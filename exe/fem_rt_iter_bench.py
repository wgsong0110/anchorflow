"""지속 PCG 반복 비용 분해 (doc/fem_realtime.md 절 K): 정답 프레임 t 상태 계에서 pcg_t2 (pcg_t6 혼합과 같은 식) 를 판정 없이 고정 N 회 돌린 시간을
  전체 / 곱 단계 뺌 / 벡터 단계 뺌 / 둘 다 뺌 (장벽 2 + 블록 합 3 뼈대), 그리고 블록 수 84 / 42 / 21 로. 사용: python exe/fem_rt_iter_bench.py --t 40"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import fem_rt_cuda2 as R2                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--t", type=int, nargs="+", default=[40])
ap.add_argument("--n", type=int, default=500)
a = ap.parse_args()
dev = "cuda"
RTC.RAWPTR[0] = True
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).float()
vol = m / float(cfg["density"])
N = int(d["x"].shape[1])
g = torch.as_tensor(cfg["g"], device=dev, dtype=torch.float32)
print(f"[장치] {torch.cuda.get_device_name(0)}", flush=True)


def gtime(fn, rep=3):
    fn()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    for _ in range(rep):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / rep


for t in a.t:
    x = d["x"][t].to(dev).float()
    v = d["v"][t].to(dev).float()
    F = DM.traj_F(d, h, dev)[t].to(dev).float().reshape(N, 3, 3)
    G = RTC.GridK(x, 0.028284)
    U, ws = RT.node_inertial(x, v, m, G, g, h)
    CF = RTC.CudaTetqK(G, x, v, F, m, vol, cfg, h, U)
    CF.warp, CF.near32, CF.sym = True, True, 1
    PC = RTC.CudaPcg(60000, dev)
    PC.sym = 1
    M = G.M
    PC.B.zero_()
    CF.hess_into(U, RTC.slots_k(G), PC.B, 1e-6, Ms=PC.Mmax)
    RTC.asm_k(G, PC, 1, PC.Mmax)
    b = CF.energy_grad(U)[1].reshape(-1).double()
    print(f"\n[t={t} 노드 {M}] 고정 {a.n} 회 (판정 없음, rtol 0)", flush=True)
    for nb_ in (0, 42, 21):
        PC.t2_nblk = nb_
        for kern, nm in (("pcg_t2_r1", "전체"), ("pcg_t2_d1", "곱 단계 뺌"), ("pcg_t2_d2", "벡터 단계 뺌"), ("pcg_t2_d3", "뼈대 (장벽 2 + 블록 합 3)")):
            ms = gtime(lambda: R2.solve_t2(PC, M, b, 0.0, a.n, kern=kern, delta=0.0, cap=1))
            print(f"  블록 {PC.t2_info[0]:3d} {nm}: {1e3 * ms / a.n:.1f} us/회", flush=True)
print("[끝] iter_bench", flush=True)
