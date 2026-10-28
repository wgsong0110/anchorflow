"""PCG 커널 판별 시간 (정답 프레임 t 상태 입자 바닥 계, fem_rt_cuda): 지속 커널 블록 수·크기, 2 커널 그래프, float32 탐색 벡터.
사용: python exe/fem_rt_pcg_bench.py --t 40"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--t", type=int, default=40)
a = ap.parse_args()
dev = "cuda"
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).float()
vol = m / float(cfg["density"])
N = int(d["x"].shape[1])
x = d["x"][a.t].to(dev).float()
v = d["v"][a.t].to(dev).float()
F = DM.traj_F(d, h, dev)[a.t].to(dev).float().reshape(N, 3, 3)
g = torch.as_tensor(cfg["g"], device=dev, dtype=torch.float32)
G = RTC.GridK(x, 0.028284)
U, ws = RT.node_inertial(x, v, m, G, g, h)
CF = RTC.CudaTetqK(G, x, v, F, m, vol, cfg, h, U)
CF.warp, CF.near32 = True, True
PC = RTC.CudaPcg(60000, dev)
M = G.M
gi = G.gi
slots = (G.tets.unsqueeze(-1) * 15 + RT.off_cls(gi[G.tets].unsqueeze(1) - gi[G.tets].unsqueeze(2))).int().contiguous()
PC.B.zero_()
CF.hess_into(U, slots, PC.B, 1e-6, Ms=PC.Mmax)
Bs = PC.B.view(15, 9, PC.Mmax)
D = Bs[0, :, :M].t().double().reshape(M, 3, 3)
PC.Dinv[:M * 9].copy_(torch.linalg.inv(D).reshape(-1))
PC.nb.view(15, PC.Mmax)[:, :M] = RT.BEll(G.gi, dtype=torch.float32).nb.t().int()
b = CF.energy_grad(U)[1].reshape(-1)


def tm(fn, rep=3):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize()
    return out, 1e3 * (time.perf_counter() - t0) / rep


sm = torch.cuda.get_device_properties(0).multi_processor_count
(x0, it0, r0, _), t0 = tm(lambda: PC.solve2(M, b, 1e-6, 20000))
print(f"[2 커널 그래프] {it0} 회 {t0:.2f} ms ({1e3 * t0 / it0:.1f} us/회) 실제 잔차 {PC.true_res(M, x0, b):.2e}", flush=True)
for bsz in (128, 256, 512):
    for mul in (1, 2, 4):
        PC.bsz, PC.nblk = bsz, mul * sm * (256 // bsz if bsz < 256 else 1)
        try:
            (xx, it, rr, _), tt = tm(lambda: PC.solve_coop(M, b, 1e-6, 20000))
            print(f"[지속 블록 {PC.nblk} x {bsz}] {it} 회 {tt:.2f} ms ({1e3 * tt / it:.1f} us/회) 실제 잔차 {PC.true_res(M, xx, b):.2e} 해 차 {float((xx - x0).norm() / x0.norm()):.2e}", flush=True)
        except Exception as e:                                               # noqa: BLE001
            print(f"[지속 블록 {PC.nblk} x {bsz}] 실패 {type(e).__name__}: {str(e)[:120]}", flush=True)
# 고정 반복 수로 장벽만의 비용: rtol 0 (끝까지), maxit 200
PC.bsz, PC.nblk = 256, 2 * sm
(_, it, _, _), tt = tm(lambda: PC.solve_coop(M, b, 0.0, 200))
print(f"[지속 200 회 고정] {tt:.2f} ms ({1e3 * tt / 200:.1f} us/회)", flush=True)
