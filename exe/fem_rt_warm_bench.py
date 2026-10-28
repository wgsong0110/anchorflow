"""PCG 시작 해 벤치 (doc/fem_realtime.md 절 J 항목 1): 정답 프레임 t 의 뉴턴 방향을 풀고, t+1 계를 0 시작 / 이전 방향 (격자 좌표 대응 x A-노름 최적 배율)
시작으로 같은 rtol 1e-6 (float64 실제 잔차) 까지 푼 반복 수. 사용: python exe/fem_rt_warm_bench.py --t 0 10 40 70"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import fem_rt_cuda2 as R2                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--t", type=int, nargs="+", default=[0, 10, 40, 70])
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
FF = DM.traj_F(d, h, dev)
PC = RTC.CudaPcg(60000, dev)
PC.sym = 1
print(f"[장치] {torch.cuda.get_device_name(0)}", flush=True)


def system(t):
    x = d["x"][t].to(dev).float()
    v = d["v"][t].to(dev).float()
    F = FF[t].to(dev).float().reshape(N, 3, 3)
    G = RTC.GridK(x, 0.028284)
    U, ws = RT.node_inertial(x, v, m, G, g, h)
    CF = RTC.CudaTetqK(G, x, v, F, m, vol, cfg, h, U)
    CF.warp, CF.near32, CF.sym = True, True, 1
    PC.B.zero_()
    CF.hess_into(U, RTC.slots_k(G), PC.B, 1e-6, Ms=PC.Mmax)
    RTC.asm_k(G, PC, 1, PC.Mmax)
    return G, CF.energy_grad(U)[1].reshape(-1).double()


for t in a.t:
    WS = R2.WarmStart()
    G0, b0 = system(t)
    x0, it0, _, _ = R2.solve_t2(PC, G0.M, b0, 1e-6, 20000, kern="pcg_t2_r1", cap=3)
    WS.store(G0, x0)
    G1, b1 = system(t + 1)
    M = G1.M
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    xc, itc, _, _ = R2.solve_t2(PC, M, b1, 1e-6, 20000, kern="pcg_t2_r1", cap=3)
    tc = 1e3 * (time.perf_counter() - t0)
    trc = PC.true_res(M, xc, b1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    xg = WS.guess(G1, PC, b1)
    xw, itw, _, _ = R2.solve_t2(PC, M, b1, 1e-6, 20000, kern="pcg_t2_r1", cap=3, x0=xg)
    tw = 1e3 * (time.perf_counter() - t0)
    trw = PC.true_res(M, xw, b1)
    r0 = float((b1 - (xg.new_zeros(xg.shape) if xg is None else 0)).norm())
    hit = float(WS.n_hit.float().mean())
    print(f"[t={t}->{t + 1}] 노드 {G0.M}->{M} (대응 {hit:.3f}): 0 시작 {itc} 회 {tc:.2f} ms 잔차 {trc:.2e} | 이전 방향 시작 {itw} 회 {tw:.2f} ms 잔차 {trw:.2e} "
          f"| 해 차 {float((xw - xc).norm() / xc.norm()):.2e}, 시작 해 대비 상대 거리 {float((xg - xc).norm() / xc.norm()):.3f}", flush=True)
print("[끝] warm_bench", flush=True)
