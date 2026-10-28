"""E·기울기 단계 벤치 (doc/fem_realtime.md 절 J 항목 4): float64 대수 (기준 tet_eg·near_eg) 대 float32 대수 (fem_rt_cuda2 tet_eg32·near_eg32,
관성·합은 float64) 의 GPU 시간과 값 차 (E 상대차, 기울기 상대차, 같은 행렬로 푼 뉴턴 방향 차), 에너지만 평가 (선탐색) 시간.
사용: python exe/fem_rt_eg_bench.py --t 0 40 70"""
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
ap.add_argument("--t", type=int, nargs="+", default=[0, 40, 70])
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


def gtime(fn, rep=10):
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
    E0, g0 = CF.energy_grad(U)
    E0 = float(E0)
    b0 = g0.reshape(-1).double()
    x0, _, _, _ = PC.solve_tile(M, b0, 1e-6, 20000, prec=32, mode=1, delta=0.1)
    tf = gtime(lambda: CF.energy_grad(U))
    te = gtime(lambda: CF.energy(U))
    print(f"\n[t={t} 노드 {M} 사면체 {G.T} 바닥 후보 {CF.nn}] float64 (기준): E+기울기 {tf:.3f} ms, E 만 {te:.3f} ms, E {E0:.9e}", flush=True)
    # 같은 식을 두 번 (원자 합 순서 비결정성 바닥)
    E0b, g0b = CF.energy_grad(U)
    print(f"  기준 두 번: E 상대차 {abs(float(E0b) - E0) / abs(E0):.2e}, 기울기 상대차 {float((g0b - g0).norm() / g0.norm()):.2e}", flush=True)
    for wh in ("tet", "near", "both"):
        E1, g1 = R2.energy_grad32(CF, U, True, wh)
        b1 = g1.reshape(-1).double()
        x1, _, _, _ = PC.solve_tile(M, b1, 1e-6, 20000, prec=32, mode=1, delta=0.1)
        t1 = gtime(lambda: R2.energy_grad32(CF, U, True, wh))
        t2 = gtime(lambda: R2.energy_grad32(CF, U, False, wh))
        # 선탐색에서 쓰는 차: E(U - p) - E(U) 를 두 판으로
        dE0 = float(CF.energy(U - x0.view(M, 3).float())) - E0
        dE1 = float(R2.energy_grad32(CF, U - x0.view(M, 3).float(), False, wh)[0]) - float(E1)
        print(f"  float32 대수 {wh}: E+기울기 {t1:.3f} ms, E 만 {t2:.3f} ms, E 상대차 {abs(float(E1) - E0) / abs(E0):.2e}, 기울기 상대차 {float((g1 - g0).norm() / g0.norm()):.2e}, "
              f"뉴턴 방향 상대차 {float((x1 - x0).norm() / x0.norm()):.2e}, 걸음 에너지 감소 E(U-p)-E(U) {dE1:.6e} (기준 {dE0:.6e})", flush=True)
print("[끝] eg_bench", flush=True)
