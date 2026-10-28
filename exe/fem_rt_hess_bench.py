"""헤시안 단계 벤치 (doc/fem_realtime.md 절 J 항목 4): 워프당 사면체 헤시안 (tet_hess_w) 의 야코비 바퀴 상한·허용치별 GPU 시간과 결과 차,
바닥 후보 헤시안 (near_hess f64 대수 / near_hess32 f32 대수) GPU 시간. 정답 프레임 t 상태, 최종 판과 같은 행렬.
사용: python exe/fem_rt_hess_bench.py --t 0 40 70"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402

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
    slots = RTC.slots_k(G)
    Ms = PC.Mmax

    def hess(sw=10, tol=1e-12, near32=True):
        CF.near32 = near32
        PC.B.zero_()
        CF.cnt.zero_()
        CF.hess_into(U, slots, PC.B, 1e-6, maxsw=sw, jtol=tol, Ms=Ms)
    hess()
    B0 = PC.B.view(15, 9, Ms)[:8, :, :M].clone()
    nneg = int(CF.cnt[1])
    tb = gtime(lambda: hess())
    print(f"\n[t={t} 노드 {M} 사면체 {G.T} 바닥 후보 {CF.nn}] 기준 (바퀴 10, tol 1e-12, near32): {tb:.3f} ms, 음 고유값 사면체 {nneg} ({nneg / G.T:.2f})", flush=True)
    RTC.asm_k(G, PC, 1, Ms)
    b = CF.energy_grad(U)[1].reshape(-1).double()
    hess()
    RTC.asm_k(G, PC, 1, Ms)
    x0, it0, _, _ = PC.solve_tile(M, b, 1e-6, 20000, prec=32, mode=1, delta=0.1)
    for sw, tol in ((8, 1e-12), (6, 1e-12), (5, 1e-12), (4, 1e-12), (10, 1e-10), (10, 1e-8)):
        tt = gtime(lambda: hess(sw, tol))
        hess(sw, tol)
        dB = float((PC.B.view(15, 9, Ms)[:8, :, :M] - B0).norm() / B0.norm())
        RTC.asm_k(G, PC, 1, Ms)
        xx, it, _, _ = PC.solve_tile(M, b, 1e-6, 20000, prec=32, mode=1, delta=0.1)
        print(f"  바퀴 {sw} tol {tol:g}: {tt:.3f} ms, 행렬 상대차 {dB:.2e}, PCG {it} (기준 {it0}), 해 차 {float((xx - x0).norm() / x0.norm()):.2e}", flush=True)
    t64 = gtime(lambda: hess(near32=False))
    print(f"  바닥 후보 f64 대수 (near32 끔): {t64:.3f} ms (near32 {tb:.3f} ms)", flush=True)
print("[끝] hess_bench", flush=True)
