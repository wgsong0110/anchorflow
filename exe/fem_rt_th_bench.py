"""16 비트 행렬 PCG·노드 재정렬 마이크로벤치 (doc/fem_realtime.md 절 J 항목 2): 정답 프레임 t 상태 입자 바닥 계 (최종 판과 같은 행렬).
  기준   solve_tile 혼합 (pcg_t32 mode 1, i_tmix_asm_r3 의 PCG)
  th<k>  fem_rt_cuda2.pcg_th<k>: 안쪽 Â = S A S 를 k (0 f32 척도, 1 f16, 2 bf16) 로, 신뢰 갱신은 float32 저장 A·float64 (멈춤 = 실제 잔차 ≤ rtol)
         +d = 대각 블록 float32, +c = 평행이동 모드 보정
  곱만   bell_mv_t32 (기준) / bell_mv_th0·1·2, Morton 재정렬한 bell_mv_t32
사용: python exe/fem_rt_th_bench.py --t 0 40"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np                                                                  # noqa: E402
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import fem_rt_cuda2 as R2                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--t", type=int, nargs="+", default=[0, 40])
a = ap.parse_args()
dev = "cuda"
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).float()
vol = m / float(cfg["density"])
N = int(d["x"].shape[1])
g = torch.as_tensor(cfg["g"], device=dev, dtype=torch.float32)
print(f"[장치] {torch.cuda.get_device_name(0)}", flush=True)


def tm(fn, rep=3):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize()
    return out, 1e3 * (time.perf_counter() - t0) / rep


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
    (x0, it0, _, _), t0 = tm(lambda: PC.solve_tile(M, b, 1e-6, 20000, prec=32, mode=1, delta=0.1))
    tr0 = PC.true_res(M, x0, b)
    print(f"\n[t={t} 노드 {M}] 기준 타일 혼합: {it0} 회 갱신 {PC.last_nup} {t0:.2f} ms ({1e3 * t0 / it0:.1f} us/회) 실제 잔차 {tr0:.2e}", flush=True)
    HM = R2.HalfMat(PC.Mmax, dev)
    for mk, dg, cp_ in ((0, False, False), (1, False, False), (2, False, False), (1, True, False), (1, True, True), (2, True, True)):
        HM.build(PC, M, mk, dg32=dg, comp=cp_)
        try:
            (xx, it, rr, neg), tt = tm(lambda: R2.solve_th(PC, HM, M, b, 1e-6, 20000, mk=mk, delta=0.1))
        except Exception as ex:                                          # noqa: BLE001
            print(f"  th{mk}{'+d' if dg else ''}{'+c' if cp_ else ''}: 실패 {ex!r}", flush=True)
            continue
        tr = PC.true_res(M, xx, b)
        print(f"  th{mk}{'+d' if dg else ''}{'+c' if cp_ else ''} 블록 {PC.th_info}: {it} 회 갱신 {PC.last_nup} {tt:.2f} ms ({1e3 * tt / max(it, 1):.1f} us/회) "
              f"실제 잔차 {tr:.2e} (커널 {rr:.2e}) 해 차 {float((xx - x0).norm() / x0.norm()):.2e} 음곡률 {neg}", flush=True)
    # 곱만 (커널 100 회 연속)
    md, md2 = RTC._mod(), R2._mod2()
    nt = (M + 31) // 32
    vv = torch.randn(3 * M, dtype=torch.float32, device=dev)
    yy = torch.empty_like(vv)

    def mv_t32(B=PC.B, nb=PC.nb):
        with RTC._Stream():
            for _ in range(100):
                md.get_function("bell_mv_t32")((nt,), (512,), (np.int32(M), RTC._cp(B), RTC._cp(nb), RTC._cp(vv), RTC._cp(yy), np.int32(PC.Mmax), np.int32(1)))
    _, tq = tm(mv_t32)
    print(f"  [곱만] t32 (기준 float32): {10 * tq:.1f} us/회", flush=True)
    for mk in (0, 1, 2):
        HM.build(PC, M, mk)
        Bp, Bd = HM.ptrs(mk)

        def mv_th(mk=mk, Bp=Bp, Bd=Bd):
            with R2._Stream():
                for _ in range(100):
                    md2.get_function(f"bell_mv_th{mk}")((nt,), (512,), (np.int32(M), R2._p(Bp), R2._p(Bd), R2._p(PC.nb), R2._p(vv), R2._p(yy), np.int32(PC.Mmax)))
        _, tq = tm(mv_th)
        print(f"  [곱만] th{mk}: {10 * tq:.1f} us/회", flush=True)
    # Morton 재정렬
    perm = R2.morton_perm(G.gi)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(M, device=dev)
    P2 = RTC.CudaPcg(60000, dev)
    P2.sym = 1
    Ms = PC.Mmax
    P2.B.view(15, 9, Ms)[:, :, :M] = PC.B.view(15, 9, Ms)[:, :, perm]
    nbo = PC.nb.view(15, Ms)[:, :M][:, perm].long()
    P2.nb.view(15, Ms)[:, :M] = torch.where(nbo >= 0, inv[nbo.clamp_min(0)], nbo).int()
    P2.Dinv.view(-1, 9)[:M] = PC.Dinv.view(-1, 9)[perm]
    bp = b.view(M, 3)[perm].reshape(-1)

    def mv_mt(B=P2.B, nb=P2.nb):
        with RTC._Stream():
            for _ in range(100):
                md.get_function("bell_mv_t32")((nt,), (512,), (np.int32(M), RTC._cp(B), RTC._cp(nb), RTC._cp(vv), RTC._cp(yy), np.int32(PC.Mmax), np.int32(1)))
    _, tq = tm(mv_mt)
    # 이웃 거리 (노드 번호 차) 중앙
    nbm = nbo.clamp_min(0)
    dist_old = (PC.nb.view(15, Ms)[:8, :M].long() - torch.arange(M, device=dev)).abs().float().median()
    dist_new = (P2.nb.view(15, Ms)[:8, :M].long() - torch.arange(M, device=dev)).abs().float().median()
    print(f"  [곱만] Morton 재정렬 t32: {10 * tq:.1f} us/회 (이웃 번호 차 중앙 {float(dist_old):.0f} -> {float(dist_new):.0f})", flush=True)
    (xp, itp, _, _), tp = tm(lambda: P2.solve_tile(M, bp, 1e-6, 20000, prec=32, mode=1, delta=0.1))
    xpo = torch.empty_like(xp.view(M, 3))
    xpo[perm] = xp.view(M, 3)
    print(f"  Morton 재정렬 타일 혼합: {itp} 회 {tp:.2f} ms ({1e3 * tp / itp:.1f} us/회) 실제 잔차 {P2.true_res(M, xp, bp):.2e} "
          f"해 차 {float((xpo.reshape(-1) - x0).norm() / x0.norm()):.2e}", flush=True)
print("[끝] th_bench", flush=True)
