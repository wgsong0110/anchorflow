"""혼합 정밀도 PCG (pcg_mix, doc/fem_realtime.md 절 I) 검산·시간: 정답 프레임 t 상태 입자 바닥 계 (대칭 반쪽 블록 ELL, 최종 판과 같은 행렬).
float64 지속 커널 (solve_coop, 최종 판) 대 float32 단독 (mode 0) 대 float32 + float64 신뢰 갱신 (mode 1, δ 여러 값).
사용: python exe/fem_rt_mix_bench.py --t 0 10 40"""
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
ap.add_argument("--t", type=int, nargs="+", default=[0, 40])
ap.add_argument("--micro", action="store_true", help="장벽만·곱만 비용 분해")
ap.add_argument("--variants", default="pcg_mix:2:0,pcg_mix:2:1,pcg_mix_lb2:2:0,pcg_mix_lb2:2:1,pcg_mix_lb3:3:1,pcg_mix_lb4:4:1,pcg_mix_lb4:4:0")
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
    gi = G.gi
    slots = (G.tets.unsqueeze(-1) * 15 + RT.off_cls(gi[G.tets].unsqueeze(1) - gi[G.tets].unsqueeze(2))).int().contiguous()
    PC.B.zero_()
    CF.hess_into(U, slots, PC.B, 1e-6, Ms=PC.Mmax)
    Bs = PC.B.view(15, 9, PC.Mmax)
    D = Bs[0, :, :M].t().double().reshape(M, 3, 3)
    PC.Dinv[:M * 9].copy_(torch.linalg.inv(D).reshape(-1))
    nbl = RT.BEll(G.gi, dtype=torch.float32)
    PC.nb.view(15, PC.Mmax)[:, :M] = torch.where(nbl.valid, nbl.nb, torch.full_like(nbl.nb, -1)).t().int()
    b = CF.energy_grad(U)[1].reshape(-1)
    (x0, it0, _, _), t0 = tm(lambda: PC.solve_coop(M, b, 1e-6, 20000))
    print(f"[t={t} 노드 {M}] float64 지속 (최종 판): {it0} 회 {t0:.2f} ms ({1e3 * t0 / it0:.1f} us/회) 실제 잔차 {PC.true_res(M, x0, b):.2e}", flush=True)
    if a.micro:
        import numpy as np
        md = RTC._mod()
        for nb_ in (21, 42, 84, 168):
            acc = torch.zeros(12, dtype=torch.float64, device=dev)
            bar = torch.zeros(2, dtype=torch.int32, device=dev)
            vv = torch.ones(3 * M, dtype=torch.float32, device=dev)

            def _bar(nb_=nb_, acc=acc, bar=bar, vv=vv):
                bar.zero_()
                with RTC._Stream():
                    md.get_function("bar_only")((nb_,), (256,), (np.int32(3 * M), np.int32(1000), RTC._cp(acc), RTC._cp(bar), RTC._cp(vv)))
            _, tb = tm(_bar)
            print(f"  [장벽만] 블록 {nb_} x 256, 1000 회: {tb:.2f} ms ({tb:.1f} us/회, 장벽 2 + 블록 합 3)", flush=True)
        z32 = torch.randn(3 * M, dtype=torch.float32, device=dev)
        y32 = torch.empty_like(z32)
        g1, b1 = RTC._grid(M)

        def _mv():
            with RTC._Stream():
                for _ in range(100):
                    md.get_function("bell_mv_f")(g1, b1, (np.int32(M), RTC._cp(PC.B), RTC._cp(PC.nb), RTC._cp(z32), RTC._cp(y32), np.int32(PC.Mmax), np.int32(1)))
        _, tmv = tm(_mv)
        print(f"  [곱만 float32 대칭 반쪽, 커널 100 회 연속] {1e3 * tmv / 100:.1f} us/회", flush=True)
        x64 = torch.randn(3 * M, dtype=torch.float64, device=dev)
        y64 = torch.empty_like(x64)

        def _mv64():
            with RTC._Stream():
                for _ in range(100):
                    md.get_function("bell_mv")(g1, b1, (RTC._cp(PC.Mp), RTC._cp(PC.B), RTC._cp(PC.nb), RTC._cp(x64), RTC._cp(y64), np.int32(PC.Mmax), np.int32(1)))
        _, tmv64 = tm(_mv64)
        print(f"  [곱만 float64 벡터 대칭 반쪽, 커널 100 회 연속] {1e3 * tmv64 / 100:.1f} us/회", flush=True)
    for prec, mode in ((64, 0), (32, 1), (32, 0)):
        (xx, it, rr, neg), tt = tm(lambda: PC.solve_tile(M, b, 1e-6, 20000, prec=prec, mode=mode, delta=0.1))
        tr = PC.true_res(M, xx, b)
        print(f"  [타일 곱 지속] float{prec} {'혼합' if mode == 1 else ''} 블록 {PC.tl_info}: {it} 회 갱신 {PC.last_nup} {tt:.2f} ms ({1e3 * tt / max(it, 1):.1f} us/회) "
              f"실제 잔차 {tr:.2e} 해 차 {float((xx - x0).norm() / x0.norm()):.2e}", flush=True)
    if a.micro:
        for prec in (32, 64):
            vv = torch.randn(3 * M, dtype=torch.float32 if prec == 32 else torch.float64, device=dev)
            yy = torch.empty_like(vv)
            nt = (M + 31) // 32

            def _mvt(vv=vv, yy=yy, prec=prec, nt=nt):
                with RTC._Stream():
                    for _ in range(100):
                        md.get_function(f"bell_mv_t{prec}")((nt,), (512,), (np.int32(M), RTC._cp(PC.B), RTC._cp(PC.nb), RTC._cp(vv), RTC._cp(yy), np.int32(PC.Mmax), np.int32(1)))
            _, tq = tm(_mvt)
            ref = PC.true_res  # noqa
            print(f"  [타일 곱만 float{prec}, 커널 100 회 연속] {1e3 * tq / 100:.1f} us/회", flush=True)
    for kern, cap, rows in [(q.split(":")[0], int(q.split(":")[1]), int(q.split(":")[2])) for q in a.variants.split(",") if q]:
        PC.mx_cap, PC.mx_kern, PC.mx_rows = cap, kern, rows
        if hasattr(PC, "mx_acc"):
            del PC.mx_acc
        for mode, dl in ((0, 0.0), (1, 0.1)):
            (xx, it, rr, neg), tt = tm(lambda: PC.solve_mix(M, b, 1e-6, 20000, mode=mode, delta=dl))
            tr = PC.true_res(M, xx, b)
            print(f"  {kern} 행 {rows} 블록 {PC.mx_nblk} {PC.mx_info} {'f32 단독' if mode == 0 else f'혼합 δ {dl:g}'}: {it} 회 갱신 {PC.last_nup} {tt:.2f} ms "
                  f"({1e3 * tt / max(it, 1):.1f} us/회) 실제 잔차 {tr:.2e} 해 차 {float((xx - x0).norm() / x0.norm()):.2e} 음곡률 {neg}", flush=True)
