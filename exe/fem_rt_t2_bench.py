"""지속 PCG 반복 비용 벤치 (doc/fem_realtime.md 절 J 항목 2): 정답 프레임 t 상태 입자 바닥 계 (최종 판과 같은 행렬).
  기준   fem_rt_cuda solve_tile (pcg_t32 혼합, i_tmix_asm_r3 의 PCG)
  t2_*   fem_rt_cuda2 pcg_t2: 같은 식, 블록이 타일 R 개 (r1/r2/r4) 를 동시에, SM 당 블록 1/2 (b2)
  곱만   커널 100 회 연속 (CuPy 배열 변환 없이 장치 주소로 -- 절 I 의 '곱만 32 µs' 는 호스트 인자 변환 시간이 섞인 값이었다)
사용: python exe/fem_rt_t2_bench.py --t 0 40 70"""
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
ap.add_argument("--t", type=int, nargs="+", default=[0, 40, 70])
ap.add_argument("--kerns", default="pcg_t2_r1,pcg_t2_r2,pcg_t2_r4,pcg_t2_r1b2,pcg_t2_r2b2,pcg_t2_r4b2")
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


def tm(fn, rep=5):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize()
    return out, 1e3 * (time.perf_counter() - t0) / rep


def gtime(fn, rep=5):
    """GPU 이벤트 시간 (호스트 시간 빼고)."""
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
    (x0, it0, _, _), t0 = tm(lambda: PC.solve_tile(M, b, 1e-6, 20000, prec=32, mode=1, delta=0.1))
    print(f"\n[t={t} 노드 {M}] 기준 타일 혼합: {it0} 회 갱신 {PC.last_nup} {t0:.2f} ms ({1e3 * t0 / it0:.1f} us/회) 실제 잔차 {PC.true_res(M, x0, b):.2e} 블록 {PC.tl_info}",
          flush=True)
    for kern in [q for q in a.kerns.split(",") if q]:
        try:
            (xx, it, rr, neg), tt = tm(lambda: R2.solve_t2(PC, M, b, 1e-6, 20000, kern=kern, delta=0.1, cap=3))
        except Exception as ex:                                          # noqa: BLE001
            print(f"  {kern}: 실패 {ex!r}", flush=True)
            continue
        print(f"  {kern} 블록 {PC.t2_info}: {it} 회 갱신 {PC.last_nup} {tt:.2f} ms ({1e3 * tt / max(it, 1):.1f} us/회) 실제 잔차 {PC.true_res(M, xx, b):.2e} "
              f"해 차 {float((xx - x0).norm() / x0.norm()):.2e} 음곡률 {neg}", flush=True)
    md, md2 = RTC._mod(), R2._mod2()
    nt = (M + 31) // 32
    vv = torch.randn(3 * M, dtype=torch.float32, device=dev)
    yy = torch.empty_like(vv)
    f_old = md.get_function("bell_mv_t32")
    f_new = md2.get_function("bell_mv_th0")
    args_old = (np.int32(M), RTC._cp(PC.B), RTC._cp(PC.nb), RTC._cp(vv), RTC._cp(yy), np.int32(PC.Mmax), np.int32(1))
    args_new = (np.int32(M), R2._p(PC.B), R2._p(None), R2._p(PC.nb), R2._p(vv), R2._p(yy), np.int32(PC.Mmax))

    def run(f, args):
        with RTC._Stream():
            for _ in range(100):
                f((nt,), (512,), args)
    print(f"  [곱만, GPU 이벤트] bell_mv_t32 (기준 커널) {10 * gtime(lambda: run(f_old, args_old)):.1f} us/회, bell_mv_th0 (__ldg) {10 * gtime(lambda: run(f_new, args_new)):.1f} us/회",
          flush=True)
    RTC.RAWPTR[0] = False
    args_cp = (np.int32(M), RTC._cp(PC.B), RTC._cp(PC.nb), RTC._cp(vv), RTC._cp(yy), np.int32(PC.Mmax), np.int32(1))

    def run_cp():
        with RTC._Stream():
            for _ in range(100):
                f_old((nt,), (512,), (np.int32(M), RTC._cp(PC.B), RTC._cp(PC.nb), RTC._cp(vv), RTC._cp(yy), np.int32(PC.Mmax), np.int32(1)))
    _, tq = tm(run_cp)
    print(f"  [곱만, 호스트 시간, CuPy 배열 변환 포함 (절 I 측정 방식)] bell_mv_t32 {10 * tq:.1f} us/회", flush=True)
    RTC.RAWPTR[0] = True
print("[끝] t2_bench", flush=True)
