"""SITReg 위상 보존 변형을 넣기 전후의 실행시간. 최적화 조합을 함께 잰다.

한 학습 스텝의 **순전파 + 역전파** 시간을 잰다 (물리손실까지 포함). 비교 축:
  전달      skin (지금 기본) / bspline (SITReg 상한이 성립하는 전달)
  상한      없음 / 있음
  합성      K = 1, 2, 4
  최적화    bf16 autocast, channels_last, torch.compile

  python exe/bench_sitreg.py --traj traj_h2/mic_clayC_t_s400706.pt
"""
import argparse
import math
import os
import time

import torch

from anchorflow import phys_resid, trilinear as TRI, vox_anchor
from anchorflow.deform import skin
from anchorflow.sitreg_warp import SITRegWarp, max_control_point_value

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, default=10)
ap.add_argument("--n_pts", type=int, default=8000)
ap.add_argument("--vox_res", type=int, default=32)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--compile", type=int, default=1)
a = ap.parse_args()

dev = "cuda"
d = torch.load(a.traj, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
ng, gl = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
gv = torch.tensor(cfg["g"], device=dev)
X = d["x"].float()
g0 = torch.Generator().manual_seed(0)
sel = torch.randperm(X.shape[1], generator=g0)[:a.n_pts].sort().values
x = X[a.t0, sel].to(dev)
v = ((X[a.t0, sel] - X[a.t0 - 1, sel]) / h).to(dev)
F0 = (d["F"][a.t0, sel].float().to(dev) if d.get("F") is not None
      else torch.eye(3, device=dev).expand(sel.numel(), 3, 3).contiguous())
mass = torch.full((sel.numel(),), float(cfg["density"]), device=dev)
vol = mass / float(cfg["density"])
ext = float((x.max(0).values - x.min(0).values).norm())
nrm = float(mass.sum()) * ext ** 2 / h ** 2

lo, hh, nn3 = vox_anchor.grid_for(x, a.vox_res ** 3)
gpos = (torch.stack(torch.meshgrid(
    *[torch.arange(int(nn3[k]), device=dev, dtype=x.dtype) for k in range(3)],
    indexing="ij"), -1).reshape(-1, 3)) * float(hh) + lo
sidx, w8 = TRI.corners(x, lo, hh, nn3)
log_r = torch.full((gpos.shape[0],), math.log(float(hh)), device=dev)
log_t = torch.zeros_like(log_r)

# 제어점 상한: 업샘플 배수는 (MPM 격자 / 제어점 격자) 가 아니라 **점 표본 밀도**로
# 본다. 여기서는 물체가 32 칸에 걸치고 입자 간격이 그보다 촘촘하므로 4 를 쓴다.
try:
    BND = max_control_point_value([4, 4, 4]) * float(hh)
    print(f"[상한] 제어점 |c| < {BND:.5f} (간격 {float(hh):.5f} 기준)", flush=True)
except Exception as e:
    BND = 0.4 * float(hh)
    print(f"[상한] SITReg 계산 실패 ({e}) -- 이론 근사 0.4*간격 = {BND:.5f}",
          flush=True)


def warp_skin(q, c):
    return skin(q, gpos, c, log_r, log_t, sidx, float(hh))[0]


def warp_bspline(q, c):
    return q + _bspline(q, lo, float(hh), nn3, c)


def _bspline(q, lo_, h_, n3, dp):
    """3 차 B-spline 업샘플로 제어점 변위를 점으로 옮긴다."""
    t = (q - lo_) / h_
    base = (t - 0.5).floor().long()
    f = t - base.to(t.dtype)
    w = torch.stack([(1 - f) ** 3 / 6,
                     (3 * f ** 3 - 6 * f ** 2 + 4) / 6,
                     (-3 * f ** 3 + 3 * f ** 2 + 3 * f + 1) / 6,
                     f ** 3 / 6], -1)                     # [N,3,4]
    n3l = n3.to(torch.long)
    out = torch.zeros_like(q)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                idx = torch.stack([(base[:, 0] + i - 1).clamp(0, int(n3l[0]) - 1),
                                   (base[:, 1] + j - 1).clamp(0, int(n3l[1]) - 1),
                                   (base[:, 2] + k - 1).clamp(0, int(n3l[2]) - 1)], -1)
                fl = (idx[:, 0] * int(n3l[1]) + idx[:, 1]) * int(n3l[2]) + idx[:, 2]
                ww = (w[:, 0, i] * w[:, 1, j] * w[:, 2, k]).unsqueeze(-1)
                out = out + ww * dp[fl]
    return out


def step(dp, mode, K, bnd):
    if mode == "skin":
        wf = warp_skin
    else:
        wf = warp_bspline
    if bnd is None:
        x2 = wf(x, dp)
    else:
        x2 = SITRegWarp(bnd, K).apply(x, dp, wf)
    E, _dl, _Ft, _pt = phys_resid.grid_ip_energy(
        x, x2 - x, v, F0, mass, vol, cfg, h, ng, gl, g=gv, norm=nrm)
    return E


def timeit(tag, mode, K, bnd, amp=False, comp=False):
    dp = torch.zeros(gpos.shape[0], 3, device=dev, requires_grad=True)
    fn = step
    if comp:
        try:
            fn = torch.compile(step, dynamic=False)
        except Exception as e:
            print(f"  {tag}: compile 실패 {e}"); comp = False
    for it in range(a.warmup + a.iters):
        if it == a.warmup:
            torch.cuda.synchronize(); t0 = time.time()
        if dp.grad is not None:
            dp.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            E = fn(dp, mode, K, bnd)
        E.backward()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / a.iters * 1000
    print(f"  {tag:46s} {ms:8.2f} ms", flush=True)
    return ms


print(f"[설정] 입자 {sel.numel()}, 제어점 {gpos.shape[0]}, 반복 {a.iters}")
print("== 적용 전 (상한·합성 없음)")
b_skin = timeit("skin 전달", "skin", 1, None)
b_bsp = timeit("bspline 전달", "bspline", 1, None)
print("== SITReg 적용 (제어점 상한 + 합성)")
for K in (1, 2, 4):
    timeit(f"bspline + 상한, 합성 K={K}", "bspline", K, BND)
print("== 최적화")
timeit("bspline + 상한 K=2, bf16", "bspline", 2, BND, amp=True)
if a.compile:
    timeit("bspline + 상한 K=2, compile", "bspline", 2, BND, comp=True)
    timeit("bspline + 상한 K=2, bf16 + compile", "bspline", 2, BND,
           amp=True, comp=True)
print(f"\n기준선: skin {b_skin:.2f} ms, bspline {b_bsp:.2f} ms")
