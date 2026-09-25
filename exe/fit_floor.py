"""이 표현이 한 스텝에서 도달할 수 있는 **하한**을 잰다.

학습이 멈춰 보일 때, 원인이 최적화인지 표현인지 갈라야 한다. 여기서는 정답 변위를
**알고 있는** 오라클이 격자점 변위 dp 만으로 최소제곱 맞춤을 한다. 네트워크도 학습도
없다. 그 결과가

  * 정지 기준선과 비슷하다  -> 표현이 한 프레임의 움직임을 담지 못한다 (표현 한계)
  * 훨씬 낮다              -> 표현은 담을 수 있다. 못 배우는 쪽이 문제다

손잡이 입자는 학습과 같은 규약으로 빼고 잰다.
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from anchorflow import trilinear as TRI                          # noqa: E402
from anchorflow import vox_anchor                                # noqa: E402
from anchorflow.deform import skin                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--vox_res", type=int, default=32)
ap.add_argument("--n_pts", type=int, default=8000)
ap.add_argument("--n_win", type=int, default=12, help="표본 창 수")
ap.add_argument("--steps", type=int, default=400, help="맞춤 반복")
ap.add_argument("--unroll", type=int, default=1)
ap.add_argument("--files", default="", help="쉼표로 구분한 궤적 파일명 (없으면 앞 8 개)")
ap.add_argument("--t0", type=int, nargs="*", default=None,
                help="창 시작 프레임을 직접 지정 (모델 평가와 같은 창으로 맞출 때)")
ap.add_argument("--dev", default="cuda")
a = ap.parse_args()
dev = a.dev

if a.files:
    files = [os.path.join(a.data, f if f.endswith(".pt") else f + ".pt")
             for f in a.files.split(",")]
else:
    files = sorted(glob.glob(os.path.join(a.data, "*.pt")))[:8]
if not files:
    raise SystemExit("궤적이 없다")
gen = torch.Generator(device=dev).manual_seed(0)
tot_r = tot_m = tot_s = 0.0
n = 0
wins = ([(f, t) for f in range(len(files)) for t in a.t0] if a.t0
        else [(wi % len(files), None) for wi in range(a.n_win)])
for wi, (fi, t0_given) in enumerate(wins):
    d = torch.load(files[fi], map_location="cpu", weights_only=False)
    X = d["x"].float()
    T = X.shape[0]
    t0 = (t0_given if t0_given is not None
          else 3 + (wi * 7) % max(T - a.unroll - 4, 1))
    gsel = torch.randperm(X.shape[1], generator=torch.Generator().manual_seed(wi)
                          )[:a.n_pts].sort().values
    x = X[t0][gsel].to(dev)
    gt = X[t0 + a.unroll][gsel].to(dev)
    # 손잡이 입자 제외 (학습과 같은 규약)
    free = torch.ones(x.shape[0], dtype=torch.bool, device=dev)
    P = d["ctrl_pos"].float()
    R = d["ctrl_R"].float()
    for kk in range(P.shape[1]):
        r = float(R[0] if R.ndim == 1 else R[0, kk])
        sel = (X[0][gsel].to(dev) - P[0, kk].to(dev)).norm(dim=-1) < r
        free &= ~sel
    ext = float((X[0].max(0).values - X[0].min(0).values).norm())

    lo, hh, nn3 = vox_anchor.grid_for(x, a.vox_res ** 3)
    sidx, _w8 = TRI.corners(x, lo, hh, nn3)
    gpos = (torch.stack(torch.meshgrid(
        *[torch.arange(int(nn3[i]), device=dev, dtype=x.dtype)
          for i in range(3)], indexing="ij"), -1).reshape(-1, 3)) * hh + lo
    import math
    log_r = torch.full((gpos.shape[0],), math.log(float(hh)), device=dev)
    log_t = torch.zeros_like(log_r)
    dp = torch.zeros(gpos.shape[0], 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([dp], lr=1e-3)
    for it in range(a.steps):
        opt.zero_grad(set_to_none=True)
        xs = skin(x, gpos, dp, log_r, log_t, sidx, float(hh))[0]
        loss = ((xs[free] - gt[free]) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        xs = skin(x, gpos, dp, log_r, log_t, sidx, float(hh))[0]
        fit = float((xs[free] - gt[free]).norm(dim=-1).mean()) / ext
        still = float((x[free] - gt[free]).norm(dim=-1).mean()) / ext
    tot_m += fit
    tot_s += still
    tot_r += fit / max(still, 1e-20)
    n += 1
    print(f"  창 {wi:2d} t0={t0:3d}  맞춤 {100*fit:.4f}%  정지 {100*still:.4f}%  "
          f"비 {fit/max(still,1e-20):.3f}", flush=True)
print(f"[하한] {n} 창 평균  맞춤 {100*tot_m/n:.4f}%  정지 {100*tot_s/n:.4f}%  "
      f"비 {tot_r/n:.3f}  (unroll {a.unroll}, 격자 {a.vox_res}^3)", flush=True)
