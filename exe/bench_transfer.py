"""격자 변위를 가우시안으로 옮기는 **전달 방식**의 비용만 따로 잰다.

우리는 셀 여덟 꼭짓점에 학습된 반경 소프트맥스로 스키닝한다. i-PG(그리고 MPM)는
이차 B-스플라인 3^3 스텐실로 G2P 한다. 어느 쪽을 쓰든 실시간(>= 30 FPS)이 되는지
보려면 전달만이 아니라 **그 격자에서 도는 3D conv 비용**까지 같이 봐야 한다 --
i-PG 를 그대로 쓰면 격자가 100^3 이라 conv 가 따라 커진다.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from anchorflow import phys_resid, trilinear as TRI, vox_anchor   # noqa: E402
from anchorflow.conv_stepper import ConvStepper                   # noqa: E402
from anchorflow.deform import skin                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n_pts", type=int, default=243621, help="가우시안 수")
ap.add_argument("--rep", type=int, default=50)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)

x = torch.rand(a.n_pts, 3, device=dev) * 1.2 + 0.4        # 물체 크기 ~1.2
mass = torch.full((a.n_pts,), 1e-6, device=dev)
print(f"[벤치] 가우시안 {a.n_pts}", flush=True)


def timed(fn, rep):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / rep * 1000.0


# ---- 1) 우리 방식: 셀 여덟 꼭짓점 + 학습 반경 스키닝 (격자 32^3)
import math                                                        # noqa: E402
lo, hh, nn3 = vox_anchor.grid_for(x, 32 ** 3)
sidx, _w8 = TRI.corners(x, lo, hh, nn3)
gpos = (torch.stack(torch.meshgrid(
    *[torch.arange(int(nn3[i]), device=dev) for i in range(3)],
    indexing="ij"), -1).reshape(-1, 3).float()) * float(hh) + lo
dp32 = torch.randn(gpos.shape[0], 3, device=dev) * 0.002
log_r = torch.full((gpos.shape[0],), math.log(float(hh)), device=dev)
log_t = torch.zeros_like(log_r)
t_skin = timed(lambda: skin(x, gpos, dp32, log_r, log_t, sidx, float(hh))[0],
               a.rep)
print(f"  스키닝(꼭짓점8, 32^3)        {t_skin:6.2f} ms", flush=True)


# ---- 2) i-PG 방식: 이차 B-스플라인 G2P
def g2p_bspline(n_grid, grid_lim=2.0):
    dx = grid_lim / n_grid
    xr = x / dx
    base, w, _ = phys_resid._bspline(xr)
    off = phys_resid._offsets(dev)
    idx3 = (base.unsqueeze(1) + off.unsqueeze(0)).clamp(0, n_grid - 1)
    flat = ((idx3[..., 0] * n_grid + idx3[..., 1]) * n_grid + idx3[..., 2])
    i0, i1, i2 = off[:, 0], off[:, 1], off[:, 2]
    ww = w[:, 0][:, i0] * w[:, 1][:, i1] * w[:, 2][:, i2]          # [N,27]
    du = torch.randn(n_grid ** 3, 3, device=dev) * 0.002
    return flat, ww, du


for ng in (32, 100):
    flat, ww, du = g2p_bspline(ng)
    t = timed(lambda: (du[flat] * ww.unsqueeze(-1)).sum(1), a.rep)
    print(f"  G2P(B-스플라인 3^3, {ng}^3)   {t:6.2f} ms", flush=True)
    del du
    torch.cuda.empty_cache()


# ---- 3) 그 격자에서 도는 conv 비용
for ng in (32, 100):
    net = ConvStepper(n_feat=30, hidden=a.hidden, depth=a.depth, h=float(hh),
                      scale=0.02, skin_out=True, arch="plain").to(dev)
    feat = torch.randn(ng ** 3, 30, device=dev)
    gp = torch.zeros(ng ** 3, 3, device=dev)
    with torch.no_grad():
        try:
            t = timed(lambda: net(gp, feat, 1 / 60,
                                  (ng, ng, ng)), a.rep if ng == 32 else 10)
            print(f"  3D conv (은닉 {a.hidden}, 깊이 {a.depth}, {ng}^3) "
                  f"{t:6.2f} ms", flush=True)
        except RuntimeError as e:
            print(f"  3D conv {ng}^3: 실패 ({str(e)[:60]})", flush=True)
    del net, feat, gp
    torch.cuda.empty_cache()
print("TRANSFER_BENCH_OK", flush=True)
