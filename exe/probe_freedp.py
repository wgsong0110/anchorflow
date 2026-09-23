"""표현력 진단: 신경망을 빼고 격자 변위 dp 를 **직접** 최적화한다.

한 창에서 dp 를 자유 변수로 두고 Adam 으로 내린다. 여기서 오차가 0 근처까지
안 내려가면 trilinear g2p 표현 자체가 목표 변위를 담지 못하는 것이고,
잘 내려가면 문제는 신경망/최적화 쪽이다. 같은 설정에서 Adam eps 도 바꿔 본다.
"""
from __future__ import annotations
import argparse, glob, os, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--vox_res", type=int, default=32)
ap.add_argument("--t0", type=int, default=5)
ap.add_argument("--n_pts", type=int, default=8000)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--lr", type=float, default=3e-4)
a = ap.parse_args()

dev = "cuda:0"
from anchorflow import trilinear as TRI, vox_anchor

f = sorted(glob.glob(os.path.join(a.data, "*.pt")))[0]
d = torch.load(f, map_location=dev, weights_only=False)
X = d["x"].float()
print(f"[궤적] {os.path.basename(f)}  x {tuple(X.shape)}", flush=True)
N_FULL = X.shape[1]
gsel = torch.arange(0, N_FULL, max(1, N_FULL // a.n_pts), device=dev)[:a.n_pts]
x = X[a.t0][gsel]
gt = X[a.t0 + 1][gsel]
EXT = float((X[0].max(0).values - X[0].min(0).values).norm())
still = float(((x - gt) ** 2).sum(-1).mean()) / EXT ** 2
print(f"[기준] EXT {EXT:.4f}  정지 오차 {100*still**0.5:.4f}%", flush=True)

lo, hh, nn3 = vox_anchor.grid_for(x, a.vox_res ** 3)
flat, w = TRI.corners(x, lo, hh, nn3)
M = int(nn3[0] * nn3[1] * nn3[2])
print(f"[격자] 격자점 {tuple(int(t) for t in nn3)} = {M}, 셀크기 {hh:.5f}", flush=True)

# 1) 최소제곱 하한 -- dp 를 자유 변수로 lr 크게 해서 충분히 내린다
for tag, lr, eps in (("lr=3e-4 eps=1e-8", 3e-4, 1e-8),
                     ("lr=3e-4 eps=1e-15", 3e-4, 1e-15),
                     ("lr=3e-2 eps=1e-8", 3e-2, 1e-8)):
    dp = torch.zeros(M, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([dp], lr=lr, eps=eps)
    for i in range(a.iters):
        opt.zero_grad()
        x2 = x + TRI.g2p(flat, w, dp)
        loss = ((x2 - gt) ** 2).sum(-1).mean() / EXT ** 2
        loss.backward()
        if i == 0:
            print(f"  [{tag}] dp 기울기 노름 {float(dp.grad.norm()):.3e}, "
                  f"원소당 {float(dp.grad.abs().mean()):.3e}", flush=True)
        opt.step()
    print(f"  [{tag}] {a.iters} 스텝 -> {100*float(loss)**0.5:.4f}%  "
          f"(정지 {100*still**0.5:.4f}%, 비 {(float(loss)/still)**0.5:.3f})", flush=True)
print("PROBE_OK", flush=True)
