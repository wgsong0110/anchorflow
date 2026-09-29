"""로컬(셀 내부 RQS) x 글로벌(상한 워프) 조합별 한 학습 스텝 실행시간.

한 스텝의 **순전파 + 역전파** 시간을 잰다 (물리손실 포함). 격자-입자 전달은
Kuhn 사면체 barycentric 하나이고, 두 단계는 독립이라 비용도 따로 얹힌다.

  python exe/bench_sitreg.py --traj traj_h2/mic_clayC_t_s400706.pt
"""
import argparse
import time

import torch

from anchorflow import phys_resid, tri_spline, vox_anchor
from anchorflow.sitreg_warp import BoundedWarp, WARP_BOUND, bary_g2p

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, default=10)
ap.add_argument("--n_pts", type=int, default=8000)
ap.add_argument("--vox_res", type=int, default=32)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--rqs_bins", type=int, default=8)
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
F0 = torch.eye(3, device=dev).expand(sel.numel(), 3, 3).contiguous()
mass = torch.full((sel.numel(),), float(cfg["density"]), device=dev)
vol = mass / float(cfg["density"])
ext = float((x.max(0).values - x.min(0).values).norm())
nrm = float(mass.sum()) * ext ** 2 / h ** 2

lo, hh, nn3 = vox_anchor.grid_for(x, a.vox_res ** 3)
hh = float(hh)
M = int(nn3[0] * nn3[1] * nn3[2])
NC = int((nn3[0] - 1) * (nn3[1] - 1) * (nn3[2] - 1))
P = tri_spline.n_params(a.rqs_bins)
BND = WARP_BOUND * hh
print(f"[상한] 격자점 변위 성분 |c| < {BND:.5f} (간격 {hh:.5f} 기준)")


def warp_bary(q, c):
    return q + bary_g2p(q, lo, hh, nn3, c)


def step(dp, mode, K, th=None):
    cw, nw = mode
    q = x
    if cw == "rqs":
        q = tri_spline.remap(q, lo, hh, nn3, th, a.rqs_bins)
    if nw == "bound":
        x2 = BoundedWarp(BND, K).apply(q, dp, warp_bary)
    else:
        x2 = warp_bary(q, dp)
    E, _dl, _Ft, _pt = phys_resid.grid_ip_energy(
        x, x2 - x, v, F0, mass, vol, cfg, h, ng, gl, g=gv, norm=nrm)
    return E


def timeit(tag, mode, K, amp=False, comp=False):
    dp = torch.zeros(M, 3, device=dev, requires_grad=True)
    th = (torch.zeros(NC, P, device=dev, requires_grad=True)
          if mode[0] == "rqs" else None)
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
        if th is not None and th.grad is not None:
            th.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            E = fn(dp, mode, K, th)
        E.backward()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / a.iters * 1000
    print(f"  {tag:40s} {ms:8.2f} ms", flush=True)
    return ms


print(f"[설정] 입자 {sel.numel()}, 격자점 {M}, 셀 {NC}, 반복 {a.iters}")
b0 = timeit("none + plain (맨 barycentric)", ("none", "plain"), 1)
timeit("rqs  + plain (재배열만)", ("rqs", "plain"), 1)
timeit("none + bound K=5 (상한만)", ("none", "bound"), 5)
for K in (1, 2, 5, 8):
    timeit(f"rqs  + bound K={K}", ("rqs", "bound"), K)
if a.compile:
    timeit("rqs + bound K=5, compile", ("rqs", "bound"), 5, comp=True)
    timeit("rqs + bound K=5, bf16 + compile", ("rqs", "bound"), 5,
           amp=True, comp=True)
print(f"\n기준선: none+plain {b0:.2f} ms")
