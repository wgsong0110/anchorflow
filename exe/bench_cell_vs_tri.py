"""셀 집계(+셀->격자점 conv) 와 8꼭짓점 trilinear 집계의 시간을 비교한다."""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch
import torch.nn as nn
from anchorflow import trilinear as TRI, vox_anchor

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=250000)
ap.add_argument("--res", type=int, nargs="+", default=[32, 50])
ap.add_argument("--feat", type=int, default=39)
ap.add_argument("--rep", type=int, default=20)
a = ap.parse_args()
dev = "cuda:0"
torch.manual_seed(0)
q = torch.randn(a.n, 3, device=dev)
q = q / q.norm(dim=1, keepdim=True) * torch.rand(a.n, 1, device=dev) ** (1/3)
x = q * 0.4 + 0.5
v = torch.randn_like(x) * 0.01
X = x.clone()
m = torch.full((a.n,), 1e-6, device=dev)

def timeit(fn, rep):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(rep): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/rep*1e3

for R in a.res:
    lo, h, n3 = vox_anchor.grid_for(x, R ** 3)
    # --- 현행: 8꼭짓점 trilinear
    flat, w = TRI.corners(x, lo, h, n3)
    rows, uniq = TRI.active(flat)
    co, pa = TRI.unflatten(uniq, n3, lo, h, x.dtype)
    t_tri = timeit(lambda: TRI.tri_feats(x, v, X, m, rows, w, pa.shape[0], pa,
                                         float(h)), a.rep)
    # --- 제안: 셀 하나에만 집계 (K=1)
    crow, cw, nc = TRI.cell_index(x, lo, h, n3)
    M_cell = nc[0] * nc[1] * nc[2]
    cen = torch.zeros(M_cell, 3, device=dev)
    t_cell = timeit(lambda: TRI.tri_feats(x, v, X, m, crow, cw, M_cell, cen,
                                          float(h)), a.rep)
    # --- 셀 -> 격자점 변환 conv (2^3, pad 1)
    F = TRI.tri_feats(x, v, X, m, crow, cw, M_cell, cen, float(h)).shape[-1]
    conv = nn.Conv3d(F, F, 2, padding=1).to(dev)
    vol = torch.randn(1, F, *nc, device=dev)
    t_conv = timeit(lambda: conv(vol), a.rep)
    out = conv(vol)
    print(f"[res {R}] 격자 {tuple(int(t) for t in n3)} 셀 {tuple(nc)} 특징 {F}채널",
          flush=True)
    print(f"  trilinear 집계 {t_tri:6.2f} ms", flush=True)
    print(f"  셀 집계 {t_cell:6.2f} ms + 셀->격자점 conv {t_conv:5.2f} ms "
          f"= {t_cell + t_conv:6.2f} ms  ({t_tri/(t_cell+t_conv):.2f} 배 빠름)",
          flush=True)
    print(f"  conv 출력 {tuple(out.shape[2:])} (격자점 {tuple(int(t) for t in n3)} 과 일치해야)",
          flush=True)
print("CELLBENCH_OK", flush=True)
