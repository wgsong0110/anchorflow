#!/usr/bin/env python
"""Parity: fused CUDA cov_warp vs the torch reference (must be quality-neutral)."""
import sys, os, torch, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow import warp as W
from anchorflow import geom
import lbs

print("cov CUDA kernel:", lbs._HAVE_COV_CUDA)
torch.manual_seed(0)
N, M, K = 1_852_335, 242, 4
cov6 = torch.rand(N, 6, device="cuda") * 0.01
cov6[:, 0] += 0.05; cov6[:, 3] += 0.05; cov6[:, 5] += 0.05   # PSD-ish diagonal
w = torch.rand(N, K, device="cuda"); w = w / w.sum(-1, keepdim=True)
idx = torch.randint(0, M, (N, K), device="cuda")
Rm = geom.quat_to_matrix(torch.randn(M, 4, device="cuda"))
quat = geom.matrix_to_quat(Rm)

def ref():
    qg = W._blend_quat(quat, w, idx)
    Rg = geom.quat_to_matrix(qg)
    S = W.cov6_to_mat3(cov6)
    return W.mat3_to_cov6(Rg @ S @ Rg.transpose(-1, -2))

with torch.no_grad():
    out_ref = ref()
    out_cuda = lbs.cov_warp(quat, w, idx, cov6)

d = (out_cuda - out_ref).abs()
rel = d / (out_ref.abs() + 1e-6)
print(f"forward max abs err: {float(d.max()):.3e}")
print(f"forward max rel err: {float(rel.max()):.3e}")
print(f"forward mean abs err: {float(d.mean()):.3e}")

def bench(fn, n=5):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        with torch.no_grad(): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

t_ref = bench(ref)
t_cuda = bench(lambda: lbs.cov_warp(quat, w, idx, cov6))
print(f"\ntorch  : {t_ref:8.2f} ms")
print(f"cuda   : {t_cuda:8.2f} ms   ({t_ref/t_cuda:.1f}x)")
print(f"25프레임: {t_ref*25:.0f} ms -> {t_cuda*25:.0f} ms")
ok = float(d.max()) < 1e-4
print("\nPARITY", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
