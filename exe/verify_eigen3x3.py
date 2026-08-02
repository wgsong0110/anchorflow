"""Numeric parity check: lib/eigen3x3 (closed-form CUDA eigh) vs torch.linalg.eigh.

MUST pass before eigh3x3() is used anywhere real (see lib/eigen3x3/__init__.py's
warning) -- eigenvector sign/ordering ambiguity means we compare eigenVALUES
directly, and eigenvectors up to per-column sign flip (both +v and -v are valid
eigenvectors; downstream code here only ever uses V @ diag @ V^T-style
reconstructions or outer products that are sign-invariant, so this is the
correct notion of "match").
"""
from __future__ import annotations

import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from eigen3x3 import eigh3x3, _HAVE_CUDA

print(f"[verify] _HAVE_CUDA={_HAVE_CUDA}")
assert _HAVE_CUDA, "extension not built -- nothing to verify"

torch.manual_seed(0)
dev = "cuda"
N = 2000

# random symmetric PSD-ish matrices spanning a range of condition numbers,
# including near-degenerate (repeated-eigenvalue) and near-diagonal cases.
X = torch.randn(N, 3, 8, device=dev)
B = X @ X.transpose(-1, -2) / 8.0    # [N,3,3] symmetric PSD, Wishart-like spread
B[: N // 10] += 1e-6 * torch.eye(3, device=dev)  # near-degenerate subset
B[N // 10 : N // 5] = torch.diag_embed(torch.rand(N // 10, 3, device=dev))  # exactly diagonal subset
B = B.contiguous()

eigval_ref, eigvec_ref = torch.linalg.eigh(B.double())
eigval_ref, eigvec_ref = eigval_ref.float(), eigvec_ref.float()

Bg = B.clone().requires_grad_(True)
eigval_cuda, eigvec_cuda = eigh3x3(Bg)

val_err = (eigval_cuda - eigval_ref).abs()
print(f"[verify] eigenvalue error: mean={val_err.mean().item():.3e} max={val_err.max().item():.3e}")

# eigenvector check up to sign: align each column's sign via dot product before diffing
sign = torch.sign((eigvec_cuda * eigvec_ref).sum(dim=1, keepdim=True))
sign = torch.where(sign == 0, torch.ones_like(sign), sign)
vec_err = (eigvec_cuda * sign - eigvec_ref).abs()
print(f"[verify] eigenvector error (sign-aligned): mean={vec_err.mean().item():.3e} max={vec_err.max().item():.3e}")

# reconstruction check: does V @ diag(L) @ V^T actually recover B? (sign-invariant,
# the real test of "is this decomposition valid", independent of eigh's own sign convention)
recon = eigvec_cuda @ torch.diag_embed(eigval_cuda) @ eigvec_cuda.transpose(-1, -2)
recon_err = (recon - B).abs()
print(f"[verify] reconstruction |V L V^T - B| error: mean={recon_err.mean().item():.3e} max={recon_err.max().item():.3e}")

# orthogonality check
ortho = eigvec_cuda.transpose(-1, -2) @ eigvec_cuda
ortho_err = (ortho - torch.eye(3, device=dev)).abs()
print(f"[verify] orthogonality |V^T V - I| error: mean={ortho_err.mean().item():.3e} max={ortho_err.max().item():.3e}")

# --- backward parity: gradcheck against torch.linalg.eigh on a small batch ---
print("\n[verify] backward parity (small batch, float64 gradcheck-style comparison)")
n_small = 8
Bs = B[:n_small].clone()


def loss_cuda(Bt):
    ev, evec = eigh3x3(Bt)
    return (ev.sum() + (evec ** 2).sum())


def loss_ref(Bt):
    ev, evec = torch.linalg.eigh(Bt)
    return (ev.sum() + (evec ** 2).sum())


Bg1 = Bs.clone().requires_grad_(True)
loss_cuda(Bg1).backward()
grad_cuda = Bg1.grad.clone()

Bg2 = Bs.clone().requires_grad_(True)
loss_ref(Bg2).backward()
grad_ref = Bg2.grad.clone()

grad_err = (grad_cuda - grad_ref).abs()
print(f"[verify] dLoss/dB error (mean={grad_err.mean().item():.3e} max={grad_err.max().item():.3e})")

# --- SCALE INVARIANCE: the bug that broke the real ficus sim ---
# Real anchor-cloud B matrices had eigenvalues ~2e-5. The kernel's absolute
# thresholds (cross-product norm guard, "already diagonal" check) silently
# misfired at that magnitude and returned ZERO eigenvectors for
# perfectly well-conditioned input, so F collapsed to 0 and the simulation
# exploded. Test the SAME matrix across magnitudes -- results must be
# identical up to the overall scale factor.
print("\n[verify] scale invariance (same well-conditioned matrix, varying magnitude)")
Xs = torch.randn(64, 3, 8, device=dev)
B_unit = (Xs @ Xs.transpose(-1, -2) / 8.0).contiguous()
worst_vnorm_err = 0.0
worst_recon_rel = 0.0
for s in [1.0, 1e-2, 1e-4, 1e-5, 1e-6, 1e-8]:
    Bs = (B_unit * s).contiguous()
    ev_s, evec_s = eigh3x3(Bs.clone())
    ev_ref_s = torch.linalg.eigh(Bs.double())[0].float()
    vnorm = evec_s.norm(dim=1)                       # each column must be unit
    vnorm_err = (vnorm - 1.0).abs().max().item()
    recon_s = evec_s @ torch.diag_embed(ev_s) @ evec_s.transpose(-1, -2)
    recon_rel = ((recon_s - Bs).abs().max() / s).item()   # relative to the scale
    val_rel = ((ev_s - ev_ref_s).abs().max() / s).item()
    worst_vnorm_err = max(worst_vnorm_err, vnorm_err)
    worst_recon_rel = max(worst_recon_rel, recon_rel)
    print(f"  scale={s:.0e}: |eigvec_col_norm - 1|max={vnorm_err:.3e}  "
          f"recon_rel={recon_rel:.3e}  eigval_rel={val_rel:.3e}")
print(f"  worst across scales: vnorm_err={worst_vnorm_err:.3e} recon_rel={worst_recon_rel:.3e} "
      f"({'PASS' if worst_vnorm_err < 1e-3 and worst_recon_rel < 1e-3 else 'FAIL'})")

# --- degenerate / repeated eigenvalues: eigenvectors must stay orthonormal ---
print("\n[verify] repeated-eigenvalue robustness (eigenvectors must stay orthonormal)")
B_deg = torch.eye(3, device=dev).expand(4, 3, 3).clone()          # all eigenvalues equal
B_deg[1] = torch.diag(torch.tensor([1.0, 1.0, 5.0], device=dev))   # two equal
B_deg[2] = torch.diag(torch.tensor([2.0, 2.0, 2.0], device=dev)) * 1e-6  # equal AND tiny
B_deg[3] = torch.zeros(3, 3, device=dev)                          # all-zero matrix
ev_d, evec_d = eigh3x3(B_deg.contiguous())
ortho_d = (evec_d.transpose(-1, -2) @ evec_d - torch.eye(3, device=dev)).abs().amax(dim=(-2, -1))
print(f"  |V^T V - I| per case: {[round(v, 6) for v in ortho_d.tolist()]}  "
      f"({'PASS' if ortho_d.max().item() < 1e-3 else 'FAIL'})")

print("\n[verify] PASS thresholds suggested: eigenvalue/recon/orthogonality max err < 1e-3, "
      "grad max err < 1e-2 (float32 + closed-form trig solver, not expecting float64-tight)")
