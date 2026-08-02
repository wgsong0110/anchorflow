import os, sys
_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)
import torch
from anchorflow.anchor_mpm import AnchorElasticSim, _polar_decompose

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
M, N, K = 64, 500, 8
anchor_canonical = torch.randn(M, 3, device=dev)
gaussian_canonical = torch.randn(N, 3, device=dev) * 1.2
sim = AnchorElasticSim(gaussian_canonical, anchor_canonical, K=K)

theta = 0.7
axis = torch.tensor([0.3, 1.0, -0.2], device=dev); axis = axis / axis.norm()
Kx = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], device=dev)
R_true = torch.eye(3, device=dev) + torch.sin(torch.tensor(theta)) * Kx + (1 - torch.cos(torch.tensor(theta))) * (Kx @ Kx)
d_true = torch.tensor([0.5, -0.3, 0.8], device=dev)
anchor_rotated = (R_true @ anchor_canonical.T).T + d_true
expected_gaussian = (R_true @ gaussian_canonical.T).T + d_true
gaussian_pos_prev = expected_gaussian.clone()

with torch.no_grad():
    w = sim._weights(gaussian_pos_prev, anchor_rotated)
    F, gaussian_pos = sim._shape_match(anchor_rotated, w)

err = (F - R_true.unsqueeze(0)).abs().amax(dim=(-2, -1))   # [N]
print("F error percentiles: p50=%.4e p90=%.4e p99=%.4e max=%.4e" % (
    err.quantile(0.5).item(), err.quantile(0.9).item(), err.quantile(0.99).item(), err.max().item()))
n_bad = (err > 0.01).sum().item()
print(f"n_bad (err>0.01) = {n_bad} / {N}")

worst = err.argmax().item()
print(f"\nworst Gaussian idx={worst}, err={err[worst].item():.4e}")
print("F[worst]:\n", F[worst])
print("R_true:\n", R_true)

nbr_rest = sim.anchor_nbr[worst]
nbr_cur = anchor_rotated[sim.nn_idx[worst]]
w_worst = w[worst]
print("\nw[worst]:", w_worst)
rest_c = (w_worst.unsqueeze(-1) * nbr_rest).sum(0)
cur_c = (w_worst.unsqueeze(-1) * nbr_cur).sum(0)
q = nbr_rest - rest_c
p = nbr_cur - cur_c
B = torch.einsum("k,ki,kj->ij", w_worst, q, q)
A = torch.einsum("k,ki,kj->ij", w_worst, p, q)
print("B[worst]:\n", B)
print("eigvals(B[worst]):", torch.linalg.eigvalsh(B))
reg = 1e-2 * (B.diagonal().sum() / 3.0)
print("reg scalar:", reg.item())
F_worst_manual = A @ torch.linalg.inv(B + reg * torch.eye(3, device=dev))
print("F_worst (manual recompute):\n", F_worst_manual)

print("\n--- polar decompose check on a KNOWN-GOOD F (=R_true) ---")
R_pd, S_pd = _polar_decompose(R_true.unsqueeze(0))
print("R_pd (should == R_true):\n", R_pd[0])
print("S_pd (should == I):\n", S_pd[0])

print("\n--- polar decompose check on F[worst] ---")
R_pd2, S_pd2 = _polar_decompose(F[worst:worst+1])
print("det(F[worst]):", torch.linalg.det(F[worst]).item())
print("R_pd2:\n", R_pd2[0])
print("S_pd2:\n", S_pd2[0])
