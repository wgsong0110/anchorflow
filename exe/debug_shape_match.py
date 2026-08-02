import os, sys
_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)
import torch
from anchorflow.anchor_mpm import AnchorElasticSim

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
M, N, K = 64, 500, 4
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
    print("weights sample [0]:", w[0])
    print("weights sum check:", w.sum(-1)[:5])

    nbr_rest = sim.anchor_nbr
    nbr_cur = anchor_rotated[sim.nn_idx]
    w_ = w.unsqueeze(-1)
    rest_centroid = (w_ * nbr_rest).sum(dim=1)
    cur_centroid = (w_ * nbr_cur).sum(dim=1)
    q = nbr_rest - rest_centroid.unsqueeze(1)
    p = nbr_cur - cur_centroid.unsqueeze(1)

    print("\nGaussian 0 debug:")
    print("  nbr_rest[0]:\n", nbr_rest[0])
    print("  nbr_cur[0]:\n", nbr_cur[0])
    print("  w[0]:", w[0])
    print("  q[0] (rest rel):\n", q[0])
    print("  p[0] (cur rel):\n", p[0])
    print("  R_true @ q[0].T (expected p[0]):\n", (R_true @ q[0].T).T)

    B = torch.einsum("nk,nki,nkj->nij", w, q, q)
    A = torch.einsum("nk,nki,nkj->nij", w, p, q)
    print("\n  B[0]:\n", B[0])
    print("  det(B[0]):", torch.linalg.det(B[0]).item())
    print("  cond(B[0]) via eigvals:", torch.linalg.eigvalsh(B[0]))
    print("  A[0]:\n", A[0])
    F0 = A[0] @ torch.linalg.inv(B[0] + 1e-9 * torch.eye(3, device=dev))
    print("  F[0] = A@Binv:\n", F0)
    print("  R_true:\n", R_true)
