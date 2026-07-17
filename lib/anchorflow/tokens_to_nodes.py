"""Semantic-guided anchor node selection (From Tokens to Nodes, arXiv:2510.02732).

Pipeline for 3DGS canonical scenes (no monocular video, no Track Anything):
  1. Render canonical 3DGS from N views → RGB images
  2. DINOv2 ViT-B/14 patch features per image → [nH*nW, D]
  3. Project Gaussians to each camera → per-patch mean depth + opacity (foreground proxy)
  4. Back-project patch centers to 3D → candidate node cloud
  5. Iterative bipartite soft matching:
       sim(i,j) = cos(z_i, z_j) - eta * M_fg(i,j)
     dynamic / high-opacity regions → preserved
     static / uniform regions → merged aggressively
  6. Return [M, 3] node positions, M ≈ n_nodes
"""
from __future__ import annotations
import math
from typing import List, Tuple
import torch
import torch.nn.functional as F


# ── DINOv2 ──────────────────────────────────────────────────────────────────

_dino_model = None

def _get_dino():
    global _dino_model
    if _dino_model is None:
        _dino_model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
        ).cuda().eval()
    return _dino_model


def _extract_dino(image: torch.Tensor, patch_size: int = 14, model=None):
    """image [3,H,W] in [0,1] on CUDA. Returns (features [nH*nW, D], nH, nW)."""
    if model is None:
        model = _get_dino()
    H, W = image.shape[1], image.shape[2]
    nH, nW = H // patch_size, W // patch_size
    x = image[:, :nH * patch_size, :nW * patch_size].unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        feats = model.get_intermediate_layers(x, n=1, return_class_token=False)[0]
    return feats.squeeze(0), nH, nW   # [nH*nW, D]


# ── Camera projection ────────────────────────────────────────────────────────

def _project(xyz: torch.Tensor, cam):
    """Project Gaussian centers to pixel space.
    Returns (uv [G,2] pixel, z_cam [G] depth in camera space)."""
    w2v = cam.world_view_transform        # [4,4]
    xyz_h = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device)], dim=-1)
    xyz_cam = (xyz_h @ w2v)[:, :3]       # [G,3]
    z_cam = xyz_cam[:, 2]
    H, W = cam.image_height, cam.image_width
    fx = W / (2 * math.tan(cam.FoVx * 0.5))
    fy = H / (2 * math.tan(cam.FoVy * 0.5))
    u = xyz_cam[:, 0] / z_cam.clamp(min=1e-4) * fx + W / 2
    v = xyz_cam[:, 1] / z_cam.clamp(min=1e-4) * fy + H / 2
    return torch.stack([u, v], dim=-1), z_cam


def _backproject(nH: int, nW: int, ps: int, cam, depths: torch.Tensor) -> torch.Tensor:
    """Patch centers → 3D world positions. depths [nH*nW]."""
    H, W = cam.image_height, cam.image_width
    fx = W / (2 * math.tan(cam.FoVx * 0.5))
    fy = H / (2 * math.tan(cam.FoVy * 0.5))
    dev = depths.device
    r = torch.arange(nH, device=dev)
    c = torch.arange(nW, device=dev)
    rr, cc = torch.meshgrid(r, c, indexing="ij")
    u = (cc.flatten() + 0.5) * ps
    v = (rr.flatten() + 0.5) * ps
    z = depths.clamp(min=1e-3)
    xyz_cam = torch.stack([(u - W/2) / fx * z, (v - H/2) / fy * z, z], dim=-1)
    w2v = cam.world_view_transform
    c2w = torch.linalg.inv(w2v)
    xyz_h = torch.cat([xyz_cam, torch.ones(xyz_cam.shape[0], 1, device=dev)], dim=-1)
    return (xyz_h @ c2w)[:, :3]


# ── Bipartite soft matching ──────────────────────────────────────────────────

def _cos_sim(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return F.normalize(A, dim=-1) @ F.normalize(B, dim=-1).T


def _merge_once(
    pos: torch.Tensor, feat: torch.Tensor, fg: torch.Tensor,
    voxel_size: float, eta: float, alpha: float, beta: float,
    r_min: float, r_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One round of voxel-based bipartite merging. Returns (pos, feat, fg)."""
    vox = (pos / voxel_size).floor().long()
    vox_min = vox.min(0).values
    vox = vox - vox_min
    vox_s = vox.max(0).values + 1
    flat = vox[:, 0] * vox_s[1] * vox_s[2] + vox[:, 1] * vox_s[2] + vox[:, 2]

    new_pos, new_feat, new_fg = [], [], []
    for uid in flat.unique():
        mask = (flat == uid).nonzero(as_tuple=True)[0]
        n = mask.shape[0]
        if n == 1:
            new_pos.append(pos[mask]); new_feat.append(feat[mask]); new_fg.append(fg[mask])
            continue

        # Dynamic tendency
        fv = feat[mask]
        gv = fg[mask]
        if n > 1:
            off = ~torch.eye(n, dtype=torch.bool, device=pos.device)
            mean_sim = _cos_sim(fv, fv)[off].mean()
        else:
            mean_sim = torch.tensor(1.0, device=pos.device)
        p_dyn = torch.sigmoid(alpha * gv.mean() - beta * mean_sim)
        r = r_min + (1.0 - p_dyn.item()) * (r_max - r_min)
        n_merge = max(0, int(n * r) - 1)

        if n_merge == 0 or n <= 2:
            new_pos.append(pos[mask]); new_feat.append(feat[mask]); new_fg.append(fg[mask])
            continue

        half = n // 2
        A_idx = mask[:half]; B_idx = mask[half:]
        sim = _cos_sim(feat[A_idx], feat[B_idx])
        fg_pen = (fg[A_idx].unsqueeze(1) + fg[B_idx].unsqueeze(0)) / 2
        sim = sim - eta * fg_pen
        best_b = sim.argmax(dim=1)
        pair_sim = sim[torch.arange(len(A_idx)), best_b]
        _, top = pair_sim.topk(min(n_merge, len(A_idx)))

        mA = A_idx[top]; mB = B_idx[best_b[top]]
        used_A = torch.zeros(len(A_idx), dtype=torch.bool, device=pos.device)
        used_B = torch.zeros(len(B_idx), dtype=torch.bool, device=pos.device)
        used_A[top] = True; used_B[best_b[top]] = True

        # merged
        new_pos.append((pos[mA] + pos[mB]) / 2)
        new_feat.append((feat[mA] + feat[mB]) / 2)
        new_fg.append((fg[mA] + fg[mB]) / 2)
        # unmerged
        keep = torch.cat([A_idx[~used_A], B_idx[~used_B]])
        new_pos.append(pos[keep]); new_feat.append(feat[keep]); new_fg.append(fg[keep])

    return torch.cat(new_pos), torch.cat(new_feat), torch.cat(new_fg)


def _bipartite_compress(
    pos: torch.Tensor, feat: torch.Tensor, fg: torch.Tensor,
    target: int, eta: float = 0.5, alpha: float = 1.0, beta: float = 1.0,
    r_min: float = 0.1, r_max: float = 0.9, growth: float = 1.5,
) -> torch.Tensor:
    extent = (pos.max(0).values - pos.min(0).values).norm().item()
    vs = extent / 30.0
    for _ in range(50):
        if pos.shape[0] <= target:
            break
        pos, feat, fg = _merge_once(pos, feat, fg, vs, eta, alpha, beta, r_min, r_max)
        vs *= growth
    # Final fallback: opacity-weighted sampling if still over target
    if pos.shape[0] > target:
        w = fg.clamp(min=0.01)
        idx = torch.multinomial(w, num_samples=target, replacement=False)
        pos = pos[idx]
    return pos


# ── Public API ───────────────────────────────────────────────────────────────

def tokens_to_nodes(
    canonical_xyz: torch.Tensor,          # [G, 3]
    canonical_opacities: torch.Tensor,    # [G, 1] or [G] — sigmoid opacity
    render_fn,                             # fn(cam) -> [3, H, W] in [0,1]
    cameras: list,
    n_nodes: int = 256,
    patch_size: int = 14,
    eta: float = 0.5,
    device: str = "cuda",
) -> torch.Tensor:
    """Returns [M, 3] anchor node positions, M ≈ n_nodes."""
    opacity = canonical_opacities.view(-1).float()
    dino = _get_dino()
    all_pos, all_feat, all_fg = [], [], []

    for cam in cameras:
        with torch.no_grad():
            img = render_fn(cam).float()                       # [3,H,W]

        feat, nH, nW = _extract_dino(img, patch_size, dino)   # [nH*nW, D]
        N_patches = nH * nW
        H_use, W_use = nH * patch_size, nW * patch_size

        uv, z_cam = _project(canonical_xyz, cam)

        patch_depth = torch.zeros(N_patches, device=device)
        patch_cnt   = torch.zeros(N_patches, device=device)
        patch_fg    = torch.zeros(N_patches, device=device)

        valid = (z_cam > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W_use) & \
                (uv[:, 1] >= 0) & (uv[:, 1] < H_use)
        if valid.sum() > 0:
            uv_v = uv[valid]; zv = z_cam[valid]; fgv = opacity[valid]
            pi = (uv_v[:, 1] / patch_size).long().clamp(0, nH - 1)
            pj = (uv_v[:, 0] / patch_size).long().clamp(0, nW - 1)
            pidx = pi * nW + pj
            patch_depth.scatter_add_(0, pidx, zv)
            patch_cnt.scatter_add_(0, pidx, torch.ones_like(zv))
            patch_fg.scatter_add_(0, pidx, fgv)

        has = patch_cnt > 0
        patch_depth[has] /= patch_cnt[has]
        patch_fg[has]    /= patch_cnt[has]
        mean_d = z_cam[valid].mean() if valid.sum() > 0 else torch.tensor(3.0, device=device)
        patch_depth[~has] = mean_d

        pos3d = _backproject(nH, nW, patch_size, cam, patch_depth).float()
        all_pos.append(pos3d)
        all_feat.append(feat.float())
        all_fg.append(patch_fg)

    pos   = torch.cat(all_pos)
    feat  = torch.cat(all_feat)
    fg    = torch.cat(all_fg)
    print(f"[tokens_to_nodes] candidates={pos.shape[0]}  target={n_nodes}")
    nodes = _bipartite_compress(pos, feat, fg, n_nodes, eta=eta)
    print(f"[tokens_to_nodes] final={nodes.shape[0]}")
    return nodes.to(device)
