"""SAM2 multi-view back-projection voting for Gaussian object segmentation.

Pipeline:
  1. Render canonical 3DGS from N cameras
  2. Run SAM2 automatic mask generation on each render
     (falls back to pixel-color K-means if SAM2 unavailable)
  3. Project each Gaussian centre onto every camera → sample the mask colour
  4. Cluster Gaussians in (position, avg-mask-colour) space → object IDs
  5. Majority-vote Gaussian IDs up to anchor level
"""
from __future__ import annotations

import numpy as np
import torch


# ── projection ────────────────────────────────────────────────────────────── #

def _project_to_pixels(xyz: torch.Tensor, cam) -> tuple:
    """Project world-space points [N,3] → (px, py, valid) all [N] on CPU."""
    N = xyz.shape[0]
    ones = torch.ones(N, 1, device=xyz.device, dtype=xyz.dtype)
    xyzw = torch.cat([xyz, ones], dim=1)              # [N, 4]
    clip  = xyzw @ cam.full_proj_transform             # [N, 4]
    w     = clip[:, 3]
    ndc   = clip[:, :2] / w.unsqueeze(1).clamp(min=1e-8)
    W, H  = cam.image_width, cam.image_height
    px    = ((ndc[:, 0] + 1.0) * 0.5 * W - 0.5).long()
    py    = ((ndc[:, 1] + 1.0) * 0.5 * H - 0.5).long()
    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (w > 0)
    return px.cpu(), py.cpu(), valid.cpu()


# ── main segmentation ─────────────────────────────────────────────────────── #

def segment_gaussians(
    gaussian_xyz: torch.Tensor,          # [N, 3] world positions (cuda)
    cameras: list,                       # list of Cam (with full_proj_transform)
    renders: list,                       # [V] tensors [3,H,W] float [0,1] (cuda)
    n_objects: int = 5,
    gaussian_colors: torch.Tensor = None,  # [N, 3] SH0 albedo (optional hint)
    sam2_checkpoint: str = None,
    sam2_cfg: str = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Returns object_ids [N] long tensor (0-indexed)."""

    N = gaussian_xyz.shape[0]
    V = len(cameras)

    # ── try SAM2 ─────────────────────────────────────────────────────────── #
    mask_gen = None
    if sam2_checkpoint is not None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SamAutomaticMaskGenerator
            _sam2 = build_sam2(sam2_cfg, sam2_checkpoint,
                               device=device, apply_postprocessing=False)
            mask_gen = SamAutomaticMaskGenerator(
                _sam2,
                points_per_side=16,
                pred_iou_thresh=0.80,
                stability_score_thresh=0.90,
                min_mask_region_area=200,
            )
            print("[segment] SAM2 loaded", flush=True)
        except Exception as e:
            print(f"[segment] SAM2 unavailable ({e}), using colour K-means", flush=True)

    # ── per-view mask-colour voting ───────────────────────────────────────── #
    # For each Gaussian, accumulate the mean-RGB of the mask it falls into
    # across all views, then cluster in (position + colour) feature space.
    colour_sum = torch.zeros(N, 3, dtype=torch.float32)
    colour_cnt = torch.zeros(N,    dtype=torch.float32)

    for v, (cam, render) in enumerate(zip(cameras, renders)):
        img_np = (render.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                  ).astype(np.uint8)
        H, W = img_np.shape[:2]

        if mask_gen is not None:
            masks_info = mask_gen.generate(img_np)
            # pixel → mean-colour of its mask
            colour_map = np.zeros((H, W, 3), dtype=np.float32)
            for m in masks_info:
                seg = m["segmentation"]  # [H, W] bool
                mc  = img_np[seg].mean(axis=0) / 255.0
                colour_map[seg] = mc
        else:
            # Fallback: use the render pixel colour directly
            colour_map = render.permute(1, 2, 0).cpu().float().numpy()

        px, py, valid = _project_to_pixels(gaussian_xyz, cam)
        px_c = px.clamp(0, W - 1).numpy()
        py_c = py.clamp(0, H - 1).numpy()
        v_np = valid.numpy()

        sampled = colour_map[py_c[v_np], px_c[v_np]]   # [n_visible, 3]
        colour_sum[v_np] += torch.from_numpy(sampled)
        colour_cnt[v_np] += 1.0

        print(f"[segment] view {v+1}/{V}  visible={int(v_np.sum())}/{N}", flush=True)

    avg_colour = colour_sum / colour_cnt.unsqueeze(1).clamp(min=1.0)  # [N, 3]

    # ── cluster in (position + colour) ──────────────────────────────────── #
    xyz_np = gaussian_xyz.cpu().float().numpy()
    xyz_n  = (xyz_np - xyz_np.mean(0)) / (xyz_np.std() + 1e-8)

    feat = np.concatenate([
        xyz_n * 1.0,               # spatial structure
        avg_colour.numpy() * 2.0,  # semantic colour (weighted more)
    ], axis=1)                      # [N, 6]

    # Optionally append SH0 albedo for extra signal
    if gaussian_colors is not None:
        feat = np.concatenate([feat, gaussian_colors.cpu().float().numpy()], axis=1)

    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n_objects, n_init=10,
                         max_iter=300, random_state=0)
    ids = km.fit_predict(feat)
    print(f"[segment] clustered {N} Gaussians → {n_objects} objects", flush=True)
    return torch.tensor(ids, dtype=torch.long)


# ── anchor-level aggregation ──────────────────────────────────────────────── #

def assign_anchor_objects(
    gaussian_object_ids: torch.Tensor,  # [N] long
    binding_idx: torch.Tensor,          # [N, K] anchor indices per Gaussian
    binding_weights: torch.Tensor,      # [N, K] weights
    n_anchors: int,
    n_objects: int,
) -> torch.Tensor:
    """Weighted majority vote: Gaussian object IDs → anchor object IDs [M]."""
    N, K = binding_idx.shape
    votes = torch.zeros(n_anchors, n_objects, dtype=torch.float32)
    for k in range(K):
        w   = binding_weights[:, k]       # [N]
        aidx = binding_idx[:, k]          # [N] anchor index
        obj  = gaussian_object_ids        # [N]
        oh   = torch.zeros(N, n_objects)
        oh.scatter_(1, obj.unsqueeze(1), w.unsqueeze(1))
        votes.scatter_add_(0, aidx.unsqueeze(1).expand(-1, n_objects), oh)
    return votes.argmax(dim=1)            # [M] long
