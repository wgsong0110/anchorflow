#!/usr/bin/env python
"""Align a DreamPhysics/PhysGaussian static 3DGS (.ply) into the SV4D dataset's
object frame, then emit a `points3d.ply` (xyz + rgb) that SC-GS uses as its
initial point cloud (full-object geometry -> no cut-off at SV4D-unseen angles).

Frame facts (from the SV4D transforms):
  * up axis = world +y ; cameras orbit the xz-plane at radius R (elev 0) ;
    object centred at origin ; view_00 (azimuth 0-of-the-ring) == the SVD/SV4D
    conditioning render.
DreamPhysics 3DGS is +z-up, off-centre, and metric-scaled, so we:
  1) centre on its bbox centre,
  2) rotate +z-up -> +y-up  (R_x(-90deg): (x,y,z)->(x, z, -y)),
  3) scale bbox-diag -> target object diameter,
  4) resolve the free azimuth (rotation about +y) by silhouette IoU against the
     real view_00 image (render each candidate, keep the best).

    python exe/align_ficus.py --ply ficus.ply --dataset ficus_ds \
        --target_diam 1.2 --out ficus_ds/points3d.ply
Runs where a gaussian-splatting renderer is importable (SC-GS image); reuses the
INRIA GaussianModel + rasteriser exactly like render_std.py.
"""
from __future__ import annotations
import argparse, os, sys, math, json
import numpy as np
import torch

sys.path.append("gaussian-splatting")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from plyfile import PlyData, PlyElement


class Pipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False


class Cam:
    def __init__(self, R, T, fovx, fovy, W, H):
        self.image_width, self.image_height = W, H
        self.FoVx, self.FoVy = fovx, fovy
        self.znear, self.zfar = 0.01, 100.0
        w2v = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        proj = getProjectionMatrix(self.znear, self.zfar, fovx, fovy).transpose(0, 1).cuda()
        self.world_view_transform = w2v
        self.full_proj_transform = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def sh_deg_of(path):
    names = [p.name for p in PlyData.read(path)["vertex"].properties
             if p.name.startswith("f_rest_")]
    return int(math.sqrt((len(names) + 3) // 3)) - 1 if names else 0


def rot_x(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def quat_mul_R(q, R):
    """Left-multiply each gaussian quaternion (w,x,y,z) by rotation matrix R."""
    from scipy.spatial.transform import Rotation as Rot
    r = Rot.from_matrix(R)
    qx = q[:, [1, 2, 3, 0]]                       # wxyz -> xyzw
    out = (r * Rot.from_quat(qx)).as_quat()       # xyzw
    return out[:, [3, 0, 1, 2]]                   # -> wxyz


def apply_world_R(g, R):
    """Rotate a loaded GaussianModel in place by world rotation R (numpy 3x3)."""
    Rt = torch.tensor(R, device="cuda")
    g._xyz = torch.nn.Parameter((g._xyz @ Rt.T).contiguous())
    q = g._rotation.detach().cpu().numpy()
    q = quat_mul_R(q, R)
    g._rotation = torch.nn.Parameter(torch.tensor(q, dtype=torch.float32, device="cuda"))


def load_view0(dataset):
    d = json.load(open(os.path.join(dataset, "transforms_train.json")))
    fovx = float(d["camera_angle_x"])
    # first frame of view_00
    f0 = sorted([f for f in d["frames"] if "/view_00/" in f["file_path"]],
                key=lambda f: f["file_path"])[0]
    c2w = np.array(f0["transform_matrix"], dtype=np.float32)
    img_path = os.path.join(dataset, f0["file_path"].lstrip("./"))
    if not os.path.exists(img_path):
        img_path += ".png"
    return c2w, fovx, img_path


def silhouette(arr, thr=0.95):
    """foreground mask = pixels not near white."""
    return (arr.mean(-1) < thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--dataset", required=True, help="SV4D ficus_ds (transforms + images)")
    ap.add_argument("--out", required=True, help="points3d.ply path")
    ap.add_argument("--target_diam", type=float, default=1.2)
    ap.add_argument("--az_step", type=float, default=15.0)
    ap.add_argument("--res", type=int, default=400)
    ap.add_argument("--dump_dir", default=None, help="save az-search renders for inspection")
    args = ap.parse_args()

    from PIL import Image
    sh = sh_deg_of(args.ply)
    g = GaussianModel(sh); g.load_ply(args.ply); g.active_sh_degree = sh
    xyz = g.get_xyz.detach()
    center = (xyz.min(0).values + xyz.max(0).values) / 2.0
    diag = float((xyz.max(0).values - xyz.min(0).values).norm())
    scale = args.target_diam / diag
    print(f"[align] N={xyz.shape[0]} sh={sh} center={center.cpu().numpy().round(3)} diag={diag:.3f} scale={scale:.4f}")

    # 1) centre, 2) z-up->y-up, 3) scale  (positions; also rotate gaussian frames)
    g._xyz = torch.nn.Parameter((g._xyz - center).contiguous())
    apply_world_R(g, rot_x(-90.0))
    g._xyz = torch.nn.Parameter((g._xyz * scale).contiguous())
    g._scaling = torch.nn.Parameter(g._scaling + math.log(scale))   # log-scale shift

    # ---- azimuth search against real view_00 ----
    c2w, fovx, img_path = load_view0(args.dataset)
    fovy = fovx
    Rc = c2w[:3, :3]; eye = c2w[:3, 3]
    T = -Rc.T @ eye
    W = H = args.res
    cam = Cam(Rc.astype(np.float32), T.astype(np.float32), fovx, fovy, W, H)
    gt = np.array(Image.open(img_path).convert("RGB").resize((W, H))) / 255.0
    gt_fg = silhouette(gt)
    pipe = Pipe(); bg = torch.tensor([1., 1, 1], device="cuda")
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        Image.fromarray((gt * 255).astype(np.uint8)).save(os.path.join(args.dump_dir, "gt_view00.png"))

    best_iou, best_az = -1.0, 0.0
    q_backup = g._rotation.detach().clone(); xyz_backup = g._xyz.detach().clone()
    for az in np.arange(0, 360, args.az_step):
        g._xyz = torch.nn.Parameter(xyz_backup.clone()); g._rotation = torch.nn.Parameter(q_backup.clone())
        apply_world_R(g, rot_y(float(az)))
        with torch.no_grad():
            out = render(cam, g, pipe, bg)["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        fg = silhouette(out)
        inter = (fg & gt_fg).sum(); union = (fg | gt_fg).sum()
        iou = inter / max(union, 1)
        if args.dump_dir:
            Image.fromarray((out * 255).astype(np.uint8)).save(os.path.join(args.dump_dir, f"az_{int(az):03d}_iou{iou:.2f}.png"))
        print(f"[align] az={az:5.0f} IoU={iou:.3f}")
        if iou > best_iou:
            best_iou, best_az = iou, float(az)
    print(f"[align] BEST az={best_az} IoU={best_iou:.3f}")

    # apply best azimuth
    g._xyz = torch.nn.Parameter(xyz_backup.clone()); g._rotation = torch.nn.Parameter(q_backup.clone())
    apply_world_R(g, rot_y(best_az))

    # ---- write points3d.ply (xyz + rgb from SH DC) ----
    from utils.sh_utils import SH2RGB
    xyz = g.get_xyz.detach().cpu().numpy()
    dc = g._features_dc.detach().squeeze(1).cpu().numpy()   # [N,3] SH band-0
    rgb = np.clip(SH2RGB(dc), 0, 1)
    elem = np.empty(xyz.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                         ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    elem["x"], elem["y"], elem["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elem["red"], elem["green"], elem["blue"] = (rgb[:, 0] * 255).astype(np.uint8), \
        (rgb[:, 1] * 255).astype(np.uint8), (rgb[:, 2] * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    PlyData([PlyElement.describe(elem, "vertex")]).write(args.out)
    print(f"[align] wrote {xyz.shape[0]} pts -> {args.out}  (az={best_az}, IoU={best_iou:.3f})")
    # also save the full aligned gaussian ply (for optional full load_ply init)
    full = os.path.splitext(args.out)[0] + "_full_gaussians.ply"
    g.save_ply(full)
    print(f"[align] wrote aligned full gaussians -> {full}")


if __name__ == "__main__":
    main()
