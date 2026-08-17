"""The camera the config specifies, and a renderer for a deformed cloud.

Three scripts had grown their own copy of this forty-line block -- the same
azimuth/elevation construction, the same MiniCam, the same pipe/background --
and a camera that drifts between scripts makes their videos incomparable for
reasons that have nothing to do with what is being compared.

Imports of SC-GS happen inside the functions: this module is pulled in by things
that never render, and the rasteriser is not always built.
"""
from __future__ import annotations

import math

import numpy as np
import torch


class MiniCam:
    def __init__(self, W, H, fovy, fovx, zn, zf, wvt, fpt):
        self.image_width, self.image_height = W, H
        self.FoVy, self.FoVx = fovy, fovx
        self.znear, self.zfar = zn, zf
        self.world_view_transform = wvt
        self.full_proj_transform = fpt
        self.camera_center = wvt.inverse()[3, :3]


def build_camera(sc, width=640, height=640, fov_x=0.6911, radius_scale=1.6):
    """The view the config asks for: its own up axis, centre, azimuth and
    elevation, at a distance set by the object's own extent."""
    from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2

    dev = sc.pos.device
    cfg = sc.cfg
    center = sc.undo(torch.tensor(cfg["mpm_space_viewpoint_center"],
                                   device=dev).unsqueeze(0))[0].cpu().numpy()
    up_mpm = (torch.tensor(cfg["mpm_space_vertical_upward_axis"], device=dev)
              + torch.tensor(cfg["mpm_space_viewpoint_center"], device=dev)).unsqueeze(0)
    up = sc.undo(up_mpm)[0].cpu().numpy() - center
    up /= (np.linalg.norm(up) + 1e-9)
    xw = sc.xyz_world[sc.keep]
    extent = float((xw.max(0).values - xw.min(0).values).norm())
    az, el = math.radians(cfg["init_azimuthm"]), math.radians(cfg["init_elevation"])
    tmp = np.array([1., 0., 0.]) if abs(np.dot(np.array([1., 0., 0.]), up)) < 0.9 \
        else np.array([0., 1., 0.])
    h1 = np.cross(up, tmp); h1 /= np.linalg.norm(h1)
    h2 = np.cross(up, h1)
    eye = center + radius_scale * extent * (
        math.cos(el) * (math.cos(az) * h1 + math.sin(az) * h2) + math.sin(el) * up)
    fwd = center - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-9)
    tup = np.cross(right, fwd)
    Rc = np.stack([right, -tup, fwd], axis=1)
    Tc = -Rc.T @ eye
    fovy = focal2fov(width / (2 * math.tan(fov_x / 2)), height)
    wvt = torch.tensor(getWorld2View2(Rc, Tc)).transpose(0, 1).float().to(dev)
    pmx = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov_x,
                               fovY=fovy).transpose(0, 1).to(dev)
    return MiniCam(width, height, fovy, fov_x, 0.01, 100.0, wvt,
                    (wvt.unsqueeze(0).bmm(pmx.unsqueeze(0))).squeeze(0))


def make_renderer(sc, ply, cam):
    """-> f(gaussian positions [N,3]) -> uint8 image.

    Takes world-space positions and passes the displacement from canonical, which
    is the interface SC-GS's renderer wants.
    """
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render as _render

    dev = sc.pos.device
    gaussians = GaussianModel(3, fea_dim=0)
    gaussians.load_ply(ply)

    class _P:
        debug = False
        compute_cov3D_python = False
        convert_SHs_python = False

    pipe = _P()
    bg = torch.tensor([1., 1., 1.], device=dev)
    d_rot = torch.zeros(sc.N, 4, device=dev); d_rot[:, 0] = 1.
    d_sc = torch.zeros(sc.N, 3, device=dev)

    def frame(xyz_world):
        d_xyz = xyz_world - sc.xyz_world
        im = torch.clamp(_render(cam, gaussians, pipe, bg, d_xyz, d_rot, d_sc,
                                  d_rot_as_res=True)["render"], 0, 1)
        return (im.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

    return frame


def label(img, text):
    """a caption in the top-left, or the image unchanged if PIL is absent"""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 22], fill=(255, 255, 255))
    d.text((6, 5), text, fill=(0, 0, 0))
    return np.asarray(im)
