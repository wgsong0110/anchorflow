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


def local_deformation(rest, K=8, ridge=1e-8):
    """-> f(x) -> F [N,3,3], the deformation gradient at every Gaussian.

    The renderer moves Gaussian centres but leaves their covariance at rest, so
    a cloud that stretches renders as splats drifting apart: holes open up and
    the surface speckles. That reads as the material breaking when nothing of
    the sort happened -- it cost a round of investigation into whether MPM
    itself had come apart on tear_bread, and it had not.

    F is taken from the positions alone, by least squares over each Gaussian's K
    rest neighbours, rather than from either simulator's own state. Both panels
    of a comparison then get the same operator, so a difference on screen is a
    difference in where the Gaussians went and not in how the two sides were
    allowed to describe themselves.
    """
    from scipy.spatial import cKDTree

    dev = rest.device
    _, ni = cKDTree(rest.cpu().numpy()).query(rest.cpu().numpy(), k=K + 1)
    idx = torch.from_numpy(ni[:, 1:]).long().to(dev)            # [N,K]
    dX = rest[idx] - rest.unsqueeze(1)                          # [N,K,3]
    B = torch.einsum("nki,nkj->nij", dX, dX)
    B = B + ridge * torch.eye(3, device=dev) * B.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-12).view(-1, 1, 1)
    Binv = torch.linalg.inv(B)

    def apply(x):
        dx = x[idx] - x.unsqueeze(1)
        A = torch.einsum("nki,nkj->nij", dx, dX)
        return A @ Binv

    return apply


def cov_deltas(F, q_raw, scale, chunk=400_000):
    """F and the Gaussians' own shape -> the (d_rotation, d_scaling) this
    renderer wants.

    A Gaussian carries Sigma = R diag(s^2) R^T; deforming it by F gives
    F Sigma F^T = M M^T with M = F R diag(s). The singular values of M are the
    new scales and its left factor is the new rotation.

    get_rotation_bias adds to the raw quaternion parameter and normalises after,
    so the delta that lands on a target unit quaternion is simply target minus
    raw. Scale is additive on the activated scale.
    """
    N = F.shape[0]
    d_rot = torch.empty_like(q_raw)
    d_sc = torch.empty_like(scale)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        q = q_raw[lo:hi]
        qn = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        R = _quat_to_mat(qn)
        M = F[lo:hi] @ R @ torch.diag_embed(scale[lo:hi])
        U, S, _ = torch.linalg.svd(M)
        # a negative determinant would be a reflection, which is not a rotation
        det = torch.linalg.det(U)
        U = torch.cat([U[..., :2], U[..., 2:] * det.view(-1, 1, 1)], dim=-1)
        qn_new = _mat_to_quat(U)
        # q and -q are the same rotation; pick the nearer one so the additive
        # delta stays small and the normalisation does not flip anything
        flip = (qn_new * qn).sum(-1, keepdim=True) < 0
        qn_new = torch.where(flip, -qn_new, qn_new)
        d_rot[lo:hi] = qn_new - q
        d_sc[lo:hi] = S - scale[lo:hi]
    return d_rot, d_sc


def _quat_to_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)


def _mat_to_quat(R):
    m = R.reshape(-1, 9).unbind(-1)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m
    t = m00 + m11 + m22
    q = torch.empty(R.shape[0], 4, device=R.device, dtype=R.dtype)
    # the branch on the largest diagonal term is what keeps this stable when the
    # trace is near -1 and the naive formula divides by something near zero
    big = torch.stack([t, m00, m11, m22], -1).argmax(-1)
    s0 = torch.sqrt((t + 1).clamp(min=1e-12)) * 2
    q0 = torch.stack([0.25 * s0, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0], -1)
    s1 = torch.sqrt((1 + m00 - m11 - m22).clamp(min=1e-12)) * 2
    q1 = torch.stack([(m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1], -1)
    s2 = torch.sqrt((1 - m00 + m11 - m22).clamp(min=1e-12)) * 2
    q2 = torch.stack([(m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2], -1)
    s3 = torch.sqrt((1 - m00 - m11 + m22).clamp(min=1e-12)) * 2
    q3 = torch.stack([(m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3], -1)
    for k, qk in enumerate((q0, q1, q2, q3)):
        q = torch.where((big == k).unsqueeze(-1), qk, q)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def make_renderer(sc, ply, cam):
    """-> f(gaussian positions in MPM space [N,3]) -> uint8 image.

    The simulators all work in MPM space and the Gaussians were trained in world
    space, so the displacement handed to the rasteriser has to cross back:
    undo(q) - xyz_world, not q - xyz_world. Skipping the undo leaves the cloud
    offset by the whole space change and scales every displacement by the wrong
    factor -- which still renders a recognisable tree that moves, so it does not
    announce itself.
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

    def frame(xyz_mpm, F=None):
        """F is the per-Gaussian deformation gradient, or None to leave every
        splat at its rest shape -- which is what this did before, and what made
        a stretching cloud look like it was coming apart."""
        d_xyz = sc.undo(xyz_mpm) - sc.xyz_world
        if F is None:
            dr, ds = d_rot, d_sc
        else:
            dr, ds = cov_deltas(F, gaussians._rotation.detach(),
                                gaussians.get_scaling.detach())
        im = torch.clamp(_render(cam, gaussians, pipe, bg, d_xyz, dr, ds,
                                  d_rot_as_res=True)["render"], 0, 1)
        return (im.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

    # a rest state must render as no displacement at all; if the space change is
    # wrong this is the cheapest place to find out
    rest = float((sc.undo(sc.pos) - sc.xyz_world).abs().max())
    if rest > 1e-3:
        raise RuntimeError(f"undo(pos) differs from xyz_world by {rest:.3g}; the "
                            f"renderer would draw a deformed rest state")
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
