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


def build_camera(sc, width=640, height=640, fov_x=0.6911, radius_scale=1.6,
                  radius=None):
    """The view the config asks for: its own up axis, centre, azimuth and
    elevation.

    거리는 radius 가 주어지면 그것을, 아니면 radius_scale * (월드 대각) 을 쓴다.
    후자는 ficus 에서 우연히 config 값과 맞았을 뿐이다 -- ficus 는 1.6*2.76=4.42
    대 config 4.11 로 가깝지만, vasedeck 은 1.6*7.59=12.14 대 config 5 로 2.4 배
    멀어 물체가 점만 하게 나온다. 실촬영 장면은 월드 대각에 배경까지 들어가서
    그렇다."""
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
    dist = float(radius) if radius is not None else radius_scale * extent
    eye = center + dist * (
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


def local_deformation(rest, K=16, ridge=1e-3):
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

    The ridge is not decoration. A Gaussian on a thin feature has neighbours
    that are very nearly coplanar, B is nearly singular, and the least squares
    answer along the thin direction is whatever noise happens to be there --
    which renders as a splat stretched into a needle several times the size of
    the object. K=16 and a ridge scaled to B's own magnitude keep that bounded;
    cov_deltas caps what survives.
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


import os as _os

# how far a splat may stretch relative to its rest shape. Too small and real
# deformation is clipped into gaps; too large and a degenerate neighbourhood
# renders as a needle. Overridable so the choice can be checked rather than
# assumed.
STRETCH_CAP = float(_os.environ.get("AF_STRETCH_CAP", 3.0))


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
        # even with the ridge, a Gaussian whose neighbourhood is degenerate can
        # come back stretched by orders of magnitude. Bound each new scale to a
        # factor of the one it came from -- the singular values are sorted, so
        # the original scales have to be sorted to compare against
        s_ref = torch.sort(scale[lo:hi], dim=-1, descending=True).values
        S = S.clamp(min=s_ref / STRETCH_CAP, max=s_ref * STRETCH_CAP)
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


def camera_from_json(sc, cam_json, index=0, width=None, height=None):
    """데이터셋에 동봉된 실제 촬영 포즈로 카메라를 만든다.

    궤도 카메라를 임의의 radius/fov 로 합성하면 안 된다 -- 실촬영 장면은 COLMAP
    포즈로 학습됐고, 그 밖의 시점에서는 배경이 무너져 보인다. cameras.json 의
    항목은 C2W 회전과 월드 위치, 그리고 픽셀 초점거리를 담고 있다.

        R = rotation (C2W),  T = -R^T @ position
        FoVx = focal2fov(fx, width),  FoVy = focal2fov(fy, height)

    해상도를 줄이려면 width/height 를 주면 된다 -- 초점거리도 같은 비율로 준다.
    """
    import json

    import numpy as np
    from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2

    dev = sc.pos.device
    cams = json.load(open(cam_json)) if isinstance(cam_json, str) else cam_json
    c = cams[index % len(cams)]
    w0, h0 = int(c["width"]), int(c["height"])
    # 원본 종횡비를 지킨다. width 만 주면 height 는 거기서 유도한다 -- 둘을
    # 따로 주면 fx 와 fy 가 다른 비율로 스케일되어 그림이 늘어난다. plane 은
    # 1060x1895(세로)인데 640x480 으로 뽑아 완전히 찌그러졌다.
    if width and not height:
        w = int(width)
        h = int(round(h0 * w / w0))
    elif height and not width:
        h = int(height)
        w = int(round(w0 * h / h0))
    else:
        w, h = int(width or w0), int(height or h0)
        if abs((w / h) - (w0 / h0)) > 1e-3:
            print(f"[cam] 종횡비 불일치: 원본 {w0}x{h0} ({w0/h0:.3f}) vs "
                  f"요청 {w}x{h} ({w/h:.3f}) -- 그림이 늘어난다", flush=True)
    fx = float(c["fx"]) * (w / w0)
    fy = float(c["fy"]) * (h / h0)
    R = np.array(c["rotation"], dtype=np.float64)
    T = -R.T @ np.array(c["position"], dtype=np.float64)
    fovx, fovy = focal2fov(fx, w), focal2fov(fy, h)
    wvt = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float().to(dev)
    pmx = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx,
                               fovY=fovy).transpose(0, 1).to(dev)
    return MiniCam(w, h, fovy, fovx, 0.01, 100.0, wvt,
                    (wvt.unsqueeze(0).bmm(pmx.unsqueeze(0))).squeeze(0))


def best_camera_index(sc, cam_json, width=640, sample=4000):
    """시뮬레이션 대상이 화면을 가장 잘 채우는 시점을 고른다.

    cameras.json 의 몇 번째가 좋은 시점인지는 장면마다 다르다 -- pillow2sofa 의
    0 번은 소파 아래를 찍고 있었다. 눈대중으로 고르는 대신, 각 카메라에 대해
    (화면 안에 들어온 대상 입자의 비율) x (그 입자들이 차지하는 화면 면적) 을
    점수로 매겨 최댓값을 쓴다.
    """
    import json

    import numpy as np

    cams = json.load(open(cam_json)) if isinstance(cam_json, str) else cam_json
    x = sc.xyz_world[sc.keep]
    if x.shape[0] > sample:
        idx = torch.randperm(x.shape[0], device=x.device)[:sample]
        x = x[idx]
    xh = torch.cat([x, torch.ones_like(x[:, :1])], -1).double().cpu().numpy()
    best, best_i = -1.0, 0
    for i, c in enumerate(cams):
        w0, h0 = int(c["width"]), int(c["height"])
        R = np.array(c["rotation"], dtype=np.float64)
        T = -R.T @ np.array(c["position"], dtype=np.float64)
        # 월드 -> 카메라
        cam_xyz = xh[:, :3] @ R + T
        z = cam_xyz[:, 2]
        front = z > 1e-6
        if front.sum() < 16:
            continue
        u = c["fx"] * cam_xyz[front, 0] / z[front] + w0 / 2.0
        v = c["fy"] * cam_xyz[front, 1] / z[front] + h0 / 2.0
        inside = (u >= 0) & (u < w0) & (v >= 0) & (v < h0)
        frac = float(inside.mean()) * float(front.mean())
        if inside.sum() < 16:
            continue
        area = (float(np.ptp(u[inside])) / w0) * (float(np.ptp(v[inside])) / h0)
        score = frac * min(area, 1.0)
        if score > best:
            best, best_i = score, i
    return best_i, best


def make_renderer(sc, ply, cam, full_scene=False, differentiable=False):
    """-> f(gaussian positions in MPM space [N,3]) -> uint8 image.

    differentiable=True 면 uint8 numpy 대신 **[3,H,W] float 텐서를 [0,1] 로**
    돌려준다. 그래디언트가 xyz 를 거쳐 앵커까지 이어지므로 SDS 처럼 렌더를 통해
    역전파하는 쓰임에 필요하다. 기본값은 그대로 uint8 numpy 다 -- 영상·그림을
    뽑는 모든 호출자가 그것을 기대한다.

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
    # scene_setup drops everything outside sim_area; the PLY on disk still has
    # all of it, and the rasteriser would be handed two million splats for a
    # displacement of eighty thousand
    # full_scene 이면 PLY 전체를 그린다. 시뮬레이션 밖 가우시안은 배경으로
    # 제자리에 서 있고 변위 0 을 받는다 -- vasedeck 은 crop 이 5.8% 만 남겨서,
    # 자르고 그리면 장면의 94% 가 사라진 그림이 나온다.
    crop = getattr(sc, "crop", None)
    n_full = gaussians._xyz.shape[0]
    if crop is not None and not full_scene:
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity",
                     "_scaling", "_rotation"):
            setattr(gaussians, name, getattr(gaussians, name)[crop])

    class _P:
        debug = False
        compute_cov3D_python = False
        convert_SHs_python = False

    pipe = _P()
    bg = torch.tensor([1., 1., 1.], device=dev)
    n_draw = gaussians._xyz.shape[0]
    d_rot = torch.zeros(n_draw, 4, device=dev); d_rot[:, 0] = 1.
    d_sc = torch.zeros(n_draw, 3, device=dev)
    wide = full_scene and crop is not None and n_full != sc.N

    def frame(xyz_mpm, F=None, hide=None):
        """F is the per-Gaussian deformation gradient, or None to leave every
        splat at its rest shape -- which is what this did before, and what made
        a stretching cloud look like it was coming apart."""
        d_xyz = sc.undo(xyz_mpm) - sc.xyz_world
        if wide:
            full = torch.zeros(n_full, 3, device=dev, dtype=d_xyz.dtype)
            full[crop] = d_xyz
            d_xyz = full
        d_op = None
        if hide is not None:
            # the Gaussians MPM never simulated are carried by their neighbours,
            # which is an interpolation and shows as tearing where the real
            # displacement is largest. Turning them off says how much of what is
            # on screen is the simulation and how much is the carry.
            d_op = torch.zeros_like(gaussians.get_opacity)
            idx = torch.nonzero(crop, as_tuple=False).squeeze(-1)[hide] if wide else hide
            d_op[idx] = -gaussians.get_opacity[idx]
        if F is None:
            dr, ds = d_rot, d_sc
        else:
            if wide:
                dr, ds = d_rot.clone(), d_sc.clone()
                _r, _s = cov_deltas(F, gaussians._rotation.detach()[crop],
                                    gaussians.get_scaling.detach()[crop])
                dr[crop], ds[crop] = _r, _s
            else:
                dr, ds = cov_deltas(F, gaussians._rotation.detach(),
                                    gaussians.get_scaling.detach())
        im = torch.clamp(_render(cam, gaussians, pipe, bg, d_xyz, dr, ds,
                                  d_opacity=d_op, d_rot_as_res=True)["render"], 0, 1)
        if differentiable:
            return im
        return (im.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")

    # a rest state must render as no displacement at all; if the space change is
    # wrong this is the cheapest place to find out
    with torch.no_grad():
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
