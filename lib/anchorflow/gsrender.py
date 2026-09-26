"""3DGS 스플랫 렌더 (PhysGaussian/GaussianFluent 와 같은 전처리·카메라·래스터라이저).

입자 위치와 변형구배만 주면 그 상태의 가우시안을 그린다. 쓰는 쪽은 교사 MPM 이든
학생이든 상관없다.

가우시안과 입자의 대응: 두 러너 모두 전처리한 가우시안 중심을 앞에 두고 채운 내부
입자를 뒤에 붙인다. h5 는 처음 순서로 저장되므로 `x[:gs_num]` 이 곧 가우시안이다.

  import taichi as ti; ti.init(arch=ti.cuda)      # 채우기 색칠이 taichi 커널이다
  sc = GSScene(model_path, config, x0_sim)
  img = sc.render(pos_sim, F, frame)              # HxWx3 uint8
"""
from __future__ import annotations

import os

import numpy as np
import torch

from utils.camera_view_utils import get_camera_view
from utils.decode_param import decode_param_json
from utils.render_utils import (convert_SH, initialize_resterize,
                                load_params_from_gs)
from utils.system_utils import searchForMaxIteration
from utils.transformation_utils import (apply_cov_rotations,
                                        apply_inverse_cov_rotations,
                                        apply_inverse_rotations, apply_rotations,
                                        generate_rotation_matrices,
                                        get_center_view_worldspace_and_observant_coordinate,
                                        shift2center111, transform2origin,
                                        undoshift2center111, undotransform2origin)
from particle_filling.filling import init_filled_particles
from scene.gaussian_model import GaussianModel


class _Pipe:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


def _upper2mat(u):
    return torch.stack([u[:, 0], u[:, 1], u[:, 2],
                        u[:, 1], u[:, 3], u[:, 4],
                        u[:, 2], u[:, 4], u[:, 5]], -1).reshape(-1, 3, 3)


def _mat2upper(m):
    return torch.stack([m[:, 0, 0], m[:, 0, 1], m[:, 0, 2],
                        m[:, 1, 1], m[:, 1, 2], m[:, 2, 2]], -1)


def _inv3(M):
    """3x3 역행렬을 딸림행렬로 직접. 배치 LU/SVD 는 여기서 극단적으로 느리다."""
    a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    d, e, f = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
    g, h, i = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]
    A = torch.stack([e * i - f * h, c * h - b * i, b * f - c * e,
                     f * g - d * i, a * i - c * g, c * d - a * f,
                     d * h - e * g, b * g - a * h, a * e - b * d], -1)
    det = a * (e * i - f * h) + b * (f * g - d * i) + c * (d * h - e * g)
    det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
    return A.reshape(-1, 3, 3) / det.reshape(-1, 1, 1)


def polar_R(F, iters=0):
    """F 에서 회전 성분만 뽑는다 -- 열벡터 그람-슈미트.

    뉴턴 극분해(역행렬 반복)는 F 가 20 배까지 늘어난 프레임에서 CUDA 를 죽였다
    (겪었다). 그람-슈미트는 원소 연산뿐이라 안전하고, SH 회전 용도로는 충분하다.
    """
    F = torch.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    a, b = F[:, :, 0], F[:, :, 1]
    na = a.norm(dim=-1, keepdim=True)
    bad = (na.squeeze(-1) < 1e-8)
    e1 = a / na.clamp(min=1e-8)
    b2 = b - (b * e1).sum(-1, keepdim=True) * e1
    nb = b2.norm(dim=-1, keepdim=True)
    bad = bad | (nb.squeeze(-1) < 1e-8)
    e2 = b2 / nb.clamp(min=1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)
    if bool(bad.any()):
        R = R.clone()
        R[bad] = torch.eye(3, device=F.device, dtype=F.dtype)
    return R


def sane_F(F):
    """퇴화한 F 는 단위행렬로. GF 는 0 프레임 f_tensor 를 전부 0 으로 쓴다."""
    if F is None:
        return None
    det = torch.linalg.det(F)
    bad = ~torch.isfinite(F).all(-1).all(-1) | (det.abs() < 1e-8)
    if bool(bad.any()):
        F = F.clone()
        F[bad] = torch.eye(3, device=F.device, dtype=F.dtype)
    return F


class GSScene:
    def __init__(self, model_path, config, x0_sim, device="cuda:0",
                 white_bg=True, sh_degree=3):
        self.dev = device
        self.model_path = model_path
        (self.material, self.bc, self.time, pre, self.cam) = \
            decode_param_json(config)
        g = GaussianModel(sh_degree)
        it = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        g.load_ply(os.path.join(model_path, "point_cloud", f"iteration_{it}",
                                "point_cloud.ply"))
        self.g = g
        self.pipe = _Pipe()
        self.bg = torch.tensor([1, 1, 1] if white_bg else [0, 0, 0],
                               dtype=torch.float32, device=device)
        p = load_params_from_gs(g, self.pipe)
        pos, cov, opa, shs = (p["pos"], p["cov3D_precomp"], p["opacity"],
                              p["shs"])
        m = opa[:, 0] > pre["opacity_threshold"]
        pos, cov, opa, shs = pos[m], cov[m], opa[m], shs[m]
        for nm in ("_xyz", "_features_dc", "_features_rest", "_opacity",
                   "_scaling", "_rotation"):
            setattr(g, nm, getattr(g, nm)[m])
        self.rots = generate_rotation_matrices(
            torch.tensor(pre["rotation_degree"]), pre["rotation_axis"])
        rotated = apply_rotations(pos, self.rots)
        # sim_area 가 있으면 그 상자 안만 시뮬한다. 밖의 가우시안은 원래 자리에
        # 그대로 그려야 한다 (움직이지 않는 배경이다) -- 빼먹으면 입자 수가
        # 가우시안 수와 어긋나 렌더가 통째로 깨진다.
        self.un = None
        area = pre.get("sim_area", None)
        if area is not None:
            m2 = torch.ones(rotated.shape[0], dtype=torch.bool, device=rotated.device)
            for i in range(3):
                m2 &= (rotated[:, i] > area[2 * i]) & (rotated[:, i] < area[2 * i + 1])
            self.un = dict(pos=pos[~m2], cov=cov[~m2], opacity=opa[~m2],
                           shs=shs[~m2])
            rotated, cov, opa, shs = rotated[m2], cov[m2], opa[m2], shs[m2]
            print(f"[sim_area] 시뮬 {int(m2.sum())} / 정지 {int((~m2).sum())} 가우시안",
                  flush=True)
        # PhysGaussian 은 config 의 `scale` 을 transform2origin 에 넘긴다
        # (최대 변을 scale 로 맞춘다). 빼먹으면 가우시안이 입자보다 크게 나온다.
        transformed, self.scale_origin, self.mean_pos = transform2origin(
            rotated, float(pre.get("scale", 1.0)))
        transformed = shift2center111(transformed)
        cov = apply_cov_rotations(cov, self.rots)
        cov = self.scale_origin * self.scale_origin * cov
        self.gs_num = transformed.shape[0]
        x0 = x0_sim.to(device).float()
        self.fit = float((x0[:self.gs_num] - transformed).abs().max())
        fill = pre["particle_filling"]
        if os.environ.get("AF_GS_ONLY") == "1":
            self.n = self.gs_num          # 채운 내부 입자는 그리지 않는다
        elif fill is not None and fill.get("visualize", False):
            shs, opa, cov = init_filled_particles(
                x0[:self.gs_num], shs, cov, opa, x0[self.gs_num:])
            self.n = x0.shape[0]
        else:
            self.n = self.gs_num
        self.cov0 = cov[:self.n].to(device)
        self.shs = shs[:self.n].to(device)
        self.opacity = opa[:self.n].to(device)
        self.screen = torch.zeros((self.n, 3), dtype=torch.float32, device=device)
        self.mark_mask = None
        # 공변 상한: 물체 지름의 몇 분의 일을 넘는 커널은 버린다
        self.cov_cap = float(((x0.max(0).values - x0.min(0).values).norm()
                              * float(os.environ.get("AF_COV_CAP", "0.10"))) ** 2)
        # 화면 반경 상한(픽셀). 이보다 커진 커널은 그리지 않는다
        self.rad_cap = float(os.environ.get("AF_RAD_CAP", "120"))
        self.mark_color = torch.tensor([0.95, 0.12, 0.10], device=device)
        vc = torch.tensor(self.cam["mpm_space_viewpoint_center"]).reshape(1, 3).cuda()
        up = torch.tensor(self.cam["mpm_space_vertical_upward_axis"]).reshape(1, 3).cuda()
        self.view_center, self.observant = \
            get_center_view_worldspace_and_observant_coordinate(
                vc, up, self.rots, self.scale_origin, self.mean_pos)

    def project(self, pts_sim, frame=0):
        """시뮬 좌표 점들을 화면 픽셀 좌표와 깊이로. (플레이트를 2D 로 그릴 때)"""
        cam = self._cam(frame)
        w = self.to_world(pts_sim.to(self.dev).float())
        h = torch.cat([w, torch.ones_like(w[:, :1])], -1) @ cam.full_proj_transform
        d = torch.cat([w, torch.ones_like(w[:, :1])], -1) @ cam.world_view_transform
        ndc = h[:, :3] / h[:, 3:4].clamp(min=1e-6)
        W, H = int(cam.image_width), int(cam.image_height)
        px = (ndc[:, 0] * 0.5 + 0.5) * W
        py = (ndc[:, 1] * 0.5 + 0.5) * H
        return torch.stack([px, py], -1).cpu().numpy(), d[:, 2].cpu().numpy()

    def mark(self, idx):
        """제어점이 잡은 가우시안을 빨갛게. idx 는 전체 번호."""
        if idx is None:
            self.mark_mask = None
            return 0
        idx = np.asarray(idx)
        idx = idx[idx < self.n]
        m = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        m[torch.from_numpy(idx.astype(np.int64)).to(self.dev)] = True
        self.mark_mask = m
        return int(m.sum())

    def to_world(self, pos_sim):
        return apply_inverse_rotations(
            undotransform2origin(undoshift2center111(pos_sim),
                                 self.scale_origin, self.mean_pos), self.rots)

    def _cam(self, frame):
        c = self.cam
        return get_camera_view(
            self.model_path, default_camera_index=c["default_camera_index"],
            center_view_world_space=self.view_center,
            observant_coordinates=self.observant, show_hint=c["show_hint"],
            init_azimuthm=c["init_azimuthm"], init_elevation=c["init_elevation"],
            init_radius=c["init_radius"], move_camera=c["move_camera"],
            current_frame=frame, delta_a=c["delta_a"], delta_e=c["delta_e"],
            delta_r=c["delta_r"])

    def set_view(self, azim=None, elev=None, radius=None):
        """카메라를 갈아끼운다. **그쪽 카메라 규약을 그대로** 쓰기 위해 config 의
        방위·고도·거리만 바꾼다 (자체 c2w 를 만들면 규약이 어긋난다)."""
        if azim is not None:
            self.cam["init_azimuthm"] = float(azim)
        if elev is not None:
            self.cam["init_elevation"] = float(elev)
        if radius is not None:
            self.cam["init_radius"] = float(radius)

    def blender_c2w(self, frame=0):
        """지금 카메라의 c2w 를 **Blender/NeRF 규약**으로 낸다.

        GausSim 의 `transforms_*.json` 이 그 규약을 읽어 c2w[:,1:3] 을 뒤집고
        역행렬을 취해 R, T 를 만든다 -- 여기서 그 과정을 정확히 거꾸로 간다.
        """
        import numpy as _np
        c = self._cam(frame)
        w2c = _np.eye(4)
        w2c[:3, :3] = _np.asarray(c.R).T
        w2c[:3, 3] = _np.asarray(c.T)
        c2w = _np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1
        return c2w, float(c.FoVx)

    def render(self, pos_sim, F=None, frame=0, hide_stretch=None, props=None):
        """이 상태의 가우시안을 그린다.

        **거르기를 SH 계산보다 먼저** 해야 한다. 폭주한 커널이나 카메라 뒤 커널을
        달고 `convert_SH` 를 부르면 그 뒤 래스터라이저가 죽는다 (겪었다).
        """
        cam = self._cam(frame)
        rast = initialize_resterize(cam, self.g, self.pipe, self.bg)
        F = sane_F(F[:self.n]) if F is not None else None
        pos = self.to_world(pos_sim[:self.n].to(self.dev).float())
        if F is None:
            cov = self.cov0
            R = torch.eye(3, device=self.dev).repeat(self.n, 1, 1)
        else:
            S = _upper2mat(self.cov0)
            cov = _mat2upper(torch.bmm(torch.bmm(F, S), F.transpose(1, 2)))
            R = polar_R(F)
        cov = apply_inverse_cov_rotations(
            cov / (self.scale_origin * self.scale_origin), self.rots)
        shs, opacity = self.shs, self.opacity
        if hide_stretch is not None and F is not None:
            far = (F.reshape(-1, 9).norm(dim=-1) > float(hide_stretch))
            opacity = opacity.clone()
            opacity[far] = 0.0
        mark = self.mark_mask
        # 걸러야 할 커널 둘: 화면 반경이 폭주한 것, 카메라 뒤/코앞에 있는 것
        tr = cov[:, 0] + cov[:, 3] + cov[:, 5]
        dep = (torch.cat([pos, torch.ones_like(pos[:, :1])], -1)
               @ cam.world_view_transform)[:, 2]
        # 고정 문턱 대신 **화면 반경**으로 거른다 -- 래스터라이저가 죽는 진짜
        # 조건은 커널이 화면에서 너무 커지는 것이다 (프레임마다 기준이 달라진다).
        import math as _m
        focal = 0.5 * float(cam.image_width) / _m.tan(float(cam.FoVx) * 0.5)
        rad_px = focal * tr.clamp(min=0.0).sqrt() / dep.clamp(min=0.2)
        keep = ((rad_px < self.rad_cap) & (dep > 0.2)
                & torch.isfinite(cov).all(-1) & torch.isfinite(pos).all(-1))
        if not bool(keep.all()):
            pos, cov, R = pos[keep], cov[keep], R[keep]
            shs, opacity = shs[keep], opacity[keep]
            if mark is not None:
                mark = mark[keep]
        if self.un is not None:                      # 시뮬 영역 밖은 제자리에
            n_sim = pos.shape[0]
            pos = torch.cat([pos, self.un["pos"]])
            cov = torch.cat([cov, self.un["cov"]])
            shs = torch.cat([shs, self.un["shs"]])
            opacity = torch.cat([opacity, self.un["opacity"]])
        # SH 회전을 넘기면(특히 expand 로 만든 단위행렬) cuBLAS 경로에서 카드가
        # 죽는 경우가 있다. AF_SH_ROT=0 이면 회전 없이 색만 계산한다.
        _rot = R if os.environ.get("AF_SH_ROT", "1") != "0" else None
        colors = convert_SH(shs, cam, self.g, pos, _rot)
        if mark is not None:
            colors = colors.clone()
            idx = torch.nonzero(mark).squeeze(-1)
            colors[idx] = self.mark_color.to(colors.dtype)
        screen = torch.zeros((pos.shape[0], 3), dtype=torch.float32,
                             device=self.dev)
        out = rast(means3D=pos.contiguous(), means2D=screen, shs=None,
                   colors_precomp=colors.float().contiguous(),
                   opacities=opacity.contiguous(), scales=None, rotations=None,
                   cov3D_precomp=cov.contiguous())
        img = out[0] if isinstance(out, (tuple, list)) else out
        img = torch.nan_to_num(img, nan=float(self.bg[0]))
        return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
