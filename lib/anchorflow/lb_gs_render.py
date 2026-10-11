"""lb 데이터셋(lego) 3DGS 렌더러를 **메모리에서 바로** 쓰는 모듈 (exe/lb_render.py 와 같은 렌더러·카메라 규약).

롤아웃이 낸 위치·F 를 디스크(h5) 를 거치지 않고 프레임마다 렌더해 영상으로 쓴다.
가우시안 ↔ 입자: 앞 NG 개가 불투명도 필터를 거친 가우시안과 같은 번호. 공분산 = F C0 Fᵀ (F 에 초기 회전 R 이 실려 있어야 한다).

  R = LegoRenderer(cfg)            # 한 번
  W = R.writer("out.mp4")
  W.frame(x[:NG], F_render[:NG])   # 프레임마다 (torch, cuda)
  W.close()
"""
import json
import math
import os
import sys

import numpy as np
import torch

W_ROOT = "/home/dkta/work"
MODEL = {"lego": "lego_whitebg-trained", "mic": "mic_whitebg-trained", "ficus": "ficus_whitebg-trained"}


class LegoRenderer:
    def __init__(self, cfg, shape="lego", dev="cuda"):
        sys.path.append(f"{W_ROOT}/i-physgaussian"); sys.path.append(f"{W_ROOT}/i-physgaussian/gaussian-splatting")
        _cwd = os.getcwd(); os.chdir(f"{W_ROOT}/i-physgaussian")
        from scene.gaussian_model import GaussianModel                       # noqa: E402
        from utils.transformation_utils import transform2origin, shift2center111   # noqa: E402
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix      # noqa: E402
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer   # noqa: E402
        os.chdir(_cwd)
        self.dev = dev
        mp = f"{W_ROOT}/pgmodel/{MODEL[shape]}"
        gs = GaussianModel(3)
        gs.load_ply(f"{mp}/point_cloud/iteration_30000/point_cloud.ply")
        with torch.no_grad():
            KIDX = torch.nonzero(gs.get_opacity[:, 0] > float(cfg.get("opacity_threshold", 0.02))).squeeze(1)
            TP, self.SO, self.MEAN = transform2origin(gs.get_xyz[KIDX], float(cfg.get("scale", 1.0)))
            self.NG = KIDX.numel()
            c6 = gs.get_covariance()[KIDX]
            C0 = torch.zeros(self.NG, 3, 3, device=dev)
            C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c6[:, 0], c6[:, 1], c6[:, 2]
            C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c6[:, 3], c6[:, 4], c6[:, 5]
            C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c6[:, 1], c6[:, 2], c6[:, 4]
            self.C0 = C0
            self.SHS, self.OPA = gs.get_features[KIDX].detach(), gs.get_opacity[KIDX].detach()
        cam = json.load(open(f"{mp}/cameras.json"))[0]
        Rw, pos = np.array(cam["rotation"]), np.array(cam["position"])
        _ctr = self.MEAN.detach().cpu().numpy().astype(np.float64)
        pos = _ctr + 1.6 * (pos - _ctr)                                  # lb_render.py 와 같은 시점
        W2C = np.linalg.inv(np.block([[Rw, pos[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
        fx_ = 2 * math.atan(cam["width"] / (2 * cam["fx"])); fy_ = 2 * math.atan(cam["height"] / (2 * cam["fy"]))
        wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
        pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fx_, fovY=fy_).transpose(0, 1).to(dev).float()
        self.RAST = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
            image_height=int(cam["height"]), image_width=int(cam["width"]), tanfovx=math.tan(fx_ * 0.5),
            tanfovy=math.tan(fy_ * 0.5), bg=torch.ones(3, device=dev), scale_modifier=1.0, viewmatrix=wv,
            projmatrix=(wv[None] @ pj[None])[0], sh_degree=3, campos=wv.inverse()[3, :3], prefiltered=False, debug=False))

    @torch.no_grad()
    def image(self, x, F):
        """x [NG,3] (시뮬 좌표), F [NG,3,3] (초기 회전 실린 렌더용) -> uint8 [H,W,3] numpy."""
        x = x[:self.NG].to(self.dev).float(); F = F[:self.NG].to(self.dev).float()
        cov = F @ self.C0 @ F.transpose(1, 2)
        cc = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
        Pm = (x - 1.0) / self.SO + self.MEAN
        img = self.RAST(means3D=Pm, means2D=torch.zeros_like(Pm), shs=self.SHS, colors_precomp=None, opacities=self.OPA,
                        scales=None, rotations=None, cov3D_precomp=cc)[0]
        return (img.clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()

    def writer(self, path, fps=30):
        import imageio.v2 as imageio
        r = self

        class _W:
            def __init__(self):
                self.w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
                self.n = 0

            def frame(self, x, F):
                self.w.append_data(r.image(x, F)); self.n += 1

            def close(self):
                self.w.close()
        return _W()
