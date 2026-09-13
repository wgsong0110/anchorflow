"""기하마다 앵커 자체를 3DGS 로 그린다.

앵커는 이미 타원체다 -- pos 가 중심, quat 이 방향, exp(log_s) 가 반지름. 그래서
장면 가우시안을 그리던 래스터라이저에 앵커를 그대로 얹으면 "이 기하가 물체를
어떻게 덮고 있는가"가 그림 하나로 나온다. 맨 왼쪽이 장면, 그 뒤가 기하별 앵커다.

앵커 색은 쥔 가우시안 수로 칠한다 -- 고아 앵커(0 개)를 눈으로 찾기 위해서다.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import imageio
import numpy as np
import torch

from anchorflow import scene_setup
from anchorflow.view import build_camera, label

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--fit", action="append", required=True, help="이름=경로")
ap.add_argument("--out", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--width", type=int, default=560)
ap.add_argument("--height", type=int, default=560)
ap.add_argument("--fov_x", type=float, default=0.6911)
ap.add_argument("--radius_scale", type=float, default=1.6)
ap.add_argument("--opacity", type=float, default=0.9)
ap.add_argument("--shrink", type=float, default=1.0,
                help="앵커 반지름 배율. 1.0 이면 커널의 실제 크기 그대로")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.anchor_sparse import load_fitted

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                       frozen_weights=True, rot_fallback=True,
                       eig_floor=args.eig_floor)
cam = build_camera(sc, args.width, args.height, args.fov_x, args.radius_scale)

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as _render

C0 = 0.28209479177387814          # SH 0차 상수


class _P:
    debug = False
    compute_cov3D_python = False
    convert_SHs_python = False


pipe = _P()
BG = torch.tensor([1., 1., 1.], device=dev)


def shoot(g, n):
    z = torch.zeros(n, 3, device=dev)
    zr = torch.zeros(n, 4, device=dev)
    im = torch.clamp(_render(cam, g, pipe, BG, z, zr, z,
                             d_rot_as_res=True)["render"], 0, 1)
    return (im.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")


def draw(xyz_world, quat, scale, rgb):
    """앵커 한 묶음을 가우시안으로 그린다. scale 은 실제 반지름."""
    n = xyz_world.shape[0]
    g = GaussianModel(0, fea_dim=0)
    g._xyz = xyz_world.contiguous()
    g._scaling = torch.log(scale.clamp(min=1e-8)).contiguous()
    g._rotation = quat.contiguous()
    op = torch.full((n, 1), args.opacity, device=dev)
    g._opacity = torch.log(op / (1 - op))                  # 역시그모이드
    f = torch.zeros(n, 1, 3, device=dev)
    f[:, 0, :] = (rgb - 0.5) / C0
    g._features_dc, g._features_rest = f, torch.zeros(n, 0, 3, device=dev)
    g.active_sh_degree = 0
    g.max_sh_degree = 0
    return shoot(g, n)


# 맨 왼쪽: 장면 가우시안 그대로 (모든 기하가 같으므로 한 번만)
gs = GaussianModel(3, fea_dim=0)
gs.load_ply(args.ply)
if getattr(sc, "crop", None) is not None:
    m = sc.crop
    for nm in ("_xyz", "_features_dc", "_features_rest", "_opacity",
               "_scaling", "_rotation"):
        setattr(gs, nm, getattr(gs, nm)[m])
panels = [label(shoot(gs, gs._xyz.shape[0]), "scene gaussians")]
print(f"[render] 장면 {gs._xyz.shape[0]} 가우시안", flush=True)

for spec in args.fit:
    name, path = spec.split("=", 1)
    if not os.path.exists(path):
        print(f"  [건너뜀] {name}: {path} 없음", flush=True)
        continue
    fit = load_fitted(sc, path, dev)[0].fit
    held = torch.zeros(fit.M, device=dev).index_add_(
        0, fit.pair_a, torch.ones(fit.pair_a.shape[0], device=dev))
    # 색: 쥔 가우시안이 적을수록 붉게, 많을수록 푸르게. 0 개면 순빨강.
    t = (held / held.clamp(min=1).median()).clamp(0, 2) / 2.0
    rgb = torch.stack([1.0 - t, t * 0.5, t], -1)
    q = fit.quat / fit.quat.norm(dim=-1, keepdim=True)
    img = draw(sc.undo(fit.pos), q, fit.log_s.exp() * args.shrink, rgb)
    orph = int((held == 0).sum())
    panels.append(label(img, f"{name}  M={fit.M}  orphan={orph}"))
    print(f"  {name}: 앵커 {fit.M}, 고아 {orph}, 쥔 짝 중앙값 {held.median():.0f}",
          flush=True)
    fit = None
    torch.cuda.empty_cache()

out = np.concatenate(panels, axis=1)
imageio.imwrite(args.out, out)
print(f"[out] {args.out}  {out.shape}", flush=True)
print("RENDER_DONE")
