"""장면의 렌더 설정을 점검한다: 가우시안이 몇 개나 참여하는지, 카메라가 물체를
제대로 담는지. 정지 자세 한 장을 여러 시야로 뽑아 눈으로 확인한다.
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
from anchorflow.view import build_camera, camera_from_json, label, make_renderer

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--n_anchors", type=int, default=1024)
ap.add_argument("--knn", type=int, default=8)
ap.add_argument("--width", type=int, default=480)
ap.add_argument("--height", type=int, default=None)
ap.add_argument("--radius", type=float, nargs="+", default=[1.0, 1.6, 2.5, 4.0],
                help="radius_scale (--abs_radius 면 절대 거리)")
ap.add_argument("--abs_radius", action="store_true",
                help="--radius 를 배율이 아니라 절대 거리로 쓴다")
ap.add_argument("--full_scene", action="store_true",
                help="crop 밖 가우시안도 배경으로 그린다")
ap.add_argument("--fov", type=float, nargs="+", default=[0.6911])
ap.add_argument("--cameras", default=None,
                help="동봉 cameras.json. 주면 실촬영 포즈를 쓴다 -- 궤도 카메라를 "
                     "합성하지 않는다")
ap.add_argument("--cam_index", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()

from plyfile import PlyData
n_raw = len(PlyData.read(args.ply)["vertex"])

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.knn, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
crop = getattr(sc, "crop", None)
n_crop = int(crop.sum()) if crop is not None else sc.N
xw = sc.xyz_world[sc.keep]
ext_keep = float((xw.max(0).values - xw.min(0).values).norm())
xa = sc.xyz_world
ext_all = float((xa.max(0).values - xa.min(0).values).norm())
cfg = sc.cfg
print(f"[가우시안] PLY 원본 {n_raw}, crop 통과 {n_crop} ({100*n_crop/max(1,n_raw):.1f}%), "
      f"sc.N {sc.N}, 시뮬 대상 sc.keep {int(sc.keep.sum())}", flush=True)
print(f"[크기] keep 만의 대각 {ext_keep:.4f}, 전체(sc.N)의 대각 {ext_all:.4f}, "
      f"sc.extent {float(sc.extent):.4f}", flush=True)
print(f"[카메라 설정] azim {cfg.get('init_azimuthm')}, elev {cfg.get('init_elevation')}, "
      f"config radius {cfg.get('init_radius')} <-> 우리가 쓰는 1.6*ext = "
      f"{1.6*ext_keep:.3f}", flush=True)
print(f"[시점] center {cfg.get('mpm_space_viewpoint_center')}, "
      f"up {cfg.get('mpm_space_vertical_upward_axis')}", flush=True)

rows = []
if args.cameras:
    row = []
    for ci in args.cam_index:
        cam = camera_from_json(sc, args.cameras, ci, args.width, args.height)
        frame = make_renderer(sc, args.ply, cam, full_scene=args.full_scene)
        row.append(label(frame(sc.pos.clone()), f"cam {ci}"))
    img = np.concatenate(row, axis=1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.imwrite(args.out, img)
    print(f"[out] {args.out} {img.shape}", flush=True)
    print("RSETUP_DONE")
    sys.exit(0)
for fov in args.fov:
    row = []
    for r in args.radius:
        cam = (build_camera(sc, args.width, args.height or args.width, fov, radius=r)
               if args.abs_radius
               else build_camera(sc, args.width, args.height or args.width, fov, r))
        frame = make_renderer(sc, args.ply, cam, full_scene=args.full_scene)
        row.append(label(frame(sc.pos.clone()),
                         f"{'dist' if args.abs_radius else 'x'}={r:g} fov={fov:.3g}"))
    rows.append(np.concatenate(row, axis=1))
img = np.concatenate(rows, axis=0)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
imageio.imwrite(args.out, img)
print(f"[out] {args.out} {img.shape}", flush=True)
print("RSETUP_DONE")
