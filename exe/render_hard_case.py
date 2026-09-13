"""가장 어려운 임펄스에서 학생들이 무엇을 하는지 본다.

(K, r) 격자에서 최악은 K=1, r=0.125x 입자 간격이다 -- 포크 하나가 입자 몇 개에만
걸리는 경우로, base 94.6%, A2 145.3% 였다. 표현 하한은 거기서도 1% 대라 실패는
전적으로 동역학이다. 숫자로는 "발산"이라고만 알 수 있어서 실제로 무엇이 일어나는지
그림으로 본다.

MPM 이 시뮬레이션하지 않는 가우시안(sc.keep 밖)은 가장 가까운 입자들의 변위로
실어 나른다 -- 제자리에 두면 나무가 움직이는 동안 유령처럼 서 있다.
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
from tqdm import tqdm

from anchorflow import scene_setup
from anchorflow.nextstate import apply_step, net_from_ckpt
from anchorflow.view import (best_camera_index, build_camera, camera_from_json,
                             label, make_renderer)

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--student", action="append", default=[], help="이름:기하.pt:학생.pt")
ap.add_argument("--K", type=int, default=1, help="포크 개수")
ap.add_argument("--r", type=float, default=0.125, help="반경, 입자 간격의 배수")
ap.add_argument("--mag", type=float, default=1.0, help="기준 힘의 배수")
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--knn", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--width", type=int, default=560)
ap.add_argument("--height", type=int, default=None,
                help="생략하면 --cameras 의 원본 종횡비에서 유도한다. 궤도 카메라 "
                     "경로에서는 width 와 같게 둔다")
ap.add_argument("--fov_x", type=float, default=0.6911)
ap.add_argument("--radius_scale", type=float, default=1.6)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--base_force", type=float, nargs=3, default=None,
                help="config 에 particle_impulse 가 없는 장면용 기준 임펄스")
ap.add_argument("--cameras", default=None,
                help="동봉 cameras.json. 실촬영 장면은 반드시 이걸 쓴다 -- 궤도 "
                     "카메라를 합성하면 학습된 시점 밖이라 그림이 무너진다")
ap.add_argument("--cam_index", type=int, default=-1,
                help="-1 이면 시뮬 대상이 화면을 가장 잘 채우는 시점을 자동 선택")
ap.add_argument("--full_scene", action="store_true",
                help="crop 밖 가우시안도 배경으로 그린다. vasedeck 은 crop 이 5.8% 뿐")
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.knn, device=dev,
                       frozen_weights=True, rot_fallback=True,
                       eig_floor=args.eig_floor)
T = MPMTeacher(sc)
mat = T.mat
dt_c = args.dt_mult * sc.sub_dt
BASE = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        BASE = torch.tensor(bc["force"], device=dev)
if args.base_force is not None:
    BASE = torch.tensor(args.base_force, device=dev)
if BASE is None:
    raise SystemExit("이 장면에는 particle_impulse 가 없다 -- --base_force 를 줄 것")

g = torch.Generator(device=dev); g.manual_seed(args.seed)
force = sc.random_multi_poke(g, args.K, args.r * sc.sim.radius,
                             BASE.norm().item() * args.mag)
hot = int((force.norm(dim=-1) > 0.05 * force.norm(dim=-1).max()).sum())
print(f"[case] K={args.K} r={args.r}x  힘을 받는 가우시안 {hot} / {sc.N} "
      f"({100*hot/sc.N:.2f}%)", flush=True)

# --- MPM 기준 ---
dv = sc.impulse_dv(force)
v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
truth = [T.pos_m.clone()]
for _ in tqdm(range(args.frames), desc="  MPM", ncols=88):
    for k in range(args.dt_mult):
        T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
    truth.append(T.solver.export_particle_x_to_torch().clone())
truth = torch.stack(truth)
# 오차는 물체 크기라는 고정 상수로 나눈다. 그 임펄스의 변위로 나누면 거의 안
# 움직이는 경우에 분모가 0 에 가까워져 잔떨림이 100% 넘게 찍힌다 -- 실제로
# (K=1, r=0.125x) 에서 MPM 이 물체 크기의 0.26% 만 움직였는데 173% 로 나왔다.
EXTENT = float(sc.extent)
ref_span = (truth - truth[0]).norm(dim=-1).max().clamp(min=1e-12)
span = EXTENT
print(f"[case] MPM 최대 변위 {float(ref_span):.4f} "
      f"(물체 크기의 {100*float(ref_span)/EXTENT:.2f}%), 정규화 상수 {EXTENT:.4f}",
      flush=True)

# --- MPM 이 안 건드리는 가우시안을 실어 나르기 ---
from scipy.spatial import cKDTree
restg = torch.nonzero(~sc.keep, as_tuple=False).squeeze(-1)
nd_np, ni_np = cKDTree(T.pos_m.cpu().numpy()).query(sc.pos[restg].cpu().numpy(), k=8)
nd = torch.from_numpy(nd_np).float().to(dev)
ni = torch.from_numpy(ni_np).long().to(dev)
cw = 1.0 / nd.clamp(min=1e-6); cw = cw / cw.sum(-1, keepdim=True)

runs, errs = {}, {}
mpm_full = []
for k in range(args.frames + 1):
    q = sc.pos.clone()
    q[mat] = truth[k]
    q[restg] = sc.pos[restg] + (cw.unsqueeze(-1) * (truth[k] - T.pos_m)[ni]).sum(1)
    mpm_full.append(q)
runs["MPM"] = torch.stack(mpm_full)

for spec in args.student:
    name, fitp, ckp = spec.split(":", 2)
    fs = load_fitted(sc, fitp, dev)[0]
    net = net_from_ckpt(torch.load(ckp, map_location=dev, weights_only=False), dev)
    p, v = fs.anchor_canonical.clone(), fs.initial_velocity(force)
    gp = fs.pos.clone()
    out, k = [gp.clone()], 0
    while k < args.frames:
        for q, d in apply_step(net, p, v, None, dt_c, fs.fixed_mask):
            p, v = q, d / dt_c
            gp = fs.skin(p, gp)
            out.append(gp.clone())
            k += 1
            if k >= args.frames:
                break
    runs[name] = torch.stack(out[: args.frames + 1])
    errs[name] = (runs[name][:, mat] - truth).norm(dim=-1).mean(-1) / span
    amp = float((runs[name][:, mat] - runs[name][0, mat]).norm(dim=-1).max() / ref_span)
    print(f"  {name}: 전 프레임 평균 {100*float(errs[name].mean()):.2f}%, "
          f"마지막 {100*float(errs[name][-1]):.2f}%, 진폭 {100*amp:.0f}%", flush=True)
    fs = net = None
    torch.cuda.empty_cache()

CI = args.cam_index
if args.cameras and CI < 0:
    CI, _sc = best_camera_index(sc, args.cameras, args.width)
    print(f"[cam] 자동 선택: {CI} (점수 {_sc:.4f})", flush=True)
cam = (camera_from_json(sc, args.cameras, CI, args.width, args.height)
       if args.cameras
       else build_camera(sc, args.width, args.height or args.width,
                         args.fov_x, args.radius_scale))
frame = make_renderer(sc, args.ply, cam, full_scene=args.full_scene)
NAMES = ["MPM"] + [s.split(":", 1)[0] for s in args.student]
vid = []
for k in tqdm(range(args.frames + 1), desc="  render", ncols=88):
    row = []
    for n in NAMES:
        tag = n if n == "MPM" else f"{n}   {100*float(errs[n][k]):.1f}%"
        row.append(label(frame(runs[n][k]), tag))
    vid.append(np.concatenate(row, axis=1))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
imageio.mimwrite(args.out, vid, fps=args.fps, codec="libx264",
                 output_params=["-pix_fmt", "yuv420p"])
print(f"[out] {args.out}  {len(vid)} 프레임", flush=True)
print("HARD_DONE")
