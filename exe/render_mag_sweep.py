"""임펄스 모양은 고정하고 세기만 바꿔 MPM 이 어떻게 반응하는지 본다.

wolf 는 소성(sand)이라 세기를 10 만 배 키워도 변위가 물체의 0.87% 에서 17.7% 까지만
갔고, 0.075~5.6 구간에서는 아예 고정이었다. 항복 뒤로는 힘이 흐름으로 빠져나가
변형이 누적되지 않는다는 뜻인데, 숫자만으로는 그것이 "안 움직인다" 인지
"흐르는데 최대 변위가 안 커진다" 인지 알 수 없다. 그림으로 가른다.

모양은 같은 씨앗의 같은 (K, r) 이고 세기만 곱해진다.
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
from anchorflow.view import build_camera, label, make_renderer

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--mags", type=float, nargs="+", required=True)
ap.add_argument("--K", type=int, default=1)
ap.add_argument("--r", type=float, default=10.0, help="반경, 입자 간격의 배수")
ap.add_argument("--uniform", action="store_true",
                help="다중 포크 대신 전 입자에 같은 방향 힘. 보정 프로브와 같은 형태")
ap.add_argument("--dir", type=float, nargs=3, default=[1.0, 0.0, 0.0])
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--check_every", type=int, default=2,
                help="도메인 검사 주기(substep). 기본 8 은 큰 임펄스에서 늦어 "
                     "warp 가 격자 밖에 쓰고 프로세스가 죽는다")
ap.add_argument("--n_anchors", type=int, default=1024)
ap.add_argument("--knn", type=int, default=8)
ap.add_argument("--width", type=int, default=480)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--fov_x", type=float, default=0.6911)
ap.add_argument("--radius_scale", type=float, default=1.6)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.mpm_teacher import MPMTeacher

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.knn, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
T = MPMTeacher(sc)
mat = T.mat
EXT = float(sc.extent)
print(f"[setup] 가우시안 {sc.N} (시뮬 {T.n}), 물체 크기 {EXT:.4f}, "
      f"고정 {int(sc.fixed_mask.sum())}", flush=True)

D = torch.tensor(args.dir, device=dev, dtype=torch.float32)
D = D / D.norm().clamp(min=1e-12)
if args.uniform:
    SHAPE = D.unsqueeze(0).expand(sc.pos.shape[0], 3).contiguous()
else:
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    SHAPE = sc.random_multi_poke(g, args.K, args.r * sc.sim.radius, 1.0)


def run(mag):
    f = SHAPE * mag
    dv = sc.impulse_dv(f)
    v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
    T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
    xs = [T.pos_m.clone()]
    for _ in range(args.frames):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % args.check_every == 0 and not T._in_domain():
                return None
        xs.append(T.solver.export_particle_x_to_torch().clone())
    return torch.stack(xs)


from scipy.spatial import cKDTree
restg = torch.nonzero(~sc.keep, as_tuple=False).squeeze(-1)
nd_np, ni_np = cKDTree(T.pos_m.cpu().numpy()).query(sc.pos[restg].cpu().numpy(), k=8)
nd = torch.from_numpy(nd_np).float().to(dev)
ni = torch.from_numpy(ni_np).long().to(dev)
cw = 1.0 / nd.clamp(min=1e-6); cw = cw / cw.sum(-1, keepdim=True)

RUNS, TAGS = [], []
for m in args.mags:
    x = run(m)
    if x is None:
        print(f"  세기 {m:g}: 격자 이탈 -- 건너뜀", flush=True)
        continue
    d = float((x - x[0]).norm(dim=-1).max())
    full = []
    for k in range(x.shape[0]):
        q = sc.pos.clone()
        q[mat] = x[k]
        q[restg] = sc.pos[restg] + (cw.unsqueeze(-1) * (x[k] - T.pos_m)[ni]).sum(1)
        full.append(q)
    RUNS.append(torch.stack(full))
    TAGS.append(f"mag {m:g}   {100*d/EXT:.1f}%")
    print(f"  세기 {m:g}: 최대 변위 {d:.5f} (물체의 {100*d/EXT:.2f}%)", flush=True)

if not RUNS:
    raise SystemExit("모두 이탈했다")
cam = build_camera(sc, args.width, args.height, args.fov_x, args.radius_scale)
frame = make_renderer(sc, args.ply, cam)
vid = []
for k in tqdm(range(args.frames + 1), desc="  render", ncols=88):
    vid.append(np.concatenate([label(frame(R[k]), t) for R, t in zip(RUNS, TAGS)],
                              axis=1))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
imageio.mimwrite(args.out, vid, fps=args.fps, codec="libx264",
                 output_params=["-pix_fmt", "yuv420p"])
print(f"[out] {args.out}  패널 {len(RUNS)}개", flush=True)
print("SWEEP_DONE")
