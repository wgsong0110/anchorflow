"""로프(가는 원기둥) 한쪽 끝을 잡고 **빠르게 내렸다 올리는** 장면을 만든다.

기준 궤적은 정지 자세(해가 아니라 기준점일 뿐)이고, 진짜 기준은 이 입자·손잡이
설정을 그대로 PG MPM 에 먹여 받는다 (exe/mk_ball_pg_inputs.py).

손잡이 명령은 프레임 구간으로 나눈다: 들고 있기 -> 아래로 -> 위로.
"""
from __future__ import annotations
import argparse
import math
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--ref", default="")
ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--r", type=float, default=0.03, help="로프 반지름")
ap.add_argument("--len", type=float, default=0.40, help="로프 길이")
ap.add_argument("--top", type=float, nargs=3, default=(1.0, 1.0, 1.55),
                help="윗끝 중심")
ap.add_argument("--hold", type=int, default=12)
ap.add_argument("--down", type=int, default=12)
ap.add_argument("--up", type=int, default=12)
ap.add_argument("--vz", type=float, default=1.5, help="내리고 올리는 속력")
ap.add_argument("--handle_r", type=float, default=0.03)
ap.add_argument("--material", default="jelly")
ap.add_argument("--E", type=float, default=2e5)
ap.add_argument("--nu", type=float, default=0.3)
ap.add_argument("--floor", type=float, default=0.95,
                help="바닥 높이 z. 음수면 바닥을 두지 않는다. PG 와 같은 "
                     "surface_collider(sticky) 형식으로 넣는다")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cfg = dict(E=a.E, nu=a.nu, g=[0.0, 0.0, -9.8], frame_dt=1.0 / 60.0,
           density=1000.0, n_grid=200, grid_lim=2.0, material=a.material,
           boundary_conditions=[])
if a.floor >= 0:
    cfg["boundary_conditions"] = [
        {"type": "surface_collider",
         "point": [1.0, 1.0, float(a.floor)],
         "normal": [0.0, 0.0, 1.0],
         "surface": "sticky", "friction": 0.0,
         "start_time": 0, "end_time": 1000.0},
    ]
if a.ref:
    try:
        rc = torch.load(a.ref, map_location="cpu", weights_only=False)["cfg"]
    except TypeError:
        rc = torch.load(a.ref, map_location="cpu")["cfg"]
    for k in ("frame_dt", "density", "grid_lim"):
        if k in rc:
            cfg[k] = rc[k]
print(f"[cfg] E {cfg['E']:g} nu {cfg['nu']} dt {cfg['frame_dt']:.5f} "
      f"n_grid {cfg['n_grid']} 재질 {cfg['material']} 바닥 "
      f"{'없음' if a.floor < 0 else f'z={a.floor}'}")

g_ = torch.Generator().manual_seed(a.seed)
u = torch.rand(a.n, generator=g_).sqrt() * a.r         # 반지름 (면적 균일)
th = torch.rand(a.n, generator=g_) * 2 * math.pi
zz = torch.rand(a.n, generator=g_) * a.len
x0 = torch.stack([u * th.cos(), u * th.sin(), -zz], -1)
x0 = x0 + torch.tensor(a.top, dtype=torch.float32)

T = a.hold + a.down + a.up + 1
xs = x0.unsqueeze(0).expand(T, a.n, 3).clone()         # 기준은 정지 자세
cid = int(x0[:, 2].argmax())                            # 윗끝 입자
vel = torch.zeros(T, 1, 3)
vel[a.hold:a.hold + a.down, 0, 2] = -a.vz
vel[a.hold + a.down:, 0, 2] = a.vz
d = dict(x=xs, F=torch.eye(3).expand(T, a.n, 3, 3).clone(), cfg=cfg,
         sel=torch.arange(a.n), n_full=a.n,
         ctrl_id=torch.tensor([cid], dtype=torch.long),
         ctrl_vel=vel, ctrl_R=torch.tensor([a.handle_r]),
         ctrl_pos=xs[:, cid].reshape(T, 1, 3).clone())
torch.save(d, a.out)
_in = int(((x0 - x0[cid]).norm(dim=-1) < a.handle_r).sum())
vol = math.pi * a.r ** 2 * a.len
print(f"[저장] {a.out}  {T} 프레임 x {a.n} 입자, 로프 반지름 {a.r} 길이 "
      f"{a.len} (부피 {vol:.3e})")
print(f"[손잡이] 입자 {cid} (z={float(x0[cid,2]):.4f}), 반경 {a.handle_r}, "
      f"반경 안 {_in} 개 | 명령: {a.hold} 프레임 들고 -> {a.down} 프레임 "
      f"-{a.vz} -> {a.up} 프레임 +{a.vz}")
