"""새 장면의 임펄스 세기를 정하고, 그 장면의 표현 하한을 함께 잰다.

ficus 는 config 에 particle_impulse 가 있어 그 값을 기준으로 삼았는데 다른
장면들에는 없다. E 와 밀도가 자릿수로 다르므로 ficus 값을 그대로 쓰면 안 된다.
세기를 로그로 훑으며 MPM 이 실제로 만드는 최대 변위를 재고, 물체 크기의 몇
퍼센트인지로 고른다 -- ficus 의 config 임펄스가 35% 를 내므로 거기 맞추면 두
장면이 같은 난이도가 된다. 추천값은 PICK= 줄로 낸다.

표현 하한도 같이 낸다: MPM 입자를 앵커로 접었다 편 잔차다. 학생이 아무리 잘해도
못 넘는 바닥이고, 알갱이가 서로 떨어지는 재료라면 여기서 드러난다 -- 구성
방정식이 소성이냐가 아니라 이웃 관계가 유지되느냐의 문제다.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--n_anchors", type=int, default=1024)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--dir", type=float, nargs=3, default=[1.0, 0.0, 0.0])
ap.add_argument("--lo", type=float, default=1e-3)
ap.add_argument("--hi", type=float, default=1e2)
ap.add_argument("--n", type=int, default=9)
ap.add_argument("--target", type=float, default=0.35,
                help="목표 변위 / 물체 크기. ficus 의 config 임펄스가 0.35 를 낸다.")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.mpm_teacher import MPMTeacher

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
T = MPMTeacher(sc)
MAT = T.mat
EXT = float(sc.extent)
NFIX = int(sc.fixed_mask.sum())
print(f"[setup] 가우시안 {sc.N} (시뮬 {T.n}), 앵커 {sc.M}, 고정 {NFIX}, "
      f"물체 크기 {EXT:.4f}, 입자 간격 {sc.sim.radius:.5f}, sub_dt {sc.sub_dt:g}",
      flush=True)
if NFIX == 0:
    print("  [경고] 고정 앵커가 없다 -- 임펄스가 물체를 통째로 밀기만 할 수 있다",
          flush=True)

D = torch.tensor(args.dir, device=dev, dtype=torch.float32)
D = D / D.norm().clamp(min=1e-12)


def mpm_particles(force):
    """MPM 입자 위치 [T+1, n, 3], 격자를 벗어나면 None."""
    dv = sc.impulse_dv(force)
    v0 = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
    T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
    xs = [T.pos_m.clone()]
    CE = 2  # 고정점 없는 씬은 8 substep 사이에 격자를 벗어난다
    if not T._vel_safe(CE):
        return None
    for _ in range(args.frames):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % CE == 0:
                if not T._in_domain() or not T._vel_safe(CE):
                    return None
        xs.append(T.solver.export_particle_x_to_torch().clone())
    return torch.stack(xs)


REC = []
print(f"\n{'세기':>12}{'최대 변위':>12}{'물체 대비':>10}{'표현 하한':>11}{'상태':>6}")
for i in range(args.n):
    u = i / max(1, args.n - 1)
    mag = args.lo * ((args.hi / args.lo) ** u)
    f = (D * mag).unsqueeze(0).expand(sc.pos.shape[0], 3).contiguous()
    p = mpm_particles(f)
    if p is None:
        print(f"{mag:>12.4g}{'--':>12}{'--':>10}{'--':>11}{'이탈':>6}", flush=True)
        continue
    disp = float((p - p[0]).norm(dim=-1).max())
    # 하한도 고정 상수(물체 크기)로 나눈다 -- 궤적 자신의 변위로 나누면 거의 안
    # 움직이는 세기에서 분모가 작아져 하한이 폭발한다.
    e = []
    for t in range(0, p.shape[0], max(1, p.shape[0] // 12)):
        gp = sc.skin(T.project(p[t]), sc.pos.clone())[MAT]
        e.append(float((gp - p[t]).norm(dim=-1).mean() / EXT))
    fl = 100 * sum(e) / len(e)
    REC.append((mag, disp / EXT))
    print(f"{mag:>12.4g}{disp:>12.5f}{100*disp/EXT:>9.2f}%{fl:>10.2f}%{'ok':>6}",
          flush=True)

ok = [(m, fr) for m, fr in REC if fr > 1e-9]
if len(ok) >= 2:
    lm = [math.log(m) for m, _ in ok]
    lf = [math.log(fr) for _, fr in ok]
    lt = math.log(args.target)
    j = min(range(len(lf)), key=lambda i: abs(lf[i] - lt))
    j2 = j - 1 if (lf[j] > lt and j > 0) else min(len(lf) - 1, j + 1)
    if j2 == j or abs(lm[j2] - lm[j]) < 1e-9:
        pick = ok[j][0] * (args.target / ok[j][1])
    else:
        a, b = sorted((j, j2))
        sl = (lf[b] - lf[a]) / (lm[b] - lm[a])
        pick = math.exp(lm[a] + (lt - lf[a]) / (sl if abs(sl) > 1e-9 else 1e-9))
    print(f"\n목표 변위 {100*args.target:.0f}% 에 맞는 추천 세기")
    print(f"PICK={pick:.6g}")
else:
    print("PICK=NONE  -- 쓸 만한 표본이 없다")
print("FORCE_DONE")
