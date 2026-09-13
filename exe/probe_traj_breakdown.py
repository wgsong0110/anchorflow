"""형상 기하 학생이 어떤 임펄스에서 무너지는가.

eval_vs_mpm 이 만들어 둔 MPM 기준 궤적 캐시를 그대로 쓴다 -- 같은 임펄스, 같은
자다. 임펄스마다 학생을 굴려 오차를 재고, 그 임펄스가 어떤 것이었는지를 나란히
놓는다. 임펄스 기술자는 힘장에서 직접 뽑는다:

  세기   힘의 RMS
  자국   힘이 최대의 5% 를 넘는 입자 비율 -- 포크 개수 x 반경^3 에 해당
  퍼짐   그 입자들의 공간 표준편차 (격자 간격 단위)
  운동   MPM 자신의 최대 변위 -- 그 임펄스가 실제로 만든 움직임의 크기
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow import scene_setup
from anchorflow.nextstate import apply_step, net_from_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--student", action="append", required=True, help="이름:학생ckpt:기하")
ap.add_argument("--cache", required=True, help="eval_vs_mpm 의 --cache 파일")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--out_json", default=None)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                       frozen_weights=True, rot_fallback=True,
                       eig_floor=args.eig_floor)
T = MPMTeacher(sc)
mat = T.mat
dt = args.dt_mult * sc.sub_dt

blob = torch.load(args.cache, map_location=dev, weights_only=False)
key = list(blob["ref"].keys())[0]
REF, FORCE = blob["ref"][key], blob["force"][key]
print(f"[ref] {args.cache}: 임펄스 {len(REF)}개", flush=True)

H = sc.sim.radius          # 입자 간격, 퍼짐의 단위
EXTENT = float(sc.extent)  # 오차의 고정 분모


def describe(f, truth):
    """임펄스가 어떤 것이었나 + 그것이 실제로 만든 움직임."""
    mag = f.norm(dim=-1)
    hot = mag > 0.05 * mag.max().clamp(min=1e-20)
    pos = sc.pos[hot]
    spread = float(pos.std(0).norm()) / H if int(hot.sum()) > 1 else 0.0
    return {"세기": float(mag.pow(2).mean().sqrt()),
            "자국": float(hot.float().mean()),
            "퍼짐": spread,
            "운동": float((truth - truth[0]).norm(dim=-1).max())}


def score(pred, truth):
    """오차는 고정 상수(물체 크기)로, 진폭만 그 궤적의 기준 운동으로 나눈다.

    임펄스마다 그 임펄스의 변위로 오차를 나누면 거의 안 움직이는 임펄스에서
    분모가 0 에 가까워져 잔떨림이 100% 넘게 찍힌다.
    """
    ref_span = (truth - truth[0]).norm(dim=-1).max().clamp(min=1e-12)
    e = (pred - truth).norm(dim=-1).mean(-1) / EXTENT
    return float(e.mean()), float(((pred - pred[0]).norm(dim=-1).max() / ref_span))


def run(net, fit, force):
    s_ = fit
    p, v = s_.anchor_canonical.clone(), s_.initial_velocity(force)
    gp = s_.pos.clone()
    out = [gp[mat].clone()]
    k = 0
    while k < args.frames:
        for q, d in apply_step(net, p, v, None, dt, s_.fixed_mask):
            p, v = q, d / dt
            if not torch.isfinite(p).all():
                return None
            gp = s_.skin(p, gp)
            out.append(gp[mat].clone())
            k += 1
            if k >= args.frames:
                break
    return torch.stack(out)


DESC = [describe(FORCE[i], REF[i]) for i in range(len(REF))]
OUT = {"desc": DESC, "students": {}}

for spec in args.student:
    name, ck_p, fit_p = spec.split(":", 2)
    # load_fitted 의 첫 원소는 래퍼다 -- anchor_canonical/skin/initial_velocity 가
    # 거기 있고, .fit 을 벗기면 롤아웃에 필요한 것들이 사라진다.
    fit = load_fitted(sc, fit_p, dev)[0]
    net = net_from_ckpt(torch.load(ck_p, map_location=dev, weights_only=False), dev)
    rows = []
    for i in range(len(REF)):
        pr = run(net, fit, FORCE[i])
        if pr is None:
            rows.append({"err": float("nan"), "amp": float("nan")})
            continue
        e, a = score(pr, REF[i])
        rows.append({"err": 100 * e, "amp": 100 * a})
    OUT["students"][name] = rows
    ok = [r["err"] for r in rows if r["err"] == r["err"]]
    print(f"  {name}: 앵커 {fit.fit.M}, 평균 {sum(ok)/len(ok):.2f}%, "
          f"최악 {max(ok):.2f}%, 최선 {min(ok):.2f}%, 발산 {len(rows)-len(ok)}", flush=True)
    fit = net = None
    torch.cuda.empty_cache()

names = list(OUT["students"])
print(f"\n{'#':>3} {'세기':>9} {'자국':>7} {'퍼짐':>7} {'운동':>8} " +
      " ".join(f"{n:>16}" for n in names))
order = sorted(range(len(REF)), key=lambda i: -OUT["students"][names[0]][i]["err"])
for i in order:
    d = DESC[i]
    cells = " ".join(f"{OUT['students'][n][i]['err']:>8.1f}%"
                     f"{OUT['students'][n][i]['amp']:>7.0f}%" for n in names)
    print(f"{i:>3} {d['세기']:>9.4f} {100*d['자국']:>6.2f}% {d['퍼짐']:>7.2f} "
          f"{d['운동']:>8.4f} {cells}")

# 상관: 어떤 기술자가 오차를 설명하나
import math
for n in names:
    e = [OUT["students"][n][i]["err"] for i in range(len(REF))]
    print(f"\n[{n}] 오차와의 피어슨 상관")
    for k in ("세기", "자국", "퍼짐", "운동"):
        x = [DESC[i][k] for i in range(len(REF))]
        mx, me = sum(x)/len(x), sum(e)/len(e)
        cov = sum((a-mx)*(b-me) for a, b in zip(x, e))
        sx = math.sqrt(sum((a-mx)**2 for a in x)); se = math.sqrt(sum((b-me)**2 for b in e))
        print(f"   {k}: {cov/(sx*se+1e-20):+.3f}")

if args.out_json:
    json.dump(OUT, open(args.out_json, "w"), ensure_ascii=False)
    print(f"[out] {args.out_json}", flush=True)
print("BREAK_DONE")
