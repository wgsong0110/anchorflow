"""자유F 학생이 어디서 무너지는가.

세 가지를 한 롤아웃에서 함께 잰다.

1. 채널 절단 -- 학생은 Δp 와 ΔF 를 둘 다 낸다. F 만 정답으로 바꿔 끼우고 굴리면
   실패가 F 예측 탓인지 위치 동역학 탓인지 갈린다. 반대로 p 만 먹여 주면 F 예측이
   혼자서도 성립하는지 보인다.
2. det(F) 궤적 -- 앵커 F 에는 행렬식 가드가 없다(apply_step_frame 은 더하기만 한다).
   부피가 실제로 붕괴하는지, 뒤집히는지를 프레임마다 본다.
3. 실패 국면 -- 프레임별 오차 곡선과 궤적별 최종 오차.

정답은 학습이 쓴 것과 같은 궤적 캐시에서 온다. 홀드아웃 앞 n_holdout 개만 쓴다.
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
from anchorflow.nextstate import NextStep, apply_step_frame

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--fit", required=True)
ap.add_argument("--traj_cache", required=True)
ap.add_argument("--n_holdout", type=int, default=5)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--u_scale", type=float, required=True,
                help="학습 로그의 '프레임 스케일: |u|' 값")
ap.add_argument("--du_scale", type=float, required=True,
                help="같은 줄의 |du| 값")
ap.add_argument("--out_json", default=None)
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
fit = load_fitted(sc, args.fit, dev)[0].fit
fixed = fit.fixed          # 피팅된 앵커 집합의 마스크. sc 쪽은 표본 512 개짜리라 안 맞는다
dt_c = args.dt_mult * sc.sub_dt
# net_from_ckpt 는 프레임 학생을 모른다 -- frame/n_extra 를 안 읽어 출력 폭이
# 3 이 되고 dec 가 어긋난다. 프레임 인자를 넣어 직접 만든다. u/du 스케일은
# 체크포인트에 안 담기므로 학습 로그의 값을 받는다.
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
a = ck["args"]
net = NextStep(a["hidden"], a["depth"], a["heads"], ck["disp_scale"],
               ck["vel_scale"], ck["acc_scale"],
               use_accel=not a.get("no_accel", False),
               chunk=a.get("chunk", 1), frame=True, n_extra=9,
               u_scale=args.u_scale, du_scale=args.du_scale).to(dev)
net.load_state_dict(ck["model"])
net.eval()
EXTENT = float(sc.extent)
print(f"[setup] 앵커 {fit.M}, n_extra {getattr(net, 'n_extra', '?')}, "
      f"dt_c {dt_c:.5g}", flush=True)

blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
TR = blob["trajs"]
HOLD = min(args.n_holdout, len(TR))
print(f"[data] 궤적 {len(TR)}, 홀드아웃 {HOLD}, 상태폭 {TR[0].shape[-1]}", flush=True)


def _p(t):
    return t[..., :3]


def _u(t):
    return t[..., 3:]


def roll(tr, mode):
    """mode: free | force_F | force_p -- 프레임별 지표."""
    T = min(args.frames, tr.shape[0] - 1)
    p = _p(tr[1]).clone()
    u = _u(tr[1]).clone()
    v = (_p(tr[1]) - _p(tr[0])) / dt_c
    # 오차는 고정 상수로, 진폭만 그 궤적의 기준 운동으로 나눈다.
    ref_span = (_p(tr[:T + 1]) - _p(tr[0])).norm(dim=-1).max().clamp(min=1e-12)
    span = EXTENT
    rec = {k: [] for k in ("err", "dF", "det_min", "det_med", "neg", "amp")}
    k = 1
    while k <= T:
        for q, d, uu, _ in apply_step_frame(net, p, v, u, None, None, dt_c, fixed):
            if k > T:
                break
            v, p, u = d / dt_c, q, uu
            tgt = tr[k]
            if mode == "force_F":
                u = _u(tgt).clone()
            elif mode == "force_p":
                p = _p(tgt).clone()
                v = (_p(tgt) - _p(tr[k - 1])) / dt_c
            det = torch.linalg.det(u.view(-1, 3, 3))
            rec["err"].append(float((p - _p(tgt))[~fixed].norm(dim=-1).mean() / span))
            rec["dF"].append(float((u - _u(tgt)).norm(dim=-1).mean()))
            rec["det_min"].append(float(det.min()))
            rec["det_med"].append(float(det.median()))
            rec["neg"].append(float((det <= 0).float().mean()))
            rec["amp"].append(float((p - _p(tr[0])).norm(dim=-1).max() / ref_span))
            k += 1
    return rec


OUT = {}
for mode in ("free", "force_F", "force_p"):
    per = [roll(TR[i], mode) for i in range(HOLD)]
    n = min(len(r["err"]) for r in per)
    agg = {key: [sum(r[key][t] for r in per) / len(per) for t in range(n)]
           for key in per[0]}
    OUT[mode] = agg
    print(f"  {mode:<8} 최종 {100*agg['err'][-1]:6.2f}%  전프레임평균 "
          f"{100*sum(agg['err'])/n:6.2f}%  진폭 {100*agg['amp'][-1]:5.0f}%  "
          f"|dF| {agg['dF'][-1]:.4f}  det최소 {min(agg['det_min']):.4f}  "
          f"det<=0 최대 {100*max(agg['neg']):.2f}%", flush=True)

print("\n[궤적별 -- free]")
for i in range(HOLD):
    r = roll(TR[i], "free")
    print(f"  traj {i}: 최종 {100*r['err'][-1]:6.2f}%  진폭 {100*r['amp'][-1]:5.0f}%  "
          f"det최소 {min(r['det_min']):.4f}  det<=0 최대 {100*max(r['neg']):.2f}%",
          flush=True)
    OUT[f"traj{i}"] = r

if args.out_json:
    json.dump(OUT, open(args.out_json, "w"))
    print(f"[out] {args.out_json}", flush=True)
print("FF_DONE")
