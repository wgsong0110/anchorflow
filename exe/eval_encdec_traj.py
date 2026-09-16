"""앵커 인코딩/디코딩을 매 스텝 끼운 궤적이 MPM 원본을 얼마나 따라가는지 잰다.

한 번만 접었다 펴는 잔차는 표현력의 하한만 말해준다. 여기서는 **매 스텝** 끼워서
오차가 실제로 누적되게 한다:

    기준 : x_{t+1} = MPM(x_t)
    비교 : x'_{t+1} = MPM(decode(encode(x'_t)))

두 궤적을 대응을 가정하지 않는 지표로 비교한다 -- 이 실험의 관심사가 찢어짐처럼
대응이 흐려지는 경우이기 때문이다.

    Chamfer      각 점에서 상대 구름의 최근접까지, 양방향 평균
    Earth Mover  최적 일대일 대응의 평균 이동량 (O(n^3) 이라 부분표본)

둘 다 물체 크기로 나눈다. 자기 변위로 나누면 덜 움직인 쪽이 유리해진다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--fit", default=None,
                help="학습된 기하. 없으면 초기 기하로 (= 학습 전)")
ap.add_argument("--dreamphysics", required=True)
ap.add_argument("--anchorflow", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="run")
ap.add_argument("--softmax_w", action="store_true",
                help="가중치를 kNN 위 softmax 로 (기본은 잘린 가우시안)")
ap.add_argument("--softmax_k", type=int, default=16)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--frames", type=int, default=40)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--emd_sample", type=int, default=2048)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

sys.path.insert(0, a.anchorflow)
sys.path.insert(0, a.dreamphysics)
dev = "cuda"
torch.set_grad_enabled(False)

import warp as wp  # noqa: E402

wp.init()
from anchorflow import scene_setup  # noqa: E402
from anchorflow.anchor_sparse import AnchorSparse, load_fitted  # noqa: E402
from anchorflow.mpm_teacher import MPMTeacher  # noqa: E402

sc = scene_setup.build(a.ply, a.config, a.n_anchors, a.K, device=dev,
                       frozen_weights=False, rot_fallback=True, eig_floor=0.02)
EXT = float(sc.extent)
FRAME_DT = float(sc.sub_dt) * a.dt_mult
print(f"[씬] 물질 {int(sc.keep.sum())}, 물체 {EXT:.4f}, frame_dt {FRAME_DT:.4g}",
      flush=True)

if a.fit:
    fit = load_fitted(sc, a.fit, device=dev)
    print(f"[기하] {a.fit}, 앵커 {fit.M}", flush=True)
else:
    fit = AnchorSparse(sc).to(dev)
    print(f"[기하] 학습 없음 (초기 기하), 앵커 {fit.M}", flush=True)
# 가중치 방식은 기하와 별개로 끼운다 -- 같은 기하 위에서 두 방식을 견주기 위해서다
fit.softmax_w = a.softmax_w
fit.softmax_k = a.softmax_k
print(f"[가중치] {'kNN softmax k=%d' % a.softmax_k if a.softmax_w else '잘린 가우시안'}",
      flush=True)


def chamfer(p, q):
    d = torch.cdist(p, q)
    return 0.5 * (d.min(1).values.mean() + d.min(0).values.mean())


def emd(p, q, n, g):
    from scipy.optimize import linear_sum_assignment
    ip = torch.randperm(p.shape[0], generator=g)[:n].to(p.device)
    iq = torch.randperm(q.shape[0], generator=g)[:n].to(q.device)
    d = torch.cdist(p[ip], q[iq]).double().cpu().numpy()
    r, c = linear_sum_assignment(d)
    return float(d[r, c].mean())


T = MPMTeacher(sc, horizon=a.frames * FRAME_DT * 1.2)
cache = fit.prepare()      # 짝 목록과 가중치를 한 번 준비한다
KEEP = sc.keep
g = torch.Generator(device=dev).manual_seed(a.seed)

# 구동은 씬 config 가 정한 것을 그대로 쓴다 (임의 임펄스가 아니다).
T._set(T.pos_m.clone(), torch.zeros(T.n, 3, device=dev), T.eye.clone(),
       torch.zeros_like(T.eye))
base = [T.pos_m.clone()]
for _ in range(a.frames - 1):
    for k in range(a.dt_mult):
        T._switch_rotation()
        T.solver.p2g2p(None, float(sc.sub_dt), device=T.wp_dev)
    base.append(T.solver.export_particle_x_to_torch().clone())
BASE = torch.stack(base)
print(f"[기준] MPM {BASE.shape[0]} 프레임, 최대 변위 "
      f"{100*float((BASE-BASE[0]).norm(dim=-1).max())/EXT:.2f}%", flush=True)

# 비교 궤적: 매 스텝 encode -> decode 를 끼운다
T._set(T.pos_m.clone(), torch.zeros(T.n, 3, device=dev), T.eye.clone(),
       torch.zeros_like(T.eye))
enc = [T.pos_m.clone()]
for t in range(a.frames - 1):
    for k in range(a.dt_mult):
        T._switch_rotation()
        T.solver.p2g2p(None, float(sc.sub_dt), device=T.wp_dev)
    x = T.solver.export_particle_x_to_torch()
    # 앵커 상태로 접었다가 (project) 다시 가우시안으로 편다 (gaussian_pos)
    ap_ = fit.project(x, cache)
    x2 = fit.gaussian_pos(ap_, cache)
    enc.append(x2.clone())
    # 편 상태를 솔버에 되돌려 다음 스텝이 그 위에서 진행되게 한다
    T.solver.import_particle_x_from_torch(x2)
ENC = torch.stack(enc)

rows = []
for t in range(BASE.shape[0]):
    cd = float(chamfer(ENC[t], BASE[t])) / EXT
    em = emd(ENC[t], BASE[t], a.emd_sample, torch.Generator().manual_seed(t)) / EXT
    rows.append(dict(frame=t, cd=cd, emd=em))
    if t % 10 == 0 or t == BASE.shape[0] - 1:
        print(f"  f{t:3d}  CD {100*cd:7.4f}%  EMD {100*em:7.4f}%", flush=True)

summ = dict(tag=a.tag, softmax_w=a.softmax_w, softmax_k=a.softmax_k,
            fit=a.fit, frames=BASE.shape[0], extent=EXT,
            cd_mean=float(np.mean([r["cd"] for r in rows])),
            emd_mean=float(np.mean([r["emd"] for r in rows])),
            cd_last=rows[-1]["cd"], emd_last=rows[-1]["emd"], rows=rows)
os.makedirs(a.out, exist_ok=True)
json.dump(summ, open(os.path.join(a.out, f"encdec_{a.tag}.json"), "w"), indent=1)
print(f"\n[요약] {a.tag}: CD 평균 {100*summ['cd_mean']:.4f}% 마지막 "
      f"{100*summ['cd_last']:.4f}% | EMD 평균 {100*summ['emd_mean']:.4f}% "
      f"마지막 {100*summ['emd_last']:.4f}%", flush=True)
print("ENCDEC_OK")
