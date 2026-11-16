"""[blx] 정답 없이 재는 물리 지표 (wolf·bread 재질 확장 베이스라인) -- 정의는 exe/stab_metrics.py (안정성 실험) 를 덤프 pred.pt 에 맞춘 것.
R2 lbours/final_fix/metrics_def.md 가 나오면 그 정의로 맞춘다 (이 파일 첫 판 작성 때 아직 없었음).

입력: --pred <덤프 pred.pt {pred [L,P,3], F [L,P,3,3] 또는 None}> (프레임 1..L, 전 입자 -- 부분표본 없음), --data <장면 .pt (x[0] 처음 위치, cfg)>.
지표 (물체 크기 Lx = 0 프레임 전 입자 바운딩박스 대각선):
  뒤집힘     det F ≤ 0 비율의 프레임 최대, det F 최소 (F = 각 방법 표현의 F: VR-GS 결속 사면체 총 F, PhysTwin 공식 데모의 누적 회전 (det 1), Spring-Gaus 는 F 없음 = I)
  부피 변화  |J−1| 평균의 프레임 최대, |J−1| > 0.1 비율의 프레임 최대 (같은 F)
  바닥 관통  바닥 (config surface_collider) 아래 입자 비율의 최대, 최대 깊이 / Lx  (바닥 없는 장면은 없음)
  에너지     E = 운동 + 중력 위치 (입자 질량 = PG 규약; 속도 = 프레임 차분 (x_t − x_{t−1}) / 프레임 간격, 0 프레임은 정답 처음 속도).
             최대 증가 max_t (E_t − E_0)/|E_0|, 마지막/처음 비 E_L / E_0. 탄성 에너지는 넣지 않음 (소성 재질은 F 에 소성 몫이 섞여 F 만으로 탄성 에너지가 안 나옴)
  자유낙하   접촉 (최저점 < 바닥 + 1.5 dx) 전 프레임에서 |Σ m v| 와 M g t 의 상대 오차 최대 -- 처음부터 고정·구동 경계가 있거나 중력 0 이면 자유낙하 구간이 아니라 없음
  떨림       위치 2 차 차분 |x_{t+1} − 2x_t + x_{t−1}| / Lx 의 평균 (전체), 정지 구간 떨림 = 정답이 정지한 프레임 (정답 프레임 평균 변위 < 1e-3 Lx) 에서의 같은 값,
             속도 시계열 스펙트럼 위쪽 절반 주파수 에너지 비율
  python exe/blx_phys_metrics.py --pred P/pred.pt --data wolf.pt --out P/phys.json [--frames 50]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stab_bl_common as C                                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=0)
ap.add_argument("--chunk", type=int, default=10)
a = ap.parse_args()
d = torch.load(a.data, map_location="cpu", weights_only=False, mmap=True)
cfg = d["cfg"]
x0 = d["x"][0].double()
P = torch.load(a.pred, map_location="cpu", weights_only=False, mmap=True)
X, FF = (P["pred"], P.get("F")) if isinstance(P, dict) else (P, None)
L = X.shape[0] if not a.frames else min(a.frames, X.shape[0])
GTX = d["x"][:L + 1]
# 질량 (stab_bl_common.load_scene 과 같은 식)
ng = int(cfg.get("n_grid", 100)); dxg = float(cfg.get("grid_lim", 2.0)) / ng
vi = (x0 / dxg).long().clamp(0, ng - 1)
fl = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cn = torch.zeros(ng ** 3, dtype=torch.float64).index_add_(0, fl, torch.ones(x0.shape[0], dtype=torch.float64))
m = (dxg ** 3) / cn[fl] * float(cfg["density"])
M = float(m.sum())
Lx = float((x0.max(0).values - x0.min(0).values).norm())
g = [float(q) for q in cfg["g"]]
h = float(cfg["frame_dt"])
zf = None
for b in cfg.get("boundary_conditions") or []:
    if b.get("type") == "surface_collider" and abs(float(b["normal"][2]) - 1.0) < 1e-6:
        zf = float(b["point"][2])
BOX, BOXW, MOV = C.parse_bcs(cfg, L * h)
constrained0 = bool(BOX or MOV or any(t0 <= 0.0 for (_p, _s, t0, _t1) in BOXW))
out = dict(frames=int(L), particles=int(x0.shape[0]), Lx=Lx, mass=M, floor=zf, F_kind=("none" if FF is None else "dump"))


def xs(t):
    return x0 if t == 0 else X[t - 1].double()


# 프레임별
detneg, detmin, jdev, jdev10, pen_r, pen_d, KE, PE, pnorm, d2, d2_rest = [], [], [], [], [], [], [], [], [], [], []
v0 = d["v"][0].double()
prev = None
for t in range(0, L + 1):
    x = xs(t)
    v = v0 if t == 0 else (x - xs(t - 1)) / h
    KE.append(float(0.5 * (m * (v * v).sum(1)).sum()))
    PE.append(float(-(m[:, None] * x * torch.tensor(g, dtype=torch.float64)).sum()))
    pnorm.append(float((m[:, None] * v).sum(0).norm()))
    if zf is not None:
        z = x[:, 2]
        pen_r.append(float((z < zf).double().mean())); pen_d.append(max(float((zf - z).max()), 0.0) / Lx)
    if t >= 1 and FF is not None:
        J = torch.linalg.det(FF[t - 1].double())
        detneg.append(float((J <= 0).double().mean())); detmin.append(float(J.min()))
        jdev.append(float((J - 1).abs().mean())); jdev10.append(float(((J - 1).abs() > 0.1).double().mean()))
    if 1 <= t <= L - 1:
        dd = float((xs(t + 1) - 2 * x + xs(t - 1)).norm(dim=1).mean()) / Lx
        d2.append(dd)
        gm = float((GTX[t + 1].double() - GTX[t].double()).norm(dim=1).mean()) / Lx
        if gm < 1e-3:
            d2_rest.append(dd)
E = np.array(KE) + np.array(PE)
out.update(e_drift_max=float(np.max((E - E[0]) / abs(E[0]))) if E[0] != 0 else None, e_last_over_first=float(E[-1] / E[0]) if E[0] != 0 else None,
           e0=float(E[0]), e_last=float(E[-1]), e_has_elastic=False)
if FF is not None:
    out.update(detneg_max=max(detneg), det_min=min(detmin), jdev_max=max(jdev), jdev10_max=max(jdev10))
else:
    out.update(detneg_max=None, det_min=None, jdev_max=None, jdev10_max=None, F_note="표현에 F 없음 (Spring-Gaus 가우시안 등방·회전 없음)")
if zf is not None:
    out.update(pen_ratio_max=max(pen_r), pen_depth_max=max(pen_d))
else:
    out.update(pen_ratio_max=None, pen_depth_max=None, pen_note="바닥 없음")
out.update(jitter_d2=float(np.mean(d2)) if d2 else None, jitter_rest_d2=float(np.mean(d2_rest)) if d2_rest else None, rest_frames=len(d2_rest))
# 스펙트럼 (전 입자, 덩이로)
if L >= 8:
    num = den = 0.0
    P_ = x0.shape[0]
    for s0 in range(0, P_, 20000):
        Xs = torch.stack([xs(t)[s0:s0 + 20000] for t in range(L + 1)]).numpy()
        vv = Xs[1:] - Xs[:-1]
        S = (np.abs(np.fft.rfft(vv - vv.mean(0), axis=0)) ** 2).sum(-1)
        hh = S.shape[0] // 2
        r = S[hh:].sum(0) / np.maximum(S.sum(0), 1e-30)
        num += float(r.sum()); den += r.shape[0]
    out["jitter_hf"] = num / den
# 자유낙하
if constrained0 or g[2] == 0.0 or zf is None:
    out.update(freefall_mom_err=None, freefall_note=("처음부터 고정·구동 경계" if constrained0 else "중력 0 또는 바닥 없음") + " -- 자유낙하 구간 없음")
else:
    T_c = next((t for t in range(L + 1) if float(xs(t)[:, 2].min()) < zf + 1.5 * dxg), L + 1)
    errs = [abs(pnorm[t] - M * abs(g[2]) * t * h) / (M * abs(g[2]) * t * h) for t in range(1, T_c)]
    out.update(freefall_mom_err=max(errs) if errs else None, freefall_frames=T_c)
json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False), flush=True)
