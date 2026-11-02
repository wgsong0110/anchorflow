"""절 AN.14 미달 궤적 진단: 예측 덤프와 정답의 프레임별 오차를 강체 몫과 나머지로 나눈다.
- 전체 오차 e = mean |x − x_gt| / EXT (판정 지표 그대로)
- 질량 중심 차 |c − c_gt| / EXT, 바닥 법선 방향 높이 차 (부호), 수평 차
- 최적 강체 (Kabsch, 질량 가중) 로 예측을 정답에 맞춘 뒤 남는 오차 e_def (변형 몫), 그 회전각 (도)
- 각 프레임 정답·예측 최저 높이 (접촉 시점), 정답·예측 질량 중심 속도 (법선, 수평)
사용: python exe/fem_an_rigid_ana.py <궤적.pt> <이름:pred.pt> [...] [--every 5]"""
import math
import sys

import numpy as np
import torch

args = [a for a in sys.argv[1:] if not a.startswith("--")]
every = 5
if "--every" in sys.argv:
    every = int(sys.argv[sys.argv.index("--every") + 1])
d = torch.load(args[0], map_location="cpu", weights_only=False)
X = d["x"].double()
EXT = float(d["_ext"])
cfg = d["cfg"]
h = float(cfg["frame_dt"])
pl = [b for b in cfg.get("boundary_conditions", []) if b.get("type") == "surface_collider"]
NR = np.array(pl[0]["normal"], dtype=np.float64) if pl else np.array([0.0, 0.0, 1.0])
NR = torch.tensor(NR / np.linalg.norm(NR), dtype=torch.float64)
PT = torch.tensor(pl[0]["point"], dtype=torch.float64) if pl else torch.zeros(3)
w = None
try:
    from anchorflow import deform_model as DM  # noqa
except Exception:
    pass
N = X.shape[1]
w = torch.ones(N, dtype=torch.float64) / N


def kabsch(P, Q):
    cp_, cq = (w[:, None] * P).sum(0), (w[:, None] * Q).sum(0)
    H = ((P - cp_) * w[:, None]).T @ (Q - cq)
    U, S, Vt = torch.linalg.svd(H)
    dd = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, float(dd)], dtype=torch.float64))
    Rm = Vt.T @ D @ U.T
    return Rm, cp_, cq


for spec in args[1:]:
    nm, pp = spec.split(":", 1)
    Pd = torch.load(pp, map_location="cpu", weights_only=False)["pred"].double()
    L = Pd.shape[0]
    print(f"## {nm} ({L} 프레임) -- t, e%, 중심차%, 높이차 mm, 수평차 mm, 변형 몫 e_def%, 강체 회전각 도, 최저 높이 정답/예측 mm, 중심 법선 속도 정답/예측 m/s")
    rows = []
    for t in range(L):
        G, P = X[t + 1], Pd[t]
        e = float((P - G).norm(dim=-1).mean()) / EXT
        cg, cpd = (w[:, None] * G).sum(0), (w[:, None] * P).sum(0)
        dc = cpd - cg
        dn = float(dc @ NR); dh = float((dc - dn * NR).norm())
        Rm, a_, b_ = kabsch(P, G)
        Pf = (P - a_) @ Rm.T + b_
        ed = float((Pf - G).norm(dim=-1).mean()) / EXT
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (float(torch.trace(Rm)) - 1) / 2))))
        hg = float(((G - PT) @ NR).min()); hp = float(((P - PT) @ NR).min())
        vg = float(((X[t + 1] - X[t]).mean(0)) @ NR) / h
        vp = float(((Pd[t] - (Pd[t - 1] if t > 0 else X[0])).mean(0)) @ NR) / h
        rows.append((t + 1, 100 * e, 100 * float(dc.norm()) / EXT, 1e3 * dn, 1e3 * dh, 100 * ed, ang, 1e3 * hg, 1e3 * hp, vg, vp))
    for r in rows:
        if r[0] % every == 0 or r[0] == 1:
            print("  " + " ".join(f"{q:.2f}" if isinstance(q, float) else str(q) for q in r))
    A = np.array(rows)
    print(f"  평균: e {A[:, 1].mean():.2f} %, 중심차 {A[:, 2].mean():.2f} %, 변형 몫 {A[:, 5].mean():.2f} %, 회전각 {A[:, 6].mean():.1f} 도")
    # 마지막 프레임 입자별 오차가 큰 곳 (정답 몸 좌표 높이 사분위별)
    G0 = X[0]; hh = (G0 - PT) @ NR
    q = torch.quantile(hh, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    eL = (Pd[-1] - X[L]).norm(dim=-1) / EXT
    bins = [hh <= q[0], (hh > q[0]) & (hh <= q[1]), (hh > q[1]) & (hh <= q[2]), hh > q[2]]
    print("  마지막 프레임 오차 % (처음 높이 사분위 아래 -> 위): " + " / ".join(f"{100 * float(eL[b].mean()):.2f}" for b in bins))
