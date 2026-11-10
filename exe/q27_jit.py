"""절 CM101 수박 꿈틀거림 지표 (읽기만): 덤프 위치 (pred.pt) 의 프레임 사이 변위 속도에서 강체 운동 (질량중심 병진 + 최소제곱 각속도) 을 뺀 RMS (질량 가중, mm/s).
v_f = (x_{f+1} − x_f)/h (프레임 평균 속도; 절 CM90 은 실행기 입자 v 순간값이라 수치가 다름), 창 = 우리 프레임 f30~f100 (배열 fi 29..99, CM90 과 같은 창).
정답 (GF) 은 wm.pt 의 프레임별 x 로 같은 식 (--gt). 비유한 입자는 그 프레임에서 뺌.
사용: python exe/q27_jit.py --data wm.pt [--pred pred.pt | --gt] --mass q27solver.pt --out jit.json
"""
import argparse
import json

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--pred", default="")
ap.add_argument("--gt", action="store_true")
ap.add_argument("--mass", default="")
ap.add_argument("--f0", type=int, default=29)
ap.add_argument("--f1", type=int, default=100)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load(a.data, map_location="cpu", weights_only=False)
h = float(d["cfg"]["frame_dt"])
if a.gt:
    X = d["x"][1:].float()                                                              # 배열 fi = 정답 프레임 fi + 1 (실행기 오차와 같은 짝)
else:
    X = torch.load(a.pred, map_location="cpu", weights_only=False)["pred"].float()
N = X.shape[1]
if a.mass:
    m = torch.load(a.mass, map_location="cpu", weights_only=False)["m"].double().reshape(-1)
else:
    m = torch.ones(N, dtype=torch.float64)
assert m.numel() == N, (m.numel(), N)
m = m.to(dev)
rows = []
f1 = min(a.f1, X.shape[0] - 1)
for fi in range(a.f0, f1):
    x0 = X[fi].to(dev).double(); x1 = X[fi + 1].to(dev).double()
    v = (x1 - x0) / h
    ok = torch.isfinite(v).all(1) & torch.isfinite(x0).all(1)
    mm, xx, vv = m[ok], x0[ok], v[ok]
    M = mm.sum()
    c = (mm[:, None] * xx).sum(0) / M
    vc = (mm[:, None] * vv).sum(0) / M
    r = xx - c; u = vv - vc
    Lm = (mm[:, None] * torch.cross(r, u, dim=1)).sum(0)
    I = (mm * (r * r).sum(1)).sum() * torch.eye(3, device=dev, dtype=torch.float64) - (mm[:, None, None] * r[:, :, None] * r[:, None, :]).sum(0)
    w = torch.linalg.solve(I, Lm)
    dv = u - torch.cross(w.expand_as(r), r, dim=1)
    rms = float(torch.sqrt((mm * (dv * dv).sum(1)).sum() / M)) * 1e3
    rows.append(dict(fi=fi, vdef_rms_mm_s=rms, vcm_mm_s=float(vc.norm()) * 1e3, w_rad_s=float(w.norm()), nbad=int((~ok).sum())))
R = [q["vdef_rms_mm_s"] for q in rows]
out = dict(src="gt" if a.gt else a.pred, h=h, f0=a.f0, f1=f1, n=len(rows), vdef_rms_mean=(sum(R) / len(R) if R else None), vdef_rms_max=(max(R) if R else None), rows=rows)
json.dump(out, open(a.out, "w"), indent=1)
print("[q27 jit]", out["src"], "창", a.f0, f1, "평균", out["vdef_rms_mean"], "최대", out["vdef_rms_max"])
