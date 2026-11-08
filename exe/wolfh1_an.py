"""절 CM52 묶음 wolfh1 진단 (읽기만, 결과 불변): wolf 정답과 우리 판의 프레임별 곡선 -- 높이 (질량중심·최대·99 백분위, 바닥 법선), 바닥 접촉 면적,
몸 퍼짐 (바닥 평면 안 회전 반경·1~99 백분위 폭), 높이 층별 접선 속력, 운동 에너지, 위치 에너지 감소, 탄성 에너지 (F 가 있으면, Hencky), 남는 몫 (= 소산: 소성·수치·고착).

사용: python exe/wolfh1_an.py <wolf.pt> <출력.json> <판 디렉토리 ...>   (판 디렉토리에 pred.pt 가 있어야; 없으면 건너뜀)
정의 (정답·우리 같은 함수):
- 높이 h = (x − 바닥 점)·n (바닥 = cfg surface_collider), 질량 가중 질량중심 높이·최대·99 백분위 (mm).
- 접촉 면적 = 바닥 위 높이 < dx_정답 (1 cm) 입자를 바닥 평면에 투영해 5 mm 칸에 넣은 점유 칸 수 x 칸 넓이 (cm²).
- 퍼짐 = 평면 안 질량 가중 회전 반경 (mm), 평면 두 축 1~99 백분위 폭 (mm).
- 속도 = (x_f − x_{f−1}) / frame_dt (정답·우리 같은 유한 차분), 운동 에너지 ½ Σ m |v|² (J); 층별 접선 속력 = 높이 [0,2)·[2,4)·[4,8)·[8,∞) cm 입자 질량 가중 평균 |v_t| (m/s).
- 위치 에너지 감소 = M g (h̄_0 − h̄_f) (J). 탄성 = Σ V_p (μ |log σ|² + λ/2 (Σ log σ)²) (J; F 가 pred.pt·정답에 있을 때만).
- 남는 몫 = 위치 에너지 감소 − 운동 − 탄성 (J; 탄성 없으면 운동만 뺌) -- 소성·수치 감쇠·바닥 고착이 가져간 에너지 합.
- 갈라짐 = |질량중심 우리 − 정답| 이 5·10·20 mm 를 처음 넘는 프레임.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402

d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
OUT = sys.argv[2]
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, "cpu").double()
M = float(m.sum())
vol = m / float(cfg["density"])
mu, lam = PR.lame(cfg["E"], cfg["nu"])
pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
NR = torch.tensor(pl["normal"], dtype=torch.float64); NR = NR / NR.norm()
PT = torch.tensor([float(q) for q in pl["point"]], dtype=torch.float64)
gv = torch.tensor([float(q) for q in cfg["g"]], dtype=torch.float64)
G = float(-(gv @ NR))
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
# 바닥 평면 두 축
a0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
if abs(float(a0 @ NR)) > 0.9:
    a0 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
E1 = a0 - (a0 @ NR) * NR; E1 = E1 / E1.norm(); E2 = torch.cross(NR, E1, dim=0)
BIN = 0.005
LAY = [(0.0, 0.02), (0.02, 0.04), (0.04, 0.08), (0.08, 9.0)]
print(f"[wolfh1_an] 입자 {m.numel()}, 질량 {M:.3f} kg, μ {mu:.4g} λ {lam:.4g}, 바닥 n {NR.tolist()} 점 {PT.tolist()}, g_n {G:.3f}, dx {DXG}, 정답 키 {sorted(k for k in d.keys())}", flush=True)


def elastic(F):
    F = F.double()
    ok = torch.isfinite(F.reshape(-1, 9)).all(-1)
    s = torch.linalg.svdvals(F[ok]).clamp_min(1e-12).log()
    w = mu * (s ** 2).sum(-1) + 0.5 * lam * s.sum(-1) ** 2
    return float((vol[ok] * w).sum())


def frame(x, xp, F=None):
    x = x.double()
    ok = torch.isfinite(x).all(-1)
    hh = (x - PT) @ NR
    mo = m * ok
    r = dict(com=float((mo * torch.where(ok, hh, 0)).sum() / mo.sum()) * 1000,
             zmax=float(hh[ok].max()) * 1000, z99=float(torch.quantile(hh[ok][::3], 0.99)) * 1000)
    low = ok & (hh < DXG)
    u, w = x @ E1, x @ E2
    if int(low.sum()):
        iu = torch.floor(u[low] / BIN).long(); iw = torch.floor(w[low] / BIN).long()
        key = torch.unique((iu - iu.min()) * 100000 + (iw - iw.min()))
        r["area_cm2"] = float(key.numel() * BIN * BIN * 1e4)
    else:
        r["area_cm2"] = 0.0
    r["n_low"] = int(low.sum())
    cu = float((mo * torch.where(ok, u, 0)).sum() / mo.sum()); cw = float((mo * torch.where(ok, w, 0)).sum() / mo.sum())
    r["rg_mm"] = float(torch.sqrt((mo * torch.where(ok, (u - cu) ** 2 + (w - cw) ** 2, 0)).sum() / mo.sum())) * 1000
    qs = torch.tensor([0.01, 0.99], dtype=torch.float64)
    qu = torch.quantile(u[ok][::3], qs); qw = torch.quantile(w[ok][::3], qs)
    r["wu_mm"] = float(qu[1] - qu[0]) * 1000; r["ww_mm"] = float(qw[1] - qw[0]) * 1000
    r["pe_drop"] = None
    if xp is not None:
        v = (x - xp.double()) / h
        okv = ok & torch.isfinite(v).all(-1)
        mv = m * okv
        r["ke"] = float(0.5 * (mv * torch.where(okv[:, None], v, 0).pow(2).sum(-1)).sum())
        vt = v - (v @ NR)[:, None] * NR
        sp = vt.norm(dim=-1)
        lay = []
        for lo, hi in LAY:
            sel = okv & (hh >= lo) & (hh < hi)
            lay.append(float((m[sel] * sp[sel]).sum() / m[sel].sum()) if int(sel.sum()) else None)
        r["vt_layers"] = lay
        r["vn_mean"] = float((mv * torch.where(okv, v @ NR, 0)).sum() / mv.sum())
    if F is not None:
        r["el"] = elastic(F)
    return r


def series(X, Fs=None, off=0):
    """X[i] = 프레임 i + off 위치 (off 0 = 정답 0 프레임부터)."""
    out = {}
    for i in range(X.shape[0]):
        f = i + off
        xp = X[i - 1] if i > 0 else (d["x"][0] if off == 1 else None)
        out[f] = frame(X[i], xp, None if Fs is None else Fs[i])
    c0 = out[0]["com"] if 0 in out else None
    return out


GTF = d.get("F")
gt = series(d["x"][:51], None if GTF is None else GTF[:51])
c0 = gt[0]["com"]
for f, r in gt.items():
    r["pe_drop"] = M * G * (c0 - r["com"]) / 1000
    r["rest"] = r["pe_drop"] - r.get("ke", 0.0) - r.get("el", 0.0) if "ke" in r else None
RES = {"gt": gt, "runs": {}, "def": __doc__}
for rd in sys.argv[3:]:
    p = os.path.join(rd, "pred.pt")
    if not os.path.isfile(p):
        continue
    P = torch.load(p, map_location="cpu", weights_only=False)
    X = P["pred"]
    s = series(X, P.get("F"), off=1)
    s[0] = gt[0]
    for f, r in s.items():
        r["pe_drop"] = M * G * (c0 - r["com"]) / 1000
        r["rest"] = r["pe_drop"] - r.get("ke", 0.0) - r.get("el", 0.0) if "ke" in r else None
    dv = {}
    for th in (5, 10, 20):
        dv[th] = next((f for f in sorted(s) if f in gt and abs(s[f]["com"] - gt[f]["com"]) > th), None)
    RES["runs"][os.path.basename(rd.rstrip("/"))] = dict(series=s, diverge=dv, part=bool(P.get("part", False)))
    f50 = s.get(50, s[max(s)])
    print(f"[wolfh1_an] {rd}: f50 질량중심 {f50['com']:.1f} mm (정답 {gt[50]['com']:.1f}), 갈라짐 5/10/20 mm = f{dv[5]}/f{dv[10]}/f{dv[20]}", flush=True)
json.dump(RES, open(OUT, "w"), ensure_ascii=False)
