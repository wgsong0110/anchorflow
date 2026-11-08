"""절 CM59 첫 접촉 소성·손상 경로 진단 (읽기만): 정답 대 우리 판, f3 높이 5 cm 층별 + 입자별.
python exe/wmcrack1_prof.py <결과 디렉토리> <gt_jp 디렉토리> <wm.pt> <GF h5 디렉토리> <판 이름,...> > 표.md   (+ <결과>/wmcrack1_prof.json)

정답 = 위치 wm.pt x[t] (exe/wmshape3_prof.py 와 같음) · 탄성 F = GF h5 f_tensor (exe/wmshape2_gt.py 와 같음) · logJp = 다시 돌린 정답 gt_jp (절 CM13.4).
우리 = <판>/wsh/xs_<t>.npz (위치·logJp float16) + <판>/wsh/fs_<t>.npz (탄성 F 특이값·det, exe/wmcrack1_wrap.py --wc1sv).
재질 식 = GF NACC (doc_nacc_compare.md 2 절, 우리 pl_rm 과 글자 같음): κ = 2μ/3 + λ, p0 = κ (1e-5 + sinh(ξ max(−logJp, 0))), 압력 p = −κ/2 (J² − 1) (NHB 키르히호프, 압축 +),
  인장 꼭짓점 Je_t = √(1 + 2βp0/κ), 압축 꼭짓점 Je_c = √(1 − 2p0/κ) (p0 < κ/2 일 때만).
층 칸 (프레임 t, 앞 저장 프레임 t′ 대비 증분): hr = 층 높이 비 중앙 (h_t/h_3), Δlj = logJp 증분 평균 (t − t′), 다져짐 % (logJp < −0.2), 손상 % (logJp > 0),
  J = 탄성 J 중앙, JJp = 전체 부피 비 J·exp(logJp + 0.04) 중앙, p0 = 응집력 중앙 (Pa), p/p0 중앙,
  압축 소성 % = 이 증분에서 logJp 가 1e-3 넘게 준 몫 (압축 꼭짓점·타원 압축 쪽), 인장 연화 % = 1e-3 넘게 는 몫 (인장 꼭짓점·타원 인장 쪽),
  인장 꼭짓점 상태 % = 끝 상태 J ≥ Je_t (1 − 1e-3) 인 몫 (꼭짓점 위 = RM 이 인장 꼭짓점으로 투영한 입자; 정답은 입자 RM 결과 그대로, 우리는 가우스 점 F 보간이라 근사).
입자별: 정답 손상 집합 (f5·f6·f8) 의 입자가 우리 판에서 어떤 상태인지 (logJp·J·p/p0 중앙, 다져짐·손상 몫), 우리 손상 집합의 정답 상태, IoU.
"""
import json
import os
import sys

import h5py
import numpy as np
import torch

O, JP, WM, H5, NS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], [q for q in sys.argv[5].split(",") if q]
FL = 0.02
TS = (3, 4, 5, 6, 8, 10)
EDG = np.arange(0.0, 0.80, 0.05)
D = torch.load(WM, map_location="cpu", weights_only=False)
GX = D["x"]
cfg = D["cfg"]
E_ = float(cfg.get("E", 2e3)); NU = float(cfg.get("nu", 0.38))
MU = E_ / (2 * (1 + NU)); LAM = E_ * NU / ((1 + NU) * (1 - 2 * NU)); KAP = 2 * MU / 3 + LAM
XI = float(cfg.get("xi", 3.0)); BETA = float(cfg.get("beta", 1.0))


def p0_of(lj):
    return KAP * (1e-5 + np.sinh(XI * np.maximum(-lj, 0.0)))


def ld_gt(t):
    with h5py.File(os.path.join(H5, "sim_%010d.h5" % t), "r") as h:
        F = np.ascontiguousarray(h["f_tensor"][:].T).reshape(-1, 3, 3).astype(np.float64)
    ok = np.isfinite(F).reshape(len(F), -1).all(1)
    Fz = np.where(ok[:, None, None], F, np.eye(3)[None])
    J = np.where(ok, np.linalg.det(Fz), np.nan)
    return GX[t].numpy().astype(np.float64), np.load(os.path.join(JP, "sc_%04d.npy" % t)).astype(np.float64), J


def ld_run(n, t):
    p, q = os.path.join(O, n, "wsh", f"xs_{t:03d}.npz"), os.path.join(O, n, "wsh", f"fs_{t:03d}.npz")
    if not (os.path.exists(p) and os.path.exists(q)):
        return None
    z, f = np.load(p), np.load(q)
    return z["x"].astype(np.float64), z["lj"].astype(np.float64), f["det"].astype(np.float64)


def stats(m, x, l, J, lp, h3, x3):
    if m.sum() < 50:
        return None
    h = x[m, 2] - FL
    p0 = p0_of(l[m])
    p = -0.5 * KAP * (J[m] ** 2 - 1.0)
    jt = np.sqrt(1.0 + 2.0 * BETA * p0 / KAP)
    dl = l[m] - lp[m]
    return dict(n=int(m.sum()), hr=float(np.median(h / h3[m])), dl=float(dl.mean()), cp=float((l[m] < -0.2).mean()), dm=float((l[m] > 0).mean()),
                J=float(np.median(J[m])), JJp=float(np.median(J[m] * np.exp(l[m] + 0.04))), p0=float(np.median(p0)), pp0=float(np.median(p / p0)),
                cpl=float((dl < -1e-3).mean()), tsf=float((dl > 1e-3).mean()), tv=float((J[m] >= jt * (1 - 1e-3)).mean()))


def prof(get):
    z3 = get(3)
    if z3 is None:
        return None
    x3, l3, J3 = z3
    h3 = x3[:, 2] - FL
    b = np.digitize(h3, EDG) - 1
    out, prev = {}, (l3, 3)
    st = {3: z3}
    for t in TS[1:]:
        z = get(t)
        if z is None:
            continue
        st[t] = z
        x, l, J = z
        ok = np.isfinite(x).all(1) & np.isfinite(l) & np.isfinite(J) & np.isfinite(prev[0]) & (h3 > 0.01)
        rows = [stats(ok & (b == k), x, l, J, prev[0], h3, x3) for k in range(len(EDG) - 1)]
        out[t] = dict(prev=prev[1], rows=rows, all=stats(ok, x, l, J, prev[0], h3, x3))
        prev = (l, t)
    return out, st


R, ST = {}, {}
for nm, get in [("정답", ld_gt)] + [(n, (lambda t, n=n: ld_run(n, t))) for n in NS]:
    r = prof(get)
    if r is not None:
        R[nm], ST[nm] = r

KEYS = ("hr", "dl", "cp", "dm", "J", "JJp", "p0", "pp0", "cpl", "tsf", "tv")
FMT = dict(hr="{:.3f}", dl="{:+.3f}", cp="{:.0%}", dm="{:.0%}", J="{:.3f}", JJp="{:.3f}", p0="{:.0f}", pp0="{:+.2f}", cpl="{:.0%}", tsf="{:.0%}", tv="{:.0%}")


def cell(q, ks):
    return "-" if q is None else " · ".join(FMT[k].format(q[k]) for k in ks)


print("# 절 CM59 첫 접촉 소성·손상 경로 -- 층별 (f3 바닥 위 높이 5 cm 층) · 입자별, 정답 대 우리 판\n")
print(__doc__.split("\n", 1)[1].strip() + "\n")
print(f"재질 상수: μ {MU:.3f} · λ {LAM:.3f} · κ {KAP:.3f} Pa · ξ {XI} · β {BETA} · 처음 logJp −0.04 → p0 {p0_of(np.array([-0.04]))[0]:.1f} Pa, 인장 꼭짓점 Je_t {np.sqrt(1 + 2 * BETA * p0_of(np.array([-0.04]))[0] / KAP):.4f}\n")
for t in TS[1:]:
    nms = [k for k in R if t in R[k]]
    if not nms:
        continue
    print(f"## f{t} (증분 기준 f{R[nms[0]][t]['prev']})\n")
    print("### (가) 층 높이 비 · Δ logJp · 다져짐 · 손상\n")
    print("| 층 cm | " + " | ".join(nms) + " |")
    print("|---|" + "---|" * len(nms))
    for k in range(len(EDG) - 1):
        print(f"| {100 * EDG[k]:.0f}~{100 * EDG[k + 1]:.0f} | " + " | ".join(cell(R[n][t]["rows"][k], ("hr", "dl", "cp", "dm")) for n in nms) + " |")
    print("| 몸 전체 | " + " | ".join(cell(R[n][t]["all"], ("hr", "dl", "cp", "dm")) for n in nms) + " |")
    print("\n### (나) 탄성 J · J·Jp · 응집력 p0 Pa · p/p0 · 압축 소성 % · 인장 연화 % · 인장 꼭짓점 상태 %\n")
    print("| 층 cm | " + " | ".join(nms) + " |")
    print("|---|" + "---|" * len(nms))
    for k in range(len(EDG) - 1):
        print(f"| {100 * EDG[k]:.0f}~{100 * EDG[k + 1]:.0f} | " + " | ".join(cell(R[n][t]["rows"][k], ("J", "JJp", "p0", "pp0", "cpl", "tsf", "tv")) for n in nms) + " |")
    print("| 몸 전체 | " + " | ".join(cell(R[n][t]["all"], ("J", "JJp", "p0", "pp0", "cpl", "tsf", "tv")) for n in nms) + " |")
    print()

# 입자별: 정답 손상 집합 · 우리 손상 집합
print("## 입자별 -- 정답 손상 입자 (그 프레임 logJp > 0) 가 우리 판에서 · 우리 손상 입자가 정답에서\n")
print("| 프레임 | 판 | 정답 손상 수 · 우리 손상 수 | IoU | 정답 손상 입자의 우리 logJp 중앙 · J 중앙 · p/p0 중앙 · 다져짐 % · 손상 % | 우리 손상 입자의 정답 logJp 중앙 · J 중앙 · 손상 % | 우리 손상 입자 바닥 위 높이 중앙 cm (정답 손상) |")
print("|---|---|---|---|---|---|---|")
PJ = {}
for t in (5, 6, 8, 10):
    if t not in ST["정답"]:
        continue
    gx, gl, gJ = ST["정답"][t]
    gd = gl > 0
    for n in R:
        if n == "정답" or t not in ST[n]:
            continue
        x, l, J = ST[n][t]
        ok = np.isfinite(l) & np.isfinite(gl) & np.isfinite(J) & np.isfinite(gJ)
        od = (l > 0) & ok
        g_ = gd & ok
        un = (od | g_).sum()
        iou = float((od & g_).sum() / un) if un else float("nan")
        p0 = p0_of(l); pp = -0.5 * KAP * (J ** 2 - 1.0) / p0

        def md(v, m):
            return float(np.median(v[m])) if m.sum() else float("nan")
        r = dict(ng=int(g_.sum()), no=int(od.sum()), iou=iou, o_l=md(l, g_), o_J=md(J, g_), o_pp=md(pp, g_), o_cp=float((l[g_] < -0.2).mean()) if g_.sum() else float("nan"),
                 o_dm=float((l[g_] > 0).mean()) if g_.sum() else float("nan"), g_l=md(gl, od), g_J=md(gJ, od), g_dm=float((gl[od] > 0).mean()) if od.sum() else float("nan"),
                 h_o=100 * md(x[:, 2] - FL, od), h_g=100 * md(gx[:, 2] - FL, g_))
        PJ.setdefault(n, {})[t] = r
        print(f"| f{t} | {n} | {r['ng']} · {r['no']} | {r['iou']:.3f} | {r['o_l']:+.3f} · {r['o_J']:.3f} · {r['o_pp']:+.2f} · {r['o_cp']:.0%} · {r['o_dm']:.0%} | "
              f"{r['g_l']:+.3f} · {r['g_J']:.3f} · {r['g_dm']:.0%} | {r['h_o']:.1f} ({r['h_g']:.1f}) |")
json.dump(dict(layers={k: {str(t): v for t, v in P.items()} for k, P in R.items()}, particles=PJ), open(os.path.join(O, "wmcrack1_prof.json"), "w"))
