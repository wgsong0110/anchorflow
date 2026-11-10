"""절 CM97 수박 첫 충돌 원인 가르기 표 (읽기만; exe/wmcrack1_prof.py 의 정의 그대로 + 압력 시간·위 끝 인장·에너지).
python exe/wmwhy_prof.py <결과 디렉토리> <gt_jp> <wm.pt> <GF h5 디렉토리> <판,...> > 표.md   (+ <결과>/wmwhy_prof.json)

정답 = 위치 wm.pt x[t] · 속도 wm.pt v[t] · 탄성 F = GF h5 f_tensor · logJp = 다시 돌린 정답 gt_jp (절 CM13.4).
우리 = <판>/wsh/xs_<t>.npz (위치·logJp f16) + fs_<t>.npz (탄성 F 특이값·det) + vs_<t>.npz (속도) (exe/wmwhy_wrap.py).
층 = f3 바닥 (z 0.02) 위 높이. 재질 식 = GF NACC·NHB: κ = 2μ/3 + λ, p0 = κ (1e-5 + sinh(ξ max(−logJp, 0))), 압력 p = −κ/2 (J² − 1) (압축 +).
에너지 (질량 = deform_model.traj_mass, 부피 = 질량 / 밀도, 양쪽 같은 입자 질량): 운동 ½ m |v|², 탄성 = NHB ψ(σ) V0 (양쪽 같은 식; 우리 SNH 판은 SNH ψ 도),
  중력 = −m g·x (f3 기준 차), 합 = 운동 + 탄성 NHB + 중력 차.
"""
import json
import os
import sys

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
from anchorflow import deform_model as DM  # noqa: E402

O, JP, WM, H5, NS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], [q for q in sys.argv[5].split(",") if q]
FL = 0.02
TS = (3, 4, 5, 6, 8)
EDG = np.arange(0.0, 0.80, 0.05)
PED = np.arange(0.0, 0.85, 0.10)                       # 압력 시간 표 층 (10 cm)
D = torch.load(WM, map_location="cpu", weights_only=False)
GX, GV = D["x"], D["v"]
cfg = D["cfg"]
E_ = float(cfg.get("E", 2e3)); NU = float(cfg.get("nu", 0.38))
MU = E_ / (2 * (1 + NU)); LAM = E_ * NU / ((1 + NU) * (1 - 2 * NU)); KAP = 2 * MU / 3 + LAM
XI = float(cfg.get("xi", 3.0)); BETA = float(cfg.get("beta", 1.0))
G = np.array([float(q) for q in cfg["g"]])
M = DM.traj_mass(D, "cpu").double().numpy()
V0 = M / float(cfg["density"])
H = float(cfg["frame_dt"])
print(f"[prof] 입자 {len(M)} 질량 합 {M.sum():.4f} 밀도 {cfg['density']} g {G.tolist()} 프레임 {H}", file=sys.stderr)


def p0_of(lj):
    return KAP * (1e-5 + np.sinh(XI * np.maximum(-lj, 0.0)))


def psi_nhb(sv):
    J = sv.prod(1)
    Jc = np.where(J > 0, J, np.nan)
    return 0.5 * MU * (Jc ** (-2.0 / 3.0) * (sv ** 2).sum(1) - 3.0) + 0.5 * KAP * (0.5 * (Jc ** 2 - 1.0) - np.log(Jc))


def psi_snh(sv):
    J = sv.prod(1)
    return 0.5 * MU * ((sv ** 2).sum(1) - 3.0) - MU * (J - 1.0) + 0.5 * (LAM + MU) * (J - 1.0) ** 2


def ld_gt(t):
    with h5py.File(os.path.join(H5, "sim_%010d.h5" % t), "r") as h:
        F = np.ascontiguousarray(h["f_tensor"][:].T).reshape(-1, 3, 3).astype(np.float64)
    ok = np.isfinite(F).reshape(len(F), -1).all(1)
    Fz = np.where(ok[:, None, None], F, np.eye(3)[None])
    sv = np.linalg.svd(Fz, compute_uv=False)
    sv[~ok] = np.nan
    J = np.where(ok, np.linalg.det(Fz), np.nan)
    return dict(x=GX[t].numpy().astype(np.float64), l=np.load(os.path.join(JP, "sc_%04d.npy" % t)).astype(np.float64), J=J, sv=sv, v=GV[t].numpy().astype(np.float64))


def ld_run(n, t):
    p, q, r = (os.path.join(O, n, "wsh", f"{k}_{t:03d}.npz") for k in ("xs", "fs", "vs"))
    if not (os.path.exists(p) and os.path.exists(q)):
        return None
    z, f = np.load(p), np.load(q)
    v = np.load(r)["v"].astype(np.float64) if os.path.exists(r) else None
    return dict(x=z["x"].astype(np.float64), l=z["lj"].astype(np.float64), J=f["det"].astype(np.float64), sv=f["sv"].astype(np.float64), v=v)


def stats(m, z, lp, h3):
    if m.sum() < 50:
        return None
    x, l, J = z["x"], z["l"], z["J"]
    h = x[m, 2] - FL
    p0 = p0_of(l[m])
    p = -0.5 * KAP * (J[m] ** 2 - 1.0)
    jt = np.sqrt(1.0 + 2.0 * BETA * p0 / KAP)
    dl = l[m] - lp[m]
    return dict(n=int(m.sum()), hr=float(np.median(h / h3[m])), dl=float(dl.mean()), cp=float((l[m] < -0.2).mean()), dm=float((l[m] > 0).mean()),
                J=float(np.median(J[m])), JJp=float(np.median(J[m] * np.exp(l[m] + 0.04))), p0=float(np.median(p0)), p=float(np.mean(p)), pmed=float(np.median(p)),
                tens=float((p < 0).mean()), cpl=float((dl < -1e-3).mean()), tsf=float((dl > 1e-3).mean()), tv=float((J[m] >= jt * (1 - 1e-3)).mean()))


def energy(z, z3):
    ok = np.isfinite(z["x"]).all(1) & np.isfinite(z["sv"]).all(1)
    r = dict(ok=int(ok.sum()))
    if z.get("v") is not None:
        okv = ok & np.isfinite(z["v"]).all(1)
        r["ke"] = float(0.5 * (M[okv] * (z["v"][okv] ** 2).sum(1)).sum())
        r["vz"] = float((M[okv] * z["v"][okv, 2]).sum() / M[okv].sum())
    pn = psi_nhb(z["sv"])
    okp = ok & np.isfinite(pn)
    r["el"] = float((pn[okp] * V0[okp]).sum())
    r["el_snh"] = float((psi_snh(z["sv"][ok]) * V0[ok]).sum())
    r["grav"] = float(-(M[ok] * ((z["x"][ok] - z3["x"][ok]) @ G)).sum())
    r["zc"] = float((M[ok] * z["x"][ok, 2]).sum() / M[ok].sum())
    r["tot"] = r.get("ke", float("nan")) + r["el"] + r["grav"]
    return r


def prof(get):
    z3 = get(3)
    if z3 is None:
        return None
    h3 = z3["x"][:, 2] - FL
    b = np.digitize(h3, EDG) - 1
    bp = np.digitize(h3, PED) - 1
    out, prev, st = {}, (z3["l"], 3), {3: z3}
    en = {3: energy(z3, z3)}
    pt = {3: [stats((np.isfinite(z3["J"]) & (bp == k) & (h3 > 0.01)), z3, z3["l"], h3) for k in range(len(PED) - 1)]}
    for t in TS[1:]:
        z = get(t)
        if z is None:
            continue
        st[t] = z
        x, l, J = z["x"], z["l"], z["J"]
        ok = np.isfinite(x).all(1) & np.isfinite(l) & np.isfinite(J) & np.isfinite(prev[0]) & (h3 > 0.01)
        out[t] = dict(prev=prev[1], rows=[stats(ok & (b == k), z, prev[0], h3) for k in range(len(EDG) - 1)], all=stats(ok, z, prev[0], h3),
                      top=stats(ok & (h3 >= 0.65) & (h3 < 0.75), z, prev[0], h3))
        pt[t] = [stats(ok & (bp == k), z, z3["l"], h3) for k in range(len(PED) - 1)]
        en[t] = energy(z, z3)
        prev = (l, t)
    return out, st, en, pt


R, ST, EN, PT = {}, {}, {}, {}
for nm, get in [("정답", ld_gt)] + [(n, (lambda t, n=n: ld_run(n, t))) for n in NS]:
    try:
        r = prof(get)
    except Exception as e_:
        print(f"[prof] {nm} 실패 {e_!r}", file=sys.stderr)
        r = None
    if r is not None:
        R[nm], ST[nm], EN[nm], PT[nm] = r
    print(f"[prof] {nm} {'완료' if r else '없음'}", file=sys.stderr)


def iou(n, t):
    if t not in ST["정답"] or t not in ST.get(n, {}):
        return None
    gl, l = ST["정답"][t]["l"], ST[n][t]["l"]
    ok = np.isfinite(l) & np.isfinite(gl)
    a, b = (l > 0) & ok, (gl > 0) & ok
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def g(n, t, k, key, f="{:.3f}"):
    try:
        q = R[n][t][k] if k in ("all", "top") else R[n][t]["rows"][k]
        return "-" if q is None else f.format(q[key])
    except (KeyError, IndexError):
        return "-"


NM = list(R)
J = {}
print("# 절 CM97 수박 첫 충돌 원인 가르기 (f3~f8) -- 정답 대 판\n")
print(__doc__.split("\n", 1)[1].strip() + "\n")
print(f"재질 상수: μ {MU:.2f} · λ {LAM:.2f} · κ {KAP:.2f} Pa · ξ {XI} · β {BETA} · 처음 p0 {p0_of(np.array([-0.04]))[0]:.1f} Pa\n")
print("## A. 요약 (f4 바닥 0~5 cm 층 높이 비 · Δ logJp · 다져짐; f4 몸 압축 소성 %; f4 · f5 다져짐 %; f5 · f6 위 끝 65~75 cm 손상 %; 손상 f5 · f6 · f8 %; 정답 IoU f5 · f6 · f8)\n")
print("| 판 | f4 바닥 0~5 cm | f4 5~25 cm Δ logJp | f4 몸 압축 소성 % | 다져짐 f4 · f5 % | 위 끝 손상 f5 · f6 % | 손상 f5 · f6 · f8 % | IoU f5 · f6 · f8 |")
print("|---|---|---|---|---|---|---|---|")
for n in NM:
    a5 = " · ".join(g(n, t, "top", "dm", "{:.1%}") for t in (5, 6))
    dm = " · ".join(g(n, t, "all", "dm", "{:.2%}") for t in (5, 6, 8))
    io = " · ".join("-" if iou(n, t) is None else f"{iou(n, t):.3f}" for t in (5, 6, 8)) if n != "정답" else "1"
    mid = "-"
    try:
        rows = [R[n][4]["rows"][k] for k in (1, 2, 3, 4)]
        mid = " · ".join("-" if q is None else f"{q['dl']:+.3f}" for q in rows)
    except KeyError:
        pass
    print(f"| {n} | {g(n, 4, 0, 'hr')} · {g(n, 4, 0, 'dl', '{:+.3f}')} · {g(n, 4, 0, 'cp', '{:.0%}')} | {mid} | {g(n, 4, 'all', 'cpl', '{:.0%}')} | "
          f"{g(n, 4, 'all', 'cp', '{:.1%}')} · {g(n, 5, 'all', 'cp', '{:.1%}')} | {a5} | {dm} | {io} |")
    J.setdefault(n, {})["iou"] = {t: iou(n, t) for t in (5, 6, 8)} if n != "정답" else None

print("\n## B. f4 층별 (층 높이 비 · Δ logJp (f4 − f3) · 다져짐 % · 압축 소성 %)\n")
print("| 층 cm | " + " | ".join(NM) + " |")
print("|---|" + "---|" * len(NM))
for k in range(len(EDG) - 1):
    print(f"| {100 * EDG[k]:.0f}~{100 * EDG[k + 1]:.0f} | " + " | ".join(
        f"{g(n, 4, k, 'hr')} · {g(n, 4, k, 'dl', '{:+.3f}')} · {g(n, 4, k, 'cp', '{:.0%}')} · {g(n, 4, k, 'cpl', '{:.0%}')}" for n in NM) + " |")
print("| 몸 전체 | " + " | ".join(f"{g(n, 4, 'all', 'hr')} · {g(n, 4, 'all', 'dl', '{:+.3f}')} · {g(n, 4, 'all', 'cp', '{:.0%}')} · {g(n, 4, 'all', 'cpl', '{:.0%}')}" for n in NM) + " |")

print("\n## C. 압축파 -- 층별 평균 압력 p Pa (압축 +) 프레임 f3 · f4 · f5 · f6 · f8 (괄호: 인장 p < 0 입자 몫 f5)\n")
print("| 층 cm (f3 높이) | " + " | ".join(NM) + " |")
print("|---|" + "---|" * len(NM))
for k in range(len(PED) - 1):
    cs = []
    for n in NM:
        v = []
        for t in TS:
            q = PT[n].get(t, [None] * 9)[k] if t in PT[n] else None
            v.append("-" if q is None else f"{q['p']:.0f}")
        q5 = PT[n].get(5, [None] * 9)[k] if 5 in PT[n] else None
        cs.append(" · ".join(v) + ("" if q5 is None else f" ({q5['tens']:.0%})"))
    print(f"| {100 * PED[k]:.0f}~{100 * PED[k + 1]:.0f} | " + " | ".join(cs) + " |")

print("\n## D. 위 끝 65~75 cm (f5 · f6): 손상 % · 인장 꼭짓점 상태 % · 인장 연화 % · J·Jp 중앙 · p0 중앙\n")
print("| 판 | f5 | f6 |")
print("|---|---|---|")
for n in NM:
    print(f"| {n} | " + " | ".join(f"{g(n, t, 'top', 'dm', '{:.1%}')} · {g(n, t, 'top', 'tv', '{:.0%}')} · {g(n, t, 'top', 'tsf', '{:.0%}')} · {g(n, t, 'top', 'JJp')} · {g(n, t, 'top', 'p0', '{:.0f}')}" for t in (5, 6)) + " |")

print("\n## E. 에너지 J (운동 · 탄성 NHB · 중력 차 (f3 기준) · 합) 와 질량중심 vz m/s · z m\n")
print("| 판 | f3 운동 · vz · z | f4 운동 · 탄성 · 합 · vz · z | f5 운동 · 탄성 · 합 · vz · z | f3 → f4 합 변화 | f4 탄성 SNH |")
print("|---|---|---|---|---|---|")
for n in NM:
    e = EN[n]

    def fe(t, full=True):
        if t not in e:
            return "-"
        q = e[t]
        return (f"{q.get('ke', float('nan')):.2f} · " + (f"{q['el']:.2f} · {q['tot']:.2f} · " if full else "") + f"{q.get('vz', float('nan')):+.2f} · {q['zc']:.3f}")
    dE = f"{e[4]['tot'] - e[3]['tot']:+.2f}" if 4 in e else "-"
    print(f"| {n} | {fe(3, False)} | {fe(4)} | {fe(5)} | {dE} | {e[4]['el_snh']:.2f} |" if 4 in e else f"| {n} | {fe(3, False)} | - | - | - | - |")

json.dump(dict(layers={k: {str(t): v for t, v in P.items()} for k, P in R.items()}, ptime={k: {str(t): v for t, v in P.items()} for k, P in PT.items()},
               energy={k: {str(t): v for t, v in P.items()} for k, P in EN.items()}, iou={k: v["iou"] for k, v in J.items()}),
          open(os.path.join(O, "wmwhy_prof.json"), "w"))
