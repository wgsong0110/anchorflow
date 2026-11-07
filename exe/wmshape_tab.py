"""절 CM40 묶음 wmshape 표: python exe/wmshape_tab.py <결과 디렉토리> <gt_jp 디렉토리> -> markdown (stdout) + <결과>/wmshape_curves.json.
판 = exe/wmshape_spec.txt 의 방식 r (진단 --wsh_rec) 수박 판, 묶음 = 이름의 끝 _숫자 뺀 것 (비결정 반복). 정답 = <결과>/gt/gt_wmshape.json (exe/wmshape_gt.py).
지표: 손상 = logJp > 0 몫 (정답 끝 6.51 %), 생긴 E = 전 구간 최대 (운동 + 중력 위치 + SNH 탄성) − 처음 7.13 J, 시간 = 첫 2 프레임 뺀 재생 GPU ms (같이 도는 판 있음; 단독은 z 판).
높이 H = 바닥 (z 0.02) 위 99.5 % 분위, 퍼짐 r95 = 처음 질량중심 축에서 수평 거리 95 % 분위, 바닥 r95 = 바닥 위 3 cm 안 입자의 r95, 모양 오차 = f4~20 평균 |H − H_정답| · |r95 − r95_정답| (m)."""
import glob
import itertools
import json
import math
import os
import re
import sys

import numpy as np

O, JP = sys.argv[1], sys.argv[2]
E0 = 7.13
FR = [3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100]
spec = [l.split(None, 4) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wmshape_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [s + [""] * (5 - len(s)) for s in spec]
runs = [(s[1], s[4]) for s in spec if s[3] in ("r", "b", "e") and os.path.exists(os.path.join(O, s[1], "fem.jsonl"))]
grp = {}
for n, fl in runs:
    grp.setdefault(re.sub(r"_\d+$", "", n), []).append(n)
gt = json.load(open(os.path.join(O, "gt", "gt_wmshape.json")))
G = {r["t"]: r for r in gt["frames"]}
REF = gt["ref"]


def rd(n):
    return [json.loads(l) for l in open(os.path.join(O, n, "fem.jsonl")) if l.strip()]


def summ(n):
    try:
        return json.load(open(os.path.join(O, n, "summary.json")))
    except Exception:
        return None


def f_(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def rng(v, d=3, pct=False):
    v = [q for q in v if q is not None and math.isfinite(q)]
    if not v:
        return "-"
    k = 100.0 if pct else 1.0
    return f"{k * np.mean(v):.{d}f} [{k * min(v):.{d}f}~{k * max(v):.{d}f}]" if len(v) > 1 else f"{k * v[0]:.{d}f}"


W = {}
for n, _ in runs:
    r = rd(n)
    W[n] = dict(rows=r, by={q["t"] + 1: q["wsh"] for q in r if q.get("wsh")}, s=summ(n))


def res(n):
    s = W[n]["s"]
    if s is None:
        return "도는 중/죽음"
    return ("발산 f%d" % s["diverged"]) if s.get("diverged") else "완주"


def dE(n):
    Ev = [q["en"]["KE"] + q["en"]["PE"] + q["en"]["Eel"] for q in W[n]["rows"] if q.get("en") and q["en"].get("Eel") is not None]
    Ev = [e for e in Ev if math.isfinite(e)]
    return (max(Ev) - E0) if Ev else None


def ms(n):
    m = [x["ms"] for x in W[n]["rows"] if x["t"] >= 2 and x.get("ms") is not None and math.isfinite(x["ms"])]
    return (float(np.mean(m)), float(np.max(m))) if m else (None, None)


def val(n, t, k, sub=None):
    q = W[n]["by"].get(t)
    if q is None:
        return None
    v = q.get(k)
    if sub is not None and v is not None:
        v = v[sub] if not isinstance(sub, tuple) else v[sub[0]][sub[1]]
    return v


def gval(t, k, sub=None):
    v = G[t].get(k)
    if sub is not None and v is not None:
        v = v[sub] if not isinstance(sub, tuple) else v[sub[0]][sub[1]]
    return v


def shape_err(n):
    eh, er = [], []
    for t in range(4, 21):
        if t in W[n]["by"]:
            eh.append(abs(val(n, t, "H") - G[t]["H"])); er.append(abs(val(n, t, "r95") - G[t]["r95"]))
    return (float(np.mean(eh)) if eh else None, float(np.mean(er)) if er else None)


def load_sv(n, t):
    p = os.path.join(O, n, "wsh", f"xs_{t:03d}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["x"].astype(np.float32), z["lj"].astype(np.float32)


GTX = None


def gt_x(t):
    global GTX
    if GTX is None:
        import torch
        GTX = torch.load("/tmp/dkta/ag_data/wm.pt", map_location="cpu", weights_only=False)["x"]
    return GTX[t].numpy()


def gt_lj(t):
    return np.load(os.path.join(JP, "sc_%04d.npy" % t)).astype(np.float32)


print("# 절 CM40 묶음 wmshape -- 수박 깨짐 모양 (정답 | D | 후보)\n")
print(__doc__.split("\n", 1)[1].strip() + "\n")
print(f"처음: H0 {REF['H0']:.3f} m, R0 {REF['R0']:.3f} m, 질량중심 축 {REF['c0']}\n")
print("## 1. 요약 (묶음마다 비결정 판들)\n")
print("| 묶음 | 판 | 결과 | 노드 상자 최대 (백만) | 끝 손상 % | f10 · f20 손상 % | 모양 오차 높이 · 퍼짐 m (f4~20) | f10 H · r95 m | 생긴 E J | 오차 % | ms 평균 · 최대 (같이 돎) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
print(f"| 정답 | - | - | - | {100 * G[100]['d0']:.2f} | {100 * G[10]['d0']:.2f} · {100 * G[20]['d0']:.2f} | 0 · 0 | {G[10]['H']:.3f} · {G[10]['r95']:.3f} | (+16.7 첫 접촉) | - | - |")
SUM = {}
for g, ns in grp.items():
    last = []
    for n in ns:
        by = W[n]["by"]
        lj_ = [q["en"]["lj0"] for q in W[n]["rows"] if (q.get("en") or {}).get("lj0") is not None]
        last.append(by[max(by)]["d0"] if by else (lj_[-1] if lj_ else None))
    se = [shape_err(n) for n in ns]
    s_ = [W[n]["s"] for n in ns]
    er = [100 * s["err_mean_done"] for s in s_ if s]
    mm = [ms(n) for n in ns]
    SUM[g] = dict(last=last, se=se)
    nnm = [max([q.get("nn") or 0 for q in W[n]["rows"]] + [0]) / 1e6 for n in ns]
    print(f"| {g} | {len(ns)} | {', '.join(res(n) for n in ns)} | {', '.join(f'{q:.1f}' for q in nnm)} | {rng(last, 2, True)} | {rng([val(n, 10, 'd0') for n in ns], 2, True)} · {rng([val(n, 20, 'd0') for n in ns], 2, True)} | "
          f"{rng([q[0] for q in se], 3)} · {rng([q[1] for q in se], 3)} | {rng([val(n, 10, 'H') for n in ns])} · {rng([val(n, 10, 'r95') for n in ns])} | {rng([dE(n) for n in ns], 2)} | {rng(er, 2)} | "
          f"{rng([q[0] for q in mm], 1)} · {rng([q[1] for q in mm], 1)} |")

for title, k, sub, d, pct in (("높이 H (m)", "H", None, 3, False), ("퍼짐 r95 (m)", "r95", None, 3, False), ("바닥 r95 (바닥 위 3 cm 입자, m)", "foot95", None, 3, False),
                              ("질량중심 z (m)", "zc", None, 3, False), ("손상 몫 % (logJp > 0)", "d0", None, 2, True), ("다져짐 몫 % (logJp < −0.2)", "c02", None, 2, True),
                              ("탄성 J 1 % 분위", "Jq", 0, 3, False), ("전체 부피 비 J·Jp 1 % 분위", "Jtq", 0, 3, False), ("전체 부피 비 J·Jp 99 % 분위", "Jtq", 4, 3, False)):
    print(f"\n## 곡선: {title}\n")
    print("| 묶음 | " + " | ".join(f"f{t}" for t in FR) + " |")
    print("|---|" + "---|" * len(FR))
    print("| 정답 | " + " | ".join((f_((100 if pct else 1) * gval(t, k, sub), d) if gval(t, k, sub) is not None else "-") for t in FR) + " |")
    for g, ns in grp.items():
        print(f"| {g} | " + " | ".join(rng([val(n, t, k, sub) for n in ns], d, pct) for t in FR) + " |")

print("\n## 2. 손상 위치 (손상 입자 중 몫 %; 높이 = 바닥 위 cm 구간 [0, 13, 27, 53, 80, 106, ∞] (= 처음 꼭대기 높이 H0 1.33 m 의 0.1·0.2·0.4·0.6·0.8 배), 반경 = 축 거리 / R0 구간 [0, 0.4, 0.7, 0.9, 1.1, ∞])\n")
print("| 묶음 | 프레임 | 손상 높이 분포 | 손상 반경 분포 | 새 손상 높이 분포 | 새 손상 반경 분포 | 새 손상 입자 |")
print("|---|---|---|---|---|---|---|")
P = lambda v: "-" if v is None else "/".join(f"{100 * q:.0f}" for q in v)
for t in (4, 5, 6, 8, 10, 20, 50):
    q = G[t]
    print(f"| 정답 | f{t} | {P(q.get('dh'))} | {P(q.get('dr'))} | {P(q.get('nh'))} | {P(q.get('nr'))} | {q.get('nnew')} |")
    for g, ns in grp.items():
        n = ns[0]
        print(f"| {g} ({n}) | f{t} | {P(val(n, t, 'dh'))} | {P(val(n, t, 'dr'))} | {P(val(n, t, 'nh'))} | {P(val(n, t, 'nr'))} | {val(n, t, 'nnew')} |")

print("\n## 3. 응력 (탄성 F_e 에서, Pa; 손상 d = logJp > 0 · 안 손상 u; 중앙값 [10~90 %]; 정답 재질 식 NHB 로 양쪽, 우리 풀이 식 SNH 는 괄호)\n")
print("| 묶음 | 프레임 | 전단 q d | 전단 q u | 압력 p d | 압력 p u | (SNH q d · u) | 응집력 p0 중앙 d · u |")
print("|---|---|---|---|---|---|---|---|")
Q3 = lambda v: "-" if not v or v[1] is None else f"{v[1]:.1f} [{v[0]:.1f}~{v[2]:.1f}]"
for t in (4, 5, 6, 10, 20, 50):
    q = G[t]
    print(f"| 정답 | f{t} | {Q3(q['q_nhb']['d'])} | {Q3(q['q_nhb']['u'])} | {Q3(q['p_nhb']['d'])} | {Q3(q['p_nhb']['u'])} | - | {f_(q['p0']['d'][0], 1)} · {f_(q['p0']['u'][0], 1)} |")
    for g, ns in grp.items():
        n = ns[0]
        w = W[n]["by"].get(t)
        if w is None or "q_nhb" not in w:
            continue
        print(f"| {g} ({n}) | f{t} | {Q3(w['q_nhb']['d'])} | {Q3(w['q_nhb']['u'])} | {Q3(w['p_nhb']['d'])} | {Q3(w['p_nhb']['u'])} | {f_(w['q_snh']['d'][1], 1)} · {f_(w['q_snh']['u'][1], 1)} | "
              f"{f_(w['p0']['d'][0], 1)} · {f_(w['p0']['u'][0], 1)} |")

print("\n## 4. 가우스 점 수준 (풀이가 힘을 계산하는 상태; 묶음 첫 판, 프레임 끝 커밋 뒤)\n")
print("dil = 프레임 시작 손상 입자 중 그 가우스 점 평균 logJp ≤ 0 (힘에 연화 안 보임) 몫, p0 비 = 손상 입자 품은 점의 '평균 logJp 응집력 / 입자 응집력 평균' 중앙 [10~90 %], 손상 몫 = 그 점 손상 입자 부피 몫 중앙,\n"
      "손상 점 부피 % = 끝 (커밋 뒤) Sq > 0 점 부피 몫, 시작 = 이 프레임 모으기 직후 (앞 프레임 끝 → 보간 교체 → 다시 모으기), RM ↑ = 커밋에서 logJp 가 는 점 몫, q·J = SNH 전단 응력·J (손상 점 d · 안 손상 u, 중앙).\n")
print("| 묶음 | 프레임 | dil | p0 비 | 손상 몫 | 손상 점 부피 % 앞 끝 → 시작 → 끝 | RM ↑ · ↓ % | Smax 시작 · 끝 | q d · u Pa | J d · u |")
print("|---|---|---|---|---|---|---|---|---|---|")
for g, ns in grp.items():
    n = ns[0]
    for t in (4, 5, 6, 8, 10, 15, 20, 30):
        w = W[n]["by"].get(t, {}).get("g")
        if not w or "err" in w:
            continue
        pr = w.get("p0r") or [None] * 3
        print(f"| {g} ({n}) | f{t} | {f_(w['dil'], 2)} | {f_(pr[1], 2)} [{f_(pr[0], 2)}~{f_(pr[2], 2)}] | {f_(w['fd'], 2)} | {f_(100 * w['Vde_prev'] if w.get('Vde_prev') is not None else None, 2)} → {f_(100 * w['Vd0'], 2)} → {f_(100 * w['Vde'], 2)} | "
              f"{f_(100 * w['rm_up'], 2)} · {f_(100 * w['rm_dn'], 2)} | {f_(w['Smax0'], 2)} · {f_(w['Smaxe'], 2)} | {f_(w['qd'][1], 1)} · {f_(w['qu'][1], 1)} | {f_(w['Jd'][1], 3)} · {f_(w['Ju'][1], 3)} |")

print("\n## 5. 손상 국소성 (손상 입자를 크기 s 상자로 묶을 때: 상자 평균 logJp > 0 몫 · 상자 손상 몫 ≥ 0.5 몫; s = 정답 dx 6.7 mm · 셀 3 가우스 점 10 mm · 셀 3 칸 20 mm)\n")
print("| 묶음 | 프레임 | 6.7 mm | 10 mm | 20 mm |")
print("|---|---|---|---|---|")
V2 = lambda v: "-" if not v or v[0] is None else f"{100 * v[0]:.0f} · {100 * v[1]:.0f}"
for t in (5, 10, 20, 50, 100):
    print(f"| 정답 | f{t} | " + " | ".join(V2(v) for v in G[t]["vox"]) + " |")
    for g, ns in grp.items():
        vv = val(ns[0], t, "vox")
        if vv:
            print(f"| {g} ({ns[0]}) | f{t} | " + " | ".join(V2(v) for v in vv) + " |")

print("\n## 6. 정답과 손상 겹침 · 비결정 흩어짐 · 바닥 미끄럼 (저장 프레임)\n")
SV = [4, 5, 6, 8, 10, 15, 20, 30, 50, 100]
print("겹침 IoU = 같은 입자 번호에서 손상 (logJp > 0) 집합 교집합 / 합집합. 흩어짐 = 같은 묶음 판 쌍의 입자 위치 평균 차 (mm) · 손상 상태가 다른 입자 몫 %. "
      "바닥 미끄럼 = f3 에 바닥 위 3 cm 안이던 입자의 f3 → 그 프레임 바깥쪽 수평 이동 중앙 (mm).\n")
print("| 묶음 | 지표 | " + " | ".join(f"f{t}" for t in SV) + " |")
print("|---|---|" + "---|" * len(SV))
CUR = dict(gt={str(t): {k: G[t].get(k) for k in ("H", "r95", "foot95", "zc", "d0")} for t in G})
g3 = gt_x(3)
c0 = np.array(REF["c0"])
bot = (g3[:, 2] - 0.02) < 0.03
r3 = g3[:, :2] - c0
u3 = r3 / np.maximum(np.linalg.norm(r3, axis=1, keepdims=True), 1e-9)


def slip(x, ref3=None):
    ref3 = g3 if ref3 is None else ref3
    dxy = x[:, :2] - ref3[:, :2]
    return float(np.nanmedian((dxy * u3)[bot].sum(1))) * 1e3


print("| 정답 | 바닥 미끄럼 mm | " + " | ".join(f_(slip(gt_x(t)), 1) for t in SV) + " |")
for g, ns in grp.items():
    iou, dv, dd, sl = [], [], [], []
    for t in SV:
        gl = gt_lj(t) > 0
        a, b, c_, s_ = [], [], [], []
        L = {}
        for n in ns:
            z = load_sv(n, t)
            if z is None:
                continue
            L[n] = z
            dm = z[1] > 0
            un = (dm | gl).sum()
            a.append(float((dm & gl).sum() / un) if un else None)
            z3 = load_sv(n, 3)
            s_.append(slip(z[0], z3[0] if z3 is not None else None))
        for n1, n2 in itertools.combinations(sorted(L), 2):
            x1, x2 = L[n1][0], L[n2][0]
            ok = np.isfinite(x1).all(1) & np.isfinite(x2).all(1)
            b.append(float(np.linalg.norm(x1[ok] - x2[ok], axis=1).mean()) * 1e3)
            c_.append(float(((L[n1][1] > 0) != (L[n2][1] > 0))[ok].mean()))
        iou.append(rng(a, 2)); dv.append(rng(b, 2)); dd.append(rng(c_, 2, True)); sl.append(rng(s_, 1))
    print(f"| {g} | 정답 겹침 IoU | " + " | ".join(iou) + " |")
    print(f"| {g} | 판 쌍 위치 차 mm | " + " | ".join(dv) + " |")
    print(f"| {g} | 판 쌍 손상 상태 다름 % | " + " | ".join(dd) + " |")
    print(f"| {g} | 바닥 미끄럼 mm | " + " | ".join(sl) + " |")
    CUR[g] = {n: {str(t): {k: W[n]["by"][t].get(k) for k in ("H", "r95", "foot95", "zc", "d0")} for t in W[n]["by"]} for n in ns}

print("\n## 7. 비결정 흩어짐이 커지는 프레임 (묶음 판들의 프레임별 손상 몫·높이 범위, 후보 켠 프레임 = 무너진 프레임 다시 하기)\n")
print("| 묶음 | 판 | 후보 켠 프레임 (X3c) | 손상 몫 처음 0.5 % 넘는 프레임 | f6 · f10 · f20 · f50 손상 % |")
print("|---|---|---|---|---|")
for g, ns in grp.items():
    for n in ns:
        on = [x["t"] + 1 for x in W[n]["rows"] if x.get("ks") and any(x["ks"].get("on") or [])]
        by = W[n]["by"]
        f05 = next((t for t in sorted(by) if by[t]["d0"] > 0.005), None)
        print(f"| {g} | {n} | {on[:16]} | {f05} | " + " · ".join(f_(100 * by[t]["d0"], 2) if t in by else "-" for t in (6, 10, 20, 50)) + " |")
json.dump(CUR, open(os.path.join(O, "wmshape_curves.json"), "w"))
