"""절 CM44 묶음 wmshape3 표: python exe/wmshape3_tab.py <결과 디렉토리> <gt_jp 디렉토리> -> markdown (stdout) + <결과>/wmshape3_curves.json.
판 = exe/wmshape3_spec.txt, 묶음 = 이름의 끝 _숫자 뺀 것 (비결정 반복). 뒤에 절 CM40 표 (exe/wmshape_tab.py, 같은 정의) 를 이 판 목록으로.
정답 = <결과>/gt/gt_wmshape.json (절 CM40). 손상 = logJp > 0 몫 (정답 끝 6.51 %), 생긴 E = 전 구간 최대 (운동 + 중력 위치 + SNH 탄성) − 처음 7.13 J,
IoU = 같은 입자 번호에서 손상 집합 교집합 / 합집합, 첫 손상 높이 = f5 손상 입자 중 바닥 위 27~53 cm 몫 (wsh dh 칸 2; 정답 57 %).
미룬 걸음 = fem_wmshape3 진단 ws3 (수렴 안 해 커밋에서 소성 상태를 확정 안 한 걸음 / 후보가 켜진 걸음, 다시 하기 시도 포함)."""
import contextlib
import io
import json
import math
import os
import runpy
import sys

import numpy as np

O, JP = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["WSH_SPEC"] = os.path.join(HERE, "wmshape3_spec.txt")
os.environ["WSH_CUR"] = "wmshape3_curves.json"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sys.argv = [os.path.join(HERE, "wmshape_tab.py"), O, JP]
    T = runpy.run_path(os.path.join(HERE, "wmshape_tab.py"), run_name="__main__")
base = buf.getvalue()
W, grp, G, rng, val, dE, load_sv, gt_lj = T["W"], T["grp"], T["G"], T["rng"], T["val"], T["dE"], T["load_sv"], T["gt_lj"]


def summ(n):
    try:
        return json.load(open(os.path.join(O, n, "summary.json")))
    except Exception:
        return None


def zms(g):
    o = []
    for r_ in ("a", "b"):
        n = f"z{g}_{r_}"
        f = os.path.join(O, n, "fem.jsonl")
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        m = [x["ms"] for x in rows if x["t"] >= 2 and x.get("ms") is not None and math.isfinite(x["ms"])]
        s = summ(n)
        o.append((r_, float(np.mean(m)) if m else None, float(np.max(m)) if m else None, (s or {}).get("diverged"), s is None))
    return o


def fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def iou(n, t):
    z = load_sv(n, t)
    if z is None:
        return None
    gl = gt_lj(t) > 0
    dm = z[1] > 0
    un = (dm | gl).sum()
    return float((dm & gl).sum() / un) if un else None


def ws3(n):
    a = b = 0
    for q in W[n]["rows"]:
        w = q.get("ws3")
        if w and w.get("on"):
            a += w["ok"]; b += w["defer"]
    return (b / (a + b)) if a + b else None


print("# 절 CM44 묶음 wmshape3 -- 수박 깨짐 모양 3 (늦은 손상 기어오름 막기 후보, BW4 위)\n")
print(__doc__.split("\n", 1)[1].strip() + "\n")
print("## A. 후보 요약 (비결정 판 + 단독 z a·b; 정답 손상 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51 %, 끝 r95 0.895 · 질량중심 0.166 m)\n")
print("| 묶음 | 판 | 발산 (판정 + 단독) | 끝 손상 % [범위] | f5 · f6 · f8 손상 % | 판 사이 표준편차 %p | f100 r95 · 질량중심 m | 정답 IoU f5 · f10 · f100 | f5 손상 27~53 cm 몫 % | 생긴 E J | 오차 % | 단독 ms 평균/최대 a · b | 미룬 걸음 몫 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
print(f"| 정답 | - | - | {100 * G[100]['d0']:.2f} | {100 * G[5]['d0']:.2f} · {100 * G[6]['d0']:.2f} · {100 * G[8]['d0']:.2f} | - | {G[100]['r95']:.3f} · {G[100]['zc']:.3f} | 1 | {100 * G[5]['dh'][2]:.0f} | (+16.7 첫 접촉) | - | - | - |")
CUR = {}
for g, ns in grp.items():
    last = []
    for n in ns:
        by = W[n]["by"]
        last.append(by[max(by)]["d0"] if by else None)
    bad = sum(1 for n in ns if (summ(n) is None) or summ(n).get("diverged"))
    z = zms(g)
    zbad = sum(1 for q in z if q[3] or q[4])
    er = [100 * s["err_mean_done"] for s in (summ(n) for n in ns) if s]
    lv = [q for q in last if q is not None]
    sd = 100 * float(np.std(lv)) if len(lv) > 1 else None
    zs = " · ".join(f"{fmt(q[1], 1)}/{fmt(q[2], 0)}" + (f" (발산 f{q[3]})" if q[3] else "") for q in z) or "-"
    dh = [val(n, 5, "dh", 2) for n in ns]
    wd = [ws3(n) for n in ns]
    print(f"| {g} | {len(ns)} | {bad}/{len(ns)} + {zbad}/{len(z)} | {rng(last, 2, True)} | {rng([val(n, 5, 'd0') for n in ns], 2, True)} · {rng([val(n, 6, 'd0') for n in ns], 2, True)} · {rng([val(n, 8, 'd0') for n in ns], 2, True)} | {fmt(sd, 2)} | "
          f"{rng([val(n, 100, 'r95') for n in ns])} · {rng([val(n, 100, 'zc') for n in ns])} | {rng([iou(n, 5) for n in ns], 2)} · {rng([iou(n, 10) for n in ns], 2)} · {rng([iou(n, 100) for n in ns], 2)} | "
          f"{rng(dh, 0, True)} | {rng([dE(n) for n in ns], 2)} | {rng(er, 2)} | {zs} | {rng(wd, 2) if any(q is not None for q in wd) else '-'} |")
    CUR[g] = {n: {str(t): W[n]["by"][t].get("d0") for t in W[n]["by"]} for n in ns}

FR = (5, 6, 8, 10, 15, 20, 30, 50, 100)
print("\n## B. 손상 곡선 (묶음 평균 %; 정답 맨 위)\n")
print("| 묶음 | " + " | ".join(f"f{t}" for t in FR) + " |")
print("|---|" + "---|" * len(FR))
print("| 정답 | " + " | ".join(f"{100 * G[t]['d0']:.2f}" for t in FR) + " |")
for g, ns in grp.items():
    print(f"| {g} | " + " | ".join(rng([val(n, t, 'd0') for n in ns], 2, True).split(" [")[0] for t in FR) + " |")

print("\n## C. 미룬 걸음 (묶음 첫 판, 후보 켜진 프레임; ok = 수렴해 커밋한 걸음, defer = 수렴 안 해 확정 안 한 걸음)\n")
print("| 묶음 | 판 | 켠 프레임 | 구간 f8~20 ok · defer | f21~50 | f51~100 |")
print("|---|---|---|---|---|---|")
for g, ns in grp.items():
    n = ns[0]
    rr = [(q["t"] + 1, q["ws3"]) for q in W[n]["rows"] if q.get("ws3")]
    if not rr:
        continue
    on = [t for t, w in rr if w.get("on")]
    cell = []
    for a, b in ((8, 20), (21, 50), (51, 100)):
        s = [w for t, w in rr if a <= t <= b and w.get("on")]
        cell.append(f"{sum(w['ok'] for w in s)} · {sum(w['defer'] for w in s)}")
    print(f"| {g} | {n} | f{on[0] if on else '-'}~ | " + " | ".join(cell) + " |")

print("\n# 절 CM40 과 같은 표 (이 판 목록)\n")
print(base.split("\n", 1)[1])
try:
    cur = json.load(open(os.path.join(O, "wmshape3_curves.json")))
    cur["d0"] = CUR
    json.dump(cur, open(os.path.join(O, "wmshape3_curves.json"), "w"))
except Exception as ex:
    print(f"(곡선 json 덧붙이기 실패 {ex!r})", file=sys.stderr)
