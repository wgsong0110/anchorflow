"""절 CM46 묶음 wmshape4 표 (절 CM44 exe/wmshape3_tab.py 와 같은 정의 + D 바닥 뚫음·이탈 (진단 D2) + E 첫 접촉 층별 진단 D4): python exe/wmshape4_tab.py <결과 디렉토리> <gt_jp 디렉토리> -> markdown (stdout) + <결과>/wmshape4_curves.json.
판 = exe/wmshape4_spec.txt, 묶음 = 이름의 끝 _숫자 뺀 것 (비결정 반복). 뒤에 절 CM40 표 (exe/wmshape_tab.py, 같은 정의) 를 이 판 목록으로.
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
os.environ["WSH_SPEC"] = os.path.join(HERE, "wmshape4_spec.txt")
os.environ["WSH_CUR"] = "wmshape4_curves.json"
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


print("# 절 CM46 묶음 wmshape4 -- 수박 V2NF 발산 원인·W̄·첫 접촉 후보 T\n")
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


def e2rows(n):
    p = os.path.join(O, n, "e2.jsonl")
    if not os.path.exists(p):
        return []
    o = []
    for l in open(p):
        try:
            q = json.loads(l)
        except Exception:
            continue
        if "f" in q:
            o.append(q)
    return o


def cause(n):
    s = summ(n)
    lg = os.path.join(O, n, "log.txt")
    t = open(lg, errors="ignore").read()[-6000:] if os.path.exists(lg) else ""
    if "OutOfMemoryError" in t:
        return "메모리"
    if s is None:
        return "죽음"
    return f"발산 f{s.get('diverged')}" if s.get("diverged") else "완주"


print("\n## D. 판별 바닥 뚫음·이탈 (진단 D2: 바닥 밑 = z < 0.02 − 한 칸, 이탈 = 몸 상자 4 칸 밖)\n")
print("| 판 | 끝 | 프레임 | 바닥 밑 입자 첫 프레임 · 최대 수 | z 최소 m | 이탈 최대 수 | 노드 수 최대 (백만) | 바닥 밑 첫 프레임 → 끝 프레임 |")
print("|---|---|---|---|---|---|---|---|")
for g, ns in grp.items():
    for n in ns:
        r = e2rows(n)
        if not r:
            continue
        nb = [q for q in r if q.get("nbelow", 0) > 0]
        nnm = max((q.get("nn", 0) for q in W[n]["rows"]), default=0)
        nnm = max([nnm] + [q.get("nn", 0) for q in r])
        print(f"| {n} | {cause(n)} | {len(W[n]['rows'])} | {('f%d' % nb[0]['f']) if nb else '-'} · {max((q['nbelow'] for q in r), default=0)} | "
              f"{min(q['zmin'] for q in r):.3f} | {max((q.get('nfar', 0) for q in r), default=0)} | {nnm / 1e6:.2f} | {(('f%d → f%d' % (nb[0]['f'], r[-1]['f'])) if nb else '-')} |")

print("\n## E. 첫 접촉 층별 진단 D4 (w4.jsonl; 프레임 시작 높이 층마다 입자 보간 dz 중앙 (m), 띠 몫 r, 도달 몫 rT, √k, 시작점 β, 전진 없음 몫)\n")
for g, ns in grp.items():
    for n in ns:
        p = os.path.join(O, n, "w4.jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p):
            q = json.loads(l)
            if q["f"] != 4:
                continue
            g2 = q.get("g2F2") or []
            fg = lambda x: "-" if x is None else f"{x:.3g}"
            ex = f" (다시 함 {q.get('redo')}, 시도 {q.get('tries')})" if q["tag"] == "fin" else ""
            print(f"\n### {n} f{q['f']} 호출 {q['call']} {q['tag']}{ex} -- 반복 {q['it']}, ‖g‖²/F² 처음 {fg(g2[0] if g2 else None)} 끝 {fg(g2[-1] if g2 else None)}\n")
            print("| 층 cm | 입자 | dz 관성 예측 | dz 목표 yh | dz 시작점 | dz 끝 | 전진 없음 | r | rT | √k | β | 노드 보폭 (반복 1·마지막) |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for rw in q["layers"]:
                kit = rw.get("kit") or []
                print(f"| {rw['z'][0]}~{rw['z'][1]} | {rw['n']} | {rw['Ut']:.4f} | {rw['yh']:.4f} | {rw['y0']:.4f} | {rw['fin']:.4f} | {rw['noprog']:.2f} | {rw['r']:.3f} | "
                      f"{rw.get('rT', float('nan')):.3f} | {rw['sk']:.3g} | {rw.get('beta', float('nan')):.3g} | {(f'{kit[0]:.3g} · {kit[-1]:.3g}') if kit else '-'} |")

print("\n# 절 CM40 과 같은 표 (이 판 목록)\n")
print(base.split("\n", 1)[1])
try:
    cur = json.load(open(os.path.join(O, "wmshape4_curves.json")))
    cur["d0"] = CUR
    json.dump(cur, open(os.path.join(O, "wmshape4_curves.json"), "w"))
except Exception as ex:
    print(f"(곡선 json 덧붙이기 실패 {ex!r})", file=sys.stderr)
