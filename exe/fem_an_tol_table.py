"""절 AN.14 허용치 훑기 표: <묶음 디렉토리> 의 <판>_<궤적>/ (summary.json, period.txt, block_ana.txt, drop_ana.txt) 를 판 x 궤적 표로.
칸: lego = 평균 오차 % (발산이면 'x프레임'), ficus = 주기 비 / 진폭 비 / 오차 %, 그리고 반복 분포 (프레임당 뉴턴 합·PCG 합 평균/최대, 뉴턴 상한 걸음·PCG 상한 풀이 평균).
사용: python exe/fem_an_tol_table.py <dir> [판 접두 순서...]"""
import glob
import json
import os
import re
import sys

import numpy as np

R = sys.argv[1]
runs = {}
for d in sorted(glob.glob(os.path.join(R, "*_*"))):
    if not os.path.isdir(d):
        continue
    b = os.path.basename(d)
    v, t = b.rsplit("_", 1)
    runs.setdefault(v, {})[t] = d


def cell(d, t):
    sp = os.path.join(d, "summary.json")
    if not os.path.exists(sp):
        lg = os.path.join(d, "log.txt")
        if os.path.exists(lg):
            tx = open(lg, errors="ignore").read()
            m = re.findall(r"프레임 (\d+)/\d+", tx)
            return ("돌는 중 " + m[-1]) if m else "시작"
        return ""
    s = json.load(open(sp))
    e = s.get("err_mean")
    o = (f"{100 * e:.2f}" if e is not None else f"x{s.get('diverged')}")
    if t == "ficus":
        pp = os.path.join(d, "period.txt")
        if os.path.exists(pp):
            m = re.search(r"정답 대비 ([\d.]+)\), 최대 진폭 [\d.]+ m \(비 ([\d.]+)\)", open(pp).read())
            if m:
                o = f"{m.group(1)} / {m.group(2)} / {o}"
            else:
                o = "주기? / " + o
    for k in ("stretch", "shear"):
        if t == k:
            bp = os.path.join(d, "block_ana.txt")
            if os.path.exists(bp):
                o += " | " + " ".join(open(bp).read().split())[-120:]
    it = s.get("it_nN")
    if it:
        o += f" [N {s['it_nN']['mean']:.1f}/{s['it_nN']['max']:.0f}, P {s['it_nP']['mean']:.0f}/{s['it_nP']['max']:.0f}, 상한N {s['it_capN']['mean']:.2f}, 상한P {s['it_capP']['mean']:.2f}]"
    return o


ts = sorted({t for v in runs.values() for t in v}, key=lambda q: (q != "ficus", q))
print("| 판 | " + " | ".join(ts) + " |")
print("|---|" + "---|" * len(ts))
for v in sorted(runs):
    print(f"| {v} | " + " | ".join(cell(runs[v][t], t) if t in runs[v] else "" for t in ts) + " |")

# 판정 요약 (판마다): lego (L000 + 0000 에서 0019) 10 % 미만 수·완주 평균·발산, ficus 주기 비·진폭 비, 블록 주기·진폭, 낙하 반발·튀어 오름, 프레임 시간 (비측정)
LEGO = ["L000"] + [f"{i:04d}" for i in range(20)]
print()
print("| 판 | lego <10 % | 완주 평균 % | 발산 | 미달 (오차 %) | ficus 주기 비 / 진폭 비 / 오차 % | 신축 주기 ms / 진폭 끝·처음 | 전단 주기 ms / 진폭 | 낙하 반발 / 튀어 오름 mm |")
print("|---|---|---|---|---|---|---|---|---|")
for v in sorted(runs):
    rr = runs[v]
    lg = [t for t in LEGO if t in rr and os.path.exists(os.path.join(rr[t], "summary.json"))]
    if not lg and "ficus" not in rr:
        continue
    ok, es, dv, bad = 0, [], [], []
    for t in lg:
        s = json.load(open(os.path.join(rr[t], "summary.json")))
        if s.get("diverged"):
            dv.append(f"{t}@{s['diverged']}")
            continue
        e = 100 * s["err_mean"]; es.append(e)
        if e < 10:
            ok += 1
        else:
            bad.append(f"{t} {e:.2f}")
    fc = cell(rr["ficus"], "ficus").split(" [")[0] if "ficus" in rr else ""
    bl = []
    for k in ("stretch", "shear"):
        o = ""
        if k in rr:
            bp = os.path.join(rr[k], "block_ana.txt")
            if os.path.exists(bp):
                m = re.search(rf"{k} {re.escape(v)}: 주기 영점 ([\d.na]+) ms .*?진폭 끝/처음 ([\d.]+)", open(bp).read())
                o = f"{m.group(1)} / {m.group(2)}" if m else "?"
        bl.append(o)
    dr = ""
    if "drop1" in rr:
        dp = os.path.join(rr["drop1"], "drop_ana.txt")
        if os.path.exists(dp):
            m = re.search(rf"{re.escape(v)}: .*?반발 계수 ([\d.-]+), 튀어 오름 ([\d.-]+) mm", open(dp).read())
            dr = f"{m.group(1)} / {m.group(2)}" if m else "?"
    print(f"| {v} | {ok}/{len(lg)} | {np.mean(es):.2f} | {len(dv)} {' '.join(dv)} | {', '.join(bad)} | {fc} | {bl[0]} | {bl[1]} | {dr} |" if es else f"| {v} | 0/{len(lg)} | | {len(dv)} | | {fc} | {bl[0]} | {bl[1]} | {dr} |")
