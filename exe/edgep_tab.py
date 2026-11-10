"""절 CM105 표 (로컬, 읽기만): R2 lbours/edgep 에서 내려받은 판 디렉토리 (summary.json · heap.md · wm.md · period.txt · epsum.json) 를 무리로 묶어 씬별 표 (한국어 markdown).

사용: python exe/edgep_tab.py <판 상위 디렉토리> > 표.md
무리: 변 벌점 E = e<씬>_*, 정규화 끔 N = n<씬>_*, H9 영상 판 = h<씬> (판 하나). 씬 글자 W wolf · M 수박 · L 레고 · F ficus · B bread.
- 오차 = summary err_mean % (발산이면 err_mean_done), 발산 = summary diverged (프레임).
- wolf (exe/wolfh10_heap.py heap.md 의 판 줄): 바닥 위 질량중심 f50 mm · 안식각 f50 ° · 입자 det F 중앙 f30·f50 · 이탈 판·최대 개수.
- 수박 (exe/tet1_wm.py wm.md 의 판 줄): 이탈 · 튐 프레임/판 · 최대 속력; 손상 = fem.jsonl en.lj0 (logJp > 0 몫) 끝 (f5 · f6 · f8) -- 가벼운 판 lj.pt 는 비어 있음.
- ficus (exe/fem_p_period.py period.txt): 주기 비 (FFT 정답 대비).
- 위반 (lib/anchorflow/fem_edgep.py ep.jsonl 요약 epsum.json): 상한 넘는 변이 있는 걸음 수 · 걸음 최대 넘는 변 수 · 최대 늘어남 d/L0.
"""
import glob
import json
import os
import re
import sys

import numpy as np

B = sys.argv[1]
SC = dict(W="wolf", M="수박", L="레고", F="ficus", B="bread")


def rows_of(md, run):
    if not os.path.exists(md):
        return None
    for l in open(md):
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(c) > 2 and (c[0] == run or c[1] == run):
            return c
    return None


def one(d):
    run = os.path.basename(d.rstrip("/"))
    r = dict(run=run)
    try:
        s = json.load(open(os.path.join(d, "summary.json")))
        e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
        r["err"] = None if e is None else 100 * e
        r["div"] = s.get("diverged")
        r["frames"] = s.get("frames_done")
    except Exception:                                                                     # 요약 없음 = 실행기가 메모리 넘침으로 죽음 (rc 1): fem.jsonl 의 마지막 프레임 다음을 발산 프레임으로, 오차는 끝난 프레임 평균
        F = [json.loads(l) for l in open(os.path.join(d, "fem.jsonl"))] if os.path.exists(os.path.join(d, "fem.jsonl")) else []
        if not F:
            r["missing"] = True
            return r
        r["err"] = 100 * float(np.mean([q["err"] for q in F])); r["div"] = int(F[-1]["t"]) + 2; r["frames"] = len(F); r["oom"] = True
    try:
        r["ep"] = json.load(open(os.path.join(d, "epsum.json")))
    except Exception:
        r["ep"] = None
    c = rows_of(os.path.join(d, "heap.md"), run)
    if c:                                                                                 # | 무리 | 판 | 발산 | 오차 | 질량중심 (바닥 위) | 꼭대기 | 안식각 | 발자국 | 채움 | det F | 이탈 |
        m = re.search(r"\(([-0-9.]+)\)", c[4]); r["com"] = float(m.group(1)) if m and m.group(1) not in ("-",) else None
        r["angle"] = c[6]; r["fill"] = c[8]; r["detF"] = c[9]; r["esc"] = c[10]; r["top"] = c[5]
    hr = os.path.join(d, "heap_rows.json")
    if os.path.exists(hr):                                                                # 프레임별 (균일 가중 바닥 위 질량중심 mm · 채움 · 입자 det F 중앙): 발산 판도 그 프레임까지
        H = json.load(open(hr)); k = [x for x in H if x != "정답"][0]; m = H[k]["runs"][0]["m"]
        r["fr"] = {f: (m[f]["comu"], m[f]["ppv"], m[f].get("detF")) for f in m if m[f]}
    c = rows_of(os.path.join(d, "wm.md"), run)
    if c:                                                                                 # | 판 | 판 수 | 발산 | 오차 | 끝 손상 | 이탈 | 튐 | 꿈틀거림 | 되튐 |
        r["esc"] = c[5]; r["jump"] = c[6]; r["wig"] = c[7]
        F = [json.loads(l) for l in open(os.path.join(d, "fem.jsonl"))] if os.path.exists(os.path.join(d, "fem.jsonl")) else []
        lj = {int(q["t"]): q["en"]["lj0"] for q in F if isinstance(q.get("en"), dict) and q["en"].get("lj0") is not None}   # 가벼운 판 (@s6l) 은 lj.pt 가 비어 손상은 fem.jsonl en.lj0 (같은 정의: logJp > 0 몫, 위치 유한 입자 분모)
        if lj:
            last = max(lj)
            r["dmg"] = "%.2f (%s)" % (100 * lj[last], " · ".join(("%.2f" % (100 * lj[t])) if t in lj else "-" for t in (4, 5, 7)))
    p = os.path.join(d, "period.txt")
    if os.path.exists(p):
        t = open(p).read()
        m = re.search(r"정답 대비 ([0-9.]+)", t); r["period"] = float(m.group(1)) if m else None
        m = re.search(r"\(비 ([0-9.]+)\)", t); r["amp"] = float(m.group(1)) if m else None
    return r


def fmt_err(R):
    E = [q["err"] for q in R if q.get("err") is not None]
    if not E:
        return "-"
    return f"{np.mean(E):.2f} [{min(E):.2f}~{max(E):.2f}]" if len(E) > 1 else f"{E[0]:.2f}"


def fmt_div(R):
    D = [q for q in R if q.get("div")]
    return f"{len(D)}/{len(R)}" + (" (" + " · ".join(f"f{q['div']}" + (" 메모리" if q.get("oom") else "") for q in D) + ")" if D else "")


def fmt_ep(R):
    E = [q["ep"] for q in R if q.get("ep")]
    if not E:
        return "-"
    return (f"{sum(e['steps_over'] for e in E) / len(E):.1f} 걸음/판 · 최대 {max(e['max_nover'] for e in E)} 변 · 최대 늘어남 {max(e['max_rmax'] for e in E):.2f}")


runs = {}
for d in sorted(glob.glob(os.path.join(B, "*/"))):
    n = os.path.basename(d.rstrip("/"))
    m = re.match(r"^([enh])([WMLFB])(_\d+)?$", n)
    if not m:
        continue
    runs.setdefault((m.group(2), m.group(1)), []).append(one(d))
GN = dict(h="H9 (영상 판 1)", n="정규화 끔 N", e="변 벌점 E")
for sk in "WMLFB":
    print(f"\n### {SC[sk]}\n")
    if sk == "W":
        print("| 판 | 발산 | 오차 % | 바닥 위 질량중심 f30 · f40 (균일, 판 수) · f50 (질량 가중) mm | 채움 f30 · f40 | 입자 det F 중앙 f30 · f40 | 안식각 f50 ° | 이탈 판 · 최대 개수 | 위반: 넘는 걸음 · 걸음 최대 넘는 변 · 최대 d/L0 |")
        print("|---|---|---|---|---|---|---|---|---|")
        print("| 정답 | - | - | 86 · 45 · 53 | 2.27 · 2.24 | 1 · 1 | 26.6 | - | - |")
    elif sk == "M":
        print("| 판 | 발산 | 오차 % | 끝 손상 % (f5 · f6 · f8) | 이탈 판 · 최대 개수 | 튐 프레임/판 · 최대 속력 m/s | 위반: 넘는 걸음 · 걸음 최대 넘는 변 · 최대 d/L0 |")
        print("|---|---|---|---|---|---|---|")
    elif sk == "F":
        print("| 판 | 발산 | 오차 % | 주기 비 (진폭 비) | 위반: 넘는 걸음 · 걸음 최대 넘는 변 · 최대 d/L0 |")
        print("|---|---|---|---|---|")
    else:
        print("| 판 | 발산 | 오차 % | 위반: 넘는 걸음 · 걸음 최대 넘는 변 · 최대 d/L0 |")
        print("|---|---|---|---|")
    for g in "hne":
        R = runs.get((sk, g), [])
        if not R:
            continue
        R = [q for q in R if not q.get("missing")]
        nm = f"{GN[g]} ({len(R)} 판)"
        if sk == "W":
            def fr(f, i):
                V = [q["fr"][f][i] for q in R if q.get("fr") and f in q["fr"] and q["fr"][f][i] is not None]
                return ("%.0f" % np.mean(V) if i == 0 else "%.2f" % np.mean(V)) + f" ({len(V)})" if V else "-"
            esc = [q.get("esc", "-") for q in R]
            nesc = sum(1 for x in esc if x not in ("-", None) and not x.startswith("0/"))
            mx = max([int(re.findall(r"(\d+)$", x)[0]) for x in esc if x and re.findall(r"(\d+)$", x)] or [0])
            C = [q["com"] for q in R if q.get("com") is not None]
            print(f"| {nm} | {fmt_div(R)} | {fmt_err(R)} | {fr('30', 0)} · {fr('40', 0)} · {('%.0f' % np.mean(C)) if C else '-'} | {fr('30', 1)} · {fr('40', 1)} | {fr('30', 2)} · {fr('40', 2)} | {' · '.join(q.get('angle', '-') for q in R)} | {nesc}/{len(R)} · {mx} | {fmt_ep(R)} |")
        elif sk == "M":
            esc = [q.get("esc", "-") for q in R]
            nesc = sum(1 for x in esc if x not in ("-", None) and not x.startswith("0/"))
            mx = max([int(re.findall(r"(\d+)$", x)[0]) for x in esc if x and re.findall(r"(\d+)$", x)] or [0])
            J = [q.get("jump", "-") for q in R]
            jf = [float(x.split("·")[0]) for x in J if x and "·" in x]
            jv = [float(x.split("·")[1]) for x in J if x and "·" in x]
            print(f"| {nm} | {fmt_div(R)} | {fmt_err(R)} | {' · '.join(q.get('dmg', '-').split(' ')[0] for q in R)} (f5 {np.mean([float(q['dmg'].split('(')[1].split(' · ')[0]) for q in R if q.get('dmg')]):.2f}) | {nesc}/{len(R)} · {mx} | {np.mean(jf) if jf else float('nan'):.1f} · {max(jv) if jv else float('nan'):.1f} | {fmt_ep(R)} |")
        elif sk == "F":
            print(f"| {nm} | {fmt_div(R)} | {fmt_err(R)} | {' · '.join(('%.2f' % q['period'] if q.get('period') is not None else '-') + ' (' + ('%.2f' % q['amp'] if q.get('amp') is not None else '-') + ')' for q in R)} | {fmt_ep(R)} |")
        else:
            print(f"| {nm} | {fmt_div(R)} | {fmt_err(R)} | {fmt_ep(R)} |")
