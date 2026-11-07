"""절 CM24 공통 세팅 후보 표: 같은 플래그를 다섯 씬 (수박·레고·ficus·wolf·bread) 에 켠 결과를 한 줄에.
사용: python exe/wmk1fix_common.py <출력 디렉토리>
칸 = 결과 · 오차 % · 단독 ms 평균 (a·b 평균; 60 fps = ≤ 16.67 ms 이면 ✓). 수박 오차는 비결정 a·b 평균, 나머지는 결정적 판정 판.
"""
import json
import os
import sys

O = sys.argv[1]
CAND = [("base", "off", "지금 기본 판 (K 1)", ""), ("ks3", "ks3", "적응 걸음 쪼개기 최대 K 8, 판정 1", "--ksplit 3"),
        ("ks3r", "ks3r", "적응 걸음 쪼개기 최대 K 8, 판정 2", "--ksplit 3 --uf_res"), ("ks5r", "ks5r", "적응 걸음 쪼개기 최대 K 32, 판정 2", "--ksplit 5 --uf_res"),
        ("uber", "uber", "덜 풀린 걸음 후퇴 오일러 갱신, 판정 2", "--ue be --uf_res")]
SC = [("L", "레고"), ("F", "ficus"), ("W", "wolf"), ("B", "bread")]


def js(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def ms(n):
    try:
        r = [json.loads(l) for l in open(f"{O}/{n}/fem.jsonl") if l.strip()][2:]
    except Exception:
        return None
    v = [q["ms"] for q in r if q["ms"] == q["ms"]]
    return sum(v) / len(v) if v else None


def res(s):
    if s is None:
        return "없음"
    return ("발산 f%d" % s["diverged"]) if s.get("diverged") else "완주"


def cell(rs, errs, t):
    e = [x for x in errs if x is not None]
    es = " · ".join(f"{x:.2f}" for x in e) if e else "-"
    ts = "-" if t is None else f"{t:.1f} ms {'✓' if t <= 1000 / 60 else '✗'}"
    return f"{rs} · {es} % · {ts}"


print("## 0. 공통 세팅 후보 (다섯 씬 같은 플래그; 칸 = 결과 · 오차 % · 단독 ms 평균 60 fps ✓/✗)\n")
print("| 공통 세팅 후보 | 플래그 | 수박 (a · b) | 레고 | ficus | wolf | bread |")
print("|---|---|---|---|---|---|---|")
for wv, rv, nm, fl in CAND:
    sa, sb = js(f"{O}/{wv}_a/summary.json"), js(f"{O}/{wv}_b/summary.json")
    wr = " · ".join(res(x) for x in (sa, sb))
    we = [100 * x["err_mean_done"] if x else None for x in (sa, sb)]
    wt = ms(f"z_{wv}")
    cells = [cell(wr, we, wt)]
    for s, _ in SC:
        sm = js(f"{O}/r_{s}_{rv}/summary.json")
        if rv == "off":
            ts = [ms(f"z_{s}_off_{r}") for r in "ab"] + [ms(f"z_{s}_off{w}_{r}") for w in ("ks3r", "ks5r", "uber") for r in "ab"]
        else:
            ts = [ms(f"z_{s}_{rv}_{r}") for r in "ab"]
        ts = [x for x in ts if x is not None]
        cells.append(cell(res(sm), [100 * sm["err_mean_done"]] if sm else [], sum(ts) / len(ts) if ts else None))
    print(f"| {nm} | `{fl}` | " + " | ".join(cells) + " |")
