"""절 AF: N 스윕 (비측정) 결과로 씬마다 N 을 고른다 + 스윕 표 (table1.md) + 단독 측정 목록.
N* = 스윕 N 중 그 N 과 그보다 큰 모든 N 이 반복 전부 완주한 가장 작은 N (안정 경계 이상 최소).
N*의 평균 오차가 10 % 이상이면, 안정하면서 평균 오차 < 10 % 인 가장 작은 N 을 후보 (N10) 로 같이 잰다 (설정 변경 아님 -- 보고용).
사용: python exe/fem_af_pick.py <스윕 루트 (…/fem_af_lego)> <표 출력.md> <측정 목록 출력.txt>"""
import collections
import json
import os
import re
import sys

root, out_md, out_list = sys.argv[1], sys.argv[2], sys.argv[3]
R = collections.defaultdict(lambda: collections.defaultdict(list))
for d in sorted(os.listdir(root)):
    m = re.fullmatch(r"af_(\d{4})_n(\d+)_r(\d+)", d)
    if not m:
        continue
    f = os.path.join(root, d, "summary.json")
    s = json.load(open(f)) if os.path.exists(f) else None
    R[m.group(1)][int(m.group(2))].append((int(m.group(3)), s, os.path.join(root, d)))


def first_cross(dd, thr=0.10):
    """프레임별 오차가 처음 thr 를 넘는 프레임 (1 부터), 마지막 기록 프레임 오차."""
    f = os.path.join(dd, "fem.jsonl")
    if not os.path.exists(f):
        return None, None
    e = [json.loads(ln) for ln in open(f)]
    e = [r for r in e if "err" in r and r.get("err") is not None]
    fc = next((r["t"] + 1 for r in e if r["err"] >= thr), None)
    return fc, (e[-1]["err"] if e else None)


L = ["| 궤적 | " + " | ".join(f"N {n}" for n in sorted({n for t in R for n in R[t]})) + " | N* (안정 경계 이상 최소) | N* 오차 | 오차 < 10 % 최소 N |",
     ""]
NS = sorted({n for t in R for n in R[t]})
L[1] = "|---|" + "---|" * len(NS) + "---|---|---|"
picks = []
for t in sorted(R):
    cells, ok, err = [], {}, {}
    for n in NS:
        runs = R[t].get(n, [])
        S = [s for _, s, _ in runs if s is not None]
        fin = [s for s in S if not s["diverged"]]
        dv = sorted(s["diverged"] for s in S if s["diverged"])
        ok[n] = len(runs) > 0 and len(fin) == len(runs)
        err[n] = sum(s["err_mean"] for s in fin) / len(fin) if fin else None
        c = f"{len(fin)}/{len(runs)}" + (f" {100 * err[n]:.2f} %" if err[n] is not None else "") + (f" (발산 {','.join(map(str, dv))})" if dv else "")
        cells.append(c)
    ns = None
    for n in NS:
        if all(ok[q] for q in NS if q >= n):
            ns = n
            break
    n10 = next((n for n in NS if ns is not None and n >= ns and err[n] is not None and err[n] < 0.10), None)
    L.append(f"| {t} | " + " | ".join(cells) + f" | {ns if ns else '없음'} | {100 * err[ns]:.2f} % |" if ns else f"| {t} | " + " | ".join(cells) + " | 없음 | - |")
    L[-1] += f" {n10 if n10 else '없음'} |"
    if ns:
        picks.append((t, ns, "star"))
        if n10 and n10 != ns:
            picks.append((t, n10, "n10"))
# 진단: 10 % 를 처음 넘는 프레임 (N* 판, 회별)
L += ["", "N* 판 프레임별 오차가 처음 10 % 를 넘는 프레임 (회별; - = 끝까지 안 넘음) · 마지막 프레임 오차:", ""]
for t, n, k in picks:
    if k != "star":
        continue
    q = []
    for r, s, dd in sorted(R[t][n]):
        fc, el = first_cross(dd)
        q.append(f"r{r} {fc if fc else '-'} · {100 * el:.1f} %" if el is not None else f"r{r} ?")
    L.append(f"- {t} N{n}: " + "; ".join(q))
open(out_md, "w").write("\n".join(L) + "\n")
open(out_list, "w").write("".join(f"{t} {n} {k}\n" for t, n, k in picks))
print("\n".join(L))
