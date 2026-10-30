"""절 AG: N 스윕 (비측정) 결과로 씬마다 N* 를 고른다 + 스윕 표.
N* = 스윕 N 중 그 N 과 그보다 큰 모든 N 이 반복 전부 완주한 가장 작은 N (안정 경계 이상 최소 -- 절 AF 와 같은 규칙).
사용: python exe/fem_ag_pick.py <lbours 루트 (/tmp/dkta/lbours)> <표.md> <측정 목록.txt> <씬...>"""
import collections
import json
import os
import re
import sys

root, out_md, out_list, scenes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
L, picks = [], []
for sc in scenes:
    d0 = os.path.join(root, f"fem_ag_{sc}")
    R = collections.defaultdict(list)
    for d in (sorted(os.listdir(d0)) if os.path.isdir(d0) else []):
        m = re.fullmatch(rf"ag_{sc}_n(\d+)_r(\d+)", d)
        if m:
            f = os.path.join(d0, d, "summary.json")
            R[int(m.group(1))].append(json.load(open(f)) if os.path.exists(f) else None)
    NS = sorted(R)
    if not NS:
        L.append(f"- {sc}: 스윕 없음")
        continue
    ok, err, cells = {}, {}, []
    for n in NS:
        S = [s for s in R[n] if s is not None]
        fin = [s for s in S if not s["diverged"]]
        dv = sorted(s["diverged"] for s in S if s["diverged"])
        ok[n] = len(R[n]) > 0 and len(fin) == len(R[n])
        err[n] = sum(s["err_mean"] for s in fin) / len(fin) if fin else None
        cells.append(f"N {n}: {len(fin)}/{len(R[n])}" + (f" {100 * err[n]:.2f} %" if err[n] is not None else "") + (f" (발산 {','.join(map(str, dv))})" if dv else ""))
    ns = next((n for n in NS if all(ok[q] for q in NS if q >= n)), None)
    L.append(f"- {sc}: " + " · ".join(cells) + f" → N* {ns if ns else '없음'}" + (f" ({100 * err[ns]:.2f} %)" if ns else ""))
    if ns:
        picks.append((sc, ns))
open(out_md, "w").write("\n".join(L) + "\n")
open(out_list, "w").write("".join(f"{s} {n}\n" for s, n in picks))
print("\n".join(L))
