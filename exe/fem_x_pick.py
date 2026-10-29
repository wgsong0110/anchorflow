"""절 X: 스윕 (fem_x_jobs.py) 에서 단독 측정할 판 고르기 -- 모든 반복 완주한 설정 중 N 이 가장 작은 것, 같은 N 이면 오차 작은 것 (fem_w_pick.py 와 같은 규칙).
사용: python exe/fem_x_pick.py <루트> <접두 (x_l_ | x_f_)>  -> "<설정> <N> <제한 인자>" 한 줄 (없으면 NONE)"""
import collections
import json
import os
import re
import sys

R, pre = sys.argv[1], sys.argv[2]
V = {"l0": "", "lL": "--lim 0.3 --lim_mode box"}
G = collections.defaultdict(list)
for b in sorted(os.listdir(R)):
    m = re.match(rf"{pre}(l0|lL)_n(\d+)_r\d+$", b)
    if not m:
        continue
    f = os.path.join(R, b, "summary.json")
    G[(m.group(1), int(m.group(2)))].append(json.load(open(f)) if os.path.exists(f) else None)
C = []
for (v, N), S in sorted(G.items()):
    ok = [s for s in S if s is not None and not s["diverged"]]
    print(f"[후보] {v} N{N}: 완주 {len(ok)}/{len(S)}, 발산 프레임 {[s['diverged'] for s in S if s is not None and s['diverged']]}", file=sys.stderr)
    if S and len(ok) == len(S):
        C.append((N, sum(s["err_mean"] for s in ok) / len(ok), v))
if not C:
    print("NONE")
else:
    N, e, v = sorted(C)[0]
    print(f"[선택] {v} N{N} 오차 {100 * e:.2f} %", file=sys.stderr)
    print(f"{v} {N} {V[v]}")
