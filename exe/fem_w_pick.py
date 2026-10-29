"""절 W: 스윕 (fem_w_jobs.py) 에서 단독 측정할 최고 판 고르기 -- 모든 반복 완주한 설정 중 N 이 가장 작은 것 (60 fps 쪽), 같은 N 이면 오차가 작은 것.
사용: python exe/fem_w_pick.py <루트 디렉토리> <접두 (w_l_ | w_f_)>  -> "<설정 이름> <N> <G2P/제한 인자>" 한 줄 (없으면 NONE)"""
import collections
import json
import os
import re
import sys

R, pre = sys.argv[1], sys.argv[2]
V = {"gb": "--g2p bary", "gs": "--g2p bs", "gbL": "--g2p bary --lim 0.3 --lim_mode box", "gsL": "--g2p bs --lim 0.3 --lim_mode box"}
G = collections.defaultdict(list)
for b in sorted(os.listdir(R)):
    m = re.match(rf"{pre}(gb|gs|gbL|gsL)_n(\d+)_r\d+$", b)
    if not m:
        continue
    f = os.path.join(R, b, "summary.json")
    s = json.load(open(f)) if os.path.exists(f) else None
    G[(m.group(1), int(m.group(2)))].append(s)
C = []
for (v, N), S in G.items():
    if S and all(s is not None and not s["diverged"] for s in S):
        C.append((N, sum(s["err_mean"] for s in S) / len(S), v))
    print(f"[후보] {v} N{N}: 완주 {sum(1 for s in S if s is not None and not s['diverged'])}/{len(S)}", file=sys.stderr)
if not C:
    print("NONE")
else:
    N, e, v = sorted(C)[0]
    print(f"[선택] {v} N{N} 오차 {100 * e:.2f} %", file=sys.stderr)
    print(f"{v} {N} {V[v]}")
