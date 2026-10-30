"""절 AG 영상 합성 (로컬): 정답 (PG/GF MPM) | 확정 설정 + 소성·점성·파괴 -- exe/fem_ex_compose.py (칸마다 ms·fps·오차, % 는 textfile + expansion=none).
R2 원본을 rclone 으로 /tmp/fem_ag_src 에 받아 쓴다. ms·fps 는 같은 N 단독 3 회 평균 (표와 같은 값), 오차는 완주 회 평균.
사용: python exe/fem_ag_compose.py <씬 hotdog|wolf|bread|wm> <N> [갈래 꼬리:이름 ...]  (기본 ":셀 증분 return mapping")"""
import json
import os
import subprocess
import sys

SRC = "/tmp/fem_ag_src"
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
here = os.path.dirname(os.path.abspath(__file__))
sc, N = sys.argv[1], int(sys.argv[2])
VAR = [v.split(":", 1) for v in (sys.argv[3:] or [":셀 증분 return mapping"])]
R = f"r2:storage/result/anchorflow/lbours/fem_ag_{sc}"
SHAPE = {"hotdog": "hotdog", "wolf": "wolf", "bread": "bread", "wm": "watermelon"}[sc]
DESC = {"hotdog": "hotdog 점소성 (PG foam: 항복 2000 Pa·점성 100, 바닥 sticky) 180 프레임",
        "wolf": "wolf 모래 (PG 공식 wolf_config: Drucker-Prager φ30°, 순차 방출, 바닥 sticky) 50 프레임",
        "bread": "bread 소성 손상 (PG plasticine 항복 200 Pa·softening 0.1, 양쪽 당김) 300 프레임",
        "wm": "수박 파괴 (GF 공식 NACC: E 2000·φ45°·ξ3·경화, 높이에서 v0 -6 낙하) 100 프레임"}[sc]


def get(rel, f):
    d = f"{SRC}/{sc}/{rel}"
    os.makedirs(os.path.dirname(f"{d}/{f}"), exist_ok=True)
    if not os.path.exists(f"{d}/{f}"):
        subprocess.run(["rclone", "copyto", f"{R}/{rel}/{f}", f"{d}/{f}"], check=True)
    return f"{d}/{f}"


cells, nfin = [], []
for sfx, vname in VAR:
    S = [json.load(open(get(f"ag_{sc}_n{N}{sfx}_t{r}", "summary.json"))) for r in (1, 2, 3)]
    fin = [s for s in S if not s["diverged"]]
    s1 = dict(S[0])
    s1["ms_mean"] = sum(s["ms_mean"] for s in S) / 3
    s1["fps"] = 1000 / s1["ms_mean"]
    s1["err_mean"] = sum(s["err_mean"] for s in fin) / len(fin) if fin else None
    sj = f"{SRC}/{sc}/ag_{sc}_n{N}{sfx}_avg.json"
    json.dump(s1, open(sj, "w"))
    ours = get(f"ag_{sc}_n{N}{sfx}_t1", f"vj/fem_{SHAPE}.mp4")
    dv = "" if not S[0]["diverged"] else f" (발산 {S[0]['diverged']})"
    cells.append(f"{vname} {len(fin)}/3 완주{dv}|{sj}|{ours}")
    nfin.append(len(fin))
gt = get("gt", f"gt_{SHAPE}.mp4")
title = (f"{DESC} · A6000 단독 · 확정 설정 (절 AD) + 정답 솔버 재질 식 (절 AG) · 공통 h_s 6.25e-5 → N {N} · ms·fps 단독 3 회 평균, 오차 완주 회 평균 (발산 판은 끝난 프레임까지)")
o = f"{OUT}/fem_ag_{sc}_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o, title, f"정답 ({'GF' if sc == 'wm' else 'PG'} MPM)|-|{gt}", *cells], check=True)
print("[저장]", o)
