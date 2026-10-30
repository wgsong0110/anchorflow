"""절 AF 영상 합성 (로컬): 정답 (MPM) | 확정 설정 (절 AD) -- exe/fem_ex_compose.py (칸마다 ms·fps·오차, % 는 textfile + expansion=none).
R2 원본을 rclone 으로 /tmp/fem_af_src 에 받아 쓴다. 단독 측정 ms 는 같은 N 단독 3 회 평균으로 덮어쓴 summary 로 (영상 칸 = 표와 같은 값).
사용: python exe/fem_af_compose.py <궤적 0000~0019> <N> (공통 설정: 영상 = 비측정 회차 1 덤프 af_<t>_c<N>_r1, ms = 단독 af_<t>_c<N>_t1~3 평균, 오차 = 6 회 중 완주 평균)"""
import json
import os
import subprocess
import sys

R = "r2:storage/result/anchorflow/lbours/fem_af_lego"
SRC = "/tmp/fem_af_src"
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
here = os.path.dirname(os.path.abspath(__file__))
t, N = sys.argv[1], int(sys.argv[2])


def get(rel, f):
    d = f"{SRC}/{rel}"
    os.makedirs(os.path.dirname(f"{d}/{f}"), exist_ok=True)
    if not os.path.exists(f"{d}/{f}"):
        subprocess.run(["rclone", "copyto", f"{R}/{rel}/{f}", f"{d}/{f}"], check=True)
    return f"{d}/{f}"


S = [json.load(open(get(f"af_{t}_c{N}_t{r}", "summary.json"))) for r in (1, 2, 3)]
A = [json.load(open(get(f"af_{t}_c{N}_r{r}", "summary.json"))) for r in (1, 2, 3)]
fin = [s for s in S + A if not s["diverged"]]
s1 = dict(A[0])
s1["ms_mean"] = sum(s["ms_mean"] for s in S) / 3
s1["fps"] = 1000 / s1["ms_mean"]
s1["err_mean"] = sum(s["err_mean"] for s in fin) / len(fin) if fin else None
sj = f"{SRC}/af_{t}_c{N}_avg.json"
json.dump(s1, open(sj, "w"))
gt = get(f"gt_{t}", "gt_lego.mp4")
ours = get(f"af_{t}_c{N}_r1", "vj/fem_lego.mp4")
title = (f"lego 홀드아웃 정답 궤적 {t} · 80 프레임 · A6000 단독 · 확정 설정 (절 AD: 명시적 lag, 탄성 부피 상한 θ0.5, 클램프 σ1/σ3 ≤ 5) · 공통 h_s 6.25e-5 s (N {N}) · "
         f"ms·fps 는 단독 3 회 평균, 오차는 완주 회 평균 ({len(fin)}/6 완주)")
o = f"{OUT}/fem_af_lego{t}_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o, title, f"정답 (MPM)|-|{gt}", f"공통 설정 N{N}|{sj}|{ours}"], check=True)
print("[저장]", o)
