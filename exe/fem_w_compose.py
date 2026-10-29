"""절 W 영상·그림 합성 (fem_r_compose.py 를 칸마다 루트를 주도록; 로컬 ffmpeg + matplotlib; % 는 fem_ex_compose.py 의 textfile + expansion=none).
  3DGS: 정답 | 칸들 (칸마다 ms·fps·오차; ms·fps 는 칸의 ms 원천 summary.json -- 최종 실행이 측정 중 오염됐으면 단독 재측정 실행의 것)
  ficus 잎·가지 끝 변위 시계열·진동 주기 (z 상위 5 % 입자, exe/fem_p_period.py 와 같은 분석, 렌더 입력 h5 에서)
사용: python exe/fem_w_compose.py lego|ficus "<칸 이름>|<루트>/<태그>[|<ms 원천 루트/태그>]" ...   (lbours/<루트>/<태그>)"""
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

L = os.path.expanduser("~/r2/result/anchorflow/lbours")
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
here = os.path.dirname(os.path.abspath(__file__))
shape = sys.argv[1]
cells = [c.split("|") for c in sys.argv[2:]]
if shape == "lego":
    base = [f"정답 (MPM)|-|{L}/fem2_lego/gt_lego.mp4",
            ]
    title = "lego L000 80 프레임 · A6000 단독 · 명시적 lag (프레임당 P2G 1 회 → 사면체 메시 서브스텝 N → G2P 1 회) · 노드 질량: 절 R 셀 안 입자 m_T/4 | 절 T 노드 B-스플라인 | 절 W 셀 B-스플라인 m_T/4"
else:
    base = [f"정답 (PG)|-|{L}/fem3_ficus/gt_ficus.mp4"]
    title = "ficus F_pg_v0 125 프레임 · A6000 단독 · 명시적 lag (lego 와 같은 코드 경로, 바닥 없음) · 노드 질량: 절 R 셀 안 입자 m_T/4 | 절 T 노드 B-스플라인 | 절 W 셀 B-스플라인 m_T/4"
args = base + [f"{c[0]}|{L}/{c[2] if len(c) > 2 else c[1]}/summary.json|{L}/{c[1]}/vj/fem_{shape}.mp4" for c in cells]
o1 = f"{OUT}/fem_w_{shape}_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o1, title, *args], check=True)
if shape == "ficus":
    F_ = os.path.expanduser("~/.fonts/NotoSerifCJKkr-Bold.otf")
    fm.fontManager.addfont(F_)
    plt.rcParams.update({"font.family": fm.FontProperties(fname=F_).get_name(), "axes.unicode_minus": False})
    dt = 0.04
    GH = f"{L}/ficus_v0/gt"                                                         # 정답 h5 (렌더 입력과 같은 형식)
    x0 = h5py.File(f"{GH}/sim_{0:010d}.h5", "r")["x"][:]
    sel = np.where(x0[:, 2] >= np.quantile(x0[:, 2], 0.95))[0]
    X0 = np.stack([h5py.File(f"{GH}/sim_{i:010d}.h5", "r")["x"][:][sel] for i in range(126)])
    d = np.array([0.977, -0.21, -0.016])
    t = np.arange(126) * dt

    def ana(u):
        s = u - u[63:].mean()
        zc = [t[i - 1] + s[i - 1] / (s[i - 1] - s[i]) * dt for i in range(1, len(s)) if s[i - 1] * s[i] < 0]
        return (2 * np.mean(np.diff(zc[:8])) if len(zc) >= 3 else np.nan), np.abs(u).max()

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ug = (X0.mean(1) - x0[sel].mean(0)) @ d
    Tg, Ag = ana(ug)
    ax.plot(t, ug, color="#0b0b0b", lw=2.6, label=f"정답 (PG MPM): 주기 {Tg:.2f} s")
    cols = ["#e04a1f", "#1f6fe0", "#1baf7a", "#a33fbf", "#c9a400"]
    for k, c in enumerate(cells):
        nm, tg = c[0], c[1]
        Xs = np.stack([h5py.File(f"{L}/{tg}/h5_fem/sim_{i:010d}.h5", "r")["x"][:][sel] for i in range(126)])   # 렌더 입력 (프레임 0 = 정답 시작)
        u = (Xs.mean(1) - x0[sel].mean(0)) @ d
        T, A = ana(u)
        ax.plot(t[:len(u)], u, color=cols[k % len(cols)], lw=1.8, label=f"{nm}: 주기 {T:.2f} s (정답 대비 {T / Tg:.2f}), 진폭 비 {A / Ag:.2f}")
        print(f"주기 {nm}: {T:.3f} s (비 {T / Tg:.2f}), 진폭 비 {A / Ag:.2f}")
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.set_xlabel("시간 (s)")
    ax.set_ylabel("주 진동 방향 변위 (z 상위 5 % 입자 평균, m)")
    ax.set_title(f"ficus 잎·가지 끝 변위 시계열 · 정답 주기 {Tg:.2f} s")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.grid(alpha=.25)
    fig.tight_layout()
    o2 = f"{OUT}/fem_w_ficus_period.png"
    fig.savefig(o2, dpi=130)
    print("[저장]", o2)
print("[저장]", o1)
