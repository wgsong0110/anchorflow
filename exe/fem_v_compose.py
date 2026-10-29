"""절 V 영상·그림 합성 (로컬 ffmpeg + matplotlib; % 는 fem_ex_compose.py 의 textfile + expansion=none).
  lego/ficus 3DGS: 정답 | 절 R 최고 | 절 T 최고 | 노드 적분 판 (칸마다 ms·fps·오차)
  ficus 잎·가지 끝 변위 시계열·진동 주기 (z 상위 5 % 입자, exe/fem_p_period.py 와 같은 분석, 렌더 입력 h5 에서)
사용: python exe/fem_v_compose.py lego|ficus "<칸 이름>|<lbours 아래 상대 경로>" ...   (각 경로에 summary.json, vj/fem_<형상>.mp4, h5_fem/)"""
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
    base = [f"정답 (MPM)|-|{L}/fem2_lego/gt_lego.mp4"]
    title = "lego L000 80 프레임 · A6000 단독 · 정답 | 셀 적분 (절 R·T 최고) | 노드 적분 FEM (절 V) -- 명시적 lag (프레임당 P2G 1 회 → 메시 서브스텝 N → G2P 1 회)"
else:
    base = [f"정답 (PG)|-|{L}/fem3_ficus/gt_ficus.mp4"]
    title = "ficus F_pg_v0 125 프레임 · A6000 단독 · 정답 | 셀 적분 (절 R·T 최고) | 노드 적분 FEM (절 V) -- lego 와 같은 코드 경로, 바닥 없음"
args = base + [f"{nm}|{L}/{p}/summary.json|{L}/{p}/vj/fem_{shape}.mp4" for nm, p in cells]
o1 = f"{OUT}/fem_v_{shape}_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o1, title, *args], check=True)
if shape == "ficus":
    F_ = os.path.expanduser("~/.fonts/NotoSerifCJKkr-Bold.otf")
    fm.fontManager.addfont(F_)
    plt.rcParams.update({"font.family": fm.FontProperties(fname=F_).get_name(), "axes.unicode_minus": False})
    dt = 0.04
    X0 = np.load("/tmp/ficper/x_gt.npy", mmap_mode="r")
    x0 = np.array(X0[0])
    sel = np.where(x0[:, 2] >= np.quantile(x0[:, 2], 0.95))[0]
    d = np.array([0.977, -0.21, -0.016])
    t = np.arange(126) * dt

    def ana(u):
        s = u - u[63:].mean()
        zc = [t[i - 1] + s[i - 1] / (s[i - 1] - s[i]) * dt for i in range(1, len(s)) if s[i - 1] * s[i] < 0]
        return (2 * np.mean(np.diff(zc[:8])) if len(zc) >= 3 else np.nan), np.abs(u).max()

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ug = (np.array(X0[:, sel, :]).mean(1) - x0[sel].mean(0)) @ d
    Tg, Ag = ana(ug)
    ax.plot(t, ug, color="#0b0b0b", lw=2.6, label=f"정답 (PG MPM): 주기 {Tg:.2f} s")
    cols = ["#e04a1f", "#1f6fe0", "#1baf7a", "#a33fbf", "#c9a400"]
    for k, (nm, p) in enumerate(cells):
        Xs = np.stack([h5py.File(f"{L}/{p}/h5_fem/sim_{i:010d}.h5", "r")["x"][:][sel] for i in range(126)])
        u = (Xs.mean(1) - x0[sel].mean(0)) @ d
        T, A = ana(u)
        ax.plot(t[:len(u)], u, color=cols[k % len(cols)], lw=1.8, label=f"{nm}: 주기 {T:.2f} s (정답 대비 {T / Tg:.2f}), 진폭 비 {A / Ag:.2f}")
        print(f"주기 {nm}: {T:.3f} s (비 {T / Tg:.2f}), 진폭 비 {A / Ag:.2f}")
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.set_xlabel("시간 (s)")
    ax.set_ylabel("주 진동 방향 변위 (z 상위 5 % 입자 평균, m)")
    ax.set_title(f"ficus 잎·가지 끝 변위 시계열 · 정답 주기 {Tg:.2f} s · 셀 적분 (절 R·T) vs 노드 적분 (절 V)")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.grid(alpha=.25)
    fig.tight_layout()
    o2 = f"{OUT}/fem_v_ficus_period.png"
    fig.savefig(o2, dpi=130)
    print("[저장]", o2)
print("[저장]", o1)
