"""절 R 영상·그림 합성 (로컬 ffmpeg + matplotlib; % 는 fem_ex_compose.py 의 textfile + expansion=none).
  lego 3DGS : 정답 | 암시적 최종 (k_final_F) | 후보 최고 판들 (칸마다 ms·fps·오차)
  ficus 3DGS: 정답 | 암시적 최종 (rt_tmix_fix) | 후보 최고 판들
  ficus 잎·가지 끝 변위 시계열·진동 주기 (z 상위 5 % 입자, exe/fem_p_period.py 와 같은 분석, pred.pt 에서)
사용: python exe/fem_q_compose.py lego|ficus "<칸 이름>|<태그>" ...   (lbours/fem_q_<형상>/<태그>)"""
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
R = f"{L}/fem_r_{shape}"
if shape == "lego":
    base = [f"정답 (MPM)|-|{L}/fem2_lego/gt_lego.mp4",
            f"암시적 FEM 최종|{L}/fem_rt_lego/k_final_F/summary.json|{L}/fem_rt_lego/k_final_F/vj/fem_lego.mp4"]
    title = "lego L000 80 프레임 · A6000 단독 · 정답 | 암시적 최종 | 명시적 lag (프레임당 P2G 1 회 → 메시 서브스텝 N → G2P 1 회) + 셀 뒤집힘 방지 (log J 장벽 · 노드 이동 제한)"
else:
    base = [f"정답 (PG)|-|{L}/fem3_ficus/gt_ficus.mp4",
            f"암시적 실시간 + 수정 판정|{L}/fem_rt_ficus/rt_tmix_fix/summary.json|{L}/fem_rt_ficus/rt_tmix_fix/v/fem_ficus.mp4"]
    title = "ficus F_pg_v0 125 프레임 · A6000 단독 · 정답 | 암시적 최종 | 명시적 lag (lego 와 같은 코드 경로, 바닥 없음) + 셀 뒤집힘 방지"
args = base + [f"{nm}|{R}/{tg}/summary.json|{R}/{tg}/vj/fem_{shape}.mp4" for nm, tg in cells]
o1 = f"{OUT}/fem_r_{shape}_cmp.mp4"
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
    for k, (nm, tg) in enumerate(cells):
        Xs = np.stack([h5py.File(f"{R}/{tg}/h5_fem/sim_{i:010d}.h5", "r")["x"][:][sel] for i in range(126)])   # 렌더 입력 (프레임 0 = 정답 시작)
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
    o2 = f"{OUT}/fem_r_ficus_period.png"
    fig.savefig(o2, dpi=130)
    print("[저장]", o2)
print("[저장]", o1)
