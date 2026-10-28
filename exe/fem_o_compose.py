"""절 O 영상·그림 합성 (로컬 ffmpeg + matplotlib, Noto CJK KR; % 는 textfile + expansion=none -- exe/fem_ex_compose.py).
  1. lego 3DGS: 정답 | 암시적 최종 (k_final_F) | 명시적 새 판 (lag + 속도 단위 바닥)          (칸마다 ms·fps·오차)
  2. ficus 3DGS: 정답 | 암시적 최종 (실시간 + 수정 판정 rt_tmix_fix) | 명시적 새 판 (같은 코드 경로, 바닥 없음)
  3. ficus 프레임별 입자 평균 ψ (정답 · 암시적 최종 · 명시적 새 판, 로그축)
  4. ficus 잎·가지 끝 변위 시계열과 진동 주기 (z 상위 5 % 입자 평균, 정답 주 방향 -- /tmp/ficper/fig.py 와 같은 분석)
사용: python exe/fem_o_compose.py <lego 태그 또는 -> <ficus 태그>   (lbours/fem_o_{lego,ficus}/<태그>)"""
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
TMP = "/tmp/fem_o_compose"
os.makedirs(TMP, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
LEGO = sys.argv[1] != "-"                                                           # "-": lego 완주 판 없음 -> lego 영상 건너뜀
LO = f"{L}/fem_o_lego/{sys.argv[1]}"
FO = f"{L}/fem_o_ficus/{sys.argv[2]}"
SL = json.load(open(f"{LO}/summary.json")) if LEGO else None
SF = json.load(open(f"{FO}/summary.json"))
SFI = json.load(open(f"{L}/fem_rt_ficus/rt_tmix_fix/summary.json"))
SLI = json.load(open(f"{L}/fem_rt_lego/k_final_F/summary.json"))
VN = lambda s: f"lag 서브스텝 {s['N']}" + (f", 속도 바닥 고착 {'질량 비율' if s.get('vfl_mode') == 'frac' else '노드 전체'}" if s.get("vfl_mode") else "") \
    + (", 아핀 전달" if s.get("affine") else "")
outs = []
# 1
o1 = f"{OUT}/fem_o_lego_cmp.mp4"
LEGO and subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o1,
                f"lego L000 80 프레임 · A6000 단독 · 정답 | 암시적 최종 (뉴턴 1 + 혼합 PCG) | 명시적 새 판 ({VN(SL)}, 사면체 등분 질량)",
                f"정답 (MPM)|-|{L}/fem2_lego/gt_lego.mp4",
                f"암시적 FEM 최종|{L}/fem_rt_lego/k_final_F/summary.json|{L}/fem_rt_lego/k_final_F/vj/fem_lego.mp4",
                f"명시적 새 판 (속도 바닥, N {SL['N']})|{LO}/summary.json|{LO}/vj/fem_lego.mp4"], check=True)
LEGO and outs.append(o1)
# 2
o2 = f"{OUT}/fem_o_ficus_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o2,
                f"ficus F_pg_v0 125 프레임 · A6000 단독 · 정답 | 암시적 최종 (실시간 + 수정 판정) | 명시적 새 판 (lego 와 같은 코드 경로, {VN(SF)}, 바닥 없음)",
                f"정답 (PG)|-|{L}/fem3_ficus/gt_ficus.mp4",
                f"암시적 실시간 + 수정 판정|{L}/fem_rt_ficus/rt_tmix_fix/summary.json|{L}/fem_rt_ficus/rt_tmix_fix/v/fem_ficus.mp4",
                f"명시적 새 판 (N {SF['N']})|{FO}/summary.json|{FO}/vj/fem_ficus.mp4"], check=True)
outs.append(o2)
# 3
F_ = os.path.expanduser("~/.fonts/NotoSerifCJKkr-Bold.otf")
fm.fontManager.addfont(F_)
plt.rcParams.update({"font.family": fm.FontProperties(fname=F_).get_name(), "axes.unicode_minus": False})
mean_curve = lambda p: np.array([a_.mean() for a_ in np.load(p, mmap_mode="r")])
C6 = np.load(f"{L}/fem_rt_ficus/rt_tmix_fix/psi/curves6.npz")
exo = mean_curve(f"{FO}/psi/psi_ov.npy")
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.plot(np.arange(len(C6["gt"])), C6["gt"], color="#222222", lw=2.4, label="정답 (PG)")
ax.plot(np.arange(len(C6["m_rtfix"])), C6["m_rtfix"], color="#7b3fbf", lw=1.6,
        label=f"암시적 최종 (실시간 + 수정 판정, {SFI['ms_mean']:.1f} ms, 오차 {100 * SFI['err_mean']:.3f}%)")
ax.plot(np.arange(len(exo)), exo, color="#e04a1f", lw=2.4, label=f"명시적 새 판 ({VN(SF)}, {SF['ms_mean']:.1f} ms, 오차 {100 * SF['err_mean']:.3f}%)")
ax.set_yscale("log")
ax.set_xlabel("프레임")
ax.set_ylabel("입자 평균 탄성 에너지 밀도 ψ [Pa] (로그)")
ax.set_title("ficus F_pg_v0 · 프레임별 입자 평균 ψ (전 입자 171,553 개, 가중 없음)")
ax.grid(alpha=.3, which="both")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
o3 = f"{OUT}/fem_o_ficus_psi_curve.png"
fig.savefig(o3, dpi=130)
plt.close(fig)
outs.append(o3)
for nm, c in (("정답", C6["gt"]), ("암시적 최종", C6["m_rtfix"]), ("명시적 새 판", exo)):
    print(f"ψ {nm}: 프레임 1 {c[1]:.4g} 40 {c[40]:.4g} 125 {c[125]:.4g} 평균(1~125) {c[1:].mean():.4g}")
# 4 진동 주기 (/tmp/ficper/fig.py 와 같은 분석: z 상위 5 % 입자, 정답 주 방향 d, 영점 교차 앞 8 개 간격 x 2)
dt = 0.04
X0 = np.load("/tmp/ficper/x_gt.npy", mmap_mode="r")
x0 = np.array(X0[0])
sel = np.where(x0[:, 2] >= np.quantile(x0[:, 2], 0.95))[0]
d = np.array([0.977, -0.21, -0.016])
t = np.arange(126) * dt


def load_h5(path):
    return np.stack([h5py.File(f"{path}/sim_{i:010d}.h5", "r")["x"][:][sel] for i in range(126)])


def ana(u):
    s = u - u[63:].mean()
    zc = [t[i - 1] + s[i - 1] / (s[i - 1] - s[i]) * dt for i in range(1, len(s)) if s[i - 1] * s[i] < 0]
    T = 2 * np.mean(np.diff(zc[:8])) if len(zc) >= 3 else np.nan
    n = 8192
    Fq = np.abs(np.fft.rfft(s * np.hanning(len(s)), n))
    f = np.fft.rfftfreq(n, dt)
    return T, 1 / f[1:][np.argmax(Fq[1:])], np.abs(u).max(), len(zc)


series = {"gt": np.array(X0[:, sel, :]), "rt": np.array(np.load("/tmp/ficper/x_rt_fix.npy", mmap_mode="r")[:, sel, :]), "new": load_h5(f"{FO}/h5_fem")}
labs = {"gt": "정답 (PG MPM)", "rt": "암시적 최종 (실시간 + 수정 판정)", "new": f"명시적 새 판 ({VN(SF)})"}
cols = {"gt": "#0b0b0b", "rt": "#1baf7a", "new": "#e04a1f"}
fig, ax = plt.subplots(figsize=(11, 5.6))
res = {}
for k, X in series.items():
    u = (X.mean(1) - x0[sel].mean(0)) @ d
    res[k] = ana(u)
    ax.plot(t, u, color=cols[k], lw=2.4 if k != "rt" else 1.6, label=f"{labs[k]}: 주기 {res[k][0]:.2f} s")
ax.axhline(0, color="#bbb", lw=0.8)
ax.set_xlabel("시간 (s)")
ax.set_ylabel("주 진동 방향 변위 (z 상위 5 % 입자 평균, m)")
ax.set_title(f"ficus 잎·가지 끝 변위 시계열 · 정답 주기 {res['gt'][0]:.2f} s, 명시적 새 판 {res['new'][0]:.2f} s (비 {res['new'][0] / res['gt'][0]:.2f})")
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.grid(alpha=.25)
fig.tight_layout()
o4 = f"{OUT}/fem_o_ficus_period.png"
fig.savefig(o4, dpi=130)
plt.close(fig)
outs.append(o4)
for k in series:
    T, Tf, A, nz = res[k]
    print(f"주기 {labs[k]}: 영점 {T:.3f} s, FFT {Tf:.3f} s, 정답 대비 {T / res['gt'][0]:.2f}, 최대 진폭 {A:.4f} m (비 {A / res['gt'][2]:.2f}), 영점 {nz}")
json.dump({k: dict(T_zc=res[k][0], T_fft=res[k][1], amp=res[k][2], nzc=res[k][3]) for k in res}, open(f"{FO}/period.json", "w"), ensure_ascii=False, indent=1)
for p in outs:
    print("[저장]", p)
