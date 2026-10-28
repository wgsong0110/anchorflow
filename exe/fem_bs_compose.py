"""절 M 영상·그림 합성 (로컬 ffmpeg + matplotlib, Noto CJK KR; % 는 textfile + expansion=none -- exe/fem_ex_compose.py).
  1. lego 3DGS: 정답 | 암시적 최종 (k_final_F) | 명시적 사면체 등분 질량 (l_final_F) | 명시적 2 차 B-스플라인   (칸마다 ms·fps·오차)
  2. ficus 3DGS: 정답 | 암시적 실시간 + 수정 판정 (rt_tmix_fix) | 명시적 사면체 등분 (f_final_F) | 명시적 B-스플라인
  3. ficus 프레임별 입자 평균 ψ (정답 · 암시적 · 명시적 사면체 등분 · 명시적 B-스플라인, 로그축)
  4. ficus ψ 색 영상 (같은 네 칸, 공통 고정 범위 log10(ψ+1e-3) −3.000~4.933 turbo -- 절 J·L 과 같은 범위, 칸별 정규화 없음)
사용: python exe/fem_bs_compose.py <lego B-스플라인 태그> <ficus B-스플라인 태그>   (lbours/fem_bs_{lego,ficus}/<태그>)"""
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

L = os.path.expanduser("~/r2/result/anchorflow/lbours")
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
TMP = "/tmp/fem_bs_compose"
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
os.makedirs(TMP, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
LB = f"{L}/fem_bs_lego/{sys.argv[1]}"
FBS = f"{L}/fem_bs_ficus/{sys.argv[2]}"
SL, SF = json.load(open(f"{LB}/summary.json")), json.load(open(f"{FBS}/summary.json"))
SLT, SFT = json.load(open(f"{L}/fem_ex_lego/l_final_F/summary.json")), json.load(open(f"{L}/fem_ex_ficus/f_final_F/summary.json"))
SFI = json.load(open(f"{L}/fem_rt_ficus/rt_tmix_fix/summary.json"))
QN = lambda s: "입자 적분점" if s.get("quad") == "particle" else f"셀 대표 K {s.get('K')}"


def tf(name, text):
    p = f"{TMP}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(textfile, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={textfile}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


outs = []
# 1
o1 = f"{OUT}/fem_bs_lego_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o1,
                f"lego L000 80 프레임 · A6000 단독 · 정답 | 암시적 최종 (뉴턴 1 + 혼합 PCG) | 명시적 사면체 등분 질량 (서브스텝 {SLT['N']}) | 명시적 2 차 B-스플라인 ({QN(SL)}, 서브스텝 {SL['N']}, 질량 Σwm)",
                f"정답 (MPM)|-|{L}/fem2_lego/gt_lego.mp4",
                f"암시적 FEM 최종|{L}/fem_rt_lego/k_final_F/summary.json|{L}/fem_rt_lego/k_final_F/vj/fem_lego.mp4",
                f"명시적 사면체 등분 (N {SLT['N']})|{L}/fem_ex_lego/l_final_F/summary.json|{L}/fem_ex_lego/l_final_F/vj/fem_lego.mp4",
                f"명시적 B-스플라인 (N {SL['N']})|{LB}/summary.json|{LB}/vj/fem_lego.mp4"], check=True)
outs.append(o1)
# 2
o2 = f"{OUT}/fem_bs_ficus_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o2,
                f"ficus F_pg_v0 125 프레임 · A6000 단독 · 정답 | 암시적 실시간 + 수정 판정 | 명시적 사면체 등분 질량 (라그랑주 {SFT['N']}) | 명시적 2 차 B-스플라인 ({QN(SF)}, 서브스텝 {SF['N']}, 질량 Σwm)",
                f"정답 (PG)|-|{L}/fem3_ficus/gt_ficus.mp4",
                f"암시적 실시간 + 수정 판정|{L}/fem_rt_ficus/rt_tmix_fix/summary.json|{L}/fem_rt_ficus/rt_tmix_fix/v/fem_ficus.mp4",
                f"명시적 사면체 등분 (N {SFT['N']})|{L}/fem_ex_ficus/f_final_F/summary.json|{L}/fem_ex_ficus/f_final_F/vj/fem_ficus.mp4",
                f"명시적 B-스플라인 (N {SF['N']})|{FBS}/summary.json|{FBS}/vj/fem_ficus.mp4"], check=True)
outs.append(o2)
# 3
F_ = os.path.expanduser("~/.fonts/NotoSerifCJKkr-Bold.otf")
fm.fontManager.addfont(F_)
plt.rcParams.update({"font.family": fm.FontProperties(fname=F_).get_name(), "axes.unicode_minus": False})
mean_curve = lambda p: np.array([a_.mean() for a_ in np.load(p, mmap_mode="r")])
C6 = np.load(f"{L}/fem_rt_ficus/rt_tmix_fix/psi/curves6.npz")
ext = mean_curve(f"{L}/fem_ex_ficus/f_final_F/psi/psi_ov.npy")
exb = mean_curve(f"{FBS}/psi/psi_ov.npy")
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.plot(np.arange(len(C6["gt"])), C6["gt"], color="#222222", lw=2.4, label="정답 (PG)")
ax.plot(np.arange(len(C6["m_rtfix"])), C6["m_rtfix"], color="#7b3fbf", lw=1.6, label=f"암시적 실시간 + 수정 판정 (오차 {100 * SFI['err_mean']:.3f}%)")
ax.plot(np.arange(len(ext)), ext, color="#0aa5a5", lw=2.0, label=f"명시적 사면체 등분 질량 (N {SFT['N']}, 오차 {100 * SFT['err_mean']:.3f}%)")
ax.plot(np.arange(len(exb)), exb, color="#e04a1f", lw=2.4, label=f"명시적 B-스플라인 {QN(SF)} (N {SF['N']}, 오차 {100 * SF['err_mean']:.3f}%)")
ax.set_yscale("log")
ax.set_xlabel("프레임")
ax.set_ylabel("입자 평균 탄성 에너지 밀도 ψ [Pa] (로그)")
ax.set_title("ficus F_pg_v0 · 프레임별 입자 평균 ψ (전 입자 171,553 개, 가중 없음)")
ax.grid(alpha=.3, which="both")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
o3 = f"{OUT}/fem_bs_ficus_psi_curve.png"
fig.savefig(o3, dpi=130)
outs.append(o3)
for nm, c in (("정답", C6["gt"]), ("암시적", C6["m_rtfix"]), ("명시적 사면체 등분", ext), ("명시적 B-스플라인", exb)):
    print(f"ψ {nm}: 프레임 1 {c[1]:.4g} 40 {c[40]:.4g} 125 {c[125]:.4g} 평균(1~125) {c[1:].mean():.4g}")
# 4
CB = f"{TMP}/cbar4.png"
_f, _a = plt.subplots(figsize=(12, 1.0))
_f.subplots_adjust(left=0.06, right=0.94, bottom=0.42, top=0.62)
_cb = matplotlib.colorbar.ColorbarBase(_a, cmap=plt.get_cmap("turbo"), norm=matplotlib.colors.Normalize(-3.0, 4.933), orientation="horizontal")
_cb.set_ticks([-3, -2, -1, 0, 1, 2, 3, 4, 4.933]); _cb.set_ticklabels(["-3.00", "-2", "-1", "0", "1", "2", "3", "4", "4.93"])
_a.set_title("log10(ψ + 0.001),  ψ = 고정 FCR 탄성 에너지 밀도 [Pa]  ·  네 칸 공통 고정 범위 -3.00 ~ 4.93, 범위 밖은 끝 색", fontsize=10)
_f.savefig(CB, dpi=240, facecolor="white"); plt.close(_f)
vids = [f"{L}/ficus_v0/psi/psi_gt_ficus.mp4", f"{L}/fem_rt_ficus/rt_tmix_fix/psi/psi_fem_ficus.mp4", f"{L}/fem_ex_ficus/f_final_F/psi/psi_fem_ficus.mp4",
        f"{FBS}/psi/psi_fem_ficus.mp4"]
labs = ["정답 (PG)", f"암시적 실시간+수정 ({100 * SFI['err_mean']:.3f}%)", f"명시적 사면체 등분 N {SFT['N']} ({100 * SFT['err_mean']:.3f}%)",
        f"명시적 B-스플라인 N {SF['N']} ({100 * SF['err_mean']:.3f}%)"]
ins, fl = [], []
for i, (v_, l_) in enumerate(zip(vids, labs)):
    ins += ["-i", v_]
    fl.append(f"[{i}:v]scale=600:600,pad=600:650:0:50:white," + dt(tf(f"p{i}", l_), FB, 26, "(w-tw)/2", 10) + f"[p{i}]")
ttl = tf("ttl", "ficus 탄성 에너지 밀도 ψ 색 · log10(ψ+1e-3) 공통 고정 범위 −3.000 ~ 4.933 (turbo) · 3DGS 렌더")
ins += ["-loop", "1", "-framerate", "30", "-i", CB]
fl.append("".join(f"[p{i}]" for i in range(4)) + "hstack=inputs=4,pad=iw:ih+70:0:70:white," + dt(ttl, FB, 30, 20, 18)
          + ",drawtext=fontfile=" + FB + ":text='프레임 %{eif\\:n\\:d}':fontsize=50:fontcolor=red:x=w-tw-30:y=8[top]")
fl.append("[4:v]scale=2400:200,format=rgb24[cb];[top][cb]vstack=inputs=2[o]")
o4 = f"{OUT}/fem_bs_ficus_psi_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-frames:v", "126",
                "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", o4], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "2", "-i", o4, "-frames:v", "1", f"{TMP}/chk_psi.png"], check=True)
outs.append(o4)
for p in outs:
    print("[저장]", p)
