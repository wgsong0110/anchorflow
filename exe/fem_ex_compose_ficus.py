"""절 L ficus 명시적 판 합성 (로컬 ffmpeg + matplotlib, Noto CJK KR):
  1. 3DGS 정답 | 암시적 실시간 (+수정 판정) | 명시적 (칸마다 ms·fps·오차)  -- exe/fem_ex_compose.py
  2. 프레임별 입자 평균 ψ (정답 · 기존 CG · PCG · 수정 판정 · 암시적 실시간 · 암시적 실시간+수정 판정 · 명시적, 로그축)
  3. ψ 색 영상 (정답 | 수정 판정 (입자 적분점) | 암시적 실시간+수정 판정 | 명시적, 공통 고정 범위 log10(ψ+1e-3) -3.000~4.933 turbo, 칸별 정규화 없음)"""
import json, os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

L = os.path.expanduser("~/r2/result/anchorflow/lbours")
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
TMP = "/tmp/fem_ex_ficus"
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
os.makedirs(TMP, exist_ok=True)
EXD = f"{L}/fem_ex_ficus/f_final_F"
IMP = f"{L}/fem_rt_ficus/rt_tmix_fix"
S = json.load(open(f"{EXD}/summary.json"))
SI = json.load(open(f"{IMP}/summary.json"))
here = os.path.dirname(os.path.abspath(__file__))


def tf(name, text):
    p = f"{TMP}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(textfile, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={textfile}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


# 1
cmp_out = f"{OUT}/fem_ex_ficus_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", cmp_out,
                f"ficus F_pg_v0 125 프레임 · A6000 단독 · 정답 | 암시적 실시간 + 수정 판정 (뉴턴 1, 혼합 PCG) | 명시적 (라그랑주 서브스텝 {S['N']}, 사면체 등분 질량 -- 정의 변경)",
                f"정답 (PG)|-|{L}/fem3_ficus/gt_ficus.mp4", f"암시적 FEM (실시간 + 수정 판정)|{IMP}/summary.json|{IMP}/v/fem_ficus.mp4",
                f"명시적 FEM (서브스텝 {S['N']})|{EXD}/summary.json|{EXD}/vj/fem_ficus.mp4"], check=True)
# 2
F_ = os.path.expanduser("~/.fonts/NotoSerifCJKkr-Bold.otf")
fm.fontManager.addfont(F_)
plt.rcParams.update({"font.family": fm.FontProperties(fname=F_).get_name(), "axes.unicode_minus": False})
C6 = np.load(f"{IMP}/psi/curves6.npz")
ex = np.array([a_.mean() for a_ in np.load(f"{EXD}/psi/psi_ov.npy", mmap_mode="r")])
ERR = {k: open(p).read().split()[-1] for k, p in (("cg", f"{L}/fem_ficus/run/err_fem.txt"), ("pcg", f"{L}/fem2_ficus/run/err_fem.txt"),
                                                   ("fix", f"{L}/fem3_ficus/run/err_fem.txt"))}
cur = [("m_cg", f"FEM 기존 CG (오차 {ERR['cg']})", "#d1495b", 1.4), ("m_pcg", f"FEM 기존 PCG (오차 {ERR['pcg']})", "#2e86ab", 1.4),
       ("m_fix", f"FEM PCG 수정 판정 (오차 {ERR['fix']})", "#3b9d5d", 1.4), ("m_rt", "암시적 실시간 판 (오차 2.163%)", "#e08a00", 1.4),
       ("m_rtfix", f"암시적 실시간 + 수정 판정 (오차 {100 * SI['err_mean']:.3f}%)", "#7b3fbf", 1.6)]
fig, ax = plt.subplots(figsize=(10, 5.6))
fr = np.arange(len(C6["gt"]))
ax.plot(fr, C6["gt"], color="#222222", lw=2.4, label="정답 (PG)")
for k, lb, c, w in cur:
    ax.plot(np.arange(len(C6[k])), C6[k], color=c, lw=w, label=lb)
ax.plot(np.arange(len(ex)), ex, color="#0aa5a5", lw=2.6, label=f"명시적 (서브스텝 {S['N']}, 오차 {100 * S['err_mean']:.3f}%)")
ax.set_yscale("log")
ax.set_xlabel("프레임")
ax.set_ylabel("입자 평균 탄성 에너지 밀도 ψ [Pa] (로그)")
ax.set_title("ficus F_pg_v0 · 프레임별 입자 평균 ψ (전 입자 171,553 개, 가중 없음)")
ax.grid(alpha=.3, which="both")
ax.legend(loc="lower right", fontsize=8.5)
fig.tight_layout()
curve_out = f"{OUT}/fem_ex_ficus_psi_curve.png"
fig.savefig(curve_out, dpi=130)
np.savez(f"{TMP}/curves7.npz", ex=ex, **{k: C6[k] for k in C6.files})
print(f"ψ 명시적: 프레임 1 {ex[1]:.4g} 40 {ex[40]:.4g} 125 {ex[125]:.4g} 평균 {ex[1:].mean():.4g} | 정답 {C6['gt'][1]:.4g} {C6['gt'][40]:.4g} {C6['gt'][125]:.4g} {C6['gt'][1:].mean():.4g}"
      f" | 암시적+수정 {C6['m_rtfix'][1]:.4g} {C6['m_rtfix'][40]:.4g} {C6['m_rtfix'][125]:.4g}")
# 3
CB = f"{TMP}/cbar4.png"
_f, _a = plt.subplots(figsize=(12, 1.0))
_f.subplots_adjust(left=0.06, right=0.94, bottom=0.42, top=0.62)
_cb = matplotlib.colorbar.ColorbarBase(_a, cmap=plt.get_cmap("turbo"), norm=matplotlib.colors.Normalize(-3.0, 4.933), orientation="horizontal")
_cb.set_ticks([-3, -2, -1, 0, 1, 2, 3, 4, 4.933]); _cb.set_ticklabels(["-3.00", "-2", "-1", "0", "1", "2", "3", "4", "4.93"])
_a.set_title("log10(ψ + 0.001),  ψ = 고정 FCR 탄성 에너지 밀도 [Pa]  ·  네 칸 공통 고정 범위 -3.00 ~ 4.93, 범위 밖은 끝 색", fontsize=10)
_f.savefig(CB, dpi=240, facecolor="white"); plt.close(_f)
vids = [f"{L}/ficus_v0/psi/psi_gt_ficus.mp4", f"{L}/fem3_ficus/psi/psi_fem_ficus.mp4", f"{IMP}/psi/psi_fem_ficus.mp4", f"{EXD}/psi/psi_fem_ficus.mp4"]
labs = ["정답 (PG)", f"입자 적분점 수정 판정 ({ERR['fix']})", f"암시적 실시간+수정 ({100 * SI['err_mean']:.3f}%)", f"명시적 서브스텝 {S['N']} ({100 * S['err_mean']:.3f}%)"]
ins, fl = [], []
for i, (v_, l_) in enumerate(zip(vids, labs)):
    ins += ["-i", v_]
    fl.append(f"[{i}:v]scale=600:600,pad=600:650:0:50:white," + dt(tf(f"p{i}", l_), FB, 28, "(w-tw)/2", 10) + f"[p{i}]")
ttl = tf("ttl", "ficus 탄성 에너지 밀도 ψ 색 · log10(ψ+1e-3) 공통 고정 범위 −3.000 ~ 4.933 (turbo) · 3DGS 렌더")
ins += ["-loop", "1", "-framerate", "30", "-i", CB]
fl.append("".join(f"[p{i}]" for i in range(4)) + "hstack=inputs=4,pad=iw:ih+70:0:70:white," + dt(ttl, FB, 30, 20, 18)
          + ",drawtext=fontfile=" + FB + ":text='프레임 %{eif\\:n\\:d}':fontsize=50:fontcolor=red:x=w-tw-30:y=8[top]")
fl.append("[4:v]scale=2400:200,format=rgb24[cb];[top][cb]vstack=inputs=2[o]")
psi_out = f"{OUT}/fem_ex_ficus_psi_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-frames:v", "126",
                "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", psi_out], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "2", "-i", psi_out, "-frames:v", "1", f"{TMP}/chk_psi.png"], check=True)
for p in (cmp_out, curve_out, psi_out):
    print("[저장]", p)
