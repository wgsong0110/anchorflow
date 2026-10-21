"""ficus C¹ N1 최적화 변수 θ 기준 기울기 (AF_VARGRAD) 정리: 그림·표.

입력: <vargrad>/{c1,bary,basenone}/vargrad.jsonl, C¹ N1 실행 inv_ov.jsonl (L-BFGS 통계; 종료 사유 추정)
출력: --png, --md
종료 사유 (ov_tol 0 이라 수렴 판정은 꺼져 있다):
  feas_lbfgs (C¹): 평가 수 >= 상한 -> 평가 상한, 반복 수 >= 상한 -> 반복 상한, 그 밖 -> 선 탐색 실패 (되추적 40 회 소진)
  torch L-BFGS (선형): 반복 >= 상한 -> 반복 상한, 평가 >= 상한 -> 평가 상한, 그 밖 -> 조기 종료 (걸음 0·비하강 = 선 탐색 실패)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

for f_ in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf",):
    if os.path.exists(f_):
        fm.fontManager.addfont(f_)
        plt.rcParams["font.family"] = [fm.FontProperties(fname=f_).get_name(), "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--vg", required=True)
ap.add_argument("--inv", required=True, help="C¹ N1 inv_ov.jsonl")
ap.add_argument("--png", required=True)
ap.add_argument("--md", required=True)
ap.add_argument("--tex", default="")
ap.add_argument("--c1dir", default="", help="C¹ vargrad.jsonl 디렉토리 (기본 <vg>/c1)")
ap.add_argument("--c1name", default="C¹ pl (N1)")
ap.add_argument("--title", default="ficus C$^1$ N1 최적화 변수 기준 기울기 (2026-10-21)")
a = ap.parse_args()


def load(p):
    if not os.path.exists(p):
        return []
    R = [json.loads(l) for l in open(p)]
    return sorted(R, key=lambda r: r["t"])


C = load(f"{a.c1dir or a.vg + '/c1'}/vargrad.jsonl")
B = load(f"{a.vg}/bary/vargrad.jsonl")
N = load(f"{a.vg}/basenone/vargrad.jsonl")
INV = {r["frame"]: r for r in (json.loads(l) for l in open(a.inv))} if os.path.exists(a.inv) else {}

RS = ["수렴 tol", "반복 상한", "평가 상한", "선 탐색 실패"]


def reason(o):
    if o is None:
        return "?"
    if "stop" in o:                       # feas_lbfgs 가 기록한 종료 사유 (8f602d1 이후)
        st_ = str(o["stop"])
        return {"max_iter": "반복 상한", "max_eval": "평가 상한", "grad_tol": "수렴 tol", "change_tol": "수렴 tol"}.get(
            st_, "선 탐색 실패" if st_.startswith("ls_fail") else st_)
    if o.get("kind") == "torch_lbfgs":
        if o["iters"] >= o["max_iter"]:
            return "반복 상한"
        if o["evals"] >= o["max_eval"]:
            return "평가 상한"
        return "선 탐색 실패"
    if o.get("evals", 0) >= o.get("max_eval", 625):
        return "평가 상한"
    if o.get("iters", 0) >= o.get("max_iter", 500):
        return "반복 상한"
    return "선 탐색 실패"


def opt_c1(t):
    r = INV.get(t + 1)
    if r is None or "opt" not in r:
        return None
    return dict(r["opt"], max_iter=500, max_eval=625)


def row(r, o):
    s, z = r["sol"], r["start"]
    nw = s.get("newton", {})
    ch = r.get("chk", {})
    sr = r.get("sol_run") or s
    # 기울기 검산은 시작점에서 (기울기가 커서 float32 에너지 차분의 반올림이 묻히지 않는다; 해에서는 차분이 잡음 수준)
    gfr = ((r.get("start_run") or z).get("grad_fd") or {}).get("rand", {})
    gft = (z.get("grad_fd") or {}).get("rand", {})
    return dict(g_run=sr["g"]["all"][0], ratio_run=r.get("ratio_run", r["ratio"]),
                gerr_run=abs(gfr.get("ad", np.nan) - gfr.get("fd", np.nan)) / max(abs(gfr.get("fd", np.nan)), 1e-30),
                gerr=abs(gft.get("ad", np.nan) - gft.get("fd", np.nan)) / max(abs(gft.get("fd", np.nan)), 1e-30),
                fr=r["t"] + 1, g=s["g"]["all"][0], ginf=s["g"]["all"][1], g0=z["g"]["all"][0], ratio=r["ratio"],
                gv=s["g"].get("val", [np.nan])[0], gg=s["g"].get("grad", [np.nan])[0], ge=s["g"].get("edge", [np.nan])[0],
                nd=nw.get("half_lam2", np.nan), neg=nw.get("neg_curv", False), cg=nw.get("cg_iters", 0), res=nw.get("res_rel", np.nan),
                drop1=(nw.get("actual_drop") or {}).get("1", np.nan), drop_max=max((nw.get("actual_drop") or {"0": np.nan}).values()),
                E=s["E"], Eel=s["E_el"], E0=z["E"], fd=min((s.get("hvp_fd_rel") or {"0": np.nan}).values()),
                xchk=ch.get("x_maxabs", np.nan), Fchk=ch.get("F_maxabs", np.nan), dchk=ch.get("dpn_maxabs", np.nan),
                grid=r.get("chk_grid", np.nan), it=(o or {}).get("iters", -1), ev=(o or {}).get("evals", -1), why=reason(o),
                nvar=s["n_var"])


RC = [row(r, opt_c1(r["t"])) for r in C]
RB = [row(r, r.get("opt")) for r in B]
RN = [row(r, r.get("opt")) for r in N]

# ---------------------------------------------------------------- 그림
COL = {"C¹ (pl)": "#2a78d6", "선형": "#eb6834", "C¹ none 첫 스텝": "#6250d6"}
WC = {"수렴 tol": "#2a78d6", "반복 상한": "#1baf7a", "평가 상한": "#e34948", "선 탐색 실패": "#6250d6", "?": "#9a9a96"}
fig, ax = plt.subplots(4, 1, figsize=(11, 13), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 3, 1]))


def ser(R, k):
    return np.array([q["fr"] for q in R]), np.array([q[k] for q in R], float)


for nm, R in (("C¹ (pl)", RC), ("선형", RB)):
    if not R:
        continue
    c = COL[nm]
    x_, y_ = ser(R, "g"); ax[0].plot(x_, y_, color=c, lw=2, label=f"{nm} 해 ‖g‖₂")
    x_, y_ = ser(R, "g0"); ax[0].plot(x_, y_, color=c, lw=1.2, ls="--", label=f"{nm} 시작점 ‖g₀‖₂")
    x_, y_ = ser(R, "ratio"); ax[1].plot(x_, y_, color=c, lw=2, label=nm + (" (참 기울기)" if R is RC else ""))
    if R is RC:
        x_, y_ = ser(R, "g_run"); ax[0].plot(x_, y_, color=c, lw=1, ls="-.", alpha=0.8, label=f"{nm} 해 ‖g‖₂ (L-BFGS 가 본 기울기)")
        x_, y_ = ser(R, "ratio_run"); ax[1].plot(x_, y_, color=c, lw=1, ls="-.", alpha=0.8, label=f"{nm} (L-BFGS 가 본 기울기)")
    x_, y_ = ser(R, "nd"); ax[2].plot(x_, y_, color=c, lw=2, label=f"{nm} 뉴턴 감소량 ½λ²")
    neg = np.array([q["neg"] for q in R])
    if neg.any():
        ax[2].scatter(x_[neg], y_[neg], s=40, facecolor="white", edgecolor=c, lw=1.5, zorder=3, label=f"{nm} 음곡률에서 CG 멈춤")
    x_, y_ = ser(R, "Eel"); ax[2].plot(x_, y_, color=c, lw=1.2, ls=":", label=f"{nm} 탄성 에너지")
for nm, R in (("C¹ none 첫 스텝", RN),):
    for q in R:
        c = COL[nm]
        ax[0].scatter([q["fr"]], [q["g"]], marker="D", s=50, color=c, zorder=4, label=f"{nm} 해 ‖g‖₂")
        ax[1].scatter([q["fr"]], [q["ratio"]], marker="D", s=50, color=c, zorder=4, label=nm)
        ax[2].scatter([q["fr"]], [q["nd"]], marker="D", s=50, color=c, zorder=4, label=f"{nm} ½λ²")
        ax[2].scatter([q["fr"]], [q["Eel"]], marker="d", s=50, facecolor="white", edgecolor=c, zorder=4, label=f"{nm} 탄성")
ax[0].set_ylabel("‖∂E/∂θ‖₂ [J/변수 단위]"); ax[0].set_title("해에서의 변수 기울기 (변수 정의가 달라 C¹·선형 크기는 직접 비교 불가)")
ax[1].set_ylabel("‖g‖ / ‖g₀‖"); ax[1].set_title("시작점 대비 기울기 감소 비")
ax[2].set_ylabel("에너지 [J]"); ax[2].set_title("이 스텝에서 아직 내려갈 수 있는 에너지 (½ gᵀH⁻¹g, CG) 와 탄성 에너지")
for k in range(3):
    ax[k].set_yscale("log"); ax[k].grid(True, which="major", color="#e4e3df", lw=0.6); ax[k].legend(fontsize=8, ncol=2, frameon=False)
for j, (nm, R) in enumerate((("C¹ (pl)", RC), ("선형", RB))):
    for q in R:
        ax[3].add_patch(plt.Rectangle((q["fr"] - 0.5, j - 0.4), 1.0, 0.8, color=WC.get(q["why"], "#9a9a96"), lw=0))
ax[3].set_yticks([0, 1]); ax[3].set_yticklabels(["C¹ (pl)", "선형"]); ax[3].set_ylim(-0.6, 1.6)
ax[3].set_title("L-BFGS 종료 사유 (500 반복 / 625 평가 상한, tol 0)")
for w, c in WC.items():
    if w != "?":
        ax[3].bar([0], [0], color=c, label=w)
ax[3].legend(fontsize=8, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.45))
ax[3].set_xlabel("프레임 (스텝 t → t+1 의 결과 프레임 t+1)"); ax[3].set_xlim(0.5, 125.5)
fig.tight_layout(); fig.savefig(a.png, dpi=130); print("[그림]", a.png)


# ---------------------------------------------------------------- 표
def med(R, k):
    v = np.array([q[k] for q in R], float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def cnt(R):
    from collections import Counter
    c = Counter(q["why"] for q in R)
    return ", ".join(f"{k} {c[k]}" for k in RS if c[k])


L = ["# " + a.title.replace("C$^1$", "C¹").replace("\\_", "_"), "",
     "측정 지점: 스텝 t (프레임 t+1 을 만드는 스텝) 마다, 프레임 t 상태(덤프)에서 그 스텝용으로 깐 격자 위에 실행이 실제로 쓴 해 θ*_t 를 "
     "넣은 점에서 ∂E_t/∂θ. E_t 는 L-BFGS 가 내리는 목적함수 그대로(증분 포텐셜 + Bernstein det 장벽 등 켜진 항 전부, J). "
     "g₀ 는 같은 스텝의 L-BFGS 시작점(관성 예측 초기화, 계수 ≤ 0 이면 1 단계 사영 뒤). "
     "뉴턴 감소량 ½λ² = ½gᵀH⁻¹g: 헤시안-벡터 곱(이중 역전파) + CG(최대 100 회, 양정치면 CG 값은 하한), 음곡률을 만나면 멈추고 그때 값.",
     ""]
if RC:
    L += [f"변수 벡터: C¹ pl {RC[0]['nvar']} 개 (자유 노드 × 54 = 값 3 + 기울기 9 + 변 42; 기울기·변은 PL 기저 위 보정), "
          + (f"선형 {RB[0]['nvar']} 개 (자유 노드 × 3 변위)" if RB else "") + ".", ""]
L += ["| 실행 | 스텝 수 | 해 ‖g‖₂ 중앙 | ‖g‖/‖g₀‖ 중앙 | ½λ² 중앙 (J) | ½λ² 최대 (J) | 탄성 중앙 (J) | ½λ²/탄성 중앙 | 음곡률 스텝 | 종료 사유 | 검산 위치 최대차 | 검산 F 최대차 |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for nm, R in ((a.c1name, RC), ("선형 (ficus_v0)", RB), ("C¹ none 첫 스텝 500 회", RN)):
    if not R:
        continue
    rat = [q["nd"] / q["Eel"] for q in R if q["Eel"] > 0]
    L.append(f"| {nm} | {len(R)} | {med(R, 'g'):.3e} | {med(R, 'ratio'):.3e} | {med(R, 'nd'):.3e} | {np.nanmax([q['nd'] for q in R]):.3e} | "
             f"{med(R, 'Eel'):.3e} | {float(np.nanmedian(rat)) if rat else float('nan'):.3e} | {sum(q['neg'] for q in R)} | {cnt(R)} | "
             f"{np.nanmax([q['xchk'] for q in R]):.2e} | {np.nanmax([q['Fchk'] for q in R]):.2e} |")
L += ["", "## 프레임별", "",
      "| 프레임 | C¹ ‖g‖ | C¹ ‖g₀‖ | C¹ 비 | C¹ ½λ² (J) | C¹ 탄성 (J) | C¹ 값/기울기/변 ‖g‖ | C¹ CG (음곡률) | C¹ 반복/평가 | C¹ 종료 | "
      "선형 ‖g‖ | 선형 비 | 선형 ½λ² (J) | 선형 탄성 (J) | 선형 반복/평가 | 선형 종료 |",
      "|" + "---|" * 16]
bm = {q["fr"]: q for q in RB}
cm = {q["fr"]: q for q in RC}
for fr in sorted(set(bm) | set(cm)):
    q, b = cm.get(fr), bm.get(fr)
    L.append(f"| {fr} | "
             + (f"{q['g']:.3e} | {q['g0']:.3e} | {q['ratio']:.2e} | {q['nd']:.3e} | {q['Eel']:.3e} | "
                f"{q['gv']:.2e}/{q['gg']:.2e}/{q['ge']:.2e} | {q['cg']}{' 음' if q['neg'] else ''} | {q['it']}/{q['ev']} | {q['why']} | "
                if q else "– | – | – | – | – | – | – | – | – | ")
             + (f"{b['g']:.3e} | {b['ratio']:.2e} | {b['nd']:.3e} | {b['Eel']:.3e} | {b['it']}/{b['ev']} | {b['why']} |" if b else "– | – | – | – | – | – |"))
L += ["", "## 검산", ""]
_BUG = bool(RC) and med(RC, "gerr_run") > 10 * max(med(RC, "gerr"), 1e-6)
if RC and not _BUG:
    L.append(f"- C¹ 기울기 방향 미분 검산(시작점) 상대차 중앙: L-BFGS 가 본 기울기 {med(RC, 'gerr_run'):.2e}, 참 기울기 {med(RC, 'gerr'):.2e} (둘이 같은 코드)")
if _BUG:
    L.append(f"- **C¹ pl 의 L-BFGS 기울기는 참 기울기가 아니다**: 시작점(기울기가 커서 float32 차분이 반올림에 안 묻히는 곳)의 임의 방향 d 에서 g·d 대 중앙차분 (E(θ+sd)-E(θ-sd))/2s 상대차 "
             f"— 실행 기울기 중앙 {med(RC, 'gerr_run'):.2e} (최대 {np.nanmax([q['gerr_run'] for q in RC]):.2e}), "
             f"중복 쓰기 원천 하나로 고친 참 기울기 중앙 {med(RC, 'gerr'):.2e}. 원인: WF.pl_c1_dofs 의 index_put(accumulate=False) 에서 "
             "같은 (노드, 변 채널) 자리를 여러 사면체가 같은 값으로 쓰는데, 역전파는 중복된 원천 모두에 기울기를 줘 값 채널 → 변 채널 경로의 "
             "기울기가 중복 수만큼 부풀려진다 (순전파 값은 같다; 기울기·변 채널 자체의 기울기는 맞다). 표·그림의 ‖g‖·비·½λ² 는 참 기울기 기준, "
             "'L-BFGS 가 본 기울기' 는 실행 그대로.")
    L.append(f"- C¹ 해 ‖g‖ 중앙: 참 {med(RC, 'g'):.3e}, L-BFGS 가 본 것 {med(RC, 'g_run'):.3e}; 비 중앙: 참 {med(RC, 'ratio'):.3e}, L-BFGS 가 본 것 {med(RC, 'ratio_run'):.3e}")
for nm, R in (("선형", RB), ("C¹ none", RN)):
    if R:
        L.append(f"- {nm} 기울기 방향 미분 검산(시작점) 상대차 중앙 {med(R, 'gerr'):.2e}")
for nm, R in (("C¹", RC), ("선형", RB)):
    if R:
        L.append(f"- {nm}: θ* 로 만든 노드 출력 최대차 {np.nanmax([q['dchk'] for q in R]):.2e}, 입자 위치 최대차(덤프 프레임 t+1) "
                 f"{np.nanmax([q['xchk'] for q in R]):.2e}, F 최대차 {np.nanmax([q['Fchk'] for q in R]):.2e}, 격자 노드 위치 최대차 "
                 f"{np.nanmax([q['grid'] for q in R]):.2e}; 헤시안-벡터 곱 대 유한차분 상대차 중앙 {med(R, 'fd'):.2e}")
open(a.md, "w").write("\n".join(L) + "\n")
print("[표]", a.md)

# ---------------------------------------------------------------- LaTeX 절 (--tex 가 있으면)
if a.tex:
    def f3(v):
        return f"{v:.3g}"
    T = [r"\section*{" + a.title + "}",
         r"스텝 $t$ (프레임 $t{+}1$ 을 만드는 스텝) 마다 프레임 $t$ 상태에 그 스텝용으로 깐 격자 위에서, 실행이 실제로 쓴 해 $\theta^*_t$ "
         r"(덤프 노드 출력 역변환; $\theta^*_t$ 로 만든 입자 위치·F 를 덤프 프레임 $t{+}1$ 과 대조) 의 $g=\partial E_t/\partial\theta$, "
         r"같은 스텝 L-BFGS 시작점의 $g_0$, 뉴턴 감소량 $\tfrac12 g^\top H^{-1}g$ (헤시안-벡터 곱 + CG 100 회, 하한). "
         r"$E_t$ 는 L-BFGS 목적함수 그대로 (J). 변수: C$^1$ pl " + (str(RC[0]["nvar"]) if RC else "-") + r" 개 (자유 노드 $\times$ 54), 선형 "
         + (str(RB[0]["nvar"]) if RB else "-") + r" 개 (자유 노드 $\times$ 3). 그림 \texttt{ovnet/ficus\_c1\_vargrad.png}, 표 \texttt{ovnet/ficus\_c1\_vargrad.md}.\\",
         r"{\small\begin{tabular}{lrrrrrrl}\toprule 실행 & 스텝 & $\|g\|/\|g_0\|$ 중앙 & $\tfrac12\lambda^2$ 중앙 (J) & $\tfrac12\lambda^2$ 최대 (J) & 탄성 중앙 (J) & $\tfrac12\lambda^2$/탄성 중앙 & 종료 사유 \\\midrule"]
    for nm, R in (("C$^1$ pl (N1)", RC), ("선형", RB), ("C$^1$ none 첫 스텝", RN)):
        if R:
            rat = [q["nd"] / q["Eel"] for q in R if q["Eel"] > 0]
            T.append(f"{nm} & {len(R)} & {f3(med(R, 'ratio'))} & {f3(med(R, 'nd'))} & {f3(np.nanmax([q['nd'] for q in R]))} & "
                     f"{f3(med(R, 'Eel'))} & {f3(float(np.nanmedian(rat)))} & {cnt(R)} \\\\")
    T.append(r"\bottomrule\end{tabular}}\\")
    if RC and _BUG:
        T.append(r"\textbf{C$^1$ pl 의 L-BFGS 기울기는 참 기울기가 아니다}: 시작점 임의 방향 방향미분 대비 상대차 중앙 "
                 + f3(med(RC, "gerr_run")) + r" (참 기울기 " + f3(med(RC, "gerr")) + r"). \texttt{WF.pl\_c1\_dofs} 의 \texttt{index\_put} "
                 r"(같은 노드·변 채널 자리를 여러 사면체가 같은 값으로 씀) 역전파가 중복 원천 모두에 기울기를 줘 값 채널 $\to$ 변 채널 경로를 부풀린다. "
                 r"수치는 참 기울기 기준. 검산: 위치 최대차 " + f"{np.nanmax([q['xchk'] for q in RC]):.1e}" + r", F 최대차 "
                 + f"{np.nanmax([q['Fchk'] for q in RC]):.1e}" + r".")
    open(a.tex, "w").write("\n".join(T) + "\n")
    print("[tex]", a.tex)
