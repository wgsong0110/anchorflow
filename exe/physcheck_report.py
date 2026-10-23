"""ipL000 물리 점검 (사용자 10-23) 그림·표: train_deform AF_PHYSCHECK (step.jsonl·static.jsonl·series.json) + AF_OVCONT_FIT (records.json).

  python exe/physcheck_report.py --dir OUT --cont CONTDIR --png OUT/ipL000_physcheck.png --md OUT/ipL000_physcheck.md

정의 (전부 train_deform 의 같은 경로에서 잰 값):
  1 한 스텝 최적성: E/E* (E* = 같은 상태 L-BFGS 500 해), 노드 변위 기울기 ∂(E·nrm)/∂U (N) 의 변위 환산 Σ|g| h²/ΣM / EXT, 잔여힘/무게 Σ|g|/(ΣM|g|)
  2 공중 자유낙하: 질량중심 가속도 (위치 2 차 차분 / h², 첫 스텝은 (c1 - c0 - h v0)/h²), 세 상태 모두 바닥 부호거리 최소 > 0 인 스텝만
  3 에너지: K = ½Σ m|v|², 중력 Ug = -Σ m g·x (정답 0 프레임 기준), 탄성 = Σ V ψ(F) (jelly FCR, psi_color_lb 와 같은 psi_of)
  4 정적 평형: Π = Σ Vψ(JF) + 바닥 - Σ m g·Δx 의 U=0 기울기 (같은 바닥 사상), 잔여힘/무게; 정지 유지: 증분 포텐셜(v=0) 의 U=0 기울기
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--cont", required=True)
ap.add_argument("--png", required=True)
ap.add_argument("--md", required=True)
a = ap.parse_args()
for f in fm.findSystemFonts():
    if "Noto" in f and ("CJK" in f or "KR" in f):
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

S = json.load(open(f"{a.dir}/series.json"))
h, g, EXT = S["h"], np.array(S["g"]), S["EXT"]
gn = float(np.linalg.norm(g)); gh = g / gn
ST = [json.loads(q) for q in open(f"{a.dir}/step.jsonl") if q.strip()]
ST.sort(key=lambda r: r["frame"])
SA = [json.loads(q) for q in open(f"{a.dir}/static.jsonl") if q.strip()]
CR = json.load(open(f"{a.cont}/records.json"))
NAME = {"gt": "정답 (MPM)", "model": "한 프레임씩 (증분 포텐셜 Adam)", "ov": "출력만 최적화", "cont": "이어 돌린 출력만 최적화"}
COL = {"gt": "#222222", "model": "#d9480f", "ov": "#1c7ed6", "cont": "#2b8a3e"}
SRC = ("gt", "model", "ov")


def ser(src, k):
    return np.array([r[k] for r in S["series"][src]["rows"]], dtype=float)


def com(src):
    return np.array([r["com"] for r in S["series"][src]["rows"]])


def contact(src):
    sd = ser(src, "sd_min")
    i = np.nonzero(sd <= 0)[0]
    return int(i[0]) if len(i) else None


md = ["# ipL000 물리 점검 (한 프레임씩 증분 포텐셜 최적화, lego L000)", "",
      f"h = {h:.5g} s, g = {g.tolist()}, |g| = {gn:.4g}, EXT = {EXT:.4g} m, 질량 {S['mass']:.5g} kg, 재질 {S['mat']} E {S['E']} ν {S['nu']}, "
      f"바닥 {S['floor']}", ""]
fig, ax = plt.subplots(4, 3, figsize=(20, 21))
ax = ax.ravel()
fr = np.array([r["frame"] for r in ST])

# ---- 1 한 스텝 최적성 ----
q = np.array([r["ratio_E_Estar"] - 1 for r in ST])
ax[0].semilogy(fr, np.maximum(q, 1e-8), c=COL["model"], lw=1.6, label="E/E* − 1 (다시 잰 값)")
ax[0].semilogy(fr, np.maximum(np.array([r["rec_E_ip"] / r["rec_E_star"] - 1 for r in ST]), 1e-8), c="#868e96", lw=1.0, ls="--", label="E/E* − 1 (records.json)")
ax[0].set_title("1 한 스텝 최적성: 증분 포텐셜 E/E* − 1"); ax[0].set_xlabel("프레임"); ax[0].legend(fontsize=9)
for key, lab, c in (("g_model", "망 출력", COL["model"]), ("g_star", "E* 해 (L-BFGS 500)", COL["ov"]), ("g_pred", "관성 예측 (시작점)", "#868e96")):
    ax[1].semilogy(fr, [r[key]["disp_pct"] for r in ST], c=c, lw=1.6, label=lab)
ax[1].set_title("1 노드 변위 기울기의 변위 환산 Σ|∂E/∂U| h²/ΣM  (% EXT)"); ax[1].set_xlabel("프레임"); ax[1].legend(fontsize=9)
ax[2].semilogy(fr, [r["pos_model_star_mean_pct"] for r in ST], c=COL["model"], lw=1.6, label="입자 위치차 평균")
ax[2].semilogy(fr, [r["pos_model_star_max_pct"] for r in ST], c=COL["model"], lw=1.0, ls="--", label="입자 위치차 최대")
ax[2].semilogy(fr, [r["node_model_star_rms_pct"] for r in ST], c="#7048e8", lw=1.2, label="노드 변위차 (질량 가중 RMS)")
ax[2].semilogy(fr, [r["g_model"]["U_mean_pct"] for r in ST], c="#868e96", lw=1.0, ls=":", label="참고: 망 노드 변위 크기 (질량 가중 평균)")
ax[2].set_title("1 망 출력 − E* 해 (같은 상태)  (% EXT)"); ax[2].set_xlabel("프레임"); ax[2].legend(fontsize=9)
rr = np.array([[r["ratio_E_Estar"], r["g_model"]["disp_pct"], r["g_star"]["disp_pct"], r["g_pred"]["disp_pct"], r["g_model"]["f_l1_w"],
                r["g_star"]["f_l1_w"], r["pos_model_star_mean_pct"], r["pos_model_star_max_pct"], r["g_model"]["g_over_inU"], r["g_star"]["g_over_inU"]] for r in ST])
md += ["## 1 한 스텝 최적성", "",
       "| 구간 | E/E* 중앙 (최대) | 기울기 변위환산 % 망 | E*해 | 관성예측 | 잔여힘/무게 망 | E*해 | 망−E* 위치 평균 % (최대) | Σ|g|/Σ(M/h²)|U| 망 | E*해 |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for lab, sl in (("프레임 1–10", slice(0, 10)), ("11–40", slice(10, 40)), ("41–80", slice(40, 80)), ("전체", slice(None))):
    z = rr[sl]
    md.append(f"| {lab} | {np.median(z[:, 0]):.4f} ({z[:, 0].max():.4f}) | {np.median(z[:, 1]):.3e} | {np.median(z[:, 2]):.3e} | {np.median(z[:, 3]):.3e} | "
              f"{np.median(z[:, 4]):.3e} | {np.median(z[:, 5]):.3e} | {np.median(z[:, 6]):.4f} ({z[:, 7].max():.4f}) | {np.median(z[:, 8]):.3e} | {np.median(z[:, 9]):.3e} |")
ck = np.array([[r["chk_model_x_max_pct"], abs(r["E_model"] / r["rec_E_ip"] - 1), r["chk_star_vs_file_mean_pct"] or 0, abs(r["E_star"] / r["rec_E_star"] - 1)] for r in ST])
md += ["", f"대조: 저장된 망으로 다시 낸 다음 상태 vs 기록 state_out 위치차 최대 {ck[:, 0].max():.2e}% EXT, E 상대차 최대 {ck[:, 1].max():.2e}; "
       f"다시 푼 E* 해 vs 기록 x_star 평균 위치차 최대 {ck[:, 2].max():.2e}%, E* 상대차 최대 {ck[:, 3].max():.2e}", ""]

# ---- 2 공중 자유낙하 ----
md += ["## 2 공중 자유낙하 (질량중심 가속도, 세 상태 모두 바닥 부호거리 최소 > 0 인 스텝)", "",
       "| 실행 | 공중 스텝 수 | 첫 접촉 프레임 | 평균 가속도 [m/s²] | |ā−g|/|g| | 스텝별 |a−g|/|g| 최대 | g 방향 성분 평균/|g| |", "|---|---|---|---|---|---|---|"]
for src in SRC:
    C = com(src); sd = ser(src, "sd_min")
    v0 = np.array(S["series"][src]["rows"][0]["vcom"])
    A = [(C[1] - C[0] - h * v0) / h ** 2] + [(C[t + 1] - 2 * C[t] + C[t - 1]) / h ** 2 for t in range(1, len(C) - 1)]
    A = np.array(A)
    air = np.array([sd[t + 1] > 0 and sd[t] > 0 and (t == 0 or sd[t - 1] > 0) for t in range(len(A))])
    along = A @ gh
    ax[3].plot(np.arange(len(A)), along / gn, c=COL[src], lw=1.6 if src != "gt" else 2.0, label=NAME[src])
    ct = contact(src)
    if ct is not None:
        ax[3].axvline(ct, c=COL[src], ls=":", lw=1.0)
    if air.any():
        Am = A[air].mean(0)
        md.append(f"| {NAME[src]} | {int(air.sum())} | {ct} | {np.round(Am, 4).tolist()} | {np.linalg.norm(Am - g) / gn:.4e} | "
                  f"{(np.linalg.norm(A[air] - g, axis=1) / gn).max():.4e} | {along[air].mean() / gn:.6f} |")
    else:
        md.append(f"| {NAME[src]} | 0 | {ct} | - | - | - | - |")
ax[3].axhline(1.0, c="#868e96", ls="--", lw=1.0, label="중력 g")
ax[3].set_ylim(-1.5, 3.0)
ax[3].set_title("2 질량중심 가속도의 g 방향 성분 / |g|  (점선 = 첫 접촉 프레임)"); ax[3].set_xlabel("스텝 (상태 t)"); ax[3].legend(fontsize=9)
md.append("")

# ---- 3 에너지 ----
U0 = S["series"]["gt"]["rows"][0]["Ug"]
K0 = S["series"]["gt"]["rows"][0]["K"]
md += ["## 3 에너지 (K + 중력 + 탄성, 정답 0 프레임 중력 기준) — 단위 J, 비율은 초기 운동 에너지 K0 대비", "",
       f"K0 = {K0:.4e} J", "",
       "| 실행 | 처음 합 | 끝 합 | 증가한 프레임 수 (> 1e-4 K0) | 한 프레임 최대 증가 / K0 | 증가분 총합 / K0 | 최대 증가 프레임 | 끝 탄성 | 끝 운동 |",
       "|---|---|---|---|---|---|---|---|---|"]
TOT = {}
for src in SRC + ("cont",):
    if src not in S["series"]:
        continue
    rows = S["series"][src]["rows"]
    fx = np.array([r["frame"] for r in rows])
    K, Ug, Ee = (np.array([r[k] for r in rows]) for k in ("K", "Ug", "Eel"))
    T_ = K + (Ug - U0) + Ee
    TOT[src] = (fx, T_, K, Ee)
    dT = np.diff(T_)
    inc = dT > 1e-4 * K0
    md.append(f"| {NAME[src]} | {T_[0]:.4e} | {T_[-1]:.4e} | {int(inc.sum())} / {len(dT)} | {dT.max() / K0:.4e} | {dT[dT > 0].sum() / K0:.4e} | "
              f"{int(fx[1:][np.argmax(dT)])} | {Ee[-1]:.4e} | {K[-1]:.4e} |")
for src, (fx, T_, K, Ee) in TOT.items():
    ax[4].plot(fx, T_, c=COL[src], lw=1.6, label=NAME[src])
    ax[5].semilogy(fx, np.maximum(K, 1e-12), c=COL[src], lw=1.6, label=NAME[src])
    ax[6].semilogy(fx, np.maximum(Ee, 1e-12), c=COL[src], lw=1.6, label=NAME[src])
ax[4].set_title("3 총 에너지 K + (Ug − Ug₀) + 탄성  [J]"); ax[5].set_title("3 운동 에너지 ½Σm|v|²  [J]"); ax[6].set_title("3 탄성 에너지 ΣVψ(F)  [J]")
for i in (4, 5, 6):
    ax[i].set_xlabel("프레임"); ax[i].legend(fontsize=9)
md.append("")

# ---- 4 정적 평형 ----
md += ["## 4 정지 상태 평형 (마지막 상태)", "",
       "| 실행 | 프레임 | 속도 최대 [m/s] | 속도 평균 | 잔여힘/무게 Σ|f|/ΣMg | 합력/무게 | 노드별 |f|/(Mg) 중앙 (90%) | 정지 유지 잔차/무게 | 정지 유지 한 스텝 이동 평균 % (최대) | det F≤0 | 바닥 아래 입자 | 최대 깊이 [m] |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
SAd = {(r["src"], r["frame"]): r for r in SA}
for src in SRC:
    ks = sorted(k for s_, k in SAd if s_ == src)
    y = [SAd[(src, k)]["static"]["f_l1_w"] for k in ks]
    ax[7].semilogy(ks, y, c=COL[src], lw=1.6, label=NAME[src])
    ax[7].semilogy(ks, [SAd[(src, k)]["hold"]["f_l1_w"] for k in ks], c=COL[src], lw=1.0, ls="--")
    rows = S["series"][src]["rows"]
    for k in (ks[-11], ks[-1]):
        r_, s_ = SAd[(src, k)], rows[k]
        md.append(f"| {NAME[src]} | {k} | {s_['v_max']:.3e} | {s_['v_mean']:.3e} | {r_['static']['f_l1_w']:.4e} | {r_['static']['f_net_w']:.4e} | "
                  f"{r_['static']['node_ratio_med']:.3e} ({r_['static']['node_ratio_p90']:.3e}) | {r_['hold']['f_l1_w']:.4e} | "
                  f"{r_['hold']['x2_move_pct']:.3e} ({r_['hold']['x2_move_max_pct']:.3e}) | {s_['det_le0']} | {s_['n_below']} | {s_['depth_max']:.3e} |")
ax[7].set_title("4 정적 평형 잔여힘/무게 (실선 Π, 점선 정지 유지 증분 포텐셜)"); ax[7].set_xlabel("프레임"); ax[7].legend(fontsize=9)
for src in SRC:
    rows = S["series"][src]["rows"]
    ax[8].semilogy([r["frame"] for r in rows], [max(r["v_max"], 1e-9) for r in rows], c=COL[src], lw=1.6, label=f"{NAME[src]} 최대")
    ax[8].semilogy([r["frame"] for r in rows], [max(r["v_mean"], 1e-9) for r in rows], c=COL[src], lw=1.0, ls="--")
ax[8].set_title("4 입자 속도 [m/s] (실선 최대, 점선 평균)"); ax[8].set_xlabel("프레임"); ax[8].legend(fontsize=9)
md += ["", "마지막 몇 프레임 속도:", ""]
for src in SRC:
    rows = S["series"][src]["rows"][-5:]
    md.append(f"- {NAME[src]}: " + ", ".join(f"프레임 {r['frame']} 최대 {r['v_max']:.3e} 평균 {r['v_mean']:.3e}" for r in rows))
md.append("")

# ---- 5 이어 돌리기 ----
cf = np.array([r["frame"] for r in CR])
ax[9].plot(cf, [r["com_disp_pct"] for r in CR], c=COL["cont"], lw=1.8, label="질량중심 이동")
ax[9].plot(cf, [r["disp_max_pct"] for r in CR], c="#7048e8", lw=1.4, label="입자 최대 이동")
ax[9].plot(cf, [r["disp_mean_pct"] for r in CR], c="#868e96", lw=1.2, ls="--", label="입자 평균 이동")
ax[9].set_title(f"5 이어 돌리기: 시작 상태 (프레임 {CR[0]['frame']}) 대비 이동  (% EXT)"); ax[9].set_xlabel("프레임"); ax[9].legend(fontsize=9)
ax[10].semilogy(cf, [max(r["K"], 1e-14) for r in CR], c=COL["cont"], lw=1.8, label="운동 에너지 [J]")
ax[10].set_title("5 이어 돌리기: 운동 에너지 [J]"); ax[10].set_xlabel("프레임"); ax[10].legend(fontsize=9)
ax[11].semilogy(cf, [r["static"]["f_l1_w"] for r in CR], c=COL["cont"], lw=1.8, label="잔여힘/무게 (Π)")
ax[11].semilogy(cf, [r["hold"]["f_l1_w"] for r in CR], c=COL["cont"], lw=1.0, ls="--", label="정지 유지 잔차/무게")
ax[11].set_title("5 이어 돌리기: 정적 평형 잔여 힘"); ax[11].set_xlabel("프레임"); ax[11].legend(fontsize=9)
dz = np.array([r["com_disp"] for r in CR[1:]]) @ gh if len(CR) > 1 else np.zeros(1)
md += [f"## 5 출력만 최적화 이어 돌리기 (시작 = 한 프레임씩 결과 프레임 {CR[0]['frame']}, {len(CR) - 1} 프레임)", "",
       "| 프레임 | 질량중심 이동 % | 그중 g 방향 [m] | 입자 최대 이동 % | 평균 이동 % | 속도 최대 | 평균 | K [J] | 잔여힘/무게 | 정지 유지 | det≤0 | 바닥 아래 |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for j in sorted(set([0, 1, 2, 5, 10, 20, 30, 60, 90, len(CR) - 1]) & set(range(len(CR)))):
    r = CR[j]
    md.append(f"| {r['frame']} | {r['com_disp_pct']:.4f} | {(np.array(r.get('com_disp', [0, 0, 0])) @ gh):.3e} | {r['disp_max_pct']:.4f} | {r['disp_mean_pct']:.4f} | "
              f"{r['v_max']:.3e} | {r['v_mean']:.3e} | {r['K']:.3e} | {r['static']['f_l1_w']:.4e} | {r['hold']['f_l1_w']:.4e} | {r['det_le0']} | {r['n_below']} |")
md += ["", f"출력만 최적화 증분 포텐셜 (최적화 전 → 후) 마지막: {CR[-1].get('E0', float('nan')):.4e} → {CR[-1].get('E1', float('nan')):.4e}", ""]
fig.suptitle("lego L000 한 프레임씩 증분 포텐셜 최적화 결과의 물리 점검", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(a.png, dpi=90)
open(a.md, "w").write("\n".join(md) + "\n")
print("\n".join(md))
print(f"[그림] {a.png}\n[표] {a.md}\nPHYSCHECK_REPORT_DONE")
