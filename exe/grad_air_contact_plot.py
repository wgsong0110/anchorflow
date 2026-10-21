"""기울기 공중/접촉 진단 (exe/gradac.sh -> train_deform AF_GRADAC 훅) 결과의 그림·표.

  python exe/grad_air_contact_plot.py --dir <grad_air_contact 결과> --out <ovnet>

분류: 공중 = 그 상태(h_state)와 모델 한 스텝 결과(h_model) 모두 바닥 위 최소 높이 > 문턱. 기본 문턱 = 20ε
(deform_model.bc_softplus_map·bc_stick_map 이 s/ε > 20 에서 위치·야코비안 모두 정확히 항등 -- 그 너머는 바닥 매핑이 손실에
아무 기여도 없다), 보조 문턱 = ε (stick_diag 의 접촉 d<0.01 과 같은 값). 망과 무관한 출발점(u=0, u₀)은 그 점의 h_end 로.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

for f in fm.findSystemFonts() + glob.glob(os.path.expanduser("~/.fonts/*")):
    if "Noto" in f and ("CJK" in f or "KR" in f):
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--suffix", default="")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
NETS = [("trained", "(a) 학습 끝 망 (15000 반복)"), ("init", "(b) 새 망 (학습 시작 시점)")]
R = {k: [json.loads(l) for l in open(f"{a.dir}/states_{k}.jsonl")] for k, _ in NETS}
TRJ = ["L000", "0000", "0001", "0002", "0003"]
MK = dict(zip(TRJ, ["o", "s", "^", "D", "v"]))
C_AIR, C_CON = "#1c7ed6", "#e8590c"
eps = R["trained"][0]["eps_map"]
TH = {"20ε": 20 * eps, "ε": eps}
TH_MAIN = "20ε"


def air(r, th, h2=None):
    return r["h_state"] > th and (r["h_model"] if h2 is None else h2) > th


def col(rows, f):
    return np.array([f(r) for r in rows], dtype=float)


def sens(r):
    return r["gtheta"] / max(r["out_err"], 1e-30)


def excess(r):
    return r["E"] - r["E_star"]


def excess_pct(r):
    return 100 * (r["E"] / r["E_star"] - 1)


# ---------------- 그림 1: 망 기울기 ----------------
PAN = [("‖∂손실/∂θ‖₂", lambda r: r["gtheta"], "log"),
       ("초과량 E − E*", excess, "symlog"),
       ("모델 출력 오차 / EXT", lambda r: r["out_err"], "log"),
       ("민감도 ‖∂손실/∂θ‖ / 출력 오차", sens, "log")]
fig, ax = plt.subplots(2, len(PAN), figsize=(5.2 * len(PAN), 8.4), sharex=True)
for ci, (ttl, f, sc) in enumerate(PAN):
    allv = np.concatenate([col(R[k], f) for k, _ in NETS])
    for ri, (k, lab) in enumerate(NETS):
        A = ax[ri, ci]
        for T in TRJ:
            rr = sorted([r for r in R[k] if r["traj"] == T], key=lambda r: r["state"])
            x = col(rr, lambda r: r["state"]); y = col(rr, f)
            ia = np.array([air(r, TH[TH_MAIN]) for r in rr])
            A.plot(x, y, "-", color="0.7", lw=0.7, zorder=1)
            A.scatter(x[ia], y[ia], s=14, marker=MK[T], color=C_AIR, zorder=2)
            A.scatter(x[~ia], y[~ia], s=14, marker=MK[T], color=C_CON, zorder=2)
        if sc == "symlog":
            lt = max(np.quantile(np.abs(allv[np.isfinite(allv)]), 0.05), 1e-12)
            A.set_yscale("symlog", linthresh=lt); A.axhline(0, color="k", lw=0.5)
        else:
            A.set_yscale("log")
        fin = allv[np.isfinite(allv)]
        if sc == "log":
            fin = fin[fin > 0]; A.set_ylim(fin.min() / 1.5, fin.max() * 1.5)
        else:
            A.set_ylim(fin.min() * (1.5 if fin.min() < 0 else 0.5), fin.max() * 1.5)
        A.grid(alpha=0.3, which="both", lw=0.4)
        if ri == 0:
            A.set_title(ttl)
        if ri == 1:
            A.set_xlabel("프레임 (상태 번호)")
        if ci == 0:
            A.set_ylabel(lab)
hs = [plt.Line2D([], [], ls="", marker="o", color=C_AIR, label=f"공중 (바닥 위 > {TH_MAIN} = {TH[TH_MAIN]:g})"),
      plt.Line2D([], [], ls="", marker="o", color=C_CON, label="접촉")] + \
     [plt.Line2D([], [], ls="", marker=MK[T], color="0.3", label=f"궤적 {T}") for T in TRJ]
fig.legend(handles=hs, loc="lower center", ncol=7, frameon=False)
fig.suptitle("lego 새 바닥 다섯 궤적 상태 적합: 상태별 손실 기울기 (학습과 같은 손실, 상태 하나씩) -- 열마다 공통 축", y=0.995)
fig.tight_layout(rect=(0, 0.04, 1, 0.98))
p1 = f"{a.out}/lego_grad_air_vs_contact{a.suffix}.png"; fig.savefig(p1, dpi=130); plt.close(fig)

# ---------------- 그림 2: 망과 무관한 출발점 u=0, u=u₀ ----------------
TR = R["trained"]
PTS = [("U_zero", "(i) u = 0 (관성 안 더함)"), ("U_inertial", "(ii) u = u₀ (관성 예측)")]
PAN2 = [("전체 ‖∂E/∂u‖ 이동량 환산 노름", lambda q: q["gu"]["all"]["mv_norm"], "log"),
        ("관성+중력 항", lambda q: q["gu"]["in"]["mv_norm"], "log"),
        ("탄성 항", lambda q: q["gu"]["el"]["mv_norm"], "log"),
        ("바닥 매핑 (켬 − 끔)", lambda q: q["gu"]["map"]["mv_norm"], "log"),
        ("E − E*", None, "symlog")]
fig, ax = plt.subplots(2, len(PAN2), figsize=(4.6 * len(PAN2), 8.4), sharex=True)
for ci, (ttl, f, sc) in enumerate(PAN2):
    def g(r, pk):
        return (r[pk]["E"] - r["E_star"]) if f is None else f(r[pk])
    allv = np.array([g(r, pk) for r in TR for pk, _ in PTS], float)
    for ri, (pk, lab) in enumerate(PTS):
        A = ax[ri, ci]
        for T in TRJ:
            rr = sorted([r for r in TR if r["traj"] == T], key=lambda r: r["state"])
            x = col(rr, lambda r: r["state"]); y = np.array([g(r, pk) for r in rr], float)
            ia = np.array([air(r, TH[TH_MAIN], r[pk]["h_end"]) for r in rr])
            A.plot(x, y, "-", color="0.7", lw=0.7, zorder=1)
            A.scatter(x[ia], y[ia], s=14, marker=MK[T], color=C_AIR, zorder=2)
            A.scatter(x[~ia], y[~ia], s=14, marker=MK[T], color=C_CON, zorder=2)
        fin = allv[np.isfinite(allv)]
        if sc == "symlog":
            lt = max(np.quantile(np.abs(fin), 0.05), 1e-12)
            A.set_yscale("symlog", linthresh=lt); A.axhline(0, color="k", lw=0.5)
            A.set_ylim(fin.min() * (1.5 if fin.min() < 0 else 0.5), fin.max() * 1.5)
        else:
            A.set_yscale("log"); pos = fin[fin > 0]
            if len(pos):
                A.set_ylim(pos.min() / 1.5, pos.max() * 1.5)
        A.grid(alpha=0.3, which="both", lw=0.4)
        if ri == 0:
            A.set_title(ttl)
        if ri == 1:
            A.set_xlabel("프레임 (상태 번호)")
        if ci == 0:
            A.set_ylabel(lab)
fig.legend(handles=hs, loc="lower center", ncol=7, frameon=False)
fig.suptitle("망 출력 0 일 때의 두 출발점에서 증분 포텐셜의 노드 변위 기울기 (이동량 환산 = ‖∂(E·nrm)/∂u_N‖·h²/m_N/EXT, 노드 전체 노름) -- 열마다 공통 축", y=0.995)
fig.tight_layout(rect=(0, 0.04, 1, 0.98))
p2 = f"{a.out}/lego_grad_u0_vs_inertial{a.suffix}.png"; fig.savefig(p2, dpi=130); plt.close(fig)


# ---------------- 표 ----------------
def ms(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (f"{np.median(v):.3e} / {np.mean(v):.3e}") if len(v) else "-"


L = ["# lego 새 바닥 다섯 궤적 상태 적합: 공중 vs 접촉 상태의 손실 기울기", "",
     f"- 자료: `{a.dir}` (states_trained.jsonl, states_init.jsonl; 400 상태 = 5 궤적 × 80), 학습과 같은 손실 경로(step_once → det_take → ip_of, 같은 nrm, phys_w·E + det 벌점)를 상태 하나씩.",
     f"- 공중 = 그 상태와 모델 한 스텝 결과 모두 바닥 위 최소 높이 > 문턱. 기본 문턱 20ε = {20 * eps:g} (바닥 매핑이 s/ε > 20 에서 정확히 항등), 보조 ε = {eps:g}.",
     "- E* = 같은 상태에서 출력만 최적화 해(덤프 노드 변위를 그대로 넣어 같은 경로로 한 스텝)의 증분 포텐셜. 초과% = 100·(E/E* − 1).",
     "- 값은 `중앙값 / 평균`.", ""]
for thn in (TH_MAIN, "ε"):
    th = TH[thn]
    L += [f"## 분류 문턱 {thn} = {th:g}", "",
          "| 망 | 그룹 | 상태 수 | ‖∂손실/∂θ‖ | E | E* | E − E* | 초과 % | 출력 오차/EXT | 셀 max\\|σ−1\\| 중앙 | 민감도 ‖g‖/출력오차 | Σ‖g‖² 비율 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, lab in NETS:
        rows = [r for r in R[k] if "E_star" in r]
        g2 = sum(r["gtheta"] ** 2 for r in rows if np.isfinite(r["gtheta"]))
        for gn, sel in (("공중", True), ("접촉", False)):
            rr = [r for r in rows if air(r, th) == sel]
            if not rr:
                L.append(f"| {lab} | {gn} | 0 | | | | | | | | | |"); continue
            L.append(f"| {lab} | {gn} | {len(rr)} | {ms(col(rr, lambda r: r['gtheta']))} | {ms(col(rr, lambda r: r['E']))} | "
                     f"{ms(col(rr, lambda r: r['E_star']))} | {ms(col(rr, excess))} | {ms(col(rr, excess_pct))} | "
                     f"{ms(col(rr, lambda r: r['out_err']))} | {ms(col(rr, lambda r: r['jcell_med']))} | {ms(col(rr, sens))} | "
                     f"{sum(r['gtheta'] ** 2 for r in rr if np.isfinite(r['gtheta'])) / g2:.4f} |")
    L.append("")
    L += ["배치 기울기에서 공중 상태 기여 비율 Σ_air‖g‖² / Σ_all‖g‖² 은 위 표 마지막 열 (공중 행). 중앙값 비 (공중/접촉):", ""]
    for k, lab in NETS:
        rows = [r for r in R[k] if "E_star" in r]
        ga = col([r for r in rows if air(r, th)], lambda r: r["gtheta"]); gc = col([r for r in rows if not air(r, th)], lambda r: r["gtheta"])
        sa = col([r for r in rows if air(r, th)], sens); sc_ = col([r for r in rows if not air(r, th)], sens)
        if len(ga) and len(gc):
            L.append(f"- {lab}: ‖g‖ 중앙 공중/접촉 = {np.median(ga) / np.median(gc):.4f}, 민감도 중앙 공중/접촉 = {np.median(sa) / np.median(sc_):.4f}")
    L.append("")
    L += ["궤적별 첫 접촉 프레임 (그 상태의 바닥 위 최소 높이 ≤ 문턱인 첫 상태, 출력만 최적화 궤적 기준):", ""]
    for T in TRJ:
        rr = sorted([r for r in R["trained"] if r["traj"] == T], key=lambda r: r["state"])
        fc = next((r["state"] for r in rr if r["h_state"] <= th), None)
        L.append(f"- {T}: {fc}  (상태 0 높이 {rr[0]['h_state']:.4f})")
    L.append("")
# 출발점 표
L += ["## 망과 무관한 출발점의 ∂E/∂u (노드 변위; 학습 끝 망 실행 (a) 에서 잼, 분류 20ε, 그 점의 한 스텝 결과 높이 기준)", "",
      "원값 = ‖∂E/∂u_N‖ (학습 정규화 E), 이동량 = ‖∂(E·nrm)/∂u_N‖·h²/m_N/EXT (지지 노드). 노름 = 노드 전체 2-노름. 바닥 매핑 항 = 같은 점에서 매핑 켬 − 끔의 기울기 차.", "",
      "| 출발점 | 그룹 | 상태 수 | 원값 중앙 | 원값 99% | 원값 노름 | 이동량 중앙 | 이동량 99% | 이동량 노름 | 관성+중력 이동량 노름 | 탄성 이동량 노름 | 바닥 매핑 이동량 노름 | E − E* | 출력 오차/EXT |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
PT_ALL = PTS + [("net", "(a) 학습 끝 망 출력"), ("star", "출력만 최적화 해 E*")]
for pk, lab in PT_ALL:
    for gn, sel in (("공중", True), ("접촉", False)):
        if pk in ("U_zero", "U_inertial"):
            rr = [r for r in TR if air(r, TH[TH_MAIN], r[pk]["h_end"]) == sel]
            q = lambda r: r[pk]; Ex = lambda r: r[pk]["E"] - r["E_star"]; oe = lambda r: r[pk]["out_err"]
            gu = lambda r: r[pk]["gu"]
        elif pk == "net":
            rr = [r for r in TR if air(r, TH[TH_MAIN]) == sel]
            gu = lambda r: r["gu_net"]; Ex = excess; oe = lambda r: r["out_err"]
        else:
            rr = [r for r in TR if air(r, TH[TH_MAIN], r["h_ov_next"]) == sel and "gu_star" in r]
            gu = lambda r: r["gu_star"]; Ex = lambda r: 0.0; oe = lambda r: r["replay_err"]
        if not rr:
            continue
        c = lambda f: ms(col(rr, f))
        mp = (c(lambda r: gu(r)["map"]["mv_norm"]) if "map" in gu(rr[0]) else "-")
        L.append(f"| {lab} | {gn} | {len(rr)} | {c(lambda r: gu(r)['all']['raw_med'])} | {c(lambda r: gu(r)['all']['raw_q99'])} | "
                 f"{c(lambda r: gu(r)['all']['raw_norm'])} | {c(lambda r: gu(r)['all']['mv_med'])} | {c(lambda r: gu(r)['all']['mv_q99'])} | "
                 f"{c(lambda r: gu(r)['all']['mv_norm'])} | {c(lambda r: gu(r)['in']['mv_norm'])} | {c(lambda r: gu(r)['el']['mv_norm'])} | "
                 f"{mp} | {c(Ex)} | {c(oe)} |")
L.append("")
rs = [r for r in TR if "replay_err" in r]
L += ["## 검산", "",
      f"- 출력만 최적화 해 재현: 덤프 노드 변위를 넣은 한 스텝 위치 vs 덤프 다음 상태, 평균 거리/EXT 최대 {max(r['replay_err'] for r in rs):.2e}, 최대 거리/EXT 최대 {max(r['replay_max'] for r in rs):.2e}",
      f"- 노드 격자 불일치 상태 수: {sum(1 for r in TR if r.get('ov_node_mismatch'))}",
      f"- 비유한 손실 상태 수: (a) {sum(1 for r in R['trained'] if not r['finite'])}, (b) {sum(1 for r in R['init'] if not r['finite'])}",
      f"- 망 시작 반복: (a) {R['trained'][0]['k0']}, (b) {R['init'][0]['k0']}", ""]
p3 = f"{a.out}/lego_grad_air_vs_contact{a.suffix}.md"
open(p3, "w").write("\n".join(L))
print("\n".join(L)); print(p1, p2, p3)
