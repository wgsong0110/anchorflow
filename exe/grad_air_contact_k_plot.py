"""기울기 진단 (b) 행 대체판: (a) 학습 끝 | (b) 반복 K | (c) 새 망 세 행 그림 + 후보 반복 표 + 재학습 결정성 대조.

  python exe/grad_air_contact_k_plot.py --dir <grad_air_contact> --kdir <grad_air_contact/k> --tb0 <원래 TB 폴더> --K 1 --cands 1,2,5,100,500,2000 --out <ovnet>

분류는 grad_air_contact_plot.py 와 같다 (20ε 기본, ε 보조; 상태·모델 한 스텝 둘 다 문턱 위면 공중).
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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

for f in fm.findSystemFonts() + glob.glob(os.path.expanduser("~/.fonts/*")):
    if "Noto" in f and ("CJK" in f or "KR" in f):
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--kdir", required=True)
ap.add_argument("--tb0", required=True)
ap.add_argument("--K", type=int, default=1)
ap.add_argument("--cands", default="1,2,5,100,500,2000")
ap.add_argument("--out", required=True)
a = ap.parse_args()
cands = [int(q) for q in a.cands.split(",")]
os.makedirs(a.out, exist_ok=True)
load = lambda p: [json.loads(l) for l in open(p)]
R = {"trained": load(f"{a.dir}/states_trained.jsonl"), "init": load(f"{a.dir}/states_init.jsonl")}
for K in cands:
    R[f"k{K}"] = load(f"{a.kdir}/states_k{K}.jsonl")
LAB = {"trained": "(a) 학습 끝 (15000 반복)", f"k{a.K}": f"(b) 반복 {a.K} (공중 손실 튄 직후)", "init": "(c) 새 망 (반복 0)"}
TRJ = ["L000", "0000", "0001", "0002", "0003"]
MK = dict(zip(TRJ, ["o", "s", "^", "D", "v"]))
C_AIR, C_CON = "#1c7ed6", "#e8590c"
eps = R["trained"][0]["eps_map"]
TH = {"20ε": 20 * eps, "ε": eps}


def air(r, th):
    return r["h_state"] > th and r["h_model"] > th


col = lambda rows, f: np.array([f(r) for r in rows], float)
sens = lambda r: r["gtheta"] / max(r["out_err"], 1e-30)
excess = lambda r: r["E"] - r["E_star"]
PAN = [("‖∂손실/∂θ‖ (2-노름)", lambda r: r["gtheta"], "log"), ("초과량 E − E*", excess, "symlog"),
       ("모델 출력 오차 / EXT", lambda r: r["out_err"], "log"), ("민감도 ‖∂손실/∂θ‖ / 출력 오차", sens, "log")]
ROWS = ["trained", f"k{a.K}", "init"]
fig, ax = plt.subplots(3, 4, figsize=(21, 12), sharex=True)
for ci, (ttl, f, sc) in enumerate(PAN):
    allv = np.concatenate([col(R[k], f) for k in ROWS]); fin = allv[np.isfinite(allv)]
    for ri, k in enumerate(ROWS):
        A = ax[ri, ci]
        for T in TRJ:
            rr = sorted([r for r in R[k] if r["traj"] == T], key=lambda r: r["state"])
            x = col(rr, lambda r: r["state"]); y = col(rr, f); ia = np.array([air(r, TH["20ε"]) for r in rr])
            A.plot(x, y, "-", color="0.7", lw=0.7, zorder=1)
            A.scatter(x[ia], y[ia], s=14, marker=MK[T], color=C_AIR, zorder=2)
            A.scatter(x[~ia], y[~ia], s=14, marker=MK[T], color=C_CON, zorder=2)
        if sc == "symlog":
            A.set_yscale("symlog", linthresh=max(np.quantile(np.abs(fin), 0.05), 1e-12)); A.axhline(0, color="k", lw=0.5)
            A.set_ylim(fin.min() * (1.5 if fin.min() < 0 else 0.5), fin.max() * 1.5)
        else:
            pos = fin[fin > 0]; A.set_yscale("log"); A.set_ylim(pos.min() / 1.5, pos.max() * 1.5)
        A.grid(alpha=0.3, which="both", lw=0.4)
        if ri == 0:
            A.set_title(ttl)
        if ri == 2:
            A.set_xlabel("프레임 (상태 번호)")
        if ci == 0:
            A.set_ylabel(LAB[k])
hs = [plt.Line2D([], [], ls="", marker="o", color=C_AIR, label=f"공중 (바닥 위 > 20ε = {20 * eps:g})"),
      plt.Line2D([], [], ls="", marker="o", color=C_CON, label="접촉")] + \
     [plt.Line2D([], [], ls="", marker=MK[T], color="0.3", label=f"궤적 {T}") for T in TRJ]
fig.legend(handles=hs, loc="lower center", ncol=7, frameon=False)
fig.suptitle("lego 새 바닥 다섯 궤적 상태 적합: 상태별 손실 기울기 -- 학습 끝 / 공중 손실 튄 직후 / 새 망 (열마다 공통 축)", y=0.995)
fig.tight_layout(rect=(0, 0.03, 1, 0.98))
p1 = f"{a.out}/lego_grad_air_vs_contact_k.png"; fig.savefig(p1, dpi=120); plt.close(fig)


def ms(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return f"{np.median(v):.3e} / {np.mean(v):.3e}" if len(v) else "-"


# ---- 재학습 결정성 대조 ----
def scal(path):
    fs = sorted(glob.glob(f"{path}/events*"))
    out = {}
    for f in fs:
        ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
        for t in ea.Tags()["scalars"]:
            out.setdefault(t, {}).update({e.step: e.value for e in ea.Scalars(t)})
    return out


S0 = scal(a.tb0)
DET = []
for K in cands:
    S1 = scal(f"{a.kdir}/tb_train_{K}")
    rel, n, same = [], 0, 0
    for t, d1 in S1.items():
        if not (t.startswith("다궤적적합/손실") or t.startswith("다궤적적합_프레임/")):
            continue
        for s, v in d1.items():
            if t in S0 and s in S0[t]:
                v0 = S0[t][s]; n += 1; same += int(v == v0)
                rel.append(abs(v - v0) / max(abs(v0), 1e-30))
    lo = S1.get("다궤적적합/손실", {}); lo0 = S0.get("다궤적적합/손실", {})
    st = sorted(set(lo) & set(lo0))
    DET.append((K, n, same, max(rel) if rel else float("nan"), float(np.median(rel)) if rel else float("nan"),
                [(s, lo0[s], lo[s]) for s in st[:3]] + ([(st[-1], lo0[st[-1]], lo[st[-1]])] if st else [])))

L = ["# lego 새 바닥 다섯 궤적: 공중 vs 접촉 기울기 -- 반복 K 시점 망 (b 행 대체)", "",
     f"- 측정·분류는 lego_grad_air_vs_contact.md 와 같다 (학습과 같은 손실을 상태 하나씩, 공중 = 상태·모델 한 스텝 모두 바닥 위 > 문턱).",
     f"- K = {a.K}: 원래 TB(다궤적적합_프레임) 공중 상태(0000:0–11, 0002:0–14, 0003:0–4) 손실이 가장 큰 반복 (반복 1 중앙 1.4e-4, 3–5 약 2.5e-5, 100 약 4e-6, 2000 약 1.3e-6, 15000 약 7e-8). "
     "TB 의 반복 k 값은 k 번 갱신한 망의 손실이고 ck_K 도 K 번 갱신 뒤 망이다.",
     f"- K 시점 망 = spstick.sh multi 와 같은 명령·시드·커밋 13471f4 로 처음부터 K 반복까지만 다시 학습 (반복 수만 바꿈). 후보 {a.cands}.",
     "- 값은 `중앙값 / 평균`.", "", "## 재학습 결정성 (다시 돌린 학습 TB vs 원래 TB, 같은 반복의 같은 태그: 반복 손실 + 상태별 손실)", "",
     "| K | 대조 값 수 | 비트 일치 | 상대차 최대 | 상대차 중앙 | 반복 손실 (반복, 원래, 다시) |", "|---|---|---|---|---|---|"]
for K, n, same, mx, md, ex in DET:
    L.append(f"| {K} | {n} | {same} | {mx:.2e} | {md:.2e} | " + "; ".join(f"{s}: {v0:.6e} vs {v1:.6e}" for s, v0, v1 in ex) + " |")
L.append("")
ORDER = ["trained"] + [f"k{K}" for K in sorted(cands, reverse=True)] + ["init"]
NM = {"trained": "학습 끝 15000", "init": "새 망 0", **{f"k{K}": f"반복 {K}" for K in cands}}
for thn in ("20ε", "ε"):
    th = TH[thn]
    L += [f"## 분류 문턱 {thn} = {th:g}", "",
          "| 망 | 그룹 | 상태 수 | ‖∂손실/∂θ‖ | E | E* | E − E* | 출력 오차/EXT | 셀 max\\|σ−1\\| 중앙 | 민감도 | Σ‖g‖² 공중 몫 | ‖g‖ 중앙 공중/접촉 | 민감도 중앙 공중/접촉 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in ORDER:
        rows = R[k]; g2 = sum(r["gtheta"] ** 2 for r in rows if np.isfinite(r["gtheta"]))
        ra = [r for r in rows if air(r, th)]; rc = [r for r in rows if not air(r, th)]
        rat = (np.median(col(ra, lambda r: r["gtheta"])) / np.median(col(rc, lambda r: r["gtheta"])),
               np.median(col(ra, sens)) / np.median(col(rc, sens))) if ra and rc else (np.nan, np.nan)
        for gn, rr in (("공중", ra), ("접촉", rc)):
            if not rr:
                continue
            L.append(f"| {NM[k]} | {gn} | {len(rr)} | {ms(col(rr, lambda r: r['gtheta']))} | {ms(col(rr, lambda r: r['E']))} | "
                     f"{ms(col(rr, lambda r: r['E_star']))} | {ms(col(rr, excess))} | {ms(col(rr, lambda r: r['out_err']))} | "
                     f"{ms(col(rr, lambda r: r['jcell_med']))} | {ms(col(rr, sens))} | "
                     f"{sum(r['gtheta'] ** 2 for r in rr) / g2:.4f} | {rat[0]:.4f} | {rat[1]:.4f} |")
    L.append("")
p2 = f"{a.out}/lego_grad_air_vs_contact_k.md"
open(p2, "w").write("\n".join(L)); print("\n".join(L)); print(p1, p2)
