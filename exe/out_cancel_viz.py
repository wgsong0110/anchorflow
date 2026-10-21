"""lego --net_inertial 롤아웃의 "멈춤" 진단: 노드 변위 = 관성 예측 u0 + 망 출력 δ 에서 δ ≈ −u0 (상쇄) 인가.

입력 (사례마다): train_deform 의 AF_DELTA_DUMP 덤프 (프레임별 노드 위치·u0·dpn=u0+δ·노드 질량·노드 바닥 거리·지지 입자
최소 바닥 거리), 같은 실행의 AF_ROLL_DUMP (pred·gt·x0 -- 정답 대비 이동량), 기존 렌더 mp4 (모델·정답).

  python exe/out_cancel_viz.py --out_vid <v20261009> --out_fig <ovnet> --json <x.json> \
      --case 이름=제목=δ덤프=롤덤프=모델mp4=정답mp4=멈춤프레임 ...

정의 (노드 = 질량 > 0 인 활성 노드, 프레임 i = 상태 i -> i+1 스텝):
  cos(δ,u0),  |δ|/|u0|,  δ·û0/|u0| (관성 방향 성분 비: −1 = 완전 상쇄),  상쇄 비율 c = −δ·û0/|u0|
  운동량 비 |Σm(u0+δ)| / |Σm u0| (0 = 노드 합 운동량이 관성 예측을 완전히 상쇄)
  접촉 노드 = 지지 입자(그 노드의 사면체 꼭짓점인 입자) 중 스텝 시작 바닥 거리 최소값 < --contact (기본 0.02 = 2ε), 나머지 공중
  이동량 = 입자 평균 |x_{i+1} − x_i| / EXT ×100 (모델·정답 각자, 고정 상수 EXT 로만 나눔)
색 범위·축은 모든 사례·프레임 공통 고정.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
import imageio.v2 as iio  # noqa: E402

for f in fm.findSystemFonts() + glob.glob(os.path.expanduser("~/.fonts/*")):
    if "Noto" in f and ("CJK" in f or "KR" in f):
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--case", action="append", required=True)
ap.add_argument("--out_vid", required=True)
ap.add_argument("--out_fig", required=True)
ap.add_argument("--json", required=True)
ap.add_argument("--contact", type=float, default=0.02)
ap.add_argument("--fps", type=int, default=8)
a = ap.parse_args()
os.makedirs(a.out_vid, exist_ok=True); os.makedirs(a.out_fig, exist_ok=True)

C_AIR, C_CON, C_ALL = "#1c7ed6", "#e8590c", "#495057"
B_COS = np.linspace(-1, 1, 61)
B_LR = np.linspace(-2, 2, 61)              # log10 |δ|/|u0|
B_PAR = np.linspace(-2.5, 1.5, 61)         # δ·û0/|u0|


def wmed(x, w):
    if x.size == 0:
        return np.nan
    o = np.argsort(x); cw = np.cumsum(w[o]); return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def load(spec):
    nm, title, dd, rd, mv, gv, st = spec.split("=")
    D = torch.load(dd, map_location="cpu", weights_only=False)
    R = torch.load(rd, map_location="cpu", weights_only=False, mmap=True)
    EXT = float(D["EXT"])
    pred, gt, x0 = R["pred"].float(), R["gt"].float(), R["x0"].float()
    L = pred.shape[0]
    prev_p = torch.cat([x0[None], pred[:-1]]); prev_g = torch.cat([x0[None], gt[:-1]])
    mot_p = (100 * (pred - prev_p).norm(dim=-1).mean(-1) / EXT).numpy()
    mot_g = (100 * (gt - prev_g).norm(dim=-1).mean(-1) / EXT).numpy()
    err = (100 * (pred - gt).norm(dim=-1).mean(-1) / EXT).numpy()
    del pred, gt, R
    fr = []
    for q in D["frames"]:
        npos, u0, dpn, m, sn, pm = [t.double().numpy() for t in q]
        k = m > 0
        u0, de, m, pm, npos, sn = u0[k], (dpn - q[1].double().numpy())[k], m[k], pm[k], npos[k], sn[k]
        nu, nd = np.linalg.norm(u0, axis=1), np.linalg.norm(de, axis=1)
        uh = u0 / np.maximum(nu, 1e-30)[:, None]
        cos = (de * uh).sum(1) / np.maximum(nd, 1e-30)
        par = (de * uh).sum(1) / np.maximum(nu, 1e-30)
        con = pm < a.contact
        Mu = (m[:, None] * u0).sum(0); Mt = (m[:, None] * (u0 + de)).sum(0)
        rec = dict(u0=u0, de=de, m=m, nu=nu, nd=nd, cos=cos, par=par, lr=np.log10(np.maximum(nd, 1e-30) / np.maximum(nu, 1e-30)),
                   con=con, pm=pm, sn=sn, mom=float(np.linalg.norm(Mt) / max(np.linalg.norm(Mu), 1e-30)))
        for g, sel in (("all", np.ones_like(con)), ("con", con), ("air", ~con)):
            s = sel.astype(bool)
            rec[f"n_{g}"] = int(s.sum()); rec[f"mass_{g}"] = float(m[s].sum())
            rec[f"medcos_{g}"] = float(np.median(cos[s])) if s.any() else np.nan
            rec[f"medcan_{g}"] = float(np.median(-par[s])) if s.any() else np.nan
            rec[f"wmedcos_{g}"] = wmed(cos[s], m[s]); rec[f"wmedcan_{g}"] = wmed(-par[s], m[s])
            rec[f"frac_cancel_{g}"] = float(((-par[s] > 0.8) & (-par[s] < 1.2)).mean()) if s.any() else np.nan
            if s.any():
                Mu_g = (m[s, None] * u0[s]).sum(0); Mt_g = (m[s, None] * (u0[s] + de[s])).sum(0)
                rec[f"mom_{g}"] = float(np.linalg.norm(Mt_g) / max(np.linalg.norm(Mu_g), 1e-30))
            else:
                rec[f"mom_{g}"] = np.nan
        rec["med_u0"] = float(100 * np.median(nu) / EXT); rec["med_de"] = float(100 * np.median(nd) / EXT)
        fr.append(rec)
    if len(fr) != L:
        raise SystemExit(f"{nm}: δ 덤프 {len(fr)} 프레임 != 롤아웃 {L}")
    return dict(nm=nm, title=title, fr=fr, EXT=EXT, mot_p=mot_p, mot_g=mot_g, err=err, mv=mv, gv=gv, stop=int(st), L=L)


CS = [load(s) for s in a.case]
# 공통 색 범위: 모든 사례·프레임의 |δ|·|u0| (EXT%) 99 분위수의 최대
VMAX = max(max(np.percentile(100 * np.concatenate([r["nd"], r["nu"]]) / c["EXT"], 99) for r in c["fr"]) for c in CS)
print(f"[색 범위] |δ|, |u0| 0 ~ {VMAX:.4f} (EXT%)", flush=True)
MOTMAX = max(max(c["mot_p"].max(), c["mot_g"].max()) for c in CS) * 1.05


def curve_keys():
    return [("medcos", "중앙값 cos(δ, u0)", (-1.05, 1.05)), ("medcan", "중앙값 상쇄 비율 −δ·û0/|u0|", (-0.5, 1.6)),
            ("mom", "운동량 비 |Σm(u0+δ)|/|Σm u0|", (0, 1.6))]


def sphere(ax, v, col, title, mk="o"):
    uu, vv = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    ax.plot_wireframe(np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv), color="#ced4da", lw=0.3)
    n = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-30)
    sc = ax.scatter(n[:, 0], n[:, 1], n[:, 2], c=col, cmap="viridis", vmin=0, vmax=VMAX, s=2, marker=mk, depthshade=False, lw=0)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1); ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-60); ax.set_title(title, fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (위)")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([-1, 0, 1])
    return sc


def hist(ax, x, r, bins, xl, vline=None, wt=False):
    for g, c, lab in (("con", C_CON, "접촉"), ("air", C_AIR, "공중")):
        s = r["con"] if g == "con" else ~r["con"]
        if s.any():
            w = r["m"][s] / r["m"][s].sum() if wt else np.full(s.sum(), 1.0 / s.sum())
            ax.hist(np.clip(x[s], bins[0], bins[-1]), bins=bins, weights=w, histtype="step", color=c, lw=1.5,
                    label=f"{lab} ({s.sum()})")
    w = r["m"] / r["m"].sum() if wt else np.full(x.size, 1.0 / x.size)
    ax.hist(np.clip(x, bins[0], bins[-1]), bins=bins, weights=w, color=C_ALL, alpha=0.25, label="전체")
    if vline is not None:
        ax.axvline(vline, color="k", ls=":", lw=1)
    ax.set_xlim(bins[0], bins[-1]); ax.set_ylim(0, 0.6); ax.set_xlabel(xl, fontsize=9)
    ax.set_ylabel(("질량 비율" if wt else "노드 비율"), fontsize=9); ax.legend(fontsize=7, loc="upper left")


def frames_of(path):
    try:
        return [f for f in iio.get_reader(path)]
    except Exception as e:  # noqa: BLE001
        print(f"[렌더 읽기 실패] {path}: {e}", flush=True); return []


for c in CS:
    MV, GV = frames_of(c["mv"]), frames_of(c["gv"])
    print(f"[{c['nm']}] 렌더 모델 {len(MV)} 프레임, 정답 {len(GV)} 프레임", flush=True)
    out = os.path.join(a.out_vid, f"lego_out_sphere_{c['nm']}.mp4")
    W = iio.get_writer(out, fps=a.fps, codec="libx264", quality=8, macro_block_size=1)
    x = np.arange(c["L"])
    for i, r in enumerate(c["fr"]):
        fig = plt.figure(figsize=(22, 15), dpi=72)
        gs = fig.add_gridspec(4, 8, height_ratios=[1.25, 1.25, 1, 1], hspace=0.45, wspace=0.55)
        fig.suptitle(f"{c['title']} — 스텝 {i}→{i + 1} (상태 {i} 렌더)  |  노드 {r['n_all']} (접촉 {r['n_con']}, 공중 {r['n_air']}),"
                     f" 위치 오차 {c['err'][i]:.2f}%, 운동량 비 {r['mom']:.3f}, 중앙값 |u0| {r['med_u0']:.3f}% |δ| {r['med_de']:.3f}% (EXT)",
                     fontsize=13)
        for j, (fr_, lab) in enumerate(((MV, "모델 렌더"), (GV, "정답 렌더"))):
            ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
            if fr_:
                ax.imshow(fr_[min(i, len(fr_) - 1)])
            ax.set_title(f"{lab} (상태 {i})", fontsize=10); ax.axis("off")
        ax = fig.add_subplot(gs[0, 4:6], projection="3d")
        sc = sphere(ax, r["de"], 100 * r["nd"] / c["EXT"], "δ 방향 (색 = |δ|, EXT%)")
        mu = (r["m"][:, None] * r["u0"]).sum(0); mu = mu / max(np.linalg.norm(mu), 1e-30)
        ax.scatter([mu[0]], [mu[1]], [mu[2]], color="red", s=200, marker="*", depthshade=False, label="질량평균 u0 방향")
        ax.scatter([-mu[0]], [-mu[1]], [-mu[2]], color="red", s=200, marker="X", depthshade=False, label="−u0 (완전 상쇄 방향)")
        ax.legend(fontsize=7, loc="lower left")
        ax = fig.add_subplot(gs[0, 6:8], projection="3d")
        sphere(ax, r["u0"], 100 * r["nu"] / c["EXT"], "u0 방향 (색 = |u0|, 같은 색 범위)", mk="^")
        cb = fig.add_axes([0.915, 0.74, 0.006, 0.14]); fig.colorbar(sc, cax=cb).set_label("크기 (EXT%)", fontsize=8)
        for row, wt in ((1, False), (2, True)):
            hist(fig.add_subplot(gs[row, 0:3]), r["cos"], r, B_COS, "cos(δ, u0)", -1, wt)
            hist(fig.add_subplot(gs[row, 3:5]), r["lr"], r, B_LR, "log10 |δ|/|u0|", 0, wt)
            hist(fig.add_subplot(gs[row, 5:8]), r["par"], r, B_PAR, "δ·û0/|u0| (−1 = 완전 상쇄)", -1, wt)
        fig.text(0.005, 0.50, "개수 기준", rotation=90, fontsize=12); fig.text(0.005, 0.30, "질량 가중", rotation=90, fontsize=12)
        for j, (k, lab, yl) in enumerate(curve_keys()):
            ax = fig.add_subplot(gs[3, 2 * j:2 * j + 2])
            for g, col, ls, gl in (("all", C_ALL, "-", "전체"), ("con", C_CON, "--", "접촉"), ("air", C_AIR, ":", "공중")):
                ax.plot(x, [q[f"{k}_{g}"] for q in c["fr"]], color=col, ls=ls, lw=1.4, label=gl)
            ax.axvline(i, color="k", lw=1); ax.axvline(c["stop"], color="red", ls="--", lw=0.8)
            ax.set_ylim(*yl); ax.set_xlim(0, c["L"] - 1); ax.set_title(lab, fontsize=10); ax.set_xlabel("프레임")
            ax.axhline(1 if k == "medcan" else (-1 if k == "medcos" else 0), color="#adb5bd", lw=0.6)
            ax.legend(fontsize=7)
        ax = fig.add_subplot(gs[3, 6:8])
        ax.plot(x, c["mot_p"], color="#c92a2a", label="모델"); ax.plot(x, c["mot_g"], color="k", label="정답")
        ax.axvline(i, color="k", lw=1); ax.axvline(c["stop"], color="red", ls="--", lw=0.8)
        ax.set_ylim(0, MOTMAX); ax.set_xlim(0, c["L"] - 1); ax.set_title("프레임 이동량 평균 |Δx| (EXT%)", fontsize=10)
        ax.set_xlabel("프레임"); ax.legend(fontsize=7)
        fig.canvas.draw()
        im = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        W.append_data(im[: im.shape[0] // 2 * 2, : im.shape[1] // 2 * 2])
        plt.close(fig)
        if i % 10 == 0:
            print(f"  [{c['nm']}] 프레임 {i}/{c['L']}", flush=True)
    W.close(); print(f"[영상] {out}", flush=True)

# ---- 요약 그림 ----
TRJ = sorted({c["nm"].split("_")[-1] for c in CS}, key=lambda t: (t != "0004", t))
MOD = []
for c in CS:
    m_ = c["nm"].rsplit("_", 1)[0]
    if m_ not in MOD:
        MOD.append(m_)
MCOL = dict(zip(MOD, ["#c92a2a", "#1971c2", "#2f9e44", "#f08c00"]))
MLAB = {c["nm"].rsplit("_", 1)[0]: c["title"].split(" · ")[0] for c in CS}
rowsK = curve_keys() + [("mot", "프레임 이동량 평균 |Δx| (EXT%)", (0, MOTMAX))]
fig, axs = plt.subplots(len(rowsK), len(TRJ), figsize=(7 * len(TRJ), 3.2 * len(rowsK)), squeeze=False)
for j, t in enumerate(TRJ):
    for c in [c for c in CS if c["nm"].endswith("_" + t)]:
        mo = c["nm"].rsplit("_", 1)[0]; x = np.arange(c["L"])
        for ri, (k, lab, yl) in enumerate(rowsK):
            ax = axs[ri, j]
            if k == "mot":
                ax.plot(x, c["mot_p"], color=MCOL[mo], label=MLAB[mo])
                continue
            for g, ls, gl in (("all", "-", "전체"), ("con", "--", "접촉"), ("air", ":", "공중")):
                ax.plot(x, [q[f"{k}_{g}"] for q in c["fr"]], color=MCOL[mo], ls=ls, lw=1.3, label=f"{MLAB[mo]} {gl}")
            ax.axvline(c["stop"], color=MCOL[mo], ls="-.", lw=0.7)
    gtc = [c for c in CS if c["nm"].endswith("_" + t)][0]
    axs[-1, j].plot(np.arange(gtc["L"]), gtc["mot_g"], color="k", label="정답")
    for ri, (k, lab, yl) in enumerate(rowsK):
        ax = axs[ri, j]; ax.set_ylim(*yl); ax.set_title(f"{t}: {lab}", fontsize=10); ax.set_xlabel("프레임")
        ax.legend(fontsize=6, ncol=2)
fig.suptitle(f"lego 롤아웃 노드 변위 u0+δ 의 상쇄 진단 (접촉 = 지지 입자 최소 바닥 거리 < {a.contact}; 세로 점선 = 멈춤 프레임)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
sp = os.path.join(a.out_fig, "lego_out_cancel_summary.png"); fig.savefig(sp, dpi=110); plt.close(fig); print(f"[그림] {sp}")

# ---- 표 ----
def wavg(c, k, lo, hi):
    v = [q[k] for q in c["fr"][lo:hi]]; v = [z for z in v if z == z]
    return np.mean(v) if v else np.nan


md = [f"# lego 롤아웃: 노드 변위 = u0 + δ 상쇄 진단\n",
      f"접촉 노드 = 그 노드를 꼭짓점으로 쓰는 입자 중 스텝 시작 바닥 거리 최소 < {a.contact}. 상쇄 비율 c = −δ·û0/|u0| (1 = 완전 상쇄), "
      "운동량 비 = |Σm(u0+δ)|/|Σm u0| (0 = 완전 상쇄), '상쇄 노드 비율' = c ∈ (0.8, 1.2) 인 노드 비율. 구간 값은 프레임 평균.\n",
      "| 사례 | 구간 | 오차 % | 이동량 모델/정답 (EXT%) | 중앙값 cos 전체/접촉/공중 | 중앙값 c 전체/접촉/공중 | 질량가중 중앙값 c 전체 | 운동량 비 전체/접촉/공중 | 상쇄 노드 비율 전체/접촉/공중 | 접촉 노드 비율 |",
      "|---|---|---|---|---|---|---|---|---|---|"]
f3 = lambda v: "-" if v != v else f"{v:.3f}"  # noqa: E731
for c in CS:
    s = c["stop"]
    for lo, hi, lab in ((0, s, f"0–{s - 1}"), (s, c["L"], f"{s}–{c['L'] - 1}"), (max(s - 2, 0), min(s + 3, c["L"]), f"멈춤 전후 {max(s - 2, 0)}–{min(s + 2, c['L'] - 1)}")):
        md.append(f"| {c['title']} | {lab} | {np.mean(c['err'][lo:hi]):.2f} | {np.mean(c['mot_p'][lo:hi]):.3f} / {np.mean(c['mot_g'][lo:hi]):.3f} | "
                  + " / ".join(f3(wavg(c, f"medcos_{g}", lo, hi)) for g in ("all", "con", "air")) + " | "
                  + " / ".join(f3(wavg(c, f"medcan_{g}", lo, hi)) for g in ("all", "con", "air")) + f" | {f3(wavg(c, 'wmedcan_all', lo, hi))} | "
                  + " / ".join(f3(wavg(c, f"mom_{g}", lo, hi)) for g in ("all", "con", "air")) + " | "
                  + " / ".join(f3(wavg(c, f"frac_cancel_{g}", lo, hi)) for g in ("all", "con", "air")) + " | "
                  + f3(np.mean([q["n_con"] / max(q["n_all"], 1) for q in c["fr"][lo:hi]])) + " |")
md.append("\n## 프레임별 (멈춤 프레임 −4 ~ +6)\n")
md.append("| 사례 | 프레임 | 오차 % | 이동량 모델/정답 | 중앙값 cos | 중앙값 c 전체/접촉/공중 | 운동량 비 전체/접촉/공중 | 접촉 노드 수/전체 |")
md.append("|---|---|---|---|---|---|---|---|")
for c in CS:
    for i in range(max(c["stop"] - 4, 0), min(c["stop"] + 7, c["L"])):
        q = c["fr"][i]
        md.append(f"| {c['title']} | {i} | {c['err'][i]:.2f} | {c['mot_p'][i]:.3f}/{c['mot_g'][i]:.3f} | {f3(q['medcos_all'])} | "
                  + "/".join(f3(q[f"medcan_{g}"]) for g in ("all", "con", "air")) + " | "
                  + "/".join(f3(q[f"mom_{g}"]) for g in ("all", "con", "air")) + f" | {q['n_con']}/{q['n_all']} |")
mp = os.path.join(a.out_fig, "lego_out_cancel.md"); open(mp, "w").write("\n".join(md) + "\n"); print(f"[표] {mp}")
SK = [k for k in CS[0]["fr"][0] if not isinstance(CS[0]["fr"][0][k], np.ndarray)]
json.dump({c["nm"]: dict(title=c["title"], stop=c["stop"], err=c["err"].tolist(), mot_p=c["mot_p"].tolist(), mot_g=c["mot_g"].tolist(),
                         frames=[{k: q[k] for k in SK} for q in c["fr"]]) for c in CS} | {"vmax_ext_pct": VMAX, "contact": a.contact},
          open(a.json, "w"), indent=1, ensure_ascii=False)
print("OUTCANCEL_VIZ_OK", flush=True)
