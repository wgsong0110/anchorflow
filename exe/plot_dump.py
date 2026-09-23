"""덤프한 활성/출력으로 분포와 정규화 실태를 그린다."""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm
for _p in (os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf"),):
    try: fm.fontManager.addfont(_p)
    except Exception: pass
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

D = torch.load(a.dump, map_location="cpu", weights_only=False)
occ = D["occ"].numpy()
fr, fn = D["feat_raw"].float().numpy(), D["feat_norm"].float().numpy()
dp, u, EXT = D["dp"].float().numpy(), D["gt_u"].float().numpy(), D["EXT"]
acts = D["acts"]
print(f"셀 {occ.size}, 찬 셀 {int(occ.sum())} ({100*occ.mean():.1f}%)")
print(f"출력 scale {D['scale']:.4f}  EXT {EXT:.4f}")

fig = plt.figure(figsize=(16.5, 10.5))
gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.26)

# --- 1) 출력 dp 분포 vs 정답 변위 -------------------------------------------
ax = fig.add_subplot(gs[0, 0])
nd, nu = np.linalg.norm(dp, axis=-1) / EXT, np.linalg.norm(u, axis=-1) / EXT
bins = np.linspace(0, max(np.percentile(nu, 99.9), 1e-9) * 1.4, 90)
ax.hist(100 * nu, bins=100 * bins, color="k", alpha=.45, label="정답 변위 |u| (입자)")
ax.hist(100 * nd, bins=100 * bins, color="crimson", alpha=.6, label="망 출력 |dp| (격자점 전부)")
ax.set_yscale("log"); ax.set_xlabel("EXT 대비 크기 [%]"); ax.set_ylabel("개수")
ax.set_title("출력 크기 분포"); ax.legend(fontsize=8)
ax.text(.98, .55, f"|dp| 중앙 {100*np.median(nd):.4f}%\n|u| 중앙 {100*np.median(nu):.4f}%\n"
        f"비 {np.median(nd)/max(np.median(nu),1e-12):.3f}",
        transform=ax.transAxes, ha="right", fontsize=8.5,
        bbox=dict(fc="w", alpha=.8))

ax = fig.add_subplot(gs[0, 1])
for i, (c, lab) in enumerate(zip("rgb", ("x", "y", "z"))):
    ax.hist(dp[:, i] / EXT * 100, bins=140, histtype="step", color=c, lw=1.5,
            label=f"dp_{lab}")
ax.hist(u.reshape(-1) / EXT * 100, bins=140, histtype="step", color="k", lw=1.5,
        ls="--", label="정답 u (성분)")
ax.set_yscale("log"); ax.set_xlabel("EXT 대비 [%]"); ax.set_title("출력 성분 분포")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
o = acts["out"].numpy()
txt = [f"찬 셀 비율            {100*occ.mean():.1f}%",
       f"|dp| 평균             {np.abs(dp).mean()/EXT*100:.5f}% EXT",
       f"|u|  평균             {np.abs(u).mean()/EXT*100:.5f}% EXT",
       f"출력 포화율 |o|>1      {100*np.mean(np.abs(o)>1):.2f}%",
       f"out 원시값 표준편차     {o.std():.4e}",
       f"scale(=0.02*EXT)      {D['scale']:.4f}",
       "",
       "--- 입력 표준화 (feat-mu)/sd ---",
       f"전체 셀   평균 {fn.mean():+.3f}  표준편차 {fn.std():.3f}",
       f"찬 셀만   평균 {fn[occ].mean():+.3f}  표준편차 {fn[occ].std():.3f}",
       f"빈 셀만   평균 {fn[~occ].mean():+.3f}  표준편차 {fn[~occ].std():.3f}",
       f"|정규화값| 최대          {np.abs(fn).max():.1f}",
       f"sd=1 로 남은(정규화 안 된) 채널 {int((np.abs(D['in_sd'].numpy()-1)<1e-9).sum())}/{fn.shape[1]}"]
ax.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=9.2)
ax.set_title("요약", loc="left")

# --- 2) 채널별 정규화: 전체 vs 찬 셀 ---------------------------------------
ax = fig.add_subplot(gs[1, 0])
ax.plot(fn.mean(0), "k.-", ms=3, lw=.8, label="전체 셀 평균")
ax.plot(fn[occ].mean(0), "r.-", ms=3, lw=.8, label="찬 셀만 평균")
ax.axhline(0, color="gray", lw=.8); ax.set_xlabel("입력 채널")
ax.set_title("표준화 후 채널 평균 — 전체는 0인데 찬 셀만 보면 크게 치우친다")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
ax.plot(fn.std(0), "k.-", ms=3, lw=.8, label="전체 셀 표준편차")
ax.plot(fn[occ].std(0), "r.-", ms=3, lw=.8, label="찬 셀만 표준편차")
ax.axhline(1, color="gray", lw=.8); ax.set_yscale("log")
ax.set_xlabel("입력 채널"); ax.set_title("표준화 후 채널 표준편차")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 2])
ax.hist(fn[~occ].reshape(-1), bins=200, range=(-8, 8), color="gray", alpha=.6,
        label="빈 셀", density=True)
ax.hist(fn[occ].reshape(-1), bins=200, range=(-8, 8), color="crimson", alpha=.6,
        label="찬 셀", density=True)
ax.set_yscale("log"); ax.set_xlabel("표준화된 입력값")
ax.set_title("표준화된 입력 분포"); ax.legend(fontsize=8)

# --- 3) 층별 활성 분포 -------------------------------------------------------
order = ["c2g", "inp"] + sorted([k for k in acts if k.startswith("body")]) + ["out"]
ax = fig.add_subplot(gs[2, :2])
pos, labs = [], []
for i, k in enumerate(order):
    v = acts[k].numpy()
    # 격자점 기준이라 셀 마스크를 그대로 못 쓴다 -> 전체/상위분위로 본다
    ax.boxplot([v.reshape(-1)[::37]], positions=[i], widths=.6, showfliers=False)
    pos.append(i); labs.append(k)
    ax.text(i, 0, f"σ={v.std():.2f}", ha="center", va="top", fontsize=7.5,
            transform=ax.get_xaxis_transform())
ax.set_xticks(pos); ax.set_xticklabels(labs, fontsize=8.5)
ax.set_title("층별 활성 분포 (GroupNorm 이 블록마다 들어 있다)")
ax.grid(alpha=.3)

ax = fig.add_subplot(gs[2, 2])
v = acts[order[-2]].numpy()
mag = np.linalg.norm(v, axis=0)
ax.hist(mag, bins=150, color="steelblue")
ax.set_yscale("log"); ax.set_title(f"마지막 블록 활성 크기 ({order[-2]})")
ax.set_xlabel("격자점별 활성 노름")

fig.suptitle("학습된 conv_r32(6000스텝) 한 프레임 진단 — 출력 분포와 정규화 실태",
             fontsize=13)
fig.savefig(a.out, dpi=118, bbox_inches="tight")
print("저장", a.out)
