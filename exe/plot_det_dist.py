"""det(F) 분포 비교: MPM 과 우리 (히스토그램 + 프레임별 백분위)."""
from __future__ import annotations
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import os as _os
for _p in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf",
           _os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")):
    try: fm.fontManager.addfont(_p)
    except Exception: pass
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--dumps", nargs="+", required=True, help="이름=덤프경로 (우리)")
ap.add_argument("--pg", default="", help="PG 궤적 .pt (F 를 가진 것) 또는 "
                                         "gt_fscal 을 가진 덤프")
ap.add_argument("--title", default="")
ap.add_argument("--lo", type=float, default=-1.0)
ap.add_argument("--hi", type=float, default=2.0)
ap.add_argument("--drop_handle", default="", help="이 덤프의 손잡이(ctrl) 안 "
                "입자를 모든 계열에서 뺀다. 'init' 이면 초기 위치 기준, "
                "'ever' 면 한 번이라도 들어간 적 있는 입자까지 뺀다")
ap.add_argument("--handle_mode", choices=["init", "ever"], default="init")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


keep = None
if a.drop_handle:
    Dh = L(a.drop_handle)
    X0 = np.asarray(Dh["x0"], dtype=np.float64)
    R = float(np.asarray(Dh["ctrl_R"]).reshape(-1)[0])
    if "ctrl_id" in Dh:
        cid = int(np.asarray(Dh["ctrl_id"]).reshape(-1)[0])
        c0 = X0[cid]
    else:
        c0 = np.asarray(Dh["ctrl_pos"], dtype=np.float64)[0, 0]
    if a.handle_mode == "init":
        keep = np.linalg.norm(X0 - c0, axis=1) > R
    else:
        P = np.asarray(Dh["pred"], dtype=np.float64)
        CP = np.asarray(Dh["ctrl_pos"], dtype=np.float64)[:, 0, :]
        T = P.shape[0]
        inside = np.zeros(P.shape[1], bool)
        for t in range(T):
            c = (P[t, cid] if "ctrl_id" in Dh else CP[min(t, len(CP) - 1)])
            inside |= np.linalg.norm(P[t] - c, axis=1) <= R
        keep = ~inside
    print(f"[손잡이 제외] {a.handle_mode}: {int((~keep).sum())} 개 뺀다 "
          f"(남는 입자 {int(keep.sum())})", flush=True)

series = []
T0 = None
for s in a.dumps:
    nm, path = s.split("=", 1)
    d = np.asarray(L(path)["fscal"])[..., 0].astype(np.float64)
    if keep is not None:
        d = d[:, keep]
    series.append((nm, d))
    T0 = d.shape[0] if T0 is None else T0
if a.pg:
    D = L(a.pg)
    if "gt_fscal" in D:
        g = np.asarray(D["gt_fscal"])[..., 0].astype(np.float64)
    else:
        F = D["F"]
        g = torch.linalg.det(F.double()).numpy()
    g = g[:T0]
    if keep is not None:
        g = g[:, keep]
    series.insert(0, ("PG MPM", g))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), dpi=120)
bins = np.linspace(a.lo, a.hi, 160)
# 한 실행의 히스토그램·중앙값·띠·최솟값은 **같은 색**이어야 읽힌다
# (기본 색 순환에 맡기면 fill_between 과 최솟값 선이 다른 색을 집어간다)
COLS = ["black", "crimson", "royalblue", "darkorange", "seagreen", "purple"]
for k, (nm, d) in enumerate(series):
    col = COLS[k % len(COLS)]
    v = d.ravel()
    ax[0].hist(np.clip(v, a.lo, a.hi), bins=bins, histtype="step", lw=1.8,
               density=True, label=nm, color=col)
    q = np.percentile(d, [0, 1, 50, 99], axis=1)
    ax[1].plot(np.arange(d.shape[0]), q[2], lw=2, label=f"{nm} 중앙값", color=col)
    ax[1].fill_between(np.arange(d.shape[0]), q[1], q[3], alpha=0.15, color=col)
    ax[1].plot(np.arange(d.shape[0]), q[0], lw=1, ls=":", alpha=0.9, color=col)
    print(f"[{nm}] det 백분위 0/1/5/50/95/100 = "
          f"{np.round(np.percentile(d, [0, 1, 5, 50, 95, 100]), 4).tolist()}, "
          f"det<0 비율 {100.0 * (d < 0).mean():.3f}%, "
          f"det<0.1 비율 {100.0 * (d < 0.1).mean():.3f}%", flush=True)
ax[0].set_xlabel("det(F)"); ax[0].set_ylabel("밀도")
ax[0].set_title(f"전 프레임·전 입자 분포 ({a.lo:g} ~ {a.hi:g} 로 자름)")
ax[0].axvline(0, color="k", lw=1, ls="--"); ax[0].set_yscale("log")
ax[1].set_xlabel("프레임"); ax[1].set_ylabel("det(F)")
ax[1].set_title("프레임별 중앙값 (띠 = 1~99%, 점선 = 최솟값)")
ax[1].axhline(0, color="k", lw=1, ls="--")
for q in ax:
    q.grid(alpha=0.25); q.legend(fontsize=9)
fig.suptitle(a.title, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)
