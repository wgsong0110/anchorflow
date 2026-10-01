"""**구 내부 빈 셀의 비율**을 (1) 격자 해상도별 (2) 같은 부피에서 층 간격별로.

입자가 없는 셀은 증분 포텐셜에 아무 기여도 하지 않아 자유롭게 일그러지고,
손잡이에서 몸통으로 가는 응력 사슬을 거기서 끊는다. "구 내부" 는 셀 중심이
공 안(초기 입자들의 중심에서 최대 반지름 안)에 드는 셀로 정의한다.
"""
from __future__ import annotations
import argparse
import math
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

from anchorflow import simplex as SX

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--nodes", type=float, nargs="+",
                default=[6, 8, 11, 16, 24, 32, 48])
ap.add_argument("--hz_mult", type=float, nargs="+",
                default=[0.125, 0.25, 0.5, 1, 2, 4])
ap.add_argument("--vol_n", type=float, default=11.0,
                help="부피를 고정할 기준 (이 n_nodes 의 정육면체 셀 부피)")
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


d = L(a.traj)
x = d["x"][a.frame].to(dev).float()
BC = x.mean(0)
BR = float((x - BC).norm(dim=-1).max())
print(f"[공] 중심 {BC.tolist()}  반지름 {BR:.4f}  입자 {x.shape[0]}", flush=True)


def empty_frac(n_nodes, hz_ratio):
    """-> (빈 비율, 내부 셀 수, h, hz, 셀 부피)"""
    lo, lat, nn = SX.grid_for_nodes(x, n_nodes, hz_ratio=hz_ratio)
    _i, _l, aux = SX.locate(x, lo, lat, nn)
    ci = aux[0].long()
    lo_i, hi_i = ci.min(0).values, ci.max(0).values
    rng = [torch.arange(int(lo_i[k]), int(hi_i[k]) + 1, device=dev)
           for k in range(3)]
    gi = torch.stack(torch.meshgrid(*rng, indexing="ij"), -1).reshape(-1, 3)
    # 셀 중심의 월드 위치 = lo + A (i + 1/2)
    cw = (gi.to(lat.A.dtype) + 0.5) @ lat.A.T + lo
    inside = ((cw - BC).norm(dim=-1) <= BR)
    key = ((gi - lo_i) * torch.tensor(
        [(hi_i[1] - lo_i[1] + 1) * (hi_i[2] - lo_i[2] + 1),
         hi_i[2] - lo_i[2] + 1, 1], device=dev)).sum(-1)
    okey = ((ci - lo_i) * torch.tensor(
        [(hi_i[1] - lo_i[1] + 1) * (hi_i[2] - lo_i[2] + 1),
         hi_i[2] - lo_i[2] + 1, 1], device=dev)).sum(-1)
    occ = torch.zeros(int(key.max()) + 1, dtype=torch.bool, device=dev)
    occ[okey] = True
    ins_key = key[inside]
    n_in = int(inside.sum())
    n_occ = int(occ[ins_key].sum())
    h = float(lat.s)
    hz = float(lat.hz)
    vol = (math.sqrt(3) / 2) * h * h * hz
    return (1.0 - n_occ / max(n_in, 1)), n_in, h, hz, vol


fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=120)

xs, ys = [], []
for n_ in a.nodes:
    f, n_in, h, hz, vol = empty_frac(n_, SX.HZ_CUBE)
    xs.append(n_); ys.append(100 * f)
    print(f"[해상도] n {n_:6.2f}  h {h:.5f}  내부 셀 {n_in:7d}  "
          f"빈 비율 {100*f:6.2f}%  셀 부피 {vol:.3e}", flush=True)
ax[0].plot(xs, ys, "o-", color="tab:blue")
ax[0].set_xlabel("n_nodes (물체를 몇 칸으로)")
ax[0].set_ylabel("구 내부 빈 셀 비율 (%)")
ax[0].set_title("격자 해상도별 (셀은 정육면체)", fontsize=11)
ax[0].grid(alpha=.3)

h0 = None
_, _, h0, _, V0 = empty_frac(a.vol_n, SX.HZ_CUBE)[0:5]
xs2, ys2 = [], []
for m in a.hz_mult:
    rho = SX.HZ_CUBE * m
    hh = (2 * V0 / (math.sqrt(3) * rho)) ** (1.0 / 3.0)
    ext = float((x.max(0).values - x.min(0).values).max())
    n_ = (ext * 1.1) / hh
    f, n_in, h, hz, vol = empty_frac(n_, rho)
    xs2.append(rho); ys2.append(100 * f)
    print(f"[층간격] hz/h {rho:.5f} (x{m})  h {h:.5f} hz {hz:.6f}  "
          f"내부 셀 {n_in:7d}  빈 비율 {100*f:6.2f}%  셀 부피 {vol:.3e}",
          flush=True)
ax[1].plot(xs2, ys2, "o-", color="tab:red")
ax[1].set_xscale("log")
ax[1].set_xlabel("hz / h (층 간격 비율, 로그)")
ax[1].set_ylabel("구 내부 빈 셀 비율 (%)")
ax[1].set_title(f"셀 부피 고정 ({V0:.3e}) 에서 층 간격별", fontsize=11)
ax[1].grid(alpha=.3, which="both")
ax[1].axvline(SX.HZ_CUBE, color="0.5", ls="--", lw=1)
ax[1].text(SX.HZ_CUBE, max(ys2) * .95, " 정육면체", fontsize=9, color="0.4")

fig.tight_layout(); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)
