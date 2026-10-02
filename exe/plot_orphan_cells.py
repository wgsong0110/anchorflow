"""**고아 셀**(26 이웃이 모두 빈 셀) 비율을 시간에 따라, 해상도별로.

고아 셀의 입자는 자기 셀 꼭짓점을 다른 입자와 전혀 공유하지 않아, 그 노드의
자유도를 그 셀이 혼자 결정한다 -- 국소적으로 미결정이 되어 거의 공짜로 움직인다.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn.functional as Fn
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
ap.add_argument("--dump", required=True, help="롤아웃 덤프 (pred 를 쓴다)")
ap.add_argument("--nodes", type=float, nargs="+", default=[90, 60, 45, 30])
ap.add_argument("--mark", type=int, nargs="+", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


D = L(a.dump)
P = D["pred"].float()
T = P.shape[0]


def orphan_frac(x, n_nodes):
    lo, lat, nn = SX.grid_for_nodes(x, n_nodes)
    _i, _l, aux = SX.locate(x, lo, lat, nn)
    ci = aux[0].long()
    lo_i = ci.min(0).values
    ci0 = ci - lo_i
    dims = (ci0.max(0).values + 1).tolist()
    occ = torch.zeros(dims, device=dev)
    occ.index_put_((ci0[:, 0], ci0[:, 1], ci0[:, 2]),
                   torch.ones(ci0.shape[0], device=dev), accumulate=True)
    occb = (occ > 0).float()
    s = Fn.conv3d(occb.reshape(1, 1, *dims),
                  torch.ones(1, 1, 3, 3, 3, device=dev),
                  padding=1).reshape(*dims)
    nb = s - occb                      # 26 이웃 중 점유 수
    sel = occb > 0
    orph = (nb[sel] == 0)
    return float(orph.float().mean()), int(sel.sum())


fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.6), dpi=120)
cm = plt.get_cmap("viridis")
for k, n_ in enumerate(a.nodes):
    ys, ns = [], []
    for t in range(T):
        f, nc = orphan_frac(P[t].to(dev), n_)
        ys.append(100 * f); ns.append(nc)
    ax.plot(np.arange(T), ys, color=cm(k / max(len(a.nodes) - 1, 1)),
            label=f"격자 {n_:g} (점유 셀 {ns[0]}→{ns[-1]})")
    print(f"[격자 {n_:6g}] 고아 셀 비율 처음 {ys[0]:6.2f}% 최대 {max(ys):6.2f}% "
          f"끝 {ys[-1]:6.2f}%  (점유 셀 {ns[0]} -> {ns[-1]})", flush=True)
for m in a.mark:
    ax.axvline(m, color="k", ls=":", lw=1.2)
    ax.text(m, ax.get_ylim()[1], f" {m}", va="top", fontsize=9)
ax.set_xlabel("프레임"); ax.set_ylabel("고아 셀 비율 (%)")
ax.set_title("시간에 따른 고아 셀 (26 이웃이 모두 빈 셀)", fontsize=11)
ax.grid(alpha=.3); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)
