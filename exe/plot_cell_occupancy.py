"""격자 해상도별로 **이웃 빈 셀 개수**의 분포를 그린다.

입자가 없는 셀은 증분 포텐셜에 아무 기여도 하지 않아 자유롭게 일그러진다.
그런 셀이 손잡이에서 몸통으로 가는 응력 사슬을 끊으므로, 점유 셀 하나가
이웃 26 칸 중 몇 칸이 비어 있는지가 그 사슬의 성김을 말해 준다.

셀은 **능면체 셀**(정수 좌표 한 칸, Kuhn 분할 전)을 센다.
"""
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

from anchorflow import simplex as SX

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--nodes", type=int, nargs="+", default=[32, 16, 11, 6])
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


d = L(a.traj)
x = d["x"][a.frame].to(dev).float()
fig, ax = plt.subplots(1, len(a.nodes), figsize=(4.6 * len(a.nodes), 4.2),
                       dpi=120, squeeze=False)
for k, nn_ in enumerate(a.nodes):
    lo, lat, nn = SX.grid_for_nodes(x, nn_)
    _idx, _lam, aux = SX.locate(x, lo, lat, nn)
    ci = aux[0].long()                                  # 입자가 든 셀 [N,3]
    lo_i = ci.min(0).values
    ci0 = ci - lo_i
    dims = (ci0.max(0).values + 1).tolist()
    occ = torch.zeros(dims, device=dev)
    occ.index_put_((ci0[:, 0], ci0[:, 1], ci0[:, 2]),
                   torch.ones(ci0.shape[0], device=dev), accumulate=True)
    occb = (occ > 0).float()
    # 26 이웃 중 빈 칸 수 = 26 - (3x3x3 합 - 자기 자신)
    s = torch.nn.functional.conv3d(
        occb.reshape(1, 1, *dims),
        torch.ones(1, 1, 3, 3, 3, device=dev), padding=1).reshape(*dims)
    empty_nb = 26.0 - (s - occb)
    sel = occb > 0
    v = empty_nb[sel].cpu().numpy()
    npc = occ[sel].cpu().numpy()
    h = float((1.1 * float((x.max(0).values - x.min(0).values).max()))
              / max(nn_, 2))
    q = ax[0][k]
    q.hist(v, bins=np.arange(-0.5, 27.5, 1.0), color="tab:blue")
    q.set_xlabel("이웃 26 칸 중 빈 칸 수"); q.set_ylabel("점유 셀 수")
    q.set_title(f"격자 {nn_}  (h {h:.4f})\n점유 셀 {int(sel.sum())}, "
                f"셀당 입자 중앙 {np.median(npc):.1f}\n"
                f"빈 이웃 중앙 {np.median(v):.0f}, 평균 {v.mean():.1f}",
                fontsize=10)
    print(f"[격자 {nn_:2d}] h {h:.4f}  점유 셀 {int(sel.sum())}  "
          f"셀당 입자 중앙 {np.median(npc):.1f}  빈 이웃 평균 {v.mean():.2f} "
          f"(26 칸 다 빈 셀 {100*float((v == 26).mean()):.1f}%)", flush=True)
fig.tight_layout(); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)
