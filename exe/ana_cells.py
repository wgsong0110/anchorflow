"""유효(점유) 셀당 입자수 분포, 빈 셀의 이웃 상황, 유효 셀의 세 방향 투영.

격자는 각 씬의 MPM dx 에 맞춘 node_h 로 만든다 (셀 부피를 dx³ 에 맞춘 것과 같다).
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
import sys
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "lib"))
from anchorflow import simplex as SX                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", nargs="+", required=True,
                help="이름=궤적.pt=node_h 형태")
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


rows = []
for sp in a.scenes:
    nm, path, hs = sp.split("=")
    d = L(path)
    X = d["x"] if "x" in d else d["pred"]
    t = min(a.frame, X.shape[0] - 1)
    x = X[t].float()
    lo, lat, nn = SX.grid_for_nodes(x, 40.0, h_fix=float(hs))
    nnl = [int(nn[k]) for k in range(3)]
    y = (x - lo) @ lat.Ai.T
    ci = y.floor().long()
    ci = torch.stack([ci[:, k].clamp(0, nnl[k] - 2) for k in range(3)], -1)
    flat = ((ci[:, 0] * nnl[1] + ci[:, 1]) * nnl[2] + ci[:, 2]).numpy()
    uq, cnt = np.unique(flat, return_counts=True)
    # 빈 셀 중 이웃에 유효 셀이 있는 것 (6 면, 26 이웃 둘 다)
    cs = np.stack([uq // (nnl[1] * nnl[2]), (uq // nnl[2]) % nnl[1],
                   uq % nnl[2]], -1)
    def nb(offsets):
        cand = set()
        for o in offsets:
            v = cs + np.asarray(o)
            ok = ((v >= 0).all(1) & (v[:, 0] < nnl[0]) & (v[:, 1] < nnl[1])
                  & (v[:, 2] < nnl[2]))
            f = ((v[ok, 0] * nnl[1] + v[ok, 1]) * nnl[2] + v[ok, 2])
            cand.update(f.tolist())
        return cand - set(uq.tolist())
    o6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    o26 = [(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)
           if (i,j,k) != (0,0,0)]
    e6, e26 = len(nb(o6)), len(nb(o26))
    V = float(abs(torch.linalg.det(lat.A.double())))
    print(f"[{nm}] 프레임 {t}, 입자 {x.shape[0]}, 격자 {nnl}, h {float(lat.s):.5f}, "
          f"셀부피 {V:.3e}", flush=True)
    print(f"   유효 셀 {len(uq)}, 셀당 입자 평균 {cnt.mean():.2f} "
          f"중앙 {int(np.median(cnt))} 1/5/95/99% "
          f"{np.percentile(cnt,[1,5,95,99]).round(1).tolist()} 최대 {cnt.max()}, "
          f"입자 1 개뿐인 셀 {(cnt==1).sum()} ({100*(cnt==1).mean():.1f}%)",
          flush=True)
    print(f"   빈 셀인데 이웃에 유효 셀 있음: 6면 기준 {e6}, 26이웃 기준 {e26} "
          f"(유효 셀의 {100*e6/len(uq):.1f}% / {100*e26/len(uq):.1f}%)", flush=True)
    rows.append((nm, cnt, cs, lat, lo, nnl, e6, e26))

n = len(rows)
fig, axs = plt.subplots(n, 4, figsize=(19, 4.4 * n), dpi=110, squeeze=False)
for r, (nm, cnt, cs, lat, lo, nnl, e6, e26) in enumerate(rows):
    q = axs[r][0]
    mx = int(np.percentile(cnt, 99.5))
    q.hist(cnt, bins=np.arange(0.5, max(mx, 2) + 1.5), color="steelblue")
    q.set_xlabel("셀당 입자수"); q.set_ylabel("셀 수")
    q.set_title(f"{nm}  유효 셀 {len(cnt)}\n평균 {cnt.mean():.1f} 중앙 "
                f"{int(np.median(cnt))}  빈셀(6면) {e6}", fontsize=10)
    q.grid(alpha=0.25)
    wc = (cs.astype(np.float64) + 0.5) @ np.asarray(lat.A.T, dtype=np.float64) \
        + np.asarray(lo, dtype=np.float64)
    for k, ((i, j), lu, lv) in enumerate(((((0, 2)), "x", "z"),
                                          (((0, 1)), "x", "y"),
                                          (((1, 2)), "y", "z"))):
        p = axs[r][k + 1]
        p.scatter(wc[:, i], wc[:, j], s=1.0, c=cnt, cmap="viridis", lw=0)
        p.set_aspect("equal"); p.grid(alpha=0.2)
        p.set_xlabel(lu); p.set_ylabel(lv)
        p.set_title(f"{nm} 유효 셀 {lu}{lv} 투영 (색 = 입자수)", fontsize=10)
fig.tight_layout(); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)
