"""셀 통계를 **사면체 기준**으로 낸다.

한 평행육면체 칸은 Kuhn 분할로 사면체 6 개다. 유효(점유) 사면체당 입자수,
빈 사면체 중 면을 맞댄 이웃에 유효 사면체가 있는 것, 그리고 유효 사면체
무게중심의 세 방향 투영을 낸다.
"""
from __future__ import annotations
import argparse
import os as _os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
for _p in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf",
           _os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")):
    try: fm.fontManager.addfont(_p)
    except Exception: pass
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "lib"))
from anchorflow import simplex as SX                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", nargs="+", required=True, help="이름=궤적.pt=node_h")
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

PERMS = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2],
                  [1, 2, 0], [2, 0, 1], [2, 1, 0]])


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


def tets_of_cells(cells, nnl):
    """셀 정수좌표 [C,3] -> 그 셀의 사면체 6 개의 꼭짓점 평탄색인 [6C,4]."""
    out = []
    for pm in PERMS:
        v = np.zeros((cells.shape[0], 4, 3), dtype=np.int64)
        v[:, 0] = cells
        acc = cells.copy()
        for r in range(3):
            acc = acc.copy()
            acc[:, pm[r]] += 1
            v[:, r + 1] = acc
        out.append(v)
    v = np.concatenate(out, 0)                       # [6C,4,3]
    return (v[..., 0] * nnl[1] + v[..., 1]) * nnl[2] + v[..., 2]


rows = []
for sp in a.scenes:
    nm, path, hs = sp.split("=")
    d = L(path)
    X = d["x"] if "x" in d else d["pred"]
    t = min(a.frame, X.shape[0] - 1)
    x = X[t].float()
    lo, lat, nn = SX.grid_for_nodes(x, 40.0, h_fix=float(hs))
    nnl = [int(nn[k]) for k in range(3)]
    idx, lam, aux = SX.locate(x, lo, lat, nn)
    tv = np.sort(idx.numpy(), axis=1)                # 입자의 사면체 (꼭짓점 4)
    uq, inv, cnt = np.unique(tv, axis=0, return_inverse=True,
                             return_counts=True)
    ci = aux[0].numpy()
    ucell = np.unique(ci, axis=0)
    # 후보: 점유 칸 + 그 6 면 이웃 칸 (면을 맞댄 사면체는 이 안에 다 들어온다)
    offs = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0],
                     [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    cand = np.concatenate([ucell + o for o in offs], 0)
    ok = ((cand >= 0).all(1) & (cand[:, 0] < nnl[0] - 1)
          & (cand[:, 1] < nnl[1] - 1) & (cand[:, 2] < nnl[2] - 1))
    cand = np.unique(cand[ok], axis=0)
    allt = np.sort(tets_of_cells(cand, nnl), axis=1)   # [T,4]
    allt = np.unique(allt, axis=0)
    # 유효 표시
    kk = np.ascontiguousarray(allt).view([('', allt.dtype)] * 4).ravel()
    ku = np.ascontiguousarray(uq).view([('', uq.dtype)] * 4).ravel()
    occ = np.isin(kk, ku)
    # 면(꼭짓점 3 개) -> 사면체 묶기
    faces = np.concatenate([allt[:, [1, 2, 3]], allt[:, [0, 2, 3]],
                            allt[:, [0, 1, 3]], allt[:, [0, 1, 2]]], 0)
    owner = np.tile(np.arange(allt.shape[0]), 4)
    fu, finv = np.unique(faces, axis=0, return_inverse=True)
    order = np.argsort(finv, kind="stable")
    fi, ow = finv[order], owner[order]
    # 같은 면을 가진 사면체 쌍에서 "빈 사면체인데 이웃이 유효" 를 센다
    bnd = np.flatnonzero(np.diff(fi)) + 1
    grp = np.split(ow, bnd)
    empty_touch = set()
    nb_empty = np.zeros(allt.shape[0], dtype=np.int32)
    for g in grp:
        if g.size < 2:
            continue
        o = occ[g]
        if o.any():
            for q in g[~o]:
                empty_touch.add(int(q))
        for q in g[o]:
            nb_empty[q] += int((~o).sum())
    V = float(abs(torch.linalg.det(lat.A.double()))) / 6.0
    print(f"[{nm}] 프레임 {t}, 입자 {x.shape[0]}, 격자 {nnl}, h {float(lat.s):.5f}, "
          f"사면체 부피 {V:.3e}", flush=True)
    print(f"   유효 사면체 {len(uq)}, 사면체당 입자 평균 {cnt.mean():.2f} "
          f"중앙 {int(np.median(cnt))} 5/95/99% "
          f"{np.percentile(cnt,[5,95,99]).round(1).tolist()} 최대 {cnt.max()}, "
          f"입자 1 개뿐 {int((cnt==1).sum())} ({100*(cnt==1).mean():.1f}%)",
          flush=True)
    print(f"   빈 사면체인데 면 맞댄 이웃에 유효 사면체 있음: {len(empty_touch)} "
          f"(유효 사면체의 {100*len(empty_touch)/len(uq):.0f}%), "
          f"유효 사면체 하나당 빈 이웃 평균 "
          f"{nb_empty[occ].mean():.2f} / 4", flush=True)
    # 무게중심
    vv = np.stack([uq // (nnl[1] * nnl[2]), (uq // nnl[2]) % nnl[1],
                   uq % nnl[2]], -1).astype(np.float64)       # [M,4,3]
    cen = vv.mean(1) @ np.asarray(lat.A.T, dtype=np.float64) \
        + np.asarray(lo, dtype=np.float64)
    rows.append((nm, cnt, cen, len(empty_touch), float(nb_empty[occ].mean())))

n = len(rows)
fig, axs = plt.subplots(n, 4, figsize=(19, 4.4 * n), dpi=110, squeeze=False)
for r, (nm, cnt, cen, et, nbm) in enumerate(rows):
    q = axs[r][0]
    mx = max(int(np.percentile(cnt, 99.5)), 2)
    q.hist(cnt, bins=np.arange(0.5, mx + 1.5), color="steelblue")
    q.set_xlabel("사면체당 입자수"); q.set_ylabel("사면체 수")
    q.set_title(f"{nm}  유효 사면체 {len(cnt)}\n평균 {cnt.mean():.2f} 중앙 "
                f"{int(np.median(cnt))}  빈 이웃 {nbm:.2f}/4", fontsize=10)
    q.grid(alpha=0.25)
    for k, ((i, j), lu, lv) in enumerate((((0, 2), "x", "z"),
                                          ((0, 1), "x", "y"),
                                          ((1, 2), "y", "z"))):
        p = axs[r][k + 1]
        p.scatter(cen[:, i], cen[:, j], s=0.8, c=cnt, cmap="viridis", lw=0)
        p.set_aspect("equal"); p.grid(alpha=0.2)
        p.set_xlabel(lu); p.set_ylabel(lv)
        p.set_title(f"{nm} 유효 사면체 {lu}{lv} 투영 (색 = 입자수)", fontsize=10)
fig.tight_layout(); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)
