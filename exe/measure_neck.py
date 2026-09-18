"""당겨지는 막대가 **목이 가늘어지다 끊겼는지**를 눈이 아니라 수치로 본다.

점 구름을 돌려 봐도 조각이 겹쳐 보여 판정이 안 된다. 그래서 두 가지를 잰다.

  목 단면   당기는 축으로 입자를 잘게 나눠 칸마다 개수를 센다. 가장 적은 칸이
            처음 대비 몇 분의 일이 되었는지가 목이 얼마나 가늘어졌나이고,
            0 이 되면 그 자리에서 끊긴 것이다
  덩어리    마지막 프레임에서 격자로 이어진 덩어리를 센다 (셀 하나 이내로 닿아
            있으면 같은 덩어리). 전체의 1% 를 넘는 덩어리가 둘 이상이면 분리다
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--h5_dir", required=True)
ap.add_argument("--tag", default="run")
ap.add_argument("--out", default=None)
ap.add_argument("--axis", type=int, default=1, help="당기는 축 (0=x,1=y,2=z)")
ap.add_argument("--bins", type=int, default=60)
ap.add_argument("--cell", type=float, default=0.0,
                help="덩어리 판정 격자 한 변. 0 이면 입자 간격 중앙값의 2 배")
ap.add_argument("--every", type=int, default=20)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
fs = sorted(glob.glob(os.path.join(a.h5_dir, "*.h5")))
if not fs:
    raise SystemExit(f"h5 가 없다: {a.h5_dir}")


def rd(p):
    with h5py.File(p, "r") as h:
        d = np.array(h["x"])
    x = torch.from_numpy(d.T if d.shape[0] == 3 else d).float()
    return x[torch.isfinite(x).all(1)]


X0 = rd(fs[0]).to(dev)
s = X0[torch.randperm(X0.shape[0], device=dev)[:3000]]
SP = float(torch.cdist(s, s).topk(2, largest=False).values[:, 1].median())
CELL = a.cell if a.cell > 0 else 2.0 * SP
print(f"[설정] {len(fs)} 프레임, 입자 {X0.shape[0]}, 간격 중앙 {SP:.5f}, "
      f"덩어리 격자 {CELL:.5f}, 당기는 축 {'xyz'[a.axis]}", flush=True)


def neck(x):
    """당기는 축으로 나눈 칸별 개수. 물체가 실제로 차지한 구간만 본다."""
    u = x[:, a.axis]
    lo, hi = float(u.quantile(0.005)), float(u.quantile(0.995))
    b = ((u - lo) / max(hi - lo, 1e-9) * a.bins).long().clamp(0, a.bins - 1)
    c = torch.bincount(b, minlength=a.bins)
    return c, hi - lo


def components(x):
    """격자 칸 단위로 이어진 덩어리 수. 칸 하나 이내면 이어진 것으로 본다."""
    v = torch.floor(x / CELL).long()
    v = v - v.min(0).values
    D = (v.max(0).values + 3).tolist()
    key = (v[:, 0] * D[1] + v[:, 1]) * D[2] + v[:, 2]
    uk, inv = torch.unique(key, return_inverse=True)
    n = uk.numel()
    kx = uk // (D[1] * D[2]); ky = (uk // D[2]) % D[1]; kz = uk % D[2]
    parent = torch.arange(n, device=dev)

    def find(i):
        while True:
            p = parent[i]
            if bool((p == i).all()):
                return i
            i = p

    # 26 이웃 중 한쪽 방향만 보면 충분하다 (합집합은 대칭이다)
    off = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1),
           (1, 1, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1), (1, 1, -1),
           (1, -1, 1), (1, -1, -1)]
    pr = parent.cpu().numpy()

    def rt(i):
        while pr[i] != i:
            pr[i] = pr[pr[i]]
            i = pr[i]
        return i

    lut = {int(k): i for i, k in enumerate(uk.cpu().numpy())}
    kxn, kyn, kzn = kx.cpu().numpy(), ky.cpu().numpy(), kz.cpu().numpy()
    for i in range(n):
        for dx, dy, dz in off:
            q = ((kxn[i] + dx) * D[1] + (kyn[i] + dy)) * D[2] + (kzn[i] + dz)
            j = lut.get(int(q))
            if j is None:
                continue
            ri, rj = rt(i), rt(j)
            if ri != rj:
                pr[ri] = rj
    root = np.array([rt(i) for i in range(n)])
    cnt = torch.bincount(torch.from_numpy(root[inv.cpu().numpy()]).long())
    big = cnt[cnt > 0.01 * x.shape[0]]
    return int(big.numel()), sorted((big / x.shape[0] * 100).tolist(),
                                    reverse=True)[:4]


c0, ext0 = neck(X0)
occ0 = float(c0[c0 > 0].float().median())
rows = []
for i in range(0, len(fs), a.every):
    x = rd(fs[i]).to(dev)
    c, ext = neck(x)
    mn = int(c.min())
    rows.append(dict(f=i, ext=ext / ext0, neck=mn / occ0, empty=int((c == 0).sum())))
    print(f"  f{i:4d}  길이 {ext / ext0:5.2f} 배  목 {mn / occ0:5.3f} "
          f"(빈 칸 {int((c == 0).sum())}/{a.bins})", flush=True)

xl = rd(fs[-1]).to(dev)
nc, share = components(xl)
cl, extl = neck(xl)
print(f"\n[마지막] 길이 {extl / ext0:.2f} 배, 목 {int(cl.min()) / occ0:.3f}, "
      f"빈 칸 {int((cl == 0).sum())}/{a.bins}", flush=True)
print(f"[덩어리] 전체의 1% 넘는 덩어리 {nc} 개, 비중 "
      + ", ".join(f"{s:.1f}%" for s in share), flush=True)
print("[판정] " + ("끊어짐 (덩어리 2 개 이상)" if nc >= 2 else
                  "안 끊어짐 (한 덩어리)"), flush=True)
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(tag=a.tag, rows=rows, n_comp=nc, share=share,
                   ext_last=extl / ext0, neck_last=int(cl.min()) / occ0),
              open(a.out, "w"), indent=1)
    print(f"[저장] {a.out}", flush=True)
print("NECK_OK", flush=True)
