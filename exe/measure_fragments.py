"""조각 개수와 **어디서 터졌는지**를 잰다.

두 가지를 본다.

1. **조각** -- t0 의 입자 간격으로 정한 반경 r 안에서 이어진 입자끼리 묶어
   연결 성분을 센다. 예전에 쓰던 *격자* 연결 성분은 틀린 잣대였다 (실제로 깨지는
   GF 수박도 "덩어리 1 개(99.9%)" 로 나왔다). 여기서는 격자가 아니라 **입자
   그래프**이고 반경이 초기 간격에 묶여 있어, 조각이 벌어지면 실제로 갈라진다.

2. **터진 자리** -- 접합 시험의 핵심. t0 이웃 쌍 중 지금 N 배 넘게 벌어진 것을
   골라, 그것이 **경계면 쌍**(다른 구름끼리)인지 **내부 쌍**(같은 구름)인지 센다.
   경계면에서 먼저 터지면 안 붙은 것이고, 내부에서 터지면 붙은 것이다.
   "틈이 닫혔다" 까지만 보면 이 질문에 답할 수 없다 -- 전에 그래서 결론을 못 냈다.

  python exe/measure_fragments.py --h5_dir DIR [--group group.npy]
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

ap = argparse.ArgumentParser()
ap.add_argument("--h5_dir", required=True)
ap.add_argument("--group", default=None, help="없으면 <h5_dir>/group.npy")
ap.add_argument("--k", type=int, default=16, help="t0 이웃 수")
ap.add_argument("--r_mult", type=float, default=1.5, help="조각 반경 = 이 값 x 초기 간격")
ap.add_argument("--escape", type=float, default=3.0, help="이 배수 넘게 벌어지면 이탈")
ap.add_argument("--n_sub", type=int, default=60000, help="이만큼만 표본으로 본다")
ap.add_argument("--min_frac", type=float, default=0.002, help="이 비율 넘는 조각만 센다")
ap.add_argument("--out", default=None)
a = ap.parse_args()

fs = sorted(glob.glob(os.path.join(a.h5_dir, "**", "sim_*.h5"), recursive=True))
if not fs:
    raise SystemExit(f"h5 가 없다: {a.h5_dir}")


def rd(p):
    with h5py.File(p, "r") as h:
        x = np.array(h["x"])
    return (x.T if x.shape[0] == 3 else x).astype(np.float64)


x0 = rd(fs[0])
gp = a.group or os.path.join(a.h5_dir, "group.npy")
grp = np.load(gp) if os.path.exists(gp) else np.zeros(len(x0), np.int32)

rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(x0), min(a.n_sub, len(x0)), replace=False))
p0, g0 = x0[idx], grp[idx]

nn = NearestNeighbors(n_neighbors=a.k + 1).fit(p0)
d0, nb = nn.kneighbors(p0)
spacing = float(np.median(d0[:, 1]))
r = a.r_mult * spacing
pairs = np.stack([np.repeat(np.arange(len(p0)), a.k), nb[:, 1:].ravel()], 1)
pd0 = d0[:, 1:].ravel()
cross = g0[pairs[:, 0]] != g0[pairs[:, 1]]
print(f"[입력] {len(fs)} 프레임, 표본 {len(p0)}/{len(x0)} 입자, 초기 간격 "
      f"{spacing:.5f}, 조각 반경 {r:.5f}, 구름 {len(np.unique(grp))} 개 "
      f"(경계면 쌍 {int(cross.sum())}/{len(pairs)})", flush=True)

rows = []
for i in (0, len(fs) // 2, len(fs) - 1):
    xx = rd(fs[i])[idx]
    ok = np.isfinite(xx).all(1)
    pd = np.linalg.norm(xx[pairs[:, 0]] - xx[pairs[:, 1]], axis=1)
    good = ok[pairs[:, 0]] & ok[pairs[:, 1]] & (pd0 > 1e-12)
    esc = (pd > a.escape * pd0) & good
    ei = esc & cross & good
    eb = esc & (~cross) & good
    # 조각 -- 같은 반경으로 이은 그래프의 연결 성분
    m = NearestNeighbors(radius=r).fit(xx[ok])
    g = m.radius_neighbors_graph(xx[ok], mode="connectivity")
    ncomp, lab = connected_components(g, directed=False)
    _, cnt = np.unique(lab, return_counts=True)
    big = cnt[cnt >= a.min_frac * ok.sum()]
    big = np.sort(big)[::-1]
    row = dict(frame=i, escaped=float(esc[good].mean()),
               iface=float(ei.sum() / max(cross[good].sum(), 1)),
               bulk=float(eb.sum() / max((~cross)[good].sum(), 1)),
               n_frag=int(len(big)), largest=float(big[0] / ok.sum()) if len(big) else 0.0,
               frag_sizes=[round(float(c / ok.sum()), 4) for c in big[:6]])
    rows.append(row)
    print(f"  f{i:4d}  이탈 {100*row['escaped']:6.2f}%  "
          f"(경계면 {100*row['iface']:6.2f}% / 내부 {100*row['bulk']:6.2f}%)  "
          f"조각 {row['n_frag']} 개 최대 {100*row['largest']:.1f}%  {row['frag_sizes']}",
          flush=True)

first, last = rows[0], rows[-1]
verdict = []
# 처음보다 **가장 큰 덩어리가 줄었을 때만** 갈라진 것으로 본다
drop = first["largest"] - last["largest"]
if drop > 0.05:
    verdict.append(f"가장 큰 덩어리가 {100*first['largest']:.0f}% -> "
                   f"{100*last['largest']:.0f}% 로 쪼개졌다")
elif last["largest"] > first["largest"] + 0.05:
    verdict.append(f"오히려 뭉쳤다 ({100*first['largest']:.0f}% -> "
                   f"{100*last['largest']:.0f}%)")
else:
    verdict.append(f"덩어리 구성이 그대로다 (최대 {100*last['largest']:.0f}%)")
if cross.sum() > 0:
    if last["iface"] > 2 * max(last["bulk"], 1e-9):
        verdict.append("경계면에서 먼저 터졌다 -- **안 붙었다**")
    elif last["bulk"] > 0 and last["iface"] <= 1.5 * last["bulk"]:
        verdict.append("경계면과 내부가 비슷하게 터졌다 -- **붙었다**")
    else:
        verdict.append("아직 안 터졌다 -- 더 당겨야 한다")
print(f"[판정] " + ", ".join(verdict), flush=True)
if a.out:
    json.dump(dict(spacing=spacing, r=r, rows=rows), open(a.out, "w"), indent=1)
    print(f"[저장] {a.out}")
print("FRAG_OK", flush=True)
