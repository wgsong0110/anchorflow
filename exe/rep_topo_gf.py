"""repflow A GF 장면 위상 지표 (exe/gfs_post.py topo 와 같은 정의, 같은 4000 표본): 이웃 이탈률 (초기 k-NN 16 이웃이 초기 거리의 3 배 넘게 벌어진 비율; 최댓값·마지막) 과 조각 수.
정답 = 흐름 파일 traj, 우리 = 결과 npz 의 traj (--traj32 로 float32 저장한 것).  python exe/rep_topo_gf.py --res R.npz --flow F.npz --out O.json"""
import argparse, json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser(); ap.add_argument("--res", required=True); ap.add_argument("--flow", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
gt = np.load(a.flow)["traj"].astype(np.float64)
pr = np.concatenate([gt[:1], np.load(a.res)["traj"].astype(np.float64)], 0)        # 결과 traj 는 프레임 1..T
T = min(gt.shape[0], pr.shape[0]); fin = [bool(np.isfinite(pr[t]).all()) for t in range(T)]
nfin = 0
while nfin < T and fin[nfin]: nfin += 1
rng = np.random.default_rng(a.seed); N = gt.shape[1]
ok0 = np.isfinite(gt[0]).all(1); idx = np.sort(rng.choice(np.where(ok0)[0], size=min(4000, int(ok0.sum())), replace=False))
x0 = gt[0][idx]; tr = cKDTree(x0); d0, nb = tr.query(x0, k=17); d0, nb = d0[:, 1:], nb[:, 1:]
def stat(X, t):
    x = X[t][idx]; xn = X[t][idx[nb]]
    okp = np.isfinite(x).all(1)
    d = np.linalg.norm(xn - x[:, None], axis=-1)
    esc = (d > 3 * d0) | ~np.isfinite(d)
    r = float(esc[okp].mean()) if okp.any() else float("nan")
    keep = (~esc) & okp[:, None]
    n = x.shape[0]; rows = np.repeat(np.arange(n), 16)
    g = coo_matrix((np.ones(keep.sum()), (rows[keep.ravel()], nb.ravel()[keep.ravel()])), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    sz = np.bincount(lab); big = int((sz >= max(1, 0.01 * n)).sum())
    return r, big, int(nc)
def series(X, TT):
    s = [stat(X, t) for t in range(TT)]; return dict(esc=[q[0] for q in s], pieces=[q[1] for q in s], comps=[q[2] for q in s])
sg = series(gt, T); so = series(pr, nfin) if nfin else dict(esc=[], pieces=[], comps=[])
o = dict(frames_gt=int(gt.shape[0]), frames_pred=int(pr.shape[0]), finite_pred_frames=nfin, sample=int(len(idx)),
         gt=dict(esc_max=max(sg["esc"]), esc_last=sg["esc"][-1], pieces_last=sg["pieces"][-1], pieces_max=max(sg["pieces"]), series=sg),
         ours=dict(esc_max=max(so["esc"]) if nfin else None, esc_last=so["esc"][-1] if nfin else None, pieces_last=so["pieces"][-1] if nfin else None,
                   pieces_max=max(so["pieces"]) if nfin else None, series=so))
json.dump(o, open(a.out, "w")); print("[위상]", {k: v for k, v in o["gt"].items() if k != "series"}, {k: v for k, v in o["ours"].items() if k != "series"}, flush=True)
