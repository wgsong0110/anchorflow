"""ficus 빠른 진동 분석 -- /home/dkta/work/ficosc.py (10-23) 와 같은 식을 인자로 받게 옮긴 것.

궤적마다: 평균 변위 벡터 m_t = mean_p (x_t − x_0) / 대각 (0 프레임 bbox 대각), 그 주성분 축 (궤적마다 자기 SVD 첫 축,
부호는 |값| 최대인 프레임이 + 가 되게) 위 성분 sig, 평균 변위 크기 mag, 프레임당 이동 speed = mean_p |x_t − x_{t−1}| / 대각,
ψ 평균 (FCR, E = 2e6, ν = 0.4: μ‖F − R‖² + λ/2 (J − 1)², 입자 단순 평균; 덤프는 F 누적, 0 프레임은 항등).
  python exe/ficus_osc_quick.py --gt_dir lbours/ficus_v0/gt --run "lin=선형=roll_ov.pt" ... --out osc_quick.json
"""
import argparse
import glob
import json

import h5py
import numpy as np
import torch
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--gt_dir", required=True, help="sim_*.h5 (x, F)")
ap.add_argument("--run", action="append", required=True, help="키=라벨=덤프")
ap.add_argument("--out", required=True)
ap.add_argument("--E", type=float, default=2e6)
ap.add_argument("--nu", type=float, default=0.4)
a = ap.parse_args()
E, nu = a.E, a.nu
mu = E / (2 * (1 + nu)); la = E * nu / ((1 + nu) * (1 - 2 * nu))


def psi(F):
    F = torch.as_tensor(F, dtype=torch.float64).reshape(-1, 3, 3)
    U, S, Vh = torch.linalg.svd(F); R = U @ Vh; J = torch.det(F)
    return (mu * ((F - R) ** 2).sum((1, 2)) + 0.5 * la * (J - 1) ** 2).numpy()


def analyze(X, Fs, name):
    X = np.asarray(X, np.float64); diag = np.linalg.norm(X[0].max(0) - X[0].min(0))
    D = X - X[0]; m = D.mean(1) / diag
    u, s, vt = np.linalg.svd(m - m.mean(0), full_matrices=False); ax = vt[0]
    sig = m @ ax; sig *= np.sign(sig[np.argmax(abs(sig))])
    mag = np.linalg.norm(D, axis=2).mean(1) / diag
    sp = np.r_[0, np.linalg.norm(np.diff(X, axis=0), axis=2).mean(1) / diag]
    ps = [float(psi(f).mean()) if f is not None else np.nan for f in tqdm(Fs, desc=f"{name} ψ")]
    return dict(name=name, axis=ax.tolist(), sig=sig.tolist(), mag=mag.tolist(), speed=sp.tolist(), psi=ps)


out = {}
fs = sorted(glob.glob(f"{a.gt_dir}/sim_*.h5"))
X, F = [], []
for f in tqdm(fs, desc="정답 h5"):
    with h5py.File(f) as h:
        X.append(np.array(h["x"])); F.append(np.array(h["F"]))
X = np.stack(X); print("gt", X.shape, flush=True)
out["gt"] = analyze(X, F, "정답 (PG)")
del X, F
for q in a.run:
    k, lab, p = q.split("=", 2)
    d = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    P = d["pred"].float().numpy(); x0 = d["x0"].float().numpy()
    Xk = np.concatenate([x0[None], P])
    Fk = ([np.eye(3)[None].repeat(len(x0), 0)] + [f.float().numpy() for f in d["F"]]) if d.get("F") is not None else [None] * len(Xk)
    print(k, Xk.shape, flush=True)
    out[k] = analyze(Xk, Fk, lab)
json.dump(out, open(a.out, "w"), ensure_ascii=False)
print("OSC_DONE", flush=True)
