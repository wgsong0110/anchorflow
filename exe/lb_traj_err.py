"""두 lb 형식 궤적(sim_*.h5 의 x) 사이 위치 오차 -- 출력만 최적화 롤아웃과 같은 정의.

오차_t = mean_i ‖x_t,i − x̂_t,i‖ / EXT,  EXT = 정답 0 프레임 전체 입자 바운딩박스 대각선 (deform_model.extent/frame_err).
프레임 1..L 평균을 낸다 (train_deform 의 [롤아웃] 평균과 같다). 같은 번호 입자끼리.

  python exe/lb_traj_err.py --gt GTDIR --sim DIR --frames 80 --out err.json
"""
import argparse
import json

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True)
ap.add_argument("--sim", required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def X(d, f):
    with h5py.File(f"{d}/sim_{f:010d}.h5") as h:
        x = np.array(h["x"], dtype=np.float64)
    return x.T if x.shape[0] == 3 else x


g0 = X(a.gt, 0)
EXT = float(np.linalg.norm(g0.max(0) - g0.min(0)))
s0 = X(a.sim, 0)
assert s0.shape == g0.shape, f"입자 수 다름 {s0.shape} vs {g0.shape}"
e0 = float(np.linalg.norm(s0 - g0, axis=1).max())
print(f"[대응] 입자 {g0.shape[0]}, 0 프레임 위치 차이 최대 {e0:.2e}, EXT {EXT:.5f}", flush=True)
E = []
for f in range(1, a.frames + 1):
    E.append(float(np.linalg.norm(X(a.sim, f) - X(a.gt, f), axis=1).mean()) / EXT)
m = float(np.mean(E))
print(f"[오차] 프레임 1~{a.frames} 평균 {100 * m:.3f}%  마지막 {100 * E[-1]:.3f}%  최대 {100 * max(E):.3f}%", flush=True)
json.dump(dict(mean=m, per_frame=E, EXT=EXT, frame0_maxdiff=e0, gt=a.gt, sim=a.sim), open(a.out, "w"), indent=1)
