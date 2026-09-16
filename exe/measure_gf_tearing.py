"""GaussianFluent 궤적이 실제로 **토폴로지가 바뀌는** 궤적인지 잰다.

bread 씬을 버린 이유가 여기 있다. 그쪽은 재질이 jelly 라 소성도 파괴도 없었고,
"찢어져 보이는" 것은 서로 반대로 당기는 구동기 아래의 탄성 신장일 뿐이었다.
그러니 새 궤적을 쓰기 전에 같은 실수를 반복하지 않도록 먼저 재고 넘어간다.

재는 양은 둘이다.

  이웃 이탈률  초기 k-NN 이웃 중 시각 t 에 초기 거리의 thresh 배 밖으로 나간 비율.
               고정 연결성(사면체 케이지·스프링)이 표현할 수 없는 바로 그 양이다.
  되돌아옴     마지막 프레임에서의 이탈률이 최댓값보다 크게 낮으면 늘어났다 돌아온
               것, 즉 탄성이다. 소성/파괴라면 이탈이 남아 있어야 한다.

입력은 GaussianFluent 러너가 `--output_h5` 로 떨군 프레임별 h5 다.
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
ap.add_argument("--h5_dir", required=True, help="sim_*.h5 가 든 디렉토리")
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="run")
ap.add_argument("--n_sample", type=int, default=4000)
ap.add_argument("--knn", type=int, default=16)
ap.add_argument("--nbr_thresh", type=float, default=3.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

files = sorted(glob.glob(os.path.join(a.h5_dir, "*.h5")))
if not files:
    raise SystemExit(f"h5 가 없다: {a.h5_dir}")
print(f"[입력] {len(files)} 프레임, {a.h5_dir}", flush=True)


def load_x(p):
    """[N,3] 로 보이는 데이터셋을 찾아 읽는다. 키 이름은 버전마다 다르다."""
    with h5py.File(p, "r") as f:
        keys = list(f.keys())
        for k in ("x", "position", "pos", "particle_x"):
            if k in f:
                d = np.array(f[k])
                return d.reshape(3, -1).T if d.shape[0] == 3 else d.reshape(-1, 3)
        for k in keys:
            d = np.array(f[k])
            if d.ndim == 2 and 3 in d.shape:
                return d.T if d.shape[0] == 3 else d
    raise KeyError(f"{p} 에서 위치를 못 찾았다. 키: {keys}")


X0 = torch.from_numpy(load_x(files[0])).float().to(dev)
N = X0.shape[0]
EXT = float((X0.max(0).values - X0.min(0).values).norm())
g = torch.Generator(device="cpu").manual_seed(a.seed)
sub = torch.randperm(N, generator=g)[:a.n_sample].to(dev)
S0 = X0[sub]
d0 = torch.cdist(S0, S0)
d0.fill_diagonal_(float("inf"))
nbr_d0, nbr_i0 = d0.topk(a.knn, largest=False)
print(f"[씬] 입자 {N}, 물체 대각 {EXT:.4f}, 표본 {sub.numel()}, k={a.knn}, "
      f"초기 이웃 거리 중앙 {float(nbr_d0.median()):.5f}", flush=True)

rows = []
for i, p in enumerate(files):
    x = torch.from_numpy(load_x(p)).float().to(dev)
    s = x[sub]
    dn = (s.unsqueeze(1) - s[nbr_i0]).norm(dim=-1)          # [S, k]
    torn = float((dn > a.nbr_thresh * nbr_d0).float().mean())
    disp = float((x - X0).norm(dim=-1).max()) / EXT
    rows.append(dict(frame=i, torn=torn, max_disp=disp,
                     stretch=float((dn / nbr_d0).median())))
    if i % 10 == 0 or i == len(files) - 1:
        print(f"  f{i:3d}  이웃 이탈 {100*torn:6.2f}%  최대 변위 {100*disp:6.2f}%  "
              f"이웃거리 배율(중앙) {rows[-1]['stretch']:.3f}", flush=True)

torn = [r["torn"] for r in rows]
t_max, t_last = max(torn), torn[-1]
# 되돌아옴 비율: 1 이면 최댓값이 그대로 남았고, 0 이면 완전히 돌아왔다 (= 탄성)
keep = t_last / t_max if t_max > 1e-9 else 0.0
verdict = ("토폴로지 변화 있음 (영구)" if t_max > 0.01 and keep > 0.5 else
           "늘어났다 돌아옴 (탄성에 가깝다)" if t_max > 0.01 else
           "이웃 구조가 거의 그대로 -- 토폴로지 변화 없음")
print(f"\n[판정] 이웃 이탈 최대 {100*t_max:.2f}% 마지막 {100*t_last:.2f}% "
      f"(잔존 {100*keep:.0f}%) -> {verdict}", flush=True)

os.makedirs(a.out, exist_ok=True)
json.dump(dict(tag=a.tag, h5_dir=a.h5_dir, n_particles=N, extent=EXT,
               knn=a.knn, nbr_thresh=a.nbr_thresh, torn_max=t_max,
               torn_last=t_last, retained=keep, verdict=verdict, rows=rows),
          open(os.path.join(a.out, f"tearing_{a.tag}.json"), "w"),
          indent=1, ensure_ascii=False)
print(f"[저장] {os.path.join(a.out, f'tearing_{a.tag}.json')}", flush=True)
print("TEARING_OK")
