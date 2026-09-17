"""변형 모델 한 프레임의 비용을 조각별로 재고, 실시간인지 판정한다.

이 모델은 한 스텝이 한 **프레임**이다 (서브스텝이 없다). 그래서 실시간 여부가
"프레임 하나를 만드는 데 걸리는 시간 < 프레임 간격"으로 바로 판정된다 -- MPM 처럼
프레임당 수백 서브스텝을 곱할 필요가 없다.

조각을 나눠 재는 이유는 호출 빈도와 성격이 다르기 때문이다.

  kNN      가우시안마다 가장 가까운 앵커 k 개. 격자 가속이라 전체 짝을 훑지 않는다.
  집계     앵커별로 자기 가우시안들을 질량 가중 요약. N x k 에 비례한다.
  순전파   어텐션. 앵커 수만 보고 가우시안 수와 무관하다.
  스키닝   변형 사상 적용. N x k.
  야코비안 모양 갱신용. 역전파 세 번이라 학습에만 필요하고 추론에는 선택이다.
  FPS      --refps 일 때만. 반복이 M 번이라 커널 실행이 M 번 나므로 따로 봐야 한다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--ckpt", default=None, help="있으면 그 구조/앵커를 쓴다")
ap.add_argument("--n_pts", type=int, nargs="+", default=[20000, 40000])
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--warmup", type=int, default=10)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
from anchorflow.deform import (DeformNet, aggregate, bc_features,   # noqa: E402
                               fps, grid_knn, jacobian_of, skin)

f = sorted(glob.glob(os.path.join(a.data, "*.pt")))[0]
d = torch.load(f, map_location="cpu", weights_only=False)
cfg = d["cfg"]
FRAME_DT = float(cfg["frame_dt"])
X0 = d["x"][0]
EXT = float((X0.max(0).values - X0.min(0).values).norm())
N_FULL = X0.shape[0]
print(f"[씬] {os.path.basename(f)}, 입자 {N_FULL}, 물체 {EXT:.4f}, "
      f"프레임 간격 {FRAME_DT*1000:.1f} ms ({1/FRAME_DT:.0f} fps)", flush=True)

ng = int(cfg.get("n_grid", 100))
dx = float(cfg.get("grid_lim", 2.0)) / ng
X0d = X0.to(dev)
vi = (X0d / dx).long().clamp(0, ng - 1)
flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cnt = torch.zeros(ng ** 3, device=dev).index_add_(
    0, flat, torch.ones(N_FULL, device=dev))
MASS_ALL = ((dx ** 3) / cnt[flat]) * float(cfg["density"])
MAT = torch.cat([torch.tensor(
    [np.log(float(cfg["E"])), float(cfg["nu"]), float(cfg.get("xi", 0.0)),
     np.log(float(cfg["density"]))], device=dev, dtype=torch.float32),
    torch.tensor(cfg["g"], device=dev, dtype=torch.float32) / 15.0])


def timeit(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(ts))


rows = {}
for N in a.n_pts:
    gs = torch.arange(min(N, N_FULL), device=dev)
    x = X0d[gs].contiguous()
    XC = x.clone()
    v = torch.zeros_like(x)
    M = a.n_anchors
    AIDX = fps(x, M)
    H = float(torch.cdist(x[AIDX], x[AIDX]).topk(2, largest=False).values[:, 1]
              .median())
    p = x[AIDX].contiguous()
    mass = MASS_ALL[gs]

    idx, _ = grid_knn(x, p, a.k)
    feat, _ = aggregate(x, v, XC, mass, idx, M, H, pa=p)
    n_bc = bc_features(p[:2], cfg).shape[-1]
    n_feat = feat.shape[-1] + MAT.numel() + n_bc
    net = DeformNet(n_feat=n_feat, hidden=a.hidden, depth=a.depth,
                    heads=a.heads, scale=0.02 * EXT, h=H, ext=EXT).to(dev).eval()
    extra = torch.cat([MAT.reshape(1, -1).expand(M, -1),
                       bc_features(p, cfg) / H], -1)
    dp, lr_, lt_ = net(p, torch.cat([feat, extra], -1), FRAME_DT)

    r = {}
    r["kNN"] = timeit(lambda: grid_knn(x, p, a.k), a.warmup, a.reps)
    r["집계"] = timeit(lambda: aggregate(x, v, XC, mass, idx, M, H, pa=p),
                     a.warmup, a.reps)
    r["순전파"] = timeit(lambda: net(p, torch.cat([feat, extra], -1), FRAME_DT),
                      a.warmup, a.reps)
    r["스키닝"] = timeit(lambda: skin(x, p, dp, lr_, lt_, idx, H), a.warmup, a.reps)
    with torch.enable_grad():
        r["야코비안"] = timeit(
            lambda: jacobian_of(lambda q: skin(q, p, dp, lr_, lt_, idx, H)[0], x),
            3, max(5, a.reps // 3))
    r["FPS"] = timeit(lambda: fps(x, M), 3, max(5, a.reps // 5))
    r["프레임(추론)"] = r["kNN"] + r["집계"] + r["순전파"] + r["스키닝"]
    r["프레임(+모양)"] = r["프레임(추론)"] + r["야코비안"]
    r["프레임(+refps)"] = r["프레임(추론)"] + r["FPS"]
    rows[N] = r
    print(f"\n[입자 {N}, 앵커 {M}]", flush=True)
    for k_ in ("kNN", "집계", "순전파", "스키닝", "야코비안", "FPS"):
        print(f"  {k_:<10} {r[k_]:7.3f} ms", flush=True)
    for k_ in ("프레임(추론)", "프레임(+모양)", "프레임(+refps)"):
        print(f"  {k_:<14} {r[k_]:7.3f} ms  = {1000/r[k_]:6.1f} fps"
              f"   {'실시간' if r[k_] < FRAME_DT*1000 else '실시간 아님'}"
              f" (기준 {FRAME_DT*1000:.1f} ms)", flush=True)

if a.out:
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(frame_dt=FRAME_DT, n_anchors=a.n_anchors, k=a.k,
                   gpu=torch.cuda.get_device_name(0),
                   rows={str(k): v for k, v in rows.items()}),
              open(os.path.join(a.out, "deform_speed.json"), "w"), indent=1,
              ensure_ascii=False)
print("DEFBENCH_OK")
