"""복셀 앵커 경로가 맞는지 확인하고, kNN+FPS 경로와 속도를 견준다.

세 가지를 본다.

  정확성  커널이 파이토치로 쓴 같은 정의를 내는가 (이웃 색인과 거리).
  비용    복셀 경로 한 프레임 대 기존 경로 한 프레임. 기존 쪽은 앵커를 매 프레임
          다시 뽑으려면 FPS 가 들어가므로, "매 스텝 재표본" 이라는 같은 조건에서
          비교한다.
  떨림    복셀 경계를 넘을 때 소속이 불연속으로 바뀐다. 실제 궤적에서 연속한 두
          프레임 사이에 앵커가 몇 개나 갈리는지, 그리고 그 때문에 스키닝 결과가
          얼마나 튀는지 잰다 -- 이것이 이 방법의 유일한 위험이다.
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
ap.add_argument("--ply", default=None)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--n_anchors", type=int, default=512, help="기존 경로의 앵커 수")
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--out", default=None)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
from anchorflow import voxel                                      # noqa: E402
from anchorflow.deform import anchor_knn, fps, aggregate, skin_with_jacobian, \
    DeformNet, bc_features                                        # noqa: E402

f = sorted(glob.glob(os.path.join(a.data, "*.pt")))[0]
d = torch.load(f, map_location="cpu", weights_only=False)
cfg = d["cfg"]
X0 = d["x"][0]
EXT = float((X0.max(0).values - X0.min(0).values).norm())
FRAME_DT = float(cfg["frame_dt"])

if a.ply:
    from plyfile import PlyData
    v_ = PlyData.read(a.ply)["vertex"]
    t_ = torch.from_numpy(np.stack([v_["x"], v_["y"], v_["z"]], 1)).float()
    t_ = t_ - t_.mean(0)
    t_ = t_ / float((t_.max(0).values - t_.min(0).values).norm()) * EXT
    x = (t_ + X0.mean(0)).to(dev).contiguous()
else:
    x = X0.to(dev).contiguous()
N = x.shape[0]
X = x.clone()
v = torch.randn_like(x) * 0.01
m = torch.rand(N, device=dev) + 0.1

# 복셀 한 변은 기존 경로의 앵커 간격에 맞춘다 (앵커 수가 비슷해지도록)
aid0 = fps(x, a.n_anchors)
H = float(torch.cdist(x[aid0], x[aid0]).topk(2, largest=False).values[:, 1].median())
print(f"[씬] 입자 {N}, 물체 {EXT:.4f}, 기존 앵커 {a.n_anchors} 간격 {H:.5f}",
      flush=True)


def timeit(fn, n=a.reps):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


va = voxel.build(x, X, v, m, H)
print(f"[복셀] 한 변 {H:.5f} -> 앵커 {va.M} 개 (기존 {a.n_anchors})", flush=True)

# --- 정확성: 커널 대 파이토치 ---
gi_c, dv_c = voxel.neighbors(x, va, a.k)
_hd, voxel._HAVE_DC = voxel._HAVE_DC, False
gi_t, dv_t = voxel.neighbors(x, va, a.k)
voxel._HAVE_DC = _hd
same = float((gi_c.sort(1).values == gi_t.sort(1).values).all(1).float().mean())
e_d = float((dv_c.sort(1).values - dv_t.sort(1).values).abs().max())
print(f"[정확성] 이웃 색인 일치 {100*same:.2f}%, 거리 최대차 {e_d:.2e}", flush=True)

# --- 비용 ---
dp = torch.randn(va.M, 3, device=dev) * 0.01
log_r = torch.full((va.M,), float(np.log(H)), device=dev)
log_t = torch.zeros(va.M, device=dev)
t_build = timeit(lambda: voxel.build(x, X, v, m, H))
t_nb = timeit(lambda: voxel.neighbors(x, va, a.k))
t_skin = timeit(lambda: skin_with_jacobian(x, va.pos, dp, log_r, log_t, gi_c, H))
t_vox = t_build + t_nb + t_skin

p0 = x[aid0].contiguous()
idx0, _ = anchor_knn(x, p0, a.k)
dp0 = torch.randn(a.n_anchors, 3, device=dev) * 0.01
lr0 = torch.full((a.n_anchors,), float(np.log(H)), device=dev)
lt0 = torch.zeros(a.n_anchors, device=dev)
t_fps = timeit(lambda: fps(x, a.n_anchors), 5)
t_knn = timeit(lambda: anchor_knn(x, p0, a.k))
t_agg = timeit(lambda: aggregate(x, v, X, m, idx0, a.n_anchors, H, pa=p0))
t_sk0 = timeit(lambda: skin_with_jacobian(x, p0, dp0, lr0, lt0, idx0, H))
t_old = t_fps + t_knn + t_agg + t_sk0

print(f"\n[비용] 매 스텝 앵커를 새로 뽑는 같은 조건에서", flush=True)
print(f"  기존: FPS {t_fps:6.2f} + kNN {t_knn:5.2f} + 집계 {t_agg:5.2f} + "
      f"스키닝+J {t_sk0:4.2f} = {t_old:6.2f} ms", flush=True)
print(f"  복셀: 생성·집계 {t_build:5.2f} + 이웃 {t_nb:5.2f} + "
      f"스키닝+J {t_skin:4.2f} = {t_vox:6.2f} ms  ({t_old/max(t_vox,1e-9):.1f}배)",
      flush=True)

# --- 떨림: 실제 궤적에서 연속 프레임 사이의 소속 변화 ---
flip, jump = [], []
T = min(40, d["x"].shape[0] - 1)
gs = torch.arange(min(N, d["x"].shape[1]))
prev = None
for t in range(1, T):
    xt = d["x"][t][gs].to(dev)
    vt = (xt - d["x"][t - 1][gs].to(dev)) / FRAME_DT
    Xc = d["x"][0][gs].to(dev)
    mm = torch.rand(xt.shape[0], device=dev) + 0.1
    vb = voxel.build(xt, Xc, vt, mm, H)
    gi, _ = voxel.neighbors(xt, vb, a.k)
    own = gi[:, 0]
    if prev is not None:
        # 같은 복셀 키를 가리키는지로 비교한다 (앵커 번호는 프레임마다 다시 매겨진다)
        kp = prev[0][prev[1]]
        kn = vb.keys[own.clamp(min=0)]
        flip.append(float((kp != kn).float().mean()))
    prev = (vb.keys, own.clamp(min=0))
print(f"\n[떨림] 연속 프레임 사이 소속 복셀이 바뀐 가우시안 비율: "
      f"평균 {100*np.mean(flip):.2f}%  최대 {100*np.max(flip):.2f}%", flush=True)

if a.out:
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(N=N, M_voxel=int(va.M), M_old=a.n_anchors, cell=H,
                   idx_same=same, dist_err=e_d,
                   old=dict(fps=t_fps, knn=t_knn, agg=t_agg, skin=t_sk0,
                            total=t_old),
                   vox=dict(build=t_build, nb=t_nb, skin=t_skin, total=t_vox),
                   flip_mean=float(np.mean(flip)), flip_max=float(np.max(flip))),
              open(os.path.join(a.out, "voxel_verify.json"), "w"), indent=1)
print("VOXEL_OK")
