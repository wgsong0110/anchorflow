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
ap.add_argument("--ens", type=int, default=4,
                help="오프셋 앙상블에 쓸 격자 수 (0 이면 안 함)")
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


# 격자는 공간에 고정한다. 궤적 전체를 담도록 여유를 두고 한 번만 잡는다.
LO = (d["x"].reshape(-1, 3).min(0).values - 2 * H).to(dev)
va = voxel.build(x, X, v, m, H, lo=LO)
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
t_build = timeit(lambda: voxel.build(x, X, v, m, H, lo=LO))
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

# --- 떨림: 실제 궤적에서 연속 프레임 사이의 앵커 위치 변화 ---
# "소속이 바뀌었나" 가 아니라 **그 가우시안이 보는 앵커 위치가 얼마나 튀는가** 를
# 재야 한다. 소속은 바뀌어도 새 앵커가 바로 옆이면 변형장은 매끄럽다.
T = min(40, d["x"].shape[0] - 1)
gs = torch.arange(min(N, d["x"].shape[1]))
# 오프셋은 칸 안에 고르게 흩어져야 한다 (앙상블이 상쇄로 이득을 보는 구조라서).
OFFS = np.array([np.random.RandomState(i).rand(3) for i in range(max(a.ens, 1))])


H_ENS = H * (max(a.ens, 1) ** (1.0 / 3.0))


def anchor_of(xt, Xc, vt, mm, mode):
    """각 가우시안이 보는 앵커 위치 [N,3]. mode: hard / soft / ens / ens_eq"""
    if mode in ("ens", "ens_eq"):
        cell = H if mode == "ens" else H_ENS
        # 격자 L 개를 **한 번에** 만든다. 각 격자에서 가장 가까운 앵커를 평균낸다.
        vb = voxel.build(xt, Xc, vt, mm, cell, lo=LO, offsets=OFFS)
        gi, _ = voxel.neighbors(xt, vb, max(a.ens, 4))
        pp = vb.pos[gi.clamp(min=0)]
        ok = (gi >= 0).float().unsqueeze(-1)
        return (pp * ok).sum(1) / ok.sum(1).clamp(min=1)
    vb = voxel.build(xt, Xc, vt, mm, H, lo=LO, soft=(mode == "soft"))
    gi, _ = voxel.neighbors(xt, vb, a.k)
    return vb.pos[gi[:, 0].clamp(min=0)]


res_flick = {}
for mode in (("hard", "soft", "ens", "ens_eq") if a.ens
             else ("hard", "soft")):
    prev, jump = None, []
    for t in range(1, T):
        xt = d["x"][t][gs].to(dev)
        vt = (xt - d["x"][t - 1][gs].to(dev)) / FRAME_DT
        Xc = d["x"][0][gs].to(dev)
        mm = torch.rand(xt.shape[0], device=dev) + 0.1
        cur = anchor_of(xt, Xc, vt, mm, mode)
        if prev is not None:
            # 가우시안 자신이 움직인 만큼을 빼고, 앵커가 **추가로** 튄 양을 본다
            mv = (xt - d["x"][t - 1][gs].to(dev))
            jump.append(float((cur - prev - mv).norm(dim=-1).mean()) / EXT)
        prev = cur
    res_flick[mode] = (float(np.mean(jump)), float(np.max(jump)))
    cc = H_ENS if mode == "ens_eq" else H
    print(f"[떨림-{mode:>6}] 앵커 위치의 추가 변동: 평균 {100*np.mean(jump):.3f}% "
          f"최대 {100*np.max(jump):.3f}%  (복셀 한 변 {100*cc/EXT:.2f}%)", flush=True)

t_soft = timeit(lambda: voxel.build(x, X, v, m, H, lo=LO, soft=True), 5)
t_ens = timeit(lambda: voxel.build(x, X, v, m, H, lo=LO, offsets=OFFS), 5)
vb_e = voxel.build(x, X, v, m, H, lo=LO, offsets=OFFS)
t_ens_nb = timeit(lambda: voxel.neighbors(x, vb_e, max(a.ens, 4)), 5)
print(f"\n[비용] 부드러운 배정(B-스플라인 3x3x3) 생성·집계 {t_soft:6.2f} ms "
      f"(하드 {t_build:5.2f} ms, {t_soft/max(t_build,1e-9):.1f}배)", flush=True)
if a.ens:
    print(f"       오프셋 앙상블 {a.ens} 개 (한 커널) 생성·집계 {t_ens:6.2f} ms + "
          f"이웃 {t_ens_nb:5.2f} ms = {t_ens + t_ens_nb:6.2f} ms", flush=True)
    print(f"       (파이썬으로 {a.ens} 번 돌면 "
          f"{a.ens * (t_build + t_nb):6.2f} ms)", flush=True)
    print(f"       앵커 {vb_e.M} 개 (격자 하나 {va.M})", flush=True)

if a.out:
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(N=N, M_voxel=int(va.M), M_old=a.n_anchors, cell=H,
                   idx_same=same, dist_err=e_d,
                   old=dict(fps=t_fps, knn=t_knn, agg=t_agg, skin=t_sk0,
                            total=t_old),
                   vox=dict(build=t_build, nb=t_nb, skin=t_skin, total=t_vox),
                   flicker={k: v for k, v in res_flick.items()},
                   soft_build=t_soft),
              open(os.path.join(a.out, "voxel_verify.json"), "w"), indent=1)
print("VOXEL_OK")
