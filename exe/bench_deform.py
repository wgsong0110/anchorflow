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
ap.add_argument("--ply", default=None,
                help="전체 가우시안 수로 재려면 원본 ply. 학습용 .pt 는 4 만 개로 "
                     "부분표본된 것이라 실제 배포 규모가 아니다")
ap.add_argument("--render", action="store_true",
                help="래스터화까지 포함해 잰다. 실시간 판정은 물리만이 아니라 "
                     "'프레임 하나를 화면에 내는 데 걸리는 시간'이어야 한다")
ap.add_argument("--width", type=int, default=800)
ap.add_argument("--height", type=int, default=800)
ap.add_argument("--sh_degree", type=int, default=3)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--k", type=int, nargs="+", default=[16])
ap.add_argument("--chunk", type=int, nargs="+", default=[8192, 32768, 131072],
                help="kNN 청크. [chunk, M] 점수판이 L2 에 들어가는지가 topk 속도를 "
                     "가른다 -- 값은 바뀌지 않는다")
ap.add_argument("--agg_sub", type=int, default=0,
                help="집계에 쓸 입자 수. 0 이면 전부")
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--warmup", type=int, default=10)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
from anchorflow.deform import (DeformNet, aggregate, anchor_knn,    # noqa: E402
                               bc_features, dense_knn, fps, grid_knn,
                               jacobian_of, skin, skin_with_jacobian)

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


FULL = None
if a.ply:
    from plyfile import PlyData                                   # noqa: E402
    v_ = PlyData.read(a.ply)["vertex"]
    xyz = np.stack([v_["x"], v_["y"], v_["z"]], 1)
    # 위치의 절대 좌표계는 타이밍에 무관하다. 분포(격자 점유)만 맞으면 되므로
    # 학습이 쓰는 MPM 프레임과 같은 크기로 맞춰 둔다.
    t_ = torch.from_numpy(xyz).float()
    t_ = t_ - t_.mean(0)
    t_ = t_ / float((t_.max(0).values - t_.min(0).values).norm()) * EXT
    FULL = (t_ + X0.mean(0)).to(dev)
    print(f"[전체] ply 가우시안 {FULL.shape[0]}", flush=True)


def raster_timer(N):
    """가우시안 N 개를 한 번 래스터화하는 시간. 값은 타이밍에 무관하므로 형태만 맞춘다."""
    from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                             GaussianRasterizer)
    W, Hh = a.width, a.height
    tanfov = float(np.tan(0.5 * np.radians(60.0)))
    vt = torch.eye(4, device=dev)
    vt[3, 2] = 4.0
    pm = torch.eye(4, device=dev)
    means = FULL[:N] if FULL is not None else X0d[:N]
    means = means - means.mean(0)
    cov = torch.zeros(N, 6, device=dev)
    cov[:, 0] = cov[:, 3] = cov[:, 5] = (0.004 * EXT) ** 2
    opa = torch.full((N, 1), 0.8, device=dev)
    col = torch.rand(N, 3, device=dev)
    scr = torch.zeros_like(means)
    st = GaussianRasterizationSettings(
        image_height=Hh, image_width=W, tanfovx=tanfov, tanfovy=tanfov,
        bg=torch.ones(3, device=dev), scale_modifier=1.0,
        viewmatrix=vt, projmatrix=vt @ pm, sh_degree=a.sh_degree,
        campos=torch.zeros(3, device=dev), prefiltered=False, debug=False)
    rast = GaussianRasterizer(raster_settings=st)

    def go():
        rast(means3D=means, means2D=scr, shs=None, colors_precomp=col,
             opacities=opa, scales=None, rotations=None, cov3D_precomp=cov)
    return go


rows = {}
for KK in a.k:
 for N in a.n_pts:
     if FULL is not None and N > N_FULL:
         x = FULL[:N].contiguous()
         gs = torch.arange(min(N, N_FULL), device=dev)
         mass_src = MASS_ALL[gs].median().expand(N).contiguous()
     else:
         gs = torch.arange(min(N, N_FULL), device=dev)
         x = X0d[gs].contiguous()
         mass_src = MASS_ALL[gs]
     XC = x.clone()
     v = torch.zeros_like(x)
     M = a.n_anchors
     AIDX = fps(x, M)
     H = float(torch.cdist(x[AIDX], x[AIDX]).topk(2, largest=False).values[:, 1]
               .median())
     p = x[AIDX].contiguous()
     mass = mass_src

     idx, _ = anchor_knn(x, p, KK)
     SUB = (torch.randperm(x.shape[0], device=dev)[:a.agg_sub]
            if 0 < a.agg_sub < x.shape[0] else None)
     feat, _ = aggregate(x, v, XC, mass, idx, M, H, pa=p, sub=SUB)
     n_bc = bc_features(p[:2], cfg).shape[-1]
     n_feat = feat.shape[-1] + MAT.numel() + n_bc
     net = DeformNet(n_feat=n_feat, hidden=a.hidden, depth=a.depth,
                     heads=a.heads, scale=0.02 * EXT, h=H, ext=EXT).to(dev).eval()
     extra = torch.cat([MAT.reshape(1, -1).expand(M, -1),
                        bc_features(p, cfg) / H], -1)
     dp, lr_, lt_ = net(p, torch.cat([feat, extra], -1), FRAME_DT)

     # 해석적 야코비안이 자동미분과 맞는지 먼저 확인한다 (틀린 것을 빨리 재봐야
     # 소용없다). 작은 부분집합에서 본다.
     with torch.enable_grad():
         xs = x[:2048]
         i2, _ = anchor_knn(xs, p, KK)
         Ja = skin_with_jacobian(xs, p, dp, lr_, lt_, i2, H)[2]
         Jb = jacobian_of(lambda q: skin(q, p, dp, lr_, lt_, i2, H)[0], xs)
         err = float((Ja - Jb).abs().max() / Jb.abs().max().clamp(min=1e-12))
     print(f"\n[야코비안 검증] 해석 대 자동미분 상대 최대오차 {err:.3e}"
           f"{'  <- 불일치' if err > 1e-3 else ''}", flush=True)

     r = {"J_check": err}
     r["kNN(격자)"] = timeit(lambda: grid_knn(x, p, KK), a.warmup, a.reps)
     r["kNN(조밀)"] = timeit(lambda: dense_knn(x, p, KK), a.warmup, a.reps)
     r["kNN(조밀,fp16)"] = timeit(lambda: dense_knn(x, p, KK, half=True),
                                 a.warmup, a.reps)
     try:
         import deformcuda as _dc
         if _dc.HAVE_CUDA and KK in (4, 6, 8, 12, 16, 24, 32):
             r["kNN(커널)"] = timeit(lambda: _dc.knn(x, p, KK), a.warmup, a.reps)
     except Exception as e:
         print(f"  kNN 커널 없음: {type(e).__name__}: {e}", flush=True)
     r["kNN"] = min([r["kNN(격자)"], r["kNN(조밀)"], r["kNN(조밀,fp16)"]]
                    + ([r["kNN(커널)"]] if "kNN(커널)" in r else []))
     r["집계"] = timeit(lambda: aggregate(x, v, XC, mass, idx, M, H, pa=p,
                                        sub=SUB), a.warmup, a.reps)
     r["순전파"] = timeit(lambda: net(p, torch.cat([feat, extra], -1), FRAME_DT),
                       a.warmup, a.reps)
     r["스키닝"] = timeit(lambda: skin(x, p, dp, lr_, lt_, idx, H), a.warmup, a.reps)
     with torch.enable_grad():
         r["야코비안(자동)"] = timeit(
             lambda: jacobian_of(lambda q: skin(q, p, dp, lr_, lt_, idx, H)[0], x),
             3, max(5, a.reps // 3))
     r["스키닝+야코비안(해석)"] = timeit(
         lambda: skin_with_jacobian(x, p, dp, lr_, lt_, idx, H), a.warmup, a.reps)
     r["야코비안"] = max(r["스키닝+야코비안(해석)"] - r["스키닝"], 0.0)
     r["FPS"] = timeit(lambda: fps(x, M), 3, max(5, a.reps // 5))
     if a.render:
         try:
             r["래스터화"] = timeit(raster_timer(x.shape[0]), 3, max(5, a.reps // 3))
         except Exception as e:
             print(f"  래스터화 실패: {type(e).__name__}: {e}", flush=True)
     r["프레임(추론)"] = r["kNN"] + r["집계"] + r["순전파"] + r["스키닝"]
     r["프레임(+모양)"] = (r["kNN"] + r["집계"] + r["순전파"]
                        + r["스키닝+야코비안(해석)"])
     r["프레임(+refps)"] = r["프레임(추론)"] + r["FPS"]
     if "래스터화" in r:
         r["프레임(+모양+렌더)"] = r["프레임(+모양)"] + r["래스터화"]
     rows[(KK, N)] = r
     print(f"\n[입자 {N}, 앵커 {M}, k={KK}]", flush=True)
     for k_ in ("kNN(격자)", "kNN(조밀)", "kNN(조밀,fp16)", "kNN(커널)", "집계",
                "순전파", "스키닝",
                "야코비안(자동)", "스키닝+야코비안(해석)", "FPS", "래스터화"):
         if k_ in r:
             print(f"  {k_:<10} {r[k_]:7.3f} ms", flush=True)
     for k_ in ("프레임(추론)", "프레임(+모양)", "프레임(+refps)",
                "프레임(+모양+렌더)"):
         if k_ not in r:
             continue
         print(f"  {k_:<16} {r[k_]:7.3f} ms  = {1000/r[k_]:6.1f} fps"
               f"   {'실시간' if r[k_] < FRAME_DT*1000 else '실시간 아님'}"
               f" (기준 {FRAME_DT*1000:.1f} ms)", flush=True)

if a.out:
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(frame_dt=FRAME_DT, n_anchors=a.n_anchors, k=a.k,
                   gpu=torch.cuda.get_device_name(0),
                   rows={f"k{k[0]}_n{k[1]}": v for k, v in rows.items()}),
              open(os.path.join(a.out, "deform_speed.json"), "w"), indent=1,
              ensure_ascii=False)
print("DEFBENCH_OK")
