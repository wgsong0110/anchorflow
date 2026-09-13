"""궤적 라벨 생성이 왜 느린가 -- 단계별로 시간을 쪼갠다.

한 프레임의 라벨을 만드는 데 드는 것:
  1. MPM 40 서브스텝
  2. F 내보내기
  3. encode_closed_F  = gram 조립 + 촐레스키 + 최소제곱 두 번
기하가 고정이면 gram 과 촐레스키는 매번 같은 값이라 한 번만 하면 된다.
그 몫이 얼마인지 잰다.
"""
from __future__ import annotations

import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--fit", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--dt_mult", type=int, default=40)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.frame_encode import FrameState

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
fit = load_fitted(sc, args.fit, dev)[0].fit
cache = fit.prepare(); w = cache[0]; Yg = fit.Xc - cache[1]
MASS = sc.volume[sc.keep].clone()
FS = FrameState(fit.pair_g, fit.pair_a, fit.N, fit.M, MASS)
T = MPMTeacher(sc, sparse=fit, frame=True)
print(f"[setup] 앵커 {fit.M}, 가우시안 {fit.N}, 짝 {fit.pair_g.shape[0]:,}, "
      f"K(최대 이웃) {FS.K}, 청크 {FS.chunk}", flush=True)

T._set(T.pos_m.clone(), torch.zeros_like(T.pos_m), T.eye.clone(), torch.zeros_like(T.eye))
def sync(): torch.cuda.synchronize()
def bench(fn, n, warm=3):
    for _ in range(warm): fn()
    sync(); t0 = time.time()
    for _ in range(n): fn()
    sync(); return (time.time() - t0) / n * 1e3

def mpm_frame():
    for _ in range(args.dt_mult):
        T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
ms_mpm = bench(mpm_frame, 5, warm=1)

x = T.solver.export_particle_x_to_torch()
F = T.solver.export_particle_F_to_torch()
def export():
    T.solver.export_particle_x_to_torch(); T.solver.export_particle_F_to_torch()
ms_exp = bench(export, args.n)

ms_gram = bench(lambda: FS.gram_mat(w, 1e-4), args.n)
G = FS.gram_mat(w, 1e-4)
ms_chol = bench(lambda: torch.linalg.cholesky(G.double()), args.n)
L = torch.linalg.cholesky(G.double())
F3 = F.reshape(-1, 3, 3)
ms_wls = bench(lambda: FS._wls(F3.reshape(-1, 9), w, L), args.n)
ms_enc = bench(lambda: FS.encode_closed_F(x, F3, w, Yg, fixed=fit.fixed, p_fix=fit.pos), args.n)
free = ~fit.fixed
ms_chol2 = bench(lambda: torch.linalg.cholesky(G[free][:, free].double()), args.n)

tot = ms_mpm + ms_exp + ms_enc
print(f"\n{'단계':<34}{'ms/프레임':>12}{'비중':>9}")
for nm, v in (("1. MPM 40 서브스텝", ms_mpm), ("2. x, F 내보내기", ms_exp),
              ("3. encode_closed_F 전체", ms_enc)):
    print(f"{nm:<34}{v:>12.2f}{100*v/tot:>8.1f}%")
print(f"{'합계 (프레임당)':<34}{tot:>12.2f}")
print(f"\n-- 3 번의 내역 --")
for nm, v in (("gram 조립 (청크 index_add)", ms_gram), ("촐레스키 (전체 MxM)", ms_chol),
              ("촐레스키 (자유 블록)", ms_chol2), ("최소제곱 _wls 1 회", ms_wls)):
    print(f"{nm:<34}{v:>12.2f}{100*v/ms_enc:>8.1f}%")
fixed_part = ms_gram + ms_chol + ms_chol2
print(f"\n기하가 고정이면 매번 다시 할 필요 없는 몫: {fixed_part:.2f} ms "
      f"(encode 의 {100*fixed_part/ms_enc:.0f}%, 프레임 전체의 {100*fixed_part/tot:.0f}%)")
print(f"250 궤적 x 60 프레임 = 15,000 회 기준 절약: "
      f"{15000*fixed_part/1000/60:.1f} 분")
print("\nLABEL_DONE")
