"""`project` 는 `skin` 의 역인가?

지금까지 "512 개 앵커의 표현 하한" 으로 인용해 온 값은 특정 인코더(`project`) 와
특정 디코더(`skin`) 를 붙인 왕복 손실이다. `project` 는 입자 변위의 가중평균 --
가중치 행렬의 전치에 해당하고, 재구성 오차의 최소화 해가 아니다. 그렇다면 그 값은
하한이 아니라 그냥 이 인코더의 성능이고, 더 나은 앵커 상태를 내는 스테퍼는 그것을
밑으로 뚫을 수 있다.

여기서 두 가지를 잰다. 둘 다 GPU 몇 분이면 끝나고, 결과에 따라 뒤따르는 기하
재피팅과 학생 재학습(며칠) 이 정당화되거나 취소된다.

  A. 항등성   앵커 시뮬레이터의 궤적 p_t 는 **정의상 표현 가능**하다. 그것을 skin 해
              가우시안으로 만든 뒤 다시 project 로 되돌리면, project 가 옳은 역이면
              정확히 p_t 가 나와야 한다. 나오지 않는 만큼이 순수한 인코더 편향이다.

  B. 느슨함   진짜 MPM 프레임에 대해 min_p ||skin(p) - x|| 을 실제로 풀어, 지금의
              project 가 주는 잔차보다 얼마나 내려가는지 본다. 내려가는 만큼이
              "하한" 이라 부르던 값의 과대평가분이다.

A 가 0 에 가깝고 B 가 거의 안 내려가면 project 는 사실상 옳은 역이고 고칠 것이 없다.
"""
from __future__ import annotations

import argparse
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--fit", required=True, help="피팅된 앵커 집합 (SK_*.pt)")
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_case", type=int, default=3, help="임펄스 몇 개로 볼지")
ap.add_argument("--ls_frames", type=int, nargs="+", default=[5, 15, 30],
                 help="B 에서 최소제곱을 풀어 볼 프레임")
ap.add_argument("--ls_iters", type=int, default=300)
ap.add_argument("--ls_lr", type=float, default=1e-3)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.anchor_sparse import load_fitted
from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
wp.init()
torch.manual_seed(args.seed)

sc0 = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor)
sc, _ = load_fitted(sc0, args.fit, dev)
fit, cache = sc.fit, sc._cache
mat = fit.mat
AC = sc.anchor_canonical
T = MPMTeacher(sc0)
BASE = torch.tensor([-0.477, 0.0, 0.0], device=dev)
g = torch.Generator(device=dev); g.manual_seed(args.seed)
print(f"[setup] 앵커 {fit.M}, 물질 가우시안 {int(mat.sum()) if mat.dtype==torch.bool else mat.shape[0]}, "
      f"{args.frames} 프레임", flush=True)


def forces(n):
    """(K, r) 계열에서 n 개. 옛 세 계열 분할은 제거했다 -- 임펄스는 포크 개수와
    반경으로만 결정된다."""
    out = []
    for _ in range(n):
        uk = torch.rand(1, device=dev, generator=g).item()
        kk = max(1, int(round(32 ** uk)))
        ur = torch.rand(1, device=dev, generator=g).item()
        lo, hi = sc0.sim.radius * 0.125, sc0.extent
        rad = lo * ((hi / lo) ** ur)
        out.append((f"K{kk} r{rad / sc0.sim.radius:.2f}x",
                    sc0.random_multi_poke(g, kk, rad, BASE.norm().item())))
    return out


# ---- A. 표현 가능한 상태에서의 항등성 -------------------------------------
print("\n=== A. skin -> project 왕복이 항등인가 (앵커 시뮬 궤적, 정의상 표현 가능) ===")
print(f"{'임펄스':>6} {'프레임':>6} {'앵커 오차/변위':>16} {'가우시안 오차/변위':>20}")
with torch.no_grad():
    for name, f in forces(args.n_case):
        p, v, gp = AC.clone(), sc.initial_velocity(f), sc.pos.clone()
        rows = []
        for t in range(args.frames):
            p, v, gp = sc.explicit_step(p, v, gp, args.dt_mult)
            if not torch.isfinite(p).all():
                print(f"  {name}: {t} 프레임에서 발산"); break
            gm = sc.skin(p, sc.pos.clone())
            p_hat = fit.project(gm[mat], cache)
            d_anchor = (p - AC).norm(dim=-1).max().clamp(min=1e-12)
            e_anchor = (p_hat - p).norm(dim=-1).mean() / d_anchor
            gm_hat = sc.skin(p_hat, sc.pos.clone())
            d_gauss = (gm[mat] - sc.pos[mat]).norm(dim=-1).max().clamp(min=1e-12)
            e_gauss = (gm_hat[mat] - gm[mat]).norm(dim=-1).mean() / d_gauss
            rows.append((t + 1, float(e_anchor), float(e_gauss)))
        for t, ea, eg in ([rows[len(rows) // 2], rows[-1]] if len(rows) > 1 else rows):
            print(f"{name:>6} {t:6d} {100*ea:15.3f}% {100*eg:19.3f}%")


# ---- B. 최소제곱은 얼마나 더 내려가나 --------------------------------------
# 피팅된 집합은 학습된 파라미터를 들고 있고 cache 는 그것들로부터 한 번 계산된
# 텐서다. 그대로 두면 매 반복이 같은 그래프를 두 번 거슬러 올라가 터진다.
# 여기서 자유변수는 앵커 위치 p 하나뿐이므로 나머지는 전부 끊는다.
for _prm in fit.parameters():
    _prm.requires_grad_(False)
cache = tuple(c.detach() if torch.is_tensor(c) else c for c in cache)
sc._cache = cache
print("\n=== B. min_p ||skin(p) - x|| 을 실제로 풀면 (진짜 MPM 프레임) ===")
print(f"{'임펄스':>6} {'프레임':>6} {'project 잔차':>14} {'최소제곱 잔차':>15} {'줄어든 비율':>12}")
for name, f in forces(args.n_case):
    traj = T.trajectory_particles(f, args.frames, args.dt_mult) \
        if hasattr(T, "trajectory_particles") else None
    if traj is None:
        # MPM 을 직접 굴려 입자 궤적을 받는다 (teacher.trajectory 는 앵커로 접어 준다)
        dv = fit.impulse_dv(f, cache)
        v0 = torch.zeros(fit.N, 3, device=dev).index_add_(
            0, fit.pair_g, cache[0].unsqueeze(-1) * dv[fit.pair_a]).contiguous()
        T._set(T.pos_m.clone(), v0, T.eye.clone(), torch.zeros_like(T.eye))
        traj = []
        for _ in range(args.frames):
            for _ in range(args.dt_mult):
                T.solver.p2g2p(None, sc0.sub_dt, device=T.wp_dev)
            traj.append(T.solver.export_particle_x_to_torch().clone())
        traj = torch.stack(traj)

    for fr in args.ls_frames:
        if fr > traj.shape[0]:
            continue
        x = traj[fr - 1]
        span = (x - T.pos_m).norm(dim=-1).max().clamp(min=1e-12)
        with torch.no_grad():
            p0 = fit.project(x, cache)
            r0 = (fit.gaussian_pos(p0, cache) - x).norm(dim=-1).mean() / span

        p = p0.clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=args.ls_lr)
        free = ~fit.fixed
        for _ in tqdm(range(args.ls_iters), desc=f"{name} f{fr}", leave=False):
            opt.zero_grad()
            loss = ((fit.gaussian_pos(p, cache) - x) ** 2).sum(-1).mean()
            loss.backward()
            if p.grad is not None:
                p.grad[fit.fixed] = 0.0      # 고정 앵커는 움직이지 않는다
            opt.step()
        with torch.no_grad():
            r1 = (fit.gaussian_pos(p, cache) - x).norm(dim=-1).mean() / span
        drop = 100 * (1 - float(r1) / max(float(r0), 1e-12))
        print(f"{name:>6} {fr:6d} {100*float(r0):13.3f}% {100*float(r1):14.3f}% "
              f"{drop:11.1f}%")

print("\n  A 가 0 에 가깝고 B 의 감소가 작으면 project 는 사실상 옳은 역이고,")
print("  '표현 하한' 이라는 이름도 유지된다. 크면 지금까지의 하한은 과대평가이고,")
print("  기하 재피팅과 학생 재학습이 정당화된다.")
