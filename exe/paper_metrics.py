"""i-PhysGaussian 부록 정의를 그대로 구현한다.

앞서 내가 쓴 정규화는 "레퍼런스가 움직인 거리" 였는데, 논문은 전부 grid_lim 기준의
고정 스케일을 쓴다. 논문이 그 이유를 명시해 두었다 -- 거의 정지한 구간에서 나눗셈이
불안정해지는 것을 피하려고. 내가 그 함정에 그대로 빠졌었다.

  COMD     = (1/T) sum_t ||c_t - c_ref_t|| / grid_lim
  mwRMSD   = (1/T) sum_t sqrt(sum_i m_i dhat^2 / sum_i m_i) / grid_lim
             dhat = grid_lim  (두 궤적 중 한쪽이라도 클램프된 입자)
                  = min(d, grid_lim)  (그 외)
  ImpulseIrr_t = ||P_t - 2P_{t-1} + P_{t-2}|| / (P_scale + eps)
  TorqueIrr_t  = ||L_t - 2L_{t-1} + L_{t-2}|| / (L_scale + eps)
  P_scale = M_total * grid_lim / dt,  L_scale = M_total * grid_lim^2 / dt
"""
import sys, os
W = os.path.expanduser("~/work")
sys.path.insert(0, W + "/anchorflow/lib")
sys.path.insert(0, W + "/DreamPhysics")
import torch, warp as wp
torch.set_grad_enabled(False)
wp.init()
from anchorflow import scene_setup
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.anchor_sparse import load_fitted

cfgp, ply, fitp = sys.argv[1], sys.argv[2], sys.argv[3]
bf = float(sys.argv[4]); frames = int(sys.argv[5]); dm = int(sys.argv[6])
GRID_LIM, EPS = 2.0, 1e-6
sc = scene_setup.build(ply, cfgp, 512, 8, device="cuda", frozen_weights=True,
                       rot_fallback=True, eig_floor=0.02)
T = MPMTeacher(sc)
mat, x0 = T.mat, T.pos_m.clone()
m = wp.to_torch(T.solver.mpm_state.particle_mass).float().clone()
Mtot = m.sum()
dt_frame = dm * sc.sub_dt
P_scale = float(Mtot) * GRID_LIM / dt_frame
L_scale = float(Mtot) * GRID_LIM ** 2 / dt_frame
force = torch.tensor([bf, 0.0, 0.0], device="cuda")
dv = sc.impulse_dv(force)
v_init = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()


def clamped(x):
    return ((x <= EPS) | (x >= GRID_LIM - EPS)).any(-1)


T._set(x0.clone(), v_init.clone(), T.eye.clone(), torch.zeros_like(T.eye))
ref, ref_c = [], []
for _ in range(frames):
    for _ in range(dm):
        T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
    x = T.solver.export_particle_x_to_torch().clone()
    ref.append(x); ref_c.append(clamped(x))
ref = torch.stack(ref); ref_c = torch.stack(ref_c)
print(f"[setup] {frames} 프레임, grid_lim={GRID_LIM}, dt_frame={dt_frame*1000:.0f}ms")
print(f"[setup] P_scale={P_scale:.4g}, L_scale={L_scale:.4g}")


def report(tag, tr, tr_c):
    T_ = tr.shape[0]
    c_k = (m.view(1, -1, 1) * tr).sum(1) / Mtot
    c_1 = (m.view(1, -1, 1) * ref).sum(1) / Mtot
    comd = float(((c_k - c_1).norm(dim=-1) / GRID_LIM).mean())
    d = (tr - ref).norm(dim=-1)
    both = tr_c | ref_c
    dhat = torch.where(both, torch.full_like(d, GRID_LIM), d.clamp(max=GRID_LIM))
    mw = float((((m.view(1, -1) * dhat ** 2).sum(-1) / Mtot).sqrt() / GRID_LIM).mean())
    v = (tr[1:] - tr[:-1]) / dt_frame
    P = (m.view(1, -1, 1) * v).sum(1)
    L = (m.view(1, -1, 1) * torch.cross(tr[:-1] - x0, v, dim=-1)).sum(1)
    d2P = (P[2:] - 2 * P[1:-1] + P[:-2]).norm(dim=-1) / (P_scale + EPS)
    d2L = (L[2:] - 2 * L[1:-1] + L[:-2]).norm(dim=-1) / (L_scale + EPS)
    bmf = float(((tr_c.float() * m.view(1, -1)).sum(-1) / Mtot).max())
    print(f"{tag:>14} {comd:11.3e} {mw:11.3e} {100*bmf:7.3f}% "
          f"{float(d2P.mean()):11.3e} {float(d2L.mean()):11.3e}")


print(f"\n{'방법':>14} {'COMD':>11} {'mwRMSD':>11} {'BMF':>8} "
      f"{'ImpulseIrr':>11} {'TorqueIrr':>11}")
report("MPM 레퍼런스", ref, ref_c)
for tag, fp in (("앵커 학습후", fitp), ("앵커 학습전", None)):
    s = load_fitted(sc, fp, "cuda")[0] if fp else sc
    p, v, gp = s.anchor_canonical.clone(), s.initial_velocity(force), s.pos.clone()
    out, oc = [], []
    for _ in range(frames):
        p, v, gp = s.explicit_step(p, v, gp, dm)
        gp = s.skin(p, gp)
        x = gp[mat]
        out.append(x.clone()); oc.append(clamped(x))
    report(tag, torch.stack(out), torch.stack(oc))
print("PM_DONE")
