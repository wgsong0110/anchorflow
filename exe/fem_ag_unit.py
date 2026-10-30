"""절 AG 단위 대조: fem_pl_cuda (ag_test: return mapping + 키르히호프 응력) vs 정답 솔버 원본 warp 커널 compute_stress_from_F_trial.

같은 F_trial·같은 상태 (항복 응력 / logJp / 손상) 를 두 쪽에 넣고 F_e, τ (대칭화 키르히호프), 갱신된 상태를 비교한다.
연속 적용 (K 걸음: F_trial_k = D_k F_e,k-1, D_k 는 두 쪽 같은 무작위) 으로 경화·손상 누적도 본다.
정답 쪽은 원본 코드 그대로 (PG: /home/dkta/work/PhysGaussian/mpm_solver_warp, GF: /home/dkta/work/GaussianFluent/mpm_solver_warp) -- 물성은 각 씬 정답 config.

  python exe/fem_ag_unit.py --solver pg --out unit_pg.json   (PG: metal·sand·foam·plasticine)
  python exe/fem_ag_unit.py --solver gf --out unit_gf.json   (GF: watermelon NACC)
"""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--solver", choices=["pg", "gf"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dump", default="", help="걸음 1 의 입력·출력 표본 (앞 2000 개) 을 이 접두로 npz 저장 (불일치 분석용)")
ap.add_argument("--only", default="", help="이 사례만 (쉼표)")
a = ap.parse_args()
ROOT = {"pg": "/home/dkta/work/PhysGaussian", "gf": "/home/dkta/work/GaussianFluent"}[a.solver]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "mpm_solver_warp"))
import numpy as np                                                                  # noqa: E402
import torch                                                                        # noqa: E402
import warp as wp                                                                   # noqa: E402
wp.init()
from mpm_solver_warp import MPM_Simulator_WARP                                      # noqa: E402  (정답 원본)
from mpm_utils import compute_stress_from_F_trial                                   # noqa: E402
from anchorflow import fem_pl_cuda as PLC                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402

dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(a.seed)


def rand_rot(n):
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).view(n, 3, 3)


def rand_F(n, sd, vol_sd=0.0):
    ls = torch.randn(n, 3, generator=g, dtype=torch.float64) * sd
    if vol_sd > 0:
        ls = ls + torch.randn(n, 1, generator=g, dtype=torch.float64) * vol_sd
    return (rand_rot(n) @ torch.diag_embed(ls.exp()) @ rand_rot(n).transpose(1, 2)).float()


# 씬 정답 config 의 물성 (출처: PG config / 우리 정답 생성 config)
CASES = {
    "pg": [
        ("metal_plane", dict(material="metal", E=1e5, nu=0.3, yield_stress=100.0, density=1000.0), 0.08, 0.02, 1e-4),                       # PG 공식 plane_config.json
        ("metal_hard", dict(material="metal", E=1e5, nu=0.3, yield_stress=100.0, hardening=1, xi=0.5, density=1000.0), 0.08, 0.02, 1e-4),    # 경화 갈래 (가상)
        ("sand_wolf", dict(material="sand", E=5e7, nu=0.3, friction_angle=30, density=2000.0), 0.01, 0.01, 2e-5),                           # PG 공식 wolf_config.json
        ("foam_hotdog", dict(material="foam", E=1e6, nu=0.3, yield_stress=2000.0, plastic_viscosity=100.0, density=1000.0), 0.02, 0.01, 2e-5),   # wmats/hotdog_viscoplastic.json
        ("foam_hotdog_dtframe", dict(material="foam", E=1e6, nu=0.3, yield_stress=2000.0, plastic_viscosity=100.0, density=1000.0), 0.02, 0.01, 1.0 / 60),
        ("plasticine_bread", dict(material="plasticine", E=2000.0, nu=0.3, yield_stress=200.0, density=200.0), 0.2, 0.05, 1e-4),            # topo/bread_plastic_config.json
    ],
    "gf": [
        ("watermelon", dict(material="watermelon", E=2000.0, nu=0.38, friction_angle=45.0, beta=1.0, xi=3.0, hardening=1.0, density=1.0), 0.05, 0.05, 1e-4),   # wm2/gf/config.json
    ],
}

res = {}
N = a.n
for name, prm_, sd, vsd, dt in CASES[a.solver]:
    if a.only and name not in a.only.split(","):
        continue
    sim = MPM_Simulator_WARP(N, n_grid=10, grid_lim=2.0)
    x0 = torch.rand(N, 3, generator=g).to(dev) * 0.5 + 0.75
    sim.load_initial_data_from_torch(x0, torch.full((N,), 1e-6, device=dev), n_grid=10, grid_lim=2.0)
    sim.set_parameters_dict(dict(prm_, n_grid=10, grid_lim=2.0))
    sim.finalize_mu_lam()
    mu, lam = PR.lame(prm_["E"], prm_["nu"])
    mat, PRM, PP = PLC.params(prm_, mu, lam, dev)
    # 상태 초기값: 항복 응력은 config 값을 중심으로 흩뜨림 (갈래를 다 밟게), logJp 는 [-0.3, 0.2]
    ys0 = torch.full((N,), float(prm_.get("yield_stress", 0.0)))
    if mat == 5:
        ys0 = ys0 * torch.rand(N, generator=g) * 1.2                                   # 일부는 손상 직전
    lj0 = (torch.rand(N, generator=g) * 0.5 - 0.3) if mat == 7 else torch.full((N,), -0.04)
    if hasattr(sim.mpm_model, "yield_stress") and prm_.get("yield_stress") is not None:
        sim.mpm_model.yield_stress = wp.from_torch(ys0.clone().to(dev).contiguous())
    sim.mpm_state.particle_Jp = wp.from_torch(lj0.clone().to(dev).contiguous())
    ys = ys0.clone().to(dev).contiguous()
    lj = lj0.clone().to(dev).contiguous()
    al = torch.ones(N, device=dev)
    Fe_g = torch.eye(3).expand(N, 3, 3).to(dev).contiguous()
    Fe_o = Fe_g.clone()
    recs = []
    for k in range(a.steps):
        D = rand_F(N, sd, vsd).to(dev)
        Ft_g = (D @ Fe_g).contiguous()
        Ft_o = (D @ Fe_o).contiguous()
        # 정답
        sim.mpm_state.particle_F_trial = wp.from_torch(Ft_g.clone(), dtype=wp.mat33)
        wp.launch(kernel=compute_stress_from_F_trial, dim=N, inputs=[sim.mpm_state, sim.mpm_model, dt], device="cuda:0")
        wp.synchronize()
        Fe_g = wp.to_torch(sim.mpm_state.particle_F).clone()
        tau_g = wp.to_torch(sim.mpm_state.particle_stress).clone()
        ys_g = wp.to_torch(sim.mpm_model.yield_stress).clone() if mat in (1, 3, 5) else None
        lj_g = wp.to_torch(sim.mpm_state.particle_Jp).clone()
        mu_g = wp.to_torch(sim.mpm_model.mu).clone()
        # 우리 (같은 입력: 정답 쪽 상태를 그대로 따라가며 한 걸음씩 대조 -- 갈림 누적을 막고 걸음마다 비교)
        Fout = torch.empty(N, 9, device=dev)
        tau = torch.empty(N, 9, device=dev)
        Fin = Ft_g.reshape(N, 9).contiguous()
        PLC.launch("ag_test", N, (np.int32(N), PLC.P(Fin), PLC.P(ys), PLC.P(lj), PLC.P(al), np.int32(mat), PLC.P(PRM), np.float32(dt), PLC.P(Fout), PLC.P(tau)))
        torch.cuda.synchronize()
        Fo = Fout.view(N, 3, 3)
        to = tau.view(N, 3, 3)
        sc_tau = float(tau_g.abs().amax(dim=(1, 2)).median()) + 1e-30
        dF = (Fo - Fe_g).abs().amax(dim=(1, 2))
        dtau = (to - tau_g).abs().amax(dim=(1, 2)) / (tau_g.abs().amax(dim=(1, 2)) + 1e-3 * sc_tau)
        dtau_mu = (to - tau_g).abs().amax(dim=(1, 2)) / mu                          # 절대 차 / μ (모래·수박처럼 응력이 거의 0 인 입자가 많으면 상대 차가 부푼다)
        ok = torch.isfinite(Fe_g.reshape(N, -1)).all(-1) & torch.isfinite(tau_g.reshape(N, -1)).all(-1)
        rec = dict(step=k + 1, n=N, n_gt_finite=int(ok.sum()), n_ours_finite=int(torch.isfinite(to.reshape(N, -1)).all(-1).sum()),
                   n_projected_gt=int(((Fe_g - Ft_g).abs().amax(dim=(1, 2)) > 1e-6).sum()),
                   dF_max=float(dF[ok].max()), dF_q99=float(torch.quantile(dF[ok], 0.99)), dF_med=float(dF[ok].median()),
                   n_dF_gt_1e3=int((dF[ok] > 1e-3).sum()),
                   dtau_rel_max=float(dtau[ok].max()), dtau_rel_q99=float(torch.quantile(dtau[ok], 0.99)), dtau_rel_med=float(dtau[ok].median()),
                   n_dtau_gt_1e3=int((dtau[ok] > 1e-3).sum()), tau_scale_med=sc_tau,
                   dtau_mu_max=float(dtau_mu[ok].max()), dtau_mu_q99=float(torch.quantile(dtau_mu[ok], 0.99)), tau_over_mu_med=float((tau_g.abs().amax(dim=(1, 2)) / mu)[ok].median()))
        if ys_g is not None:
            dy = (ys - ys_g).abs() / (ys_g.abs() + 1e-6 * max(float(prm_.get("yield_stress", 1.0)), 1.0))
            rec.update(dys_rel_max=float(dy[ok].max()), dys_rel_q99=float(torch.quantile(dy[ok], 0.99)), n_dys_gt_1e3=int((dy[ok] > 1e-3).sum()))
        if mat == 5:
            dmg_g = mu_g <= 0
            dmg_o = al <= 0
            rec.update(dmg_gt=int(dmg_g.sum()), dmg_ours=int(dmg_o.sum()), dmg_mismatch=int((dmg_g != dmg_o).sum()))
        if mat == 7:
            dl = (lj - lj_g).abs()
            rec.update(dlogJp_max=float(dl[ok].max()), dlogJp_q99=float(torch.quantile(dl[ok], 0.99)), n_dlogJp_gt_1e4=int((dl[ok] > 1e-4).sum()),
                       logJp_gt_pos=int((lj_g > 0).sum()), logJp_ours_pos=int((lj > 0).sum()))
        recs.append(rec)
        if a.dump and k == 0:
            np.savez(f"{a.dump}_{name}.npz", Ft=Ft_g[:2000].cpu().numpy(), Fe_g=Fe_g[:2000].cpu().numpy(), tau_g=tau_g[:2000].cpu().numpy(),
                     Fe_o=Fo[:2000].cpu().numpy(), tau_o=to[:2000].cpu().numpy(), lj0=lj0[:2000].numpy(), mu=mu, lam=lam, kap=float(PP["kappa"]))
        print(f"[{name}] 걸음 {k + 1}: 정답 투영 {rec['n_projected_gt']}/{N}, |ΔF_e| 최대 {rec['dF_max']:.2e} (99 % {rec['dF_q99']:.2e}, >1e-3 {rec['n_dF_gt_1e3']}), "
              f"τ 상대 차 최대 {rec['dtau_rel_max']:.2e} (99 % {rec['dtau_rel_q99']:.2e}, >1e-3 {rec['n_dtau_gt_1e3']}), |Δτ|/μ 최대 {rec['dtau_mu_max']:.2e} (99 % {rec['dtau_mu_q99']:.2e}; |τ|/μ 중앙 {rec['tau_over_mu_med']:.2e})"
              + (f", ys 상대 차 최대 {rec['dys_rel_max']:.2e}" if ys_g is not None else "")
              + (f", 손상 정답 {rec['dmg_gt']} 우리 {rec['dmg_ours']} 불일치 {rec['dmg_mismatch']}" if mat == 5 else "")
              + (f", logJp 차 최대 {rec['dlogJp_max']:.2e} (>1e-4 {rec['n_dlogJp_gt_1e4']}), logJp>0 정답 {rec['logJp_gt_pos']} 우리 {rec['logJp_ours_pos']}" if mat == 7 else ""),
              flush=True)
        # 다음 걸음은 정답 상태에서 (같은 입력)
        Fe_o = Fe_g.clone()
        if ys_g is not None:
            ys.copy_(ys_g)
        lj.copy_(lj_g)
        if mat == 5:
            al.copy_((mu_g > 0).float())
    res[name] = dict(params=prm_, dt=dt, mat=mat, prm=PP, steps=recs)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print(f"[저장] {a.out}", flush=True)
