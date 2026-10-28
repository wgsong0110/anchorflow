"""ficus ψ 가 화분 고정 상자 윗면 띠에 몰리는 원인 확인 (사용자 10-28).

같은 정답 시작 상태 (프레임 0) 에서 굴린 여러 판의 롤아웃 덤프 (F float32) 를 같은 프레임에서 비교한다:
  - 띠 (0 프레임 z ∈ 상자 윗면 ± band) 입자의 ψ (상자 안 / 밖 입자 따로), ψ 합 중 띠 비중, max|σ-1|
  - 격자 (0 프레임 상태 GridK, 고정 노드 = cfg_bc_nodes) 의 띠 사면체가 (i) 고정 노드와 자유 노드를 섞는지 (ii) 상자 안·밖 입자를 섞는지
  - 프레임 1 (모든 판이 같은 시작 F₀ 에서 한 스텝): 각 판의 실제 입자 F₁ 에서 사면체 A_T = F₁ F₀⁻¹ (사면체 안 평균) 을 되찾아
    사면체 적분점 에너지 V_T ψ(A_T F̄₀) 와 입자 적분점 에너지 Σ V_p ψ(A_T F₀_p) 를 비교 -- F̄ (셀 대표 F) 가 F₀ 의 입자 간 퍼짐을 덮는지.
사용: python exe/fem_rt_psi_band.py --data <F_pg_v0.pt> --run 이름=roll_fem.pt ... --t 1,2,3,40,125 --out <json>"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--run", action="append", default=[])
ap.add_argument("--t", default="1,2,3,40,125")
ap.add_argument("--node_h", type=float, default=0.056569)
ap.add_argument("--band", type=float, default=0.06)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
DM.REGION_DENSITY[0] = True
m = DM.traj_mass(d, dev).float()
vol = m / d["_rho"].to(dev).float()
x0 = d["x"][0].to(dev).float()
N = x0.shape[0]
Fgt = DM.traj_F(d, h, dev)
F0 = Fgt[0].to(dev).float().reshape(N, 3, 3)
box = [b for b in cfg["boundary_conditions"] if b["type"] == "cuboid"][0]
q, sz = torch.tensor(box["point"], device=dev), torch.tensor(box["size"], device=dev)
inbox = ((x0 - q).abs() < sz).all(-1)
ztop = float(q[2] + sz[2])
band = (x0[:, 2] - ztop).abs() < a.band
psi = lambda F: torch.cat([PR.psi_of(F[s:s + 65536], cfg, h)[0] for s in range(0, F.shape[0], 65536)])


def smax(F):
    S = torch.linalg.svdvals(F.double())
    return (S - 1).abs().amax(1)


out = dict(args=vars(a), ztop=ztop, n_band=int(band.sum()), n_inbox=int(inbox.sum()), n_band_inbox=int((band & inbox).sum()))
# ---- 격자 (프레임 0) 섞임
G = RTC.GridK(x0, a.node_h)
sel = RT.cfg_bc_sel(cfg, x0)
fx, _ = RT.cfg_bc_nodes(cfg, sel, G.npos, G.rows, G.lam, m, 0.0, h)
nfx_t = fx[G.tets].sum(1)                                                          # 사면체의 고정 꼭짓점 수
T = G.T
cnt = torch.zeros(T, device=dev).index_add_(0, G.pinv, torch.ones(N, device=dev))
cin = torch.zeros(T, device=dev).index_add_(0, G.pinv, inbox.float())
tband = torch.zeros(T, device=dev).index_add_(0, G.pinv, band.float()) > 0
mixP = (cin > 0) & (cin < cnt)
part = (nfx_t > 0) & (nfx_t < 4)
out["grid"] = dict(T=T, band_tets=int(tband.sum()), band_tets_partially_fixed=int((tband & part).sum()), band_tets_all_fixed=int((tband & (nfx_t == 4)).sum()),
                   band_tets_free=int((tband & (nfx_t == 0)).sum()), band_tets_mixed_inbox_outbox=int((tband & mixP).sum()),
                   all_tets_partially_fixed=int(part.sum()), all_tets_mixed=int(mixP.sum()),
                   particles_in_partially_fixed_tets=int(part[G.pinv].sum()), band_particles_in_partially_fixed=int((part[G.pinv] & band).sum()))
# F₀ 의 사면체 안 퍼짐 (F̄₀ 와의 차)
VT = torch.zeros(T, device=dev).index_add_(0, G.pinv, vol)
FT0 = torch.zeros(T, 3, 3, device=dev).index_add_(0, G.pinv, vol[:, None, None] * F0) / VT[:, None, None]
dF0 = (F0 - FT0[G.pinv]).flatten(1).norm(dim=1)
out["F0_spread"] = dict(all_mean=float(dF0.mean()), band_mean=float(dF0[band].mean()), band_max=float(dF0[band].max()),
                        F0_dev_from_I_band_mean=float((F0[band] - torch.eye(3, device=dev)).flatten(1).norm(dim=1).mean()))
runs = {"정답": None}
for r_ in a.run:
    k_, p_ = r_.split("=", 1)
    runs[k_] = p_
ts = [int(z) for z in a.t.split(",")]
res = {}
for name, path in runs.items():
    R = {}
    D = None if path is None else torch.load(path, map_location="cpu", weights_only=False)
    L = Fgt.shape[0] - 1 if D is None else D["F"].shape[0]
    for t in ts:
        if t > L:
            continue
        F = (Fgt[t] if D is None else D["F"][t - 1]).to(dev).float().reshape(N, 3, 3)
        p = psi(F)
        sm = smax(F)
        R[t] = dict(psi_mean=float(p.mean()), psi_band_frac=float(p[band].sum() / p.sum()),
                    psi_band_inbox_mean=float(p[band & inbox].mean()), psi_band_outbox_mean=float(p[band & ~inbox].mean()),
                    psi_nonband_mean=float(p[~band].mean()), smax_band_p99=float(sm[band].quantile(0.99)), smax_nonband_p99=float(sm[~band].quantile(0.99)),
                    psi_partfix_tet_particles_mean=float(p[part[G.pinv]].mean()), psi_other_particles_mean=float(p[~part[G.pinv]].mean()))
        if t == 1:
            # 사면체 A_T = F₁ F₀⁻¹ (사면체 안 질량 무관 평균; 선형 사면체면 사면체 안 모든 입자에서 같아야 한다 -- 격자는 프레임 0 상태 GridK)
            Ap = F @ torch.linalg.inv(F0)
            AT = torch.zeros(T, 3, 3, device=dev).index_add_(0, G.pinv, Ap) / cnt[:, None, None]
            Adev = (Ap - AT[G.pinv]).flatten(1).norm(dim=1)
            eT = VT * psi(AT @ FT0)                                                 # 사면체 적분점
            eP = torch.zeros(T, device=dev).index_add_(0, G.pinv, vol * psi(AT[G.pinv] @ F0))   # 입자 적분점, 같은 A_T
            bt = tband
            R[t].update(A_within_tet_dev_band_max=float(Adev[band].max()), A_within_tet_dev_all_max=float(Adev.max()),
                        E_tetq_band=float(eT[bt].sum()), E_partq_same_A_band=float(eP[bt].sum()),
                        E_tetq_all=float(eT.sum()), E_partq_same_A_all=float(eP.sum()),
                        E_tetq_band_partfix=float(eT[bt & part].sum()), E_partq_band_partfix=float(eP[bt & part].sum()),
                        strain_band_tets_p99=float(smax(AT[bt]).quantile(0.99)), strain_partfix_band_tets_mean=float(smax(AT[bt & part]).mean()),
                        strain_free_band_tets_mean=float(smax(AT[bt & (nfx_t == 0)]).mean()) if bool((bt & (nfx_t == 0)).any()) else None)
        print(name, t, json.dumps(R[t], ensure_ascii=False), flush=True)
    res[name] = R
    del D
out["runs"] = res
print(json.dumps(out["grid"], ensure_ascii=False), json.dumps(out["F0_spread"], ensure_ascii=False), flush=True)
json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
print("BAND_DONE", flush=True)
