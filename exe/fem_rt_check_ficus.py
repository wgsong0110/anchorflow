"""ficus FEM 실시간 판 대조 (사용자 10-28): train_deform 사면체 적분점 경로 (AF_FEM_TETQ=1, AF_FEM_FAST=tet) 가 같은 상태에서 낸 값과
fem_rt (fem_rt_sim --floor none --cfg_bc --region_density --pg_vdamp 의 커널 경로) 를 직접 비교한다.

입력: train_deform 덤프 (AF_FEM_TETQ_DUMP, 프레임 t 첫 뉴턴 걸음 시작: 상태 x·v·F·m·vol, 노드 위치, 고정 노드, U0, E, 기울기, 사면체 헤시안 투영 전).
(a) 같은 상태·같은 U 에서 E·기울기 (자유 노드)·투영 조립 헤시안 (H·v, 자기 블록)·뉴턴 방향 상대차. 노드는 위치로 짝짓는다.
    U 는 U0 (관성 예측) 와 U0 + 무작위 변위 (자유 노드, 0.05·node_h) 둘 다 -- 뒤의 것은 덤프의 FemFrame 을 같은 클래스로 다시 만들어 잰다
    (다시 만든 FemFrame 이 U0 에서 덤프 값을 비트 수준으로 재현하는지 먼저 확인).
(b) 경계·감쇠: 고정 노드 집합 일치, 네 노드가 모두 묶인 입자 관성 무게 (w) 일치, 질량·부피 (영역 밀도) 일치, 속도 감쇠 배율 일치, U0 (관성 예측 + 고정) 일치.
사용: python exe/fem_rt_check_ficus.py --dump <폴더> --t 0,1,2 --data <F_pg_v0.pt> --out <json>"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
os.environ.setdefault("AF_FEM_FAST_K", "manual")
os.environ.setdefault("AF_FEM_FAST_NEAR", "analytic")
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_fast as FF                                               # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--t", default="0,1,2")
ap.add_argument("--data", required=True)
ap.add_argument("--node_h", type=float, default=0.056569)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
DM.REGION_DENSITY[0] = True
m_rt = DM.traj_mass(d, dev).float()
vol_rt = m_rt / d["_rho"].to(dev).float()
sel = RT.cfg_bc_sel(cfg, d["x"][0].to(dev).float())
g = torch.as_tensor(cfg["g"], device=dev, dtype=torch.float32)
rel = lambda p, q: float((p.double() - q.double()).norm() / q.double().norm().clamp_min(1e-300))
out = dict(args=vars(a), frames=[])
PC = RTC.CudaPcg(20000, dev)
PC.sym = 1


def key(npos):
    return [tuple(r) for r in torch.round(npos.double().cpu() / 1e-4).long().tolist()]


for t in [int(q) for q in a.t.split(",")]:
    D = torch.load(os.path.join(a.dump, f"tetq_t{t:04d}.pt"), map_location="cpu", weights_only=False)
    r = dict(t=t)
    x, v, F = D["x"].to(dev).float(), D["v"].to(dev).float(), D["F"].to(dev).float().reshape(-1, 3, 3)
    N = x.shape[0]
    # ---- (b) 질량·부피·감쇠
    r["mass_rel_max"] = float(((D["m"].to(dev).float() - m_rt).abs() / m_rt).max())
    r["vol_rel_max"] = float(((D["vol"].to(dev).float() - vol_rt).abs() / vol_rt).max())
    r["vdamp_train_deform"] = D["vdamp"]
    r["vdamp_fem_rt"] = RT.pg_vdamp_scale(cfg, h)
    if t == 0:
        r["x0_same_as_data"] = bool(torch.equal(D["x"].float(), d["x"][0].float()))
        r["v0_same_as_data"] = bool(torch.equal(D["v"].float(), d["v"][0].float()))
    # ---- 격자·노드 짝
    G = RTC.GridK(x, a.node_h)
    kt, kr = key(D["npos"]), key(G.npos)
    pos_r = {k_: i for i, k_ in enumerate(kr)}
    perm = torch.tensor([pos_r.get(k_, -1) for k_ in kt], device=dev)            # train_deform 노드 i -> fem_rt 노드 perm[i]
    r["nodes_train_deform"], r["nodes_fem_rt"], r["nodes_unmatched"] = len(kt), len(kr), int((perm < 0).sum())
    assert int((perm < 0).sum()) == 0 and len(kt) == len(kr), r
    M = G.M
    # 입자 -> 노드 (rows) 가 같은 노드를 가리키는지 (barycentric 도)
    rw_t = perm[D["rw"].to(dev)]
    r["rows_same"] = bool(torch.equal(torch.sort(rw_t, 1).values, torch.sort(G.rows, 1).values))
    def to_rt(Xt):
        Y = torch.zeros((M,) + tuple(Xt.shape[1:]), dtype=Xt.dtype, device=dev)
        Y[perm] = Xt.to(dev)
        return Y
    # ---- (b) 고정 노드
    fx_t = to_rt(D["fixd"].to(dev))
    fx_r, vel = RT.cfg_bc_nodes(cfg, sel, G.npos, G.rows, G.lam, m_rt, t * h, h)
    r["fix_train_deform"], r["fix_fem_rt"] = int(fx_t.sum()), int(fx_r.sum())
    r["fix_same"] = bool(torch.equal(fx_t, fx_r))
    # ---- (b) U0 = 관성 예측 + 고정
    U_r, ws = RT.node_inertial(x, v, m_rt, G, g, h)
    U_r = torch.where(fx_r.unsqueeze(-1), (h * vel).to(U_r.dtype), U_r)
    U_t = to_rt(D["U0"].to(dev).float())
    r["U0_rel"] = rel(U_r, U_t)
    # ---- fem_rt 커널 프레임 (fem_rt_sim 과 같은 생성)
    CF = RTC.CudaTetqK(G, x, v, F, m_rt, vol_rt, cfg, h, U_t, fixd=fx_r, floor=False)
    CF.warp, CF.near32, CF.sym = True, True, 1
    w_r = (~fx_r[G.rows].all(1)).float()
    r["w_same"] = bool(torch.equal(w_r.cpu(), D["w"].float().cpu())) if D.get("w") is not None else None
    r["w_zero_particles"] = int((w_r == 0).sum())
    # ---- 덤프 FemFrame 다시 만들기 (train_deform 과 같은 클래스·같은 인자)
    fr = FF.FemFrame(D["x"].to(dev), D["v"].to(dev), D["F"].to(dev), D["m"].to(dev), D["vol"].to(dev), D["rw"].to(dev), D["lm"].to(dev),
                     D["npos"].to(dev), D["fixd"].to(dev), cfg, h, eps_sp=0.0, eps_st=0.0, stick_xn=False, dtype=D["x"].dtype)
    U0t = D["U0"].to(dev)
    fr.tetq_setup(U0t)
    E0, g0, _ = fr.energy_grad(U0t)
    H0, _ = fr.hess_tetq(U0t)
    r["recon_E_rel"] = abs(float(E0) - D["E"]) / abs(D["E"])
    r["recon_g_rel"] = rel(g0.double().cpu(), D["g"])
    r["recon_H_rel"] = rel(H0.cpu(), D["H"]) if D["H"] is not None else None
    tets_t = perm[fr.uq]                                                           # train_deform 사면체 -> fem_rt 노드 번호
    slots = RTC.slots_k(G)
    Ms = PC.Mmax
    free = ~fx_r
    fr3 = free.unsqueeze(-1).expand(-1, 3).reshape(-1)
    for nm, dU in (("U0", None), ("U0+rand", 0.05 * a.node_h)):
        q = dict()
        if dU is None:
            Ut_t, Et, gt, Ht = U0t, D["E"], D["g"].to(dev), D["H"].to(dev)
        else:
            pert = (torch.rand_like(U0t) * 2 - 1) * dU * (~D["fixd"].to(dev)).unsqueeze(-1)
            Ut_t = U0t + pert
            Et, gt, _ = fr.energy_grad(Ut_t)
            Et = float(Et)
            Ht, _ = fr.hess_tetq(Ut_t)
            gt = gt.double()
        Ur = to_rt(Ut_t.float())
        Er, gr = CF.energy_grad(Ur)
        gr = gr.masked_fill(fx_r.unsqueeze(-1), 0.0)
        gtr = to_rt(gt.double()).masked_fill(fx_r.unsqueeze(-1), 0.0)
        q["E_train_deform"], q["E_fem_rt"] = Et, float(Er)
        q["E_rel"] = abs(float(Er) - Et) / abs(Et)
        q["g_rel_free"] = rel(gr, gtr)
        # 투영 조립 헤시안: train_deform 쪽 = 사면체 12x12 |λ| 투영 (바닥 1e-6, float32 고유분해 -- AF_FEM_EIG_F32·EIG_PROJ abs) -> 블록 ELL -> 고정 행·열 제거·대각 I
        Hp, negf = RT.project(Ht, 1e-6)
        Be = RT.BEll(G.gi)
        Be.add_tets(tets_t, Hp)
        Bv = Be.B.view(M, 15, 3, 3)
        kill = fx_r.unsqueeze(1) | fx_r[Be.nb]
        kill[:, 0] = fx_r
        Bv.mul_((~kill)[..., None, None].to(Bv.dtype))
        Bv[fx_r, 0] = torch.eye(3, device=dev, dtype=Bv.dtype)
        # fem_rt 쪽 = 실행과 같은 커널 (워프 사면체 헤시안 + 투영, 대칭 반쪽, asm_k, pin_rows)
        PC.B.zero_()
        CF.cnt.zero_()
        CF.hess_into(Ur, slots, PC.B, 1e-6, 10, 1e-12, Ms=Ms)
        RTC.asm_k(G, PC, 1, Ms)
        RTC.pin_rows(PC, fx_r, M, Ms, 1)
        Bs = PC.B.view(15, 9, Ms)
        nbl = RT.BEll(G.gi, dtype=torch.float32)
        Bd = Bs[:, :, :M].permute(2, 0, 1).reshape(M, 15, 3, 3).clone()
        for k_ in range(1, 8):
            j_ = nbl.nb[:, 7 + k_]
            Bd[:, 7 + k_] = torch.where(nbl.valid[:, 7 + k_, None, None], Bd[j_, k_].transpose(1, 2), torch.zeros_like(Bd[:, 0]))
        Bd = Bd.double()
        mv_r = lambda z: torch.einsum("mkab,mkb->ma", Bd, z.reshape(M, 3)[nbl.nb]).reshape(-1)
        q["diag_block_rel"] = rel(Bd[:, 0], Bv[:, 0])
        hv = []
        for _ in range(3):
            z = torch.randn(3 * M, device=dev, dtype=torch.float64) * fr3
            hv.append(rel(mv_r(z), Be.mv(z)))
        zg = gtr.reshape(-1)
        hv.append(rel(mv_r(zg), Be.mv(zg)))
        q["Hv_rel_rand3_and_g"] = hv
        q["neg_tet_frac_train_deform"] = negf
        q["neg_tet_frac_fem_rt"] = int(CF.cnt[1]) / CF.T
        # 뉴턴 방향: 같은 rtol 1e-6, train_deform 쪽은 float64 블록 야코비 PCG (torch), fem_rt 쪽은 실행과 같은 타일 혼합 정밀도 PCG
        b_t = gtr.reshape(-1)
        D_ = Bv[:, 0].clone()
        Di = torch.linalg.inv(D_)
        pt_, it_t, _, _ = RT.pcg(Be.mv, lambda rr: (Di @ rr.reshape(-1, 3, 1)).reshape(-1), b_t, 10000, 1e-6, 50)
        pr_, it_r, _, _ = PC.solve_tile(M, gr.reshape(-1).double(), 1e-6, 10000, prec=32, mode=1, delta=0.1)
        q["pcg_iters_train_deform_f64"], q["pcg_iters_fem_rt_mix"] = it_t, it_r
        q["p_rel"] = rel(pr_, pt_)
        q["true_res_fem_rt"] = PC.true_res(M, pr_, gr.reshape(-1).double())
        r[nm] = q
    out["frames"].append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
print("CHECK_DONE", flush=True)
