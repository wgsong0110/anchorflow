"""ficus_v0_greg 덤프의 프레임별 노드 출력(변위 3 + 반경 1)을 나눠 넣어, 같은 상태에서 탄성 에너지 변화 ΔE_el 을 잰다 (진단, CPU).

스텝 t (상태 = 프레임 t: x = x0 또는 pred[t-1], F = I 또는 F[t-1]) 에서 덤프 해 nodes[t][1] = [U | ρ_raw] 를
  (a) U 만, ρ_raw = 0      (글로벌 변형만; 반경은 시작값)
  (b) ρ_raw 만, U = 0      (로컬 변형만)
  (c) U 와 ρ_raw 둘 다     (= 실제 스텝)
  (d) U 를 barycentric 으로 (로컬 변형 없는 매핑, 참고)
로 넣고 F' = (I + ∇u) F, ΔE_el = Σ V ψ(F') − Σ V ψ(F).
검산: (c) 의 입자 위치·F' 가 덤프 pred[t]·F[t] 와 같은지, 다시 만든 격자 노드 위치가 덤프 노드 위치와 같은지.

  python exe/greg_split_el.py --dump .../ficus_v0_greg/roll_ov.pt --data .../F_pg_v0.pt --t1 11 --out .../greg_split_el.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                     # noqa: E402
from anchorflow import simplex as SX                             # noqa: E402
from anchorflow import phys_resid                                # noqa: E402
from anchorflow import skin as SK                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--t1", type=int, default=11)
ap.add_argument("--t0", type=int, default=0, help="시험용: 이 스텝부터")
ap.add_argument("--node_h", type=float, default=0.056569)
ap.add_argument("--n_nodes", type=float, default=32)
ap.add_argument("--hz_ratio", type=float, default=1.0 / 6 ** 0.5)
ap.add_argument("--dt", type=float, default=0.04)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", required=True)
ap.add_argument("--skin", choices=["v0", "fix"], default="v0",
                help="v0 = 옛 구현(du = ΣW U, ficus_v0_greg 덤프), fix = 고친 로컬 변형(λ' = λ + W(ρ) − W(ρ_ref), x' = Σλ'Y; ficus_v0_greg_fix)")
a = ap.parse_args()
gregory_skin = SK.gregory_skin_v0 if a.skin == "v0" else SK.gregory_skin
torch.set_num_threads(a.threads)
torch.set_grad_enabled(False)
dt64 = torch.float64
f32 = torch.float32            # 실행과 같은 정밀도로 격자·스키닝 (에너지 합만 float64)

D = torch.load(a.dump, map_location="cpu", weights_only=False, mmap=True)
G = torch.load(a.data, map_location="cpu", weights_only=False, mmap=True)
cfg = G["cfg"]
x0 = D["x0"].to(f32)
N = x0.shape[0]
# 부피 = 질량 / 밀도 = dx³ / (그 MPM 칸의 입자 수) -- DM.traj_mass 와 _pvol 을 합친 것 (밀도와 무관)
ng_ = int(cfg.get("n_grid", 100)); dx_ = float(cfg.get("grid_lim", 2.0)) / ng_
X0d = G["x"][0].to(dt64)
vi_ = (X0d.float() / dx_).long().clamp(0, ng_ - 1)
fl_ = (vi_[:, 0] * ng_ + vi_[:, 1]) * ng_ + vi_[:, 2]
cn_ = torch.zeros(ng_ ** 3, dtype=dt64).index_add_(0, fl_, torch.ones(N, dtype=dt64))
vol = (dx_ ** 3) / cn_[fl_]
I3 = torch.eye(3, dtype=f32)


def E_el(F):
    psi, _ = phys_resid.psi_of(F.to(f32), cfg, a.dt)
    return float((vol * psi.to(dt64)).sum())


def bary(x, npos, rows, U):
    Xv = npos[rows]
    Minv = torch.linalg.inv((Xv[:, 1:] - Xv[:, :1]).transpose(1, 2))
    l3 = (Minv @ (x - Xv[:, 0])[..., None]).squeeze(-1)
    lam = torch.cat([1.0 - l3.sum(1, keepdim=True), l3], 1)
    dlam = torch.cat([-Minv.sum(1, keepdim=True), Minv], 1)
    Ur = U[rows]
    return (lam[..., None] * Ur).sum(1), (Ur[..., :, None] * dlam[..., None, :]).sum(1)


rows_out = []
print(f"[시작] 입자 {N}, 프레임 0..{a.t1 - 1}, 덤프 키 {sorted(D.keys())}", flush=True)
for t in range(a.t0, a.t1):
    x = x0 if t == 0 else D["pred"][t - 1].to(f32)
    F = I3.expand(N, 3, 3).clone() if t == 0 else D["F"][t - 1].to(f32)
    lo, lat, nn = SX.grid_for_nodes(x, a.n_nodes, hz_ratio=a.hz_ratio, off=None, h_fix=a.node_h)
    idx, lam, _ = SX.locate(x, lo, lat, nn)
    rows, uniq = SX.active_nodes(idx)
    npos = SX.node_pos(lo, lat, nn, uniq)
    nd = D["nodes"][t]
    tp, dpn = nd[0].to(f32), nd[1].to(f32)
    # 노드 짝: 개수 일치를 요구하지 않고 격자 정수 좌표로 맞춘다 (덤프 노드 집합이 그 프레임의 기준).
    # 다시 깐 격자에만 있는 노드(덤프 위치가 반올림으로 다른 칸에 든 입자 때문)는 출력 0 으로 두고 수를 기록한다.
    nnl = [int(nn[k]) for k in range(3)]
    ijk = torch.round((tp.double() - lo.double()) @ lat.Ai.double().T).long()
    inb = ((ijk >= 0) & (ijk < torch.tensor(nnl))).all(1)
    fl_d = (ijk[:, 0] * nnl[1] + ijk[:, 1]) * nnl[2] + ijk[:, 2]
    lut = torch.full((nnl[0] * nnl[1] * nnl[2],), -1, dtype=torch.long)
    lut[fl_d[inb]] = torch.nonzero(inb).squeeze(1)
    m_ = lut[uniq]                                                # 다시 깐 노드 -> 덤프 노드 색인 (-1 = 없음)
    have = m_ >= 0
    n_new_only = int((~have).sum())
    n_dump_only = int(tp.shape[0] - have.sum())
    dpn_full = torch.zeros(npos.shape[0], dpn.shape[1], dtype=dpn.dtype)
    dpn_full[have] = dpn[m_[have]]
    p_aff = int((~have)[rows].any(1).sum())
    pdiff = float((tp[m_[have]] - npos[have]).abs().max())
    dpn = dpn_full
    U, R = dpn[:, :3], dpn[:, 3]
    Z3, Z1 = torch.zeros_like(U), torch.zeros_like(R)
    E0 = E_el(F)
    rec = {"t": t, "nodes": int(npos.shape[0]), "nodes_dump": int(tp.shape[0]), "nodes_rebuilt_only": n_new_only,
           "nodes_dump_only": n_dump_only, "particles_on_unmatched": p_aff, "node_pos_maxdiff": pdiff, "E_el_state": E0,
           "U_med_over_h": float(U.norm(dim=-1).median() / a.node_h),
           "rho_raw_absmax": float(R.abs().max()), "rho_raw_med": float(R.abs().median())}
    for k, (UU, RR) in {"a_U_only": (U, Z1), "b_rho_only": (Z3, R), "c_both": (U, R)}.items():
        du, Du = gregory_skin(x, npos, rows, UU, RR, a.node_h)
        F2 = (I3 + Du) @ F
        rec[k] = {"dE_el": E_el(F2) - E0, "Du_med": float(Du.flatten(1).norm(dim=-1).median()),
                  "du_med_over_h": float(du.norm(dim=-1).median() / a.node_h)}
        if k == "c_both":
            rec["chk_x_maxdiff"] = float((x + du - D["pred"][t].to(f32)).abs().max())
            rec["chk_F_maxdiff"] = float((F2 - D["F"][t].to(f32)).abs().max())
            rec["chk_F_frac_gt1e-3"] = float(((F2 - D["F"][t].to(f32)).abs().flatten(1).max(1).values > 1e-3).float().mean())
            Dc = Du
    dub, Dub = bary(x, npos, rows, U)
    rec["d_bary"] = {"dE_el": E_el((I3 + Dub) @ F) - E0, "Du_med": float(Dub.flatten(1).norm(dim=-1).median()),
                     "du_med_over_h": float(dub.norm(dim=-1).median() / a.node_h)}
    du_a, Du_a = gregory_skin(x, npos, rows, U, Z1, a.node_h)
    rec["c_minus_a_Du_max"] = float((Dc - Du_a).abs().max())
    # 로컬 -> 글로벌 순서 분해 (사용자 지정): 글로벌 = 노드 변위 U 의 barycentric(조각 선형) 보간,
    # 로컬 = 셀 안 Gregory·방사형 스키닝(반경 포함)이 그 위에 더하는 나머지 (실제 ∇u − barycentric ∇u).
    #   F_loc = (I + Du_c − Du_bary) F,  F_full = (I + Du_c) F  ->  ΔE_loc = E(F_loc) − E(F), ΔE_glob = E(F_full) − E(F_loc)
    E_loc = E_el((I3 + Dc - Dub) @ F)
    E_full = E0 + rec["c_both"]["dE_el"]
    rec["dE_loc"], rec["dE_glob"], rec["dE"] = E_loc - E0, E_full - E_loc, E_full - E0
    rows_out.append(rec)
    print(f"t={t:2d} 노드위치차 {pdiff:.1e} 짝없음(새 {n_new_only}, 덤프 {n_dump_only}, 입자 {p_aff}) | E_el(상태) {E0:.4e} | ΔE_el a(U만) {rec['a_U_only']['dE_el']:+.4e} "
          f"b(ρ만) {rec['b_rho_only']['dE_el']:+.4e} c(둘다) {rec['c_both']['dE_el']:+.4e} d(bary) {rec['d_bary']['dE_el']:+.4e} "
          f"| |ρ_raw|max {rec['rho_raw_absmax']:.1e} | c-a ∇u 최대차 {rec['c_minus_a_Du_max']:.1e} "
          f"| ΔE {rec['dE']:+.4e} = 로컬 {rec['dE_loc']:+.4e} + 글로벌 {rec['dE_glob']:+.4e} | 검산 x {rec['chk_x_maxdiff']:.1e} F {rec['chk_F_maxdiff']:.1e} (차>1e-3 입자 비율 {rec['chk_F_frac_gt1e-3']:.1e})", flush=True)
    json.dump(rows_out, open(a.out, "w"), ensure_ascii=False, indent=1)
print("GREG_SPLIT_DONE", flush=True)
