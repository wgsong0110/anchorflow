"""고친 Gregory 로컬 변형에서 탄성 에너지가 반경에 대해 매끄러운지 (진단, CPU).

덤프 스텝 t 의 상태(x, F)와 노드 출력 [U | ρ_raw] 에서
  (1) 반경 방향 훑기: U 고정, ρ_raw = s·ρ_dump (s ∈ [−2, 2]) -> E_el(s)
  (2) 반경 방향 훑기 (U = 0): 로컬만
  (3) 변위 방향 훑기: ρ_raw = 0, U = s·U_dump -> E_el(s) (비교: 매끄러워야 한다)
  s = 0 양쪽 한쪽 기울기 (E(±ε) − E(0)) / ε 와 껍질(ρ_ref 와 ρ 사이 r) 입자 수, 그 입자들의 ‖∇u‖ 를 적는다.
꺾임(r = ρ)이 원인이면 (1)(2) 가 s = 0 에서 V 자(양쪽 기울기 부호 같음, 크기 O(1))이고 (3) 은 매끄럽다.
  python exe/greg_kink_scan.py --dump roll_ov.pt --data F_pg_v0.pt --steps 0 1 5 10 --out scan.json
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
ap.add_argument("--steps", type=int, nargs="+", default=[0, 1, 5, 10])
ap.add_argument("--node_h", type=float, default=0.056569)
ap.add_argument("--n_nodes", type=float, default=32)
ap.add_argument("--hz_ratio", type=float, default=1.0 / 6 ** 0.5)
ap.add_argument("--dt", type=float, default=0.04)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.set_grad_enabled(False)
f32, f64 = torch.float32, torch.float64
D = torch.load(a.dump, map_location="cpu", weights_only=False, mmap=True)
G = torch.load(a.data, map_location="cpu", weights_only=False, mmap=True)
cfg = G["cfg"]
x0 = D["x0"].to(f32)
N = x0.shape[0]
ng_ = int(cfg.get("n_grid", 100)); dx_ = float(cfg.get("grid_lim", 2.0)) / ng_
vi_ = (G["x"][0].float() / dx_).long().clamp(0, ng_ - 1)
fl_ = (vi_[:, 0] * ng_ + vi_[:, 1]) * ng_ + vi_[:, 2]
cn_ = torch.zeros(ng_ ** 3, dtype=f64).index_add_(0, fl_, torch.ones(N, dtype=f64))
vol = (dx_ ** 3) / cn_[fl_]
I3 = torch.eye(3, dtype=f32)
h = a.node_h


def E_el(F):
    psi, _ = phys_resid.psi_of(F.to(f32), cfg, a.dt)
    return float((vol * psi.to(f64)).sum())


S = [-2, -1, -0.5, -0.1, -0.01, -0.001, 0.0, 0.001, 0.01, 0.1, 0.5, 1, 2]
out = []
for t in a.steps:
    x = x0 if t == 0 else D["pred"][t - 1].to(f32)
    F = I3.expand(N, 3, 3).clone() if t == 0 else D["F"][t - 1].to(f32)
    lo, lat, nn = SX.grid_for_nodes(x, a.n_nodes, hz_ratio=a.hz_ratio, off=None, h_fix=h)
    idx, _, _ = SX.locate(x, lo, lat, nn)
    rows, uniq = SX.active_nodes(idx)
    npos = SX.node_pos(lo, lat, nn, uniq)
    tp, dpn = D["nodes"][t][0].to(f32), D["nodes"][t][1].to(f32)
    if tp.shape[0] != npos.shape[0]:
        print(f"t={t}: 노드 수 다름 -- 건너뜀"); continue
    U, R = dpn[:, :3], dpn[:, 3]
    Z3, Z1 = torch.zeros_like(U), torch.zeros_like(R)
    rec = {"t": t, "rho_raw_absmax": float(R.abs().max()), "U_med_h": float(U.norm(dim=1).median() / h)}
    for name, fn in (("rho_withU", lambda s: (U, s * R)), ("rho_U0", lambda s: (Z3, s * R)), ("U_rho0", lambda s: (s * U, Z1))):
        Es = []
        for s in S:
            UU, RR = fn(s)
            _, Du = SK.gregory_skin(x, npos, rows, UU, RR, h)
            Es.append(E_el((I3 + Du) @ F))
        e0 = Es[S.index(0.0)]
        sl = {eps: ((Es[S.index(eps)] - e0) / eps, (Es[S.index(-eps)] - e0) / eps) for eps in (0.001, 0.01)}
        rec[name] = {"s": S, "E": Es, "slope_plus_minus": {str(k): v for k, v in sl.items()}}
        print(f"t={t} {name}: E(s) " + " ".join(f"{s:g}:{e - e0:+.3e}" for s, e in zip(S, Es))
              + " | 한쪽 기울기 (E(+ε)−E0)/ε, (E(−ε)−E0)/ε: "
              + ", ".join(f"ε={k}: {v[0]:+.3e} {v[1]:+.3e}" for k, v in sl.items()), flush=True)
    # 껍질: ρ_ref 와 ρ_dump 사이 거리의 (입자, 꼭짓점)
    Xv = npos[rows]
    r = (x[:, None] - Xv).norm(dim=-1)
    rho = h * (0.05 + 0.95 * torch.sigmoid(R)[rows]); rho0 = h * 0.525
    shell = ((r - rho0) * (r - rho) < 0).any(1)
    _, Du = SK.gregory_skin(x, npos, rows, Z3, R, h)
    g = Du.flatten(1).norm(dim=1)
    rec.update(shell_particles=int(shell.sum()), grad_shell_med=float(g[shell].median()) if shell.any() else 0.0,
               grad_shell_max=float(g[shell].max()) if shell.any() else 0.0, grad_out_max=float(g[~shell].max()))
    print(f"t={t}: 껍질 입자 {int(shell.sum())} / {N}, 그 ‖∇u_loc‖ 중앙 {rec['grad_shell_med']:.3e} 최대 {rec['grad_shell_max']:.3e}; "
          f"껍질 밖 최대 {rec['grad_out_max']:.3e}", flush=True)
    out.append(rec)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
print("GREG_KINK_SCAN_DONE", flush=True)
