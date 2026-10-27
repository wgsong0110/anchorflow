"""FEM pcg 실시간 판 (1) 사면체 축약 검증 (사용자 10-27). AF_FEM_DUMP 덤프(dump_tXXXX.pt) 하나를 읽어 float64 로:

  a. 바닥 매핑(--bc_softplus / --bc_stick, s/ε ≤ 20) 또는 bc_energy 고착 띠(법선 거리 < dx)에 걸린 입자 = "바닥 후보" 비율
  b. 후보 밖 입자의 국소 탐침 미분 B_p = ∂vec(Jd_p)/∂U_T (9x12) 가 사면체 안에서 같은지 (barycentric -> 사면체 상수)
  c. 입자 12x12 HL (기존 경로: 이중 역전파 12 회) = 관성 m/h² (λλᵀ⊗I) + V_p B_pᵀ K_p B_p 인지 (K_p = ∂²ψ(A F_p)/∂A², 9x9, 같은
     hess_mode 닫힌 식 _fcr_grad_h 를 한 번 더 미분) -- 후보 밖 입자, 사면체 합 기준 상대오차
  d. 사용자 제안 (사면체 안 F 상수): F_p^n 의 사면체 안 퍼짐, Σ V_p ψ(J_T F_p) vs Σ_T V_T ψ(J_T F̄_T) 의 에너지·헤시안 차
  e. 관성 항 일관 질량 축약: Σ m/(2h²)|Σλu - c|² vs ½uᵀMu - bᵀu + const (float64)
  f. 축약 계산 시간 (같은 GPU): K_p (9x9) + 사면체 합 vs 덤프의 기존 t_hess
사용: python exe/fem_reduce_check.py <dump.pt> [--out 요약.json]
"""
import sys, os, json, time, argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import phys_resid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--out", default=None)
A = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
f64 = torch.float64
D = torch.load(A.dump, map_location="cpu", weights_only=False)
R = {}
t = D["t"]; h = float(D["h"]); args = D["args"]
N = D["HL"].shape[0]
HL = D["HL"].to(dev, f64)
Bp = D["Bp"].to(dev, f64)
Jd = D["Jd"].to(dev, f64).reshape(N, 3, 3)
F = D["F"].to(dev, f64).reshape(N, 3, 3)
x = D["x"].to(dev, f64); v = D["v"].to(dev, f64)
m = D["m"].to(dev, f64)
vol = D["vol"].to(dev, f64) if torch.is_tensor(D["vol"]) else torch.full((N,), float(D["vol"]), device=dev, dtype=f64)
lm = D["lm"].to(dev, f64); rw = D["rw"].to(dev); inv = D["inv"].to(dev); uq = D["uq"].to(dev)
U0 = D["U0"].to(dev, f64)[:, :3]
T = int(uq.shape[0])
mu, lam = phys_resid.lame(D["E"], D["nu"])
mu = float(mu.float().mean()) if torch.is_tensor(mu) else float(mu)
lam = float(lam.float().mean()) if torch.is_tensor(lam) else float(lam)
R.update(t=t, N=N, T=T, h=h, mat=D["mat"], mu=mu, lam=lam)
print(f"[덤프] {A.dump} t={t} 입자 {N} 사면체 {T} h {h:.5f} 재질 {D['mat']} μ {mu:.4g} λ {lam:.4g}")

# a. 바닥 후보
q = x + (lm.unsqueeze(-1) * U0[rw]).sum(1)                     # 매핑 전 스텝 끝 위치
dx = float(D["grid_lim"] or 2.0) / float(D["n_grid"])
near = torch.zeros(N, dtype=torch.bool, device=dev)
eps_sp, eps_st = float(args.get("bc_softplus", 0) or 0), float(args.get("bc_stick", 0) or 0)
for bc in (D["bcs"] or []):
    if bc.get("type") != "surface_collider":
        continue
    pt = torch.tensor(bc["point"], device=dev, dtype=f64); nr = torch.tensor(bc["normal"], device=dev, dtype=f64); nr = nr / nr.norm()
    s = ((q - pt) * nr).sum(-1); s0 = ((x - pt) * nr).sum(-1)
    for e in (eps_sp, eps_st):
        if e > 0:
            near |= s / e <= 20.0
    near |= s0 < dx                                                  # bc_energy 고착 띠
tet_near = torch.zeros(T, dtype=torch.bool, device=dev).index_put_((inv,), near, accumulate=True)
R.update(near_frac=float(near.double().mean()), near_n=int(near.sum()), tet_near_frac=float(tet_near.double().mean()))
print(f"[a] 바닥 후보 입자 {int(near.sum())} ({100 * R['near_frac']:.2f}%), 후보 포함 사면체 {100 * R['tet_near_frac']:.2f}%")

# b. B_p 사면체 안 상수?
far = ~near
ref = torch.zeros(T, dtype=torch.long, device=dev).scatter_(0, inv.flip(0), torch.arange(N, device=dev).flip(0))  # 사면체별 첫 입자
dB = (Bp - Bp[ref[inv]]).abs().amax((1, 2))
farT = ~tet_near
mB = float(dB[far & farT[inv]].max()) if bool((far & farT[inv]).any()) else 0.0
R["B_within_tet_maxabs"] = mB; R["B_scale"] = float(Bp.abs().max())
print(f"[b] 후보 없는 사면체에서 B_p 사면체 안 최대 차 {mB:.3e} (|B| 최대 {R['B_scale']:.3e})")

# c. HL = 관성 + 탄성 (후보 밖)
I3 = torch.eye(3, device=dev, dtype=f64)


def grad_A(Av, Fp):          # ∂ψ(A F)/∂A = P(AF) Fᵀ  (hess_mode 닫힌 식)
    Am = Av.reshape(3, 3)
    P = phys_resid._fcr_grad_h((Am @ Fp).unsqueeze(0), mu, lam)[0]
    return (P @ Fp.transpose(0, 1)).reshape(9)


Kfun = torch.func.vmap(torch.func.jacfwd(grad_A), in_dims=(0, 0))
CH = 65536


def K_all(Aa, Ff):
    return torch.cat([Kfun(Aa[i:i + CH].reshape(-1, 9), Ff[i:i + CH]) for i in range(0, Aa.shape[0], CH)])


torch.cuda.synchronize() if dev == "cuda" else None
t0 = time.time()
K = K_all(Jd, F)                                                     # [N,9,9]
torch.cuda.synchronize() if dev == "cuda" else None
tK = time.time() - t0
sig = torch.linalg.svdvals(Jd @ F)
low = sig[:, -1] <= 0.01
Hel = vol[:, None, None] * Bp.transpose(1, 2) @ K @ Bp
L12 = (lm[:, :, None, None, None] * lm[:, None, None, :, None] * I3[None, None, :, None, :]).reshape(N, 12, 12)
Hin = (m / (h * h))[:, None, None] * L12
Hmod = Hel + Hin
ok = far & ~low
rel_p = (HL - Hmod).flatten(1).norm(dim=1) / HL.flatten(1).norm(dim=1).clamp_min(1e-300)
HT_true = torch.zeros(T, 12, 12, device=dev, dtype=f64).index_add_(0, inv[ok], HL[ok])
HT_mod = torch.zeros(T, 12, 12, device=dev, dtype=f64).index_add_(0, inv[ok], Hmod[ok])
R["H_rel_tetsum"] = float((HT_true - HT_mod).norm() / HT_true.norm())
R["H_rel_particle_max"] = float(rel_p[ok].max()); R["H_rel_particle_med"] = float(rel_p[ok].median())
R["low_sigma_n"] = int(low.sum())
print(f"[c] 후보 밖 입자 HL vs 관성+탄성 모형: 사면체 합 상대오차 {R['H_rel_tetsum']:.3e}, 입자별 중앙 {R['H_rel_particle_med']:.3e} 최대 {R['H_rel_particle_max']:.3e} "
      f"(σ_min≤0.01 입자 {R['low_sigma_n']} 제외)")
# 후보 입자: HL - 모형 = 바닥 항 (+ 매핑 탓 차이) 크기
if bool(near.any()):
    R["near_extra_rel"] = float((HL[near] - Hmod[near]).norm() / HL[near].norm())
    print(f"    후보 입자 HL 중 관성+탄성(매핑 없음) 모형으로 안 설명되는 몫 {R['near_extra_rel']:.3e} (바닥 매핑·바닥 항 -- 입자 경로 그대로 둘 부분)")

# d. 사용자 제안: 사면체 안 F 상수
cnt = torch.zeros(T, device=dev, dtype=f64).index_add_(0, inv, torch.ones(N, device=dev, dtype=f64))
VT = torch.zeros(T, device=dev, dtype=f64).index_add_(0, inv, vol)
Fbar = torch.zeros(T, 9, device=dev, dtype=f64).index_add_(0, inv, (vol[:, None] * F.reshape(N, 9))) / VT[:, None]
Fbar = Fbar.reshape(T, 3, 3)
spread = (F - Fbar[inv]).flatten(1).norm(dim=1)
R["F_spread_max"] = float(spread.max()); R["F_spread_mean"] = float(spread.mean())
R["F_dev_from_I_mean"] = float((F - I3).flatten(1).norm(dim=1).mean())


def psi(Fm):
    s_ = torch.linalg.svdvals(Fm).clamp_min(0.01)
    return phys_resid._psi_fcr(s_, mu, lam)


Eel = float((vol * psi(Jd @ F)).sum())
JT = Jd[ref]                                                           # 후보 밖 사면체는 사면체 상수
Ered_F = float((VT * psi(JT @ Fbar)).sum())
Eel_farT = float((vol * psi(Jd @ F))[farT[inv]].sum()); Ered_farT = float((VT * psi(JT @ Fbar))[farT].sum())
R.update(E_el=Eel, E_el_reducedF=Ered_F, E_el_reducedF_rel=abs(Ered_F - Eel) / abs(Eel),
         E_el_reducedF_rel_farT=abs(Ered_farT - Eel_farT) / abs(Eel_farT))
KT = K_all(JT, Fbar)
HelT_F = VT[:, None, None] * Bp[ref].transpose(1, 2) @ KT @ Bp[ref]
HelT_true = torch.zeros(T, 12, 12, device=dev, dtype=f64).index_add_(0, inv, Hel)
R["H_el_reducedF_rel_farT"] = float((HelT_F - HelT_true)[farT].norm() / HelT_true[farT].norm())
print(f"[d] F^n 사면체 안 퍼짐 |F_p-F̄_T| 평균 {R['F_spread_mean']:.3e} 최대 {R['F_spread_max']:.3e} (|F-I| 평균 {R['F_dev_from_I_mean']:.3e}); "
      f"탄성 에너지 Σ V ψ(J F_p) {Eel:.10e} vs Σ_T V_T ψ(J_T F̄_T) {Ered_F:.10e} 상대차 {R['E_el_reducedF_rel']:.3e} "
      f"(후보 없는 사면체만 {R['E_el_reducedF_rel_farT']:.3e}), 탄성 헤시안 사면체 합 상대차 {R['H_el_reducedF_rel_farT']:.3e}")
# 정확 축약 (A 공간): Σ_p V_p K_p 를 사면체로 모은 뒤 B_Tᵀ (.) B_T -- 후보 없는 사면체에서 입자별과 같은가
KTs = torch.zeros(T, 9, 9, device=dev, dtype=f64).index_add_(0, inv, vol[:, None, None] * K)
HelT_A = Bp[ref].transpose(1, 2) @ KTs @ Bp[ref]
R["H_el_exactA_rel_farT"] = float((HelT_A - HelT_true)[farT].norm() / HelT_true[farT].norm())
print(f"    정확 축약 (입자 9x9 K_p 를 사면체 합 후 B_T 로 12x12): 입자별 12x12 합과 상대차 {R['H_el_exactA_rel_farT']:.3e} (후보 없는 사면체)")

# e. 관성 일관 질량 축약
gv = torch.tensor(D["g"], device=dev, dtype=f64)
c = h * v + h * h * gv                                                 # du 의 목표
du = (lm.unsqueeze(-1) * U0[rw]).sum(1)
Ein_dir = float((0.5 * m / (h * h) * ((du - c) ** 2).sum(-1)).sum())
M_ = U0.shape[0]
ii = rw.unsqueeze(2).expand(-1, 4, 4).reshape(-1); jj = rw.unsqueeze(1).expand(-1, 4, 4).reshape(-1)
Mv = (m[:, None, None] * lm.unsqueeze(2) * lm.unsqueeze(1)).reshape(-1)
Mc = torch.sparse_coo_tensor(torch.stack([ii, jj]), Mv, (M_, M_)).coalesce()
bvec = torch.zeros(M_, 3, device=dev, dtype=f64).index_add_(0, rw.reshape(-1), (m[:, None, None] * lm.unsqueeze(-1) * c.unsqueeze(1)).reshape(-1, 3))
cst = float((m * (c * c).sum(-1)).sum())
Ein_red = float(0.5 / (h * h) * ((U0 * (Mc @ U0)).sum() - 2 * (bvec * U0).sum() + cst))
R.update(E_in_direct=Ein_dir, E_in_consistent=Ein_red, E_in_rel=abs(Ein_red - Ein_dir) / abs(Ein_dir))
print(f"[e] 관성 직접 {Ein_dir:.12e} vs 일관 질량 {Ein_red:.12e} 상대차 {R['E_in_rel']:.3e} (자유 마스크 없음 가정)")

# f. 시간
R["t_K_particles"] = tK
print(f"[f] 입자 9x9 K_p (vmap jacfwd, float64) {1e3 * tK:.1f} ms -- 기존 12 회 이중 역전파 헤시안은 doc 의 t_hess 참고")
if A.out:
    json.dump(R, open(A.out, "w"), indent=1, ensure_ascii=False)
    print(f"[저장] {A.out}")
