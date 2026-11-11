"""절 CM110: wolf 모래 재질 정의 대조 (numpy 만, 로컬). 실행: python3 exe/cm110_sand_check.py
정답 = PhysGaussian mpm_utils.py sand_return_mapping + kirchoff_stress_drucker_prager 를 numpy 로 글자 그대로 옮김
(로컬 사본: DreamPhysics·PhysDreamer warp_mpm/mpm_utils.py -- 두 사본 sand_return_mapping 글자 같음; PG 원본의 μ·λ 는 1e7 배율 없음).
우리 = lib/anchorflow/fem_ao_rt.py AO_SRC 의 ao_dpf + ao_rm3 + ao_elem_s 응력 줄 (τ, P = U diag(τ/σ) Vᵀ) 을 글자 그대로 옮김 (부호 있는 SVD).
비교량: F_E, 키르히호프 τ (정답 τ = P(F_E) F_Eᵀ 대칭화, 우리 τ = P(X) Xᵀ)."""
import math
import numpy as np

E, NU, PHI = 5e7, 0.3, 30.0                                   # wolf.pt cfg (PhysGaussian/config/wolf_config.json)
MU = E / (2 * (1 + NU)); LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
SP = math.sin(PHI / 180.0 * 3.14159265)
ALPHA = math.sqrt(2.0 / 3.0) * 2.0 * SP / (3.0 - SP)
CDP = (3 * LAM + 2 * MU) / (2 * MU) * ALPHA                  # fem_ao_cuda.cdp_of


def rsvd(F):                                                  # 회전 U,V + 부호 있는 σ (warp svd3·우리 ssvdf 와 같은 꼴)
    U, s, Vt = np.linalg.svd(F); V = Vt.T
    if np.linalg.det(U) < 0: U[:, 2] *= -1; s[2] *= -1
    if np.linalg.det(V) < 0: V[:, 2] *= -1; s[2] *= -1
    return U, s, V


# ---- 정답 (PG) ----
def pg_rm(Ft):
    U, sig, V = rsvd(Ft)
    eps = np.log(np.maximum(np.abs(sig), 1e-14))
    tr = eps.sum(); eh = eps - tr / 3; en = np.linalg.norm(eh)
    dg = en + (3 * LAM + 2 * MU) / (2 * MU) * tr * ALPHA
    if dg <= 0: return Ft.copy(), 0
    if tr > 0: return U @ V.T, 2
    H = eps - eh * (dg / en)
    return U @ np.diag(np.exp(H)) @ V.T, 1


def pg_tau(FE):
    U, sig, V = rsvd(FE)
    with np.errstate(invalid="ignore"):
        ls = np.log(sig)
    c = 2 * MU * ls / sig + LAM * ls.sum() / sig
    t = U @ np.diag(c) @ V.T @ FE.T
    return 0.5 * (t + t.T)


# ---- 우리 (ao_dpf / ao_rm3 / ao_elem_s 응력) ----
def ao_dpf(S):
    sgn = np.where(S < 0, -1.0, 1.0); sg = np.maximum(np.abs(S), 1e-14); ep = np.log(sg)
    tr = ep.sum(); e = ep - tr / 3; rho = np.linalg.norm(e)
    dg = rho + CDP * tr
    if not dg > 0: return 0, sgn, sg, ep, e, tr, rho, 1.0
    if tr > 0: return 2, sgn, sg, ep, e, tr, rho, 0.0
    return 1, sgn, sg, ep, e, tr, rho, -CDP * tr / rho


def our_rm(X):
    U, S, V = rsvd(X)
    st, sgn, sg, ep, e, tr, rho, s = ao_dpf(S)
    if st == 0: return X.copy(), 0
    sn = sgn if st == 2 else sgn * np.exp(tr / 3 + s * e)
    return U @ np.diag(sn) @ V.T, st


def our_tau(X):
    U, S, V = rsvd(X)
    st, sgn, sg, ep, e, tr, rho, s = ao_dpf(S)
    K3 = LAM + 2 * MU / 3
    tau = np.zeros(3) if st == 2 else ((K3 * tr + 2 * MU * s * e) if st == 1 else (2 * MU * ep + LAM * tr))
    ss = sgn * sg
    P = U @ np.diag(tau / ss) @ V.T
    return P @ X.T


def R(ax, th):
    c, s = math.cos(th), math.sin(th); i, j = [(1, 2), (0, 2), (0, 1)][ax]
    M = np.eye(3); M[i, i] = c; M[j, j] = c; M[i, j] = -s; M[j, i] = s; return M


def iso(J): return np.eye(3) * J ** (1 / 3)


def shear(g): M = np.eye(3); M[0, 1] = g; return M


def uni(l, ax=2): M = np.eye(3); M[ax, ax] = l; return M


Q = R(0, 0.3) @ R(1, -0.7) @ R(2, 1.1)
CASES = [("순수 압축 J 0.9", iso(0.9)), ("순수 압축 J 0.5", iso(0.5)), ("순수 압축 J 0.1", iso(0.1)),
         ("단순 전단 γ 0.05", shear(0.05)), ("단순 전단 γ 0.3", shear(0.3)),
         ("일축 인장 1.05", uni(1.05)), ("등방 인장 J 1.1", iso(1.1)),
         ("일축 압축 0.95", uni(0.95)), ("일축 압축 0.7", uni(0.7)),
         ("압축 J0.8 + 전단 0.02", iso(0.8) @ shear(0.02)), ("압축 J0.8 + 전단 0.3", iso(0.8) @ shear(0.3)),
         ("회전 Q · 압축 0.7 · 전단 0.1", Q @ uni(0.7) @ shear(0.1))]
rng = np.random.default_rng(0)
for k in range(4):
    CASES.append((f"무작위 {k}", Q @ (np.eye(3) + 0.15 * rng.standard_normal((3, 3)))))
CASES.append(("뒤집힘 det<0 (탄성 갈래)", np.diag([1.0, 1.0, -0.98])))
CASES.append(("뒤집힘 det<0 (투영 갈래)", np.diag([0.6, 1.3, -0.9])))

print(f"μ {MU:.4e} λ {LAM:.4e} α {ALPHA:.5f} cdp {CDP:.5f}")
print("| 경우 | det F | 갈래 정답/우리 | ‖ΔF_E‖ | ‖Δτ‖/μ | 정답 p = −tr τ/3 (kPa) | 우리 p (kPa) |")
print("|---|---|---|---|---|---|---|")
for nm, F in CASES:
    Fp, sp = pg_rm(F); Fo, so = our_rm(F)
    tp = pg_tau(Fp); to = our_tau(F)
    dF = np.linalg.norm(Fp - Fo); dt = np.linalg.norm(tp - to) / MU
    pp, po = -np.trace(tp) / 3 / 1e3, -np.trace(to) / 3 / 1e3
    print(f"| {nm} | {np.linalg.det(F):.3f} | {sp}/{so} | {dF:.1e} | {dt:.1e} | {pp:.4g} | {po:.4g} |")

# 압축 복원력: 순수 부피 압축은 tr<0, 편차 0 -> dg = cdp tr < 0 -> 탄성 갈래 (투영 없음). 정답·우리 모두 Hencky p = −K tr ε = −K ln J
print("\n| J | Hencky 키르히호프 p = −K ln J (MPa, 정답·우리 같음) | (참고, 안 씀) FCR 키르히호프 p (MPa) |")
print("|---|---|---|")
K = LAM + 2 * MU / 3
for J in (0.9, 0.5, 0.3, 0.2, 0.1):
    j = J ** (1 / 3); pf = -(2 * MU * (j - 1) * j + LAM * (J - 1) * J)       # τ = 2μ(F−R)Fᵀ + λ(J−1)J I, F = j I
    print(f"| {J} | {-K * math.log(J) / 1e6:.2f} | {pf / 1e6:.2f} |")

# 걸음 크기 (정답은 2e-5 s 서브스텝마다 RM, 우리는 0.04 s 걸음에 한 번) -- 정의가 같아도 비비례 경로는 결과가 갈림 (참고 수치)
print("\n경로: 전단 γ 0.1 다음 압축 J 0.9 (같은 걸음 안)")
Fend = iso(0.9) @ shear(0.1)
Fo, so = our_rm(Fend); to = our_tau(Fend)
n = 2000; Fe = np.eye(3)
for i in range(n):                                            # 앞 절반 전단, 뒤 절반 압축 (곱 증분)
    if i < n // 2: dF = shear(0.1 / (n // 2))
    else: dF = iso(0.9 ** (1 / (n - n // 2)))
    Fe, _ = pg_rm(dF @ Fe)
tp = pg_tau(Fe)
dev = lambda t: np.linalg.norm(t - np.trace(t) / 3 * np.eye(3))
print(f"  정답 2000 서브스텝: p {-np.trace(tp)/3/1e3:.4g} kPa |dev τ| {dev(tp)/1e3:.4g} kPa   우리 한 걸음: p {-np.trace(to)/3/1e3:.4g} kPa |dev τ| {dev(to)/1e3:.4g} kPa (갈래 {so})")
