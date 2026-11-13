"""절 CM128 로컬 numpy 검사 (f64, GPU 없음): --xsinv 정의 (fem_sinv.py) 를 커널 글자대로 옮겨
 (1) 탄성 갈래에서 P = ∂W/∂X (W = μ Σ ε² + λ/2 (Σ ε)², ε = log σ | σ − 1) 를 중앙 차분과,
 (2) 투영 전 헤시안 (주값 D + 쌍 lt·lf, 커널 식) 을 P 의 중앙 차분과,
 (3) 뒤집힌 점에서 −P 방향 (힘 방향) 으로 조금 움직였을 때 σ_3 가 커지는지 (뒤집힘 풀림) 와 지금 판 (log|σ|) 을 대조,
 (4) 갈래 판정·응력 τ 를 지금 판과 대조 (뒤집히지 않은 점은 같아야 함), (5) ao_rm3 F_E 의 det 부호와 ε 보존.
사용: python exe/sinv_check.py"""
import numpy as np

E, NU = 5e7, 0.3
MU, LAM = E / (2 * (1 + NU)), E * NU / ((1 + NU) * (1 - 2 * NU))
PHI = np.deg2rad(30.0)
AL = np.sqrt(2 / 3) * 2 * np.sin(PHI) / (3 - np.sin(PHI))
CDP = (3 * LAM + 2 * MU) * AL / (2 * MU)


def ssvd(X):
    """부호 있는 SVD: U, V 회전 (det +1), 가장 작은 |σ| 에 부호 (ssvdf 와 같은 규약)."""
    U, s, Vt = np.linalg.svd(X)
    V = Vt.T
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1; s[2] *= -1
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1; s[2] *= -1
    return U, s, V


def eps(s, new):
    if not new:
        return np.log(np.maximum(np.abs(s), 1e-14)), 1 / s, -1 / s ** 2   # 지금 판: ε = log|σ|, ε' = 1/σ (부호 있는 σ 로 나눔)
    pos = s > 0
    e = np.where(pos, np.log(np.maximum(np.abs(s), 1e-14)), s - 1)
    return e, np.where(pos, 1 / s, 1.0), np.where(pos, -1 / s ** 2, 0.0)


def dpf(s, new):
    ep, d1, d2 = eps(s, new)
    tr = ep.sum(); e = ep - tr / 3; rho = np.linalg.norm(e); dg = rho + CDP * tr
    if not dg > 0:
        return 0, ep, d1, d2, tr, e, rho, 1.0
    if tr > 0:
        return 2, ep, d1, d2, tr, e, rho, 0.0
    return 1, ep, d1, d2, tr, e, rho, -CDP * tr / rho


def W(X, new):
    U, s, V = ssvd(X); ep = eps(s, new)[0]
    return MU * (ep ** 2).sum() + 0.5 * LAM * ep.sum() ** 2


def P_el(X, new):
    U, s, V = ssvd(X); ep, d1, d2 = eps(s, new)
    tau = 2 * MU * ep + LAM * ep.sum()
    return U @ np.diag(tau * d1) @ V.T


def tau_rm(X, new):
    U, s, V = ssvd(X); st, ep, d1, d2, tr, e, rho, sc = dpf(s, new)
    K3 = LAM + 2 * MU / 3
    tau = np.zeros(3) if st == 2 else (K3 * tr + 2 * MU * sc * e if st == 1 else 2 * MU * ep + LAM * tr)
    return st, tau, U @ np.diag(tau * d1) @ V.T


def hess(X, new):
    """커널 식 (투영 전): 주값 D, 쌍 lt (회전 e_t), lf (e_f)."""
    U, s, V = ssvd(X); ep, d1, d2 = eps(s, new)
    tau = 2 * MU * ep + LAM * ep.sum(); pe = tau * d1
    D = (2 * MU * np.eye(3) + LAM) * np.outer(d1, d1) + np.diag(tau * d2)
    H = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            ei = np.outer(U[:, i], V[:, i]).ravel(); ej = np.outer(U[:, j], V[:, j]).ravel()
            H += D[i, j] * np.outer(ei, ej)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        lt = (pe[i] + pe[j]) / (s[i] + s[j]); lf = (pe[i] - pe[j]) / (s[i] - s[j])
        x1 = np.outer(U[:, i], V[:, j]); x2 = np.outer(U[:, j], V[:, i])
        et = ((x1 - x2) / np.sqrt(2)).ravel(); ef = ((x1 + x2) / np.sqrt(2)).ravel()
        H += lt * np.outer(et, et) + lf * np.outer(ef, ef)
    return H


def rm3(X, new):
    U, s, V = ssvd(X); st, ep, d1, d2, tr, e, rho, sc = dpf(s, new)
    sgn = np.where(s > 0, 1.0, -1.0) if new else np.where(s < 0, -1.0, 1.0)
    if st == 0 and (not new or (sgn > 0).all()):
        return X
    if new:
        sn = np.ones(3) if st == 2 else np.exp(tr / 3 + sc * e)
    else:
        sn = sgn if st == 2 else sgn * np.exp(tr / 3 + sc * e)
    return U @ np.diag(sn) @ V.T


rng = np.random.default_rng(0)


def rot():
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


cases = []
for s in ([1.8, 1.2, -0.6], [1.3, 1.1, -0.9], [1.05, 1.0, -0.3], [2.2, 1.5, -1.2], [0.9, 0.8, -0.05], [1.0, 0.95, -1.0],
          [1.2, 1.1, 0.9], [0.9, 0.85, 0.7], [1.1, 1.05, 1.02]):
    cases.append((s, rot() @ np.diag(s) @ rot().T))
print(f"μ {MU:.4e} λ {LAM:.4e} cdp {CDP:.5f}")
print("| σ (부호) | det | 갈래 지금/새 | τ 지금 (MPa) | τ 새 (MPa) | P FD 상대오차 새 | H FD 상대오차 새 | 힘 방향으로 σ3 변화 지금 / 새 | F_E det 지금 / 새 | ε 보존 새 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for s, X in cases:
    h = 1e-6
    # (1) P = ∂W/∂X (탄성 갈래 식)
    G = np.zeros(9)
    for k in range(9):
        d = np.zeros(9); d[k] = h
        G[k] = (W(X + d.reshape(3, 3), True) - W(X - d.reshape(3, 3), True)) / (2 * h)
    Pn = P_el(X, True).ravel(); ep1 = np.linalg.norm(Pn - G) / np.linalg.norm(G)
    # (2) H = ∂P/∂X
    HF = np.zeros((9, 9))
    for k in range(9):
        d = np.zeros(9); d[k] = h
        HF[:, k] = (P_el(X + d.reshape(3, 3), True).ravel() - P_el(X - d.reshape(3, 3), True).ravel()) / (2 * h)
    Hn = hess(X, True); eh = np.linalg.norm(Hn - HF) / np.linalg.norm(HF)
    # (3) 반환 사상 뒤 힘 (−P̂) 방향으로 작은 걸음 -> σ3 변화
    ds = []
    for new in (False, True):
        st, tau, Pm = tau_rm(X, new)
        nP = np.linalg.norm(Pm)
        if nP == 0:
            ds.append(0.0); continue
        Y = X - 1e-4 * Pm / nP
        ds.append(ssvd(Y)[1][2] - ssvd(X)[1][2])
    st0, t0, _ = tau_rm(X, False); st1, t1, _ = tau_rm(X, True)
    F0 = rm3(X, False); F1 = rm3(X, True)
    ee = dpf(ssvd(X)[1], True)
    if st1 == 0:
        e_keep = np.abs(eps(ssvd(F1)[1], True)[0] - ee[1]).max()
    elif st1 == 1:
        e_keep = np.abs(eps(ssvd(F1)[1], True)[0] - (ee[4] / 3 + ee[7] * ee[5])).max()
    else:
        e_keep = np.abs(eps(ssvd(F1)[1], True)[0]).max()
    print(f"| {s} | {np.linalg.det(X):.3f} | {st0}/{st1} | {np.round(t0 / 1e6, 2).tolist()} | {np.round(t1 / 1e6, 2).tolist()} | {ep1:.1e} | {eh:.1e} | "
          f"{ds[0]:+.2e} / {ds[1]:+.2e} | {np.linalg.det(F0):+.3f} / {np.linalg.det(F1):+.3f} | {e_keep:.1e} |")
