"""절 CM129 로컬 원리 시험 (numpy 만, CPU): 모래 (PG sand Drucker-Prager, Hencky) 기둥 무너짐에서
뉴턴 (시험 X 의 Hencky 탄성 접선 + PSD 투영, 지금 판과 같은 꼴) 대 local-global (Projective Dynamics 꼴:
local = 가우스점마다 반환 사상 정확히 -> 투영 변형 / 힘, global = 걸음마다 한 번 만든 상수 SPD 행렬 M/τ² + s·K0) 의 반복당 수축 비교.

모형: 2 차원 평면 변형 (F33 = 1), 사각 쌍선형 요소 2x2 가우스, 갱신 라그랑주 (걸음 시작 형상 기준 ∇N), 걸음 τ = 0.04 s, 후퇴 오일러,
  E 5e7 · ν 0.3 · ρ 2000 · φ 30° (wolf 와 같은 값), 바닥 y = 0 고착 (바닥 노드 디리클레, 반복 중 바닥 밑으로 가면 그 걸음 고정).
  힘 = Σ V0 τ̂ f⁻ᵀ ∇N (τ̂ = 반환 사상 뒤 키르히호프, f = 걸음 증분 변형) -- 절 CM110 의 우리 식과 같음.
PD 행렬 K0 = Σ V0 Bᵀ C0 B (B: 노드 변위 -> 증분 f, C0 = 쉼 상태 선형 탄성 = Hencky 헤시안 at I), s = 배율.
  고정점 x <- x − A⁻¹ g(x) 는 g 가 참 잔차라 수렴하면 뉴턴과 같은 해 (판정 2 의 모형판).
사용: python exe/pd_model.py [h] [frames] [모드...]   모드 = newton:N · pd:N:s · pdaa:N:s:m (Anderson m) · ref
"""
import sys
import time

import numpy as np

E, NU, RHO, PHI = 5e7, 0.3, 2000.0, 30.0
MU = E / (2 * (1 + NU)); LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
SPHI = np.sin(np.radians(PHI)); ALPHA = np.sqrt(2 / 3) * 2 * SPHI / (3 - SPHI)
CDP = (3 * LAM + 2 * MU) * ALPHA / (2 * MU)
TAU = 0.04; G = 9.8


def mesh(h, W=0.4, H=0.8):
    nx, ny = int(round(W / h)), int(round(H / h))
    xs, ys = np.meshgrid(np.arange(nx + 1) * h, np.arange(ny + 1) * h, indexing="ij")
    X = np.stack([xs.ravel(), ys.ravel()], 1)
    idx = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    el = np.stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()], 1)
    return X, el


GP = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / np.sqrt(3)


def dN_ref(xi, eta):
    return 0.25 * np.array([[-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)], [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]])


DNR = np.stack([dN_ref(*g) for g in GP])  # (4 gp, 4 node, 2)


def grads(X, el):
    xe = X[el]  # (ne,4,2)
    J = np.einsum("gai,eaj->egij", DNR, xe)  # dx/dxi (ne,4,2,2)  J[i,j] = Σ x_i dN/dxi_j ...
    J = np.einsum("eaj,gai->egji", xe, DNR)  # J_ji = dx_j/dxi_i -> use as (xi -> x): J[e,g,j,i]
    det = np.linalg.det(J)
    Jinv = np.linalg.inv(J)  # dxi/dx: Jinv[i,j] = dxi_i/dx_j
    dN = np.einsum("gai,egij->egaj", DNR, Jinv)  # dN_a/dx_j
    return dN.reshape(-1, 4, 2), det.reshape(-1)


def svd3(F):
    U, s, Vt = np.linalg.svd(F)
    # 회전으로 (det U, det V = 1), 가장 작은 σ 부호 반영
    dU = np.linalg.det(U); dV = np.linalg.det(Vt)
    U[:, :, 2] *= dU[:, None]; Vt[:, 2, :] *= dV[:, None]; s[:, 2] *= dU * dV
    return U, s, Vt


def rm(FE):
    """PG sand_return_mapping 그대로 (exe/cm110_sand_check.py 와 같은 식). 돌림: 반환 뒤 F_E, 키르히호프 τ̂, 상태."""
    U, s, Vt = svd3(FE)
    eps = np.log(np.maximum(np.abs(s), 1e-14))
    tr = eps.sum(1)
    eh = eps - tr[:, None] / 3
    en = np.linalg.norm(eh, axis=1)
    dg = en + CDP * tr
    st = np.zeros(len(FE), int)
    epsE = eps.copy()
    ap = (dg > 0) & (tr > 0)
    tl = (dg > 0) & ~(tr > 0)
    st[ap] = 2; st[tl] = 1
    epsE[ap] = 0
    epsE[tl] = eps[tl] - (dg[tl] / np.maximum(en[tl], 1e-30))[:, None] * eh[tl]
    sg = np.sign(s); sg[sg == 0] = 1
    FEn = np.einsum("nij,nj,njk->nik", U, sg * np.exp(epsE), Vt)
    tk = 2 * MU * epsE + LAM * epsE.sum(1)[:, None]
    T = np.einsum("nij,nj,nkj->nik", U, tk, U)
    return FEn, T, st


def hencky_P(X):
    U, s, Vt = svd3(X)
    eps = np.log(np.maximum(np.abs(s), 1e-14))
    tk = 2 * MU * eps + LAM * eps.sum(1)[:, None]
    return np.einsum("nij,nj,njk->nik", U, tk / s, Vt)


IDX = [(0, 0), (0, 1), (1, 0), (1, 1)]


class Sim:
    def __init__(s, h):
        s.h = h
        s.X, s.el = mesh(h)
        s.nn = len(s.X)
        s.x = s.X.copy(); s.v = np.zeros_like(s.X)
        dN0, det0 = grads(s.X, s.el)
        s.V0 = det0  # 가우스 무게 1 (2x2 쌍선형, 기준 넓이 4 -> det 이 넓이/4)
        m = np.zeros(s.nn)
        Nval = np.array([[0.25 * (1 + a * g[0]) * (1 + b * g[1]) for (a, b) in [(-1, -1), (1, -1), (1, 1), (-1, 1)]] for g in GP])
        for gq in range(4):
            np.add.at(m, s.el, RHO * det0.reshape(-1, 4)[:, gq:gq + 1] * Nval[gq][None, :])
        s.m = m
        s.FE = np.tile(np.eye(3), (len(s.el) * 4, 1, 1))
        s.gel = np.repeat(s.el, 4, axis=0)  # 가우스점 -> 노드 4

    # 걸음 준비
    def begin(s):
        s.dN, _ = grads(s.x, s.el)
        s.xs = s.x.copy()
        s.yt = s.x + TAU * s.v + TAU ** 2 * np.array([0, -G])
        s.fix = s.xs[:, 1] <= 1e-9

    def finc(s, x):
        return np.einsum("qai,qaj->qij", x[s.gel] - s.xs[s.gel], s.dN) + np.eye(2)

    def trial(s, f):
        X = s.FE.copy()
        X[:, :2, :2] = np.einsum("qij,qjk->qik", f, s.FE[:, :2, :2])
        return X

    def grad(s, x):
        f = s.finc(x)
        X = s.trial(f)
        _, T, st = rm(X)
        fi = np.linalg.inv(f)
        gN = np.einsum("qij,qaj->qai", fi.transpose(0, 2, 1), s.dN)  # f⁻ᵀ ∇N
        fq = s.V0[:, None, None] * np.einsum("qij,qaj->qai", T[:, :2, :2], gN)
        fint = np.zeros_like(x)
        np.add.at(fint, s.gel, fq)
        g = s.m[:, None] / TAU ** 2 * (x - s.yt) + fint
        g[s.fix] = 0
        return g, st

    def Bmat(s, withF):
        """노드 변위 (q 의 4 노드 x 2) -> dX (4 성분) : (nq, 4, 8)"""
        nq = len(s.gel)
        B = np.zeros((nq, 4, 8))
        Fq = s.FE[:, :2, :2] if withF else np.tile(np.eye(2), (nq, 1, 1))
        for c, (i, j) in enumerate(IDX):
            for a in range(4):
                for d in range(2):  # dx_a,d
                    if d != i:
                        continue
                    # dX_ij = Σ_k dx_a,i dN_a,k F_kj
                    B[:, c, 2 * a + d] = np.einsum("qk,qk->q", s.dN[:, a, :], Fq[:, :, j])
        return B

    def assemble(s, B, C):
        Ke = np.einsum("qci,qcd,qdj->qij", B, s.V0[:, None, None] * C, B)
        n = 2 * s.nn
        K = np.zeros((n, n))
        dofs = np.stack([2 * s.gel, 2 * s.gel + 1], 2).reshape(-1, 8)
        for i in range(8):
            for j in range(8):
                np.add.at(K, (dofs[:, i], dofs[:, j]), Ke[:, i, j])
        K[np.diag_indices(n)] += np.repeat(s.m, 2) / TAU ** 2
        return K

    def hess_newton(s, x):
        """시험 X 의 Hencky 탄성 헤시안 (4x4 평면 성분, 중앙 차분) + 고윳값 0 자르기 -- 지금 판과 같은 꼴"""
        f = s.finc(x); X = s.trial(f)
        Fq = s.FE[:, :2, :2]
        C = np.zeros((len(X), 4, 4)); e = 1e-6
        for c, (i, j) in enumerate(IDX):
            Xp = X.copy(); Xp[:, i, j] += e; Xm = X.copy(); Xm[:, i, j] -= e
            dP = (hencky_P(Xp) - hencky_P(Xm)) / (2 * e)
            C[:, :, c] = np.stack([dP[:, a, b] for (a, b) in IDX], 1)
        C = 0.5 * (C + C.transpose(0, 2, 1))
        w, Q = np.linalg.eigh(C)
        C = np.einsum("qij,qj,qkj->qik", Q, np.maximum(w, 0), Q)
        # dX = df F_q, f 기준 ∇N 은 걸음 시작 형상 -- 다만 지금 반복점의 f 로 B 를 만듦 (∂X/∂x 는 x 무관: X = f F_q, f 선형)
        return s.assemble(s.Bmat(True), C)

    def K0(s, sc):
        C0 = np.zeros((4, 4))
        for c, (i, j) in enumerate(IDX):
            for d, (k, l) in enumerate(IDX):
                C0[c, d] = MU * ((i == k) * (j == l) + (i == l) * (j == k)) + LAM * (i == j) * (k == l)
        return s.assemble(s.Bmat(False), sc * np.tile(C0, (len(s.gel), 1, 1)))

    def end(s, x):
        f = s.finc(x)
        X = s.trial(f)
        FEn, _, _ = rm(X)
        s.FE = FEn
        s.v = (x - s.xs) / TAU
        s.x = x


def free(s, x):
    return ~s.fix


def solve_step(s, mode, nit, sc=1.0, aam=0, tol=0.0, rec=None):
    s.begin()
    x = s.xs + TAU * s.v; x[s.fix] = s.xs[s.fix]  # 시작점 = 등속 예측 (바닥 고착 노드 그대로)
    g, st = s.grad(x)
    g0 = np.linalg.norm(s.m * G); hist = [np.linalg.norm(g) / g0]  # 힘 척도 = 무게 ‖m g‖
    A = None; Lc = None
    if mode.startswith("pd"):
        A = s.K0(sc)
    G_hist, X_hist = [], []
    for it in range(nit):
        fx = np.repeat(s.fix, 2)
        if mode == "newton":
            A = s.hess_newton(x)
            Lc = None
        if Lc is None or Lc[1] is not None and not np.array_equal(Lc[1], fx):
            Af = A[np.ix_(~fx, ~fx)]
            Lc = (np.linalg.cholesky(Af), fx.copy())
        L = Lc[0]
        r = g.ravel()[~fx]
        d = np.linalg.solve(L.T, np.linalg.solve(L, r))
        dx = np.zeros(2 * s.nn); dx[~fx] = -d
        xn = x + dx.reshape(-1, 2)
        if aam:  # Anderson (고정점 함수 Φ(x) = x − A⁻¹ g(x))
            X_hist.append(xn.ravel().copy()); G_hist.append((xn - x).ravel().copy())
            if len(G_hist) > aam + 1:
                X_hist.pop(0); G_hist.pop(0)
            if len(G_hist) >= 2:
                dG = np.stack([G_hist[k + 1] - G_hist[k] for k in range(len(G_hist) - 1)], 1)
                dXh = np.stack([X_hist[k + 1] - X_hist[k] for k in range(len(X_hist) - 1)], 1)
                th, *_ = np.linalg.lstsq(dG, G_hist[-1], rcond=None)
                xn = (X_hist[-1] - dXh @ th).reshape(-1, 2)
        # 바닥: 밑으로 간 노드는 이 걸음 고정 (위치 0)
        below = (xn[:, 1] < 0) & ~s.fix
        if below.any():
            xn[below, 1] = 0.0
            s.fix = s.fix | below
            if mode.startswith("pd"):
                Lc = None
        x = xn
        g, st = s.grad(x)
        hist.append(np.linalg.norm(g) / max(g0, 1e-30))
        if tol and hist[-1] < tol:
            break
    s.end(x)
    return hist, st


def run(h, frames, mode, nit, sc=1.0, aam=0, tol=0.0, quiet=False):
    s = Sim(h)
    out = []
    t0 = time.time()
    for fr in range(1, frames + 1):
        hist, st = solve_step(s, mode, nit, sc, aam, tol)
        com = s.x[:, 1] @ s.m / s.m.sum()
        rho = [hist[i + 1] / hist[i] for i in range(len(hist) - 1)]
        _, det = grads(s.x, s.el); r = det / s.V0
        out.append(dict(f=fr, com=com, hist=hist, rho=rho, st=np.bincount(st, minlength=3) / len(st), xmax=s.x[:, 0].max(), r07=int((r < 0.7).sum()), rmin=r.min()))
        if not quiet:
            print(f"  f{fr:02d} 질량중심 {com*1e3:6.1f} mm 앞끝 {s.x[:,0].max()*1e3:6.0f} mm |g|/‖mg‖ 시작 {hist[0]:.1e} 끝 {hist[-1]:.2e} 반복 {len(hist)-1} "
                  f"k=3·6·10·20·40 {' '.join(f'{hist[k]:.1e}' if k < len(hist) else '-' for k in (3,6,10,20,40))} ρ 0~1 {rho[0]:.2f} 중앙 {np.median(rho):.3f} r<0.7 {out[-1]['r07']} rmin {out[-1]['rmin']:.2f} 상태 탄성·꼬리·꼭짓점 {np.round(out[-1]['st'],2)}", flush=True)
    return out, s, time.time() - t0


if __name__ == "__main__":
    h = float(sys.argv[1]); frames = int(sys.argv[2])
    for md in sys.argv[3:]:
        p = md.split(":")
        kind = p[0]
        if kind == "newton":
            print(f"== 뉴턴 {p[1]} 회", flush=True); out, s, t = run(h, frames, "newton", int(p[1]), tol=float(p[2]) if len(p) > 2 else 0)
        elif kind == "pd":
            print(f"== PD {p[1]} 회 s={p[2]}", flush=True); out, s, t = run(h, frames, "pd", int(p[1]), float(p[2]), tol=float(p[3]) if len(p) > 3 else 0)
        elif kind == "pdaa":
            print(f"== PD+Anderson {p[1]} 회 s={p[2]} m={p[3]}", flush=True); out, s, t = run(h, frames, "pd", int(p[1]), float(p[2]), int(p[3]))
        print(f"   {t:.1f} s", flush=True)
