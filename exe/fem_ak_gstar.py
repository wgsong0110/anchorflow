"""절 AK.30: Kuhn 칸 순수 굽힘 패치 시험으로 칸 안 분산 에너지 계수 γ*(ν).
한 칸 (simplex.lattice 정육면체, 몸대각선 z, Kuhn 6 사면체) 두께의 보에 선형 탄성 순수 굽힘 정해 (St. Venant: u_s = −κ s ζ, u_ζ = κ/2 (s² + ν (ζ² − η²)), u_η = κ ν η ζ,
σ_ss = −E κ ζ 만) 를 노드 값으로 주고, 칸 안 균일 입자 (사면체 P1 사상) 에서
  E_연속 = ∫ E κ² ζ² / 2,  E_평균 = V_c ψ(ε̄_c) (λ 몫 포함),  E_분산 = Σ_p w_p [μ |∇u_p − ∇u̅|² + λ/2 tr(∇u_p − ∇u̅)²]
를 재고, 굽힘 방향 (축·두께 법선) 을 고르게 뽑은 평균 에너지 비 γ* = ⟨E_연속 − E_평균⟩ / ⟨E_분산⟩ (ν 만 들어간다; E 는 약분).
사용: python exe/fem_ak_gstar.py [ν ...]"""
import itertools
import math
import sys

import numpy as np


def gstar(nu, ndir=200, npt=20000, seed=0):
    rng = np.random.default_rng(seed)
    r, hz = 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(6.0)
    u = [(1.0, 0.0), (-0.5, math.sqrt(3.0) / 2), (-0.5, -math.sqrt(3.0) / 2)]
    A = np.array([[r * u[i][0] for i in range(3)], [r * u[i][1] for i in range(3)], [hz] * 3])
    perms = list(itertools.permutations(range(3)))

    def verts(p):
        v, c = [np.zeros(3)], np.zeros(3)
        for k in p:
            c = c.copy(); c[k] += 1; v.append(c)
        return np.array(v)
    mu, lam = 1.0 / (2 * (1 + nu)), nu / ((1 + nu) * (1 - 2 * nu))
    c0 = A @ np.full(3, 0.5)
    num = den = 0.0
    for _ in range(ndir):
        q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        t, n = q[:, 0], q[:, 1]
        b = np.cross(t, n)
        G = {}
        for p in perms:
            X = verts(p) @ A.T
            d = X - c0
            s, z, e = d @ t, d @ n, d @ b
            U = np.outer(-s * z, t) + np.outer(0.5 * (s * s + nu * (z * z - e * e)), n) + np.outer(nu * e * z, b)
            G[p] = (U[1:] - U[0]).T @ np.linalg.inv((X[1:] - X[0]).T)
        y = rng.random((npt, 3))
        x = y @ A.T
        w = abs(np.linalg.det(A)) / npt
        GP = np.array([G[tuple(k)] for k in np.argsort(-y, axis=1)])
        z = (x - c0) @ n
        Ec = (z * z / 2).sum() * w
        Gb = GP.mean(0)
        eb = 0.5 * (Gb + Gb.T)
        Ea = npt * w * (mu * (eb ** 2).sum() + lam / 2 * np.trace(eb) ** 2)
        Dm = GP - Gb
        Ev = (mu * (Dm ** 2).sum((1, 2)) + lam / 2 * np.trace(Dm, axis1=1, axis2=2) ** 2).sum() * w
        num += Ec - Ea
        den += Ev
    return num / den


if __name__ == "__main__":
    for nu in [float(v) for v in (sys.argv[1:] or ["0.3", "0.4"])]:
        print(f"ν {nu}: γ* {gstar(nu):.4f}", flush=True)
