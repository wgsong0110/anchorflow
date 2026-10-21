"""--integrator midpoint 단위 시험 (CPU, float64).

(A) 질점-스프링 1 자유도 (m=1, k=ω²): 후퇴 오일러 = 증분 포텐셜
        E(u) = m/(2h²)(u − h v)² + ½k(x+u)²
    최소화 (L-BFGS), 중점법 = 같은 증분 포텐셜을 h/2 로 풀고 lib/anchorflow/midpoint.mid_to_end 로 외삽.
    스텝당 진폭비 (E_N/E_0)^(1/2N) 를 이론값 (후퇴 오일러 1/√(1+(ωh)²), 중점법 1) 과 비교.

(B) 작은 격자 진동: 실제 코드 경로의 조각을 그대로 쓴다 --
    simplex.grid_for_nodes(h_fix)·locate·active_nodes·g2p_pre·g2p_jac_pre (매 스텝 현재 위치에서 격자 재구성, 롤아웃과 같음),
    phys_resid.pts_ip_energy (train_deform.ip_of 가 --obj pts 에서 부르는 증분 포텐셜 그 함수, FCR jelly),
    midpoint.mid_to_end·extrap_jac (train_deform._ov_frame 의 --integrator midpoint 가 부르는 함수).
    초기 속도 = 정지 상태 선형화의 일반화 고유모드 Kφ = ω² M φ (M = Wᵀ diag(m) W, K = 탄성 기울기 중심차분),
    작은 진폭. 에너지 ½Σm|v|² + Σ V ψ(F) 의 스텝당 비를 이론값과 비교.

통과 조건: (A) 중점법 |진폭비 − 1| < 1e-6, 후퇴 오일러 상대오차 < 1e-6;
          (B) 중점법 에너지 상대 변화 < 2e-3 (N 스텝 전체), 후퇴 오일러 스텝당 에너지비 / (1/(1+(ωh)²)) 상대오차 < 2e-2.
마지막 줄 MIDPOINT_TEST PASS / FAIL.
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch
from anchorflow import simplex as SX
from anchorflow import phys_resid as PR
from anchorflow import midpoint as MIDP

torch.set_default_dtype(torch.float64)
OK = True


def lbfgs_min(f0, u0, iters=400, tol=1e-11):
    """L-BFGS (강한 Wolfe). 목적함수를 시작점 기울기 최대값으로 나눠 tol 을 상대값으로 쓴다 -> (u, 상대 기울기 최대)."""
    u = u0.clone().requires_grad_(True)
    with torch.enable_grad():
        g0, = torch.autograd.grad(f0(u), u)
    s = 1.0 / max(float(g0.abs().max()), 1e-300)
    f = lambda q: f0(q) * s
    opt = torch.optim.LBFGS([u], lr=1.0, max_iter=iters, tolerance_grad=tol, tolerance_change=1e-30,
                            history_size=50, line_search_fn="strong_wolfe")

    def clo():
        opt.zero_grad()
        e = f(u)
        e.backward()
        return e
    for _ in range(5):
        opt.step(clo)
        with torch.enable_grad():
            g, = torch.autograd.grad(f(u), u)
        if float(g.abs().max()) < tol * 10:
            break
    return u.detach(), float(g.abs().max())


# ---------------- (A) 1 자유도 ----------------
print("[A] 질점-스프링 1 자유도 (m=1, k=ω², x0=1, v0=0), 스텝 40")
for wh in (0.1, 0.5, 1.0, 2.0):
    w, h, N = 1.0, wh, 40
    m, k = 1.0, w * w
    res = {}
    for integ in ("be", "midpoint"):
        x, v = torch.tensor([1.0]), torch.tensor([0.0])
        E0 = 0.5 * m * float(v @ v) + 0.5 * k * float(x @ x)
        for _ in range(N):
            hh = h if integ == "be" else 0.5 * h
            f = lambda u: 0.5 * m / hh ** 2 * ((u - hh * v) ** 2).sum() + 0.5 * k * ((x + u) ** 2).sum()
            u, gm = lbfgs_min(f, hh * v)
            if integ == "be":
                x, v = x + u, u / h
            else:
                x, v = MIDP.mid_to_end(x, x + u, v, h)
        EN = 0.5 * m * float(v @ v) + 0.5 * k * float(x @ x)
        res[integ] = (EN / E0) ** (1.0 / (2 * N))
    th = 1.0 / math.sqrt(1.0 + wh * wh)
    e_be = abs(res["be"] / th - 1.0); e_mp = abs(res["midpoint"] - 1.0)
    ok = e_be < 1e-6 and e_mp < 1e-6
    OK &= ok
    print(f"  ωh={wh:4.1f}: 후퇴 오일러 진폭비 {res['be']:.10f} (이론 1/√(1+(ωh)²) = {th:.10f}, 상대오차 {e_be:.1e}) | "
          f"중점법 {res['midpoint']:.12f} (|−1| {e_mp:.1e}) {'통과' if ok else '실패'}")

# ---------------- (B) 작은 격자 ----------------
cfg = {"material": "jelly", "E": 2e4, "nu": 0.3, "density": 1000.0, "n_grid": 100, "grid_lim": 2.0, "g": [0, 0, 0],
       "boundary_conditions": []}
L_, sp, nh = 0.2, 0.0125, 0.05
g1 = torch.arange(sp / 2, L_, sp)
X0 = torch.stack(torch.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3) + 0.9
Np = X0.shape[0]
m = torch.full((Np,), cfg["density"] * sp ** 3)
vol = m / cfg["density"]
I3 = torch.eye(3)


def build(x):
    lo, lat, nn = SX.grid_for_nodes(x, 32, off=None, h_fix=nh)
    idx, lam, aux = SX.locate(x, lo, lat, nn)
    rows, uniq = SX.active_nodes(idx)
    return rows, lam, aux, lat, int(uniq.shape[0])


def field(st, U):
    rows, lam, aux, lat, _ = st
    du = SX.g2p_pre(rows, lam, U)
    _, G = SX.g2p_jac_pre(rows, lam, aux, lat, U)
    return du, I3 + G


def ip(x, v, F, h, st):
    def f(U):
        du, J = field(st, U)
        E, _, _, _ = PR.pts_ip_energy(x, du, v, F, J, m, vol, cfg, h, cfg["n_grid"], cfg["grid_lim"])
        return E
    return f


def energy(v, F):
    psi, _ = PR.psi_of(F, cfg, 0.0)
    return 0.5 * float((m * (v * v).sum(-1)).sum()), float((vol * psi).sum())


# 정지 상태 선형화 -> 일반화 고유모드
st0 = build(X0)
M0 = st0[4]
rows, lam = st0[0], st0[1]
W = torch.zeros(Np, M0)
W.scatter_add_(1, rows, lam)                                  # u_p = Σ_i W_pi U_i (성분별)
Mn = W.T @ (m.unsqueeze(1) * W)                               # 노드 질량 (일관)
F0 = I3.expand(Np, 3, 3).clone()


def grad_el(Uf):
    U = Uf.reshape(M0, 3).clone().requires_grad_(True)
    _, J = field(st0, U)
    psi, _ = PR.psi_of(J @ F0, cfg, 0.0)
    g, = torch.autograd.grad((vol * psi).sum(), U)
    return g.reshape(-1)


eps = 1e-6
K = torch.stack([(grad_el(eps * e) - grad_el(-eps * e)) / (2 * eps) for e in torch.eye(3 * M0)], 1)
K = 0.5 * (K + K.T)
M3 = torch.kron(Mn, torch.eye(3))
M3 = M3 + 1e-12 * float(M3.diagonal().mean()) * torch.eye(3 * M0)    # 가중치가 정확히 0 인 노드 대비
Lc = torch.linalg.cholesky(M3)
Li = torch.linalg.inv(Lc)
ev, Q = torch.linalg.eigh(Li @ K @ Li.T)
nz = int((ev < 1e-6 * ev.max()).sum())
kk = nz                                                        # 강체 6 개 다음 첫 변형 모드
w = math.sqrt(float(ev[kk]))
phi = (Li.T @ Q[:, kk]).reshape(M0, 3)
print(f"[B] 입자 {Np}, 노드 {M0}, 영 고유값 {nz} 개 (강체 6), 첫 변형 모드 ω = {w:.3f} rad/s, 다음 {math.sqrt(float(ev[kk + 1])):.3f}")
amp = 1e-4 * L_ * w                                             # 변위 진폭 ~ 1e-4 L
V0 = W @ phi
V0 = V0 * (amp / float(V0.norm(dim=-1).max()))
N = 30
for wh in (0.5, 1.0):
    h = wh / w
    out = {}
    for integ in ("be", "midpoint"):
        x, v, F = X0.clone(), V0.clone(), F0.clone()
        K0, E0 = energy(v, F)
        T0 = K0 + E0
        gmax = 0.0
        for _ in range(N):
            st = build(x)
            hh = h if integ == "be" else 0.5 * h
            U0 = torch.zeros(st[4], 3)
            U, gm = lbfgs_min(ip(x, v, F, hh, st), U0)
            gmax = max(gmax, gm)
            with torch.no_grad():
                du, J = field(st, U)
                if integ == "be":
                    x, v, F = x + du, du / h, J @ F
                else:
                    x, v = MIDP.mid_to_end(x, x + du, v, h)
                    F = MIDP.extrap_jac(J) @ F
        K1, E1 = energy(v, F)
        out[integ] = ((K1 + E1) / T0, gmax)
    r_be = out["be"][0] ** (1.0 / N); th = 1.0 / (1.0 + wh * wh)
    e_be = abs(r_be / th - 1.0); e_mp = abs(out["midpoint"][0] - 1.0)
    ok = e_be < 2e-2 and e_mp < 2e-3
    OK &= ok
    print(f"  ωh={wh:.1f} (h={h:.4e} s, {N} 스텝): 후퇴 오일러 스텝당 에너지비 {r_be:.6f} (이론 1/(1+(ωh)²) = {th:.6f}, "
          f"상대오차 {e_be:.1e}; {N} 스텝 총 {out['be'][0]:.3e}) | 중점법 {N} 스텝 에너지비 {out['midpoint'][0]:.8f} "
          f"(|−1| {e_mp:.1e}) | 최대 기울기 {max(out['be'][1], out['midpoint'][1]):.1e} {'통과' if ok else '실패'}")

print("MIDPOINT_TEST " + ("PASS" if OK else "FAIL"))
