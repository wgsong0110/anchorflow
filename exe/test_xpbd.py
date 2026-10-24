"""lib/anchorflow/xpbd.py 단위 시험 (CPU float64). 끝에 XPBD_TEST PASS / FAIL.

  0) 격자: 같은 색 사면체 꼭짓점 겹침 0, Σλ X_node = x, 셀 사상 J = D_s D_m⁻¹ 가 train_deform 의 I + ∇u (SX.g2p_jac_pre) 와 같다
  0b) 구속 에너지: ½C_R²/α_R + ½C_V²/α_V = Σ V_p ψ_FCR(J F_p) (입자별 회전이라 셀 안 F 가 달라도 정확)
  1) 정지: 중력·바닥 없이 정지 물체 (F = I, v = 0) -> 위치·F 그대로 (N=4 K=1, N=1 K=10)
  2) 자유낙하: 중력만 -> 질량중심 속도 = g t, 프레임 2 계 차분 = g Δt², 모양 그대로
  3) 사면체 하나를 x 로 1.3 배 늘였다 놓기 -> 정지 모양을 지나 되돌아오고 (σ_max 가 1 근처까지), 에너지(운동+탄성)가 처음보다 늘지 않고, 뒤집히지 않는다
  4) 바닥: 비스듬히 떨어지는 정육면체 -> 입자 바닥 침투 ≤ 0.02 h, (det F·닿은 뒤 접선 미끄러짐·끝 속도는 출력만 -- 뒤집힘은 실험에서 재는 결과)
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                     # noqa: E402

from anchorflow import simplex as SX                             # noqa: E402
from anchorflow import xpbd as XP                                # noqa: E402

torch.manual_seed(0)
dt64 = torch.float64
OK = []


def chk(name, cond, msg):
    OK.append(bool(cond))
    print(f"[{'통과' if cond else '실패'}] {name}: {msg}", flush=True)


def cube(c, side, sp):
    n = int(round(side / sp))
    g = (torch.arange(n, dtype=dt64) + 0.5) * sp - side / 2
    X = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return X + torch.tensor(c, dtype=dt64)


E, nu, rho = 1e5, 0.3, 1000.0
mu, la = XP.lame(E, nu)
h = 0.05
sp = h / 3
x0 = cube([0.5, 0.5, 0.5], 0.2, sp)
Np = x0.shape[0]
Vp = torch.full((Np,), sp ** 3, dtype=dt64)
m = rho * Vp
I3 = torch.eye(3, dtype=dt64).expand(Np, 3, 3).clone()
print(f"[설정] 입자 {Np}, node_h {h}, μ {mu:.4g} λ {la:.4g} (fixed corotated)", flush=True)

# ---- 0) 격자 ----
S = XP.build_lattice(x0, h)
bad = XP.check_coloring(S["tets"], S["groups"])
rec = (S["lam"].to(dt64).unsqueeze(-1) * S["npos"].to(dt64)[S["rows"]]).sum(1)
dp = 0.01 * torch.randn(S["npos"].shape[0], 3, dtype=dt64)
DmI, Vt = XP.tet_rest(S["npos"], S["tets"])
Jt = XP.tet_J(S["npos"].to(dt64) + dp, S["tets"], DmI)[S["pinv"]]
_, G = SX.g2p_jac_pre(S["rows"], S["lam"].to(dt64), S["aux"], SX.lattice(S["lat"].s, S["lat"].hz, dtype=dt64), dp)
dJ = float((Jt - (torch.eye(3, dtype=dt64) + G)).abs().max())
chk("격자", bad == 0 and float((rec - x0).abs().max()) < 1e-12 and dJ < 1e-10,
    f"사면체 {S['T']} 색 {len(S['groups'])} 겹친 색 {bad}, |Σλ X − x| 최대 {float((rec - x0).abs().max()):.2e}, |J − (I+∇u)| 최대 {dJ:.2e}, "
    f"사면체 부피 {float(Vt.mean()):.4e} (h³/6 = {h ** 3 / 6:.4e})")

# ---- 0b) 구속 에너지 = V ψ_FCR ----
Fr = torch.linalg.matrix_exp(0.3 * torch.randn(Np, 3, 3, dtype=dt64))            # 입자마다 다른 F
Tn = S["T"]
first = torch.full((Tn,), Np, dtype=torch.long).scatter_reduce_(0, S["pinv"], torch.arange(Np), "amin")
Fsame = Fr[first][S["pinv"]]                                                      # 셀 안 같은 F
Jr = torch.linalg.matrix_exp(0.2 * torch.randn(Tn, 3, 3, dtype=dt64))
ok0 = []
for nm_, FF in (("셀 안 같은 F", Fsame), ("입자마다 다른 F", Fr)):
    V_, B_, M_, m1_, m2_ = XP.aggregates(S["pinv"], Tn, Vp, FF)
    CR_, CV_ = XP.cell_constraints_particle(Jr, S["pinv"], Tn, FF, Vp, V_, m1_, m2_)
    Ec = float((V_ * (mu * CR_ ** 2 + 0.5 * la * CV_ ** 2)).sum())                # ½C²/α 합
    Ep = float((Vp * XP.psi_fcr(Jr[S["pinv"]] @ FF, mu, la)).sum())               # 입자별 Σ V ψ_FCR(J F_p)
    # C_V 는 상수만 다르다: 그 상수를 더해 비교
    cst = float((V_ * 0.5 * la * (1 - m1_ ** 2 / m2_)).sum())      # Σ V_p (j a_p − 1)² = V m2 (j − m1/m2)² + V (1 − m1²/m2)
    ok0.append((nm_, Ec + cst, Ep))
r0 = abs(ok0[0][1] / ok0[0][2] - 1)
r1 = abs(ok0[1][1] / ok0[1][2] - 1)
chk("구속 에너지", r0 < 1e-10 and r1 < 1e-10,
    f"셀 안 같은 F: Σ V(½C_R²/α_R+½C_V²/α_V) {ok0[0][1]:.10e} vs Σ V_p ψ_FCR(J F_p) {ok0[0][2]:.10e} (상대차 {r0:.1e}); "
    f"입자마다 다른 F: {ok0[1][1]:.10e} vs {ok0[1][2]:.10e} (상대차 {r1:.1e})")

# ---- 1) 정지 ----
for (N, K) in ((4, 1), (1, 10)):
    sol = XP.Solver(mu, la, N, K)
    x, v, F = x0.clone(), torch.zeros_like(x0), I3.clone()
    for i in range(10):
        x, v, F, _, st = XP.advance_frame(sol, x, v, F, m, Vp, h, [0, 0, 0], 1 / 60)
    dx = float((x - x0).abs().max()); dF = float((F - torch.eye(3, dtype=dt64)).abs().max()); vm = float(v.abs().max())
    chk(f"정지 N={N} K={K}", dx < 1e-9 * h and dF < 1e-9 and vm < 1e-9, f"10 프레임 뒤 |x − x0| 최대 {dx:.2e} (h {h}), |F − I| 최대 {dF:.2e}, |v| 최대 {vm:.2e}")

# ---- 2) 자유낙하 ----
gz = [0.0, 0.0, -9.8]
dt = 1 / 60
sol = XP.Solver(mu, la, 4, 1)
x, v, F = x0.clone(), torch.zeros_like(x0), I3.clone()
com = [x.mean(0)]
for i in range(10):
    x, v, F, _, st = XP.advance_frame(sol, x, v, F, m, Vp, h, gz, dt)
    com.append((m.unsqueeze(1) * x).sum(0) / m.sum())
t = 10 * dt
vc = (m.unsqueeze(1) * v).sum(0) / m.sum()
a2 = (com[-1] - 2 * com[-2] + com[-3]) / dt ** 2
shape = float(((x - com[-1]) - (x0 - com[0])).abs().max())
chk("자유낙하", abs(float(vc[2]) - (-9.8 * t)) < 1e-9 * 9.8 * t and float(vc[:2].abs().max()) < 1e-12 and abs(float(a2[2]) + 9.8) < 1e-6 and shape < 1e-9,
    f"질량중심 속도 z {float(vc[2]):.10f} (g t = {-9.8 * t:.10f}), 프레임 2 계 차분 가속도 {float(a2[2]):.8f}, 모양 변화 최대 {shape:.2e}")

# ---- 3) 사면체 하나 늘였다 놓기 ----
lat = SX.lattice(0.1, 0.1 * XP.HZ, dtype=dt64)
X = torch.stack([torch.zeros(3, dtype=dt64), lat.A[:, 0], lat.A[:, 0] + lat.A[:, 1], lat.A.sum(1)], 0)
S1 = dict(tets=torch.tensor([[0, 1, 2, 3]]), groups=[torch.tensor([0])], T=1,
          pgroups=[(torch.tensor([0]), torch.tensor([0]))])                     # 입자 하나 (F = I, 부피 = 사면체 부피)
DmI1, V1 = XP.tet_rest(X, S1["tets"])
cen = X.mean(0)
xs = X.clone(); xs[:, 0] = cen[0] + 1.3 * (X[:, 0] - cen[0]); xs0_ = xs.clone()
vs = torch.zeros_like(xs)
w = torch.full((4,), 1.0 / (1000.0 * float(V1) / 4), dtype=dt64)
E1, nu1 = 1e4, 0.3
mu1, la1 = XP.lame(E1, nu1)
sol = XP.Solver(mu1, la1, 10, 1)
Bm, Mbm, m1v, m2v = torch.zeros(1, 3, 3, dtype=dt64), torch.eye(3, dtype=dt64)[None], torch.ones(1, dtype=dt64), torch.ones(1, dtype=dt64)
smax, dets, j00, etot = [], [], [], []
m4 = 1000.0 * float(V1) / 4
Etot = lambda xq, vq: float(V1 * XP.psi_fcr(XP.tet_J(xq, S1["tets"], DmI1), mu1, la1)) + 0.5 * m4 * float((vq * vq).sum())   # noqa: E731
s0 = float(torch.linalg.svdvals(XP.tet_J(xs, S1["tets"], DmI1))[0, 0])
E_0 = None
for i in range(600):
    xs, vs, _ = sol.frame(S1, xs, vs, w, [0, 0, 0], 1e-3, Fp=torch.eye(3, dtype=dt64)[None], Vp=V1.clone(), V=V1, B=Bm, Mb=Mbm, m1=m1v, m2=m2v, DmI=DmI1)
    J = XP.tet_J(xs, S1["tets"], DmI1)
    etot.append(Etot(xs, vs))
    smax.append(float(torch.linalg.svdvals(J)[0, 0])); dets.append(float(torch.linalg.det(J)[0])); j00.append(float(J[0, 0, 0]))
first_back = next((i for i, s in enumerate(j00) if s < 1.0), None)
E_0 = Etot(xs0_, torch.zeros_like(xs0_))
chk("사면체 늘였다 놓기", first_back is not None and min(smax) < 1.05 and max(etot) <= E_0 * (1 + 1e-6) and min(dets) > 0,
    f"처음 σ_max {s0:.4f} (J_xx 1.3), J_xx 가 1 아래로 (정지 모양을 지나) 간 첫 프레임 {first_back} (프레임 1 ms), J_xx 최소 {min(j00):.4f}, "
    f"σ_max 최대 {max(smax):.4f} 최소 {min(smax):.4f}, 에너지(운동+탄성) 처음 {E_0:.4e} 최대 {max(etot):.4e} 끝 {etot[-1]:.4e}, "
    f"평균 {sum(smax) / len(smax):.4f}, det 최소 {min(dets):.4f} 최대 {max(dets):.4f}")

# ---- 4) 바닥 ----
cfgf = dict(boundary_conditions=[dict(type="surface_collider", point=[0, 0, 0.0], normal=[0, 0, 1], surface="sticky")])
for (N, K) in ((1, 1), (4, 1), (1, 10)):
    sol = XP.Solver(mu, la, N, K, planes=XP.planes_of(cfgf), mu_s=math.inf)
    x = x0.clone(); x[:, 2] += -x0[:, 2].min() + 0.03          # 바닥 위 3 cm
    v = torch.zeros_like(x); v[:, 0] = 0.5; v[:, 2] = -2.0
    F = I3.clone()
    pen, touch, xt0 = [], None, None
    for i in range(40):
        x, v, F, _, st = XP.advance_frame(sol, x, v, F, m, Vp, h, gz, dt)
        pen.append(float(x[:, 2].min()))
        if touch is None and pen[-1] < 1e-3:
            touch = i; xt0 = float((m * x[:, 0]).sum() / m.sum())
    slide = float((m * x[:, 0]).sum() / m.sum()) - xt0 if xt0 is not None else float("nan")
    vend = float(v.norm(dim=-1).mean())
    detm = float(torch.linalg.det(F).min())
    chk(f"바닥 N={N} K={K}", min(pen) >= -0.02 * h and touch is not None,
        f"입자 최저 높이 최소 {min(pen):.3e} (허용 −{0.02 * h:.1e}), 닿은 프레임 {touch}, 닿은 뒤 질량중심 x 이동 {slide:.4e} m (닿기 전 접선 속도 0.5 m/s), "
        f"끝 |v| 평균 {vend:.3e}, det F 최소 {detm:.4f}")

print("XPBD_TEST " + ("PASS" if all(OK) else "FAIL"), flush=True)
