"""--interp bspline / nodeavg 단위 시험 (CPU float64, 빌드 없음): deform_model.node_feats + advance 경로 그대로

  1) 선형장 재현: 노드 변위가 아핀(B·X + c)이면 입자 변위 = B·x + c, 한 스텝 야코비안 = I + B (세 보간 모두)
  2) bspline: 27 이웃이 모두 활성 노드에 있고 가중치 합 1
  3) bspline: 야코비안 = 입자 변위의 위치 미분 (autograd, 가중치를 위치로 다시 계산)
     nodeavg: 사면체별 grad u 가 barycentric 변위의 위치 미분 (autograd)과 같다
  4) 두 보간의 야코비안이 노드 변수에 대해 맞게 미분된다 (gradcheck)
  5) 셀 경계(격자면·Kuhn 사면체 내부면)를 사이에 둔 두 점의 야코비안 차: bary 는 O(1) 로 튀고 bspline·nodeavg 는 O(δ)
  6) advance 의 jvp(속도, --v_from_dt) 가 중심차분과 같고, 역전파 기울기가 유한하다
"""
import math
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch

from anchorflow import deform_model as DM
from anchorflow import simplex as SX

torch.manual_seed(0)
dt = torch.float64
H = 0.056569
ok = True


def chk(name, cond, info=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'통과' if cond else '실패'}] {name} {info}", flush=True)


def mk_a(interp, v_from_dt=False, net_inertial=False):
    return types.SimpleNamespace(
        hex_jac=False, agg="bary", rel_pe=0, bary_pe=0, interp=interp, n_nodes=32, hz_ratio=SX.HZ_CUBE, node_h=H,
        net_inertial=net_inertial, bc_ipg=False, gregory=False, bc_stick=0.0, bc_softplus=0.0, bc_soft=False,
        bc_mode="project", bc_level="node", v_from_dt=v_from_dt, dt_cond=v_from_dt)


cfg = {"boundary_conditions": [], "grid_lim": 2.0, "n_grid": 100, "g": [0.0, 0.0, 0.0]}
# 덩어리 하나 + 셀 경계 양쪽 점 쌍
N0 = 4000
X = 0.7 + 0.6 * torch.rand(N0, 3, dtype=dt)
lo0, lat0, nn0 = SX.grid_for_nodes(X, 32, hz_ratio=SX.HZ_CUBE, h_fix=H)
DEL = 1e-7
yb = ((X[:500] - lo0) @ lat0.Ai.T)
pairs = []
for ax in range(3):                                   # 격자면: y_ax = 정수 ± δ
    y = yb.clone(); y[:, ax] = torch.round(y[:, ax])
    pairs.append((y.clone(), ax, "격자면"))
y = yb.clone(); fr = y - y.floor()                   # Kuhn 내부면: f_0 = f_1 ± δ
y[:, 1] = y[:, 1].floor() + fr[:, 0]
pairs.append((y.clone(), None, "사면체 내부면"))
P_m, P_p, lab = [], [], []
for y, ax, nm in pairs:
    e = torch.zeros(3, dtype=dt)
    if ax is None:
        e[0] = DEL; e[1] = -DEL
    else:
        e[ax] = DEL
    P_m.append((y - e) @ lat0.A.T + lo0); P_p.append((y + e) @ lat0.A.T + lo0); lab.append(nm)
nP = P_m[0].shape[0]
x = torch.cat([X] + P_m + P_p)
N = x.shape[0]
mass = torch.ones(N, dtype=dt)
v = torch.zeros(N, 3, dtype=dt)


def run(interp, dp_fn, v_from_dt=False, inertial=None, h=0.04):
    a = mk_a(interp, v_from_dt, inertial is not None)
    ens = DM.node_feats(a, cfg, x, mass, x, v, 1.0)
    dp = dp_fn(ens[1])
    x2, vn, _dpn, Jf, _c, _act = DM.advance(None, a, [ens], x, h, cfg, dp_fast=lambda M, _d: (dp,),
                                            inertial=([inertial(ens)] if inertial is not None else None))
    return ens, dp, x2, vn, Jf


I3 = torch.eye(3, dtype=dt)
B = 0.05 * torch.randn(3, 3, dtype=dt); c = 0.01 * torch.randn(3, dtype=dt)
for mode in ("bary", "bspline", "nodeavg"):
    ens, dp, x2, _, Jf = run(mode, lambda npos: npos @ B.T + c)
    e_u = float((x2 - x - (x @ B.T + c)).abs().max()); e_J = float((Jf - I3 - B).abs().max())
    chk(f"1) 선형장 재현 {mode}", e_u < 1e-12 and e_J < 1e-11, f"변위 오차 {e_u:.1e}, 야코비안 오차 {e_J:.1e}, 노드 {ens[1].shape[0]}")

# 2) bspline 활성 노드
a = mk_a("bspline")
ens = DM.node_feats(a, cfg, x, mass, x, v, 1.0)
_, lo, (lo_, lat, nn), rows, lam, uniq, _, aux = ens[0], ens[1], ens[2], ens[3], ens[4], ens[5], ens[6], ens[7]
f27, w27, dw27 = SX.bspline27(x, lo_, lat, nn)
chk("2) bspline 27 이웃 전부 활성", bool((uniq[rows] == f27).all()) and rows.shape[1] == 27,
    f"활성 노드 {uniq.numel()} (사면체 꼭짓점만이면 {SX.active_nodes(SX.locate(x, lo_, lat, nn)[0])[1].numel()})")
chk("2) bspline 가중치 합 1", float((lam.sum(1) - 1).abs().max()) < 1e-13)

# 3) 야코비안 = 위치 미분 (autograd)
dpr = 1e-3 * torch.randn(uniq.numel(), 3, dtype=dt)
Jb = SX.bspline_jac_pre(rows, aux.bsp, lat, dpr)
sel = torch.randperm(N0)[:64]


def u_bsp(xq):
    fl, w, _ = SX.bspline27(xq.unsqueeze(0), lo_, lat, nn)
    r = torch.searchsorted(uniq, fl)
    return (w.unsqueeze(-1) * dpr[r]).sum(1)[0]


ea = max(float((torch.autograd.functional.jacobian(u_bsp, x[i]) - Jb[i]).abs().max()) for i in sel.tolist())
chk("3) bspline 야코비안 = autograd(변위)", ea < 1e-12, f"최대 차 {ea:.1e} (|J| ~ {float(Jb.abs().max()):.1e})")
a2 = mk_a("nodeavg")
ens2 = DM.node_feats(a2, cfg, x, mass, x, v, 1.0)
lo2, lat2, nn2 = ens2[2]; rows2, lam2, uniq2, aux2 = ens2[3], ens2[4], ens2[5], ens2[7]
dpr2 = 1e-3 * torch.randn(uniq2.numel(), 3, dtype=dt)
tets, trank, tw, ws = aux2.nav
_, Gp = SX.g2p_jac_pre(rows2, lam2, aux2, lat2, dpr2)


def u_bar(xq):
    idx, lm, _ = SX.locate(xq.unsqueeze(0), lo2, lat2, nn2)
    return (lm.unsqueeze(-1) * dpr2[torch.searchsorted(uniq2, idx)]).sum(1)[0]


ea2 = max(float((torch.autograd.functional.jacobian(u_bar, x[i]) - Gp[i]).abs().max()) for i in sel.tolist())
chk("3) nodeavg 사면체 grad u = autograd(barycentric 변위)", ea2 < 1e-12, f"최대 차 {ea2:.1e}")
_, inv = torch.unique(rows2, dim=0, return_inverse=True)
chk("3) nodeavg 사면체 집합 = 입자가 든 사면체", bool((tets[inv] == rows2).all()), f"사면체 {tets.shape[0]}")

# 4) gradcheck (노드 변수)
sub = torch.randperm(N)[:40]
dq = (1e-3 * torch.randn(uniq.numel(), 3, dtype=dt)).requires_grad_(True)
chk("4) gradcheck bspline 야코비안", torch.autograd.gradcheck(
    lambda q: SX.bspline_jac_pre(rows[sub], aux.bsp[sub], lat, q), (dq,), eps=1e-6, atol=1e-8))
dq2 = (1e-3 * torch.randn(uniq2.numel(), 3, dtype=dt)).requires_grad_(True)
chk("4) gradcheck nodeavg 야코비안", torch.autograd.gradcheck(
    lambda q: SX.nodeavg_jac_pre(rows2[sub], lam2[sub], aux2.nav, lat2, q), (dq2,), eps=1e-6, atol=1e-8))

# 5) 셀 경계 연속성 (노드 변위 무작위 = 매끄럽지 않은 장에서도)
for mode in ("bary", "bspline", "nodeavg"):
    torch.manual_seed(1)
    _, dp, _, _, Jf = run(mode, lambda npos: 1e-3 * torch.randn(npos.shape[0], 3, dtype=dt))
    msg = []
    worst = 0.0
    for k, nm in enumerate(lab):
        jm = Jf[N0 + k * nP:N0 + (k + 1) * nP]
        jp = Jf[N0 + (len(lab) + k) * nP:N0 + (len(lab) + k + 1) * nP]
        jmp = float((jm - jp).abs().amax((1, 2)).max())
        msg.append(f"{nm}{'' if nm != '격자면' else k} {jmp:.1e}")
        worst = max(worst, jmp)
    scale = float((Jf - I3).abs().max())
    if mode == "bary":
        chk("5) bary 는 경계에서 튄다 (대조)", worst > 1e-3 * scale, f"|ΔJ| 최대 {worst:.1e} (|∇u| 최대 {scale:.1e}): " + ", ".join(msg))
    else:
        chk(f"5) {mode} 경계에서 연속", worst < 1e-5 * scale,
            f"|ΔJ| 최대 {worst:.1e} (|∇u| 최대 {scale:.1e}): " + ", ".join(msg))

# 6) jvp 속도 = 중심차분, 역전파 유한
for mode in ("bspline", "nodeavg"):
    torch.manual_seed(2)
    dpf = 1e-3 * torch.randn(20000, 3, dtype=dt)
    vn_g = 0.3 * torch.randn(20000, 3, dtype=dt)
    gq = torch.tensor([0.0, 0.0, -9.8], dtype=dt)
    def inert(e):
        return (vn_g[:e[1].shape[0]], gq)
    _, _, x2, vj, _ = run(mode, lambda npos: dpf[:npos.shape[0]], True, inert, 0.04)
    e = 1e-6
    _, _, xa, _, _ = run(mode, lambda npos: dpf[:npos.shape[0]], False, inert, 0.04 + e)
    _, _, xb, _, _ = run(mode, lambda npos: dpf[:npos.shape[0]], False, inert, 0.04 - e)
    fd = (xa - xb) / (2 * e)
    ev = float((vj - fd).abs().max())
    chk(f"6) {mode} jvp 속도 = 중심차분", ev < 1e-6, f"최대 차 {ev:.1e} (|v| 최대 {float(vj.abs().max()):.2f})")
    a6 = mk_a(mode)
    ens6 = DM.node_feats(a6, cfg, x, mass, x, v, 1.0)
    q6 = (1e-3 * torch.randn(ens6[1].shape[0], 3, dtype=dt)).requires_grad_(True)
    x26, _, _, J6, _, _ = DM.advance(None, a6, [ens6], x, 0.04, cfg, dp_fast=lambda M, _d: (q6,))
    L = (x26 ** 2).sum() + (torch.linalg.det(J6) - 1).pow(2).sum()
    L.backward()
    chk(f"6) {mode} 역전파 기울기 유한", bool(torch.isfinite(q6.grad).all()) and float(q6.grad.abs().max()) > 0,
        f"|g| 최대 {float(q6.grad.abs().max()):.2e}")

print("INTERP_SMOOTH_TEST " + ("ALL_PASS" if ok else "FAIL"), flush=True)
sys.exit(0 if ok else 1)
