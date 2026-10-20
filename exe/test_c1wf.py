"""--interp c1wf (Worsey–Farin C1 3 차) 단위 시험 + 3D 준정적 시연 (torch float64; CPU 또는 GPU)

  1) 아핀 재현: 노드 자유도를 아핀장에서 채우면 입자 변위 = B·x + c, 한 스텝 야코비안 = I + B (deform_model.advance 경로)
  2) 3 차 재현: 3 차 다항식장에서 채우면 값·기울기 정확
  3) C1: 무작위 자유도에서 격자면·Kuhn 내부면 양쪽 ±δ 의 야코비안 차가 O(δ) (bary 는 O(1))
  4) det 보장서: 입자 det(I+∇u) = 자기 조각의 6 차 Bernstein 다항식 값 (정확), 계수 최소 ≤ 입자 det,
     "조각 계수 모두 > 0 ⇒ 그 조각 입자 det > 0" 위반 0
  5) autograd: jac_pre·det_coefs·장벽의 노드 변수 기울기 (gradcheck)
  6) 입자는 자기 조각 안 (barycentric ≥ 0), 바닥 평면: 제어점 최저 ≤ 조밀 표본 최저 (볼록 껍질)
  7) 준정적: 정육면체를 위에서 짓눌러 (StVK, 뒤집힘에 저항 없음) Bernstein 로그 장벽 유무·bary 비교
  8) 바닥 시연: 외팔보가 중력으로 처져 바닥 평면에 닿을 때 입자만 장벽 vs 제어점 장벽
결과 JSON: $OUT/ct3d_results.json, 그림 $OUT/ct3d_*.png. 마지막 줄 CT3D_TEST_PASS / CT3D_TEST_FAIL
"""
import json
import math
import os
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import torch

from anchorflow import deform_model as DM
from anchorflow import simplex as SX
from anchorflow import wf as WF
from anchorflow.feas_lbfgs import FeasLBFGS

OUT = os.environ.get("CT3D_OUT", "/tmp/ct3d_out")
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() and os.environ.get("CT3D_CPU") is None else "cpu"
dt = torch.float64
torch.manual_seed(0)
H = 0.056569
ok = True
RES = {"device": dev}


def chk(name, cond, info=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'통과' if cond else '실패'}] {name} {info}", flush=True)
    RES.setdefault("checks", []).append(dict(name=name, ok=bool(cond), info=info))


def mk_a(interp):
    return types.SimpleNamespace(
        hex_jac=False, agg="bary", rel_pe=0, bary_pe=0, interp=interp, n_nodes=32, hz_ratio=SX.HZ_CUBE, node_h=H,
        net_inertial=False, bc_ipg=False, gregory=False, bc_stick=0.0, bc_softplus=0.0, bc_soft=False,
        bc_mode="project", bc_level="node", v_from_dt=False, dt_cond=False)


cfg = {"boundary_conditions": [], "grid_lim": 2.0, "n_grid": 100, "g": [0.0, 0.0, 0.0]}
N0 = 6000
X = (0.7 + 0.4 * torch.rand(N0, 3, dtype=dt)).to(dev)
lo0, lat0, nn0 = SX.grid_for_nodes(X, 32, hz_ratio=SX.HZ_CUBE, h_fix=H)
DEL = 1e-7
yb = ((X[:500] - lo0) @ lat0.Ai.T)
pairs = []
for ax in range(3):
    y = yb.clone(); y[:, ax] = torch.round(y[:, ax]); pairs.append((y.clone(), ax, "격자면"))
y = yb.clone(); fr = y - y.floor(); y[:, 1] = y[:, 1].floor() + fr[:, 0]; pairs.append((y.clone(), None, "Kuhn 내부면"))
P_m, P_p, lab = [], [], []
for y, ax, nm in pairs:
    e = torch.zeros(3, dtype=dt, device=dev)
    if ax is None:
        e[0] = DEL; e[1] = -DEL
    else:
        e[ax] = DEL
    P_m.append((y - e) @ lat0.A.T + lo0); P_p.append((y + e) @ lat0.A.T + lo0); lab.append(nm)
nP = P_m[0].shape[0]
x = torch.cat([X] + P_m + P_p)
N = x.shape[0]
mass = torch.ones(N, dtype=dt, device=dev)
v = torch.zeros(N, 3, dtype=dt, device=dev)


def dofs_from_field(npos, lat, f, J):
    """노드 자유도 [M,54]: 값, ∂u/∂y = J A, 변 가로 미분 = J(mid) A t (변의 아래쪽 끝 노드에)."""
    M = npos.shape[0]
    A = lat.A.to(dt)
    out = torch.zeros(M, WF.C1_NOUT, dtype=dt, device=npos.device)
    out[:, :3] = f(npos)
    Gy = J(npos) @ A                                                  # [M,3c,3d]
    out[:, 3:12] = Gy.reshape(M, 9)
    TR = torch.as_tensor(WF._TRANS, dtype=dt, device=npos.device)    # [7,2,3]
    CL = torch.as_tensor(WF.CLASSES, dtype=dt, device=npos.device)
    for k in range(7):
        mid = npos + 0.5 * (CL[k] @ A.T)
        Jm = J(mid) @ A
        for s in range(2):
            out[:, 12 + (k * 2 + s) * 3:12 + (k * 2 + s) * 3 + 3] = Jm @ TR[k, s]
    return out


def run(interp, dp_fn, xx=None, h=0.04):
    xx = x if xx is None else xx
    a = mk_a(interp)
    ens = DM.node_feats(a, cfg, xx, torch.ones(xx.shape[0], dtype=dt, device=dev), xx, torch.zeros_like(xx), 1.0)
    dp = dp_fn(ens[1], ens[2][1])
    x2, vn, _dpn, Jf, _c, _act = DM.advance(None, a, [ens], xx, h, cfg, dp_fast=lambda M, _d: (dp,))
    return ens, dp, x2, Jf


I3 = torch.eye(3, dtype=dt, device=dev)
# 1) 아핀
B = 0.05 * torch.randn(3, 3, dtype=dt, device=dev); c = 0.01 * torch.randn(3, dtype=dt, device=dev)
ens, dp, x2, Jf = run("c1wf", lambda npos, lat: dofs_from_field(npos, lat, lambda p: p @ B.T + c, lambda p: B.expand(p.shape[0], 3, 3)))
e_u = float((x2 - x - (x @ B.T + c)).abs().max()); e_J = float((Jf - I3 - B).abs().max())
chk("1) 아핀 재현 (advance 경로)", e_u < 1e-12 and e_J < 1e-11, f"변위 오차 {e_u:.1e}, 야코비안 오차 {e_J:.1e}, 노드 {ens[1].shape[0]}")
RES["affine"] = dict(u_err=e_u, J_err=e_J)
# 2) 3 차
CF = 0.3 * torch.randn(3, 20, dtype=dt, device=dev)
MON = [(i, j, k) for i in range(4) for j in range(4) for k in range(4) if i + j + k <= 3]


def fc(p):
    q = p - 0.9
    return torch.stack([q[:, 0] ** m[0] * q[:, 1] ** m[1] * q[:, 2] ** m[2] for m in MON], 1) @ CF.T


def Jc(p):
    q = p - 0.9
    cols = []
    for d in range(3):
        cols.append(torch.stack([(m[d] * q[:, d] ** max(m[d] - 1, 0) if m[d] > 0 else torch.zeros_like(q[:, 0]))
                                 * math.prod(q[:, e] ** m[e] for e in range(3) if e != d) for m in MON], 1) @ CF.T)
    return torch.stack(cols, -1)


ens, dp, x2, Jf = run("c1wf", lambda npos, lat: dofs_from_field(npos, lat, fc, Jc))
e_u = float((x2 - x - fc(x)).abs().max()); e_J = float((Jf - I3 - Jc(x)).abs().max())
chk("2) 3 차 다항식 재현", e_u < 1e-11 and e_J < 1e-9, f"변위 오차 {e_u:.1e}, 야코비안 오차 {e_J:.1e} (|u| ~ {float(fc(x).abs().max()):.2f})")
RES["cubic"] = dict(u_err=e_u, J_err=e_J)
# 3) C1
RES["jump"] = {}
for mode in ("bary", "c1wf"):
    torch.manual_seed(1)
    nch = 3 if mode == "bary" else WF.C1_NOUT
    _, dp, _, Jf = run(mode, lambda npos, lat: 1e-3 * torch.randn(npos.shape[0], nch, dtype=dt, device=dev))
    worst, msg = 0.0, []
    for k, nm in enumerate(lab):
        jm = Jf[N0 + k * nP:N0 + (k + 1) * nP]; jp = Jf[N0 + (len(lab) + k) * nP:N0 + (len(lab) + k + 1) * nP]
        jmp = float((jm - jp).abs().amax((1, 2)).max()); msg.append(f"{nm}{k if nm == '격자면' else ''} {jmp:.1e}"); worst = max(worst, jmp)
    scale = float((Jf - I3).abs().max())
    RES["jump"][mode] = dict(max=worst, grad_scale=scale)
    if mode == "bary":
        chk("3) bary 는 면에서 튄다 (대조)", worst > 1e-3 * scale, f"|ΔJ| 최대 {worst:.1e} (|∇u| {scale:.1e}): " + ", ".join(msg))
    else:
        chk("3) c1wf 면에서 연속 (C1)", worst < 1e-5 * scale, f"|ΔJ| 최대 {worst:.1e} (|∇u| {scale:.1e}, δ={DEL:g}): " + ", ".join(msg))
# 4) det 보장서 (큰 무작위 변형 여러 규모)
RES["det"] = []
for amp in (0.01, 0.03, 0.06, 0.1):
    torch.manual_seed(3)
    ens, dp, x2, Jf = run("c1wf", lambda npos, lat: amp * torch.randn(npos.shape[0], WF.C1_NOUT, dtype=dt, device=dev))
    aux = ens[7]; ws, wt = aux.wf, aux.wft
    cco = WF.det_coefs(dp, wt, ens[2][1])                                # [T,12,84]
    key = torch.cat([ens[3], ws["ty"].unsqueeze(1)], 1)
    _, inv = torch.unique(key, dim=0, return_inverse=True)
    cp = cco[inv, ws["sb"]]                                              # [N,84]
    T_ = WF.tables(dev, dt)
    poly = (WF._bern_t(ws["L"], T_["I6"], torch.as_tensor([WF.mcoef(m) for m in WF.I6], dtype=dt, device=dev)) * cp).sum(1)
    dJ = torch.linalg.det(Jf)
    e_poly = float((poly - dJ).abs().max())
    cmin = cp.min(1).values
    viol = int(((cmin > 0) & (dJ <= 0)).sum()); bound = int((dJ < cmin - 1e-10).sum())
    pos = cco.min(-1).values > 0
    r = dict(amp=amp, subtets=int(cco.shape[0] * 12), subtets_coef_pos=int(pos.sum()), particles=int(N),
             particle_det_neg=int((dJ <= 0).sum()), viol=viol, bound_viol=bound, poly_err=e_poly,
             coef_min=float(cco.min()), det_min=float(dJ.min()),
             ratio_med=float((cmin[cmin > 0] / dJ[cmin > 0]).median()) if bool((cmin > 0).any()) else None)
    RES["det"].append(r)
    chk(f"4) det 보장서 amp={amp}", viol == 0 and bound == 0 and e_poly < 1e-9,
        f"입자 det = 조각 Bernstein 다항식 (오차 {e_poly:.1e}), 계수>0 인데 det≤0 {viol}, 계수최소>det {bound}, "
        f"계수>0 조각 {r['subtets_coef_pos']}/{r['subtets']}, 입자 det≤0 {r['particle_det_neg']}")
    if amp == 0.03:
        np.savez(os.path.join(OUT, "ct3d_det_scatter.npz"), cmin=cmin.cpu().numpy(), det=dJ.cpu().numpy())
    # 6) 자기 조각 안
    if amp == 0.06:
        Lmin = float(ws["L"].min()); Lsum = float((ws["L"].sum(1) - 1).abs().max())
        chk("6) 입자는 자기 조각 안 (barycentric ≥ 0)", Lmin > -1e-12 and Lsum < 1e-12, f"최소 {Lmin:.1e}, 합-1 {Lsum:.1e}")
# 5) autograd
a5 = mk_a("c1wf")
ens5 = DM.node_feats(a5, cfg, x, mass, x, v, 1.0)
sub = torch.randperm(N, device=dev)[:8]
M5 = ens5[1].shape[0]
base5 = 1e-3 * torch.randn(M5, WF.C1_NOUT, dtype=dt, device=dev)


def embed(nodes):
    def f(qs):
        return base5.index_copy(0, nodes, qs)
    return f


nd1 = torch.unique(ens5[3][sub])                                   # 이 입자들이 쓰는 노드만 변수로 (gradcheck 비용)
ws5 = {k: (vv[sub] if torch.is_tensor(vv) and vv.shape[0] == N else vv) for k, vv in ens5[7].wf.items()}
e1 = embed(nd1)
g1 = torch.autograd.gradcheck(lambda q: WF.jac_pre(ens5[3][sub], ws5, ens5[2][1], e1(q)), (base5[nd1].clone().requires_grad_(True),), eps=1e-6, atol=1e-8)
wt5 = ens5[7].wft
wts = dict(rows=wt5["rows"][:2], ty=wt5["ty"][:2], svol=wt5["svol"][:2])
nd2 = torch.unique(wts["rows"]); e2 = embed(nd2)
g2 = torch.autograd.gradcheck(lambda q: WF.det_coefs(e2(q), wts, ens5[2][1]), (base5[nd2].clone().requires_grad_(True),), eps=1e-6, atol=1e-8)
g3 = torch.autograd.gradcheck(lambda q: WF.barrier_energy(WF.det_coefs(e2(q), wts, ens5[2][1]), wts["svol"], 1.0, 1.5),
                              (base5[nd2].clone().requires_grad_(True),), eps=1e-6, atol=1e-7)
chk("5) gradcheck jac_pre / det_coefs / 로그 장벽", g1 and g2 and g3)


# 6b) 바닥: 제어점 볼록 껍질
def ctrl_points(dpn, wt, lat, lo):
    """조각마다 사상 x+u 의 Bézier 제어점 [T,12,20,3] (월드)."""
    T_ = WF.tables(dpn.device, dpn.dtype)
    D = WF.gather_dofs(dpn, wt["rows"], wt["ty"])
    uo = torch.einsum("tskn,tnc->tskc", T_["ord"][wt["ty"]], D)
    subs = torch.as_tensor(np.stack([t["sub"] for t in WF.TYPES]), dtype=dpn.dtype, device=dpn.device)[wt["ty"]]  # [T,12,4,3] 셀 국소 y
    I3i = T_["I3"].to(dpn.dtype)                                                  # [20,4]
    yl = torch.einsum("kj,tsjd->tskd", I3i, subs) / 3.0
    return wt["base"].unsqueeze(1).unsqueeze(1) + yl @ lat.A.T + uo


def with_base(wt, uniq, lo, lat, nn):
    nnl = [int(nn[k]) for k in range(3)]
    fl = uniq[wt["rows"][:, 0]]
    ijk = torch.stack([fl // (nnl[1] * nnl[2]), (fl // nnl[2]) % nnl[1], fl % nnl[2]], -1).to(lat.A.dtype)
    wt = dict(wt); wt["base"] = ijk @ lat.A.T + lo                              # v0 의 월드 위치
    return wt


torch.manual_seed(5)
ens6 = DM.node_feats(a5, cfg, x, mass, x, v, 1.0)
lo6, lat6, nn6 = ens6[2]
dp6 = dofs_from_field(ens6[1], lat6, lambda p: torch.stack([0.1 * torch.sin(4 * p[:, 1]), 0 * p[:, 0], -0.2 * (p[:, 0] - 0.7) ** 2], 1),
                      lambda p: torch.stack([torch.stack([0 * p[:, 0], 0.4 * torch.cos(4 * p[:, 1]), 0 * p[:, 0]], -1),
                                             torch.zeros(p.shape[0], 3, dtype=dt, device=dev),
                                             torch.stack([-0.4 * (p[:, 0] - 0.7), 0 * p[:, 0], 0 * p[:, 0]], -1)], 1))
dp6 = dp6 + 0.02 * torch.randn_like(dp6)
wt6 = with_base(ens6[7].wft, ens6[5], lo6, lat6, nn6)
cp6 = ctrl_points(dp6, wt6, lat6, lo6)
# 조밀 표본: 점유 사면체마다 무작위 점
Ls = torch.distributions.Dirichlet(torch.ones(4, dtype=dt, device=dev)).sample((wt6["rows"].shape[0], 12, 200))
Bs = WF._bern_t(Ls.reshape(-1, 4), WF.tables(dev, dt)["I3"], WF.tables(dev, dt)["mc3"]).reshape(Ls.shape[:3] + (20,))
smp = torch.einsum("tspk,tskc->tspc", Bs, cp6)
ctr_min = cp6[..., 2].amin((2,)); smp_min = smp[..., 2].amin((2,))
# 표본이 실제 사상과 같은지 (제어점 다항식 = advance 경로의 x+u)
xs = smp.reshape(-1, 3)[:2000]
fl_ok = bool((smp_min >= ctr_min - 1e-12).all())
# 제어점 다항식이 실제 x+u 와 같은지: 입자에서 비교
ws6 = ens6[7].wf
key6 = torch.cat([ens6[3], ws6["ty"].unsqueeze(1)], 1)
_, inv6 = torch.unique(key6, dim=0, return_inverse=True)
xp_ctrl = torch.einsum("nk,nkc->nc", WF._bern_t(ws6["L"], WF.tables(dev, dt)["I3"], WF.tables(dev, dt)["mc3"]), cp6[inv6, ws6["sb"]])
u6, _ = WF.jac_pre(ens6[3], ws6, lat6, dp6)
e_cp = float((xp_ctrl - (x + u6)).abs().max())
chk("6) 바닥 평면: 조각마다 제어점 최저 z ≤ 조밀 표본 최저 z (볼록 껍질), 제어점 다항식 = x+u", fl_ok and e_cp < 1e-12,
    f"제어점 최저 {float(ctr_min.min()):.4f}, 표본 최저 {float(smp_min.min()):.4f}, 입자에서 제어점식 오차 {e_cp:.1e}")
RES["floor_hull"] = dict(ctrl_min=float(ctr_min.min()), sample_min=float(smp_min.min()), ctrl_poly_err=e_cp)


# ------------------------------------------------------------------ 7) 준정적 짓누르기
def stvk(Fp, mu, lam):
    E_ = 0.5 * (Fp.transpose(-1, -2) @ Fp - torch.eye(3, dtype=Fp.dtype, device=Fp.device))
    return mu * (E_ * E_).sum((-1, -2)) + 0.5 * lam * E_.diagonal(dim1=-2, dim2=-1).sum(-1) ** 2


def quasi_static(interp, barrier, push=(0.0, 0.0, -0.6), steps=12, iters=150, nside=24, h=1.0 / 6, jhat=0.3, kap=1.0):
    g = torch.linspace(0.0, 1.0, nside, dtype=dt, device=dev)
    P = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) * 0.999 + 5e-4
    lo, lat, nn = SX.grid_for_nodes(P, 8, hz_ratio=SX.HZ_CUBE, h_fix=h)
    idx, lam_, aux = SX.locate(P, lo, lat, nn)
    rows, uniq = SX.active_nodes(idx)
    npos = SX.node_pos(lo, lat, nn, uniq)
    M = npos.shape[0]
    zlo, zhi = float(P[:, 2].min()), float(P[:, 2].max())
    # 바닥·천장 근처 노드 (입자 덩어리 위아래 판에 박힌 노드) -- 판 두께 h
    bot = npos[:, 2] < zlo + 0.5 * h
    top = npos[:, 2] > zhi - 0.5 * h
    fixed = bot | top
    vol = torch.full((P.shape[0],), 1.0 / P.shape[0], dtype=dt, device=dev)
    mu, lmb = 0.385, 0.577
    nch = WF.C1_NOUT if interp == "c1wf" else 3
    if interp == "c1wf":
        aux = SX.Aux(aux); ws = WF.particle_struct(P, lo, lat, nn, aux, rows); wt = WF.tet_struct(rows, ws["ty"], lat)
    else:
        tets, tinv = torch.unique(rows, dim=0, return_inverse=True)
        trep = torch.zeros(tets.shape[0], dtype=torch.long, device=dev).scatter_(0, tinv, torch.arange(rows.shape[0], device=dev))
        tvol = abs(float(torch.linalg.det(lat.A))) / 6.0
    # 위·아래 판 노드는 **값(변위 3 성분)만** 고정 -- 기울기·변 미분은 자유 (2D 의 "값 + 접선 미분" 고정에 해당하는 최소 구속)
    fm = torch.ones(M, nch, dtype=torch.bool, device=dev)
    fm[fixed, :3] = False
    pv = torch.as_tensor(push, dtype=dt, device=dev)
    var = torch.zeros(M, nch, dtype=dt, device=dev).requires_grad_(True)
    hist = []
    A_ = lat.A.to(dt)
    # 아핀 증분장 Δu(x) = frac(z)·Δ, frac = (z-zlo)/(zhi-zlo) 의 자유도 (값·∂/∂y·변 가로 미분)
    frac = ((npos[:, 2] - zlo) / (zhi - zlo))
    dzdy = A_[2] / (zhi - zlo)                                            # ∂frac/∂y [3]
    TR = torch.as_tensor(WF._TRANS, dtype=dt, device=dev)

    def inc_field(delta):
        f = torch.zeros(M, nch, dtype=dt, device=dev)
        f[:, :3] = frac.unsqueeze(-1) * delta
        if nch == WF.C1_NOUT:
            f[:, 3:12] = (delta.unsqueeze(-1) * dzdy.unsqueeze(0)).reshape(9)
            for k in range(7):
                for s_ in range(2):
                    f[:, 12 + (k * 2 + s_) * 3:12 + (k * 2 + s_) * 3 + 3] = delta * float(dzdy @ TR[k, s_])
        return f

    def full(var, s):
        presc = torch.zeros(M, nch, dtype=dt, device=dev)
        presc[top, :3] = s * pv
        return torch.where(fm, var, presc)

    def energy(var, s, parts=False):
        d = full(var, s)
        if interp == "c1wf":
            u, G = WF.jac_pre(rows, ws, lat, d)
        else:
            u, G = SX.g2p_jac_pre(rows, lam_, aux, lat, d)
        Fp = torch.eye(3, dtype=dt, device=dev) + G
        Ee = (vol * stvk(Fp, mu, lmb)).sum()
        Eb = torch.zeros((), dtype=dt, device=dev)
        if interp == "c1wf":
            cc = WF.det_coefs(d, wt, lat)
            if barrier:
                Eb = WF.barrier_energy(cc, wt["svol"], kap * mu, jhat)
            cmin = cc.min()
        else:
            dT = torch.linalg.det(Fp[trep])
            if barrier:
                if bool((dT <= 0).any()):
                    Eb = torch.full((), float("inf"), dtype=dt, device=dev)
                else:
                    dc = torch.minimum(dT, torch.full_like(dT, jhat))
                    Eb = kap * mu * tvol * (-((dc - jhat) ** 2) * torch.log(dc / jhat)).sum()
            cmin = dT.min()
        if parts:
            return Ee, Eb, cmin, torch.linalg.det(Fp), u, G
        return Ee + Eb

    t0 = time.time()
    s_done, ds = 0.0, 1.0 / steps
    while s_done < 1.0 - 1e-12:
        dl = min(ds, 1.0 - s_done); s = s_done + dl
        with torch.no_grad():
            trial = var + inc_field(dl * pv)                               # 아핀 증분으로 초기 추정 (경계와 일치)
            E0 = energy(trial, s)
        if barrier and not math.isfinite(float(E0)):
            ds *= 0.5                                                      # 실현 불가면 하중 증분을 반으로
            hist.append(dict(s=s, halve=True)); print(f"    [준정적] s={s:.4f} 실현 불가 -> 증분 {ds:.2e}", flush=True)
            if ds < 1e-4:
                raise SystemExit("[준정적] 하중 증분이 너무 작아졌다")
            continue
        with torch.no_grad():
            var.copy_(trial)
        opt = FeasLBFGS([var], max_iter=iters, history_size=30) if barrier else torch.optim.LBFGS(
            [var], lr=1.0, max_iter=iters, history_size=30, tolerance_grad=0.0, tolerance_change=0.0, line_search_fn="strong_wolfe")

        def clo():
            opt.zero_grad(set_to_none=True)
            E = energy(var, s)
            if torch.isfinite(E):
                E.backward()
            return E
        opt.step(clo)
        s_done = s
        ds = min(2.0 * ds, 1.0 / steps)                                    # 성공하면 증분을 되돌려 키운다
        Ee, Eb, cmin, dJ, u, G = energy(var.detach(), s, parts=True)
        hist.append(dict(s=s, E_el=float(Ee), E_bar=float(Eb), coef_min=float(cmin), particle_det_min=float(dJ.min()),
                         particle_det_neg=int((dJ <= 0).sum()), stats=getattr(opt, "stats", None)))
        print(f"    [준정적 {interp} {'장벽' if barrier else '없음'}] s={s:.4f} 탄성 {float(Ee):.3e} 장벽 {float(Eb):.2e} 계수 최소 {float(cmin):.3e} "
              f"입자 det 최소 {float(dJ.min()):.3e} ({time.time() - t0:.0f} 초) {getattr(opt, 'stats', '')}", flush=True)
    d = full(var.detach(), 1.0)
    res = dict(interp=interp, barrier=barrier, sec=round(time.time() - t0, 1), hist=hist, M=M, particles=int(P.shape[0]),
               free_dofs=int(fm.sum()))
    Ee, Eb, cmin, dJ, u, G = energy(var.detach(), 1.0, parts=True)
    res.update(E_el=float(Ee), coef_min=float(cmin), particle_det_min=float(dJ.min()), particle_det_neg=int((dJ <= 0).sum()))
    if interp == "c1wf":
        # 조밀 표본 det (점유 사면체마다 조각당 60 점)
        cc = WF.det_coefs(d, wt, lat)
        res["dense_note"] = "조각 Bernstein 계수 최소 = 하한"
        res["subtets_coef_neg"] = int((cc.min(-1).values <= 0).sum())
        res["subtets"] = int(cc.shape[0] * 12)
        Ls_ = torch.distributions.Dirichlet(torch.ones(4, dtype=dt, device=dev)).sample((cc.shape[0], 12, 60))
        I6 = WF.tables(dev, dt)["I6"]; mc6 = torch.as_tensor([WF.mcoef(m) for m in WF.I6], dtype=dt, device=dev)
        Bd = WF._bern_t(Ls_.reshape(-1, 4), I6, mc6).reshape(Ls_.shape[:3] + (84,))
        dd = (Bd * cc.unsqueeze(2)).sum(-1)
        res["dense_det_min"] = float(dd.min()); res["dense_det_neg"] = int((dd <= 0).sum())
    else:
        res["tets"] = int(tets.shape[0]); res["tets_det_neg"] = int((torch.linalg.det(torch.eye(3, dtype=dt, device=dev) + G[trep]) <= 0).sum())
    res["_P"] = P.cpu().numpy(); res["_x"] = (P + u).detach().cpu().numpy(); res["_det"] = dJ.detach().cpu().numpy()
    return res


RES["qs"] = {}
for interp, bar in (("c1wf", True), ("c1wf", False), ("bary", True), ("bary", False)):
    r = quasi_static(interp, bar)
    key = f"{interp}|{'장벽' if bar else '장벽없음'}"
    RES["qs"][key] = {k: vv for k, vv in r.items() if not k.startswith("_")}
    np.savez(os.path.join(OUT, f"ct3d_qs_{interp}_{int(bar)}.npz"), P=r["_P"], x=r["_x"], det=r["_det"])
    print(f"[준정적] {key}: 탄성 {r['E_el']:.4e} 계수/셀det 최소 {r['coef_min']:.3e} 입자 det 최소 {r['particle_det_min']:.3e} "
          f"입자 det≤0 {r['particle_det_neg']} " + (f"조밀 det 최소 {r.get('dense_det_min', float('nan')):.3e} 조밀 det≤0 {r.get('dense_det_neg')}"
                                                    if interp == 'c1wf' else f"셀 det≤0 {r.get('tets_det_neg')}") + f" ({r['sec']} 초)", flush=True)
qb = RES["qs"]["c1wf|장벽"]
chk("7) 준정적 c1wf+장벽: 모든 단계 계수>0, 조밀 표본·입자 det>0", qb["coef_min"] > 0 and qb["dense_det_neg"] == 0 and qb["particle_det_neg"] == 0
    and all(h_.get("coef_min", 1) > 0 for h_ in qb["hist"]) and qb["hist"][-1].get("s", 0) > 1 - 1e-9, f"계수 최소 {qb['coef_min']:.3e}, 조밀 det 최소 {qb['dense_det_min']:.3e}")


# ------------------------------------------------------------------ 8) 바닥 시연 (외팔보)
def cantilever(mode, iters=400, nl=(30, 10, 10), h=1.0 / 8, floor=-0.18, ghat=0.02, gs=(0.5, 1.0, 1.5, 2.0, 3.0)):
    gx = torch.linspace(0, 1.0, nl[0], dtype=dt, device=dev); gy = torch.linspace(0, 0.3, nl[1], dtype=dt, device=dev)
    gz = torch.linspace(0, 0.3, nl[2], dtype=dt, device=dev)
    P = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3) * 0.999 + 5e-4
    lo, lat, nn = SX.grid_for_nodes(P, 8, hz_ratio=SX.HZ_CUBE, h_fix=h)
    idx, lam_, aux = SX.locate(P, lo, lat, nn)
    rows, uniq = SX.active_nodes(idx)
    npos = SX.node_pos(lo, lat, nn, uniq); M = npos.shape[0]
    aux = SX.Aux(aux); ws = WF.particle_struct(P, lo, lat, nn, aux, rows); wt = with_base(WF.tet_struct(rows, ws["ty"], lat), uniq, lo, lat, nn)
    fixed = npos[:, 0] < float(P[:, 0].min()) + 0.5 * h
    free = (~fixed).nonzero(as_tuple=True)[0]
    z = torch.zeros(M, WF.C1_NOUT, dtype=dt, device=dev)
    var = z[free].clone().requires_grad_(True)
    vol = torch.full((P.shape[0],), 0.09 / P.shape[0], dtype=dt, device=dev)
    mu, lmb = 0.385, 0.577

    def energy(var, gg, parts=False):
        d = z.index_copy(0, free, var)
        u, G = WF.jac_pre(rows, ws, lat, d)
        Fp = torch.eye(3, dtype=dt, device=dev) + G
        E = (vol * stvk(Fp, mu, lmb)).sum() + gg * (vol * (P[:, 2] + u[:, 2])).sum()
        cc = WF.det_coefs(d, wt, lat)
        E = E + WF.barrier_energy(cc, wt["svol"], mu, 0.3)
        if mode == "particle":
            hz = P[:, 2] + u[:, 2] - floor; w_ = vol
        else:
            cp = ctrl_points(d, wt, lat, lo); hz = cp[..., 2].reshape(-1) - floor
            w_ = (wt["svol"] / 20.0).unsqueeze(-1).expand(-1, -1, 20).reshape(-1)
        if bool((hz <= 0).any()):
            return torch.full((), float("inf"), dtype=dt, device=dev)
        hc = torch.minimum(hz, torch.full_like(hz, ghat))
        E = E + 1e2 * mu * (w_ * (-((hc - ghat) ** 2) * torch.log(hc / ghat))).sum()
        if parts:
            return E, u, d, cc
        return E

    for gg in gs:
        opt = FeasLBFGS([var], max_iter=iters, history_size=30)

        def clo():
            opt.zero_grad(set_to_none=True)
            E = energy(var, gg)
            if torch.isfinite(E):
                E.backward()
            return E
        opt.step(clo)
    E, u, d, cc = energy(var.detach(), gs[-1], parts=True)
    cp = ctrl_points(d, wt, lat, lo)
    Ls_ = torch.distributions.Dirichlet(torch.ones(4, dtype=dt, device=dev)).sample((cp.shape[0], 12, 300))
    Bs_ = WF._bern_t(Ls_.reshape(-1, 4), WF.tables(dev, dt)["I3"], WF.tables(dev, dt)["mc3"]).reshape(Ls_.shape[:3] + (20,))
    smp_ = torch.einsum("tspk,tskc->tspc", Bs_, cp)
    # 실제 물체 경계 = 입자 덩어리 경계면; 여기서는 점유 사면체 전체의 상(사면체가 물체를 감싼다)을 본다
    xp = P + u
    return dict(mode=mode, particle_min_minus_floor=float(xp[:, 2].min() - floor), ctrl_min_minus_floor=float(cp[..., 2].min() - floor),
                dense_min_minus_floor=float(smp_[..., 2].min() - floor), coef_min=float(cc.min()),
                _smp=smp_.reshape(-1, 3)[::7].cpu().numpy(), _xp=xp.detach().cpu().numpy(), _cp=cp.reshape(-1, 3).detach().cpu().numpy())


RES["floor"] = {}
for mode in ("particle", "ctrl"):
    r = cantilever(mode)
    RES["floor"][mode] = {k: vv for k, vv in r.items() if not k.startswith("_")}
    np.savez(os.path.join(OUT, f"ct3d_floor_{mode}.npz"), smp=r["_smp"], xp=r["_xp"], cp=r["_cp"])
    print(f"[바닥] {mode}: 입자 최저-바닥 {r['particle_min_minus_floor']:+.4e}, 제어점 최저-바닥 {r['ctrl_min_minus_floor']:+.4e}, "
          f"조밀 표본 최저-바닥 {r['dense_min_minus_floor']:+.4e}, det 계수 최소 {r['coef_min']:.3e}", flush=True)
fc_ = RES["floor"]["ctrl"]
chk("8) 제어점 바닥 장벽: 요소 상 전체가 바닥 위", fc_["dense_min_minus_floor"] > 0 and fc_["ctrl_min_minus_floor"] > 0, json.dumps(fc_))

# 9) 요소 말림 감시 (정보: 통과 조건 아님 -- 단, 항등 사상에서 0 이어야 감시가 쓸모 있다)
lo9, lat9, nn9 = ens5[2]
c_id = WF.curl_check(torch.zeros(M5, WF.C1_NOUT, dtype=dt, device=dev), ens5[7].wft, lo9, lat9, nn9, ens5[5])
torch.manual_seed(9)
c_r = [WF.curl_check(amp * torch.randn(M5, WF.C1_NOUT, dtype=dt, device=dev), ens5[7].wft, lo9, lat9, nn9, ens5[5]) for amp in (0.003, 0.03, 0.3)]
RES["curl"] = dict(identity=c_id, random=c_r, tets=int(ens5[7].wft["rows"].shape[0]))
chk("9) 말림 감시: 항등 사상에서 말린 요소 0 (무작위 0.003/0.03/0.3 은 참고)", c_id[0] == 0, f"항등 {c_id}, 무작위 {c_r}, 사면체 {RES['curl']['tets']}")
# 통과 조건 (사용자 10-20): 성질 1 (계수>0 ⇒ det>0, 장벽으로 뒤집힘 0) 과 성질 3 (C1) 및 그 전제(재현·autograd·감시 0).
# 6) 자기 조각/바닥 볼록 껍질, 8) 바닥 시연 은 기록만.
REQ = ("1)", "2)", "3)", "4)", "5)", "7)", "9)")
ok = all(c["ok"] for c in RES["checks"] if c["name"].startswith(REQ))
RES["pass"] = bool(ok)
json.dump(RES, open(os.path.join(OUT, "ct3d_results.json"), "w"), ensure_ascii=False, indent=1, default=float)
print("CT3D_TEST_PASS" if ok else "CT3D_TEST_FAIL", flush=True)
