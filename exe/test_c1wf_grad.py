"""--interp c1wf 기울기 시험 (float64): autograd 기울기가 참 기울기인지.

  A) pl_c1_dofs 수반(adjoint) 시험: 선형 사상 L 에 대해 <L d, w> = <d, Lᵀw> (autograd) -- 상대오차.
     고친 판(기본)과 레거시(WF.PL_LEGACY_DUP, 고치기 전 중복 쓰기) 둘 다 재고, 순전파 값이 정확히 같은지.
  B) advance 경로 전체 (노드 출력 θ -> [pl 기저] -> 입자 위치·F, Bernstein det 계수 로그 장벽, 손잡이 노드 torch.where)
     의 스칼라 E(θ) 에서 방향미분: 중심 유한차분 (E(θ+εd)-E(θ-εd))/2ε 대 ∇E·d.
     방향: 임의 전체, 값 채널만(0:3), 기울기 채널만(3:12), 변 채널만(12:54), -∇E. c1wf pl · c1wf none · bary(선형) 각각.
     레거시 판의 어긋남도 함께 적는다 (검출력 확인).
마지막 줄 C1WF_GRAD_TEST PASS / FAIL
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch

from anchorflow import deform_model as DM
from anchorflow import simplex as SX
from anchorflow import wf as WF

dt = torch.float64
dev = "cuda" if torch.cuda.is_available() and os.environ.get("CT3D_CPU") is None else "cpu"
torch.manual_seed(0)
H = 0.056569
ok = True
TOL_ADJ, TOL_FD = 1e-10, 1e-6


def chk(n, c, i=""):
    global ok
    ok = ok and bool(c)
    print(f"[{'통과' if c else '실패'}] {n} {i}", flush=True)


def mk(interp, base):
    return types.SimpleNamespace(hex_jac=False, agg="bary", rel_pe=0, bary_pe=0, interp=interp, c1_base=base, n_nodes=32,
                                 hz_ratio=SX.HZ_CUBE, node_h=H, net_inertial=False, bc_ipg=False, gregory=False, bc_stick=0.0,
                                 bc_softplus=0.0, bc_soft=False, bc_mode="project", bc_level="node", v_from_dt=False, dt_cond=False)


cfg = {"boundary_conditions": [], "grid_lim": 2.0, "n_grid": 100, "g": [0.0, 0.0, 0.0]}
NP = int(os.environ.get("C1G_NP", 3000))
x = (0.7 + 0.3 * torch.rand(NP, 3, dtype=dt)).to(dev)
mass = torch.ones(NP, dtype=dt, device=dev)

# ------------------------------------------------------------------ A) 수반 시험
ensA = DM.node_feats(mk("c1wf", "pl"), cfg, x, mass, x, torch.zeros_like(x), 1.0)
wt = ensA[7].wft
M = ensA[1].shape[0]
for leg in (False, True):
    WF.PL_LEGACY_DUP[0] = leg
    nm = "레거시(고치기 전)" if leg else "고친 판"
    errs = []
    for r in range(5):
        d = torch.randn(M, 3, dtype=dt, device=dev)
        w = torch.randn(M, 51, dtype=dt, device=dev)
        lhs = float((WF.pl_c1_dofs(d, wt, M) * w).sum())
        v = torch.zeros(M, 3, dtype=dt, device=dev, requires_grad=True)
        (WF.pl_c1_dofs(v, wt, M) * w).sum().backward()
        rhs = float((d * v.grad).sum())
        errs.append(abs(lhs - rhs) / max(abs(lhs), 1e-300))
    e = max(errs)
    if leg:
        print(f"[정보] A) pl_c1_dofs 수반 시험 {nm}: 상대오차 최대 {e:.3e} (클수록 버그 검출)", flush=True)
        chk("A) 레거시 판은 수반 시험에서 어긋난다 (검출력)", e > 1e-3, f"{e:.3e}")
    else:
        chk("A) pl_c1_dofs 수반 시험 <Ld,w> = <d,Lᵀw> (고친 판)", e < TOL_ADJ, f"상대오차 최대 {e:.3e} (5 회)")
WF.PL_LEGACY_DUP[0] = False
vv = torch.randn(M, 3, dtype=dt, device=dev)
o_fix = WF.pl_c1_dofs(vv, wt, M)
WF.PL_LEGACY_DUP[0] = True
o_leg = WF.pl_c1_dofs(vv, wt, M)
WF.PL_LEGACY_DUP[0] = False
dval = float((o_fix - o_leg).abs().max())
chk("A) 순전파 값: 고친 판 = 레거시 (정확히)", dval == 0.0, f"최대 차 {dval:.1e}")


# ------------------------------------------------------------------ B) advance 경로 방향미분
def setup(interp, base):
    a = mk(interp, base)
    ens = DM.node_feats(a, cfg, x, mass, x, torch.zeros_like(x), 1.0)
    Mn = ens[1].shape[0]
    nch = WF.C1_NOUT if interp == "c1wf" else 3
    th0 = 0.005 * H * torch.randn(Mn, nch, dtype=dt, device=dev)
    th0[:, :3] += 0.05 * H * torch.randn(Mn, 3, dtype=dt, device=dev)
    hm = torch.rand(Mn, device=dev) < 0.1                       # 손잡이 노드 (명령 평행이동, 변수에서 빠진다)
    hv = 0.5 * H * torch.randn(Mn, 3, dtype=dt, device=dev)
    w1 = torch.randn(NP, 3, dtype=dt, device=dev)
    w2 = torch.randn(NP, 3, 3, dtype=dt, device=dev)

    def E(th):
        x2, _vn, _dpn, Jf, _c, _act = DM.advance(None, a, [ens], x, 0.04, cfg, node_mask=[hm], node_vel=[hv],
                                                 ctrl_scale=1.0, dp_fast=lambda _M, _d: (th,))
        e = (w1 * (x2 - x)).sum() / H + (w2 * Jf).sum() * 0.1
        if interp == "c1wf":
            e = e + 0.01 * WF.barrier_energy(DM.C1_LAST[0][0], DM.C1_LAST[0][1], 1.0, 1.5)
        return e
    return E, th0


GROUPS = {"임의 전체": None, "값 채널 0:3": (0, 3), "기울기 채널 3:12": (3, 12), "변 채널 12:54": (12, 54)}


def fd_check(E, th0, nrep=3, eps=1e-6):
    th = th0.clone().requires_grad_(True)
    e0 = E(th)
    assert torch.isfinite(e0), "E 가 유한하지 않다 (장벽 계수 ≤0?)"
    g, = torch.autograd.grad(e0, th)
    out = {}
    for gname, sl in GROUPS.items():
        if sl is not None and sl[0] >= th0.shape[1]:
            continue
        rs = []
        for r in range(nrep):
            d = torch.randn_like(th0)
            if sl is not None:
                m = torch.zeros_like(d); m[:, sl[0]:sl[1]] = 1.0; d = d * m
            d = d / d.norm() * th0.norm() * 1e-2
            with torch.no_grad():
                fd = (float(E(th0 + eps * d)) - float(E(th0 - eps * d))) / (2 * eps)
            gd = float((g * d).sum())
            rs.append(abs(fd - gd) / max(abs(fd), abs(gd), 1e-300))
        out[gname] = max(rs)
    d = -g / g.norm() * th0.norm() * 1e-2
    with torch.no_grad():
        fd = (float(E(th0 + eps * d)) - float(E(th0 - eps * d))) / (2 * eps)
    gd = float((g * d).sum())
    out["-∇E"] = abs(fd - gd) / max(abs(fd), abs(gd), 1e-300)
    return out


for interp, base in (("c1wf", "pl"), ("c1wf", "none"), ("bary", "none")):
    E, th0 = setup(interp, base)
    nm = f"{interp}{' ' + base if interp == 'c1wf' else ' (선형)'}"
    WF.PL_LEGACY_DUP[0] = False
    r = fd_check(E, th0)
    chk(f"B) 방향미분 {nm}", max(r.values()) < TOL_FD, ", ".join(f"{k} {v:.2e}" for k, v in r.items()))
    if interp == "c1wf" and base == "pl":
        WF.PL_LEGACY_DUP[0] = True
        rl = fd_check(E, th0)
        WF.PL_LEGACY_DUP[0] = False
        print(f"[정보] B) 레거시(고치기 전) {nm}: " + ", ".join(f"{k} {v:.2e}" for k, v in rl.items()), flush=True)
        chk("B) 레거시 판은 값 채널 방향에서 어긋난다 (검출력)", rl["값 채널 0:3"] > 1e-3, f"{rl['값 채널 0:3']:.2e}")
print(f"[장치] {dev}, 입자 {NP}", flush=True)
print("C1WF_GRAD_TEST " + ("PASS" if ok else "FAIL"), flush=True)
