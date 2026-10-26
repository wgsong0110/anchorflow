"""--interp bspline2 / bspline3 단위 시험 (CPU float64, 빌드 없음): deform_model.node_feats + advance 경로 그대로

  0) 활성 노드 = 사면체 꼭짓점 ∪ B-스플라인 지지 전부, 가중치 합 1, 표본점 수
  1) 항등: 노드 변위 0 -> x' = x, F_step = I (두 판, (B) 상한 사상 포함)
  2) 아핀 재현: U = B·X + c -> x' - x = B·x + c, F_step = I + B (2 차; 3 차는 상한 사상 뒤 U = ū + s(BX+c-ū) 도 아핀 -> 그대로 확인)
  3) 야코비안 = 변위의 위치 중심차분 (무작위 노드 변위)
  4) 셀 경계 양쪽 F 연속 (다항식 조각 경계면 세 축 + Kuhn 사면체 내부면), 최대 차; bary 대조
  5) (B) 큰 무작위 변위 (이웃 차 ~Δ): 상한 사상 뒤 이웃 차 최대 < 0.48Δ, det 최소 > 0 (입자·조각 표본점·무작위 조밀 표본);
     상한 없이는 뒤집힘 (대조); 사상 기울기 gradcheck
  6) (A) 장벽: det ≥ Ĵ 이면 0, det -> 0 에서 커진다; 장벽 기울기 반대 방향 한 걸음이 표본 det 최소를 올린다
  7) 역전파 기울기 유한, jvp 속도 = 중심차분
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch

from anchorflow import bspfd as BF
from anchorflow import deform_model as DM
from anchorflow import phys_resid
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
N0 = 4000
X = 0.7 + 0.6 * torch.rand(N0, 3, dtype=dt)
lo0, lat0, nn0 = SX.grid_for_nodes(X, 32, hz_ratio=SX.HZ_CUBE, h_fix=H)
DEL = 1e-7
yb = ((X[:500] - lo0) @ lat0.Ai.T)
I3 = torch.eye(3, dtype=dt)


def mk_pairs(shift):
    """조각 경계면 (y_ax = 정수 + shift ± δ) 세 축 + Kuhn 내부면 (f0 = f1 ± δ)."""
    out = []
    for ax in range(3):
        y = yb.clone(); y[:, ax] = torch.round(y[:, ax] - shift) + shift
        e = torch.zeros(3, dtype=dt); e[ax] = DEL
        out.append((y - e, y + e, f"조각면{ax}"))
    y = yb.clone(); fr = y - y.floor(); y[:, 1] = y[:, 1].floor() + fr[:, 0]
    e = torch.zeros(3, dtype=dt); e[0] = DEL; e[1] = -DEL
    out.append((y - e, y + e, "사면체 내부면"))
    return out


PAIRS = {"bary": mk_pairs(0.0), "bspline2": mk_pairs(0.5), "bspline3": mk_pairs(0.0)}
XS = {}
for k, pr in PAIRS.items():
    XS[k] = torch.cat([X] + [(p[0] @ lat0.A.T + lo0) for p in pr] + [(p[1] @ lat0.A.T + lo0) for p in pr])
nP = 500


def run(interp, dp_fn, v_from_dt=False, inertial=None, h=0.04, xq=None, node_mask=None):
    xq = XS[interp] if xq is None else xq
    a = mk_a(interp, v_from_dt, inertial is not None)
    N = xq.shape[0]
    ens = DM.node_feats(a, cfg, xq, torch.ones(N, dtype=dt), xq, torch.zeros(N, 3, dtype=dt), 1.0)
    dp = dp_fn(ens[1])
    DM.BSP_LAST[0] = None
    x2, vn, dpn, Jf, _c, _act = DM.advance(None, a, [ens], xq, h, cfg, dp_fast=lambda M, _d: (dp,),
                                           inertial=([inertial(ens)] if inertial is not None else None),
                                           node_mask=node_mask)
    return ens, dp, x2, vn, Jf, dpn


# 0) 활성 노드·가중치
for mode, J in (("bspline2", 27), ("bspline3", 64)):
    ens, *_ = run(mode, lambda npos: torch.zeros(npos.shape[0], 3, dtype=dt))
    lo_, lat_, nn_ = ens[2]; rows, lam, uniq, aux = ens[3], ens[4], ens[5], ens[7]
    fb, wb, _ = BF.weights(XS[mode], lo_, lat_, nn_, 2 if J == 27 else 3)
    ntet = SX.active_nodes(SX.locate(XS[mode], lo_, lat_, nn_)[0])[1].numel()
    chk(f"0) {mode} {J} 지지 노드 전부 활성", rows.shape[1] == J and bool((uniq[rows] == fb).all()),
        f"활성 노드 {uniq.numel()} (사면체 꼭짓점만이면 {ntet}, {uniq.numel() / ntet:.2f} 배)")
    chk(f"0) {mode} 가중치 합 1", float((lam.sum(1) - 1).abs().max()) < 1e-13,
        f"점유 조각 {aux.smp['n_piece']}, 표본점 {aux.smp['w'].numel()}, 가중 합 {float(aux.smp['w'].sum()):.3e} "
        f"(조각 부피 합 {aux.smp['n_piece'] * abs(float(torch.linalg.det(lat_.A))):.3e})"
        + (f", 축 이웃 쌍 {aux.cap[0].numel()}" if aux.cap is not None else ""))

# 1) 항등
for mode in ("bspline2", "bspline3"):
    _, _, x2, _, Jf, _ = run(mode, lambda npos: torch.zeros(npos.shape[0], 3, dtype=dt))
    xq = XS[mode]
    e1, e2 = float((x2 - xq).abs().max()), float((Jf - I3).abs().max())
    chk(f"1) 항등 {mode}", e1 == 0.0 and e2 == 0.0, f"|x'-x| 최대 {e1:.1e}, |F-I| 최대 {e2:.1e}")

# 2) 아핀 재현
B = 0.05 * torch.randn(3, 3, dtype=dt); c = 0.01 * torch.randn(3, dtype=dt)
for mode in ("bspline2", "bspline3"):
    ens, dp, x2, _, Jf, dpn = run(mode, lambda npos: npos @ B.T + c)
    xq = XS[mode]
    if mode == "bspline3":
        st = DM.BSP_LAST[0]["cap"]; s = float(st[2])
        ref = dp.mean(0)
        tgt_u = ref + s * (xq @ B.T + c - ref); tgt_J = I3 + s * B
        extra = f", 상한 사상 s = {s:.10f} (이웃 차 최대 {float(st[0]):.3f}Δ, 매끄러운 최대 {float(st[1]):.3f}Δ)"
    else:
        tgt_u = xq @ B.T + c; tgt_J = I3 + B; extra = ""
    e_u = float((x2 - xq - tgt_u).abs().max()); e_J = float((Jf - tgt_J).abs().max())
    chk(f"2) 아핀 재현 {mode}", e_u < 1e-12 and e_J < 1e-11, f"변위 오차 {e_u:.1e}, 야코비안 오차 {e_J:.1e}{extra}")

# 3) 야코비안 = 위치 중심차분 (노드 변위 무작위, 사상 뒤 노드 변위를 고정해 위치만 바꿔 다시 보간)
for mode in ("bspline2", "bspline3"):
    torch.manual_seed(3)
    ens, dp, x2, _, Jf, dpn = run(mode, lambda npos: 2e-3 * torch.randn(npos.shape[0], 3, dtype=dt))
    lo_, lat_, nn_ = ens[2]; uniq = ens[5]
    U = dpn[:, :3]
    sel = torch.randperm(N0)[:200]

    def u_at(q):
        fl, w, _ = BF.weights(q, lo_, lat_, nn_, 2 if mode == "bspline2" else 3)
        return (w.unsqueeze(-1) * U[torch.searchsorted(uniq, fl)]).sum(1)
    e = 1e-6
    Jfd = torch.stack([(u_at(XS[mode][sel] + e * I3[k]) - u_at(XS[mode][sel] - e * I3[k])) / (2 * e) for k in range(3)], -1)
    ea = float((Jfd - (Jf[sel] - I3)).abs().max())
    chk(f"3) {mode} 야코비안 = 중심차분", ea < 1e-8, f"최대 차 {ea:.1e} (|∇u| 최대 {float((Jf - I3).abs().max()):.1e})")

# 4) 셀 경계 연속성
for mode in ("bary", "bspline2", "bspline3"):
    torch.manual_seed(1)
    _, _, _, _, Jf, _ = run(mode, lambda npos: 1e-3 * torch.randn(npos.shape[0], 3, dtype=dt))
    msg, worst = [], 0.0
    L = len(PAIRS[mode])
    for k, (_, _, nm) in enumerate(PAIRS[mode]):
        jm = Jf[N0 + k * nP:N0 + (k + 1) * nP]; jp = Jf[N0 + (L + k) * nP:N0 + (L + k + 1) * nP]
        d_ = float((jm - jp).abs().amax((1, 2)).max()); worst = max(worst, d_); msg.append(f"{nm} {d_:.1e}")
    scale = float((Jf - I3).abs().max())
    if mode == "bary":
        chk("4) bary 는 경계에서 튄다 (대조)", worst > 1e-3 * scale, f"|ΔF| 최대 {worst:.1e} (|∇u| 최대 {scale:.1e}): " + ", ".join(msg))
    else:
        chk(f"4) {mode} 경계에서 F 연속", worst < 1e-5 * scale, f"|ΔF| 최대 {worst:.1e} (|∇u| 최대 {scale:.1e}, 거리 2e-7 격자단위): " + ", ".join(msg))

# 5) (B) 큰 변위
DLT = H / 2 ** 0.5
for amp in (0.5, 2.0):
    torch.manual_seed(5)
    ens, dp, x2, _, Jf, dpn = run("bspline3", lambda npos: amp * DLT * torch.randn(npos.shape[0], 3, dtype=dt))
    st = DM.BSP_LAST[0]["cap"]; dsm = DM.BSP_LAST[0]["det"]
    lo_, lat_, nn_ = ens[2]; uniq, aux = ens[5], ens[7]
    # 조밀 무작위 표본: 점유 조각 안 균일 (지지 64 노드가 모두 활성인 점)
    yq = ((XS["bspline3"] - lo_) @ lat_.Ai.T).floor().unique(dim=0)
    yr = (yq.repeat(20, 1) + torch.rand(yq.shape[0] * 20, 3, dtype=dt))
    qr = yr @ lat_.A.T + lo_
    fl, w, dw = BF.weights(qr, lo_, lat_, nn_, 3)
    Jr = SX.bspline_jac_pre(torch.searchsorted(uniq, fl), dw, lat_, dpn)
    dr = torch.linalg.det(I3 + Jr)
    dp_ = torch.linalg.det(Jf)
    nrm2 = float(torch.linalg.matrix_norm(Jr, ord=2).max())
    chk(f"5) (B) 진폭 {amp}Δ: 사상 뒤 이웃 차 < 0.48Δ, det > 0",
        float(st[3]) < BF.CAP_C and float(dp_.min()) > 0 and float(dsm.min()) > 0 and float(dr.min()) > 0 and nrm2 < 1.0,
        f"사상 전 이웃 차 최대 {float(st[0]):.3f}Δ -> 후 {float(st[3]):.4f}Δ (s {float(st[2]):.4f}); det 최소 입자 {float(dp_.min()):.4f}, "
        f"조각 표본 {float(dsm.min()):.4f} ({dsm.numel()} 점), 조밀 무작위 {float(dr.min()):.4f} ({dr.numel()} 점); ‖∇u‖₂ 최대 {nrm2:.3f} (< 1, 하한 det ≥ {(1 - 3 ** 0.5 * BF.CAP_C) ** 3:.4f})")
    # 대조: 상한 없이 (같은 V 를 그대로)
    Jn = SX.bspline_jac_pre(torch.searchsorted(uniq, fl), dw, lat_, dp)
    dn = torch.linalg.det(I3 + Jn)
    chk(f"5) (B) 대조: 상한 없이 진폭 {amp}Δ", float(dn.min()) < 0, f"det 최소 {float(dn.min()):.3f}, 뒤집힌 비율 {float((dn <= 0).double().mean()):.3f}")

# 5b) 구속 노드는 그대로, 사상 기울기 gradcheck
ens, *_ = run("bspline3", lambda npos: torch.zeros(npos.shape[0], 3, dtype=dt))
aux = ens[7]; M = ens[1].shape[0]
pin = torch.zeros(M, dtype=torch.bool); pin[:M // 5] = True
V = DLT * torch.randn(M, 3, dtype=dt); V[pin] = 0.0
U, st = BF.cap_map(V, aux.cap, aux.delta, pin=pin)
chk("5b) (B) 구속 노드 값 그대로·이웃 차 < 0.48Δ", float(U[pin].abs().max()) == 0.0 and float(st[3]) < BF.CAP_C,
    f"이웃 차 {float(st[0]):.3f}Δ -> {float(st[3]):.4f}Δ")
sub = torch.randperm(aux.cap[0].numel())[:300]
pairs_s = (aux.cap[0][sub], aux.cap[1][sub])
nodes_s = torch.unique(torch.cat(pairs_s))
remap = torch.full((M,), -1, dtype=torch.long); remap[nodes_s] = torch.arange(nodes_s.numel())
ps = (remap[pairs_s[0]], remap[pairs_s[1]])
Vq = (0.3 * DLT * torch.randn(nodes_s.numel(), 3, dtype=dt)).requires_grad_(True)
chk("5b) (B) 상한 사상 gradcheck", torch.autograd.gradcheck(lambda q: BF.cap_map(q, ps, aux.delta)[0], (Vq,), eps=1e-7, atol=1e-6))

# 6) (A) 장벽
jh = 0.3
Jt = torch.tensor([1.0, 0.5, 0.31, 0.3, 0.2, 0.1, 0.01], dtype=dt)
bv = phys_resid._barrier_b(Jt, jh, ext_kind="linear")
chk("6) (A) 장벽 b(det): det ≥ Ĵ 이면 0, Ĵ 아래로 단조 증가", float(bv[:4].abs().max()) == 0.0 and bool((bv[4:].diff() > 0).all()),
    "b = " + ", ".join(f"{float(q):.3e}" for q in bv))
torch.manual_seed(6)
ens, dp, x2, _, Jf, dpn = run("bspline2", lambda npos: 0.6 * DLT * torch.randn(npos.shape[0], 3, dtype=dt))
lat_ = ens[2][1]; smp = ens[7].smp
q = dp.clone().requires_grad_(True)
d0 = BF.smp_det(smp, lat_, q)
Eb = (smp["w"] * phys_resid._barrier_b(d0, jh, ext_kind="linear")).sum()
g, = torch.autograd.grad(Eb, q)
best = None
for al in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
    d1 = BF.smp_det(smp, lat_, dp - al * g / g.norm() * DLT)
    if best is None or float(d1.min()) > best[1]:
        best = (al, float(d1.min()), int((d1 < jh).sum()))
chk("6) (A) 장벽 기울기 반대 방향 걸음이 표본 det 최소를 올린다", float(Eb) > 0 and best[1] > float(d0.min()),
    f"장벽 {float(Eb):.3e}, det 최소 {float(d0.min()):.4f} -> {best[1]:.4f} (걸음 {best[0]}Δ), Ĵ 아래 표본 {int((d0 < jh).sum())} -> {best[2]}")

# 7) 역전파·jvp
for mode in ("bspline2", "bspline3"):
    torch.manual_seed(2)
    dpf = 1e-3 * torch.randn(40000, 3, dtype=dt)
    vn_g = 0.3 * torch.randn(40000, 3, dtype=dt)
    gq = torch.tensor([0.0, 0.0, -9.8], dtype=dt)

    def inert(e):
        return (vn_g[:e[1].shape[0]], gq)
    _, _, _, vj, _, _ = run(mode, lambda npos: dpf[:npos.shape[0]], True, inert, 0.04)
    e = 1e-6
    _, _, xa, _, _, _ = run(mode, lambda npos: dpf[:npos.shape[0]], False, inert, 0.04 + e)
    _, _, xb, _, _, _ = run(mode, lambda npos: dpf[:npos.shape[0]], False, inert, 0.04 - e)
    ev = float((vj - (xa - xb) / (2 * e)).abs().max())
    chk(f"7) {mode} jvp 속도 = 중심차분", ev < 1e-6, f"최대 차 {ev:.1e} (|v| 최대 {float(vj.abs().max()):.2f})")
    a7 = mk_a(mode)
    xq = XS[mode]; N = xq.shape[0]
    ens7 = DM.node_feats(a7, cfg, xq, torch.ones(N, dtype=dt), xq, torch.zeros(N, 3, dtype=dt), 1.0)
    q7 = (1e-3 * torch.randn(ens7[1].shape[0], 3, dtype=dt)).requires_grad_(True)
    x27, _, _, J7, _, _ = DM.advance(None, a7, [ens7], xq, 0.04, cfg, dp_fast=lambda M, _d: (q7,))
    Ls = (x27 ** 2).sum() + (torch.linalg.det(J7) - 1).pow(2).sum() + DM.BSP_LAST[0]["det"].sum()
    Ls.backward()
    chk(f"7) {mode} 역전파 기울기 유한", bool(torch.isfinite(q7.grad).all()) and float(q7.grad.abs().max()) > 0,
        f"|g| 최대 {float(q7.grad.abs().max()):.2e}")

print("BSPLINE23_TEST " + ("ALL_PASS" if ok else "FAIL"), flush=True)
sys.exit(0 if ok else 1)
