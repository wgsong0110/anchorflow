"""변형장의 t 미분(속도)과 공간 야코비안(F 밀기)의 정합성 검사.

  A) 정확 검사 -- dtfilm 을 0 으로, --dt_scale 을 켜고 **선형** 전달(tri)을
     쓰면 dp(t) = c·t 이므로 Phi_t(x) = x + W c t 다. 그러면 dPhi/dt 가 할선
     (x2-x)/t 와 **기계 정밀도로 같아야** 한다. 순방향 AD 배선이 맞는지 여기서
     갈린다 (값이 뭐든 맞아떨어질 여지가 없다).
  B) 수치 검사 -- 일반 망에서 dPhi/dt 를 t 중심차분과 댄다. fp32 차분은 상쇄로
     못 쓰므로 **float64** 로 잰다.
  C) 공간 야코비안 -- grad_x Phi 를 공간 중심차분과 댄다. rqs 는 셀 면에서
     접선 불연속이 **허용**되므로 면에서 떨어진 내부점만 쓴다.
  D) det grad_x Phi > 0 (접힘 없음).
  E) 참고 -- g2p_grad 로 민 F 와 변형장 야코비안으로 민 F 가 얼마나 다른가.

  python exe/test_field_deriv.py --traj traj_h2/mic_clayC_t_s400706.pt
"""
import argparse

import torch
import torch.autograd.forward_ad as fwAD

from anchorflow import phys_resid, tri_spline, vox_anchor
from anchorflow.conv_stepper import ConvStepper
from anchorflow.deform import jacobian_of
from anchorflow.sitreg_warp import (BoundedWarp, WARP_BOUND,
                                    bary_g2p, bary_g2p_jac)

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--t0", type=int, default=10)
ap.add_argument("--n_pts", type=int, default=3000)
ap.add_argument("--vox_res", type=int, default=32)
ap.add_argument("--bins", type=int, default=8)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
d = torch.load(a.traj, map_location="cpu", weights_only=False)
cfg = d["cfg"]
HDT = float(cfg["frame_dt"])
X = d["x"].float()
sel = torch.randperm(X.shape[1],
                     generator=torch.Generator().manual_seed(0)
                     )[:a.n_pts].sort().values
x32 = X[a.t0, sel].to(dev)
v32 = ((X[a.t0, sel] - X[a.t0 - 1, sel]) / HDT).to(dev)
mass32 = torch.full((sel.numel(),), float(cfg["density"]), device=dev)
lo32, hh, nn3 = vox_anchor.grid_for(x32, a.vox_res ** 3)
hh = float(hh)
NC = int((nn3[0] - 1) * (nn3[1] - 1) * (nn3[2] - 1))
P = tri_spline.n_params(a.bins)
CELLS = tuple(int(nn3[k]) - 1 for k in range(3))
GRID = tuple(int(c) for c in nn3)
BND = WARP_BOUND * hh
feat32 = torch.randn(NC, 8, device=dev)
ok = True


def chk(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}", flush=True)


def mknet(cw, wake=True, dtype=torch.float32):
    torch.manual_seed(1)
    n = ConvStepper(n_feat=8, hidden=32, depth=2, h=hh, scale=0.02,
                    dt_cond=True, dt_ref=HDT, dt_scale=True,
                    rqs_dim=(P if cw == "rqs" else 0)).to(dev).to(dtype)
    with torch.no_grad():
        n.out.weight.normal_(0, 0.05)
        if cw == "rqs":
            n.out_rqs.weight.normal_(0, 0.05)
        last = [m for m in n.dtfilm.mlp.modules()
                if isinstance(m, torch.nn.Linear)][-1]
        if wake:            # 학습된 상태를 흉내낸다 (0 이면 t 의존이 없다)
            last.weight.normal_(0, 0.05); last.bias.normal_(0, 0.05)
        else:               # 항등 FiLM -- dp 가 t 에 정확히 비례하게 둔다
            last.weight.zero_(); last.bias.zero_()
        n.dtfilm._cache.clear()
    return n


def field(net, kind, tau, q, lo, nn3_, sidx, feat):
    """kind = (로컬 cell_warp, 글로벌 node_warp). 두 축은 직교다."""
    cw, nw = kind
    out = net(None, feat, tau, GRID, cells=CELLS)
    dp = out[0]
    if cw == "rqs":
        q = tri_spline.remap(q, lo, hh, nn3_, out[-1], a.bins)
    if nw == "bound":
        return BoundedWarp(BND, 5).apply(
            q, dp, lambda z, c: z + bary_g2p(z, lo, hh, nn3_, c))
    return q + bary_g2p(q, lo, hh, nn3_, dp)


def field_jac(net, kind, tau, q, lo, nn3_, feat):
    """train_deform 의 --jac analytic 과 같은 닫힌 형식 야코비안."""
    cw, nw = kind
    out = net(None, feat, tau, GRID, cells=CELLS)
    dp = out[0]
    kj = bary_g2p_jac
    sl = None
    if cw == "rqs":
        q, sl = tri_spline.remap(q, lo, hh, nn3_, out[-1], a.bins,
                                 return_jac=True)
    if nw == "bound":
        _, J = BoundedWarp(BND, 5).apply_jac(
            q, dp, lambda z, c: kj(z, lo, hh, nn3_, c))
    else:
        _u, G = kj(q, lo, hh, nn3_, dp)
        J = torch.eye(3, device=dev, dtype=q.dtype) + G
    if sl is not None:
        J = J * sl.unsqueeze(1)
    return J



def dphi_dt(net, kind, tau_v, q, lo, nn3_, sidx, feat):
    """출하 경로와 **같은** 방법으로 잰다: torch.func.jvp.

    수동 dual_level 은 쓰지 않는다 -- 레벨을 벗어난 primal 이 무효가 되어
    학습 언롤의 역전파가 깨지고(실측), rqs 의 detach 와도 어긋났다.
    """
    tau0 = torch.as_tensor(tau_v, device=dev, dtype=q.dtype)
    return torch.func.jvp(
        lambda t: field(net, kind, t, q, lo, nn3_, sidx, feat),
        (tau0,), (torch.ones_like(tau0),))


# ---- A) 정확 검사: 선형 전달 + 항등 FiLM 이면 dPhi/dt == 할선 -----------------
print("== A) 배선 정확 검사 (none+plain, 항등 dtfilm)")
netA = mknet("none", wake=False)
x2A, vA = dphi_dt(netA, ("none", "plain"), HDT, x32, lo32, nn3, None, feat32)
secA = (x2A - x32) / HDT
relA = float((vA - secA).norm() / secA.norm().clamp_min(1e-30))
chk("dPhi/dt == (x2-x)/t (dp ∝ t 이므로 정확히 같아야)", relA < 1e-5,
    f"상대오차 {relA:.2e}")

# ---- B~D) float64 로 수치 검증 ------------------------------------------------
x64 = x32.double()
lo64 = lo32.double()
feat64 = feat32.double()
# 로컬 x 글로벌 네 조합 -- 두 축이 직교인지 확인
for kind in (("none", "plain"), ("none", "bound"),
             ("rqs", "plain"), ("rqs", "bound")):
    print(f"== {kind[0]} + {kind[1]}")
    net = mknet(kind[0], wake=True, dtype=torch.float64)
    # B) t 미분
    x2, vd = dphi_dt(net, kind, HDT, x64, lo64, nn3, None, feat64)
    eps = HDT * 1e-4
    with torch.no_grad():
        fd = (field(net, kind, torch.as_tensor(HDT + eps, device=dev,
                                               dtype=torch.float64),
                    x64, lo64, nn3, None, feat64)
              - field(net, kind, torch.as_tensor(HDT - eps, device=dev,
                                                 dtype=torch.float64),
                      x64, lo64, nn3, None, feat64)) / (2 * eps)
    _pp = ((vd - fd).norm(dim=-1) / fd.norm(dim=-1).clamp_min(1e-30))
    _nb = int((_pp > 1e-3).sum())
    # 판정은 **입자별** 상대오차로 한다. 전역 노름비는 소수의 이상점이 지배한다.
    chk(f"{kind}: dPhi/dt == t 중심차분", float(_pp.median()) < 1e-5,
        f"중앙 {float(_pp.median()):.2e}, 1e-3 초과 {_nb}/{_pp.numel()} (면집합)")
    # 이상점이 **국소 비평활** 때문인지(eps 를 줄이면 사라진다) AD 가 틀린
    # 것인지(모든 eps 에서 남는다) 를 스윕으로 가른다 -- 추측하지 않는다.
    if _nb:
        line = []
        for _ee in (1e-3, 1e-4, 1e-5, 1e-6):
            _e = HDT * _ee
            with torch.no_grad():
                _f2 = (field(net, kind, torch.as_tensor(
                            HDT + _e, device=dev, dtype=torch.float64),
                        x64, lo64, nn3, None, feat64)
                       - field(net, kind, torch.as_tensor(
                            HDT - _e, device=dev, dtype=torch.float64),
                        x64, lo64, nn3, None, feat64)) / (2 * _e)
            _q = ((vd - _f2).norm(dim=-1)
                  / _f2.norm(dim=-1).clamp_min(1e-30))
            line.append(f"eps={_ee:.0e}:{int((_q > 1e-3).sum())}")
        print(f"      이상점 수의 eps 의존: {' '.join(line)}  "
              f"(줄어들면 국소 비평활, 남으면 AD 오류)")
    sec = (x2 - x64) / HDT
    print(f"      (참고) 할선과의 차이 "
          f"{100*float((vd-sec).norm()/sec.norm()):.1f}% -- 차분이 담지 못한 몫")

    # C/D) 공간 야코비안. rqs 는 셀 면 접선 불연속이 허용되므로 내부점만.
    if kind[0] == "rqs":
        ti = (x64 - lo64) / hh
        frac = ti - ti.floor()
        keep = ((frac > 0.15) & (frac < 0.85)).all(-1)
        xs = x64[keep]
    else:
        xs = x64
    Ja = jacobian_of(lambda z: field(net, kind, torch.as_tensor(
        HDT, device=dev, dtype=torch.float64), z, lo64, nn3, None, feat64),
        xs)
    with torch.no_grad():
        I3 = torch.eye(3, device=dev, dtype=torch.float64)
        sp = 1e-6
        cols = []
        for k in range(3):
            fp = field(net, kind, torch.as_tensor(HDT, device=dev,
                                                  dtype=torch.float64),
                       xs + sp * I3[k], lo64, nn3, None, feat64)
            fm = field(net, kind, torch.as_tensor(HDT, device=dev,
                                                  dtype=torch.float64),
                       xs - sp * I3[k], lo64, nn3, None, feat64)
            cols.append((fp - fm) / (2 * sp))
        Jfd = torch.stack(cols, -1)
    # 판정은 입자별로 한다. PL 전달은 C0 이라 사면체 면에서 기울기가 꺾이고,
    # bound 는 K 번 합성하며 중간 위치들이 또 면을 밟을 수 있어 꺾임 집합이
    # K 배 촘촘하다 -- 그 위를 ±sp 로 걸치는 소수 입자의 중심차분은 미분이
    # 아니다 (t 미분 때와 같은 부류, AD 가 a.e. 의 옳은 값).
    _pj = ((Ja - Jfd).reshape(xs.shape[0], -1).norm(dim=-1)
           / Jfd.reshape(xs.shape[0], -1).norm(dim=-1).clamp_min(1e-30))
    _njb = int((_pj > 1e-3).sum())
    Jan = field_jac(net, kind, torch.as_tensor(HDT, device=dev,
                                               dtype=torch.float64),
                    xs, lo64, nn3, feat64).detach()
    # 입자별로 본다. 전역 노름비는 셀 면/동률(argsort·floor 비평활)을 밟은
    # 소수 점이 지배한다 -- 거기서는 autograd 도 analytic 도 한쪽 값을 주지만
    # 어느 쪽 셀을 잡느냐가 갈릴 수 있다 (측도 0 집합, a.e. 에서 둘은 같다).
    _pa = ((Jan - Ja).reshape(xs.shape[0], -1).norm(dim=-1)
           / Ja.reshape(xs.shape[0], -1).norm(dim=-1).clamp_min(1e-30))
    _na = int((_pa > 1e-6).sum())
    # PL·C0 사상이라 셀 면/동률(비평활 집합)에서는 서로 다른 한쪽 값을 줄 수
    # 있다 -- 개수로 FAIL 하지 않고 중앙값으로 본다.
    chk(f"{kind}: 해석 야코비안 == autograd", float(_pa.median()) < 1e-9,
        f"중앙 {float(_pa.median()):.1e}, 1e-6 초과 {_na}/{xs.shape[0]} (면집합)")
    det = torch.linalg.det(Ja)
    # bary 는 사면체 면이 큐브 면보다 많아 비평활 집합이 촘촘하다 -- 중앙값 판정
    chk(f"{kind}: grad_x Phi == 공간 중심차분", float(_pj.median()) < 1e-6,
        f"중앙 {float(_pj.median()):.2e}, 1e-3 초과 {_njb}/{xs.shape[0]} (면집합)")
    if kind[1] == "bound":
        # 단사 보장은 글로벌이 bound 일 때다 (로컬 RQS 는 자체 단사라 합성 유지)
        chk(f"{kind}: det grad_x Phi > 0", bool((det > 0).all()),
            f"최소 {float(det.min()):.4f} 중앙 {float(det.median()):.4f}")
    else:
        print(f"      (참고) det 최소 {float(det.min()):.4f} -- 상한이 없어 "
              f"보장 없음")
    # E) g2p_grad 로 민 F 와 변형장 야코비안으로 민 F 가 얼마나 다른가.
    #    야코비안은 자동미분이 필요하므로 no_grad 밖에서 부른다.
    Jfull = jacobian_of(lambda z: field(net, kind, torch.as_tensor(
        HDT, device=dev, dtype=torch.float64), z, lo64, nn3, None,
        feat64), x64).detach()
    with torch.no_grad():
        I3 = torch.eye(3, device=dev, dtype=torch.float64)
        F0 = I3.expand(x64.shape[0], 3, 3).contiguous()
        _m, duI, vI, info, _fr = phys_resid.p2g_increment(
            x64, x2 - x64, v32.double(), mass32.double(), int(cfg["n_grid"]),
            float(cfg.get("grid_lim", 2.0)))
        gu = phys_resid.g2p_grad(x64, duI, info, int(cfg["n_grid"]))
        Fg = (I3 + gu) @ F0
        Fj = Jfull @ F0
        rr = float((Fg - Fj).norm() / (Fj - I3).norm().clamp_min(1e-30))
        print(f"      g2p_grad F vs 변형장 야코비안 F: 증분 대비 {100*rr:.1f}%"
              f"  (det 중앙 g2p {float(torch.linalg.det(Fg).median()):.4f} / "
              f"장 {float(torch.linalg.det(Fj).median()):.4f})")

print("ALL-OK" if ok else "SOME-FAIL")
raise SystemExit(0 if ok else 1)
