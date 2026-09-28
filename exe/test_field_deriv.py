"""변형장의 t 미분(속도)과 공간 야코비안(F 밀기)의 정합성 검사.

  A) 정확 검사 -- dtfilm 을 0 으로, --dt_scale 을 켜고 **선형** 전달(bspline)을
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
from anchorflow.deform import skin, skin_with_jacobian, jacobian_of
from anchorflow.sitreg_warp import (SITRegWarp, FALLBACK_BOUND_444,
                                    cubic_bspline_g2p, max_control_point_value)

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
try:
    BND = max_control_point_value([4, 4, 4]) * hh
except Exception:
    BND = FALLBACK_BOUND_444 * hh
sidx32 = vox_anchor.knn(x32, lo32 - 0.5 * hh, hh, nn3, 16)
feat32 = torch.randn(NC, 8, device=dev)
ok = True


def chk(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}", flush=True)


def mknet(kind, wake=True, dtype=torch.float32):
    torch.manual_seed(1)
    n = ConvStepper(n_feat=8, hidden=32, depth=2, h=hh, scale=0.02,
                    skin_out=(kind == "skin"), dt_cond=True, dt_ref=HDT,
                    dt_scale=True,
                    rqs_dim=(P if kind == "rqs" else 0)).to(dev).to(dtype)
    with torch.no_grad():
        n.out.weight.normal_(0, 0.05)
        if kind == "rqs":
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
    out = net(None, feat, tau, GRID, cells=CELLS)
    dp = out[0]
    if kind == "rqs":
        qr = tri_spline.remap(q, lo, hh, nn3_, out[-1], a.bins)
        return SITRegWarp(BND, 2).apply(
            qr, dp, lambda z, c: z + cubic_bspline_g2p(z, lo, hh, nn3_, c))
    if kind == "bspline":
        return q + cubic_bspline_g2p(q, lo, hh, nn3_, dp)
    return skin(q, _gpos(lo, q.dtype), dp, out[1], out[2], sidx, hh)[0]


def _gpos(lo, dt_):
    return (torch.stack(torch.meshgrid(
        *[torch.arange(int(nn3[k]), device=dev, dtype=dt_) for k in range(3)],
        indexing="ij"), -1).reshape(-1, 3)) * hh + lo


def dphi_dt(net, kind, tau_v, q, lo, nn3_, sidx, feat):
    tau0 = torch.as_tensor(tau_v, device=dev, dtype=q.dtype)
    with fwAD.dual_level():
        r = field(net, kind, fwAD.make_dual(tau0, torch.ones_like(tau0)),
                  q, lo, nn3_, sidx, feat)
        pr, tg = fwAD.unpack_dual(r)
        return pr.clone(), (None if tg is None else tg.clone())


# ---- A) 정확 검사: 선형 전달 + 항등 FiLM 이면 dPhi/dt == 할선 -----------------
print("== A) 배선 정확 검사 (bspline 전달, 항등 dtfilm)")
netA = mknet("bspline", wake=False)
x2A, vA = dphi_dt(netA, "bspline", HDT, x32, lo32, nn3, sidx32, feat32)
secA = (x2A - x32) / HDT
relA = float((vA - secA).norm() / secA.norm().clamp_min(1e-30))
chk("dPhi/dt == (x2-x)/t (dp ∝ t 이므로 정확히 같아야)", relA < 1e-5,
    f"상대오차 {relA:.2e}")

# ---- B~D) float64 로 수치 검증 ------------------------------------------------
x64 = x32.double()
lo64 = lo32.double()
feat64 = feat32.double()
for kind in ("skin", "bspline", "rqs"):
    print(f"== 전달 {kind}")
    net = mknet(kind, wake=True, dtype=torch.float64)
    # B) t 미분
    x2, vd = dphi_dt(net, kind, HDT, x64, lo64, nn3, sidx32, feat64)
    eps = HDT * 1e-4
    with torch.no_grad():
        fd = (field(net, kind, torch.as_tensor(HDT + eps, device=dev,
                                               dtype=torch.float64),
                    x64, lo64, nn3, sidx32, feat64)
              - field(net, kind, torch.as_tensor(HDT - eps, device=dev,
                                                 dtype=torch.float64),
                      x64, lo64, nn3, sidx32, feat64)) / (2 * eps)
    r = float((vd - fd).norm() / fd.norm().clamp_min(1e-30))
    chk(f"{kind}: dPhi/dt == t 중심차분", r < 1e-5, f"상대오차 {r:.2e}")
    sec = (x2 - x64) / HDT
    print(f"      (참고) 할선과의 차이 "
          f"{100*float((vd-sec).norm()/sec.norm()):.1f}% -- 차분이 담지 못한 몫")

    # C/D) 공간 야코비안. rqs 는 셀 면 접선 불연속이 허용되므로 내부점만.
    if kind == "rqs":
        ti = (x64 - lo64) / hh
        frac = ti - ti.floor()
        keep = ((frac > 0.15) & (frac < 0.85)).all(-1)
        xs = x64[keep]
    else:
        xs = x64
    Ja = jacobian_of(lambda z: field(net, kind, torch.as_tensor(
        HDT, device=dev, dtype=torch.float64), z, lo64, nn3, sidx32, feat64),
        xs)
    with torch.no_grad():
        I3 = torch.eye(3, device=dev, dtype=torch.float64)
        sp = 1e-6
        cols = []
        for k in range(3):
            fp = field(net, kind, torch.as_tensor(HDT, device=dev,
                                                  dtype=torch.float64),
                       xs + sp * I3[k], lo64, nn3, sidx32, feat64)
            fm = field(net, kind, torch.as_tensor(HDT, device=dev,
                                                  dtype=torch.float64),
                       xs - sp * I3[k], lo64, nn3, sidx32, feat64)
            cols.append((fp - fm) / (2 * sp))
        Jfd = torch.stack(cols, -1)
    r = float((Ja - Jfd).norm() / Jfd.norm())
    det = torch.linalg.det(Ja)
    chk(f"{kind}: grad_x Phi == 공간 중심차분", r < 1e-4,
        f"상대오차 {r:.2e} (점 {xs.shape[0]})")
    chk(f"{kind}: det grad_x Phi > 0", bool((det > 0).all()),
        f"최소 {float(det.min()):.4f} 중앙 {float(det.median()):.4f}")
    if kind == "skin":
        with torch.no_grad():
            o = net(None, feat64, torch.as_tensor(HDT, device=dev,
                                                  dtype=torch.float64),
                    GRID, cells=CELLS)
            Js = skin_with_jacobian(xs, _gpos(lo64, torch.float64), o[0],
                                    o[1], o[2], sidx32, hh)[2]
        rr = float((Js - Jfd).norm() / Jfd.norm())
        chk("skin: 해석 야코비안 == 공간 중심차분", rr < 1e-4, f"상대오차 {rr:.2e}")

    # E) g2p_grad 로 민 F 와 변형장 야코비안으로 민 F 가 얼마나 다른가.
    #    야코비안은 자동미분이 필요하므로 no_grad 밖에서 부른다.
    Jfull = jacobian_of(lambda z: field(net, kind, torch.as_tensor(
        HDT, device=dev, dtype=torch.float64), z, lo64, nn3, sidx32,
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
