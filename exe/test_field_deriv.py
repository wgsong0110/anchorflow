"""변형장을 t 로 미분해 얻은 속도와, 변형장 야코비안으로 민 F 의 정합성 검사.

검사 대상은 개념 셋이다.
  1) dPhi/dt (순방향 AD) == t 중심차분  -- 속도가 진짜 변형장의 시간 미분인가
  2) grad_x Phi (해석/자동미분) == 공간 중심차분  -- F 를 미는 야코비안이 맞나
  3) g2p_grad 로 민 F 와 grad_x Phi 로 민 F 가 **다르다**  -- 지금까지 F 가
     실제 사상과 어긋나 있었다는 증거 (RQS 셀 내부 항·스키닝 가중치 누락)

  python exe/test_field_deriv.py --traj traj_h2/mic_clayC_t_s400706.pt
"""
import argparse
import math

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
ap.add_argument("--n_pts", type=int, default=4000)
ap.add_argument("--vox_res", type=int, default=32)
ap.add_argument("--bins", type=int, default=8)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
d = torch.load(a.traj, map_location="cpu", weights_only=False)
cfg = d["cfg"]
H_DT = float(cfg["frame_dt"])
X = d["x"].float()
sel = torch.randperm(X.shape[1],
                     generator=torch.Generator().manual_seed(0)
                     )[:a.n_pts].sort().values
x = X[a.t0, sel].to(dev)
v = ((X[a.t0, sel] - X[a.t0 - 1, sel]) / H_DT).to(dev)
F0 = torch.eye(3, device=dev).expand(sel.numel(), 3, 3).contiguous()
mass = torch.full((sel.numel(),), float(cfg["density"]), device=dev)
lo, hh, nn3 = vox_anchor.grid_for(x, a.vox_res ** 3)
hh = float(hh)
gpos = (torch.stack(torch.meshgrid(
    *[torch.arange(int(nn3[k]), device=dev, dtype=x.dtype) for k in range(3)],
    indexing="ij"), -1).reshape(-1, 3)) * hh + lo
M = gpos.shape[0]
NC = int((nn3[0] - 1) * (nn3[1] - 1) * (nn3[2] - 1))
P = tri_spline.n_params(a.bins)
try:
    BND = max_control_point_value([4, 4, 4]) * hh
except Exception:
    BND = FALLBACK_BOUND_444 * hh
ok = True


def chk(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}", flush=True)


def mknet(rqs):
    torch.manual_seed(1)
    n = ConvStepper(n_feat=8, hidden=32, depth=2, h=hh, scale=0.02,
                    skin_out=not rqs, dt_cond=True, dt_ref=H_DT,
                    dt_scale=True,
                    rqs_dim=(P if rqs else 0)).to(dev)
    with torch.no_grad():          # 학습된 상태를 흉내낸다 (0 이면 볼 게 없다)
        n.out.weight.normal_(0, 0.05)
        last = [m for m in n.dtfilm.mlp.modules()
                if isinstance(m, torch.nn.Linear)][-1]
        last.weight.normal_(0, 0.05); last.bias.normal_(0, 0.05)
        n.dtfilm._cache.clear()
        if rqs:
            n.out_rqs.weight.normal_(0, 0.05)
    return n


cells = tuple(int(nn3[k]) - 1 for k in range(3))
feat = torch.randn(NC, 8, device=dev)
sidx = vox_anchor.knn(x, lo - 0.5 * hh, hh, nn3, 16)


def field(net, tau, q=None, rqs=False):
    """변형장 Phi_tau(q). tau 는 텐서(dual 가능), q 기본은 x."""
    qq = x if q is None else q
    out = net(None, feat, tau, tuple(int(c) for c in nn3), cells=cells)
    dp = out[0]
    if rqs:
        th = out[-1]
        qr = tri_spline.remap(qq, lo, hh, nn3, th, a.bins)
        return SITRegWarp(BND, 2).apply(
            qr, dp, lambda z, c: z + cubic_bspline_g2p(z, lo, hh, nn3, c))
    return skin(qq, gpos, dp, out[1], out[2], sidx, hh)[0]


for RQS in (False, True):
    tag = "rqs" if RQS else "skin"
    net = mknet(RQS)
    print(f"== 전달 {tag}")
    tau0 = torch.as_tensor(H_DT, device=dev, dtype=x.dtype)

    # 1) dPhi/dt : 순방향 AD vs 중심차분
    with fwAD.dual_level():
        q = field(net, fwAD.make_dual(tau0, torch.ones_like(tau0)), rqs=RQS)
        x2, vdot = fwAD.unpack_dual(q)
        x2 = x2.clone()
        vdot = vdot.clone()
    eps = H_DT * 1e-3
    with torch.no_grad():
        fd = (field(net, tau0 + eps, rqs=RQS)
              - field(net, tau0 - eps, rqs=RQS)) / (2 * eps)
    rel = float((vdot - fd).norm() / fd.norm().clamp_min(1e-20))
    chk(f"{tag}: dPhi/dt == 중심차분", rel < 2e-3,
        f"상대오차 {rel:.3e}, |v| 중앙 {float(vdot.norm(dim=-1).median()):.4f}")
    # 차분(x2-x)/h 와는 **다르다** -- 그게 이번 변경의 요점이다
    sec = (x2 - x) / H_DT
    dif = float((vdot - sec).norm() / sec.norm().clamp_min(1e-20))
    print(f"      (참고) 할선 (x2-x)/h 와의 차이 {100*dif:.1f}%")

    # 2) grad_x Phi : 자동미분 vs 공간 중심차분
    Ja = jacobian_of(lambda z: field(net, tau0, q=z, rqs=RQS), x)
    with torch.no_grad():
        I3 = torch.eye(3, device=dev)
        sp = 1e-4 * hh
        cols = [(field(net, tau0, q=x + sp * I3[k], rqs=RQS)
                 - field(net, tau0, q=x - sp * I3[k], rqs=RQS)) / (2 * sp)
                for k in range(3)]
        Jfd = torch.stack(cols, -1)
    rel = float((Ja - Jfd).norm() / Jfd.norm())
    chk(f"{tag}: grad_x Phi == 공간 중심차분", rel < 5e-3,
        f"상대오차 {rel:.3e}, det 중앙 {float(torch.linalg.det(Ja).median()):.4f}")
    chk(f"{tag}: det grad_x Phi > 0", bool((torch.linalg.det(Ja) > 0).all()),
        f"최소 {float(torch.linalg.det(Ja).min()):.3e}")
    if not RQS:
        with torch.no_grad():
            o = net(None, feat, tau0, tuple(int(c) for c in nn3), cells=cells)
            Js = skin_with_jacobian(x, gpos, o[0], o[1], o[2], sidx, hh)[2]
        r = float((Js - Jfd).norm() / Jfd.norm())
        chk("skin: 해석 야코비안 == 공간 중심차분", r < 5e-3, f"상대오차 {r:.3e}")

    # 3) g2p_grad 로 민 F 와 변형장 야코비안으로 민 F 의 차이
    with torch.no_grad():
        du = x2 - x
        _m, duI, vI, info, _fr = phys_resid.p2g_increment(
            x, du, v, mass, int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0)))
        gu = phys_resid.g2p_grad(x, duI, info, int(cfg["n_grid"]))
        Fg = (torch.eye(3, device=dev) + gu) @ F0
        Fj = Jfd @ F0
    r = float((Fg - Fj).norm()
              / (Fj - torch.eye(3, device=dev)).norm().clamp_min(1e-20))
    print(f"      g2p_grad 로 민 F 와 변형장 야코비안으로 민 F 의 차이: "
          f"증분 대비 {100*r:.1f}%  "
          f"(det 중앙 g2p {float(torch.linalg.det(Fg).median()):.4f} / "
          f"장 {float(torch.linalg.det(Fj).median()):.4f})")

print("ALL-OK" if ok else "SOME-FAIL")
raise SystemExit(0 if ok else 1)
