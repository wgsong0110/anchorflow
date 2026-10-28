"""절 M 진단: 2 차 B-스플라인 명시적 판의 노드 국소 안정 걸음 2/ω (프레임 0, 절 L.1 진단과 같은 꼴).
  노드 질량 m_n = Σ_p w_n(x_p) m_p (P2G 그대로)
  입자 적분점 : ω_n² = Σ_p V_p (λ+2μ) |∇w_n(x_p)|² / m_n          (∇w 가 w 와 함께 0 으로 -- |∇w|²/w 유한)
  셀 대표 K 점 : ω_n² = Σ_c V_c (λ+2μ) |ḡ_{n-b_c}|² / m_n           (ḡ = 셀 안 K 점 ∇w 평균 -- 셀 입자가 노드에서 멀어 w → 0 이어도 ḡ 는 그대로)
--cfg_bc 이면 묶인 노드 (cuboid 상자 안) 는 뺀다. 출력: 분위 (0, 1e-4, 1e-3, 1e-2, 0.1, 0.5) 와 필요 N = h / (2/ω)_min, 질량 비 분위.
사용: python exe/fem_bs_diag.py --data .. --node_h .. [--region_density] [--cfg_bc]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_bs_cuda as BSC                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--node_h", type=float, default=0.028284)
ap.add_argument("--region_density", action="store_true")
ap.add_argument("--cfg_bc", action="store_true")
a = ap.parse_args()
dev = "cuda"
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
DM.REGION_DENSITY[0] = bool(a.region_density)
m = DM.traj_mass(d, dev).double()
rho = d["_rho"].to(dev).double() if a.region_density else None
vol = (m / rho) if a.region_density else (m / float(cfg["density"]))
x = d["x"][0].to(dev).double()
mu, lam = PR.lame(cfg["E"], cfg["nu"])
kap = lam + 2 * mu
lo, lat, nn = SX.grid_for_nodes(x, 32, margin=0.1, hz_ratio=SX.HZ_CUBE, h_fix=a.node_h)
lat = SX.Lat(lat.A.double(), lat.Ai.double(), lat.s, lat.hz)
lo = lo.double()
flat, w, dwdy = SX.bspline27(x, lo, lat, nn)
P = int(nn[0] * nn[1] * nn[2])
mn = torch.zeros(P, device=dev, dtype=torch.float64).index_add_(0, flat.reshape(-1), (w * m[:, None]).reshape(-1))
gx = dwdy @ lat.Ai                                                                   # [N,27,3]
kd_p = torch.zeros(P, device=dev, dtype=torch.float64).index_add_(0, flat.reshape(-1), (vol[:, None] * kap * (gx ** 2).sum(-1)).reshape(-1))
base = flat[:, 0]                                                                    # o = (0,0,0) 노드 = 셀 번호
Vc = torch.zeros(P, device=dev, dtype=torch.float64).index_add_(0, base, vol)
off = flat[0] - flat[0, 0]                                                           # 27 노드 평탄 오프셋 (모든 입자 같다)
fix = torch.zeros(P, dtype=torch.bool, device=dev)
if a.cfg_bc:
    ijk = torch.stack(torch.meshgrid(*[torch.arange(int(nn[k]), device=dev) for k in range(3)], indexing="ij"), -1).reshape(-1, 3).double()
    npos = (ijk @ lat.A.T + lo).float()
    fx, _ = RT.cfg_bc_nodes(cfg, [], npos, torch.zeros(1, 4, dtype=torch.long, device=dev), torch.zeros(1, 4, device=dev), m[:1].float(), 0.0, h)
    if fx is not None:
        fix = fx
ok0 = (mn > 0) & ~fix
print(f"[진단] 입자 {x.shape[0]}, 노드 (질량 > 0, 묶임 제외) {int(ok0.sum())}, dx {float(lat.A[:, 0].norm()):.5f}, λ+2μ {kap:.4g}, h {h:.5f}", flush=True)
qq = torch.tensor([0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5], dtype=torch.float64)


def rep(name, kd):
    ok = ok0 & (kd > 0)
    dtl = 2.0 / (kd[ok] / mn[ok]).sqrt()
    qs = torch.quantile(dtl.float().cpu(), qq.float())
    print(f"  {name}: 2/ω 분위 (0, 1e-4, 1e-3, 1e-2, 0.1, 0.5) = " + ", ".join(f"{float(q):.2e}" for q in qs)
          + f" s -> 필요 N (최소) {h / float(qs[0]):.0f}, (1e-4 분위) {h / float(qs[1]):.0f}", flush=True)
    k = int(torch.argmin(dtl))
    idx = ok.nonzero().squeeze(1)[k]
    print(f"     최소 노드: 질량 {float(mn[idx]):.3e} (질량 중앙 {float(mn[ok].median()):.3e}), ω²·m {float(kd[idx]):.3e} (중앙 {float(kd[ok].median()):.3e})", flush=True)


rep("입자 적분점", kd_p)
for K in (1, 8):
    gK = BSC.cell_grad_table(lat.Ai, K).to(dev)                                        # [27,3]
    kd_c = torch.zeros(P, device=dev, dtype=torch.float64)
    occ = (Vc > 0).nonzero().squeeze(1)
    for o in range(27):
        kd_c.index_add_(0, occ + off[o], Vc[occ] * kap * float((gK[o] ** 2).sum()))
    rep(f"셀 대표 K {K}", kd_c)
r = (mn[ok0] / mn[ok0].median()).float().cpu()
print("  노드 질량 / 중앙 분위 (0, 1e-4, 1e-3, 1e-2): " + ", ".join(f"{float(q):.2e}" for q in torch.quantile(r, qq[:4].float())), flush=True)
print("DIAG_DONE", flush=True)
