"""절 CM103 띠 근거: 정답 궤적에서 관찰되는 사면체 F 특이값·det 범위 (읽기만).

사용: python exe/tet2_gtsv.py --data <궤적.pt> --cell <셀 배수> --frames <프레임> --tag <씬> --out <json> [--margin 1.25]
정의:
- 격자 = 실행기와 같은 능면체 격자 (SX.lattice(cell·dx·√2, HZ_CUBE·...), 칸 모서리 = cell·dx), 칸마다 Kuhn 6 사면체 (fem_tet1 규칙: 칸 안 좌표 내림차순).
- 사면체 F (정답) = 그 프레임에 그 사면체 안에 있는 정답 입자 F 의 질량 가중 로그 신장 평균 (실행기 rt_gauss 와 같은 꼴: 우 신장 V diag(ln σ) Vᵀ 평균의 고윳값 exp),
  det = exp(tr). 입자 det ≤ 0 · 비유한 입자는 빼고 셈 (수를 기록).
- 띠: σ_lo = (사면체 σ 최소) / m, σ_hi = (사면체 σ 최대) x m, 장벽 활성 폭 d̂_lo = σ_min^정답 − σ_lo · d̂_hi = σ_hi − σ_max^정답 (정답 범위 안에서는 장벽 0),
  det 장벽 활성 d̂_J = 사면체 det 최소 (정답 최소 아래에서만 켜짐, 0 에서 +∞).
"""
import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
from anchorflow import simplex as SX  # noqa: E402
from anchorflow import deform_model as DM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--cell", type=float, required=True)
ap.add_argument("--frames", type=int, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--margin", type=float, default=1.25)
ap.add_argument("--region_density", action="store_true")
a = ap.parse_args()
dev = "cuda"
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
DM.REGION_DENSITY[0] = bool(a.region_density)
m = DM.traj_mass(d, dev).double().reshape(-1)
F = DM.traj_F(d, h, dev)
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
NODE_H = a.cell * DXG * math.sqrt(2.0)
lat = SX.lattice(NODE_H, SX.HZ_CUBE * NODE_H, device=dev, dtype=torch.float64)
T = min(a.frames + 1, int(d["x"].shape[0]))
Q = torch.tensor([1e-4, 1e-3, 0.5, 0.999, 0.9999], device=dev, dtype=torch.float64)
rows = []
G = dict(p_smin=math.inf, p_smax=-math.inf, p_jmin=math.inf, p_jmax=-math.inf, t_smin=math.inf, t_smax=-math.inf, t_jmin=math.inf, t_jmax=-math.inf,
         n_bad=0, n_inv=0, n_tet=0)
for f in range(T):
    x = d["x"][f].to(dev).double()
    Ff = F[f].to(dev).double().reshape(-1, 3, 3)
    ok = torch.isfinite(x).all(1) & torch.isfinite(Ff.reshape(-1, 9)).all(1)
    J = torch.linalg.det(Ff[ok])
    pos = J > 0
    nbad, ninv = int((~ok).sum()), int((~pos).sum())
    idx = torch.nonzero(ok).reshape(-1)[pos]
    Fp, mp, xp = Ff[idx], m[idx], x[idx]
    Uu, S, Vh = torch.linalg.svd(Fp)
    V = Vh.transpose(1, 2)
    LS = V @ torch.diag_embed(torch.log(S)) @ Vh                                           # 우 신장 로그 V ln Σ Vᵀ
    y = xp @ lat.Ai.T
    b = torch.floor(y)
    fr = y - b
    ordr = torch.sort(fr, dim=1, descending=True, stable=True).indices
    t = 2 * ordr[:, 0] + (ordr[:, 1] > ordr[:, 2]).long()
    bi = b.long()
    bi = bi - bi.min(0).values
    nb = bi.max(0).values + 1
    key = ((bi[:, 0] * nb[1] + bi[:, 1]) * nb[2] + bi[:, 2]) * 6 + t
    uk, inv = torch.unique(key, return_inverse=True)
    nt = uk.numel()
    W = torch.zeros(nt, device=dev, dtype=torch.float64).index_add_(0, inv, mp)
    M6 = torch.zeros(nt, 9, device=dev, dtype=torch.float64).index_add_(0, inv, mp[:, None] * LS.reshape(-1, 9))
    Mt = (M6 / W[:, None]).reshape(-1, 3, 3)
    Mt = 0.5 * (Mt + Mt.transpose(1, 2))
    ev = torch.linalg.eigvalsh(Mt)
    ts = torch.exp(ev)
    tj = torch.exp(ev.sum(1))
    ps = S
    r = dict(f=f, n_tet=nt, n_bad=nbad, n_inv=ninv, p_smin=float(ps.min()), p_smax=float(ps.max()), p_jmin=float(J[pos].min()), p_jmax=float(J[pos].max()),
             t_smin=float(ts.min()), t_smax=float(ts.max()), t_jmin=float(tj.min()), t_jmax=float(tj.max()),
             t_sq=[float(q) for q in torch.quantile(ts.reshape(-1)[torch.randperm(ts.numel(), device=dev)[:2_000_000]], Q)])
    rows.append(r)
    for k in ("p_smin", "p_jmin", "t_smin", "t_jmin"):
        G[k] = min(G[k], r[k])
    for k in ("p_smax", "p_jmax", "t_smax", "t_jmax"):
        G[k] = max(G[k], r[k])
    G["n_bad"] = max(G["n_bad"], nbad); G["n_inv"] = max(G["n_inv"], ninv); G["n_tet"] = max(G["n_tet"], nt)
    if f % 10 == 0:
        print(f"[{a.tag}] f{f} 사면체 {nt} σ [{r['t_smin']:.4f}, {r['t_smax']:.4f}] det [{r['t_jmin']:.4f}, {r['t_jmax']:.4f}] 입자 σ [{r['p_smin']:.4f}, {r['p_smax']:.4f}] 뒤집힘 {ninv} 비유한 {nbad}", flush=True)
mg = a.margin
slo, shi = G["t_smin"] / mg, G["t_smax"] * mg
band = dict(slo=slo, shi=shi, dlo=G["t_smin"] - slo, dhi=shi - G["t_smax"], ja=G["t_jmin"], margin=mg)
out = dict(tag=a.tag, data=a.data, cell=a.cell, frames=T - 1, dx=DXG, edge=a.cell * DXG, G=G, band=band, rows=rows)
json.dump(out, open(a.out, "w"), indent=1)
print(f"[{a.tag}] 정답 사면체 σ [{G['t_smin']:.4f}, {G['t_smax']:.4f}] det [{G['t_jmin']:.4f}, {G['t_jmax']:.4f}] (입자 σ [{G['p_smin']:.4f}, {G['p_smax']:.4f}]) "
      f"-> 띠 [{slo:.4f}, {shi:.4f}] d̂ {band['dlo']:.4f}·{band['dhi']:.4f} d̂_J {band['ja']:.4f}", flush=True)
print("T2B=%.6e,%.6e,%.6e,%.6e,%.6e" % (slo, shi, band["dlo"], band["dhi"], band["ja"]))
