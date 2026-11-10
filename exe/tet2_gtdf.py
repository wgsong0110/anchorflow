"""절 CM103 판 r 범위 근거: 정답 궤적의 프레임 간 사상 ΔF (프레임 f 위치 -> f+1 위치의 기울기) 특이값 분포 (읽기만).

사용: python exe/tet2_gtdf.py --data <궤적.pt> --cell <셀 배수> --frames <프레임> --tag <씬> --out <json> [--margin 1.25] [--q 1e-4]
정의:
- 격자 = 실행기와 같은 능면체 격자 (칸 모서리 cell·dx). 프레임 f 위치로 칸에 묶은 정답 입자의 (x_f, x_{f+1}) 에 질량 가중 아핀 맞춤
  x_{f+1} ≈ ΔF x_f + c (ΔF = C_yx C_xx⁻¹, 무게중심 뺀 공분산) -- 칸 단위 (사면체 6 개 묶음; 사면체당 정답 입자가 몇 개뿐이라 사면체 맞춤은 불안정),
  입자 10 개 이상·C_xx 조건수 1e3 이하·비유한 없음인 칸만.
- 범위: 모든 프레임·칸의 ΔF 특이값 분포의 q · 1 − q 분위 (기본 1e-4) 를 로그 여유 m 배: s_lo = exp(m ln q_lo), s_hi = exp(m ln q_hi).
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
ap.add_argument("--q", type=float, default=1e-4)
ap.add_argument("--region_density", action="store_true")
a = ap.parse_args()
dev = "cuda"
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
DM.REGION_DENSITY[0] = bool(a.region_density)
m = DM.traj_mass(d, dev).double().reshape(-1)
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
NODE_H = a.cell * DXG * math.sqrt(2.0)
lat = SX.lattice(NODE_H, SX.HZ_CUBE * NODE_H, device=dev, dtype=torch.float64)
T = min(a.frames + 1, int(d["x"].shape[0]))
allS, rows = [], []
for f in range(T - 1):
    x0 = d["x"][f].to(dev).double(); x1 = d["x"][f + 1].to(dev).double()
    ok = torch.isfinite(x0).all(1) & torch.isfinite(x1).all(1)
    x0, x1, w = x0[ok], x1[ok], m[ok]
    b = torch.floor(x0 @ lat.Ai.T).long(); b = b - b.min(0).values; nb = b.max(0).values + 1
    key = (b[:, 0] * nb[1] + b[:, 1]) * nb[2] + b[:, 2]
    uk, inv = torch.unique(key, return_inverse=True); n = uk.numel()
    W = torch.zeros(n, device=dev, dtype=torch.float64).index_add_(0, inv, w)
    cnt = torch.zeros(n, device=dev).index_add_(0, inv, torch.ones_like(w, dtype=torch.float32))
    mx = torch.zeros(n, 3, device=dev, dtype=torch.float64).index_add_(0, inv, w[:, None] * x0) / W[:, None]
    my = torch.zeros(n, 3, device=dev, dtype=torch.float64).index_add_(0, inv, w[:, None] * x1) / W[:, None]
    dx = x0 - mx[inv]; dy = x1 - my[inv]
    Cxx = torch.zeros(n, 9, device=dev, dtype=torch.float64).index_add_(0, inv, (w[:, None, None] * dx[:, :, None] * dx[:, None, :]).reshape(-1, 9)).reshape(-1, 3, 3)
    Cyx = torch.zeros(n, 9, device=dev, dtype=torch.float64).index_add_(0, inv, (w[:, None, None] * dy[:, :, None] * dx[:, None, :]).reshape(-1, 9)).reshape(-1, 3, 3)
    ev = torch.linalg.eigvalsh(Cxx)
    good = (cnt >= 10) & (ev[:, 0] > 0) & (ev[:, 2] / ev[:, 0].clamp_min(1e-300) < 1e3)
    DF = Cyx[good] @ torch.linalg.inv(Cxx[good])
    S = torch.linalg.svdvals(DF)
    S = S[torch.isfinite(S).all(1)]
    allS.append(S.reshape(-1).float().cpu())
    rows.append(dict(f=f, ncell=n, ngood=int(good.sum()), smin=float(S.min()), smax=float(S.max()), q=[float(v) for v in torch.quantile(S.reshape(-1)[:2_000_000], torch.tensor([1e-3, 0.5, 0.999], device=dev, dtype=S.dtype))]))
    if f % 10 == 0:
        print(f"[{a.tag}] f{f}->{f + 1} 칸 {n} (맞춤 {int(good.sum())}) ΔF σ [{float(S.min()):.4f}, {float(S.max()):.4f}]", flush=True)
A = torch.cat(allS).double()
idx = torch.randperm(A.numel())[:16_000_000]
ql, qh = [float(v) for v in torch.quantile(A[idx], torch.tensor([a.q, 1 - a.q], dtype=torch.float64))]
slo, shi = math.exp(a.margin * math.log(ql)), math.exp(a.margin * math.log(qh))
out = dict(tag=a.tag, cell=a.cell, edge=a.cell * DXG, frames=T - 1, n=int(A.numel()), amin=float(A.min()), amax=float(A.max()), qlo=ql, qhi=qh, q=a.q, margin=a.margin,
           band3=dict(slo=slo, shi=shi), rows=rows)
json.dump(out, open(a.out, "w"), indent=1)
print(f"[{a.tag}] 정답 프레임 간 ΔF σ: 전체 [{float(A.min()):.4f}, {float(A.max()):.4f}], 분위 {a.q:g}/{1 - a.q:g} = {ql:.4f}/{qh:.4f} -> 범위 [{slo:.4f}, {shi:.4f}] (로그 여유 {a.margin})", flush=True)
