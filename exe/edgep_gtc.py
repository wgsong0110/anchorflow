"""절 CM105 변 길이 상한 c 근거: 정답 궤적을 실행기 노드 격자에 사상한 프레임 간 노드 변 늘어남 (읽기만).

사용: python exe/edgep_gtc.py --data <궤적.pt> --cell <셀 배수> --frames <프레임> --tag <씬> --out <json> [--margin 1.25] [--region_density]
정의 (실행기와 같은 사상):
- 격자 = 실행기와 같은 능면체 격자 (exe/fem_an_rt_sim.py: SX.lattice(cell·dx·√2, HZ_CUBE·…), 노드 = 격자 정수점, 축 a 의 변 길이 L0_a = |A e_a|).
- 프레임 f 정답 입자 위치 x_f 로 이차 B-스플라인 가중 (실행기 rt_coord·bwf 와 같은 식: b = floor(Ai x − ½), 27 노드) 을 잡고,
  노드 변위 d_i = Σ_p w_ip m_p (x_{f+1,p} − x_{f,p}) / Σ_p w_ip m_p (실행기 P2G 와 같은 질량 가중 평균), 활성 노드 = Σ_p w_ip m_p > 0 (실행기 mn > 0).
- 변 = 축 이웃 활성 노드 쌍 (i, i + e_a), 걸음 끝 길이 = |A e_a + d_{i+e_a} − d_i| (프레임당 걸음 하나 = 실행기 걸음 시작 격자에서 끝까지의 변).
  늘어남 비 r = 길이 / L0_a. 모든 프레임·변의 최대 r_max (+ 분위).
- 상한 c = r_max 를 로그 여유 m 배 (절 CM103 범위와 같은 규칙): c = exp(m ln r_max), 기본 m = 1.25.
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
DM.REGION_DENSITY[0] = bool(a.region_density)
m = DM.traj_mass(d, dev).double().reshape(-1)
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
NODE_H = a.cell * DXG * math.sqrt(2.0)
lat = SX.lattice(NODE_H, SX.HZ_CUBE * NODE_H, device=dev, dtype=torch.float64)
A, Ai = lat.A, lat.Ai
L0 = [float(torch.linalg.norm(A[:, q])) for q in range(3)]
T = min(a.frames + 1, int(d["x"].shape[0]))
QS = [0.5, 0.9, 0.99, 0.999, 0.9999]
rows, samp = [], []
rmax_all, rmin_all, nedge_all = 0.0, 1e30, 0
for f in range(T - 1):
    x0 = d["x"][f].to(dev).double(); x1 = d["x"][f + 1].to(dev).double()
    ok = torch.isfinite(x0).all(1) & torch.isfinite(x1).all(1)
    x0, x1, w = x0[ok], x1[ok], m[ok]
    u = x0 @ Ai.T
    b = torch.floor(u - 0.5)
    fx = u - b
    b = b.long()
    bmin = b.min(0).values
    b = b - bmin
    nb = (b.max(0).values + 3).tolist()
    W1 = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
    nn = nb[0] * nb[1] * nb[2]
    Wn = torch.zeros(nn, device=dev, dtype=torch.float64)
    Dn = torch.zeros(nn, 3, device=dev, dtype=torch.float64)
    du = x1 - x0
    for o in range(27):
        i, j, k = o // 9, (o // 3) % 3, o % 3
        ww = W1[i][:, 0] * W1[j][:, 1] * W1[k][:, 2] * w
        nid = ((b[:, 0] + i) * nb[1] + (b[:, 1] + j)) * nb[2] + (b[:, 2] + k)
        Wn.index_add_(0, nid, ww)
        Dn.index_add_(0, nid, ww[:, None] * du)
    act = Wn > 0
    dn = torch.where(act[:, None], Dn / Wn.clamp_min(1e-300)[:, None], torch.zeros_like(Dn))
    idx = torch.arange(nn, device=dev)
    ia = [idx // (nb[1] * nb[2]), (idx // nb[2]) % nb[1], idx % nb[2]]
    st = [nb[1] * nb[2], nb[2], 1]
    rs = []
    for q in range(3):
        e = (ia[q] + 1 < nb[q]) & act
        n_ = idx[e]
        j_ = n_ + st[q]
        e2 = act[j_]
        n_, j_ = n_[e2], j_[e2]
        r = A[:, q][None, :] + dn[j_] - dn[n_]
        rs.append(torch.linalg.norm(r, dim=1) / L0[q])
    R = torch.cat(rs)
    nedge_all += R.numel()
    rmx, rmn = float(R.max()), float(R.min())
    rmax_all, rmin_all = max(rmax_all, rmx), min(rmin_all, rmn)
    sel = R if R.numel() <= 200_000 else R[torch.randint(0, R.numel(), (200_000,), device=dev)]
    samp.append(sel.float().cpu())
    rows.append(dict(f=f, nnode=int(act.sum()), nedge=int(R.numel()), rmax=rmx, rmin=rmn, q=[float(v) for v in torch.quantile(sel.double(), torch.tensor(QS, device=dev, dtype=torch.float64))]))
    if f % 10 == 0:
        print(f"[{a.tag}] f{f}->{f + 1} 활성 노드 {int(act.sum())} 변 {R.numel()} 늘어남 비 [{rmn:.4f}, {rmx:.4f}]", flush=True)
S = torch.cat(samp).double()
qv = [float(v) for v in torch.quantile(S[torch.randperm(S.numel())[:16_000_000]], torch.tensor(QS, dtype=torch.float64))]
c = math.exp(a.margin * math.log(rmax_all))
fm = max(rows, key=lambda r: r["rmax"])
out = dict(tag=a.tag, cell=a.cell, dxg=DXG, L0=L0, A=[float(v) for v in A.reshape(-1)], frames=T - 1, nedge=nedge_all, rmax=rmax_all, rmin=rmin_all,
           frame_of_max=fm["f"], q=dict(zip([str(q) for q in QS], qv)), margin=a.margin, c=c, rows=rows)
json.dump(out, open(a.out, "w"), indent=1)
print(f"[{a.tag}] 정답 노드 변 늘어남 비: 최대 {rmax_all:.4f} (f{fm['f']}), 최소 {rmin_all:.4f}, 분위 " + " ".join(f"{q:g}:{v:.4f}" for q, v in zip(QS, qv))
      + f" -> 상한 c = exp({a.margin} ln r_max) = {c:.4f} (L0 {L0[0]:.5f} m)", flush=True)
