"""롤아웃 덤프(AF_ROLL_DUMP, AF_ROLL_DUMP_F=1) 여러 개와 정답 궤적의 프레임별 물리량 분포 -> json (대화형 보기용).

물리량 (입자마다): 속력 |v|, det F, Green 변형률 크기 |E| (E = (FᵀF − I)/2), 탄성에너지 밀도 ψ (고정 공회전, cfg E·ν).
단위 세 가지:
  입자  -- 입자 하나하나
  셀    -- 모델의 사면체 셀 (train_deform 과 같은 격자: grid_for_nodes(h_fix=node_h, hz_ratio), Kuhn 분할) 안 입자 평균
  격자  -- MPM 격자 칸 (dx = grid_lim / n_grid) 안 입자 평균
분포 = 고정 구간 히스토그램 (구간은 모든 실행·프레임 공통; ψ 는 log10).

  python exe/dist_frames.py --gt lbpt/test_low/L000.pt --run 망원래=roll_a.pt --run 망Adam=roll_b.pt --cfg lbdata/config.json --out dist.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import simplex as SX  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True)
ap.add_argument("--run", action="append", required=True, help="이름=롤아웃 덤프")
ap.add_argument("--cfg", required=True)
ap.add_argument("--node_h", type=float, default=0.028284)
ap.add_argument("--hz_ratio", type=float, default=1.0 / 6 ** 0.5)
ap.add_argument("--bins", type=int, default=60)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open(a.cfg))
E_, nu_ = float(cfg["E"]), float(cfg["nu"])
mu, lam = E_ / (2 * (1 + nu_)), E_ * nu_ / ((1 + nu_) * (1 - 2 * nu_))
dxg = float(cfg.get("grid_lim", 2.0)) / int(cfg["n_grid"])


def quantities(v, F):
    F = F.float(); v = v.float()
    J = torch.linalg.det(F)
    C = F.transpose(1, 2) @ F
    Eg = 0.5 * (C - torch.eye(3, device=F.device))
    en = Eg.flatten(1).norm(dim=1)
    U, S, Vh = torch.linalg.svd(F)
    psi = mu * ((S - 1) ** 2).sum(1) + 0.5 * lam * (J - 1) ** 2
    return {"속력": v.norm(dim=1), "det F": J, "Green 변형률 |E|": en,
            "탄성에너지 밀도 log10 ψ": torch.log10(psi.clamp_min(1e-6))}


def group_mean(key, val):
    _, inv = torch.unique(key, return_inverse=True)
    n = int(inv.max()) + 1
    s = torch.zeros(n, device=val.device).index_add_(0, inv, val)
    c = torch.zeros(n, device=val.device).index_add_(0, inv, torch.ones_like(val))
    return s / c


def levels(x, q):
    lo, lat, nn = SX.grid_for_nodes(x, 32, hz_ratio=a.hz_ratio, h_fix=a.node_h)
    _, _, (ci, rank, perm) = SX.locate(x, lo, lat, nn)
    pid = perm[:, 0] * 3 + perm[:, 1]                       # 순열 6 개 -> 사면체 번호
    nnl = [int(nn[k]) for k in range(3)]
    cell = ((ci[:, 0] * nnl[1] + ci[:, 1]) * nnl[2] + ci[:, 2]) * 9 + pid
    gi = (x / dxg).floor().long()
    grid = (gi[:, 0] * 4096 + gi[:, 1]) * 4096 + gi[:, 2]
    out = {}
    for k, val in q.items():
        val = torch.nan_to_num(val, nan=0.0, posinf=1e6, neginf=-1e6)
        out[k] = {"입자": val, "셀": group_mean(cell, val), "격자": group_mean(grid, val)}
    return out


# 정답
g = torch.load(a.gt, map_location="cpu", weights_only=False)
runs = {"정답": dict(x=g["x"][1:], v=g["v"][1:], F=g["F"][1:])}
for r in a.run:
    nm, p = r.split("=", 1)
    d = torch.load(p, map_location="cpu", weights_only=False)
    runs[nm] = dict(x=d["pred"], v=d["v"], F=d["F"])
T = min(r["x"].shape[0] for r in runs.values())
print(f"[분포] 프레임 {T}, 실행 {list(runs)}", flush=True)

raw = {}                                                        # 이름 -> 프레임 -> 물리량 -> 단위 -> 값
for nm, r in runs.items():
    raw[nm] = []
    for t in range(T):
        x = r["x"][t].to(dev).float(); v = r["v"][t].to(dev).float(); F = r["F"][t].to(dev).float().reshape(-1, 3, 3)
        lv = levels(x, quantities(v, F))
        raw[nm].append({k: {u: vv.cpu().numpy() for u, vv in d_.items()} for k, d_ in lv.items()})
    print(f"  {nm} 끝", flush=True)

QK = list(raw["정답"][0].keys()); UN = ["입자", "셀", "격자"]
edges = {}
for k in QK:
    for u in UN:
        allv = np.concatenate([raw[nm][t][k][u] for nm in raw for t in range(0, T, 4)])
        lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
        if hi <= lo:
            hi = lo + 1e-6
        edges[(k, u)] = np.linspace(lo, hi, a.bins + 1)
out = {"frames": T, "quantities": QK, "units": UN, "runs": list(raw), "edges": {}, "hist": {}, "stats": {}}
for (k, u), e in edges.items():
    out["edges"][f"{k}|{u}"] = [float(f"{q:.5g}") for q in e]
for nm in raw:
    for t in range(T):
        for k in QK:
            for u in UN:
                val = raw[nm][t][k][u]
                e = edges[(k, u)]
                h, _ = np.histogram(np.clip(val, e[0], e[-1]), bins=e)
                out["hist"][f"{nm}|{t}|{k}|{u}"] = (h / max(len(val), 1)).round(5).tolist()
                out["stats"][f"{nm}|{t}|{k}|{u}"] = [float(f"{np.mean(val):.4g}"), float(f"{np.median(val):.4g}"),
                                                     float(f"{np.percentile(val, 1):.4g}"), float(f"{np.percentile(val, 99):.4g}"),
                                                     int(len(val))]
json.dump(out, open(a.out, "w"), ensure_ascii=False)
print(f"[저장] {a.out}", flush=True)
