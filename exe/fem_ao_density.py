"""절 AO 공통 설정 근거: 씬마다 입자 밀도 (셀 배수별 점유 칸당 입자 수) 와 정답 최대 속력·프레임 간격.
셀 = 절 AN 격자 (simplex.lattice 회전 정육면체, 칸 변 = 셀 배수 x 정답 dx). 출력 json 한 줄씩.
사용: python exe/fem_ao_density.py <이름>:<데이터.pt> ... > out.jsonl"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                         # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
for arg in sys.argv[1:]:
    nm, path = arg.split(":", 1)
    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = d["cfg"]
    x = d["x"][0].to(dev).double()
    N = x.shape[0]
    DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
    h = float(cfg["frame_dt"])
    # 이웃 거리 중앙값 (무작위 2 만 입자의 가장 가까운 입자)
    g = torch.Generator(device="cpu").manual_seed(0)
    idx = torch.randperm(N, generator=g)[:20000].to(dev)
    dmin = []
    for ch in idx.split(2000):
        dd = torch.cdist(x[ch], x)
        dd[torch.arange(ch.numel(), device=dev), ch] = float("inf")
        dmin.append(dd.min(1).values)
    nn = torch.cat(dmin)
    # 정답 최대 속력 (프레임 차분, 프레임마다 입자 최대 -> 그 99 백분위·최대), 프레임 안 최대 이동
    X = d["x"].float()
    vmax = []
    for i in range(X.shape[0] - 1):
        dx = (X[i + 1] - X[i]).to(dev)
        ok = torch.isfinite(dx).all(-1)
        sp = dx[ok].norm(dim=-1) / h
        vmax.append([float(sp.max()), float(torch.quantile(sp[torch.randperm(sp.numel(), device=dev)[:200000]], 0.99))])
    rows = {}
    for c in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        nh = c * DXG * math.sqrt(2.0)
        lat = SX.lattice(nh, SX.HZ_CUBE * nh, device=dev, dtype=torch.float64)
        y = x @ lat.Ai.T
        b = torch.floor(y - 0.5).long()
        b = b - b.min(0).values
        mx = b.max(0).values + 1
        cid = (b[:, 0] * mx[1] + b[:, 1]) * mx[2] + b[:, 2]
        _, cnt = torch.unique(cid, return_counts=True)
        cnt = cnt.double()
        rows[str(c)] = dict(cell_m=float(lat.A[:, 0].norm()), n_cells=int(cnt.numel()), mean_ppc=float(cnt.mean()), median_ppc=float(cnt.median()),
                            p10_ppc=float(torch.quantile(cnt, 0.1)), frac_ge8=float((cnt >= 8).double().mean()), frac_le1=float((cnt <= 1).double().mean()),
                            n_q=int(cnt.numel() * 8))
    print(json.dumps(dict(scene=nm, N=N, dx_gt=DXG, frame_dt=h, substep_dt=float(cfg["substep_dt"]), material=cfg.get("material"), E=cfg.get("E"), density=cfg.get("density"),
                          nn_median=float(nn.median()), nn_p90=float(torch.quantile(nn, 0.9)), vmax_frame_max=max(v[0] for v in vmax), v99_frame_max=max(v[1] for v in vmax),
                          cells=rows)), flush=True)
