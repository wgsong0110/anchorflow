"""주어진 입력만으로 낼 수 있는 기준선들을 직접 잰다.

학생이 쓰는 것과 같은 격자/전달로 (1) 관성 dp=v*dt, (2) 셀평균 속도를 격자로
옮긴 관성, (3) 최적 스칼라 배율 관성, (4) 강체 평행이동, (5) 자유 dp 하한 을
재서, 국소 함수가 얼마나 줄일 수 있는지 본다.
"""
from __future__ import annotations
import argparse, glob, os
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--vox_res", type=int, default=32)
ap.add_argument("--n_pts", type=int, default=8000)
ap.add_argument("--n_traj", type=int, default=8)
ap.add_argument("--n_win", type=int, default=6)
a = ap.parse_args()

dev = "cuda:0"
from anchorflow import trilinear as TRI, vox_anchor

files = sorted(glob.glob(os.path.join(a.data, "*.pt")))[:a.n_traj]
acc = {k: 0.0 for k in ("정지", "관성", "관성격자", "관성최적", "평행이동", "자유")}
cnt = 0
for f in files:
    d = torch.load(f, map_location=dev, weights_only=False)
    X = d["x"].float()
    N_FULL = X.shape[1]
    gsel = torch.arange(0, N_FULL, max(1, N_FULL // a.n_pts), device=dev)[:a.n_pts]
    EXT = float((X[0].max(0).values - X[0].min(0).values).norm())
    T = X.shape[0]
    for t0 in torch.linspace(2, T - 3, a.n_win).long().tolist():
        x = X[t0][gsel]
        xp = X[t0 - 1][gsel]
        gt = X[t0 + 1][gsel]
        u = gt - x                                  # 정답 변위
        v = x - xp                                  # 한 프레임 변위 (=v*dt)

        def e(p):
            return float(((p - u) ** 2).sum(-1).mean()) / EXT ** 2

        lo, hh, nn3 = vox_anchor.grid_for(x, a.vox_res ** 3)
        flat, w = TRI.corners(x, lo, hh, nn3)
        M = int(nn3[0] * nn3[1] * nn3[2])
        # 셀평균 속도를 격자로 흩뿌렸다가 다시 모은다 (학생이 쓰는 경로)
        num = torch.zeros(M, 3, device=dev)
        den = torch.zeros(M, 1, device=dev)
        num.index_add_(0, flat.reshape(-1), (w.unsqueeze(-1) * v.unsqueeze(1)).reshape(-1, 3))
        den.index_add_(0, flat.reshape(-1), w.reshape(-1, 1))
        vg = num / den.clamp(min=1e-12)
        v_smooth = TRI.g2p(flat, w, vg)
        s = float((u * v).sum() / (v * v).sum().clamp(min=1e-20))
        dp = torch.zeros(M, 3, device=dev, requires_grad=True)
        opt = torch.optim.Adam([dp], lr=3e-2)
        for _ in range(300):
            opt.zero_grad()
            (((x + TRI.g2p(flat, w, dp)) - gt) ** 2).sum(-1).mean().div(EXT ** 2).backward()
            opt.step()
        acc["정지"] += e(torch.zeros_like(u)); acc["관성"] += e(v)
        acc["관성격자"] += e(v_smooth); acc["관성최적"] += e(s * v)
        acc["평행이동"] += e(u.mean(0, keepdim=True).expand_as(u))
        acc["자유"] += e(TRI.g2p(flat, w, dp.detach()))
        cnt += 1
    print(f"  {os.path.basename(f)} 완료", flush=True)

base = (acc["정지"] / cnt) ** 0.5
print(f"\n창 {cnt} 개 평균 (EXT 대비 RMSE)")
for k, s in acc.items():
    r = (s / cnt) ** 0.5
    print(f"  {k:6} {100*r:.4f}%   정지 대비 비 {r/base:.3f}")
print("BASE_OK", flush=True)
