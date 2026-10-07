"""표현력 비교 결과 하나에 대한 추가 측정: EMD(전체 점, 정확) · 물리 지표 · 시각 품질.

  python exe/rep_metrics.py --res repflow/r9/wolf_ours.npz --flow repflow/gflow_wolf.npz \
      --video repflow/r9/wolf_ours.mp4 --target_video repflow/r9/wolf_target.mp4 --out ....json

- EMD: rep_track2 가 10 프레임마다 저장한 위치로, kNN 후보 희소 최소 가중 완전 매칭 (emd_sparse).
- 물리 지표: 같은 질량 1/N 입자의 운동에너지 KE, 선운동량 P, 각운동량 Lang (프레임 차분 속도),
  부피비 mean det F. 목표 흐름에서 같은 식으로 잰 값과의 차이를 고정 상수로 나눈다
  (목표 값의 프레임 평균 크기 -- 자기 자신으로 나누지 않는다).
- 시각 품질: 같은 카메라로 그린 목표 영상 대비 프레임별 PSNR · SSIM · LPIPS(alex), 프레임 집합 FID.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emd_sparse                                                  # noqa: E402
import gauss_flow as gf                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True)
ap.add_argument("--flow", required=True)
ap.add_argument("--video", default="")
ap.add_argument("--target_video", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=30.0)
a = ap.parse_args()
dev = "cuda"
Z = np.load(a.res, allow_pickle=True)
L = float(Z["L"])
out = {"res": a.res}

# ---------------- EMD (정확)
ev = []
for t, y, g in zip(Z["emd_t"], Z["emd_y"], Z["emd_tgt"]):
    v, k = emd_sparse.emd(torch.as_tensor(y, device=dev), torch.as_tensor(g, device=dev))
    ev.append((int(t), v / L, k))
    print(f"  EMD t={int(t):3d} {100 * v / L:.3f}%  (k {k})", flush=True)
out["emd_frames"] = ev
out["EMD_pct"] = 100 * float(np.mean([e[1] for e in ev]))

# ---------------- 물리 지표: 목표 흐름에서 같은 식
D = np.load(a.flow)
field = D["field"]
T = int(Z["metrics"].shape[0])
period = 10
x = torch.as_tensor(D["traj"][0], dtype=torch.float64, device=dev)
F = torch.eye(3, dtype=torch.float64, device=dev).expand(x.shape[0], 3, 3).clone()
tp = []
xp = x.clone()
for t in range(T):
    f = torch.as_tensor(field[t // period], dtype=torch.float64, device=dev)
    x, F, _, _ = gf.advance(x, F, f, 1.0 / a.fps, 0.5)
    vel = (x - xp) * a.fps
    com = x.mean(0)
    tp.append((t + 1, float(0.5 * (vel * vel).sum(1).mean()), *vel.mean(0).tolist(),
               *torch.cross(x - com, vel, dim=-1).mean(0).tolist(), float(torch.linalg.det(F).mean())))
    xp = x.clone()
tp = np.array(tp)
mp = Z["phys"]
n = min(len(tp), len(mp))
tp, mp = tp[:n], mp[:n]
ke_s = np.abs(tp[:, 1]).mean()
p_s = np.linalg.norm(tp[:, 2:5], axis=1).mean() + 1e-12
l_s = np.linalg.norm(tp[:, 5:8], axis=1).mean() + 1e-12
out["phys"] = {
    "KE_err_pct": 100 * float(np.abs(mp[:, 1] - tp[:, 1]).mean() / ke_s),
    "P_err_pct": 100 * float(np.linalg.norm(mp[:, 2:5] - tp[:, 2:5], axis=1).mean() / p_s),
    "Lang_err_pct": 100 * float(np.linalg.norm(mp[:, 5:8] - tp[:, 5:8], axis=1).mean() / l_s),
    "vol_err_pct": 100 * float(np.abs(mp[:, 8] - tp[:, 8]).mean()),          # 부피비(질량 밀도 역수) 차
    "KE_drift_pct": 100 * float(np.abs(np.diff(mp[:, 1])).mean() / ke_s),    # 프레임간 에너지 요동
}
print("  물리", out["phys"], flush=True)

# ---------------- 시각 품질
if a.video and a.target_video and os.path.exists(a.video) and os.path.exists(a.target_video):
    import imageio.v2 as imageio
    from torchmetrics.image import (PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image.fid import FrechetInceptionDistance
    A = np.stack(list(imageio.get_reader(a.video)))
    B = np.stack(list(imageio.get_reader(a.target_video)))
    m = min(len(A), len(B))
    A = torch.as_tensor(A[:m]).permute(0, 3, 1, 2).float().div(255).to(dev)
    B = torch.as_tensor(B[:m]).permute(0, 3, 1, 2).float().div(255).to(dev)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    vis = {"PSNR": [], "SSIM": [], "LPIPS": []}
    try:
        lp = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)
    except Exception as ex:                                             # 가중치 내려받기 실패 등
        lp = None
        print("  LPIPS 불가:", ex, flush=True)
    for i in range(m):
        vis["PSNR"].append(float(psnr(A[i:i + 1], B[i:i + 1])))
        vis["SSIM"].append(float(ssim(A[i:i + 1], B[i:i + 1])))
        if lp is not None:
            vis["LPIPS"].append(float(lp(A[i:i + 1].clamp(0, 1), B[i:i + 1].clamp(0, 1))))
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    fid.update(B, real=True)
    fid.update(A, real=False)
    out["visual"] = {k: float(np.mean(v)) for k, v in vis.items() if v}
    out["visual"]["FID"] = float(fid.compute())
    print("  시각", out["visual"], flush=True)

json.dump(out, open(a.out, "w"), indent=1)
print(f"[측정] {a.out}", flush=True)
