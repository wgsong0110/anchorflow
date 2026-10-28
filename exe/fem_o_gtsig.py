"""절 O 진단: 정답 lego L000 입자 F 의 특이값 분포 (프레임별 최대·99.9 %·최소, det 최소) -- 우리 lag 속도 바닥 판의 σ(F̄) 와 비교."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                         # noqa: E402
from anchorflow import deform_model as DM                                            # noqa: E402
d = torch.load("/home/dkta/work/one/L000/L000.pt", map_location="cpu", weights_only=False)
h = float(d["cfg"]["frame_dt"])
F = DM.traj_F(d, h, "cuda")
for t in (1, 5, 10, 13, 15, 17, 20, 27, 40, 60, 80):
    Ft = F[t].cuda().double().reshape(-1, 3, 3)
    s = torch.linalg.svdvals(Ft)
    print(f"정답 프레임 {t}: σ 최대 {float(s[:, 0].max()):.3f} (99.9% {float(torch.quantile(s[:, 0].float(), 0.999)):.3f}) 최소 {float(s[:, -1].min()):.3f} "
          f"(0.1% {float(torch.quantile(s[:, -1].float(), 0.001)):.3f}), det 최소 {float(torch.linalg.det(Ft).min()):.3f}", flush=True)
