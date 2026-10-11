"""EvalModel 한 스텝의 커널 단위 프로파일 (torch.profiler) -- 롤아웃 최적화용.

  python exe/prof_step.py --ckpt MAIN_best.pt --tag 0000 --steps 5
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                     # noqa: E402
from anchorflow import deform_model as DM                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--traj_dir", default="/home/dkta/work/lbpt/test")
ap.add_argument("--tag", default="0000")
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--tf32", action="store_true")
ap.add_argument("--fast", action="store_true", help="빠른 경로 (fast_roll)")
a = ap.parse_args()
if a.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
M = DM.EvalModel(a.ckpt, fast=a.fast)
d = torch.load(f"{a.traj_dir}/{a.tag}.pt", map_location="cuda", weights_only=False)
x, v = M.init(d, 0)
for _ in range(3):                                   # 예열
    x, v, _ = M.frame(d, x, v)
torch.cuda.synchronize(); t = time.time()
for _ in range(10):
    x, v, _ = M.frame(d, x, v)
torch.cuda.synchronize(); print(f"[시간] 프레임당 {(time.time()-t)/10*1e3:.1f} ms (동기화 없이)", flush=True)
from torch.profiler import profile, ProfilerActivity                # noqa: E402
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(a.steps):
        x, v, _ = M.frame(d, x, v)
    torch.cuda.synchronize()
ka = prof.key_averages()
print(ka.table(sort_by="cuda_time_total", row_limit=35, max_name_column_width=70))
n_k = sum(e.count for e in ka if e.device_type == torch.autograd.DeviceType.CUDA)
print(f"[커널] 스텝당 CUDA 커널 호출 {n_k / a.steps:.0f} 회", flush=True)
print(ka.table(sort_by="cpu_time_total", row_limit=25, max_name_column_width=70))
