"""rep_track2 의 Sinkhorn EMD 근사를 정확해(헝가리안)와 비교한다 (작은 점 집합)."""
import math
import os
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rep_track2.py")).read()
ns = {"torch": torch, "math": math}
exec(src[src.index("def emd(A, B"):src.index("RENDER = None")], ns)
emd = ns["emd"]
torch.manual_seed(0)
X = torch.as_tensor(np.load("/home/dkta/work/repflow/aux_wolf_pts.npy"), device="cuda")
for n in [3000]:
    A = X[torch.randperm(X.shape[0], device="cuda")[:n]]
    for sh in [0.001, 0.005, 0.02, 0.08]:
        B = A + sh * torch.randn_like(A) + torch.tensor([sh, 0, 0], device="cuda")
        C = torch.cdist(A.double(), B.double()).cpu().numpy(); r, c = linear_sum_assignment(C)
        print(f"n {n} 이동 {sh}: 정확 {C[r, c].mean():.6f}  Sinkhorn {emd(A, B):.6f}", flush=True)
A = X; B = X + 0.01 * torch.randn_like(X)
torch.cuda.synchronize(); t = time.time(); v = emd(A, B); torch.cuda.synchronize()
print(f"전체 {X.shape[0]} 점: {v:.6f}  {time.time() - t:.1f}s", flush=True)
