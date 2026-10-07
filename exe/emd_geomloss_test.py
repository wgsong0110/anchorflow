"""geomloss(다중 해상도 Sinkhorn, KeOps)의 W1 근사를 정확해(헝가리안)와 비교하고 13.9 만 점 시간을 잰다."""
import time

import numpy as np
import torch
from geomloss import SamplesLoss
from scipy.optimize import linear_sum_assignment

dev = "cuda"
torch.manual_seed(0)
X = torch.as_tensor(np.load("/home/dkta/work/repflow/aux_wolf_pts.npy"), device=dev)
for blur, scal in ((1e-3, 0.9), (3e-4, 0.95)):
    L = SamplesLoss("sinkhorn", p=1, blur=blur, scaling=scal, debias=True, backend="multiscale")
    Lo = SamplesLoss("sinkhorn", p=1, blur=blur, scaling=scal, debias=True, backend="online")
    for n in (3000, 8000):
        A = X[torch.randperm(X.shape[0], device=dev)[:n]]
        for sh in (0.001, 0.005, 0.02, 0.08, 0.2):
            B = A + sh * torch.randn_like(A) + torch.tensor([sh, 0, 0], device=dev)
            C = torch.cdist(A.double(), B.double()).cpu().numpy(); r, c = linear_sum_assignment(C)
            ex = C[r, c].mean()
            v = float(Lo(A, B)); vm = float(L(A, B))
            print(f"blur {blur} sc {scal} n {n} 이동 {sh}: 정확 {ex:.6f}  online {100*(v-ex)/ex:+.3f}%  "
                  f"multiscale {100*(vm-ex)/ex:+.3f}%", flush=True)
    for sh in (0.005, 0.02, 0.08):
        B = X + sh * torch.randn_like(X) + torch.tensor([sh, 0, 0], device=dev)
        torch.cuda.synchronize(); t = time.time(); v = float(L(X, B)); torch.cuda.synchronize()
        print(f"blur {blur} 전체 {X.shape[0]} 이동 {sh}: {v:.6f}  {time.time()-t:.1f}s", flush=True)
print("GEOM_DONE")
