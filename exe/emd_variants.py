"""EMD 근사 후보들을 정확해(헝가리안)와 비교한다 (rep_emd.py 의 방식 고르기용)."""
import math
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

dev = "cuda"


def softmin(P, Q, h, lw, e, ch=2048):
    out = torch.empty(P.shape[0], device=P.device, dtype=P.dtype)
    for i in range(0, P.shape[0], ch):
        out[i:i + ch] = -e * torch.logsumexp(lw + (h[None] - torch.cdist(P[i:i + ch], Q)) / e, 1)
    return out


def solve(A, B, eps_end, shrink, tol=1e-3, max_extra=500):
    n, m = A.shape[0], B.shape[0]
    la, lb = -math.log(n), -math.log(m)
    f = torch.zeros(n, device=dev, dtype=A.dtype); g = torch.zeros(m, device=dev, dtype=A.dtype)
    e = 2.0 * float(torch.cdist(A[:1], B).max()); it = 0
    while True:
        e = max(e, eps_end)
        f = softmin(A, B, g, lb, e); g = softmin(B, A, f, la, e); it += 1
        if e <= eps_end:
            break
        e *= shrink
    for k in range(max_extra):
        fn = softmin(A, B, g, lb, e)
        err = float((torch.exp((f - fn) / e) - 1).abs().mean())
        f = fn; g = softmin(B, A, f, la, e); it += 1
        if err < tol:
            break
    return f, g, e, it, err


def primal(A, B, f, g, e, ch=2048):
    la, lb = -math.log(A.shape[0]), -math.log(B.shape[0])
    c = 0.0
    for i in range(0, A.shape[0], ch):
        C = torch.cdist(A[i:i + ch], B)
        c += float((torch.exp(la + lb + (f[i:i + ch, None] + g[None] - C) / e) * C).double().sum())
    return c


def variant(A, B, eps_end, shrink, debias):
    f, g, e, it, err = solve(A, B, eps_end, shrink)
    p = primal(A, B, f, g, e)
    d = float(f.double().mean() + g.double().mean())
    if debias:
        fa, ga, *_ = solve(A, A, eps_end, shrink); fb, gb, *_ = solve(B, B, eps_end, shrink)
        d = d - 0.5 * float(fa.double().mean() + ga.double().mean()) \
            - 0.5 * float(fb.double().mean() + gb.double().mean())
    return p, d, it, err


torch.manual_seed(0)
X = torch.as_tensor(np.load("/home/dkta/work/repflow/aux_wolf_pts.npy"), device=dev)
VARS = [(1e-3, 0.9), (3e-4, 0.9), (1e-3, 0.7), (3e-4, 0.95)]
for n in (3000,):
    A = X[torch.randperm(X.shape[0], device=dev)[:n]]
    for sh in (0.001, 0.005, 0.02, 0.08, 0.2):
        B = A + sh * torch.randn_like(A) + torch.tensor([sh, 0, 0], device=dev)
        C = torch.cdist(A.double(), B.double()).cpu().numpy(); r, c = linear_sum_assignment(C)
        ex = C[r, c].mean()
        out = [f"이동 {sh}: 정확 {ex:.5f}"]
        for (ee, s) in VARS:
            for dt in (torch.float32, torch.float64):
                p, d, it, err = variant(A.to(dt), B.to(dt), ee, s, debias=True)
                out.append(f"  ε{ee:g}/{s} {str(dt)[-7:]}: 원시 {100*(p-ex)/ex:+.1f}% "
                           f"편향제거 {100*(d-ex)/ex:+.1f}% (반복 {it}, 오차 {err:.0e})")
        print("\n".join(out), flush=True)
B = X + 0.02 * torch.randn_like(X)
for (ee, s) in VARS[:2]:
    torch.cuda.synchronize(); t = time.time()
    f, g, e, it, err = solve(X, B, ee, s); torch.cuda.synchronize()
    print(f"전체 {X.shape[0]}: ε{ee:g}/{s} 반복 {it} 오차 {err:.0e}  {time.time()-t:.0f}s", flush=True)
print("VAR_DONE")
