"""EMD (같은 개수 두 점 집합의 최적 일대일 매칭 평균 거리) 를 전체 점에서 정확하게 잰다.

후보 짝을 서로의 k 최근접으로 제한한 희소 이분 그래프에서 최소 가중 완전 매칭을 푼다
(scipy LAPJVsp). 최적 짝이 후보 안에 있으면 정확해와 같고, 아니면 위쪽 한계다. 완전 매칭이
없거나 정확해와 다르면 k 를 늘린다. --test 는 작은 집합에서 헝가리안과 비교한다.
"""
import argparse
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def knn(A, B, k, ch=4096):
    d, i = [], []
    for s in range(0, A.shape[0], ch):
        dd, ii = torch.cdist(A[s:s + ch], B).topk(k, largest=False)
        d.append(dd.double().cpu()); i.append(ii.cpu())
    return torch.cat(d).numpy(), torch.cat(i).numpy()


def _match(A, B, k):
    n = A.shape[0]
    _, iab = knn(A, B, k)
    _, iba = knn(B, A, k)
    # 후보: 서로의 k 최근접 + 같은 번호 짝(i, i) -- 같은 입자끼리의 짝이 늘 있어 완전 매칭이 존재한다
    rows = np.concatenate([np.repeat(np.arange(n), k), iba.reshape(-1), np.arange(n)])
    cols = np.concatenate([iab.reshape(-1), np.repeat(np.arange(n), k), np.arange(n)])
    G = csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n, n))
    r, c = G.nonzero()
    w = (A[torch.as_tensor(r, device=A.device)] - B[torch.as_tensor(c, device=A.device)]) \
        .norm(dim=1).double().cpu().numpy() + 1e-12
    G = csr_matrix((w, (r, c)), shape=(n, n))
    ri, ci = min_weight_full_bipartite_matching(G)
    return float((A[torch.as_tensor(ri, device=A.device)] - B[torch.as_tensor(ci, device=A.device)])
                 .norm(dim=1).double().mean())


def emd(A, B, k=32, kmax=256, tol=1e-4):
    """A, B: [N,3] (같은 입자 순서). k 를 두 배씩 늘려 값의 상대 변화가 tol 아래면 멈춘다.
    후보 제한 매칭은 정확해의 위쪽 한계이고 k 가 커지면 정확해로 내려간다. 반환 (평균 거리, 쓴 k)."""
    A, B = A.float().contiguous(), B.float().contiguous()
    prev = _match(A, B, k)
    while k < kmax:
        k *= 2
        cur = _match(A, B, k)
        if abs(prev - cur) <= tol * cur:
            return cur, k
        prev = cur
    return prev, k


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pts", default="/home/dkta/work/repflow/aux_wolf_pts.npy")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    X = torch.as_tensor(np.load(a.pts), device=dev)
    for n in (3000, 8000):
        A = X[torch.randperm(X.shape[0], device=dev)[:n]]
        for sh in (0.001, 0.005, 0.02, 0.08, 0.2):
            B = A + sh * torch.randn_like(A) + torch.tensor([sh, 0, 0], device=dev)
            B = B[torch.randperm(n, device=dev)]
            C = torch.cdist(A.double(), B.double()).cpu().numpy(); r, c = linear_sum_assignment(C)
            ex = C[r, c].mean()
            v, ku = emd(A, B)
            print(f"n {n} 이동 {sh}: 정확 {ex:.6f}  희소 {v:.6f} ({100 * (v - ex) / ex:+.4f}%, k {ku})",
                  flush=True)
    for sh in (0.005, 0.02, 0.08):
        B = X + sh * torch.randn_like(X) + torch.tensor([sh, 0, 0], device=dev)
        t = time.time()
        v, ku = emd(X, B)
        print(f"전체 {X.shape[0]} 이동 {sh}: {v:.6f} (k {ku})  {time.time() - t:.0f}s", flush=True)
    print("EMDSP_DONE")
