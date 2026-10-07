"""EMD (같은 개수 두 점 집합의 최적 일대일 매칭 평균 거리) -- GPU 경매 알고리즘 (Bertsekas, ε 스케일링).

후보를 제한하지 않고 모든 쌍 거리를 덩어리로 다시 계산한다. 마지막 ε 에서 ε-최적이라
평균 거리 오차는 ε 이하다 (eps_end 를 충분히 작게 두면 정확해와 같다).
야코비 경매: 짝 없는 사람이 모두 동시에 입찰하고, 물건마다 가장 높은 입찰을 받는다.

  python exe/emd_auction.py --test
"""
import argparse
import time

import numpy as np
import torch


def _best2(Bc, prices, Ap, ch):
    """사람 Ap 각각에 대해 가치 -|a-b| - p 의 최댓값·두 번째 값·물건 번호."""
    w1, w2, j1 = [], [], []
    for s in range(0, Ap.shape[0], ch):
        v = -torch.cdist(Ap[s:s + ch], Bc) - prices[None]
        tv, ti = v.topk(2, dim=1)
        w1.append(tv[:, 0]); w2.append(tv[:, 1]); j1.append(ti[:, 0])
    return torch.cat(w1), torch.cat(w2), torch.cat(j1)


def emd(A, B, eps_end=1e-6, factor=5.0, ch=2048, max_rounds=100000):
    """반환 (평균 거리, 라운드 수)."""
    A = A.float().contiguous(); B = B.float().contiguous()
    n = A.shape[0]
    dev = A.device
    prices = torch.zeros(n, device=dev)
    eps = float(torch.cdist(A[:256], B[:256]).max()) / 4
    rounds = 0
    while True:
        owner = torch.full((n,), -1, dtype=torch.long, device=dev)      # 물건 -> 사람
        assign = torch.full((n,), -1, dtype=torch.long, device=dev)     # 사람 -> 물건
        while True:
            free = torch.nonzero(assign < 0).squeeze(1)
            if free.numel() == 0:
                break
            w1, w2, j1 = _best2(B, prices, A[free], ch)
            bid = prices[j1] + (w1 - w2) + eps
            # 물건마다 가장 높은 입찰 (동률이면 아무나)
            best = torch.full((n,), -float("inf"), device=dev)
            best.scatter_reduce_(0, j1, bid, reduce="amax")
            win = bid >= best[j1]
            # 같은 물건에 같은 최고가가 여럿이면 하나만
            cand = torch.full((n,), n, dtype=torch.long, device=dev)
            cand.scatter_reduce_(0, j1[win], torch.arange(win.sum(), device=dev), reduce="amin")
            wi = torch.nonzero(win).squeeze(1)
            keep = cand[j1[wi]] == torch.arange(wi.numel(), device=dev)
            wi = wi[keep]
            persons, objs = free[wi], j1[wi]
            prev = owner[objs]
            assign[prev[prev >= 0]] = -1
            owner[objs] = persons
            assign[persons] = objs
            prices[objs] = bid[wi]
            rounds += 1
            if rounds > max_rounds:
                raise RuntimeError("경매가 끝나지 않는다")
        if eps <= eps_end:
            break
        eps = max(eps / factor, eps_end)
    return float((A - B[assign]).norm(dim=1).double().mean()), rounds


if __name__ == "__main__":
    from scipy.optimize import linear_sum_assignment
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
            t = time.time(); v, rd = emd(A, B)
            print(f"n {n} 이동 {sh}: 정확 {ex:.6f}  경매 {v:.6f} ({100 * (v - ex) / ex:+.4f}%)  "
                  f"라운드 {rd}  {time.time() - t:.1f}s", flush=True)
    for sh in (0.005, 0.02, 0.08, 0.2):
        B = X + sh * torch.randn_like(X) + torch.tensor([sh, 0, 0], device=dev)
        torch.cuda.synchronize(); t = time.time()
        v, rd = emd(X, B)
        print(f"전체 {X.shape[0]} 이동 {sh}: {v:.6f}  라운드 {rd}  {time.time() - t:.0f}s", flush=True)
    print("AUCTION_DONE")
