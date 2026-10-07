"""EMD (같은 개수 두 점 집합의 최적 일대일 매칭 평균 거리) -- 전체 그래프에서 ε-최적 = 사실상 정확해.

1) 후보 = 각 점의 k 최근접 + 같은 번호 짝. 후보 위에서 희소 야코비 경매(ε 스케일링).
2) 마지막 ε 에서 **모든 쌍**을 덩어리로 훑어 ε-상보성(가치 −c_ij − p_j 가 배정 가치 + ε 보다
   큰 j 가 있는가)을 검사한다. 어기는 쌍을 후보에 넣고 같은 ε 에서 이어 경매한다.
3) 어기는 쌍이 없으면 끝 -- 그 배정은 전체 그래프에서 ε-최적이라 평균 거리 오차가 ε 이하다.

  python exe/emd_exact.py --test
"""
import argparse
import time

import numpy as np
import torch


def _cands(A, B, k, ch=4096):
    out = []
    for s in range(0, A.shape[0], ch):
        out.append(torch.cdist(A[s:s + ch], B).topk(k, largest=False).indices)
    I = torch.cat(out)
    return torch.cat([I, torch.arange(A.shape[0], device=A.device)[:, None]], 1)


def _auction(A, B, cand, prices, assign, owner, eps, max_rounds):
    """후보 위 희소 야코비 경매. assign/owner/prices 를 제자리 갱신. 반환 라운드 수."""
    n, k = cand.shape
    C = (A[:, None] - B[cand]).norm(dim=-1)                         # [n,k]
    rounds = 0
    while True:
        free = torch.nonzero(assign < 0).squeeze(1)
        if free.numel() == 0:
            return rounds
        cf = cand[free]
        val = -C[free] - prices[cf]
        tv, ti = val.topk(2, dim=1)
        j1 = cf.gather(1, ti[:, :1]).squeeze(1)
        bid = prices[j1] + (tv[:, 0] - tv[:, 1]) + eps
        best = torch.full((n,), -float("inf"), device=A.device, dtype=bid.dtype)
        best.scatter_reduce_(0, j1, bid, reduce="amax")
        win = torch.nonzero(bid >= best[j1]).squeeze(1)
        first = torch.full((n,), win.numel(), dtype=torch.long, device=A.device)
        first.scatter_reduce_(0, j1[win], torch.arange(win.numel(), device=A.device), reduce="amin")
        win = win[first[j1[win]] == torch.arange(win.numel(), device=A.device)]
        persons, objs = free[win], j1[win]
        prev = owner[objs]
        assign[prev[prev >= 0]] = -1
        owner[objs] = persons
        assign[persons] = objs
        prices[objs] = bid[win]
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError("경매가 끝나지 않는다")


def emd(A, B, k=16, eps_end=1e-7, factor=4.0, ch=2048, max_rounds=2_000_000, log=False):
    """반환 (평균 거리, 정보 dict)."""
    A = A.double().contiguous(); B = B.double().contiguous()
    n = A.shape[0]; dev = A.device
    cand = _cands(A.float(), B.float(), k)
    prices = torch.zeros(n, device=dev, dtype=torch.float64)
    eps = float((A - B).norm(dim=1).max()) / 2 + 1e-9
    rounds = 0; added = 0; checks = 0
    while True:                                                        # ε 스케일링
        assign = torch.full((n,), -1, dtype=torch.long, device=dev)
        owner = torch.full((n,), -1, dtype=torch.long, device=dev)
        rounds += _auction(A, B, cand, prices, assign, owner, eps, max_rounds)
        if eps <= eps_end:
            break
        eps = max(eps / factor, eps_end)
    while True:                                                        # 전체 쌍 ε-상보성 검사
        checks += 1
        av = -(A - B[assign]).norm(dim=1) - prices[assign]
        viol_i, viol_j = [], []
        for s in range(0, n, ch):
            v = -torch.cdist(A[s:s + ch], B) - prices[None]
            bv, bj = v.max(1)
            bad = bv > av[s:s + ch] + eps
            if bad.any():
                ii = torch.nonzero(bad).squeeze(1)
                viol_i.append(ii + s); viol_j.append(bj[ii])
        if not viol_i:
            break
        vi, vj = torch.cat(viol_i), torch.cat(viol_j)
        added += vi.numel()
        cand = torch.cat([cand, torch.full((n, 1), -1, dtype=torch.long, device=dev)], 1)
        cand[:, -1] = cand[:, 0]                                       # 빈 칸은 첫 후보로 채운다
        cand[vi, -1] = vj
        # 어긴 사람은 배정을 풀고 같은 ε 에서 이어 경매
        owner[assign[vi]] = -1
        assign[vi] = -1
        rounds += _auction(A, B, cand, prices, assign, owner, eps, max_rounds)
        if log:
            print(f"    검사 {checks}: 위반 {vi.numel()} 쌍 추가", flush=True)
    return float((A - B[assign]).norm(dim=1).mean()), dict(rounds=rounds, added=added, checks=checks,
                                                           k=int(cand.shape[1]))


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
            C = torch.cdist(A.double(), B.double()).cpu().numpy(); r, c = linear_sum_assignment(C)
            ex = C[r, c].mean()
            t = time.time(); v, info = emd(A, B)
            print(f"n {n} 이동 {sh}: 정확 {ex:.7f}  ε최적 {v:.7f} ({100 * (v - ex) / ex:+.5f}%)  {info}  "
                  f"{time.time() - t:.1f}s", flush=True)
    for sh in (0.005, 0.02, 0.08):
        B = X + sh * torch.randn_like(X) + torch.tensor([sh, 0, 0], device=dev)
        torch.cuda.synchronize(); t = time.time(); v, info = emd(X, B, log=True)
        print(f"전체 {X.shape[0]} 이동 {sh}: {v:.7f}  {info}  {time.time() - t:.0f}s", flush=True)
    print("EXACT_DONE")
