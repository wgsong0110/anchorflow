"""EMD (전체 점, 부분표본 없음): rep_track2 가 저장한 프레임별 위치(emd_y, emd_tgt)로 잰다.

정확해(헝가리안)는 13.9 만 점이면 비용 행렬만 155GB 라 풀 수 없어, 같은 개수 두 점
집합의 최적 수송을 로그 영역 Sinkhorn 으로 근사한다 (비용 = 유클리드 거리, 균등 질량).
ε 를 지름에서 eps_end 까지 줄여 가며(담금질) 반복하고, 마지막 ε 에서 행 주변분포 오차가
tol 아래로 내려갈 때까지 더 돈다. 값은 쌍대 목적 <a,f> + <b,g> (수렴하면 수송 비용).

  python exe/rep_emd.py --res repflow/r5/wolf_ours.npz           # 결과에 EMD 를 적는다
  python exe/rep_emd.py --test                                    # 작은 집합에서 정확해와 비교
"""
import argparse
import math
import time

import numpy as np
import torch


def _softmin(P, Q, h, lw, e, ch):
    """-e·logsumexp_j(lw + (h_j - |p_i - q_j|)/e)."""
    out = torch.empty(P.shape[0], device=P.device, dtype=P.dtype)
    for i in range(0, P.shape[0], ch):
        out[i:i + ch] = -e * torch.logsumexp(lw + (h[None] - torch.cdist(P[i:i + ch], Q)) / e, 1)
    return out


def emd(A, B, eps_end=1e-4, shrink=0.7, tol=1e-3, max_extra=300, ch=2048, log=False):
    A, B = A.float().contiguous(), B.float().contiguous()
    n, m = A.shape[0], B.shape[0]
    la, lb = -math.log(n), -math.log(m)
    f = torch.zeros(n, device=A.device); g = torch.zeros(m, device=A.device)
    e = 2.0 * float(torch.cdist(A[:1], B).max())
    while True:                                       # 담금질: ε 마다 대칭 갱신 한 번
        e = max(e, eps_end)
        f, g = 0.5 * (f + _softmin(A, B, g, lb, e, ch)), 0.5 * (g + _softmin(B, A, f, la, e, ch))
        if e <= eps_end:
            break
        e *= shrink
    err = float("nan")
    for k in range(max_extra):                        # 마지막 ε 에서 수렴까지
        fn = _softmin(A, B, g, lb, e, ch)
        err = float((torch.exp((f - fn) / e) - 1).abs().mean())   # 행 주변분포 상대 오차
        f = fn
        g = _softmin(B, A, f, la, e, ch)
        if err < tol:
            break
    if log:
        print(f"    (Sinkhorn 추가 반복 {k + 1}, 주변분포 오차 {err:.1e})", flush=True)
    return float(f.double().mean() + g.double().mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", nargs="*", default=[])
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--pts", default="/home/dkta/work/repflow/aux_wolf_pts.npy")
    a = ap.parse_args()
    dev = "cuda"
    if a.test:
        from scipy.optimize import linear_sum_assignment
        torch.manual_seed(0)
        X = torch.as_tensor(np.load(a.pts), device=dev)
        for n in (3000, 8000):
            A = X[torch.randperm(X.shape[0], device=dev)[:n]]
            for sh in (0.001, 0.005, 0.02, 0.08, 0.2):
                B = A + sh * torch.randn_like(A) + torch.tensor([sh, 0, 0], device=dev)
                B = B[torch.randperm(n, device=dev)]
                C = torch.cdist(A.double(), B.double()).cpu().numpy()
                r, c = linear_sum_assignment(C)
                ex = C[r, c].mean()
                ap_ = emd(A, B, log=True)
                print(f"n {n} 이동 {sh}: 정확 {ex:.6f}  Sinkhorn {ap_:.6f}  "
                      f"상대 {100 * (ap_ - ex) / ex:+.2f}%", flush=True)
        B = X + 0.02 * torch.randn_like(X)
        torch.cuda.synchronize(); t = time.time(); v = emd(X, B, log=True); torch.cuda.synchronize()
        print(f"전체 {X.shape[0]} 점: {v:.6f}  {time.time() - t:.1f}s", flush=True)
    for p in a.res:
        Z = dict(np.load(p, allow_pickle=True))
        L = float(Z["L"])
        ev = []
        for t, y, g in zip(Z["emd_t"], Z["emd_y"], Z["emd_tgt"]):
            v = emd(torch.as_tensor(y, device=dev), torch.as_tensor(g, device=dev)) / L
            ev.append(v)
            Z["metrics"][int(t) - 1, 3] = v
            print(f"  [{p}] t={int(t):3d}  EMD {100 * v:.3f}%", flush=True)
        Z["emd"] = np.array(ev)
        np.savez_compressed(p, **Z)
        print(f"[EMD] {p}  평균 {100 * np.mean(ev):.3f}%", flush=True)
