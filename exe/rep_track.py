"""표현력 비교: 목표 궤적(exe/gauss_flow.py)을 각 표현이 자기 자유도로 따라간다.

프레임마다 이전 해에서 출발해 같은 반복 수의 Adam 으로

    loss = mean_p |y_p(θ) - y*_p|²  +  λ_inv · mean_p relu(τ - det F_p)²

를 줄인다. det F 는 **모든 방법에 같은 방식**으로 잰다 -- 정지 상태 이웃 k 개로
최소제곱 F_p = (Σ dy dxᵀ)(Σ dx dxᵀ)⁻¹ (방법마다 사상의 미분 형태가 달라 이렇게
통일한다). 자유도 예산은 --dof 로 맞춘다 (실제 개수를 결과에 남긴다).

표현 (모두 정지 위치 X 의 함수 y = f_θ(X)):
  ours       사면체 격자 꼭짓점 변위 u_i + 꼭짓점별 꺾임 반경 ρ_i (4/꼭짓점)
             y = X + Σ_i w_i ψ_i(r_i) u_i / Σ_j w_j ψ_j(r_j)
             w_i = λ_i² / Σ λ_j²  (Gregory 볼록 결합: 맞은편 면에서 값·기울기 0)
             ψ_i(r) = 1 - a (r/ρ_i)²            (r < ρ_i, 중심 기울기 0)
                    = (1-a) / (1 + (r-ρ_i)/ℓ)    (r ≥ ρ_i, 기울기가 꺾인다)
  vrgs       GS-Verse(VR-GS): 표면 삼각형 메시 꼭짓점 (3/꼭짓점)
             y = v0 + a1 e1 + a2 e2 + b n (lib/anchorflow/gsverse.py 결합 그대로)
  phystwin   PhysTwin: 제어점 변위 (3/제어점). 제어점마다 이웃 16 개로 회전을
             맞추고(Procrustes), 입자는 가까운 16 개 제어점 변환을 역거리 가중으로
             섞는다 (PhysTwin gaussian_splatting/dynamic_utils.interpolate_motions)
  gaussim    GausSim: k-means 군집(CMS)마다 위치+변형기울기 (12/군집)
             y = c_k + t_k + M_k (X - c_k)   (입자는 자기 군집 하나를 따른다)
  simplicits Simplicits: 신경 스키닝 가중치 W(X) (정지 형상에서 학습) +
             핸들별 3x4 변환 (12/핸들). y = X + Σ_k W_k(X) T_k [X;1]
             (가중치 학습은 exe/simplicits_weights.py, kaolin 환경)

  python exe/rep_track.py --flow repflow/flow_wolf.npz --method ours \
      --out repflow/res_wolf_ours.npz
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--flow", required=True)
ap.add_argument("--method", required=True,
                choices=["ours", "vrgs", "phystwin", "gaussim", "simplicits"])
ap.add_argument("--out", required=True)
ap.add_argument("--dof", type=int, default=3000)
ap.add_argument("--iters", type=int, default=200, help="프레임당 반복")
ap.add_argument("--iters0", type=int, default=400, help="첫 프레임 반복")
ap.add_argument("--lr", type=float, default=1e-3, help="학습률 탐색에서 네 방법 모두 1e-3 이 최선")
ap.add_argument("--lam_inv", type=float, default=100.0)
ap.add_argument("--tau", type=float, default=0.1, help="det F 장벽 문턱")
ap.add_argument("--knn_F", type=int, default=8)
ap.add_argument("--emd_n", type=int, default=4096)
ap.add_argument("--emd_every", type=int, default=10)
ap.add_argument("--aux", default="", help="함께 옮길 정지 점 (영상용 가우시안 중심, npy)")
ap.add_argument("--simp_w", default="", help="simplicits: 학습된 가중치 npz")
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)

D = np.load(a.flow)
X0 = torch.as_tensor(D["X0"], device=dev)                  # [N,3]
TRAJ = torch.as_tensor(D["traj"], device=dev)              # [T+1,N,3]
T = TRAJ.shape[0] - 1
N = X0.shape[0]
L = float((X0.max(0).values - X0.min(0).values).norm())    # 고정 정규화 상수
AUX = (torch.as_tensor(np.load(a.aux), dtype=torch.float32, device=dev)
       if a.aux else torch.zeros(0, 3, device=dev))
XALL = torch.cat([X0, AUX], 0)                             # 결합은 둘 다에 한다


# ============================================================== 표현들
class Ours(torch.nn.Module):
    def __init__(self, X, dof):
        super().__init__()
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
        from anchorflow import simplex as sx
        n_nodes = 6.0
        for _ in range(40):                                 # 꼭짓점 수를 dof/4 에 맞춘다
            lo, lat, nn = sx.grid_for_nodes(X0, n_nodes)
            idx, lam, _ = sx.locate(X, lo, lat, nn)
            rows, uniq = sx.active_nodes(idx)
            if 4 * uniq.numel() >= dof:
                break
            n_nodes *= 1.06
        self.rows, self.lam = rows, lam
        self.Xn = sx.node_pos(lo, lat, nn, uniq)            # [M,3]
        M = uniq.numel()
        self.h = lat.s
        self.u = torch.nn.Parameter(torch.zeros(M, 3, device=X.device))
        self.rho_raw = torch.nn.Parameter(torch.zeros(M, device=X.device))
        self.a, self.ell = 0.5, 0.5 * self.h
        self.dof = 4 * M
        print(f"[ours] 꼭짓점 {M}  간격 h {self.h:.4f}  자유도 {self.dof}", flush=True)

    def forward(self, X):
        rows, lam = self.rows, self.lam                     # [N,4]
        Xv = self.Xn[rows]                                  # [N,4,3]
        r = (X[:, None] - Xv).norm(dim=-1)                  # [N,4]
        rho = self.h * (0.05 + 0.95 * torch.sigmoid(self.rho_raw)[rows])  # 꺾임 위치
        # where 의 안 쓰는 갈래도 유한해야 한다 (무한대면 기울기가 NaN 이 된다)
        inner = 1.0 - self.a * (torch.minimum(r, rho) / rho) ** 2
        outer = (1.0 - self.a) / (1.0 + (r - rho).clamp_min(0.0) / self.ell)
        psi = torch.where(r < rho, inner, outer).clamp_min(1e-6)
        w = lam * lam
        w = w / w.sum(1, keepdim=True).clamp_min(1e-12)     # Gregory 볼록 결합
        g = w * psi
        W = g / g.sum(1, keepdim=True).clamp_min(1e-12)
        return X + (W[..., None] * self.u[rows]).sum(1)


class VRGS(torch.nn.Module):
    def __init__(self, X, dof):
        super().__init__()
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
        from anchorflow import gsverse as gv
        self.gv = gv
        best = None
        for ng in range(8, 80, 2):                          # 꼭짓점 수를 dof/3 에 맞춘다
            v, f = gv.mesh_from_points(X0, n_grid=ng, grid_lim=1.0)
            if best is None or abs(3 * v.shape[0] - dof) < abs(3 * best[0].shape[0] - dof):
                best = (v, f, ng)
            if 3 * v.shape[0] > dof:
                break
        v, f, ng = best
        # 쓰이는 꼭짓점만 남긴다
        used = torch.unique(f)
        remap = torch.full((v.shape[0],), -1, dtype=torch.long, device=v.device)
        remap[used] = torch.arange(used.numel(), device=v.device)
        self.vr, self.f = v[used], remap[f]
        self.ti, self.a1, self.a2, self.b = gv.bind_points(X, self.vr, self.f)
        self.v = torch.nn.Parameter(self.vr.clone())
        self.dof = 3 * self.vr.shape[0]
        print(f"[vrgs] 꼭짓점 {self.vr.shape[0]}  삼각형 {self.f.shape[0]}  격자 {ng}  "
              f"자유도 {self.dof}", flush=True)

    def forward(self, X):
        v0, e1, e2, n = self.gv._tri_frame(self.v, self.f)
        t = self.ti
        return (v0[t] + self.a1[:, None] * e1[t] + self.a2[:, None] * e2[t]
                + self.b[:, None] * n[t])


def fps(X, k):
    """가장 먼 점 샘플링 (제어점 뽑기)."""
    i = [int(torch.randint(X.shape[0], (1,)))]
    d = (X - X[i[0]]).norm(dim=1)
    for _ in range(k - 1):
        j = int(d.argmax())
        i.append(j)
        d = torch.minimum(d, (X - X[j]).norm(dim=1))
    return torch.tensor(i, device=X.device)


class PhysTwin(torch.nn.Module):
    def __init__(self, X, dof, K=16):
        super().__init__()
        nb = dof // 3
        self.B = X0[fps(X0, nb)]                            # 제어점 정지 위치
        d = torch.cdist(self.B, self.B)
        self.rel = d.topk(K + 1, largest=False).indices[:, 1:]   # 이웃 16 (자기 제외)
        dd, ii = torch.cdist(X, self.B).topk(K, largest=False)
        w = 1.0 / (dd + 1e-6)
        self.wi, self.ww = ii, w / w.sum(1, keepdim=True)   # knn_weights 그대로
        self.m = torch.nn.Parameter(torch.zeros(nb, 3, device=X.device))
        self.dof = 3 * nb
        print(f"[phystwin] 제어점 {nb}  이웃 {K}  자유도 {self.dof}", flush=True)

    def forward(self, X):
        B, m, rel = self.B, self.m, self.rel
        A0 = B[rel] - B[:, None]                            # [nb,K,3]
        A1 = (B[rel] + m[rel]) - (B[:, None] + m[:, None])
        Fm = A1.transpose(1, 2) @ A0                        # interpolate_motions 의 F
        U, S, Vh = torch.linalg.svd(Fm)
        dfix = torch.ones_like(S)
        dfix[:, -1] = torch.sign(torch.linalg.det(U @ Vh))
        R = U @ torch.diag_embed(dfix) @ Vh                 # 회전 (det -1 은 뒤집어 맞춤)
        k = self.wi                                         # [N,K]
        loc = X[:, None] - B[k]                             # [N,K,3]
        moved = (R[k] @ loc[..., None]).squeeze(-1) + B[k] + m[k]
        return (self.ww[..., None] * moved).sum(1)


class GausSim(torch.nn.Module):
    def __init__(self, X, dof):
        super().__init__()
        K = dof // 12
        C = X0[fps(X0, K)]
        for _ in range(20):                                 # k-means (군집 = CMS)
            lab = torch.cdist(X0, C).argmin(1)
            C = torch.stack([X0[lab == k].mean(0) if (lab == k).any() else C[k]
                             for k in range(K)])
        self.C = C
        self.lab = torch.cdist(X, C).argmin(1)
        self.t = torch.nn.Parameter(torch.zeros(K, 3, device=X.device))
        self.Mx = torch.nn.Parameter(torch.zeros(K, 3, 3, device=X.device))  # M = I + Mx
        self.dof = 12 * K
        print(f"[gaussim] 군집 {K}  자유도 {self.dof}", flush=True)

    def forward(self, X):
        k = self.lab
        M = torch.eye(3, device=X.device) + self.Mx[k]
        return self.C[k] + self.t[k] + (M @ (X - self.C[k])[..., None]).squeeze(-1)


class Simplicits(torch.nn.Module):
    def __init__(self, X, dof, wpath):
        super().__init__()
        Wd = np.load(wpath)
        Wfull = torch.as_tensor(Wd["W"], dtype=torch.float32, device=X.device)  # [N+aux,K]
        assert Wfull.shape[0] == X.shape[0], "가중치는 X0+aux 에 대해 학습해야 한다"
        self.W = Wfull
        K = Wfull.shape[1]
        self.Tm = torch.nn.Parameter(torch.zeros(K, 3, 4, device=X.device))
        self.dof = 12 * K
        print(f"[simplicits] 핸들 {K}  자유도 {self.dof}", flush=True)

    def forward(self, X):
        Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)   # [N,4]
        d = torch.einsum("kij,nj->nki", self.Tm, Xh)        # [N,K,3]
        return X + (self.W[..., None] * d).sum(1)


# ============================================================== 공통
if a.method == "ours":
    rep = Ours(XALL, a.dof)
elif a.method == "vrgs":
    rep = VRGS(XALL, a.dof)
elif a.method == "phystwin":
    rep = PhysTwin(XALL, a.dof)
elif a.method == "gaussim":
    rep = GausSim(XALL, a.dof)
else:
    rep = Simplicits(XALL, a.dof, a.simp_w)
rep = rep.to(dev)

# det F: 정지 이웃 k 개 최소제곱 (모든 방법 공통)
nbr = torch.cdist(X0, X0).topk(a.knn_F + 1, largest=False).indices[:, 1:] \
    if N <= 40000 else None
dX = X0[nbr] - X0[:, None]                                  # [N,k,3]
Binv = torch.linalg.inv(dX.transpose(1, 2) @ dX
                        + 1e-9 * torch.eye(3, device=dev))


def detF(Y):
    dY = Y[nbr] - Y[:, None]
    F = (dY.transpose(1, 2) @ dX) @ Binv
    return torch.linalg.det(F)


def chamfer(A, B, ch=4096):
    def one(P, Q):
        s = 0.0
        for i in range(0, P.shape[0], ch):
            s += torch.cdist(P[i:i + ch], Q).min(1).values.sum()
        return s / P.shape[0]
    return float(0.5 * (one(A, B) + one(B, A)))


EI = torch.as_tensor(np.random.default_rng(0).choice(N, min(a.emd_n, N),
                                                     replace=False), device=dev)


def emd(A, B):
    """같은 인덱스 부분표본끼리의 최적 일대일 매칭 평균 거리."""
    from scipy.optimize import linear_sum_assignment
    P, Q = A[EI].double(), B[EI].double()
    C = torch.cdist(P, Q).cpu().numpy()
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean())


with torch.no_grad():
    y0 = rep(XALL)[:N]
    print(f"[결합] t=0 재현 오차 최대 {float((y0 - X0).norm(dim=1).max()):.2e}  "
          f"지름 L {L:.4f}  입자 {N}  보조 {AUX.shape[0]}", flush=True)

opt = torch.optim.Adam(rep.parameters(), lr=a.lr)
rows, Yh, Ah = [], [y0.cpu().numpy().astype(np.float16)], []
if AUX.shape[0]:
    with torch.no_grad():
        Ah.append(rep(XALL)[N:].cpu().numpy().astype(np.float16))
t0 = time.time()
for t in range(1, T + 1):
    tgt = TRAJ[t]
    for it in range(a.iters0 if t == 1 else a.iters):
        opt.zero_grad(set_to_none=True)
        Y = rep(XALL)[:N]
        l2 = ((Y - tgt) ** 2).sum(1).mean()
        dt_ = detF(Y)
        bar = torch.relu(a.tau - dt_).pow(2).mean()
        (l2 + a.lam_inv * bar).backward()
        prev = [q.detach().clone() for q in rep.parameters()]
        opt.step()
        # det F > 0 을 **항상** 지킨다: 스텝 뒤 뒤집힌 입자가 생기면 스텝을 반씩
        # 줄여 되돌린다 (시작은 det=1 이라 늘 실현가능한 쪽에 머문다)
        with torch.no_grad():
            for _bt in range(10):
                if float(detF(rep(XALL)[:N]).min()) > 0:
                    break
                for q, q0 in zip(rep.parameters(), prev):
                    q.copy_(q0 + 0.5 * (q - q0))
            else:
                for q, q0 in zip(rep.parameters(), prev):
                    q.copy_(q0)
    with torch.no_grad():
        Yall = rep(XALL)
        Y = Yall[:N]
        dt_ = detF(Y)
        rmse = float(((Y - tgt) ** 2).sum(1).mean().sqrt()) / L
        cd = chamfer(Y, tgt) / L
        e = emd(Y, tgt) / L if (t % a.emd_every == 0 or t == T) else float("nan")
        rows.append((t, rmse, cd, e, float(dt_.min()), float((dt_ <= 0).float().mean())))
        Yh.append(Y.cpu().numpy().astype(np.float16))
        if AUX.shape[0]:
            Ah.append(Yall[N:].cpu().numpy().astype(np.float16))
    if t % 10 == 0 or t == 1:
        print(f"  [t={t:3d}] RMSE {100*rmse:.3f}%  CD {100*cd:.3f}%  "
              + (f"EMD {100*e:.3f}%  " if e == e else "")
              + f"det 최소 {rows[-1][4]:.3f} (≤0 {100*rows[-1][5]:.2f}%)  "
              f"{time.time()-t0:.0f}s", flush=True)

R = np.array(rows)
emd_v = R[:, 3][~np.isnan(R[:, 3])]
print(f"[요약] {a.method}  자유도 {rep.dof}  RMSE {100*R[:,1].mean():.3f}%  "
      f"CD {100*R[:,2].mean():.3f}%  EMD {100*emd_v.mean():.3f}%  "
      f"det 최소 {R[:,4].min():.3f}  뒤집힘 최대 {100*R[:,5].max():.2f}%", flush=True)
np.savez_compressed(a.out, metrics=R, dof=rep.dof, L=L, Y=np.stack(Yh),
                    **({"AUXY": np.stack(Ah)} if AUX.shape[0] else {}))
print(f"[저장] {a.out}", flush=True)
