"""표현력 비교: 목표 궤적(exe/gauss_flow.py)을 각 표현이 자기 자유도로 따라간다.

프레임마다 이전 해에서 출발해 같은 반복 수의 Adam 으로

    loss = mean_p |y_p(θ) - y*_p|²  +  λ_inv · mean_p relu(τ - det F_p)²

를 줄인다. det F 는 **모든 방법에 같은 방식**으로 잰다 -- 정지 상태 이웃 k 개로
최소제곱 F_p = (Σ dy dxᵀ)(Σ dx dxᵀ)⁻¹ (방법마다 사상의 미분 형태가 달라 이렇게
통일한다). 자유도 예산은 --dof 로 맞춘다 (실제 개수를 결과에 남긴다).

표현 (모두 정지 위치 X 의 함수 y = f_θ(X)):
  ours       **매 프레임 직전 위치에 사면체 격자를 새로 깔아** 입자-격자 대응을
             다시 잡고, 그 프레임의 격자점 증분 변위 u_i + 꺾임 반경 ρ_i (4/꼭짓점)
             y = P + Σ_i w_i ψ_i(r_i) u_i / Σ_j w_j ψ_j(r_j)   (P = 직전 위치)
             w_i = λ_i² / Σ λ_j²  (Gregory 볼록 결합: 맞은편 면에서 값·기울기 0)
             ψ_i(r) = 1 - a (r/ρ_i)²            (r < ρ_i, 중심 기울기 0)
                    = (1-a) / (1 + (r-ρ_i)/ℓ)    (r ≥ ρ_i, 기울기가 꺾인다)
  vrgs       GS-Verse(VR-GS): 표면 삼각형 메시 꼭짓점 (3/꼭짓점)
             y = v0 + a1 e1 + a2 e2 + b n (lib/anchorflow/gsverse.py 결합 그대로)
  phystwin   PhysTwin: 제어점 변위 (3/제어점). 입자 가중치는 공식처럼 매 프레임
             직전 위치로 다시 계산한다. 제어점마다 이웃 16 개로 회전을
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
                choices=["ours", "tet", "vrgs", "phystwin", "gaussim", "simplicits"])
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
ap.add_argument("--gaussim_official", action="store_true",
                help="gaussim 표현을 공식대로: 3 단 계층(downsample 0.01, 0.01), "
                     "F = U·diag(exp s 부피 정규화)·Vᵀ (|s|≤5)")
ap.add_argument("--pt_noflip", action="store_true",
                help="phystwin: 질량점마다 처음 이웃 16 개로 잰 국소 F 의 det>0 을 지킨다 (뒤집힘 금지)")
ap.add_argument("--pt_spring", type=float, default=0.0,
                help="phystwin: 처음 이웃 간 거리를 묶는 스프링 항 계수 (상대 변형률² 평균)")
ap.add_argument("--fixed_bind", action="store_true",
                help="ours/tet: 격자·입자-격자 대응을 처음(정지 위치)에 한 번 잡고 고정 (매 프레임 재설정 안 함)")
ap.add_argument("--tb", default="auto",
                help="TensorBoard 디렉토리 (auto: /home/dkta/work/tbrf/<폴더>_<파일>, none: 끔)")
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
def bind_lattice(P, per_node, dof, n0=6.0):
    """현재 위치 P 에 사면체 격자를 새로 깔고 대응을 잡는다 (꼭짓점 수를 dof/per_node 에)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
    from anchorflow import simplex as sx
    n_nodes = n0
    for _ in range(80):
        lo, lat, nn = sx.grid_for_nodes(P[:N], n_nodes)
        idx, lam, _ = sx.locate(P, lo, lat, nn)
        rows, uniq = sx.active_nodes(idx)
        if per_node * uniq.numel() >= dof:
            break
        n_nodes *= 1.03
    return rows, lam.detach(), sx.node_pos(lo, lat, nn, uniq), lat.s, n_nodes


class Ours(torch.nn.Module):
    """매 프레임 **직전 위치**에 사면체 격자를 새로 깔고(입자-격자 대응 재설정),
    그 프레임의 격자점 증분 변위로 스키닝한다:
        y = P + Σ_i W_i(P) u_i,  W_i = w_i ψ_i / Σ_j w_j ψ_j,  w_i = λ_i²/Σλ_j²
    자유도는 꼭짓점당 4 (u 3 + 꺾임 반경 ρ 1)."""
    rebind_each_frame = True

    def __init__(self, X, dof):
        super().__init__()
        self.dof_budget, self.n0 = dof, 6.0
        self.a = 0.5
        self.rebind(X)

    def rebind(self, P):
        dev_ = P.device
        self.rows, self.lam, self.Xn, self.h, self.n0 = bind_lattice(
            P.detach(), 4, self.dof_budget, max(self.n0 / 1.2, 3.0))
        M = self.Xn.shape[0]
        self.ell = 0.5 * self.h
        self.u = torch.nn.Parameter(torch.zeros(M, 3, device=dev_))
        self.rho_raw = torch.nn.Parameter(torch.zeros(M, device=dev_))
        self.dof = 4 * M

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


class TetOnly(torch.nn.Module):
    """우리 표현에서 Gregory·방사형 함수를 뺀 것: 사면체 무게중심(선형) 보간만.
    격자 재설정은 우리와 같다. y = P + Σ_i λ_i u_i  (꼭짓점당 자유도 3)."""
    rebind_each_frame = True

    def __init__(self, X, dof):
        super().__init__()
        self.dof_budget, self.n0 = dof, 6.0
        self.rebind(X)

    def rebind(self, P):
        self.rows, self.lam, self.Xn, self.h, self.n0 = bind_lattice(
            P.detach(), 3, self.dof_budget, max(self.n0 / 1.2, 3.0))
        M = self.Xn.shape[0]
        self.u = torch.nn.Parameter(torch.zeros(M, 3, device=P.device))
        self.dof = 3 * M

    def forward(self, X):
        return X + (self.lam[..., None] * self.u[self.rows]).sum(1)


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
    """제어점(뼈) 변위로 입자를 옮긴다. 공식 gs_render_dynamics.py 처럼 이웃 관계는
    처음 위치로 한 번(relations, K=16), 입자 가중치는 **매 프레임 직전 위치**로
    다시 계산한다(knn_weights, K=16). 그 프레임의 뼈 이동 m 이 자유도다."""
    rebind_each_frame = True

    def __init__(self, X, dof, K=16):
        super().__init__()
        nb = dof // 3
        self.K = K
        self.B = X0[fps(X0, nb)].clone()                    # 뼈 위치 (매 프레임 갱신)
        d = torch.cdist(self.B, self.B)
        self.rel = d.topk(K + 1, largest=False).indices[:, 1:]   # 처음 위치로 한 번
        # 질량점 토폴로지: 처음 위치 기준 이웃 차이 / 거리 (뒤집힘·찢어짐 판정용)
        self.B0 = self.B.clone()
        A0 = self.B0[self.rel] - self.B0[:, None]
        self.A0inv = torch.linalg.inv(A0.transpose(1, 2) @ A0)
        self.L0 = A0.norm(dim=-1)
        self.dof = 3 * nb
        self.rebind(X)

    def rebind(self, P):
        if hasattr(self, "m"):                              # 직전 프레임의 뼈 이동 반영
            self.B = (self.B + self.m.detach()).clone()
        dd, ii = [], []
        for i in range(0, P.shape[0], 20000):
            d_, i_ = torch.cdist(P[i:i + 20000].detach(), self.B).topk(self.K, largest=False)
            dd.append(d_); ii.append(i_)
        dd, ii = torch.cat(dd), torch.cat(ii)
        w = 1.0 / (dd + 1e-6)
        self.wi, self.ww = ii, w / w.sum(1, keepdim=True)
        self.m = torch.nn.Parameter(torch.zeros_like(self.B))

    def node_det(self):
        """질량점마다 처음 이웃 대비 국소 F 의 det (최소제곱)."""
        Bn = self.B + self.m
        A1 = Bn[self.rel] - Bn[:, None]
        A0 = self.B0[self.rel] - self.B0[:, None]
        F = (A1.transpose(1, 2) @ A0) @ self.A0inv
        return torch.linalg.det(F)

    def spring(self):
        """처음 이웃 간 거리 보존 (상대 변형률² 평균)."""
        Bn = self.B + self.m
        L = (Bn[self.rel] - Bn[:, None]).norm(dim=-1)
        return ((L / self.L0 - 1.0) ** 2).mean()

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


def quat_mat(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                       -1).reshape(*q.shape[:-1], 3, 3)


class GausSimOfficial(torch.nn.Module):
    """GausSim 공식 표현 (rep_track2.GausSim 과 같다): 3 단 계층 -- 점 -> 군집(1%) ->
    위 군집(그 1%), 군집 변환 F = U·diag(exp s / (Π exp s)^(1/3))·Vᵀ (|s|≤5, 부피 보존),
    점 y = P + p + F (X - P)  (P = 위 군집 중심, p = 위 군집 이동). 자유도는 공식 값."""

    def __init__(self, X):
        super().__init__()
        NG = X0.shape[0]
        k1 = max(int(round(0.01 * NG)), 2)
        k2 = max(int(round(0.01 * k1)), 1)
        torch.manual_seed(0)
        self.C1 = X0[fps(X0, k1)]
        torch.manual_seed(1)
        self.C2 = self.C1[fps(self.C1, k2)]
        self.par = torch.cdist(self.C1, self.C2).argmin(1)
        self.lab = torch.cat([torch.cdist(X[i:i + 20000], self.C1).argmin(1)
                              for i in range(0, X.shape[0], 20000)])
        self.p2 = torch.nn.Parameter(torch.zeros(k2, 3, device=X.device))
        z = torch.zeros(k1, 11, device=X.device)
        z[:, 0] = 1.0; z[:, 7] = 1.0
        self.dg = torch.nn.Parameter(z)
        self.dof = 3 * k2 + 11 * k1
        print(f"[gaussim 공식] 군집 {k1} / 위 군집 {k2}  자유도 {self.dof}", flush=True)

    def forward(self, X):
        U = quat_mat(self.dg[:, 0:4])
        s = torch.exp(self.dg[:, 4:7].clamp(-5, 5))
        s = s / torch.prod(s, -1, keepdim=True).pow(1 / 3)
        V = quat_mat(self.dg[:, 7:11])
        F = (U @ torch.diag_embed(s) @ V.transpose(1, 2))[self.lab]
        P = self.C2[self.par[self.lab]]
        return P + self.p2[self.par[self.lab]] + (F @ (X - P)[..., None]).squeeze(-1)


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
elif a.method == "tet":
    rep = TetOnly(XALL, a.dof)
elif a.method == "vrgs":
    rep = VRGS(XALL, a.dof)
elif a.method == "phystwin":
    rep = PhysTwin(XALL, a.dof)
elif a.method == "gaussim":
    rep = GausSimOfficial(XALL) if a.gaussim_official else GausSim(XALL, a.dof)
else:
    rep = Simplicits(XALL, a.dof, a.simp_w)
rep = rep.to(dev)

# det F: 정지 이웃 k 개 최소제곱 (모든 방법 공통)
nbr = torch.cat([torch.cdist(X0[i:i + 8192], X0).topk(a.knn_F + 1, largest=False).indices[:, 1:]
                 for i in range(0, N, 8192)])                 # 큰 N 은 나눠서
def _ref(Pref):
    dX = Pref[nbr] - Pref[:, None]                          # [N,k,3]
    return dX, torch.linalg.inv(dX.transpose(1, 2) @ dX
                                + 1e-12 * torch.eye(3, device=dev))


REF0 = _ref(X0)


def detF(Y, ref=None):
    """기준(ref) 대비 Y 의 이웃 최소제곱 F 의 det. 격자를 매 프레임 새로 까는
    방법은 기준이 **직전 프레임 위치**(증분 사상), 나머지는 정지 위치다."""
    dX, Binv = ref if ref is not None else REF0
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

if a.fixed_bind:
    rep.rebind_each_frame = False                           # 처음 대응 고정: u 는 정지 대비 누적 변위
REB = getattr(rep, "rebind_each_frame", False)
TBW = None
if a.tb != "none":
    import os
    from torch.utils.tensorboard import SummaryWriter
    _tb = a.tb if a.tb != "auto" else os.path.join(
        "/home/dkta/work/tbrf", os.path.basename(os.path.dirname(os.path.abspath(a.out)))
        + "_" + os.path.splitext(os.path.basename(a.out))[0])
    TBW = SummaryWriter(_tb)
    print(f"[TB] {_tb}", flush=True)
opt = torch.optim.Adam(rep.parameters(), lr=a.lr)
rows, Yh, Ah = [], [y0.cpu().numpy().astype(np.float16)], []
CURVE = []          # 프레임별 러닝 커브: [(t, it, L2, 장벽, 되돌림 횟수)]
DOFS = []
if AUX.shape[0]:
    with torch.no_grad():
        Ah.append(rep(XALL)[N:].cpu().numpy().astype(np.float16))
Pref = XALL.clone()          # 격자 재설정 방법의 기준 위치 (직전 프레임)
t0 = time.time()
for t in range(1, T + 1):
    tgt = TRAJ[t]
    if REB:
        if t > 1:
            rep.rebind(Pref)                                # 입자-격자 대응을 새로
            rep.to(dev)
            opt = torch.optim.Adam(rep.parameters(), lr=a.lr)
        ref = _ref(Pref[:N])
        inp = Pref
    else:
        ref, inp = None, XALL
    DOFS.append(rep.dof)
    for it in range(a.iters0 if t == 1 else a.iters):
        opt.zero_grad(set_to_none=True)
        Y = rep(inp)[:N]
        l2 = ((Y - tgt) ** 2).sum(1).mean()
        dt_ = detF(Y, ref)
        bar = torch.relu(a.tau - dt_).pow(2).mean()
        extra = 0.0
        if a.method == "phystwin" and a.pt_noflip:
            bar = bar + torch.relu(a.tau - rep.node_det()).pow(2).mean()
        if a.method == "phystwin" and a.pt_spring > 0:
            extra = a.pt_spring * rep.spring()
        (l2 + a.lam_inv * bar + extra).backward()
        prev = [q.detach().clone() for q in rep.parameters()]
        opt.step()
        # det F > 0 을 **항상** 지킨다: 스텝 뒤 뒤집힌 입자가 생기면 스텝을 반씩
        # 줄여 되돌린다 (재설정 방법은 증분 사상 기준, 시작은 det=1)
        nbt = 0
        with torch.no_grad():
            for _bt in range(10):
                ok = float(detF(rep(inp)[:N], ref).min()) > 0
                if ok and a.method == "phystwin" and a.pt_noflip:
                    ok = float(rep.node_det().min()) > 0
                if ok:
                    break
                nbt += 1
                for q, q0 in zip(rep.parameters(), prev):
                    q.copy_(q0 + 0.5 * (q - q0))
            else:
                for q, q0 in zip(rep.parameters(), prev):
                    q.copy_(q0)
        CURVE.append((t, it, float(l2), float(bar), nbt))
    with torch.no_grad():
        Yall = rep(inp)
        Y = Yall[:N]
        dt_ = detF(Y, ref)                                  # 이 프레임 사상의 det
        rmse = float(((Y - tgt) ** 2).sum(1).mean().sqrt()) / L
        cd = chamfer(Y, tgt) / L
        e = (emd(Y, tgt) / L if a.emd_n > 0 and (t % a.emd_every == 0 or t == T)
             else float("nan"))                             # --emd_n 0: 저장한 Y 로 따로 잰다
        rows.append((t, rmse, cd, e, float(dt_.min()), float((dt_ <= 0).float().mean())))
        Yh.append(Y.cpu().numpy().astype(np.float16))
        if AUX.shape[0]:
            Ah.append(Yall[N:].cpu().numpy().astype(np.float16))
        Pref = Yall.detach().clone()
    if TBW is not None:
        for k_, v_ in (("RMSE_pct", 100 * rmse), ("CD_pct", 100 * cd), ("knn_det_min", rows[-1][4]),
                       ("inverted_pct", 100 * rows[-1][5]), ("dof", rep.dof),
                       ("time_s", time.time() - t0)):
            TBW.add_scalar("frame/" + k_, v_, t)
        for c_ in [c for c in CURVE if c[0] == t]:
            TBW.add_scalar("iter/l2", c_[2], len(CURVE) - sum(1 for c in CURVE if c[0] == t) + c_[1])
        TBW.flush()
    if t % 10 == 0 or t == 1:
        print(f"  [t={t:3d}] RMSE {100*rmse:.3f}%  CD {100*cd:.3f}%  "
              + (f"EMD {100*e:.3f}%  " if e == e else "")
              + f"det 최소 {rows[-1][4]:.3f} (≤0 {100*rows[-1][5]:.2f}%)  "
              f"자유도 {rep.dof}  "
              + (f"질량점 det 최소 {float(rep.node_det().min()):.3f}  변형률 RMS "
                 f"{float(rep.spring().sqrt()):.3f}  " if a.method == "phystwin" else "")
              + f"{time.time()-t0:.0f}s", flush=True)

R = np.array(rows)
emd_v = R[:, 3][~np.isnan(R[:, 3])] if (~np.isnan(R[:, 3])).any() else np.array([np.nan])
print(f"[요약] {a.method}  자유도 {np.mean(DOFS):.0f} (프레임 평균)  RMSE {100*R[:,1].mean():.3f}%  "
      f"CD {100*R[:,2].mean():.3f}%  EMD {100*emd_v.mean():.3f}%  "
      f"det 최소 {R[:,4].min():.3f}  뒤집힘 최대 {100*R[:,5].max():.2f}%", flush=True)
np.savez_compressed(a.out, metrics=R, dof=np.array(DOFS), L=L, Y=np.stack(Yh),
                    curve=np.array(CURVE, dtype=np.float64),
                    AUXY=(np.stack(Ah) if AUX.shape[0] else np.stack(Yh)))  # 보조 없으면 입자 자체
print(f"[저장] {a.out}", flush=True)
