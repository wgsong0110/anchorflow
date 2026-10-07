"""표현력 비교 (공식 설정판): 가우시안 중심을 목표 흐름대로 따라가게 한다.

입자 = 원본 3DGS 가우시안 중심 (내부 채움 없음). 목표 = 같은 속도장으로 그 점들을
흘린 것 (exe/gauss_flow.py --pts). 최적화·평가·유효 격자·영상 모두 같은 부분표본 N 개.

각 방법은 **논문·공식 코드의 설정 그대로** 쓴다 (자유도도 공식 값, 결과에 기록):
  ours       사면체 격자 (node_h = PG dx·(2√2)^(1/3)), 매 프레임 직전 위치에 격자를 다시
             깔아 대응 재설정, 증분 변위 + 꺾임 반경, Gregory 볼록 결합 w=λ²/Σλ²
  tet        위에서 Gregory·방사형을 뺀 무게중심 선형 보간 (같은 격자)
  phystwin   스프링 질점 = 가우시안 중심 11024 점을 복셀 0.005 로 줄인 것
             (data_process_sample.py 기본값; 길이는 그쪽 물체 25 cm 대비 비율로 옮김),
             이웃 16 (처음 위치), 입자 가중치 16 (매 프레임 직전 위치, 역거리),
             뼈 회전 = 이웃 Procrustes (interpolate_motions)
  gaussim    3 단 계층 (기본 설정 downsample 0.01, 0.01, 거리 기반 표본),
             F = U·diag(exp s 정규화)·Vᵀ (|s|≤5, 부피 보존), 점 x = p + F(X - P)
  simplicits kaolin 기본 학습 가중치 (핸들 10) + 핸들별 3x4 변환
  vrgs       GS-Verse: 표면 메시(가우시안 중심 marching cubes, n_grid 100) 꼭짓점,
             가우시안은 가까운 삼각형의 국소 틀 좌표로 결합 (gsverse.bind_points)

최적화는 우리 원래 설정을 모두에 같게 쓴다: 프레임마다 L-BFGS (lr 1, 반복 500,
이력 50, strong Wolfe), 손실 = 입자 L2 평균 + κ·det F 로그 장벽 (κ=1, Ĵ=0.3, J≤0
선형 연장). det F 는 각 방법 사상의 **해석적/자동미분 야코비안**이고 정지 상태 대비
누적이다 (격자 재설정 방법은 증분 F 를 곱해 누적한다).

렌더 공분산도 방법별 공식 방식: ours/tet/simplicits/gaussim 은 FΣFᵀ, phystwin 은
뼈 회전 섞기 RΣRᵀ, vrgs 는 삼각형 틀 F 로 FΣFᵀ.

  python exe/rep_track2.py --flow repflow/gflow_wolf.npz --aux repflow/aux_wolf.npz \
      --fill pgfill_wolf.npy --method ours --out repflow/r2/wolf_ours.npz \
      --video repflow/r2/wolf_ours.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
from anchorflow import simplex as sx                                # noqa: E402
from anchorflow import gsverse as gv                                # noqa: E402
from anchorflow.phys_resid import _barrier_b                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--flow", required=True)
ap.add_argument("--aux", required=True)
ap.add_argument("--fill", default="", help="(쓰지 않는다 -- 내부 채움 없는 3DGS 만)")
ap.add_argument("--method", required=True,
                choices=["ours", "tet", "phystwin", "gaussim", "simplicits", "vrgs"])
ap.add_argument("--out", required=True)
ap.add_argument("--video", default="")
ap.add_argument("--simp", default="", help="simplicits: 학습된 가중치 함수 (.pt)")
ap.add_argument("--max_iter", type=int, default=500)
ap.add_argument("--kappa", type=float, default=1.0)
ap.add_argument("--jhat", type=float, default=0.3)
ap.add_argument("--emd_every", type=int, default=10)
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--fp64", action="store_true", help="모든 계산을 float64 로")
ap.add_argument("--iters0", type=int, default=400, help="adam_bt: 첫 프레임 반복")
ap.add_argument("--iters", type=int, default=200, help="adam_bt: 이후 프레임 반복")
ap.add_argument("--lr", type=float, default=1e-3, help="adam_bt: Adam 학습률")
ap.add_argument("--lam_inv", type=float, default=100.0, help="adam_bt: 장벽 계수")
ap.add_argument("--tau", type=float, default=0.1, help="adam_bt: det 장벽 문턱")
ap.add_argument("--knn_F", type=int, default=8, help="adam_bt: det 를 잴 이웃 수")
ap.add_argument("--riem_eps", type=float, default=1e-2,
                help="riem: 계량 G = JᵀJ + ε·λmax·I 의 상대 ε (λmax 는 프레임마다 거듭제곱법)")
ap.add_argument("--profile", type=int, default=0,
                help="riem: 첫 프레임 이 반복 수만 돌며 구간별 시간 + torch 프로파일러 표를 내고 끝낸다")
ap.add_argument("--riem_energy", default="elastic", choices=["elastic", "logbarrier"],
                help="riem 계량의 에너지: elastic (스프링/StVK 막/고정공회전) | logbarrier "
                     "(det 의 로그 장벽, 잔차 log J, 계량 Σ ∇logJ ∇logJᵀ: ours/tet 격자 셀, "
                     "vrgs 삼각형 셀, gaussim 입자)")
ap.add_argument("--tet_quality", type=float, default=0.05,
                help="phystwin logbarrier: 4-클릭 사면체 중 정지 품질 6√2·V/l_rms³ 이 이보다 작은(납작한) 것은 버린다")
ap.add_argument("--ipm_mu0", type=float, default=1e-4, help="ipm: 첫 장벽 계수 μ")
ap.add_argument("--ipm_stages", type=int, default=4, help="ipm: μ 를 10 배씩 줄이는 단계 수 (프레임마다)")
ap.add_argument("--save_traj", action="store_true",
                help="모든 프레임의 가우시안 위치(float16)를 저장 (물리 지표·EMD 재측정용)")
ap.add_argument("--riem_k", type=float, default=1.0,
                help="riem: 탄성 강성 배수 k (G = k·JᵀJ + ε·λmax(k=1)·I -- ε 는 k=1 기준으로 고정)")
ap.add_argument("--riem_lr", type=float, default=1.0, help="riem: 고정 보폭 η")
ap.add_argument("--riem_cg", type=int, default=50, help="riem: CG 최대 반복")
ap.add_argument("--pt_realtime", action="store_true",
                help="phystwin: 공식 실시간 데모 방식 -- 이웃 질량점 인덱스는 첫 프레임에 한 번, "
                     "매 프레임 그 이웃까지 거리로 가중치만 다시")
ap.add_argument("--tb", default="auto",
                help="TensorBoard 디렉토리 (auto: /home/dkta/work/tbrf/<폴더>_<파일>, none: 끔)")
ap.add_argument("--opt", default="lbfgs", choices=["lbfgs", "gd", "adam_bt", "adam", "riem", "ipm"],
                help="lbfgs (원래 설정) / gd: 고정 학습률 경사하강")
ap.add_argument("--gd_lr", type=float, default=1.0)
ap.add_argument("--frames", type=int, default=0, help="앞 몇 프레임만 (0 이면 전부)")
ap.add_argument("--autograd_jac", action="store_true",
                help="야코비안을 예전처럼 자동미분 2 차로 (검증·비교용, 느리다)")
ap.add_argument("--no_compile", action="store_true", help="torch.compile 끄기")
ap.add_argument("--check_jac", action="store_true",
                help="해석적 야코비안을 자동미분과 비교하고 끝낸다")
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)
if a.fp64:
    torch.set_default_dtype(torch.float64)
DT = torch.float64 if a.fp64 else torch.float32

D = np.load(a.flow)
AX = np.load(a.aux, allow_pickle=True)
G0 = torch.as_tensor(AX["G"], dtype=DT, device=dev)                # 전체 가우시안
FI = torch.as_tensor(D["idx"], device=dev)                           # 맞추는 부분표본
TRAJ = torch.as_tensor(D["traj"], dtype=DT, device=dev)              # [T+1,N,3]
# 최적화·평가·유효 격자·영상을 **같은 점 집합**(흐름의 부분표본)으로 통일한다
GIDX_SUB = FI.cpu().numpy()
G0 = G0[FI].contiguous()
FI = torch.arange(FI.numel(), device=dev)
T, N, NG = TRAJ.shape[0] - 1, FI.numel(), G0.shape[0]
if a.frames:
    T = min(T, a.frames)
assert torch.allclose(G0[FI], TRAJ[0], atol=1e-5), "흐름과 가우시안 집합이 다르다"
L = float((G0.max(0).values - G0.min(0).values).norm())              # 고정 정규화
S_NORM = float(AX["s"])                                              # 정규화 1 = 시뮬 S_NORM
LO, OFF = AX["lo"], AX["off"]
# 내부 채움 입자는 어디에도 쓰지 않는다 (공식대로 원본 3DGS 가우시안만)


def barrier(J):
    return _barrier_b(J, a.jhat, ext_kind="linear")


def fps(X, k, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    i = [int(torch.randint(X.shape[0], (1,), generator=g))]
    d = (X - X[i[0]]).norm(dim=1)
    for _ in range(k - 1):
        j = int(d.argmax()); i.append(j)
        d = torch.minimum(d, (X - X[j]).norm(dim=1))
    return torch.tensor(i, device=X.device)


def quat_mat(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                       -1).reshape(*q.shape[:-1], 3, 3)


def det3(F):
    """3x3 행렬식 (torch.linalg.det 과 같은 값; 컴파일 그래프가 끊기지 않게 전개)."""
    return (F[:, 0, 0] * (F[:, 1, 1] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 1])
            - F[:, 0, 1] * (F[:, 1, 0] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 0])
            + F[:, 0, 2] * (F[:, 1, 0] * F[:, 2, 1] - F[:, 1, 1] * F[:, 2, 0]))


def outer_sum(A, B):
    """Σ_k A[:,k,:] ⊗ B[:,k,:] -> [n,3,3]. einsum 이 작은 배치 행렬곱(sgemm)으로 가 느려 원소별로."""
    return (A[..., :, None] * B[..., None, :]).sum(1)


def mm3(A, B):
    """[n,3,3] @ [n,3,3] 을 원소별로 (같은 이유)."""
    return (A[..., :, :, None] * B[..., None, :, :]).sum(-2)


def eye_plus(M):
    return M + torch.eye(3, device=M.device, dtype=M.dtype)


def jac(fn, X):
    """y = fn(X) 와 J = ∂y/∂X (점별, create_graph)."""
    X = X.detach().requires_grad_(True)
    y = fn(X)
    rows = [torch.autograd.grad(y[:, i].sum(), X, create_graph=True)[0] for i in range(3)]
    return y, torch.stack(rows, 1)


# ============================================================== 계량용 순수 잔차 (컴파일용)
def res_lattice_lb(x, Xn, cells, cD0i):
    u = x[:Xn.numel()].reshape(-1, 3)
    X = (Xn + u)[cells]
    D = (X[:, 1:] - X[:, :1]).transpose(1, 2)
    J = det3(mm3(D, cD0i))
    return torch.log(J.clamp_min(1e-6)) * (cells.shape[0] ** -0.5)


def res_spring(x, B, rel, L0):
    Bn = B + x.reshape(-1, 3)
    return ((Bn[rel] - Bn[:, None]).norm(dim=-1) - L0).reshape(-1)


def res_tri_lb(x, f, n0, A0):
    v = x.reshape(-1, 3)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    c = torch.cross(v1 - v0, v2 - v0, dim=-1)
    J = (c * n0).sum(-1) / (2 * A0)
    return A0.sqrt() * torch.log(J.clamp_min(1e-6))


def res_simp_lb(x, W, dW, Xh):
    Tm = x.reshape(-1, 3, 4)
    TX = (Tm[None] * Xh[:, None, None, :]).sum(-1)                    # [n,K,3]
    F = eye_plus((W[..., None, None] * Tm[None, :, :, :3]).sum(1) + outer_sum(TX, dW))
    return torch.log(det3(F).clamp_min(1e-6)) * (W.shape[0] ** -0.5)


def _gn_prod(fn, th, u, *args):
    """가우스-뉴턴 곱 Jᵀ J u (J = ∂fn/∂θ)."""
    from torch.func import jvp, vjp
    _, Ju = jvp(lambda z: fn(z, *args), (th,), (u,))
    _, vf = vjp(lambda z: fn(z, *args), th)
    return vf(Ju)[0]


GN_PROD = None


# ============================================================== 표현들
class Lattice(torch.nn.Module):
    """ours / tet: 매 프레임 직전 위치에 격자를 새로 깔고 증분 변위로 스키닝."""
    rebind_each_frame = True

    def __init__(self, gregory):
        super().__init__()
        self.greg = gregory
        # 원래 조합: node_h = MPM dx·(2√2)^(1/3). PG 채우기 격자 dx = 2/100 (시뮬)
        self.h = (2.0 / 100.0) * (2.0 * math.sqrt(2.0)) ** (1.0 / 3.0) / S_NORM
        self.a = 0.5

    def rebind(self, Pall):
        lo, lat, nn = sx.grid_for_nodes(Pall, 1.0, h_fix=self.h)
        idx, lam, _ = sx.locate(Pall, lo, lat, nn)
        rows, uniq = sx.active_nodes(idx)
        self.rows = rows
        self.Xn = sx.node_pos(lo, lat, nn, uniq)
        Xv = self.Xn[rows]                                          # [NG,4,3]
        self.v0 = Xv[:, 0]
        self.Minv = torch.linalg.inv((Xv[:, 1:] - Xv[:, :1]).transpose(1, 2))
        M = uniq.numel()
        # 프레임 안에서 X(=직전 위치)와 격자는 고정 -> λ, ∇λ, r, ∇r, w, ∇w 는 상수
        X = Pall
        l3 = (self.Minv @ (X - self.v0)[..., None]).squeeze(-1)
        self.lam = torch.cat([1.0 - l3.sum(1, keepdim=True), l3], 1)        # [N,4]
        self.dlam = torch.cat([-self.Minv.sum(1, keepdim=True), self.Minv], 1)  # [N,4,3]
        if self.greg:
            d = X[:, None] - self.Xn[rows]
            self.r = d.norm(dim=-1)
            self.dr = d / self.r[..., None].clamp_min(1e-30)
            S = (self.lam * self.lam).sum(1, keepdim=True)
            self.w = self.lam * self.lam / S.clamp_min(1e-12)
            dS = (2 * self.lam[..., None] * self.dlam).sum(1, keepdim=True)
            self.dw = 2 * self.lam[..., None] * self.dlam / S[..., None] \
                - (self.lam * self.lam)[..., None] * dS / (S * S)[..., None]
        # 셀: 이 프레임 입자가 들어 있는 사면체들 (꼭짓점 순서 그대로 -> 정지 det 부호 유지)
        self.cells = torch.unique(rows, dim=0)
        Xc = self.Xn[self.cells]
        self.cD0i = torch.linalg.inv((Xc[:, 1:] - Xc[:, :1]).transpose(1, 2))
        self.u = torch.nn.Parameter(torch.zeros(M, 3, device=dev))
        self.rho_raw = torch.nn.Parameter(torch.zeros(M, device=dev)) if self.greg else None
        self.dof = (4 if self.greg else 3) * M

    def params(self):
        return [self.u] + ([self.rho_raw] if self.greg else [])

    def yJ(self, X, sel):
        return self.yJ_p(X, sel, self.u, self.rho_raw)

    def cellJ(self, u, rho_raw=None):
        """셀 det 와 가중치 w² (Kuhn 사면체는 부피가 같아 1/셀 수)."""
        X = (self.Xn + u)[self.cells]
        D = (X[:, 1:] - X[:, :1]).transpose(1, 2)
        J = det3(mm3(D, self.cD0i))
        return J, torch.full_like(J, 1.0 / self.cells.shape[0])

    def logbarrier_residual(self, u, rho_raw=None):
        """셀(사면체) det 로그 장벽: log J_c, J_c = det(D(Xn+u) D0⁻¹) -- 이 프레임 격자 대비 증분.
        Kuhn 사면체는 부피가 모두 같아 가중치는 1. ρ 는 셀 det 에 들어가지 않는다."""
        X = (self.Xn + u)[self.cells]
        D = (X[:, 1:] - X[:, :1]).transpose(1, 2)
        J = det3(mm3(D, self.cD0i))
        return torch.log(J.clamp_min(1e-6)) * (1.0 / math.sqrt(self.cells.shape[0]))

    def yJ_p(self, X, sel, u, rho_raw):
        """해석적 변위 y - X 와 ∂y/∂X (forward 와 같은 식을 손으로 미분). 매개변수를 인자로."""
        rows = self.rows[sel]
        U = u[rows]                                                 # [n,4,3]
        if not self.greg:
            dy = (self.lam[sel][..., None] * U).sum(1)
            return dy, eye_plus(outer_sum(U, self.dlam[sel]))
        r, dr, w, dw = self.r[sel], self.dr[sel], self.w[sel], self.dw[sel]
        rho = self.h * (0.05 + 0.95 * torch.sigmoid(rho_raw)[rows])
        ins = r < rho
        inner = 1.0 - self.a * (torch.minimum(r, rho) / rho) ** 2
        q = 1.0 + (r - rho).clamp_min(0.0) / (0.5 * self.h)
        outer = (1.0 - self.a) / q
        psi_raw = torch.where(ins, inner, outer)
        psi = psi_raw.clamp_min(1e-6)
        dpsi = torch.where(ins, -2.0 * self.a * r / (rho * rho),
                           -(1.0 - self.a) / (0.5 * self.h) / (q * q)) * (psi_raw > 1e-6)
        g = w * psi
        dg = dw * psi[..., None] + (w * dpsi)[..., None] * dr
        G = g.sum(1, keepdim=True).clamp_min(1e-12)
        W = g / G
        dW = dg / G[..., None] - W[..., None] * dg.sum(1, keepdim=True) / G[..., None]
        dy = (W[..., None] * U).sum(1)
        return dy, eye_plus(outer_sum(U, dW))

    def forward(self, X, sel):
        rows = self.rows[sel]
        l3 = (self.Minv[sel] @ (X - self.v0[sel])[..., None]).squeeze(-1)
        lam = torch.cat([1.0 - l3.sum(1, keepdim=True), l3], 1)    # X 의 함수
        if not self.greg:
            return X + (lam[..., None] * self.u[rows]).sum(1)
        r = (X[:, None] - self.Xn[rows]).norm(dim=-1)
        rho = self.h * (0.05 + 0.95 * torch.sigmoid(self.rho_raw)[rows])
        inner = 1.0 - self.a * (torch.minimum(r, rho) / rho) ** 2
        outer = (1.0 - self.a) / (1.0 + (r - rho).clamp_min(0.0) / (0.5 * self.h))
        psi = torch.where(r < rho, inner, outer).clamp_min(1e-6)
        w = lam * lam
        w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
        g = w * psi
        W = g / g.sum(1, keepdim=True).clamp_min(1e-12)
        return X + (W[..., None] * self.u[rows]).sum(1)


class PhysTwin(torch.nn.Module):
    rebind_each_frame = True

    def __init__(self, K=16):
        super().__init__()
        # data_process_sample.py: 표면 1024 + 1 만 점, 복셀 0.005 (그쪽 물체 ~0.25 m).
        # 내부 채움 없이 가우시안 중심에서 뽑는다.
        g = torch.Generator(device="cpu").manual_seed(0)
        P = G0[torch.randperm(NG, generator=g)[:11024].to(dev)]
        vox = 0.005 / 0.25
        key = torch.floor((P - P.min(0).values) / vox).long()
        _, inv = torch.unique(key, dim=0, return_inverse=True)
        B = torch.zeros(inv.max() + 1, 3, device=dev).index_reduce_(
            0, inv, P, "mean", include_self=False)
        self.B, self.K = B, K
        self.rel = torch.cdist(B, B).topk(K + 1, largest=False).indices[:, 1:]
        self.B0 = B.clone()                                  # 스프링 정지 길이 기준 (처음 위치)
        self.L0 = (B[self.rel] - B[:, None]).norm(dim=-1)
        self.m = None
        self.dof = 3 * B.shape[0]

    def rebind(self, Pall):
        if self.m is not None:
            self.B = (self.B + self.m.detach()).clone()
        if a.pt_realtime and getattr(self, "wi", None) is not None:
            ii = self.wi                                     # 실시간 데모: 이웃 인덱스 고정
        else:
            ii = torch.cat([torch.cdist(Pall[i:i + 20000], self.B).topk(self.K, largest=False).indices
                            for i in range(0, Pall.shape[0], 20000)])
        self.wi = ii
        # 프레임 안에서 X 와 B 는 고정 -> 역거리 가중치와 그 기울기는 상수
        d = Pall[:, None] - self.B[ii]
        dn = d.norm(dim=-1)
        qd = 1.0 / (dn + 1e-6)
        Q = qd.sum(1, keepdim=True)
        dq = -(qd * qd)[..., None] * d / dn[..., None].clamp_min(1e-30)
        self.w = qd / Q
        self.dw = dq / Q[..., None] - qd[..., None] * dq.sum(1, keepdim=True) / (Q * Q)[..., None]
        self.d = d
        self.m = torch.nn.Parameter(torch.zeros_like(self.B))

    def params(self):
        return [self.m]

    def build_tets(self, qmin):
        """질량점 스프링 그래프(처음 이웃 16, 대칭화)의 4-클릭을 사면체 셀로. 납작한 것은 버린다."""
        n = self.B0.shape[0]
        A = torch.zeros(n, n, dtype=torch.bool, device=dev)
        A[torch.arange(n, device=dev)[:, None], self.rel] = True
        A = A | A.T
        iu = torch.nonzero(torch.triu(A, 1))                         # 변 (i<j)
        tris = []
        for c in range(0, iu.shape[0], 4096):                         # 삼각형 (i<j<k)
            e = iu[c:c + 4096]
            cm = A[e[:, 0]] & A[e[:, 1]]
            cm &= torch.arange(n, device=dev)[None] > e[:, 1:2]
            r, k = torch.nonzero(cm, as_tuple=True)
            tris.append(torch.cat([e[r], k[:, None]], 1))
        tris = torch.cat(tris)
        tets = []
        for c in range(0, tris.shape[0], 4096):                       # 4-클릭 (i<j<k<l)
            tr = tris[c:c + 4096]
            cm = A[tr[:, 0]] & A[tr[:, 1]] & A[tr[:, 2]]
            cm &= torch.arange(n, device=dev)[None] > tr[:, 2:3]
            r, l = torch.nonzero(cm, as_tuple=True)
            tets.append(torch.cat([tr[r], l[:, None]], 1))
        tets = torch.cat(tets)
        X = self.B0[tets]
        D0 = (X[:, 1:] - X[:, :1]).transpose(1, 2)                    # [C,3,3] 열 = 변
        vol = torch.linalg.det(D0) / 6
        lr = ((X[:, :, None] - X[:, None]).norm(dim=-1) ** 2).sum((1, 2)).div(12).sqrt()
        q = 6 * math.sqrt(2) * vol.abs() / lr ** 3                     # 정사면체 1
        keep = q > qmin
        self.tets, self.D0i, self.V0 = tets[keep], torch.linalg.inv(D0[keep]), vol[keep].abs()
        print(f"[phystwin 셀] 변 {iu.shape[0]}  삼각형 {tris.shape[0]}  4-클릭 {tets.shape[0]}  "
              f"품질>{qmin} {int(keep.sum())}", flush=True)

    def logbarrier_residual(self, m):
        """셀 det 로그 장벽: √V₀ · log J,  J = det(D(x) D0⁻¹) (정지 1)."""
        X = (self.B + m)[self.tets]
        D = (X[:, 1:] - X[:, :1]).transpose(1, 2)
        J = det3(mm3(D, self.D0i))
        return self.V0.sqrt() * torch.log(J.clamp_min(1e-6))

    def elastic_residual(self, m):
        """스프링 에너지 E = ½ Σ (|x_i - x_j| - L0_ij)² 의 잔차 (처음 이웃 16 개, 단위 강성)."""
        Bn = self.B + m
        return ((Bn[self.rel] - Bn[:, None]).norm(dim=-1) - self.L0).reshape(-1)

    def bone_R(self):
        B, m, rel = self.B, self.m, self.rel
        A0 = B[rel] - B[:, None]
        A1 = (B[rel] + m[rel]) - (B[:, None] + m[:, None])
        U, S, Vh = torch.linalg.svd(A1.transpose(1, 2) @ A0)
        dfix = torch.ones_like(S)
        dfix[:, -1] = torch.sign(torch.linalg.det(U @ Vh))
        return U @ torch.diag_embed(dfix) @ Vh

    def yJ(self, X, sel):
        k = self.wi[sel]
        w, dw = self.w[sel], self.dw[sel]
        Rk = self.bone_R()[k]                                         # [n,K,3,3]
        dsel = self.d[sel]
        Rd = (Rk @ dsel[..., None]).squeeze(-1)
        moved = Rd + self.B[k] + self.m[k]
        dy = (w[..., None] * (Rd - dsel + self.m[k])).sum(1)
        J = (w[..., None, None] * Rk).sum(1) + outer_sum(moved, dw)
        return dy, J

    def forward(self, X, sel):
        k = self.wi[sel]
        d = (X[:, None] - self.B[k]).norm(dim=-1)
        w = 1.0 / (d + 1e-6)
        w = w / w.sum(1, keepdim=True)                              # knn_weights (X 의 함수)
        R = self.bone_R()
        moved = (R[k] @ (X[:, None] - self.B[k])[..., None]).squeeze(-1) + self.B[k] + self.m[k]
        return (w[..., None] * moved).sum(1)

    def cov_R(self, X, sel):
        """공식: 입자 회전 = 뼈 회전 쿼터니언의 가중 평균."""
        k = self.wi[sel]
        d = (X[:, None] - self.B[k]).norm(dim=-1)
        w = 1.0 / (d + 1e-6); w = w / w.sum(1, keepdim=True)
        R = self.bone_R()
        q = rot_to_quat(R)[k]                                        # [n,K,4]
        q = q * torch.sign((q * q[:, :1]).sum(-1, keepdim=True)).clamp_min(-1)
        return quat_mat((w[..., None] * q).sum(1))


def rot_to_quat(R):
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    w = torch.sqrt((1 + tr).clamp_min(1e-12)) / 2
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)
    return torch.stack([w, x, y, z], -1)


class GausSim(torch.nn.Module):
    rebind_each_frame = False

    def __init__(self):
        super().__init__()
        k1 = max(int(round(0.01 * NG)), 2)                          # downsample 0.01
        k2 = max(int(round(0.01 * k1)), 1)                          # downsample 0.01
        self.C1 = G0[fps(G0, k1)]
        self.C2 = self.C1[fps(self.C1, k2, 1)]
        self.par = torch.cdist(self.C1, self.C2).argmin(1)          # 군집 -> 위 군집
        self.lab = torch.cdist(G0, self.C1).argmin(1)               # 점 -> 군집
        self.p2 = torch.nn.Parameter(torch.zeros(k2, 3, device=dev))   # 맨 위 위치 이동
        z = torch.zeros(k1, 11, device=dev)
        z[:, 0] = 1.0; z[:, 7] = 1.0                                # init_quant 1 (단위 회전)
        self.dg = torch.nn.Parameter(z)
        self.dof = 3 * k2 + 11 * k1
        print(f"[gaussim] 군집 {k1} / 위 군집 {k2}", flush=True)

    def params(self):
        return [self.p2, self.dg]

    def F_of(self, p2, dg):
        """군집 F (매개변수를 인자로; p2 는 쓰지 않는다)."""
        U = quat_mat(dg[:, 0:4])
        s = torch.exp(dg[:, 4:7].clamp(-5, 5))
        s = s / torch.prod(s, -1, keepdim=True).pow(1 / 3)
        V = quat_mat(dg[:, 7:11])
        return U @ torch.diag_embed(s) @ V.transpose(1, 2)

    def F1(self):
        U = quat_mat(self.dg[:, 0:4])
        s = torch.exp(self.dg[:, 4:7].clamp(-5, 5))
        s = s / torch.prod(s, -1, keepdim=True).pow(1 / 3)          # 부피 보존 (norm_volumn)
        V = quat_mat(self.dg[:, 7:11])
        return U @ torch.diag_embed(s) @ V.transpose(1, 2)

    def yJ(self, X, sel):
        k = self.lab[sel]
        F = self.F1()[k]
        P = self.C2[self.par[k]]
        FmI = F - torch.eye(3, device=F.device, dtype=F.dtype)
        return self.p2[self.par[k]] + (FmI @ (X - P)[..., None]).squeeze(-1), F

    def forward(self, X, sel):
        k = self.lab[sel]
        F = self.F1()[k]
        P = self.C2[self.par[k]]
        return P + self.p2[self.par[k]] + (F @ (X - P)[..., None]).squeeze(-1)


class Simplicits(torch.nn.Module):
    rebind_each_frame = False

    def __init__(self, path):
        super().__init__()
        self.fcn = torch.load(path, weights_only=False).to(dev)
        for q in self.fcn.parameters():
            q.requires_grad_(False)
        with torch.no_grad():
            K = self.fcn(G0[:2].float()).shape[1]
        self.Tm = torch.nn.Parameter(torch.zeros(K, 3, 4, device=dev))
        self.dof = 12 * K
        # 결합은 정지 위치 G0 에 고정 -> 가중치와 그 기울기는 한 번만 구한다
        Ws, dWs = [], []
        for i in range(0, NG, 20000):
            Xc = G0[i:i + 20000].detach().requires_grad_(True)
            Wc = self.fcn(Xc.float()).to(Xc.dtype)
            g = [torch.autograd.grad(Wc[:, j].sum(), Xc, retain_graph=True)[0] for j in range(K)]
            Ws.append(Wc.detach()); dWs.append(torch.stack(g, 1).detach())
        self.W, self.dW = torch.cat(Ws), torch.cat(dWs)                     # [N,K], [N,K,3]

    def params(self):
        return [self.Tm]

    def yJ(self, X, sel):
        W, dW = self.W[sel], self.dW[sel]
        Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)
        TX = torch.einsum("kij,nj->nki", self.Tm, Xh)                 # [n,K,3]
        dy = (W[..., None] * TX).sum(1)
        J = (W[..., None, None] * self.Tm[None, :, :, :3]).sum(1) + outer_sum(TX, dW)
        return dy, eye_plus(J)

    def forward(self, X, sel):
        W = self.fcn(X.float()).to(X.dtype)                # kaolin 신경망은 float32
        Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)
        return X + (W[..., None] * torch.einsum("kij,nj->nki", self.Tm, Xh)).sum(1)


class VRGS(torch.nn.Module):
    rebind_each_frame = False

    def __init__(self):
        super().__init__()
        g_sim = (G0 - torch.as_tensor(OFF, device=dev, dtype=torch.float32)) * S_NORM \
            + torch.as_tensor(LO, device=dev, dtype=torch.float32)
        v, f = gv.mesh_from_points(g_sim, n_grid=100, grid_lim=2.0)   # 가우시안 중심으로
        v = v.to(DT)
        v = (v - torch.as_tensor(LO, device=dev, dtype=torch.float32)) / S_NORM \
            + torch.as_tensor(OFF, device=dev, dtype=torch.float32)
        used = torch.unique(f)
        remap = torch.full((v.shape[0],), -1, dtype=torch.long, device=dev)
        remap[used] = torch.arange(used.numel(), device=dev)
        self.vr, self.f = v[used], remap[f]
        self.ti, self.a1, self.a2, self.b = gv.bind_points(G0, self.vr, self.f)
        self.v = torch.nn.Parameter(self.vr.clone())
        self.A0i = torch.linalg.inv(torch.stack(gv._tri_frame(self.vr, self.f)[1:], -1))
        self.dof = 3 * self.vr.shape[0]
        # StVK 막 에너지용 정지 삼각형: 국소 2D 틀에서 D0 = [e1 e2] 와 넓이
        v0, v1, v2 = self.vr[self.f[:, 0]], self.vr[self.f[:, 1]], self.vr[self.f[:, 2]]
        e1, e2 = v1 - v0, v2 - v0
        t1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        nrm = torch.cross(e1, e2, dim=-1)
        self.A0 = 0.5 * nrm.norm(dim=-1)
        self.n0 = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        t2 = torch.cross(nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12), t1, dim=-1)
        D0 = torch.stack([torch.stack([(e1 * t1).sum(-1), (e1 * t2).sum(-1)], -1),
                          torch.stack([(e2 * t1).sum(-1), (e2 * t2).sum(-1)], -1)], -1)   # [T,2,2]
        self.D0i = torch.linalg.inv(D0)
        nu = 0.3                                              # 라메 상수 비 (μ=1)
        self.mu, self.lam = 1.0, 2 * nu / (1 - 2 * nu)
        print(f"[vrgs] 꼭짓점 {self.vr.shape[0]} 삼각형 {self.f.shape[0]}", flush=True)

    def params(self):
        return [self.v]

    def forward(self, X, sel):
        v0, e1, e2, n = gv._tri_frame(self.v, self.f)
        t = self.ti[sel]
        return (v0[t] + self.a1[sel, None] * e1[t] + self.a2[sel, None] * e2[t]
                + self.b[sel, None] * n[t])

    def F(self, sel):
        A = torch.stack(gv._tri_frame(self.v, self.f)[1:], -1)
        return (A @ self.A0i)[self.ti[sel]]

    def yJ(self, X, sel):
        return self(X, sel) - X, self.F(sel)

    def cellJ(self, v):
        """삼각형 셀 부호 넓이비와 가중치 w² = A0."""
        v0, v1, v2 = v[self.f[:, 0]], v[self.f[:, 1]], v[self.f[:, 2]]
        c = torch.cross(v1 - v0, v2 - v0, dim=-1)
        return (c * self.n0).sum(-1) / (2 * self.A0), self.A0

    def logbarrier_residual(self, v):
        """삼각형 셀 det 로그 장벽: √A₀ · log J,  J = (e1×e2)·n̂₀ / |e1⁰×e2⁰| (정지 법선 기준 부호 있는 넓이비)."""
        v0, v1, v2 = v[self.f[:, 0]], v[self.f[:, 1]], v[self.f[:, 2]]
        c = torch.cross(v1 - v0, v2 - v0, dim=-1)
        J = (c * self.n0).sum(-1) / (2 * self.A0)
        return self.A0.sqrt() * torch.log(J.clamp_min(1e-6))

    def elastic_residual(self, v):
        """삼각형 StVK 막 에너지 Σ A0 (μ|E|² + λ/2 tr(E)²) 의 잔차 (E = ½(FᵀF - I), F 는 3x2)."""
        v0, v1, v2 = v[self.f[:, 0]], v[self.f[:, 1]], v[self.f[:, 2]]
        e1, e2, Di = v1 - v0, v2 - v0, self.D0i
        # F = [e1 e2] D0⁻¹ (3x2) 을 원소별로 (작은 배치 행렬곱이 sgemm 으로 가서 느렸다)
        f1 = e1 * Di[:, 0, 0, None] + e2 * Di[:, 1, 0, None]
        f2 = e1 * Di[:, 0, 1, None] + e2 * Di[:, 1, 1, None]
        E11, E22 = 0.5 * ((f1 * f1).sum(-1) - 1), 0.5 * ((f2 * f2).sum(-1) - 1)
        E12 = 0.5 * (f1 * f2).sum(-1)
        w = self.A0.sqrt()
        return torch.cat([w * math.sqrt(self.mu) * E11, w * math.sqrt(self.mu) * E22,
                          w * math.sqrt(2 * self.mu) * E12,
                          w * math.sqrt(self.lam / 2) * (E11 + E22)])


if a.method in ("ours", "tet"):
    rep = Lattice(gregory=(a.method == "ours"))
elif a.method == "phystwin":
    rep = PhysTwin()
elif a.method == "gaussim":
    rep = GausSim()
elif a.method == "simplicits":
    rep = Simplicits(a.simp)
else:
    rep = VRGS()
REB = rep.rebind_each_frame
# 해석적 야코비안 (+ torch.compile). 크기가 프레임마다 바뀌는 격자 방법이 있어 dynamic
YJ = rep.yJ if a.no_compile else torch.compile(rep.yJ, dynamic=True)


MAP_DY = [None]


def map_and_F(Pref, sel, Fprev=None, need_graph=True):
    """선택 점의 위치와 누적 F (정지 대비)."""
    X = Pref[sel]
    if not a.autograd_jac:
        # yJ 는 변위 dy = y - X 를 직접 낸다: fp32 에서 y(≈0.5) 를 만든 뒤 목표를 빼면
        # 작은 변화가 반올림에 묻혀 직선 탐색이 보폭 0 으로 멈춘다 (같은 식, 계산 순서만)
        if need_graph:
            dy, J = YJ(X, sel)
        else:
            with torch.no_grad():
                dy, J = YJ(X, sel)
        F = mm3(J, Fprev[sel]) if (REB and Fprev is not None) else J
        MAP_DY[0] = dy
        return X + dy, F
    if a.method == "vrgs":
        y = rep(X, sel)
        F = rep.F(sel)
        return y, F
    if need_graph:
        y, J = jac(lambda Z: rep(Z, sel), X)
    else:
        with torch.enable_grad():
            y, J = jac(lambda Z: rep(Z, sel), X)
        y, J = y.detach(), J.detach()
    F = J @ Fprev[sel] if (REB and Fprev is not None) else J
    return y, F


# ============================================================== 측정·렌더 준비
def chamfer(A, B, ch=4096):
    def one(P, Q):
        return sum(torch.cdist(P[i:i + ch], Q).min(1).values.sum()
                   for i in range(0, P.shape[0], ch)) / P.shape[0]
    return float(0.5 * (one(A, B) + one(B, A)))


RENDER = None
if a.video:
    sys.path.append(a.pg)
    sys.path.append(os.path.join(a.pg, "gaussian-splatting"))
    _cwd = os.getcwd(); os.chdir(a.pg)
    from scene.gaussian_model import GaussianModel                   # noqa: E402
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402
    from diff_gaussian_rasterization import (                         # noqa: E402
        GaussianRasterizationSettings, GaussianRasterizer)
    import imageio.v2 as imageio                                     # noqa: E402
    os.chdir(_cwd)
    model = str(AX["model"])
    torch.set_default_dtype(torch.float32)          # 3DGS 모델은 float32 로 적재
    gs = GaussianModel(3)
    gs.load_ply(f"{model}/point_cloud/iteration_30000/point_cloud.ply")
    gi = torch.as_tensor(AX["gidx"][GIDX_SUB], device=dev)   # 같은 부분표본 가우시안만
    c6 = gs.get_covariance()[gi].detach()
    C0 = torch.zeros(NG, 3, 3, device=dev)
    C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c6[:, 0], c6[:, 1], c6[:, 2]
    C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c6[:, 3], c6[:, 4], c6[:, 5]
    C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c6[:, 1], c6[:, 2], c6[:, 4]
    SHS, OPA = gs.get_features[gi].detach(), gs.get_opacity[gi].detach()
    torch.set_default_dtype(DT)
    so, mean = float(AX["scale_origin"]), torch.as_tensor(AX["mean"], device=dev).float()
    lo_t = torch.as_tensor(LO, device=dev).float(); off_t = torch.as_tensor(OFF, device=dev).float()
    cam = json.load(open(f"{model}/cameras.json"))[0]
    Rw, pos = np.array(cam["rotation"]), np.array(cam["position"])
    W2C = np.linalg.inv(np.block([[Rw, pos[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
    fx_ = 2 * math.atan(cam["width"] / (2 * cam["fx"]))
    fy_ = 2 * math.atan(cam["height"] / (2 * cam["fy"]))
    wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
    pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fx_, fovY=fy_).transpose(0, 1).to(dev).float()
    st = GaussianRasterizationSettings(
        image_height=int(cam["height"]), image_width=int(cam["width"]),
        tanfovx=math.tan(fx_ * 0.5), tanfovy=math.tan(fy_ * 0.5), bg=torch.ones(3, device=dev, dtype=torch.float32),
        scale_modifier=1.0, viewmatrix=wv, projmatrix=(wv[None] @ pj[None])[0],
        sh_degree=3, campos=wv.inverse()[3, :3], prefiltered=False, debug=False)
    RAST = GaussianRasterizer(raster_settings=st)
    WR = imageio.get_writer(a.video, fps=30, codec="libx264", quality=8)
    RENDER = True


def render(P, Fall=None, Rall=None):
    P = P.float()
    Pm = ((P - off_t) * S_NORM + lo_t - 1.0) / so + mean
    M = (Rall if Rall is not None else Fall).float()
    cov = M @ C0 @ M.transpose(1, 2)
    c6 = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2],
                      cov[:, 2, 2]], 1)
    torch.set_default_dtype(torch.float32)          # 래스터라이저가 빈 텐서를 기본형으로 만든다
    with torch.no_grad():
        img = RAST(means3D=Pm, means2D=torch.zeros_like(Pm), shs=SHS, colors_precomp=None,
                   opacities=OPA, scales=None, rotations=None, cov3D_precomp=c6)[0]
    WR.append_data((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    torch.set_default_dtype(DT)


def eval_all(Pref, Fprev):
    """전체 가우시안의 위치·누적 F (그리고 phystwin 은 공식 회전)."""
    ys, Fs, Rs = [], [], []
    for i in range(0, NG, 40000):
        sel = ALL[i:i + 40000]
        y, F = map_and_F(Pref, sel, Fprev, need_graph=False)
        ys.append(y.detach()); Fs.append(F.detach())
        if a.method == "phystwin":
            with torch.no_grad():
                Rs.append(rep.cov_R(Pref[sel], sel))
    return torch.cat(ys), torch.cat(Fs), (torch.cat(Rs) if Rs else None)


# ============================================================== 추적
ALL = torch.arange(NG, device=dev)
TBW = None
if a.tb != "none":
    from torch.utils.tensorboard import SummaryWriter
    _tb = a.tb if a.tb != "auto" else os.path.join(
        "/home/dkta/work/tbrf", os.path.basename(os.path.dirname(os.path.abspath(a.out)))
        + "_" + os.path.splitext(os.path.basename(a.out))[0])
    TBW = SummaryWriter(_tb)
    print(f"[TB] {_tb}", flush=True)
NCUR = [0]
OPT = None
if a.opt == "adam_bt":
    # 예전 설정(rep_track.py)의 최적화: det F 는 정지 이웃 k 개 최소제곱으로 재고,
    # 격자 재설정 방법은 직전 위치 기준(증분), 나머지는 정지 위치 기준
    NBR = torch.cat([torch.cdist(G0[i:i + 8192], G0).topk(a.knn_F + 1, largest=False).indices[:, 1:]
                     for i in range(0, NG, 8192)])

    def knn_ref(P):
        dX = P[NBR] - P[:, None]
        return dX, torch.linalg.inv(dX.transpose(1, 2) @ dX + 1e-12 * torch.eye(3, device=dev))

    def knn_det(Y, ref):
        dX, Binv = ref
        dY = Y[NBR] - Y[:, None]
        return det3((dY.transpose(1, 2) @ dX) @ Binv)
    REF0 = knn_ref(G0)
    OPT = None
Pref = G0.clone()
Fcum = torch.eye(3, device=dev).expand(NG, 3, 3).clone()
Rcum = torch.eye(3, device=dev).expand(NG, 3, 3).clone()
if REB:
    rep.rebind(Pref)
rep.to(dev)
print(f"[{a.method}] 가우시안 {NG} (맞추기 {N})  자유도 {rep.dof}  지름 L {L:.4f}", flush=True)
if a.check_jac:
    gch = torch.Generator(device="cpu").manual_seed(1)
    with torch.no_grad():
        for q in rep.params():
            q.add_(0.02 * torch.randn(q.shape, generator=gch).to(q))
    sel = torch.randperm(NG, generator=gch)[:5000].to(dev)
    X = Pref[sel]
    if a.method == "vrgs":                          # vrgs 는 원래 해석적 F (자동미분 비교 없음)
        y0, J0 = rep(X, sel), rep.F(sel)
    else:
        y0, J0 = jac(lambda Z: rep(Z, sel), X)
    y1, J1 = rep.yJ(X, sel); y1 = X + y1
    y2, J2 = YJ(X, sel)
    l0 = (y0.square().sum() + det3(J0).sum()); g0 = torch.autograd.grad(l0, rep.params())
    l1 = (y1.square().sum() + det3(J1).sum()); g1 = torch.autograd.grad(l1, rep.params())
    rel = lambda A, B: float((A - B).abs().max() / B.abs().max().clamp_min(1e-30))
    print(f"[검사] y {rel(y1, y0):.2e}  J {rel(J1, J0):.2e}  컴파일 J {rel(J2, J0):.2e}  "
          f"기울기 " + " ".join(f"{rel(p, q):.2e}" for p, q in zip(g1, g0)), flush=True)
    import time as _t
    for name, fn in (("자동미분", None), ("해석", rep.yJ), ("해석+컴파일", YJ)):
        for rep_i in range(4):
            torch.cuda.synchronize(); t1 = _t.time()
            if fn is None:
                yy, JJ = ((rep(Pref, ALL), rep.F(ALL)) if a.method == "vrgs"
                          else jac(lambda Z: rep(Z, ALL), Pref))
            else:
                yy, JJ = fn(Pref, ALL); yy = Pref + yy
            (yy.square().mean() + det3(JJ).mean()).backward()
            torch.cuda.synchronize()
        print(f"[시간] {name}: 전체 {NG} 점 한 번 평가+역전파 {1000*(_t.time()-t1):.1f} ms", flush=True)
    sys.exit(0)
if RENDER:
    render(G0, Fall=Fcum)
rows, CURVE, DOFS, STOP, EMDP = [], [], [], [], []
PHYS, TRAJS, YPREV = [], [], [G0[FI].clone()]
t0 = time.time()
for t in range(1, T + 1):
    tgt = TRAJ[t]
    RES0 = Pref[FI] - tgt                               # 직전 위치 - 목표 (가까운 두 수의 차)
    if REB and t > 1:
        rep.rebind(Pref); rep.to(dev)
    DOFS.append(rep.dof)
    it = [0]

    def closure():
        opt.zero_grad(set_to_none=True)
        y, F = map_and_F(Pref, FI, Fcum)
        J = det3(F)
        res = (RES0 + MAP_DY[0]) if not a.autograd_jac else (y - tgt)
        l2 = (res ** 2).sum(1).mean()
        bar = barrier(J).mean()
        loss = l2 + a.kappa * bar
        loss.backward()
        CURVE.append((t, it[0], float(l2), float(bar)))
        it[0] += 1
        return loss
    if a.opt in ("riem", "ipm"):
        # 탄성 에너지의 가우스-뉴턴 헤시안을 계량으로: Δ = (JᵀJ + ε·λmax·I)⁻¹ ∇L, θ -= ηΔ.
        # 계량은 매 스텝 현재 상태에서 다시 선형화한다. 장벽·되돌림·선탐색 없음
        from torch.func import jvp, vjp
        PL = rep.params()
        shapes = [q.shape for q in PL]
        sizes = [q.numel() for q in PL]

        def unflat(x):
            return [c.reshape(sh) for c, sh in zip(torch.split(x, sizes), shapes)]

        def flat(ts):
            return torch.cat([q.reshape(-1) for q in ts])
        theta = flat([q.detach() for q in PL])
        X_ = Pref[FI]
        if a.method in ("ours", "tet") and a.riem_energy == "logbarrier":
            def make_res(th):
                return lambda x: rep.logbarrier_residual(*unflat(x))
        elif a.method in ("ours", "tet"):
            # 입자 변형 구배로 잰 탄성 에너지 (고정 공회전, PhysGaussian 기본 탄성):
            # ψ(F) = μ|F - R|² + λ/2 (J - 1)²,  F = J_inc(θ) · F_prev (정지 대비 누적).
            # 가우스-뉴턴에서 R 은 선형화 점의 극분해로 고정한다
            nu_ = 0.3
            mu_, la_ = 1.0, 2 * nu_ / (1 - 2 * nu_)
            Fp_ = Fcum[FI]
            wv_ = 1.0 / math.sqrt(FI.numel())

            def make_res(th):
                with torch.no_grad():
                    _, J0 = rep.yJ_p(X_, FI, *(unflat(th) if rep.greg else unflat(th) + [None]))
                    U_, _, Vh_ = torch.linalg.svd(J0 @ Fp_)
                    Dg = torch.ones(U_.shape[0], 3, device=dev, dtype=U_.dtype)
                    Dg[:, 2] = torch.sign(torch.linalg.det(U_ @ Vh_))
                    R_ = U_ @ torch.diag_embed(Dg) @ Vh_

                def res(x):
                    _, Jx = rep.yJ_p(X_, FI, *(unflat(x) if rep.greg else unflat(x) + [None]))
                    F_ = mm3(Jx, Fp_)
                    return torch.cat([(wv_ * math.sqrt(mu_) * (F_ - R_)).reshape(-1),
                                      wv_ * math.sqrt(la_ / 2) * (det3(F_) - 1.0)])
                return res
        else:
            def make_res(th):
                if a.riem_energy == "logbarrier":
                    if a.method == "gaussim":
                        # 입자 det 로그 장벽: log det F_p (13.9 만 입자, 가중 1/√N)
                        wv = 1.0 / math.sqrt(FI.numel())
                        return lambda x: wv * torch.log(
                            det3(GausSim.F_of(rep, *unflat(x))[rep.lab[FI]]).clamp_min(1e-6))
                    if a.method == "simplicits":
                        # 입자 det 로그 장벽: log det F_p, F_p = I + Σ W T[:, :3] + Σ (T[X;1]) ⊗ ∇W
                        wv = 1.0 / math.sqrt(FI.numel())
                        W_, dW_ = rep.W[FI], rep.dW[FI]
                        Xh_ = torch.cat([X_, torch.ones_like(X_[:, :1])], 1)

                        def res_s(x):
                            Tm = unflat(x)[0]
                            TX = torch.einsum("kij,nj->nki", Tm, Xh_)
                            F_ = eye_plus((W_[..., None, None] * Tm[None, :, :, :3]).sum(1)
                                          + outer_sum(TX, dW_))
                            return wv * torch.log(det3(F_).clamp_min(1e-6))
                        return res_s
                    return lambda x: rep.logbarrier_residual(*unflat(x))     # vrgs: 삼각형 셀 det
                return lambda x: rep.elastic_residual(*unflat(x))

        # 셀/입자 det 함수 (ipm 의 장벽·가능성 판정용)
        if a.method == "simplicits":
            W_i, dW_i = rep.W[FI], rep.dW[FI]
            Xh_i = torch.cat([X_, torch.ones_like(X_[:, :1])], 1)

            def Jfun(x):
                Tm = unflat(x)[0]
                TX = torch.einsum("kij,nj->nki", Tm, Xh_i)
                F_ = eye_plus((W_i[..., None, None] * Tm[None, :, :, :3]).sum(1) + outer_sum(TX, dW_i))
                J_ = det3(F_)
                return J_, torch.full_like(J_, 1.0 / J_.numel())
        elif hasattr(rep, "cellJ"):
            def Jfun(x):
                return rep.cellJ(*unflat(x))
        else:
            Jfun = None

        PURE = None                                         # (순수 잔차, 인자) -- 컴파일 경로
        if a.riem_energy == "logbarrier" and a.method in ("ours", "tet"):
            PURE = (res_lattice_lb, (rep.Xn, rep.cells, rep.cD0i))
        elif a.riem_energy == "logbarrier" and a.method == "vrgs":
            PURE = (res_tri_lb, (rep.f, rep.n0, rep.A0))
        elif a.riem_energy == "logbarrier" and a.method == "simplicits":
            PURE = (res_simp_lb, (rep.W[FI], rep.dW[FI], torch.cat([X_, torch.ones_like(X_[:, :1])], 1)))
        elif a.riem_energy == "elastic" and a.method == "phystwin":
            PURE = (res_spring, (rep.B, rep.rel, rep.L0))
        if PURE is not None and GN_PROD is None and not a.no_compile:
            GN_PROD = torch.compile(_gn_prod, dynamic=True)

        def GN(th):
            if PURE is not None:
                fn_, args_ = PURE
                pr = GN_PROD if GN_PROD is not None else _gn_prod
                return lambda u: pr(fn_, th, u, *args_)
            rf = make_res(th)
            _, vjp_fn = vjp(rf, th)
            return lambda u: vjp_fn(jvp(rf, (th,), (u,))[1])[0]
        with torch.no_grad():                                 # 프레임마다 λmax (거듭제곱법 20 번)
            Hv = GN(theta.detach())
            u = torch.randn_like(theta)
            for _ in range(20):
                u = Hv(u); lmax = float(u.norm()); u = u / max(lmax, 1e-30)
        if a.profile:
            import collections
            TM = collections.defaultdict(float)
            NCG = [0]

            def tic():
                torch.cuda.synchronize(); return time.time()
            WARM = 3                                          # 컴파일·첫 호출은 빼고 잰다
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
        eps = a.riem_eps * max(lmax, 1e-12)                   # k=1 기준 -> k 를 키우면 계량이 더 딱딱해진다
        if a.opt == "ipm":
            # 내부점법 (장벽법): 매 단계 min L2 + μ·B, B = -Σ w² log J (J>0 영역 안에서만),
            # μ 는 단계마다 1/10. 방향은 같은 계량의 Δ = -G⁻¹∇f, 보폭은 가능성(J>0) + Armijo 되돌림
            assert Jfun is not None, "ipm 은 로그 장벽(셀/입자 det)이 있는 방법만"
            ntot = a.iters0 if t == 1 else a.iters
            nst = max(ntot // a.ipm_stages, 1)
            step = 1.0

            def fval(x):
                with torch.no_grad():
                    ps = unflat(x)
                    saved = [q.detach().clone() for q in PL]
                    for q, pp in zip(PL, ps):
                        q.copy_(pp)
                    dy_, _ = YJ(X_, FI)
                    l2_ = float(((RES0 + dy_) ** 2).sum(1).mean())
                    for q, pp in zip(PL, saved):
                        q.copy_(pp)
                    J_, w2_ = Jfun(x)
                    if float(J_.min()) <= 0:
                        return float("inf"), l2_
                    return l2_ + mu * float(-(w2_ * torch.log(J_)).sum()), l2_
            for it_ in range(ntot):
                mu = a.ipm_mu0 * (0.1 ** min(it_ // nst, a.ipm_stages - 1))
                for q in PL:
                    q.grad = None
                dy, _ = YJ(X_, FI)
                l2 = ((RES0 + dy) ** 2).sum(1).mean()
                th_ = flat(list(PL))
                Jc, w2c = Jfun(th_)
                f = l2 + mu * (-(w2c * torch.log(Jc.clamp_min(1e-30))).sum())
                f.backward()
                gk = flat([q.grad if q.grad is not None else torch.zeros_like(q) for q in PL]).detach()
                theta = flat([q.detach() for q in PL])
                with torch.no_grad():
                    Hv = GN(theta)
                    Gv = lambda u: Hv(u) + eps * u
                    x = torch.zeros_like(gk); r = gk.clone(); pdir = r.clone(); rr = (r * r).sum()
                    for _ in range(a.riem_cg):
                        Gp = Gv(pdir); al = rr / (pdir * Gp).sum().clamp_min(1e-30)
                        x += al * pdir; r -= al * Gp
                        rr_new = (r * r).sum()
                        if rr_new.sqrt() < 1e-4 * gk.norm():
                            break
                        pdir = r + (rr_new / rr) * pdir; rr = rr_new
                    f0 = float(f); slope = -float((gk * x).sum())
                    step = min(step * 2.0, 1e6)
                    ok = False
                    for _bt in range(40):
                        f1, _ = fval(theta - step * x)
                        if f1 <= f0 + 1e-4 * step * slope:
                            ok = True; break
                        step *= 0.5
                    if ok:
                        for q, dq in zip(PL, unflat(theta - step * x)):
                            q.copy_(dq)
                CURVE.append((t, it_, float(l2), mu, step if ok else 0.0))
            it[0] = ntot
            STOP.append((t, ntot, ntot, float(Jfun(flat([q.detach() for q in PL]))[0].min()), eps, "ipm"))
        for it_ in range(0 if a.opt == "ipm" else (a.iters0 if t == 1 else a.iters)):
            if a.profile and it_ == WARM:
                TM.clear(); NCG[0] = 0
                prof.__enter__()
            if a.profile and it_ == WARM + a.profile:
                prof.__exit__(None, None, None)
                tot = sum(TM.values()); nit = a.profile
                print(f"[프로파일] {a.method}  반복 {nit} (앞 {WARM} 번 제외)  CG 평균 {NCG[0] / nit:.1f} 회/반복  "
                      f"합계 {1000 * tot / nit:.1f} ms/반복", flush=True)
                for k_, v_ in sorted(TM.items(), key=lambda z: -z[1]):
                    print(f"  {k_:28s} {1000 * v_ / nit:9.2f} ms/반복  ({100 * v_ / tot:5.1f}%)", flush=True)
                print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12), flush=True)
                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12), flush=True)
                sys.exit(0)
            if a.profile:
                t_ = tic()
            for q in PL:
                q.grad = None
            dy, _ = YJ(X_, FI)
            l2 = ((RES0 + dy) ** 2).sum(1).mean()
            l2.backward()
            gk = flat([q.grad if q.grad is not None else torch.zeros_like(q) for q in PL]).detach()
            theta = flat([q.detach() for q in PL])
            if a.profile:
                t2_ = tic(); TM["손실 앞·뒤 (L2 기울기)"] += t2_ - t_; t_ = t2_
            with torch.no_grad():
                Hv = GN(theta)
                if a.profile:
                    t2_ = tic(); TM["계량 선형화 (make_res+vjp)"] += t2_ - t_; t_ = t2_
                Gv = lambda u: a.riem_k * Hv(u) + eps * u
                x = torch.zeros_like(gk); r = gk.clone(); pdir = r.clone(); rr = (r * r).sum()
                for _ in range(a.riem_cg):                    # CG: G x = g
                    Gp = Gv(pdir); al = rr / (pdir * Gp).sum().clamp_min(1e-30)
                    x += al * pdir; r -= al * Gp
                    rr_new = (r * r).sum()
                    if a.profile:
                        NCG[0] += 1
                    if rr_new.sqrt() < 1e-4 * gk.norm():
                        break
                    pdir = r + (rr_new / rr) * pdir; rr = rr_new
                if a.profile:
                    t2_ = tic(); TM["CG (jvp+vjp 곱)"] += t2_ - t_; t_ = t2_
                for q, dq in zip(PL, unflat(x)):
                    q -= a.riem_lr * dq
            CURVE.append((t, it_, float(l2), 0.0, 0))
        if a.opt == "riem":
            it[0] = it_ + 1
            STOP.append((t, it_ + 1, it_ + 1, float("nan"), eps, "riem"))
    elif a.opt == "adam":
        # 뒤집힘 고려 없음: 입자 L2 만 Adam 으로 (장벽·되돌림 없음). det 는 야코비안으로 기록만
        if REB or OPT is None:
            OPT = torch.optim.Adam(rep.params(), lr=a.lr)
        X_ = Pref[FI]
        for it_ in range(a.iters0 if t == 1 else a.iters):
            OPT.zero_grad(set_to_none=True)
            dy, _ = YJ(X_, FI)
            l2 = ((RES0 + dy) ** 2).sum(1).mean()
            l2.backward()
            OPT.step()
            CURVE.append((t, it_, float(l2), 0.0, 0))
        it[0] = it_ + 1
        STOP.append((t, it_ + 1, it_ + 1, float("nan"), 0.0, "adam"))
    elif a.opt == "adam_bt":
        if REB or OPT is None:                       # 매개변수가 새로 생기면 Adam 도 새로
            OPT = torch.optim.Adam(rep.params(), lr=a.lr)
        ref = knn_ref(Pref) if REB else REF0
        X_ = Pref[FI]
        nbt_tot = 0
        for it_ in range(a.iters0 if t == 1 else a.iters):
            OPT.zero_grad(set_to_none=True)
            dy, _ = YJ(X_, FI)
            Y = X_ + dy
            l2 = ((RES0 + dy) ** 2).sum(1).mean()
            bar = torch.relu(a.tau - knn_det(Y, ref)).pow(2).mean()
            (l2 + a.lam_inv * bar).backward()
            prev = [q.detach().clone() for q in rep.params()]
            OPT.step()
            nbt = 0                                    # det>0 을 지키는 되돌림 (반씩, 최대 10 번)
            with torch.no_grad():
                for _bt in range(10):
                    dy2, _ = YJ(X_, FI)
                    if float(knn_det(X_ + dy2, ref).min()) > 0:
                        break
                    nbt += 1
                    for q, q0 in zip(rep.params(), prev):
                        q.copy_(q0 + 0.5 * (q - q0))
                else:
                    for q, q0 in zip(rep.params(), prev):
                        q.copy_(q0)
            nbt_tot += nbt
            CURVE.append((t, it_, float(l2), float(bar), nbt))
        it[0] = it_ + 1
        with torch.no_grad():
            dy2, _ = YJ(X_, FI)
            KD = knn_det(X_ + dy2, ref)
        STOP.append((t, it_ + 1, it_ + 1, float(KD.min()), float(nbt_tot), "adam_bt"))
    elif a.opt == "lbfgs":
        opt = torch.optim.LBFGS(rep.params(), lr=1.0, max_iter=a.max_iter, history_size=50,
                                tolerance_grad=0.0, tolerance_change=0.0,
                                line_search_fn="strong_wolfe")
        opt.step(closure)
        # 종료 사유: torch LBFGS 의 내부 상태로 판정한다
        stt = opt.state[opt._params[0]]
        gmax = float(torch.cat([q.grad.reshape(-1) for q in rep.params()]).abs().max())
        dstep = float((stt["d"] * stt["t"]).abs().max()) if "d" in stt else float("nan")
        reason = ("max_iter" if stt["n_iter"] >= a.max_iter else
                  "max_eval" if stt["func_evals"] >= int(a.max_iter * 1.25) else
                  "grad=0" if gmax <= 0 else
                  "step=0" if dstep <= 0 else
                  "loss 변화 0 / 하강방향 아님")
        STOP.append((t, stt["n_iter"], stt["func_evals"], gmax, dstep, reason))
    else:
        opt = torch.optim.SGD(rep.params(), lr=a.gd_lr)
        for _ in range(a.max_iter):
            opt.step(closure)
        STOP.append((t, a.max_iter, a.max_iter, float("nan"), float("nan"), "gd"))
    yall, Fall, Rall = eval_all(Pref, Fcum)
    with torch.no_grad():
        y = yall[FI]
        J = torch.linalg.det(Fall[FI])
        rmse = float(((y - tgt) ** 2).sum(1).mean().sqrt()) / L
        cd = chamfer(y, tgt) / L
        e = float("nan")                            # EMD 는 저장한 위치로 rep_emd.py 가 따로 잰다
        if t % a.emd_every == 0 or t == T:
            EMDP.append((t, y.float().cpu().numpy(), tgt.float().cpu().numpy()))
        rows.append((t, rmse, cd, e, float(J.min()), float((J <= 0).float().mean())))
        # 물리 지표 (같은 질량 1/N 입자): 운동에너지·선운동량·각운동량 (프레임 차분 속도), 부피비 Σ det F / N
        vel = (y - YPREV[0]) * 30.0
        com = y.mean(0)
        PHYS.append((t, float(0.5 * (vel * vel).sum(1).mean()), *vel.mean(0).tolist(),
                     *torch.cross(y - com, vel, dim=-1).mean(0).tolist(), float(J.mean())))
        YPREV[0] = y.clone()
        if a.save_traj:
            TRAJS.append(yall.half().cpu().numpy())
        if TBW is not None:
            for k_, v_ in (("RMSE_pct", 100 * rmse), ("CD_pct", 100 * cd),
                           ("detJ_min", rows[-1][4]), ("inverted_pct", 100 * rows[-1][5]),
                           ("dof", rep.dof), ("time_s", time.time() - t0)):
                TBW.add_scalar("frame/" + k_, v_, t)
            if a.opt == "adam_bt":
                TBW.add_scalar("frame/knn_det_min", STOP[-1][3], t)
                TBW.add_scalar("frame/backtracks", STOP[-1][4], t)
            else:
                TBW.add_scalar("frame/evals", STOP[-1][2], t)
                TBW.add_scalar("frame/early_stop", float(STOP[-1][5] == "step=0"), t)
            for c_ in CURVE[NCUR[0]:]:                 # 반복별 러닝 커브 (전체 반복 번호)
                TBW.add_scalar("iter/l2", c_[2], NCUR[0]); TBW.add_scalar("iter/barrier", c_[3], NCUR[0])
                NCUR[0] += 1
            TBW.flush()
        if REB:
            Fcum = Fall.clone()
            if Rall is not None:
                Rcum = Rall @ Rcum
            Pref = yall.clone()
        else:
            Fcum = Fall
    if RENDER:
        render(yall, Fall=Fall, Rall=(Rcum if a.method == "phystwin" else None))
    if t % 10 == 0 or t == 1:
        print(f"  [t={t:3d}] RMSE {100*rmse:.3f}%  CD {100*cd:.3f}%  "
              + (f"EMD {100*e:.3f}%  " if e == e else "")
              + f"det 최소 {rows[-1][4]:.3f} (≤0 {100*rows[-1][5]:.2f}%)  자유도 {rep.dof}  "
              + (f"이웃 det 최소 {STOP[-1][3]:.3f}  되돌림 {int(STOP[-1][4])}  "
                 if a.opt == "adam_bt" else
                 f"평가 {it[0]}  종료 {STOP[-1][5]} (반복 {STOP[-1][1]}, |g|max {STOP[-1][3]:.1e}, "
                 f"|step|max {STOP[-1][4]:.1e})  ")
              + f"{time.time()-t0:.0f}s", flush=True)
if RENDER:
    WR.close()
R = np.array(rows)
print(f"[요약] {a.method}  자유도 {np.mean(DOFS):.0f}  RMSE {100*R[:,1].mean():.3f}%  "
      f"CD {100*R[:,2].mean():.3f}%  det 최소 {R[:,4].min():.4f}  "
      f"뒤집힘 최대 {100*R[:,5].max():.2f}%", flush=True)
np.savez_compressed(a.out, metrics=R, dof=np.array(DOFS), L=L,
                    curve=np.array(CURVE, dtype=np.float64),
                    stop=np.array([q[:5] for q in STOP], dtype=np.float64),
                    stop_reason=np.array([q[5] for q in STOP]),
                    emd_t=np.array([q[0] for q in EMDP]), emd_y=np.stack([q[1] for q in EMDP]),
                    emd_tgt=np.stack([q[2] for q in EMDP]),
                    phys=np.array(PHYS, dtype=np.float64),     # t, KE, P(3), Lang(3), mean detF
                    **({"traj": np.stack(TRAJS)} if TRAJS else {}))
print(f"[저장] {a.out}" + (f"  영상 {a.video}" if a.video else ""), flush=True)
