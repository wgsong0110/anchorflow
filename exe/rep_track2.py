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
ap.add_argument("--emd_n", type=int, default=4096)
ap.add_argument("--emd_every", type=int, default=10)
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--fp64", action="store_true", help="모든 계산을 float64 로")
ap.add_argument("--opt", default="lbfgs", choices=["lbfgs", "gd"],
                help="lbfgs (원래 설정) / gd: 고정 학습률 경사하강")
ap.add_argument("--gd_lr", type=float, default=1.0)
ap.add_argument("--frames", type=int, default=0, help="앞 몇 프레임만 (0 이면 전부)")
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


def jac(fn, X):
    """y = fn(X) 와 J = ∂y/∂X (점별, create_graph)."""
    X = X.detach().requires_grad_(True)
    y = fn(X)
    rows = [torch.autograd.grad(y[:, i].sum(), X, create_graph=True)[0] for i in range(3)]
    return y, torch.stack(rows, 1)


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
        self.u = torch.nn.Parameter(torch.zeros(M, 3, device=dev))
        self.rho_raw = torch.nn.Parameter(torch.zeros(M, device=dev)) if self.greg else None
        self.dof = (4 if self.greg else 3) * M

    def params(self):
        return [self.u] + ([self.rho_raw] if self.greg else [])

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
        self.m = None
        self.dof = 3 * B.shape[0]

    def rebind(self, Pall):
        if self.m is not None:
            self.B = (self.B + self.m.detach()).clone()
        ii = torch.cat([torch.cdist(Pall[i:i + 20000], self.B).topk(self.K, largest=False).indices
                        for i in range(0, Pall.shape[0], 20000)])
        self.wi = ii
        self.m = torch.nn.Parameter(torch.zeros_like(self.B))

    def params(self):
        return [self.m]

    def bone_R(self):
        B, m, rel = self.B, self.m, self.rel
        A0 = B[rel] - B[:, None]
        A1 = (B[rel] + m[rel]) - (B[:, None] + m[:, None])
        U, S, Vh = torch.linalg.svd(A1.transpose(1, 2) @ A0)
        dfix = torch.ones_like(S)
        dfix[:, -1] = torch.sign(torch.linalg.det(U @ Vh))
        return U @ torch.diag_embed(dfix) @ Vh

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

    def F1(self):
        U = quat_mat(self.dg[:, 0:4])
        s = torch.exp(self.dg[:, 4:7].clamp(-5, 5))
        s = s / torch.prod(s, -1, keepdim=True).pow(1 / 3)          # 부피 보존 (norm_volumn)
        V = quat_mat(self.dg[:, 7:11])
        return U @ torch.diag_embed(s) @ V.transpose(1, 2)

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

    def params(self):
        return [self.Tm]

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


def map_and_F(Pref, sel, Fprev=None, need_graph=True):
    """선택 점의 위치와 누적 F (정지 대비)."""
    X = Pref[sel]
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


EI = torch.as_tensor(np.random.default_rng(0).choice(N, min(a.emd_n, N), replace=False),
                     device=dev)


def emd(A, B):
    from scipy.optimize import linear_sum_assignment
    C = torch.cdist(A[EI].double(), B[EI].double()).cpu().numpy()
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean())


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
    with torch.no_grad():
        img = RAST(means3D=Pm, means2D=torch.zeros_like(Pm), shs=SHS, colors_precomp=None,
                   opacities=OPA, scales=None, rotations=None, cov3D_precomp=c6)[0]
    WR.append_data((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


ALL = torch.arange(NG, device=dev)


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
Pref = G0.clone()
Fcum = torch.eye(3, device=dev).expand(NG, 3, 3).clone()
Rcum = torch.eye(3, device=dev).expand(NG, 3, 3).clone()
if REB:
    rep.rebind(Pref)
rep.to(dev)
print(f"[{a.method}] 가우시안 {NG} (맞추기 {N})  자유도 {rep.dof}  지름 L {L:.4f}", flush=True)
if RENDER:
    render(G0, Fall=Fcum)
rows, CURVE, DOFS, STOP = [], [], [], []
t0 = time.time()
for t in range(1, T + 1):
    tgt = TRAJ[t]
    if REB and t > 1:
        rep.rebind(Pref); rep.to(dev)
    DOFS.append(rep.dof)
    it = [0]

    def closure():
        opt.zero_grad(set_to_none=True)
        y, F = map_and_F(Pref, FI, Fcum)
        J = torch.linalg.det(F)
        l2 = ((y - tgt) ** 2).sum(1).mean()
        bar = barrier(J).mean()
        loss = l2 + a.kappa * bar
        loss.backward()
        CURVE.append((t, it[0], float(l2), float(bar)))
        it[0] += 1
        return loss
    if a.opt == "lbfgs":
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
        e = emd(y, tgt) / L if (t % a.emd_every == 0 or t == T) else float("nan")
        rows.append((t, rmse, cd, e, float(J.min()), float((J <= 0).float().mean())))
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
              f"평가 {it[0]}  종료 {STOP[-1][5]} (반복 {STOP[-1][1]}, |g|max {STOP[-1][3]:.1e}, "
              f"|step|max {STOP[-1][4]:.1e})  {time.time()-t0:.0f}s", flush=True)
if RENDER:
    WR.close()
R = np.array(rows)
ev = R[:, 3][~np.isnan(R[:, 3])]
print(f"[요약] {a.method}  자유도 {np.mean(DOFS):.0f}  RMSE {100*R[:,1].mean():.3f}%  "
      f"CD {100*R[:,2].mean():.3f}%  EMD {100*ev.mean():.3f}%  det 최소 {R[:,4].min():.4f}  "
      f"뒤집힘 최대 {100*R[:,5].max():.2f}%", flush=True)
np.savez_compressed(a.out, metrics=R, dof=np.array(DOFS), L=L,
                    curve=np.array(CURVE, dtype=np.float64),
                    stop=np.array([q[:5] for q in STOP], dtype=np.float64),
                    stop_reason=np.array([q[5] for q in STOP]))
print(f"[저장] {a.out}" + (f"  영상 {a.video}" if a.video else ""), flush=True)
