"""표현력 비교의 변형 표현들 (repflow B: 증분 포텐셜 추적용, 점 집합을 인자로 받는 판).

exe/rep_track2.py 의 클래스와 같은 식이다 (공식 설정 · 해석적 야코비안). 다른 점은
전역 변수 대신 점 집합 X0 (시뮬 좌표, 내부 채움 포함)를 받고, 계량 텐서장용 순수 잔차를
(함수, 인자) 로 내준다는 것뿐이다. 여러 물체는 Multi 로 물체마다 따로 표현을 둔다.

각 표현:
  rebind_each_frame       매 프레임 직전 위치에 다시 결합하는가 (ours·PhysTwin)
  rebind(P)               (그런 경우) 직전 위치 P 에 결합
  params()                최적화 변수
  yJ(X, sel)              변위 dy = y - X 와 야코비안 ∂y/∂X (선택 점)
  metric()                계량 잔차 (순수 함수, 인자들) -- 없으면 None (제약 없는 Adam)
"""
from __future__ import annotations

import math

import torch

from anchorflow import simplex as sx
from anchorflow import gsverse as gv


# ---------------------------------------------------------------- 공통 작은 연산
def det3(F):
    return (F[:, 0, 0] * (F[:, 1, 1] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 1])
            - F[:, 0, 1] * (F[:, 1, 0] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 0])
            + F[:, 0, 2] * (F[:, 1, 0] * F[:, 2, 1] - F[:, 1, 1] * F[:, 2, 0]))


def outer_sum(A, B):
    return (A[..., :, None] * B[..., None, :]).sum(1)


def mm3(A, B):
    return (A[..., :, :, None] * B[..., None, :, :]).sum(-2)


def eye_plus(M):
    return M + torch.eye(3, device=M.device, dtype=M.dtype)


def quat_mat(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                       -1).reshape(*q.shape[:-1], 3, 3)


def rot_to_quat(R):
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    w = torch.sqrt((1 + tr).clamp_min(1e-12)) / 2
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)
    return torch.stack([w, x, y, z], -1)


def fps(X, k, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    i = [int(torch.randint(X.shape[0], (1,), generator=g))]
    d = (X - X[i[0]]).norm(dim=1)
    for _ in range(k - 1):
        j = int(d.argmax()); i.append(j)
        d = torch.minimum(d, (X - X[j]).norm(dim=1))
    return torch.tensor(i, device=X.device)


# ---------------------------------------------------------------- 계량 순수 잔차
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
    TX = (Tm[None] * Xh[:, None, None, :]).sum(-1)
    F = eye_plus((W[..., None, None] * Tm[None, :, :, :3]).sum(1) + outer_sum(TX, dW))
    return torch.log(det3(F).clamp_min(1e-6)) * (W.shape[0] ** -0.5)


def gn_prod(fn, th, u, *args):
    """가우스-뉴턴 곱 Jᵀ J u."""
    from torch.func import jvp, vjp
    _, Ju = jvp(lambda z: fn(z, *args), (th,), (u,))
    _, vf = vjp(lambda z: fn(z, *args), th)
    return vf(Ju)[0]


# ---------------------------------------------------------------- 표현들
class Lattice(torch.nn.Module):
    """ours: 매 프레임 직전 위치에 사면체 격자를 새로 깔고 증분 변위 + Gregory·방사형 스키닝."""
    rebind_each_frame = True

    def __init__(self, X0, h, gregory=True):
        super().__init__()
        self.greg, self.h, self.a = gregory, h, 0.5
        self.dev = X0.device

    def rebind(self, P):
        lo, lat, nn = sx.grid_for_nodes(P, 1.0, h_fix=self.h)
        idx, lam, _ = sx.locate(P, lo, lat, nn)
        rows, uniq = sx.active_nodes(idx)
        self.rows = rows
        self.Xn = sx.node_pos(lo, lat, nn, uniq)
        Xv = self.Xn[rows]
        self.v0 = Xv[:, 0]
        self.Minv = torch.linalg.inv((Xv[:, 1:] - Xv[:, :1]).transpose(1, 2))
        M = uniq.numel()
        l3 = (self.Minv @ (P - self.v0)[..., None]).squeeze(-1)
        self.lam = torch.cat([1.0 - l3.sum(1, keepdim=True), l3], 1)
        self.dlam = torch.cat([-self.Minv.sum(1, keepdim=True), self.Minv], 1)
        if self.greg:
            d = P[:, None] - self.Xn[rows]
            self.r = d.norm(dim=-1)
            self.dr = d / self.r[..., None].clamp_min(1e-30)
            S = (self.lam * self.lam).sum(1, keepdim=True)
            self.w = self.lam * self.lam / S.clamp_min(1e-12)
            dS = (2 * self.lam[..., None] * self.dlam).sum(1, keepdim=True)
            self.dw = 2 * self.lam[..., None] * self.dlam / S[..., None] \
                - (self.lam * self.lam)[..., None] * dS / (S * S)[..., None]
        self.cells = torch.unique(rows, dim=0)
        Xc = self.Xn[self.cells]
        self.cD0i = torch.linalg.inv((Xc[:, 1:] - Xc[:, :1]).transpose(1, 2))
        self.u = torch.nn.Parameter(torch.zeros(M, 3, device=self.dev))
        self.rho_raw = torch.nn.Parameter(torch.zeros(M, device=self.dev)) if self.greg else None
        self.dof = (4 if self.greg else 3) * M

    def params(self):
        return [self.u] + ([self.rho_raw] if self.greg else [])

    def yJ(self, X, sel):
        rows = self.rows[sel]
        U = self.u[rows]
        if not self.greg:
            return (self.lam[sel][..., None] * U).sum(1), eye_plus(outer_sum(U, self.dlam[sel]))
        r, dr, w, dw = self.r[sel], self.dr[sel], self.w[sel], self.dw[sel]
        rho = self.h * (0.05 + 0.95 * torch.sigmoid(self.rho_raw)[rows])
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
        return (W[..., None] * U).sum(1), eye_plus(outer_sum(U, dW))

    def metric(self):
        return res_lattice_lb, (self.Xn, self.cells, self.cD0i)


class PhysTwin(torch.nn.Module):
    """질량점 + 공식 interpolate_motions (실시간 데모 대응: 이웃 인덱스는 첫 프레임에 한 번)."""
    rebind_each_frame = True

    def __init__(self, X0, K=16, seed=0):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = X0.shape[0]
        P = X0[torch.randperm(n, generator=g)[:min(11024, n)].to(X0.device)]
        vox = 0.005 / 0.25
        key = torch.floor((P - P.min(0).values) / vox).long()
        _, inv = torch.unique(key, dim=0, return_inverse=True)
        B = torch.zeros(int(inv.max()) + 1, 3, device=X0.device, dtype=X0.dtype).index_reduce_(
            0, inv, P, "mean", include_self=False)
        self.B, self.K = B, K
        self.rel = torch.cdist(B, B).topk(K + 1, largest=False).indices[:, 1:]
        self.L0 = (B[self.rel] - B[:, None]).norm(dim=-1)
        self.m, self.wi = None, None
        self.dof = 3 * B.shape[0]

    def rebind(self, P):
        if self.m is not None:
            self.B = (self.B + self.m.detach()).clone()
        if self.wi is None:
            self.wi = torch.cat([torch.cdist(P[i:i + 20000], self.B).topk(self.K, largest=False).indices
                                 for i in range(0, P.shape[0], 20000)])
        ii = self.wi
        d = P[:, None] - self.B[ii]
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
        Rk = self.bone_R()[k]
        dsel = self.d[sel]
        Rd = (Rk @ dsel[..., None]).squeeze(-1)
        moved = Rd + self.B[k] + self.m[k]
        dy = (w[..., None] * (Rd - dsel + self.m[k])).sum(1)
        J = (w[..., None, None] * Rk).sum(1) + outer_sum(moved, dw)
        return dy, J

    def metric(self):
        return res_spring, (self.B, self.rel, self.L0)


class GausSim(torch.nn.Module):
    """3 단 계층 (downsample 0.01, 0.01), 부피 보존 F."""
    rebind_each_frame = False

    def __init__(self, X0):
        super().__init__()
        n = X0.shape[0]
        k1 = max(int(round(0.01 * n)), 2)
        k2 = max(int(round(0.01 * k1)), 1)
        self.C1 = X0[fps(X0, k1)]
        self.C2 = self.C1[fps(self.C1, k2, 1)]
        self.par = torch.cdist(self.C1, self.C2).argmin(1)
        self.lab = torch.cat([torch.cdist(X0[i:i + 20000], self.C1).argmin(1)
                              for i in range(0, n, 20000)])
        self.p2 = torch.nn.Parameter(torch.zeros(k2, 3, device=X0.device))
        z = torch.zeros(k1, 11, device=X0.device)
        z[:, 0] = 1.0; z[:, 7] = 1.0
        self.dg = torch.nn.Parameter(z)
        self.dof = 3 * k2 + 11 * k1

    def params(self):
        return [self.p2, self.dg]

    def F1(self):
        U = quat_mat(self.dg[:, 0:4])
        s = torch.exp(self.dg[:, 4:7].clamp(-5, 5))
        s = s / torch.prod(s, -1, keepdim=True).pow(1 / 3)
        V = quat_mat(self.dg[:, 7:11])
        return U @ torch.diag_embed(s) @ V.transpose(1, 2)

    def yJ(self, X, sel):
        k = self.lab[sel]
        F = self.F1()[k]
        P = self.C2[self.par[k]]
        FmI = F - torch.eye(3, device=F.device, dtype=F.dtype)
        return self.p2[self.par[k]] + (FmI @ (X - P)[..., None]).squeeze(-1), F

    def metric(self):
        return None


class Simplicits(torch.nn.Module):
    """kaolin 기본 학습 가중치 (핸들 10) + 핸들별 3x4 변환."""
    rebind_each_frame = False

    def __init__(self, X0, fcn_path, center):
        super().__init__()
        fcn = torch.load(fcn_path, weights_only=False).to(X0.device)
        for q in fcn.parameters():
            q.requires_grad_(False)
        # 가중치는 학습 때 좌표(물체 중심 기준)로 잰다 -- 여러 물체면 물체마다 자기 중심
        Xl = X0 - center
        with torch.no_grad():
            K = fcn(Xl[:2].float()).shape[1]
        self.Tm = torch.nn.Parameter(torch.zeros(K, 3, 4, device=X0.device))
        self.dof = 12 * K
        Ws, dWs = [], []
        for i in range(0, X0.shape[0], 20000):
            Xc = Xl[i:i + 20000].detach().requires_grad_(True)
            Wc = fcn(Xc.float()).to(Xc.dtype)
            g = [torch.autograd.grad(Wc[:, j].sum(), Xc, retain_graph=True)[0] for j in range(K)]
            Ws.append(Wc.detach()); dWs.append(torch.stack(g, 1).detach())
        self.W, self.dW = torch.cat(Ws), torch.cat(dWs)
        self.Xh = torch.cat([X0, torch.ones_like(X0[:, :1])], 1)

    def params(self):
        return [self.Tm]

    def yJ(self, X, sel):
        W, dW = self.W[sel], self.dW[sel]
        Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)
        TX = (self.Tm[None] * Xh[:, None, None, :]).sum(-1)
        dy = (W[..., None] * TX).sum(1)
        J = (W[..., None, None] * self.Tm[None, :, :, :3]).sum(1) + outer_sum(TX, dW)
        return dy, eye_plus(J)

    def metric(self):
        return res_simp_lb, (self.W, self.dW, self.Xh)


class VRGS(torch.nn.Module):
    """GS-Verse: 표면 메시(n_grid 100 marching cubes, 표면 가우시안 중심으로) + 삼각형 국소 틀 결합."""
    rebind_each_frame = False

    def __init__(self, X0, surf):
        super().__init__()
        c = surf.mean(0)
        sh = torch.full_like(c, 1.0) - c                    # 메시 상자 [0,2] 안으로 옮겨 만든다
        v, f = gv.mesh_from_points((surf + sh).float(), n_grid=100, grid_lim=2.0)
        v = v.to(X0.dtype) - sh
        used = torch.unique(f)
        remap = torch.full((v.shape[0],), -1, dtype=torch.long, device=X0.device)
        remap[used] = torch.arange(used.numel(), device=X0.device)
        self.vr, self.f = v[used], remap[f]
        self.ti, self.a1, self.a2, self.b = gv.bind_points(X0, self.vr, self.f)
        self.v = torch.nn.Parameter(self.vr.clone())
        self.A0i = torch.linalg.inv(torch.stack(gv._tri_frame(self.vr, self.f)[1:], -1))
        self.dof = 3 * self.vr.shape[0]
        v0, v1, v2 = self.vr[self.f[:, 0]], self.vr[self.f[:, 1]], self.vr[self.f[:, 2]]
        nrm = torch.cross(v1 - v0, v2 - v0, dim=-1)
        self.A0 = 0.5 * nrm.norm(dim=-1)
        self.n0 = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def params(self):
        return [self.v]

    def yJ(self, X, sel):
        v0, e1, e2, n = gv._tri_frame(self.v, self.f)
        t = self.ti[sel]
        y = v0[t] + self.a1[sel, None] * e1[t] + self.a2[sel, None] * e2[t] + self.b[sel, None] * n[t]
        A = torch.stack([e1, e2, n], -1)
        return y - X, (A @ self.A0i)[t]

    def metric(self):
        return res_tri_lb, (self.f, self.n0, self.A0)


class Multi:
    """여러 물체: 물체마다 따로 표현. 점 순서는 전체 배열의 번호(idx[o])로 이어 붙인다."""

    def __init__(self, reps, idx):
        self.reps, self.idx = reps, idx
        self.rebind_each_frame = reps[0].rebind_each_frame

    def rebind(self, P):
        for r, ii in zip(self.reps, self.idx):
            r.rebind(P[ii])

    def params(self):
        return [q for r in self.reps for q in r.params()]

    @property
    def dof(self):
        return sum(r.dof for r in self.reps)

    def yJ(self, X):
        N = X.shape[0]
        dy = torch.zeros_like(X)
        J = torch.zeros(N, 3, 3, device=X.device, dtype=X.dtype)
        parts_dy, parts_J = [], []
        for r, ii in zip(self.reps, self.idx):
            d_, J_ = r.yJ(X[ii], torch.arange(ii.numel(), device=X.device))
            parts_dy.append((ii, d_)); parts_J.append((ii, J_))
        dy = dy.index_put((torch.cat([p[0] for p in parts_dy]),), torch.cat([p[1] for p in parts_dy]))
        J = J.index_put((torch.cat([p[0] for p in parts_J]),), torch.cat([p[1] for p in parts_J]))
        return dy, J

    def metrics(self):
        return [r.metric() for r in self.reps]
