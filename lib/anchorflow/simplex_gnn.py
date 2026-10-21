"""사면체 복합체 위의 메시지 패싱 스테퍼.

노드는 **점유 사면체의 꼭짓점**뿐이고 (빈 곳은 계산하지 않는다), 간선은 그
사면체들의 실제 변이다. 간선 종류는 2 배 격자에서의 상대 오프셋으로 정하는데
**방향까지 구분**한다 -- {-1,0,1}^3 의 0 아닌 26 가지 (축 6 + 면대각 12 +
체대각 8). +x 와 -x 는 다른 간선이다.

노드 특징은 입자 물리량을 자기 사면체의 **barycentric 가중으로 4 꼭짓점에
뿌려** 모은다 (MPM 의 P2G 와 같은 구조, 전달 가중치를 그대로 재사용).

출력은 노드별 변위 dp 이고 0 초기화라 시작은 항등이다. dt 는 조건 변수로
층마다 DtFiLM 으로 들어간다 (conv 경로와 같은 모듈).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .deform import _inv3
from .nextstate import DtFiLM
from .simplex import N_EDGE_CLASS

__all__ = ["scatter_to_nodes", "node_moments", "node_centroids",
           "node_moments_finish", "SimplexGNN"]


def scatter_to_nodes(vals, rows, lam, n_node, mass=None):
    """입자 값 [N,F] -> 노드 값 [M,F]. barycentric 가중 평균.

    mass 가 있으면 질량가중 (관성 같은 외연량에 맞다). 가중치 합으로 나눠
    입자 밀도가 다른 노드끼리 크기가 어긋나지 않게 한다.
    """
    w = lam if mass is None else lam * mass.reshape(-1, 1)
    num = torch.zeros(n_node, vals.shape[-1], device=vals.device,
                      dtype=vals.dtype)
    den = torch.zeros(n_node, 1, device=vals.device, dtype=vals.dtype)
    r = rows.reshape(-1)
    num.index_add_(0, r, (w.unsqueeze(-1) * vals.unsqueeze(1)).reshape(
        -1, vals.shape[-1]))
    den.index_add_(0, r, w.reshape(-1, 1))
    return num / den.clamp_min(1e-12)


def node_moments(x, v, X, m, rows, lam, n_node, npos, hn):
    """노드마다 **국소 통계**를 barycentric 질량가중으로 쌓는다.

    격자 시절 tri_feats 가 주던 것과 같은 항목이다 (가중치만 trilinear ->
    barycentric). 변위·속도 평균만으로는 탄성항이 보는 것을 담지 못한다 --
    2 차 모멘트(국소 형상), 각속도, **국소 변형구배 추정 F = A B^-1** 이
    있어야 한다.

    -> [M, 1+3+3+3+1+6+3+9 = 29]
    """
    dev = x.device
    wm = lam * m.unsqueeze(1)                              # [N,4]
    K = rows.shape[1]
    r = rows.reshape(-1)

    def acc(vals):
        F = vals.shape[-1]
        out = torch.zeros(n_node, F, device=dev, dtype=vals.dtype)
        return out.index_add_(0, r, vals.reshape(-1, F))

    wmf = wm.unsqueeze(-1)
    ones = torch.ones_like(wmf)
    g1 = acc(torch.cat([wmf, wmf * x.unsqueeze(1), wmf * X.unsqueeze(1),
                        wmf * v.unsqueeze(1), ones], -1))
    Wa, cx, cX, cv = node_centroids(g1)
    dx = x.unsqueeze(1) - cx[rows]
    dX = X.unsqueeze(1) - cX[rows]
    dv = v.unsqueeze(1) - cv[rows]
    ww = wm.reshape(-1, K, 1, 1)
    g2 = acc(torch.cat([
        (ww * (dx.unsqueeze(-1) * dx.unsqueeze(-2))).reshape(-1, K, 9),
        wmf * torch.cross(dx, dv, dim=-1),
        (ww * (dx.unsqueeze(-1) * dX.unsqueeze(-2))).reshape(-1, K, 9),
        (ww * (dX.unsqueeze(-1) * dX.unsqueeze(-2))).reshape(-1, K, 9)], -1))
    return node_moments_finish(g1, g2, npos, hn)


PFREQ_NS = 15        # --pfreq 가 인코딩하는 입자-노드 스칼라 수: 상대위치 3 + 속도 3 + (F - I) 9


def pfreq_channels(K):
    """--pfreq K 노드 특징 폭 = 15 × (1 + 2K)."""
    return PFREQ_NS * (1 + 2 * int(K))


def particle_freq_feats(x, v, F, m, rows, lam, n_node, npos, hn, K):
    """**--pfreq K** 노드 특징 (node_moments 29 채널 대신, 사용자 10-21).

    입자 p 가 자기 원소 꼭짓점 노드 j (rows[p, c] = j, 가중 lam[p, c] = λ_pj) 에 대해 갖는 스칼라 15 개
        s = [ (x_p - x_j)/hn (3),  v_p (3, 부르는 쪽이 vel_scale 로 나눈 값),  F_p - I (9, 행 우선) ]
    를 하나씩 [s, sin(2^k π s) (k<K), cos(2^k π s) (k<K)] 로 인코딩해 노드 j 로 λ_pj·m_p 가중 평균한다
    (node_moments 와 같은 가중·같은 분모 Σλm, 같은 하한 1e-12 -- 가중 0 인 노드는 0).
    채널 순서: 스칼라 i (위 순서) 마다 1+2K 칸 [s, sin_0..sin_{K-1}, cos_0..cos_{K-1}].

    x, v: [N,3], F: [N,3,3] 또는 [N,9], m: [N], rows/lam: [N,C], npos: [M,3] -> [M, 15(1+2K)]
    """
    dev, dt = x.device, x.dtype
    K = int(K)
    N, C = rows.shape
    F9 = F.reshape(N, 9).to(dt) - torch.eye(3, device=dev, dtype=dt).reshape(1, 9)
    v = v.to(dt)
    fr = (2.0 ** torch.arange(K, device=dev, dtype=dt)) * math.pi                  # [K]
    nch = pfreq_channels(K)
    num = torch.zeros(n_node, nch, device=dev, dtype=dt)
    den = torch.zeros(n_node, 1, device=dev, dtype=dt)
    for c in range(C):                     # 꼭짓점 열마다 (메모리: N × 15(1+2K) 하나씩)
        r = rows[:, c]
        w = (lam[:, c].to(dt) * m.to(dt)).unsqueeze(-1)                            # [N,1]
        s = torch.cat([(x - npos[r]) / hn, v, F9], -1)                              # [N,15]
        ang = s.unsqueeze(-1) * fr                                                  # [N,15,K]
        enc = torch.cat([s.unsqueeze(-1), torch.sin(ang), torch.cos(ang)], -1)      # [N,15,1+2K]
        num.index_add_(0, r, w * enc.reshape(N, nch))
        den.index_add_(0, r, w)
    return num / den.clamp(min=1e-12)


def node_centroids(g1):
    """1 차 누적 g1 [M,11] -> (가중치 합 Wa, 질량중심 cx, cX, cv).

    node_moments 와 롤아웃 빠른 경로(fast_roll) 가 함께 쓴다 (같은 연산).
    """
    Wa = g1[:, 0].clamp(min=1e-12)
    Wi = Wa.unsqueeze(-1)
    return Wa, g1[:, 1:4] / Wi, g1[:, 4:7] / Wi, g1[:, 7:10] / Wi


def node_moments_finish(g1, g2, npos, hn):
    """누적 g1 [M,11] (가중치·1 차) 과 g2 [M,30] (중심 둘레 2 차) -> 노드 특징 [M,29].

    node_moments 의 뒷부분을 떼어 낸 것 -- 빠른 경로가 누적만 융합 커널로 하고
    여기부터는 같은 연산을 쓰게 하려는 것이다.
    """
    dev, n_node = g1.device, g1.shape[0]
    Wa, cx, cX, cv = node_centroids(g1)
    cnt = g1[:, 10:11]
    S = g2[:, :9].reshape(n_node, 3, 3) / Wa.reshape(-1, 1, 1)
    iu = torch.triu_indices(3, 3, device=dev)
    S6 = S[:, iu[0], iu[1]] / (hn * hn)
    L = g2[:, 9:12]
    tr = S.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(-1, 1, 1)
    I3 = torch.eye(3, device=dev, dtype=g1.dtype)
    Ii, _ = _inv3((tr * I3 - S) * Wa.reshape(-1, 1, 1) + 1e-8 * I3)
    om = (Ii @ L.unsqueeze(-1)).squeeze(-1)
    A = g2[:, 12:21].reshape(n_node, 3, 3)
    B = g2[:, 21:30].reshape(n_node, 3, 3)
    Bi, _ = _inv3(B + (1e-6 * hn * hn) * I3)
    Fa = (A @ Bi).clamp(-20.0, 20.0).reshape(n_node, 9)
    return torch.cat([torch.log1p(Wa.unsqueeze(-1) * 1e3), (cx - npos) / hn,
                      (cX - npos) / hn, cv, torch.log1p(cnt),
                      S6, om * hn, Fa - I3.reshape(1, 9)], -1)


# 진단 스위치 -- 격자 conv 에서 GroupNorm 이 크기 정보를 지운 전례가 있어
# (_NORM=False 옵션), 같은 함정을 가릴 수 있게 둔다.
_MP_NORM = True      # False 면 갱신의 LayerNorm 제거
_MP_MEAN = True      # False 면 메시지 집계를 평균 대신 합으로


class _MPLayer(nn.Module):
    """한 층: 간선 종류별 선형 변환 + 집계 + 노드 갱신 (잔차)."""

    def __init__(self, hidden):
        super().__init__()
        # 간선 종류 임베딩을 곱해 메시지를 만든다. 종류가 26 개뿐이라
        # 종류별 가중행렬을 통째로 두는 대신 임베딩 변조로 둔다 (파라미터 절약).
        self.emb = nn.Embedding(N_EDGE_CLASS, hidden)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.msg = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))
        self.nrm = nn.LayerNorm(hidden) if _MP_NORM else nn.Identity()

    def forward(self, h, src, dst, cls):
        e = self.emb(cls)                                   # [E,H]
        m = self.msg(torch.cat([h[src] * e, h[dst]], -1))   # [E,H]
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        if _MP_MEAN:
            # 평균은 이웃 수가 제각각인 경계 노드에서 신호를 희석한다 --
            # 합이 나은지 실측으로 가린다
            cnt = torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype)
            cnt.index_add_(0, dst, torch.ones_like(m[:, :1]))
            agg = agg / cnt.clamp_min(1.0)
        return h + self.nrm(self.upd(torch.cat([h, agg], -1)))


class SimplexGNN(nn.Module):
    """forward(feat, src, dst, cls, dt) -> (dp [M,3],)

    feat 은 **노드** 기준 [M,F] 다 (scatter_to_nodes 로 만든다).
    """

    def __init__(self, n_feat, hidden=128, layers=8, scale=1.0,
                 dt_cond=True, dt_ref=1.0, dt_scale=False, n_mat=0, n_out=3):
        super().__init__()
        self.scale = scale
        self.register_buffer("in_mu", torch.zeros(n_feat))
        self.register_buffer("in_sd", torch.ones(n_feat))
        self.inp = nn.Linear(n_feat, hidden)
        self.body = nn.ModuleList([_MPLayer(hidden) for _ in range(layers)])
        # n_out 4 = 변위 3 + Gregory·방사형 스키닝 반경 ρ_raw 1 (lib/anchorflow/skin.py, --gregory)
        self.n_out = int(n_out)
        self.out = nn.Linear(hidden, self.n_out)
        nn.init.zeros_(self.out.weight)     # 시작은 항등 (변위 0)
        nn.init.zeros_(self.out.bias)
        # 물성은 씬 안에서 상수라 특징에 붙이면 채널을 상수로 채우는 셈이다.
        self.n_mat = int(n_mat)
        if self.n_mat:
            self.mfilm = nn.Sequential(nn.Linear(self.n_mat, hidden), nn.SiLU(),
                                       nn.Linear(hidden, 2 * hidden))
            nn.init.zeros_(self.mfilm[-1].weight)
            nn.init.zeros_(self.mfilm[-1].bias)
        self.dt_ref = float(dt_ref) if dt_ref else 1.0
        self.dt_scale = bool(dt_scale)
        self.dtfilm = DtFiLM(hidden, layers + 1) if dt_cond else None

    def set_input_stats(self, feats):
        f = feats.detach().float().reshape(-1, feats.shape[-1])
        sd = f.std(0)
        self.in_mu.copy_(f.mean(0))
        self.in_sd.copy_(torch.where(sd > 1e-4 * sd.max().clamp(min=1e-12),
                                     sd, torch.ones_like(sd)))

    def _dtmod(self, h, dt, i):
        if self.dtfilm is None:
            return h
        g, b = self.dtfilm(dt, h.device)
        j = min(i, self.dtfilm.n_sites - 1)
        return g[j].to(h.dtype) * h + b[j].to(h.dtype)

    def forward(self, feat, src, dst, cls, dt, mat=None):
        h = self.inp((feat - self.in_mu) / self.in_sd)
        if self.n_mat and mat is not None:
            g, b = self.mfilm(mat).chunk(2, -1)
            h = (1.0 + g) * h + b
        h = self._dtmod(h, dt, 0)
        for i, blk in enumerate(self.body):
            h = blk(h, src, dst, cls)
            h = self._dtmod(h, dt, i + 1)
        # 변위는 1 차로 v*dt 라 dt 에 비례한다 (--dt_scale)
        sc = (self.scale * (dt / self.dt_ref) if self.dt_scale else self.scale)
        o = self.out(h)
        if self.n_out == 4:
            # 반경 채널은 무차원 (σ 로 들어간다) -- 변위 크기·dt 비례를 곱하지 않는다
            return (torch.cat([o[:, :3] * sc, o[:, 3:4]], -1),)
        return (o * sc,)
