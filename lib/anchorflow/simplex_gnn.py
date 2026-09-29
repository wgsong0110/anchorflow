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

import torch
import torch.nn as nn

from .deform import _inv3
from .nextstate import DtFiLM
from .simplex import N_EDGE_CLASS

__all__ = ["scatter_to_nodes", "node_moments", "SimplexGNN"]


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
    Wa = g1[:, 0].clamp(min=1e-12)
    Wi = Wa.unsqueeze(-1)
    cx, cX, cv = g1[:, 1:4] / Wi, g1[:, 4:7] / Wi, g1[:, 7:10] / Wi
    cnt = g1[:, 10:11]
    dx = x.unsqueeze(1) - cx[rows]
    dX = X.unsqueeze(1) - cX[rows]
    dv = v.unsqueeze(1) - cv[rows]
    ww = wm.reshape(-1, K, 1, 1)
    g2 = acc(torch.cat([
        (ww * (dx.unsqueeze(-1) * dx.unsqueeze(-2))).reshape(-1, K, 9),
        wmf * torch.cross(dx, dv, dim=-1),
        (ww * (dx.unsqueeze(-1) * dX.unsqueeze(-2))).reshape(-1, K, 9),
        (ww * (dX.unsqueeze(-1) * dX.unsqueeze(-2))).reshape(-1, K, 9)], -1))
    S = g2[:, :9].reshape(n_node, 3, 3) / Wa.reshape(-1, 1, 1)
    iu = torch.triu_indices(3, 3, device=dev)
    S6 = S[:, iu[0], iu[1]] / (hn * hn)
    L = g2[:, 9:12]
    tr = S.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(-1, 1, 1)
    I3 = torch.eye(3, device=dev, dtype=x.dtype)
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
                 dt_cond=True, dt_ref=1.0, dt_scale=False, n_mat=0):
        super().__init__()
        self.scale = scale
        self.register_buffer("in_mu", torch.zeros(n_feat))
        self.register_buffer("in_sd", torch.ones(n_feat))
        self.inp = nn.Linear(n_feat, hidden)
        self.body = nn.ModuleList([_MPLayer(hidden) for _ in range(layers)])
        self.out = nn.Linear(hidden, 3)
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
        return (self.out(h) * sc,)
