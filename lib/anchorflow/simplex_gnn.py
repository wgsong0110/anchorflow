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
           "node_moments_finish", "SimplexGNN", "p2g_feats", "P2G_NCH",
           "LocalAttnGNN", "ring_edges", "ring_offset_table",
           "GraphUNetGNN", "grid_ijk", "coarsen_grid", "gunet_plan"]


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


P2G_NCH = 13         # --node_feat p2g 노드 입력 폭: log1p(1e3·m_i) 1 + v_i 3 + (F_i - I) 9


def p2g_feats(v, F, m, rows, w, n_node):
    """**--node_feat p2g** 노드 특징 (사용자 10-23): MPM P2G 그대로 입자 -> 노드.

    rows/w: 입자마다 27 노드 (2 차 B-스플라인, 격자 좌표 기준 텐서곱) 의 노드 색인·가중 w_ip [N,27].
    활성 노드가 아닌 이웃은 부르는 쪽이 덤프 노드(n_node-1 번)로 보내고 가중 0 으로 둔다.
      m_i = Σ_p w_ip m_p,  v_i = Σ_p w_ip m_p v_p / m_i,  F_i = Σ_p w_ip m_p F_p / m_i
    정규화는 node_moments 규약 그대로: 질량 칸 log1p(1e3·m_i) (node_moments 의 log1p(1e3·Σλm) 과 같은 꼴),
    속도는 부르는 쪽이 vel_scale 로 나눈 값, F 는 F_i - I (node_moments 의 Fa - I 와 같은 꼴). 분모 하한 1e-12.

    v: [N,3], F: [N,3,3] 또는 [N,9], m: [N] -> [M, 13] = [log1p(1e3 m_i), v_i (3), F_i - I (9, 행 우선)]
    """
    dev, dt = v.device, v.dtype
    N, C = rows.shape
    F9 = F.reshape(N, 9).to(dt)
    wm = w.to(dt) * m.to(dt).unsqueeze(1)                                  # [N,27]
    r = rows.reshape(-1)
    num = torch.zeros(n_node, 1 + 3 + 9, device=dev, dtype=dt)
    vals = torch.cat([torch.ones_like(v[:, :1]), v, F9], -1)               # [N,13]
    num.index_add_(0, r, (wm.unsqueeze(-1) * vals.unsqueeze(1)).reshape(-1, 13))
    mi = num[:, :1]
    den = mi.clamp(min=1e-12)
    I9 = torch.eye(3, device=dev, dtype=dt).reshape(1, 9)
    return torch.cat([torch.log1p(mi * 1e3), num[:, 1:4] / den, num[:, 4:13] / den - I9], -1)


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


class _LAttnLayer(nn.Module):
    """**국소 어텐션** 한 층 (--gnn_layer lattn, 사용자 10-23).

    각 노드(질의)는 메시지 패싱이 쓰는 **같은 이웃 집합** (src -> dst 간선, edges_of) 에만 주의를 둔다.
    다중 헤드; 상대 위치 인코딩 = 간선의 격자 상대 오프셋 (EDGE_OFFSETS[cls], 정수 3) 을 MLP 로
      키 편향 r_k, 값 편향 r_v (폭 hidden) 와 헤드별 로짓 편향 b (헤드 수) 로 넣는다:
        logit_e,h = <q_dst, k_src + r_k(cls)>_h / sqrt(d_h) + b_h(cls),  softmax 는 같은 dst 의 간선끼리
        out_dst  = Σ_e α_e,h (v_src + r_v(cls))_h
    잔차 + LayerNorm (pre-norm): h <- h + Wo·attn(LN(h)), h <- h + FFN(LN(h)).
    """

    def __init__(self, hidden, heads, ring=1):
        super().__init__()
        if hidden % heads:
            raise SystemExit(f"--attn_heads {heads} 가 --hidden {hidden} 을 나누지 못한다")
        from .simplex import EDGE_OFFSETS
        self.H, self.dh = int(heads), hidden // int(heads)
        # ring 1: 간선 종류 14 가지 표 (cls = edges_of 의 클래스). ring K>1: [-K,K]^3 전부의 표 (cls = ring_edges 의 색인)
        _tab = EDGE_OFFSETS if int(ring) == 1 else ring_offset_table(int(ring))
        self.register_buffer("off", torch.tensor(_tab, dtype=torch.float32), persistent=False)
        self.ln1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.rpe = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, 2 * hidden + self.H))
        self.o = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden))

    def forward(self, h, src, dst, cls):
        M, D = h.shape
        H, dh = self.H, self.dh
        q, k, v = self.qkv(self.ln1(h)).split(D, -1)
        rt = self.rpe(self.off.to(h.dtype))                                 # [14, 2D+H] (간선 종류 14 가지뿐; ring K>1 이면 (2K+1)^3)
        rk, rv, rb = rt[:, :D], rt[:, D:2 * D], rt[:, 2 * D:]
        ke = (k[src] + rk[cls]).reshape(-1, H, dh)
        ve = (v[src] + rv[cls]).reshape(-1, H, dh)
        lg = (q[dst].reshape(-1, H, dh) * ke).sum(-1) / math.sqrt(dh) + rb[cls]     # [E,H]
        # 같은 dst 끼리 softmax: 최댓값 빼기는 상수 이동이라 값에 무관 -- 기울기를 끊어 둔다
        mx = torch.full((M, H), -1e30, device=h.device, dtype=lg.dtype).scatter_reduce(
            0, dst.unsqueeze(-1).expand(-1, H), lg.detach(), reduce="amax", include_self=True)
        ex = torch.exp(lg - mx[dst])
        den = torch.zeros(M, H, device=h.device, dtype=lg.dtype).index_add(0, dst, ex)
        al = ex / den[dst].clamp_min(1e-30)                                  # [E,H]
        agg = torch.zeros(M, H, dh, device=h.device, dtype=h.dtype).index_add(0, dst, al.unsqueeze(-1) * ve)
        h = h + self.o(agg.reshape(M, D))
        return h + self.ff(self.ln2(h))


def ring_offset_table(K):
    """[-K,K]^3 정수 오프셋 전부 ((2K+1)^3 행). 색인 = (ox+K)(2K+1)^2 + (oy+K)(2K+1) + (oz+K)."""
    r = range(-K, K + 1)
    return [(i, j, k) for i in r for j in r for k in r]


def ring_edges(src, dst, cls, K, n_node=None):
    """1 링 간선 (src -> dst, 종류 cls = EDGE_OFFSETS 색인) -> **K 링** 간선 (사용자 10-23, --attn_ring K).

    dst 의 K 링 = 간선을 K 번 이하로 건너 닿는 노드 전부 (자기 자신 제외, 중복 제거).
    희소 확장: 쌍 (d, s, o = pos[d]-pos[s]) 마다 s 로 들어오는 1 링 간선 (s2 -> s, o2) 을 붙여 (d, s2, o + o2) --
    한 바퀴에 **새로 생긴 쌍만** 다음 바퀴로 넘긴다 (BFS 경계). 키 d·M + s 로 torch.unique.
    오프셋은 경로에 무관하다 (pos[d]-pos[s2]) -- 그래서 어느 경로로 왔든 같은 값.
    K = 1 이면 들어온 그대로 돌려준다 (cls 도 EDGE_OFFSETS 색인 그대로).
    -> (src [E_K], dst [E_K], cls [E_K] = ring_offset_table(K) 색인), (src, dst) 키 오름차순 정렬
    """
    K = int(K)
    if K <= 1:
        return src, dst, cls
    from .simplex import EDGE_OFFSETS
    dev = src.device
    M = int(n_node) if n_node is not None else int(torch.maximum(src.max(), dst.max())) + 1
    src, dst = src.long(), dst.long()
    o1 = torch.tensor(EDGE_OFFSETS, device=dev, dtype=torch.long)[cls.long()]       # pos[dst] - pos[src]
    order = torch.argsort(dst, stable=True)
    in_src, in_off = src[order], o1[order]                    # dst 별 들어오는 간선 (CSR)
    deg = torch.bincount(dst, minlength=M)
    ptr = torch.zeros(M + 1, device=dev, dtype=torch.long)
    ptr[1:] = torch.cumsum(deg, 0)
    all_key = dst * M + src
    all_off = o1
    fd, fs, fo = dst, src, o1                                 # 경계 (지난 바퀴에 새로 생긴 쌍)
    for _ in range(K - 1):
        n = deg[fs]
        tot = int(n.sum())
        if tot == 0:
            break
        rep = torch.repeat_interleave(torch.arange(fs.numel(), device=dev), n)
        base = torch.cumsum(n, 0) - n
        e = ptr[fs][rep] + (torch.arange(tot, device=dev) - base[rep])
        nd, ns, no = fd[rep], in_src[e], fo[rep] + in_off[e]
        keep = nd != ns                                       # 자기 자신 제외
        nd, ns, no = nd[keep], ns[keep], no[keep]
        key = nd * M + ns
        uk, inv = torch.unique(key, return_inverse=True)
        uo = torch.zeros(uk.numel(), 3, device=dev, dtype=torch.long).index_copy_(
            0, inv, no)                                       # 같은 키는 같은 오프셋 (경로 무관)
        new = ~torch.isin(uk, all_key)
        uk, uo = uk[new], uo[new]
        all_key = torch.cat([all_key, uk])
        all_off = torch.cat([all_off, uo])
        fd, fs, fo = uk // M, uk % M, uo
    # (src, dst) 오름차순 (edges_of 와 같은 정렬 규약)
    sk = (all_key % M) * M + all_key // M
    srt = torch.argsort(sk)
    all_key, all_off = all_key[srt], all_off[srt]
    d_, s_ = all_key // M, all_key % M
    W = 2 * K + 1
    c_ = (all_off[:, 0] + K) * W * W + (all_off[:, 1] + K) * W + (all_off[:, 2] + K)
    if bool((all_off.abs() > K).any()):
        raise RuntimeError("ring_edges: 오프셋이 [-K,K] 를 넘었다")
    return s_, d_, c_


class LocalAttnGNN(SimplexGNN):
    """SimplexGNN 과 같은 입출력·입력 정규화·dt/물성 FiLM·0 초기화 출력, 몸통 층만 국소 어텐션 (--gnn_layer lattn).

    ring (--attn_ring K, 사용자 10-23): 어텐션 이웃 = K 링 (1 = 지금, 메시지 패싱 이웃 그대로). K>1 이면 forward 가
    받은 1 링 간선에서 ring_edges 로 K 링 간선을 만든다 -- 같은 간선 텐서 객체가 다시 오면 (한 프레임 안의 Adam 반복)
    만들어 둔 것을 쓴다. ring_stats: 마지막으로 만든 이웃 목록의 노드 수·간선 수·노드당 평균/최소/최대 이웃·구성 ms.
    """

    def __init__(self, n_feat, hidden=128, layers=8, heads=8, ring=1, **kw):
        super().__init__(n_feat, hidden=hidden, layers=layers, **kw)
        self.ring = int(ring)
        if self.ring < 1:
            raise SystemExit(f"--attn_ring {ring} 는 1 이상")
        self.body = nn.ModuleList([_LAttnLayer(hidden, heads, ring=self.ring) for _ in range(layers)])
        self._rc = None              # (src, dst, cls 원본 객체, 결과) -- 원본을 붙잡아 두어 같은 객체 판정이 안전하다
        self.ring_stats = None

    def forward(self, feat, src, dst, cls, dt, mat=None):
        rc = self._rc
        if rc is None or rc[0] is not src or rc[1] is not dst or rc[2] is not cls:
            import time as _t
            if feat.is_cuda:
                torch.cuda.synchronize(feat.device)
            t0 = _t.time()
            out = ring_edges(src, dst, cls, self.ring, n_node=feat.shape[0])
            if feat.is_cuda:
                torch.cuda.synchronize(feat.device)
            ms = 1e3 * (_t.time() - t0)
            nb = torch.bincount(out[1], minlength=feat.shape[0])
            self.ring_stats = dict(K=self.ring, M=int(feat.shape[0]), E=int(out[1].numel()), avg=float(nb.float().mean()),
                                   min=int(nb.min()), max=int(nb.max()), build_ms=ms)
            self._rc = rc = (src, dst, cls, out)
        return super().forward(feat, *rc[3], dt, mat=mat)


# ---------------------------------------------------------------- Graph U-Net (--gnn_layer unet, 사용자 10-24)
def grid_ijk(uniq, nn):
    """압축 노드 색인 uniq [M] (평탄 색인) -> 정수 격자좌표 [M,3] (edges_of 와 같은 풀이)."""
    nnl = [int(nn[k]) for k in range(3)]
    z = uniq % nnl[2]
    y = (uniq // nnl[2]) % nnl[1]
    xx = uniq // (nnl[1] * nnl[2])
    return torch.stack([xx, y, z], -1).long()


def _off_cls(off):
    """정수 오프셋 [E,3] (성분 {-1,0,1}) -> 간선 종류 (EDGE_OFFSETS 색인, 없으면 -1)."""
    from .simplex import EDGE_OFFSETS
    tbl = torch.full((27,), -1, device=off.device, dtype=torch.long)
    for c, o in enumerate(EDGE_OFFSETS):
        tbl[(o[0] + 1) * 9 + (o[1] + 1) * 3 + (o[2] + 1)] = c
    ok = (off.abs() <= 1).all(-1)
    key = ((off[:, 0] + 1) * 9 + (off[:, 1] + 1) * 3 + (off[:, 2] + 1)).clamp(0, 26)
    return torch.where(ok, tbl[key], torch.full_like(key, -1))


def coarsen_grid(pos, src, dst):
    """한 수준 거칠게 (간격 h -> 2h): 거친 노드 = 정수 좌표를 2 로 내림 나눈 칸 (점유 칸만).

    거친 간선 = 고운 간선 (s -> d) 의 부모 쌍 (P[s] -> P[d]), 자기 자신 제외, 중복 제거 --
    물체 위상을 따른다 (고운 수준에서 이어진 칸끼리만 이웃). 고운 Kuhn 변은 성분 부호가 한쪽뿐
    ({0,1}^3 또는 {0,-1}^3) 이라 부모 오프셋도 같은 14 종류 안에 든다 (아니면 오류).
    -> (parent [M] 거친 색인, cpos [Mc,3], (csrc, cdst, ccls), cnt [Mc] 자식 수)
    """
    cp = torch.div(pos, 2, rounding_mode="floor")
    mn = cp.min(0).values
    q = cp - mn
    R = q.max(0).values + 1
    key = (q[:, 0] * R[1] + q[:, 1]) * R[2] + q[:, 2]
    uk, parent = torch.unique(key, return_inverse=True)
    Mc = int(uk.numel())
    cpos = torch.zeros(Mc, 3, device=pos.device, dtype=pos.dtype).index_copy_(0, parent, cp)
    cnt = torch.bincount(parent, minlength=Mc)
    cs, cd = parent[src], parent[dst]
    keep = cs != cd
    cs, cd = cs[keep], cd[keep]
    if cs.numel():
        e = torch.unique(torch.stack([cs, cd], -1), dim=0)
        cs, cd = e[:, 0], e[:, 1]
    ccls = _off_cls(cpos[cd] - cpos[cs])
    if bool((ccls < 0).any()):
        raise RuntimeError("coarsen_grid: 거친 간선 오프셋이 14 종류 밖이다")
    return parent, cpos, (cs, cd, ccls), cnt


def gunet_plan(levels, enc=2, bot=2, dec=1):
    """층 배분: [(수준, 'enc'|'bot'|'dec', 층 수)] -- 인코더 0..L-2 각 enc, 바닥 L-1 bot, 디코더 L-2..0 각 dec."""
    L = int(levels)
    if L < 2:
        raise SystemExit(f"--gunet_levels {levels} 는 2 이상")
    return ([(l, "enc", enc) for l in range(L - 1)] + [(L - 1, "bot", bot)]
            + [(l, "dec", dec) for l in range(L - 2, -1, -1)])


class GraphUNetGNN(SimplexGNN):
    """**Graph U-Net** 몸통 (Gao & Ji 2019 구조를 우리 격자에 맞게, --gnn_layer unet, 사용자 10-24).

    수준 0 = 원 격자 (node_h), 수준 l = 간격 2^l h. 풀링은 top-k (gPool) 대신 **격자 거칠게 묶기**
    (coarsen_grid: 정수 좌표 // 2, 특징은 자식 평균), 언풀링은 부모 특징을 자식에 복사한 뒤 같은 수준
    인코더 특징과 스킵 (이어붙이기 -> 선형, 잔차: h = skip + W[skip, up]). 각 수준의 층은 SimplexGNN 의
    메시지 전달 층 (_MPLayer, 간선 종류 14 그대로) 이다. 층 배분 gunet_plan(levels): 기본 3 수준이면
    인코더 0·1 각 2 층, 바닥(4h) 2 층, 디코더 1·0 각 1 층 = 8 층 (기존 8 층 128 과 같은 층 수).
    입력 정규화·물성 FiLM·dt FiLM (층마다 한 자리, 순서대로)·0 초기화 출력은 SimplexGNN 그대로.

    forward 는 grid=(uniq, nn[, lo, lat]) (노드 압축 색인 -> 정수 좌표; lo·lat 가 있으면 격자 원점을 더한 전역 정수 좌표로 묶어
    물체가 움직여도 묶는 칸이 월드에 고정된다) 가 필요하다 (deform_model.advance 가 needs_grid 를 보고 넘긴다).
    같은 (uniq, src) 객체가 다시 오면 (한 프레임 안의 Adam 반복) 만들어 둔 계층을 쓴다. 노드 수가 바뀌면 새로 만든다.
    level_stats: 마지막 계층의 수준별 노드·간선 수.
    """

    needs_grid = True

    def __init__(self, n_feat, hidden=128, levels=3, enc=2, bot=2, dec=1, **kw):
        kw.pop("layers", None)
        self.plan = gunet_plan(levels, enc, bot, dec)
        n_layers = sum(n for _, _, n in self.plan)
        super().__init__(n_feat, hidden=hidden, layers=n_layers, **kw)
        self.levels = int(levels)
        self.body = nn.ModuleList([_MPLayer(hidden) for _ in range(n_layers)])
        self.fuse = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(self.levels - 1)])
        self._hc = None
        self.level_stats = None

    def hierarchy(self, src, dst, cls, grid):
        uniq, nn_ = grid[0], grid[1]
        hc = self._hc
        if hc is not None and hc[0] is uniq and hc[1] is src:
            return hc[2]
        pos = grid_ijk(uniq, nn_)
        if len(grid) >= 4 and grid[2] is not None:
            # 격자 원점 lo (월드) = A·imin (grid_for_nodes) -> 전역 정수 좌표 = 국소 + imin. 바운딩박스가 움직여
            # imin 이 프레임마다 바뀌어도 묶는 칸(정수 // 2^l)이 월드에 고정된다 (off 로 민 앙상블 격자는 반올림 원점)
            lo, lat = grid[2], grid[3]
            imin = torch.round(lat.Ai.to(lo.dtype) @ lo).long()
            pos = pos + imin.to(pos.device)
        lv = [dict(src=src, dst=dst, cls=cls, M=int(uniq.numel()))]
        for _ in range(self.levels - 1):
            parent, pos, (cs, cd, cc), cnt = coarsen_grid(pos, lv[-1]["src"], lv[-1]["dst"])
            lv[-1]["parent"] = parent
            lv.append(dict(src=cs, dst=cd, cls=cc, M=int(cnt.numel()), cnt=cnt))
        self.level_stats = [dict(M=d["M"], E=int(d["src"].numel())) for d in lv]
        self._hc = (uniq, src, lv)
        return lv

    def forward(self, feat, src, dst, cls, dt, mat=None, grid=None):
        if grid is None:
            raise RuntimeError("GraphUNetGNN: grid=(uniq, nn[, lo, lat]) 가 필요하다")
        lv = self.hierarchy(src, dst, cls, grid)
        h = self.inp((feat - self.in_mu) / self.in_sd)
        if self.n_mat and mat is not None:
            g, b = self.mfilm(mat).chunk(2, -1)
            h = (1.0 + g) * h + b
        h = self._dtmod(h, dt, 0)
        i, skips, cur = 0, {}, 0
        for (l, kind, n) in self.plan:
            if kind == "dec":                                       # 언풀링: 부모 복사 + 스킵
                up = h[lv[l]["parent"]]
                sk = skips[l]
                h = sk + self.fuse[l](torch.cat([sk, up], -1))
            elif l != cur:                                          # 풀링: 자식 평균
                d = lv[l]
                agg = torch.zeros(d["M"], h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, lv[l - 1]["parent"], h)
                h = agg / d["cnt"].clamp_min(1).unsqueeze(-1).to(h.dtype)
            cur = l
            d = lv[l]
            for _ in range(n):
                h = self.body[i](h, d["src"], d["dst"], d["cls"])
                i += 1
                h = self._dtmod(h, dt, i)
            if kind == "enc":
                skips[l] = h
        sc = (self.scale * (dt / self.dt_ref) if self.dt_scale else self.scale)
        o = self.out(h)
        if self.n_out == 4:
            return (torch.cat([o[:, :3] * sc, o[:, 3:4]], -1),)
        return (o * sc,)
