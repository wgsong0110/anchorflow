"""앵커가 한 프레임의 **변형**을 직접 내놓는 모델.

지금까지의 사슬은 기하(누가 누구에 매이는가)를 한 번 학습해 고정해두고 그 위에서
동역학을 배웠다. 그 고정이 문제였다 -- 짝 목록이 정준 배치에서 정해져 끝까지 가므로,
물체가 갈라져도 떨어져 나간 조각이 반대편 앵커에 계속 매여 있다. GaussianFluent
watermelon 궤적에서 초기 이웃의 13.4%가 영구히 갈라지는데, 그 13.4%를 표현할 수
없는 구조다.

여기서는 그 둘을 뒤집는다.

  기하를 배우지 않는다   앵커는 t=0 에 FPS 로 고르고, 이후 위치는 모델이 낸
                         변위로만 갱신된다. 학습되는 기하 파라미터가 없다.
  동역학을 배우지 않는다 힘도 질량도 적분도 없다. 상태를 받아 **한 프레임의 변형**을
                         곧바로 낸다.

한 프레임은 이렇게 돈다.

  1. 가우시안마다 **현재 위치 기준** 가장 가까운 k 개 앵커를 고른다 (격자 가속).
     소속이 매 프레임 다시 정해지므로 찢어지면 저절로 갈아탄다.
  2. 앵커마다 자기에게 모인 가우시안들을 **질량 가중**으로 집계한다. 스키닝
     가중치는 쓰지 않는다 -- 그것은 모델 출력 r 에 의존해 순환이 된다.
  3. 어텐션이 (앵커 위치 + 집계 특징) 을 받아 앵커마다 (변위 dp, 반경 r, 온도 t) 를 낸다.
  4. phi(x) = x + sum_a w_a(x) dp_a 로 가우시안을 옮긴다. w 는 현재 거리와 r 로
     정해지고 kNN 위 softmax 다.
  5. 같은 사상의 야코비안 J = d(phi)/dx 로 가우시안 모양을 갱신한다.

순환이 없다: w 는 **현재** 위치·앵커·r 만 보고 만드는 것은 **다음** 위치다.

온도에 대한 주의. 온도를 앵커별로 두면 logit_a / t_a = -d^2 / (2 r_a^2 t_a) 가 되어
r_a^2 <- r_a^2 t_a 와 정확히 같아진다 -- 반경과 완전히 중복이다. 그래서 네트워크는
앵커마다 t_a 를 내되, 가우시안의 온도는 이웃들의 t_a 를 **고정** 커널로 섞어 만든다.
그러면 r 은 각 앵커의 도달 범위를, t 는 그 가우시안 배분의 뾰족함을 맡는다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import torch.nn.functional as Fn

from .nextstate import DtFiLM, mlp


# --------------------------------------------------------------- 격자 kNN
def grid_knn(x, p, k, occupancy=2.0, chunk=200_000):
    """가우시안마다 가장 가까운 앵커 k 개. 전체 짝을 훑지 않는다.

    앵커를 균일 격자에 담고 질의는 자기 셀과 26 개 이웃 셀만 본다. 셀 크기는
    셀당 앵커가 평균 `occupancy` 개가 되도록 잡으므로, 27 개 셀에서 나오는 후보가
    k 보다 넉넉하다. 후보가 모자란 가우시안만(경계에 홀로 떨어진 경우) 전체 앵커와
    다시 재는 예비 경로를 탄다 -- 근사가 아니라 정확한 kNN 이다.

    -> (idx [N,k] long, d [N,k])
    """
    N, M = x.shape[0], p.shape[0]
    dev = x.device
    lo = torch.minimum(x.min(0).values, p.min(0).values) - 1e-5
    hi = torch.maximum(x.max(0).values, p.max(0).values) + 1e-5
    span = (hi - lo).clamp(min=1e-6)
    cell = float((float(span.prod()) * occupancy / max(M, 1)) ** (1.0 / 3.0))
    cell = max(cell, 1e-6)
    dims = (span / cell).ceil().long().clamp(min=1)
    D0, D1, D2 = [int(v) for v in dims]

    def flat(c):
        return (c[..., 0] * D1 + c[..., 1]) * D2 + c[..., 2]

    pc = ((p - lo) / cell).long().clamp(torch.zeros(3, dtype=torch.long, device=dev),
                                        dims - 1)
    pf = flat(pc)
    order = torch.argsort(pf)
    pf_s = pf[order]
    n_cell = D0 * D1 * D2
    cnt = torch.bincount(pf_s, minlength=n_cell)
    start = torch.cumsum(cnt, 0) - cnt
    cap = int(cnt.max())

    off = torch.stack(torch.meshgrid(
        *[torch.arange(-1, 2, device=dev)] * 3, indexing="ij"), -1).reshape(-1, 3)
    slot = torch.arange(cap, device=dev)

    idx_out = torch.empty(N, k, dtype=torch.long, device=dev)
    d_out = torch.empty(N, k, device=dev)
    short = []
    for s in range(0, N, chunk):
        xs = x[s:s + chunk]
        c = ((xs - lo) / cell).long().clamp(
            torch.zeros(3, dtype=torch.long, device=dev), dims - 1)
        nb = (c.unsqueeze(1) + off.unsqueeze(0)).clamp(
            torch.zeros(3, dtype=torch.long, device=dev), dims - 1)   # [n,27,3]
        nf = flat(nb)                                                 # [n,27]
        st = start[nf].unsqueeze(-1) + slot                           # [n,27,cap]
        ok = slot < cnt[nf].unsqueeze(-1)
        cand = order[st.clamp(max=M - 1)]                             # [n,27,cap]
        cand = cand.reshape(xs.shape[0], -1)
        ok = ok.reshape(xs.shape[0], -1)
        d = (xs.unsqueeze(1) - p[cand]).norm(dim=-1)
        d = torch.where(ok, d, torch.full_like(d, float("inf")))
        # 같은 앵커가 여러 셀에서 중복으로 잡히지는 않는다 (셀 분할이 배타적)
        kk = min(k, d.shape[1])
        dv, di = d.topk(kk, dim=1, largest=False)
        gi = torch.gather(cand, 1, di)
        if kk < k or not torch.isfinite(dv[:, -1]).all():
            bad = torch.nonzero(~torch.isfinite(dv[:, -1]), as_tuple=False).squeeze(-1) \
                if kk == k else torch.arange(xs.shape[0], device=dev)
            short.append((s, bad))
        if kk < k:
            pad = k - kk
            dv = torch.cat([dv, dv[:, -1:].expand(-1, pad)], 1)
            gi = torch.cat([gi, gi[:, -1:].expand(-1, pad)], 1)
        idx_out[s:s + chunk] = gi
        d_out[s:s + chunk] = dv
    for s, bad in short:
        if bad.numel() == 0:
            continue
        g = bad + s
        dd = torch.cdist(x[g], p)
        dv, di = dd.topk(k, dim=1, largest=False)
        idx_out[g] = di
        d_out[g] = dv
    return idx_out, d_out


# --------------------------------------------------- 상대위치 어텐션
class RelPos(nn.Module):
    """위치에서 나오는 것들을 **한 번만** 만들어 모든 층이 나눠 쓴다.

    두 가지를 낸다.

    회전 각도  RoPE-3D. R(p) 가 직교이고 R(a)R(b)=R(a+b) 이면
               (R(p_i)q_i)^T (R(p_j)k_j) = q_i^T R(p_j-p_i) k_j 가 되어 내적이
               **상대 위치만의 함수**가 된다. 바이어스 행렬 대신 q,k 를 돌리므로
               SDPA 에게는 평범한 어텐션이다. 주파수는 축정렬이 아니라 head 마다
               임의의 3D 벡터로 둔다 (RoPE-Mixed, ECCV 2024 / LieRE, ICML 2025) --
               축정렬은 d_head 를 3 으로 나눠야 하고 대각 방향을 표현하지 못한다.
               파장은 앵커 간격의 두 배부터 물체 지름의 두 배까지 log 등간격이다.
               더 짧으면 이웃 사이에서 위상이 감겨 같은 각도가 다른 거리로 읽힌다.

    게이트 채널  거리 바이어스를 q/k 채널로 분해한 것. softmax 가 행 상수를
               지우므로
                   -(p_i-p_j)^T M (p_i-p_j) = [행 상수] + 2 p_i^T M p_j - p_j^T M p_j
               이고, 오른쪽 두 항은 q 에 [2 M p_i, 1], k 에 [p_j, -p_j^T M p_j] 를
               붙인 내적과 **정확히** 같다. 근사가 아니다. M = A^T A 라 양정치가
               보장되고 head 마다 수용 반경과 방향성이 다르다.

    층마다 따로 두면 각도와 게이트를 depth 번 다시 계산한다. 위치는 층 사이에
    변하지 않는데 M=512 에서는 연산이 작아 **커널 실행 횟수**가 시간을 정하므로,
    그 중복이 그대로 손해다 (실측: 층별 6.9 -> 17.9 ms).
    """

    def __init__(self, heads, dc, lam_min, lam_max, sigma0, seed=0):
        super().__init__()
        assert dc % 2 == 0, "RoPE 는 채널을 쌍으로 쓴다"
        P = dc // 2
        g = torch.Generator().manual_seed(seed)
        mag = 2.0 * math.pi / torch.logspace(math.log10(lam_min),
                                             math.log10(lam_max), P)
        d = torch.randn(heads, P, 3, generator=g)
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        self.omega = nn.Parameter(d * mag.reshape(1, P, 1))
        self.A = nn.Parameter(torch.eye(3).repeat(heads, 1, 1) / sigma0)

    def forward(self, pos):
        """pos [B,M,3] -> (cos, sin, q_geo, k_geo)"""
        ang = torch.einsum("bmc,hpc->bhmp", pos, self.omega)
        Mh = self.A.transpose(-1, -2) @ self.A                  # [H,3,3] 양정치
        Mp = torch.einsum("hcd,bmd->bhmc", Mh, pos)             # [B,H,M,3]
        quad = (pos.unsqueeze(1) * Mp).sum(-1, keepdim=True)    # p^T M p
        qg = torch.cat([2.0 * Mp, torch.ones_like(quad)], -1)
        kg = torch.cat([pos.unsqueeze(1).expand_as(Mp), -quad], -1)
        return ang.cos(), ang.sin(), qg, kg


def _rope(x, c, s):
    """채널을 앞뒤 절반으로 짝지어 돌린다 (GPT-NeoX 식).

    이웃한 두 채널을 짝짓는 원래 형태와 수학적으로 같다 -- 채널 순열일 뿐이고
    그 순열은 앞의 학습되는 qkv 선형이 흡수한다. 대신 슬라이스가 연속이라
    스트라이드 접근과 stack/flatten 이 사라진다. M=512 에서는 연산이 작아
    커널 실행 횟수가 시간을 정하므로 이것이 그대로 이득이다.
    """
    x1, x2 = x.chunk(2, -1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)


class RelAttention(nn.Module):
    """상대 위치가 q/k 안에 들어가는 어텐션. [M,M] 바이어스를 만들지 않는다.

    전에 쓰던 경로는 쌍마다 (p_i-p_j, log(1+d)) 를 작은 MLP 에 통과시켜 head 별
    바이어스를 만들었다. 파라미터가 292 개인 층이 학습 반복의 35% 를 먹었는데,
    이유는 **[M,M,32] 중간 텐서를 만들고 역전파용으로 붙잡고 있기** 때문이다
    (Point Transformer V3 도 같은 것을 재고 RPE 를 아예 버렸다).

    기하 채널은 head 차원 **안에서** 뗀다. 덧붙이면 32 -> 36 이 되어 flash 커널의
    지원 차원을 벗어난다.
    """

    EXTRA = 4
    # SDPA 로 부를지, q@k^T 를 그대로 만들지. 바이어스를 없앴으므로 어텐션 행렬은
    # [B,H,M,M] 하나뿐이고 M=512 에서 4MB 다 -- 피해야 했던 것은 [M,M,32] 쪽이었다.
    # 그 크기에서는 fp32 SDPA(= flash 불가, mem-efficient 폴백) 보다 평범한 GEMM 이
    # 빠를 수 있어 둘을 모두 둔다.
    USE_SDPA = False

    def __init__(self, hidden, heads):
        super().__init__()
        assert hidden % heads == 0
        self.h, self.d = heads, hidden // heads
        self.dc = self.d - self.EXTRA
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        # scale 은 **내용 채널 기준**이다. 기본값 d^-1/2 를 쓰면 내용 항이 잘못
        # 스케일된다. 기하 항 크기는 학습되는 A 가 흡수한다.
        self.scale = self.dc ** -0.5

    def forward(self, x, ctx):
        c, s, qg, kg = ctx
        B, M, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = (t.view(B, M, self.h, self.d).transpose(1, 2) for t in (q, k, v))
        # q 와 k 를 한 덩어리로 묶어 회전과 이어붙이기를 한 번씩만 한다
        qk = torch.stack([q[..., :self.dc], k[..., :self.dc]], 0)
        qk = torch.cat([_rope(qk, c, s), torch.stack([qg, kg], 0)], -1)
        q, k = qk[0], qk[1]
        if self.USE_SDPA:
            o = Fn.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                scale=self.scale)
        else:
            o = ((q @ k.transpose(-1, -2)) * self.scale).softmax(-1) @ v
        return self.proj(o.transpose(1, 2).reshape(B, M, -1))


class RelBlock(nn.Module):
    """pre-norm 블록. 어텐션만 위 것으로 바뀌었다."""

    def __init__(self, hidden, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.att = RelAttention(hidden, heads)
        self.ffn = mlp([hidden, 2 * hidden, hidden], layernorm=False)

    def forward(self, x, ctx):
        x = x + self.att(self.n1(x), ctx)
        return x + self.ffn(self.n2(x))


# --------------------------------------------------------------- 집계
def aggregate(x, v, X, m, idx, M, h, pa=None, Fg=None):
    """앵커별로 자기에게 모인 가우시안들을 질량 가중으로 요약한다.

    가중치는 **질량뿐**이다. 스키닝 가중치를 쓰면 그것이 모델 출력 r 의 함수라
    순환이 된다.

    x  [N,3] 현재 위치     v [N,3] 현재 속도      X [N,3] 정준 위치
    m  [N]   질량          idx [N,k] 각 가우시안이 고른 앵커
    h  스칼라. 길이 정규화에 쓰는 앵커 간격.
    pa [M,3] 앵커의 현재 위치. 소속 가우시안들의 질량가중 위치를 앵커 자기 위치
             기준으로 넣기 위해 받는다 -- 절대 위치는 네트워크 입력의 p 와
             중복이고, 상대값이라야 "내가 쥔 재질의 무게중심이 나로부터 어느
             쪽으로 얼마나 밀렸나" 가 된다. 찢어짐이 여기서 먼저 보인다.
    Fg [N,3,3] 가우시안이 들고 있는 변형구배 (있으면 함께 요약)

    -> [M, C] 특징
    """
    N, k = idx.shape
    dev = x.device
    a = idx.reshape(-1)                                  # [N*k]
    w = m.unsqueeze(1).expand(N, k).reshape(-1)          # 질량만

    def sca(val):                                        # [N*k, D] -> [M, D]
        D = val.shape[-1]
        out = torch.zeros(M, D, device=dev, dtype=val.dtype)
        return out.index_add_(0, a, val)

    Wa = sca(w.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12)      # [M] 질량 합
    rep = lambda t: t.unsqueeze(1).expand(N, k, t.shape[-1]).reshape(N * k, -1)
    xr, vr, Xr = rep(x), rep(v), rep(X)
    cx = sca(w.unsqueeze(-1) * xr) / Wa.unsqueeze(-1)           # [M,3] 질량중심
    cX = sca(w.unsqueeze(-1) * Xr) / Wa.unsqueeze(-1)
    cv = sca(w.unsqueeze(-1) * vr) / Wa.unsqueeze(-1)           # [M,3] 평균 속도

    dx = xr - cx[a]
    dX = Xr - cX[a]
    dv = vr - cv[a]
    # 2차 모멘트: 이 앵커 주변이 얼마나, 어느 방향으로 퍼져 있나
    S = sca((w.reshape(-1, 1, 1) * (dx.unsqueeze(-1) * dx.unsqueeze(-2))
             ).reshape(N * k, 9)).reshape(M, 3, 3) / Wa.reshape(M, 1, 1)
    iu = torch.triu_indices(3, 3, device=dev)
    S6 = S[:, iu[0], iu[1]] / (h * h)
    # 각운동량 -> 각속도. 강체처럼 돌고 있는 성분을 분리해 준다
    L = sca(w.unsqueeze(-1) * torch.cross(dx, dv, dim=-1))      # [M,3]
    tr = S.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(M, 1, 1)
    I = (tr * torch.eye(3, device=dev) - S) * Wa.reshape(M, 1, 1)
    om = torch.linalg.solve(I + 1e-8 * torch.eye(3, device=dev), L.unsqueeze(-1)
                            ).squeeze(-1)
    # 국소 변형구배: 정준 배치 대비 얼마나 찌그러졌나 (최소제곱)
    A = sca((w.reshape(-1, 1, 1) * (dx.unsqueeze(-1) * dX.unsqueeze(-2))
             ).reshape(N * k, 9)).reshape(M, 3, 3)
    B = sca((w.reshape(-1, 1, 1) * (dX.unsqueeze(-1) * dX.unsqueeze(-2))
             ).reshape(N * k, 9)).reshape(M, 3, 3)
    Fa = A @ torch.linalg.inv(B + (1e-6 * h * h) * torch.eye(3, device=dev))
    detF = torch.linalg.det(Fa).reshape(M, 1)

    feats = [
        torch.log(Wa).reshape(M, 1),                    # 질량
        torch.log1p(sca(torch.ones_like(w).unsqueeze(-1))),   # 개수
        (cx - cX) / h,                                  # 정준 대비 이동
        ((cx - pa) / h if pa is not None
         else torch.zeros_like(cx)),                    # 질량가중 위치 (앵커 기준)
        cv,                                             # 평균 속도 (호출자가 정규화)
        S6,                                             # 퍼진 모양
        om,                                             # 회전 속도
        Fa.reshape(M, 9),                               # 찌그러진 정도
        torch.sign(detF) * torch.log(detF.abs().clamp(min=1e-6)),
    ]
    if Fg is not None:
        Fr = Fg.unsqueeze(1).expand(N, k, 3, 3).reshape(N * k, 9)
        feats.append(sca(w.unsqueeze(-1) * Fr) / Wa.unsqueeze(-1))
    return torch.cat(feats, -1), cx


# --------------------------------------------------------------- 스키닝
def skin(x, p, dp, log_r, log_t, idx, h):
    """phi(x) = x + sum_a w_a(x) dp_a. w 는 kNN 위 softmax.

    logit_a = -d_a^2 / (2 r_a^2),  온도는 가우시안별로 이웃의 t_a 를 **고정**
    커널로 섞어 만든다 -- 앵커별 온도를 그대로 쓰면 반경과 중복이기 때문이다.
    """
    pa = p[idx]                                    # [N,k,3]
    d2 = ((x.unsqueeze(1) - pa) ** 2).sum(-1)      # [N,k]
    r2 = (2.0 * log_r[idx]).exp().clamp(min=1e-12)
    logit = -0.5 * d2 / r2
    # 섞는 커널은 고정 폭 h 다. 학습되는 양에 의존하면 온도가 다시 반경과 얽힌다.
    u = torch.softmax(-0.5 * d2 / (h * h), dim=1)
    tau = (u * log_t[idx].exp()).sum(1, keepdim=True).clamp(min=1e-4)
    w = torch.softmax(logit / tau, dim=1)          # [N,k]
    return x + (w.unsqueeze(-1) * dp[idx]).sum(1), w


def jacobian_of(fn, x):
    """pointwise 사상의 야코비안 [N,3,3]. 역전파 세 번이면 된다.

    x 를 떼어낸 잎으로 두는 것이 중요하다: 앵커 출력은 **모든** 가우시안의 집계에
    의존하므로, 붙어 있는 x 로 미분하면 다른 가우시안을 거쳐 오는 항까지 섞여
    "이 점에서의 변형 사상의 야코비안" 이 아니게 된다. 앵커 출력 쪽 그래프는
    그대로 살아 있어 J 에 건 손실이 네트워크로 흐른다.
    """
    xd = x.detach().requires_grad_(True)
    y = fn(xd)
    rows = []
    for i in range(3):
        g, = torch.autograd.grad(y[:, i].sum(), xd, create_graph=True)
        rows.append(g)
    return torch.stack(rows, 1)                    # [N,3,3]


# --------------------------------------------------------------- 모델
class DeformNet(nn.Module):
    """앵커 상태 -> (변위, 반경, 온도). 구조는 학생 스테퍼와 같은 어텐션이다.

    국소 연산자로는 수용영역이 모자란다 -- 앵커 하나를 흔들었을 때 k=8, depth 4 로
    512 개 중 29 개만 움직였다. 찢어짐은 경계를 따라 멀리 전파되므로 한 층에서
    모든 앵커가 서로를 보는 편이 맞고, M=512 면 전체 맵이 4MB 다.
    """

    def __init__(self, n_feat, hidden=128, depth=4, heads=4, n_static=0,
                 scale=1.0, h=1.0, ext=1.0, zero_init=True, seed=0):
        super().__init__()
        self.scale = scale                 # 변위 단위 (전형적 한 프레임 변위)
        self.h = h                         # 길이 단위 (앵커 간격)
        self.n_static = n_static
        # 앵커의 **절대** 위치는 넣지 않는다. 어텐션 바이어스가 이미 상대 오프셋과
        # 거리만 보고, 집계 특징도 전부 상대량이라, 여기에 p 를 넣는 순간 연산자가
        # 평행이동 등변성을 잃는다 -- 물체가 1 만큼 옮겨간 같은 파괴를 다른 입력으로
        # 보게 된다. 경계까지의 거리처럼 위치가 필요한 정보는 static 채널이
        # 경계 기준 상대량으로 이미 들고 있다.
        self.node_enc = mlp([n_feat + n_static, hidden, hidden])
        self.film = DtFiLM(hidden, depth + 1)
        # 파장 범위: 가장 짧은 것은 앵커 간격의 두 배 (그보다 짧으면 이웃 사이에서
        # 위상이 감긴다), 가장 긴 것은 물체 지름의 두 배.
        self.relpos = RelPos(heads, hidden // heads - RelAttention.EXTRA,
                             lam_min=2.0 * h, lam_max=2.0 * ext, sigma0=h,
                             seed=seed)
        self.blocks = nn.ModuleList([RelBlock(hidden, heads)
                                     for _ in range(depth)])
        self.dec = mlp([hidden, hidden, 5], layernorm=False)
        if zero_init:
            # 출력이 거의 0 이면 dp~0, r~h, tau~1 -- 아무것도 움직이지 않는 항등
            # 변형에서 시작한다. 학습이 "가만히 있기" 를 먼저 배울 필요가 없다.
            #
            # 다만 가중치를 **정확히** 0 으로 두면 안 된다. 그 층의 입력 쪽
            # 기울기가 grad_out @ W = 0 이라, 첫 스텝에 상류 전체(인코더, 어텐션
            # 블록, RoPE 주파수, 거리 게이트)가 기울기를 하나도 못 받는다. 실측:
            # 64 개 파라미터 중 62 개가 정확히 0. 편향만 0 으로 두고 가중치는
            # 기본 초기화의 1/100 로 줄이면 출력은 여전히 항등에 가깝고 경로는
            # 살아 있다.
            last = [m for m in self.dec.modules() if isinstance(m, nn.Linear)][-1]
            with torch.no_grad():
                last.weight.mul_(0.01)
            nn.init.zeros_(last.bias)

    def forward(self, p, feat, dt, static=None):
        """p [M,3], feat [M,F] -> (dp [M,3], log_r [M], log_tau [M])

        p 는 어텐션 바이어스에만 쓰이고 -- 거기서도 쌍의 상대 오프셋과 거리로만
        들어간다 -- 노드 특징으로는 들어가지 않는다.
        """
        f = [feat]
        if static is not None:
            f.append(static)
        h = self.node_enc(torch.cat(f, -1)).unsqueeze(0)
        gamma, beta = self.film(dt, p.device)
        h = gamma[0] * h + beta[0]
        # 좌표는 앵커 무게중심 기준으로 옮겨 쓴다. 평행이동 불변은 그대로이고
        # (무게중심이 함께 움직인다), 거리 게이트가 |p|^2 ~ 300 짜리 두 항의
        # 차이로 |Δ|^2 ~ 1 을 만드는 자리끼리 상쇄를 피한다.
        pc = (p - p.mean(0, keepdim=True)).unsqueeze(0)
        ctx = self.relpos(pc)
        for i, blk in enumerate(self.blocks):
            h = blk(gamma[i + 1] * h + beta[i + 1], ctx)
        o = self.dec(h).squeeze(0)
        dp = o[:, :3] * self.scale
        log_r = o[:, 3] + math.log(self.h)
        log_t = o[:, 4]
        return dp, log_r.clamp(math.log(self.h) - 3.0, math.log(self.h) + 3.0), \
            log_t.clamp(-4.0, 4.0)


def bc_features(p, cfg):
    """경계 조건을 앵커마다 읽는다. 조건 자체는 안 변하지만 앵커가 움직이므로
    값은 매 프레임 다시 계산한다."""
    dev = p.device
    out = []
    for bc in cfg.get("boundary_conditions", []):
        t = bc["type"]
        if t == "surface_collider":
            n = torch.tensor(bc["normal"], device=dev, dtype=p.dtype)
            n = n / n.norm().clamp(min=1e-12)
            q = torch.tensor(bc["point"], device=dev, dtype=p.dtype)
            out.append(((p - q) * n).sum(-1, keepdim=True))       # 부호 있는 거리
            out.append(n.reshape(1, 3).expand(p.shape[0], 3))
        elif t in ("cuboid", "enforce_particle_translation"):
            q = torch.tensor(bc["point"], device=dev, dtype=p.dtype)
            s = torch.tensor(bc["size"], device=dev, dtype=p.dtype)
            out.append(((p - q).abs() <= s).all(-1, keepdim=True).to(p.dtype))
    if not out:
        return torch.zeros(p.shape[0], 0, device=dev, dtype=p.dtype)
    return torch.cat(out, -1)
