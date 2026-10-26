"""텐서곱 B-스플라인 FFD (--interp bspline2 / bspline3, 사용자 10-26).

노드 = 우리 Kuhn 격자의 정육면체 꼭짓점 (기존 노드 그대로). 격자 기저 A 의 열은 서로 직교하고 길이가
Δ = node_h/√2 인 정육면체 격자다 (simplex.lattice: a_i·a_j = h²/3·(-1/2) + h²/6 = 0, |a_i|² = h²/2).
격자 좌표 y = A⁻¹(x - lo) 에서 축별 1 차원 B-스플라인의 곱 N_i(x) 로

    x' = x + Σ_i N_i(x) U_i,        F_step = I + Σ_i U_i ⊗ ∇N_i(x),        F' = F_step F

  bspline2: 2 차 (C¹, 27 노드, MPM 과 같은 가중치 -- simplex.bspline27 그대로)
  bspline3: 3 차 (C², 64 노드)

두 판 모두 분할의 합이 1 이고 1 차 다항식을 재현한다 (U 가 아핀이면 x' 도 아핀). U = 0 이면 항등.

det > 0 보장:
  (A) bspline2: 입자 det F_step 에 기존 --cell_barrier 로그 장벽 (phys_resid._cell_pen) + 점유 다항식 조각마다
      표본점 8 개 (조각 국소 좌표 {1/4, 3/4}³) 에 같은 장벽 (조각 부피 Δ³ 를 8 등분한 가중).
  (B) bspline3: 이웃 노드(격자축 방향) 변위 차의 상한. 3 차 B-스플라인의 축 미분은 그 축 전진 차분의 볼록 결합이다
          ∂u/∂y_a = Σ B2·B3·B3 (U_{i+e_a} - U_i)
      그래서 모든 축 이웃 쌍에서 |U_{i+e_a} - U_i| ≤ cΔ 이면 ∂u/∂y 의 각 열 노름 ≤ cΔ,
      ‖∇_x u‖₂ = ‖∂u/∂y‖₂/Δ ≤ ‖∂u/∂y‖_F/Δ ≤ √3 c.  c < 1/√3 (≈0.577) 이면 ‖∇u‖₂ < 1 이라
      det F_step ≥ (1 - √3 c)³ > 0 이 지지 노드 64 개가 모두 활성인 **모든 점**에서 성립하고, 그 영역의 볼록 부분집합
      위에서 u 가 축약(Lipschitz < 1)이라 x + u 가 단사다. 기본 c = 0.48 (사용자 제안 값; det ≥ (1-0.831)³ = 0.0048).
      (참고: Choi & Lee 2000 의 3D 충분조건은 제어점 **변위 자체** |Δc| < Δ/K, K≈2.48 -> 0.4033Δ 이고, 0.48 은 2D 값이다.
       변위 자체 조건은 평행이동에도 걸리므로 여기서는 평행이동에 불변인 차분 조건을 쓴다.)
      사상: m = τ·logsumexp(|D_e|/(Δτ)) (≥ max_e |D_e|/Δ, 매끄러운 최대), s = (1 + (m/c)^k)^(-1/k),
            U = ū + s (V - ū)   ->  |D_e(U)| = s|D_e(V)| ≤ s·m·Δ < cΔ  (정확히, 모든 쌍).
      m ≪ c 이면 s ≈ 1 - (m/c)^k/k (k=8 이라 m = c/2 에서 1 - 5e-4) -- 거의 항등, 노드 변위 0 이면 정확히 항등.
      ū = 손잡이(구속) 노드 명령 변위의 평균 (없으면 전체 평균) -- 평행이동은 줄이지 않는다. 구속 노드는 사상 뒤 다시 박는다
      (명령이 한결같은 평행이동이면 다시 박아도 값이 같아 보장이 그대로다; ficus 화분은 0).
"""
import math

import torch

from . import simplex as SX

CAP_C = 0.48          # (B) 축 이웃 변위 차 상한 / Δ  (< 1/√3)
CAP_K = 8             # 포화 지수
CAP_TAU = 0.01        # 매끄러운 최대의 온도 (Δ 단위)
N_SMP_AX = 2          # 조각당 축별 표본 수 -> 8 개
_O64 = {}


def _o64(device):
    if device not in _O64:
        _O64[device] = torch.tensor([[i, j, k] for i in range(4) for j in range(4) for k in range(4)],
                                    device=device, dtype=torch.long)
    return _O64[device]


def bspline64(q, lo, lat, nn):
    """3 차 균일 B-스플라인 (격자 좌표 y 에서 축별 곱, 64 노드).

    base = floor(y) - 1, f = y - floor(y) ∈ [0,1),
    w = [(1-f)³, 3f³-6f²+4, -3f³+3f²+3f+1, f³]/6,  ∂w/∂f = [-(1-f)², 3f²-4f, -3f²+2f+1, f²]/2.
    -> (flat [N,64], w [N,64], dwdy [N,64,3])
    """
    nnl = [int(nn[k]) for k in range(3)]
    y = (q - lo) @ lat.Ai.T
    fl = torch.floor(y)
    f = y - fl
    g = 1.0 - f
    wq = torch.stack([g ** 3, 3 * f ** 3 - 6 * f ** 2 + 4, -3 * f ** 3 + 3 * f ** 2 + 3 * f + 1, f ** 3], 1) / 6.0
    dq = torch.stack([-g ** 2, 3 * f ** 2 - 4 * f, -3 * f ** 2 + 2 * f + 1, f ** 2], 1) / 2.0     # [N,4(o),3(축)]
    o = _o64(q.device)
    w0, w1, w2 = wq[:, o[:, 0], 0], wq[:, o[:, 1], 1], wq[:, o[:, 2], 2]
    d0, d1, d2 = dq[:, o[:, 0], 0], dq[:, o[:, 1], 1], dq[:, o[:, 2], 2]
    w = w0 * w1 * w2
    dwdy = torch.stack([d0 * w1 * w2, w0 * d1 * w2, w0 * w1 * d2], -1)
    ijk = (fl.long() - 1).unsqueeze(1) + o.unsqueeze(0)
    lim = torch.tensor(nnl, device=q.device)
    if bool(((ijk < 0) | (ijk >= lim)).any()):
        raise SystemExit("[bspline64] 64 이웃 노드가 격자 밖이다 -- 격자 여백이 모자라다")
    flat = (ijk[..., 0] * nnl[1] + ijk[..., 1]) * nnl[2] + ijk[..., 2]
    return flat, w, dwdy


def weights(q, lo, lat, nn, order):
    return SX.bspline27(q, lo, lat, nn) if int(order) == 2 else bspline64(q, lo, lat, nn)


def piece_of(y, order):
    """다항식 조각 정수 좌표: 2 차는 노드 중심 쌍대 셀 floor(y-1/2), 3 차는 격자 셀 floor(y)."""
    return torch.floor(y - 0.5) if int(order) == 2 else torch.floor(y)


def samples(x, lo, lat, nn, order, uniq):
    """점유 조각마다 표본점 N_SMP_AX³ 개 -> dict(rows [S,J], dw [S,J,3], w [S], n_piece, q [S,3]).

    조각 = 입자가 하나라도 든 다항식 조각 (그 지지 노드는 모두 활성이다 -- 같은 조각 입자의 지지와 같다).
    가중 w = 조각 부피 / 표본 수.
    """
    y = (x.detach() - lo) @ lat.Ai.T
    pc = torch.unique(piece_of(y, order).long(), dim=0)                         # [P,3]
    sh = 0.5 if int(order) == 2 else 0.0
    t = (torch.arange(N_SMP_AX, device=x.device, dtype=x.dtype) + 0.5) / N_SMP_AX   # 1/4, 3/4
    g = torch.stack(torch.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)   # [8,3]
    ys = (pc.to(x.dtype).unsqueeze(1) + sh + g.unsqueeze(0)).reshape(-1, 3)
    qs = ys @ lat.A.T.to(x.dtype) + lo
    fl, _w, dw = weights(qs, lo, lat, nn, order)
    r = torch.searchsorted(uniq, fl).clamp(max=uniq.numel() - 1)
    if not bool((uniq[r] == fl).all()):
        raise SystemExit("[bspfd] 표본점 지지 노드가 활성 노드 밖이다")
    vol = abs(float(torch.linalg.det(lat.A.double())))
    ws = torch.full((qs.shape[0],), vol / g.shape[0], device=x.device, dtype=x.dtype)
    return dict(rows=r, dw=dw.to(x.dtype), w=ws, n_piece=int(pc.shape[0]), q=qs)


def axis_pairs(uniq, nn):
    """활성 노드 중 격자축 방향 이웃 쌍 (src, dst) -- 세 축 모두 (+e_a 방향 하나씩)."""
    nnl = [int(nn[k]) for k in range(3)]
    k2 = uniq % nnl[2]
    k1 = (uniq // nnl[2]) % nnl[1]
    k0 = uniq // (nnl[1] * nnl[2])
    ijk = torch.stack([k0, k1, k2], -1)
    src, dst = [], []
    for ax, st in enumerate((nnl[1] * nnl[2], nnl[2], 1)):
        ok = ijk[:, ax] + 1 < nnl[ax]
        tgt = uniq + st
        p = torch.searchsorted(uniq, tgt).clamp(max=uniq.numel() - 1)
        hit = ok & (uniq[p] == tgt)
        src.append(torch.nonzero(hit).squeeze(1)); dst.append(p[hit])
    return torch.cat(src), torch.cat(dst)


def cap_map(dpn, pairs, delta, pin=None, c=CAP_C, k=CAP_K, tau=CAP_TAU):
    """(B) 노드 변위 V (dpn[:, :3]) -> U = ū + s(V-ū), 모든 축 이웃 쌍 |ΔU| < cΔ. -> (dpn', 통계 텐서 [4])

    통계 = (참 최대 |ΔV|/Δ, 매끄러운 최대 m, s, 사상 뒤 최대 |ΔU|/Δ).
    """
    src, dst = pairs
    V = dpn[:, :3]
    if src.numel() == 0:
        z = torch.zeros(4, device=V.device, dtype=V.dtype)
        return dpn, z
    D = V[dst] - V[src]
    n = torch.sqrt((D * D).sum(-1) + 1e-24) / float(delta)
    m = tau * torch.logsumexp(n / tau, 0)
    s = (1.0 + (m / c) ** k) ** (-1.0 / k)
    if pin is not None and bool(pin.any()):
        ref = V[pin].mean(0)
    else:
        ref = V.mean(0)
    U = ref + s * (V - ref)
    if pin is not None:
        U = torch.where(pin.unsqueeze(-1), V, U)
    st = torch.stack([n.max().detach(), m.detach(), s.detach(),
                      ((U[dst] - U[src]).norm(dim=-1).max() / float(delta)).detach()])
    out = U if dpn.shape[-1] == 3 else torch.cat([U, dpn[:, 3:]], -1)
    return out, st


def smp_det(smp, lat, dpn):
    """표본점 det F_step [S]."""
    J = SX.bspline_jac_pre(smp["rows"], smp["dw"], lat, dpn)
    return torch.linalg.det(torch.eye(3, device=J.device, dtype=J.dtype) + J)
