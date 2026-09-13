"""앵커가 회전과 신축을 상태로 들고, 그것으로 가우시안의 변형구배를 만든다.

지금까지 F 는 앵커 위치장의 미분이었다. 매끄러운 저차원 기저로 위치를 0.1% 까지
맞춰도 그 미분은 뭉개진다 -- 측정하면 어떤 앵커 위치를 골라도 F 잔차가 40% 아래로
내려가지 않고, 실제 형상 매칭은 94% 다.

여기서는 F 를 미분으로 얻지 않는다. 앵커마다 쿼터니언과 로그 스케일을 상태로 두고,
가우시안은 그것들을 각각의 공간에서 블렌딩해 자기 회전과 자기 스케일을 받는다.

    o_g = normalize( sum_a w_ga o_a )        회전은 쿼터니언 공간에서
    s_g = sum_a w_ga s_a                     신축은 로그 공간에서
    F_g = R(o_g) diag(exp(s_g))

행렬을 그대로 선형 블렌딩하면(sum w R S) 이웃의 회전이 어긋날 때 결과가 회전이
아니게 되고 부피가 쭈그러든다 -- 선형 블렌드 스키닝의 오래된 문제다. 쿼터니언
공간에서 섞고 정규화하면 항상 유효한 회전이 나온다.

3DGS 가 가우시안을 (쿼터니언, 스케일) 로 표현하므로, 이것은 물리 상태와 렌더링
표현을 같은 것으로 만드는 일이기도 하다.

한계: F = R diag 는 자유도가 6 이다(회전 3 + 신축 3). 일반적인 F 는 9 이고 극분해의
U 가 대칭 6 자유도이므로, 신축 주축이 몸체에 대해 도는 변형은 담지 못한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def quat_to_R(o):
    """[.,4] (w,x,y,z) -> [.,3,3]. 입력은 정규화돼 있다고 본다."""
    w, x, y, z = o[..., 0], o[..., 1], o[..., 2], o[..., 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _R_to_quat_local(R):
    """회전행렬 -> 쿼터니언 (w,x,y,z). 수치적으로 안전한 분기."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = torch.zeros(*m.shape[:-2], 4, device=m.device, dtype=m.dtype)
    s0 = t > 0
    s = torch.sqrt((t.clamp(min=-0.999999) + 1.0).clamp(min=1e-12)) * 2
    q[..., 0] = 0.25 * s
    q[..., 1] = (m[..., 2, 1] - m[..., 1, 2]) / s.clamp(min=1e-12)
    q[..., 2] = (m[..., 0, 2] - m[..., 2, 0]) / s.clamp(min=1e-12)
    q[..., 3] = (m[..., 1, 0] - m[..., 0, 1]) / s.clamp(min=1e-12)
    q = torch.where(s0.unsqueeze(-1), q, torch.cat(
        [torch.ones_like(t).unsqueeze(-1), torch.zeros_like(m[..., 0, :])], -1))
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def _qconj(a):
    return torch.stack([a[..., 0], -a[..., 1], -a[..., 2], -a[..., 3]], -1)


def _qlog(a):
    """단위 쿼터니언 -> 회전벡터의 절반. |v| 가 0 근처에서 안전하게."""
    v = a[..., 1:]
    n = v.norm(dim=-1, keepdim=True)
    wc = a[..., :1].clamp(-1.0, 1.0)
    th = torch.atan2(n, wc)
    return v * torch.where(n > 1e-8, th / n.clamp(min=1e-8), torch.ones_like(n))


def _qexp(u):
    n = u.norm(dim=-1, keepdim=True)
    sc = torch.where(n > 1e-8, torch.sin(n) / n.clamp(min=1e-8), torch.ones_like(n))
    return torch.cat([torch.cos(n), u * sc], -1)


class AnchorFrame(nn.Module):
    """앵커의 회전·신축 상태와, 그것으로부터의 가우시안 변형구배.

    상태는 self 에 두지 않고 호출자가 실어 나른다 -- 체크포인팅이 순전파와
    재계산에서 같은 텐서를 보아야 하기 때문이다(앞서 이것으로 죽은 적이 있다).
    """

    def __init__(self, dev="cuda"):
        super().__init__()
        self.dev = dev
        # 회전·신축의 관성. 크면 둔하게 반응한다. 학습 대상.
        self.log_Iw = nn.Parameter(torch.zeros((), device=dev))
        self.log_Is = nn.Parameter(torch.zeros((), device=dev))
        # 정지 배치로 되돌리는 복원 강성. 0 이면 자유롭게 떠다닌다.
        self.log_kw = nn.Parameter(torch.zeros((), device=dev))
        self.log_ks = nn.Parameter(torch.zeros((), device=dev))
        self.damping = 1.0

    @staticmethod
    def target(F_mpm):
        """MPM 의 F 를 가우시안별 (쿼터니언, 로그 신축) 목표로 쪼갠다.

        데이터 전처리다 -- F_mpm 은 상수이므로 여기서 미분이 필요 없고, 극분해의
        고유값 간격 나눗셈이 역전파에 들어오지 않는다. 앵커별로 극분해하던
        예전 인코더는 그 미분이 등방 요소에서 발산해 학습이 죽었다.
        """
        from .anchor_fit import closest_rotation
        with torch.no_grad():
            R = closest_rotation(F_mpm, 8, 1e-6)
            U = R.transpose(-1, -2) @ F_mpm
            sv = U.diagonal(dim1=-2, dim2=-1)
            ls = torch.log(torch.nn.functional.softplus(sv * 4.0) / 4.0 + 1e-6)
            return _R_to_quat_local(R), ls

    @staticmethod
    def deconv(og_t, sg_t, w, pair_g, pair_a, M, ridge=1e-4):
        """가우시안별 목표 -> 앵커별 값. 블렌딩의 역합성.

        블렌딩 s_g = sum_a w_ga s_a 는 선형이고 평활화다. 앵커에 국소 변형을 그냥
        재어 넣으면(형상 매칭) 평활화된 값을 다시 평활화하는 셈이라 F 가 뭉개진다
        -- 실측으로 88.8% 대 12%. 그래서 정합이 아니라 최소제곱 역문제를 푼다.

        W 가 파라미터(가중치)에 의존하므로 이 풀이 전체가 미분 가능하다.
        """
        dev = w.device
        # 정규방정식 W^T W a = W^T t 를 앵커 공간에서 조립한다
        idx = pair_a * M + pair_a
        G = torch.zeros(M * M, device=dev)
        # 같은 가우시안을 공유하는 앵커 쌍이 결합을 만든다: 짝-짝 곱을 모은다
        order = torch.argsort(pair_g)
        pg, pa, pw = pair_g[order], pair_a[order], w[order]
        cnt = torch.bincount(pg, minlength=int(pair_g.max()) + 1)
        off = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), cnt.cumsum(0)])
        K = int(cnt.max())
        # [Ng, K] 로 채워 외적을 한 번에
        sel = torch.arange(K, device=dev).unsqueeze(0) + off[:-1].unsqueeze(1)
        valid = torch.arange(K, device=dev).unsqueeze(0) < cnt.unsqueeze(1)
        sa = torch.where(valid, pa[sel.clamp(max=pa.shape[0] - 1)], torch.zeros_like(sel))
        sw = torch.where(valid, pw[sel.clamp(max=pw.shape[0] - 1)], torch.zeros_like(pw[:1]))
        outer = (sw.unsqueeze(-1) * sw.unsqueeze(-2)).reshape(-1)
        pidx = (sa.unsqueeze(-1) * M + sa.unsqueeze(-2)).reshape(-1)
        G = G.index_add_(0, pidx, outer).reshape(M, M)
        G = G + ridge * torch.diagonal(G).mean().detach() * torch.eye(M, device=dev)
        L = torch.linalg.cholesky(G.double())
        out = []
        for t in (og_t, sg_t):
            rhs = torch.zeros(M, t.shape[-1], device=dev).index_add_(
                0, pair_a, w.unsqueeze(-1) * t[pair_g])
            out.append(torch.cholesky_solve(rhs.double(), L).float())
        oa, sa_out = out
        oa = oa / oa.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return oa, sa_out

    def rest(self, M):
        """정지 상태: 항등 회전, 신축 0, 속도 0."""
        o = torch.zeros(M, 4, device=self.dev); o[:, 0] = 1.0
        return (o, torch.zeros(M, 3, device=self.dev),
                torch.zeros(M, 3, device=self.dev), torch.zeros(M, 3, device=self.dev))

    @staticmethod
    def _align(o_pair, pair_g, N, w):
        """대척점 정렬. q 와 -q 는 같은 회전이라, 이웃이 반대 반구에 있으면
        선형 합이 상쇄되어 블렌딩이 무의미해진다. 가우시안마다 기준을 하나
        정하고 나머지를 그 반구로 뒤집는다."""
        ref = torch.zeros(N, 4, device=o_pair.device)
        # 기준: 가중치가 가장 큰 짝의 쿼터니언
        best = torch.zeros(N, device=o_pair.device).index_reduce_(
            0, pair_g, w, "amax", include_self=False)
        pick = (w >= best[pair_g] - 1e-12)
        ref.index_copy_(0, pair_g[pick], o_pair[pick])
        sgn = torch.sign((o_pair * ref[pair_g]).sum(-1, keepdim=True))
        return o_pair * torch.where(sgn == 0, torch.ones_like(sgn), sgn)

    def blend(self, o_a, s_a, w, pair_g, pair_a, N, mode="geo"):
        """앵커의 (o, s) -> 가우시안의 (R, S) -> F. w 는 가우시안마다 합이 1.

        mode="nlerp" 는 선형 합 후 정규화(1 차 근사),
        mode="geo"  는 기준 쿼터니언 둘레의 log/exp 측지 평균이다.
        """
        op = self._align(o_a[pair_a], pair_g, N, w)
        sg = torch.zeros(N, 3, device=o_a.device).index_add_(
            0, pair_g, w.unsqueeze(-1) * s_a[pair_a])
        if mode == "nlerp":
            og = torch.zeros(N, 4, device=o_a.device).index_add_(
                0, pair_g, w.unsqueeze(-1) * op)
            og = og / og.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            # 기준 q0 (정렬에 쓴 것) 둘레에서 log 를 취해 평균 내고 되돌린다
            q0 = torch.zeros(N, 4, device=o_a.device)
            best = torch.zeros(N, device=o_a.device).index_reduce_(
                0, pair_g, w, "amax", include_self=False)
            pick = (w >= best[pair_g] - 1e-12)
            q0.index_copy_(0, pair_g[pick], op[pick])
            q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            d = _qmul(_qconj(q0[pair_g]), op)              # q0^-1 q_a
            u = _qlog(d)                                   # [P,3] 회전벡터/2
            ub = torch.zeros(N, 3, device=o_a.device).index_add_(
                0, pair_g, w.unsqueeze(-1) * u)
            og = _qmul(q0, _qexp(ub))
            og = og / og.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        R = quat_to_R(og)
        return R * sg.exp().unsqueeze(-2), og, sg          # F = R diag(e^s)

    def step(self, o, s, wv, sv, gE_o, gE_s, dt):
        """일반화 힘으로 한 스텝. gE_* 는 dE/d(앵커의 o, s)."""
        tau_o = -gE_o - self.log_kw.exp() * torch.stack(
            [o[:, 0] - 1.0, o[:, 1], o[:, 2], o[:, 3]], -1)
        tau_s = -gE_s - self.log_ks.exp() * s
        wv = (wv + dt * self._quat_rate(o, tau_o) / self.log_Iw.exp()) * self.damping
        sv = (sv + dt * tau_s / self.log_Is.exp()) * self.damping
        ox, oy, oz = wv[:, 0], wv[:, 1], wv[:, 2]
        qw, qx, qy, qz = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        do = 0.5 * torch.stack([-ox * qx - oy * qy - oz * qz,
                                 ox * qw + oy * qz - oz * qy,
                                 -ox * qz + oy * qw + oz * qx,
                                 ox * qy - oy * qx + oz * qw], -1)
        o = o + dt * do
        o = o / o.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        s = s + dt * sv
        return o, s, wv, sv

    @staticmethod
    def _quat_rate(o, tau4):
        """쿼터니언 4 성분의 일반화 힘을 몸체 각속도 3 성분으로 옮긴다."""
        qw, qx, qy, qz = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        tw, tx, ty, tz = tau4[:, 0], tau4[:, 1], tau4[:, 2], tau4[:, 3]
        return 0.5 * torch.stack([
            -tw * qx + tx * qw - ty * qz + tz * qy,
            -tw * qy + tx * qz + ty * qw - tz * qx,
            -tw * qz - tx * qy + ty * qx + tz * qw], -1)


class FrameDynamics(nn.Module):
    """프레임 표현 위의 동역학.

    F 를 앵커 위치의 미분으로 얻지 않고 앵커가 든 (회전, 신축) 에서 조립하므로,
    탄성 에너지가 앵커 위치에 의존하지 않는다 -- 그대로 두면 앵커에 힘이 안 걸린다.
    그래서 결합 항을 둔다:

      E = sum_g V_g Psi(F_g^frame)                      (o, s) 에 힘
        + lam_c sum_a V_a || R_a S_a - F_a^shape(p) ||^2   p 와 (o,s) 를 잇는다

    F_a^shape 는 앵커 그래프의 형상 매칭이다(AnchorStress 의 stencil 재사용).
    lam_c 가 "F 를 위치에서 얼마나 풀어줄 것인가" 의 손잡이가 된다.
    """

    def __init__(self, dev="cuda"):
        super().__init__()
        self.dev = dev
        # 앵커별로 둔다. 스칼라 셋으로는 학습이 바꿀 것이 없어 40 회를 돌려도
        # rollout 이 21.22 -> 21.20 밖에 안 움직였다(실측).
        self.log_Iw = nn.Parameter(torch.zeros(0, device=dev))    # 회전 관성
        self.log_Is = nn.Parameter(torch.zeros(0, device=dev))    # 신축 관성
        self.log_lam = nn.Parameter(torch.zeros(0, device=dev))   # 결합 강성
        self.log_k = nn.Parameter(torch.zeros(0, device=dev))     # 탄성 배율
        self.damping = 1.0
        self.frame = AnchorFrame(dev)

    def relax(self, o, s, F_shape, alpha_o, alpha_s):
        """1 차 이완으로 F_shape 를 따라간다.

        2 차 미분방정식으로 즉시 추종시키려면 유효 강성이 1/dt^2 규모여야 하는데
        (앵커 부피가 1e-5 라 log(lam) - log(I) ~ 28 이 필요하다), 명시적 적분이
        그 강성을 못 버틴다. 1 차 이완은 alpha in (0,1] 이면 언제나 안정하고,
        alpha=1 이 곧 형상 매칭과 동일한 동작이라 기존의 상위집합이 된다.
        alpha<1 은 변형 이력을 담는 누출 적분기가 된다 -- 위치가 표현하지 못하는
        것을 담는다는 이 방향의 요지가 바로 그것이다.
        """
        from .anchor_fit import closest_rotation
        Rt = closest_rotation(F_shape, 8, 1e-6)
        Ut = Rt.transpose(-1, -2) @ F_shape
        st = torch.log(torch.nn.functional.softplus(
            Ut.diagonal(dim1=-2, dim2=-1) * 4.0) / 4.0 + 1e-6)
        st = 3.0 * torch.tanh(st / 3.0)
        ot = _R_to_quat_local(Rt)
        ot = ot * torch.sign((ot * o).sum(-1, keepdim=True)).clamp(min=-1.0).where(
            (ot * o).sum(-1, keepdim=True) != 0, torch.ones_like(ot[:, :1]))
        s = s + alpha_s.unsqueeze(-1) * (st - s)
        o = o + alpha_o.unsqueeze(-1) * (ot - o)
        return o / o.norm(dim=-1, keepdim=True).clamp(min=1e-12), s

    def resize(self, M):
        """밀도 제어로 앵커 수가 바뀌면 다시 잡는다.

        분할·제거로 인덱스 대응이 부분적이라 초기값으로 되돌린다. 밀도 제어
        직후에는 어차피 Adam 의 모멘트도 재시작된다.
        """
        return self.size_to(M, self._lam0, self._I0)

    def size_to(self, M, lam0=0.0, I0=0.0):
        """앵커 수에 맞춰 파라미터를 잡는다. 밀도 제어로 M 이 바뀌면 다시 부른다.

        lam0 을 크게, I0 을 작게 잡으면 A 가 F_shape 를 거의 즉시 따라가 기존
        형상 매칭과 같아진다 -- 프레임이 기존 동작의 상위집합이 되는 출발점이다.
        관성 1, 결합 1 로 두면 응답이 느려 rollout 이 13.13% 대신 21.22% 에서
        시작한다(실측).
        """
        self._lam0, self._I0 = lam0, I0
        z = torch.zeros(M, device=self.dev)
        self.log_Iw = nn.Parameter(z.clone() + I0)
        self.log_Is = nn.Parameter(z.clone() + I0)
        self.log_lam = nn.Parameter(z.clone() + lam0)
        self.log_k = nn.Parameter(z.clone())
        # 이완율. 0 이면 sigmoid(0)=0.5, 큰 양수면 1 에 가까워 형상 매칭과 같다.
        self.a_o = nn.Parameter(z.clone() + 4.0)
        self.a_s = nn.Parameter(z.clone() + 4.0)
        return self

    def forces(self, o, s, w, pair_g, pair_a, N, vol, mu, lam, F_shape, vol_a):
        """(o, s) 에 대한 일반화 힘과 p 에 대한 결합력의 재료를 함께 낸다."""
        from .anchor_fit import closest_rotation, det3, inv3
        F, og, sg = self.frame.blend(o, s, w, pair_g, pair_a, N, "nlerp")
        eye = torch.eye(3, device=F.device)
        R = quat_to_R(og)
        kk = self.log_k.exp()
        J = det3(F)
        n_ = F.reshape(-1, 9).norm(dim=-1).clamp(min=1e-12).reshape(-1, 1, 1)
        FiT = inv3(F + 1e-3 * n_ * eye, eps=1e-30).transpose(-1, -2)
        Js = J / (1.0 + J.abs() / 20.0)                    # 부피항 포화
        P = (2 * mu.reshape(-1, 1, 1) * kk * (F - R)
             + lam.reshape(-1, 1, 1) * kk * (Js - 1).reshape(-1, 1, 1)
             * Js.reshape(-1, 1, 1) * FiT)
        VP = vol.reshape(-1, 1, 1) * P
        # dE/ds_g = diag(R^T P) * e^{s}   (F = R diag(e^s))
        dEs_g = (R.transpose(-1, -2) @ VP).diagonal(dim1=-2, dim2=-1) * sg.exp()
        # 회전: dE/dR = P S^T -> 몸체 토크는 반대칭 부분
        S = sg.exp()
        dER = VP * S.unsqueeze(-2)
        Wm = R.transpose(-1, -2) @ dER
        tau_g = -0.5 * torch.stack([Wm[:, 2, 1] - Wm[:, 1, 2],
                                     Wm[:, 0, 2] - Wm[:, 2, 0],
                                     Wm[:, 1, 0] - Wm[:, 0, 1]], -1)
        M = o.shape[0]
        dEs = torch.zeros(M, 3, device=F.device).index_add_(
            0, pair_a, w.unsqueeze(-1) * dEs_g[pair_g])
        tau = torch.zeros(M, 3, device=F.device).index_add_(
            0, pair_a, w.unsqueeze(-1) * tau_g[pair_g])
        # 결합: A_a = R_a S_a 를 앵커 그래프의 F_a^shape 로 끌어당긴다
        Ra = quat_to_R(o)
        Aa = Ra * s.exp().unsqueeze(-2)
        D = (Aa - F_shape) * (self.log_lam.exp() * vol_a).reshape(-1, 1, 1)
        dEs = dEs + (Ra.transpose(-1, -2) @ D).diagonal(dim1=-2, dim2=-1) * s.exp()
        Wc = Ra.transpose(-1, -2) @ (D * s.exp().unsqueeze(-2))
        tau = tau - 0.5 * torch.stack([Wc[:, 2, 1] - Wc[:, 1, 2],
                                        Wc[:, 0, 2] - Wc[:, 2, 0],
                                        Wc[:, 1, 0] - Wc[:, 0, 1]], -1)
        return dEs, tau, -2.0 * D, F                       # 마지막은 dE/dF_shape

    def step_frame(self, o, s, wv, sv, tau, dEs, dt):
        wv = (wv + dt * tau / self.log_Iw.exp().unsqueeze(-1)) * self.damping
        sv = (sv + dt * (-dEs) / self.log_Is.exp().unsqueeze(-1)) * self.damping
        ox, oy, oz = wv[:, 0], wv[:, 1], wv[:, 2]
        qw, qx, qy, qz = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        do = 0.5 * torch.stack([-ox * qx - oy * qy - oz * qz,
                                 ox * qw + oy * qz - oz * qy,
                                 -ox * qz + oy * qw + oz * qx,
                                 ox * qy - oy * qx + oz * qw], -1)
        o = o + dt * do
        o = o / o.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        # 신축은 tanh 로 유계: e^{-3} ~ e^{3}
        s = 3.0 * torch.tanh((s + dt * sv) / 3.0)
        return o, s, wv, sv
