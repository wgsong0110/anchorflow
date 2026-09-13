"""앵커 그래프 위에서 직접 계산하는 연속체 응력.

지금까지는 가우시안 17 만 개를 적분점으로 써서 변형구배를 만들었다. 여기서는
앵커가 자기 이웃 앵커만 보고 같은 일을 한다 -- 적분점이 512 개로 줄고, 짝이
350 만에서 8 천으로 줄고, 가우시안은 힘 계산에 전혀 참여하지 않는다.

중심력(O_only)이 실패한 이유가 여기서 해결된다. 쌍을 잇는 방향의 힘만으로는
거리를 바꾸지 않는 전단 변형에 저항할 수 없어 에너지가 0 인 모드가 남고,
실제로 400 회 중 319 회가 발산했다. 변형구배를 거치면 그 모드에도 응력이 생긴다.

  F_a = [ Σ_b w_ab (p_b − p_a) q_ab^T ] B_a^{-1},   B_a = Σ_b w_ab q_ab q_ab^T
  P_a = 2μ_a (F_a − R_a) + λ_a (J−1) J F_a^{-T}
  f_b = −V_a w_ab P_a B_a^{-1} q_ab,   f_a = +Σ_b (같은 것)

힘이 에너지 Σ_a V_a Ψ(F_a) 의 그래디언트라 운동량은 구조적으로 보존된다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree


class AnchorStress(nn.Module):
    def __init__(self, k=16, eig_floor=0.02, dev="cuda"):
        super().__init__()
        self.k = k
        self.eig_floor = eig_floor
        self.dev = dev
        # 힘 법칙은 자기 파라미터를 가진다. 스키닝의 log_s / quat 은 가우시안을
        # 앵커에 붙이는 지지 반경이고, 여기 log_h 는 앵커가 이웃 앵커를 보는
        # 스텐실 폭이다. 하는 일이 다르므로 공유하면 한 텐서에 상충하는
        # 그래디언트가 겹친다.
        self.polar_iters = 8
        # 경계는 클램프가 아니라 tanh 로 준다. log_ka 가 커지면 exp 가 넘쳐
        # 응력이 곧바로 무한이 되는데(추적이 "F 는 유한한데 P 가 서브스텝 2 에서
        # 무한" 이라고 찍었다), 하드 클램프로 막으면 경계에서 그래디언트가 정확히
        # 0 이 되어 그 앵커가 죽고 Adam 의 모멘텀만 매 스텝 낭비된다.
        #
        #   k_a = exp( L tanh(u/L) )      u ~ 0 에서는 exp(u) 와 같고
        #                                 |u| 가 커지면 e^{±L} 로 매끄럽게 수렴
        #
        # 파라미터 자체는 무한대까지 자유롭고, 물리량만 유계다.
        self.ka_lim = 2.0                 # 강성 배율 e^{-2} ~ e^{2}
        self.h_lim = float(np.log(3.0))   # 스텐실 폭 h0/3 ~ 3 h0
        # 부피항 lam (J-1) J F^-T 는 J 에 대해 이차라, B 가 한 방향으로 랭크를
        # 잃으면 J = det(A)/det(B) 가 1e13 까지 뛰고 응력이 1e28 이 된다.
        # 대각합도 B^-1 의 최대원소도 멀쩡한 채로 행렬식만 무너지므로 그 둘로는
        # 못 잡는다. J 를 부드럽게 포화시켜 응력을 유계로 만든다.
        self.J_max = 20.0
        # 정지 스텐실은 pos 와 log_h 에만 의존하므로 롤아웃 내내 상수다. 예전에는
        # 서브스텝마다(코스 프레임당 40 회) 다시 만들었는데, 앵커 512 개짜리
        # 텐서에 커널을 수백 개 던지는 것이 이 경로의 비용 전부라 그만큼 낭비였다.
        self._rest = None
        # NaN 추적을 위임할 시뮬레이터. 그냥 대입하면 nn.Module 로 등록되어
        # 그쪽 파라미터가 이 모듈의 parameters() 에도 딸려 들어오고, 옵티마이저가
        # "some parameters appear in more than one parameter group" 로 죽는다.
        # 리스트에 담아 등록을 피한다.
        self._owner = []
        self.log_h = nn.Parameter(torch.zeros(0, device=dev))
        self.log_ka = nn.Parameter(torch.zeros(0, device=dev))
        for n in ("ea", "eb"):
            self.register_buffer(n, torch.zeros(0, dtype=torch.long, device=dev))
        for n in ("vol", "mu", "lam", "h0"):
            self.register_buffer(n, torch.zeros(0, device=dev))
        # B 와 함께 사라지지 않는 릿지의 바닥. 상대 릿지만 쓰면 B 가 통째로
        # 작아질 때 릿지도 같이 작아져 정작 필요할 때 없어지고, B^-1 이 폭주해
        # F 는 유한한데 J^2 F^-T 가 float 를 넘긴다 -- 추적이 "F 는 유한, P 는
        # 서브스텝 2 에서 무한" 이라고 찍은 것이 그 모양이었다. 가우시안 경로는
        # 이미 같은 이유로 B_ref 를 두고 있다.
        self.register_buffer("B_ref", torch.zeros((), device=dev))

    @torch.no_grad()
    def rebuild(self, fit):
        """정지 배치에서 이웃과 그 정지 상수들을 만든다. 앵커 재질은 그 앵커가
        쥐고 있는 가우시안의 부피가중 평균이다."""
        pos = fit.pos.detach()
        m = pos.shape[0]
        k = min(self.k + 1, m)
        _, nb = cKDTree(pos.cpu().numpy()).query(pos.cpu().numpy(), k=k)
        ea = torch.from_numpy(np.repeat(np.arange(m), k - 1)).to(self.dev)
        eb = torch.from_numpy(nb[:, 1:].reshape(-1).astype(np.int64)).to(self.dev)
        # 스텐실 폭은 이웃까지의 실제 간격에서 출발한다. 앵커 수가 바뀌면
        # 다시 잡고, 그대로면 학습된 값을 유지한다.
        r0 = (pos[eb] - pos[ea]).norm(dim=-1).clamp(min=1e-12)
        if self.log_h.numel() != m:
            sp = torch.zeros(m, device=self.dev).index_add_(0, ea, r0) / (k - 1)
            self.register_buffer("h0", sp.clamp(min=1e-9))
            # 절대값이 아니라 h0 에 대한 상대 오프셋. 0 이면 정지 간격 그대로다.
            self.log_h = nn.Parameter(torch.zeros(m, device=self.dev))
            self.log_ka = nn.Parameter(torch.zeros(m, device=self.dev))
        # 정지 배치에서의 B 규모를 바닥으로 삼는다. 스텐실이 좁아져도 이 아래로는
        # 릿지가 내려가지 않는다.
        w0 = torch.exp(-0.5 * (r0 / self.h0[ea].clamp(min=1e-12)) ** 2)
        t0 = torch.zeros(m, device=self.dev).index_add_(0, ea, w0).clamp(min=1e-12)
        w0 = w0 / t0[ea]
        q0 = pos[eb] - pos[ea]
        B0 = torch.zeros(m, 3, 3, device=self.dev).index_add_(
            0, ea, w0.reshape(-1, 1, 1) * (q0.unsqueeze(-1) * q0.unsqueeze(-2)))
        self.B_ref = (B0.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0).median().detach()
        # 앵커가 쥔 가우시안에서 부피와 재질을 모은다
        wg = fit.weights().detach()
        vol = torch.zeros(m, device=self.dev).index_add_(
            0, fit.pair_a, fit.vol[fit.pair_g] * wg)
        mu = torch.zeros(m, device=self.dev).index_add_(
            0, fit.pair_a, fit.mu[fit.pair_g] * fit.vol[fit.pair_g] * wg)
        lam = torch.zeros(m, device=self.dev).index_add_(
            0, fit.pair_a, fit.lam[fit.pair_g] * fit.vol[fit.pair_g] * wg)
        v = vol.clamp(min=1e-20)
        self.ea, self.eb = ea, eb
        self.vol, self.mu, self.lam = vol, mu / v, lam / v
        return ea.shape[0]

    @staticmethod
    def _soft(u, lim):
        """유계 재매개화. u=0 근처에서는 항등이고 |u|->inf 에서 ±lim 으로 수렴."""
        return lim * torch.tanh(u / lim)

    @property
    def ka(self):
        return self._soft(self.log_ka, self.ka_lim).exp()

    @property
    def h(self):
        return self.h0 * self._soft(self.log_h, self.h_lim).exp()

    def clamp_(self):
        # tanh 가 이미 유계로 만들므로 잘라낼 것이 없다
        pass

    def reg(self):
        """정지 설정에서 멀어진 정도. 손실의 정규화 항이 이것도 보게 한다."""
        return self.log_ka.pow(2).mean() + self.log_h.pow(2).mean()

    def prepare_rest(self, pos):
        """롤아웃 시작에 한 번. 그래디언트는 그대로 흐른다."""
        self._rest = self.rest(pos)
        return self._rest

    def rest(self, pos):
        """정지 스텐실. log_h 와 앵커 위치에서 매번 새로 만들므로 둘 다
        힘을 통해 그래디언트를 받는다."""
        from .anchor_sparse import inv3
        m = pos.shape[0]
        q = pos[self.eb] - pos[self.ea]
        r = q.norm(dim=-1).clamp(min=1e-12)
        w = torch.exp(-0.5 * (r / self.h[self.ea].clamp(min=1e-12)) ** 2)
        tot = torch.zeros(m, device=pos.device).index_add_(0, self.ea, w).clamp(min=1e-12)
        w = w / tot[self.ea]
        B = torch.zeros(m, 3, 3, device=pos.device).index_add_(
            0, self.ea, w.reshape(-1, 1, 1) * (q.unsqueeze(-1) * q.unsqueeze(-2)))
        tr = (B.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0).clamp(min=1e-20)
        eye = torch.eye(3, device=pos.device)
        # 하드 clamp(min=B_ref) 은 바닥 아래에서 미분이 정확히 0 이라, 좁아진
        # 앵커가 그 상태에 갇힌 채 되돌아올 신호를 못 받는다. softplus 로 부드럽게
        # 받친다: tr >> B_ref 면 tr 그대로, tr -> 0 이면 log2 * B_ref 로 수렴하고
        # 그 사이 어디서도 그래디언트가 끊기지 않는다.
        b = self.B_ref.clamp(min=1e-30)
        eps = self.eig_floor * b * torch.nn.functional.softplus(tr / b)
        Binv = inv3(B + eps.reshape(-1, 1, 1) * eye)
        B_det = B
        if self._owner:
            o = self._owner[0]
            o._chk("스텐실_h", self.h)
            o._chk("스텐실_w", w)
            o._chk_min("앵커응력_trB", tr)
            o._chk_min("앵커응력_detB", torch.linalg.det(B))
            o._chk("앵커응력_Binv", Binv)
        return w, q, Binv

    def forward(self, p, pos0):
        if self.ea.numel() == 0:
            return torch.zeros_like(p)
        from .anchor_sparse import closest_rotation, det3, inv3
        m = p.shape[0]
        w, q, Binv = self._rest if self._rest is not None else self.rest(pos0)
        d = p[self.eb] - p[self.ea]
        A = torch.zeros(m, 3, 3, device=p.device).index_add_(
            0, self.ea, w.reshape(-1, 1, 1) * (d.unsqueeze(-1) * q.unsqueeze(-2)))
        F = A @ Binv
        if self._owner:
            o = self._owner[0]
            o._chk("앵커응력_A", A)
            o._chk("앵커응력_F", F)
        R = closest_rotation(F, self.polar_iters, 1e-6)
        J = det3(F)
        # |J| << J_max 면 J 그대로, 커지면 ±J_max 로 매끄럽게 포화. 부호는 보존한다
        J = J / (1.0 + J.abs() / self.J_max)
        if self._owner:
            o._chk("앵커응력_R", R)
            o._chk("앵커응력_J", J)
            o._chk("앵커응력_강성", self.ka)
        n_ = F.reshape(-1, 9).norm(dim=-1).clamp(min=1e-12).reshape(-1, 1, 1)
        eye = torch.eye(3, device=p.device)
        FiT = inv3(F + 1e-6 * n_ * eye, eps=1e-30).transpose(-1, -2)
        kk = self.ka.reshape(-1, 1, 1)
        mu = self.mu.reshape(-1, 1, 1) * kk
        lam = self.lam.reshape(-1, 1, 1) * kk
        if self._owner:
            o._chk("앵커응력_Finv", FiT)
            o._chk("앵커응력_mu", mu)
            o._chk("앵커응력_lam", lam)
        P = 2 * mu * (F - R) + lam * (J - 1).reshape(-1, 1, 1) * J.reshape(-1, 1, 1) * FiT
        if self._owner:
            self._owner[0]._chk("앵커응력_P", P)
        PB = (P @ Binv)[self.ea]
        c = -(self.vol[self.ea] * w).unsqueeze(-1) * \
            torch.einsum("pij,pj->pi", PB, q)
        f = torch.zeros_like(p)
        f.index_add_(0, self.eb, c)
        f.index_add_(0, self.ea, -c)
        return f
