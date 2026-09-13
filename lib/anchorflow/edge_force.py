"""앵커 쌍이 직접 주고받는, 학습되는 힘.

표현 하한 측정(2026-08-31)에서 앵커 512 개는 MPM 을 0.03~0.13% 로 담을 수 있는데
시뮬레이터는 3~7% 에 머문다는 것이 확인됐다. 자유도가 아니라 그 상태를 다음
상태로 옮기는 법칙이 부족하다는 뜻이라, 연속체 응력에서 유도하지 않는 항을
하나 더한다.

세 가지를 구조로 강제한다.

  회전 등변   -- 입력은 회전 불변 스칼라뿐이고 출력은 두 앵커를 잇는 방향뿐이다
  운동량 보존 -- 무방향 간선마다 한 번 계산해 양끝에 +f, -f 로 뿌린다
  각운동량 보존 -- 중심력이라 팔에 대한 모멘트가 상쇄된다

마지막 층을 0 으로 초기화하므로 학습 전에는 f = 0 이고, 시뮬레이터는 지금
그대로다. 그래서 이 항이 손해를 끼치면 그것 자체가 결과가 된다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

N_IN = 7


class EdgeForce(nn.Module):
    def __init__(self, k=16, hidden=32, dev="cuda"):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(N_IN, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        ).to(dev)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.log_gain = nn.Parameter(torch.zeros((), device=dev))
        # 해석적 선형 스프링. 대체형(가우시안 응력 없이 앵커끼리만)에서는 MLP 가
        # 0 에서 출발하므로 힘이 아예 없는 상태가 되어 학습이 성립하지 않는다.
        # 그래서 스프링을 바닥에 깔고 MLP 는 그 위의 보정을 배운다. 강성은
        # 초기에 물리항과 최소제곱으로 맞춘다(calibrate).
        self.log_ke = nn.Parameter(torch.zeros((), device=dev))
        self.spring = False
        self.register_buffer("ei", torch.zeros(0, dtype=torch.long, device=dev))
        self.register_buffer("ej", torch.zeros(0, dtype=torch.long, device=dev))
        self.register_buffer("r0", torch.zeros(0, device=dev))
        self.R = 1.0
        self.A0 = 1.0

    @torch.no_grad()
    def rebuild(self, pos, radius, dt, n_sub=40):
        """정지 배치에서 무방향 kNN 간선을 만든다.

        앵커는 밀도 제어로 늘고 줄므로 refresh 마다 다시 만든다. 간선 수가
        앵커당 k 개라 짝 350 만 개짜리 가우시안 경로에 비하면 무시할 만하다.
        """
        x = pos.detach().cpu().numpy()
        m = x.shape[0]
        k = min(self.k + 1, m)
        _, nb = cKDTree(x).query(x, k=k)
        i = np.repeat(np.arange(m), k - 1)
        j = nb[:, 1:].reshape(-1)
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        uniq = np.unique(lo.astype(np.int64) * m + hi.astype(np.int64))
        ei, ej = (uniq // m).astype(np.int64), (uniq % m).astype(np.int64)
        dev = pos.device
        self.ei = torch.from_numpy(ei).to(dev)
        self.ej = torch.from_numpy(ej).to(dev)
        self.r0 = (pos[self.ej] - pos[self.ei]).norm(dim=-1).clamp(min=1e-9).detach()
        self.R = float(radius)
        # 가속도 단위. 프레임 하나에 반경만큼 움직이는 크기라, 물리항이 내는
        # 값과 자릿수가 비슷하다. 정확할 필요는 없다 -- log_gain 이 마저 맞춘다.
        self.A0 = float(radius) / float(dt * n_sub) ** 2
        return self.ei.shape[0]

    def spring_basis(self, p):
        """강성 1 일 때의 스프링 힘. 보정 계수를 최소제곱으로 구할 때 쓴다."""
        a, b = self.ei, self.ej
        d = p[b] - p[a]
        r = d.norm(dim=-1).clamp(min=1e-9)
        f = ((r - self.r0) / r).unsqueeze(-1) * d
        out = torch.zeros_like(p)
        out.index_add_(0, a, f)
        out.index_add_(0, b, -f)
        return out

    @torch.no_grad()
    def calibrate(self, p, f_phys):
        """물리항이 내던 힘에 스프링 하나를 최소제곱으로 맞춘다."""
        g = self.spring_basis(p)
        ke = float((f_phys * g).sum() / (g * g).sum().clamp(min=1e-30))
        self.log_ke.fill_(float(np.log(max(ke, 1e-12))))
        return ke

    def forward(self, p, v, mass, log_s, log_k):
        if self.ei.numel() == 0:
            return torch.zeros_like(p)
        a, b = self.ei, self.ej
        d = p[b] - p[a]
        r = d.norm(dim=-1).clamp(min=1e-9)
        dh = d / r.unsqueeze(-1)
        dv = v[b] - v[a]
        vr = (dv * dh).sum(-1)
        vp = (dv - vr.unsqueeze(-1) * dh).norm(dim=-1)
        tau = self.r0 / max(self.A0, 1e-30) ** 0.5      # 이 간선의 시간 단위
        ka, kb = log_k[a], log_k[b]
        sa, sb = log_s[a].mean(-1), log_s[b].mean(-1)
        feat = torch.stack([
            r / self.r0 - 1.0,                           # 변형률
            torch.log(self.r0 / self.R),                 # 정지 길이
            vr * tau / self.r0,                          # 지름 방향 변형률 속도
            vp * tau / self.r0,                          # 접선 방향
            0.5 * (ka + kb),                             # 강성 (대칭)
            (ka - kb).abs(),
            0.5 * (sa + sb) - float(np.log(self.R)),     # 크기 (대칭)
        ], dim=-1)
        phi = self.net(feat).squeeze(-1) * self.log_gain.exp()
        # 호출자에 따라 [M] 로도 [M,1] 로도 들어온다
        mv = mass.reshape(mass.shape[0], -1)[:, 0]
        mu_e = (mv[a] * mv[b]) / (mv[a] + mv[b]).clamp(min=1e-12)
        f = (mu_e * self.A0 * phi).unsqueeze(-1) * dh
        if self.spring:
            f = f + (self.log_ke.exp() * (r - self.r0)).unsqueeze(-1) * dh
        out = torch.zeros_like(p)
        out.index_add_(0, a, f)
        out.index_add_(0, b, -f)
        return out
