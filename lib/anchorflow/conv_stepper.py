"""복셀 격자 위에서 도는 3D 컨볼루션 스테퍼.

앵커가 격자 중심이면 앵커 특징은 [C, nx, ny, nz] 부피가 되므로 어텐션 대신
컨볼루션으로 이웃을 섞을 수 있다. 상호작용이 국소적이고 평행이동 등변이며
비용이 앵커 수에 선형이다. 대신 먼 거리는 깊이(또는 풀링)로만 닿는다.

`plain` 은 같은 해상도로 컨볼루션을 쌓고, `unet` 은 풀링·업샘플로 다중 스케일을
줘서 적은 깊이로도 물체 반대편까지 잇는다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn


_ZERO_OUT = True    # False 면 출력층 가중치를 기본 초기화의 1/100 로 둔다
_NORM = True        # False 면 블록 안 GroupNorm 을 빼서 **크기 정보를 남긴다**


def _nrm(c):
    """부피의 96% 가 빈 칸이라 GroupNorm 은 배경 통계로 크기를 지운다.

    창마다 필요한 변위 크기가 네 배 넘게 다른데 정규화 뒤에는 출력 크기가
    입력 크기와 무관해져, 망이 진폭을 따라갈 수 없다.
    """
    return nn.GroupNorm(8, c) if _NORM else nn.Identity()


def _blk_sep(cin, cout):
    """축별 분해 블록: 3x3x3 한 번 대신 3x1x1, 1x3x1, 1x1x3 세 번.

    커널당 곱셈이 27 -> 9 로 준다. 수용영역은 같다.
    """
    def tri(ci, co):
        return nn.Sequential(
            nn.Conv3d(ci, co, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.Conv3d(co, co, (1, 3, 1), padding=(0, 1, 0), bias=False),
            nn.Conv3d(co, co, (1, 1, 3), padding=(0, 0, 1), bias=False),
            _nrm(co), nn.SiLU())

    return nn.Sequential(tri(cin, cout), tri(cout, cout))


class _AxisPar(nn.Module):
    """세 축 1D 컨볼루션을 **같은 입력에 병렬로** 걸고 합친다.

    순차 분해와 달리 축 방향 십자(7칸)만 덮는다 -- 대각 이웃은 다음 층에서 닿는다.
    세 가지가 서로 의존하지 않아 GPU 에서 겹쳐 실행될 수 있다.
    """

    def __init__(self, cin, cout):
        super().__init__()
        self.cx = nn.Conv3d(cin, cout, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.cy = nn.Conv3d(cin, cout, (1, 3, 1), padding=(0, 1, 0), bias=False)
        self.cz = nn.Conv3d(cin, cout, (1, 1, 3), padding=(0, 0, 1), bias=False)
        self.nrm = _nrm(cout)
        self.act = nn.SiLU()

    def forward(self, v):
        return self.act(self.nrm(self.cx(v) + self.cy(v) + self.cz(v)))


def _blk_par(cin, cout):
    return nn.Sequential(_AxisPar(cin, cout), _AxisPar(cout, cout))


def _blk(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1), _nrm(cout), nn.SiLU(),
        nn.Conv3d(cout, cout, 3, padding=1), _nrm(cout), nn.SiLU())


class ConvStepper(nn.Module):
    """forward(p, feat, dt, grid, cells=...) -> (dp [M,3],) 또는 (dp, dmg)

    feat 은 **셀** 기준 [M_cell, F] 이고 c2g 가 격자점으로 옮긴다.
    격자 순서는 (ix*ny + iy)*nz + iz 다.
    """

    def __init__(self, n_feat, hidden=64, depth=4, h=0.05, scale=1.0,
                 skin_out=False,
                 arch="plain", damage=False):
        super().__init__()
        self.h, self.scale, self.arch, self.damage = h, scale, arch, damage
        # skin_out: 변위와 함께 **스키닝 반경**을 낸다 (DeformNet 과 같은 매개화).
        # 고정 trilinear 가중치로 전달하면 같은 조건에서 비가 0.25 -> 0.58 로
        # 나빠진다 -- 전달 가중치가 학습돼야 한다.
        self.skin_out = bool(skin_out)
        self.register_buffer("in_mu", torch.zeros(n_feat))
        self.register_buffer("in_sd", torch.ones(n_feat))
        self.film = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 2 * hidden))
        # 셀 집계 결과를 격자점으로 옮기는 층 (2^3, pad 1 -> 셀 n -> 격자점 n+1)
        self.c2g = nn.Conv3d(n_feat, n_feat, 2, padding=1)
        self.inp = nn.Conv3d(n_feat, hidden, 1)
        B = (_blk_par if arch.endswith("_par")
             else _blk_sep if arch.endswith("_sep") else _blk)
        base = arch.replace("_sep", "").replace("_par", "")
        self.arch = base
        if base == "unet":
            self.d1, self.d2 = B(hidden, hidden), B(hidden, 2 * hidden)
            self.mid = B(2 * hidden, 2 * hidden)
            self.u2 = B(4 * hidden, hidden)
            self.u1 = B(2 * hidden, hidden)
        else:
            self.body = nn.ModuleList([B(hidden, hidden) for _ in range(depth)])
        self.out = nn.Conv3d(hidden, 4 if (damage or skin_out) else 3, 1)
        # 가중치를 **정확히** 0 으로 두면 상류로 가는 기울기가 grad_out @ W = 0
        # 이라 첫 스텝에 인코더·블록이 기울기를 하나도 못 받는다. 어텐션 쪽에서
        # 겪은 함정이다 -- 편향만 0 으로 두고 가중치는 기본 초기화의 1/100 로.
        if _ZERO_OUT:
            nn.init.zeros_(self.out.weight)
        else:
            with torch.no_grad():
                self.out.weight.mul_(0.01)
        nn.init.zeros_(self.out.bias)


    def set_input_stats(self, feats):
        """[S, F] 표본에서 채널별 평균/표준편차를 잡아 버퍼에 넣는다."""
        f = feats.detach().float().reshape(-1, feats.shape[-1])
        sd = f.std(0)
        self.in_mu.copy_(f.mean(0))
        self.in_sd.copy_(torch.where(sd > 1e-4 * sd.max().clamp(min=1e-12),
                                     sd, torch.ones_like(sd)))

    def forward(self, p, feat, dt, grid, static=None, cells=None):
        """grid 는 격자점 크기. cells 가 주어지면 feat 은 **셀** 기준이고,
        c2g 가 격자점으로 옮긴다."""
        nx, ny, nz = grid
        f = feat if static is None else torch.cat([feat, static], -1)
        f = (f - self.in_mu) / self.in_sd
        if cells is not None:
            v = self.c2g(f.t().reshape(1, -1, *cells))
        else:
            v = f.t().reshape(1, -1, nx, ny, nz)
        v = self.inp(v)
        g, b = self.film(torch.as_tensor([[float(dt)]], device=v.device,
                                         dtype=v.dtype)).chunk(2, -1)
        v = g.view(1, -1, 1, 1, 1) * v + b.view(1, -1, 1, 1, 1)
        if self.arch == "unet":
            s1 = self.d1(v)
            s2 = self.d2(Fn.avg_pool3d(s1, 2, ceil_mode=True))
            m = self.mid(Fn.avg_pool3d(s2, 2, ceil_mode=True))
            m = Fn.interpolate(m, size=s2.shape[2:], mode="nearest")
            m = self.u2(torch.cat([m, s2], 1))
            m = Fn.interpolate(m, size=s1.shape[2:], mode="nearest")
            v = self.u1(torch.cat([m, s1], 1))
        else:
            for blk in self.body:
                v = v + blk(v)
        o = self.out(v).reshape(-1, nx * ny * nz).t()          # [M, C]
        dp = o[:, :3] * self.scale
        if self.skin_out:
            lr = (o[:, 3] + math.log(self.h)).clamp(
                math.log(self.h) - 3.0, math.log(self.h) + 3.0)
            return dp, lr, torch.zeros_like(lr)
        # trilinear 전달이라 스키닝 반경·온도가 없다. 변위만 낸다.
        if self.damage:
            return dp, Fn.softplus(o[:, 3] - 3.0)
        return (dp,)
