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


def _blk(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class ConvStepper(nn.Module):
    """forward(p, feat, dt, grid) -> (dp [M,3], log_r [M], log_t [M])

    feat [M,F] 는 격자 순서 (ix*ny + iy)*nz + iz 로 정렬돼 있어야 한다.
    grid 는 (nx, ny, nz).
    """

    def __init__(self, n_feat, hidden=64, depth=4, h=0.05, scale=1.0,
                 arch="plain", damage=False):
        super().__init__()
        self.h, self.scale, self.arch, self.damage = h, scale, arch, damage
        self.register_buffer("in_mu", torch.zeros(n_feat))
        self.register_buffer("in_sd", torch.ones(n_feat))
        self.film = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 2 * hidden))
        self.inp = nn.Conv3d(n_feat, hidden, 1)
        if arch == "unet":
            self.d1, self.d2 = _blk(hidden, hidden), _blk(hidden, 2 * hidden)
            self.mid = _blk(2 * hidden, 2 * hidden)
            self.u2 = _blk(4 * hidden, hidden)
            self.u1 = _blk(2 * hidden, hidden)
        else:
            self.body = nn.ModuleList([_blk(hidden, hidden) for _ in range(depth)])
        self.out = nn.Conv3d(hidden, 6 if damage else 4, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, p, feat, dt, grid, static=None):
        nx, ny, nz = grid
        f = feat if static is None else torch.cat([feat, static], -1)
        f = (f - self.in_mu) / self.in_sd
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
        log_r = o[:, 3] + math.log(self.h)
        log_t = o[:, 4] if self.damage else torch.zeros_like(log_r)
        out = (dp, log_r.clamp(math.log(self.h) - 3.0, math.log(self.h) + 3.0),
               log_t.clamp(-4.0, 4.0))
        if self.damage:
            out = out + (Fn.softplus(o[:, 5] - 3.0),)
        return out


# ---------------------------------------------------------------- 희소 판
class SparseConvStepper(nn.Module):
    """점유 칸에서만 도는 submanifold 희소 컨볼루션 스테퍼.

    조밀 판은 빈 칸까지 전부 연산한다. 물체가 bbox 의 일부만 채우면 그만큼이
    낭비다. 여기서는 점유 칸만 좌표로 들고 다닌다.

    forward(p, feat, dt, meta) -- meta = (coords [M,3] long, grid (nx,ny,nz))
    """

    def __init__(self, n_feat, hidden=64, depth=4, h=0.05, scale=1.0,
                 arch="plain", damage=False):
        super().__init__()
        import spconv.pytorch as spc
        from spconv.core import ConvAlgo
        self.spc = spc
        # 이 환경에서 기본 MaskImplicitGemm 커널이 FPE 로 죽는다. Native 로 고정한다.
        _algo = ConvAlgo.Native
        self.h, self.scale, self.arch, self.damage = h, scale, arch, damage
        self.register_buffer("in_mu", torch.zeros(n_feat))
        self.register_buffer("in_sd", torch.ones(n_feat))
        self.film = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 2 * hidden))

        def sblk(cin, cout):
            return spc.SparseSequential(
                spc.SubMConv3d(cin, cout, 3, bias=False, algo=_algo),
                nn.GroupNorm(8, cout), nn.SiLU(),
                spc.SubMConv3d(cout, cout, 3, bias=False, algo=_algo),
                nn.GroupNorm(8, cout), nn.SiLU())

        self.inp = spc.SubMConv3d(n_feat, hidden, 1, bias=True, algo=_algo)
        self.body = nn.ModuleList([sblk(hidden, hidden) for _ in range(depth)])
        self.out = spc.SubMConv3d(hidden, 6 if damage else 4, 1, bias=True,
                          algo=_algo)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, p, feat, dt, meta, static=None):
        coords, grid = meta
        f = feat if static is None else torch.cat([feat, static], -1)
        f = (f - self.in_mu) / self.in_sd
        b = torch.zeros(coords.shape[0], 1, dtype=torch.int32, device=f.device)
        ind = torch.cat([b, coords.int()], 1)
        x = self.spc.SparseConvTensor(f, ind, list(grid), 1)
        x = self.inp(x)
        g, bb = self.film(torch.as_tensor([[float(dt)]], device=f.device,
                                          dtype=f.dtype)).chunk(2, -1)
        x = x.replace_feature(g * x.features + bb)
        for blk in self.body:
            y = blk(x)
            x = x.replace_feature(x.features + y.features)
        o = self.out(x).features                                # [M, C]
        dp = o[:, :3] * self.scale
        log_r = o[:, 3] + math.log(self.h)
        log_t = o[:, 4] if self.damage else torch.zeros_like(log_r)
        out = (dp, log_r.clamp(math.log(self.h) - 3.0, math.log(self.h) + 3.0),
               log_t.clamp(-4.0, 4.0))
        if self.damage:
            out = out + (Fn.softplus(o[:, 5] - 3.0),)
        return out
