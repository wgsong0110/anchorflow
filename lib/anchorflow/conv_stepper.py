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

from .nextstate import DtFiLM


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
                 arch="plain", damage=False, n_mat=0, drop=0.0, rqs_dim=0,
                 dt_cond=False, dt_ref=1.0, dt_scale=False):
        super().__init__()
        # 블록 사이의 채널 드롭아웃. 부피가 대부분 비어 있어 **화소별** 드롭은
        # 빈 칸만 지우기 쉬우므로, 채널 통째로 떨어뜨리는 Dropout3d 를 쓴다.
        self.drop = nn.Dropout3d(float(drop)) if drop > 0 else nn.Identity()
        self.h, self.scale, self.arch, self.damage = h, scale, arch, damage
        # 격자-입자 전달은 Kuhn 사면체 barycentric 고정이다. 예전 학습 반경(skin)
        # 헤드는 "셀 내부 변화가 고정"이던 시절의 땜질이라 제거했다 -- 그
        # 자유도는 셀 내부 RQS 재배열(rqs_dim 헤드)이 든다.
        self.register_buffer("in_mu", torch.zeros(n_feat))
        self.register_buffer("in_sd", torch.ones(n_feat))
        self.film = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 2 * hidden))
        # 물성은 씬 안에서 **상수**라 셀 특징에 붙이면 채널 하나를 상수로 채우는
        # 셈이다. 대신 블록마다 채널별 스케일·시프트로 넣는다 (FiLM).
        self.n_mat = int(n_mat)
        if self.n_mat:
            self.mfilm = nn.ModuleList([
                nn.Sequential(nn.Linear(self.n_mat, hidden), nn.SiLU(),
                              nn.Linear(hidden, 2 * hidden))
                for _ in range(depth + 1)])
            for m in self.mfilm:                   # 처음에는 항등 변조
                nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
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
        self.out = nn.Conv3d(hidden, 4 if damage else 3, 1)
        # 가중치를 **정확히** 0 으로 두면 상류로 가는 기울기가 grad_out @ W = 0
        # 이라 첫 스텝에 인코더·블록이 기울기를 하나도 못 받는다. 어텐션 쪽에서
        # 겪은 함정이다 -- 편향만 0 으로 두고 가중치는 기본 초기화의 1/100 로.
        if _ZERO_OUT:
            nn.init.zeros_(self.out.weight)
        else:
            with torch.no_grad():
                self.out.weight.mul_(0.01)
        nn.init.zeros_(self.out.bias)
        # 셀 내부 RQS 재배열 파라미터 헤드. 격자점에서 내고 8 꼭짓점 평균으로
        # 셀 값으로 바꾼다 -- 이웃 셀이 꼭짓점을 나눠 가져 파라미터장이 저절로
        # 상관되고, 원시값 0 이 항등이라 초기화도 변위 헤드와 같은 규약이다.
        self.rqs_dim = int(rqs_dim)
        if self.rqs_dim:
            self.out_rqs = nn.Conv3d(hidden, self.rqs_dim, 1)
            with torch.no_grad():
                self.out_rqs.weight.mul_(0.01)
            nn.init.zeros_(self.out_rqs.bias)
        # --- 스텝 크기 dt 를 **조건 변수**로 받는다 -----------------------------
        # 기본 self.film 은 dt 원시값을 Linear(1,.) 에 넣는다. dt 를 고정해 쓰면
        # 상수라 사실상 편향이지만, dt 를 흔들면 0.042 ~ 0.0004 처럼 두 자리
        # 넘게 걸쳐 조건이 나빠진다. 어텐션 경로가 쓰던 DtFiLM 을 그대로 쓴다 --
        # log10(dt) 를 푸리에로 펴서 층마다 (gamma, beta) 로 넣고, 0 초기화라
        # 켜는 순간에는 항등이다 (그래서 옛 체크포인트에 얹어도 값이 안 변한다).
        self.dt_ref = float(dt_ref) if dt_ref else 1.0
        self.dt_scale = bool(dt_scale)
        self.dtfilm = DtFiLM(hidden, depth + 1) if dt_cond else None


    def set_input_stats(self, feats):
        """[S, F] 표본에서 채널별 평균/표준편차를 잡아 버퍼에 넣는다."""
        f = feats.detach().float().reshape(-1, feats.shape[-1])
        sd = f.std(0)
        self.in_mu.copy_(f.mean(0))
        self.in_sd.copy_(torch.where(sd > 1e-4 * sd.max().clamp(min=1e-12),
                                     sd, torch.ones_like(sd)))

    def _dtmod(self, v, dt, i):
        if self.dtfilm is None:
            return v
        g, b = self.dtfilm(dt, v.device)
        j = min(i, self.dtfilm.n_sites - 1)
        return (g[j].to(v.dtype).view(1, -1, 1, 1, 1) * v
                + b[j].to(v.dtype).view(1, -1, 1, 1, 1))

    def _mod(self, v, mat, i):
        if not self.n_mat or mat is None:
            return v
        g, b = self.mfilm[min(i, len(self.mfilm) - 1)](mat).chunk(2, -1)
        return (1.0 + g).view(1, -1, 1, 1, 1) * v + b.view(1, -1, 1, 1, 1)

    def forward(self, p, feat, dt, grid, static=None, cells=None, mat=None,
                ens=1):
        """grid 는 격자점 크기. cells 가 주어지면 feat 은 **셀** 기준이고,
        c2g 가 격자점으로 옮긴다.

        ens > 1 이면 feat 이 [ens*M, F] 로 쌓여 들어온 것으로 보고 **한 번의**
        컨볼루션으로 처리한다. 어긋난 격자 여러 개를 따로 돌리면 커널 실행만
        그만큼 늘어 GPU 가 비는데, 가중치가 같으므로 배치 차원으로 묶으면 된다.
        """
        nx, ny, nz = grid
        f = feat if static is None else torch.cat([feat, static], -1)
        f = (f - self.in_mu) / self.in_sd
        if cells is not None:
            _c = f.reshape(ens, -1, f.shape[-1]).permute(0, 2, 1)
            v = self.c2g(_c.reshape(ens, -1, *cells))
        else:
            _c = f.reshape(ens, -1, f.shape[-1]).permute(0, 2, 1)
            v = _c.reshape(ens, -1, nx, ny, nz)
        v = self.inp(v)
        # 이 film 은 dt 원시값을 받는 **무작위 초기화** Linear 라, dt 를 고정해
        # 쓰던 동안에는 학습된 편향이나 마찬가지였다. dt 를 두 자리에 걸쳐
        # 흔들면 여기도 같이 흔들려 (조건이 나쁜) 두 번째 dt 경로가 된다.
        # 조건화를 켜면 이쪽은 기준 dt 로 못박아 편향 역할만 남기고, dt 의존은
        # DtFiLM 하나로 모은다 -- 그래야 --dt_scale 의 비례가 정확해지고 옛
        # 체크포인트도 기준 dt 에서 값이 그대로 나온다.
        if self.dtfilm is not None:
            _dtf = torch.full((1, 1), self.dt_ref, device=v.device,
                              dtype=v.dtype)
        elif torch.is_tensor(dt):
            _dtf = dt.reshape(1, 1).to(device=v.device, dtype=v.dtype)
        else:
            _dtf = torch.full((1, 1), float(dt), device=v.device,
                              dtype=v.dtype)
        g, b = self.film(_dtf).chunk(2, -1)
        v = g.view(1, -1, 1, 1, 1) * v + b.view(1, -1, 1, 1, 1)
        v = self._mod(v, mat, 0)
        v = self._dtmod(v, dt, 0)
        if self.arch == "unet":
            s1 = self.d1(v)
            s2 = self.d2(Fn.avg_pool3d(s1, 2, ceil_mode=True))
            m = self.mid(Fn.avg_pool3d(s2, 2, ceil_mode=True))
            m = Fn.interpolate(m, size=s2.shape[2:], mode="nearest")
            m = self.drop(self.u2(torch.cat([m, s2], 1)))
            m = Fn.interpolate(m, size=s1.shape[2:], mode="nearest")
            v = self.u1(torch.cat([m, s1], 1))
        else:
            for _i, blk in enumerate(self.body):
                v = v + self.drop(blk(v))
                v = self._mod(v, mat, _i + 1)
                v = self._dtmod(v, dt, _i + 1)
        o = self.out(v)                                        # [E,C,nx,ny,nz]
        rq = None
        if self.rqs_dim:
            # 격자점 -> 셀: 2^3 평균 (격자점 수 n -> 셀 수 n-1)
            r = Fn.avg_pool3d(self.out_rqs(v), 2, stride=1)
            rq = r.reshape(ens, r.shape[1], -1).permute(0, 2, 1).reshape(
                -1, r.shape[1])                                # [E*Mc, P]
        o = o.reshape(ens, o.shape[1], -1).permute(0, 2, 1).reshape(
            -1, o.shape[1])                                    # [E*M, C]
        # 변위는 1 차로 v*dt 라 dt 에 비례한다. --dt_scale 이면 그 비례를
        # 구조로 박아 망이 dt 의존을 처음부터 안 배워도 되게 한다.
        # float() 로 감싸면 dt 에 대한 미분 경로가 끊긴다 -- 텐서는 그대로 쓴다
        _sc = (self.scale * (dt / self.dt_ref) if self.dt_scale
               else self.scale)
        dp = o[:, :3] * _sc
        if self.damage:
            out = (dp, Fn.softplus(o[:, 3] - 3.0))
        else:
            out = (dp,)
        return out + (rq,) if rq is not None else out
