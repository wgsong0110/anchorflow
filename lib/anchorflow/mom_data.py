"""운동량 잔차 변위 망의 자기지도 학습 상태 생성 (사용자 10-27) -- 정답 궤적과 무관.

lego 형상(L000 0 프레임 입자 = 기준 배치 X, 질량·부피 그대로) 위에서 한 스텝 시작 상태 (x, v, F, h) 를 무작위로 만든다.
  회전   R: 확률 p_rot_full 로 SO(3) 균등, 나머지는 축 정렬 자세 24 개 중 하나 -> 연직축 균등 회전 뒤 무작위 축 기울기 logU[1e-4, tilt_max] (바닥에 평평히)
  무변형 확률 p_rest 로 변형·퍼짐 없음 (F = R, 힘 0 근처)
  변형   y = F_aff (X + u(X)),  u = Σ_k a_k sin(ω_k·X + φ_k)  (매끈한 무작위 변형장, 파장 logU[3 h_node, 3 EXT])
         변형 크기 S ~ logU[strain_min, strain_max] (모드 변형률 |a_k||ω_k| = S·U/√K), F_aff = Q diag(exp ε) Qᵀ, |ε| ≤ S_aff ~ logU
         F_p = R_p R F_aff (I + ∇u(X_p)) (I + σ N_p)  -- 입자별 퍼짐 σ ~ logU[spread_min, spread_max],
         일부(비율 U[0, out_frac]) 입자는 추가 회전 R_p (각 logU[1e-3, out_ang]) -- 사면체 안 F 퍼짐 (실제 롤아웃의 이력 차)
         det F 가 det_min 아래로 가면 변형 크기를 절반으로 줄여 다시 (최대 6 번)
  눌림   확률 p_flat 로 바닥 법선 방향 비 U[flat_min, 1] 로 눌러 x·F 에 같이 (바닥에 납작해진 상태)
  위치   x = R y + t,  바닥 최소 거리 c: 확률 (p_pen, p_near, 나머지) 로 U[−0.5, 0]·h_node / U[0, 2]·h_node / U[2 h_node, clear_max·EXT],
         바닥 면 안 방향 이동 U[−2, 2]·h_node (격자 정렬 바꾸기)
  속도   v = v_lin + ω × (x − c) + v_sm(X) + 입자 잡음: |v_lin| ~ logU[v_min, v_max], |ω| ~ logU[w_min, w_max],
         v_sm 진폭 ~ logU[vs_min, vs_max] (같은 꼴의 매끈한 장), 입자 잡음 σ_v ~ logU[1e-5, vnoise_max]
  dt     확률 p_frame 로 1/60, 나머지 logU[1/60/dt_sub_max, 1/60]
"""
from __future__ import annotations

import math

import torch

DEFAULTS = dict(p_rest=0.15, p_rot_full=0.3, tilt_max=0.6, K=6, strain_min=1e-6, strain_max=0.2, aff_max=0.15, p_flat=0.25, flat_min=0.4, spread_min=1e-6, spread_max=0.05,
                out_frac=0.03, out_ang=1.5, det_min=0.15, p_pen=0.15, p_near=0.55, clear_max=0.5, v_min=1e-3, v_max=6.0, w_min=1e-3,
                w_max=10.0, vs_min=1e-4, vs_max=2.0, vnoise_max=0.1, p_frame=0.25, dt_sub_max=64)


def _logu(gen, lo, hi, n=(), dev="cuda"):
    u = torch.rand(n, generator=gen, device=dev)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def _unit(gen, n, dev):
    q = torch.randn(n, 3, generator=gen, device=dev)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _rodrigues(axis, ang):
    """axis [...,3] 단위, ang [...] -> R [...,3,3]."""
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=axis.device, dtype=axis.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -axis[..., 2], axis[..., 1], -axis[..., 0]
    K = K - K.transpose(-1, -2)
    s, c = torch.sin(ang)[..., None, None], torch.cos(ang)[..., None, None]
    I = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(K)
    return I + s * K + (1 - c) * (K @ K)


def _cube_rots(dev):
    """정육면체 회전군 24 개 (부호 있는 치환 행렬, det +1)."""
    import itertools
    out = []
    for pm in itertools.permutations(range(3)):
        for sg in itertools.product((1.0, -1.0), repeat=3):
            M = torch.zeros(3, 3)
            for i, j in enumerate(pm):
                M[i, j] = sg[i]
            if float(torch.linalg.det(M)) > 0:
                out.append(M)
    return torch.stack(out).to(dev)


def _rand_rot(gen, dev):
    q = torch.randn(4, generator=gen, device=dev)
    q = q / q.norm()
    w, x, y, z = q
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
                        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
                        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)])])


class Fourier:
    """u(X) = Σ_k a_k sin(ω_k·X + φ_k), ∇u = Σ_k a_k ω_kᵀ cos(...)."""

    def __init__(self, gen, K, lam_lo, lam_hi, amp_strain, dev):
        lam = _logu(gen, lam_lo, lam_hi, (K,), dev)
        self.w = _unit(gen, K, dev) * (2 * math.pi / lam).unsqueeze(-1)            # [K,3]
        self.ph = torch.rand(K, generator=gen, device=dev) * 2 * math.pi
        s = amp_strain * torch.rand(K, generator=gen, device=dev) / math.sqrt(K)    # 모드 변형률 |a||ω|
        self.a = _unit(gen, K, dev) * (s / self.w.norm(dim=-1)).unsqueeze(-1)       # [K,3]

    def val(self, X):
        return torch.sin(X @ self.w.T + self.ph) @ self.a

    def grad(self, X):
        c = torch.cos(X @ self.w.T + self.ph)                                        # [N,K]
        return torch.einsum("nk,ki,kj->nij", c, self.a, self.w)


class StateGen:
    def __init__(self, sc, dev="cuda", **kw):
        self.sc, self.dev = sc, dev
        self.o = dict(DEFAULTS, **kw)
        self.c0 = sc.x0.mean(0)
        self.X = sc.x0 - self.c0                    # 기준 배치 (중심 원점)
        self.N = self.X.shape[0]
        self.cube = _cube_rots(dev)

    def sample(self, gen):
        o, sc, dev, X, N = self.o, self.sc, self.dev, self.X, self.N
        I3 = torch.eye(3, device=dev)
        # 회전
        if float(torch.rand((), generator=gen, device=dev)) < o["p_rot_full"]:
            R = _rand_rot(gen, dev)
        else:                                       # 연직축 회전 (균등) 뒤 작은 기울기 logU[1e-4, tilt_max] -- 바닥에 평평히 놓인 자세 (실제 롤아웃의 넓은 접촉)
            nz = sc.planes[0][1] if sc.planes else torch.tensor([0.0, 0.0, 1.0], device=dev)
            Ry = _rodrigues(nz, torch.rand((), generator=gen, device=dev) * 2 * math.pi)
            Rc = self.cube[int(torch.randint(0, 24, (), generator=gen, device=dev))]   # 축 정렬 자세 24 개 중 하나 (어느 면이든 바닥에 눕게)
            R = _rodrigues(_unit(gen, 1, dev)[0], _logu(gen, 1e-4, o["tilt_max"], (), dev)) @ Ry @ Rc
        # 변형
        S = float(_logu(gen, o["strain_min"], o["strain_max"], (), dev))
        Sa = float(_logu(gen, o["strain_min"], o["aff_max"], (), dev))
        Q = _rand_rot(gen, dev)
        eps = (2 * torch.rand(3, generator=gen, device=dev) - 1) * Sa
        fu = Fourier(gen, o["K"], 3 * sc.node_h, 3 * sc.EXT, 1.0, dev)
        sig = float(_logu(gen, o["spread_min"], o["spread_max"], (), dev))
        Np = torch.randn(N, 3, 3, generator=gen, device=dev)
        fo = float(torch.rand((), generator=gen, device=dev)) * o["out_frac"]
        om = torch.rand(N, generator=gen, device=dev) < fo
        oang = _logu(gen, 1e-3, o["out_ang"], (N,), dev)
        Rp = _rodrigues(_unit(gen, N, dev), oang * om.float())
        rest = float(torch.rand((), generator=gen, device=dev)) < o["p_rest"]   # 변형 없는 상태 (회전만, 힘 0 근처 -- 실제 0 프레임 같은 것)
        scale = 0.0 if rest else 1.0
        for _ in range(6):
            Fa = Q @ torch.diag(torch.exp(eps * scale)) @ Q.T
            Fs = Fa @ (I3 + scale * fu.grad(X))
            Fp = (Rp if not rest else I3) @ (R @ Fs) @ (I3 + scale * sig * Np)
            if float(torch.linalg.det(Fp).min()) >= o["det_min"]:
                break
            scale *= 0.5
        y = (X + scale * fu.val(X)) @ Fa.T
        x = y @ R.T
        # 바닥 법선 방향 눌림 (확률 p_flat, 비 U[flat_min, 1]): 실제 롤아웃은 바닥에서 높이가 반까지 납작해진다 (정답 80 프레임 높이 1.056 -> 0.52)
        if sc.planes and not rest and float(torch.rand((), generator=gen, device=dev)) < o["p_flat"]:
            nf = sc.planes[0][1]
            sq = o["flat_min"] + (1 - o["flat_min"]) * float(torch.rand((), generator=gen, device=dev))
            Sq = I3 + (sq - 1.0) * torch.outer(nf, nf)
            x = x @ Sq.T
            Fp = Sq @ Fp
        else:
            sq = 1.0
        # 바닥 높이·면 안 이동
        if sc.planes:
            p0, n = sc.planes[0]
            r = float(torch.rand((), generator=gen, device=dev))
            u = float(torch.rand((), generator=gen, device=dev))
            if r < o["p_pen"]:
                c = -0.5 * sc.node_h * u
            elif r < o["p_pen"] + o["p_near"]:
                c = 2.0 * sc.node_h * u
            else:
                c = 2.0 * sc.node_h + u * (o["clear_max"] * sc.EXT - 2.0 * sc.node_h)
            t = self.c0.clone()
            x = x + t
            sd = ((x - p0) @ n).min()
            x = x + (c - sd) * n
            tan = (2 * torch.rand(3, generator=gen, device=dev) - 1) * 2 * sc.node_h
            x = x + tan - (tan @ n) * n
        else:
            x = x + self.c0
        # 속도
        vl = _unit(gen, 1, dev)[0] * float(_logu(gen, o["v_min"], o["v_max"], (), dev))
        wv = _unit(gen, 1, dev)[0] * float(_logu(gen, o["w_min"], o["w_max"], (), dev))
        fv = Fourier(gen, o["K"], 3 * sc.node_h, 3 * sc.EXT, 1.0, dev)
        vs = float(_logu(gen, o["vs_min"], o["vs_max"], (), dev))
        vsm = fv.val(X)
        vsm = vsm / vsm.norm(dim=-1).max().clamp_min(1e-12) * vs
        vn_ = float(_logu(gen, 1e-5, o["vnoise_max"], (), dev))
        xc = x.mean(0)
        v = vl + torch.cross(wv.expand(N, 3), x - xc, dim=-1) + vsm @ R.T + vn_ * torch.randn(N, 3, generator=gen, device=dev)
        # dt
        if float(torch.rand((), generator=gen, device=dev)) < o["p_frame"]:
            h = sc.frame_dt
        else:
            h = float(_logu(gen, sc.frame_dt / o["dt_sub_max"], sc.frame_dt, (), dev))
        info = dict(strain=S * scale, aff=Sa * scale, spread=sig * scale, out_frac=fo, v_lin=float(vl.norm()), w=float(wv.norm()),
                    v_sm=vs, h=h, rest=rest, flat=sq)
        return x.contiguous(), v.contiguous(), Fp.contiguous(), h, info


@torch.no_grad()
def state_stats(sc, P, x, F, h_eval=None):
    """분포 비교용 상태 통계 (한 상태) -> {이름: 값 텐서 (cpu, 부분표본)}. P = mom_core.prepare 결과."""
    from . import mom_core as MC
    S = P.S
    out = {}
    sv = torch.linalg.svdvals(P.Fb.double()).float()
    out["F̄ σ_min"] = sv[:, -1]
    out["F̄ σ_max"] = sv[:, 0]
    out["det F̄"] = torch.linalg.det(P.Fb)
    out["|F_p − F̄_T|"] = (F - P.Fb[S.pinv]).flatten(1).norm(dim=-1)
    out["|v_n| m/s"] = P.vn.norm(dim=-1)
    out["|f_n/m_n| / |g|"] = (P.f / P.mn.clamp_min(sc.m_floor).unsqueeze(-1)).norm(dim=-1) / sc.gmag
    if sc.planes:
        p0, n = sc.planes[0]
        out["바닥 최소 거리/h_node"] = (((x - p0) @ n).min() / sc.node_h).reshape(1)
    near = P.feat[:, 9]
    out["바닥 근접 노드 비율"] = near.mean().reshape(1)
    xt = MC.explicit_dx(sc, P)
    J = torch.linalg.det(MC.cell_A(S, xt))
    out["명시해 det A_T"] = J
    out["명시해 뒤집힌 사면체 비율"] = (J <= 0).float().mean().reshape(1)
    out["명시해 뒤집힌 사면체 있음"] = (J <= 0).any().float().reshape(1)
    out["h"] = torch.tensor([P.h])
    for k, t in list(out.items()):
        t = t.reshape(-1).float().cpu()
        if t.numel() > 20000:
            t = t[torch.randperm(t.numel(), generator=torch.Generator().manual_seed(0))[:20000]]
        out[k] = t
    return out
