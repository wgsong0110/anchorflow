"""VR-GS (Jiang et al., SIGGRAPH 2024, arXiv 2401.16663) 직접 구현 -- XPBD + 고정 위상 사면체 케이지에 3DGS 임베딩 (탄성).

코드가 공개되지 않아 논문 본문에 있는 것은 그대로, 없는 것은 표준 XPBD 관행으로 메운다 ( 에 전부 적음).

논문에 있는 것
  - 시뮬레이터 XPBD (Macklin et al. 2016): 서브스텝마다 예측 -> 구속 반복 -> 속도 = 위치 차 / dt, 라그랑주 승수 λ 는 서브스텝마다 0.
  - 탄성 = 변형 에너지 구속 ("strain energy constraint"), 물성 E·ν·ρ.
  - 가우시안 중심 = 담고 있는 사면체의 무게중심 보간 μ = Σ w_i x_i (rest 에서 정한 w), 공분산 Σ = F Σ0 Fᵀ (F = Ds Dm⁻¹).
  - 케이지 정점 수 10K~30K.
  - 바닥 충돌 (물성 조정에 씀). 서브스텝당 반복 수는 장면마다 1~50 (표 1), PhysGaussian 비교는 dt 1e-4 s·반복 11.
논문에 없는 것 (표준 관행)
  - 케이지: 논문은 VDB -> marching cubes -> TetGen. 여기는 가우시안 중심 점유 격자 (1 칸 팽창) -> 칸마다 Kuhn 6 사면체 (meshgs/tetcage.py 와 같은 구조).
    점유 격자 칸 크기는 정점 수가 상한 (기본 30K) 이하인 가장 작은 칸.
  - 변형 에너지 구속의 꼴: 정답 재질 (fixed corotated, ψ = μ‖F−R‖² + λ/2 (det F − 1)²) 을 두 구속으로 나눠 에너지를 정확히 맞춤
      C_R = ‖F − R‖_F, α_R = 1/(2μV);  C_V = det F − 1, α_V = 1/(λV)   (½C_R²/α_R + ½C_V²/α_V = V ψ)
    사면체마다 두 구속을 2x2 로 함께 푼다 (Macklin & Müller 2021 의 결합 풀이). R 은 Müller et al. 2016 반복 회전 추출 (사면체별 사원수 따뜻한 시작, 5 회).
  - 풀이 순서: 가우스-자이델, 사면체 탐욕 색칠 (같은 색은 꼭짓점을 공유하지 않음) -- GPU XPBD 표준.
  - 질량: 사면체 집중 질량 ρ V_T / 4 (ρ = 사면체 안 가우시안의 입자 밀도 평균, 없으면 config 밀도).
  - 바닥: 정점 비관통 z ≥ 바닥 (반복마다 투영) + 고착 (닿은 정점은 이번 서브스텝 접선 이동 취소) = 정답 sticky collider.
  - 고정 경계 (ficus cuboid): rest 위치가 상자 안인 정점 역질량 0. 시간 구간 cuboid (pillow2sofa) 는 PG 와 같이 start <= t < end 동안만 (프레임 시작 시각 판정).
  - 감쇠: 씬 config grid_v_damping_scale 를 같은 시간율로 정점 속도에 (서브스텝마다 s^(dt/substep_dt)); lego 는 1 (없음).
  - 초기 속도: 입자 속도의 질량·무게중심 가중 평균 (한 번, 프레임 0).
  - 물체-물체 충돌: 논문은 충돌을 다루지만 (4.3.1 "relative density between two objects based on their deformations under collision",
    6.2 "dining table and objects on it serve as collision boundaries ... [Deul et al. 2016]", 표 1 "CD: collision detections per step" 5~20) 알고리즘 세부가 없음 ->
    XPBD 표준: 물체 = 케이지 연결 성분, 다른 물체 정점 쌍의 비관통 부등식 구속 |x_i − x_j| >= d (d = 케이지 칸 h, α = 0), 마찰 없음 (cfric 0; 1 = 고착),
    해시 격자 검출을 반복마다 (서브스텝당 CD = 반복 수), 야코비 + 구속 평균. 성분 하나 (레고) 면 커널 없음.
  - PG bounding_box: 격자 노드 i < 3 / i >= n_grid − 3 의 바깥 방향 속도 0 (미끄럼 벽) -> 정점 축별 [2 dx, grid_lim − 3 dx] 투영.
서브스텝 안에서는 입자를 쓰지 않는다 (가우시안 갱신은 프레임 끝 한 번).

Runner (old exe/pareto_vrgs.py; library part = old fusedgs/vrgs_xpbd.py):
  python exe/baseline_vrgs.py --mode fit|run|scale --data <scene>.pt --tag <tag> --out <dir> --set NxIT ...
"""
from __future__ import annotations
import os
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib')
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
import math
import numpy as np
import torch
import hexfem as RTC
import xpbd as _XPBD
_SRC = _XPBD.cu('_SRC')
_MOD = [None]

def _mod():
    if _MOD[0] is None:
        import cupy as cp
        _MOD[0] = cp.RawModule(code=_SRC, options=('--std=c++14', '--use_fast_math'))
    return _MOD[0]
_P = RTC._cp
_CORN = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
_TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]

def _occ(x, h, dil=1):
    lo = x.min(0).values - (dil + 1) * h
    idx = torch.floor((x - lo) / h).long()
    R = (idx.max(0).values + dil + 2).tolist()
    occ = torch.zeros(R, dtype=torch.bool, device=x.device)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if dil > 0:
        occ = torch.nn.functional.max_pool3d(occ.float()[None, None], 2 * dil + 1, 1, dil)[0, 0] > 0
    return (occ, lo, idx)

def build_cage(x, h, dil=1):
    """점유 격자 (칸 h, dil 칸 팽창) -> 정점 [V,3], 사면체 [T,4], 입자 결속 (사면체 번호, 무게중심 4)."""
    dev = x.device
    (occ, lo, idx) = _occ(x, h, dil)
    R = occ.shape
    cell = occ[:-1, :-1, :-1].clone()
    for (a, b, c) in _CORN[1:]:
        cell &= occ[a:R[0] - 1 + a, b:R[1] - 1 + b, c:R[2] - 1 + c]
    ci = torch.nonzero(cell)
    off = torch.tensor(_CORN, device=dev)
    vid = (ci[:, None, :] + off[None]).reshape(-1, 3)
    key = (vid[:, 0] * R[1] + vid[:, 1]) * R[2] + vid[:, 2]
    (uk, inv) = torch.unique(key, return_inverse=True)
    V = torch.stack([uk // (R[1] * R[2]), uk // R[2] % R[1], uk % R[2]], -1).double() * h + lo.double()
    ck = inv.reshape(-1, 8)
    T = torch.stack([ck[:, list(t)] for t in _TETS], 1).reshape(-1, 4)
    cid = torch.full((R[0] - 1, R[1] - 1, R[2] - 1), -1, dtype=torch.long, device=dev)
    cid[ci[:, 0], ci[:, 1], ci[:, 2]] = torch.arange(ci.shape[0], device=dev)
    pc = cid[idx[:, 0].clamp(max=R[0] - 2), idx[:, 1].clamp(max=R[1] - 2), idx[:, 2].clamp(max=R[2] - 2)]
    assert bool((pc >= 0).all()), '결속 실패: 입자가 든 칸이 케이지에 없음'
    xd = x.double()
    best_t = torch.zeros(x.shape[0], dtype=torch.long, device=dev)
    best_w = torch.zeros(x.shape[0], 4, dtype=torch.float64, device=dev)
    best_m = torch.full((x.shape[0],), -1e+30, dtype=torch.float64, device=dev)
    for k in range(6):
        t = pc * 6 + k
        P = V[T[t]]
        A = torch.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], -1)
        l = torch.linalg.solve(A, (xd - P[:, 0]).unsqueeze(-1)).squeeze(-1)
        wv = torch.cat([1.0 - l.sum(-1, keepdim=True), l], -1)
        m = wv.min(-1).values
        upd = m > best_m
        best_t = torch.where(upd, t, best_t)
        best_w = torch.where(upd[:, None], wv, best_w)
        best_m = torch.where(upd, m, best_m)
    return (V, T, best_t, best_w, float(best_m.min()))

def cage_for_max_verts(x, vmax, dil=1):
    """정점 수 ≤ vmax 인 가장 작은 칸 h (이분 탐색, 로그 척도)."""
    ext = float((x.max(0).values - x.min(0).values).max())
    (lo_h, hi_h) = (ext / 400.0, ext / 4.0)

    def nv(h):
        (occ, _, _) = _occ(x, h, dil)
        return int(occ.sum())
    for _ in range(30):
        mid = math.sqrt(lo_h * hi_h)
        if nv(mid) > vmax:
            lo_h = mid
        else:
            hi_h = mid
    h = hi_h
    while True:
        (V, T, tid, bw, mm) = build_cage(x, h, dil)
        if V.shape[0] <= vmax:
            return (h, V, T, tid, bw, mm)
        h *= 1.01

def greedy_color(T, nv):
    """탐욕 색칠 (꼭짓점 공유 사면체는 다른 색) -- 공간 순서 (사면체 번호 = 칸 순서) 그대로."""
    Tn = T.cpu().numpy()
    vm = np.zeros(nv, dtype=np.uint64)
    col = np.empty(Tn.shape[0], dtype=np.int64)
    one = np.uint64(1)
    for t in range(Tn.shape[0]):
        (a, b, c, d) = Tn[t]
        m = int(vm[a] | vm[b] | vm[c] | vm[d])
        k = (~m & m + 1).bit_length() - 1
        assert k < 64
        col[t] = k
        bit = one << np.uint64(k)
        vm[a] |= bit
        vm[b] |= bit
        vm[c] |= bit
        vm[d] |= bit
    return torch.from_numpy(col)

class VRGS:
    """케이지 XPBD. x0 [Np,3] 입자 rest, v0 [Np,3], m [Np] 입자 질량, rho_p [Np] 입자 밀도."""

    def __init__(s, x0, v0, m, rho_p, E, nu, g, frame_dt, N, iters, vmax=30000, floor_z=None, fix_boxes=(), damp_scale=1.0, damp_dt=None, h=None, nrot=5, rot_bcs=(), fix_boxes_t=(), origin=None, collide=True, bbox=None, hcap=16, cfric=0.0, plast=None, mov_boxes=()):
        dev = 'cuda'
        s.Np = x0.shape[0]
        x0 = x0.to(dev).double()
        if h is None:
            (s.h, V, T, tid, bw, s.bind_min) = cage_for_max_verts(x0, vmax)
        else:
            s.h = h
            (V, T, tid, bw, s.bind_min) = build_cage(x0, h)
        (s.nv, s.nt) = (V.shape[0], T.shape[0])
        col = greedy_color(T, s.nv).to(dev)
        order = torch.argsort(col, stable=True)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=dev)
        T = T[order]
        tid = inv[tid]
        cnt = torch.bincount(col, minlength=int(col.max()) + 1).tolist()
        s.colors = [(int(sum(cnt[:c])), int(cnt[c])) for c in range(len(cnt)) if cnt[c] > 0]
        P = V[T]
        Dm = torch.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], -1)
        s.vol = torch.linalg.det(Dm).abs() / 6.0
        s.Dmi = torch.linalg.inv(Dm).float().contiguous().reshape(-1)
        s.Dmi0 = s.Dmi.clone()
        s.plast = plast
        s.mat = {'sand': 2, 'plasticine': 5}.get((plast or {}).get('material'), 0)
        if s.mat == 5:
            assert int(plast.get('hardening', 0)) == 0, 'PG hardening 1 은 미지원 (config 에 없으면 0)'
        s.ys0 = float((plast or {}).get('yield_stress', 0.0))
        s.ys = torch.full((s.nt,), s.ys0, device=dev, dtype=torch.float32)
        s.dead = torch.zeros(s.nt, device=dev, dtype=torch.uint8)
        s.ml = torch.zeros(2, device=dev, dtype=torch.float32)
        rho_p = rho_p.to(dev).double()
        rs = torch.zeros(s.nt, device=dev, dtype=torch.float64).index_add_(0, tid, rho_p)
        rc = torch.zeros(s.nt, device=dev, dtype=torch.float64).index_add_(0, tid, torch.ones_like(rho_p))
        rho_t = torch.where(rc > 0, rs / rc.clamp(min=1), m.to(dev).double().sum() / (m.to(dev).double() / rho_p).sum())
        Mv = torch.zeros(s.nv, device=dev, dtype=torch.float64).index_add_(0, T.reshape(-1), (rho_t * s.vol / 4.0).repeat_interleave(4))
        w = 1.0 / Mv
        s.fixed = torch.zeros(s.nv, dtype=torch.bool, device=dev)
        for (pt, sz) in fix_boxes:
            s.fixed |= ((V - torch.tensor(pt, device=dev, dtype=torch.float64)).abs() < torch.tensor(sz, device=dev, dtype=torch.float64)).all(-1)
        s.rotg = list(rot_bcs)
        s.kin = torch.zeros(s.nv, dtype=torch.bool, device=dev)
        s.kin_g = []
        for (pt, nv_, hh, rr, wins) in s.rotg:
            pt_ = torch.tensor(pt, device=dev, dtype=torch.float64)
            n_ = torch.tensor(nv_, device=dev, dtype=torch.float64)
            off = V - pt_
            ax = off @ n_
            rp = off - ax[:, None] * n_
            msk = (ax.abs() < hh) & (rp.norm(dim=-1) < rr)
            s.kin_g.append(msk)
            s.kin |= msk
        s.fixed |= s.kin
        s.fixw = []
        for (pt, sz, t0, t1) in fix_boxes_t:
            mw = ((V - torch.tensor(pt, device=dev, dtype=torch.float64)).abs() < torch.tensor(sz, device=dev, dtype=torch.float64)).all(-1) & ~s.fixed
            s.fixw.append((mw, float(t0), float(t1)))
        s.movw = []
        for (pt, sz, t0, t1, vel) in mov_boxes:
            mw = ((V - torch.tensor(pt, device=dev, dtype=torch.float64)).abs() < torch.tensor(sz, device=dev, dtype=torch.float64)).all(-1) & ~s.fixed
            s.movw.append((mw, float(t0), float(t1), [float(q) for q in vel]))
        s.w_free = w.float().contiguous().clone()
        s.w_free[s.fixed] = 0.0
        w[s.fixed] = 0.0
        for (mw, t0, t1) in s.fixw:
            if t0 <= 1e-06 < t1:
                w[mw] = 0.0
        for (mw, t0, t1, _v) in s.movw:
            if t0 <= 1e-06 < t1:
                w[mw] = 0.0
        s.fr = 0
        s.vd = torch.zeros(s.nv, 3, device=dev, dtype=torch.float32)
        s.kin_u8 = s.kin.to(torch.uint8).contiguous()
        s.mass_cage = float(Mv.sum())
        s.mass_part = float(m.sum())
        m_ = m.to(dev).double()
        v0 = v0.to(dev).double()
        wp_ = bw.clamp(min=0.0) * m_[:, None]
        Ms = torch.zeros(s.nv, device=dev, dtype=torch.float64).index_add_(0, T[tid].reshape(-1), wp_.reshape(-1))
        Ps = torch.zeros(s.nv, 3, device=dev, dtype=torch.float64).index_add_(0, T[tid].reshape(-1), (wp_[:, :, None] * v0[:, None, :]).reshape(-1, 3))
        vbar = (m_[:, None] * v0).sum(0) / m_.sum()
        vv = torch.where(Ms[:, None] > 0, Ps / Ms.clamp(min=1e-300)[:, None], vbar[None])
        vv[s.fixed] = 0.0
        vv[w == 0.0] = 0.0
        for (mw, t0, t1, vel) in s.movw:
            if t0 <= 1e-06 < t1:
                vv[mw] = torch.tensor(vel, device=dev, dtype=torch.float64)
        s.origin = None if origin is None else torch.as_tensor(origin, device=dev, dtype=torch.float64)
        if s.origin is not None:
            V = V - s.origin
        s.X0 = V.float().contiguous()
        s.V0 = vv.float().contiguous()
        s.x = s.X0.clone()
        s.xp = s.X0.clone()
        s.v = s.V0.clone()
        s.w = w.float().contiguous()
        s.T = T.int().contiguous()
        s.tid = tid.int().contiguous()
        s.bw = bw.float().contiguous()
        s.lam = torch.zeros(2 * s.nt, device=dev)
        s.q = torch.zeros(s.nt, 4, device=dev)
        s.q[:, 3] = 1.0
        s.xo = torch.zeros(s.Np, 3, device=dev)
        s.g = [float(q) for q in g]
        s.frame_dt = float(frame_dt)
        s.nu = float(nu)
        s.fz = -1e+30 if floor_z is None else float(floor_z) - (0.0 if s.origin is None else float(s.origin[2]))
        (s.damp_scale, s.damp_dt) = (float(damp_scale), damp_dt)
        s.nrot = int(nrot)
        s.aR = torch.zeros(s.nt, device=dev)
        s.aV = torch.zeros(s.nt, device=dev)
        lab = torch.arange(s.nv, device=dev)
        Tl = T.long()
        while True:
            tm = lab[Tl].min(1).values
            nl = lab.scatter_reduce(0, Tl.reshape(-1), tm.repeat_interleave(4), reduce='amin')
            nl = nl[nl]
            if bool((nl == lab).all()):
                break
            lab = nl
        (_, comp) = torch.unique(lab, return_inverse=True)
        s.ncomp = int(comp.max()) + 1
        s.comp_sizes = torch.bincount(comp).sort(descending=True).values[:8].tolist()
        s.collide = bool(collide) and s.ncomp > 1
        s.comp = comp.int().contiguous()
        s.cd_d = float(s.h)
        s.hcap = int(hcap)
        s.cfric = float(cfric)
        hs = 1
        while hs < 2 * s.nv:
            hs *= 2
        s.hsize = hs
        if s.collide:
            s.hcnt = torch.zeros(hs, device=dev, dtype=torch.int32)
            s.hslot = torch.zeros(hs * s.hcap, device=dev, dtype=torch.int32)
            s.cdx = torch.zeros(s.nv, 3, device=dev)
        s.bbox = None
        if bbox is not None:
            o_ = [0.0, 0.0, 0.0] if s.origin is None else s.origin.tolist()
            s.bbox = ([float(bbox[0][k]) - o_[k] for k in range(3)], [float(bbox[1][k]) - o_[k] for k in range(3)])
        s.set_setting(N, iters)
        s.set_E(E)
        s.g_sub = s.g_skin = None

    def set_setting(s, N, iters):
        (s.N, s.iters) = (int(N), int(iters))
        s.dt = s.frame_dt / s.N
        s.damp = s.damp_scale ** (s.dt / s.damp_dt) if s.damp_dt and s.damp_scale < 1.0 else 1.0
        s.g_sub = s.g_skin = None

    def set_E(s, E):
        s.E = float(E)
        mu = s.E / (2.0 * (1.0 + s.nu))
        la = s.E * s.nu / ((1.0 + s.nu) * (1.0 - 2.0 * s.nu))
        dt2 = s.dt * s.dt
        s.aR.copy_((1.0 / (2.0 * mu * s.vol) / dt2).float())
        s.aV.copy_((1.0 / (la * s.vol) / dt2).float())
        s.ml.copy_(torch.tensor([mu, la], dtype=torch.float32))

    def reset(s):
        s.fr = 0
        s.x.copy_(s.X0)
        s.xp.copy_(s.X0)
        s.v.copy_(s.V0)
        s.q.zero_()
        s.q[:, 3] = 1.0
        s.Dmi.copy_(s.Dmi0)
        s.ys.fill_(s.ys0)
        s.dead.zero_()
        if s.fixw or s.movw:
            s._pin()

    def _pin(s):
        """[final_pa_bl] 프레임 시작 시각 t 에 켜진 구간 고정 상자의 정점 역질량 0 (+ 속도 0), 꺼진 상자는 원래 역질량. 그래프는 s.w 를 그대로 읽음."""
        t = s.fr * s.frame_dt
        w = s.w_free.clone()
        on = torch.zeros_like(s.fixed)
        for (mw, t0, t1) in s.fixw:
            if t0 <= t + 1e-06 < t1:
                on |= mw
        w[on] = 0.0
        s.v[on] = 0.0
        if s.movw:
            kin = s.kin.clone()
            vd = s.vd.clone() if s.rotg else torch.zeros_like(s.vd)
            for (mw, t0, t1, vel) in s.movw:
                if t0 <= t + 1e-06 < t1:
                    w[mw] = 0.0
                    kin |= mw
                    vd[mw] = torch.tensor(vel, device=vd.device, dtype=vd.dtype)
                    s.v[mw] = vd[mw]
            s.kin_u8.copy_(kin.to(torch.uint8))
            s.vd.copy_(vd)
        s.w.copy_(w)

    def _rec_sub(s):
        M = _mod()
        (f_pred, f_zero, f_solve, f_floor, f_vel, f_kin) = (M.get_function(k) for k in ('vg_predict', 'vg_zero', 'vg_solve', 'vg_floor', 'vg_vel', 'vg_kin'))
        (f_hclear, f_hins, f_contact, f_addx, f_box) = (M.get_function(k) for k in ('vg_hclear', 'vg_hins', 'vg_contact', 'vg_addx', 'vg_box'))
        B = 128
        gv = ((s.nv + B - 1) // B,)
        (i32, f32) = (np.int32, np.float32)
        with RTC._Stream():
            f_pred(gv, (B,), (i32(s.nv), _P(s.x), _P(s.xp), _P(s.v), _P(s.w), f32(s.g[0]), f32(s.g[1]), f32(s.g[2]), f32(s.dt), f32(s.damp)))
            if s.rotg or s.movw:
                f_kin(gv, (B,), (i32(s.nv), _P(s.x), _P(s.vd), _P(s.kin_u8), f32(s.dt)))
            f_zero(((2 * s.nt + B - 1) // B,), (B,), (i32(2 * s.nt), _P(s.lam)))
            for _ in range(s.iters):
                for (o, c) in s.colors:
                    f_solve(((c + B - 1) // B,), (B,), (i32(o), i32(c), _P(s.T), _P(s.x), _P(s.w), _P(s.Dmi), _P(s.aR), _P(s.aV), _P(s.lam), _P(s.q), i32(s.nrot), i32(s.mat), _P(s.dead)))
                if s.collide:
                    (ih, mk, hc) = (f32(1.0 / s.cd_d), np.uint32(s.hsize - 1), i32(s.hcap))
                    f_hclear(((s.hsize + B - 1) // B,), (B,), (i32(s.hsize), _P(s.hcnt)))
                    f_hins(gv, (B,), (i32(s.nv), _P(s.x), _P(s.comp), ih, mk, hc, _P(s.hcnt), _P(s.hslot)))
                    f_contact(gv, (B,), (i32(s.nv), _P(s.x), _P(s.xp), _P(s.w), _P(s.comp), ih, mk, hc, _P(s.hcnt), _P(s.hslot), f32(s.cd_d), f32(s.cfric), _P(s.cdx)))
                    f_addx(gv, (B,), (i32(s.nv), _P(s.x), _P(s.cdx)))
                if s.bbox is not None:
                    (lo_, hi_) = s.bbox
                    f_box(gv, (B,), (i32(s.nv), _P(s.x), _P(s.w), f32(lo_[0]), f32(lo_[1]), f32(lo_[2]), f32(hi_[0]), f32(hi_[1]), f32(hi_[2])))
                if s.fz > -1e+29:
                    f_floor(gv, (B,), (i32(s.nv), _P(s.x), _P(s.xp), _P(s.w), f32(s.fz)))
            if s.mat:
                M.get_function('vg_plast')(((s.nt + B - 1) // B,), (B,), (i32(s.nt), _P(s.T), _P(s.x), _P(s.Dmi), _P(s.ys), _P(s.dead), i32(s.mat), _P(s.ml), f32(s.plast.get('cdp', 0.0)), f32(s.plast.get('softening', 0.1))))
            f_vel(gv, (B,), (i32(s.nv), _P(s.x), _P(s.xp), _P(s.v), f32(1.0 / s.dt)))

    def _rec_skin(s):
        f = _mod().get_function('vg_skin')
        with RTC._Stream():
            f(((s.Np + 127) // 128,), (128,), (np.int32(s.Np), _P(s.tid), _P(s.bw), _P(s.T), _P(s.x), _P(s.xo)))

    def capture(s):
        bk = (s.x.clone(), s.xp.clone(), s.v.clone(), s.q.clone(), s.Dmi.clone(), s.ys.clone(), s.dead.clone())
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            s._rec_sub()
            s._rec_skin()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        s.g_sub = torch.cuda.CUDAGraph()
        with torch.cuda.graph(s.g_sub):
            s._rec_sub()
        s.g_skin = torch.cuda.CUDAGraph()
        with torch.cuda.graph(s.g_skin):
            s._rec_skin()
        torch.cuda.synchronize()
        s.x.copy_(bk[0])
        s.xp.copy_(bk[1])
        s.v.copy_(bk[2])
        s.q.copy_(bk[3])
        s.Dmi.copy_(bk[4])
        s.ys.copy_(bk[5])
        s.dead.copy_(bk[6])

    def frame(s):
        """한 프레임 (N 서브스텝 + 가우시안 중심 갱신). 돌려줌: GPU 시간 ms (CUDA 이벤트), 입자 위치 s.xo."""
        if s.g_sub is None:
            s.capture()
        if s.rotg:
            s._drive()
        if s.fixw or s.movw:
            s._pin()
        (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        e0.record()
        for _ in range(s.N):
            s.g_sub.replay()
        s.g_skin.replay()
        e1.record()
        torch.cuda.synchronize()
        if s.origin is not None:
            s.xo += s.origin.float()
        s.fr += 1
        return e0.elapsed_time(e1)

    def _drive(s):
        """프레임 시각 [t, t+h) 와 겹친 몫 f 만큼 곱한 회전 구동 속도 (PG: 시각 start <= t < end, v = rs (r_perp x n) + ts n, r 은 현재 위치) 를 구동 정점에."""
        t = s.fr * s.frame_dt
        vd = torch.zeros(s.nv, 3, device=s.x.device, dtype=torch.float32)
        for ((pt, nv_, hh, rr, wins), msk) in zip(s.rotg, s.kin_g):
            rs = ts = 0.0
            for (r_, t_, t0, t1) in wins:
                f = max(0.0, min(t + s.frame_dt, t1) - max(t, t0)) / s.frame_dt
                rs += r_ * f
                ts += t_ * f
            n_ = torch.tensor(nv_, device=s.x.device, dtype=torch.float32)
            off = s.x - torch.tensor(pt, device=s.x.device, dtype=torch.float32) + (0.0 if s.origin is None else s.origin.float())
            rp = off - (off @ n_)[:, None] * n_
            v = rs * torch.cross(rp, n_.expand_as(rp), dim=-1) + ts * n_
            vd[msk] = v[msk]
        s.vd.copy_(vd)

    def F_part(s):
        """입자 (가우시안) 마다 결속 사면체의 변형 기울기 F = Ds Dm^-1 [Np,3,3] (렌더 공분산 F Σ0 Fᵀ, 논문 그대로; 시간 밖)."""
        P = s.x[s.T.long()]
        Ds = torch.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], -1)
        return (Ds @ s.Dmi0.reshape(-1, 3, 3))[s.tid.long()]

    def info(s):
        return dict(h=s.h, nv=s.nv, nt=s.nt, ncolor=len(s.colors), ncomp=s.ncomp, comp_sizes=s.comp_sizes, collide=s.collide, contact_d=s.cd_d, contact_fric=s.cfric, bbox=s.bbox, nfixed=int(s.fixed.sum()), nfixed_t=[int(q[0].sum()) for q in s.fixw], nmov=[int(q[0].sum()) for q in s.movw], mat=s.mat, plast=s.plast, ndead=int(s.dead.sum()), mass_cage=s.mass_cage, mass_part=s.mass_part, bind_min=s.bind_min, N=s.N, iters=s.iters, dt=s.dt, damp=s.damp, E=s.E, nu=s.nu)
if __name__ == '__main__':
    import argparse
    import json
    import math
    import os
    import sys
    import time
    import numpy as np
    import torch
    import gsrender as DM
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=80)
    ap.add_argument('--floor', choices=['vel', 'none'], default='vel')
    ap.add_argument('--cfg_bc', action='store_true')
    ap.add_argument('--region_density', action='store_true')
    ap.add_argument('--pg_vdamp', action='store_true')
    ap.add_argument('--vmax', type=int, default=30000, help='케이지 정점 상한 (논문 10K~30K)')
    ap.add_argument('--mode', choices=['fit', 'run', 'scale'], default='fit')
    ap.add_argument('--set', nargs='+', default=['16x5'], help='설정 목록 NxIT (서브스텝 x 반복)')
    ap.add_argument('--fit_json', default=None, help='run: 설정별 E (fit 출력 fit.json 들, 쉼표)')
    ap.add_argument('--E', type=float, default=0.0, help='run: 고정 E (0 = fit_json)')
    ap.add_argument('--nfit', type=int, default=10)
    ap.add_argument('--span', type=float, default=2.0, help='탐색 폭 (정답 E 의 10^±span)')
    ap.add_argument('--warm', type=int, default=2)
    ap.add_argument('--no_collide', action='store_true', help='[final_fix_vrgs] 물체-물체 충돌 끄기 (예전 판; 케이지 성분 하나면 원래 없음)')
    ap.add_argument('--vg_contact', choices=['auto', 'on', 'off'], default='auto', help='[paper 합침] 물체-물체 충돌·bounding_box (final_fix_vrgs) 를 쓸지: auto = 탄성 재질만 (final_fix_vrgs 레고·pillow2sofa 판), sand·plasticine 은 끔 (final_pa_bl bread·wolf 판)')
    ap.add_argument('--contact_fric', choices=['none', 'stick'], default='none', help='[final_fix_vrgs] 물체-물체 접촉 접선 처리 (논문·Deul 2016 인용에 계수 없음 -> 기본 마찰 없음)')
    ap.add_argument('--dump_F', action='store_true', help='run: 첫 설정 덤프에 입자 F (결속 사면체 F) 도 -> pred.pt {pred, F} (안정 서브스텝 표 렌더용, doc/pareto.md 절 S)')
    a = ap.parse_args()
    RTC.RAWPTR[0] = True
    os.makedirs(a.out, exist_ok=True)
    dev = 'cuda'
    d = torch.load(a.data, map_location='cpu', weights_only=False)
    cfg = d['cfg']
    DM.REGION_DENSITY[0] = bool(a.region_density)
    m = DM.traj_mass(d, dev).double()
    EXT = float(d['_ext'])
    rho_p = d['_rho'].to(dev).double() if a.region_density and '_rho' in d else torch.full_like(m, float(cfg['density']))
    L = min(a.frames, int(d['x'].shape[0]) - 1)
    FZ = None
    if a.floor == 'vel':
        pl = [b for b in cfg['boundary_conditions'] if b.get('type') == 'surface_collider'][0]
        assert abs(float(pl['normal'][2]) - 1.0) < 1e-06, '바닥 법선이 z 가 아님'
        FZ = float(pl['point'][2])
    import baseline_spring as BC
    (BOX, BOXW, MOV) = BC.parse_bcs(cfg, L * float(cfg['frame_dt'])) if a.cfg_bc else ([], [], [])
    for _b in BOXW:
        assert len(_b) == 4
    PLAST = None
    if cfg.get('material', 'jelly') in ('sand', 'plasticine'):
        PLAST = BC.plastic_params(cfg)
    elif cfg.get('material', 'jelly') != 'jelly':
        raise SystemExit(f"VR-GS 재질 미지원: {cfg.get('material')}")
    VGC = PLAST is None if a.vg_contact == 'auto' else a.vg_contact == 'on'
    BBOX = None
    if a.cfg_bc and VGC:
        for bc in cfg.get('boundary_conditions') or []:
            if bc.get('type') == 'bounding_box':
                assert float(bc.get('start_time', 0.0)) <= 0.0 and float(bc.get('end_time', 999.0)) > L * float(cfg['frame_dt']), '시간 구간이 있는 bounding_box 는 미지원'
                (_gl, _ng) = (float(cfg.get('grid_lim', 2.0)), int(cfg['n_grid']))
                _dx = _gl / _ng
                BBOX = ([2 * _dx] * 3, [(_ng - 3) * _dx] * 3)
    ROT = {}
    if a.cfg_bc:
        import numpy as _np
        for bc in cfg.get('boundary_conditions') or []:
            if bc.get('type') == 'enforce_particle_velocity_rotation':
                nv_ = _np.asarray(bc['normal'], dtype=_np.float64)
                nv_ = nv_ / _np.linalg.norm(nv_)
                k_ = (tuple((float(q) for q in bc['point'])), tuple(nv_.tolist()), float(bc['half_height_and_radius'][0]), float(bc['half_height_and_radius'][1]))
                ROT.setdefault(k_, []).append((float(bc['rotation_scale']), float(bc['translation_scale']), float(bc.get('start_time', 0.0)), float(bc.get('end_time', 1000.0))))
        ROT = [(k_[0], list(k_[1]), k_[2], k_[3], w_) for (k_, w_) in ROT.items()]
        for _g in ROT:
            _w = sorted(_g[4], key=lambda q: q[2])
            assert _w[0][2] == 0.0 and all((abs(_w[i][3] - _w[i + 1][2]) < 1e-09 for i in range(len(_w) - 1))) and (_w[-1][3] >= 1.0), '회전 구동 구간이 시간을 덮지 않음'
    else:
        ROT = []
    DSC = float(cfg.get('grid_v_damping_scale', 1.0)) if a.pg_vdamp else 1.0
    (E_GT, NU) = (float(cfg['E']), float(cfg['nu']))
    GT = d['x'][1:L + 1].float()
    x0 = d['x'][0].to(dev).float()

    def parse(sv):
        (n, it) = sv.split('x')
        return (int(n), int(it))
    SETS = [parse(q) for q in a.set]
    S = VRGS(x0, d['v'][0], m, rho_p, E_GT, NU, cfg['g'], float(cfg['frame_dt']), SETS[0][0], SETS[0][1], vmax=a.vmax, floor_z=FZ, fix_boxes=BOX, damp_scale=DSC, damp_dt=float(cfg['substep_dt']), rot_bcs=ROT, fix_boxes_t=BOXW, origin=(0.5 * (x0.min(0).values + x0.max(0).values)).tolist() if BOXW or MOV else None, collide=VGC and (not a.no_collide), bbox=BBOX, cfric=1.0 if a.contact_fric == 'stick' else 0.0, plast=PLAST, mov_boxes=MOV)
    INFO = S.info()
    print(f"[VR-GS] {a.tag} 입자 {S.Np}, 케이지 정점 {S.nv} 사면체 {S.nt} 칸 {S.h:.5f} 색 {len(S.colors)} 고정 정점 {INFO['nfixed']} (구간 고정 {INFO.get('nfixed_t')}), 질량 케이지 {S.mass_cage:.4g} / 입자 {S.mass_part:.4g}, 결속 무게중심 최소 {S.bind_min:.2e}, 물체 (케이지 성분) {S.ncomp} 개 (큰 순 {S.comp_sizes}) 충돌 {('켬 d %.5f 마찰 %s' % (S.cd_d, a.contact_fric) if S.collide else '없음')}, 경계 상자 {BBOX}, 재질 {INFO['mat']} {PLAST}, 구동 상자 정점 {INFO['nmov']}, 바닥 {FZ}, 감쇠 {DSC}, 정답 E {E_GT:g} ν {NU}, 프레임 {L}, EXT {EXT:.5f}", flush=True)
    print(f"[명령] {' '.join(sys.argv)}", flush=True)

    def rollout(E, keep=False):
        S.set_E(E)
        S.reset()
        (errs, ms) = ([], [])
        P = torch.empty(L, S.Np, 3) if keep else None
        PF = torch.empty(L, S.Np, 3, 3) if keep and a.dump_F else None
        div = None
        for f in range(L):
            t = S.frame()
            xo = S.xo
            gt = GT[f].to(dev)
            e = float((xo - gt).norm(dim=-1).mean()) / EXT
            if not math.isfinite(e) or e > 2.0:
                div = f + 1
                errs.append(float('nan'))
                break
            errs.append(e)
            ms.append(t)
            if keep:
                P[f] = xo.cpu()
            if PF is not None:
                PF[f] = S.F_part().cpu()
        sc = ms[a.warm:] if len(ms) > a.warm else ms
        return (dict(E=E, err=float(np.mean(errs)) if div is None else None, err_done=float(np.nanmean(errs)) if np.isfinite(errs).any() else None, div=div, frames=len(errs), ndead=int(S.dead.sum()), ms_mean=float(np.mean(sc)) if sc else None, ms_max=float(np.max(sc)) if sc else None, errs=errs), P if PF is None else (P, PF))

    def obj(r):
        return r['err'] if r['err'] is not None else 1000000000.0
    if a.mode == 'fit':
        res = []
        for (N, it) in SETS:
            S.set_setting(N, it)
            t0 = time.time()
            (lo, hi) = (math.log10(E_GT) - a.span, math.log10(E_GT) + a.span)
            gr = (math.sqrt(5) - 1) / 2
            (c_, d_) = (hi - gr * (hi - lo), lo + gr * (hi - lo))
            ev = []
            (rc, _) = rollout(10 ** c_)
            ev.append(rc)
            (rd, _) = rollout(10 ** d_)
            ev.append(rd)
            while len(ev) < a.nfit:
                if obj(rc) <= obj(rd):
                    (hi, d_, rd) = (d_, c_, rc)
                    c_ = hi - gr * (hi - lo)
                    (rc, _) = rollout(10 ** c_)
                    ev.append(rc)
                else:
                    (lo, c_, rc) = (c_, d_, rd)
                    d_ = lo + gr * (hi - lo)
                    (rd, _) = rollout(10 ** d_)
                    ev.append(rd)
            best = min(ev, key=obj)
            r = dict(N=N, iters=it, dt=S.dt, E_best=best['E'], err=best['err'], div=best['div'], ms_mean_nonexcl=best['ms_mean'], evals=[dict(E=q['E'], err=q['err'], div=q['div']) for q in ev], sec=time.time() - t0)
            res.append(r)
            print(f"  [맞춤] {a.tag} N {N} 반복 {it}: E {best['E']:.4g} (정답 {E_GT:g} 의 {best['E'] / E_GT:.3g} 배) 오차 {('발산' if best['err'] is None else '%.3f%%' % (100 * best['err']))}, {r['sec']:.0f}s, 평가 " + ' '.join((f"{q['E']:.3g}:{('x' if q['err'] is None else '%.2f' % (100 * q['err']))}" for q in ev)), flush=True)
            json.dump(dict(info=INFO, tag=a.tag, E_gt=E_GT, res=res), open(os.path.join(a.out, 'fit.json'), 'w'), indent=1, ensure_ascii=False)
    elif a.mode == 'run':
        EB = {}
        if a.fit_json:
            for fj in a.fit_json.split(','):
                for r in json.load(open(fj))['res']:
                    EB[r['N'], r['iters']] = r['E_best']
        res = []
        for (N, it) in SETS:
            E = a.E if a.E > 0 else EB.get((N, it))
            if E is None:
                print(f'  [없음] {a.tag} {N}x{it}: 맞춘 E 없음 -- 건너뜀', flush=True)
                continue
            S.set_setting(N, it)
            (r, P) = rollout(E, keep=(N, it) == SETS[0])
            r.update(N=N, iters=it, dt=S.dt)
            res.append(r)
            if isinstance(P, tuple):
                if r['div'] is None:
                    torch.save({'pred': P[0], 'F': P[1]}, os.path.join(a.out, 'pred.pt'))
            elif P is not None:
                torch.save(P, os.path.join(a.out, f'pred_{N}x{it}.pt'))
            print(f"  [실행] {a.tag} N {N} 반복 {it} E {E:.4g}: 오차 {('발산 %d' % r['div'] if r['err'] is None else '%.3f%%' % (100 * r['err']))}, 프레임 {r['ms_mean']:.3f} ms (최대 {r['ms_max']:.3f})", flush=True)
            json.dump(dict(info=INFO, tag=a.tag, E_gt=E_GT, res=[{k: v for (k, v) in q.items() if k != 'errs'} | dict(errs=q['errs']) for q in res]), open(os.path.join(a.out, 'run.json'), 'w'), indent=1, ensure_ascii=False)
    else:
        PAPER = [('Horse', 10611, 1295223, 5, 115.4), ('Bear', 19472, 2525891, 5, 76.9), ('Fox', 28205, 304875, 1, 161.2)]
        out = []
        for (nm, nv, ng, it_p, fps) in PAPER:
            T = VRGS(x0, d['v'][0], m, rho_p, E_GT, NU, cfg['g'], float(cfg['frame_dt']), 1, 1, vmax=nv, floor_z=FZ, fix_boxes=BOX, rot_bcs=ROT, damp_scale=DSC, damp_dt=float(cfg['substep_dt']))
            rep = (ng + T.Np - 1) // T.Np
            T.tid = T.tid.repeat(rep)[:ng].contiguous()
            T.bw = T.bw.repeat(rep, 1)[:ng].contiguous()
            T.Np = ng
            T.xo = torch.zeros(ng, 3, device=dev)
            row = dict(scene=nm, nv_paper=nv, nv=T.nv, nt=T.nt, ncolor=len(T.colors), gauss=ng, iters_paper=it_p, fps_paper=fps, sub_ms={})
            for it in sorted({1, 5, 11, it_p}):
                T.set_setting(1, it)
                T.set_E(E_GT)
                T.capture()
                for _ in range(20):
                    T.g_sub.replay()
                torch.cuda.synchronize()
                (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                e0.record()
                for _ in range(200):
                    T.g_sub.replay()
                e1.record()
                torch.cuda.synchronize()
                row['sub_ms'][it] = e0.elapsed_time(e1) / 200
            (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(5):
                T.g_skin.replay()
            e0.record()
            for _ in range(50):
                T.g_skin.replay()
            e1.record()
            torch.cuda.synchronize()
            row['skin_ms'] = e0.elapsed_time(e1) / 50
            row['max_sub_in_budget'] = (1000.0 / fps - row['skin_ms']) / row['sub_ms'][it_p]
            out.append(row)
            print(f'  [규모] {nm}: 정점 {T.nv} (논문 {nv}) 사면체 {T.nt} 색 {len(T.colors)}, 서브스텝 1 회 ' + ' '.join((f'반복 {k} {v:.4f} ms' for (k, v) in row['sub_ms'].items())) + f", 가우시안 {ng} 중심 갱신 {row['skin_ms']:.4f} ms -> 논문 {fps} fps (프레임 {1000 / fps:.2f} ms, 렌더 포함) 안에 서브스텝 최대 {row['max_sub_in_budget']:.1f} 회 (반복 {it_p})", flush=True)
            del T
            torch.cuda.empty_cache()
        json.dump(out, open(os.path.join(a.out, 'scale.json'), 'w'), indent=1, ensure_ascii=False)
    print('[끝]', flush=True)
