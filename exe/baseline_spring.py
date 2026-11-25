"""Spring-mass baselines run from their official code: PhysTwin and Spring-Gaus.

Subcommands (old script -> new):
  phystwin     exe/stab_phystwin.py   --mode fit | run
  springgaus   exe/stab_sg.py         --mode prep | train | run   (prep input = evaluation.py multiview)
Shared scene/metric helpers = old exe/stab_bl_common.py; exact resume of the Spring-Gaus dynamic stage = old exe/sg_resume.py.
This file does not import the fusedgs library (the baselines run in their own environments).
"""
import json
import math
import os
import time
import numpy as np
import torch

def add_scene_args(ap):
    ap.add_argument('--data', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=80)
    ap.add_argument('--floor', choices=['vel', 'none'], default='vel')
    ap.add_argument('--cfg_bc', action='store_true', help='config 의 cuboid 고정 경계 (ficus 화분)')
    ap.add_argument('--region_density', action='store_true')
    ap.add_argument('--pg_vdamp', action='store_true', help='(받기만 함 -- 베이스라인은 자기 감쇠 규약을 씀)')
    ap.add_argument('--warm', type=int, default=2, help='시간 평균에서 뺄 첫 프레임 수')
    ap.add_argument('--dump', action='store_true')
    ap.add_argument('--dump_F', action='store_true')

def load_scene(a, dev='cpu'):
    d = torch.load(a.data, map_location='cpu', weights_only=False)
    cfg = d['cfg']
    T = int(d['x'].shape[0]) - 1
    L = min(a.frames, T)
    x0 = d['x'][0].double()
    ng = int(cfg.get('n_grid', 100))
    dx = float(cfg.get('grid_lim', 2.0)) / ng
    vi = (x0 / dx).long().clamp(0, ng - 1)
    fl = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
    cn = torch.zeros(ng ** 3, dtype=torch.float64).index_add_(0, fl, torch.ones(x0.shape[0], dtype=torch.float64))
    rho = torch.full((x0.shape[0],), float(cfg['density']), dtype=torch.float64)
    if a.region_density:
        for r in cfg.get('additional_material_params') or []:
            if float(r.get('E', cfg['E'])) != float(cfg['E']) or float(r.get('nu', cfg['nu'])) != float(cfg['nu']):
                raise SystemExit(f'--region_density: E/nu 가 다른 영역은 없음 ({r})')
            q = torch.as_tensor(r['point'], dtype=torch.float64)
            sz = torch.as_tensor(r['size'], dtype=torch.float64)
            rho[((x0 > q - sz) & (x0 < q + sz)).all(-1)] = float(r['density'])
    vol = dx ** 3 / cn[fl]
    FZ = None
    if a.floor == 'vel':
        pl = [b for b in cfg['boundary_conditions'] if b.get('type') == 'surface_collider'][0]
        assert abs(float(pl['normal'][2]) - 1.0) < 1e-06, '바닥 법선이 z 가 아님'
        FZ = float(pl['point'][2])
    (BOX, BOXW, MOV) = parse_bcs(cfg, L * float(cfg['frame_dt'])) if a.cfg_bc else ([], [], [])
    ROT = []
    if a.cfg_bc:
        _grp = {}
        for bc in cfg.get('boundary_conditions') or []:
            if bc.get('type') == 'enforce_particle_velocity_rotation':
                nv_ = torch.as_tensor(bc['normal'], dtype=torch.float64)
                nv_ = (nv_ / nv_.norm()).tolist()
                k_ = (tuple((float(q) for q in bc['point'])), tuple(nv_), float(bc['half_height_and_radius'][0]), float(bc['half_height_and_radius'][1]))
                _grp.setdefault(k_, []).append((float(bc['rotation_scale']), float(bc['translation_scale']), float(bc.get('start_time', 0.0)), float(bc.get('end_time', 1000.0))))
        ROT = [(k_[0], list(k_[1]), k_[2], k_[3], w_) for (k_, w_) in _grp.items()]
    g = [float(q) for q in cfg['g']]
    assert abs(g[0]) < 1e-12 and abs(g[1]) < 1e-12 and (g[2] <= 0), f'중력이 -z 또는 0 이 아님 {g}'
    S = dict(d=d, cfg=cfg, L=L, x0=x0, X=d['x'][:L + 1], v0=d['v'][0].double(), vol=vol, rho=rho, mass=vol * rho, EXT=float((x0.max(0).values - x0.min(0).values).norm()), FZ=FZ, BOX=BOX, BOXW=BOXW, MOV=MOV, ROT=ROT, g=g, frame_dt=float(cfg['frame_dt']), E=float(cfg['E']), nu=float(cfg['nu']))
    return S

def parse_bcs(cfg, horizon):
    """config 고정·구동 경계 -> (BOX 상시 고정 [(점, 크기)], BOXW 구간 고정 [(점, 크기, 시작, 끝)], MOV 구간 등속 구동 [(점, 크기, 시작, 끝, 속도)]).
    상자 안 판정은 처음 위치 (PG selection_enforce_particle_velocity_translation 은 경계 추가 때 = 0 프레임 입자 위치로 마스크).
      cuboid (격자 속도 0): 예전 그대로 (구간이 롤아웃 전체면 BOX, 아니면 BOXW).
      enforce_particle_translation: 속도 0 이고 구간이 롤아웃 전체면 BOX (예전 그대로), 속도 0 구간 -> BOXW,
        [blx] 속도 ≠ 0 -> MOV (bread: 위·아래 띠를 ±0.2 m/s 로 당김, 아래 띠는 2.2 s 까지).
      [blx] release_particles_sequentially (wolf): PG mpm_solver_warp.release_particles_sequentially 와 같은 50 개 속도 0 상자
        (법선 축: 중심 end_position, 반폭 |start−end|/50 x (50−i), 다른 축: 중심 1 반폭 1; 구간 [start_time, end_time/50 x (i+1))) -> BOXW 50 개."""
    (BOX, BOXW, MOV) = ([], [], [])
    for bc in cfg.get('boundary_conditions') or []:
        ty = bc.get('type')
        if ty == 'cuboid':
            assert max((abs(float(q)) for q in bc['velocity'])) == 0.0
            w_ = cuboid_window(bc, horizon)
            if w_ is None:
                BOX.append(([float(q) for q in bc['point']], [float(q) for q in bc['size']]))
            else:
                BOXW.append(([float(q) for q in bc['point']], [float(q) for q in bc['size']], w_[0], w_[1]))
        elif ty == 'enforce_particle_translation':
            (pt, sz) = ([float(q) for q in bc['point']], [float(q) for q in bc['size']])
            vel = [float(q) for q in bc['velocity']]
            (t0, t1) = (float(bc.get('start_time', 0.0)), float(bc.get('end_time', 1000.0)))
            if max((abs(q) for q in vel)) == 0.0:
                if t0 <= 0.0 and t1 > horizon:
                    BOX.append((pt, sz))
                else:
                    BOXW.append((pt, sz, t0, t1))
            else:
                MOV.append((pt, sz, t0, t1, vel))
        elif ty == 'release_particles_sequentially':
            nl = 50
            nv = [float(q) for q in bc['normal']]
            (pt, sz, ax) = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], -1)
            for k in range(3):
                if nv[k] == 0:
                    (pt[k], sz[k]) = (1.0, 1.0)
                else:
                    (ax, pt[k]) = (k, float(bc['end_position']))
            hl = abs(float(bc['start_position']) - float(bc['end_position'])) / nl
            etp = float(bc['end_time']) / nl
            for i in range(nl):
                s2 = list(sz)
                s2[ax] = hl * (nl - i)
                BOXW.append((list(pt), s2, float(bc['start_time']), etp * (i + 1)))
    return (BOX, BOXW, MOV)

def plastic_params(cfg, E=None):
    """[blx] 재질 확장 (원 논문은 탄성 전용): PG config -> 소성 매개변수. E 는 각 방법이 논문 절차로 맞춘 값이 있으면 그것 (반환 사상의 mu·lam), 없으면 config.
    PG mpm_solver_warp 기본값: softening 0.1, hardening 0, xi 0, friction_angle 25 (sin 은 3.14159265 로)."""
    mat = cfg.get('material', 'jelly')
    E = float(cfg['E']) if E is None else float(E)
    nu = float(cfg['nu'])
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    phi = float(cfg.get('friction_angle', 25.0))
    sp = math.sin(phi / 180.0 * 3.14159265)
    alpha = math.sqrt(2.0 / 3.0) * 2.0 * sp / (3.0 - sp)
    ys = float(cfg.get('yield_stress', 0.0))
    return dict(material=mat, E=E, nu=nu, mu=mu, lam=lam, friction_angle=phi, alpha=alpha, cdp=(3.0 * lam + 2.0 * mu) / (2.0 * mu) * alpha, yield_stress=ys, softening=float(cfg.get('softening', 0.1)), hardening=int(cfg.get('hardening', 0)), eps_y=math.sqrt(1.5) * ys / E if E > 0 else 0.0)

def spring_plastic_update(cur, rest, pp, tr=None):
    """[blx] 스프링 소성 휴지 길이 (표준 방식: 변형률이 항복을 넘으면 휴지 길이가 따라감). cur·rest 같은 모양 텐서, 로그 변형률 ε = log(cur/rest).
      plasticine (PG von Mises): |ε| > ε_y 이면 rest = cur exp(−sign(ε) ε_y)  (ε_y = sqrt(3/2) σy / E; 단축 응력 환산).
      sand (PG Drucker–Prager, 스프링 근사 -- 정확히 옮길 수 없음): tr = 스프링 양 끝 점의 체적 변형률 추정 (점마다 이어진 스프링 ε 평균 x 3) 의 평균,
        축 편차 e = ε − tr/3, |ε̂| ≈ sqrt(3/2)|e| (축대칭 상태), δγ = sqrt(3/2)|e| + cdp tr.
        δγ ≤ 0: 그대로 / δγ > 0 · tr > 0 (인장): rest = cur (응력 0, PG F_e = U Vᵀ) / δγ > 0 · tr ≤ 0: e 를 δγ 만큼 줄임 (PG H = ε − ε̂ δγ/|ε̂|) -> rest = cur exp(−(tr/3 + e'))."""
    eps = torch.log(cur.clamp(min=1e-12) / rest.clamp(min=1e-12))
    if pp['material'] == 'plasticine':
        ey = pp['eps_y']
        if ey <= 0:
            return rest
        over = eps.abs() > ey
        return torch.where(over, cur * torch.exp(-torch.sign(eps) * ey), rest)
    if pp['material'] == 'sand':
        e = eps - tr / 3.0
        en = math.sqrt(1.5) * e.abs()
        dg = en + pp['cdp'] * tr
        e2 = e * (1.0 - dg / en.clamp(min=1e-30))
        r_c = cur * torch.exp(-(tr / 3.0 + e2))
        r_t = cur
        return torch.where(dg <= 0, rest, torch.where(tr > 0, r_t, r_c))
    return rest

def cuboid_window(bc, horizon):
    """[final_pa_bl] PG cuboid 의 시간 구간 (PhysGaussian mpm_utils set_velocity_on_cuboid: start_time <= t < end_time 동안만 상자 안 격자 속도 = velocity,
    그 뒤 자유). 구간이 롤아웃 [0, horizon] 전체를 덮으면 None (= 예전처럼 상시 고정 -- 결과 불변), 아니면 (start, end).
    reset=1 (end 뒤 15 dt 동안 전체 격자 속도 0) 은 end 가 롤아웃 밖일 때만 허용."""
    (t0, t1) = (float(bc.get('start_time', 0.0)), float(bc.get('end_time', 1000.0)))
    if t0 <= 0.0 and t1 > horizon:
        return None
    assert int(bc.get('reset', 0)) == 0, f'구간 안에서 끝나는 reset cuboid 는 미지원 {bc}'
    return (t0, t1)

def box_active(t0, t1, t, eps=1e-06):
    """PG 조건 start <= t < end (t 는 프레임 시작 시각; 부동소수 오차 eps)."""
    return t0 <= t + eps and t + eps < t1

def time_scale(Kl, g):
    """닮음 변환 시간 배율 (절 S.3 공식 밖 (1)): 길이 Kl 배일 때 중력이 9.8 이 되게 c = sqrt(Kl |g| / 9.8).
    중력 0 (ficus 공식 config) 이면 맞출 중력이 없어 c = 1 (길이만 바꿈; 사용자 결정 11-05)."""
    gz = abs(float(g[2]))
    return (Kl * gz / 9.8) ** 0.5 if gz > 0 else 1.0

def rot_membership(x, rot):
    """회전 구동 원기둥 (처음 위치 기준) 안 입자 마스크 [(점, 법선, 반 높이, 반지름, 구간들)] -> 그룹마다 bool [N]."""
    out = []
    for (pt, nv, hh, rr, _w) in rot:
        n_ = torch.as_tensor(nv, device=x.device, dtype=x.dtype)
        off = x - torch.as_tensor(pt, device=x.device, dtype=x.dtype)
        ax = off @ n_
        rp = off - ax[:, None] * n_
        out.append((ax.abs() < hh) & (rp.norm(dim=-1) < rr))
    return out

def rot_traj(x0, rot, masks, L, h):
    """구동 입자의 프레임별 위치 [L+1, N, 3] (PG 정의: v = rs (r_perp x n) + ts n = 각속도 −rs n 의 강체 회전 + n 방향 이동, 프레임 [(f−1)h, f h) 와 겹친 몫 f 만큼 곱해 합침).
    프레임 f 위치는 f−1 위치에서 한 프레임 회전 (로드리게스) -- 우리 판 노드 구속·VR-GS 구동과 같은 프레임 평균 속도."""
    X = x0.unsqueeze(0).repeat(L + 1, 1, 1).clone()
    for ((pt, nv, hh, rr, wins), msk) in zip(rot, masks):
        n_ = torch.as_tensor(nv, dtype=x0.dtype)
        p_ = torch.as_tensor(pt, dtype=x0.dtype)
        for f in range(1, L + 1):
            t = (f - 1) * h
            rs = ts = 0.0
            for (r_, t_, t0, t1) in wins:
                fr = max(0.0, min(t + h, t1) - max(t, t0)) / h
                rs += r_ * fr
                ts += t_ * fr
            th = -rs * h
            r = X[f - 1][msk] - p_
            ax = (r @ n_)[:, None] * n_
            rp = r - ax
            rot_ = ax + rp * torch.cos(torch.tensor(th, dtype=x0.dtype)) + torch.cross(n_.expand_as(rp), rp, dim=-1) * torch.sin(torch.tensor(th, dtype=x0.dtype))
            X[f][msk] = p_ + rot_ + ts * h * n_
    return X

def in_boxes(x, boxes):
    """|x − point| < size (모든 축) -- vrgs_xpbd 고정 상자와 같은 판정."""
    m = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for b_ in boxes:
        (pt, sz) = (b_[0], b_[1])
        m |= ((x - torch.as_tensor(pt, device=x.device, dtype=x.dtype)).abs() < torch.as_tensor(sz, device=x.device, dtype=x.dtype)).all(-1)
    return m

class Roll:
    """프레임마다 예측 전 입자 위치 (와 F) 를 받아 오차·발산·시간·덤프를 모은다."""

    def __init__(s, a, S, N, extra=None):
        (s.a, s.S, s.N) = (a, S, N)
        (s.errs, s.ms, s.div) = ([], [], None)
        s.extra = dict(extra or {})
        s.P = torch.empty(S['L'], S['x0'].shape[0], 3) if a.dump else None
        s.PF = torch.empty(S['L'], S['x0'].shape[0], 3, 3) if a.dump and a.dump_F else None
        os.makedirs(a.out, exist_ok=True)

    def frame(s, f, x, F=None, ms=None):
        """f = 1..L, x [P,3] (예측 전 입자, 시뮬 좌표). 반환: 계속하면 True."""
        x = x.detach().to('cpu', torch.float64)
        gt = s.S['X'][f].double()
        e = float((x - gt).norm(dim=-1).mean()) / s.S['EXT'] if torch.isfinite(x).all() else float('nan')
        if not math.isfinite(e) or e > 2.0:
            s.div = f
            s.errs.append(float('nan'))
            return False
        s.errs.append(e)
        if ms is not None:
            s.ms.append(ms)
        if s.P is not None:
            s.P[f - 1] = x.float()
        if s.PF is not None:
            s.PF[f - 1] = torch.eye(3).expand(x.shape[0], 3, 3) if F is None else F.detach().to('cpu', torch.float32)
        if f % 10 == 0 or f == s.S['L']:
            print(f"  [프레임 {f}/{s.S['L']}] 오차 {100 * e:.3f}%" + (f' {ms:.2f} ms' if ms is not None else ''), flush=True)
        return True

    def fail(s, f, why):
        s.div = f
        s.errs.append(float('nan'))
        s.extra['fail'] = why
        print(f'  [실패] 프레임 {f}: {why}', flush=True)

    def finish(s):
        a = s.a
        sc = s.ms[a.warm:] if len(s.ms) > a.warm else s.ms
        fin = [q for q in s.errs if math.isfinite(q)]
        r = dict(tag=a.tag, N=s.N, frames=len(s.errs), diverged=s.div, err_mean=float(np.mean(s.errs)) if s.div is None else None, err_mean_done=float(np.mean(fin)) if fin else None, ms_mean=float(np.mean(sc)) if sc else None, ms_max=float(np.max(sc)) if sc else None, errs=s.errs, **s.extra)
        json.dump(r, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1, ensure_ascii=False)
        if s.P is not None and s.div is None:
            torch.save({'pred': s.P, 'F': s.PF}, os.path.join(a.out, 'pred.pt'))
        print(f'[끝] {a.tag} N {s.N}: ' + (f'발산 프레임 {s.div}' if s.div else f"오차 {100 * r['err_mean']:.3f}%") + (f", 평균 {r['ms_mean']:.2f} ms (최대 {r['ms_max']:.2f})" if r['ms_mean'] is not None else ''), flush=True)
        return r

def sync_ms(t0):
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0)
import glob
import pickle
import random
import shutil
import sys
SGR_EVERY = int(os.environ.get('SG_RESUME_EVERY', '5'))

def _sgr_log(s):
    print(f"[sg_resume {time.strftime('%F %T')}] {s}", flush=True)

def _sgr_rng_get():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

def _sgr_rng_set(r):
    random.setstate(r['py'])
    np.random.set_state(r['np'])
    torch.set_rng_state(r['torch'])
    torch.cuda.set_rng_state_all(r['cuda'])

def _sgr_extra(sim):
    """state_dict 밖의 텐서·스칼라 속성 (공식 + stab_sg 덧붙임 af_*). 목록·None 은 구성 때 정해지는 값이라 뺌."""
    pk = set(dict(sim.named_parameters()).keys()) | set(dict(sim.named_buffers()).keys())
    out = {}
    for (k, v) in vars(sim).items():
        if k.startswith('_') or k in pk:
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().clone()
        elif isinstance(v, (int, float, bool)) and (not isinstance(v, np.generic)):
            out[k] = v
    return out

def _sgr_cands(FIT, dump_cfg, n_iter):
    c = []
    for n in ('cur', 'old'):
        f = f'{FIT}/sg_resume/{n}/checkpoint/state.pt'
        if os.path.exists(f):
            try:
                s = torch.load(f, map_location='cpu', weights_only=False)
                if s.get('dump_cfg') == dump_cfg:
                    c.append((s['iter'], 1, os.path.dirname(f), s))
                else:
                    _sgr_log(f'{f}: cfg 다름 -> 안 씀')
            except Exception as ex:
                _sgr_log(f'{f}: 읽기 실패 {ex!r}')
    for d in glob.glob(f'{FIT}/exp/*/checkpoints_dynamic/checkpoint/') if os.environ.get('SG_RESUME_LEGACY', '0') == '1' else []:
        need = ('train_param.pth.tar', 'Spring_Mass.pth.tar', 'random_state.pkl', 'gaussians.ply', 'anchors.pt', 'dy_n_step.json')
        if not all((os.path.exists(d + q) for q in need)):
            continue
        dc = os.path.join(d, '../../dump_cfg.yaml')
        if not (os.path.exists(dc) and open(dc).read() == dump_cfg):
            _sgr_log(f'옛 체크포인트 {d}: cfg 다름 -> 안 씀')
            continue
        try:
            ep = int(torch.load(d + 'train_param.pth.tar', map_location='cpu', weights_only=False)['epoch'])
        except Exception as ex:
            _sgr_log(f'{d}: train_param 읽기 실패 {ex!r}')
            continue
        c.append((ep, 0, d.rstrip('/'), None))
    c = [q for q in c if 0 < q[0]]
    return sorted(c, key=lambda q: (q[0], q[1]))[-1] if c else None

def sg_resume_install(T, FIT, arg, cfg, exp_dir):
    if os.environ.get('SG_RESUME', '1') == '0':
        _sgr_log('꺼짐 (SG_RESUME=0)')
        return
    dump_cfg = open(f'{exp_dir}/dump_cfg.yaml').read()
    n_iter = int(cfg.DYNAMIC.ITERATIONS)
    best = _sgr_cands(FIT, dump_cfg, n_iter)
    ctx = dict(start=1, load=None)
    if best is not None:
        (it, new, ck, st) = best
        arg.dy_reload = f'{ck}/Spring_Mass.pth.tar'
        fz = cfg.is_frozen() if hasattr(cfg, 'is_frozen') else False
        if fz:
            cfg.defrost()
        cfg.DYNAMIC.PRETRAINED = arg.dy_reload
        if fz:
            cfg.freeze()
        ctx.update(start=it + 1, load=(new, ck, st))
        _sgr_log(f"재개: 반복 {it} 끝난 저장점 {ck} ({('새 형식 정확' if new else '옛 체크포인트 근사 재개 (가우시안 Adam 모멘트만 새로)')}) -> 반복 {it + 1} 부터")
    else:
        _sgr_log('저장점 없음 -> 처음부터')
    (get_sim0, train_step0, etqdm0) = (T.get_simulator, T.train_step, T.etqdm)

    def get_simulator(*a, **k):
        (sim, g) = get_sim0(*a, **k)
        (ctx['sim'], ctx['g']) = (sim, g)
        return (sim, g)

    def save(it):
        (sim, g) = (ctx['sim'], ctx['g'])
        R = f'{FIT}/sg_resume'
        tmp = f'{R}/tmp/checkpoint'
        shutil.rmtree(f'{R}/tmp', ignore_errors=True)
        os.makedirs(tmp)
        torch.save(sim.state_dict(), f'{tmp}/Spring_Mass.pth.tar')
        x0 = g._xyz
        g._xyz = sim.init_xyz_all.detach().clone()
        g.save_ply(f'{tmp}/gaussians.ply')
        g._xyz = x0
        torch.save(sim.init_xyz.detach().clone().cpu(), f'{tmp}/anchors.pt')
        json.dump(sim.n_step, open(f'{tmp}/dy_n_step.json', 'w'))
        torch.save(dict(iter=it, n_step=sim.n_step, sim_opt=sim.optimizer.state_dict(), sim_sched=sim.scheduler.state_dict() if sim.scheduler is not None else None, g_opt=g.optimizer.state_dict() if g.optimizer is not None else None, rng=_sgr_rng_get(), extra=_sgr_extra(sim), dump_cfg=dump_cfg, t=time.time()), f'{tmp}/state.pt')
        if os.path.isdir(f'{R}/cur'):
            shutil.rmtree(f'{R}/old', ignore_errors=True)
            os.rename(f'{R}/cur', f'{R}/old')
        os.rename(f'{R}/tmp', f'{R}/cur')
        shutil.rmtree(f'{R}/old', ignore_errors=True)
        _sgr_log(f'저장 반복 {it} n_step {sim.n_step}')

    class Bar:

        def __init__(self, it_, **k):
            self.bar = etqdm0([i for i in it_ if i >= ctx['start']], **k)

        def __iter__(self):
            first = True
            for i in self.bar:
                if first and ctx.get('rng') is not None:
                    _sgr_rng_set(ctx.pop('rng'))
                if not first and (i - 1) % SGR_EVERY == 0:
                    save(i - 1)
                first = False
                yield i

        def __getattr__(self, n):
            return getattr(self.bar, n)

    def train_step(*a, **k):
        names = ('args', 'cfg', 'cfg_stage', 'stage', 'max_frame', 'simulator', 'scene', 'gaussians')
        kw = dict(zip(names, a), **k)
        if kw['stage'] != 'dynamic':
            return train_step0(*a, **k)
        (simulator, gaussians) = (kw['simulator'], kw['gaussians'])
        (ctx['sim'], ctx['g']) = (simulator, gaussians)
        if ctx['load'] is not None:
            (new, ck, st) = ctx['load']
            if new:
                simulator.optimizer.load_state_dict(st['sim_opt'])
                if st['sim_sched'] is not None and simulator.scheduler is not None:
                    simulator.scheduler.load_state_dict(st['sim_sched'])
                if st['g_opt'] is not None and gaussians.optimizer is not None:
                    gaussians.optimizer.load_state_dict(st['g_opt'])
                for (kk, v) in st['extra'].items():
                    cur = getattr(simulator, kk, None)
                    if isinstance(v, torch.Tensor):
                        if isinstance(cur, torch.Tensor) and cur.shape == v.shape and (not torch.equal(cur.detach().cpu(), v)):
                            _sgr_log(f'속성 {kk} 가 저장값과 다름 (최대 차 {float((cur.detach().cpu().double() - v.double()).abs().max()):.3g}) -> 저장값으로')
                        setattr(simulator, kk, v.to(cur.device if isinstance(cur, torch.Tensor) else 'cuda').requires_grad_(bool(cur.requires_grad) if isinstance(cur, torch.Tensor) else False))
                    else:
                        setattr(simulator, kk, v)
                simulator.n_step = st['n_step']
                ctx['rng'] = st['rng']
            else:
                tp = torch.load(f'{ck}/train_param.pth.tar', map_location='cpu', weights_only=False)
                simulator.optimizer.load_state_dict(tp['optimizer'])
                rs = pickle.load(open(f'{ck}/random_state.pkl', 'rb'))
                ctx['rng'] = dict(py=rs.random_rng_state, np=rs.numpy_rng_state, torch=rs.torch_rng_state, cuda=rs.torch_cuda_rng_state_all)
            _sgr_log(f"적재 끝: 반복 {ctx['start']} 부터, n_step {simulator.n_step}")
        T.etqdm = Bar
        ec0 = torch.cuda.empty_cache
        if os.environ.get('SG_NO_EMPTY_CACHE', '0') == '1':
            torch.cuda.empty_cache = lambda : None
            _sgr_log('torch.cuda.empty_cache 끔 (SG_NO_EMPTY_CACHE=1)')
        try:
            return train_step0(*a, **k)
        finally:
            T.etqdm = etqdm0
            torch.cuda.empty_cache = ec0
    T.get_simulator = get_simulator
    T.train_step = train_step
COMMANDS = ('phystwin', 'springgaus')
if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit('usage: python exe/baseline_spring.py {' + '|'.join(COMMANDS) + '} [options]')
    CMD = sys.argv.pop(1)
else:
    CMD = None
if CMD == 'phystwin':
    import argparse
    import glob
    import importlib.util
    import json
    import os
    import pickle
    import sys
    import time
    import types
    from unittest import mock
    if os.environ.get('WARP_CACHE_PATH'):
        os.environ['WARP_CACHE_PATH'] += '_phystwin'
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    ap.add_argument('--mode', choices=['fit', 'run'], required=True)
    ap.add_argument('--phystwin', default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/PhysTwin')
    ap.add_argument('--fit', default='', help='run: fit 출력 폴더')
    ap.add_argument('--N', type=int, default=0, help='run: 프레임당 서브스텝 (0 = 맞춤 때의 N0)')
    ap.add_argument('--nobs', type=int, default=11024)
    ap.add_argument('--cma_iter', type=int, default=20, help='optimize_cma.py 기본값')
    ap.add_argument('--train_iters', type=int, default=0, help='0 = real.yaml iterations (200)')
    ap.add_argument('--knn', type=int, default=16, help='가우시안 이동 이웃 (공식 데모 16)')
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    FIT = os.path.abspath(a.fit or a.out)
    os.environ.setdefault('WANDB_MODE', 'disabled')
    import numpy as np
    import torch
    S = load_scene(a)
    (L, x0) = (S['L'], S['x0'])
    P = x0.shape[0]
    ext_side = float((x0.max(0).values - x0.min(0).values).max())
    K = 0.25 / ext_side
    c = time_scale(K, S['g'])
    G0 = abs(S['g'][2]) == 0.0
    FDT = S['frame_dt'] * c
    if S['FZ'] is not None:
        Z0 = S['FZ']
    else:
        Z0 = float(S['X'][..., 2].min()) - 0.05 * ext_side
    N0 = max(1, round(FDT / 5e-05))

    def to_pt(x):
        """시뮬 좌표 -> PhysTwin 좌표."""
        y = x.double().clone()
        y[..., 2] -= Z0
        return y * K

    def from_pt(y):
        x = y.double() / K
        x[..., 2] += Z0
        return x
    g_ = torch.Generator().manual_seed(0)
    sel = torch.randperm(P, generator=g_)[:min(a.nobs, P)]
    Q0 = to_pt(x0[sel])
    key = torch.floor((Q0 - Q0.min(0).values) / 0.005).long()
    (seen, first) = (set(), [])
    for (i, k) in enumerate(map(tuple, key.tolist())):
        if k not in seen:
            seen.add(k)
            first.append(i)
    psel = sel[torch.as_tensor(first)]
    ctrl = in_boxes(x0[psel], S['BOX']) if S['BOX'] else torch.zeros(len(psel), dtype=torch.bool)
    RMASK = rot_membership(x0[psel], S['ROT']) if S['ROT'] else []
    for _m in RMASK:
        ctrl = ctrl | _m
    REAL = bool(ctrl.any())
    PINW = [(in_boxes(x0[psel][~ctrl], [b_]), b_[2], b_[3], (0.0, 0.0, 0.0)) for b_ in S['BOXW']]
    PINW += [(in_boxes(x0[psel][~ctrl], [b_]), b_[2], b_[3], tuple((float(q) * K / c for q in b_[4]))) for b_ in S['MOV']]
    PLAST = plastic_params(S['cfg']) if S['cfg'].get('material', 'jelly') in ('sand', 'plasticine') else None
    if PLAST is not None:
        assert not REAL, '소성 확장은 조작점 없는 synthetic 경로만 (wolf·bread)'
    (obj_idx, ctl_idx) = (psel[~ctrl], psel[ctrl])
    B = to_pt(S['X'][:, obj_idx])
    CP = to_pt(x0[ctl_idx])
    if RMASK:
        _xc = rot_traj(x0[ctl_idx], S['ROT'], [m_[ctrl].clone() for m_ in RMASK], L, S['frame_dt'])
        CPT = to_pt(_xc)
    else:
        CPT = CP[None].repeat(L + 1, 1, 1)
    vel = S['v0'][obj_idx] * (K / c)
    print(f"[PhysTwin] {a.tag} 입자 {P}, 질량점 {B.shape[1]} + 조작점 {CP.shape[0]} ({('real' if REAL else 'synthetic')} 경로), K {K:.4g} 시간 {c:.4g} 배, 프레임 {FDT:.5g} s (PhysTwin 단위) N0 {N0}, 바닥 z {Z0:.4g}, 초기 속력 최대 {float(vel.norm(dim=1).max()):.3g}", flush=True)
    print(f"[명령] {' '.join(sys.argv)}", flush=True)
    os.makedirs(FIT, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    DP = f'{FIT}/final_data.pkl' if REAL else f'{FIT}/pts.npy'
    if a.mode == 'fit':
        np.save(f'{FIT}/vel.npy', vel.float().numpy())
        if REAL:
            (T1, M) = (B.shape[0], B.shape[1])
            pickle.dump(dict(object_points=B.float().numpy(), object_colors=np.ones((T1, M, 3), np.float32), object_visibilities=np.ones((T1, M), bool), object_motions_valid=np.ones((T1, M), bool), controller_points=CPT.float().numpy(), surface_points=np.zeros((0, 3)), interior_points=np.zeros((0, 3))), open(DP, 'wb'))
        else:
            np.save(DP, B.float().numpy())

    class _Stub(types.ModuleType):

        def __getattr__(self, k):
            return mock.MagicMock(name=f'{self.__name__}.{k}')
    sys.path.insert(0, a.phystwin)
    os.chdir(a.phystwin)
    for _m in ['pynput', 'gaussian_splatting', 'gaussian_splatting.scene', 'gaussian_splatting.scene.gaussian_model', 'gaussian_splatting.scene.cameras', 'gaussian_splatting.gaussian_renderer', 'gaussian_splatting.utils', 'gaussian_splatting.utils.graphics_utils', 'gaussian_splatting.rotation_utils', 'gs_render']:
        sys.modules.setdefault(_m, _Stub(_m))
    try:
        import pyrender
    except Exception:
        sys.modules['pyrender'] = _Stub('pyrender')
    try:
        import wandb
    except Exception:
        _nop = lambda *x, **k: None
        sys.modules['wandb'] = types.SimpleNamespace(init=_nop, log=_nop, finish=_nop, Video=_nop)
    _sp = importlib.util.spec_from_file_location('gaussian_splatting.dynamic_utils', f'{a.phystwin}/gaussian_splatting/dynamic_utils.py')
    DU = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(DU)
    sys.modules['gaussian_splatting.dynamic_utils'] = DU

    class _IpdbStop(RuntimeError):
        pass

    def _no_trace(*x, **k):
        raise _IpdbStop('dynamic_utils 의 ipdb.set_trace (SVD 실패·det 이상)')
    sys.modules['ipdb'] = types.SimpleNamespace(set_trace=_no_trace)
    import open3d as _o3d
    if int(np.__version__.split('.')[0]) >= 2 and tuple((int(q) for q in _o3d.__version__.split('.')[:2])) < (0, 19):
        sys.exit(f'open3d {_o3d.__version__} + numpy {np.__version__}: the PhysTwin spring search (KDTreeFlann) segfaults with this pair. Use open3d >= 0.19 or numpy < 2.')
    _v3 = _o3d.utility.Vector3dVector
    _o3d.utility.Vector3dVector = lambda x: _v3(np.ascontiguousarray(np.asarray(x), dtype=np.float64))
    import warp as wp
    from qqtt.utils import cfg
    import qqtt.data.simple_data as _sd
    import qqtt.data.real_data as _rd
    _sd.SimpleData.visualize_data = lambda self, visualize=False: None
    _rd.RealData.visualize_data = lambda self, visualize=False, save_gt=True: None
    import qqtt.model.diff_simulator.spring_mass_warp as _smw
    if G0:

        @wp.kernel
        def _update_vel_from_force_g0(v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), masses: wp.array(dtype=wp.float32), dt: float, drag_damping: float, reverse_factor: float, v_new: wp.array(dtype=wp.vec3)):
            tid = wp.tid()
            v0 = v[tid]
            f0 = f[tid]
            m0 = masses[tid]
            drag_damping_factor = wp.exp(-dt * drag_damping)
            all_force = f0
            a = all_force / m0
            v1 = v0 + a * dt
            v2 = v1 * drag_damping_factor
            v_new[tid] = v2
        _smw.update_vel_from_force = _update_vel_from_force_g0
        print('[PhysTwin] config 중력 0 -> 공식 커널의 중력 9.8 항 뺀 커널로', flush=True)
    _PL = dict(sp=None)
    if PLAST is not None:

        @wp.kernel(enable_backward=False)
        def _pl_zero(acc: wp.array(dtype=float), cnt: wp.array(dtype=float)):
            tid = wp.tid()
            acc[tid] = 0.0
            cnt[tid] = 0.0

        @wp.kernel(enable_backward=False)
        def _pl_acc(x: wp.array(dtype=wp.vec3), springs: wp.array(dtype=wp.vec2i), rest: wp.array(dtype=float), nobj: int, acc: wp.array(dtype=float), cnt: wp.array(dtype=float)):
            tid = wp.tid()
            i1 = springs[tid][0]
            i2 = springs[tid][1]
            if i1 < nobj and i2 < nobj:
                e = wp.log(wp.max(wp.length(x[i2] - x[i1]), 1e-12) / wp.max(rest[tid], 1e-12))
                wp.atomic_add(acc, i1, e)
                wp.atomic_add(acc, i2, e)
                wp.atomic_add(cnt, i1, 1.0)
                wp.atomic_add(cnt, i2, 1.0)

        @wp.kernel(enable_backward=False)
        def _pl_upd(x: wp.array(dtype=wp.vec3), springs: wp.array(dtype=wp.vec2i), rest: wp.array(dtype=float), nobj: int, acc: wp.array(dtype=float), cnt: wp.array(dtype=float), mat: int, ey: float, cdp: float):
            tid = wp.tid()
            i1 = springs[tid][0]
            i2 = springs[tid][1]
            if i1 < nobj and i2 < nobj:
                l = wp.max(wp.length(x[i2] - x[i1]), 1e-12)
                r = wp.max(rest[tid], 1e-12)
                e = wp.log(l / r)
                if mat == 5:
                    if wp.abs(e) > ey:
                        sg = 1.0
                        if e < 0.0:
                            sg = -1.0
                        rest[tid] = l * wp.exp(-sg * ey)
                if mat == 2:
                    tr = 1.5 * (acc[i1] / wp.max(cnt[i1], 1.0) + acc[i2] / wp.max(cnt[i2], 1.0))
                    ed = e - tr / 3.0
                    en = 1.224744871391589 * wp.abs(ed)
                    dg = en + cdp * tr
                    if dg > 0.0:
                        if tr > 0.0:
                            rest[tid] = l
                        else:
                            e2 = ed * (1.0 - dg / wp.max(en, 1e-30))
                            rest[tid] = l * wp.exp(-(tr / 3.0 + e2))
        _nob = int((~ctrl).sum())
        _PL_BUF = {_nob: (wp.zeros(_nob, dtype=float, device='cuda:0'), wp.zeros(_nob, dtype=float, device='cuda:0'))}
        _PL_MAT = {'sand': 2, 'plasticine': 5}[PLAST['material']]
        print(f"[PhysTwin] 재질 확장 {PLAST['material']}: ε_y {PLAST['eps_y']:.4g} cdp {PLAST['cdp']:.4g} (마찰각 {PLAST['friction_angle']}, ν {PLAST['nu']})", flush=True)

    def _plastic_after(x_new):
        if PLAST is None or _PL['sp'] is None:
            return
        (springs, rest, nobj) = _PL['sp']
        n = int(x_new.shape[0])
        if n not in _PL_BUF:
            _PL_BUF[n] = (wp.zeros(n, dtype=float, device='cuda:0'), wp.zeros(n, dtype=float, device='cuda:0'))
        (acc, cnt) = _PL_BUF[n]
        if _PL_MAT == 2:
            wp.launch(_pl_zero, dim=n, inputs=[acc, cnt])
            wp.launch(_pl_acc, dim=int(springs.shape[0]), inputs=[x_new, springs, rest, int(nobj), acc, cnt])
        wp.launch(_pl_upd, dim=int(springs.shape[0]), inputs=[x_new, springs, rest, int(nobj), acc, cnt, _PL_MAT, float(PLAST['eps_y']), float(PLAST['cdp'])])
    if PLAST is not None and (not PINW):
        _int0 = _smw.integrate_ground_collision

        class _WpLaunchProxyP:

            def __getattr__(self, k):
                return getattr(wp, k)

            def launch(self, *args, **kw):
                k_ = kw.get('kernel', args[0] if args else None)
                if k_ is _smw.eval_springs:
                    _PL['sp'] = (kw['inputs'][5], kw['inputs'][6], kw['inputs'][4])
                r_ = wp.launch(*args, **kw)
                if k_ is _int0:
                    _plastic_after(kw['outputs'][0])
                return r_
        _smw.wp = _WpLaunchProxyP()
    if PLAST is not None:
        _init_sms0 = _smw.SpringMassSystemWarp.__init__

        def _init_sms(self, *x, **k):
            rl = k['init_rest_lengths'] if 'init_rest_lengths' in k else x[2]
            self.af_rest0 = rl.detach().clone()
            _init_sms0(self, *x, **k)
        _smw.SpringMassSystemWarp.__init__ = _init_sms
    if PINW:

        @wp.kernel
        def _integrate_ground_collision_pin(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), collide_elas: wp.array(dtype=float), collide_fric: wp.array(dtype=float), dt: float, reverse_factor: float, pin: wp.array(dtype=wp.int32), pinv: wp.array(dtype=wp.vec3), x_new: wp.array(dtype=wp.vec3), v_new: wp.array(dtype=wp.vec3)):
            tid = wp.tid()
            x0 = x[tid]
            v0 = v[tid]
            normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor
            x_z = x0[2]
            v_z = v0[2]
            next_x_z = (x_z + v_z * dt) * reverse_factor
            if next_x_z < 0.0 and v_z * reverse_factor < -0.0001:
                v_normal = wp.dot(v0, normal) * normal
                v_tao = v0 - v_normal
                v_normal_length = wp.length(v_normal)
                v_tao_length = wp.max(wp.length(v_tao), 1e-06)
                clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
                clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)
                v_normal_new = -clamp_collide_elas * v_normal
                a = wp.max(0.0, 1.0 - clamp_collide_fric * (1.0 + clamp_collide_elas) * v_normal_length / v_tao_length)
                v_tao_new = a * v_tao
                v1 = v_normal_new + v_tao_new
                toi = -x_z / v_z
            else:
                v1 = v0
                toi = 0.0
            xn = x0 + v0 * toi + v1 * (dt - toi)
            if pin[tid] != 0:
                v1 = pinv[tid]
                xn = x0 + v1 * dt
            x_new[tid] = xn
            v_new[tid] = v1
        _PIN = wp.zeros(int((~ctrl).sum()), dtype=wp.int32, device='cuda:0')
        _PINV = wp.zeros(int((~ctrl).sum()), dtype=wp.vec3, device='cuda:0')
        _PIN_ON = {}
        _PIN_LAST = [None]

        class _WpLaunchProxy:

            def __getattr__(self, k):
                return getattr(wp, k)

            def launch(self, *args, **kw):
                k_ = kw.get('kernel', args[0] if args else None)
                if k_ is _smw.eval_springs:
                    _PL['sp'] = (kw['inputs'][5], kw['inputs'][6], kw['inputs'][4])
                if k_ is _integrate_ground_collision_pin:
                    assert int(kw['dim']) == _PIN.shape[0], f"질량점 수 {kw['dim']} != 고정 배열 {_PIN.shape[0]}"
                    kw['inputs'] = list(kw['inputs']) + [_PIN, _PINV]
                r_ = wp.launch(*args, **kw)
                if k_ is _integrate_ground_collision_pin:
                    _plastic_after(kw['outputs'][0])
                return r_
        _smw.integrate_ground_collision = _integrate_ground_collision_pin
        _smw.wp = _WpLaunchProxy()
        _sct0 = _smw.SpringMassSystemWarp.set_controller_target

        def _set_controller_target(self, frame_idx, pure_inference=False):
            t = (frame_idx - 1) * S['frame_dt']
            key = tuple((i for (i, (m_, t0_, t1_, v_)) in enumerate(PINW) if box_active(t0_, t1_, t)))
            if _PIN_LAST[0] != key:
                if key not in _PIN_ON:
                    on = torch.zeros(_PIN.shape[0], dtype=torch.int32)
                    vv = torch.zeros(_PIN.shape[0], 3, dtype=torch.float32)
                    for i in key:
                        on[PINW[i][0]] = 1
                        vv[PINW[i][0]] = torch.tensor(PINW[i][3], dtype=torch.float32)
                    _PIN_ON[key] = (wp.from_torch(on.cuda().contiguous(), dtype=wp.int32), wp.from_torch(vv.cuda().contiguous(), dtype=wp.vec3))
                wp.copy(_PIN, _PIN_ON[key][0])
                wp.copy(_PINV, _PIN_ON[key][1])
                _PIN_LAST[0] = key
            return _sct0(self, frame_idx, pure_inference)
        _smw.SpringMassSystemWarp.set_controller_target = _set_controller_target
        print(f'[PhysTwin] 시간 구간 고정·구동: ' + ', '.join((f'{int(m_.sum())} 질량점 [{t0_}, {t1_}) s v {v_}' for (m_, t0_, t1_, v_) in PINW[:8])) + (' ...' if len(PINW) > 8 else ''), flush=True)
    if PLAST is not None:
        _sct1 = _smw.SpringMassSystemWarp.set_controller_target

        def _set_controller_target_pl(self, frame_idx, pure_inference=False):
            if int(frame_idx) == 1:
                wp.to_torch(self.wp_rest_lengths, requires_grad=False).copy_(self.af_rest0)
            return _sct1(self, frame_idx, pure_inference)
        _smw.SpringMassSystemWarp.set_controller_target = _set_controller_target_pl
    from qqtt.engine.cma_optimize_warp import OptimizerCMA
    from qqtt.engine.trainer_warp import InvPhyTrainerWarp
    import wandb
    wandb.log = lambda *x, **k: None
    wandb.Video = lambda *x, **k: None
    InvPhyTrainerWarp.visualize_sim = lambda self, *x, **k: None
    _ef = OptimizerCMA.error_func
    OptimizerCMA.error_func = lambda self, p, visualize=False, video_path=None: _ef(self, p, False, None)

    def set_cfg(N):
        cfg.load_from_yaml('configs/real.yaml')
        cfg.data_type = 'real' if REAL else 'synthetic'
        cfg.reverse_z = False
        cfg.FPS = 1.0 / FDT
        cfg.num_substeps = int(N)
        cfg.dt = FDT / int(N)

    def set_vel(sim):
        """(3) real 경로는 velocity_path 를 안 받음 -> 시뮬레이터 초기 속도 배열에 직접."""
        if REAL:
            wp.copy(sim.wp_init_velocities, wp.from_torch(vel.float().cuda().contiguous(), dtype=wp.vec3))
    VP = f'{FIT}/vel.npy'
    if a.mode == 'fit':
        set_cfg(N0)
        t0 = time.time()
        if not os.path.exists(f'{FIT}/optimizeCMA/optimal_params.pkl'):
            opt = OptimizerCMA(data_path=DP, base_dir=FIT, train_frame=L + 1, velocity_path=None if REAL else VP)
            if REAL:
                opt.init_velocities = vel.float().cuda()
            opt.optimize(max_iter=a.cma_iter)
            os.replace(f'{FIT}/optimal_params.pkl', f'{FIT}/optimizeCMA/optimal_params.pkl')
        optimal = pickle.load(open(f'{FIT}/optimizeCMA/optimal_params.pkl', 'rb'))
        print(f'[1 단계 CMA-ES] {optimal} ({time.time() - t0:.0f}s)', flush=True)
        set_cfg(N0)
        cfg.set_optimal_params(dict(optimal))
        if a.train_iters:
            cfg.iterations = a.train_iters
        tr = InvPhyTrainerWarp(data_path=DP, base_dir=FIT, train_frame=L + 1, velocity_path=None if REAL else VP)
        set_vel(tr.simulator)
        if not glob.glob(f'{FIT}/train/best_*.pth'):
            tr.train()
        best = sorted(glob.glob(f'{FIT}/train/best_*.pth'), key=lambda p: int(p.split('_')[-1].split('.')[0]))[-1]
        Y = torch.load(best, map_location='cpu')['spring_Y'].float()
        json.dump(dict(tag=a.tag, N0=N0, K=K, c=c, g0=G0, Z0=Z0, FDT=FDT, real=REAL, n_mass=int(B.shape[1]), n_ctrl=int(CP.shape[0]), optimal={k: float(v) for (k, v) in optimal.items()}, best=best, iterations=int(cfg.iterations), lr=float(cfg.base_lr), spring_Y_median=float(Y.median()), spring_Y_min=float(Y.min()), spring_Y_max=float(Y.max()), n_springs=int(Y.numel()), sec=time.time() - t0), open(f'{FIT}/fit.json', 'w'), indent=1)
        print(f'[2 단계 경사 학습] {best} 스프링 {Y.numel()} 강성 중앙 {float(Y.median()):.4g} ({time.time() - t0:.0f}s)\nPTFIT_OK', flush=True)
        sys.exit(0)
    fj = json.load(open(f'{FIT}/fit.json'))
    N = a.N or N0
    set_cfg(N)
    cfg.set_optimal_params(dict(fj['optimal']))
    tr = InvPhyTrainerWarp(data_path=DP, base_dir=a.out, train_frame=L + 1, velocity_path=None if REAL else VP, pure_inference_mode=True)
    ck = torch.load(fj['best'], map_location=cfg.device)
    sim = tr.simulator
    assert len(ck['spring_Y']) == sim.n_springs, '맞춤 때와 스프링 연결이 다름'
    sim.set_spring_Y(torch.log(ck['spring_Y']).detach().clone())
    sim.set_collide(ck['collide_elas'].detach().clone(), ck['collide_fric'].detach().clone())
    sim.set_collide_object(ck['collide_object_elas'].detach().clone(), ck['collide_object_fric'].detach().clone())
    set_vel(sim)
    dev = 'cuda'
    xyz = to_pt(x0).float().to(dev)
    quat = torch.zeros(P, 4, device=dev)
    quat[:, 0] = 1
    CPd = CPT.float().to(dev)
    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
    prev = torch.cat([wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).clone(), CPd[0]])
    rel = DU.get_topk_indices(prev, K=a.knn)
    widx = torch.cat([DU.knn_weights_sparse(prev, xyz[i:i + 4000], K=a.knn)[1] for i in range(0, P, 4000)])
    R = Roll(a, S, N, extra=dict(method='phystwin', N0=N0, dt=cfg.dt, real=REAL, n_mass=int(B.shape[1]), n_ctrl=int(CP.shape[0])))
    print(f'[run] N {N} dt {cfg.dt:.4g} (PhysTwin 단위), 스프링 {sim.n_springs}', flush=True)
    for f in range(1, L + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if REAL or PINW or PLAST is not None:
            sim.set_controller_target(f, pure_inference=True)
        if sim.object_collision_flag:
            sim.update_collision_graph()
        wp.capture_launch(sim.forward_graph)
        x = torch.cat([wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False), CPd[f]])
        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
        if not torch.isfinite(x).all():
            R.fail(f, '질량점 비유한')
            break
        try:
            w = DU.calc_weights_vals_from_indices(prev, xyz, widx)
            (xyz, quat, _) = DU.interpolate_motions_speedup(bones=prev, motions=x - prev, relations=rel, weights=w, weights_indices=widx, xyz=xyz, quat=quat)
        except Exception as ex:
            R.fail(f, f'가우시안 이동 실패 ({type(ex).__name__}: {ex})')
            break
        prev = x.clone()
        ms = sync_ms(t0)
        F = DU.quat2mat(quat) if a.dump_F else None
        if not R.frame(f, from_pt(xyz), F=F, ms=ms):
            break
    R.finish()
if CMD == 'springgaus':
    import argparse
    import glob
    import json
    import math
    import os
    import sys
    import time
    import types
    ap = argparse.ArgumentParser()
    add_scene_args(ap)
    ap.add_argument('--mode', choices=['prep', 'train', 'run'], required=True)
    ap.add_argument('--sg', default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/Spring-Gaus')
    ap.add_argument('--fit', required=True, help='학습 작업 폴더 (데이터·설정·checkpoints·exp)')
    ap.add_argument('--mv', default='', help='prep: evaluation.py multiview 출력')
    ap.add_argument('--N', type=int, default=0, help='run: n_step (0 = 학습 끝 n_step)')
    ap.add_argument('--smoke', type=int, default=0, help='prep: 연기 시험 -- 학습 반복 수만 아주 작게 (정적 300·속도 2·동역학 2·다듬기 10; 결과는 경로 확인용)')
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    FIT = os.path.abspath(a.fit)
    SG = os.path.abspath(a.sg)
    import numpy as np
    import torch
    S = load_scene(a)
    (L, x0) = (S['L'], S['x0'])
    os.makedirs(FIT, exist_ok=True)
    print(f"[명령] {' '.join(sys.argv)}", flush=True)
    XF = f'{FIT}/af_xf.json'
    if a.mode == 'prep':
        mvm = json.load(open(f'{a.mv}/mv_meta.json'))
        assert mvm['n_frames'] == L + 1, f"다시점 영상 프레임 수 {mvm['n_frames']} != L+1 {L + 1}"
        side = float((x0.max(0).values - x0.min(0).values).max())
        Ks = 0.4 / side
        c = time_scale(Ks, S['g'])
        ctr = [float(q) for q in mvm['ctr']]
        json.dump(dict(Ks=Ks, c=c, ctr=ctr, side=side), open(XF, 'w'), indent=1)
    xf = json.load(open(XF))
    (Ks, c, CTR) = (xf['Ks'], xf['c'], torch.tensor(xf['ctr'], dtype=torch.float64))

    def to_sg(x):
        return (x.double().cpu() - CTR) * Ks + 0.5

    def from_sg(y):
        return (y.double().cpu() - 0.5) / Ks + CTR
    DT = S['frame_dt'] * c
    OBJ = a.tag
    CFG = f'{FIT}/{OBJ}.yaml'
    BCJ = f'{FIT}/af_bc.json'
    if a.mode == 'prep':
        D = f'{FIT}/data/{OBJ}'
        os.makedirs(D, exist_ok=True)
        cams = json.load(open(f'{a.mv}/cams_sim.json'))
        for cm in cams:
            M = np.array(cm['c2w'])
            M[:3, 3] = (M[:3, 3] - np.array(xf['ctr'])) * Ks + 0.5
            cm['c2w'] = M.tolist()
            lk = f"{D}/{cm['camera']}"
            if not os.path.exists(lk):
                os.symlink(os.path.abspath(f"{a.mv}/{cm['camera']}"), lk)
        json.dump(cams, open(f'{D}/camera.json', 'w'), indent=1)
        Xs = to_sg(S['X'].reshape(-1, 3)).reshape(S['X'].shape)
        (lo, hi) = (Xs.reshape(-1, 3).min(0).values, Xs.reshape(-1, 3).max(0).values)
        pad = 0.1 * float((hi - lo).max())
        (lo, hi) = ((lo - pad).tolist(), (hi + pad).tolist())
        fix = bool(S['BOX']) or bool(S['ROT']) or bool(S['BOXW']) or bool(S['MOV'])
        if S['FZ'] is not None:
            gz = float(to_sg(torch.tensor([[0.0, 0.0, S['FZ']]]))[0, 2])
            zmin = Xs[..., 2].amin(dim=1)
            hit = next((f for f in range(1, L + 1) if float(zmin[f]) <= gz + 0.005), L)
        else:
            gz = float(Xs[..., 2].min()) - 0.05
            hit = L
        hit = max(2, min(hit, L))
        GV = '[0, 0, -9.8]' if abs(S['g'][2]) > 0 else '[0, 0, 0]'
        VIT = 2 if a.smoke else 100
        SMK = '' if not a.smoke else f'\nSTATIC:\n  ITERATIONS: 300\n  TEST_ITERATIONS: [300]\n  SAVE_ITERATIONS: [300]\n  CHECKPOINT_ITERATIONS: [300]\n'
        CK = f'{FIT}/checkpoints/{OBJ}'
        os.makedirs(CK, exist_ok=True)
        y = f"DEFAULT: {SG}/config/mpm_synthetic/default.yaml\nCHECKPOINTS_ROOT: {CK}\n\nDATA:\n  TYPE: MPM_Synthetic\n  DATA_ROOT: {FIT}/data\n  OBJ_NAME: {OBJ}\n  N_CAM: {len(cams)}\n  N_FRAME: {L + 1}\n  FRAME_ALL: {L + 1}\n  HIT_FRAME: {hit}\n  H: {cams[0]['height']}\n  W: {cams[0]['width']}\n  WHITE_BKG: False\n  XYZ_MIN: {[round(q, 5) for q in lo]}\n  XYZ_MAX: {[round(q, 5) for q in hi]}\n  BC: [[[0, 0, {gz:.6f}], [0, 0, 1]]]\n\n  DT: {DT:.8f}\n  GLOBAL_M: 1\n  GLOBAL_K: 1000\n  GLOBAL_DAMP: 0.1\n\n  EVAL_FREQ: {1.0 / DT:.6f}\n  EVAL_FRAME: {L + 1}\n\nVELOCITY:\n  G: {GV}\n  VIZ_INTERVAL: 999999\n" + (f'  ITERATIONS: {VIT}\n  SAVE_INTERVAL: 1\n' if a.smoke else '') + f'\nDYNAMIC:\n  G: {GV}\n  VIZ_INTERVAL: 999999\n' + ('  ITERATIONS: 2\n  ITER_REFINE: 10\n  SAVE_INTERVAL: 1\n' if a.smoke else '') + SMK
        open(CFG, 'w').write(y)
        bc = dict(fix_boxes_sg=[], v0=None)
        if fix:
            for (pt, sz) in S['BOX']:
                bc['fix_boxes_sg'].append((to_sg(torch.tensor([pt]))[0].tolist(), [float(q) * Ks for q in sz]))
            if S['BOXW']:
                bc['fix_boxes_sg_w'] = [(to_sg(torch.tensor([b_[0]]))[0].tolist(), [float(q) * Ks for q in b_[1]], b_[2], b_[3], c) for b_ in S['BOXW']]
            if S['MOV']:
                bc['mov_sg'] = [(to_sg(torch.tensor([b_[0]]))[0].tolist(), [float(q) * Ks for q in b_[1]], b_[2], b_[3], [float(q) * Ks / c for q in b_[4]], c) for b_ in S['MOV']]
            bc['rot_sg'] = [dict(pt=to_sg(torch.tensor([g_[0]]))[0].tolist(), n=g_[1], hh=g_[2] * Ks, rr=g_[3] * Ks, wins=g_[4], c=c) for g_ in S['ROT']]
            torch.save(dict(x0=to_sg(x0).float(), v0=(S['v0'] * (Ks / c)).float()), f'{FIT}/af_v0.pt')
            bc['v0'] = f'{FIT}/af_v0.pt'
            json.dump([0.0, 0.0, 0.0], open(f'{CK}/init_velocity_{VIT}.json', 'w'))
        json.dump(bc, open(BCJ, 'w'), indent=1)
        print(f'[prep] {CFG}\n{y}\n[BC] {bc}\nSGPREP_OK', flush=True)
        sys.exit(0)
    os.chdir(FIT)
    sys.path.insert(0, SG)
    GPU = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    BC = json.load(open(BCJ))
    import lib.models.spring_mass.Spring_Mass as SMM
    if BC['fix_boxes_sg'] or BC['v0'] or BC.get('rot_sg') or BC.get('fix_boxes_sg_w') or BC.get('mov_sg'):
        _V = torch.load(BC['v0'])
        (_init0, _step0) = (SMM.Spring_Mass.initialize, SMM.Spring_Mass.step)

        def _initialize(self, xyz):
            _init0(self, xyz)
            X0 = _V['x0'].to(xyz.device)
            V0 = _V['v0'].to(xyz.device)
            idx = torch.cat([torch.cdist(self.init_xyz[i:i + 256].float(), X0).topk(8, largest=False).indices for i in range(0, self.n_points, 256)])
            self.af_fix = in_boxes(self.init_xyz.float(), BC['fix_boxes_sg']) if BC['fix_boxes_sg'] else torch.zeros(self.n_points, dtype=torch.bool, device=xyz.device)
            self.af_rot = []
            for g_ in BC.get('rot_sg') or []:
                n_ = torch.tensor(g_['n'], device=xyz.device, dtype=torch.float32)
                p_ = torch.tensor(g_['pt'], device=xyz.device, dtype=torch.float32)
                off = self.init_xyz.float() - p_
                ax = off @ n_
                rp = off - ax[:, None] * n_
                m_ = (ax.abs() < g_['hh']) & (rp.norm(dim=-1) < g_['rr'])
                self.af_rot.append((m_, n_, p_, g_))
                self.af_fix = self.af_fix | m_
            self.af_nstep = 0
            self.af_fixw = [(in_boxes(self.init_xyz.float(), [b_[:2]]) & ~self.af_fix, b_[2], b_[3], b_[4]) for b_ in BC.get('fix_boxes_sg_w') or []]
            self.af_mov = [(in_boxes(self.init_xyz.float(), [b_[:2]]) & ~self.af_fix, b_[2], b_[3], torch.tensor(b_[4], device=xyz.device), b_[5]) for b_ in BC.get('mov_sg') or []]
            v = V0[idx].mean(1)
            v[self.af_fix] = 0.0
            for (m_, t0_, t1_, c_) in self.af_fixw:
                if box_active(t0_, t1_, 0.0):
                    v[m_] = 0.0
            for (m_, t0_, t1_, vv_, c_) in self.af_mov:
                if box_active(t0_, t1_, 0.0):
                    v[m_] = vv_.to(v.dtype)
            self.init_v = v.to(self.init_xyz.dtype)
            print(f'[ficus 처리] 앵커 {self.n_points}, 고정 {int(self.af_fix.sum())}, 구간 고정 {[(int(q[0].sum()), q[1], q[2]) for q in self.af_fixw]}, 초기 속력 최대 {float(v.norm(dim=1).max()):.3g}', flush=True)

        def _step(self, xyz, v, K, m, rebound_k, fric_k, damp, dt):
            xyz_in = xyz
            (xyz, v) = _step0(self, xyz, v, K, m, rebound_k, fric_k, damp, dt)
            fx = self.af_fix[:, None]
            vnew = torch.zeros_like(v)
            for (m_, n_, p_, g_) in self.af_rot:
                t_sim = self.af_nstep * float(dt) / g_['c']
                rs = sum((r_ for (r_, t_, t0, t1) in g_['wins'] if t0 <= t_sim < t1))
                ts = sum((t_ for (r_, t_, t0, t1) in g_['wins'] if t0 <= t_sim < t1))
                th = -rs * float(dt) / g_['c']
                off = xyz_in - p_
                ax = (off @ n_)[:, None] * n_
                rp = off - ax
                xr = p_ + ax + rp * math.cos(th) + torch.cross(n_.expand_as(rp), rp, dim=-1) * math.sin(th) + ts * float(dt) / g_['c'] * n_
                mm = m_[:, None]
                xyz = torch.where(mm, xr.to(xyz.dtype), xyz)
                vnew = torch.where(mm, ((xr - xyz_in) / float(dt)).to(v.dtype), vnew)
                fx = fx & ~mm
            t_w = self.af_nstep * float(dt)
            self.af_nstep += 1
            keep = self.af_fix[:, None]
            for (m_, _n, _p, _g) in self.af_rot:
                keep = keep & ~m_[:, None]
            if self.af_fixw:
                on = torch.zeros_like(self.af_fix)
                for (m_, t0_, t1_, c_) in self.af_fixw:
                    if box_active(t0_, t1_, t_w / c_):
                        on = on | m_
                if bool(on.any()):
                    keep = keep | on[:, None]
                    v = torch.where(on[:, None], torch.zeros_like(v), v)
            (xyz, v) = (torch.where(keep, self.init_xyz, xyz), torch.where(self.af_fix[:, None], vnew, v))
            for (m_, t0_, t1_, vv_, c_) in self.af_mov:
                if box_active(t0_, t1_, t_w / c_):
                    mm = m_[:, None]
                    xyz = torch.where(mm, (xyz_in + vv_ * float(dt)).to(xyz.dtype), xyz)
                    v = torch.where(mm, vv_.to(v.dtype).expand_as(v), v)
            return (xyz, v)
        _fwd0 = SMM.Spring_Mass.forward

        def _forward(self, xyz_all, xyz, v, frame_id, *x, **k):
            if self.af_fixw or self.af_mov:
                self.af_nstep = (int(frame_id) - 1) * int(self.n_step)
            return _fwd0(self, xyz_all, xyz, v, frame_id, *x, **k)
        SMM.Spring_Mass.forward = _forward
        SMM.Spring_Mass.initialize = _initialize
        SMM.Spring_Mass.step = _step
    PLAST = plastic_params(S['cfg']) if S['cfg'].get('material', 'jelly') in ('sand', 'plasticine') else None
    if PLAST is not None:
        (_cf0, _st1, _fw1) = (SMM.Spring_Mass.compute_force, SMM.Spring_Mass.step, SMM.Spring_Mass.forward)

        def _compute_force_pl(self, xyz, v, K, damp):
            r = getattr(self, 'af_rest', None)
            if r is None:
                return _cf0(self, xyz, v, K, damp)
            o = self.origin_len
            self.origin_len = r
            try:
                return _cf0(self, xyz, v, K, damp)
            finally:
                self.origin_len = o

        def _step_pl(self, xyz, v, K, m, rebound_k, fric_k, damp, dt):
            (xyz, v) = _st1(self, xyz, v, K, m, rebound_k, fric_k, damp, dt)
            with torch.no_grad():
                if getattr(self, 'af_rest', None) is None:
                    self.af_rest = self.origin_len.detach().clone()
                xd = xyz.detach()
                cur = torch.norm(xd[self.knn_index] - xd.unsqueeze(1), dim=2)
                tr = None
                if PLAST['material'] == 'sand':
                    ev = 3.0 * torch.log(cur.clamp(min=1e-12) / self.af_rest.clamp(min=1e-12)).mean(1)
                    tr = 0.5 * (ev[:, None] + ev[self.knn_index])
                self.af_rest = spring_plastic_update(cur, self.af_rest, PLAST, tr)
            return (xyz, v)

        def _forward_pl(self, xyz_all, xyz, v, frame_id, *x, **k):
            f = int(frame_id)
            if f == 1:
                self.af_rest = self.origin_len.detach().clone()
            elif getattr(self, 'af_snap', None) is not None and self.af_snap[0] == f:
                self.af_rest = self.af_snap[1].clone()
            if getattr(self, 'af_rest', None) is not None:
                self.af_snap = (f, self.af_rest.detach().clone())
            return _fw1(self, xyz_all, xyz, v, frame_id, *x, **k)
        SMM.Spring_Mass.compute_force = _compute_force_pl
        SMM.Spring_Mass.step = _step_pl
        SMM.Spring_Mass.forward = _forward_pl
        print(f"[Spring-Gaus] 재질 확장 {PLAST['material']}: ε_y {PLAST['eps_y']:.4g} cdp {PLAST['cdp']:.4g}", flush=True)
    import train as T
    if a.mode == 'train':
        sys.argv = ['train.py', '-g', GPU, '--cfg', CFG, '--exp_id', f'stab_{OBJ}', '--eval_cam', '0']
        arg = T.config_parser()
        cfg = T.get_config_merge_default(config_file=arg.cfg, arg=arg)
        os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu_id
        import open3d as o3d
        T.o3d = o3d
        T.setup_seed(cfg.SEED)
        recorder = T.Recorder(arg.exp_id, cfg, rank=0, time_f=time.time())
        T.recorder = recorder
        (T.cfg, T.arg) = (cfg, arg)
        exp_path = f'{recorder.exp_id}_{recorder.timestamp}'
        os.makedirs('checkpoints', exist_ok=True)
        static = os.path.exists(os.path.join(cfg.CHECKPOINTS_ROOT, 'static_gaussians'))
        scene = T.Scene(cfg, exp_path, shuffle=False, load_static=not static)
        T.scene = scene
        t0 = time.time()
        if not static:
            T.logger.info('Begin Train Static...')
            T.train_static(arg, cfg, scene, cfg.CHECKPOINTS_ROOT)
        if not os.path.exists(os.path.join(cfg.CHECKPOINTS_ROOT, f'init_velocity_{cfg.VELOCITY.ITERATIONS}.json')):
            T.logger.info('Begin Train Velocity...')
            T.train_velocity(arg, cfg, scene, recorder)
        T.eval_fitting = T.eval_prediction = lambda *x, **k: T.logger.info('[stab] 공식 평가 건너뜀')
        sg_resume_install(T, FIT, arg, cfg, f'{FIT}/exp/{exp_path}')
        T.train_dynamic(arg, cfg, scene, recorder)
        cks = sorted(glob.glob(f'{FIT}/exp/{exp_path}/checkpoints_dynamic/checkpoint/Spring_Mass.pth.tar'))
        assert cks, '동역학 체크포인트 없음'
        json.dump(dict(tag=OBJ, exp=f'{FIT}/exp/{exp_path}', dy_reload=cks[-1], n_step=json.load(open(f'{FIT}/exp/{exp_path}/checkpoints_dynamic/checkpoint/dy_n_step.json')), init_velocity=json.load(open(os.path.join(cfg.CHECKPOINTS_ROOT, f'init_velocity_{cfg.VELOCITY.ITERATIONS}.json'))), sec=time.time() - t0), open(f'{FIT}/sg_train.json', 'w'), indent=1)
        print(f'[학습 끝] {cks[-1]} ({time.time() - t0:.0f}s)\nSG_TRAIN_DONE', flush=True)
        sys.exit(0)
    tj = json.load(open(f'{FIT}/sg_train.json'))
    sys.argv = ['test.py', '-g', GPU, '--cfg', CFG, '--exp_id', f'stabrun_{OBJ}', '--dy_reload', tj['dy_reload']]
    arg = T.config_parser()
    cfg = T.get_config_merge_default(config_file=arg.cfg, arg=arg)

    class _Scene:
        """get_simulator 가 쓰는 set_gaussians 만 (합성 데이터 경로 = load_ply; 영상 적재 생략)."""

        def __init__(s, cfg):
            s.cfg = cfg

        def set_gaussians(s, gaussians, load_path=None):
            s.gaussians = gaussians
            gaussians.load_ply(load_path)
    load_velocity = json.load(open(os.path.join(cfg.CHECKPOINTS_ROOT, f'init_velocity_{cfg.VELOCITY.ITERATIONS}.json')))
    (sim, gaussians) = T.get_simulator(arg, cfg, _Scene(cfg), cfg_stage=cfg.DYNAMIC, init_velocity=load_velocity, load_g=None)
    sim.eval()
    NT = int(sim.n_step)
    N = a.N or NT
    dev = 'cuda'
    ours0 = to_sg(x0).float().to(dev)
    sim.set_all_particle(ours0)
    sim.stage = 'velocity'
    sim.n_step = N
    R = Roll(a, S, N, extra=dict(method='springgaus', n_step_train=NT, dt=sim.dt / N, n_anchor=int(sim.n_points), init_velocity=load_velocity, dy_reload=tj['dy_reload']))
    print(f'[run] 학습 n_step {NT} -> {N}, 앵커 {sim.n_points}, 입자 {ours0.shape[0]}, 프레임 간격 {sim.dt:.5g} (Spring-Gaus 단위)', flush=True)
    with torch.no_grad():
        (xyz_all, xyz, v) = (ours0.clone(), sim.init_xyz.detach().clone(), sim.init_v.detach().clone())
        for f in range(1, L + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            (xyz_all, xyz, v, is_nan) = sim(xyz_all, xyz, v, f)
            ms = sync_ms(t0)
            if bool(is_nan):
                R.fail(f, '공식 is_nan (앵커 NaN)')
                break
            if not R.frame(f, from_sg(xyz_all), F=None, ms=ms):
                break
    R.finish()
