"""Expressiveness experiment: tracking targets and the tracking comparison.

Subcommands (old script -> min_aux file -> new subcommand):
  track       exe/rep_track_sg.py   expressiveness/track.py       track the target flow with one deformation representation (see the comment
                                                                  at the top of the `track` block)
  flow        exe/gauss_flow.py     expressiveness/prep.py flow   target deformation from a sequence of single Gaussian velocity fields
  gf-target   exe/rep_prep_gf.py    expressiveness/prep.py gf-target  ground-truth simulator trajectory (sim_*.h5) -> tracking target
Helpers: GS-Verse binding (old fusedgs/gsverse.py), velocity fields (old exe/gauss_flow.py).
"""
from __future__ import annotations
import argparse
import math
import os
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib')
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
import numpy as np
import torch

def mesh_from_points(x, n_grid=100, grid_lim=2.0, iso=0.5, smooth=1):
    """채우기 입자에서 표면 메시를 뽑는다 (PG 채우기가 쓰는 marching cubes).

    반환: (정점 [V,3] 시뮬 좌표, 삼각형 [T,3])
    """
    import mcubes
    dx = grid_lim / n_grid
    idx = (x / dx).long().clamp(0, n_grid - 1)
    occ = torch.zeros(n_grid, n_grid, n_grid, device=x.device)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    for _ in range(max(smooth, 0)):
        occ = torch.nn.functional.avg_pool3d(occ[None, None], 3, stride=1, padding=1)[0, 0]
        occ = occ / occ.max().clamp_min(1e-12)
    (v, f) = mcubes.marching_cubes(occ.detach().cpu().numpy().astype(np.float32), float(iso))
    v = torch.from_numpy(np.ascontiguousarray(v)).float().to(x.device) * dx
    f = torch.from_numpy(np.ascontiguousarray(f.astype(np.int64))).to(x.device)
    ar = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=-1).norm(dim=-1) * 0.5
    keep = ar > 0.001 * float(ar.median())
    return (v, f[keep])

def _tri_frame(v, f):
    """삼각형의 국소 틀. 반환 (원점 v0 [T,3], e1, e2, n)."""
    (v0, v1, v2) = (v[f[:, 0]], v[f[:, 1]], v[f[:, 2]])
    (e1, e2) = (v1 - v0, v2 - v0)
    n = torch.cross(e1, e2, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (v0, e1, e2, n)

def bind_points(x, v, f, chunk=4096):
    """입자를 가장 가까운 삼각형에 묶는다. 반환 (삼각형 색인, a1, a2, b).

    x = v0 + a1 e1 + a2 e2 + b n 이 되도록 국소 틀에서 좌표를 푼다 (정확히
    복원되므로 t=0 에서 위치가 한 비트도 달라지지 않는다).
    """
    (v0, e1, e2, n) = _tri_frame(v, f)
    ctr = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0
    A = torch.stack([e1, e2, n], -1).double()
    ti = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for s in range(0, x.shape[0], chunk):
        e = min(s + chunk, x.shape[0])
        ti[s:e] = torch.cdist(x[s:e], ctr).argmin(1)
    loc = torch.linalg.solve(A[ti], (x - v0[ti]).double().unsqueeze(-1))
    loc = loc.squeeze(-1).to(x.dtype)
    return (ti, loc[:, 0], loc[:, 1], loc[:, 2])
KINDS = ['smooth', 'split', 'point_src', 'line_src', 'twist_tear', 'sink']
SINK_EPS = 0.02
STEP_DET_MIN = float('inf')

def field_seq(seed, n_win, c_lo=0.2, c_hi=0.8, sig=(0.15, 0.35), amp=(0.3, 0.8), kinds=None):
    """창별 (c3, d3, σ, A, 종류). 시드가 같으면 형상과 무관하게 같다.

    kinds 를 주면 창마다 그 종류를 차례로 돈다 (기본: 부드러운 가우시안만).
    """
    r = np.random.default_rng(seed)
    out = []
    for w in range(n_win):
        c = r.uniform(c_lo, c_hi, 3)
        d = r.normal(size=3)
        d /= np.linalg.norm(d)
        k = KINDS.index(kinds[w % len(kinds)]) if kinds else 0
        out.append(np.concatenate([c, d, [r.uniform(*sig), r.uniform(*amp), k]]))
    return np.array(out, dtype=np.float64)

def velocity(x, f):
    """창 f 의 속도 v(x) (x: [N,3])."""
    (c, d, sg, A) = (f[:3], f[3:6], f[6], f[7])
    kind = KINDS[int(round(float(f[8])))] if f.shape[0] > 8 else 'smooth'
    dx = x - c
    r2 = (dx * dx).sum(1)
    g = torch.exp(-r2 / (sg * sg))
    if kind == 'smooth':
        return A * g[:, None] * d[None]
    if kind == 'split':
        return A * (g * torch.sign(dx @ d))[:, None] * d[None]
    if kind == 'point_src':
        n = dx / r2.sqrt().clamp_min(1e-09)[:, None]
        return A * g[:, None] * n
    if kind == 'line_src':
        xp = dx - (dx @ d)[:, None] * d[None]
        rp2 = (xp * xp).sum(1)
        n = xp / rp2.sqrt().clamp_min(1e-09)[:, None]
        return A * torch.exp(-rp2 / (sg * sg))[:, None] * n
    if kind == 'twist_tear':
        rot = torch.cross(d.expand_as(dx), dx, dim=1) / sg
        return A * (g * torch.sign(dx @ d))[:, None] * rot
    if kind == 'sink':
        r = r2.sqrt()
        n = dx / r.clamp_min(1e-09)[:, None]
        cut = (r / SINK_EPS).clamp(0.0, 1.0)
        return -A * (g * cut)[:, None] * n
    raise ValueError(kind)

def vel_and_grad(x, f):
    """v(x) 와 ∇v(x) -- 자동미분 (끊기는 곳 밖에서 정확)."""
    with torch.enable_grad():
        xx = x.detach().requires_grad_(True)
        v = velocity(xx, f)
        J = torch.stack([torch.autograd.grad(v[:, i].sum(), xx, retain_graph=i < 2)[0] for i in range(3)], 1)
    return (v.detach(), J.detach())

def dt_limit(f, safety):
    """Δt < σ²/A (흡입은 중심을 넘어가지 않게 Δt < ε/A 도)."""
    (sg, A) = (float(f[6]), float(f[7]))
    dt = safety * sg * sg / A
    if f.shape[0] > 8 and KINDS[int(round(float(f[8])))] == 'sink':
        dt = min(dt, safety * SINK_EPS / A)
    return dt

def advance(x, F, f, T, safety=0.5):
    """창 f 로 시간 T 만큼 RK4 적분 (Δt = safety·σ²/A 이하). F 도 함께."""
    dt_max = dt_limit(f, safety)
    n = max(1, math.ceil(T / dt_max))
    dt = T / n
    global STEP_DET_MIN
    for _ in range(n):
        Fi = torch.eye(3, dtype=x.dtype, device=x.device).expand(x.shape[0], 3, 3)

        def rhs(xx, FF):
            (v, J) = vel_and_grad(xx, f)
            return (v, J @ FF)
        (k1x, k1F) = rhs(x, F)
        (k2x, k2F) = rhs(x + 0.5 * dt * k1x, F + 0.5 * dt * k1F)
        (k3x, k3F) = rhs(x + 0.5 * dt * k2x, F + 0.5 * dt * k2F)
        (k4x, k4F) = rhs(x + dt * k3x, F + dt * k3F)
        (_, s1) = rhs(x, Fi)
        (_, s2) = rhs(x + 0.5 * dt * k1x, Fi + 0.5 * dt * s1)
        (_, s3) = rhs(x + 0.5 * dt * k2x, Fi + 0.5 * dt * s2)
        (_, s4) = rhs(x + dt * k3x, Fi + dt * s3)
        Fstep = Fi + dt / 6.0 * (s1 + 2 * s2 + 2 * s3 + s4)
        STEP_DET_MIN = min(STEP_DET_MIN, float(torch.linalg.det(Fstep).min()))
        x = x + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        F = F + dt / 6.0 * (k1F + 2 * k2F + 2 * k3F + k4F)
    return (x, F, n, dt)

def run_flow(x0, field, frames, period, frame_dt, safety=0.5, keep_F=False):
    """x0 [N,3] 텐서를 frames 프레임 흘린다 -> traj [T+1,N,3], F, 기록."""
    x = x0.clone()
    F = torch.eye(3, dtype=x.dtype, device=x.device).expand(x.shape[0], 3, 3).clone()
    (traj, Fs, log) = ([x.clone()], [F.clone()] if keep_F else None, [])
    for t in range(frames):
        f = torch.as_tensor(field[t // period], dtype=x.dtype, device=x.device)
        (x, F, n, dt) = advance(x, F, f, frame_dt, safety)
        traj.append(x.clone())
        if keep_F:
            Fs.append(F.clone())
        det = torch.linalg.det(F)
        log.append((t, n, dt, float(det.min()), float(det.max())))
    return (torch.stack(traj), torch.stack(Fs) if keep_F else None, log)
COMMANDS = ('track', 'flow', 'gf-target')
if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit('usage: python exe/expressiveness.py {' + '|'.join(COMMANDS) + '} [options]')
    CMD = sys.argv.pop(1)
else:
    CMD = None
if CMD == 'flow':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fill', required=True, help='PG 공식 채우기 입자 (npy) -- 정규화 기준')
    ap.add_argument('--pts', default='', help='흘릴 점을 따로 준다 (rep_aux 가 만든 정규화된 가우시안 중심 npy). 주면 채우기 입자 대신 이 점들이 목표 입자다')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=30000, help='부분표본 입자 수 (0 이면 전부)')
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--period', type=int, default=10, help='가우시안 교체 주기(프레임)')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--seed', type=int, default=0, help='속도장 시드 (형상 공통)')
    ap.add_argument('--sub_seed', type=int, default=0, help='부분표본 시드')
    ap.add_argument('--safety', type=float, default=0.5, help='Δt = safety·σ²/A')
    ap.add_argument('--save_F', action='store_true')
    ap.add_argument('--amp', type=float, nargs=2, default=[2.0, 5.0], help='세기 A 범위 (정규화 단위/초)')
    ap.add_argument('--sig', type=float, nargs=2, default=[0.1, 0.3], help='폭 σ 범위')
    ap.add_argument('--kinds', default=','.join(KINDS), help="창마다 돌아가며 쓸 장의 종류 (쉼표). 'smooth' 만이면 위상 보존")
    a = ap.parse_args()
    X = np.load(a.fill).astype(np.float64)
    lo = X.min(0)
    s = float((X.max(0) - lo).max())
    Xn = (X - lo) / s
    Xn = Xn + (1.0 - Xn.max(0)) / 2.0
    if a.pts:
        Xn = np.load(a.pts).astype(np.float64)
    r = np.random.default_rng(a.sub_seed)
    idx = np.arange(len(Xn)) if a.n <= 0 else np.sort(r.choice(len(Xn), min(a.n, len(Xn)), replace=False))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    x0 = torch.as_tensor(Xn[idx], dtype=torch.float64, device=dev)
    field = field_seq(a.seed, (a.frames + a.period - 1) // a.period, sig=tuple(a.sig), amp=tuple(a.amp), kinds=a.kinds.split(','))
    (traj, Fs, log) = run_flow(x0, field, a.frames, a.period, 1.0 / a.fps, a.safety, keep_F=a.save_F)
    dmin = min((q[3] for q in log))
    nsub = [q[1] for q in log]
    disp = float((traj[-1] - traj[0]).norm(dim=1).mean())
    print(f'[흐름] 입자 {len(idx)}  프레임 {a.frames}  주기 {a.period}  창 {len(field)}  서브스텝 {min(nsub)}~{max(nsub)}  det F 최소 {dmin:.4f}  평균 변위 {disp:.4f} (정규화 단위)', flush=True)
    print(f'[스텝] 한 스텝 사상 det 최소 {STEP_DET_MIN:.4f} (모든 입자·모든 스텝)', flush=True)
    assert dmin > 0 and STEP_DET_MIN > 0, 'det <= 0 -- 흐름이 접혔다'
    np.savez_compressed(a.out, X0=Xn[idx].astype(np.float32), traj=traj.cpu().numpy().astype(np.float32), idx=idx, lo=lo, s=s, field=field, **{'F': Fs.cpu().numpy().astype(np.float32)} if a.save_F else {})
    print(f'[저장] {a.out}', flush=True)
if CMD == 'gf-target':
    import argparse, glob, os
    import h5py, numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5_dir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--frames', type=int, default=0, help='0 이면 있는 프레임 전부')
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(a.h5_dir, 'sim_*.h5')))
    if a.frames:
        fs = fs[:a.frames + 1]

    def rd(f):
        with h5py.File(f, 'r') as h:
            return np.asarray(h['x'], dtype=np.float64).T
    x0 = rd(fs[0])
    N = x0.shape[0]
    assert np.isfinite(x0).all(), '프레임 0 에 비유한 입자'
    lo = x0.min(0)
    s = float((x0.max(0) - lo).max())
    off = (1.0 - (x0.max(0) - lo) / s) / 2.0
    tr = np.empty((len(fs), N, 3), dtype=np.float32)
    bad = 0
    for (t, f) in enumerate(fs):
        x = rd(f)
        assert x.shape[0] == N, f'{f}: 입자 수 {x.shape[0]} != {N}'
        bad += int((~np.isfinite(x)).any(1).sum())
        tr[t] = ((x - lo) / s + off).astype(np.float32)
    os.makedirs(a.out_dir, exist_ok=True)
    G = tr[0].copy()
    np.savez(f'{a.out_dir}/gflow_{a.name}.npz', X0=G, traj=tr, idx=np.arange(N), lo=lo, s=s, field=np.zeros((0, 9)))
    np.savez(f'{a.out_dir}/aux_{a.name}.npz', G=G, gidx=np.arange(N), lo=lo, s=s, off=off, scale_origin=1.0, mean=np.zeros(3), model='', opacity_threshold=0.0)
    print(f'[gf] {a.name}: 입자 {N}  프레임 {len(fs) - 1}  비유한 값 {bad}  bbox 긴 축 {s:.4f}  이동 최대 {float(np.abs(tr[-1] - tr[0]).max()):.3f}', flush=True)
if CMD == 'track':
    import argparse
    import json
    import math
    import os
    import sys
    import time
    import numpy as np
    import torch
    import gsrender as sx
    import baseline_vrgs as vx
    from gsrender import _barrier_b as _barrier_b
    ap = argparse.ArgumentParser()
    ap.add_argument('--flow', required=True)
    ap.add_argument('--aux', required=True)
    ap.add_argument('--fill', default='', help='(쓰지 않는다 -- 내부 채움 없는 3DGS 만)')
    ap.add_argument('--method', required=True, choices=['ours', 'tet', 'phystwin', 'gaussim', 'simplicits', 'vrgs', 'hex', 'vrgsx', 'springgaus'])
    ap.add_argument('--hex_dx', type=float, default=0.02, help='hex: 씬 MPM dx (시뮬 좌표). repflow 세 형상 = PG 격자 dx 2/100 (기존 ours 와 같은 dx)')
    ap.add_argument('--vmax', type=int, default=30000, help='vrgsx: 케이지 정점 상한 (VR-GS 논문 10K~30K, 우리 구현 기본)')
    ap.add_argument('--hex_cell', type=float, default=1.0, help='hex: 칸 = cell·dx (실시간 판 --cell, 기본 1)')
    ap.add_argument('--ckpt_every', type=int, default=10, help='이어 돌리기 체크포인트 간격 (프레임, 0 이면 끔) -- <out>.ckpt.pt')
    ap.add_argument('--resume', action='store_true', help='<out>.ckpt.pt 에서 이어 돌린다 (영상은 이어 쓴 조각을 끝에 합친다)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--video', default='')
    ap.add_argument('--simp', default='', help='simplicits: 학습된 가중치 함수 (.pt)')
    ap.add_argument('--max_iter', type=int, default=500)
    ap.add_argument('--kappa', type=float, default=1.0)
    ap.add_argument('--jhat', type=float, default=0.3)
    ap.add_argument('--emd_every', type=int, default=10)
    ap.add_argument('--pg', default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/i-physgaussian')
    ap.add_argument('--fp64', action='store_true', help='모든 계산을 float64 로')
    ap.add_argument('--iters0', type=int, default=400, help='adam_bt: 첫 프레임 반복')
    ap.add_argument('--iters', type=int, default=200, help='adam_bt: 이후 프레임 반복')
    ap.add_argument('--lr', type=float, default=0.001, help='adam_bt: Adam 학습률')
    ap.add_argument('--lam_inv', type=float, default=100.0, help='adam_bt: 장벽 계수')
    ap.add_argument('--tau', type=float, default=0.1, help='adam_bt: det 장벽 문턱')
    ap.add_argument('--knn_F', type=int, default=8, help='adam_bt: det 를 잴 이웃 수')
    ap.add_argument('--riem_eps', type=float, default=0.01, help='riem: 계량 G = JᵀJ + ε·λmax·I 의 상대 ε (λmax 는 프레임마다 거듭제곱법)')
    ap.add_argument('--profile', type=int, default=0, help='riem: 첫 프레임 이 반복 수만 돌며 구간별 시간 + torch 프로파일러 표를 내고 끝낸다')
    ap.add_argument('--riem_energy', default='elastic', choices=['elastic', 'logbarrier'], help='riem 계량의 에너지: elastic (스프링/StVK 막/고정공회전) | logbarrier (det 의 로그 장벽, 잔차 log J, 계량 Σ ∇logJ ∇logJᵀ: ours/tet 격자 셀, vrgs 삼각형 셀, gaussim 입자)')
    ap.add_argument('--tet_quality', type=float, default=0.05, help='phystwin logbarrier: 4-클릭 사면체 중 정지 품질 6√2·V/l_rms³ 이 이보다 작은(납작한) 것은 버린다')
    ap.add_argument('--ipm_mu0', type=float, default=0.0001, help='ipm: 첫 장벽 계수 μ')
    ap.add_argument('--ipm_stages', type=int, default=4, help='ipm: μ 를 10 배씩 줄이는 단계 수 (프레임마다)')
    ap.add_argument('--save_F', type=int, default=0, help="앞 K 점 (GF: 가우시안) 의 프레임별 변형 기울기 F (float16) 도 저장 (GF 렌더용, result['Fs'] [T,K,9])")
    ap.add_argument('--traj32', action='store_true', help='--save_traj 를 float32 로 (위상 지표용; 기본 float16)')
    ap.add_argument('--save_traj', action='store_true', help='모든 프레임의 가우시안 위치(float16)를 저장 (물리 지표·EMD 재측정용)')
    ap.add_argument('--hex_pbar', action='store_true', help='hex (rep4 11-15): 계량에 입자 (가우시안) 위치 det 로그 장벽 log det F_p (가중 1/√N) 을 가우스 점 장벽과 함께 넣는다')
    ap.add_argument('--hex_lmax_gauss', action='store_true', help='hex --hex_pbar: ε·λmax 의 λmax 를 가우스 점 장벽 계량만으로 잰다 (기존 판과 같은 감쇠 크기)')
    ap.add_argument('--feas', action='store_true', help='riem (rep4 11-15): Armijo 되돌림이 로그 장벽의 정의역을 지킨다 -- 시도 보폭에서 입자 증분 det F_p > 0 (그리고 hex 는 가우스 점 det > 0) 일 때만 받아들인다 (아니면 반으로)')
    ap.add_argument('--hex_diag', action='store_true', help='hex: 프레임 끝마다 입자 증분 det<=0 수, 그 중 자기 칸 가우스 점 8 개가 모두 > 0 인 수, 가우스 점 det<=0 수를 찍는다')
    ap.add_argument('--ls_max', type=int, default=30, help='riem: Armijo 되돌림 최대 횟수 (0 이면 고정 보폭)')
    ap.add_argument('--riem_k', type=float, default=1.0, help='riem: 탄성 강성 배수 k (G = k·JᵀJ + ε·λmax(k=1)·I -- ε 는 k=1 기준으로 고정)')
    ap.add_argument('--riem_lr', type=float, default=1.0, help='riem: 고정 보폭 η')
    ap.add_argument('--riem_cg', type=int, default=50, help='riem: CG 최대 반복')
    ap.add_argument('--pt_realtime', action='store_true', help='phystwin: 공식 실시간 데모 방식 -- 이웃 질량점 인덱스는 첫 프레임에 한 번, 매 프레임 그 이웃까지 거리로 가중치만 다시')
    ap.add_argument('--tb', default='auto', help='TensorBoard 디렉토리 (auto: /workspace/tbrf/<폴더>_<파일>, none: 끔)')
    ap.add_argument('--opt', default='lbfgs', choices=['lbfgs', 'gd', 'adam_bt', 'adam', 'riem', 'ipm'], help='lbfgs (원래 설정) / gd: 고정 학습률 경사하강')
    ap.add_argument('--gd_lr', type=float, default=1.0)
    ap.add_argument('--frames', type=int, default=0, help='앞 몇 프레임만 (0 이면 전부)')
    ap.add_argument('--autograd_jac', action='store_true', help='야코비안을 예전처럼 자동미분 2 차로 (검증·비교용, 느리다)')
    ap.add_argument('--no_compile', action='store_true', help='torch.compile 끄기')
    ap.add_argument('--check_jac', action='store_true', help='해석적 야코비안을 자동미분과 비교하고 끝낸다')
    a = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(0)
    if a.fp64:
        torch.set_default_dtype(torch.float64)
    DT = torch.float64 if a.fp64 else torch.float32
    D = np.load(a.flow)
    AX = np.load(a.aux, allow_pickle=True)
    G0 = torch.as_tensor(AX['G'], dtype=DT, device=dev)
    FI = torch.as_tensor(D['idx'], device=dev)
    TRAJ = torch.as_tensor(D['traj'], dtype=DT, device=dev)
    GIDX_SUB = FI.cpu().numpy()
    G0 = G0[FI].contiguous()
    FI = torch.arange(FI.numel(), device=dev)
    (T, N, NG) = (TRAJ.shape[0] - 1, FI.numel(), G0.shape[0])
    if a.frames:
        T = min(T, a.frames)
    assert torch.allclose(G0[FI], TRAJ[0], atol=1e-05), '흐름과 가우시안 집합이 다르다'
    L = float((G0.max(0).values - G0.min(0).values).norm())
    S_NORM = float(AX['s'])
    (LO, OFF) = (AX['lo'], AX['off'])

    def barrier(J):
        return _barrier_b(J, a.jhat, ext_kind='linear')

    def fps(X, k, seed=0):
        g = torch.Generator(device='cpu').manual_seed(seed)
        i = [int(torch.randint(X.shape[0], (1,), generator=g))]
        d = (X - X[i[0]]).norm(dim=1)
        for _ in range(k - 1):
            j = int(d.argmax())
            i.append(j)
            d = torch.minimum(d, (X - X[j]).norm(dim=1))
        return torch.tensor(i, device=X.device)

    def quat_mat(q):
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        (w, x, y, z) = q.unbind(-1)
        return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)

    def det3(F):
        """3x3 행렬식 (torch.linalg.det 과 같은 값; 컴파일 그래프가 끊기지 않게 전개)."""
        return F[:, 0, 0] * (F[:, 1, 1] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 1]) - F[:, 0, 1] * (F[:, 1, 0] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 0]) + F[:, 0, 2] * (F[:, 1, 0] * F[:, 2, 1] - F[:, 1, 1] * F[:, 2, 0])

    def outer_sum(A, B):
        """Σ_k A[:,k,:] ⊗ B[:,k,:] -> [n,3,3]. einsum 이 작은 배치 행렬곱(sgemm)으로 가 느려 원소별로."""
        return (A[..., :, None] * B[..., None, :]).sum(1)

    def mm3(A, B):
        """[n,3,3] @ [n,3,3] 을 원소별로 (같은 이유)."""
        return (A[..., :, :, None] * B[..., None, :, :]).sum(-2)

    def eye_plus(M):
        return M + torch.eye(3, device=M.device, dtype=M.dtype)

    def jac(fn, X):
        """y = fn(X) 와 J = ∂y/∂X (점별, create_graph)."""
        X = X.detach().requires_grad_(True)
        y = fn(X)
        rows = [torch.autograd.grad(y[:, i].sum(), X, create_graph=True)[0] for i in range(3)]
        return (y, torch.stack(rows, 1))

    def res_lattice_lb(x, Xn, cells, cD0i):
        u = x[:Xn.numel()].reshape(-1, 3)
        X = (Xn + u)[cells]
        D = (X[:, 1:] - X[:, :1]).transpose(1, 2)
        J = det3(mm3(D, cD0i))
        return torch.log(J.clamp_min(1e-06)) * cells.shape[0] ** (-0.5)

    def hex_gauss_F(u, cn, BGn):
        """가우스 점 F_q = I + Σ_o u[cn[c,o]] ⊗ BGn[q,o] -> [C*8,3,3] (칸 하나의 27 노드 x 팔분 8 점, 행렬곱 한 번)."""
        U = u[cn]
        C = U.shape[0]
        Ft = U.transpose(1, 2).reshape(C * 3, 27) @ BGn.permute(1, 0, 2).reshape(27, 24)
        Fq = Ft.reshape(C, 3, 8, 3).permute(0, 2, 1, 3).reshape(C * 8, 3, 3)
        return eye_plus(Fq)

    def res_hex_lb(x, cn, BGn):
        J = det3(hex_gauss_F(x.reshape(-1, 3), cn, BGn))
        return torch.log(J.clamp_min(1e-06)) * J.shape[0] ** (-0.5)

    def res_hex_lb_p(x, cn, BGn, rows, G27):
        """hex --hex_pbar: 가우스 점 장벽 (가중 1/√Q) + 입자 위치 장벽 log det F_p, F_p = I + Σ u_o ⊗ ∇w_o(X_p) (가중 1/√N)."""
        u = x.reshape(-1, 3)
        Jq = det3(hex_gauss_F(u, cn, BGn))
        Jp = det3(eye_plus(outer_sum(u[rows], G27)))
        return torch.cat([torch.log(Jq.clamp_min(1e-06)) * Jq.shape[0] ** (-0.5), torch.log(Jp.clamp_min(1e-06)) * Jp.shape[0] ** (-0.5)])

    def res_spring(x, B, rel, L0):
        Bn = B + x.reshape(-1, 3)
        return ((Bn[rel] - Bn[:, None]).norm(dim=-1) - L0).reshape(-1)

    def res_tri_lb(x, f, n0, A0):
        v = x.reshape(-1, 3)
        (v0, v1, v2) = (v[f[:, 0]], v[f[:, 1]], v[f[:, 2]])
        c = torch.cross(v1 - v0, v2 - v0, dim=-1)
        J = (c * n0).sum(-1) / (2 * A0)
        return A0.sqrt() * torch.log(J.clamp_min(1e-06))

    def res_simp_lb(x, W, dW, Xh):
        Tm = x.reshape(-1, 3, 4)
        TX = (Tm[None] * Xh[:, None, None, :]).sum(-1)
        F = eye_plus((W[..., None, None] * Tm[None, :, :, :3]).sum(1) + outer_sum(TX, dW))
        return torch.log(det3(F).clamp_min(1e-06)) * W.shape[0] ** (-0.5)

    def _gn_prod(fn, th, u, *args):
        """가우스-뉴턴 곱 Jᵀ J u (J = ∂fn/∂θ)."""
        from torch.func import jvp, vjp
        (_, Ju) = jvp(lambda z: fn(z, *args), (th,), (u,))
        (_, vf) = vjp(lambda z: fn(z, *args), th)
        return vf(Ju)[0]
    GN_PROD = None
    NFEAS = [0]
    HDIAG = []

    class Lattice(torch.nn.Module):
        """ours / tet: 매 프레임 직전 위치에 격자를 새로 깔고 증분 변위로 스키닝."""
        rebind_each_frame = True

        def __init__(self, gregory):
            super().__init__()
            self.greg = gregory
            self.h = 2.0 / 100.0 * (2.0 * math.sqrt(2.0)) ** (1.0 / 3.0) / S_NORM
            self.a = 0.5

        def rebind(self, Pall):
            (lo, lat, nn) = sx.grid_for_nodes(Pall, 1.0, h_fix=self.h)
            (idx, lam, _) = sx.locate(Pall, lo, lat, nn)
            (rows, uniq) = sx.active_nodes(idx)
            self.rows = rows
            self.Xn = sx.node_pos(lo, lat, nn, uniq)
            Xv = self.Xn[rows]
            self.v0 = Xv[:, 0]
            self.Minv = torch.linalg.inv((Xv[:, 1:] - Xv[:, :1]).transpose(1, 2))
            M = uniq.numel()
            X = Pall
            l3 = (self.Minv @ (X - self.v0)[..., None]).squeeze(-1)
            self.lam = torch.cat([1.0 - l3.sum(1, keepdim=True), l3], 1)
            self.dlam = torch.cat([-self.Minv.sum(1, keepdim=True), self.Minv], 1)
            if self.greg:
                d = X[:, None] - self.Xn[rows]
                self.r = d.norm(dim=-1)
                self.dr = d / self.r[..., None].clamp_min(1e-30)
                S = (self.lam * self.lam).sum(1, keepdim=True)
                self.w = self.lam * self.lam / S.clamp_min(1e-12)
                dS = (2 * self.lam[..., None] * self.dlam).sum(1, keepdim=True)
                self.dw = 2 * self.lam[..., None] * self.dlam / S[..., None] - (self.lam * self.lam)[..., None] * dS / (S * S)[..., None]
            self.cells = torch.unique(rows, dim=0)
            Xc = self.Xn[self.cells]
            self.cD0i = torch.linalg.inv((Xc[:, 1:] - Xc[:, :1]).transpose(1, 2))
            self.u = torch.nn.Parameter(torch.zeros(M, 3, device=dev))
            self.rho_raw = torch.nn.Parameter(torch.zeros(M, device=dev)) if self.greg else None
            self.dof = (4 if self.greg else 3) * M

        def params(self):
            return [self.u] + ([self.rho_raw] if self.greg else [])

        def yJ(self, X, sel):
            return self.yJ_p(X, sel, self.u, self.rho_raw)

        def cellJ(self, u, rho_raw=None):
            """셀 det 와 가중치 w² (Kuhn 사면체는 부피가 같아 1/셀 수)."""
            X = (self.Xn + u)[self.cells]
            D = (X[:, 1:] - X[:, :1]).transpose(1, 2)
            J = det3(mm3(D, self.cD0i))
            return (J, torch.full_like(J, 1.0 / self.cells.shape[0]))

        def logbarrier_residual(self, u, rho_raw=None):
            """셀(사면체) det 로그 장벽: log J_c, J_c = det(D(Xn+u) D0⁻¹) -- 이 프레임 격자 대비 증분.
        Kuhn 사면체는 부피가 모두 같아 가중치는 1. ρ 는 셀 det 에 들어가지 않는다."""
            X = (self.Xn + u)[self.cells]
            D = (X[:, 1:] - X[:, :1]).transpose(1, 2)
            J = det3(mm3(D, self.cD0i))
            return torch.log(J.clamp_min(1e-06)) * (1.0 / math.sqrt(self.cells.shape[0]))

        def yJ_p(self, X, sel, u, rho_raw):
            """해석적 변위 y - X 와 ∂y/∂X (forward 와 같은 식을 손으로 미분). 매개변수를 인자로."""
            rows = self.rows[sel]
            U = u[rows]
            if not self.greg:
                dy = (self.lam[sel][..., None] * U).sum(1)
                return (dy, eye_plus(outer_sum(U, self.dlam[sel])))
            (r, dr, w, dw) = (self.r[sel], self.dr[sel], self.w[sel], self.dw[sel])
            rho = self.h * (0.05 + 0.95 * torch.sigmoid(rho_raw)[rows])
            ins = r < rho
            inner = 1.0 - self.a * (torch.minimum(r, rho) / rho) ** 2
            q = 1.0 + (r - rho).clamp_min(0.0) / (0.5 * self.h)
            outer = (1.0 - self.a) / q
            psi_raw = torch.where(ins, inner, outer)
            psi = psi_raw.clamp_min(1e-06)
            dpsi = torch.where(ins, -2.0 * self.a * r / (rho * rho), -(1.0 - self.a) / (0.5 * self.h) / (q * q)) * (psi_raw > 1e-06)
            g = w * psi
            dg = dw * psi[..., None] + (w * dpsi)[..., None] * dr
            G = g.sum(1, keepdim=True).clamp_min(1e-12)
            W = g / G
            dW = dg / G[..., None] - W[..., None] * dg.sum(1, keepdim=True) / G[..., None]
            dy = (W[..., None] * U).sum(1)
            return (dy, eye_plus(outer_sum(U, dW)))

        def forward(self, X, sel):
            rows = self.rows[sel]
            l3 = (self.Minv[sel] @ (X - self.v0[sel])[..., None]).squeeze(-1)
            lam = torch.cat([1.0 - l3.sum(1, keepdim=True), l3], 1)
            if not self.greg:
                return X + (lam[..., None] * self.u[rows]).sum(1)
            r = (X[:, None] - self.Xn[rows]).norm(dim=-1)
            rho = self.h * (0.05 + 0.95 * torch.sigmoid(self.rho_raw)[rows])
            inner = 1.0 - self.a * (torch.minimum(r, rho) / rho) ** 2
            outer = (1.0 - self.a) / (1.0 + (r - rho).clamp_min(0.0) / (0.5 * self.h))
            psi = torch.where(r < rho, inner, outer).clamp_min(1e-06)
            w = lam * lam
            w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
            g = w * psi
            W = g / g.sum(1, keepdim=True).clamp_min(1e-12)
            return X + (W[..., None] * self.u[rows]).sum(1)

    class PhysTwin(torch.nn.Module):
        rebind_each_frame = True

        def __init__(self, K=16):
            super().__init__()
            g = torch.Generator(device='cpu').manual_seed(0)
            P = G0[torch.randperm(NG, generator=g)[:11024].to(dev)]
            vox = 0.005 / 0.25
            key = torch.floor((P - P.min(0).values) / vox).long()
            (_, inv) = torch.unique(key, dim=0, return_inverse=True)
            B = torch.zeros(inv.max() + 1, 3, device=dev).index_reduce_(0, inv, P, 'mean', include_self=False)
            (self.B, self.K) = (B, K)
            self.rel = torch.cdist(B, B).topk(K + 1, largest=False).indices[:, 1:]
            self.B0 = B.clone()
            self.L0 = (B[self.rel] - B[:, None]).norm(dim=-1)
            self.m = None
            self.dof = 3 * B.shape[0]

        def rebind(self, Pall):
            if self.m is not None:
                self.B = (self.B + self.m.detach()).clone()
            if a.pt_realtime and getattr(self, 'wi', None) is not None:
                ii = self.wi
            else:
                ii = torch.cat([torch.cdist(Pall[i:i + 20000], self.B).topk(self.K, largest=False).indices for i in range(0, Pall.shape[0], 20000)])
            self.wi = ii
            d = Pall[:, None] - self.B[ii]
            dn = d.norm(dim=-1)
            qd = 1.0 / (dn + 1e-06)
            Q = qd.sum(1, keepdim=True)
            dq = -(qd * qd)[..., None] * d / dn[..., None].clamp_min(1e-30)
            self.w = qd / Q
            self.dw = dq / Q[..., None] - qd[..., None] * dq.sum(1, keepdim=True) / (Q * Q)[..., None]
            self.d = d
            self.m = torch.nn.Parameter(torch.zeros_like(self.B))

        def params(self):
            return [self.m]

        def logbarrier_residual(self, m):
            """셀 det 로그 장벽: √V₀ · log J,  J = det(D(x) D0⁻¹) (정지 1)."""
            X = (self.B + m)[self.tets]
            D = (X[:, 1:] - X[:, :1]).transpose(1, 2)
            J = det3(mm3(D, self.D0i))
            return self.V0.sqrt() * torch.log(J.clamp_min(1e-06))

        def elastic_residual(self, m):
            """스프링 에너지 E = ½ Σ (|x_i - x_j| - L0_ij)² 의 잔차 (처음 이웃 16 개, 단위 강성)."""
            Bn = self.B + m
            return ((Bn[self.rel] - Bn[:, None]).norm(dim=-1) - self.L0).reshape(-1)

        def bone_R(self):
            (B, m, rel) = (self.B, self.m, self.rel)
            A0 = B[rel] - B[:, None]
            A1 = B[rel] + m[rel] - (B[:, None] + m[:, None])
            (U, S, Vh) = torch.linalg.svd(A1.transpose(1, 2) @ A0)
            dfix = torch.ones_like(S)
            dfix[:, -1] = torch.sign(torch.linalg.det(U @ Vh))
            return U @ torch.diag_embed(dfix) @ Vh

        def yJ(self, X, sel):
            k = self.wi[sel]
            (w, dw) = (self.w[sel], self.dw[sel])
            Rk = self.bone_R()[k]
            dsel = self.d[sel]
            Rd = (Rk @ dsel[..., None]).squeeze(-1)
            moved = Rd + self.B[k] + self.m[k]
            dy = (w[..., None] * (Rd - dsel + self.m[k])).sum(1)
            J = (w[..., None, None] * Rk).sum(1) + outer_sum(moved, dw)
            return (dy, J)

        def forward(self, X, sel):
            k = self.wi[sel]
            d = (X[:, None] - self.B[k]).norm(dim=-1)
            w = 1.0 / (d + 1e-06)
            w = w / w.sum(1, keepdim=True)
            R = self.bone_R()
            moved = (R[k] @ (X[:, None] - self.B[k])[..., None]).squeeze(-1) + self.B[k] + self.m[k]
            return (w[..., None] * moved).sum(1)

        def cov_R(self, X, sel):
            """공식: 입자 회전 = 뼈 회전 쿼터니언의 가중 평균."""
            k = self.wi[sel]
            d = (X[:, None] - self.B[k]).norm(dim=-1)
            w = 1.0 / (d + 1e-06)
            w = w / w.sum(1, keepdim=True)
            R = self.bone_R()
            q = rot_to_quat(R)[k]
            q = q * torch.sign((q * q[:, :1]).sum(-1, keepdim=True)).clamp_min(-1)
            return quat_mat((w[..., None] * q).sum(1))

    def rot_to_quat(R):
        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        w = torch.sqrt((1 + tr).clamp_min(1e-12)) / 2
        x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
        y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
        z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)
        return torch.stack([w, x, y, z], -1)

    class GausSim(torch.nn.Module):
        rebind_each_frame = False

        def __init__(self):
            super().__init__()
            k1 = max(int(round(0.01 * NG)), 2)
            k2 = max(int(round(0.01 * k1)), 1)
            self.C1 = G0[fps(G0, k1)]
            self.C2 = self.C1[fps(self.C1, k2, 1)]
            self.par = torch.cdist(self.C1, self.C2).argmin(1)
            self.lab = torch.cdist(G0, self.C1).argmin(1)
            self.p2 = torch.nn.Parameter(torch.zeros(k2, 3, device=dev))
            z = torch.zeros(k1, 11, device=dev)
            z[:, 0] = 1.0
            z[:, 7] = 1.0
            self.dg = torch.nn.Parameter(z)
            self.dof = 3 * k2 + 11 * k1
            print(f'[gaussim] 군집 {k1} / 위 군집 {k2}', flush=True)

        def params(self):
            return [self.p2, self.dg]

        def F_of(self, p2, dg):
            """군집 F (매개변수를 인자로; p2 는 쓰지 않는다)."""
            U = quat_mat(dg[:, 0:4])
            s = torch.exp(dg[:, 4:7].clamp(-5, 5))
            s = s / torch.prod(s, -1, keepdim=True).pow(1 / 3)
            V = quat_mat(dg[:, 7:11])
            return U @ torch.diag_embed(s) @ V.transpose(1, 2)

        def F1(self):
            U = quat_mat(self.dg[:, 0:4])
            s = torch.exp(self.dg[:, 4:7].clamp(-5, 5))
            s = s / torch.prod(s, -1, keepdim=True).pow(1 / 3)
            V = quat_mat(self.dg[:, 7:11])
            return U @ torch.diag_embed(s) @ V.transpose(1, 2)

        def yJ(self, X, sel):
            k = self.lab[sel]
            F = self.F1()[k]
            P = self.C2[self.par[k]]
            FmI = F - torch.eye(3, device=F.device, dtype=F.dtype)
            return (self.p2[self.par[k]] + (FmI @ (X - P)[..., None]).squeeze(-1), F)

        def forward(self, X, sel):
            k = self.lab[sel]
            F = self.F1()[k]
            P = self.C2[self.par[k]]
            return P + self.p2[self.par[k]] + (F @ (X - P)[..., None]).squeeze(-1)

    class Simplicits(torch.nn.Module):
        rebind_each_frame = False

        def __init__(self, path):
            super().__init__()
            self.fcn = torch.load(path, weights_only=False).to(dev)
            for q in self.fcn.parameters():
                q.requires_grad_(False)
            with torch.no_grad():
                K = self.fcn(G0[:2].float()).shape[1]
            self.Tm = torch.nn.Parameter(torch.zeros(K, 3, 4, device=dev))
            self.dof = 12 * K
            (Ws, dWs) = ([], [])
            for i in range(0, NG, 20000):
                Xc = G0[i:i + 20000].detach().requires_grad_(True)
                Wc = self.fcn(Xc.float()).to(Xc.dtype)
                g = [torch.autograd.grad(Wc[:, j].sum(), Xc, retain_graph=True)[0] for j in range(K)]
                Ws.append(Wc.detach())
                dWs.append(torch.stack(g, 1).detach())
            (self.W, self.dW) = (torch.cat(Ws), torch.cat(dWs))

        def params(self):
            return [self.Tm]

        def yJ(self, X, sel):
            (W, dW) = (self.W[sel], self.dW[sel])
            Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)
            TX = torch.einsum('kij,nj->nki', self.Tm, Xh)
            dy = (W[..., None] * TX).sum(1)
            J = (W[..., None, None] * self.Tm[None, :, :, :3]).sum(1) + outer_sum(TX, dW)
            return (dy, eye_plus(J))

        def forward(self, X, sel):
            W = self.fcn(X.float()).to(X.dtype)
            Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)
            return X + (W[..., None] * torch.einsum('kij,nj->nki', self.Tm, Xh)).sum(1)

    class VRGS(torch.nn.Module):
        rebind_each_frame = False

        def __init__(self):
            super().__init__()
            g_sim = (G0 - torch.as_tensor(OFF, device=dev, dtype=torch.float32)) * S_NORM + torch.as_tensor(LO, device=dev, dtype=torch.float32)
            (v, f) = mesh_from_points(g_sim, n_grid=100, grid_lim=2.0)
            v = v.to(DT)
            v = (v - torch.as_tensor(LO, device=dev, dtype=torch.float32)) / S_NORM + torch.as_tensor(OFF, device=dev, dtype=torch.float32)
            used = torch.unique(f)
            remap = torch.full((v.shape[0],), -1, dtype=torch.long, device=dev)
            remap[used] = torch.arange(used.numel(), device=dev)
            (self.vr, self.f) = (v[used], remap[f])
            (self.ti, self.a1, self.a2, self.b) = bind_points(G0, self.vr, self.f)
            self.v = torch.nn.Parameter(self.vr.clone())
            self.A0i = torch.linalg.inv(torch.stack(_tri_frame(self.vr, self.f)[1:], -1))
            self.dof = 3 * self.vr.shape[0]
            (v0, v1, v2) = (self.vr[self.f[:, 0]], self.vr[self.f[:, 1]], self.vr[self.f[:, 2]])
            (e1, e2) = (v1 - v0, v2 - v0)
            t1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            nrm = torch.cross(e1, e2, dim=-1)
            self.A0 = 0.5 * nrm.norm(dim=-1)
            self.n0 = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            t2 = torch.cross(nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12), t1, dim=-1)
            D0 = torch.stack([torch.stack([(e1 * t1).sum(-1), (e1 * t2).sum(-1)], -1), torch.stack([(e2 * t1).sum(-1), (e2 * t2).sum(-1)], -1)], -1)
            self.D0i = torch.linalg.inv(D0)
            nu = 0.3
            (self.mu, self.lam) = (1.0, 2 * nu / (1 - 2 * nu))
            print(f'[vrgs] 꼭짓점 {self.vr.shape[0]} 삼각형 {self.f.shape[0]}', flush=True)

        def params(self):
            return [self.v]

        def forward(self, X, sel):
            (v0, e1, e2, n) = _tri_frame(self.v, self.f)
            t = self.ti[sel]
            return v0[t] + self.a1[sel, None] * e1[t] + self.a2[sel, None] * e2[t] + self.b[sel, None] * n[t]

        def F(self, sel):
            A = torch.stack(_tri_frame(self.v, self.f)[1:], -1)
            return (A @ self.A0i)[self.ti[sel]]

        def yJ(self, X, sel):
            return (self(X, sel) - X, self.F(sel))

        def cellJ(self, v):
            """삼각형 셀 부호 넓이비와 가중치 w² = A0."""
            (v0, v1, v2) = (v[self.f[:, 0]], v[self.f[:, 1]], v[self.f[:, 2]])
            c = torch.cross(v1 - v0, v2 - v0, dim=-1)
            return ((c * self.n0).sum(-1) / (2 * self.A0), self.A0)

        def logbarrier_residual(self, v):
            """삼각형 셀 det 로그 장벽: √A₀ · log J,  J = (e1×e2)·n̂₀ / |e1⁰×e2⁰| (정지 법선 기준 부호 있는 넓이비)."""
            (v0, v1, v2) = (v[self.f[:, 0]], v[self.f[:, 1]], v[self.f[:, 2]])
            c = torch.cross(v1 - v0, v2 - v0, dim=-1)
            J = (c * self.n0).sum(-1) / (2 * self.A0)
            return self.A0.sqrt() * torch.log(J.clamp_min(1e-06))

        def elastic_residual(self, v):
            """삼각형 StVK 막 에너지 Σ A0 (μ|E|² + λ/2 tr(E)²) 의 잔차 (E = ½(FᵀF - I), F 는 3x2)."""
            (v0, v1, v2) = (v[self.f[:, 0]], v[self.f[:, 1]], v[self.f[:, 2]])
            (e1, e2, Di) = (v1 - v0, v2 - v0, self.D0i)
            f1 = e1 * Di[:, 0, 0, None] + e2 * Di[:, 1, 0, None]
            f2 = e1 * Di[:, 0, 1, None] + e2 * Di[:, 1, 1, None]
            (E11, E22) = (0.5 * ((f1 * f1).sum(-1) - 1), 0.5 * ((f2 * f2).sum(-1) - 1))
            E12 = 0.5 * (f1 * f2).sum(-1)
            w = self.A0.sqrt()
            return torch.cat([w * math.sqrt(self.mu) * E11, w * math.sqrt(self.mu) * E22, w * math.sqrt(2 * self.mu) * E12, w * math.sqrt(self.lam / 2) * (E11 + E22)])

    def res_spring_w(x, B, nbr, L0, sw):
        Bn = B + x.reshape(-1, 3)
        return (sw * ((Bn[nbr] - Bn[:, None]).norm(dim=-1) - L0)).reshape(-1)

    class SpringGaus(torch.nn.Module):
        """springgaus: Spring-Gaus 공식 표현 (질량점 2048 · 스프링 이웃 256 · 가우시안 결속 16), 위 설명 참고."""
        rebind_each_frame = False

        def __init__(self, n_sample=2048, k_nb=256, k_bind=16, voxel=0.01, n_fix=16, seed=123):
            import random
            super().__init__()
            self.eps = 1e-14
            sc = torch.floor(G0 / voxel).int() + 0.5
            grid = torch.unique(sc, dim=0)
            cloud = (grid * voxel).to(DT)
            try:
                random.seed(seed)
                sel = random.sample(range(cloud.shape[0]), n_sample)
            except ValueError:
                sc = torch.floor(G0 / (voxel * 0.1)).int() + 0.5
                cloud = (torch.unique(sc, dim=0) * (voxel * 0.1)).to(DT)
                random.seed(seed)
                sel = random.sample(range(cloud.shape[0]), n_sample)
            self.B = cloud[torch.as_tensor(sel, device=dev)].contiguous()
            M = self.B.shape[0]
            d = torch.cdist(self.B.double(), self.B.double())
            (dv, ix) = d.topk(k_nb + 1, largest=False)
            self.nbr = ix[:, 1:].contiguous()
            self.L0 = dv[:, 1:].to(DT).contiguous()
            kvec = torch.ones(k_nb, dtype=DT, device=dev)
            j = torch.arange(k_nb - n_fix, dtype=DT, device=dev) / (k_nb - n_fix)
            kvec[n_fix:] = torch.clamp(2 - torch.pow(torch.exp(torch.nn.functional.softplus(torch.zeros((), dtype=DT, device=dev))), j), 0, 1)
            self.sw = torch.sqrt(kvec[None] / (self.L0 + self.eps))
            (ids, cs, gs) = ([], [], [])
            for i in range(0, NG, 20000):
                Xc = G0[i:i + 20000]
                dd = torch.cdist(Xc.double(), self.B.double())
                (_, ii) = dd.topk(k_bind, largest=False)
                (c, gc) = self.coef(Xc, ii)
                ids.append(ii)
                cs.append(c.detach())
                gs.append(gc.detach())
            (self.idx, self.c, self.dc) = (torch.cat(ids), torch.cat(cs), torch.cat(gs))
            self.u = torch.nn.Parameter(torch.zeros(M, 3, device=dev, dtype=DT))
            self.dof = 3 * M
            print(f'[springgaus] 질량점 {M} (voxel 표집 {cloud.shape[0]} 개 중), 스프링 {M * k_nb}, 결속 {k_bind}', flush=True)

        def coef(self, X, ii):
            """공식 set_all_particle 식: c_k = a_k / Σ a,  a_k = (|X - P_k|)^-0.5. 돌려줌 (c [n,K], ∂c/∂X [n,K,3] 해석)."""
            r = X[:, None] - self.B[ii]
            dist = r.norm(dim=-1).clamp_min(1e-08)
            a = 1.0 / (dist ** 0.5 + self.eps)
            S = a.sum(1, keepdim=True)
            c = a / S
            da = (-0.5 * dist ** (-2.5))[..., None] * r
            dS = da.sum(1, keepdim=True)
            return (c, (da - c[..., None] * dS) / S[..., None])

        def params(self):
            return [self.u]

        def yJ(self, X, sel):
            return self.yJ_p(X, sel, self.u)

        def yJ_p(self, X, sel, u):
            U = u[self.idx[sel]]
            dy = (self.c[sel][..., None] * U).sum(1)
            return (dy, eye_plus(outer_sum(U, self.dc[sel])))

        def elastic_residual(self, u):
            return res_spring_w(u.reshape(-1), self.B, self.nbr, self.L0, self.sw)

        def forward(self, X, sel):
            """검사용 (자동미분 대조): 가중치를 X 의 함수로 (이웃은 정지 위치에서 고정)."""
            (c, _) = self.coef(X, self.idx[sel])
            return X + (c[..., None] * self.u[self.idx[sel]]).sum(1)

    class VRGSX(torch.nn.Module):
        """vrgsx: VR-GS 케이지 (fusedgs/vrgs_xpbd.py 와 같은 메시·결속). 미지수 u = 정점 변위, 고정 위상."""
        rebind_each_frame = False

        def __init__(self, vmax):
            super().__init__()
            (h, V, T, tid, bw, bmin) = vx.cage_for_max_verts(G0.double(), vmax)
            self.h = h
            self.V = V.to(DT).contiguous()
            self.T = T.long().contiguous()
            P = V[T]
            Dm = torch.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], -1)
            self.DmI = torch.linalg.inv(Dm).to(DT).contiguous()
            self.tid = tid.long().contiguous()
            self.bw = bw.to(DT).contiguous()
            self.u = torch.nn.Parameter(torch.zeros(V.shape[0], 3, device=dev, dtype=DT))
            self.dof = 3 * V.shape[0]
            self.nt = T.shape[0]
            print(f'[vrgsx] 케이지 정점 {V.shape[0]} 사면체 {self.nt} 칸 {h:.5f} (정규화) 결속 최소 무게중심 {bmin:.3e} (가우시안 {G0.shape[0]})', flush=True)

        def params(self):
            return [self.u]

        def yJ(self, X, sel):
            return self.yJ_p(X, sel, self.u)

        def yJ_p(self, X, sel, u):
            t = self.tid[sel]
            Tt = self.T[t]
            dy = (self.bw[sel][..., None] * u[Tt]).sum(1)
            Ud = torch.stack([u[Tt[:, 1]] - u[Tt[:, 0]], u[Tt[:, 2]] - u[Tt[:, 0]], u[Tt[:, 3]] - u[Tt[:, 0]]], -1)
            return (dy, eye_plus(mm3(Ud, self.DmI[t])))

        def cellJ(self, u):
            X = (self.V + u)[self.T]
            D = (X[:, 1:] - X[:, :1]).transpose(1, 2)
            J = det3(mm3(D, self.DmI))
            return (J, torch.full_like(J, 1.0 / J.numel()))

        def logbarrier_residual(self, u):
            return res_lattice_lb(u.reshape(-1), self.V, self.T, self.DmI)

        def forward(self, X, sel):
            t = self.tid[sel]
            return X + (self.bw[sel][..., None] * self.u[self.T[t]]).sum(1)

    class Hex(torch.nn.Module):
        """hex: 이차 B-스플라인 육면체 격자 (실시간 판과 같은 격자·칸 규칙), 매 프레임 직전 위치로 다시 깐다.
    좌표: 추적은 정규화 좌표 P, 격자는 시뮬 좌표 x = (P - off)·s + lo 에서 y = x Aiᵀ (정수점 = 시뮬 원점)."""
        rebind_each_frame = True

        def __init__(self, dx_sim, cell):
            super().__init__()
            nh = cell * dx_sim * math.sqrt(2.0)
            lat = sx.lattice(nh, sx.HZ_CUBE * nh, device=dev, dtype=torch.float64)
            (self.A64, self.Ai64) = (lat.A, lat.Ai)
            self.edge_sim = float(lat.A[:, 0].norm())
            self.h = self.edge_sim / S_NORM
            self.S = float(S_NORM)
            self.lo64 = torch.as_tensor(np.asarray(LO, dtype=np.float64), device=dev)
            self.off64 = torch.as_tensor(np.asarray(OFF, dtype=np.float64), device=dev)
            O = torch.tensor([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], device=dev)
            self.O = O
            GG = 0.5 / math.sqrt(3.0)
            BG = torch.zeros(8, 27, 3, dtype=torch.float64, device=dev)
            for oc in range(8):
                fxv = [1.0 + (GG if oc >> 2 - q & 1 else -GG) for q in range(3)]
                w = [[0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2] for f in fxv]
                dw = [[f - 1.5, -2 * (f - 1), f - 0.5] for f in fxv]
                for o in range(27):
                    (i, j, k) = (o // 9, o // 3 % 3, o % 3)
                    BG[oc, o] = torch.tensor([dw[0][i] * w[1][j] * w[2][k], w[0][i] * dw[1][j] * w[2][k], w[0][i] * w[1][j] * dw[2][k]], dtype=torch.float64, device=dev)
            self.BGn = (BG @ self.Ai64 * self.S).to(DT)
            print(f'[hex] 칸 변 {self.edge_sim:.6f} (시뮬; dx {dx_sim} x cell {cell}) = {self.h:.6f} (정규화), 노드 간격 h {nh:.6f}, 정수점 = 시뮬 원점', flush=True)

        def _lat(self, P):
            return ((P.double() - self.off64) * self.S + self.lo64) @ self.Ai64.T

        @staticmethod
        def _wd(fx):
            w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2], -1)
            dw = torch.stack([fx - 1.5, -2 * (fx - 1), fx - 0.5], -1)
            return (w, dw)

        def _w27(self, fx, grad=True):
            (w, dw) = self._wd(fx)
            (i, j, k) = (self.O[:, 0], self.O[:, 1], self.O[:, 2])
            W = w[:, 0, i] * w[:, 1, j] * w[:, 2, k]
            if not grad:
                return (W, None)
            gy = torch.stack([dw[:, 0, i] * w[:, 1, j] * w[:, 2, k], w[:, 0, i] * dw[:, 1, j] * w[:, 2, k], w[:, 0, i] * w[:, 1, j] * dw[:, 2, k]], -1)
            return (W, gy)

        def rebind(self, Pall):
            y = self._lat(Pall)
            b = torch.floor(y - 0.5)
            (W, gy) = self._w27(y - b)
            bi = b.long()
            nb = bi[:, None, :] + self.O[None]
            mn = nb.reshape(-1, 3).min(0).values
            ext = nb.reshape(-1, 3).max(0).values - mn + 1
            key = ((nb[..., 0] - mn[0]) * ext[1] + (nb[..., 1] - mn[1])) * ext[2] + (nb[..., 2] - mn[2])
            (uk, inv) = torch.unique(key.reshape(-1), return_inverse=True)
            self.rows = inv.reshape(-1, 27)
            M = uk.numel()
            ni = torch.stack([uk // (ext[1] * ext[2]), uk // ext[2] % ext[1], uk % ext[2]], 1) + mn
            self.Xn = ((ni.double() @ self.A64.T - self.lo64) / self.S + self.off64).to(DT)
            (cb, self.pcell) = torch.unique(bi, dim=0, return_inverse=True)
            cnk = cb[:, None, :] + self.O[None]
            ckey = ((cnk[..., 0] - mn[0]) * ext[1] + (cnk[..., 1] - mn[1])) * ext[2] + (cnk[..., 2] - mn[2])
            self.cn = torch.searchsorted(uk, ckey.reshape(-1)).reshape(-1, 27)
            assert bool((uk[self.cn] == ckey).all()), '칸 노드가 활성 노드에 없다'
            self.ncell = cb.shape[0]
            self.base = b.to(DT)
            self.W27 = W.to(DT)
            self.G27 = (gy @ self.Ai64 * self.S).to(DT)
            self.u = torch.nn.Parameter(torch.zeros(M, 3, device=dev))
            self.dof = 3 * M

        def params(self):
            return [self.u]

        def yJ(self, X, sel):
            return self.yJ_p(X, sel, self.u)

        def yJ_p(self, X, sel, u):
            U = u[self.rows[sel]]
            dy = (self.W27[sel][..., None] * U).sum(1)
            return (dy, eye_plus(outer_sum(U, self.G27[sel])))

        def cellJ(self, u):
            J = det3(hex_gauss_F(u, self.cn, self.BGn))
            return (J, torch.full_like(J, 1.0 / J.numel()))

        def logbarrier_residual(self, u):
            return res_hex_lb(u.reshape(-1), self.cn, self.BGn)

        def forward(self, X, sel):
            """검사용: 가중치를 X 의 함수로 (칸 base 는 깔 때 값 고정)."""
            xs = (X - self.off64.to(X.dtype)) * self.S + self.lo64.to(X.dtype)
            y = xs @ self.Ai64.T.to(X.dtype)
            (W, _) = self._w27(y - self.base[sel], grad=False)
            return X + (W[..., None] * self.u[self.rows[sel]]).sum(1)
    if a.method in ('ours', 'tet'):
        rep = Lattice(gregory=a.method == 'ours')
    elif a.method == 'hex':
        assert a.opt == 'riem' and a.riem_energy == 'logbarrier', 'hex: riem + logbarrier (가우스 점 det 로그 장벽) 전용'
        rep = Hex(a.hex_dx, a.hex_cell)
    elif a.method == 'vrgsx':
        assert a.opt == 'riem' and a.riem_energy == 'logbarrier', 'vrgsx: riem + logbarrier (사면체 det 로그 장벽) 전용'
        rep = VRGSX(a.vmax)
    elif a.method == 'springgaus':
        assert a.opt == 'riem' and a.riem_energy == 'elastic', 'springgaus: riem + elastic (스프링 에너지) 전용'
        rep = SpringGaus()
    elif a.method == 'phystwin':
        rep = PhysTwin()
    elif a.method == 'gaussim':
        rep = GausSim()
    elif a.method == 'simplicits':
        rep = Simplicits(a.simp)
    else:
        rep = VRGS()
    REB = rep.rebind_each_frame
    YJ = rep.yJ if a.no_compile else torch.compile(rep.yJ, dynamic=True)
    MAP_DY = [None]

    def map_and_F(Pref, sel, Fprev=None, need_graph=True):
        """선택 점의 위치와 누적 F (정지 대비)."""
        X = Pref[sel]
        if not a.autograd_jac:
            if need_graph:
                (dy, J) = YJ(X, sel)
            else:
                with torch.no_grad():
                    (dy, J) = YJ(X, sel)
            F = mm3(J, Fprev[sel]) if REB and Fprev is not None else J
            MAP_DY[0] = dy
            return (X + dy, F)
        if a.method == 'vrgs':
            y = rep(X, sel)
            F = rep.F(sel)
            return (y, F)
        if need_graph:
            (y, J) = jac(lambda Z: rep(Z, sel), X)
        else:
            with torch.enable_grad():
                (y, J) = jac(lambda Z: rep(Z, sel), X)
            (y, J) = (y.detach(), J.detach())
        F = J @ Fprev[sel] if REB and Fprev is not None else J
        return (y, F)

    def chamfer(A, B, ch=4096):
        ch = max(64, min(ch, int(200000000.0 // max(A.shape[0], B.shape[0], 1))))

        def one(P, Q):
            return sum((torch.cdist(P[i:i + ch], Q).min(1).values.sum() for i in range(0, P.shape[0], ch))) / P.shape[0]
        return float(0.5 * (one(A, B) + one(B, A)))

    def nn_dist(P, Q, ch=4096, k=8):
        """(추가 기록용, 표에는 안 씀) 정확한 최근접 거리: 행렬곱 cdist 로 후보 k 개, 거리는 직접 (rep_track2 현재 판 nn_dist).
    표의 CD 는 r10 과 같은 chamfer (행렬곱 cdist, float32) 로 둔다 -- 비교 방법 기존 값과 같은 식."""
        out = []
        ch = max(64, min(ch, int(200000000.0 // max(Q.shape[0], 1))))
        for i in range(0, P.shape[0], ch):
            Pi = P[i:i + ch]
            j = torch.cdist(Pi, Q).topk(min(k, Q.shape[0]), largest=False).indices
            out.append((Pi[:, None] - Q[j]).norm(dim=-1).min(1).values)
        return torch.cat(out)
    CKPT = a.out + '.ckpt.pt'
    CK = torch.load(CKPT, map_location=dev, weights_only=False) if a.resume and os.path.exists(CKPT) else None
    FRD = a.video + '.frames' if a.video else None
    (FR, FCHUNKS) = ([], CK['fchunks'] if CK is not None else [])
    RENDER = None
    if a.video:
        sys.path.append(a.pg)
        sys.path.append(os.path.join(a.pg, 'gaussian-splatting'))
        _cwd = os.getcwd()
        os.chdir(a.pg)
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        import imageio.v2 as imageio
        os.chdir(_cwd)
        model = str(AX['model'])
        torch.set_default_dtype(torch.float32)
        gs = GaussianModel(3)
        gs.load_ply(f'{model}/point_cloud/iteration_30000/point_cloud.ply')
        gi = torch.as_tensor(AX['gidx'][GIDX_SUB], device=dev)
        c6 = gs.get_covariance()[gi].detach()
        C0 = torch.zeros(NG, 3, 3, device=dev)
        (C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2]) = (c6[:, 0], c6[:, 1], c6[:, 2])
        (C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2]) = (c6[:, 3], c6[:, 4], c6[:, 5])
        (C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1]) = (c6[:, 1], c6[:, 2], c6[:, 4])
        (SHS, OPA) = (gs.get_features[gi].detach(), gs.get_opacity[gi].detach())
        torch.set_default_dtype(DT)
        (so, mean) = (float(AX['scale_origin']), torch.as_tensor(AX['mean'], device=dev).float())
        lo_t = torch.as_tensor(LO, device=dev).float()
        off_t = torch.as_tensor(OFF, device=dev).float()
        cam = json.load(open(f'{model}/cameras.json'))[0]
        (Rw, pos) = (np.array(cam['rotation']), np.array(cam['position']))
        W2C = np.linalg.inv(np.block([[Rw, pos[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
        fx_ = 2 * math.atan(cam['width'] / (2 * cam['fx']))
        fy_ = 2 * math.atan(cam['height'] / (2 * cam['fy']))
        wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
        pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fx_, fovY=fy_).transpose(0, 1).to(dev).float()
        st = GaussianRasterizationSettings(image_height=int(cam['height']), image_width=int(cam['width']), tanfovx=math.tan(fx_ * 0.5), tanfovy=math.tan(fy_ * 0.5), bg=torch.ones(3, device=dev, dtype=torch.float32), scale_modifier=1.0, viewmatrix=wv, projmatrix=(wv[None] @ pj[None])[0], sh_degree=3, campos=wv.inverse()[3, :3], prefiltered=False, debug=False)
        RAST = GaussianRasterizer(raster_settings=st)
        os.makedirs(FRD, exist_ok=True)
        RENDER = True

    def render(P, Fall=None, Rall=None):
        P = P.float()
        Pm = ((P - off_t) * S_NORM + lo_t - 1.0) / so + mean
        M = (Rall if Rall is not None else Fall).float()
        cov = M @ C0 @ M.transpose(1, 2)
        c6 = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
        torch.set_default_dtype(torch.float32)
        with torch.no_grad():
            img = RAST(means3D=Pm, means2D=torch.zeros_like(Pm), shs=SHS, colors_precomp=None, opacities=OPA, scales=None, rotations=None, cov3D_precomp=c6)[0]
        FR.append((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        torch.set_default_dtype(DT)

    def eval_all(Pref, Fprev):
        """전체 가우시안의 위치·누적 F (그리고 phystwin 은 공식 회전)."""
        (ys, Fs, Rs) = ([], [], [])
        for i in range(0, NG, 40000):
            sel = ALL[i:i + 40000]
            (y, F) = map_and_F(Pref, sel, Fprev, need_graph=False)
            ys.append(y.detach())
            Fs.append(F.detach())
            if a.method == 'phystwin':
                with torch.no_grad():
                    Rs.append(rep.cov_R(Pref[sel], sel))
        return (torch.cat(ys), torch.cat(Fs), torch.cat(Rs) if Rs else None)
    ALL = torch.arange(NG, device=dev)
    TBW = None
    if a.tb != 'none':
        from torch.utils.tensorboard import SummaryWriter
        _tb = a.tb if a.tb != 'auto' else os.path.join(os.environ.get('FUSEDGS_WORK', '/workspace') + '/tbrf', os.path.basename(os.path.dirname(os.path.abspath(a.out))) + '_' + os.path.splitext(os.path.basename(a.out))[0])
        TBW = SummaryWriter(_tb)
        print(f'[TB] {_tb}', flush=True)
    NCUR = [0]
    OPT = None
    if a.opt == 'adam_bt':
        NBR = torch.cat([torch.cdist(G0[i:i + 8192], G0).topk(a.knn_F + 1, largest=False).indices[:, 1:] for i in range(0, NG, 8192)])

        def knn_ref(P):
            dX = P[NBR] - P[:, None]
            return (dX, torch.linalg.inv(dX.transpose(1, 2) @ dX + 1e-12 * torch.eye(3, device=dev)))

        def knn_det(Y, ref):
            (dX, Binv) = ref
            dY = Y[NBR] - Y[:, None]
            return det3(dY.transpose(1, 2) @ dX @ Binv)
        REF0 = knn_ref(G0)
        OPT = None
    Pref = G0.clone()
    Fcum = torch.eye(3, device=dev).expand(NG, 3, 3).clone()
    Rcum = torch.eye(3, device=dev).expand(NG, 3, 3).clone()
    if REB:
        rep.rebind(Pref)
    rep.to(dev)
    print(f'[{a.method}] 가우시안 {NG} (맞추기 {N})  자유도 {rep.dof}  지름 L {L:.4f}', flush=True)
    if a.check_jac:
        gch = torch.Generator(device='cpu').manual_seed(1)
        with torch.no_grad():
            for q in rep.params():
                q.add_(0.02 * torch.randn(q.shape, generator=gch).to(q))
        sel = torch.randperm(NG, generator=gch)[:5000].to(dev)
        X = Pref[sel]
        if a.method == 'vrgs':
            (y0, J0) = (rep(X, sel), rep.F(sel))
        else:
            (y0, J0) = jac(lambda Z: rep(Z, sel), X)
        (y1, J1) = rep.yJ(X, sel)
        y1 = X + y1
        (y2, J2) = YJ(X, sel)
        l0 = y0.square().sum() + det3(J0).sum()
        g0 = torch.autograd.grad(l0, rep.params())
        l1 = y1.square().sum() + det3(J1).sum()
        g1 = torch.autograd.grad(l1, rep.params())
        rel = lambda A, B: float((A - B).abs().max() / B.abs().max().clamp_min(1e-30))
        print(f'[검사] y {rel(y1, y0):.2e}  J {rel(J1, J0):.2e}  컴파일 J {rel(J2, J0):.2e}  기울기 ' + ' '.join((f'{rel(p, q):.2e}' for (p, q) in zip(g1, g0))), flush=True)
        import time as _t
        for (name, fn) in (('자동미분', None), ('해석', rep.yJ), ('해석+컴파일', YJ)):
            for rep_i in range(4):
                torch.cuda.synchronize()
                t1 = _t.time()
                if fn is None:
                    (yy, JJ) = (rep(Pref, ALL), rep.F(ALL)) if a.method == 'vrgs' else jac(lambda Z: rep(Z, ALL), Pref)
                else:
                    (yy, JJ) = fn(Pref, ALL)
                    yy = Pref + yy
                (yy.square().mean() + det3(JJ).mean()).backward()
                torch.cuda.synchronize()
            print(f'[시간] {name}: 전체 {NG} 점 한 번 평가+역전파 {1000 * (_t.time() - t1):.1f} ms', flush=True)
        sys.exit(0)
    if RENDER and CK is None:
        render(G0, Fall=Fcum)
    (rows, CURVE, DOFS, STOP, EMDP) = ([], [], [], [], [])
    (PHYS, TRAJS, YPREV) = ([], [], [G0[FI].clone()])
    FSAVE = []
    t0 = time.time()
    T_START = 1
    if CK is not None:
        (Pref, Fcum, Rcum) = (CK['Pref'], CK['Fcum'], CK['Rcum'])
        (rows, CURVE, DOFS, STOP, EMDP, PHYS, TRAJS, YPREV) = [CK['L'][k] for k in ('rows', 'CURVE', 'DOFS', 'STOP', 'EMDP', 'PHYS', 'TRAJS', 'YPREV')]
        FSAVE = CK['L'].get('FS', [])
        HDIAG[:] = CK['L'].get('HD', [])
        torch.set_rng_state(CK['rng'].cpu())
        torch.cuda.set_rng_state(CK['cuda_rng'].cpu())
        T_START = CK['t'] + 1
        t0 -= CK['elapsed']
        print(f"[이어 돌리기] {CKPT}: 프레임 {CK['t']} 까지 완료 -> {T_START} 부터 (영상 조각 {len(FCHUNKS)})", flush=True)
    for t in range(T_START, T + 1):
        tgt = TRAJ[t]
        RES0 = Pref[FI] - tgt
        if REB and t > 1:
            rep.rebind(Pref)
            rep.to(dev)
        DOFS.append(rep.dof)
        it = [0]

        def closure():
            opt.zero_grad(set_to_none=True)
            (y, F) = map_and_F(Pref, FI, Fcum)
            J = det3(F)
            res = RES0 + MAP_DY[0] if not a.autograd_jac else y - tgt
            l2 = (res ** 2).sum(1).mean()
            bar = barrier(J).mean()
            loss = l2 + a.kappa * bar
            loss.backward()
            CURVE.append((t, it[0], float(l2), float(bar)))
            it[0] += 1
            return loss
        if a.opt in ('riem', 'ipm'):
            from torch.func import jvp, vjp
            PL = rep.params()
            shapes = [q.shape for q in PL]
            sizes = [q.numel() for q in PL]

            def unflat(x):
                return [c.reshape(sh) for (c, sh) in zip(torch.split(x, sizes), shapes)]

            def flat(ts):
                return torch.cat([q.reshape(-1) for q in ts])
            theta = flat([q.detach() for q in PL])
            X_ = Pref[FI]
            if a.method in ('ours', 'tet', 'hex', 'vrgsx') and a.riem_energy == 'logbarrier':

                def make_res(th):
                    return lambda x: rep.logbarrier_residual(*unflat(x))
            elif a.method in ('ours', 'tet'):
                nu_ = 0.3
                (mu_, la_) = (1.0, 2 * nu_ / (1 - 2 * nu_))
                Fp_ = Fcum[FI]
                wv_ = 1.0 / math.sqrt(FI.numel())

                def make_res(th):
                    with torch.no_grad():
                        (_, J0) = rep.yJ_p(X_, FI, *(unflat(th) if rep.greg else unflat(th) + [None]))
                        (U_, _, Vh_) = torch.linalg.svd(J0 @ Fp_)
                        Dg = torch.ones(U_.shape[0], 3, device=dev, dtype=U_.dtype)
                        Dg[:, 2] = torch.sign(torch.linalg.det(U_ @ Vh_))
                        R_ = U_ @ torch.diag_embed(Dg) @ Vh_

                    def res(x):
                        (_, Jx) = rep.yJ_p(X_, FI, *(unflat(x) if rep.greg else unflat(x) + [None]))
                        F_ = mm3(Jx, Fp_)
                        return torch.cat([(wv_ * math.sqrt(mu_) * (F_ - R_)).reshape(-1), wv_ * math.sqrt(la_ / 2) * (det3(F_) - 1.0)])
                    return res
            else:

                def make_res(th):
                    if a.riem_energy == 'logbarrier':
                        if a.method == 'gaussim':
                            wv = 1.0 / math.sqrt(FI.numel())
                            return lambda x: wv * torch.log(det3(GausSim.F_of(rep, *unflat(x))[rep.lab[FI]]).clamp_min(1e-06))
                        if a.method == 'simplicits':
                            wv = 1.0 / math.sqrt(FI.numel())
                            (W_, dW_) = (rep.W[FI], rep.dW[FI])
                            Xh_ = torch.cat([X_, torch.ones_like(X_[:, :1])], 1)

                            def res_s(x):
                                Tm = unflat(x)[0]
                                TX = torch.einsum('kij,nj->nki', Tm, Xh_)
                                F_ = eye_plus((W_[..., None, None] * Tm[None, :, :, :3]).sum(1) + outer_sum(TX, dW_))
                                return wv * torch.log(det3(F_).clamp_min(1e-06))
                            return res_s
                        return lambda x: rep.logbarrier_residual(*unflat(x))
                    return lambda x: rep.elastic_residual(*unflat(x))
            if a.method == 'simplicits':
                (W_i, dW_i) = (rep.W[FI], rep.dW[FI])
                Xh_i = torch.cat([X_, torch.ones_like(X_[:, :1])], 1)

                def Jfun(x):
                    Tm = unflat(x)[0]
                    TX = torch.einsum('kij,nj->nki', Tm, Xh_i)
                    F_ = eye_plus((W_i[..., None, None] * Tm[None, :, :, :3]).sum(1) + outer_sum(TX, dW_i))
                    J_ = det3(F_)
                    return (J_, torch.full_like(J_, 1.0 / J_.numel()))
            elif hasattr(rep, 'cellJ'):

                def Jfun(x):
                    return rep.cellJ(*unflat(x))
            else:
                Jfun = None
            PURE = None
            if a.riem_energy == 'logbarrier' and a.method in ('ours', 'tet'):
                PURE = (res_lattice_lb, (rep.Xn, rep.cells, rep.cD0i))
            elif a.riem_energy == 'logbarrier' and a.method == 'hex' and a.hex_pbar:
                PURE = (res_hex_lb_p, (rep.cn, rep.BGn, rep.rows[FI], rep.G27[FI]))
            elif a.riem_energy == 'logbarrier' and a.method == 'hex':
                PURE = (res_hex_lb, (rep.cn, rep.BGn))
            elif a.riem_energy == 'logbarrier' and a.method == 'vrgsx':
                PURE = (res_lattice_lb, (rep.V, rep.T, rep.DmI))
            elif a.riem_energy == 'logbarrier' and a.method == 'vrgs':
                PURE = (res_tri_lb, (rep.f, rep.n0, rep.A0))
            elif a.riem_energy == 'logbarrier' and a.method == 'simplicits':
                PURE = (res_simp_lb, (rep.W[FI], rep.dW[FI], torch.cat([X_, torch.ones_like(X_[:, :1])], 1)))
            elif a.riem_energy == 'elastic' and a.method == 'phystwin':
                PURE = (res_spring, (rep.B, rep.rel, rep.L0))
            elif a.riem_energy == 'elastic' and a.method == 'springgaus':
                PURE = (res_spring_w, (rep.B, rep.nbr, rep.L0, rep.sw))
            if PURE is not None and GN_PROD is None and (not a.no_compile):
                GN_PROD = torch.compile(_gn_prod, dynamic=True)

            def GN(th):
                if PURE is not None:
                    (fn_, args_) = PURE
                    pr = GN_PROD if GN_PROD is not None else _gn_prod
                    return lambda u: pr(fn_, th, u, *args_)
                rf = make_res(th)
                (_, vjp_fn) = vjp(rf, th)
                return lambda u: vjp_fn(jvp(rf, (th,), (u,))[1])[0]
            with torch.no_grad():
                Hv = GN(theta.detach())
                if a.method == 'hex' and a.hex_pbar and a.hex_lmax_gauss:
                    _th0 = theta.detach()
                    Hv = lambda v_: _gn_prod(res_hex_lb, _th0, v_, rep.cn, rep.BGn)
                u = torch.randn_like(theta)
                for _ in range(20):
                    u = Hv(u)
                    lmax = float(u.norm())
                    u = u / max(lmax, 1e-30)
            if a.profile:
                import collections
                TM = collections.defaultdict(float)
                NCG = [0]

                def tic():
                    torch.cuda.synchronize()
                    return time.time()
                WARM = 3
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            eps = a.riem_eps * max(lmax, 1e-12)
            if a.opt == 'ipm':
                assert Jfun is not None, 'ipm 은 로그 장벽(셀/입자 det)이 있는 방법만'
                ntot = a.iters0 if t == 1 else a.iters
                nst = max(ntot // a.ipm_stages, 1)
                step = 1.0

                def fval(x):
                    with torch.no_grad():
                        ps = unflat(x)
                        saved = [q.detach().clone() for q in PL]
                        for (q, pp) in zip(PL, ps):
                            q.copy_(pp)
                        (dy_, _) = YJ(X_, FI)
                        l2_ = float(((RES0 + dy_) ** 2).sum(1).mean())
                        for (q, pp) in zip(PL, saved):
                            q.copy_(pp)
                        (J_, w2_) = Jfun(x)
                        if float(J_.min()) <= 0:
                            return (float('inf'), l2_)
                        return (l2_ + mu * float(-(w2_ * torch.log(J_)).sum()), l2_)
                for it_ in range(ntot):
                    mu = a.ipm_mu0 * 0.1 ** min(it_ // nst, a.ipm_stages - 1)
                    for q in PL:
                        q.grad = None
                    (dy, _) = YJ(X_, FI)
                    l2 = ((RES0 + dy) ** 2).sum(1).mean()
                    th_ = flat(list(PL))
                    (Jc, w2c) = Jfun(th_)
                    f = l2 + mu * -(w2c * torch.log(Jc.clamp_min(1e-30))).sum()
                    f.backward()
                    gk = flat([q.grad if q.grad is not None else torch.zeros_like(q) for q in PL]).detach()
                    theta = flat([q.detach() for q in PL])
                    with torch.no_grad():
                        Hv = GN(theta)
                        Gv = lambda u: Hv(u) + eps * u
                        x = torch.zeros_like(gk)
                        r = gk.clone()
                        pdir = r.clone()
                        rr = (r * r).sum()
                        for _ in range(a.riem_cg):
                            Gp = Gv(pdir)
                            al = rr / (pdir * Gp).sum().clamp_min(1e-30)
                            x += al * pdir
                            r -= al * Gp
                            rr_new = (r * r).sum()
                            if rr_new.sqrt() < 0.0001 * gk.norm():
                                break
                            pdir = r + rr_new / rr * pdir
                            rr = rr_new
                        f0 = float(f)
                        slope = -float((gk * x).sum())
                        step = min(step * 2.0, 1000000.0)
                        ok = False
                        for _bt in range(40):
                            (f1, _) = fval(theta - step * x)
                            if f1 <= f0 + 0.0001 * step * slope:
                                ok = True
                                break
                            step *= 0.5
                        if ok:
                            for (q, dq) in zip(PL, unflat(theta - step * x)):
                                q.copy_(dq)
                    CURVE.append((t, it_, float(l2), mu, step if ok else 0.0))
                it[0] = ntot
                STOP.append((t, ntot, ntot, float(Jfun(flat([q.detach() for q in PL]))[0].min()), eps, 'ipm'))
            for it_ in range(0 if a.opt == 'ipm' else a.iters0 if t == 1 else a.iters):
                if a.profile and it_ == WARM:
                    TM.clear()
                    NCG[0] = 0
                    prof.__enter__()
                if a.profile and it_ == WARM + a.profile:
                    prof.__exit__(None, None, None)
                    tot = sum(TM.values())
                    nit = a.profile
                    print(f'[프로파일] {a.method}  반복 {nit} (앞 {WARM} 번 제외)  CG 평균 {NCG[0] / nit:.1f} 회/반복  합계 {1000 * tot / nit:.1f} ms/반복', flush=True)
                    for (k_, v_) in sorted(TM.items(), key=lambda z: -z[1]):
                        print(f'  {k_:28s} {1000 * v_ / nit:9.2f} ms/반복  ({100 * v_ / tot:5.1f}%)', flush=True)
                    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=12), flush=True)
                    print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=12), flush=True)
                    sys.exit(0)
                if a.profile:
                    t_ = tic()
                for q in PL:
                    q.grad = None
                (dy, _) = YJ(X_, FI)
                l2 = ((RES0 + dy) ** 2).sum(1).mean()
                l2.backward()
                gk = flat([q.grad if q.grad is not None else torch.zeros_like(q) for q in PL]).detach()
                theta = flat([q.detach() for q in PL])
                if a.profile:
                    t2_ = tic()
                    TM['손실 앞·뒤 (L2 기울기)'] += t2_ - t_
                    t_ = t2_
                with torch.no_grad():
                    Hv = GN(theta)
                    if a.profile:
                        t2_ = tic()
                        TM['계량 선형화 (make_res+vjp)'] += t2_ - t_
                        t_ = t2_
                    Gv = lambda u: a.riem_k * Hv(u) + eps * u
                    x = torch.zeros_like(gk)
                    r = gk.clone()
                    pdir = r.clone()
                    rr = (r * r).sum()
                    for _ in range(a.riem_cg):
                        Gp = Gv(pdir)
                        al = rr / (pdir * Gp).sum().clamp_min(1e-30)
                        x += al * pdir
                        r -= al * Gp
                        rr_new = (r * r).sum()
                        if a.profile:
                            NCG[0] += 1
                        if rr_new.sqrt() < 0.0001 * gk.norm():
                            break
                        pdir = r + rr_new / rr * pdir
                        rr = rr_new
                    if a.profile:
                        t2_ = tic()
                        TM['CG (jvp+vjp 곱)'] += t2_ - t_
                        t_ = t2_
                    f0 = float(l2)
                    sl = float((gk * x).sum())
                    stp = a.riem_lr
                    nbt = 0
                    if a.ls_max == 0:
                        for (q, dq) in zip(PL, unflat(x)):
                            q -= stp * dq
                    for _bt in range(a.ls_max):
                        for (q, q0, dq) in zip(PL, unflat(theta), unflat(x)):
                            q.copy_(q0 - stp * dq)
                        (dyt, Jt_) = YJ(X_, FI)
                        ft = float(((RES0 + dyt) ** 2).sum(1).mean())
                        if a.feas and ft == ft:
                            ok_ = bool(det3(Jt_).min() > 0)
                            if ok_ and a.method == 'hex':
                                ok_ = bool(rep.cellJ(PL[0].detach())[0].min() > 0)
                            if not ok_:
                                NFEAS[0] += 1
                                ft = float('nan')
                        if ft == ft and ft <= f0 - 0.0001 * stp * sl:
                            break
                        stp *= 0.5
                        nbt += 1
                    else:
                        if a.ls_max:
                            for (q, q0) in zip(PL, unflat(theta)):
                                q.copy_(q0)
                CURVE.append((t, it_, float(l2), stp, nbt))
            if a.opt == 'riem':
                it[0] = it_ + 1
                STOP.append((t, it_ + 1, it_ + 1, float('nan'), eps, 'riem'))
        elif a.opt == 'adam':
            if REB or OPT is None:
                OPT = torch.optim.Adam(rep.params(), lr=a.lr)
            X_ = Pref[FI]
            for it_ in range(a.iters0 if t == 1 else a.iters):
                OPT.zero_grad(set_to_none=True)
                (dy, _) = YJ(X_, FI)
                l2 = ((RES0 + dy) ** 2).sum(1).mean()
                l2.backward()
                OPT.step()
                CURVE.append((t, it_, float(l2), 0.0, 0))
            it[0] = it_ + 1
            STOP.append((t, it_ + 1, it_ + 1, float('nan'), 0.0, 'adam'))
        elif a.opt == 'adam_bt':
            if REB or OPT is None:
                OPT = torch.optim.Adam(rep.params(), lr=a.lr)
            ref = knn_ref(Pref) if REB else REF0
            X_ = Pref[FI]
            nbt_tot = 0
            for it_ in range(a.iters0 if t == 1 else a.iters):
                OPT.zero_grad(set_to_none=True)
                (dy, _) = YJ(X_, FI)
                Y = X_ + dy
                l2 = ((RES0 + dy) ** 2).sum(1).mean()
                bar = torch.relu(a.tau - knn_det(Y, ref)).pow(2).mean()
                (l2 + a.lam_inv * bar).backward()
                prev = [q.detach().clone() for q in rep.params()]
                OPT.step()
                nbt = 0
                with torch.no_grad():
                    for _bt in range(10):
                        (dy2, _) = YJ(X_, FI)
                        if float(knn_det(X_ + dy2, ref).min()) > 0:
                            break
                        nbt += 1
                        for (q, q0) in zip(rep.params(), prev):
                            q.copy_(q0 + 0.5 * (q - q0))
                    else:
                        for (q, q0) in zip(rep.params(), prev):
                            q.copy_(q0)
                nbt_tot += nbt
                CURVE.append((t, it_, float(l2), float(bar), nbt))
            it[0] = it_ + 1
            with torch.no_grad():
                (dy2, _) = YJ(X_, FI)
                KD = knn_det(X_ + dy2, ref)
            STOP.append((t, it_ + 1, it_ + 1, float(KD.min()), float(nbt_tot), 'adam_bt'))
        elif a.opt == 'lbfgs':
            opt = torch.optim.LBFGS(rep.params(), lr=1.0, max_iter=a.max_iter, history_size=50, tolerance_grad=0.0, tolerance_change=0.0, line_search_fn='strong_wolfe')
            opt.step(closure)
            stt = opt.state[opt._params[0]]
            gmax = float(torch.cat([q.grad.reshape(-1) for q in rep.params()]).abs().max())
            dstep = float((stt['d'] * stt['t']).abs().max()) if 'd' in stt else float('nan')
            reason = 'max_iter' if stt['n_iter'] >= a.max_iter else 'max_eval' if stt['func_evals'] >= int(a.max_iter * 1.25) else 'grad=0' if gmax <= 0 else 'step=0' if dstep <= 0 else 'loss 변화 0 / 하강방향 아님'
            STOP.append((t, stt['n_iter'], stt['func_evals'], gmax, dstep, reason))
        else:
            opt = torch.optim.SGD(rep.params(), lr=a.gd_lr)
            for _ in range(a.max_iter):
                opt.step(closure)
            STOP.append((t, a.max_iter, a.max_iter, float('nan'), float('nan'), 'gd'))
        if a.method == 'hex' and (a.hex_diag or a.feas):
            with torch.no_grad():
                (_, Jinc) = YJ(Pref[FI], FI)
                jp = det3(Jinc)
                jq = rep.cellJ(rep.u.detach())[0].reshape(-1, 8)
                badp = jp <= 0
                qpos = (jq > 0).all(1)[rep.pcell[FI]]
                HDIAG.append((t, int(badp.sum()), int((badp & qpos).sum()), int((jq <= 0).sum()), float(jp.min()), float(jq.min()), NFEAS[0]))
            if t % 10 == 0 or t == 1 or HDIAG[-1][1] > 0:
                print(f'  [진단 t={t:3d}] 입자 증분 det<=0 {HDIAG[-1][1]} (그 중 칸 가우스 점 8 개 모두 >0: {HDIAG[-1][2]})  가우스 점 det<=0 {HDIAG[-1][3]}/{jq.numel()}  입자 증분 det 최소 {HDIAG[-1][4]:.3g}  가우스 점 최소 {HDIAG[-1][5]:.3g}  정의역 되돌림 {NFEAS[0]}', flush=True)
        NFEAS[0] = 0
        (yall, Fall, Rall) = eval_all(Pref, Fcum)
        with torch.no_grad():
            y = yall[FI]
            J = torch.linalg.det(Fall[FI])
            rmse = float(((y - tgt) ** 2).sum(1).mean().sqrt()) / L
            cd = chamfer(y, tgt) / L
            e = float('nan')
            if t % a.emd_every == 0 or t == T:
                EMDP.append((t, y.float().cpu().numpy(), tgt.float().cpu().numpy()))
            cdx = float(0.5 * (nn_dist(y, tgt).mean() + nn_dist(tgt, y).mean())) / L
            rows.append((t, rmse, cd, e, float(J.min()), float((J <= 0).float().mean()), cdx))
            vel = (y - YPREV[0]) * 30.0
            com = y.mean(0)
            PHYS.append((t, float(0.5 * (vel * vel).sum(1).mean()), *vel.mean(0).tolist(), *torch.cross(y - com, vel, dim=-1).mean(0).tolist(), float(J.mean())))
            YPREV[0] = y.clone()
            if a.save_traj:
                TRAJS.append((yall if a.traj32 else yall.half()).cpu().numpy())
            if TBW is not None:
                for (k_, v_) in (('RMSE_pct', 100 * rmse), ('CD_pct', 100 * cd), ('detJ_min', rows[-1][4]), ('inverted_pct', 100 * rows[-1][5]), ('dof', rep.dof), ('time_s', time.time() - t0)):
                    TBW.add_scalar('frame/' + k_, v_, t)
                if a.opt == 'adam_bt':
                    TBW.add_scalar('frame/knn_det_min', STOP[-1][3], t)
                    TBW.add_scalar('frame/backtracks', STOP[-1][4], t)
                else:
                    TBW.add_scalar('frame/evals', STOP[-1][2], t)
                    TBW.add_scalar('frame/early_stop', float(STOP[-1][5] == 'step=0'), t)
                for c_ in CURVE[NCUR[0]:]:
                    TBW.add_scalar('iter/l2', c_[2], NCUR[0])
                    TBW.add_scalar('iter/barrier', c_[3], NCUR[0])
                    NCUR[0] += 1
                TBW.flush()
            if REB:
                Fcum = Fall.clone()
                if Rall is not None:
                    Rcum = Rall @ Rcum
                Pref = yall.clone()
            else:
                Fcum = Fall
        if a.save_F:
            FSAVE.append((Rcum if a.method == 'phystwin' else Fall)[:a.save_F].reshape(-1, 9).half().cpu().numpy())
        if RENDER:
            render(yall, Fall=Fall, Rall=Rcum if a.method == 'phystwin' else None)
        if a.ckpt_every > 0 and t % a.ckpt_every == 0 and (t < T):
            if RENDER and FR:
                _fc = os.path.join(FRD, f'f{t:03d}.npy')
                np.save(_fc, np.stack(FR))
                FCHUNKS.append(_fc)
                FR = []
            _L = dict(rows=rows, CURVE=CURVE, DOFS=DOFS, STOP=STOP, EMDP=EMDP, PHYS=PHYS, TRAJS=TRAJS, YPREV=YPREV, FS=FSAVE, HD=HDIAG)
            torch.save(dict(t=t, Pref=Pref, Fcum=Fcum, Rcum=Rcum, L=_L, fchunks=FCHUNKS, elapsed=time.time() - t0, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), CKPT + '.tmp')
            os.replace(CKPT + '.tmp', CKPT)
        if t % 10 == 0 or t == 1:
            print(f'  [t={t:3d}] RMSE {100 * rmse:.3f}%  CD {100 * cd:.3f}%  정확 CD {100 * cdx:.3f}%  ' + (f'EMD {100 * e:.3f}%  ' if e == e else '') + f'det 최소 {rows[-1][4]:.3f} (≤0 {100 * rows[-1][5]:.2f}%)  자유도 {rep.dof}  ' + (f'이웃 det 최소 {STOP[-1][3]:.3f}  되돌림 {int(STOP[-1][4])}  ' if a.opt == 'adam_bt' else f'평가 {it[0]}  종료 {STOP[-1][5]} (반복 {STOP[-1][1]}, |g|max {STOP[-1][3]:.1e}, |step|max {STOP[-1][4]:.1e})  ') + f'{time.time() - t0:.0f}s', flush=True)
    if RENDER:
        WR = imageio.get_writer(a.video, fps=30, codec='libx264', quality=8)
        for _fc in FCHUNKS:
            for _im in np.load(_fc):
                WR.append_data(_im)
        for _im in FR:
            WR.append_data(_im)
        WR.close()
    R = np.array(rows)
    print(f'[요약] {a.method}  자유도 {np.mean(DOFS):.0f}  RMSE {100 * R[:, 1].mean():.3f}%  CD {100 * R[:, 2].mean():.3f}%  det 최소 {R[:, 4].min():.4f}  뒤집힘 최대 {100 * R[:, 5].max():.2f}%  정확 CD {100 * R[:, 6].mean():.3f}%', flush=True)
    np.savez_compressed(a.out, metrics=R, dof=np.array(DOFS), L=L, curve=np.array(CURVE, dtype=np.float64), stop=np.array([q[:5] for q in STOP], dtype=np.float64), stop_reason=np.array([q[5] for q in STOP]), emd_t=np.array([q[0] for q in EMDP]), emd_y=np.stack([q[1] for q in EMDP]), emd_tgt=np.stack([q[2] for q in EMDP]), phys=np.array(PHYS, dtype=np.float64), **{'traj': np.stack(TRAJS)} if TRAJS else {}, **{'Fs': np.stack(FSAVE)} if FSAVE else {}, **{'hex_diag': np.array(HDIAG, dtype=np.float64)} if HDIAG else {}, **{'hex_edge_sim': rep.edge_sim, 'hex_edge_norm': rep.h, 'hex_dx': a.hex_dx, 'hex_cell': a.hex_cell, 'gauss_note': 'det 는 칸당 2x2x2 가우스 점에서만 장벽; 칸 안 모든 점 det>0 보장 아님'} if a.method == 'hex' else {})
    print(f'[저장] {a.out}' + (f'  영상 {a.video}' if a.video else ''), flush=True)
