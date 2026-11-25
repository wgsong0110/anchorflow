"""Evaluation: substep stability, speed-accuracy physics / visual metrics, 3DGS rendering, zero-energy modes.

Subcommands (old script -> min_aux file -> new subcommand):
  analyze            exe/stab1115_an.py     evaluation/stability.py analyze  one scene: minimum stable N, reference trajectory, N-error curve,
                                                                              physics checks -> an_<scene>.json
  extra              exe/stab_extra.py      evaluation/stability.py extra    extra physics (energy, volume), no-reference visual metrics
  is-stable          exe/stab_is_stable.py  evaluation/stability.py is-stable  prints 1 / 0 / - for one run
  metrics            exe/stab_metrics.py    evaluation/stability.py metrics  all metrics of the early stability sweep
  stability-vis      exe/stab_vis.py        evaluation/stability.py vis      PSNR / SSIM / LPIPS against the PhysGaussian N=1024 rendering
  phys               exe/ffx_phys.py        evaluation/phys_vis.py phys      no-ground-truth physics metrics of one speed-accuracy cell
  pareto-vis         exe/ffx_vis.py         evaluation/phys_vis.py vis       no-reference IQA, RAFT warping error, PSNR/SSIM/LPIPS vs. ground-truth rendering
  render             exe/lb_render.py       evaluation/render.py render      trajectory folder (meta.json, sim_*.h5 with x, F) -> video / PNG frames
  multiview          exe/stab_mv_render.py  evaluation/render.py multiview   ground-truth trajectory -> Spring-Gaus multi-view input
  roll-to-h5         exe/lb_roll_to_h5.py   evaluation/render.py roll-to-h5  dump pred.pt {pred, F} -> trajectory folder read by `render`
  zero-energy-modes  exe/q27_eig.py         (new)                            zero / near-zero eigenmodes of the first-iterate elastic Hessian
Library (importable: `from evaluation import subset, frame_stats, KEYS`): frame statistics and safe-mode helpers (old fusedgs/stabstat.py),
result loading and trajectory metrics (old exe/stab_metrics.py), scene constants and the stability judgement (old exe/stab1115_lib.py),
RAFT warping error, renderer helpers (cell overlay = old exe/cell_overlay.py, particle marking = old exe/dc_mark.py).
Baked-in boot-time patches (old paper_wt/exe/pr2_patch_models.py): MODEL += vasedeck / plane / pillow2sofa; multiview crops the Gaussians
to the config sim_area for vasedeck / plane / pillow2sofa (render already crops to sim_area for every scene unless LB_NO_SIMAREA=1).
"""
import math
import os
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib')
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
import json
import re
import numpy as np
NSUB = 32768

def subset(n, k=NSUB, seed=0):
    """모든 방법·모든 서브스텝 수에서 같은 번호 (같은 입력 입자 순서 기준)."""
    return np.sort(np.random.default_rng(seed).choice(n, min(k, n), replace=False))
KEYS = ('ke', 'pe', 'nan', 'out', 'vmax', 'detneg', 'detmin', 'ee', 'jdev', 'jdev10', 'penr', 'pend', 'px', 'py', 'pz')
SAFE_KEYS = ('safe_n', 'safe_split', 'safe_inert')

def elastic_fcr(F, vol, E, nu):
    """고정 공회전(fixed corotated) 탄성 에너지 합 Σ V (μ‖F−R‖² + λ/2 (J−1)²). 방법마다 구성모델이 조금씩 달라도
    같은 자로 비교하려고 하나로 고정한다 (PG jelly 와 같은 모델)."""
    (mu, lam) = (E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)))
    (U, S, Vt) = np.linalg.svd(F)
    R = U @ Vt
    J = np.prod(S, axis=1)
    psi = mu * ((F - R) ** 2).sum((1, 2)) + 0.5 * lam * (J - 1) ** 2
    return float((vol * psi).sum())

def frame_stats(x, v, F, mass, g, floor_z, box=(0.0, 2.0), vol=None, E=None, nu=None):
    x = np.asarray(x, np.float64)
    v = np.asarray(v, np.float64)
    fin = np.isfinite(x).all(1) & np.isfinite(v).all(1)
    out = fin & ((x < box[0]) | (x > box[1])).any(1)
    (xm, vm, mm) = (x[fin], v[fin], mass[fin])
    gz = -float(g[2])
    ke = 0.5 * float((mm * (vm * vm).sum(1)).sum())
    pe = gz * float((mm * (xm[:, 2] - floor_z)).sum())
    st = dict(ke=ke, pe=pe, nan=float(1 - fin.mean()), out=float(out.mean()), vmax=float(np.sqrt((vm * vm).sum(1)).max()) if len(vm) else float('nan'))
    if F is not None:
        F = np.asarray(F, np.float64).reshape(-1, 3, 3)
        d = np.linalg.det(F[fin]) if fin.any() else np.array([np.nan])
        st['detneg'] = float((d <= 0).mean())
        st['detmin'] = float(np.nanmin(d))
        st['jdev'] = float(np.nanmean(np.abs(d - 1)))
        st['jdev10'] = float((np.abs(d - 1) > 0.1).mean())
        if vol is not None and E is not None and fin.any():
            st['ee'] = elastic_fcr(F[fin], np.asarray(vol, np.float64)[fin], float(E), float(nu))
    if len(xm):
        dep = float(floor_z) - xm[:, 2]
        st['penr'] = float((dep > 0).mean())
        st['pend'] = float(max(dep.max(), 0.0))
        p = (mm[:, None] * vm).sum(0)
        (st['px'], st['py'], st['pz']) = (float(q) for q in p)
    return st
SIG_HI = 100.0

def clamp_F(F):
    """F [N,3,3] torch 제자리: 특이값이 SIG_HI 를 넘는 입자만 U diag(min(σ, SIG_HI)) Vᵀ 로 바꾼다 (뒤집힘 부호는 U·V 에 남는다).
    -> 바꾼 입자 수"""
    import torch
    F3 = F.reshape(-1, 3, 3)
    ok = torch.isfinite(F3.reshape(F3.shape[0], -1)).all(1)
    if not bool(ok.any()):
        return 0
    S = torch.linalg.svdvals(torch.where(ok[:, None, None], F3, torch.eye(3, device=F3.device, dtype=F3.dtype)))
    bad = ok & (S > SIG_HI).any(1)
    n = int(bad.sum())
    if n:
        (U, S2, Vh) = torch.linalg.svd(F3[bad])
        F3[bad] = U @ torch.diag_embed(S2.clamp(max=SIG_HI)) @ Vh
    return n

def safe_params(x0, floor_z, g, grid_lim, n_grid):
    gz = abs(float(g[2])) if hasattr(g, '__len__') else abs(float(g))
    H = float(np.asarray(x0)[:, 2].max() - floor_z)
    dx = float(grid_lim) / int(n_grid)
    return dict(lo=3 * dx, hi=float(grid_lim) - 3 * dx, vcap=10.0 * np.sqrt(2 * gz * max(H, 1e-06)))

def safe_fix(x, v, F, x_prev, sp):
    """torch 텐서 제자리 수정. x,v [N,3], F [N,3,3] 또는 None, x_prev [N,3]. -> 손댄 입자 수 (int)"""
    import torch
    bad = ~(torch.isfinite(x).all(1) & torch.isfinite(v).all(1))
    if F is not None:
        bad = bad | ~torch.isfinite(F.reshape(F.shape[0], -1)).all(1)
    n = int(bad.sum())
    if n:
        x[bad] = x_prev[bad]
        v[bad] = 0.0
        if F is not None:
            F[bad] = torch.eye(3, device=F.device, dtype=F.dtype)
    sp_ = v.norm(dim=1)
    over = sp_ > sp['vcap']
    m = int(over.sum())
    if m:
        v[over] *= (sp['vcap'] / sp_[over]).unsqueeze(1)
    out = ((x < sp['lo']) | (x > sp['hi'])).any(1)
    k = int(out.sum())
    if k:
        x.clamp_(sp['lo'], sp['hi'])
    return n + m + k

def load(res, m, n, nf=60):
    """-> dict(status, rows{key:[T]}, X[T,S,3]) 또는 None (실행 없음)"""
    (npz, log) = (f'{res}/{m}_N{n}.npz', f'{res}/{m}_N{n}.log')
    ck = next((p for p in (f'{res}/{m}_N{n}.npz.ckpt.pt', f'{res}/{m}_N{n}.main.npz.ckpt.pt') if os.path.exists(p)), None)
    if os.path.exists(npz):
        z = np.load(npz)
        keys = [str(k) for k in z['keys']]
        rows = {k: z['stats'][:, i] for (i, k) in enumerate(keys)}
        st = '끝'
        if np.nanmax(rows['nan']) > 0 or len(z['x']) < nf + 1:
            st = '발산'
        return dict(status=st, rows=rows, X=z['x'])
    died = os.path.exists(log) and re.search('illegal memory|Error|Traceback|발산', open(log, errors='ignore').read())
    if ck:
        import torch
        c = torch.load(ck, map_location='cpu', weights_only=False)
        (R, X) = (c['stab']['STROWS'], c['stab']['STX']) if c.get('stab') else (c['rows'], c['XS'])
        keys = sorted({k for r in R for k in r})
        return dict(status='발산' if died else '진행', rows={k: np.array([r.get(k, np.nan) for r in R]) for k in keys}, X=np.stack([np.asarray(x) for x in X]))
    if os.path.exists(log):
        txt = open(log, errors='ignore').read()
        if re.search('illegal memory|Error|error|Traceback|발산', txt):
            return dict(status='발산', rows=None, X=None)
    return None

def metrics(r, ref, L, zf, g, dt, M, dxg):
    (rows, X) = (r['rows'], r['X'])
    out = {}
    if rows is not None:
        E = rows['ke'] + rows['pe'] + (np.nan_to_num(rows['ee']) if 'ee' in rows and np.isfinite(rows['ee']).any() else 0.0)
        out['e_drift_max'] = float(np.nanmax((E - E[0]) / abs(E[0])))
        out['e_drift_last'] = float((E[-1] - E[0]) / abs(E[0]))
        out['e_has_elastic'] = bool('ee' in rows and np.isfinite(rows['ee']).any())
        out['detneg_max'] = float(np.nanmax(rows['detneg'])) if 'detneg' in rows else None
        for k in ('jdev', 'jdev10'):
            out[k + '_max'] = float(np.nanmax(rows[k])) if k in rows and np.isfinite(rows[k]).any() else None
    if X is not None and len(X) >= 3:
        z = X[..., 2]
        out['pen_ratio_max'] = float((z < zf).mean(1).max())
        out['pen_depth_max'] = float(max((zf - z).max(), 0.0) / L)
        d2 = np.linalg.norm(X[2:] - 2 * X[1:-1] + X[:-2], axis=-1)
        out['jitter_d2'] = float(np.nanmean(d2) / L)
        v = X[1:] - X[:-1]
        if len(v) >= 8:
            P = np.abs(np.fft.rfft(v - v.mean(0), axis=0)) ** 2
            P = P.sum(-1)
            h = P.shape[0] // 2
            out['jitter_hf'] = float(np.nanmean(P[h:].sum(0) / np.maximum(P.sum(0), 1e-30)))
        hit = np.where(z.min(1) < zf + 1.5 * dxg)[0]
        T_c = int(hit[0]) if len(hit) else len(z)
        out['freefall_frames'] = T_c
        if rows is not None:
            ke = rows['ke']
            errs = [abs(np.sqrt(2 * M * max(ke[t], 0.0)) - M * abs(g) * t * dt) / (M * abs(g) * t * dt) for t in range(1, min(T_c, len(ke)))]
            out['freefall_mom_err'] = float(max(errs)) if errs else None
        if ref is not None:
            from scipy.spatial import cKDTree
            T = min(len(X), len(ref))
            cds = []
            for t in range(1, T):
                if not np.isfinite(X[t]).all():
                    break
                (ka, kb) = (cKDTree(X[t]), cKDTree(ref[t]))
                cds.append(0.5 * (kb.query(X[t])[0].mean() + ka.query(ref[t])[0].mean()) / L)
            out['cd_mpm'] = float(np.mean(cds)) if cds else None
            out['cd_frames'] = len(cds)
    return out

def metrics_main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('FUSEDGS_WORK', '/workspace') + '/stab')
    ap.add_argument('--out', required=True)
    ap.add_argument('--nmax', type=int, default=128)
    a = ap.parse_args()
    NS = [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if n <= a.nmax]
    res_all = {}
    for S in ('lego', 'mic', 'ficus'):
        res = f'{a.root}/{S}/res'
        cfg = json.load(open(f'{a.root}/{S}/config.json'))
        zf = float([b for b in cfg['boundary_conditions'] if b['type'] == 'surface_collider'][0]['point'][2])
        g = float(cfg['g'][2])
        dt = float(cfg['frame_dt'])
        import h5py
        with h5py.File(f'{a.root}/{S}/init/sim_0000000000.h5') as h:
            x0 = np.array(h['x'])
            x0 = x0.T if x0.shape[0] == 3 else x0
        dxg = float(cfg['grid_lim']) / int(cfg['n_grid'])
        M = float(cfg['density']) * dxg ** 3 * len(np.unique(np.floor(x0 / dxg).astype(np.int64), axis=0))
        refr = load(res, 'pg', 1024)
        ref = refr['X'] if refr else None
        L = float(np.linalg.norm(ref[0].max(0) - ref[0].min(0)))
        res_all[S] = {}
        for m in ('pg', 'gf', 'ipg', 'ours', 'ovs', 'pg_safe', 'gf_safe', 'ipg_safe'):
            res_all[S][m] = {}
            for n in NS:
                r = load(res, m, n)
                if r is None:
                    continue
                d = dict(status=r['status'])
                if r['rows'] is not None and 'safe_n' in r['rows']:
                    d['safe_n'] = float(np.nanmax(r['rows']['safe_n']))
                    d['safe_split'] = float(np.nanmax(r['rows']['safe_split']))
                    d['safe_inert'] = float(np.nanmax(r['rows']['safe_inert']))
                d.update(metrics(r, ref, L, zf, g, dt, M, dxg))
                if d['status'] == '끝' and (not m.endswith('_safe')) and (not np.isfinite(d.get('e_drift_max', 0.0)) or d.get('e_drift_max', 0.0) > 1.0 or (d.get('pen_depth_max') or 0.0) > 0.5):
                    d['status'] = '발산'
                res_all[S][m][n] = d
                print(S, m, n, d['status'], {k: round(v, 6) if isinstance(v, float) else v for (k, v) in d.items() if k != 'status'}, flush=True)
    json.dump(res_all, open(a.out, 'w'), indent=1)
    print('METRICS_DONE', flush=True)
ST = os.environ.get('S1115_ROOT', os.environ.get('FUSEDGS_WORK', '/workspace') + '/stab')
REPS = ('a', 'b', 'c')
POW = [2 ** i for i in range(12)]
NSHAPE = ('ship_plastic_lego', 'mic_metal_lego')

def nframes(S):
    """[stabfinal] 장면 프레임 수 = config frame_num (기존 낙하 장면 60, 새 형상 80)."""
    try:
        return int(json.load(open(f'{ST}/{S}/config.json'))['frame_num'])
    except Exception:
        return 60

def res_dir(S, rep):
    return f'{ST}/{S}/res' if rep == 'a' else f'{ST}/{S}/res_{rep}'

def hex_dir(S, n, rep):
    return f'{ST}/{S}/hex/{S}_N{n}_{rep}'

def hex_load(d, nf=60):
    p = f'{d}/res.npz'
    if not os.path.exists(p):
        return None
    z = np.load(p)
    keys = [str(k) for k in z['keys']]
    rows = {k: z['stats'][:, i] for (i, k) in enumerate(keys)}
    st = '끝' if len(z['x']) >= nf + 1 and np.nanmax(rows['nan']) == 0 and np.isfinite(z['x']).all() else '발산'
    stop = str(z['stop']) if 'stop' in z.files else ''
    if st == '발산' and 'CUDA error: out of memory' in stop and (len(z['x']) <= 3):
        st = '메모리부족'
    return dict(status=st, rows=rows, X=z['x'], stop=stop, K=int(z['substeps']))

def load_run(S, m, n, rep):
    if m == 'ours':
        return hex_load(hex_dir(S, n, rep), nframes(S))
    return load(res_dir(S, rep), m, n, nframes(S))

def scene_consts(S):
    cfg = json.load(open(f'{ST}/{S}/config.json'))
    zf = float([b for b in cfg['boundary_conditions'] if b['type'] == 'surface_collider'][0]['point'][2])
    return (cfg, zf, float(cfg['g'][2]), float(cfg['frame_dt']), float(cfg['grid_lim']) / int(cfg['n_grid']))

def judge(r, S):
    """안정 = 전 프레임 (60 또는 80) 끝 + 운동+위치 에너지 변화 최대 <= 1 + 바닥 관통 <= 0.5 L (stabhex_an.judge 와 같음). -> (bool|None, 지표 dict)"""
    if r is None or r['status'] == '진행':
        return (None, {})
    if r['status'] != '끝':
        return (False, {})
    (cfg, zf, g, dt, dxg) = scene_consts(S)
    X = r['X']
    L = float(np.linalg.norm(X[0].max(0) - X[0].min(0)))
    rr = dict(r)
    rr['rows'] = {k: v for (k, v) in r['rows'].items() if k != 'ee'} if r['rows'] is not None else None
    mt = metrics(rr, None, L, zf, g, dt, 1.0, dxg)
    ok = bool(np.isfinite(mt.get('e_drift_max', 0.0)) and mt.get('e_drift_max', 0.0) <= 1.0 and ((mt.get('pen_depth_max') or 0.0) <= 0.5))
    return (ok, mt)

def load_raft(dev):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    W8 = Raft_Large_Weights.C_T_SKHT_V2
    return (raft_large(weights=W8, progress=False).to(dev).eval(), W8.transforms())

def _pad8(x):
    import torch.nn.functional as Fn
    (H, W) = x.shape[-2:]
    return (Fn.pad(x, (0, (8 - W % 8) % 8, 0, (8 - H % 8) % 8), mode='replicate'), H, W)

def _flow(RAFT, TF, i1, i2):
    (a_, b_) = TF(i1, i2)
    (a_, H, W) = _pad8(a_)
    (b_, _, _) = _pad8(b_)
    return RAFT(a_, b_)[-1][..., :H, :W]

def _warp(x, f):
    """x [1,C,H,W] 를 f [1,2,H,W] (x 좌표 = 픽셀 + f) 에서 끌어옴. -> (끌어온 값, 영상 안 여부)"""
    import torch
    import torch.nn.functional as Fn
    (_, _, H, W) = x.shape
    (yy, xx) = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
    gx = xx[None] + f[:, 0]
    gy = yy[None] + f[:, 1]
    inside = (gx >= 0) & (gx <= W - 1) & (gy >= 0) & (gy <= H - 1)
    grid = torch.stack([2 * gx / (W - 1) - 1, 2 * gy / (H - 1) - 1], -1)
    return (Fn.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True), inside)

def warp_error(prev, cur, RAFT, TF):
    """연속 두 프레임 [1,3,H,W] (0~1) -> (warping error, 가림 비율)."""
    import torch
    with torch.no_grad():
        fb = _flow(RAFT, TF, cur, prev)
        ff = _flow(RAFT, TF, prev, cur)
        (wprev, ins) = _warp(prev, fb)
        (ffw, _) = _warp(ff, fb)
        Mk = (((fb + ffw) ** 2).sum(1) < 0.01 * ((fb ** 2).sum(1) + (ffw ** 2).sum(1)) + 0.5) & ins
        e = ((cur - wprev) ** 2).sum(1)
        return (float((e * Mk).sum() / Mk.sum().clamp_min(1)), float(1 - Mk.float().mean()))
W = os.environ.get('FUSEDGS_WORK', '/workspace')
MODEL = {'vasedeck': 'vasedeck_whitebg-trained', 'plane': 'plane-trained', 'pillow2sofa': 'pillow2sofa_whitebg-trained', 'lego': 'lego_whitebg-trained', 'mic': 'mic_whitebg-trained', 'ficus': 'ficus_whitebg-trained', 'hotdog': 'hotdog_whitebg-trained', 'wolf': 'wolf_whitebg-trained', 'bread': 'bread-trained'}
MODEL.update({'mic_elastic': 'mic', 'ship_elastic': 'ship', 'chair_elastic': 'chair', 'ship_plastic': 'ship', 'mic_plastic': 'mic', 'mic_metal': 'mic', 'ship_plastic_lego': 'ship', 'mic_metal_lego': 'mic', 'hotdog_plastic_lego': 'hotdog'})
MODEL.update({'chair_plasticine': 'chair', 'drums_foam': 'drums', 'materials_sand': 'materials', 'materials_metal': 'materials', 'ball_metal': 'ball'})
MODEL.update({'chair_paste': 'chair', 'drums_honey': 'drums', 'bread_foam_drop': 'bread-trained', 'chair_paste_lego': 'chair', 'drums_honey_lego': 'drums', 'bread_foam_drop_lego': 'bread-trained'})

def cfg_rotation(cfg, dev):
    """PG 전처리 회전 (config rotation_degree/axis -- PG generate_rotation_matrix 와 같은 식, apply_rotations 순서대로) -> (R, 회전 있음).
    없으면 항등."""
    _RT = torch.eye(3, device=dev)
    for (_dg, _ax) in zip(cfg.get('rotation_degree') or [], cfg.get('rotation_axis') or []):
        (_c, _s) = (math.cos(_dg / 180.0 * 3.1415926), math.sin(_dg / 180.0 * 3.1415926))
        _R = {0: [[1, 0, 0], [0, _c, -_s], [0, _s, _c]], 1: [[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]], 2: [[_c, -_s, 0], [_s, _c, 0], [0, 0, 1]]}[int(_ax)]
        _RT = torch.tensor(_R, device=dev, dtype=torch.float32) @ _RT
    return (_RT, bool((_RT - torch.eye(3, device=dev)).abs().max() > 0))

def cov6_to_mat(c6):
    """3DGS 공분산 6 성분 [NG,6] -> [NG,3,3]."""
    (NG, dev) = (c6.shape[0], c6.device)
    C0 = torch.zeros(NG, 3, 3, device=dev)
    (C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2]) = (c6[:, 0], c6[:, 1], c6[:, 2])
    (C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2]) = (c6[:, 3], c6[:, 4], c6[:, 5])
    (C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1]) = (c6[:, 1], c6[:, 2], c6[:, 4])
    return C0
FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]

def draw_cells(fr, px, dep, cols, alpha):
    from PIL import Image, ImageDraw
    out = fr.astype(np.float32)
    (H, W) = fr.shape[:2]
    for ci in np.argsort(-dep.mean(1)):
        (P8, d8) = (px[ci], dep[ci])
        if not np.isfinite(P8).all():
            continue
        c = np.asarray(cols[ci], np.float32)
        for fc in sorted(FACES, key=lambda q: -d8[list(q)].mean()):
            poly = P8[list(fc)]
            (x0, y0) = np.floor(poly.min(0)).astype(int)
            (x1, y1) = np.ceil(poly.max(0)).astype(int)
            (x0, y0) = (max(x0, 0), max(y0, 0))
            (x1, y1) = (min(x1, W - 1), min(y1, H - 1))
            if x1 < x0 or y1 < y0:
                continue
            m = Image.new('L', (x1 - x0 + 1, y1 - y0 + 1), 0)
            dr = ImageDraw.Draw(m)
            pts = [(float(p[0] - x0), float(p[1] - y0)) for p in poly]
            dr.polygon(pts, fill=1)
            mk = np.asarray(m).astype(bool)
            sub = out[y0:y1 + 1, x0:x1 + 1]
            sub[mk] = sub[mk] * (1 - alpha) + c * alpha
            e = Image.new('L', m.size, 0)
            ImageDraw.Draw(e).line(pts + [pts[0]], fill=1, width=1)
            ek = np.asarray(e).astype(bool)
            sub[ek] = c * 0.7
    return np.clip(out, 0, 255).astype(np.uint8)
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
RED = (1.0, 0.0, 1.0)

def sh_rgb(shs, pos, campos):
    """shs [N,16,3] (GaussianModel.get_features 순서), pos [N,3], campos [3] -> rgb [N,3]."""
    d = pos - campos[None]
    d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
    (x, y, z) = (d[:, 0:1], d[:, 1:2], d[:, 2:3])
    s = shs
    r = SH_C0 * s[:, 0]
    r = r - SH_C1 * y * s[:, 1] + SH_C1 * z * s[:, 2] - SH_C1 * x * s[:, 3]
    (xx, yy, zz, xy, yz, xz) = (x * x, y * y, z * z, x * y, y * z, x * z)
    r = r + SH_C2[0] * xy * s[:, 4] + SH_C2[1] * yz * s[:, 5] + SH_C2[2] * (2.0 * zz - xx - yy) * s[:, 6] + SH_C2[3] * xz * s[:, 7] + SH_C2[4] * (xx - yy) * s[:, 8]
    r = r + SH_C3[0] * y * (3 * xx - yy) * s[:, 9] + SH_C3[1] * xy * z * s[:, 10] + SH_C3[2] * y * (4 * zz - xx - yy) * s[:, 11] + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * s[:, 12] + SH_C3[4] * x * (4 * zz - xx - yy) * s[:, 13] + SH_C3[5] * z * (xx - yy) * s[:, 14] + SH_C3[6] * x * (xx - 3 * yy) * s[:, 15]
    return (r + 0.5).clamp_min(0.0)

def load_mark(path, ng, dev):
    idx = torch.as_tensor(np.load(path).astype(np.int64), device=dev)
    return (idx[idx < ng], idx[idx >= ng])

def extra_cov(F, s):
    """F [K,3,3], s 축 길이 -> 6 성분 공분산."""
    cov = s * s * (F @ F.transpose(1, 2))
    return (torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1), cov)

def project_px(pos, full_proj, W, H):
    """pos [K,3] -> 픽셀 [K,2] (full_proj = 행 벡터 규약 [4,4], 3DGS 와 같다)."""
    ph = torch.cat([pos, torch.ones_like(pos[:, :1])], 1) @ full_proj
    nd = ph[:, :3] / (ph[:, 3:4] + 1e-07)
    return torch.stack([((nd[:, 0] + 1.0) * W - 1.0) * 0.5, ((nd[:, 1] + 1.0) * H - 1.0) * 0.5], 1)
COMMANDS = ('analyze', 'extra', 'is-stable', 'metrics', 'stability-vis', 'phys', 'pareto-vis', 'render', 'multiview', 'roll-to-h5', 'zero-energy-modes')
if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit('usage: python exe/evaluation.py {' + '|'.join(COMMANDS) + '} [options]')
    CMD = sys.argv.pop(1)
else:
    CMD = None
if CMD == 'analyze':
    import argparse, glob, json, os, sys
    import numpy as np
    import h5py
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tol', type=float, default=0.005)
    ap.add_argument('--hextag', default='')
    a = ap.parse_args()
    S = a.scene
    (cfg, zf, g, dt, dxg) = scene_consts(S)
    FR = nframes(S)
    with h5py.File(f'{ST}/{S}/init/sim_0000000000.h5') as h:
        x0 = np.array(h['x'])
        x0 = x0.T if x0.shape[0] == 3 else x0
    sub = subset(len(x0))
    Lc = float(np.linalg.norm(x0[sub].max(0) - x0[sub].min(0)))
    H = float(x0[:, 2].max() - zf)
    vcap = 10.0 * np.sqrt(2 * abs(g) * H)
    Mtot = float(cfg['density']) * dxg ** 3 * len(np.unique(np.floor(x0 / dxg).astype(np.int64), axis=0))
    refr = load_run(S, 'pg', 1024, 'a')
    PREF = refr['X']
    LP = float(np.linalg.norm(PREF[0].max(0) - PREF[0].min(0)))
    REFM = metrics(refr, PREF, LP, zf, g, dt, Mtot, dxg)
    THR = dict(detneg=0.0, pen_mult=2.0, pen_floor=0.01, jit_mult=2.0, ff=0.05, cd=0.05)
    SO = dict(scene=S, L=Lc, n_particles=int(len(x0)), floor_z=zf, phys_ref=REFM, methods={})

    def rmse(XA, XB):
        T = min(len(XA), len(XB))
        return np.sqrt(((XA[:T].astype(np.float64) - XB[:T].astype(np.float64)) ** 2).sum(-1).mean(-1)) / Lc

    def all_ns(m):
        ns = set()
        for rep in REPS:
            if m == 'ours':
                for p in glob.glob(f'{ST}/{S}/hex/{S}_N*_{rep}{a.hextag}'):
                    try:
                        ns.add(int(os.path.basename(p).split('_N')[1].split('_')[0]))
                    except Exception:
                        pass
            else:
                for p in glob.glob(res_dir(S, rep) + f'/{m}_N*.fin'):
                    ns.add(int(os.path.basename(p)[len(m) + 2:-4]))
        return sorted(ns)

    def loadrep(m, n, rep):
        if m == 'ours':
            return hex_load(hex_dir(S, n, rep) + a.hextag, FR)
        return load_run(S, m, n, rep)
    for m in ('pg', 'ipg', 'omni', 'ours'):
        ns = all_ns(m)
        if not ns:
            continue
        M = dict(runs={}, ref=None, ref_pairs=[], err={}, floor={}, N_min=None)
        byN = {}
        for n in ns:
            for rep in REPS:
                r = loadrep(m, n, rep)
                if r is None or r['status'] == '진행':
                    continue
                (ok, mt) = judge(r, S)
                rec = dict(N=n, rep=rep, status=r['status'], stable=ok, metrics={k: float(v) if isinstance(v, (int, float, np.floating)) and v is not None else v for (k, v) in mt.items()})
                if r['status'] == '끝':
                    mt2 = metrics(r, PREF, LP, zf, g, dt, Mtot, dxg)
                    rec['phys'] = {'detneg': [mt2.get('detneg_max'), (mt2.get('detneg_max') or 0.0) <= THR['detneg']], 'pen': [mt2.get('pen_depth_max'), (mt2.get('pen_depth_max') or 0.0) <= max(THR['pen_mult'] * (REFM.get('pen_depth_max') or 0.0), THR['pen_floor'])], 'jit': [mt2.get('jitter_d2'), (mt2.get('jitter_d2') or 0.0) <= THR['jit_mult'] * REFM['jitter_d2']], 'ff': [mt2.get('freefall_mom_err'), None if mt2.get('freefall_mom_err') is None else mt2['freefall_mom_err'] <= THR['ff']], 'cd': [mt2.get('cd_mpm'), (mt2.get('cd_mpm') or 1.0) <= THR['cd']]}
                M['runs'][f'{m}_N{n}_{rep}'] = rec
                byN.setdefault(n, []).append((rep, r, rec))
        st = {}
        for (n, lst) in byN.items():
            if all((rec['status'] != '메모리부족' for (_, _, rec) in lst)):
                st[n] = [bool(rec['stable']) for (_, _, rec) in lst]
        M['stable_by_N'] = {n: [int(sum(v)), len(v)] for (n, v) in sorted(st.items())}
        good = [n for (n, v) in sorted(st.items()) if all(v) and len(v) >= (3 if m != 'x' else 1)]
        pl = f"{ST}/{S}/plan_{m}{(a.hextag if m == 'ours' else '')}.json"
        if os.path.exists(pl):
            M['N_min'] = json.load(open(pl)).get('N_min')
        if M['N_min'] is None and good:
            M['N_min'] = good[0]
        fin = {n: [r for (rep, r, rec) in lst if rep == 'a'][0]['X'] for (n, lst) in byN.items() if any((rep == 'a' and r['status'] == '끝' and np.isfinite(r['X']).all() for (rep, r, rec) in lst)) and n & n - 1 == 0}
        Ns = sorted(fin)
        for n in Ns:
            if 2 * n in fin:
                d = float(rmse(fin[n], fin[2 * n])[1:FR + 1].mean())
                M['ref_pairs'].append([n, 2 * n, d])
                if M['ref'] is None and d < a.tol:
                    M['ref'] = 2 * n
        M['ref_converged'] = M['ref'] is not None
        if M['ref'] is None and Ns:
            c_ = Ns[0]
            while 2 * c_ in fin:
                c_ *= 2
            M['ref'] = c_
        if M['ref'] is not None:
            XR = fin[M['ref']]
            for n in Ns:
                e = rmse(fin[n], XR)
                M['err'][n] = dict(mean=float(e[1:FR + 1].mean()), last=float(e[min(FR, len(e) - 1)]))
        for (n, lst) in byN.items():
            ok_ = [r for (rep, r, rec) in lst if r['status'] == '끝']
            if len(ok_) >= 2:
                e = rmse(ok_[0]['X'], ok_[1]['X'])
                M['floor'][n] = dict(mean=float(e[1:FR + 1].mean()), max=float(e[1:FR + 1].max()))
        SO['methods'][m] = M
        print(S, m, 'N_min', M['N_min'], '기준', M['ref'], '수렴' if M['ref_converged'] else '미수렴', flush=True)

    def _j(o):
        if isinstance(o, dict):
            return {str(k): _j(v) for (k, v) in o.items()}
        if isinstance(o, (list, tuple)):
            return [_j(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    json.dump(_j(SO), open(a.out, 'w'), ensure_ascii=False)
    print('[stab1115 분석] ->', a.out)
if CMD == 'extra':
    import argparse, glob, json, os, sys
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--hextag', default='_f5')
    ap.add_argument('--no_visual', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    S = a.scene
    D = f'{ST}/{S}'
    V = f'{D}/vis'
    (cfg, zf, g, dt, dxg) = scene_consts(S)
    FR = nframes(S)
    OUT = a.out or f'{D}/extra.json'
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}

    def items():
        it = [('pg_N1024', 'pg', 1024)]
        for m in ('pg', 'ipg', 'omni', 'ours'):
            p = f"{D}/plan_{m}{(a.hextag if m == 'ours' else '')}.json"
            if os.path.exists(p):
                n = json.load(open(p)).get('N_min')
                if n:
                    it.append((f"{m}{(a.hextag if m == 'ours' else '')}_N{n}", m, int(n)))
        return it

    def load_extra(m, n, rep):
        if m == 'ours':
            return hex_load(hex_dir(S, n, rep) + a.hextag, FR)
        r = load_run(S, m, n, rep)
        if r is None and rep == 'a' and os.path.exists(f'{V}/{m}_N{n}.npz'):
            r = load(V, m, n, FR)
        return r

    def phys(r):
        o = {}
        (rows, X) = (r.get('rows'), r.get('X'))
        if rows is not None and 'ke' in rows and ('pe' in rows):
            E = np.asarray(rows['ke'], np.float64) + np.asarray(rows['pe'], np.float64)
            o['energy_inc'] = float(max(0.0, np.nanmax((E - E[0]) / abs(E[0]))))
            o['e_drift_max'] = float(np.nanmax((E - E[0]) / abs(E[0])))
            o['e_ratio_last'] = float(E[-1] / E[0])
        if rows is not None and 'jdev' in rows and np.isfinite(rows['jdev']).any():
            o['jdev_max'] = float(np.nanmax(rows['jdev']))
        if rows is not None and 'out' in rows:
            o['out_max'] = float(np.nanmax(rows['out']))
        if rows is not None and 'detneg' in rows and np.isfinite(rows['detneg']).any():
            o['detneg_max'] = float(np.nanmax(rows['detneg']))
        if X is not None and len(X) > 1:
            h = 2 * dxg
            vs = []
            for t in range(len(X)):
                x = X[t]
                if not np.isfinite(x).all():
                    break
                vs.append(len(np.unique(np.floor(x / h).astype(np.int64), axis=0)))
            vs = np.array(vs, np.float64)
            o['vol_change'] = float(np.max(np.abs(vs / vs[0] - 1.0)))
            o['vol_change_last'] = float(abs(vs[-1] / vs[0] - 1.0))
        return o
    PH = {}
    for (name, m, n) in items():
        runs = []
        for rep in ('a',) if name == 'pg_N1024' else REPS:
            r = load_extra(m, n, rep)
            if r is not None and r.get('status') == '끝':
                runs.append(phys(r))
        if runs:
            PH[name] = {k: float(np.mean([q[k] for q in runs if k in q])) for k in sorted({k for q in runs for k in q})} | {'runs': len(runs)}
            clean = [q for q in runs if q.get('out_max', 0.0) == 0.0 and 'jdev_max' in q]
            if clean and len(clean) < len(runs) and ('jdev_max' in PH[name]):
                PH[name]['jdev_max_all'] = PH[name]['jdev_max']
                PH[name]['jdev_max'] = float(np.mean([q['jdev_max'] for q in clean]))
            PH[name]['jdev_runs'] = len(clean) if clean else len(runs)
        print('[물리]', name, PH.get(name), flush=True)
    res.setdefault('phys', {}).update(PH)
    json.dump(res, open(OUT, 'w'), indent=1)
    if os.environ.get('SX_PHYS_ONLY') or a.no_visual:
        pass
    else:
        import imageio.v2 as imageio
        import torch
        import pyiqa
        dev = 'cuda'
        IQ = {k: pyiqa.create_metric(k, device=dev) for k in ('niqe', 'brisque', 'clipiqa', 'musiq')}
        (RAFT, TF) = load_raft(dev)

        def img(p):
            return torch.as_tensor(imageio.imread(p)[..., :3].astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
        VIS = {}
        for (name, m, n) in items():
            P = sorted(glob.glob(f'{V}/png_{name}/*.png'))
            if len(P) < 3:
                print('[시각] PNG 없음', name, flush=True)
                continue
            q = {k: [] for k in IQ}
            (we, oc) = ([], [])
            with torch.no_grad():
                prev = img(P[0])
                for p in P[1:]:
                    cur = img(p)
                    for (k, f) in IQ.items():
                        q[k].append(float(f(cur.clamp(0, 1))))
                    (w_, o_) = warp_error(prev, cur, RAFT, TF)
                    we.append(w_)
                    oc.append(o_)
                    prev = cur
            VIS[name] = {k: float(np.mean(v)) for (k, v) in q.items()} | dict(warp_err=float(np.mean(we)), warp_occ_frac=float(np.mean(oc)), frames=len(P) - 1, pyiqa_version=getattr(pyiqa, '__version__', '?'), def_='metrics_def.md (final_fix)')
            print('[시각]', name, VIS[name], flush=True)
        res.setdefault('visual', {}).update(VIS)
        json.dump(res, open(OUT, 'w'), indent=1)
    print('[stab_extra] ->', OUT)
if CMD == 'is-stable':
    import json
    import os
    import sys
    import numpy as np
    (d, m, n) = (sys.argv[1], sys.argv[2], int(sys.argv[3]))
    r = load(d, m, n)
    if r is None or r['status'] == '진행':
        print('-')
        sys.exit(0)
    if r['status'] != '끝':
        print(0)
        sys.exit(0)
    root = sys.argv[4] if len(sys.argv) > 4 else os.path.dirname(d.rstrip('/'))
    cfg = json.load(open(f'{root}/config.json'))
    _fl = [b for b in cfg['boundary_conditions'] if b['type'] == 'surface_collider']
    X = r['X']
    if not _fl:
        ref = load(f'{root}/res', 'pg', 1024)
        if ref is None or ref['status'] != '끝':
            print('-')
            sys.exit(0)
        _E = lambda rr: rr['rows']['ke'] + rr['rows']['pe']
        e_ref = float(np.nanmax(np.abs(_E(ref))))
        E = _E(r)
        ok = bool(np.isfinite(E).all()) and float(np.max((E - e_ref) / e_ref)) <= 1.0 and (len(E) == len(_E(ref)))
        print(1 if ok else 0)
        sys.exit(0)
    zf = float(_fl[0]['point'][2])
    L = float(np.linalg.norm(X[0].max(0) - X[0].min(0)))
    if r['rows'] is not None:
        r['rows'] = {k: v for (k, v) in r['rows'].items() if k != 'ee'}
    mt = metrics(r, None, L, zf, float(cfg['g'][2]), float(cfg['frame_dt']), 1.0, float(cfg['grid_lim']) / int(cfg['n_grid']))
    ok = np.isfinite(mt.get('e_drift_max', 0.0)) and mt.get('e_drift_max', 0.0) <= 1.0 and ((mt.get('pen_depth_max') or 0.0) <= 0.5)
    print(1 if ok else 0)
if CMD == 'metrics':
    metrics_main()
if CMD == 'stability-vis':
    import argparse
    import glob
    import json
    import imageio.v2 as imageio
    import numpy as np
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import lpips
    from skimage.metrics import structural_similarity as ssim
    LP = lpips.LPIPS(net='alex').cuda().eval()
    R = sorted(glob.glob(f'{a.ref}/*.png'))
    res = {}
    for spec in a.runs:
        (name, d) = spec.split('=', 1)
        F = sorted(glob.glob(f'{d}/*.png'))
        T = min(len(F), len(R))
        (ps, ss, ls) = ([], [], [])
        for t in range(1, T):
            x = imageio.imread(F[t])[..., :3].astype(np.float32) / 255
            y = imageio.imread(R[t])[..., :3].astype(np.float32) / 255
            mse = float(((x - y) ** 2).mean())
            ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
            ss.append(float(ssim(x, y, channel_axis=2, data_range=1.0)))
            with torch.no_grad():
                tx = torch.as_tensor(x).permute(2, 0, 1)[None].cuda() * 2 - 1
                ty = torch.as_tensor(y).permute(2, 0, 1)[None].cuda() * 2 - 1
                ls.append(float(LP(tx, ty)))
        res[name] = dict(frames=T - 1, psnr=float(np.mean(ps)), ssim=float(np.mean(ss)), lpips=float(np.mean(ls)), psnr_last=ps[-1], lpips_last=ls[-1])
        print(name, res[name], flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)
if CMD == 'phys':
    import argparse
    import glob
    import json
    import os
    import sys
    import numpy as np
    import torch
    import evaluation as ss
    import gsrender as DM
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--N', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--pred', default='', help='덤프 pred.pt {pred [T,N,3] = 프레임 1..T, F [T,N,3,3]} (프레임 0 = 자료 x[0], F = I)')
    ap.add_argument('--gt_h5', default='', help='정답 렌더용 궤적 폴더 (sim_*.h5 의 x, F = F_sim R; meta.json 의 R) -- 위치는 자료 x, F 는 h5')
    ap.add_argument('--frames', type=int, default=0, help='0 = 덤프 길이 전부')
    ap.add_argument('--floor', default='vel')
    ap.add_argument('--no_F', action='store_true')
    ap.add_argument('--note', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d = torch.load(a.data, map_location='cpu', weights_only=False)
    cfg = d['cfg']
    h = float(cfg['frame_dt'])
    g = np.array(cfg.get('g', [0.0, 0.0, -9.8]), np.float64)
    dxg = float(cfg.get('grid_lim', 2.0)) / int(cfg.get('n_grid', 100))
    m = DM.traj_mass(d, 'cpu').double().numpy()
    Np = m.shape[0]
    vol = m / float(cfg['density'])
    XD = d['x']
    x0 = XD[0].double().numpy()
    HAS_FLOOR = a.floor != 'none'
    zf = None
    if HAS_FLOOR:
        fl = [b for b in cfg.get('boundary_conditions', []) if b.get('type') == 'surface_collider']
        zf = float(fl[0]['point'][2]) if fl else float(cfg.get('floor_z', 0.0))
    zref = zf if HAS_FLOOR else float(x0[:, 2].min())
    if 'v' in d and d['v'] is not None:
        v0 = d['v'][0].double().numpy()
    else:
        v0 = np.tile(np.array(cfg.get('init_velocity', [0.0, 0.0, 0.0]), np.float64), (Np, 1))
    note = [a.note] if a.note else []
    if a.pred:
        P = torch.load(a.pred, map_location='cpu', weights_only=False)
        (PX, PF) = (P['pred'], P.get('F'))
        T = PX.shape[0] if not a.frames else min(a.frames, PX.shape[0])
        if P.get('part'):
            note.append(f"덤프 일부 (발산 프레임 {P.get('diverged')})")

        def X(t):
            return x0 if t == 0 else PX[t - 1].double().numpy()

        def Fm(t):
            if a.no_F or PF is None:
                return None
            return np.tile(np.eye(3), (Np, 1, 1)) if t == 0 else PF[t - 1].double().numpy().reshape(-1, 3, 3)
    else:
        import h5py
        R = np.array(json.load(open(f'{a.gt_h5}/meta.json'))['R'], np.float64)
        H5 = sorted(glob.glob(f'{a.gt_h5}/sim_*.h5'))
        T = min(len(H5) - 1, XD.shape[0] - 1)
        if a.frames:
            T = min(T, a.frames)

        def X(t):
            return XD[t].double().numpy()

        def Fm(t):
            if a.no_F:
                return None
            with h5py.File(H5[t]) as f_:
                Fr = np.array(f_['F'], np.float64)
            Fr = Fr.T if Fr.shape[0] == 9 and Fr.shape[1] != 9 else Fr
            return Fr.reshape(-1, 3, 3) @ R.T
    SUB = ss.subset(Np)
    (rows, XS, dV) = ([], [], [])
    (xp, xc) = (None, X(0))
    for t in range(T + 1):
        xn = X(t + 1) if t < T else None
        if t == 0:
            v = v0
        elif xn is not None:
            v = (xn - xp) / (2 * h)
        else:
            v = (xc - xp) / h
        F = Fm(t)
        st = ss.frame_stats(xc, v, F, m, g, zref)
        rows.append(st)
        XS.append(xc[SUB].astype(np.float32))
        if F is not None:
            J = np.linalg.det(F)
            dV.append(float((vol * (J - 1)).sum() / vol.sum()))
        (xp, xc) = (xc, xn)
    keys = sorted({k for r in rows for k in r})
    RW = {k: np.array([r.get(k, np.nan) for r in rows], np.float64) for k in keys}
    XS = np.stack(XS)
    Lc = float(np.linalg.norm(XS[0].max(0) - XS[0].min(0)))
    M = float(m.sum())
    rr = dict(rows={k: v for (k, v) in RW.items() if k != 'ee'}, X=XS)
    mt = ss.metrics(rr, None, Lc, zf if HAS_FLOOR else zref, g[2], h, M, dxg)
    E = RW['ke'] + RW['pe']
    p0 = np.array([RW['px'][0], RW['py'][0], RW['pz'][0]])
    out = dict(method=a.method, scene=a.scene, N=a.N, frames=int(T), n_particles=int(Np), L=Lc, floor_z=zf, pe_ref_z=zref, detneg_max=None, detmin=None, jdev_max=None, dV_last=None, dV_absmax=None, pen_ratio_max=None, pen_depth_max=None, e_drift_max=mt.get('e_drift_max'), e_ratio_last=float(E[-1] / E[0]) if E[0] != 0 else None, E0=float(E[0]), freefall_mom_err=None, freefall_frames=None, jitter_d2=mt.get('jitter_d2'), jitter_hf=mt.get('jitter_hf'), nan_max=float(np.nanmax(RW['nan'])))
    if not (a.no_F or (a.pred and PF is None)):
        out.update(detneg_max=float(np.nanmax(RW['detneg'])), detmin=float(np.nanmin(RW['detmin'])), jdev_max=float(np.nanmax(RW['jdev'])), dV_last=dV[-1], dV_absmax=float(np.max(np.abs(dV))))
    else:
        note.append('F 없음 (스프링-질량: 변형 기울기가 표현에 없음) -> 뒤집힘·부피 해당 없음')
    if HAS_FLOOR:
        out.update(pen_ratio_max=mt.get('pen_ratio_max'), pen_depth_max=mt.get('pen_depth_max'), freefall_frames=mt.get('freefall_frames'))
        if np.linalg.norm(p0) <= 1e-09 * M:
            out['freefall_mom_err'] = mt.get('freefall_mom_err')
        else:
            Tc = mt.get('freefall_frames') or 0
            gv = np.array([0.0, 0.0, g[2]])
            errs = [np.linalg.norm(np.array([RW['px'][t], RW['py'][t], RW['pz'][t]]) - (p0 + M * gv * t * h)) / (M * abs(g[2]) * t * h) for t in range(1, min(Tc, T + 1))]
            out['freefall_mom_err'] = float(max(errs)) if errs else None
            note.append('처음 속도 0 아님 -> 자유낙하 = |p_t - (p_0 + M g t h)| / (M |g| t h)')
    else:
        note.append('바닥 없음 (--floor none) -> 관통·자유낙하 해당 없음; 위치 에너지 기준 = 0 프레임 최저 입자 z')
    out['note'] = '; '.join(note)
    out['dV_series'] = dV
    out['E_series'] = E.tolist()
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for (k, v) in out.items() if not k.endswith('_series')}, ensure_ascii=False), flush=True)
if CMD == 'pareto-vis':
    import argparse
    import glob
    import json
    import imageio.v2 as imageio
    import numpy as np
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--png', required=True)
    ap.add_argument('--ref', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--iqa', default='niqe,brisque,musiq,clipiqa')
    ap.add_argument('--ref_only', action='store_true', help='기존 --out 의 값은 두고 정답 렌더 대비 (PSNR·SSIM·LPIPS) 만 다시 재어 합침')
    a = ap.parse_args()
    dev = 'cuda'
    FR = sorted(glob.glob(f'{a.png}/*.png'))
    RF = sorted(glob.glob(f'{a.ref}/*.png')) if a.ref else []

    def img(p):
        return imageio.imread(p)[..., :3].astype(np.float32) / 255

    def tens(x):
        return torch.as_tensor(x).permute(2, 0, 1)[None].to(dev)
    assert not a.ref or RF, f'정답 렌더 PNG 없음: {a.ref}'
    res = dict(frames=len(FR) - 1)
    if a.ref_only:
        res = json.load(open(a.out))
    if not a.ref_only:
        import pyiqa
        res['pyiqa_version'] = getattr(pyiqa, '__version__', '?')
    for name in [] if a.ref_only else [q for q in a.iqa.split(',') if q]:
        try:
            M = pyiqa.create_metric(name, device=dev)
            vs = []
            with torch.no_grad():
                for p in FR[1:]:
                    vs.append(float(M(tens(img(p)).clamp(0, 1))))
            res[name] = float(np.mean(vs))
            res[name + '_lower_better'] = bool(M.lower_better)
            del M
            torch.cuda.empty_cache()
        except Exception as ex:
            res[name] = None
            res[name + '_err'] = repr(ex)[:300]
        print(name, res.get(name), flush=True)
    if a.ref_only:
        FR_W = []
    else:
        FR_W = FR
    from evaluation import load_raft, warp_error
    (RAFT, TF) = load_raft(dev)
    (ws, occ) = ([], [])
    prev = tens(img(FR[0])) if FR_W else None
    for p in FR_W[1:]:
        cur = tens(img(p))
        (w_, o_) = warp_error(prev, cur, RAFT, TF)
        ws.append(w_)
        occ.append(o_)
        prev = cur
    if FR_W:
        res['warp_err'] = float(np.mean(ws))
        res['warp_occ_frac'] = float(np.mean(occ))
        res['raft_weights'] = 'torchvision raft_large C_T_SKHT_V2'
        print('warp_err', res['warp_err'], flush=True)
    del RAFT
    torch.cuda.empty_cache()
    if RF:
        import lpips
        from skimage.metrics import structural_similarity as ssim
        LP = lpips.LPIPS(net='alex').to(dev).eval()
        T = min(len(FR), len(RF))
        (ps, sv, ls) = ([], [], [])
        for t in range(1, T):
            (x, y) = (img(FR[t]), img(RF[t]))
            ps.append(10 * np.log10(1.0 / max(float(((x - y) ** 2).mean()), 1e-12)))
            sv.append(float(ssim(x, y, channel_axis=2, data_range=1.0)))
            with torch.no_grad():
                ls.append(float(LP(tens(x) * 2 - 1, tens(y) * 2 - 1)))
        res.update(psnr=float(np.mean(ps)), ssim=float(np.mean(sv)), lpips=float(np.mean(ls)), ref_frames=T - 1)
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False), flush=True)
if CMD == 'render':
    import torch
    import argparse
    import glob
    import json
    import math
    import os
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--shape', default='lego')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--iter', type=int, default=-1, help='3DGS 체크포인트 반복. -1 이면 PG load_checkpoint 처럼 가장 큰 반복')
    ap.add_argument('--colors', default='', help='가우시안 색 덮어쓰기 .npy [T,>=NG,3] (h5 행 순서, 앞 NG 개를 쓴다) (0~1, 프레임 순서 = sim_*.h5 정렬 순). 주면 SH 대신 colors_precomp 로 래스터화한다 (불투명도·공분산은 그대로)')
    ap.add_argument('--iso', type=float, default=0.0, help='공분산을 등방형 (iso × 가우시안 축 길이 중앙값)² I 로 (F 무시). 0 이면 FΣFᵀ')
    ap.add_argument('--frame_label', type=int, default=None, help='주면 프레임 번호 (이 값 + 영상 프레임 순번) 를 크게 빨간색으로 찍는다')
    ap.add_argument('--cam_back', type=float, default=1.6, help='카메라를 물체 중심에서 이 배로 뒤로 (기본 1.6, 기존 그대로)')
    ap.add_argument('--elev', type=float, default=None, help='elastic_new: 위에서 내려다보는 카메라 고도각 (도; 같은 중심, 거리 = cam_back 거리 x --back2, 방위 = cameras.json 첫 장). 없으면 기존 그대로')
    ap.add_argument('--up', default='0,0,1', help='--elev 의 위 방향')
    ap.add_argument('--back2', type=float, default=1.0)
    ap.add_argument('--caption', default='', help='--frame_label 과 함께 왼쪽 위에 찍을 설명 (한국어 가능)')
    ap.add_argument('--frames', type=int, default=0, help='앞 이 상태 번호까지만 (0 = 전부; sim_*.h5 정렬 순 0..frames)')
    ap.add_argument('--mark', default='', help='사용자 10-30 발산 셀 입자 표시: 입자 번호 .npy -- 그 입자를 빨강으로 (가우시안 < NG 는 색만, 채움 입자 ≥ NG 는 등방 가우시안 (축 = 가우시안 축 길이 중앙값 x --mark_size, 공분산 F s² Fᵀ) 을 더해 그린다). 나머지는 SH 그대로 (래스터라이저와 같은 식으로 colors_precomp)')
    ap.add_argument('--mark_fade', type=float, default=1.0, help='--mark: 표시 안 한 가우시안 불투명도 배율 (투시 칸, 1 = 그대로)')
    ap.add_argument('--mark_size', type=float, default=1.0)
    ap.add_argument('--aniso_json', default='', help='절 CM194: 주면 프레임마다 가우시안 공분산 FΣFᵀ 의 σmax/σmin (고윳값 비의 제곱근) 최대·상위 1%% 값 (그리고 정지 Σ 대비 배율) -> json (렌더 불변)')
    ap.add_argument('--aniso_only', type=int, default=0, help='절 CM194: 1 이면 이방성 json 만 (렌더·영상 쓰기 건너뜀)')
    ap.add_argument('--png', default='', help='주면 프레임마다 무손실 PNG 도 이 폴더에 (%04d.png; 시각 품질 PSNR·SSIM·LPIPS 용, doc/pareto.md 절 S)')
    ap.add_argument('--mark_px', default='', help='--mark: 프레임마다 표시 입자 화면 위치 (중앙·범위 픽셀) -> json (확대 칸 중심)')
    ap.add_argument('--cells', default='', help='절 CM14.3: 칸별 장벽 에너지 bener.pt (FusedGS --bar_ecell) -- 렌더 위에 그 칸 (프레임 끝 꼭짓점 8 개) 을 같은 카메라로 반투명 상자 (보이는 면) 로 겹침, 색 = log10 에너지 (--cell_key) 를 --erange 고정 범위로 (칸별·판별 정규화 없음). 영상 프레임 fi (h5 상태 fi) 의 칸 = bener frames[fi − 1] (fi = 0 은 처음 상태라 없음)')
    ap.add_argument('--erange', default='-6,0', help='--cells: 색 범위 log10 에너지 "lo,hi" (범위 밖은 끝 색)')
    ap.add_argument('--cell_key', default='em', choices=['em', 'el'], help='--cells: em = 프레임 안 걸음 끝 최대, el = 프레임 끝')
    ap.add_argument('--cell_alpha', type=float, default=0.4)
    a = ap.parse_args()
    import h5py
    import imageio.v2 as imageio
    import numpy as np
    import torch
    torch.set_grad_enabled(False)
    dev = 'cuda'
    cfg = json.load(open(a.config))
    meta = json.load(open(f'{a.sim}/meta.json'))
    sys.path.append(f'{W}/i-physgaussian')
    sys.path.append(f'{W}/i-physgaussian/gaussian-splatting')
    _cwd = os.getcwd()
    os.chdir(f'{W}/i-physgaussian')
    from scene.gaussian_model import GaussianModel
    from utils.transformation_utils import transform2origin, shift2center111
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    os.chdir(_cwd)
    mp = f'{W}/pgmodel/{MODEL[a.shape]}'
    gs = GaussianModel(3)
    _it = a.iter if a.iter >= 0 else max((int(q.split('_')[-1]) for q in os.listdir(f'{mp}/point_cloud') if q.startswith('iteration_')))
    print(f'[3DGS] {mp} iteration_{_it}', flush=True)
    gs.load_ply(f'{mp}/point_cloud/iteration_{_it}/point_cloud.ply')
    KIDX = torch.nonzero(gs.get_opacity[:, 0] > float(cfg.get('opacity_threshold', 0.02))).squeeze(1)
    (_RT, ROT) = cfg_rotation(cfg, dev)
    if cfg.get('sim_area') is not None and os.environ.get('LB_NO_SIMAREA', '0') != '1':
        _XR = gs.get_xyz[KIDX] @ _RT.T if ROT else gs.get_xyz[KIDX]
        _b = cfg['sim_area']
        _m = torch.ones(_XR.shape[0], dtype=torch.bool, device=_XR.device)
        for _i in range(3):
            _m = _m & (_XR[:, _i] > _b[2 * _i]) & (_XR[:, _i] < _b[2 * _i + 1])
        KIDX = KIDX[_m]
    (TP, SO, MEAN) = transform2origin(gs.get_xyz[KIDX] @ _RT.T if ROT else gs.get_xyz[KIDX], float(cfg.get('scale', 1.0)))
    TP = shift2center111(TP)
    NG = KIDX.numel()
    with h5py.File(meta['h5']) as h:
        x0 = np.array(h['x'])
        x0 = x0.T if x0.shape[0] == 3 else x0
    err = float((torch.as_tensor(x0[:NG], device=dev).float() - TP).norm(dim=1).max())
    print(f'[대응] 가우시안 {NG}, 회전 전 입자 앞 {NG} 개와 위치 차이 최대 {err:.2e}', flush=True)
    assert err < 0.0001, '채움 캐시 앞부분이 가우시안과 같은 번호가 아니다'
    c6 = gs.get_covariance()[KIDX]
    C0 = cov6_to_mat(c6)
    if ROT:
        C0 = _RT @ C0 @ _RT.T
    (SHS, OPA) = (gs.get_features[KIDX], gs.get_opacity[KIDX])
    cam = json.load(open(f'{mp}/cameras.json'))[0]
    (Rw, pos) = (np.array(cam['rotation']), np.array(cam['position']))
    _ctr = (MEAN @ _RT if ROT else MEAN).cpu().numpy().astype(np.float64)
    pos = _ctr + a.cam_back * (pos - _ctr)
    if a.elev is not None:
        _up = np.array([float(q) for q in a.up.split(',')])
        _up /= np.linalg.norm(_up)
        _el = math.radians(a.elev)
        _rel = pos - _ctr
        _d = np.linalg.norm(_rel) * a.back2
        _h = _rel - _up * (_rel @ _up)
        _h /= np.linalg.norm(_h)
        pos = _ctr + _d * (math.cos(_el) * _h + math.sin(_el) * _up)
        _f = (_ctr - pos) / np.linalg.norm(_ctr - pos)
        _r = np.cross(_f, _up)
        _r /= np.linalg.norm(_r)
        _dn = np.cross(_f, _r)
        Rw = np.stack([_r, _dn, _f], 1)
        print('[cam] elev', a.elev, 'up', _up, 'pos', pos, flush=True)
    W2C = np.linalg.inv(np.block([[Rw, pos[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
    fx_ = 2 * math.atan(cam['width'] / (2 * cam['fx']))
    fy_ = 2 * math.atan(cam['height'] / (2 * cam['fy']))
    wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
    pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fx_, fovY=fy_).transpose(0, 1).to(dev).float()
    RAST = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(image_height=int(cam['height']), image_width=int(cam['width']), tanfovx=math.tan(fx_ * 0.5), tanfovy=math.tan(fy_ * 0.5), bg=torch.ones(3, device=dev), scale_modifier=1.0, viewmatrix=wv, projmatrix=(wv[None] @ pj[None])[0], sh_degree=3, campos=wv.inverse()[3, :3], prefiltered=False, debug=False))
    _FONT = None

    def _label(fr, n):
        """프레임 번호를 크게 빨간색으로 (그리고 --caption)."""
        global _FONT
        from PIL import Image, ImageDraw, ImageFont
        if _FONT is None:
            from matplotlib import font_manager as fm
            _cand = [f for f in fm.findSystemFonts() if 'Noto' in f and ('CJK' in f or 'KR' in f)] or [f for f in fm.findSystemFonts() if 'DejaVuSans-Bold' in f] or [fm.findfont('DejaVu Sans')]
            _H = fr.shape[0]
            _FONT = (ImageFont.truetype(_cand[0], int(_H * 0.09)), ImageFont.truetype(_cand[0], int(_H * 0.035)))
        im = Image.fromarray(fr)
        dr = ImageDraw.Draw(im)
        dr.text((int(fr.shape[1] * 0.03), int(fr.shape[0] * 0.02)), f'{n}', fill=(230, 0, 0), font=_FONT[0])
        if a.caption:
            dr.text((int(fr.shape[1] * 0.03), int(fr.shape[0] * 0.13)), a.caption, fill=(230, 0, 0), font=_FONT[1])
        return np.asarray(im)
    WR = imageio.get_writer(a.out, fps=a.fps, codec='libx264', quality=8)
    files = sorted(glob.glob(f'{a.sim}/sim_*.h5'))
    COL = None
    if a.colors:
        COL = np.load(a.colors, mmap_mode='r')
        assert COL.shape[0] == len(files) and COL.shape[1] >= NG and (COL.shape[2] == 3), (COL.shape, len(files), NG)
        print(f'[색 덮어쓰기] {a.colors} {tuple(COL.shape)}', flush=True)
    if a.frames:
        files = files[:a.frames + 1]
    MK = None
    if a.mark:
        (MG, MF) = load_mark(a.mark, NG, dev)
        MK = torch.zeros(NG, dtype=torch.bool, device=dev)
        MK[MG] = True
        S_EX = float(gs.get_scaling[KIDX].median()) * a.mark_size
        CAMPOS = wv.inverse()[3, :3]
        FPROJ = (wv[None] @ pj[None])[0]
        print(f'[표시] {a.mark}: 가우시안 {MG.numel()} + 채움 입자 {MF.numel()} (등방 축 {S_EX:.4g}), 나머지 불투명도 x {a.mark_fade}', flush=True)
        PXL = []
    CEL = None
    if a.cells:
        import matplotlib
        CEL = torch.load(a.cells, weights_only=False)['frames']
        (_ELO, _EHI) = [float(q) for q in a.erange.split(',')]
        _CMAP = matplotlib.colormaps['inferno']
        _CPROJ = (wv[None] @ pj[None])[0]
        print(f'[칸 겹치기] {a.cells} {len(CEL)} 프레임, 색 log10 범위 {_ELO}~{_EHI} ({a.cell_key})', flush=True)

        def ecol(e):
            """에너지 -> RGB (0~255), log10 고정 범위, inferno 0.05~0.9 (흰 배경에서 보이게 밝은 끝 자름)."""
            t = (np.log10(np.maximum(e, 1e-300)) - _ELO) / (_EHI - _ELO)
            return (np.array(_CMAP(0.05 + 0.85 * np.clip(t, 0, 1)))[..., :3] * 255).astype(np.float32)

        def _cells(fr, cl):
            e = cl[a.cell_key].numpy().astype(np.float64)
            keep = e > 0
            if not keep.any():
                return fr
            cor = cl['cor'][torch.from_numpy(keep)].to(dev).float().reshape(-1, 3)
            Pc = (cor - 1.0) / SO + MEAN
            if ROT:
                Pc = Pc @ _RT
            ph = torch.cat([Pc, torch.ones_like(Pc[:, :1])], 1)
            dep = (ph @ wv)[:, 2].reshape(-1, 8).cpu().numpy()
            px = project_px(Pc, _CPROJ, fr.shape[1], fr.shape[0]).reshape(-1, 8, 2).cpu().numpy()
            return draw_cells(fr, px, dep, ecol(e[keep]), a.cell_alpha)
    ANI = []
    for (fi, p) in enumerate(files):
        with h5py.File(p) as h:
            x = torch.as_tensor(np.array(h['x'][:NG]), device=dev).float()
            F = torch.as_tensor(np.array(h['F'][:NG]), device=dev).float().reshape(-1, 3, 3)
            if MK is not None and MF.numel():
                _xa = np.array(h['x'])
                _xa = _xa.T if _xa.shape[0] == 3 else _xa
                _Fa = np.array(h['F'])
                _Fa = _Fa.T if _Fa.shape[0] == 9 else _Fa
                _sel = MF.cpu().numpy()
                xe = torch.as_tensor(_xa[_sel], device=dev).float()
                Fe = torch.as_tensor(_Fa.reshape(-1, 9)[_sel], device=dev).float().reshape(-1, 3, 3)
        if a.iso > 0:
            cov = torch.eye(3, device=dev).expand(NG, 3, 3) * (a.iso * float(gs.get_scaling[KIDX].median())) ** 2
        else:
            cov = F @ C0 @ F.transpose(1, 2)
            if ROT:
                cov = _RT.T @ cov @ _RT
        cc = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
        if a.aniso_json:
            if '_L0' not in globals():
                from utils.general_utils import build_rotation as _bro
                _L0 = _bro(gs._rotation[KIDX]).double() * gs.get_scaling[KIDX].double()[:, None, :]
                if ROT:
                    _L0 = _RT.double() @ _L0
                _s0 = torch.linalg.svdvals(_L0)
                _r0 = _s0[:, 0] / _s0[:, 2].clamp_min(1e-300)
            _ok = torch.isfinite(F.reshape(NG, 9)).all(1)
            _sv = torch.linalg.svdvals(F[_ok].double() @ _L0[_ok])
            _r = _sv[:, 0] / _sv[:, 2].clamp_min(1e-300)
            _rr = _r / _r0[_ok]
            ANI.append(dict(fi=fi, n=int(NG), nbad=int((~_ok).sum()), max=float(_r.max()) if _r.numel() else None, p99=float(torch.quantile(_r.float(), 0.99)) if _r.numel() else None, rel_max=float(_rr.max()) if _rr.numel() else None, rel_p99=float(torch.quantile(_rr.float(), 0.99)) if _rr.numel() else None))
            if a.aniso_only:
                continue
        Pm = (x - 1.0) / SO + MEAN
        if ROT:
            Pm = Pm @ _RT
        _cp = torch.as_tensor(np.asarray(COL[fi][:NG], dtype=np.float32), device=dev) if COL is not None else None
        _op = OPA
        if MK is not None:
            if _cp is None:
                _cp = sh_rgb(SHS, Pm, CAMPOS)
            _cp = _cp.clone()
            _cp[MK] = torch.tensor(RED, device=dev)
            _op = OPA.clone()
            if a.mark_fade != 1.0:
                _op[~MK] = _op[~MK] * a.mark_fade
            _mp = [Pm[MK]]
            if MF.numel():
                _ok = torch.isfinite(xe).all(1) & torch.isfinite(Fe).all(2).all(1)
                Pe = (xe[_ok] - 1.0) / SO + MEAN
                (ce, cove) = extra_cov(Fe[_ok], S_EX)
                if ROT:
                    Pe = Pe @ _RT
                    cove = _RT.T @ cove @ _RT
                    ce = torch.stack([cove[:, 0, 0], cove[:, 0, 1], cove[:, 0, 2], cove[:, 1, 1], cove[:, 1, 2], cove[:, 2, 2]], 1)
                Pm = torch.cat([Pm, Pe])
                cc = torch.cat([cc, ce])
                _cp = torch.cat([_cp, torch.tensor(RED, device=dev).expand(Pe.shape[0], 3)])
                _op = torch.cat([_op, torch.ones(Pe.shape[0], 1, device=dev)])
                _mp.append(Pe)
            _mp = torch.cat(_mp)
            _mp = _mp[torch.isfinite(_mp).all(1)]
            if _mp.shape[0]:
                _px = project_px(_mp, FPROJ, int(cam['width']), int(cam['height']))
                PXL.append(dict(fi=fi, med=_px.median(0).values.tolist(), lo=_px.min(0).values.tolist(), hi=_px.max(0).values.tolist(), n=int(_mp.shape[0])))
            else:
                PXL.append(dict(fi=fi, med=None))
        img = RAST(means3D=Pm, means2D=torch.zeros_like(Pm), shs=SHS if _cp is None else None, colors_precomp=_cp, opacities=_op, scales=None, rotations=None, cov3D_precomp=cc)[0]
        _fr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        if CEL is not None and fi >= 1 and (fi - 1 < len(CEL)):
            _fr = _cells(_fr, CEL[fi - 1])
        if a.frame_label is not None:
            _fr = _label(_fr, a.frame_label + fi)
        WR.append_data(_fr)
        if a.png:
            os.makedirs(a.png, exist_ok=True)
            imageio.imwrite(os.path.join(a.png, f'{fi:04d}.png'), _fr)
    WR.close()
    if a.aniso_json:
        json.dump(dict(frames=ANI, max=max((q['max'] for q in ANI if q['max'] is not None), default=None), p99=max((q['p99'] for q in ANI if q['p99'] is not None), default=None), rel_max=max((q['rel_max'] for q in ANI if q['rel_max'] is not None), default=None), rel_p99=max((q['rel_p99'] for q in ANI if q['rel_p99'] is not None), default=None), rest_max=ANI[0]['max'] if ANI else None, rest_p99=ANI[0]['p99'] if ANI else None, nbad=sum((q['nbad'] for q in ANI))), open(a.aniso_json, 'w'), indent=1)
    if MK is not None and a.mark_px:
        json.dump(dict(W=int(cam['width']), H=int(cam['height']), frames=PXL, n_gauss=int(MG.numel()), n_fill=int(MF.numel())), open(a.mark_px, 'w'))
    print(f'[영상] {a.out} ({len(files)} 프레임)', flush=True)
if CMD == 'multiview':
    import torch
    import argparse
    import glob
    import json
    import math
    import os
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--shape', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--ncam', type=int, default=10)
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--cam_back', type=float, default=1.6)
    ap.add_argument('--iter', type=int, default=-1)
    a = ap.parse_args()
    import h5py
    import imageio.v2 as imageio
    import numpy as np
    import torch
    torch.set_grad_enabled(False)
    dev = 'cuda'
    cfg = json.load(open(a.config))
    meta = json.load(open(f'{a.sim}/meta.json'))
    sys.path.append(f'{W}/i-physgaussian')
    sys.path.append(f'{W}/i-physgaussian/gaussian-splatting')
    _cwd = os.getcwd()
    os.chdir(f'{W}/i-physgaussian')
    from scene.gaussian_model import GaussianModel
    from utils.transformation_utils import transform2origin, shift2center111
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    from utils.sh_utils import eval_sh
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    os.chdir(_cwd)
    mp = f'{W}/pgmodel/{MODEL[a.shape]}'
    gs = GaussianModel(3)
    _it = a.iter if a.iter >= 0 else max((int(q.split('_')[-1]) for q in os.listdir(f'{mp}/point_cloud') if q.startswith('iteration_')))
    gs.load_ply(f'{mp}/point_cloud/iteration_{_it}/point_cloud.ply')
    KIDX = torch.nonzero(gs.get_opacity[:, 0] > float(cfg.get('opacity_threshold', 0.02))).squeeze(1)
    (_RT, ROT) = cfg_rotation(cfg, dev)
    _XR = gs.get_xyz[KIDX] @ _RT.T if ROT else gs.get_xyz[KIDX]
    if a.shape in ('vasedeck', 'plane', 'pillow2sofa') and cfg.get('sim_area') is not None:
        _b = cfg['sim_area']
        _m = torch.ones(_XR.shape[0], dtype=torch.bool, device=_XR.device)
        for _i in range(3):
            _m = _m & (_XR[:, _i] > _b[2 * _i]) & (_XR[:, _i] < _b[2 * _i + 1])
        KIDX = KIDX[_m]
        _XR = _XR[_m]
    (TP, SO, MEAN) = transform2origin(_XR, float(cfg.get('scale', 1.0)))
    TP = shift2center111(TP)
    NG = KIDX.numel()
    with h5py.File(meta['h5']) as h:
        x0 = np.array(h['x'])
        x0 = x0.T if x0.shape[0] == 3 else x0
    err = float((torch.as_tensor(x0[:NG], device=dev).float() - TP).norm(dim=1).max())
    assert err < 0.0001, f'채움 캐시 앞부분이 가우시안과 같은 번호가 아님 ({err})'
    c6 = gs.get_covariance()[KIDX]
    C0 = cov6_to_mat(c6)
    if ROT:
        C0 = _RT @ C0 @ _RT.T
    C0 = C0 * float(SO) ** 2
    (SHS, OPA) = (gs.get_features[KIDX], gs.get_opacity[KIDX])
    files = sorted(glob.glob(f'{a.sim}/sim_*.h5'))
    if a.frames:
        files = files[:a.frames + 1]
    X = []
    for p in files:
        with h5py.File(p) as h:
            X.append(np.array(h['x'][:NG]))
    X = np.stack(X)
    (lo, hi) = (X.reshape(-1, 3).min(0), X.reshape(-1, 3).max(0))
    ctr = 0.5 * (lo + hi)
    cam = json.load(open(f'{mp}/cameras.json'))[0]
    pos_w = _RT.cpu().numpy().astype(np.float64) @ np.array(cam['position'])
    mean_w = MEAN.cpu().numpy().astype(np.float64)
    dist = a.cam_back * float(np.linalg.norm(pos_w - mean_w)) * float(SO)
    el0 = math.degrees(math.asin(max(-1.0, min(1.0, (pos_w - mean_w)[2] / np.linalg.norm(pos_w - mean_w)))))
    el0 = min(max(el0, 10.0), 60.0)
    fov = 2 * math.atan(cam['width'] / (2 * cam['fx']))
    f_px = a.res / (2 * math.tan(fov / 2))
    pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov).transpose(0, 1).to(dev).float()
    (CAMS, RAST, CPOS) = ([], [], [])
    for i in range(a.ncam):
        (az, el) = (math.radians(36.0 * i * 10 / a.ncam), math.radians(el0 + (20.0 if i % 2 else 0.0)))
        p = ctr + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        fw = (ctr - p) / np.linalg.norm(ctr - p)
        rt = np.cross(fw, [0.0, 0.0, 1.0])
        rt /= np.linalg.norm(rt)
        up = np.cross(rt, fw)
        c2w_b = np.eye(4)
        (c2w_b[:3, 0], c2w_b[:3, 1], c2w_b[:3, 2], c2w_b[:3, 3]) = (rt, up, -fw, p)
        c2w_c = c2w_b.copy()
        c2w_c[:3, 1:3] *= -1
        W2C = np.linalg.inv(c2w_c)
        wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
        RAST.append(GaussianRasterizer(raster_settings=GaussianRasterizationSettings(image_height=a.res, image_width=a.res, tanfovx=math.tan(fov * 0.5), tanfovy=math.tan(fov * 0.5), bg=torch.zeros(3, device=dev), scale_modifier=1.0, viewmatrix=wv, projmatrix=(wv[None] @ pj[None])[0], sh_degree=3, campos=wv.inverse()[3, :3], prefiltered=False, debug=False)))
        CPOS.append(torch.tensor(p, device=dev).float())
        CAMS.append(dict(camera=f'cam_{i}', K=[[f_px, 0.0, a.res / 2.0], [0.0, f_px, a.res / 2.0], [0.0, 0.0, 1.0]], c2w=c2w_b.tolist(), width=a.res, height=a.res))
        os.makedirs(f'{a.out}/cam_{i}', exist_ok=True)
    for (fi, p) in enumerate(files):
        with h5py.File(p) as h:
            x = torch.as_tensor(np.array(h['x'][:NG]), device=dev).float()
            F = torch.as_tensor(np.array(h['F'][:NG]), device=dev).float().reshape(-1, 3, 3)
        cov = F @ C0 @ F.transpose(1, 2)
        cc = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
        for (i, RS) in enumerate(RAST):
            cp = None
            if ROT:
                d = x - CPOS[i]
                d = d / d.norm(dim=1, keepdim=True) @ _RT
                cp = torch.clamp_min(eval_sh(3, SHS.transpose(1, 2), d) + 0.5, 0.0)
            img = RS(means3D=x, means2D=torch.zeros_like(x), shs=SHS if cp is None else None, colors_precomp=cp, opacities=OPA, scales=None, rotations=None, cov3D_precomp=cc)[0]
            imageio.imwrite(f'{a.out}/cam_{i}/{fi:03d}.png', (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    json.dump(CAMS, open(f'{a.out}/cams_sim.json', 'w'), indent=1)
    json.dump(dict(ctr=ctr.tolist(), lo=lo.tolist(), hi=hi.tolist(), dist=dist, el0=el0, fov=fov, n_frames=len(files), NG=NG, SO=float(SO)), open(f'{a.out}/mv_meta.json', 'w'), indent=1)
    print(f'[다시점] {a.out}: 카메라 {a.ncam} x 프레임 {len(files)}, 가우시안 {NG}, 거리 {dist:.3f} 고도 {el0:.1f}°\nMV_OK', flush=True)
if CMD == 'roll-to-h5':
    import torch
    import argparse
    import json
    import os
    import h5py
    import numpy as np
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--traj', required=True, help='원래 lbdata 궤적 디렉토리 (meta.json, sim_0000000000.h5)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--first_from_dump', action='store_true', help='프레임 0 을 원래 궤적 0 프레임 대신 덤프 pred[0]·F[0] 으로 (이어 돌리기 덤프: pred[0] = 시작 상태), 그 뒤 pred[1:]')
    ap.add_argument('--x0_add', default='', help='절 CM11: 원래 궤적 프레임 0 의 x 에 더할 벡터 "dx,dy,dz" (띄운 판의 처음 상태; F 는 그대로)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = torch.load(a.dump, weights_only=False)
    assert d.get('F') is not None, '덤프에 F 가 없다 -- AF_ROLL_DUMP_F=1 로 다시'
    meta = json.load(open(f'{a.traj}/meta.json'))
    R = torch.as_tensor(np.array(meta['R']), dtype=torch.float32)
    json.dump(meta, open(f'{a.out}/meta.json', 'w'), indent=1)
    (P, F) = (d['pred'].float(), d['F'].float())
    if a.first_from_dump:
        (P, F) = (P[1:], F[1:])
        with h5py.File(f'{a.out}/sim_0000000000.h5', 'w') as o:
            o.create_dataset('x', data=d['pred'][0].float().numpy().astype(np.float32))
            o.create_dataset('F', data=(d['F'][0].float() @ R).reshape(-1, 9).numpy().astype(np.float32))
    else:
        with h5py.File(f'{a.traj}/sim_0000000000.h5') as h, h5py.File(f'{a.out}/sim_0000000000.h5', 'w') as o:
            for k in ('x', 'F'):
                _v = np.array(h[k])
                if k == 'x' and a.x0_add:
                    _o = np.array([float(q) for q in a.x0_add.split(',')], dtype=_v.dtype)
                    _v = _v + (_o[:, None] if _v.shape[0] == 3 and _v.shape[1] != 3 else _o[None, :])
                o.create_dataset(k, data=_v)
    for i in range(P.shape[0]):
        Fr = (F[i] @ R).reshape(-1, 9)
        with h5py.File(f'{a.out}/sim_{i + 1:010d}.h5', 'w') as o:
            o.create_dataset('x', data=P[i].numpy().astype(np.float32))
            o.create_dataset('F', data=Fr.numpy().astype(np.float32))
    _Fl = F[-1] if F.shape[0] else d['F'][0].float()
    print(f'[저장] {a.out}  {P.shape[0] + 1} 프레임, det F 범위 {float(torch.linalg.det(_Fl).min()):.3f}~{float(torch.linalg.det(_Fl).max()):.3f}', flush=True)
if CMD == 'zero-energy-modes':
    import argparse
    import json
    import math
    import time
    import numpy as np
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--solver', required=True)
    ap.add_argument('--frames', default='0,c')
    ap.add_argument('--q', type=int, default=8)
    ap.add_argument('--kb', type=int, default=2400)
    ap.add_argument('--deg', type=int, default=24)
    ap.add_argument('--iters', type=int, default=14)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dev', default='cuda')
    ap.add_argument('--hg', type=float, default=0.0, help='절 CM194: --q 1 위 모래시계 벌점 γ (fusedgs/fem_gq1.py 와 같은 식: γμ Σ_g V/8 ‖(∇_g − ∇_c) y F‖²)')
    a = ap.parse_args()
    dev = a.dev
    torch.manual_seed(0)
    SV = torch.load(a.solver, map_location='cpu', weights_only=False)
    D = torch.load(a.pred, map_location='cpu', weights_only=False)
    XP = D['pred']
    FP = D.get('F')
    A = SV['A'].double().reshape(3, 3).to(dev)
    Ai = SV['Ai'].double().reshape(3, 3).to(dev)
    vol = SV['vol'].double().reshape(-1).to(dev)
    mu = float(SV['mu'].double().mean()) if torch.is_tensor(SV['mu']) else float(SV['mu'])
    lam = float(SV['lam'].double().mean()) if torch.is_tensor(SV['lam']) else float(SV['lam'])
    kap = SV.get('goct')
    kap = float(kap) if kap is not None and float(kap) >= 0 else 0.0
    cell = float(A.norm(dim=1).mean())

    def contact_frame():
        zmin = torch.stack([torch.nan_to_num(XP[f, :, 2].float(), nan=1000000000.0).min() for f in range(XP.shape[0])])
        lo = float(zmin.min())
        for f in range(XP.shape[0]):
            if float(zmin[f]) <= lo + 0.5 * cell:
                return f
        return int(zmin.argmin())

    def axis_tab(q):
        if q == 1:
            d = [0.0]
            w = [1.0]
            edges = []
        elif q == 8:
            d = [-0.5 / math.sqrt(3), 0.5 / math.sqrt(3)]
            w = [0.5, 0.5]
            edges = [1.0]
        elif q == 64:
            ga = math.sqrt(3 / 7 - 2 / 7 * math.sqrt(6 / 5))
            gb = math.sqrt(3 / 7 + 2 / 7 * math.sqrt(6 / 5))
            wa = (18 + math.sqrt(30)) / 36
            wb = (18 - math.sqrt(30)) / 36
            d = [-0.5 * gb, -0.5 * ga, 0.5 * ga, 0.5 * gb]
            w = [0.5 * wb, 0.5 * wa, 0.5 * wa, 0.5 * wb]
            edges = [0.5 + w[0], 0.5 + w[0] + w[1], 0.5 + w[0] + w[1] + w[2]]
        else:
            s = 0.5 * math.sqrt(0.6)
            d = [-s, 0.0, s]
            w = [5 / 18, 8 / 18, 5 / 18]
            edges = [0.5 + 5 / 18, 0.5 + 13 / 18]
        return (d, w, edges)

    def n_components(ids_c, nn):
        """절 CM195: 노드를 함께 쓰는 칸끼리 이은 연결 성분 수 (성분마다 강체 모드 6 개 = 탄성 헤시안의 자명한 영 모드)."""
        lab = torch.arange(nn, device=ids_c.device)
        for _ in range(100000):
            cl = lab[ids_c].min(1).values
            new = lab.clone().scatter_reduce_(0, ids_c.reshape(-1), cl[:, None].expand_as(ids_c).reshape(-1), reduce='amin')
            new = new[new]
            if torch.equal(new, lab):
                break
            lab = new
        return int(torch.unique(lab[ids_c.reshape(-1)]).numel())

    def build(fi, q):
        x = XP[fi].to(dev).double()
        F = FP[fi].to(dev).double().reshape(-1, 3, 3) if FP is not None else torch.eye(3, device=dev, dtype=torch.float64).expand(x.shape[0], 3, 3)
        ok = torch.isfinite(x).all(1) & torch.isfinite(F.reshape(-1, 9)).all(1)
        (x, F, v) = (x[ok], F[ok], vol[ok])
        y = x @ Ai.T
        b = torch.floor(y - 0.5)
        fx = y - b
        b = b.long()
        bmin = b.min(0).values
        b = b - bmin
        nbx = (b.max(0).values + 3).tolist()
        ck = (b[:, 0] * nbx[1] + b[:, 1]) * nbx[2] + b[:, 2]
        (cu, cinv) = torch.unique(ck, return_inverse=True)
        nc = cu.numel()
        cb = torch.stack([cu // (nbx[1] * nbx[2]), cu // nbx[2] % nbx[1], cu % nbx[2]], 1)
        (d1, w1, edges) = axis_tab(q)
        n1 = len(d1)
        sub = torch.zeros_like(fx, dtype=torch.long)
        for e in edges:
            sub += (fx >= e).long()
        so = (sub[:, 0] * n1 + sub[:, 1]) * n1 + sub[:, 2]
        nq1 = n1 ** 3
        qi = cinv * nq1 + so
        Vs = torch.zeros(nc * nq1, device=dev, dtype=torch.float64).index_add_(0, qi, v)
        Fs = torch.zeros(nc * nq1, 9, device=dev, dtype=torch.float64).index_add_(0, qi, v[:, None] * F.reshape(-1, 9))
        Vc = torch.zeros(nc, device=dev, dtype=torch.float64).index_add_(0, cinv, v)
        Fc = torch.zeros(nc, 9, device=dev, dtype=torch.float64).index_add_(0, cinv, v[:, None] * F.reshape(-1, 9))
        Wo = torch.tensor([w1[i] * w1[j] * w1[k] for i in range(n1) for j in range(n1) for k in range(n1)], device=dev, dtype=torch.float64)
        Vq = (1 - kap) * Vs + kap * (Vc[:, None] * Wo[None, :]).reshape(-1)
        Fq = torch.where((Vs > 0)[:, None], Fs / Vs.clamp_min(1e-300)[:, None], (Fc / Vc[:, None]).repeat_interleave(nq1, 0)).reshape(-1, 3, 3)
        BG = torch.zeros(nq1, 27, 3, device=dev, dtype=torch.float64)
        for o in range(nq1):
            ii = (o // (n1 * n1), o // n1 % n1, o % n1)
            fxv = [1.0 + d1[ii[t]] for t in range(3)]
            W = [[0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2] for f in fxv]
            dW = [[f - 1.5, -2 * (f - 1), f - 0.5] for f in fxv]
            for n in range(27):
                (i, j, k) = (n // 9, n // 3 % 3, n % 3)
                gy = torch.tensor([dW[0][i] * W[1][j] * W[2][k], W[0][i] * dW[1][j] * W[2][k], W[0][i] * W[1][j] * dW[2][k]], device=dev, dtype=torch.float64)
                BG[o, n] = gy @ Ai
        off = torch.tensor([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], device=dev)
        nodes = (cb[:, None, :] + off[None]).reshape(-1, 3)
        nk = (nodes[:, 0] * nbx[1] + nodes[:, 1]) * nbx[2] + nodes[:, 2]
        (nu, ninv) = torch.unique(nk, return_inverse=True)
        ids_c = ninv.reshape(nc, 27)
        keep = Vq > 0
        qcell = torch.arange(nc * nq1, device=dev) // nq1
        qo = torch.arange(nc * nq1, device=dev) % nq1
        (qcell, qo, Vq, Fq) = (qcell[keep], qo[keep], Vq[keep], Fq[keep])
        G = torch.einsum('qij,qni->qnj', Fq, BG[qo])
        ids = ids_c[qcell]
        occ = torch.zeros(nu.numel(), device=dev, dtype=torch.long).index_add_(0, ids_c.reshape(-1), torch.ones(ids_c.numel(), device=dev, dtype=torch.long))
        I3 = torch.eye(3, device=dev, dtype=torch.float64)
        f0 = dict(F_dev_max=float((F - I3).abs().amax()), x_sum=float(x.sum()), F_sum=float(F.sum()))
        R = dict(nc=nc, nq=int(keep.sum()), nn=int(nu.numel()), ids=ids, G=G.float(), V=Vq.float(), occ=occ, np=int(ok.sum()), qcell=qcell, ids_c=ids_c, ncomp=n_components(ids_c, int(nu.numel())), f0=f0)
        if q == 1 and a.hg > 0:
            g8 = 0.5 / math.sqrt(3)
            BG8 = torch.zeros(8, 27, 3, device=dev, dtype=torch.float64)
            for o in range(8):
                fxv = [1.0 + (g8 if o >> 2 - t & 1 else -g8) for t in range(3)]
                W = [[0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2] for f in fxv]
                dW = [[f - 1.5, -2 * (f - 1), f - 0.5] for f in fxv]
                for n in range(27):
                    (i, j, k) = (n // 9, n // 3 % 3, n % 3)
                    gy = torch.tensor([dW[0][i] * W[1][j] * W[2][k], W[0][i] * dW[1][j] * W[2][k], W[0][i] * W[1][j] * dW[2][k]], device=dev, dtype=torch.float64)
                    BG8[o, n] = gy @ Ai
            Dd = BG8 - BG[0][None]
            R['HGd'] = torch.einsum('qij,gni->qgnj', Fq, Dd).float()
        return R

    def make_op(S):
        """칸마다 81x81 강성 (점 합) 을 모아 희소 행렬 (CSR) 로 -- 야코비 배율 Â = D^-½ K D^-½ 곱은 cuSPARSE SpMM."""
        nn = S['nn']
        (ids, G, V) = (S['ids'], S['G'].double(), S['V'].double())
        nq = ids.shape[0]
        qc = S['qcell']
        nc = S['nc']
        Kc = torch.zeros(nc, 27, 3, 27, 3, device=dev, dtype=torch.float32)
        I3 = torch.eye(3, device=dev, dtype=torch.float64)
        CH = 4096
        for q0 in range(0, nq, CH):
            q1 = min(nq, q0 + CH)
            g = G[q0:q1]
            v = V[q0:q1]
            t1 = lam * torch.einsum('qna,qmb->qnamb', g, g)
            t3 = mu * torch.einsum('qnb,qma->qnamb', g, g)
            gg = torch.einsum('qnj,qmj->qnm', g, g)
            t2 = mu * gg[:, :, None, :, None] * I3[None, None, :, None, :]
            Kc.index_add_(0, qc[q0:q1], ((t1 + t2 + t3) * v[:, None, None, None, None]).float())
            del t1, t2, t3, gg
            if 'HGd' in S:
                d = S['HGd'][q0:q1].double()
                hh = torch.einsum('qgnj,qgmj->qnm', d, d) * (a.hg * mu / 8.0)
                Kc.index_add_(0, qc[q0:q1], (hh[:, :, None, :, None] * I3[None, None, :, None, :] * v[:, None, None, None, None]).float())
                del d, hh
        ic = S['ids_c']
        dof = (ic[:, :, None] * 3 + torch.arange(3, device=dev)[None, None, :]).reshape(nc, 81)
        rows = dof[:, :, None].expand(nc, 81, 81).reshape(-1)
        cols = dof[:, None, :].expand(nc, 81, 81).reshape(-1)
        vals = Kc.reshape(-1)
        del Kc
        n = nn * 3
        K = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n)).coalesce()
        del rows, cols, vals
        idx = K.indices()
        val = K.values()
        dg = torch.zeros(n, device=dev, dtype=torch.float64)
        m = idx[0] == idx[1]
        dg.index_add_(0, idx[0][m], val[m].double())
        act = dg > 1e-12 * dg.max()
        s = torch.zeros_like(dg)
        s[act] = dg[act].rsqrt()
        val = (val.double() * s[idx[0]] * s[idx[1]]).float()
        A = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce().to_sparse_csr()
        S['nnz'] = int(val.numel())
        del idx, val, K

        def Ah(Z):
            return torch.sparse.mm(A, Z)
        return (Ah, act, s)
    KB = [a.kb]

    def chfsi(Ah, act, n, kb):
        na = int(act.sum())
        kb = min(kb, na - 1)
        KB[0] = kb
        am = act.float()
        x = torch.randn(n, 1, device=dev) * am[:, None]
        for _ in range(40):
            x = Ah(x)
            x = x / x.norm()
        lmax = float((x * Ah(x)).sum()) * 1.05
        X = torch.randn(n, kb, device=dev) * am[:, None]
        (X, _) = torch.linalg.qr(X.double())
        X = X.float()
        cut = 0.02 * lmax
        hist = []
        for it in range(a.iters):
            e = (lmax - cut) / 2
            c = (lmax + cut) / 2
            Y0 = X
            Y1 = (Ah(X) - c * X) / e
            for _ in range(2, a.deg + 1):
                Y2 = 2 * (Ah(Y1) - c * Y1) / e - Y0
                (Y0, Y1) = (Y1, Y2)
                nr = Y1.norm(dim=0).max()
                if nr > 1000000000000000.0:
                    Y0 = Y0 / nr
                    Y1 = Y1 / nr
            (Q, _) = torch.linalg.qr(Y1.double())
            Q = Q.float()
            AQ = Ah(Q)
            Hs = Q.double().T @ AQ.double()
            Hs = 0.5 * (Hs + Hs.T)
            (ev, W) = torch.linalg.eigh(Hs)
            X = (Q.double() @ W).float()
            AX = (AQ.double() @ W).float()
            res = (AX - X * ev.float()[None, :]).norm(dim=0) / lmax
            cut = float(ev[-1]) if float(ev[-1]) > 0 else cut
            nz5 = int((ev < 1e-05 * lmax).sum())
            nz3 = int((ev < 0.001 * lmax).sum())
            hist.append(dict(it=it, cut_rel=cut / lmax, n_lt1e5=nz5, n_lt1e3=nz3, res_max_lt1e3=float(res[ev < 0.001 * lmax].max()) if nz3 else None))
            print('[q27 eig] it', it, hist[-1], flush=True)
        return (ev.cpu().numpy(), X, res.cpu().numpy(), lmax, hist)
    t0 = time.time()
    fr = []
    for t in a.frames.split(','):
        fr.append(contact_frame() if t == 'c' else int(t))
    OUT = dict(pred=a.pred, q=a.q, hg=a.hg, mu=mu, lam=lam, kap=kap, cell=cell, cases=[])
    for fi in fr:
        S = build(fi, a.q)
        (Ah, act, sc) = make_op(S)
        n = S['nn'] * 3
        (ev, X, res, lmax, hist) = chfsi(Ah, act, n, a.kb)
        z5 = ev < 1e-05 * lmax
        z3 = ev < 0.001 * lmax
        conv = res < 0.0001
        loc = {}
        occ = S['occ']
        bins = [(1, 8), (9, 17), (18, 26), (27, 27)]
        for (nm, msk) in (('zero', z5), ('near', z3 & ~z5)):
            k = int(msk.sum())
            row = dict(n=k)
            if k:
                U = sc.float()[:, None] * X[:, torch.from_numpy(msk).to(dev)]
                U = U / U.norm(dim=0, keepdim=True).clamp_min(1e-30)
                wn = (U * U).sum(1).reshape(-1, 3).sum(1) / k
                for (lo, hi) in bins:
                    m = (occ >= lo) & (occ <= hi)
                    row['%d-%d' % (lo, hi)] = dict(node_frac=float(m.float().mean()), mode_frac=float(wn[m].sum()))
            loc[nm] = row
        rec = dict(frame=fi, frame_label='f%d' % (fi + 1), nc=S['nc'], nq=S['nq'], nn=S['nn'], np=S['np'], nnz=S.get('nnz'), ndof_act=int(act.sum()), kb=KB[0], lmax=lmax, n_zero=int(z5.sum()), ncomp=S['ncomp'], n_rigid=6 * S['ncomp'], n_zero_nonrigid=int(z5.sum()) - 6 * S['ncomp'], f0=S['f0'], n_near=int((z3 & ~z5).sum()), n_zero_conv=int((z5 & conv).sum()), all_kb_small=bool(z3.all()), ev_head=[float(q) / lmax for q in ev[:12]], ev_at=[float(ev[min(len(ev) - 1, j)]) / lmax for j in (6, 30, 100, 300, 1000)], loc=loc, hist=hist[-3:], sec=round(time.time() - t0, 1))
        OUT['cases'].append(rec)
        print('[q27 eig] 프레임', fi, 'q', a.q, {k: rec[k] for k in ('nc', 'nq', 'nn', 'n_zero', 'ncomp', 'n_zero_nonrigid', 'n_near', 'n_zero_conv', 'all_kb_small')}, flush=True)
        json.dump(OUT, open(a.out, 'w'), indent=1)
        del Ah, X
        if dev == 'cuda':
            torch.cuda.empty_cache()
    print('[q27 eig] 끝', round(time.time() - t0, 1), 's')
