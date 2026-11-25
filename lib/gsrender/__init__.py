from __future__ import annotations
# ---------------- rtrender1 ----------------
import os


def _cu_sources():
    """CUDA sources of this package (compiled at run time by cupy). Each entry of gsrender.cu is
    '//@@ <name> <length>' on its own line followed by exactly <length> characters of source."""
    global _CU
    if _CU is None:
        _CU = {}
        t = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gsrender.cu'), encoding='utf-8', newline='').read()
        p = 0
        while p < len(t):
            e = t.index('\n', p)
            _, name, n = t[p:e].split(' ')
            _CU[name] = t[e + 1:e + 1 + int(n)]
            p = e + 1 + int(n) + 1
    return _CU


def cu(name):
    return _cu_sources()[name]


_CU = None

import json
import math
import os
import sys
import time
import numpy as np
import torch
W = os.environ.get('FUSEDGS_WORK', os.environ.get('FUSEDGS_WORK', '/workspace'))
MODEL = {'vasedeck': 'vasedeck_whitebg-trained', 'plane': 'plane-trained', 'pillow2sofa': 'pillow2sofa_whitebg-trained', 'lego': 'lego_whitebg-trained', 'ficus': 'ficus_whitebg-trained', 'wolf': 'wolf_whitebg-trained', 'bread': 'bread-trained'}
MODEL.update({'chair_plasticine': 'chair', 'drums_foam': 'drums', 'materials_sand': 'materials', 'materials_metal': 'materials', 'ball_metal': 'ball'})
MODEL.update({'chair_paste': 'chair', 'drums_honey': 'drums', 'bread_foam_drop': 'bread-trained', 'chair_paste_lego': 'chair', 'drums_honey_lego': 'drums', 'bread_foam_drop_lego': 'bread-trained'})
MODEL.update({'chair_elastic': 'chair', 'ship_plastic': 'ship', 'mic_metal': 'mic', 'mic_plastic': 'mic', 'mic_elastic': 'mic', 'ship_elastic': 'ship', 'ship_plastic_lego': 'ship', 'mic_metal_lego': 'mic', 'hotdog_plastic_lego': 'hotdog'})

class Cfg:
    kind = 'pg'
    shape = 'lego'
    traj = ''
    cfg = ''
    model = ''
    gf = f'{W}/GaussianFluent'
    fast = 0
    chk = ()
    out = '.'
C = Cfg()
ST = dict(ready=False, fi=0, log=None, clog=None, dep=0)

def _purge(root):
    for k in list(sys.modules):
        if k.split('.')[0] in ('utils', 'scene'):
            f = getattr(sys.modules[k], '__file__', None) or ''
            if not os.path.abspath(f).startswith(os.path.abspath(root)):
                print(f'[절 CM54 rtrender1] 모듈 {k} ({f}) 빼고 {root} 것으로', flush=True)
                sys.modules.pop(k)

def _setup_pg():
    cfg = json.load(open(C.cfg))
    meta = json.load(open(f'{C.traj}/meta.json'))
    sys.path.append(f'{W}/i-physgaussian')
    sys.path.append(f'{W}/i-physgaussian/gaussian-splatting')
    cwd = os.getcwd()
    os.chdir(f'{W}/i-physgaussian')
    try:
        _purge(f'{W}/i-physgaussian')
        from scene.gaussian_model import GaussianModel
        from utils.transformation_utils import transform2origin, shift2center111
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    finally:
        os.chdir(cwd)
    import h5py
    dev = 'cuda'
    mp = f'{W}/pgmodel/{MODEL[C.shape]}'
    gs = GaussianModel(3)
    it = max((int(q.split('_')[-1]) for q in os.listdir(f'{mp}/point_cloud') if q.startswith('iteration_')))
    gs.load_ply(f'{mp}/point_cloud/iteration_{it}/point_cloud.ply')
    KIDX = torch.nonzero(gs.get_opacity[:, 0] > float(cfg.get('opacity_threshold', 0.02))).squeeze(1)
    RT = torch.eye(3, device=dev)
    for (dg, ax) in zip(cfg.get('rotation_degree') or [], cfg.get('rotation_axis') or []):
        (c_, s_) = (math.cos(dg / 180.0 * 3.1415926), math.sin(dg / 180.0 * 3.1415926))
        R_ = {0: [[1, 0, 0], [0, c_, -s_], [0, s_, c_]], 1: [[c_, 0, s_], [0, 1, 0], [-s_, 0, c_]], 2: [[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]]}[int(ax)]
        RT = torch.tensor(R_, device=dev, dtype=torch.float32) @ RT
    ROT = bool((RT - torch.eye(3, device=dev)).abs().max() > 0)
    _XR = gs.get_xyz[KIDX] @ RT.T if ROT else gs.get_xyz[KIDX]
    if C.shape in ('vasedeck', 'plane') and cfg.get('sim_area') is not None:
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
    c6 = gs.get_covariance()[KIDX]
    C0 = torch.zeros(NG, 3, 3, device=dev)
    (C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2]) = (c6[:, 0], c6[:, 1], c6[:, 2])
    (C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2]) = (c6[:, 3], c6[:, 4], c6[:, 5])
    (C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1]) = (c6[:, 1], c6[:, 2], c6[:, 4])
    if ROT:
        C0 = RT @ C0 @ RT.T
    cam = json.load(open(f'{mp}/cameras.json'))[0]
    (Rw, pos) = (np.array(cam['rotation']), np.array(cam['position']))
    ctr = (MEAN @ RT if ROT else MEAN).cpu().numpy().astype(np.float64)
    pos = ctr + 1.6 * (pos - ctr)
    W2C = np.linalg.inv(np.block([[Rw, pos[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
    fx_ = 2 * math.atan(cam['width'] / (2 * cam['fx']))
    fy_ = 2 * math.atan(cam['height'] / (2 * cam['fy']))
    wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
    pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fx_, fovY=fy_).transpose(0, 1).to(dev).float()
    RAST = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(image_height=int(cam['height']), image_width=int(cam['width']), tanfovx=math.tan(fx_ * 0.5), tanfovy=math.tan(fy_ * 0.5), bg=torch.ones(3, device=dev), scale_modifier=1.0, viewmatrix=wv, projmatrix=(wv[None] @ pj[None])[0], sh_degree=3, campos=wv.inverse()[3, :3], prefiltered=False, debug=False))
    Rm = torch.as_tensor(np.array(meta['R']), dtype=torch.float32, device=dev)
    G = dict(NG=NG, H=int(cam['height']), Wd=int(cam['width']), C0=C0.contiguous(), RT=RT, ROT=ROT, SO=SO, MEAN=MEAN, Rm=Rm, SHS=gs.get_features[KIDX].contiguous(), OPA=gs.get_opacity[KIDX].contiguous(), RAST=RAST, corr_err=err)
    G['C0R'] = (Rm @ C0 @ Rm.T).contiguous()
    return G

def _setup_gf():
    dev = 'cuda'
    sys.path.insert(0, C.gf)
    sys.path.insert(1, os.path.join(C.gf, 'gaussian-splatting'))
    cwd = os.getcwd()
    os.chdir(C.gf)
    try:
        _purge(C.gf)
        from utils.decode_param import decode_param_json
        import types
        import utils.transformation_utils as _TU
        import utils.camera_view_utils as _CV
        import utils.render_utils as _RU
        from scene.gaussian_model import GaussianModel
        from utils.system_utils import searchForMaxIteration
        NS = {}
        for _m in (_TU, _CV, _RU):
            NS.update({k: v for (k, v) in vars(_m).items() if not k.startswith('_')})
        TU = CV = RU = types.SimpleNamespace(**NS)

        class PP:
            convert_SHs_python = False
            compute_cov3D_python = True
            debug = False
        (material_params, bc_params, time_params, pp, camera_params) = decode_param_json(C.cfg)
        d = os.path.join(C.model, 'point_cloud')
        gaussians = GaussianModel(3)
        gaussians.load_ply(os.path.join(d, f'iteration_{searchForMaxIteration(d)}', 'point_cloud.ply'))
        pipeline = PP()
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=dev)
        params = RU.load_params_from_gs(gaussians, pipeline)
        (init_pos, init_cov, init_opacity, init_shs) = (params['pos'], params['cov3D_precomp'], params['opacity'], params['shs'])
        m = init_opacity[:, 0] > pp['opacity_threshold']
        (init_pos, init_cov, init_opacity, init_shs) = (init_pos[m], init_cov[m], init_opacity[m], init_shs[m])
        for k in ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation'):
            setattr(gaussians, k, getattr(gaussians, k)[m])
        R = TU.generate_rotation_matrices(torch.tensor(pp['rotation_degree']), pp['rotation_axis'])
        rp = TU.apply_rotations(init_pos, R)
        if pp['sim_area'] is not None:
            b = pp['sim_area']
            m = torch.ones(rp.shape[0], dtype=torch.bool, device=dev)
            for i in range(3):
                m = m & (rp[:, i] > b[2 * i]) & (rp[:, i] < b[2 * i + 1])
            (rp, init_cov, init_opacity, init_shs) = (rp[m], init_cov[m], init_opacity[m], init_shs[m])
        (tp, so, mp_) = TU.transform2origin(rp)
        tp = TU.shift2center111(tp)
        init_cov = TU.apply_cov_rotations(init_cov, R) * so * so
        NG = tp.shape[0]
        C0 = torch.zeros(NG, 3, 3, device=dev)
        c = init_cov
        (C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2]) = (c[:, 0], c[:, 1], c[:, 2])
        (C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2]) = (c[:, 3], c[:, 4], c[:, 5])
        (C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1]) = (c[:, 1], c[:, 2], c[:, 4])
        (vc, oc) = CV.get_center_view_worldspace_and_observant_coordinate(torch.tensor(camera_params['mpm_space_viewpoint_center']).reshape((1, 3)).cuda(), torch.tensor(camera_params['mpm_space_vertical_upward_axis']).reshape((1, 3)).cuda(), R, so, mp_)
        CAMS = {}
        for t in range(0, 1001):
            if t > 0 and (not camera_params['move_camera']):
                CAMS[t] = CAMS[0]
                continue
            cam = CV.get_camera_view(C.model, default_camera_index=camera_params['default_camera_index'], center_view_world_space=vc, observant_coordinates=oc, show_hint=camera_params['show_hint'], init_azimuthm=camera_params['init_azimuthm'], init_elevation=camera_params['init_elevation'], init_radius=camera_params['init_radius'], move_camera=camera_params['move_camera'], current_frame=t, delta_a=camera_params['delta_a'], delta_e=camera_params['delta_e'], delta_r=camera_params['delta_r'], width=600, height=600)
            CAMS[t] = (cam, RU.initialize_resterize(cam, gaussians, pipeline, background))
            if t > 200:
                break
    finally:
        os.chdir(cwd)
    meta = json.load(open(f'{C.traj}/meta.json'))
    Rm = torch.as_tensor(np.array(meta['R']), dtype=torch.float32, device=dev)
    E = torch.eye(3, device=dev, dtype=torch.float32)
    f_pos = lambda X: TU.apply_inverse_rotations(TU.undotransform2origin(TU.undoshift2center111(X), so, mp_), R)
    b0 = f_pos(torch.zeros(1, 3, device=dev))
    A = (f_pos(E) - b0).T
    f_cov = lambda c6_: TU.apply_inverse_cov_rotations(c6_ / (so * so), R)
    L6 = f_cov(torch.eye(6, device=dev)).T
    xt = torch.randn(1000, 3, device=dev) + 1.0
    ct = torch.randn(1000, 6, device=dev)
    lin_err = [float((f_pos(xt) - (xt @ A.T + b0)).abs().max()), float((f_cov(ct) - ct @ L6.T).abs().max())]
    G = dict(NG=NG, H=600, Wd=600, C0=C0.contiguous(), C0R=(Rm @ C0 @ Rm.T).contiguous(), Rm=Rm, SHS=init_shs.contiguous(), OPA=init_opacity.contiguous(), CAMS=CAMS, gaussians=gaussians, RU=RU, TU=TU, R=R, so=so, mp=mp_, A=A.contiguous(), b0=b0.reshape(3).contiguous(), L6=L6.contiguous(), lin_err=lin_err, sh_deg=int(gaussians.active_sh_degree), shs_shape=list(init_shs.shape))
    return G

def rtrender1__inv3(X):
    (a, b, c) = (X[:, 0, 0], X[:, 0, 1], X[:, 0, 2])
    (d, e, f) = (X[:, 1, 0], X[:, 1, 1], X[:, 1, 2])
    (g, h, i) = (X[:, 2, 0], X[:, 2, 1], X[:, 2, 2])
    (A_, B_, C_) = (e * i - f * h, -(d * i - f * g), d * h - e * g)
    det = a * A_ + b * B_ + c * C_
    adj = torch.stack([A_, -(b * i - c * h), b * f - c * e, B_, a * i - c * g, -(a * f - c * d), C_, -(a * h - b * g), a * e - b * d], 1).reshape(-1, 3, 3)
    return adj / det[:, None, None]

def polar_t(F, it=10):
    X = F.double()
    for _ in range(it):
        Xi = rtrender1__inv3(X)
        g = torch.sqrt(Xi.flatten(1).norm(dim=1) / X.flatten(1).norm(dim=1))[:, None, None]
        X = 0.5 * (g * X + Xi.transpose(1, 2) / g)
    return X.float()

def upd_pg0(G, s, gi):
    x = s.x[gi]
    F = s.F.reshape(-1, 9)[gi].reshape(-1, 3, 3) @ G['Rm']
    cov = F @ G['C0'] @ F.transpose(1, 2)
    if G['ROT']:
        cov = G['RT'].T @ cov @ G['RT']
    cc = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
    Pm = (x - 1.0) / G['SO'] + G['MEAN']
    if G['ROT']:
        Pm = Pm @ G['RT']
    return dict(P=Pm, cc=cc, shs=G['SHS'], col=None, opa=G['OPA'])

def ras_pg(G, U, t):
    P = U['P']
    return G['RAST'](means3D=P, means2D=torch.zeros_like(P), shs=U['shs'], colors_precomp=U['col'], opacities=U['opa'], scales=None, rotations=None, cov3D_precomp=U['cc'])[0]

def upd_gf0(G, s, gi, t):
    (TU, RU) = (G['TU'], G['RU'])
    X = s.x[gi]
    Fg = s.F.reshape(-1, 9)[gi].reshape(-1, 3, 3) @ G['Rm']
    ok = torch.isfinite(X).all(1) & torch.isfinite(Fg).all(2).all(1)
    cov = Fg @ G['C0'] @ Fg.transpose(1, 2)
    c6 = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
    rot = polar_t(Fg)
    pos = TU.apply_inverse_rotations(TU.undotransform2origin(TU.undoshift2center111(X), G['so'], G['mp']), G['R'])
    c6 = TU.apply_inverse_cov_rotations(c6 / (G['so'] * G['so']), G['R'])
    cam = G['CAMS'][t][0]
    col = RU.convert_SH(G['SHS'][ok], cam, G['gaussians'], pos[ok], rot[ok]).float()
    return dict(P=pos[ok], cc=c6[ok], shs=None, col=col, opa=G['OPA'][ok], rot=rot, ok=ok)

def ras_gf(G, U, t):
    P = U['P']
    return G['CAMS'][t][1](means3D=P, means2D=torch.zeros_like(P), shs=None, colors_precomp=U['col'], opacities=U['opa'], scales=None, rotations=None, cov3D_precomp=U['cc'])[0]
_KSRC = cu('_KSRC')
_K = {}

def _kern(name):
    if name not in _K:
        import cupy as cp
        _K[name] = cp.RawModule(code=_KSRC, options=('--std=c++14',)).get_function(name)
    return _K[name]

def _launch(name, NG, args):
    import cupy as cp
    aa = []
    for a in args:
        if torch.is_tensor(a):
            assert a.is_cuda and a.is_contiguous() and (a.dtype in (torch.float32, torch.int64)), (name, a.dtype, a.shape)
            aa.append(cp.asarray(a))
        else:
            aa.append(a)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        _kern(name)(((NG + 255) // 256,), (256,), tuple(aa))

def upd_pg1(G, s, gi):
    NG = G['NG']
    if 'bufP' not in G:
        G['bufP'] = torch.empty(NG, 3, device='cuda')
        G['bufC'] = torch.empty(NG, 6, device='cuda')
        G['M'] = torch.cat([G['RT'].reshape(9), G['MEAN'].float().reshape(3), (1.0 / G['SO']).float().reshape(1)]).contiguous()
    _launch('rr_pg', NG, (s.x, s.F, gi, G['C0R'], G['M'], np.int32(1 if G['ROT'] else 0), G['bufP'], G['bufC'], np.int32(NG)))
    return dict(P=G['bufP'], cc=G['bufC'], shs=G['SHS'], col=None, opa=G['OPA'])

def upd_gf1(G, s, gi, t):
    NG = G['NG']
    if 'bufP' not in G:
        G['bufP'] = torch.empty(NG, 3, device='cuda')
        G['bufC'] = torch.empty(NG, 6, device='cuda')
        G['bufK'] = torch.empty(NG, 3, device='cuda')
        G['CP'] = {}
    if t not in G['CP']:
        G['CP'][t] = G['CAMS'][t][0].camera_center.float().reshape(3).contiguous().cuda()
    _launch('rr_gf', NG, (s.x, s.F, gi, G['C0'], G['Rm'], G['A'], G['b0'], G['L6'], G['SHS'], G['CP'][t], G['bufP'], G['bufC'], G['bufK'], np.int32(NG), np.int32(1 if C.fast >= 2 else 0)))
    return dict(P=G['bufP'], cc=G['bufC'], shs=None, col=G['bufK'], opa=G['OPA'])

def frame(s):
    if C.kind == 'pg':
        return _frame(s, lambda G, s_, gi, t: (upd_pg1 if C.fast else upd_pg0)(G, s_, gi), lambda G, U, t: ras_pg(G, U, t))
    return _frame(s, lambda G, s_, gi, t: (upd_gf1 if C.fast else upd_gf0)(G, s_, gi, t), lambda G, U, t: ras_gf(G, U, t))

def _img_u8(img):
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

def _frame(s, upd, ras):
    fi = ST['fi']
    ST['fi'] += 1
    t = fi + 1
    sim_ms = float(s.last_ms)
    with torch.no_grad():
        if not ST['ready']:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            G = _setup_pg() if C.kind == 'pg' else _setup_gf()
            if C.fast and C.kind == 'gf':
                assert G['sh_deg'] == 3 and G['shs_shape'][1:] == [16, 3], ('합친 커널은 SH 3 차 (N,16,3) 만', G['sh_deg'], G['shs_shape'])
            ST['G'] = G
            ST['ready'] = True
            ST['log'] = open(os.path.join(C.out, 'rr.jsonl'), 'w')
            meta = {}
            for (k, v) in G.items():
                try:
                    json.dumps(v)
                    meta[k] = v
                except TypeError:
                    pass
            meta.update(kind=C.kind, shape=C.shape, fast=C.fast, setup_s=time.perf_counter() - t0, Np=int(s.x.shape[0]))
            json.dump(meta, open(os.path.join(C.out, 'rr_meta.json'), 'w'), ensure_ascii=False, indent=1)
            print(f"[절 CM54 rtrender1] {C.kind} {C.shape} 가우시안 {G['NG']} 해상도 {G['Wd']}x{G['H']} 갱신 방식 {C.fast} (준비 {meta['setup_s']:.1f} s)", flush=True)
        G = ST['G']
        if ST.get('perm') is not s.perm or getattr(s, 'det', False):
            inv = torch.empty_like(s.perm)
            inv[s.perm] = torch.arange(s.perm.numel(), device=s.perm.device)
            ST['gi'] = inv[:G['NG']].long().contiguous()
            ST['perm'] = s.perm
        gi = ST['gi']
        assert s.x.dtype == torch.float32 and s.F.dtype == torch.float32 and s.x.is_contiguous() and s.F.is_contiguous()
        torch.cuda.synchronize()
        (e0, e1, e2) = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        w0 = time.perf_counter()
        e0.record()
        U = upd(G, s, gi, t)
        e1.record()
        img = ras(G, U, t)
        e2.record()
        e2.synchronize()
        w1 = time.perf_counter()
        (b, c) = (e0.elapsed_time(e1), e1.elapsed_time(e2))
        rec = dict(t=fi, a=sim_ms, b=b, c=c, tot=sim_ms + b + c, wall_bc=1000.0 * (w1 - w0))
        if t in C.chk:
            os.makedirs(os.path.join(C.out, 'rrimg'), exist_ok=True)
            u8 = _img_u8(img)
            import imageio.v2 as imageio
            imageio.imwrite(os.path.join(C.out, 'rrimg', f'{t:04d}.png'), u8)
            ck = dict(t=t)
            if C.fast:
                U0 = upd_pg0(G, s, gi) if C.kind == 'pg' else upd_gf0(G, s, gi, t)
                img0 = ras(G, U0, t)
                if C.kind == 'gf':
                    ok = U0['ok']
                    ck.update(nok=int((~ok).sum()), dP=float((U['P'][ok] - U0['P']).abs().max()), dC=float((U['cc'][ok] - U0['cc']).abs().max()), rC=float(((U['cc'][ok] - U0['cc']).abs().max(1).values / U0['cc'].abs().max(1).values.clamp_min(1e-30)).max()), dK=float((U['col'][ok] - U0['col']).abs().max()))
                else:
                    ck.update(dP=float((U['P'] - U0['P']).abs().max()), dC=float((U['cc'] - U0['cc']).abs().max()), rC=float(((U['cc'] - U0['cc']).abs().max(1).values / U0['cc'].abs().max(1).values.clamp_min(1e-30)).max()))
                d = (img - img0).abs()
                u80 = _img_u8(img0)
                ck.update(img_max=float(d.max()), img_mean=float(d.mean()), u8_ndiff=int((u8 != u80).any(2).sum()), u8_max=int(np.abs(u8.astype(int) - u80).max()))
            if C.kind == 'gf':
                Fg = s.F.reshape(-1, 9)[gi].reshape(-1, 3, 3) @ G['Rm']
                okk = torch.isfinite(Fg).all(2).all(1)
                (Uc, _, Vh) = torch.linalg.svd(Fg[okk].cpu())
                rp = polar_t(Fg[okk]).cpu()
                dd = (rp - Uc @ Vh).abs().flatten(1).max(1).values
                ck.update(polar_max=float(dd.max()), polar_p999=float(torch.quantile(dd[::7].double(), 0.999)), polar_n1e3=int((dd > 0.001).sum()), detF_min=float(torch.linalg.det(Fg[okk].double()).min()))
            if ST['clog'] is None:
                ST['clog'] = open(os.path.join(C.out, 'rr_chk.jsonl'), 'w')
            ST['clog'].write(json.dumps(ck) + '\n')
            ST['clog'].flush()
        ST['log'].write(json.dumps(rec) + '\n')
        ST['log'].flush()

def install(kind, shape, traj, cfg, model, gf, fast, chk, out):
    (C.kind, C.shape, C.traj, C.cfg, C.model, C.gf, C.fast, C.out) = (kind, shape, traj, cfg, model, gf or C.gf, int(fast), out)
    C.chk = tuple((int(q) for q in chk.split(',') if q.strip())) if chk else ()
    import hexfem as ART
    import hexfem as AORT
    import hexfem as WRX

    def outer(fn):

        def w(s, *a, **kw):
            ST['dep'] += 1
            try:
                r = fn(s, *a, **kw)
            finally:
                ST['dep'] -= 1
            if ST['dep'] == 0:
                frame(s)
            return r
        return w
    ART.an_rt_split_step = outer(ART.an_rt_split_step)
    AORT.ao_rt_split_step = outer(AORT.ao_rt_split_step)
    WRX.split_col = outer(WRX.split_col)
    print(f'[절 CM54 rtrender1] 훅 설치 {kind} {shape} 갱신 방식 {fast} 확인 프레임 {C.chk}', flush=True)

# ---------------- wf ----------------
import itertools
C1_NOUT = 54
CLASSES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], float)

def _trans_pairs():
    E = np.eye(3)
    out = []
    for d in CLASSES:
        for (a_, b_) in ((0, 1), (0, 2), (1, 2)):
            if abs(np.linalg.det(np.stack([d, E[a_], E[b_]]))) > 0.5:
                out.append((E[a_], E[b_]))
                break
    return np.array(out)
_TRANS = _trans_pairs()
PL_LEGACY_DUP = [False]

def multi4(n):
    return [m for m in itertools.product(range(n + 1), repeat=4) if sum(m) == n][::-1]
(I1, I2, I3, I4, I6) = (multi4(1), multi4(2), multi4(3), multi4(4), multi4(6))
ID3 = {m: k for (k, m) in enumerate(I3)}

def mcoef(m):
    return math.factorial(sum(m)) / np.prod([math.factorial(x) for x in m])

def bern_np(n, L):
    L = np.atleast_2d(L)
    return np.stack([mcoef(m) * np.prod([L[:, i] ** m[i] for i in range(4)], 0) for m in multi4(n)], 1)

def _add(m, e):
    return tuple((x + y for (x, y) in zip(m, e)))
EYE4 = [tuple((int(i == j) for j in range(4))) for i in range(4)]
PERMS = list(itertools.permutations(range(3)))

def chain(p):
    v = [np.zeros(3)]
    for r in range(3):
        v.append(v[-1] + np.eye(3)[p[r]])
    return np.array(v)

def incenter(V):
    A = []
    for i in range(4):
        f = [V[j] for j in range(4) if j != i]
        A.append(0.5 * np.linalg.norm(np.cross(f[1] - f[0], f[2] - f[0])))
    A = np.array(A)
    return (A[:, None] * V).sum(0) / A.sum()

def neighbor_vertex(V, i):
    return V[(i - 1) % 4] + V[(i + 1) % 4] - V[i]

def bary4(P, x):
    A = np.vstack([P.T, np.ones(4)])
    return np.linalg.solve(A, np.r_[x, 1.0])

def build_type(p):
    V = chain(p)
    z = incenter(V)
    W = {}
    for i in range(4):
        F = [V[j] for j in range(4) if j != i]
        Vn = np.array(F + [neighbor_vertex(V, i)])
        zn = incenter(Vn)
        nrm = np.cross(F[1] - F[0], F[2] - F[0])
        t = np.dot(F[0] - z, nrm) / np.dot(zn - z, nrm)
        w = z + t * (zn - z)
        lb = np.linalg.lstsq(np.vstack([np.array(F).T, np.ones(3)]), np.r_[w, 1.0], rcond=None)[0]
        assert lb.min() > 1e-06, ('면 점이 면 밖', p, i, lb)
        W[i] = (w, lb)
    subs = []
    sub_lbl = []
    for i in range(4):
        f = [j for j in range(4) if j != i]
        for k in range(3):
            (a_, b_) = (f[k], f[(k + 1) % 3])
            subs.append(np.array([V[a_], V[b_], W[i][0], z]))
            sub_lbl.append((a_, b_, i))
    subs = np.array(subs)
    key = lambda x: tuple(np.round(x * 3000000.0).astype(np.int64))
    pts = {}
    sidx = np.zeros((12, 20), int)
    for (s, S) in enumerate(subs):
        for (k, m) in enumerate(I3):
            x = (np.array(m)[:, None] * S).sum(0) / 3.0
            kk = key(x)
            if kk not in pts:
                pts[kk] = len(pts)
            sidx[s, k] = pts[kk]
    U = len(pts)
    (rows, rhs) = ([], [])
    for s1 in range(12):
        for s2 in range(s1 + 1, 12):
            k1 = {key(x): j for (j, x) in enumerate(subs[s1])}
            sh = [key(x) for x in subs[s2] if key(x) in k1]
            if len(sh) != 3:
                continue
            e2 = [j for (j, x) in enumerate(subs[s2]) if key(x) not in k1][0]
            lam = bary4(subs[s1], subs[s2][e2])
            sh1 = [k1[kk] for kk in sh]
            sh2 = [j for (j, x) in enumerate(subs[s2]) if key(x) in k1]
            for mu in itertools.product(range(3), repeat=3):
                if sum(mu) != 2:
                    continue
                base = sum((mu[q] * subs[s1][sh1[q]] for q in range(3)))
                r = np.zeros(U)
                r[pts[key((base + subs[s2][e2]) / 3.0)]] -= 1
                for j in range(4):
                    r[pts[key((base + subs[s1][j]) / 3.0)]] += lam[j]
                rows.append(r)
                rhs.append(np.zeros(28))
    nC = len(rows)
    EDG = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    def sub_with(*verts):
        for (s, S) in enumerate(subs):
            ks = {key(x) for x in S}
            if all((key(v) in ks for v in verts)):
                return s
        raise RuntimeError
    for i in range(4):
        r = np.zeros(U)
        r[pts[key(V[i])]] = 1
        e = np.zeros(28)
        e[4 * i] = 1
        rows.append(r)
        rhs.append(e)
        s = sub_with(V[i])
        S = subs[s]
        for q in S:
            if np.allclose(q, V[i]):
                continue
            r = np.zeros(U)
            r[pts[key((2 * V[i] + q) / 3)]] += 3
            r[pts[key(V[i])]] -= 3
            e = np.zeros(28)
            e[4 * i + 1:4 * i + 4] = q - V[i]
            rows.append(r)
            rhs.append(e)
    cls_of = []
    for (ei, (i, j)) in enumerate(EDG):
        d = V[j] - V[i]
        c = int(np.where(np.abs(CLASSES - d).sum(1) < 1e-09)[0][0])
        cls_of.append(c)
        mid = 0.5 * (V[i] + V[j])
        s = sub_with(V[i], V[j])
        S = subs[s]
        for sdir in range(2):
            t = _TRANS[c, sdir]
            dl = np.linalg.solve(np.vstack([S.T, np.ones(4)]), np.r_[t, 0.0])
            Lm = bary4(S, mid)
            B2 = bern_np(2, Lm)[0]
            r = np.zeros(U)
            for (g_, gm) in enumerate(I2):
                for j4 in range(4):
                    r[pts[key((np.array(_add(gm, EYE4[j4]))[:, None] * S).sum(0) / 3)]] += 3 * B2[g_] * dl[j4]
            e = np.zeros(28)
            e[16 + 2 * ei + sdir] = 1
            rows.append(r)
            rhs.append(e)
    A = np.array(rows)
    B = np.array(rhs)
    (sol, res, rk, sv) = np.linalg.lstsq(A, B, rcond=None)
    assert rk == U, ('WF 계가 유일해를 안 준다', rk, U)
    resid = float(np.abs(A @ sol - B).max())
    assert resid < 1e-09, ('WF 계 잔차', resid)
    ordm = sol[sidx]
    GD = np.zeros((12, 10, 3, 28))
    for (s, S) in enumerate(subs):
        Em = np.stack([S[1] - S[0], S[2] - S[0], S[3] - S[0]], 1)
        Ei = np.linalg.inv(Em)
        for (g_, gm) in enumerate(I2):
            D = np.stack([3 * (ordm[s, ID3[_add(gm, EYE4[k])]] - ordm[s, ID3[_add(gm, EYE4[0])]]) for k in (1, 2, 3)], 0)
            GD[s, g_] = Ei.T @ D
    Hinv = np.stack([np.linalg.inv(np.vstack([S.T, np.ones(4)])) for S in subs])
    return dict(V=V, z=z, W=W, sub=subs, ord=ordm, GD=GD, Hinv=Hinv, cls=np.array(cls_of), EDG=EDG, U=U, nC=nC, resid=resid, sub_lbl=sub_lbl)
TYPES = [build_type(p) for p in PERMS]

def _prod_table(na, nb):
    (Ia, Ib, Ic) = (multi4(na), multi4(nb), multi4(na + nb))
    ic = {m: k for (k, m) in enumerate(Ic)}
    (A, B, C, Wt) = ([], [], [], [])
    for (a_, ma) in enumerate(Ia):
        for (b_, mb) in enumerate(Ib):
            mc = _add(ma, mb)
            A.append(a_)
            B.append(b_)
            C.append(ic[mc])
            Wt.append(mcoef(ma) * mcoef(mb) / mcoef(mc))
    return (np.array(A), np.array(B), np.array(C), np.array(Wt), len(Ic))
P22 = _prod_table(2, 2)
P42 = _prod_table(4, 2)
_TT = {}

def tables(device, dtype):
    raise NotImplementedError('not used by the paper settings')

def _face_tess(T):
    key = lambda x: tuple(np.round(x * 3000000.0).astype(np.int64))
    (pts, P_sub, P_L, tris) = ({}, [], [], [])
    for (s, (a_, b_, i)) in enumerate(T['sub_lbl']):
        S = T['sub'][s]
        loc = {}
        for (p, q, r) in ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (0, 1, 1), (1, 0, 1)):
            L = np.array([p, q, r, 0]) / 2.0
            x = L @ S
            k = key(x)
            if k not in pts:
                pts[k] = len(pts)
                P_sub.append(s)
                P_L.append(L)
            loc[p, q, r] = pts[k]
        for tr in (((2, 0, 0), (1, 1, 0), (1, 0, 1)), ((1, 1, 0), (0, 2, 0), (0, 1, 1)), ((1, 0, 1), (0, 1, 1), (0, 0, 2)), ((1, 1, 0), (0, 1, 1), (1, 0, 1))):
            tris.append([loc[t] for t in tr])
    tris = np.array(tris)
    E = sorted({tuple(sorted((t[j], t[(j + 1) % 3]))) for t in tris for j in range(3)})
    E = np.array(E)
    ok = np.ones((len(E), len(tris)), bool)
    for (e, (p, q)) in enumerate(E):
        for (t, tr) in enumerate(tris):
            if p in tr or q in tr:
                ok[e, t] = False
    return (np.array(P_sub), np.array(P_L), tris, E, ok)
_TESS = [_face_tess(t) for t in TYPES]

# ---------------- simplex ----------------
from collections import namedtuple
simplex___all__ = ['Lat', 'lattice', 'grid_for_nodes', 'node_pos', 'locate', 'g2p', 'g2p_jac', 'g2p_pre', 'g2p_jac_pre', 'tet_det', 'tet_id', 'active_nodes', 'edges_of', 'EDGE_OFFSETS', 'N_EDGE_CLASS', 'HZ_CUBE']
HZ_CUBE = 1.0 / math.sqrt(6.0)
Lat = namedtuple('Lat', 'A Ai s hz')

def lattice(h, hz, device=None, dtype=torch.float32):
    r = float(h) / math.sqrt(3.0)
    (c, s3) = (0.5, math.sqrt(3.0) / 2.0)
    u = ((1.0, 0.0), (-c, s3), (-c, -s3))
    A = torch.tensor([[r * u[i][0] for i in range(3)], [r * u[i][1] for i in range(3)], [float(hz)] * 3], device=device, dtype=dtype)
    return Lat(A, torch.linalg.inv(A), float(h), float(hz))

def grid_for_nodes(x, n_nodes, margin=0.05, hz_ratio=HZ_CUBE, off=None, h_fix=None):
    xd = x.detach()
    lo_ = xd.min(0).values
    hi_ = xd.max(0).values
    ext = (hi_ - lo_).max().clamp_min(1e-12)
    pad = margin * ext
    h = float(h_fix) if h_fix else float((ext + 2 * pad) / max(float(n_nodes), 2.0))
    lat = lattice(h, hz_ratio * h, device=x.device, dtype=x.dtype)
    (b0, b1) = (lo_ - pad, hi_ + pad)
    cor = torch.stack([torch.stack([b0[0] if m & 1 else b1[0], b0[1] if m & 2 else b1[1], b0[2] if m & 4 else b1[2]]) for m in range(8)], 0)
    y = cor @ lat.Ai.T
    imin = torch.floor(y.min(0).values).long() - 1
    imax = torch.ceil(y.max(0).values).long() + 1
    nn = (imax - imin + 1).clamp_min(2)
    lo = lat.A @ imin.to(lat.A.dtype)
    if off is not None:
        _o = torch.as_tensor(off, device=lo.device, dtype=lat.A.dtype)
        lo = lo - lat.A @ _o
        nn = nn + 1
    return (lo, lat, nn)

def node_pos(lo, lat, nn, uniq):
    nnl = [int(nn[k]) for k in range(3)]
    k2 = uniq % nnl[2]
    k1 = uniq // nnl[2] % nnl[1]
    k0 = uniq // (nnl[1] * nnl[2])
    ijk = torch.stack([k0, k1, k2], -1).to(lat.A.dtype)
    return ijk @ lat.A.T + lo

def locate(q, lo, lat, nn):
    nnl = [int(nn[k]) for k in range(3)]
    y = (q - lo) @ lat.Ai.T
    ci = y.floor().long()
    ci = torch.stack([ci[:, k].clamp(0, nnl[k] - 2) for k in range(3)], -1)
    f = (y - ci).clamp(0.0, 1.0)
    perm = torch.argsort(f.detach(), dim=1, descending=True)
    rank = torch.argsort(perm, dim=1)
    sv = f.gather(1, perm)
    lam = torch.stack([1.0 - sv[:, 0], sv[:, 0] - sv[:, 1], sv[:, 1] - sv[:, 2], sv[:, 2]], -1)
    eye = torch.eye(3, device=q.device, dtype=torch.long)
    dirs = eye[perm]
    verts = torch.cat([torch.zeros_like(dirs[:, :1]), dirs.cumsum(1)], 1) + ci.unsqueeze(1)
    idx = (verts[..., 0] * nnl[1] + verts[..., 1]) * nnl[2] + verts[..., 2]
    return (idx, lam, (ci, rank, perm))

def g2p(q, lo, lat, nn, dp):
    raise NotImplementedError('not used by the paper settings')

def g2p_jac(q, lo, lat, nn, dp):
    raise NotImplementedError('not used by the paper settings')

def g2p_pre(rows, lam, dp_act):
    raise NotImplementedError('not used by the paper settings')
LOCAL_PROBE = [None]

def g2p_jac_pre(rows, lam, aux, lat, dp_act):
    raise NotImplementedError('not used by the paper settings')

def tet_det(G):
    raise NotImplementedError('not used by the paper settings')

def tet_id(lo, lat, nn, aux):
    raise NotImplementedError('not used by the paper settings')

def active_nodes(idx):
    (uniq, inv) = torch.unique(idx.reshape(-1), return_inverse=True)
    return (inv.reshape(idx.shape), uniq)
_POS_OFFSETS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
EDGE_OFFSETS = _POS_OFFSETS + [(-i, -j, -k) for (i, j, k) in _POS_OFFSETS]
simplex_N_EDGE_CLASS = len(EDGE_OFFSETS)
_OFF2CLS = {o: c for (c, o) in enumerate(EDGE_OFFSETS)}

def edges_of(idx_rows, uniq, nn):
    raise NotImplementedError('not used by the paper settings')

class Aux(tuple):
    bsp = None
    nav = None
_O27 = {}

def nodeavg_jac_pre(rows, lam, nav, lat, dp_act):
    raise NotImplementedError('not used by the paper settings')

# ---------------- nextstate ----------------
import torch.nn as nn
import torch.nn.functional as F

def nextstate_mlp(sizes, layernorm=True, act=nn.SiLU):
    raise NotImplementedError('not used by the paper settings')
try:
    from geobias import fused_geo_bias, HAVE_CUDA as _HAVE_GEOBIAS
except Exception:
    _HAVE_GEOBIAS = False

class nextstate_DtFiLM(nn.Module):

    def __init__(self, hidden, n_sites, n_freq=6):
        raise NotImplementedError('not used by the paper settings')

    def forward(self, dt, device):
        raise NotImplementedError('not used by the paper settings')

class GeoAttentionBias(nn.Module):

    def __init__(self, heads, hidden=32):
        raise NotImplementedError('not used by the paper settings')

    def forward(self, p):
        raise NotImplementedError('not used by the paper settings')
USE_SDPA = False

class GeoAttentionBlock(nn.Module):

    def __init__(self, hidden, heads=4):
        raise NotImplementedError('not used by the paper settings')

    def forward(self, x, bias):
        raise NotImplementedError('not used by the paper settings')

class NextStep(nn.Module):

    def __init__(self, hidden=128, depth=4, heads=4, scale=1.0, vel_scale=1.0, acc_scale=1.0, zero_init=False, use_accel=True, chunk=1, n_static=0, n_embed=0, m_anchors=0, frame=False, u_scale=1.0, s_scale=1.0, du_scale=1.0, ds_scale=1.0, n_extra=6):
        raise NotImplementedError('not used by the paper settings')

    def forward(self, p, v, a, dt, u=None, s=None):
        raise NotImplementedError('not used by the paper settings')

# ---------------- deform ----------------
import torch.nn.functional as Fn
deform_DtFiLM = nextstate_DtFiLM
deform_mlp = nextstate_mlp
try:
    import deformcuda as _dc
    _HAVE_DC = _dc.HAVE_CUDA
except Exception:
    (_dc, _HAVE_DC) = (None, False)
_DC_K = (4, 6, 8, 12, 16, 24, 32)

class RelPos(nn.Module):

    def __init__(self, heads, dc, lam_min, lam_max, sigma0, seed=0):
        raise NotImplementedError('not used by the paper settings')

    def forward(self, pos):
        raise NotImplementedError('not used by the paper settings')

class RelAttention(nn.Module):
    EXTRA = 4
    USE_SDPA = False

    def __init__(self, hidden, heads):
        raise NotImplementedError('not used by the paper settings')

    def forward(self, x, ctx):
        raise NotImplementedError('not used by the paper settings')

class RelBlock(nn.Module):

    def __init__(self, hidden, heads):
        raise NotImplementedError('not used by the paper settings')

    def forward(self, x, ctx):
        raise NotImplementedError('not used by the paper settings')

def deform__inv3(A):
    raise NotImplementedError('not used by the paper settings')
DMG_FLOOR = 0.02

class DeformNet(nn.Module):

    def __init__(self, n_feat, hidden=128, depth=4, heads=4, n_static=0, damage=False, scale=1.0, h=1.0, ext=1.0, zero_init=True, seed=0):
        raise NotImplementedError('not used by the paper settings')

    def forward(self, p, feat, dt, static=None):
        raise NotImplementedError('not used by the paper settings')

def deform_bc_features(p, cfg):
    raise NotImplementedError('not used by the paper settings')

# ---------------- bspfd ----------------
SX = sys.modules[__name__]
CAP_C = 0.48
CAP_K = 8
CAP_TAU = 0.01
N_SMP_AX = 2
_O64 = {}

def weights(q, lo, lat, nn, order):
    raise NotImplementedError('not used by the paper settings')

# ---------------- localrot ----------------
KAPPA = 0.9
_ZERO = bool(os.environ.get('AF_LOCAL_ROT_ZERO'))
STAT_KEYS = ('detJloc_min', 'out_of_cell', 'lam_y_min', 'lam_x_min', 'phi_max', 'phi_mean', 'thp_over_max', 'K_fro_max', 'theta_max_med', 'n')
LAST = [None]

# ---------------- skin ----------------
A_IN = 0.5

def skin_gregory_skin(x, npos, rows, U, rho_raw, h, return_parts=False):
    raise NotImplementedError('not used by the paper settings')

# ---------------- phys_resid ----------------
import contextlib
from typing import NamedTuple
phys_resid___all__ = ['lame', 'psi_of', 'plastic_step', 'Plast', 'ip_energy', 'residual', 'det3', 'cdmpm_reset', 'cdmpm_state', 'set_inv_barrier', 'set_cell_barrier', 'smooth_noise', 'mat_name', 'bc_node_mask', 'bc_energy', 'set_hardening', 'grid_ip_sub', 'grid_ip_pts', 'jac_neighbors', 'grad_from_points', 'p2g_ls', 'g2p_from_nodes']

def lame(E, nu):
    (E, nu) = (float(E), float(nu))
    return (E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

def mat_name(cfg):
    raise NotImplementedError('not used by the paper settings')
_JIT = None
_SIG_DT = torch.float64 if os.environ.get('AF_SIG_F64') else torch.float32

class Plast(NamedTuple):
    dlog: torch.Tensor
    V: torch.Tensor
    jp: 'torch.Tensor | None' = None
_JP = [None]
_HARD = [None]

def set_hardening(h):
    raise NotImplementedError('not used by the paper settings')

def det3(M):
    raise NotImplementedError('not used by the paper settings')

def cdmpm_reset(n, device=None, dtype=torch.float32, alpha0=-0.04):
    raise NotImplementedError('not used by the paper settings')

def cdmpm_state():
    raise NotImplementedError('not used by the paper settings')

class _FcrPsi(torch.autograd.Function):

    @staticmethod
    def forward(ctx, F, mu, lam, clamp):
        raise NotImplementedError('not used by the paper settings')
_CV = {}

def _sym_eig3(C):
    raise NotImplementedError('not used by the paper settings')
_FCR_FWD = [None]
_FCR_GRAD = [None]

class _FcrPsiFast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, F, mu, lam, clamp):
        raise NotImplementedError('not used by the paper settings')
_HESS = [False]

class _FcrPsiH(torch.autograd.Function):

    @staticmethod
    def forward(ctx, F, mu, lam, clamp):
        raise NotImplementedError('not used by the paper settings')

def psi_of(F_trial, cfg, dt):
    raise NotImplementedError('not used by the paper settings')

@torch.no_grad()
def plastic_step(F_trial, pl):
    raise NotImplementedError('not used by the paper settings')

def ip_energy(x2, xtil, F_trial, mass, vol, cfg, h, free=None, g=None, norm=None):
    raise NotImplementedError('not used by the paper settings')

def residual(E, x2, mass, ext):
    raise NotImplementedError('not used by the paper settings')

def rebuild_F(x_all, cfg, dt, k=16, chunk=0):
    raise NotImplementedError('not used by the paper settings')

def smooth_noise(x, sigma, ext, gen, n_wave=6):
    raise NotImplementedError('not used by the paper settings')
_OFF3 = None

def p2g_ls(x, du, mass, n_grid, grid_lim, ridge=1e-06):
    raise NotImplementedError('not used by the paper settings')

def g2p_from_nodes(x, G_I, info, n_grid):
    raise NotImplementedError('not used by the paper settings')

def bc_node_mask(uniq, n_grid, dx, cfg, dtype, dev):
    raise NotImplementedError('not used by the paper settings')
_BARRIER = [None]
_INVB = [None]
_CELLB = [None]

def set_inv_barrier(kappa=None, jhat=0.3, ext='linear'):
    raise NotImplementedError('not used by the paper settings')

def set_cell_barrier(kappa=None, jhat=0.3, ext='linear'):
    raise NotImplementedError('not used by the paper settings')

def _barrier_b(d, dhat, eps_r=0.001, ext_kind='quad'):
    if ext_kind == 'none':
        b = -(d - dhat) ** 2 * torch.log(d / dhat)
        return torch.where(d < dhat, b, torch.zeros_like(b))
    eps = eps_r * dhat
    dc = d.clamp_min(eps)
    b = -(dc - dhat) ** 2 * torch.log(dc / dhat)
    _l = math.log(eps / dhat)
    _k = eps - dhat
    be = -_k ** 2 * _l
    bp = -2.0 * _k * _l - _k ** 2 / eps
    bpp = -2.0 * _l - 4.0 * _k / eps + _k ** 2 / eps ** 2
    ext = be + bp * (d - eps)
    if ext_kind == 'quad':
        ext = ext + 0.5 * bpp * (d - eps) ** 2
    b = torch.where(d < eps, ext, b)
    return torch.where(d < dhat, b, torch.zeros_like(b))

def bc_energy(x, du, mass, cfg, h, grid_lim, n_grid, stiff=None):
    raise NotImplementedError('not used by the paper settings')

def jac_neighbors(x0, k=16, chunk=2048):
    raise NotImplementedError('not used by the paper settings')

def grad_from_points(x0, du, nb):
    raise NotImplementedError('not used by the paper settings')

def grid_ip_pts(x, du, vel, F, mass, vol, cfg, h, n_grid, grid_lim, g=None, norm=None, free=None, nb=None):
    raise NotImplementedError('not used by the paper settings')

def grid_ip_sub(x, du, vel, F, mass, vol, cfg, h, n_grid, grid_lim, g=None, norm=None, free=None, K=1):
    raise NotImplementedError('not used by the paper settings')

# ---------------- simplex_gnn ----------------
simplex_gnn__inv3 = deform__inv3
simplex_gnn_DtFiLM = nextstate_DtFiLM
simplex_gnn_N_EDGE_CLASS = simplex_N_EDGE_CLASS
simplex_gnn___all__ = ['scatter_to_nodes', 'node_moments', 'node_centroids', 'node_moments_finish', 'SimplexGNN', 'p2g_feats', 'P2G_NCH', 'p2g_stress_feats', 'P2G_APIC_NCH', 'P2G_STRESS_NCH', 'LocalAttnGNN', 'ring_edges', 'ring_offset_table', 'GraphUNetGNN', 'grid_ijk', 'coarsen_grid', 'gunet_plan']

def simplex_gnn_scatter_to_nodes(vals, rows, lam, n_node, mass=None):
    raise NotImplementedError('not used by the paper settings')

def simplex_gnn_node_moments(x, v, X, m, rows, lam, n_node, npos, hn):
    raise NotImplementedError('not used by the paper settings')
PFREQ_NS = 15
P2G_NCH = 13
P2G_APIC_NCH = 9
P2G_STRESS_NCH = 9

def p2g_feats(v, F, m, rows, w, n_node, C=None, dx=None, hn=None):
    raise NotImplementedError('not used by the paper settings')

def p2g_stress_feats(tau, vol0, m, rows, w, gradw, n_node, E, hn):
    raise NotImplementedError('not used by the paper settings')

def node_centroids(g1):
    raise NotImplementedError('not used by the paper settings')

def node_moments_finish(g1, g2, npos, hn):
    raise NotImplementedError('not used by the paper settings')
_MP_NORM = True
_MP_MEAN = True

class _MPLayer(nn.Module):

    def __init__(self, hidden):
        raise NotImplementedError('not used by the paper settings')

    def forward(self, h, src, dst, cls):
        raise NotImplementedError('not used by the paper settings')

class simplex_gnn_SimplexGNN(nn.Module):

    def __init__(self, n_feat, hidden=128, layers=8, scale=1.0, dt_cond=True, dt_ref=1.0, dt_scale=False, n_mat=0, n_out=3):
        raise NotImplementedError('not used by the paper settings')

    def forward(self, feat, src, dst, cls, dt, mat=None):
        raise NotImplementedError('not used by the paper settings')

class _LAttnLayer(nn.Module):

    def __init__(self, hidden, heads, ring=1):
        raise NotImplementedError('not used by the paper settings')

    def forward(self, h, src, dst, cls):
        raise NotImplementedError('not used by the paper settings')

def ring_offset_table(K):
    raise NotImplementedError('not used by the paper settings')

def ring_edges(src, dst, cls, K, n_node=None):
    raise NotImplementedError('not used by the paper settings')

class LocalAttnGNN(simplex_gnn_SimplexGNN):

    def __init__(self, n_feat, hidden=128, layers=8, heads=8, ring=1, **kw):
        raise NotImplementedError('not used by the paper settings')

    def forward(self, feat, src, dst, cls, dt, mat=None):
        raise NotImplementedError('not used by the paper settings')

def grid_ijk(uniq, nn):
    raise NotImplementedError('not used by the paper settings')

def coarsen_grid(pos, src, dst):
    raise NotImplementedError('not used by the paper settings')

def gunet_plan(levels, enc=2, bot=2, dec=1):
    raise NotImplementedError('not used by the paper settings')

class GraphUNetGNN(simplex_gnn_SimplexGNN):
    needs_grid = True

    def __init__(self, n_feat, hidden=128, levels=3, enc=2, bot=2, dec=1, bc_mode=None, bc_near_h=1.0, **kw):
        raise NotImplementedError('not used by the paper settings')

    def forward(self, feat, src, dst, cls, dt, mat=None, grid=None):
        raise NotImplementedError('not used by the paper settings')

# ---------------- deform_model ----------------
import glob
import types
phys_resid = sys.modules[__name__]
SX = sys.modules[__name__]
WF = sys.modules[__name__]
_SGM = sys.modules[__name__]
deform_model_bc_features = deform_bc_features
deform_model_SimplexGNN = simplex_gnn_SimplexGNN
deform_model_node_moments = simplex_gnn_node_moments
deform_model_scatter_to_nodes = simplex_gnn_scatter_to_nodes
deform_model_gregory_skin = skin_gregory_skin
LR = sys.modules[__name__]
BF = sys.modules[__name__]
FCAP = float(os.environ.get('AF_FEAT_CAP', '50'))
N_MAT_FULL = 13

def extent(x0):
    return float((x0.max(0).values - x0.min(0).values).norm())
REGION_DENSITY = [False]

def region_rho(cfg, x0):
    regs = cfg.get('additional_material_params') or []
    if not REGION_DENSITY[0] or not regs:
        return None
    rho = torch.full((x0.shape[0],), float(cfg['density']), device=x0.device, dtype=torch.float32)
    for r in regs:
        if float(r.get('E', cfg['E'])) != float(cfg['E']) or float(r.get('nu', cfg['nu'])) != float(cfg['nu']):
            raise SystemExit(f'--pg_region_density: E/nu 가 다른 영역은 아직 없다 ({r})')
        q = torch.as_tensor(r['point'], device=x0.device, dtype=x0.dtype)
        sz = torch.as_tensor(r['size'], device=x0.device, dtype=x0.dtype)
        ins = ((x0 > q - sz) & (x0 < q + sz)).all(-1)
        rho[ins] = float(r['density'])
    return rho

def traj_mass(d, dev='cuda'):
    if '_mass' in d:
        return d['_mass']
    c = d['cfg']
    x0 = d['x'][0].to(dev).float()
    ng_ = int(c.get('n_grid', 100))
    dx_ = float(c.get('grid_lim', 2.0)) / ng_
    vi_ = (x0 / dx_).long().clamp(0, ng_ - 1)
    fl_ = (vi_[:, 0] * ng_ + vi_[:, 1]) * ng_ + vi_[:, 2]
    cn_ = torch.zeros(ng_ ** 3, device=dev).index_add_(0, fl_, torch.ones(x0.shape[0], device=dev))
    _rho = region_rho(c, x0)
    if _rho is None:
        d['_mass'] = dx_ ** 3 / cn_[fl_] * float(c['density'])
    else:
        d['_mass'] = dx_ ** 3 / cn_[fl_] * _rho
        d['_rho'] = _rho
    d['_ext'] = extent(x0)
    return d['_mass']

def traj_F(d, frame_dt, dev='cuda', keep_gpu=True, on_rebuild=None):
    if d.get('_Fok') is None:
        _Fin = d['F']
        _id = float((_Fin[:3].float().to(dev) - torch.eye(3, device=dev)).abs().max()) < 1e-06
        if _id and (not d.get('F_is_true')):
            import time
            t_ = time.time()
            Fr = phys_resid.rebuild_F(d['x'].to(dev).float(), d['cfg'], frame_dt)
            d['F'] = Fr if keep_gpu else Fr.cpu()
            if on_rebuild is not None:
                on_rebuild(time.time() - t_)
        d['_Fok'] = True
    return d['F']
_TAU_CACHE = [None]
IP_FEAT_N = 18
IPF_LAST = [None]
_CONST = {}
PROBE = [None, None]
BSP_LAST = [None]
BSP_INFO = []
C1_LAST = [None]
INPUT_HOOK = [None]
_ARG_DEFAULTS = dict(arch='sgnn', hidden=128, gnn_layers=8, gnn_no_norm=False, gnn_sum=False, n_nodes=32, hz_ratio=1.0 / 6 ** 0.5, node_h=0.0, hex_jac=False, fe_state=False, control=False, net_inertial=False, v_from_dt=False, dt_cond=False, dt_scale=False, dt_sub=1, eval_dt_sub=0, ens_off=0, interp='bary', bc_soft=False, bc_mode='project', bc_level='node', bc_ipg=False, bc_softplus=0.0, bc_stick=0.0, bc_stick_xn=False, ov_init='predict', no_mat=False, mat_film=False, v0_traj=False, smooth_nb=0, cell_poly=False, ov_roll=0, refps=False, hold_traj=None, pool=True, node_feat='mom', gnn_layer='mp', attn_heads=8, apic=False, p2g_stress=False)

class EvalModel:
    UNSUPPORTED = ('control', 'fe_state', 'hex_jac', 'cell_poly', 'refps')

    def __init__(self, ckpt, dev='cuda', fast=False, fast_graph=True, fast_triton=True, fast_edge_triton=True):
        raise NotImplementedError('not used by the paper settings')

    def init(self, d, t0):
        raise NotImplementedError('not used by the paper settings')

    def step(self, d, x, v, h=None, sec=None):
        raise NotImplementedError('not used by the paper settings')

    def frame(self, d, x, v, sec=None):
        raise NotImplementedError('not used by the paper settings')
