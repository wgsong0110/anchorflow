"""FusedGS: run one simulation of a paper scene with one of the paper's settings.

  python exe/run_fusedgs.py --scene bread --setting default --data <scene>.pt --gt <gt dir> --out <dir>

  --data    FusedGS input: initial particles of the scene (tools.py pg-h5-to-pt)
  --gt      PhysGaussian ground-truth trajectory folder (sim_*.h5); used for the error metric and the in-loop 3DGS render
  --gt_cfg  scene config (default <gt>/config.json)
  --out     output folder: log.txt-style stdout, per-frame error / timing (*.jsonl), and, for settings that dump,
            the particle trajectory pred.pt {pred [T,N,3], F [T,N,3,3]} used by evaluation.py

Settings (cfg/settings.json; each expands to the exact solver configuration of the paper runs):
  default                       speed-accuracy (Fig. Pareto), tearing figure, ablation / robustness reference
  gauss1 / gauss27 / gauss64    1x1x1, 3x3x3, 4x4x4 Gauss points per cell (ablation, robustness)
  gauss1_timing                 1x1x1 run used for the timing column of the ablation table
  no_ftransfer                  elastic / plastic deformation-gradient transfer swapped (ablation)
  cell0.75 / cell1.5 / cell2    cell size x0.75 / x1.5 / x2 (robustness)
  substepsN, N = 1,2,...,128    N substeps per frame (robustness)
  stability                     stability table (lego, pillow2sofa, ship, mic); --N substeps, --stats 0 for timing runs
The solver uses a 16.7 ms per-frame time budget (real-time setting), so results vary slightly from run to run.
"""
import argparse
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'lib'))

SETTINGS = {sc: {k: (v['frames'], v['args']) for k, v in d.items()}
            for sc, d in json.load(open(os.path.join(ROOT, 'cfg', 'settings.json'))).items()}   # paper settings


# ---------------- wrapper chain (one function per wrapper of the recorded runs) ----------------
import os
import runpy
import sys
_REAL_RUN_PATH = runpy.run_path
WRAPPERS = {}

def w_escape1_wrap():
    import json
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    MF = 0.0
    if os.environ.get('AF_E1_DG', '0') == '1':
        import torch
        import hexfem as ART
        import hexfem as AORT
        import hexfem as WRX
        LOG = open(os.path.join(out, 'e1.jsonl'), 'w')
        ST = dict(dep=0, far=None)

        def _bw(fx):
            return torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], dim=-1)

        @torch.no_grad()
        def diag(s, fi):
            IP = s.IP.cpu()
            (nb0, nb1, nb2, nn) = (int(IP[6]), int(IP[7]), int(IP[8]), int(IP[9]))
            mn = s.mn[:nn].float()
            has = mn > 0
            med = float(mn[has].median()) if bool(has.any()) else 1.0
            r = mn / med
            rh = r[has]
            sp = s.vk.reshape(-1, 3)[:nn].float().norm(dim=-1)
            fast = has & (sp > 3.0)
            if getattr(s, '_e1km', None) and int(IP[11]) > 0:
                mf_ = s._e1km / (8.0 * int(IP[11]))
                rec0 = dict(mf_ratio=mf_ / med, n_mf=int((has & (mn < mf_)).sum()))
            else:
                rec0 = {}
            rec = dict(f=fi + 1, nn=nn, nact=int(has.sum()), rmin=float(rh.min()) if rh.numel() else None, n2=int((rh < 0.01).sum()), n4=int((rh < 0.0001).sum()), n6=int((rh < 1e-06).sum()), nfast=int(fast.sum()), fast_rmed=float(r[fast].median()) if bool(fast.any()) else None, fast_rmin=float(r[fast].min()) if bool(fast.any()) else None)
            rec.update(rec0)
            N = int(getattr(s, 'N', s.x.shape[0]))
            x = s.x[:N].float()
            ok = torch.isfinite(x).all(-1)
            cell = float(s.A[:, 0].norm())
            xs = x[ok][::7].double()
            q = torch.quantile(xs, torch.tensor([0.001, 0.999], dtype=torch.float64, device=xs.device), dim=0).float()
            d = torch.clamp(torch.maximum(q[0] - x, x - q[1]), min=0).norm(dim=-1)
            d = torch.where(ok, d, torch.full_like(d, float('inf')))
            far = d > 4 * cell
            rec.update(nfar=int(far.sum()), dmax=float(d[ok].max()) if bool(ok.any()) else None, nonfin=int((~ok).sum()))
            prev = ST['far']
            new = far & (~prev if prev is not None and prev.numel() == far.numel() else torch.ones_like(far))
            ST['far'] = far
            nnew = int(new.sum())
            rec['new'] = dict(n=nnew)
            if 0 < nnew:
                idx = torch.nonzero(new).squeeze(1)
                pb = s.Np.reshape(-1, 3)[:N][idx].long()
                w = _bw(s.fx.reshape(-1, 3)[:N][idx].float())
                rmin_p = torch.full((idx.numel(),), float('inf'), device=x.device)
                vn_p = torch.zeros(idx.numel(), device=x.device)
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            n = ((pb[:, 0] + i) * nb1 + (pb[:, 1] + j)) * nb2 + (pb[:, 2] + k)
                            okn = (n >= 0) & (n < nn)
                            n = n.clamp(0, nn - 1)
                            ww = w[:, 0, i] * w[:, 1, j] * w[:, 2, k]
                            use = okn & (ww > 0.001)
                            rr = torch.where(r[n] > 0, r[n], torch.full_like(r[n], float('inf')))
                            rmin_p = torch.where(use, torch.minimum(rmin_p, rr), rmin_p)
                            vn_p = torch.where(use, torch.maximum(vn_p, sp[n]), vn_p)
                fin = torch.isfinite(rmin_p)
                vp = s.v[:N][idx].float().norm(dim=-1)
                if bool(fin.any()):
                    rf = rmin_p[fin]
                    rec['new'].update(rmin_med=float(rf.median()), rmin_min=float(rf.min()), frac_lt1e4=float((rf < 0.0001).float().mean()), frac_lt1e2=float((rf < 0.01).float().mean()))
                rec['new'].update(vn_med=float(vn_p.median()), vp_med=float(vp[torch.isfinite(vp)].median()) if bool(torch.isfinite(vp).any()) else None, d_med=float(d[idx].median()))
            K = int(getattr(s, 'Kcur', 1))
            for (fl, gg) in (('cjl', 'cjg'), ('ufl', 'ufg')):
                (L_, G_) = (getattr(s, fl, None), getattr(s, gg, None))
                if L_ is not None and G_ is not None and (L_.numel() >= K) and (G_.numel() >= K) and (L_.numel() < 4096):
                    rec['step'] = dict(K=K, fail=[int(v) for v in L_[:K].cpu()], g2=[float(v) for v in G_[:K].cpu()])
                    break
            LOG.write(json.dumps(rec) + '\n')
            LOG.flush()

        def outer(fn):

            def w(s, t, h, *a, **kw):
                ST['dep'] += 1
                try:
                    res = fn(s, t, h, *a, **kw)
                finally:
                    ST['dep'] -= 1
                if ST['dep'] == 0:
                    try:
                        diag(s, int(round(t / h)))
                    except Exception:
                        import traceback
                        with open(os.path.join(out, 'e1_err.txt'), 'a') as f_:
                            f_.write(traceback.format_exc()[-2000:] + '\n')
                return res
            return w
        ART.an_rt_split_step = outer(ART.an_rt_split_step)
        AORT.ao_rt_split_step = outer(AORT.ao_rt_split_step)
        WRX.split_col = outer(WRX.split_col)
        print('[절 CM42 진단] 프레임마다 노드 질량 비·입자 이탈 -> e1.jsonl (결과 불변)', flush=True)
    sys.argv = [os.path.join(HERE, 'speed3_wrap.py')] + argv
    runpy.run_path(sys.argv[0], run_name='__main__')
WRAPPERS['escape1_wrap.py'] = w_escape1_wrap

def w_escape2_wrap():
    import json
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    CAND = {}
    while argv and argv[0].startswith('--e2'):
        CAND[argv[0][2:]] = argv[1]
        del argv[:2]
    if os.environ.get('AF_E2_DG', '0') == '1':
        pass
    sys.argv = [os.path.join(HERE, 'escape1_wrap.py')] + argv
    runpy.run_path(sys.argv[0], run_name='__main__')
WRAPPERS['escape2_wrap.py'] = w_escape2_wrap

def w_escape3_wrap():
    import json
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    CAND = {}
    while argv and argv[0].startswith('--e3'):
        CAND[argv[0][2:]] = argv[1]
        del argv[:2]
    W = int(float(CAND.get('e3w', 0)))
    G = int(float(CAND.get('e3g', 0)))
    OLD_G = False
    if W == 1:
        assert not OLD_G and G == 0, '--e3w 1 은 원래 G2P 기저 판 (G 없음)'
    if W or G:
        import stability as E3
        E3.escape3_install(w=W, g=W, newg=bool(G))
        print(f'[절 CM45] 장벽 표본점 기저 W̄ {W} (0 끔 1 원래 G2P 2 G 판), 새 실행기 G {G}', flush=True)
    if os.environ.get('AF_E3_DG', '0') == '1':
        pass
    sys.argv = [os.path.join(HERE, 'escape2_wrap.py')] + argv
    runpy.run_path(sys.argv[0], run_name='__main__')
WRAPPERS['escape3_wrap.py'] = w_escape3_wrap

def w_escape4_wrap():
    import json
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    CAND = {}
    while argv and argv[0].startswith('--e5'):
        CAND[argv[0][2:]] = argv[1]
        del argv[:2]
    R_ = int(float(CAND.get('e5r', 0)))
    A_ = int(float(CAND.get('e5a', 0)))
    W_ = int(float(argv[argv.index('--e3w') + 1])) if '--e3w' in argv else 0
    if R_ or A_:
        assert W_ == 1, '--e5r·--e5a 는 W̄ (--e3w 1) 위에서만'
        import stability as E5
        E5.escape4_install(r=R_, a=A_)
        print(f'[절 CM49] R {R_} · A {A_} (W̄ {W_})', flush=True)
    if os.environ.get('AF_E5_DG', '0') == '1':
        pass
    sys.argv = [os.path.join(HERE, 'escape3_wrap.py')] + argv
    runpy.run_path(sys.argv[0], run_name='__main__')
WRAPPERS['escape4_wrap.py'] = w_escape4_wrap

def w_ext2_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    C = dict(e='0', d='2', k='0', t='0', x='0', ck='0')
    while argv and argv[0].startswith(('--w8', '--x1')):
        C[argv[0][4:]] = argv[1]
        del argv[:2]
    ON = float(C['e']) != 0 or float(C['k']) != 0
    HEAD = argv[:argv.index('--out')]
    SP = any((a.startswith(('--s6', '--s7', '--s8', '--s9')) for a in HEAD if a.startswith('--')))
    S9 = any((a.startswith('--s9') for a in HEAD if a.startswith('--')))
    _orig = runpy.run_path
    _st = dict(done=False, nmf=0)

    def run_path(path, *a, **kw):
        b = os.path.basename(path)
        if b == 'mfix_wrap.py' and SP:
            _st['nmf'] += 1
            if _st['nmf'] == 1:
                path = os.path.join(HERE, 'speed9_wrap.py' if S9 else 'speed8_wrap.py')
                sys.argv[0] = path
        if ON and (not _st['done']) and (b in ('simulate.py', 'fem_ao_rt_sim.py')):
            _st['done'] = True
            if float(C['k']) != 0:
                import stability as W6
                W6.wolfh6_install(out, K=1)
            if float(C['e']) != 0:
                import stability as X2
                X2.ext2_install(out, D=int(float(C['d'])), T=int(float(C['t'])), X=int(float(C['x'])), ck=int(float(C['ck'])))
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'lightg4_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['ext2_wrap.py'] = w_ext2_wrap

def w_finterp_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    O = dict(fi='0', fidg='0')
    while argv and argv[0] in ('--fi', '--fidg'):
        O[argv[0][2:]] = argv[1]
        del argv[:2]
    (MODE, DG) = (int(float(O['fi'])), int(float(O['fidg'])))
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if (MODE or DG) and (not _done[0]) and (os.path.basename(path) in ('simulate.py', 'fem_ao_rt_sim.py')):
            _done[0] = True
            import quadrature as FI
            FI.finterp_install(out, MODE, DG)
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'wolfh11_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['finterp_wrap.py'] = w_finterp_wrap

def w_ghost6_wrap():
    import atexit
    import json
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    CAND = {}
    while argv and (argv[0].startswith('--g4') or argv[0].startswith('--g5') or argv[0].startswith('--g6')):
        CAND[argv[0][2:]] = argv[1]
        del argv[:2]
    if 'g4' in CAND:
        os.environ['AF_G4_GAMMA'] = str(float(CAND['g4']))
        os.environ['AF_G4_MASS'] = str(int(float(CAND.get('g4m', 1))))
        os.environ['AF_G4_THIN'] = str(int(float(CAND.get('g4t', 1))))
        os.environ['AF_G4_FIXX'] = str(int(float(CAND.get('g4x', 0))))
        os.environ['AF_G5_W'] = str(int(float(CAND.get('g5w', 0))))
        os.environ['AF_G5_WMAX'] = str(float(CAND.get('g5wmax', 3)))
        os.environ['AF_G6_CONS'] = str(int(float(CAND.get('g6c', 0))))
        import stability as G3
        G3.ghost6__env_init()
        import hexfem as ART
        import hexfem as V0
        G3.ghost6_install()
        print(f"[절 CM57] ghost penalty 6 켬 (보존 판 {G3.CONS}: 강성 인자 = {('u0 + y (기준 형상 = 첫 프레임 입자 위치, 입자 F 로 노드까지 아핀 연장)' if G3.CONS else 'y (걸음 변위, ghost5 그대로)')})", flush=True)
        print(f'[절 CM55] ghost penalty 5 켬 (γ̄ {G3.GAMMA:g}, 질량 {G3.MASS}, 면 한정 방식 {G3.THIN}, 구속 노드 면 빼기 {G3.FIXX}, 면 가중 방식 {G3.WMODE} w_max {G3.WMAX:g}, 칸별 커널)', flush=True)
        FED = {int(q) for q in CAND.get('g5ed', '').split(',') if q.strip()}
        FD = {int(q) for q in CAND.get('g4fd', '').split(',') if q.strip()}
        FDUMP = {int(q) for q in CAND.get('g4dump', '').split(',') if q.strip()}
        FD6 = {int(q) for q in CAND.get('g6fd', '').split(',') if q.strip()}
        EN6 = int(float(CAND.get('g6en', 0)))

        def dump(s, fi):
            raise NotImplementedError('not used by the paper settings')
        DG1 = os.environ.get('AF_G1_DG', '0') == '1'
        L1 = open(os.path.join(out, 'g1.jsonl'), 'w') if DG1 else None
        ST = dict(n=0, done=set())

        def err(tag, e_):
            raise NotImplementedError('not used by the paper settings')

        def torch_ok(s):
            raise NotImplementedError('not used by the paper settings')

        def wrap_step(cls):
            _os = cls.step

            def step(s, t, *a, **kw):
                d = getattr(s, '_g3dep', 0)
                if d == 0 and G3.CONS and (getattr(s, 'g6X0', None) is None):
                    s.g6X0 = s.x.detach().clone().contiguous()
                    print(f'[절 CM57] 기준 형상 X0 = 첫 프레임 입자 위치 {tuple(s.g6X0.shape)}', flush=True)
                s._g3dep = d + 1
                try:
                    r = _os(s, t, *a, **kw)
                finally:
                    s._g3dep = d
                if d != 0 or getattr(s, 'gpcnt', None) is None:
                    return r
                ST['n'] += 1
                h = float(getattr(s, 'h', 0.0) or 0.0)
                fi = int(round(t / h)) + 1 if h > 0 else ST['n']
                try:
                    if L1 is not None:
                        c = s.gpcnt.cpu().tolist()
                        L1.write(json.dumps(dict(f=fi, ncut=int(c[0]), nface=int(c[1]), nface_all=int(c[2]), nthick=int(c[3]), nc=int(s.IP[11].item()), k=float(s._gpk), kM=float(s._g3kM), it2=float(getattr(s, '_g3it2', 0.0)))) + '\n')
                        L1.flush()
                    if fi in FDUMP:
                        dump(s, fi)
                    if EN6:
                        G3.endiag(s, fi, out, 0.1 * float(s._gpkap) * float(s._gph))
                    if fi in FD6 and bool(torch_ok(s)):
                        rs6 = G3.check6(s, fi, out)
                        print(f'[절 CM57 검사] 프레임 {fi}: {json.dumps(rs6, ensure_ascii=False)[:1500]}', flush=True)
                    if fi in FED:
                        rs5 = G3.ediag(s, fi, out)
                        print(f'[절 CM55 진단] 프레임 {fi}: {json.dumps(rs5, ensure_ascii=False)[:1200]}', flush=True)
                    if fi in FD and fi not in ST['done'] and bool(torch_ok(s)):
                        ST['done'].add(fi)
                        rs = G3.ghost6_check(s, fi, out)
                        print(f'[절 CM53 검사] 프레임 {fi}: {json.dumps(rs, ensure_ascii=False)[:1500]}', flush=True)
                except Exception as e_:
                    err(f'프레임 {fi}', e_)
                return r
            cls.step = step
        for cls_ in (ART.an_rt_RTSolver, V0.an_rt_v0_RTSolver):
            wrap_step(cls_)

        @atexit.register
        def _info():
            try:
                json.dump(dict(cons=G3.CONS, gamma=G3.GAMMA, mass=G3.MASS, thin=G3.THIN, wmode=G3.WMODE, wmax=G3.WMAX, fixx=G3.FIXX, nk=G3.ghost6_ST['nk'], said=G3.ghost6_ST['said'], nmass=G3.ghost6_ST['nmass'], nstep=G3.ghost6_ST['nstep']), open(os.path.join(out, 'g6info.json'), 'w'), ensure_ascii=False)
            except Exception as e_:
                err('info', e_)
    sys.argv = [os.path.join(HERE, 'escape4_wrap.py')] + argv
    runpy.run_path(sys.argv[0], run_name='__main__')
WRAPPERS['ghost6_wrap.py'] = w_ghost6_wrap

def w_gqj_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    O = dict(gq_q1='0', gq_hg='0', gq_hgm='0', gq_vs='1', gq_ck='0')
    while argv and argv[0][2:] in O:
        O[argv[0][2:]] = argv[1]
        del argv[:2]
    if any((float(O[k]) != (1.0 if k == 'gq_vs' else 0.0) for k in O)):
        import quadrature as GQ
        GQ.gq1_install(out, q1=int(float(O['gq_q1'])), hg=float(O['gq_hg']), hgm=int(float(O['gq_hgm'])), vs=float(O['gq_vs']), ck=int(float(O['gq_ck'])))
    sys.argv = list(argv)
    runpy.run_path(sys.argv[0], run_name='__main__')
WRAPPERS['gqj_wrap.py'] = w_gqj_wrap

def w_hhtc2_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    HC = None
    TCM = None
    (RR, S4) = ([], [])
    while argv and (argv[0].startswith('--hc') or argv[0].startswith('--rr') or argv[0].startswith('--s4') or (argv[0] == '--tc')):
        (k_, v_) = (argv[0], argv[1])
        del argv[:2]
        if k_ == '--hc':
            HC = int(float(v_))
        elif k_ == '--tc':
            TCM = int(float(v_))
        elif k_.startswith('--rr'):
            RR += [k_, v_]
        else:
            S4 += [k_, v_]
    if TCM is not None:
        assert '--tb' in argv and (RR or not TCM & 8), '--tc 는 --tb 판, 비트 8 (앞 프레임 렌더 빼기) 은 렌더 판 (--rr) 위에서만'
        import realtime as TC
        TC.t2chain_install(TCM, out)
    assert not S4 or RR, '--s4* 는 --rr (렌더 판) 과 함께만 (speed4_wrap 을 끼우는 자리)'
    _orig = runpy.run_path
    _st = dict(rr=False, hc=False)

    def run_path(path, *a, **kw):
        b = os.path.basename(path)
        if RR and (not _st['rr']) and (b == 'escape4_wrap.py'):
            _st['rr'] = True
            p = os.path.join(HERE, 'rtrender1_wrap.py')
            sys.argv = [p] + RR + S4 + sys.argv[1:]
            print(f"[절 CM58] 렌더 판: ghost6_wrap -> rtrender1_wrap {' '.join(RR)} -> speed4_wrap {' '.join(S4)} -> escape4_wrap", flush=True)
            return _orig(p, *a, **kw)
        if HC is not None and (not _st['hc']) and (b in ('simulate.py', 'fem_ao_rt_sim.py')):
            _st['hc'] = True
            import stability as HCM
            HCM.hhtc2_install(HC, out)
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'ghost6_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['hhtc2_wrap.py'] = w_hhtc2_wrap

def w_lightg3_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    C = {}
    while argv and argv[0].startswith('--l3'):
        C[argv[0][4:]] = argv[1]
        del argv[:2]
    if '' in C:
        os.environ['AF_L3_G'] = str(float(C['']))
        os.environ['AF_L3_SC'] = str(int(float(C.get('sc', 0))))
        os.environ['AF_L3_FR'] = str(int(float(C.get('fr', 1))))
        os.environ['AF_L3_R0'] = str(int(float(C.get('r0', 0))))
        os.environ['AF_L3_DG'] = str(int(float(C.get('dg', 0))))
        os.environ['AF_L3_CK'] = str(int(float(C.get('ck', 0))))
        os.environ['AF_L3_FD'] = C.get('fd', '')
    _orig = runpy.run_path
    _st = dict(on=False)

    def run_path(path, *a, **kw):
        b = os.path.basename(path)
        if '' in C and (not _st['on']) and (b in ('simulate.py', 'fem_ao_rt_sim.py')):
            _st['on'] = True
            import stability as L3
            L3.lightg3__env_init()
            L3.lightg3_install(out)
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'wolfh5_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['lightg3_wrap.py'] = w_lightg3_wrap

def w_lightg4_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    C = {}
    while argv and argv[0].startswith('--l4'):
        C[argv[0][4:]] = argv[1]
        del argv[:2]
    ON = int(float(C.get('', 0))) == 1
    _orig = runpy.run_path
    _st = dict(on=False)

    def run_path(path, *a, **kw):
        b = os.path.basename(path)
        if ON and (not _st['on']) and (b in ('simulate.py', 'fem_ao_rt_sim.py')):
            _st['on'] = True
            import stability as L4
            L4.lightg4_install(out)
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'lightg3_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['lightg4_wrap.py'] = w_lightg4_wrap

def w_mfix_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    C = {}
    while argv and argv[0].startswith('--mf'):
        C[argv[0][4:]] = argv[1]
        del argv[:2]
    if C:
        os.environ['AF_MF1'] = str(int(float(C.get('1', 0))))
        os.environ['AF_MF2'] = str(int(float(C.get('2', 0))))
    _orig = runpy.run_path
    _st = dict(on=False)

    def run_path(path, *a, **kw):
        b = os.path.basename(path)
        if C and (not _st['on']) and (b in ('simulate.py', 'fem_ao_rt_sim.py')):
            _st['on'] = True
            import stability as MF
            MF.mfix__env_init()
            MF.mfix_install(out)
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'wmfloor_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['mfix_wrap.py'] = w_mfix_wrap

def w_nlwhy_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 16000))
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if not _done[0] and os.path.basename(path) == 'simulate.py':
            _done[0] = True
            import stability as NL
            NL.nlwhy_install(out)
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    assert argv and argv[0].endswith('.py'), 'nlwhy_wrap: 다음 감싸개 .py 가 첫 인자'
    sys.argv = argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['nlwhy_wrap.py'] = w_nlwhy_wrap

def w_q27_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 18000))
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    O = dict(q27='0', q27cap='0', q27dg='0', q27jit='0', q1pt='0', q27ph='all')
    SEEN = False
    while argv and argv[0] in ('--q27', '--q27cap', '--q27dg', '--q27jit', '--q1pt', '--q27ph'):
        O[argv[0][2:]] = argv[1]
        del argv[:2]
        SEEN = True
    (MODE, CAP, DG) = (int(float(O['q27'])), int(float(O['q27cap'])), int(float(O['q27dg'])))
    if int(float(O['q1pt'])):
        MODE = 2
    _orig = runpy.run_path
    _done = [False]
    print('[절 CM179 q27_wrap] 단계 %s 실행 (SEEN %s)' % (O['q27ph'], SEEN), flush=True)
    if SEEN and O['q27ph'] == 'src':
        import quadrature as Q27S
        Q27S.q27_install(out, MODE, CAP, DG, 'simulate.py', int(float(O['q27jit'])), 'src')
        _done = [True]

    def run_path(path, *a, **kw):
        if os.path.basename(path) in ('simulate.py', 'fem_ao_rt_sim.py'):
            print('[절 CM179 q27_wrap] 단계 %s 훅이 실행기 경로를 봄 (done %s)' % (O['q27ph'], _done[0]), flush=True)
        if SEEN and (not _done[0]) and (os.path.basename(path) in ('simulate.py', 'fem_ao_rt_sim.py')):
            _done[0] = True
            import quadrature as Q27
            Q27.q27_install(out, MODE, CAP, DG, os.path.basename(path), int(float(O['q27jit'])), O['q27ph'])
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = list(argv)
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['q27_wrap.py'] = w_q27_wrap

def w_q27b_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 18000))
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    O = dict(q27n='3', q27='0', q27cap='0', q27dg='0', q27jit='0', q1pt='0', q27ph='all')
    SEEN = False
    while argv and argv[0] in ('--q27n', '--q27', '--q27cap', '--q27dg', '--q27jit', '--q1pt', '--q27ph'):
        O[argv[0][2:]] = argv[1]
        del argv[:2]
        SEEN = True
    (MODE, CAP, DG) = (int(float(O['q27'])), int(float(O['q27cap'])), int(float(O['q27dg'])))
    if int(float(O['q1pt'])):
        pass
    _orig = runpy.run_path
    _done = [False]
    print('[절 CM183 q27b_wrap] 단계 %s 실행 (SEEN %s)' % (O['q27ph'], SEEN), flush=True)
    if SEEN and O['q27ph'] == 'src':
        import quadrature as Q27S
        Q27S.q27b_install(out, MODE, CAP, DG, 'simulate.py', int(float(O['q27jit'])), 'src', int(float(O['q27n'])))
        _done = [True]

    def run_path(path, *a, **kw):
        if os.path.basename(path) in ('simulate.py', 'fem_ao_rt_sim.py'):
            print('[절 CM183 q27b_wrap] 단계 %s 훅이 실행기 경로를 봄 (done %s)' % (O['q27ph'], _done[0]), flush=True)
        if SEEN and (not _done[0]) and (os.path.basename(path) in ('simulate.py', 'fem_ao_rt_sim.py')):
            _done[0] = True
            import quadrature as Q27
            Q27.q27b_install(out, MODE, CAP, DG, os.path.basename(path), int(float(O['q27jit'])), O['q27ph'], int(float(O['q27n'])))
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = list(argv)
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['q27b_wrap.py'] = w_q27b_wrap

def w_rotbc_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 15000))
    import stability as _STB
    KERNEL = _STB.cu('w_rotbc_wrap_KERNEL')

    def install():
        import numpy as np
        import torch
        import hexfem as ART
        anchor = '// 선탐색 (융합): 후보 c = ¼'
        assert anchor in ART.an_rt_SRC, '[rotbc] 커널 삽입 위치를 못 찾음'
        ART.an_rt_SRC = ART.an_rt_SRC.replace(anchor, KERNEL + anchor, 1)
        (orig_init, orig_k, orig_set, orig_alloc) = (ART.an_rt_RTSolver.__init__, ART.an_rt_RTSolver.k, ART.an_rt_RTSolver.set_bc, ART.an_rt_RTSolver.alloc)

        def init(s, *a, **kw):
            g = None
            f = sys._getframe(1)
            while f is not None:
                if 'cfg' in f.f_globals and 'h' in f.f_globals and (f.f_globals.get('__name__') == '__main__'):
                    g = f.f_globals
                    break
                f = f.f_back
            assert g is not None, '[rotbc] 실행기 전역 (cfg) 을 못 찾음'
            cfg = g['cfg']
            bcs = cfg.get('boundary_conditions') or []
            rot = [b for b in bcs if b.get('type') == 'enforce_particle_velocity_rotation']
            mat_path = kw.get('mat') is not None or kw.get('trel') is not None or kw.get('pvb') is not None
            tr = [] if mat_path else [b for b in bcs if b.get('type') == 'enforce_particle_translation']
            lo = g.get('LOFF')
            assert not ((rot or tr) and lo is not None and (float(lo.abs().max()) > 0)), '[rotbc] 띄움 (--lift) 과 같이는 안 됨'
            bc = list(kw.get('bc') or [])
            s._trb = [([float(q) for q in b['velocity']], float(b.get('start_time', 0.0)), float(b.get('end_time', 1000.0))) for b in tr]
            groups = {}
            for b in rot:
                nv = np.asarray(b['normal'], dtype=np.float64)
                nv = nv / np.linalg.norm(nv)
                (hh, rr) = (float(q) for q in b['half_height_and_radius'])
                key = (tuple((float(q) for q in b['point'])), tuple(nv.tolist()), hh, rr)
                groups.setdefault(key, []).append((float(b['rotation_scale']), float(b['translation_scale']), float(b.get('start_time', 0.0)), float(b.get('end_time', 1000.0))))
            if (groups or tr) and (not bc):
                bc.append(([1000000000.0] * 3, [1e-09] * 3, [0.0] * 3, 1000000000.0, 2000000000.0))
            kw['bc'] = bc or None
            s._rbc_allow = bool(groups)
            if tr:
                x0 = (a[0] if a else kw['x']).to('cuda').double()
                pid = torch.full((x0.shape[0],), -1, device='cuda', dtype=torch.int32)
                for (ib, b) in enumerate(tr):
                    msk = ((x0 - torch.tensor(b['point'], device='cuda', dtype=torch.float64)).abs() < torch.tensor(b['size'], device='cuda', dtype=torch.float64)).all(-1)
                    pid[msk] = ib
                s.trpid = pid.contiguous()
                s.TP = torch.zeros(len(tr) * 4, device='cuda', dtype=torch.float32)
                print(f'[rotbc] translation 입자 선택 (PG 정의): 구속 {[int((pid == ib).sum()) for ib in range(len(tr))]} / {x0.shape[0]} 입자 -> 노드 (구속 질량 > 절반)', flush=True)
                del x0
            if tr:
                s._trmref = float((a[3] if len(a) > 3 else kw['m']).max())
                s._trvn = int(any((any((q != 0.0 for q in v_)) for (v_, _, _) in s._trb)))
            orig_init(s, *a, **kw)
            s._rbc = list(groups.items())
            s.RP = torch.zeros(max(1, len(s._rbc)) * 12, device='cuda', dtype=torch.float32)
            print(f'[rotbc] 회전 구동기 {len(rot)} 개 (영역 {len(s._rbc)} 곳으로 합침), 탄성 판 translation 상자 {len(tr)} 개', flush=True)

        def k(s, name, n, args, b=128):
            if name == 'rt_bc_box' and getattr(s, '_trb', None) and (int(args[3]) == 0):
                assert s.PA.numel() >= s.capN * 4, '[rotbc] PA 용량'
                orig_k(s, 'rt_trp_zero', n, (args[0], ART.an_rt_P(s.PA)), b=256)
                orig_k(s, 'rt_trp_acc', s.N, (np.int32(s.N), args[0], ART.an_rt_P(s.Ai), ART.an_rt_P(s.x), ART.an_rt_P(s.m), ART.an_rt_P(s.perm), ART.an_rt_P(s.trpid), ART.an_rt_P(s.TP), np.float32(s._trmref), np.int32(s._trvn), ART.an_rt_P(s.PA)), b=128)
                orig_k(s, 'rt_trp_fix', n, (args[0], ART.an_rt_P(s.mn), ART.an_rt_P(s.PA), args[4], args[5]), b=256)
            if name == 'rt_bc_box' and getattr(s, '_rbc', None) and (int(args[3]) == len(s.bc) - 1):
                for jj in range(len(s._rbc)):
                    orig_k(s, 'rt_bc_rot', n, (args[0], args[1], ART.an_rt_P(s.U), ART.an_rt_P(s.RP), np.int32(jj), args[4], args[5]), b=256)
            return orig_k(s, name, n, args, b)

        def set_bc(s, t):
            orig_set(s, t)
            if getattr(s, '_trb', None):
                v = []
                for (vel, t0, t1) in s._trb:
                    v += list(vel) + [1.0 if t0 <= t < t1 else 0.0]
                s.TP.copy_(torch.tensor(v, dtype=torch.float32))
            if getattr(s, '_rbc', None):
                v = []
                for ((pt, nv, hh, rr), wins) in s._rbc:
                    rs = ts = fs = 0.0
                    for (r_, t_, t0, t1) in wins:
                        f = max(0.0, min(t + s.h, t1) - max(t, t0)) / s.h
                        rs += r_ * f
                        ts += t_ * f
                        fs += f
                    v += list(pt) + list(nv) + [hh, rr, rs, ts, 0.0, 1.0 if fs > 0 else 0.0]
                s.RP.copy_(torch.tensor(v, dtype=torch.float32))

        def alloc(s, need):
            orig_alloc(s, need)
            if getattr(s, '_trb', None):
                s.PA = torch.zeros(s.capN * 4, device='cuda', dtype=torch.float32)
        (ART.an_rt_RTSolver.__init__, ART.an_rt_RTSolver.k, ART.an_rt_RTSolver.set_bc, ART.an_rt_RTSolver.alloc) = (init, k, set_bc, alloc)
    if True:
        argv = sys.argv[1:]
        ON = 0
        while argv and argv[0] == '--rot_bc':
            ON = int(float(argv[1]))
            del argv[:2]
        _orig = runpy.run_path
        _done = [False]

        def run_path(path, *a, **kw):
            if ON and (not _done[0]) and (os.path.basename(path) == 'simulate.py'):
                _done[0] = True
                install()
            return _orig(path, *a, **kw)
        runpy.run_path = run_path
        sys.argv = [os.path.abspath(argv[0])] + argv[1:]
        _orig(sys.argv[0], run_name='__main__')
WRAPPERS['rotbc_wrap.py'] = w_rotbc_wrap

def w_rtrender1_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    O = dict(rr='pg', rr_shape='lego', rr_traj='', rr_cfg='', rr_model='', rr_gf='', rr_fast='0', rr_chk='')
    while argv and argv[0].startswith('--rr'):
        O[argv[0][2:]] = argv[1]
        del argv[:2]
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if not _done[0] and os.path.basename(path) in ('simulate.py', 'fem_ao_rt_sim.py'):
            _done[0] = True
            import gsrender as RR
            RR.install(O['rr'], O['rr_shape'], O['rr_traj'], O['rr_cfg'], O['rr_model'], O['rr_gf'], int(O['rr_fast']), O['rr_chk'], out)
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'speed4_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['rtrender1_wrap.py'] = w_rtrender1_wrap

def w_speed12_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    O = dict(s12z='0', s12dg='0')
    while argv and argv[0].startswith('--s12'):
        O[argv[0][2:]] = argv[1]
        del argv[:2]
    ON = any((float(v) != 0 for v in O.values()))
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if ON and (not _done[0]) and (os.path.basename(path) == 'fem_ao_rt_sim.py'):
            _done[0] = True
            import fem_speed12 as S12
            S12.install(out, z=int(float(O['s12z'])), dg=int(float(O['s12dg'])))
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'ext2_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['speed12_wrap.py'] = w_speed12_wrap

def w_speed3_wrap():
    import os
    import runpy
    import sys
    import hexfem as ART
    import realtime as S3
    (opt, i) = ({}, 1)
    while i < len(sys.argv) and (not sys.argv[i].endswith('.py')):
        opt[sys.argv[i].lstrip('-')] = sys.argv[i + 1]
        i += 2
    sim = sys.argv[i]
    argv = sys.argv[i:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    if os.environ.get('AF_S3_UL', '0') == '1':
        pass
    if 'tb' in opt:
        S3.tb_install(float(opt['tb']), float(opt.get('tb_m', 0.5)), out, int(opt.get('tb_w', 1)))
        print(f"[절 CM39 후보 T] 시간 예산 {opt['tb']} ms (여유 {opt.get('tb_m', 0.5)} ms, 자리 {opt.get('tb_w', 1)})", flush=True)
    if opt.get('inv', '0') != '0':
        pass
    if opt.get('w3', '0') != '0':
        pass
    if opt.get('w1', '0') != '0':
        pass
    if opt:
        S3.frame_hook_install(out)
    sys.argv = argv
    runpy.run_path(sim, run_name='__main__')
WRAPPERS['speed3_wrap.py'] = w_speed3_wrap

def w_speed4_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    OPT = {}
    while argv and argv[0].startswith('--s4'):
        OPT[argv[0][2:]] = int(float(argv[1]))
        del argv[:2]
    OPT = {k: v for (k, v) in OPT.items() if v}
    if OPT:
        import realtime as S4
        if OPT.get('s4p'):
            pass
        if OPT.get('s4w'):
            S4.ws4_install(OPT['s4w'])
        if OPT.get('s4y'):
            S4.yl4_install(OPT['s4y'])
        if OPT.get('s4g'):
            pass
        if OPT.get('s4x'):
            S4.x1_install(OPT['s4x'], float(OPT.get('s4e', 0)))
        S4.speed4_frame_install(out)
        print(f'[절 CM50] speed4 {OPT}', flush=True)
    sys.argv = [os.path.join(HERE, 'escape4_wrap.py')] + argv
    runpy.run_path(sys.argv[0], run_name='__main__')
WRAPPERS['speed4_wrap.py'] = w_speed4_wrap

def w_speed7_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    O7 = dict(s7kp='', s7nm='0', s7z='0', s7src='0', s7ovf='0', s7p='0', s7pm='1e-4', s7y='0', s7ym='4', s7pk='')
    OPT = dict(s6w='0', s6o='0', s6l='0', s6prof='', s6b='0', s6r='0', s6rw='0')
    while argv and (argv[0].startswith('--s7') or argv[0].startswith('--s6')):
        k_ = argv[0][2:]
        (O7 if k_.startswith('s7') else OPT)[k_] = argv[1]
        del argv[:2]
    ON7 = any((v not in ('', '0') for (k, v) in O7.items() if k not in ('s7pm', 's7ym')))
    ON = ON7 or any((v not in ('', '0') for v in OPT.values()))
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if ON and (not _done[0]) and (os.path.basename(path) in ('simulate.py', 'fem_ao_rt_sim.py')):
            _done[0] = True
            import realtime as S6
            S6.speed6_install(out, o=int(float(OPT['s6o'])), lean=int(float(OPT['s6l'])), prof=OPT['s6prof'] or None, b=int(float(OPT['s6b'])), r=int(float(OPT['s6r'])), rw=int(float(OPT['s6rw'])))
            if ON7:
                import realtime as S7
                S7.speed7_install(out, kp=O7['s7kp'] or None, nm=int(float(O7['s7nm'])), z=int(float(O7['s7z'])), src=int(float(O7['s7src'])), ovf=int(float(O7['s7ovf'])), p=int(float(O7['s7p'])), pm=float(O7['s7pm']), y=int(float(O7['s7y'])), ym=int(float(O7['s7ym'])), pk=O7['s7pk'] or None)
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'mfix_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['speed7_wrap.py'] = w_speed7_wrap

def w_speed8_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    O8 = dict(s8w='0', s8g='0')
    while argv and argv[0].startswith('--s8'):
        O8[argv[0][2:]] = argv[1]
        del argv[:2]
    ON = any((float(v) != 0 for v in O8.values()))
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if ON and (not _done[0]) and (os.path.basename(path) in ('simulate.py', 'fem_ao_rt_sim.py')):
            _done[0] = True
            import realtime as S8
            S8.speed8_install(out, w=int(float(O8['s8w'])), g=float(O8['s8g']))
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'speed7_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['speed8_wrap.py'] = w_speed8_wrap

def w_speed9_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    O9 = dict(s9dg='0', s9c='0', s9y='0', s9e='0', s9r='0', s9p='0')
    while argv and argv[0].startswith('--s9'):
        O9[argv[0][2:]] = argv[1]
        del argv[:2]
    ON = any((float(v) != 0 for v in O9.values()))
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if ON and (not _done[0]) and (os.path.basename(path) in ('simulate.py', 'fem_ao_rt_sim.py')):
            _done[0] = True
            import realtime as S9
            S9.speed9_install(out, dg=int(float(O9['s9dg'])), cm=int(float(O9['s9c'])), yl=int(float(O9['s9y'])), e3=int(float(O9['s9e'])), r=int(float(O9['s9r'])), ew=float(O9['s9p']))
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'speed8_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['speed9_wrap.py'] = w_speed9_wrap

def w_ssfix_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    C = dict(xgT='0', alog='0', best='0', tol='0', tolp='0', kex='0', split='0', bigT='1.0', wrx='1', dumpfail='', det='1', bec='0', tsum='0', kmax='256', wrxr='0', fpic='0', fpik='0', cgk='0', cgmax='64', cgm='1', fpg='0', detj='0')
    while argv and argv[0].startswith('--ssf_'):
        C[argv[0][6:]] = argv[1]
        del argv[:2]
    _orig = runpy.run_path
    _done = [False]
    _BEC = {}

    def _install():
        import hexfem as ART
        if float(C['xgT']) != 0:
            import hexfem as C2
            old = 'big = res && cjg[k] > 1.f;'
            assert C2.C2_GATE_SRC.count(old) == 1
            C2.C2_GATE_SRC = C2.C2_GATE_SRC.replace(old, 'big = res && cjg[k] > %.9ef;' % float(C['xgT']))
            print(f"[substep_fix] --xit_from 문 문턱 ‖g‖²/F² > {float(C['xgT']):g}", flush=True)
        oi = ART.an_rt_RTSolver.__init__

        def init(s, *a, **kw):
            oi(s, *a, **kw)
            K = max(int(s.K), 1)
            if float(C['tol']) != 0 and s.tolN >= 0:
                s.tolN = s.tolN / K ** float(C['tol'])
            if float(C['tolp']) != 0 and s.tolP >= 0:
                s.tolP = s.tolP / K ** float(C['tolp'])
            if float(C['kex']) != 0 and getattr(s, 'floor', None):
                s.floor['KEX'] = s.floor['KEX'] * float(C['kex'])
            print(f"[substep_fix] KEX {getattr(s, 'floor', None) and s.floor['KEX']} tolN {s.tolN:.4g} tolP {s.tolP:.4g} (K {K}, ssf_tol {C['tol']}, ssf_tolp {C['tolp']})", flush=True)
        ART.an_rt_RTSolver.__init__ = init
        if float(C['cgk']) != 0:
            oinit_c = ART.an_rt_RTSolver.__init__

            def init_c(s, *a, **kw):
                oinit_c(s, *a, **kw)
                s._cg0 = int(s.cg)
                if int(float(C['cgm'])) == 0 or hasattr(type(s), 'mat_g2p'):
                    N_ = max(int(s.K), 1)
                    s.cg = int(min(int(float(C['cgmax'])), max(s._cg0, round(s._cg0 * N_ ** float(C['cgk'])))))
                print(f"[d192 cgk] N {s.K} PCG 상한 {s._cg0} -> {s.cg} (P {C['cgk']}, 재질 판만 {C['cgm']}, 재질 판 {hasattr(type(s), 'mat_g2p')})", flush=True)
            ART.an_rt_RTSolver.__init__ = init_c
        if float(C['bec']) != 0:
            okc = ART.an_rt_RTSolver.k
            BS = dict(k=0, on=float(C['bec']) != 2)

            def kc(s, name, n, args, b=128):
                r = okc(s, name, n, args, b)
                if name == 'rt_reset':
                    BS['k'] = 0
                if name == 'cj_end':
                    if BS['on']:
                        s.cjl[BS['k']:BS['k'] + 1].fill_(1)
                    BS['k'] += 1
                return r
            ART.an_rt_RTSolver.k = kc
            _BEC['bs'] = BS
        D = int(float(C['split']))
        if D > 0 or C['dumpfail'] or float(C['fpic']) != 0 or (float(C['fpg']) != 0):
            import torch
            ostep = ART.an_rt_RTSolver.step
            oinit2 = ART.an_rt_RTSolver.__init__

            def init2(s, *a, **kw):
                oinit2(s, *a, **kw)
                s.kcap = max(int(getattr(s, 'kcap', 0) or 0), min(int(s.K) << D, max(int(s.K), int(float(C['kmax'])))))
                s.alloc(None)
            ART.an_rt_RTSolver.__init__ = init2
            T = float(C['bigT'])
            ST = dict(retry=0, frames=0, hist={}, fpic=0)

            def _fpi_rescue(s, J0=None):
                import numpy as np
                if not getattr(s, 'matk', 0) or getattr(s, 'fpi', False) or (not hasattr(s, '_src0')):
                    return
                P = ART.an_rt_P
                if getattr(s, '_fpm', None) is None:
                    s._fpm = ART.cp.RawModule(code=ART.build_src(s._src0, s.adapt, s.goct, s.floor, fk=getattr(s, 'fk', False), hspec=getattr(s, 'hspec', False), ncomp=getattr(s, 'ncomp', False), gqm=getattr(s, 'gqm', False), fpi=True, cj=getattr(s, 'cjj', False), cue=bool(getattr(s, 'cue', 0)), uel=getattr(s, 'uel', 0), ufw=getattr(s, 'ufw', False), ls0=getattr(s, 'ls0', False), c2=getattr(s, 'c2', None)), options=('--std=c++14',))
                M = s._fpm
                (i32, f) = (np.int32, np.float32)
                capQ = s.capQc * 8
                N = s.N
                IP = s.IP
                K = int(s.K)
                Fe = torch.zeros(capQ, 28, device=s.dev, dtype=torch.float32)
                Uk = s.Ust[K - 1].contiguous()
                (mod0, fn0) = (s.mod, s.fn)
                _sv = {nm: getattr(s, nm, None) for nm in ('Fe', 'fiJ0', 'fiJX', 'fiJp0', 'fiD')}
                _Z = lambda n_: torch.zeros(n_, device=s.dev, dtype=torch.float32)
                (s.Fe, s.fiJ0, s.fiJX, s.fiJp0, s.fiD) = (Fe, _Z(capQ), _Z(capQ), _Z(N), _Z(N * 8))
                (s.mod, s.fn) = (M, {})
                try:
                    if s.aq:
                        s.k('aq_commit', capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(Uk), i32(s._mat), P(s.prm), P(s.Sq), f(s.dt)))
                    else:
                        s.k('ao_commit', capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(Uk), f(s.cdp)))
                    s.k('fpi_q', capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(Uk), P(Fe)))
                    if J0 is None:
                        (tF, tys, tlj, tal) = (s.F, getattr(s, 'ysp', None), getattr(s, 'ljp', None), getattr(s, 'alp', None))
                    else:
                        tF = s.F.clone()
                        (tys, tlj, tal) = (s.ysp.clone(), s.ljp.clone(), s.alp.clone()) if s.aq else (None, None, None)
                    s.k('fpi_p', N, (i32(N), P(IP), P(s.fx), P(s.pcd), P(s.cidx), P(Fe), i32(0), i32(0), P(tF), P(s.Sq if s.aq else None), P(tys if s.aq else None), P(tlj if s.aq else None), P(tal if s.aq else None)))
                finally:
                    (s.mod, s.fn) = (mod0, fn0)
                    for (nm, v_) in _sv.items():
                        setattr(s, nm, v_)
                torch.cuda.synchronize()
                if J0 is not None:
                    J = torch.linalg.det(s.F[:N].reshape(-1, 3, 3).double())
                    m = ~((J > J0) & (J < 1.0 / J0))
                    nm = int(m.sum())
                    if nm:
                        s.F[:N].view(-1, 9)[m] = tF[:N].view(-1, 9)[m]
                        if s.aq:
                            for (d_, t_) in ((s.ysp, tys), (s.ljp, tlj), (s.alp, tal)):
                                d_[:N][m] = t_[:N][m]
                        ST['fpgn'] = ST.get('fpgn', 0) + nm
                        ST['fpgf'] = ST.get('fpgf', 0) + 1
                        print(f"[d192 fpg] 프레임 {ST['frames']} K {K}: J 범위 밖 입자 {nm} -> 가우스 점 보간 (J 최소 {float(J.min()):.3g} 최대 {float(J.max()):.3g})", flush=True)
                    return
                ST['fpic'] += 1
                print(f"[d192 fpic] 프레임 {ST['frames']} K {K}: 입자 F·상태 = 가우스 점 보간 (재질 판)", flush=True)

            def collapsed(s, K):
                if not bool(torch.isfinite(s.x).all()):
                    return True
                if float(C['det']) != 0 and float(torch.linalg.det(s.F.reshape(-1, 3, 3)).min()) <= float(C['detj']):
                    return True
                fl = s.cjl[:K] > 0
                big = ~(s.cjg[:K] <= T)
                stall = s.cjs[:K] <= 0.0
                return bool((fl & big).any())
            AL = float(C['alog']) != 0
            BE = float(C['best']) != 0

            def meas(s, K, oom):
                if oom:
                    return dict(K=K, oom=1, bad=(2, 0, 0.0))
                if not bool(torch.isfinite(s.x).all()):
                    return dict(K=K, fin=0, bad=(2, 0, 0.0))
                nd = int((torch.linalg.det(s.F.reshape(-1, 3, 3)) <= 0.0).sum())
                fl = s.cjl[:K] > 0
                g = s.cjg[:K]
                gm = float(torch.where(fl, g, torch.zeros_like(g)).max()) if K > 0 else 0.0
                ext = [round(float(q), 3) for q in s.x.max(0).values - s.x.min(0).values]
                return dict(K=K, ndet=nd, g2max=gm, nbig=int((fl & ~(g <= T)).sum()), ext=ext, ms=round(float(getattr(s, 'last_ms', 0.0) or 0.0), 1), bad=(1 if nd > 0 else 0, nd, gm))

            def step(s, t):
                ST['frames'] += 1
                bk = s._bk()
                K0 = int(s.K)
                h = float(s.h)
                cm0 = s.cap_mul
                w0 = getattr(s, '_wrxv', 0) if float(C['wrxr']) != 0 else 0
                oom = False
                try:
                    redo = ostep(s, t)
                except torch.OutOfMemoryError:
                    oom = True
                    redo = False
                dp = 0
                tw = 0.0
                if float(C['bec']) == 2 and _BEC.get('bs'):
                    _BEC['bs']['on'] = False
                TR = []
                first = ST['frames'] == 1
                while dp < D and K0 << dp + 1 <= int(float(C['kmax'])) and (oom or (collapsed(s, int(s.K)) and (not first)) or (first and (not bool(torch.isfinite(s.x).all())))):
                    if (AL or BE) and dp == 0:
                        TR.append((meas(s, int(s.K), oom), None if oom else s._bk() if BE else None))
                    tw += float(getattr(s, 'last_ms', 0.0) or 0.0) if not oom else 0.0
                    dp += 1
                    ST['retry'] += 1
                    if oom:
                        torch.cuda.empty_cache()
                    s._rs(bk)
                    if oom or s.cap_mul != cm0:
                        s.cap_mul = cm0
                        s.alloc(None)
                        s.graphs.clear()
                        s.graph = None
                    oom = False
                    if hasattr(s, 'wrx') and float(C['wrx']) != 0:
                        import hexfem as WRX
                        WRX.on(s, 1)
                    ART.set_K_an(s, K0 << dp, h)
                    if float(C['bec']) == 2 and _BEC.get('bs'):
                        _BEC['bs']['on'] = True
                    try:
                        redo = ostep(s, t) or redo
                    except (torch.OutOfMemoryError, RuntimeError) as ex_:
                        if not isinstance(ex_, torch.OutOfMemoryError) and 'out of memory' not in str(ex_):
                            raise
                        oom = True
                    if AL or BE:
                        TR.append((meas(s, int(s.K), oom), None if oom else s._bk() if BE else None))
                fin_col = dp > 0 and (oom or collapsed(s, int(s.K))) if BE or AL else None
                if BE and fin_col:
                    cand = [(m['bad'], i) for (i, (m, sn)) in enumerate(TR) if sn is not None]
                    if cand:
                        bi = min(cand)[1]
                        if bi != len(TR) - 1:
                            s._rs(TR[bi][1])
                            oom = False
                            print(f"[substep_fix] 프레임 t={t:.4f} 끝까지 무너짐 -> 시도 {bi} (K {TR[bi][0]['K']}) 끝 상태를 받음", flush=True)
                if AL and dp:
                    print(f'[substep_fix] 시도 t={t:.4f} ' + ' | '.join((str({k: v for (k, v) in m.items() if k != 'bad'}) for (m, _) in TR)), flush=True)
                del TR
                if C['dumpfail'] and dp == D and (oom or collapsed(s, int(s.K))) and (not ST.get('dumped')):
                    ST['dumped'] = True
                    torch.save(dict(bk=[q.cpu() for q in bk], t=float(t), K0=K0, fi=ST['frames'] - 1), C['dumpfail'])
                    print(f"[substep_fix] 끝까지 무너진 프레임 시작 상태 저장 {C['dumpfail']} (프레임 {ST['frames'] - 1})", flush=True)
                ST['hist'][dp] = ST['hist'].get(dp, 0) + 1
                fpm = int(float(C['fpic']))
                if fpm and oom is False and (dp > 0 if fpm == 2 else collapsed(s, int(s.K))) and (int(s.K) >= int(float(C['fpik']))):
                    _fpi_rescue(s)
                if float(C['fpg']) > 0 and oom is False and bool(torch.isfinite(s.x).all()):
                    J_ = torch.linalg.det(s.F[:s.N].reshape(-1, 3, 3).double())
                    if not bool(((J_ > float(C['fpg'])) & (J_ < 1.0 / float(C['fpg']))).all()):
                        _fpi_rescue(s, float(C['fpg']))
                if dp and float(C['tsum']) != 0:
                    s.last_ms = float(s.last_ms) + tw
                if dp:
                    print(f'[substep_fix] 프레임 t={t:.4f} 무너짐 -> K {K0} 에서 {K0 << dp} 로 다시 풂 (깊이 {dp}; 끝 판정 무너짐 {oom or collapsed(s, int(s.K))})', flush=True)
                    ART.set_K_an(s, K0, h)
                    if hasattr(s, 'wrx') and float(C['wrx']) != 0:
                        import hexfem as WRX
                        WRX.on(s, w0)
                    if oom:
                        s.x.fill_(float('nan'))
                return redo
            ART.an_rt_RTSolver.step = step
            import atexit
            atexit.register(lambda : print(f"[substep_fix] 프레임 {ST['frames']} 다시 풂 {ST['retry']} 깊이별 {ST['hist']}" + (f" fpic {ST['fpic']}" if int(float(C['fpic'])) else '') + (f" fpg 프레임 {ST.get('fpgf', 0)} 입자 {ST.get('fpgn', 0)}" if float(C['fpg']) else ''), flush=True))

    def run_path(path, *a, **kw):
        if not _done[0] and os.path.basename(path) == 'simulate.py':
            _done[0] = True
            if float(C['tol']) != 0 or float(C['tolp']) != 0 or float(C['kex']) != 0 or (float(C['split']) != 0) or C['dumpfail'] or (float(C['bec']) != 0) or (float(C['xgT']) != 0) or (float(C['fpic']) != 0) or (float(C['cgk']) != 0) or (float(C['fpg']) != 0):
                _install()
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.abspath(argv[0])] + argv[1:]
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['ssfix_wrap.py'] = w_ssfix_wrap

def w_stabhex_wrap():
    import os
    import runpy
    import signal
    import sys
    import time
    HERE = _HERE
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 14000))
    argv = sys.argv[1:]
    OPT = dict(npz='', stats='1', direct='0')
    while argv and argv[0].startswith('--sh_'):
        OPT[argv[0][5:]] = argv[1]
        del argv[:2]
    ON = OPT['stats'] not in ('0', '') and OPT['npz'] != ''

    def _arg(name, default=None):
        return argv[len(argv) - 1 - argv[::-1].index(name) + 1] if name in argv else default
    if '--rr_shape' in argv:
        import gsrender as _RR1
        _RR1.MODEL.setdefault('mic', 'mic_whitebg-trained')
    REC = dict(rows=[], X=[], T=[], IT=[], stop='끝까지', t0=time.time())
    from concurrent.futures import ThreadPoolExecutor as _TPE
    _SX = _TPE(max_workers=int(os.environ.get('STABHEX_STAT_W', '6')))
    _orig = runpy.run_path
    _done = [False]

    def _install():
        import numpy as np
        import torch
        import gsrender as DM
        import hexfem as ART
        import stability as SS
        d = torch.load(_arg('--data'), map_location='cpu', weights_only=False)
        cfg = d['cfg']
        m = DM.traj_mass(d, 'cpu').double().numpy()
        N = m.shape[0]
        vol = m / float(cfg['density'])
        fl = [b for b in cfg['boundary_conditions'] if b.get('type') == 'surface_collider']
        FZ = float(fl[0]['point'][2]) if fl else 0.0
        G = [float(q) for q in cfg['g']]
        EK = dict(vol=vol, E=float(cfg['E']), nu=float(cfg['nu']))
        SUB = SS.subset(N)
        x0 = d['x'][0].double().numpy()
        v0 = d['v'][0].double().numpy()
        REC['rows'].append(SS.frame_stats(x0, v0, np.tile(np.eye(3), (N, 1, 1)), m, G, FZ, **EK))
        REC['X'].append(x0[SUB].astype(np.float32))
        REC.update(sub=SUB, h=float(cfg['frame_dt']), N=N)
        del d
        _oc = ART.an_rt_RTSolver.counts
        _last = [time.time()]

        def counts(s):
            o = _oc(s)
            xo = s.x_orig()
            vo = torch.empty_like(s.v)
            vo[s.perm] = s.v
            Fo = torch.empty_like(s.F)
            Fo[s.perm] = s.F
            (xn, vn, Fn) = (xo.double().cpu().numpy(), vo.double().cpu().numpy(), Fo.reshape(-1, 3, 3).double().cpu().numpy())
            REC['rows'].append(_SX.submit(SS.frame_stats, xn, vn, Fn, m, G, FZ, **EK))
            REC['X'].append(xn[SUB].astype(np.float32))
            it = o.get('it') or {}
            REC['IT'].append([it.get(k, -1) for k in ('nN', 'nP', 'capN', 'capP', 'steps', 'solves')])
            t = time.time()
            REC['T'].append(t - _last[0])
            _last[0] = t
            return o
        ART.an_rt_RTSolver.counts = counts
        if OPT['direct'] not in ('0', ''):
            _direct()
        print(f"[stabhex] 안정성 기록 설치: 입자 {N}, 부분표본 {len(SUB)}, 바닥 z {FZ}, -> {OPT['npz']}", flush=True)

    def _direct():
        import hexfem as C3

        class _D:

            def __init__(self, s):
                self.s = s

            def replay(self):
                self.s.record(0.0, self.s.Kcur)

        def capture(s):
            g = _D(s)
            s.graph = g
            s.graphs[s.Kcur] = g
        C3.capture = capture
        print('[stabhex] 그래프 없이 바로 띄움 (--sh_direct)', flush=True)

    def _save():
        if not ON or not REC['X']:
            return
        import numpy as np
        import stability as SS
        K = SS.KEYS
        os.makedirs(os.path.dirname(os.path.abspath(OPT['npz'])), exist_ok=True)
        tmp = OPT['npz'] + '.tmp.npz'
        np.savez_compressed(tmp, sub=REC['sub'], x=np.stack(REC['X']).astype(np.float32), t=np.array(REC['T']), keys=np.array(K), stats=np.array([[r.get(k, np.nan) for k in K] for r in [q.result() if hasattr(q, 'result') else q for q in REC['rows']]]), substeps=int(_arg('--K', 1)), dt=REC['h'], newton=int(_arg('--newton', -1)), cg=int(_arg('--cg', -1)), tolN=float(_arg('--tolN', -1)), tolP=float(_arg('--tolP', -1)), it=np.array(REC['IT'], dtype=np.float64).reshape(-1, 6), frames_done=len(REC['X']) - 1, stop=REC['stop'])
        os.replace(tmp, OPT['npz'])
        print(f"[stabhex] 저장 {OPT['npz']}: 프레임 {len(REC['X']) - 1}, 끝 '{REC['stop']}'", flush=True)

    def run_path(path, *a, **kw):
        if ON and (not _done[0]) and (os.path.basename(path) == 'simulate.py'):
            _done[0] = True
            _install()
        return _orig(path, *a, **kw)

    def _term(signum, frame):
        raise NotImplementedError('not used by the paper settings')
    if ON:
        runpy.run_path = run_path
        signal.signal(signal.SIGTERM, _term)
    sys.argv = [os.path.abspath(argv[0])] + argv[1:]
    try:
        _orig(sys.argv[0], run_name='__main__')
    except BaseException as ex:
        if REC['stop'] == '끝까지':
            REC['stop'] = f'예외 {type(ex).__name__}: {str(ex)[:200]}'
        _save()
        raise
    _save()
WRAPPERS['stabhex_wrap.py'] = w_stabhex_wrap

def w_startpt_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 14000))
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    ON = 0
    while argv and argv[0] == '--sp':
        ON = int(float(argv[1]))
        del argv[:2]
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if ON and (not _done[0]) and (os.path.basename(path) == 'simulate.py'):
            _done[0] = True
            import stability as CF
            CF.startpt_install(out)
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.abspath(argv[0])] + argv[1:]
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['startpt_wrap.py'] = w_startpt_wrap

def w_wmfloor_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    C = {}
    while argv and argv[0].startswith('--wf'):
        C[argv[0][4:]] = argv[1]
        del argv[:2]
    _orig = runpy.run_path
    _st = dict(on=False)

    def run_path(path, *a, **kw):
        b = os.path.basename(path)
        if C and (not _st['on']) and (b in ('simulate.py', 'fem_ao_rt_sim.py')):
            _st['on'] = True
            import fem_wmfloor as WF
            WF.install(out)
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'hhtc2_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['wmfloor_wrap.py'] = w_wmfloor_wrap

def w_wolfh11_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 8000))
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    O = dict(w11='0', w11dg='0')
    while argv and argv[0].startswith('--w11'):
        O[argv[0][2:]] = argv[1]
        del argv[:2]
    MODE = int(float(O['w11']))
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if MODE and (not _done[0]) and (os.path.basename(path) == 'simulate.py'):
            _done[0] = True
            import fem_wolfh11 as W11
            W11.install(out, MODE, int(float(O['w11dg'])))
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'wolfh9_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['wolfh11_wrap.py'] = w_wolfh11_wrap

def w_wolfh12_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 12000))
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    O = dict(w12='0', w12dg='0')
    while argv and argv[0] in ('--w12', '--w12dg'):
        O[argv[0][2:]] = argv[1]
        del argv[:2]
    (MODE, DG) = (int(float(O['w12'])), int(float(O['w12dg'])))
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if (MODE or DG) and (not _done[0]) and (os.path.basename(path) == 'simulate.py'):
            _done[0] = True
            import stability as W12
            W12.wolfh12_install(out, MODE, DG)
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'finterp_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['wolfh12_wrap.py'] = w_wolfh12_wrap

def w_wolfh5_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    CAND = {}
    while argv and argv[0].startswith('--w5'):
        CAND[argv[0][4:]] = int(float(argv[1]))
        del argv[:2]
    ON = any(CAND.values())

    def rel_of(path):
        import torch
        try:
            d = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        except Exception:
            d = torch.load(path, map_location='cpu', weights_only=False)
        cfg = d['cfg']
        o = dict(dxg=float(cfg.get('grid_lim', 2.0)) / int(cfg.get('n_grid', 100)))
        for bc in cfg.get('boundary_conditions') or []:
            if bc.get('type') == 'release_particles_sequentially':
                ax = [k for k in range(3) if bc['normal'][k] != 0][0]
                H0 = abs(float(bc['start_position']) - float(bc['end_position']))
                o.update(ax=ax, c0=float(bc['end_position']), H0=H0, vf=H0 / float(bc['end_time']))
        del d
        return o
    _orig = runpy.run_path
    _st = dict(done=False)

    def run_path(path, *a, **kw):
        if ON and (not _st['done']) and (os.path.basename(path) in ('simulate.py',)):
            _st['done'] = True
            if CAND.get('fb', 0):
                import fem_wolfh4 as W4
                W4.install(fb=1, dg=0, out=out)
            import stability as W5
            W5.wolfh5_install(dg=CAND.get('dg', 0), s=CAND.get('s', 0), c=CAND.get('c', 0), g=CAND.get('g', 0), gk0=CAND.get('gk0', 0), gw=CAND.get('gw', 0), out=out, rel=rel_of(argv[argv.index('--data') + 1]))
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'mfix_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['wolfh5_wrap.py'] = w_wolfh5_wrap

def w_wolfh9_wrap():
    import os
    import runpy
    import sys
    HERE = _HERE
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 6000))
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else '.'
    os.makedirs(out, exist_ok=True)
    O = dict(w9='0')
    while argv and argv[0].startswith('--w9'):
        O[argv[0][2:]] = argv[1]
        del argv[:2]
    MODE = int(float(O['w9']))
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if MODE and (not _done[0]) and (os.path.basename(path) in ('simulate.py', 'fem_ao_rt_sim.py')):
            _done[0] = True
            import fem_wolfh9 as W9
            W9.install(out, MODE)
        return _orig(path, *a, **kw)
    runpy.run_path = run_path
    sys.argv = [os.path.join(HERE, 'speed12_wrap.py')] + argv
    _orig(sys.argv[0], run_name='__main__')
WRAPPERS['wolfh9_wrap.py'] = w_wolfh9_wrap
_HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', 'hexfem')

def _dispatch(path, *a, **kw):
    b = os.path.basename(path)
    if b in WRAPPERS:
        return WRAPPERS[b]()
    if b == 'simulate.py':
        path = os.path.join(_HERE, 'simulate.py')
    return _REAL_RUN_PATH(path, *a, **kw)

def run(argv):
    runpy.run_path = _dispatch
    sys.argv = list(argv)
    runpy.run_path(sys.argv[0], run_name='__main__')


def main():
    if not sys.flags.utf8_mode:                       # logs and json files contain non-ASCII text: always run in UTF-8 mode
        os.execv(sys.executable, [sys.executable, "-X", "utf8"] + sys.argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, choices=sorted(SETTINGS))
    ap.add_argument("--setting", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--gt_cfg", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=0, help="0 = the paper's frame count")
    ap.add_argument("--N", type=int, default=1, help="substeps (stability setting)")
    ap.add_argument("--stats", type=int, default=1, help="stability setting: 1 = record per-frame statistics (res.npz), 0 = timing run")
    ap.add_argument("--direct", action="store_true", help="stability setting: run frames without the CUDA graph (automatic for N >= 512)")
    ap.add_argument("--dump", action="store_true", help="also dump particle positions and F (stability setting; other settings dump when the paper runs did)")
    ap.add_argument("--extra", default="", help="extra flags, e.g. '--tb 1e5' replaces the 16.7 ms time budget")
    a = ap.parse_args()
    if a.setting not in SETTINGS[a.scene]:
        sys.exit(f"setting {a.setting!r} is not available for {a.scene}: {sorted(SETTINGS[a.scene])}")
    frames, cmd = SETTINGS[a.scene][a.setting]
    ph = dict(data=os.path.abspath(a.data), gt=os.path.abspath(a.gt), out=os.path.abspath(a.out),
              gt_cfg=os.path.abspath(a.gt_cfg or os.path.join(a.gt, "config.json")), frames=str(a.frames or frames),
              N=str(a.N), stats=str(a.stats), direct=str(1 if (a.N >= 512 or a.direct) else 0))
    argv = []
    for t in cmd.split():
        for k, v in ph.items():
            t = t.replace("{" + k + "}", v)
        argv.append(t)
    if a.dump and "--dump" not in argv:
        argv += ["--dump", "--dump_F"]
    ex = a.extra.split()
    while ex:                                             # a flag already in the command gets the new value, others are appended
        k = ex.pop(0); v = ex.pop(0) if ex and not ex[0].startswith("--") else None
        if k in argv and v is not None and argv.index(k) + 1 < len(argv):
            argv[argv.index(k) + 1] = v
        else:
            argv += [k] + ([v] if v is not None else [])
    os.makedirs(ph["out"], exist_ok=True)
    run(argv)


if __name__ == "__main__":
    main()
