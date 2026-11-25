"""Scene data preparation and the substep-stability driver.

Subcommands (old script -> min_aux file -> new subcommand):
  pg-h5-to-pt         exe/pg_h5_to_pt.py        tools/data.py pg-h5-to-pt   PhysGaussian h5 sequence -> FusedGS input .pt (+ trajectory folder)
  plastfix-lego-cfg   exe/plastfix_lego_cfg.py  tools/data.py plastfix-lego-cfg  writes cfg/nshape/<scene>_lego.json from cfg/nshape/<scene>.json
  hex-input           exe/stabhex_data.py       stability_driver.py hex-input   stability scene (init h5 + config) -> run_fusedgs.py --data .pt
  stability-ladder    exe/stab1115_drv.py       stability_driver.py             per scene: minimum stable N of PG / i-PG / OmniPhysGS / ours,
                                                                                N ladder, visual-metric renders, our timing (see the block comment)
"""
import json
import os
import sys
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _rd(h, k):
    if k not in h:
        return None
    v = np.array(h[k])
    return v.T if v.ndim == 2 and v.shape[0] in (3, 9) and (v.shape[0] != v.shape[-1]) else v

def make_hex_input(src, out, T=60):
    import h5py
    import torch
    with h5py.File(f'{src}/init/sim_0000000000.h5', 'r') as h:
        x0 = _rd(h, 'x').astype(np.float32)
        v0 = _rd(h, 'v')
    cfg = json.load(open(f'{src}/config.json'))
    N = x0.shape[0]
    v0 = np.zeros_like(x0) if v0 is None else v0.reshape(N, 3).astype(np.float32)
    X = torch.from_numpy(x0)[None].expand(T + 1, N, 3)
    d = dict(cfg=cfg, x=X, v=torch.from_numpy(v0)[None], F=torch.eye(3).reshape(1, 1, 3, 3).expand(1, N, 3, 3).contiguous(), F_is_true=True, stabhex=dict(src=src, frames=T, note='x 는 0 프레임 반복 (정답 없음)'))
    torch.save(d, out)
    print(f"[stabhex 입력] {src} -> {out}: 입자 {N}, 프레임 {T}, |v0| 최대 {float(np.abs(v0).max()):.3g}, z 최소 {float(x0[:, 2].min()):.4f}, 바닥 {[b for b in cfg['boundary_conditions'] if b['type'] == 'surface_collider'][0]['point'][2]}", flush=True)
COMMANDS = ('pg-h5-to-pt', 'plastfix-lego-cfg', 'hex-input', 'stability-ladder')
if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit('usage: python exe/tools.py {' + '|'.join(COMMANDS) + '} [options]')
    CMD = sys.argv.pop(1)
else:
    CMD = None
if CMD == 'pg-h5-to-pt':
    import argparse
    import glob
    import json
    import os
    import re
    import h5py
    import numpy as np
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5_dir', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out_pt', required=True)
    ap.add_argument('--v0_from_bc', action='store_true', help="config 의 enforce_particle_translation 중 시작 0·끝 <= substep_dt 인 것(= 첫 서브스텝 P2G 직전에만 입자 속도를 박는 PG 공식 초기 속도 지정)을 0 프레임 v 로 옮긴다. PG 는 0 프레임 h5 를 첫 서브스텝 전에 써서 v[0]=0 으로 남는다. 그 경계조건은 저장 cfg 에서 빼고 cfg['init_velocity_bc'] 에 적는다 (--cfg_bc 가 첫 스텝 노드를 박지 않게 -- 우리 한 스텝은 PG 400 서브스텝)")
    ap.add_argument('--allow_nonfinite', action='store_true', help='절 AG (GF 수박): GF 는 진행 중 발산 입자를 비유한값으로 떨군다 (quarantine) -- x 는 그대로 두고 (지표에서 거른다) F 의 비유한 값만 항등으로')
    ap.add_argument('--out_dir_n', type=int, default=0, help='--out_dir 에 앞 K 프레임만 (0 = 전부; 절 AG 수박: 렌더 대응용 0 프레임만)')
    ap.add_argument('--out_dir', default='', help='lb_render·lb_roll_to_h5 용 궤적 디렉토리 (x (N,3), F (N,9), meta.json R=I)')
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(a.h5_dir, '**', 'sim_*.h5'), recursive=True), key=lambda p: int(re.findall('(\\d+)', os.path.splitext(os.path.basename(p))[0])[-1]))
    assert fs, a.h5_dir
    cfg = json.load(open(a.config))
    cfg.setdefault('n_grid', 50)
    cfg.setdefault('grid_lim', 2.0)
    (X, V, F) = ([], [], [])
    for f in fs:
        with h5py.File(f, 'r') as h:
            x = np.array(h['x'])
            v = np.array(h['v'])
            Fm = np.array(h['f_tensor'])
        assert x.shape[0] == 3 and Fm.shape[0] == 9, (f, x.shape, Fm.shape)
        X.append(torch.from_numpy(x.T.copy()).float())
        V.append(torch.from_numpy(v.T.copy()).half())
        F.append(torch.from_numpy(Fm.T.copy()).reshape(-1, 3, 3).half())
    (X, V, F) = (torch.stack(X), torch.stack(V), torch.stack(F))
    if float(F[0].float().abs().max()) == 0.0:
        F[0] = torch.eye(3).half().expand_as(F[0])
        print('[F] 0 프레임 f_tensor 가 전부 0 (PG 미초기화 저장) -> 항등으로', flush=True)
    if a.v0_from_bc:
        (keep, moved) = ([], [])
        sdt = float(cfg['substep_dt'])
        for bc in cfg.get('boundary_conditions') or []:
            if bc.get('type') == 'enforce_particle_translation' and float(bc['start_time']) == 0.0 and (float(bc['end_time']) <= sdt):
                q = torch.tensor(bc['point'], dtype=torch.float32)
                sz = torch.tensor(bc['size'], dtype=torch.float32)
                m_ = ((X[0] - q).abs() < sz).all(-1)
                v0 = V[0].float()
                v0[m_] = torch.tensor(bc['velocity'], dtype=torch.float32)
                V[0] = v0.half()
                moved.append(bc)
                print(f"[초기 속도] enforce_particle_translation {bc['velocity']} (시각 [0, {bc['end_time']}) = 첫 서브스텝만) -> 0 프레임 v, 입자 {int(m_.sum())}/{m_.numel()}", flush=True)
            else:
                keep.append(bc)
        assert moved, '--v0_from_bc: 시작 0·끝 <= substep_dt 인 enforce_particle_translation 이 없다'
        cfg['boundary_conditions'] = keep
        cfg['init_velocity_bc'] = moved
    if a.allow_nonfinite:
        _bad = ~torch.isfinite(F.float()).reshape(F.shape[0], F.shape[1], 9).all(-1)
        F[_bad] = torch.eye(3).half()
        print(f'[비유한] x 비유한 입자 (프레임별 최대) {int((~torch.isfinite(X).all(-1)).sum(-1).max())}, F 비유한 -> 항등 {int(_bad.sum())} (입자 x 프레임)', flush=True)
        assert torch.isfinite(X[0]).all(), '0 프레임 위치가 비유한'
    else:
        assert torch.isfinite(X).all() and torch.isfinite(F.float()).all()
    err0 = float((F[0].float() - torch.eye(3)).abs().max())
    os.makedirs(os.path.dirname(a.out_pt), exist_ok=True)
    meta = {'h5': fs[0], 'config': os.path.abspath(a.config), 'R': np.eye(3).tolist(), 'src': 'PhysGaussian gs_simulation', 'n_grid_default_filled': True}
    torch.save({'x': X, 'v': V, 'F': F, 'sel': torch.arange(X.shape[1]), 'cfg': cfg, 'n_full': int(X.shape[1]), 'meta': meta, 'gaussians_first': int(X.shape[1])}, a.out_pt)
    print(f"[저장] {a.out_pt}  {X.shape[0]} 프레임 x {X.shape[1]} 입자, F0-I 최대 {err0:.2e}, 0 프레임 |v| 최대 {float(V[0].float().norm(dim=-1).max()):.3g}, 1 프레임 |v| 평균 {float(V[1].float().norm(dim=-1).mean()):.4f}, n_grid {cfg['n_grid']} grid_lim {cfg['grid_lim']}", flush=True)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        json.dump(cfg, open(f'{a.out_dir}/config.json', 'w'), indent=1)
        meta['config'] = f'{a.out_dir}/config.json'
        json.dump(meta, open(f'{a.out_dir}/meta.json', 'w'), indent=1)
        for i in range(X.shape[0] if a.out_dir_n <= 0 else min(a.out_dir_n, X.shape[0])):
            with h5py.File(f'{a.out_dir}/sim_{i:010d}.h5', 'w') as o:
                o.create_dataset('x', data=X[i].numpy().astype(np.float32))
                o.create_dataset('F', data=F[i].float().reshape(-1, 9).numpy().astype(np.float32))
        print(f'[저장] {a.out_dir}  렌더용 {X.shape[0]} 프레임', flush=True)
if CMD == 'plastfix-lego-cfg':
    import copy, json, os
    D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cfg', 'nshape')
    LEGO = dict(rotation_degree=[119.5764, -6.4234, 138.3083], rotation_axis=[0, 1, 2], substep_dt=2.000800320128051e-05, frame_dt=1 / 60, frame_num=80, n_grid=100, _floor_gap=0.003729709149228453)
    BC = [{'type': 'bounding_box'}, {'type': 'surface_collider', 'point': [1, 1, 0.2], 'normal': [0.0, 0.0, 1.0], 'surface': 'sticky', 'friction': 0.0, 'start_time': 0, 'end_time': 1000.0}]
    for sc in ('chair_paste', 'drums_honey', 'bread_foam_drop'):
        c = json.load(open(f'{D}/{sc}.json'))
        c.update(copy.deepcopy(LEGO))
        c['boundary_conditions'] = copy.deepcopy(BC)
        c['_plastfix_lego'] = f'{sc}.json 에서 레고 L000 세팅으로 (조정자 11-15): 회전 L000 초기 R, substep_dt 2.0008e-5, frame_dt 1/60, 80 프레임, 초기 속도 0, n_grid 100, 경계 = bounding_box + sticky 바닥 (z = 최저점 - lift 0.00373, 실행 때 elastic_new_floor.py), 재질·채우기는 원래 장면 그대로'
        json.dump(c, open(f'{D}/{sc}_lego.json', 'w'), indent=1, ensure_ascii=False)
        print(sc, {k: c[k] for k in ('material', 'E', 'nu', 'density', 'yield_stress', 'rotation_degree', 'frame_num')}, c.get('plastic_viscosity'), c.get('scale'))
if CMD == 'hex-input':
    make_hex_input(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 60)
if CMD == 'stability-ladder':
    import argparse, glob, re, shutil, subprocess, threading, time
    from concurrent.futures import ThreadPoolExecutor
    import evaluation as L
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--par', type=int, default=4)
    ap.add_argument('--stages', default='main,vis,speed')
    ap.add_argument('--methods', default='pg,ipg,omni,ours')
    ap.add_argument('--cap', default='pg:1024,ipg:16,omni:256,ours:256,ipgr:64', help='방법별 사다리 상한 N')
    ap.add_argument('--hextag', default='', help='우리 결과 폴더 접미 (수정판 구분)')
    ap.add_argument('--extra_runs', default='', help='main 단계에서 사다리 없이 돌릴 판 (예 pg:1024 = 기준 궤적만)')
    ap.add_argument('--ipgr_env', default='', help='[stabfinal ipgr] i-PG 수정판 환경 (쉼표), 예 AF_FD_EPS=1e-5,AF_LS_REJECT=1,AF_IPG_GPU=1,AF_IPG_FAST=1')
    ap.add_argument('--onerep', type=int, default=0, help='[stabfinal] 1 = 확정 단계도 N 마다 1 회 (판 a 만)')
    ap.add_argument('--timeout', type=int, default=10800)
    ap.add_argument('--hexgraph', type=int, default=0, help='1 = 우리 사다리·덤프 판도 프레임 CUDA 그래프 (N >= 512 만 바로 띄움; 0 = 사다리·덤프는 --sh_direct 1)')
    ap.add_argument('--gt_ns', default='1,8,32')
    ap.add_argument('--gt_fr', type=int, default=12)
    a = ap.parse_args()
    CHILD_ENV = {k: v for k, v in os.environ.items() if k != 'MKL_THREADING_LAYER'} | {'PYTHONUTF8': '1'}   # set by MKL when this process imported numpy/torch; children start clean
    S = a.scene
    ST = L.ST
    W = os.path.dirname(ST)
    PY = sys.executable
    PG = f'{W}/i-physgaussian'
    CAP = {kv.split(':')[0]: int(kv.split(':')[1]) for kv in a.cap.split(',')}
    SEM = threading.BoundedSemaphore(a.par)
    OSEM = threading.BoundedSemaphore(int(os.environ.get('S1115_OPAR', '1')))
    os.makedirs(f'{ST}/{S}', exist_ok=True)
    LOGF = open(f"{ST}/{S}/drv{('_' + os.environ['S1115_TAG'] if os.environ.get('S1115_TAG') else '')}.log", 'a')
    LK = threading.Lock()
    if 'ipgr' in a.methods or 'ipgr' in a.extra_runs:
        subprocess.run([PY, f'{ROOT}/baseline_mpm.py', 'patch-noisefloor', '--ipg', PG], check=True)
    FR = L.nframes(S)
    NS = S in L.NSHAPE

    def say(*s):
        with LK:
            m = time.strftime('%m-%d %H:%M:%S ') + ' '.join((str(q) for q in s))
            print(m, flush=True)
            LOGF.write(m + '\n')
            LOGF.flush()

    def sh(cmd, log, timeout=None):
        with open(log, 'w') as f:
            try:
                p = subprocess.run(['bash', '-c', cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout or a.timeout, stdin=subprocess.DEVNULL, env=CHILD_ENV)
                return p.returncode
            except subprocess.TimeoutExpired:
                f.write('\n[timeout] 시간 초과\n')
                return -9

    def old_cmd(m, n, out, dump=''):
        """[stabfinal] i-PG 통합 커널 (AF_IPG_FUSED, 같은 식의 가속) 은 탄성 재질 0 만 -> 새 형상 (plasticine·metal) 은 끔."""
        c = f'{ST}/{S}'
        if m == 'omni':
            c2 = f'cd {ROOT} && {PY} -u baseline_mpm.py omni --repo {W}/OmniPhysGS --h5 {c}/init/sim_0000000000.h5 --config {c}/config.json --substeps {n} --out {out}.npz'
            return c2 + (f' --dump {dump}' if dump else '')
        if m in ('pg', 'ipg', 'ipgr'):
            ev = a.ipgr_env.replace(',', ' ') if m == 'ipgr' else ''
            fu = 0 if NS or m == 'ipgr' else 1
            c2 = f"cd {PG} && {ev} AF_IPG_FUSED={fu} OMP_NUM_THREADS=2 {PY} -u {ROOT}/baseline_mpm.py run --method {('ipg' if m == 'ipgr' else m)} --h5 {c}/init/sim_0000000000.h5 --config {c}/config.json --substeps {n} --out {out}.npz --ckpt_every 0"
            return c2 + (f' --dump {dump}' if dump else '')
        raise ValueError(f'방법 {m}')

    def hex_cmd(n, d, dump=False, stats=1, direct=None, frames=FR, speed=False):
        """One FusedGS run of the ladder = run_fusedgs.py --setting stability (scene flags live in run_fusedgs.py; the ladder varies --N)."""
        gt, cfg = (f'{ST}/{S}/gt', f'{ST}/{S}/gt/config.json') if NS else (f'{ST}/{S}/rr', f'{ST}/{S}/config.json')
        return (f"cd {ROOT} && {PY} -u run_fusedgs.py --scene {S.split('_')[0]} --setting stability --N {n} --frames {frames} "
                f"--data {ST}/{S}/hex_in.pt --gt {gt} --gt_cfg {cfg} --out {d} --stats {0 if speed else stats}"
                + (' --direct' if (n >= 512 or direct) else '') + (' --dump' if dump else ''))

    def oom_artifact(log):
        """다른 프로세스와 카드 메모리를 나눠 쓰다 난 메모리 부족 (발산 아님) -> 다시. 우리 판의 20 GiB 넘는 할당 시도는 발산 (실행기 관례)."""
        try:
            t = open(log, errors='ignore').read()
        except Exception:
            return False
        if not re.search('out of memory|OutOfMemory|MemoryError', t, re.I):
            return False
        big = [float(x) for x in re.findall('Tried to allocate ([0-9.]+) GiB', t)]
        if big and max(big) >= 20:
            return False
        if re.search('Process \\d+ has [0-9.]+ [GM]iB memory in use', t):
            return True
        fr = [int(x) for x in re.findall('프레임 (\\d+)/%d' % FR, t)]
        if fr and max(fr) >= 5:
            return False
        return True
    SETUP_ERR = re.compile('ModuleNotFoundError|ImportError|Failed to find forward kernel|Failed to load CUDA module|통합 커널은 탄성')

    def setup_error(log):
        """[stabfinal] 실행 환경 오류 (모듈 없음·warp 커널 캐시 동시 컴파일 경합·재질 가드) 로 첫 프레임 전에 죽은 판 = 발산 아님 -> 다시."""
        try:
            t = open(log, errors='ignore').read()
        except Exception:
            return False
        return bool(SETUP_ERR.search(t)) and (not re.search('\\| *[1-9]\\d*/\\d+ \\[', t))

    def run(m, n, rep):
        for k in range(4):
            run1(m, n, rep)
            lg = L.hex_dir(S, n, rep) + a.hextag + '/log.txt' if m == 'ours' else f'{L.res_dir(S, rep)}/{m}_N{n}.log'
            fin = L.hex_dir(S, n, rep) + a.hextag + '/FIN' if m == 'ours' else f'{L.res_dir(S, rep)}/{m}_N{n}.fin'
            if os.path.exists(fin) and oom_artifact(lg) and ('rc=0' not in open(fin).read()):
                say('메모리 부족 (다른 프로세스와 카드 공유) 다시', S, m, n, rep, k)
                os.remove(fin)
                time.sleep(40)
                continue
            if os.path.exists(fin) and 'rc=0' not in open(fin).read() and setup_error(lg):
                say('실행 환경 오류 (발산 아님) 다시', S, m, n, rep, k)
                os.remove(fin)
                for q in (f'{L.res_dir(S, rep)}/{m}_N{n}.npz',):
                    os.path.exists(q) and m != 'ours' and os.remove(q)
                time.sleep(20 + 10 * k)
                continue
            break

    def stale_fin(fin, log):
        """[stabfinal] 옛 실행 환경 오류의 .fin 위에 다시 돌린 판이 중간에 끊긴 경우 (로그가 .fin 보다 1 분 넘게 새것) -> 이 인스턴스의 .fin 을 치우고 다시 돌림."""
        try:
            return os.path.exists(fin) and os.path.getmtime(log) > os.path.getmtime(fin) + 60
        except OSError:
            return False

    def run1(m, n, rep):
        """한 실행 (있으면 건너뜀). 결과는 L.load_run(S, m, n, rep)."""
        if m == 'ours':
            d0 = L.hex_dir(S, n, rep) + a.hextag
            if stale_fin(f'{d0}/FIN', f'{d0}/log.txt'):
                say('끊긴 판 다시', S, m, n, rep)
                os.remove(f'{d0}/FIN')
        else:
            o0 = f'{L.res_dir(S, rep)}/{m}_N{n}'
            if stale_fin(o0 + '.fin', o0 + '.log'):
                say('끊긴 판 다시', S, m, n, rep)
                os.remove(o0 + '.fin')
            elif os.path.exists(o0 + '.fin'):
                mm = re.match('rc=-9 (\\d+)s', open(o0 + '.fin').read().strip())
                if mm and int(mm.group(1)) < a.timeout - 60:
                    say('시간 제한 판 다시 (제한', a.timeout, 's)', S, m, n, rep)
                    os.remove(o0 + '.fin')
        if m == 'ours':
            d = L.hex_dir(S, n, rep) + a.hextag
            if os.path.exists(f'{d}/FIN'):
                return
            os.makedirs(d, exist_ok=True)
            with OSEM, SEM:
                t0 = time.time()
                rc = sh(f'CUDA_VISIBLE_DEVICES=0 {hex_cmd(n, d, direct=not a.hexgraph)}', f'{d}/log.txt')
            open(f'{d}/FIN', 'w').write(f'rc={rc} {time.time() - t0:.0f}s')
            shutil.rmtree(f'{d}/pred.pt', ignore_errors=True)
            if os.path.exists(f'{d}/pred.pt'):
                os.remove(f'{d}/pred.pt')
        else:
            rd = L.res_dir(S, rep)
            os.makedirs(rd, exist_ok=True)
            out = f'{rd}/{m}_N{n}'
            if os.path.exists(out + '.fin'):
                return
            with SEM:
                t0 = time.time()
                rc = sh(f'CUDA_VISIBLE_DEVICES=0 {old_cmd(m, n, out)}', out + '.log')
            for t in glob.glob(out + '_tmp*'):
                shutil.rmtree(t, ignore_errors=True) if os.path.isdir(t) else os.remove(t)
            open(out + '.fin', 'w').write(f'rc={rc} {time.time() - t0:.0f}s')
        say('실행', S, m, n, rep, 'rc', rc, f'{time.time() - t0:.0f}s')

    def stable(m, n, rep):
        r = L.hex_load(L.hex_dir(S, n, rep) + a.hextag, FR) if m == 'ours' else L.load_run(S, m, n, rep)
        (ok, mt) = L.judge(r, S)
        return ok

    def pmap(fn, items):
        with ThreadPoolExecutor(max_workers=max(len(items), 1)) as ex:
            return list(ex.map(fn, items))

    def plan_path(m):
        return f"{ST}/{S}/plan_{m}{(a.hextag if m == 'ours' else '')}.json"

    def method_main(m):
        if os.path.exists(plan_path(m)) and json.load(open(plan_path(m))).get('N_min') is not None:
            return
        cap = CAP[m]
        pows = [p for p in L.POW if p <= cap]
        pmap(lambda n: run(m, n, 'a'), sorted(pows, reverse=True))
        st = {n: stable(m, n, 'a') for n in pows}
        say(S, m, '사다리', {n: st[n] for n in pows})
        good = [n for n in pows if st[n]]
        plan = dict(method=m, ladder=st, N_min=None)
        if good:
            hi = good[0]
            cand = [hi]
            if hi > 1:
                lo = hi // 2
                ints = list(range(lo + 1, hi))
                st_i = []
                for i in range(0, len(ints), max(a.par, 1)):
                    bt = ints[i:i + max(a.par, 1)]
                    pmap(lambda n: run(m, n, 'a'), bt)
                    st_i = [n for n in bt if stable(m, n, 'a')]
                    if st_i:
                        break
                cand = st_i + [hi]
            n = cand[0]
            tried = []
            while n <= 2 * hi:
                reps = ('a',) if a.onerep else L.REPS
                pmap(lambda r: run(m, n, r), list(reps))
                res = [stable(m, n, r) for r in reps]
                tried.append((n, res))
                if all(res):
                    plan['N_min'] = n
                    break
                n += 1
            plan['confirm_tried'] = tried
        json.dump(plan, open(plan_path(m), 'w'))
        say(S, m, 'N_min', plan['N_min'])

    def stage_main():
        ms = a.methods.split(',')
        ex = [x.split(':') for x in a.extra_runs.split(',') if x]
        pmap(lambda kv: run(kv[0], int(kv[1]), kv[2] if len(kv) > 2 else 'a'), ex)
        ms = [m for m in ms if m in CAP]
        ts = {m: threading.Thread(target=method_main, args=(m,)) for m in ms}
        [t.start() for t in ts.values()]
        [t.join() for t in ts.values()]

    def up_vec():
        """렌더러는 회전 전 모델 좌표로 그림 -> 세계 위 (중력 반대) = config 회전 R 의 셋째 행 (vast_elastic_new.sh 와 같은 식; 회전 없는 장면은 0,0,1)."""
        import math
        import numpy as np
        c = json.load(open(f'{ST}/{S}/config.json'))
        R = np.eye(3)
        for (dg, ax) in zip(c.get('rotation_degree') or [], c.get('rotation_axis') or []):
            (cs, sn) = (math.cos(dg / 180 * 3.1415926), math.sin(dg / 180 * 3.1415926))
            R = np.array({0: [[1, 0, 0], [0, cs, -sn], [0, sn, cs]], 1: [[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]], 2: [[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]]}[int(ax)]) @ R
        return ','.join(('%.6f' % q for q in R[2]))

    def traj0():
        """lb_render·lb_roll_to_h5 용 0 프레임 궤적 폴더 (meta.json: h5 = 처음 입자, R = I; sim_0 = 처음 x, F = I)."""
        T0 = f'{ST}/{S}/traj0'
        if not os.path.exists(f'{T0}/sim_0000000000.h5'):
            os.makedirs(T0, exist_ok=True)
            sh(f"cd {ROOT} && {PY} - <<'PYEOF'\nimport h5py, json, numpy as np\nwith h5py.File('{ST}/{S}/init/sim_0000000000.h5') as h:\n    x = np.array(h['x']); x = x.T if x.shape[0] == 3 else x\nwith h5py.File('{T0}/sim_0000000000.h5', 'w') as o:\n    o.create_dataset('x', data=x.astype(np.float32)); o.create_dataset('F', data=np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (len(x), 1)))\njson.dump(dict(h5='{ST}/{S}/init/sim_0000000000.h5', R=np.eye(3).tolist()), open('{T0}/meta.json', 'w'))\nPYEOF", f'{ST}/{S}/traj0.log')
        return T0
    CAM = '--cam_back 1.6 --elev 40 --back2 1.1'

    def vis_one(name, m, n):
        V = f'{ST}/{S}/vis'
        os.makedirs(V, exist_ok=True)
        png = f'{V}/png_{name}'
        if os.path.exists(f'{png}/DONE'):
            return png
        dump = f'{V}/dump_{name}'
        shutil.rmtree(dump, ignore_errors=True)
        T0 = traj0()
        rc = -1
        with SEM:
            if m == 'ours':
                d = f'{V}/hexdump_{name}'
                os.makedirs(d, exist_ok=True)
                sh(f'CUDA_VISIBLE_DEVICES=0 {hex_cmd(n, d, dump=True, stats=1, direct=not a.hexgraph)}', f'{d}/log.txt')
                sh(f'cd {ROOT} && {PY} -u evaluation.py roll-to-h5 --dump {d}/pred.pt --traj {T0} --out {dump}', f'{V}/{name}_conv.log')
                shutil.rmtree(d, ignore_errors=True)
            else:
                sh(f"CUDA_VISIBLE_DEVICES=0 {old_cmd(m, n, f'{V}/{name}', dump=dump)}", f'{V}/{name}_dump.log')
                shutil.copyfile(f'{T0}/meta.json', f'{dump}/meta.json') if os.path.isdir(dump) else None
            if os.path.exists(f'{dump}/meta.json'):
                rc = sh(f'cd {ROOT} && CUDA_VISIBLE_DEVICES=0 {PY} -u evaluation.py render --sim {dump} --config {ST}/{S}/config.json --shape {S} --out {V}/{name}.mp4 --png {png} {CAM} --up {up_vec()} --fps 30', f'{V}/{name}_render.log')
            shutil.rmtree(dump, ignore_errors=True)
        if os.path.isdir(png) and len(glob.glob(f'{png}/*.png')) >= FR:
            open(f'{png}/DONE', 'w').write(str(rc))
        else:
            say('렌더 실패', S, name, 'rc', rc)
        return png

    def stage_vis():
        V = f'{ST}/{S}/vis'
        os.makedirs(V, exist_ok=True)
        items = [('pg_N1024', 'pg', 1024)]
        for m in a.methods.split(','):
            if os.path.exists(plan_path(m)):
                nm = json.load(open(plan_path(m)))['N_min']
                if nm:
                    items.append((f"{m}{(a.hextag if m == 'ours' else '')}_N{nm}", m, nm))
        for (nm, m, n) in items:
            vis_one(nm, m, n)
        ref = f'{V}/png_pg_N1024'
        runs = ' '.join((f'{nm}={V}/png_{nm}' for (nm, m, n) in items if nm != 'pg_N1024' and os.path.exists(f'{V}/png_{nm}/DONE')))
        if runs and os.path.exists(f'{ref}/DONE'):
            sh(f'cd {ROOT} && CUDA_VISIBLE_DEVICES=0 {PY} -u evaluation.py stability-vis --ref {ref} --runs {runs} --out {V}/vis.json', f'{V}/vis.log')
        say(S, 'vis 끝', open(f'{V}/vis.json').read()[:300] if os.path.exists(f'{V}/vis.json') else 'vis.json 없음')

    def stage_speed():
        SP = f'{ST}/{S}/speed'
        os.makedirs(SP, exist_ok=True)
        ns = [1, 2, 4, 8, 16]
        if os.path.exists(plan_path('ours')):
            nm = json.load(open(plan_path('ours'))).get('N_min')
            if nm and nm not in ns:
                ns.append(int(nm))
        for n in ns:
            d = f'{SP}/t_N{n}{a.hextag}'
            if os.path.exists(f'{d}/FIN'):
                continue
            os.makedirs(d, exist_ok=True)
            cmd = hex_cmd(n, d, stats=0, speed=True)
            if os.path.exists(plan_path('ours')):
                lad = json.load(open(plan_path('ours'))).get('ladder', {})
                if lad.get(str(n)) is False:
                    open(f'{d}/FIN', 'w').write('skip=발산')
                    continue
            rc = sh(f'CUDA_VISIBLE_DEVICES=0 {cmd}', f'{d}/log.txt')
            open(f'{d}/FIN', 'w').write(f'rc={rc}')
            say('단독 시간', S, n, 'rc', rc)

    def stage_cmd():
        """[CM192 측정] 사다리 판과 같은 우리 명령을 --gt_ns 의 N 마다 출력 (앞 --gt_fr 프레임, 결과 폴더 <장면>/prof/N<n>)"""
        for n in [int(q) for q in a.gt_ns.split(',')]:
            d = f'{ST}/{S}/prof/N{n}'
            os.makedirs(d, exist_ok=True)
            print('CMD', hex_cmd(n, d, direct=not a.hexgraph, frames=a.gt_fr), flush=True)
    if 'ours' in a.methods.split(',') or 'speed' in a.stages:
        os.makedirs(f'{ST}/{S}/rr', exist_ok=True)
        json.dump({'h5': f'{ST}/{S}/init/sim_0000000000.h5', 'R': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, open(f'{ST}/{S}/rr/meta.json', 'w'))
        if not os.path.exists(f'{ST}/{S}/hex_in.pt'):
            assert not NS, '새 형상 장면의 hex_in.pt 는 부팅이 pg_h5_to_pt 로 만듦'
            make_hex_input(f'{ST}/{S}', f'{ST}/{S}/hex_in.pt', FR)
    for st in a.stages.split(','):
        {'main': stage_main, 'vis': stage_vis, 'speed': stage_speed, 'cmd': stage_cmd}[st]()
    open(f"{ST}/{S}/DRV_DONE_{a.stages.replace(',', '_')}{a.hextag}", 'w').write(time.strftime('%F %T'))
    say('끝', S, a.stages)
