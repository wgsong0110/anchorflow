"""절 CM102 수박 지표 (읽기만, torch 없이 -- .pt 를 메모리 맵으로): 판 디렉토리 (pred.pt · lj.pt · summary.json · log.txt) 와 정답 wm.pt.

사용: python exe/tet1_wm.py <wm.pt> <무리=판디렉토리[,판디렉토리...]> ... > 표.md   (+ TET1_WM_ROWS json)
정의 (모두 같은 함수로 정답·판):
- 오차 = summary err_mean (발산이면 err_mean_done).
- 손상 = logJp > 0 몫 (위치 유한 입자 분모), f5·f6·f8·끝 (정답 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51 %).
- 이탈 = 몸 상자 (축별 0.1~99.9 백분위) 밖 거리 > 4 칸 (칸 = 실행기 로그 '셀 … m') 입자 수 최대 (프레임 중), > 20 이면 이탈 판 (exe/speed1_bbox.py 와 같은 정의).
- 튐 = 프레임 사이 입자 속력 (|x_{t+1} − x_t|/Δt) 최대 > 정답의 모든 프레임 최대 속력 인 프레임 수, 판 최대 속력.
- 꿈틀거림 = 강체 뺀 속도 RMS: 프레임마다 유한 입자 (7 개마다 하나) 속도에 강체 운동 v = V + ω × (x − c) 를 최소제곱으로 맞춘 나머지의 RMS (m/s),
  프레임 구간 [1,30) · [30,100) 평균 (정답도 같은 함수 -- 정답은 깨진 조각의 실제 상대 운동이 들어감).
- 되튐 = 질량중심 z (균일 가중) 의 그때까지 최저점 대비 오름 최대 (m) (정답 0.092)."""
import json
import os
import pickle
import re
import sys
import zipfile

import numpy as np

DT = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64, 'IntStorage': np.int32,
      'HalfStorage': np.float16, 'BoolStorage': np.bool_, 'ByteStorage': np.uint8}


class _S:
    def __init__(s, n): s.n = n


def load(path):
    """.pt (zip, 비압축) -> dict, 텐서는 np.memmap 위 보기 (필요한 것만 디스크에서 읽음)."""
    z = zipfile.ZipFile(path); pre = z.namelist()[0].split('/')[0]
    fh = open(path, 'rb')

    def mm(key, dt):
        zi = z.getinfo(f'{pre}/data/{key}')
        fh.seek(zi.header_offset); h = fh.read(30)
        off = zi.header_offset + 30 + int.from_bytes(h[26:28], 'little') + int.from_bytes(h[28:30], 'little')
        return np.memmap(path, dtype=dt, mode='r', offset=off, shape=(zi.file_size // np.dtype(dt).itemsize,))
    cache = {}

    def pload(pid):
        typ, st, key, loc, n = pid
        if key not in cache:
            cache[key] = mm(key, DT[st.n] if isinstance(st, _S) else np.float32)
        return cache[key]

    def rebuild(storage, off, size, stride, *a):
        if len(size) == 0:
            return storage[off]
        return np.lib.stride_tricks.as_strided(storage[off:], shape=size, strides=[q * storage.itemsize for q in stride])

    class U(pickle.Unpickler):
        def find_class(s, mod, name):
            if name == '_rebuild_tensor_v2':
                return rebuild
            if mod == 'torch' and name.endswith('Storage'):
                return _S(name)
            if mod.startswith('torch') or mod.startswith('collections'):
                return (lambda *a, **k: {}) if name == 'OrderedDict' else (lambda *a, **k: None)
            return super().find_class(mod, name)

        def persistent_load(s, pid):
            return pload(pid)
    return U(z.open(f'{pre}/data.pkl')).load()


def rigid_rms(x0, x1, dt):
    ok = np.isfinite(x0).all(1) & np.isfinite(x1).all(1)
    x0 = x0[ok][::7].astype(np.float64); v = (x1[ok][::7].astype(np.float64) - x0) / dt
    if x0.shape[0] < 10:
        return float('nan'), float('nan')
    c = x0.mean(0); r = x0 - c
    # v ≈ V + ω × r : 미지수 [V(3), ω(3)], ω × r = −[r]x ω
    n = r.shape[0]; A = np.zeros((n, 3, 6)); A[:, :, :3] = np.eye(3)
    A[:, 0, 4], A[:, 0, 5] = r[:, 2], -r[:, 1]
    A[:, 1, 3], A[:, 1, 5] = -r[:, 2], r[:, 0]
    A[:, 2, 3], A[:, 2, 4] = r[:, 1], -r[:, 0]
    sol, *_ = np.linalg.lstsq(A.reshape(-1, 6), v.reshape(-1), rcond=None)
    res = v.reshape(-1) - A.reshape(-1, 6) @ sol
    return float(np.sqrt((res.reshape(n, 3) ** 2).sum(1).mean())), float(np.linalg.norm(v, axis=1).max())


def far(x, cell):
    ok = np.isfinite(x).all(1); x = x[ok]
    q0, q1 = np.quantile(x[::7], 0.001, axis=0), np.quantile(x[::7], 0.999, axis=0)
    d = np.linalg.norm(np.clip(np.maximum(q0 - x, x - q1), 0, None), axis=1)
    return int((d > 4 * cell).sum())


def traj_stats(P, dt, cell, LJ=None):
    L = P.shape[0]
    r = dict(rms=[], vmax=[], far=[], com=[])
    for t in range(L):
        x = np.asarray(P[t]); ok = np.isfinite(x).all(1)
        r['com'].append(float(x[ok][:, 2].mean()) if ok.any() else float('nan'))
        r['far'].append(far(x, cell) if ok.sum() > 100 else 0)
        if t + 1 < L:
            a, b = rigid_rms(x, np.asarray(P[t + 1]), dt); r['rms'].append(a); r['vmax'].append(b)
    z = np.array(r['com']); mn = np.minimum.accumulate(np.nan_to_num(z, nan=np.inf)); r['bounce'] = float(np.nanmax(z - mn))
    if LJ is not None:
        r['dmg'] = {}
        for f in (5, 6, 8, L):
            if f - 1 < LJ.shape[0]:
                l = np.asarray(LJ[f - 1]); ok = np.isfinite(np.asarray(P[f - 1])).all(1)
                r['dmg'][f] = float((l[ok] > 0).mean()) if ok.any() else float('nan')
    return r


def main():
    gt = load(sys.argv[1]); X = gt['x']; cfg = gt.get('cfg') or {}
    dt = float(cfg.get('frame_dt', 1 / 30)) if isinstance(cfg, dict) else 1 / 30
    rows = {}
    cell0 = None
    for a in sys.argv[2:]:
        g, ds = a.split('=', 1); R = []
        for d in ds.split(','):
            s = json.load(open(os.path.join(d, 'summary.json'))) if os.path.exists(os.path.join(d, 'summary.json')) else {}
            lg = open(os.path.join(d, 'log.txt'), errors='ignore').read() if os.path.exists(os.path.join(d, 'log.txt')) else ''
            m = re.search(r"셀 ([0-9.]+) m", lg); cell = float(m.group(1)) if m else 0.03
            cell0 = cell0 or cell
            e = s.get('err_mean') if s.get('err_mean') is not None else s.get('err_mean_done')
            rec = dict(dir=d, div=s.get('diverged'), err=e, cell=cell)
            pp = os.path.join(d, 'pred.pt')
            if os.path.exists(pp):
                P = load(pp)['pred']; lj = load(os.path.join(d, 'lj.pt'))['logJp'] if os.path.exists(os.path.join(d, 'lj.pt')) else None
                rec.update(traj_stats(P, dt, cell, lj))
            R.append(rec)
        rows[g] = R
    G = traj_stats(np.asarray(X[1:]), dt, cell0 or 0.03)
    vg = max(G['vmax'])
    for g, R in rows.items():
        for r_ in R:
            if 'vmax' in r_:
                r_['jump'] = int(sum(1 for v in r_['vmax'] if v > vg))
    json.dump(dict(gt=G, rows=rows, dt=dt), open(os.environ.get('TET1_WM_ROWS', 'tet1_wm_rows.json'), 'w'), ensure_ascii=False, default=float)

    def mm_(v, p=2):
        v = [q for q in v if q is not None and np.isfinite(q)]
        return '-' if not v else (f"{np.mean(v):.{p}f}" + (f" [{min(v):.{p}f}~{max(v):.{p}f}]" if len(v) > 1 else ''))
    w = lambda r_, a, b: float(np.nanmean(r_['rms'][a:b])) if len(r_.get('rms', [])) > a else float('nan')
    print(f"정답 최대 입자 속력 {vg:.2f} m/s, 꿈틀거림 [1,30) {np.nanmean(G['rms'][0:29]):.3f} · [30,100) {np.nanmean(G['rms'][29:]):.3f} m/s, 되튐 {G['bounce']:.3f} m\n")
    print("| 판 | 판 수 | 발산 | 오차 % | 끝 손상 % (f5 · f6 · f8) | 이탈 판 · 최대 개수 | 튐 프레임/판 · 최대 속력 m/s | 꿈틀거림 RMS [1,30) · [30,100) m/s | 되튐 m |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g, R in rows.items():
        D = [r_ for r_ in R if 'rms' in r_]
        nd = sum(1 for r_ in R if r_['div'])
        dm = lambda f: mm_([100 * r_['dmg'].get(f, np.nan) for r_ in D if r_.get('dmg') and f in r_['dmg']])
        last = mm_([100 * list(r_['dmg'].values())[-1] for r_ in D if r_.get('dmg')])
        nf = [max(r_['far']) for r_ in D]
        print(f"| {g} | {len(R)} | {nd} | {mm_([100 * r_['err'] for r_ in R if r_['err'] is not None])} | {last} ({dm(5)} · {dm(6)} · {dm(8)}) | "
              f"{sum(1 for q in nf if q > 20)}/{len(D)} · {max(nf) if nf else '-'} | {mm_([r_['jump'] for r_ in D], 1)} · {mm_([max(r_['vmax']) for r_ in D], 1)} | "
              f"{mm_([w(r_, 0, 29) for r_ in D], 3)} · {mm_([w(r_, 29, 200) for r_ in D], 3)} | {mm_([r_['bounce'] for r_ in D], 3)} |")


if __name__ == '__main__':
    main()
