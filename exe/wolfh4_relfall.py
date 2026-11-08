"""절 CM68 방출 뒤 낙하 몫 (읽기만): 순차 방출 (wolf cfg release_particles_sequentially, x 1.5 -> 0.5, 1.5 s, 50 층) 로 풀린 입자 중 바닥 위 10 cm 넘는 입자의
수직 속도 (프레임 중심 차분) 를 방출 뒤 시간 τ 칸별로 평균, 괄호 = 자유 낙하 −g τ 대비 몫. 정답 = wolf.pt x, 판 = pred.pt (f1..f50). torch 없이 numpy 로 읽음.
사용: python exe/wolfh4_relfall.py <wolf.pt> <판이름=pred.pt> ...   (프레임 f1~f24)
"""
import zipfile, pickle, io, numpy as np
DT = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64, 'IntStorage': np.int32, 'BoolStorage': np.bool_, 'HalfStorage': np.float16, 'ByteStorage': np.uint8}
def load(path):
    z = zipfile.ZipFile(path)
    nm = [n for n in z.namelist() if n.endswith('data.pkl')][0]
    pre = nm[:-len('data.pkl')]
    class St:
        def __init__(s, key, dt): s.key, s.dt = key, dt
    def rebuild(storage, off, size, stride, *a):
        buf = np.frombuffer(z.read(pre + 'data/' + storage.key), dtype=storage.dt)
        if len(size) == 0: return buf[off]
        return np.lib.stride_tricks.as_strided(buf[off:], shape=tuple(size), strides=tuple(s * buf.itemsize for s in stride)).copy()
    class U(pickle.Unpickler):
        def find_class(s, mod, name):
            if name == '_rebuild_tensor_v2': return rebuild
            if name in DT: return name
            if mod == 'collections' and name == 'OrderedDict':
                import collections; return collections.OrderedDict
            if mod.startswith('torch'):
                return lambda *a, **k: None
            return super().find_class(mod, name)
        def persistent_load(s, pid):
            typ, st, key, loc, n = pid
            return St(key, DT.get(st if isinstance(st, str) else getattr(st, '__name__', ''), np.float32))
    return U(io.BytesIO(z.read(nm))).load()


def main():
    import sys
    g = load(sys.argv[1])['x'].astype(np.float64)
    x0 = g[0]
    kmax = np.ceil(50 - np.abs(x0[:, 0] - 0.5) / 0.02) - 1
    trel = np.where(kmax >= 0, 0.03 * (kmax + 1), 0.0)
    FL = 0.48
    bins = [0, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.48]
    runs = {'정답': g}
    for a in sys.argv[2:]:
        n, p = a.split('=')
        runs[n] = np.concatenate([x0[None], load(p)['pred'].astype(np.float64)], 0)
    print('| 판 | ' + ' | '.join(f'τ [{a:.2f},{b:.2f}) s' for a, b in zip(bins[:-1], bins[1:])) + ' |')
    print('|---|' + '---|' * (len(bins) - 1))
    print('| 자유 낙하 −g τ̄ (m/s) | ' + ' | '.join(f'{-9.8 * (a + b) / 2:.2f}' for a, b in zip(bins[:-1], bins[1:])) + ' |')
    for n, X in runs.items():
        S = np.zeros(len(bins) - 1); C = np.zeros(len(bins) - 1); T = np.zeros(len(bins) - 1)
        for f in range(1, 25):
            vz = (X[f + 1, :, 2] - X[f - 1, :, 2]) / 0.08
            tau = 0.04 * f - trel
            ok = (tau > 0) & (np.minimum(X[f - 1, :, 2], X[f + 1, :, 2]) - FL > 0.10)
            b = np.digitize(tau, bins) - 1
            for i in range(len(bins) - 1):
                m = ok & (b == i)
                S[i] += vz[m].sum(); C[i] += m.sum(); T[i] += tau[m].sum()
        print(f'| {n} | ' + ' | '.join('-' if C[i] == 0 else f'{S[i] / C[i]:.2f} ({S[i] / C[i] / (-9.8 * T[i] / C[i]):.2f})' for i in range(len(bins) - 1)) + ' |')


if __name__ == "__main__":
    main()
