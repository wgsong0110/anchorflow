"""torch 없이 .pt (zip) 를 numpy 로 읽음 (읽기만)."""
import pickle, zipfile, numpy as np, collections
DT = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64, 'IntStorage': np.int32,
      'HalfStorage': np.float16, 'BoolStorage': np.bool_, 'ByteStorage': np.uint8, 'ShortStorage': np.int16, 'CharStorage': np.int8}
class _S:
    def __init__(s, n): s.n = n
def load(path):
    z = zipfile.ZipFile(path); names = z.namelist(); pre = names[0].split('/')[0]
    cache = {}
    def pload(pid):
        typ, st, key, loc, n = pid
        if key not in cache:
            dt = DT[st.n] if isinstance(st, _S) else np.float32
            cache[key] = np.frombuffer(z.read(f'{pre}/data/{key}'), dtype=dt)
        return cache[key]
    def rebuild(storage, off, size, stride, *a):
        if len(size) == 0: return storage[off]
        it = storage.itemsize
        return np.lib.stride_tricks.as_strided(storage[off:], shape=size, strides=[s * it for s in stride])
    class U(pickle.Unpickler):
        def find_class(s, mod, name):
            if name == '_rebuild_tensor_v2': return rebuild
            if mod == 'torch' and name.endswith('Storage'): return _S(name)
            if name == 'OrderedDict': return collections.OrderedDict
            if mod.startswith('torch'): return lambda *a, **k: None
            return super().find_class(mod, name)
        def persistent_load(s, pid): return pload(pid)
    return U(z.open(f'{pre}/data.pkl')).load()
