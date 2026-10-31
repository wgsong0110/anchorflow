# torch 없이 torch.save zip 을 numpy 로 읽는 최소 로더 (읽기 전용; 절 AG.14 wolf 뷰어·수박 뷰어용, arm64 로컬에 torch 없음)
import zipfile, pickle, numpy as np, collections
DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16, "LongStorage": np.int64,
      "IntStorage": np.int32, "ShortStorage": np.int16, "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
      "BFloat16Storage": np.uint16}
class _Stor:
    def __init__(self, name): self.name = name
def load(path):
    z = zipfile.ZipFile(path); pre = z.namelist()[0].split("/")[0]
    cache = {}
    def rebuild(storage, offset, size, stride, *a):
        arr = storage
        if len(size) == 0: return arr[offset]
        it = arr.itemsize
        return np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size), strides=tuple(s * it for s in stride)).copy()
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"): return rebuild
            if mod == "torch" and name.endswith("Storage"): return _Stor(name)
            if mod == "collections" and name == "OrderedDict": return collections.OrderedDict
            if mod == "torch" and name == "device": return lambda *a: str(a)
            if mod == "builtins": return getattr(__builtins__, name) if not isinstance(__builtins__, dict) else __builtins__[name]
            if mod == "numpy.core.multiarray" or mod == "numpy._core.multiarray" or mod == "numpy":
                import importlib; return getattr(importlib.import_module(mod), name)
            raise pickle.UnpicklingError(f"{mod}.{name}")
        def persistent_load(self, pid):
            _, st, key, loc, numel = pid
            if key not in cache:
                cache[key] = np.frombuffer(z.read(f"{pre}/data/{key}"), dtype=DT[st.name])
            return cache[key]
    return U(z.open(f"{pre}/data.pkl")).load()
