"""절 CM130: torch 없이 torch.save zip (pred.pt) 를 numpy 로 읽기 (로컬 arm64 에 torch 없음). load(path) -> dict, 텐서는 np.memmap 기반 ndarray."""
import pickle, zipfile, numpy as np, collections

DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16, "LongStorage": np.int64,
      "IntStorage": np.int32, "ShortStorage": np.int16, "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
      "BFloat16Storage": np.uint16}


class _S:
    def __init__(self, name): self.name = name


def load(path):
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pk = [n for n in names if n.endswith("data.pkl")][0]
    pre = pk[:-len("data.pkl")]
    infos = {i.filename: i for i in zf.infolist()}
    cache = {}

    def storage(key, dt):
        if key not in cache:
            info = infos[pre + "data/" + key]
            assert info.compress_type == 0
            with open(path, "rb") as fh:
                fh.seek(info.header_offset + 26)
                n1, n2 = np.frombuffer(fh.read(4), np.uint16)
            off = info.header_offset + 30 + int(n1) + int(n2)
            cache[key] = np.memmap(path, dtype=dt, mode="r", offset=off, shape=(info.file_size // np.dtype(dt).itemsize,))
        return cache[key]

    def rebuild(st, off, size, stride, *a):
        arr, dt = st
        it = arr.itemsize
        if len(size) == 0:
            return arr[off]
        return np.lib.stride_tricks.as_strided(arr[off:], shape=tuple(size), strides=tuple(s * it for s in stride))

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name in DT:
                return _S(name)
            if mod.startswith("torch._utils") and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                return rebuild
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if mod.startswith("torch"):
                return lambda *a, **k: None
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            _, st, key, loc, n = pid
            if hasattr(st, "name"):
                dt = DT[st.name]
            else:  # 새 판: torch.storage.UntypedStorage + dtype 따로 -- 여기선 float32 로 가정
                dt = np.float32
            a = storage(key, dt)
            return (a, a.dtype)

    with zf.open(pk) as f:
        return U(f).load()
