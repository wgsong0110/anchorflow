"""절 AG.19 (사용자 지시) 로컬 판 (numpy, torch 없이): exe/fem_ag19_flip.py 와 같은 식. .pt (zip, 압축 없음) 에서 'x' 텐서의 프레임만 직접 읽는다.
사용: python3 exe/fem_ag19_flip_np.py <정답 .pt> <node_h> <프레임,...> <바닥 z 또는 none> <출력 json>"""
import io
import itertools
import json
import math
import pickle
import struct
import sys
import zipfile

import numpy as np

def open_x(path):
    z = zipfile.ZipFile(path)
    pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    class U(pickle.Unpickler):
        def find_class(s, mod, name):
            if name == "_rebuild_tensor_v2":
                return lambda *a: ("T", a)
            if mod.startswith("torch") and name.endswith("Storage"):
                return name
            if mod.startswith("torch"):
                return lambda *a, **k: (mod, name, a)
            try:
                return super().find_class(mod, name)
            except Exception:
                return lambda *a, **k: (mod, name, a)
        def persistent_load(s, pid):
            return ("S", pid)
    obj = U(io.BytesIO(z.read(pk))).load()
    t = obj["x"][1]
    st, off, shape, stride = t[0][1], t[1], t[2], t[3]
    dtype = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64}[st[1]]
    key = st[2]
    base = pk.rsplit("/", 1)[0]
    info = z.getinfo(f"{base}/data/{key}")
    assert info.compress_type == 0
    f = open(path, "rb"); f.seek(info.header_offset)
    hdr = f.read(30); nl, el = struct.unpack("<HH", hdr[26:30])
    start = info.header_offset + 30 + nl + el
    return f, start, off, shape, np.dtype(dtype)

def frame(fh, start, off, shape, dt, k):
    n = shape[1] * shape[2]
    fh.seek(start + (off + k * n) * dt.itemsize)
    return np.frombuffer(fh.read(n * dt.itemsize), dtype=dt).reshape(shape[1], shape[2]).astype(np.float64)

path, nh = sys.argv[1], float(sys.argv[2])
frames = [int(q) for q in sys.argv[3].split(",")]
zfl = None if sys.argv[4] == "none" else float(sys.argv[4])
fh, start, off, shape, dt = open_x(path)
r = nh / math.sqrt(3.0); c, s3 = 0.5, math.sqrt(3.0) / 2.0; hz = nh / math.sqrt(6.0)
u = ((1.0, 0.0), (-c, s3), (-c, -s3))
A = np.array([[r * u[i][0] for i in range(3)], [r * u[i][1] for i in range(3)], [hz] * 3])
Ai = np.linalg.inv(A)
a_edge = float(np.linalg.norm(A[:, 0])); Vtet = a_edge ** 3 / 6.0
rng = np.random.default_rng(0)
combos = {n: np.array(list(itertools.combinations(range(n), 4))) for n in range(4, 12)}
out = []
for fr_ in frames:
    x0 = frame(fh, start, off, shape, dt, fr_); x1 = frame(fh, start, off, shape, dt, fr_ + 1)
    dl = float(np.abs(x0).max()) * 2.0 ** -24 if dt == np.float32 else float(np.abs(x0).max()) * 2.0 ** -11
    thr = 10.0 * (1.0 / 6.0) * 3.0 * (2.0 * dl) * (3.0 * a_edge ** 2)
    y = x0 @ Ai.T
    b = np.floor(y).astype(np.int64); fr = y - b
    perm = np.argsort(-fr, axis=1)
    pid = perm[:, 0] * 3 + perm[:, 1]
    b -= b.min(0); nb = b.max(0) + 1
    key = ((b[:, 0] * nb[1] + b[:, 1]) * nb[2] + b[:, 2]) * 9 + pid
    order = np.argsort(key, kind="stable"); ks = key[order]
    cut = np.flatnonzero(np.diff(ks)) + 1
    starts = np.concatenate([[0], cut]); cnt = np.diff(np.concatenate([starts, [ks.size]]))
    zmin = float(x0[:, 2].min())
    n_t = n_any = n_half = 0; nb_t = nb_any = nb_half = 0; n_q = n_qf = 0
    for s_, c_ in zip(starts[cnt >= 4], cnt[cnt >= 4]):
        idx = order[s_:s_ + c_]
        cb = combos[c_] if c_ in combos else np.stack([rng.choice(c_, 4, replace=False) for _ in range(500)])
        if cb.shape[0] > 500:
            cb = cb[rng.choice(cb.shape[0], 500, replace=False)]
        P0 = x0[idx][cb]; P1 = x1[idx][cb]
        V0 = np.linalg.det(P0[:, 1:] - P0[:, :1]) / 6.0
        V1 = np.linalg.det(P1[:, 1:] - P1[:, :1]) / 6.0
        ok = np.abs(V0) >= thr
        if not ok.any():
            continue
        fl = np.sign(V0[ok]) != np.sign(V1[ok])
        n_t += 1; n_q += int(ok.sum()); n_qf += int(fl.sum())
        anyf = bool(fl.any()); half = float(fl.mean()) >= 0.5
        n_any += anyf; n_half += half
        if zfl is not None and float(x0[idx, 2].min()) <= max(zmin, zfl) + 3.0 * nh:
            nb_t += 1; nb_any += anyf; nb_half += half
    rr = dict(frame=fr_, tets_ge4=int((cnt >= 4).sum()), tets_judged=n_t, frac_any=n_any / max(n_t, 1), frac_half=n_half / max(n_t, 1),
              quads=n_q, frac_quads_flipped=n_qf / max(n_q, 1), thr_over_Vtet=thr / Vtet,
              bottom_tets=nb_t, bottom_frac_any=(nb_any / nb_t) if nb_t else None, bottom_frac_half=(nb_half / nb_t) if nb_t else None, zmin=zmin)
    out.append(rr)
    print(json.dumps(rr, ensure_ascii=False), flush=True)
json.dump(out, open(sys.argv[5], "w"), indent=1)
