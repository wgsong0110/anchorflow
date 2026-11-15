"""절 CM184 진단 (결과 불변): 비유한 값 첫 발생 커널 찾기. RTSolver.k 로 띄우는 커널마다 뒤에 검사 커널을 붙여 (그래프 캡처에도 같이 들어감)
감시 버퍼 (f, y, U, eta, yh, Fq, Vq, B, bGn, bKn, mn, SC) 에 비유한 값이 처음 생긴 '띄움 순번' 을 버퍼별로 원자 최소. 순번 -> 커널 이름 표는 띄울 때 기록.
프레임마다 (record 끝) 결과를 읽어 nanprobe.jsonl 에 쓰고 초기화. AF_NANPROBE=1 일 때 hexff 감싸개가 설치."""
import json
import os

import numpy as np
import torch

SRC = r"""
extern "C" {
__global__ void np_f(const float* a, long long n, int seq, int* out) {
    long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!isfinite(a[i])) atomicMin(out, seq);
}
__global__ void np_d(const double* a, long long n, int seq, int* out) {
    long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!isfinite(a[i])) atomicMin(out, seq);
}
}
"""
BUFS = ("f", "y", "U", "eta", "yh", "Fq", "Vq", "B", "bGn", "bKn", "mn", "SC")
ST = dict(mod=None, names={}, seq=0, out=None, fr=0, f=None, busy=False)


def _mod():
    import cupy as cp
    if ST["mod"] is None:
        ST["mod"] = cp.RawModule(code=SRC)
        ST["out"] = cp.full((len(BUFS),), 2 ** 30, dtype=cp.int32)
        ST["dq"] = cp.full((3,), 2 ** 30, dtype=cp.int32); ST["dq"][2] = 0
        ST["pq"] = cp.zeros((4096, 3), dtype=cp.int32)
    return ST["mod"]


def _probe(s, name):
    m = _mod()
    ST["seq"] += 1
    sq = ST["seq"]
    ST["names"][sq] = name
    try:                                                                       # 가우스 점 det 검사 (hexff 커널이 모듈에 있을 때)
        if "hf_cnt" not in s.fn:
            s.fn["hf_cnt"] = s.mod.get_function("hf_cnt")
        cq = int(s.Fq.shape[0])
        s.fn["hf_cnt"](((cq + 127) // 128,), (128,), (np.uint64(s.IP.data_ptr()), np.uint64(s.cb.data_ptr()), np.uint64(s.BG.data_ptr()), np.uint64(s.Fq.data_ptr()),
                       np.uint64(s.Vq.data_ptr()), np.uint64(s.y.data_ptr()), np.int32(sq), np.uint64(ST["pq"].data.ptr + 12 * (sq % 4096))))
    except Exception as e:
        ST["err"] = repr(e)[:200]
    for j, nm in enumerate(BUFS):
        t = getattr(s, nm, None)
        if t is None or not torch.is_tensor(t) or t.numel() == 0:
            continue
        n = t.numel()
        fn = m.get_function("np_d" if t.dtype == torch.float64 else "np_f")
        if t.dtype not in (torch.float64, torch.float32):
            continue
        g = (n + 255) // 256
        fn((g,), (256,), (np.uint64(t.data_ptr()), np.int64(n), np.int32(sq), np.uint64(ST["out"].data.ptr + 4 * j)))


def install(out):
    from . import fem_an_rt as ART
    ST["f"] = open(os.path.join(out, "nanprobe.jsonl"), "w")
    C = ART.RTSolver
    k0 = C.__dict__["k"]

    def k(s, name, n, args, b=128):
        if name == "aq_elem" and getattr(s, "_pkc", None) is None:              # 재질 커널 직전: 이 y 의 det X ≤ 0 점 수 (aq_elem 띄움마다 칸)
            import cupy as cp
            if "ae" not in ST:
                ST["ae"] = cp.zeros((4096, 3), dtype=cp.int32); ST["ae_n"] = 0; ST["ae_seq"] = {}
            i = ST["ae_n"] % 4096; ST["ae_n"] += 1; ST["ae_seq"][i] = ST["seq"]
            try:
                from . import fem_rt_cuda as RTC
                if "hf_cnt" not in s.fn:
                    s.fn["hf_cnt"] = s.mod.get_function("hf_cnt")
                cq = int(s.Fq.shape[0])
                with RTC._Stream():
                    s.fn["hf_cnt"](((cq + 127) // 128,), (128,), (np.uint64(s.IP.data_ptr()), np.uint64(s.cb.data_ptr()), np.uint64(s.BG.data_ptr()), np.uint64(s.Fq.data_ptr()),
                                   np.uint64(s.Vq.data_ptr()), np.uint64(args[5]), np.int32(1), np.uint64(ST["ae"].data.ptr + 12 * i)))
            except Exception as e:
                ST["err"] = repr(e)[:200]
        r = k0(s, name, n, args, b)
        if name in ("rt_ga_y0", "yl_mix") and getattr(s, "_pkc", None) is None:      # 진단: 뒤집힌 점의 노드 pm 별
            import cupy as cp
            if "who" not in ST:
                ST["who"] = cp.zeros((64, 16), dtype=cp.float32); ST["who_n"] = 0; ST["who_nm"] = {}
            i = ST["who_n"] % 64; ST["who_n"] += 1; ST["who_nm"][i] = name
            try:
                from . import fem_rt_cuda as RTC
                if "hf_who" not in s.fn:
                    s.fn["hf_who"] = s.mod.get_function("hf_who")
                cq = int(s.Fq.shape[0]); U_ = lambda t: np.uint64(t.data_ptr())
                with RTC._Stream():
                    s.fn["hf_who"](((cq + 127) // 128,), (128,), (U_(s.IP), U_(s.cb), U_(s.BG), U_(s.Fq), U_(s.Vq), U_(s.y), U_(s.U), U_(s.pm), U_(s.mn), np.uint64(ST["who"].data.ptr + 64 * i)))
            except Exception as e:
                ST["err"] = repr(e)[:200]
        if not ST["busy"] and getattr(s, "_pkc", None) is None:
            ST["busy"] = True
            try:
                from . import fem_rt_cuda as RTC
                with RTC._Stream():
                    _probe(s, name)
            finally:
                ST["busy"] = False
        return r
    C.k = k
    o_ss = ART.split_step

    def split_step(*a, **kw):
        r = o_ss(*a, **kw)
        if not torch.cuda.is_current_stream_capturing() and ST["out"] is not None:
            torch.cuda.synchronize()
            v = ST["out"].get().tolist()
            dq = ST["dq"].get().tolist()
            rec = dict(call=ST["fr"], nseq=ST["seq"], first={nm: (int(q), ST["names"].get(int(q))) for nm, q in zip(BUFS, v) if q < 2 ** 30},
                       detF=(dq[0], ST["names"].get(dq[0])), detX=(dq[1], ST["names"].get(dq[1])), nXbad_sum=dq[2], err=ST.get("err"))
            ST["dq"][:] = 2 ** 30; ST["dq"][2] = 0
            if "ae" in ST:
                a_ = ST["ae"].get(); rec["aq_elem_badX"] = [[i, ST["ae_seq"].get(i), int(a_[i, 2])] for i in range(min(ST["ae_n"], 4096)) if a_[i, 2] > 0][:40]
                rec["n_aq_elem"] = ST["ae_n"]; ST["ae"][:] = 0
            pq = ST["pq"].get()[:, 2]; prev = 0; ch = []
            for q_ in range(1, min(ST["seq"], 4095) + 1):
                if pq[q_] != prev: ch.append([q_, ST["names"].get(q_), int(pq[q_])]); prev = pq[q_]
            rec["detX_changes"] = ch[:80]; ST["pq"][:] = 0
            if "who" in ST:
                w_ = ST["who"].get(); rec["who"] = [[i, ST["who_nm"].get(i), [round(float(v), 6) for v in w_[i, :13]]] for i in range(min(ST["who_n"], 64)) if w_[i, 12] > 0]
                ST["who"][:] = 0
            ST["f"].write(json.dumps(rec, ensure_ascii=False) + "\n"); ST["f"].flush()
            ST["out"].fill(2 ** 30)
        ST["fr"] += 1
        return r
    ART.split_step = split_step
    print("[절 CM184 nanprobe] 비유한 첫 발생 커널 기록 -> nanprobe.jsonl (결과 불변)", flush=True)
