"""절 CM62 묶음 speed5 (11-08 위임, 렌더 포함 60 fps): 공통 후보 G (= F + A3 + P1X + H4 + Rf) 의 프레임별·단계별 시간 + 시뮬·렌더 스트림 겹치기.
감싸개 exe/speed5_wrap.py 가 실행기를 띄우는 순간 (다른 감싸개가 모두 설치된 뒤 = 가장 바깥) 부름. 인자 없으면 이 모듈을 안 부름.

(M) 프레임 시각 기록 (--s5m 1, 결과 불변; P·O 면 늘 켬): 가장 바깥 쪼개기 함수 (ART·AORT split_step, WRX split_col) 앞뒤에 CUDA 이벤트
    ep0 (시뮬 시작, 시뮬 스트림) · ep1 (시뮬 + 렌더 갱신 + (겹치기 아니면) 래스터화 끝, 시뮬 스트림) -> <out>/s5.jsonl
    per = ep0 -> ep1 (프레임 한 장이 시뮬 스트림을 붙든 시간, 그래프 사이 호스트 판정 빈틈 포함), int = 앞 프레임 ep0 -> 이번 ep0 (처리량 간격),
    lat = ep0 -> 그 프레임 래스터화 끝 (지연: 시뮬 시작 -> 화면).

(P) 단계 시간 기록 (--s5p 1, 결과 불변): 절 CM50 speed4 (P) 와 같은 globaltimer 표식을 옛 실행기 (수박) · 새 실행기 (레고·wolf·ficus·bread) 둘 다에.
    가장 안쪽 커널 띄우기 (풀이기 s.fn 사전) 를 감싸 그래프 캡처 때 단계가 바뀌는 커널 앞에 한 스레드 커널 sp_st 를 넣음 -- 다른 모듈이 원래 k 를 직접 불러도 잡힘.
    단계: P2G · 걸음 시작 · W̄ (e3_) · R2·A2 (e5_) · 벌점 (P1X ghost: gp_·gq·g4~6_) · H4 판정 (hc_) · 뉴턴 [요소 · 장벽 기울기 · 기울기·판정 · PCG · 보폭·선탐색]
    · 걸음 끝 (RM 포함) · G2P · 구간 사이 (나눈 그래프 경계 호스트 읽기·띄우기) · 시도 사이. 예산 커널 (tb_·tc_)·0 채우기·제어는 앞 단계에 붙음.

(O) 시뮬·렌더 스트림 겹치기 (--s5o 1): 래스터라이저 (diff_gaussian_rasterization) 는 커널·cub·cudaMemcpy 를 레거시 기본 스트림 0 에 띄움 ->
    시뮬 전체 (torch·cupy 현재 스트림) 를 비차단 스트림 S 로 옮기고 래스터화만 기본 스트림에. 프레임 t: [S: 시뮬 t] ‖ [0: 래스터화 t−1]
    -> S 가 래스터화 t−1 끝을 기다린 뒤 갱신 t (가우시안 버퍼를 덮으므로) -> 0 이 갱신 t 끝을 기다린 뒤 래스터화 t (호스트는 전처리 개수 읽기까지만 막힘).
    풀이 상태는 읽기만, 시뮬 커널·순서·입력 같음 (결정적 판 끝 해시로 확인). rr.jsonl 의 b = 래스터화 t−1 기다림 + 갱신, c = 래스터화 띄우기 동안 호스트가 막힌 시간
    (= 시뮬 스트림이 실제로 렌더에 내준 몫) -> Rf (절 CM56 비트 8) 는 그 몫을 뺌 (예산 값이 G 와 달라짐 = 시간 따라 바뀌는 결과, 대조는 정확도 표로).
    s5.jsonl craw = 래스터화 자체 GPU 시간 (기본 스트림 이벤트).
"""
import json
import os

import numpy as np
import torch

SP_SRC = r"""
extern "C" {
#define SPN 65536
__device__ unsigned long long SP_T[SPN];
__device__ int SP_C[SPN];
__device__ int SP_N;
__global__ void sp5_st(int c) {
    int i = SP_N;
    if (i < SPN) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); SP_T[i] = t; SP_C[i] = c; }
    SP_N = i + 1;
}
}
"""
SPN = 65536
CATS = ["P2G", "걸음 시작", "W̄", "R2·A2", "벌점", "H4 판정", "요소", "장벽 기울기", "기울기·판정", "PCG", "보폭·선탐색", "걸음 끝", "G2P", "구간 사이", "시도 사이", "기타"]
CI = {c: i for i, c in enumerate(CATS)}
NEWTON_C = {CI[c] for c in ("요소", "장벽 기울기", "기울기·판정", "PCG", "보폭·선탐색", "벌점")}
BEGIN = 1 << 14
INHERIT = {"rt_zero_i", "rt_zero_f", "rt_zero_d", "rt_ctl", "rt_sc_clear", "rt_reset", "sp5_st", "sp_st"}
STEP0 = {"ao_step0_ga", "rt_step0", "ao_step0_th", "rt_ga_step0", "rt_step0g", "rt_bdf_base"}
N_ELEM = {"aq_elem", "ao_elem", "ao_elem_s", "rt_elem", "rt_elem_c"}
N_BAR = {"rt_bar_q", "rt_bar_n", "rt_barn_q", "wb_cnt", "rt_bar_q_c", "rt_barn_q_c"}
N_GRAD = {"rt_grad", "rt_grad_c", "uf_g", "en_gate", "c2_gate"}
N_PCG_P = ("rt_ap", "rt_cg", "rt_mg_", "ao_mv", "rt_bar_mv", "rt_barn_mv", "rt_dotz", "rt_pinit", "ao_warm", "rt_sc_roll", "mg_", "rt_lm", "rt_jac")
T_G2P_P = ("ao_g2p", "aq_g2p", "rt_g2p", "fpi_", "lji_", "pf_g2p", "g2p")


class SP:
    mod = None
    fn = None
    on = False


class O5:
    on = False
    S = None
    prev_end = None
    pend = []


def _cap():
    return bool(torch.cuda.is_current_stream_capturing())


def _stamp(code):
    from . import fem_rt_cuda as RTC
    with RTC._Stream():
        SP.fn((1,), (1,), (np.int32(code),))


def _cat(s, name):
    if name in INHERIT or name.startswith("tb_") or name.startswith("tc_"):
        return None
    if name.startswith("e3_"):
        return CI["W̄"]
    if name.startswith("e5_"):
        return CI["R2·A2"]
    if name.startswith(("gp_", "gq", "g4_", "g5_", "g6_", "g3_")):
        return CI["벌점"]
    if name.startswith("hc_"):
        return CI["H4 판정"]
    ph = getattr(s, "_s5ph", "pre")
    if ph == "pre":
        return CI["P2G"]
    if ph == "start":
        return CI["걸음 시작"]
    if ph == "newton":
        if name in N_ELEM:
            return CI["요소"]
        if name in N_BAR:
            return CI["장벽 기울기"]
        if name in N_GRAD:
            return CI["기울기·판정"]
        if name.startswith(N_PCG_P):
            return CI["PCG"]
        return CI["보폭·선탐색"]
    if name.startswith(T_G2P_P):
        return CI["G2P"]
    return CI["걸음 끝"]


def _emit(s, cid, begin=False):
    it = int(getattr(s, "_s5it", 0)) if cid in NEWTON_C else 0
    ks = max(0, int(getattr(s, "_s5ks", 0)))
    code = cid + 32 * min(it, 15) + 512 * min(ks, 7) + (BEGIN if begin else 0)
    if code != getattr(s, "_s5last", None) or begin:
        _stamp(code)
        s._s5last = code


class _FnDict(dict):
    """풀이기 s.fn (이름 -> cupy 함수) 를 감쌈: 꺼낼 때 그래프 캡처 중이면 단계 표식을 먼저 띄움 (가장 안쪽 = 모든 s.k 경로)."""
    def __init__(s_, src, owner):
        super().__init__(src)
        s_.owner = owner

    def __getitem__(s_, name):
        f = dict.__getitem__(s_, name)
        s = s_.owner
        if SP.on and _cap():
            if name in STEP0:
                s._s5ph, s._s5it = "start", 0
                s._s5ks = int(getattr(s, "_s5ks", -1)) + 1
            c = _cat(s, name)
            if c is not None:
                _emit(s, c)
        return f


def _sg(name, dt, n=1):
    import cupy as cp
    return cp.ndarray((n,), dtype=dt, memptr=SP.mod.get_global(name))


def sp_frame():
    torch.cuda.synchronize()
    n = int(_sg("SP_N", np.int32)[0])
    m = min(n, SPN)
    if m == 0:
        return dict(n=0)
    T = _sg("SP_T", np.uint64, SPN)[:m].get().astype(np.int64)
    C = _sg("SP_C", np.int32, SPN)[:m].get()
    tries, cur = [], None
    for i in range(m):
        c = int(C[i])
        if c & BEGIN:
            cur = dict(cat={}, it={}, t0=int(T[i]), t1=None)
            tries.append(cur)
        if cur is None:
            continue
        cid = c & 31
        if cid == CI["시도 사이"]:
            cur["t1"] = int(T[i])
        if i + 1 < m:
            d = (int(T[i + 1]) - int(T[i])) * 1e-6
            nm = CATS[cid]
            cur["cat"][nm] = cur["cat"].get(nm, 0.0) + d
            if cid in NEWTON_C:
                it = (c >> 5) & 15
                cur["it"][it] = cur["it"].get(it, 0.0) + d
    out = []
    for q in tries:
        gpu = ((q["t1"] - q["t0"]) * 1e-6) if q["t1"] else None
        out.append(dict(ms=round(gpu, 4) if gpu is not None else None, cat={k: round(v, 4) for k, v in q["cat"].items() if k != "시도 사이"},
                        gap=round(q["cat"].get("시도 사이", 0.0), 4), it={str(k): round(v, 4) for k, v in sorted(q["it"].items())}))
    _sg("SP_N", np.int32)[0] = 0
    torch.cuda.synchronize()
    return dict(n=n, tries=out)


def sp_install():
    import cupy as cp
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_common3 as C3
    SP.mod = cp.RawModule(code=SP_SRC)
    SP.fn = SP.mod.get_function("sp5_st")
    SP.on = True
    for cls in (ART.RTSolver, AORT.AORTSolver):
        o_init, o_rec, o_ctl = cls.__init__, cls.record, cls.ctl

        def __init__(s, *a, _o=o_init, **kw):
            _o(s, *a, **kw)
            if not isinstance(s.fn, _FnDict):
                s.fn = _FnDict(s.fn, s)

        def record(s, t_frame, K=None, _o=o_rec):
            if not isinstance(s.fn, _FnDict):
                s.fn = _FnDict(s.fn, s)
            s._s5ph, s._s5it, s._s5ks, s._s5last = "pre", 0, -1, None
            cap = _cap()
            if cap:
                _emit(s, CI["P2G"], begin=True)
            r = _o(s, t_frame, K)
            if cap:
                _stamp(CI["시도 사이"])
                s._s5last = None
            return r

        def ctl(s, op, it=0, _o=o_ctl):
            if op == 2:
                s._s5ph, s._s5it = "newton", int(it)
            elif op == 8:
                s._s5ph = "tail"
            return _o(s, op, it)
        cls.__init__, cls.record, cls.ctl = __init__, record, ctl
    o_mark = C3.mark

    def mark(s, kind):
        if getattr(s, "_c3cap", None) is not None:
            _stamp(CI["구간 사이"])
            s._s5last = None
        return o_mark(s, kind)
    C3.mark = mark


# ======================= (O) 겹치기 =======================
def ov_install():
    """시뮬을 비차단 스트림 S 로 (torch·cupy 현재 스트림), 래스터화만 기본 스트림 0 (래스터라이저가 늘 쓰는 스트림)."""
    import cupy as cp
    from . import rtrender1 as RR
    O5.on = True
    O5.S = torch.cuda.Stream()
    torch.cuda.set_stream(O5.S)
    O5.cps = cp.cuda.ExternalStream(O5.S.cuda_stream)
    O5.cps.use()
    D = torch.cuda.default_stream()

    def wu(fn):
        def u(*a, **kw):
            if O5.prev_end is not None:                                                  # 갱신이 가우시안 버퍼를 덮기 전에 앞 래스터화 끝을 기다림
                torch.cuda.current_stream().wait_event(O5.prev_end)
            return fn(*a, **kw)
        return u

    def wr(fn):
        def r(G, U, t):
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
            D.wait_event(ev)
            es, ee = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(D):
                es.record(D)
                img = fn(G, U, t)
                ee.record(D)
            O5.prev_end = ee
            O5.pend.append((es, ee))
            return img
        return r
    for nm in ("upd_pg0", "upd_pg1", "upd_gf0", "upd_gf1"):
        setattr(RR, nm, wu(getattr(RR, nm)))
    for nm in ("ras_pg", "ras_gf"):
        setattr(RR, nm, wr(getattr(RR, nm)))


# ======================= (M) 프레임 기록 =======================
def frame_install(out, mode):
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX
    log = open(os.path.join(out, "s5.jsonl"), "w")
    st = dict(dep=0, fr=0, prev=None)
    q = []                                                                                 # 아직 못 쓴 프레임 (지연은 래스터화 끝을 기다려야 함)

    def flush(force=False):
        while q:
            rec, ep0, ep1, ras = q[0]
            if ras is not None and not force and not ras[1].query():
                break
            q.pop(0)
            ep1.synchronize()
            rec["per"] = round(ep0.elapsed_time(ep1), 4)
            if ras is not None:
                ras[1].synchronize()
                rec["lat"] = round(ep0.elapsed_time(ras[1]), 4)
                rec["craw"] = round(ras[0].elapsed_time(ras[1]), 4)
            else:
                rec["lat"] = rec["per"]
            log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()

    def outer(fn):
        def w(s, *a, **kw):
            top = st["dep"] == 0
            if top:
                if SP.on:
                    torch.cuda.synchronize()
                    _sg("SP_N", np.int32)[0] = 0
                    torch.cuda.synchronize()
                ep0 = torch.cuda.Event(enable_timing=True)
                ep0.record()
                O5.pend = []
            st["dep"] += 1
            try:
                r = fn(s, *a, **kw)
            finally:
                st["dep"] -= 1
            if top:
                ep1 = torch.cuda.Event(enable_timing=True)
                ep1.record()
                rec = dict(t=st["fr"], o=int(O5.on), a=float(getattr(s, "last_ms", float("nan"))))
                st["fr"] += 1
                if st["prev"] is not None:
                    ep0.synchronize()
                    rec["int"] = round(st["prev"].elapsed_time(ep0), 4)
                st["prev"] = ep0
                if SP.on:
                    rec["sp"] = sp_frame()
                ras = O5.pend[-1] if (O5.on and O5.pend) else None
                q.append((rec, ep0, ep1, ras))
                flush()
            return r
        return w
    ART.split_step = outer(ART.split_step)
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    import atexit
    atexit.register(lambda: flush(True))


def install(out, p=0, o=0, m=1):
    if o:
        ov_install()
    if p:
        sp_install()
    frame_install(out, dict(p=p, o=o))
    print(f"[절 CM62] speed5 켬 단계 기록 {p} 겹치기 {o} 시뮬 스트림 {torch.cuda.current_stream()}", flush=True)
