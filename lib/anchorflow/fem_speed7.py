"""절 CM72 묶음 speed7 (11-08 위임, 수박 렌더 포함 벽시계 60 fps): 공통 후보 H (= G'' + O + L + B + Rw, 절 CM65·CM67) 위에서
수박 보통 프레임의 커널 단위 프로파일 + 결과 불변 가속. 감싸개 exe/speed7_wrap.py 가 부름 (인자 없으면 이 모듈을 안 부름).

(KP) 커널 프로파일 (--s7kp F0,F1, 결과 불변): 프레임 F0 시작 ~ F1 시작을 torch.profiler (CUDA 활동만 -- 파이썬 호출 쌓기 없음, 호스트 느려짐 최소)
     로 받아 <out>/s7trace.json.gz. 프레임 경계는 가장 바깥 쪼개기 함수 들어갈 때 시뮬 스트림에 띄우는 빈 커널 s7_fm (트레이스에 이름으로 보임).
     그래프 안 커널도 이름·시간이 하나씩 남음 (CUPTI). 분석은 exe/speed7_kp.py (프레임·걸음·뉴턴 반복별 커널 시간).
(NM) mfix 진단 끔 (--s7nm 1, 결과 불변): 절 CM67 fem_mfix 가 프레임마다 그래프 밖에서 하는 진단 (장치 동기화 + 노드 통계 .item() 열몇 번 -> mf.jsonl)
     을 안 함 -- 가벼운 판 L (절 CM65) 이 기록 전용 훅을 걷어낸 것과 같은 일을 G'' 의 새 훅에. 장치 일 (mf_set·mf_mark·mf_save) 은 그대로.
(Z) 지연 백업 (--s7z 1, 결과 불변 -- 옛 실행기 (수박) 만): 프레임 시작 입자 상태 백업 (x·v·F·m·vol·perm·pa·ysp·ljp·alp + R2 h_p, 수박 약 190 MB, 프레임마다
     복사 0.65 ms) 을 바로 뜨지 않고 '아직 안 뜬 백업' 표시만. 근거: 옛 실행기 K 1 그래프는 X1 (절 CM50) 로 꼬리 (걸음 끝·RM 커밋·G2P·R2 끝) 를 버려질 시도에서
     안 띄우고, 넘침 (IP[13]) 이면 커널이 일찍 돌아가므로, 같은 프레임 안 되돌리기 (_rs) 때 입자 상태는 늘 프레임 시작 그대로 -> 되돌리기는 할 일 없음.
     상태를 바꾸는 일 앞에서는 실제로 뜸: (가) 그래프 캡처 (데움 실행이 상태를 바꿈), (나) K ≥ 2 쪼갠 시도 (꼬리가 늘 돎) -- set_K_ga(K > 1) 때.
     확인 판 (--s7z 2): 위와 같이 하되 프레임 시작에 실제 백업도 떠 두고, 할 일 없는 되돌리기마다 지금 입자 상태와 비트 대조 (다르면 실제 백업으로 되돌리고 셈),
     지연 백업을 실제로 뜰 때도 프레임 시작 백업과 비트 대조 -> <out>/s7z.jsonl (프레임마다 셈).
"""
import gzip
import json
import os
import time

import numpy as np
import torch


class C:
    out = "."
    kp = None
    nm = 0
    z = 0


class LZ:
    mode = 0
    ob = None
    orr = None
    cur = None
    st = dict(lazy=0, mat=0, rs_noop=0, rs_real=0, mis=0, mis_mat=0, cmp=0)
    log = None


def _names(s):
    return tuple(s._bk_names())


def _bits_eq(a, b):
    if a is None or b is None:
        return a is b
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def _cmp_state(s, ref):
    """지금 입자 상태 대 백업 ref (원래 _bk 돌려줌) 비트 대조 -> 다른 배열 이름 목록."""
    bad = [nm for nm, t in zip(_names(s), ref) if not _bits_eq(getattr(s, nm), t)]
    h = getattr(s, "e5hr", None)
    if h is not None and getattr(ref, "h", None) is not None and not _bits_eq(h, ref.h):
        bad.append("e5hr")
    LZ.st["cmp"] += 1
    return bad


class _Lazy:
    """아직 안 뜬 백업. real = 실제 백업 (뜬 뒤), ref = 확인 판의 프레임 시작 실제 백업."""

    def __init__(s_, s, ref=None):
        s_.s, s_.real, s_.ref = s, None, ref

    def mat(s_):
        if s_.real is None:
            s_.real = LZ.ob(s_.s)
            LZ.st["mat"] += 1
            if s_.ref is not None:
                bad = [nm for nm, a, b in zip(_names(s_.s), s_.real, s_.ref) if not _bits_eq(a, b)]
                if getattr(s_.ref, "h", None) is not None and not _bits_eq(getattr(s_.real, "h", None), s_.ref.h):
                    bad.append("e5hr")
                if bad:
                    LZ.st["mis_mat"] += 1
                    LZ.st.setdefault("mis_names", []).append(bad)
        return s_.real

    def __getitem__(s_, i):
        if s_.real is None and i == 0:
            return s_.s.x                                                               # 넘침 판정의 비유한 검사 (step) -- 되돌리기 전 상태 = 프레임 시작 (확인 판이 대조)
        return s_.mat()[i]

    def __len__(s_):
        return len(_names(s_.s))


def lz_install(out):
    from . import fem_ao_rt as AORT
    cls = AORT.AORTSolver
    LZ.ob, LZ.orr = cls._bk, cls._rs
    LZ.log = open(os.path.join(out, "s7z.jsonl"), "w")

    def _bk(s):
        ref = LZ.ob(s) if LZ.mode == 2 else None
        b = _Lazy(s, ref)
        LZ.cur = b
        LZ.st["lazy"] += 1
        return b

    def _rs(s, bk):
        if isinstance(bk, _Lazy):
            if bk.real is None:
                LZ.st["rs_noop"] += 1
                if bk.ref is not None:
                    bad = _cmp_state(s, bk.ref)
                    if bad:
                        LZ.st["mis"] += 1
                        LZ.st.setdefault("mis_names", []).append(bad)
                        return LZ.orr(s, bk.ref)
                return None
            LZ.st["rs_real"] += 1
            return LZ.orr(s, bk.real)
        return LZ.orr(s, bk)
    cls._bk, cls._rs = _bk, _rs
    ocap = cls.capture

    def capture(s, *a, **kw):
        if LZ.cur is not None and LZ.cur.s is s:
            LZ.cur.mat()
        return ocap(s, *a, **kw)
    cls.capture = capture
    oK = AORT.set_K_ga

    def set_K_ga(s, K, *a, **kw):
        if int(K) > 1 and isinstance(s, cls) and LZ.cur is not None and LZ.cur.s is s:
            LZ.cur.mat()
        return oK(s, K, *a, **kw)
    AORT.set_K_ga = set_K_ga
    print(f"[절 CM72] 지연 백업 Z 켬 (모드 {LZ.mode}) {cls.__module__}.{cls.__name__}", flush=True)


def lz_frame_end(fr):
    if LZ.log is None:
        return
    st = dict(LZ.st)
    if any(st[k] for k in ("lazy", "mat", "rs_noop", "rs_real", "mis", "mis_mat")) or "mis_names" in st:
        LZ.log.write(json.dumps(dict(f=fr, **st), ensure_ascii=False) + "\n")
        LZ.log.flush()
    for k in LZ.st:
        LZ.st[k] = 0 if not isinstance(LZ.st[k], list) else LZ.st[k]
    LZ.st.pop("mis_names", None)
    LZ.cur = None


class KP:
    prof = None
    mod = None
    fn = None
    dep = 0
    nfr = 0
    t0 = None


FM_SRC = r"""
extern "C" __global__ void s7_fm(int f) { }
"""


def _fm(fi):
    import cupy as cp
    if KP.fn is None:
        KP.mod = cp.RawModule(code=FM_SRC)
        KP.fn = KP.mod.get_function("s7_fm")
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        KP.fn((1,), (1,), (np.int32(fi),))


def _kp_start(fr):
    torch.cuda.synchronize()
    KP.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
    KP.prof.__enter__()
    KP.t0 = time.perf_counter()
    print(f"[절 CM72] 커널 프로파일 시작 프레임 {fr}", flush=True)


def _kp_stop(fr):
    torch.cuda.synchronize()
    p = KP.prof
    KP.prof = None
    p.__exit__(None, None, None)
    fn = os.path.join(C.out, "s7trace.json")
    try:
        p.export_chrome_trace(fn)
        with open(fn, "rb") as f, gzip.open(fn + ".gz", "wb") as g:
            g.write(f.read())
        os.remove(fn)
        print(f"[절 CM72] 커널 프로파일 끝 프레임 {fr} ({KP.nfr} 프레임, {time.perf_counter() - KP.t0:.1f} s) -> s7trace.json.gz", flush=True)
    except Exception:
        import traceback
        open(os.path.join(C.out, "s7kp_err.txt"), "w").write(traceback.format_exc())


def kp_install():
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX
    from . import fem_speed6 as S6
    F0, F1 = C.kp

    def outer(fn):
        def w(s, *a, **kw):
            top = KP.dep == 0
            if top:
                fr = S6.W.fr
                if fr == F0:
                    _kp_start(fr)
                if fr == F1 and KP.prof is not None:
                    _kp_stop(fr)
                if KP.prof is not None:
                    _fm(fr)
                    KP.nfr += 1
            KP.dep += 1
            try:
                return fn(s, *a, **kw)
            finally:
                KP.dep -= 1
        return w
    ART.split_step = outer(ART.split_step)
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    import atexit

    def _end():
        if KP.prof is not None:
            _kp_stop(-1)
    atexit.register(_end)


def nm_install():
    from . import fem_mfix as MF

    def post(s, fi):
        return None
    MF.post = post
    print("[절 CM72] mfix 프레임 진단 끔 (장치 일은 그대로)", flush=True)


def frame_install():
    """가장 바깥 프레임 끝 훅 (지연 백업 셈 기록)."""
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX
    from . import fem_speed6 as S6
    dep = [0]

    def outer(fn):
        def w(s, *a, **kw):
            dep[0] += 1
            try:
                return fn(s, *a, **kw)
            finally:
                dep[0] -= 1
                if dep[0] == 0:
                    lz_frame_end(S6.W.fr)
        return w
    ART.split_step = outer(ART.split_step)
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)


def src_install(out):
    """(SRC) 진단 (--s7src 1, 결과 불변): cupy RawModule 소스 글자를 <out>/s7src_<n>.cu 로 (커널 글자 바꾸기 설계용)."""
    import cupy as cp
    o_init = cp.RawModule.__init__
    n_ = [0]

    def __init__(self, *a, **kw):
        code = kw.get("code", a[0] if a else None)
        if isinstance(code, str) and n_[0] < 12:
            open(os.path.join(out, f"s7src_{n_[0]}.cu"), "w").write(code)
            n_[0] += 1
        return o_init(self, *a, **kw)
    cp.RawModule.__init__ = __init__


def install(out, kp=None, nm=0, z=0, src=0):
    """실행기를 runpy 로 띄우는 순간, speed6 설치 뒤 (가장 바깥) 에 부름."""
    C.out, C.nm, C.z = out, int(nm), int(z)
    if int(src):
        src_install(out)
    if C.z:
        LZ.mode = C.z
        lz_install(out)
        frame_install()
    if kp:
        C.kp = tuple(int(q) for q in str(kp).split(","))
    if C.nm:
        nm_install()
    if C.kp:
        kp_install()
    print(f"[절 CM72] speed7 켬 프로파일 {C.kp} mfix 진단 끔 {C.nm} 지연 백업 {C.z}", flush=True)
