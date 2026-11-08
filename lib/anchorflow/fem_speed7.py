"""절 CM72 묶음 speed7 (11-08 위임, 수박 렌더 포함 벽시계 60 fps): 공통 후보 H (= G'' + O + L + B + Rw, 절 CM65·CM67) 위에서
수박 보통 프레임의 커널 단위 프로파일 + 결과 불변 가속. 감싸개 exe/speed7_wrap.py 가 부름 (인자 없으면 이 모듈을 안 부름).

(KP) 커널 프로파일 (--s7kp F0,F1, 결과 불변): 프레임 F0 시작 ~ F1 시작을 torch.profiler (CUDA 활동만 -- 파이썬 호출 쌓기 없음, 호스트 느려짐 최소)
     로 받아 <out>/s7trace.json.gz. 프레임 경계는 가장 바깥 쪼개기 함수 들어갈 때 시뮬 스트림에 띄우는 빈 커널 s7_fm (트레이스에 이름으로 보임).
     그래프 안 커널도 이름·시간이 하나씩 남음 (CUPTI). 분석은 exe/speed7_kp.py (프레임·걸음·뉴턴 반복별 커널 시간).
(NM) mfix 진단 끔 (--s7nm 1, 결과 불변): 절 CM67 fem_mfix 가 프레임마다 그래프 밖에서 하는 진단 (장치 동기화 + 노드 통계 .item() 열몇 번 -> mf.jsonl)
     을 안 함 -- 가벼운 판 L (절 CM65) 이 기록 전용 훅을 걷어낸 것과 같은 일을 G'' 의 새 훅에. 장치 일 (mf_set·mf_mark·mf_save) 은 그대로.
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


def install(out, kp=None, nm=0):
    """실행기를 runpy 로 띄우는 순간, speed6 설치 뒤 (가장 바깥) 에 부름."""
    C.out, C.nm = out, int(nm)
    if kp:
        C.kp = tuple(int(q) for q in str(kp).split(","))
    if C.nm:
        nm_install()
    if C.kp:
        kp_install()
    print(f"[절 CM72] speed7 켬 프로파일 {C.kp} mfix 진단 끔 {C.nm}", flush=True)
