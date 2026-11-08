"""절 CM56 묶음 t2chain (조정자 위임 11-08): 공통 설정의 시간 예산 T2 (절 CM39, lib/anchorflow/fem_speed3.py) 가 만드는 '덜 풀린 걸음 연쇄' 를 끊는 정의.

현상 (절 CM53.4): T2 가 뉴턴을 자른 걸음이 덜 풀린 채 (‖g‖²/F² 8e3·3e3, 보폭 1/16~1/4) 다음 프레임으로 넘어가고, 다음 프레임들도 반복 2 판정 뒤에서 잘려 덜 풀림이
이어짐 -> 빠른 노드 (> 3 m/s) 3 천~2 만 -> 입자 이탈 -> 노드 상자 커짐 -> 메모리 넘침. 예산 끄면 0.

이 모듈은 fem_speed3 의 시간 예산 장치 소스 TB_SRC 를 tb_install 앞에서 고쳐 (감싸개 exe/t2chain_wrap.py 가 맨 앞에서 부름) 아래를 넣는다. 모드 0 = 진단만 (결과 불변 -- 그래프에 작은
기록 커널 두 개가 들어가 프레임 시간이 수 µs 늘 뿐, 예산 판정 식은 그대로).

정의 (모두 문턱·클램프 없음, 비트 합 --tc M):
  1 누적 이월 (마감 시각): 프레임 n 의 예산 = (B − m) + Σ_{k<n} (B − t_k).  t_k = 앞 프레임의 GPU 시계 프레임 시간 (첫 시도 첫 커널 -> 마지막 시도 끝, 다시 하기·쪼개기 포함),
     B = 16.7 ms. 시뮬이 실시간 시계보다 앞서 있는 만큼만 더 풀고, 뒤처지면 그만큼 덜 품 -> 프레임 시간 누적 합 ≤ n·B + (마지막 프레임 초과분). 그래프 밖 프레임 (첫 프레임·다시 캡처) 은 셈에서 뺌.
  4 한 프레임 이월: 프레임 n 의 예산 = (B − m) + (B − t_{n−1}) -- 바로 앞 프레임이 남긴 (또는 넘긴) 시간만. 이웃 두 프레임 합 ≤ 2B + 초과분.
  2 자른 걸음 몸 전체 후퇴 오일러: 예산으로 자른 걸음이 덜 풀림으로 판정되면 (cj_end, 절 CM24 판정 2) 걸음 끝 갱신을 --ue_loc 노드 집합 S 가 아니라 몸 전체 노드에
     후퇴 오일러 (U1 = y, v1 = (y − U)/h, 가속도 기억 0) 로 -- 절 CM35 비트 32 (무너진 걸음 ‖g‖²/F² > 1e6 이면 몸 전체) 와 같은 갱신을 '예산으로 자름' 에도.
     HHT 가속도 기억·속도 외삽이 덜 풀린 y 를 다음 프레임 관성 예측으로 넘기지 않게 함. 자르지 않은 걸음 (풀림·뉴턴 상한) 은 그대로.

진단 (늘, 결과 불변): 프레임마다 <out>/t2c.jsonl -- 반복별 시작 ‖g‖² (SC[1])·이 걸음 힘 척도 F² (CD[2])·그때 멈춤 표시 (마지막 시도), 이월 잔고 (ms), 자른 걸음 수,
  몸 전체 후퇴 오일러로 바꾼 걸음 수, 그 프레임 예산 한도 (ms).
"""
import json
import os

import numpy as np
import torch

from . import fem_speed3 as S3

MODE = 0
NIT = 16

_T0_OLD = """    if (TB_NEWF) { TB_T0 = n; TB_NEWF = 0; TB_FR = (TB_FR + 1) & 7; TB_MIT[TB_FR] = 0ull; TB_MTL[TB_FR] = 0ull; TB_CUT = 0; TB_GATES = 0; }
    TB_RUN = 0; TB_TP = n; TB_CS = 0;"""
_T0_NEW = """    if (TB_NEWF) {
        if (TC_PEND) { long long d_ = TC_BF - (long long)(TB_TE - TB_T0); TC_LAST = d_; if (TC_MODE & 4) TC_BANK = d_; else TC_BANK += d_; }   // 절 CM56: 앞 (그래프 안) 프레임 몫 이월
        TC_PEND = 1; TC_NALL = 0; TC_NCK = 0;
        TB_T0 = n; TB_NEWF = 0; TB_FR = (TB_FR + 1) & 7; TB_MIT[TB_FR] = 0ull; TB_MTL[TB_FR] = 0ull; TB_CUT = 0; TB_GATES = 0; }
    TB_RUN = 0; TB_TP = n; TB_CS = 0; TC_CK = 0; for (int i_ = 0; i_ < 3*TC_NIT; i_++) TC_IT[i_] = -1.0;   // 진단은 시도마다 (마지막 시도가 남음)"""
_GATE_OLD = "    if (tb_now() - TB_T0 + ci + ct > B) { CT[0] = 1; *SKIP = 1; TB_CUT++; TB_CS = 1; }"
_GATE_NEW = """    long long lim_ = (long long)B + ((TC_MODE & 5) ? TC_BANK : 0ll);                 // 절 CM56 비트 1·4: 이월 잔고만큼 한도를 옮김 (음수면 줄임)
    TC_LIM = lim_;
    if ((long long)(tb_now() - TB_T0 + ci + ct) > lim_) { CT[0] = 1; *SKIP = 1; TB_CUT++; TB_CS = 1; if (TC_MODE & 2) { TC_CK = 1; TC_NCK++; } }"""
_GLOB_OLD = "__device__ int TB_NEWF, TB_RUN, TB_CUT, TB_FR, TB_GATES, TB_CS;"
_GLOB_NEW = _GLOB_OLD + """
#define TC_NIT %d
__device__ long long TC_BANK, TC_BF, TC_LAST, TC_LIM;
__device__ int TC_PEND, TC_CK, TC_MODE, TC_NALL, TC_NCK;
__device__ double TC_IT[3*TC_NIT];""" % NIT
_TAIL = """
// 절 CM56 비트 2: 이 시도에서 예산으로 자른 걸음이면 --ue_loc 노드 집합을 몸 전체로 (ul_st 앞; 갱신 커널이 덜 풀린 걸음 (cjl) 일 때만 집합을 씀)
__global__ void tc_all(const int* IP, int* m) {
    if (!TC_CK) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) atomicAdd(&TC_NALL, 1);
    if (n < IP[9]) m[n] = 1;
}
// 진단 (결과 불변): rt_ctl op 3 뒤 반복 it 의 시작 ‖g‖² (SC[1]), 힘 척도 F² (CD[2]), 멈춤 표시 (CT[0]) -- 시도마다 덮어씀 (마지막 시도가 남음)
__global__ void tc_rec(const double* SC, const double* CD, const int* CT, int it) {
    if (it >= TC_NIT) return;
    TC_IT[3*it] = SC[1]; TC_IT[3*it+1] = CD[2]; TC_IT[3*it+2] = (double)CT[0];
}
"""


def patch_src(src):
    for o_, n_ in ((_GLOB_OLD, _GLOB_NEW), (_T0_OLD, _T0_NEW), (_GATE_OLD, _GATE_NEW)):
        assert src.count(o_) == 1, o_[:60]
        src = src.replace(o_, n_, 1)
    i = src.rstrip().rindex("}")                                                           # extern "C" 닫는 괄호
    return src[:i] + _TAIL.lstrip("\n") + src[i:]


class TC:
    fn = {}
    log = None
    out = "."
    nall_tot = 0


def _g(name, dt, n=1):
    import cupy as cp
    return cp.ndarray((n,), dtype=dt, memptr=S3.TB.mod.get_global(name))


def _launch(name, grid, block, args):
    from . import fem_rt_cuda as RTC
    with RTC._Stream():
        TC.fn[name]((grid,), (block,), args)


def install(mode, out):
    """감싸개 맨 앞에서 (fem_speed3.tb_install 보다 먼저) 부름. mode = 비트 합 (0 = 진단만)."""
    global MODE
    MODE = int(mode)
    assert not (MODE & 1 and MODE & 4), "비트 1 (누적) 과 4 (한 프레임) 는 함께 안 씀"
    TC.out = out
    S3.TB_SRC = patch_src(S3.TB_SRC)
    o_inst = S3.tb_install

    def tb_install(B, m, out_, where=1):
        o_inst(B, m, out_, where)
        for nm in ("tc_all", "tc_rec"):                                                      # 캡처 밖에서 미리 적재
            TC.fn[nm] = S3.TB.mod.get_function(nm)
        _g("TC_MODE", np.int32)[0] = MODE
        _g("TC_BF", np.int64)[0] = int(round(float(B) * 1e6))
        _g("TC_BANK", np.int64)[0] = 0
        _g("TC_PEND", np.int32)[0] = 0
        TC.log = open(os.path.join(out_, "t2c.jsonl"), "w")
        print(f"[절 CM56 t2chain] 모드 {MODE} (1 누적 이월 · 4 한 프레임 이월 · 2 자른 걸음 몸 전체 후퇴 오일러; 0 = 진단만), B {B} ms 여유 {m} ms 자리 {where}", flush=True)
    S3.tb_install = tb_install

    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    for cls in (ART.RTSolver, AORT.AORTSolver):
        _wrap_k(cls)
        _wrap_ctl(cls)
    o_ic = S3.inv_counts

    def inv_counts(s):                                                                     # fem_speed3 프레임 훅 (가장 바깥 쪼개기 함수 끝) 이 프레임마다 부름 -- 여기에 기록을 얹음
        r = o_ic(s)
        try:
            frame_rec(s)
        except Exception:
            import traceback
            with open(os.path.join(TC.out, "t2c_err.txt"), "a") as f_:
                f_.write(traceback.format_exc()[-2000:] + "\n")
        return r
    S3.inv_counts = inv_counts


def _wrap_k(cls):
    ok = cls.k

    def k(s, name, n, args, b=128):
        if name == "ul_st" and (MODE & 2) and S3.TB.mod is not None and getattr(s, "_tbcap", False):
            _launch("tc_all", (int(n) + 255) // 256, 256, (args[0], args[1]))
            s._tc_nins = getattr(s, "_tc_nins", 0) + 1
        return ok(s, name, n, args, b)
    cls.k = k


def _wrap_ctl(cls):
    oc = cls.ctl

    def ctl(s, op, it=0):
        oc(s, op, it)
        if op == 3 and S3.TB.mod is not None and getattr(s, "adapt", False) and getattr(s, "_tbcap", False):
            P = lambda t: np.uint64(t.data_ptr())
            _launch("tc_rec", 1, 1, (P(s.SC), P(s.CD), P(s.CT), np.int32(it)))
    cls.ctl = ctl


def frame_rec(s):
    if TC.log is None or S3.TB.mod is None:
        return
    it = _g("TC_IT", np.float64, 3 * NIT).get().reshape(NIT, 3)
    rows = [[float(a), float(b), int(c)] for a, b, c in it if a >= 0]
    nall = int(_g("TC_NALL", np.int32)[0])
    rec = dict(bank=round(int(_g("TC_BANK", np.int64)[0]) * 1e-6, 3), last=round(int(_g("TC_LAST", np.int64)[0]) * 1e-6, 3),
               lim=round(int(_g("TC_LIM", np.int64)[0]) * 1e-6, 3), cut=int(S3.TB.g("TB_CUT", np.int32)[0]), nck=int(_g("TC_NCK", np.int32)[0]),
               nall=nall, ins=int(getattr(s, "_tc_nins", 0)), it=rows)
    TC.log.write(json.dumps(rec) + "\n"); TC.log.flush()
    _ = torch
