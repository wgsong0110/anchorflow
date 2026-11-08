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
     상태를 바꾸는 일 앞에서는 실제로 뜸: (가) 그래프 밖 record (캡처의 데움 실행이 상태를 바꿈), (나) K ≥ 2 쪼갠 시도 (꼬리가 늘 돎) -- set_K_ga(K > 1) 때.
     확인 판 (--s7z 2): 위와 같이 하되 프레임 시작에 실제 백업도 떠 두고, 할 일 없는 되돌리기마다 지금 입자 상태와 비트 대조 (다르면 실제 백업으로 되돌리고 셈),
     지연 백업을 실제로 뜰 때도 프레임 시작 백업과 비트 대조 -> <out>/s7z.jsonl (프레임마다 셈).
(P) 장벽 표본점 거르기 (--s7p 1, 결과 불변 -- 옛 실행기 (수박) 만): 장벽 커널 (rt_bar_y0e·rt_bar_kfe·ws_bk·rt_bar_q·wb_floor_q·mf_mark) 은 활성 표본점마다
     27 노드를 모아 위치 x_q = x̄_q + Σ_o w_qo (U_o + θ⁻¹ (y_o − U_o)) 와 바닥 거리 d = n·x_q − p_n 을 셈 (몸 거의 전부 = 수박 약 1.6 백만 점).
     w ≥ 0, Σ_o w = 1 (B-스플라인·W̄ 모두) 이라 d ≥ d̄_q + min_o n·t_o (d̄_q = n·x̄_q − p_n, t_o = U_o + θ⁻¹ (y_o − U_o)). 커널을 띄우기 바로 앞에
     노드 4x4x4 덩이마다 min n·t 를 모으고 (s7_bmin, 그 커널이 쓰는 y 그대로; 비유한이면 −∞), 점은 자기 27 노드가 걸친 덩이 (≤ 8) 최소로 하한을 잡아
     하한 > 문턱 T + 여유 (T = d̂ 또는 0 -- 커널이 그 점을 '띠 밖' 으로 다루는 조건) 이면 27 노드 모으기를 건너뛰고 d = 1e30 (띠 밖 갈래 그대로 = 같은 출력).
     커널 뒤 s7_brst 가 덩이 표를 +∞ 로·준비 표시를 0 으로 (다른 경로로 띄운 같은 커널은 거르기 안 함). 확인 판 (--s7p 2): 거르지 않고 다 센 뒤
     '하한이 거르라는데 실제 d 가 문턱 밑' 인 점 (어김) 과 거를 점 수를 셈 -> <out>/s7p.jsonl. 고칠 커널 = --s7pk (기본 PSEL: 덩이 표 값 (부름당 ~16 µs) 보다
     모으기 몫이 큰 것만 -- s4 프로파일).
(Y) 시작점 후보 에너지 나눠 셈 (--s7y 1, 결과 불변 -- 옛 실행기): rt_bar_y0e 는 β_i = 2^-(k0+i) 후보 59 개의 장벽 에너지를 블록 합 60 번으로 한꺼번에 셈
     (띠 안 점이 든 블록마다; 접촉이 퍼질수록 0.1 -> 1.0 ms). 고르기 rt_y0epick 은 '앞에서부터 E_i ≤ E_0 인 첫 i' 라 앞 M 개 (--s7ym, 기본 4) 안에서 찾으면
     뒤 후보 값은 안 쓰임 -> 앞 M 개 + E_0 만 먼저 셈, 장치에서 앞 M 개 안에 고를 것이 있는지 보고 (s7_ychk, rt_y0epick 과 같은 조건) 없을 때만
     같은 재생 안에서 나머지 후보를 셈 (같은 커널·같은 점별 값·같은 블록 합; c_i 는 2 의 거듭제곱이라 ldexpf 시작값 = 되풀이 반감과 비트 같음).
     확인 판 (--s7y 2): 나머지도 늘 세고, 고른 kk 가 '앞 M 개 안에 있다' 는 판정과 맞는지 셈 -> s7p.jsonl ychk·yviol.
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
    p = 0


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
    orec = cls.record

    def record(s, *a, **kw):                                                            # 데움 (그래프 밖 record = 상태를 바꿈) 앞 -- C3 (--xg) 는 인스턴스 capture 를 바꾸므로 여기서
        if LZ.cur is not None and LZ.cur.s is s and not torch.cuda.is_current_stream_capturing():
            LZ.cur.mat()
        return orec(s, *a, **kw)
    cls.record = record
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


P_SRC = r"""
// ---- 절 CM72 (P) 장벽 표본점 거르기 ----
__device__ int* S7B;
__device__ int S7ON;
__device__ int S7RDY;
__device__ unsigned long long* S7VIOL;
__device__ float S7MARG;
__device__ __forceinline__ int s7key(float f) { int i = __float_as_int(f); return i >= 0 ? i : (i ^ 0x7fffffff); }
__device__ __forceinline__ float s7unkey(int k) { return __int_as_float(k >= 0 ? k : (k ^ 0x7fffffff)); }
#define S7_CNT(f_, v_) if (S7ON == 2) { atomicAdd(&S7VIOL[2], 1ULL); if (f_) { atomicAdd(&S7VIOL[1], 1ULL); if (v_) atomicAdd(&S7VIOL[0], 1ULL); } }
// sk1·sk2 = 뒤 커널이 바로 돌아가는 장치 표시 (AF_SKIP·WS4_DN) 자리 -- 켜져 있으면 덩이 표를 안 만들고 준비 표시 0
__global__ void s7_bmin(const int* IP, const float* U, const float* ya, const float* yb, float thinv, float n0, float n1, float n2, const int* sk1, const int* sk2) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if ((sk1 && *sk1) || (sk2 && *sk2)) { if (n == 0) S7RDY = 0; return; }
    if (n == 0) S7RDY = 1;
    if (n >= IP[9] || IP[13]) return;
    int nb1 = IP[7], nb2 = IP[8];
    int i = n / (nb1*nb2), j = (n / nb2) % nb1, k = n % nb2;
    float u0 = U[n*3], u1 = U[n*3+1], u2 = U[n*3+2];
    float ta = (u0 + thinv*(ya[n*3] - u0))*n0 + (u1 + thinv*(ya[n*3+1] - u1))*n1 + (u2 + thinv*(ya[n*3+2] - u2))*n2;
    float t = ta; bool ok = isfinite(ta);
    if (yb) { float tb = (u0 + thinv*(yb[n*3] - u0))*n0 + (u1 + thinv*(yb[n*3+1] - u1))*n1 + (u2 + thinv*(yb[n*3+2] - u2))*n2; ok = ok && isfinite(tb); t = fminf(t, tb); }
    if (!ok) t = __int_as_float(0xff800000);
    int b = ((i >> 2)*((nb1 + 3) >> 2) + (j >> 2))*((nb2 + 3) >> 2) + (k >> 2);
    atomicMin(&S7B[b], s7key(t));
}
__global__ void s7_brst(const int* IP, int cap) {
    int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b == 0) S7RDY = 0;
    int nb = ((IP[6] + 3) >> 2)*((IP[7] + 3) >> 2)*((IP[8] + 3) >> 2);
    if (b < nb && b < cap) S7B[b] = 0x7f800000;
}
// (Y) 시작점 후보 에너지 나눠 셈
__device__ int S7YI0 = 0;
__device__ int S7YI1 = 1073741824;
__device__ int S7YON = 0;
__device__ int S7YGT = 0;
__device__ int S7YK0 = 0;
__device__ int S7YM = 0;
__global__ void s7_yset(int i0, int i1, int on) { S7YI0 = i0; S7YI1 = i1; S7YON = on; }
__global__ void s7_ychk(const double* SC, const int* kk, int M, int cnt) {
    double E0 = SC[0]; int g = 0;
    for (int i=0;i<M;i++) { double E1 = SC[1+i]; if (isfinite(E1) && E1 <= E0) { g = 1; break; } }
    S7YGT = g; S7YK0 = kk[0]; S7YM = M;
    if (cnt) { atomicAdd(&S7VIOL[3], 1ULL); if (g) atomicAdd(&S7VIOL[4], 1ULL); }
}
__global__ void s7_ycmp(const int* kk) { if (S7YGT && !(kk[0] >= S7YK0 && kk[0] < S7YK0 + S7YM)) atomicAdd(&S7VIOL[5], 1ULL); }
// 점 q 의 거리 하한 > 문턱 T + 여유 이면 참 (S7RDY 0 = 앞에 s7_bmin 이 안 돌았음 -> 거짓)
__device__ __forceinline__ bool s7far(int q, const int* IP, const int* cb, float db, float T) {
    if (S7ON == 0 || S7RDY == 0 || IP[13]) return false;
    int c = q >> 3; int b0 = cb[c*3], b1 = cb[c*3+1], b2 = cb[c*3+2];
    int B1 = (IP[7] + 3) >> 2, B2 = (IP[8] + 3) >> 2;
    int m = 0x7fffffff;
    for (int x = b0 >> 2; x <= (b0 + 2) >> 2; x++) for (int y = b1 >> 2; y <= (b1 + 2) >> 2; y++) for (int z = b2 >> 2; z <= (b2 + 2) >> 2; z++) m = min(m, S7B[(x*B1 + y)*B2 + z]);
    float mn = s7unkey(m);
    return db + mn > T + S7MARG;
}
"""
# 커널 이름 -> (U, ya, yb (None = 한 위치), thinv, n0) 인자 자리 (커널 글자 고침은 _p_patch)
PK = {"rt_bar_y0e": (6, 4, 5, 7, 8), "rt_bar_kfe": (5, 3, 4, 6, 7), "ws_bk": (5, 3, 4, 6, 7), "yl_kq": (6, 4, 5, 7, 8),
      "rt_bar_q": (5, 4, None, 6, 7), "wb_floor_q": (5, 4, None, 6, 7), "yl_chk": (4, 3, None, 5, 6), "en_phi_q": (7, 6, None, 8, 13),
      "mf_mark": (4, 4, 3, 5, 6), "hc_bar": (4, 4, 3, 5, 6)}
PSK2 = {"ws_bk"}                                                                         # 뒤 커널이 WS4_DN 이면 바로 돌아감 (절 CM50 S1)


def _db(Q):
    return f"(AF_BX[{Q}*4]/bm)*n0 + (AF_BX[{Q}*4+1]/bm)*n1 + (AF_BX[{Q}*4+2]/bm)*n2 - pn"


def _pre(Q, T, loop):
    """27 노드 모으기 고리 앞: 거르기 판정 + 거르면 (모드 1) 고리 건너뜀."""
    ind = loop[:len(loop) - len(loop.lstrip())]
    return f"{ind}bool s7f_ = s7far({Q}, IP, cb, {_db(Q)}, {T});\n{ind}if (!(s7f_ && S7ON == 1))\n" + loop


# 커널 이름 -> [(옛 글자, 새 글자)] (덤프 s7src: 장벽 판 옛 실행기 + W̄ (E3W)·mfix (MFW)·hhtc2 (HC_W) 고친 뒤 글자)
L2 = "for (int o=0;o<27;o++) { float w = E3W(q, o); int id = ids[o];\n"
D2 = "    float d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn, d1 = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn;\n"
PATCH = {
    "rt_bar_y0e": [("        for (int o=0;o<27;o++) { float w = E3W(qq, o); int id = ids[o];\n", _pre("qq", "dh", "        for (int o=0;o<27;o++) { float w = E3W(qq, o); int id = ids[o];\n")),
                   ("        d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn; d1 = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn;\n",
                    "        d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn; d1 = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn;\n        S7_CNT(s7f_, !(fminf(d0, d1) >= dh)); if (s7f_ && S7ON == 1) { d0 = 1e30f; d1 = 1e30f; }\n")],
    "rt_bar_kfe": [("    " + L2, _pre("q", "0.f", "    " + L2)), (D2, D2 + "    S7_CNT(s7f_, !(d1 > 0.f)); if (s7f_ && S7ON == 1) return;\n")],
    "ws_bk": [("    " + L2, _pre("q", "0.f", "    " + L2)), (D2, D2 + "    S7_CNT(s7f_, !(d1 > 0.f)); if (s7f_ && S7ON == 1) return;\n")],
    "yl_kq": [("    " + L2, _pre("q", "dh", "    " + L2)), (D2, D2 + "    S7_CNT(s7f_, !(fminf(d0, d1) >= dh)); if (s7f_ && S7ON == 1) { d0 = 1e30f; d1 = 1e30f; }\n")],
    "rt_bar_q": [("        for (int o=0;o<27;o++) { float w = E3W(qq, o); int id = ids[o];\n", _pre("qq", "dh", "        for (int o=0;o<27;o++) { float w = E3W(qq, o); int id = ids[o];\n")),
                 ("        float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;\n",
                  "        float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;\n        S7_CNT(s7f_, !(d >= dh)); if (s7f_ && S7ON == 1) d = 1e30f;\n")],
    "wb_floor_q": [("    " + L2, _pre("q", "dh", "    " + L2)),
                   ("    float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;\n    if (!(d < dh)) return;\n",
                    "    float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;\n    S7_CNT(s7f_, !(d >= dh)); if (s7f_ && S7ON == 1) return;\n    if (!(d < dh)) return;\n")],
    "yl_chk": [("    for (int o=0;o<27;o++) { float w = E3W(q, o); int id = ids[o]; for (int a=0;a<3;a++) xa[a] +=",
                _pre("q", "0.f", "    for (int o=0;o<27;o++) { float w = E3W(q, o); int id = ids[o]; for (int a=0;a<3;a++) xa[a] +=")),
               ("    if (xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn > 0.f) return;\n",
                "    { float d_ = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn; S7_CNT(s7f_, !(d_ > 0.f)); if (s7f_ && S7ON == 1) return; if (d_ > 0.f) return; }\n")],
    "en_phi_q": [("            for (int o=0;o<27;o++) { float w = E3W(q, o); int id = ids[o]; for (int a=0;a<3;a++) xq[a] +=",
                  _pre("q", "dh", "            for (int o=0;o<27;o++) { float w = E3W(q, o); int id = ids[o]; for (int a=0;a<3;a++) xq[a] +=")),
                 ("            float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;\n            if (d < dh) {\n",
                  "            float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;\n            S7_CNT(s7f_, !(d >= dh)); if (s7f_ && S7ON == 1) d = 1e30f;\n            if (d < dh) {\n")],
    "mf_mark": [("    for (int o=0;o<27;o++) { float w = MFW(q, oc, o); int id = ids[o];\n", _pre("q", "dh", "    for (int o=0;o<27;o++) { float w = MFW(q, oc, o); int id = ids[o];\n")),
                ("    float ds = xs[0]*n0 + xs[1]*n1 + xs[2]*n2 - pn, dp = xp[0]*n0 + xp[1]*n1 + xp[2]*n2 - pn;\n",
                 "    float ds = xs[0]*n0 + xs[1]*n1 + xs[2]*n2 - pn, dp = xp[0]*n0 + xp[1]*n1 + xp[2]*n2 - pn;\n    S7_CNT(s7f_, !(fminf(ds, dp) >= dh)); if (s7f_ && S7ON == 1) return;\n")],
    "hc_bar": [("    for (int o=0;o<27;o++) { float w = HC_W(q, oc, o); int id = ids[o];\n", _pre("q", "dh", "    for (int o=0;o<27;o++) { float w = HC_W(q, oc, o); int id = ids[o];\n")),
               ("    float ds = xs[0]*n0 + xs[1]*n1 + xs[2]*n2 - pn, de = xe[0]*n0 + xe[1]*n1 + xe[2]*n2 - pn;\n",
                "    float ds = xs[0]*n0 + xs[1]*n1 + xs[2]*n2 - pn, de = xe[0]*n0 + xe[1]*n1 + xe[2]*n2 - pn;\n    S7_CNT(s7f_, !(fminf(ds, de) >= dh)); if (s7f_ && S7ON == 1) return;\n")],
}


PSEL = ("rt_bar_kfe", "rt_bar_q", "wb_floor_q", "mf_mark", "hc_bar", "rt_bar_y0e", "yl_kq")
YPATCH = [("{ if (AF_SKIP) return;\n", "{ if (AF_SKIP) return; if (S7YON && S7YGT) return;\n"),
          ("    blocksum_add(e, &SC[0]);\n    float c = ldexpf(1.f, -kk[0]);\n    for (int i=0;i<NE;i++) {\n",
           "    if (S7YI0 == 0) blocksum_add(e, &SC[0]);\n    float c = ldexpf(1.f, -kk[0] - S7YI0);\n    for (int i=S7YI0;i<min(NE, S7YI1);i++) {\n")]


class PP:
    y = 0
    ym = 4
    sel = PSEL
    mode = 0
    marg = 1e-4
    patched = set()
    log = None
    bad = set()
    said = set()


def _p_patch(src):
    """장벽 커널 글자 고침 (PATCH 표; 소스에 그 커널이 없으면 건너뜀). 고친 커널 이름 -> PP.patched."""
    import re
    items = [(nm, list(r)) for nm, r in PATCH.items() if (PP.mode and nm in PP.sel)]
    if PP.y:
        d_ = dict(items)
        d_["rt_bar_y0e"] = d_.get("rt_bar_y0e", []) + YPATCH
        items = list(d_.items())
    for nm, reps in items:
        m = re.search(r"__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?" + nm + r"\s*\(", src)
        if not m:
            continue
        i = m.start()
        j = src.find("\n__global__", m.end())
        j = len(src) if j < 0 else j
        body = src[i:j]
        ok = True
        for old, new in reps:
            if body.count(old) != 1:
                ok = False
                PP.bad.add((nm, old[:60], body.count(old)))
                break
            body = body.replace(old, new, 1)
        if ok:
            src = src[:i] + body + src[j:]
            PP.patched.add(nm)
    return src


def _p_src(src):
    if "s7_bmin(" in src or "__device__ float* AF_BX;" not in src:
        return src
    src = _p_patch(src)
    k = src.find("__global__")                                                         # 장치 함수·전역은 첫 커널 앞에 (고친 커널이 s7far 를 부름)
    k = src.rfind("\n", 0, k) + 1
    return src[:k] + P_SRC + src[k:]


class _PFn(dict):
    """풀이기 s.fn (이름 -> cupy 함수): 고친 장벽 커널을 꺼내면 앞에 s7_bmin, 뒤에 s7_brst 를 붙여 띄우는 함수를 돌려줌."""

    def __init__(s_, src, owner):
        super().__init__(src)
        s_.owner = owner

    def __getitem__(s_, name):
        f = dict.__getitem__(s_, name)
        if PP.y and name == "rt_bar_y0e" and name in PP.patched:
            return _YCall(s_.owner, name, f)
        if PP.y == 2 and name == "rt_y0epick":
            return _YPick(s_.owner, f)
        if PP.mode and name in PP.patched and name in PK:
            return _PCall(s_.owner, name, f)
        return f


class _PCall:
    def __init__(s_, s, name, f):
        s_.s, s_.name, s_.f = s, name, f

    def pre(s_, args):
        s = s_.s
        iU, ia, ib, it, i0 = PK[s_.name]
        a = args
        ok = len(a) > i0 + 2 and all(isinstance(a[q], np.uint64) for q in (iU, ia) + ((ib,) if ib is not None else ())) and all(isinstance(a[q], np.float32) for q in (it, i0, i0 + 1, i0 + 2))
        if not ok:
            if s_.name not in PP.said:
                PP.said.add(s_.name)
                print(f"[절 CM72] 장벽 거르기: {s_.name} 인자 꼴이 달라 거르지 않음 ({[type(q).__name__ for q in a]})", flush=True)
            return False
        sk1, sk2 = _skp(s, "AF_SKIP"), (_skp(s, "WS4_DN") if s_.name in PSK2 else np.uint64(0))
        capN = int(s.capN)
        s.fn.get_raw("s7_bmin")(((capN + 255) // 256,), (256,), (a[0], a[iU], a[ia], a[ib] if ib is not None else np.uint64(0), a[it], a[i0], a[i0 + 1], a[i0 + 2], sk1, sk2))
        return True

    def post(s_, args):
        cap = int(s_.s.s7b.numel())
        s_.s.fn.get_raw("s7_brst")(((cap + 255) // 256,), (256,), (args[0], np.int32(cap)))

    def __call__(s_, grid, block, args, **kw):
        ok = s_.pre(args)
        r = s_.f(grid, block, args, **kw)
        if ok:
            s_.post(args)
        return r


class _YCall:
    """rt_bar_y0e: (P 면 덩이 표) -> 앞 M 후보 -> 판정 -> 나머지 (판정이 찾았으면 장치에서 건너뜀) -> 기본값 되돌림."""

    def __init__(s_, s, name, f):
        s_.s, s_.name, s_.f = s, name, f

    def __call__(s_, grid, block, args, **kw):
        s = s_.s
        fs, fc = s.fn.get_raw("s7_yset"), s.fn.get_raw("s7_ychk")
        NE = int(args[16])
        M = min(int(PP.ym), NE)
        pc = _PCall(s, s_.name, s_.f) if (PP.mode and s_.name in PP.sel) else None
        ok = pc.pre(args) if pc is not None else False
        fs((1,), (1,), (np.int32(0), np.int32(M), np.int32(0)))
        s_.f(grid, block, args, **kw)
        fc((1,), (1,), (args[15], args[14], np.int32(M), np.int32(1 if PP.y == 2 else 0)))
        fs((1,), (1,), (np.int32(M), np.int32(NE), np.int32(1 if PP.y == 1 else 0)))
        r = s_.f(grid, block, args, **kw)
        fs((1,), (1,), (np.int32(0), np.int32(1 << 30), np.int32(0)))
        if ok:
            pc.post(args)
        return r


class _YPick:
    def __init__(s_, s, f):
        s_.s, s_.f = s, f

    def __call__(s_, grid, block, args, **kw):
        r = s_.f(grid, block, args, **kw)
        s_.s.fn.get_raw("s7_ycmp")((1,), (1,), (args[1],))
        return r


def _skp(s, nm):
    c = getattr(s, "_s7skp", None)
    if c is None:
        c = s._s7skp = {}
    if nm not in c:
        try:
            c[nm] = np.uint64(s.mod.get_global(nm).ptr)
        except Exception:
            c[nm] = np.uint64(0)
    return c[nm]


def _pfn_get_raw(self, name):
    if not dict.__contains__(self, name):
        dict.__setitem__(self, name, self.owner.mod.get_function(name))
    return dict.__getitem__(self, name)


_PFn.get_raw = _pfn_get_raw


def p_bind(s):
    import cupy as cp
    capN = int(s.capN)
    cap = capN // 8 + 1024
    if getattr(s, "s7b", None) is None or s.s7b.numel() < cap:
        s.s7b = torch.full((cap,), 0x7f800000, device=s.x.device, dtype=torch.int32)
    if getattr(s, "s7v", None) is None:
        s.s7v = torch.zeros(8, device=s.x.device, dtype=torch.int64)
    m = s.mod
    cp.ndarray((1,), dtype=cp.uint64, memptr=m.get_global("S7B"))[0] = int(s.s7b.data_ptr())
    cp.ndarray((1,), dtype=cp.uint64, memptr=m.get_global("S7VIOL"))[0] = int(s.s7v.data_ptr())
    cp.ndarray((1,), dtype=cp.int32, memptr=m.get_global("S7ON"))[0] = int(PP.mode)
    cp.ndarray((1,), dtype=cp.int32, memptr=m.get_global("S7RDY"))[0] = 0
    cp.ndarray((1,), dtype=cp.float32, memptr=m.get_global("S7MARG"))[0] = float(PP.marg)
    if not isinstance(s.fn, _PFn):
        s.fn = _PFn(s.fn, s)


def p_install(out, mode, marg=1e-4, y=0, ym=4, sel=None):
    from . import fem_ao_rt as AORT
    PP.mode, PP.marg, PP.y, PP.ym = int(mode), float(marg), int(y), int(ym)
    if sel:
        PP.sel = tuple(q for q in str(sel).split(",") if q)
    PP.log = open(os.path.join(out, "s7p.jsonl"), "w")
    o_s = AORT._src
    AORT._src = lambda *a, **kw: _p_src(o_s(*a, **kw))
    cls = AORT.AORTSolver
    ob = cls.bind_act

    def bind_act(s, *a, **kw):
        r = ob(s, *a, **kw)
        if "s7_bmin(" in getattr(s, "_s7code", "s7_bmin(") and getattr(s, "mod", None) is not None:
            try:
                p_bind(s)
            except Exception as e_:
                PP.bad.add(("bind", str(e_)[:200], 0))
        return r
    cls.bind_act = bind_act
    print(f"[절 CM72] 장벽 거르기 P 켬 (모드 {PP.mode}, 여유 {PP.marg} m, 커널 {PP.sel if PP.mode else ()}) · 시작점 후보 나눠 셈 Y {PP.y} (앞 {PP.ym})", flush=True)


def p_frame_end(s, fr):
    if PP.log is None or getattr(s, "s7v", None) is None:
        return
    rec = dict(f=fr)
    if fr <= 3 or PP.mode == 2:
        rec["patched"] = sorted(PP.patched) if fr <= 3 else None
        rec["bad"] = [list(b) for b in sorted(PP.bad)] if fr <= 3 else None
        rec["fnwrap"] = isinstance(s.fn, _PFn) if fr <= 3 else None
    if PP.mode == 2 or PP.y == 2:
        torch.cuda.synchronize()
        v = s.s7v.tolist()
        s.s7v.zero_()
        if PP.mode == 2:
            rec.update(viol=v[0], far=v[1], act=v[2])
        if PP.y == 2:
            rec.update(ychk=v[3], ygate=v[4], yviol=v[5])
    if len(rec) > 1:
        PP.log.write(json.dumps({k: v for k, v in rec.items() if v is not None}, ensure_ascii=False) + "\n")
        PP.log.flush()


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
                    if C.p:
                        p_frame_end(s, S6.W.fr)
        return w
    ART.split_step = outer(ART.split_step)
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)


def src_install(out):
    """(SRC) 진단 (--s7src 1, 결과 불변): cupy RawModule 소스 글자를 <out>/s7src_<n>.cu 로 (커널 글자 바꾸기 설계용). RawModule 은 Cython 형이라 이름을 함수로 바꿔 끼움."""
    import cupy as cp
    orig = cp.RawModule
    n_ = [0]

    def RawModule(*a, **kw):
        code = kw.get("code", a[0] if a else None)
        if isinstance(code, str) and n_[0] < 12:
            open(os.path.join(out, f"s7src_{n_[0]}.cu"), "w").write(code)
            n_[0] += 1
        return orig(*a, **kw)
    cp.RawModule = RawModule


def ovf_install(out):
    """(OVF) 진단 (--s7ovf 1, 결과 불변): 새 실행기 넘침 다시 할당 때 (alloc(None) 부름) 넘침 비트 IP[13]·상자 수·용량 -> <out>/s7ovf.jsonl."""
    from . import fem_an_rt as ART
    f = open(os.path.join(out, "s7ovf.jsonl"), "w")
    oa = ART.RTSolver.alloc

    def alloc(s, need, *a, **kw):
        try:
            if need is None and getattr(s, "IP", None) is not None and getattr(s, "capN", None) is not None:
                ip = [int(v) for v in s.IP[:16].tolist()]
                ml = [{k: (int(v) if isinstance(v, (int, np.integer)) else None) for k, v in L.items() if k in ("capn", "capq", "cap")} for L in getattr(s, "ML", [])]
                f.write(json.dumps(dict(ip=ip, capN=int(s.capN), capC=int(s.capC), capQc=int(s.capQc), cap_mul=float(s.cap_mul), ml=ml)) + "\n")
                f.flush()
        except Exception as e_:
            f.write(json.dumps(dict(err=str(e_))) + "\n")
            f.flush()
        return oa(s, need, *a, **kw)
    ART.RTSolver.alloc = alloc


def install(out, kp=None, nm=0, z=0, src=0, ovf=0, p=0, pm=1e-4, y=0, ym=4, pk=None):
    """실행기를 runpy 로 띄우는 순간, speed6 설치 뒤 (가장 바깥) 에 부름."""
    C.out, C.nm, C.z, C.p = out, int(nm), int(z), int(p) or int(y)
    if C.p:
        p_install(out, int(p), pm, y=int(y), ym=int(ym), sel=pk)
    if int(src):
        src_install(out)
    if int(ovf):
        ovf_install(out)
    if C.z:
        LZ.mode = C.z
        lz_install(out)
    if C.z or C.p:
        frame_install()
    if kp:
        C.kp = tuple(int(q) for q in str(kp).split(","))
    if C.nm:
        nm_install()
    if C.kp:
        kp_install()
    print(f"[절 CM72] speed7 켬 프로파일 {C.kp} mfix 진단 끔 {C.nm} 지연 백업 {C.z}", flush=True)
