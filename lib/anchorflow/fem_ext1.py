"""절 CM83 묶음 ext1 (11-08 위임): 공통 후보 H6 = H5 + S_구속 + C + P1X 강성 × κ + T1E2 -- 확장 B-스플라인 T1E2 (절 CM81, lib/anchorflow/fem_wolfh8.py) 를
수박이 쓰는 옛 실행기 (lib/anchorflow/fem_ao_rt.py AORTSolver, exe/fem_ao_rt_sim.py · exe/s6/fem_ao_rt_sim.py) 에 같은 정의로 옮김.

정의 (절 CM81.0 그대로; 새 문턱·클램프 없음): 내부 칸 = 점유 칸 중 26 이웃 칸이 모두 점유 (T1), 내부 노드 = 받침에 내부 칸이 있는 활성 노드, 외부 노드 o 는
블록 가운데 칸 거리 ≤ 2 인 가장 가까운 내부 칸의 27 노드에 축별 2 차 라그랑주 외삽 (E2) 으로 묶음 (뉴턴 걸음 증분 아핀 확장, y_o − y0_o = Σ_j c_oj (y_j − y0_j)),
그런 칸이 없으면 고립 (원래 자유도). 커널 글자는 fem_wolfh8.W8_SRC 그대로 (w8_icell2 · w8_class · w8_free · w8_fold · w8_fold2 · w8_expand) + 옛 실행기 경로에만 있는 자리를 위한 커널 (아래 X1_SRC).

옛 실행기 훅 (AORTSolver 의 alloc · k · ctl 감쌈; 새 실행기 판 wolfh8 과 같은 자리 + 옛 실행기에만 있는 자리):
  - 걸음 시작 ctl(1): 분류 + 외부 노드 변수 바꿈 항등 (w8_free) -- 새 실행기와 같음.
  - rt_grad(_c) 뒤: 잔차 접기 (w8_fold·w8_fold2) + |g|² 다시 (x1_gg, 압축 판은 AO_ACT 목록) -- 같음.
    + 블록 야코비 판 (MG 층 1 = 수박): rt_grad 가 접기 전 r 로 만든 z = D⁻¹ r · p = z · Tp = T z · r·z (SC[0]) 를 접은 r 로 다시 (x1_jz, rt_grad 와 같은 식) 하고 p 펼치기.
      (새 실행기 판은 MG 가 z·p 를 다시 써서 이것이 필요 없었음. 외부 노드 r_o = 0 -> z_o = 0 -> 전처리 = 줄어든 문제 주 부분행렬 블록 대각.)
  - rt_pinit · rt_cg2(_c) 뒤: 방향 펼치기 (w8_expand) -- 같음.  rt_ap(_c) 뒤: 헤시안 곱 접기 -- 같음.
  - PCG 첫 추정 (ao_tvec(d0 -> td) · ao_warm, 쪼갠 걸음 둘째부터): d0 를 지금 분류로 펼친 뒤 (td 도) 곱하고, ao_warm 뒤 잔차 접기 + x1_jz + 펼치기.
  - 노드별 보폭 (절 CM30 (b)·CM32 --ws_bar --ws_fi, 수박): 노드 k_n 이 달라 외부 노드를 자기 k_o 로 움직이면 확장 관계가 깨짐 ->
      ws_cand 뒤 외부 노드 후보를 주인 노드 실제 보폭으로 (y1_o = y_o + Σ c_oj 2^-k_j (T δ)_j, 방향 dd 같은 식; x1_wscand),
      ws_merge 뒤 외부 노드 k_o 를 주인 노드에 올림 (k_j = max(k_j, k_o) -- 외부 노드에 걸린 표본점·가우스 점의 27 노드 규칙을 확장 사상으로 옮긴 것; x1_kprop),
      ws_lsup 뒤 외부 노드 갱신을 주인 노드 실제 증분으로 (η_o += Σ c 2^-k_j (T δ)_j − 2^-k_o δ_o, y_o = ŷ_o + η_o; x1_wslsup). 되돌림 (전역 보폭) 은 모두 같은 k 라 원래와 같음.
  - 기록: 걸음 시작마다 분류 수·질량을 그래프 안에서 장치 버퍼에 (x1_log, 호스트 동기화 없음) -> 끝에서 w8.jsonl. 새 실행기는 fem_wolfh8 훅 그대로 + 프레임마다
    호스트 읽기 (step 감쌈) 를 빼고 같은 그래프 안 기록으로 (결과 불변; 단독 측정에 동기화가 끼지 않게).
검사 (--x1ck 1, 그래프 밖 record = 캡처 앞 데움 실행에서만; 결과 불변, 상태는 되돌림) -> x1ck.jsonl:
  C1 펼치기·접기 GPU 커널 = 파이썬 E·Eᵀ, 수반 u·(Eᵀ v) = (E u)·v.  C2 접은 잔차 = Eᵀ r0, SC[1] = |r|².  C3 (야코비) z = D⁻¹ r, z_o = 0, SC[0] = r·z, p_o = Σ c (T p)_j.
  C4 줄어든 헤시안 곱 (실제 곱 사슬 + 접기) 대칭·양정치·외부 몫 0.  C5 줄어든 기울기 중심 차분 (기울기 사슬 전체 다시: 요소·벌점 훅·장벽·rt_grad·접기) 대 C4 곱 (원래 곱 대 차분도 같이).
  C6 확장 관계 y_o − y0_o = Σ c (y_j − y0_j) 를 뉴턴 갱신 (rt_lsup · ws_lsup · en_bt) 마다.  C7 (MG 판) r_o = 0, r·z > 0, p_o 펼침.
"""
import atexit
import json
import os
import weakref

import numpy as np
import torch

ST = dict(D=2, T=1, X=0, ck=0, out=".", nsrc=0, nlog_src=0, nalloc=0, nclass=0, nfold=0, nexp=0, ngg=0, njz=0, nwarm=0, nwscand=0, nkprop=0, nwslsup=0, nlog=0,
          nck=0, ninv=0, said=False, err=None, ck_err=None)
MARK = "// ---- 절 CM83 ext1"
MARKL = "// ---- 절 CM83 ext1 기록"
HCAP = 8192                                                                            # 기록 줄 상한 (걸음 시작 수)
_SOL = []

X1_LOG = r'''
// ---- 절 CM83 ext1 기록: 걸음 시작 분류 수·질량을 장치 버퍼에 (그래프 안), 다음 걸음을 위해 0
__global__ void x1_log(const int* IP, int* cnt, float* cf, int* hp, int* H, float* HF, int cap, int ks, int K) {
    int s = hp[0]; hp[0] = s + 1;
    if (s < cap) {
        for (int i=0;i<12;i++) H[s*16 + i] = cnt[i];
        H[s*16 + 12] = ks; H[s*16 + 13] = K; H[s*16 + 14] = IP[11]; H[s*16 + 15] = IP[9];
        for (int i=0;i<4;i++) HF[s*4 + i] = cf[i];
    }
    for (int i=0;i<16;i++) cnt[i] = 0;
    for (int i=0;i<4;i++) cf[i] = 0.f;
}
'''

X1_SRC = r'''
// ---- 절 CM83 ext1: 옛 실행기 자리 (lib/anchorflow/fem_ext1.py)
// 블록 야코비: 접은 r 로 z = D⁻¹ r, p = z, Tp = T z, SC[0] += r·z (rt_grad 와 같은 식; cmp 1 = 압축 판 AO_ACT 목록)
__global__ void x1_jz(const int* IP, const float* Dinv, const float* sk, const int* pm, float n0, float n1, float n2, const float* r, float* z, float* p, float* Tp, double* SC, int cmp) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int lim = cmp ? IP[14] : IP[9];
    if (blockIdx.x*blockDim.x >= lim) return;
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    double rz = 0.0;
    if (ii < lim) {
        int n = cmp ? AO_ACT[ii] : ii;
        const float* D = Dinv + n*9;
        float rr[3] = {r[n*3], r[n*3+1], r[n*3+2]};
        float zz[3];
        for (int a=0;a<3;a++) { zz[a] = D[a*3]*rr[0] + D[a*3+1]*rr[1] + D[a*3+2]*rr[2]; z[n*3+a] = zz[a]; p[n*3+a] = zz[a]; rz += (double)rr[a]*zz[a]; }
        float tp[3]; Tapply(sk[n], pm[n], n0, n1, n2, zz, tp); for (int a=0;a<3;a++) Tp[n*3+a] = tp[a];
    }
    blocksum_add(rz, &SC[0]);
}
// |g|² = |접은 r|² (SC[1]); cmp 1 = AO_ACT 목록 (압축 판은 빈 노드 r 을 안 씀)
__global__ void x1_gg(const int* IP, const float* r, double* SC, int cmp) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int lim = cmp ? IP[14] : IP[9];
    if (blockIdx.x*blockDim.x >= lim) return;
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    double s = 0.0;
    if (ii < lim) { int n = cmp ? AO_ACT[ii] : ii; for (int d=0;d<3;d++) { double q = r[n*3+d]; s += q*q; } }
    blocksum_add(s, &SC[1]);
}
// 노드별 보폭 후보: 외부 노드 = 주인 노드 실제 보폭 증분의 외삽 (y1_o = y_o + Σ c 2^-k_j td_j, dd_o 같은 값)
__global__ void x1_wscand(const int* IP, const int* base, const float* w, const int* kn, const float* td, const float* y, float* y1, float* dd) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    const float* W = w + n*9;
    float t[3] = {0.f, 0.f, 0.f};
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c; float sq = ldexpf(1.f, -kn[q]);
          for (int d=0;d<3;d++) t[d] += ww*sq*td[q*3+d]; } } }
    for (int d=0;d<3;d++) { dd[n*3+d] = t[d]; y1[n*3+d] = y[n*3+d] + t[d]; }
}
// 외부 노드 k_o 를 주인 노드에 (k_j = max(k_j, k_o))
__global__ void x1_kprop(const int* IP, const int* base, const float* w, int* kn) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n], k = kn[n];
    const float* W = w + n*9;
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { if (W[a]*W[3+b]*W[6+c] == 0.f) continue;
          atomicMax(&kn[b0 + (a*nb1 + b)*nb2 + c], k); } } }
}
// 노드별 보폭 갱신 뒤: 외부 노드 증분 = 주인 노드 실제 증분의 외삽 (T_o = I)
__global__ void x1_wslsup(const int* IP, const int* base, const float* w, const int* kn, const float* td, const float* dl, const float* yh, float* eta, float* y) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || base[n] < 0) return;
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    const float* W = w + n*9;
    float t[3] = {0.f, 0.f, 0.f};
    for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;
      for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;
        for (int c=0;c<3;c++) { float ww = W[a]*W[3+b]*W[6+c]; if (ww == 0.f) continue;
          int q = b0 + (a*nb1 + b)*nb2 + c; float sq = ldexpf(1.f, -kn[q]);
          for (int d=0;d<3;d++) t[d] += ww*sq*td[q*3+d]; } } }
    float so = ldexpf(1.f, -kn[n]);
    for (int d=0;d<3;d++) { float e = eta[n*3+d] + t[d] - so*dl[n*3+d]; eta[n*3+d] = e; y[n*3+d] = yh[n*3+d] + e; }
}
'''


def P(t):
    return np.uint64(0 if t is None else t.data_ptr())


def _ins(src, body, mark):
    if mark in src:
        return src
    assert "Tapply(" in src and "blocksum_add(" in src, "ext1: Tapply·blocksum_add 없는 모듈"
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + body + "\n}\n"


def _add_ao(src):
    if MARK in src:
        return src
    from . import fem_wolfh8 as W8
    assert "__device__ int* AO_ACT;" in src, "ext1: 옛 실행기 소스 (AO_ACT) 가 아님"
    ST["nsrc"] += 1
    return _ins(src, W8.W8_SRC + X1_SRC + X1_LOG, MARK)


def _add_log(src):
    if MARKL in src:
        return src
    ST["nlog_src"] += 1
    return _ins(src, X1_LOG, MARKL)


def _hist(s):
    if getattr(s, "x1H", None) is None:
        s.x1H = torch.zeros(HCAP, 16, device=s.dev, dtype=torch.int32)
        s.x1HF = torch.zeros(HCAP, 4, device=s.dev, dtype=torch.float32)
        s.x1hp = torch.zeros(1, device=s.dev, dtype=torch.int32)
        _SOL.append(weakref.ref(s))


def _log(s, ok):
    _hist(s)
    ks = int(getattr(s, "_x1ks", 0))
    ok(s, "x1_log", 1, (P(s.IP), P(s.w8cnt), P(s.w8cf), P(s.x1hp), P(s.x1H), P(s.x1HF), np.int32(HCAP), np.int32(ks), np.int32(int(getattr(s, "K", 1)))), 1)
    s._x1ks = ks + 1
    ST["nlog"] += 1


# ------------------------------------------------------------------ 새 실행기: wolfh8 그대로 + 그래프 안 기록
def _install_an(out, D, T, X):
    from . import fem_an_rt as ART
    from . import fem_wolfh8 as W8
    R = ART.RTSolver
    st0 = R.step
    W8.install(out, D=D, T=T, X=X)
    R.step = st0                                                                       # wolfh8 의 프레임마다 호스트 읽기 (w8cnt.tolist) 를 뺌 -- 기록은 아래 그래프 안
    ob = ART.build_src
    ART.build_src = lambda *a, **kw: _add_log(ob(*a, **kw))
    oc, ok = R.ctl, R.k

    def ctl(s, op, it=0):
        r = oc(s, op, it)
        if op == 0:
            s._x1ks = 0
        elif op == 1 and W8._on(s):
            _log(s, ok)
        return r
    R.ctl = ctl


# ------------------------------------------------------------------ 옛 실행기
def _install_ao():
    from . import fem_ao_rt as AORT
    oas = AORT._src
    AORT._src = lambda *a, **kw: _add_ao(oas(*a, **kw))
    C = AORT.AORTSolver
    oa, ok, oc = C.alloc, C.k, C.ctl
    i32 = np.int32

    def on(s):
        return getattr(s, "w8b", None) is not None and getattr(s, "_x1on", False)

    def alloc(s, need):
        r = oa(s, need)
        dev = s.dev
        s.w8ic = torch.zeros(s.capC, device=dev, dtype=torch.int32)
        s.w8b = torch.full((s.capN,), -1, device=dev, dtype=torch.int32)
        s.w8w = torch.zeros(s.capN * 9, device=dev, dtype=torch.float32)
        s.w8acc = torch.zeros(s.capN * 3, device=dev, dtype=torch.float32)
        s.w8cnt = torch.zeros(16, device=dev, dtype=torch.int32)
        s.w8cf = torch.zeros(4, device=dev, dtype=torch.float32)
        s._x1on = True
        ST["nalloc"] += 1
        return r

    def cmp(s):
        return i32(1 if getattr(s, "compact", False) else 0)

    def fold(s, v, sk, pm, nr):
        ok(s, "w8_fold", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), v, P(s.w8acc)), 256)
        ok(s, "w8_fold2", s.capN, (P(s.IP), sk, pm, *nr, P(s.w8acc), v), 256)
        ST["nfold"] += 1

    def expand(s, p, Tp):
        ok(s, "w8_expand", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), p, Tp), 256)
        ST["nexp"] += 1

    def jz(s, Dinv, sk, pm, nr, r, z, p, Tp, SC):
        ok(s, "rt_sc_clear", 1, (SC, i32(0), i32(1)), 1)
        ok(s, "x1_jz", s.capN, (P(s.IP), Dinv, sk, pm, *nr, r, z, p, Tp, SC, cmp(s)), 256)
        ST["njz"] += 1
        expand(s, p, Tp)

    def k(s, name, n, args, b=128):
        if on(s) and ST["ck"] and not getattr(s, "_x1in", False) and name in ("aq_elem", "ao_elem_s", "ao_elem", "rt_bar_q", "rt_bar_n"):
            s._x1rec = getattr(s, "_x1rec", {}); s._x1rec[name] = tuple(args)
        r = ok(s, name, n, args, b)
        if not on(s):
            return r
        if name in ("rt_grad", "rt_grad_c"):                                           # IP mn it2 sk pm n0 n1 n2 eta f B Dinv r z p dl Tp SC r0
            nr = tuple(args[5:8])
            fold(s, args[12], args[3], args[4], nr)
            ok(s, "rt_sc_clear", 1, (args[17], i32(1), i32(2)), 1)
            ok(s, "x1_gg", s.capN, (P(s.IP), args[12], args[17], cmp(s)), 256)
            ST["ngg"] += 1
            if getattr(s, "nl", 1) <= 1:                                               # 블록 야코비: z·p·Tp·r·z 를 접은 r 로 다시 (MG 판은 V 순환 + rt_pinit 이 다시 씀)
                jz(s, args[11], args[3], args[4], nr, args[12], args[13], args[14], args[16], args[17])
            if ST["ck"] and not getattr(s, "_x1in", False):
                s._x1g = dict(name=name, args=tuple(args))
                _check_iter(s, ok)
        elif name in ("rt_ap", "rt_ap_c"):                                             # IP mn it2 sk pm n0 n1 n2 p KTp Ap SC [Kb]
            fold(s, args[10], args[3], args[4], tuple(args[5:8]))
        elif name == "rt_pinit":                                                       # IP z sk pm n0 n1 n2 p Tp
            expand(s, args[7], args[8])
            if ST["ck"] and not getattr(s, "_x1in", False):
                _check_mg(s)
        elif name in ("rt_cg2", "rt_cg2_c"):                                           # IP SC sk pm n0 n1 n2 z p Tp [KTz]
            expand(s, args[8], args[9])
        elif name == "ao_tvec":                                                        # IP sk pm n0 n1 n2 v o: PCG 첫 추정 d0 -> td 면 d0 를 지금 분류로 펼침
            if int(args[6]) == int(s.d0.data_ptr()):
                expand(s, args[6], args[7])
        elif name == "ao_warm":                                                        # IP mn it2 sk pm n0 n1 n2 d0 KTd Dinv dl r z p Tp SC
            nr = tuple(args[5:8])
            fold(s, args[12], args[3], args[4], nr)
            jz(s, args[10], args[3], args[4], nr, args[12], args[13], args[14], args[15], args[16])
            ST["nwarm"] += 1
        elif name == "ws_cand":                                                        # IP kn sk pm n0 n1 n2 yh eta dl td y1 dd
            ok(s, "x1_wscand", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), args[1], args[10], P(s.y), args[11], args[12]), 256)
            ST["nwscand"] += 1
        elif name == "ws_merge":                                                       # IP kn kn2
            ok(s, "x1_kprop", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), args[1]), 256)
            ST["nkprop"] += 1
        elif name == "ws_lsup":                                                        # IP kn sk pm n0 n1 n2 yh eta dl y cmp
            ok(s, "x1_wslsup", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), args[1], P(s.awtd), args[9], args[7], args[8], args[10]), 256)
            ST["nwslsup"] += 1
            if ST["ck"] and not getattr(s, "_x1in", False):
                _check_inv(s, "ws_lsup")
        elif name in ("rt_lsup", "rt_lsup_c", "en_bt_fin"):
            if ST["ck"] and not getattr(s, "_x1in", False):
                _check_inv(s, name)
        return r

    def ctl(s, op, it=0):
        r = oc(s, op, it)
        if op == 0:
            s._x1ks = 0
        if op == 1 and on(s):
            ok(s, "rt_zero_i", s.capC, (i32(s.capC), P(s.w8ic)), 256)
            if ST["T"] in (1, 2):
                ok(s, "w8_icell2", s.capQc, (P(s.IP), P(s.cb), P(s.occ), P(s.w8ic), P(s.w8cnt), i32(1 if ST["T"] == 2 else 0)), 128)
            else:
                ok(s, "w8_icell", s.capQc, (P(s.IP), P(s.cb), P(s.Aacc), P(s.w8ic), P(s.w8cnt)), 128)
            ok(s, "w8_class", s.capN, (P(s.IP), P(s.mn), P(s.w8ic), i32(ST["D"]), P(s.w8b), P(s.w8w), P(s.w8cnt), P(s.w8cf), i32(ST["X"])), 256)
            ok(s, "w8_free", s.capN, (P(s.IP), P(s.w8b), P(s.sk), P(s.pm), P(s.yh), P(s.y), P(s.eta)), 256)
            ST["nclass"] += 1
            _log(s, ok)
            if ST["ck"] and not torch.cuda.is_current_stream_capturing():
                s._x1y0 = s.y.clone()
            if not ST["said"]:
                ST["said"] = True
                print(f"[절 CM83 ext1] 옛 실행기 확장 훅 -- 차수 {ST['D']} 내부 칸 정의 {ST['T']} 고립 {ST['X']}, 걸음 시작 분류·변수 바꿈 항등, rt_grad 접기 + |g|² 다시"
                      f"{' + 야코비 z·p 다시' if getattr(s, 'nl', 1) <= 1 else ''}, rt_pinit·rt_cg2 펼치기, rt_ap 접기, 첫 추정 d0 펼치기·ao_warm 접기, "
                      f"노드별 보폭 (ws_cand·ws_merge·ws_lsup) 외부 노드 = 주인 노드 실제 증분 (MG 층 {getattr(s, 'nl', 1)}, 압축 {bool(getattr(s, 'compact', False))}, 검사 {ST['ck']})", flush=True)
                _dump()
        return r
    C.alloc, C.k, C.ctl = alloc, k, ctl


def install(out, D=2, T=1, X=0, ck=0):
    """실행기 runpy 순간 (다른 감싸개 다 설치 뒤, 가장 바깥). 두 실행기 모두 (그 판이 쓰는 쪽만 걸림)."""
    ST.update(D=int(D), T=int(T), X=int(X), ck=int(ck), out=out)
    _install_an(out, ST["D"], ST["T"], ST["X"])
    _install_ao()
    print(f"[절 CM83] ext1 켬 -- 확장 B-스플라인 차수 {ST['D']} 내부 칸 정의 {ST['T']} (새 실행기 = fem_wolfh8 훅, 옛 실행기 = fem_ext1 훅), 검사 {ST['ck']}", flush=True)
    atexit.register(_fin)


def _fin():
    rows = []
    for wr in _SOL:
        s = wr()
        if s is None or getattr(s, "x1H", None) is None:
            continue
        try:
            torch.cuda.synchronize()
            n = min(int(s.x1hp.item()), HCAP)
            H = s.x1H[:n].cpu().tolist(); HF = s.x1HF[:n].cpu().tolist()
            for i in range(n):
                c, m = H[i], HF[i]
                mt = sum(m[:3])
                rows.append(dict(seq=i, K=c[13], ks=c[12], icell=c[3], ncell=c[14], nn=c[15], inner=c[0], outer=c[1], iso=c[2], ax1=c[5], ax2=c[6], ax3=c[7], xiso=c[8],
                                 m_inner=m[0], m_outer=m[1], m_iso=m[2], m_xiso=m[3], mf_outer=(m[1] / mt if mt > 0 else None), mf_iso=(m[2] / mt if mt > 0 else None)))
        except Exception as e_:
            ST["err"] = f"fin {type(e_).__name__}: {e_}"
    try:
        with open(os.path.join(ST["out"], "w8.jsonl"), "w") as f_:
            for q in rows:
                f_.write(json.dumps(q) + "\n")
    except Exception as e_:
        ST["err"] = f"w8.jsonl {type(e_).__name__}: {e_}"
    ST["nrows"] = len(rows)
    _dump()


def _dump():
    try:
        with open(os.path.join(ST["out"], "x1info.json"), "w") as f_:
            json.dump({k: v for k, v in ST.items()}, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


# ------------------------------------------------------------------ 검사 (그래프 밖 record 만, 결과 불변)
NCK, NINV = 24, 400


def _ckw(d):
    try:
        with open(os.path.join(ST["out"], "x1ck.jsonl"), "a") as f_:
            f_.write(json.dumps(d, default=float) + "\n")
    except Exception:
        pass


def _skip(s):
    if torch.cuda.is_current_stream_capturing():
        return True
    torch.cuda.synchronize()
    if not getattr(s, "adapt", False):
        return False
    import cupy as cp
    g = s.mod.get_global("AF_SKIP")
    return int(cp.ndarray((1,), dtype=cp.int32, memptr=g)[0]) != 0


def _v3(t, nn):
    return t.reshape(-1)[:nn * 3].view(nn, 3)


def _T(v, sk, pm, nr):
    n_ = torch.tensor([float(q) for q in nr], device=v.device, dtype=v.dtype)
    vn = (v * n_).sum(-1, keepdim=True)
    skc = sk[:, None].to(v.dtype)
    o1 = skc * (v - vn * n_) + vn * n_
    o3 = v - (1 - skc) * vn * n_
    pmc = pm[:, None]
    o = torch.where(pmc == 0, skc * v, o1)
    o = torch.where(pmc == 3, o3, o)
    return torch.where(pmc == 2, torch.zeros_like(v), o)


def _emap(s):
    nn = int(s.IP[9]); nb1, nb2 = int(s.IP[7]), int(s.IP[8])
    b = s.w8b[:nn].long()
    ext = (b >= 0).nonzero().squeeze(1)
    W = s.w8w[:s.capN * 9].view(-1, 9)[ext].double()
    ia = torch.arange(3, device=s.dev)
    A, B_, C_ = torch.meshgrid(ia, ia, ia, indexing="ij")
    A, B_, C_ = A.reshape(-1), B_.reshape(-1), C_.reshape(-1)
    ww = W[:, A] * W[:, 3 + B_] * W[:, 6 + C_]
    q = b[ext][:, None] + ((A * nb1 + B_) * nb2 + C_)[None, :]
    return nn, ext, q, ww


def _E(u, ext, q, ww, sk, pm, nr):
    """η 꼴 펼치기: 외부 노드 = Σ c (T u)_j (외부 T = I)."""
    Tu = _T(u, sk, pm, nr)
    o = u.clone()
    o[ext] = (ww[:, :, None] * Tu[q]).sum(1)
    return o


def _Et(v, ext, q, ww, sk, pm, nr):
    """접기 Eᵀ: v_j += T_j Σ c v_o, v_o = 0."""
    acc = torch.zeros_like(v)
    acc.index_add_(0, q.reshape(-1), (ww[:, :, None] * v[ext][:, None, :]).reshape(-1, 3))
    o = v.clone()
    o[ext] = 0
    return o + _T(acc, sk, pm, nr)


def _rel(a, b):
    return float((a - b).norm() / max(float(b.norm()), 1e-300))


def _gpu_expand(s, ok, u, sk, pm, nr):
    nn = u.shape[0]
    p = torch.zeros(s.capN, 3, device=s.dev, dtype=torch.float32); Tp = torch.zeros_like(p)
    p[:nn] = u.float(); Tp[:nn] = _T(u, sk, pm, nr).float()
    ok(s, "w8_expand", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), P(p), P(Tp)), 256)
    return p, Tp


def _gpu_fold(s, ok, v, nr):
    nn = v.shape[0]
    t = torch.zeros(s.capN, 3, device=s.dev, dtype=torch.float32)
    t[:nn] = v.float()
    ok(s, "w8_fold", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), P(t), P(s.w8acc)), 256)
    ok(s, "w8_fold2", s.capN, (P(s.IP), P(s.sk), P(s.pm), *nr, P(s.w8acc), P(t)), 256)
    return t


def _check_iter(s, ok):
    if ST["nck"] >= NCK or _skip(s):
        return
    s._x1in = True
    s._g3chk_bk = getattr(s, "_g3chk", False); s._g3chk = True
    out = dict(kind="iter", K=int(getattr(s, "K", 1)), ks=int(getattr(s, "_x1ks", 0)) - 1)
    try:
        torch.cuda.synchronize()
        a = s._x1g["args"]; nr = tuple(a[5:8]); it2 = a[2]
        nn, ext, q, ww = _emap(s)
        out.update(nn=nn, next=int(ext.numel()))
        if nn <= 0 or ext.numel() == 0:
            _ckw(out); return
        sk, pm, mn = s.sk[:nn], s.pm[:nn], s.mn[:nn]
        act = mn > 0
        nonext = act.clone(); nonext[ext] = False
        g = torch.Generator(device=s.dev).manual_seed(77 + ST["nck"])
        rnd = lambda m: torch.randn(nn, 3, device=s.dev, generator=g, dtype=torch.float64) * m[:, None]
        # C1 펼치기·접기 커널 = 파이썬 E·Eᵀ, 수반
        u, v = rnd(nonext), rnd(act)
        pE, _ = _gpu_expand(s, ok, u, sk, pm, nr)
        Eu = _E(u, ext, q, ww, sk, pm, nr)
        Fv = _Et(v, ext, q, ww, sk, pm, nr)
        Fg = _gpu_fold(s, ok, v, nr)
        out["C1"] = dict(expand_rel=_rel(pE[:nn].double(), Eu), fold_rel=_rel(Fg[:nn].double(), Fv),
                         adj=abs(float((u * Fv).sum()) - float((Eu * v).sum())) / max(abs(float((Eu * v).sum())), 1e-300))
        # C2 접은 잔차 = Eᵀ r0, SC[1] = |r|²
        r0 = _v3(s.g0buf, nn).double(); r = _v3(s.r, nn).double()
        rr = r[act]
        out["C2"] = dict(fold_r0_rel=_rel(r[act], _Et(r0, ext, q, ww, sk, pm, nr)[act]), r_ext_max=float(r[ext].abs().max()),
                         gg_rel=abs(float(s.SC[1]) - float((rr * rr).sum())) / max(float((rr * rr).sum()), 1e-300), g_unf=float(r0[act].norm()), g_red=float(rr.norm()))
        # C3 야코비
        if getattr(s, "nl", 1) <= 1:
            D = s.Dinv.reshape(-1)[:nn * 9].view(nn, 3, 3).double()
            z = _v3(s.z, nn).double(); p = _v3(s.p, nn).double(); Tp = _v3(s.Tp, nn).double()
            zr = torch.einsum("nab,nb->na", D, r)
            pe = p.clone(); pe[ext] = (ww[:, :, None] * Tp[q]).sum(1)
            out["C3"] = dict(z_rel=_rel(z[act], zr[act]), z_ext_max=float(z[ext].abs().max()), rz_rel=abs(float(s.SC[0]) - float((r[act] * z[act]).sum())) / max(abs(float((r[act] * z[act]).sum())), 1e-300),
                             rz=float(s.SC[0]), p_ext_rel=_rel(p[ext], pe[ext]), Tp_rel=_rel(Tp[act], _T(p, sk, pm, nr)[act]))
        # C4 줄어든 헤시안 곱 (실제 곱 사슬 + 접기)
        cell = float(s.A[:, 0].norm()) if hasattr(s, "A") else 1.0
        u2, v2 = rnd(nonext), rnd(nonext)
        Au, Ap_full_u = _hv(s, ok, u2, sk, pm, nr, it2)
        Av, _ = _hv(s, ok, v2, sk, pm, nr, it2)
        vAu, uAv = float((v2 * Au).sum()), float((u2 * Av).sum())
        out["C4"] = dict(sym=abs(vAu - uAv) / max(abs(vAu) + abs(uAv), 1e-300), uAu=float((u2 * Au).sum()), vAv=float((v2 * Av).sum()), Ap_ext_max=float(Au[ext].abs().max()),
                         fold_rel=_rel(Au[act], _Et(Ap_full_u, ext, q, ww, sk, pm, nr)[act]))
        # C5 줄어든 기울기 중심 차분 (기울기 사슬 전체) 대 C4 곱
        Eu2 = _E(u2, ext, q, ww, sk, pm, nr)
        sc = cell / max(float(Eu2.norm(dim=-1).max()), 1e-30)
        c5 = {}
        for ep in (1e-3, 1e-4):
            h = ep * sc
            gp, g0p = _grad_at(s, ok, Eu2 * h, nr)
            gm, g0m = _grad_at(s, ok, -Eu2 * h, nr)
            if gp is None or gm is None:
                c5[f"e{ep:g}"] = "기울기 사슬 기록 없음"; continue
            jr = (gp - gm) / (2 * h)
            jf = (g0p - g0m) / (2 * h)
            c5[f"e{ep:g}"] = dict(red_rel=_rel(jr[act], Au[act]), red_cos=float((jr[act] * Au[act]).sum() / max(float(jr[act].norm() * Au[act].norm()), 1e-300)),
                                  full_rel=_rel(jf[act], Ap_full_u[act]), dir=float((u2 * jr).sum()) / max(abs(float((u2 * Au).sum())), 1e-300))
        out["C5"] = c5
        ST["nck"] += 1
    except Exception as e_:
        import traceback
        ST["ck_err"] = f"{type(e_).__name__}: {e_} | {traceback.format_exc()[-600:]}"
        out["err"] = ST["ck_err"]
    finally:
        s._x1in = False
        s._g3chk = s._g3chk_bk
    _ckw(out)


def _hv(s, ok, u, sk, pm, nr, it2):
    """줄어든 헤시안 곱 A_red u = Eᵀ A (E u) 를 실제 곱 사슬 (mv 커널 + 벌점 훅 + 장벽 곱 + rt_ap) + 접기로; 원래 A (E u) 도 (rt_ap 접기 전 사본)."""
    nn = u.shape[0]
    p, Tp = _gpu_expand(s, ok, u, sk, pm, nr)
    KTp = torch.zeros(s.capN, 3, device=s.dev, dtype=torch.float32); Ap = torch.zeros_like(KTp)
    capQ = s.capQc * 8
    SCb = s.SC.clone()
    s.k(s.mvname(), capQ, (P(s.IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.H), P(Tp), P(KTp), np.int32(capQ)) + ((np.float32(2.0 * s.mu + s.lam),) if getattr(s, "h16", False) else ()))
    if getattr(s, "wbar", None) is not None:
        s.k("rt_bar_mv", s.capN, (P(s.IP), P(s.bKn), *nr, P(Tp), P(KTp)), 256)
    nm = "rt_ap_c" if getattr(s, "compact", False) else "rt_ap"
    ok(s, nm, s.capN, (P(s.IP), P(s.mn), it2, P(s.sk), P(s.pm), *nr, P(p), P(KTp), P(Ap), P(s.SC)), 256)    # 원래 곱 (접기 전)
    full = _v3(Ap, nn).double().clone()
    ok(s, "w8_fold", s.capN, (P(s.IP), P(s.w8b), P(s.w8w), P(Ap), P(s.w8acc)), 256)
    ok(s, "w8_fold2", s.capN, (P(s.IP), P(s.sk), P(s.pm), *nr, P(s.w8acc), P(Ap)), 256)
    torch.cuda.synchronize()
    s.SC.copy_(SCb)
    return _v3(Ap, nn).double().clone(), full


_BK = ("y", "eta", "f", "B", "H", "r", "z", "p", "dl", "Tp", "Dinv", "g0buf", "SC", "bGn", "bKn", "bE", "qb", "cnt")


def _grad_at(s, ok, deta, nr):
    """η + deta (외부 노드는 펼친 증분) 에서 기울기 사슬 (요소 커널 + 벌점 훅 · 장벽 · rt_grad + 접기 훅) 을 다시 돌려 (접은 g, 원래 g) -> 상태 되돌림."""
    R = getattr(s, "_x1rec", {}) or {}
    en = next((n for n in ("aq_elem", "ao_elem_s", "ao_elem") if n in R), None)
    if en is None:
        return None, None
    nn = deta.shape[0]
    bk = {n: getattr(s, n).clone() for n in _BK if torch.is_tensor(getattr(s, n, None))}
    try:
        _v3(s.eta, nn).add_(deta.float())                                              # y = ŷ + T η 이므로 y += T Δη (다른 노드는 그대로)
        _v3(s.y, nn).add_(_T(deta, s.sk[:nn], s.pm[:nn], nr).float())
        s.f.zero_(); s.B.zero_()
        ok(s, "rt_sc_clear", 1, (P(s.SC), np.int32(0), np.int32(16)), 1)
        if getattr(s, "adapt", False):
            ok(s, "rt_sc_clear", 1, (P(s.SC), np.int32(56), np.int32(58)), 1)
        a = list(R[en])
        if en == "aq_elem":
            a[14] = P(None); a[18] = P(None)
        elif len(a) > 14:
            a[14] = P(None)
        s.k(en, s.capQc * 8, tuple(a))                                                 # 벌점 훅 (ghost6·lightg) 이 이 이름 뒤에 붙음
        if "rt_bar_q" in R and getattr(s, "wbar", None) is not None:
            s.bGn.zero_(); s.bKn.zero_()
            s.k("rt_bar_q", s.capQc * 8, R["rt_bar_q"])
            s.k("rt_bar_n", s.capN, R["rt_bar_n"], 256)
        g = s._x1g
        ok(s, g["name"], s.capN, g["args"], 256)                                       # 원래 rt_grad (접기 전)
        g0 = -_v3(s.g0buf, nn).double().clone()
        k_ = type(s).k
        k_(s, g["name"], s.capN, g["args"], 256)                                       # 이 모듈 훅까지 (접기) -- 같은 입력이라 같은 원래 값 + 접기
        torch.cuda.synchronize()
        return -_v3(s.r, nn).double().clone(), g0
    finally:
        for n, t in bk.items():
            getattr(s, n).copy_(t)
        torch.cuda.synchronize()


def _check_inv(s, where):
    if ST["ninv"] >= NINV or getattr(s, "_x1y0", None) is None or _skip(s):
        return
    try:
        torch.cuda.synchronize()
        nn, ext, q, ww = _emap(s)
        if ext.numel() == 0:
            return
        d = _v3(s.y, nn).double() - _v3(s._x1y0, nn).double()
        pred = (ww[:, :, None] * d[q]).sum(1)
        dev = float((d[ext] - pred).norm(dim=-1).max())
        _ckw(dict(kind="inv", at=where, K=int(getattr(s, "K", 1)), ks=int(getattr(s, "_x1ks", 0)) - 1, next=int(ext.numel()), dev=dev, dmax=float(d.norm(dim=-1).max()),
                  rel=dev / max(float(d.norm(dim=-1).max()), 1e-300)))
        ST["ninv"] += 1
    except Exception as e_:
        ST["ck_err"] = f"inv {type(e_).__name__}: {e_}"


def _check_mg(s):
    if ST["nck"] >= NCK or _skip(s):
        return
    try:
        torch.cuda.synchronize()
        nn, ext, q, ww = _emap(s)
        if ext.numel() == 0:
            return
        r = _v3(s.r, nn).double(); z = _v3(s.z, nn).double(); p = _v3(s.p, nn).double(); Tp = _v3(s.Tp, nn).double()
        act = s.mn[:nn] > 0
        pe = (ww[:, :, None] * Tp[q]).sum(1)
        _ckw(dict(kind="mg", next=int(ext.numel()), r_ext_max=float(r[ext].abs().max()), rz=float((r[act] * z[act]).sum()), z_ext_max=float(z[ext].abs().max()), p_ext_rel=_rel(p[ext], pe)))
    except Exception as e_:
        ST["ck_err"] = f"mg {type(e_).__name__}: {e_}"
