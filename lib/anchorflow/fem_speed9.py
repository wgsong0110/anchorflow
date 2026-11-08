"""절 CM76 묶음 speed9 (11-08 위임, 수박 렌더 포함 벽시계 60 fps 이어서): 공통 후보 H3 (= H2 + W + X3g, 절 CM74) 위 진단·가속·후보. 감싸개 exe/speed9_wrap.py 가 부름.

(D) 진단 (--s9dg 1, 결과 불변 -- 셈만): 걸음 시작 rt_bar_y0e 의 활성 장벽 표본점 수 · 띠 안 점 수 (min(d0, d1) < d̂, 블록 문과 같은 식) · 띠 점이 든 블록 수,
    rt_ctl 의 뉴턴 반복 시작 잔차 비 ‖g‖²/F² · PCG 반복마다 잔차 비 r²/rf · 노드·표본점·칸 수 (IP[9]·IP[12]·IP[11]) -> <out>/s9.jsonl (프레임마다, 동기화 있음).
(C) 걸음 시작 장벽 에너지 압축 (--s9c 1, 결과 불변 = 원자 합 순서 잡음 수준): rt_bar_y0e 는 띠 점이 하나라도 든 블록마다 후보 59 개 + E0 의 블록 합 60 번을 함
    (블록 = 바닥에 수직인 칸 기둥이라 접촉이 퍼지면 거의 모든 블록). 같은 커널이 d0·d1 을 같은 식으로 구한 뒤 띠 점 (블록 문과 같은 조건 = 이 점의 모든 몫이 0 이
    아닐 수 있는 점) 만 (q, d0, d1) 압축 목록에 붙이고 (워프 모아 원자 하나) 돌아감 -> s9_y0b 가 목록만 블록 합 (같은 e 식). 빠진 점의 몫은 정확히 0 (원래도 블록
    문이 0 으로 다룸) 이라 바뀌는 것은 합의 묶음뿐 (원래도 블록 사이 원자 합 순서는 정해지지 않음).
    확인 판 (--s9c 2): 원래 커널로 진짜 SC 를 쓰고 (+ 목록), 원래 커널을 한 번 더 그림자 SC' 에 (원자 순서 잡음 기준), 압축 판을 그림자 SC'' 에 -> s9_cmp 가
    후보마다 |SC − SC''| ≤ 2 (n − 1) u Σ|e| (u = 2^-53, 모든 e ≥ 0 이라 Σ|e| = SC) 대조 + 고른 후보 (rt_y0epick 과 같은 식) 가 다른 수 (압축 · 원래 두 번).
(Y) 노드별 시작점 판정 조기 끝 (--s9y 1, 결과 불변 -- 단조성 가정, 확인 판 2 로 셈): yl_kq 의 점마다 E_q(2^-(k+i)) ≤ E_q(0) 인 첫 i 를 59 후보까지 f64 log 로 찾는
    고리를 f32 비교로: d0 ≤ 0 이면 i 0 (E0 = 1e30 이 최대), d1 ≥ d0 이면 d ≥ d0 라 i 0, d1 < d0 이면 d ≤ d0 이므로 d0 ≥ d̂ 면 d ≥ d̂ 인 첫 i (E 0), d0 < d̂ 면
    d == d0 인 첫 i (E 는 d 에 순감소). 확인 판 (--s9y 2): 두 고리를 다 돌고 원래 i 를 씀, 다른 수를 셈.
(E) W̄ 모으기 워프 모으기 (--s9e 1, --s8w 위; 결과 불변 = 원자 합 순서 잡음): 절 CM74 W 와 같은 방법을 e3_wq (입자당 27 원자 합) 에. 확인은 --s8w 2 (칸 6 = e3wq).
(R) X3c 다시 하기를 첫 시도 끝 상태에서 이어 하기 (--s9r 1, 결과 바뀜 -- 후보): X3g ('r', 반복 1 시작)·X2 ('x', 걸음 끝) 로 되돌릴 때 지금 y (첫 시도 반복 0 뒤 /
    걸음 끝) 를 떠 두고, 다시 하는 시도는 ao_floor_n (켠 구속: 고착 상한·속도 수준 접촉) 만 돌린 뒤 시작점 고르기 (rt_ga_y0 ~ aw_y0z_b) 를 건너뛰고 s9_cont 가
    떠 둔 물리 위치를 새 구속의 변수로 옮김 (η = T⁻¹ (y − ŷ), ao_floor_n 의 y0uk 식과 같은 꼴; 고정 노드 그대로, 고착 s = 0 노드는 접선이 ŷ 에 묶임). 법선 성분이
    그대로라 장벽 거리는 첫 시도 받아들인 값. 떠 둔 y 에 비유한이 있으면 원래 다시 하기. 경계 = 'x1' 그래프에 't' (ao_floor_n 뒤)·'u' (ctl op 1 앞) 를 더함.
(P) PCG 멈춤 기준을 뉴턴 잔차 비로 (--s9p γ, 결과 바뀜 -- 후보): Eisenstat–Walker choice 2 -- 반복 0 은 η = tolP 그대로, 반복 k ≥ 1 은 η_k = γ (‖g_k‖/‖g_{k−1}‖)²,
    γ η_{k−1}² > 0.1 이면 η_k ≥ γ η_{k−1}², η_k ≤ 0.9. PCG 는 r² ≤ η_k² rf 이면 멈춤 (상한 --cg 그대로).
"""
import json
import os
import re

import numpy as np
import torch


class C:
    out = "."
    dg = 0
    cm = 0
    yl = 0
    e3 = 0
    r = 0
    ew = 0.0
    log = None
    patched = set()
    bad = []
    said = False
    rsaid = False
    rc = dict(cont=0, fb=0)


HDR = r"""
// ---- 절 CM76 speed9 ----
__device__ int S9DG = 0;
__device__ int S9CM = 0;
__device__ int S9YL = 0;
__device__ int S9EW = 0;
__device__ unsigned long long* S9C;
__device__ int* S9L;
__device__ unsigned int* S9N;
__device__ double* S9SC;
__device__ double* S9LG;
__device__ int* S9LN;
__device__ double S9GP = 0.0;
__device__ double S9ETA = 0.0;
__device__ double S9RF = 0.0;
__device__ double S9EWG = 0.9;
__device__ double S9EWM = 0.9;
__device__ __forceinline__ void s9_cnt(bool a, bool b) {
    unsigned ma = __ballot_sync(0xffffffffu, a), mb = __ballot_sync(0xffffffffu, b);
    if ((threadIdx.x & 31) == 0) { if (ma) atomicAdd(&S9C[0], (unsigned long long)__popc(ma)); if (mb) atomicAdd(&S9C[1], (unsigned long long)__popc(mb)); }
}
__device__ __forceinline__ void s9_push(int q, float d0, float d1, bool b) {
    unsigned m = __ballot_sync(0xffffffffu, b);
    if (!m) return;
    int l = threadIdx.x & 31; unsigned base = 0;
    if (l == 0) base = atomicAdd(S9N, (unsigned)__popc(m));
    base = __shfl_sync(0xffffffffu, base, 0);
    if (b) { unsigned i = base + __popc(m & ((1u << l) - 1u)); S9L[(size_t)i*3] = q; S9L[(size_t)i*3+1] = __float_as_int(d0); S9L[(size_t)i*3+2] = __float_as_int(d1); }
}
__device__ __forceinline__ void s9_lg(int code, double v) {
    if (!S9LG || !S9LN) return;
    int i = atomicAdd(S9LN, 1);
    if (i < 1024) { S9LG[2*i] = (double)code; S9LG[2*i+1] = v; }
}
__device__ __forceinline__ double s9_ew(int it, double eP2, double g2) {
    if (!S9EW || !(eP2 >= 0.0)) return eP2;
    double eta;
    if (it == 0 || !(S9GP > 0.0) || !isfinite(g2)) eta = sqrt(eP2);
    else { double r = g2/S9GP; eta = S9EWG*r; double sg = S9EWG*S9ETA*S9ETA; if (sg > 0.1) eta = fmax(eta, sg); eta = fmin(eta, S9EWM); }
    S9GP = g2; S9ETA = eta;
    return eta*eta;
}
__device__ __forceinline__ int s9_ylf(float d0, float d1, float dh, float c) {
    if (!(d0 > 0.f) || !(d1 < d0)) return 0;
    int i = 0;
    for (; i<59; i++) { float d = d0 + c*(d1 - d0); if (d0 < dh ? (d == d0) : !(d < dh)) break; c *= 0.5f; }
    return i;
}
"""

TAIL = r"""
// ---- 절 CM76 speed9 커널 ----
__global__ void s9_set(int cm) { S9CM = cm; }
__global__ void s9_rst(int full) { int i = threadIdx.x; if (i == 0) *S9N = 0; if (full && i < 128) S9SC[i] = 0.0; }
__global__ void __launch_bounds__(128) s9_y0b(const float* Vq, float dh, float ks, const int* kk, double* SC, int NE) { if (AF_SKIP) return;
    unsigned n = *S9N;
    if (blockIdx.x*blockDim.x >= n) return;
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = i < n;
    int q = on ? S9L[(size_t)i*3] : 0;
    float d0 = on ? __int_as_float(S9L[(size_t)i*3+1]) : 0.f, d1 = on ? __int_as_float(S9L[(size_t)i*3+2]) : 0.f;
    double kap = on ? (double)ks*(double)Vq[q] : 0.0;
    double e = 0.0;
    if (on && d0 < dh) { if (d0 > 0.f) { double dd = (double)d0 - dh; e = -kap*dd*dd*log((double)d0/dh); } else e = 1e30; }
    blocksum_add(e, &SC[0]);
    float c = ldexpf(1.f, -kk[0]);
    for (int j=0;j<NE;j++) {
        e = 0.0;
        if (on) { float d = d0 + c*(d1 - d0);
            if (d < dh) { if (d > 0.f) { double dd = (double)d - dh; e = -kap*dd*dd*log((double)d/dh); } else e = 1e30; } }
        blocksum_add(e, &SC[1+j]);
        c *= 0.5f;
    }
}
__device__ int s9_pick(const double* S, int NE) { double E0 = S[0]; for (int i=0;i<NE;i++) { double E1 = S[1+i]; if (isfinite(E1) && E1 <= E0) return i; } return -1; }
__global__ void s9_cmp(const double* SC, int NE) { if (AF_SKIP) return;
    unsigned n = *S9N;
    S9C[8] += 1; S9C[4] += n;
    double u = 1.1102230246251565e-16, nn = (n > 1) ? (double)(n - 1) : 0.0;
    float r1 = __uint_as_float((unsigned)S9C[13]), r2 = __uint_as_float((unsigned)S9C[14]);
    for (int k=0;k<=NE;k++) {
        double a = SC[k], b = S9SC[k], c = S9SC[64+k];
        double s1 = fmax(fabs(a), fabs(b)), s2 = fmax(fabs(a), fabs(c));
        double d1 = fabs(a - b), d2 = fabs(a - c);
        if (!(d1 <= 2.0*nn*u*s1*1.0001 + 1e-300)) S9C[9] += 1;
        if (!(d2 <= 2.0*nn*u*s2*1.0001 + 1e-300)) S9C[12] += 1;
        if (nn > 0.0 && s1 > 0.0) r1 = fmaxf(r1, (float)(d1/(nn*u*s1)));
        if (nn > 0.0 && s2 > 0.0) r2 = fmaxf(r2, (float)(d2/(nn*u*s2)));
    }
    S9C[13] = __float_as_uint(r1); S9C[14] = __float_as_uint(r2);
    int pa = s9_pick(SC, NE), pb = s9_pick(S9SC, NE), pc = s9_pick(S9SC + 64, NE);
    if (pa != pb) S9C[10] += 1;
    if (pa != pc) S9C[11] += 1;
}
__global__ void s9_cont(const int* IP, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* ys, float* eta, float* y) { if (AF_SKIP) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    int p = pm[n];
    if (p == 2) return;
    float s = sk[n], isk = (s > 1e-6f) ? 1.f/s : 0.f;
    float dd[3], et[3]; for (int a=0;a<3;a++) dd[a] = ys[n*3+a] - yh[n*3+a];
    float nv[3] = {n0, n1, n2};
    if (p == 0) { for (int a=0;a<3;a++) et[a] = isk*dd[a]; }
    else if (p == 3) { float dn = dd[0]*n0 + dd[1]*n1 + dd[2]*n2; for (int a=0;a<3;a++) et[a] = (dd[a] - dn*nv[a]) + isk*dn*nv[a]; }
    else { float dn = dd[0]*n0 + dd[1]*n1 + dd[2]*n2; for (int a=0;a<3;a++) et[a] = dn*nv[a] + isk*(dd[a] - dn*nv[a]); }
    float ty[3]; Tapply(s, p, n0, n1, n2, et, ty);
    for (int a=0;a<3;a++) { eta[n*3+a] = et[a]; y[n*3+a] = yh[n*3+a] + ty[a]; }
}
"""

Y0E_OLD = "    if (!__syncthreads_or(act && fminf(d0, d1) < dh + 1e-5f*(fabsf(d0) + fabsf(d1)))) return;"
Y0E_NEW = ("    bool s9b_ = act && fminf(d0, d1) < dh + 1e-5f*(fabsf(d0) + fabsf(d1));   // 절 CM76 (D)·(C)\n"
           "    if (S9DG) s9_cnt(act, s9b_);\n"
           "    if (S9CM) s9_push(qq, d0, d1, s9b_);\n"
           "    if (S9CM == 1) return;\n"
           "    if (!__syncthreads_or(s9b_)) return;\n"
           "    if (S9DG && threadIdx.x == 0) atomicAdd(&S9C[2], 1ULL);")
YL_OLD = ("        float c = ldexpf(1.f, -k); int i = 0;\n"
          "        for (; i<59; i++) { double E = yl_e(d0 + c*(d1 - d0), dh, kap); if (E <= E0) break; c *= 0.5f; }")
YL_NEW = ("        float c = ldexpf(1.f, -k); int i = 0;\n"
          "        if (S9YL && isfinite(d0) && isfinite(d1)) { i = s9_ylf(d0, d1, dh, c);   // 절 CM76 (Y)\n"
          "            if (S9YL == 2) { int j = 0; float c2 = c; for (; j<59; j++) { double E = yl_e(d0 + c2*(d1 - d0), dh, kap); if (E <= E0) break; c2 *= 0.5f; }\n"
          "                             atomicAdd(&S9C[5], 1ULL); if (j != i) atomicAdd(&S9C[6], 1ULL); i = j; } }\n"
          "        else for (; i<59; i++) { double E = yl_e(d0 + c*(d1 - d0), dh, kap); if (E <= E0) break; c *= 0.5f; }")
CTL3_OLD = "        double g2 = SC[1];\n"
CTL3_NEW = "        double g2 = SC[1];\n        if (S9DG) s9_lg(100 + it, it == 0 ? g2/SC[56] : g2/CD[2]);   // 절 CM76 (D)\n"
CTL4_OLD = "    if (op == 4) { if (CT[0]) return; double rf = fmin(SC[1], CD[2]); CD[1] = (eP2 >= 0.0) ? eP2*rf : -1.0; CT[1] = 0; CT[7]++; SC[57] = 0.0; return; }"
CTL4_NEW = ("    if (op == 4) { if (CT[0]) return; double rf = fmin(SC[1], CD[2]); double e2_ = s9_ew(it, eP2, SC[1]); S9RF = rf;   // 절 CM76 (P)\n"
            "        if (S9DG && S9EW) s9_lg(300 + it, sqrt(e2_));\n"
            "        CD[1] = (eP2 >= 0.0) ? e2_*rf : -1.0; CT[1] = 0; CT[7]++; SC[57] = 0.0; return; }")
CTL6_OLD = "CT[3]++; if (CD[1] >= 0.0 && r2 <= CD[1]) CT[1] = 1; return; }"
CTL6_NEW = "CT[3]++; if (S9DG) s9_lg(200 + it, S9RF > 0.0 ? r2/S9RF : -1.0); if (CD[1] >= 0.0 && r2 <= CD[1]) CT[1] = 1; return; }"
PATCH = {"rt_bar_y0e": [(Y0E_OLD, Y0E_NEW)], "yl_kq": [(YL_OLD, YL_NEW)], "rt_ctl": [(CTL3_OLD, CTL3_NEW), (CTL4_OLD, CTL4_NEW), (CTL6_OLD, CTL6_NEW)]}

E3_OLD = "    for (int o=0;o<27;o++) atomicAdd(&E3W(q, o), ww[o]*sc);"
E3_NEW = ("    { unsigned s8m = s8_act(p, N); int s9q = (int)q; unsigned gq = __match_any_sync(s8m, s9q); bool lq = s8_lead(gq);   // 절 CM76 (E)\n"
          "      for (int o=0;o<27;o++) { float t = ww[o]*sc; float a0 = s8_gsum(gq, t); if (lq) atomicAdd(&E3W(q, o), a0); s8_sh(6, (size_t)o*AF_WQN + q, t); }\n"
          "      s8_ld(4, lq); }")


def _patch(src):
    for nm, reps in PATCH.items():
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
                C.bad.append([nm, old[:70], body.count(old)])
                break
            body = body.replace(old, new, 1)
        if ok:
            src = src[:i] + body + src[j:]
            C.patched.add(nm)
    return src


def _src(src):
    if "s9_y0b(" in src or not re.search(r"__global__\s+void\s+__launch_bounds__\(128\)\s+rt_bar_y0e\s*\(", src):
        return src
    src = _patch(src)
    k = src.find("__global__")
    k = src.rfind("\n", 0, k) + 1
    src = src[:k] + HDR + src[k:]
    e = src.rstrip().rfind("\n}")
    src = src[:e] + "\n" + TAIL + src[e:]
    if not C.said:
        C.said = True
        print(f"[절 CM76] 소스 고침 {sorted(C.patched)} 못 고침 {C.bad}", flush=True)
    return src


def _P(t):
    return np.uint64(0 if t is None else t.data_ptr())


def _g(s, nm, dt):
    import cupy as cp
    return cp.ndarray((1,), dtype=dt, memptr=s.mod.get_global(nm))


def bind(s):
    import cupy as cp
    try:
        s.mod.get_global("S9C")
    except Exception:
        return
    dev = s.x.device
    capQ = int(s.capQc) * 8
    if getattr(s, "s9c", None) is None:
        s.s9c = torch.zeros(16, device=dev, dtype=torch.int64)
        s.s9n = torch.zeros(1, device=dev, dtype=torch.int32)
        s.s9sc = torch.zeros(128, device=dev, dtype=torch.float64)
        s.s9lg = torch.zeros(2048, device=dev, dtype=torch.float64)
        s.s9ln = torch.zeros(1, device=dev, dtype=torch.int32)
    if C.cm and (getattr(s, "s9l", None) is None or s.s9l.numel() < capQ * 3):
        s.s9l = torch.zeros(capQ * 3 + 3, device=dev, dtype=torch.int32)
    for nm, t in (("S9C", s.s9c), ("S9N", s.s9n), ("S9SC", s.s9sc), ("S9LG", s.s9lg), ("S9LN", s.s9ln), ("S9L", getattr(s, "s9l", None))):
        _g(s, nm, cp.uint64)[0] = 0 if t is None else int(t.data_ptr())
    _g(s, "S9DG", cp.int32)[0] = int(C.dg)
    _g(s, "S9CM", cp.int32)[0] = 1 if C.cm == 1 else 0
    _g(s, "S9YL", cp.int32)[0] = int(C.yl)
    _g(s, "S9EW", cp.int32)[0] = 1 if C.ew > 0 else 0
    if C.ew > 0:
        _g(s, "S9EWG", cp.float64)[0] = float(C.ew)


class _YCall:
    """rt_bar_y0e (P 거르기 감싼 것 그대로 f): (C) 압축 / 확인."""

    def __init__(s_, s, f):
        s_.s, s_.f = s, f

    def __call__(s_, grid, block, args, **kw):
        s = s_.s
        fn = s.fn
        capQ = int(s.capQc) * 8
        g = ((capQ + 127) // 128,)
        Vq, dh, ks, kk, SC, NE = args[3], args[12], args[13], args[14], args[15], args[16]
        if C.cm == 1:
            fn.get_raw("s9_rst")((1,), (128,), (np.int32(0),))
            r = s_.f(grid, block, args, **kw)
            fn.get_raw("s9_y0b")(g, (128,), (Vq, dh, ks, kk, SC, NE))
            return r
        fn.get_raw("s9_rst")((1,), (128,), (np.int32(1),))
        fn.get_raw("s9_set")((1,), (1,), (np.int32(2),))
        r = s_.f(grid, block, args, **kw)
        fn.get_raw("s9_set")((1,), (1,), (np.int32(0),))
        a2 = list(args)
        a2[15] = np.uint64(int(s.s9sc.data_ptr()) + 64 * 8)
        s_.f(grid, block, tuple(a2), **kw)
        fn.get_raw("s9_y0b")(g, (128,), (Vq, dh, ks, kk, np.uint64(int(s.s9sc.data_ptr())), NE))
        fn.get_raw("s9_cmp")((1,), (1,), (SC, NE))
        return r


class _ECall:
    """e3_wq 확인 (--s8w 2 + --s9e): 절 CM74 s8_pre / s8_cmp 를 칸 6 (e3wq) 에."""

    def __init__(s_, s, f):
        s_.s, s_.f = s, f

    def __call__(s_, grid, block, args, **kw):
        s = s_.s
        sh = (getattr(s, "s8sh", None) or {}).get(6)
        t = getattr(s, "e3wq", None)
        if sh is None or t is None or getattr(s, "s8st", None) is None:
            return s_.f(grid, block, args, **kw)
        a, h, c_ = sh
        n = t.numel()
        gg = ((min(n, 1 << 22) + 255) // 256,)
        s.fn.get_raw("s8_pre")(gg, (256,), (_P(t), _P(a), _P(h), _P(c_), np.int64(n)))
        r = s_.f(grid, block, args, **kw)
        s.fn.get_raw("s8_cmp")(gg, (256,), (_P(t), _P(a), _P(h), _P(c_), np.int64(n), _P(s.s8st), np.int32(6)))
        return r


def _install_src():
    import cupy as cp
    o_rm = cp.RawModule

    def RawModule(*a, **kw):
        if "code" in kw and isinstance(kw["code"], str):
            kw["code"] = _src(kw["code"])
        elif a and isinstance(a[0], str):
            a = (_src(a[0]),) + tuple(a[1:])
        return o_rm(*a, **kw)
    cp.RawModule = RawModule


def _install_fn():
    from . import fem_speed7 as S7
    o_gi = S7._PFn.__getitem__

    def gi(s_, name):
        f = o_gi(s_, name)
        if name == "rt_bar_y0e" and C.cm and "rt_bar_y0e" in C.patched:
            return _YCall(s_.owner, f)
        if name == "e3_wq" and C.e3:
            from . import fem_speed8 as S8
            if S8.C.w == 2 and "e3_wq" in S8.C.patched:
                return _ECall(s_.owner, f)
        return f
    S7._PFn.__getitem__ = gi


def _install_e3():
    from . import fem_speed8 as S8
    S8.WPATCH["e3_wq"] = [(E3_OLD, E3_NEW)]
    S8.SLOT[6] = "e3wq"
    S8.KNAME[4] = "e3_wq"


# ---------------- (R) 이어 하기 ----------------
class _TProxy:
    def __init__(s_, s, g):
        s_.s, s_.g = s, g

    def replay(s_):
        s = s_.s
        if getattr(s, "_s9cont", False):
            s._s9cont = False
            from . import fem_rt_cuda as RTC
            nr = s.floor["NR"] if s.floor else [0.0, 0.0, 1.0]
            f = s.fn.get_raw("s9_cont")
            capN = int(s.capN)
            with RTC._Stream():
                f(((capN + 255) // 256,), (256,), (_P(s.IP), _P(s.sk), _P(s.pm), *[np.float32(q) for q in nr], _P(s.yh), _P(s._s9ys), _P(s.eta), _P(s.y)))
            C.rc["cont"] += 1
            return
        return s_.g.replay()


def _install_r():
    from . import fem_ao_rt as AORT
    from . import fem_common3 as C3
    from . import fem_speed4 as S4
    from . import fem_wmrx as WRX
    assert S4.X1.on and S4.X1.level == 2, "--s9r 는 X2 (@s4x 2) 위"
    cls = AORT.AORTSolver
    o_k, o_ctl = cls.k, cls.ctl

    def cap1(s):
        return getattr(s, "_c3cap", None) is not None and getattr(s, "_x1K", 0) == 1 and getattr(s, "wbar", None) is not None

    def k(s, name, n, args, b=128):
        r = o_k(s, name, n, args, b)
        if name == "ao_floor_n" and cap1(s) and int(getattr(s, "_x1ks", -1)) == 0:
            C3.mark(s, "t")
        return r

    def ctl(s, op, it=0):
        if op == 1 and cap1(s):
            C3.mark(s, "u")
        return o_ctl(s, op, it)
    cls.k, cls.ctl = k, ctl
    o_init = C3.SegGraph.__init__

    def __init__(self, s, segs):
        o_init(self, s, segs)
        k4 = getattr(self, "k4", None)
        if k4 is not None and "t" in k4 and "u" in k4 and "s" in k4:
            i = k4.index("t")
            assert k4.index("s") < i < k4.index("u") and k4.count("t") == 1 and k4.count("u") == 1, k4
            cond, g = self.segs[i]
            assert not cond
            self.segs[i] = (cond, _TProxy(s, g))
            if not C.rsaid:
                C.rsaid = True
                print(f"[절 CM76] 이어 하기 경계 {k4}", flush=True)
    C3.SegGraph.__init__ = __init__
    o_rs = S4._restore

    def _restore(s):
        torch.cuda.synchronize()
        nn = int(s.IP[9])
        if getattr(s, "_s9ys", None) is None or s._s9ys.numel() != s.y.numel():
            s._s9ys = torch.empty_like(s.y)
        s._s9ys.copy_(s.y)
        s._s9ok = bool(torch.isfinite(s.y.reshape(-1)[:nn * 3]).all())
        return o_rs(s)
    S4._restore = _restore
    o_on = WRX.on

    def on(s, v):
        if int(v) == 1 and getattr(s, "_s9ok", None) is not None:
            if s._s9ok:
                s._s9cont = True
            else:
                C.rc["fb"] += 1
            s._s9ok = None
        elif int(v) == 0:
            s._s9cont = False
            s._s9ok = None
        return o_on(s, v)
    WRX.on = on


def frame_end(s, fr):
    rec = dict(f=fr)
    if fr <= 3:
        rec.update(patched=sorted(C.patched), bad=C.bad)
    if C.r:
        rec["rc"] = dict(C.rc)
        C.rc = dict(cont=0, fb=0)
    if (C.dg or C.cm == 2 or C.yl == 2) and getattr(s, "s9c", None) is not None:
        torch.cuda.synchronize()
        c = s.s9c.tolist()
        s.s9c.zero_()
        ip = s.IP.tolist()
        rec["ip"] = dict(nn=ip[9], nq=ip[12], nc=ip[11])
        if C.dg:
            rec["bar"] = dict(act=c[0], band=c[1], bblk=c[2])
            n = min(int(s.s9ln.item()), 1024)
            lg = s.s9lg[:2 * n].tolist()
            s.s9ln.zero_()
            rec["lg"] = [[int(lg[2 * i]), lg[2 * i + 1]] for i in range(n)]
        if C.cm == 2:
            u = lambda v: float(np.array([int(v) & 0xffffffff], dtype=np.uint32).view(np.float32)[0])
            rec["chk"] = dict(calls=c[8], nlist=c[4], viol=c[9], pick=c[10], viol_ref=c[12], pick_ref=c[11], rmax=u(c[13]), rmax_ref=u(c[14]))
        if C.yl == 2:
            rec["ylchk"] = dict(n=c[5], mis=c[6])
    if len(rec) > 1:
        C.log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        C.log.flush()


def frame_install():
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
                if dep[0] == 0 and isinstance(s, AORT.AORTSolver):
                    frame_end(s, S6.W.fr)
        return w
    ART.split_step = outer(ART.split_step)
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)


def install(out, dg=0, cm=0, yl=0, e3=0, r=0, ew=0.0):
    """실행기를 runpy 로 띄우는 순간, speed6·7·8 설치 뒤 (가장 바깥) 에 부름."""
    from . import fem_ao_rt as AORT
    C.out, C.dg, C.cm, C.yl, C.e3, C.r, C.ew = out, int(dg), int(cm), int(yl), int(e3), int(r), float(ew)
    C.log = open(os.path.join(out, "s9.jsonl"), "w")
    if C.e3:
        _install_e3()
    _install_src()
    _install_fn()
    cls = AORT.AORTSolver
    ob = cls.bind_act

    def bind_act(s, *a, **kw):
        rr = ob(s, *a, **kw)
        try:
            bind(s)
        except Exception as e_:
            C.bad.append(["bind", str(e_)[:200]])
        return rr
    cls.bind_act = bind_act
    if C.r:
        _install_r()
    frame_install()
    print(f"[절 CM76] speed9 켬 진단 {C.dg} 압축 {C.cm} 시작점 조기 끝 {C.yl} W̄ 워프 모으기 {C.e3} 이어 하기 {C.r} EW γ {C.ew}", flush=True)
