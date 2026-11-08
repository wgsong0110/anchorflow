"""절 CM74 묶음 speed8 (11-08 위임, 수박 렌더 포함 벽시계 60 fps 이어서): 공통 후보 H2 (= H + A, 절 CM72) 위 가속·후보. 감싸개 exe/speed8_wrap.py 가 부름.

(W) 워프 모으기 (--s8w 1, 결과 불변 = 원자 합 순서 잡음 수준 -- 옛 실행기 (수박) 만): 입자 -> 팔분 점 (q)·칸 (c) 원자 합 커널 넷
    (rt_p2g 29 개 · aq_p2g_s 8 개 · gq_p2q 6 개 · wb_p2q 4 개 / 입자) 을, 워프 안에서 같은 q (또는 c) 를 가진 스레드끼리 (__match_any_sync) 먼저 더한 뒤
    (모둠 안 레인 순서대로 셔플 합) 모둠 대표 (가장 낮은 레인) 하나만 원자 합. 입자는 실행기가 첫 프레임 칸 순서로 정렬해 둠 (fem_an_rt 2536) 이라
    이웃 스레드가 같은 칸·팔분 점을 많이 공유. 합의 순서만 바뀜 (모둠 크기 1 이면 0 + v = v 로 원래와 비트 같은 덧셈).
(W 확인, --s8w 2): W 로 진짜 버퍼에 쓰면서, 같은 커널 안에서 원래 방식 (입자마다 원자 합) 을 그림자 버퍼 sh1 에, |항| 합을 sha 에, 항 수를 shn 에 더함
    (커널 앞에 sh1 = 진짜 버퍼 사본, sha = |진짜|, shn = 1). 커널 뒤 s8_cmp 가 원소마다 |W − 원래| 와 반올림 한계 2 (n − 1) u Σ|항| (u = 2^-24,
    어떤 순서의 되풀이 합도 오차 ≤ (n − 1) u Σ|항| 이므로 두 순서의 차는 그 두 배 이하) 를 대조 -> 넘는 원소 = 어김. 정규화 차 max |Δ| / (u (n − 1) Σ|항|)
    도 기록. 워프 모둠 수 (대표 수 / 레인 수) 도 셈 -> <out>/s8w.jsonl (프레임마다).
(G) X3g (--s8g T, 결과 바뀜 -- 후보): 절 CM50 X3e4 (첫 시도 반복 0 '시작' 잔차 > 1e6 이면 켬) 와 달리, 반복 0 을 한 번 푼 뒤 반복 1 시작 잔차
    ‖g‖²/F² 가 T 를 넘거나 비유한이면 (= 무너짐 판정 (--ue_bigT 1e6) 을 이미 넘은 첫 시도) 반복 1·판정을 안 풀고 켜서 걸음 시작부터 (X2 와 같은 되돌림 +
    예산 시계 TB_RUN 0). 구현 = 절 CM50 X3e4 의 'r' 경계·되돌림을 그대로 쓰고 경계 자리만 반복 1 (speed4 의 반복 0 경계는 끔). 놓친 무너짐은 X2.
    반복 1 이 조건 구간 (--xit_from 1) 이면 경계가 조건을 바꾸므로 안 씀 (assert).
"""
import json
import os

import numpy as np
import torch


class C:
    out = "."
    w = 0
    g = 0.0
    log = None
    patched = set()
    bad = []
    said = False
    gcnt = 0
    gtot = 0


# ---------------- (W) 워프 모으기 ----------------
S8_HDR = r"""
// ---- 절 CM74 W: 워프 모으기 (같은 열쇠 레인끼리 셔플 합 -> 대표 하나만 원자 합) ----
__device__ int S8CHK;                       // 1 = 확인 판 (그림자 원자 합)
__device__ float* S8P[24];                  // 칸 k: [3k] sh1 (원래 방식 합), [3k+1] sha (|항| 합), [3k+2] shn (항 수)
__device__ unsigned long long* S8LD;        // [2k] 레인 수, [2k+1] 대표 수
__device__ __forceinline__ unsigned s8_act(int p, int N) { int b = p & ~31; int r = N - b; return r >= 32 ? 0xffffffffu : ((1u << r) - 1u); }
__device__ __forceinline__ float s8_gsum(unsigned g, float v) { float s = 0.f; unsigned r = g; while (r) { int l = __ffs(r) - 1; r &= r - 1; s += __shfl_sync(g, v, l); } return s; }
__device__ __forceinline__ bool s8_lead(unsigned g) { return (int)(threadIdx.x & 31) == __ffs(g) - 1; }
__device__ __forceinline__ void s8_sh(int k, size_t off, float v) { if (S8CHK) { atomicAdd(S8P[3*k] + off, v); atomicAdd(S8P[3*k+1] + off, fabsf(v)); atomicAdd(S8P[3*k+2] + off, 1.f); } }
__device__ __forceinline__ void s8_ld(int k, bool l) { if (S8CHK && S8LD) { atomicAdd(S8LD + 2*k, 1ull); if (l) atomicAdd(S8LD + 2*k + 1, 1ull); } }
extern "C" __global__ void s8_pre(const float* real, float* sh1, float* sha, float* shn, long long n) {
    for (long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x*blockDim.x) { float v = real[i]; sh1[i] = v; sha[i] = fabsf(v); shn[i] = 1.f; }
}
// st[k*6]: 0 어김 수, 1 비교 원소 (항 2 개 이상), 2 max 정규화 차 (float 비트), 3 max |Δ|/Σ|항| (float 비트), 4 비유한 다름, 5 비트 다름 원소
extern "C" __global__ void s8_cmp(const float* real, const float* sh1, const float* sha, const float* shn, long long n, unsigned long long* st, int k) {
    for (long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x*blockDim.x) {
        float a = real[i], b = sh1[i], s = sha[i], c = shn[i];
        if (c < 1.5f) { if (__float_as_int(a) != __float_as_int(b)) atomicAdd(st + k*6 + 5, 1ull); continue; }
        atomicAdd(st + k*6 + 1, 1ull);
        if (__float_as_int(a) != __float_as_int(b)) atomicAdd(st + k*6 + 5, 1ull);
        if (!isfinite(a) || !isfinite(b)) { if (__float_as_int(a) != __float_as_int(b)) atomicAdd(st + k*6 + 4, 1ull); continue; }
        double d = fabs((double)a - (double)b), u = 5.9604644775390625e-8, bnd = 2.0 * ((double)c - 1.0) * u * (double)s;
        if (d > bnd * 1.0001 + 1e-38) atomicAdd(st + k*6, 1ull);
        float r = (float)(d / fmax(((double)c - 1.0) * u * (double)s, 1e-300)), q = (float)(d / fmax((double)s, 1e-300));
        atomicMax((unsigned long long*)(st + k*6 + 2), (unsigned long long)__float_as_uint(r));
        atomicMax((unsigned long long*)(st + k*6 + 3), (unsigned long long)__float_as_uint(q));
    }
}
"""

# (커널, 옛 글자, 새 글자) -- 옛 글자는 절 CM72 kZ2_a/s7src_1.cu (H2 수박 최종 소스) 그대로
WPATCH = {
    "rt_p2g": [(
        "    for (int k=0;k<9;k++) { atomicAdd(&aq[k], w*R[k]); atomicAdd(&aq[9+k], w*L[k]); atomicAdd(&aq[18+k], w*Fp[k]); }\n"
        "    atomicAdd(&aq[27], w); atomicAdd(&ac[29], m[p]);",
        "    { unsigned s8m = s8_act(p, N); unsigned gq = __match_any_sync(s8m, q), gc = __match_any_sync(s8m, c); bool lq = s8_lead(gq), lc = s8_lead(gc);   // 절 CM74 W\n"
        "      for (int k=0;k<9;k++) { float a0 = s8_gsum(gq, w*R[k]), a1 = s8_gsum(gq, w*L[k]), a2 = s8_gsum(gq, w*Fp[k]);\n"
        "                              if (lq) { atomicAdd(&aq[k], a0); atomicAdd(&aq[9+k], a1); atomicAdd(&aq[18+k], a2); }\n"
        "                              s8_sh(0, (size_t)q*28+k, w*R[k]); s8_sh(0, (size_t)q*28+9+k, w*L[k]); s8_sh(0, (size_t)q*28+18+k, w*Fp[k]); }\n"
        "      float a3 = s8_gsum(gq, w), a4 = s8_gsum(gc, m[p]); if (lq) atomicAdd(&aq[27], a3); if (lc) atomicAdd(&ac[29], a4);\n"
        "      s8_sh(0, (size_t)q*28+27, w); s8_sh(1, (size_t)c*30+29, m[p]); s8_ld(0, lq); }"),
    ],
    "aq_p2g_s": [(
        "    for (int k=0;k<4;k++) { atomicAdd(&Sa[(size_t)q*4+k], v4[k]); atomicAdd(&Sc[(size_t)c*4+k], v4[k]); }",
        "    { unsigned s8m = s8_act(p, N); unsigned gq = __match_any_sync(s8m, q), gc = __match_any_sync(s8m, c); bool lq = s8_lead(gq), lc = s8_lead(gc);   // 절 CM74 W\n"
        "      for (int k=0;k<4;k++) { float a0 = s8_gsum(gq, v4[k]), a1 = s8_gsum(gc, v4[k]); if (lq) atomicAdd(&Sa[(size_t)q*4+k], a0); if (lc) atomicAdd(&Sc[(size_t)c*4+k], a1);\n"
        "                              s8_sh(2, (size_t)q*4+k, v4[k]); s8_sh(3, (size_t)c*4+k, v4[k]); }\n"
        "      s8_ld(1, lq); }"),
    ],
    "gq_p2q": [(
        "    for (int a=0;a<3;a++) { float t = mp*v[p*3+a]; atomicAdd(&PQ[(size_t)q*6+a], t); }\n"
        "    if (pa) for (int a=0;a<3;a++) { float t = mp*pa[p*3+a]; atomicAdd(&PQ[(size_t)q*6+3+a], t); }",
        "    unsigned s8m = s8_act(p, N); unsigned gq = __match_any_sync(s8m, q); bool lq = s8_lead(gq);   // 절 CM74 W\n"
        "    for (int a=0;a<3;a++) { float t = mp*v[p*3+a]; float a0 = s8_gsum(gq, t); if (lq) atomicAdd(&PQ[(size_t)q*6+a], a0); s8_sh(4, (size_t)q*6+a, t); }\n"
        "    if (pa) for (int a=0;a<3;a++) { float t = mp*pa[p*3+a]; float a0 = s8_gsum(gq, t); if (lq) atomicAdd(&PQ[(size_t)q*6+3+a], a0); s8_sh(4, (size_t)q*6+3+a, t); }\n"
        "    s8_ld(2, lq);"),
    ],
    "wb_p2q": [(
        "    for (int a=0;a<3;a++) atomicAdd(&AF_BX[q*4+a], m[p]*x[p*3+a]); atomicAdd(&AF_BX[q*4+3], m[p]);",
        "    { unsigned s8m = s8_act(p, N); unsigned gq = __match_any_sync(s8m, q); bool lq = s8_lead(gq);   // 절 CM74 W\n"
        "      for (int a=0;a<3;a++) { float t = m[p]*x[p*3+a]; float a0 = s8_gsum(gq, t); if (lq) atomicAdd(&AF_BX[q*4+a], a0); s8_sh(5, (size_t)q*4+a, t); }\n"
        "      float a3 = s8_gsum(gq, m[p]); if (lq) atomicAdd(&AF_BX[q*4+3], a3); s8_sh(5, (size_t)q*4+3, m[p]); s8_ld(3, lq); }"),
    ],
}
# 확인 판 그림자 칸: 칸 번호 -> 풀이기 속성 이름 (진짜 버퍼)
SLOT = {0: "Aacc", 1: "Cacc", 2: "Sa", 3: "Sc", 4: "gqP", 5: "bx"}
KSLOT = {"rt_p2g": (0, 1), "aq_p2g_s": (2, 3), "gq_p2q": (4,), "wb_p2q": (5,)}
KNAME = {0: "rt_p2g", 1: "aq_p2g_s", 2: "gq_p2q", 3: "wb_p2q"}


def _w_patch(src):
    import re
    for nm, reps in WPATCH.items():
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


def _w_src(src):
    if "s8_gsum(" in src or "void rt_p2g(" not in src:
        return src
    src = _w_patch(src)
    k = src.find("__global__")
    k = src.rfind("\n", 0, k) + 1
    src = src[:k] + S8_HDR + src[k:]
    if not C.said:
        C.said = True
        print(f"[절 CM74] 워프 모으기 W 켬 (모드 {C.w}) 고친 커널 {sorted(C.patched)} 못 고침 {C.bad}", flush=True)
    return src


def _P(t):
    return np.uint64(0 if t is None else t.data_ptr())


def w_bind(s):
    """확인 판: 그림자 버퍼를 진짜 버퍼 크기로 (다시 할당 뒤마다) + 장치 전역."""
    import cupy as cp
    m = s.mod
    try:
        g = m.get_global("S8CHK")
    except Exception:
        return
    cp.ndarray((1,), dtype=cp.int32, memptr=g)[0] = 1 if C.w == 2 else 0
    if C.w != 2:
        return
    if getattr(s, "s8st", None) is None:
        s.s8st = torch.zeros(6 * 8, device=s.x.device, dtype=torch.int64)
        s.s8ld = torch.zeros(16, device=s.x.device, dtype=torch.int64)
    sh = getattr(s, "s8sh", None) or {}
    ptrs = np.zeros(24, dtype=np.uint64)
    for k, nm in SLOT.items():
        t = getattr(s, nm, None)
        if not torch.is_tensor(t):
            continue
        n = t.numel()
        if k not in sh or sh[k][0].numel() != n:
            sh[k] = [torch.empty(n, device=t.device, dtype=torch.float32) for _ in range(3)]
        for j in range(3):
            ptrs[3 * k + j] = sh[k][j].data_ptr()
    s.s8sh = sh
    cp.ndarray((24,), dtype=cp.uint64, memptr=m.get_global("S8P"))[:] = cp.asarray(ptrs)
    cp.ndarray((1,), dtype=cp.uint64, memptr=m.get_global("S8LD"))[0] = int(s.s8ld.data_ptr())


def w_install(out, mode):
    from . import fem_ao_rt as AORT
    C.w = int(mode)
    import cupy as cp                                                                 # 글자 고침은 RawModule 에서 (실행기 인자 --sp2 가 _src 를 나중에 감싸 rt_p2g·gq_p2q 를 고치므로 _src 에서는 최종 글자가 아님 -- s0k 첫 판)
    o_rm = cp.RawModule

    def RawModule(*a, **kw):
        if "code" in kw and isinstance(kw["code"], str):
            kw["code"] = _w_src(kw["code"])
        elif a and isinstance(a[0], str):
            a = (_w_src(a[0]),) + tuple(a[1:])
        return o_rm(*a, **kw)
    cp.RawModule = RawModule
    cls = AORT.AORTSolver
    ob = cls.bind_act

    def bind_act(s, *a, **kw):
        r = ob(s, *a, **kw)
        try:
            w_bind(s)
        except Exception as e_:
            C.bad.append(["bind", str(e_)[:200]])
        return r
    cls.bind_act = bind_act
    if C.w != 2:
        return
    o_k = cls.k

    def k(s, name, n, args, b=128):
        sl = KSLOT.get(name) if (name in C.patched and getattr(s, "s8sh", None) is not None and getattr(s, "ao_ready", False)) else None
        if not sl:
            return o_k(s, name, n, args, b)
        L = []
        for kk in sl:
            t = getattr(s, SLOT[kk], None)
            if not torch.is_tensor(t) or kk not in s.s8sh:
                continue
            a, h, c_ = s.s8sh[kk]
            L.append((kk, t, a, h, c_))
            o_k(s, "s8_pre", min(t.numel(), 1 << 22), (_P(t), _P(a), _P(h), _P(c_), np.int64(t.numel())), 256)
        r = o_k(s, name, n, args, b)
        for kk, t, a, h, c_ in L:
            o_k(s, "s8_cmp", min(t.numel(), 1 << 22), (_P(t), _P(a), _P(h), _P(c_), np.int64(t.numel()), _P(s.s8st), np.int32(kk)), 256)
        return r
    cls.k = k


def _f32(u):
    return float(np.array([int(u) & 0xffffffff], dtype=np.uint32).view(np.float32)[0])


def w_frame_end(s, fr):
    rec = dict(f=fr)
    if fr <= 3:
        rec.update(patched=sorted(C.patched), bad=C.bad)
    if C.w == 2 and getattr(s, "s8st", None) is not None:
        torch.cuda.synchronize()
        st = s.s8st.tolist(); ld = s.s8ld.tolist()
        s.s8st.zero_(); s.s8ld.zero_()
        rec["chk"] = {SLOT[k]: dict(viol=st[k * 6], n=st[k * 6 + 1], rmax=_f32(st[k * 6 + 2]), qmax=_f32(st[k * 6 + 3]), nfin=st[k * 6 + 4], ndiff=st[k * 6 + 5]) for k in SLOT}
        rec["grp"] = {KNAME[k]: [ld[2 * k], ld[2 * k + 1]] for k in KNAME}
    if C.g > 0:
        rec["x3g"] = C.gcnt
        C.gtot += C.gcnt
        C.gcnt = 0
    if len(rec) > 1:
        C.log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        C.log.flush()


# ---------------- (G) X3g 후보 ----------------
def g_install(T):
    from . import fem_ao_rt as AORT
    from . import fem_common3 as C3
    from . import fem_speed4 as S4
    assert S4.X1.on and S4.X1.level == 2, "--s8g 는 X2 (@s4x 2) 위"
    C.g = float(T)
    S4.X1.T = C.g                                                                     # replay 의 'r' 갈래를 켬 (경계는 아래 ctl 이 반복 1 에만)
    cls = AORT.AORTSolver
    o_ctl = cls.ctl

    def ctl(s, op, it=0):
        cap = getattr(s, "_c3cap", None) is not None
        if op == 3 and it == 0:
            if cap and getattr(s, "_x1K", 0) == 1 and getattr(s, "_xbd", None) is not None:
                s._xbd[2:4].zero_()                                                    # 그래프 안: 지난 'r' 값 지움 (반복 1 이 안 돌면 0 으로 남음)
            T0 = S4.X1.T
            S4.X1.T = 0.0                                                              # speed4 의 반복 0 경계 (X3e4) 를 끔
            try:
                return o_ctl(s, op, it)
            finally:
                S4.X1.T = T0
        r = o_ctl(s, op, it)
        if op == 3 and it == 1 and cap and getattr(s, "_x1K", 0) == 1:
            assert C3.mark is not None
            S4._xb("xb_r", 1, (_P(s.IP), _P(s.SC), _P(s.CD), np.float64(C.g), _P(s._xbd)))
            S4._d2h(s)
            C3.mark(s, "r")
        return r
    cls.ctl = ctl
    o_rep = C3.SegGraph.replay

    def replay(self):
        s = self.s
        xn = getattr(s, "_xbn", None)
        if xn is not None:
            xn[2] = 0; xn[3] = 0                                                       # 호스트 사본도 (반복 1 구간이 건너뛰어지면 지난 값이 남지 않게)
        e0 = S4.X1.x3e
        r = o_rep(self)
        if S4.X1.x3e > e0:
            C.gcnt += S4.X1.x3e - e0
        return r
    C3.SegGraph.replay = replay


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
                    w_frame_end(s, S6.W.fr)
        return w
    ART.split_step = outer(ART.split_step)
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)


def install(out, w=0, g=0.0):
    """실행기를 runpy 로 띄우는 순간, speed6·speed7 설치 뒤 (가장 바깥) 에 부름."""
    C.out = out
    C.log = open(os.path.join(out, "s8w.jsonl"), "w")
    if int(w):
        w_install(out, int(w))
    if float(g) > 0:
        g_install(float(g))
    frame_install()
    print(f"[절 CM74] speed8 켬 워프 모으기 {C.w} X3g {C.g}", flush=True)
