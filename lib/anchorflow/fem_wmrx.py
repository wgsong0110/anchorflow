"""절 CM38 재설계 후보 (수박 옛 실행기 exe/fem_ao_rt_sim.py 만; --wrx 0 이면 이 모듈을 안 부름 = 생성 소스 글자 그대로).

원인 (절 CM38.1): 무너진 걸음의 첫 반복 잔차는 모두 관성 몫 M η/τ² 이고, η 는 고착 띠 노드의 √k 척도 변수 (y = ŷ + T η, T 의 접선·전체 몫이 √k,
k = (1 − r)^KEX, KEX = dt / substep 300) -- 시작점이 걸음 시작 위치 U 쪽 (ao_floor_n y0uk·--y0loc) 이면 η = T⁻¹(U − ŷ) 가 1/√k 배로 부풂.
그 뉴턴 방향은 η 공간에서 −η (PCG 정확, 접선 일관) 이고 y 공간에서는 관통하는 ŷ 쪽이라 장벽·뒤집힘 필터가 보폭을 0 에 가깝게 자름 -> 진전 없음.

후보 (비트, 장치 전역 AF_WRXON 이 1 일 때만):
  1 = 고착 상한 (sk_cut): √k < AF_SKC 이면 √k = 0 (완전 고착 -- 그 노드는 붙은 목표 h 에 묶이고 η 자유도 없음; KEX → ∞ 극한과 같은 꼴, 1/√k 부풂 없음).
  2 = 속도 수준 접촉 (vproj): 띠 몫 r > 0 인 노드의 붙은 목표 h 에서 U 기준 들어가는 법선 이동을 r 만큼 지움 (MPM 격자 경계 조건과 같은 비탄성 충격을
      풀이 전에 먼저) -- 관성 예측 자체가 바닥을 덜 뚫어 시작점 U 와 ŷ 의 거리가 줄어듦.
켜는 방식: --wrx_col 0 = 모든 걸음, 1 = 무너진 프레임만 (K 1 시도가 ‖g‖²/F² > --ue_bigT 이면 그 프레임을 되돌리고 켠 채 K 1 부터 다시 쪼개기)."""
import math

import numpy as np

HEAD = "__device__ int AF_WRXON = 0;\n__device__ int AF_WRXB = 0;\n__device__ float AF_SKC = 0.f;\n"
A1 = "    float s = sqrtf(k);\n    float d[3];\n"
A1N = ("    float s = sqrtf(k);\n"
       "    if (AF_WRXON && (AF_WRXB & 1) && s < AF_SKC) s = 0.f;                                  // 절 CM38 후보 1: 고착 상한\n"
       "    float d[3];\n")
A2 = "    float h[3]; for (int a=0;a<3;a++) h[a] = Ut[n*3+a] + (1.f-k)*Pd[a];\n"
A2N = A2 + ("    if (AF_WRXON && (AF_WRXB & 2) && floor_) {                                              // 절 CM38 후보 2: 속도 수준 접촉 (들어가는 법선 이동을 띠 몫만큼 지움)\n"
            "        float r_ = (mqn[n] > 0.f) ? fminf(an[n]/mqn[n], 1.f) : 0.f;\n"
            "        float dn_ = (h[0]-U[n*3])*n0 + (h[1]-U[n*3+1])*n1 + (h[2]-U[n*3+2])*n2;\n"
            "        if (r_ > 0.f && dn_ < 0.f) { h[0] -= r_*dn_*n0; h[1] -= r_*dn_*n1; h[2] -= r_*dn_*n2; }\n"
            "    }\n")


KNAMES = ("ao_floor_n", "rt_floor_n", "mt_floor_n")                                 # 옛 실행기 (수박)·새 실행기 탄성·재질 판 바닥 노드 커널 (같은 식)
A1G = "    float s = sqrtf(k);\n"
A1GN = A1G + "    if (AF_WRXON && (AF_WRXB & 1) && s < AF_SKC) s = 0.f;                                  // 절 CM38 후보 1: 고착 상한\n"


def apply(src):
    """있는 바닥 노드 커널 몸통마다 같은 두 곳을 바꿈 (전역 선언은 첫 커널 앞 한 번)."""
    out, pos, head = [], 0, False
    for nm in KNAMES:
        key = "__global__ void " + nm + "("
        if key not in src:
            continue
        i = src.index(key)
        j = src.index("__global__", i + 10)
        body = src[i:j]
        assert body.count(A1G) == 1 and body.count(A2) == 1, f"--wrx: {nm} 글자가 바뀜"
        src = src[:i] + body.replace(A1G, A1GN, 1).replace(A2, A2N, 1) + src[j:]
        head = True
    assert head, "--wrx: 바닥 노드 커널 없음"
    i0 = min(src.index("__global__ void " + nm + "(") for nm in KNAMES if ("__global__ void " + nm + "(") in src)
    i0 = src.rfind("\n", 0, i0) + 1                                                   # 전역 선언은 가장 앞 바닥 커널 앞 (재질 판은 mt_floor_n 이 rt_floor_n 보다 앞)
    return src[:i0] + HEAD + src[i0:]


def install(new=False):
    """옛 실행기: fem_ao_rt._src 감쌈. new=True: 새 실행기 fem_an_rt.build_src 감쌈 (fem_mat_rt 도 같은 함수를 부름)."""
    if new:
        from . import fem_an_rt as ART
        if getattr(ART, "_wrx_on", False):
            return
        ART._wrx_on = True
        o_b = ART.build_src

        def build_src(*a, **kw):
            return apply(o_b(*a, **kw))
        ART.build_src = build_src
        return
    from . import fem_ao_rt as AORT
    if getattr(AORT, "_wrx_on", False):
        return
    AORT._wrx_on = True
    o_src = AORT._src

    def _src(*a, **kw):
        return apply(o_src(*a, **kw))
    AORT._src = _src


def _set(s, name, val, dt):
    from . import fem_an_rt as ART
    g = s.mod.get_global(name)
    ART.cp.ndarray((1,), dtype=dt, memptr=g)[0] = val


def setup(s, bits, skc):
    from . import fem_an_rt as ART
    s.wrx = dict(bits=int(bits), skc=float(skc))
    _set(s, "AF_WRXB", int(bits), ART.cp.int32)
    _set(s, "AF_SKC", float(skc), ART.cp.float32)


def on(s, v):
    from . import fem_an_rt as ART
    _set(s, "AF_WRXON", int(v), ART.cp.int32)
    s._wrxv = int(v)                                                                   # 절 CM192: 호스트 쪽 기억 (exe/ssfix_wrap.py --ssf_wrxr 1 이 다시 풀기 뒤 되돌림)


def split_col(s, t, h, sub_dt, depth, fn=None):
    """무너진 프레임만 켬: 끈 채 K 1 -> (무너진 걸음 있으면) 되돌리고 켠 채 split_step (K 1 부터 2^depth 까지). 돌려줌: (다시 했는지, K, 시도 목록 [(K, 무너진 걸음, ms, 켬)])."""
    if fn is None:                                                                     # 옛 실행기 split_step (t, h, sub_dt, depth); 새 실행기는 fn = lambda d: ART.split_step(s, t, h, d)
        from . import fem_ao_rt as AORT
        fn = lambda d: AORT.split_step(s, t, h, sub_dt, d)
    bk = s._bk()
    on(s, 0)
    r0, K0, tr0 = fn(0)
    tries = [(a, b, c, 0) for (a, b, c) in tr0]
    if tr0[-1][1] == 0 or not math.isfinite(tr0[-1][2]):
        return r0, K0, tries
    tot0 = sum(q[2] for q in tr0)
    s._rs(bk)
    on(s, 1)
    r1, K1, tr1 = fn(depth)
    on(s, 0)
    tries += [(a, b, c, 1) for (a, b, c) in tr1]
    s.last_ms = tot0 + float(s.last_ms)
    return (r0 or r1), K1, tries
