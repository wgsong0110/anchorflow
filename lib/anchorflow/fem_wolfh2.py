"""절 CM63 묶음 wolfh2: wolf 높게 쌓임 -- 공통 후보 G' (F + A3 + P1X + H5 + Rf) 위에서 바닥 풀기 (N·PT2) + 벌점 기전 가르기 + 정답 sticky 에 가까운 고착 몫 S.
감싸개 exe/wolfh2_wrap.py 가 실행기 (fem_an_rt_sim.py) 를 띄우는 순간 install() (다른 감싸개가 모두 설치된 뒤 = ghost6 의 환경 변수·k 감싸기 다음). 인자 없으면 이 모듈을 안 부름.

n  (--w2n 1)  N (절 CM52, lib/anchorflow/fem_wolfh1.py 그대로): 국소 시작점 섞기 β 를 바닥 법선 성분에만 (접선 시작 = 관성 예측).
s  (--w2s 1)  S (정의 변경, --fl_pt 위): 팔분 표본점 q 의 고착 몫 = 그 팔분 칸 질량의 바닥 법선 높이 분포 위에서 정답 sticky 점별 고착 1 − (1 − w_gt(z))^κ 의 평균,
              노드 k = 1 − r̄ (--fl_pt 와 같음 -- 노드 거듭제곱 없음; κ = Δt/dt_정답 = AF_KEXQ).
     φ_q = ∫ B₂(t) [1 − (1 − w_gt(d_q + p_n + t δ))^κ] dt,   δ = h / (2√3)  (h = 우리 칸 = --cell × dx_정답).
     근거: 우리 격자 (simplex.lattice, HZ_CUBE) 의 칸은 몸 대각선이 바닥 법선인 정육면체 (세 기저 모두 법선 성분 h/√3). 팔분 칸 (모서리 h/2) 을 질량이 고르게 채우면
     법선 높이 = 질량중심 + (h/(2√3)) (u₁ + u₂ + u₃ − 3/2), uᵢ 균등 -> 분포가 정확히 간격 δ 의 이차 B-스플라인 (Irwin–Hall 3). w_gt 는 정답 MPM (dx_정답 격자) 의
     sticky 가중 (평면 아래 정답 노드의 이차 B-스플라인 몫 합 -- rt_floor_q 와 같은 식), 1 − (1 − w)^κ 는 한 프레임 κ 서브스텝 동안 정답 입자가 잃는 속도 몫.
     즉 '정답 격자 간격 (dx_정답) 의 고착 층이 우리 팔분 칸 (법선 두께 3δ) 질량 중 차지하는 몫'. 띠 고착 (--fl_bst) 의 띠 안 완전 고착 (몫 1) 도 이것으로 바뀜.
     적분은 [−1.5, 1.5] 64 점 중점 규칙 (이산화; 받침 밖 (d_q − 1.5 δ ≥ 1.5 dx_정답) 이면 정확히 0 이라 건너뜀). 문턱·클램프 없음 (fminf(w, 1) 은 기존 PT 와 같은 반올림 막이).
     걸음 쪼개기 (ksplit frac < 1) 에서도 κ 는 프레임 값 (기존 --fl_pt 와 같은 한계).
gk0 (--w2gk0 1)  진단 (기전 가르기): ghost penalty 강성 k = 0 (고스트 질량만 남김).
gs  (--w2gs 1)   진단·후보: 면 36 노드 중 바닥 고착 몫 an > 0 인 노드가 있는 면을 뺌 (P1X 의 구속 노드 면 빼기 @g4x 와 같은 꼴; 걸음마다 rt_floor_n·mt_floor_n 뒤, 프레임 면 표지 위 멱등).
"""
import atexit
import json
import os

import numpy as np

ST = dict(n=0, s=0, gk0=0, gs=0, cell=None, out=".", nstk=0, said=False)

FLJ_OLD = "    mw = mq[q]*(1.f - expf(AF_KEXQ*log1pf(-fminf(wst, 1.f))));"
FLJ_NEW = ("#ifdef AF_W2S\n"
           "    mw = mq[q]*w2_phi(sw_, pn, dxg);                                           // 절 CM63 S: 팔분 칸 높이 분포 위 정답 sticky 점별 고착 몫 평균\n"
           "#else\n" + FLJ_OLD + "\n#endif")

S_DEV = r'''
// ---- 절 CM63 S (lib/anchorflow/fem_wolfh2.py): 팔분 표본점 고착 몫 φ = ∫ B₂(t) [1 − (1 − w_gt(z_q + t δ))^κ] dt, δ = AF_W2C dx_정답 / (2√3) ----
#if defined(AF_W2S) && defined(AF_FLJ)
__device__ __forceinline__ float w2_wgt(float zc, float pn, float dxg) {          // 정답 sticky 가중 (rt_floor_q 와 같은 식): zc = 법선 좌표 x·n
    float u = zc/dxg; float bf = floorf(u - 0.5f); float f = u - bf;
    float w3[3] = {0.5f*(1.5f-f)*(1.5f-f), 0.75f-(f-1.f)*(f-1.f), 0.5f*(f-0.5f)*(f-0.5f)};
    float w = 0.f; for (int i=0;i<3;i++) if ((bf + (float)i)*dxg - pn < 0.f) w += w3[i];
    return w;
}
__device__ float w2_phi(float sw, float pn, float dxg) {
    const float dl = AF_W2C*dxg*0.288675134594813f;                               // δ = h/(2√3), h = AF_W2C dx_정답
    if (sw - 1.5f*dl >= 1.5f*dxg) return 0.f;                                     // 받침 밖: 평면 아래 정답 노드는 모두 < p_n 이라 z ≥ p_n + 1.5 dx 에서 w_gt = 0 (정확히 0)
    float acc = 0.f;
    for (int i=0;i<64;i++) {
        float t = -1.5f + (i + 0.5f)*(3.f/64.f); float at = fabsf(t);
        float b = (at < 0.5f) ? (0.75f - t*t) : 0.5f*(1.5f - at)*(1.5f - at);
        float w = w2_wgt(sw + pn + t*dl, pn, dxg);
        acc += b*(1.f - expf(AF_KEXQ*log1pf(-fminf(w, 1.f))));
    }
    return acc*(3.f/64.f);
}
#endif
'''

GS_SRC = r'''
// ---- 절 CM63 gs (lib/anchorflow/fem_wolfh2.py): 면 36 노드 (칸 27 + 그 축 바깥 층 9) 중 바닥 고착 몫 an > 0 노드가 있으면 면 표지 비트를 지움 (fem_ghost4 gq_fix 와 같은 배치) ----
__global__ void gq_w2stk(const int* IP, const int* cb, const float* an, unsigned char* fm, int* cnt) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    if (!m) return;
    int nb1 = IP[7], nb2 = IP[8]; int s1 = nb2, s0 = nb1*nb2;
    int base = (cb[c*3]*nb1 + cb[c*3+1])*nb2 + cb[c*3+2];
    int cf = 0;
    for (int o=0;o<27;o++) if (an[base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3)] > 0.f) { cf = 1; break; }
    int m2 = m, nr = 0;
    for (int a=0;a<3;a++) {
        if (!(m & (1 << a))) continue;
        int f = cf;
        for (int i=0;i<3 && !f;i++) for (int j=0;j<3 && !f;j++) {
            int id = (a == 0) ? (base + 3*s0 + i*s1 + j) : ((a == 1) ? (base + i*s0 + 3*s1 + j) : (base + i*s0 + j*s1 + 3));
            if (an[id] > 0.f) f = 1;
        }
        if (f) { m2 &= ~(1 << a); nr++; }
    }
    if (m2 != m) { fm[c] = (unsigned char)m2; atomicAdd(&cnt[0], nr); }
}
'''


def patch_src(src):
    """build_src 출력에 S·gs 를 넣음 (S: rt_floor_q 의 --fl_pt 줄을 바꿈, gs: 커널을 모듈 끝에)."""
    if ST["s"]:
        assert "#define AF_FLJ 1" in src, "절 CM63 S 는 --fl_pt 위"
        i = src.index("__global__ void rt_floor_q(")
        j = src.index("\n}\n", i) + 1
        assert src[i:j].count(FLJ_OLD) == 1, "절 CM63 S: rt_floor_q 의 --fl_pt 줄을 못 찾음"
        src = src[:i] + S_DEV + src[i:j].replace(FLJ_OLD, FLJ_NEW, 1) + src[j:]
        src = src.replace('extern "C" {', 'extern "C" {\n#define AF_W2S 1\n#define AF_W2C %.9ef' % float(ST["cell"]), 1)
    if ST["gs"] and "gq_w2stk(" not in src:
        k = src.rstrip().rfind("}")
        src = src[:k] + GS_SRC + "\n}\n"
    return src


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def install(n=0, s=0, gk0=0, gs=0, cell=None, out="."):
    from . import fem_an_rt as ART
    ST.update(n=int(n), s=int(s), gk0=int(gk0), gs=int(gs), cell=cell, out=out)
    if ST["n"]:
        from . import fem_wolfh1 as H1
        H1.install(n=1)
    if ST["s"] or ST["gs"]:
        _ob = ART.build_src
        ART.build_src = lambda *a, **kw: patch_src(_ob(*a, **kw))
    if ST["gk0"]:
        from . import fem_ghost6 as G6
        assert G6.ST.get("on"), "절 CM63 gk0 는 ghost penalty (P1X) 위"
        _ok0 = G6._k_of

        def _k_of(s_):
            first = getattr(s_, "_w2k0", None) is None
            _ok0(s_)
            if first:
                s_._w2k0 = float(s_._gpk)
                print(f"[절 CM63] gk0: ghost penalty 강성 k {s_._gpk:.6g} -> 0 (고스트 질량 k_M {s_._g3kM:.6g} 그대로)", flush=True)
            s_._gpk = 0.0
            return s_._gpk
        G6._k_of = _k_of
    if ST["gs"]:
        from . import fem_ghost6 as G6
        assert G6.ST.get("on"), "절 CM63 gs 는 ghost penalty (P1X) 위"
        import torch
        for cls in (ART.RTSolver,):
            _k = cls.k

            def k(s_, name, nn, args, b=128, _k=_k):
                r = _k(s_, name, nn, args, b)
                if name in ("rt_floor_n", "mt_floor_n") and getattr(s_, "g3fm", None) is not None:      # 재질 판 (wolf 모래) 은 mt_floor_n
                    if getattr(s_, "w2cnt", None) is None:
                        s_.w2cnt = torch.zeros(2, device=s_.dev, dtype=torch.int32)
                        ST["sol"] = s_
                    _k(s_, "gq_w2stk", int(s_.capQc), (args[0], P(s_.cb), args[1], P(s_.g3fm), P(s_.w2cnt)), 128)
                return r
            cls.k = k
    print(f"[절 CM63] wolfh2 켬: N {ST['n']} · S {ST['s']} (셀 {cell}) · 강성 0 {ST['gk0']} · 고착 노드 면 빼기 {ST['gs']}", flush=True)

    def _fin():
        info = {k_: ST[k_] for k_ in ("n", "s", "gk0", "gs", "cell")}
        sol = ST.get("sol")
        if sol is not None:
            try:
                import torch
                torch.cuda.synchronize()
                info["gs_removed_faces_total"] = int(sol.w2cnt[0].item())
            except Exception as e:                                                       # noqa: BLE001
                info["gs_err"] = repr(e)
        json.dump(info, open(os.path.join(out, "w2info.json"), "w"))
    atexit.register(_fin)
