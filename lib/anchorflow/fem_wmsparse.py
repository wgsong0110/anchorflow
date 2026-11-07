"""절 CM30 (조정자 결정 11-08, 위임 범위 안 정의 변경): 수박 성긴 가우스 점 정의 + 장벽 실현 가능 보폭 국소화.
옛 재질 실행기 (fem_ao_rt.AORTSolver) 에만, s.wf["ws"] = dict(oct, red, bar) 가 있을 때만 -- 없으면 소스·실행 글자 그대로.
장벽 바닥 (--fl_bar) 판 전제: 팔분 칸 실제 입자 질량 = 표본점 버퍼 AF_BX[q*4+3] (wb_p2q, 프레임 시작 P2G, rt_gauss 앞) 을 그대로 씀.

(a) oct = κ  점유 가중 (새 실행기 --gq_oct κ 와 같은 식, 절 AN.err4): 가우스 점 부피·질량 = (1 − κ) x 그 팔분 칸에 실제로 든 입자 부피·질량
             + κ x 칸 합 / 8, 노드 운동량·가속도 몫도 같은 가중 (gq_pq). κ = 0 이면 입자 없는 팔분 칸 점은 V_q = m_q = 0 -- 요소 힘·헤시안·
             뒤집힘 필터 (aw_inv_q 는 V_q ≤ 0 점을 건너뜀)·장벽 (표본점 없음) 어디에도 안 들어감.
    red      재분배 판: 빈 팔분 점 0, 칸 합을 입자 있는 팔분 점에 똑같이 (칸 합 / 입자 있는 팔분 수), 운동량도 같은 몫 (칸 평균 속도).
    두 판 모두 입자 F·재질 상태 보간 (--fp_interp fpi_p, --lj_interp lji_p) 의 칸 안 8 점 삼선형 가중에서 빈 팔분 점을 빼고 다시 정규화
    (빈 점은 적분에서 빠져 F_q 가 묶이지 않으므로 그 값을 입자에 옮기지 않음).
(b) bar = R  장벽 실현 가능 보폭 국소화 (절 CM29 (a) 노드별 보폭과 같은 방식): 원래는 온 걸음 후보에서 관통할 (d ≤ 0) 활성 표본점마다 d > 0 인
             최소 k_q 를 구해 몸 전체 보폭 min(뒤집힘 필터 2^-kg, 2^-max k_q). 여기서는 노드 보폭 2^-kn 를 모두 kg (전역 필터 그대로) 로 시작해
             확인 R 라운드: 후보 y + 2^-kn ∘ T δ 에서 (1) 관통할 표본점은 그 27 노드에 kn + k_q, (2) 뒤집힘 필터 판정 (aw_inv_q 와 같은 식, --ls_end 포함)
             을 어기는 가우스 점도 그 27 노드에 kn + k_q (노드마다 보폭이 달라 방향이 바뀌므로). 최종 확인에서도 어기면 그 반복은 원래 전역 보폭
             2^-max(kg, kb) 로 되돌림. SC[14] (덜 풀린 걸음 판정) = 노드 보폭 최소.
"""
import numpy as np
import torch

from .fem_ao_rt import P

WS_G = r"""
#define WS_G __global__
"""

# ---- (a) 소스 바꾸기 ----
GA_OLD = "    Vq[q] = ac[28]*0.125f; float mm = ac[29]*0.125f; mq[q] = mm;\n"
GA_OCT = ("    Vq[q] = ac[28]*0.125f; float mm = ac[29]*0.125f;\n"
          "    { float mo_ = AF_BX[(size_t)q*4+3]; Vq[q] = (1.f - WS_K)*aq[27] + WS_K*Vq[q]; mm = (1.f - WS_K)*mo_ + WS_K*mm; }   // 절 CM30 (a) 점유 가중\n"
          "    mq[q] = mm;\n")
GA_RED = ("    Vq[q] = ac[28]*0.125f; float mm = ac[29]*0.125f;\n"
          "    { float fr_ = ws_fr(q); Vq[q] = ac[28]*fr_; mm = ac[29]*fr_; }                      // 절 CM30 (a) 재분배\n"
          "    mq[q] = mm;\n")
PQ_OLD = "pq[a] = oct ? ((1.f - kap)*PQ[(size_t)q*6+a] + kap*pc) : pc;"
PQ_OCT = "pq[a] = (1.f - WS_K)*PQ[(size_t)q*6+a] + WS_K*pc;"
PQ_RED = "pq[a] = PC[(size_t)c*6+a]*ws_fr(q);"
FR_SRC = r"""
// 절 CM30 (a) 재분배: 입자 있는 팔분 점 몫 = 1 / (그 칸 입자 있는 팔분 수), 빈 팔분 0 (팔분 질량 = 표본점 버퍼 AF_BX[.,3])
__device__ __forceinline__ float ws_fr(int q) {
    int c = q >> 3; if (!(AF_BX[(size_t)q*4+3] > 0.f)) return 0.f;
    int n = 0; for (int o=0;o<8;o++) if (AF_BX[(size_t)(c*8+o)*4+3] > 0.f) n++;
    return 1.f/(float)n;
}
"""
# fpi_p (절 CM20) · lji_p (절 CM21): 빈 팔분 점 빼고 다시 정규화
FP_A_OLD = "    bool anyb = false;\n    for (int oc=0; oc<8; oc++) {\n"
FP_A_NEW = "    bool anyb = false; float ws_w = 0.f;\n    for (int oc=0; oc<8; oc++) {\n"
FP_B_OLD = "        if (!(w > 0.f)) continue;\n        const float* e = Fe + (size_t)(c*8 + oc)*28;\n"
FP_B_NEW = "        if (!(w > 0.f) || !(AF_BX[(size_t)(c*8 + oc)*4+3] > 0.f)) continue;\n        ws_w += w;\n        const float* e = Fe + (size_t)(c*8 + oc)*28;\n"
FP_C_OLD = "    float out[9]; bool bad = false;\n    rs_fin(sR, sL, 1.f, out, &bad);\n"
FP_C_NEW = ("    if (ws_w > 0.f) { float iw_ = 1.f/ws_w; for (int k=0;k<9;k++) { sR[k] *= iw_; sL[k] *= iw_; sX[k] *= iw_; } for (int k=0;k<3;k++) s3[k] *= iw_; }\n"
            "    float out[9]; bool bad = false;\n    rs_fin(sR, sL, 1.f, out, &bad);\n")
LJ_A_OLD = "    float s = 0.f;\n    for (int oc=0; oc<8; oc++) {\n"
LJ_A_NEW = "    float s = 0.f, ws_w = 0.f;\n    for (int oc=0; oc<8; oc++) {\n"
LJ_B_OLD = "        if (!(w > 0.f)) continue;\n        s += w*Sq[(size_t)(c*8 + oc)*4+1];\n    }\n    ljp[p] = s;\n"
LJ_B_NEW = "        if (!(w > 0.f) || !(AF_BX[(size_t)(c*8 + oc)*4+3] > 0.f)) continue;\n        ws_w += w; s += w*Sq[(size_t)(c*8 + oc)*4+1];\n    }\n    ljp[p] = ws_w > 0.f ? s/ws_w : s;\n"

# ---- (b) 커널 ----
WS_SRC = r"""
// ---- 절 CM30 (b) 장벽 실현 가능 보폭 국소화 ----
// 노드 k 시작값 = 전역 필터 k (SC[14] = 2^-kg)
__global__ void ws_init(const int* IP, const double* SC, int* kn, int* kn2, int* kg) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int k = -ilogb(SC[14]); if (k < 0) k = 0;
    if (n == 0) kg[0] = k;
    if (n >= IP[9]) return;
    kn[n] = k; kn2[n] = 0;
}
// 후보 y1 = ŷ + T (η + 2^-kn δ), 방향 dd = 2^-kn T δ (td = T δ)
__global__ void ws_cand(const int* IP, const int* kn, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* eta, const float* dl,
                        const float* td, float* y1, float* dd) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float s_ = ldexpf(1.f, -kn[n]);
    float e[3]; for (int a=0;a<3;a++) e[a] = eta[n*3+a] + s_*dl[n*3+a];
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) { y1[n*3+a] = yh[n*3+a] + t[a]; dd[n*3+a] = s_*td[n*3+a]; }
}
// 관통할 활성 표본점 (rt_bar_kfe 와 같은 식) -> 그 27 노드 kn2 = max(kn2, kn + k_q), 어긴 점 수 fl
__global__ void __launch_bounds__(128) ws_bk(const int* IP, const int* cb, const float* BW, const float* y0, const float* y1, const float* U, float thinv,
                         float n0, float n1, float n2, float pn, const int* kn, int* kn2, int* fl, int scat) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3];
    if (!(bm > 0.f) || !AF_BACT[q]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xa[3], xb[3];
    for (int a=0;a<3;a++) { xa[a] = AF_BX[q*4+a]/bm; xb[a] = xa[a]; }
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
        for (int a=0;a<3;a++) { xa[a] += w*(U[id*3+a] + thinv*(y0[id*3+a] - U[id*3+a])); xb[a] += w*(U[id*3+a] + thinv*(y1[id*3+a] - U[id*3+a])); } }
    float d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn, d1 = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn;
    if (d1 > 0.f) return;
    int k = 0; float c = 1.f;
    while (true) { c *= 0.5f; k++; if (c == 0.f) break; if (d0 + c*(d1 - d0) > 0.f) break; }
    atomicAdd(&fl[0], 1);
    if (scat) for (int o=0;o<27;o++) if (BW[oc*27+o] != 0.f) atomicMax(&kn2[ids[o]], min(kn[ids[o]] + k, 60));
}
// 뒤집힘 판정을 어기는 가우스 점 (ws_inv_q 의 kq > 0) -> 27 노드 kn2 = max(kn2, kn + k_q), fl[1]
__global__ void ws_nk(const int* IP, const int* cb, const int* kq, const int* kn, int* kn2, int* fl, int scat) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int k = kq[q]; if (k <= 0) return;
    atomicAdd(&fl[1], 1);
    if (!scat) return;
    int ids[27]; qnodes(q, IP, cb, ids);
    for (int o=0;o<27;o++) atomicMax(&kn2[ids[o]], min(kn[ids[o]] + k, 60));
}
__global__ void ws_merge(const int* IP, int* kn, int* kn2) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    kn[n] = max(kn[n], kn2[n]); kn2[n] = 0;
}
// 최종 확인: 어긴 점이 있으면 (fl[0] + fl[1] > 0) 원래 전역 보폭 max(kg, kb) 로 되돌림; 노드 k 최대·자른 노드 수
__global__ void ws_fb(const int* IP, const int* fl, const int* kg, const int* kb, int* kn, int* km, int* st) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    bool fb = (fl[0] + fl[1]) > 0;
    int kgl = max(kg[0], kb[0]);
    if (fb) kn[n] = kgl;
    atomicMax(km, kn[n]);
    if (!fb && kn[n] > kg[0]) atomicAdd(&st[3], 1);
    if (n == 0) { st[4] += 1; if (fb) st[1] += 1; st[5] = max(st[5], kg[0]); st[6] = max(st[6], kb[0]); }
}
__global__ void ws_sc(double* SC, const int* km, const int* kg, int* st) {
    SC[14] = (double)ldexpf(1.f, -km[0]); st[2] = max(st[2], km[0]);
}
// 갱신 = rt_lsup (full 0) 과 같은 식, 보폭만 노드별 2^-kn; cmp 1 = 압축 판 (AO_ACT 노드만)
__global__ void ws_lsup(const int* IP, const int* kn, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, float* eta, const float* dl, float* y, int cmp) {
    int n;
    if (cmp) { int ii_ = blockIdx.x*blockDim.x + threadIdx.x; if (ii_ >= IP[14]) return; n = AO_ACT[ii_]; }
    else { n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= IP[9]) return; }
    float st = ldexpf(1.f, -kn[n]);
    float e[3]; for (int a=0;a<3;a++) { e[a] = eta[n*3+a] + st*dl[n*3+a]; eta[n*3+a] = e[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y[n*3+a] = yh[n*3+a] + t[a];
}
WS_G void ws_zero(int n, int* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = 0; }
"""


def _rep(s, o, n, c=1):
    assert s.count(o) == c, (o[:80], s.count(o))
    return s.replace(o, n)


def _inv_q():
    """aw_inv_q (절 AS) 글자 그대로 + 점별 k 출력 kq (fem_wminv._inv_q 와 같은 바꾸기, 이름만 ws_)."""
    from .fem_ao_rt import AW_SRC
    i = AW_SRC.index("__global__ void aw_inv_q(")
    j = AW_SRC.index("\n}\n", i) + 3
    b = AW_SRC[i:j].replace("aw_inv_q(", "ws_inv_q(", 1)
    b = _rep(b, "float e1, float e0) {", "float e1, float e0, int* kq) {")
    b = _rep(b, "    if (k > 0) atomicMax(&kk[0], k);\n", "    kq[q] = k;\n    if (k > 0) atomicMax(&kk[0], k);\n")
    return b


def apply(r, ws):
    """r = fem_ao_rt._src 결과 (skipify 앞). ws = dict(oct κ 또는 None, red bool, bar R)."""
    if not ws:
        return r
    assert "__device__ float* AF_BX;" in r, "절 CM30 은 장벽 바닥 (--fl_bar) 판 위에만"
    oc, red = ws.get("oct"), bool(ws.get("red"))
    if oc is not None or red:
        assert not (oc is not None and red), "--gq_oct 와 --gq_ored 는 따로"
        assert "gq_q2n(" in r, "절 CM30 (a) 는 --gq_mass 판 위에만"
        if red:
            r = _rep(r, GA_OLD, GA_RED)
            r = _rep(r, PQ_OLD, PQ_RED)
            i = r.index("__global__ void rt_gauss(")
            r = r[:i] + FR_SRC + r[i:]
        else:
            r = _rep(r, GA_OLD, GA_OCT)
            r = _rep(r, PQ_OLD, PQ_OCT)
            r = r.replace('extern "C" {', 'extern "C" {\n#define WS_K %.9ef' % float(oc), 1)
        if "__global__ void fpi_p(" in r:
            for o_, n_ in ((FP_A_OLD, FP_A_NEW), (FP_B_OLD, FP_B_NEW), (FP_C_OLD, FP_C_NEW)):
                r = _rep(r, o_, n_)
        if "__global__ void lji_p(" in r:
            for o_, n_ in ((LJ_A_OLD, LJ_A_NEW), (LJ_B_OLD, LJ_B_NEW)):
                r = _rep(r, o_, n_)
    if int(ws.get("bar", 0)) > 0:
        r = r.replace('extern "C" {', 'extern "C" {' + WS_G, 1)
        i = r.rfind("}")
        r = r[:i] + _inv_q() + WS_SRC + r[i:]
    return r


def bar_on(s):
    return int(getattr(s, "wf", {}).get("ws", {}).get("bar", 0)) > 0


def alloc(s):
    """fem_ao_rt.AORTSolver.alloc 끝에서 (용량 바뀔 때마다)."""
    if not bar_on(s):
        return
    I = lambda *sh: torch.zeros(*sh, device=s.dev, dtype=torch.int32)
    s.wskn = I(s.capN); s.wskn2 = I(s.capN); s.wskq = I(s.capQc * 8); s.wskk = I(2)
    s.wskg = I(1); s.wskm = I(1); s.wsfl = I(64)
    s.wsy1 = torch.zeros(s.capN, 3, device=s.dev, dtype=torch.float32); s.wsdd = torch.zeros(s.capN, 3, device=s.dev, dtype=torch.float32)
    s.wsst = I(8)


def frame_stats(s):
    """마지막 시도 통계: [-, 되돌림 반복, 최대 노드 k, 자른 노드 합 (kn > kg), 반복 수, 최대 필터 kg, 최대 전역 장벽 kb, -]."""
    return s.wsst.cpu().tolist() if hasattr(s, "wsst") else None


def newton(s, kstep, it, capN, capQ, nr, thv, WB, NR):
    """뉴턴 반복 끝 (원래 뒤집힘 필터 블록 다음, SC[14] = 2^-kg, s.awtd = T δ). 원래 장벽 보폭·rt_lsup 를 대신함 (부르는 쪽이 건너뜀)."""
    i32, f = np.int32, np.float32
    IP = s.IP; R = int(s.wf["ws"]["bar"]); F3 = [f(q) for q in nr]
    assert WB is not None and s.lsinv
    if kstep == 0 and it == 0:
        s.k("ws_zero", 8, (i32(8), P(s.wsst)), b=32)
    # 원래 전역 장벽 k (온 걸음 후보) -- 되돌림·진단용
    s.k("rt_ytry", capN, (P(IP), P(s.sk), P(s.pm), *F3, P(s.yh), P(s.eta), P(s.dl), P(s.wbyc)), b=256)
    s.zf(s.kk)
    s.k("rt_bar_kfe", capQ, (P(IP), P(s.cb), P(s.BW), P(s.y), P(s.wbyc), P(s.U), thv, *F3, f(WB["PN"]), P(s.kk)))
    s.k("ws_init", capN, (P(IP), P(s.SC), P(s.wskn), P(s.wskn2), P(s.wskg)), b=256)
    s.zf(s.wsfl)
    aend = s._aw_end()
    bargs = lambda r_, sc: (P(IP), P(s.cb), P(s.BW), P(s.y), P(s.wsy1), P(s.U), thv, *F3, f(WB["PN"]), P(s.wskn), P(s.wskn2), P(s.wsfl[2 * r_:]), i32(sc))

    def check(r_, sc):
        s.k("ws_cand", capN, (P(IP), P(s.wskn), P(s.sk), P(s.pm), *F3, P(s.yh), P(s.eta), P(s.dl), P(s.awtd), P(s.wsy1), P(s.wsdd)), b=256)
        s.zf(s.wskq); s.zf(s.wskk)
        s.k("ws_inv_q", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), P(s.wsdd), P(s.wskk)) + aend + (P(s.wskq),))
        s.k("ws_nk", capQ, (P(IP), P(s.cb), P(s.wskq), P(s.wskn), P(s.wskn2), P(s.wsfl[2 * r_:]), i32(sc)))
        s.k("ws_bk", capQ, bargs(r_, sc))
    for r_ in range(R):
        check(r_, 1)
        s.k("ws_merge", capN, (P(IP), P(s.wskn), P(s.wskn2)), b=256)
    check(R, 0)
    s.zf(s.wskm)
    s.k("ws_fb", capN, (P(IP), P(s.wsfl[2 * R:]), P(s.wskg), P(s.kk), P(s.wskn), P(s.wskm), P(s.wsst)), b=256)
    s.k("ws_sc", 1, (P(s.SC), P(s.wskm), P(s.wskg), P(s.wsst)), b=1)
    if NR:
        j = kstep * s.nt + it
        s.NRkb[j:j + 1].copy_(s.kk[:1]); s.NRst[j:j + 1].copy_(s.SC[14:15])
    s.k("ws_lsup", capN, (P(IP), P(s.wskn), P(s.sk), P(s.pm), *F3, P(s.yh), P(s.eta), P(s.dl), P(s.y), i32(1 if s.compact else 0)), b=256)
