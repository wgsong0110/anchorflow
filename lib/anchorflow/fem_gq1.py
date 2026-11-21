"""절 CM194 (gq_justify, 조정자 11-16 위임): 칸당 가우스 점 2x2x2 의 정당화 실험 -- 1x1x1 (칸 가운데 1 점) 과 모래시계 억제 벌점을
기본 판과 같은 실행기·같은 CUDA 커널·같은 그래프 (칸당 8 레인 배치 그대로) 위에 얹는 모듈. 끄면 (모두 0) 아무것도 안 바꿈.

(1) 1x1x1 (--gq_q1 1): 칸 8 레인 배치는 그대로 두고 8 레인이 모두 같은 점 (칸 가운데) 을 맡게 함.
    - 고운 층 기저 표 BW·BG: 8 줄 모두 칸 가운데 (국소 좌표 f = 1) 의 값·월드 기울기. 거친 층 (전처리기, rt_mg_·cvc 커널) 은 원래 2x2x2 표 (절 CM179 q1pt 와 같음).
    - rt_gauss: 가우스 점 F 는 칸 전체 입자 평균 (팔분 평균 대신), V_q = 칸 부피 / 8·m_q = 칸 질량 / 8 (점유 가중 κ 없음 -- 1 점이면 (1 − κ)·칸 실제 + κ·칸 합 = 칸 부피),
      점 위치 xq0 = 칸 가운데 (GG = 0). 재질 판 (소성) 가우스 점 상태 aq_gauss_s 도 칸 평균.
    -> 8 레인의 X_q 가 모두 같아 에너지 = V_c Ψ(F_c), 힘·헤시안·블록 대각·선탐색·장벽 표본점이 모두 1 점 규칙과 같음. 입자 F 보간 (fpi) 은 8 점이 같은 값이라 F_c.
       절 CM179 q1pt (exe/q27_wrap.py --q1pt 1, 칸당 32 자리 중 1 자리) 와 같은 정의, 계산 배치만 기본 판 (8 자리) 것.
(2) 모래시계 억제 (--gq_hg γ, 1x1x1 위에서만): E_hg = (γ μ / 2) Σ_칸 Σ_{g ∈ 2x2x2} (V_c / 8) ‖F_g − F_c‖²,
    F_g = (I + Σ_n y_n ⊗ ∇w_n(ξ_g)) F_c^0, F_c = (I + Σ_n y_n ⊗ ∇w_n(ξ_c)) F_c^0 -- 칸 안 2x2x2 가우스 점 변형 기울기와 칸 (가운데 1 점) 변형 기울기의 차에 γμ 를 곱한 벌점.
    걸음 변위 y 의 이차식 (프레임 시작 F_c^0 고정) -> 칸 강성 K_hg = γ μ Σ_g (V_c/8) (B_g − B_c)ᵀ (B_g − B_c) ⊗ I, Flanagan–Belytschko (1981) 모래시계 강성과 같은 꼴
    (그쪽 모래시계 기저 대신 2x2x2 점 기울기 차). 격자가 프레임마다 새로 깔리므로 프레임 사이에 모래시계 힘을 들고 가지 않음 (FB 의 증분 강성에서 누적 없는 판).
    실행기 연결 = 절 CM105 fem_edgep 과 같은 자리 (클래스 수준 RTSolver.k 감싸기): 요소 커널 (rt_elem·aq_elem·ao_elem·ao_elem_s) 뒤 힘·에너지·블록 대각,
    헤시안 곱 (rt_mv·rt_mv2·rt_mv2s) 뒤 out += K_hg v, rt_bd 뒤 블록 대각, 선탐색 후보 rt_elem_lsN 뒤 후보 에너지. 모두 그래프에 함께 캡처 (시간에 포함).
(3) 모래시계 에너지 비 (--gq_hgm 1, 진단, 결과 불변, 그래프 밖): 프레임 (s.step) 끝 노드 변위 U 로 점유 칸마다 2x2x2 점 변위 기울기 G_g (월드),
    칸 평균 Ḡ; ψ(G) = |sym G|² + ν/(1 − 2ν) (tr G)² (선형 등방 탄성 에너지 / μ); E_hg = Σ_칸 Σ_g ψ(G_g − Ḡ), E_tot = Σ ψ(G_g) (칸 무게 같음).
    Σ_g (G_g − Ḡ) = 0 이라 E_tot = E_평균 + E_hg (0 ≤ 비 ≤ 1). 칸 평균 기울기가 0 인 성분 = 1 점 규칙이 못 보는 칸 단위 변형 (강체 운동은 ψ 가 0).
    가우스 점 배치와 무관 (칸 목록 cb·노드 U 만) -> 모든 판 (1x1x1, 2x2x2, 3x3x3 32 자리) 에 같은 자.
(4) 충격 x s (--gq_vs s): 자료 cfg 의 중력 x s² (낙하 충격 속도 x s), 경계 조건 속도 (enforce_particle_translation velocity, 회전 구동 rotation/translation scale) x s.
"""
import json
import math
import os

import numpy as np

CFG = dict(q1=0, hg=0.0, hgm=0, vs=1.0, ck=0)
INFO = dict(cfg=None, mods=0, gauss_patch=0, sgauss_patch=0, hooks={}, names={}, solvers=[], err=None, check=None, coarse_swap=0)
OUT = ["."]
ELEM = ("rt_elem", "ao_elem", "ao_elem_s", "aq_elem")
MV = ("rt_mv", "rt_mv2", "rt_mv2s")
COARSE_K = ("rt_mg_", "ao_mg_", "cvc")
UNSUP = ("rt_elem_ls", "en_phi_n", "ao_mv_q", "ao_mv_h", "ao_mv_c", "ao_mv_s", "rt_mg_mv2_f")   # 이 판에서 안 고친 에너지·곱 커널 -- 불리면 멈춤 (벌점 판만)

SRC = r'''
// ---- 절 CM194 모래시계 억제 벌점 (lib/anchorflow/fem_gq1.py): E = ½ kg V_q ‖Σ_n y_n ⊗ g_n‖², g_n = F_c^0ᵀ (b_g,n − b_c,n) ----
__device__ __forceinline__ void hg_g(const float* F, const float* B8, const float* Bc, int oc, int o, float* g) {
    float d0 = B8[(oc*27+o)*3] - Bc[o*3], d1 = B8[(oc*27+o)*3+1] - Bc[o*3+1], d2 = B8[(oc*27+o)*3+2] - Bc[o*3+2];
    g[0] = F[0]*d0 + F[3]*d1 + F[6]*d2; g[1] = F[1]*d0 + F[4]*d1 + F[7]*d2; g[2] = F[2]*d0 + F[5]*d1 + F[8]*d2;
}
__global__ void __launch_bounds__(128) hg_force(const int* IP, const int* cb, const float* Fq, const float* Vq, const float* B8, const float* Bc, float kg,
                                                const float* y, double* f, float* B, double* E) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12] && !IP[13]; int qq = (q < IP[12]) ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float w = on ? kg*Vq[qq] : 0.f;
    float H[9] = {0,0,0,0,0,0,0,0,0};
    if (f || E) for (int o=0;o<27;o++) { float g[3]; hg_g(F, B8, Bc, oc, o, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = on ? y[id*3+a] : 0.f; H[a*3] += ya*g[0]; H[a*3+1] += ya*g[1]; H[a*3+2] += ya*g[2]; } }
    if (E) { double e = 0.0; for (int k=0;k<9;k++) e += (double)H[k]*(double)H[k]; blocksum_add(0.5*(double)w*e, E); }
    if (f) for (int o=0;o<27;o++) { float g[3]; hg_g(F, B8, Bc, oc, o, g);
        for (int a=0;a<3;a++) { double s = seg8d((double)(w*(H[a*3]*g[0] + H[a*3+1]*g[1] + H[a*3+2]*g[2]))); if (on && head) atomicAdd(&f[ids[o]*3+a], s); } }
    if (B) for (int o=0;o<27;o++) { float g[3]; hg_g(F, B8, Bc, oc, o, g);
        float s = seg8f(w*(g[0]*g[0] + g[1]*g[1] + g[2]*g[2])); if (on && head) for (int a=0;a<3;a++) atomicAdd(&B[ids[o]*9+a*4], s); }
}
__global__ void __launch_bounds__(128) hg_mv(const int* IP, const int* cb, const float* Fq, const float* Vq, const float* B8, const float* Bc, float kg, const float* v, float* out) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12] && !IP[13]; int qq = (q < IP[12]) ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float w = on ? kg*Vq[qq] : 0.f;
    float H[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { float g[3]; hg_g(F, B8, Bc, oc, o, g); int id = ids[o];
        for (int a=0;a<3;a++) { float va = on ? v[id*3+a] : 0.f; H[a*3] += va*g[0]; H[a*3+1] += va*g[1]; H[a*3+2] += va*g[2]; } }
    for (int o=0;o<27;o++) { float g[3]; hg_g(F, B8, Bc, oc, o, g);
        for (int a=0;a<3;a++) { float s = seg8f(w*(H[a*3]*g[0] + H[a*3+1]*g[1] + H[a*3+2]*g[2])); if (on && head) atomicAdd(&out[ids[o]*3+a], s); } }
}
// 선탐색 후보 c_i = 2^-i (i0 ≤ i < NC): y_c = ŷ + T(η + c_i δ) -> SC[40+i] += E_hg(y_c)  (rt_elem_lsN 과 같은 후보·같은 T; T 선형이라 H(c) = H0 + c Hd)
__global__ void __launch_bounds__(128) hg_ls(const int* IP, const int* cb, const float* Fq, const float* Vq, const float* B8, const float* Bc, float kg,
                                             const float* yh, const float* eta, const float* dl, const float* sk, const int* pm, float n0, float n1, float n2,
                                             double* SC, int NC, int i0, const double* gate) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12] && !IP[13]; int qq = (q < IP[12]) ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    double w = on ? (double)kg*(double)Vq[qq] : 0.0;
    double H0[9] = {0,0,0,0,0,0,0,0,0}, Hd[9] = {0,0,0,0,0,0,0,0,0};
    if (on) for (int o=0;o<27;o++) { float g[3]; hg_g(F, B8, Bc, oc, o, g); int id = ids[o];
        float t0[3], td[3], e0[3] = {eta[id*3], eta[id*3+1], eta[id*3+2]}, ed[3] = {dl[id*3], dl[id*3+1], dl[id*3+2]};
        Tapply(sk[id], pm[id], n0, n1, n2, e0, t0); Tapply(sk[id], pm[id], n0, n1, n2, ed, td);
        for (int a=0;a<3;a++) { double ya = (double)yh[id*3+a] + (double)t0[a], da = (double)td[a];
            for (int b=0;b<3;b++) { H0[a*3+b] += ya*(double)g[b]; Hd[a*3+b] += da*(double)g[b]; } } }
    double c = 1.0; for (int i=0;i<i0;i++) c *= 0.5;
    for (int i=i0;i<NC;i++) {
        double e = 0.0; for (int k=0;k<9;k++) { double h = H0[k] + c*Hd[k]; e += h*h; }
        blocksum_add(0.5*w*e, &SC[40+i]);
        c *= 0.5;
    }
}
'''

# 진단 모듈 (따로 컴파일, 그래프 밖): 모래시계 에너지 비
HM_SRC = r'''
extern "C" {
__device__ __forceinline__ double hm_seg8(double s) {
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2); s += __shfl_xor_sync(0xffffffffu, s, 4); return s;
}
__device__ __forceinline__ double hm_psi(const double* G, double r) {
    double t = G[0] + G[4] + G[8], s = 0.0;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { double e = 0.5*(G[a*3+b] + G[b*3+a]); s += e*e; }
    return s + r*t*t;
}
// 점유 칸마다 8 레인 (칸 c = t >> 3, 점 g = t & 7): G_g = Σ_n U_n ⊗ b_g,n (월드). out[0] += Σ ψ(G_g − Ḡ), out[1] += Σ ψ(G_g), out[2] += 칸 수
__global__ void hm_cell(const int* IP, const int* cb, const float* U, const float* B8, double r, double* out) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int nc = IP[11];
    if (blockIdx.x*blockDim.x >= nc*8) return;
    bool on = (t < nc*8) && !IP[13];
    int c = on ? (t >> 3) : 0, g = t & 7;
    int nb1 = IP[7], nb2 = IP[8];
    double G[9] = {0,0,0,0,0,0,0,0,0};
    if (on) for (int o=0;o<27;o++) { int i = o/9, j = (o/3)%3, k = o%3; int id = ((cb[c*3]+i)*nb1 + (cb[c*3+1]+j))*nb2 + (cb[c*3+2]+k);
        const float* b = B8 + (g*27+o)*3;
        for (int a=0;a<3;a++) { double ua = (double)U[id*3+a]; G[a*3] += ua*b[0]; G[a*3+1] += ua*b[1]; G[a*3+2] += ua*b[2]; } }
    double D[9]; for (int k=0;k<9;k++) D[k] = G[k] - 0.125*hm_seg8(G[k]);
    double eh = on ? hm_psi(D, r) : 0.0, et = on ? hm_psi(G, r) : 0.0;
    if (!(eh == eh) || !(et == et)) { eh = 0.0; et = 0.0; if (on && g == 0) atomicAdd(&out[3], 1.0); }
    atomicAdd(&out[0], eh); atomicAdd(&out[1], et);
    if (on && g == 0) atomicAdd(&out[2], 1.0);
}
}
'''


def _dump():
    try:
        json.dump(INFO, open(os.path.join(OUT[0], "gq1info.json"), "w"), ensure_ascii=False, indent=1, default=str)
    except Exception:
        pass


def _P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _sub(src, old, new, name):
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"[절 CM194 gq1] {name}: '{old[:70]}' 수 {n} (기대 1)")
    return src.replace(old, new)


def _fn_span(src, name):
    i = src.find("void " + name + "(")
    if i < 0:
        return None
    j = src.find("\n__global__", i + 10)
    j2 = src.find("\n__device__", i + 10)
    ends = [q for q in (j, j2) if q > 0]
    return i, (min(ends) if ends else len(src))


def transform(src):
    """해 모듈이면: (1) 1 점 판의 rt_gauss·aq_gauss_s 고침, (2) 벌점 판의 커널 덧붙임."""
    did = False
    if CFG["q1"]:
        sp = _fn_span(src, "rt_gauss")
        if sp is not None:
            a, b = sp
            body = src[a:b]
            body = _sub(body, "if (aq[27] > 0.f) {", "if (false) {   // 절 CM194 1 점: 칸 평균 F", "rt_gauss")
            body = _sub(body, "const float GG = 0.28867513459481287f;", "const float GG = 0.f;   // 절 CM194 1 점: 칸 가운데", "rt_gauss")
            if "AF_GOCT" in body:
                body = _sub(body, "Vq[q] = (1.f - AF_GOCT)*aq[27] + AF_GOCT*Vq[q]; mm = (1.f - AF_GOCT)*AF_MO[q] + AF_GOCT*mm;", "   // 절 CM194 1 점: V = 칸 부피 / 8 (κ 무관)", "rt_gauss")
            src = src[:a] + body + src[b:]
            INFO["gauss_patch"] += 1; did = True
        sp = _fn_span(src, "aq_gauss_s")
        if sp is not None:
            a, b = sp
            body = _sub(src[a:b], "const float* a = (Sa[(size_t)q*4+3] > 0.f) ? Sa + (size_t)q*4 : Sc + (size_t)c*4;",
                        "const float* a = Sc + (size_t)c*4;   // 절 CM194 1 점: 칸 평균 상태", "aq_gauss_s")
            src = src[:a] + body + src[b:]
            INFO["sgauss_patch"] += 1; did = True
    if CFG["hg"] > 0 and 'extern "C" {' in src and "void Tapply(" in src and "void blocksum_add(" in src and "void qnodes(" in src and "hg_force(" not in src:
        j = src.rindex("}")
        src = src[:j] + SRC + "\n" + src[j:]
        did = True
    if did:
        INFO["mods"] += 1
        _dump()
    return src


def tables(Ai, offs):
    """offs = 점 국소 좌표 (f 축마다) 목록 -> BW [n,27], BG [n,27,3] (실행기 RTSolver.__init__ 과 같은 식)."""
    Ai64 = Ai.double().cpu().numpy().reshape(3, 3)
    n = len(offs)
    BW = np.zeros((n, 27)); BG = np.zeros((n, 27, 3))
    for oc, fxv in enumerate(offs):
        w = [[0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2] for f in fxv]
        dw = [[f - 1.5, -2 * (f - 1), f - 0.5] for f in fxv]
        for o in range(27):
            i, j, k = o // 9, (o // 3) % 3, o % 3
            BW[oc, o] = w[0][i] * w[1][j] * w[2][k]
            gy = np.array([dw[0][i] * w[1][j] * w[2][k], w[0][i] * dw[1][j] * w[2][k], w[0][i] * w[1][j] * dw[2][k]])
            BG[oc, o] = gy @ Ai64
    return BW, BG


GG = 0.5 / math.sqrt(3.0)
OFF8 = [[1.0 + (GG if (oc >> (2 - a)) & 1 else -GG) for a in range(3)] for oc in range(8)]


class St:
    def __init__(h, s):
        import torch
        h.s, h.torch = s, torch
        mu, lam = getattr(s, "mu", None), getattr(s, "lam", None)
        prm = getattr(s, "prm", None)
        try:
            mu, lam = float(mu), float(lam)
        except Exception:
            mu, lam = float(prm[0]), float(prm[1])
        h.mu, h.lam = mu, lam
        dev = s.BG.device
        BW8, BG8 = tables(s.Ai, OFF8)
        if s.BG.numel() == BG8.size:
            h.err8 = float(np.abs(BG8.reshape(-1) - s.BG.double().cpu().numpy()).max() / max(np.abs(BG8).max(), 1e-30))
            h.errw = float(np.abs(BW8.reshape(-1) - s.BW.double().cpu().numpy()).max())
        else:
            h.err8 = h.errw = None                                                        # 3x3x3 (32 자리) 판: 진단 (에너지 비) 만
        assert not CFG["q1"] or (h.err8 < 1e-5 and h.errw < 1e-6), f"[절 CM194] 실행기 2x2x2 표와 다름 {h.err8} {h.errw}"
        BWc, BGc = tables(s.Ai, [[1.0, 1.0, 1.0]])
        f32 = torch.float32
        h.BG8 = torch.tensor(BG8, device=dev, dtype=f32).reshape(-1).contiguous()
        h.BW8 = torch.tensor(BW8, device=dev, dtype=f32).reshape(-1).contiguous()
        h.BGc = torch.tensor(BGc, device=dev, dtype=f32).reshape(-1).contiguous()
        h.kg = float(CFG["hg"]) * h.mu
        if CFG["q1"]:
            s.BW8_orig, s.BG8_orig = s.BW, s.BG                                           # 거친 층은 원래 표
            s.BW = torch.tensor(np.repeat(BWc, 8, 0), device=dev, dtype=f32).reshape(-1).contiguous()
            s.BG = torch.tensor(np.repeat(BGc, 8, 0), device=dev, dtype=f32).reshape(-1).contiguous()
        h.rows = []
        h.hm = None
        if CFG["hgm"]:
            import cupy as cp
            h.hm = cp.RawModule(code=HM_SRC, options=("--std=c++14",)).get_function("hm_cell")
            h.hmout = torch.zeros(4, device=dev, dtype=torch.float64)
            nu = h.lam / (2.0 * (h.lam + h.mu))
            h.r = nu / (1.0 - 2.0 * nu)
        INFO["solvers"].append(dict(cls=type(s).__name__, mu=h.mu, lam=h.lam, kg=h.kg, q1=CFG["q1"], hg=CFG["hg"], tab_err8=h.err8, tab_errw=h.errw,
                                    BWc_sum=float(BWc.sum()), BGc_rowsum=float(np.abs(BGc.sum(1)).max())))
        print(f"[절 CM194 gq1] {type(s).__name__}: 1 점 {CFG['q1']} 벌점 γ {CFG['hg']:g} (kg = γμ = {h.kg:.6g}) 에너지 비 {CFG['hgm']} 표 검사 {h.err8}", flush=True)
        _dump()

    def frame(h, t):
        """s.step(t) 한 번 뒤 (그래프 밖): 프레임 노드 변위 U 의 모래시계 에너지 비."""
        if h.hm is None:
            return
        s, torch = h.s, h.torch
        import cupy as cp
        torch.cuda.synchronize()
        ip = s.IP.cpu().numpy().tolist()
        nc = int(ip[11])
        h.hmout.zero_()
        if nc > 0 and int(ip[13]) == 0:
            n = nc * 8
            with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                h.hm(((n + 127) // 128,), (128,), (_P(s.IP), _P(s.cb), _P(s.U), _P(h.BG8), np.float64(h.r), _P(h.hmout)))
        torch.cuda.synchronize()
        o = h.hmout.cpu().numpy().tolist()
        h.rows.append(dict(t=float(t), ehg=o[0], etot=o[1], nc=int(o[2]), nbad=int(o[3]), ip=ip[6:14]))
        with open(os.path.join(OUT[0], "hgm.jsonl"), "a") as f_:
            f_.write(json.dumps(h.rows[-1]) + "\n")


def _hook(s, _ok, name, args):
    h = getattr(s, "_gq1", None)
    if h is None or not (CFG["hg"] > 0):
        return
    if name in UNSUP:
        raise RuntimeError(f"[절 CM194] 벌점 판에서 안 고친 커널 {name}")
    f32, i32 = np.float32, np.int32
    G = (_P(s.IP), _P(s.cb), _P(s.Fq), _P(s.Vq), _P(h.BG8), _P(h.BGc), f32(h.kg))
    capQ = int(s.Fq.shape[0])
    if name in ELEM:
        if name == "rt_elem":
            mode = int(args[8]); y, E, f_, B = args[5], args[9], args[10], args[13]
        elif name == "aq_elem":
            mode = 1 | 8; y, E, f_, B = args[5], None, args[10], args[13]
        else:
            mode = int(args[9]); y, E, f_, B = args[5], None, args[10], args[13]
        if mode & 13:
            _ok(s, "hg_force", capQ, G + (y, f_ if mode & 1 else _P(None), B if mode & 8 else _P(None), E if (E is not None and mode & 4) else _P(None)), 128)
        INFO["hooks"]["%s:%d" % (name, mode)] = INFO["hooks"].get("%s:%d" % (name, mode), 0) + 1
    elif name in MV:
        _ok(s, "hg_mv", capQ, G + (args[6], args[7]), 128)
        INFO["hooks"][name] = INFO["hooks"].get(name, 0) + 1
    elif name == "rt_bd":
        _ok(s, "hg_force", capQ, G + (_P(s.y), _P(None), args[6], _P(None)), 128)
        INFO["hooks"][name] = INFO["hooks"].get(name, 0) + 1
    elif name == "rt_elem_lsN":
        if len(args) >= 19:
            NC, i0, gate = int(args[16]), int(args[17]), args[18]
        else:
            NC, i0, gate = int(args[16]), 0, _P(None)
        _ok(s, "hg_ls", capQ, G + (args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[15], i32(NC), i32(i0), gate), 128)
        INFO["hooks"][name] = INFO["hooks"].get(name, 0) + 1


def selfcheck(h):
    """벌점 커널 대 torch f64 참값 (그래프 밖, 해 상태 안 바꿈): 지금 칸 목록·F_q·V_q 로 흔든 y, v 에서 E·기울기·블록 대각·헤시안 곱·선탐색 후보."""
    s, torch = h.s, h.torch
    dev = s.y.device
    IP = s.IP.cpu().numpy()
    nq, nc = int(IP[12]), int(IP[11])
    nb1, nb2 = int(IP[7]), int(IP[8])
    if nq <= 0 or int(IP[13]):
        return dict(err="빈 상자")
    cap = s.y.numel() // 3
    cb = s.cb[:nc].long()
    off = torch.tensor([[o // 9, (o // 3) % 3, o % 3] for o in range(27)], device=dev)
    ids = ((cb[:, None, 0] + off[None, :, 0]) * nb1 + (cb[:, None, 1] + off[None, :, 1])) * nb2 + (cb[:, None, 2] + off[None, :, 2])     # [nc, 27]
    Fq = s.Fq[:nq].double().view(nc, 8, 3, 3); Vq = s.Vq[:nq].double().view(nc, 8)
    D = (h.BG8.double().view(8, 27, 3) - h.BGc.double().view(1, 27, 3))                     # [8, 27, 3]
    Gm = torch.einsum("cgkj,gok->cgoj", Fq, D)                                                 # g = Fᵀ d  [nc, 8, 27, 3]
    W = h.kg * Vq
    g_ = torch.Generator(device=dev); g_.manual_seed(7)
    nn = int(ids.max()) + 1
    y0 = (torch.randn(cap, 3, generator=g_, device=dev, dtype=torch.float64) * 1e-3).float().double()
    v0 = (torch.randn(cap, 3, generator=g_, device=dev, dtype=torch.float64) * 1e-3).float().double()

    def Eref(y):
        Y = y[ids]                                                                             # [nc, 27, 3]
        H = torch.einsum("coa,cgob->cgab", Y, Gm)
        return 0.5 * (W[:, :, None, None] * H * H).sum()
    yy = y0.clone().requires_grad_(True)
    e = Eref(yy)
    gr, = torch.autograd.grad(e, yy, create_graph=True)
    Hv, = torch.autograd.grad((gr * v0).sum(), yy)
    Bd = torch.zeros(cap, dtype=torch.float64, device=dev).index_add_(0, ids.reshape(-1).repeat(8), (W[:, :, None] * (Gm * Gm).sum(-1)).permute(1, 0, 2).reshape(-1))
    yk = y0.float().reshape(-1).contiguous(); vk = v0.float().reshape(-1).contiguous()
    fk = torch.zeros(cap * 3, dtype=torch.float64, device=dev); Bk = torch.zeros(cap * 9, dtype=torch.float32, device=dev)
    Ek = torch.zeros(1, dtype=torch.float64, device=dev); ok_ = torch.zeros(cap * 3, dtype=torch.float32, device=dev)
    f32, i32 = np.float32, np.int32
    G = (_P(s.IP), _P(s.cb), _P(s.Fq), _P(s.Vq), _P(h.BG8), _P(h.BGc), f32(h.kg))
    capQ = int(s.Fq.shape[0])
    raw = getattr(s, "_gq1_ok", None)
    raw(s, "hg_force", capQ, G + (_P(yk), _P(fk), _P(Bk), _P(Ek)), 128)
    raw(s, "hg_mv", capQ, G + (_P(vk), _P(ok_)), 128)
    NC = 6
    SC = torch.zeros(64, dtype=torch.float64, device=dev)
    sk = torch.ones(cap, dtype=torch.float32, device=dev); pm = torch.zeros(cap, dtype=torch.int32, device=dev)
    eta0 = torch.zeros(cap * 3, dtype=torch.float32, device=dev)
    raw(s, "hg_ls", capQ, G + (_P(yk), _P(eta0), _P(vk), _P(sk), _P(pm), f32(0), f32(0), f32(1), _P(SC), i32(NC), i32(0), _P(None)), 128)
    torch.cuda.synchronize()
    rel = lambda a_, b_: float(torch.linalg.norm((a_ - b_).reshape(-1)) / max(float(torch.linalg.norm(b_.reshape(-1))), 1e-300))
    out = dict(nq=nq, nc=nc, kg=h.kg, E_ref=float(e))
    out["E_rel"] = abs(float(Ek[0]) - float(e)) / max(abs(float(e)), 1e-300)
    out["grad_rel"] = rel(fk.view(-1, 3), gr.detach())
    out["B_rel"] = rel(Bk.view(-1, 3, 3)[:, 0, 0].double(), Bd)
    out["Hv_rel"] = rel(ok_.view(-1, 3).double(), Hv.detach())
    Ec = [float(Eref(y0 + (0.5 ** i) * v0)) for i in range(NC)]
    out["ls_rel"] = max(abs(float(SC[40 + i]) - Ec[i]) / max(abs(Ec[i]), 1e-300) for i in range(NC))
    out["ok"] = bool(max(out["E_rel"], out["grad_rel"], out["B_rel"], out["Hv_rel"], out["ls_rel"]) < 1e-3)
    return out


def install(out, q1=0, hg=0.0, hgm=0, vs=1.0, ck=0):
    import cupy as cp
    import torch
    OUT[0] = out
    os.makedirs(out, exist_ok=True)
    CFG.update(q1=int(q1), hg=float(hg), hgm=int(hgm), vs=float(vs), ck=int(ck))
    assert CFG["hg"] == 0 or CFG["q1"], "--gq_hg 는 --gq_q1 1 위에서만"
    INFO["cfg"] = dict(CFG)
    if CFG["vs"] != 1.0:                                                                   # (4) 충격 x s: 자료 cfg 고침 (torch.load 감싸기)
        tl = torch.load

        def load(*a, **kw):
            d = tl(*a, **kw)
            if isinstance(d, dict) and isinstance(d.get("cfg"), dict) and "x" in d and not d.get("_gq_vs"):
                c = dict(d["cfg"]); sc = CFG["vs"]
                c["g"] = [float(q) * sc * sc for q in c["g"]]
                bcs = []
                for b in (c.get("boundary_conditions") or []):
                    b = dict(b)
                    if "velocity" in b:
                        b["velocity"] = [float(q) * sc for q in b["velocity"]]
                    for kk in ("rotation_scale", "translation_scale"):
                        if kk in b:
                            b[kk] = float(b[kk]) * sc
                    bcs.append(b)
                c["boundary_conditions"] = bcs
                d["cfg"] = c; d["_gq_vs"] = True
                INFO["vs_cfg"] = dict(g=c["g"], bc=bcs)
                _dump()
                print(f"[절 CM194 gq1] 충격 x{sc:g}: 중력 {c['g']}, 경계 속도 x{sc:g}", flush=True)
            return d
        torch.load = load
    if CFG["q1"] or CFG["hg"] > 0:
        base = cp.RawModule
        if not getattr(base, "_gq1", False):
            def RawModule(*a, **kw):
                a = list(a)
                code = kw.get("code", a[0] if a else None)
                if isinstance(code, str):
                    code2 = transform(code)
                    if code2 is not code:
                        if "code" in kw:
                            kw["code"] = code2
                        else:
                            a[0] = code2
                return base(*a, **kw)
            RawModule._gq1 = True
            cp.RawModule = RawModule
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from . import fem_an_rt as ART
    for cls in (ART.RTSolver,):
        if getattr(cls.k, "_gq1", False):
            continue
        _ok = cls.k

        def k(s, name, n, args, b=128, _ok=_ok):
            INFO["names"][name] = INFO["names"].get(name, 0) + 1
            if CFG["q1"] and name.startswith(COARSE_K) and getattr(s, "BG8_orig", None) is not None:
                pb, pw = _P(s.BG), _P(s.BW)
                args = tuple((_P(s.BG8_orig) if (isinstance(x, np.uint64) and x == pb) else (_P(s.BW8_orig) if (isinstance(x, np.uint64) and x == pw) else x)) for x in args)
                INFO["coarse_swap"] += 1
            r = _ok(s, name, n, args, b)
            if getattr(s, "_gqin", False) or name.startswith("hg_"):
                return r
            s._gqin = True
            try:
                _hook(s, _ok, name, args)
            except Exception as e_:
                INFO["err"] = f"{name}: {e_}"[:600]; _dump(); raise
            finally:
                s._gqin = False
            return r
        k._gq1 = True
        cls.k = k
    C = ART.RTSolver
    if not getattr(C.__init__, "_gq1", False):
        f0 = C.__init__

        def init(s, *a, _f0=f0, **kw):
            _f0(s, *a, **kw)
            if getattr(s, "_gq1", None) is not None:
                return
            s._gq1 = St(s)
            s._gq1_ok = _ok_base
            st0 = s.step
            nstep = [0]

            def step(t, *a2, **kw2):
                r = st0(t, *a2, **kw2)
                nstep[0] += 1
                h = s._gq1
                try:
                    h.frame(t)
                except Exception as e_:
                    INFO["hm_err"] = repr(e_)[:300]; _dump()
                if nstep[0] == 2:
                    INFO["first_names"] = dict(INFO["names"]); _dump()
                if CFG["ck"] and CFG["hg"] > 0 and not getattr(h, "_ckd", False) and nstep[0] >= 3:
                    h._ckd = True
                    try:
                        o = selfcheck(h)
                    except Exception as e_:
                        o = dict(err=repr(e_)[:600])
                    INFO["check"] = o
                    json.dump(o, open(os.path.join(OUT[0], "gq1ck.json"), "w"), indent=1, default=str)
                    print(f"[절 CM194 gq1] 검사 {'통과' if o.get('ok') else '실패'}: {json.dumps(o, default=str)}", flush=True)
                    _dump()
                if CFG["q1"] and nstep[0] == 1:
                    INFO["bg_is_center"] = bool(torch.equal(s.BG, s._gq1_center_BG)) if hasattr(s, "_gq1_center_BG") else None
                return r
            s.step = step
            if CFG["q1"]:
                s._gq1_center_BG = s.BG
        init._gq1 = True
        C.__init__ = init
    _ok_base = ART.RTSolver.k                                                                # 훅 달린 k (hg_ 이름은 훅을 건너뜀)
    print(f"[절 CM194] gq1 켬 -- {json.dumps(CFG)}", flush=True)
    _dump()


def summarize(path):
    """hgm.jsonl -> 같은 t 는 마지막 시도만; 비 = Σ E_hg / Σ E_tot, 프레임 최대."""
    L = [json.loads(l) for l in open(path)]
    by = {}
    for q in L:
        by[round(q["t"], 9)] = q
    R = [by[k] for k in sorted(by)]
    eh = sum(q["ehg"] for q in R); et = sum(q["etot"] for q in R)
    fr = [q["ehg"] / q["etot"] if q["etot"] > 0 else None for q in R]
    frv = [q for q in fr if q is not None]
    return dict(n=len(R), ratio=(eh / et if et > 0 else None), frame_max=(max(frv) if frv else None), frame_mean=(float(np.mean(frv)) if frv else None),
                nbad=sum(q.get("nbad", 0) for q in R), frames=fr)
