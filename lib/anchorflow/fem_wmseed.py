"""절 CM60 (사용자 11-08 위임, 정의 변경, 수박 옛 재질 실행기 exe/fem_ao_rt_sim.py + --seedb 만): GF 씨 영역 (β 3e9) 을 입자 수준으로 들고 감.
감싸개 exe/wmseed1_wrap.py --sd1 1 이 install() 을 부를 때만 -- 안 부르면 소스·커널 열 글자 그대로.

원본 (GF gs_simulation_watermelon.py + mpm_utils.py NonAssociativeCamClay_return_mapping, doc 절 CM60.1):
  씨 입자 (입자 β 3e9) 는 첫 '항복면 + 경화' 호출에서 이차식 판별식 B² − 4AC 가 float32 를 넘쳐 logJp = −inf 가 되고, 그 뒤 p0 = ∞ 라
  세 갈래 모두 거짓 -- 영원히 탄성 (σ ≥ 0 깎기만). 그 전에도 인장 꼭짓점 (p < −β p0) 은 사실상 없음.
절 CM59 NS (--seedb 만) 가 터진 원인 (확인): 가우스 점 상태 = 팔분 입자 logJp 의 산술 평균이라 −inf 하나가 점 전체를 −inf 로 만들고,
  프레임 끝 보간 교체 (fpi_p) 가 그 −inf 를 이웃 비씨 입자에 씀 -> 다음 프레임 더 많은 점이 −inf ... (NS_1 비씨 입자 −inf f4 2 % -> f10 43 % -> f20 97 % -> f30 100 %).

정의 (이 파일):
  입자: 씨 입자 = 입자 β (ysp, --seedb) ≥ (β + β_씨)/2 (원본 판정 그대로의 마스크). 씨 입자의 F·logJp 는 입자 수준에서만 바뀜 --
        프레임 끝 G2P (aq_g2p, 기존 커널) 의 걸음별 증분 F ← RM(A_k A_{k−1}⁻¹ F) 를 입자 β 3e9·자기 logJp 로 (원본 함수 그대로 -> −inf 출구도 그대로),
        보간 교체 (fpi_p) 는 씨 입자를 건너뜀 (F·logJp 를 덮지 않음).
  가우스 점 (프레임 시작에 입자에서, 뉴턴 반복 밖):
    소성 몫 = 비씨 입자만: 상태 Sq (aq_p2g_s 에서 씨 입자 뺌), 씨·비씨가 섞인 팔분 점의 F_q = 비씨 입자만의 회전·로그 평균
              (rt_p2g 누적 − 씨 입자 누적, rt_gauss 와 같은 규칙 (--fq_rl)), β 1.
    씨 몫   φ_s = 씨 입자 부피 / 팔분 입자 부피, F_s = 씨 입자만의 회전·로그 평균 (같은 규칙),
              얼은 몫 f = 씨 입자 중 logJp = −inf 부피 몫, 안 언 씨 logJp = 유한 logJp 부피 가중 평균 (−inf 는 평균에 안 섞음).
    응력 (뉴턴 반복마다, aq_elem): Q = (1 − φ_s) V_q P(RM_β(X; Sq)) F_qᵀ + φ_s V_q [(1 − f) P(RM_βs(X_s; logJp_안언씨)) + f P(RM_βs(X_s; −inf))] F_sᵀ,
              X = (I + ∇u) F_q, X_s = (I + ∇u) F_s (같은 노드 변위), RM = 원본 pl_rm (지역 사본, 상태 안 씀), −inf 몫 = 원본 함수에 −inf 를 넣은 것 (= σ ≥ 0 깎기만).
    에너지 선탐색 (en_phi_q, --en_mo) 도 같은 몫으로: Ψ = (1 − φ_s) Ψ_SNH(RM_β(X)) + φ_s [(1 − f) Ψ_SNH(RM_βs(X_s)) + f Ψ_SNH(RM_βs(X_s; −inf))].
    접선 (헤시안) 은 그대로 (F_q 의 시험 σ 탄성 접선 x V_q) -- 고정점 (힘) 이 아닌 뉴턴 수렴에만 영향.
    커밋 (aq_commit) 은 소성 몫만 (지금처럼); 씨 몫 상태는 가우스 점에 안 남음 (입자가 들고 감).
  보간 교체 (fpi_p, 비씨 입자): 칸 안 8 점 중 비씨 입자가 든 팔분 점만으로 다시 정규화 (씨만 든 점의 F_q·상태를 비씨 입자에 옮기지 않음;
        자기 팔분 점은 늘 비씨 입자가 있어 가중 > 0).
  새 문턱·클램프 없음 (씨 판정은 입자 β 두 값의 가운데 = 절 CM28 (a) SEED_NEW 와 같은 식).
전제: --fp_interp (fpi_p), --gq_oct 0 (빈 팔분 점 V_q = 0 -> 칸 대체 없음), --seedb, 비결정 판 (aq_p2g_s; 결정 판 aq_p2g_sd 는 안 고침), fpi_p 는 절 CM30 (a) 빈 점 빼기 판.
"""
import numpy as np
import torch

GLOB = "\n__device__ float* AF_SD;                                                     // 절 CM60: 가우스 점 씨 몫 [q*16]: F_s (0..8), φ_s (9), 얼은 몫 f (10), 안 언 씨 logJp (11), 비씨 입자 있음 (12)\n__device__ float AF_SDT;                                                      // 절 CM60: 씨 판정 = 입자 β ≥ (β + β_씨)/2\n"

DEV = r"""
// ---- 절 CM60 (lib/anchorflow/fem_wmseed.py): 가우스 점 씨 몫 응력·에너지 ----
__device__ __forceinline__ void sd_X(const float* d, const float* A, float* Fs, float* X) {
    for (int k=0;k<9;k++) Fs[k] = d[k];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) X[a*3+b] = Fs[a*3+b] + A[a*3]*Fs[b] + A[a*3+1]*Fs[3+b] + A[a*3+2]*Fs[6+b];
}
// Q = (1 − φ_s) (Pm F_qᵀ) + φ_s V_q Σ_갈래 w (P_s F_sᵀ); Pm = V_q 를 곱한 소성 몫 P. φ_s = 0 이면 Q = Pm F_qᵀ (원래 식 그대로)
__device__ __noinline__ void sd_q(int q, const float* A, int mat, const float* prm, float dt, float al, float vq, const float* Pm, const float* F, float* Q) {
    for (int a=0;a<3;a++) for (int c=0;c<3;c++) Q[a*3+c] = Pm[a*3]*F[c*3] + Pm[a*3+1]*F[c*3+1] + Pm[a*3+2]*F[c*3+2];
    const float* d = AF_SD + (size_t)q*16;
    float ph = d[9];
    if (!(ph > 0.f)) return;
    float Fs[9], X[9]; sd_X(d, A, Fs, X);
    float U[9], S[3], V[9]; ssvdf(X, U, S, V);
    float sev = (mat == 7) ? prm[14] : 0.f;
    float ss[3]; for (int i=0;i<3;i++) ss[i] = (sev > 0.f) ? fmaxf(S[i], sev) : ((fabsf(S[i]) > 1e-6f) ? S[i] : ((S[i] < 0.f) ? -1e-6f : 1e-6f));
    float pp[3] = {0.f, 0.f, 0.f}, fz = d[10];
    for (int br=0; br<2; br++) {
        float wb = br ? fz : 1.f - fz;
        if (!(wb > 0.f)) continue;
        float se[3]; for (int i=0;i<3;i++) se[i] = (sev > 0.f) ? fmaxf(S[i], sev) : S[i];
        float ys_ = prm[11], lj_ = br ? __int_as_float(0xff800000) : d[11], al_ = al;
        if (!(prm[15] > 0.5f)) pl_rm(mat, se, prm, ys_, lj_, al_, dt);
        float Pe[3]; pl_pi(mat, se, prm, al, Pe);
        for (int i=0;i<3;i++) pp[i] += wb*Pe[i]*se[i]/ss[i];
    }
    float Ps[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Ps[a*3+b] = U[a*3]*pp[0]*V[b*3] + U[a*3+1]*pp[1]*V[b*3+1] + U[a*3+2]*pp[2]*V[b*3+2];
    float wp = 1.f - ph;                                                      // 씨만 든 점 (φ_s = 1) 은 소성 몫을 아예 안 더함 (그 점 Sq 는 칸 대체·0)
    for (int a=0;a<3;a++) for (int c=0;c<3;c++) Q[a*3+c] = ((wp > 0.f) ? wp*Q[a*3+c] : 0.f) + ph*vq*(Ps[a*3]*Fs[c*3] + Ps[a*3+1]*Fs[c*3+1] + Ps[a*3+2]*Fs[c*3+2]);
}
// 에너지 몫 (en_phi_q 와 같은 SNH 식): (1 − φ_s) ps + φ_s Σ_갈래 w Ψ_SNH(RM_βs(X_s))
__device__ __noinline__ double sd_psi(int q, const float* A, int mat, const float* prm, float dt, float al, double ps) {
    const float* d = AF_SD + (size_t)q*16;
    float ph = d[9];
    if (!(ph > 0.f)) return ps;
    float Fs[9], X[9]; sd_X(d, A, Fs, X);
    float U[9], S[3], V[9]; ssvdf(X, U, S, V);
    float sev = (mat == 7) ? prm[14] : 0.f, fz = d[10];
    double mu = prm[0], ls = (double)prm[1] + mu, es = 0.0;
    for (int br=0; br<2; br++) {
        float wb = br ? fz : 1.f - fz;
        if (!(wb > 0.f)) continue;
        float se[3]; for (int i=0;i<3;i++) se[i] = (sev > 0.f) ? fmaxf(S[i], sev) : S[i];
        float ys_ = prm[11], lj_ = br ? __int_as_float(0xff800000) : d[11], al_ = al;
        if (!(prm[15] > 0.5f)) pl_rm(mat, se, prm, ys_, lj_, al_, dt);
        double J = (double)se[0]*se[1]*se[2], ic = (double)se[0]*se[0] + (double)se[1]*se[1] + (double)se[2]*se[2];
        es += (double)wb*(0.5*mu*(ic - 3.0) - mu*(J - 1.0) + 0.5*ls*(J - 1.0)*(J - 1.0));
    }
    double wp = 1.0 - (double)ph;
    return ((wp > 0.0) ? wp*ps : 0.0) + (double)ph*es;
}
"""

KERN = r"""
// ---- 절 CM60 (lib/anchorflow/fem_wmseed.py): 씨 입자 누적·가우스 점 씨 몫 ----
__global__ void sd_zero(int n, float* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = 0.f; }
// 씨 입자만: 팔분 점 누적 DA[q*32] = R (0..8), log S (9..17), F (18..26), 부피 (27), 뒤집힌 입자 수 (28), −inf 부피 (29), 유한 logJp 부피 x logJp (30), 유한 logJp 부피 (31)
__global__ void sd_p2g(int N, const int* IP, const float* fx, const int* pcd, const int* cidx, const float* vol, const float* F, const float* ys, const float* lj, float* DA) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    if (!(ys[p] >= AF_SDT)) return;
    int c = cidx[pcd[p]];
    if (c < 0) return;
    int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
    int q = c*8 + oc; float w = vol[p];
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    float U[9], S[3], V[9]; ssvdf(Fp, U, S, V);
    bool ng = !(S[2] > 0.f);
    float R[9], L[9], l[3];
    for (int i=0;i<3;i++) l[i] = ng ? 0.f : logf(S[i]);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
        R[a*3+b] = U[a*3]*V[b*3] + U[a*3+1]*V[b*3+1] + U[a*3+2]*V[b*3+2];
        L[a*3+b] = V[a*3]*l[0]*V[b*3] + V[a*3+1]*l[1]*V[b*3+1] + V[a*3+2]*l[2]*V[b*3+2];
    }
    float* d = DA + (size_t)q*32;
    for (int k=0;k<9;k++) { atomicAdd(&d[k], w*R[k]); atomicAdd(&d[9+k], w*L[k]); atomicAdd(&d[18+k], w*Fp[k]); }
    atomicAdd(&d[27], w);
    if (ng) atomicAdd(&d[28], 1.f);
    float lv = lj[p];
    if (lv == __int_as_float(0xff800000)) atomicAdd(&d[29], w);
    else if (isfinite(lv)) { atomicAdd(&d[30], w*lv); atomicAdd(&d[31], w); }
}
__device__ __forceinline__ float sd_det9(const float* M) { return M[0]*(M[4]*M[8]-M[5]*M[7]) - M[1]*(M[3]*M[8]-M[5]*M[6]) + M[2]*(M[3]*M[7]-M[4]*M[6]); }
// rt_gauss 와 같은 규칙: 회전·로그 평균, rs_fin 실패면 유클리드, 뒤집힌 입자가 있으면 (rl 1 = --fq_rl) 유클리드 평균 det > 0 일 때만 유클리드 (rl 0 = 늘 유클리드)
__device__ void sd_fin(const float* R, const float* L, const float* Fx, float W, bool anyb, int rl, float* out) {
    bool bad = false;
    rs_fin(R, L, W, out, &bad);
    float eu[9]; for (int k=0;k<9;k++) eu[k] = Fx[k]/W;
    if (bad || (anyb && (!rl || sd_det9(eu) > 0.f))) for (int k=0;k<9;k++) out[k] = eu[k];
}
// 가우스 점: 씨 몫 SD[q*16] + 섞인 팔분 점의 소성 몫 F_q = 비씨 입자만 (rt_p2g 누적 A − 씨 누적), 비씨 부피 = Sa[q*4+3] (aq_p2g_s, 씨 뺀 판)
__global__ void sd_gauss(const int* IP, const float* Aacc, const int* badq, const float* Sa, const float* DA, int rl, float* Fq, float* SD) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    const float* d = DA + (size_t)q*32; float* o = SD + (size_t)q*16;
    float Ws = d[27], Wn = Sa[(size_t)q*4+3];
    o[12] = (Wn > 0.f) ? 1.f : 0.f;
    if (!(Ws > 0.f)) { for (int k=0;k<9;k++) o[k] = Fq[(size_t)q*9+k]; o[9] = 0.f; o[10] = 0.f; o[11] = 0.f; return; }
    o[9] = Ws/(Ws + fmaxf(Wn, 0.f));
    float wt = d[29] + d[31];
    o[10] = (wt > 0.f) ? d[29]/wt : 1.f;
    o[11] = (d[31] > 0.f) ? d[30]/d[31] : 0.f;
    float Fs[9]; sd_fin(d, d + 9, d + 18, Ws, d[28] > 0.f, rl, Fs);
    for (int k=0;k<9;k++) o[k] = Fs[k];
    if (Wn > 0.f) {
        const float* a = Aacc + (size_t)q*28;
        float nR[9], nL[9], nF[9];
        for (int k=0;k<9;k++) { nR[k] = a[k] - d[k]; nL[k] = a[9+k] - d[9+k]; nF[k] = a[18+k] - d[18+k]; }
        float out[9]; sd_fin(nR, nL, nF, Wn, badq[q] != 0, rl, out);
        bool ok = true; for (int k=0;k<9;k++) if (!isfinite(out[k])) ok = false;
        if (ok) for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = out[k];
    }
}
"""

ELEM_HEAD = "__global__ void __launch_bounds__(128) aq_elem("
ELEM_OLD = ("    for (int k=0;k<9;k++) Pm[k] *= vq;\n"
            "    for (int a=0;a<3;a++) for (int c=0;c<3;c++) sQ[tl][a*3+c] = Pm[a*3]*F[c*3] + Pm[a*3+1]*F[c*3+1] + Pm[a*3+2]*F[c*3+2];\n")
ELEM_NEW = ("    for (int k=0;k<9;k++) Pm[k] *= vq;\n"
            "    { float Q_[9]; sd_q(qq, A, mat, prm, dt, Sq[(size_t)qq*4+2], vq, Pm, F, Q_);       // 절 CM60: 씨 몫 (φ_s = 0 이면 원래 식)\n"
            "      for (int a=0;a<3;a++) for (int c=0;c<3;c++) sQ[tl][a*3+c] = Q_[a*3+c]; }\n")
P2G_HEAD = "__global__ void aq_p2g_s("
P2G_OLD = "    int q = c*8 + oc; float w = vol[p];\n"
P2G_NEW = "    int q = c*8 + oc; float w = vol[p];\n    if (ys[p] >= AF_SDT) return;                                             // 절 CM60: 씨 입자는 소성 몫 상태에서 뺌 (sd_p2g 가 따로)\n"
EN_OLD = "            e += (ps == ps) ? (double)vq*ps : 1e30;\n"
EN_NEW = "            ps = sd_psi(q, A, mat, prm, dt, Sq[(size_t)q*4+2], ps);                 // 절 CM60: 씨 몫\n            e += (ps == ps) ? (double)vq*ps : 1e30;\n"
FPI_HEAD = "__global__ void fpi_p("
FPI_C_OLD = "    if (c < 0) return;\n"
FPI_C_NEW = "    if (c < 0) return;\n    if (ysp[p] >= AF_SDT) return;                                          // 절 CM60: 씨 입자는 교체 안 함 (F·logJp = G2P 입자 RM 그대로)\n"
FPI_W_OLD = "        if (!(w > 0.f) || !(AF_BX[(size_t)(c*8 + oc)*4+3] > 0.f)) continue;\n"
FPI_W_NEW = "        if (!(w > 0.f) || !(AF_BX[(size_t)(c*8 + oc)*4+3] > 0.f) || !(AF_SD[(size_t)(c*8 + oc)*16+12] > 0.f)) continue;   // 절 CM60: 비씨 입자 든 점만\n"
SEED_MARK = "// 절 CM28 (a) 씨 영역 β"


def _span(s, head):
    assert s.count(head) == 1, ("절 CM60 커널 머리", head, s.count(head))
    i = s.index(head)
    j = s.find("__global__", i + len(head))
    return i, (len(s) if j < 0 else j)


def _rep_in(s, head, o, n):
    i, j = _span(s, head)
    b = s[i:j]
    assert b.count(o) == 1, ("절 CM60 바꿀 글자", head, o[:100], b.count(o))
    return s[:i] + b.replace(o, n) + s[j:]


def patch(r):
    assert SEED_MARK in r, "절 CM60: --seedb (절 CM28 (a) β 판정) 위에서만"
    assert "AF_SD;" not in r
    r = r.replace("extern \"C\" {", "extern \"C\" {" + GLOB, 1)
    r = _rep_in(r, P2G_HEAD, P2G_OLD, P2G_NEW)
    i, _ = _span(r, ELEM_HEAD)
    r = r[:i] + DEV + r[i:]
    r = _rep_in(r, ELEM_HEAD, ELEM_OLD, ELEM_NEW)
    if "en_phi_q(" in r:
        r = _rep_in(r, "__global__ void __launch_bounds__(128) en_phi_q(", EN_OLD, EN_NEW)
    r = _rep_in(r, FPI_HEAD, FPI_C_OLD, FPI_C_NEW)
    r = _rep_in(r, FPI_HEAD, FPI_W_OLD, FPI_W_NEW)
    r = r.rstrip()
    k = r.rfind("}")
    return r[:k] + KERN + "\n}\n"


ST = {"on": False, "done": False, "said": False}


def install():
    from . import fem_ao_rt as AORT
    if getattr(AORT, "_wmseed", False):
        return
    AORT._wmseed = True
    ART = AORT.ART
    o_src, o_skip = AORT._src, AORT._skipify

    def _skip(src):
        if ST["on"]:
            src = patch(src)
            ST["done"] = True
        return o_skip(src)
    AORT._skipify = _skip

    def _src(*a, **kw):
        wf = kw.get("wf") or {}
        if not wf.get("seedb"):
            return o_src(*a, **kw)
        ST["on"], ST["done"] = True, False
        try:
            r = o_src(*a, **kw)
        finally:
            ST["on"] = False
        if not ST["done"]:
            r = patch(r)
        return r
    AORT._src = _src

    def bind(s):
        if not hasattr(s, "mod"):
            return
        pv = s.prm.detach().float().cpu().numpy() if torch.is_tensor(s.prm) else np.asarray(s.prm, dtype=np.float32)
        t = 0.5 * (float(pv[5]) + float(pv[11]))
        g = s.mod.get_global("AF_SDT")
        ART.cp.ndarray((1,), dtype=ART.cp.float32, memptr=g)[0] = np.float32(t)
        if hasattr(s, "sdDA"):
            g = s.mod.get_global("AF_SD")
            ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=g)[0] = int(s.sdSD.data_ptr())
        if not ST["said"]:
            ST["said"] = True
            print(f"[절 CM60] 씨 입자 몫 켬: 씨 판정 입자 β ≥ {t:.6g} (β {float(pv[5]):g}, β_씨 {float(pv[11]):g}), fq_rl {int(bool(getattr(s, 'fqrl', False)))}", flush=True)

    o_alloc = AORT.AORTSolver.alloc

    def alloc(s, need):
        o_alloc(s, need)
        nq = int(s.capQc) * 8
        s.sdDA = torch.zeros(nq * 32, device=s.dev, dtype=torch.float32)
        s.sdSD = torch.zeros(nq * 16, device=s.dev, dtype=torch.float32)
        if getattr(s, "ao_ready", False):
            bind(s)
    AORT.AORTSolver.alloc = alloc
    o_bind = AORT.AORTSolver.bind_act

    def bind_act(s):
        o_bind(s)
        bind(s)
    AORT.AORTSolver.bind_act = bind_act
    ok = AORT.AORTSolver.k
    P = AORT.P
    i32 = np.int32

    def k(s, name, n, args, b=128):
        if name == "aq_p2g_sd":
            raise RuntimeError("절 CM60: 결정적 판 (aq_p2g_sd) 은 안 고침")
        if name == "aq_p2g_s":
            nz = int(s.sdDA.numel())
            ok(s, "sd_zero", nz, (i32(nz), P(s.sdDA)), 256)
            r = ok(s, name, n, args, b)
            ok(s, "sd_p2g", int(s.N), (i32(int(s.N)), P(s.IP), P(s.fx), P(s.pcd), P(s.cidx), P(s.vol), P(s.F), P(s.ysp), P(s.ljp), P(s.sdDA)), 128)
            return r
        r = ok(s, name, n, args, b)
        if name == "aq_gauss_s":
            ok(s, "sd_gauss", int(s.capQc) * 8, (P(s.IP), P(s.Aacc), P(s.badq), P(s.Sa), P(s.sdDA), i32(1 if getattr(s, "fqrl", False) else 0), P(s.Fq), P(s.sdSD)), 128)
        return r
    AORT.AORTSolver.k = k
