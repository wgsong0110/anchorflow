"""절 CM68 묶음 wolfh4: wolf 높게 쌓임 4 -- 셀 2 B-스플라인 요소의 체적 잠김 (가우스 점 소성·응력·J 칸 안 분산 진단) + F-bar (칸 평균 체적 변화).
감싸개 exe/wolfh4_wrap.py 가 실행기 (fem_an_rt_sim.py) 를 띄우는 순간 install() (다른 감싸개가 모두 설치된 뒤 = 가장 바깥). 인자 없으면 이 모듈을 안 부름.

fb (--w4fb 1) F-bar (정의 변경, 요소 운동학만; 재질 식·RM 은 그대로):
    칸 c 의 가우스 점 q (팔분 8 점, q = 8c + o) 마다 걸음 시작 상태 F_q 와 시험 X_q = (I + ∇u) F_q 에서 증분 체적비 j_q = det X_q / det F_q,
    칸 평균 j̄_c = Σ_q W_q j_q / Σ_q W_q (W_q = V_q det F_q = 걸음 시작 부피; 같은 칸 점 중 det > 0 이고 V_q > 0 인 점만),
    X̄_q = (j̄_c / j_q)^{1/3} X_q   (증분 F-bar: f̄ = (j̄/j)^{1/3} f, X̄ = f̄ F_q -- de Souza Neto et al. 1996 IJSS 33 의 F-bar 를 증분 꼴로, Coombs et al. 2018 CMAME 333 MPM 판과 같은 꼴).
    응력·RM·커밋은 X̄ 로 (모래: PG Drucker-Prager 투영 그대로). 힘은 퍼텐셜 꼴 Σ_q V_q P(X̄_q) : δX̄_q 의 변분 (탄성이면 Ψ = Σ V ψ(X̄) 의 기울기) --
    정리하면 점마다 P_eff = [τ̄_q − p_q I + J_q p̂_c I] X_q⁻ᵀ, τ̄ = X̄ 의 키르히호프 응력, p_q = tr τ̄_q / 3, J_q = det X_q, p̂_c = Σ V p / Σ V J (칸 코시 압력)
    = 편차 응력은 점마다, 압력은 칸 평균 (평균 팽창 Nagtegaal–Parks–Rice 1974 / Simo–Taylor–Pister 1985 Q1/P0 와 같은 체적 몫).
    접선 (뉴턴·PCG 곱; 정의가 아니라 풀이): 점마다 헤시안의 체적 몫 K = λ + 2μ/3 을 빼고 (편차 몫 c² 배) 칸마다 순위 1 체적 항 K ΣV (δ ln j̄)² 를 더함
    (곱 커널 fb_mv: g_qi = √(K ΣV) ω_q / σ_qi, ω_q = V_q J_q / Σ V J, P̂_ii += g_qi Σ_칸 g X̂_kk). 노드 대각 (전처리) 은 점 체적 몫을 그대로 둠 (보수적).
    문턱·클램프 없음. 칸에 유효 점이 하나면 그대로 (c = 1). det ≤ 0 (뒤집힌) 점은 로그가 정의되지 않아 칸 평균에서 빠지고 자기 X 그대로.
    지금은 모래 (재질 판 mat 2, --hspec) 경로만 -- 다른 경로 (탄성 rt_elem·plasticine aq_elem·옛 실행기 수박) 는 같은 식을 통과 후보가 생기면 넣음 (그 전엔 assert).
dg (--w4dg 1) 진단 (결과 불변, 커널 시간만): 걸음 끝 (커밋 앞) 가우스 점마다 시험 X (fb 면 X̄) 의 RM 상태·소성 증분·응력·체적을 바닥 높이 칸별 합 (w4dg.json).
"""
import atexit
import json
import os

import numpy as np

ST = dict(fb=0, dg=0, out=".", marks=[], sol=None)
NB = 7           # 높이 칸: [0,1) [1,2) [2,4) [4,8) [8,16) [16,32) [32,∞) cm (바닥 아래는 0 칸)
NF = 16
CAPR = 4096
FIELDS = ["n", "V", "V_cone", "V_apex", "V_dep_dev", "V_dlogJp", "V_p", "V_sdev", "V_varc_lnj", "V_lnj2", "V_varc_lnJ", "V_varc_p", "V_lnj", "V_mc_lnj2", "V_rho", "V_lnc_abs"]

FB_SRC = r'''
// ---- 절 CM68 wolfh4 (lib/anchorflow/fem_wolfh4.py): F-bar (칸 평균 증분 체적) + 가우스 점 소성·응력·체적 진단 ----
__device__ __forceinline__ float fb_s8(float v) { v += __shfl_xor_sync(0xffffffffu, v, 1); v += __shfl_xor_sync(0xffffffffu, v, 2); v += __shfl_xor_sync(0xffffffffu, v, 4); return v; }
__device__ __forceinline__ float fb_det3(const float* M) { return M[0]*(M[4]*M[8]-M[5]*M[7]) - M[1]*(M[3]*M[8]-M[5]*M[6]) + M[2]*(M[3]*M[7]-M[4]*M[6]); }
// 칸 평균 증분 체적비로 배율 c = (j̄/j)^{1/3}; 8 레인 (한 칸) 모두 불러야 함
__device__ __forceinline__ float fb_scale(bool on, float vq, const float* F, const float* X, bool& w, float& JX) {
    JX = fb_det3(X); float JF = fb_det3(F);
    w = on && vq > 0.f && JX > 0.f && JF > 0.f;
    float sW = fb_s8(w ? vq*JF : 0.f), sWj = fb_s8(w ? vq*JX : 0.f);
    if (!(w && sW > 0.f && sWj > 0.f)) { w = false; return 1.f; }
    return cbrtf((sWj/sW)*(JF/JX));
}
__global__ void __launch_bounds__(128) fb_mv(const int* IP, const int* cb, const float* BG, const float* H, const float* dy, float* out, int capQ) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    __shared__ float sY[16][27][3]; __shared__ float sP[128][9];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int tl = threadIdx.x, l8 = tl & 7, b8 = tl & ~7, cl = tl >> 3; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    for (int o = l8; o < 27; o += 8) { int id = ids[o]; for (int a=0;a<3;a++) sY[cl][o][a] = dy[id*3+a]; }
    __syncwarp();
    float A[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; for (int a=0;a<3;a++) { float ya = sY[cl][o][a]; A[a*3] += ya*bg[0]; A[a*3+1] += ya*bg[1]; A[a*3+2] += ya*bg[2]; } }
    float U[9], M[9], g[3];
    for (int k=0;k<9;k++) { U[k] = H[(size_t)k*capQ + qq]; M[k] = H[(size_t)(9+k)*capQ + qq]; }
    for (int i=0;i<3;i++) g[i] = on ? FB_G[(size_t)qq*3+i] : 0.f;
    float T[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) T[a*3+b] = A[a*3]*M[b] + A[a*3+1]*M[3+b] + A[a*3+2]*M[6+b];
    float sl = 0.f; for (int i=0;i<3;i++) sl += g[i]*(U[i]*T[i] + U[3+i]*T[3+i] + U[6+i]*T[6+i]);     // Σ_i g_i X̂_ii, X̂ = Uᵀ A M
    float Sc = fb_s8(sl);
    for (int a=0;a<3;a++) for (int c=0;c<3;c++) sP[tl][a*3+c] = Sc*(U[a*3]*g[0]*M[c*3] + U[a*3+1]*g[1]*M[c*3+1] + U[a*3+2]*g[2]*M[c*3+2]);
    __syncwarp();
    for (int o = l8; o < 27; o += 8) {
        float sv[3] = {0.f, 0.f, 0.f};
        for (int j=0;j<8;j++) { const float* bg = BG + (j*27+o)*3; const float* Q = sP[b8+j];
            for (int a=0;a<3;a++) sv[a] += Q[a*3]*bg[0] + Q[a*3+1]*bg[1] + Q[a*3+2]*bg[2]; }
        if (on) for (int a=0;a<3;a++) if (sv[a] != 0.f) atomicAdd(&out[ids[o]*3+a], sv[a]);
    }
}
// 걸음 끝 커밋 (ao_commit 을 F-bar 로): F_q ← F_q X⁻¹ Y, Y = RM(X̄) (RM 안 걸리면 X̄)
__global__ void __launch_bounds__(128) fb_commit(const int* IP, const int* cb, const float* BG, float* Fq, const float* U, float cdp, double* cnt) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ua = U[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; } }
    bool w; float JX; float c = fb_scale(on, on ? FB_VQ[qq] : 0.f, F, X, w, JX);
    if (!on) return;
    float Xb[9]; for (int k=0;k<9;k++) Xb[k] = c*X[k];
    float Y[9];
    bool rm = ao_rm3(Xb, cdp, Y);
    if (!rm) { if (!w || c == 1.f) return; for (int k=0;k<9;k++) Y[k] = Xb[k]; }
    float Xi[9]; inv3(X, Xi);
    float T1[9], T2[9]; ao_mm3(Xi, Y, T1); ao_mm3(F, T1, T2);
    bool ok = true; for (int k=0;k<9;k++) if (!isfinite(T2[k])) ok = false;
    if (ok) for (int k=0;k<9;k++) Fq[(size_t)qq*9+k] = T2[k];
    if (cnt) { atomicAdd(&cnt[0], 1.0); if (w) { float l = logf(c); atomicAdd(&cnt[1], 1.0); atomicAdd(&cnt[2], (double)fabsf(l)); atomicAdd(&cnt[3], (double)l*l); } if (!ok) atomicAdd(&cnt[4], 1.0); }
}
// 진단: 걸음 끝 시험 X (fb 면 X̄) 의 RM 상태·소성 증분·응력·체적, 바닥 높이 칸별 합 (V 가중). 행 = ctr[0]
__global__ void __launch_bounds__(128) fb_dg(const int* IP, const int* cb, const float* BG, const float* BW, const float* Fq, const float* Vq, const float* U,
                                             float mu, float lam, float cdp, float n0, float n1, float n2, float pn, double* D, const int* ctr, int cap) {
    __shared__ double sh[7*16];
    for (int i=threadIdx.x;i<7*16;i+=blockDim.x) sh[i] = 0.0;
    __syncthreads();
    int row = ctr[0];
    if (blockIdx.x*blockDim.x < IP[12]) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    float xq[3] = {0.f, 0.f, 0.f};
    float bm = on ? AF_BX[qq*4+3] : 0.f;
    if (bm > 0.f) for (int a=0;a<3;a++) xq[a] = AF_BX[qq*4+a]/bm;
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o]; float ww = BW[oc*27+o];
        for (int a=0;a<3;a++) { float ua = U[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; xq[a] += ww*ua; } }
    float vq = on ? Vq[qq] : 0.f;
    bool live = on && bm > 0.f && vq > 0.f;
    bool w; float JX; float c = 1.f;
    if (FB_ON) c = fb_scale(live, vq, F, X, w, JX); else { JX = fb_det3(X); w = live && JX > 0.f && fb_det3(F) > 0.f; }
    float JF = fb_det3(F);
    float Xb[9]; for (int k=0;k<9;k++) Xb[k] = c*X[k];
    float Uu[9], S[3], V[9]; ssvdf(Xb, Uu, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    float K3 = lam + 2.f*mu/3.f;
    float pr = (st == 2) ? 0.f : -K3*tr;                                       // 키르히호프 압력 (압축 +)
    float sd = (st == 2) ? 0.f : 2.f*mu*((st == 1) ? s : 1.f)*rho;               // |dev τ|
    float dep = (st == 1) ? (1.f - s)*rho : ((st == 2) ? rho : 0.f);            // 편차 소성 변형 증분 (Hencky 노름)
    float dlj = (st == 2) ? tr : 0.f;                                            // Δ log Jp (꼭짓점 투영이 버린 체적)
    float lnj = (w ? logf(JX/JF) : 0.f), lnJ = (w ? logf(JX) : 0.f);           // 증분·전체 (X, 막대 전) 체적
    float wv = w ? vq : 0.f;
    float sV = fb_s8(wv), mlj = fb_s8(wv*lnj), mlJ = fb_s8(wv*lnJ), mp = fb_s8(wv*pr);
    if (sV > 0.f) { mlj /= sV; mlJ /= sV; mp /= sV; }
    if (live) {
        float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;
        float dc = d*100.f; int b = (dc < 1.f) ? 0 : ((dc < 2.f) ? 1 : ((dc < 4.f) ? 2 : ((dc < 8.f) ? 3 : ((dc < 16.f) ? 4 : ((dc < 32.f) ? 5 : 6)))));
        double* o = sh + b*16;
        atomicAdd(o+0, 1.0); atomicAdd(o+1, (double)vq);
        if (st == 1) atomicAdd(o+2, (double)vq);
        if (st == 2) atomicAdd(o+3, (double)vq);
        atomicAdd(o+4, (double)vq*dep); atomicAdd(o+5, (double)vq*dlj); atomicAdd(o+6, (double)vq*pr); atomicAdd(o+7, (double)vq*sd);
        if (w) { atomicAdd(o+8, (double)vq*(lnj-mlj)*(lnj-mlj)); atomicAdd(o+9, (double)vq*lnj*lnj); atomicAdd(o+10, (double)vq*(lnJ-mlJ)*(lnJ-mlJ));
                 atomicAdd(o+12, (double)vq*lnj); atomicAdd(o+13, (double)vq*mlj*mlj); atomicAdd(o+15, (double)vq*fabsf(logf(c))); }
        atomicAdd(o+11, (double)vq*(pr-mp)*(pr-mp)); atomicAdd(o+14, (double)vq*rho);
    }
    }
    __syncthreads();
    if (row < cap) for (int i=threadIdx.x;i<7*16;i+=blockDim.x) if (sh[i] != 0.0) atomicAdd(&D[(size_t)row*112 + i], sh[i]);
}
__global__ void fb_inc(int* ctr) { ctr[0] += 1; }
'''

# ao_elem_s 본문 안 바꿈 (모두 'if (FB_ON)' 아래 -- FB_ON 0 이면 원래 식 그대로)
ELEM_SUBS = [
    ("    float U[9], S[3], V[9];\n    ssvdf(X, U, S, V);\n",
     "    float U[9], S[3], V[9];\n    ssvdf(X, U, S, V);\n"
     "    float fbc = 1.f, fbJX = 1.f, fb_b = 0.f; bool fbw = false;\n"
     "    if (FB_ON) { fbc = fb_scale(on, on ? Vq[qq] : 0.f, F, X, fbw, fbJX); if (fbw) for (int i=0;i<3;i++) S[i] *= fbc; }   // 절 CM68 F-bar: X̄ = c X (주값만 c 배)\n"),
    ("    float vq = on ? Vq[qq] : 0.f;\n    // 힘:",
     "    float vq = on ? Vq[qq] : 0.f;\n"
     "    if (FB_ON) {                                                                 // 절 CM68: P_eff = [τ̄ − p I + J p̂_칸 I] X⁻ᵀ, 순위 1 체적 접선 무게\n"
     "        float t_[3]; for (int i=0;i<3;i++) t_[i] = p[i]*ss[i];\n"
     "        float pq = (t_[0] + t_[1] + t_[2])/3.f;\n"
     "        float sP = fb_s8(fbw ? vq*pq : 0.f), sJ = fb_s8(fbw ? vq*fbJX : 0.f), sV = fb_s8(fbw ? vq : 0.f);\n"
     "        if (fbw && sJ > 0.f) { float ph = sP/sJ; for (int i=0;i<3;i++) p[i] = (t_[i] - pq + fbJX*ph)*fbc/ss[i]; fb_b = sqrtf(K3*sV)*vq*fbJX/sJ; }\n"
     "    }\n"
     "    // 힘:"),
    ("D[i][j] = (((i==j) ? 2.f*mu : 0.f) + lam)/(ss[i]*ss[j])",
     "D[i][j] = (((i==j) ? 2.f*mu : 0.f) + lam - ((fb_b > 0.f) ? K3 : 0.f))/(ss[i]*ss[j])"),
    ("    if (on) { for (int k=0;k<9;k++) { H[(size_t)k*capQ + qq] = U[k]; H[(size_t)(9+k)*capQ + qq] = M[k]; } for (int k=0;k<12;k++) H[(size_t)(18+k)*capQ + qq] = C[k]; }\n",
     "    if (FB_ON) { float c2 = fbc*fbc; for (int k=0;k<12;k++) C[k] *= c2;           // 절 CM68: X̄ 접선을 X 로 (c²), 순위 1 무게 g = b c / σ̄\n"
     "        if (on) for (int i=0;i<3;i++) FB_G[(size_t)qq*3+i] = fb_b*fbc/ss[i]; }\n"
     "    if (on) { for (int k=0;k<9;k++) { H[(size_t)k*capQ + qq] = U[k]; H[(size_t)(9+k)*capQ + qq] = M[k]; } for (int k=0;k<12;k++) H[(size_t)(18+k)*capQ + qq] = C[k]; }\n"
     "    if (FB_ON && fb_b > 0.f) { const int RI_[6] = {0,1,2,0,0,1}, RJ_[6] = {0,1,2,1,2,2}; float c2 = fbc*fbc;   // 노드 대각 (전처리) 만: 점 체적 몫 되돌림\n"
     "        for (int k=0;k<6;k++) C[k] += vq*K3*c2/(ss[RI_[k]]*ss[RJ_[k]]); }\n"),
]


def _sub_fn(src, fn, old, new):
    i = src.index("__global__ void __launch_bounds__(128) " + fn + "(")
    j = src.find("__global__", i + 10)
    j = len(src) if j < 0 else j
    body = src[i:j]
    assert body.count(old) == 1, (fn, old[:60], body.count(old))
    return src[:i] + body.replace(old, new) + src[j:]


def patch_src(src):
    if "fb_mv(" in src or "ao_elem_s(" not in src:
        return src
    for old, new in ELEM_SUBS:
        src = _sub_fn(src, "ao_elem_s", old, new)
    decl = "\n__device__ int FB_ON;\n__device__ float* FB_G;\n__device__ const float* FB_VQ;\n"
    i = src.index("__global__ void __launch_bounds__(128) ao_elem_s(")
    i = src.rfind("\n", 0, src.rfind("__device__ __forceinline__ void ao_pairs", 0, i))
    src = src[:i] + decl + FB_SRC.split("__global__ void __launch_bounds__(128) fb_mv(")[0] + src[i:]
    k = src.rstrip().rfind("}")
    rest = "__global__ void __launch_bounds__(128) fb_mv(" + FB_SRC.split("__global__ void __launch_bounds__(128) fb_mv(")[1]
    if "AF_BX" not in src:                                                        # 장벽 없는 판: 진단 커널 빼기
        rest = rest.split("// 진단:")[0] + "__global__ void fb_inc(int* ctr) { ctr[0] += 1; }\n"
    return src[:k] + rest + "\n}\n"


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _bind(s_):
    import torch
    capQ = int(s_.capQc) * 8
    dev = s_.dev
    if getattr(s_, "_w4capQ", -1) != capQ:
        s_._w4capQ = capQ
        s_._w4G = torch.zeros(capQ * 3, device=dev, dtype=torch.float32)
    if getattr(s_, "_w4cnt", None) is None:
        s_._w4cnt = torch.zeros(8, device=dev, dtype=torch.float64)
        s_._w4ctr = torch.zeros(1, device=dev, dtype=torch.int32)
        s_._w4D = torch.zeros(CAPR * NB * NF, device=dev, dtype=torch.float64)
    ST["sol"] = s_
    mod = getattr(s_, "mod", None)
    if mod is None:
        return
    from . import fem_an_rt as ART
    try:
        g = mod.get_global("FB_ON")
    except Exception:                                                             # noqa: BLE001  (재질 모듈이 아닌 다른 모듈)
        return
    ART.cp.ndarray((1,), dtype=ART.cp.int32, memptr=g)[0] = int(ST["fb"])
    for nm, t in (("FB_G", s_._w4G), ("FB_VQ", s_.Vq)):
        ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=mod.get_global(nm))[0] = int(t.data_ptr())


def install(fb=0, dg=0, out="."):
    from . import fem_an_rt as ART
    from . import fem_mat_rt as MRT
    ST.update(fb=int(fb), dg=int(dg), out=out)
    _ob = ART.build_src
    ART.build_src = lambda *a, **kw: patch_src(_ob(*a, **kw))
    i32, f32 = np.int32, np.float32
    _mb = MRT.MatRT.mat_bind

    def mat_bind(s_):
        r = _mb(s_)
        assert not s_.aq and s_.hspec and not getattr(s_, "det", False), "절 CM68 F-bar 는 지금 모래 (mat 2) --hspec 비결정 판만"
        _bind(s_)
        return r
    MRT.MatRT.mat_bind = mat_bind
    _k = ART.RTSolver.k

    def k(s_, name, nn, args, b=128):
        if name == "ao_commit":
            _bind(s_) if getattr(s_, "_w4cnt", None) is None else None
            if ST["dg"] and s_.floor:
                FL = s_.floor; nr = FL["NR"]
                _k(s_, "fb_dg", nn, (args[0], args[1], args[2], P(s_.BW), args[3], P(s_.Vq), args[4], f32(s_.mu), f32(s_.lam), args[5],
                                     *[f32(q) for q in nr], f32(FL["PN"]), P(s_._w4D), P(s_._w4ctr), i32(CAPR)), 128)
                _k(s_, "fb_inc", 1, (P(s_._w4ctr),), 1)
            if ST["fb"]:
                return _k(s_, "fb_commit", nn, tuple(args) + (P(s_._w4cnt),), 128)
            return _k(s_, name, nn, args, b)
        r = _k(s_, name, nn, args, b)
        if ST["fb"] and name == "rt_mv2s" and int(args[8]) == int(s_.capQc) * 8 and int(args[2]) == int(s_.BG.data_ptr()):
            _k(s_, "fb_mv", nn, (args[0], args[1], args[2], args[5], args[6], args[7], args[8]), 128)
        return r
    ART.RTSolver.k = k
    _os = ART.RTSolver.step

    def step(s_, t, *a, **kw):
        d = getattr(s_, "_w4dep", 0)
        s_._w4dep = d + 1
        try:
            r = _os(s_, t, *a, **kw)
        finally:
            s_._w4dep = d
        if d == 0 and ST["dg"]:
            c = getattr(s_, "_w4ctr", None)
            ST["marks"].append((int(round(t / s_.h)), int(c.item()) if c is not None else -1))
        return r
    ART.RTSolver.step = step
    print(f"[절 CM68] wolfh4 켬: F-bar {ST['fb']} · 진단 {ST['dg']}", flush=True)

    def _fin():
        info = dict(fb=ST["fb"], dg=ST["dg"])
        sol = ST.get("sol")
        if sol is not None:
            try:
                import torch
                torch.cuda.synchronize()
                c = sol._w4cnt.cpu().tolist()
                info.update(commit_pts=c[0], fb_pts=c[1], lnc_abs_mean=(c[2] / c[1] if c[1] else None), lnc_rms=((c[3] / c[1]) ** 0.5 if c[1] else None), nonfinite=c[4])
                if ST["dg"]:
                    nr = int(sol._w4ctr.item())
                    D = sol._w4D.view(CAPR, NB, NF)[:min(nr, CAPR)].cpu().numpy()
                    np.savez_compressed(os.path.join(out, "w4dg.npz"), D=D, marks=np.array(ST["marks"]))
                    fr, lo = [], 0
                    for f_, hi in ST["marks"]:
                        hi = min(hi, D.shape[0])
                        if hi > lo:
                            fr.append(dict(f=f_, rows=[lo, hi], last=D[hi - 1].tolist()))
                        lo = hi
                    json.dump(dict(edges_cm=[0, 1, 2, 4, 8, 16, 32], fields=FIELDS, nrows=nr, frames=fr), open(os.path.join(out, "w4dg.json"), "w"))
                    info["dg_rows"] = nr
            except Exception as e:                                                       # noqa: BLE001
                info["err"] = repr(e)
        json.dump(info, open(os.path.join(out, "w4info.json"), "w"))
    atexit.register(_fin)
