"""절 CM20 --fp_interp (사용자 승인 11-07 정의 변경): 입자 변형 구배 F_p 를 G2P 누적 F_p ← A(x_p) F_p 대신 걸음 (프레임) 끝 가우스 점 변형 구배의 보간으로 '교체'.

정의 (doc/fem_realtime.md 절 CM20):
- 원천 = 풀이가 실제로 들고 있는 가우스 점 상태: F_q^end = A(x_q) F_q = F_q + Σ_n U_n ⊗ (F_qᵀ ∇w_n(x_q)) (U = 프레임 끝 노드 변위, rt_elem·rt_qjd 와 같은 식).
  재질 판 (수박) 은 마지막 걸음도 가우스 점에서 커밋 (aq_commit: RM, 상태 Sq 갱신) 한 뒤의 F_q 로 같은 식 -> RM 뒤 탄성 F.
- 보간 = 입자가 속한 B-스플라인 칸 (프레임 시작 위치 기준, P2G 와 같은 칸) 의 2x2x2 가우스 점 삼선형 가중, 칸 좌표 fx ∈ [½, 3/2), 점 1 ± g (g = 1/(2√3)):
  t_a = clamp((fx_a − (1 − g)) / (2g), 0, 1), w_o = Π_a (o 의 a 비트 ? t_a : 1 − t_a) -- 가중 ≥ 0, 합 1 (밖으로 외삽 안 함; 팔분 바깥쪽 입자는 자기 팔분 점 값 그대로).
- 평균 규칙 = P2G 의 F_q 구성 (rt_gauss) 과 같은 회전·신장 분리: 점마다 부호 있는 SVD X = U S Vᵀ, R = U Vᵀ, L = V log S Vᵀ (S_min ≤ 0 = 뒤집힘이면 L = 0, 표시),
  F_p = polar(Σ w R) · exp(sym Σ w L) (rs_fin 그대로), 대체 = Σ w X (유클리드):
    rl = 0 (새 실행기 rt_gauss): rs_fin 실패 또는 가중 > 0 인 점 중 뒤집힘이 있으면 유클리드.
    rl = 1 (옛 재질 실행기 --fq_rl): rs_fin 실패면 유클리드, 뒤집힘만 있으면 유클리드 평균의 det > 0 일 때만 유클리드 (아니면 회전·로그 평균).
- 재질 상태 (ys, logJp, al) 도 같은 가중의 선형 평균으로 교체 (Sq != NULL 일 때).
끄면 이 소스는 어디에도 들어가지 않는다 (생성 CUDA 소스 글자 그대로)."""

FPI_SRC = r'''
// ---- 절 CM20 --fp_interp: 입자 F = 프레임 끝 가우스 점 변형 구배의 칸 안 삼선형 보간 (회전·로그 신장 분리 평균) ----
// 점: Fe[q*28] = R (0..8), L (9..17), X (18..26), 뒤집힘 (27)
__global__ void fpi_q(const int* IP, const int* cb, const float* BG, const float* Fq, const float* U, float* Fe) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12] || IP[13]) return;
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9], X[9]; for (int k=0;k<9;k++) { F[k] = Fq[(size_t)q*9+k]; X[k] = F[k]; }
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ua = U[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; } }
    float Uu[9], S[3], V[9];
    ssvdf(X, Uu, S, V);
    bool ng = !(S[2] > 0.f);
    float* e = Fe + (size_t)q*28;
    float l[3]; for (int i=0;i<3;i++) l[i] = ng ? 0.f : logf(S[i]);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
        e[a*3+b] = Uu[a*3]*V[b*3] + Uu[a*3+1]*V[b*3+1] + Uu[a*3+2]*V[b*3+2];
        e[9+a*3+b] = V[a*3]*l[0]*V[b*3] + V[a*3+1]*l[1]*V[b*3+1] + V[a*3+2]*l[2]*V[b*3+2];
        e[18+a*3+b] = X[a*3+b];
    }
    e[27] = ng ? 1.f : 0.f;
}
__device__ __forceinline__ float fpi_det9(const float* M) { return M[0]*(M[4]*M[8]-M[5]*M[7]) - M[1]*(M[3]*M[8]-M[5]*M[6]) + M[2]*(M[3]*M[7]-M[4]*M[6]); }
// 입자: 자기 칸 8 점 삼선형 가중 (밖으로 외삽 안 함), F 교체 (guard = 비유한이면 안 씀), Sq 있으면 재질 상태도 교체
__global__ void fpi_p(int N, const int* IP, const float* fx, const int* pcd, const int* cidx, const float* Fe, int rl, int guard, float* F,
                      const float* Sq, float* ysp, float* ljp, float* alp) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    if (c < 0) return;
    const float GG = 0.28867513459481287f;
    float t[3]; for (int a=0;a<3;a++) t[a] = fminf(fmaxf((fx[p*3+a] - (1.f - GG)) / (2.f*GG), 0.f), 1.f);
    float sR[9], sL[9], sX[9], s3[3] = {0.f, 0.f, 0.f};
    for (int k=0;k<9;k++) { sR[k] = 0.f; sL[k] = 0.f; sX[k] = 0.f; }
    bool anyb = false;
    for (int oc=0; oc<8; oc++) {
        float w = ((oc & 4) ? t[0] : 1.f - t[0]) * ((oc & 2) ? t[1] : 1.f - t[1]) * ((oc & 1) ? t[2] : 1.f - t[2]);
        if (!(w > 0.f)) continue;
        const float* e = Fe + (size_t)(c*8 + oc)*28;
        for (int k=0;k<9;k++) { sR[k] += w*e[k]; sL[k] += w*e[9+k]; sX[k] += w*e[18+k]; }
        if (e[27] > 0.5f) anyb = true;
        if (Sq) for (int k=0;k<3;k++) s3[k] += w*Sq[(size_t)(c*8 + oc)*4+k];
    }
    float out[9]; bool bad = false;
    rs_fin(sR, sL, 1.f, out, &bad);
    if (rl) { if (bad || (anyb && fpi_det9(sX) > 0.f)) for (int k=0;k<9;k++) out[k] = sX[k]; }
    else if (bad || anyb) for (int k=0;k<9;k++) out[k] = sX[k];
    bool ok = true;
    if (guard) for (int k=0;k<9;k++) if (!isfinite(out[k])) ok = false;
    if (ok) for (int k=0;k<9;k++) F[(size_t)p*9+k] = out[k];
    if (Sq) { ysp[p] = s3[0]; ljp[p] = s3[1]; alp[p] = s3[2]; }
}
'''


def fpify(src):
    """extern "C" 블록 끝 (마지막 '}') 앞에 FPI_SRC 를 넣고 AF_FPI 를 정의."""
    s = src.rstrip()
    i = s.rfind("}")
    s = s[:i] + FPI_SRC + "\n}\n"
    return s.replace('extern "C" {', 'extern "C" {\n#define AF_FPI 1', 1)
