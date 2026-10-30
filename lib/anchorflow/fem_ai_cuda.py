"""절 AI (사용자 10-30 합의 원인): 프레임 시작 응력 폭주의 두 원인을 클램프 없이 근본적으로 고친다 -- 확정 경로 (절 AD/AF, --cfuse --cfuse_mode 2) 위.

원인 (1) P2G 에서 셀 안 입자 F 를 유클리드 공간에서 그냥 평균 -> 회전이 섞이면 가짜 압축 (Σ w R_p 의 특이값 < 1).
원인 (2) G2P 에서 셀 안 입자를 모두 같은 A_T 로 갱신 -> 셀 경계에서 갱신이 불연속, 입자 F 이웃 벌어짐 누적.

A. 셀 탄성 모델 (P2G):
  A1 (--ai_p2g rs)  F_p = R_p S_p (극분해, det F_p < 0 이면 가장 작은 특이값 방향을 뒤집어 R_p 를 회전으로 -- 수를 센다),
                    R̄ = polar(Σ w R_p) (회전의 현 (chordal) 평균), S̄ = exp(Σ w log S_p) (로그 유클리드 평균), F̄ = R̄ S̄, w = V_p / V_T.
  A2 (--ai_stress)  셀 기준 응력 = 프레임 시작 입자 Kirchhoff 응력의 부피 가중 평균 τ̄₀ = Σ w τ(F_p).
                    서브스텝 셀 응력 τ̄(A) = τ(A F̄) + R_A (τ̄₀ − τ(F̄)) R_Aᵀ  (R_A = polar(A_T); 프레임 시작 (A = I) 에서 τ̄₀ 와 같고,
                    셀 변형 증분은 셀 F̄ 의 FCR 로, 응력 차 D₀ = τ̄₀ − τ(F̄) 는 셀 회전으로 돌려 나른다 -- 객관적).
                    힘 f_i = −V_T τ̄ A⁻ᵀ ∇λ_i  (= 기존 −V_T P(A F̄) F̄ᵀ ∇λ_i  +  −V_T R_A D₀ R_Aᵀ A⁻ᵀ ∇λ_i).
                    τ(F) = P(F) Fᵀ = 2μ (F − Q) Fᵀ + λ (J − 1) |det F| I  (FCR, Q = F (FᵀF)^{-1/2}, J = Π max(σ, 0.01) -- 힘 커널과 같은 정의).
B. 입자 F 갱신 (G2P, 위치·속도는 barycentric 그대로):
  B1 = 절 AB --fupd navg --fupd_b 1 (기존 코드).
  B2 (--ai_g2p bs)     F ← (I + Σ_n U_n ⊗ ∇w̃_n(x_p)) F,  w = 노드 격자 2 차 B-스플라인 (절 T 와 같은 식), 질량 있는 메시 노드만,
                       Shepard 정규화 w̃ = w / Σw (∇w̃ = (∇w − w̃ ∇Σw)/Σw -> 강체 이동이면 기울기 0). 보간 사상의 해석 기울기 (이웃 최소제곱 아님).
  B3 (--ai_g2p dev)    F ← A_T F̄_T + R_A (F − F̄_T)  (셀 평균 몫은 셀 사상, 셀 평균에서의 편차는 회전만).
  B3+B1 (devnavg)      Ã 로 같은 식, B3+B2 (devbs) A_bs 로 같은 식.
"""
from __future__ import annotations

from . import fem_rt_cuda as RTC
from . import fem_ex_cuda as EX
from . import fem_c_cuda as CCU

_AI = r'''
__device__ float OFFMIN = 0.f;     // A2 응력 차 몫을 건너뛸 문턱 (Pa) -- set_offmin
extern "C" {
// ---- 대칭 3x3 고유분해 (순환 야코비, double) : A (9, 대칭) -> l (대각), V (열 = 고유벡터) ----
template <typename R> __device__ __forceinline__ void ai_rot(R* A, R* V, int p, int q) {
    R apq = A[p*3+q];
    if (fabs(apq) < (R)1e-37) return;
    R th = (A[q*3+q] - A[p*3+p]) / ((R)2*apq);
    R t = (th >= (R)0 ? (R)1 : (R)-1) / (fabs(th) + sqrt(th*th + (R)1));
    R c = (R)1/sqrt(t*t + (R)1), s = t*c;
    for (int k=0;k<3;k++) { R akp=A[k*3+p], akq=A[k*3+q]; A[k*3+p]=c*akp - s*akq; A[k*3+q]=s*akp + c*akq; }
    for (int k=0;k<3;k++) { R apk=A[p*3+k], aqk=A[q*3+k]; A[p*3+k]=c*apk - s*aqk; A[q*3+k]=s*apk + c*aqk; }
    for (int k=0;k<3;k++) { R vkp=V[k*3+p], vkq=V[k*3+q]; V[k*3+p]=c*vkp - s*vkq; V[k*3+q]=s*vkp + c*vkq; }
}
template <typename R> __device__ __forceinline__ void ai_eig(const R* S, R* l, R* V) {
    R A[9]; for (int k=0;k<9;k++) { A[k]=S[k]; V[k]=(k%4==0)?(R)1:(R)0; }
    R sc = fabs(A[0]) + fabs(A[4]) + fabs(A[8]) + (R)1e-30;
    const R tol = sizeof(R) == 8 ? (R)1e-30 : (R)1e-14;
    for (int sw=0; sw<10; sw++) {
        R off = A[1]*A[1] + A[2]*A[2] + A[5]*A[5];
        if (off < tol*sc*sc) break;
        ai_rot(A, V, 0, 1); ai_rot(A, V, 0, 2); ai_rot(A, V, 1, 2);
    }
    l[0]=A[0]; l[1]=A[4]; l[2]=A[8];
}
// V diag(d) Vᵀ
template <typename R> __device__ __forceinline__ void ai_vdv(const R* V, const R* d, R* M) {
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) { R s=(R)0; for (int k=0;k<3;k++) s += V[r*3+k]*d[k]*V[c*3+k]; M[r*3+c]=s; }
}
// F (9) -> 회전 R (det +1; det F < 0 이면 가장 작은 특이값 방향 반전, inv = 1), log S (9 대칭, |σ| 의 로그), Q = F (FᵀF)^{-1/2} (반전 없음), J = Π max(σ,0.01)
template <typename T> __device__ __forceinline__ void ai_polar(const T* F, T* R, T* Lg, T* Q, T& J, int& inv) {
    T C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k] = F[i]*F[k] + F[3+i]*F[3+k] + F[6+i]*F[6+k];
    T l[3], V[9]; ai_eig(C, l, V);
    T s[3]; int imin = 0;
    for (int k=0;k<3;k++) { s[k] = sqrt(fmax(l[k], (T)1e-24)); if (s[k] < s[imin]) imin = k; }
    T dt = det3(F);
    inv = dt < (T)0 ? 1 : 0;
    T di[3], dq[3], dl[3];
    J = (T)1;
    for (int k=0;k<3;k++) { dq[k] = (T)1/s[k]; di[k] = (inv && k == imin) ? -dq[k] : dq[k]; dl[k] = log(fmax(s[k], (T)1e-12)); J *= fmax(s[k], (T)0.01); }
    T Wi[9], Wq[9];
    ai_vdv(V, di, Wi); ai_vdv(V, dq, Wq);
    if (R) mm3(F, Wi, R);
    if (Q) mm3(F, Wq, Q);
    if (Lg) ai_vdv(V, dl, Lg);
}
// τ(F) = 2μ (F − Q) Fᵀ + λ (J − 1) |det F| I  -> 대칭 6 (xx, yy, zz, xy, xz, yz)
template <typename R> __device__ __forceinline__ void ai_tau(const R* F, R mu, R lam, R* t6) {
    R Q[9], J; int inv;
    ai_polar(F, (R*)nullptr, (R*)nullptr, Q, J, inv);
    R D[9]; for (int k=0;k<9;k++) D[k] = F[k] - Q[k];
    R T[9]; mmt3(D, F, T);
    R pr = lam*(J - (R)1)*fabs(det3(F));
    t6[0] = (R)2*mu*T[0] + pr; t6[1] = (R)2*mu*T[4] + pr; t6[2] = (R)2*mu*T[8] + pr;
    t6[3] = mu*(T[1] + T[3]); t6[4] = mu*(T[2] + T[6]); t6[5] = mu*(T[5] + T[7]);
}

// ---- A1: 회전·신장 분리 평균 (P2G, 입자당 스레드; 모든 입자가 사면체 F̄ 에 들어가는 경로 -- --floor vel / none) ----
__global__ void ai_rs_acc(int N, const long long* pinv, const float* vol, const float* Fp, float* RS, float* LS, float* cnt) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    long long t = pinv[p];
    float F[9]; for (int k=0;k<9;k++) F[k] = Fp[p*9+k];                    // 입자 극분해는 float (A6000 FP64 가 느리다 -- 절 AI 속도)
    float R[9], Lg[9], J; int inv;
    ai_polar(F, R, Lg, (float*)nullptr, J, inv);
    float V = vol[p];
    for (int k=0;k<9;k++) atomicAdd(&RS[t*9+k], (float)(V*R[k]));
    atomicAdd(&LS[t*6+0], (float)(V*Lg[0])); atomicAdd(&LS[t*6+1], (float)(V*Lg[4])); atomicAdd(&LS[t*6+2], (float)(V*Lg[8]));
    atomicAdd(&LS[t*6+3], (float)(V*Lg[1])); atomicAdd(&LS[t*6+4], (float)(V*Lg[2])); atomicAdd(&LS[t*6+5], (float)(V*Lg[5]));
    if (inv) atomicAdd(&cnt[0], 1.f);
}
__global__ void ai_rs_fin(const int* szp, const float* VTs, const float* RS, const float* LS, float* FT, float* cnt) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VTs[t];
    if (!(V > 0.f)) return;
    double M[9]; for (int k=0;k<9;k++) M[k] = (double)RS[t*9+k] / (double)V;
    double Rb[9], J; int inv;
    ai_polar(M, Rb, (double*)nullptr, (double*)nullptr, J, inv);
    if (inv) atomicAdd(&cnt[1], 1.f);
    double L[9];
    L[0] = LS[t*6+0]/(double)V; L[4] = LS[t*6+1]/(double)V; L[8] = LS[t*6+2]/(double)V;
    L[1] = L[3] = LS[t*6+3]/(double)V; L[2] = L[6] = LS[t*6+4]/(double)V; L[5] = L[7] = LS[t*6+5]/(double)V;
    double l[3], W[9]; ai_eig(L, l, W);
    double e[3]; for (int k=0;k<3;k++) e[k] = exp(l[k]);
    double S[9]; ai_vdv(W, e, S);
    double Fb[9]; mm3(Rb, S, Fb);
    for (int k=0;k<9;k++) FT[t*9+k] = (float)Fb[k];
}
// ---- A2: 응력 쪽 평균 (P2G) ----
__global__ void ai_tau_acc(int N, const long long* pinv, const float* vol, const float* Fp, float mu, float lam, float* TS) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    long long t = pinv[p];
    float F[9]; for (int k=0;k<9;k++) F[k] = Fp[p*9+k];
    float t6[6]; ai_tau(F, mu, lam, t6);
    float V = vol[p];
    for (int k=0;k<6;k++) atomicAdd(&TS[t*6+k], (float)(V*t6[k]));
}
// τ̄₀ = TS / V (TB), D₀ = τ̄₀ − τ(F̄) (D0). VTs 는 상한 전 (입자 부피 합) -- _sparse 앞에서 부른다.
__global__ void ai_tau_fin(const int* szp, const float* VTs, const float* TS, const float* FT, float mu, float lam, float* TB, float* D0) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VTs[t];
    if (!(V > 0.f)) { for (int k=0;k<6;k++) { TB[t*6+k] = 0.f; D0[t*6+k] = 0.f; } return; }
    double Fb[9]; for (int k=0;k<9;k++) Fb[k] = FT[t*9+k];
    double tf[6]; ai_tau(Fb, (double)mu, (double)lam, tf);
    for (int k=0;k<6;k++) { double tb = (double)TS[t*6+k]/(double)V; TB[t*6+k] = (float)tb; D0[t*6+k] = (float)(tb - tf[k]); }
}
// ---- A2 힘: c_fc (절 C 융합 판: 힘 + 바닥 접촉 판정) 와 같고, 사면체 블록에 응력 차 몫 −V R_A D₀ R_Aᵀ A⁻ᵀ ∇λ 를 더한다 ----
__device__ __forceinline__ void ai_off_body(int t, const long long* tets, const float* U, const float* gl, const float* VT, const float* D0, float* f) {
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    float dm = 0.f; for (int k=0;k<6;k++) dm = fmaxf(dm, fabsf(D0[t*6+k]));
    if (dm < OFFMIN) return;                                                       // 응력 차가 사실상 0 (입자 1 개 셀 등) -- 건너뜀
    float D[9] = {D0[t*6+0], D0[t*6+3], D0[t*6+4], D0[t*6+3], D0[t*6+1], D0[t*6+5], D0[t*6+4], D0[t*6+5], D0[t*6+2]};
    float Q[9]; c_polar_fast(A, Q);
    float QD[9], T1[9]; mm3(Q, D, QD); mmt3(QD, Q, T1);
    float cA[9]; cof3(A, cA);
    float dA = A[0]*cA[0] + A[1]*cA[1] + A[2]*cA[2];
    if (fabsf(dA) < 1e-6f) dA = copysignf(1e-6f, dA);
    float Gm[9]; mm3(T1, cA, Gm);       // T1 cof(A) / det A = T1 A⁻ᵀ  (cof 는 행렬 cof3 -- A⁻ᵀ = cof(A)/det A)
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*s/dA);
    }
}
__global__ void ai_fc(int nbT, const int* szp, const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                      float mu, float lam, float* f, float* rst, int fpol, float* eF,
                      const float* s0c, const unsigned int* dn, const long long* nidx, const float* x, const long long* rows, const float* lamb, const float* m, int tetmass,
                      float p0, float p1, float p2, float n0, float n1, float n2, float* acc, int* ncon, const float* D0) {
    if ((int)blockIdx.x < nbT) {
        int i0 = blockIdx.x*blockDim.x + threadIdx.x;
        int t;
        if (tl) { if (i0 >= tlc[0]) return; t = tl[i0]; }
        else { if (i0 >= szp[1]) return; t = i0; }
        c_force_body(t, tets, U, gl, FT, VT, mu, lam, f, 0.f, 0.f, 0, rst, fpol, eF);
        if (D0) ai_off_body(t, tets, U, gl, VT, D0, f);
        return;
    }
    int q = (blockIdx.x - nbT)*blockDim.x + threadIdx.x;
    if (q >= szp[2]) return;
    if (s0c[q] > __uint_as_float(*dn) + 1e-5f) return;
    long long p = nidx[q];
    float u[3] = {0,0,0};
    long long nd[4]; float lm[4];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm[i]*U[nd[i]*3+a]; }
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(s < 0.f)) return;
    float mp = m[p];
    for (int i=0;i<4;i++) atomicAdd(&acc[nd[i]], (tetmass ? 0.25f : lm[i])*mp);
    if (ncon) atomicAdd(ncon, 1);
}
// 같은 응력 차 몫만 (비융합 경로·검산용)
__global__ void ai_off(const int* szp, const long long* tets, const float* U, const float* gl, const float* VT, const float* D0, float* f) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= szp[1]) return;
    ai_off_body(t, tets, U, gl, VT, D0, f);
}

// ---- B: G2P (위치·속도는 ex_g2p_lagF 와 같은 barycentric 식), F 갱신만 mode 로 ----
//   2 bs     : F ← A_bs F,   A_bs = I + Σ_n U_n ⊗ ∇w̃_n(x_p)   (2 차 B-스플라인, 질량 있는 메시 노드만, Shepard 정규화)
//   3 dev    : F ← A_T F̄_T + R(A_T) (F − F̄_T)
//   4 devnavg: F ← Ã F̄_T + R(Ã) (F − F̄_T),   Ã = Σ λ_i Ā_n (fa_tet_acc which 0)
//   5 devbs  : F ← A_bs F̄_T + R(A_bs) (F − F̄_T)
//   bst[0] += 정규화 합 Σw 가 1 보다 작은 입자 수 (메시 밖·질량 0 노드를 뺀 입자), bst[1] += 그 1 − Σw 합
__global__ void ai_g2p(int N, const float* x, const float* Ucum, const float* Vn, const long long* rows, const float* lamb, const float* glT,
                       const long long* tinv, const float* Fp, float* x2, float* v2, float* F2, const float* FT, const float* NA, int mode,
                       const float* gpar, const int* ipar, const long long* ncm, const float* mn, float* bst) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double u[3]={0,0,0}, vv[3]={0,0,0}, A[9]={1,0,0,0,1,0,0,0,1}, Nt[9]={0,0,0,0,0,0,0,0,0};
    long long t = tinv[p];
    const float* gl = glT + t*12;
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; double lm=lamb[p*4+i];
        for (int a=0;a<3;a++){ double Ua=Ucum[nd*3+a]; double Va=(double)Vn[nd*3+a]; u[a]+=lm*Ua; vv[a]+=lm*Va;
            for (int b=0;b<3;b++) A[a*3+b]+=Ua*gl[i*3+b]; }
        if (mode == 4) { double W = (double)NA[nd*10+9]; if (W > 0.0) for (int k=0;k<9;k++) Nt[k] += lm*(double)NA[nd*10+k]/W; } }
    for (int k=0;k<3;k++) { x2[p*3+k] = (float)((double)x[p*3+k] + u[k]); v2[p*3+k] = (float)vv[k]; }
    double F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    double M[9];                           // 이 입자에 쓸 사상
    if (mode == 2 || mode == 5) {
        int n0=ipar[0], n1=ipar[1], n2=ipar[2];
        const float* Ai = gpar + 3;
        float y[3]; bsl_y(x + p*3, gpar, y);
        int b[3]; float W[3][3], D[3][3];
        for (int k=0;k<3;k++) bsl_w(y[k], b[k], W[k], D[k]);
        double Win = 0.0, gW[3] = {0,0,0};
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
            int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
            if (a0<0||a1<0||a2<0||a0>n0-1||a1>n1-1||a2>n2-1) continue;
            long long nd = ncm[((long long)a0*n1 + a1)*n2 + a2];
            if (nd < 0 || !(mn[nd] > 0.f)) continue;
            Win += (double)(W[0][i]*W[1][j]*W[2][l]);
            float gy0 = D[0][i]*W[1][j]*W[2][l], gy1 = W[0][i]*D[1][j]*W[2][l], gy2 = W[0][i]*W[1][j]*D[2][l];
            for (int c=0;c<3;c++) gW[c] += gy0*Ai[c] + gy1*Ai[3+c] + gy2*Ai[6+c];
        }
        if (Win < 0.999999) { atomicAdd(&bst[0], 1.f); atomicAdd(&bst[1], (float)(1.0 - Win)); }
        for (int k=0;k<9;k++) M[k] = (k%4==0) ? 1.0 : 0.0;
        if (Win > 1e-12) {
            for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
                int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
                if (a0<0||a1<0||a2<0||a0>n0-1||a1>n1-1||a2>n2-1) continue;
                long long nd = ncm[((long long)a0*n1 + a1)*n2 + a2];
                if (nd < 0 || !(mn[nd] > 0.f)) continue;
                double w = (double)(W[0][i]*W[1][j]*W[2][l]) / Win;
                float gy0 = D[0][i]*W[1][j]*W[2][l], gy1 = W[0][i]*D[1][j]*W[2][l], gy2 = W[0][i]*W[1][j]*D[2][l];
                double gx[3]; for (int c=0;c<3;c++) gx[c] = ((double)(gy0*Ai[c] + gy1*Ai[3+c] + gy2*Ai[6+c]) - w*gW[c]) / Win;
                for (int a=0;a<3;a++) { double Ua = Ucum[nd*3+a]; for (int c=0;c<3;c++) M[a*3+c] += Ua*gx[c]; }
            }
        } else {
            for (int k=0;k<9;k++) M[k] = A[k];
        }
    } else if (mode == 4) {
        for (int k=0;k<9;k++) M[k] = Nt[k];
    } else {
        for (int k=0;k<9;k++) M[k] = A[k];
    }
    double F2d[9];
    if (mode == 3 || mode == 4 || mode == 5) {
        double Fb[9]; for (int k=0;k<9;k++) Fb[k] = FT[t*9+k];
        double R[9], J; int inv;
        ai_polar(M, R, (double*)nullptr, (double*)nullptr, J, inv);
        double dF[9]; for (int k=0;k<9;k++) dF[k] = F[k] - Fb[k];
        double P1[9], P2[9]; mm3(M, Fb, P1); mm3(R, dF, P2);
        for (int k=0;k<9;k++) F2d[k] = P1[k] + P2[k];
    } else {
        mm3(M, F, F2d);
    }
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
}
}  // extern "C" (절 AI)
'''

_MOD = [None]


def mod():
    if _MOD[0] is None:
        import cupy as cp
        _MOD[0] = cp.RawModule(code=(RTC._SRC + EX._EX + CCU._C + _AI).encode("ascii", "ignore").decode(), options=("--std=c++14",))
    return _MOD[0]


def launch(name, n, args, b=128, grid=None):
    with RTC._Stream():
        g = grid if grid is not None else (int(n) + b - 1) // b
        mod().get_function(name)((g,), (b,), args)


P = EX.P


def set_offmin(v):
    import cupy as cp
    import numpy as np
    cp.ndarray((1,), cp.float32, mod().get_global("OFFMIN"))[:] = cp.asarray(np.asarray([v], dtype=np.float32))
