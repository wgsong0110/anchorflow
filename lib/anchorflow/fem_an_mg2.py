"""절 AN.20 MG 곱 가속 (결과 정의 같음 -- 반올림 순서만 다름) + 거친 층 점유 칸 목록 + 작은 거친 층 V 순환 한 블록 융합.

(1) 칸 협동 헤시안 곱 (--mv2): 칸 하나 = 8 레인 (팔분 가우스 점), 칸의 27 노드 dy 를 8 레인이 나눠 공유 메모리로 모으고,
    G_qn = F_qᵀ ∇w 를 노드마다 다시 만들지 않고 Y = Σ_o dy_o ⊗ ∇w_o (3x3) 로 한 번에: δX = Y F, δP = V H δX, Q = δP Fᵀ,
    노드 몫 = Σ_oc Q_oc ∇w_{oc,o} (8 레인이 27 노드를 나눠 원자 합 -- 머리 레인 하나가 81 번 하던 것). 식: δX_ab = Σ_o dy_oa g_ob, g = Fᵀ bg 이므로 δX = Y F;
    out_oa = Σ_b δP_ab g_ob = Σ_k (δP Fᵀ)_ak bg_ok. 같은 곱을 다른 순서로 더함 (반올림만 다름). 거친 층 기울기 배율 sc = 2^-l 은 Y·결과에 곱함 (2 의 거듭제곱이라 정확).
(2) 거친 층 점유 칸 목록 (--mg_occ): 조밀 상자 칸 중 부피 > 0 인 칸만 (칸 번호 오름차순, 한 블록 스캔). 빈 칸은 부피 0 이라 곱·대각에 기여가 원래 0 --
    연산자는 같고 일 (스레드) 만 줄어듦. 목록 용량 = 첫 프레임 상자에서 센 점유 칸 x cap_mul (넘치면 넘침 표시 -> 다시 할당).
(3) 작은 거친 층 V 순환 한 블록 (--mg_fuse L): 층 L 이상 (L ≥ 1) 의 V 순환 (앞 평활·잔차·제한·가장 거친 층 평활·연장·뒤 평활) 을 1024 스레드 한 블록 커널 하나로.
    곱은 칸 버퍼 (목록 자리 x 27 x 3) 에 쓰고 노드가 이웃 27 칸을 고정 순서로 모음 -> 원자 합 없음 (결정적 판·비결정 판이 같은 비트). 그래프 판의 같은 층 연산과 식이 같음.
"""
import numpy as np
import torch

MG2_SRC = r'''
// ================= 절 AN.20: 칸 협동 곱, 점유 칸 목록, 거친 층 한 블록 V 순환 =================
// 칸 협동 곱 핵심 (워프의 32 레인이 모두 부름; 칸 하나 = 연속 8 레인, j = 팔분). sW = 이 칸의 공유 메모리 81 칸 (dy 모음 뒤 Q 로 다시 씀).
// nid[r] = 노드 o = j + 8r 의 번호 (o < 27). res[r][a] = 노드 o 의 몫 (sc 배율 포함).
__device__ __forceinline__ void mv2_core(int j, bool on, size_t q, const int* nid, const float* dy, const float* BG, float sc, float* sW,
                                         const float* Fq, const float* Vq, const float* H, int capQ, float res[4][3]) {
    #pragma unroll
    for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) { float a0 = 0.f, a1 = 0.f, a2 = 0.f;
        if (on) { int n = nid[r]; a0 = dy[n*3]; a1 = dy[n*3+1]; a2 = dy[n*3+2]; } sW[o*3] = a0; sW[o*3+1] = a1; sW[o*3+2] = a2; } }
    __syncwarp();
    float Y[9] = {0,0,0,0,0,0,0,0,0};
    const float* bq = BG + j*81;
    if (on) {
        #pragma unroll 3
        for (int o=0;o<27;o++) { float y0 = sW[o*3], y1 = sW[o*3+1], y2 = sW[o*3+2]; float b0 = __ldg(bq + o*3), b1 = __ldg(bq + o*3+1), b2 = __ldg(bq + o*3+2);
            Y[0] += y0*b0; Y[1] += y0*b1; Y[2] += y0*b2; Y[3] += y1*b0; Y[4] += y1*b1; Y[5] += y1*b2; Y[6] += y2*b0; Y[7] += y2*b1; Y[8] += y2*b2; }
    }
    float Qv[9] = {0,0,0,0,0,0,0,0,0};
    float vq = on ? Vq[q] : 0.f;
    if (on && vq > 0.f) {
#ifdef AF_HSPEC
        // 절 CM3 --hspec (거친 층 rt_mg_mv2): 주값 분해 30 성분 -- Q = U P̂ Mᵀ, P̂ = C(Uᵀ (sc Y) M) (rt_mv2s 와 같은 식)
        float U[9], M[9], C[12];
        for (int k=0;k<9;k++) { U[k] = H[(size_t)k*capQ + q]; M[k] = H[(size_t)(9+k)*capQ + q]; }
        for (int k=0;k<12;k++) C[k] = H[(size_t)(18+k)*capQ + q];
        float T[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) T[a*3+b] = sc*(Y[a*3]*M[b] + Y[a*3+1]*M[3+b] + Y[a*3+2]*M[6+b]);
        float Xh[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Xh[a*3+b] = U[a]*T[b] + U[3+a]*T[3+b] + U[6+a]*T[6+b];
        float Ph[9];
        Ph[0] = C[0]*Xh[0] + C[3]*Xh[4] + C[4]*Xh[8];
        Ph[4] = C[3]*Xh[0] + C[1]*Xh[4] + C[5]*Xh[8];
        Ph[8] = C[4]*Xh[0] + C[5]*Xh[4] + C[2]*Xh[8];
        Ph[1] = C[6]*Xh[1] + C[9]*Xh[3];  Ph[3] = C[9]*Xh[1] + C[6]*Xh[3];
        Ph[2] = C[7]*Xh[2] + C[10]*Xh[6]; Ph[6] = C[10]*Xh[2] + C[7]*Xh[6];
        Ph[5] = C[8]*Xh[5] + C[11]*Xh[7]; Ph[7] = C[11]*Xh[5] + C[8]*Xh[7];
        float W[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) W[a*3+b] = U[a*3]*Ph[b] + U[a*3+1]*Ph[3+b] + U[a*3+2]*Ph[6+b];
        for (int a=0;a<3;a++) for (int k=0;k<3;k++) Qv[a*3+k] = W[a*3]*M[k*3] + W[a*3+1]*M[k*3+1] + W[a*3+2]*M[k*3+2];
#else
        float F[9]; for (int k=0;k<9;k++) F[k] = Fq[q*9+k];
        float dX[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) dX[a*3+b] = sc*(Y[a*3]*F[b] + Y[a*3+1]*F[3+b] + Y[a*3+2]*F[6+b]);
        float dP[9]; for (int r=0;r<9;r++) { float t = 0.f; for (int c=0;c<9;c++) t += H[(size_t)(r*9+c)*capQ + q]*dX[c]; dP[r] = vq*t; }
        for (int a=0;a<3;a++) for (int k=0;k<3;k++) Qv[a*3+k] = dP[a*3]*F[k*3] + dP[a*3+1]*F[k*3+1] + dP[a*3+2]*F[k*3+2];
#endif
    }
    __syncwarp();                                                                   // dy 읽기 끝 -> 같은 칸에 Q 쓰기
    for (int k=0;k<9;k++) sW[j*9+k] = Qv[k];
    __syncwarp();
    #pragma unroll
    for (int r=0;r<4;r++) { int o = j + 8*r; res[r][0] = res[r][1] = res[r][2] = 0.f; if (o >= 27) continue;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f;
        for (int oc=0; oc<8; oc++) { const float* Qo = sW + oc*9; const float* bg = BG + oc*81 + o*3; float b0 = __ldg(bg), b1 = __ldg(bg+1), b2 = __ldg(bg+2);
            s0 += Qo[0]*b0 + Qo[1]*b1 + Qo[2]*b2; s1 += Qo[3]*b0 + Qo[4]*b1 + Qo[5]*b2; s2 += Qo[6]*b0 + Qo[7]*b1 + Qo[8]*b2; }
        res[r][0] = sc*s0; res[r][1] = sc*s1; res[r][2] = sc*s2; }
    __syncwarp();                                                                   // 다음 칸이 sW 를 다시 쓰기 전
}
// 고운 층 곱 out += K dy (압축 칸, 칸 좌표 cb). 블록 128 = 칸 16.
__global__ void __launch_bounds__(128) rt_mv2(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* H, const float* dy, float* out, int capQ) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    __shared__ float sW[16*81];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int j = threadIdx.x & 7, c = qq >> 3;
    int nb1 = IP[7], nb2 = IP[8];
    int cx = cb[c*3], cy = cb[c*3+1], cz = cb[c*3+2];
    int nid[4];
    for (int r=0;r<4;r++) { int o = j + 8*r; if (o > 26) o = 26; nid[r] = ((cx + o/9)*nb1 + (cy + (o/3)%3))*nb2 + (cz + o%3); }
    float res[4][3];
    mv2_core(j, on, (size_t)qq, nid, dy, BG, 1.f, sW + (threadIdx.x >> 3)*81, Fq, Vq, H, capQ, res);
    if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) atomicAdd(&out[nid[r]*3+a], res[r][a]); }
}
// 절 CM3 --hspec: 고운 층 곱의 주값 분해 판 (rt_mv2 와 같은 칸 협동 틀). 점마다 H[k*capQ+q]: U 9, M = F V 9, C 12 (V_q 곱해 둠; 절 AO.16 ao_mv_q 와 같은 꼴).
// rt_mv2 의 Q = V_q (H (Y F)) Fᵀ 를 Q = U P̂ Mᵀ, P̂ = C(Uᵀ Y M) 로 (같은 선형 사상 -- 헤시안 = Σ 순위 1 항 9 개, e = U Ê Vᵀ; 반올림 순서만 다름). 읽기 30 개 (rt_mv2 90 개).
__global__ void __launch_bounds__(128) rt_mv2s(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* H, const float* dy, float* out, int capQ) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    __shared__ float sW[16*81];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int j = threadIdx.x & 7, c = qq >> 3;
    int nb1 = IP[7], nb2 = IP[8];
    int cx = cb[c*3], cy = cb[c*3+1], cz = cb[c*3+2];
    int nid[4];
    for (int r=0;r<4;r++) { int o = j + 8*r; if (o > 26) o = 26; nid[r] = ((cx + o/9)*nb1 + (cy + (o/3)%3))*nb2 + (cz + o%3); }
    float* sw = sW + (threadIdx.x >> 3)*81;
    #pragma unroll
    for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) { float a0 = 0.f, a1 = 0.f, a2 = 0.f;
        if (on) { int n = nid[r]; a0 = dy[n*3]; a1 = dy[n*3+1]; a2 = dy[n*3+2]; } sw[o*3] = a0; sw[o*3+1] = a1; sw[o*3+2] = a2; } }
    __syncwarp();
    float Y[9] = {0,0,0,0,0,0,0,0,0};
    const float* bq = BG + j*81;
    if (on) {
        #pragma unroll 3
        for (int o=0;o<27;o++) { float y0 = sw[o*3], y1 = sw[o*3+1], y2 = sw[o*3+2]; float b0 = __ldg(bq + o*3), b1 = __ldg(bq + o*3+1), b2 = __ldg(bq + o*3+2);
            Y[0] += y0*b0; Y[1] += y0*b1; Y[2] += y0*b2; Y[3] += y1*b0; Y[4] += y1*b1; Y[5] += y1*b2; Y[6] += y2*b0; Y[7] += y2*b1; Y[8] += y2*b2; }
    }
    float Qv[9] = {0,0,0,0,0,0,0,0,0};
    float vq = on ? Vq[qq] : 0.f;
    if (on && vq > 0.f) {
        float U[9], M[9], C[12];
        for (int k=0;k<9;k++) { U[k] = H[(size_t)k*capQ + qq]; M[k] = H[(size_t)(9+k)*capQ + qq]; }
        for (int k=0;k<12;k++) C[k] = H[(size_t)(18+k)*capQ + qq];
        float T[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) T[a*3+b] = Y[a*3]*M[b] + Y[a*3+1]*M[3+b] + Y[a*3+2]*M[6+b];
        float Xh[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Xh[a*3+b] = U[a]*T[b] + U[3+a]*T[3+b] + U[6+a]*T[6+b];
        float Ph[9];
        Ph[0] = C[0]*Xh[0] + C[3]*Xh[4] + C[4]*Xh[8];
        Ph[4] = C[3]*Xh[0] + C[1]*Xh[4] + C[5]*Xh[8];
        Ph[8] = C[4]*Xh[0] + C[5]*Xh[4] + C[2]*Xh[8];
        Ph[1] = C[6]*Xh[1] + C[9]*Xh[3];  Ph[3] = C[9]*Xh[1] + C[6]*Xh[3];
        Ph[2] = C[7]*Xh[2] + C[10]*Xh[6]; Ph[6] = C[10]*Xh[2] + C[7]*Xh[6];
        Ph[5] = C[8]*Xh[5] + C[11]*Xh[7]; Ph[7] = C[11]*Xh[5] + C[8]*Xh[7];
        float W[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) W[a*3+b] = U[a*3]*Ph[b] + U[a*3+1]*Ph[3+b] + U[a*3+2]*Ph[6+b];
        for (int a=0;a<3;a++) for (int k=0;k<3;k++) Qv[a*3+k] = W[a*3]*M[k*3] + W[a*3+1]*M[k*3+1] + W[a*3+2]*M[k*3+2];
    }
    __syncwarp();
    for (int k=0;k<9;k++) sw[j*9+k] = Qv[k];
    __syncwarp();
    float res[4][3];
    #pragma unroll
    for (int r=0;r<4;r++) { int o = j + 8*r; res[r][0] = res[r][1] = res[r][2] = 0.f; if (o >= 27) continue;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f;
        for (int oc=0; oc<8; oc++) { const float* Qo = sw + oc*9; const float* bg = BG + oc*81 + o*3; float b0 = __ldg(bg), b1 = __ldg(bg+1), b2 = __ldg(bg+2);
            s0 += Qo[0]*b0 + Qo[1]*b1 + Qo[2]*b2; s1 += Qo[3]*b0 + Qo[4]*b1 + Qo[5]*b2; s2 += Qo[6]*b0 + Qo[7]*b1 + Qo[8]*b2; }
        res[r][0] = s0; res[r][1] = s1; res[r][2] = s2; }
    if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) atomicAdd(&out[nid[r]*3+a], res[r][a]); }
}
// 거친 층 곱 (조밀 상자 칸 번호; list 가 있으면 점유 칸 목록만)
__global__ void __launch_bounds__(128) rt_mg_mv2(int l, const int* LM, const float* BG, float sc, const float* Fq, const float* Vq, const float* H, int capQ, const float* dy, float* out,
                                                 const int* list, const int* cnt) {
    const int* L = LM + l*8;
    int ncell = list ? cnt[l] : (L[7] >> 3);
    if (blockIdx.x*16 >= ncell) return;
    __shared__ float sW[16*81];
    int slot = blockIdx.x*16 + (threadIdx.x >> 3), j = threadIdx.x & 7;
    bool on = slot < ncell;
    int c = on ? (list ? list[slot] : slot) : 0;
    if (on && !(Vq[(size_t)c*8] > 0.f)) on = false;                                 // 빈 칸 (부피 0): 기여 0
    int cz = c % L[6], cy = (c / L[6]) % L[5], cx = c / (L[5]*L[6]);
    int nid[4];
    for (int r=0;r<4;r++) { int o = j + 8*r; if (o > 26) o = 26; nid[r] = ((cx + o/9)*L[1] + (cy + (o/3)%3))*L[2] + (cz + o%3); }
    float res[4][3];
    mv2_core(j, on, (size_t)c*8 + j, nid, dy, BG, sc, sW + (threadIdx.x >> 3)*81, Fq, Vq, H, capQ, res);
    if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) atomicAdd(&out[nid[r]*3+a], res[r][a]); }
}
// 제한 (rt_mg_restrict 와 같은 식·같은 더하는 순서; 고리를 펼쳐 64 x 3 읽기를 한꺼번에 띄움 -- 원래 판은 continue 고리라 읽기 지연이 줄줄이 쌓임)
__global__ void rt_mg_restrict2(int lf, const int* LM, const float* in, float* out, int ch) {
    const int* A = LM + lf*8; const int* B = LM + (lf+1)*8;
    int J = blockIdx.x*blockDim.x + threadIdx.x;
    if (J >= B[3]) return;
    int jz = J % B[2], jy = (J / B[2]) % B[1], jx = J / (B[1]*B[2]);
    const float cw[4] = {0.25f, 0.75f, 0.75f, 0.25f};
    float acc[3] = {0.f, 0.f, 0.f};
    #pragma unroll
    for (int k=0;k<4;k++) { int ix = 2*jx - 3 + k; bool okx = ix >= 0 && ix < A[0];
      #pragma unroll
      for (int m=0;m<4;m++) { int iy = 2*jy - 3 + m; bool oky = okx && iy >= 0 && iy < A[1];
        #pragma unroll
        for (int n=0;n<4;n++) { int iz = 2*jz - 3 + n; bool ok = oky && iz >= 0 && iz < A[2];
          if (ok) { float w = cw[k]*cw[m]*cw[n]; int i = (ix*A[1] + iy)*A[2] + iz;
            if (ch == 3) { acc[0] += w*__ldg(in + i*3); acc[1] += w*__ldg(in + i*3+1); acc[2] += w*__ldg(in + i*3+2); } else acc[0] += w*__ldg(in + i); } } } }
    for (int c=0;c<ch;c++) out[J*ch+c] = acc[c];
}
// 제한 3 (같은 식, 더하는 순서만 다름): 거친 노드 하나 = 연속 16 레인, 레인 (k, m) 이 n = 0..3 네 탭을 더한 뒤 16 레인 나비 합 (고정 순서 -> 결정적).
__global__ void rt_mg_restrict3(int lf, const int* LM, const float* in, float* out, int ch) {
    const int* A = LM + lf*8; const int* B = LM + (lf+1)*8;
    int g = blockIdx.x*blockDim.x + threadIdx.x;
    int J = g >> 4, t = threadIdx.x & 15;
    if ((blockIdx.x*blockDim.x >> 4) >= B[3]) return;                               // 블록 전체가 밖이면 (워프 단위 나비 합 앞이라 블록 단위로만)
    bool on = J < B[3];
    int JJ = on ? J : 0;
    int jz = JJ % B[2], jy = (JJ / B[2]) % B[1], jx = JJ / (B[1]*B[2]);
    const float cw[4] = {0.25f, 0.75f, 0.75f, 0.25f};
    int k = t >> 2, m = t & 3;
    int ix = 2*jx - 3 + k, iy = 2*jy - 3 + m;
    bool oxy = on && ix >= 0 && ix < A[0] && iy >= 0 && iy < A[1];
    float a0 = 0.f, a1 = 0.f, a2 = 0.f;
    #pragma unroll
    for (int n=0;n<4;n++) { int iz = 2*jz - 3 + n; bool ok = oxy && iz >= 0 && iz < A[2];
        int i = ok ? (ix*A[1] + iy)*A[2] + iz : 0; float w = ok ? cw[k]*cw[m]*cw[n] : 0.f;
        if (ch == 3) { float v0 = ok ? __ldg(in + i*3) : 0.f, v1 = ok ? __ldg(in + i*3+1) : 0.f, v2 = ok ? __ldg(in + i*3+2) : 0.f; a0 += w*v0; a1 += w*v1; a2 += w*v2; }
        else { float v0 = ok ? __ldg(in + i) : 0.f; a0 += w*v0; } }
    #pragma unroll
    for (int o=1;o<16;o<<=1) { a0 += __shfl_xor_sync(0xffffffffu, a0, o); a1 += __shfl_xor_sync(0xffffffffu, a1, o); a2 += __shfl_xor_sync(0xffffffffu, a2, o); }
    if (on && t == 0) { out[J*ch] = a0; if (ch == 3) { out[J*3+1] = a1; out[J*3+2] = a2; } }
}
// 점유 칸 목록 (한 블록 1024, 칸 번호 오름차순): list[i] = 칸, pos[칸] = 자리 또는 −1, cnt[l] = 수. 용량 넘치면 넘침 표시 (IP[13] |= 4).
__global__ void __launch_bounds__(1024) rt_mg_occ(int l, const int* LM, const float* Vq, int* list, int* pos, int* cnt, int cap, int* IP) {
    __shared__ int sh[1024];
    int n = LM[l*8+7] >> 3, carry = 0;
    for (int base = 0; base < n; base += 1024) {
        int i = base + threadIdx.x;
        int v = (i < n && Vq[(size_t)i*8] > 0.f) ? 1 : 0;
        sh[threadIdx.x] = v;
        __syncthreads();
        for (int o=1; o<1024; o<<=1) { int t = (threadIdx.x >= o) ? sh[threadIdx.x - o] : 0; __syncthreads(); sh[threadIdx.x] += t; __syncthreads(); }
        int p = carry + sh[threadIdx.x] - 1;
        if (i < n) { if (v && p < cap) { list[p] = i; pos[i] = p; } else pos[i] = -1; }
        int tot = sh[1023];
        __syncthreads();
        carry += tot;
    }
    if (threadIdx.x == 0) { cnt[l] = carry < cap ? carry : cap; if (carry > cap) atomicOr(&IP[13], 4); }
}
// ---- 거친 층 V 순환 한 블록 512 스레드 (층 lb .. nl−1) ----
// 층 표 MP[l*16 + k] (장치 주소): 0 Fq, 1 Vq, 2 H, 3 mc, 4 D, 5 r, 6 z, 7 t (잔차), 8 KT (칸 몫 모음 결과), 9 list, 10 pos, 11 cbuf (자리 x 27 x 3)
// 정수 MI[l*4 + k]: 0 capq
__device__ void cvc_Kz(int l, const int* L, const unsigned long long* MP, const int* MI, const int* cnt, const float* BG, const float* z, float* KT, float* sW) {
    const float* Fq = (const float*)MP[l*16+0]; const float* Vq = (const float*)MP[l*16+1]; const float* H = (const float*)MP[l*16+2];
    const int* list = (const int*)MP[l*16+9]; const int* pos = (const int*)MP[l*16+10]; float* cbuf = (float*)MP[l*16+11];
    int capQ = MI[l*4];
    float sc = ldexpf(1.f, -l);
    int ncell = cnt[l];
    int j = threadIdx.x & 7;
    for (int base = 0; base < ncell; base += (blockDim.x >> 3)) {                    // 칸 몫 -> 칸 버퍼 (블록 전체가 같은 횟수 돎)
        int slot = base + (threadIdx.x >> 3);
        bool on = slot < ncell;
        int c = on ? list[slot] : 0;
        int cz = c % L[6], cy = (c / L[6]) % L[5], cx = c / (L[5]*L[6]);
        int nid[4];
        for (int r=0;r<4;r++) { int o = j + 8*r; if (o > 26) o = 26; nid[r] = ((cx + o/9)*L[1] + (cy + (o/3)%3))*L[2] + (cz + o%3); }
        float res[4][3];
        mv2_core(j, on, (size_t)c*8 + j, nid, z, BG, sc, sW + (threadIdx.x >> 3)*81, Fq, Vq, H, capQ, res);
        if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) cbuf[((size_t)slot*27+o)*3+a] = res[r][a]; }
    }
    __syncthreads();
    for (int n = threadIdx.x; n < L[3]; n += blockDim.x) {                           // 노드가 이웃 27 칸을 고정 순서로
        int ni = n / (L[1]*L[2]), nj = (n / L[2]) % L[1], nk = n % L[2];
        float s0 = 0.f, s1 = 0.f, s2 = 0.f;
        int pp[27];
        #pragma unroll
        for (int o=0;o<27;o++) {                                                     // 자리 읽기를 먼저 한꺼번에 (지연 겹침), 더하는 순서는 o 순서 그대로
            int ci = ni - o/9, cj = nj - (o/3)%3, ck = nk - o%3;
            bool ok = !(ci < 0 || cj < 0 || ck < 0 || ci >= L[4] || cj >= L[5] || ck >= L[6]);
            pp[o] = ok ? pos[(ci*L[5] + cj)*L[6] + ck] : -1;
        }
        #pragma unroll
        for (int o=0;o<27;o++) { if (pp[o] < 0) continue;
            const float* cv = cbuf + ((size_t)pp[o]*27+o)*3; s0 += cv[0]; s1 += cv[1]; s2 += cv[2];
        }
        KT[n*3] = s0; KT[n*3+1] = s1; KT[n*3+2] = s2;
    }
    __syncthreads();
}
__device__ __forceinline__ void cvc_restrict(const int* A, const int* B, int J, const float* in, float* out) {
    int jz = J % B[2], jy = (J / B[2]) % B[1], jx = J / (B[1]*B[2]);
    const float cw[4] = {0.25f, 0.75f, 0.75f, 0.25f};
    float acc[3] = {0.f, 0.f, 0.f};
    #pragma unroll
    for (int k=0;k<4;k++) { int ix = 2*jx - 3 + k; bool okx = ix >= 0 && ix < A[0];
      #pragma unroll
      for (int m=0;m<4;m++) { int iy = 2*jy - 3 + m; bool oky = okx && iy >= 0 && iy < A[1];
        #pragma unroll
        for (int n=0;n<4;n++) { int iz = 2*jz - 3 + n; if (oky && iz >= 0 && iz < A[2]) {
          float w = cw[k]*cw[m]*cw[n]; int i = (ix*A[1] + iy)*A[2] + iz;
          for (int c=0;c<3;c++) acc[c] += w*in[i*3+c]; } } } }
    for (int c=0;c<3;c++) out[J*3+c] = acc[c];
}
__device__ __forceinline__ void cvc_prolong_add(const int* A, const int* B, int I, const float* in, float* out) {
    int iz = I % A[2], iy = (I / A[2]) % A[1], ix = I / (A[1]*A[2]);
    const float cw[4] = {0.25f, 0.75f, 0.75f, 0.25f};
    int ii[3] = {ix, iy, iz};
    int Jc[3][2]; float Wc[3][2];
    for (int a=0;a<3;a++) { int j0 = (ii[a] + 3) >> 1; int k0 = ii[a] + 3 - 2*j0; Jc[a][0] = j0; Wc[a][0] = cw[k0]; Jc[a][1] = j0 - 1; Wc[a][1] = cw[k0 + 2]; }
    float acc[3] = {0.f, 0.f, 0.f};
    for (int p=0;p<2;p++) { int jx = Jc[0][p]; if (jx < 0 || jx >= B[0]) continue;
      for (int q=0;q<2;q++) { int jy = Jc[1][q]; if (jy < 0 || jy >= B[1]) continue;
        for (int r=0;r<2;r++) { int jz = Jc[2][r]; if (jz < 0 || jz >= B[2]) continue;
          float w = Wc[0][p]*Wc[1][q]*Wc[2][r]; int J = (jx*B[1] + jy)*B[2] + jz;
          for (int c=0;c<3;c++) acc[c] += w*in[J*3+c]; } } }
    for (int c=0;c<3;c++) out[I*3+c] = out[I*3+c] + acc[c];
}
// mode 0: z = w D r; 1: z += w D (r − (Mi z + KT)); 2: t = r − (Mi z + KT)
__device__ __forceinline__ void cvc_node(int l, const int* L, const unsigned long long* MP, int mode, float w, float itau2) {
    const float* mc = (const float*)MP[l*16+3]; const float* D = (const float*)MP[l*16+4]; const float* r = (const float*)MP[l*16+5];
    float* z = (float*)MP[l*16+6]; float* t = (float*)MP[l*16+7]; const float* KT = (const float*)MP[l*16+8];
    for (int n = threadIdx.x; n < L[3]; n += blockDim.x) {
        const float* d = D + n*9;
        float e[3];
        if (mode == 0) { for (int a=0;a<3;a++) e[a] = r[n*3+a]; }
        else { float Mi = mc[n]*itau2; for (int a=0;a<3;a++) e[a] = r[n*3+a] - (Mi*z[n*3+a] + KT[n*3+a]); }
        if (mode == 2) { for (int a=0;a<3;a++) t[n*3+a] = e[a]; continue; }
        for (int a=0;a<3;a++) { float u = w*(d[a*3]*e[0] + d[a*3+1]*e[1] + d[a*3+2]*e[2]); z[n*3+a] = (mode == 0) ? u : z[n*3+a] + u; }
    }
    __syncthreads();
}
__global__ void __launch_bounds__(512) rt_mg_cvc(int lb, int nl, int nc, const int* LM, const unsigned long long* MP, const int* MI, const int* cnt, const double* W,
                                                  const float* BG, float itau2) {
    __shared__ float sW[64*81];
    for (int l = lb; l < nl - 1; l++) {                                              // 내려감: z = ωDr, 잔차, 제한
        const int* L = LM + l*8; float w = (float)W[l];
        cvc_node(l, L, MP, 0, w, itau2);
        cvc_Kz(l, L, MP, MI, cnt, BG, (const float*)MP[l*16+6], (float*)MP[l*16+8], sW);
        cvc_node(l, L, MP, 2, w, itau2);
        const int* Lc = LM + (l+1)*8; const float* t = (const float*)MP[l*16+7]; float* rc = (float*)MP[(l+1)*16+5];
        for (int J = threadIdx.x; J < Lc[3]; J += blockDim.x) cvc_restrict(L, Lc, J, t, rc);
        __syncthreads();
    }
    { int l = nl - 1; const int* L = LM + l*8; float w = (float)W[l];                // 가장 거친 층 평활
      cvc_node(l, L, MP, 0, w, itau2);
      for (int it = 1; it < nc; it++) { cvc_Kz(l, L, MP, MI, cnt, BG, (const float*)MP[l*16+6], (float*)MP[l*16+8], sW); cvc_node(l, L, MP, 1, w, itau2); } }
    for (int l = nl - 2; l >= lb; l--) {                                             // 올라감: 연장 더하기, 뒤 평활
        const int* L = LM + l*8; const int* Lc = LM + (l+1)*8; float w = (float)W[l];
        float* z = (float*)MP[l*16+6]; const float* zc = (const float*)MP[(l+1)*16+6];
        for (int I = threadIdx.x; I < L[3]; I += blockDim.x) cvc_prolong_add(L, Lc, I, zc, z);
        __syncthreads();
        cvc_Kz(l, L, MP, MI, cnt, BG, z, (float*)MP[l*16+8], sW);
        cvc_node(l, L, MP, 1, w, itau2);
    }
    for (int l = lb; l < nl; l++) { const int* L = LM + l*8; float* KT = (float*)MP[l*16+8];   // KT 비움 (그래프 판 곱·거듭제곱법은 원자 합 바탕 0 을 기대)
        for (int i = threadIdx.x; i < L[3]*3; i += blockDim.x) KT[i] = 0.f; }
}
'''


def mg2_src(src):
    """fem_an_rt SRC (닫는 괄호 포함) 끝에 MG2 커널을 넣는다."""
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + MG2_SRC + "\n}\n"


def det_mg2(s):
    """결정적 판: 칸 협동 곱의 원자 합을 칸 버퍼 쓰기로 (모으기 커널은 원래 판 것 그대로: 고운 층 ao_gath_f, 거친 층 ao_mg_gath)."""
    from .fem_an_det import _sub_in
    s = _sub_in(s, "rt_mv2", "if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) atomicAdd(&out[nid[r]*3+a], res[r][a]); }",
                "if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) { if (AO_DET) AO_CV[((size_t)c*27+o)*3+a] = res[r][a]; else atomicAdd(&out[nid[r]*3+a], res[r][a]); } }")
    s = _sub_in(s, "rt_mv2s", "if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) atomicAdd(&out[nid[r]*3+a], res[r][a]); }",
                "if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) { if (AO_DET) AO_CV[((size_t)c*27+o)*3+a] = res[r][a]; else atomicAdd(&out[nid[r]*3+a], res[r][a]); } }")
    s = _sub_in(s, "rt_mg_mv2", "if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) atomicAdd(&out[nid[r]*3+a], res[r][a]); }",
                "if (on) for (int r=0;r<4;r++) { int o = j + 8*r; if (o < 27) for (int a=0;a<3;a++) { if (AO_DET) AO_MGV[((size_t)c*27+o)*3+a] = res[r][a]; else atomicAdd(&out[nid[r]*3+a], res[r][a]); } }")
    return s


def occ_counts(cells, nl, cap_mul):
    """첫 프레임 고운 층 점유 칸 좌표 (n x 3, 상자 기준) -> 층 1..nl−1 점유 칸 수 (rt_mg_acc 의 사상과 같은 식: 출처 칸의 팔분 8 점 -> 거친 칸). 목록 용량 = 수 x cap_mul + 64."""
    GG = 0.28867513459481287
    dev = cells.device
    offs = torch.tensor([[(GG if (oc >> (2 - a)) & 1 else -GG) for a in range(3)] for oc in range(8)], device=dev, dtype=torch.float64)
    cur = cells.double()
    caps = [0]
    for l in range(1, nl):
        tq = (cur[:, None, :] + 1.0 + offs[None]).reshape(-1, 3)
        t = 0.5 * (tq + 1.5)
        b = torch.floor(t - 0.5)
        cur = torch.unique(b, dim=0)
        caps.append(int(cur.shape[0] * cap_mul) + 64)
    return caps


class MG2Mixin:
    """RTSolver 에 섞는 판 (fem_an_rt.RTSolver 가 플래그를 보고 부름)."""

    def mg2_alloc(s, cells):
        dev = s.dev
        s.mg_cnt = torch.zeros(max(s.nl, 1), device=dev, dtype=torch.int32)
        if s.nl <= 1:
            return
        caps = occ_counts(cells, s.nl, s.cap_mul) if (s.mg_occ or s.mg_fuse) else [0] * s.nl
        for l in range(1, s.nl):
            L = s.ML[l - 1]
            dense = L["capq"] // 8
            L["lcap"] = min(dense, caps[l]) if (s.mg_occ or s.mg_fuse) else dense
            L["list"] = torch.zeros(L["lcap"], device=dev, dtype=torch.int32)
            L["pos"] = torch.zeros(dense, device=dev, dtype=torch.int32)
            if s.mg_fuse and l >= s.mg_fuse:
                L["cbuf"] = torch.zeros(L["lcap"] * 81, device=dev, dtype=torch.float32)
        if s.mg_fuse:
            MP = np.zeros((s.nl, 16), dtype=np.uint64); MI = np.zeros((s.nl, 4), dtype=np.int32)
            for l in range(s.mg_fuse, s.nl):
                L = s.ML[l - 1]
                for k_, t_ in enumerate((L["Fq"], L["Vq"], L["H"], L["mc"], L["D"], L["r"], L["z"], L["t"], L["KT"], L["list"], L["pos"], L["cbuf"])):
                    MP[l, k_] = np.uint64(t_.data_ptr())
                MI[l, 0] = L["capq"]
            s.mg_MP = torch.from_numpy(MP.view(np.int64).reshape(-1).copy()).to(dev)
            s.mg_MI = torch.from_numpy(MI.reshape(-1).copy()).to(dev)

    def mg2_occ(s, l):
        """mg_build 안: 층 l 의 rt_mg_fin 뒤 점유 칸 목록."""
        if not (s.mg_occ or (s.mg_fuse and l >= s.mg_fuse)):
            return
        L = s.ML[l - 1]
        s.k("rt_mg_occ", 1024, (np.int32(l), P(s.LM), P(L["Vq"]), P(L["list"]), P(L["pos"]), P(s.mg_cnt), np.int32(L["lcap"]), P(s.IP)), b=1024)

    def mg2_mv(s, l, z, KT):
        i32, f = np.int32, np.float32
        if l == 0:
            s.k(s.fmv_name(), s.capQc * 8, (P(s.IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.fmv_H()), P(z), P(KT), i32(s.capQc * 8)))
            return
        L = s.ML[l - 1]
        use = s.mg_occ
        n = (L["lcap"] if use else L["capq"] // 8) * 8
        s.k("rt_mg_mv2", n, (i32(l), P(s.LM), P(s.BG), f(0.5 ** l), P(L["Fq"]), P(L["Vq"]), P(L["H"]), i32(L["capq"]), P(z), P(KT),
                             P(L["list"] if use else None), P(s.mg_cnt if use else None)))

    def mg2_cvc(s, r, z):
        """층 mg_fuse 이상 V 순환 한 블록. r·z 는 그 층의 r·z (ML 버퍼와 같음)."""
        L = s.ML[s.mg_fuse - 1]
        assert r.data_ptr() == L["r"].data_ptr() and z.data_ptr() == L["z"].data_ptr()
        it2 = np.float32(1.0 / (getattr(s, 'mg_tau', s.tau) ** 2))
        s.k("rt_mg_cvc", 512, (np.int32(s.mg_fuse), np.int32(s.nl), np.int32(s.mg_nc), P(s.LM), P(s.mg_MP), P(s.mg_MI), P(s.mg_cnt), P(s.W), P(s.BG), it2), b=512)

    def kbench(s, reps=20, iters=20):
        """커널 하나씩 그래프 (같은 커널 reps 번) 로 재생 시간 -> {이름: µs/회}. 상태는 되돌림 (값은 의미 없음, 시간만)."""
        i32, f = np.int32, np.float32
        bk = s._bk()
        saved = {k_: t_.clone() for k_, t_ in (("m0u", s.m0u), ("m0KT", s.m0KT), ("KTp", s.KTp))}
        for mod in s._mods():
            try:
                g = mod.get_global("AF_SKIP"); import cupy as cp; cp.ndarray((1,), dtype=cp.int32, memptr=g)[0] = 0
            except Exception:
                pass
        nr = s.floor["NR"] if s.floor else [0.0, 0.0, 1.0]
        it2 = f(1.0 / (getattr(s, 'mg_tau', s.tau) ** 2))
        cases = []
        capQ = s.capQc * 8
        cases.append(("empty", lambda: s.k("rt_sc_copy", 1, (P(s.SC), i32(63), i32(63)), b=1)))
        cases.append(("rt_mv", lambda: s.k("rt_mv", capQ, (P(s.IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.H), P(s.m0u), P(s.m0KT), i32(capQ)))))
        cases.append(("rt_mv2", lambda: s.k("rt_mv2", capQ, (P(s.IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.H), P(s.m0u), P(s.m0KT), i32(capQ)))))
        cases.append(("node0", lambda: s.k("rt_mg_node", s.capN, (i32(0), P(s.LM), i32(1), P(s.meff), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.Dinv), P(s.W), P(s.r), P(s.m0KT), P(s.m0t), P(None), P(None)), b=256)))
        cases.append(("rt_cg2", lambda: s.k("rt_cg2", s.capN, (P(s.IP), P(s.SC), P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.zo), P(s.p0), P(s.yc), P(None)), b=256)))
        cases.append(("rt_dotz", lambda: s.k("rt_dotz", s.capN, (P(s.IP), P(s.r), P(s.m0t), P(s.zo), P(s.SC), i32(0)), b=256)))
        for l in range(1, s.nl):
            L = s.ML[l - 1]
            cases.append((f"mg_mv:{l}", lambda l=l, L=L: s.k("rt_mg_mv", L["capq"], (i32(l), P(s.LM), P(s.BG), f(0.5 ** l), P(L["Fq"]), P(L["Vq"]), P(L["H"]), i32(L["capq"]), P(L["y"]), P(L["KT"])))))
            cases.append((f"mg_mv2d:{l}", lambda l=l, L=L: s.k("rt_mg_mv2", L["capq"], (i32(l), P(s.LM), P(s.BG), f(0.5 ** l), P(L["Fq"]), P(L["Vq"]), P(L["H"]), i32(L["capq"]), P(L["y"]), P(L["KT"]), P(None), P(None)))))
            if s.mg_occ or s.mg_fuse:
                cases.append((f"mg_mv2o:{l}", lambda l=l, L=L: s.k("rt_mg_mv2", L["lcap"] * 8, (i32(l), P(s.LM), P(s.BG), f(0.5 ** l), P(L["Fq"]), P(L["Vq"]), P(L["H"]), i32(L["capq"]), P(L["y"]), P(L["KT"]), P(L["list"]), P(s.mg_cnt)))))
            cases.append((f"node:{l}", lambda l=l, L=L: s.k("rt_mg_node", L["capn"], (i32(l), P(s.LM), i32(1), P(L["mc"]), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(L["D"]), P(s.W), P(L["r"]), P(L["KT"]), P(L["y"]), P(None), P(None)), b=256)))
            cases.append((f"restrict:{l - 1}", lambda l=l, L=L: s.k("rt_mg_restrict", L["capn"], (i32(l - 1), P(s.LM), P(s.m0t if l == 1 else s.ML[l - 2]["t"]), P(L["t"]), i32(3)), b=256)))
            cases.append((f"restrict2:{l - 1}", lambda l=l, L=L: s.k("rt_mg_restrict2", L["capn"], (i32(l - 1), P(s.LM), P(s.m0t if l == 1 else s.ML[l - 2]["t"]), P(L["t"]), i32(3)), b=256)))
            cases.append((f"restrict3:{l - 1}", lambda l=l, L=L: s.k("rt_mg_restrict3", L["capn"] * 16, (i32(l - 1), P(s.LM), P(s.m0t if l == 1 else s.ML[l - 2]["t"]), P(L["t"]), i32(3)), b=256)))
            cases.append((f"prolong:{l - 1}", lambda l=l, L=L: s.k("rt_mg_prolong", (s.capN if l == 1 else s.ML[l - 2]["capn"]), (i32(l - 1), P(s.LM), P(L["y"]), P(s.zo if l == 1 else s.ML[l - 2]["y"]), i32(0)), b=256)))
        if s.mg_fuse:
            def cvc():
                L = s.ML[s.mg_fuse - 1]
                s.mg2_cvc(L["r"], L["z"])
            cases.append((f"cvc:{s.mg_fuse}", cvc))
        out = {}
        for nm, fn in cases:
            try:
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    fn()
                torch.cuda.current_stream().wait_stream(st)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(reps):
                        fn()
                g.replay(); torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(iters):
                    g.replay()
                e1.record(); torch.cuda.synchronize()
                out[nm] = round(1000.0 * e0.elapsed_time(e1) / (reps * iters), 2)
                del g
            except Exception as ex:
                out[nm] = "err %s" % str(ex)[:120]
        s._rs(bk)
        for k_, t_ in saved.items():
            getattr(s, k_).copy_(t_)
        return out


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())
