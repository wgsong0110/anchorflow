"""FEM 실시간 판 손으로 쓴 CUDA 커널 (사용자 10-28 "커스텀 커널로 짜서 최적화할 수 있는 거 다", 목표 60 fps; doc/fem_realtime.md 절 H).

CuPy RawModule (NVRTC -- 호스트 컴파일러 불필요) 로 컴파일해 torch 텐서 포인터에 바로 띄운다 (같은 CUDA 스트림).
fem_rt.NodeFrame (--floor node, 사면체 적분점 + 노드 바닥 항) 의 같은 식을 사면체당 스레드 하나로:

  tet_eg      E·기울기 (float64): 관성 ½UᵀM4U − B4·U, F = A F̄, ψ = μΣ(σ−1)² + λ/2(J−1)² (σ = √eig(FᵀF), 하한 0.01 -- PR._fcr_fwd_core 와 같은 닫힌 식),
              dψ/dF = 2μ(F − Q) + λ(J−1) sgn(det F) cof F (Q = 크기 조정 뉴턴 극분해 10 회 -- PR._fcr_grad_core), 노드로 원자 합.
  tet_hess    사면체 12x12 = M4⊗I + Bᵀ K B (K = V ∂(P F̄ᵀ)/∂A -- fem_fast._k_manual 과 같은 전진 미분 극분해 10 회, σ_min ≤ 0.01 이면 0),
              대칭화, 12x12 순환 야코비 고유분해 (float32, 레지스터/지역 메모리), |λ| 투영 (바닥 floor·max|λ|), 블록 ELL 칸으로 원자 합.
  pcg_*       블록 3x3 ELL (노드당 15 칸) 곱 + 내적 융합, 블록 야코비, 스칼라 판정을 GPU 에서 (동기화 없음), 반복 묶음을 CUDA 그래프로.
              행렬 float32, 벡터·내적 float64. 버퍼는 최대 크기로 미리 잡아 그래프를 프레임 넘어 재사용 (노드 수는 장치 정수로).
"""
from __future__ import annotations

import numpy as np
import torch

_SRC = r'''
template <typename T> __device__ __forceinline__ void cof3(const T* A, T* C) {
    T a=A[0],b=A[1],c=A[2],d=A[3],e=A[4],f=A[5],g=A[6],h=A[7],i=A[8];
    C[0]=e*i-f*h; C[1]=f*g-d*i; C[2]=d*h-e*g;
    C[3]=c*h-b*i; C[4]=a*i-c*g; C[5]=b*g-a*h;
    C[6]=b*f-c*e; C[7]=c*d-a*f; C[8]=a*e-b*d;
}
template <typename T> __device__ __forceinline__ void coflin(const T* A, const T* D, T* C) {
    T a=A[0],b=A[1],c=A[2],d=A[3],e=A[4],f=A[5],g=A[6],h=A[7],i=A[8];
    T da=D[0],db=D[1],dc=D[2],dd=D[3],de=D[4],df=D[5],dg=D[6],dh=D[7],di=D[8];
    C[0]=de*i+e*di-df*h-f*dh; C[1]=df*g+f*dg-dd*i-d*di; C[2]=dd*h+d*dh-de*g-e*dg;
    C[3]=dc*h+c*dh-db*i-b*di; C[4]=da*i+a*di-dc*g-c*dg; C[5]=db*g+b*dg-da*h-a*dh;
    C[6]=db*f+b*df-dc*e-c*de; C[7]=dc*d+c*dd-da*f-a*df; C[8]=da*e+a*de-db*d-b*dd;
}
template <typename T> __device__ __forceinline__ T det3(const T* Q) {
    return Q[0]*(Q[4]*Q[8]-Q[5]*Q[7]) - Q[1]*(Q[3]*Q[8]-Q[5]*Q[6]) + Q[2]*(Q[3]*Q[7]-Q[4]*Q[6]);
}
template <typename T> __device__ __forceinline__ void polar(const T* F, T* Q) {
    for (int k=0;k<9;k++) Q[k]=F[k];
    T C[9];
    for (int it=0; it<10; it++) {
        T dt = det3(Q);
        T ds = (fabs(dt) > (T)1e-30) ? dt : (T)1e-30;
        T gm = pow(fabs(ds), (T)(-1.0/3.0));
        cof3(Q, C);
        for (int k=0;k<9;k++) Q[k] = (T)0.5*(gm*Q[k] + C[k]/(ds*gm));
    }
}
// 대칭 3x3 의 고유값 오름차순 (phys_resid._sym_eig3 과 같은 삼각 닫힌 식)
template <typename T> __device__ __forceinline__ void symeig3(const T* C, T* l) {
    T a=C[0], b=C[4], c=C[8], d=C[1], e=C[5], f=C[2];
    T q=(a+b+c)/(T)3; T p1=d*d+e*e+f*f;
    T p2=(a-q)*(a-q)+(b-q)*(b-q)+(c-q)*(c-q)+(T)2*p1;
    T p=sqrt(fmax(p2/(T)6,(T)1e-30)); T ip=(T)1/p;
    T B00=(a-q)*ip, B11=(b-q)*ip, B22=(c-q)*ip, B01=d*ip, B12=e*ip, B02=f*ip;
    T r=(T)0.5*(B00*(B11*B22-B12*B12)-B01*(B01*B22-B12*B02)+B02*(B01*B12-B11*B02));
    r = fmin(fmax(r,(T)-1),(T)1);
    T phi=acos(r)/(T)3;
    T l3=q+(T)2*p*cos(phi);
    T l1=q+(T)2*p*cos(phi+(T)(2.0*3.14159265358979323846/3.0));
    l[0]=l1; l[2]=l3; l[1]=(T)3*q-l1-l3;
}
template <typename T> __device__ __forceinline__ void mm3(const T* A, const T* B, T* C) {  // C = A B
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) C[r*3+c]=A[r*3]*B[c]+A[r*3+1]*B[3+c]+A[r*3+2]*B[6+c];
}
template <typename T> __device__ __forceinline__ void mmt3(const T* A, const T* B, T* C) { // C = A Bᵀ
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) C[r*3+c]=A[r*3]*B[c*3]+A[r*3+1]*B[c*3+1]+A[r*3+2]*B[c*3+2];
}

__device__ __forceinline__ double warp_sum(double v) {
    for (int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ void block_add(double v, double* out) {
    __shared__ double sh[32];
    __syncthreads();
    v = warp_sum(v);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane==0) sh[w]=v;
    __syncthreads();
    if (w==0) {
        v = (lane < (blockDim.x>>5)) ? sh[lane] : 0.0;
        v = warp_sum(v);
        if (lane==0) atomicAdd(out, v);
    }
}

extern "C" {

// ---------------- E·기울기 (float64) ----------------
__global__ void tet_eg(int T, const long long* tets, const float* U, const double* gl, const double* FT, const double* VT,
                       const double* M4, const double* B4, double mu, double lam, int want_g, double* E, double* g, int* nlow) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0.0;
    if (t < T) {
        long long n[4]; double u[4][3], G[12];
        for (int i=0;i<4;i++){ n[i]=tets[t*4+i]; for(int c=0;c<3;c++) u[i][c]=(double)U[n[i]*3+c]; }
        for (int k=0;k<12;k++) G[k]=gl[t*12+k];
        // 관성
        const double* m4 = M4 + t*16; const double* b4 = B4 + t*12;
        double gi[4][3];
        for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
            double s=0; for (int j=0;j<4;j++) s += m4[i*4+j]*u[j][c];
            e += 0.5*s*u[i][c] - b4[i*3+c]*u[i][c];
            gi[i][c] = s - b4[i*3+c];
        }
        // A = I + Σ u_i ⊗ ∇λ_i
        double A[9];
        for (int c=0;c<3;c++) for (int b=0;b<3;b++) {
            double s = (c==b)?1.0:0.0;
            for (int i=0;i<4;i++) s += u[i][c]*G[i*3+b];
            A[c*3+b]=s;
        }
        const double* Fb = FT + t*9;
        double X[9]; mm3(A, Fb, X);
        double C[9];
        for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
        double l[3]; symeig3(C, l);
        double S[3], J=1.0, ps=0.0;
        for (int k=0;k<3;k++){ S[k]=fmax(sqrt(fmax(l[k],1e-12)),0.01); J*=S[k]; ps += (S[k]-1.0)*(S[k]-1.0); }
        double V = VT[t];
        e += V*(mu*ps + 0.5*lam*(J-1.0)*(J-1.0));
        if (S[0] <= 0.01) atomicAdd(nlow, 1);
        if (want_g) {
            double Q[9], cf[9], P[9], Gm[9];
            polar(X, Q); cof3(X, cf);
            double dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
            double sg = (dt>0)?1.0:((dt<0)?-1.0:0.0);
            for (int k=0;k<9;k++) P[k] = 2.0*mu*(X[k]-Q[k]) + lam*(J-1.0)*sg*cf[k];
            mmt3(P, Fb, Gm);
            for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
                double s = gi[i][c];
                for (int b=0;b<3;b++) s += V*Gm[c*3+b]*G[i*3+b];
                atomicAdd(&g[n[i]*3+c], s);
            }
        }
    }
    block_add(e, E);
}

// ---------------- 사면체 헤시안 + |λ| 투영 + 블록 ELL 원자 합 ----------------
__global__ void tet_hess(int T, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                         const float* M4, float mu, float lam, float efloor, const int* slots, float* Bell, int* nneg, int maxsw, float jtol) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float u[4][3], G[12];
    for (int i=0;i<4;i++){ long long n=tets[t*4+i]; for(int c=0;c<3;c++) u[i][c]=U[n*3+c]; }
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9];
    for (int c=0;c<3;c++) for (int b=0;b<3;b++) {
        float s=(c==b)?1.f:0.f; for (int i=0;i<4;i++) s += u[i][c]*G[i*3+b]; A[c*3+b]=s;
    }
    const float* Fb = FT + t*9;
    float X[9]; mm3(A, Fb, X);
    float Cx[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) Cx[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(Cx, l);
    float smin = sqrtf(fmaxf(l[0],1e-12f));
    float V = VT[t];
    float K[81];
    if (smin <= 0.01f) { for (int k=0;k<81;k++) K[k]=0.f; }
    else {
        float cf[9]; cof3(X, cf);
        float det = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
        float sg = (det>0)?1.f:((det<0)?-1.f:0.f);
        for (int j=0;j<9;j++) {
            int a=j/3, bb=j%3;
            float dX[9];
            for (int r=0;r<3;r++) for (int c=0;c<3;c++) dX[r*3+c] = (r==a) ? Fb[bb*3+c] : 0.f;
            float Q[9], dQ[9], C[9], dC[9];
            for (int k=0;k<9;k++){ Q[k]=X[k]; dQ[k]=dX[k]; }
            for (int it=0; it<10; it++) {
                cof3(Q, C); coflin(Q, dQ, dC);
                float dt = det3(Q);
                float ddt=0; for (int k=0;k<9;k++) ddt += C[k]*dQ[k];
                bool big = fabsf(dt) > 1e-30f;
                float ds = big ? dt : 1e-30f, dds = big ? ddt : 0.f;
                float gm = powf(fabsf(ds), -1.f/3.f);
                float dgm = -(1.f/3.f)*gm*dds/ds;
                float sgm = ds*gm, dsg = dds*gm + ds*dgm;
                for (int k=0;k<9;k++) {
                    float qn = 0.5f*(gm*Q[k] + C[k]/sgm);
                    dQ[k] = 0.5f*(dgm*Q[k] + gm*dQ[k] + dC[k]/sgm - C[k]*(dsg/(sgm*sgm)));
                    Q[k] = qn;
                }
            }
            float dcf[9]; coflin(X, dX, dcf);
            float ddet=0; for (int k=0;k<9;k++) ddet += cf[k]*dX[k];
            float dP[9];
            for (int k=0;k<9;k++) dP[k] = 2.f*mu*(dX[k]-dQ[k]) + lam*ddet*cf[k] + lam*(det-sg)*dcf[k];
            float dG[9]; mmt3(dP, Fb, dG);
            for (int k=0;k<9;k++) K[k*9+j] = V*dG[k];
        }
    }
    // H12[(i,c),(j,d)] = M4_ij δ_cd + Σ_{b,e} ∇λ_ib K[(c,b),(d,e)] ∇λ_je
    float H[144], W[144];
    const float* m4 = M4 + t*16;
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float KB[3][4];   // Σ_e K[(c,b),(d,e)] ∇λ_je  -> 그 다음 b 로 합
        for (int j=0;j<4;j++) for (int d=0;d<3;d++) {
            float s=0;
            for (int b=0;b<3;b++) { float kb=0; for (int e2=0;e2<3;e2++) kb += K[(c*3+b)*9 + d*3+e2]*G[j*3+e2]; s += G[i*3+b]*kb; }
            H[(i*3+c)*12 + j*3+d] = s + ((c==d) ? m4[i*4+j] : 0.f);
        }
    }
    for (int p=0;p<12;p++) for (int q=p+1;q<12;q++) { float s=0.5f*(H[p*12+q]+H[q*12+p]); H[p*12+q]=s; H[q*12+p]=s; }
    for (int k=0;k<144;k++) W[k] = (k%13==0) ? 1.f : 0.f;
    float fro=0; for (int k=0;k<144;k++) fro += H[k]*H[k];
    float skip = sqrtf(jtol*fro)*1e-3f;
    for (int sw=0; sw<maxsw; sw++) {
        float off=0; for (int p=0;p<12;p++) for (int q=p+1;q<12;q++) off += H[p*12+q]*H[p*12+q];
        if (off <= jtol*fro) break;
        for (int p=0;p<11;p++) for (int q=p+1;q<12;q++) {
            float apq = H[p*12+q];
            if (fabsf(apq) <= skip) continue;
            float th = 0.5f*(H[q*12+q]-H[p*12+p])/apq;
            float tt = 1.f/(fabsf(th)+sqrtf(1.f+th*th)); if (th<0) tt=-tt;
            float c = 1.f/sqrtf(1.f+tt*tt), s = tt*c, tau = s/(1.f+c);
            H[p*12+p] -= tt*apq; H[q*12+q] += tt*apq; H[p*12+q]=0.f; H[q*12+p]=0.f;
            for (int j=0;j<12;j++) if (j!=p && j!=q) {
                float gg=H[j*12+p], hh=H[j*12+q];
                float np_=gg-s*(hh+gg*tau), nq=hh+s*(gg-hh*tau);
                H[j*12+p]=np_; H[p*12+j]=np_; H[j*12+q]=nq; H[q*12+j]=nq;
            }
            for (int j=0;j<12;j++) { float gg=W[j*12+p], hh=W[j*12+q]; W[j*12+p]=gg-s*(hh+gg*tau); W[j*12+q]=hh+s*(gg-hh*tau); }
        }
    }
    float ev[12], mx=0; int neg=0;
    for (int k=0;k<12;k++){ ev[k]=H[k*13]; if (ev[k]<0) neg=1; mx=fmaxf(mx,fabsf(ev[k])); }
    if (neg) atomicAdd(nneg, 1);
    for (int k=0;k<12;k++) ev[k] = fmaxf(fabsf(ev[k]), efloor*mx);
    const int* sl = slots + t*16;
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) {
        float* Bo = Bell + (long long)sl[i*4+j]*9;
        for (int c=0;c<3;c++) for (int d=0;d<3;d++) {
            int r=i*3+c, q=j*3+d; float s=0;
            for (int k=0;k<12;k++) s += W[r*12+k]*ev[k]*W[q*12+k];
            atomicAdd(&Bo[c*3+d], s);
        }
    }
}

// ---------------- PCG (블록 ELL float32 행렬, float64 벡터; 상태 st: 0 rz 1 dAd 2 rr 3 rz2 4 thr 5 alive 6 nit 7 neg 8 alpha 9 beta) ----------------
__global__ void pcg_spmv(const int* Mp, const float* B, const int* nb, const double* d, double* y, double* st) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    int M = *Mp;
    double part = 0.0;
    if (m < M && st[5] > 0.5) {
        double y0=0,y1=0,y2=0;
        for (int k=0;k<15;k++) {
            int j = nb[m*15+k];
            const float* b = B + ((long long)m*15+k)*9;
            double x0=d[j*3], x1=d[j*3+1], x2=d[j*3+2];
            y0 += b[0]*x0 + b[1]*x1 + b[2]*x2;
            y1 += b[3]*x0 + b[4]*x1 + b[5]*x2;
            y2 += b[6]*x0 + b[7]*x1 + b[8]*x2;
        }
        y[m*3]=y0; y[m*3+1]=y1; y[m*3+2]=y2;
        part = d[m*3]*y0 + d[m*3+1]*y1 + d[m*3+2]*y2;
    }
    block_add(part, &st[1]);
}
// 2 커널 PCG: scal[2][4] = (rz, alive, nit, neg) 패리티별, acc[2][3] = (dAd, rr, rz2) 패리티별
__global__ void pcg2_b(int p, const int* Mp, const double* Dinv, double* x, double* r, double* z, const double* d, const double* Ad,
                       const double* scal, double* acc) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    int M = *Mp;
    const double* S = scal + p*4; double* A = acc + p*3;
    double dAd = A[0];
    bool ae = (S[1] > 0.5) && (dAd > 0);
    double al = ae ? S[0]/dAd : 0.0;
    double prr=0, prz=0;
    if (m < M && ae) {
        double rr[3];
        for (int c=0;c<3;c++){ x[m*3+c] += al*d[m*3+c]; r[m*3+c] -= al*Ad[m*3+c]; rr[c]=r[m*3+c]; }
        const double* D = Dinv + m*9;
        for (int c=0;c<3;c++){ double zz = D[c*3]*rr[0]+D[c*3+1]*rr[1]+D[c*3+2]*rr[2]; z[m*3+c]=zz; prz += rr[c]*zz; prr += rr[c]*rr[c]; }
    }
    if (blockIdx.x==0 && threadIdx.x==0) { double* O = acc + (1-p)*3; O[0]=0; O[1]=0; O[2]=0; }
    block_add(prr, &A[1]);
    block_add(prz, &A[2]);
}
__global__ void pcg2_d(int p, const int* Mp, const float* B, const int* nb, const double* z, double* d, double* Ad,
                       double* scal, double* acc, double thr) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    int M = *Mp;
    const double* S = scal + p*4; const double* A = acc + p*3;
    bool alive = S[1] > 0.5;
    bool ae = alive && (A[0] > 0);
    bool an = ae && !(A[1] < thr);
    double be = an ? A[2]/S[0] : 0.0;
    double part = 0.0;
    if (m < M && an) {
        double y0=0,y1=0,y2=0;
        for (int k=0;k<15;k++) {
            int j = nb[m*15+k];
            const float* b = B + ((long long)m*15+k)*9;
            double x0=z[j*3], x1=z[j*3+1], x2=z[j*3+2];
            y0 += b[0]*x0 + b[1]*x1 + b[2]*x2;
            y1 += b[3]*x0 + b[4]*x1 + b[5]*x2;
            y2 += b[6]*x0 + b[7]*x1 + b[8]*x2;
        }
        double q[3] = {y0,y1,y2};
        for (int c=0;c<3;c++){ double dd = z[m*3+c] + be*d[m*3+c]; double aa = q[c] + be*Ad[m*3+c]; d[m*3+c]=dd; Ad[m*3+c]=aa; part += dd*aa; }
    }
    if (blockIdx.x==0 && threadIdx.x==0) {
        double* O = scal + (1-p)*4;
        O[0] = an ? A[2] : S[0]; O[1] = an ? 1.0 : 0.0; O[2] = S[2] + (ae ? 1.0 : 0.0); O[3] = (S[3] > 0.5 || (alive && !(A[0] > 0))) ? 1.0 : 0.0;
    }
    block_add(part, &acc[(1-p)*3]);
}
__global__ void bell_mv(const int* Mp, const float* B, const int* nb, const double* v, double* y) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= *Mp) return;
    double y0=0,y1=0,y2=0;
    for (int k=0;k<15;k++) {
        int j = nb[m*15+k];
        const float* b = B + ((long long)m*15+k)*9;
        double x0=v[j*3], x1=v[j*3+1], x2=v[j*3+2];
        y0 += b[0]*x0 + b[1]*x1 + b[2]*x2; y1 += b[3]*x0 + b[4]*x1 + b[5]*x2; y2 += b[6]*x0 + b[7]*x1 + b[8]*x2;
    }
    y[m*3]=y0; y[m*3+1]=y1; y[m*3+2]=y2;
}
__global__ void pcg_s1(double* st) {
    if (st[5] > 0.5) {
        if (!(st[1] > 0)) { st[7]=1; st[5]=0; st[8]=0; }
        else { st[8] = st[0]/st[1]; st[2]=0; st[3]=0; }
    } else st[8] = 0;
}
__global__ void pcg_upd(const int* Mp, const double* Dinv, double* x, double* r, double* z, const double* d, const double* y, double* st) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    int M = *Mp;
    double prr=0, prz=0;
    if (m < M && st[5] > 0.5) {
        double al = st[8];
        double rr[3];
        for (int c=0;c<3;c++){ x[m*3+c] += al*d[m*3+c]; r[m*3+c] -= al*y[m*3+c]; rr[c]=r[m*3+c]; }
        const double* D = Dinv + m*9;
        for (int c=0;c<3;c++){ double zz = D[c*3]*rr[0]+D[c*3+1]*rr[1]+D[c*3+2]*rr[2]; z[m*3+c]=zz; prz += rr[c]*zz; prr += rr[c]*rr[c]; }
    }
    block_add(prr, &st[2]);
    block_add(prz, &st[3]);
}
__global__ void pcg_s2(double* st) {
    if (st[5] > 0.5) {
        st[6] += 1;
        if (st[2] < st[4]) st[5] = 0;
    }
    if (st[5] > 0.5) { st[9] = st[3]/st[0]; st[0] = st[3]; } else st[9] = 0;
    st[1] = 0;
}
__global__ void pcg_dir(const int* Mp, double* d, const double* z, const double* st) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k < (*Mp)*3 && st[5] > 0.5) d[k] = z[k] + st[9]*d[k];
}
}
'''

_MOD = [None]


def _mod():
    if _MOD[0] is None:
        import cupy as cp
        _MOD[0] = cp.RawModule(code=_SRC.encode("ascii", "ignore").decode(), options=("--std=c++14",))
    return _MOD[0]


def _cp(t):
    import cupy as cp
    return cp.asarray(t)


class _Stream:
    """torch 지금 스트림에 CuPy 커널을 띄운다."""
    def __enter__(self):
        import cupy as cp
        self.s = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        self.s.__enter__()
        return self

    def __exit__(self, *e):
        self.s.__exit__(*e)


def _grid(n, b=128):
    return ((int(n) + b - 1) // b,), (b,)


class CudaFem:
    """프레임마다 NodeFrame 의 사면체 자료로 커널 입력을 만든다 (float64 E·기울기용, float32 헤시안용)."""

    def __init__(self, fr):
        self.fr = fr
        G = fr.G
        self.T = G.T
        self.tets = G.tets.contiguous()
        self.gl8 = fr.gl.contiguous()
        self.FT8 = fr.FT.contiguous()
        self.VT8 = fr.VT.contiguous()
        self.M48 = fr.M4.contiguous()
        self.B48 = fr.B4.contiguous()
        self.gl4, self.FT4, self.VT4, self.M44 = self.gl8.float(), self.FT8.float(), self.VT8.float(), self.M48.float()
        self.mu, self.lam = float(fr.mu), float(fr.lam_)
        self.dev = G.tets.device
        self.cnt = torch.zeros(2, dtype=torch.int32, device=self.dev)

    def energy_grad(self, U, want_g=True):
        M = self.fr.G.M
        E = torch.zeros(1, dtype=torch.float64, device=self.dev)
        g = torch.zeros(M, 3, dtype=torch.float64, device=self.dev)
        Uf = U.float().contiguous()
        k = _mod().get_function("tet_eg")
        gr, bl = _grid(self.T)
        with _Stream():
            k(gr, bl, (np.int32(self.T), _cp(self.tets), _cp(Uf), _cp(self.gl8), _cp(self.FT8), _cp(self.VT8), _cp(self.M48), _cp(self.B48),
                       np.float64(self.mu), np.float64(self.lam), np.int32(1 if want_g else 0), _cp(E), _cp(g), _cp(self.cnt[0:1])))
        Ef, gf = self.fr._floor_eg(U.double())
        return E[0] + self.fr.C0 + Ef, g + gf

    def energy(self, U):
        return self.energy_grad(U, want_g=False)[0]

    def hess_into(self, U, slots, Bell, efloor, maxsw=10, jtol=1e-12):
        k = _mod().get_function("tet_hess")
        gr, bl = _grid(self.T, 64)
        Uf = U.float().contiguous()
        with _Stream():
            k(gr, bl, (np.int32(self.T), _cp(self.tets), _cp(Uf), _cp(self.gl4), _cp(self.FT4), _cp(self.VT4), _cp(self.M44),
                       np.float32(self.mu), np.float32(self.lam), np.float32(efloor), _cp(slots), _cp(Bell), _cp(self.cnt[1:2]),
                       np.int32(maxsw), np.float32(jtol)))


class CudaPcg:
    """미리 잡은 최대 크기 버퍼 위의 PCG. 반복 ck 회를 CUDA 그래프 하나로 잡아 (처음 한 번) 프레임 넘어 재생."""

    def __init__(self, Mmax, dev, ck=10):
        self.Mmax, self.ck, self.dev = int(Mmax), int(ck), dev
        f8 = torch.float64
        self.B = torch.zeros(self.Mmax * 15 * 9, dtype=torch.float32, device=dev)
        self.nb = torch.zeros(self.Mmax * 15, dtype=torch.int32, device=dev)
        self.Dinv = torch.zeros(self.Mmax * 9, dtype=f8, device=dev)
        self.x, self.r, self.z, self.d, self.y = [torch.zeros(self.Mmax * 3, dtype=f8, device=dev) for _ in range(5)]
        self.st = torch.zeros(10, dtype=f8, device=dev)
        self.Mp = torch.zeros(1, dtype=torch.int32, device=dev)
        self.graph = None

    def _iter(self):
        md = _mod()
        g1, b1 = _grid(self.Mmax)
        g3, b3 = _grid(self.Mmax * 3)
        md.get_function("pcg_spmv")(g1, b1, (self.cMp, self.cB, self.cnb, self.cd, self.cy, self.cst))
        md.get_function("pcg_s1")((1,), (1,), (self.cst,))
        md.get_function("pcg_upd")(g1, b1, (self.cMp, self.cDinv, self.cx, self.cr, self.cz, self.cd, self.cy, self.cst))
        md.get_function("pcg_s2")((1,), (1,), (self.cst,))
        md.get_function("pcg_dir")(g3, b3, (self.cMp, self.cd, self.cz, self.cst))

    def solve2(self, M, b, rtol, ncg, use_graph=True):
        """2 커널 PCG (pcg2_b: x·r·z 갱신 + rr·rz 합, pcg2_d: A z 곱 + 방향 d·Ad 갱신 + dAd 합; 스칼라 판정은 각 스레드가 같은 값으로).
        같은 식 (Ad = A z + β Ad 로 곱 한 번), 같은 판정 |r|² < rtol²|b|²."""
        n = 3 * M
        self.Mp.fill_(M)
        if not hasattr(self, "Ad"):
            self.Ad = torch.zeros(self.Mmax * 3, dtype=torch.float64, device=self.dev)
            self.scal = torch.zeros(8, dtype=torch.float64, device=self.dev)
            self.acc = torch.zeros(6, dtype=torch.float64, device=self.dev)
            self.thr = torch.zeros(1, dtype=torch.float64, device=self.dev)
        if not hasattr(self, "cB"):
            self.cB, self.cnb, self.cDinv, self.cx, self.cr, self.cz, self.cd, self.cy, self.cst, self.cMp = \
                [_cp(q) for q in (self.B, self.nb, self.Dinv, self.x, self.r, self.z, self.d, self.y, self.st, self.Mp)]
        if not hasattr(self, "cAd"):
            self.cAd, self.cscal, self.cacc = _cp(self.Ad), _cp(self.scal), _cp(self.acc)
        self.x.zero_()
        self.r.zero_()
        self.r[:n].copy_(b)
        Di = self.Dinv[:M * 9].view(M, 3, 3)
        z = (Di @ b.view(M, 3, 1)).view(-1)
        self.z.zero_()
        self.z[:n].copy_(z)
        self.d.copy_(self.z)
        g1, b1 = _grid(self.Mmax)
        with _Stream():
            _mod().get_function("bell_mv")(g1, b1, (self.cMp, self.cB, self.cnb, self.cd, self.cAd))
        bb = (b * b).sum().clamp_min(1e-300)
        thr = float(bb) * rtol * rtol
        self.scal.zero_()
        self.acc.zero_()
        self.scal[0] = (b * z).sum()
        self.scal[1] = 1.0
        self.acc[0] = (self.d[:n] * self.Ad[:n]).sum()
        md = _mod()
        kb, kd = md.get_function("pcg2_b"), md.get_function("pcg2_d")

        def it2(p, thr_):
            kb(g1, b1, (np.int32(p), self.cMp, self.cDinv, self.cx, self.cr, self.cz, self.cd, self.cAd, self.cscal, self.cacc))
            kd(g1, b1, (np.int32(p), self.cMp, self.cB, self.cnb, self.cz, self.cd, self.cAd, self.cscal, self.cacc, np.float64(thr_)))
        ck = self.ck - (self.ck % 2)
        it = 0
        if use_graph:
            # 판정 문턱은 프레임마다 다르다 -- 장치 값으로 넘기면 좋지만 커널 인자로 박혀 있어 그래프를 문턱마다 다시 잡지 않으려고
            # 문턱 대신 b 를 |b| 로 나눠 푼다 (해 x 도 |b| 배로 되돌린다; 같은 판정 rr/|b|² < rtol²)
            pass
        sc = float(bb.sqrt())
        self.r[:n].div_(sc); self.z[:n].div_(sc); self.d[:n].div_(sc); self.Ad[:n].div_(sc)
        self.scal[0] /= sc * sc
        self.acc[0] /= sc * sc
        thr1 = rtol * rtol
        if use_graph and getattr(self, "graph2", None) is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s), _Stream():
                gr.capture_begin()
                for k in range(ck):
                    it2(k % 2, thr1)
                gr.capture_end()
            torch.cuda.current_stream().wait_stream(s)
            self.graph2 = gr
        while it < ncg:
            if use_graph:
                self.graph2.replay()
            else:
                with _Stream():
                    for k in range(ck):
                        it2(k % 2, thr1)
            it += ck
            if float(self.scal[1]) < 0.5:
                break
        x = self.x[:n] * sc
        rr = float((self.r[:n] * self.r[:n]).sum())
        return x, int(self.scal[2]), rr ** 0.5, bool(self.scal[3] > 0.5)

    def solve(self, M, b, rtol, ncg, use_graph=True):
        """B·nb·Dinv 는 호출 전에 채워 둔다 (앞 M 노드). b [3M] float64. -> (x [3M], 반복, |r|/|b|, 음곡률)."""
        n = 3 * M
        self.Mp.fill_(M)
        self.x.zero_()
        self.r.zero_()
        self.r[:n].copy_(b)
        Di = self.Dinv[:M * 9].view(M, 3, 3)
        z = (Di @ b.view(M, 3, 1)).view(-1)
        self.z.zero_()
        self.z[:n].copy_(z)
        self.d.copy_(self.z)
        rz = (b * z).sum()
        bb = (b * b).sum().clamp_min(1e-300)
        self.st.zero_()
        self.st[0] = rz
        self.st[4] = bb * rtol * rtol
        self.st[5] = 1.0
        if not hasattr(self, "cB"):
            self.cB, self.cnb, self.cDinv, self.cx, self.cr, self.cz, self.cd, self.cy, self.cst, self.cMp = \
                [_cp(q) for q in (self.B, self.nb, self.Dinv, self.x, self.r, self.z, self.d, self.y, self.st, self.Mp)]
        it = 0
        if use_graph and self.graph is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s), _Stream():
                gr.capture_begin()
                for _ in range(self.ck):
                    self._iter()
                gr.capture_end()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = gr
        while it < ncg:
            if use_graph:
                self.graph.replay()
            else:
                with _Stream():
                    for _ in range(self.ck):
                        self._iter()
            it += self.ck
            if float(self.st[5]) < 0.5:
                break
        return self.x[:n].clone(), int(self.st[6]), float((self.st[2] / bb).sqrt()), bool(self.st[7] > 0.5)
