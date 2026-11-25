//@@ _SRC 87574

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


// K = V d(P(AF) F^T)/dA (9x9, K[k*9+j] = d gA_k / d A_j) -- fem_fast._k_manual (Newton polar 10 iters forward-mode), smin<=0.01 -> 0
__device__ void kmanual(const float* X, const float* Fb, float mu, float lam, float V, float smin, float* K) {
    if (smin <= 0.01f || V == 0.f) { for (int k=0;k<81;k++) K[k]=0.f; return; }
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



template <typename V> __device__ __forceinline__ float ell_row32(int m, int c, int Ms, const float* B, const int* nb, const float* v) {
    float y = 0.f;
    for (int k=0;k<15;k++) {
        int j = nb[k*Ms+m];
        const float* bb = B + ((long long)k*9 + c*3)*Ms + m;
        y += bb[0]*__ldcg(&v[j*3]) + bb[Ms]*__ldcg(&v[j*3+1]) + bb[2*Ms]*__ldcg(&v[j*3+2]);
    }
    return y;
}

template <bool CG> __device__ __forceinline__ double ell_row_sym(int m, int c, int Ms, const float* B, const int* nb, const double* v) {
    double y = 0;
    for (int k=0;k<8;k++) {
        int j = nb[k*Ms+m]; if (j < 0) continue;
        const float* bb = B + ((long long)k*9 + c*3)*Ms + m;
        double x0, x1, x2;
        if (CG) { x0=__ldcg(&v[j*3]); x1=__ldcg(&v[j*3+1]); x2=__ldcg(&v[j*3+2]); } else { x0=v[j*3]; x1=v[j*3+1]; x2=v[j*3+2]; }
        y += bb[0]*x0 + bb[Ms]*x1 + bb[2*Ms]*x2;
    }
    for (int k=1;k<8;k++) {
        int j = nb[(7+k)*Ms+m]; if (j < 0) continue;
        const float* bb = B + ((long long)k*9 + c)*Ms + j;      // (B_k at node j)^T row c = column c
        double x0, x1, x2;
        if (CG) { x0=__ldcg(&v[j*3]); x1=__ldcg(&v[j*3+1]); x2=__ldcg(&v[j*3+2]); } else { x0=v[j*3]; x1=v[j*3+1]; x2=v[j*3+2]; }
        y += bb[0]*x0 + bb[3*(long long)Ms]*x1 + bb[6*(long long)Ms]*x2;
    }
    return y;
}
template <bool CG> __device__ __forceinline__ double ell_row_sym64(int m, int c, int Ms, const double* B, const int* nb, const double* v) {
    double y = 0;
    for (int k=0;k<8;k++) {
        int j = nb[k*Ms+m]; if (j < 0) continue;
        const double* bb = B + ((long long)k*9 + c*3)*Ms + m;
        double x0, x1, x2;
        if (CG) { x0=__ldcg(&v[j*3]); x1=__ldcg(&v[j*3+1]); x2=__ldcg(&v[j*3+2]); } else { x0=v[j*3]; x1=v[j*3+1]; x2=v[j*3+2]; }
        y += bb[0]*x0 + bb[Ms]*x1 + bb[2*Ms]*x2;
    }
    for (int k=1;k<8;k++) {
        int j = nb[(7+k)*Ms+m]; if (j < 0) continue;
        const double* bb = B + ((long long)k*9 + c)*Ms + j;      // (B_k at node j)^T row c = column c
        double x0, x1, x2;
        if (CG) { x0=__ldcg(&v[j*3]); x1=__ldcg(&v[j*3+1]); x2=__ldcg(&v[j*3+2]); } else { x0=v[j*3]; x1=v[j*3+1]; x2=v[j*3+2]; }
        y += bb[0]*x0 + bb[3*(long long)Ms]*x1 + bb[6*(long long)Ms]*x2;
    }
    return y;
}
extern "C" {
// ---------------- warp per tet: same as tet_hess (K, 12x12, |lambda| projection, block-ELL atomics), parallel round-robin Jacobi in shared memory ----------------
__global__ void tet_hess_w(int T, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                           const float* M4, float mu, float lam, float efloor, const int* slots, float* Bell, int* nneg, int maxsw, float jtol,
                           const float* Ht, int Ms, int sym) {
    __shared__ float sH[4][12][13], sV[4][12][13], sK[4][81], sR[4][6][2], sL[4][12], sG[4][12], sF[4][9], sM[4][16];
    int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int t = blockIdx.x*4 + w;
    if (t >= T) return;   // whole warp exits together
    float (*H)[13] = sH[w]; float (*V)[13] = sV[w]; float* K = sK[w];
    if (lane < 12) sG[w][lane] = gl[t*12+lane];
    if (lane < 9) sF[w][lane] = FT[t*9+lane];
    if (lane < 16) sM[w][lane] = M4[t*16+lane];
    __syncwarp();
    float A[9];
    for (int c=0;c<3;c++) for (int b=0;b<3;b++) {
        float sacc=(c==b)?1.f:0.f;
        for (int i=0;i<4;i++) sacc += U[tets[t*4+i]*3+c]*sG[w][i*3+b];
        A[c*3+b]=sacc;
    }
    float X[9]; mm3(A, sF[w], X);
    float V_ = VT[t];
    // K: lane j < 9 computes column j
    if (lane < 9) {
        float Cx[9]; for (int i=0;i<3;i++) for (int k=0;k<3;k++) Cx[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
        float l[3]; symeig3(Cx, l);
        float smin = sqrtf(fmaxf(l[0],1e-12f));
        int j = lane;
        if (smin <= 0.01f || V_ == 0.f) { for (int k=0;k<9;k++) K[k*9+j]=0.f; }
        else {
            const float* Fb = sF[w];
            float cf[9]; cof3(X, cf);
            float det = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
            float sg = (det>0)?1.f:((det<0)?-1.f:0.f);
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
            for (int k=0;k<9;k++) K[k*9+j] = V_*dG[k];
        }
    }
    __syncwarp();
    const float* G = sG[w];
    for (int e=lane; e<144; e+=32) {
        int r = e/12, q = e%12; int i=r/3, c=r%3, j=q/3, d=q%3;
        float sacc=0;
        for (int b=0;b<3;b++) { float kb=0; for (int e2=0;e2<3;e2++) kb += K[(c*3+b)*9 + d*3+e2]*G[j*3+e2]; sacc += G[i*3+b]*kb; }
        sacc += (c==d) ? sM[w][i*4+j] : 0.f;
        if (Ht != 0) sacc += Ht[(long long)t*144 + e];
        H[r][q] = sacc;
        V[r][q] = (r==q) ? 1.f : 0.f;
    }
    __syncwarp();
    for (int e=lane; e<144; e+=32) { int r=e/12, q=e%12; if (r<q) { float sv=0.5f*(H[r][q]+H[q][r]); H[r][q]=sv; H[q][r]=sv; } }
    __syncwarp();
    float fro=0; for (int e=lane; e<144; e+=32) { float hv=H[e/12][e%12]; fro += hv*hv; }
    for (int o=16;o>0;o>>=1) fro += __shfl_xor_sync(0xffffffff, fro, o);
    for (int sw=0; sw<maxsw; sw++) {
        float off=0; for (int e=lane; e<144; e+=32) { int r=e/12, q=e%12; if (r<q) off += H[r][q]*H[r][q]; }
        for (int o=16;o>0;o>>=1) off += __shfl_xor_sync(0xffffffff, off, o);
        if (off <= jtol*fro) break;
        for (int rd=0; rd<11; rd++) {
            // circle method: player 11 fixed, others rotate
            if (lane < 6) {
                int pa, pb;
                if (lane == 0) { pa = 11; pb = rd; } else { pa = (rd + lane) % 11; pb = (rd + 11 - lane) % 11; }
                int p_ = min(pa,pb), q_ = max(pa,pb);
                float apq = H[p_][q_];
                float cc=1.f, ss=0.f;
                if (fabsf(apq) > 1e-30f) {
                    float th = 0.5f*(H[q_][q_]-H[p_][p_])/apq;
                    float tt = 1.f/(fabsf(th)+sqrtf(1.f+th*th)); if (th<0) tt=-tt;
                    cc = 1.f/sqrtf(1.f+tt*tt); ss = tt*cc;
                }
                sR[w][lane][0]=cc; sR[w][lane][1]=ss;
                sL[w][lane*2]=(float)p_; sL[w][lane*2+1]=(float)q_;
            }
            __syncwarp();
            // A <- A J (columns p,q) and V <- V J
            for (int e=lane; e<144; e+=32) {
                int pr = e/24, rem = e%24, row = rem%12; bool isV = rem >= 12;
                int p_ = (int)sL[w][pr*2], q_ = (int)sL[w][pr*2+1]; float cc=sR[w][pr][0], ss=sR[w][pr][1];
                if (!isV) { float ap=H[row][p_], aq=H[row][q_]; H[row][p_]=cc*ap-ss*aq; H[row][q_]=ss*ap+cc*aq; }
                else { float ap=V[row][p_], aq=V[row][q_]; V[row][p_]=cc*ap-ss*aq; V[row][q_]=ss*ap+cc*aq; }
            }
            __syncwarp();
            // A <- J^T A (rows p,q)
            for (int e=lane; e<72; e+=32) {
                int pr = e/12, col = e%12;
                int p_ = (int)sL[w][pr*2], q_ = (int)sL[w][pr*2+1]; float cc=sR[w][pr][0], ss=sR[w][pr][1];
                float ap=H[p_][col], aq=H[q_][col]; H[p_][col]=cc*ap-ss*aq; H[q_][col]=ss*ap+cc*aq;
            }
            __syncwarp();
        }
    }
    float ev = (lane < 12) ? H[lane][lane] : 0.f;
    int neg = __any_sync(0xffffffff, lane < 12 && ev < 0.f);
    float mx = fabsf(ev); for (int o=16;o>0;o>>=1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    if (lane < 12) sL[w][lane] = fmaxf(fabsf(ev), efloor*mx);
    if (lane == 0 && neg) atomicAdd(nneg, 1);
    __syncwarp();
    const int* sl = slots + t*16;
    for (int e=lane; e<144; e+=32) {
        int r=e/12, q=e%12; float sacc=0;
        for (int k=0;k<12;k++) sacc += V[r][k]*sL[w][k]*V[q][k];
        int i=r/3, c=r%3, j=q/3, d=q%3;
        int sv = sl[i*4+j]; long long node = sv/15, cls = sv%15;
        if (sym && cls >= 8) continue;
        atomicAdd(&Bell[((long long)cls*9 + c*3+d)*Ms + node], sacc);
    }
}

}
// ---------------- bottom-candidate particle (stick -> softplus maps), fem_fast._near_core closed form ----------------
template <typename R> struct NearMapT {
    R s, w, w1, w2, w3, g, g1, g2, sp;
    R n[3], a[3], r[3];
    R Nm[9], Pm[9], IG[9], Jst[9], K1[9], Mn[9], Ru[9];
};
template <typename R> __device__ void near_map(const R* x, const R* u, const R* G, const R* c, const R* pt, const R* nv,
                         R eps_st, R eps_sp, R xn, NearMapT<R>& M) {
    for (int k=0;k<3;k++) M.n[k]=nv[k];
    R s = 0; for (int k=0;k<3;k++) s += (x[k]+u[k]-pt[k])*nv[k];
    M.s = s;
    if (eps_st > 0) {
        R e=eps_st; bool far = s/e > R(20); R sg = R(1)/(R(1)+exp(-s/e));
        M.w = far?R(1):sg; M.w1 = far?R(0):sg*(1-sg)/e; M.w2 = far?R(0):sg*(1-sg)*(1-2*sg)/(e*e); M.w3 = far?R(0):sg*(1-sg)*(1-6*sg+6*sg*sg)/(e*e*e);
    } else { M.w=1; M.w1=0; M.w2=0; M.w3=0; }
    if (eps_sp > 0) {
        R e=eps_sp; bool far = s/e > R(20); R sg = R(1)/(R(1)+exp(-s/e));
        M.g = far?R(1):sg; M.g1 = far?R(0):sg*(1-sg)/e; M.g2 = far?R(0):sg*(1-sg)*(1-2*sg)/(e*e);
        M.sp = far ? s : e*log1p(exp(s/e));
    } else { M.g=1; M.g1=0; M.g2=0; M.sp=s; }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) { M.Nm[i*3+j]=nv[i]*nv[j]; M.Pm[i*3+j]=((i==j)?R(1):R(0))-nv[i]*nv[j]; M.IG[i*3+j]=((i==j)?R(1):R(0))+G[i*3+j]; }
    for (int i=0;i<3;i++){ R t=0; for (int j=0;j<3;j++) t += u[j]*M.Pm[j*3+i]; M.a[i]=t; }
    for (int i=0;i<3;i++) M.r[i] = u[i] - (1-M.w)*M.a[i] + (M.sp-s)*nv[i] - c[i];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        R I_ = (i==j)?R(1):R(0);
        M.Ru[i*3+j] = I_ - (1-M.w)*M.Pm[i*3+j] + M.w1*M.a[i]*nv[j] + (M.g-1)*M.Nm[i*3+j];
        M.Jst[i*3+j] = M.Nm[i*3+j] + M.w*M.Pm[i*3+j] + M.w1*M.a[i]*nv[j];
        M.Mn[i*3+j] = I_ + (M.g-1)*M.Nm[i*3+j];
    }
    R T[9]; mm3(M.Jst, M.IG, T);
    for (int k=0;k<9;k++) M.K1[k] = T[k] + xn*(1-M.w)*M.Pm[k];
}
// Bc[c] = w1 n_c P + w2 n_c a n^T + w1 P[:,c] n^T ; dK1[c] = Bc[c] IG - xn w1 n_c P
template <typename R> __device__ void near_Bc(const NearMapT<R>& M, int c, R* Bc) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++)
        Bc[i*3+j] = M.w1*M.n[c]*M.Pm[i*3+j] + M.w2*M.n[c]*M.a[i]*M.n[j] + M.w1*M.Pm[i*3+c]*M.n[j];
}
typedef NearMapT<double> NearMap;

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

// 3 threads per node (row c), node index contiguous across threads (coalesced SoA reads)
template <bool CG> __device__ __forceinline__ double ell_row(int m, int c, int Ms, const float* B, const int* nb, const double* v) {
    double y = 0;
    #pragma unroll 5
    for (int k=0;k<15;k++) {
        int j = nb[k*Ms+m];
        const float* bb = B + ((long long)k*9 + c*3)*Ms + m;
        double x0, x1, x2;
        if (CG) { x0=__ldcg(&v[j*3]); x1=__ldcg(&v[j*3+1]); x2=__ldcg(&v[j*3+2]); } else { x0=v[j*3]; x1=v[j*3+1]; x2=v[j*3+2]; }
        y += bb[0]*x0 + bb[Ms]*x1 + bb[2*Ms]*x2;
    }
    return y;
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
                         const float* M4, float mu, float lam, float efloor, const int* slots, float* Bell, int* nneg, int maxsw, float jtol, const float* Ht, int Ms, float gtau, int* ngate) {
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
    kmanual(X, Fb, mu, lam, V, smin, K);
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
    if (Ht != 0) { const float* ht = Ht + (long long)t*144; for (int k=0;k<144;k++) H[k] += ht[k]; }
    for (int p=0;p<12;p++) for (int q=p+1;q<12;q++) { float s=0.5f*(H[p*12+q]+H[q*12+p]); H[p*12+q]=s; H[q*12+p]=s; }
    float fro=0; for (int k=0;k<144;k++) fro += H[k]*H[k];
    const int* sl = slots + t*16;
    if (gtau > 0.f) {
        // Cholesky gate: if H + tau*|H|_F*I is SPD, use it as is (|lambda| projection differs only for eigenvalues within tau*|H|_F of 0)
        float sh = gtau*sqrtf(fro);
        for (int k=0;k<144;k++) W[k] = H[k];
        for (int k=0;k<12;k++) W[k*13] += sh;
        bool ok = true;
        for (int j=0;j<12 && ok;j++) {
            float dsum = W[j*13];
            for (int k=0;k<j;k++) dsum -= W[j*12+k]*W[j*12+k];
            if (!(dsum > 0.f)) { ok = false; break; }
            float ljj = sqrtf(dsum); W[j*13] = ljj;
            for (int i=j+1;i<12;i++) { float s2 = W[i*12+j]; for (int k=0;k<j;k++) s2 -= W[i*12+k]*W[j*12+k]; W[i*12+j] = s2/ljj; }
        }
        if (ok) {
            atomicAdd(ngate, 1);
            for (int i=0;i<4;i++) for (int j=0;j<4;j++) {
                int sv = sl[i*4+j]; long long node = sv/15, cls = sv%15;
                for (int c=0;c<3;c++) for (int d=0;d<3;d++) {
                    int r=i*3+c, q=j*3+d;
                    atomicAdd(&Bell[((long long)cls*9 + c*3+d)*Ms + node], H[r*12+q] + ((r==q)?sh:0.f));
                }
            }
            return;
        }
    }
    for (int k=0;k<144;k++) W[k] = (k%13==0) ? 1.f : 0.f;
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
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) {
        int sv = sl[i*4+j]; long long node = sv/15, cls = sv%15;
        for (int c=0;c<3;c++) for (int d=0;d<3;d++) {
            int r=i*3+c, q=j*3+d; float s=0;
            for (int k=0;k<12;k++) s += W[r*12+k]*ev[k]*W[q*12+k];
            atomicAdd(&Bell[((long long)cls*9 + c*3+d)*Ms + node], s);
        }
    }
}


// E, grad for bottom-candidate particles (float64). gl: per-particle [N,12], F [N,9], kk = w m/h^2, c = h v + h^2 g
__global__ void near_eg(int nn_, const long long* idx, const float* x, const float* U, const long long* rows, const float* lamb,
                        const float* glp, const float* Fp, const float* vol, const float* kk, const float* cc,
                        double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn,
                        double mu, double lam, int want_g, double* E, double* g) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0.0;
    if (q < nn_) {
        long long p = idx[q];
        long long nd[4]; double lm[4], gl[12], u[3]={0,0,0}, G[9]={0,0,0,0,0,0,0,0,0}, xx[3], c[3];
        for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; }
        for (int k=0;k<12;k++) gl[k]=glp[p*12+k];
        for (int i=0;i<4;i++) for (int a=0;a<3;a++){ double Ua=(double)U[nd[i]*3+a]; u[a]+=lm[i]*Ua; for (int b=0;b<3;b++) G[a*3+b]+=Ua*gl[i*3+b]; }
        for (int k=0;k<3;k++){ xx[k]=x[p*3+k]; c[k]=cc[p*3+k]; }
        double pt[3]={p0,p1,p2}, nv[3]={n0,n1,n2};
        NearMap M; near_map(xx, u, G, c, pt, nv, eps_st, eps_sp, xn, M);
        double Jf[9]; mm3(M.Mn, M.K1, Jf);
        double F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
        double X[9]; mm3(Jf, F, X);
        double C[9]; for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
        double l[3]; symeig3(C, l);
        double S[3], J=1, ps=0; for (int k=0;k<3;k++){ S[k]=fmax(sqrt(fmax(l[k],1e-12)),0.01); J*=S[k]; ps+=(S[k]-1)*(S[k]-1); }
        double V = vol[p], k_ = kk[p];
        double rr = M.r[0]*M.r[0]+M.r[1]*M.r[1]+M.r[2]*M.r[2];
        e = 0.5*k_*rr + V*(mu*ps + 0.5*lam*(J-1)*(J-1));
        if (want_g) {
            double Q[9], cf[9], P[9], Pc[9];
            polar(X, Q); cof3(X, cf);
            double dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
            double sg = (dt>0)?1.0:((dt<0)?-1.0:0.0);
            for (int k=0;k<9;k++) P[k] = 2.0*mu*(X[k]-Q[k]) + lam*(J-1.0)*sg*cf[k];
            mmt3(P, F, Pc); for (int k=0;k<9;k++) Pc[k]*=V;
            double gy[12];
            // u part: k Ru^T r + <Pc, dJu[c]>,  dJu[c] = g1 n_c N K1 + Mn dK1[c]
            double NK1[9]; mm3(M.Nm, M.K1, NK1);
            for (int c3=0;c3<3;c3++) {
                double t=0; for (int i=0;i<3;i++) t += M.Ru[i*3+c3]*M.r[i];
                double Bc[9], dK[9], MdK[9]; near_Bc(M, c3, Bc); mm3(Bc, M.IG, dK);
                for (int k=0;k<9;k++) dK[k] -= xn*M.w1*M.n[c3]*M.Pm[k];
                mm3(M.Mn, dK, MdK);
                double pj=0; for (int k=0;k<9;k++) pj += Pc[k]*(M.g1*M.n[c3]*NK1[k] + MdK[k]);
                gy[c3] = k_*t + pj;
            }
            double MJ[9]; mm3(M.Mn, M.Jst, MJ);
            for (int a=0;a<3;a++) for (int b=0;b<3;b++){ double t=0; for (int i=0;i<3;i++) t += MJ[i*3+a]*Pc[i*3+b]; gy[3+a*3+b]=t; }
            for (int i=0;i<4;i++) for (int c3=0;c3<3;c3++) {
                double t = lm[i]*gy[c3];
                for (int b=0;b<3;b++) t += gl[i*3+b]*gy[3+c3*3+b];
                atomicAdd(&g[nd[i]*3+c3], t);
            }
        }
    }
    block_add(e, E);
}

// Hessian (12x12 local) of bottom-candidate particles, atomically added to per-tet buffer Ht [T,144] (before projection)
__global__ void near_hess(int nn_, const long long* idx, const long long* tinv, const float* x, const float* U, const long long* rows, const float* lamb,
                          const float* glp, const float* Fp, const float* vol, const float* kk, const float* cc,
                          double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn,
                          float mu, float lam, float* Ht) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = idx[q];
    long long nd[4]; double lm[4], gl[12], u[3]={0,0,0}, G[9]={0,0,0,0,0,0,0,0,0}, xx[3], c[3];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; }
    for (int k=0;k<12;k++) gl[k]=glp[p*12+k];
    for (int i=0;i<4;i++) for (int a=0;a<3;a++){ double Ua=(double)U[nd[i]*3+a]; u[a]+=lm[i]*Ua; for (int b=0;b<3;b++) G[a*3+b]+=Ua*gl[i*3+b]; }
    for (int k=0;k<3;k++){ xx[k]=x[p*3+k]; c[k]=cc[p*3+k]; }
    double pt[3]={p0,p1,p2}, nv[3]={n0,n1,n2};
    NearMap M; near_map(xx, u, G, c, pt, nv, eps_st, eps_sp, xn, M);
    double Jf[9]; mm3(M.Mn, M.K1, Jf);
    float Ff[9], Xf[9], Af[9];
    for (int k=0;k<9;k++){ Ff[k]=Fp[p*9+k]; Af[k]=(float)Jf[k]; }
    mm3(Af, Ff, Xf);
    float Cx[9]; for (int i=0;i<3;i++) for (int k=0;k<3;k++) Cx[i*3+k]=Xf[i]*Xf[k]+Xf[3+i]*Xf[3+k]+Xf[6+i]*Xf[6+k];
    float l[3]; symeig3(Cx, l);
    float smin = sqrtf(fmaxf(l[0],1e-12f));
    float V = vol[p];
    float K[81]; kmanual(Xf, Ff, mu, lam, V, smin, K);
    double Pc[9];
    { float Q[9], cf[9], P[9], Pf[9]; polar(Xf, Q); cof3(Xf, cf);
      float det = Xf[0]*cf[0]+Xf[1]*cf[1]+Xf[2]*cf[2]; float sg=(det>0)?1.f:((det<0)?-1.f:0.f);
      for (int k=0;k<9;k++) P[k] = 2.f*mu*(Xf[k]-Q[k]) + lam*(det-sg)*cf[k];
      mmt3(P, Ff, Pf); for (int k=0;k<9;k++) Pc[k]=(double)(V*Pf[k]); }
    double k_ = kk[p];
    // JJ [9 x 12]: col c (u): vec(dJu[c]), col 3+ab: vec(MJ E_ab)
    double JJ[108];
    double NK1[9]; mm3(M.Nm, M.K1, NK1);
    double dK1s[27];
    for (int c3=0;c3<3;c3++) {
        double Bc[9], dK[9], MdK[9]; near_Bc(M, c3, Bc); mm3(Bc, M.IG, dK);
        for (int k=0;k<9;k++) dK[k] -= xn*M.w1*M.n[c3]*M.Pm[k];
        for (int k=0;k<9;k++) dK1s[c3*9+k]=dK[k];
        mm3(M.Mn, dK, MdK);
        for (int k=0;k<9;k++) JJ[k*12+c3] = M.g1*M.n[c3]*NK1[k] + MdK[k];
    }
    double MJ[9]; mm3(M.Mn, M.Jst, MJ);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) for (int i=0;i<3;i++) for (int j=0;j<3;j++) JJ[(i*3+j)*12 + 3+a*3+b] = (j==b) ? MJ[i*3+a] : 0.0;
    double Hy[144];
    { double KJ[108];
      for (int r=0;r<9;r++) for (int cc2=0;cc2<12;cc2++){ double t=0; for (int k=0;k<9;k++) t += (double)K[r*9+k]*JJ[k*12+cc2]; KJ[r*12+cc2]=t; }
      for (int r=0;r<12;r++) for (int cc2=0;cc2<12;cc2++){ double t=0; for (int k=0;k<9;k++) t += JJ[k*12+r]*KJ[k*12+cc2]; Hy[r*12+cc2]=t; } }
    // Huu
    double ra = M.r[0]*M.a[0]+M.r[1]*M.a[1]+M.r[2]*M.a[2], rn = M.r[0]*M.n[0]+M.r[1]*M.n[1]+M.r[2]*M.n[2];
    double Pr[3]; for (int i=0;i<3;i++){ double t=0; for (int j=0;j<3;j++) t += M.r[j]*M.Pm[j*3+i]; Pr[i]=t; }
    double PN=0; for (int k=0;k<9;k++) PN += Pc[k]*NK1[k];
    double PNdK[3];
    for (int c3=0;c3<3;c3++){ double NdK[9]; mm3(M.Nm, dK1s+c3*9, NdK); double t=0; for (int k=0;k<9;k++) t += Pc[k]*NdK[k]; PNdK[c3]=t; }
    for (int c3=0;c3<3;c3++) for (int d3=0;d3<3;d3++) {
        double t=0; for (int i=0;i<3;i++) t += M.Ru[i*3+c3]*M.Ru[i*3+d3];
        double huu = k_*(t + (M.w2*ra + M.g1*rn)*M.Nm[c3*3+d3] + M.w1*(Pr[c3]*M.n[d3] + M.n[c3]*Pr[d3]));
        double ncd = M.n[c3]*M.n[d3];
        double D2[9];
        for (int i=0;i<3;i++) for (int j=0;j<3;j++)
            D2[i*3+j] = M.w2*ncd*M.Pm[i*3+j] + M.w3*ncd*M.a[i]*M.n[j] + M.w2*(M.n[c3]*M.Pm[i*3+d3]*M.n[j] + M.n[d3]*M.Pm[i*3+c3]*M.n[j]);
        double D2I[9], MD2[9]; mm3(D2, M.IG, D2I);
        for (int k=0;k<9;k++) D2I[k] -= xn*M.w2*ncd*M.Pm[k];
        mm3(M.Mn, D2I, MD2);
        double pm=0; for (int k=0;k<9;k++) pm += Pc[k]*MD2[k];
        double phi = M.g2*PN*ncd + M.g1*(M.n[c3]*PNdK[d3] + M.n[d3]*PNdK[c3]) + pm;
        Hy[c3*12+d3] += huu + phi;
    }
    // Phi_uG[c][ab] = (Mc[c]^T Pc)_ab,  Mc[c] = g1 n_c N Jst + Mn Bc[c]
    double NJ[9]; mm3(M.Nm, M.Jst, NJ);
    for (int c3=0;c3<3;c3++) {
        double Bc[9], MB[9]; near_Bc(M, c3, Bc); mm3(M.Mn, Bc, MB);
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
            double t=0; for (int i=0;i<3;i++) t += (M.g1*M.n[c3]*NJ[i*3+a] + MB[i*3+a])*Pc[i*3+b];
            Hy[c3*12 + 3+a*3+b] += t; Hy[(3+a*3+b)*12 + c3] += t;
        }
    }
    // local = L^T Hy L ; L[a][(i,c)] = lm_i d_ac ; L[3+a*3+b][(i,c)] = d_ac gl_ib
    float* Ho = Ht + tinv[p]*144;
    for (int i=0;i<4;i++) for (int c3=0;c3<3;c3++) {
        double Lr[12]; for (int y=0;y<12;y++) Lr[y]=0; Lr[c3]=lm[i]; for (int b=0;b<3;b++) Lr[3+c3*3+b]=gl[i*3+b];
        double HL[12]; for (int y2=0;y2<12;y2++){ double t=0; for (int y=0;y<12;y++) t += Lr[y]*Hy[y*12+y2]; HL[y2]=t; }
        for (int j=0;j<4;j++) for (int d3=0;d3<3;d3++) {
            double t = HL[d3]*lm[j]; for (int b=0;b<3;b++) t += HL[3+d3*3+b]*gl[j*3+b];
            atomicAdd(&Ho[(i*3+c3)*12 + j*3+d3], (float)t);
        }
    }
}
__global__ void near_hess32(int nn_, const long long* idx, const long long* tinv, const float* x, const float* U, const long long* rows, const float* lamb,
                          const float* glp, const float* Fp, const float* vol, const float* kk, const float* cc,
                          double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn,
                          float mu, float lam, float* Ht) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = idx[q];
    long long nd[4]; float lm[4], gl[12], u[3]={0,0,0}, G[9]={0,0,0,0,0,0,0,0,0}, xx[3], c[3];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; }
    for (int k=0;k<12;k++) gl[k]=glp[p*12+k];
    for (int i=0;i<4;i++) for (int a=0;a<3;a++){ float Ua=(float)U[nd[i]*3+a]; u[a]+=lm[i]*Ua; for (int b=0;b<3;b++) G[a*3+b]+=Ua*gl[i*3+b]; }
    for (int k=0;k<3;k++){ xx[k]=x[p*3+k]; c[k]=cc[p*3+k]; }
    float pt[3]={(float)p0,(float)p1,(float)p2}, nv[3]={(float)n0,(float)n1,(float)n2};
    NearMapT<float> M; near_map<float>(xx, u, G, c, pt, nv, (float)eps_st, (float)eps_sp, (float)xn, M);
    float Jf[9]; mm3(M.Mn, M.K1, Jf);
    float Ff[9], Xf[9], Af[9];
    for (int k=0;k<9;k++){ Ff[k]=Fp[p*9+k]; Af[k]=(float)Jf[k]; }
    mm3(Af, Ff, Xf);
    float Cx[9]; for (int i=0;i<3;i++) for (int k=0;k<3;k++) Cx[i*3+k]=Xf[i]*Xf[k]+Xf[3+i]*Xf[3+k]+Xf[6+i]*Xf[6+k];
    float l[3]; symeig3(Cx, l);
    float smin = sqrtf(fmaxf(l[0],1e-12f));
    float V = vol[p];
    float K[81]; kmanual(Xf, Ff, mu, lam, V, smin, K);
    float Pc[9];
    { float Q[9], cf[9], P[9], Pf[9]; polar(Xf, Q); cof3(Xf, cf);
      float det = Xf[0]*cf[0]+Xf[1]*cf[1]+Xf[2]*cf[2]; float sg=(det>0)?1.f:((det<0)?-1.f:0.f);
      for (int k=0;k<9;k++) P[k] = 2.f*mu*(Xf[k]-Q[k]) + lam*(det-sg)*cf[k];
      mmt3(P, Ff, Pf); for (int k=0;k<9;k++) Pc[k]=(float)(V*Pf[k]); }
    float k_ = kk[p];
    // JJ [9 x 12]: col c (u): vec(dJu[c]), col 3+ab: vec(MJ E_ab)
    float JJ[108];
    float NK1[9]; mm3(M.Nm, M.K1, NK1);
    float dK1s[27];
    for (int c3=0;c3<3;c3++) {
        float Bc[9], dK[9], MdK[9]; near_Bc(M, c3, Bc); mm3(Bc, M.IG, dK);
        for (int k=0;k<9;k++) dK[k] -= xn*M.w1*M.n[c3]*M.Pm[k];
        for (int k=0;k<9;k++) dK1s[c3*9+k]=dK[k];
        mm3(M.Mn, dK, MdK);
        for (int k=0;k<9;k++) JJ[k*12+c3] = M.g1*M.n[c3]*NK1[k] + MdK[k];
    }
    float MJ[9]; mm3(M.Mn, M.Jst, MJ);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) for (int i=0;i<3;i++) for (int j=0;j<3;j++) JJ[(i*3+j)*12 + 3+a*3+b] = (j==b) ? MJ[i*3+a] : 0.0;
    float Hy[144];
    { float KJ[108];
      for (int r=0;r<9;r++) for (int cc2=0;cc2<12;cc2++){ float t=0; for (int k=0;k<9;k++) t += (float)K[r*9+k]*JJ[k*12+cc2]; KJ[r*12+cc2]=t; }
      for (int r=0;r<12;r++) for (int cc2=0;cc2<12;cc2++){ float t=0; for (int k=0;k<9;k++) t += JJ[k*12+r]*KJ[k*12+cc2]; Hy[r*12+cc2]=t; } }
    // Huu
    float ra = M.r[0]*M.a[0]+M.r[1]*M.a[1]+M.r[2]*M.a[2], rn = M.r[0]*M.n[0]+M.r[1]*M.n[1]+M.r[2]*M.n[2];
    float Pr[3]; for (int i=0;i<3;i++){ float t=0; for (int j=0;j<3;j++) t += M.r[j]*M.Pm[j*3+i]; Pr[i]=t; }
    float PN=0; for (int k=0;k<9;k++) PN += Pc[k]*NK1[k];
    float PNdK[3];
    for (int c3=0;c3<3;c3++){ float NdK[9]; mm3(M.Nm, dK1s+c3*9, NdK); float t=0; for (int k=0;k<9;k++) t += Pc[k]*NdK[k]; PNdK[c3]=t; }
    for (int c3=0;c3<3;c3++) for (int d3=0;d3<3;d3++) {
        float t=0; for (int i=0;i<3;i++) t += M.Ru[i*3+c3]*M.Ru[i*3+d3];
        float huu = k_*(t + (M.w2*ra + M.g1*rn)*M.Nm[c3*3+d3] + M.w1*(Pr[c3]*M.n[d3] + M.n[c3]*Pr[d3]));
        float ncd = M.n[c3]*M.n[d3];
        float D2[9];
        for (int i=0;i<3;i++) for (int j=0;j<3;j++)
            D2[i*3+j] = M.w2*ncd*M.Pm[i*3+j] + M.w3*ncd*M.a[i]*M.n[j] + M.w2*(M.n[c3]*M.Pm[i*3+d3]*M.n[j] + M.n[d3]*M.Pm[i*3+c3]*M.n[j]);
        float D2I[9], MD2[9]; mm3(D2, M.IG, D2I);
        for (int k=0;k<9;k++) D2I[k] -= xn*M.w2*ncd*M.Pm[k];
        mm3(M.Mn, D2I, MD2);
        float pm=0; for (int k=0;k<9;k++) pm += Pc[k]*MD2[k];
        float phi = M.g2*PN*ncd + M.g1*(M.n[c3]*PNdK[d3] + M.n[d3]*PNdK[c3]) + pm;
        Hy[c3*12+d3] += huu + phi;
    }
    // Phi_uG[c][ab] = (Mc[c]^T Pc)_ab,  Mc[c] = g1 n_c N Jst + Mn Bc[c]
    float NJ[9]; mm3(M.Nm, M.Jst, NJ);
    for (int c3=0;c3<3;c3++) {
        float Bc[9], MB[9]; near_Bc(M, c3, Bc); mm3(M.Mn, Bc, MB);
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
            float t=0; for (int i=0;i<3;i++) t += (M.g1*M.n[c3]*NJ[i*3+a] + MB[i*3+a])*Pc[i*3+b];
            Hy[c3*12 + 3+a*3+b] += t; Hy[(3+a*3+b)*12 + c3] += t;
        }
    }
    // local = L^T Hy L ; L[a][(i,c)] = lm_i d_ac ; L[3+a*3+b][(i,c)] = d_ac gl_ib
    float* Ho = Ht + tinv[p]*144;
    for (int i=0;i<4;i++) for (int c3=0;c3<3;c3++) {
        float Lr[12]; for (int y=0;y<12;y++) Lr[y]=0; Lr[c3]=lm[i]; for (int b=0;b<3;b++) Lr[3+c3*3+b]=gl[i*3+b];
        float HL[12]; for (int y2=0;y2<12;y2++){ float t=0; for (int y=0;y<12;y++) t += Lr[y]*Hy[y*12+y2]; HL[y2]=t; }
        for (int j=0;j<4;j++) for (int d3=0;d3<3;d3++) {
            float t = HL[d3]*lm[j]; for (int b=0;b<3;b++) t += HL[3+d3*3+b]*gl[j*3+b];
            atomicAdd(&Ho[(i*3+c3)*12 + j*3+d3], (float)t);
        }
    }
}


// ---------------- grid: locate (simplex.locate same formula) ----------------
__global__ void locate_k(int N, const float* x, const float* lo, const float* Ai, int n0, int n1, int n2,
                         long long* idx, float* lamb, long long* tid, unsigned char* nmark, unsigned char* tmark) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    float q[3] = {x[p*3]-lo[0], x[p*3+1]-lo[1], x[p*3+2]-lo[2]};
    float y[3]; for (int k=0;k<3;k++) y[k] = Ai[k*3]*q[0] + Ai[k*3+1]*q[1] + Ai[k*3+2]*q[2];
    int nl[3] = {n0, n1, n2};
    int ci[3]; float f[3];
    for (int k=0;k<3;k++){ int c = (int)floorf(y[k]); c = min(max(c, 0), nl[k]-2); ci[k]=c; f[k] = fminf(fmaxf(y[k]-(float)c, 0.f), 1.f); }
    int pr[3] = {0,1,2};
    // descending sort of f with index
    if (f[pr[0]] < f[pr[1]]) { int t=pr[0]; pr[0]=pr[1]; pr[1]=t; }
    if (f[pr[1]] < f[pr[2]]) { int t=pr[1]; pr[1]=pr[2]; pr[2]=t; }
    if (f[pr[0]] < f[pr[1]]) { int t=pr[0]; pr[0]=pr[1]; pr[1]=t; }
    float s0=f[pr[0]], s1=f[pr[1]], s2=f[pr[2]];
    lamb[p*4]=1.f-s0; lamb[p*4+1]=s0-s1; lamb[p*4+2]=s1-s2; lamb[p*4+3]=s2;
    int v[3] = {ci[0], ci[1], ci[2]};
    for (int r=0;r<4;r++) {
        if (r>0) v[pr[r-1]] += 1;
        long long fl = ((long long)v[0]*n1 + v[1])*n2 + v[2];
        idx[p*4+r] = fl; nmark[fl] = 1;
    }
    long long cell = ((long long)ci[0]*n1 + ci[1])*n2 + ci[2];
    long long tt = cell*9 + pr[0]*3 + pr[1];
    tid[p] = tt; tmark[tt] = 1;
}
// per-tet closed-form gradient of barycentric (float64): Dm = [X1-X0, X2-X0, X3-X0] columns, Di = inv(Dm) rows = grad lambda_{1..3}
__global__ void tet_geo(int T, const long long* tets, const float* npos, double* gl) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    double X[4][3];
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) X[i][c] = (double)npos[tets[t*4+i]*3+c];
    double D[9];
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) D[r*3+c] = X[c+1][r]-X[0][r];
    double C[9]; cof3(D, C);
    double det = D[0]*C[0]+D[1]*C[1]+D[2]*C[2];
    // inv = cof^T / det
    for (int i=0;i<3;i++) for (int b=0;b<3;b++) gl[t*12 + (i+1)*3 + b] = C[b*3+i]/det;
    for (int b=0;b<3;b++) gl[t*12+b] = -(gl[t*12+3+b]+gl[t*12+6+b]+gl[t*12+9+b]);
}
// ---------------- G2P with stick -> softplus maps (fem_fast.FemFrame._maps), float64 inside ----------------
__global__ void g2p_part(int N, const float* x, const float* U, const long long* rows, const float* lamb, const float* glT, const long long* tinv,
                         const float* Fp, double p0, double p1, double p2, double n0, double n1, double n2,
                         double eps_st, double eps_sp, double xn, double h, float* x2, float* v2, float* F2) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double u[3]={0,0,0}, G[9]={0,0,0,0,0,0,0,0,0}, xx[3], c0[3]={0,0,0};
    const float* gl = glT + tinv[p]*12;
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; double lm=lamb[p*4+i]; for (int a=0;a<3;a++){ double Ua=U[nd*3+a]; u[a]+=lm*Ua; for (int b=0;b<3;b++) G[a*3+b]+=Ua*gl[i*3+b]; } }
    for (int k=0;k<3;k++) xx[k]=x[p*3+k];
    double pt[3]={p0,p1,p2}, nv[3]={n0,n1,n2};
    NearMap M; near_map(xx, u, G, c0, pt, nv, eps_st, eps_sp, xn, M);
    double Jf[9]; mm3(M.Mn, M.K1, Jf);
    for (int k=0;k<3;k++) { double qk = xx[k] + M.r[k]; x2[p*3+k] = (float)qk; v2[p*3+k] = (float)((qk - xx[k])/h); }
    double F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    double F2d[9]; mm3(Jf, F, F2d);
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
}



// ---------------- G2P without floor maps (--floor none): x2 = x + sum lam U, F2 = (I + sum U_i (x) gl_i) F, float64 inside ----------------
__global__ void g2p_lin(int N, const float* x, const float* U, const long long* rows, const float* lamb, const float* glT, const long long* tinv,
                        const float* Fp, double h, float* x2, float* v2, float* F2) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double u[3]={0,0,0}, A[9]={1,0,0,0,1,0,0,0,1};
    const float* gl = glT + tinv[p]*12;
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; double lm=lamb[p*4+i]; for (int a=0;a<3;a++){ double Ua=U[nd*3+a]; u[a]+=lm*Ua; for (int b=0;b<3;b++) A[a*3+b]+=Ua*gl[i*3+b]; } }
    for (int k=0;k<3;k++) { double xk = x[p*3+k]; double qk = xk + u[k]; x2[p*3+k] = (float)qk; v2[p*3+k] = (float)((qk - xk)/h); }
    double F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    double F2d[9]; mm3(A, F, F2d);
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
}

// 16 lanes per node block-ELL row: lane k<15 does slot k (3x3 f32 block x f64 3-vector), shuffle-reduce to lane 0 of the 16-group
__device__ __forceinline__ void ell16(int m, int k, bool on, int Ms, const float* B, const int* nb, const double* v, double* y) {
    double y0=0, y1=0, y2=0;
    if (on && k < 15) {
        int j = nb[k*Ms+m];
        const float* bb = B + (long long)k*9*Ms + m;
        double x0=__ldcg(&v[j*3]), x1=__ldcg(&v[j*3+1]), x2=__ldcg(&v[j*3+2]);
        y0 = bb[0]*x0 + bb[Ms]*x1 + bb[2*Ms]*x2;
        y1 = bb[3*Ms]*x0 + bb[4*Ms]*x1 + bb[5*Ms]*x2;
        y2 = bb[6*Ms]*x0 + bb[7*Ms]*x1 + bb[8*Ms]*x2;
    }
    for (int o=8;o>0;o>>=1) { y0 += __shfl_down_sync(0xffffffff, y0, o, 16); y1 += __shfl_down_sync(0xffffffff, y1, o, 16); y2 += __shfl_down_sync(0xffffffff, y2, o, 16); }
    y[0]=y0; y[1]=y1; y[2]=y2;
}
__global__ void pcg2_d16(int p, const int* Mp, const float* B, const int* nb, const double* z, double* d, double* Ad,
                         double* scal, double* acc, double thr, int Ms) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int m = gid >> 4, k = gid & 15;
    int M = *Mp;
    const double* S = scal + p*4; const double* A = acc + p*3;
    bool alive = S[1] > 0.5;
    bool ae = alive && (A[0] > 0);
    bool an = ae && !(A[1] < thr);
    double be = an ? A[2]/S[0] : 0.0;
    double part = 0.0;
    double q[3];
    ell16(m, k, an && m < M, Ms, B, nb, z, q);
    if (an && m < M && k == 0) {
        for (int c=0;c<3;c++){ double dd = z[m*3+c] + be*d[m*3+c]; double aa = q[c] + be*Ad[m*3+c]; d[m*3+c]=dd; Ad[m*3+c]=aa; part += dd*aa; }
    }
    if (blockIdx.x==0 && threadIdx.x==0) {
        double* O = scal + (1-p)*4;
        O[0] = an ? A[2] : S[0]; O[1] = an ? 1.0 : 0.0; O[2] = S[2] + (ae ? 1.0 : 0.0); O[3] = (S[3] > 0.5 || (alive && !(A[0] > 0))) ? 1.0 : 0.0;
    }
    block_add(part, &acc[(1-p)*3]);
}


__global__ void pcg2_d3(int p, const int* Mp, const float* B, const int* nb, const double* z, double* d, double* Ad,
                        double* scal, double* acc, double thr, int Ms) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int c = gid / Ms, m = gid - c*Ms;
    int M = *Mp;
    const double* S = scal + p*4; const double* A = acc + p*3;
    bool alive = S[1] > 0.5;
    bool ae = alive && (A[0] > 0);
    bool an = ae && !(A[1] < thr);
    double be = an ? A[2]/S[0] : 0.0;
    double part = 0.0;
    if (an && c < 3 && m < M) {
        double q = ell_row<false>(m, c, Ms, B, nb, z);
        double dd = z[m*3+c] + be*d[m*3+c]; double aa = q + be*Ad[m*3+c]; d[m*3+c]=dd; Ad[m*3+c]=aa; part = dd*aa;
    }
    if (blockIdx.x==0 && threadIdx.x==0) {
        double* O = scal + (1-p)*4;
        O[0] = an ? A[2] : S[0]; O[1] = an ? 1.0 : 0.0; O[2] = S[2] + (ae ? 1.0 : 0.0); O[3] = (S[3] > 0.5 || (alive && !(A[0] > 0))) ? 1.0 : 0.0;
    }
    block_add(part, &acc[(1-p)*3]);
}

// ---------------- persistent PCG (one launch, software grid barrier, all blocks co-resident) ----------------
__device__ void gbar(unsigned int* cnt, volatile unsigned int* gen, unsigned int nblk) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int g = *gen;
        __threadfence();
        if (atomicAdd(cnt, 1u) == nblk - 1) { *cnt = 0; __threadfence(); atomicAdd((unsigned int*)gen, 1u); }
        else { while (*gen == g) { } }
        __threadfence();
    }
    __syncthreads();
}
__device__ double block_sum(double v) {
    __shared__ double sh2[32];
    __syncthreads();
    v = warp_sum(v);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane==0) sh2[w]=v;
    __syncthreads();
    double t = 0;
    if (w==0) { t = (lane < (blockDim.x>>5)) ? sh2[lane] : 0.0; t = warp_sum(t); }
    return t;  // valid in thread 0
}
__global__ void pcg_coop(int M, int Ms, const float* B, const int* nb, const double* Dinv, double* x, double* r, double* z, double* d, double* Ad,
                         double* acc, double* sc, unsigned int* bar, double thr, int maxit) {
    // sc: [0] rz0, [1] dAd0 (input); out [2] nit, [3] rr, [4] neg
    unsigned int nblk = gridDim.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x, nth = gridDim.x*blockDim.x;
    volatile double* va = acc;
    double rz = sc[0], dAd = sc[1];
    int nit = 0, neg = 0, p = 0;
    double rr = 0;
    for (int it=0; it<maxit; it++) {
        if (!(dAd > 0)) { neg = 1; break; }
        double al = rz/dAd;
        double prr=0, prz=0;
        for (int m=tid; m<M; m+=nth) {
            double rv[3];
            for (int c=0;c<3;c++){ x[m*3+c] += al*d[m*3+c]; double rc = r[m*3+c] - al*Ad[m*3+c]; r[m*3+c]=rc; rv[c]=rc; }
            const double* D = Dinv + m*9;
            for (int c=0;c<3;c++){ double zz = D[c*3]*rv[0]+D[c*3+1]*rv[1]+D[c*3+2]*rv[2]; z[m*3+c]=zz; prz += rv[c]*zz; prr += rv[c]*rv[c]; }
        }
        double b1 = block_sum(prr); double b2 = block_sum(prz);
        if (threadIdx.x==0) { atomicAdd(&acc[p*3+1], b1); atomicAdd(&acc[p*3+2], b2); }
        if (tid==0) { acc[(1-p)*3]=0; acc[(1-p)*3+1]=0; acc[(1-p)*3+2]=0; }
        gbar(bar, bar+1, nblk);
        rr = va[p*3+1]; double rz2 = va[p*3+2];
        nit++;
        if (rr < thr) break;
        double be = rz2/rz; rz = rz2;
        double pd = 0;
        for (int w2 = tid; w2 < 3*M; w2 += nth) {
            int c = w2 / M, m = w2 - c*M;
            double q = ell_row<true>(m, c, Ms, B, nb, z);
            double dd = z[m*3+c] + be*d[m*3+c]; double aa = q + be*Ad[m*3+c]; d[m*3+c]=dd; Ad[m*3+c]=aa; pd += dd*aa;
        }
        double b3 = block_sum(pd);
        if (threadIdx.x==0) atomicAdd(&acc[(1-p)*3], b3);
        gbar(bar, bar+1, nblk);
        dAd = va[(1-p)*3];
        p = 1-p;
    }
    if (tid==0) { sc[2]=nit; sc[3]=rr; sc[4]=neg; }
}

// persistent PCG with float32 search vectors (z, d, Ad) and float32 SpMV; x, r and all scalars float64 (same stopping test on the recursive residual)
__global__ void pcg_coop32(int M, int Ms, const float* B, const int* nb, const double* Dinv, double* x, double* r, float* z, float* d, float* Ad,
                           double* acc, double* sc, unsigned int* bar, double thr, int maxit) {
    unsigned int nblk = gridDim.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x, nth = gridDim.x*blockDim.x;
    volatile double* va = acc;
    double rz = sc[0], dAd = sc[1];
    int nit = 0, neg = 0, p = 0;
    double rr = 0;
    for (int it=0; it<maxit; it++) {
        if (!(dAd > 0)) { neg = 1; break; }
        double al = rz/dAd;
        double prr=0, prz=0;
        for (int m=tid; m<M; m+=nth) {
            double rv[3];
            for (int c=0;c<3;c++){ x[m*3+c] += al*(double)d[m*3+c]; double rc = r[m*3+c] - al*(double)Ad[m*3+c]; r[m*3+c]=rc; rv[c]=rc; }
            const double* D = Dinv + m*9;
            for (int c=0;c<3;c++){ double zz = D[c*3]*rv[0]+D[c*3+1]*rv[1]+D[c*3+2]*rv[2]; z[m*3+c]=(float)zz; prz += rv[c]*zz; prr += rv[c]*rv[c]; }
        }
        double b1 = block_sum(prr); double b2 = block_sum(prz);
        if (threadIdx.x==0) { atomicAdd(&acc[p*3+1], b1); atomicAdd(&acc[p*3+2], b2); }
        if (tid==0) { acc[(1-p)*3]=0; acc[(1-p)*3+1]=0; acc[(1-p)*3+2]=0; }
        gbar(bar, bar+1, nblk);
        rr = va[p*3+1]; double rz2 = va[p*3+2];
        nit++;
        if (rr < thr) break;
        double be = rz2/rz; rz = rz2;
        double pd = 0;
        float bef = (float)be;
        for (int m=tid; m<M; m+=nth) {
            for (int c=0;c<3;c++) {
                float q = ell_row32<float>(m, c, Ms, B, nb, z);
                float dd = z[m*3+c] + bef*d[m*3+c]; float aa = q + bef*Ad[m*3+c]; d[m*3+c]=dd; Ad[m*3+c]=aa; pd += (double)dd*(double)aa;
            }
        }
        double b3 = block_sum(pd);
        if (threadIdx.x==0) atomicAdd(&acc[(1-p)*3], b3);
        gbar(bar, bar+1, nblk);
        dAd = va[(1-p)*3];
        p = 1-p;
    }
    if (tid==0) { sc[2]=nit; sc[3]=rr; sc[4]=neg; }
}

__global__ void pcg_coop1(int M, int Ms, const float* B, const int* nb, const double* Dinv, double* x, double* r, double* z, double* d, double* Ad,
                         double* acc, double* sc, unsigned int* bar, double thr, int maxit, int sym) {
    // sc: [0] rz0, [1] dAd0 (input); out [2] nit, [3] rr, [4] neg
    unsigned int nblk = gridDim.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x, nth = gridDim.x*blockDim.x;
    volatile double* va = acc;
    double rz = sc[0], dAd = sc[1];
    int nit = 0, neg = 0, p = 0;
    double rr = 0;
    for (int it=0; it<maxit; it++) {
        if (!(dAd > 0)) { neg = 1; break; }
        double al = rz/dAd;
        double prr=0, prz=0;
        for (int m=tid; m<M; m+=nth) {
            double rv[3];
            for (int c=0;c<3;c++){ x[m*3+c] += al*d[m*3+c]; double rc = r[m*3+c] - al*Ad[m*3+c]; r[m*3+c]=rc; rv[c]=rc; }
            const double* D = Dinv + m*9;
            for (int c=0;c<3;c++){ double zz = D[c*3]*rv[0]+D[c*3+1]*rv[1]+D[c*3+2]*rv[2]; z[m*3+c]=zz; prz += rv[c]*zz; prr += rv[c]*rv[c]; }
        }
        double b1 = block_sum(prr); double b2 = block_sum(prz);
        if (threadIdx.x==0) { atomicAdd(&acc[p*3+1], b1); atomicAdd(&acc[p*3+2], b2); }
        if (tid==0) { acc[(1-p)*3]=0; acc[(1-p)*3+1]=0; acc[(1-p)*3+2]=0; }
        gbar(bar, bar+1, nblk);
        rr = va[p*3+1]; double rz2 = va[p*3+2];
        nit++;
        if (rr < thr) break;
        double be = rz2/rz; rz = rz2;
        double pd = 0;
        for (int m=tid; m<M; m+=nth) {
            double q[3];
            for (int c=0;c<3;c++) q[c] = sym ? ell_row_sym<true>(m, c, Ms, B, nb, z) : ell_row<true>(m, c, Ms, B, nb, z);
            for (int c=0;c<3;c++){ double dd = z[m*3+c] + be*d[m*3+c]; double aa = q[c] + be*Ad[m*3+c]; d[m*3+c]=dd; Ad[m*3+c]=aa; pd += dd*aa; }
        }
        double b3 = block_sum(pd);
        if (threadIdx.x==0) atomicAdd(&acc[(1-p)*3], b3);
        gbar(bar, bar+1, nblk);
        dAd = va[(1-p)*3];
        p = 1-p;
    }
    if (tid==0) { sc[2]=nit; sc[3]=rr; sc[4]=neg; }
}
__global__ void pcg_coop64(int M, int Ms, const double* B, const int* nb, const double* Dinv, double* x, double* r, double* z, double* d, double* Ad,
                         double* acc, double* sc, unsigned int* bar, double thr, int maxit, int sym) {
    // sc: [0] rz0, [1] dAd0 (input); out [2] nit, [3] rr, [4] neg
    unsigned int nblk = gridDim.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x, nth = gridDim.x*blockDim.x;
    volatile double* va = acc;
    double rz = sc[0], dAd = sc[1];
    int nit = 0, neg = 0, p = 0;
    double rr = 0;
    for (int it=0; it<maxit; it++) {
        if (!(dAd > 0)) { neg = 1; break; }
        double al = rz/dAd;
        double prr=0, prz=0;
        for (int m=tid; m<M; m+=nth) {
            double rv[3];
            for (int c=0;c<3;c++){ x[m*3+c] += al*d[m*3+c]; double rc = r[m*3+c] - al*Ad[m*3+c]; r[m*3+c]=rc; rv[c]=rc; }
            const double* D = Dinv + m*9;
            for (int c=0;c<3;c++){ double zz = D[c*3]*rv[0]+D[c*3+1]*rv[1]+D[c*3+2]*rv[2]; z[m*3+c]=zz; prz += rv[c]*zz; prr += rv[c]*rv[c]; }
        }
        double b1 = block_sum(prr); double b2 = block_sum(prz);
        if (threadIdx.x==0) { atomicAdd(&acc[p*3+1], b1); atomicAdd(&acc[p*3+2], b2); }
        if (tid==0) { acc[(1-p)*3]=0; acc[(1-p)*3+1]=0; acc[(1-p)*3+2]=0; }
        gbar(bar, bar+1, nblk);
        rr = va[p*3+1]; double rz2 = va[p*3+2];
        nit++;
        if (rr < thr) break;
        double be = rz2/rz; rz = rz2;
        double pd = 0;
        for (int m=tid; m<M; m+=nth) {
            double q[3];
            for (int c=0;c<3;c++) q[c] = ell_row_sym64<true>(m, c, Ms, B, nb, z);
            for (int c=0;c<3;c++){ double dd = z[m*3+c] + be*d[m*3+c]; double aa = q[c] + be*Ad[m*3+c]; d[m*3+c]=dd; Ad[m*3+c]=aa; pd += dd*aa; }
        }
        double b3 = block_sum(pd);
        if (threadIdx.x==0) atomicAdd(&acc[(1-p)*3], b3);
        gbar(bar, bar+1, nblk);
        dAd = va[(1-p)*3];
        p = 1-p;
    }
    if (tid==0) { sc[2]=nit; sc[3]=rr; sc[4]=neg; }
}


// ---------------- VBD (Vertex Block Descent): per-color local gradients (float32) ----------------
__global__ void vbd_tet_g(int T, const long long* tets, const int* color, int cc, const float* U, const float* gl, const float* FT, const float* VT,
                          const float* M4, const float* B4, float mu, float lam, float* g) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    long long n[4]; int ii = -1;
    for (int i=0;i<4;i++){ n[i]=tets[t*4+i]; if (color[n[i]]==cc) ii=i; }
    if (ii < 0) return;
    float u[4][3];
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) u[i][c]=U[n[i]*3+c];
    const float* G = gl + t*12;
    float gi[3];
    for (int c=0;c<3;c++){ float s=0; for (int j=0;j<4;j++) s += M4[t*16+ii*4+j]*u[j][c]; gi[c] = s - B4[t*12+ii*3+c]; }
    float V = VT[t];
    if (V > 0.f) {
        float A[9];
        for (int c=0;c<3;c++) for (int b=0;b<3;b++){ float s=(c==b)?1.f:0.f; for (int i=0;i<4;i++) s += u[i][c]*G[i*3+b]; A[c*3+b]=s; }
        const float* Fb = FT + t*9;
        float X[9]; mm3(A, Fb, X);
        float Q[9], cf[9], P[9], Gm[9];
        polar(X, Q); cof3(X, cf);
        float det = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
        float sg = (det>0)?1.f:((det<0)?-1.f:0.f);
        for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(det-sg)*cf[k];
        mmt3(P, Fb, Gm);
        for (int c=0;c<3;c++) for (int b=0;b<3;b++) gi[c] += V*Gm[c*3+b]*G[ii*3+b];
    }
    for (int c=0;c<3;c++) atomicAdd(&g[n[ii]*3+c], gi[c]);
}
__global__ void vbd_near_g(int nn_, const long long* idx, const float* x, const float* U, const long long* rows, const float* lamb,
                           const float* glp, const float* Fp, const float* vol, const float* kk, const float* ccv,
                           double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn,
                           float mu, float lam, const int* color, int cc, float* g) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = idx[q];
    long long nd[4]; int ii=-1;
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; if (color[nd[i]]==cc) ii=i; }
    if (ii < 0) return;
    float lm[4], gl[12], u[3]={0,0,0}, G[9]={0,0,0,0,0,0,0,0,0}, xx[3], c[3];
    for (int i=0;i<4;i++) lm[i]=lamb[p*4+i];
    for (int k=0;k<12;k++) gl[k]=glp[p*12+k];
    for (int i=0;i<4;i++) for (int a=0;a<3;a++){ float Ua=U[nd[i]*3+a]; u[a]+=lm[i]*Ua; for (int b=0;b<3;b++) G[a*3+b]+=Ua*gl[i*3+b]; }
    for (int k=0;k<3;k++){ xx[k]=x[p*3+k]; c[k]=ccv[p*3+k]; }
    float pt[3]={(float)p0,(float)p1,(float)p2}, nv[3]={(float)n0,(float)n1,(float)n2};
    NearMapT<float> M; near_map<float>(xx, u, G, c, pt, nv, (float)eps_st, (float)eps_sp, (float)xn, M);
    float Jf[9]; mm3(M.Mn, M.K1, Jf);
    float F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    float X[9]; mm3(Jf, F, X);
    float Q[9], cf[9], P[9], Pc[9];
    polar(X, Q); cof3(X, cf);
    float det = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2]; float sg=(det>0)?1.f:((det<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(det-sg)*cf[k];
    float V = vol[p];
    mmt3(P, F, Pc); for (int k=0;k<9;k++) Pc[k]*=V;
    float k_ = kk[p];
    float gy[12];
    float NK1[9]; mm3(M.Nm, M.K1, NK1);
    for (int c3=0;c3<3;c3++) {
        float t=0; for (int i=0;i<3;i++) t += M.Ru[i*3+c3]*M.r[i];
        float Bc[9], dK[9], MdK[9]; near_Bc(M, c3, Bc); mm3(Bc, M.IG, dK);
        for (int k=0;k<9;k++) dK[k] -= (float)xn*M.w1*M.n[c3]*M.Pm[k];
        mm3(M.Mn, dK, MdK);
        float pj=0; for (int k=0;k<9;k++) pj += Pc[k]*(M.g1*M.n[c3]*NK1[k] + MdK[k]);
        gy[c3] = k_*t + pj;
    }
    float MJ[9]; mm3(M.Mn, M.Jst, MJ);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++){ float t=0; for (int i=0;i<3;i++) t += MJ[i*3+a]*Pc[i*3+b]; gy[3+a*3+b]=t; }
    for (int c3=0;c3<3;c3++) {
        float t = lm[ii]*gy[c3];
        for (int b=0;b<3;b++) t += gl[ii*3+b]*gy[3+c3*3+b];
        atomicAdd(&g[nd[ii]*3+c3], t);
    }
}
// node update for color cc: U -= D^-1 g (3x3 block), g reset
__global__ void vbd_node(int M, const int* color, int cc, const float* Dinv, float* g, float* U, float step, float maxstep) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= M || color[m] != cc) return;
    float g0=g[m*3], g1=g[m*3+1], g2=g[m*3+2];
    const float* D = Dinv + m*9;
    float d0 = step*(D[0]*g0 + D[1]*g1 + D[2]*g2), d1 = step*(D[3]*g0 + D[4]*g1 + D[5]*g2), d2 = step*(D[6]*g0 + D[7]*g1 + D[8]*g2);
    float nd = sqrtf(d0*d0+d1*d1+d2*d2);
    if (maxstep > 0.f && nd > maxstep) { float sc = maxstep/nd; d0*=sc; d1*=sc; d2*=sc; }
    if (isfinite(nd)) { U[m*3] -= d0; U[m*3+1] -= d1; U[m*3+2] -= d2; }
    g[m*3]=0; g[m*3+1]=0; g[m*3+2]=0;
}

// ---------------- PCG (블록 ELL float32 행렬, float64 벡터; 상태 st: 0 rz 1 dAd 2 rr 3 rz2 4 thr 5 alive 6 nit 7 neg 8 alpha 9 beta) ----------------
__global__ void pcg_spmv(const int* Mp, const float* B, const int* nb, const double* d, double* y, double* st, int Ms) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    int M = *Mp;
    double part = 0.0;
    if (m < M && st[5] > 0.5) {
        double y0=0,y1=0,y2=0;
        for (int k=0;k<15;k++) {
            int j = nb[k*Ms+m];
            const float* b = B + (long long)k*9*Ms + m;
            double x0=d[j*3], x1=d[j*3+1], x2=d[j*3+2];
            y0 += b[0]*x0 + b[Ms]*x1 + b[2*Ms]*x2;
            y1 += b[3*Ms]*x0 + b[4*Ms]*x1 + b[5*Ms]*x2;
            y2 += b[6*Ms]*x0 + b[7*Ms]*x1 + b[8*Ms]*x2;
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
                       double* scal, double* acc, double thr, int Ms) {
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
            int j = nb[k*Ms+m];
            const float* b = B + (long long)k*9*Ms + m;
            double x0=z[j*3], x1=z[j*3+1], x2=z[j*3+2];
            y0 += b[0]*x0 + b[Ms]*x1 + b[2*Ms]*x2;
            y1 += b[3*Ms]*x0 + b[4*Ms]*x1 + b[5*Ms]*x2;
            y2 += b[6*Ms]*x0 + b[7*Ms]*x1 + b[8*Ms]*x2;
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
__global__ void bell_mv(const int* Mp, const float* B, const int* nb, const double* v, double* y, int Ms, int sym) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= *Mp) return;
    for (int c=0;c<3;c++) y[m*3+c] = sym ? ell_row_sym<false>(m, c, Ms, B, nb, v) : ell_row<false>(m, c, Ms, B, nb, v);
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
// ---------------- assembly helpers (result-invariant replacements of the torch glue) ----------------
// slots[t, a*4+b] = tets[t,a]*15 + class(gi[tets[t,b]] - gi[tets[t,a]])   (class table: 27 -> 0..14 or -1)
__global__ void slots_k(int T, const long long* tets, const long long* gi, const int* tbl, int* slots) {
    int e = blockIdx.x*blockDim.x + threadIdx.x;
    if (e >= T*16) return;
    int t = e >> 4, a = (e >> 2) & 3, b = e & 3;
    long long na = tets[t*4+a], nb_ = tets[t*4+b];
    long long d0 = gi[nb_*3]-gi[na*3], d1 = gi[nb_*3+1]-gi[na*3+1], d2 = gi[nb_*3+2]-gi[na*3+2];
    int cls = -1;
    if (d0 >= -1 && d0 <= 1 && d1 >= -1 && d1 <= 1 && d2 >= -1 && d2 <= 1) cls = tbl[(d0+1)*9 + (d1+1)*3 + (d2+1)];
    slots[e] = (int)(na*15 + cls);
}
// per node: block-Jacobi inverse of the self block (cls 0) in float64 (identity if a diagonal entry is not > 0), and the 15-neighbor table
// (lattice index of node = uniq[m]; neighbor = ncm[lattice + offset] if marked; invalid -> -1 when sym else self)
__global__ void asm_k(int M, int Ms, const float* B, double* Dinv, float* Dinv32, const long long* uniq, int n0, int n1, int n2,
                      const unsigned char* nmark, const long long* ncm, const int* offs, int sym, int* nb) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= M) return;
    double D[9];
    for (int e=0;e<9;e++) D[e] = (double)B[(long long)e*Ms + m];
    if (!(D[0] > 0 && D[4] > 0 && D[8] > 0)) { for (int e=0;e<9;e++) D[e] = (e%4==0) ? 1.0 : 0.0; }
    double C[9]; cof3(D, C);
    double det = D[0]*C[0] + D[1]*C[1] + D[2]*C[2];
    // inverse = cof^T / det
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) { double v = C[c*3+r]/det; Dinv[m*9+r*3+c] = v; if (Dinv32) Dinv32[m*9+r*3+c] = (float)v; }
    long long L = uniq[m];
    int k2 = (int)(L % n2), k1 = (int)((L / n2) % n1), k0 = (int)(L / ((long long)n1*n2));
    for (int k=0;k<15;k++) {
        int a0 = k0 + offs[k*3], a1 = k1 + offs[k*3+1], a2 = k2 + offs[k*3+2];
        int j = -1;
        if (a0 >= 0 && a0 < n0 && a1 >= 0 && a1 < n1 && a2 >= 0 && a2 < n2) {
            long long q = ((long long)a0*n1 + a1)*n2 + a2;
            if (nmark[q]) j = (int)ncm[q];
        }
        nb[k*Ms+m] = (j >= 0) ? j : (sym ? -1 : m);
    }
}
// ---------------- VBD extras: node floor terms, per-node local line search ----------------
__device__ __forceinline__ void barrier3_d(double d, double dhat, double* b, double* b1) {
    double eps = 1e-3*dhat;
    if (!(d < dhat)) { *b = 0; *b1 = 0; return; }
    double le = log(eps/dhat), ke = eps - dhat;
    if (d < eps) {
        double be = -(ke*ke)*le, bp = -2.0*ke*le - ke*ke/eps, bpp = -2.0*le - 4.0*ke/eps + ke*ke/(eps*eps);
        double de = d - eps;
        *b = be + bp*de + 0.5*bpp*de*de; *b1 = bp + bpp*de; return;
    }
    double L = log(d/dhat), k = d - dhat;
    *b = -(k*k)*L; *b1 = -2.0*k*L - k*k/d;
}
// node floor gradient (barrier kb b(c_n + n.U_n) + stick ks w |P U_n|^2 / 2) for nodes of color cc, added to g
__global__ void vbd_floor_g(int M, const int* color, int cc, const float* U, const double* cn, const double* w, const double* kb, const double* ks,
                            double n0, double n1, double n2, double dhat, float* g) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= M || color[m] != cc) return;
    double nv[3] = {n0, n1, n2}, u[3] = {U[m*3], U[m*3+1], U[m*3+2]};
    double un = u[0]*n0 + u[1]*n1 + u[2]*n2;
    double b, b1; barrier3_d(cn[m] + un, dhat, &b, &b1);
    for (int c=0;c<3;c++) g[m*3+c] += (float)(kb[m]*b1*nv[c] + ks[m]*w[m]*(u[c] - un*nv[c]));
}
// step dl = D^-1 g (clamped to maxstep, non-finite -> 0) for nodes of color cc; g reset
__global__ void vbd_step(int M, const int* color, int cc, const float* Dinv, float* g, float* dl, float maxstep) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= M || color[m] != cc) return;
    float g0=g[m*3], g1=g[m*3+1], g2=g[m*3+2];
    const float* D = Dinv + m*9;
    float d0 = D[0]*g0 + D[1]*g1 + D[2]*g2, d1 = D[3]*g0 + D[4]*g1 + D[5]*g2, d2 = D[6]*g0 + D[7]*g1 + D[8]*g2;
    float nd = sqrtf(d0*d0+d1*d1+d2*d2);
    if (!isfinite(nd)) { d0=0; d1=0; d2=0; }
    else if (maxstep > 0.f && nd > maxstep) { float sc = maxstep/nd; d0*=sc; d1*=sc; d2*=sc; }
    dl[m*3]=d0; dl[m*3+1]=d1; dl[m*3+2]=d2;
    g[m*3]=0; g[m*3+1]=0; g[m*3+2]=0;
}
__device__ __forceinline__ float psi_f(const float* X, float mu, float lam) {
    float C[9]; for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J=1.f, ps=0.f;
    for (int k=0;k<3;k++){ float S=fmaxf(sqrtf(fmaxf(l[k],1e-12f)),0.01f); J*=S; ps += (S-1.f)*(S-1.f); }
    return mu*ps + 0.5f*lam*(J-1.f)*(J-1.f);
}
// local energies of tets touching a color-cc node at U_n - a dl_n, a in {0, 1, 1/2, ..., 2^-(K-2)} -> Eo[n*K + k] (double atomics)
__global__ void vbd_tet_e(int T, const long long* tets, const int* color, int cc, const float* U, const float* dl, const float* gl, const float* FT,
                          const float* VT, const float* M4, const float* B4, float mu, float lam, int K, double* Eo) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    long long n[4]; int ii = -1;
    for (int i=0;i<4;i++){ n[i]=tets[t*4+i]; if (color[n[i]]==cc) ii=i; }
    if (ii < 0) return;
    float u0[4][3];
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) u0[i][c]=U[n[i]*3+c];
    float dd[3] = {dl[n[ii]*3], dl[n[ii]*3+1], dl[n[ii]*3+2]};
    if (dd[0]==0.f && dd[1]==0.f && dd[2]==0.f) return;
    const float* G = gl + t*12; const float* Fb = FT + t*9; float V = VT[t];
    for (int k=0;k<K;k++) {
        float al = (k==0) ? 0.f : ldexpf(1.f, -(k-1));
        float u[4][3];
        for (int i=0;i<4;i++) for (int c=0;c<3;c++) u[i][c] = u0[i][c] - ((i==ii) ? al*dd[c] : 0.f);
        // inertia: only terms involving node ii matter for the comparison: 0.5 M_ii |u_ii|^2 + sum_{j!=ii} M_ij u_ii.u_j - B_ii.u_ii
        float e = 0.f;
        for (int c=0;c<3;c++) {
            float s = 0.5f*M4[t*16+ii*4+ii]*u[ii][c];
            for (int j=0;j<4;j++) if (j != ii) s += M4[t*16+ii*4+j]*u[j][c];
            e += s*u[ii][c] - B4[t*12+ii*3+c]*u[ii][c];
        }
        if (V > 0.f) {
            float A[9];
            for (int c=0;c<3;c++) for (int b=0;b<3;b++){ float s=(c==b)?1.f:0.f; for (int i=0;i<4;i++) s += u[i][c]*G[i*3+b]; A[c*3+b]=s; }
            float X[9]; mm3(A, Fb, X);
            e += V*psi_f(X, mu, lam);
        }
        atomicAdd(&Eo[n[ii]*K + k], (double)e);
    }
}
__global__ void vbd_near_e(int nn_, const long long* idx, const float* x, const float* U, const long long* rows, const float* lamb,
                           const float* glp, const float* Fp, const float* vol, const float* kk, const float* ccv,
                           double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn,
                           float mu, float lam, const int* color, int cc, const float* dl, int K, double* Eo) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = idx[q];
    long long nd[4]; int ii=-1;
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; if (color[nd[i]]==cc) ii=i; }
    if (ii < 0) return;
    float dd[3] = {dl[nd[ii]*3], dl[nd[ii]*3+1], dl[nd[ii]*3+2]};
    if (dd[0]==0.f && dd[1]==0.f && dd[2]==0.f) return;
    float lm[4], gl[12], xx[3], c[3], F[9];
    for (int i=0;i<4;i++) lm[i]=lamb[p*4+i];
    for (int k=0;k<12;k++) gl[k]=glp[p*12+k];
    for (int k=0;k<3;k++){ xx[k]=x[p*3+k]; c[k]=ccv[p*3+k]; }
    for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    float pt[3]={(float)p0,(float)p1,(float)p2}, nv[3]={(float)n0,(float)n1,(float)n2};
    float V = vol[p], k_ = kk[p];
    for (int k=0;k<K;k++) {
        float al = (k==0) ? 0.f : ldexpf(1.f, -(k-1));
        float u[3]={0,0,0}, G[9]={0,0,0,0,0,0,0,0,0};
        for (int i=0;i<4;i++) for (int a=0;a<3;a++){ float Ua=U[nd[i]*3+a] - ((i==ii) ? al*dd[a] : 0.f); u[a]+=lm[i]*Ua; for (int b=0;b<3;b++) G[a*3+b]+=Ua*gl[i*3+b]; }
        NearMapT<float> Mp; near_map<float>(xx, u, G, c, pt, nv, (float)eps_st, (float)eps_sp, (float)xn, Mp);
        float Jf[9]; mm3(Mp.Mn, Mp.K1, Jf);
        float X[9]; mm3(Jf, F, X);
        float rr = Mp.r[0]*Mp.r[0]+Mp.r[1]*Mp.r[1]+Mp.r[2]*Mp.r[2];
        atomicAdd(&Eo[nd[ii]*K + k], (double)(0.5f*k_*rr + V*psi_f(X, mu, lam)));
    }
}
// pick the largest a with E(a) <= E(0) (node floor terms added here when fl != 0), U -= a dl, reset Eo
__global__ void vbd_node_ls(int M, const int* color, int cc, const float* dl, double* Eo, int K, float* U, int* nrej,
                            int fl, const double* cn, const double* w, const double* kb, const double* ks, double n0, double n1, double n2, double dhat) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= M || color[m] != cc) return;
    double nv[3] = {n0, n1, n2};
    int best = -1; double e0 = 0;
    for (int k=0;k<K;k++) {
        double al = (k==0) ? 0.0 : ldexp(1.0, -(k-1));
        double e = Eo[m*K+k];
        if (fl) {
            double u[3]; for (int c=0;c<3;c++) u[c] = (double)U[m*3+c] - al*(double)dl[m*3+c];
            double un = u[0]*nv[0]+u[1]*nv[1]+u[2]*nv[2];
            double b, b1; barrier3_d(cn[m] + un, dhat, &b, &b1);
            double pu = 0; for (int c=0;c<3;c++){ double t = u[c]-un*nv[c]; pu += t*t; }
            e += kb[m]*b + 0.5*ks[m]*w[m]*pu;
        }
        if (k==0) e0 = e; else if (best < 0 && e <= e0) best = k;
        Eo[m*K+k] = 0;
    }
    if (best < 0) { if (dl[m*3]!=0.f || dl[m*3+1]!=0.f || dl[m*3+2]!=0.f) atomicAdd(nrej, 1); return; }
    float al = ldexpf(1.f, -(best-1));
    for (int c=0;c<3;c++) U[m*3+c] -= al*dl[m*3+c];
}
// ---------------- float32 PCG (+ optional float64 reliable updates = mixed-precision refinement), persistent kernel ----------------
// row of y = A v with float32 vector (sym = half storage: self + positive offsets 0..7, negative offsets = neighbor's transpose)
__device__ __forceinline__ float ell_row_f(int m, int c, int Ms, const float* B, const int* nb, const float* v, int sym) {
    float y = 0.f;
    if (!sym) {
        for (int k=0;k<15;k++) {
            int j = nb[k*Ms+m];
            const float* bb = B + ((long long)k*9 + c*3)*Ms + m;
            y += bb[0]*__ldcg(&v[j*3]) + bb[Ms]*__ldcg(&v[j*3+1]) + bb[2*Ms]*__ldcg(&v[j*3+2]);
        }
        return y;
    }
    for (int k=0;k<8;k++) {
        int j = nb[k*Ms+m]; if (j < 0) continue;
        const float* bb = B + ((long long)k*9 + c*3)*Ms + m;
        y += bb[0]*__ldcg(&v[j*3]) + bb[Ms]*__ldcg(&v[j*3+1]) + bb[2*Ms]*__ldcg(&v[j*3+2]);
    }
    for (int k=1;k<8;k++) {
        int j = nb[(7+k)*Ms+m]; if (j < 0) continue;
        const float* bb = B + ((long long)k*9 + c)*Ms + j;
        y += bb[0]*__ldcg(&v[j*3]) + bb[3*(long long)Ms]*__ldcg(&v[j*3+1]) + bb[6*(long long)Ms]*__ldcg(&v[j*3+2]);
    }
    return y;
}
__global__ void bell_mv_f(int M, const float* B, const int* nb, const float* v, float* y, int Ms, int sym) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= M) return;
    for (int c=0;c<3;c++) y[m*3+c] = ell_row_f(m, c, Ms, B, nb, v, sym);
}
// mode 0: pure float32 PCG (stop on recursive float32 residual).
// mode 1: float32 PCG with float64 reliable updates (x_hi += x_lo; r = b - A x_hi in float64) whenever |r|^2 < delta2 * max|r|^2 since
//         the last update, and always before stopping: the stop test |r|^2 < thr is made on the float64 true residual only.
// sc in: [0] rz0 [1] dAd0 ; out: [2] nit [3] rr (true when mode 1) [4] neg [5] n_updates.   acc: 6 x 2 parities (+ 2 for update sums)
__device__ __forceinline__ void pcg_mix_body(int M, int Ms, const float* B, const int* nb, const float* Dinv, const double* bs, double* xh, double* rh,
                        float* x, float* r, float* z, float* d, float* Ad, double* acc, double* sc, unsigned int* bar,
                        double thr, int maxit, int sym, int mode, double delta2, int rows) {
    unsigned int nblk = gridDim.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x, nth = gridDim.x*blockDim.x;
    volatile double* va = acc;
    double rz = sc[0], dAd = sc[1];
    int nit = 0, neg = 0, p = 0, nup = 0, q = 0;
    double rr = 1.0, rmax = 1.0;
    for (int it=0; it<maxit; it++) {
        if (!(dAd > 0)) { neg = 1; break; }
        float al = (float)(rz/dAd);
        float prr=0.f, prz=0.f;
        for (int m=tid; m<M; m+=nth) {
            float rv[3];
            for (int c=0;c<3;c++){ x[m*3+c] += al*d[m*3+c]; float rc = r[m*3+c] - al*Ad[m*3+c]; r[m*3+c]=rc; rv[c]=rc; }
            const float* D = Dinv + m*9;
            for (int c=0;c<3;c++){ float zz = D[c*3]*rv[0]+D[c*3+1]*rv[1]+D[c*3+2]*rv[2]; z[m*3+c]=zz; prz += rv[c]*zz; prr += rv[c]*rv[c]; }
        }
        double b1 = block_sum((double)prr); double b2 = block_sum((double)prz);
        if (threadIdx.x==0) { atomicAdd(&acc[p*3+1], b1); atomicAdd(&acc[p*3+2], b2); }
        if (tid==0) { acc[(1-p)*3]=0; acc[(1-p)*3+1]=0; acc[(1-p)*3+2]=0; }
        gbar(bar, bar+1, nblk);
        rr = va[p*3+1]; double rz2 = va[p*3+2];
        nit++;
        if (mode == 1 && (rr < delta2*rmax || rr < thr)) {
            // reliable update: x_hi += x_lo, r = b - A x_hi (float64 accumulation of the float32-stored matrix), z = D^-1 r
            for (int m=tid; m<M; m+=nth) for (int c=0;c<3;c++){ xh[m*3+c] += (double)x[m*3+c]; x[m*3+c] = 0.f; }
            if (tid==0) { acc[6+(1-q)*2]=0; acc[6+(1-q)*2+1]=0; }
            gbar(bar, bar+1, nblk);
            double trr=0, trz=0;
            for (int m=tid; m<M; m+=nth) {
                double rv[3];
                for (int c=0;c<3;c++) {
                    double ax = sym ? ell_row_sym<true>(m, c, Ms, B, nb, xh) : ell_row<true>(m, c, Ms, B, nb, xh);
                    rv[c] = bs[m*3+c] - ax; rh[m*3+c] = rv[c]; r[m*3+c] = (float)rv[c];
                }
                const float* D = Dinv + m*9;
                for (int c=0;c<3;c++){ float zz = D[c*3]*(float)rv[0]+D[c*3+1]*(float)rv[1]+D[c*3+2]*(float)rv[2]; z[m*3+c]=zz; trz += rv[c]*(double)zz; trr += rv[c]*rv[c]; }
            }
            double c1 = block_sum(trr); double c2 = block_sum(trz);
            if (threadIdx.x==0) { atomicAdd(&acc[6+q*2], c1); atomicAdd(&acc[6+q*2+1], c2); }
            gbar(bar, bar+1, nblk);
            rr = va[6+q*2]; rz2 = va[6+q*2+1];
            q = 1-q; nup++;
            rmax = rr;
            if (rr < thr) break;
        } else if (mode == 0 && rr < thr) {
            break;
        } else if (rr > rmax) rmax = rr;
        double be = rz2/rz; rz = rz2;
        float bef = (float)be;
        float pd = 0.f;
        if (rows) {
            // one thread per (row c, node m), c-major so that B reads coalesce across m
            for (int w2=tid; w2<3*M; w2+=nth) {
                int c = w2 / M, m = w2 - c*M;
                float qq = ell_row_f(m, c, Ms, B, nb, z, sym);
                float dd = z[m*3+c] + bef*d[m*3+c]; float aa = qq + bef*Ad[m*3+c]; d[m*3+c]=dd; Ad[m*3+c]=aa; pd += dd*aa;
            }
        } else {
        for (int m=tid; m<M; m+=nth) {
            float qq[3];
            for (int c=0;c<3;c++) qq[c] = ell_row_f(m, c, Ms, B, nb, z, sym);
            for (int c=0;c<3;c++){ float dd = z[m*3+c] + bef*d[m*3+c]; float aa = qq[c] + bef*Ad[m*3+c]; d[m*3+c]=dd; Ad[m*3+c]=aa; pd += dd*aa; }
        }
        }
        double b3 = block_sum((double)pd);
        if (threadIdx.x==0) atomicAdd(&acc[(1-p)*3], b3);
        gbar(bar, bar+1, nblk);
        dAd = va[(1-p)*3];
        p = 1-p;
    }
    if (tid==0) { sc[2]=nit; sc[3]=rr; sc[4]=neg; sc[5]=nup; }
}
}  // extern "C" (templates below need C++ linkage)
// ---------------- tiled block-ELL matvec: a block of 512 threads = 16 warps handles a tile of 32 nodes; warp k (< 15) takes neighbor slot k
// (coalesced over the 32 nodes), partial 3-vectors are summed over slots in shared memory. Much more parallelism than one thread per node.
template <typename V> __device__ __forceinline__ void tile_part(int m, int k, int M, int Ms, const float* B, const int* nb, const V* v, int sym,
                                                                V& a0, V& a1, V& a2) {
    a0 = 0; a1 = 0; a2 = 0;
    if (m >= M || k >= 15) return;
    int j = nb[k*Ms+m];
    if (j < 0) return;
    V x0 = __ldcg(&v[j*3]), x1 = __ldcg(&v[j*3+1]), x2 = __ldcg(&v[j*3+2]);
    if (!sym || k < 8) {
        const float* bb = B + (long long)k*9*Ms + m;
        a0 = (V)bb[0]*x0 + (V)bb[Ms]*x1 + (V)bb[2*(long long)Ms]*x2;
        a1 = (V)bb[3*(long long)Ms]*x0 + (V)bb[4*(long long)Ms]*x1 + (V)bb[5*(long long)Ms]*x2;
        a2 = (V)bb[6*(long long)Ms]*x0 + (V)bb[7*(long long)Ms]*x1 + (V)bb[8*(long long)Ms]*x2;
    } else {
        const float* bb = B + (long long)(k-7)*9*Ms + j;          // transpose of neighbor's positive block
        a0 = (V)bb[0]*x0 + (V)bb[3*(long long)Ms]*x1 + (V)bb[6*(long long)Ms]*x2;
        a1 = (V)bb[Ms]*x0 + (V)bb[4*(long long)Ms]*x1 + (V)bb[7*(long long)Ms]*x2;
        a2 = (V)bb[2*(long long)Ms]*x0 + (V)bb[5*(long long)Ms]*x1 + (V)bb[8*(long long)Ms]*x2;
    }
}
template <typename V> __device__ void bell_mv_tile_body(int M, const float* B, const int* nb, const V* v, V* y, int Ms, int sym) {
    __shared__ V sy[16][3][33];
    int lane = threadIdx.x & 31, k = threadIdx.x >> 5;
    int m = blockIdx.x*32 + lane;
    V a0, a1, a2; tile_part<V>(m, k, M, Ms, B, nb, v, sym, a0, a1, a2);
    sy[k][0][lane]=a0; sy[k][1][lane]=a1; sy[k][2][lane]=a2;
    __syncthreads();
    if (k < 3 && m < M) { V s=0; for (int q=0;q<15;q++) s += sy[q][k][lane]; y[m*3+k] = s; }
}
// persistent PCG on the tiled matvec. V = float: float32 vectors (mode 1 = float64 reliable updates, mode 0 = float32 alone);
// V = double: float64 vectors (mode 0 = plain float64 PCG, same algebra as pcg_coop1). blockDim must be 512.
template <typename V> __device__ void pcg_tile_body(int M, int Ms, const float* B, const int* nb, const V* Dinv, const double* bs, double* xh,
                        V* x, V* r, V* z, V* d, V* Ad, double* acc, double* sc, unsigned int* bar, double thr, int maxit, int sym, int mode, double delta2) {
    __shared__ V sy[16][3][33];
    unsigned int nblk = gridDim.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x, nth = gridDim.x*blockDim.x;
    int lane = threadIdx.x & 31, kw = threadIdx.x >> 5;
    int ntile = (M + 31) / 32;
    volatile double* va = acc;
    double rz = sc[0], dAd = sc[1];
    int nit = 0, neg = 0, p = 0, nup = 0, q = 0;
    double rr = 1.0, rmax = 1.0;
    for (int it=0; it<maxit; it++) {
        if (!(dAd > 0)) { neg = 1; break; }
        V al = (V)(rz/dAd);
        V prr=0, prz=0;
        for (int m=tid; m<M; m+=nth) {
            V rv[3];
            for (int c=0;c<3;c++){ x[m*3+c] += al*d[m*3+c]; V rc = r[m*3+c] - al*Ad[m*3+c]; r[m*3+c]=rc; rv[c]=rc; }
            const V* D = Dinv + m*9;
            for (int c=0;c<3;c++){ V zz = D[c*3]*rv[0]+D[c*3+1]*rv[1]+D[c*3+2]*rv[2]; z[m*3+c]=zz; prz += rv[c]*zz; prr += rv[c]*rv[c]; }
        }
        double b1 = block_sum((double)prr); double b2 = block_sum((double)prz);
        if (threadIdx.x==0) { atomicAdd(&acc[p*3+1], b1); atomicAdd(&acc[p*3+2], b2); }
        if (tid==0) { acc[(1-p)*3]=0; acc[(1-p)*3+1]=0; acc[(1-p)*3+2]=0; }
        gbar(bar, bar+1, nblk);
        rr = va[p*3+1]; double rz2 = va[p*3+2];
        nit++;
        if (mode == 1 && (rr < delta2*rmax || rr < thr)) {
            for (int m=tid; m<M; m+=nth) for (int c=0;c<3;c++){ xh[m*3+c] += (double)x[m*3+c]; x[m*3+c] = 0; }
            if (tid==0) { acc[6+(1-q)*2]=0; acc[6+(1-q)*2+1]=0; }
            gbar(bar, bar+1, nblk);
            double trr=0, trz=0;
            for (int m=tid; m<M; m+=nth) {
                double rv[3];
                for (int c=0;c<3;c++) {
                    double ax = sym ? ell_row_sym<true>(m, c, Ms, B, nb, xh) : ell_row<true>(m, c, Ms, B, nb, xh);
                    rv[c] = bs[m*3+c] - ax; r[m*3+c] = (V)rv[c];
                }
                const V* D = Dinv + m*9;
                for (int c=0;c<3;c++){ V zz = D[c*3]*(V)rv[0]+D[c*3+1]*(V)rv[1]+D[c*3+2]*(V)rv[2]; z[m*3+c]=zz; trz += rv[c]*(double)zz; trr += rv[c]*rv[c]; }
            }
            double c1 = block_sum(trr); double c2 = block_sum(trz);
            if (threadIdx.x==0) { atomicAdd(&acc[6+q*2], c1); atomicAdd(&acc[6+q*2+1], c2); }
            gbar(bar, bar+1, nblk);
            rr = va[6+q*2]; rz2 = va[6+q*2+1];
            q = 1-q; nup++;
            rmax = rr;
            if (rr < thr) break;
        } else if (mode != 1 && rr < thr) {
            break;
        } else if (rr > rmax) rmax = rr;
        double be = rz2/rz; rz = rz2;
        V bev = (V)be;
        V pd = 0;
        for (int tile = blockIdx.x; tile < ntile; tile += gridDim.x) {
            int m = tile*32 + lane;
            V a0, a1, a2; tile_part<V>(m, kw, M, Ms, B, nb, z, sym, a0, a1, a2);
            sy[kw][0][lane]=a0; sy[kw][1][lane]=a1; sy[kw][2][lane]=a2;
            __syncthreads();
            if (kw < 3 && m < M) {
                V s=0; for (int qq=0;qq<15;qq++) s += sy[qq][kw][lane];
                int e = m*3+kw;
                V dd = z[e] + bev*d[e]; V aa = s + bev*Ad[e]; d[e]=dd; Ad[e]=aa; pd += dd*aa;
            }
            __syncthreads();
        }
        double b3 = block_sum((double)pd);
        if (threadIdx.x==0) atomicAdd(&acc[(1-p)*3], b3);
        gbar(bar, bar+1, nblk);
        dAd = va[(1-p)*3];
        p = 1-p;
    }
    if (tid==0) { sc[2]=nit; sc[3]=rr; sc[4]=neg; sc[5]=nup; }
}
extern "C" {
__global__ void bell_mv_t32(int M, const float* B, const int* nb, const float* v, float* y, int Ms, int sym) { bell_mv_tile_body<float>(M, B, nb, v, y, Ms, sym); }
__global__ void bell_mv_t64(int M, const float* B, const int* nb, const double* v, double* y, int Ms, int sym) { bell_mv_tile_body<double>(M, B, nb, v, y, Ms, sym); }
#define PCG_T_ARGS(V) int M, int Ms, const float* B, const int* nb, const V* Dinv, const double* bs, double* xh, V* x, V* r, V* z, V* d, V* Ad, \
    double* acc, double* sc, unsigned int* bar, double thr, int maxit, int sym, int mode, double delta2
__global__ void pcg_t32(PCG_T_ARGS(float)) { pcg_tile_body<float>(M, Ms, B, nb, Dinv, bs, xh, x, r, z, d, Ad, acc, sc, bar, thr, maxit, sym, mode, delta2); }
__global__ void pcg_t64(PCG_T_ARGS(double)) { pcg_tile_body<double>(M, Ms, B, nb, Dinv, bs, xh, x, r, z, d, Ad, acc, sc, bar, thr, maxit, sym, mode, delta2); }
// micro benchmark: nit iterations of (per-thread work over M rows, 2 block sums + atomics + grid barrier) x 2 -- the PCG skeleton without the math
__global__ void bar_only(int M, int nit, double* acc, unsigned int* bar, float* v) {
    unsigned int nblk = gridDim.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x, nth = gridDim.x*blockDim.x;
    int p = 0;
    for (int it=0; it<nit; it++) {
        float s = 0.f;
        for (int m=tid; m<M; m+=nth) s += v[m];
        double b1 = block_sum((double)s); double b2 = block_sum((double)s);
        if (threadIdx.x==0) { atomicAdd(&acc[p*3+1], b1); atomicAdd(&acc[p*3+2], b2); }
        if (tid==0) { acc[(1-p)*3]=0; acc[(1-p)*3+1]=0; acc[(1-p)*3+2]=0; }
        gbar(bar, bar+1, nblk);
        double b3 = block_sum((double)s);
        if (threadIdx.x==0) atomicAdd(&acc[(1-p)*3], b3);
        gbar(bar, bar+1, nblk);
        p = 1-p;
    }
}
#define PCG_MIX_ARGS int M, int Ms, const float* B, const int* nb, const float* Dinv, const double* bs, double* xh, double* rh, \
    float* x, float* r, float* z, float* d, float* Ad, double* acc, double* sc, unsigned int* bar, double thr, int maxit, int sym, int mode, double delta2, int rows
#define PCG_MIX_CALL pcg_mix_body(M, Ms, B, nb, Dinv, bs, xh, rh, x, r, z, d, Ad, acc, sc, bar, thr, maxit, sym, mode, delta2, rows)
__global__ void pcg_mix(PCG_MIX_ARGS) { PCG_MIX_CALL; }
__global__ void __launch_bounds__(256, 2) pcg_mix_lb2(PCG_MIX_ARGS) { PCG_MIX_CALL; }
__global__ void __launch_bounds__(256, 3) pcg_mix_lb3(PCG_MIX_ARGS) { PCG_MIX_CALL; }
__global__ void __launch_bounds__(256, 4) pcg_mix_lb4(PCG_MIX_ARGS) { PCG_MIX_CALL; }
}

//@@ _EX 96695

// 진단 전용 배율 (절 N.2, --near_diag): [0] 바닥 후보 관성 보정 항 m/h² (J_qᵀ r - (u - c)), [1] 변위 쪽 탄성 항 pj, [2] 변위 기울기 쪽 탄성 항 gG. 기본 1 (원래 식)
__device__ float EXD[3] = {1.f, 1.f, 1.f};
template <typename R, typename S>
__device__ __forceinline__ void ex_tet_force_body(int t, const long long* tets, const float* U, const S* gl, const S* FT, const S* VT, R mu, R lam, float* f) {
    R V = (R)VT[t];
    if (!(V > (R)0)) return;
    long long n[4]; R G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=(R)gl[t*12+k];
    R A[9] = {1,0,0, 0,1,0, 0,0,1};
    if (U) {
        for (int i=0;i<4;i++) for (int c=0;c<3;c++) { R u=(R)U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    }
    R Fb[9]; for (int k=0;k<9;k++) Fb[k]=(R)FT[t*9+k];
    R X[9]; mm3(A, Fb, X);
    R C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    R l[3]; symeig3(C, l);
    R J = 1;
    for (int k=0;k<3;k++){ R s = sqrt(l[k] > (R)1e-12 ? l[k] : (R)1e-12); s = s > (R)0.01 ? s : (R)0.01; J *= s; }
    R Q[9], cf[9], P[9], Gm[9];
    polar(X, Q); cof3(X, cf);
    R dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    R sg = (dt>0)?(R)1:((dt<0)?(R)-1:(R)0);
    for (int k=0;k<9;k++) P[k] = (R)2*mu*(X[k]-Q[k]) + lam*(J-(R)1)*sg*cf[k];
    mmt3(P, Fb, Gm);
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        R s = 0; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], (float)(-V*s));
    }
}

template <typename R> __device__ __forceinline__ void ex_near_force_body(int q, int nn_, const long long* nidx, const float* x0c, const float* F0c, const float* cc,
                              const float* Ucum, const float* xcur, const float* Fcur, const float* F0inv,
                              const long long* rows, const float* lamb, const float* glT, const long long* tinv, const float* mp, const float* vol, R h,
                              R p0, R p1, R p2, R n0, R n1, R n2, R eps_st, R eps_sp, R xn, R mu, R lam, float* gn) {
    if (q >= nn_) return;
    long long p = nidx[q];
    long long nd[4]; R lm[4], gl[12], u[3]={0,0,0}, G[9]={0,0,0,0,0,0,0,0,0}, xx[3], c[3], IGt[9]={1,0,0,0,1,0,0,0,1};
    const float* glp = glT + tinv[p]*12;
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; }
    for (int k=0;k<12;k++) gl[k]=glp[k];
    for (int k=0;k<3;k++){ xx[k]=x0c[q*3+k]; c[k]=cc[q*3+k]; }
    if (Ucum) {
        for (int i=0;i<4;i++) for (int a=0;a<3;a++){ R Ua=(R)Ucum[nd[i]*3+a]; u[a]+=lm[i]*Ua; for (int b=0;b<3;b++) G[a*3+b]+=Ua*gl[i*3+b]; }
    } else {
        for (int k=0;k<3;k++) u[k] = (R)xcur[p*3+k] - xx[k];
        R Fc[9], Fi[9]; for (int k=0;k<9;k++){ Fc[k]=Fcur[p*9+k]; Fi[k]=F0inv[q*9+k]; }
        mm3(Fc, Fi, IGt);
        for (int k=0;k<9;k++) G[k] = IGt[k] - ((k%4==0)?(R)1:(R)0);
    }
    R pt[3]={p0,p1,p2}, nv[3]={n0,n1,n2};
    NearMapT<R> M; near_map(xx, u, G, c, pt, nv, eps_st, eps_sp, xn, M);
    R Jf[9]; mm3(M.Mn, M.K1, Jf);
    R F[9]; for (int k=0;k<9;k++) F[k]=F0c[q*9+k];
    R X[9]; mm3(Jf, F, X);
    R C[9]; for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    R l[3]; symeig3(C, l);
    R J=1; for (int k=0;k<3;k++){ R s=fmax(sqrt(fmax(l[k],(R)1e-12)),(R)0.01); J*=s; }
    R V = vol[p], k_ = (R)mp[p]/(h*h);
    R Q[9], cf[9], P[9], Pc[9];
    polar(X, Q); cof3(X, cf);
    R dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    R sg = (dt>0)?(R)1:((dt<0)?(R)-1:(R)0);
    for (int k=0;k<9;k++) P[k] = (R)2*mu*(X[k]-Q[k]) + lam*(J-(R)1)*sg*cf[k];
    mmt3(P, F, Pc); for (int k=0;k<9;k++) Pc[k]*=V;
    R gy[12];
    R NK1[9]; mm3(M.Nm, M.K1, NK1);
    for (int c3=0;c3<3;c3++) {
        R t=0; for (int i=0;i<3;i++) t += M.Ru[i*3+c3]*M.r[i];
        t -= u[c3] - c[c3];                                            // - 매핑 없는 관성 (노드 질량 동역학이 이미 갖는 몫)
        R Bc[9], dK[9], MdK[9]; near_Bc(M, c3, Bc); mm3(Bc, M.IG, dK);
        for (int k=0;k<9;k++) dK[k] -= xn*M.w1*M.n[c3]*M.Pm[k];
        mm3(M.Mn, dK, MdK);
        R pj=0; for (int k=0;k<9;k++) pj += Pc[k]*(M.g1*M.n[c3]*NK1[k] + MdK[k]);
        gy[c3] = (R)EXD[0]*k_*t + (R)EXD[1]*pj;
    }
    R MJ[9]; mm3(M.Mn, M.Jst, MJ);
    R gG[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++){ R t=0; for (int i=0;i<3;i++) t += MJ[i*3+a]*Pc[i*3+b]; gG[a*3+b]=t; }
    if (!Ucum) { R T2[9]; mmt3(gG, IGt, T2); for (int k=0;k<9;k++) gG[k]=T2[k]; }     // sub: ∂/∂δU 는 gG (I+G)ᵀ
    for (int k=0;k<9;k++) gy[3+k]=(R)EXD[2]*gG[k];
    for (int i=0;i<4;i++) for (int c3=0;c3<3;c3++) {
        R t = lm[i]*gy[c3];
        for (int b=0;b<3;b++) t += gl[i*3+b]*gy[3+c3*3+b];
        atomicAdd(&gn[nd[i]*3+c3], (float)t);
    }
}


extern "C" {
// szp (선택): 장치 크기 [M, T, nn, 넘침] -- 그래프 재생에서 프레임마다 바뀌는 크기를 장치에서 읽는다 (없으면 인자 값)
__global__ void ex_tet_force_dd(const int* szp, int Targ, const long long* tets, const float* U, const double* gl, const double* FT, const double* VT,
                                double mu, double lam, float* f) {
    int T = szp ? szp[1] : Targ;
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < T) ex_tet_force_body<double, double>(t, tets, U, gl, FT, VT, mu, lam, f);
}
__global__ void ex_tet_force_ff(const int* szp, int Targ, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                                float mu, float lam, float* f) {
    int T = szp ? szp[1] : Targ;
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < T) ex_tet_force_body<float, float>(t, tets, U, gl, FT, VT, mu, lam, f);
}

__global__ void ex_node_step(const int* szp, int Marg, const float* mn, const float* f, const float* gn, float* vn, float* dU, float* Ucum,
                             float g0, float g1, float g2, float hs, float damp, const unsigned char* fixd, const float* vbc) {
    int M = szp ? szp[0] : Marg;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float gg[3] = {g0, g1, g2};
    float m = mn[n];
    for (int c=0;c<3;c++) {
        float v;
        if (fixd && fixd[n]) v = vbc[n*3+c];
        else if (m > 0.f) v = (vn[n*3+c] + hs*((f[n*3+c] - (gn ? gn[n*3+c] : 0.f))/m + gg[c]))*damp;
        else v = 0.f;
        vn[n*3+c] = v; dU[n*3+c] = hs*v;
        if (Ucum) Ucum[n*3+c] += hs*v;
    }
}

// 바닥 후보 입자의 프레임 바닥 에너지 (기존 정의, 프레임 걸음 h·프레임 시작 x0·F0·c = h v + h² g 그대로):
//   E_fl = m/(2h²)(|q(x0+u) - x0 - c|² - |u - c|²) + V ψ(J_f(u, G) F0)
//   (매핑이 항등이면 첫 항 0; 탄성은 이 입자들이 사면체 F̄ 에서 빠져 있으므로 여기서), u·G = 프레임 시작부터의 변위·변위 기울기
//   lag: u = Σλ U_cum, G = Σ U_cum ⊗ ∇λ (변수 U_cum)       sub: u = x - x0, G = F F0⁻¹ - I (변수 = 지금 서브스텝 노드 변위 δU, δG = D (I + G))
//   기울기를 노드로 gn (힘 = -gn) -- 서브스텝마다 지금 상태에서 (명시적)
__global__ void ex_near_force(const int* szp, int nnarg, const long long* nidx, const float* x0c, const float* F0c, const float* cc,
                              const float* Ucum, const float* xcur, const float* Fcur, const float* F0inv,
                              const long long* rows, const float* lamb, const float* glT, const long long* tinv, const float* mp, const float* vol, double h,
                              double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn,
                              double mu, double lam, float* gn) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    ex_near_force_body<double>(q, szp ? szp[2] : nnarg, nidx, x0c, F0c, cc, Ucum, xcur, Fcur, F0inv, rows, lamb, glT, tinv, mp, vol, h,
                               p0, p1, p2, n0, n1, n2, eps_st, eps_sp, xn, mu, lam, gn);
}
// float32 대수 판 (암시적 판 --near32 / --eg32 와 같은 방식)
__global__ void ex_near_force32(const int* szp, int nnarg, const long long* nidx, const float* x0c, const float* F0c, const float* cc,
                              const float* Ucum, const float* xcur, const float* Fcur, const float* F0inv,
                              const long long* rows, const float* lamb, const float* glT, const long long* tinv, const float* mp, const float* vol, double h,
                              double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn,
                              double mu, double lam, float* gn) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    ex_near_force_body<float>(q, szp ? szp[2] : nnarg, nidx, x0c, F0c, cc, Ucum, xcur, Fcur, F0inv, rows, lamb, glT, tinv, mp, vol, (float)h,
                              (float)p0, (float)p1, (float)p2, (float)n0, (float)n1, (float)n2, (float)eps_st, (float)eps_sp, (float)xn, (float)mu, (float)lam, gn);
}

// 프레임 끝 바닥 후보 입자: 기존 G2P 매핑을 프레임 변위에 한 번 (암시적 판 G2P 와 같은 식) -- x2 = q(x0 + u), F2 = J_f F0,
// v2 = (선형 G2P 가 이미 쓴 v2) + (q - x0 - u)/h  (매핑이 옮긴 만큼을 프레임 속도로)
__global__ void ex_near_fin(const int* szp, int nnarg, const long long* nidx, const float* x0c, const float* F0c,
                            const float* Ucum, const float* xcur, const float* Fcur, const float* F0inv,
                            const long long* rows, const float* lamb, const float* glT, const long long* tinv,
                            double p0, double p1, double p2, double n0, double n1, double n2, double eps_st, double eps_sp, double xn, double h,
                            float* x2, float* v2, float* F2) {
    int nn_ = szp ? szp[2] : nnarg;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = nidx[q];
    double u[3]={0,0,0}, G[9]={0,0,0,0,0,0,0,0,0}, xx[3], c0[3]={0,0,0};
    for (int k=0;k<3;k++) xx[k]=x0c[q*3+k];
    if (Ucum) {
        const float* gl = glT + tinv[p]*12;
        for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; double lm=lamb[p*4+i]; for (int a=0;a<3;a++){ double Ua=Ucum[nd*3+a]; u[a]+=lm*Ua; for (int b=0;b<3;b++) G[a*3+b]+=Ua*gl[i*3+b]; } }
    } else {
        for (int k=0;k<3;k++) u[k] = (double)xcur[p*3+k] - xx[k];
        double Fc[9], Fi[9], IG[9]; for (int k=0;k<9;k++){ Fc[k]=Fcur[p*9+k]; Fi[k]=F0inv[q*9+k]; }
        mm3(Fc, Fi, IG);
        for (int k=0;k<9;k++) G[k] = IG[k] - ((k%4==0)?1.0:0.0);
    }
    double pt[3]={p0,p1,p2}, nv[3]={n0,n1,n2};
    NearMap M; near_map(xx, u, G, c0, pt, nv, eps_st, eps_sp, xn, M);
    double Jf[9]; mm3(M.Mn, M.K1, Jf);
    for (int k=0;k<3;k++) { x2[p*3+k] = (float)(xx[k] + M.r[k]); v2[p*3+k] = (float)((double)v2[p*3+k] + (M.r[k] - u[k])/h); }
    double F[9]; for (int k=0;k<9;k++) F[k]=F0c[q*9+k];
    double F2d[9]; mm3(Jf, F, F2d);
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
}

// (lag) 프레임 끝 선형 G2P: x + Σλ U, v = Σλ v_n, F <- (I + Σ U ⊗ ∇λ) F  (바닥 후보는 뒤이어 ex_near_fin 이 덮어쓴다)
__global__ void ex_g2p_lag(int N, const float* x, const float* Ucum, const float* Vn, const long long* rows, const float* lamb, const float* glT,
                           const long long* tinv, const float* Fp, float* x2, float* v2, float* F2, float* C2, int vq) {
    // C2 (선택, --affine): 입자 속도 기울기 C = (Σ v_n ⊗ ∇λ_n) A⁻¹ (사면체 안 선형 속도장의 현재 배치 기울기) -- 다음 프레임 P2G 의 아핀 전달 (APIC 와 같은 꼴)
    //   (정답 PG 의 APIC C = B D⁻¹, B = Σ w v_i (x_i - x_p)ᵀ, D = Σ w (x_i - x_p)(x_i - x_p)ᵀ 와 같은 값: 선형 사면체 보간은 아핀장에 정확하므로
    //    B = (Σ v_i ⊗ ∇λ_i) D 이고, D 는 입자가 면·모서리 위면 특이하므로 닫힌 꼴로 쓴다)
    // vq = 1 (절 Q (a) --rep_w tet4): 입자 속도 v = ¼ Σ v_n (P2G 사면체 등분 가중 m_p/4 와 같은 가중) -- 위치·F 는 그대로 λ·A
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double u[3]={0,0,0}, vv[3]={0,0,0}, A[9]={1,0,0,0,1,0,0,0,1}, Gv[9]={0,0,0,0,0,0,0,0,0};
    const float* gl = glT + tinv[p]*12;
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; double lm=lamb[p*4+i];
        for (int a=0;a<3;a++){ double Ua=Ucum[nd*3+a]; double Va=(double)Vn[nd*3+a]; u[a]+=lm*Ua; vv[a]+=lm*Va;
            for (int b=0;b<3;b++) { A[a*3+b]+=Ua*gl[i*3+b]; Gv[a*3+b]+=Va*gl[i*3+b]; } } }
    if (vq) { for (int a=0;a<3;a++) { double t=0; for (int i=0;i<4;i++) t += (double)Vn[rows[p*4+i]*3+a]; vv[a] = 0.25*t; } }
    for (int k=0;k<3;k++) { x2[p*3+k] = (float)((double)x[p*3+k] + u[k]); v2[p*3+k] = (float)vv[k]; }
    double F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    double F2d[9]; mm3(A, F, F2d);
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
    if (C2) {
        double cf[9]; cof3(A, cf);
        double dA = A[0]*cf[0]+A[1]*cf[1]+A[2]*cf[2];
        double Ci[9];
        if (fabs(dA) > 1e-12) { for (int i=0;i<3;i++) for (int j=0;j<3;j++) Ci[i*3+j] = cf[j*3+i]/dA; }   // A⁻¹ = cofᵀ / det
        else { for (int k=0;k<9;k++) Ci[k] = (k%4==0)?1.0:0.0; }
        double Cc[9]; mm3(Gv, Ci, Cc);
        for (int k=0;k<9;k++) C2[p*9+k] = (float)Cc[k];
    }
}
// ======== 절 O: lag 판 속도 단위 바닥 (--floor vel, 사용자 10-28 "2번") ========
// 정답 (PG mpm_solver_warp add_surface_collider, surface "sticky"): 격자 노드 (x_n - p)·n < 0 이면 grid_v_out = 0 (세 성분 모두),
//   서브스텝마다 격자 정규화·중력·감쇠 뒤, G2P 앞.
// 여기서는 판정을 **입자** 로: 바닥 후보 입자의 지금 보간 위치 x_p + Σλ U (서브스텝 시작) 의 높이 s = (x - p)·n < 0 이면 접촉.
// 접촉 입자가 걸린 사면체 네 노드에 P2G 와 같은 질량 가중 w_pn (사면체 등분 0.25 m_p / Σλm 판은 λ m_p) 을 쌓는다 -> acc_n.
__global__ void ex_vfl_contact(const int* szp, const long long* nidx, const float* x, const float* Ucum, const long long* rows, const float* lamb,
                               const float* m, int tetmass, float p0, float p1, float p2, float n0, float n1, float n2, float* acc, int* ncon) {
    int nn_ = szp[2];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = nidx[q];
    float u[3] = {0,0,0};
    long long nd[4]; float lm[4];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm[i]*Ucum[nd[i]*3+a]; }
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(s < 0.f)) return;
    float mp = m[p];
    for (int i=0;i<4;i++) atomicAdd(&acc[nd[i]], (tetmass ? 0.25f : lm[i])*mp);
    if (ncon) atomicAdd(ncon, 1);
}
// lag 노드 걸음 (ex_node_step 과 같은 식) + 속도 바닥 + 쓴 뒤 f·gn·acc 비우기 (채우기 커널 없음):
//   v = (v + h_s((f - gn)/m + g))·감쇠  (--cfg_bc 묶인 노드 = v_bc)
//   acc_n > 0 (접촉 입자가 걸린 노드):
//     mode 0 (질량 비율 고착): v <- (1 - α) v,  α = min(acc_n / m_n, 1)  -- 접촉 입자 몫의 운동량을 0 으로 (정답 sticky 의 v = 0 을 접촉 입자 질량에만)
//     mode 1 (노드 전체 고착): v <- 0  -- 정답 sticky 와 같은 노드 속도 0 (판정만 입자)
//     mode 2 (절 AG 미끄럼, --vfl_mode slip): 접선 성분 그대로 (np = 1 이면 안쪽 법선 성분만 제거 -- GF/PG bounding_box 의 한쪽 방향 속도 0)
//     np = 1 (비관통): 그 뒤 v·n < 0 이면 법선 성분 제거 -> 접촉 입자의 보간 속도 Σλ v_n (볼록 결합) 의 법선 성분 ≥ 0
//   U += h_s v
__global__ void ex_node_step_lag(const int* szp, const float* mn, float* f, float* gn, float* vn, float* Ucum, float g0, float g1, float g2, float hs, float damp,
                                 const unsigned char* fixd, const float* vbc, float* acc, int mode, int np_, float n0, float n1, float n2, int nodisp) {
    // nodisp = 1 (절 R --lim): 속도만 쓰고 U 갱신은 뒤이은 색칠 이동 제한 커널 (ex_lim_color) 이 한다
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float gg[3] = {g0, g1, g2};
    float m = mn[n];
    float v[3];
    for (int c=0;c<3;c++) {
        if (fixd && fixd[n]) v[c] = vbc[n*3+c];
        else if (m > 0.f) v[c] = (vn[n*3+c] + hs*((f[n*3+c] - (gn ? gn[n*3+c] : 0.f))/m + gg[c]))*damp;
        else v[c] = 0.f;
        f[n*3+c] = 0.f; if (gn) gn[n*3+c] = 0.f;
    }
    if (acc) {
        float a = acc[n];
        acc[n] = 0.f;
        if (a > 0.f && !(fixd && fixd[n])) {
            float k = (mode == 1) ? 0.f : ((mode == 2) ? 1.f : fmaxf(1.f - (m > 0.f ? a/m : 1.f), 0.f));   // mode 2 (절 AG slip): 접선 그대로, 아래 비관통만 (정답 bounding_box: 바깥 법선 성분만 0)
            for (int c=0;c<3;c++) v[c] *= k;
            if (np_) { float vn_ = v[0]*n0 + v[1]*n1 + v[2]*n2; if (vn_ < 0.f) { v[0] -= vn_*n0; v[1] -= vn_*n1; v[2] -= vn_*n2; } }
        }
    }
    for (int c=0;c<3;c++) { vn[n*3+c] = v[c]; if (!nodisp) Ucum[n*3+c] += hs*v[c]; }
}
// ======== 절 R: lag 판 셀 뒤집힘 방지 (사용자 10-29 "두 가지 다") ========
// (1) FCR + log J 장벽: 에너지 밀도에 κ μ b(J; Ĵ) (phys_resid._barrier_b, 연장 linear: J < 1e-3 Ĵ 에서 값·기울기 이어 붙임) 를 더한다.
//     fbon 0: J = det(A_T F̄_T) (셀 변형 전체 = 기존 F 장벽 꼴), 1: J = det A_T (프레임 안 셀 사상 = 기존 셀 장벽 꼴)
//     b'(J) = -2 (J - Ĵ) ln(J/Ĵ) - (J - Ĵ)² / J  (J < Ĵ), 0 (J ≥ Ĵ)
__device__ __forceinline__ float bar_db(float d, float jh) {
    if (!(d < jh)) return 0.f;
    float eps = 1e-3f*jh;
    float dc = d < eps ? eps : d;
    float k = dc - jh, l = logf(dc/jh);
    return -2.f*k*l - k*k/dc;
}
// 사면체 힘 (ex_tet_force_body<float> 와 같은 FCR 식) + 장벽 + 뒤집힘 계수 rst[0] += 1 (점유 사면체, det A_T ≤ 0 인 서브스텝마다)
__global__ void ex_tet_force_R(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                               float mu, float lam, float* f, float fbk, float fbj, int fbon, float* rst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    float cA[9]; cof3(A, cA);
    float dA = A[0]*cA[0]+A[1]*cA[1]+A[2]*cA[2];
    if (rst && !(dA > 0.f)) atomicAdd(&rst[0], 1.f);
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    float X[9]; mm3(A, Fb, X);
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9], P[9], Gm[9];
    polar(X, Q); cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
    if (fbk > 0.f && fbon == 0) { float db = bar_db(dt, fbj); if (db != 0.f) { float s = fbk*mu*db; for (int k=0;k<9;k++) P[k] += s*cf[k]; } }
    mmt3(P, Fb, Gm);
    if (fbk > 0.f && fbon == 1) { float db = bar_db(dA, fbj); if (db != 0.f) { float s = fbk*mu*db; for (int k=0;k<9;k++) Gm[k] += s*cA[k]; } }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*s);
    }
}
// (2') 상자 판 (--lim_mode box, 서브스텝당 커널 2 개 -- 색 8 번보다 싸다): det(A0 + Σ α_i d_i ⊗ ∇λ_i) 는 각 α_i 에 대해 아핀 (행렬식 보조정리) 이라
//   상자 [0,s]⁴ 위 최솟값은 꼭짓점 (부분집합 S 의 노드만 s 배로 움직인 15 + 1 배치) 에서 난다. 사면체마다 꼭짓점 det ≥ thr 를 지키는 최대 s_T 를 찾고
//   (s = 1 이 되면 끝, 아니면 이분 14 번 -- lo 는 늘 유효), 노드 α_n = min_T s_T (원자 min, 음 아닌 float 비트 순서 = 정수 순서).
//   모든 노드가 α_n ≤ s_T 이므로 실제 배치는 그 사면체의 상자 안 -> det A_T ≥ thr 보장 (Jacobi, 순서 무관·결정적). thr = min(jmin, 지금 det) (더 줄이지만 않게).
//   (--cfg_bc 묶인 노드는 α 1 로 움직인다 -- 그 노드가 든 사면체는 보장 밖)
__device__ __forceinline__ float det_sub(const float* A0, const float* Mo, int S, float s) {
    float B[9]; for (int k=0;k<9;k++) { float t = 0.f; for (int i=0;i<4;i++) if (S & (1<<i)) t += Mo[i*9+k]; B[k] = A0[k] + s*t; }
    return det3(B);
}
__global__ void ex_lim_box_tet(const int* szp, const long long* tets, const float* gl, const float* VT, const float* vn, const float* Ucum, float hs, float jmin,
                               unsigned int* alph, float* rst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T || !(VT[t] > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A0[9] = {1,0,0, 0,1,0, 0,0,1};
    float Mo[36];
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=Ucum[n[i]*3+c], d=hs*vn[n[i]*3+c]; for (int b=0;b<3;b++) { A0[c*3+b] += u*G[i*3+b]; Mo[i*9+c*3+b] = d*G[i*3+b]; } }
    float D0 = det3(A0);
    float thr = fminf(jmin, D0);
    bool ok = true;
    for (int S=1; S<16 && ok; S++) if (det_sub(A0, Mo, S, 1.f) < thr) ok = false;
    if (ok) return;
    float lo = 0.f, hi = 1.f;
    for (int it=0; it<14; it++) {
        float mid = 0.5f*(lo+hi); bool g = true;
        for (int S=1; S<16 && g; S++) if (det_sub(A0, Mo, S, mid) < thr) g = false;
        if (g) lo = mid; else hi = mid;
    }
    unsigned int bits = __float_as_uint(lo);
    for (int i=0;i<4;i++) atomicMin(&alph[n[i]], bits);
    atomicAdd(&rst[7], 1.f);
}
__global__ void ex_lim_box_apply(const int* szp, const float* mn, float* vn, float* Ucum, float hs, unsigned int* alph, int vmode, const unsigned char* fixd, float* rst) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float al = __uint_as_float(alph[n]);
    alph[n] = 0x3f800000u;                                         // 1.0f (다음 서브스텝)
    if (fixd && fixd[n]) al = 1.f;
    float v[3] = {vn[n*3], vn[n*3+1], vn[n*3+2]};
    for (int c=0;c<3;c++) Ucum[n*3+c] += al*hs*v[c];
    if (al < 1.f) {
        float m = mn[n], w = 1.f - al;
        atomicAdd(&rst[1], 1.f);
        if (vmode == 0) {
            for (int c=0;c<3;c++) { vn[n*3+c] = al*v[c]; atomicAdd(&rst[2+c], w*m*v[c]); }
            float v2 = v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
            atomicAdd(&rst[5], 0.5f*m*v2*(1.f - al*al));
            atomicAdd(&rst[6], w*m*sqrtf(v2));
        }
    }
}
// (2) 노드 이동 제한용 노드 -> 사면체 목록 (프레임마다, 격자 만든 뒤): nt[n*24 + s] = t*4 + (n 의 사면체 안 꼭짓점 번호), ntc[n] = 개수 (Kuhn 최대 24)
__global__ void ex_node_tets(const int* szp, const long long* tets, int* ntc, int* nt, int* ovf) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    for (int i=0;i<4;i++) { long long n = tets[t*4+i]; int s = atomicAdd(&ntc[n], 1); if (s < 24) nt[n*24+s] = t*4+i; else atomicAdd(ovf, 1); }
}
// (2) 색 col 의 노드만 (Kuhn 격자 지표 홀짝 8 색: 한 사면체의 네 노드는 지표 차가 {0,1}³\0 이라 모두 다른 색 -> 같은 색 노드는 사면체를 공유하지 않는다)
//   이번 서브스텝 변위 d = h_s v_n 을 α ∈ [0,1] 배로 줄여, 이 노드가 든 점유 사면체 (V_T > 0) 마다 det A_T ≥ jmin 을 지킨다.
//   한 노드만 움직이면 det(A + α d ⊗ ∇λ_i) = det A + α dᵀ cof(A) ∇λ_i (행렬식 보조정리) 로 α 에 선형 -> 닫힌 꼴 최대 α.
//   다른 색 노드는 그 사이 고정 (앞 색은 이미 갱신, 뒤 색은 아직) -- 색 순서 0..7 로 Gauss-Seidel, 같은 색 안은 병렬·순서 무관 (결정적).
//   vmode 0: v ← α v (속도도 이동과 맞춘다 -- 운동량 (1-α) m v·운동에너지 ½ m |v|² (1-α²) 를 rst 에 기록), 1: v 그대로 (이동만 줄임)
//   rst: [1] 제한된 (노드·서브스텝) 수, [2..4] 줄인 운동량 Σ (1-α) m v, [5] 줄인 운동에너지, [6] Σ |(1-α) m v|
__global__ void ex_lim_color(const int* szp, int col, const unsigned char* ncol, const int* ntc, const int* nt, const long long* tets, const float* gl,
                             const float* VT, const float* mn, float* vn, float* Ucum, float hs, float jmin, int vmode, const unsigned char* fixd, float* rst) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M || ncol[n] != col) return;
    float v[3] = {vn[n*3], vn[n*3+1], vn[n*3+2]};
    float d[3] = {hs*v[0], hs*v[1], hs*v[2]};
    float al = 1.f;
    if (!(fixd && fixd[n])) {
        int k = min(ntc[n], 24);
        for (int s=0;s<k;s++) {
            int e = nt[n*24+s]; int t = e >> 2, i = e & 3;
            if (!(VT[t] > 0.f)) continue;
            const float* G = gl + (long long)t*12;
            float A[9] = {1,0,0, 0,1,0, 0,0,1};
            for (int j=0;j<4;j++) { long long q = tets[(long long)t*4+j]; for (int c=0;c<3;c++) { float u=Ucum[q*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[j*3+b]; } }
            float cA[9]; cof3(A, cA);
            float D0 = A[0]*cA[0]+A[1]*cA[1]+A[2]*cA[2];
            float r = 0.f;
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) r += d[a]*cA[a*3+b]*G[i*3+b];
            if (r < 0.f) { float am = D0 > jmin ? (D0 - jmin)/(-r) : 0.f; al = fminf(al, am); }
        }
        al = fminf(fmaxf(al, 0.f), 1.f);
    }
    for (int c=0;c<3;c++) Ucum[n*3+c] += al*d[c];
    if (al < 1.f) {
        float m = mn[n], w = 1.f - al;
        atomicAdd(&rst[1], 1.f);
        if (vmode == 0) {
            for (int c=0;c<3;c++) { vn[n*3+c] = al*v[c]; atomicAdd(&rst[2+c], w*m*v[c]); }
            float v2 = v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
            atomicAdd(&rst[5], 0.5f*m*v2*(1.f - al*al));
            atomicAdd(&rst[6], w*m*sqrtf(v2));
        }
    }
}
// ======== 절 Z: A_T 특이값 비율 σ1/σ3 (사용자 10-29 "특이값 비율로 두 방식 각각") ========
// r(A) = σ1(A)/σ3(A) (AᵀA 고윳값 l0 ≤ l1 ≤ l2 -> sqrt(l2/l0)). 클램프 판정에서는 det A ≤ 0 이면 r = +inf (뒤집힌 배치는 비율 무한대로 본다).
__device__ __forceinline__ float sr_ratio(const float* A) {
    float d = det3(A);
    if (!(d > 0.f)) return __int_as_float(0x7f800000);
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=A[i]*A[k]+A[3+i]*A[3+k]+A[6+i]*A[6+k];
    float l[3]; symeig3(C, l);
    return sqrtf(fmaxf(l[2], 0.f)/fmaxf(l[0], 1e-30f));
}
// (1) 클램프 (--lim_sr c, 절 R 상자 판과 같은 Jacobi 틀: 사면체별 s_T -> 노드 α_n = min_T s_T (원자 min) -> ex_lim_box_apply 가 이동·속도 α 배):
//   판정 feas(s): 상자 [0,s]⁴ 의 꼭짓점 15 개 (부분집합 S 의 노드만 s 배 이동) 모두 r ≤ thrR (= max(c, 지금 r)) [+ jon 이면 det ≥ thrJ (= min(Ĵ, 지금 det)), 기존 상자 판 그대로].
//   빠른 통과: ‖Σ α_i d_i ∇λ_iᵀ‖₂ ≤ ρ = Σ |d_i||∇λ_i| (모든 α ∈ [0,1]⁴) -> Weyl 로 σ1 ≤ σ1⁰ + ρ, σ3 ≥ σ3⁰ - ρ; (σ1⁰+ρ) ≤ thrR (σ3⁰-ρ) 이면 비율은 상자 전체에서 보장 (꼭짓점 검사 생략).
//   아니면 s = 1 꼭짓점 검사, 실패하면 이분 14 번. 비율은 α 에 아핀이 아니라 꼭짓점 판정은 상자 내부를 보장하지 않는다 -> 이동 뒤 실제 비율 위반을 ex_sr_viol 로 센다 (rst[8]).
__device__ __forceinline__ bool sr_feas(const float* A0, const float* Mo, float s, float thrR, int jon, float thrJ) {
    for (int S=1; S<16; S++) {
        float B[9]; for (int k=0;k<9;k++) { float t = 0.f; for (int i=0;i<4;i++) if (S & (1<<i)) t += Mo[i*9+k]; B[k] = A0[k] + s*t; }
        if (jon && det3(B) < thrJ) return false;
        if (sr_ratio(B) > thrR) return false;
    }
    return true;
}
__global__ void ex_lim_sr_tet(const int* szp, const long long* tets, const float* gl, const float* VT, const float* vn, const float* Ucum, float hs,
                              float csr, int jon, float jmin, unsigned int* alph, float* rst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T || !(VT[t] > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A0[9] = {1,0,0, 0,1,0, 0,0,1};
    float Mo[36];
    float rho = 0.f;
    for (int i=0;i<4;i++) {
        float dd = 0.f, gg = 0.f;
        for (int c=0;c<3;c++) { float u=Ucum[n[i]*3+c], d=hs*vn[n[i]*3+c]; dd += d*d; for (int b=0;b<3;b++) { A0[c*3+b] += u*G[i*3+b]; Mo[i*9+c*3+b] = d*G[i*3+b]; } }
        for (int b=0;b<3;b++) gg += G[i*3+b]*G[i*3+b];
        rho += sqrtf(dd*gg);
    }
    float D0 = det3(A0);
    float r0 = sr_ratio(A0);
    float thrR = fmaxf(csr, r0);
    float thrJ = fminf(jmin, D0);
    bool ok = true;
    if (jon) { for (int S=1; S<16 && ok; S++) if (det_sub(A0, Mo, S, 1.f) < thrJ) ok = false; }
    if (ok && D0 > 0.f) {
        float C[9];
        for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=A0[i]*A0[k]+A0[3+i]*A0[3+k]+A0[6+i]*A0[6+k];
        float l[3]; symeig3(C, l);
        float s1 = sqrtf(fmaxf(l[2], 0.f)), s3 = sqrtf(fmaxf(l[0], 0.f));
        if (s3 - rho > 0.f && (s1 + rho) <= thrR*(s3 - rho)) return;          // Weyl: 상자 전체 보장
    }
    if (sr_feas(A0, Mo, 1.f, thrR, jon, thrJ)) return;
    float lo = 0.f, hi = 1.f;
    for (int it=0; it<14; it++) {
        float mid = 0.5f*(lo+hi);
        if (sr_feas(A0, Mo, mid, thrR, jon, thrJ)) lo = mid; else hi = mid;
    }
    unsigned int bits = __float_as_uint(lo);
    for (int i=0;i<4;i++) atomicMin(&alph[n[i]], bits);
    atomicAdd(&rst[7], 1.f);
}
// 클램프 검산 (--lim_sr_chk, 측정 판에는 끔): 이동 뒤 점유 사면체의 실제 r > c (1 + 1e-3) 인 (사면체 x 서브스텝) 수 -> rst[8], det ≤ 0 수 -> rst[9]
__global__ void ex_sr_viol(const int* szp, const long long* tets, const float* gl, const float* VT, const float* Ucum, float csr, float* rst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T || !(VT[t] > 0.f)) return;
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) { long long q = tets[t*4+i]; for (int c=0;c<3;c++) { float u=Ucum[q*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*gl[t*12+i*3+b]; } }
    float r = sr_ratio(A);
    if (!(det3(A) > 0.f)) atomicAdd(&rst[9], 1.f);
    else if (r > csr*1.001f) atomicAdd(&rst[8], 1.f);
}
// (2) 로그 장벽 (--sb κ --sb_c c --sb_dhat d̂): 사면체 에너지에 V_T κ μ b(c - r(A_T); d̂) (b 는 기존 bar_db 꼴 -(d-d̂)² ln(d/d̂), d < 1e-3 d̂ 선형 연장).
//   ∂r/∂A = A (P_max/(σ1σ3) - σ1 P_min/σ3³),  P_k = AᵀA 의 고유 사영 (단순 고윳값: Π_{j≠k}(C - l_j I)/(l_k - l_j), 겹친 고윳값: 두 차원 사영의 1/2 -- 대칭 부분기울기)
//   ∂E/∂A = -κ μ b'(d) ∂r/∂A  (b' < 0 이라 r 을 줄이는 쪽), 사면체 힘 f_i = -V (∂E/∂A) ∇λ_i. 장벽 부분은 double.
//   나머지 (FCR 탄성·뒤집힘 계수·det 장벽 fbk) 는 ex_tet_force_R 와 같다.
__device__ __forceinline__ double bar_db_d(double d, double jh) {
    if (!(d < jh)) return 0.0;
    double eps = 1e-3*jh;
    double dc = d < eps ? eps : d;
    double k = dc - jh, l = log(dc/jh);
    return -2.0*k*l - k*k/dc;
}
__device__ __forceinline__ void sr_grad(const float* Af, double* Gr, double* rr) {   // Gr = ∂r/∂A, rr = r
    double A[9]; for (int k=0;k<9;k++) A[k] = (double)Af[k];
    double C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=A[i]*A[k]+A[3+i]*A[3+k]+A[6+i]*A[6+k];
    double l[3]; symeig3(C, l);
    double l0 = fmax(l[0], 1e-30), l1 = l[1], l2 = l[2];
    double s1 = sqrt(l2), s3 = sqrt(l0);
    *rr = s1/s3;
    double tol = 1e-6*l2;
    double Pmax[9], Pmin[9], I3[9] = {1,0,0, 0,1,0, 0,0,1};
    double X0[9], X1[9], X2[9];
    for (int k=0;k<9;k++) { X0[k] = C[k] - l0*I3[k]; X1[k] = C[k] - l1*I3[k]; X2[k] = C[k] - l2*I3[k]; }
    if (l2 - l1 > tol) { double Q[9]; mm3(X1, X0, Q); double s = 1.0/((l2-l1)*(l2-l0)); for (int k=0;k<9;k++) Pmax[k] = Q[k]*s; }
    else { double s = 0.5/(l2-l0); for (int k=0;k<9;k++) Pmax[k] = X0[k]*s; }
    if (l1 - l0 > tol) { double Q[9]; mm3(X2, X1, Q); double s = 1.0/((l0-l2)*(l0-l1)); for (int k=0;k<9;k++) Pmin[k] = Q[k]*s; }
    else { double s = 0.5/(l0-l2); for (int k=0;k<9;k++) Pmin[k] = X2[k]*s; }
    double W[9]; double a = 1.0/(s1*s3), b = s1/(s3*s3*s3);
    for (int k=0;k<9;k++) W[k] = a*Pmax[k] - b*Pmin[k];
    mm3(A, W, Gr);
}
__global__ void ex_tet_force_SR(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                                float mu, float lam, float* f, float fbk, float fbj, int fbon, float* rst, float srk, float src, float srd) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    float cA[9]; cof3(A, cA);
    float dA = A[0]*cA[0]+A[1]*cA[1]+A[2]*cA[2];
    if (rst && !(dA > 0.f)) atomicAdd(&rst[0], 1.f);
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    float X[9]; mm3(A, Fb, X);
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9], P[9], Gm[9];
    polar(X, Q); cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
    if (fbk > 0.f && fbon == 0) { float db = bar_db(dt, fbj); if (db != 0.f) { float s = fbk*mu*db; for (int k=0;k<9;k++) P[k] += s*cf[k]; } }
    mmt3(P, Fb, Gm);
    if (fbk > 0.f && fbon == 1) { float db = bar_db(dA, fbj); if (db != 0.f) { float s = fbk*mu*db; for (int k=0;k<9;k++) Gm[k] += s*cA[k]; } }
    if (srk > 0.f) {
        float CA[9];
        for (int i=0;i<3;i++) for (int k=0;k<3;k++) CA[i*3+k]=A[i]*A[k]+A[3+i]*A[3+k]+A[6+i]*A[6+k];
        float la[3]; symeig3(CA, la);
        float rq = sqrtf(fmaxf(la[2], 0.f)/fmaxf(la[0], 1e-30f));
        if ((float)src - rq < 1.01f*srd) {                                     // 활성 근처만 double 로 다시
            double Gr[9], r;
            sr_grad(A, Gr, &r);
            double db = bar_db_d((double)src - r, (double)srd);
            if (db != 0.0) {
                double s = -(double)srk*(double)mu*db;
                for (int k=0;k<9;k++) Gm[k] += (float)(s*Gr[k]);
                if (rst) atomicAdd(&rst[10], 1.f);
            }
        }
    }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*s);
    }
}
// ======== 절 P: lag 판 K 서브스텝마다 재평균 (--rep K --rep_mode vel, 사용자 10-29 "해") ========
// 속도만 재평균 (입자 이동·재배치 없음, 같은 rows/λ·같은 노드 질량): v_p = Σλ v_n (G2P, PIC) -> (mv)_n = Σ w_pn v_p, v_n = (mv)_n / m_n (P2G)
//   w_pn = P2G 와 같은 질량 가중 (사면체 등분 0.25 m_p / Σλm 판은 λ m_p). mv 는 호출 전에 비워 둔다.
// 절 Q 선택지:
//   g2pq = 1 ((a) --rep_w tet4): G2P 도 P2G 와 같은 사면체 등분 가중 v_p = ¼ Σ v_n
//   tetmass = 0 + 정규화 질량 ml = Σλ m_p ((a) --rep_w lam): P2G 를 G2P 와 같은 λ m_p 가중 (동역학 노드 질량 m_T/4 는 그대로, 정규화만 Σλm)
//   glT ((b) --affine, APIC): C_p = Σ v_n ⊗ ∇λ_n (정답 PG 의 C = B D⁻¹ 와 같은 값), P2G 값 v_p + C_p (x_n - x_p) -- 같은 rows 로 되돌리므로 x_n - x_p 는
//     기준 배치 (npos - x0) 로 계산 (지금 배치의 A_T (X_n - X_p) 와 C A_T = Σ v ⊗ ∇λ 가 상쇄 -- 같은 값). 이 경우 각 기여가 정확히 v_n 이 되어 재평균은 항등이다.
__global__ void ex_rep_vel_p2g(int N, const long long* rows, const float* lamb, const float* m, int tetmass, const float* vn, float* mv,
                               int g2pq, const float* glT, const long long* tinv, const float* npos, const float* x0) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    long long nd[4]; float lm[4]; float vp[3] = {0,0,0};
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; float q = g2pq ? 0.25f : lm[i]; for (int c=0;c<3;c++) vp[c] += q*vn[nd[i]*3+c]; }
    float Gv[9] = {0,0,0,0,0,0,0,0,0};
    if (glT) { const float* gl = glT + tinv[p]*12;
        for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float vc = vn[nd[i]*3+c]; for (int b=0;b<3;b++) Gv[c*3+b] += vc*gl[i*3+b]; } }
    float mp = m[p];
    for (int i=0;i<4;i++){ float w=(tetmass ? 0.25f : lm[i])*mp;
        float d[3] = {0,0,0};
        if (glT) for (int b=0;b<3;b++) d[b] = npos[nd[i]*3+b] - x0[p*3+b];
        for (int c=0;c<3;c++) atomicAdd(&mv[nd[i]*3+c], w*(vp[c] + Gv[c*3]*d[0] + Gv[c*3+1]*d[1] + Gv[c*3+2]*d[2])); }
}
__global__ void ex_rep_vel_fin(const int* szp, const float* mn, float* mv, float* vn) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float m = mn[n];
    for (int c=0;c<3;c++) { if (m > 1e-30f) vn[n*3+c] = mv[n*3+c]/m; mv[n*3+c] = 0.f; }
}
// 프레임 시작 바닥 후보 압축 배열: x0, F0, F0⁻¹, c = h v + h² g
__global__ void ex_near_init(const int* szp, int nnarg, const long long* nidx, const float* x, const float* v, const float* Fp, float h, float g0, float g1, float g2,
                             float* x0c, float* F0c, float* F0inv, float* cc) {
    int nn_ = szp ? szp[2] : nnarg;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = nidx[q];
    float gg[3] = {g0, g1, g2};
    for (int k=0;k<3;k++){ x0c[q*3+k]=x[p*3+k]; cc[q*3+k]=h*v[p*3+k] + h*h*gg[k]; }
    double A[9], C[9];
    for (int k=0;k<9;k++){ A[k]=Fp[p*9+k]; F0c[q*9+k]=Fp[p*9+k]; }
    cof3(A, C);
    double det = A[0]*C[0]+A[1]*C[1]+A[2]*C[2];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) F0inv[q*9+i*3+j] = (float)(C[j*3+i]/det);
}

// (sub) 서브스텝 G2P: 매핑 포함 (floor=1, g2p_part 와 같은 식) 또는 선형, v = (q - x)/h_s * damp
__global__ void ex_g2p_sub(int N, const float* x, const float* dU, const long long* rows, const float* lamb, const float* glT, const long long* tinv,
                           const float* Fp, int floor, double p0, double p1, double p2, double n0, double n1, double n2,
                           double eps_st, double eps_sp, double xn, double hs, double damp, float* x2, float* v2, float* F2) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double u[3]={0,0,0}, G[9]={0,0,0,0,0,0,0,0,0}, xx[3], c0[3]={0,0,0};
    const float* gl = glT + tinv[p]*12;
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; double lm=lamb[p*4+i]; for (int a=0;a<3;a++){ double Ua=dU[nd*3+a]; u[a]+=lm*Ua; for (int b=0;b<3;b++) G[a*3+b]+=Ua*gl[i*3+b]; } }
    for (int k=0;k<3;k++) xx[k]=x[p*3+k];
    double Jf[9], r[3];
    if (floor) {
        double pt[3]={p0,p1,p2}, nv[3]={n0,n1,n2};
        NearMap M; near_map(xx, u, G, c0, pt, nv, eps_st, eps_sp, xn, M);
        mm3(M.Mn, M.K1, Jf);
        for (int k=0;k<3;k++) r[k]=M.r[k];
    } else {
        for (int k=0;k<9;k++) Jf[k] = G[k] + ((k%4==0)?1.0:0.0);
        for (int k=0;k<3;k++) r[k]=u[k];
    }
    for (int k=0;k<3;k++) { x2[p*3+k] = (float)(xx[k] + r[k]); v2[p*3+k] = (float)(r[k]/hs*damp); }
    double F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    double F2d[9]; mm3(Jf, F, F2d);
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
}

// ---------------- 최적화 판 격자: 셀 점유 표시 -> (팽창) -> 노드·사면체 압축, 서브스텝마다 장치 표로 locate ----------------
// gpar = [lo0, lo1, lo2, Ai(9)], ipar = [n0, n1, n2]
__global__ void ex_mark_cells(int N, const float* x, const float* gpar, const int* ipar, unsigned char* cmark) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    float q[3] = {x[p*3]-gpar[0], x[p*3+1]-gpar[1], x[p*3+2]-gpar[2]};
    const float* Ai = gpar + 3;
    int nl[3] = {n0, n1, n2}, ci[3];
    for (int k=0;k<3;k++){ float y = Ai[k*3]*q[0] + Ai[k*3+1]*q[1] + Ai[k*3+2]*q[2]; int c=(int)floorf(y); ci[k]=min(max(c,0),nl[k]-2); }
    cmark[((long long)ci[0]*n1 + ci[1])*n2 + ci[2]] = 1;
}
__global__ void ex_dilate(const int* ipar, int r, const unsigned char* cin, unsigned char* cout) {
    long long cidx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    long long P = (long long)n0*n1*n2;
    if (cidx >= P) return;
    int c2 = cidx % n2, c1 = (cidx / n2) % n1, c0 = cidx / ((long long)n1*n2);
    if (c0 > n0-2 || c1 > n1-2 || c2 > n2-2) { cout[cidx]=0; return; }
    unsigned char on = 0;
    for (int a=-r;a<=r && !on;a++) for (int b=-r;b<=r && !on;b++) for (int d=-r;d<=r && !on;d++) {
        int e0=c0+a, e1=c1+b, e2=c2+d;
        if (e0<0||e1<0||e2<0||e0>n0-2||e1>n1-2||e2>n2-2) continue;
        if (cin[((long long)e0*n1+e1)*n2+e2]) on = 1;
    }
    cout[cidx] = on;
}
__global__ void ex_mark_nt(const int* ipar, const unsigned char* cmark, unsigned char* nmark, unsigned char* tmark) {
    long long cidx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    long long P = (long long)n0*n1*n2;
    if (cidx >= P || !cmark[cidx]) return;
    int c2 = cidx % n2, c1 = (cidx / n2) % n1, c0 = cidx / ((long long)n1*n2);
    for (int a=0;a<2;a++) for (int b=0;b<2;b++) for (int d=0;d<2;d++) nmark[((long long)(c0+a)*n1+(c1+b))*n2+(c2+d)] = 1;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) if (a!=b) tmark[cidx*9+a*3+b] = 1;
}
// 압축 사면체 꼭짓점: locate_k 와 같은 순서 (기준 꼭짓점, +e_pr0, +e_pr1, +e_pr2)
__global__ void ex_build_tets(const int* ipar, const unsigned char* tmark, const long long* tcm, const long long* ncm, long long* tets) {
    long long s = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    long long P = (long long)n0*n1*n2;
    if (s >= P*9 || !tmark[s]) return;
    long long cidx = s / 9; int a = (s % 9) / 3, b = s % 3, c = 3 - a - b;
    int v[3] = {(int)(cidx / ((long long)n1*n2)), (int)((cidx / n2) % n1), (int)(cidx % n2)};
    long long t = tcm[s];
    int pr[3] = {a, b, c};
    for (int r=0;r<4;r++) {
        if (r>0) v[pr[r-1]] += 1;
        tets[t*4+r] = ncm[((long long)v[0]*n1 + v[1])*n2 + v[2]];
    }
}
// 서브스텝 locate (locate_k 와 같은 식) + 압축 번호; 팽창 영역 밖이면 넘침 표시 szp[3]
__global__ void ex_locate(int N, const float* x, const float* gpar, const int* ipar, const long long* ncm, const long long* tcm,
                          const unsigned char* tmark, long long* rows, float* lamb, long long* pinv, int* szp) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    float q[3] = {x[p*3]-gpar[0], x[p*3+1]-gpar[1], x[p*3+2]-gpar[2]};
    const float* Ai = gpar + 3;
    float y[3]; for (int k=0;k<3;k++) y[k] = Ai[k*3]*q[0] + Ai[k*3+1]*q[1] + Ai[k*3+2]*q[2];
    int nl[3] = {n0, n1, n2};
    int ci[3]; float f[3];
    for (int k=0;k<3;k++){ int c = (int)floorf(y[k]); c = min(max(c, 0), nl[k]-2); ci[k]=c; f[k] = fminf(fmaxf(y[k]-(float)c, 0.f), 1.f); }
    int pr[3] = {0,1,2};
    if (f[pr[0]] < f[pr[1]]) { int t=pr[0]; pr[0]=pr[1]; pr[1]=t; }
    if (f[pr[1]] < f[pr[2]]) { int t=pr[1]; pr[1]=pr[2]; pr[2]=t; }
    if (f[pr[0]] < f[pr[1]]) { int t=pr[0]; pr[0]=pr[1]; pr[1]=t; }
    float s0=f[pr[0]], s1=f[pr[1]], s2=f[pr[2]];
    lamb[p*4]=1.f-s0; lamb[p*4+1]=s0-s1; lamb[p*4+2]=s1-s2; lamb[p*4+3]=s2;
    long long cell = ((long long)ci[0]*n1 + ci[1])*n2 + ci[2];
    long long tt = cell*9 + pr[0]*3 + pr[1];
    if (!tmark[tt]) { atomicAdd(&szp[3], 1); }
    pinv[p] = tmark[tt] ? tcm[tt] : 0;
    int v[3] = {ci[0], ci[1], ci[2]};
    for (int r=0;r<4;r++) {
        if (r>0) v[pr[r-1]] += 1;
        long long fl = ((long long)v[0]*n1 + v[1])*n2 + v[2];
        long long cm = ncm[fl];
        rows[p*4+r] = cm >= 0 ? cm : 0;
    }
}
// P2G 질량·운동량 (float 원자 합)
// tetmass = 0: 노드 질량 Σ λ m_p (기본, 암시적 판 관성 행렬의 행 합), 1: 사면체 등분 Σ_T m_T / 4 (정의 변경 -- 선형 사면체 표준 질량 집중)
__global__ void ex_p2g_mv(int N, const long long* rows, const float* lamb, const float* m, const float* v, float* mn, float* mv, int tetmass,
                          const float* Cp, const float* xp, const float* npos, float* ml) {
    // Cp (선택, --affine): 운동량에 m w C (x_n - x_p) -- 아핀 속도장은 노드 값이 정확히 v(x_n)
    // ml (선택, 절 Q (a) --rep_w lam): 속도 전달 가중을 G2P 와 같은 λ m_p 로 (운동량 mv 에 λ m_p v, 정규화 질량 ml = Σ λ m_p), 노드 질량 mn 은 tetmass 정의 그대로
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    float mp = m[p];
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; float w=(tetmass ? 0.25f : lamb[p*4+i])*mp; atomicAdd(&mn[nd], w);
        if (ml) { w = lamb[p*4+i]*mp; atomicAdd(&ml[nd], w); }
        float d[3] = {0,0,0};
        if (Cp) for (int b=0;b<3;b++) d[b] = npos[nd*3+b] - xp[p*3+b];
        for (int c=0;c<3;c++) { float vc = v[p*3+c]; if (Cp) vc += Cp[p*9+c*3]*d[0] + Cp[p*9+c*3+1]*d[1] + Cp[p*9+c*3+2]*d[2];
            atomicAdd(&mv[nd*3+c], w*vc); } }
}
__global__ void ex_vfin(const int* szp, int Marg, const float* mn, const float* mv, float* vn, const float* ml) {
    // ml (선택, --rep_w lam): 속도 = 운동량 / Σλm (그 값이 0 인 노드 -- 모든 입자의 λ 가 0 -- 는 속도 0)
    int M = szp ? szp[0] : Marg;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float m = ml ? ml[n] : mn[n];
    for (int c=0;c<3;c++) vn[n*3+c] = m > 1e-30f ? mv[n*3+c]/m : 0.f;
}
// 분할 (CudaTetqK 와 같은 정의: 예측 변위 u0 = Σλ (H v_n + H² g) 의 끝 높이 s/ε ≤ 20 이면 바닥 후보) + 항등 입자만 사면체 V·V F 합
__global__ void ex_split(int N, const float* x, const long long* rows, const float* lamb, const long long* pinv, const float* vn, float H,
                         float g0, float g1, float g2, int floor, float p0, float p1, float p2, float q0, float q1, float q2, float eps_sp, float eps_st,
                         const float* vol, const float* Fp, float* VTs, float* FTs, int* nslot, long long* nidx, int* szp, int keep, int near_in_tet) {
    // near_in_tet = 1 (--floor vel, 절 O): 바닥 후보도 사면체 F̄ 에 넣는다 (에너지 바닥이 없으므로 탄성은 사면체가 갖는다) -- 후보 표시는 접촉 판정용
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    bool near = false;
    if (keep) {
        // 절 P (--rep full): 프레임 안 재 P2G 는 keep=1 -- 바닥 후보 표시는 프레임 것 그대로, near_in_tet=1 (--floor vel) 이면 후보도 F̄ 에
        if (nslot[p] < 0 || near_in_tet) { long long t = pinv[p]; float V = vol[p]; atomicAdd(&VTs[t], V); for (int k=0;k<9;k++) atomicAdd(&FTs[t*9+k], V*Fp[p*9+k]); }
        return;
    }
    if (floor) {
        float u[3]={0,0,0}, gg[3]={g0,g1,g2};
        for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; float lm=lamb[p*4+i]; for (int c=0;c<3;c++) u[c] += lm*(H*vn[nd*3+c] + H*H*gg[c]); }
        float s = (x[p*3]+u[0]-p0)*q0 + (x[p*3+1]+u[1]-p1)*q1 + (x[p*3+2]+u[2]-p2)*q2;
        if (eps_sp > 0.f && s/eps_sp <= 20.f) near = true;
        if (eps_st > 0.f && s/eps_st <= 20.f) near = true;
    }
    if (near) {
        int q = atomicAdd(&szp[2], 1);
        nidx[q] = p; nslot[p] = q;
        if (near_in_tet) { long long t = pinv[p]; float V = vol[p]; atomicAdd(&VTs[t], V); for (int k=0;k<9;k++) atomicAdd(&FTs[t*9+k], V*Fp[p*9+k]); }
    } else {
        nslot[p] = -1;
        long long t = pinv[p]; float V = vol[p];
        atomicAdd(&VTs[t], V);
        for (int k=0;k<9;k++) atomicAdd(&FTs[t*9+k], V*Fp[p*9+k]);
    }
}
__global__ void ex_ffin(const int* szp, int Targ, const float* VTs, const float* FTs, float* FT) {
    int T = szp ? szp[1] : Targ;
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VTs[t];
    for (int k=0;k<9;k++) FT[t*9+k] = V > 0.f ? FTs[t*9+k]/V : ((k%4==0)?1.f:0.f);
}
__global__ void ex_gather(const int* szp, int nnarg, const long long* nidx, const float* x, const float* Fp, float* xs, float* Fn) {
    int nn_ = szp ? szp[2] : nnarg;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = nidx[q];
    for (int k=0;k<3;k++) xs[q*3+k]=x[p*3+k];
    for (int k=0;k<9;k++) Fn[q*9+k]=Fp[p*9+k];
}
// 사면체 ∇λ float (tet_geo 의 float 판)
__global__ void ex_tet_geo32(const int* szp, const long long* tets, const float* npos, float* gl) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    double X[4][3];
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) X[i][c] = (double)npos[tets[t*4+i]*3+c];
    double D[9];
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) D[r*3+c] = X[c+1][r]-X[0][r];
    double C[9]; cof3(D, C);
    double det = D[0]*C[0]+D[1]*C[1]+D[2]*C[2];
    double g[12];
    for (int i=0;i<3;i++) for (int b=0;b<3;b++) g[(i+1)*3+b] = C[b*3+i]/det;
    for (int b=0;b<3;b++) g[b] = -(g[3+b]+g[6+b]+g[9+b]);
    for (int k=0;k<12;k++) gl[t*12+k] = (float)g[k];
}
// ======== sub 판 융합 커널 (최적화: 값은 같은 식, 소비하며 비우기 -> 채우기 커널 없음) ========
// 사면체 힘 + F̄ 나누기 융합: F̄ = FTs/VTs 를 그 자리에서, 쓴 뒤 VTs·FTs 를 0 으로 (다음 서브스텝 P2G 가 다시 쌓는다)
__global__ void ex_tet_force_sub(const int* szp, const long long* tets, const float* gl, float* FTs, float* VTs, float mu, float lam, float* f) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VTs[t];
    if (!(V > 0.f)) return;
    float Fb[9]; for (int k=0;k<9;k++) { Fb[k] = FTs[t*9+k]/V; FTs[t*9+k] = 0.f; }
    VTs[t] = 0.f;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float X[9]; for (int k=0;k<9;k++) X[k]=Fb[k];
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9], P[9], Gm[9];
    polar(X, Q); cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
    mmt3(P, Fb, Gm);
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*s);
    }
}
// 노드 걸음 + 속도 나누기 융합: v = mv/m + h_s((f - gn)/m + g), 쓴 뒤 mn·mv·f·gn 을 0 으로
__global__ void ex_node_step_sub(const int* szp, float* mn, float* mv, float* f, float* gn, float* dU, float g0, float g1, float g2, float hs,
                                 const unsigned char* fixd, const float* vbc) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    float gg[3] = {g0, g1, g2};
    float m = mn[n];
    for (int c=0;c<3;c++) {
        float v;
        if (fixd && fixd[n]) v = vbc[n*3+c];
        else if (m > 0.f) v = mv[n*3+c]/m + hs*((f[n*3+c] - (gn ? gn[n*3+c] : 0.f))/m + gg[c]);
        else v = 0.f;
        dU[n*3+c] = hs*v;
        mv[n*3+c] = 0.f; f[n*3+c] = 0.f; if (gn) gn[n*3+c] = 0.f;
    }
    mn[n] = 0.f;
}
// 선형 G2P (float) + 다음 서브스텝 locate + P2G (질량·운동량, 항등 입자 V·VF) 융합
__global__ void ex_g2p_p2g(int N, const float* xa, const float* Fa, const float* dU, long long* rows, float* lamb, long long* pinv, const float* glT,
                           float hs, float damp, float* xb, float* vb, float* Fb, int do_p2g,
                           const float* gpar, const int* ipar, const long long* ncm, const long long* tcm, const unsigned char* tmark,
                           const float* m, const float* vol, const int* nslot, int tetmass, float* mn, float* mv, float* VTs, float* FTs, int* szp) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    float u[3]={0,0,0}, A[9]={1,0,0,0,1,0,0,0,1};
    const float* gl = glT + pinv[p]*12;
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; float lm=lamb[p*4+i]; for (int a=0;a<3;a++){ float Ua=dU[nd*3+a]; u[a]+=lm*Ua; for (int b=0;b<3;b++) A[a*3+b]+=Ua*gl[i*3+b]; } }
    float xn[3], vn_[3], F[9], F2[9];
    for (int k=0;k<3;k++) { xn[k] = xa[p*3+k] + u[k]; vn_[k] = u[k]/hs*damp; xb[p*3+k] = xn[k]; vb[p*3+k] = vn_[k]; }
    for (int k=0;k<9;k++) F[k]=Fa[p*9+k];
    mm3(A, F, F2);
    for (int k=0;k<9;k++) Fb[p*9+k] = F2[k];
    if (!do_p2g) return;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    float q[3] = {xn[0]-gpar[0], xn[1]-gpar[1], xn[2]-gpar[2]};
    const float* Ai = gpar + 3;
    float y[3]; for (int k=0;k<3;k++) y[k] = Ai[k*3]*q[0] + Ai[k*3+1]*q[1] + Ai[k*3+2]*q[2];
    int nl[3] = {n0, n1, n2};
    int ci[3]; float fr[3];
    for (int k=0;k<3;k++){ int c = (int)floorf(y[k]); c = min(max(c, 0), nl[k]-2); ci[k]=c; fr[k] = fminf(fmaxf(y[k]-(float)c, 0.f), 1.f); }
    int pr[3] = {0,1,2};
    if (fr[pr[0]] < fr[pr[1]]) { int t=pr[0]; pr[0]=pr[1]; pr[1]=t; }
    if (fr[pr[1]] < fr[pr[2]]) { int t=pr[1]; pr[1]=pr[2]; pr[2]=t; }
    if (fr[pr[0]] < fr[pr[1]]) { int t=pr[0]; pr[0]=pr[1]; pr[1]=t; }
    float s0=fr[pr[0]], s1=fr[pr[1]], s2=fr[pr[2]];
    float lmb[4] = {1.f-s0, s0-s1, s1-s2, s2};
    long long cell = ((long long)ci[0]*n1 + ci[1])*n2 + ci[2];
    long long tt = cell*9 + pr[0]*3 + pr[1];
    if (!tmark[tt]) atomicAdd(&szp[3], 1);
    long long t = tmark[tt] ? tcm[tt] : 0;
    pinv[p] = t;
    int v[3] = {ci[0], ci[1], ci[2]};
    float mp = m[p];
    for (int r=0;r<4;r++) {
        if (r>0) v[pr[r-1]] += 1;
        long long cm = ncm[((long long)v[0]*n1 + v[1])*n2 + v[2]];
        long long nd = cm >= 0 ? cm : 0;
        rows[p*4+r] = nd; lamb[p*4+r] = lmb[r];
        float w = (tetmass ? 0.25f : lmb[r])*mp;
        atomicAdd(&mn[nd], w);
        for (int c=0;c<3;c++) atomicAdd(&mv[nd*3+c], w*vn_[c]);
    }
    if (nslot[p] < 0) {
        float V = vol[p];
        atomicAdd(&VTs[t], V);
        for (int k=0;k<9;k++) atomicAdd(&FTs[t*9+k], V*F2[k]);
    }
}

// 정렬된 입자 (같은 사면체 연속) 의 워프 구간 합: 같은 키가 이어진 레인끼리 포함 누적합 (5 단계), 구간 끝 레인만 원자 합
__device__ __forceinline__ float seg_scan(float v, int key, int lane) {
    for (int d=1; d<32; d<<=1) {
        float o = __shfl_up_sync(0xffffffff, v, d);
        int ko = __shfl_up_sync(0xffffffff, key, d);
        if (lane >= d && ko == key) v += o;
    }
    return v;
}
// ex_g2p_p2g 와 같은 식, P2G 원자 합만 워프 구간 합으로 (입자는 프레임 시작에 사면체 번호로 정렬) -- 모든 레인이 끝까지 간다
__global__ void ex_g2p_p2g_seg(int N, const float* xa, const float* Fa, const float* dU, long long* rows, float* lamb, long long* pinv, const float* glT,
                           float hs, float damp, float* xb, float* vb, float* Fb, int do_p2g,
                           const float* gpar, const int* ipar, const long long* ncm, const long long* tcm, const unsigned char* tmark,
                           const float* m, const float* vol, const int* nslot, int tetmass, float* mn, float* mv, float* VTs, float* FTs, int* szp) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool on = p < N;
    float vn_[3] = {0,0,0}, F2[9] = {0,0,0,0,0,0,0,0,0}, xn[3] = {0,0,0};
    if (on) {
        float u[3]={0,0,0}, A[9]={1,0,0,0,1,0,0,0,1};
        const float* gl = glT + pinv[p]*12;
        for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; float lm=lamb[p*4+i]; for (int a=0;a<3;a++){ float Ua=dU[nd*3+a]; u[a]+=lm*Ua; for (int b=0;b<3;b++) A[a*3+b]+=Ua*gl[i*3+b]; } }
        float F[9];
        for (int k=0;k<3;k++) { xn[k] = xa[p*3+k] + u[k]; vn_[k] = u[k]/hs*damp; xb[p*3+k] = xn[k]; vb[p*3+k] = vn_[k]; }
        for (int k=0;k<9;k++) F[k]=Fa[p*9+k];
        mm3(A, F, F2);
        for (int k=0;k<9;k++) Fb[p*9+k] = F2[k];
    }
    if (!do_p2g) return;
    int key = -1 - lane;                                       // 꺼진 레인은 고유 키
    long long nd4[4] = {0,0,0,0};
    float wv[4] = {0,0,0,0};
    float V = 0.f;
    long long t = 0;
    if (on) {
        int n0=ipar[0], n1=ipar[1], n2=ipar[2];
        float q[3] = {xn[0]-gpar[0], xn[1]-gpar[1], xn[2]-gpar[2]};
        const float* Ai = gpar + 3;
        float y[3]; for (int k=0;k<3;k++) y[k] = Ai[k*3]*q[0] + Ai[k*3+1]*q[1] + Ai[k*3+2]*q[2];
        int nl[3] = {n0, n1, n2};
        int ci[3]; float fr[3];
        for (int k=0;k<3;k++){ int c = (int)floorf(y[k]); c = min(max(c, 0), nl[k]-2); ci[k]=c; fr[k] = fminf(fmaxf(y[k]-(float)c, 0.f), 1.f); }
        int pr[3] = {0,1,2};
        if (fr[pr[0]] < fr[pr[1]]) { int tt_=pr[0]; pr[0]=pr[1]; pr[1]=tt_; }
        if (fr[pr[1]] < fr[pr[2]]) { int tt_=pr[1]; pr[1]=pr[2]; pr[2]=tt_; }
        if (fr[pr[0]] < fr[pr[1]]) { int tt_=pr[0]; pr[0]=pr[1]; pr[1]=tt_; }
        float s0=fr[pr[0]], s1=fr[pr[1]], s2=fr[pr[2]];
        float lmb[4] = {1.f-s0, s0-s1, s1-s2, s2};
        long long cell = ((long long)ci[0]*n1 + ci[1])*n2 + ci[2];
        long long tt = cell*9 + pr[0]*3 + pr[1];
        if (!tmark[tt]) atomicAdd(&szp[3], 1);
        t = tmark[tt] ? tcm[tt] : 0;
        pinv[p] = t;
        key = (int)t;
        int v[3] = {ci[0], ci[1], ci[2]};
        float mp = m[p];
        for (int r=0;r<4;r++) {
            if (r>0) v[pr[r-1]] += 1;
            long long cm = ncm[((long long)v[0]*n1 + v[1])*n2 + v[2]];
            nd4[r] = cm >= 0 ? cm : 0;
            rows[p*4+r] = nd4[r]; lamb[p*4+r] = lmb[r];
            wv[r] = (tetmass ? 0.25f : lmb[r])*mp;
        }
        if (nslot[p] < 0) V = vol[p];
    }
    int kn = __shfl_down_sync(0xffffffff, key, 1);
    bool last = (lane == 31) || (kn != key);
    // 같은 사면체 = 같은 네 노드 (같은 순서) 이므로 키 하나로 노드 합도 묶는다
    for (int r=0;r<4;r++) {
        float sw = seg_scan(wv[r], key, lane);
        float sx = seg_scan(wv[r]*vn_[0], key, lane), sy = seg_scan(wv[r]*vn_[1], key, lane), sz = seg_scan(wv[r]*vn_[2], key, lane);
        if (on && last) { long long nd = nd4[r]; atomicAdd(&mn[nd], sw); atomicAdd(&mv[nd*3], sx); atomicAdd(&mv[nd*3+1], sy); atomicAdd(&mv[nd*3+2], sz); }
    }
    float sV = seg_scan(V, key, lane);
    float sF[9]; for (int k=0;k<9;k++) sF[k] = seg_scan(V*F2[k], key, lane);
    if (on && last && sV > 0.f) { atomicAdd(&VTs[t], sV); for (int k=0;k<9;k++) atomicAdd(&FTs[t*9+k], sF[k]); }
}
// ======== 절 S: lag 판 P2G 를 2 차 B-스플라인으로 (--p2g bs, 사용자 10-29) ========
// 격자 = 사면체 메시 노드 격자 (Kuhn 정육면체 격자, 지표 좌표 y = Ai (x - lo), 노드 = 정수 y, 셀 c = [c, c+1]³, 셀 중심 = c + ½).
// 축마다 2 차 B-스플라인 N(t) (|t| < ½: ¾ - t², ½ ≤ |t| < 3/2: ½ (3/2 - |t|)²), 분할 1 (Σ_k N(y - k) = 1, 어떤 정수 격자에서나).
//   노드:  base = floor(y - ½), f = y - base, 노드 base..base+2 의 w = [½(1.5-f)², ¾-(f-1)², ½(f-½)²]   (fem_bs_cuda 와 같은 식)
//   셀:   셀 중심 격자는 y_c = y - ½ 의 정수 격자 -> 같은 식을 y - ½ 에 (셀 base_c..base_c+2 = 주변 3³ 셀)
__device__ __forceinline__ void bsl_w(float y, int& b, float* W, float* D) {
    b = (int)floorf(y - 0.5f);
    float f = y - (float)b;
    W[0] = 0.5f*(1.5f-f)*(1.5f-f); W[1] = 0.75f-(f-1.f)*(f-1.f); W[2] = 0.5f*(f-0.5f)*(f-0.5f);
    D[0] = f-1.5f; D[1] = -2.f*(f-1.f); D[2] = f-0.5f;
}
__device__ __forceinline__ void bsl_y(const float* x, const float* gpar, float* y) {
    float q[3] = {x[0]-gpar[0], x[1]-gpar[1], x[2]-gpar[2]};
    const float* Ai = gpar + 3;
    for (int k=0;k<3;k++) y[k] = Ai[k*3]*q[0] + Ai[k*3+1]*q[1] + Ai[k*3+2]*q[2];
}
// 프레임 시작 P2G (입자마다 한 번):
//   m_n += w(x_n - x_p) m_p,  (mv)_n += w m_p v_p                                  (27 노드, 압축 번호 ncm; 메시 밖 노드면 넘침 szp[3])
//   cV_c += w(x_c - x_p) V_p,  cVF_c += w(x_c - x_p) V_p F_p                         (27 셀, 밀집 셀 번호; 메시 밖 셀이면 넘침)
//   바닥 후보 (floor=1): 프레임 끝 예측 높이 (x + H v + H² g - p)·n ≤ margin 이면 nidx 에 (접촉 판정 대상 표시만 -- 판정은 서브스텝마다 정확히)
__global__ void bsl_p2g(int N, const float* x, const float* v, const float* F, const float* m, const float* vol, const float* gpar, const int* ipar,
                        const long long* ncm, const unsigned char* cmk, float* mn, float* mv, float* cV, float* cVF, int* szp,
                        int floor, float H, float g0, float g1, float g2, float p0, float p1, float p2, float q0, float q1, float q2, float margin,
                        long long* nidx, int nrm, float* bst) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    float xp[3] = {x[p*3], x[p*3+1], x[p*3+2]}, vp[3] = {v[p*3], v[p*3+1], v[p*3+2]};
    float y[3]; bsl_y(xp, gpar, y);
    float mp = m[p], Vp = vol[p];
    int b[3]; float W[3][3], D[3][3];
    for (int k=0;k<3;k++) bsl_w(y[k], b[k], W[k], D[k]);
    int ovf = 0;
    // 절 T (nrm = 1): 메시에 있는 노드에만 쓰고, 입자마다 메시 안 노드 가중 합 Win 으로 나눠 Σ_n w = 1 (총 질량·운동량 보존). 메시 밖 몫은 넘침이 아니다.
    float sn = 1.f;
    if (nrm) {
        float Win = 0.f;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
            int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
            if (a0<0||a1<0||a2<0||a0>n0-1||a1>n1-1||a2>n2-1) continue;
            if (ncm[((long long)a0*n1 + a1)*n2 + a2] < 0) continue;
            Win += W[0][i]*W[1][j]*W[2][l];
        }
        if (Win > 0.f) { sn = 1.f/Win; if (bst) atomicAdd(&bst[2], (1.f - Win)*mp); } else ovf = 1;
    }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>n0-1||a1>n1-1||a2>n2-1) { if (!nrm) ovf = 1; continue; }
        long long nd = ncm[((long long)a0*n1 + a1)*n2 + a2];
        if (nd < 0) { if (!nrm) ovf = 1; continue; }
        float w = W[0][i]*W[1][j]*W[2][l]*sn*mp;
        atomicAdd(&mn[nd], w);
        for (int c=0;c<3;c++) atomicAdd(&mv[nd*3+c], w*vp[c]);
    }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[p*9+k];
    for (int k=0;k<3;k++) bsl_w(y[k] - 0.5f, b[k], W[k], D[k]);
    // 절 T (nrm = 1): 셀도 메시에 있는 셀에만, 메시 안 셀 가중 합 Cin 으로 나눠 Σ_c w = 1 (총 부피 보존)
    float sc = 1.f;
    if (nrm) {
        float Cin = 0.f;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
            int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
            if (a0<0||a1<0||a2<0||a0>n0-2||a1>n1-2||a2>n2-2) continue;
            if (!cmk[((long long)a0*n1 + a1)*n2 + a2]) continue;
            Cin += W[0][i]*W[1][j]*W[2][l];
        }
        if (Cin > 0.f) { sc = 1.f/Cin; if (bst) atomicAdd(&bst[3], (1.f - Cin)*Vp); } else ovf = 1;
    }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>n0-2||a1>n1-2||a2>n2-2) { if (!nrm) ovf = 1; continue; }
        long long c = ((long long)a0*n1 + a1)*n2 + a2;
        if (!cmk[c]) { if (!nrm) ovf = 1; continue; }
        float w = W[0][i]*W[1][j]*W[2][l]*sc*Vp;
        atomicAdd(&cV[c], w);
        for (int k=0;k<9;k++) atomicAdd(&cVF[c*9+k], w*Fp[k]);
    }
    if (ovf) atomicAdd(&szp[3], 1);
    if (floor) {
        float gg[3] = {g0, g1, g2};
        float s = 0.f, nn[3] = {q0, q1, q2}, pp[3] = {p0, p1, p2};
        for (int c=0;c<3;c++) s += (xp[c] + H*vp[c] + H*H*gg[c] - pp[c])*nn[c];
        if (s <= margin) { int q = atomicAdd(&szp[2], 1); nidx[q] = p; }
    }
}
// 사면체 상태: 셀 (정육면체 = Kuhn 사면체 6 개, 같은 부피) 의 F̄_c = cVF_c / cV_c, V_T = cV_c / 6.
//   네 노드 중 질량 0 인 노드가 있으면 그 사면체는 힘에서 뺀다 (V_T = 0; 질량 0 노드에 힘을 걸 수 없다) -- 뺀 부피는 bst[0] 에 기록.
__global__ void bsl_tfin(const int* szp, const long long* tcell, const long long* tets, const float* mn, const float* cV, const float* cVF,
                         float* VTs, float* FT, float* bst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    long long c = tcell[t];
    float V = cV[c];
    if (V > 0.f) {
        bool z = false;
        for (int i=0;i<4;i++) if (!(mn[tets[t*4+i]] > 0.f)) z = true;
        if (z) { VTs[t] = 0.f; atomicAdd(&bst[0], V/6.f); atomicAdd(&bst[1], 1.f); for (int k=0;k<9;k++) FT[t*9+k] = (k%4==0)?1.f:0.f; return; }
        VTs[t] = V/6.f;
        for (int k=0;k<9;k++) FT[t*9+k] = cVF[c*9+k]/V;
    } else {
        VTs[t] = 0.f;
        for (int k=0;k<9;k++) FT[t*9+k] = (k%4==0)?1.f:0.f;
    }
}
// 사면체 -> 밀집 셀 번호 (격자 만들 때)
__global__ void ex_tet_cell(const int* ipar, const unsigned char* tmark, const long long* tcm, long long* tcell) {
    long long s = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    long long P = (long long)ipar[0]*ipar[1]*ipar[2];
    if (s >= P*9 || !tmark[s]) return;
    tcell[tcm[s]] = s / 9;
}
// 입자의 프레임 변위 보간: bs = 1 -> u = Σ w(x_n - x_p⁰) U_n (B-스플라인), 0 -> Σλ U (사면체)
__device__ __forceinline__ void bsl_interp_u(long long p, const float* x, const float* Ucum, const long long* rows, const float* lamb,
                                             const float* gpar, const int* ipar, const long long* ncm, int bs, float* u, int nrm) {
    u[0]=u[1]=u[2]=0.f;
    if (!bs) {
        for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; float lm=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm*Ucum[nd*3+a]; }
        return;
    }
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    float y[3]; bsl_y(x + p*3, gpar, y);
    int b[3]; float W[3][3], D[3][3];
    for (int k=0;k<3;k++) bsl_w(y[k], b[k], W[k], D[k]);
    float Win = 0.f;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>n0-1||a1>n1-1||a2>n2-1) continue;
        long long nd = ncm[((long long)a0*n1 + a1)*n2 + a2];
        if (nd < 0) continue;
        float w = W[0][i]*W[1][j]*W[2][l];
        Win += w;
        for (int a=0;a<3;a++) u[a] += w*Ucum[nd*3+a];
    }
    if (nrm && Win > 0.f) for (int a=0;a<3;a++) u[a] /= Win;                     // 절 T: P2G 와 같은 정규화 (메시 안 노드 가중 합)
}
// 속도 바닥 접촉 (절 O 와 같은 판정·분배, 질량 가중만 B-스플라인): 바닥 후보 입자의 지금 위치 x_p⁰ + u (G2P 와 같은 보간) 높이 < 0 이면
//   P2G 와 같은 질량 가중 w m_p 를 27 노드 acc 에 -> ex_node_step_lag 가 frac 고착 (1 - acc/m) + 비관통
__global__ void bsl_vfl_contact(const int* szp, const long long* nidx, const float* x, const float* Ucum, const long long* rows, const float* lamb,
                                const float* gpar, const int* ipar, const long long* ncm, const float* m, int g2pbs,
                                float p0, float p1, float p2, float n0, float n1, float n2, float* acc, int* ncon, int nrm) {
    int nn_ = szp[2];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = nidx[q];
    float u[3]; bsl_interp_u(p, x, Ucum, rows, lamb, gpar, ipar, ncm, g2pbs, u, nrm);
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(s < 0.f)) return;
    int N0=ipar[0], N1=ipar[1], N2=ipar[2];
    float y[3]; bsl_y(x + p*3, gpar, y);
    int b[3]; float W[3][3], D[3][3];
    for (int k=0;k<3;k++) bsl_w(y[k], b[k], W[k], D[k]);
    float mp = m[p];
    if (nrm) {                                                                   // 절 T: P2G 와 같은 정규화 가중 (메시 안 노드 합 = 1)
        float Win = 0.f;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
            int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
            if (a0<0||a1<0||a2<0||a0>N0-1||a1>N1-1||a2>N2-1) continue;
            if (ncm[((long long)a0*N1 + a1)*N2 + a2] < 0) continue;
            Win += W[0][i]*W[1][j]*W[2][l];
        }
        if (Win > 0.f) mp /= Win;
    }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>N0-1||a1>N1-1||a2>N2-1) continue;
        long long nd = ncm[((long long)a0*N1 + a1)*N2 + a2];
        if (nd < 0) continue;
        atomicAdd(&acc[nd], W[0][i]*W[1][j]*W[2][l]*mp);
    }
    if (ncon) atomicAdd(ncon, 1);
}
// 프레임 끝 B-스플라인 G2P (질량 가중과 같은 w): x = x⁰ + Σ w U_n, v = Σ w v_n, F ← (I + Σ U_n ⊗ ∇w_n(x⁰)) F
__global__ void bsl_g2p(int N, const float* x, const float* Ucum, const float* Vn, const float* gpar, const int* ipar, const long long* ncm,
                        const float* Fp, float* x2, float* v2, float* F2, int nrm) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    const float* Ai = gpar + 3;
    float y[3]; bsl_y(x + p*3, gpar, y);
    int b[3]; float W[3][3], D[3][3];
    for (int k=0;k<3;k++) bsl_w(y[k], b[k], W[k], D[k]);
    // 절 T (nrm = 1): 메시 안 노드만, 정규화 가중 w~ = w / Win, 기울기 ∇w~ = (∇w - w~ ∇Win) / Win (Σ ∇w~ = 0 -- 강체 이동이면 A = I)
    double Win = 1.0, gW[3] = {0,0,0};
    if (nrm) {
        Win = 0.0;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
            int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
            if (a0<0||a1<0||a2<0||a0>n0-1||a1>n1-1||a2>n2-1) continue;
            if (ncm[((long long)a0*n1 + a1)*n2 + a2] < 0) continue;
            Win += (double)(W[0][i]*W[1][j]*W[2][l]);
            float gy0 = D[0][i]*W[1][j]*W[2][l], gy1 = W[0][i]*D[1][j]*W[2][l], gy2 = W[0][i]*W[1][j]*D[2][l];
            for (int c=0;c<3;c++) gW[c] += gy0*Ai[c] + gy1*Ai[3+c] + gy2*Ai[6+c];
        }
        if (!(Win > 0.0)) { Win = 1.0; gW[0]=gW[1]=gW[2]=0.0; }
    }
    double u[3]={0,0,0}, vv[3]={0,0,0}, A[9]={1,0,0,0,1,0,0,0,1};
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>n0-1||a1>n1-1||a2>n2-1) continue;
        long long nd = ncm[((long long)a0*n1 + a1)*n2 + a2];
        if (nd < 0) continue;
        double w = (double)(W[0][i]*W[1][j]*W[2][l]) / Win;
        float gy0 = D[0][i]*W[1][j]*W[2][l], gy1 = W[0][i]*D[1][j]*W[2][l], gy2 = W[0][i]*W[1][j]*D[2][l];
        double gx[3]; for (int c=0;c<3;c++) gx[c] = ((double)(gy0*Ai[c] + gy1*Ai[3+c] + gy2*Ai[6+c]) - w*gW[c]) / Win;
        for (int a=0;a<3;a++) { double Ua = Ucum[nd*3+a]; u[a] += w*Ua; vv[a] += w*(double)Vn[nd*3+a]; for (int c=0;c<3;c++) A[a*3+c] += Ua*gx[c]; }
    }
    for (int k=0;k<3;k++) { x2[p*3+k] = (float)((double)x[p*3+k] + u[k]); v2[p*3+k] = (float)vv[k]; }
    double F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    double F2d[9]; mm3(A, F, F2d);
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
}
// ======== 절 W (사용자 10-29): 셀 중심 커널 하나로 질량·운동량·F̄·V 모두 셀에, 노드는 일반 FEM 집중 질량 m_T/4 ========
// 셀 가중 = 절 T 의 셀 가중 그대로 (셀 중심 2 차 B-스플라인, 메시 안 셀만, 입자마다 메시 안 합 Cin 으로 정규화 -> Σ_c w~ = 1):
//   cM_c += w~ m_p, cMV_c += w~ m_p v_p, cV_c += w~ V_p, cVF_c += w~ V_p F_p.  바닥 후보 표시는 bsl_p2g 와 같다.
//   bst[2] = 메시 밖으로 갈 몫의 질량 Σ(1-Cin) m_p, bst[3] = 부피 Σ(1-Cin) V_p (정규화로 메시 안 셀에 옮긴 양)
__global__ void bsc_p2g(int N, const float* x, const float* v, const float* F, const float* m, const float* vol, const float* gpar, const int* ipar,
                        const unsigned char* cmk, float* cM, float* cMV, float* cV, float* cVF, int* szp,
                        int floor, float H, float g0, float g1, float g2, float p0, float p1, float p2, float q0, float q1, float q2, float margin,
                        long long* nidx, float* bst) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    int n0=ipar[0], n1=ipar[1], n2=ipar[2];
    float xp[3] = {x[p*3], x[p*3+1], x[p*3+2]}, vp[3] = {v[p*3], v[p*3+1], v[p*3+2]};
    float y[3]; bsl_y(xp, gpar, y);
    float mp = m[p], Vp = vol[p];
    int b[3]; float W[3][3], D[3][3];
    for (int k=0;k<3;k++) bsl_w(y[k] - 0.5f, b[k], W[k], D[k]);
    float Cin = 0.f;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>n0-2||a1>n1-2||a2>n2-2) continue;
        if (!cmk[((long long)a0*n1 + a1)*n2 + a2]) continue;
        Cin += W[0][i]*W[1][j]*W[2][l];
    }
    if (!(Cin > 0.f)) { atomicAdd(&szp[3], 1); return; }
    float sc = 1.f/Cin;
    if (bst) { atomicAdd(&bst[2], (1.f - Cin)*mp); atomicAdd(&bst[3], (1.f - Cin)*Vp); }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[p*9+k];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>n0-2||a1>n1-2||a2>n2-2) continue;
        long long c = ((long long)a0*n1 + a1)*n2 + a2;
        if (!cmk[c]) continue;
        float w = W[0][i]*W[1][j]*W[2][l]*sc;
        float wm = w*mp, wv = w*Vp;
        atomicAdd(&cM[c], wm);
        for (int k=0;k<3;k++) atomicAdd(&cMV[c*3+k], wm*vp[k]);
        atomicAdd(&cV[c], wv);
        for (int k=0;k<9;k++) atomicAdd(&cVF[c*9+k], wv*Fp[k]);
    }
    if (floor) {
        float gg[3] = {g0, g1, g2};
        float s = 0.f, nn[3] = {q0, q1, q2}, pp[3] = {p0, p1, p2};
        for (int c=0;c<3;c++) s += (xp[c] + H*vp[c] + H*H*gg[c] - pp[c])*nn[c];
        if (s <= margin) { int q = atomicAdd(&szp[2], 1); nidx[q] = p; }
    }
}
// 셀 -> 사면체 -> 노드 (일반 FEM 집중 질량): 셀 = 같은 부피 Kuhn 사면체 6 개 -> m_T = cM_c/6, (mv)_T = cMV_c/6, V_T = cV_c/6, F̄_T = cVF_c/cV_c;
//   노드 m_n = Σ_{T∋n} m_T/4, (mv)_n = Σ_{T∋n} (mv)_T/4  (속도는 ex_vfin 이 (mv)_n / m_n)
__global__ void bsc_tfin(const int* szp, const long long* tcell, const long long* tets, const float* cM, const float* cMV, const float* cV, const float* cVF,
                         float* VTs, float* FT, float* mn, float* mv) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    long long c = tcell[t];
    float V = cV[c];
    if (V > 0.f) { VTs[t] = V/6.f; for (int k=0;k<9;k++) FT[t*9+k] = cVF[c*9+k]/V; }
    else { VTs[t] = 0.f; for (int k=0;k<9;k++) FT[t*9+k] = (k%4==0)?1.f:0.f; }
    float q = cM[c]*(1.f/24.f);
    float pv[3] = {cMV[c*3]*(1.f/24.f), cMV[c*3+1]*(1.f/24.f), cMV[c*3+2]*(1.f/24.f)};
    if (q > 0.f) for (int i=0;i<4;i++) { long long nd = tets[t*4+i]; atomicAdd(&mn[nd], q); for (int k=0;k<3;k++) atomicAdd(&mv[nd*3+k], pv[k]); }
}
// 속도 바닥 접촉 (절 O·T 와 같은 판정 -- 지금 위치 = G2P 와 같은 보간; 분배는 P2G 와 같은 질량 경로): 접촉 입자 m_p 를 셀 가중 w~_pc 로 셀 accC 에,
//   bsc_acc_tet 이 사면체마다 accC_c/24 를 네 노드 acc 에 (= 노드 질량과 같은 m_T/4 경로) -> ex_node_step_lag 의 frac 고착 (1 - acc/m) + 비관통
__global__ void bsc_vfl_contact(const int* szp, const long long* nidx, const float* x, const float* Ucum, const long long* rows, const float* lamb,
                                const float* gpar, const int* ipar, const long long* ncm, const unsigned char* cmk, const float* m, int g2pbs,
                                float p0, float p1, float p2, float n0, float n1, float n2, float* accC, int* ncon) {
    int nn_ = szp[2];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    long long p = nidx[q];
    float u[3]; bsl_interp_u(p, x, Ucum, rows, lamb, gpar, ipar, ncm, g2pbs, u, 1);
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(s < 0.f)) return;
    int N0=ipar[0], N1=ipar[1], N2=ipar[2];
    float y[3]; bsl_y(x + p*3, gpar, y);
    int b[3]; float W[3][3], D[3][3];
    for (int k=0;k<3;k++) bsl_w(y[k] - 0.5f, b[k], W[k], D[k]);
    float Cin = 0.f;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>N0-2||a1>N1-2||a2>N2-2) continue;
        if (!cmk[((long long)a0*N1 + a1)*N2 + a2]) continue;
        Cin += W[0][i]*W[1][j]*W[2][l];
    }
    if (!(Cin > 0.f)) return;
    float mp = m[p]/Cin;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int l=0;l<3;l++) {
        int a0=b[0]+i, a1=b[1]+j, a2=b[2]+l;
        if (a0<0||a1<0||a2<0||a0>N0-2||a1>N1-2||a2>N2-2) continue;
        long long c = ((long long)a0*N1 + a1)*N2 + a2;
        if (!cmk[c]) continue;
        atomicAdd(&accC[c], W[0][i]*W[1][j]*W[2][l]*mp);
    }
    if (ncon) atomicAdd(ncon, 1);
}
__global__ void bsc_acc_tet(const int* szp, const long long* tcell, const long long* tets, const float* accC, float* acc) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float q = accC[tcell[t]]*(1.f/24.f);
    if (q > 0.f) for (int i=0;i<4;i++) atomicAdd(&acc[tets[t*4+i]], q);
}
// 입자 재배열 (프레임 시작): dst[i] = src[perm[i]] (행 길이 k 의 float)
__global__ void ex_gather_rows(int N, int k, const long long* perm, const float* src, float* dst) {
    long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long long)N*k) return;
    long long r = i / k; int c = i % k;
    dst[i] = src[perm[r]*k + c];
}
// 되돌리기 (프레임 끝): dst[perm[i]] = src[i]
__global__ void ex_scatter_rows(int N, int k, const long long* perm, const float* src, float* dst) {
    long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long long)N*k) return;
    long long r = i / k; int c = i % k;
    dst[perm[r]*k + c] = src[i];
}
// ======== 절 V: 노드 적분 FEM (사용자 10-29 "그것도 일단 해 보라고 해") -- 평균 노드 변형 사면체 (Bonet & Burton 1998, Dohrmann et al. 2000) ========
// 틀·질량·바닥·F̄ 상태·G2P 는 절 R lag 판 그대로. 바뀐 것은 내부력 하나:
//   서브스텝마다
//   (1) nint_cell : F_T = A_T(U) F̄_T,  사면체 버퍼 X_T = (V_T/4) F_T                (V_T > 0 사면체만)
//   (2) nint_node : 노드 -> 사면체 목록으로 S_n = Σ_{T∋n} X_T, V_n = Σ V_T/4, F_n = S_n / V_n (= Σ V_T F_T / Σ V_T),
//                   P_n = P_FCR(F_n) (ex_tet_force_R 와 같은 FCR 식)
//   (3) nint_force: f_a -= (V_T/4) [(1-β) Σ_{n∈T} P_n + 4β P_FCR(F_T)] F̄_Tᵀ ∇λ_a
//   = 에너지 E(U) = (1-β) Σ_n V_n ψ(F_n) + β Σ_T V_T ψ(F_T) 의 -∂E/∂U_a 그대로
//     (∂F_n/∂U_a·δ = Σ_{T∋n,a} (V_T/4)/V_n δ ⊗ (F̄_Tᵀ ∇λ_a) 이라 V_n 이 상쇄된다). β = 0 이 순수 노드 적분, β > 0 은 셀 적분 혼합 (영에너지 모드 안정화 선택지).
//   사면체마다 Σ_a ∇λ_a = 0 -> 내력 합 0 (운동량 보존), E 가 회전 불변 -> 각운동량 보존.
__device__ __forceinline__ void fcr_P(const float* X, float mu, float lam, float* P) {
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9];
    polar(X, Q); cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
}
__device__ __forceinline__ float nint_AX(int t, const long long* tets, const float* U, const float* gl, const float* FT, long long* n, float* G, float* X) {
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    mm3(A, Fb, X);
    return det3(A);
}
// (1) 사면체마다 (V_T/4) F_T 를 사면체 버퍼에 (원자 합 없음 -- 노드 쪽이 목록으로 모은다; 목록 순서는 ex_node_tets 원자 슬롯이라 합 순서는 실행마다 다를 수 있다)
__global__ void nint_cell(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT, float* XT, float* rst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12], X[9];
    float dA = nint_AX(t, tets, U, gl, FT, n, G, X);
    if (rst && !(dA > 0.f)) atomicAdd(&rst[0], 1.f);
    float q = 0.25f*V;
    for (int k=0;k<9;k++) XT[t*9+k] = q*X[k];
}
// (2) 노드마다 노드 -> 사면체 목록 (ex_node_tets, 프레임마다) 으로 S_n = Σ (V_T/4) F_T, V_n = Σ V_T/4 -> F_n = S_n / V_n, P_n = P_FCR(F_n)
__global__ void nint_node(const int* szp, const int* ntc, const int* nt, const float* VT, const float* XT, float* Pn, float* Vn, float mu, float lam) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= M) return;
    int k = min(ntc[n], 24);
    float S[9] = {0,0,0,0,0,0,0,0,0}, v = 0.f;
    for (int s=0;s<k;s++) {
        int t = nt[n*24+s] >> 2;
        float V = VT[t];
        if (!(V > 0.f)) continue;
        v += 0.25f*V;
        for (int c=0;c<9;c++) S[c] += XT[(long long)t*9+c];
    }
    float P[9] = {0,0,0,0,0,0,0,0,0};
    if (v > 0.f) {
        float X[9]; for (int c=0;c<9;c++) X[c] = S[c]/v;
        fcr_P(X, mu, lam, P);
    }
    for (int c=0;c<9;c++) Pn[n*9+c] = P[c];
    if (Vn) Vn[n] = v;
}
__global__ void nint_force(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT, const float* Pn,
                           float mu, float lam, float beta, float* f) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float S[9];
    for (int k=0;k<9;k++) S[k] = (1.f-beta)*(Pn[n[0]*9+k] + Pn[n[1]*9+k] + Pn[n[2]*9+k] + Pn[n[3]*9+k]);
    if (beta > 0.f) {
        long long n2[4]; float G2[12], X[9], PT[9];
        nint_AX(t, tets, U, gl, FT, n2, G2, X);
        fcr_P(X, mu, lam, PT);
        for (int k=0;k<9;k++) S[k] += 4.f*beta*PT[k];
    }
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    float Gm[9]; mmt3(S, Fb, Gm);
    float q = 0.25f*V;
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -q*s);
    }
}
// ======== 절 AA: 셀 에너지 = 셀 안 입자 에너지 평균의 2 차 근사 (사용자 10-30 "셀 평균 ψ 의 근사로") ========
//   E_T = V_T [ ψ(X̄) + ½ ⟨∂²ψ(X̄), S⟩ ],  X̄ = A_T F̄_T,  S = (A⊗I) Σ_F (A⊗I)ᵀ  (= vec(A F_p) 의 공분산, vec 은 행 우선 9)
//   Σ_F = Σ_p V_p vec(F_p - F̄) vec(F_p - F̄)ᵀ / V_T  -- 프레임 시작 P2G 에서 한 번 (두 번째 패스, 대칭 9x9 위 삼각 45 개 저장)
//   -∂E_T/∂x = -V_T [ (P(X̄) + ∂_X q) F̄ᵀ + G_S ] ∇λ_i
//     G_S[a][b] = Σ_c Σ_m H[(a,c)][m] T[m][(b,c)],  T = (A⊗I) Σ_F   (∂S/∂A 몫, H = ∂²ψ(X̄))
//     ∂_X q (S 고정, ∂³ψ 몫 -- 근사 없음, 닫힌 식):
//       λ 몫 q_λ = λ/2 [ cᵀ S c + 2 (det X - sg) X:C_S ]  (c = cof X, C_S = Σ_q cof(s_q) = S 의 2x2 소행렬식 합)
//            ∂_X q_λ = λ [ coflin(X, Sc) + (X:C_S) cof X + (det X - sg) C_S ]
//       μ 몫 q_μ = μ [ tr S - h ],  h = tr(M⁻¹ B S Bᵀ),  M = tr(Ŝ) I - Ŝ (X = R Ŝ 극분해), B s = ax(Rᵀ s - sᵀ R)
//            (s : DR[s] = aᵀ M⁻¹ a 에서 -- FCR 의 μ 헤시안 2μ(I - DR)),  ∂h/∂X_m = -⟨dM_m, M⁻¹KM⁻¹⟩ + 2 Σ_i ax_i(dR_mᵀ y_i - y_iᵀ dR_m)
//            (dR = R [w]×, w = M⁻¹ B e_m, dŜ = sym(Rᵀ e_m - [w]× Ŝ), K = B S Bᵀ, y_i = (S Bᵀ M⁻¹) 의 i 열)
//   H: hmode 0 = 기존 해석 헤시안 장치 함수 kmanual (전진 미분 극분해 10 회), 1 = 같은 헤시안의 닫힌 식 (H s = 2μ(s - R[M⁻¹ B s]×) + λ[(c:s) c + (det - sg) coflin(X, s)])
//   σ_min(X̄) ≤ 0.01 이거나 tr Σ_F = 0 (입자 1 개 셀 등) 이면 보정 0 (= 기존 ψ(X̄)).
__device__ __forceinline__ int cq_si(int i, int j) { if (i > j) { int t=i; i=j; j=t; } return i*9 - (i*(i-1))/2 + (j-i); }
__global__ void ex_cov_acc(int N, const long long* pinv, const int* nslot, int nit, const float* vol, const float* Fp, const float* FT, float* CS) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    if (!(nslot[p] < 0 || nit)) return;
    long long t = pinv[p];
    float d[9]; for (int k=0;k<9;k++) d[k] = Fp[p*9+k] - FT[t*9+k];
    float V = vol[p];
    int s = 0;
    for (int i=0;i<9;i++) for (int j=i;j<9;j++) { atomicAdd(&CS[t*45+s], V*d[i]*d[j]); s++; }
}
__global__ void ex_cov_fin(const int* szp, const float* VT, float* CS) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    float iv = V > 0.f ? 1.f/V : 0.f;
    for (int k=0;k<45;k++) CS[(long long)t*45+k] *= iv;
}
__device__ __forceinline__ void cq_ax(const float* Z, float* a) { a[0] = Z[7]-Z[5]; a[1] = Z[2]-Z[6]; a[2] = Z[3]-Z[1]; }   // ax(Z - Zᵀ)
__device__ __forceinline__ void cq_Bs(const float* R, const float* s, float* a) {   // a = ax(Rᵀ s - sᵀ R)
    float Y[9]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) Y[i*3+j] = R[i]*s[j] + R[3+i]*s[3+j] + R[6+i]*s[6+j];
    cq_ax(Y, a);
}
// 보정 응력: dq (∂_X q, 9) 와 G_S (3x3). 반환 0 이면 보정 없음.
__device__ int cq_corr(const float* A, const float* X, const float* Q, const float* cf, float dt, float sg, float smin, const float* CSt,
                       float mu, float lam, int hmode, float* dq, float* GS) {
    float tr = CSt[cq_si(0,0)] + CSt[cq_si(1,1)] + CSt[cq_si(2,2)] + CSt[cq_si(3,3)] + CSt[cq_si(4,4)] + CSt[cq_si(5,5)] + CSt[cq_si(6,6)] + CSt[cq_si(7,7)] + CSt[cq_si(8,8)];
    if (!(tr > 0.f) || !(smin > 0.01f)) return 0;
    float W1[81], W2[81];                       // W1 = Σ_F -> S_X,  W2 = T
    for (int i=0;i<9;i++) for (int j=0;j<9;j++) W1[i*9+j] = CSt[cq_si(i,j)];
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) for (int j=0;j<9;j++)
        W2[(r*3+c)*9+j] = A[r*3]*W1[c*9+j] + A[r*3+1]*W1[(3+c)*9+j] + A[r*3+2]*W1[(6+c)*9+j];
    for (int i=0;i<9;i++) for (int r=0;r<3;r++) for (int c=0;c<3;c++)                   // S_X = T Lᵀ (W1 덮어씀 -- Σ_F 는 더 안 쓴다)
        W1[i*9+r*3+c] = W2[i*9+c]*A[r*3] + W2[i*9+3+c]*A[r*3+1] + W2[i*9+6+c]*A[r*3+2];
    // 극분해 Ŝ = sym(Qᵀ X), M = tr Ŝ I - Ŝ, M⁻¹
    float Sh[9];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) Sh[i*3+j] = Q[i]*X[j] + Q[3+i]*X[3+j] + Q[6+i]*X[6+j];
    for (int i=0;i<3;i++) for (int j=i+1;j<3;j++) { float m_ = 0.5f*(Sh[i*3+j]+Sh[j*3+i]); Sh[i*3+j] = m_; Sh[j*3+i] = m_; }
    float tS = Sh[0]+Sh[4]+Sh[8];
    float Mm[9], Mi[9];
    for (int k=0;k<9;k++) Mm[k] = -Sh[k];
    Mm[0] += tS; Mm[4] += tS; Mm[8] += tS;
    cof3(Mm, Mi); { float dm = Mm[0]*Mi[0]+Mm[1]*Mi[1]+Mm[2]*Mi[2]; float idm = 1.f/dm; for (int k=0;k<9;k++) Mi[k] *= idm; }   // M 대칭 -> cof 대칭 = adj
    float Bm[27];                               // B (3x9): B[:,m] = ax(Rᵀ e_m - e_mᵀ R)
    for (int m=0;m<9;m++) { float e[9] = {0,0,0,0,0,0,0,0,0}; e[m] = 1.f; float a3[3]; cq_Bs(Q, e, a3); Bm[m] = a3[0]; Bm[9+m] = a3[1]; Bm[18+m] = a3[2]; }
    float dd = dt - sg;
    // ---- G_S = Σ_c (H T)[(a,c),(b,c)] ----
    for (int k=0;k<9;k++) GS[k] = 0.f;
    if (hmode == 0) {
        float H[81]; const float I3[9] = {1,0,0, 0,1,0, 0,0,1};
        kmanual(X, I3, mu, lam, 1.f, smin, H);
        for (int i=0;i<9;i++) for (int j=i+1;j<9;j++) { float m_ = 0.5f*(H[i*9+j]+H[j*9+i]); H[i*9+j] = m_; H[j*9+i] = m_; }
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float s = 0.f;
            for (int c=0;c<3;c++) for (int m=0;m<9;m++) s += H[(a*3+c)*9+m]*W2[m*9+b*3+c];
            GS[a*3+b] = s; }
    } else {
        for (int b=0;b<3;b++) for (int c=0;c<3;c++) {          // T 의 열 (b,c) 에 H 를 곱한다
            float s9[9]; for (int m=0;m<9;m++) s9[m] = W2[m*9+b*3+c];
            float a3[3], w[3]; cq_Bs(Q, s9, a3);
            for (int i=0;i<3;i++) w[i] = Mi[i*3]*a3[0] + Mi[i*3+1]*a3[1] + Mi[i*3+2]*a3[2];
            float Wx[9] = {0.f,-w[2],w[1], w[2],0.f,-w[0], -w[1],w[0],0.f};
            float dR[9]; mm3(Q, Wx, dR);
            float cs = 0.f; for (int k=0;k<9;k++) cs += cf[k]*s9[k];
            float cl[9]; coflin(X, s9, cl);
            for (int a=0;a<3;a++) {
                float hs_ = 0.f;
                for (int cc=0;cc<3;cc++) if (cc == c) { int k = a*3+cc; hs_ = 2.f*mu*(s9[k]-dR[k]) + lam*(cs*cf[k] + dd*cl[k]); }
                GS[a*3+b] += hs_;
            }
        }
    }
    // ---- λ 몫 ∂_X q_λ ----
    float Sc[9], CSx[9];
    for (int i=0;i<9;i++) { float s = 0.f; for (int j=0;j<9;j++) s += W1[i*9+j]*cf[j]; Sc[i] = s; }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        int i1 = (i+1)%3, i2 = (i+2)%3, j1 = (j+1)%3, j2 = (j+2)%3;
        CSx[i*3+j] = W1[(i1*3+j1)*9 + i2*3+j2] - W1[(i1*3+j2)*9 + i2*3+j1];
    }
    float XC = 0.f; for (int k=0;k<9;k++) XC += X[k]*CSx[k];
    float cl[9]; coflin(X, Sc, cl);
    for (int k=0;k<9;k++) dq[k] = lam*(cl[k] + XC*cf[k] + dd*CSx[k]);
    // ---- μ 몫 -μ ∂h/∂X ----
    float BS[27];
    for (int l=0;l<3;l++) for (int m=0;m<9;m++) { float s = 0.f; for (int j=0;j<9;j++) s += Bm[l*9+j]*W1[j*9+m]; BS[l*9+m] = s; }
    float K[9];
    for (int l=0;l<3;l++) for (int r=0;r<3;r++) { float s = 0.f; for (int m=0;m<9;m++) s += BS[l*9+m]*Bm[r*9+m]; K[l*3+r] = s; }
    float T1[9], MKM[9]; mm3(Mi, K, T1); mm3(T1, Mi, MKM);
    float Yc[27];                               // Yc[i*9+m] = (S Bᵀ M⁻¹)[m][i]
    for (int i=0;i<3;i++) for (int m=0;m<9;m++) Yc[i*9+m] = BS[m]*Mi[i] + BS[9+m]*Mi[3+i] + BS[18+m]*Mi[6+i];
    for (int m=0;m<9;m++) {
        int k = m/3, j = m%3;
        float a3[3] = {Bm[m], Bm[9+m], Bm[18+m]}, w[3];
        for (int i=0;i<3;i++) w[i] = Mi[i*3]*a3[0] + Mi[i*3+1]*a3[1] + Mi[i*3+2]*a3[2];
        float Wx[9] = {0.f,-w[2],w[1], w[2],0.f,-w[0], -w[1],w[0],0.f};
        float dR[9]; mm3(Q, Wx, dR);
        float WS[9]; mm3(Wx, Sh, WS);
        float dS[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) dS[a*3+b] = ((b==j) ? Q[k*3+a] : 0.f) - WS[a*3+b];
        for (int a=0;a<3;a++) for (int b=a+1;b<3;b++) { float m_ = 0.5f*(dS[a*3+b]+dS[b*3+a]); dS[a*3+b] = m_; dS[b*3+a] = m_; }
        float tdS = dS[0]+dS[4]+dS[8];
        float t1 = 0.f;
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float dM = ((a==b) ? tdS : 0.f) - dS[a*3+b]; t1 += dM*MKM[a*3+b]; }
        float t2 = 0.f;
        for (int i=0;i<3;i++) {
            float Z[9];                         // Z = dRᵀ y_i
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) Z[a*3+b] = dR[a]*Yc[i*9+b] + dR[3+a]*Yc[i*9+3+b] + dR[6+a]*Yc[i*9+6+b];
            float az[3]; cq_ax(Z, az);
            t2 += az[i];
        }
        dq[m] += -mu*(-t1 + 2.f*t2);
    }
    return 1;
}
__global__ void ex_tet_force_CQ(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT, const float* CS,
                                float mu, float lam, float* f, float* rst, int hmode, float* gdbg) {
    // gdbg (검산 전용, 보통 null): 사면체마다 [P F̄ᵀ (9), ∂_X q F̄ᵀ (9), G_S (9)]
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    if (rst) { float dA = det3(A); if (!(dA > 0.f)) atomicAdd(&rst[0], 1.f); }
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    float X[9]; mm3(A, Fb, X);
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9], P[9], Gm[9];
    polar(X, Q); cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
    float dq[9], GS[9];
    int on = cq_corr(A, X, Q, cf, dt, sg, sqrtf(fmaxf(l[0], 0.f)), CS + (long long)t*45, mu, lam, hmode, dq, GS);
    if (gdbg) { float g0[9]; mmt3(P, Fb, g0); for (int k=0;k<9;k++) { gdbg[(long long)t*27+k] = g0[k]; gdbg[(long long)t*27+9+k] = 0.f; gdbg[(long long)t*27+18+k] = 0.f; }
        if (on) { float g1[9]; mmt3(dq, Fb, g1); for (int k=0;k<9;k++) { gdbg[(long long)t*27+9+k] = g1[k]; gdbg[(long long)t*27+18+k] = GS[k]; } } }
    if (on) for (int k=0;k<9;k++) P[k] += dq[k];
    mmt3(P, Fb, Gm);
    if (on) { for (int k=0;k<9;k++) Gm[k] += GS[k]; if (rst) atomicAdd(&rst[11], 1.f); }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*s);
    }
}

// ======== 절 AB (갈래 A, 사용자 10-30 "해볼 수 있는 거 다"): 프레임 끝 G2P 의 입자 F 갱신 방식 ========
// 진단 (절 Y·AA): 입자 F 이웃 벌어짐이 정답의 10~20 배 -- 이웃 사면체의 서로 다른 A_T 를 F_p <- A_T F_p 로 각자 누적. 위치·속도 G2P 는 그대로, F 갱신만 바꾼다.
// fa_tet_acc: 점유 사면체 (V_T > 0) 의 X_T 를 네 노드에 V_T 가중 합 -- NA[n*10+k] += V_T X_T[k], NA[n*10+9] += V_T
//   which 0: X_T = A_T = I + Σ_i U_i ∇λ_iᵀ (프레임 셀 사상), 1: X_T = F̄_T (프레임 시작 셀 평균 F, P2G 것)
__global__ void fa_tet_acc(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT, int which, float* NA) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; for (int i=0;i<4;i++) n[i] = tets[t*4+i];
    float X[9];
    if (which == 0) {
        for (int k=0;k<9;k++) X[k] = (k%4==0) ? 1.f : 0.f;
        for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u = U[n[i]*3+c]; for (int b=0;b<3;b++) X[c*3+b] += u*gl[t*12+i*3+b]; }
    } else {
        for (int k=0;k<9;k++) X[k] = FT[t*9+k];
    }
    for (int i=0;i<4;i++) { for (int k=0;k<9;k++) atomicAdd(&NA[n[i]*10+k], V*X[k]); atomicAdd(&NA[n[i]*10+9], V); }
}
// ex_g2p_lagF: 위치·속도는 ex_g2p_lag (vq 0, C 없음) 와 같은 식, F 갱신만 mode 로:
//   0 기존: F <- A_T F
//   1 bar  (셀 균질화/부분 이완): F <- A_T [(1-β) F + β F̄_T]          (β 1 = 셀 평균으로 덮어씀)
//   2 nbar (이웃 이완): F <- A_T [(1-β) F + β F̃],  F̃ = Σ_i λ_i F̄_n (F̄_n = 노드에 닿는 점유 사면체 F̄_T 의 V_T 가중 평균, NA which 1)
//   3 navg (매끈한 속도 기울기): F <- [(1-β) A_T + β Ã] F,  Ã = Σ_i λ_i Ā_n (Ā_n = 노드에 닿는 점유 사면체 A_T 의 V_T 가중 평균, NA which 0 -- simplex.nodeavg_jac_pre 꼴)
//   4 polar (회전/신장 분리): F = R S, F̃ = R̃ S̃ (극분해) -> F <- A_T R [(1-β) S + β S̃]  (회전은 입자 것, 신장만 이웃 (노드 평균 F̄) 쪽으로)
//   5 navgbar (3 + 2): F <- [(1-β) A_T + β Ã] [(1-β2) F + β2 F̃]  (NA = A 노드 평균, NB = F̄ 노드 평균)
__global__ void ex_g2p_lagF(int N, const float* x, const float* Ucum, const float* Vn, const long long* rows, const float* lamb, const float* glT,
                            const long long* tinv, const float* Fp, float* x2, float* v2, float* F2, const float* FT, const float* NA, int mode, float beta, const float* NB, float beta2,
                            float* C2) {
    // C2 (선택, 절 AK --affine): 입자 속도 기울기 C = (Σ v_n ⊗ ∇λ_n) A_T⁻¹ (ex_g2p_lag 의 C2 와 같은 식) -- 다음 프레임 P2G 아핀 전달
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double u[3]={0,0,0}, vv[3]={0,0,0}, A[9]={1,0,0,0,1,0,0,0,1}, Nt[9]={0,0,0,0,0,0,0,0,0}, Nb[9]={0,0,0,0,0,0,0,0,0}, Gv[9]={0,0,0,0,0,0,0,0,0};
    long long t = tinv[p];
    const float* gl = glT + t*12;
    for (int i=0;i<4;i++){ long long nd=rows[p*4+i]; double lm=lamb[p*4+i];
        for (int a=0;a<3;a++){ double Ua=Ucum[nd*3+a]; double Va=(double)Vn[nd*3+a]; u[a]+=lm*Ua; vv[a]+=lm*Va;
            for (int b=0;b<3;b++) { A[a*3+b]+=Ua*gl[i*3+b]; Gv[a*3+b]+=Va*gl[i*3+b]; } }
        if (mode >= 2) { double W = (double)NA[nd*10+9]; if (W > 0.0) for (int k=0;k<9;k++) Nt[k] += lm*(double)NA[nd*10+k]/W; }
        if (mode == 5) { double W = (double)NB[nd*10+9]; if (W > 0.0) for (int k=0;k<9;k++) Nb[k] += lm*(double)NB[nd*10+k]/W; } }
    for (int k=0;k<3;k++) { x2[p*3+k] = (float)((double)x[p*3+k] + u[k]); v2[p*3+k] = (float)vv[k]; }
    double F[9]; for (int k=0;k<9;k++) F[k]=Fp[p*9+k];
    double B = beta, G[9], F2d[9];
    if (mode == 1) { for (int k=0;k<9;k++) G[k] = ((double)1-B)*F[k] + B*(double)FT[t*9+k]; mm3(A, G, F2d); }
    else if (mode == 2) { for (int k=0;k<9;k++) G[k] = ((double)1-B)*F[k] + B*Nt[k]; mm3(A, G, F2d); }
    else if (mode == 3) { for (int k=0;k<9;k++) G[k] = ((double)1-B)*A[k] + B*Nt[k]; mm3(G, F, F2d); }
    else if (mode == 4) {
        double R[9], Rt[9], S[9], St[9], RS[9];
        polar(F, R); polar(Nt, Rt);
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) { double s1=0, s2=0; for (int k=0;k<3;k++) { s1 += R[k*3+r]*F[k*3+c]; s2 += Rt[k*3+r]*Nt[k*3+c]; } S[r*3+c]=s1; St[r*3+c]=s2; }
        for (int k=0;k<9;k++) G[k] = ((double)1-B)*S[k] + B*St[k];
        mm3(R, G, RS); mm3(A, RS, F2d);
    }
    else if (mode == 5) { double H[9]; for (int k=0;k<9;k++) { G[k] = ((double)1-B)*A[k] + B*Nt[k]; H[k] = ((double)1-(double)beta2)*F[k] + (double)beta2*Nb[k]; } mm3(G, H, F2d); }
    else mm3(A, F, F2d);
    for (int k=0;k<9;k++) F2[p*9+k] = (float)F2d[k];
    if (C2) {
        double cf[9]; cof3(A, cf);
        double dA = A[0]*cf[0]+A[1]*cf[1]+A[2]*cf[2];
        double Ci[9];
        if (fabs(dA) > 1e-12) { for (int i=0;i<3;i++) for (int j=0;j<3;j++) Ci[i*3+j] = cf[j*3+i]/dA; }
        else { for (int k=0;k<9;k++) Ci[k] = (k%4==0)?1.0:0.0; }
        double Cc[9]; mm3(Gv, Ci, Cc);
        for (int k=0;k<9;k++) C2[p*9+k] = (float)Cc[k];
    }
}
}
extern "C" {
// ======== 절 AD: 희소 셀 병합 (사용자 10-30 "해볼 수 있는 거 다", 갈래 B) ========
// 묶음 g (사면체 집합, 루트 r) 의 셀 에너지 E_g = V_g ψ(Ā_g F̄_g),  Ā_g = Σ_{t∈g} V_t A_t / V_g (입자 부피 가중),
//   F̄_g = Σ V_t F̄_t / V_g,  V_g = Σ V_t (입자 부피).  힘 f_{t,i} = -V_t P(Ā F̄) F̄ᵀ ∇λ_{t,i}  (= -∂E_g/∂x, 묶인 사면체마다 입자 부피 V_t 비례 분배 -- 노드 질량 m_T/4 와 같은 비라 강성/질량 비 유지).
//   (처음 판은 기하 부피 = 같은 몫 (V_g/n) 이었는데 입자 1 개 사면체의 가벼운 노드에 큰 강성이 걸려 프레임 2 에 발산 -- 연기 시험)
//   Σ_i ∇λ_{t,i} = 0 -> 운동량 보존, ψ 회전 불변 + Ā 가 강체 회전과 같이 돈다 -> 각운동량 보존 (에너지의 기울기이므로).
// grp[t] = 루트 (묶음 크기 ≥ 2) 또는 -1 (홀로: ex_tet_force_R 과 같은 식). AG = 서브스텝 홀짝 두 벌 (G1 이 이번 벌에 쌓고, G2 가 읽은 뒤 다음 벌을 비운다).
__device__ __forceinline__ void sp_fcr_P(const float* X, float mu, float lam, float* P) {
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9];
    polar(X, Q); cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
}
__global__ void sp_force_g1(const int* szp, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                            float mu, float lam, float* f, const int* grp, float* AG, float* rst) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    if (rst && !(det3(A) > 0.f)) atomicAdd(&rst[0], 1.f);
    int r = grp[t];
    if (r >= 0) { for (int k=0;k<9;k++) atomicAdd(&AG[(long long)r*9+k], V*A[k]); return; }
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    float X[9]; mm3(A, Fb, X);
    float P[9], Gm[9]; sp_fcr_P(X, mu, lam, P);
    mmt3(P, Fb, Gm);
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*s);
    }
}
__global__ void sp_force_g2(const int* szp, const long long* tets, const float* gl, float mu, float lam, float* f, const int* grp, const float* AG, float* AGz,
                            const float* FG, const float* VG, const float* VT, const float* U, float hg) {
    // hg > 0: 모래시계 억제 γ μ Σ_t V_t |A_t - Ā|²  ->  사면체 t 힘에 -V_t 2 γ μ (A_t - Ā) ∇λ_{t,i} 더함 (Σ_t V_t (A_t - Ā) = 0 이라 Ā 경유 몫 없음)
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) return;
    int r = grp[t];
    if (r < 0) return;
    float vg = VG[r];
    float A[9]; for (int k=0;k<9;k++) A[k] = AG[(long long)r*9+k]/vg;
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FG[(long long)r*9+k];
    float X[9]; mm3(A, Fb, X);
    float P[9], Gm[9]; sp_fcr_P(X, mu, lam, P);
    mmt3(P, Fb, Gm);
    float w = VT[t];
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    if (hg > 0.f) {
        float At[9] = {1,0,0, 0,1,0, 0,0,1};
        for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) At[c*3+b] += u*G[i*3+b]; }
        for (int k=0;k<9;k++) Gm[k] += 2.f*hg*mu*(At[k]-A[k]);
    }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -w*s);
    }
    if (AGz) for (int k=0;k<9;k++) AGz[(long long)r*9+k] = 0.f;              // 다음 서브스텝 벌 비우기 (같은 묶음 모두 0 을 쓴다)
}
}  // extern "C" (절 AD)

//@@ _K 16101

extern "C" {
__device__ void jac3(double a[3][3], double v[3][3]) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) v[i][j] = (i==j) ? 1.0 : 0.0;
    for (int sw=0; sw<40; sw++) {
        double off = a[0][1]*a[0][1] + a[0][2]*a[0][2] + a[1][2]*a[1][2];
        double dg = a[0][0]*a[0][0] + a[1][1]*a[1][1] + a[2][2]*a[2][2];
        if (!(off > 1e-34*dg)) break;
        for (int p=0;p<2;p++) for (int q=p+1;q<3;q++) {
            if (a[p][q] == 0.0) continue;
            double th = 0.5*(a[q][q]-a[p][p])/a[p][q];
            double t = (th >= 0.0 ? 1.0 : -1.0)/(fabs(th) + sqrt(th*th + 1.0));
            double c = 1.0/sqrt(t*t + 1.0), s = t*c;
            for (int k=0;k<3;k++) { double x1 = a[k][p], x2 = a[k][q]; a[k][p] = c*x1 - s*x2; a[k][q] = s*x1 + c*x2; }
            for (int k=0;k<3;k++) { double x1 = a[p][k], x2 = a[q][k]; a[p][k] = c*x1 - s*x2; a[q][k] = s*x1 + c*x2; }
            for (int k=0;k<3;k++) { double x1 = v[k][p], x2 = v[k][q]; v[k][p] = c*x1 - s*x2; v[k][q] = s*x1 + c*x2; }
        }
    }
}
// 부호 있는 SVD: X = U diag(S) Vᵀ, U·V 회전 (det +1), S0 >= S1 >= |S2|, S2 부호 = det X 부호. 행 우선 [a*3+b].
__device__ void ssvd(const double* X, double* U, double* S, double* V) {
    double A[3][3], v[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) { double t = 0.0; for (int k=0;k<3;k++) t += X[k*3+i]*X[k*3+j]; A[i][j] = t; }
    jac3(A, v);
    int o[3] = {0,1,2};
    double e[3] = {A[0][0], A[1][1], A[2][2]};
    if (e[o[0]] < e[o[1]]) { int t = o[0]; o[0] = o[1]; o[1] = t; }
    if (e[o[1]] < e[o[2]]) { int t = o[1]; o[1] = o[2]; o[2] = t; }
    if (e[o[0]] < e[o[1]]) { int t = o[0]; o[0] = o[1]; o[1] = t; }
    for (int i=0;i<3;i++) for (int c=0;c<3;c++) V[i*3+c] = v[i][o[c]];
    double dv = V[0]*(V[4]*V[8]-V[5]*V[7]) - V[1]*(V[3]*V[8]-V[5]*V[6]) + V[2]*(V[3]*V[7]-V[4]*V[6]);
    if (dv < 0.0) for (int i=0;i<3;i++) V[i*3+2] = -V[i*3+2];
    double b[3][3];
    for (int c=0;c<3;c++) for (int a=0;a<3;a++) { double t = 0.0; for (int k=0;k<3;k++) t += X[a*3+k]*V[k*3+c]; b[c][a] = t; }
    double u1[3], u2[3], u3[3];
    double n1 = sqrt(b[0][0]*b[0][0] + b[0][1]*b[0][1] + b[0][2]*b[0][2]);
    if (n1 > 0.0) for (int a=0;a<3;a++) u1[a] = b[0][a]/n1; else { u1[0] = 1.0; u1[1] = 0.0; u1[2] = 0.0; }
    double d12 = u1[0]*b[1][0] + u1[1]*b[1][1] + u1[2]*b[1][2];
    for (int a=0;a<3;a++) u2[a] = b[1][a] - d12*u1[a];
    double n2 = sqrt(u2[0]*u2[0] + u2[1]*u2[1] + u2[2]*u2[2]);
    if (n2 > 1e-14*n1 && n2 > 0.0) { for (int a=0;a<3;a++) u2[a] /= n2; }
    else {                                                                      // 퇴화 (σ2 ≈ 0): u1 에 수직인 아무 단위 벡터
        int ax = 0; if (fabs(u1[1]) < fabs(u1[ax])) ax = 1; if (fabs(u1[2]) < fabs(u1[ax])) ax = 2;
        double ea[3] = {0.0, 0.0, 0.0}; ea[ax] = 1.0;
        u2[0] = u1[1]*ea[2]-u1[2]*ea[1]; u2[1] = u1[2]*ea[0]-u1[0]*ea[2]; u2[2] = u1[0]*ea[1]-u1[1]*ea[0];
        double nn = sqrt(u2[0]*u2[0] + u2[1]*u2[1] + u2[2]*u2[2]); for (int a=0;a<3;a++) u2[a] /= nn;
    }
    u3[0] = u1[1]*u2[2]-u1[2]*u2[1]; u3[1] = u1[2]*u2[0]-u1[0]*u2[2]; u3[2] = u1[0]*u2[1]-u1[1]*u2[0];
    for (int a=0;a<3;a++) { U[a*3] = u1[a]; U[a*3+1] = u2[a]; U[a*3+2] = u3[a]; }
    S[0] = u1[0]*b[0][0] + u1[1]*b[0][1] + u1[2]*b[0][2];
    S[1] = u2[0]*b[1][0] + u2[1]*b[1][1] + u2[2]*b[1][2];
    S[2] = u3[0]*b[2][0] + u3[1]*b[2][1] + u3[2]*b[2][2];
}
__device__ __forceinline__ double seg8(double s) {                              // 같은 칸 8 레인 합 (모든 레인이 불러야 함)
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    return s;
}
// mode: 1 기울기, 2 헤시안, 4 에너지
__global__ void an_elem(int nq, const int* qn, const double* G, const double* Fq, const double* Vq, const double* y,
                        double mu, double lam, int mode, double* E, double* f, double* H, float* H32) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    double X[9];
    for (int k=0;k<9;k++) X[k] = Fq[(size_t)qq*9+k];
    for (int n=0;n<27;n++) {
        int id = qn[(size_t)qq*27+n];
        const double* g = G + ((size_t)qq*27+n)*3;
        for (int a=0;a<3;a++) { double ya = y[id*3+a]; for (int b=0;b<3;b++) X[a*3+b] += ya*g[b]; }
    }
    double U[9], S[3], V[9];
    ssvd(X, U, S, V);
    double J = S[0]*S[1]*S[2];
    double gJ[3] = {S[1]*S[2], S[0]*S[2], S[0]*S[1]};
    double vq = on ? Vq[qq] : 0.0;
    if (mode & 4) { if (on) E[qq] = vq*(mu*((S[0]-1.0)*(S[0]-1.0) + (S[1]-1.0)*(S[1]-1.0) + (S[2]-1.0)*(S[2]-1.0)) + 0.5*lam*(J-1.0)*(J-1.0)); }
    if (mode & 1) {
        double ph[3]; for (int i=0;i<3;i++) ph[i] = 2.0*mu*(S[i]-1.0) + lam*(J-1.0)*gJ[i];
        double P[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) P[a*3+b] = U[a*3]*ph[0]*V[b*3] + U[a*3+1]*ph[1]*V[b*3+1] + U[a*3+2]*ph[2]*V[b*3+2];
        for (int n=0;n<27;n++) {
            const double* g = G + ((size_t)qq*27+n)*3;
            double s[3];
            for (int a=0;a<3;a++) s[a] = vq*(P[a*3]*g[0] + P[a*3+1]*g[1] + P[a*3+2]*g[2]);
            for (int a=0;a<3;a++) s[a] = seg8(s[a]);
            if (on && head) { int id = qn[(size_t)qq*27+n]; for (int a=0;a<3;a++) atomicAdd(&f[id*3+a], s[a]); }
        }
    }
    if ((mode & 2) && on) {
        double D[3][3], Ev[3][3];
        double Sg[3][3] = {{0.0, S[2], S[1]}, {S[2], 0.0, S[0]}, {S[1], S[0], 0.0}};
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = ((i==j) ? 2.0*mu : 0.0) + lam*gJ[i]*gJ[j] + lam*(J-1.0)*Sg[i][j];
        jac3(D, Ev);
        double h[81];
        for (int k=0;k<81;k++) h[k] = 0.0;
        for (int m=0;m<3;m++) {
            double dm = (mode & 16) ? D[m][m] : fmax(D[m][m], 0.0);                  // mode&16: 투영 없음 (진단)
            if (dm == 0.0) continue;
            double e[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) e[a*3+b] = Ev[0][m]*U[a*3]*V[b*3] + Ev[1][m]*U[a*3+1]*V[b*3+1] + Ev[2][m]*U[a*3+2]*V[b*3+2];
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += dm*e[r]*e[c];
        }
        const int PI[3] = {0, 0, 1}, PJ[3] = {1, 2, 2}, PK[3] = {2, 1, 0};
        const double r2 = 0.70710678118654752440;
        for (int pp=0; pp<3; pp++) {
            int i = PI[pp], j = PJ[pp], k = PK[pp];
            double lt = 2.0*mu*(1.0 - 2.0/(S[i]+S[j])) + lam*(J-1.0)*S[k];
            double lf = 2.0*mu - lam*(J-1.0)*S[k];
            if (mode & 16) { if (!isfinite(lt)) lt = 0.0; }
            else { lt = isfinite(lt) ? fmax(lt, 0.0) : 0.0; lf = fmax(lf, 0.0); }
            double et[9], ef[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
                double x1 = U[a*3+i]*V[b*3+j], x2 = U[a*3+j]*V[b*3+i];
                et[a*3+b] = r2*(x1 - x2); ef[a*3+b] = r2*(x1 + x2);
            }
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += lt*et[r]*et[c] + lf*ef[r]*ef[c];
        }
        if (mode & 8) { for (int k=0;k<81;k++) H32[(size_t)qq*81+k] = (float)h[k]; }
        else { for (int k=0;k<81;k++) H[(size_t)qq*81+k] = h[k]; }
    }
}
__global__ void an_mv(int nq, const int* qn, const double* G, const double* H, const double* Vq, const double* dy, double* out) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    double dX[9] = {0,0,0,0,0,0,0,0,0};
    for (int n=0;n<27;n++) {
        int id = qn[(size_t)qq*27+n];
        const double* g = G + ((size_t)qq*27+n)*3;
        for (int a=0;a<3;a++) { double ya = dy[id*3+a]; for (int b=0;b<3;b++) dX[a*3+b] += ya*g[b]; }
    }
    double vq = on ? Vq[qq] : 0.0;
    double dP[9];
    const double* h = H + (size_t)qq*81;
    for (int r=0;r<9;r++) { double t = 0.0; for (int c=0;c<9;c++) t += h[r*9+c]*dX[c]; dP[r] = vq*t; }
    for (int n=0;n<27;n++) {
        const double* g = G + ((size_t)qq*27+n)*3;
        double s[3];
        for (int a=0;a<3;a++) s[a] = dP[a*3]*g[0] + dP[a*3+1]*g[1] + dP[a*3+2]*g[2];
        for (int a=0;a<3;a++) s[a] = seg8(s[a]);
        if (on && head) { int id = qn[(size_t)qq*27+n]; for (int a=0;a<3;a++) atomicAdd(&out[id*3+a], s[a]); }
    }
}
__global__ void an_bd(int nq, const int* qn, const double* G, const double* H, const double* Vq, double* B) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    double vq = on ? Vq[qq] : 0.0;
    const double* h = H + (size_t)qq*81;
    for (int n=0;n<27;n++) {
        const double* g = G + ((size_t)qq*27+n)*3;
        double m[9];
        for (int a=0;a<3;a++) for (int c=0;c<3;c++) {
            double t = 0.0;
            for (int b=0;b<3;b++) for (int d=0;d<3;d++) t += h[(a*3+b)*9 + c*3+d]*g[b]*g[d];
            m[a*3+c] = vq*t;
        }
        for (int k=0;k<9;k++) m[k] = seg8(m[k]);
        if (on && head) { int id = qn[(size_t)qq*27+n]; for (int k=0;k<9;k++) atomicAdd(&B[id*9+k], m[k]); }
    }
}
// 입자 준비: R = U Vᵀ, LS = V diag(log S) Vᵀ, neg = det F <= 0
__global__ void an_rsp(int N, const double* F, double* R, double* LS, unsigned char* neg) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double U[9], S[3], V[9];
    ssvd(F + (size_t)p*9, U, S, V);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) R[(size_t)p*9+a*3+b] = U[a*3]*V[b*3] + U[a*3+1]*V[b*3+1] + U[a*3+2]*V[b*3+2];
    bool ng = !(S[2] > 0.0);
    neg[p] = ng ? 1 : 0;
    double l[3]; for (int i=0;i<3;i++) l[i] = ng ? 0.0 : log(S[i]);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) LS[(size_t)p*9+a*3+b] = V[a*3]*l[0]*V[b*3] + V[a*3+1]*l[1]*V[b*3+1] + V[a*3+2]*l[2]*V[b*3+2];
}
// 점 마무리: R̄ = polar(sR) (부호 있는 SVD 의 U Vᵀ = 가장 가까운 회전), S̄ = exp(sym(sL) / sw), RS = R̄ S̄; 비유한이면 nf
__global__ void an_rsf(int nq, const double* sR, const double* sL, const double* sw, double* RS, unsigned char* nf) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nq) return;
    double U[9], S[3], V[9];
    ssvd(sR + (size_t)q*9, U, S, V);
    double Rb[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Rb[a*3+b] = U[a*3]*V[b*3] + U[a*3+1]*V[b*3+1] + U[a*3+2]*V[b*3+2];
    double w = sw[q];
    double L[3][3], Ev[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) L[i][j] = 0.5*(sL[(size_t)q*9+i*3+j] + sL[(size_t)q*9+j*3+i])/w;
    jac3(L, Ev);
    double ex[3]; for (int i=0;i<3;i++) ex[i] = exp(L[i][i]);
    double Sb[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Sb[a*3+b] = Ev[a][0]*ex[0]*Ev[b][0] + Ev[a][1]*ex[1]*Ev[b][1] + Ev[a][2]*ex[2]*Ev[b][2];
    bool bad = false;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
        double t = Rb[a*3]*Sb[b] + Rb[a*3+1]*Sb[3+b] + Rb[a*3+2]*Sb[6+b];
        RS[(size_t)q*9+a*3+b] = t;
        if (!isfinite(t)) bad = true;
    }
    nf[q] = bad ? 1 : 0;
}

__device__ __forceinline__ float seg8f(float s) {
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    return s;
}
// float32 헤시안 곱 (H32·G32·V32, dy·out float32)
__global__ void an_mv32(int nq, const int* qn, const float* G, const float* H, const float* Vq, const float* dy, float* out) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    float dX[9] = {0,0,0,0,0,0,0,0,0};
    int ids[27];
    for (int n=0;n<27;n++) {
        int id = qn[(size_t)qq*27+n]; ids[n] = id;
        const float* g = G + ((size_t)qq*27+n)*3;
        float g0 = g[0], g1 = g[1], g2 = g[2];
        for (int a=0;a<3;a++) { float ya = dy[id*3+a]; dX[a*3] += ya*g0; dX[a*3+1] += ya*g1; dX[a*3+2] += ya*g2; }
    }
    float vq = on ? Vq[qq] : 0.f;
    float dP[9];
    const float* h = H + (size_t)qq*81;
    for (int r=0;r<9;r++) { float t = 0.f; for (int c=0;c<9;c++) t += h[r*9+c]*dX[c]; dP[r] = vq*t; }
    for (int n=0;n<27;n++) {
        const float* g = G + ((size_t)qq*27+n)*3;
        float g0 = g[0], g1 = g[1], g2 = g[2];
        float s[3];
        for (int a=0;a<3;a++) s[a] = seg8f(dP[a*3]*g0 + dP[a*3+1]*g1 + dP[a*3+2]*g2);
        if (on && head) { int id = ids[n]; for (int a=0;a<3;a++) atomicAdd(&out[id*3+a], s[a]); }
    }
}
__global__ void an_bd32(int nq, const int* qn, const float* G, const float* H, const float* Vq, float* B) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    float vq = on ? Vq[qq] : 0.f;
    const float* h = H + (size_t)qq*81;
    for (int n=0;n<27;n++) {
        const float* g = G + ((size_t)qq*27+n)*3;
        float m[9];
        for (int a=0;a<3;a++) for (int c=0;c<3;c++) {
            float t = 0.f;
            for (int b=0;b<3;b++) for (int d=0;d<3;d++) t += h[(a*3+b)*9 + c*3+d]*g[b]*g[d];
            m[a*3+c] = vq*t;
        }
        for (int k=0;k<9;k++) m[k] = seg8f(m[k]);
        if (on && head) { int id = qn[(size_t)qq*27+n]; for (int k=0;k<9;k++) atomicAdd(&B[id*9+k], m[k]); }
    }
}
// 노드 3x3 블록 곱 z = D r (float32), 블록 대각 평활·전처리용
__global__ void an_b3(int n, const float* D, const float* r, float* z, float w) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* d = D + (size_t)i*9;
    float r0 = r[i*3], r1 = r[i*3+1], r2 = r[i*3+2];
    z[i*3]   = w*(d[0]*r0 + d[1]*r1 + d[2]*r2);
    z[i*3+1] = w*(d[3]*r0 + d[4]*r1 + d[5]*r2);
    z[i*3+2] = w*(d[6]*r0 + d[7]*r1 + d[8]*r2);
}
// P2G / G2P 공통: 입자 격자 좌표 y = Ai x (torch 에서 구한 값을 그대로 받아 칸 판정이 torch 쪽과 같게), base = floor(y − ½) − bmin (상자 안), fx = y − floor(y − ½), 27 노드 (상자 번호 → 압축 번호 cmp)
__device__ __forceinline__ void bw64(double f, double* w, double* d) {
    w[0] = 0.5*(1.5-f)*(1.5-f); w[1] = 0.75-(f-1.0)*(f-1.0); w[2] = 0.5*(f-0.5)*(f-0.5);
    d[0] = f-1.5; d[1] = -2.0*(f-1.0); d[2] = f-0.5;
}
__global__ void an_p2g(int N, const double* Y, const double* v, const double* m, const double* Ai, const long long* bmin, const int* nb, const long long* cmp,
                       double* mn, double* mv) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double y[3]; int b[3]; double w[3][3], dw[3][3];
    for (int a=0;a<3;a++) { y[a] = Y[p*3+a]; double fb = floor(y[a] - 0.5); b[a] = (int)(fb - (double)bmin[a]); bw64(y[a] - fb, w[a], dw[a]); }
    double mp = m[p];
    for (int o=0;o<27;o++) {
        int i = o / 9, j = (o / 3) % 3, k = o % 3;
        long long nid = ((long long)(b[0]+i)*nb[1] + (b[1]+j))*nb[2] + (b[2]+k);
        long long c = cmp ? cmp[nid] : nid;
        double ww = w[0][i]*w[1][j]*w[2][k]*mp;
        atomicAdd(&mn[c], ww);
        if (mv) for (int a=0;a<3;a++) atomicAdd(&mv[c*3+a], ww*v[p*3+a]);
    }
}
__global__ void an_g2p(int N, const double* Y, const double* x, const double* F, const double* Ai, const long long* bmin, const int* nb, const long long* cmp,
                       const double* U, const double* vn, double* x2, double* v2, double* F2) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    double y[3]; int b[3]; double w[3][3], dw[3][3];
    for (int a=0;a<3;a++) { y[a] = Y[p*3+a]; double fb = floor(y[a] - 0.5); b[a] = (int)(fb - (double)bmin[a]); bw64(y[a] - fb, w[a], dw[a]); }
    double u[3] = {0,0,0}, vv[3] = {0,0,0};
    double A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int o=0;o<27;o++) {
        int i = o / 9, j = (o / 3) % 3, k = o % 3;
        long long nid = ((long long)(b[0]+i)*nb[1] + (b[1]+j))*nb[2] + (b[2]+k);
        long long c = cmp[nid];
        double ww = w[0][i]*w[1][j]*w[2][k];
        double gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        double gx[3]; for (int bb=0;bb<3;bb++) gx[bb] = Ai[0*3+bb]*gy[0] + Ai[1*3+bb]*gy[1] + Ai[2*3+bb]*gy[2];
        for (int a=0;a<3;a++) { double uu = U[c*3+a]; u[a] += ww*uu; vv[a] += ww*vn[c*3+a]; for (int bb=0;bb<3;bb++) A[a*3+bb] += uu*gx[bb]; }
    }
    for (int a=0;a<3;a++) { x2[p*3+a] = x[p*3+a] + u[a]; v2[p*3+a] = vv[a]; }
    for (int a=0;a<3;a++) for (int bb=0;bb<3;bb++) { double t = 0.0; for (int kk=0;kk<3;kk++) t += A[a*3+kk]*F[(size_t)p*9+kk*3+bb]; F2[(size_t)p*9+a*3+bb] = t; }
}
}

//@@ _C 19644

#define CFULL 0xffffffffu
extern "C" {
// ---- (3) 점유 사면체 목록 (프레임마다, tlc 는 미리 0) ----
__global__ void c_tet_list(const int* szp, const float* VT, int* tl, int* tlc) {
    int T = szp[1];
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool o = (t < T) && (VT[t] > 0.f);
    unsigned b = __ballot_sync(CFULL, o);
    int base = 0;
    if (lane == 0 && b) base = atomicAdd(tlc, __popc(b));
    base = __shfl_sync(CFULL, base, 0);
    if (o) tl[base + __popc(b & ((1u << lane) - 1u))] = t;
}
// ---- (1) 바닥 후보 시작 높이 s0 (프레임마다) ----
__global__ void c_s0(const int* szp, const long long* nidx, const float* x, float p0, float p1, float p2, float n0, float n1, float n2, float* s0c) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= szp[2]) return;
    long long p = nidx[q];
    s0c[q] = (x[p*3]-p0)*n0 + (x[p*3+1]-p1)*n1 + (x[p*3+2]-p2)*n2;
}
// (1) 걸러 낸 접촉 판정: 통과한 후보는 ex_vfl_contact 와 같은 식
__global__ void c_vfl_contact(const int* szp, const float* s0c, const unsigned int* dn, const long long* nidx, const float* x, const float* Ucum,
                              const long long* rows, const float* lamb, const float* m, int tetmass, float p0, float p1, float p2, float n0, float n1, float n2,
                              float* acc, int* ncon) {
    int nn_ = szp[2];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nn_) return;
    if (s0c[q] > __uint_as_float(*dn) + 1e-5f) return;
    long long p = nidx[q];
    float u[3] = {0,0,0};
    long long nd[4]; float lm[4];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm[i]*Ucum[nd[i]*3+a]; }
    float s = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (!(s < 0.f)) return;
    float mp = m[p];
    for (int i=0;i<4;i++) atomicAdd(&acc[nd[i]], (tetmass ? 0.25f : lm[i])*mp);
    if (ncon) atomicAdd(ncon, 1);
}
// ---- 상자 판 적용 (ex_lim_box_apply 그대로) + 목록 개수 비움 + (1) 의 D 갱신 (dn != null) ----
__global__ void c_lim_apply(const int* szp, const float* mn, float* vn, float* Ucum, float hs, unsigned int* alph, int vmode, const unsigned char* fixd, float* rst,
                            int* lst, unsigned int* dn, float n0, float n1, float n2) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (lst && n == 0) lst[0] = 0;
    float dl = 0.f;
    if (n < M) {
        float al = __uint_as_float(alph[n]);
        alph[n] = 0x3f800000u;
        if (fixd && fixd[n]) al = 1.f;
        float v[3] = {vn[n*3], vn[n*3+1], vn[n*3+2]};
        float U[3];
        for (int c=0;c<3;c++) { Ucum[n*3+c] += al*hs*v[c]; U[c] = Ucum[n*3+c]; }
        if (al < 1.f) {
            float m = mn[n], w = 1.f - al;
            atomicAdd(&rst[1], 1.f);
            if (vmode == 0) {
                for (int c=0;c<3;c++) { vn[n*3+c] = al*v[c]; atomicAdd(&rst[2+c], w*m*v[c]); }
                float v2 = v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
                atomicAdd(&rst[5], 0.5f*m*v2*(1.f - al*al));
                atomicAdd(&rst[6], w*m*sqrtf(v2));
            }
        }
        dl = fmaxf(-(U[0]*n0 + U[1]*n1 + U[2]*n2), 0.f);
    }
    if (dn) {
        unsigned b = __float_as_uint(dl);
        for (int o=16; o>0; o>>=1) b = max(b, __shfl_xor_sync(CFULL, b, o));
        if ((threadIdx.x & 31) == 0 && b) atomicMax(dn, b);
    }
}
// ---- (2) 클램프 판정 ----
__device__ __forceinline__ void c_sr_load(int t, const long long* tets, const float* gl, const float* vn, const float* Ucum, float hs,
                                          long long* n, float* A0, float* Mo, float& rho) {
    float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    for (int k=0;k<9;k++) A0[k] = (k % 4 == 0) ? 1.f : 0.f;
    rho = 0.f;
    for (int i=0;i<4;i++) {
        float dd = 0.f, gg = 0.f;
        for (int c=0;c<3;c++) { float u=Ucum[n[i]*3+c], d=hs*vn[n[i]*3+c]; dd += d*d; for (int b=0;b<3;b++) { A0[c*3+b] += u*G[i*3+b]; Mo[i*9+c*3+b] = d*G[i*3+b]; } }
        for (int b=0;b<3;b++) gg += G[i*3+b]*G[i*3+b];
        rho += sqrtf(dd*gg);
    }
}
// 빠른 보장 판정 (사면체당 스레드): 보장되면 아무것도 안 함, 아니면 lst 에 넣는다. tl != null 이면 점유 목록 (i < tlc[0]) 위로.
__global__ void c_sr_pre(const int* szp, const int* tl, const int* tlc, const long long* tets, const float* gl, const float* VT, const float* vn, const float* Ucum,
                         float hs, float csr, int jon, float jmin, int* lst, const float* eF) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int t;
    if (tl) { if (i >= tlc[0]) return; t = tl[i]; }
    else { if (i >= szp[1] || !(VT[i] > 0.f)) return; t = i; }
    if (eF) {   // (6) ‖A0 - I‖_F² 는 같은 서브스텝 힘 커널이 같은 U 로 남긴 값 -- U 를 다시 읽지 않는다 (판정 보수성은 같다)
        float rho = 0.f;
        for (int a=0;a<4;a++) {
            long long na = tets[t*4+a];
            float dd = 0.f, gg = 0.f;
            for (int c=0;c<3;c++) { float d=hs*vn[na*3+c]; dd += d*d; float g_=gl[t*12+a*3+c]; gg += g_*g_; }
            rho += sqrtf(dd*gg);
        }
        float del = sqrtf(eF[t]) + rho;
        bool fast = del < 0.999f*(csr - 1.f)/(csr + 1.f);
        if (fast && jon) { float s3 = 1.f - del; fast = s3*s3*s3 >= jmin*1.001f; }
        if (fast) return;
        int slot = atomicAdd(&lst[0], 1);
        lst[1 + slot] = t;
        return;
    }
    long long n[4]; float G[12];
    for (int k=0;k<4;k++) n[k]=tets[t*4+k];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A0[9] = {1,0,0, 0,1,0, 0,0,1};
    float rho = 0.f;
    for (int a=0;a<4;a++) {
        float dd = 0.f, gg = 0.f;
        for (int c=0;c<3;c++) { float u=Ucum[n[a]*3+c], d=hs*vn[n[a]*3+c]; dd += d*d; for (int b=0;b<3;b++) A0[c*3+b] += u*G[a*3+b]; }
        for (int b=0;b<3;b++) gg += G[a*3+b]*G[a*3+b];
        rho += sqrtf(dd*gg);
    }
    float e2 = 0.f;
    for (int k=0;k<9;k++) { float d = A0[k] - ((k % 4 == 0) ? 1.f : 0.f); e2 += d*d; }
    float del = sqrtf(e2) + rho;
    bool fast = del < 0.999f*(csr - 1.f)/(csr + 1.f);
    if (fast && jon) { float s3 = 1.f - del; fast = s3*s3*s3 >= jmin*1.001f; }
    if (fast) return;
    int slot = atomicAdd(&lst[0], 1);
    lst[1 + slot] = t;
}
__device__ __forceinline__ bool c_feasw(int lane, const float* A0, const float* Mo, float s, float thrR, int jon, float thrJ) {
    bool o = true;
    if (lane < 15) {
        int S = lane + 1;
        float B[9]; for (int k=0;k<9;k++) { float t = 0.f; for (int i=0;i<4;i++) if (S & (1<<i)) t += Mo[i*9+k]; B[k] = A0[k] + s*t; }
        if (jon && det3(B) < thrJ) o = false;
        else if (sr_ratio(B) > thrR) o = false;
    }
    return __all_sync(CFULL, o);
}
// 목록 사면체당 워프 하나 -- ex_lim_sr_tet 과 같은 순서·식 (꼭짓점 15 개를 레인에), 격자 크기 고정 + 워프 단위 반복
__global__ void c_sr_warp(const int* lst, const long long* tets, const float* gl, const float* vn, const float* Ucum, float hs,
                          float csr, int jon, float jmin, unsigned int* alph, float* rst) {
    int cnt = lst[0];
    int lane = threadIdx.x & 31;
    int gw = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    int nw = (gridDim.x*blockDim.x) >> 5;
    for (int q = gw; q < cnt; q += nw) {
        int t = lst[1 + q];
        long long n[4]; float A0[9], Mo[36], rho;
        c_sr_load(t, tets, gl, vn, Ucum, hs, n, A0, Mo, rho);
        float D0 = det3(A0);
        float r0 = sr_ratio(A0);
        float thrR = fmaxf(csr, r0);
        float thrJ = fminf(jmin, D0);
        bool ok = true;
        if (jon) { bool o = (lane >= 15) || !(det_sub(A0, Mo, lane + 1, 1.f) < thrJ); ok = __all_sync(CFULL, o); }
        if (ok && D0 > 0.f) {
            float C[9];
            for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=A0[i]*A0[k]+A0[3+i]*A0[3+k]+A0[6+i]*A0[6+k];
            float l[3]; symeig3(C, l);
            float s1 = sqrtf(fmaxf(l[2], 0.f)), s3 = sqrtf(fmaxf(l[0], 0.f));
            if (s3 - rho > 0.f && (s1 + rho) <= thrR*(s3 - rho)) continue;
        }
        if (c_feasw(lane, A0, Mo, 1.f, thrR, jon, thrJ)) continue;
        float lo = 0.f, hi = 1.f;
        for (int it=0; it<14; it++) {
            float mid = 0.5f*(lo+hi);
            if (c_feasw(lane, A0, Mo, mid, thrR, jon, thrJ)) lo = mid; else hi = mid;
        }
        if (lane == 0) {
            unsigned int bits = __float_as_uint(lo);
            for (int i=0;i<4;i++) atomicMin(&alph[n[i]], bits);
            atomicAdd(&rst[7], 1.f);
        }
    }
}
// ---- 힘: ex_tet_force_R 과 같은 식 (+ 점유 목록, (4) 빠른 극분해 선택) ----
__device__ __forceinline__ void c_polar_fast(const float* F, float* Q) {
    for (int k=0;k<9;k++) Q[k]=F[k];
    float C[9];
    for (int it=0; it<10; it++) {
        float dt = det3(Q);
        float ds = (fabsf(dt) > 1e-30f) ? dt : 1e-30f;
        float gm = rcbrtf(fabsf(ds));
        cof3(Q, C);
        float mx = 0.f;
        for (int k=0;k<9;k++) { float qn = 0.5f*(gm*Q[k] + C[k]/(ds*gm)); mx = fmaxf(mx, fabsf(qn - Q[k])); Q[k] = qn; }
        if (mx < 1e-7f) break;
    }
}
__device__ __forceinline__ void c_force_body(int t, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                                             float mu, float lam, float* f, float fbk, float fbj, int fbon, float* rst, int fpol, float* eF) {
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    if (eF) { float e2 = 0.f; for (int k=0;k<9;k++) { float d = A[k] - ((k % 4 == 0) ? 1.f : 0.f); e2 += d*d; } eF[t] = e2; }   // (6) 클램프 빠른 판정용 ‖A - I‖_F²
    float cA[9]; cof3(A, cA);
    float dA = A[0]*cA[0]+A[1]*cA[1]+A[2]*cA[2];
    if (rst && !(dA > 0.f)) atomicAdd(&rst[0], 1.f);
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    float X[9]; mm3(A, Fb, X);
    float C[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
    float l[3]; symeig3(C, l);
    float J = 1.f;
    for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
    float Q[9], cf[9], P[9], Gm[9];
    if (fpol) c_polar_fast(X, Q); else polar(X, Q);
    cof3(X, cf);
    float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
    float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
    for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
    if (fbk > 0.f && fbon == 0) { float db = bar_db(dt, fbj); if (db != 0.f) { float s = fbk*mu*db; for (int k=0;k<9;k++) P[k] += s*cf[k]; } }
    mmt3(P, Fb, Gm);
    if (fbk > 0.f && fbon == 1) { float db = bar_db(dA, fbj); if (db != 0.f) { float s = fbk*mu*db; for (int k=0;k<9;k++) Gm[k] += s*cA[k]; } }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float s = 0.f; for (int b=0;b<3;b++) s += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*s);
    }
}
__global__ void c_tet_force_R(const int* szp, const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                              float mu, float lam, float* f, float fbk, float fbj, int fbon, float* rst, int fpol, float* eF) {
    int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    int t;
    if (tl) { if (i0 >= tlc[0]) return; t = tl[i0]; }
    else { if (i0 >= szp[1]) return; t = i0; }
    c_force_body(t, tets, U, gl, FT, VT, mu, lam, f, fbk, fbj, fbon, rst, fpol, eF);
}

// ======== (5) 융합 판 (--cfuse): 서브스텝 커널 6 개 -> 4 개, 같은 식 ========
//   K1 c_fc      : [사면체 블록] c_tet_force_R 와 같은 힘 + [바닥 후보 블록] c_vfl_contact 와 같은 접촉 판정 (둘 다 노드 배열에 원자 합만 -- 서로 독립)
//   K2 c_sr_pre2 : c_sr_pre 와 같은 판정, 노드 속도는 ex_node_step_lag 와 같은 식으로 그 자리에서 (c_vnew)
//   K3 c_sr_warp2: c_sr_warp 와 같은 판정, 속도는 c_vnew
//   K4 c_apply2  : ex_node_step_lag (속도 갱신·바닥 고착·비관통, f·gn·acc 비움) + c_lim_apply (α, U, 기록, 목록 비움, D) 를 한 번에
// ex_node_step_lag 와 같은 식 (nodisp = 1 판) -- 쓰지 않고 값만
__device__ __forceinline__ void c_vnew(long long n, const float* mn, const float* f, const float* gn, const float* vn, float g0, float g1, float g2, float hs, float damp,
                                       const unsigned char* fixd, const float* vbc, const float* acc, int mode, int np_, float n0, float n1, float n2, float* v) {
    float gg[3] = {g0, g1, g2};
    float m = mn[n];
    for (int c=0;c<3;c++) {
        if (fixd && fixd[n]) v[c] = vbc[n*3+c];
        else if (m > 0.f) v[c] = (vn[n*3+c] + hs*((f[n*3+c] - (gn ? gn[n*3+c] : 0.f))/m + gg[c]))*damp;
        else v[c] = 0.f;
    }
    if (acc) {
        float a = acc[n];
        if (a > 0.f && !(fixd && fixd[n])) {
            float k = (mode == 1) ? 0.f : ((mode == 2) ? 1.f : fmaxf(1.f - (m > 0.f ? a/m : 1.f), 0.f));
            for (int c=0;c<3;c++) v[c] *= k;
            if (np_) { float vn_ = v[0]*n0 + v[1]*n1 + v[2]*n2; if (vn_ < 0.f) { v[0] -= vn_*n0; v[1] -= vn_*n1; v[2] -= vn_*n2; } }
        }
    }
}
struct CVS { const float* mn; const float* f; const float* gn; const float* vn; float g0, g1, g2, hs, damp; const unsigned char* fixd; const float* vbc; const float* acc; int mode, np_; float n0, n1, n2; };
__device__ __forceinline__ void c_vget(const CVS& S, long long n, float* v) {
    c_vnew(n, S.mn, S.f, S.gn, S.vn, S.g0, S.g1, S.g2, S.hs, S.damp, S.fixd, S.vbc, S.acc, S.mode, S.np_, S.n0, S.n1, S.n2, v);
}
__global__ void c_fc(int nbT, const int* szp, const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                     float mu, float lam, float* f, float* rst, int fpol, float* eF,
                     const float* s0c, const unsigned int* dn, const long long* nidx, const float* x, const long long* rows, const float* lamb, const float* m, int tetmass,
                     float p0, float p1, float p2, float n0, float n1, float n2, float* acc, int* ncon) {
    if ((int)blockIdx.x < nbT) {
        int i0 = blockIdx.x*blockDim.x + threadIdx.x;
        int t;
        if (tl) { if (i0 >= tlc[0]) return; t = tl[i0]; }
        else { if (i0 >= szp[1]) return; t = i0; }
        c_force_body(t, tets, U, gl, FT, VT, mu, lam, f, 0.f, 0.f, 0, rst, fpol, eF);
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
__global__ void c_sr_pre2(const int* szp, const int* tl, const int* tlc, const long long* tets, const float* gl, const float* VT, const float* Ucum, CVS S,
                          float csr, int jon, float jmin, int* lst) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int t;
    if (tl) { if (i >= tlc[0]) return; t = tl[i]; }
    else { if (i >= szp[1] || !(VT[i] > 0.f)) return; t = i; }
    long long n[4]; float G[12];
    for (int k=0;k<4;k++) n[k]=tets[t*4+k];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A0[9] = {1,0,0, 0,1,0, 0,0,1};
    float rho = 0.f;
    for (int a=0;a<4;a++) {
        float vv[3]; c_vget(S, n[a], vv);
        float dd = 0.f, gg = 0.f;
        for (int c=0;c<3;c++) { float u=Ucum[n[a]*3+c], d=S.hs*vv[c]; dd += d*d; for (int b=0;b<3;b++) A0[c*3+b] += u*G[a*3+b]; }
        for (int b=0;b<3;b++) gg += G[a*3+b]*G[a*3+b];
        rho += sqrtf(dd*gg);
    }
    float e2 = 0.f;
    for (int k=0;k<9;k++) { float d = A0[k] - ((k % 4 == 0) ? 1.f : 0.f); e2 += d*d; }
    float del = sqrtf(e2) + rho;
    bool fast = del < 0.999f*(csr - 1.f)/(csr + 1.f);
    if (fast && jon) { float s3 = 1.f - del; fast = s3*s3*s3 >= jmin*1.001f; }
    if (fast) return;
    int slot = atomicAdd(&lst[0], 1);
    lst[1 + slot] = t;
}
__global__ void c_sr_warp2(const int* lst, const long long* tets, const float* gl, const float* Ucum, CVS S,
                           float csr, int jon, float jmin, unsigned int* alph, float* rst) {
    int cnt = lst[0];
    int lane = threadIdx.x & 31;
    int gw = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    int nw = (gridDim.x*blockDim.x) >> 5;
    float hs = S.hs;
    for (int q = gw; q < cnt; q += nw) {
        int t = lst[1 + q];
        long long n[4]; float A0[9], Mo[36], rho, G[12];
        for (int i=0;i<4;i++) n[i]=tets[t*4+i];
        for (int k=0;k<12;k++) G[k]=gl[t*12+k];
        for (int k=0;k<9;k++) A0[k] = (k % 4 == 0) ? 1.f : 0.f;
        rho = 0.f;
        for (int i=0;i<4;i++) {
            float vv[3]; c_vget(S, n[i], vv);
            float dd = 0.f, gg = 0.f;
            for (int c=0;c<3;c++) { float u=Ucum[n[i]*3+c], d=hs*vv[c]; dd += d*d; for (int b=0;b<3;b++) { A0[c*3+b] += u*G[i*3+b]; Mo[i*9+c*3+b] = d*G[i*3+b]; } }
            for (int b=0;b<3;b++) gg += G[i*3+b]*G[i*3+b];
            rho += sqrtf(dd*gg);
        }
        float D0 = det3(A0);
        float r0 = sr_ratio(A0);
        float thrR = fmaxf(csr, r0);
        float thrJ = fminf(jmin, D0);
        bool ok = true;
        if (jon) { bool o = (lane >= 15) || !(det_sub(A0, Mo, lane + 1, 1.f) < thrJ); ok = __all_sync(CFULL, o); }
        if (ok && D0 > 0.f) {
            float C[9];
            for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=A0[i]*A0[k]+A0[3+i]*A0[3+k]+A0[6+i]*A0[6+k];
            float l[3]; symeig3(C, l);
            float s1 = sqrtf(fmaxf(l[2], 0.f)), s3 = sqrtf(fmaxf(l[0], 0.f));
            if (s3 - rho > 0.f && (s1 + rho) <= thrR*(s3 - rho)) continue;
        }
        if (c_feasw(lane, A0, Mo, 1.f, thrR, jon, thrJ)) continue;
        float lo = 0.f, hi = 1.f;
        for (int it=0; it<14; it++) {
            float mid = 0.5f*(lo+hi);
            if (c_feasw(lane, A0, Mo, mid, thrR, jon, thrJ)) lo = mid; else hi = mid;
        }
        if (lane == 0) {
            unsigned int bits = __float_as_uint(lo);
            for (int i=0;i<4;i++) atomicMin(&alph[n[i]], bits);
            atomicAdd(&rst[7], 1.f);
        }
    }
}
__global__ void c_apply2(const int* szp, CVS S, float* vn, float* f, float* gn, float* acc, float* Ucum, unsigned int* alph, int vmode, float* rst,
                         int* lst, unsigned int* dn) {
    int M = szp[0];
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (lst && n == 0) lst[0] = 0;
    float dl = 0.f;
    if (n < M) {
        float v[3]; c_vget(S, n, v);
        for (int c=0;c<3;c++) { f[n*3+c] = 0.f; if (gn) gn[n*3+c] = 0.f; }
        if (acc) acc[n] = 0.f;
        float al = __uint_as_float(alph[n]);
        alph[n] = 0x3f800000u;
        if (S.fixd && S.fixd[n]) al = 1.f;
        float U[3];
        for (int c=0;c<3;c++) { vn[n*3+c] = v[c]; Ucum[n*3+c] += al*S.hs*v[c]; U[c] = Ucum[n*3+c]; }
        if (al < 1.f) {
            float m = S.mn[n], w = 1.f - al;
            atomicAdd(&rst[1], 1.f);
            if (vmode == 0) {
                for (int c=0;c<3;c++) { vn[n*3+c] = al*v[c]; atomicAdd(&rst[2+c], w*m*v[c]); }
                float v2 = v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
                atomicAdd(&rst[5], 0.5f*m*v2*(1.f - al*al));
                atomicAdd(&rst[6], w*m*sqrtf(v2));
            }
        }
        dl = fmaxf(-(U[0]*S.n0 + U[1]*S.n1 + U[2]*S.n2), 0.f);
    }
    if (dn) {
        unsigned b = __float_as_uint(dl);
        for (int o=16; o>0; o>>=1) b = max(b, __shfl_xor_sync(CFULL, b, o));
        if ((threadIdx.x & 31) == 0 && b) atomicMax(dn, b);
    }
}
}  // extern "C" (절 C)

//@@ _PL 22815

extern "C" {
#ifndef AF_VN_DEF
#define AF_VN_DEF
__device__ const float* AFVN;                                                       // 절 AG.18 --vfl_pred: 노드 속도 (null = 끔) -- 비관통 판정을 서브스텝 끝 예측 입자 위치 s + h_s (Σλ v)·n 로
__device__ float AFHS;
#endif
// ---------------- 3x3 SVD (C = XᵀX 순환 Jacobi) ----------------
__device__ __forceinline__ void pl_svd(const float* X, float* U, float* s, float* V) {
    float A[9];
    for (int i=0;i<3;i++) for (int k=0;k<3;k++) A[i*3+k] = X[i]*X[k] + X[3+i]*X[3+k] + X[6+i]*X[6+k];
    for (int k=0;k<9;k++) V[k] = (k%4==0) ? 1.f : 0.f;
    for (int sw=0; sw<10; sw++) {
        float off = A[1]*A[1] + A[2]*A[2] + A[5]*A[5];
        float dg = A[0]*A[0] + A[4]*A[4] + A[8]*A[8];
        if (off <= 1e-14f*dg || off < 1e-36f) break;
        for (int r=0;r<3;r++) {
            int p = (r==2) ? 1 : 0, q = (r==0) ? 1 : 2;
            float apq = A[p*3+q];
            if (fabsf(apq) < 1e-30f) continue;
            float th = (A[q*3+q] - A[p*3+p]) / (2.f*apq);
            float t = (th >= 0.f ? 1.f : -1.f) / (fabsf(th) + sqrtf(th*th + 1.f));
            float c = rsqrtf(t*t + 1.f), sn = t*c;
            for (int k=0;k<3;k++) { float akp=A[k*3+p], akq=A[k*3+q]; A[k*3+p]=c*akp - sn*akq; A[k*3+q]=sn*akp + c*akq; }
            for (int k=0;k<3;k++) { float apk=A[p*3+k], aqk=A[q*3+k]; A[p*3+k]=c*apk - sn*aqk; A[q*3+k]=sn*apk + c*aqk; }
            for (int k=0;k<3;k++) { float vkp=V[k*3+p], vkq=V[k*3+q]; V[k*3+p]=c*vkp - sn*vkq; V[k*3+q]=sn*vkp + c*vkq; }
        }
    }
    // 내림차순 정렬 (열 교환)
    float l[3] = {A[0], A[4], A[8]};
    for (int i=0;i<2;i++) for (int j=0;j<2-i;j++) if (l[j] < l[j+1]) {
        float tl=l[j]; l[j]=l[j+1]; l[j+1]=tl;
        for (int k=0;k<3;k++) { float tv=V[k*3+j]; V[k*3+j]=V[k*3+j+1]; V[k*3+j+1]=tv; }
    }
    // V 를 회전으로 (det +1)
    float dv = V[0]*(V[4]*V[8]-V[5]*V[7]) - V[1]*(V[3]*V[8]-V[5]*V[6]) + V[2]*(V[3]*V[7]-V[4]*V[6]);
    if (dv < 0.f) { V[2] = -V[2]; V[5] = -V[5]; V[8] = -V[8]; }
    float xv[3][3];
    for (int j=0;j<3;j++) for (int r=0;r<3;r++) xv[j][r] = X[r*3]*V[j] + X[r*3+1]*V[3+j] + X[r*3+2]*V[6+j];
    float s0 = sqrtf(fmaxf(l[0], 0.f));
    float u0[3], u1[3], u2[3];
    float n0 = sqrtf(xv[0][0]*xv[0][0] + xv[0][1]*xv[0][1] + xv[0][2]*xv[0][2]);
    if (n0 > 1e-20f) { for (int r=0;r<3;r++) u0[r] = xv[0][r]/n0; } else { u0[0]=1.f; u0[1]=0.f; u0[2]=0.f; }
    float d01 = u0[0]*xv[1][0] + u0[1]*xv[1][1] + u0[2]*xv[1][2];
    for (int r=0;r<3;r++) u1[r] = xv[1][r] - d01*u0[r];
    float n1 = sqrtf(u1[0]*u1[0] + u1[1]*u1[1] + u1[2]*u1[2]);
    if (n1 > 1e-20f) { for (int r=0;r<3;r++) u1[r] /= n1; }
    else {   // u0 에 수직인 아무 단위 벡터
        float e[3] = {0.f, 0.f, 0.f}; int k = (fabsf(u0[0]) < 0.6f) ? 0 : ((fabsf(u0[1]) < 0.6f) ? 1 : 2); e[k] = 1.f;
        float d = u0[k]; for (int r=0;r<3;r++) u1[r] = e[r] - d*u0[r];
        float nn = sqrtf(u1[0]*u1[0] + u1[1]*u1[1] + u1[2]*u1[2]); for (int r=0;r<3;r++) u1[r] /= nn;
    }
    u2[0] = u0[1]*u1[2] - u0[2]*u1[1]; u2[1] = u0[2]*u1[0] - u0[0]*u1[2]; u2[2] = u0[0]*u1[1] - u0[1]*u1[0];
    s[0] = n0 > 1e-20f ? n0 : s0;
    s[1] = u1[0]*xv[1][0] + u1[1]*xv[1][1] + u1[2]*xv[1][2];
    s[2] = u2[0]*xv[2][0] + u2[1]*xv[2][1] + u2[2]*xv[2][2];
    for (int r=0;r<3;r++) { U[r*3] = u0[r]; U[r*3+1] = u1[r]; U[r*3+2] = u2[r]; }
}
__device__ __forceinline__ void pl_usv(const float* U, const float* d, const float* V, float* Y) {   // Y = U diag(d) Vᵀ
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) Y[r*3+c] = U[r*3]*d[0]*V[c*3] + U[r*3+1]*d[1]*V[c*3+1] + U[r*3+2]*d[2]*V[c*3+2];
}
// ---------------- return mapping (주값; s 를 덮어쓴다; 바뀌었으면 true) ----------------
//  upd = 0 이면 ys·lj·al 은 지역 사본 (셀 투영), 1 이면 입자 상태 갱신
__device__ __forceinline__ bool pl_rm(int mat, float* s, const float* prm, float& ys, float& lj, float& al, float dt) {
    float mu = prm[0], lam = prm[1];
    if (mat == 1 || mat == 5) {                                     // PG von_mises_return_mapping(_with_damage)
        if (mat == 5 && !(al > 0.f)) return false;                  // 손상 입자: μ = λ = 0 -> cond 0 > ys (≤ 0) -> F_trial 그대로
        float sg[3], ep[3];
        for (int i=0;i<3;i++) { sg[i] = fmaxf(s[i], 0.01f); ep[i] = logf(sg[i]); }
        float tr = ep[0] + ep[1] + ep[2], tmp = tr/3.f;
        float tau[3]; for (int i=0;i<3;i++) tau[i] = 2.f*mu*ep[i] + lam*tr;
        float st = tau[0] + tau[1] + tau[2];
        float c0 = tau[0] - st/3.f, c1 = tau[1] - st/3.f, c2 = tau[2] - st/3.f;
        if (sqrtf(c0*c0 + c1*c1 + c2*c2) > ys) {
            if (mat == 5 && ys <= 0.f) return false;
            float eh[3]; for (int i=0;i<3;i++) eh[i] = ep[i] - tmp;
            float ehn = sqrtf(eh[0]*eh[0] + eh[1]*eh[1] + eh[2]*eh[2]) + 1e-6f;
            float dg = ehn - ys/(2.f*mu);
            for (int i=0;i<3;i++) ep[i] = ep[i] - (dg/ehn)*eh[i];
            float muh = mu;
            if (mat == 5) {
                float r = dg/ehn; float de = fabsf(r)*sqrtf(eh[0]*eh[0] + eh[1]*eh[1] + eh[2]*eh[2]);
                ys = ys - prm[8]*de;
                if (ys <= 0.f) { al = 0.f; muh = 0.f; }
            }
            for (int i=0;i<3;i++) s[i] = expf(ep[i]);
            if (prm[7] == 1.f) ys = ys + 2.f*muh*prm[6]*dg;
            return true;
        }
        return false;
    }
    if (mat == 3) {                                                 // PG viscoplasticity_return_mapping_with_StVK
        float sg[3], ep[3], b[3];
        for (int i=0;i<3;i++) { sg[i] = fmaxf(s[i], 0.01f); b[i] = sg[i]*sg[i]; ep[i] = logf(sg[i]); }
        float tr = ep[0] + ep[1] + ep[2];
        float st[3]; for (int i=0;i<3;i++) st[i] = 2.f*mu*(ep[i] - tr/3.f);
        float stn = sqrtf(st[0]*st[0] + st[1]*st[1] + st[2]*st[2]);
        float y = stn - sqrtf(2.f/3.f)*ys;
        if (y > 0.f) {
            float muh = mu*(b[0] + b[1] + b[2])/3.f;
            float snn = stn - y/(1.f + prm[9]/(2.f*muh*dt));
            for (int i=0;i<3;i++) s[i] = expf((snn/stn)*st[i]/(2.f*mu) + tr/3.f);
            return true;
        }
        return false;
    }
    if (mat == 2) {                                                 // PG sand_return_mapping
        float ep[3]; for (int i=0;i<3;i++) ep[i] = logf(fmaxf(fabsf(s[i]), 1e-14f));
        float tr = ep[0] + ep[1] + ep[2];
        float eh[3]; for (int i=0;i<3;i++) eh[i] = ep[i] - tr/3.f;
        float ehn = sqrtf(eh[0]*eh[0] + eh[1]*eh[1] + eh[2]*eh[2]);
        float dg = ehn + (3.f*lam + 2.f*mu)/(2.f*mu)*tr*prm[3];
        if (dg <= 0.f) return false;
        if (tr > 0.f) { s[0] = s[1] = s[2] = 1.f; return true; }
        for (int i=0;i<3;i++) s[i] = expf(ep[i] - eh[i]*(dg/ehn));
        return true;
    }
    if (mat == 7) {                                                 // GF NonAssociativeCamClay_return_mapping
        float kap = prm[2], M = prm[4], beta = prm[5], xi = prm[6];
        float sg[3]; for (int i=0;i<3;i++) sg[i] = fmaxf(s[i], 0.f);
        float p0 = kap*(0.00001f + sinhf(xi*fmaxf(-lj, 0.f)));
        float J = sg[0]*sg[1]*sg[2];
        float B[3]; for (int i=0;i<3;i++) B[i] = sg[i]*sg[i];
        float Bm = (B[0] + B[1] + B[2])/3.f;
        float Jp = powf(J, -2.f/3.f);
        float sh[3]; for (int i=0;i<3;i++) sh[i] = mu*Jp*(B[i] - Bm);
        float prime = kap/2.f*(J - 1.f/J);
        float ptr = -prime*J;
        float ysh = (6.f - 3.f)/2.f*(1.f + 2.f*beta);
        float yph = M*M*(ptr + beta*p0)*(ptr - p0);
        float ssq = sh[0]*sh[0] + sh[1]*sh[1] + sh[2]*sh[2];
        float y = ysh*ssq + yph;
        float sf[3] = {sg[0], sg[1], sg[2]};
        float ljn = lj, pmin = beta*p0;
        int hard = prm[7] > 0.5f;
        if (ptr > p0) {
            float Je = sqrtf(-2.f*p0/kap + 1.f);
            sf[0] = sf[1] = sf[2] = powf(Je, 1.f/3.f);
            if (hard) ljn = lj + logf(J/Je);
        } else if (ptr < -pmin) {
            float Je = sqrtf(2.f*pmin/kap + 1.f);
            sf[0] = sf[1] = sf[2] = powf(Je, 1.f/3.f);
            if (hard) ljn = lj + logf(J/Je);
        } else if (y >= 1e-4f) {
            float shn = fmaxf(sqrtf(ssq), 1e-10f);
            float sqf = sqrtf(-yph/ysh);
            float J2 = powf(J, 2.f/3.f);
            float sc = J2/mu*sqf/shn;
            for (int i=0;i<3;i++) sf[i] = sqrtf(sc*sh[i] + Bm);
            if (hard && p0 > 1e-4f && ptr < p0 - 1e-4f && ptr > 1e-4f - pmin) {
                float pc = (p0 - pmin)*0.5f;
                float qt = sqrtf((6.f - 3.f)/2.f)*shn;
                float dp = pc - ptr, dq = 0.f - qt;
                float dn = fmaxf(sqrtf(dp*dp + dq*dq), 1e-10f);
                dp = dp/dn;                                         // GF 원본: direction_q 는 나누지 않는다 (그대로)
                float C = M*M*(pc + beta*p0)*(pc - p0);
                float Bq = M*M*dp*(2.f*pc - p0 + beta*p0);
                float Aq = M*M*dp*dp + (1.f + 2.f*beta)*dq*dq;
                float disc = Bq*Bq - 4.f*Aq*C;
                float l1 = (-Bq + sqrtf(disc))/(2.f*Aq);
                float l2 = (-Bq - sqrtf(disc))/(2.f*Aq);
                float p1 = pc + l1*dp, p2 = pc + l2*dp;
                float pf = ((ptr - pc)*(p1 - pc) > 0.f) ? p1 : p2;
                float Jef = sqrtf(fabsf(-2.f*pf/kap + 1.f));
                if (Jef > 1e-4f && hard) ljn = lj + logf(J/Jef);
            }
        }
        lj = ljn;
        s[0] = sf[0]; s[1] = sf[1]; s[2] = sf[2];
        return true;                                                // GF 는 늘 U diag(σ_final) Vᵀ 로 다시 짠다
    }
    return false;
}
// ---------------- 응력 주값 P_i (P = U diag(P_i) Vᵀ, 키르히호프 τ = P Fᵀ) ----------------
__device__ __forceinline__ void pl_pi(int mat, const float* s, const float* prm, float al, float* Pi) {
    float mu = prm[0], lam = prm[1];
    if (mat == 1 || mat == 3) {                                     // PG kirchoff_stress_StVK: τ = U diag(τ_i) Vᵀ Fᵀ -> P_i = τ_i
        float ep[3]; for (int i=0;i<3;i++) ep[i] = logf(fmaxf(s[i], 0.01f));
        float tr = ep[0] + ep[1] + ep[2];
        for (int i=0;i<3;i++) Pi[i] = 2.f*mu*ep[i] + lam*tr;
    } else if (mat == 2) {                                          // PG kirchoff_stress_drucker_prager: P_i = (2μ ln σ_i + λ Σ ln σ)/σ_i
        float sg[3], ep[3]; for (int i=0;i<3;i++) { sg[i] = fmaxf(fabsf(s[i]), 1e-14f); ep[i] = logf(sg[i]); }
        float tr = ep[0] + ep[1] + ep[2];
        for (int i=0;i<3;i++) Pi[i] = (2.f*mu*ep[i] + lam*tr)/sg[i];
    } else if (mat == 7) {                                          // GF kirchoff_stress_neoHookeanBoarden: τ = μ J^{-2/3} dev(B) + κ/2 (J² - 1) I
#ifdef AF_SNH
        // 절 CM13 판 (2) (사용자 승인 정의 변경): 탄성 부분을 Stable Neo-Hookean (Smith 2018 의 로그 항 없는 꼴, Kim·Eberle 2022) 으로 --
        // Ψ = μ/2 (I_C − 3) − μ (J − 1) + λ_s/2 (J − 1)², P = μ F + (λ_s (J − 1) − μ) cof F -> 주값 P_i = μ σ_i + (λ_s (J − 1) − μ) σ_j σ_k.
        // 다항식이라 J ≤ 0 에서도 유한·매끄러움 (나눗셈·거듭제곱 없음, 클램프 없음). 같은 E·ν 대응: 선형화 P ≈ μ (H + Hᵀ) + (λ_s − μ) tr H I -> μ_s = μ, λ_s = λ + μ (prm[0] μ, prm[1] λ).
        { float lss = prm[1] + mu, J = s[0]*s[1]*s[2], cc = lss*(J - 1.f) - mu;
          Pi[0] = mu*s[0] + cc*s[1]*s[2]; Pi[1] = mu*s[1] + cc*s[0]*s[2]; Pi[2] = mu*s[2] + cc*s[0]*s[1];
#ifdef AF_DMG
          // 절 CM19 (사용자 승인 정의 변경): 손상 열화 ψ = g ψ⁺ + ψ⁻ (al = g = (1 − d)², 부르는 쪽 aq_elem 이 넘김). 분해 = 체적 압축 몫만 유지:
          // ψ⁻ = κ/2 (J − 1)² (J < 1 일 때만, κ = λ + 2μ/3), ψ⁺ = Ψ_SNH − ψ⁻ -> P_i = g P_i^SNH + (1 − g) [J < 1] κ (J − 1) σ_j σ_k.
          { float gd = al, cm = (J < 1.f) ? (1.f - gd)*prm[2]*(J - 1.f) : 0.f;
            Pi[0] = gd*Pi[0] + cm*s[1]*s[2]; Pi[1] = gd*Pi[1] + cm*s[0]*s[2]; Pi[2] = gd*Pi[2] + cm*s[0]*s[1]; }
#endif
        }
        return;
#endif
        float kap = prm[2];
        float J = s[0]*s[1]*s[2];
        float Bm = (s[0]*s[0] + s[1]*s[1] + s[2]*s[2])/3.f;
        float Jp = powf(J, -2.f/3.f);
        for (int i=0;i<3;i++) { float sd = (fabsf(s[i]) > 1e-6f) ? s[i] : ((s[i] < 0.f) ? -1e-6f : 1e-6f);
            Pi[i] = (mu*Jp*(s[i]*s[i] - Bm) + 0.5f*kap*(J*J - 1.f))/sd; }
    } else {                                                        // PG kirchoff_stress_FCR (jelly, plasticine): P = 2μ(F - R) + λ J (J - 1) F⁻ᵀ
        float J = s[0]*s[1]*s[2];
        float co[3] = {s[1]*s[2], s[0]*s[2], s[0]*s[1]};
        for (int i=0;i<3;i++) Pi[i] = al*(2.f*mu*(s[i] - 1.f) + lam*(J - 1.f)*co[i]);
    }
}
// GF kirchoff_stress_neoHookeanBoarden 를 F_e 에서 바로 (GF 와 같은 식: B = F Fᵀ, J = det F -- SVD 거치지 않아 float 오차가 정답 수준)
__device__ __forceinline__ void pl_nhb_tau(const float* Fe, float mu, float kap, float* tau) {
    float B[9]; mmt3(Fe, Fe, B);
    float J = det3(Fe);
    float tb = (B[0] + B[4] + B[8])/3.f;
    float sd = mu*powf(J, -2.f/3.f), sv = 0.5f*kap*(J*J - 1.f);
    for (int k=0;k<9;k++) tau[k] = sd*(B[k] - ((k%4==0) ? tb : 0.f)) + ((k%4==0) ? sv : 0.f);
}
// ---------------- 사면체 힘 (c_force_body 틀 + 재질) ----------------
__device__ __forceinline__ void ag_force_body(int t, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                                              float mu, float lam, float* f, float* rst, int fpol,
                                              int mat, const float* prm, const float* ysT, const float* ljT, float dtc, const float* vn, float visc, int cflag,
                                              float* FeC, float* XpC, float* ysC, float* ljC) {
    // cflag (진단 전용): 1 = 셀 투영 끔 (재질 응력만, return mapping 은 프레임 끝 입자에만), 2 = 응력을 FCR 로 (투영은 그대로)
    if (mat == 0 && !(visc > 0.f)) { c_force_body(t, tets, U, gl, FT, VT, mu, lam, f, 0.f, 0.f, 0, rst, fpol, (float*)0); return; }
    float V = VT[t];
    if (!(V > 0.f)) return;
    long long n[4]; float G[12];
    for (int i=0;i<4;i++) n[i]=tets[t*4+i];
    for (int k=0;k<12;k++) G[k]=gl[t*12+k];
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float u=U[n[i]*3+c]; for (int b=0;b<3;b++) A[c*3+b] += u*G[i*3+b]; }
    float cA[9]; cof3(A, cA);
    float dA = A[0]*cA[0]+A[1]*cA[1]+A[2]*cA[2];
    if (rst && !(dA > 0.f)) atomicAdd(&rst[0], 1.f);
    float Fb[9]; for (int k=0;k<9;k++) Fb[k]=FT[t*9+k];
    float X[9]; mm3(A, Fb, X);
    float Gm[9];
    if (mat == 0) {
        float C[9];
        for (int i=0;i<3;i++) for (int k=0;k<3;k++) C[i*3+k]=X[i]*X[k]+X[3+i]*X[3+k]+X[6+i]*X[6+k];
        float l[3]; symeig3(C, l);
        float J = 1.f;
        for (int k=0;k<3;k++){ float s = sqrtf(fmaxf(l[k], 1e-12f)); s = fmaxf(s, 0.01f); J *= s; }
        float Q[9], cf[9], P[9];
        if (fpol) c_polar_fast(X, Q); else polar(X, Q);
        cof3(X, cf);
        float dt = X[0]*cf[0]+X[1]*cf[1]+X[2]*cf[2];
        float sg = (dt>0)?1.f:((dt<0)?-1.f:0.f);
        for (int k=0;k<9;k++) P[k] = 2.f*mu*(X[k]-Q[k]) + lam*(J-1.f)*sg*cf[k];
        mmt3(P, Fb, Gm);
    } else {
        float Uu[9], s[3], Vv[9];
        float ys = ysT ? ysT[t] : 0.f, lj = ljT ? ljT[t] : 0.f, al = 1.f;
        if (FeC) {                                                  // 셀 증분 return mapping (정답처럼 서브스텝마다): F_trial = X X_prev⁻¹ F_e,prev, 셀 상태 갱신
            float Xp[9], Fp[9]; for (int k=0;k<9;k++) { Xp[k] = XpC[t*9+k]; Fp[k] = FeC[t*9+k]; }
            float cX[9]; cof3(Xp, cX); float dX = det3(Xp); float id = (fabsf(dX) > 1e-20f) ? 1.f/dX : 0.f;
            float Xi[9]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) Xi[i*3+j] = cX[j*3+i]*id;
            float D[9]; mm3(X, Xi, D);
            float Ft[9]; mm3(D, Fp, Ft);
            ys = ysC[t]; lj = ljC[t];
            pl_svd(Ft, Uu, s, Vv);
            bool ch = pl_rm(mat, s, prm, ys, lj, al, dtc);
            float Fe[9]; if (ch) pl_usv(Uu, s, Vv, Fe); else for (int k=0;k<9;k++) Fe[k] = Ft[k];
            for (int k=0;k<9;k++) { FeC[t*9+k] = Fe[k]; XpC[t*9+k] = X[k]; }
            ysC[t] = ys; ljC[t] = lj;
            // 응력은 F_e 에서, 힘은 Gm = P(F_e) F_e⁻ᵀ ... 대신 τ X⁻ᵀ F̄ᵀ: P_X = τ X⁻ᵀ (현재 배치 응력은 τ/J 로 같다)
            float tau[9];
            if (mat == 7 && !(cflag & 2)) pl_nhb_tau(Fe, prm[0], prm[2], tau);
            else { float U2[9], s2[3], V2[9]; pl_svd(Fe, U2, s2, V2); float Pi[3]; pl_pi((cflag & 2) ? 5 : mat, s2, prm, al, Pi);
                   float P_[9]; pl_usv(U2, Pi, V2, P_); mmt3(P_, Fe, tau); }
            float cXX[9]; cof3(X, cXX); float dXX = det3(X); float idd = (fabsf(dXX) > 1e-20f) ? 1.f/dXX : 0.f;
            float XiT[9]; for (int k=0;k<9;k++) XiT[k] = cXX[k]*idd;                       // X⁻ᵀ = cof(X)/det
            float PX[9]; mm3(tau, XiT, PX);
            mmt3(PX, Fb, Gm);
            goto visc_part;
        }
        pl_svd(X, Uu, s, Vv);
        if (!(cflag & 1)) pl_rm(mat, s, prm, ys, lj, al, dtc);    // 셀 투영 (상태는 지역 사본 -- 입자 상태는 프레임 끝 ag_rm 만 바꾼다)
        if (mat == 7 && !(cflag & 2)) {                             // τ 를 F_e 에서 바로: Gm = τ F_e⁻ᵀ F̄ᵀ = τ (F̄ F_e⁻¹)ᵀ
            float Fe[9]; pl_usv(Uu, s, Vv, Fe);
            float tau[9]; pl_nhb_tau(Fe, prm[0], prm[2], tau);
            float cF[9]; cof3(Fe, cF);
            float dF = det3(Fe); float id = (fabsf(dF) > 1e-20f) ? 1.f/dF : 0.f;
            float Fi[9]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) Fi[i*3+j] = cF[j*3+i]*id;     // F_e⁻¹
            float M_[9]; mm3(Fb, Fi, M_);                            // F̄ F_e⁻¹
            mmt3(tau, M_, Gm);
        } else {
            float Pi[3]; pl_pi((cflag & 2) ? 5 : mat, s, prm, al, Pi);
            float P[9]; pl_usv(Uu, Pi, Vv, P);
            mmt3(P, Fb, Gm);
        }
    }
visc_part:
    if (visc > 0.f && vn) {                                         // 점성 (정답 솔버에는 없음): τ_v = 2 η J sym(L), L = (Σ v_n ⊗ ∇λ_n) A⁻¹, 힘 -V τ_v A⁻ᵀ ∇λ
        float Gv[9] = {0,0,0,0,0,0,0,0,0};
        for (int i=0;i<4;i++) for (int c=0;c<3;c++) { float vv = vn[n[i]*3+c]; for (int b=0;b<3;b++) Gv[c*3+b] += vv*G[i*3+b]; }
        float id = (fabsf(dA) > 1e-12f) ? 1.f/dA : 0.f;
        float Ai[9]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) Ai[i*3+j] = cA[j*3+i]*id;     // A⁻¹ = cofᵀ / det
        float L[9]; mm3(Gv, Ai, L);
        float J = det3(X);
        float tv[9]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) tv[i*3+j] = visc*J*(L[i*3+j] + L[j*3+i]);
        float Gw[9]; mmt3(tv, Ai, Gw);
        for (int k=0;k<9;k++) Gm[k] += Gw[k];
    }
    for (int i=0;i<4;i++) for (int c=0;c<3;c++) {
        float sacc = 0.f; for (int b=0;b<3;b++) sacc += Gm[c*3+b]*G[i*3+b];
        atomicAdd(&f[n[i]*3+c], -V*sacc);
    }
}
// c_fc 와 같은 틀 (사면체 블록 nbT 개 + 바닥 후보 블록) -- 사면체 블록만 ag_force_body
__global__ void ag_fc(int nbT, const int* szp, const int* tl, const int* tlc, const long long* tets, const float* U, const float* gl, const float* FT, const float* VT,
                      float mu, float lam, float* f, float* rst, int fpol,
                      int mat, const float* prm, const float* ysT, const float* ljT, float dtc, const float* vn, float visc, int cflag,
                      float* FeC, float* XpC, float* ysC, float* ljC,
                      const float* s0c, const unsigned int* dn, const long long* nidx, const float* x, const long long* rows, const float* lamb, const float* m, int tetmass,
                      float p0, float p1, float p2, float n0, float n1, float n2, float* acc, int* ncon) {
    if ((int)blockIdx.x < nbT) {
        int i0 = blockIdx.x*blockDim.x + threadIdx.x;
        int t;
        if (tl) { if (i0 >= tlc[0]) return; t = tl[i0]; }
        else { if (i0 >= szp[1]) return; t = i0; }
        ag_force_body(t, tets, U, gl, FT, VT, mu, lam, f, rst, fpol, mat, prm, ysT, ljT, dtc, vn, visc, cflag, FeC, XpC, ysC, ljC);
        return;
    }
    int q = (blockIdx.x - nbT)*blockDim.x + threadIdx.x;
    if (q >= szp[2]) return;
    if (!AFVN && s0c[q] > __uint_as_float(*dn) + 1e-5f) return;
    long long p = nidx[q];
    float u[3] = {0,0,0};
    long long nd[4]; float lm[4];
    for (int i=0;i<4;i++){ nd[i]=rows[p*4+i]; lm[i]=lamb[p*4+i]; for (int a=0;a<3;a++) u[a] += lm[i]*U[nd[i]*3+a]; }
    float sd = (x[p*3]+u[0]-p0)*n0 + (x[p*3+1]+u[1]-p1)*n1 + (x[p*3+2]+u[2]-p2)*n2;
    if (AFVN) { float vp = 0.f; for (int i=0;i<4;i++) vp += lm[i]*(AFVN[nd[i]*3]*n0 + AFVN[nd[i]*3+1]*n1 + AFVN[nd[i]*3+2]*n2); sd += AFHS*vp; }   // 절 AG.18 --vfl_pred
    if (!(sd < 0.f)) return;
    float mp = m[p];
    for (int i=0;i<4;i++) atomicAdd(&acc[nd[i]], (tetmass ? 0.25f : lm[i])*mp);
    if (ncon) atomicAdd(ncon, 1);
}
// 셀 증분 상태 초기화 (서브스텝 그래프 첫 커널): F_e = X_prev = F̄, 상태 = 입자 평균
__global__ void ag_cinit(const int* szp, const float* FT, const float* ysT, const float* ljT, float* FeC, float* XpC, float* ysC, float* ljC) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= szp[1]) return;
    for (int k=0;k<9;k++) { FeC[t*9+k] = FT[t*9+k]; XpC[t*9+k] = FT[t*9+k]; }
    ysC[t] = ysT ? ysT[t] : 0.f; ljC[t] = ljT ? ljT[t] : 0.f;
}
// 프레임 끝 입자 return mapping (상태 갱신) -- rmc[0] += 투영된 입자 수, rmc[1] += 새로 손상된 입자 수
__global__ void ag_rm(int N, float* F, float* ys, float* lj, float* al, int mat, const float* prm, float dt, float* rmc) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    float X[9]; for (int k=0;k<9;k++) X[k] = F[p*9+k];
    float Uu[9], s[3], Vv[9]; pl_svd(X, Uu, s, Vv);
    float y_ = ys ? ys[p] : 0.f, l_ = lj ? lj[p] : 0.f, a_ = al ? al[p] : 1.f, a0 = a_;
    bool ch = pl_rm(mat, s, prm, y_, l_, a_, dt);
    if (ys) ys[p] = y_;
    if (lj) lj[p] = l_;
    if (al) al[p] = a_;
    if (ch) {
        float Y[9]; pl_usv(Uu, s, Vv, Y);
        for (int k=0;k<9;k++) F[p*9+k] = Y[k];
        if (rmc && mat != 7) atomicAdd(&rmc[0], 1.f);
    }
    if (rmc && a0 > 0.f && !(a_ > 0.f)) atomicAdd(&rmc[1], 1.f);
}
// 입자 속도 구속의 노드 몫 섞기 (노드 걸음 뒤): v <- (1 - α) v + pv,  pv = α v̄_bc (프레임마다 P2G 질량 가중으로)
__global__ void ag_pbc(const int* szp, float* vn, const float* pa, const float* pv) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= szp[0]) return;
    float a = pa[n];
    if (!(a > 0.f)) return;
    for (int c=0;c<3;c++) vn[n*3+c] = (1.f - a)*vn[n*3+c] + pv[n*3+c];
}
// 단위 대조: return mapping (갱신) + 키르히호프 응력 τ = P F_eᵀ (대칭화)
__global__ void ag_test(int N, const float* Fin, float* ys, float* lj, float* al, int mat, const float* prm, float dt, float* Fout, float* tau) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    float X[9]; for (int k=0;k<9;k++) X[k] = Fin[p*9+k];
    float Uu[9], s[3], Vv[9]; pl_svd(X, Uu, s, Vv);
    float y_ = ys[p], l_ = lj[p], a_ = al[p];
    bool ch = pl_rm(mat, s, prm, y_, l_, a_, dt);
    ys[p] = y_; lj[p] = l_; al[p] = a_;
    float Fe[9];
    if (ch) pl_usv(Uu, s, Vv, Fe); else for (int k=0;k<9;k++) Fe[k] = X[k];
    for (int k=0;k<9;k++) Fout[p*9+k] = Fe[k];
    float T[9];
    if (mat == 7) pl_nhb_tau(Fe, prm[0], prm[2], T);
    else {
        float U2[9], s2[3], V2[9]; pl_svd(Fe, U2, s2, V2);
        float Pi[3]; pl_pi(mat, s2, prm, a_, Pi);
        float P[9]; pl_usv(U2, Pi, V2, P);
        mmt3(P, Fe, T);
    }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) tau[p*9+i*3+j] = 0.5f*(T[i*3+j] + T[j*3+i]);
}
}

//@@ _SAND 6388

__device__ __forceinline__ int ao_dp(const double* S, double cdp, double* sgn, double* sg, double* ep, double* e, double& tr, double& rho, double& s) {
    for (int i=0;i<3;i++) { sgn[i] = (S[i] < 0.0) ? -1.0 : 1.0; sg[i] = fmax(fabs(S[i]), 1e-14); ep[i] = log(sg[i]); }
    tr = ep[0] + ep[1] + ep[2];
    for (int i=0;i<3;i++) e[i] = ep[i] - tr/3.0;
    rho = sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    double dg = rho + cdp*tr;
    s = 1.0;
    if (!(dg > 0.0)) return 0;
    if (tr > 0.0) { s = 0.0; return 2; }
    s = -cdp*tr/rho;
    return 1;
}
__global__ void ao_elem(int nq, const int* qn, const double* G, const double* Fq, const double* Vq, const double* y,
                        double mu, double lam, double cdp, int mode, double* f, double* H, float* H32, unsigned char* pst) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    double X[9];
    for (int k=0;k<9;k++) X[k] = Fq[(size_t)qq*9+k];
    for (int n=0;n<27;n++) {
        int id = qn[(size_t)qq*27+n];
        const double* g = G + ((size_t)qq*27+n)*3;
        for (int a=0;a<3;a++) { double ya = y[id*3+a]; for (int b=0;b<3;b++) X[a*3+b] += ya*g[b]; }
    }
    double U[9], S[3], V[9];
    ssvd(X, U, S, V);
    double sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dp(S, cdp, sgn, sg, ep, e, tr, rho, s);
    if (pst && on) pst[qq] = (unsigned char)st;
    double K3 = lam + 2.0*mu/3.0;
    double tau[3], ss[3], p[3], pe[3];
    for (int i=0;i<3;i++) {
        tau[i] = (st == 2) ? 0.0 : ((st == 1) ? (K3*tr + 2.0*mu*s*e[i]) : (2.0*mu*ep[i] + lam*tr));
        ss[i] = sgn[i]*sg[i];
        p[i] = tau[i]/ss[i];
        pe[i] = (2.0*mu*ep[i] + lam*tr)/ss[i];                                  // 시험 상태 탄성 (Hencky) 응력 주값 -- mode 64 접선용
    }
    const bool elt = (mode & 64) != 0;                                          // mode 64: 탄성 접선 (초기 강성법) = 시험 X 의 Hencky 헤시안 (투영), 응력은 RM 그대로
    double vq = on ? Vq[qq] : 0.0;
    if (mode & 1) {
        double P[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) P[a*3+b] = U[a*3]*p[0]*V[b*3] + U[a*3+1]*p[1]*V[b*3+1] + U[a*3+2]*p[2]*V[b*3+2];
        for (int n=0;n<27;n++) {
            const double* g = G + ((size_t)qq*27+n)*3;
            double sv[3];
            for (int a=0;a<3;a++) sv[a] = vq*(P[a*3]*g[0] + P[a*3+1]*g[1] + P[a*3+2]*g[2]);
            for (int a=0;a<3;a++) sv[a] = seg8(sv[a]);
            if (on && head) { int id = qn[(size_t)qq*27+n]; for (int a=0;a<3;a++) atomicAdd(&f[id*3+a], sv[a]); }
        }
    }
    if ((mode & 2) && on) {
        double C[3][3], D[3][3], Ev[3][3];
        double eh[3] = {0.0, 0.0, 0.0};
        if (st == 1 && rho > 0.0) for (int i=0;i<3;i++) eh[i] = e[i]/rho;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
            double c;
            if (st == 0 || elt) c = ((i==j) ? 2.0*mu : 0.0) + lam;
            else if (st == 1) c = K3 + 2.0*mu*s*(((i==j) ? 1.0 : 0.0) - 1.0/3.0 - eh[i]*eh[j]) - ((mode & 32) ? 2.0*mu*cdp*eh[i] : 0.0);   // mode 32: 일관 접선 (비대칭 결합 항 −2μc ê_i 포함)
            else c = 0.0;
            C[i][j] = c;
        }
        const double* pp_ = elt ? pe : p;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = C[i][j]/(ss[i]*ss[j]) - ((i==j) ? pp_[i]/ss[i] : 0.0);
        double Dk[3][3]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) Dk[i][j] = D[i][j];
        double h[81];
        for (int k=0;k<81;k++) h[k] = 0.0;
        if (mode & 32) {                                                        // 일관 접선: D 그대로 (투영·대칭 없음) h = Σ_mn D_mn e_m ⊗ e_n, e_m = U_m V_mᵀ
            double em[3][9];
            for (int m=0;m<3;m++) for (int a=0;a<3;a++) for (int b=0;b<3;b++) em[m][a*3+b] = U[a*3+m]*V[b*3+m];
            for (int m=0;m<3;m++) for (int n=0;n<3;n++) { double dmn = Dk[m][n]; if (!isfinite(dmn)) continue;
                for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += dmn*em[m][r]*em[n][c]; }
        } else {
        jac3(D, Ev);
        for (int m=0;m<3;m++) {
            double dm = fmax(D[m][m], 0.0);
            if (!(dm > 0.0) || !isfinite(dm)) continue;
            double ee[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) ee[a*3+b] = Ev[0][m]*U[a*3]*V[b*3] + Ev[1][m]*U[a*3+1]*V[b*3+1] + Ev[2][m]*U[a*3+2]*V[b*3+2];
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += dm*ee[r]*ee[c];
        }
        }
        const int PI[3] = {0, 0, 1}, PJ[3] = {1, 2, 2};
        const double r2 = 0.70710678118654752440;
        for (int pp=0; pp<3; pp++) {
            int i = PI[pp], j = PJ[pp];
            double den = ss[i] + ss[j];
            double lt = (fabs(den) > 1e-12) ? (pp_[i] + pp_[j])/den : 0.0;
            double dd = ss[i] - ss[j];
            double lf = (fabs(dd) > 1e-9*(fabs(ss[i]) + fabs(ss[j]))) ? (pp_[i] - pp_[j])/dd : (Dk[i][i] - Dk[i][j]);
            if (mode & 32) { lt = isfinite(lt) ? lt : 0.0; lf = isfinite(lf) ? lf : 0.0; }
            else { lt = isfinite(lt) ? fmax(lt, 0.0) : 0.0; lf = isfinite(lf) ? fmax(lf, 0.0) : 0.0; }
            double et[9], ef[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
                double x1 = U[a*3+i]*V[b*3+j], x2 = U[a*3+j]*V[b*3+i];
                et[a*3+b] = r2*(x1 - x2); ef[a*3+b] = r2*(x1 + x2);
            }
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += lt*et[r]*et[c] + lf*ef[r]*ef[c];
        }
        if (mode & 8) { for (int k=0;k<81;k++) H32[(size_t)qq*81+k] = (float)h[k]; }
        else { for (int k=0;k<81;k++) H[(size_t)qq*81+k] = h[k]; }
    }
}
// 정답 return mapping (3x3 묶음): 탄성이면 그대로, 꼭짓점이면 U diag(부호) Vᵀ, 꼬리 투영이면 U diag(부호 exp ε̂) Vᵀ
__global__ void ao_rm(int N, const double* Fin, double cdp, double* Fout, unsigned char* pst) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    const double* X = Fin + (size_t)p*9;
    double U[9], S[3], V[9];
    ssvd(X, U, S, V);
    double sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dp(S, cdp, sgn, sg, ep, e, tr, rho, s);
    if (pst) pst[p] = (unsigned char)st;
    if (st == 0) { for (int k=0;k<9;k++) Fout[(size_t)p*9+k] = X[k]; return; }
    double sn[3];
    for (int i=0;i<3;i++) sn[i] = (st == 2) ? sgn[i] : sgn[i]*exp(tr/3.0 + s*e[i]);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Fout[(size_t)p*9+a*3+b] = U[a*3]*sn[0]*V[b*3] + U[a*3+1]*sn[1]*V[b*3+1] + U[a*3+2]*sn[2]*V[b*3+2];
}

//@@ UL_SRC 1721

// 절 CM31 --ue_loc / --uf_w (fusedgs/fem_common1.py)
__global__ void ul_zero(const int* IP, int* m) { int n = blockIdx.x*blockDim.x + threadIdx.x; if (n < IP[9]) m[n] = 0; }
__global__ void ul_kn(const int* IP, const int* kn, int* m) { int n = blockIdx.x*blockDim.x + threadIdx.x; if (n < IP[9] && kn[n] > 0) m[n] = 1; }
__global__ void ul_res(const int* IP, const float* r0, const float* me, float gg, int* m) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    double s = 0.0; for (int a=0;a<3;a++) s += (double)r0[n*3+a]*r0[n*3+a];
    double w = (double)me[n]*gg;
    if (s > w*w) m[n] = 1;
}
// 진단: [0] S 노드 수, [1] 몸 노드 수 (질량 > 0), [2] S ∩ 몸
__global__ void ul_st(const int* IP, const int* m, const float* mn, int* st) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (m[n]) atomicAdd(&st[0], 1);                                             // 정수 원자 -> 결정적
    if (mn[n] > 0.f) { atomicAdd(&st[1], 1); if (m[n]) atomicAdd(&st[2], 1); }
}
// Σ_n (m_n |g|)² -- 블록 하나 (256 스레드) 가 고정 순서로 (결정적)
__global__ void ul_wsum(const int* IP, const float* me, float gg, double* w, int k) {
    __shared__ double sh[256];
    double s = 0.0;
    for (int n = threadIdx.x; n < IP[9]; n += 256) { double t = (double)me[n]*gg; s += t*t; }
    sh[threadIdx.x] = s; __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) w[k] = sh[0];
}
__global__ void ul_gw(const double* SC, const double* CD, const double* w, float* cjg, int k) {
    double F2 = CD[2] + w[k]; double r = (F2 > 0.0) ? SC[1]/F2 : 0.0; cjg[k] = (r == r) ? (float)fmin(r, 3.0e38) : 3.0e38f; }

//@@ UL_BAR_SRC 742

__global__ void __launch_bounds__(128) ul_bar(const int* IP, const int* cb, const float* BW, const float* y, const float* U, float thinv, float n0, float n1, float n2, float pn, float dh, int* m) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3];
    if (!(bm > 0.f) || !AF_BACT[q]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xa[3];
    for (int a=0;a<3;a++) xa[a] = AF_BX[q*4+a]/bm;
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o]; for (int a=0;a<3;a++) xa[a] += w*(U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a])); }
    if (xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn < dh) { for (int o=0;o<27;o++) if (BW[oc*27+o] != 0.f) m[ids[o]] = 1; }
}

//@@ C2_SRC 1649

// ---- 절 CM35 (fusedgs/fem_common2.py) ----
__global__ void ul_dil(const int* IP, const int* src, int* dst) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (src[n]) { dst[n] = 1; return; }
    int n0 = IP[6], n1 = IP[7], n2 = IP[8];
    int i = n/(n1*n2), j = (n/n2) % n1, k = n % n2;
    int r = 0;
    for (int a=-1;a<=1 && !r;a++) for (int b=-1;b<=1 && !r;b++) for (int c=-1;c<=1 && !r;c++) {
        int ii = i+a, jj = j+b, kk = k+c;
        if (ii < 0 || jj < 0 || kk < 0 || ii >= n0 || jj >= n1 || kk >= n2) continue;
        if (src[(ii*n1 + jj)*n2 + kk]) r = 1;
    }
    dst[n] = r;
}
__global__ void ul_cpy(const int* IP, const int* src, int* dst) { int n = blockIdx.x*blockDim.x + threadIdx.x; if (n < IP[9]) dst[n] = src[n]; }
// 비트 16: 덜 풀린 걸음 끝 HHT 갱신이 노드 속도를 뒤집는 노드 (v1_HHT = al (y − z0), v1_HHT·v0 < 0) -- 두 실행기 걸음 끝 커널과 같은 v1 식, 갱신 전 vk = v0
__global__ void ul_flip(const int* IP, float al, const float* y, const float* z0, const float* vk, int* m) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float d = 0.f, s = 0.f;
    for (int a=0;a<3;a++) { float v0 = vk[n*3+a]; d += al*(y[n*3+a] - z0[n*3+a])*v0; s += v0*v0; }
    if (s > 0.f && d < 0.f) m[n] = 1;
}
// 비트 32: 이 걸음 마지막으로 돈 뉴턴 반복 시작 ‖g‖²/F² (판정 2 값) 가 T 넘게 크면 (풀이가 무너진 걸음) 몸 전체
__global__ void ul_big(const int* IP, const float* g2, int k, float T, int* m) { int n = blockIdx.x*blockDim.x + threadIdx.x; if (n < IP[9] && !(g2[k] <= T)) m[n] = 1; }
__global__ void ul_cut(const int* IP, const int* cut, int* m) { int n = blockIdx.x*blockDim.x + threadIdx.x; if (n < IP[9] && cut[n] > 0) m[n] = 1; }

//@@ C2_GATE_SRC 220

__global__ void c2_gate(int* CT, const float* cjs, const float* cjg, int k, int res) {
    if (CT[0]) return;
    bool cut = cjs[k] < 1.f, big = res && cjg[k] > 1.f;
    if (!(cut || big)) { CT[0] = 1; AF_SKIP = 1; }
}

//@@ FPI_SRC 2953

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

//@@ MG2_SRC 20170

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

//@@ an_rt_SRC 90288

#define NQ8 8
extern "C" {
__device__ __forceinline__ float seg8f(float s) {
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2); s += __shfl_xor_sync(0xffffffffu, s, 4); return s;
}
__device__ __forceinline__ double seg8d(double s) {
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2); s += __shfl_xor_sync(0xffffffffu, s, 4); return s;
}
__device__ __forceinline__ double warpsum(double s) {
    for (int o=16;o>0;o>>=1) s += __shfl_xor_sync(0xffffffffu, s, o); return s;
}
// 블록 합 (f64) -> 원자 합 하나
__device__ void blocksum_add(double v, double* out) {
    __shared__ double sh[32];
    v = warpsum(v);
    int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (l == 0) sh[w] = v;
    __syncthreads();
    if (w == 0) { v = (l < (blockDim.x >> 5)) ? sh[l] : 0.0; v = warpsum(v); if (l == 0) atomicAdd(out, v); }
    __syncthreads();
}
__device__ void jac3f(float a[3][3], float v[3][3]) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) v[i][j] = (i==j) ? 1.f : 0.f;
    for (int sw=0; sw<20; sw++) {
        float off = a[0][1]*a[0][1] + a[0][2]*a[0][2] + a[1][2]*a[1][2];
        float dg = a[0][0]*a[0][0] + a[1][1]*a[1][1] + a[2][2]*a[2][2];
        if (!(off > 1e-14f*dg)) break;
        for (int p=0;p<2;p++) for (int q=p+1;q<3;q++) {
            if (a[p][q] == 0.f) continue;
            float th = 0.5f*(a[q][q]-a[p][p])/a[p][q];
            float t = (th >= 0.f ? 1.f : -1.f)/(fabsf(th) + sqrtf(th*th + 1.f));
            float c = rsqrtf(t*t + 1.f), s = t*c;
            for (int k=0;k<3;k++) { float x1 = a[k][p], x2 = a[k][q]; a[k][p] = c*x1 - s*x2; a[k][q] = s*x1 + c*x2; }
            for (int k=0;k<3;k++) { float x1 = a[p][k], x2 = a[q][k]; a[p][k] = c*x1 - s*x2; a[q][k] = s*x1 + c*x2; }
            for (int k=0;k<3;k++) { float x1 = v[k][p], x2 = v[k][q]; v[k][p] = c*x1 - s*x2; v[k][q] = s*x1 + c*x2; }
        }
    }
}
// 부호 있는 SVD (f32), fem_an_cuda.ssvd 와 같은 식
__device__ void ssvdf(const float* X, float* U, float* S, float* V) {
    float A[3][3], v[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) { float t = 0.f; for (int k=0;k<3;k++) t += X[k*3+i]*X[k*3+j]; A[i][j] = t; }
    jac3f(A, v);
    int o[3] = {0,1,2}; float e[3] = {A[0][0], A[1][1], A[2][2]};
    if (e[o[0]] < e[o[1]]) { int t = o[0]; o[0] = o[1]; o[1] = t; }
    if (e[o[1]] < e[o[2]]) { int t = o[1]; o[1] = o[2]; o[2] = t; }
    if (e[o[0]] < e[o[1]]) { int t = o[0]; o[0] = o[1]; o[1] = t; }
    for (int i=0;i<3;i++) for (int c=0;c<3;c++) V[i*3+c] = v[i][o[c]];
    float dv = V[0]*(V[4]*V[8]-V[5]*V[7]) - V[1]*(V[3]*V[8]-V[5]*V[6]) + V[2]*(V[3]*V[7]-V[4]*V[6]);
    if (dv < 0.f) for (int i=0;i<3;i++) V[i*3+2] = -V[i*3+2];
    float b[3][3];
    for (int c=0;c<3;c++) for (int a=0;a<3;a++) { float t = 0.f; for (int k=0;k<3;k++) t += X[a*3+k]*V[k*3+c]; b[c][a] = t; }
    float u1[3], u2[3], u3[3];
    float n1 = sqrtf(b[0][0]*b[0][0] + b[0][1]*b[0][1] + b[0][2]*b[0][2]);
    if (n1 > 0.f) for (int a=0;a<3;a++) u1[a] = b[0][a]/n1; else { u1[0] = 1.f; u1[1] = 0.f; u1[2] = 0.f; }
    float d12 = u1[0]*b[1][0] + u1[1]*b[1][1] + u1[2]*b[1][2];
    for (int a=0;a<3;a++) u2[a] = b[1][a] - d12*u1[a];
    float n2 = sqrtf(u2[0]*u2[0] + u2[1]*u2[1] + u2[2]*u2[2]);
    if (n2 > 1e-6f*n1 && n2 > 0.f) { for (int a=0;a<3;a++) u2[a] /= n2; }
    else {
        int ax = 0; if (fabsf(u1[1]) < fabsf(u1[ax])) ax = 1; if (fabsf(u1[2]) < fabsf(u1[ax])) ax = 2;
        float ea[3] = {0.f, 0.f, 0.f}; ea[ax] = 1.f;
        u2[0] = u1[1]*ea[2]-u1[2]*ea[1]; u2[1] = u1[2]*ea[0]-u1[0]*ea[2]; u2[2] = u1[0]*ea[1]-u1[1]*ea[0];
        float nn = sqrtf(u2[0]*u2[0] + u2[1]*u2[1] + u2[2]*u2[2]); for (int a=0;a<3;a++) u2[a] /= nn;
    }
    u3[0] = u1[1]*u2[2]-u1[2]*u2[1]; u3[1] = u1[2]*u2[0]-u1[0]*u2[2]; u3[2] = u1[0]*u2[1]-u1[1]*u2[0];
    for (int a=0;a<3;a++) { U[a*3] = u1[a]; U[a*3+1] = u2[a]; U[a*3+2] = u3[a]; }
    S[0] = u1[0]*b[0][0] + u1[1]*b[0][1] + u1[2]*b[0][2];
    S[1] = u2[0]*b[1][0] + u2[1]*b[1][1] + u2[2]*b[1][2];
    S[2] = u3[0]*b[2][0] + u3[1]*b[2][1] + u3[2]*b[2][2];
}
__device__ __forceinline__ void bwf(float f, float* w, float* d) {
    w[0] = 0.5f*(1.5f-f)*(1.5f-f); w[1] = 0.75f-(f-1.f)*(f-1.f); w[2] = 0.5f*(f-0.5f)*(f-0.5f);
    d[0] = f-1.5f; d[1] = -2.f*(f-1.f); d[2] = f-0.5f;
}
// IP (int): 0..2 bmin, 3..5 bmax, 6..8 nb (노드 상자), 9 노드 수, 10 칸 상자 수, 11 압축 칸 수 nc, 12 가우스 점 수 nq, 13 넘침
// ---- P2G ----
__global__ void rt_reset(int* IP) { IP[0] = IP[1] = IP[2] = 2147483647; IP[3] = IP[4] = IP[5] = -2147483647; IP[13] = 0; IP[14] = 0; }
__global__ void rt_coord(int N, const float* x, const float* Ai, int* pb, float* fx, int* IP) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = p < N;
    if (__all_sync(0xffffffffu, !on)) return;
    for (int a=0;a<3;a++) {                                                    // 상자 최소·최대: 워프 안에서 먼저 줄인 뒤 원자 연산 하나 (정수라 결과 같음, 경합만 줄임)
        int b0 = 2147483647, b1 = -2147483647;
        if (on) {
            float y = Ai[a*3]*x[p*3] + Ai[a*3+1]*x[p*3+1] + Ai[a*3+2]*x[p*3+2];
            float fb = floorf(y - 0.5f);
            int b = (int)fb; pb[p*3+a] = b; fx[p*3+a] = y - fb; b0 = b; b1 = b;
        }
        b0 = __reduce_min_sync(0xffffffffu, b0); b1 = __reduce_max_sync(0xffffffffu, b1);
        if ((threadIdx.x & 31) == 0) { atomicMin(&IP[a], b0); atomicMax(&IP[3+a], b1); }
    }
}
__global__ void rt_box(int* IP, int capN, int capC) {
    int nb0 = IP[3]-IP[0]+3, nb1 = IP[4]-IP[1]+3, nb2 = IP[5]-IP[2]+3;
    IP[6] = nb0; IP[7] = nb1; IP[8] = nb2;
    long long nn = (long long)nb0*nb1*nb2, nc = (long long)(nb0-2)*(nb1-2)*(nb2-2);
    IP[9] = (int)nn; IP[10] = (int)nc;
    if (nn > capN || nc > capC) { IP[13] = 1; IP[9] = 0; IP[10] = 0; }
}
__global__ void rt_zero_f(int n, const int* cnt, int ci, int mul, float* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; int m = cnt ? cnt[ci]*mul : n; if (i < m && i < n) a[i] = 0.f; }
__global__ void rt_zero_d(int n, const int* cnt, int ci, int mul, double* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; int m = cnt ? cnt[ci]*mul : n; if (i < m && i < n) a[i] = 0.0; }
__global__ void rt_zero_i(int n, int* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = 0; }
// 입자: 상자 안 base, 칸 상자 번호 표시
__global__ void rt_mark(int N, int* pb, const float* fx, const int* IP, int* pcd, int* occ) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int b0 = pb[p*3]-IP[0], b1 = pb[p*3+1]-IP[1], b2 = pb[p*3+2]-IP[2];
    pb[p*3] = b0; pb[p*3+1] = b1; pb[p*3+2] = b2;
    int cd = (b0*(IP[7]-2) + b1)*(IP[8]-2) + b2;
    pcd[p] = cd; occ[cd] = 1;
}
// 한 블록 배타 스캔: occ -> cidx (압축 번호, 비점유 -1), 칸 base 좌표 cb, nc·nq
__global__ void rt_scan(int* IP, const int* occ, int* cidx, int* cb, int capQc) {
    __shared__ int sh[1024];
    int n = IP[13] ? 0 : IP[10];
    int carry = 0;
    int n1 = IP[7]-2, n2 = IP[8]-2;
    for (int base = 0; base < n; base += 1024) {
        int i = base + threadIdx.x;
        int v = (i < n) ? occ[i] : 0;
        sh[threadIdx.x] = v;
        __syncthreads();
        for (int o=1; o<1024; o<<=1) { int t = (threadIdx.x >= o) ? sh[threadIdx.x - o] : 0; __syncthreads(); sh[threadIdx.x] += t; __syncthreads(); }
        int inc = sh[threadIdx.x];
        if (i < n) {
            int id = carry + inc - v;
            if (v) { if (id < capQc) { cidx[i] = id; cb[id*3] = i / (n1*n2); cb[id*3+1] = (i / n2) % n1; cb[id*3+2] = i % n2; } }
            else cidx[i] = -1;
        }
        int tot = sh[1023];
        __syncthreads();
        carry += tot;
    }
    if (threadIdx.x == 0) { IP[11] = carry; IP[12] = carry*8; if (carry > capQc) { IP[13] = 1; IP[11] = 0; IP[12] = 0; } }
}
// 절 CM3 (--ncomp): rt_scan 과 같은 출력 (정수, 비트 같음) 을 격자 3 단계로 -- 칸 상자 수십만 칸을 한 블록이 1024 씩 도는 rt_scan (0.7 ms) 대신.
__global__ void rt_scan_a(const int* IP, const int* occ, int* bc) {                 // 블록 256 고정: 블록별 점유 칸 수
    int n = IP[13] ? 0 : IP[10];
    if (blockIdx.x*blockDim.x >= n) return;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int c = __syncthreads_count(i < n && occ[i]);
    if (threadIdx.x == 0) bc[blockIdx.x] = c;
}
__global__ void rt_scan_b(int* IP, const int* bc, int* bo, int capQc) {               // 한 블록 1024: 블록 수 배타 스캔 -> nc·nq (넘치면 표시)
    __shared__ int sh[1024];
    int nb = ((IP[13] ? 0 : IP[10]) + 255) / 256, carry = 0;
    for (int base = 0; base < nb; base += 1024) {
        int i = base + threadIdx.x;
        int v = (i < nb) ? bc[i] : 0;
        sh[threadIdx.x] = v;
        __syncthreads();
        for (int o=1; o<1024; o<<=1) { int t = (threadIdx.x >= o) ? sh[threadIdx.x - o] : 0; __syncthreads(); sh[threadIdx.x] += t; __syncthreads(); }
        if (i < nb) bo[i] = carry + sh[threadIdx.x] - v;
        int tot = sh[1023];
        __syncthreads();
        carry += tot;
    }
    if (threadIdx.x == 0) { IP[11] = carry; IP[12] = carry*8; if (carry > capQc) { IP[13] = 1; IP[11] = 0; IP[12] = 0; } }
}
__global__ void rt_scan_c(const int* IP, const int* occ, const int* bo, int* cidx, int* cb, int capQc) {   // 블록 256 고정: 자리 = 블록 시작 + 블록 안 순번
    __shared__ int ws[8];
    int n = IP[10];
    if (blockIdx.x*blockDim.x >= n) return;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int v = (i < n) ? occ[i] : 0;
    unsigned int m = __ballot_sync(0xffffffffu, v);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) ws[w] = __popc(m);
    __syncthreads();
    if (i >= n) return;
    int id = bo[blockIdx.x];
    for (int k=0;k<w;k++) id += ws[k];
    id += __popc(m & ((1u << lane) - 1u));
    int n1 = IP[7]-2, n2 = IP[8]-2;
    if (v) { if (id < capQc) { cidx[i] = id; cb[id*3] = i / (n1*n2); cb[id*3+1] = (i / n2) % n1; cb[id*3+2] = i % n2; } }
    else cidx[i] = -1;
}
// 입자 -> 가우스 점 누적 (회전 R, log S, F, 가중) + 칸 누적 + 칸 부피·질량 + 노드 질량·운동량
// A: [q*28]: 0..8 R, 9..17 LS, 18..26 F, 27 w;  C: [c*30]: 0..27 같은 것, 28 V, 29 M;  bad: q 와 c (int, 뒤집힌 입자)
__global__ void rt_p2g(int N, const float* x, const float* v, const float* F, const float* m, const float* vol, const int* pb, const float* fx, const int* pcd,
                       const int* cidx, const int* IP, float* A, float* C, int* badq, int* badc, float* mn, float* mv, const float* pa, float* ma) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
    int q = c*8 + oc;
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    float U[9], S[3], V[9];
    ssvdf(Fp, U, S, V);
    float w = vol[p];
    bool ng = !(S[2] > 0.f);
    float R[9], L[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) R[a*3+b] = U[a*3]*V[b*3] + U[a*3+1]*V[b*3+1] + U[a*3+2]*V[b*3+2];
    float l[3]; for (int i=0;i<3;i++) l[i] = ng ? 0.f : logf(S[i]);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) L[a*3+b] = V[a*3]*l[0]*V[b*3] + V[a*3+1]*l[1]*V[b*3+1] + V[a*3+2]*l[2]*V[b*3+2];
    float* aq = A + (size_t)q*28; float* ac = C + (size_t)c*30;
    for (int k=0;k<9;k++) { atomicAdd(&aq[k], w*R[k]); atomicAdd(&aq[9+k], w*L[k]); atomicAdd(&aq[18+k], w*Fp[k]);
                            atomicAdd(&ac[k], w*R[k]); atomicAdd(&ac[9+k], w*L[k]); atomicAdd(&ac[18+k], w*Fp[k]); }
    atomicAdd(&aq[27], w); atomicAdd(&ac[27], w); atomicAdd(&ac[28], w); atomicAdd(&ac[29], m[p]);
#ifdef AF_BAR
    for (int a=0;a<3;a++) atomicAdd(&AF_BX[q*4+a], m[p]*x[p*3+a]); atomicAdd(&AF_BX[q*4+3], m[p]);   // 장벽 판: 팔분 점 입자 질량 중심 (표본점)
#endif
#ifdef AF_OCT
    atomicAdd(&AF_MO[q], m[p]);                                                     // 팔분 칸 실제 입자 질량 (점유 가중, 절 AN.err4 정의)
#endif
    if (ng) { atomicOr(&badq[q], 1); atomicOr(&badc[c], 1); }
#ifndef AF_GQM
    float wx[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], wx[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float mp = m[p];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = wx[0][i]*wx[1][j]*wx[2][k]*mp;
        atomicAdd(&mn[nid], ww);
        for (int a=0;a<3;a++) atomicAdd(&mv[nid*3+a], ww*v[p*3+a]);
        if (ma) for (int a=0;a<3;a++) atomicAdd(&ma[nid*3+a], ww*pa[p*3+a]);   // generalized-α: 입자 가속도 -> 노드 (질량 가중)
    }
#endif
}
__device__ void rs_fin(const float* sR, const float* sL, float w, float* out, bool* bad) {
    float U[9], S[3], V[9];
    ssvdf(sR, U, S, V);
    float Rb[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Rb[a*3+b] = U[a*3]*V[b*3] + U[a*3+1]*V[b*3+1] + U[a*3+2]*V[b*3+2];
    float Lm[3][3], Ev[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) Lm[i][j] = 0.5f*(sL[i*3+j] + sL[j*3+i])/w;
    jac3f(Lm, Ev);
    float ex[3]; for (int i=0;i<3;i++) ex[i] = expf(Lm[i][i]);
    float Sb[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Sb[a*3+b] = Ev[a][0]*ex[0]*Ev[b][0] + Ev[a][1]*ex[1]*Ev[b][1] + Ev[a][2]*ex[2]*Ev[b][2];
    *bad = false;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = Rb[a*3]*Sb[b] + Rb[a*3+1]*Sb[3+b] + Rb[a*3+2]*Sb[6+b]; out[a*3+b] = t; if (!isfinite(t)) *bad = true; }
}
// 가우스 점 마무리: F_q, V_q, m_q, 접촉용 시작 위치 xq0 (월드), 노드 몫 질량 mqn (원자 합)
__global__ void rt_gauss(const int* IP, const float* A, const float* C, const int* badq, const int* badc, const int* cb, const float* Am, const float* BW,
                         float* Fq, float* Vq, float* mq, float* xq0, float* mqn, int* nbad) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int c = q >> 3, oc = q & 7;
    const float* aq = A + (size_t)q*28; const float* ac = C + (size_t)c*30;
    float out[9]; bool bad = false;
    if (aq[27] > 0.f) {
        rs_fin(aq, aq + 9, aq[27], out, &bad);
        if (bad || badq[q]) { for (int k=0;k<9;k++) out[k] = aq[18+k]/aq[27]; atomicAdd(nbad, 1); }
    } else {
        rs_fin(ac, ac + 9, ac[27], out, &bad);
        if (bad || badc[c]) for (int k=0;k<9;k++) out[k] = ac[18+k]/ac[27];
    }
    for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = out[k];
    Vq[q] = ac[28]*0.125f; float mm = ac[29]*0.125f;
#ifdef AF_OCT
    Vq[q] = (1.f - AF_GOCT)*aq[27] + AF_GOCT*Vq[q]; mm = (1.f - AF_GOCT)*AF_MO[q] + AF_GOCT*mm;   // 점유 가중: (1 − κ) 팔분 칸 실제 부피·질량 + κ 칸 합 / 8
#endif
    mq[q] = mm;
    const float GG = 0.28867513459481287f;
    float yq[3] = {(float)(IP[0] + cb[c*3]) + 1.f + ((oc & 4) ? GG : -GG), (float)(IP[1] + cb[c*3+1]) + 1.f + ((oc & 2) ? GG : -GG), (float)(IP[2] + cb[c*3+2]) + 1.f + ((oc & 1) ? GG : -GG)};
    for (int a=0;a<3;a++) xq0[q*3+a] = Am[a*3]*yq[0] + Am[a*3+1]*yq[1] + Am[a*3+2]*yq[2];
    int nb1 = IP[7], nb2 = IP[8];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((cb[c*3]+i)*nb1 + (cb[c*3+1]+j))*nb2 + (cb[c*3+2]+k);
        atomicAdd(&mqn[nid], BW[oc*27+o]*mm);
    }
}
__global__ void rt_vn(const int* IP, const float* mn, const float* mv, float* vk, float* U) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n];
    for (int a=0;a<3;a++) { vk[n*3+a] = (m > 0.f) ? mv[n*3+a]/m : 0.f; U[n*3+a] = 0.f; }
}
// ---- 풀이 ----
// 가우스 점 공통: 27 노드 번호, G_qn = F_qᵀ ∇w (BG 월드 기울기 표)
__device__ __forceinline__ void qnodes(int q, const int* IP, const int* cb, int* ids) {
    int c = q >> 3; int nb1 = IP[7], nb2 = IP[8];
    for (int o=0;o<27;o++) { int i = o/9, j = (o/3)%3, k = o%3; ids[o] = ((cb[c*3]+i)*nb1 + (cb[c*3+1]+j))*nb2 + (cb[c*3+2]+k); }
}
__device__ __forceinline__ void gqn(const float* Fq, const float* bg, float* g) {
    g[0] = Fq[0]*bg[0] + Fq[3]*bg[1] + Fq[6]*bg[2];
    g[1] = Fq[1]*bg[0] + Fq[4]*bg[1] + Fq[7]*bg[2];
    g[2] = Fq[2]*bg[0] + Fq[5]*bg[1] + Fq[8]*bg[2];
}
// 걸음 시작: Ut = U + τ v + τ² g, z0 = U + β τ v, y = Ut (고정 노드는 U + τ vbc)
__global__ void rt_step0(const int* IP, const float* U, const float* vk, float tau, float beta, float g0, float g1, float g2, const unsigned char* fixd, const float* vbc,
                         float* Ut, float* z0, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float gg[3] = {g0, g1, g2};
    for (int a=0;a<3;a++) {
        float u = U[n*3+a], vv = vk[n*3+a];
        float ut = u + tau*vv + tau*tau*gg[a];
        Ut[n*3+a] = ut; z0[n*3+a] = u + beta*tau*vv;
        y[n*3+a] = (fixd && fixd[n]) ? u + tau*vbc[n*3+a] : ut;
    }
}
// 바닥: 가우스 점 걸음 끝 위치 -> 정답 sticky 가중 (s < 0 만) -> 노드 몫 an (원자 합)
__global__ void rt_floor_q(const int* IP, const int* cb, const float* BW, const float* xq0, const float* mq, const float* y, const float* U, float thinv,
                           float p0, float p1, float p2, float n0, float n1, float n2, float pn, float dxg, float* an) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
#ifdef AF_FSTK
    bool qa_ = dxg < 0.f; dxg = fabsf(dxg);                                     // 절 CM12 (--fl_stk): dxg < 0 으로 부르면 (A) 가우스 점 고착 판정 (CM8 a 판 정의 그대로: 가우스 점 xq0, s < 0, 정답 sticky 가중)
#endif
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xq[3] = {xq0[q*3], xq0[q*3+1], xq0[q*3+2]};
#ifdef AF_BAR
#ifdef AF_FSTK
    float bm_ = qa_ ? 0.f : AF_BX[q*4+3]; if (bm_ > 0.f) for (int a=0;a<3;a++) xq[a] = AF_BX[q*4+a]/bm_;
#else
    float bm_ = AF_BX[q*4+3]; if (bm_ > 0.f) for (int a=0;a<3;a++) xq[a] = AF_BX[q*4+a]/bm_;   // 장벽 판: 고착 판정도 표본점 (팔분 입자 질량 중심) 에서
#endif
#endif
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
        for (int a=0;a<3;a++) { float ye = U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a]); xq[a] += w*ye; } }
    float s = (xq[0]-p0)*n0 + (xq[1]-p1)*n1 + (xq[2]-p2)*n2;
#ifdef AF_BAR
#ifdef AF_FSTK
    float sw_ = s; if (!qa_) s = (bm_ > 0.f) ? -1.f : 1.f;
#else
    float sw_ = s; s = (bm_ > 0.f) ? -1.f : 1.f;                                 // 장벽 판: 정답 sticky 가중 전체 (평면 위 띠 포함), 빈 팔분 점 없음
#endif
#else
    float sw_ = s;
#endif
    if (!(s < 0.f)) return;
    float u = (sw_ + pn)/dxg; float bf = floorf(u - 0.5f); float f = u - bf;
    float w3[3] = {0.5f*(1.5f-f)*(1.5f-f), 0.75f-(f-1.f)*(f-1.f), 0.5f*(f-0.5f)*(f-0.5f)};
    float wst = 0.f; for (int i=0;i<3;i++) if ((bf + (float)i)*dxg - pn < 0.f) wst += w3[i];
#ifdef AF_BST
#ifdef AF_FSTK
    if (qa_) ; else
#endif
    if (sw_ < AF_BDHC) wst = 1.f;                                               // 띠 고착 (--fl_bst, 절 AN.21): 장벽 띠 안 (d < d̂, 평면 아래 포함) 표본점은 완전 고착 (정답 sticky 노드 v = 0 과 같은 몫 1)
    else if (AF_BST == 1) wst = 0.f;                                            //   1 = 띠 밖은 고착 없음, 2 = 띠 밖은 정답 sticky 가중 그대로
#endif
    float mw = mq[q]*wst;
#ifdef AF_FLJ
    mw = mq[q]*(1.f - expf(AF_KEXQ*log1pf(-fminf(wst, 1.f))));                            // 점별 고착 (--fl_pt, 절 AN.17): 점 질량 x (1 − (1 − w_st)^(Δt/dt_정답)) -- 거듭제곱을 점에서 한 뒤 노드로 평균
#endif
    if (mw > 0.f) for (int o=0;o<27;o++) atomicAdd(&an[ids[o]], BW[oc*27+o]*mw);
}
// 노드: 고착 인자 k, 변수 바꿈 (sk, pm), ŷ, η, y = ŷ + T η. pm 0 = P I, 1 = P (I − nnᵀ) (바깥 법선 이탈), 2 = 고정 (T 0)
__device__ __forceinline__ void Tapply(float sk, int pm, float n0, float n1, float n2, const float* v, float* o) {
#if defined(AF_FSTK) || defined(AF_FUP)
    if (pm < 0) { float sn_ = __int_as_float(-pm - 1); float vn_ = v[0]*n0 + v[1]*n1 + v[2]*n2;          // 절 CM12: 접선 √k_t·법선 √k_n 따로 (pm = −1 − 비트(√k_n))
        o[0] = sk*(v[0]-vn_*n0) + sn_*vn_*n0; o[1] = sk*(v[1]-vn_*n1) + sn_*vn_*n1; o[2] = sk*(v[2]-vn_*n2) + sn_*vn_*n2; return; }
#endif
#ifdef AF_NCL
    if (pm == 3) { float vn3 = v[0]*n0 + v[1]*n1 + v[2]*n2; o[0] = sk*(v[0]-vn3*n0); o[1] = sk*(v[1]-vn3*n1); o[2] = sk*(v[2]-vn3*n2); return; }   // 절 CM16 노드 법선 클램프: 접선 √k 만, 법선 0
#endif
    if (pm == 2) { o[0] = o[1] = o[2] = 0.f; return; }
    if (pm == 0) { o[0] = sk*v[0]; o[1] = sk*v[1]; o[2] = sk*v[2]; return; }
    float vn = v[0]*n0 + v[1]*n1 + v[2]*n2;                                   // P v = v − (v·n) n (접선), (I − P) v = (v·n) n
    o[0] = sk*(v[0]-vn*n0) + vn*n0; o[1] = sk*(v[1]-vn*n1) + vn*n1; o[2] = sk*(v[2]-vn*n2) + vn*n2;
}
#ifdef AF_FSTK
// 절 CM12 노드 변수 바꿈 (접선 k_t, 법선 k_n 따로): ŷ = Ũ + (1 − k_t) P d + (1 − k_n) (I − P) d (d = z0' − Ũ), η = √k_t P (y − ŷ) + √k_n (I − P)(y − ŷ), y = ŷ + T η.
// k_n = 1 이면 pm 1 경로와 같은 식, k_n = k_t 이면 pm 0 경로와 같은 식 (이 함수는 k_n ≠ 1 일 때만 부름).
__device__ void fstk_node(int n, float kt, float kn, float n0, float n1, float n2, const float* d, const float* Ut, float* y, float* sk, int* pm, float* yh, float* eta) {
    float st = sqrtf(kt), sn = sqrtf(kn);
    float dn = d[0]*n0 + d[1]*n1 + d[2]*n2; float nn[3] = {n0, n1, n2};
    float h[3]; for (int a=0;a<3;a++) h[a] = Ut[n*3+a] + (1.f-kt)*(d[a]-dn*nn[a]) + (1.f-kn)*dn*nn[a];
    float e2[3]; for (int a=0;a<3;a++) e2[a] = y[n*3+a] - h[a];
    float c2 = e2[0]*n0 + e2[1]*n1 + e2[2]*n2;
    int pmv = -1 - __float_as_int(sn);
    float et[3]; for (int a=0;a<3;a++) et[a] = st*(e2[a] - c2*nn[a]) + sn*c2*nn[a];
    float ty[3]; Tapply(st, pmv, n0, n1, n2, et, ty);
    for (int a=0;a<3;a++) { yh[n*3+a] = h[a]; eta[n*3+a] = et[a]; y[n*3+a] = h[a] + ty[a]; }
    sk[n] = st; pm[n] = pmv;
}
#endif
__global__ void rt_floor_n(const int* IP, const float* an, const float* mqn, float kex, int floor_, int vout, float al, float n0, float n1, float n2,
                           const unsigned char* fixd, const float* vbc, const float* U, float tau, const float* Ut, const float* z0, float* y, float* sk, int* pm, float* yh, float* eta) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float k = 1.f; int p = 0;
#ifdef AF_FSTK
    float kfn_ = 1.f;
#endif
    if (floor_) {
        float r = (mqn[n] > 0.f) ? fminf(an[n]/mqn[n], 1.f) : 0.f;
        k = expf(kex*log1pf(-r));
#ifdef AF_FLJ
        k = 1.f - r;                                                             // 점별 고착: an 이 이미 거듭제곱한 몫의 질량 합
#endif
        float vv = al*((y[n*3]-z0[n*3])*n0 + (y[n*3+1]-z0[n*3+1])*n1 + (y[n*3+2]-z0[n*3+2])*n2);
        p = (vout && vv > 0.f) ? 1 : 0;
#ifdef AF_BAR
        p = 1;                                                                   // 장벽 판: 법선은 장벽, 고착은 접선에만
#endif
#ifdef AF_FSTK
        float ra_ = (mqn[n] > 0.f) ? fminf(AF_ANA[n]/mqn[n], 1.f) : 0.f;       // 절 CM12: (A) 가우스 점 고착 몫 -> k_A = (1 − r̄_A)^κ, 법선은 바깥 이탈 (vout, vv > 0) 이 아니면 k_A
        float ka_ = expf(kex*log1pf(-ra_));
        kfn_ = (vout && vv > 0.f) ? 1.f : ka_; k *= ka_;
#endif
    }
    if (fixd && fixd[n]) p = 2;
#ifdef AF_FSTK
    if (p != 2 && kfn_ != 1.f) { float dd_[3]; for (int a=0;a<3;a++) dd_[a] = z0[n*3+a] - Ut[n*3+a]; fstk_node(n, k, kfn_, n0, n1, n2, dd_, Ut, y, sk, pm, yh, eta); return; }
#endif
    float s = sqrtf(k);
    float d[3], yy[3];
    for (int a=0;a<3;a++) d[a] = z0[n*3+a] - Ut[n*3+a];
    if (p == 2) { for (int a=0;a<3;a++) { yy[a] = U[n*3+a] + tau*vbc[n*3+a]; yh[n*3+a] = yy[a]; eta[n*3+a] = 0.f; y[n*3+a] = yy[a]; } sk[n] = 0.f; pm[n] = 2; return; }
    float Pd[3];
    if (p == 0) { for (int a=0;a<3;a++) Pd[a] = d[a]; }
    else { float dn = d[0]*n0 + d[1]*n1 + d[2]*n2; Pd[0] = d[0]-dn*n0; Pd[1] = d[1]-dn*n1; Pd[2] = d[2]-dn*n2; }
    float h[3]; for (int a=0;a<3;a++) h[a] = Ut[n*3+a] + (1.f-k)*Pd[a];
    float e1[3], e2[3];
    for (int a=0;a<3;a++) { e1[a] = y[n*3+a] - Ut[n*3+a]; e2[a] = y[n*3+a] - h[a]; }
    float et[3];
    if (p == 0) { for (int a=0;a<3;a++) et[a] = s*e2[a]; }
    else { float c1 = e1[0]*n0 + e1[1]*n1 + e1[2]*n2; float c2 = e2[0]*n0 + e2[1]*n1 + e2[2]*n2;
           for (int a=0;a<3;a++) { float nn_ = (a==0) ? n0 : ((a==1) ? n1 : n2); et[a] = c1*nn_ + s*(e2[a] - c2*nn_); } }
    float ty[3]; Tapply(s, p, n0, n1, n2, et, ty);
    for (int a=0;a<3;a++) { yh[n*3+a] = h[a]; eta[n*3+a] = et[a]; y[n*3+a] = h[a] + ty[a]; }
    sk[n] = s; pm[n] = p;
}
// 요소: mode 1 기울기 (f64 원자, 8 레인 구간 합), 2 헤시안 (f32 저장), 4 에너지 (f64 블록 합)
__global__ void __launch_bounds__(128) rt_elem(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, float mu, float lam, int mode,
                        double* E, double* f, float* H, int capQ, float* B) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    int nq = IP[12];
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = y[id*3+a]; X[a*3] += ya*g[0]; X[a*3+1] += ya*g[1]; X[a*3+2] += ya*g[2]; } }
    float U[9], S[3], V[9];
    ssvdf(X, U, S, V);
    float J = S[0]*S[1]*S[2];
    float gJ[3] = {S[1]*S[2], S[0]*S[2], S[0]*S[1]};
    float vq = on ? Vq[qq] : 0.f;
    if (mode & 4) {
#ifdef AF_SNH
        double e = on ? snh_e((double)vq, S, mu, lam) : 0.0;                    // 절 CM18 SNH
#else
        double e = on ? (double)vq*((double)mu*(((double)S[0]-1.0)*((double)S[0]-1.0) + ((double)S[1]-1.0)*((double)S[1]-1.0) + ((double)S[2]-1.0)*((double)S[2]-1.0)) + 0.5*(double)lam*((double)J-1.0)*((double)J-1.0)) : 0.0;
#endif
        blocksum_add(e, E);
    }
    if (mode & 1) {
#ifdef AF_SNH
        float ph[3]; snh_ph(S, mu, lam, ph);                                       // 절 CM18 SNH: P_i = μ σ_i + (λs (J − 1) − μ) ∂J/∂σ_i
#else
        float ph[3]; for (int i=0;i<3;i++) ph[i] = 2.f*mu*(S[i]-1.f) + lam*(J-1.f)*gJ[i];
#endif
        float P[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) P[a*3+b] = U[a*3]*ph[0]*V[b*3] + U[a*3+1]*ph[1]*V[b*3+1] + U[a*3+2]*ph[2]*V[b*3+2];
        for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
            for (int a=0;a<3;a++) { double s = seg8d((double)(vq*(P[a*3]*g[0] + P[a*3+1]*g[1] + P[a*3+2]*g[2])));
                if (on && head) atomicAdd(&f[ids[o]*3+a], s); } }
    }
    if (mode & 2) {                                                             // 헤시안 (모든 레인이 들어옴: 대각 구간 합 때문에; 밖 레인은 0)
        float h[81]; for (int k=0;k<81;k++) h[k] = 0.f;
        if (on) {
        float D[3][3], Ev[3][3];
        float Sg[3][3] = {{0.f, S[2], S[1]}, {S[2], 0.f, S[0]}, {S[1], S[0], 0.f}};
#ifdef AF_SNH
        snh_D(S, mu, lam, D);                                                      // 절 CM18 SNH: 주값 헤시안 (투영 전)
#else
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = ((i==j) ? 2.f*mu : 0.f) + lam*gJ[i]*gJ[j] + lam*(J-1.f)*Sg[i][j];
#endif
        jac3f(D, Ev);
        for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (!(dm > 0.f)) continue;
            float e[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) e[a*3+b] = Ev[0][m]*U[a*3]*V[b*3] + Ev[1][m]*U[a*3+1]*V[b*3+1] + Ev[2][m]*U[a*3+2]*V[b*3+2];
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += dm*e[r]*e[c]; }
        const int PI[3] = {0,0,1}, PJ[3] = {1,2,2}, PK[3] = {2,1,0};
        const float r2 = 0.70710678f;
        for (int pp=0; pp<3; pp++) { int i = PI[pp], j = PJ[pp], k = PK[pp];
#ifdef AF_SNH
            float lt, lf; snh_tf(S, k, mu, lam, &lt, &lf);                             // 절 CM18 SNH: 비틀림·뒤집힘 고윳값 (나눗셈 없음)
#else
            float lt = 2.f*mu*(1.f - 2.f/(S[i]+S[j])) + lam*(J-1.f)*S[k]; float lf = 2.f*mu - lam*(J-1.f)*S[k];
#endif
            lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = fmaxf(lf, 0.f);
            float et[9], ef[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float x1 = U[a*3+i]*V[b*3+j], x2 = U[a*3+j]*V[b*3+i]; et[a*3+b] = r2*(x1-x2); ef[a*3+b] = r2*(x1+x2); }
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += lt*et[r]*et[c] + lf*ef[r]*ef[c]; }
#ifdef AF_HSPEC
        {                                                                         // 절 CM3 --hspec: 같은 헤시안의 주값 분해 30 성분 (U 9, M = F V 9, C 12 x V_q) -- rt_mv2s 가 읽음
            float vq_ = vq, Cs[12];
            const int RI[6] = {0,1,2,0,0,1}, RJ[6] = {0,1,2,1,2,2};
            for (int k=0;k<6;k++) { float t = 0.f; for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (dm > 0.f) t += dm*Ev[RI[k]][m]*Ev[RJ[k]][m]; } Cs[k] = vq_*t; }
            for (int pp=0; pp<3; pp++) { int i = PI[pp], j = PJ[pp], k = PK[pp];
#ifdef AF_SNH
                float lt, lf; snh_tf(S, k, mu, lam, &lt, &lf);                             // 절 CM18 SNH: 비틀림·뒤집힘 고윳값 (나눗셈 없음)
#else
                float lt = 2.f*mu*(1.f - 2.f/(S[i]+S[j])) + lam*(J-1.f)*S[k]; float lf = 2.f*mu - lam*(J-1.f)*S[k];
#endif
                lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = fmaxf(lf, 0.f);
                Cs[6+pp] = vq_*0.5f*(lt + lf); Cs[9+pp] = vq_*0.5f*(lf - lt); }
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) H[(size_t)(9 + a*3+b)*capQ + qq] = F[a*3]*V[b] + F[a*3+1]*V[3+b] + F[a*3+2]*V[6+b];
            for (int k=0;k<9;k++) H[(size_t)k*capQ + qq] = U[k];
            for (int k=0;k<12;k++) H[(size_t)(18+k)*capQ + qq] = Cs[k];
        }
#else
        for (int k=0;k<81;k++) H[(size_t)k*capQ + qq] = h[k];                     // 성분 우선 배치 (곱 커널 읽기가 합쳐지게)
#endif
        }
        if (mode & 8) {                                                           // 노드 블록 대각 (융합): m_ac = V Σ_b g_b Σ_d H[(ab),(cd)] g_d
            for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
                float m[9];
                for (int a=0;a<3;a++) for (int c=0;c<3;c++) { float t = 0.f;
                    for (int b=0;b<3;b++) { float hg = h[(a*3+b)*9 + c*3]*g[0] + h[(a*3+b)*9 + c*3+1]*g[1] + h[(a*3+b)*9 + c*3+2]*g[2]; t += g[b]*hg; }
                    m[a*3+c] = vq*t; }
                for (int k=0;k<9;k++) m[k] = seg8f(m[k]);
                if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], m[k]); }
        }
    }
}
// 노드 블록 대각 Σ V Σ_bd H[(ab),(cd)] g_b g_d (f32, 8 레인 구간 합)
__global__ void __launch_bounds__(128) rt_bd(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* H, float* B) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    const float* h = H + (size_t)qq*81;
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
        float m[9];
        for (int a=0;a<3;a++) for (int c=0;c<3;c++) { float t = 0.f;
            for (int b=0;b<3;b++) for (int d=0;d<3;d++) t += h[(a*3+b)*9 + c*3+d]*g[b]*g[d];
            m[a*3+c] = vq*t; }
        for (int k=0;k<9;k++) m[k] = seg8f(m[k]);
        if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], m[k]); }
}
// 헤시안 곱 out += Σ V (H δX) g, δX = Σ dy ⊗ g (f32)
__global__ void __launch_bounds__(128) rt_mv(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* H, const float* dy, float* out, int capQ) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float dX[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = dy[id*3+a]; dX[a*3] += ya*g[0]; dX[a*3+1] += ya*g[1]; dX[a*3+2] += ya*g[2]; } }
    float vq = on ? Vq[qq] : 0.f;
    float dP[9]; for (int r=0;r<9;r++) { float t = 0.f; for (int c=0;c<9;c++) t += H[(size_t)(r*9+c)*capQ + qq]*dX[c]; dP[r] = vq*t; }
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
        for (int a=0;a<3;a++) { float s = seg8f(dP[a*3]*g[0] + dP[a*3+1]*g[1] + dP[a*3+2]*g[2]); if (on && head) atomicAdd(&out[ids[o]*3+a], s); } }
}
// 노드: 기울기 g = Mi η + T f (f64), r = −g (f32), 블록 야코비 Dinv = (Mi I + T B T)⁻¹, z = D r, p = z; 스칼라 SC: 0 rz, 1 |g|², ...
__device__ __forceinline__ void inv3(const float* M, float* Mi) {
    float c0 = M[4]*M[8]-M[5]*M[7], c1 = M[5]*M[6]-M[3]*M[8], c2 = M[3]*M[7]-M[4]*M[6];
    float d = M[0]*c0 + M[1]*c1 + M[2]*c2;
    if (d == 0.f || !isfinite(d)) { for (int k=0;k<9;k++) Mi[k] = 0.f; return; }
    float id = 1.f/d;
    Mi[0] = c0*id; Mi[1] = (M[2]*M[7]-M[1]*M[8])*id; Mi[2] = (M[1]*M[5]-M[2]*M[4])*id;
    Mi[3] = c1*id; Mi[4] = (M[0]*M[8]-M[2]*M[6])*id; Mi[5] = (M[2]*M[3]-M[0]*M[5])*id;
    Mi[6] = c2*id; Mi[7] = (M[1]*M[6]-M[0]*M[7])*id; Mi[8] = (M[0]*M[4]-M[1]*M[3])*id;
}
__device__ __forceinline__ void Tmat(float sk, int pm, float n0, float n1, float n2, float* T) {
    float nn[3] = {n0, n1, n2};
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
        float I = (a==b) ? 1.f : 0.f;
#if defined(AF_FSTK) || defined(AF_FUP)
        if (pm < 0) { float Pab = I - nn[a]*nn[b]; T[a*3+b] = sk*Pab + __int_as_float(-pm - 1)*(I - Pab); continue; }
#endif
#ifdef AF_NCL
        if (pm == 3) { T[a*3+b] = sk*(I - nn[a]*nn[b]); continue; }                  // 절 CM16 노드 법선 클램프 (Tapply pm 3 과 같은 T)
#endif
        if (pm == 2) T[a*3+b] = 0.f; else if (pm == 0) T[a*3+b] = sk*I; else { float Pab = I - nn[a]*nn[b]; T[a*3+b] = sk*Pab + (I - Pab); }
    }
}
__global__ void rt_grad(const int* IP, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* eta, const double* f,
                        const float* B, float* Dinv, float* r, float* z, float* p, float* dl, float* Tp, double* SC, float* r0) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double rz = 0.0, gg = 0.0, ff = 0.0;
    if (n < IP[9]) {
        float T[9]; Tmat(sk[n], pm[n], n0, n1, n2, T);
        float Mi = mn[n]*itau2;
        double g[3];
        for (int a=0;a<3;a++) g[a] = (double)Mi*eta[n*3+a] + (double)T[a*3]*f[n*3] + (double)T[a*3+1]*f[n*3+1] + (double)T[a*3+2]*f[n*3+2];
#ifdef AF_ADAPT
        for (int a=0;a<3;a++) { double tf = g[a] - (double)Mi*eta[n*3+a]; ff += tf*tf; }      // 탄성·접촉 힘 몫 |T f|² (절 AN.14 기준 척도)
#endif
        float M[9], TB[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += T[a*3+k]*B[n*9+k*3+b]; TB[a*3+b] = t; }
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = (a==b) ? Mi : 0.f; for (int k=0;k<3;k++) t += TB[a*3+k]*T[k*3+b]; M[a*3+b] = t; }
        float Di[9]; inv3(M, Di);
        for (int k=0;k<9;k++) Dinv[n*9+k] = Di[k];
        float rr[3]; for (int a=0;a<3;a++) { rr[a] = (float)(-g[a]); r[n*3+a] = rr[a]; r0[n*3+a] = rr[a]; dl[n*3+a] = 0.f; gg += g[a]*g[a]; }
        float zz[3]; for (int a=0;a<3;a++) { zz[a] = Di[a*3]*rr[0] + Di[a*3+1]*rr[1] + Di[a*3+2]*rr[2]; z[n*3+a] = zz[a]; p[n*3+a] = zz[a]; rz += (double)rr[a]*zz[a]; }
        float tp[3]; Tapply(sk[n], pm[n], n0, n1, n2, zz, tp); for (int a=0;a<3;a++) Tp[n*3+a] = tp[a];
    }
    blocksum_add(rz, &SC[0]); blocksum_add(gg, &SC[1]);
#ifdef AF_ADAPT
    blocksum_add(ff, &SC[56]);
#endif
}
// Ap = Mi p + T (K T p) 의 노드 마무리 + p·Ap (SC[2])
__global__ void rt_ap(const int* IP, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* p, const float* KTp, float* Ap, double* SC, const float* Kb) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s = 0.0;
    if (n < IP[9]) {
        float kt[3] = {KTp[n*3], KTp[n*3+1], KTp[n*3+2]};
        if (Kb) { float K = Kb[n];                                                // 절 AN.22 --mgbar: 장벽 덩어리 곱 K n (n·T p) 융합 (rt_bar_mv 와 같은 식)
            if (K != 0.f) { float pp[3] = {p[n*3], p[n*3+1], p[n*3+2]}, tp[3]; Tapply(sk[n], pm[n], n0, n1, n2, pp, tp); float c = K*(tp[0]*n0 + tp[1]*n1 + tp[2]*n2); kt[0] += c*n0; kt[1] += c*n1; kt[2] += c*n2; } }
        float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, kt, t);
        float Mi = mn[n]*itau2;
        for (int a=0;a<3;a++) { float v = Mi*p[n*3+a] + t[a]; Ap[n*3+a] = v; s += (double)p[n*3+a]*v; }
    }
    blocksum_add(s, &SC[2]);
}
// α = rz/pAp; dl += α p; r −= α Ap; zo = z; z = D r; SC[3] += r·z, SC[4] += r·(z − zo)
__global__ void rt_cg1(const int* IP, const double* SC, const float* Dinv, const float* p, const float* Ap, float* dl, float* r, float* z, double* SC2) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s3 = 0.0, s4 = 0.0, s5 = 0.0;
    if (n < IP[9]) {
        double pAp = SC[2]; float al = (pAp > 0.0) ? (float)(SC[0]/pAp) : 0.f;
        float rr[3];
        for (int a=0;a<3;a++) { dl[n*3+a] += al*p[n*3+a]; rr[a] = r[n*3+a] - al*Ap[n*3+a]; r[n*3+a] = rr[a]; }
        const float* D = Dinv + n*9;
        for (int a=0;a<3;a++) { float zn = D[a*3]*rr[0] + D[a*3+1]*rr[1] + D[a*3+2]*rr[2]; float zo = z[n*3+a]; z[n*3+a] = zn; s3 += (double)rr[a]*zn; s4 += (double)rr[a]*(zn - zo); s5 += (double)rr[a]*rr[a]; }
    }
    blocksum_add(s3, &SC2[3]); blocksum_add(s4, &SC2[4]);
#ifdef AF_ADAPT
    blocksum_add(s5, &SC2[57]);
#endif
}
// β = SC4/SC0 (폴라크-리비에르); p = z + β p; Tp = T p; 다음을 위해 SC0 ← SC3 (단일 스레드 쪽에서)
__global__ void rt_cg2(const int* IP, const double* SC, const float* sk, const int* pm, float n0, float n1, float n2, const float* z, float* p, float* Tp, float* KTz) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (KTz) { KTz[n*3] = 0.f; KTz[n*3+1] = 0.f; KTz[n*3+2] = 0.f; }                // 다음 헤시안 곱의 원자 합 바탕 (0 채우기 커널 융합)
    float be = (SC[0] > 0.0) ? (float)(SC[4]/SC[0]) : 0.f;
    float pp[3]; for (int a=0;a<3;a++) { pp[a] = z[n*3+a] + be*p[n*3+a]; p[n*3+a] = pp[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, pp, t); for (int a=0;a<3;a++) Tp[n*3+a] = t[a];
}
__global__ void rt_sc_roll(double* SC) { SC[0] = SC[3]; SC[2] = 0.0; SC[3] = 0.0; SC[4] = 0.0; }
__global__ void rt_sc_clear(double* SC, int a, int b) { for (int i=a;i<b;i++) SC[i] = 0.0; }
// 선탐색: 후보 c 의 y_c = ŷ + T (η + c δ) 와 관성 몫 ½ Σ Mi |η + c δ|² (SC[slot]), 그리고 g·δ (SC[9])
__global__ void rt_lsy(const int* IP, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* eta, const float* dl,
                       float c, float* yc, double* SC, int slot, const float* r) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s = 0.0, gd = 0.0;
    if (n < IP[9]) {
        float e[3]; for (int a=0;a<3;a++) e[a] = eta[n*3+a] + c*dl[n*3+a];
        float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
        float Mi = mn[n]*itau2;
        for (int a=0;a<3;a++) { yc[n*3+a] = yh[n*3+a] + t[a]; s += 0.5*(double)Mi*e[a]*e[a]; if (r) gd += -(double)r[n*3+a]*dl[n*3+a]; }
    }
    blocksum_add(s, &SC[slot]);
    if (r) blocksum_add(gd, &SC[9]);
}
// 보폭 고르기: SC[10..13] = E 탄성 (c = 0, ¼, ½, 1), SC[5..8] = 관성 몫; 받아들여지는 가장 큰 c -> SC[14]
__global__ void rt_lspick(double* SC) {
    double E0 = SC[5] + SC[10]; double gd = SC[9];
    const double cs[3] = {0.25, 0.5, 1.0};
    double st = 0.0;
    for (int i=0;i<3;i++) { double E1 = SC[6+i] + SC[11+i]; if (isfinite(E1) && E1 <= E0 + 1e-4*cs[i]*gd) st = cs[i]; }
    SC[14] = st;
}
__global__ void rt_lsup(const int* IP, const double* SC, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, float* eta, const float* dl, float* y, int full) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float st = full ? 1.f : (float)SC[14];
    float e[3]; for (int a=0;a<3;a++) { e[a] = eta[n*3+a] + st*dl[n*3+a]; eta[n*3+a] = e[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y[n*3+a] = yh[n*3+a] + t[a];
}
// 걸음 끝: U, v (중점·후퇴 오일러), 고정 노드 v = vbc, 감쇠
__global__ void rt_stepend(const int* IP, int mid, float tau, float al, const float* y, const float* z0, float* U, float* vk, const unsigned char* fixd, const float* vbc, float damp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) {
        float yy = y[n*3+a], u = U[n*3+a];
        float v1 = mid ? al*(yy - z0[n*3+a]) : (yy - u)/tau;
        if (fixd && fixd[n]) v1 = vbc[n*3+a];
        U[n*3+a] = mid ? 2.f*yy - u : yy;
        vk[n*3+a] = v1*damp;
    }
}
// G2P (제자리): x += Σ w U, v = Σ w v, F ← (I + Σ U ⊗ ∇w) F
__global__ void rt_g2p(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* U, const float* vk, float* x, float* v, float* F) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float u[3] = {0,0,0}, vv[3] = {0,0,0}, A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k];
        float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
        for (int a=0;a<3;a++) { float uu = U[nid*3+a]; u[a] += ww*uu; vv[a] += ww*vk[nid*3+a]; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }
    }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*Fp[k*3+b]; F[(size_t)p*9+a*3+b] = t; }
}
// 고정 노드 (ficus 등 cuboid 구속): 상자 안 노드 -> fixd, vbc (프레임 시각 기준, 이동 상자)
__global__ void rt_bc_box(const int* IP, const float* Am, const float* BP, int ib, unsigned char* fixd, float* vbc) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    const float* b = BP + ib*10;                                              // 중심 3, 반 크기 3, 속도 3, 켜짐 1 (호스트가 프레임마다 채움)
    if (b[9] == 0.f) return;
    float c0 = b[0], c1 = b[1], c2 = b[2], s0 = b[3], s1 = b[4], s2 = b[5], v0 = b[6], v1 = b[7], v2 = b[8];
    int nb1 = IP[7], nb2 = IP[8];
    int i = n / (nb1*nb2), j = (n / nb2) % nb1, k = n % nb2;
    float yq[3] = {(float)(IP[0] + i), (float)(IP[1] + j), (float)(IP[2] + k)};
    float xp[3]; for (int a=0;a<3;a++) xp[a] = Am[a*3]*yq[0] + Am[a*3+1]*yq[1] + Am[a*3+2]*yq[2];
    bool in = fabsf(xp[0]-c0) < s0 && fabsf(xp[1]-c1) < s1 && fabsf(xp[2]-c2) < s2;
    if (in) { fixd[n] = 1; vbc[n*3] = v0; vbc[n*3+1] = v1; vbc[n*3+2] = v2; }
}
// 선탐색 (융합): 후보 c = ¼, ½, 1 의 탄성 에너지를 한 순회로 (y_c = ŷ + T (η + c δ) 를 모으며 계산) -> SC[11..13]
__global__ void __launch_bounds__(128) rt_elem_ls(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* yh, const float* eta, const float* dl,
                        const float* sk, const int* pm, float n0, float n1, float n2, float mu, float lam, double* SC) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    const float cs[3] = {0.25f, 0.5f, 1.f};
    float X[3][9]; for (int c=0;c<3;c++) for (int k=0;k<9;k++) X[c][k] = F[k];
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int c=0;c<3;c++) {
            float e[3]; for (int a=0;a<3;a++) e[a] = eta[id*3+a] + cs[c]*dl[id*3+a];
            float t[3]; Tapply(sk[id], pm[id], n0, n1, n2, e, t);
            for (int a=0;a<3;a++) { float ya = yh[id*3+a] + t[a]; X[c][a*3] += ya*g[0]; X[c][a*3+1] += ya*g[1]; X[c][a*3+2] += ya*g[2]; } } }
    float vq = on ? Vq[qq] : 0.f;
    for (int c=0;c<3;c++) {
        float U[9], S[3], V[9]; ssvdf(X[c], U, S, V);
        double J = (double)S[0]*S[1]*S[2];
#ifdef AF_SNH
        double e = on ? snh_e((double)vq, S, mu, lam) : 0.0;                    // 절 CM18 SNH
#else
        double e = on ? (double)vq*((double)mu*(((double)S[0]-1.0)*((double)S[0]-1.0) + ((double)S[1]-1.0)*((double)S[1]-1.0) + ((double)S[2]-1.0)*((double)S[2]-1.0)) + 0.5*(double)lam*(J-1.0)*(J-1.0)) : 0.0;
#endif
        blocksum_add(e, &SC[11+c]);
    }
}
__global__ void rt_lsy4(const int* IP, const float* mn, float itau2, const float* eta, const float* dl, const float* r0, double* SC) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s[4] = {0,0,0,0}, gd = 0.0;
    const float cs[4] = {0.f, 0.25f, 0.5f, 1.f};
    if (n < IP[9]) { float Mi = mn[n]*itau2;
        for (int a=0;a<3;a++) { float e0 = eta[n*3+a], d = dl[n*3+a]; for (int c=0;c<4;c++) { float e = e0 + cs[c]*d; s[c] += 0.5*(double)Mi*e*e; } gd += -(double)r0[n*3+a]*d; } }
    for (int c=0;c<4;c++) blocksum_add(s[c], &SC[5+c]);
    blocksum_add(gd, &SC[9]);
}
// G2P + 마지막 걸음 노드 갱신 (융합): 노드의 U_1 = 2y − U (중점) 또는 y, v_1 = α (y − z0) 또는 (y − U)/τ, 고정 노드 v = vbc, 감쇠
__global__ void rt_g2p_end(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* y, const float* z0, const float* U, int mid, float tau, float al,
                           const unsigned char* fixd, const float* vbc, float damp, float* x, float* v, float* F) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float u[3] = {0,0,0}, vv[3] = {0,0,0}, A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k];
        float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
        bool fx_ = fixd && fixd[nid];
        for (int a=0;a<3;a++) {
            float yy = y[nid*3+a], uk = U[nid*3+a];
            float uu = mid ? 2.f*yy - uk : yy;
            float v1 = fx_ ? vbc[nid*3+a] : (mid ? al*(yy - z0[nid*3+a]) : (yy - uk)/tau);
            u[a] += ww*uu; vv[a] += ww*v1*damp; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }
    }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*Fp[k*3+b]; F[(size_t)p*9+a*3+b] = t; }
}
// ================= 멀티그리드 (거친 층 = 조밀 상자, 프레임당 한 번 구성) =================
// LM[l*8 + ..]: 0..2 노드 상자 차원, 3 노드 수, 4..6 칸 상자 차원, 7 가우스 점 수 (= 칸 x 8). 층 0 = 고운 층 (IP).
__global__ void rt_mg_dims(int* IP, int* LM, int nl, const int* capn, const int* capq) {
    int n0 = IP[6], n1 = IP[7], n2 = IP[8];
    LM[0] = n0; LM[1] = n1; LM[2] = n2; LM[3] = IP[9];
    int ovf = 0;
    for (int l=1; l<nl; l++) {
        n0 = n0/2 + 3; n1 = n1/2 + 3; n2 = n2/2 + 3;
        int* o = LM + l*8; o[0] = n0; o[1] = n1; o[2] = n2; o[3] = n0*n1*n2; o[4] = n0-2; o[5] = n1-2; o[6] = n2-2; o[7] = (n0-2)*(n1-2)*(n2-2)*8;
        if (o[3] > capn[l] || o[7] > capq[l]) ovf = 1;
    }
    // 절 CM7: 넘친 프레임 (앞 단계 IP[13] 또는 거친 층) 이면 모든 층 차원·수를 0 -- 다시 할 프레임이라 값은 버림.
    // 전에는 넘친 층의 수만 0 이라 (층 0 은 IP[9] = 0 이어도 차원 IP[6..8] 은 큰 상자 그대로) 제한 rt_mg_restrict*·ao_mg_acc_det 가
    // 차원만 보고 용량 밖 칸 번호 (cidx·mn·r·bKn) 를 읽어 불법 접근 (wolf K 8 프레임 40 등).
#ifndef AF_MGDIMS_OLD
    if (ovf || IP[13]) { for (int l=0; l<nl; l++) for (int k=0;k<8;k++) LM[l*8+k] = 0; if (ovf) atomicOr(&IP[13], 2); }
#else
    for (int l=1; l<nl; l++) { int* o = LM + l*8; if (o[3] > capn[l] || o[7] > capq[l]) { o[3] = 0; o[7] = 0; atomicOr(&IP[13], 2); } }   // 옛 판 (진단 대조 전용)
#endif
}
// 거친 층 l 가우스 점 누적: 출처 = 층 l−1 의 가우스 점 (위치 tq = 칸 좌표 + 팔분 위치, X = F, 가중 V)
__global__ void rt_mg_acc(int l, const int* IP, const int* LM, const int* cbF, const float* Fsrc, const float* Vsrc, float* A, float* C, int* badq, int* badc) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    int nsrc = (l == 1) ? IP[12] : LM[(l-1)*8+7];
#ifndef AF_MGDIMS_OLD
    if (q >= nsrc || LM[l*8+7] == 0) return;                                          // 절 CM7: 넘친 프레임 (층 차원 0) 이면 쓰지 않음
#else
    if (q >= nsrc) return;
#endif
    float w = Vsrc[q]; if (!(w > 0.f)) return;
    int c = q >> 3, oc = q & 7;
    const float GG = 0.28867513459481287f;
    int cx, cy, cz;
    if (l == 1) { cx = cbF[c*3]; cy = cbF[c*3+1]; cz = cbF[c*3+2]; }
    else { const int* pl = LM + (l-1)*8; cz = c % pl[6]; cy = (c / pl[6]) % pl[5]; cx = c / (pl[5]*pl[6]); }
    float tq[3] = {cx + 1.f + ((oc & 4) ? GG : -GG), cy + 1.f + ((oc & 2) ? GG : -GG), cz + 1.f + ((oc & 1) ? GG : -GG)};
    const int* L = LM + l*8;
    int b[3]; float fx[3];
    for (int a=0;a<3;a++) { float t = 0.5f*(tq[a] + 1.5f); float fb = floorf(t - 0.5f); b[a] = (int)fb; fx[a] = t - fb; }
    int cc = (b[0]*L[5] + b[1])*L[6] + b[2];
    int qc = cc*8 + (fx[0] >= 1.f ? 4 : 0) + (fx[1] >= 1.f ? 2 : 0) + (fx[2] >= 1.f ? 1 : 0);
    float X[9]; for (int k=0;k<9;k++) X[k] = Fsrc[(size_t)q*9+k];
    float U[9], S[3], V[9]; ssvdf(X, U, S, V);
    bool ng = !(S[2] > 0.f);
    float R[9], Lg[9]; float lg[3]; for (int i=0;i<3;i++) lg[i] = ng ? 0.f : logf(S[i]);
    for (int a=0;a<3;a++) for (int bb=0;bb<3;bb++) { R[a*3+bb] = U[a*3]*V[bb*3] + U[a*3+1]*V[bb*3+1] + U[a*3+2]*V[bb*3+2];
        Lg[a*3+bb] = V[a*3]*lg[0]*V[bb*3] + V[a*3+1]*lg[1]*V[bb*3+1] + V[a*3+2]*lg[2]*V[bb*3+2]; }
    float* aq = A + (size_t)qc*28; float* ac = C + (size_t)cc*30;
    for (int k=0;k<9;k++) { atomicAdd(&aq[k], w*R[k]); atomicAdd(&aq[9+k], w*Lg[k]); atomicAdd(&aq[18+k], w*X[k]);
                            atomicAdd(&ac[k], w*R[k]); atomicAdd(&ac[9+k], w*Lg[k]); atomicAdd(&ac[18+k], w*X[k]); }
    atomicAdd(&aq[27], w); atomicAdd(&ac[27], w); atomicAdd(&ac[28], w);
    if (ng) { atomicOr(&badq[qc], 1); atomicOr(&badc[cc], 1); }
}
__global__ void rt_mg_fin(int l, const int* LM, const float* A, const float* C, const int* badq, const int* badc, float* Fq, float* Vq) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= LM[l*8+7]) return;
    int c = q >> 3;
    const float* aq = A + (size_t)q*28; const float* ac = C + (size_t)c*30;
    float out[9]; bool bad = false;
    if (!(ac[27] > 0.f)) { for (int k=0;k<9;k++) out[k] = (k % 4 == 0) ? 1.f : 0.f; Vq[q] = 0.f; }
    else {
        if (aq[27] > 0.f) { rs_fin(aq, aq + 9, aq[27], out, &bad); if (bad || badq[q]) for (int k=0;k<9;k++) out[k] = aq[18+k]/aq[27]; }
        else { rs_fin(ac, ac + 9, ac[27], out, &bad); if (bad || badc[c]) for (int k=0;k<9;k++) out[k] = ac[18+k]/ac[27]; }
        Vq[q] = ac[28]*0.125f;
    }
    for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = out[k];
}
__device__ __forceinline__ void qnodes_d(int q, const int* L, int* ids) {
    int c = q >> 3; int cz = c % L[6], cy = (c / L[6]) % L[5], cx = c / (L[5]*L[6]);
    for (int o=0;o<27;o++) { int i = o/9, j = (o/3)%3, k = o%3; ids[o] = ((cx+i)*L[1] + (cy+j))*L[2] + (cz+k); }
}
// 거친 층 헤시안 (y = 0 -> X = F_q) 성분 우선 저장 + 노드 대각 융합. BG 기울기에 sc = 2^-l
__global__ void __launch_bounds__(128) rt_mg_elem(int l, const int* LM, const float* BG, float sc, const float* Fq, const float* Vq, float mu, float lam, float* H, int capQ, float* B) {
    const int* L = LM + l*8;
    if (blockIdx.x*blockDim.x >= L[7]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < L[7]; int qq = on ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes_d(qq, L, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    if (__all_sync(0xffffffffu, !(vq > 0.f))) return;
    float h[81]; for (int k=0;k<81;k++) h[k] = 0.f;
    if (on && vq > 0.f) {
        float U[9], S[3], V[9]; ssvdf(F, U, S, V);
        float J = S[0]*S[1]*S[2]; float gJ[3] = {S[1]*S[2], S[0]*S[2], S[0]*S[1]};
        float D[3][3], Ev[3][3];
        float Sg[3][3] = {{0.f, S[2], S[1]}, {S[2], 0.f, S[0]}, {S[1], S[0], 0.f}};
#ifdef AF_SNH
        snh_D(S, mu, lam, D);                                                      // 절 CM18 SNH: 주값 헤시안 (투영 전)
#else
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = ((i==j) ? 2.f*mu : 0.f) + lam*gJ[i]*gJ[j] + lam*(J-1.f)*Sg[i][j];
#endif
        jac3f(D, Ev);
        for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (!(dm > 0.f)) continue;
            float e[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) e[a*3+b] = Ev[0][m]*U[a*3]*V[b*3] + Ev[1][m]*U[a*3+1]*V[b*3+1] + Ev[2][m]*U[a*3+2]*V[b*3+2];
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += dm*e[r]*e[c]; }
        const int PI[3] = {0,0,1}, PJ[3] = {1,2,2}, PK[3] = {2,1,0};
        const float r2 = 0.70710678f;
        for (int pp=0; pp<3; pp++) { int i = PI[pp], j = PJ[pp], k = PK[pp];
#ifdef AF_SNH
            float lt, lf; snh_tf(S, k, mu, lam, &lt, &lf);                             // 절 CM18 SNH: 비틀림·뒤집힘 고윳값 (나눗셈 없음)
#else
            float lt = 2.f*mu*(1.f - 2.f/(S[i]+S[j])) + lam*(J-1.f)*S[k]; float lf = 2.f*mu - lam*(J-1.f)*S[k];
#endif
            lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = fmaxf(lf, 0.f);
            float et[9], ef[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float x1 = U[a*3+i]*V[b*3+j], x2 = U[a*3+j]*V[b*3+i]; et[a*3+b] = r2*(x1-x2); ef[a*3+b] = r2*(x1+x2); }
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += lt*et[r]*et[c] + lf*ef[r]*ef[c]; }
#ifdef AF_HSPEC
        {                                                                         // 절 CM3 --hspec: 거친 층도 주값 분해 30 성분 (U, M = F V, C x V_q) -- rt_mg_mv2 (mv2_core) 가 읽음
            float Cs[12];
            const int RI[6] = {0,1,2,0,0,1}, RJ[6] = {0,1,2,1,2,2};
            for (int k=0;k<6;k++) { float t = 0.f; for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (dm > 0.f) t += dm*Ev[RI[k]][m]*Ev[RJ[k]][m]; } Cs[k] = vq*t; }
            for (int pp=0; pp<3; pp++) { int i = PI[pp], j = PJ[pp], k = PK[pp];
#ifdef AF_SNH
                float lt, lf; snh_tf(S, k, mu, lam, &lt, &lf);                             // 절 CM18 SNH: 비틀림·뒤집힘 고윳값 (나눗셈 없음)
#else
                float lt = 2.f*mu*(1.f - 2.f/(S[i]+S[j])) + lam*(J-1.f)*S[k]; float lf = 2.f*mu - lam*(J-1.f)*S[k];
#endif
                lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = fmaxf(lf, 0.f);
                Cs[6+pp] = vq*0.5f*(lt + lf); Cs[9+pp] = vq*0.5f*(lf - lt); }
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) H[(size_t)(9 + a*3+b)*capQ + qq] = F[a*3]*V[b] + F[a*3+1]*V[3+b] + F[a*3+2]*V[6+b];
            for (int k=0;k<9;k++) H[(size_t)k*capQ + qq] = U[k];
            for (int k=0;k<12;k++) H[(size_t)(18+k)*capQ + qq] = Cs[k];
        }
#endif
    }
#ifndef AF_HSPEC
    if (on) for (int k=0;k<81;k++) H[(size_t)k*capQ + qq] = h[k];
#endif
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; float bs[3] = {sc*bg[0], sc*bg[1], sc*bg[2]}; float g[3]; gqn(F, bs, g);
        float m[9];
        for (int a=0;a<3;a++) for (int c=0;c<3;c++) { float t = 0.f;
            for (int b=0;b<3;b++) { float hg = h[(a*3+b)*9 + c*3]*g[0] + h[(a*3+b)*9 + c*3+1]*g[1] + h[(a*3+b)*9 + c*3+2]*g[2]; t += g[b]*hg; }
            m[a*3+c] = vq*t; }
        for (int k=0;k<9;k++) m[k] = seg8f(m[k]);
        if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], m[k]); }
}
// 거친 층 헤시안 곱 (조밀)
__global__ void __launch_bounds__(128) rt_mg_mv(int l, const int* LM, const float* BG, float sc, const float* Fq, const float* Vq, const float* H, int capQ, const float* dy, float* out) {
    const int* L = LM + l*8;
    if (blockIdx.x*blockDim.x >= L[7]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < L[7]; int qq = on ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes_d(qq, L, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    if (__all_sync(0xffffffffu, !(vq > 0.f))) return;                           // 빈 칸 (부피 0) 만 든 워프는 기여 0
    float dX[9] = {0,0,0,0,0,0,0,0,0};
    float gs[27][3];
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; float bs[3] = {sc*bg[0], sc*bg[1], sc*bg[2]}; gqn(F, bs, gs[o]); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = dy[id*3+a]; dX[a*3] += ya*gs[o][0]; dX[a*3+1] += ya*gs[o][1]; dX[a*3+2] += ya*gs[o][2]; } }
    float dP[9]; for (int r=0;r<9;r++) { float t = 0.f; if (vq > 0.f) for (int c=0;c<9;c++) t += H[(size_t)(r*9+c)*capQ + qq]*dX[c]; dP[r] = vq*t; }
    for (int o=0;o<27;o++) for (int a=0;a<3;a++) { float s = seg8f(dP[a*3]*gs[o][0] + dP[a*3+1]*gs[o][1] + dP[a*3+2]*gs[o][2]); if (on && head) atomicAdd(&out[ids[o]*3+a], s); }
}
// 제한 out[J] = Σ c_k c_l c_m in[2J − 3 + (k,l,m)] (c = 1,3,3,1 / 4), 고운 층 차원 nf (LM 층 lf), 거친 층 차원 (LM 층 lf+1). ch = 1 또는 3
__global__ void rt_mg_restrict(int lf, const int* LM, const float* in, float* out, int ch) {
    const int* A = LM + lf*8; const int* B = LM + (lf+1)*8;
    int J = blockIdx.x*blockDim.x + threadIdx.x;
    if (J >= B[3]) return;
    int jz = J % B[2], jy = (J / B[2]) % B[1], jx = J / (B[1]*B[2]);
    const float cw[4] = {0.25f, 0.75f, 0.75f, 0.25f};
    float acc[3] = {0.f, 0.f, 0.f};
    for (int k=0;k<4;k++) { int ix = 2*jx - 3 + k; if (ix < 0 || ix >= A[0]) continue;
      for (int m=0;m<4;m++) { int iy = 2*jy - 3 + m; if (iy < 0 || iy >= A[1]) continue;
        for (int n=0;n<4;n++) { int iz = 2*jz - 3 + n; if (iz < 0 || iz >= A[2]) continue;
          float w = cw[k]*cw[m]*cw[n]; int i = (ix*A[1] + iy)*A[2] + iz;
          for (int c=0;c<ch;c++) acc[c] += w*in[i*ch+c]; } } }
    for (int c=0;c<ch;c++) out[J*ch+c] = acc[c];
}
// 연장 out[i] (+)= Σ_J c_{i−2J+3} in[J]
__global__ void rt_mg_prolong(int lf, const int* LM, const float* in, float* out, int add) {
    const int* A = LM + lf*8; const int* B = LM + (lf+1)*8;
    int I = blockIdx.x*blockDim.x + threadIdx.x;
    if (I >= A[3]) return;
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
    for (int c=0;c<3;c++) out[I*3+c] = (add ? out[I*3+c] : 0.f) + acc[c];
}
// 층 l 배열 0 채우기 (실제 크기만): which 0 = 노드 수, 1 = 가우스 점 수, 2 = 칸 수
__global__ void rt_zero_lm(int l, const int* LM, int which, int mul, float* a, int* ai) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int n = (which == 0) ? LM[l*8+3] : ((which == 1) ? LM[l*8+7] : LM[l*8+7]/8);
    if (i >= n*mul) return;
    if (a) a[i] = 0.f; else ai[i] = 0;
}
// 거친 층 노드: D = (Mi I + B)⁻¹ (질량·대각 0 이면 0)
__global__ void rt_mg_D(int l, const int* LM, const float* mc, float itau2, const float* B, float* D) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= LM[l*8+3]) return;
    float M[9]; float Mi = mc[n]*itau2;
    for (int k=0;k<9;k++) M[k] = B[n*9+k] + ((k % 4 == 0) ? Mi : 0.f);
    float Di[9]; inv3(M, Di); for (int k=0;k<9;k++) D[n*9+k] = Di[k];
}
// 층 노드 연산: (0) z = w D r; (1) z += w D (r − Az); (2) res = r − Az;  Az = Mi z + KTz 마무리 (층 0 은 T 끼움)
// 층 0: Az = Mi z + T (K (T z)); 여기서 KTz 는 이미 계산된 K(Tz) 또는 K z. 모드 3: Tz = T z (층 0), 모드 4: out = T v
__global__ void rt_mg_node(int l, const int* LM, int mode, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2,
                           const float* D, const double* W, const float* r, float* KTz, float* z, float* res, const float* Kb) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= LM[l*8+3]) return;
    float w = (float)W[l];
    const float* d = D + n*9;
    float rr[3], az[3];
    for (int a=0;a<3;a++) rr[a] = r ? r[n*3+a] : 0.f;
    if (mode == 0 || mode == 6) { float zz[3]; for (int a=0;a<3;a++) { zz[a] = w*(d[a*3]*rr[0] + d[a*3+1]*rr[1] + d[a*3+2]*rr[2]); z[n*3+a] = zz[a]; }
        if (mode == 6) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, zz, t); for (int a=0;a<3;a++) res[n*3+a] = t[a]; }       // 층 0: 다음 Az 의 T z 를 함께 (융합)
        return; }
    if (mode == 3 || mode == 4) { const float* v = (mode == 3) ? z : r; float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, v + n*3, t); for (int a=0;a<3;a++) res[n*3+a] = t[a]; return; }
    if (mode == 5 || mode == 9) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, r + n*3, t); for (int a=0;a<3;a++) z[n*3+a] += t[a];
        if (mode == 9) { float u[3] = {z[n*3], z[n*3+1], z[n*3+2]}; Tapply(sk[n], pm[n], n0, n1, n2, u, t); for (int a=0;a<3;a++) res[n*3+a] = t[a]; }   // 층 0: 다음 Az 의 T z (융합)
        return; }
    float Mi = mn[n]*itau2;
    float kt[3] = {KTz[n*3], KTz[n*3+1], KTz[n*3+2]};
    KTz[n*3] = 0.f; KTz[n*3+1] = 0.f; KTz[n*3+2] = 0.f;                         // 읽은 뒤 비움 (다음 곱의 원자 합 바탕; 0 채우기 커널 없앰)
    if (l == 0) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, kt, t); for (int a=0;a<3;a++) kt[a] = t[a]; }
    for (int a=0;a<3;a++) az[a] = Mi*z[n*3+a] + kt[a];
    if (Kb) { float K = Kb[n];                                                   // 절 AN.22 --mgbar: 장벽 덩어리 헤시안을 노드 연산에 융합 (층 0: T (K n nᵀ) T z, 거친 층: K n nᵀ z)
        if (K != 0.f) { float zz[3] = {z[n*3], z[n*3+1], z[n*3+2]}, tz[3]; if (l == 0) Tapply(sk[n], pm[n], n0, n1, n2, zz, tz); else { tz[0] = zz[0]; tz[1] = zz[1]; tz[2] = zz[2]; }
            float c = K*(tz[0]*n0 + tz[1]*n1 + tz[2]*n2); float bv[3] = {c*n0, c*n1, c*n2}, tb[3];
            if (l == 0) Tapply(sk[n], pm[n], n0, n1, n2, bv, tb); else { tb[0] = bv[0]; tb[1] = bv[1]; tb[2] = bv[2]; }
            for (int a=0;a<3;a++) az[a] += tb[a]; } }
    float e[3]; for (int a=0;a<3;a++) e[a] = rr[a] - az[a];
    if (mode == 2) { for (int a=0;a<3;a++) res[n*3+a] = e[a]; return; }
    if (mode == 7) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t); for (int a=0;a<3;a++) res[n*3+a] = t[a]; return; }   // 층 0: T (r − Az) (제한 입력, 융합)
    for (int a=0;a<3;a++) z[n*3+a] += w*(d[a*3]*e[0] + d[a*3+1]*e[1] + d[a*3+2]*e[2]);
}
// 거듭제곱법 한 걸음: z ← D A z / |D A z|, λ = (z·DAz)/(z·z) -> SCM[l]; 마지막에 W[l] = ω/λ
__global__ void rt_mg_pw1(int l, const int* LM, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* D, const float* z, float* KTz,
                          float* y, double* SCM) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int nn = LM[l*8+3];
    if (blockIdx.x*blockDim.x >= nn) return;
    if (n < nn) {
        float Mi = mn[n]*itau2;
        float kt[3] = {KTz[n*3], KTz[n*3+1], KTz[n*3+2]};
        KTz[n*3] = 0.f; KTz[n*3+1] = 0.f; KTz[n*3+2] = 0.f;
        if (l == 0) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, kt, t); for (int a=0;a<3;a++) kt[a] = t[a]; }
        float az[3]; for (int a=0;a<3;a++) az[a] = Mi*z[n*3+a] + kt[a];
        const float* d = D + n*9;
        for (int a=0;a<3;a++) { float v = d[a*3]*az[0] + d[a*3+1]*az[1] + d[a*3+2]*az[2]; y[n*3+a] = v; s1 += (double)z[n*3+a]*v; s2 += (double)z[n*3+a]*z[n*3+a]; s3 += (double)v*v; }
    }
    blocksum_add(s1, &SCM[l*4]); blocksum_add(s2, &SCM[l*4+1]); blocksum_add(s3, &SCM[l*4+2]);
}
__global__ void rt_mg_pw2(int l, const int* LM, const float* y, float* z, double* SCM, double* W, float omega, int last) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0 && last) { double lm = SCM[l*4] / fmax(SCM[l*4+1], 1e-300); W[l] = omega / fmax(lm, 1e-30); }
    if (n >= LM[l*8+3]) return;
    double nr = sqrt(fmax(SCM[l*4+2], 1e-300));
#ifdef AF_FK
    double inr = 1.0/nr; for (int a=0;a<3;a++) z[n*3+a] = (float)(y[n*3+a]*inr);    // 절 AN.22 --fk: f64 나눗셈 셋 -> 하나
#else
    for (int a=0;a<3;a++) z[n*3+a] = (float)(y[n*3+a] / nr);
#endif
}
__global__ void rt_mg_pw0(int l, double* SCM) { SCM[l*4] = 0.0; SCM[l*4+1] = 0.0; SCM[l*4+2] = 0.0; }
__global__ void rt_fill_rand(int l, const int* LM, float* z) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= LM[l*8+3]) return;
    unsigned int h = (unsigned int)n*2654435761u + 12345u*(l+1);
    for (int a=0;a<3;a++) { h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15; z[n*3+a] = ((float)(h & 0xffffff) / 16777216.f) - 0.5f; }
}
// PCG + MG: α 갱신만 (z 는 V 순환이 따로); 그다음 내적 r·z, r·(z − zo) 와 zo ← z
__global__ void rt_cg1m(const int* IP, const double* SC, const float* p, const float* Ap, float* dl, float* r) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    double pAp = SC[2]; float al = (pAp > 0.0) ? (float)(SC[0]/pAp) : 0.f;
    for (int a=0;a<3;a++) { dl[n*3+a] += al*p[n*3+a]; r[n*3+a] -= al*Ap[n*3+a]; }
}
__global__ void rt_dotz(const int* IP, const float* r, const float* z, float* zo, double* SC, int first) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s3 = 0.0, s4 = 0.0, s5 = 0.0;
    if (n < IP[9]) for (int a=0;a<3;a++) { float zn = z[n*3+a]; s3 += (double)r[n*3+a]*zn; s4 += (double)r[n*3+a]*(zn - zo[n*3+a]); s5 += (double)r[n*3+a]*r[n*3+a]; zo[n*3+a] = zn; }
    blocksum_add(s3, first ? &SC[0] : &SC[3]); if (!first) blocksum_add(s4, &SC[4]);
#ifdef AF_ADAPT
    if (!first) blocksum_add(s5, &SC[57]);                                         // PCG 잔차 |r|² (절 AN.14 판정)
#endif
}
__global__ void rt_pinit(const int* IP, const float* z, const float* sk, const int* pm, float n0, float n1, float n2, float* p, float* Tp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, z + n*3, t);
    for (int a=0;a<3;a++) { p[n*3+a] = z[n*3+a]; Tp[n*3+a] = t[a]; }
}
// 하강 방향 보장 (--descent): PCG 결과 δ 가 g·δ ≥ 0 이면 δ ← α0 p0 (p0 = 첫 방향 = M r0, α0 = r0·M r0 / p0·A p0). SC[16] = r0·Mr0, SC[17] = p0·Ap0, SC[18] 바꾼 횟수
__global__ void rt_fixdir(const int* IP, double* SC, const float* p0, float* dl) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    bool bad = !(SC[9] < 0.0);
    if (n == 0 && bad) SC[62] += 1.0;
    if (n == 0) SC[59] = bad ? 0.0 : 1.0;                                       // 절 AN.22: 방향을 안 바꾸면 1 (두 번째 관성 몫 계산 생략 문)
    if (n >= IP[9] || !bad) return;
    float a0 = (SC[61] > 0.0) ? (float)(SC[60]/SC[61]) : 0.f;
    for (int a=0;a<3;a++) dl[n*3+a] = a0*p0[n*3+a];
}
__global__ void rt_sc_copy(double* SC, int dst, int src) { SC[dst] = SC[src]; }
// 절 CM3 노드 압축 (--ncomp): 점유 칸 27 개 중 하나라도 이웃인 노드 (가우스 점 곱·질량·대각이 닿는 노드 전부) 목록 ACT[1..ACT[0]] (노드 번호 오름차순, 결정적).
// 그 밖 노드는 질량·대각·힘·곱이 모두 0 이라 PCG·V 순환 값이 목록 노드에 주는 몫이 0 -- 목록만 돌아도 목록 노드 값은 같다 (블록 합 순서만 다름).
__device__ __forceinline__ int rt_act_on(const int* IP, const int* occ, int n) {
    if (n >= IP[9]) return 0;
    int nb1 = IP[7], nb2 = IP[8];
    int ni = n / (nb1*nb2), nj = (n / nb2) % nb1, nk = n % nb2;
    int c0 = IP[6]-2, c1 = nb1-2, c2 = nb2-2;
    for (int i=0;i<3;i++) { int ci = ni - i; if (ci < 0 || ci >= c0) continue;
      for (int j=0;j<3;j++) { int cj = nj - j; if (cj < 0 || cj >= c1) continue;
        for (int k=0;k<3;k++) { int ck = nk - k; if (ck < 0 || ck >= c2) continue;
          if (occ[(ci*c1 + cj)*c2 + ck]) return 1; } } }
    return 0;
}
__global__ void rt_act_cnt(const int* IP, const int* occ, int* bc) {                // 블록 256 고정
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int c = __syncthreads_count(rt_act_on(IP, occ, n));
    if (threadIdx.x == 0) bc[blockIdx.x] = c;
}
__global__ void rt_act_scan(const int* IP, const int* bc, int* bo, int* ACT) {       // 한 블록 1024: 블록 수 배타 스캔
    __shared__ int sh[1024];
    int nb = (IP[9] + 255) / 256, carry = 0;
    for (int base = 0; base < nb; base += 1024) {
        int i = base + threadIdx.x;
        int v = (i < nb) ? bc[i] : 0;
        sh[threadIdx.x] = v;
        __syncthreads();
        for (int o=1; o<1024; o<<=1) { int t = (threadIdx.x >= o) ? sh[threadIdx.x - o] : 0; __syncthreads(); sh[threadIdx.x] += t; __syncthreads(); }
        if (i < nb) bo[i] = carry + sh[threadIdx.x] - v;
        int tot = sh[1023];
        __syncthreads();
        carry += tot;
    }
    if (threadIdx.x == 0) ACT[0] = carry;
}
__global__ void rt_act_wr(const int* IP, const int* occ, const int* bo, int* ACT) {  // 블록 256 고정
    __shared__ int ws[8];
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int v = rt_act_on(IP, occ, n);
    unsigned int m = __ballot_sync(0xffffffffu, v);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) ws[w] = __popc(m);
    __syncthreads();
    int off = bo[blockIdx.x];
    for (int k=0;k<w;k++) off += ws[k];
    if (v) ACT[1 + off + __popc(m & ((1u << lane) - 1u))] = n;
}
// 선탐색 넓힘 (--ls_n NC): 후보 c_i = 2^-i (i = 0..NC−1). 관성 몫 SC[20] (c = 0), SC[21+i]; 탄성 SC[40+i]; g·δ SC[9]; 보폭 SC[14]
// 절 AN.22: 후보 고리를 컴파일 시 12 로 펼쳐 sc[] 를 레지스터에 (원래는 실행 시 NC 고리라 지역 메모리 -- 같은 연산·같은 순서, 결과 비트 같음). gate != 0 이고 *gate != 0 이면 아무것도 안 함 (하강 방향을 안 바꾼 반복의 두 번째 계산 생략)
__global__ void rt_lsyN(const int* IP, const float* mn, float itau2, const float* eta, const float* dl, const float* r0, double* SC, int NC, const double* gate) {
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s0 = 0.0, gd = 0.0, sc[12];
#pragma unroll
    for (int i=0;i<12;i++) sc[i] = 0.0;
    if (n < IP[9]) { float Mi = mn[n]*itau2;
        for (int a=0;a<3;a++) { float e0 = eta[n*3+a], d = dl[n*3+a]; s0 += 0.5*(double)Mi*e0*e0; gd += -(double)r0[n*3+a]*d;
            float c = 1.f;
#pragma unroll
            for (int i=0;i<12;i++) { if (i < NC) { float e = e0 + c*d; sc[i] += 0.5*(double)Mi*e*e; } c *= 0.5f; } } }
    blocksum_add(s0, &SC[20]); blocksum_add(gd, &SC[9]);
#pragma unroll
    for (int i=0;i<12;i++) if (i < NC) blocksum_add(sc[i], &SC[21+i]);
}
// 절 AN.22 --fk (같은 식, 반올림만 다름): 관성 몫 후보를 이차식으로 -- ½ Σ Mi |η + c δ|² = S0 + c S1 + c² S2 (S0 = SC20, S1 = Σ Mi η·δ -> SC33, S2 = ½ Σ Mi |δ|² -> SC34),
// 후보마다 블록 합 12 번 대신 4 번 (f64 처리량이 낮은 A6000 에서 rt_lsyN 이 노드 커널 중 가장 비쌌음). rt_lsyfin 이 SC[21+i] 를 채움.
__global__ void rt_lsyQ(const int* IP, const float* mn, float itau2, const float* eta, const float* dl, const float* r0, double* SC, const double* gate) {
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s0 = 0.0, gd = 0.0, s1 = 0.0, s2 = 0.0;
    if (n < IP[9]) { double Mi = (double)(mn[n]*itau2);
        for (int a=0;a<3;a++) { double e0 = eta[n*3+a], d = dl[n*3+a]; s0 += 0.5*Mi*e0*e0; gd += -(double)r0[n*3+a]*d; s1 += Mi*e0*d; s2 += 0.5*Mi*d*d; } }
    blocksum_add(s0, &SC[20]); blocksum_add(gd, &SC[9]); blocksum_add(s1, &SC[33]); blocksum_add(s2, &SC[34]);
}
__global__ void rt_lsyfin(double* SC, int NC, const double* gate) {
    if (gate && *gate != 0.0) return;
    double c = 1.0; for (int i=0;i<NC;i++) { SC[21+i] = SC[20] + c*SC[33] + c*c*SC[34]; c *= 0.5; }
}
// 절 AN.22 선탐색 조기 판정 (결과 같음): 후보 c = 1 이 아르미호를 만족하면 SC[63] = 1 -> 나머지 후보 커널 (gate = SC+63) 생략. rt_lspickN 은 첫 만족 후보를 고르므로 결과가 같다.
__global__ void rt_lsgate(double* SC) {
    double E0 = SC[20] + SC[10]; double gd = SC[9]; double E1 = SC[21] + SC[40];
    SC[63] = (isfinite(E1) && E1 <= E0 + 1e-4*gd) ? 1.0 : 0.0;
}
__global__ void rt_sc_clear_g(double* SC, int a, int b, const double* gate) { if (*gate != 0.0) return; for (int i=a;i<b;i++) SC[i] = 0.0; }
__global__ void __launch_bounds__(128) rt_elem_lsN(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* yh, const float* eta, const float* dl,
                        const float* sk, const int* pm, float n0, float n1, float n2, float mu, float lam, double* SC, int NC, int i0, const double* gate) {
    if (gate && *gate != 0.0) return;                                           // 절 AN.22: 후보 c = 1 이 받아들여졌으면 나머지 후보 생략
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    float c = 1.f; for (int i=0;i<i0;i++) c *= 0.5f;
    for (int i=i0;i<NC;i++) {
        float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
        for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
            float e[3]; for (int a=0;a<3;a++) e[a] = eta[id*3+a] + c*dl[id*3+a];
            float t[3]; Tapply(sk[id], pm[id], n0, n1, n2, e, t);
            for (int a=0;a<3;a++) { float ya = yh[id*3+a] + t[a]; X[a*3] += ya*g[0]; X[a*3+1] += ya*g[1]; X[a*3+2] += ya*g[2]; } }
        float U[9], S[3], V[9]; ssvdf(X, U, S, V);
        double J = (double)S[0]*S[1]*S[2];
#ifdef AF_SNH
        double en = on ? snh_e((double)vq, S, mu, lam) : 0.0;                   // 절 CM18 SNH
#else
        double en = on ? (double)vq*((double)mu*(((double)S[0]-1.0)*((double)S[0]-1.0) + ((double)S[1]-1.0)*((double)S[1]-1.0) + ((double)S[2]-1.0)*((double)S[2]-1.0)) + 0.5*(double)lam*(J-1.0)*(J-1.0)) : 0.0;
#endif
        blocksum_add(en, &SC[40+i]);
        c *= 0.5f;
    }
}
__global__ void rt_lspickN(double* SC, int NC) {
    double E0 = SC[20] + SC[10]; double gd = SC[9];
    double st = 0.0, c = 1.0;
    for (int i=0;i<NC;i++) { double E1 = SC[21+i] + SC[40+i]; if (st == 0.0 && isfinite(E1) && E1 <= E0 + 1e-4*c*gd) st = c; c *= 0.5; }
    SC[14] = st;
}
// 절 AN.22 시작점 탐색 (--y0ls, 결과 정의 같음 -- 풀리는 증분 포텐셜·장벽 활성 판정 (걸음 시작 U) 그대로, 뉴턴 시작점만):
// rt_floor_n 뒤 η = η(Ũ) (관성 예측) 를 dl 에 둔 뒤 rt_ga_y0 가 η = η(U) 로 바꾸면 dl ← η(Ũ) − η(U). 선탐색과 같은 후보 커널로 c = 2^-i (i = 0..NC−1) 의 전체 에너지 (관성 + 탄성 + 장벽) -> 유한한 것 중 가장 작은 c -> SC[14].
__global__ void rt_y0dir(const int* IP, const float* eta, float* dl) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) dl[n*3+a] -= eta[n*3+a];
}
__global__ void rt_y0pick(double* SC, int NC) {
    double best = 0.0, st = 0.0, c = 1.0; bool have = false;
    for (int i=0;i<NC;i++) { double E1 = SC[21+i] + SC[40+i]; if (isfinite(E1) && E1 < 1e29 && (!have || E1 < best)) { best = E1; st = c; have = true; } c *= 0.5; }
    SC[14] = st;
}
// ---- 단계 일반화 (θ 법·TR-BDF2) ----
// 단계 시작: Ut = Ub + τ vb + τ² g, z0 = Ub + β τ vb, y = Ut (고정 노드 y = Uf + dtf vbc). θ 단계: Ub = U, vb = v, β = 1 − θ; BDF2 단계: Ub = w, vb = ŵ, β = 0.
__global__ void rt_step0g(const int* IP, const float* Ub, const float* vb, float tau, float beta, float g0, float g1, float g2, const unsigned char* fixd, const float* vbc,
                          const float* Uf, float dtf, float* Ut, float* z0, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float gg[3] = {g0, g1, g2};
    for (int a=0;a<3;a++) {
        float u = Ub[n*3+a], vv = vb[n*3+a];
        float ut = u + tau*vv + tau*tau*gg[a];
        Ut[n*3+a] = ut; z0[n*3+a] = u + beta*tau*vv;
        y[n*3+a] = (fixd && fixd[n]) ? Uf[n*3+a] + dtf*vbc[n*3+a] : ut;
    }
}
// 단계 끝: U ← U + thinv (y − U) (θ 단계 thinv = 1/θ, BDF2 1), v ← α (y − z0) (고정 노드 vbc), 감쇠
__global__ void rt_stepend_g(const int* IP, float thinv, float al, const float* y, const float* z0, float* U, float* vk, const unsigned char* fixd, const float* vbc, float damp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) {
        float yy = y[n*3+a], u = U[n*3+a];
        float v1 = al*(yy - z0[n*3+a]);
        if (fixd && fixd[n]) v1 = vbc[n*3+a];
        U[n*3+a] = u + thinv*(yy - u);
        vk[n*3+a] = v1*damp;
    }
}
// BDF2 기저: w = c1 Uγ − c0 U0, ŵ = c1 vγ − c0 v0
__global__ void rt_bdf_base(const int* IP, float c1, float c0, const float* Ug, const float* vg, const float* U0, const float* v0, float* w, float* wv) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) { w[n*3+a] = c1*Ug[n*3+a] - c0*U0[n*3+a]; wv[n*3+a] = c1*vg[n*3+a] - c0*v0[n*3+a]; }
}
// 관성 질량 하한: m' = max(m, cs tr(B)/3) (cs = c τ², B = 걸음 첫 뉴턴의 노드 탄성 대각 블록), cs = 0 이면 m 그대로
__global__ void rt_meff(const int* IP, const float* mn, const float* B, float cs, float* meff) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n];
    meff[n] = (cs > 0.f) ? fmaxf(m, cs*(B[n*9] + B[n*9+4] + B[n*9+8])*(1.f/3.f)) : m;
}
// G2P + 마지막 단계 끝 (일반): uu = U + thinv (y − U), v1 = α (y − z0) 또는 vbc
__global__ void rt_g2p_endg(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* y, const float* z0, const float* U, float thinv, float al,
                           const unsigned char* fixd, const float* vbc, float damp, float* x, float* v, float* F) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float u[3] = {0,0,0}, vv[3] = {0,0,0}, A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k];
        float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
        bool fx_ = fixd && fixd[nid];
        for (int a=0;a<3;a++) {
            float yy = y[nid*3+a], uk = U[nid*3+a];
            float uu = uk + thinv*(yy - uk);
            float v1 = fx_ ? vbc[nid*3+a] : al*(yy - z0[nid*3+a]);
            u[a] += ww*uu; vv[a] += ww*v1*damp; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }
    }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*Fp[k*3+b]; F[(size_t)p*9+a*3+b] = t; }
}
// 접촉 적응: 고착 노드 수 -> IP[14] (rt_reset 이 비움)
__global__ void rt_count_stuck(const int* IP, const float* an, int* IPw) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (an[n] > 0.f) atomicAdd(&IPw[14], 1);
}
// 가장 거친 층 평활 (융합, 한 블록): z = ω D r 뒤 nc−1 번 z += ω D (r − (Mi z + K z)). z·곱 누적 KT 는 전역 (한 블록이라 블록 동기화로 차례; 크기 제한 없음), KT 는 끝에 비움.
__global__ void __launch_bounds__(1024) rt_mg_csolve(int l, const int* LM, const float* BG, float sc, const float* Fq, const float* Vq, const float* H, int capQ,
                                                     const float* mc, float itau2, const float* D, const double* W, const float* r, float* z, float* ks, int nc) {
    const int* L = LM + l*8;
    int nn = L[3], nq = L[7];
    float w = (float)W[l];
    for (int n = threadIdx.x; n < nn; n += blockDim.x) {
        const float* d = D + n*9; float r0 = r[n*3], r1 = r[n*3+1], r2 = r[n*3+2];
        for (int a=0;a<3;a++) { z[n*3+a] = w*(d[a*3]*r0 + d[a*3+1]*r1 + d[a*3+2]*r2); ks[n*3+a] = 0.f; }
    }
    __syncthreads();
    for (int it = 1; it < nc; it++) {
        for (int q = threadIdx.x; q < nq; q += blockDim.x) {
            float vq = Vq[q]; if (!(vq > 0.f)) continue;
            int oc = q & 7; int ids[27]; qnodes_d(q, L, ids);
            float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
            float gs[27][3]; float dX[9] = {0,0,0,0,0,0,0,0,0};
            for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; float bs[3] = {sc*bg[0], sc*bg[1], sc*bg[2]}; gqn(F, bs, gs[o]); int id = ids[o];
                for (int a=0;a<3;a++) { float ya = z[id*3+a]; dX[a*3] += ya*gs[o][0]; dX[a*3+1] += ya*gs[o][1]; dX[a*3+2] += ya*gs[o][2]; } }
            float dP[9]; for (int rr=0;rr<9;rr++) { float t = 0.f; for (int c=0;c<9;c++) t += H[(size_t)(rr*9+c)*capQ + q]*dX[c]; dP[rr] = vq*t; }
            for (int o=0;o<27;o++) for (int a=0;a<3;a++) atomicAdd(&ks[ids[o]*3+a], dP[a*3]*gs[o][0] + dP[a*3+1]*gs[o][1] + dP[a*3+2]*gs[o][2]);
        }
        __syncthreads();
        for (int n = threadIdx.x; n < nn; n += blockDim.x) {
            float Mi = mc[n]*itau2; const float* d = D + n*9;
            float e[3]; for (int a=0;a<3;a++) { e[a] = r[n*3+a] - (Mi*z[n*3+a] + ks[n*3+a]); ks[n*3+a] = 0.f; }
            for (int a=0;a<3;a++) z[n*3+a] += w*(d[a*3]*e[0] + d[a*3+1]*e[1] + d[a*3+2]*e[2]);
        }
        __syncthreads();
    }
}
// ---- generalized-α / HHT (implicit-err 절 AN.err3 증분 포텐셜 꼴) ----
// 노드 가속도 a = ma / m (질량 0 이면 0, 고정 노드 0)
__global__ void rt_ga_anode(const int* IP, const float* mn, const float* ma, const unsigned char* fixd, float* ak) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n];
    for (int a=0;a<3;a++) ak[n*3+a] = (m > 0.f && !(fixd && fixd[n])) ? ma[n*3+a]/m : 0.f;
}
// 걸음 시작: xp = U + dt v + dt²(½−β) a, ys = αf U + (1−αf) xp, Ũ = ys + τ² (g − αm a), z0 = ys − (v + dt (1−γ) a)/α, y = U (시작점 = 걸음 시작 위치; 고정 노드 U + (1−αf) dt vbc)
__global__ void rt_ga_step0(const int* IP, const float* U, const float* vk, const float* ak, float dt, float am, float af, float gm, float be, float tau, float al,
                            float g0, float g1, float g2, const unsigned char* fixd, const float* vbc, float* Ut, float* z0, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float gg[3] = {g0, g1, g2};
    for (int a=0;a<3;a++) {
        float u = U[n*3+a], v = vk[n*3+a], ac = ak[n*3+a];
        float xp = u + dt*v + dt*dt*(0.5f - be)*ac;
        float ys = af*u + (1.f - af)*xp;
        Ut[n*3+a] = ys + tau*tau*(gg[a] - am*ac);
        z0[n*3+a] = ys - (v + dt*(1.f - gm)*ac)/al;
        y[n*3+a] = (fixd && fixd[n]) ? u + (1.f - af)*dt*vbc[n*3+a] : Ut[n*3+a];    // 고착 판정은 관성 예측 Ũ 에서 (implicit-err 와 같게); 시작점은 rt_ga_y0 가 U 로
    }
}
// 뉴턴 시작점 = 걸음 시작 위치 U (고착 판정 뒤, implicit-err floor_state --y0 uk 와 같은 식): dd = U − ŷ, η = (I − P) dd + (1/√k) P dd (√k ≤ 1e-6 이면 그 성분 0), y = ŷ + T η
__global__ void rt_ga_y0(const int* IP, const float* U, const float* yh, const float* sk, const int* pm, float n0, float n1, float n2, float* eta, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    int p = pm[n];
    if (p == 2) return;                                                                // 고정 노드: rt_floor_n 이 정한 그대로
    float s = sk[n];
    float isk = (s > 1e-6f) ? 1.f/s : 0.f;
    float dd[3]; for (int a=0;a<3;a++) dd[a] = U[n*3+a] - yh[n*3+a];
    float e[3];
#if defined(AF_FSTK) || defined(AF_FUP)
    if (p < 0) { float sn_ = __int_as_float(-p - 1); float isn_ = (sn_ > 1e-6f) ? 1.f/sn_ : 0.f;               // 절 CM12: 법선 성분은 1/√k_n
        float dn = dd[0]*n0 + dd[1]*n1 + dd[2]*n2; float nn[3] = {n0, n1, n2}; for (int a=0;a<3;a++) e[a] = isn_*dn*nn[a] + isk*(dd[a] - dn*nn[a]); } else
#endif
    if (p == 0) { for (int a=0;a<3;a++) e[a] = isk*dd[a]; }
    else { float dn = dd[0]*n0 + dd[1]*n1 + dd[2]*n2; float nn[3] = {n0, n1, n2}; for (int a=0;a<3;a++) e[a] = dn*nn[a] + isk*(dd[a] - dn*nn[a]); }
    float t[3]; Tapply(s, p, n0, n1, n2, e, t);
    for (int a=0;a<3;a++) { eta[n*3+a] = e[a]; y[n*3+a] = yh[n*3+a] + t[a]; }
}
// 걸음 끝: v1 = α (y − z0) (고정 노드 vbc), a1 = (v1 − v − dt (1−γ) a)/(γ dt), U1 = U + dt v + dt² ((½−β) a + β a1) (Newmark 위치 식), 고착 노드 (an > 0) 의 a1 = 0 (areset), 고정 노드 a1 = 0, 감쇠
__global__ void rt_ga_end(const int* IP, float dt, float gm, float be, float al, const float* y, const float* z0, float* U, float* vk, float* ak,
                          const float* an, int areset, const unsigned char* fixd, const float* vbc, float damp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    bool fx_ = fixd && fixd[n];
    bool stk = areset && an && an[n] > 0.f;
#ifdef AF_BAR
    bool stkb = stk; stk = false;                                               // 장벽 판: 고착은 접선뿐이라 가속도 0 도 접선 성분만 (법선 반동은 장벽 몫)
#ifdef AF_FSTK
    stk = areset && AF_ANA[n] > 0.f;                                            // 절 CM12: (A) 가우스 점 고착 노드는 a 판 그대로 세 성분 가속도 0
#endif
#endif
    for (int a=0;a<3;a++) {
        float u = U[n*3+a], v = vk[n*3+a], ac = ak[n*3+a];
        float v1 = fx_ ? vbc[n*3+a] : al*(y[n*3+a] - z0[n*3+a]);
        float a1 = (v1 - v - dt*(1.f - gm)*ac)/(gm*dt);
        U[n*3+a] = u + dt*v + dt*dt*((0.5f - be)*ac + be*a1);
        vk[n*3+a] = v1*damp;
        ak[n*3+a] = (fx_ || stk) ? 0.f : a1;
    }
#ifdef AF_BAR
    if (stkb && !fx_) { float an_ = ak[n*3]*AF_BNRM[0] + ak[n*3+1]*AF_BNRM[1] + ak[n*3+2]*AF_BNRM[2]; for (int a=0;a<3;a++) ak[n*3+a] = an_*AF_BNRM[a]; }
#endif
}
// 첫 프레임 가속도 a0 = g − f(0)/m', m' = max(m, mlow) (고정 노드 0)
__global__ void rt_ga_a0(const int* IP, const float* mn, const double* f, float mlow, float g0, float g1, float g2, const unsigned char* fixd, float* ak) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float me = fmaxf(mn[n], mlow); float gg[3] = {g0, g1, g2};
    for (int a=0;a<3;a++) ak[n*3+a] = (fixd && fixd[n]) ? 0.f : (gg[a] - (float)f[n*3+a]/me);
}
// G2P (generalized-α): x += Σ w U, v = Σ w v, a_p = Σ w a, F ← (I + Σ U ⊗ ∇w) F
__global__ void rt_g2p_ga(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* U, const float* vk, const float* ak, float* x, float* v, float* pa, float* F) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float u[3] = {0,0,0}, vv[3] = {0,0,0}, aa[3] = {0,0,0}, A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k];
        float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
        for (int a=0;a<3;a++) { float uu = U[nid*3+a]; u[a] += ww*uu; vv[a] += ww*vk[nid*3+a]; aa[a] += ww*ak[nid*3+a]; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }
    }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; pa[p*3+a] = aa[a]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*Fp[k*3+b]; F[(size_t)p*9+a*3+b] = t; }
}
// 절 CM2 시작점 섞기: η = (1 − β) η_U + β η_ŷ (β = 2^-kk[0]), y = ŷ + T η (kk[0] = 0 이면 η_ŷ 그대로 -> y = 관성 예측 ŷ 와 비트 같음)
__global__ void rt_y0mix(const int* IP, const int* kk, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* eS, float* eta, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (pm[n] == 2) return;
    float be = ldexpf(1.f, -kk[0]);
    float e[3]; for (int a=0;a<3;a++) { e[a] = (1.f - be)*eta[n*3+a] + be*eS[n*3+a]; eta[n*3+a] = e[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y[n*3+a] = yh[n*3+a] + t[a];
}
// 절 CM2 재질 판 (선탐색 없음) 장벽 걸음: 온 걸음 후보 y1 = ŷ + T (η + δ)
__global__ void rt_ytry(const int* IP, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* eta, const float* dl, float* y1) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float e[3]; for (int a=0;a<3;a++) e[a] = eta[n*3+a] + dl[n*3+a];
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y1[n*3+a] = yh[n*3+a] + t[a];
}
// 보폭 SC[14] = 2^-kk[0] (rt_lsup full = 0 이 씀)
__global__ void rt_kstep(const int* kk, double* SC) { SC[14] = (double)ldexpf(1.f, -kk[0]); }
// 허용치 조기 종료 제어 (절 AN.14, 단일 스레드). skipify() 가 AF_SKIP 선언과 다른 모든 커널 머리의 "AF_SKIP 이면 바로 돌아감" 을 넣는다 (그 판에서만 컴파일).
// CT (int): 0 ND (이 걸음 뉴턴 끝), 1 PD (이 풀이 PCG 끝), 2 뉴턴 반복 수, 3 PCG 반복 수, 4 뉴턴 상한까지 간 걸음 수, 5 PCG 상한까지 간 풀이 수, 6 걸음 수, 7 풀이 수 (프레임 합)
// CD (double): 0 걸음 첫 |g|², 1 이 풀이의 PCG 목표 |r|², 2 기준 척도 F² = |T f(y0)|² (걸음 시작점의 탄성·접촉 힘 몫, rt_grad 의 SC[56]). eN2 = εN², eP2 = εP² (음수면 그 판정 없음)
// 규칙 (두 적분기 공통): 뉴턴은 |g_k| <= εN F 이면 끝 (k >= 1), PCG 는 |r_i| <= εP min(|g_k|, F) 이면 끝 (|r|² = rt_dotz·rt_cg1 의 SC[57]).
// 후퇴 오일러는 시작점 = 관성 예측 Ũ 라 |g0| = F (바닥 변수 바꿈 몫 빼고); HHT 는 시작점 = 걸음 시작 위치 U 라 |g0| 가 관성 불일치로 부풀어 있어 F 를 따로 둔다.
#ifdef AF_ADAPT
__global__ void rt_ctl(int op, int it, double* SC, int* CT, double* CD, double eN2, double eP2) {
    if (op == 0) { for (int i=0;i<16;i++) CT[i] = 0; AF_SKIP = 0; return; }                         // 프레임 시작
    if (op == 1) { CT[0] = 0; CT[1] = 0; CT[6]++; AF_SKIP = 0; return; }                              // 걸음 시작
    if (op == 2) { CT[1] = 0; AF_SKIP = CT[0]; return; }                                              // 뉴턴 반복 시작
    if (op == 3) {                                                                                     // rt_grad 뒤: |g|² = SC[1], |T f|² = SC[56]
        if (CT[0]) return;
        double g2 = SC[1];
        if (it == 0) { CD[0] = g2; CD[2] = SC[56]; if (!(g2 > 0.0)) { CT[0] = 1; AF_SKIP = 1; return; } }
        else if (eN2 >= 0.0 && g2 <= eN2*CD[2]) { CT[0] = 1; AF_SKIP = 1; return; }
        CT[2]++; return;
    }
    if (op == 4) { if (CT[0]) return; double rf = fmin(SC[1], CD[2]); CD[1] = (eP2 >= 0.0) ? eP2*rf : -1.0; CT[1] = 0; CT[7]++; SC[57] = 0.0; return; }   // PCG 시작: 목표 |r|²
    if (op == 5) { AF_SKIP = CT[0] | CT[1]; return; }                                                 // PCG 반복 시작
    if (op == 6) { double r2 = SC[57]; SC[57] = 0.0; if (CT[0] | CT[1]) return; CT[3]++; if (CD[1] >= 0.0 && r2 <= CD[1]) CT[1] = 1; return; }   // PCG 반복 끝
    if (op == 7) { if (!CT[0] && !CT[1]) CT[5]++; AF_SKIP = CT[0]; return; }                          // PCG 끝 -> 선탐색·갱신
    if (op == 8) { if (!CT[0]) CT[4]++; AF_SKIP = 0; return; }                                        // 걸음 끝 (뉴턴 고리 밖)
}
#endif
}

//@@ GQ_SRC 2424

__global__ void gq_p2q(int N, const float* v, const float* m, const float* pa, const float* fx, const int* pcd, const int* cidx, const int* IP, float* PQ, float* PC) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
    int q = c*8 + oc;
    float mp = m[p];
    for (int a=0;a<3;a++) { float t = mp*v[p*3+a]; atomicAdd(&PQ[(size_t)q*6+a], t); atomicAdd(&PC[(size_t)c*6+a], t); }
    if (pa) for (int a=0;a<3;a++) { float t = mp*pa[p*3+a]; atomicAdd(&PQ[(size_t)q*6+3+a], t); atomicAdd(&PC[(size_t)c*6+3+a], t); }
}
// 결정적 판: 정렬된 입자의 칸 구간 (st) 을 칸 하나가 차례로
__global__ void gq_p2q_d(const int* IP, const int* st, const float* v, const float* m, const float* pa, const float* fx, float* PQ, float* PC) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    for (int p = st[c]; p < st[c+1]; p++) {
        int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
        int q = c*8 + oc;
        float mp = m[p];
        for (int a=0;a<3;a++) { float t = mp*v[p*3+a]; PQ[(size_t)q*6+a] += t; PC[(size_t)c*6+a] += t; }
        if (pa) for (int a=0;a<3;a++) { float t = mp*pa[p*3+a]; PQ[(size_t)q*6+3+a] += t; PC[(size_t)c*6+3+a] += t; }
    }
}
__device__ __forceinline__ void gq_pq(int q, const float* PQ, const float* PC, float kap, int oct, int na, float* pq) {
    int c = q >> 3;
    for (int a=0;a<na;a++) { float pc = PC[(size_t)c*6+a]*0.125f; pq[a] = oct ? ((1.f - kap)*PQ[(size_t)q*6+a] + kap*pc) : pc; }
}
// 점 -> 노드 (원자 합)
__global__ void gq_q2n(const int* IP, const int* cb, const float* BW, const float* mq, const float* PQ, const float* PC, float kap, int oct, float* mn, float* mv, float* ma) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int c = q >> 3, oc = q & 7;
    float pq[6]; gq_pq(q, PQ, PC, kap, oct, ma ? 6 : 3, pq);
    float mm = mq[q];
    int nb1 = IP[7], nb2 = IP[8];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((cb[c*3]+i)*nb1 + (cb[c*3+1]+j))*nb2 + (cb[c*3+2]+k);
        float w = BW[oc*27+o];
        atomicAdd(&mn[nid], w*mm);
        for (int a=0;a<3;a++) atomicAdd(&mv[nid*3+a], w*pq[a]);
        if (ma) for (int a=0;a<3;a++) atomicAdd(&ma[nid*3+a], w*pq[3+a]);
    }
}

//@@ BAR_SRC 12944

// ---- 장벽 판 (--fl_bar, 절 AN.19): 바닥 법선 = 표본점 IPC 거리 장벽, 고착 = 접선만 ----
// 표본점 q = 팔분 칸 입자 질량 중심 (프레임 시작, P2G), 걸음 안 이동 = Σ_o BW (U + thinv (y − U)) (가우스 점 기저); d = (x − p)·n.
// b(d) = −κ (d − d̂)² ln(d/d̂) (0 < d < d̂), κ = ks V_q (ks = (λ + 2μ)/d̂², d̂ = ½ dx_정답). 걸음 시작 d ≤ 0 인 점은 그 걸음 장벽에서 뺌 (AF_BACT 0).
// mode 0: 걸음 시작 (y = 시작점) 판정; mode 1: 기울기 스칼라 c = b' thinv, 덩어리 헤시안 스칼라 h = b'' thinv², 에너지 -> E.
#if defined(AF_BCROSS) || defined(AF_BCNT)
// 절 CM10 (--bar_cross, 사용자 승인 정의 변경; --bar_cnt 진단도 같은 판정): 표본점이 속한 B-스플라인 칸이 바닥 평면을 관통하는가.
// 칸 범위 = 코드의 칸 정의 그대로 (rt_coord: 격자 좌표 L = Ai x 에서 base b = floor(L − ½), 칸 = L ∈ [b+½, b+1½)³ -- 입자 배정 범위, 노드 b..b+2 의 27 개).
// 판정 위치 = 걸음 시작 위치 (rt_bar_q mode 0 과 같은 y): 칸 꼭짓점 8 개 (L = b + ½ + {0,1}³) 의 기준 위치 A L 에 같은 이차 B-스플라인 기저로 노드 이동 U + thinv (y − U) 를 더한 점의
// 평면 부호 거리 d 의 최소 < 0 < 최대 이면 관통 칸 (꼭짓점 축별 기저: f = ½ -> (½, ½, 0), f = 1½ -> (0, ½, ½)).
__device__ __forceinline__ bool bar_cross(int qq, const int* IP, const int* cb, const int* ids, const float* y, const float* U, float thinv, float n0, float n1, float n2, float pn) {
    int c = qq >> 3; float dmin = 3.0e38f, dmax = -3.0e38f;
    for (int cn=0; cn<8; cn++) {
        int ci[3] = {(cn>>2)&1, (cn>>1)&1, cn&1};
        float L[3], w[3][3];
        for (int a=0;a<3;a++) { L[a] = (float)(IP[a] + cb[c*3+a]) + 0.5f + (float)ci[a];
            w[a][0] = ci[a] ? 0.f : 0.5f; w[a][1] = 0.5f; w[a][2] = ci[a] ? 0.5f : 0.f; }
        float x[3]; for (int a=0;a<3;a++) x[a] = AF_BAM[a*3]*L[0] + AF_BAM[a*3+1]*L[1] + AF_BAM[a*3+2]*L[2];
        for (int o=0;o<27;o++) { float ww = w[0][o/9]*w[1][(o/3)%3]*w[2][o%3]; if (ww == 0.f) continue; int id = ids[o];
            for (int a=0;a<3;a++) x[a] += ww*(U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a])); }
        float d = x[0]*n0 + x[1]*n1 + x[2]*n2 - pn;
        dmin = fminf(dmin, d); dmax = fmaxf(dmax, d);
    }
    return dmin < 0.f && dmax > 0.f;
}
#endif
__global__ void __launch_bounds__(128) rt_bar_q(const int* IP, const int* cb, const float* BW, const float* Vq, const float* y, const float* U, float thinv,
                         float n0, float n1, float n2, float pn, float dh, float ks, int mode, float* Gn, float* Kn, double* E) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    double e = 0.0; float c = 0.f, h = 0.f;
    float bm = on ? AF_BX[qq*4+3] : 0.f;
    if (bm > 0.f) {
        int ids[27]; qnodes(qq, IP, cb, ids);
        float xq[3] = {AF_BX[qq*4]/bm, AF_BX[qq*4+1]/bm, AF_BX[qq*4+2]/bm};
        for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
            for (int a=0;a<3;a++) xq[a] += w*(U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a])); }
        float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;
        if (mode == 0) AF_BACT[qq] = (d > 0.f) ? 1 : 0;
#if defined(AF_BCROSS) || defined(AF_BCNT)
        if (mode == 0) { bool cr_ = bar_cross(qq, IP, cb, ids, y, U, thinv, n0, n1, n2, pn);
#ifdef AF_BCNT
            if (cr_) AF_BCF[qq] |= 2;                                           // 진단: 이 프레임 어느 걸음 시작에서든 관통 칸
#endif
#ifdef AF_BCROSS
            if (!cr_) AF_BACT[qq] = 0;                                          // 절 CM10: 관통 칸이 아닌 칸의 표본점은 이 걸음 장벽에서 뺌 (걸음 시작 d ≤ 0 인 점과 같은 처리)
#endif
        }
#endif
        else if (AF_BACT[qq] && d < dh) {
            double kap = (double)ks*(double)Vq[qq];
#ifdef AF_BCNT
            AF_BCF[qq] |= 1;                                                    // 진단: 이 프레임 어느 뉴턴 반복에서든 장벽이 걸림 (활성이고 d < d̂)
#endif
            if (d > 0.f) {
                double dd = (double)d - dh, L = log((double)d/dh);
                e = -kap*dd*dd*L;
                c = (float)(-kap*(2.0*dd*L + dd*dd/d))*thinv;
                h = (float)(-kap*(2.0*L + 4.0*dd/d - dd*dd/((double)d*d)))*thinv*thinv;
            } else e = 1e30;
        }
        if (mode == 1 && !AF_BDET && (c != 0.f || h != 0.f)) for (int o=0;o<27;o++) { float w = BW[oc*27+o]; atomicAdd(&Gn[ids[o]], w*c); atomicAdd(&Kn[ids[o]], w*h); }
    }
    if (mode == 1) {
        if (on && AF_BDET) { AF_BPG[qq] = c; AF_BPH[qq] = h; }
#ifdef AF_BFULL
        if (on) AF_BPH[qq] = h;                                                 // 비덩어리 헤시안 곱 (rt_bar_mvq) 이 쓰는 점 스칼라
#endif
        blocksum_add(e, E);
    }
}
// 노드: f += G n, B += K n nᵀ (덩어리 헤시안: 행 합이라 참 헤시안보다 크다 -> 하강 방향 유지)
__global__ void rt_bar_n(const int* IP, const float* Gn, const float* Kn, float n0, float n1, float n2, double* f, float* B) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float G = Gn[n], K = Kn[n]; if (G == 0.f && K == 0.f) return;
    float nn[3] = {n0, n1, n2};
    for (int a=0;a<3;a++) { f[n*3+a] += (double)(G*nn[a]); for (int b=0;b<3;b++) B[n*9+a*3+b] += K*nn[a]*nn[b]; }
}
// 헤시안 곱 몫: KTp += K n (n·Tp)
__global__ void rt_bar_mv(const int* IP, const float* Kn, float n0, float n1, float n2, const float* Tp, float* KTp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float K = Kn[n]; if (K == 0.f) return;
    float t = K*(Tp[n*3]*n0 + Tp[n*3+1]*n1 + Tp[n*3+2]*n2);
    KTp[n*3] += t*n0; KTp[n*3+1] += t*n1; KTp[n*3+2] += t*n2;
}
// 비덩어리 헤시안 곱 (--bar_full, 절 AN.21): (H Tp)_i += n Σ_q w_qi h_q Σ_j w_qj (n·Tp_j) -- 점 스칼라 t_q = h_q Σ_j w_qj (n·Tp_j) 를 노드 스칼라 bS 로 모은 뒤 rt_bar_mvn 이 n 방향으로 더함
// (덩어리 판 rt_bar_mv 는 diag(Σ_q h_q w_q) 라 상대 변위 몫까지 강성을 줌; 블록 대각 전처리 B 에는 덩어리 값을 그대로 씀 -- 전처리만)
__global__ void __launch_bounds__(128) rt_bar_mvq(const int* IP, const int* cb, const float* BW, float n0, float n1, float n2, const float* Tp, float* bS, float* tq) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float h = AF_BPH[q];
    if (h == 0.f) { if (AF_BDET) tq[q] = 0.f; return; }
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float t = 0.f;
    for (int o=0;o<27;o++) { int id = ids[o]; t += BW[oc*27+o]*(Tp[id*3]*n0 + Tp[id*3+1]*n1 + Tp[id*3+2]*n2); }
    t *= h;
    if (AF_BDET) { tq[q] = t; return; }
    for (int o=0;o<27;o++) atomicAdd(&bS[ids[o]], BW[oc*27+o]*t);
}
__global__ void rt_bar_mvn(const int* IP, const float* bS, float n0, float n1, float n2, float* KTp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float t = bS[n]; if (t == 0.f) return;
    KTp[n*3] += t*n0; KTp[n*3+1] += t*n1; KTp[n*3+2] += t*n2;
}
// 선탐색 후보 NC 개의 장벽 에너지 -> SC[40+i] (rt_elem_lsN 과 같은 후보 y = ŷ + T (η + c δ), c = 2^-i)
__global__ void __launch_bounds__(128) rt_bar_lsN(const int* IP, const int* cb, const float* BW, const float* Vq, const float* yh, const float* eta, const float* dl,
                         const float* sk, const int* pm, const float* U, float thinv, float n0, float n1, float n2, float pn, float dh, float ks, double* SC, int NC, int i0, const double* gate) {
    if (gate && *gate != 0.0) return;                                           // 절 AN.22: 후보 c = 1 이 받아들여졌으면 나머지 후보 생략
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    float bm = on ? AF_BX[qq*4+3] : 0.f;
    bool act = bm > 0.f && AF_BACT[qq];
    int ids[27]; qnodes(qq, IP, cb, ids);
    float x0[3] = {0.f, 0.f, 0.f};
    if (act) for (int a=0;a<3;a++) x0[a] = AF_BX[qq*4+a]/bm;
    double kap = act ? (double)ks*(double)Vq[qq] : 0.0;
    float c = 1.f; for (int i=0;i<i0;i++) c *= 0.5f;
    for (int i=i0;i<NC;i++) {
        double e = 0.0;
        if (act) {
            float xq[3] = {x0[0], x0[1], x0[2]};
            for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
                float ee[3]; for (int a=0;a<3;a++) ee[a] = eta[id*3+a] + c*dl[id*3+a];
                float t[3]; Tapply(sk[id], pm[id], n0, n1, n2, ee, t);
                for (int a=0;a<3;a++) { float ya = yh[id*3+a] + t[a]; xq[a] += w*(U[id*3+a] + thinv*(ya - U[id*3+a])); } }
            float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;
            if (d < dh) { if (d > 0.f) { double dd = (double)d - dh; e = -kap*dd*dd*log((double)d/dh); } else e = 1e30; }
        }
        blocksum_add(e, &SC[40+i]);
        c *= 0.5f;
    }
}
// 절 AN.22 (--mgbar 2, 전처리만): 거친 층 장벽 덩어리 헤시안. Kc_l = R Kc_{l−1} (제한 = 연장의 전치, 1 의 분할이라 Pᵀ diag(K) P 의 행 합 덩어리와 같음 -- 거친 질량 mc = R m 과 같은 꼴),
// 곱 KT += Kc n (n·z), 블록 대각 역 D_b = (D⁻¹ + Kc n nᵀ)⁻¹ = D − Kc (D n)(D n)ᵀ / (1 + Kc nᵀ D n) (셔먼-모리슨, D 대칭)
__global__ void rt_bar_mvl(int l, const int* LM, const float* Kc, float n0, float n1, float n2, const float* z, float* KT) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= LM[l*8+3]) return;
    float K = Kc[n]; if (K == 0.f) return;
    float t = K*(z[n*3]*n0 + z[n*3+1]*n1 + z[n*3+2]*n2);
    KT[n*3] += t*n0; KT[n*3+1] += t*n1; KT[n*3+2] += t*n2;
}
// 절 CM2 (정의 변경 셋째 --y0ut, 둘째 재질 판 장벽): 장벽 실현 가능 보폭. 활성 표본점 (AF_BACT) 의 d 는 y 에 선형이라 y0 -> y1 의 β = 2^-k 에서 d(β) = d0 + β (d1 − d0);
// d(β) > 0 인 가장 작은 k 를 점마다 구해 최대를 kk 에 (정수 원자 최대 -- 순서 무관). k 상한 없음 (β 가 f32 에서 0 이 되면 멈춤 -- 표현 한계).
__global__ void __launch_bounds__(128) rt_bar_kfe(const int* IP, const int* cb, const float* BW, const float* y0, const float* y1, const float* U, float thinv,
                         float n0, float n1, float n2, float pn, int* kk) {
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
    atomicMax(kk, k);
}
// 절 CM8 (B, --y0eb, 정의 변경): 시작점 섞기 β 를 장벽 에너지 기준으로 -- "장벽 에너지가 걸음 시작 위치 (β = 0, 시작점 U) 의 장벽 에너지보다 크지 않은 가장 큰 2^-k".
// 후보 β_i = 2^-(k0+i) (k0 = rt_bar_kfe 의 실현 가능 k, i = 0..NE−1) 의 장벽 에너지 -> SC[1+i], β = 0 -> SC[0]. 활성 표본점 (U 에서 판정, AF_BACT) 의 d 는 y 에 선형
// (y(β) = (1−β) y_U + β y_ŷ) 이라 두 끝 d0 (U)·d1 (ŷ) 만 모으고 후보마다 d = d0 + β (d1 − d0) (rt_bar_kfe 와 같은 식). 장벽 에너지는 β 에 볼록이라 조건을 만족하는 β 는 [0, β*] 구간.
__global__ void __launch_bounds__(128) rt_bar_y0e(const int* IP, const int* cb, const float* BW, const float* Vq, const float* y0, const float* y1, const float* U, float thinv,
                         float n0, float n1, float n2, float pn, float dh, float ks, const int* kk, double* SC, int NE) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    float bm = on ? AF_BX[qq*4+3] : 0.f;
    bool act = bm > 0.f && AF_BACT[qq];
    float d0 = 0.f, d1 = 0.f; double kap = 0.0;
    if (act) {
        int ids[27]; qnodes(qq, IP, cb, ids);
        float xa[3], xb[3];
        for (int a=0;a<3;a++) { xa[a] = AF_BX[qq*4+a]/bm; xb[a] = xa[a]; }
        for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
            for (int a=0;a<3;a++) { xa[a] += w*(U[id*3+a] + thinv*(y0[id*3+a] - U[id*3+a])); xb[a] += w*(U[id*3+a] + thinv*(y1[id*3+a] - U[id*3+a])); } }
        d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn; d1 = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn;
        kap = (double)ks*(double)Vq[qq];
    }
    double e = 0.0;
    if (act && d0 < dh) { if (d0 > 0.f) { double dd = (double)d0 - dh; e = -kap*dd*dd*log((double)d0/dh); } else e = 1e30; }
    blocksum_add(e, &SC[0]);
    float c = ldexpf(1.f, -kk[0]);
    for (int i=0;i<NE;i++) {
        e = 0.0;
        if (act) { float d = d0 + c*(d1 - d0);
            if (d < dh) { if (d > 0.f) { double dd = (double)d - dh; e = -kap*dd*dd*log((double)d/dh); } else e = 1e30; } }
        blocksum_add(e, &SC[1+i]);
        c *= 0.5f;
    }
}
// 후보 중 E(β_i) ≤ E(0) 인 첫 i (가장 큰 β) -> kk = k0 + i; 없으면 β = 0 (kk 1000: ldexpf(1, −1000) = 0 -> rt_y0mix 가 η_U 그대로 = 시작점 U)
__global__ void rt_y0epick(double* SC, int* kk, int NE) {
    double E0 = SC[0]; int k0 = kk[0];
    for (int i=0;i<NE;i++) { double E1 = SC[1+i]; if (isfinite(E1) && E1 <= E0) { kk[0] = k0 + i; return; } }
    kk[0] = 1000;
}
__global__ void rt_mg_dbar(int l, const int* LM, const float* D, const float* Kc, float n0, float n1, float n2, float* Db) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= LM[l*8+3]) return;
    const float* d = D + n*9; float K = Kc[n];
    float u[3]; for (int a=0;a<3;a++) u[a] = d[a*3]*n0 + d[a*3+1]*n1 + d[a*3+2]*n2;
    float den = 1.f + K*(u[0]*n0 + u[1]*n1 + u[2]*n2);
    float c = (K != 0.f && den > 0.f) ? K/den : 0.f;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Db[n*9+a*3+b] = d[a*3+b] - c*u[a]*u[b];
}

//@@ BARN_SRC 5984

// ---- 절 CM14 노드 장벽 (--bar_node add|only, 사용자 승인 11-05 정의 변경): B-스플라인 노드 위치의 바닥 위 높이 d_n 에 같은 IPC 거리 장벽 ----
// 노드 위치 x_n = A L_n + U_n + thinv (y_n − U_n) (L_n = 노드 격자 좌표 IP[0..2] + 상자 번호, 표본점 이동과 같은 걸음 끝 식), d_n = (x_n − p)·n.
// b(d) = −κ_n (d − d̂_n)² ln(d/d̂_n) (0 < d < d̂_n), κ_n = ks_n V_n, ks_n = (λ + 2μ)/d̂_n², V_n = Σ_q w_qn V_q (노드에 모이는 부피, Σ_n V_n = Σ_q V_q).
// 노드 하나의 변위에만 걸리므로 기울기 c_n n (c_n = b' thinv)·헤시안 h_n n nᵀ (h_n = b'' thinv²) 은 노드 대각 (정확) -> 표본점 장벽의 노드 스칼라 bGn·bKn 에 더함.
// 걸음 시작 d_n ≤ 0 (또는 V_n = 0) 인 노드는 그 걸음 장벽에서 뺌 (AF_BNACT 0; 표본점 규칙과 같음). AF_BNF: 진단 비트 1 = 이 프레임 장벽 걸림 (활성이고 d < d̂_n), 2 = 어느 걸음 시작에서 d ≤ 0.
__device__ __forceinline__ float barn_d(int n, const int* IP, const float* y, const float* U, float thinv, float n0, float n1, float n2, float pn) {
    int nb1 = IP[7], nb2 = IP[8];
    float L[3] = {(float)(IP[0] + n/(nb1*nb2)), (float)(IP[1] + (n/nb2)%nb1), (float)(IP[2] + n%nb2)};
    float x[3];
    for (int a=0;a<3;a++) x[a] = AF_BAM[a*3]*L[0] + AF_BAM[a*3+1]*L[1] + AF_BAM[a*3+2]*L[2] + U[n*3+a] + thinv*(y[n*3+a] - U[n*3+a]);
    return x[0]*n0 + x[1]*n1 + x[2]*n2 - pn;
}
__device__ __forceinline__ double barn_e(float d, float dh, double kap) {
    if (!(d < dh)) return 0.0;
    if (!(d > 0.f)) return 1e30;
    double dd = (double)d - dh; return -kap*dd*dd*log((double)d/dh);
}
// 노드 부피 (비결정 판; 결정적 판은 ao_gath_q 고정 순서)
__global__ void __launch_bounds__(128) rt_barn_vq(const int* IP, const int* cb, const float* BW, const float* Vq, float* Vn) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float v = Vq[q]; if (!(v > 0.f)) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    for (int o=0;o<27;o++) atomicAdd(&Vn[ids[o]], BW[oc*27+o]*v);
}
// mode 0: 걸음 시작 판정; mode 1: Gn += c, Kn += h, 에너지 -> E
__global__ void rt_barn_q(const int* IP, const float* y, const float* U, float thinv, float n0, float n1, float n2, float pn, float dh, float ks, int mode, float* Gn, float* Kn, double* E) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = n < IP[9];
    float V = on ? AF_BNV[n] : 0.f;
    double e = 0.0;
    if (V > 0.f) {
        float d = barn_d(n, IP, y, U, thinv, n0, n1, n2, pn);
        if (mode == 0) { AF_BNACT[n] = (d > 0.f) ? 1 : 0; if (!(d > 0.f)) AF_BNF[n] |= 2; }
        else if (AF_BNACT[n] && d < dh) {
            AF_BNF[n] |= 1;
            double kap = (double)ks*(double)V;
            if (d > 0.f) {
                double dd = (double)d - dh, L = log((double)d/dh);
                e = -kap*dd*dd*L;
                Gn[n] += (float)(-kap*(2.0*dd*L + dd*dd/d))*thinv;
                Kn[n] += (float)(-kap*(2.0*L + 4.0*dd/d - dd*dd/((double)d*d)))*thinv*thinv;
            } else e = 1e30;
        }
    } else if (on && mode == 0) AF_BNACT[n] = 0;
    if (mode == 1) blocksum_add(e, E);
}
// 선탐색 후보 NC 개 (rt_bar_lsN 과 같은 후보 y = ŷ + T (η + c δ), c = 2^-i) 의 노드 장벽 에너지 -> SC[40+i]
__global__ void rt_barn_lsN(const int* IP, const float* yh, const float* eta, const float* dl, const float* sk, const int* pm, const float* U, float thinv,
                            float n0, float n1, float n2, float pn, float dh, float ks, double* SC, int NC, int i0, const double* gate) {
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = n < IP[9]; int nn = on ? n : 0;
    float V = on ? AF_BNV[nn] : 0.f;
    bool act = V > 0.f && AF_BNACT[nn];
    double kap = act ? (double)ks*(double)V : 0.0;
    float c = 1.f; for (int i=0;i<i0;i++) c *= 0.5f;
    for (int i=i0;i<NC;i++) {
        double e = 0.0;
        if (act) {
            float ee[3]; for (int a=0;a<3;a++) ee[a] = eta[nn*3+a] + c*dl[nn*3+a];
            float t[3]; Tapply(sk[nn], pm[nn], n0, n1, n2, ee, t);
            float yy[3]; for (int a=0;a<3;a++) yy[a] = yh[nn*3+a] + t[a];
            int nb1 = IP[7], nb2 = IP[8];
            float L[3] = {(float)(IP[0] + nn/(nb1*nb2)), (float)(IP[1] + (nn/nb2)%nb1), (float)(IP[2] + nn%nb2)};
            float x[3]; for (int a=0;a<3;a++) x[a] = AF_BAM[a*3]*L[0] + AF_BAM[a*3+1]*L[1] + AF_BAM[a*3+2]*L[2] + U[nn*3+a] + thinv*(yy[a] - U[nn*3+a]);
            e = barn_e(x[0]*n0 + x[1]*n1 + x[2]*n2 - pn, dh, kap);
        }
        blocksum_add(e, &SC[40+i]);
        c *= 0.5f;
    }
}
// 실현 가능 보폭 (rt_bar_kfe 와 같은 식, 노드): y0 -> y1 의 β = 2^-k 에서 활성 노드 모두 d > 0 인 가장 작은 k -> kk (정수 원자 최대)
__global__ void rt_barn_kfe(const int* IP, const float* y0, const float* y1, const float* U, float thinv, float n0, float n1, float n2, float pn, int* kk) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (!(AF_BNV[n] > 0.f) || !AF_BNACT[n]) return;
    float d0 = barn_d(n, IP, y0, U, thinv, n0, n1, n2, pn), d1 = barn_d(n, IP, y1, U, thinv, n0, n1, n2, pn);
    if (d1 > 0.f) return;
    int k = 0; float c = 1.f;
    while (true) { c *= 0.5f; k++; if (c == 0.f) break; if (d0 + c*(d1 - d0) > 0.f) break; }
    atomicMax(kk, k);
}
// 시작점 섞기 β (--y0eb, rt_bar_y0e 와 같은 식, 노드): β = 0 -> SC[0], 후보 β_i = 2^-(k0+i) -> SC[1+i]
__global__ void rt_barn_y0e(const int* IP, const float* y0, const float* y1, const float* U, float thinv, float n0, float n1, float n2, float pn, float dh, float ks, const int* kk, double* SC, int NE) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = n < IP[9]; int nn = on ? n : 0;
    float V = on ? AF_BNV[nn] : 0.f;
    bool act = V > 0.f && AF_BNACT[nn];
    float d0 = 0.f, d1 = 0.f; double kap = 0.0;
    if (act) { d0 = barn_d(nn, IP, y0, U, thinv, n0, n1, n2, pn); d1 = barn_d(nn, IP, y1, U, thinv, n0, n1, n2, pn); kap = (double)ks*(double)V; }
    blocksum_add(act ? barn_e(d0, dh, kap) : 0.0, &SC[0]);
    float c = ldexpf(1.f, -kk[0]);
    for (int i=0;i<NE;i++) {
        blocksum_add(act ? barn_e(d0 + c*(d1 - d0), dh, kap) : 0.0, &SC[1+i]);
        c *= 0.5f;
    }
}

//@@ BARE_SRC 2547

// ---- 절 CM14.3 진단 --bar_ecell (결과 불변: 풀이 버퍼는 읽기만, 쓰는 곳은 진단 버퍼 EC·EL·EM·ED·ET 뿐) ----
// 걸음 끝 (뉴턴 고리 뒤, rt_ga_end 전 -- y = 걸음 끝 반복값, 이동 = U + thinv (y − U), 풀이 장벽과 같은 식) 의 장벽 에너지를 칸 단위로 모음.
// 칸 = 표본점 칸 (qq >> 3) 의 가운데 노드 (ids[13] = 칸 base + (1,1,1)) 번호로 셈. 표본점 장벽 -> 그 점이 속한 칸, 노드 장벽 -> 그 노드가 가운데인 칸 (base = 노드 − 1).
// 활성 (AF_BACT / AF_BNACT) 이고 0 < d < d̂ 인 점·노드만 (풀이 에너지와 같은 항); d ≤ 0 (풀이에서는 1e30) 은 에너지에 넣지 않고 ET[2]·ET[3] 에 셈.
__global__ void __launch_bounds__(128) rt_bare_q(const int* IP, const int* cb, const float* BW, const float* y, const float* U, float thinv,
                         float n0, float n1, float n2, float pn, float dh, float ks, const float* Vq, float* EC, double* ET) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3];
    if (!(bm > 0.f) || !AF_BACT[q]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xq[3] = {AF_BX[q*4]/bm, AF_BX[q*4+1]/bm, AF_BX[q*4+2]/bm};
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
        for (int a=0;a<3;a++) xq[a] += w*(U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a])); }
    float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;
    if (!(d < dh)) return;
    if (!(d > 0.f)) { atomicAdd(&ET[2], 1.0); return; }
    double kap = (double)ks*(double)Vq[q], dd = (double)d - dh;
    double e = -kap*dd*dd*log((double)d/dh);
    atomicAdd(&EC[ids[13]], (float)e); atomicAdd(&ET[0], e);
}
#ifdef AF_BNODE
__global__ void rt_bare_n(const int* IP, const float* y, const float* U, float thinv, float n0, float n1, float n2, float pn, float dh, float ks, float* EC, double* ET) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float V = AF_BNV[n];
    if (!(V > 0.f) || !AF_BNACT[n]) return;
    float d = barn_d(n, IP, y, U, thinv, n0, n1, n2, pn);
    if (!(d < dh)) return;
    if (!(d > 0.f)) { atomicAdd(&ET[3], 1.0); return; }
    double kap = (double)ks*(double)V, dd = (double)d - dh;
    double e = -kap*dd*dd*log((double)d/dh);
    atomicAdd(&EC[n], (float)e); atomicAdd(&ET[1], e);
}
#endif
// 걸음 끝: EL = 이 걸음 칸 에너지 (마지막 걸음이면 프레임 끝 값), EM = 프레임 안 걸음 끝 최대, ED = 걸음 끝 노드 이동 (칸 꼭짓점 그리기용), EC 비움
__global__ void rt_bare_acc(const int* IP, const float* y, const float* U, float thinv, float* EC, float* EL, float* EM, float* ED) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float e = EC[n]; EL[n] = e; EM[n] = fmaxf(EM[n], e); EC[n] = 0.f;
    for (int a=0;a<3;a++) ED[n*3+a] = U[n*3+a] + thinv*(y[n*3+a] - U[n*3+a]);
}

//@@ VRES_SRC 3774

// ---- 절 CM15 −z 저항 바닥 (--fl_vres k, 사용자 승인 11-05 정의 변경): 장벽·고착 대신 바닥 밑 표본점의 법선 반대 방향 속도에 한쪽 이차 벌점 ----
// 표본점 q = 장벽과 같은 팔분 칸 입자 질량 중심 x̄_q (AF_BX, 프레임 시작 P2G), 질량 m_q = AF_BX[4q+3]. 걸음 시작 위치 x(U) = x̄ + Σ_o w_qo U_o 의 평면 부호 거리 d0 < 0 인 점만 (mode 0 에서 AF_BACT 1).
// 걸음 이동 법선 성분 s = (x(y) − x(U))·n = thinv Σ_o w_qo (y_o − U_o)·n (= h v_n). E_q = ½ c m_q s² (s < 0), 0 (s ≥ 0)  [= ½ c m_q h² min(0, v_n)²].
// 기울기 (정확): dE/dy_o = c m_q s thinv w_qo n -> 점 스칼라 c_q = c m_q s thinv (장벽과 같은 노드 모으기 f += G n);
// 헤시안: 점 스칼라 h_q = c m_q thinv² (s < 0) 를 장벽처럼 행 합 덩어리 (노드 K n nᵀ, 참 헤시안 c m thinv² w wᵀ ⊗ n nᵀ 보다 크다 -- 블록 대각·PCG 곱·거친 층 같은 경로).
__global__ void __launch_bounds__(128) rt_vres_q(const int* IP, const int* cb, const float* BW, const float* y, const float* U, float thinv,
                         float n0, float n1, float n2, float pn, float cv, int mode, float* Gn, float* Kn, double* E) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    double e = 0.0; float c = 0.f, h = 0.f;
    float bm = on ? AF_BX[qq*4+3] : 0.f;
    if (bm > 0.f) {
        int ids[27]; qnodes(qq, IP, cb, ids);
        if (mode == 0) {
            float xq[3] = {AF_BX[qq*4]/bm, AF_BX[qq*4+1]/bm, AF_BX[qq*4+2]/bm};
            for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o]; for (int a=0;a<3;a++) xq[a] += w*U[id*3+a]; }
            float d0 = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;
            AF_BACT[qq] = (d0 < 0.f) ? 1 : 0;
        } else if (AF_BACT[qq]) {
            float s = 0.f;
            for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
                s += w*((y[id*3]-U[id*3])*n0 + (y[id*3+1]-U[id*3+1])*n1 + (y[id*3+2]-U[id*3+2])*n2); }
            s *= thinv;
            if (s < 0.f) { double km = (double)cv*(double)bm; e = 0.5*km*(double)s*(double)s; c = (float)(km*(double)s)*thinv; h = (float)km*thinv*thinv; }
        }
        if (mode == 1 && !AF_BDET && (c != 0.f || h != 0.f)) for (int o=0;o<27;o++) { float w = BW[oc*27+o]; atomicAdd(&Gn[ids[o]], w*c); atomicAdd(&Kn[ids[o]], w*h); }
    }
    if (mode == 1) {
        if (on && AF_BDET) { AF_BPG[qq] = c; AF_BPH[qq] = h; }
        blocksum_add(e, E);
    }
}
// 선탐색 후보 NC 개 (rt_bar_lsN 과 같은 후보 y = ŷ + T (η + c δ), c = 2^-i) 의 벌점 에너지 -> SC[40+i] (pn 은 rt_vres_q 와 같은 인자 묶음을 받으려고 둠, 안 씀)
__global__ void __launch_bounds__(128) rt_vres_lsN(const int* IP, const int* cb, const float* BW, const float* yh, const float* eta, const float* dl,
                         const float* sk, const int* pm, const float* U, float thinv, float n0, float n1, float n2, float pn, float cv, double* SC, int NC, int i0, const double* gate) {
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    float bm = on ? AF_BX[qq*4+3] : 0.f;
    bool act = bm > 0.f && AF_BACT[qq];
    int ids[27]; qnodes(qq, IP, cb, ids);
    double km = act ? (double)cv*(double)bm : 0.0;
    float c = 1.f; for (int i=0;i<i0;i++) c *= 0.5f;
    for (int i=i0;i<NC;i++) {
        double e = 0.0;
        if (act) {
            float s = 0.f;
            for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
                float ee[3]; for (int a=0;a<3;a++) ee[a] = eta[id*3+a] + c*dl[id*3+a];
                float t[3]; Tapply(sk[id], pm[id], n0, n1, n2, ee, t);
                s += w*((yh[id*3]+t[0]-U[id*3])*n0 + (yh[id*3+1]+t[1]-U[id*3+1])*n1 + (yh[id*3+2]+t[2]-U[id*3+2])*n2); }
            s *= thinv;
            if (s < 0.f) e = 0.5*km*(double)s*(double)s;
        }
        blocksum_add(e, &SC[40+i]);
        c *= 0.5f;
    }
}

//@@ Y0L_SRC 4902

// 절 CM27 (c) --y0loc (정의 변경은 뉴턴 시작점뿐): 장벽 판 시작점 섞기를 국소로 -- 몸 전체 스칼라 β 대신 노드별 β_n = 2^-kn[n].
// 표본점마다 전역 규칙 (rt_bar_kfe 실현 가능 + --y0eb 장벽 에너지) 을 그 점 하나에만 적용한 k_q -> 그 점이 걸친 노드 (B-스플라인 가중 ≠ 0) 에 원자 최대.
// 노드마다 β 가 달라 d 가 β 에 선형이 아니므로, 섞은 뒤 활성 표본점 d > 0 을 확인해 어긴 점의 노드를 (그 점 노드 최대 k) + 1 로 (라운드 R 번),
// 마지막 확인에도 어긴 점이 있으면 모든 노드를 전역 k (지금 규칙) 로 (실현 가능 보장). 정수 원자만 -> 결정적.
__device__ __forceinline__ double yl_e(float d, float dh, double kap) {
    if (!(d < dh)) return 0.0;
    if (!(d > 0.f)) return 1e30;
    double dd = (double)d - dh; return -kap*dd*dd*log((double)d/dh);
}
__global__ void __launch_bounds__(128) yl_kq(const int* IP, const int* cb, const float* BW, const float* Vq, const float* y0, const float* y1, const float* U, float thinv,
                         float n0, float n1, float n2, float pn, float dh, float ks, int eb, int* kn) {
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
#if AF_BLOC >= 2
    // 절 CM189 --bloc 2: 관통 쪽 몫만 S⁻ = Σ_n w_n min(s_n, 0) (s_n = thinv (ŷ − y_U)_n·n̂) -> 노드 β_n ≤ 2^-k_q 이면 어떤 섞임에서도 d ≥ d0 + 2^-k_q S⁻
    // (실현 가능·장벽 에너지 둘 다 이 하한으로 판정 -> 섞은 뒤 어긴 점 없음, 몸 전체 되돌림 (yl_fb) 이 안 걸림)
    { float sm = 0.f;
      for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
          float sn = thinv*((y1[id*3]-y0[id*3])*n0 + (y1[id*3+1]-y0[id*3+1])*n1 + (y1[id*3+2]-y0[id*3+2])*n2);
          sm += w*fminf(sn, 0.f); }
      d1 = d0 + sm; }
#endif
    int k = 0;
    if (!(d1 > 0.f)) { float c = 1.f; while (true) { c *= 0.5f; k++; if (c == 0.f) break; if (d0 + c*(d1 - d0) > 0.f) break; } }
    if (eb) {                                                                    // --y0eb 를 이 점 하나에: E_q(2^-(k+i)) ≤ E_q(0) 인 첫 i, 없으면 β 0
        double kap = (double)ks*(double)Vq[q], E0 = yl_e(d0, dh, kap);
        float c = ldexpf(1.f, -k); int i = 0;
        for (; i<59; i++) { double E = yl_e(d0 + c*(d1 - d0), dh, kap); if (E <= E0) break; c *= 0.5f; }
        k = (i == 59) ? 1000 : k + i;
    }
    if (k == 0) return;
    for (int o=0;o<27;o++) if (BW[oc*27+o] != 0.f) atomicMax(&kn[ids[o]], k);
}
__global__ void yl_mix(const int* IP, const int* kn, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* eU, const float* eS, float* eta, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (pm[n] == 2) return;
    float be = ldexpf(1.f, -kn[n]);
    float e[3]; for (int a=0;a<3;a++) { e[a] = (1.f - be)*eU[n*3+a] + be*eS[n*3+a]; eta[n*3+a] = e[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y[n*3+a] = yh[n*3+a] + t[a];
}
__global__ void __launch_bounds__(128) yl_chk(const int* IP, const int* cb, const float* BW, const float* y, const float* U, float thinv, float n0, float n1, float n2, float pn,
                         const int* kn, int* kn2, int* fl) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3];
    if (!(bm > 0.f) || !AF_BACT[q]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xa[3];
    for (int a=0;a<3;a++) xa[a] = AF_BX[q*4+a]/bm;
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o]; for (int a=0;a<3;a++) xa[a] += w*(U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a])); }
    if (xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn > 0.f) return;
    atomicAdd(fl, 1);
    int km = 0; for (int o=0;o<27;o++) if (BW[oc*27+o] != 0.f) km = max(km, kn[ids[o]]);
    int kt = (km >= 999) ? 1000 : km + 1;
    for (int o=0;o<27;o++) if (BW[oc*27+o] != 0.f) atomicMax(&kn2[ids[o]], kt);
}
__global__ void yl_merge(const int* IP, int* kn, const int* kn2) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    kn[n] = max(kn[n], kn2[n]);
}
__global__ void yl_fb(const int* IP, const int* fl, const int* kk, int* kn) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || fl[0] == 0) return;
    kn[n] = kk[0];
}
// 진단: [0] k > 0 노드 수, [1] 최대 k, [2] β 0 (k ≥ 1000) 노드 수, [3] 몸 노드 수 (질량 > 0)
__global__ void yl_st(const int* IP, const int* kn, const float* mn, int* st) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    int k = kn[n];
    if (k > 0) { atomicAdd(&st[0], 1); atomicMax(&st[1], k); }
    if (k >= 1000) atomicAdd(&st[2], 1);
    if (mn[n] > 0.f) atomicAdd(&st[3], 1);
}

//@@ NCL_SRC 3568

// ---- 절 CM16 노드 법선 하드 클램프 바닥 (--fl_nclamp, 사용자 지시 하드 클램프 11-05): 바닥 아래 노드의 걸음 변위 법선 성분이 음수면 0 (한쪽 구속, 위로·접선 자유, 마찰 없음) ----
// 후보 (걸음마다, 걸음 시작에 판정): 걸음 시작 노드 위치 x_n(U) = A L_n + U_n 의 평면 부호 거리 d0 = (x_n − p)·n ≤ 0 이고 변수 바꿈이 pm 0·1 인 노드 (pm 2 고정 노드·pm < 0 은 제외).
// 활성 집합 (고착·고정 노드와 같은 변수 바꿈 y = ŷ + T η 로 구속): 활성 노드는 pm 3 (T = √k P -- 법선 성분 0), ŷ 의 법선 성분 = U·n, η 법선 0 -> y·n = U·n (걸음 변위 법선 0).
//   이 노드의 법선 방향은 뉴턴·PCG·선탐색에서 빠지고 (기울기 T f·헤시안 T K T·블록 대각·MG 층 0 모두 같은 T), 관성 몫의 법선 상수는 그 반복 안에서 같아 선탐색 비교에 영향 없음.
// ncf: 0 비후보, 1 후보 (자유), 2 활성 (원래 pm 0), 3 활성 (원래 pm 1). yrn = 활성으로 바꿀 때의 원래 ŷ·n (풀 때 되돌림).
// mode 0 걸음 시작 (시작점 정한 뒤): 후보 판정 + 넣기. mode 1 뉴턴 반복마다 (힘 f 뒤, rt_grad 앞): 빼기 -- 활성 노드의 자유 에너지 법선 기울기 g_n = Mi (y·n − yrn)/s_n² + f·n < 0
//   (위로 가면 에너지가 줄어듦 = 구속 반력이 아래로 당겨야 하는 경우) 이면 풀어 원래 pm 로 (y 그대로). mode 2 뉴턴 반복마다 (갱신 rt_lsup 뒤): 넣기 -- 자유 후보의 (y − U)·n < 0 이면 법선 성분을 U·n 으로 투영하고 활성.
// mode 3 걸음 끝 (진단, 결과 불변): 활성 노드 수. cnt[0] 후보 수, [1] 넣기 횟수, [2] 빼기 횟수, [3] 걸음 끝 활성 수 (모두 질량 > 0 노드; 정수 원자 -- 순서 무관).
__device__ __forceinline__ void ncl_on(int n, float n0, float n1, float n2, const float* U, const float* sk, float* y, float* yh, float* eta, int* pm, int* ncf, float* yrn) {
    float nn[3] = {n0, n1, n2};
    float yhn = yh[n*3]*n0 + yh[n*3+1]*n1 + yh[n*3+2]*n2;
    float un = U[n*3]*n0 + U[n*3+1]*n1 + U[n*3+2]*n2;
    float en = eta[n*3]*n0 + eta[n*3+1]*n1 + eta[n*3+2]*n2;
    float s = sk[n];
    for (int a=0;a<3;a++) { yh[n*3+a] += (un - yhn)*nn[a]; eta[n*3+a] -= en*nn[a]; y[n*3+a] = yh[n*3+a] + s*eta[n*3+a]; }
    ncf[n] = 2 + pm[n]; yrn[n] = yhn; pm[n] = 3;
}
__global__ void rt_ncl(const int* IP, int mode, const float* Am, float n0, float n1, float n2, float pn, const float* U, const float* mn, const float* meff, float itau2, const double* f,
                       const float* sk, float* y, float* yh, float* eta, int* pm, int* ncf, float* yrn, int* cnt) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    bool mp = mn[n] > 0.f;
    if (mode == 0) {
        int nb1 = IP[7], nb2 = IP[8];
        float L[3] = {(float)(IP[0] + n/(nb1*nb2)), (float)(IP[1] + (n/nb2)%nb1), (float)(IP[2] + n%nb2)};
        float d0 = -pn; float nn[3] = {n0, n1, n2};
        for (int a=0;a<3;a++) d0 += (Am[a*3]*L[0] + Am[a*3+1]*L[1] + Am[a*3+2]*L[2] + U[n*3+a])*nn[a];
        int p = pm[n];
        ncf[n] = (d0 <= 0.f && (p == 0 || p == 1)) ? 1 : 0;
        if (ncf[n] && mp) atomicAdd(&cnt[0], 1);
    }
    if (mode == 0 || mode == 2) {
        if (ncf[n] != 1) return;
        float dn = (y[n*3]-U[n*3])*n0 + (y[n*3+1]-U[n*3+1])*n1 + (y[n*3+2]-U[n*3+2])*n2;
        if (dn < 0.f) { ncl_on(n, n0, n1, n2, U, sk, y, yh, eta, pm, ncf, yrn); if (mp) atomicAdd(&cnt[1], 1); }
        return;
    }
    if (mode == 1) {
        int c = ncf[n]; if (c < 2) return;
        int p0 = c - 2; float sn = (p0 == 0) ? sk[n] : 1.f;
        if (!(sn > 1e-6f)) return;                                                       // 법선 √k ≈ 0 (구속으로 거의 고정) 은 풀지 않음
        float yn = y[n*3]*n0 + y[n*3+1]*n1 + y[n*3+2]*n2;
        float gn = meff[n]*itau2*(yn - yrn[n])/(sn*sn) + (float)(f[n*3]*n0 + f[n*3+1]*n1 + f[n*3+2]*n2);
        if (!(gn < 0.f)) return;
        float yhn = yh[n*3]*n0 + yh[n*3+1]*n1 + yh[n*3+2]*n2; float nn[3] = {n0, n1, n2};
        float en = (yn - yrn[n])/sn;
        for (int a=0;a<3;a++) { yh[n*3+a] += (yrn[n] - yhn)*nn[a]; eta[n*3+a] += en*nn[a]; }
        pm[n] = p0; ncf[n] = 1;
        if (mp) atomicAdd(&cnt[2], 1);
        return;
    }
    if (mode == 3 && ncf[n] >= 2 && mp) atomicAdd(&cnt[3], 1);
}

//@@ FUP_SRC 5196

// ---- 절 CM17 (A) 바닥 위 허용 (--fl_up 1|2, 사용자 승인 11-06 정의 변경): (A) 가우스 점 고착 몫의 법선 성분을 한쪽 구속으로 (아래 −n 만 막고 위 +n 자유) ----
// 후보 (걸음 시작, rt_floor_n 뒤): 고착 몫 an > 0 이고 변수 바꿈이 pm 0·1 인 노드 (고정 노드 pm 2 제외 -- rt_ga_end 의 고착 가속도 0 처치 대상과 같은 집합). fsk = 그 노드의 고착 √k (rt_floor_n 의 k = (1 − r̄)^κ 그대로).
// 변수 바꿈 (rt_floor_n 과 같은 꼴, 접선·법선 따로): ŷ = Ũ + (1 − k_t) P d + (1 − k_n)(I − P) d (d = z0 − Ũ), T = √k_t P + √k_n (I − P), y = ŷ + T η.
//   접선 √k_t = √k (AF_FUP 1, 판 (i) -- 지금처럼 고착) 또는 1 (AF_FUP 2, 판 (ii) -- 마찰 없음). 법선 √k_n = √k (구속, ufl 2) 또는 1 (자유, ufl 1).
//   (i) 구속 = 원래 pm 0 과 같은 식, 자유 = 원래 pm 1 (바깥 법선 이탈) 과 같은 식 -> 원래 (A) 와 다른 것은 법선을 묶는 판정뿐 (원래: 걸음 시작 관성 예측 속도 법선 성분 > 0 이면 자유, 걸음 안 고정).
// mode 0 걸음 시작 (시작점 = ŷ 그대로인 (A) 판): 후보 판정 + 관성 예측 걸음 변위 (Ũ − U)·n < 0 이면 구속, 아니면 자유 -- 시작 y 는 rt_floor_n 과 같은 사상 y = ŷ + T T (Ũ − ŷ).
// mode 1 뉴턴 반복마다 (힘 f 뒤, rt_grad 앞) = 빼기: 구속 노드의 법선 자유 에너지 기울기 g_n = Mi (y·n − Ũ·n) + f·n < 0 (위로 가면 에너지가 줄어듦) 이면 법선 자유로 (y 그대로, ŷ·n = Ũ·n, η·n = y·n − Ũ·n).
//   (구속 중인 노드는 법선 이동이 k 배로 눌려 걸음 변위 부호로는 풀 수 없음 -- 절 CM16 빼기 규칙과 같은 꼴.)
// mode 2 뉴턴 반복마다 (갱신 rt_lsup 뒤) = 넣기: 자유 후보의 걸음 변위 법선 성분 (y − U)·n < 0 이면 구속 -- 법선 성분에 rt_floor_n 과 같은 고착 사상 y·n ← ŷ·n + k (y·n − ŷ·n) (ŷ·n = Ũ·n + (1 − k) d·n).
// mode 3 걸음 끝 (진단, 결과 불변): 구속 노드 수. mode 4 rt_ga_end 뒤 (고착 가속도 0 처치를 묶인 성분에만): 판 (i) 구속 = 세 성분 0 (원래 그대로), 자유 = 접선만 0; 판 (ii) 구속 = 법선만 0, 자유 = 0 처치 없음.
// cnt[0] 후보 수, [1] 넣기 횟수 (걸음 시작 구속 포함), [2] 빼기 횟수, [3] 걸음 끝 구속 수 (모두 질량 > 0 노드; 정수 원자 -- 순서 무관).
__device__ __forceinline__ void fup_pm(int n, float st, float sn, float* sk, int* pm) {
    sk[n] = st; pm[n] = (sn == st) ? 0 : ((sn == 1.f) ? 1 : (-1 - __float_as_int(sn)));
}
__global__ void rt_fup(const int* IP, int mode, float n0, float n1, float n2, const float* U, const float* Ut, const float* z0, const float* an, const float* mn, const float* meff, float itau2,
                       const double* f, float* sk, int* pm, float* y, float* yh, float* eta, int* ufl, float* fsk, int* cnt, float dt, float gm, float al, const float* vo, const float* ao, float* ak) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    bool mp = mn[n] > 0.f;
    float nn[3] = {n0, n1, n2};
    if (mode == 0) {
        int p = pm[n];
        if (!(an[n] > 0.f) || !(p == 0 || p == 1)) { ufl[n] = 0; return; }
        float s = sk[n]; fsk[n] = s;
        float st = (AF_FUP == 1) ? s : 1.f;
        float dU = 0.f, dn = 0.f;
        for (int a=0;a<3;a++) { dU += (Ut[n*3+a] - U[n*3+a])*nn[a]; dn += (z0[n*3+a] - Ut[n*3+a])*nn[a]; }
        bool act = dU < 0.f;
        float sn = act ? s : 1.f, kt = st*st, kn = sn*sn;
        float h[3], e[3];
        for (int a=0;a<3;a++) { float d = z0[n*3+a] - Ut[n*3+a]; h[a] = Ut[n*3+a] + (1.f - kt)*(d - dn*nn[a]) + (1.f - kn)*dn*nn[a]; e[a] = Ut[n*3+a] - h[a]; }
        float en = e[0]*n0 + e[1]*n1 + e[2]*n2;
        float et[3]; for (int a=0;a<3;a++) et[a] = st*(e[a] - en*nn[a]) + sn*en*nn[a];
        fup_pm(n, st, sn, sk, pm);
        float ty[3]; Tapply(sk[n], pm[n], n0, n1, n2, et, ty);
        for (int a=0;a<3;a++) { yh[n*3+a] = h[a]; eta[n*3+a] = et[a]; y[n*3+a] = h[a] + ty[a]; }
        ufl[n] = act ? 2 : 1;
        if (mp) { atomicAdd(&cnt[0], 1); if (act) atomicAdd(&cnt[1], 1); }
        return;
    }
    int u = ufl[n];
    if (u == 0) return;
    if (mode == 1) {
        if (u != 2) return;
        float yn = y[n*3]*n0 + y[n*3+1]*n1 + y[n*3+2]*n2, utn = Ut[n*3]*n0 + Ut[n*3+1]*n1 + Ut[n*3+2]*n2;
        float gn = meff[n]*itau2*(yn - utn) + (float)(f[n*3]*n0 + f[n*3+1]*n1 + f[n*3+2]*n2);
        if (!(gn < 0.f)) return;
        float yhn = yh[n*3]*n0 + yh[n*3+1]*n1 + yh[n*3+2]*n2, en = eta[n*3]*n0 + eta[n*3+1]*n1 + eta[n*3+2]*n2;
        for (int a=0;a<3;a++) { yh[n*3+a] += (utn - yhn)*nn[a]; eta[n*3+a] += ((yn - utn) - en)*nn[a]; }
        fup_pm(n, sk[n], 1.f, sk, pm); ufl[n] = 1;
        if (mp) atomicAdd(&cnt[2], 1);
        return;
    }
    if (mode == 2) {
        if (u != 1) return;
        float dn = (y[n*3]-U[n*3])*n0 + (y[n*3+1]-U[n*3+1])*n1 + (y[n*3+2]-U[n*3+2])*n2;
        if (!(dn < 0.f)) return;
        float s = fsk[n], kn = s*s;
        float dz = (z0[n*3]-Ut[n*3])*n0 + (z0[n*3+1]-Ut[n*3+1])*n1 + (z0[n*3+2]-Ut[n*3+2])*n2;
        float hn = Ut[n*3]*n0 + Ut[n*3+1]*n1 + Ut[n*3+2]*n2 + (1.f - kn)*dz;
        float yn = y[n*3]*n0 + y[n*3+1]*n1 + y[n*3+2]*n2;
        float enw = s*(yn - hn), ynw = hn + s*enw;
        float yhn = yh[n*3]*n0 + yh[n*3+1]*n1 + yh[n*3+2]*n2, en = eta[n*3]*n0 + eta[n*3+1]*n1 + eta[n*3+2]*n2;
        for (int a=0;a<3;a++) { yh[n*3+a] += (hn - yhn)*nn[a]; eta[n*3+a] += (enw - en)*nn[a]; y[n*3+a] += (ynw - yn)*nn[a]; }
        fup_pm(n, sk[n], s, sk, pm); ufl[n] = 2;
        if (mp) atomicAdd(&cnt[1], 1);
        return;
    }
    if (mode == 3) { if (u == 2 && mp) atomicAdd(&cnt[3], 1); return; }
    if (mode == 4) {
        float a1[3];
        for (int a=0;a<3;a++) { float v1 = al*(y[n*3+a] - z0[n*3+a]); a1[a] = (v1 - vo[n*3+a] - dt*(1.f - gm)*ao[n*3+a])/(gm*dt); }
        float an_ = a1[0]*n0 + a1[1]*n1 + a1[2]*n2;
        bool tz = (AF_FUP == 1), nz = (u == 2);
        for (int a=0;a<3;a++) ak[n*3+a] = (tz ? 0.f : a1[a] - an_*nn[a]) + (nz ? 0.f : an_*nn[a]);
    }
}

//@@ SNH_SRC 1235

#define AF_SNH 1
__device__ __forceinline__ double snh_e(double vq, const float* S, float mu, float lam) {
    double ms = (double)mu, ls = (double)lam + (double)mu, J = (double)S[0]*(double)S[1]*(double)S[2];
    return vq*(0.5*ms*((double)S[0]*S[0] + (double)S[1]*S[1] + (double)S[2]*S[2] - 3.0) - ms*(J - 1.0) + 0.5*ls*(J - 1.0)*(J - 1.0));
}
__device__ __forceinline__ void snh_ph(const float* S, float mu, float lam, float* ph) {
    float ls = lam + mu, J = S[0]*S[1]*S[2], c = ls*(J - 1.f) - mu;
    float gJ[3] = {S[1]*S[2], S[0]*S[2], S[0]*S[1]};
    for (int i=0;i<3;i++) ph[i] = mu*S[i] + c*gJ[i];
}
__device__ __forceinline__ void snh_D(const float* S, float mu, float lam, float D[3][3]) {
    float ls = lam + mu, J = S[0]*S[1]*S[2], c = ls*(J - 1.f) - mu;
    float gJ[3] = {S[1]*S[2], S[0]*S[2], S[0]*S[1]};
    float Sg[3][3] = {{0.f, S[2], S[1]}, {S[2], 0.f, S[0]}, {S[1], S[0], 0.f}};
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = ((i==j) ? mu : 0.f) + ls*gJ[i]*gJ[j] + c*Sg[i][j];
}
__device__ __forceinline__ void snh_tf(const float* S, int k, float mu, float lam, float* lt, float* lf) {
    float ls = lam + mu, J = S[0]*S[1]*S[2], c = ls*(J - 1.f) - mu;
    *lt = mu + c*S[k]; *lf = mu - c*S[k];
}

//@@ QJD_SRC 1156

extern "C" {
__global__ void rt_qjd(const int* IP, const int* cb, const float* BG, const float* BW, const float* Fq, const float* xq0, const float* U, double* Jq, float* Xq) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int c = q >> 3, oc = q & 7, nb1 = IP[7], nb2 = IP[8];
    float F[9], X[9]; for (int k=0;k<9;k++) { F[k] = Fq[(size_t)q*9+k]; X[k] = F[k]; }
    float xq[3] = {xq0[q*3], xq0[q*3+1], xq0[q*3+2]};
    for (int o=0;o<27;o++) { int i = o/9, j = (o/3)%3, k = o%3; int id = ((cb[c*3]+i)*nb1 + (cb[c*3+1]+j))*nb2 + (cb[c*3+2]+k);
        const float* bg = BG + (oc*27+o)*3;
        float g[3] = {F[0]*bg[0] + F[3]*bg[1] + F[6]*bg[2], F[1]*bg[0] + F[4]*bg[1] + F[7]*bg[2], F[2]*bg[0] + F[5]*bg[1] + F[8]*bg[2]};
        float w = BW[oc*27+o];
        for (int a=0;a<3;a++) { float u = U[id*3+a]; X[a*3] += u*g[0]; X[a*3+1] += u*g[1]; X[a*3+2] += u*g[2]; xq[a] += w*u; } }
    double x0 = X[0], x1 = X[1], x2 = X[2], x3 = X[3], x4 = X[4], x5 = X[5], x6 = X[6], x7 = X[7], x8 = X[8];
    Jq[q] = x0*(x4*x8 - x5*x7) - x1*(x3*x8 - x5*x6) + x2*(x3*x7 - x4*x6);
    for (int a=0;a<3;a++) Xq[q*3+a] = xq[a];
}
}

//@@ CJ_SRC 646

__global__ void cj_init(float* cjs, float* cjg, int k) { cjs[k] = 1.f; cjg[k] = 0.f; }
__global__ void cj_min(const double* SC, float* cjs, int k) { float s = (float)SC[14]; if (!(s == s)) s = 0.f; cjs[k] = fminf(cjs[k], s); }
__global__ void cj_g(const double* SC, const double* CD, float* cjg, int k) { double r = (CD[2] > 0.0) ? SC[1]/CD[2] : 0.0; cjg[k] = (r == r) ? (float)fmin(r, 3.0e38) : 3.0e38f; }
__global__ void cj_end(const int* CT, const float* cjs, const float* cjg, int* cjl, int k, int adapt, int res) {
    bool cut = cjs[k] < 1.f, big = res && cjg[k] > 1.f;
    cjl[k] = ((adapt ? (CT[0] == 0) : 1) && (cut || big)) ? 1 : 0; }

//@@ an_det_DET_DECL 1296

__device__ int AO_DET;
__device__ double* AO_SCB;
__device__ double* AO_PART;
__device__ int* AO_BMAX;
__device__ double* AO_CF;                                                         // 칸 x 27 x 3 (f64 힘)
__device__ float* AO_CB;                                                          // 칸 x 27 x 9 (노드 대각)
__device__ float* AO_CV;                                                          // 칸 x 27 x 3 (헤시안 곱)
__device__ float* AO_MW;                                                          // 점별 고착 질량
__device__ double* AO_SCM;                                                        // MG 거듭제곱법 스칼라 (부분합 칸 64 이후)
__device__ float* AO_MGB;                                                         // 거친 층 칸 x 27 x 9 (노드 대각)
__device__ float* AO_MGV;                                                         // 거친 층 칸 x 27 x 3 (헤시안 곱)
__device__ __forceinline__ void ao_bs_out(double* out, double v) {
    long long sl = out - AO_SCB;
    if (AO_DET && sl >= 0 && sl < 64) { AO_PART[sl*65536 + blockIdx.x] = v; atomicMax(&AO_BMAX[sl], (int)blockIdx.x + 1); return; }
    long long sm = AO_SCM ? (out - AO_SCM) : -1;
    if (AO_DET && AO_SCM && sm >= 0 && sm < 32) { AO_PART[(64+sm)*65536 + blockIdx.x] = v; atomicMax(&AO_BMAX[64+sm], (int)blockIdx.x + 1); return; }
    atomicAdd(out, v);
}

//@@ an_det_DET_SRC 12503

__global__ void ao_bsfin() {
    int sl = blockIdx.x;
    int nb = AO_BMAX[sl];
    if (nb == 0) return;
    __shared__ double sh[256];
    double s = 0.0;
    for (int b = threadIdx.x; b < nb; b += 256) s += AO_PART[(size_t)sl*65536 + b];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) { if (sl < 64) AO_SCB[sl] += sh[0]; else AO_SCM[sl-64] += sh[0]; AO_BMAX[sl] = 0; }
}
// 노드 -> (이웃 칸 c, 그 칸에서 이 노드의 자리 o) 를 고정 순서로: o = (i,j,k), 칸 좌표 = 노드 좌표 − (i,j,k)
__device__ __forceinline__ int ao_ncell(const int* IP, const int* cidx, int n, int o) {
    int nb1 = IP[7], nb2 = IP[8];
    int ni = n / (nb1*nb2), nj = (n / nb2) % nb1, nk = n % nb2;
    int ci = ni - o/9, cj = nj - (o/3)%3, ck = nk - o%3;
    int c1 = nb1 - 2, c2 = nb2 - 2;
    if (ci < 0 || cj < 0 || ck < 0 || ci >= IP[6]-2 || cj >= c1 || ck >= c2) return -1;
    return cidx[(ci*c1 + cj)*c2 + ck];
}
__device__ int* AO_ACTL;                                                          // 압축 판이면 질량 있는 노드 목록 (IP[14] 개), 아니면 null
__device__ __forceinline__ int ao_node(const int* IP, int i) {
    if (AO_ACTL) return (i < IP[14]) ? AO_ACTL[i] : -1;
    return (i < IP[9]) ? i : -1;
}
__global__ void ao_gath_d3(const int* IP, const int* cidx, double* out) {
    int n = ao_node(IP, blockIdx.x*blockDim.x + threadIdx.x);
    if (n < 0) return;
    double s[3] = {0.0, 0.0, 0.0};
    for (int o=0;o<27;o++) { int c = ao_ncell(IP, cidx, n, o); if (c < 0) continue; for (int a=0;a<3;a++) s[a] += AO_CF[((size_t)c*27+o)*3+a]; }
    for (int a=0;a<3;a++) out[n*3+a] += s[a];
}
__global__ void ao_gath_f(const int* IP, const int* cidx, int D, const float* buf, float* out) {
    int n = ao_node(IP, blockIdx.x*blockDim.x + threadIdx.x);
    if (n < 0) return;
    float s[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { int c = ao_ncell(IP, cidx, n, o); if (c < 0) continue; for (int a=0;a<D;a++) s[a] += buf[((size_t)c*27+o)*D+a]; }
    for (int a=0;a<D;a++) out[n*D+a] += s[a];
}
// 점 값 x 기저 (BW) 를 노드가 모음: mode 0 = mq (rt_gauss 의 mqn), 1 = AO_MW (rt_floor_q 의 an)
__global__ void ao_gath_q(const int* IP, const int* cidx, const float* BW, const float* qv, float* out) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float s = 0.f;
    for (int o=0;o<27;o++) { int c = ao_ncell(IP, cidx, n, o); if (c < 0) continue; for (int oc=0;oc<8;oc++) s += BW[oc*27+o]*qv[(size_t)c*8+oc]; }
    out[n] += s;
}
// 절 CM9 --gq_mass 결정적 판: 노드가 이웃 27 칸 x 8 점의 (m_q, p_q) x 기저를 고정 순서로 (gq_q2n 과 같은 식)
__global__ void gq_gath(const int* IP, const int* cidx, const float* BW, const float* mq, const float* PQ, const float* PC, float kap, int oct, float* mn, float* mv, float* ma) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    int na = ma ? 6 : 3;
    float sm = 0.f, sp[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int o=0;o<27;o++) { int c = ao_ncell(IP, cidx, n, o); if (c < 0) continue;
        for (int oc=0;oc<8;oc++) { int q = c*8 + oc; float w = BW[oc*27+o]; float pq[6]; gq_pq(q, PQ, PC, kap, oct, na, pq);
            sm += w*mq[q]; for (int a=0;a<na;a++) sp[a] += w*pq[a]; } }
    mn[n] += sm; for (int a=0;a<3;a++) mv[n*3+a] += sp[a];
    if (ma) for (int a=0;a<3;a++) ma[n*3+a] += sp[3+a];
}
// 노드 목록 (번호 순서, 한 블록 스캔)
__global__ void ao_act_scan(int* IP, const float* mn, int* act) {
    __shared__ int sh[1024];
    int n = IP[9], carry = 0;
    for (int base = 0; base < n; base += 1024) {
        int i = base + threadIdx.x;
        int v = (i < n && mn[i] > 0.f) ? 1 : 0;
        sh[threadIdx.x] = v;
        __syncthreads();
        for (int o=1; o<1024; o<<=1) { int t = (threadIdx.x >= o) ? sh[threadIdx.x - o] : 0; __syncthreads(); sh[threadIdx.x] += t; __syncthreads(); }
        if (v) act[carry + sh[threadIdx.x] - 1] = i;
        int tot = sh[1023];
        __syncthreads();
        carry += tot;
    }
    if (threadIdx.x == 0) IP[14] = carry;
}
// 정렬된 입자의 칸 구간 시작 st[c] (st[칸 수] = N)
__global__ void ao_seg(int N, const int* IP, const int* pcd, const int* cidx, int* st) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p > N || IP[13]) return;
    if (p == N) { st[IP[11]] = N; return; }
    if (p == 0 || pcd[p] != pcd[p-1]) st[cidx[pcd[p]]] = p;
}
// P2G 칸 몫: 칸 하나가 자기 입자를 차례로 (rt_p2g 와 같은 식)
__global__ void ao_p2g_q(int N, const float* F, const float* m, const float* vol, const float* fx, const int* st, const int* IP, float* A, float* C, int* badq, int* badc) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    for (int p = st[c]; p < st[c+1]; p++) {
        int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
        int q = c*8 + oc;
        float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
        float U[9], S[3], V[9];
        ssvdf(Fp, U, S, V);
        float w = vol[p];
        bool ng = !(S[2] > 0.f);
        float R[9], L[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) R[a*3+b] = U[a*3]*V[b*3] + U[a*3+1]*V[b*3+1] + U[a*3+2]*V[b*3+2];
        float l[3]; for (int i=0;i<3;i++) l[i] = ng ? 0.f : logf(S[i]);
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) L[a*3+b] = V[a*3]*l[0]*V[b*3] + V[a*3+1]*l[1]*V[b*3+1] + V[a*3+2]*l[2]*V[b*3+2];
        float* aq = A + (size_t)q*28; float* ac = C + (size_t)c*30;
        for (int k=0;k<9;k++) { aq[k] += w*R[k]; aq[9+k] += w*L[k]; aq[18+k] += w*Fp[k]; ac[k] += w*R[k]; ac[9+k] += w*L[k]; ac[18+k] += w*Fp[k]; }
        aq[27] += w; ac[27] += w; ac[28] += w; ac[29] += m[p];
#ifdef AF_BAR
        for (int a=0;a<3;a++) AF_BX[q*4+a] += m[p]*AF_XP[p*3+a]; AF_BX[q*4+3] += m[p];   // 장벽 판 표본점 (절 AN.19)
#endif
#ifdef AF_OCT
        AF_MO[q] += m[p];
#endif
        if (ng) { badq[q] = 1; badc[c] = 1; }
    }
}
// P2G 노드 몫 (빠른 판): 칸 하나가 자기 입자의 27 노드 몫 (질량, 운동량 3) 을 고정 순서로 더해 칸 버퍼 [칸 x 27 x 4] 에 -> ao_gath_mv 가 노드마다 모음
__global__ void ao_p2g_c(const int* IP, const int* st, const float* fx, const float* m, const float* v, float* cbuf) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    float acc[27][4];
    for (int o=0;o<27;o++) for (int a=0;a<4;a++) acc[o][a] = 0.f;
    for (int p = st[c]; p < st[c+1]; p++) {
        float wx[3][3], dw[3][3];
        bwf(fx[p*3], wx[0], dw[0]); bwf(fx[p*3+1], wx[1], dw[1]); bwf(fx[p*3+2], wx[2], dw[2]);
        float mp = m[p], v0 = v[p*3], v1 = v[p*3+1], v2 = v[p*3+2];
        for (int o=0;o<27;o++) { float ww = wx[0][o/9]*wx[1][(o/3)%3]*wx[2][o%3]*mp; acc[o][0] += ww; acc[o][1] += ww*v0; acc[o][2] += ww*v1; acc[o][3] += ww*v2; }
    }
    for (int o=0;o<27;o++) for (int a=0;a<4;a++) cbuf[((size_t)c*27+o)*4+a] = acc[o][a];
}
__global__ void ao_gath_mv(const int* IP, const int* cidx, const float* cbuf, float* mn, float* mv) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || n >= IP[9]) return;
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    for (int o=0;o<27;o++) { int c = ao_ncell(IP, cidx, n, o); if (c < 0) continue; for (int a=0;a<4;a++) s[a] += cbuf[((size_t)c*27+o)*4+a]; }
    mn[n] += s[0]; for (int a=0;a<3;a++) mv[n*3+a] += s[1+a];
}
// P2G 노드 몫: 노드 하나가 이웃 27 칸의 입자를 차례로 (질량·운동량), 구속 질량 (KS > 0 이면, ao_pbc_acc 와 같은 끊는 표본)
__global__ void ao_p2g_n(const int* IP, const int* cidx, const int* st, const float* fx, const float* m, const float* v, float* mn, float* mv,
                         const float* Trel, const float* tbuf, int K, int S, float dt, float* acc) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || n >= IP[9]) return;
    float sm = 0.f, sv[3] = {0.f, 0.f, 0.f};
    int KS = K*S;
    float t0 = tbuf ? tbuf[0] : 0.f;
    for (int o=0;o<27;o++) {
        int c = ao_ncell(IP, cidx, n, o); if (c < 0) continue;
        int i = o/9, j = (o/3)%3, k = o%3;
        for (int p = st[c]; p < st[c+1]; p++) {
            float wx[3][3], dw[3][3];
            bwf(fx[p*3], wx[0], dw[0]); bwf(fx[p*3+1], wx[1], dw[1]); bwf(fx[p*3+2], wx[2], dw[2]);
            float ww = wx[0][i]*wx[1][j]*wx[2][k]*m[p];
            if (mn) { sm += ww; for (int a=0;a<3;a++) sv[a] += ww*v[p*3+a]; }
            if (acc && Trel[p] > t0) {
                int ncut = 0;
                for (int ks=0; ks<KS; ks++) { int kk = ks / S, ss = ks % S; float ts = t0 + ((float)kk + ((float)ss + 0.5f)/(float)S)*dt; if (Trel[p] > ts) ncut = ks + 1; }
                if (ncut > 0) acc[(size_t)n*KS + ncut - 1] += ww;
            }
        }
    }
    if (mn) { mn[n] += sm; for (int a=0;a<3;a++) mv[n*3+a] += sv[a]; }
}

// ---- MG 결정적 판 ----
// 거친 층 노드가 이웃 27 칸 (조밀 상자) 의 칸 버퍼를 고정 순서로 모음; 빈 칸 (부피 0) 은 건너뜀 (그 칸 워프가 일찍 끝나 버퍼를 안 썼을 수 있음)
__global__ void ao_mg_gath(int l, const int* LM, int D, const float* buf, const float* Vq, float* out) {
    const int* L = LM + l*8;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= L[3]) return;
    int ni = n / (L[1]*L[2]), nj = (n / L[2]) % L[1], nk = n % L[2];
    float s[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) {
        int ci = ni - o/9, cj = nj - (o/3)%3, ck = nk - o%3;
        if (ci < 0 || cj < 0 || ck < 0 || ci >= L[4] || cj >= L[5] || ck >= L[6]) continue;
        int c = (ci*L[5] + cj)*L[6] + ck;
        if (!(Vq[(size_t)c*8] > 0.f)) continue;
        for (int a=0;a<D;a++) s[a] += buf[((size_t)c*27+o)*D+a];
    }
    for (int a=0;a<D;a++) out[n*D+a] += s[a];
}
// 거친 점 하나가 자기에게 사상되는 출처 점 (출처 칸 i ∈ {2b−1, 2b} 각 축, 칸 안 팔분 0..7) 을 고정 순서로 모아 R·LS·F·가중 (rt_mg_acc 와 같은 식)
__global__ void ao_mg_acc_det(int l, const int* IP, const int* LM, const int* cidx, const float* Fsrc, const float* Vsrc, float* A, int* badq) {
    const int* L = LM + l*8;
    int qc = blockIdx.x*blockDim.x + threadIdx.x;
    if (qc >= L[7]) return;
    int cc = qc >> 3, occ = qc & 7;
    int bz = cc % L[6], by = (cc / L[6]) % L[5], bx = cc / (L[5]*L[6]);
    int bb[3] = {bx, by, bz};
    const float GG = 0.28867513459481287f;
    int S0, S1, S2;                                                               // 출처 칸 상자 차원
    if (l == 1) { S0 = IP[6]-2; S1 = IP[7]-2; S2 = IP[8]-2; } else { const int* pl = LM + (l-1)*8; S0 = pl[4]; S1 = pl[5]; S2 = pl[6]; }
    float acc[28]; for (int k=0;k<28;k++) acc[k] = 0.f;
    int bad = 0;
    for (int di=0; di<2; di++) for (int dj=0; dj<2; dj++) for (int dk=0; dk<2; dk++) {
        int ii[3] = {2*bx - 1 + di, 2*by - 1 + dj, 2*bz - 1 + dk};
        if (ii[0] < 0 || ii[1] < 0 || ii[2] < 0 || ii[0] >= S0 || ii[1] >= S1 || ii[2] >= S2) continue;
        int cs = (ii[0]*S1 + ii[1])*S2 + ii[2];
        if (l == 1) { cs = cidx[cs]; if (cs < 0) continue; }
        for (int os=0; os<8; os++) {
            int q = cs*8 + os;
            float w = Vsrc[q]; if (!(w > 0.f)) continue;
            int ok = 1, oc = 0;
            for (int a=0;a<3;a++) {
                int bit = (a == 0) ? (os & 4) : ((a == 1) ? (os & 2) : (os & 1));
                float tq = (float)ii[a] + 1.f + (bit ? GG : -GG);
                float t = 0.5f*(tq + 1.5f); float fb = floorf(t - 0.5f);
                if ((int)fb != bb[a]) ok = 0;
                if (t - fb >= 1.f) oc |= (a == 0) ? 4 : ((a == 1) ? 2 : 1);
            }
            if (!ok || oc != occ) continue;
            float X[9]; for (int k=0;k<9;k++) X[k] = Fsrc[(size_t)q*9+k];
            float U[9], S[3], V[9]; ssvdf(X, U, S, V);
            bool ng = !(S[2] > 0.f);
            float lg[3]; for (int i=0;i<3;i++) lg[i] = ng ? 0.f : logf(S[i]);
            for (int a=0;a<3;a++) for (int b2=0;b2<3;b2++) {
                acc[a*3+b2] += w*(U[a*3]*V[b2*3] + U[a*3+1]*V[b2*3+1] + U[a*3+2]*V[b2*3+2]);
                acc[9+a*3+b2] += w*(V[a*3]*lg[0]*V[b2*3] + V[a*3+1]*lg[1]*V[b2*3+1] + V[a*3+2]*lg[2]*V[b2*3+2]);
                acc[18+a*3+b2] += w*X[a*3+b2]; }
            acc[27] += w;
            if (ng) bad = 1;
        }
    }
    for (int k=0;k<28;k++) A[(size_t)qc*28+k] = acc[k];
    badq[qc] = bad;
}
// 거친 칸 합 = 그 칸 8 점 합 (고정 순서), V 합 = 가중 합
__global__ void ao_mg_cell(int l, const int* LM, const float* A, const int* badq, float* C, int* badc) {
    const int* L = LM + l*8;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= L[7]/8) return;
    float s[28]; for (int k=0;k<28;k++) s[k] = 0.f;
    int bd = 0;
    for (int oc=0; oc<8; oc++) { const float* a = A + ((size_t)c*8+oc)*28; for (int k=0;k<28;k++) s[k] += a[k]; bd |= badq[c*8+oc]; }
    for (int k=0;k<28;k++) C[(size_t)c*30+k] = s[k];
    C[(size_t)c*30+28] = s[27];
    badc[c] = bd;
}

//@@ an_rt_v0_SRC 59856

#define NQ8 8
extern "C" {
__device__ __forceinline__ float seg8f(float s) {
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2); s += __shfl_xor_sync(0xffffffffu, s, 4); return s;
}
__device__ __forceinline__ double seg8d(double s) {
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2); s += __shfl_xor_sync(0xffffffffu, s, 4); return s;
}
__device__ __forceinline__ double warpsum(double s) {
    for (int o=16;o>0;o>>=1) s += __shfl_xor_sync(0xffffffffu, s, o); return s;
}
// 블록 합 (f64) -> 원자 합 하나
__device__ void blocksum_add(double v, double* out) {
    __shared__ double sh[32];
    v = warpsum(v);
    int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (l == 0) sh[w] = v;
    __syncthreads();
    if (w == 0) { v = (l < (blockDim.x >> 5)) ? sh[l] : 0.0; v = warpsum(v); if (l == 0) atomicAdd(out, v); }
    __syncthreads();
}
__device__ void jac3f(float a[3][3], float v[3][3]) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) v[i][j] = (i==j) ? 1.f : 0.f;
    for (int sw=0; sw<20; sw++) {
        float off = a[0][1]*a[0][1] + a[0][2]*a[0][2] + a[1][2]*a[1][2];
        float dg = a[0][0]*a[0][0] + a[1][1]*a[1][1] + a[2][2]*a[2][2];
        if (!(off > 1e-14f*dg)) break;
        for (int p=0;p<2;p++) for (int q=p+1;q<3;q++) {
            if (a[p][q] == 0.f) continue;
            float th = 0.5f*(a[q][q]-a[p][p])/a[p][q];
            float t = (th >= 0.f ? 1.f : -1.f)/(fabsf(th) + sqrtf(th*th + 1.f));
            float c = rsqrtf(t*t + 1.f), s = t*c;
            for (int k=0;k<3;k++) { float x1 = a[k][p], x2 = a[k][q]; a[k][p] = c*x1 - s*x2; a[k][q] = s*x1 + c*x2; }
            for (int k=0;k<3;k++) { float x1 = a[p][k], x2 = a[q][k]; a[p][k] = c*x1 - s*x2; a[q][k] = s*x1 + c*x2; }
            for (int k=0;k<3;k++) { float x1 = v[k][p], x2 = v[k][q]; v[k][p] = c*x1 - s*x2; v[k][q] = s*x1 + c*x2; }
        }
    }
}
// 부호 있는 SVD (f32), fem_an_cuda.ssvd 와 같은 식
__device__ void ssvdf(const float* X, float* U, float* S, float* V) {
    float A[3][3], v[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) { float t = 0.f; for (int k=0;k<3;k++) t += X[k*3+i]*X[k*3+j]; A[i][j] = t; }
    jac3f(A, v);
    int o[3] = {0,1,2}; float e[3] = {A[0][0], A[1][1], A[2][2]};
    if (e[o[0]] < e[o[1]]) { int t = o[0]; o[0] = o[1]; o[1] = t; }
    if (e[o[1]] < e[o[2]]) { int t = o[1]; o[1] = o[2]; o[2] = t; }
    if (e[o[0]] < e[o[1]]) { int t = o[0]; o[0] = o[1]; o[1] = t; }
    for (int i=0;i<3;i++) for (int c=0;c<3;c++) V[i*3+c] = v[i][o[c]];
    float dv = V[0]*(V[4]*V[8]-V[5]*V[7]) - V[1]*(V[3]*V[8]-V[5]*V[6]) + V[2]*(V[3]*V[7]-V[4]*V[6]);
    if (dv < 0.f) for (int i=0;i<3;i++) V[i*3+2] = -V[i*3+2];
    float b[3][3];
    for (int c=0;c<3;c++) for (int a=0;a<3;a++) { float t = 0.f; for (int k=0;k<3;k++) t += X[a*3+k]*V[k*3+c]; b[c][a] = t; }
    float u1[3], u2[3], u3[3];
    float n1 = sqrtf(b[0][0]*b[0][0] + b[0][1]*b[0][1] + b[0][2]*b[0][2]);
    if (n1 > 0.f) for (int a=0;a<3;a++) u1[a] = b[0][a]/n1; else { u1[0] = 1.f; u1[1] = 0.f; u1[2] = 0.f; }
    float d12 = u1[0]*b[1][0] + u1[1]*b[1][1] + u1[2]*b[1][2];
    for (int a=0;a<3;a++) u2[a] = b[1][a] - d12*u1[a];
    float n2 = sqrtf(u2[0]*u2[0] + u2[1]*u2[1] + u2[2]*u2[2]);
    if (n2 > 1e-6f*n1 && n2 > 0.f) { for (int a=0;a<3;a++) u2[a] /= n2; }
    else {
        int ax = 0; if (fabsf(u1[1]) < fabsf(u1[ax])) ax = 1; if (fabsf(u1[2]) < fabsf(u1[ax])) ax = 2;
        float ea[3] = {0.f, 0.f, 0.f}; ea[ax] = 1.f;
        u2[0] = u1[1]*ea[2]-u1[2]*ea[1]; u2[1] = u1[2]*ea[0]-u1[0]*ea[2]; u2[2] = u1[0]*ea[1]-u1[1]*ea[0];
        float nn = sqrtf(u2[0]*u2[0] + u2[1]*u2[1] + u2[2]*u2[2]); for (int a=0;a<3;a++) u2[a] /= nn;
    }
    u3[0] = u1[1]*u2[2]-u1[2]*u2[1]; u3[1] = u1[2]*u2[0]-u1[0]*u2[2]; u3[2] = u1[0]*u2[1]-u1[1]*u2[0];
    for (int a=0;a<3;a++) { U[a*3] = u1[a]; U[a*3+1] = u2[a]; U[a*3+2] = u3[a]; }
    S[0] = u1[0]*b[0][0] + u1[1]*b[0][1] + u1[2]*b[0][2];
    S[1] = u2[0]*b[1][0] + u2[1]*b[1][1] + u2[2]*b[1][2];
    S[2] = u3[0]*b[2][0] + u3[1]*b[2][1] + u3[2]*b[2][2];
}
__device__ __forceinline__ void bwf(float f, float* w, float* d) {
    w[0] = 0.5f*(1.5f-f)*(1.5f-f); w[1] = 0.75f-(f-1.f)*(f-1.f); w[2] = 0.5f*(f-0.5f)*(f-0.5f);
    d[0] = f-1.5f; d[1] = -2.f*(f-1.f); d[2] = f-0.5f;
}
// IP (int): 0..2 bmin, 3..5 bmax, 6..8 nb (노드 상자), 9 노드 수, 10 칸 상자 수, 11 압축 칸 수 nc, 12 가우스 점 수 nq, 13 넘침
// ---- P2G ----
__global__ void rt_reset(int* IP) { IP[0] = IP[1] = IP[2] = 2147483647; IP[3] = IP[4] = IP[5] = -2147483647; IP[13] = 0; IP[14] = 0; }
__global__ void rt_coord(int N, const float* x, const float* Ai, int* pb, float* fx, int* IP) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    for (int a=0;a<3;a++) {
        float y = Ai[a*3]*x[p*3] + Ai[a*3+1]*x[p*3+1] + Ai[a*3+2]*x[p*3+2];
        float fb = floorf(y - 0.5f);
        int b = (int)fb; pb[p*3+a] = b; fx[p*3+a] = y - fb;
        atomicMin(&IP[a], b); atomicMax(&IP[3+a], b);
    }
}
__global__ void rt_box(int* IP, int capN, int capC) {
    int nb0 = IP[3]-IP[0]+3, nb1 = IP[4]-IP[1]+3, nb2 = IP[5]-IP[2]+3;
    IP[6] = nb0; IP[7] = nb1; IP[8] = nb2;
    long long nn = (long long)nb0*nb1*nb2, nc = (long long)(nb0-2)*(nb1-2)*(nb2-2);
    IP[9] = (int)nn; IP[10] = (int)nc;
    if (nn > capN || nc > capC) { IP[13] = 1; IP[9] = 0; IP[10] = 0; }
}
__global__ void rt_zero_f(int n, const int* cnt, int ci, int mul, float* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; int m = cnt ? cnt[ci]*mul : n; if (i < m && i < n) a[i] = 0.f; }
__global__ void rt_zero_d(int n, const int* cnt, int ci, int mul, double* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; int m = cnt ? cnt[ci]*mul : n; if (i < m && i < n) a[i] = 0.0; }
__global__ void rt_zero_i(int n, int* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = 0; }
// 입자: 상자 안 base, 칸 상자 번호 표시
__global__ void rt_mark(int N, int* pb, const float* fx, const int* IP, int* pcd, int* occ) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int b0 = pb[p*3]-IP[0], b1 = pb[p*3+1]-IP[1], b2 = pb[p*3+2]-IP[2];
    pb[p*3] = b0; pb[p*3+1] = b1; pb[p*3+2] = b2;
    int cd = (b0*(IP[7]-2) + b1)*(IP[8]-2) + b2;
    pcd[p] = cd; occ[cd] = 1;
}
// 한 블록 배타 스캔: occ -> cidx (압축 번호, 비점유 -1), 칸 base 좌표 cb, nc·nq
__global__ void rt_scan(int* IP, const int* occ, int* cidx, int* cb, int capQc) {
    __shared__ int sh[1024];
    int n = IP[13] ? 0 : IP[10];
    int carry = 0;
    int n1 = IP[7]-2, n2 = IP[8]-2;
    for (int base = 0; base < n; base += 1024) {
        int i = base + threadIdx.x;
        int v = (i < n) ? occ[i] : 0;
        sh[threadIdx.x] = v;
        __syncthreads();
        for (int o=1; o<1024; o<<=1) { int t = (threadIdx.x >= o) ? sh[threadIdx.x - o] : 0; __syncthreads(); sh[threadIdx.x] += t; __syncthreads(); }
        int inc = sh[threadIdx.x];
        if (i < n) {
            int id = carry + inc - v;
            if (v) { if (id < capQc) { cidx[i] = id; cb[id*3] = i / (n1*n2); cb[id*3+1] = (i / n2) % n1; cb[id*3+2] = i % n2; } }
            else cidx[i] = -1;
        }
        int tot = sh[1023];
        __syncthreads();
        carry += tot;
    }
    if (threadIdx.x == 0) { IP[11] = carry; IP[12] = carry*8; if (carry > capQc) { IP[13] = 1; IP[11] = 0; IP[12] = 0; } }
}
// 입자 -> 가우스 점 누적 (회전 R, log S, F, 가중) + 칸 누적 + 칸 부피·질량 + 노드 질량·운동량
// A: [q*28]: 0..8 R, 9..17 LS, 18..26 F, 27 w;  C: [c*30]: 0..27 같은 것, 28 V, 29 M;  bad: q 와 c (int, 뒤집힌 입자)
__global__ void rt_p2g(int N, const float* x, const float* v, const float* F, const float* m, const float* vol, const int* pb, const float* fx, const int* pcd,
                       const int* cidx, const int* IP, float* A, float* C, int* badq, int* badc, float* mn, float* mv) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
    int q = c*8 + oc;
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    float U[9], S[3], V[9];
    ssvdf(Fp, U, S, V);
    float w = vol[p];
    bool ng = !(S[2] > 0.f);
    float R[9], L[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) R[a*3+b] = U[a*3]*V[b*3] + U[a*3+1]*V[b*3+1] + U[a*3+2]*V[b*3+2];
    float l[3]; for (int i=0;i<3;i++) l[i] = ng ? 0.f : logf(S[i]);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) L[a*3+b] = V[a*3]*l[0]*V[b*3] + V[a*3+1]*l[1]*V[b*3+1] + V[a*3+2]*l[2]*V[b*3+2];
    float* aq = A + (size_t)q*28; float* ac = C + (size_t)c*30;
    for (int k=0;k<9;k++) { atomicAdd(&aq[k], w*R[k]); atomicAdd(&aq[9+k], w*L[k]); atomicAdd(&aq[18+k], w*Fp[k]);
                            atomicAdd(&ac[k], w*R[k]); atomicAdd(&ac[9+k], w*L[k]); atomicAdd(&ac[18+k], w*Fp[k]); }
    atomicAdd(&aq[27], w); atomicAdd(&ac[27], w); atomicAdd(&ac[28], w); atomicAdd(&ac[29], m[p]);
    if (ng) { atomicOr(&badq[q], 1); atomicOr(&badc[c], 1); }
    float wx[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], wx[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float mp = m[p];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = wx[0][i]*wx[1][j]*wx[2][k]*mp;
        atomicAdd(&mn[nid], ww);
        for (int a=0;a<3;a++) atomicAdd(&mv[nid*3+a], ww*v[p*3+a]);
    }
}
__device__ void rs_fin(const float* sR, const float* sL, float w, float* out, bool* bad) {
    float U[9], S[3], V[9];
    ssvdf(sR, U, S, V);
    float Rb[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Rb[a*3+b] = U[a*3]*V[b*3] + U[a*3+1]*V[b*3+1] + U[a*3+2]*V[b*3+2];
    float Lm[3][3], Ev[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) Lm[i][j] = 0.5f*(sL[i*3+j] + sL[j*3+i])/w;
    jac3f(Lm, Ev);
    float ex[3]; for (int i=0;i<3;i++) ex[i] = expf(Lm[i][i]);
    float Sb[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Sb[a*3+b] = Ev[a][0]*ex[0]*Ev[b][0] + Ev[a][1]*ex[1]*Ev[b][1] + Ev[a][2]*ex[2]*Ev[b][2];
    *bad = false;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = Rb[a*3]*Sb[b] + Rb[a*3+1]*Sb[3+b] + Rb[a*3+2]*Sb[6+b]; out[a*3+b] = t; if (!isfinite(t)) *bad = true; }
}
// 가우스 점 마무리: F_q, V_q, m_q, 접촉용 시작 위치 xq0 (월드), 노드 몫 질량 mqn (원자 합)
__global__ void rt_gauss(const int* IP, const float* A, const float* C, const int* badq, const int* badc, const int* cb, const float* Am, const float* BW,
                         float* Fq, float* Vq, float* mq, float* xq0, float* mqn, int* nbad) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int c = q >> 3, oc = q & 7;
    const float* aq = A + (size_t)q*28; const float* ac = C + (size_t)c*30;
    float out[9]; bool bad = false;
    if (aq[27] > 0.f) {
        rs_fin(aq, aq + 9, aq[27], out, &bad);
        if (bad || badq[q]) { for (int k=0;k<9;k++) out[k] = aq[18+k]/aq[27]; atomicAdd(nbad, 1); }
    } else {
        rs_fin(ac, ac + 9, ac[27], out, &bad);
        if (bad || badc[c]) for (int k=0;k<9;k++) out[k] = ac[18+k]/ac[27];
    }
    for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = out[k];
    Vq[q] = ac[28]*0.125f; float mm = ac[29]*0.125f; mq[q] = mm;
    const float GG = 0.28867513459481287f;
    float yq[3] = {(float)(IP[0] + cb[c*3]) + 1.f + ((oc & 4) ? GG : -GG), (float)(IP[1] + cb[c*3+1]) + 1.f + ((oc & 2) ? GG : -GG), (float)(IP[2] + cb[c*3+2]) + 1.f + ((oc & 1) ? GG : -GG)};
    for (int a=0;a<3;a++) xq0[q*3+a] = Am[a*3]*yq[0] + Am[a*3+1]*yq[1] + Am[a*3+2]*yq[2];
    int nb1 = IP[7], nb2 = IP[8];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((cb[c*3]+i)*nb1 + (cb[c*3+1]+j))*nb2 + (cb[c*3+2]+k);
        atomicAdd(&mqn[nid], BW[oc*27+o]*mm);
    }
}
__global__ void rt_vn(const int* IP, const float* mn, const float* mv, float* vk, float* U) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n];
    for (int a=0;a<3;a++) { vk[n*3+a] = (m > 0.f) ? mv[n*3+a]/m : 0.f; U[n*3+a] = 0.f; }
}
// ---- 풀이 ----
// 가우스 점 공통: 27 노드 번호, G_qn = F_qᵀ ∇w (BG 월드 기울기 표)
__device__ __forceinline__ void qnodes(int q, const int* IP, const int* cb, int* ids) {
    int c = q >> 3; int nb1 = IP[7], nb2 = IP[8];
    for (int o=0;o<27;o++) { int i = o/9, j = (o/3)%3, k = o%3; ids[o] = ((cb[c*3]+i)*nb1 + (cb[c*3+1]+j))*nb2 + (cb[c*3+2]+k); }
}
__device__ __forceinline__ void gqn(const float* Fq, const float* bg, float* g) {
    g[0] = Fq[0]*bg[0] + Fq[3]*bg[1] + Fq[6]*bg[2];
    g[1] = Fq[1]*bg[0] + Fq[4]*bg[1] + Fq[7]*bg[2];
    g[2] = Fq[2]*bg[0] + Fq[5]*bg[1] + Fq[8]*bg[2];
}
// 걸음 시작: Ut = U + τ v + τ² g, z0 = U + β τ v, y = Ut (고정 노드는 U + τ vbc)
__global__ void rt_step0(const int* IP, const float* U, const float* vk, float tau, float beta, float g0, float g1, float g2, const unsigned char* fixd, const float* vbc,
                         float* Ut, float* z0, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float gg[3] = {g0, g1, g2};
    for (int a=0;a<3;a++) {
        float u = U[n*3+a], vv = vk[n*3+a];
        float ut = u + tau*vv + tau*tau*gg[a];
        Ut[n*3+a] = ut; z0[n*3+a] = u + beta*tau*vv;
        y[n*3+a] = (fixd && fixd[n]) ? u + tau*vbc[n*3+a] : ut;
    }
}
// 바닥: 가우스 점 걸음 끝 위치 -> 정답 sticky 가중 (s < 0 만) -> 노드 몫 an (원자 합)
__global__ void rt_floor_q(const int* IP, const int* cb, const float* BW, const float* xq0, const float* mq, const float* y, const float* U, float thinv,
                           float p0, float p1, float p2, float n0, float n1, float n2, float pn, float dxg, float* an) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xq[3] = {xq0[q*3], xq0[q*3+1], xq0[q*3+2]};
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
        for (int a=0;a<3;a++) { float ye = U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a]); xq[a] += w*ye; } }
    float s = (xq[0]-p0)*n0 + (xq[1]-p1)*n1 + (xq[2]-p2)*n2;
    if (!(s < 0.f)) return;
    float u = (s + pn)/dxg; float bf = floorf(u - 0.5f); float f = u - bf;
    float w3[3] = {0.5f*(1.5f-f)*(1.5f-f), 0.75f-(f-1.f)*(f-1.f), 0.5f*(f-0.5f)*(f-0.5f)};
    float wst = 0.f; for (int i=0;i<3;i++) if ((bf + (float)i)*dxg - pn < 0.f) wst += w3[i];
    float mw = mq[q]*wst;
    if (mw > 0.f) for (int o=0;o<27;o++) atomicAdd(&an[ids[o]], BW[oc*27+o]*mw);
}
// 노드: 고착 인자 k, 변수 바꿈 (sk, pm), ŷ, η, y = ŷ + T η. pm 0 = P I, 1 = P (I − nnᵀ) (바깥 법선 이탈), 2 = 고정 (T 0)
__device__ __forceinline__ void Tapply(float sk, int pm, float n0, float n1, float n2, const float* v, float* o) {
    if (pm == 2) { o[0] = o[1] = o[2] = 0.f; return; }
    if (pm == 0) { o[0] = sk*v[0]; o[1] = sk*v[1]; o[2] = sk*v[2]; return; }
    float vn = v[0]*n0 + v[1]*n1 + v[2]*n2;                                   // P v = v − (v·n) n (접선), (I − P) v = (v·n) n
    o[0] = sk*(v[0]-vn*n0) + vn*n0; o[1] = sk*(v[1]-vn*n1) + vn*n1; o[2] = sk*(v[2]-vn*n2) + vn*n2;
}
__global__ void rt_floor_n(const int* IP, const float* an, const float* mqn, float kex, int floor_, int vout, float al, float n0, float n1, float n2,
                           const unsigned char* fixd, const float* vbc, const float* U, float tau, const float* Ut, const float* z0, float* y, float* sk, int* pm, float* yh, float* eta) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float k = 1.f; int p = 0;
    if (floor_) {
        float r = (mqn[n] > 0.f) ? fminf(an[n]/mqn[n], 1.f) : 0.f;
        k = expf(kex*log1pf(-r));
        float vv = al*((y[n*3]-z0[n*3])*n0 + (y[n*3+1]-z0[n*3+1])*n1 + (y[n*3+2]-z0[n*3+2])*n2);
        p = (vout && vv > 0.f) ? 1 : 0;
    }
    if (fixd && fixd[n]) p = 2;
    float s = sqrtf(k);
    float d[3], yy[3];
    for (int a=0;a<3;a++) d[a] = z0[n*3+a] - Ut[n*3+a];
    if (p == 2) { for (int a=0;a<3;a++) { yy[a] = U[n*3+a] + tau*vbc[n*3+a]; yh[n*3+a] = yy[a]; eta[n*3+a] = 0.f; y[n*3+a] = yy[a]; } sk[n] = 0.f; pm[n] = 2; return; }
    float Pd[3];
    if (p == 0) { for (int a=0;a<3;a++) Pd[a] = d[a]; }
    else { float dn = d[0]*n0 + d[1]*n1 + d[2]*n2; Pd[0] = d[0]-dn*n0; Pd[1] = d[1]-dn*n1; Pd[2] = d[2]-dn*n2; }
    float h[3]; for (int a=0;a<3;a++) h[a] = Ut[n*3+a] + (1.f-k)*Pd[a];
    float e1[3], e2[3];
    for (int a=0;a<3;a++) { e1[a] = y[n*3+a] - Ut[n*3+a]; e2[a] = y[n*3+a] - h[a]; }
    float et[3];
    if (p == 0) { for (int a=0;a<3;a++) et[a] = s*e2[a]; }
    else { float c1 = e1[0]*n0 + e1[1]*n1 + e1[2]*n2; float c2 = e2[0]*n0 + e2[1]*n1 + e2[2]*n2;
           for (int a=0;a<3;a++) { float nn_ = (a==0) ? n0 : ((a==1) ? n1 : n2); et[a] = c1*nn_ + s*(e2[a] - c2*nn_); } }
    float ty[3]; Tapply(s, p, n0, n1, n2, et, ty);
    for (int a=0;a<3;a++) { yh[n*3+a] = h[a]; eta[n*3+a] = et[a]; y[n*3+a] = h[a] + ty[a]; }
    sk[n] = s; pm[n] = p;
}
// 요소: mode 1 기울기 (f64 원자, 8 레인 구간 합), 2 헤시안 (f32 저장), 4 에너지 (f64 블록 합)
__global__ void __launch_bounds__(128) rt_elem(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, float mu, float lam, int mode,
                        double* E, double* f, float* H, int capQ, float* B) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    int nq = IP[12];
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = y[id*3+a]; X[a*3] += ya*g[0]; X[a*3+1] += ya*g[1]; X[a*3+2] += ya*g[2]; } }
    float U[9], S[3], V[9];
    ssvdf(X, U, S, V);
    float J = S[0]*S[1]*S[2];
    float gJ[3] = {S[1]*S[2], S[0]*S[2], S[0]*S[1]};
    float vq = on ? Vq[qq] : 0.f;
    if (mode & 4) {
        double e = on ? (double)vq*((double)mu*(((double)S[0]-1.0)*((double)S[0]-1.0) + ((double)S[1]-1.0)*((double)S[1]-1.0) + ((double)S[2]-1.0)*((double)S[2]-1.0)) + 0.5*(double)lam*((double)J-1.0)*((double)J-1.0)) : 0.0;
        blocksum_add(e, E);
    }
    if (mode & 1) {
        float ph[3]; for (int i=0;i<3;i++) ph[i] = 2.f*mu*(S[i]-1.f) + lam*(J-1.f)*gJ[i];
        float P[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) P[a*3+b] = U[a*3]*ph[0]*V[b*3] + U[a*3+1]*ph[1]*V[b*3+1] + U[a*3+2]*ph[2]*V[b*3+2];
        for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
            for (int a=0;a<3;a++) { double s = seg8d((double)(vq*(P[a*3]*g[0] + P[a*3+1]*g[1] + P[a*3+2]*g[2])));
                if (on && head) atomicAdd(&f[ids[o]*3+a], s); } }
    }
    if (mode & 2) {                                                             // 헤시안 (모든 레인이 들어옴: 대각 구간 합 때문에; 밖 레인은 0)
        float h[81]; for (int k=0;k<81;k++) h[k] = 0.f;
        if (on) {
        float D[3][3], Ev[3][3];
        float Sg[3][3] = {{0.f, S[2], S[1]}, {S[2], 0.f, S[0]}, {S[1], S[0], 0.f}};
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = ((i==j) ? 2.f*mu : 0.f) + lam*gJ[i]*gJ[j] + lam*(J-1.f)*Sg[i][j];
        jac3f(D, Ev);
        for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (!(dm > 0.f)) continue;
            float e[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) e[a*3+b] = Ev[0][m]*U[a*3]*V[b*3] + Ev[1][m]*U[a*3+1]*V[b*3+1] + Ev[2][m]*U[a*3+2]*V[b*3+2];
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += dm*e[r]*e[c]; }
        const int PI[3] = {0,0,1}, PJ[3] = {1,2,2}, PK[3] = {2,1,0};
        const float r2 = 0.70710678f;
        for (int pp=0; pp<3; pp++) { int i = PI[pp], j = PJ[pp], k = PK[pp];
            float lt = 2.f*mu*(1.f - 2.f/(S[i]+S[j])) + lam*(J-1.f)*S[k]; float lf = 2.f*mu - lam*(J-1.f)*S[k];
            lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = fmaxf(lf, 0.f);
            float et[9], ef[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float x1 = U[a*3+i]*V[b*3+j], x2 = U[a*3+j]*V[b*3+i]; et[a*3+b] = r2*(x1-x2); ef[a*3+b] = r2*(x1+x2); }
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += lt*et[r]*et[c] + lf*ef[r]*ef[c]; }
        for (int k=0;k<81;k++) H[(size_t)k*capQ + qq] = h[k];                     // 성분 우선 배치 (곱 커널 읽기가 합쳐지게)
        }
        if (mode & 8) {                                                           // 노드 블록 대각 (융합): m_ac = V Σ_b g_b Σ_d H[(ab),(cd)] g_d
            for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
                float m[9];
                for (int a=0;a<3;a++) for (int c=0;c<3;c++) { float t = 0.f;
                    for (int b=0;b<3;b++) { float hg = h[(a*3+b)*9 + c*3]*g[0] + h[(a*3+b)*9 + c*3+1]*g[1] + h[(a*3+b)*9 + c*3+2]*g[2]; t += g[b]*hg; }
                    m[a*3+c] = vq*t; }
                for (int k=0;k<9;k++) m[k] = seg8f(m[k]);
                if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], m[k]); }
        }
    }
}
// 노드 블록 대각 Σ V Σ_bd H[(ab),(cd)] g_b g_d (f32, 8 레인 구간 합)
__global__ void __launch_bounds__(128) rt_bd(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* H, float* B) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    const float* h = H + (size_t)qq*81;
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
        float m[9];
        for (int a=0;a<3;a++) for (int c=0;c<3;c++) { float t = 0.f;
            for (int b=0;b<3;b++) for (int d=0;d<3;d++) t += h[(a*3+b)*9 + c*3+d]*g[b]*g[d];
            m[a*3+c] = vq*t; }
        for (int k=0;k<9;k++) m[k] = seg8f(m[k]);
        if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], m[k]); }
}
// 헤시안 곱 out += Σ V (H δX) g, δX = Σ dy ⊗ g (f32)
__global__ void __launch_bounds__(128) rt_mv(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* H, const float* dy, float* out, int capQ) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float dX[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = dy[id*3+a]; dX[a*3] += ya*g[0]; dX[a*3+1] += ya*g[1]; dX[a*3+2] += ya*g[2]; } }
    float vq = on ? Vq[qq] : 0.f;
    float dP[9]; for (int r=0;r<9;r++) { float t = 0.f; for (int c=0;c<9;c++) t += H[(size_t)(r*9+c)*capQ + qq]*dX[c]; dP[r] = vq*t; }
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
        for (int a=0;a<3;a++) { float s = seg8f(dP[a*3]*g[0] + dP[a*3+1]*g[1] + dP[a*3+2]*g[2]); if (on && head) atomicAdd(&out[ids[o]*3+a], s); } }
}
// 노드: 기울기 g = Mi η + T f (f64), r = −g (f32), 블록 야코비 Dinv = (Mi I + T B T)⁻¹, z = D r, p = z; 스칼라 SC: 0 rz, 1 |g|², ...
__device__ __forceinline__ void inv3(const float* M, float* Mi) {
    float c0 = M[4]*M[8]-M[5]*M[7], c1 = M[5]*M[6]-M[3]*M[8], c2 = M[3]*M[7]-M[4]*M[6];
    float d = M[0]*c0 + M[1]*c1 + M[2]*c2;
    if (d == 0.f || !isfinite(d)) { for (int k=0;k<9;k++) Mi[k] = 0.f; return; }
    float id = 1.f/d;
    Mi[0] = c0*id; Mi[1] = (M[2]*M[7]-M[1]*M[8])*id; Mi[2] = (M[1]*M[5]-M[2]*M[4])*id;
    Mi[3] = c1*id; Mi[4] = (M[0]*M[8]-M[2]*M[6])*id; Mi[5] = (M[2]*M[3]-M[0]*M[5])*id;
    Mi[6] = c2*id; Mi[7] = (M[1]*M[6]-M[0]*M[7])*id; Mi[8] = (M[0]*M[4]-M[1]*M[3])*id;
}
__device__ __forceinline__ void Tmat(float sk, int pm, float n0, float n1, float n2, float* T) {
    float nn[3] = {n0, n1, n2};
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
        float I = (a==b) ? 1.f : 0.f;
        if (pm == 2) T[a*3+b] = 0.f; else if (pm == 0) T[a*3+b] = sk*I; else { float Pab = I - nn[a]*nn[b]; T[a*3+b] = sk*Pab + (I - Pab); }
    }
}
__global__ void rt_grad(const int* IP, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* eta, const double* f,
                        const float* B, float* Dinv, float* r, float* z, float* p, float* dl, float* Tp, double* SC, float* r0) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double rz = 0.0, gg = 0.0;
    if (n < IP[9]) {
        float T[9]; Tmat(sk[n], pm[n], n0, n1, n2, T);
        float Mi = mn[n]*itau2;
        double g[3];
        for (int a=0;a<3;a++) g[a] = (double)Mi*eta[n*3+a] + (double)T[a*3]*f[n*3] + (double)T[a*3+1]*f[n*3+1] + (double)T[a*3+2]*f[n*3+2];
        float M[9], TB[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += T[a*3+k]*B[n*9+k*3+b]; TB[a*3+b] = t; }
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = (a==b) ? Mi : 0.f; for (int k=0;k<3;k++) t += TB[a*3+k]*T[k*3+b]; M[a*3+b] = t; }
        float Di[9]; inv3(M, Di);
        for (int k=0;k<9;k++) Dinv[n*9+k] = Di[k];
        float rr[3]; for (int a=0;a<3;a++) { rr[a] = (float)(-g[a]); r[n*3+a] = rr[a]; r0[n*3+a] = rr[a]; dl[n*3+a] = 0.f; gg += g[a]*g[a]; }
        float zz[3]; for (int a=0;a<3;a++) { zz[a] = Di[a*3]*rr[0] + Di[a*3+1]*rr[1] + Di[a*3+2]*rr[2]; z[n*3+a] = zz[a]; p[n*3+a] = zz[a]; rz += (double)rr[a]*zz[a]; }
        float tp[3]; Tapply(sk[n], pm[n], n0, n1, n2, zz, tp); for (int a=0;a<3;a++) Tp[n*3+a] = tp[a];
    }
    blocksum_add(rz, &SC[0]); blocksum_add(gg, &SC[1]);
}
// Ap = Mi p + T (K T p) 의 노드 마무리 + p·Ap (SC[2])
__global__ void rt_ap(const int* IP, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* p, const float* KTp, float* Ap, double* SC) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s = 0.0;
    if (n < IP[9]) {
        float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, KTp + n*3, t);
        float Mi = mn[n]*itau2;
        for (int a=0;a<3;a++) { float v = Mi*p[n*3+a] + t[a]; Ap[n*3+a] = v; s += (double)p[n*3+a]*v; }
    }
    blocksum_add(s, &SC[2]);
}
// α = rz/pAp; dl += α p; r −= α Ap; zo = z; z = D r; SC[3] += r·z, SC[4] += r·(z − zo)
__global__ void rt_cg1(const int* IP, const double* SC, const float* Dinv, const float* p, const float* Ap, float* dl, float* r, float* z, double* SC2) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s3 = 0.0, s4 = 0.0;
    if (n < IP[9]) {
        double pAp = SC[2]; float al = (pAp > 0.0) ? (float)(SC[0]/pAp) : 0.f;
        float rr[3];
        for (int a=0;a<3;a++) { dl[n*3+a] += al*p[n*3+a]; rr[a] = r[n*3+a] - al*Ap[n*3+a]; r[n*3+a] = rr[a]; }
        const float* D = Dinv + n*9;
        for (int a=0;a<3;a++) { float zn = D[a*3]*rr[0] + D[a*3+1]*rr[1] + D[a*3+2]*rr[2]; float zo = z[n*3+a]; z[n*3+a] = zn; s3 += (double)rr[a]*zn; s4 += (double)rr[a]*(zn - zo); }
    }
    blocksum_add(s3, &SC2[3]); blocksum_add(s4, &SC2[4]);
}
// β = SC4/SC0 (폴라크-리비에르); p = z + β p; Tp = T p; 다음을 위해 SC0 ← SC3 (단일 스레드 쪽에서)
__global__ void rt_cg2(const int* IP, const double* SC, const float* sk, const int* pm, float n0, float n1, float n2, const float* z, float* p, float* Tp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float be = (SC[0] > 0.0) ? (float)(SC[4]/SC[0]) : 0.f;
    float pp[3]; for (int a=0;a<3;a++) { pp[a] = z[n*3+a] + be*p[n*3+a]; p[n*3+a] = pp[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, pp, t); for (int a=0;a<3;a++) Tp[n*3+a] = t[a];
}
__global__ void rt_sc_roll(double* SC) { SC[0] = SC[3]; SC[2] = 0.0; SC[3] = 0.0; SC[4] = 0.0; }
__global__ void rt_sc_clear(double* SC, int a, int b) { for (int i=a;i<b;i++) SC[i] = 0.0; }
// 선탐색: 후보 c 의 y_c = ŷ + T (η + c δ) 와 관성 몫 ½ Σ Mi |η + c δ|² (SC[slot]), 그리고 g·δ (SC[9])
__global__ void rt_lsy(const int* IP, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* eta, const float* dl,
                       float c, float* yc, double* SC, int slot, const float* r) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s = 0.0, gd = 0.0;
    if (n < IP[9]) {
        float e[3]; for (int a=0;a<3;a++) e[a] = eta[n*3+a] + c*dl[n*3+a];
        float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
        float Mi = mn[n]*itau2;
        for (int a=0;a<3;a++) { yc[n*3+a] = yh[n*3+a] + t[a]; s += 0.5*(double)Mi*e[a]*e[a]; if (r) gd += -(double)r[n*3+a]*dl[n*3+a]; }
    }
    blocksum_add(s, &SC[slot]);
    if (r) blocksum_add(gd, &SC[9]);
}
// 보폭 고르기: SC[10..13] = E 탄성 (c = 0, ¼, ½, 1), SC[5..8] = 관성 몫; 받아들여지는 가장 큰 c -> SC[14]
__global__ void rt_lspick(double* SC) {
    double E0 = SC[5] + SC[10]; double gd = SC[9];
    const double cs[3] = {0.25, 0.5, 1.0};
    double st = 0.0;
    for (int i=0;i<3;i++) { double E1 = SC[6+i] + SC[11+i]; if (isfinite(E1) && E1 <= E0 + 1e-4*cs[i]*gd) st = cs[i]; }
    SC[14] = st;
}
__global__ void rt_lsup(const int* IP, const double* SC, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, float* eta, const float* dl, float* y, int full) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float st = full ? 1.f : (float)SC[14];
    float e[3]; for (int a=0;a<3;a++) { e[a] = eta[n*3+a] + st*dl[n*3+a]; eta[n*3+a] = e[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y[n*3+a] = yh[n*3+a] + t[a];
}
// 걸음 끝: U, v (중점·후퇴 오일러), 고정 노드 v = vbc, 감쇠
__global__ void rt_stepend(const int* IP, int mid, float tau, float al, const float* y, const float* z0, float* U, float* vk, const unsigned char* fixd, const float* vbc, float damp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) {
        float yy = y[n*3+a], u = U[n*3+a];
        float v1 = mid ? al*(yy - z0[n*3+a]) : (yy - u)/tau;
        if (fixd && fixd[n]) v1 = vbc[n*3+a];
        U[n*3+a] = mid ? 2.f*yy - u : yy;
        vk[n*3+a] = v1*damp;
    }
}
// G2P (제자리): x += Σ w U, v = Σ w v, F ← (I + Σ U ⊗ ∇w) F
__global__ void rt_g2p(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* U, const float* vk, float* x, float* v, float* F) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float u[3] = {0,0,0}, vv[3] = {0,0,0}, A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k];
        float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
        for (int a=0;a<3;a++) { float uu = U[nid*3+a]; u[a] += ww*uu; vv[a] += ww*vk[nid*3+a]; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }
    }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*Fp[k*3+b]; F[(size_t)p*9+a*3+b] = t; }
}
// 고정 노드 (ficus 등 cuboid 구속): 상자 안 노드 -> fixd, vbc (프레임 시각 기준, 이동 상자)
__global__ void rt_bc_box(const int* IP, const float* Am, const float* BP, int ib, unsigned char* fixd, float* vbc) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    const float* b = BP + ib*10;                                              // 중심 3, 반 크기 3, 속도 3, 켜짐 1 (호스트가 프레임마다 채움)
    if (b[9] == 0.f) return;
    float c0 = b[0], c1 = b[1], c2 = b[2], s0 = b[3], s1 = b[4], s2 = b[5], v0 = b[6], v1 = b[7], v2 = b[8];
    int nb1 = IP[7], nb2 = IP[8];
    int i = n / (nb1*nb2), j = (n / nb2) % nb1, k = n % nb2;
    float yq[3] = {(float)(IP[0] + i), (float)(IP[1] + j), (float)(IP[2] + k)};
    float xp[3]; for (int a=0;a<3;a++) xp[a] = Am[a*3]*yq[0] + Am[a*3+1]*yq[1] + Am[a*3+2]*yq[2];
    bool in = fabsf(xp[0]-c0) < s0 && fabsf(xp[1]-c1) < s1 && fabsf(xp[2]-c2) < s2;
    if (in) { fixd[n] = 1; vbc[n*3] = v0; vbc[n*3+1] = v1; vbc[n*3+2] = v2; }
}
// 선탐색 (융합): 후보 c = ¼, ½, 1 의 탄성 에너지를 한 순회로 (y_c = ŷ + T (η + c δ) 를 모으며 계산) -> SC[11..13]
__global__ void __launch_bounds__(128) rt_elem_ls(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* yh, const float* eta, const float* dl,
                        const float* sk, const int* pm, float n0, float n1, float n2, float mu, float lam, double* SC) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    const float cs[3] = {0.25f, 0.5f, 1.f};
    float X[3][9]; for (int c=0;c<3;c++) for (int k=0;k<9;k++) X[c][k] = F[k];
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int c=0;c<3;c++) {
            float e[3]; for (int a=0;a<3;a++) e[a] = eta[id*3+a] + cs[c]*dl[id*3+a];
            float t[3]; Tapply(sk[id], pm[id], n0, n1, n2, e, t);
            for (int a=0;a<3;a++) { float ya = yh[id*3+a] + t[a]; X[c][a*3] += ya*g[0]; X[c][a*3+1] += ya*g[1]; X[c][a*3+2] += ya*g[2]; } } }
    float vq = on ? Vq[qq] : 0.f;
    for (int c=0;c<3;c++) {
        float U[9], S[3], V[9]; ssvdf(X[c], U, S, V);
        double J = (double)S[0]*S[1]*S[2];
        double e = on ? (double)vq*((double)mu*(((double)S[0]-1.0)*((double)S[0]-1.0) + ((double)S[1]-1.0)*((double)S[1]-1.0) + ((double)S[2]-1.0)*((double)S[2]-1.0)) + 0.5*(double)lam*(J-1.0)*(J-1.0)) : 0.0;
        blocksum_add(e, &SC[11+c]);
    }
}
__global__ void rt_lsy4(const int* IP, const float* mn, float itau2, const float* eta, const float* dl, const float* r0, double* SC) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s[4] = {0,0,0,0}, gd = 0.0;
    const float cs[4] = {0.f, 0.25f, 0.5f, 1.f};
    if (n < IP[9]) { float Mi = mn[n]*itau2;
        for (int a=0;a<3;a++) { float e0 = eta[n*3+a], d = dl[n*3+a]; for (int c=0;c<4;c++) { float e = e0 + cs[c]*d; s[c] += 0.5*(double)Mi*e*e; } gd += -(double)r0[n*3+a]*d; } }
    for (int c=0;c<4;c++) blocksum_add(s[c], &SC[5+c]);
    blocksum_add(gd, &SC[9]);
}
// G2P + 마지막 걸음 노드 갱신 (융합): 노드의 U_1 = 2y − U (중점) 또는 y, v_1 = α (y − z0) 또는 (y − U)/τ, 고정 노드 v = vbc, 감쇠
__global__ void rt_g2p_end(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* y, const float* z0, const float* U, int mid, float tau, float al,
                           const unsigned char* fixd, const float* vbc, float damp, float* x, float* v, float* F) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float u[3] = {0,0,0}, vv[3] = {0,0,0}, A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k];
        float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
        bool fx_ = fixd && fixd[nid];
        for (int a=0;a<3;a++) {
            float yy = y[nid*3+a], uk = U[nid*3+a];
            float uu = mid ? 2.f*yy - uk : yy;
            float v1 = fx_ ? vbc[nid*3+a] : (mid ? al*(yy - z0[nid*3+a]) : (yy - uk)/tau);
            u[a] += ww*uu; vv[a] += ww*v1*damp; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }
    }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*Fp[k*3+b]; F[(size_t)p*9+a*3+b] = t; }
}
// ================= 멀티그리드 (거친 층 = 조밀 상자, 프레임당 한 번 구성) =================
// LM[l*8 + ..]: 0..2 노드 상자 차원, 3 노드 수, 4..6 칸 상자 차원, 7 가우스 점 수 (= 칸 x 8). 층 0 = 고운 층 (IP).
__global__ void rt_mg_dims(int* IP, int* LM, int nl, const int* capn, const int* capq) {
    int n0 = IP[6], n1 = IP[7], n2 = IP[8];
    LM[0] = n0; LM[1] = n1; LM[2] = n2; LM[3] = IP[9];
    for (int l=1; l<nl; l++) {
        n0 = n0/2 + 3; n1 = n1/2 + 3; n2 = n2/2 + 3;
        int* o = LM + l*8; o[0] = n0; o[1] = n1; o[2] = n2; o[3] = n0*n1*n2; o[4] = n0-2; o[5] = n1-2; o[6] = n2-2; o[7] = (n0-2)*(n1-2)*(n2-2)*8;
        if (o[3] > capn[l] || o[7] > capq[l]) { o[3] = 0; o[7] = 0; atomicOr(&IP[13], 2); }
    }
}
// 거친 층 l 가우스 점 누적: 출처 = 층 l−1 의 가우스 점 (위치 tq = 칸 좌표 + 팔분 위치, X = F, 가중 V)
__global__ void rt_mg_acc(int l, const int* IP, const int* LM, const int* cbF, const float* Fsrc, const float* Vsrc, float* A, float* C, int* badq, int* badc) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    int nsrc = (l == 1) ? IP[12] : LM[(l-1)*8+7];
    if (q >= nsrc) return;
    float w = Vsrc[q]; if (!(w > 0.f)) return;
    int c = q >> 3, oc = q & 7;
    const float GG = 0.28867513459481287f;
    int cx, cy, cz;
    if (l == 1) { cx = cbF[c*3]; cy = cbF[c*3+1]; cz = cbF[c*3+2]; }
    else { const int* pl = LM + (l-1)*8; cz = c % pl[6]; cy = (c / pl[6]) % pl[5]; cx = c / (pl[5]*pl[6]); }
    float tq[3] = {cx + 1.f + ((oc & 4) ? GG : -GG), cy + 1.f + ((oc & 2) ? GG : -GG), cz + 1.f + ((oc & 1) ? GG : -GG)};
    const int* L = LM + l*8;
    int b[3]; float fx[3];
    for (int a=0;a<3;a++) { float t = 0.5f*(tq[a] + 1.5f); float fb = floorf(t - 0.5f); b[a] = (int)fb; fx[a] = t - fb; }
    int cc = (b[0]*L[5] + b[1])*L[6] + b[2];
    int qc = cc*8 + (fx[0] >= 1.f ? 4 : 0) + (fx[1] >= 1.f ? 2 : 0) + (fx[2] >= 1.f ? 1 : 0);
    float X[9]; for (int k=0;k<9;k++) X[k] = Fsrc[(size_t)q*9+k];
    float U[9], S[3], V[9]; ssvdf(X, U, S, V);
    bool ng = !(S[2] > 0.f);
    float R[9], Lg[9]; float lg[3]; for (int i=0;i<3;i++) lg[i] = ng ? 0.f : logf(S[i]);
    for (int a=0;a<3;a++) for (int bb=0;bb<3;bb++) { R[a*3+bb] = U[a*3]*V[bb*3] + U[a*3+1]*V[bb*3+1] + U[a*3+2]*V[bb*3+2];
        Lg[a*3+bb] = V[a*3]*lg[0]*V[bb*3] + V[a*3+1]*lg[1]*V[bb*3+1] + V[a*3+2]*lg[2]*V[bb*3+2]; }
    float* aq = A + (size_t)qc*28; float* ac = C + (size_t)cc*30;
    for (int k=0;k<9;k++) { atomicAdd(&aq[k], w*R[k]); atomicAdd(&aq[9+k], w*Lg[k]); atomicAdd(&aq[18+k], w*X[k]);
                            atomicAdd(&ac[k], w*R[k]); atomicAdd(&ac[9+k], w*Lg[k]); atomicAdd(&ac[18+k], w*X[k]); }
    atomicAdd(&aq[27], w); atomicAdd(&ac[27], w); atomicAdd(&ac[28], w);
    if (ng) { atomicOr(&badq[qc], 1); atomicOr(&badc[cc], 1); }
}
__global__ void rt_mg_fin(int l, const int* LM, const float* A, const float* C, const int* badq, const int* badc, float* Fq, float* Vq) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= LM[l*8+7]) return;
    int c = q >> 3;
    const float* aq = A + (size_t)q*28; const float* ac = C + (size_t)c*30;
    float out[9]; bool bad = false;
    if (!(ac[27] > 0.f)) { for (int k=0;k<9;k++) out[k] = (k % 4 == 0) ? 1.f : 0.f; Vq[q] = 0.f; }
    else {
        if (aq[27] > 0.f) { rs_fin(aq, aq + 9, aq[27], out, &bad); if (bad || badq[q]) for (int k=0;k<9;k++) out[k] = aq[18+k]/aq[27]; }
        else { rs_fin(ac, ac + 9, ac[27], out, &bad); if (bad || badc[c]) for (int k=0;k<9;k++) out[k] = ac[18+k]/ac[27]; }
        Vq[q] = ac[28]*0.125f;
    }
    for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = out[k];
}
__device__ __forceinline__ void qnodes_d(int q, const int* L, int* ids) {
    int c = q >> 3; int cz = c % L[6], cy = (c / L[6]) % L[5], cx = c / (L[5]*L[6]);
    for (int o=0;o<27;o++) { int i = o/9, j = (o/3)%3, k = o%3; ids[o] = ((cx+i)*L[1] + (cy+j))*L[2] + (cz+k); }
}
// 거친 층 헤시안 (y = 0 -> X = F_q) 성분 우선 저장 + 노드 대각 융합. BG 기울기에 sc = 2^-l
__global__ void __launch_bounds__(128) rt_mg_elem(int l, const int* LM, const float* BG, float sc, const float* Fq, const float* Vq, float mu, float lam, float* H, int capQ, float* B) {
    const int* L = LM + l*8;
    if (blockIdx.x*blockDim.x >= L[7]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < L[7]; int qq = on ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes_d(qq, L, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    if (__all_sync(0xffffffffu, !(vq > 0.f))) return;
    float h[81]; for (int k=0;k<81;k++) h[k] = 0.f;
    if (on && vq > 0.f) {
        float U[9], S[3], V[9]; ssvdf(F, U, S, V);
        float J = S[0]*S[1]*S[2]; float gJ[3] = {S[1]*S[2], S[0]*S[2], S[0]*S[1]};
        float D[3][3], Ev[3][3];
        float Sg[3][3] = {{0.f, S[2], S[1]}, {S[2], 0.f, S[0]}, {S[1], S[0], 0.f}};
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = ((i==j) ? 2.f*mu : 0.f) + lam*gJ[i]*gJ[j] + lam*(J-1.f)*Sg[i][j];
        jac3f(D, Ev);
        for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (!(dm > 0.f)) continue;
            float e[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) e[a*3+b] = Ev[0][m]*U[a*3]*V[b*3] + Ev[1][m]*U[a*3+1]*V[b*3+1] + Ev[2][m]*U[a*3+2]*V[b*3+2];
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += dm*e[r]*e[c]; }
        const int PI[3] = {0,0,1}, PJ[3] = {1,2,2}, PK[3] = {2,1,0};
        const float r2 = 0.70710678f;
        for (int pp=0; pp<3; pp++) { int i = PI[pp], j = PJ[pp], k = PK[pp];
            float lt = 2.f*mu*(1.f - 2.f/(S[i]+S[j])) + lam*(J-1.f)*S[k]; float lf = 2.f*mu - lam*(J-1.f)*S[k];
            lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = fmaxf(lf, 0.f);
            float et[9], ef[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float x1 = U[a*3+i]*V[b*3+j], x2 = U[a*3+j]*V[b*3+i]; et[a*3+b] = r2*(x1-x2); ef[a*3+b] = r2*(x1+x2); }
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += lt*et[r]*et[c] + lf*ef[r]*ef[c]; }
    }
    if (on) for (int k=0;k<81;k++) H[(size_t)k*capQ + qq] = h[k];
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; float bs[3] = {sc*bg[0], sc*bg[1], sc*bg[2]}; float g[3]; gqn(F, bs, g);
        float m[9];
        for (int a=0;a<3;a++) for (int c=0;c<3;c++) { float t = 0.f;
            for (int b=0;b<3;b++) { float hg = h[(a*3+b)*9 + c*3]*g[0] + h[(a*3+b)*9 + c*3+1]*g[1] + h[(a*3+b)*9 + c*3+2]*g[2]; t += g[b]*hg; }
            m[a*3+c] = vq*t; }
        for (int k=0;k<9;k++) m[k] = seg8f(m[k]);
        if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], m[k]); }
}
// 거친 층 헤시안 곱 (조밀)
__global__ void __launch_bounds__(128) rt_mg_mv(int l, const int* LM, const float* BG, float sc, const float* Fq, const float* Vq, const float* H, int capQ, const float* dy, float* out) {
    const int* L = LM + l*8;
    if (blockIdx.x*blockDim.x >= L[7]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < L[7]; int qq = on ? q : 0; bool head = (threadIdx.x & 7) == 0; int oc = qq & 7;
    int ids[27]; qnodes_d(qq, L, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    if (__all_sync(0xffffffffu, !(vq > 0.f))) return;                           // 빈 칸 (부피 0) 만 든 워프는 기여 0
    float dX[9] = {0,0,0,0,0,0,0,0,0};
    float gs[27][3];
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; float bs[3] = {sc*bg[0], sc*bg[1], sc*bg[2]}; gqn(F, bs, gs[o]); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = dy[id*3+a]; dX[a*3] += ya*gs[o][0]; dX[a*3+1] += ya*gs[o][1]; dX[a*3+2] += ya*gs[o][2]; } }
    float dP[9]; for (int r=0;r<9;r++) { float t = 0.f; if (vq > 0.f) for (int c=0;c<9;c++) t += H[(size_t)(r*9+c)*capQ + qq]*dX[c]; dP[r] = vq*t; }
    for (int o=0;o<27;o++) for (int a=0;a<3;a++) { float s = seg8f(dP[a*3]*gs[o][0] + dP[a*3+1]*gs[o][1] + dP[a*3+2]*gs[o][2]); if (on && head) atomicAdd(&out[ids[o]*3+a], s); }
}
// 제한 out[J] = Σ c_k c_l c_m in[2J − 3 + (k,l,m)] (c = 1,3,3,1 / 4), 고운 층 차원 nf (LM 층 lf), 거친 층 차원 (LM 층 lf+1). ch = 1 또는 3
__global__ void rt_mg_restrict(int lf, const int* LM, const float* in, float* out, int ch) {
    const int* A = LM + lf*8; const int* B = LM + (lf+1)*8;
    int J = blockIdx.x*blockDim.x + threadIdx.x;
    if (J >= B[3]) return;
    int jz = J % B[2], jy = (J / B[2]) % B[1], jx = J / (B[1]*B[2]);
    const float cw[4] = {0.25f, 0.75f, 0.75f, 0.25f};
    float acc[3] = {0.f, 0.f, 0.f};
    for (int k=0;k<4;k++) { int ix = 2*jx - 3 + k; if (ix < 0 || ix >= A[0]) continue;
      for (int m=0;m<4;m++) { int iy = 2*jy - 3 + m; if (iy < 0 || iy >= A[1]) continue;
        for (int n=0;n<4;n++) { int iz = 2*jz - 3 + n; if (iz < 0 || iz >= A[2]) continue;
          float w = cw[k]*cw[m]*cw[n]; int i = (ix*A[1] + iy)*A[2] + iz;
          for (int c=0;c<ch;c++) acc[c] += w*in[i*ch+c]; } } }
    for (int c=0;c<ch;c++) out[J*ch+c] = acc[c];
}
// 연장 out[i] (+)= Σ_J c_{i−2J+3} in[J]
__global__ void rt_mg_prolong(int lf, const int* LM, const float* in, float* out, int add) {
    const int* A = LM + lf*8; const int* B = LM + (lf+1)*8;
    int I = blockIdx.x*blockDim.x + threadIdx.x;
    if (I >= A[3]) return;
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
    for (int c=0;c<3;c++) out[I*3+c] = (add ? out[I*3+c] : 0.f) + acc[c];
}
// 층 l 배열 0 채우기 (실제 크기만): which 0 = 노드 수, 1 = 가우스 점 수, 2 = 칸 수
__global__ void rt_zero_lm(int l, const int* LM, int which, int mul, float* a, int* ai) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int n = (which == 0) ? LM[l*8+3] : ((which == 1) ? LM[l*8+7] : LM[l*8+7]/8);
    if (i >= n*mul) return;
    if (a) a[i] = 0.f; else ai[i] = 0;
}
// 거친 층 노드: D = (Mi I + B)⁻¹ (질량·대각 0 이면 0)
__global__ void rt_mg_D(int l, const int* LM, const float* mc, float itau2, const float* B, float* D) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= LM[l*8+3]) return;
    float M[9]; float Mi = mc[n]*itau2;
    for (int k=0;k<9;k++) M[k] = B[n*9+k] + ((k % 4 == 0) ? Mi : 0.f);
    float Di[9]; inv3(M, Di); for (int k=0;k<9;k++) D[n*9+k] = Di[k];
}
// 층 노드 연산: (0) z = w D r; (1) z += w D (r − Az); (2) res = r − Az;  Az = Mi z + KTz 마무리 (층 0 은 T 끼움)
// 층 0: Az = Mi z + T (K (T z)); 여기서 KTz 는 이미 계산된 K(Tz) 또는 K z. 모드 3: Tz = T z (층 0), 모드 4: out = T v
__global__ void rt_mg_node(int l, const int* LM, int mode, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2,
                           const float* D, const double* W, const float* r, float* KTz, float* z, float* res) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= LM[l*8+3]) return;
    float w = (float)W[l];
    const float* d = D + n*9;
    float rr[3], az[3];
    for (int a=0;a<3;a++) rr[a] = r ? r[n*3+a] : 0.f;
    if (mode == 0 || mode == 6) { float zz[3]; for (int a=0;a<3;a++) { zz[a] = w*(d[a*3]*rr[0] + d[a*3+1]*rr[1] + d[a*3+2]*rr[2]); z[n*3+a] = zz[a]; }
        if (mode == 6) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, zz, t); for (int a=0;a<3;a++) res[n*3+a] = t[a]; }       // 층 0: 다음 Az 의 T z 를 함께 (융합)
        return; }
    if (mode == 3 || mode == 4) { const float* v = (mode == 3) ? z : r; float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, v + n*3, t); for (int a=0;a<3;a++) res[n*3+a] = t[a]; return; }
    if (mode == 5) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, r + n*3, t); for (int a=0;a<3;a++) z[n*3+a] += t[a]; return; }
    float Mi = mn[n]*itau2;
    float kt[3] = {KTz[n*3], KTz[n*3+1], KTz[n*3+2]};
    KTz[n*3] = 0.f; KTz[n*3+1] = 0.f; KTz[n*3+2] = 0.f;                         // 읽은 뒤 비움 (다음 곱의 원자 합 바탕; 0 채우기 커널 없앰)
    if (l == 0) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, kt, t); for (int a=0;a<3;a++) kt[a] = t[a]; }
    for (int a=0;a<3;a++) az[a] = Mi*z[n*3+a] + kt[a];
    float e[3]; for (int a=0;a<3;a++) e[a] = rr[a] - az[a];
    if (mode == 2) { for (int a=0;a<3;a++) res[n*3+a] = e[a]; return; }
    if (mode == 7) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t); for (int a=0;a<3;a++) res[n*3+a] = t[a]; return; }   // 층 0: T (r − Az) (제한 입력, 융합)
    for (int a=0;a<3;a++) z[n*3+a] += w*(d[a*3]*e[0] + d[a*3+1]*e[1] + d[a*3+2]*e[2]);
}
// 거듭제곱법 한 걸음: z ← D A z / |D A z|, λ = (z·DAz)/(z·z) -> SCM[l]; 마지막에 W[l] = ω/λ
__global__ void rt_mg_pw1(int l, const int* LM, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* D, const float* z, float* KTz,
                          float* y, double* SCM) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int nn = LM[l*8+3];
    if (blockIdx.x*blockDim.x >= nn) return;
    if (n < nn) {
        float Mi = mn[n]*itau2;
        float kt[3] = {KTz[n*3], KTz[n*3+1], KTz[n*3+2]};
        KTz[n*3] = 0.f; KTz[n*3+1] = 0.f; KTz[n*3+2] = 0.f;
        if (l == 0) { float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, kt, t); for (int a=0;a<3;a++) kt[a] = t[a]; }
        float az[3]; for (int a=0;a<3;a++) az[a] = Mi*z[n*3+a] + kt[a];
        const float* d = D + n*9;
        for (int a=0;a<3;a++) { float v = d[a*3]*az[0] + d[a*3+1]*az[1] + d[a*3+2]*az[2]; y[n*3+a] = v; s1 += (double)z[n*3+a]*v; s2 += (double)z[n*3+a]*z[n*3+a]; s3 += (double)v*v; }
    }
    blocksum_add(s1, &SCM[l*4]); blocksum_add(s2, &SCM[l*4+1]); blocksum_add(s3, &SCM[l*4+2]);
}
__global__ void rt_mg_pw2(int l, const int* LM, const float* y, float* z, double* SCM, double* W, float omega, int last) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0 && last) { double lm = SCM[l*4] / fmax(SCM[l*4+1], 1e-300); W[l] = omega / fmax(lm, 1e-30); }
    if (n >= LM[l*8+3]) return;
    double nr = sqrt(fmax(SCM[l*4+2], 1e-300));
    for (int a=0;a<3;a++) z[n*3+a] = (float)(y[n*3+a] / nr);
}
__global__ void rt_mg_pw0(int l, double* SCM) { SCM[l*4] = 0.0; SCM[l*4+1] = 0.0; SCM[l*4+2] = 0.0; }
__global__ void rt_fill_rand(int l, const int* LM, float* z) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= LM[l*8+3]) return;
    unsigned int h = (unsigned int)n*2654435761u + 12345u*(l+1);
    for (int a=0;a<3;a++) { h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15; z[n*3+a] = ((float)(h & 0xffffff) / 16777216.f) - 0.5f; }
}
// PCG + MG: α 갱신만 (z 는 V 순환이 따로); 그다음 내적 r·z, r·(z − zo) 와 zo ← z
__global__ void rt_cg1m(const int* IP, const double* SC, const float* p, const float* Ap, float* dl, float* r) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    double pAp = SC[2]; float al = (pAp > 0.0) ? (float)(SC[0]/pAp) : 0.f;
    for (int a=0;a<3;a++) { dl[n*3+a] += al*p[n*3+a]; r[n*3+a] -= al*Ap[n*3+a]; }
}
__global__ void rt_dotz(const int* IP, const float* r, const float* z, float* zo, double* SC, int first) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s3 = 0.0, s4 = 0.0;
    if (n < IP[9]) for (int a=0;a<3;a++) { float zn = z[n*3+a]; s3 += (double)r[n*3+a]*zn; s4 += (double)r[n*3+a]*(zn - zo[n*3+a]); zo[n*3+a] = zn; }
    blocksum_add(s3, first ? &SC[0] : &SC[3]); if (!first) blocksum_add(s4, &SC[4]);
}
__global__ void rt_pinit(const int* IP, const float* z, const float* sk, const int* pm, float n0, float n1, float n2, float* p, float* Tp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, z + n*3, t);
    for (int a=0;a<3;a++) { p[n*3+a] = z[n*3+a]; Tp[n*3+a] = t[a]; }
}
// 하강 방향 보장 (--descent): PCG 결과 δ 가 g·δ ≥ 0 이면 δ ← α0 p0 (p0 = 첫 방향 = M r0, α0 = r0·M r0 / p0·A p0). SC[16] = r0·Mr0, SC[17] = p0·Ap0, SC[18] 바꾼 횟수
__global__ void rt_fixdir(const int* IP, double* SC, const float* p0, float* dl) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    bool bad = !(SC[9] < 0.0);
    if (n == 0 && bad) SC[62] += 1.0;
    if (n >= IP[9] || !bad) return;
    float a0 = (SC[61] > 0.0) ? (float)(SC[60]/SC[61]) : 0.f;
    for (int a=0;a<3;a++) dl[n*3+a] = a0*p0[n*3+a];
}
__global__ void rt_sc_copy(double* SC, int dst, int src) { SC[dst] = SC[src]; }
// 선탐색 넓힘 (--ls_n NC): 후보 c_i = 2^-i (i = 0..NC−1). 관성 몫 SC[20] (c = 0), SC[21+i]; 탄성 SC[40+i]; g·δ SC[9]; 보폭 SC[14]
__global__ void rt_lsyN(const int* IP, const float* mn, float itau2, const float* eta, const float* dl, const float* r0, double* SC, int NC) {
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s0 = 0.0, gd = 0.0, sc[12];
    for (int i=0;i<12;i++) sc[i] = 0.0;
    if (n < IP[9]) { float Mi = mn[n]*itau2;
        for (int a=0;a<3;a++) { float e0 = eta[n*3+a], d = dl[n*3+a]; s0 += 0.5*(double)Mi*e0*e0; gd += -(double)r0[n*3+a]*d;
            float c = 1.f; for (int i=0;i<NC;i++) { float e = e0 + c*d; sc[i] += 0.5*(double)Mi*e*e; c *= 0.5f; } } }
    blocksum_add(s0, &SC[20]); blocksum_add(gd, &SC[9]);
    for (int i=0;i<NC;i++) blocksum_add(sc[i], &SC[21+i]);
}
__global__ void __launch_bounds__(128) rt_elem_lsN(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* yh, const float* eta, const float* dl,
                        const float* sk, const int* pm, float n0, float n1, float n2, float mu, float lam, double* SC, int NC) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float vq = on ? Vq[qq] : 0.f;
    float c = 1.f;
    for (int i=0;i<NC;i++) {
        float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
        for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
            float e[3]; for (int a=0;a<3;a++) e[a] = eta[id*3+a] + c*dl[id*3+a];
            float t[3]; Tapply(sk[id], pm[id], n0, n1, n2, e, t);
            for (int a=0;a<3;a++) { float ya = yh[id*3+a] + t[a]; X[a*3] += ya*g[0]; X[a*3+1] += ya*g[1]; X[a*3+2] += ya*g[2]; } }
        float U[9], S[3], V[9]; ssvdf(X, U, S, V);
        double J = (double)S[0]*S[1]*S[2];
        double en = on ? (double)vq*((double)mu*(((double)S[0]-1.0)*((double)S[0]-1.0) + ((double)S[1]-1.0)*((double)S[1]-1.0) + ((double)S[2]-1.0)*((double)S[2]-1.0)) + 0.5*(double)lam*(J-1.0)*(J-1.0)) : 0.0;
        blocksum_add(en, &SC[40+i]);
        c *= 0.5f;
    }
}
__global__ void rt_lspickN(double* SC, int NC) {
    double E0 = SC[20] + SC[10]; double gd = SC[9];
    double st = 0.0, c = 1.0;
    for (int i=0;i<NC;i++) { double E1 = SC[21+i] + SC[40+i]; if (st == 0.0 && isfinite(E1) && E1 <= E0 + 1e-4*c*gd) st = c; c *= 0.5; }
    SC[14] = st;
}
// ---- 단계 일반화 (θ 법·TR-BDF2) ----
// 단계 시작: Ut = Ub + τ vb + τ² g, z0 = Ub + β τ vb, y = Ut (고정 노드 y = Uf + dtf vbc). θ 단계: Ub = U, vb = v, β = 1 − θ; BDF2 단계: Ub = w, vb = ŵ, β = 0.
__global__ void rt_step0g(const int* IP, const float* Ub, const float* vb, float tau, float beta, float g0, float g1, float g2, const unsigned char* fixd, const float* vbc,
                          const float* Uf, float dtf, float* Ut, float* z0, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float gg[3] = {g0, g1, g2};
    for (int a=0;a<3;a++) {
        float u = Ub[n*3+a], vv = vb[n*3+a];
        float ut = u + tau*vv + tau*tau*gg[a];
        Ut[n*3+a] = ut; z0[n*3+a] = u + beta*tau*vv;
        y[n*3+a] = (fixd && fixd[n]) ? Uf[n*3+a] + dtf*vbc[n*3+a] : ut;
    }
}
// 단계 끝: U ← U + thinv (y − U) (θ 단계 thinv = 1/θ, BDF2 1), v ← α (y − z0) (고정 노드 vbc), 감쇠
__global__ void rt_stepend_g(const int* IP, float thinv, float al, const float* y, const float* z0, float* U, float* vk, const unsigned char* fixd, const float* vbc, float damp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) {
        float yy = y[n*3+a], u = U[n*3+a];
        float v1 = al*(yy - z0[n*3+a]);
        if (fixd && fixd[n]) v1 = vbc[n*3+a];
        U[n*3+a] = u + thinv*(yy - u);
        vk[n*3+a] = v1*damp;
    }
}
// BDF2 기저: w = c1 Uγ − c0 U0, ŵ = c1 vγ − c0 v0
__global__ void rt_bdf_base(const int* IP, float c1, float c0, const float* Ug, const float* vg, const float* U0, const float* v0, float* w, float* wv) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) { w[n*3+a] = c1*Ug[n*3+a] - c0*U0[n*3+a]; wv[n*3+a] = c1*vg[n*3+a] - c0*v0[n*3+a]; }
}
// 관성 질량 하한: m' = max(m, cs tr(B)/3) (cs = c τ², B = 걸음 첫 뉴턴의 노드 탄성 대각 블록), cs = 0 이면 m 그대로
__global__ void rt_meff(const int* IP, const float* mn, const float* B, float cs, float* meff) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n];
    meff[n] = (cs > 0.f) ? fmaxf(m, cs*(B[n*9] + B[n*9+4] + B[n*9+8])*(1.f/3.f)) : m;
}
// G2P + 마지막 단계 끝 (일반): uu = U + thinv (y − U), v1 = α (y − z0) 또는 vbc
__global__ void rt_g2p_endg(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* y, const float* z0, const float* U, float thinv, float al,
                           const unsigned char* fixd, const float* vbc, float damp, float* x, float* v, float* F) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float u[3] = {0,0,0}, vv[3] = {0,0,0}, A[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k];
        float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
        float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
        bool fx_ = fixd && fixd[nid];
        for (int a=0;a<3;a++) {
            float yy = y[nid*3+a], uk = U[nid*3+a];
            float uu = uk + thinv*(yy - uk);
            float v1 = fx_ ? vbc[nid*3+a] : al*(yy - z0[nid*3+a]);
            u[a] += ww*uu; vv[a] += ww*v1*damp; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }
    }
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = 0.f; for (int k=0;k<3;k++) t += A[a*3+k]*Fp[k*3+b]; F[(size_t)p*9+a*3+b] = t; }
}
// 접촉 적응: 고착 노드 수 -> IP[14] (rt_reset 이 비움)
__global__ void rt_count_stuck(const int* IP, const float* an, int* IPw) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (an[n] > 0.f) atomicAdd(&IPw[14], 1);
}
}

//@@ det_rt_DET_DECL 864

__device__ int AO_DET;
__device__ double* AO_SCB;
__device__ double* AO_PART;
__device__ int* AO_BMAX;
__device__ double* AO_CF;                                                         // 칸 x 27 x 3 (f64 힘)
__device__ float* AO_CB;                                                          // 칸 x 27 x 9 (노드 대각)
__device__ float* AO_CV;                                                          // 칸 x 27 x 3 (헤시안 곱)
__device__ float* AO_MW;                                                          // 점별 고착 질량
__device__ int* AO_NCUT;                                                          // 입자별 구속 끊는 표본 수
__device__ __forceinline__ void ao_bs_out(double* out, double v) {
    long long sl = out - AO_SCB;
    if (AO_DET && sl >= 0 && sl < 64) { AO_PART[sl*16384 + blockIdx.x] = v; atomicMax(&AO_BMAX[sl], (int)blockIdx.x + 1); }
    else atomicAdd(out, v);
}

//@@ det_rt_DET_SRC 7973

__global__ void ao_bsfin() {
    int sl = blockIdx.x;
    int nb = AO_BMAX[sl];
    if (nb == 0) return;
    __shared__ double sh[256];
    double s = 0.0;
    for (int b = threadIdx.x; b < nb; b += 256) s += AO_PART[(size_t)sl*16384 + b];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) { AO_SCB[sl] += sh[0]; AO_BMAX[sl] = 0; }
}
// 노드 -> (이웃 칸 c, 그 칸에서 이 노드의 자리 o) 를 고정 순서로: o = (i,j,k), 칸 좌표 = 노드 좌표 − (i,j,k)
__device__ __forceinline__ int ao_ncell(const int* IP, const int* cidx, int n, int o) {
    int nb1 = IP[7], nb2 = IP[8];
    int ni = n / (nb1*nb2), nj = (n / nb2) % nb1, nk = n % nb2;
    int ci = ni - o/9, cj = nj - (o/3)%3, ck = nk - o%3;
    int c1 = nb1 - 2, c2 = nb2 - 2;
    if (ci < 0 || cj < 0 || ck < 0 || ci >= IP[6]-2 || cj >= c1 || ck >= c2) return -1;
    return cidx[(ci*c1 + cj)*c2 + ck];
}
__device__ int* AO_ACTL;                                                          // 압축 판이면 질량 있는 노드 목록 (IP[14] 개), 아니면 null
__device__ __forceinline__ int ao_node(const int* IP, int i) {
    if (AO_ACTL) return (i < IP[14]) ? AO_ACTL[i] : -1;
    return (i < IP[9]) ? i : -1;
}
__global__ void ao_gath_d3(const int* IP, const int* cidx, double* out) {
    int n = ao_node(IP, blockIdx.x*blockDim.x + threadIdx.x);
    if (n < 0) return;
    double s[3] = {0.0, 0.0, 0.0};
    for (int o=0;o<27;o++) { int c = ao_ncell(IP, cidx, n, o); if (c < 0) continue; for (int a=0;a<3;a++) s[a] += AO_CF[((size_t)c*27+o)*3+a]; }
    for (int a=0;a<3;a++) out[n*3+a] += s[a];
}
__global__ void ao_gath_f(const int* IP, const int* cidx, int D, const float* buf, float* out) {
    int n = ao_node(IP, blockIdx.x*blockDim.x + threadIdx.x);
    if (n < 0) return;
    float s[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { int c = ao_ncell(IP, cidx, n, o); if (c < 0) continue; for (int a=0;a<D;a++) s[a] += buf[((size_t)c*27+o)*D+a]; }
    for (int a=0;a<D;a++) out[n*D+a] += s[a];
}
// 점 값 x 기저 (BW) 를 노드가 모음: mode 0 = mq (rt_gauss 의 mqn), 1 = AO_MW (rt_floor_q 의 an)
__global__ void ao_gath_q(const int* IP, const int* cidx, const float* BW, const float* qv, float* out) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float s = 0.f;
    for (int o=0;o<27;o++) { int c = ao_ncell(IP, cidx, n, o); if (c < 0) continue; for (int oc=0;oc<8;oc++) s += BW[oc*27+o]*qv[(size_t)c*8+oc]; }
    out[n] += s;
}
// 노드 목록 (번호 순서, 한 블록 스캔)
__global__ void ao_act_scan(int* IP, const float* mn, int* act) {
    __shared__ int sh[1024];
    int n = IP[9], carry = 0;
    for (int base = 0; base < n; base += 1024) {
        int i = base + threadIdx.x;
        int v = (i < n && mn[i] > 0.f) ? 1 : 0;
        sh[threadIdx.x] = v;
        __syncthreads();
        for (int o=1; o<1024; o<<=1) { int t = (threadIdx.x >= o) ? sh[threadIdx.x - o] : 0; __syncthreads(); sh[threadIdx.x] += t; __syncthreads(); }
        if (v) act[carry + sh[threadIdx.x] - 1] = i;
        int tot = sh[1023];
        __syncthreads();
        carry += tot;
    }
    if (threadIdx.x == 0) IP[14] = carry;
}
// 정렬된 입자의 칸 구간 시작 st[c] (st[칸 수] = N)
__global__ void ao_seg(int N, const int* IP, const int* pcd, const int* cidx, int* st) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p > N || IP[13]) return;
    if (p == N) { st[IP[11]] = N; return; }
    if (p == 0 || pcd[p] != pcd[p-1]) st[cidx[pcd[p]]] = p;
}
// P2G 칸 몫: 칸 하나가 자기 입자를 차례로 (rt_p2g 와 같은 식)
__global__ void ao_p2g_q(int N, const float* F, const float* m, const float* vol, const float* fx, const int* st, const int* IP, float* A, float* C, int* badq, int* badc) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    for (int p = st[c]; p < st[c+1]; p++) {
        int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
        int q = c*8 + oc;
        float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
        float U[9], S[3], V[9];
        ssvdf(Fp, U, S, V);
        float w = vol[p];
        bool ng = !(S[2] > 0.f);
        float R[9], L[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) R[a*3+b] = U[a*3]*V[b*3] + U[a*3+1]*V[b*3+1] + U[a*3+2]*V[b*3+2];
        float l[3]; for (int i=0;i<3;i++) l[i] = ng ? 0.f : logf(S[i]);
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) L[a*3+b] = V[a*3]*l[0]*V[b*3] + V[a*3+1]*l[1]*V[b*3+1] + V[a*3+2]*l[2]*V[b*3+2];
        float* aq = A + (size_t)q*28; float* ac = C + (size_t)c*30;
        for (int k=0;k<9;k++) { aq[k] += w*R[k]; aq[9+k] += w*L[k]; aq[18+k] += w*Fp[k]; ac[k] += w*R[k]; ac[9+k] += w*L[k]; ac[18+k] += w*Fp[k]; }
        aq[27] += w; ac[27] += w; ac[28] += w; ac[29] += m[p];
        if (ng) { badq[q] = 1; badc[c] = 1; }
    }
}
// P2G 노드 몫 (빠른 판): 칸 하나가 자기 입자의 27 노드 몫 (질량, 운동량 3) 을 고정 순서로 더해 칸 버퍼 [칸 x 27 x 4] 에 -> ao_gath_mv 가 노드마다 모음
__global__ void ao_p2g_c(const int* IP, const int* st, const float* fx, const float* m, const float* v, float* cbuf) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    float acc[27][4];
    for (int o=0;o<27;o++) for (int a=0;a<4;a++) acc[o][a] = 0.f;
    for (int p = st[c]; p < st[c+1]; p++) {
        float wx[3][3], dw[3][3];
        bwf(fx[p*3], wx[0], dw[0]); bwf(fx[p*3+1], wx[1], dw[1]); bwf(fx[p*3+2], wx[2], dw[2]);
        float mp = m[p], v0 = v[p*3], v1 = v[p*3+1], v2 = v[p*3+2];
        for (int o=0;o<27;o++) { float ww = wx[0][o/9]*wx[1][(o/3)%3]*wx[2][o%3]*mp; acc[o][0] += ww; acc[o][1] += ww*v0; acc[o][2] += ww*v1; acc[o][3] += ww*v2; }
    }
    for (int o=0;o<27;o++) for (int a=0;a<4;a++) cbuf[((size_t)c*27+o)*4+a] = acc[o][a];
}
__global__ void ao_gath_mv(const int* IP, const int* cidx, const float* cbuf, float* mn, float* mv) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || n >= IP[9]) return;
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    for (int o=0;o<27;o++) { int c = ao_ncell(IP, cidx, n, o); if (c < 0) continue; for (int a=0;a<4;a++) s[a] += cbuf[((size_t)c*27+o)*4+a]; }
    mn[n] += s[0]; for (int a=0;a<3;a++) mv[n*3+a] += s[1+a];
}
__global__ void ao_ncut(int N, const float* Trel, const float* tbuf, int K, int S, float dt) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    float t0 = tbuf[0], tr = Trel[p]; int KS = K*S, ncut = 0;
    if (tr > t0) for (int ks=0; ks<KS; ks++) { int kk = ks / S, ss = ks % S; float ts = t0 + ((float)kk + ((float)ss + 0.5f)/(float)S)*dt; if (tr > ts) ncut = ks + 1; }
    AO_NCUT[p] = ncut;
}
// P2G 노드 몫: 노드 하나가 이웃 27 칸의 입자를 차례로 (질량·운동량), 구속 질량 (KS > 0 이면, ao_pbc_acc 와 같은 끊는 표본)
__global__ void ao_p2g_n(const int* IP, const int* cidx, const int* st, const float* fx, const float* m, const float* v, float* mn, float* mv,
                         const float* Trel, const float* tbuf, int K, int S, float dt, float* acc) {
    int n = mn ? (blockIdx.x*blockDim.x + threadIdx.x) : ao_node(IP, blockIdx.x*blockDim.x + threadIdx.x);   // 구속 질량만이면 질량 있는 노드 목록 (압축 판)
    if (IP[13] || n < 0 || n >= IP[9]) return;
    float sm = 0.f, sv[3] = {0.f, 0.f, 0.f};
    int KS = K*S;
    float t0 = tbuf ? tbuf[0] : 0.f;
    for (int o=0;o<27;o++) {
        int c = ao_ncell(IP, cidx, n, o); if (c < 0) continue;
        int i = o/9, j = (o/3)%3, k = o%3;
        for (int p = st[c]; p < st[c+1]; p++) {
            int ncut = 0;
            if (!mn) { ncut = AO_NCUT[p]; if (ncut == 0) continue; }            // 구속 질량만: 입자별 끊는 표본 (미리 계산), 풀린 입자 건너뜀
            float wx[3][3], dw[3][3];
            bwf(fx[p*3], wx[0], dw[0]); bwf(fx[p*3+1], wx[1], dw[1]); bwf(fx[p*3+2], wx[2], dw[2]);
            float ww = wx[0][i]*wx[1][j]*wx[2][k]*m[p];
            if (mn) { sm += ww; for (int a=0;a<3;a++) sv[a] += ww*v[p*3+a]; }
            if (acc && ncut > 0) acc[(size_t)n*KS + ncut - 1] += ww;
        }
    }
    if (mn) { mn[n] += sm; for (int a=0;a<3;a++) mv[n*3+a] += sv[a]; }
}

//@@ NSD_SRC 4119

#define NSD_UNSET 0x7fffffff
__global__ void nsd_fill_i(int n, int* a, int v) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = v; }
__global__ void nsd_fill_f(int n, float* a, float v) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = v; }
__global__ void nsd_node(const int* IP, const float* v, int* mk, int code) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || n >= IP[9] || mk[n] != NSD_UNSET) return;
    if (!isfinite(v[n*3]) || !isfinite(v[n*3+1]) || !isfinite(v[n*3+2])) mk[n] = code;
}
__global__ void nsd_node_d(const int* IP, const double* v, int* mk, int code) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || n >= IP[9] || mk[n] != NSD_UNSET) return;
    if (!isfinite(v[n*3]) || !isfinite(v[n*3+1]) || !isfinite(v[n*3+2])) mk[n] = code;
}
__global__ void nsd_jump(const int* IP, const float* U, const float* Up, float cinv, float* jm, int* jk, int k) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || n >= IP[9]) return;
    float d0 = U[n*3] - (Up ? Up[n*3] : 0.f), d1 = U[n*3+1] - (Up ? Up[n*3+1] : 0.f), d2 = U[n*3+2] - (Up ? Up[n*3+2] : 0.f);
    float d = sqrtf(d0*d0 + d1*d1 + d2*d2)*cinv;
    if (isfinite(d) && d > jm[n]) { jm[n] = d; jk[n] = k; }
}
__global__ void nsd_sc(const double* SC, int* g, int code) {
    if (g[0] != NSD_UNSET) return;
    for (int i=0;i<16;i++) if (!isfinite(SC[i])) { g[0] = code; return; }
}
// 가우스 점: X = F_q + A F_q (A = Σ_o y_o ⊗ ∇w_o, aq_elem 과 같은 식), y = 0 이면 X = F_q
__global__ void nsd_q(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, const float* Sq,
                      int mat, const float* prm, float dt, int dostress, int code, int* mJ, int* mS, int* mQ, float* jmin) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || q >= IP[12]) return;
    int oc = q & 7;
    float F[9]; bool qbad = false;
    for (int k=0;k<9;k++) { F[k] = Fq[(size_t)q*9+k]; if (!isfinite(F[k])) qbad = true; }
    if (Sq) for (int k=0;k<3;k++) if (!isfinite(Sq[(size_t)q*4+k])) qbad = true;
    if (qbad && mQ[q] == NSD_UNSET) mQ[q] = code;
    float vq = Vq[q];
    if (!(vq > 0.f)) return;
    float A[9] = {0,0,0,0,0,0,0,0,0};
    if (y) { int ids[27]; qnodes(q, IP, cb, ids);
        for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; int id = ids[o];
            for (int a=0;a<3;a++) { float ya = y[id*3+a]; A[a*3] += ya*bg[0]; A[a*3+1] += ya*bg[1]; A[a*3+2] += ya*bg[2]; } } }
    float X[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) X[a*3+b] = F[a*3+b] + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b];
    double J = (double)X[0]*((double)X[4]*X[8] - (double)X[5]*X[7]) - (double)X[1]*((double)X[3]*X[8] - (double)X[5]*X[6]) + (double)X[2]*((double)X[3]*X[7] - (double)X[4]*X[6]);
    if (isfinite(J)) { if ((float)J < jmin[q]) jmin[q] = (float)J; if (J <= 0.0 && mJ[q] == NSD_UNSET) mJ[q] = code; }
    if (!dostress || mS[q] != NSD_UNSET) return;
    float U[9], S[3], V[9];
    ssvdf(X, U, S, V);
    bool bad = false;
    for (int k=0;k<9;k++) if (!isfinite(U[k]) || !isfinite(V[k])) bad = true;
    float ys = Sq[(size_t)q*4], lj = Sq[(size_t)q*4+1], al = Sq[(size_t)q*4+2];
    float sev = (mat == 7) ? prm[14] : 0.f;
    float se[3]; for (int i=0;i<3;i++) { se[i] = (sev > 0.f) ? fmaxf(S[i], sev) : S[i]; if (!isfinite(S[i])) bad = true; }
    if (!(prm[15] > 0.5f)) pl_rm(mat, se, prm, ys, lj, al, dt);
#ifdef AF_DMG
    al = dmg_g(Sq, (size_t)q);                                               // 절 CM19: aq_elem 과 같은 열화 g
#endif
    float Pe[3]; pl_pi(mat, se, prm, al, Pe);
    float ss[3], p[3];
    for (int i=0;i<3;i++) { ss[i] = (sev > 0.f) ? fmaxf(S[i], sev) : ((fabsf(S[i]) > 1e-6f) ? S[i] : ((S[i] < 0.f) ? -1e-6f : 1e-6f)); p[i] = Pe[i]*se[i]/ss[i]; }
#ifdef AF_JBAR
    aq_jbar_p(S, prm, p);                                                     // 절 CM13 판 (1): aq_elem 과 같은 장벽 몫
#endif
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = U[a*3]*p[0]*V[b*3] + U[a*3+1]*p[1]*V[b*3+1] + U[a*3+2]*p[2]*V[b*3+2]; if (!isfinite(t)) bad = true; }
    if (bad) mS[q] = code;
}

//@@ AO_SRC 62896

#ifndef AF_CTAN_MODE
#define AF_CTAN_MODE 0
#endif

__device__ __forceinline__ int ao_dpf(const float* S, float cdp, float* sgn, float* sg, float* ep, float* e, float& tr, float& rho, float& s) {
    for (int i=0;i<3;i++) { sgn[i] = (S[i] < 0.f) ? -1.f : 1.f; sg[i] = fmaxf(fabsf(S[i]), 1e-14f); ep[i] = logf(sg[i]); }
    tr = ep[0] + ep[1] + ep[2];
    for (int i=0;i<3;i++) e[i] = ep[i] - tr/3.f;
    rho = sqrtf(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    float dg = rho + cdp*tr;
    s = 1.f;
    if (!(dg > 0.f)) return 0;
    if (tr > 0.f) { s = 0.f; return 2; }
    s = -cdp*tr/rho;
    return 1;
}
__global__ void ao_vzero(int N, const int* IP, const float* Trel, const float* tbuf, float* v) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    if (Trel[p] > tbuf[0]) { v[p*3] = 0.f; v[p*3+1] = 0.f; v[p*3+2] = 0.f; }
}
// 구속 입자 질량을 걸음 k x 표본 s 시각마다 노드에 (acc[n*KS + k*S + s])
__global__ void ao_pbc_acc(int N, const int* IP, const int* pb, const float* fx, const float* m, const float* Trel, const float* tbuf, int K, int S, float dt, float* acc) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float tr = Trel[p], t0 = tbuf[0];
    if (!(tr > t0)) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8], KS = K*S;
    float mp = m[p];
    int ncut = 0;
    for (int ks=0; ks<KS; ks++) { int kk = ks / S, ss = ks % S; float ts = t0 + ((float)kk + ((float)ss + 0.5f)/(float)S)*dt; if (tr > ts) ncut = ks + 1; }
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k]*mp;
        if (ncut > 0) atomicAdd(&acc[(size_t)nid*KS + ncut - 1], ww);       // 표본 시각은 커지는 순서 -> 구속된 표본은 앞쪽 ncut 개 (끝 자리에만 더하고 노드에서 뒤쪽 누적합)
    }
}
__global__ void ao_pbc_k(const int* IP, const float* mn, const float* acc, int K, int S, int k, float expo, float* kb) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n], lk = 0.f; int KS = K*S;
    if (m > 0.f) {
        float suf = 0.f;                                                        // 표본 ks 의 구속 질량 = Σ_{j ≥ ks} acc[j]
        for (int j=KS-1; j>=k*S; j--) { suf += acc[(size_t)n*KS + j]; if (j < (k+1)*S) { float al = fminf(suf/m, 1.f); lk += log1pf(-al); } }
    }
    kb[n] = expf(lk*expo);
}
__global__ void ao_floor_n(const int* IP, const float* an, const float* mqn, float kex, int floor_, int vout, float al, float n0, float n1, float n2,
                           const unsigned char* fixd, const float* vbc, const float* U, float tau, const float* Ut, const float* z0, float* y, float* sk, int* pm, float* yh, float* eta,
                           const float* kb, const float* mn, float mthr, int y0uk, const float* vbt, int slip) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float k = 1.f; int p = 0;
    if (floor_) {
        float r = (mqn[n] > 0.f) ? fminf(an[n]/mqn[n], 1.f) : 0.f;
        k = expf(kex*log1pf(-r));
        float vv = al*((y[n*3]-z0[n*3])*n0 + (y[n*3+1]-z0[n*3+1])*n1 + (y[n*3+2]-z0[n*3+2])*n2);
        p = (vout && vv > 0.f) ? 1 : 0;
        if (slip) { if (vv > 0.f) { k = 1.f; p = 0; } else p = 3; }        // 미끄럼 바닥 (GF bounding_box): 들어가는 법선 성분만 고착 몫
    }
    float vt[3] = {0.f, 0.f, 0.f};
    if (kb) { float b = kb[n]; if (b < 1.f) { k *= b; p = 0; if (vbt) for (int a=0;a<3;a++) vt[a] = vbt[n*3+a]/al; } }
    if (fixd && fixd[n]) p = 2;
    if (p == 2) { for (int a=0;a<3;a++) { float yy = U[n*3+a] + tau*vbc[n*3+a]; yh[n*3+a] = yy; eta[n*3+a] = 0.f; y[n*3+a] = yy; } sk[n] = 0.f; pm[n] = 2; return; }
    if (mn[n] < mthr) { for (int a=0;a<3;a++) { float yy = Ut[n*3+a]; yh[n*3+a] = yy; eta[n*3+a] = 0.f; y[n*3+a] = yy; } sk[n] = 0.f; pm[n] = 2; return; }
    float s = sqrtf(k);
    float d[3];
    for (int a=0;a<3;a++) d[a] = z0[n*3+a] + vt[a] - Ut[n*3+a];
    float Pd[3];
    if (p == 0) { for (int a=0;a<3;a++) Pd[a] = d[a]; }
    else if (p == 3) { float dn = d[0]*n0 + d[1]*n1 + d[2]*n2; Pd[0] = dn*n0; Pd[1] = dn*n1; Pd[2] = dn*n2; }
    else { float dn = d[0]*n0 + d[1]*n1 + d[2]*n2; Pd[0] = d[0]-dn*n0; Pd[1] = d[1]-dn*n1; Pd[2] = d[2]-dn*n2; }
    float h[3]; for (int a=0;a<3;a++) h[a] = Ut[n*3+a] + (1.f-k)*Pd[a];
    float e1[3], e2[3];
    for (int a=0;a<3;a++) { e1[a] = y[n*3+a] - Ut[n*3+a]; e2[a] = y[n*3+a] - h[a]; }
    float et[3];
    if (p == 0) { for (int a=0;a<3;a++) et[a] = s*e2[a]; }
    else if (p == 3) { float c1 = e1[0]*n0 + e1[1]*n1 + e1[2]*n2; float c2 = e2[0]*n0 + e2[1]*n1 + e2[2]*n2;
           for (int a=0;a<3;a++) { float nn_ = (a==0) ? n0 : ((a==1) ? n1 : n2); et[a] = (e1[a] - c1*nn_) + s*c2*nn_; } }
    else { float c1 = e1[0]*n0 + e1[1]*n1 + e1[2]*n2; float c2 = e2[0]*n0 + e2[1]*n1 + e2[2]*n2;
           for (int a=0;a<3;a++) { float nn_ = (a==0) ? n0 : ((a==1) ? n1 : n2); et[a] = c1*nn_ + s*(e2[a] - c2*nn_); } }
    if (y0uk) {                                                                  // 뉴턴 시작점 = 걸음 시작 위치 U (T 의 역: 법선 성분 그대로, 나머지 /√k, k = 0 이면 0; 절 AN.err3 처치 (a))
        float isk = (s > 1e-6f) ? 1.f/s : 0.f;
        float dd[3]; for (int a=0;a<3;a++) dd[a] = U[n*3+a] - h[a];
        if (p == 0) { for (int a=0;a<3;a++) et[a] = isk*dd[a]; }
        else if (p == 3) { float dn = dd[0]*n0 + dd[1]*n1 + dd[2]*n2; float nv[3] = {n0, n1, n2}; for (int a=0;a<3;a++) et[a] = (dd[a] - dn*nv[a]) + isk*dn*nv[a]; }
        else { float dn = dd[0]*n0 + dd[1]*n1 + dd[2]*n2; float nv[3] = {n0, n1, n2}; for (int a=0;a<3;a++) et[a] = dn*nv[a] + isk*(dd[a] - dn*nv[a]); }
    }
    float ty[3]; Tapply(s, p, n0, n1, n2, et, ty);
    for (int a=0;a<3;a++) { yh[n*3+a] = h[a]; eta[n*3+a] = et[a]; y[n*3+a] = h[a] + ty[a]; }
    sk[n] = s; pm[n] = p;
}
// 모래 요소: mode 1 기울기 (RM 응력), 2 헤시안 (시험 X Hencky 탄성, 투영), 8 노드 대각 융합. st 수 IPq[0..1] (꼬리, 꼭짓점) 원자 합 (진단)
__global__ void __launch_bounds__(128) ao_elem(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, float mu, float lam, float cdp, int mode,
                        double* f, float* H, int capQ, float* B, int* cnt) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    int nq = IP[12];
    bool on = q < nq;
    int qq = on ? q : 0;
    bool head = (threadIdx.x & 7) == 0;
    int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = y[id*3+a]; X[a*3] += ya*g[0]; X[a*3+1] += ya*g[1]; X[a*3+2] += ya*g[2]; } }
    float U[9], S[3], V[9];
    ssvdf(X, U, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    if (cnt && on && st > 0 && (mode & 1)) atomicAdd(&cnt[st-1], 1);
    float K3 = lam + 2.f*mu/3.f;
    float ss[3], p[3], pe[3];
    for (int i=0;i<3;i++) {
        float tau = (st == 2) ? 0.f : ((st == 1) ? (K3*tr + 2.f*mu*s*e[i]) : (2.f*mu*ep[i] + lam*tr));
        ss[i] = sgn[i]*sg[i]; p[i] = tau/ss[i]; pe[i] = (2.f*mu*ep[i] + lam*tr)/ss[i];
    }
    float vq = on ? Vq[qq] : 0.f;
    if (mode & 1) {
        float Pm[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) Pm[a*3+b] = U[a*3]*p[0]*V[b*3] + U[a*3+1]*p[1]*V[b*3+1] + U[a*3+2]*p[2]*V[b*3+2];
        for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
            for (int a=0;a<3;a++) { double sv = seg8d((double)(vq*(Pm[a*3]*g[0] + Pm[a*3+1]*g[1] + Pm[a*3+2]*g[2])));
                if (on && head) atomicAdd(&f[ids[o]*3+a], sv); } }
    }
    if (mode & 2) {
        float h[81]; for (int k=0;k<81;k++) h[k] = 0.f;
        if (on) {
        float D[3][3], Ev[3][3], Dk[3][3];
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) { D[i][j] = (((i==j) ? 2.f*mu : 0.f) + lam)/(ss[i]*ss[j]) - ((i==j) ? pe[i]/ss[i] : 0.f); Dk[i][j] = D[i][j]; }
        jac3f(D, Ev);
        for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (!(dm > 0.f) || !isfinite(dm)) continue;
            float ee[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) ee[a*3+b] = Ev[0][m]*U[a*3]*V[b*3] + Ev[1][m]*U[a*3+1]*V[b*3+1] + Ev[2][m]*U[a*3+2]*V[b*3+2];
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += dm*ee[r]*ee[c]; }
        const int PI[3] = {0,0,1}, PJ[3] = {1,2,2};
        const float r2 = 0.70710678f;
        for (int pp=0; pp<3; pp++) { int i = PI[pp], j = PJ[pp];
            float den = ss[i] + ss[j]; float lt = (fabsf(den) > 1e-12f) ? (pe[i] + pe[j])/den : 0.f;
            float dd = ss[i] - ss[j]; float lf = (fabsf(dd) > 1e-6f*(fabsf(ss[i]) + fabsf(ss[j]))) ? (pe[i] - pe[j])/dd : (Dk[i][i] - Dk[i][j]);
            lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = isfinite(lf) ? fmaxf(lf, 0.f) : 0.f;
            float et[9], ef[9];
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float x1 = U[a*3+i]*V[b*3+j], x2 = U[a*3+j]*V[b*3+i]; et[a*3+b] = r2*(x1-x2); ef[a*3+b] = r2*(x1+x2); }
            for (int r=0;r<9;r++) for (int c=0;c<9;c++) h[r*9+c] += lt*et[r]*et[c] + lf*ef[r]*ef[c]; }
        if (mode & 512) { int k = 0; for (int r=0;r<9;r++) for (int c=r;c<9;c++) ((__half*)H)[(size_t)(k++)*capQ + qq] = __float2half(h[r*9+c]/(2.f*mu + lam)); }   // f16 위 삼각
        else if (mode & 128) { int k = 0; for (int r=0;r<9;r++) for (int c=r;c<9;c++) H[(size_t)(k++)*capQ + qq] = h[r*9+c]; }   // 위 삼각 45 개 (h 는 비트 대칭)
        else for (int k=0;k<81;k++) H[(size_t)k*capQ + qq] = h[k];
        }
        if ((mode & 8) && (mode & 256)) {                                        // 노드 대각 (공유 메모리 판): 칸 8 점의 헤시안·F·V 를 공유 메모리에, 레인마다 노드 o = 레인, 레인+8, ... 을 맡아 8 점을 직접 합 (셔플 없음)
            __shared__ float shH[128][45]; __shared__ float shF[128][9]; __shared__ float shV[128];
            int tl = threadIdx.x;
            { int k = 0; for (int r=0;r<9;r++) for (int c=r;c<9;c++) shH[tl][k++] = h[r*9+c]; }
            for (int k=0;k<9;k++) shF[tl][k] = F[k];
            shV[tl] = vq;
            __syncwarp();
            int l8 = tl & 7, b8 = tl & ~7;
            for (int o = l8; o < 27; o += 8) {
                float mm[9] = {0,0,0,0,0,0,0,0,0};
                for (int j=0;j<8;j++) {
                    const float* hj = shH[b8+j]; float g[3]; gqn(shF[b8+j], BG + (j*27+o)*3, g); float vj = shV[b8+j];
                    for (int a=0;a<3;a++) for (int c=0;c<3;c++) { float t = 0.f;
                        for (int b=0;b<3;b++) for (int dd=0;dd<3;dd++) { int r_ = a*3+b, c_ = c*3+dd; int lo = r_ < c_ ? r_ : c_, hi = r_ < c_ ? c_ : r_;
                            t += hj[lo*9 - lo*(lo-1)/2 + (hi - lo)]*g[b]*g[dd]; }
                        mm[a*3+c] += vj*t; }
                }
                if (on) for (int k=0;k<9;k++) { if (AO_DET) AO_CB[((size_t)(qq>>3)*27+o)*9+k] = mm[k]; else atomicAdd(&B[ids[o]*9+k], mm[k]); }
            }
        } else if (mode & 8) {
            for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g);
                float mm[9];
                for (int a=0;a<3;a++) for (int c=0;c<3;c++) { float t = 0.f;
                    for (int b=0;b<3;b++) { float hg = h[(a*3+b)*9 + c*3]*g[0] + h[(a*3+b)*9 + c*3+1]*g[1] + h[(a*3+b)*9 + c*3+2]*g[2]; t += g[b]*hg; }
                    mm[a*3+c] = vq*t; }
                for (int k=0;k<9;k++) mm[k] = seg8f(mm[k]);
                if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], mm[k]); }
        }
    }
}
__device__ __forceinline__ bool ao_rm3(const float* X, float cdp, float* Y) {
    float U[9], S[3], V[9];
    ssvdf(X, U, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    if (st == 0) { for (int k=0;k<9;k++) Y[k] = X[k]; return false; }
#if AF_AO_RMPOS
    // wolffix 11-15 (--ao_rmpos, env AF_AO_RMPOS=1): PhysGaussian 공식 sand_return_mapping 과 같게 -- 소성 (꼬리·꼭짓점) 반환 결과 특이값은 부호 없이 exp(H) (꼭짓점 U Vᵀ).
    // 옛 판은 부호 sgn 을 남겨 뒤집힌 X 가 반환 뒤에도 뒤집힌 채 (꼭짓점이면 det −1 거울 무응력 상태) 로 굳었다 (wolf 끝 프레임 det F −1 입자, 위에 얹힌 뒤집힌 덩어리).
    float sn[3]; for (int i=0;i<3;i++) sn[i] = (st == 2) ? 1.f : expf(tr/3.f + s*e[i]);
#else
    float sn[3]; for (int i=0;i<3;i++) sn[i] = (st == 2) ? sgn[i] : sgn[i]*expf(tr/3.f + s*e[i]);
#endif
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Y[a*3+b] = U[a*3]*sn[0]*V[b*3] + U[a*3+1]*sn[1]*V[b*3+1] + U[a*3+2]*sn[2]*V[b*3+2];
    return true;
}
__device__ __forceinline__ void ao_mm3(const float* A, const float* B, float* C) {
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) C[a*3+b] = A[a*3]*B[b] + A[a*3+1]*B[3+b] + A[a*3+2]*B[6+b];
}
// 걸음 끝 커밋: X = F_q + Σ U ⊗ g (U = 걸음 끝 변위), F_q ← F_q X⁻¹ RM(X)
__global__ void ao_commit(const int* IP, const int* cb, const float* BG, float* Fq, const float* U, float cdp) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ua = U[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; } }
    float Y[9];
    if (!ao_rm3(X, cdp, Y)) return;
    float Xi[9]; inv3(X, Xi);
    float T1[9], T2[9]; ao_mm3(Xi, Y, T1); ao_mm3(F, T1, T2);
    bool ok = true; for (int k=0;k<9;k++) if (!isfinite(T2[k])) ok = false;
    if (ok) for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = T2[k];
}
// G2P (모래): 걸음별 U_k (Ust[k*capN*3 + ...]) 로 F ← RM(A_k A_{k−1}⁻¹ F), x += Σ w U_K, v = Σ w v_K. 입자 RM 수 cnt[2..3]
__global__ void ao_g2p(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* Ust, int K, int capN, const float* vk, float cdp,
                       float* x, float* v, float* F, int* cnt) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    float Ap[9] = {1,0,0, 0,1,0, 0,0,1};
    float u[3] = {0,0,0}, vv[3] = {0,0,0};
    int nrm = 0;
    for (int kk=0; kk<K; kk++) {
        const float* Uk = Ust + (size_t)kk*capN*3;
        float A[9] = {1,0,0, 0,1,0, 0,0,1};
        bool last = kk == K-1;
        for (int o=0;o<27;o++) {
            int i = o/9, j = (o/3)%3, k = o%3;
            int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
            float ww = w[0][i]*w[1][j]*w[2][k];
            float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
            float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
            for (int a=0;a<3;a++) { float uu = Uk[nid*3+a]; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; if (last) { u[a] += ww*uu; vv[a] += ww*vk[nid*3+a]; } }
        }
        float Api[9]; inv3(Ap, Api);
        float Ainc[9], Ft[9]; ao_mm3(A, Api, Ainc); ao_mm3(Ainc, Fp, Ft);
        float Y[9]; if (ao_rm3(Ft, cdp, Y)) nrm++;
        for (int k=0;k<9;k++) { Fp[k] = Y[k]; Ap[k] = A[k]; }
    }
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; }
    for (int k=0;k<9;k++) F[(size_t)p*9+k] = Fp[k];
    if (cnt && nrm) atomicAdd(&cnt[2], 1);
}
// θ 방식 걸음 끝 (절 AN.err2 정의): y = U_k + θ (U_{k+1} − U_k), v_{k+1} = (y − z0)/(θτ), U_{k+1} = U_k + (y − U_k)/θ
__global__ void ao_stepend_th(const int* IP, float ith, float al, const float* y, const float* z0, float* U, float* vk, const unsigned char* fixd, const float* vbc, float damp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) {
        float yy = y[n*3+a], u = U[n*3+a];
        float v1 = al*(yy - z0[n*3+a]);
        if (fixd && fixd[n]) v1 = vbc[n*3+a];
        U[n*3+a] = u + (yy - u)*ith;
        vk[n*3+a] = v1*damp;
    }
}
__global__ void ao_act_reset(int* IP) { IP[14] = 0; }
__global__ void ao_act(int* IP, const float* mn) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (mn[n] > 0.f) { int k = atomicAdd(&IP[14], 1); AO_ACT[k] = n; }
}
// PCG 첫 추정 (앞 걸음 첫 뉴턴의 δ): rt_grad 뒤에 dl = d0, r = −g − A d0, z = D r, p = z, Tp = T p, SC0 = r·z
__global__ void ao_tvec(const int* IP, const float* sk, const int* pm, float n0, float n1, float n2, const float* v, float* o) {
    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;
    if (ii_ >= IP[14]) return;
    int n = AO_ACT[ii_];
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, v + n*3, t); for (int a=0;a<3;a++) o[n*3+a] = t[a];
}
__global__ void ao_warm(const int* IP, const float* mn, float itau2, const float* sk, const int* pm, float n0, float n1, float n2, const float* d0, const float* KTd,
                        const float* Dinv, float* dl, float* r, float* z, float* p, float* Tp, double* SC) {
    if (blockIdx.x*blockDim.x >= IP[14]) return;
    int ii_ = blockIdx.x*blockDim.x + threadIdx.x;
    double rz = 0.0;
    if (ii_ < IP[14]) {
        int n = AO_ACT[ii_];
        float Mi = mn[n]*itau2;
        float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, KTd + n*3, t);
        float rr[3];
        for (int a=0;a<3;a++) { float d = d0[n*3+a]; dl[n*3+a] = d; rr[a] = r[n*3+a] - (Mi*d + t[a]); r[n*3+a] = rr[a]; }
        const float* D = Dinv + n*9;
        float zz[3]; for (int a=0;a<3;a++) { zz[a] = D[a*3]*rr[0] + D[a*3+1]*rr[1] + D[a*3+2]*rr[2]; z[n*3+a] = zz[a]; p[n*3+a] = zz[a]; rz += (double)rr[a]*zz[a]; }
        float tp[3]; Tapply(sk[n], pm[n], n0, n1, n2, zz, tp); for (int a=0;a<3;a++) Tp[n*3+a] = tp[a];
    }
    blocksum_add(rz, &SC[0]);
}

// 헤시안 곱 (칸 공유 메모리 판, 위 삼각 45 저장): 칸 27 노드 입력을 공유 메모리에 한 번, 레인마다 노드 o = 레인 + 8k 를 맡아 8 점의 V (H δX)·g 를 직접 합
__global__ void __launch_bounds__(128) ao_mv_c(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* H, const float* dy, float* out, int capQ) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    __shared__ float sY[16][27][3]; __shared__ float sP[128][9]; __shared__ float sF[128][9];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int tl = threadIdx.x, l8 = tl & 7, b8 = tl & ~7, cl = tl >> 3; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    for (int o = l8; o < 27; o += 8) { int id = ids[o]; for (int a=0;a<3;a++) sY[cl][o][a] = dy[id*3+a]; }
    float F[9]; for (int k=0;k<9;k++) { F[k] = Fq[(size_t)qq*9+k]; sF[tl][k] = F[k]; }
    __syncwarp();
    float dX[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); for (int a=0;a<3;a++) { float ya = sY[cl][o][a]; dX[a*3] += ya*g[0]; dX[a*3+1] += ya*g[1]; dX[a*3+2] += ya*g[2]; } }
    float vq = on ? Vq[qq] : 0.f;
    for (int r=0;r<9;r++) { float t = 0.f; for (int c=0;c<9;c++) { int a_ = r < c ? r : c, b_ = r < c ? c : r; t += H[(size_t)(a_*9 - a_*(a_-1)/2 + (b_ - a_))*capQ + qq]*dX[c]; } sP[tl][r] = vq*t; }
    __syncwarp();
    for (int o = l8; o < 27; o += 8) {
        float sv[3] = {0.f, 0.f, 0.f};
        for (int j=0;j<8;j++) { float g[3]; gqn(sF[b8+j], BG + (j*27+o)*3, g); const float* dP = sP[b8+j];
            for (int a=0;a<3;a++) sv[a] += dP[a*3]*g[0] + dP[a*3+1]*g[1] + dP[a*3+2]*g[2]; }
        if (on) for (int a=0;a<3;a++) { if (AO_DET) AO_CV[((size_t)(qq>>3)*27+o)*3+a] = sv[a]; else atomicAdd(&out[ids[o]*3+a], sv[a]); }
    }
}
// ---- 주값 분해 저장 판 (--hspec): 탄성 접선 헤시안 = Σ_m dm e_m e_mᵀ + Σ_쌍 (lt e_t e_tᵀ + lf e_f e_fᵀ), e = U Ê Vᵀ (순위 1 항 9 개).
// 점마다 H[k*capQ + q] (k = 0..29): U (9), M = F V (9), D⁺ = Ev diag(dm) Evᵀ 위 삼각 (00 11 22 01 02 12) x V_q, c1 = (lt+lf)/2 x V_q (쌍 01 02 12), c2 = (lf−lt)/2 x V_q.
// 곱: dX = dA F (dA = Σ_o y_o ⊗ ∇w_o), X̂ = Uᵀ dA M, P̂_ii = (D⁺ diag X̂)_i, P̂_ij = c1 X̂_ij + c2 X̂_ji, 노드 o 몫 = (U P̂ Mᵀ) ∇w_o.
// 노드 대각: h = Mᵀ ∇w_o, S = diag(h) D⁺ diag(h) + 쌍 ((i,i) += c1 h_j², (j,j) += c1 h_i², (i,j) += c2 h_i h_j), 블록 = U S Uᵀ.
__device__ __forceinline__ void ao_pairs(int pp, int& i, int& j) { i = (pp == 2) ? 1 : 0; j = (pp == 0) ? 1 : 2; }
__global__ void __launch_bounds__(128) ao_elem_s(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, float mu, float lam, float cdp, int mode,
                        double* f, float* H, int capQ, float* B, int* cnt) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    __shared__ float sY[16][27][3]; __shared__ float sQ[128][9]; __shared__ float sU[128][9]; __shared__ float sM[128][9]; __shared__ float sC[128][12];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    int nq = IP[12];
    bool on = q < nq;
    int qq = on ? q : 0;
    int tl = threadIdx.x, l8 = tl & 7, b8 = tl & ~7, cl = tl >> 3, oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    for (int o = l8; o < 27; o += 8) { int id = ids[o]; for (int a=0;a<3;a++) sY[cl][o][a] = y[id*3+a]; }
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    __syncwarp();
    float A[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; for (int a=0;a<3;a++) { float ya = sY[cl][o][a]; A[a*3] += ya*bg[0]; A[a*3+1] += ya*bg[1]; A[a*3+2] += ya*bg[2]; } }
    float X[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) X[a*3+b] = F[a*3+b] + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b];
    float U[9], S[3], V[9];
    ssvdf(X, U, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    if (cnt && on && st > 0) atomicAdd(&cnt[st-1], 1);
    float K3 = lam + 2.f*mu/3.f;
    float ss[3], p[3], pe[3];
    for (int i=0;i<3;i++) {
        float tau = (st == 2) ? 0.f : ((st == 1) ? (K3*tr + 2.f*mu*s*e[i]) : (2.f*mu*ep[i] + lam*tr));
        ss[i] = sgn[i]*sg[i]; p[i] = tau/ss[i]; pe[i] = (2.f*mu*ep[i] + lam*tr)/ss[i];
    }
    float vq = on ? Vq[qq] : 0.f;
    // 힘: Qf = V_q P Fᵀ (P = U diag(p) Vᵀ), 노드 o 몫 = Qf ∇w_o
    { float Pm[9];
      for (int a=0;a<3;a++) for (int b=0;b<3;b++) Pm[a*3+b] = vq*(U[a*3]*p[0]*V[b*3] + U[a*3+1]*p[1]*V[b*3+1] + U[a*3+2]*p[2]*V[b*3+2]);
      for (int a=0;a<3;a++) for (int c=0;c<3;c++) sQ[tl][a*3+c] = Pm[a*3]*F[c*3] + Pm[a*3+1]*F[c*3+1] + Pm[a*3+2]*F[c*3+2]; }
    // 헤시안 주값 계수
    float M[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) M[a*3+b] = F[a*3]*V[b] + F[a*3+1]*V[3+b] + F[a*3+2]*V[6+b];
    float C[12];
    // 절 CM138 --ctan: 헤시안을 반환 사상과 일관되게 (AF_CTAN_MODE 0 = 옛 판: 시험 X 의 탄성 접선 ('초기 강성법'); 1 = 꼭짓점 (st 2, 응력 0) 점의 접선 0; 2 = 1 + 꼬리 (st 1) 점의 DP 반환 일관 접선 (대칭화))
    float Cc[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) Cc[i][j] = ((i==j) ? 2.f*mu : 0.f) + lam;
#if AF_CTAN_MODE >= 1
    if (st == 2) { for (int i=0;i<3;i++) { pe[i] = 0.f; for (int j=0;j<3;j++) Cc[i][j] = 0.f; } }
#endif
#if AF_CTAN_MODE == 3
    // 절 CM175 --ctan 3: 모드 1 + 꼭짓점 (st 2) 점의 접선을 체적 압축 방향 탄성 체적 강성만 (K3 1 1ᵀ, 편차·기하 항 0) -- 꼭짓점에서 압축으로 가면 탄성으로 돌아오므로 (반환 사상 접선의 압축 쪽 요소)
    if (st == 2) { for (int i=0;i<3;i++) for (int j=0;j<3;j++) Cc[i][j] = K3; }
#endif
#if AF_CTAN_MODE == 2
    if (st == 1) {
        float nn_[3], beta = -2.f*mu*cdp;
        for (int i=0;i<3;i++) nn_[i] = e[i]/rho;
        float Ct[3][3];
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) Ct[i][j] = K3 + beta*(nn_[i] + (tr/rho)*(((i==j) ? 1.f : 0.f) - 1.f/3.f - nn_[i]*nn_[j]));
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) Cc[i][j] = 0.5f*(Ct[i][j] + Ct[j][i]);
        for (int i=0;i<3;i++) pe[i] = p[i];
    }
#endif
    { float D[3][3], Ev[3][3], Dk[3][3];
      for (int i=0;i<3;i++) for (int j=0;j<3;j++) { D[i][j] = Cc[i][j]/(ss[i]*ss[j]) - ((i==j) ? pe[i]/ss[i] : 0.f); Dk[i][j] = D[i][j]; }
      jac3f(D, Ev);
      float dm[3]; for (int m=0;m<3;m++) { float d_ = fmaxf(D[m][m], 0.f); dm[m] = (d_ > 0.f && isfinite(d_)) ? d_ : 0.f; }
      const int RI[6] = {0,1,2,0,0,1}, RJ[6] = {0,1,2,1,2,2};
      for (int k=0;k<6;k++) C[k] = vq*(dm[0]*Ev[RI[k]][0]*Ev[RJ[k]][0] + dm[1]*Ev[RI[k]][1]*Ev[RJ[k]][1] + dm[2]*Ev[RI[k]][2]*Ev[RJ[k]][2]);
      for (int pp=0; pp<3; pp++) { int i, j; ao_pairs(pp, i, j);
          float den = ss[i] + ss[j]; float lt = (fabsf(den) > 1e-12f) ? (pe[i] + pe[j])/den : 0.f;
          float dd = ss[i] - ss[j]; float lf = (fabsf(dd) > 1e-6f*(fabsf(ss[i]) + fabsf(ss[j]))) ? (pe[i] - pe[j])/dd : (Dk[i][i] - Dk[i][j]);
          lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = isfinite(lf) ? fmaxf(lf, 0.f) : 0.f;
          C[6+pp] = vq*0.5f*(lt + lf); C[9+pp] = vq*0.5f*(lf - lt); } }
    if (on) { for (int k=0;k<9;k++) { H[(size_t)k*capQ + qq] = U[k]; H[(size_t)(9+k)*capQ + qq] = M[k]; } for (int k=0;k<12;k++) H[(size_t)(18+k)*capQ + qq] = C[k]; }
    for (int k=0;k<9;k++) { sU[tl][k] = U[k]; sM[tl][k] = M[k]; }
    for (int k=0;k<12;k++) sC[tl][k] = C[k];
    __syncwarp();
    for (int o = l8; o < 27; o += 8) {
        double fs[3] = {0.0, 0.0, 0.0};
        float bb[6] = {0,0,0,0,0,0};
        for (int j=0;j<8;j++) {
            const float* bg = BG + (j*27+o)*3;
            const float* Qj = sQ[b8+j];
            for (int a=0;a<3;a++) fs[a] += (double)(Qj[a*3]*bg[0] + Qj[a*3+1]*bg[1] + Qj[a*3+2]*bg[2]);
            const float* Mj = sM[b8+j]; const float* Cj = sC[b8+j]; const float* Uj = sU[b8+j];
            float h[3]; for (int b=0;b<3;b++) h[b] = Mj[b]*bg[0] + Mj[3+b]*bg[1] + Mj[6+b]*bg[2];
            float S6[6] = {Cj[0]*h[0]*h[0] + Cj[6]*h[1]*h[1] + Cj[7]*h[2]*h[2],
                           Cj[1]*h[1]*h[1] + Cj[6]*h[0]*h[0] + Cj[8]*h[2]*h[2],
                           Cj[2]*h[2]*h[2] + Cj[7]*h[0]*h[0] + Cj[8]*h[1]*h[1],
                           (Cj[3] + Cj[9])*h[0]*h[1], (Cj[4] + Cj[10])*h[0]*h[2], (Cj[5] + Cj[11])*h[1]*h[2]};
            float Sm[9] = {S6[0], S6[3], S6[4], S6[3], S6[1], S6[5], S6[4], S6[5], S6[2]};
            float T[9]; for (int a=0;a<3;a++) for (int c=0;c<3;c++) T[a*3+c] = Uj[a*3]*Sm[c] + Uj[a*3+1]*Sm[3+c] + Uj[a*3+2]*Sm[6+c];
            const int RI[6] = {0,1,2,0,0,1}, RJ[6] = {0,1,2,1,2,2};
            for (int k=0;k<6;k++) bb[k] += T[RI[k]*3]*Uj[RJ[k]*3] + T[RI[k]*3+1]*Uj[RJ[k]*3+1] + T[RI[k]*3+2]*Uj[RJ[k]*3+2];
        }
        float mm[9] = {bb[0], bb[3], bb[4], bb[3], bb[1], bb[5], bb[4], bb[5], bb[2]};
        if (on) {
            for (int a=0;a<3;a++) { if (AO_DET) AO_CF[((size_t)(qq>>3)*27+o)*3+a] = fs[a]; else atomicAdd(&f[ids[o]*3+a], fs[a]); }
            for (int k=0;k<9;k++) { if (AO_DET) AO_CB[((size_t)(qq>>3)*27+o)*9+k] = mm[k]; else atomicAdd(&B[ids[o]*9+k], mm[k]); }
        }
    }
}
__global__ void __launch_bounds__(128) ao_mv_q(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* H, const float* dy, float* out, int capQ) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    __shared__ float sY[16][27][3]; __shared__ float sP[128][9];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12]; int qq = on ? q : 0; int tl = threadIdx.x, l8 = tl & 7, b8 = tl & ~7, cl = tl >> 3; int oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    for (int o = l8; o < 27; o += 8) { int id = ids[o]; for (int a=0;a<3;a++) sY[cl][o][a] = dy[id*3+a]; }
    __syncwarp();
    float A[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; for (int a=0;a<3;a++) { float ya = sY[cl][o][a]; A[a*3] += ya*bg[0]; A[a*3+1] += ya*bg[1]; A[a*3+2] += ya*bg[2]; } }
    float U[9], M[9], C[12];
    for (int k=0;k<9;k++) { U[k] = H[(size_t)k*capQ + qq]; M[k] = H[(size_t)(9+k)*capQ + qq]; }
    for (int k=0;k<12;k++) C[k] = on ? H[(size_t)(18+k)*capQ + qq] : 0.f;
    float T[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) T[a*3+b] = A[a*3]*M[b] + A[a*3+1]*M[3+b] + A[a*3+2]*M[6+b];
    float Xh[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Xh[a*3+b] = U[a]*T[b] + U[3+a]*T[3+b] + U[6+a]*T[6+b];
    float Ph[9];
    Ph[0] = C[0]*Xh[0] + C[3]*Xh[4] + C[4]*Xh[8];
    Ph[4] = C[3]*Xh[0] + C[1]*Xh[4] + C[5]*Xh[8];
    Ph[8] = C[4]*Xh[0] + C[5]*Xh[4] + C[2]*Xh[8];
    Ph[1] = C[6]*Xh[1] + C[9]*Xh[3];  Ph[3] = C[9]*Xh[1] + C[6]*Xh[3];
    Ph[2] = C[7]*Xh[2] + C[10]*Xh[6]; Ph[6] = C[10]*Xh[2] + C[7]*Xh[6];
    Ph[5] = C[8]*Xh[5] + C[11]*Xh[7]; Ph[7] = C[11]*Xh[5] + C[8]*Xh[7];
    float W[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) W[a*3+b] = U[a*3]*Ph[b] + U[a*3+1]*Ph[3+b] + U[a*3+2]*Ph[6+b];
    for (int a=0;a<3;a++) for (int c=0;c<3;c++) sP[tl][a*3+c] = W[a*3]*M[c*3] + W[a*3+1]*M[c*3+1] + W[a*3+2]*M[c*3+2];
    __syncwarp();
    for (int o = l8; o < 27; o += 8) {
        float sv[3] = {0.f, 0.f, 0.f};
        for (int j=0;j<8;j++) { const float* bg = BG + (j*27+o)*3; const float* Q = sP[b8+j];
            for (int a=0;a<3;a++) sv[a] += Q[a*3]*bg[0] + Q[a*3+1]*bg[1] + Q[a*3+2]*bg[2]; }
        if (on) for (int a=0;a<3;a++) { if (AO_DET) AO_CV[((size_t)(qq>>3)*27+o)*3+a] = sv[a]; else atomicAdd(&out[ids[o]*3+a], sv[a]); }
    }
}
// ---- generalized-α / HHT (절 AN.err3 증분 포텐셜 꼴): 힘을 y = (1−αf) U_{n+1} + αf U_n 에서, 노드 가속도 상태 ga (3 x 노드)
__global__ void ao_step0_ga(const int* IP, const float* U, const float* vk, const float* ga, float dt, float am, float af, float gm, float be, float tau, float g0, float g1, float g2,
                            float* Ut, float* z0, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float gg[3] = {g0, g1, g2};
    float c1 = (1.f - af)*be*dt*dt, al = gm*dt/c1;
    for (int a=0;a<3;a++) {
        float u = U[n*3+a], v = vk[n*3+a], an = ga[n*3+a];
        float xp = u + dt*v + dt*dt*(0.5f - be)*an;
        float ys = af*u + (1.f - af)*xp;
        z0[n*3+a] = ys - (v + dt*(1.f - gm)*an)/al;
        float ut = ys + tau*tau*(gg[a] - am*an);
        Ut[n*3+a] = ut; y[n*3+a] = ut;                                       // 고착 판정·바깥 법선 판정은 관성 예측으로 (시작점은 ao_floor_n y0uk)
    }
}
// 걸음 끝: v = α (y − z0), a_{n+1} = (v − v_n − dt (1−γ) a_n)/(γ dt) (Newmark 속도 식, 고착 뒤 속도와 맞게), U = U_n + dt v_n + dt² ((½−β) a_n + β a_{n+1}) (처치 (c)),
// 고착 몫 있는 노드 (an/mqn > 0) 와 구속 노드 (k_구속 < 1) 는 a_{n+1} = 0 (처치 (b); 구속은 같은 이유로 덧붙임)
__global__ void ao_stepend_ga(const int* IP, float dt, float gm, float be, float al, const float* y, const float* z0, float* U, float* vk, float* ga,
                              const float* anf, const float* mqn, const float* kb, float damp) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    bool st = (anf && mqn[n] > 0.f && anf[n] > 0.f) || (kb && kb[n] < 1.f);
    for (int a=0;a<3;a++) {
        float u = U[n*3+a], v = vk[n*3+a], an = ga[n*3+a];
        float v1 = al*(y[n*3+a] - z0[n*3+a]);
        float a1 = (v1 - v - dt*(1.f - gm)*an)/(gm*dt);
        U[n*3+a] = u + dt*v + dt*dt*((0.5f - be)*an + be*a1);
        vk[n*3+a] = v1*damp;
        ga[n*3+a] = st ? 0.f : a1;
    }
}
__global__ void ao_p2g_a(int N, const int* IP, const int* pb, const float* fx, const float* m, const float* pa, float* acc) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float mp = m[p];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k]*mp;
        for (int a=0;a<3;a++) atomicAdd(&acc[nid*3+a], ww*pa[p*3+a]);
    }
}
// 질량 0 노드 (가우스 점 이웃 27 노드 안인데 입자 B-스플라인 가중이 정확히 0 -- 격자 정렬 채움 입자에서 생김) 의 속도·가속도:
// 이웃 26 노드 중 질량 있는 노드의 질량 가중 평균 (없으면 그대로 0). 질량 있는 노드만 읽으므로 제자리 갱신해도 경합 없음. 절 AQ.
__global__ void aq_vfill(const int* IP, const float* mn, float* vk, float* ga) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || mn[n] > 0.f) return;
    int nb1 = IP[7], nb2 = IP[8], nb0 = IP[6];
    int i0 = n / (nb1*nb2), j0 = (n / nb2) % nb1, k0 = n % nb2;
    float sw = 0.f, sv[3] = {0.f, 0.f, 0.f}, sa[3] = {0.f, 0.f, 0.f};
    for (int di=-1; di<=1; di++) for (int dj=-1; dj<=1; dj++) for (int dk=-1; dk<=1; dk++) {
        int i = i0+di, j = j0+dj, k = k0+dk;
        if (i < 0 || j < 0 || k < 0 || i >= nb0 || j >= nb1 || k >= nb2) continue;
        int m_ = (i*nb1 + j)*nb2 + k; float w = mn[m_];
        if (!(w > 0.f)) continue;
        sw += w; for (int a=0;a<3;a++) { sv[a] += w*vk[m_*3+a]; if (ga) sa[a] += w*ga[m_*3+a]; }
    }
    if (sw > 0.f) for (int a=0;a<3;a++) { vk[n*3+a] = sv[a]/sw; if (ga) ga[n*3+a] = sa[a]/sw; }
}
__global__ void ao_adiv(const int* IP, const float* mn, float* ga) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n];
    for (int a=0;a<3;a++) ga[n*3+a] = (m > 0.f) ? ga[n*3+a]/m : 0.f;
}
__global__ void ao_azero(int N, const float* Trel, const float* tbuf, float* pa) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    if (Trel[p] > tbuf[0]) { pa[p*3] = 0.f; pa[p*3+1] = 0.f; pa[p*3+2] = 0.f; }
}
// 입자 가속도 = Σ w a_K (마지막 걸음 노드 가속도), 다음 프레임 P2G 로 노드에
__global__ void ao_g2p_a(int N, const int* IP, const int* pb, const float* fx, const float* ga, float* pa) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float s[3] = {0.f, 0.f, 0.f};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k];
        for (int a=0;a<3;a++) s[a] += ww*ga[nid*3+a];
    }
    for (int a=0;a<3;a++) pa[p*3+a] = s[a];
}
// ======== 절 AQ: 재질 일반 (정답 pl_rm·pl_pi; 재질 3 foam, 5 plasticine, 7 watermelon) ========
// 가우스 점 상태 Sq[q*4] = (ys, logJp, 살아있는 몫 al, 0): 프레임 시작에 그 팔분 영역 입자 부피 가중 평균 (없으면 칸 평균)
__global__ void aq_p2g_s(int N, const int* IP, const float* fx, const int* pcd, const int* cidx, const float* vol, const float* ys, const float* lj, const float* al, float* Sa, float* Sc) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
    int q = c*8 + oc; float w = vol[p];
    float v4[4] = {w*ys[p], w*lj[p], w*al[p], w};
    for (int k=0;k<4;k++) { atomicAdd(&Sa[(size_t)q*4+k], v4[k]); atomicAdd(&Sc[(size_t)c*4+k], v4[k]); }
}
// 결정적 판: 칸마다 자기 입자 (칸 번호로 정렬, st 구간) 를 차례로
__global__ void aq_p2g_sd(const int* IP, const int* st, const float* fx, const float* vol, const float* ys, const float* lj, const float* al, float* Sa, float* Sc) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= IP[11] || IP[13]) return;
    float a[8][4]; for (int o=0;o<8;o++) for (int k=0;k<4;k++) a[o][k] = 0.f;
    float cc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int p = st[c]; p < st[c+1]; p++) {
        int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
        float w = vol[p]; float v4[4] = {w*ys[p], w*lj[p], w*al[p], w};
        for (int k=0;k<4;k++) { a[oc][k] += v4[k]; cc[k] += v4[k]; }
    }
    for (int o=0;o<8;o++) for (int k=0;k<4;k++) Sa[((size_t)c*8+o)*4+k] = a[o][k];
    for (int k=0;k<4;k++) Sc[(size_t)c*4+k] = cc[k];
}
__global__ void aq_gauss_s(const int* IP, const float* Sa, const float* Sc, float* Sq) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int c = q >> 3;
    const float* a = (Sa[(size_t)q*4+3] > 0.f) ? Sa + (size_t)q*4 : Sc + (size_t)c*4;
    float w = fmaxf(a[3], 1e-30f);
    Sq[(size_t)q*4] = a[0]/w; Sq[(size_t)q*4+1] = a[1]/w; Sq[(size_t)q*4+2] = a[2]/w; Sq[(size_t)q*4+3] = 0.f;
}
#ifdef AF_DMG
// 절 CM19 (사용자 승인 정의 변경, 재질 판 수박 + --elx snh 만): 손상 모델을 B-스플라인 가우스 점 구조에 얹음. AF_DMG 1 = 연속체 손상 (CDM), 2 = 위상장 파괴 (AT2).
//   ψ⁺ = Ψ_SNH(σ) − ψ⁻, ψ⁻ = κ/2 (J − 1)² [J < 1] (체적 압축 몫만 열화 안 함), σ = RM 뒤 주값 (탄성 상태). 구동량은 0 아래 자름, 넘침 막이 1e10 (물리 매개변수 아님).
//   상태: CDM 은 입자 alp = d_p (이력 최대), 가우스 점 Sq[2] = 팔분 영역 부피 가중 평균 d (걸음 끝 커밋에서 최대 갱신);
//         위상장은 입자 alp = H_p = max ψ⁺, 가우스 점 Sq[2] = 평균 H (커밋에서 최대 갱신), Sq[3] = 노드 d 의 그 점 보간 (pf_q), 입자 ysp = 노드 d 의 입자 보간 (출력만).
//   prm[10] = CDM ψ_c / 위상장 G_c, prm[11] = 위상장 l0.
__device__ __forceinline__ float dmg_psip(const float* s, const float* prm) {
    float mu = prm[0], lss = prm[1] + mu, kap = prm[2];
    float J = s[0]*s[1]*s[2], jm = J - 1.f;
    float ps = 0.5f*mu*(s[0]*s[0] + s[1]*s[1] + s[2]*s[2] - 3.f) - mu*jm + 0.5f*lss*jm*jm;
    if (J < 1.f) ps -= 0.5f*kap*jm*jm;
    return (ps > 0.f) ? fminf(ps, 1e10f) : 0.f;                                 // NaN -> 0
}
// CDM 국소 법칙 (매개변수 ψ_c 하나): ψ⁺ ≤ ψ_c 이면 d = 0, 넘으면 g(d) = (1 − d)² = ψ_c / ψ⁺ -- 열화된 인장 에너지 g ψ⁺ 가 ψ_c 에 묶임 (응력 ∝ 1/변형으로 연화)
__device__ __forceinline__ float dmg_cdm(float ps, float psc) { return (ps > psc) ? 1.f - sqrtf(psc/ps) : 0.f; }
__device__ __forceinline__ float dmg_g(const float* Sq, size_t q) {
#if AF_DMG == 1
    float d = Sq[q*4+2];
#else
    float d = Sq[q*4+3];
#endif
    d = fminf(fmaxf(d, 0.f), 1.f);
    return (1.f - d)*(1.f - d);
}
#endif
// 탄성 주값 1 차 PK (시험 σ, RM 없음) 와 주값 공간 대칭 접선 (중심 차분)
__device__ __noinline__ void aq_tan9(int mat, float s0, float s1, float s2, const float* prm, float al, float* pe, float* D9) {
    // 절 AR 고침: 예전 판 (2 차원 배열 인자·동적 첨자 sp[j], forceinline) 은 aq_elem 안에서 D[0][0]·D[0][1] 이 (1, 0) 으로 덮여 접선이 틀렸다
    // (단위 시험에서는 맞고 커널 안에서만 -- 대조 --check_aq: 모래를 재질 커널로 돌리면 해석 커널과 계수 35 % 차). 열마다 따로 펼쳐 평탄 배열로.
#ifdef AF_SNH
    float a0 = s0, a1 = s1, a2 = s2;                                             // 절 CM13 판 (2): Stable Neo-Hookean 은 뒤집힌 σ 에서도 정의 -> 부호 그대로 (크기로 바꾸지 않음)
#else
    float a0 = (mat == 7) ? fabsf(s0) : s0, a1 = (mat == 7) ? fabsf(s1) : s1, a2 = (mat == 7) ? fabsf(s2) : s2;   // neo-Hookean 은 J^{-2/3} 라 뒤집힌 시험 σ 에서 nan -> 크기로 (접선만)
#endif
    float c[3] = {a0, a1, a2};
    pl_pi(mat, c, prm, al, pe);
    float Dt[9];
    #pragma unroll
    for (int j=0;j<3;j++) {
        float sj = (j == 0) ? a0 : ((j == 1) ? a1 : a2);
        float h = fmaxf(1e-3f*fabsf(sj), 1e-4f);
        float sp[3] = {a0 + ((j == 0) ? h : 0.f), a1 + ((j == 1) ? h : 0.f), a2 + ((j == 2) ? h : 0.f)};
        float sm[3] = {a0 - ((j == 0) ? h : 0.f), a1 - ((j == 1) ? h : 0.f), a2 - ((j == 2) ? h : 0.f)};
        float pp[3], pm[3]; pl_pi(mat, sp, prm, al, pp); pl_pi(mat, sm, prm, al, pm);
        float ih = 0.5f/h;
        Dt[j] = (pp[0] - pm[0])*ih; Dt[3+j] = (pp[1] - pm[1])*ih; Dt[6+j] = (pp[2] - pm[2])*ih;
    }
    D9[0] = Dt[0]; D9[4] = Dt[4]; D9[8] = Dt[8];
    D9[1] = D9[3] = 0.5f*(Dt[1] + Dt[3]); D9[2] = D9[6] = 0.5f*(Dt[2] + Dt[6]); D9[5] = D9[7] = 0.5f*(Dt[5] + Dt[7]);
    for (int k=0;k<9;k++) if (!isfinite(D9[k])) D9[k] = 0.f;
    for (int k=0;k<3;k++) if (!isfinite(pe[k])) pe[k] = 0.f;
}
#ifdef AF_JBAR
// 절 CM13 판 (1) (사용자 승인 정의 변경): 가우스 점마다 J 로그 장벽 E_b = κ_J V_q φ(J), φ(J) = −log J + J − 1 (J = det X, X = 시험 변형 기울기).
//   문턱 없음 (d̂ 없음): φ(1) = φ'(1) = 0 이라 쉼 상태 응력·위치는 그대로, J → 0⁺ 에서 +∞. κ_J = prm[13] (= --jbar x μ).
//   주값: ∂φ/∂σ_i = (J − 1)/σ_i, ∂²φ/∂σ_i² = 1/σ_i², ∂²φ/∂σ_i∂σ_j = J/(σ_i σ_j) (i ≠ j); 비틀림 몫은 pe 를 통해 (아래 aq_elem 의 λ_t·λ_f 식, 음수는 0 으로 투영).
//   J ≤ 0 (시험 X 가 이미 뒤집힘) 에서는 장벽이 정의되지 않아 몫 0 (그 점은 진단 q_inv 로 셈).
__device__ __forceinline__ void aq_jbar_p(const float* S, const float* prm, float* p) {
    float J = S[0]*S[1]*S[2], kJ = prm[13];
    if (!(J > 0.f)) return;
    for (int i=0;i<3;i++) p[i] += kJ*(J - 1.f)/S[i];
}
__device__ __forceinline__ void aq_jbar_t(const float* S, const float* prm, float* pe, float D[3][3]) {
    float J = S[0]*S[1]*S[2], kJ = prm[13];
    if (!(J > 0.f)) return;
    for (int i=0;i<3;i++) { pe[i] += kJ*(J - 1.f)/S[i]; for (int j=0;j<3;j++) D[i][j] += (i == j) ? kJ/(S[i]*S[i]) : kJ*J/(S[i]*S[j]); }
}
#endif
#if AF_PTAN
// 절 CM189 --ptan 1: 반환 사상과 일관된 접선 (재질 1·3·5 = PG von Mises 계열). 합성 주값 응력 p_i(σ) = P_i(RM(σ)) RM(σ)_i / σ_i (aq_elem 의 힘과 같은 식,
// 상태 ys·logJp·al 은 걸음 시작값 지역 사본) 를 시험 σ 에서 중심 차분 -> 대칭화. 항복 안 한 점은 탄성 접선과 같음.
__device__ __forceinline__ void aq_cpi(int mat, const float* sg, const float* prm, float ys, float lj, float al, float dt, float* p) {
    float se[3] = {sg[0], sg[1], sg[2]}; float y_ = ys, l_ = lj, a_ = al;
    pl_rm(mat, se, prm, y_, l_, a_, dt);
    float Pe[3]; pl_pi(mat, se, prm, a_, Pe);
    for (int i=0;i<3;i++) p[i] = Pe[i]*se[i]/sg[i];
}
__device__ __noinline__ void aq_tan9c(int mat, float s0, float s1, float s2, const float* prm, float ys, float lj, float al, float dt, float* pe, float* D9) {
    float c[3] = {s0, s1, s2};
    aq_cpi(mat, c, prm, ys, lj, al, dt, pe);
    float Dt[9];
    #pragma unroll
    for (int j=0;j<3;j++) {
        float sj = (j == 0) ? s0 : ((j == 1) ? s1 : s2);
        float h = fmaxf(1e-3f*fabsf(sj), 1e-4f);
        float sp[3] = {s0 + ((j == 0) ? h : 0.f), s1 + ((j == 1) ? h : 0.f), s2 + ((j == 2) ? h : 0.f)};
        float sm[3] = {s0 - ((j == 0) ? h : 0.f), s1 - ((j == 1) ? h : 0.f), s2 - ((j == 2) ? h : 0.f)};
        float pp[3], pm[3]; aq_cpi(mat, sp, prm, ys, lj, al, dt, pp); aq_cpi(mat, sm, prm, ys, lj, al, dt, pm);
        float ih = 0.5f/h;
        Dt[j] = (pp[0] - pm[0])*ih; Dt[3+j] = (pp[1] - pm[1])*ih; Dt[6+j] = (pp[2] - pm[2])*ih;
    }
    D9[0] = Dt[0]; D9[4] = Dt[4]; D9[8] = Dt[8];
    D9[1] = D9[3] = 0.5f*(Dt[1] + Dt[3]); D9[2] = D9[6] = 0.5f*(Dt[2] + Dt[6]); D9[5] = D9[7] = 0.5f*(Dt[5] + Dt[7]);
    for (int k=0;k<9;k++) if (!isfinite(D9[k])) D9[k] = 0.f;
    for (int k=0;k<3;k++) if (!isfinite(pe[k])) pe[k] = 0.f;
}
#endif
__device__ __forceinline__ void aq_tan(int mat, const float* ss0, const float* prm, float al, float* pe, float D[3][3]) {
    float D9[9]; aq_tan9(mat, ss0[0], ss0[1], ss0[2], prm, al, pe, D9);
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = D9[i*3+j];
}
// 재질 요소 (주값 분해 저장, ao_elem_s 와 같은 틀): 응력 = 시험 X 에 정답 RM (상태 지역 사본) -> τ̂ = pl_pi(σ̂) σ̂, P = U diag(τ̂_i/σ_i) Vᵀ;
// 헤시안 = 시험 X 의 탄성 접선 (초기 강성법, 투영). 등가 점성 (eta > 0): 노드 속도 v = α (y − z0) 로 Ẋ, τ_v = 2 η J sym(Ẋ X⁻¹), P += τ_v X⁻ᵀ, 접선에 α η 몫.
__global__ void __launch_bounds__(128) aq_elem(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, int mat, const float* prm, const float* Sq, float dt,
                        double* f, float* H, int capQ, float* B, int* cnt, const float* z0, float alI, float eta, int* qb) {
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    __shared__ float sY[16][27][3]; __shared__ float sQ[128][9]; __shared__ float sU[128][9]; __shared__ float sM[128][9]; __shared__ float sC[128][12];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    bool on = q < IP[12];
    int qq = on ? q : 0;
    int tl = threadIdx.x, l8 = tl & 7, b8 = tl & ~7, cl = tl >> 3, oc = qq & 7;
    int ids[27]; qnodes(qq, IP, cb, ids);
    for (int o = l8; o < 27; o += 8) { int id = ids[o]; for (int a=0;a<3;a++) sY[cl][o][a] = y[id*3+a]; }
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)qq*9+k];
    __syncwarp();
    float A[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; for (int a=0;a<3;a++) { float ya = sY[cl][o][a]; A[a*3] += ya*bg[0]; A[a*3+1] += ya*bg[1]; A[a*3+2] += ya*bg[2]; } }
    float X[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) X[a*3+b] = F[a*3+b] + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b];
    float Ad[9] = {0,0,0,0,0,0,0,0,0};
    if (eta > 0.f) {                                                          // 노드 속도 기울기 (z0 도 같은 칸 공유 메모리로)
        __syncwarp();
        for (int o = l8; o < 27; o += 8) { int id = ids[o]; for (int a=0;a<3;a++) sY[cl][o][a] = alI*(y[id*3+a] - z0[id*3+a]); }
        __syncwarp();
        for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; for (int a=0;a<3;a++) { float va = sY[cl][o][a]; Ad[a*3] += va*bg[0]; Ad[a*3+1] += va*bg[1]; Ad[a*3+2] += va*bg[2]; } }
    }
    float U[9], S[3], V[9];
    ssvdf(X, U, S, V);
    float ys = Sq[(size_t)qq*4], lj = Sq[(size_t)qq*4+1], al = Sq[(size_t)qq*4+2];
    float sev = (mat == 7) ? prm[14] : 0.f;                                      // 뒤집힌 시험 바닥 (--nhb_eps; 0 = 끔): 주값 σ̂ = max(σ, ε) 에서 RM·응력·접선
    float se[3]; for (int i=0;i<3;i++) se[i] = (sev > 0.f) ? fmaxf(S[i], sev) : S[i];
    bool ch = (prm[15] > 0.5f) ? false : pl_rm(mat, se, prm, ys, lj, al, dt);
    if (cnt && on && ch && mat != 7) atomicAdd(&cnt[0], 1);
#ifdef AF_DMG
    al = dmg_g(Sq, (size_t)qq);                                              // 절 CM19: 응력·접선의 al 자리에 열화 g (재질 7 RM 은 al 을 안 씀·안 바꿈)
#endif
    float Pe[3]; pl_pi(mat, se, prm, al, Pe);
    float ss[3], p[3];
    for (int i=0;i<3;i++) { ss[i] = (sev > 0.f) ? fmaxf(S[i], sev) : ((fabsf(S[i]) > 1e-6f) ? S[i] : ((S[i] < 0.f) ? -1e-6f : 1e-6f)); p[i] = Pe[i]*se[i]/ss[i]; }
#ifdef AF_JBAR
    aq_jbar_p(S, prm, p);                                                     // 절 CM13 판 (1): J 로그 장벽 몫 (시험 X 의 J > 0 일 때만 정의)
#endif
    float vq = on ? Vq[qq] : 0.f;
    float Pm[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Pm[a*3+b] = U[a*3]*p[0]*V[b*3] + U[a*3+1]*p[1]*V[b*3+1] + U[a*3+2]*p[2]*V[b*3+2];
    float Jx = ss[0]*ss[1]*ss[2];
    if (eta > 0.f) {                                                          // τ_v X⁻ᵀ, L = Ẋ X⁻¹ = (Ad F) X⁻¹
        float Xd[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Xd[a*3+b] = Ad[a*3]*F[b] + Ad[a*3+1]*F[3+b] + Ad[a*3+2]*F[6+b];
        float Xi[9]; inv3(X, Xi);
        float L[9]; ao_mm3(Xd, Xi, L);
        float T[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) T[a*3+b] = eta*Jx*(L[a*3+b] + L[b*3+a]);
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) Pm[a*3+b] += T[a*3]*Xi[b*3] + T[a*3+1]*Xi[b*3+1] + T[a*3+2]*Xi[b*3+2];   // T X⁻ᵀ
    }
    if (qb && on && vq > 0.f) {                                                // 진단: 뒤집힌 시험 X, 비유한 응력
        if (!(S[0]*S[1]*S[2] > 0.f)) atomicAdd(&qb[0], 1);
        bool nf = false; for (int k=0;k<9;k++) if (!isfinite(Pm[k])) nf = true;
        if (nf) atomicAdd(&qb[1], 1);
    }
    for (int k=0;k<9;k++) Pm[k] *= vq;
    for (int a=0;a<3;a++) for (int c=0;c<3;c++) sQ[tl][a*3+c] = Pm[a*3]*F[c*3] + Pm[a*3+1]*F[c*3+1] + Pm[a*3+2]*F[c*3+2];
    float M[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) M[a*3+b] = F[a*3]*V[b] + F[a*3+1]*V[3+b] + F[a*3+2]*V[6+b];
    float C[12];
    { float al0 = Sq[(size_t)qq*4+2];
#ifdef AF_DMG
      al0 = al;
#endif
      float pe[3], D[3][3], Ev[3][3], Dk[3][3];
#if AF_PTAN
      if ((mat == 1 || mat == 3 || mat == 5) && !(prm[15] > 0.5f)) {
          float D9c[9]; aq_tan9c(mat, ss[0], ss[1], ss[2], prm, Sq[(size_t)qq*4], Sq[(size_t)qq*4+1], al0, dt, pe, D9c);
          for (int i=0;i<3;i++) for (int j=0;j<3;j++) D[i][j] = D9c[i*3+j];
      } else
#endif
      aq_tan(mat, ss, prm, al0, pe, D);
#ifdef AF_JBAR
      aq_jbar_t(S, prm, pe, D);                                               // 절 CM13 판 (1): 장벽 주값 헤시안·응력 (투영은 아래 그대로)
#endif
      if (eta > 0.f) for (int i=0;i<3;i++) D[i][i] += 2.f*alI*eta*fabsf(Jx)/(ss[i]*ss[i]);
      for (int i=0;i<3;i++) for (int j=0;j<3;j++) Dk[i][j] = D[i][j];
      jac3f(D, Ev);
      float dm[3]; for (int m=0;m<3;m++) { float d_ = fmaxf(D[m][m], 0.f); dm[m] = (d_ > 0.f && isfinite(d_)) ? d_ : 0.f; }
      const int RI[6] = {0,1,2,0,0,1}, RJ[6] = {0,1,2,1,2,2};
      for (int k=0;k<6;k++) C[k] = vq*(dm[0]*Ev[RI[k]][0]*Ev[RJ[k]][0] + dm[1]*Ev[RI[k]][1]*Ev[RJ[k]][1] + dm[2]*Ev[RI[k]][2]*Ev[RJ[k]][2]);
      for (int pp=0; pp<3; pp++) { int i, j; ao_pairs(pp, i, j);
          float den = ss[i] + ss[j]; float lt = (fabsf(den) > 1e-12f) ? (pe[i] + pe[j])/den : 0.f;
          float dd = ss[i] - ss[j]; float lf = (fabsf(dd) > 1e-6f*(fabsf(ss[i]) + fabsf(ss[j]))) ? (pe[i] - pe[j])/dd : (Dk[i][i] - Dk[i][j]);
          lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = isfinite(lf) ? fmaxf(lf, 0.f) : 0.f;
          float c1 = 0.5f*(lt + lf), c2 = 0.5f*(lf - lt);
          if (eta > 0.f) { float cv = alI*eta*fabsf(Jx); c1 += cv*0.5f*(1.f/(ss[i]*ss[i]) + 1.f/(ss[j]*ss[j])); c2 += cv/(ss[i]*ss[j]); }
          C[6+pp] = vq*c1; C[9+pp] = vq*c2; }
      if (AQ_DBG && on && qq == AQ_DBGQ) {                                     // 진단 (절 AR): 한 점의 중간값
          float* o = AQ_DBG;
          for (int k=0;k<9;k++) { o[k] = X[k]; o[9+k] = Dk[k/3][k%3]; o[18+k] = Ev[k/3][k%3]; }
          for (int i=0;i<3;i++) { o[27+i] = S[i]; o[30+i] = ss[i]; o[33+i] = pe[i]; o[36+i] = dm[i]; o[39+i] = D[i][i]; o[42+i] = se[i]; }
          o[45] = vq; o[46] = al0; o[47] = (float)mat; for (int k=0;k<12;k++) o[48+k] = C[k]; }
    }
    if (on) { for (int k=0;k<9;k++) { H[(size_t)k*capQ + qq] = U[k]; H[(size_t)(9+k)*capQ + qq] = M[k]; } for (int k=0;k<12;k++) H[(size_t)(18+k)*capQ + qq] = C[k]; }
    for (int k=0;k<9;k++) { sU[tl][k] = U[k]; sM[tl][k] = M[k]; }
    for (int k=0;k<12;k++) sC[tl][k] = C[k];
    __syncwarp();
    for (int o = l8; o < 27; o += 8) {
        double fs[3] = {0.0, 0.0, 0.0};
        float bb[6] = {0,0,0,0,0,0};
        for (int j=0;j<8;j++) {
            const float* bg = BG + (j*27+o)*3;
            const float* Qj = sQ[b8+j];
            for (int a=0;a<3;a++) fs[a] += (double)(Qj[a*3]*bg[0] + Qj[a*3+1]*bg[1] + Qj[a*3+2]*bg[2]);
            const float* Mj = sM[b8+j]; const float* Cj = sC[b8+j]; const float* Uj = sU[b8+j];
            float h[3]; for (int b=0;b<3;b++) h[b] = Mj[b]*bg[0] + Mj[3+b]*bg[1] + Mj[6+b]*bg[2];
            float S6[6] = {Cj[0]*h[0]*h[0] + Cj[6]*h[1]*h[1] + Cj[7]*h[2]*h[2],
                           Cj[1]*h[1]*h[1] + Cj[6]*h[0]*h[0] + Cj[8]*h[2]*h[2],
                           Cj[2]*h[2]*h[2] + Cj[7]*h[0]*h[0] + Cj[8]*h[1]*h[1],
                           (Cj[3] + Cj[9])*h[0]*h[1], (Cj[4] + Cj[10])*h[0]*h[2], (Cj[5] + Cj[11])*h[1]*h[2]};
            float Sm[9] = {S6[0], S6[3], S6[4], S6[3], S6[1], S6[5], S6[4], S6[5], S6[2]};
            float T[9]; for (int a=0;a<3;a++) for (int c=0;c<3;c++) T[a*3+c] = Uj[a*3]*Sm[c] + Uj[a*3+1]*Sm[3+c] + Uj[a*3+2]*Sm[6+c];
            const int RI[6] = {0,1,2,0,0,1}, RJ[6] = {0,1,2,1,2,2};
            for (int k=0;k<6;k++) bb[k] += T[RI[k]*3]*Uj[RJ[k]*3] + T[RI[k]*3+1]*Uj[RJ[k]*3+1] + T[RI[k]*3+2]*Uj[RJ[k]*3+2];
        }
        float mm[9] = {bb[0], bb[3], bb[4], bb[3], bb[1], bb[5], bb[4], bb[5], bb[2]};
        if (on) {
            for (int a=0;a<3;a++) { if (AO_DET) AO_CF[((size_t)(qq>>3)*27+o)*3+a] = fs[a]; else atomicAdd(&f[ids[o]*3+a], fs[a]); }
            for (int k=0;k<9;k++) { if (AO_DET) AO_CB[((size_t)(qq>>3)*27+o)*9+k] = mm[k]; else atomicAdd(&B[ids[o]*9+k], mm[k]); }
        }
    }
}
// 걸음 끝 커밋 (재질): X = F_q + dA F_q, RM (가우스 점 상태 갱신) -> F_q ← F_q X⁻¹ Y
__global__ void aq_commit(const int* IP, const int* cb, const float* BG, float* Fq, const float* U, int mat, const float* prm, float* Sq, float dt) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float A[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; int id = ids[o]; for (int a=0;a<3;a++) { float ua = U[id*3+a]; A[a*3] += ua*bg[0]; A[a*3+1] += ua*bg[1]; A[a*3+2] += ua*bg[2]; } }
    float X[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) X[a*3+b] = F[a*3+b] + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b];
    float Uu[9], s[3], Vv[9]; ssvdf(X, Uu, s, Vv);
    if (mat == 7 && prm[14] > 0.f) for (int i=0;i<3;i++) s[i] = fmaxf(s[i], prm[14]);
    float ys = Sq[(size_t)q*4], lj = Sq[(size_t)q*4+1], al = Sq[(size_t)q*4+2];
    bool ch = (prm[15] > 0.5f) ? false : pl_rm(mat, s, prm, ys, lj, al, dt);
    Sq[(size_t)q*4] = ys; Sq[(size_t)q*4+1] = lj; Sq[(size_t)q*4+2] = al;
#ifdef AF_DMG
    { float ps = dmg_psip(s, prm);                                            // 절 CM19: RM 뒤 탄성 σ 의 ψ⁺ 로 가우스 점 이력 갱신 (비가역 최대)
#if AF_DMG == 1
      Sq[(size_t)q*4+2] = fmaxf(Sq[(size_t)q*4+2], dmg_cdm(ps, prm[10]));
#else
      Sq[(size_t)q*4+2] = fmaxf(Sq[(size_t)q*4+2], ps);
#endif
    }
#endif
    if (!ch) return;
    float Y[9]; pl_usv(Uu, s, Vv, Y);
    float Xi[9]; inv3(X, Xi);
    float T1[9], T2[9]; ao_mm3(Xi, Y, T1); ao_mm3(F, T1, T2);
    bool ok = true; for (int k=0;k<9;k++) if (!isfinite(T2[k])) ok = false;
    if (ok) for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = T2[k];
}
// G2P (재질): 걸음별 F ← RM(A_k A_{k−1}⁻¹ F) (입자 상태 ys·logJp·al 갱신), x·v 는 마지막 걸음
__global__ void aq_g2p(int N, const int* IP, const float* Ai, const int* pb, const float* fx, const float* Ust, int K, int capN, const float* vk, int mat, const float* prm, float dt,
                       float* x, float* v, float* F, float* ysp, float* ljp, float* alp, int* cnt) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float Fp[9]; for (int k=0;k<9;k++) Fp[k] = F[(size_t)p*9+k];
    float Ap[9] = {1,0,0, 0,1,0, 0,0,1};
    float u[3] = {0,0,0}, vv[3] = {0,0,0};
    float ys = ysp[p], lj = ljp[p], al = alp[p], a0 = al;
    int nrm = 0;
    for (int kk=0; kk<K; kk++) {
        const float* Uk = Ust + (size_t)kk*capN*3;
        float A[9] = {1,0,0, 0,1,0, 0,0,1};
        bool last = kk == K-1;
        for (int o=0;o<27;o++) {
            int i = o/9, j = (o/3)%3, k = o%3;
            int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
            float ww = w[0][i]*w[1][j]*w[2][k];
            float gy[3] = {dw[0][i]*w[1][j]*w[2][k], w[0][i]*dw[1][j]*w[2][k], w[0][i]*w[1][j]*dw[2][k]};
            float gx[3]; for (int b=0;b<3;b++) gx[b] = Ai[0*3+b]*gy[0] + Ai[1*3+b]*gy[1] + Ai[2*3+b]*gy[2];
            for (int a=0;a<3;a++) { float uu = Uk[nid*3+a]; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; if (last) { u[a] += ww*uu; vv[a] += ww*vk[nid*3+a]; } }
        }
        float Api[9]; inv3(Ap, Api);
        float Ainc[9], Ft[9]; ao_mm3(A, Api, Ainc); ao_mm3(Ainc, Fp, Ft);
        float Uu[9], s[3], Vv[9]; ssvdf(Ft, Uu, s, Vv);
        if (mat == 7 && prm[14] > 0.f) for (int i=0;i<3;i++) s[i] = fmaxf(s[i], prm[14]);
        if (!(prm[15] > 0.5f) && pl_rm(mat, s, prm, ys, lj, al, dt)) { pl_usv(Uu, s, Vv, Fp); nrm++; } else for (int k=0;k<9;k++) Fp[k] = Ft[k];
#ifdef AF_DMG
        { float ps = dmg_psip(s, prm);                                        // 절 CM19: 입자 이력 (걸음마다, RM 뒤 탄성 σ)
#if AF_DMG == 1
          al = fmaxf(al, dmg_cdm(ps, prm[10]));
#else
          al = fmaxf(al, ps);
#endif
        }
#endif
        for (int k=0;k<9;k++) Ap[k] = A[k];
    }
    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; }
    bool ok = true; for (int k=0;k<9;k++) if (!isfinite(Fp[k])) ok = false;
    if (ok) for (int k=0;k<9;k++) F[(size_t)p*9+k] = Fp[k];
    ysp[p] = ys; ljp[p] = lj; alp[p] = al;
    if (cnt && nrm && mat != 7) atomicAdd(&cnt[2], 1);
    if (cnt && a0 > 0.f && !(al > 0.f)) atomicAdd(&cnt[3], 1);
}
#if AF_DMG == 2
// 절 CM19 위상장 (AT2, Miehe 2010 이력장): 노드 손상장 d (B-스플라인 기저 w_n), 가우스 점 적분 (기준 부피 V_q, 기준 기울기 ∇_X w = F_qᵀ ∇_x w):
//   Σ_q V_q [ (G_c/l0 + 2 H_q) d_q w_n + G_c l0 ∇d_q·∇w_n ] = Σ_q V_q 2 H_q w_n  (대칭 양정치) -> 블록 야코비 (대각) PCG, 고정 반복 (그래프 안).
__global__ void pf_asm(const int* IP, const int* cb, const float* BW, const float* BG, const float* Fq, const float* Vq, const float* Sq, const float* prm, float* rhs, float* dg) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float vq = Vq[q]; if (!(vq > 0.f)) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float Gc = prm[10], l0 = prm[11], H = Sq[(size_t)q*4+2];
    float aa = Gc/l0 + 2.f*H, bb = Gc*l0;
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; float g[3]; gqn(F, BG + (oc*27+o)*3, g);
        atomicAdd(&rhs[ids[o]], vq*2.f*H*w); atomicAdd(&dg[ids[o]], vq*(aa*w*w + bb*(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]))); }
}
__global__ void pf_mv(const int* IP, const int* cb, const float* BW, const float* BG, const float* Fq, const float* Vq, const float* Sq, const float* prm, const float* x, float* out) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float vq = Vq[q]; if (!(vq > 0.f)) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float Gc = prm[10], l0 = prm[11], H = Sq[(size_t)q*4+2];
    float aa = Gc/l0 + 2.f*H, bb = Gc*l0;
    float xv = 0.f, gr[3] = {0.f, 0.f, 0.f};
    for (int o=0;o<27;o++) { float w = BW[oc*27+o], xo = x[ids[o]]; float g[3]; gqn(F, BG + (oc*27+o)*3, g);
        xv += w*xo; gr[0] += g[0]*xo; gr[1] += g[1]*xo; gr[2] += g[2]*xo; }
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; float g[3]; gqn(F, BG + (oc*27+o)*3, g);
        atomicAdd(&out[ids[o]], vq*(aa*xv*w + bb*(gr[0]*g[0] + gr[1]*g[1] + gr[2]*g[2]))); }
}
// r = b − A d, z = r / diag, p = z; SP[0] += r·z, SP[3] += b·b
__global__ void pf_init(const int* IP, const float* rhs, const float* Ad, const float* dg, float* r, float* z, float* p, double* SP) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double rz = 0.0, bb = 0.0;
    if (n < IP[9]) { float rr = rhs[n] - Ad[n]; float zz = (dg[n] > 0.f) ? rr/dg[n] : 0.f; r[n] = rr; z[n] = zz; p[n] = zz; rz = (double)rr*zz; bb = (double)rhs[n]*rhs[n]; }
    blocksum_add(rz, &SP[0]); blocksum_add(bb, &SP[3]);
}
__global__ void pf_dot(const int* IP, const float* p, const float* Ap, double* SP) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double t = (n < IP[9]) ? (double)p[n]*Ap[n] : 0.0;
    blocksum_add(t, &SP[1]);
}
__global__ void pf_up1(const int* IP, const double* SPc, const float* p, const float* Ap, const float* dg, float* d, float* r, float* z, double* SP) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double rz = 0.0, rr2 = 0.0;
    if (n < IP[9]) { double pa = SPc[1]; float al = (pa > 0.0) ? (float)(SPc[0]/pa) : 0.f;
        d[n] += al*p[n]; float rr = r[n] - al*Ap[n]; r[n] = rr; float zz = (dg[n] > 0.f) ? rr/dg[n] : 0.f; z[n] = zz; rz = (double)rr*zz; rr2 = (double)rr*rr; }
    blocksum_add(rz, &SP[2]); blocksum_add(rr2, &SP[4]);
}
__global__ void pf_up2(const int* IP, const double* SP, const float* z, float* p) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    double r0 = SP[0]; float be = (r0 > 0.0) ? (float)(SP[2]/r0) : 0.f;
    p[n] = z[n] + be*p[n];
}
// 반복 끝: rz ← rz_new, 마지막 |r|² 를 SP[5] 에; res[k] = |r| / |b| (풀이 k 의 마지막 반복)
__global__ void pf_roll(double* SP, float* res, int k) {
    SP[0] = SP[2]; SP[2] = 0.0; SP[1] = 0.0; SP[5] = SP[4]; SP[4] = 0.0;
    res[k] = (SP[3] > 0.0) ? (float)sqrt(SP[5]/SP[3]) : 0.f;
}
// 노드 d -> 가우스 점 d_q = clamp(Σ w_n d_n, 0, 1) (Sq[3])
__global__ void pf_q(const int* IP, const int* cb, const float* BW, const float* d, float* Sq) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float s = 0.f; for (int o=0;o<27;o++) s += BW[oc*27+o]*d[ids[o]];
    Sq[(size_t)q*4+3] = fminf(fmaxf(s, 0.f), 1.f);
}
// 노드 d -> 입자 (출력·지표만: ysp = d_p, 재질 7 은 ys 를 안 씀)
__global__ void pf_g2p(int N, const int* IP, const int* pb, const float* fx, const float* d, float* dp) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float s = 0.f;
    for (int o=0;o<27;o++) { int i = o/9, j = (o/3)%3, k = o%3;
        s += w[0][i]*w[1][j]*w[2][k]*d[((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k)]; }
    dp[p] = fminf(fmaxf(s, 0.f), 1.f);
}
#endif
// 0 아닌 구속 속도 (bread enforce_particle_translation): 프레임 시작 구속 입자 v = v_bc, 입자 구속 속도 pvc (풀린 입자 0) 와 구속 표시 pmk
__global__ void aq_vset(int N, const float* Trel, const float* tbuf, const float* vbc, float* v, float* pvc, float* pmk) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    bool c = Trel[p] > tbuf[0];
    for (int a=0;a<3;a++) { float vb = c ? vbc[p*3+a] : 0.f; if (c) v[p*3+a] = vb; pvc[p*3+a] = vb; pmk[p*3+a] = c ? 1.f : 0.f; }
}
__global__ void aq_p2g_v(int N, const int* IP, const int* pb, const float* fx, const float* m, const float* pvc, const float* pmk, float* vm, float* cm) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13] || !(pmk[p*3] > 0.f)) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float mp = m[p];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        float ww = w[0][i]*w[1][j]*w[2][k]*mp;
        for (int a=0;a<3;a++) atomicAdd(&vm[nid*3+a], ww*pvc[p*3+a]);
        atomicAdd(&cm[nid*3], ww);
    }
}
// 노드 구속 목표 속도 = Σ w m v_bc / Σ w m (구속 입자)
__global__ void aq_vbar(const int* IP, const float* cm, float* vm) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float c = cm[n*3];
    for (int a=0;a<3;a++) vm[n*3+a] = (c > 0.f) ? vm[n*3+a]/c : 0.f;
}
#if AF_BLOC
// 절 CM189 --bloc: 장벽 실현 가능 보폭 노드별. 표본점 q 의 d 는 노드 위치에 선형: d(β) = d0 + Σ_n w_n β_n s_n (s_n = thinv (y1 − y0)_n·n̂).
// 관통 쪽 몫만 모은 S⁻ = Σ_n w_n min(s_n, 0) 으로 d0 + 2^-k S⁻ > 0 인 가장 작은 k_q (보수적: 노드 β_n ≤ 2^-k_q 이면 어떤 섞임에서도 d > 0) -> 그 점이 걸친 노드 (w ≠ 0) 에 원자 최대.
__global__ void __launch_bounds__(128) bl_kq(const int* IP, const int* cb, const float* BW, const float* y0, const float* y1, const float* U, float thinv,
                         float n0, float n1, float n2, float pn, int* kn) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3];
    if (!(bm > 0.f) || !AF_BACT[q]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xa[3], sm = 0.f;
    for (int a=0;a<3;a++) xa[a] = AF_BX[q*4+a]/bm;
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
        for (int a=0;a<3;a++) xa[a] += w*(U[id*3+a] + thinv*(y0[id*3+a] - U[id*3+a]));
        float sn = thinv*((y1[id*3]-y0[id*3])*n0 + (y1[id*3+1]-y0[id*3+1])*n1 + (y1[id*3+2]-y0[id*3+2])*n2);
        sm += w*fminf(sn, 0.f); }
    float d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn;
    if (d0 + sm > 0.f) return;
    int k = 0; float c = 1.f;
    while (true) { c *= 0.5f; k++; if (c == 0.f) break; if (d0 + c*sm > 0.f) break; }
    for (int o=0;o<27;o++) if (BW[oc*27+o] != 0.f) atomicMax(&kn[ids[o]], k);
}
__global__ void bl_lsup(const int* IP, const double* SC, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, float* eta, const float* dl, float* y, const int* kn) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float st = fminf((float)SC[14], ldexpf(1.f, -kn[n]));
    float e[3]; for (int a=0;a<3;a++) { e[a] = eta[n*3+a] + st*dl[n*3+a]; eta[n*3+a] = e[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y[n*3+a] = yh[n*3+a] + t[a];
}
#endif

//@@ TOL_CTL 1150

#ifdef AF_ADAPT
__global__ void rt_ctl(int op, int it, double* SC, int* CT, double* CD, double eN2, double eP2) {
    if (op == 0) { for (int i=0;i<16;i++) CT[i] = 0; AF_SKIP = 0; return; }
    if (op == 1) { CT[0] = 0; CT[1] = 0; CT[6]++; AF_SKIP = 0; return; }
    if (op == 2) { CT[1] = 0; AF_SKIP = CT[0]; return; }
    if (op == 3) {
        if (CT[0]) return;
        double g2 = SC[1];
        if (it == 0) { CD[0] = g2; CD[2] = SC[56]; if (g2 == 0.0) { CT[0] = 1; AF_SKIP = 1; return; } }      // 비유한 |g| 는 건너뛰지 않음 (발산을 숨기지 않게)
        else if (eN2 >= 0.0 && g2 <= eN2*CD[2]) { CT[0] = 1; AF_SKIP = 1; return; }
        CT[2]++; return;
    }
    if (op == 4) { if (CT[0]) return; double rf = fmin(SC[1], CD[2]); CD[1] = (eP2 >= 0.0) ? eP2*rf : -1.0; CT[1] = 0; CT[7]++; SC[57] = 0.0; return; }
    if (op == 5) { AF_SKIP = CT[0] | CT[1]; return; }
    if (op == 6) { double r2 = SC[57]; SC[57] = 0.0; if (CT[0] | CT[1]) return; CT[3]++; if (CD[1] >= 0.0 && r2 <= CD[1]) CT[1] = 1; return; }
    if (op == 7) { if (!CT[0] && !CT[1]) CT[5]++; AF_SKIP = CT[0]; return; }
    if (op == 8) { if (!CT[0]) CT[4]++; AF_SKIP = 0; return; }
}
#endif

//@@ AW_SRC 6474

__device__ __forceinline__ double aw_p3(double c0, double c1, double c2, double c3, double a) { return c0 + a*(c1 + a*(c2 + a*c3)); }
__device__ bool aw_pos(double c0, double c1, double c2, double c3, double a) {
    if (!(aw_p3(c0, c1, c2, c3, a) > 0.0)) return false;
    double A = 3.0*c3, B = 2.0*c2, C = c1, t[2]; int nt = 0;                  // 극점 (p'(t) = 0) 이 (0, a) 안에 있으면 그 값도 양수여야
    if (A == 0.0) { if (B != 0.0) t[nt++] = -C/B; }
    else { double D = B*B - 4.0*A*C; if (D >= 0.0) { double q = -0.5*(B + copysign(sqrt(D), B)); if (q != 0.0) { t[nt++] = q/A; t[nt++] = C/q; } else t[nt++] = 0.0; } }
    for (int i=0;i<nt;i++) if (t[i] > 0.0 && t[i] < a && !(aw_p3(c0, c1, c2, c3, t[i]) > 0.0)) return false;
    return true;
}
__device__ __forceinline__ void aw_X(const int* ids, int oc, const float* BG, const float* F, const float* v, int withF, float* X) {
    float A[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; int id = ids[o]; for (int a=0;a<3;a++) { float va = v[id*3+a]; A[a*3] += va*bg[0]; A[a*3+1] += va*bg[1]; A[a*3+2] += va*bg[2]; } }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) X[a*3+b] = (withF ? F[a*3+b] : 0.f) + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b];
}
__device__ __forceinline__ void aw_cof(const double* M, double* C) {
    C[0] = M[4]*M[8]-M[5]*M[7]; C[1] = M[5]*M[6]-M[3]*M[8]; C[2] = M[3]*M[7]-M[4]*M[6];
    C[3] = M[2]*M[7]-M[1]*M[8]; C[4] = M[0]*M[8]-M[2]*M[6]; C[5] = M[1]*M[6]-M[0]*M[7];
    C[6] = M[1]*M[5]-M[2]*M[4]; C[7] = M[2]*M[3]-M[0]*M[5]; C[8] = M[0]*M[4]-M[1]*M[3];
}
__global__ void aw_inv_q(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* yb, const float* d, int* kk, const float* Ue, float e1, float e0) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    if (!(Vq[q] > 0.f)) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float X0f[9], Dxf[9]; aw_X(ids, oc, BG, F, yb, 1, X0f); aw_X(ids, oc, BG, F, d, 0, Dxf);
    double X0[9], Dx[9]; for (int k=0;k<9;k++) { X0[k] = (double)X0f[k]; Dx[k] = (double)Dxf[k]; }
    double C0[9], CD[9]; aw_cof(X0, C0); aw_cof(Dx, CD);
    double c0 = X0[0]*C0[0] + X0[1]*C0[1] + X0[2]*C0[2], c3 = Dx[0]*CD[0] + Dx[1]*CD[1] + Dx[2]*CD[2], c1 = 0.0, c2 = 0.0;
    for (int k=0;k<9;k++) { c1 += C0[k]*Dx[k]; c2 += CD[k]*X0[k]; }
    if (!(c0 > 0.0)) { atomicAdd(&kk[1], 1); return; }
    int k = 0; while (k < 1074 && !aw_pos(c0, c1, c2, c3, ldexp(1.0, -k))) k++;
    if (Ue) {                                                                  // --ls_end: 걸음 끝 변위 U_e = e1 y + e0 U (HHT 1/(1−αf), −αf/(1−αf)) 의 X 도 J > 0
        float A[9] = {0,0,0,0,0,0,0,0,0};
        for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; int id = ids[o]; for (int a=0;a<3;a++) { float va = e1*yb[id*3+a] + e0*Ue[id*3+a]; A[a*3] += va*bg[0]; A[a*3+1] += va*bg[1]; A[a*3+2] += va*bg[2]; } }
        double Xe[9], De[9], Ce[9], Cd[9];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) { Xe[a*3+b] = (double)(F[a*3+b] + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b]); De[a*3+b] = (double)e1*Dx[a*3+b]; }
        aw_cof(Xe, Ce); aw_cof(De, Cd);
        double f0 = Xe[0]*Ce[0] + Xe[1]*Ce[1] + Xe[2]*Ce[2], f3 = De[0]*Cd[0] + De[1]*Cd[1] + De[2]*Cd[2], f1 = 0.0, f2 = 0.0;
        for (int j=0;j<9;j++) { f1 += Ce[j]*De[j]; f2 += Cd[j]*Xe[j]; }
        if (!(f0 > 0.0)) atomicAdd(&kk[1], 1);
        else { int k2 = 0; while (k2 < 1074 && !aw_pos(f0, f1, f2, f3, ldexp(1.0, -k2))) k2++; k = max(k, k2); }
    }
    if (k > 0) atomicMax(&kk[0], k);
}
// 보폭 2^-k 를 out[0] 에, 통계 st[0] 줄어든 횟수, st[1] 최대 k, st[2] 시작에서 뒤집힌 점-회, st[3] 호출 수; kk 비움
__global__ void aw_pick(int* kk, double* out, int* st) {
    int k = kk[0]; out[0] = ldexp(1.0, -k);
    st[0] += (k > 0) ? 1 : 0; st[1] = max(st[1], k); st[2] += kk[1]; st[3] += 1;
    kk[0] = 0; kk[1] = 0;
}
// 비압축 판 방향 T dl (모든 노드)
__global__ void aw_tdir(const int* IP, const float* sk, const int* pm, float n0, float n1, float n2, const float* v, float* o) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, v + n*3, t); for (int a=0;a<3;a++) o[n*3+a] = t[a];
}
// 시작점 (--y0z): 기준 y_z = ŷ + T η_z, η_z = T 의 역으로 z0 (영속도 위치) 에 맞춘 값 (k = 0 방향은 ŷ 그대로); 방향 dy = y_시작 − y_z
__global__ void aw_y0z_a(const int* IP, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* z0, const float* y, float* ez, float* yb, float* dy, const float* mn, float mthr, int yl) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float s = sk[n]; int p = pm[n];
    if (p == 2 && yl && mn[n] < mthr) { for (int a=0;a<3;a++) { ez[n*3+a] = 0.f; yb[n*3+a] = z0[n*3+a]; dy[n*3+a] = y[n*3+a] - z0[n*3+a]; } return; }   // --y0l: 거의 빈 노드도 기준에서 시작점으로
    if (p == 2) { for (int a=0;a<3;a++) { ez[n*3+a] = 0.f; yb[n*3+a] = y[n*3+a]; dy[n*3+a] = 0.f; } return; }
    float isk = (s > 0.f) ? 1.f/s : 0.f;
    float dd[3]; for (int a=0;a<3;a++) dd[a] = z0[n*3+a] - yh[n*3+a];
    float nv[3] = {n0, n1, n2}, et[3];
    float dn = dd[0]*n0 + dd[1]*n1 + dd[2]*n2;
    if (p == 0) { for (int a=0;a<3;a++) et[a] = isk*dd[a]; }
    else if (p == 3) { for (int a=0;a<3;a++) et[a] = (dd[a] - dn*nv[a]) + isk*dn*nv[a]; }
    else { for (int a=0;a<3;a++) et[a] = dn*nv[a] + isk*(dd[a] - dn*nv[a]); }
    float t[3]; Tapply(s, p, n0, n1, n2, et, t);
    for (int a=0;a<3;a++) { ez[n*3+a] = et[a]; float b = yh[n*3+a] + t[a]; yb[n*3+a] = b; dy[n*3+a] = y[n*3+a] - b; }
}
// 보폭 β = 1 이면 손대지 않음 (비트 그대로), 아니면 η = η_z + β (η − η_z), y = ŷ + T η
__global__ void aw_y0z_b(const int* IP, const double* bt, const float* sk, const int* pm, float n0, float n1, float n2, float* yh, const float* ez, float* eta, float* y, const float* mn, float mthr, int yl, const float* yb, const float* dy) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    double b = bt[0];
    if (b == 1.0) return;
    if (pm[n] == 2) { if (yl && mn[n] < mthr) for (int a=0;a<3;a++) { float v = yb[n*3+a] + (float)b*dy[n*3+a]; yh[n*3+a] = v; y[n*3+a] = v; } return; }   // --y0l: 고정 위치 = 시작점
    float bf = (float)b, e[3];
    for (int a=0;a<3;a++) { e[a] = ez[n*3+a] + bf*(eta[n*3+a] - ez[n*3+a]); eta[n*3+a] = e[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y[n*3+a] = yh[n*3+a] + t[a];
}

//@@ AW_DET9 154
__device__ __forceinline__ float aw_det9(const float* M) { return M[0]*(M[4]*M[8]-M[5]*M[7]) - M[1]*(M[3]*M[8]-M[5]*M[6]) + M[2]*(M[3]*M[7]-M[4]*M[6]); }

//@@ WB_SRC 2564

// ---- 절 CM21 장벽 바닥 (옛 재질 실행기): 표본점 = 팔분 칸 입자 질량 중심 (프레임 시작; 새 실행기 rt_p2g 의 AF_BAR 누적과 같은 식, 여기서는 따로 한 커널) ----
__global__ void wb_p2q(int N, const int* IP, const float* x, const float* m, const float* fx, const int* pcd, const int* cidx) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
    int q = c*8 + oc;
    for (int a=0;a<3;a++) atomicAdd(&AF_BX[q*4+a], m[p]*x[p*3+a]); atomicAdd(&AF_BX[q*4+3], m[p]);
}
// 띠 고착 (절 AN.21 --fl_bst 1): 걸음 시작 관성 예측 y 의 걸음 끝 위치 식에서 표본점 높이 d < d̂ (평면 아래 포함) 이면 고착 몫 1, 밖 0 -> 노드 몫 an (새 실행기 rt_floor_q AF_BAR·AF_BST 1 갈래와 같은 식)
__global__ void wb_floor_q(const int* IP, const int* cb, const float* BW, const float* mq, const float* y, const float* U, float thinv, float n0, float n1, float n2, float pn, float dh, float* an) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3]; if (!(bm > 0.f)) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xq[3] = {AF_BX[q*4]/bm, AF_BX[q*4+1]/bm, AF_BX[q*4+2]/bm};
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
        for (int a=0;a<3;a++) { float ye = U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a]); xq[a] += w*ye; } }
    float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;
    if (!(d < dh)) return;
    float mw = mq[q];
    if (mw > 0.f) for (int o=0;o<27;o++) atomicAdd(&an[ids[o]], BW[oc*27+o]*mw);
}
// 보폭 = min(뒤집힘 필터 보폭 SC[14] (aw_pick), 장벽 실현 가능 보폭 2^-kk) -- 재질 판 장벽 (절 CM2 rt_kstep) 을 --ls_inv 와 함께
__global__ void wb_kmin(const int* kk, double* SC, int has) { double b = (double)ldexpf(1.f, -kk[0]); SC[14] = has ? fmin(SC[14], b) : b; }
// 진단 (--bar_cnt, 결과 불변): 장벽 기울기와 같은 y 에서 활성이고 d < d̂ 인 표본점 표시 (절 CM10 AF_BCNT 비트 1 과 같은 판정)
__global__ void wb_cnt(const int* IP, const int* cb, const float* BW, const float* y, const float* U, float thinv, float n0, float n1, float n2, float pn, float dh, int* bcf) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3]; if (!(bm > 0.f) || !AF_BACT[q]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xq[3] = {AF_BX[q*4]/bm, AF_BX[q*4+1]/bm, AF_BX[q*4+2]/bm};
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
        for (int a=0;a<3;a++) xq[a] += w*(U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a])); }
    float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;
    if (d < dh) bcf[q] |= 1;
}

//@@ LJI_SRC 840

// ---- 절 CM21 --lj_interp: 입자 logJp = 프레임 끝 (마지막 걸음 커밋 RM 뒤) 가우스 점 logJp 의 칸 안 삼선형 보간 (절 CM20 fpi_p 와 같은 가중: t_a = clamp((fx_a − (1 − g)) / 2g, 0, 1), 외삽 없음) ----
__global__ void lji_p(int N, const int* IP, const float* fx, const int* pcd, const int* cidx, const float* Sq, float* ljp) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    if (c < 0) return;
    const float GG = 0.28867513459481287f;
    float t[3]; for (int a=0;a<3;a++) t[a] = fminf(fmaxf((fx[p*3+a] - (1.f - GG)) / (2.f*GG), 0.f), 1.f);
    float s = 0.f;
    for (int oc=0; oc<8; oc++) {
        float w = ((oc & 4) ? t[0] : 1.f - t[0]) * ((oc & 2) ? t[1] : 1.f - t[1]) * ((oc & 1) ? t[2] : 1.f - t[2]);
        if (!(w > 0.f)) continue;
        s += w*Sq[(size_t)(c*8 + oc)*4+1];
    }
    ljp[p] = s;
}

//@@ UF_SRC 620

__global__ void uf_init(float* ufs, int k) { ufs[k] = 1.f; }
__global__ void uf_min(const double* SC, float* ufs, int k) { float s = (float)SC[14]; if (!(s == s)) s = 0.f; ufs[k] = fminf(ufs[k], s); }
__global__ void uf_g(const double* SC, const double* CD, float* ufg, int k) { double r = (CD[2] > 0.0) ? SC[1]/CD[2] : 0.0; ufg[k] = (r == r) ? (float)fmin(r, 3.0e38) : 3.0e38f; }
__global__ void uf_end(const int* CT, const float* ufs, int* ufl, int k, int adapt, const float* ufg, int res) {
    bool cut = ufs[k] < 1.f, big = res && ufg[k] > 1.f;
    ufl[k] = ((adapt ? (CT[0] == 0) : 1) && (cut || big)) ? 1 : 0; }

//@@ MT_SRC 4980

// ---- 절 CM2 재질 판 연결 커널 ----
// rt_floor_n (이 가지) + 재질 판 몫 (절 AO ao_floor_n 과 같은 식): 구속 인자 k_구속 < 1 이면 k *= k_구속, 세 성분 모두 (p = 0), 목표 속도 vbt;
// 질량 < 문턱 노드는 관성 예측 Ũ 에 고정 (pm 2).
__global__ void mt_floor_n(const int* IP, const float* an, const float* mqn, float kex, int floor_, int vout, float al, float n0, float n1, float n2,
                           const unsigned char* fixd, const float* vbc, const float* U, float tau, const float* Ut, const float* z0, float* y, float* sk, int* pm, float* yh, float* eta,
                           const float* kb, const float* mn, float mthr, const float* vbt) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float k = 1.f; int p = 0;
#ifdef AF_FSTK
    float kfn_ = 1.f;
#endif
    if (floor_) {
        float r = (mqn[n] > 0.f) ? fminf(an[n]/mqn[n], 1.f) : 0.f;
        k = expf(kex*log1pf(-r));
#ifdef AF_FLJ
        k = 1.f - r;
#endif
        float vv = al*((y[n*3]-z0[n*3])*n0 + (y[n*3+1]-z0[n*3+1])*n1 + (y[n*3+2]-z0[n*3+2])*n2);
        p = (vout && vv > 0.f) ? 1 : 0;
#ifdef AF_BAR
        p = 1;
#endif
#ifdef AF_FSTK
        float ra_ = (mqn[n] > 0.f) ? fminf(AF_ANA[n]/mqn[n], 1.f) : 0.f;       // 절 CM12: (A) 가우스 점 고착 (rt_floor_n 과 같은 식)
        float ka_ = expf(kex*log1pf(-ra_));
        kfn_ = (vout && vv > 0.f) ? 1.f : ka_; k *= ka_;
#endif
    }
    float vt[3] = {0.f, 0.f, 0.f};
#ifdef AF_FSTK
    if (kb) { float b = kb[n]; if (b < 1.f) { kfn_ = 1.f; } }                  // 구속 노드는 세 성분 모두 k (p 0 경로) -- 아래 원래 줄
#endif
    if (kb) { float b = kb[n]; if (b < 1.f) { k *= b; p = 0; if (vbt) for (int a=0;a<3;a++) vt[a] = vbt[n*3+a]/al; } }
    if (fixd && fixd[n]) p = 2;
    if (p == 2) { for (int a=0;a<3;a++) { float yy = U[n*3+a] + tau*vbc[n*3+a]; yh[n*3+a] = yy; eta[n*3+a] = 0.f; y[n*3+a] = yy; } sk[n] = 0.f; pm[n] = 2; return; }
    if (mn[n] < mthr) { for (int a=0;a<3;a++) { float yy = Ut[n*3+a]; yh[n*3+a] = yy; eta[n*3+a] = 0.f; y[n*3+a] = yy; } sk[n] = 0.f; pm[n] = 2; return; }
#ifdef AF_FSTK
    if (kfn_ != 1.f) { float dd_[3]; for (int a=0;a<3;a++) dd_[a] = z0[n*3+a] + vt[a] - Ut[n*3+a]; fstk_node(n, k, kfn_, n0, n1, n2, dd_, Ut, y, sk, pm, yh, eta); return; }
#endif
    float s = sqrtf(k);
    float d[3], Pd[3];
    for (int a=0;a<3;a++) d[a] = z0[n*3+a] + vt[a] - Ut[n*3+a];
    if (p == 0) { for (int a=0;a<3;a++) Pd[a] = d[a]; }
    else { float dn = d[0]*n0 + d[1]*n1 + d[2]*n2; Pd[0] = d[0]-dn*n0; Pd[1] = d[1]-dn*n1; Pd[2] = d[2]-dn*n2; }
    float h[3]; for (int a=0;a<3;a++) h[a] = Ut[n*3+a] + (1.f-k)*Pd[a];
    float e1[3], e2[3];
    for (int a=0;a<3;a++) { e1[a] = y[n*3+a] - Ut[n*3+a]; e2[a] = y[n*3+a] - h[a]; }
    float et[3];
    if (p == 0) { for (int a=0;a<3;a++) et[a] = s*e2[a]; }
    else { float c1 = e1[0]*n0 + e1[1]*n1 + e1[2]*n2; float c2 = e2[0]*n0 + e2[1]*n1 + e2[2]*n2;
           for (int a=0;a<3;a++) { float nn_ = (a==0) ? n0 : ((a==1) ? n1 : n2); et[a] = c1*nn_ + s*(e2[a] - c2*nn_); } }
    float ty[3]; Tapply(s, p, n0, n1, n2, et, ty);
    for (int a=0;a<3;a++) { yh[n*3+a] = h[a]; eta[n*3+a] = et[a]; y[n*3+a] = h[a] + ty[a]; }
    sk[n] = s; pm[n] = p;
}
// 구속 노드 (k_구속 < 1) 걸음 끝 가속도 0 (절 AO 처치 (b) 의 구속 몫; 바닥 몫은 rt_ga_end)
__global__ void mt_akb(const int* IP, const float* kb, float* ak) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (kb[n] < 1.f) { ak[n*3] = 0.f; ak[n*3+1] = 0.f; ak[n*3+2] = 0.f; }
}
// 주값 분해 30 성분 (aq_elem: U 9, M = F V 9, C 12 -- V_q 곱함) -> 81 성분 dP/dX (rt_mv 꼴: dP = V_q H dX, dX = A F)
__global__ void mt_h81(const int* IP, const float* Fq, const float* Vq, const float* H30, float* H, int capQ) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float U[9], M[9], C[12], F[9], Fi[9], V[9];
    for (int k=0;k<9;k++) { U[k] = H30[(size_t)k*capQ + q]; M[k] = H30[(size_t)(9+k)*capQ + q]; F[k] = Fq[(size_t)q*9+k]; }
    for (int k=0;k<12;k++) C[k] = H30[(size_t)(18+k)*capQ + q];
    inv3(F, Fi);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) V[a*3+b] = Fi[a*3]*M[b] + Fi[a*3+1]*M[3+b] + Fi[a*3+2]*M[6+b];
    float vq = Vq[q]; float iv = (vq > 0.f) ? 1.f/vq : 0.f;
    for (int c=0;c<9;c++) {
        int i0 = c/3, j0 = c%3;
        float Xh[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Xh[a*3+b] = U[i0*3+a]*V[j0*3+b];
        float Ph[9];
        Ph[0] = C[0]*Xh[0] + C[3]*Xh[4] + C[4]*Xh[8];
        Ph[4] = C[3]*Xh[0] + C[1]*Xh[4] + C[5]*Xh[8];
        Ph[8] = C[4]*Xh[0] + C[5]*Xh[4] + C[2]*Xh[8];
        Ph[1] = C[6]*Xh[1] + C[9]*Xh[3];  Ph[3] = C[9]*Xh[1] + C[6]*Xh[3];
        Ph[2] = C[7]*Xh[2] + C[10]*Xh[6]; Ph[6] = C[10]*Xh[2] + C[7]*Xh[6];
        Ph[5] = C[8]*Xh[5] + C[11]*Xh[7]; Ph[7] = C[11]*Xh[5] + C[8]*Xh[7];
        float W[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) W[a*3+b] = U[a*3]*Ph[b] + U[a*3+1]*Ph[3+b] + U[a*3+2]*Ph[6+b];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) H[(size_t)((a*3+b)*9 + c)*capQ + q] = iv*(W[a*3]*V[b*3] + W[a*3+1]*V[b*3+1] + W[a*3+2]*V[b*3+2]);
    }
}

//@@ MT_PHI_SRC 641

__global__ void mt_pbc_kphi(const int* IP, const float* mn, const float* acc, int K, int S, int k, float expo, float* kb) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n], lk = 0.f; int KS = K*S;
    if (m > 0.f) {
        float suf = 0.f;
        for (int j=KS-1; j>=k*S; j--) { suf += acc[(size_t)n*KS + j]; if (j < (k+1)*S) { float al = fminf(suf/m, 1.f); lk += log1pf(-al); } }
    }
    float x = -lk*expo, r;
    if (!(x > 0.f)) r = 1.f;
    else if (x < 1e-3f) r = 1.f - x/3.f + x*x/12.f;
    else if (isinf(x)) r = 0.f;
    else r = 2.f*(x - 1.f + expf(-x))/(x*x);
    kb[n] = r;
}

//@@ vcap_SRC 1214

extern "C" {
__global__ void vc_cap(const int* IP, const unsigned char* fixd, float vmax, float dt, float* U, float* vk, float* ak, int* cnt) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || (fixd && fixd[n])) return;
    float v0 = vk[n*3], v1 = vk[n*3+1], v2 = vk[n*3+2];
    float s2 = v0*v0 + v1*v1 + v2*v2;
    if (!(s2 > vmax*vmax)) return;
    float r = vmax*rsqrtf(s2);
    for (int e=0;e<3;e++) { float v = vk[n*3+e]; float vc = v*r; U[n*3+e] += dt*(vc - v); vk[n*3+e] = vc; ak[n*3+e] = 0.f; }
    if (cnt) atomicAdd(cnt, 1);
}
// 절 CM196 --be_sub_mmed a: K > 1 걸음 관성 질량 하한 (질량 단위, 걸음 길이 무관) meff = max(meff, a x 첫 프레임 노드 질량 중앙값) (질량 있는 노드만)
// cs > 0 이면 하한 = min(mm, cs tr(B)/3) (cs = c τ²: 걸음이 못 푸는 진동수 (ωτ > 1/√c) 노드만, 최대 mm)
__global__ void vc_mmed(const int* IP, const float* mn, float mm, const float* B, float cs, float* meff) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float fl = mm;
    if (cs > 0.f) fl = fminf(mm, cs*(B[n*9] + B[n*9+4] + B[n*9+8])*(1.f/3.f));
    if (mn[n] > 0.f && meff[n] < fl) meff[n] = fl;
}
// 절 CM196 --be_sub_skc: 장치 전역 float 하나 쓰기 (그래프 안에서 K 별 값)
__global__ void vc_setf(float* p, float v) { *p = v; }
}

