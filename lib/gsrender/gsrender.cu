//@@ _KSRC 5956

__device__ __forceinline__ void mm3(const float* A, const float* B, float* O) {
  #pragma unroll
  for (int r = 0; r < 3; r++)
    #pragma unroll
    for (int c = 0; c < 3; c++) O[r*3+c] = A[r*3+0]*B[0*3+c] + A[r*3+1]*B[1*3+c] + A[r*3+2]*B[2*3+c];
}
extern "C" __global__ void rr_pg(const float* __restrict__ x, const float* __restrict__ F, const long long* __restrict__ gi,
    const float* __restrict__ C0R, const float* __restrict__ M, int rot, float* __restrict__ P, float* __restrict__ cc, int NG) {
  // M: RT[9], MEAN[3], 1/SO
  int n = blockIdx.x * blockDim.x + threadIdx.x; if (n >= NG) return;
  long long j = gi[n];
  float f[9], c0[9], T[9], Cv[9];
  #pragma unroll
  for (int k = 0; k < 9; k++) { f[k] = F[j*9+k]; c0[k] = C0R[(long long)n*9+k]; }
  mm3(f, c0, T);
  #pragma unroll
  for (int r = 0; r < 3; r++)
    #pragma unroll
    for (int c = 0; c < 3; c++) Cv[r*3+c] = T[r*3+0]*f[c*3+0] + T[r*3+1]*f[c*3+1] + T[r*3+2]*f[c*3+2];
  float p[3];
  #pragma unroll
  for (int d = 0; d < 3; d++) p[d] = (x[j*3+d] - 1.0f) * M[12] + M[9+d];
  if (rot) {
    float Rt[9], Q[9], R2[9]; for (int k = 0; k < 9; k++) Rt[k] = M[k];
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) Q[r*3+c] = Rt[0*3+r]*Cv[0*3+c] + Rt[1*3+r]*Cv[1*3+c] + Rt[2*3+r]*Cv[2*3+c];
    mm3(Q, Rt, R2); for (int k = 0; k < 9; k++) Cv[k] = R2[k];
    float q[3]; for (int c = 0; c < 3; c++) q[c] = p[0]*Rt[0*3+c] + p[1]*Rt[1*3+c] + p[2]*Rt[2*3+c];
    p[0] = q[0]; p[1] = q[1]; p[2] = q[2];
  }
  P[n*3+0] = p[0]; P[n*3+1] = p[1]; P[n*3+2] = p[2];
  cc[n*6+0] = Cv[0]; cc[n*6+1] = Cv[1]; cc[n*6+2] = Cv[2]; cc[n*6+3] = Cv[4]; cc[n*6+4] = Cv[5]; cc[n*6+5] = Cv[8];
}
template <typename T> __device__ void polar_n(const float* f, float* Rr, int maxit, T tol) {
  T X[9]; for (int k = 0; k < 9; k++) X[k] = (T)f[k];
  for (int it = 0; it < maxit; it++) {
    T a = X[0], b = X[1], c = X[2], d = X[3], e = X[4], ff = X[5], g = X[6], h = X[7], i = X[8];
    T A_ = e*i - ff*h, B_ = -(d*i - ff*g), C_ = d*h - e*g;
    T det = a*A_ + b*B_ + c*C_;
    T adj[9] = {A_, -(b*i - c*h), b*ff - c*e, B_, a*i - c*g, -(a*ff - c*d), C_, -(a*h - b*g), a*e - b*d};
    T nx = 0, ni = 0, Xi[9];
    for (int k = 0; k < 9; k++) { Xi[k] = adj[k] / det; nx += X[k]*X[k]; ni += Xi[k]*Xi[k]; }
    T gm = sqrt(sqrt(ni) / sqrt(nx));
    T Y[9], dd = 0;
    for (int r = 0; r < 3; r++) for (int q = 0; q < 3; q++) Y[r*3+q] = (T)0.5 * (gm * X[r*3+q] + Xi[q*3+r] / gm);
    for (int k = 0; k < 9; k++) { dd += (Y[k] - X[k]) * (Y[k] - X[k]); X[k] = Y[k]; }
    if (tol > 0 && dd < tol * tol) break;
  }
  for (int k = 0; k < 9; k++) Rr[k] = (float)X[k];
}
extern "C" __global__ void rr_gf(const float* __restrict__ x, const float* __restrict__ F, const long long* __restrict__ gi,
    const float* __restrict__ C0, const float* __restrict__ Rm, const float* __restrict__ A, const float* __restrict__ b0, const float* __restrict__ L6,
    const float* __restrict__ shs, const float* __restrict__ cp, float* __restrict__ P, float* __restrict__ cc, float* __restrict__ col, int NG, int prec) {
  int n = blockIdx.x * blockDim.x + threadIdx.x; if (n >= NG) return;
  long long j = gi[n];
  float f0[9], f[9], c0[9], T[9], Cv[9];
  #pragma unroll
  for (int k = 0; k < 9; k++) { f0[k] = F[j*9+k]; c0[k] = C0[(long long)n*9+k]; }
  mm3(f0, Rm, f);                                                    // lb_roll_to_h5: F · R
  float xx = x[j*3+0], xy = x[j*3+1], xz = x[j*3+2];
  bool ok = isfinite(xx) && isfinite(xy) && isfinite(xz);
  #pragma unroll
  for (int k = 0; k < 9; k++) ok = ok && isfinite(f[k]);
  if (!ok) {                                                         // wm_render 는 빼고 그림 -> 카메라 위치에 두어 잘림 (같은 화면)
    P[n*3+0] = cp[0]; P[n*3+1] = cp[1]; P[n*3+2] = cp[2];
    for (int k = 0; k < 6; k++) cc[n*6+k] = (k == 0 || k == 3 || k == 5) ? 1e-8f : 0.f;
    col[n*3+0] = col[n*3+1] = col[n*3+2] = 0.f; return;
  }
  mm3(f, c0, T);
  #pragma unroll
  for (int r = 0; r < 3; r++)
    #pragma unroll
    for (int c = 0; c < 3; c++) Cv[r*3+c] = T[r*3+0]*f[c*3+0] + T[r*3+1]*f[c*3+1] + T[r*3+2]*f[c*3+2];
  float c6[6] = {Cv[0], Cv[1], Cv[2], Cv[4], Cv[5], Cv[8]};
  #pragma unroll
  for (int r = 0; r < 6; r++) { float s = 0.f; for (int k = 0; k < 6; k++) s += L6[r*6+k] * c6[k]; cc[n*6+r] = s; }
  float p[3];
  #pragma unroll
  for (int r = 0; r < 3; r++) p[r] = A[r*3+0]*xx + A[r*3+1]*xy + A[r*3+2]*xz + b0[r];
  P[n*3+0] = p[0]; P[n*3+1] = p[1]; P[n*3+2] = p[2];
  // 극분해 (배율 뉴턴): prec 0 = f64 10 회 고정 (polar_t 와 같은 식), 1 = f32 바뀜 ‖ΔX‖_F < 1e-6 이면 멈춤 (최대 12 회)
  float Rr[9];
  if (prec == 0) polar_n<double>(f, Rr, 10, 0.0); else polar_n<float>(f, Rr, 12, 1e-6f);
  float dx = p[0] - cp[0], dy = p[1] - cp[1], dz = p[2] - cp[2];
  float ux = Rr[0]*dx + Rr[1]*dy + Rr[2]*dz, uy = Rr[3]*dx + Rr[4]*dy + Rr[5]*dz, uz = Rr[6]*dx + Rr[7]*dy + Rr[8]*dz;
  float nr = sqrtf(ux*ux + uy*uy + uz*uz); ux /= nr; uy /= nr; uz /= nr;
  const float* s = shs + (long long)n * 48;                          // (16, 3)
  float xx2 = ux*ux, yy2 = uy*uy, zz2 = uz*uz, xyv = ux*uy, yzv = uy*uz, xzv = ux*uz;
  float bs[16];
  bs[0] = 0.28209479177387814f;
  bs[1] = -0.4886025119029199f * uy; bs[2] = 0.4886025119029199f * uz; bs[3] = -0.4886025119029199f * ux;
  bs[4] = 1.0925484305920792f * xyv; bs[5] = -1.0925484305920792f * yzv; bs[6] = 0.31539156525252005f * (2.f*zz2 - xx2 - yy2);
  bs[7] = -1.0925484305920792f * xzv; bs[8] = 0.5462742152960396f * (xx2 - yy2);
  bs[9] = -0.5900435899266435f * uy * (3.f*xx2 - yy2); bs[10] = 2.890611442640554f * xyv * uz; bs[11] = -0.4570457994644658f * uy * (4.f*zz2 - xx2 - yy2);
  bs[12] = 0.3731763325901154f * uz * (2.f*zz2 - 3.f*xx2 - 3.f*yy2); bs[13] = -0.4570457994644658f * ux * (4.f*zz2 - xx2 - yy2);
  bs[14] = 1.445305721320277f * uz * (xx2 - yy2); bs[15] = -0.5900435899266435f * ux * (xx2 - 3.f*yy2);
  for (int ch = 0; ch < 3; ch++) { float v = 0.f; for (int k = 0; k < 16; k++) v += bs[k] * s[k*3+ch]; v += 0.5f; col[n*3+ch] = v > 0.f ? v : 0.f; }
}

