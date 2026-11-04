"""절 AR 진단: 재질 커널 접선 aq_tan (주값 중심 차분) 과 해석 Hencky 접선을 같은 σ 에서 GPU 로 직접 비교 (모듈 _src 그대로 + 시험 커널).
사용: python exe/fem_ar_tan_test.py   (mat 2 PG 모래 탄성 = Hencky, mat 5 FCR; μ λ 는 수박 값)"""
import sys

import cupy as cp
import numpy as np

sys.path.insert(0, "lib")
from anchorflow import fem_ao_rt as AORT

TEST = r'''
extern "C" __global__ void ar_tan_test(int n, const float* S, int mat, const float* prm, float* out) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= n) return;
    float ss[3] = {S[t*3], S[t*3+1], S[t*3+2]};
    float pe[3], D[3][3];
    aq_tan(mat, ss, prm, 1.f, pe, D);
    float* o = out + t*24;
    for (int i=0;i<3;i++) { o[i] = pe[i]; for (int j=0;j<3;j++) o[3+i*3+j] = D[i][j]; }
    float mu = prm[0], lam = prm[1];
    float ep[3]; for (int i=0;i<3;i++) ep[i] = logf(fabsf(ss[i]));
    float tr = ep[0] + ep[1] + ep[2];
    for (int i=0;i<3;i++) { float pa = (2.f*mu*ep[i] + lam*tr)/ss[i]; o[12+i] = pa;
        for (int j=0;j<3;j++) o[15+i*3+j] = (((i==j) ? 2.f*mu : 0.f) + lam)/(ss[i]*ss[j]) - ((i==j) ? pa/ss[i] : 0.f); }
}
'''
_t = open("lib/anchorflow/fem_ao_rt.py", encoding="utf-8").read()
def cblock(name):
    i = _t.index("__global__ void __launch_bounds__(128) " + name + "(")
    a = _t.index("    float C[12];", i); b = _t.index("    if (on) {", a)
    return _t[a:b]
TEST += r'''
extern "C" __global__ void ar_c_test(int n, const float* S, int mat, const float* prm, const float* Sq, float* out) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= n) return;
    float mu = prm[0], lam = prm[1], vq = 1.f, eta = 0.f, alI = 0.f;
    int qq = t;
    float ss[3] = {S[t*3], S[t*3+1], S[t*3+2]};
    float Jx = ss[0]*ss[1]*ss[2];
    float* o = out + t*24;
    { float pe[3]; for (int i=0;i<3;i++) pe[i] = (2.f*mu*logf(fabsf(ss[i])) + lam*(logf(fabsf(ss[0]))+logf(fabsf(ss[1]))+logf(fabsf(ss[2]))))/ss[i];
@SAND@
      for (int k=0;k<12;k++) o[12+k] = C[k]; }
    {
@AQ@
      for (int k=0;k<12;k++) o[k] = C[k]; }
}
'''.replace("@SAND@", cblock("ao_elem_s")).replace("@AQ@", cblock("aq_elem"))
src = AORT._src(tol=False)
i = src.rfind("}")
mod = cp.RawModule(code=src[:i] + "}\n" + TEST, options=("--std=c++14",))
fn = mod.get_function("ar_tan_test")
mu, lam = 2000 / 2.76, 2000 * 0.38 / (1.38 * 0.24)
prm = cp.zeros(16, dtype=cp.float32); prm[0] = mu; prm[1] = lam
S = cp.asarray(np.array([[1, 1, 1], [1.001, 0.9995, 1.0002], [1.05, 0.97, 0.99], [1.2, 1.0, 0.8]], dtype=np.float32))
n = S.shape[0]
for mat in (2, 5):
    out = cp.zeros(n * 24, dtype=cp.float32)
    fn((1,), (32,), (np.int32(n), S, np.int32(mat), prm, out))
    o = out.get().reshape(n, 24)
    for t in range(n):
        print(f"mat {mat} σ {S[t].get().tolist()}")
        print("  pe(aq)", np.round(o[t, :3], 4).tolist(), " pe(해석 Hencky)", np.round(o[t, 12:15], 4).tolist())
        print("  D(aq)\n", np.round(o[t, 3:12].reshape(3, 3), 2))
        if mat == 2:
            print("  D(해석)\n", np.round(o[t, 15:24].reshape(3, 3), 2))

fc = mod.get_function("ar_c_test")
Sq = cp.zeros(n * 4, dtype=cp.float32); Sq[2::4] = 1.0
out = cp.zeros(n * 24, dtype=cp.float32)
fc((1,), (32,), (np.int32(n), S, np.int32(2), prm, Sq, out))
o = out.get().reshape(n, 24)
for t in range(n):
    print(f"계수 C (D+ 위 삼각 6, c1 3, c2 3) σ {S[t].get().tolist()}")
    print("  aq_elem  ", np.round(o[t, :12], 2).tolist())
    print("  ao_elem_s", np.round(o[t, 12:], 2).tolist())
