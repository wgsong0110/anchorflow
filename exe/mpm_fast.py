"""절 AM: 정답 MPM (lego L000 을 만든 PG 공식 warp 솔버, exe/gt_sens.py 기본 경로) 자체를 결과 불변으로 가속하면 실시간 (60 fps, 16.7 ms/프레임) 이 되는가.

정답 설정 그대로 (gt_sens 기본 경로 = lb_gen.py): 입자 313,750, n_grid 100 (dx 0.02), grid_lim 2, substep_dt = frame_dt/833 (2.0008e-5 s),
jelly (고정 FCR, 매 서브스텝 svd3), APIC (rpic 0), 중력, bounding_box (padding 3), 바닥 surface_collider sticky z 0.48, 감쇠 없음 (scale 1.1).
서브스텝 수·dt·재질·경계는 바꾸지 않는다. 초기 상태는 PG 솔버 객체를 gt_sens 와 똑같이 만든 뒤 그 warp 배열 (x, v, C, F_trial, mass, vol, mu, lam) 을 그대로 읽는다.

--mode pg   : 원래 솔버 그대로 (sol.p2g2p 833 회/프레임). 구간별 시간 = 솔버 자체의 wp.ScopedTimer (synchronize=True, 원래 코드에 있는 것) 기록.
--mode fast : 같은 식을 CuPy (NVRTC) 손 커널로. 가속 항목을 하나씩 켠다 (--sort --fuse --sparse --wred --graph).
              svd3 는 warp 설치본 native/svd.h 의 같은 소스를 그대로 잘라 넣는다 (같은 알고리즘·같은 연산 순서). 행렬·벡터 곱도 warp mat.h 와 같은 순서.
              결과가 달라질 수 있는 곳은 원자 합 순서 (정렬·워프 합) 뿐이다.
--mode floor: 서브스텝마다 입자 상태를 한 번 읽고 쓰는 것만 (읽기 26 float, 쓰기 24 float / 입자) -- 메모리 대역폭 바닥.

출력 DIR/: x.pt (float32 [F, N, 3], 원래 입자 순서), res.json (프레임별 ms, 정답 h5 와의 오차, 구간별 ms, 최대 |v| (CFL))
  cd <i-physgaussian> && python mpm_fast.py --out DIR --mode fast --sort --fuse --sparse --wred --graph
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--mode", default="fast", choices=["pg", "fast", "floor"])
ap.add_argument("--gt", default="/home/dkta/work/lbdata/test_low/L000")
ap.add_argument("--pt", default="/home/dkta/work/lbpt/test_low/L000.pt")
ap.add_argument("--config", default="/home/dkta/work/lbdata/config.json")
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--sort", action="store_true", help="프레임 시작마다 입자를 격자 칸 순서 (x 우선 선형 색인 = 격자 메모리 순서) 로 정렬, 내보낼 때 원래 순서로")
ap.add_argument("--fuse", action="store_true", help="G2P(s-1) + 응력 + P2G(s) 를 입자 커널 하나로, 정규화·중력·경계·비우기를 격자 커널 하나로 (서브스텝당 2 회 실행)")
ap.add_argument("--sparse", action="store_true", help="격자 커널을 입자가 닿은 4x4x4 블록만 (--fuse 필요)")
ap.add_argument("--wred", action="store_true", help="P2G 를 워프 안 같은 칸 입자끼리 셔플로 먼저 합친 뒤 칸 대표만 원자 합 (--fuse 필요)")
ap.add_argument("--shared", action="store_true", help="P2G 를 블록 공유 메모리 타일 (블록 입자 칸 상자 + 2, 최대 1536 노드) 에 먼저 모은 뒤 0 아닌 값만 전역 원자 합 (--fuse --sparse 필요)")
ap.add_argument("--graph", action="store_true", help="프레임 하나 (커널 1,667 개) 를 CUDA 그래프로")
ap.add_argument("--prof", default="10,40,70", help="구간별 시간을 잴 프레임 (CUDA 이벤트, 그래프 없이). 이 프레임들은 프레임 시간 통계에서 뺀다")
ap.add_argument("--block", type=int, default=128)
ap.add_argument("--no_dump", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.pg)
os.chdir(a.pg)
import h5py                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import warp as wp                                                # noqa: E402
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP   # noqa: E402

PROF = set(int(q) for q in a.prof.split(",") if q.strip()) if a.prof else set()
# ---------------- gt_sens.py 기본 경로와 똑같이 정답 솔버를 세운다 ----------------
cfg = json.load(open(a.config))
meta = json.load(open(os.path.join(a.gt, "meta.json")))
GL = float(cfg["grid_lim"])
n_grid = int(cfg["n_grid"])
dx0 = GL / n_grid
frame_dt = float(cfg["frame_dt"])
nsub = int(round(frame_dt / float(cfg["substep_dt"])))
dt = frame_dt / nsub
G = [float(q) for q in cfg["g"]]
floor = [b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]
d = torch.load(a.pt, map_location="cpu", weights_only=False, mmap=True)
x0d = d["x"][0].float()
EXT = float((x0d.max(0).values - x0d.min(0).values).norm())
del d, x0d


def h5x(t, key="x"):
    with h5py.File(os.path.join(a.gt, f"sim_{t:010d}.h5")) as h:
        r = np.array(h[key])
    return r.T.copy() if (r.ndim == 2 and r.shape[0] == 3 and r.shape[1] != 3) else r


x0 = h5x(0).astype(np.float32)
v0 = h5x(0, "v").astype(np.float32)
N = x0.shape[0]
L = min(a.frames, int(meta.get("frames", cfg.get("frame_num", a.frames))))
X = torch.as_tensor(x0).cuda().contiguous()
cell = (X / dx0).floor().long()
key = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
_, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
VOL = (dx0 ** 3 / cnt[inv].float()).contiguous()
wp.init()
sol = MPM_Simulator_WARP(10)
sol.load_initial_data_from_torch(X, VOL, None, n_grid=n_grid, grid_lim=GL)
sol.set_parameters_dict({"material": "jelly", "E": float(cfg["E"]), "nu": float(cfg["nu"]), "density": float(cfg["density"]),
                         "g": G, "n_grid": n_grid, "grid_lim": GL, "grid_v_damping_scale": 1.1, "rpic_damping": 0.0})
sol.add_bounding_box()
sol.add_surface_collider(tuple(floor["point"]), tuple(floor["normal"]), floor["surface"], float(floor.get("friction", 0.0)))
sol.finalize_mu_lam()
sol.import_particle_v_from_torch(torch.as_tensor(v0).cuda().contiguous())
wp.synchronize()
dev = torch.cuda.get_device_name(0)
print(f"[절 AM] 모드 {a.mode}, 입자 {N}, n_grid {n_grid} (dx {dx0:.5f}), 서브스텝 {dt:.4e} x {nsub}, 바닥 {floor['point']} {floor['surface']}, EXT {EXT:.6f}, {dev}", flush=True)
print(f"[명령] {' '.join(sys.argv)}", flush=True)
os.makedirs(a.out, exist_ok=True)
XD = None if a.no_dump else torch.empty(L, N, 3, dtype=torch.float32)
errs, fms, vmax = [], [], []
prof = {}


def record(f, xt, vt, ms):
    gt = torch.as_tensor(h5x(f)).cuda()
    e = float((xt - gt).norm(dim=-1).mean()) / EXT
    if not torch.isfinite(xt).all():
        e = float("nan")
    errs.append(e)
    fms.append(ms)
    vmax.append(float(vt.norm(dim=-1).max()))
    if XD is not None:
        XD[f - 1] = xt.cpu()
    if f <= 3 or f % 10 == 0 or f == L:
        print(f"  프레임 {f}/{L} 오차 {100 * e:.4f}% (누적 {100 * np.nanmean(errs):.4f}%) {ms:.2f} ms 최대|v| {vmax[-1]:.3f}", flush=True)


# ================================ 원래 솔버 그대로 ================================
if a.mode == "pg":
    step = 0
    for f in range(1, L + 1):
        sol.time_profile = {}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(nsub):
            sol.p2g2p(step, dt)
            step += 1
        wp.synchronize()
        ms = 1e3 * (time.perf_counter() - t0)
        if f in PROF or f >= 2:
            for k, vv in sol.time_profile.items():
                prof.setdefault(k, []).append(float(np.sum(vv)))
            prof.setdefault("_frame", []).append(ms)
        record(f, sol.export_particle_x_to_torch().clone(), wp.to_torch(sol.mpm_state.particle_v).clone(), ms)
    PRO = {k: float(np.mean(vv)) for k, vv in prof.items()}
    PRO["기타 (zero_grid·동기화·파이썬)"] = PRO["_frame"] - sum(vv for k, vv in PRO.items() if not k.startswith("_"))

# ================================ 손 커널 판 ================================
else:
    import cupy as cp
    SVD = open(os.path.join(os.path.dirname(wp.__file__), "native", "svd.h")).read()
    i0 = SVD.index("#define _gamma")
    i1 = SVD.index("template<typename Type>\ninline CUDA_CALLABLE void svd3(")
    SVD = SVD[i0:i1]
    SRC = r'''
#define CUDA_CALLABLE __device__
namespace wsvd {
__device__ __forceinline__ float sqrt(float x) { return sqrtf(x); }
__device__ __forceinline__ float abs(float x) { return fabsf(x); }
__device__ __forceinline__ float max(float a, float b) { return a > b ? a : b; }
''' + SVD + r'''
}
// 3x3 행렬은 warp mat33 과 같은 행 우선 data[i][j] = M[i*3+j]. 곱의 합 순서도 warp mat.h 와 같게 (0 에서 시작해 k 순서로 더함).
__device__ __forceinline__ void mm(const float* A, const float* B, float* T) {
  _Pragma("unroll") for (int i=0;i<3;i++) _Pragma("unroll") for (int j=0;j<3;j++) { float t = 0.f; _Pragma("unroll") for (int k=0;k<3;k++) t += A[i*3+k]*B[k*3+j]; T[i*3+j] = t; } }
__device__ __forceinline__ void mmT(const float* A, const float* B, float* T) {           // A * B^T
  _Pragma("unroll") for (int i=0;i<3;i++) _Pragma("unroll") for (int j=0;j<3;j++) { float t = 0.f; _Pragma("unroll") for (int k=0;k<3;k++) t += A[i*3+k]*B[j*3+k]; T[i*3+j] = t; } }

// 응력 (compute_stress_from_F_trial, jelly = 재질 0): F = F_trial, J = det, svd3, FCR, 대칭화
__device__ __forceinline__ void stress_fcr(const float* F, float mu, float lam, float* S) {
  // determinant = dot(row0, cross(row1, row2))
  float c0 = F[4]*F[8] - F[5]*F[7], c1 = F[5]*F[6] - F[3]*F[8], c2 = F[3]*F[7] - F[4]*F[6];
  float J = 0.f; J += F[0]*c0; J += F[1]*c1; J += F[2]*c2;
  float U[9], V[9], sg[9];
  wsvd::_svd(F[0],F[1],F[2],F[3],F[4],F[5],F[6],F[7],F[8],
             U[0],U[1],U[2],U[3],U[4],U[5],U[6],U[7],U[8],
             sg[0],sg[1],sg[2],sg[3],sg[4],sg[5],sg[6],sg[7],sg[8],
             V[0],V[1],V[2],V[3],V[4],V[5],V[6],V[7],V[8]);
  float R[9]; mmT(U, V, R);
  float tmu = 2.0f*mu, A[9];
  _Pragma("unroll") for (int k=0;k<9;k++) A[k] = tmu*(F[k]-R[k]);
  float T[9]; mmT(A, F, T);
  float dg = lam*J*(J-1.0f);
  float st[9];
  _Pragma("unroll") for (int i=0;i<3;i++) _Pragma("unroll") for (int j=0;j<3;j++) st[i*3+j] = T[i*3+j] + (i==j ? dg : 0.f*lam*J*(J-1.0f));
  _Pragma("unroll") for (int i=0;i<3;i++) _Pragma("unroll") for (int j=0;j<3;j++) S[i*3+j] = (st[i*3+j] + st[j*3+i]) / 2.0f;
}

// 2 차 B-스플라인 가중 (p2g_apic_with_stress / g2p 와 같은 식). w[a][i] = 축 a, 스텐실 i (warp mat33(c0,c1,c2) 는 열 생성자)
__device__ __forceinline__ void bspline(const float* xp, float inv_dx, int* b, float* fx, float w[3][3], float dw[3][3]) {
  _Pragma("unroll") _Pragma("unroll") for (int a=0;a<3;a++) {
    float gp = xp[a]*inv_dx;
    b[a] = (int)(gp - 0.5f);
    fx[a] = gp - (float)b[a];
    float wa = 1.5f - fx[a], wb = fx[a] - 1.0f, wc = fx[a] - 0.5f;
    w[a][0] = (wa*wa)*0.5f; w[a][1] = (0.0f - wb*wb) + 0.75f; w[a][2] = (wc*wc)*0.5f;
    dw[a][0] = fx[a] - 1.5f; dw[a][1] = -2.0f*(fx[a] - 1.0f); dw[a][2] = fx[a] - 0.5f;
  }
}

// G2P (g2p 커널과 같은 식): x, v, C, F (= F_trial) 를 갱신
__device__ __forceinline__ void g2p_body(float* x, float* v, float* C, float* F, const float* __restrict__ vout, int n, float inv_dx, float dt) {
  int b[3]; float fx[3], w[3][3], dw[3][3];
  bspline(x, inv_dx, b, fx, w, dw);
  float nv[3] = {0.f,0.f,0.f}, nC[9], nF[9];
  _Pragma("unroll") _Pragma("unroll") for (int k=0;k<9;k++) { nC[k] = 0.f; nF[k] = 0.f; }
  _Pragma("unroll") _Pragma("unroll") for (int i=0;i<3;i++) _Pragma("unroll") for (int j=0;j<3;j++) _Pragma("unroll") for (int k=0;k<3;k++) {
    long long id = ((long long)(b[0]+i)*n + (b[1]+j))*n + (b[2]+k);
    float dp[3] = {(float)i - fx[0], (float)j - fx[1], (float)k - fx[2]};
    float wt = w[0][i]*w[1][j]*w[2][k];
    float g[3] = {vout[id*3], vout[id*3+1], vout[id*3+2]};
    _Pragma("unroll") _Pragma("unroll") for (int r=0;r<3;r++) nv[r] = nv[r] + g[r]*wt;
    float cf = wt*inv_dx*4.0f;
    _Pragma("unroll") _Pragma("unroll") for (int r=0;r<3;r++) _Pragma("unroll") for (int c=0;c<3;c++) nC[r*3+c] = nC[r*3+c] + (g[r]*dp[c])*cf;
    float dwt[3] = {dw[0][i]*w[1][j]*w[2][k]*inv_dx, w[0][i]*dw[1][j]*w[2][k]*inv_dx, w[0][i]*w[1][j]*dw[2][k]*inv_dx};
    _Pragma("unroll") _Pragma("unroll") for (int r=0;r<3;r++) _Pragma("unroll") for (int c=0;c<3;c++) nF[r*3+c] = nF[r*3+c] + g[r]*dwt[c];
  }
  _Pragma("unroll") _Pragma("unroll") for (int r=0;r<3;r++) { v[r] = nv[r]; x[r] = x[r] + dt*nv[r]; }
  _Pragma("unroll") _Pragma("unroll") for (int k=0;k<9;k++) C[k] = nC[k];
  float A[9]; for (int k=0;k<9;k++) A[k] = (k%4==0 ? 1.0f : 0.0f) + nF[k]*dt;
  float T[9]; mm(A, F, T);
  _Pragma("unroll") _Pragma("unroll") for (int k=0;k<9;k++) F[k] = T[k];
}

#define SHCAP 1536
// 워프 안 구간 합 (입자가 칸 순서로 정렬돼 있을 때 같은 칸 = 같은 27 노드). stride = 다음 구간 머리까지 거리
__device__ __forceinline__ float seg_sum(float val, int stride) {
  _Pragma("unroll") _Pragma("unroll") for (int s=1; s<32; s<<=1) { float t = __shfl_down_sync(0xffffffffu, val, s); if (s < stride) val += t; }
  return val;
}

// P2G (p2g_apic_with_stress 와 같은 식, rpic 0 -> C 그대로)
template <int RED, int MARK, int SH = 0>
__device__ __forceinline__ void p2g_body(bool act, const float* x, const float* v, const float* C, const float* S, float mass, float vol,
                                         float* vin, float* gm, int* flags, int n, int nb, float dx, float inv_dx, float dt,
                                         float* sh = nullptr, bool useSh = false, int ox = 0, int oy = 0, int oz = 0, int ey = 0, int ez = 0) {
  int b[3] = {0,0,0}; float fx[3] = {0.f,0.f,0.f}, w[3][3], dw[3][3];
  if (act) bspline(x, inv_dx, b, fx, w, dw);
  else { for (int q=0;q<3;q++) _Pragma("unroll") for (int r=0;r<3;r++) { w[q][r] = 0.f; dw[q][r] = 0.f; } }
  long long key = act ? (((long long)b[0]*n + b[1])*n + b[2]) : -1 - (long long)(threadIdx.x & 31);
  int stride = 1; bool head = act;
  if (RED) {
    int lane = threadIdx.x & 31;
    long long kp = __shfl_up_sync(0xffffffffu, key, 1);
    head = (lane == 0) || (kp != key);
    unsigned hm = __ballot_sync(0xffffffffu, head);
    unsigned above = (lane == 31) ? 0u : (hm & ~((2u << lane) - 1u));
    int nexth = above ? (__ffs(above) - 1) : 32;
    stride = nexth - lane;
    head = head && act;
  }
  if (MARK && head) {
    _Pragma("unroll") _Pragma("unroll") for (int q=0;q<8;q++) {
      int bx = (b[0] + ((q>>2)&1)*2) >> 2, by = (b[1] + ((q>>1)&1)*2) >> 2, bz = (b[2] + (q&1)*2) >> 2;
      flags[(bx*nb + by)*nb + bz] = 1;
    }
  }
  float M[9]; for (int k=0;k<9;k++) M[k] = (-vol)*S[k];
  _Pragma("unroll") _Pragma("unroll") for (int i=0;i<3;i++) _Pragma("unroll") for (int j=0;j<3;j++) _Pragma("unroll") for (int k=0;k<3;k++) {
    float dp[3] = {((float)i - fx[0])*dx, ((float)j - fx[1])*dx, ((float)k - fx[2])*dx};
    float wt = w[0][i]*w[1][j]*w[2][k];
    float dwt[3] = {dw[0][i]*w[1][j]*w[2][k]*inv_dx, w[0][i]*dw[1][j]*w[2][k]*inv_dx, w[0][i]*w[1][j]*dw[2][k]*inv_dx};
    float add[4];
    float wm = wt*mass;
    _Pragma("unroll") _Pragma("unroll") for (int r=0;r<3;r++) {
      float cd = C[r*3]*dp[0]; cd = cd + C[r*3+1]*dp[1]; cd = cd + C[r*3+2]*dp[2];
      float ef = M[r*3]*dwt[0]; ef = ef + M[r*3+1]*dwt[1]; ef = ef + M[r*3+2]*dwt[2];
      add[r] = wm*(v[r] + cd) + dt*ef;
    }
    add[3] = wm;
    if (RED) { for (int r=0;r<4;r++) add[r] = seg_sum(add[r], stride); }
    if (head) {
      if (SH && useSh) {                                                  // 블록 공유 메모리 타일에 먼저 (블록 끝에 한 번에 전역으로)
        int li = ((b[0]+i-ox)*ey + (b[1]+j-oy))*ez + (b[2]+k-oz);
        atomicAdd(&sh[li], add[0]); atomicAdd(&sh[SHCAP+li], add[1]); atomicAdd(&sh[2*SHCAP+li], add[2]); atomicAdd(&sh[3*SHCAP+li], add[3]);
      } else {
        long long id = ((long long)(b[0]+i)*n + (b[1]+j))*n + (b[2]+k);
        atomicAdd(&vin[id*3], add[0]); atomicAdd(&vin[id*3+1], add[1]); atomicAdd(&vin[id*3+2], add[2]);
        atomicAdd(&gm[id], add[3]);
      }
    }
  }
}

#define LD3(P,p,o) { o[0]=P[(p)*3]; o[1]=P[(p)*3+1]; o[2]=P[(p)*3+2]; }
#define LD9(P,p,o) { for (int _k=0;_k<9;_k++) o[_k]=P[(p)*9+_k]; }
#define ST3(P,p,o) { P[(p)*3]=o[0]; P[(p)*3+1]=o[1]; P[(p)*3+2]=o[2]; }
#define ST9(P,p,o) { for (int _k=0;_k<9;_k++) P[(p)*9+_k]=o[_k]; }

// 융합 입자 커널: [G2P (지난 서브스텝 격자)] + 응력 + P2G (이번 서브스텝 격자)
template <int G2P, int RED, int MARK, int SH = 0>
__device__ void fused(int N, int n, int nb, float dx, float inv_dx, float dt, float mu, float lam,
                      float* X, float* Vv, float* Cc, float* Ff, const float* __restrict__ mass, const float* __restrict__ vol,
                      const float* __restrict__ vout, float* vin, float* gm, int* flags) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  bool act = p < N;
  float x[3] = {0,0,0}, v[3] = {0,0,0}, C[9], F[9], S[9];
  float m = 0.f, vl = 0.f;
  for (int k=0;k<9;k++) { C[k]=0.f; F[k]=0.f; S[k]=0.f; }
  if (act) {
    LD3(X,p,x); LD3(Vv,p,v); LD9(Cc,p,C); LD9(Ff,p,F); m = mass[p]; vl = vol[p];
    if (G2P) g2p_body(x, v, C, F, vout, n, inv_dx, dt);
    stress_fcr(F, mu, lam, S);
  }
  if (SH) {
    // 블록 입자들의 칸 상자 (정렬돼 있으면 작다) -> 상자 + 2 노드가 SHCAP 이하이면 공유 메모리 타일에 모은 뒤 0 아닌 값만 전역 원자 합
    __shared__ int bb[6];
    __shared__ float sh[4*SHCAP];
    if (threadIdx.x == 0) { bb[0] = bb[1] = bb[2] = 0x7fffffff; bb[3] = bb[4] = bb[5] = -0x7fffffff; }
    __syncthreads();
    if (act) for (int q=0;q<3;q++) { int bq = (int)(x[q]*inv_dx - 0.5f); atomicMin(&bb[q], bq); atomicMax(&bb[3+q], bq); }
    __syncthreads();
    int ox = bb[0], oy = bb[1], oz = bb[2];
    int ex = bb[3]-ox+3, ey = bb[4]-oy+3, ez = bb[5]-oz+3;
    bool useSh = (bb[3] >= bb[0]) && (ex*ey*ez <= SHCAP);
    int sz = useSh ? ex*ey*ez : 0;
    for (int q = threadIdx.x; q < sz; q += blockDim.x) { sh[q] = 0.f; sh[SHCAP+q] = 0.f; sh[2*SHCAP+q] = 0.f; sh[3*SHCAP+q] = 0.f; }
    __syncthreads();
    p2g_body<RED, MARK, 1>(act, x, v, C, S, m, vl, vin, gm, flags, n, nb, dx, inv_dx, dt, sh, useSh, ox, oy, oz, ey, ez);
    __syncthreads();
    for (int q = threadIdx.x; q < sz; q += blockDim.x) {
      int lz = q % ez, ly = (q / ez) % ey, lx = q / (ez*ey);
      long long id = ((long long)(ox+lx)*n + (oy+ly))*n + (oz+lz);
      float a0 = sh[q], a1 = sh[SHCAP+q], a2 = sh[2*SHCAP+q], a3 = sh[3*SHCAP+q];
      if (a0 != 0.f) atomicAdd(&vin[id*3], a0);
      if (a1 != 0.f) atomicAdd(&vin[id*3+1], a1);
      if (a2 != 0.f) atomicAdd(&vin[id*3+2], a2);
      if (a3 != 0.f) atomicAdd(&gm[id], a3);
    }
  } else {
    p2g_body<RED, MARK>(act, x, v, C, S, m, vl, vin, gm, flags, n, nb, dx, inv_dx, dt);
  }
  if (act && G2P) { ST3(X,p,x); ST3(Vv,p,v); ST9(Cc,p,C); ST9(Ff,p,F); }
}

// 격자: 정규화 + 중력 (grid_normalization_and_gravity) -> bounding_box (padding 3) -> 바닥 sticky (z*dx - pz < 0 이면 0)
__device__ __forceinline__ void grid_node(int gx, int gy, int gz, int n, float dx, float dt, float gxa, float gya, float gza, float px, float py, float pz,
                                          float nx, float ny, float nz, const float* vin, const float* gm, float* vout) {
  long long id = ((long long)gx*n + gy)*n + gz;
  float mm_ = gm[id];
  float o[3] = {0.f, 0.f, 0.f};
  if (mm_ > 1e-15f) {
    float r = 1.0f / mm_;
    o[0] = vin[id*3]*r; o[1] = vin[id*3+1]*r; o[2] = vin[id*3+2]*r;
    o[0] = o[0] + dt*gxa; o[1] = o[1] + dt*gya; o[2] = o[2] + dt*gza;
  }
  const int pad = 3;
  if (gx < pad && o[0] < 0) o[0] = 0.f;
  if (gx >= n - pad && o[0] > 0) o[0] = 0.f;
  if (gy < pad && o[1] < 0) o[1] = 0.f;
  if (gy >= n - pad && o[1] > 0) o[1] = 0.f;
  if (gz < pad && o[2] < 0) o[2] = 0.f;
  if (gz >= n - pad && o[2] > 0) o[2] = 0.f;
  float ox = __fsub_rn(__fmul_rn((float)gx, dx), px), oy = __fsub_rn(__fmul_rn((float)gy, dx), py), oz = __fsub_rn(__fmul_rn((float)gz, dx), pz);
  float dd = 0.f; dd = __fadd_rn(dd, __fmul_rn(ox, nx)); dd = __fadd_rn(dd, __fmul_rn(oy, ny)); dd = __fadd_rn(dd, __fmul_rn(oz, nz));
  if (dd < 0.0f) { o[0] = 0.f; o[1] = 0.f; o[2] = 0.f; }
  vout[id*3] = o[0]; vout[id*3+1] = o[1]; vout[id*3+2] = o[2];
}

#define GRID_ARGS int n, float dx, float dt, float gxa, float gya, float gza, float px, float py, float pz, float nx, float ny, float nz
#define GRID_PASS n, dx, dt, gxa, gya, gza, px, py, pz, nx, ny, nz

extern "C" {
#define FUSED(NM, G, R, M) FUSEDS(NM, G, R, M, 0)
#define FUSEDS(NM, G, R, M, SHV) __global__ void NM(int N, int n, int nb, float dx, float inv_dx, float dt, float mu, float lam, float* X, float* Vv, float* Cc, float* Ff, \
   const float* __restrict__ mass, const float* __restrict__ vol, const float* __restrict__ vout, float* vin, float* gm, int* flags) \
   { fused<G, R, M, SHV>(N, n, nb, dx, inv_dx, dt, mu, lam, X, Vv, Cc, Ff, mass, vol, vout, vin, gm, flags); }
FUSED(k_fused_000, 0, 0, 0) FUSED(k_fused_100, 1, 0, 0)
FUSED(k_fused_001, 0, 0, 1) FUSED(k_fused_101, 1, 0, 1)
FUSED(k_fused_011, 0, 1, 1) FUSED(k_fused_111, 1, 1, 1)
FUSED(k_fused_010, 0, 1, 0) FUSED(k_fused_110, 1, 1, 0)
FUSEDS(k_fused_001s, 0, 0, 1, 1) FUSEDS(k_fused_101s, 1, 0, 1, 1)
FUSEDS(k_fused_011s, 0, 1, 1, 1) FUSEDS(k_fused_111s, 1, 1, 1, 1)

// 원래 솔버와 같은 나눔 (비교판): 비우기 / 응력 / P2G / 정규화 / 상자 / 바닥 / G2P 따로
__global__ void k_zero(int G3, float* gm, float* vin, float* vout) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= G3) return;
  gm[i] = 0.f; vin[i*3] = 0.f; vin[i*3+1] = 0.f; vin[i*3+2] = 0.f; vout[i*3] = 0.f; vout[i*3+1] = 0.f; vout[i*3+2] = 0.f; }
__global__ void k_stress(int N, float mu, float lam, const float* Ff, float* Ss) {
  int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= N) return;
  float F[9], S[9]; LD9(Ff,p,F); stress_fcr(F, mu, lam, S); ST9(Ss,p,S); }
__global__ void k_p2g(int N, int n, float dx, float inv_dx, float dt, const float* X, const float* Vv, const float* Cc, const float* Ss,
                      const float* mass, const float* vol, float* vin, float* gm) {
  int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= N) return;
  float x[3], v[3], C[9], S[9]; LD3(X,p,x); LD3(Vv,p,v); LD9(Cc,p,C); LD9(Ss,p,S);
  p2g_body<0, 0>(true, x, v, C, S, mass[p], vol[p], vin, gm, nullptr, n, 0, dx, inv_dx, dt); }
__global__ void k_gnorm(int n, float dt, float gxa, float gya, float gza, const float* vin, const float* gm, float* vout) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n*n*n) return;
  float mm_ = gm[i];
  if (mm_ > 1e-15f) { float r = 1.0f / mm_; vout[i*3] = vin[i*3]*r + dt*gxa; vout[i*3+1] = vin[i*3+1]*r + dt*gya; vout[i*3+2] = vin[i*3+2]*r + dt*gza; } }
__global__ void k_gbc(GRID_ARGS, const float* vin, const float* gm, float* vout) {    // 상자 + 바닥 (정규화 끝난 vout 에)
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n*n*n) return;
  int gz = i % n, gy = (i / n) % n, gx = i / (n*n);
  float o[3] = {vout[i*3], vout[i*3+1], vout[i*3+2]};
  const int pad = 3;
  if (gx < pad && o[0] < 0) o[0] = 0.f;
  if (gx >= n - pad && o[0] > 0) o[0] = 0.f;
  if (gy < pad && o[1] < 0) o[1] = 0.f;
  if (gy >= n - pad && o[1] > 0) o[1] = 0.f;
  if (gz < pad && o[2] < 0) o[2] = 0.f;
  if (gz >= n - pad && o[2] > 0) o[2] = 0.f;
  float ox = __fsub_rn(__fmul_rn((float)gx, dx), px), oy = __fsub_rn(__fmul_rn((float)gy, dx), py), oz = __fsub_rn(__fmul_rn((float)gz, dx), pz);
  float dd = 0.f; dd = __fadd_rn(dd, __fmul_rn(ox, nx)); dd = __fadd_rn(dd, __fmul_rn(oy, ny)); dd = __fadd_rn(dd, __fmul_rn(oz, nz));
  if (dd < 0.0f) { o[0] = 0.f; o[1] = 0.f; o[2] = 0.f; }
  vout[i*3] = o[0]; vout[i*3+1] = o[1]; vout[i*3+2] = o[2]; }
__global__ void k_g2p(int N, int n, float inv_dx, float dt, float* X, float* Vv, float* Cc, float* Ff, const float* __restrict__ vout) {
  int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= N) return;
  float x[3], v[3], C[9], F[9]; LD3(X,p,x); LD3(Vv,p,v); LD9(Cc,p,C); LD9(Ff,p,F);
  g2p_body(x, v, C, F, vout, n, inv_dx, dt);
  ST3(X,p,x); ST3(Vv,p,v); ST9(Cc,p,C); ST9(Ff,p,F); }

// 융합 격자 커널 (조밀): 전 노드 정규화·경계 -> vout, 그리고 vin·m 비우기 (다음 서브스텝 P2G 용)
__global__ void k_grid_dense(GRID_ARGS, float* vin, float* gm, float* vout) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n*n*n) return;
  int gz = i % n, gy = (i / n) % n, gx = i / (n*n);
  grid_node(gx, gy, gz, GRID_PASS, vin, gm, vout);
  gm[i] = 0.f; vin[i*3] = 0.f; vin[i*3+1] = 0.f; vin[i*3+2] = 0.f; }
// 융합 격자 커널 (희소): 블록 하나 = 4x4x4 노드, 이번 서브스텝 P2G 가 표시한 블록만
__global__ void k_grid_sparse(GRID_ARGS, int nb, int* flags, float* vin, float* gm, float* vout) {
  int bi = blockIdx.x;
  int f = flags[bi];
  __syncthreads();
  if (!f) return;
  int bz = bi % nb, by = (bi / nb) % nb, bx = bi / (nb*nb);
  int t = threadIdx.x;
  int gx = bx*4 + (t >> 4), gy = by*4 + ((t >> 2) & 3), gz = bz*4 + (t & 3);
  if (gx < n && gy < n && gz < n) {
    long long i = ((long long)gx*n + gy)*n + gz;
    grid_node(gx, gy, gz, GRID_PASS, vin, gm, vout);
    gm[i] = 0.f; vin[i*3] = 0.f; vin[i*3+1] = 0.f; vin[i*3+2] = 0.f;
  }
  if (t == 0) flags[bi] = 0;
}
// 대역폭 바닥: 입자 상태 읽기 26 float (x v C F m vol), 쓰기 24 float (x v C F) -- 계산 없음
__global__ void k_floor(int N, float* X, float* Vv, float* Cc, float* Ff, const float* mass, const float* vol) {
  int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= N) return;
  float x[3], v[3], C[9], F[9]; LD3(X,p,x); LD3(Vv,p,v); LD9(Cc,p,C); LD9(Ff,p,F);
  float s = mass[p] + vol[p];
  x[0] += s*1e-30f;
  ST3(X,p,x); ST3(Vv,p,v); ST9(Cc,p,C); ST9(Ff,p,F); }
}
'''
    mod = cp.RawModule(code=SRC, options=("--std=c++14",))
    st = sol.mpm_state
    md = sol.mpm_model
    Xt = wp.to_torch(st.particle_x).clone().contiguous()                              # [N,3]
    Vt = wp.to_torch(st.particle_v).clone().contiguous()
    Ct = wp.to_torch(st.particle_C).clone().reshape(N, 9).contiguous()
    Ft = wp.to_torch(st.particle_F_trial).clone().reshape(N, 9).contiguous()          # 첫 응력 커널은 F <- F_trial (= I)
    Mt = wp.to_torch(st.particle_mass).clone().contiguous()
    Vo = wp.to_torch(st.particle_vol).clone().contiguous()
    mu_t, lam_t = wp.to_torch(md.mu), wp.to_torch(md.lam)
    assert bool((mu_t == mu_t[0]).all()) and bool((lam_t == lam_t[0]).all())
    MU, LAM = np.float32(float(mu_t[0])), np.float32(float(lam_t[0]))
    DX, IDX = np.float32(md.dx), np.float32(md.inv_dx)
    ga = md.gravitational_accelaration
    GXA, GYA, GZA = np.float32(ga[0]), np.float32(ga[1]), np.float32(ga[2])
    cpar = sol.collider_params[1]
    PX, PY, PZ = (np.float32(cpar.point[k]) for k in range(3))
    NX, NY, NZ = (np.float32(cpar.normal[k]) for k in range(3))
    assert sol.collider_params[1].surface_type == 0 and len(sol.grid_postprocess) == 2
    DT = np.float32(dt)
    n = n_grid
    G3 = n ** 3
    NB = (n + 3) // 4
    gm = torch.zeros(G3, device="cuda")
    vin = torch.zeros(G3 * 3, device="cuda")
    vout = torch.zeros(G3 * 3, device="cuda")
    flags = torch.zeros(NB ** 3, dtype=torch.int32, device="cuda")
    Ss = torch.zeros(N, 9, device="cuda")
    order = torch.arange(N, device="cuda")
    print(f"[손 커널] dx {DX} inv_dx {IDX} mu {MU} lam {LAM} g {GXA},{GYA},{GZA} 바닥 {PX},{PY},{PZ} n {NX},{NY},{NZ} dt {DT}; 정렬 {a.sort} 융합 {a.fuse} 희소 {a.sparse} 워프합 {a.wred} 공유타일 {a.shared} 블록 {a.block} 그래프 {a.graph}", flush=True)
    for nm in ("k_fused_111s", "k_fused_111", "k_fused_011", "k_fused_101", "k_g2p", "k_p2g", "k_stress"):
        fn = mod.get_function(nm)
        print(f"  {nm}: 레지스터 {fn.num_regs}, 국소 메모리 {fn.local_size_bytes} B", flush=True)
    U64 = lambda t: np.uint64(t.data_ptr())                                          # noqa: E731
    _XS = {}

    def stream():
        sp = torch.cuda.current_stream().cuda_stream
        if sp not in _XS:
            _XS[sp] = cp.cuda.ExternalStream(sp)
        return _XS[sp]

    B = a.block
    gP = ((N + B - 1) // B,)
    gG = ((G3 + 255) // 256,)
    PA = lambda: (U64(Xt), U64(Vt), U64(Ct), U64(Ft), U64(Mt), U64(Vo))              # noqa: E731
    GA = (np.int32(n), DX, DT, GXA, GYA, GZA, PX, PY, PZ, NX, NY, NZ)
    fz = {k: mod.get_function(k) for k in ("k_fused_000", "k_fused_100", "k_fused_001", "k_fused_101", "k_fused_011", "k_fused_111", "k_fused_010", "k_fused_110",
                                           "k_fused_001s", "k_fused_101s", "k_fused_011s", "k_fused_111s")}
    kz, ks, kp, kn, kb, kg = (mod.get_function(k) for k in ("k_zero", "k_stress", "k_p2g", "k_gnorm", "k_gbc", "k_g2p"))
    kgd, kgs, kfl = mod.get_function("k_grid_dense"), mod.get_function("k_grid_sparse"), mod.get_function("k_floor")
    EV = {}

    def ev(name, fn):
        """구간별 시간 (prof 프레임만): 이벤트 두 개로 감싼다."""
        if not PROFILING[0]:
            fn()
            return
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        EV.setdefault(name, []).append((e0, e1))

    PROFILING = [False]

    def fused_launch(g2p):
        k = fz[f"k_fused_{int(g2p)}{int(a.wred)}{int(a.sparse)}" + ("s" if a.shared else "")]
        fn = lambda: k(gP, (B,), (np.int32(N), np.int32(n), np.int32(NB), DX, IDX, DT, MU, LAM, *PA(), U64(vout), U64(vin), U64(gm), U64(flags)))  # noqa: E731
        ev("입자 (G2P+응력+P2G 융합)" if g2p else "입자 (응력+P2G, 프레임 첫 서브스텝)", fn)

    def grid_launch():
        if a.sparse:
            fn = lambda: kgs((NB ** 3,), (64,), (*GA, np.int32(NB), U64(flags), U64(vin), U64(gm), U64(vout)))  # noqa: E731
        else:
            fn = lambda: kgd(gG, (256,), (*GA, U64(vin), U64(gm), U64(vout)))  # noqa: E731
        ev("격자 (정규화+중력+경계+비우기)", fn)

    def g2p_launch():
        ev("G2P (프레임 끝)" if a.fuse else "G2P", lambda: kg(gP, (B,), (np.int32(N), np.int32(n), IDX, DT, U64(Xt), U64(Vt), U64(Ct), U64(Ft), U64(vout))))

    def frame_body():
        with stream():
            if a.mode == "floor":
                for s in range(nsub):
                    kfl(gP, (B,), (np.int32(N), *PA()))
                return
            if a.fuse:
                for s in range(nsub):
                    fused_launch(s > 0)
                    grid_launch()
                g2p_launch()
            else:
                for s in range(nsub):
                    ev("비우기", lambda: kz(gG, (256,), (np.int32(G3), U64(gm), U64(vin), U64(vout))))
                    ev("응력", lambda: ks(gP, (B,), (np.int32(N), MU, LAM, U64(Ft), U64(Ss))))
                    ev("P2G", lambda: kp(gP, (B,), (np.int32(N), np.int32(n), DX, IDX, DT, U64(Xt), U64(Vt), U64(Ct), U64(Ss), U64(Mt), U64(Vo), U64(vin), U64(gm))))
                    ev("격자 정규화+중력", lambda: kn(gG, (256,), (np.int32(n), DT, GXA, GYA, GZA, U64(vin), U64(gm), U64(vout))))
                    ev("격자 경계 (상자+바닥)", lambda: kb(gG, (256,), (*GA, U64(vin), U64(gm), U64(vout))))
                    g2p_launch()

    def do_sort():
        b = (Xt * float(IDX) - 0.5).to(torch.int32).long()
        k = (b[:, 0] * n + b[:, 1]) * n + b[:, 2]
        perm = torch.argsort(k, stable=True)
        for t in (Xt, Vt, Ct, Ft, Mt, Vo):
            t.copy_(t[perm])
        order.copy_(order[perm])

    def sort_launch():
        if a.sort:
            ev("정렬", do_sort)

    graph = None
    if a.graph:
        torch.cuda.synchronize()
        s_ = torch.cuda.Stream()
        s_.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        # 잡는 동안에는 실행되지 않으므로 상태를 바꾸지 않는다
        with torch.cuda.stream(s_):
            graph.capture_begin()
            frame_body()
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(s_)
        torch.cuda.synchronize()

    def export():
        xo = torch.empty_like(Xt)
        xo[order] = Xt
        vo_ = torch.empty_like(Vt)
        vo_[order] = Vt
        return xo, vo_

    for f in range(1, L + 1):
        PROFILING[0] = f in PROF
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sort_launch()
        if graph is not None and not PROFILING[0]:
            graph.replay()
        else:
            frame_body()
        torch.cuda.synchronize()
        ms = 1e3 * (time.perf_counter() - t0)
        xt, vt = export()
        record(f, xt, vt, ms)
        if a.mode == "floor" and f >= 5:
            break
    PRO = {k: float(sum(e0.elapsed_time(e1) for e0, e1 in vv)) / max(len(PROF), 1) for k, vv in EV.items()}
    if PROF:
        PRO["_prof_frame_ms"] = float(np.mean([fms[f - 1] for f in PROF if f <= len(fms)]))

# ================================ 정리 ================================
keep = [fms[i] for i in range(len(fms)) if (i + 1) >= 2 and (i + 1) not in PROF]
cfl_v = max(vmax) * dt / dx0 if vmax else float("nan")
E_, nu_, rho = float(cfg["E"]), float(cfg["nu"]), float(cfg["density"])
mu_, lam_ = E_ / (2 * (1 + nu_)), E_ * nu_ / ((1 + nu_) * (1 - 2 * nu_))
c_p = float(np.sqrt((lam_ + 2 * mu_) / rho))
summ = dict(args=vars(a), device=dev, N=N, nsub=nsub, dt=dt, dx=dx0, EXT=EXT, frames_done=len(errs),
            err_mean=float(np.mean(errs)) if errs else float("nan"), err=errs, frame_ms=fms,
            frame_ms_median=float(np.median(keep)) if keep else float("nan"), frame_ms_mean=float(np.mean(keep)) if keep else float("nan"),
            substep_us_median=1e3 * float(np.median(keep)) / nsub if keep else float("nan"),
            prof_ms_per_frame=PRO, vmax=vmax, cfl_v=cfl_v, c_p=c_p, cfl_c=c_p * dt / dx0,
            dt_cfl_v1=dx0 / max(vmax) if vmax else None, dt_cfl_c1=dx0 / c_p)
json.dump(summ, open(os.path.join(a.out, "res.json"), "w"), ensure_ascii=False, indent=1)
if XD is not None:
    torch.save(XD[:len(errs)].clone(), os.path.join(a.out, "x.pt"))
print(f"[절 AM 결과] {os.path.basename(a.out)}: 정답 대비 {100 * summ['err_mean']:.4f}% ({len(errs)}/{L}), 프레임 중앙값 {summ['frame_ms_median']:.2f} ms "
      f"(서브스텝 {summ['substep_us_median']:.1f} us), 구간 {json.dumps({k: round(vv, 2) for k, vv in PRO.items()}, ensure_ascii=False)}, "
      f"CFL 속도 {cfl_v:.4f} 탄성파 {summ['cfl_c']:.4f}", flush=True)
