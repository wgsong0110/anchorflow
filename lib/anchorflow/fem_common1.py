"""절 CM31 묶음 common1 (새 실행기 fem_an_rt 훅만): 덜 풀린 걸음 후퇴 오일러 갱신을 노드 국소로 (--ue_loc), 덜 풀린 걸음 판정 2 의 힘 척도에 무게를 더함 (--uf_w).

정의 (끄면 생성 CUDA 소스 글자 그대로 -- 훅은 플래그 켤 때만 소스·커널 열을 바꿈):
- --ue_loc M (비트 합, --ue be/a0 위): 덜 풀린 걸음 (판정 커널 cjl = 1) 의 걸음 끝 갱신 바꿈을 노드 집합 S 에만, 나머지 노드는 HHT 그대로.
    1: 걸음 끝 y 에서 장벽이 걸린 표본점 (활성, 0 < d < d̂ 또는 d ≤ 0) 이 걸친 노드 (B-스플라인 가중 ≠ 0) -- 접촉 노드
    2: --y0loc 의 시작점 β_n < 1 노드 (kn > 0 = ŷ 에서 관통할 표본점이 걸친 노드) -- 관통 표본점 주변
    4: 마지막으로 돈 뉴턴 반복 시작 잔차 힘 |g_n| 이 그 노드 무게 m_n |g| 보다 큰 노드 (잔차 가속도 ≥ 중력 가속도)
- --uf_w (--uf_res 위): 판정 2 의 ‖g‖ > F 에서 F² = ‖T f_0‖² (탄성·접촉 힘, 지금) 대신 ‖T f_0‖² + Σ_n (m_n |g|)² (무게를 더한 몸 전체 힘 척도) --
    자유 낙하·탄성 힘이 작은 걸음에서 F 가 작아 잔차 비가 커지는 것을 막음 (문턱 1 은 그대로).
"""

UL_SRC = r"""
// 절 CM31 --ue_loc / --uf_w (lib/anchorflow/fem_common1.py)
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
"""

UL_BAR_SRC = r"""
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
"""


def ulify(src, uel, ufw, bar):
    """모듈 끝에 UL 커널 (끄면 소스 그대로)."""
    if not (uel or ufw):
        return src
    i = src.rstrip().rfind("}")
    return src[:i] + UL_SRC + (UL_BAR_SRC if bar else "") + "\n}\n"


def uelify(src):
    """cjueify 가 만든 rt_ga_end_uf 를 복사해 노드 집합 ulm 만 바꾸는 rt_ga_end_ufl 을 그 뒤에 (원래 두 커널 그대로)."""
    i = src.index("__global__ void rt_ga_end_uf(")
    j = src.index("\n}\n", i) + 3
    k = src[i:j].replace("rt_ga_end_uf(", "rt_ga_end_ufl(", 1)
    for o_, n_ in (("float damp, const int* cjl, int um) {", "float damp, const int* cjl, int um, const int* ulm) {"),
                   ("    if (n >= IP[9]) return;\n", "    if (n >= IP[9]) return;\n    uf_ = uf_ && ulm[n];                                                        // 절 CM31 --ue_loc: 노드 집합만\n")):
        assert k.count(o_) == 1, o_
        k = k.replace(o_, n_, 1)
    return src[:j] + k + src[j:]
