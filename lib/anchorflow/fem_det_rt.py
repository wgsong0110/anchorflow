"""결정적 합산 (같은 명령 두 번이 비트 단위로 같게) -- 절 AN 실시간 판 (fem_an_rt.RTSolver) 과 절 AO 모래 판 공통.
절 AN 커널 원본은 고치지 않는다: (1) det_src() 가 소스 문자열에서 원자 합 줄을 AO_DET 분기로 바꾼 판을 만들고 (AO_DET = 0 이면 원래 동작 그대로),
(2) DetMixin.k() 가 커널 실행을 가로채 결정적 짝 커널을 이어 실행한다. 쓰는 쪽: class X(DetMixin) 또는 기존 하위 클래스의 기반으로 (예: fem_ao_rt.AORTSolver, RTAdapt).
원자 합을 바꾼 곳과 방식:
- 가우스 점 -> 노드 흩어 쓰기 (rt_elem·ao_elem 힘 f64·노드 대각 B, rt_mv 곱): 칸마다 (8 레인 구간 합 뒤 머리 레인) 칸 버퍼 [칸 x 27 x 성분] 에 한 번 쓰기 ->
  ao_gath_* 가 노드마다 이웃 27 칸을 고정 순서 (o = 0..26) 로 모은다.
- rt_gauss 노드 질량 몫 mqn, rt_floor_q 고착 몫 an: 점 값 (mq, 점별 mw) 에서 노드가 이웃 27 칸 x 8 점을 고정 순서로 모은다.
- P2G (rt_p2g) 와 구속 누적 (ao_pbc_acc): 프레임마다 입자를 칸 번호로 안정 정렬 (torch.sort stable, 입자 배열·원래 순번 perm 도 같이 재배열),
  칸 구간 시작 st[c] -> 칸 하나가 자기 입자를 차례로 (A·C·bad), 노드 하나가 이웃 27 칸 입자를 차례로 (질량·운동량·구속 질량).
- 블록 합 (blocksum_add, 내적·에너지): 대상이 SC 배열이면 블록 부분합을 AO_PART[칸][블록] 에 쓰고 ao_bsfin 이 고정 순서 (스레드별 보폭 합 + 공유 메모리 트리) 로 더한다.
- 노드 목록 (ao_act, 압축 판): 원자 카운터 대신 한 블록 스캔 (ao_act_scan) 으로 번호 순서 그대로.
남은 원자 연산은 정수 (상자 최소·최대, 칸 점유, 뒤집힘 표시, 진단 수) 뿐이라 순서와 무관하다. MG 커널은 바꾸지 않았다 (결정적 모드에서는 MG 를 쓰지 말 것).
"""
import numpy as np
import torch

from . import fem_an_rt as ART

P = ART.P
MAXB = 16384                                                                      # 블록 부분합 칸당 최대 블록 수
NSL = 32                                                                          # SC 칸 수 (fem_an_rt: SC = zeros(32))

DET_DECL = r'''
__device__ int AO_DET;
__device__ double* AO_SCB;
__device__ double* AO_PART;
__device__ int* AO_BMAX;
__device__ double* AO_CF;                                                         // 칸 x 27 x 3 (f64 힘)
__device__ float* AO_CB;                                                          // 칸 x 27 x 9 (노드 대각)
__device__ float* AO_CV;                                                          // 칸 x 27 x 3 (헤시안 곱)
__device__ float* AO_MW;                                                          // 점별 고착 질량
__device__ __forceinline__ void ao_bs_out(double* out, double v) {
    long long sl = out - AO_SCB;
    if (AO_DET && sl >= 0 && sl < 32) { AO_PART[sl*16384 + blockIdx.x] = v; atomicMax(&AO_BMAX[sl], (int)blockIdx.x + 1); }
    else atomicAdd(out, v);
}
'''

DET_SRC = r'''
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
'''


def _kslice(src, name):
    import re
    mt = re.search(r"__global__ void (?:__launch_bounds__\(\d+\) )?" + name + r"\(", src)
    i = mt.start()
    j = src.index("\n}\n", i) + 3
    return i, j


def _sub_in(src, name, old, new, count=1):
    i, j = _kslice(src, name)
    body = src[i:j]
    assert old in body, (name, old[:40])
    return src[:i] + body.replace(old, new, count) + src[j:]


def det_src(head):
    """절 AN SRC (닫는 괄호 앞까지) -> 결정적 분기를 넣은 판 (AO_DET = 0 이면 원래와 같음)."""
    s = head.replace('extern "C" {', 'extern "C" {\n' + DET_DECL, 1)
    s = s.replace("if (l == 0) atomicAdd(out, v); }", "if (l == 0) ao_bs_out(out, v); }", 1)
    s = _sub_in(s, "rt_elem", "if (on && head) atomicAdd(&f[ids[o]*3+a], s);",
                "if (on && head) { if (AO_DET) AO_CF[((size_t)(qq>>3)*27+o)*3+a] = s; else atomicAdd(&f[ids[o]*3+a], s); }")
    s = _sub_in(s, "rt_elem", "if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], m[k]);",
                "if (on && head) for (int k=0;k<9;k++) { if (AO_DET) AO_CB[((size_t)(qq>>3)*27+o)*9+k] = m[k]; else atomicAdd(&B[ids[o]*9+k], m[k]); }")
    s = _sub_in(s, "rt_mv", "if (on && head) atomicAdd(&out[ids[o]*3+a], s);",
                "if (on && head) { if (AO_DET) AO_CV[((size_t)(qq>>3)*27+o)*3+a] = s; else atomicAdd(&out[ids[o]*3+a], s); }")
    s = _sub_in(s, "rt_gauss", "atomicAdd(&mqn[nid], BW[oc*27+o]*mm);", "if (!AO_DET) atomicAdd(&mqn[nid], BW[oc*27+o]*mm);")
    s = _sub_in(s, "rt_floor_q", "if (!(s < 0.f)) return;", "if (!(s < 0.f)) { if (AO_DET) AO_MW[q] = 0.f; return; }")
    s = _sub_in(s, "rt_floor_q", "if (mw > 0.f) for (int o=0;o<27;o++) atomicAdd(&an[ids[o]], BW[oc*27+o]*mw);",
                "if (AO_DET) AO_MW[q] = mw; else if (mw > 0.f) for (int o=0;o<27;o++) atomicAdd(&an[ids[o]], BW[oc*27+o]*mw);")
    return s


def det_ao(ao_src):
    """절 AO 모래 커널 (fem_ao_rt.AO_SRC) 의 원자 합 분기."""
    s = _sub_in(ao_src, "ao_elem", "if (on && head) atomicAdd(&f[ids[o]*3+a], sv); } }",
                "if (on && head) { if (AO_DET) AO_CF[((size_t)(qq>>3)*27+o)*3+a] = sv; else atomicAdd(&f[ids[o]*3+a], sv); } } }")
    s = _sub_in(s, "ao_elem", "if (on && head) for (int k=0;k<9;k++) atomicAdd(&B[ids[o]*9+k], mm[k]); }",
                "if (on && head) for (int k=0;k<9;k++) { if (AO_DET) AO_CB[((size_t)(qq>>3)*27+o)*9+k] = mm[k]; else atomicAdd(&B[ids[o]*9+k], mm[k]); } }")
    return s


BS_KERNELS = {"rt_elem", "rt_grad", "rt_grad_c", "rt_ap", "rt_ap_c", "rt_cg1", "rt_cg1_c", "rt_lsy", "rt_lsy4", "rt_elem_ls", "ao_warm", "rt_fixdir"}


class DetMixin(ART.RTSolver):
    """결정적 합산 모드 (s.det = True). 하위 클래스의 모듈은 det_src·DET_SRC 를 포함해 컴파일해야 한다 (fem_ao_rt._src 가 그렇게 함)."""

    def det_alloc(s):
        Z = lambda *sh, dt=torch.float32: torch.zeros(*sh, device=s.dev, dtype=dt)
        s.d_part = Z(NSL * MAXB, dt=torch.float64)
        s.d_bmax = Z(NSL, dt=torch.int32)
        s.d_cf = Z(s.capQc * 27 * 3, dt=torch.float64)
        s.d_cb = Z(s.capQc * 27 * 9)
        s.d_cv = Z(s.capQc * 27 * 3)
        s.d_mw = Z(s.capQc * 8)
        s.d_st = torch.zeros(s.capQc + 1, device=s.dev, dtype=torch.int32)

    def det_bind(s):
        def setp(nm, t, dt):
            g = s.mod.get_global(nm)
            ART.cp.ndarray((1,), dtype=dt, memptr=g)[0] = int(t.data_ptr()) if t is not None else 0
        setp("AO_SCB", s.SC, ART.cp.uint64); setp("AO_PART", s.d_part, ART.cp.uint64); setp("AO_BMAX", s.d_bmax, ART.cp.uint64)
        setp("AO_CF", s.d_cf, ART.cp.uint64); setp("AO_CB", s.d_cb, ART.cp.uint64); setp("AO_CV", s.d_cv, ART.cp.uint64); setp("AO_MW", s.d_mw, ART.cp.uint64)
        import os
        setp("AO_ACTL", s.act if (getattr(s, "compact", False) and os.environ.get("AO_DET_ACTG", "1") == "1") else None, ART.cp.uint64)
        g = s.mod.get_global("AO_DET")
        ART.cp.ndarray((1,), dtype=ART.cp.int32, memptr=g)[0] = 1 if s.det else 0

    def det_sort(s):
        """입자를 칸 번호 (pcd) 로 안정 정렬, 입자 배열과 원래 순번 perm 을 같이 재배열 (그래프 안에서 torch 연산)."""
        o = torch.sort(s.pcd, stable=True).indices
        for nm in ("x", "v", "F", "m", "vol", "fx", "Np", "pcd", "perm") + (("trel",) if getattr(s, "trel", None) is not None else ()):
            t = getattr(s, nm)
            t.copy_(t[o])

    def k(s, name, n, args, b=128):
        if not getattr(s, "det", False):
            return super().k(name, n, args, b)
        i32 = np.int32
        IP = s.IP
        if name == "rt_p2g":                                                    # 정렬 + 칸 몫 + 노드 몫
            s.det_sort()
            s.zf(s.d_st)
            super().k("ao_seg", s.N + 1, (i32(s.N), P(IP), P(s.pcd), P(s.cidx), P(s.d_st)), 256)
            super().k("ao_p2g_q", s.capQc, (i32(s.N), P(s.F), P(s.m), P(s.vol), P(s.fx), P(s.d_st), P(IP), P(s.Aacc), P(s.Cacc), P(s.badq), P(s.badc)), 128)
            import os
            if os.environ.get("AO_DET_P2GC", "0") == "1":
                super().k("ao_p2g_c", s.capQc, (P(IP), P(s.d_st), P(s.fx), P(s.m), P(s.v), P(s.d_cb)), 64)
                super().k("ao_gath_mv", s.capN, (P(IP), P(s.cidx), P(s.d_cb), P(s.mn), P(s.mv)), 256)
            else:
                super().k("ao_p2g_n", s.capN, (P(IP), P(s.cidx), P(s.d_st), P(s.fx), P(s.m), P(s.v), P(s.mn), P(s.mv), P(None), P(None), i32(1), i32(1), np.float32(0), P(None)), 128)
            return
        if name == "ao_pbc_acc":                                                 # 구속 질량 (노드가 모음)
            # args: N, IP, Np, fx, m, trel, tbuf, K, S, dt, acc
            super().k("ao_p2g_n", s.capN, (P(IP), P(s.cidx), P(s.d_st), P(s.fx), P(s.m), P(None), P(None), P(None), P(s.trel), P(s.tbuf), args[7], args[8], args[9], args[10]), 128)
            return
        if name == "ao_act":
            super().k("ao_act_scan", 1024, (P(IP), P(s.mn), P(s.act)), 1024)
            return
        if name == "ao_act_reset":
            return
        super().k(name, n, args, b)
        if name in BS_KERNELS:
            super().k("ao_bsfin", NSL * 256, (), 256)
        if name in ("rt_elem", "ao_elem"):
            mode = int(args[8]) if name == "rt_elem" else int(args[9])
            fptr = args[10] if name == "rt_elem" else args[10]
            bptr = args[13] if name == "rt_elem" else args[13]
            if mode & 1:
                super().k("ao_gath_d3", s.capN, (P(IP), P(s.cidx), fptr), 256)
            if (mode & 2) and (mode & 8):
                super().k("ao_gath_f", s.capN, (P(IP), P(s.cidx), i32(9), P(s.d_cb), bptr), 256)
        elif name == "rt_mv":
            super().k("ao_gath_f", s.capN, (P(IP), P(s.cidx), i32(3), P(s.d_cv), args[7]), 256)
        elif name == "rt_gauss":
            super().k("ao_gath_q", s.capN, (P(IP), P(s.cidx), P(s.BW), P(s.mq), P(s.mqn)), 256)
        elif name == "rt_floor_q":
            super().k("ao_gath_q", s.capN, (P(IP), P(s.cidx), P(s.BW), P(s.d_mw), args[-1]), 256)
