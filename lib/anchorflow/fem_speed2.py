"""절 CM36 --sp2 (조정자 승인 11-08: 반올림 수준으로만 결과가 바뀌는 커널 재작성): 수박 실행기 (옛 실행기 fem_ao_rt.AORTSolver) 의 프로파일 상위 커널 재작성.

끄면 (--sp2 0) 아무것도 바꾸지 않는다 (이 모듈을 부르지 않음). 켜면 install(bits) 가 fem_ao_rt._src (생성 소스 끝에 글자 바꾸기·커널 덧붙임) 와
AORTSolver.k (커널 실행 가로채기) 를 감싼다 -- fem_ao_rt.py·fem_an_rt.py 의 글자는 그대로 (새 실행기 = 레고·ficus·wolf·bread 경로는 이 모듈을 안 씀).

비트:
  1  rt_scan (한 블록 1024 스레드 칸 스캔) -> 새 실행기 절 CM3 --ncomp 의 격자 3 단계 스캔 (sp_scan_a/b/c, 같은 정수 출력 = 비트 같음).
  2  blocksum_add 의 블록 합이 정확히 0 이면 원자 합을 안 함 (+0.0 더하기는 값 그대로) + rt_bar_y0e 는 블록 안에 장벽 띠 (d < d̂) 에 들 수 있는
     표본점이 없으면 60 개 블록 합을 건너뜀 (그 블록 몫은 모두 정확히 0) -- 결과 같음 (부호 있는 0 만 예외).
  4  rt_p2g 의 칸 누적 C[0..28] (가우스 점 누적 A 와 같은 값) 과 gq_p2q 의 칸 운동량 PC 를 원자 합 대신 팔분 8 점 합으로 (sp_csum, rt_gauss 앞) -- 합 순서만 바뀜 (반올림 수준).
  8  gq_q2n (점마다 27 노드 x 7 원자 합) -> 칸 하나 = 32 스레드 (노드 o 마다 한 스레드) 가 8 점을 먼저 모은 뒤 원자 합 (8 배 적은 원자 합) -- 합 순서만 바뀜.
  16 aw_inv_q (뒤집힘 필터, f64 삼차식) 앞에 f32 증명 빠른 길: 계수 c0..c3 를 f32 로 셈해 c0 + Σ min(0, c_i) > 1e-3 m³ (m = X·D 성분 최대 절댓값;
     f32 계수 오차 상한 약 1e-4 m³ 의 10 배) 이면 [0, 1] 에서 det > 0 이 확실해 f64 경로도 k = 0·부작용 없음 -> 바로 돌아감; 아니면 f64 경로 그대로 (같은 k).
"""
import numpy as np
import torch

from . import fem_an_rt_v0 as ART

P = ART.P
B_SCAN, B_BSUM, B_P2G, B_Q2N, B_INV = 1, 2, 4, 8, 16

SCAN_SRC = r'''
// ---- 절 CM36 --sp2 비트 1: rt_scan 과 같은 출력 (fem_an_rt rt_scan_a/b/c 글자 그대로, 이름만 sp_) ----
__global__ void sp_scan_a(const int* IP, const int* occ, int* bc) {@SK
    int n = IP[13] ? 0 : IP[10];
    if (blockIdx.x*blockDim.x >= n) return;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int c = __syncthreads_count(i < n && occ[i]);
    if (threadIdx.x == 0) bc[blockIdx.x] = c;
}
__global__ void sp_scan_b(int* IP, const int* bc, int* bo, int capQc) {@SK
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
__global__ void sp_scan_c(const int* IP, const int* occ, const int* bo, int* cidx, int* cb, int capQc) {@SK
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
'''

CSUM_SRC = r'''
// ---- 절 CM36 --sp2 비트 4: 칸 누적 C[c*30 + 0..28] = 팔분 8 점 A[q*28 + 0..27] 합 (28 = 부피 = 27 과 같은 합), PC[c*6 + a] = Σ PQ[q*6 + a] ----
// 칸 하나 = 32 스레드 (j = 성분); C[29] (질량) 은 rt_p2g 원자 합 그대로.
__global__ void sp_csum(const int* IP, const float* A, float* C, const float* PQ, float* PC) {@SK
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int c = t >> 5, j = t & 31;
    if (c >= IP[11]) return;
    if (j < 29) { int jj = (j == 28) ? 27 : j; float s = 0.f;
        for (int oc=0; oc<8; oc++) s += A[(size_t)(c*8 + oc)*28 + jj];
        C[(size_t)c*30 + j] = s; }
    if (PQ && j < 6) { float s = 0.f;
        for (int oc=0; oc<8; oc++) s += PQ[(size_t)(c*8 + oc)*6 + j];
        PC[(size_t)c*6 + j] = s; }
}
'''

Q2N_SRC = r'''
// ---- 절 CM36 --sp2 비트 8: gq_q2n 과 같은 식 (점 -> 노드), 칸 하나 = 32 스레드 (o < 27 = 노드), 칸의 8 점이 같은 27 노드를 쓰므로 8 점을 먼저 모은 뒤 원자 합 ----
__global__ void sp_q2n(const int* IP, const int* cb, const float* BW, const float* mq, const float* PQ, const float* PC, float kap, int oct, float* mn, float* mv, float* ma) {@SK
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int c = t >> 5, o = t & 31;
    if (c >= IP[11] || o >= 27) return;
    int na = ma ? 6 : 3;
    float sm = 0.f, sp[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int oc=0; oc<8; oc++) {
        int q = c*8 + oc; float w = BW[oc*27+o];
        float pq[6]; gq_pq(q, PQ, PC, kap, oct, na, pq);
        sm += w*mq[q];
        for (int a=0;a<na;a++) sp[a] += w*pq[a];
    }
    int nb1 = IP[7], nb2 = IP[8];
    int i = o/9, j = (o/3)%3, k = o%3;
    int nid = ((cb[c*3]+i)*nb1 + (cb[c*3+1]+j))*nb2 + (cb[c*3+2]+k);
    atomicAdd(&mn[nid], sm);
    for (int a=0;a<3;a++) atomicAdd(&mv[nid*3+a], sp[a]);
    if (ma) for (int a=0;a<3;a++) atomicAdd(&ma[nid*3+a], sp[3+a]);
}
'''

CERT_SRC = r'''
// ---- 절 CM36 --sp2 비트 16: 뒤집힘 필터 f32 증명 -- det(X + t D) = c0 + c1 t + c2 t² + c3 t³ 가 [0, 1] 에서 확실히 양수인가 ----
// 하한 c0 + Σ min(0, c_i) (t^i ∈ [0, 1]) 이 1e-3 m³ (m = X·D 성분 최대 절댓값) 보다 크면 참. f32 계수 오차 (계수마다 삼중곱 ≤ 18 개, 각 ≤ m³) 는 약 1e-4 m³ 이하.
// 참이면 f64 경로 (aw_pos(c, 1)) 도 참: p(1) > 0 이고 (0, 1) 안 극점 값 ≥ 하한 > 0. 비유한이면 거짓 (f64 경로로).
__device__ __forceinline__ bool sp_cert(const float* X, const float* D) {
    float C0[9], CD[9];
    C0[0] = X[4]*X[8]-X[5]*X[7]; C0[1] = X[5]*X[6]-X[3]*X[8]; C0[2] = X[3]*X[7]-X[4]*X[6];
    C0[3] = X[2]*X[7]-X[1]*X[8]; C0[4] = X[0]*X[8]-X[2]*X[6]; C0[5] = X[1]*X[6]-X[0]*X[7];
    C0[6] = X[1]*X[5]-X[2]*X[4]; C0[7] = X[2]*X[3]-X[0]*X[5]; C0[8] = X[0]*X[4]-X[1]*X[3];
    CD[0] = D[4]*D[8]-D[5]*D[7]; CD[1] = D[5]*D[6]-D[3]*D[8]; CD[2] = D[3]*D[7]-D[4]*D[6];
    CD[3] = D[2]*D[7]-D[1]*D[8]; CD[4] = D[0]*D[8]-D[2]*D[6]; CD[5] = D[1]*D[6]-D[0]*D[7];
    CD[6] = D[1]*D[5]-D[2]*D[4]; CD[7] = D[2]*D[3]-D[0]*D[5]; CD[8] = D[0]*D[4]-D[1]*D[3];
    float c0 = X[0]*C0[0] + X[1]*C0[1] + X[2]*C0[2], c3 = D[0]*CD[0] + D[1]*CD[1] + D[2]*CD[2], c1 = 0.f, c2 = 0.f, m = 0.f;
    for (int k=0;k<9;k++) { c1 += C0[k]*D[k]; c2 += CD[k]*X[k]; m = fmaxf(m, fmaxf(fabsf(X[k]), fabsf(D[k]))); }
    float lb = c0 + fminf(c1, 0.f) + fminf(c2, 0.f) + fminf(c3, 0.f);
    return isfinite(lb) && isfinite(m) && lb > 1e-3f*m*m*m;
}
'''

INV_OLD = "    double X0[9], Dx[9]; for (int k=0;k<9;k++) { X0[k] = (double)X0f[k]; Dx[k] = (double)Dxf[k]; }\n"
INV_NEW = ("    if (sp_cert(X0f, Dxf)) {                                                   // 절 CM36 --sp2 비트 16: f32 증명이면 f64 경로도 k = 0·부작용 없음\n"
           "        bool ok_ = true;\n"
           "        if (Ue) { float A_[9] = {0,0,0,0,0,0,0,0,0}, Xe_[9], De_[9];\n"
           "            for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; int id = ids[o]; for (int a=0;a<3;a++) { float va = e1*yb[id*3+a] + e0*Ue[id*3+a]; A_[a*3] += va*bg[0]; A_[a*3+1] += va*bg[1]; A_[a*3+2] += va*bg[2]; } }\n"
           "            for (int a=0;a<3;a++) for (int b=0;b<3;b++) { Xe_[a*3+b] = F[a*3+b] + A_[a*3]*F[b] + A_[a*3+1]*F[3+b] + A_[a*3+2]*F[6+b]; De_[a*3+b] = e1*Dxf[a*3+b]; }\n"
           "            ok_ = sp_cert(Xe_, De_); }\n"
           "        if (ok_) return;\n"
           "    }\n" + INV_OLD)

P2G_OLD = ("    for (int k=0;k<9;k++) { atomicAdd(&aq[k], w*R[k]); atomicAdd(&aq[9+k], w*L[k]); atomicAdd(&aq[18+k], w*Fp[k]);\n"
           "                            atomicAdd(&ac[k], w*R[k]); atomicAdd(&ac[9+k], w*L[k]); atomicAdd(&ac[18+k], w*Fp[k]); }\n"
           "    atomicAdd(&aq[27], w); atomicAdd(&ac[27], w); atomicAdd(&ac[28], w); atomicAdd(&ac[29], m[p]);\n")
P2G_NEW = ("    for (int k=0;k<9;k++) { atomicAdd(&aq[k], w*R[k]); atomicAdd(&aq[9+k], w*L[k]); atomicAdd(&aq[18+k], w*Fp[k]); }\n"
           "    atomicAdd(&aq[27], w); atomicAdd(&ac[29], m[p]);                    // 절 CM36 --sp2 비트 4: 칸 C[0..28] 은 sp_csum (팔분 합)\n")
P2Q_OLD = ("    for (int a=0;a<3;a++) { float t = mp*v[p*3+a]; atomicAdd(&PQ[(size_t)q*6+a], t); atomicAdd(&PC[(size_t)c*6+a], t); }\n"
           "    if (pa) for (int a=0;a<3;a++) { float t = mp*pa[p*3+a]; atomicAdd(&PQ[(size_t)q*6+3+a], t); atomicAdd(&PC[(size_t)c*6+3+a], t); }\n")
P2Q_NEW = ("    for (int a=0;a<3;a++) { float t = mp*v[p*3+a]; atomicAdd(&PQ[(size_t)q*6+a], t); }\n"
           "    if (pa) for (int a=0;a<3;a++) { float t = mp*pa[p*3+a]; atomicAdd(&PQ[(size_t)q*6+3+a], t); }   // 절 CM36 --sp2 비트 4: PC 는 sp_csum\n")
BS_OLD = "    else atomicAdd(out, v);\n}"
BS_NEW = "    else if (v != 0.0) atomicAdd(out, v);                                  // 절 CM36 --sp2 비트 2: +0.0 더하기 생략 (값 그대로)\n}"
Y0E_OLD = "        kap = (double)ks*(double)Vq[qq];\n    }\n    double e = 0.0;\n"
Y0E_NEW = ("        kap = (double)ks*(double)Vq[qq];\n    }\n"
           "    if (!__syncthreads_or(act && fminf(d0, d1) < dh + 1e-5f*(fabsf(d0) + fabsf(d1)))) return;   // 절 CM36 --sp2 비트 2: 띠 밖 블록은 모든 몫이 정확히 0 (d 는 d0·d1 사이 볼록 결합)\n"
           "    double e = 0.0;\n")


def _kernel_span(r, name):
    import re
    m = re.search(r"__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?" + name + r"\s*\(", r)
    assert m, name
    i = m.start()
    j = r.find("\n__global__", m.end())
    return i, (len(r) if j < 0 else j)


def _sub_in(r, name, old, new):
    i, j = _kernel_span(r, name)
    body = r[i:j]
    assert body.count(old) == 1, (name, old[:60])
    return r[:i] + body.replace(old, new, 1) + r[j:]


def apply(r, bits):
    """생성 소스 r (fem_ao_rt._src 의 출력) 에 비트별 바꾸기·덧붙이기."""
    bits = int(bits)
    if not bits:
        return r
    sk = " if (AF_SKIP) return;" if "__device__ int AF_SKIP;" in r else ""
    add = ""
    if bits & B_SCAN:
        add += SCAN_SRC
    if bits & B_BSUM:
        i = r.index("__device__ __forceinline__ void ao_bs_out(")
        j = r.index("\n}", i) + 2
        assert r[i:j].count(BS_OLD) == 1
        r = r[:i] + r[i:j].replace(BS_OLD, BS_NEW, 1) + r[j:]
        if "rt_bar_y0e(" in r:
            r = _sub_in(r, "rt_bar_y0e", Y0E_OLD, Y0E_NEW)
    if bits & B_P2G:
        r = _sub_in(r, "rt_p2g", P2G_OLD, P2G_NEW)
        if "void gq_p2q(" in r:
            r = _sub_in(r, "gq_p2q", P2Q_OLD, P2Q_NEW)
        add += CSUM_SRC
    if bits & B_Q2N:
        assert "void gq_q2n(" in r, "--sp2 비트 8 은 --gq_mass 판만"
        add += Q2N_SRC
    if bits & B_INV:
        r = _sub_in(r, "aw_inv_q", INV_OLD, INV_NEW)
        i = r.index("__global__ void aw_inv_q(")
        r = r[:i] + CERT_SRC.lstrip("\n") + r[i:]
    if add:
        i = r.rstrip().rfind("}")                                                  # extern "C" { ... } 닫는 괄호 앞
        r = r[:i] + add.replace("@SK", sk) + "\n" + r[i:]
    return r


def _ibuf(s, nm, n):
    t = getattr(s, nm, None)
    if t is None or t.numel() < n:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("--sp2: 버퍼는 데움 record 에서 잡아야 함")
        t = torch.zeros(n, device=s.dev, dtype=torch.int32)
        setattr(s, nm, t)
    return t


def install(bits):
    """fem_ao_rt._src 와 AORTSolver.k 를 감쌈 (bits 0 이면 아무것도 안 함)."""
    bits = int(bits)
    if not bits:
        return
    from . import fem_ao_rt as AORT
    if getattr(AORT, "_sp2_bits", None) is not None:
        assert AORT._sp2_bits == bits
        return
    AORT._sp2_bits = bits
    o_src = AORT._src

    def _src(*a, **kw):
        return apply(o_src(*a, **kw), bits)
    AORT._src = _src
    ok = AORT.AORTSolver.k
    i32 = np.int32

    def k(s, name, n, args, b=128):
        assert not getattr(s, "det", False), "--sp2 는 비결정 판만"
        if (bits & B_SCAN) and name == "rt_scan":
            nb = (int(s.capC) + 255) // 256 + 1
            scb, sco = _ibuf(s, "_sp_scb", nb), _ibuf(s, "_sp_sco", nb)
            ok(s, "sp_scan_a", s.capC, (P(s.IP), P(s.occ), P(scb)), 256)
            ok(s, "sp_scan_b", 1024, (P(s.IP), P(scb), P(sco), i32(s.capQc)), 1024)
            ok(s, "sp_scan_c", s.capC, (P(s.IP), P(s.occ), P(sco), P(s.cidx), P(s.cb), i32(s.capQc)), 256)
            return
        if (bits & B_P2G) and name == "rt_gauss":
            gq = getattr(s, "gqm", False)
            ok(s, "sp_csum", s.capQc * 32, (P(s.IP), P(s.Aacc), P(s.Cacc), P(s.gqP if gq else None), P(s.gqC if gq else None)), 128)
            return ok(s, name, n, args, b)
        if (bits & B_Q2N) and name == "gq_q2n":
            return ok(s, "sp_q2n", s.capQc * 32, args, 128)
        return ok(s, name, n, args, b)
    AORT.AORTSolver.k = k
