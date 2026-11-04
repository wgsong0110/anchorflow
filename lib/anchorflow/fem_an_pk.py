"""절 AN.18 지속형 커널 판 (`--pk`): 프레임 하나 = 커널 호출 한 번.
RTSolver.record() 가 내는 커널 순서 (그래프 판과 같은 것) 를 명령 목록으로 모으고 (RTSolver.k 의 모으기 모드), 지속형 커널 pk_run 하나가 그 목록을 차례로 돈다.
- 소스 변환 (pk_src): 모듈 소스 (결정적 판이면 det_src 를 거친 것) 의 모든 __global__ 커널을 __device__ 함수 pkd_<이름>(VB_, VT_, VD_, VS_, 원래 인자) 로 바꾼다.
  몸 안의 blockIdx.x -> VB_ (가상 블록 번호 = 원래 실행의 블록 번호), threadIdx.x -> VT_, blockDim.x -> VD_ (원래 블록 크기), __syncthreads() -> 가상 블록 이름 붙은 장벽 (bar.sync 1+VS_, VD_).
  물리 블록 (PKB 스레드) 하나가 원래 블록 크기 VD_ 의 가상 블록 PKB/VD_ 개를 동시에 (VS_ = 그 안 번호), 격자 단위 반복으로 nvb 개 모두를 돈다.
  블록 합 blocksum_add 는 가상 블록마다 같은 순서 (워프 합 -> 워프 0 이 합) 로: 원래 블록 번호·블록 크기가 그대로라 결정적 판의 블록 부분합 AO_PART[칸][블록] 도 같은 자리·같은 값.
- 명령 사이: 격자 장벽 (cooperative_groups grid.sync 와 같은 식: 블록 대표가 원자 더하기 하나로 최상위 비트를 뒤집고 기다림). 기본은 내보내기 원자 더하기 + 받기 읽기 (-DPK_FENCE 면 grid.sync 처럼 앞뒤 __threadfence; 결정적 판 비트 대조 둘 다 통과, 내보내기·받기가 3 % 빠름). 협동 실행 (cudaLaunchCooperativeKernel) 이라 모든 블록이 동시 상주.
  블록 수 = cudaOccupancyMaxActiveBlocksPerMultiprocessor x SM 수.
- 작은 명령 (가상 블록 1 개: 스칼라 커널 rt_ctl·rt_sc_* 등) 과 MG 작은 층 (층 >= pk_loc) 명령은 블록 0 하나가 차례로 돌고 사이는 블록 장벽만; 이어진 묶음 끝에 격자 장벽 하나.
- 허용치 조기 종료 (절 AN.14): rt_ctl 이 켜는 AF_SKIP 를 커널 안 분기로 -- 건너뛰는 명령은 장벽도 없이 넘어감 (펼친 상한 목록을 비용 거의 0 으로 건너뜀).
  모든 블록이 같은 판정을 하도록, 블록 0 대표가 장벽 도착 직전 AF_SKIP 를 장벽 번호 홀짝 칸 PK_SNAP 에 적고 모든 블록이 장벽을 지난 직후 그 칸을 읽는다.
- 그래프 판의 torch 연산 (복사·0 채우기·결정적 판 입자 정렬 torch.sort) 은 같은 일을 하는 장치 명령으로 (pk_copy32, pk_zero_b, pk_sort_*: 칸별 개수 -> 스캔 -> 놓기 -> 칸 안 원래 번호 오름차순 = 안정 정렬과 같은 순서).
  칸 압축 스캔 (rt_scan, 한 블록 1024 스레드) 은 같은 출력의 격자 3 단계 스캔 (pk_scan_a/b/c) 으로.
"""
import re
from collections import namedtuple

PKSrc = namedtuple("PKSrc", "code meta")

EXCLUDE = ("rt_scan", "rt_mg_csolve", "ao_act_scan")                    # 블록 크기 1024 가정 커널: pk 판에서는 다른 명령으로 바꾸거나 (rt_scan) 쓰지 않음
EXEMPT = ("rt_reset", "rt_ctl", "pk_copy32", "pk_zero_b")                # AF_SKIP 와 무관하게 늘 도는 명령 (그래프 판에서 건너뛰기 줄이 없던 것: 제어 커널과 torch 연산)

PK_HEAD = r'''
#ifndef PKB
#define PKB 256
#endif
#ifndef PKMINB
#define PKMINB 1
#endif
#ifdef PK_INLINE
#define PK_FNATTR __forceinline__
#else
#define PK_FNATTR __noinline__
#endif
#ifndef PK_EXATTR
#define PK_EXATTR
#endif
#define PK_VSYNC() asm volatile("bar.sync %0, %1;" :: "r"(VS_ + 1), "r"(VD_) : "memory")
#ifdef AF_ADAPT
#define PK_SKIPV (*(volatile int*)&AF_SKIP)
#else
#define PK_SKIPV 0
#endif
__device__ __forceinline__ void pk_bs_out(double* out, double v, int VB_) {
#ifdef PK_DET
    long long sl = out - AO_SCB;
    if (AO_DET && sl >= 0 && sl < 64) { AO_PART[sl*65536 + VB_] = v; atomicMax(&AO_BMAX[sl], VB_ + 1); return; }
    long long sm = AO_SCM ? (out - AO_SCM) : -1;
    if (AO_DET && AO_SCM && sm >= 0 && sm < 32) { AO_PART[(64+sm)*65536 + VB_] = v; atomicMax(&AO_BMAX[64+sm], VB_ + 1); return; }
#endif
    atomicAdd(out, v);
}
// 가상 블록 합 (blocksum_add 와 같은 순서: 워프 합 -> 워프 0 이 워프 합들을 다시 워프 합)
__device__ void pk_bsum(double v, double* out, int VB_, int VT_, int VD_, int VS_) {
    __shared__ double pk_sh[PKB/32];
    v = warpsum(v);
    int l = VT_ & 31, w = VT_ >> 5, base = VS_*(VD_ >> 5);
    if (l == 0) pk_sh[base + w] = v;
    PK_VSYNC();
    if (w == 0) { v = (l < (VD_ >> 5)) ? pk_sh[base + l] : 0.0; v = warpsum(v); if (l == 0) pk_bs_out(out, v, VB_); }
    PK_VSYNC();
}
#define blocksum_add(v, out) pk_bsum(v, out, VB_, VT_, VD_, VS_)
// 가상 블록 배타 스캔 (정수): 돌려줌 = 앞 스레드 합, *tot = 블록 합
__device__ int pk_bscan(int v, int* tot, int VT_, int VD_, int VS_) {
    __shared__ int pk_ws[PKB/32];
    int l = VT_ & 31, w = VT_ >> 5, nw = VD_ >> 5, base = VS_*nw;
    int x = v;
    for (int o=1;o<32;o<<=1) { int y = __shfl_up_sync(0xffffffffu, x, o); if (l >= o) x += y; }
    if (l == 31) pk_ws[base + w] = x;
    PK_VSYNC();
    int off = 0, t = 0;
    for (int k=0;k<nw;k++) { int s_ = pk_ws[base + k]; if (k < w) off += s_; t += s_; }
    PK_VSYNC();
    *tot = t; return off + x - v;
}
#define PK_BSCAN(v, t) pk_bscan(v, t, VT_, VD_, VS_)
'''

PK_EXTRA = r'''
__global__ void pk_zero_b(int n, unsigned char* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = 0; }
__global__ void pk_copy32(int n, unsigned int* d, const unsigned int* s) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) d[i] = s[i]; }
// 칸 압축 (rt_scan 과 같은 출력): a = 256 칸 덩어리 합, b = 덩어리 합 배타 스캔 + nc·nq·넘침 (한 가상 블록), c = 덩어리 안 스캔 + 덩어리 앞 합 -> cidx·cb
__global__ void pk_scan_a(const int* IP, const int* occ, int* bs) {
    int n = IP[13] ? 0 : IP[10];
    if (blockIdx.x*blockDim.x >= n) return;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int v = (i < n) ? occ[i] : 0; int tot;
    PK_BSCAN(v, &tot);
    if (threadIdx.x == 0) bs[blockIdx.x] = tot;
}
__global__ void pk_scan_b(int* IP, int* bs, int capQc, int capB) {
    int n = IP[13] ? 0 : IP[10];
    int nb = (n + 255) / 256;
    int carry = 0;
    for (int base = 0; base < nb; base += blockDim.x) {
        int i = base + threadIdx.x; int v = (i < nb) ? bs[i] : 0; int tot;
        int ex = PK_BSCAN(v, &tot);
        if (i < nb) bs[i] = carry + ex;
        carry += tot;
    }
    if (threadIdx.x == 0) { bs[capB] = n; IP[11] = carry; IP[12] = carry*8; if (carry > capQc) { IP[13] = 1; IP[11] = 0; IP[12] = 0; } }
}
__global__ void pk_scan_c(const int* IP, const int* occ, const int* bs, int* cidx, int* cb, int capQc, int capB) {
    int n = bs[capB];
    if (blockIdx.x*blockDim.x >= n) return;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int v = (i < n) ? occ[i] : 0; int tot;
    int ex = PK_BSCAN(v, &tot);
    int n1 = IP[7]-2, n2 = IP[8]-2;
    if (i < n) {
        int id = bs[blockIdx.x] + ex;
        if (v) { if (id < capQc) { cidx[i] = id; cb[id*3] = i / (n1*n2); cb[id*3+1] = (i / n2) % n1; cb[id*3+2] = i % n2; } }
        else cidx[i] = -1;
    }
}
// 결정적 판 입자 정렬 (torch.sort(pcd, stable=True) 와 같은 순서): 칸별 개수 -> 배타 스캔 st -> 놓기 (원자 자리) -> 칸 안 원래 번호 오름차순 -> 배열 모으기
__global__ void pk_sort_cnt(int N, const int* IP, const int* pcd, const int* cidx, int* cnt) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    atomicAdd(&cnt[cidx[pcd[p]]], 1);
}
__global__ void pk_sort_scan(int N, const int* IP, const int* cnt, int* st, int* cur) {
    int nc = IP[13] ? 0 : IP[11];
    int carry = 0;
    for (int base = 0; base < nc; base += blockDim.x) {
        int i = base + threadIdx.x; int v = (i < nc) ? cnt[i] : 0; int tot;
        int ex = PK_BSCAN(v, &tot);
        if (i < nc) { st[i] = carry + ex; cur[i] = 0; }
        carry += tot;
    }
    if (threadIdx.x == 0 && nc > 0) st[nc] = N;
}
__global__ void pk_sort_put(int N, const int* IP, const int* pcd, const int* cidx, const int* st, int* cur, int* idx) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    idx[st[c] + atomicAdd(&cur[c], 1)] = p;
}
__global__ void pk_sort_cell(const int* IP, const int* st, int* idx) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int a = st[c], b = st[c+1];
    for (int i = a+1; i < b; i++) { int k = idx[i]; int j = i-1; while (j >= a && idx[j] > k) { idx[j+1] = idx[j]; j--; } idx[j+1] = k; }
}
__global__ void pk_gather(int N, const int* IP, const int* idx, int nw, const unsigned int* src, unsigned int* dst) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N*nw) return;
    int p = i / nw, w = i - p*nw;
    int q = IP[13] ? p : idx[p];
    dst[i] = src[(size_t)q*nw + w];
}
'''

PK_RUN = r'''
__device__ unsigned int PK_BAR;
__device__ int PK_SNAP[2];
__device__ unsigned long long* PK_TP;                                             // 시간 기록 (null = 끔): [장벽 번호] = (전역 시계 ns, 명령 번호)
__device__ __forceinline__ void pk_tstamp(int e, int c) {
    if (PK_TP && blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); PK_TP[2*e] = t; PK_TP[2*e+1] = (unsigned long long)c; }
}
// 격자 장벽 (cooperative_groups grid.sync 와 같은 식) + 건너뛰기 판정 공유 (블록 0 대표가 도착 직전 AF_SKIP 를 홀짝 칸에, 모두 지난 직후 읽음)
__device__ __forceinline__ void pk_gsync(int* ep, int* skip, int c) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) PK_SNAP[*ep & 1] = PK_SKIPV;
        unsigned int nb = (blockIdx.x == 0) ? (0x80000000u - (gridDim.x - 1u)) : 1u;
#ifndef PK_FENCE
        unsigned int old, cur;                                                     // 내보내기·받기 의미 원자 연산 (전체 울타리 대신)
        asm volatile("atom.add.release.gpu.u32 %0, [%1], %2;" : "=r"(old) : "l"(&PK_BAR), "r"(nb) : "memory");
        do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(cur) : "l"(&PK_BAR) : "memory"); } while (((old ^ cur) & 0x80000000u) == 0u);
#else
        __threadfence();
        unsigned int old = atomicAdd(&PK_BAR, nb);
        while (((old ^ *(volatile unsigned int*)&PK_BAR) & 0x80000000u) == 0u) { }
        __threadfence();
#endif
    }
    __syncthreads();
    *skip = *(volatile int*)&PK_SNAP[*ep & 1];
    pk_tstamp(*ep + 1, c);
    *ep += 1;
}
// 명령 하나: CMD[8] = 커널 번호, 가상 블록 수, 가상 블록 크기, 표시 (1 블록 0 전용 묶음, 2 건너뛰기 무관, 4 묶음에 무관 명령 있음), 인자 위치, 묶음 끝, -, -
__device__ __forceinline__ void pk_cmd(const int* cm, const unsigned long long* A0, int local) {
    int kid = cm[0], nvb = cm[1], vbd = cm[2];
    const unsigned long long* A = A0 + cm[4];
    if (vbd == 1) { if (blockIdx.x == 0 && threadIdx.x == 0) pk_exec(kid, A, 0, 1, nvb, 0, 1, 0); return; }
    int nsub = PKB / vbd; int vs = threadIdx.x / vbd; int vt = threadIdx.x - vs*vbd;
    if (vs >= nsub) return;
    // 격자 명령: 가상 블록을 블록 (= SM) 먼저 퍼뜨림 (vb = vs x 격자 + 블록) -- 작은 명령이 SM 하나에 몰리지 않게. 같은 vb 는 늘 같은 스레드 (이음 생략 조건)
    int start = local ? vs : (vs*(int)gridDim.x + (int)blockIdx.x), stride = (local ? 1 : (int)gridDim.x)*nsub;
    if (start < nvb) pk_exec(kid, A, start, stride, nvb, vt, vbd, vs);
}
__global__ void __launch_bounds__(PKB, PKMINB) pk_run(const int* CMD, int ncmd, const unsigned long long* ARGS) {
    int ep = 0, skip = 0;
    pk_tstamp(0, -1);
    for (int c = 0; c < ncmd; ) {
        const int* cm = CMD + c*8;
        int fl = cm[3];
        if (fl & 1) {
            int re = cm[5];
            if (skip && !(fl & 4)) { c = re + 1; continue; }
            if (blockIdx.x == 0) {
                int sk = skip;
                for (int j = c; j <= re; j++) {
                    const int* cj = CMD + j*8;
                    if (!sk || (cj[3] & 2)) { pk_cmd(cj, ARGS, 1); __syncthreads(); sk = PK_SKIPV; }
                }
            }
            pk_gsync(&ep, &skip, c);
            c = re + 1;
        } else {
            if (!skip || (fl & 2)) { pk_cmd(cm, ARGS, 0); if (!(fl & 8)) pk_gsync(&ep, &skip, c); }   // 8: 다음 명령과 같은 자리끼리만 이어져 장벽 생략
            c++;
        }
    }
}
'''


def _match_paren(s, i, o="(", c=")"):
    """s[i] == o 인 자리에서 짝 닫는 자리 (// 줄 주석 건너뜀)."""
    d = 0
    j = i
    n = len(s)
    while j < n:
        ch = s[j]
        if ch == "/" and j + 1 < n and s[j + 1] == "/":
            j = s.index("\n", j)
            continue
        if ch == o:
            d += 1
        elif ch == c:
            d -= 1
            if d == 0:
                return j
        j += 1
    raise ValueError("짝 없음")


def _kind(t):
    t = t.strip()
    if "*" in t:
        return "p"
    base = t.replace("const", "").strip()
    return {"int": "i", "float": "f", "double": "d"}[base]


def _cast(t, k, i):
    if k == "p":
        return "(%s)A[%d]" % (t.strip(), i)
    if k == "i":
        return "(int)(unsigned int)A[%d]" % i
    if k == "f":
        return "__uint_as_float((unsigned int)A[%d])" % i
    return "__longlong_as_double((long long)A[%d])" % i


def pk_src(src, adapt, goct, det):
    """모듈 소스 (extern "C" { ... } 꼴, 마지막 } 가 extern 닫기) -> 지속형 판 소스와 커널 표 {이름: (번호, [인자 종류])}."""
    s = src
    if goct is not None and goct >= 0:
        s = s.replace('extern "C" {', 'extern "C" {\n#define AF_OCT 1\n__device__ float* AF_MO;\n__device__ float AF_GOCT;', 1)
    if adapt:
        s = s.replace('extern "C" {', 'extern "C" {\n__device__ int AF_SKIP;\n#define AF_ADAPT 1', 1)
    if det:
        s = s.replace('extern "C" {', 'extern "C" {\n#define PK_DET 1', 1)
    # 블록 합 원래 정의는 이름만 바꿔 남기고 (안 씀), 바로 뒤에 지속형 머리 (가상 블록 합 매크로 등)
    old = "__device__ void blocksum_add(double v, double* out) {"
    assert old in s
    i = s.index(old)
    j = _match_paren(s, s.index("{", i), "{", "}")
    s = s[:i] + old.replace("blocksum_add", "blocksum_add_orig_") + s[i + len(old):j + 1] + "\n" + PK_HEAD + s[j + 1:]
    k = s.rstrip().rfind("}")
    s = s[:k] + PK_EXTRA + "\n" + s[k:]
    out, pos, meta, cases = [], 0, {}, []
    pat = re.compile(r"__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?(\w+)\s*\(")
    for m in pat.finditer(s):
        if m.start() < pos:
            continue
        name = m.group(1)
        pe = _match_paren(s, m.end() - 1)
        params = s[m.end():pe]
        b0 = s.index("{", pe)
        b1 = _match_paren(s, b0, "{", "}")
        out.append(s[pos:m.start()])
        pos = b1 + 1
        if name in EXCLUDE:
            continue
        body = s[b0:b1 + 1]
        for a_, b_ in (("blockIdx.x", "VB_"), ("threadIdx.x", "VT_"), ("blockDim.x", "VD_"), ("__syncthreads()", "PK_VSYNC()")):
            body = body.replace(a_, b_)
        assert "gridDim" not in body, name
        if "__shared__" in body:
            assert name == "ao_bsfin", name                                       # 블록 크기 256 고정 커널 하나: 가상 블록마다 공유 칸
            body = body.replace("__shared__ double sh[256];", "__shared__ double pk_bf[(PKB >= 256 ? PKB/256 : 1)][256]; double* sh = pk_bf[VS_];")
            assert "__shared__ double pk_bf" in body
        plist = [q.strip() for q in params.split(",") if q.strip()]
        kinds, decl = [], []
        for q in plist:
            mm = re.match(r"(.*?)(\w+)\s*$", q, re.S)
            kinds.append(_kind(mm.group(1)))
            decl.append((mm.group(1), mm.group(2)))
        kid = len(meta)
        meta[name] = (kid, kinds)
        hdr = "__device__ PK_FNATTR void pkd_%s(const int VB_, const int VT_, const int VD_, const int VS_%s)" % (name, "".join(", " + q for q in plist))
        out.append(hdr + " " + body)
        ld = "".join("        %s a%d = %s;\n" % (t.strip(), ii, _cast(t, kd, ii)) for ii, ((t, _), kd) in enumerate(zip(decl, kinds)))
        call = "pkd_%s(vb, VT_, VD_, VS_%s)" % (name, "".join(", a%d" % ii for ii in range(len(decl))))
        guard = name == "rt_ctl"                                                     # 소스에서 #ifdef AF_ADAPT 안
        cases.append(("#ifdef AF_ADAPT\n" if guard else "") + "    case %d: {\n%s        for (int vb = v0; vb < nvb; vb += vst) %s;\n    } break;\n" % (kid, ld, call) + ("#endif\n" if guard else ""))
    out.append(s[pos:])
    s = "".join(out)
    disp = ("__device__ PK_EXATTR void pk_exec(int kid, const unsigned long long* A, int v0, int vst, int nvb, int VT_, int VD_, int VS_) {\n    switch (kid) {\n" + "".join(cases) + "    default: break;\n    }\n}\n")
    k = s.rstrip().rfind("}")
    s = s[:k] + disp + PK_RUN + "\n" + s[k:]
    return PKSrc(s, meta)


def pack(a, kd):
    import numpy as np
    if kd == "p":
        assert isinstance(a, np.uint64), type(a)
        return int(a)
    if kd == "i":
        assert isinstance(a, np.int32), type(a)
        return int(a) & 0xffffffff
    if kd == "f":
        assert isinstance(a, np.float32), type(a)
        return int(np.float32(a).view(np.uint32))
    assert isinstance(a, np.float64), type(a)
    return int(np.float64(a).view(np.uint64))


def build_cmds(cmds, meta, chain=None):
    """모은 명령 [(이름, 스레드 수, 블록 크기, 인자, 블록 0 전용)] -> (CMD int32 [n x 8], ARGS uint64, 통계). 가상 블록 1 개 명령은 블록 0 전용으로."""
    import numpy as np
    C, A = [], []
    for (name, n, b, args, loc) in cmds:
        kid, kinds = meta[name]
        assert len(args) == len(kinds), (name, len(args), len(kinds))
        off = len(A)
        A.extend(pack(a, kd) for a, kd in zip(args, kinds))
        nvb = (int(n) + int(b) - 1) // int(b)
        local = bool(loc) or nvb <= 1
        C.append([kid, nvb, int(b), (1 if local else 0) | (2 if name in EXEMPT else 0), off, 0, 0, 0])
    for i, (name, n, b, args, loc) in enumerate(cmds):                                # 이음 표시 (RTSolver.pk_chain): 같은 크기 가상 블록·둘 다 격자 명령일 때만
        if chain is not None and i in chain:
            assert i + 1 < len(C) and not (C[i][3] & 1) and not (C[i + 1][3] & 1) and C[i][2] == C[i + 1][2] and C[i][1] == C[i + 1][1] and not (C[i][3] & 2) and not (C[i + 1][3] & 2), (name, cmds[i + 1][0])
            C[i][3] |= 8
    i, nrun = 0, 0
    while i < len(C):
        if C[i][3] & 1:
            j = i
            while j + 1 < len(C) and (C[j + 1][3] & 1):
                j += 1
            ex = any(C[q][3] & 2 for q in range(i, j + 1))
            for q in range(i, j + 1):
                C[q][5] = j
            if ex:
                C[i][3] |= 4
            nrun += 1
            i = j + 1
        else:
            i += 1
    nglob = sum(1 for c in C if not (c[3] & 1))
    nch = sum(1 for c in C if c[3] & 8)
    st = dict(ncmd=len(C), nargs=len(A), grid_cmds=nglob, local_runs=nrun, chained=nch, syncs_max=nglob + nrun - nch)
    Cn = np.array(C, dtype=np.int32).reshape(-1, 8)
    An = np.array(A or [0], dtype=np.uint64)
    return Cn, An, st
