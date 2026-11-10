"""절 CM105 (사용자 11-10 지시): 노드 변 길이 상한 벌점 -- 이차 B-스플라인 (육면체) 판에서 ghost penalty 류 정규화를 모두 뺀 자리에 하나만.

정의 (doc/fem_realtime.md 절 CM105.0):
- 변 e = 노드 격자에서 축 a 방향으로 이웃한 활성 노드 쌍 (i, j = i + e_a), 활성 = 노드 질량 mn > 0 (실행기 풀이 노드와 같은 판정).
  기준 길이 L0_a = |A e_a| (격자 축 a 의 월드 길이 = 격자 간격). 걸음 끝 (뉴턴 미지수) 노드 위치 x_i = ξ_i + y_i 이므로 변 벡터 r = A e_a + y_j − y_i.
- E_edge = k Σ_e max(0, |r_e| − c L0_a)²  (상한 c L0 안에서는 0, 넘을 때만 제곱).
  기울기 ∂E/∂y_j = −∂E/∂y_i = 2k (d − cL0) r/d,  변 헤시안 H_e = 2k [(1 − cL0/d) I + (cL0/d) r rᵀ/d²] (d > cL0 에서 양의 준정치), 쌍 블록 [H −H; −H H].
- k = ½ (2μ + λ) L0 : 변 하나가 상한을 δ 넘을 때 E = ½ (2μ+λ) (δ/L0)² L0³ = 변형률 δ/L0 의 일축 변형 (P 파 계수) 탄성 에너지를 칸 부피 L0³ 만큼.
- c = 씬별 정답 궤적 (프레임 간 노드 변 늘어남 최대) 에서 (exe/edgep_gtc.py), 다섯 씬 같은 규칙.
실행기 연결 (두 실행기 같은 정의, 클래스 수준 RTSolver.k 감싸기 -- fem_ghost6 와 같은 자리):
- 요소 커널 (rt_elem·ao_elem·ao_elem_s·aq_elem) 뒤: 힘 f (모드 1) · 에너지 E (rt_elem 모드 4) · 블록 대각 B (모드 8) 에 같은 y 로 더하고, 그 y 에서의 변 상태 (r, cL0/d) 를 저장.
- 헤시안 곱 커널 (rt_mv·rt_mv2·rt_mv2s·ao_mv_*; MG 층 0 곱 포함) 뒤: out += H v (저장한 변 상태, 직전 요소 커널의 y 에서 선형화).
- rt_bd (블록 대각 따로) 뒤: s.y 에서 B 에 더함. 선탐색 후보 rt_elem_lsN 뒤: 후보마다 SC[40+i] 에 E(y_c). 옛 실행기 --en_mo Φ (en_phi_n) 뒤: PH[1] 에 E(y).
- 걸음 시작 (rt_ga_step0 등) 에서 변 상태를 0 으로 (앞 프레임 노드 번호 상태가 곱에 안 들어가게).
- 기록 (걸음 끝 커널 앞, 결과 불변): 활성 변 수, 상한 넘는 변 수, 최대 늘어남 d/L0, Σ((d − cL0)/L0)² -> ep.jsonl (프레임마다).
- 모드 0 = 기록만 (힘·곱·에너지 안 바꿈 = 정규화 끈 대조 판), 모드 1 = 벌점 켬.
"""
import json
import os
import time

import numpy as np

MAXROW = 64
NCMAX = 24
STEP0 = {"rt_step0", "rt_step0g", "rt_ga_step0", "ao_step0_ga", "ao_step0_th"}
ENDK = {"rt_stepend", "rt_stepend_g", "rt_ga_end", "rt_ga_end_uf", "rt_ga_end_ufl", "rt_ga_end_hc", "ao_stepend_ga", "ao_stepend_th", "ao_stepend_uf", "ao_stepend_ufl", "ao_stepend_hc", "en_stepend"}   # _hc = fem_hhtc2 가 장벽 씬에서 _ufl 대신 부르는 복사본
ELEM = ("rt_elem", "ao_elem", "ao_elem_s", "aq_elem")
MV = ("rt_mv", "rt_mv2", "rt_mv2s", "ao_mv_q", "ao_mv_h", "ao_mv_c", "ao_mv_s")
CFG = dict(mode=None, c=None, kscale=1.0, dg=1, ck=0)
INFO = dict(mode=None, c=None, hooks={}, solvers=[], err=None, mods=0, check=None)
OUT = ["."]

SRC = r'''
// ---- 절 CM105 노드 변 길이 상한 벌점 (lib/anchorflow/fem_edgep.py) ----
__device__ __forceinline__ bool ep_nb(const int* IP, const float* mn, int n, int a, int* j) {
    int nb0 = IP[6], nb1 = IP[7], nb2 = IP[8];
    int ia = (a == 0) ? n/(nb1*nb2) : ((a == 1) ? (n/nb2) % nb1 : n % nb2);
    int na = (a == 0) ? nb0 : ((a == 1) ? nb1 : nb2);
    if (ia + 1 >= na) return false;
    *j = n + ((a == 0) ? nb1*nb2 : ((a == 1) ? nb2 : 1));
    return mn[n] > 0.f && mn[*j] > 0.f;
}
// 변 (n, a) 의 r = A e_a + y_j − y_i (배정밀도), L0
__device__ __forceinline__ double ep_r(const float* Am, const float* y, int n, int j, int a, double* r) {
    double L2 = 0.0;
    for (int q=0;q<3;q++) { double aq = (double)Am[q*3+a]; L2 += aq*aq; r[q] = aq + (double)y[j*3+q] - (double)y[n*3+q]; }
    return sqrt(L2);
}
// 힘 (기울기) f · 블록 대각 B · 에너지 (반환) · 변 상태 S (r, cL0/d; 비활성 0) -- 노드 n 이 축 +a 변 셋을 맡음
__device__ double ep_body(const int* IP, const float* mn, const float* Am, float c, float k, const float* y, double* f, float* B, float* S, int n) {
    double e = 0.0;
    for (int a=0;a<3;a++) {
        int j = 0; float st[4] = {0.f, 0.f, 0.f, 0.f};
        if (ep_nb(IP, mn, n, a, &j)) {
            double r[3]; double L0 = ep_r(Am, y, n, j, a, r);
            double d = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]), a0 = (double)c*L0;
            if (d > a0) {
                double ex = d - a0;
                e += (double)k*ex*ex;
                double gs = 2.0*(double)k*ex/d;
                if (f) for (int q=0;q<3;q++) { atomicAdd(&f[j*3+q], gs*r[q]); atomicAdd(&f[n*3+q], -gs*r[q]); }
                double rho = a0/d;
                if (B) for (int p=0;p<3;p++) for (int q=0;q<3;q++) {
                    float h = (float)(2.0*(double)k*(((p == q) ? (1.0 - rho) : 0.0) + rho*r[p]*r[q]/(d*d)));
                    atomicAdd(&B[n*9+p*3+q], h); atomicAdd(&B[j*9+p*3+q], h); }
                st[0] = (float)r[0]; st[1] = (float)r[1]; st[2] = (float)r[2]; st[3] = (float)rho;
            }
        }
        if (S) { float* s_ = S + ((size_t)n*3 + a)*4; s_[0] = st[0]; s_[1] = st[1]; s_[2] = st[2]; s_[3] = st[3]; }
    }
    return e;
}
__global__ void __launch_bounds__(128) ep_force(const int* IP, const float* mn, const float* Am, float c, float k, const float* y, double* f, float* B, double* E, float* S) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0.0;
    if (n < IP[9] && !IP[13]) e = ep_body(IP, mn, Am, c, k, y, f, B, S, n);
    if (E) blocksum_add(e, E);
}
// 옛 실행기 --en_mo Φ: PH[1] += E(y) (en_phi_n 과 같은 MO 문)
__global__ void __launch_bounds__(128) ep_phi(const int* IP, const float* mn, const float* Am, float c, float k, const float* y, const int* MO, double* PH) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (!MO[0]) return;
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0.0;
    if (n < IP[9] && !IP[13]) e = ep_body(IP, mn, Am, c, k, y, (double*)0, (float*)0, (float*)0, n);
    blocksum_add(e, &PH[1]);
}
// 헤시안 곱: out += H v (저장한 변 상태)
__global__ void __launch_bounds__(128) ep_mv(const int* IP, float k, const float* S, const float* v, float* out) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13]) return;
    int nb1 = IP[7], nb2 = IP[8];
    for (int a=0;a<3;a++) {
        const float* st = S + ((size_t)n*3 + a)*4;
        float rho = st[3];
        if (!(rho > 0.f)) continue;
        int j = n + ((a == 0) ? nb1*nb2 : ((a == 1) ? nb2 : 1));
        float r[3] = {st[0], st[1], st[2]};
        float d2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
        float dv[3]; for (int q=0;q<3;q++) dv[q] = v[j*3+q] - v[n*3+q];
        float rd = (r[0]*dv[0] + r[1]*dv[1] + r[2]*dv[2])/d2;
        for (int q=0;q<3;q++) { float h = 2.f*k*((1.f - rho)*dv[q] + rho*r[q]*rd); atomicAdd(&out[j*3+q], h); atomicAdd(&out[n*3+q], -h); }
    }
}
__global__ void ep_szero(const int* IP, float* S) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int t=0;t<12;t++) S[(size_t)n*12 + t] = 0.f;
}
// 선탐색 후보 c_i = 2^-i (i0 ≤ i < NC): y_c = ŷ + T(η + c_i δ) -> SC[40+i] += E(y_c)  (rt_elem_lsN 과 같은 후보·같은 T)
__global__ void __launch_bounds__(128) ep_ls(const int* IP, const float* mn, const float* Am, float c, float k, const float* yh, const float* eta, const float* dl,
                                             const float* sk, const int* pm, float n0, float n1, float n2, double* SC, int NC, int i0, const double* gate) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double e[24];
    for (int i=0;i<24;i++) e[i] = 0.0;
    if (n < IP[9] && !IP[13]) {
        for (int a=0;a<3;a++) {
            int j = 0;
            if (!ep_nb(IP, mn, n, a, &j)) continue;
            double L2 = 0.0; for (int q=0;q<3;q++) L2 += (double)Am[q*3+a]*(double)Am[q*3+a];
            double a0 = (double)c*sqrt(L2);
            float cc = 1.f; for (int i=0;i<i0;i++) cc *= 0.5f;
            for (int i=i0;i<NC;i++) {
                float ev[3], ej[3], tn[3], tj[3];
                for (int q=0;q<3;q++) { ev[q] = eta[n*3+q] + cc*dl[n*3+q]; ej[q] = eta[j*3+q] + cc*dl[j*3+q]; }
                Tapply(sk[n], pm[n], n0, n1, n2, ev, tn); Tapply(sk[j], pm[j], n0, n1, n2, ej, tj);
                double r[3]; for (int q=0;q<3;q++) r[q] = (double)Am[q*3+a] + ((double)yh[j*3+q] + (double)tj[q]) - ((double)yh[n*3+q] + (double)tn[q]);
                double d = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
                if (d > a0) e[i] += (double)k*(d - a0)*(d - a0);
                cc *= 0.5f;
            }
        }
    }
    for (int i=i0;i<NC;i++) blocksum_add(e[i], &SC[40+i]);
}
// ---- 기록 (결과 불변): 걸음마다 행 하나 -- R[row*8 + 0..3] = 활성 변 수 · 상한 넘는 변 수 · Σ((d − cL0)/L0)² · 활성 노드 수, RI[row] = 최대 d/L0 (float 비트)
__global__ void ep_rowbeg(int* EI, double* R, int* RI, int maxrow) {
    int r = EI[0];
    if (r < maxrow) { for (int t=0;t<8;t++) R[r*8+t] = 0.0; RI[r] = 0; } else EI[2] = 1;
    EI[1] = r; EI[0] = r + 1;
}
__global__ void __launch_bounds__(128) ep_stat(const int* IP, const float* mn, const float* Am, float c, const float* y, const int* EI, double* R, int* RI, int maxrow) {
    int row = EI[1];
    if (row >= maxrow) return;
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double na = 0.0, no = 0.0, ex2 = 0.0, nn = 0.0; float rmax = 0.f;
    if (n < IP[9] && !IP[13]) {
        if (mn[n] > 0.f) nn = 1.0;
        for (int a=0;a<3;a++) {
            int j = 0;
            if (!ep_nb(IP, mn, n, a, &j)) continue;
            double r[3]; double L0 = ep_r(Am, y, n, j, a, r);
            double d = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]), a0 = (double)c*L0;
            na += 1.0;
            if (d > a0) { no += 1.0; ex2 += (d - a0)*(d - a0)/(L0*L0); }
            float q = (float)(d/L0); if (isfinite(q)) rmax = fmaxf(rmax, q); else rmax = 3.0e38f;
        }
    }
    blocksum_add(na, &R[row*8]); blocksum_add(no, &R[row*8+1]); blocksum_add(ex2, &R[row*8+2]); blocksum_add(nn, &R[row*8+3]);
    if (rmax > 0.f) atomicMax(&RI[row], __float_as_int(rmax));
}
'''


def transform(src):
    """해 모듈 (Tapply·blocksum_add 있음) 이면 벌점 커널을 extern C 끝에 덧붙임."""
    if not ('extern "C" {' in src and "void Tapply(" in src and "void blocksum_add(" in src):
        return src, False
    if "ep_force(" in src:
        return src, False
    j = src.rindex("}")
    return src[:j] + SRC + "\n" + src[j:], True


def _dump():
    try:
        json.dump(INFO, open(os.path.join(OUT[0], "epinfo.json"), "w"), ensure_ascii=False, indent=1, default=str)
    except Exception:
        pass


def _P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _cnt(name):
    INFO["hooks"][name] = INFO["hooks"].get(name, 0) + 1


class St:
    """해 인스턴스별 상태 (버퍼·k·기록)."""

    def __init__(h, s):
        import torch
        h.s, h.torch = s, torch
        mu, lam = getattr(s, "mu", None), getattr(s, "lam", None)
        prm = getattr(s, "prm", None)
        p01 = [float(prm[0]), float(prm[1])] if (prm is not None and hasattr(prm, "numel") and prm.numel() >= 2) else None
        if mu is None and p01 is not None:
            mu, lam = p01
        h.mu, h.lam = float(mu), float(lam)
        A = s.A.detach().double().reshape(3, 3).cpu()
        h.L0 = [float(torch.linalg.norm(A[:, a])) for a in range(3)]
        h.L0m = float(np.mean(h.L0))
        h.k = 0.5 * CFG["kscale"] * (2.0 * h.mu + h.lam) * h.L0m
        h.Am = s.A.detach().float().reshape(9).contiguous().clone()
        h.S = None
        h.phase = 0
        h.ncall = 0
        h.f = open(os.path.join(OUT[0], "ep.jsonl"), "a") if CFG["dg"] else None
        h.EI = torch.zeros(4, dtype=torch.int32, device=s.A.device)
        h.R = torch.zeros(MAXROW * 8, dtype=torch.float64, device=s.A.device)
        h.RI = torch.zeros(MAXROW, dtype=torch.int32, device=s.A.device)
        if getattr(s, "y", None) is not None:
            h.alloc()
        INFO["solvers"].append(dict(cls=type(s).__name__, mu=h.mu, lam=h.lam, prm01=p01, L0=h.L0, A=[float(v) for v in A.reshape(-1)], k=h.k, c=CFG["c"], mode=CFG["mode"]))
        print(f"[절 CM105 edgep] {type(s).__name__}: 모드 {CFG['mode']} ({'벌점 켬' if CFG['mode'] else '기록만'}), c {CFG['c']:.6g}, L0 {h.L0m:.6g} m (축 {', '.join('%.6g' % v for v in h.L0)}), "
              f"μ {h.mu:.6g}, λ {h.lam:.6g} -> k = ½ (2μ+λ) L0 x {CFG['kscale']:g} = {h.k:.6g} Pa·m", flush=True)
        _dump()

    def alloc(h):
        s, torch = h.s, h.torch
        ny = s.y.numel()
        if h.S is not None and h.S.numel() >= ny // 3 * 12:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[절 CM105] 캡처 중 버퍼 할당")
        h.S = torch.zeros(ny // 3 * 12 * 2, dtype=torch.float32, device=s.y.device)       # 용량 2 배 여유 (해 용량이 늘어도 캡처 안에서 다시 할당 안 하게)
        INFO["nalloc"] = INFO.get("nalloc", 0) + 1

    def nn(h):
        return int(h.s.y.numel() // 3)

    def read(h, t):
        """프레임 (s.step 한 번) 뒤 걸음 행 -> ep.jsonl."""
        if h.f is None:
            return
        EI = h.EI.cpu().numpy()
        nr = min(int(EI[0]), MAXROW)
        R = h.R[:nr * 8].view(nr, 8).cpu().numpy() if nr else np.zeros((0, 8))
        RI = h.RI[:nr].cpu().numpy() if nr else np.zeros((0,), np.int32)
        rows = [dict(nact=int(R[i, 0]), nover=int(R[i, 1]), ex2=float(R[i, 2]), nnode=int(R[i, 3]), rmax=float(np.array([RI[i]], np.int32).view(np.float32)[0])) for i in range(nr)]
        h.f.write(json.dumps(dict(c=h.ncall, t=float(t), steps=rows, over=int(EI[2]))) + "\n"); h.f.flush()
        h.ncall += 1
        h.EI.zero_()


def _go(s, _ok, name, n, args, b=128):
    _cnt(name)
    _ok(s, name, n, args, b)


def _hook(s, _ok, name, args):
    h = getattr(s, "_eph", None)
    if h is None:
        return
    f32, i32 = np.float32, np.int32
    nn = h.nn()
    c, on = float(CFG["c"]), bool(CFG["mode"])
    k = f32(h.k)
    if name in STEP0:
        h.alloc()
        _go(s, _ok, "ep_szero", nn, (_P(s.IP), _P(h.S)), 256)
        h.phase = 1
        return
    if name in ENDK:
        if h.phase and CFG["dg"]:
            _go(s, _ok, "ep_rowbeg", 1, (_P(h.EI), _P(h.R), _P(h.RI), i32(MAXROW)), 1)
            _go(s, _ok, "ep_stat", nn, (_P(s.IP), _P(s.mn), _P(h.Am), f32(c), _P(s.y), _P(h.EI), _P(h.R), _P(h.RI), i32(MAXROW)), 128)
        h.phase = 0
        return
    if not on:
        return
    G = (_P(s.IP), _P(s.mn), _P(h.Am), f32(c), k)
    if name in ELEM:
        if name == "rt_elem":
            mode = int(args[8]); y, E, f_, B = args[5], args[9], args[10], args[13]
        elif name == "aq_elem":
            mode = 1 | 8; y, E, f_, B = args[5], None, args[10], args[13]
        else:
            mode = int(args[9]); y, E, f_, B = args[5], None, args[10], args[13]
        st = bool(mode & 11)
        _go(s, _ok, "ep_force", nn, G + (y, f_ if mode & 1 else _P(None), B if mode & 8 else _P(None), E if (E is not None and mode & 4) else _P(None), _P(h.S) if st else _P(None)), 128)
        INFO["hooks"]["mode:%s:%d" % (name, mode)] = INFO["hooks"].get("mode:%s:%d" % (name, mode), 0) + 1
    elif name in MV:
        _go(s, _ok, "ep_mv", nn, (_P(s.IP), k, _P(h.S), args[6], args[7]), 128)
    elif name == "rt_bd":
        _go(s, _ok, "ep_force", nn, G + (_P(s.y), _P(None), args[6], _P(None), _P(h.S)), 128)
    elif name == "rt_elem_lsN":
        if len(args) >= 19:
            NC, i0, gate = int(args[16]), int(args[17]), args[18]
        else:
            NC, i0, gate = int(args[16]), 0, _P(None)
        assert NC <= NCMAX, f"[절 CM105] 선탐색 후보 {NC} > {NCMAX}"
        _go(s, _ok, "ep_ls", nn, G + (args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[15], i32(NC), i32(i0), gate), 128)
    elif name == "en_phi_n":
        _go(s, _ok, "ep_phi", nn, G + (args[3], args[5], args[6]), 128)
    elif name == "rt_elem_ls":
        raise RuntimeError(f"[절 CM105] {name} 경로는 안 고침")


def selfcheck(h, out):
    """유한차분·참값 검사 (그래프 밖, 해 상태 안 바꿈): 지금 프레임 끝 노드 상자·질량으로, 활성 변이 많게 시험 상한 c_t 와 흔든 y 에서
    (1) 커널 (f32 y, f64 합) 대 torch f64 참값 -- E · 기울기 · 블록 대각 · 헤시안 곱 · 선탐색 후보 에너지 · Φ,
    (2) 참값의 기울기·헤시안 곱 대 중심 유한차분 (f64), (3) 커널 에너지 중심 차분 대 커널 기울기."""
    s, torch = h.s, h.torch
    dev = s.y.device
    L = lambda s_, *a_: s_.k(*a_)                                                         # ep_ 이름은 훅이 건너뜀
    IP = s.IP.cpu().numpy()
    nb = [int(IP[6]), int(IP[7]), int(IP[8])]
    nnode = int(IP[9])
    if nnode <= 0 or int(IP[13]):
        return dict(err="빈 상자")
    mn = s.mn.reshape(-1)[:nnode].float()
    act = mn > 0
    idx = torch.arange(nnode, device=dev)
    i0, i1, i2 = idx // (nb[1] * nb[2]), (idx // nb[2]) % nb[1], idx % nb[2]
    A = h.Am.double().reshape(3, 3)
    E_list = []
    for a, (ia, st) in enumerate(((i0, nb[1] * nb[2]), (i1, nb[2]), (i2, 1))):
        ok = (ia + 1 < nb[a]) & act
        j = idx + st
        ok = ok & act[j.clamp(max=nnode - 1)]
        n_ = idx[ok]
        E_list.append((n_, n_ + st, a))
    ne = sum(int(q[0].numel()) for q in E_list)
    if ne == 0:
        return dict(err="활성 변 없음")
    ct = 0.97
    kk = float(h.k)
    g = torch.Generator(device=dev); g.manual_seed(1234)
    y0 = s.y.reshape(-1)[:nnode * 3].double().view(-1, 3).clone()
    y0 = y0 + 0.03 * h.L0m * torch.randn(y0.shape, generator=g, device=dev, dtype=torch.float64) * act[:, None]
    y0 = y0.float().double()                                                               # 커널과 같은 f32 값
    v = (torch.randn(y0.shape, generator=g, device=dev, dtype=torch.float64) * act[:, None] * h.L0m).float().double()

    def Eref(y):
        e = torch.zeros((), dtype=torch.float64, device=dev)
        for n_, j_, a in E_list:
            r = A[:, a][None, :] + y[j_] - y[n_]
            d = torch.linalg.norm(r, dim=1)
            e = e + kk * torch.clamp(d - ct * h.L0[a], min=0.0).pow(2).sum()
        return e

    def gref(y):
        yy = y.clone().requires_grad_(True)
        e = Eref(yy)
        gg, = torch.autograd.grad(e, yy, create_graph=True)
        return e, gg, yy

    e_r, g_r, yy = gref(y0)
    Hv_r, = torch.autograd.grad((g_r * v).sum(), yy)
    g_r = g_r.detach()
    Bref = torch.zeros(nnode, 3, 3, dtype=torch.float64, device=dev)
    for n_, j_, a in E_list:
        r = A[:, a][None, :] + y0[j_] - y0[n_]
        d = torch.linalg.norm(r, dim=1)
        a0 = ct * h.L0[a]
        m = d > a0
        rho = torch.where(m, a0 / d, torch.zeros_like(d))
        He = 2 * kk * ((1 - rho)[:, None, None] * torch.eye(3, dtype=torch.float64, device=dev)[None] + rho[:, None, None] * r[:, :, None] * r[:, None, :] / (d * d)[:, None, None])
        He = He * m[:, None, None]
        Bref.index_add_(0, n_, He); Bref.index_add_(0, j_, He)
    cap = s.y.numel() // 3
    yk = torch.zeros(cap * 3, dtype=torch.float32, device=dev); yk[:nnode * 3] = y0.reshape(-1).float()
    vk = torch.zeros(cap * 3, dtype=torch.float32, device=dev); vk[:nnode * 3] = v.reshape(-1).float()
    fk = torch.zeros(cap * 3, dtype=torch.float64, device=dev)
    Bk = torch.zeros(cap * 9, dtype=torch.float32, device=dev)
    Ek = torch.zeros(1, dtype=torch.float64, device=dev)
    Sk = torch.zeros(cap * 12, dtype=torch.float32, device=dev)
    ok_ = torch.zeros(cap * 3, dtype=torch.float32, device=dev)
    f32, i32 = np.float32, np.int32
    G = (_P(s.IP), _P(s.mn), _P(h.Am), f32(ct), f32(kk))
    L(s, "ep_force", cap, G + (_P(yk), _P(fk), _P(Bk), _P(Ek), _P(Sk)), 128)
    L(s, "ep_mv", cap, (_P(s.IP), f32(kk), _P(Sk), _P(vk), _P(ok_)), 128)
    NC = 10
    SC = torch.zeros(64, dtype=torch.float64, device=dev)
    sk = torch.ones(cap, dtype=torch.float32, device=dev); pm = torch.zeros(cap, dtype=torch.int32, device=dev)
    eta0 = torch.zeros(cap * 3, dtype=torch.float32, device=dev)
    L(s, "ep_ls", cap, G + (_P(yk), _P(eta0), _P(vk), _P(sk), _P(pm), f32(0), f32(0), f32(1), _P(SC), i32(NC), i32(0), _P(None)), 128)
    MO = torch.ones(4, dtype=torch.int32, device=dev); PH = torch.zeros(4, dtype=torch.float64, device=dev)
    L(s, "ep_phi", cap, G + (_P(yk), _P(MO), _P(PH)), 128)
    torch.cuda.synchronize()
    rel = lambda a_, b_: float(torch.linalg.norm((a_ - b_).reshape(-1)) / max(float(torch.linalg.norm(b_.reshape(-1))), 1e-300))
    out = dict(nnode=nnode, ne=ne, c_test=ct, k=kk, L0=h.L0m, E_ref=float(e_r))
    nact = 0
    for n_, j_, a in E_list:
        r = A[:, a][None, :] + y0[j_] - y0[n_]
        nact += int((torch.linalg.norm(r, dim=1) > ct * h.L0[a]).sum())
    out["nact_test"] = nact
    out["E_kernel_rel"] = abs(float(Ek[0]) - float(e_r)) / max(abs(float(e_r)), 1e-300)
    out["grad_kernel_rel"] = rel(fk[:nnode * 3].view(-1, 3), g_r)
    out["B_kernel_rel"] = rel(Bk[:nnode * 9].double().view(-1, 3, 3), Bref)
    out["Hv_kernel_rel"] = rel(ok_[:nnode * 3].double().view(-1, 3), Hv_r.detach())       # 커널 v 는 f32 -> 참값도 같은 f32 값 v
    cs = [0.5 ** i for i in range(NC)]
    Ec_ref = [float(Eref(y0 + cc * v)) for cc in cs]
    out["ls_kernel_rel"] = max(abs(float(SC[40 + i]) - Ec_ref[i]) / max(abs(Ec_ref[i]), 1e-300) for i in range(NC))
    out["ls_E"] = [[float(SC[40 + i]), Ec_ref[i]] for i in range(NC)]
    out["phi_kernel_rel"] = abs(float(PH[1]) - float(e_r)) / max(abs(float(e_r)), 1e-300)
    # (2) 참값 (f64) 기울기·헤시안 곱 대 중심 유한차분: y ± ε v (v ~ L0 크기). 헤시안은 상한 d = cL0 에서 끊기므로 (기울기는 연속)
    #     헤시안 곱 차분은 y0 에서 |d − cL0| < 1e-3 L0 인 변 (ε 흔들림에 활성이 바뀔 수 있는 변) 을 뺀 같은 에너지로 비교 (뺀 변 수 기록).
    kink = []
    for n_, j_, a in E_list:
        r = A[:, a][None, :] + y0[j_] - y0[n_]
        kink.append((torch.linalg.norm(r, dim=1) - ct * h.L0[a]).abs() >= 1e-3 * h.L0[a])

    def Emask(y):
        e = torch.zeros((), dtype=torch.float64, device=dev)
        for (n_, j_, a), km in zip(E_list, kink):
            r = A[:, a][None, :] + y[j_[km]] - y[n_[km]]
            e = e + kk * torch.clamp(torch.linalg.norm(r, dim=1) - ct * h.L0[a], min=0.0).pow(2).sum()
        return e

    def gmask(y):
        yy_ = y.clone().requires_grad_(True)
        gg_, = torch.autograd.grad(Emask(yy_), yy_, create_graph=True)
        return gg_, yy_

    gm0, ym0 = gmask(y0)
    Hvm, = torch.autograd.grad((gm0 * v).sum(), ym0)
    out["n_kink_excl"] = int(sum(int((~km).sum()) for km in kink))
    fd = {}
    gv = float((g_r * v).sum())
    for eps in (1e-4, 1e-6):
        Ep, Em = float(Eref(y0 + eps * v)), float(Eref(y0 - eps * v))
        gp, _ = gmask(y0 + eps * v); gmn, _ = gmask(y0 - eps * v)
        Hfd = (gp.detach() - gmn.detach()) / (2 * eps)
        fd["%g" % eps] = dict(gv=gv, gv_fd=(Ep - Em) / (2 * eps), gv_rel=abs((Ep - Em) / (2 * eps) - gv) / max(abs(gv), 1e-300), Hv_rel=rel(Hfd, Hvm.detach()))
    out["fd_ref"] = fd
    # (3) 커널 유한차분 (f32 y): E_k(y ± ε v) 중심 차분 대 커널 기울기 · v
    eps = 1e-3
    for sg, key in ((1, "p"), (-1, "m")):
        yk2 = yk.clone(); yk2[:nnode * 3] = (y0 + sg * eps * v).reshape(-1).float()
        Ek2 = torch.zeros(1, dtype=torch.float64, device=dev)
        L(s, "ep_force", cap, G + (_P(yk2), _P(None), _P(None), _P(Ek2), _P(None)), 128)
        torch.cuda.synchronize()
        out["Ek_" + key] = float(Ek2[0])
    gvk = float((fk[:nnode * 3].view(-1, 3) * v).sum())
    out["kernel_fd"] = dict(eps=eps, gv=gvk, gv_fd=(out["Ek_p"] - out["Ek_m"]) / (2 * eps), rel=abs((out["Ek_p"] - out["Ek_m"]) / (2 * eps) - gvk) / max(abs(gvk), 1e-300))
    out["ok"] = bool(out["E_kernel_rel"] < 1e-5 and out["grad_kernel_rel"] < 1e-4 and out["B_kernel_rel"] < 1e-4 and out["Hv_kernel_rel"] < 1e-4 and out["ls_kernel_rel"] < 1e-5
                     and out["phi_kernel_rel"] < 1e-5 and fd["1e-06"]["gv_rel"] < 1e-6 and fd["1e-06"]["Hv_rel"] < 1e-5 and out["kernel_fd"]["rel"] < 1e-2 and nact > 0)
    return out


def install(out, mode, c, kscale=1.0, dg=1, ck=0):
    import cupy as cp
    OUT[0] = out
    CFG.update(mode=int(mode), c=float(c), kscale=float(kscale), dg=int(dg), ck=int(ck))
    INFO.update(mode=int(mode), c=float(c), kscale=float(kscale))
    for fn in ("ep.jsonl", "epck.json"):
        try:
            os.remove(os.path.join(out, fn))
        except OSError:
            pass
    base = cp.RawModule
    if not getattr(base, "_edgep", False):
        def RawModule(*a, **kw):
            a = list(a)
            code = kw.get("code", a[0] if a else None)
            if isinstance(code, str):
                code2, did = transform(code)
                if did:
                    INFO["mods"] += 1; _dump()
                    if "code" in kw:
                        kw["code"] = code2
                    else:
                        a[0] = code2
            return base(*a, **kw)
        RawModule._edgep = True
        cp.RawModule = RawModule
    from . import fem_an_rt as ART
    from . import fem_an_rt_v0 as V0
    for cls in (ART.RTSolver, V0.RTSolver):                                              # 클래스 수준 k 감싸기 (fem_ghost6 와 같은 자리: super().k 경로도 다 지남)
        if getattr(cls.k, "_edgep", False):
            continue
        _ok = cls.k

        def k(s, name, n, args, b=128, _ok=_ok):
            r = _ok(s, name, n, args, b)
            if getattr(s, "_epin", False) or name.startswith("ep_"):
                return r
            s._epin = True
            try:
                _hook(s, _ok, name, args)
            except Exception as e_:
                INFO["err"] = f"{name}: {e_}"[:600]; _dump(); raise
            finally:
                s._epin = False
            return r
        k._edgep = True
        cls.k = k
    mods = [ART, V0]
    for nm in ("fem_ao_rt", "fem_mat_rt"):
        try:
            mods.append(__import__("anchorflow." + nm, fromlist=[nm]))
        except Exception as e_:
            INFO.setdefault("imp_err", []).append(f"{nm}: {e_}"[:200])
    for M in mods:
        for cn in ("RTSolver", "AORTSolver", "RTAdapt", "MatRT"):
            C = getattr(M, cn, None)
            if C is None or getattr(C.__init__, "_edgep", False):
                continue
            f0 = C.__init__

            def init(s, *a, _f0=f0, **kw):
                _f0(s, *a, **kw)
                if getattr(s, "_eph", None) is not None:
                    return
                s._eph = St(s)
                st0 = s.step

                def step(t, *a2, **kw2):
                    s._eph.alloc()                                                          # 캡처 밖에서 용량 확인
                    r = st0(t, *a2, **kw2)
                    h = s._eph
                    if CFG["dg"]:
                        h.read(t)
                    if CFG["ck"] and not getattr(h, "_ckd", False) and h.ncall >= 3:
                        h._ckd = True
                        try:
                            o = selfcheck(h, OUT[0])
                        except Exception as e_:
                            o = dict(err=str(e_)[:600])
                        INFO["check"] = o
                        json.dump(o, open(os.path.join(OUT[0], "epck.json"), "w"), indent=1, default=str)
                        print(f"[절 CM105 edgep] 검사 {'통과' if o.get('ok') else '실패'}: " + json.dumps({q: o.get(q) for q in ('ne', 'nact_test', 'E_kernel_rel', 'grad_kernel_rel', 'B_kernel_rel', 'Hv_kernel_rel', 'ls_kernel_rel', 'phi_kernel_rel', 'err')}, default=str), flush=True)
                        _dump()
                    return r
                s.step = step
            init._edgep = True
            C.__init__ = init
    print(f"[절 CM105] edgep 켬 -- 모드 {mode} ({'벌점 켬' if mode else '기록만 (결과 불변)'}), c {c:.6g}, k 배율 {kscale:g}, 기록 {dg}, 검사 {ck}", flush=True)
    _dump()


def summarize(path):
    """ep.jsonl -> 판 요약 (같은 t 는 마지막 시도만)."""
    L = [json.loads(l) for l in open(path)]
    by = {}
    for q in L:
        by[round(q["t"], 9)] = q
    S = [r for kk in sorted(by) for r in by[kk]["steps"]]
    if not S:
        return {}
    return dict(nstep=len(S), steps_over=sum(r["nover"] > 0 for r in S), max_nover=max(r["nover"] for r in S), sum_nover=sum(r["nover"] for r in S),
                max_rmax=max(r["rmax"] for r in S), mean_nact=float(np.mean([r["nact"] for r in S])), max_ex2=max(r["ex2"] for r in S),
                frame_max=[max((r["rmax"] for r in by[kk]["steps"]), default=0.0) for kk in sorted(by)],
                frame_nover=[max((r["nover"] for r in by[kk]["steps"]), default=0) for kk in sorted(by)])
