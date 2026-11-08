"""절 CM49 묶음 escape4: 장벽 표본점의 '걸음 끝 기준 높이' 를 다음 걸음 시작 높이로 이어 받기 (R) + 걸음 시작 d ≤ 0 표본점을 빼지 않는 상대 장벽 (A).
인자 없으면 이 모듈을 안 부름 = 소스·그래프 그대로. W̄ (절 CM45 --e3w 1) 위에서만 (R 의 정확성이 W̄ 의 '표본점 이동 = 칸 입자 G2P 이동 질량 평균' 에 기댐).

원인 (절 CM46.1·이 절 진단 D5): 장벽은 걸음 끝에 '그 프레임 팔분 칸 q 의 입자 질량 평균 높이 > 0' 만 보장한다. 다음 프레임은 입자를 새 팔분 칸에 다시 모으므로
(칸 경계는 세계에 고정, 수박 바닥 z 0.02 는 셀 3 격자의 팔분 경계와 겹침) mm 단위로 바닥 밑에 담긴 입자들만 든 새 칸이 생기고, 그 칸의 걸음 시작 높이
(질량 평균) 는 바닥 밑 -> '걸음 시작 d ≤ 0 이면 장벽에서 뺌' 규칙으로 빠져 자유 낙하. 즉 걸음 끝 보장의 대상 (앞 프레임 칸 평균) 과 다음 걸음 시작 판정의
대상 (새 칸 평균) 이 다른 정의.

R (--e5r 1, 정의 변경): 입자마다 '자기 칸 표본점의 걸음 끝 장벽 기준 높이' h_p 를 이어 받는다.
    프레임 끝 (G2P 뒤):   h_p = d̃_q(p) (끝) = δ_q + Σ_{p'∈q} m_p' d(x_p') / Σ m_p'     (d(x) = x·n − p_n, δ_q = 그 프레임 표본점 기준 높이 − 실제 질량 평균 높이)
    프레임 시작 (P2G 뒤): d̃_q (시작) = Σ_{p∈q} m_p h_p / Σ m_p,  δ_q = d̃_q − d_q (실제 질량 평균 높이),
                          표본점 기준 위치 x̄_q ← x̄_q + δ_q n  (AF_BX 의 질량 가중 합에 m_q δ_q n 을 더함 -- 장벽·띠 고착·시작점 판정 커널 전부가 같은 기준을 봄)
  걸음 안 표본점 이동은 그대로 Σ W̄ (U + thinv (y − U)) (W̄ = 칸 입자 G2P 기저 질량 평균), 그래서 걸음 끝 기준 높이 = d̃_q (시작) + 이동 = δ_q + 칸 입자 실제 끝 높이 평균
  (G2P 가 U 에 선형이라 정확히) -> 장벽이 보장한 값 (> 0) 이 곧 다음 프레임 시작 기준 높이의 재료 (h_p). 다시 모아도 새 칸의 시작 기준 높이는 그 입자들이 받은
  보장 값의 질량 평균이라 > 0 (앞 칸이 모두 활성이었으면). 처음 h_p = d(x_p) (δ = 0). 질량 가중 Σ_q m_q δ_q 는 다시 모으기에 보존 (Σ_{p∈q} m_p (x̄_q − x_p) = 0) -> 처음 0 이면 0.
  입자 배열 읽기·쓰기는 프레임에 두 번 (P2G 뒤·G2P 뒤, W̄ 모으기와 같은 자리) -- 걸음·뉴턴 안에서는 입자를 안 읽음. 문턱·클램프·노드 바닥 없음.
A (--e5a 1, 정의 변경): 걸음 시작 기준 높이 d̃_q ≤ 0 인 표본점을 장벽에서 빼지 않고, 그 점의 장벽 원점을 걸음 시작 위치로 옮긴 상대 장벽:
    d̃_q (시작) ← d̂ (δ_q = d̂ − d_q) -- 시작 위치가 장벽 받침 가장자리 (에너지 0) 가 되고, 시작 위치보다 d̂ 더 들어가는 이동에서 장벽이 무한 -> '더 들어가는 이동만 막음'.
  d̂ 는 장벽 자체의 받침 두께 (새 값 아님). R 과 함께면 옮긴 기준이 h_p 로 이어져 다음 프레임에도 유지.
  둘 다 프레임 시작에 한 번 (K = 1 이면 걸음 시작 = 프레임 시작; 쪼갠 걸음 (ksplit) 의 뒤 걸음은 앞 걸음 끝이 장벽 안이라 그대로).

진단 D5 (AF_E5_DG=1, 옛 실행기 수박, 결과 불변 -- 읽기만): exe/escape4_wrap.py.
"""
import numpy as np
import torch

ST = dict(r=0, a=0)
G2PK = ("aq_g2p", "ao_g2p", "rt_g2p_ga", "rt_g2p_endg")                               # 입자 위치를 쓰는 G2P 커널 (그 뒤 h_p)

E5_SRC = r'''
// ---- 절 CM49 R·A: 표본점 걸음 끝 장벽 기준 높이 이어 받기 (AF_BX 위) ----
__device__ __forceinline__ size_t e5_q(int p, const float* fx, const int* pcd, const int* cidx) {
    int c = cidx[pcd[p]];
    int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
    return (size_t)c*8 + oc;
}
__global__ void e5_z(const int* IP, float* R, int* cnt) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q == 0) { cnt[0] = 0; cnt[1] = 0; }
    if (q >= IP[12]) return;
    R[q] = 0.f;
}
__global__ void e5_rp(int N, const int* IP, const float* fx, const float* m, const int* pcd, const int* cidx, const float* hr, float* R) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    atomicAdd(&R[e5_q(p, fx, pcd, cidx)], m[p]*hr[p]);
}
// 시작: d_q = 실제 질량 평균 높이, d̃_q = R ? Σ m h / Σ m : d_q; A 면 d̃ ≤ 0 -> d̂. δ_q = d̃ − d_q, x̄_q += δ_q n. cnt[0] = d̃ ≤ 0 (옮기기 전) 표본점 수, cnt[1] = A 로 옮긴 수
__global__ void e5_sh(const int* IP, const float* R, float* DQ, float n0, float n1, float n2, float pn, float dh, int useR, int lift, int* cnt) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3];
    if (!(bm > 0.f)) { DQ[q] = 0.f; return; }
    float da = (AF_BX[q*4]*n0 + AF_BX[q*4+1]*n1 + AF_BX[q*4+2]*n2)/bm - pn;
    float dr = useR ? R[q]/bm : da;
    if (!(dr > 0.f)) { atomicAdd(&cnt[0], 1); if (lift) { dr = dh; atomicAdd(&cnt[1], 1); } }
    float de = dr - da;
    DQ[q] = de;
    if (de != 0.f) { AF_BX[q*4] += bm*de*n0; AF_BX[q*4+1] += bm*de*n1; AF_BX[q*4+2] += bm*de*n2; }
}
// 끝 (G2P 뒤): S_q = Σ m d(x_p) (그 프레임 칸), h_p = δ_q + S_q / m_q
__global__ void e5_ep(int N, const int* IP, const float* fx, const float* m, const int* pcd, const int* cidx, const float* x, float n0, float n1, float n2, float pn, float* S) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    atomicAdd(&S[e5_q(p, fx, pcd, cidx)], m[p]*(x[p*3]*n0 + x[p*3+1]*n1 + x[p*3+2]*n2 - pn));
}
__global__ void e5_eh(int N, const int* IP, const float* fx, const int* pcd, const int* cidx, const float* S, const float* DQ, float* hr) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    size_t q = e5_q(p, fx, pcd, cidx);
    float bm = AF_BX[q*4+3];
    if (bm > 0.f) hr[p] = DQ[q] + S[q]/bm;
}
'''


def _apply(src):
    if "__device__ float* AF_BX;" not in src:
        return src
    i = src.rstrip().rfind("}")
    return src[:i] + E5_SRC + "\n}\n"


class _BK(tuple):
    """원래 _bk 튜플 그대로 (bk[0] = x 등) + h_p 사본 (속성)."""
    h = None


def plane(s):
    """(n, p_n, d̂) -- 옛 실행기 s.wbar (PN·DH) + s.floor NR, 새 실행기 s.floor (PN·DH·NR)."""
    W = getattr(s, "wbar", None) or s.floor
    nr = (s.floor or {}).get("NR", [0.0, 0.0, 1.0])
    return [float(q) for q in nr], float(W["PN"]), float(W["DH"])


def hasbar(s):
    return bool(getattr(s, "bar", False)) or getattr(s, "wbar", None) is not None


def init_hr(s):
    """h_p 처음 = d(x_p) (δ = 0). 첫 _bk 때 (어떤 기록·되돌림보다 앞)."""
    if getattr(s, "e5hr", None) is not None or not hasbar(s):
        return
    nr, pn, _ = plane(s)
    N = int(getattr(s, "N", s.x.shape[0]))
    s.e5hr = (s.x[:N].double() @ torch.tensor(nr, device=s.x.device, dtype=torch.float64) - pn).float().contiguous()


def install(r=0, a=0):
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_an_rt_v0 as V0
    from . import fem_det_rt as DET
    from . import fem_mat_rt as MAT
    from . import fem_escape3 as E3                                                    # 표본점 커널 이름 (BARK); W̄ 설치는 이 뒤 escape3_wrap 이 (감싸개가 --e3w 1 확인)
    ST["r"], ST["a"] = int(r), int(a)
    P = ART.P
    _obs, _oas = ART.build_src, AORT._src
    ART.build_src = lambda *a_, **k_: _apply(_obs(*a_, **k_))
    AORT._src = lambda *a_, **k_: _apply(_oas(*a_, **k_))

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            if getattr(s, "_e5in", False):
                return _ok(s, name, n, args, b)
            if name == "rt_reset":
                s._e5st = not hasbar(s)
            elif name in E3.BARK and not getattr(s, "_e5st", True):
                s._e5st = True
                s._e5in = True
                try:
                    init_hr(s)
                    nr, pn, dh = plane(s)
                    i32, f = np.int32, np.float32
                    N = int(s.N)
                    nq = s.capQc * 8
                    _ok(s, "e5_z", nq, (P(s.IP), P(s.e5R), P(s.e5cnt)), 256)
                    if ST["r"]:
                        _ok(s, "e5_rp", N, (i32(N), P(s.IP), P(s.fx), P(s.m), P(s.pcd), P(s.cidx), P(s.e5hr), P(s.e5R)), 256)
                    _ok(s, "e5_sh", nq, (P(s.IP), P(s.e5R), P(s.e5DQ), *[f(q) for q in nr], f(pn), f(dh), i32(ST["r"]), i32(ST["a"]), P(s.e5cnt)), 256)
                finally:
                    s._e5in = False
                if not ST.get("said"):
                    ST["said"] = True
                    print(f"[절 CM49] 표본점 기준 높이 이어 받기 R {ST['r']} · 상대 장벽 A {ST['a']} (d̂ {dh:g}), 기준 옮김 자리 = {name} 앞", flush=True)
            r_ = _ok(s, name, n, args, b)
            if name in G2PK and ST["r"] and hasbar(s) and getattr(s, "e5hr", None) is not None:
                s._e5in = True
                try:
                    nr, pn, _ = plane(s)
                    i32, f = np.int32, np.float32
                    N = int(s.N)
                    nq = s.capQc * 8
                    _ok(s, "e5_z", nq, (P(s.IP), P(s.e5S), P(s.e5cnt2)), 256)
                    _ok(s, "e5_ep", N, (i32(N), P(s.IP), P(s.fx), P(s.m), P(s.pcd), P(s.cidx), P(s.x), *[f(q) for q in nr], f(pn), P(s.e5S)), 256)
                    _ok(s, "e5_eh", N, (i32(N), P(s.IP), P(s.fx), P(s.pcd), P(s.cidx), P(s.e5S), P(s.e5DQ), P(s.e5hr)), 256)
                finally:
                    s._e5in = False
            return r_
        cls.k = k

    def wrap_alloc(cls):
        _oa = cls.alloc

        def alloc(s, need):
            r_ = _oa(s, need)
            nq = s.capQc * 8                                                           # 표본점 용량 (AF_BX 와 같은 capQc·8; bx 는 하위 alloc 이 뒤에 만들 수 있어 용량으로)
            Z = lambda: torch.zeros(nq, device=s.dev, dtype=torch.float32)
            s.e5R, s.e5S, s.e5DQ = Z(), Z(), Z()
            s.e5cnt = torch.zeros(2, device=s.dev, dtype=torch.int32)
            s.e5cnt2 = torch.zeros(2, device=s.dev, dtype=torch.int32)
            return r_
        cls.alloc = alloc

    def wrap_bk(cls):                                                                   # h_p 를 프레임 시작 상태에 (다시 하기·넘침 다시 할당·쪼개기·첫 접촉 다시 풀기가 되돌림)
        _ob, _or = cls.__dict__["_bk"], cls.__dict__["_rs"]

        def _bk(s):
            init_hr(s)
            b_ = _BK(_ob(s))
            h = getattr(s, "e5hr", None)
            b_.h = h.clone() if h is not None else None
            return b_

        def _rs(s, bk):
            _or(s, bk)
            h = getattr(s, "e5hr", None)
            if h is not None and isinstance(bk, _BK) and bk.h is not None:
                h.copy_(bk.h)
        cls._bk = _bk
        cls._rs = _rs

    for cls in (ART.RTSolver, V0.RTSolver):
        wrap_k(cls)
        wrap_alloc(cls)
    for cls in {ART.RTSolver, V0.RTSolver, DET.DetMixin, MAT.MatRT, AORT.AORTSolver}:
        if "_bk" in cls.__dict__ and "_rs" in cls.__dict__:
            wrap_bk(cls)
