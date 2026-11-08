"""절 CM47 묶음 ghost1: B-스플라인 작은 받침 안정화 -- ghost penalty (면 기울기 점프 벌점). 감싸개 인자 없으면 이 모듈을 안 부름 = 소스·그래프 그대로.

문헌식 (Hansbo, Larson, Larsson 2017, "Cut finite element methods for linear elasticity problems", arXiv:1703.04377, 식 (2.18)·(2.20)·5 절;
같은 꼴 = Burman 2010 "Ghost penalty" C. R. Math. 348, Burman·Hansbo 2012 Appl. Numer. Math. 62, de Prenter 외 2023 리뷰 arXiv:2208.08538 식 (4.5)):
    j_h(v, w) = Σ_{F ∈ F_h(∂Ω)} Σ_{l=1}^{p} h^{2l+1} ([D^l_n v], [D^l_n w])_F,       a_h = a + γ_a h^{-2} j_h,     γ_a = (2μ + λ) · 10^{-4}
  F_h(∂Ω) = 경계에 걸린 요소 (K ∩ ∂Ω ≠ ∅) 의 내부 면 (두 요소 모두 활성), p = 2 (이차 B-스플라인).
이 판에 옮김 (문턱·클램프 없음, γ 는 위 문헌 기본값 그대로):
  - 요소 = 칸 (이차 B-스플라인 매듭 구간, 노드 base b..b+2), 활성 = 입자 있는 칸 (cidx ≥ 0). 경계 칸 = 3x3x3 이웃 26 칸 중 비활성 (또는 칸 상자 밖) 이 있는 활성 칸
    (Ω_h = 활성 칸 합집합일 때 K 의 닫힘이 ∂Ω_h 에 닿는 칸 -- 위상 판정, 조절 값 없음). 면 = 축 방향 이웃 두 칸이 모두 활성이고 둘 중 하나가 경계 칸.
  - 이차 B-스플라인은 C¹ 이라 l = 1 점프는 0 (정확히). l = 2: 칸 안 ∂²_n u = (u_b − 2u_{b+1} + u_{b+2}) / h² (상수), 면 점프 = (−u_b + 3u_{b+1} − 3u_{b+2} + u_{b+3}) / h²,
    면 위 가로 두 방향은 이차 B-스플라인 조각 (면 질량 M = [[6,13,1],[13,54,13],[1,13,6]]/120 의 텐서곱) 으로 정확히 적분.
    -> 에너지 E_g = ½ γ_a h^{-2} h^5 ∫_F |[∂²_n y]|² = ½ k Σ_F Σ_{t,t'} (M⊗M)_{tt'} D_t·D_t',  k = γ_a h,  D_t = 3 차 차분 (칸 단위 변위 y, 길이 m).
  - 변수 = 걸음 풀이 변위 y (요소 커널과 같은 y; 프레임 시작 배치 기준), h = |det A|^{1/3} (칸 크기 m), μ·λ = 풀이기 s.mu·s.lam.
  - 걸음 풀이에 탄성 에너지처럼 더함: 기울기 f += G y (요소 커널 뒤), 블록 대각 B += diag G, 헤시안 곱 KTp += G Tp (모든 고운 층 곱 커널 뒤 -- MG 층 0 연산자 포함, 거친 층은 그대로),
    에너지 (새 실행기 선탐색 rt_elem 모드 4·rt_elem_lsN 후보, 옛 실행기 --en_mo Φ en_phi_q) += ½ yᵀ G y. 관성·G2P·질량·장벽은 그대로.
"""
import math

import numpy as np
import torch

GAMMA_A = 1e-4                                                                          # Hansbo·Larson·Larsson 2017 5 절: γ_a = (2μ + λ) · 10^-4
ST = dict(on=False, said=False)
ELEM = ("rt_elem", "aq_elem", "ao_elem", "ao_elem_s")
MV = ("rt_mv", "rt_mv2", "rt_mv2s", "ao_mv_q", "ao_mv_h", "ao_mv_c", "ao_mv_s")

GP_SRC = r'''
// ---- 절 CM47 ghost penalty (lib/anchorflow/fem_ghost1.py) ----
__device__ __forceinline__ float gp_m(int a, int b) { const float M[9] = {6.f/120.f, 13.f/120.f, 1.f/120.f, 13.f/120.f, 54.f/120.f, 13.f/120.f, 1.f/120.f, 13.f/120.f, 6.f/120.f}; return M[a*3+b]; }
__device__ __forceinline__ float gp_cl(int l) { return (l == 0) ? -1.f : ((l == 1) ? 3.f : ((l == 2) ? -3.f : 1.f)); }
__device__ __forceinline__ void gp_ids(const int* IP, const int* cb, int fc, int* ids) {   // 면 fc = c*3 + 축 -> 36 노드 (층 l 0..3 x 가로 i, j 0..2)
    int c = fc / 3, a = fc % 3, t1 = (a == 0) ? 1 : 0, t2 = (a == 2) ? 1 : 2;
    int nb1 = IP[7], nb2 = IP[8];
    for (int l=0;l<4;l++) for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        int x[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x[a] += l; x[t1] += i; x[t2] += j;
        ids[l*9 + i*3 + j] = (x[0]*nb1 + x[1])*nb2 + x[2];
    }
}
__device__ __forceinline__ void gp_D(const int* ids, const float* u, float D[9][3]) {
    for (int t=0;t<9;t++) for (int a=0;a<3;a++) { float s = 0.f; for (int l=0;l<4;l++) s += gp_cl(l)*u[ids[l*9+t]*3+a]; D[t][a] = s; }
}
__device__ __forceinline__ void gp_MD(float D[9][3], float MD[9][3]) {
    for (int t=0;t<9;t++) for (int a=0;a<3;a++) { float s = 0.f; for (int t2=0;t2<9;t2++) s += gp_m(t/3, t2/3)*gp_m(t%3, t2%3)*D[t2][a]; MD[t][a] = s; }
}
__global__ void gp_zero(int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    cnt[0] = 0; cnt[1] = 0;
}
__global__ void gp_cut(const int* IP, const int* cb, const int* cidx, int* cut, int* cnt) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n0 = IP[6]-2, n1 = IP[7]-2, n2 = IP[8]-2;
    int x = cb[c*3], y = cb[c*3+1], z = cb[c*3+2], k = 0;
    for (int dx=-1;dx<=1 && !k;dx++) for (int dy=-1;dy<=1 && !k;dy++) for (int dz=-1;dz<=1 && !k;dz++) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        int X = x+dx, Y = y+dy, Z = z+dz;
        if (X < 0 || Y < 0 || Z < 0 || X >= n0 || Y >= n1 || Z >= n2 || cidx[(X*n1 + Y)*n2 + Z] < 0) k = 1;
    }
    cut[c] = k;
    if (k) atomicAdd(&cnt[0], 1);
}
__global__ void gp_face(const int* IP, const int* cb, const int* cidx, const int* cut, int* cnt, int* fl) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int c = i / 3, a = i % 3;
    if (IP[13] || c >= IP[11]) return;
    int n[3] = {IP[6]-2, IP[7]-2, IP[8]-2};
    int x[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x[a] += 1;
    if (x[a] >= n[a]) return;
    int c2 = cidx[(x[0]*n[1] + x[1])*n[2] + x[2]];
    if (c2 < 0 || !(cut[c] || cut[c2])) return;
    int s = atomicAdd(&cnt[1], 1);
    fl[s] = i;
}
// f += k G y (f64), B 대각 += k c_l² M_tt (블록 I3), E += ½ k yᵀ G y
__global__ void gp_force(const int* IP, const int* cb, const int* cnt, const int* fl, float k, const float* y, double* f, float* B, double* E) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (blockIdx.x*blockDim.x >= cnt[1]) return;
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0.0;
    if (s < cnt[1]) {
        int ids[36]; gp_ids(IP, cb, fl[s], ids);
        float D[9][3], MD[9][3]; gp_D(ids, y, D); gp_MD(D, MD);
        for (int t=0;t<9;t++) for (int a=0;a<3;a++) e += 0.5*(double)k*(double)D[t][a]*(double)MD[t][a];
        if (f) for (int l=0;l<4;l++) for (int t=0;t<9;t++) for (int a=0;a<3;a++) atomicAdd(&f[ids[l*9+t]*3+a], (double)(k*gp_cl(l)*MD[t][a]));
        if (B) for (int l=0;l<4;l++) for (int t=0;t<9;t++) { float v = k*gp_cl(l)*gp_cl(l)*gp_m(t/3, t/3)*gp_m(t%3, t%3); int id = ids[l*9+t];
            atomicAdd(&B[id*9], v); atomicAdd(&B[id*9+4], v); atomicAdd(&B[id*9+8], v); }
    }
    if (E) blocksum_add(e, E);
}
// out += k G dy (f32)
__global__ void gp_mv(const int* IP, const int* cb, const int* cnt, const int* fl, float k, const float* dy, float* out) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if (s >= cnt[1]) return;
    int ids[36]; gp_ids(IP, cb, fl[s], ids);
    float D[9][3], MD[9][3]; gp_D(ids, dy, D); gp_MD(D, MD);
    for (int l=0;l<4;l++) for (int t=0;t<9;t++) for (int a=0;a<3;a++) atomicAdd(&out[ids[l*9+t]*3+a], k*gp_cl(l)*MD[t][a]);
}
// 선탐색 후보 y_c = ŷ + T(η + c δ) = a + c b (T 선형) -> Q_aa·Q_ab·Q_bb (GQ 0..2)
__global__ void gp_ls(const int* IP, const int* cb, const int* cnt, const int* fl, const float* yh, const float* eta, const float* dl, const float* sk, const int* pm,
                      float n0, float n1, float n2, double* GQ, const double* gate) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (gate && *gate != 0.0) return;
    if (blockIdx.x*blockDim.x >= cnt[1]) return;
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    double qa = 0.0, qab = 0.0, qb = 0.0;
    if (s < cnt[1]) {
        int ids[36]; gp_ids(IP, cb, fl[s], ids);
        float Da[9][3], Db[9][3];
        for (int t=0;t<9;t++) for (int a=0;a<3;a++) { Da[t][a] = 0.f; Db[t][a] = 0.f; }
        for (int l=0;l<4;l++) for (int t=0;t<9;t++) { int id = ids[l*9+t];
            float e[3] = {eta[id*3], eta[id*3+1], eta[id*3+2]}, d[3] = {dl[id*3], dl[id*3+1], dl[id*3+2]}, te[3], td[3];
            Tapply(sk[id], pm[id], n0, n1, n2, e, te); Tapply(sk[id], pm[id], n0, n1, n2, d, td);
            for (int a=0;a<3;a++) { Da[t][a] += gp_cl(l)*(yh[id*3+a] + te[a]); Db[t][a] += gp_cl(l)*td[a]; } }
        float Ma[9][3], Mb[9][3]; gp_MD(Da, Ma); gp_MD(Db, Mb);
        for (int t=0;t<9;t++) for (int a=0;a<3;a++) { qa += (double)Da[t][a]*Ma[t][a]; qab += (double)Da[t][a]*Mb[t][a]; qb += (double)Db[t][a]*Mb[t][a]; }
    }
    blocksum_add(qa, &GQ[0]); blocksum_add(qab, &GQ[1]); blocksum_add(qb, &GQ[2]);
}
__global__ void gp_lsfin(double* SC, int NC, int i0, const double* gate, double* GQ, float k) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (gate && *gate != 0.0) return;
    float c = 1.f; for (int i=0;i<i0;i++) c *= 0.5f;
    for (int i=i0;i<NC;i++) { double cd = (double)c; SC[40+i] += 0.5*(double)k*(GQ[0] + 2.0*cd*GQ[1] + cd*cd*GQ[2]); c *= 0.5f; }
    GQ[0] = 0.0; GQ[1] = 0.0; GQ[2] = 0.0;
}
// 옛 실행기 --en_mo Φ: PH[1] += ½ k yᵀ G y (en_phi_q 와 같은 MO 문)
__global__ void gp_phi(const int* IP, const int* cb, const int* cnt, const int* fl, float k, const float* y, const int* MO, double* PH) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    if (!MO[0]) return;
    if (blockIdx.x*blockDim.x >= cnt[1]) return;
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0.0;
    if (s < cnt[1]) {
        int ids[36]; gp_ids(IP, cb, fl[s], ids);
        float D[9][3], MD[9][3]; gp_D(ids, y, D); gp_MD(D, MD);
        for (int t=0;t<9;t++) for (int a=0;a<3;a++) e += 0.5*(double)k*(double)D[t][a]*(double)MD[t][a];
    }
    blocksum_add(e, &PH[1]);
}
'''


def _append(src):
    if "gp_force(" in src:
        return src
    assert "blocksum_add(" in src and "Tapply(" in src, "ghost: 바탕 소스에 blocksum_add·Tapply 없음"
    i = src.rstrip().rfind("}")
    return src[:i] + GP_SRC + "\n}\n"


def _k_of(s):
    """k = γ_a h, γ_a = (2μ + λ) 10^-4, h = |det A|^{1/3}."""
    if getattr(s, "_gpk", None) is None:
        A = s.A.double().reshape(3, 3).cpu()
        h = abs(float(torch.linalg.det(A))) ** (1.0 / 3.0)
        kap = 2.0 * float(s.mu) + float(s.lam)
        s._gpk = GAMMA_A * kap * h
        s._gph, s._gpkap = h, kap
        print(f"[절 CM47 ghost] γ_a = (2μ+λ)·{GAMMA_A:g} (μ {float(s.mu):.6g}, λ {float(s.lam):.6g}, 2μ+λ {kap:.6g}), h {h:.6g} m, k = γ_a h = {s._gpk:.6g}", flush=True)
    return s._gpk


def _alloc(s):
    cap = int(s.capQc)
    if getattr(s, "gpcut", None) is None or s.gpcut.numel() < cap:
        s.gpcut = torch.zeros(cap, device=s.dev, dtype=torch.int32)
        s.gpfl = torch.zeros(3 * cap, device=s.dev, dtype=torch.int32)
    if getattr(s, "gpcnt", None) is None:
        s.gpcnt = torch.zeros(4, device=s.dev, dtype=torch.int32)
        s.gpGQ = torch.zeros(4, device=s.dev, dtype=torch.float64)


def install():
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_an_rt_v0 as V0
    P = ART.P
    ST["on"] = True
    _obs, _oas = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: _append(_obs(*a, **kw))
    AORT._src = lambda *a, **kw: _append(_oas(*a, **kw))
    f32, i32 = np.float32, np.int32

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            r = _ok(s, name, n, args, b)
            if getattr(s, "_gpin", False) or name.startswith("gp_"):
                return r
            s._gpin = True
            try:
                _hook(s, _ok, name, args)
            finally:
                s._gpin = False
            return r
        cls.k = k

    def _hook(s, _ok, name, args):
        if name == "rt_gauss":                                                           # 칸 압축·cidx·cb 가 찬 뒤: 경계 칸·면 목록 (프레임마다)
            assert not getattr(s, "det", False), "ghost: 결정적 판 (det) 은 안 고침"
            assert getattr(s, "pkmod", None) is None, "ghost: 지속형 판은 안 고침"
            _alloc(s)
            _k_of(s)
            cq = int(s.capQc)
            _ok(s, "gp_zero", 1, (P(s.gpcnt),), 1)
            _ok(s, "gp_cut", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.gpcut), P(s.gpcnt)), 128)
            _ok(s, "gp_face", 3 * cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.gpcut), P(s.gpcnt), P(s.gpfl)), 128)
            if not ST["said"]:
                ST["said"] = True
                print("[절 CM47 ghost] 켬: 경계 칸·면 목록 (rt_gauss 뒤), 요소·곱·에너지 커널 뒤 G 몫", flush=True)
            return
        if getattr(s, "gpcnt", None) is None:
            return
        k = f32(s._gpk)
        nf = 3 * int(s.capQc)
        G = (P(s.IP), P(s.cb), P(s.gpcnt), P(s.gpfl), k)
        if name in ELEM:
            if name == "rt_elem":
                mode = int(args[8]); y, E, f_, B = args[5], args[9], args[10], args[13]
                _ok(s, "gp_force", nf, G + (y, f_ if mode & 1 else P(None), B if mode & 8 else P(None), E if mode & 4 else P(None)), 128)
            elif name == "aq_elem":
                _ok(s, "gp_force", nf, G + (args[5], args[10], args[13], P(None)), 128)
            else:
                mode = int(args[9])
                _ok(s, "gp_force", nf, G + (args[5], args[10] if mode & 1 else P(None), args[13] if mode & 8 else P(None), P(None)), 128)
        elif name in MV:
            _ok(s, "gp_mv", nf, G + (args[6], args[7]), 128)
        elif name == "rt_bd":
            _ok(s, "gp_force", nf, G[:4] + (k, P(s.y), P(None), args[6], P(None)), 128)
        elif name == "rt_elem_lsN":
            yh, eta, dl, sk, pm, n0, n1, n2, SC, NC, i0, gate = args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[15], args[16], args[17], args[18]
            _ok(s, "gp_ls", nf, G[:4] + (yh, eta, dl, sk, pm, n0, n1, n2, P(s.gpGQ), gate), 128)
            _ok(s, "gp_lsfin", 1, (SC, NC, i0, gate, P(s.gpGQ), k), 1)
        elif name == "en_phi_q":
            _ok(s, "gp_phi", nf, G + (args[6], args[19], args[20]), 128)
        elif name == "rt_elem_ls":
            raise RuntimeError(f"ghost: {name} 경로는 안 고침")

    for cls in (ART.RTSolver, V0.RTSolver):
        wrap_k(cls)
