"""절 CM132 묶음 extfix (조정자 11-13, 사용자 wolf 완전 위임): 확장 B-스플라인 T1E2 (절 CM81 fem_wolfh8) 의 외부 노드 정의 두 가지 고침 -- 새 플래그, 기본 0 (기본 판 불변).

배경 (절 CM131): 예산 안 H9 의 걸음 잔차 정체는 확장 주인 노드에 접힌 잔차 (끝 몫 0.999). 묶인 외부 노드가 무겁고 (질량비 0.3 이상, 받침 입자 48~3300)
  라그랑주 외삽 가중 (축마다 (1, −3, 3) -> |c| ≤ 27, Σc² ≤ 2800) 으로 주인 노드에 접혀 옴.
(다) --xfl 1  바닥 평면이 자른 칸: 점유를 바닥 위 부분만으로 잼 -- 내부 칸 판정 (T1: 점유 칸 중 26 이웃 칸이 모두 점유, w8_icell2) 에서
       이웃 칸이 비었거나 상자 밖이어도 그 칸이 통째로 바닥 아래 (꼭짓점 8 개 모두 n·x − p·n ≤ 0) 이면 빈 칸으로 세지 않음 (적분 영역 = 바닥 위).
       바닥 평면이 자른 칸은 원래대로 그 칸의 입자 (모두 바닥 위) 로 점유를 잼. 칸 좌표 = 실행기 정의 그대로 (격자 좌표 L ∈ [b+½, b+1½)³, 세계 = A L).
(가) --xls 1|2  외부 노드 o 의 확장 계수를 라그랑주 외삽 대신 P2G 와 같은 입자·같은 가중의 최소제곱 사영으로:
       c_o = argmin_c Σ_p m_p (N_o(x_p) − Σ_j c_j N_j(x_p))²,  p = 모든 입자 (프레임 시작 위치, P2G 의 pb·fx·m 그대로), j = 묶인 내부 칸 블록 27 노드 (묶는 칸 선택은 w8_class 그대로).
       정규 방정식 G c = b, G_jk = M_jk · b_j = M_oj (M = P2G 가중의 일관 질량 행렬, 필요한 노드 쌍만 f64 원자 합) 을 대각 피벗 LDLᵀ 로 풂 -- 피벗이 수치 0 (≤ 27 ε_f64 max diag, LAPACK pstrf 기본 판정) 이면
       그 기저는 사영에 안 씀 (계수 0, 기본 해). 입자가 닿지 않는 블록 노드는 G 행이 정확히 0 이라 계수 0.
       계수는 27 개 (텐서 곱 아님) -> 확장 가중 버퍼 w8w 를 노드당 27 로 바꾸고 w8_fold · w8_expand 를 27 가중으로 (같은 접기·펼치기 식, 가중만 다름).
       w8_class 는 그 자리에 라그랑주 텐서 곱 27 개를 먼저 씀 (xls 2 의 특수해).
       xls 1 = 위 순수 사영 (지시 그대로). 순수 사영은 Σ_j c_j ≠ 1 이라 몸 전체 평행 이동·균일 변형을 외부 노드가 따라가지 못함 (안쪽이 δ 만큼 움직이면 외부는 (Σc) δ).
       xls 2 = 같은 최소제곱을 아핀 재현 제약 (Σ c_j = 1, Σ c_j ξ_j = ξ_o: 평행 이동·회전·균일 변형을 그대로 옮김) 아래에서: c = c_L + Z z,
         c_L = 라그랑주 (제약을 만족하는 특수해), Z = 제약 영공간 (27x23 상수); 탈락 방향은 라그랑주 그대로. 2 차 재현 (라그랑주) 만 버림.
훅: 감싸개 (exe/extfix_wrap.py) 가 import 때 fem_wolfh8.install 을 감쌈 -> wolfh8 설치 직전에 RTSolver.k 를 안쪽에서 감싸
     (wolfh8 의 w8_icell2·w8_class 호출이 이 감쌈을 지남): w8_icell2 -> xf_icell2 (xfl), w8_class 뒤 xf_fz·xf_mark·xf_mz·xf_mass·xf_solve (xls).
     모두 걸음 시작 ctl(1) 안 (그래프 안, 호스트 동기화 없음). 새 실행기 (fem_an_rt RTSolver) 만.
기록: xfinfo.json (설정·호출 수), xf.jsonl (걸음 시작마다: xfl 바닥 아래 이웃 덕에 내부가 된 칸 수 · xls 외부 노드 수·Σc² 평균·최대·|c| 최대 (최소제곱 / 라그랑주)·피벗 탈락 노드 수·평균 계수·넘침).
"""
import atexit
import json
import os
import weakref

import numpy as np
import torch

ST = dict(fl=0, ls=0, out=".", nsrc=0, nicell=0, nls=0, nalloc=0, nlog=0, said=False, err=None)
MARK = "// ---- 절 CM132 extfix"
HCAP = 8192
NH = 16
_SOL = []

XF_SRC = r'''
// ---- 절 CM132 extfix (lib/anchorflow/fem_extfix.py)
__device__ __forceinline__ int xf_below(const int* IP, const float* A, int a, int b, int d, float n0, float n1, float n2, float pn) {   // 칸 (상자 상대 base a,b,d) 이 통째로 바닥 아래
    for (int o=0;o<8;o++) {
        float L0 = (float)(a + IP[0]) + 0.5f + (float)((o >> 2) & 1), L1 = (float)(b + IP[1]) + 0.5f + (float)((o >> 1) & 1), L2 = (float)(d + IP[2]) + 0.5f + (float)(o & 1);
        float x0 = A[0]*L0 + A[1]*L1 + A[2]*L2, x1 = A[3]*L0 + A[4]*L1 + A[5]*L2, x2 = A[6]*L0 + A[7]*L1 + A[8]*L2;
        if (x0*n0 + x1*n1 + x2*n2 - pn > 0.f) return 0; }
    return 1;
}
__global__ void xf_icell2(const int* IP, const int* cb, const int* occ, int* ic, int* cnt, int face, const float* A, float n0, float n1, float n2, float pn, double* XS) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n0_ = IP[6]-2, n1_ = IP[7]-2, n2_ = IP[8]-2;
    int a0 = cb[c*3], b0 = cb[c*3+1], c0_ = cb[c*3+2], nbl = 0;
    for (int a=a0-1;a<=a0+1;a++) for (int b=b0-1;b<=b0+1;b++) for (int d=c0_-1;d<=c0_+1;d++) {
        if (face && (a != a0) + (b != b0) + (d != c0_) > 1) continue;
        int inb = !(a < 0 || b < 0 || d < 0 || a >= n0_ || b >= n1_ || d >= n2_);
        if (inb && occ[(a*n1_ + b)*n2_ + d]) continue;
        if (xf_below(IP, A, a, b, d, n0, n1, n2, pn)) { nbl++; continue; }
        return; }
    ic[(a0*n1_ + b0)*n2_ + c0_] = 1;
    atomicAdd(&cnt[3], 1);
    if (nbl) atomicAdd(&XS[0], 1.0);
}
// (가) 일관 질량 행렬 M_ab = Σ_p m_p N_a(x_p) N_b(x_p) (P2G 와 같은 입자·가중) 을 필요한 노드 (외부 노드 + 그 묶인 블록 27 노드) 쌍만 모음.
//   저장: 노드 a 의 행에 b − a ∈ [−2, 2]³ 인 125 칸 (a < b 인 쌍만, 읽을 때 대칭), f64 원자 합.
__global__ void xf_fz(const int* IP, int* fl) { int n = blockIdx.x*blockDim.x + threadIdx.x; if (n < IP[9]) fl[n] = 0; }
__global__ void xf_mark(const int* IP, const int* base, int* fl) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13] || base[n] < 0) return;
    fl[n] = 1;
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) for (int c=0;c<3;c++) fl[b0 + (a*nb1 + b)*nb2 + c] = 1;
}
__global__ void xf_mz(const int* IP, const int* fl, double* Mc) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || !fl[n]) return;
    for (int i=0;i<125;i++) Mc[(size_t)n*125 + i] = 0.0;
}
__device__ __forceinline__ int xf_off(int di, int dj, int dk) { return ((di+2)*5 + (dj+2))*5 + (dk+2); }
__global__ void xf_mass(int N, const int* IP, const int* pb, const float* fx, const float* m, const int* fl, double* Mc) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float wx[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], wx[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    int p0 = pb[p*3], p1 = pb[p*3+1], p2 = pb[p*3+2];
    double mp = (double)m[p];
    for (int u=0;u<27;u++) {
        int i = u/9, j = (u/3)%3, k = u%3;
        int na = ((p0+i)*nb1 + (p1+j))*nb2 + (p2+k);
        if (!fl[na]) continue;
        double wa = mp*(double)wx[0][i]*(double)wx[1][j]*(double)wx[2][k];
        if (wa == 0.0) continue;
        for (int v=u;v<27;v++) {                                                           // 스텐실 순서 = 노드 번호 순서 (u ≤ v 이면 na ≤ nb)
            int i2 = v/9, j2 = (v/3)%3, k2 = v%3;
            int nb_ = ((p0+i2)*nb1 + (p1+j2))*nb2 + (p2+k2);
            if (!fl[nb_]) continue;
            double w = wa*(double)wx[0][i2]*(double)wx[1][j2]*(double)wx[2][k2];
            if (w == 0.0) continue;
            atomicAdd(&Mc[(size_t)na*125 + xf_off(i2-i, j2-j, k2-k)], w);
        }
    }
}
__device__ __forceinline__ double xf_M(const double* Mc, int nb1, int nb2, int a, int b) {   // 노드 a, b 의 일관 질량 (축 거리 > 2 면 0)
    if (a > b) { int t = a; a = b; b = t; }
    int ai = a/(nb1*nb2), aj = (a/nb2)%nb1, ak = a%nb2, bi = b/(nb1*nb2), bj = (b/nb2)%nb1, bk = b%nb2;
    int di = bi-ai, dj = bj-aj, dk = bk-ak;
    if (di < -2 || di > 2 || dj < -2 || dj > 2 || dk < -2 || dk > 2) return 0.0;
    return Mc[(size_t)a*125 + xf_off(di, dj, dk)];
}
__device__ __forceinline__ void xf_amax(double* a, double v) {                       // v ≥ 0
    unsigned long long* q = (unsigned long long*)a; unsigned long long o = *q, nv = (unsigned long long)__double_as_longlong(v);
    while (__longlong_as_double((long long)o) < v) { unsigned long long r = atomicCAS(q, o, nv); if (r == o) break; o = r; }
}
// XS: 0 xfl 칸 수 · 1 외부 노드 (풂) · 2 Σ Σc² (LS) · 3 max Σc² (LS) · 4 max |c| (LS) · 5 Σ Σc² (라그랑주) · 6 max Σc² (라그랑주) · 7 max |c| (라그랑주)
//     8 피벗 탈락 노드 (계수 < 양의 대각 수) · 9 Σ 계수 · 10 Σ 양의 대각 수 · 11 (안 씀) · 12 Σ (Σ_j c_j − 1)² (분할 단위 어긋남) · 13 max |Σ_j c_j − 1|
// mode 1: 순수 최소제곱 (G c = b 의 기본 해, 탈락 방향 계수 0). mode 2: 아핀 재현 제약 (Σc = 1, Σ c ξ_j = ξ_o) 아래 최소제곱 -- c = c_L + Z z
//   (c_L = 라그랑주 = 특수해, Z = 제약 영공간 정규직교 기저 27x23 (상수, 파이썬 SVD)), (Zᵀ G Z) z = Zᵀ (b − G c_L), 탈락 방향 z = 0 (라그랑주 그대로).
__global__ void xf_solve(const int* IP, const int* base, const double* Mc, float* w, double* XS, int mode, const double* Z) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || IP[13] || base[n] < 0) return;
    float* W = w + (size_t)n*27;
    double cl[27], sl2 = 0.0, slm = 0.0;
    for (int l=0;l<27;l++) { cl[l] = W[l]; sl2 += cl[l]*cl[l]; slm = fmax(slm, fabs(cl[l])); }
    int nb1 = IP[7], nb2 = IP[8], b0 = base[n];
    int bn[27];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) for (int c=0;c<3;c++) bn[(a*3+b)*3+c] = b0 + (a*nb1 + b)*nb2 + c;
    double M[27][27], bb[27], cc[27];
    int piv[27]; unsigned char used[27];
    int nv = (mode == 2) ? 23 : 27;
    if (mode == 2) {
        double Gf[27][27], gz[27][23], rb[27];
        for (int u=0;u<27;u++) for (int v=u;v<27;v++) { double q = xf_M(Mc, nb1, nb2, bn[u], bn[v]); Gf[u][v] = q; Gf[v][u] = q; }
        for (int u=0;u<27;u++) { double t = xf_M(Mc, nb1, nb2, n, bn[u]); for (int v=0;v<27;v++) t -= Gf[u][v]*cl[v]; rb[u] = t; }   // b − G c_L
        for (int u=0;u<27;u++) for (int a=0;a<23;a++) { double t = 0.0; for (int v=0;v<27;v++) t += Gf[u][v]*Z[v*23 + a]; gz[u][a] = t; }
        for (int a=0;a<23;a++) { double t = 0.0; for (int u=0;u<27;u++) t += Z[u*23 + a]*rb[u]; bb[a] = t;
            for (int c=a;c<23;c++) { double q = 0.0; for (int u=0;u<27;u++) q += Z[u*23 + a]*gz[u][c]; M[a][c] = q; M[c][a] = q; } }
    } else {
        for (int u=0;u<27;u++) { bb[u] = xf_M(Mc, nb1, nb2, n, bn[u]); for (int v=u;v<27;v++) { double q = xf_M(Mc, nb1, nb2, bn[u], bn[v]); M[u][v] = q; M[v][u] = q; } }
    }
    double dmax = 0.0; int nd = 0;
    for (int u=0;u<nv;u++) { used[u] = 0; cc[u] = 0.0; if (M[u][u] > 0.0) nd++; dmax = fmax(dmax, M[u][u]); }
    double tol = (double)nv*2.220446049250313e-16*dmax;
    int r = 0;
    if (dmax > 0.0) for (int it=0; it<nv; it++) {
        int j = -1; double dj = tol;
        for (int i=0;i<nv;i++) if (!used[i] && M[i][i] > dj) { dj = M[i][i]; j = i; }
        if (j < 0) break;
        used[j] = 1; piv[r++] = j;
        double d = M[j][j];
        for (int i=0;i<nv;i++) { if (used[i]) continue;
            double l = M[i][j]/d;
            if (l != 0.0) { for (int k=0;k<nv;k++) if (!used[k]) M[i][k] -= l*M[j][k]; bb[i] -= l*bb[j]; }
            M[i][j] = l; }
    }
    for (int t=r-1;t>=0;t--) { int j = piv[t]; double x = bb[j]/M[j][j];
        for (int q=t+1;q<r;q++) x -= M[piv[q]][j]*cc[piv[q]];
        cc[j] = x; }
    double co[27];
    if (mode == 2) { for (int u=0;u<27;u++) { double t = cl[u]; for (int a=0;a<23;a++) t += Z[u*23 + a]*cc[a]; co[u] = t; } }
    else for (int u=0;u<27;u++) co[u] = cc[u];
    double s2 = 0.0, sm = 0.0, su = 0.0;
    for (int l=0;l<27;l++) { W[l] = (float)co[l]; s2 += co[l]*co[l]; sm = fmax(sm, fabs(co[l])); su += co[l]; }
    atomicAdd(&XS[1], 1.0); atomicAdd(&XS[2], s2); xf_amax(&XS[3], s2); xf_amax(&XS[4], sm);
    atomicAdd(&XS[5], sl2); xf_amax(&XS[6], sl2); xf_amax(&XS[7], slm);
    if (r < nd) atomicAdd(&XS[8], 1.0);
    atomicAdd(&XS[9], (double)r); atomicAdd(&XS[10], (double)nd);
    atomicAdd(&XS[12], (su - 1.0)*(su - 1.0)); xf_amax(&XS[13], fabs(su - 1.0));
}
__global__ void xf_log(double* XS, const int* ns, int* hp, double* H, int cap) {          // 걸음 시작마다 한 줄 (그래프 안), 다음 걸음 위해 0
    int s = hp[0]; hp[0] = s + 1;
    if (s < cap) { for (int i=0;i<14;i++) H[s*16 + i] = XS[i]; H[s*16 + 14] = ns ? (double)ns[0] : -1.0; H[s*16 + 15] = ns ? (double)ns[1] : -1.0; }
    for (int i=0;i<16;i++) XS[i] = 0.0;
}
'''


def add_src(src):
    if MARK in src:
        return src
    assert "bwf(" in src, "extfix: bwf 없는 모듈"
    ST["nsrc"] += 1
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + XF_SRC + "\n}\n"


def patch_w8_src(src):
    """(가) 판: w8w 를 노드당 27 가중으로 (w8_class 는 라그랑주 텐서 곱 27 개를 씀, w8_fold·w8_expand 는 27 가중을 읽음)."""
    reps = [
        ("w8_wts(xs[0], dg, w + n*9); w8_wts(xs[1], dg, w + n*9 + 3); w8_wts(xs[2], dg, w + n*9 + 6);",
         "{ float t9[9]; w8_wts(xs[0], dg, t9); w8_wts(xs[1], dg, t9 + 3); w8_wts(xs[2], dg, t9 + 6);\n"
         "      for (int a=0;a<3;a++) for (int b=0;b<3;b++) for (int c=0;c<3;c++) w[(size_t)n*27 + (a*3+b)*3+c] = t9[a]*t9[3+b]*t9[6+c]; }   // 절 CM132 xls: 27 가중", 1),
        ("const float* W = w + n*9;", "const float* W = w + (size_t)n*27;", 2),
        ("for (int a=0;a<3;a++) { if (W[a] == 0.f) continue;", "for (int a=0;a<3;a++) {", 2),
        ("for (int b=0;b<3;b++) { if (W[3+b] == 0.f) continue;", "for (int b=0;b<3;b++) {", 2),
        ("float ww = W[a]*W[3+b]*W[6+c];", "float ww = W[(a*3+b)*3+c];", 2),
    ]
    for a, b, c in reps:
        assert src.count(a) == c, f"extfix: W8_SRC 고칠 글자 수 {src.count(a)} != {c}: {a[:60]}"
        src = src.replace(a, b)
    assert "n*9" not in src, "extfix: W8_SRC 에 n*9 가 남음"
    return src


def affine_null():
    """아핀 재현 제약 행렬 C (4x27: 1, ξa, ξb, ξc; 블록 노드 (a,b,c) 의 ξ = (a−1, b−1, c−1), 지역 번호 (a*3+b)*3+c) 의 영공간 정규직교 기저 Z (27x23, 행 우선)."""
    xi = np.array([[a - 1, b - 1, c - 1] for a in range(3) for b in range(3) for c in range(3)], dtype=np.float64)
    C = np.concatenate([np.ones((27, 1)), xi], 1).T
    _, sv, vt = np.linalg.svd(C)
    assert (sv > 1e-9).sum() == 4
    Z = vt[4:].T
    assert Z.shape == (27, 23) and np.abs(C @ Z).max() < 1e-12
    return Z


def P(t):
    return np.uint64(0 if t is None else t.data_ptr())


def _on(s):
    return getattr(s, "w8b", None) is not None and getattr(s, "_w8on", None) is not None


def install(fl=0, ls=0, out="."):
    """감싸개 import 때. fem_wolfh8.install 을 감싸 그 직전에 RTSolver.k (안쪽)·build_src, 직후에 alloc 을 감쌈."""
    from . import fem_wolfh8 as W8
    ST.update(fl=int(fl), ls=int(ls), out=out)
    if not (ST["fl"] or ST["ls"]):
        return
    w8_install = W8.install

    def w8_install_x(out_, D=2, T=0, X=0):
        from . import fem_an_rt as ART
        R = ART.RTSolver
        assert int(T) in (1, 2) or not ST["fl"], "extfix --xfl: 내부 칸 정의 T1·T2 (w8_icell2) 만"
        ob = ART.build_src
        ART.build_src = lambda *a, **kw: add_src(ob(*a, **kw))
        if ST["ls"]:
            W8.W8_SRC = patch_w8_src(W8.W8_SRC)
        ok0 = R.k

        def k(s, name, n, args, b=128):
            if not _on(s):
                return ok0(s, name, n, args, b)
            i32, f32 = np.int32, np.float32
            if name == "w8_icell2" and ST["fl"]:
                FL = s.floor
                assert FL is not None, "extfix --xfl: 바닥 씬만"
                nr = [float(q) for q in FL["NR"]]
                r = ok0(s, "xf_icell2", n, tuple(args) + (P(s.A), f32(nr[0]), f32(nr[1]), f32(nr[2]), f32(FL["PN"]), P(s.xfXS)), b)
                ST["nicell"] += 1
                return r
            r = ok0(s, name, n, args, b)
            if name == "w8_class" and ST["ls"]:
                ok0(s, "xf_fz", s.capN, (P(s.IP), P(s.xffl)), 256)
                ok0(s, "xf_mark", s.capN, (P(s.IP), P(s.w8b), P(s.xffl)), 256)
                ok0(s, "xf_mz", s.capN, (P(s.IP), P(s.xffl), P(s.xfM)), 256)
                ok0(s, "xf_mass", s.N, (i32(s.N), P(s.IP), P(s.Np), P(s.fx), P(s.m), P(s.xffl), P(s.xfM)), 128)
                ok0(s, "xf_solve", s.capN, (P(s.IP), P(s.w8b), P(s.xfM), P(s.w8w), P(s.xfXS), i32(ST["ls"]), P(s.xfZ)), 64)
                ST["nls"] += 1
            if name == "w8_class":
                ok0(s, "xf_log", 1, (P(s.xfXS), np.uint64(0), P(s.xfhp), P(s.xfH), i32(HCAP)), 1)
                ST["nlog"] += 1
                if not ST["said"]:
                    ST["said"] = True
                    print(f"[절 CM132 extfix] 걸음 시작 훅 -- (다) 바닥 아래 칸 빈 칸 아님 {ST['fl']} · (가) 최소제곱 사영 계수 {ST['ls']} (입자 {s.N}, 일관 질량 행 {s.capN} x 125 f64)", flush=True)
                    _dump()
            return r
        R.k = k
        w8_install(out_, D=D, T=T, X=X)
        oa = R.alloc

        def alloc(s, need):
            r = oa(s, need)
            if getattr(s, "w8b", None) is not None:
                dev = s.dev
                if ST["ls"]:
                    s.w8w = torch.zeros(s.capN * 27, device=dev, dtype=torch.float32)
                    s.xffl = torch.zeros(s.capN, device=dev, dtype=torch.int32)
                    s.xfM = torch.zeros(s.capN * 125, device=dev, dtype=torch.float64)
                    s.xfZ = torch.tensor(affine_null(), device=dev, dtype=torch.float64).contiguous()
                s.xfXS = torch.zeros(16, device=dev, dtype=torch.float64)
                if getattr(s, "xfH", None) is None:
                    s.xfH = torch.zeros(HCAP, NH, device=dev, dtype=torch.float64)
                    s.xfhp = torch.zeros(1, device=dev, dtype=torch.int32)
                    _SOL.append(weakref.ref(s))
                ST["nalloc"] += 1
            return r
        R.alloc = alloc
        print(f"[절 CM132] extfix 켬 -- (다) xfl {ST['fl']} · (가) xls {ST['ls']} (wolfh8 차수 {D} 내부 칸 정의 {T})", flush=True)

    W8.install = w8_install_x
    atexit.register(_dump)


def _dump():
    try:
        rows = []
        for wr in _SOL:
            s = wr()
            if s is None:
                continue
            n = int(s.xfhp.item())
            H = s.xfH[:min(n, HCAP)].cpu().numpy()
            for i, h in enumerate(H):
                no = h[1]
                rows.append(dict(k=i, icell_floor=int(h[0]), outer=int(no), ls_s2_mean=(h[2] / no if no > 0 else None), ls_s2_max=h[3], ls_cmax=h[4],
                                 lag_s2_mean=(h[5] / no if no > 0 else None), lag_s2_max=h[6], lag_cmax=h[7], piv_drop=int(h[8]),
                                 rank_mean=(h[9] / no if no > 0 else None), nd_mean=(h[10] / no if no > 0 else None),
                                 pu_rms=(float(np.sqrt(h[12] / no)) if no > 0 else None), pu_max=h[13]))
        if rows:
            with open(os.path.join(ST["out"], "xf.jsonl"), "w") as f_:
                for r_ in rows:
                    f_.write(json.dumps(r_) + "\n")
    except Exception as e_:
        ST["err"] = f"{type(e_).__name__}: {e_}"
    try:
        with open(os.path.join(ST["out"], "xfinfo.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass
