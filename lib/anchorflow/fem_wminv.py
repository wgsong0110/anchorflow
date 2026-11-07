"""절 CM29 (조정자 위임 11-08, 위임 범위 안 풀이 방식 변경): 수박 첫 접촉의 전역 뒤집힘 필터 (--ls_inv, 절 AS) 를 국소로.
옛 재질 실행기 (fem_ao_rt.AORTSolver) 에만, s.wf["wi"] = dict(mode, R, kj, rec) 가 있을 때만 -- 없으면 소스·실행 글자 그대로.

mode "node"  (a) 노드별 보폭: 원래 필터는 가우스 점마다 [0, 2^-k] 에서 J > 0 인 최소 k_q 를 구해 몸 전체 보폭 2^-max k_q.
             여기서는 노드 n 의 보폭 2^-kn (kn = 걸친 가우스 점 k_q 의 최대) -- 나머지 노드는 온 보폭. 노드마다 다른 보폭이면
             가우스 점의 방향이 바뀌므로 확인 라운드 R 번: 바뀐 방향 (2^-kn ∘ T δ) 으로 같은 판정 (wi_inv_q, --ls_end 끝 변위 포함) 을 다시 해
             아직 어기는 점 q 는 그 점 27 노드의 kn 에 k_q 를 더함 (그 점에서는 [0, 2^-k_q] 경로와 같아 안전). R 번 뒤에도 어기면 (최종 확인)
             그 반복은 원래 전역 보폭 (모든 노드 2^-max k_q, 첫 판정 값) 으로 되돌림. 장벽 실현 가능 보폭 (rt_bar_kfe) 은 국소 보폭을 준
             후보 y + 2^-kn ∘ T δ 위에서 전역 2^-kb 로 곱함 (원래는 온 걸음 후보에서 구한 2^-kb 와 필터 보폭의 작은 쪽).
             SC[14] (덜 풀린 걸음 판정·진단) = 노드 보폭 최소 2^-(max kn + kb). 자른 노드가 없으면 원래와 같은 산술 (같은 비트).
mode "jbar"  (b) 필터 대신 J 로그 장벽 (절 CM13 판 (1) 식 aq_jbar_p·aq_jbar_t, κ_J = kj) 을 표시된 가우스 점에만: 뉴턴 반복마다 그 반복 방향으로
             원래 필터 판정 (보폭은 자르지 않음) 을 해 k_q > 0 인 점을 표시, 표시는 그 프레임 (시도) 끝까지 유지 -- 다음 요소 계산부터 장벽.
             프레임 마지막 반복 뒤 표시를 지움 (가우스 점 번호는 프레임마다 새로 짜임). 실행기는 --ls_inv 없이 (필터 끔).
rec          진단 (결과 불변): 반복마다 원래 필터가 k_q > 0 으로 거는 가우스 점 -- (걸음·반복, q, k, 이 반복 시작 J, 온 걸음 J, [0,1] 최소 J,
             V_q, 27 노드 질량 최소·합, 팔분 표본점 질량·위치 (프레임 시작 칸 입자 질량 중심), det F_q) 와 반복별 수·최대 k. 시도마다 그 프레임
             기록을 s.wi_att 에 (rec_frames 프레임만).
"""
import numpy as np
import torch

from .fem_ao_rt import P

REC_W = 15                                                                            # 기록 한 줄 float 수
REC_CAP = 1 << 17
NCODE = 4096                                                                          # 걸음·반복 칸 (kstep * nt + it)

# 커널 이름은 _skipify 의 정규식 (__global__ void) 에 안 걸리게 WI_G 로 -- 프레임 끝 표시 지우기·시도 시작 비우기는 허용치로 멈춘 뒤에도 돌아야 함
WI_HEAD = r"""
// ---- 절 CM29 (fem_wminv): 뒤집힘 필터 국소화 ----
#define WI_G __global__
__device__ int* WI_M;
"""

WI_SRC = r"""
// ---- 절 CM29 커널 ----
__device__ __forceinline__ void wi_cub(const int* ids, int oc, const float* BG, const float* F, const float* yb, const float* d, double* c) {
    float X0f[9], Dxf[9]; aw_X(ids, oc, BG, F, yb, 1, X0f); aw_X(ids, oc, BG, F, d, 0, Dxf);
    double X0[9], Dx[9]; for (int k=0;k<9;k++) { X0[k] = (double)X0f[k]; Dx[k] = (double)Dxf[k]; }
    double C0[9], CD[9]; aw_cof(X0, C0); aw_cof(Dx, CD);
    c[0] = X0[0]*C0[0] + X0[1]*C0[1] + X0[2]*C0[2]; c[3] = Dx[0]*CD[0] + Dx[1]*CD[1] + Dx[2]*CD[2]; c[1] = 0.0; c[2] = 0.0;
    for (int k=0;k<9;k++) { c[1] += C0[k]*Dx[k]; c[2] += CD[k]*X0[k]; }
}
// 진단: k_q > 0 인 점 기록 (원래 필터가 거는 점)
__global__ void wi_rec(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* yb, const float* d, const float* mn, const int* kq,
                       const float* bx, int code, int* cnt, float* R, int cap, int* ck) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int k = kq[q]; if (k <= 0) return;
    atomicAdd(&ck[code*2], 1); atomicMax(&ck[code*2+1], k);
    int i = atomicAdd(cnt, 1); if (i >= cap) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int a=0;a<9;a++) F[a] = Fq[(size_t)q*9+a];
    double c[4]; wi_cub(ids, oc, BG, F, yb, d, c);
    double jm = c[0]; for (int t=1;t<=64;t++) { double s_ = t/64.0; double v_ = aw_p3(c[0], c[1], c[2], c[3], s_); jm = fmin(jm, v_); }
    float mmin = 3e38f, msum = 0.f; for (int o=0;o<27;o++) { float m_ = mn[ids[o]]; mmin = fminf(mmin, m_); msum += m_; }
    float dF = F[0]*(F[4]*F[8]-F[5]*F[7]) - F[1]*(F[3]*F[8]-F[5]*F[6]) + F[2]*(F[3]*F[7]-F[4]*F[6]);
    float* o = R + (size_t)i*15;
    o[0] = (float)code; o[1] = __int_as_float(q); o[2] = (float)k; o[3] = (float)c[0]; o[4] = (float)(c[0]+c[1]+c[2]+c[3]); o[5] = (float)jm; o[6] = Vq[q];
    o[7] = mmin; o[8] = msum; float bm = bx ? bx[q*4+3] : 0.f; o[9] = bm;
    for (int a=0;a<3;a++) o[10+a] = (bx && bm > 0.f) ? bx[q*4+a]/bm : 0.f;
    o[13] = dF; o[14] = 0.f;
}
// (a) 노드 k: k_q > 0 인 점의 27 노드에 (kn + k_q) 의 최대 -> kn2
__global__ void wi_nk(const int* IP, const int* cb, const int* kq, const int* kn, int* kn2) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int k = kq[q]; if (k <= 0) return;
    int ids[27]; qnodes(q, IP, cb, ids);
    for (int o=0;o<27;o++) atomicMax(&kn2[ids[o]], (kn ? kn[ids[o]] : 0) + k);
}
__global__ void wi_merge(const int* IP, int* kn, int* kn2, int* km) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    int a = min(max(kn[n], kn2[n]), 60); kn[n] = a; kn2[n] = 0;
    if (a > 0) atomicMax(km, a);
}
__global__ void wi_dsc(const int* IP, const int* kn, const float* td, float* o) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float s_ = ldexpf(1.f, -kn[n]); for (int a=0;a<3;a++) o[n*3+a] = s_*td[n*3+a];
}
// 최종 확인이 어기면 (kk[0] > 0) 원래 전역 보폭 kg 로 되돌림; 자른 노드 수 셈
__global__ void wi_fb(const int* IP, const int* kk, const int* kg, int* kn, int* km, int* st) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    bool fb = kk[0] > 0;
    if (fb) kn[n] = kg[0];
    if (n == 0) { if (fb) { km[0] = kg[0]; st[1] += 1; } if (kg[0] > 0) st[0] += 1; st[2] = max(st[2], fb ? kg[0] : km[0]); }
    if (kn[n] > 0) atomicAdd(&st[3], 1);
}
__global__ void wi_ytry(const int* IP, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* eta, const float* dl, const int* kn, float* y1) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float s_ = ldexpf(1.f, -kn[n]);
    float e[3]; for (int a=0;a<3;a++) e[a] = eta[n*3+a] + s_*dl[n*3+a];
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y1[n*3+a] = yh[n*3+a] + t[a];
}
// SC[14] = 노드 보폭 최소 (덜 풀린 걸음 판정·진단), si[0] = 원래 전역 필터 보폭 (진단)
__global__ void wi_sc(double* SC, const int* km, const int* kb, const int* kg, double* si, int* st) {
    SC[14] = (double)ldexpf(1.f, -(km[0] + kb[0])); si[0] = ldexp(1.0, -kg[0]); st[6] += 1;
}
// 갱신 = rt_lsup (full 0) 과 같은 식, 보폭만 노드별 2^-(kn + kb); cmp 1 = 압축 판 (AO_ACT 노드만, rt_lsup_c 와 같음)
__global__ void wi_lsup(const int* IP, const int* kn, const int* kb, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, float* eta, const float* dl, float* y, int cmp) {
    int n;
    if (cmp) { int ii_ = blockIdx.x*blockDim.x + threadIdx.x; if (ii_ >= IP[14]) return; n = AO_ACT[ii_]; }
    else { n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= IP[9]) return; }
    float st = ldexpf(1.f, -(kn[n] + kb[0]));
    float e[3]; for (int a=0;a<3;a++) { e[a] = eta[n*3+a] + st*dl[n*3+a]; eta[n*3+a] = e[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y[n*3+a] = yh[n*3+a] + t[a];
}
// (b) 표시
__global__ void wi_mark(const int* IP, const int* kq, int* M, int* st) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    if (kq[q] > 0) { if (M[q] == 0) atomicAdd(&st[5], 1); M[q] = 1; }
}
WI_G void wi_mclr(int n, int* M) { int q = blockIdx.x*blockDim.x + threadIdx.x; if (q < n) M[q] = 0; }
WI_G void wi_zero(int n, int* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = 0; }
WI_G void wi_cnt(int* st, const int* M, int n) { int q = blockIdx.x*blockDim.x + threadIdx.x; if (q < n && M[q]) atomicAdd(&st[4], 1); }
"""


def _rep(s, o, n, c=1):
    assert s.count(o) == c, (o[:80], s.count(o))
    return s.replace(o, n)


def _inv_q():
    """aw_inv_q (절 AS, fem_ao_rt.AW_SRC) 글자 그대로 + 점별 k 출력 kq."""
    from .fem_ao_rt import AW_SRC
    i = AW_SRC.index("__global__ void aw_inv_q(")
    j = AW_SRC.index("\n}\n", i) + 3
    b = AW_SRC[i:j].replace("aw_inv_q(", "wi_inv_q(", 1)
    b = _rep(b, "float e1, float e0) {", "float e1, float e0, int* kq) {")
    b = _rep(b, "    if (k > 0) atomicMax(&kk[0], k);\n", "    kq[q] = k;\n    if (k > 0) atomicMax(&kk[0], k);\n")
    return b


def apply(r, wi):
    """r = fem_ao_rt._src 결과 (skipify 앞). wi = dict(mode "", "node", "jbar"; kj; rec)."""
    if not wi:
        return r
    mode = wi.get("mode", "")
    assert mode in ("", "node", "jbar"), mode
    h = WI_HEAD
    if mode == "jbar":
        assert "#define AF_JBAR 1" not in r and "#define AF_SNH 1" in r, "--wi jbar 는 --elx snh 위 (전역 jbar 와 같이 쓰지 않음)"
        h += "#define AF_JBAR 1\n#define WI_KJ %.9ef\n" % float(wi["kj"])
        r = _rep(r, "    float J = S[0]*S[1]*S[2], kJ = prm[13];\n", "    float J = S[0]*S[1]*S[2], kJ = WI_KJ;\n", 2)
        r = _rep(r, "    aq_jbar_p(S, prm, p);", "    if (WI_M && on && WI_M[qq]) aq_jbar_p(S, prm, p);")
        r = _rep(r, "      aq_jbar_t(S, prm, pe, D);", "      if (WI_M && on && WI_M[qq]) aq_jbar_t(S, prm, pe, D);")
    r = r.replace("extern \"C\" {", "extern \"C\" {" + h, 1)
    i = r.rfind("}")
    return r[:i] + _inv_q() + WI_SRC + r[i:]


def alloc(s):
    """fem_ao_rt.AORTSolver.alloc 끝에서 (용량 바뀔 때마다)."""
    w = s.wf["wi"]
    I = lambda *sh: torch.zeros(*sh, device=s.dev, dtype=torch.int32)
    s.wikq = I(s.capQc * 8); s.wiM = I(s.capQc * 8); s.wkk = I(2)
    s.wikn = I(s.capN); s.wikn2 = I(s.capN); s.wikm = I(1); s.wikg = I(1); s.wiz = I(2)
    s.widd = torch.zeros(s.capN, 3, device=s.dev, dtype=torch.float32)
    s.wisi = torch.zeros(1, device=s.dev, dtype=torch.float64)
    s.wist = I(8)
    if w.get("rec"):
        s.wiR = torch.zeros(REC_CAP * REC_W, device=s.dev, dtype=torch.float32); s.wic = I(1); s.wick = I(NCODE * 2)
    if not hasattr(s, "wi_att"):
        s.wi_att = []
        s._wi_step0 = s.step
        s.step = lambda t: _step(s, t)


def bind(s):
    """fem_ao_rt.AORTSolver.bind_act 에서 (다시 할당 뒤마다)."""
    from .fem_ao_rt import ART
    g = s.mod.get_global("WI_M")
    ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=g)[0] = int(s.wiM.data_ptr()) if s.wf["wi"].get("mode") == "jbar" else 0


def _step(s, t):
    r = s._wi_step0(t)
    w = s.wf["wi"]
    fr = int(round(t / s.h)) + 1
    if w.get("rec") and fr in w.get("rec_frames", ()):
        n = min(int(s.wic[0]), REC_CAP)
        nc = s.K * s.nt
        s.wi_att.append(dict(frame=fr, K=int(s.K), n=int(s.wic[0]), rows=s.wiR[:n * REC_W].view(n, REC_W).cpu().numpy().copy(),
                             ck=s.wick[:2 * nc].view(nc, 2).cpu().numpy().copy(), st=s.wist.cpu().tolist()))
    return r


def frame_stats(s):
    """프레임 (모든 시도 합) 통계: [반복 중 국소 자름, 되돌림, 최대 kn, 자른 노드 합, 표시 점 (jbar), 새 표시, 반복 수] -- 마지막 시도 값."""
    return s.wist.cpu().tolist()


def newton(s, K, kstep, it, capN, capQ, nr, thv, WB, NR, BA):
    """뉴턴 반복 끝 (PCG 뒤, 원래 필터 블록 다음) 에서. node 판은 원래 장벽 보폭·rt_lsup 를 대신함 (부르는 쪽이 건너뜀)."""
    i32, f = np.int32, np.float32
    w = s.wf["wi"]; mode = w.get("mode", ""); IP = s.IP
    F3 = [f(q) for q in nr]
    if kstep == 0 and it == 0:                                                         # 시도 시작: 통계·기록 비움 (허용치와 무관하게 도는 커널)
        s.k("wi_zero", 8, (i32(8), P(s.wist)), b=32)
        if w.get("rec"):
            s.k("wi_zero", 1, (i32(1), P(s.wic)), b=32)
            s.k("wi_zero", NCODE * 2, (i32(NCODE * 2), P(s.wick)), b=256)
    if mode or not s.lsinv:                                                            # 방향 T δ (원래 필터 블록과 같은 커널; 필터 블록이 돌았으면 이미 있음)
        if s.compact:
            s.zf(s.awtd, 9, 3)
            s.k("ao_tvec", capN, (P(IP), P(s.sk), P(s.pm), *F3, P(s.dl), P(s.awtd)), b=256)
        else:
            s.k("aw_tdir", capN, (P(IP), P(s.sk), P(s.pm), *F3, P(s.dl), P(s.awtd)), b=256)
    aend = s._aw_end()

    def invq(d):
        s.zf(s.wikq); s.zf(s.wkk)
        s.k("wi_inv_q", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), P(d), P(s.wkk)) + aend + (P(s.wikq),))

    invq(s.awtd)
    if w.get("rec"):
        code = min(kstep * s.nt + it, NCODE - 1)
        s.k("wi_rec", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), P(s.awtd), P(s.mn), P(s.wikq), P(s.bx if WB is not None else None),
                             i32(code), P(s.wic), P(s.wiR), i32(REC_CAP), P(s.wick)))
    if mode == "jbar":
        s.k("wi_mark", capQ, (P(IP), P(s.wikq), P(s.wiM), P(s.wist)))
        if kstep == K - 1 and it == s.nt - 1:                                          # 프레임 (시도) 끝: 표시 수 세고 지움
            n = s.capQc * 8
            s.k("wi_cnt", n, (P(s.wist), P(s.wiM), i32(n)), b=256)
            s.k("wi_mclr", n, (i32(n), P(s.wiM)), b=256)
    if mode != "node":
        return
    s.wikg.copy_(s.wkk[:1])                                                            # 원래 전역 필터 k
    s.zf(s.wikn); s.zf(s.wikn2); s.zf(s.wikm)
    s.k("wi_nk", capQ, (P(IP), P(s.cb), P(s.wikq), P(None), P(s.wikn2)))
    s.k("wi_merge", capN, (P(IP), P(s.wikn), P(s.wikn2), P(s.wikm)), b=256)
    for _ in range(int(w.get("R", 4))):
        s.k("wi_dsc", capN, (P(IP), P(s.wikn), P(s.awtd), P(s.widd)), b=256)
        invq(s.widd)
        s.k("wi_nk", capQ, (P(IP), P(s.cb), P(s.wikq), P(s.wikn), P(s.wikn2)))
        s.k("wi_merge", capN, (P(IP), P(s.wikn), P(s.wikn2), P(s.wikm)), b=256)
    s.k("wi_dsc", capN, (P(IP), P(s.wikn), P(s.awtd), P(s.widd)), b=256)
    invq(s.widd)
    s.k("wi_fb", capN, (P(IP), P(s.wkk), P(s.wikg), P(s.wikn), P(s.wikm), P(s.wist)), b=256)
    if WB is not None:
        s.k("wi_ytry", capN, (P(IP), P(s.sk), P(s.pm), *F3, P(s.yh), P(s.eta), P(s.dl), P(s.wikn), P(s.wbyc)), b=256)
        s.zf(s.kk)
        s.k("rt_bar_kfe", capQ, (P(IP), P(s.cb), P(s.BW), P(s.y), P(s.wbyc), P(s.U), thv, *F3, f(WB["PN"]), P(s.kk)))
        kb = s.kk
    else:
        kb = s.wiz
    s.k("wi_sc", 1, (P(s.SC), P(s.wikm), P(kb), P(s.wikg), P(s.wisi), P(s.wist)), b=1)
    if NR:
        j = kstep * s.nt + it
        s.NRsi[j:j + 1].copy_(s.wisi[:1]); s.NRkb[j:j + 1].copy_(kb[:1]); s.NRst[j:j + 1].copy_(s.SC[14:15])
    s.k("wi_lsup", capN, (P(IP), P(s.wikn), P(kb), P(s.sk), P(s.pm), *F3, P(s.yh), P(s.eta), P(s.dl), P(s.y), i32(1 if s.compact else 0)), b=256)


def rec_summary(att, mn_med, nump_cell, floor_pn, nrm=(0.0, 0.0, 1.0), cell=None, hcell=None):
    """시도 기록 -> 요약 (호스트). nump_cell: 칸별 입자 수 (cpu numpy, 칸 번호 = q >> 3)."""
    rows = att["rows"]
    out = dict(frame=att["frame"], K=att["K"], n=att["n"], st=att["st"])
    ck = att["ck"]
    out["per_it"] = [[int(i), int(ck[i, 0]), int(ck[i, 1])] for i in range(ck.shape[0]) if ck[i, 0] > 0]
    if rows.shape[0] == 0:
        return out
    q = np.ascontiguousarray(rows[:, 1]).view(np.int32)
    uq, first = np.unique(q, return_index=True)
    r = rows[first]
    emp = r[:, 9] <= 0                                                               # 팔분에 입자 없음 (표본점 없음) -> 칸 입자 평균 높이 (프레임 끝)
    h = r[:, 10:13] @ np.asarray(nrm, dtype=np.float64) - floor_pn
    if hcell is not None:
        h = np.where(emp, hcell[(uq >> 3).clip(0, len(hcell) - 1)], h)
    npc = nump_cell[(uq >> 3).clip(0, len(nump_cell) - 1)] if nump_cell is not None else np.zeros(len(uq))
    mr = r[:, 7] / mn_med
    qs = lambda a: [float(x) for x in np.quantile(a, [0.0, 0.1, 0.5, 0.9, 1.0])] if len(a) else None
    out.update(nq=int(len(uq)), k=qs(r[:, 2]), J0=qs(r[:, 3]), J1=qs(r[:, 4]), Jmin=qs(r[:, 5]), Vq=qs(r[:, 6]), mmin_rel=qs(mr), msum_rel=qs(r[:, 8] / mn_med),
               h=qs(h), h_cells=(qs(h / cell) if cell else None), nump=qs(npc), detFq=qs(r[:, 13]),
               frac_np1=float((npc <= 1).mean()), frac_np2=float((npc <= 2).mean()), frac_h1=(float((h < cell).mean()) if cell else None),
               frac_mr1e2=float((mr < 1e-2).mean()), frac_oct_empty=float(emp.mean()), frac_J0neg=float((r[:, 3] <= 0).mean()), frac_J1neg=float((r[:, 4] <= 0).mean()),
               xyz_mean=[float(x) for x in r[:, 10:13].mean(0)], xyz_lo=[float(x) for x in r[:, 10:13].min(0)], xyz_hi=[float(x) for x in r[:, 10:13].max(0)])
    return out
