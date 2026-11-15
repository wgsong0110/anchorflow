"""절 CM184 묶음 hexff (GF 장면 바닥 접촉 뒤집힘 수정, 정의 변경): 육면체 2차 B-스플라인 판의 뉴턴 시작점·보폭에 가우스 점 뒤집힘 불가능 조건.

원인 (절 CM184): 재질 판 (fem_an_rt_sim 의 MatRT) 은 퍼텐셜이 없어 에너지 선탐색이 없고 (a.ls = 0) 보폭 규칙은 장벽 실현 가능 보폭 (rt_bar_kfe) 뿐이다.
  시작점 섞기 --y0loc 는 장벽 근처 노드만 β 를 줄이고 (k 큼) 나머지 노드는 관성 예측 ŷ 쪽으로 그대로 보내므로, 한 프레임 이동이 칸 크기의 수십 배인 장면
  (GF: 6~8 m/s, 칸 0.0067~0.02 m, 프레임 0.03 s -> 9~36 칸) 에서는 바닥 위 3~4 칸에서 가우스 점 X = (I + A(y)) F 의 det 가 0 이하가 되어 재질 응력이 비유한
  (F2 = NaN) -> 뉴턴 0 반복 -> 덜 풀린 걸음 갱신이 뒤집힌 시작점을 그대로 받음 -> 발산.
수정 (모든 장면 같은 규칙, 기존 장벽 규칙과 같은 꼴):
  (1) 보폭: 장벽 실현 가능 보폭 kk 를 구하는 모든 자리 (걸음 시작 rt_bar_kfe, 뉴턴 반복 rt_bar_kfe) 뒤에 hf_kfe -- 입자 있는 가우스 점 (V_q > 0) 마다
      y0 -> y1 의 c = 2^-k 에서 det X(y(c)) > τ det X(y0) 이고 걸음 끝 외삽 X(U + thinv (y(c) − U)) 도 같은 조건인 가장 작은 k 를 kk 에 원자 최대.
      (y0 에서 이미 det ≤ 0 인 점은 고칠 수 없으니 뺌.)
  (2) 시작점 섞기 (--y0loc) 확인 라운드: yl_chk 뒤 hf_chk -- 섞은 시작점의 가우스 점 det 이 0 이하 (또는 외삽 det 이 0 이하) 면 위반 수에 더하고 그 칸 노드 k 를
      (칸 노드 최대 k) 로 맞춤 (yl_merge 가 합침). 마지막 라운드에도 위반이 있으면 기존 규칙대로 모든 노드 전역 k (kk, (1) 이 들어간 값) 로.
  τ = --hff_tau (기본 0: det > 0 만). 위반 없는 걸음은 보폭·시작점이 비트 단위로 그대로 (PG 장면은 기존 결과와 같아야 함 -- 확인 대상).
서브스텝 (걸음) 안에서 입자는 안 씀 (가우스 점 F_q·V_q·노드 변수만).
"""
import numpy as np
import torch

INFO = dict(tau=0.0, mods=[], n_kfe=0, n_chk=0, n_lmix=0, lmix=0, kfe=1, skip=(2,))

HF_SRC = r"""
// ---- 절 CM184 hexff: 가우스 점 뒤집힘 불가능 보폭·시작점 확인 ----
__device__ unsigned char* AF_HFREL;                                          // 이 걸음 풀어 준 거의 빈 노드 표시 (mode 2; 걸음 시작 hf_relclr 로 지움)
__global__ void hf_relclr(const int* IP) { int n = blockIdx.x*blockDim.x + threadIdx.x; if (n < IP[9] && AF_HFREL) AF_HFREL[n] = 0; }
__device__ __forceinline__ float hf_det(const float* X) { return X[0]*(X[4]*X[8]-X[5]*X[7]) - X[1]*(X[3]*X[8]-X[5]*X[6]) + X[2]*(X[3]*X[7]-X[4]*X[6]); }
// X(y) = (I + Σ_o y_o ⊗ bg_o) F  (재질 커널 ao_elem 과 같은 식), Xe = 같은 식에 U + thinv (y − U)
__device__ __forceinline__ void hf_X(const int* ids, const float* BGo, const float* F, const float* y, const float* U, float thinv, int ext, float* X) {
    float A[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BGo + o*3; int id = ids[o];
        for (int a=0;a<3;a++) { float ya = y[id*3+a]; if (ext) ya = U[id*3+a] + thinv*(ya - U[id*3+a]);
            A[a*3] += ya*bg[0]; A[a*3+1] += ya*bg[1]; A[a*3+2] += ya*bg[2]; } }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) X[a*3+b] = F[a*3+b] + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b];
}
// 거의 빈 노드 고정 자리 (mt_floor_n: 질량 < m̄ 이면 관성 예측 Ũ 에 고정, pm 2) 를 시작점 섞기와 같은 β 로: y = ŷ = U + β_n (Ũ − U), β_n = 2^-k (노드별 kn 또는 전역 kk).
// β = 1 (장벽이 시작점을 안 줄인 걸음) 이면 지금 판과 같음. 경계 조건 고정 노드 (fixd) 는 그대로.
// mode 2 (풀어 줌): 그 걸음 β < 1 인 거의 빈 노드는 고정하지 않고 자유 노드 (pm 0, √k 1, ŷ = Ũ) 로 -- 시작 위치 U + β (Ũ − U), η = y − Ũ (rt_floor_n 의 자유 노드와 같은 변수 바꿈).
__global__ void hf_lmix(const int* IP, const int* kn, const int* kk, int* pm, float* sk, float* eta, const float* mn, float mthr, const unsigned char* fixd, const float* U, const float* Ut, float* yh, float* y, int mode) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (!(mn[n] < mthr)) return;
    if (pm[n] != 2 && !(mode == 2 && AF_HFREL && AF_HFREL[n])) return;
    if (fixd && fixd[n]) return;
    int k = kn ? kn[n] : kk[0];
    if (k == 0) { if (mode == 2 && AF_HFREL && AF_HFREL[n]) { for (int a=0;a<3;a++) { yh[n*3+a] = Ut[n*3+a]; eta[n*3+a] = 0.f; y[n*3+a] = Ut[n*3+a]; } } return; }
    float be = ldexpf(1.f, -k);
    if (mode == 2) {
        if (AF_HFREL) AF_HFREL[n] = 1;
        pm[n] = 0; sk[n] = 1.f;
        for (int a=0;a<3;a++) { float v = U[n*3+a] + be*(Ut[n*3+a] - U[n*3+a]); yh[n*3+a] = Ut[n*3+a]; eta[n*3+a] = v - Ut[n*3+a]; y[n*3+a] = v; }
        return;
    }
    for (int a=0;a<3;a++) { float v = U[n*3+a] + be*(Ut[n*3+a] - U[n*3+a]); yh[n*3+a] = v; y[n*3+a] = v; }
}
__global__ void __launch_bounds__(128) hf_kfe(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y0, const float* y1, const float* U, float thinv, float tau, int* kk) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    if (!(Vq[q] > 0.f)) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    const float* BGo = BG + oc*27*3;
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float X0[9], X1[9], E0[9], E1[9];
    hf_X(ids, BGo, F, y0, U, thinv, 0, X0); hf_X(ids, BGo, F, y1, U, thinv, 0, X1);
    hf_X(ids, BGo, F, y0, U, thinv, 1, E0); hf_X(ids, BGo, F, y1, U, thinv, 1, E1);
    float d0 = hf_det(X0), e0 = hf_det(E0);
    if (!(d0 > 0.f) || !(e0 > 0.f)) return;                                    // 시작이 이미 뒤집힘: 고칠 수 없음 (기존 판 그대로)
    float t0 = tau*d0, te = tau*e0;
    int k = 0; float c = 1.f;
    while (true) {
        float Xc[9], Ec[9];
        for (int i=0;i<9;i++) { Xc[i] = X0[i] + c*(X1[i] - X0[i]); Ec[i] = E0[i] + c*(E1[i] - E0[i]); }
        if (hf_det(Xc) > t0 && hf_det(Ec) > te) break;
        c *= 0.5f; k++; if (c == 0.f || k >= 60) break;
    }
    if (k > 0) atomicMax(kk, k);
}
__global__ void __launch_bounds__(128) hf_cnt(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, int seq, int* out) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;                               // 진단 (절 CM184 nanprobe, 결과 불변): det F_q ≤ 0 · det X(y) ≤ 0 인 점이 처음 생긴 띄움 순번
    if (q >= IP[12]) return;
    if (!(Vq[q] > 0.f)) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    if (!(hf_det(F) > 0.f)) atomicMin(&out[0], seq);
    float X[9]; hf_X(ids, BG + oc*27*3, F, y, y, 1.f, 0, X);
    if (!(hf_det(X) > 0.f)) { atomicMin(&out[1], seq); atomicAdd(&out[2], 1); }
}
__global__ void __launch_bounds__(128) hf_who(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, const float* U, const int* pm, const float* mn, float* out) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;                               // 진단: det X(y) ≤ 0 인 점의 27 노드를 pm 별로 -- out[pm*4+0] 수, +1 최대 |y − U|, +2 질량 합, +3 (pm 0 인데 |y−U| 최대)
    if (q >= IP[12]) return;
    if (!(Vq[q] > 0.f)) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float X[9]; hf_X(ids, BG + oc*27*3, F, y, y, 1.f, 0, X);
    if (hf_det(X) > 0.f) return;
    atomicAdd(&out[12], 1.f);
    for (int o=0;o<27;o++) { int id = ids[o]; int p = pm[id]; p = p < 0 ? 0 : (p > 2 ? 2 : p);
        float d = 0.f; for (int a=0;a<3;a++) { float t = y[id*3+a] - U[id*3+a]; d += t*t; } d = sqrtf(d);
        atomicAdd(&out[p*4], 1.f); atomicMax((int*)&out[p*4+1], __float_as_int(d)); atomicAdd(&out[p*4+2], mn[id]); }
}
__global__ void __launch_bounds__(128) hf_chk(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, const float* U, float thinv,
                         const int* kn, int* kn2, int* fl) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    if (!(Vq[q] > 0.f)) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    const float* BGo = BG + oc*27*3;
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float X[9], E[9];
    hf_X(ids, BGo, F, y, U, thinv, 0, X); hf_X(ids, BGo, F, y, U, thinv, 1, E);
    if (hf_det(X) > 0.f && hf_det(E) > 0.f) return;
    atomicAdd(fl, 1);
    int km = 0; for (int o=0;o<27;o++) km = max(km, kn[ids[o]]);
    if (km == 0) km = 1;
    for (int o=0;o<27;o++) atomicMax(&kn2[ids[o]], km);
}
"""
_ANCHOR = "__global__ void __launch_bounds__(128) rt_bar_kfe("


def _inject(code):
    if not isinstance(code, str) or _ANCHOR not in code or "hf_kfe(" in code:
        return code, 0
    return code.replace(_ANCHOR, HF_SRC + _ANCHOR, 1), 1


def _wrap_k(k0):
    def k(s, name, n, args, b=128):
        if INFO["lmix"] == 2 and getattr(s, "_hff", False) and (getattr(s, "_hfrel", None) is None or s._hfrel.numel() < s.capN) \
                and not torch.cuda.is_current_stream_capturing() and hasattr(s, "mod"):
            try:                                                               # 풀어 준 노드 표시 버퍼 (그래프 밖에서 한 번; 용량 늘면 다시)
                import cupy as cp
                s._hfrel = torch.zeros(s.capN, device=s.U.device, dtype=torch.uint8)
                g = s.mod.get_global("AF_HFREL"); cp.ndarray((1,), dtype=cp.uint64, memptr=g)[0] = int(s._hfrel.data_ptr())
            except Exception as e:
                INFO["rel_err"] = repr(e)[:200]
        r = k0(s, name, n, args, b)
        if getattr(s, "_hff", False) and getattr(s, "_pkc", None) is None and name == "mt_floor_n" and INFO["lmix"] == 2 and getattr(s, "_hfrel", None) is not None:
            k0(s, "hf_relclr", n, (args[0],), 256)
        if getattr(s, "_hff", False) and getattr(s, "_pkc", None) is None and name in ("yl_mix", "rt_y0mix") and getattr(s, "mthr", 0.0) > 0 and INFO["lmix"]:
            P = lambda t: np.uint64(t.data_ptr()) if t is not None else np.uint64(0)
            if name == "yl_mix":            # (IP, kn, sk, pm, n0, n1, n2, yh, eU, eS, eta, y)
                kn_, kk_, yh_, y_ = args[1], P(s.kk), args[7], args[11]
            else:                           # rt_y0mix (IP, kk, sk, pm, n0, n1, n2, yh, eS, eta, y)
                kn_, kk_, yh_, y_ = np.uint64(0), args[1], args[7], args[10]
            k0(s, "hf_lmix", n, (args[0], kn_, kk_, args[3], P(s.sk), P(s.eta), P(s.mn), np.float32(s.mthr), P(getattr(s, "fixd", None)), P(s.U), P(s.Ut), yh_, y_, np.int32(INFO["lmix"])), 256)
            INFO["n_lmix"] += 1
        if getattr(s, "_hff", False) and getattr(s, "_pkc", None) is None and name in ("rt_bar_kfe", "yl_chk") \
                and getattr(s, "_mat", None) not in INFO["skip"]:      # 기본: 모래 (재질 2, Drucker-Prager 반환 사상이 주값 크기 |σ| 로 정의돼 뒤집힘을 스스로 다룸) 빼고 전부 (탄성 실행기 포함)
            P = lambda t: np.uint64(t.data_ptr())
            if (name == "rt_bar_kfe" and len(args) != 12) or (name == "yl_chk" and len(args) != 13):
                INFO.setdefault("bad_args", set()).add((name, len(args)))
                return r
            if name == "rt_bar_kfe" and not INFO["kfe"]:
                return r
            if name == "rt_bar_kfe":       # (IP, cb, BW, y0, y1, U, thinv, n0, n1, n2, pn, kk)
                k0(s, "hf_kfe", n, (args[0], args[1], P(s.BG), P(s.Fq), P(s.Vq), args[3], args[4], args[5], args[6], np.float32(INFO["tau"]), args[11]), b)
                INFO["n_kfe"] += 1
            else:                           # yl_chk (IP, cb, BW, y, U, thinv, n0, n1, n2, pn, kn, kn2, fl)
                k0(s, "hf_chk", n, (args[0], args[1], P(s.BG), P(s.Fq), P(s.Vq), args[3], args[4], args[5], args[10], args[11], args[12]), b)
                INFO["n_chk"] += 1
        return r
    k._hff = True
    return k


def install(tau=0.0, lmix=0, kfe=1, skip=(2,)):
    import cupy as cp
    from . import fem_an_rt as ART
    INFO["tau"] = float(tau); INFO["lmix"] = int(lmix); INFO["kfe"] = int(kfe); INFO["skip"] = tuple(skip)
    base = cp.RawModule
    if not getattr(base, "_hff", False):
        def RawModule(*a, **kw):
            a = list(a)
            code = kw.get("code", a[0] if a else None)
            code2, cnt = _inject(code)
            if cnt:
                INFO["mods"].append(cnt)
                if "code" in kw:
                    kw["code"] = code2
                else:
                    a[0] = code2
            return base(*a, **kw)
        RawModule._hff = True
        cp.RawModule = RawModule
    C = ART.RTSolver
    if not getattr(C.__dict__["k"], "_hff", False):
        C.k = _wrap_k(C.__dict__["k"])
    f0 = C.__init__
    if not getattr(f0, "_hff", False):
        def init(s, *a, _f0=f0, **kw):
            _f0(s, *a, **kw)
            s._hff = True
        init._hff = True
        C.__init__ = init
    print(f"[절 CM184 hexff] 가우스 점 뒤집힘 불가능 보폭 (rt_bar_kfe 뒤 hf_kfe) + 시작점 섞기 확인 (yl_chk 뒤 hf_chk), τ {tau}, 보폭 조건 {kfe}, 뺀 재질 {skip}, 거의 빈 노드 고정 자리 섞기 {lmix}", flush=True)
