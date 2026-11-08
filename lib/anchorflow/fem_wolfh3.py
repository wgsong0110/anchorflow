"""절 CM66 묶음 wolfh3: wolf 높게 쌓임 3 -- --y0eb 가 높이를 붙드는지 걸음별 확인 + 장벽 에너지 판정을 바닥 법선 몫에만 두는 Y5.
감싸개 exe/wolfh3_wrap.py 가 실행기 (fem_an_rt_sim.py) 를 띄우는 순간 install() (wolfh2 등 다른 감싸개가 모두 설치된 뒤 = 가장 바깥 k 감싸기). 인자 없으면 이 모듈을 안 부름.

y  (--w3y 1)  Y5 (정의 변경은 뉴턴 시작점뿐, --y0loc 위): 국소 시작점 섞기 (yl_kq -> yl_mix) 의 노드 k 를 둘로 나눔.
     k_f (실현 가능 몫) = 지금 yl_kq 를 eb 0 으로 (활성 표본점 d > 0 유지에 필요한 k, 라운드 yl_chk 보탬·되돌림 yl_fb 그대로).
     k_E (장벽 에너지 몫) = 표본점마다 법선 1 차원 증분 퍼텐셜 E_q(β) = ½ M_q (d(β) − d_ŷ)² + κ_q B(d(β)) 를 후보 β = 2^-(k_f,q + i) (i 0..59) 와 β 0 중 최소로 고른 k
       (M_q = m_q / (τ² θ⁻²) = 표본점 질량의 법선 관성, d 는 세계 법선 높이, y 공간 관성 ½ m/τ² |y − ŷ|² 를 x = U + θ⁻¹ (y − U) 로 옮긴 것; κ_q B = rt_bar_q 와 같은 장벽 에너지)
       -> 그 점 노드에 원자 최대 (지금 규칙과 같은 퍼뜨림).
     섞기: Δ = η(ŷ) − η(U), η = η(U) + 2^-max(k_f, k_E) (Δ·n) n + 2^-k_f (Δ − (Δ·n) n)  -- 장벽 에너지 판정은 법선 성분에만, 접선은 실현 가능 규칙까지만 (--y0eb 없는 시작점 = 관성 예측 쪽).
     지금 --y0eb (점별 E_q(β) ≤ E_q(0)) 는 띠 안에서 바닥 쪽으로 가는 점이면 늘 β ≈ 0 (부동소수 k 25~33) -- 몸이 걸음 시작 높이에 묶임. Y5 는 그 판정을 관성 + 장벽
     최소로 바꿔 (장벽 에너지만 보면 법선으로 한 걸음도 못 내려감) 장벽이 관성과 맞서는 높이까지 내려가게 함. 문턱·클램프 없음 (d ≥ d̂ 구간만 지나는 점은 장벽 0 이라 β = 2^-k_f 와 같음).
     yl_chk·ul_kn·yl_st 는 k_M = max(k_f, k_E) 를 봄 (어긴 점의 노드 올림이 실제 법선 β 기준, 덜 풀린 걸음 갱신 집합 S 도 지금과 같은 '시작점이 묶인 노드').
     전역 --y0eb (rt_bar_y0e·kk) 는 그대로 (되돌림 yl_fb 와 진단 y0k 에만 쓰임).
nt (--w3nt 1) Y5 위 N: 접선 2^-k_f 대신 1 (관성 예측 그대로) = 절 CM52 N 과 같은 꼴 (fem_wolfh1 대신 이 모듈의 섞기에서).
dg (--w3dg 1) 진단 (결과 불변이 목적, 커널 시간만 보탬): 걸음마다 표본점을 걸음 시작 바닥 높이 d0 칸 [0,0.5,1,2,4,8,16,32,∞) cm 로 나눠
     시작점 (섞기 뒤) 과 걸음 끝에서 관성 예측 대비 실현 몫 -- 법선 Σm (d − d0) / Σm (d_ŷ − d0), 접선 Σm (t − t0)·(t_ŷ − t0) / Σm |t_ŷ − t0|²,
     지금 --y0eb 규칙이 k 를 올린 점 수 (k_eb > k_f)·k_eb ≥ 20 점 수, 노드 실효 k (그 점 27 노드 최대) ≥ 20 점 수 (= 걸음 시작에 묶인 점), 장벽 에너지 Σ κB (d0·d).
     장치 쪽 행 카운터 (그래프 안에서 걸음 끝마다 +1), 프레임 경계는 step() 뒤 읽음. 끝에 w3dg.npz (행) + w3dg.json (프레임별 합).
"""
import atexit
import json
import os

import numpy as np

ST = dict(y=0, nt=0, dg=0, out=".", marks=[], said=False)
NB = 8           # 높이 칸 수
NF = 12          # 칸마다 값 수
CAPR = 4096      # 행 (걸음) 상한

W3_SRC = r'''
// ---- 절 CM66 wolfh3 (lib/anchorflow/fem_wolfh3.py): Y5 (장벽 에너지 판정을 법선 몫에만) + 걸음별 시작점·걸음 끝 실현 몫 진단 ----
__device__ __forceinline__ void w3_xq(int q, const int* ids, const float* BW, int oc, const float* y, const float* U, float thinv, float bm, float* x) {
    for (int a=0;a<3;a++) x[a] = AF_BX[q*4+a]/bm;
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o]; for (int a=0;a<3;a++) x[a] += w*(U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a])); }
}
__device__ __forceinline__ int w3_kf(float d0, float d1) {                    // yl_kq 와 같은 실현 가능 k
    int k = 0;
    if (!(d1 > 0.f)) { float c = 1.f; while (true) { c *= 0.5f; k++; if (c == 0.f) break; if (d0 + c*(d1 - d0) > 0.f) break; } }
    return k;
}
__device__ __forceinline__ int w3_keb(int k, float d0, float d1, float dh, double kap) {   // yl_kq 의 --y0eb 규칙 (점 하나)
    double E0 = yl_e(d0, dh, kap); float c = ldexpf(1.f, -k); int i = 0;
    for (; i<59; i++) { double E = yl_e(d0 + c*(d1 - d0), dh, kap); if (E <= E0) break; c *= 0.5f; }
    return (i == 59) ? 1000 : k + i;
}
__global__ void w3_zi(int n, int* a) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = 0; }
__global__ void __launch_bounds__(128) y5_kq(const int* IP, const int* cb, const float* BW, const float* Vq, const float* mq, const float* y0, const float* y1, const float* U, float thinv,
                         float n0, float n1, float n2, float pn, float dh, float ks, float itau2, int* kE, int* cnt) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3];
    if (!(bm > 0.f) || !AF_BACT[q]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xa[3], xb[3];
    w3_xq(q, ids, BW, oc, y0, U, thinv, bm, xa); w3_xq(q, ids, BW, oc, y1, U, thinv, bm, xb);
    float d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn, d1 = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn;
    int k = w3_kf(d0, d1), kk = k;
    if (!(fminf(d0, d1) >= dh)) {                                               // 선분이 장벽 띠를 지나는 점만 (아니면 장벽 0 -> 관성만 -> 최대 β = 2^-k_f)
        double kap = (double)ks*(double)Vq[q];
        double M = (double)mq[q]*(double)itau2/((double)thinv*(double)thinv);
        double best = 1e300; int bi = -1;                                       // 후보는 β 가 줄어드는 순서 -> '<' 로 같은 값 중 가장 큰 β
        float c = ldexpf(1.f, -k);
        for (int i=0;i<60;i++) { float d = d0 + c*(d1 - d0); double r = (double)d - (double)d1; double E = 0.5*M*r*r + yl_e(d, dh, kap);
            if (E < best) { best = E; bi = i; } c *= 0.5f; }
        { double r = (double)d0 - (double)d1; double E = 0.5*M*r*r + yl_e(d0, dh, kap); if (E < best) { best = E; bi = -1; } }     // β 0 (걸음 시작)
        kk = (bi >= 0) ? k + bi : 1000;
        if (kk > k) atomicAdd(&cnt[0], 1);
        if (kk >= 1000) atomicAdd(&cnt[1], 1);
    }
    atomicAdd(&cnt[2], 1);
    if (kk == 0) return;
    for (int o=0;o<27;o++) if (BW[oc*27+o] != 0.f) atomicMax(&kE[ids[o]], kk);
}
__global__ void y5_kmax(const int* IP, const int* kn, const int* kE, int* kM) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    kM[n] = max(kn[n], kE[n]);
}
__global__ void y5_mix(const int* IP, const int* kn, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, const float* eU, const float* eS, float* eta, float* y,
                       const int* kE, int nt) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (pm[n] == 2) return;
    float bn = ldexpf(1.f, -max(kn[n], kE[n])), bt = nt ? 1.f : ldexpf(1.f, -kn[n]);
    float dl_[3]; for (int a=0;a<3;a++) dl_[a] = eS[n*3+a] - eU[n*3+a];
    float dn_ = dl_[0]*n0 + dl_[1]*n1 + dl_[2]*n2; float nn_[3] = {n0, n1, n2};
    float e[3]; for (int a=0;a<3;a++) { e[a] = eU[n*3+a] + bn*dn_*nn_[a] + bt*(dl_[a] - dn_*nn_[a]); eta[n*3+a] = e[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y[n*3+a] = yh[n*3+a] + t[a];
}
// 진단: D[(row*2 + phase)*8 + 칸][12]
__global__ void __launch_bounds__(128) w3_dg(const int* IP, const int* cb, const float* BW, const float* Vq, const float* mq, const float* y0, const float* y1, const float* y, const float* U, float thinv,
                         float n0, float n1, float n2, float pn, float dh, float ks, const int* kn, const int* kE, int phase, double* D, const int* ctr, int cap) {
    __shared__ double sh[8*12];
    for (int i=threadIdx.x;i<8*12;i+=blockDim.x) sh[i] = 0.0;
    __syncthreads();
    int row = ctr[0];
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    float bm = (q < IP[12]) ? AF_BX[q*4+3] : 0.f;
    if (row < cap && bm > 0.f) {
        int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
        float xa[3], xb[3], xm[3];
        w3_xq(q, ids, BW, oc, y0, U, thinv, bm, xa); w3_xq(q, ids, BW, oc, y1, U, thinv, bm, xb); w3_xq(q, ids, BW, oc, y, U, thinv, bm, xm);
        float d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn, d1 = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn, dm = xm[0]*n0 + xm[1]*n1 + xm[2]*n2 - pn;
        float nv[3] = {n0, n1, n2}; double t1 = 0.0, tm = 0.0, t11 = 0.0, tmm = 0.0;
        for (int a=0;a<3;a++) { double e1 = (double)(xb[a] - xa[a]) - (double)(d1 - d0)*nv[a], em = (double)(xm[a] - xa[a]) - (double)(dm - d0)*nv[a];
            t1 += em*e1; t11 += e1*e1; tmm += em*em; }
        const float ed[7] = {0.005f, 0.01f, 0.02f, 0.04f, 0.08f, 0.16f, 0.32f};
        int b = 0; for (int i=0;i<7;i++) if (d0 >= ed[i]) b = i + 1;
        double m = (double)mq[q];
        double* s = sh + b*12;
        atomicAdd(s+0, 1.0); atomicAdd(s+1, m); atomicAdd(s+2, m*((double)dm - d0)); atomicAdd(s+3, m*((double)d1 - d0));
        atomicAdd(s+4, m*t1); atomicAdd(s+5, m*t11); atomicAdd(s+11, m*tmm);
        bool act = AF_BACT[q] != 0;
        double kap = (double)ks*(double)Vq[q];
        if (act) {
            if (phase == 0) { int kf = w3_kf(d0, d1), ke = w3_keb(kf, d0, d1, dh, kap); if (ke > kf) atomicAdd(s+6, 1.0); if (ke >= 20) atomicAdd(s+7, 1.0); }
            double e0 = yl_e(d0, dh, kap), em_ = yl_e(dm, dh, kap);
            if (e0 < 1e29) atomicAdd(s+9, e0);
            if (em_ < 1e29) atomicAdd(s+10, em_);
        }
        int km = 0; for (int o=0;o<27;o++) if (BW[oc*27+o] != 0.f) { int id = ids[o]; int kk = kn[id]; if (kE) kk = max(kk, kE[id]); km = max(km, kk); }
        if (km >= 20) atomicAdd(s+8, 1.0);
    }
    __syncthreads();
    if (row < cap) for (int i=threadIdx.x;i<8*12;i+=blockDim.x) if (sh[i] != 0.0) atomicAdd(&D[((size_t)(row*2 + phase))*96 + i], sh[i]);
}
__global__ void w3_inc(int* ctr) { ctr[0] += 1; }
'''


def patch_src(src):
    if "y5_kq(" in src:
        return src
    assert "yl_kq(" in src and "AF_BACT" in src, "절 CM66 은 --y0loc 장벽 판 위"
    k = src.rstrip().rfind("}")
    return src[:k] + W3_SRC + "\n}\n"


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _bufs(s_):
    import torch
    capN = int(s_.capN)
    if getattr(s_, "_w3cap", -1) != capN:
        dev = s_.dev
        s_._w3cap = capN
        s_._w3kE = torch.zeros(capN, device=dev, dtype=torch.int32)
        s_._w3kM = torch.zeros(capN, device=dev, dtype=torch.int32)
        s_._w3y0 = torch.zeros(capN, 3, device=dev, dtype=torch.float32)
        s_._w3y1 = torch.zeros(capN, 3, device=dev, dtype=torch.float32)
    if getattr(s_, "_w3cnt", None) is None:
        s_._w3cnt = torch.zeros(4, device=s_.dev, dtype=torch.int32)
        s_._w3ctr = torch.zeros(1, device=s_.dev, dtype=torch.int32)
        s_._w3D = torch.zeros(CAPR * 2 * NB * NF, device=s_.dev, dtype=torch.float64)
        ST["sol"] = s_


def install(y=0, nt=0, dg=0, out="."):
    from . import fem_an_rt as ART
    ST.update(y=int(y), nt=int(nt), dg=int(dg), out=out)
    assert not (ST["nt"] and not ST["y"]), "--w3nt 는 --w3y 위"
    _ob = ART.build_src
    ART.build_src = lambda *a, **kw: patch_src(_ob(*a, **kw))
    i32 = np.int32
    cls = ART.RTSolver
    _oa = cls.alloc

    def alloc(s_, need):
        r = _oa(s_, need)
        _bufs(s_)
        return r
    cls.alloc = alloc
    _k = cls.k

    def k(s_, name, nn, args, b=128):
        Y = ST["y"]
        if name == "rt_ga_step0":
            s_._w3tau = float(args[9])
        if name == "yl_kq":
            _bufs(s_)
            if ST["dg"]:
                s_.dcopy(s_._w3y0, s_.ylU); s_.dcopy(s_._w3y1, s_.ySv)
            s_._w3ka = tuple(args)                                                       # 진단용 (thinv·법선·pn·dh·ks)
            if Y:
                a = list(args); a[14] = i32(0)                                           # 실현 가능 몫만 (eb 0)
                r = _k(s_, name, nn, tuple(a), b)
                _k(s_, "w3_zi", int(s_.capN), (i32(int(s_.capN)), P(s_._w3kE)), 256)
                it2 = 1.0 / (s_._w3tau * s_._w3tau)
                _k(s_, "y5_kq", nn, (args[0], args[1], args[2], args[3], P(s_.mq), args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13],
                                     np.float32(it2), P(s_._w3kE), P(s_._w3cnt)), 128)
                return r
            return _k(s_, name, nn, args, b)
        if Y and name == "yl_mix":
            return _k(s_, "y5_mix", nn, tuple(args) + (P(s_._w3kE), i32(ST["nt"])), b)
        if Y and name in ("yl_chk", "ul_kn", "yl_st"):
            ix = {"yl_chk": 10, "ul_kn": 1, "yl_st": 1}[name]
            _k(s_, "y5_kmax", int(s_.capN), (args[0], args[ix], P(s_._w3kE), P(s_._w3kM)), 256)
            a = list(args); a[ix] = P(s_._w3kM)
            r = _k(s_, name, nn, tuple(a), b)
            if name == "yl_st" and ST["dg"]:
                _dg(s_, _k, 0)
            return r
        if name == "yl_st" and ST["dg"]:
            r = _k(s_, name, nn, args, b)
            _dg(s_, _k, 0)
            return r
        if name == "ul_zero" and ST["dg"] and getattr(s_, "_w3ka", None) is not None:
            _dg(s_, _k, 1)
            _k(s_, "w3_inc", 1, (P(s_._w3ctr),), 1)
        return _k(s_, name, nn, args, b)
    cls.k = k

    def _dg(s_, _k, phase):
        a = s_._w3ka
        nq = int(s_.capQc) * 8
        _k(s_, "w3_dg", nq, (a[0], a[1], a[2], a[3], P(s_.mq), P(s_._w3y0), P(s_._w3y1), P(s_.y), a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
                            P(s_.ykn), P(s_._w3kE) if ST["y"] else P(None), i32(phase), P(s_._w3D), P(s_._w3ctr), i32(CAPR)), 128)

    if ST["dg"]:
        _os = cls.step

        def step(s_, t, *a, **kw):
            d = getattr(s_, "_w3dep", 0)
            s_._w3dep = d + 1
            try:
                r = _os(s_, t, *a, **kw)
            finally:
                s_._w3dep = d
            if d == 0:
                c = getattr(s_, "_w3ctr", None)
                ST["marks"].append((int(round(t / s_.h)), int(c.item()) if c is not None else -1))
            return r
        cls.step = step
    print(f"[절 CM66] wolfh3 켬: Y5 {ST['y']} · 접선 관성 예측 (N 꼴) {ST['nt']} · 진단 {ST['dg']}", flush=True)

    def _fin():
        info = {k_: ST[k_] for k_ in ("y", "nt", "dg")}
        sol = ST.get("sol")
        if sol is not None:
            try:
                import torch
                torch.cuda.synchronize()
                c = sol._w3cnt.cpu().tolist()
                info.update(y5_raised=c[0], y5_beta0=c[1], y5_pts=c[2])
                nq = int(sol.IP[12]); nn_ = int(sol.IP[9])
                info["mq_sum"] = float(sol.mq[:nq].double().sum()); info["mn_sum"] = float(sol.mn[:nn_].double().sum())
                if ST["dg"]:
                    nr = int(sol._w3ctr.item())
                    D = sol._w3D.view(CAPR, 2, NB, NF)[:min(nr, CAPR)].cpu().numpy()
                    np.savez_compressed(os.path.join(out, "w3dg.npz"), D=D, marks=np.array(ST["marks"]))
                    fr = []
                    lo = 0
                    for f_, hi in ST["marks"]:
                        hi = min(hi, D.shape[0])
                        fr.append(dict(f=f_, rows=[lo, hi], S=D[lo:hi, 0].sum(0).tolist(), E=D[lo:hi, 1].sum(0).tolist()))
                        lo = hi
                    json.dump(dict(edges_cm=[0, 0.5, 1, 2, 4, 8, 16, 32], fields=["n", "m", "m_dn_real", "m_dn_pred", "m_dt_dot", "m_dt_pred2", "n_eb_raise", "n_eb_ge20", "n_node_k_ge20",
                                                                                   "E0", "Em", "m_dt_real2"], nrows=nr, frames=fr), open(os.path.join(out, "w3dg.json"), "w"))
                    info["dg_rows"] = nr
            except Exception as e:                                                       # noqa: BLE001
                info["err"] = repr(e)
        json.dump(info, open(os.path.join(out, "w3info.json"), "w"))
    atexit.register(_fin)
