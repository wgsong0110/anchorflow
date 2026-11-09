"""절 CM86 묶음 speed13 (11-08 위임, 수박 두 시도 프레임 = 시작점 뒤집힘 필터 β < 1 걸음): 벌점이 시작점 뒤집힘을 늘리는 이유 진단 + 고침.
감싸개 exe/speed13_wrap.py 가 부름 (exe/speed12_wrap.py 바깥). 옛 실행기 (수박, AORTSolver) 만.

(D) --s13dg 1 (결과 불변 -- 읽기만): 'r' 경계 (H8 = 반복 0 시작, 시작점이 정해진 뒤 첫 뉴턴 갱신 앞) 판정 값을 읽는 순간 그래프 밖에서
    가우스 점마다 J/det F_q 를 여러 위치에서 계산:
      J0 = 시작점 기준 y_z (aw_y0z_a 의 yb) · Jt = 필터가 보는 선분 끝 y_z + dy (β 1 시작점) · Jh = ŷ (고착 섞은 관성 예측 yh) ·
      Ju = 고착 섞기 전 관성 예측 Ut (= HHT ys + τ²(g − α_m a_n)) · Jn = 가속도 몫 뺀 관성 예측 Yn = α_f U + (1 − α_f)(U + dt v) + τ² g ·
      Je = 걸음 끝 외삽 e1 (y_z + dy) + e0 U (--ls_end 와 같은 식) -- 그리고 필터와 같은 점별 k (aw_inv_q 글자 그대로 + kq 출력).
    노드 표지: 가벼움 벌점 면 노드 = 면 가중 fw − 1[P1X] > 0 인 면 (칸 c, 축 a) 의 36 노드 (lightg3/lightg4 의 r κ λ 몫이 실제로 선 면).
    점마다 27 노드 중 벌점 면 노드 있음 · 바닥 띠 노드 (an > 0) 있음 · 0 < √k ≤ 1e-6 노드 있음 · 최소 노드 질량 / m̄ · 노드 가속도 몫 |Ut − Yn| 최대 · 속도 몫 |(1 − α_f) dt v| 최대.
    노드 요약: 벌점 면 노드 대 나머지 활성 노드의 |Ut − Yn| · |(1 − α_f) dt v| 중앙·평균.
    -> <out>/s13.jsonl (걸음마다 한 줄; k > 0 점 표본 최대 24 개).
(X) --s13x 1 (결과 바뀜; H9 = H8 + X): X2 신호 S 를 '시작점 필터를 자른 칸의 노드' 로 국소화.
    H8 (절 CM85 (나)): S = (1 − β)² Σ_{√k > 0} (M/τ²)² |T⁻¹ P (U − ŷ)|² -- 몸 전체 √k > 0 노드 합. β 가 전역이라 몸 안 어느 칸이든 (바닥과 먼 무거운 칸의
      관성 예측 압축 포함) 필터를 자르면 멀리 있는 √k ≈ 0 바닥 노드의 1/k 몫이 S 로 들어와 X2 를 켬 (절 CM86.1: 벌점 판의 늘어난 β < 1 걸음은
      √k ≈ 0 노드가 없는 칸에서 잘림).
    X: S = (1 − β)² Σ_{n ∈ N_cut, √k > 0} (같은 몫),  N_cut = 이 걸음 시작점 필터 (aw_inv_q, --ls_end 끝 외삽 포함) 에서 k_q > 0 인 가우스 점의 27 노드.
      = 옛 판 (H7) 에서 '자른 칸에 걸친 노드의 시작점 몫' 만 판정에 넣음. 시작점·β·문턱 (1e6) 그대로, 새 값 없음.
    구현: aw_y0z_a 의 노드 몫을 S12A 원자 합 대신 노드 버퍼 S13Q[n] 에, aw_inv_q (시작점 호출) 가 k > 0 점의 27 노드를 S13M 에 표시
      (aw_y0z_a 가 걸음마다 둘 다 비움), 반복 0 'r' 판정 (speed12 s12_r) 바로 앞에 s13_sum 이 Σ_{S13M} S13Q 를 S12A 에 모음.
"""
import json
import os

import numpy as np
import torch


class C:
    out = "."
    dg = 0
    x = 0
    sum_fn = None
    log = None
    fn = {}
    nrec = 0
    nerr = 0
    done = set()


S13_SRC = r"""
// ---- 절 CM86 speed13 진단 (lib/anchorflow/fem_speed13.py; 그래프 밖에서만 부름, 결과 불변) ----
#define S13_G __global__
__device__ __forceinline__ double s13_det(const int* ids, int oc, const float* BG, const float* F, const float* v) {
    float Xf[9]; aw_X(ids, oc, BG, F, v, 1, Xf);
    double X[9], Cf[9]; for (int k=0;k<9;k++) X[k] = (double)Xf[k]; aw_cof(X, Cf);
    return X[0]*Cf[0] + X[1]*Cf[1] + X[2]*Cf[2];
}
// 가벼움 벌점 면 (fw − 1[P1X] > 0) 의 36 노드 표지
S13_G void s13_nf(const int* IP, const int* cb, const unsigned char* fm, const float* fw, const float* wid, int* nf) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= IP[11]) return;
    int m = fm[c]; if (!m) return;
    int nb1 = IP[7], nb2 = IP[8], s1 = nb2, s0 = nb1*nb2;
    int base = (cb[c*3]*nb1 + cb[c*3+1])*nb2 + cb[c*3+2];
    for (int a=0;a<3;a++) {
        if (!(m & (1 << a))) continue;
        if (!(fw[c*3+a] - wid[c*3+a] > 0.f)) continue;
        for (int o=0;o<27;o++) nf[base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3)] = 1;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
            int e = (a == 0) ? (base + 3*s0 + i*s1 + j) : ((a == 1) ? (base + i*s0 + 3*s1 + j) : (base + i*s0 + j*s1 + 3));
            nf[e] = 1;
        }
    }
}
S13_G void s13_q(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq,
                 const float* yz, const float* yt, const float* yh, const float* ut, const float* yn, const float* ue,
                 const int* kq, const int* nf, const float* mn, float mbar, const float* an, const float* sk, const float* nacc, const float* nvel,
                 int* cnt, float* rows, int cap) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    if (!(Vq[q] > 0.f)) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    double Fd[9], Cf[9]; for (int k=0;k<9;k++) Fd[k] = (double)F[k]; aw_cof(Fd, Cf);
    double dF = Fd[0]*Cf[0] + Fd[1]*Cf[1] + Fd[2]*Cf[2];
    double J0 = s13_det(ids, oc, BG, F, yz)/dF, Jt = s13_det(ids, oc, BG, F, yt)/dF, Jh = s13_det(ids, oc, BG, F, yh)/dF;
    double Ju = s13_det(ids, oc, BG, F, ut)/dF, Jn = s13_det(ids, oc, BG, F, yn)/dF, Je = ue ? s13_det(ids, oc, BG, F, ue)/dF : 1.0;
    int fN = 0, fF = 0, fW = 0; float mi = 3.0e38f, am = 0.f, vm = 0.f;
    for (int o=0;o<27;o++) { int id = ids[o]; if (nf[id]) fN = 1; if (an[id] > 0.f) fF = 1; float s = sk[id]; if (s > 0.f && s <= 1e-6f) fW = 1;
                             float m = mn[id]; if (m > 0.f) mi = fminf(mi, m); am = fmaxf(am, nacc[id]); vm = fmaxf(vm, nvel[id]); }
    int k = kq[q];
    atomicAdd(&cnt[0], 1); if (fN) atomicAdd(&cnt[1], 1);
    if (k > 0) { atomicAdd(&cnt[2], 1); if (fN) atomicAdd(&cnt[3], 1); if (fF) atomicAdd(&cnt[17], 1); if (mi < mbar) atomicAdd(&cnt[18], 1); if (fW) atomicAdd(&cnt[19], 1);
                 if (Ju <= 0.0) atomicAdd(&cnt[15], 1); if (Jn <= 0.0) atomicAdd(&cnt[16], 1); if (Jh <= 0.0) atomicAdd(&cnt[20], 1); if (Jt <= 0.0) atomicAdd(&cnt[21], 1); if (Je <= 0.0) atomicAdd(&cnt[22], 1); }
    if (Ju <= 0.0) { atomicAdd(&cnt[4], 1); if (fN) atomicAdd(&cnt[5], 1); }
    if (Jn <= 0.0) { atomicAdd(&cnt[6], 1); if (fN) atomicAdd(&cnt[7], 1); }
    if (Jh <= 0.0) { atomicAdd(&cnt[8], 1); if (fN) atomicAdd(&cnt[9], 1); }
    if (Jt <= 0.0) { atomicAdd(&cnt[10], 1); if (fN) atomicAdd(&cnt[11], 1); }
    if (Je <= 0.0) { atomicAdd(&cnt[12], 1); if (fN) atomicAdd(&cnt[13], 1); }
    if (J0 <= 0.0) atomicAdd(&cnt[14], 1);
    if (fF) atomicAdd(&cnt[23], 1);
    if (mi < mbar) atomicAdd(&cnt[24], 1);
    if (k > 0 || Jt <= 0.0 || Ju <= 0.0 || Je <= 0.0) {
        int r = atomicAdd(&cnt[31], 1);
        if (r < cap) { float* w = rows + (size_t)r*16; int c = q >> 3;
            w[0] = (float)q; w[1] = (float)k; w[2] = (float)J0; w[3] = (float)Jt; w[4] = (float)Jh; w[5] = (float)Ju; w[6] = (float)Jn; w[7] = (float)Je;
            w[8] = Vq[q]; w[9] = (float)fN; w[10] = (mbar > 0.f) ? mi/mbar : mi; w[11] = (float)fF; w[12] = (float)fW; w[13] = am; w[14] = vm; w[15] = (float)cb[c*3+2]; }
    }
}
"""


def _inv_src():
    """aw_inv_q (fem_ao_rt.AW_SRC) 글자 그대로 + 점별 k 출력 kq (fem_wminv._inv_q 와 같은 바꾸기, 이름만 s13_)."""
    from .fem_ao_rt import AW_SRC
    i = AW_SRC.index("__global__ void aw_inv_q(")
    j = AW_SRC.index("\n}\n", i) + 3
    b = AW_SRC[i:j].replace("__global__ void aw_inv_q(", "S13_G void s13_inv_q(", 1)
    o1, n1 = "float e1, float e0) {", "float e1, float e0, int* kq) {"
    o2, n2 = "    if (k > 0) atomicMax(&kk[0], k);\n", "    kq[q] = k;\n    if (k > 0) atomicMax(&kk[0], k);\n"
    assert b.count(o1) == 1 and b.count(o2) == 1
    return b.replace(o1, n1).replace(o2, n2)


X_SIG_O = "atomicAdd(&S12A, m_*m_*q_/(s_*s_)); }"
X_SIG_N = "if (S13Q) S13Q[n] = m_*m_*q_/(s_*s_); }   // 절 CM86 X: 노드 버퍼 (합은 s13_sum 이 자른 칸 노드만)"
X_A_O = "    float s = sk[n]; int p = pm[n];\n    if (p == 2 && yl && mn[n] < mthr)"
X_A_N = "    if (S13Q) S13Q[n] = 0.0; if (S13M) S13M[n] = 0;   // 절 CM86 X: 걸음마다 비움\n" + X_A_O
X_I_O = "    if (k > 0) atomicMax(&kk[0], k);\n"
X_I_N = "    if (k > 0) { atomicMax(&kk[0], k); if (S13M) for (int o_=0;o_<27;o_++) S13M[ids[o_]] = 1; }   // 절 CM86 X: 자른 점의 27 노드 표시\n"
X_SUM = r"""
// ---- 절 CM86 X: S12A = Σ_{표시 노드} S13Q (반복 0 'r' 판정 s12_r 바로 앞; S12A 는 ao_floor_n 이 걸음마다 0)
S13_G void s13_sum(const int* IP) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || !S13M || !S13Q) return;
    if (S13M[n]) { double v = S13Q[n]; if (v != 0.0) atomicAdd(&S12A, v); }
}
"""


def _patch_x(src):
    assert src.count(X_SIG_O) == 1 and src.count(X_A_O) == 1, ("절 CM86 X 패치 자리 (speed12 z 1 위)", src.count(X_SIG_O), src.count(X_A_O))
    i = src.index("void aw_inv_q(")
    j = src.index(X_I_O, i)
    assert j < src.index("\n}\n", i)
    src = src[:j] + X_I_N + src[j + len(X_I_O):]
    src = src.replace(X_SIG_O, X_SIG_N).replace(X_A_O, X_A_N)
    src = src.replace('extern "C" {', 'extern "C" {\n#define S13_G __global__\n__device__ double* S13Q;\n__device__ int* S13M;', 1)
    i = src.rstrip().rfind("}")
    src = src[:i] + X_SUM + "\n}\n"
    C.done.add("X 신호 국소화")
    return src


def _patch(src):
    if not isinstance(src, str) or "aw_y0z_a(" not in src or "s13_" in src:
        return src
    if C.x:
        src = _patch_x(src)
    if C.dg:
        i = src.rstrip().rfind("}")
        src = src[:i] + (S13_SRC.replace("#define S13_G __global__\n", "") if C.x else S13_SRC) + _inv_src() + "\n}\n"
        C.done.add("진단 커널")
    return src


def _install_src():
    from . import fem_ao_rt as AORT
    oa = AORT._src

    def az(*a, **kw):
        r = _patch(oa(*a, **kw))
        print(f"[절 CM86] speed13 소스 {sorted(C.done)}", flush=True)
        return r
    AORT._src = az


def _P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _fn(s, name):
    mods = [s.mod] + (list(s._mods()) if hasattr(s, "_mods") else [])
    for m in mods:
        try:
            return m.get_function(name)
        except Exception:
            continue
    raise RuntimeError(f"{name} 없음")


CAP = 1 << 16


def rdiag(s, trig):
    from . import fem_rt_cuda as RTC
    try:
        from . import fem_speed6 as S6
        fr = int(getattr(S6.W, "fr", -1))
    except Exception:
        fr = -1
    try:
        IPh = s.IP.cpu().tolist() if torch.is_tensor(s.IP) else list(s.IP)
        nN, nC, nQ = int(IPh[9]), int(IPh[11]), int(IPh[12])
        dev = s.dev
        if getattr(s, "_s13cnt", None) is None:
            s._s13cnt = torch.zeros(32, device=dev, dtype=torch.int32)
            s._s13rows = torch.zeros(CAP * 16, device=dev, dtype=torch.float32)
        if getattr(s, "_s13kq", None) is None or s._s13kq.numel() < max(nQ, 1):
            s._s13kq = torch.zeros(max(nQ, 1) * 2, device=dev, dtype=torch.int32)
            s._s13kk = torch.zeros(2, device=dev, dtype=torch.int32)
        if getattr(s, "_s13nf", None) is None or s._s13nf.numel() < max(nN, 1):
            s._s13nf = torch.zeros(max(nN, 1) * 2, device=dev, dtype=torch.int32)
        ga = s.ga_c
        af = float(ga["af"])
        n3 = nN * 3
        U = s.U.reshape(-1)[:n3].float()
        vk = s.vk.reshape(-1)[:n3].float()
        g = torch.tensor([float(q) for q in s.g], device=dev, dtype=torch.float32).repeat(nN)
        tau, dt = float(s.tau), float(s.dt)
        yz = s.awyb.reshape(-1)[:n3].contiguous()
        dy = s.awdy.reshape(-1)[:n3]
        yt = (yz + dy).contiguous()
        yn = (af * U + (1.0 - af) * (U + dt * vk) + tau * tau * g).contiguous()
        ut = s.Ut.reshape(-1)[:n3].contiguous()
        yh = s.yh.reshape(-1)[:n3].contiguous()
        aw_end = s._aw_end()
        e1, e0 = float(aw_end[1]), float(aw_end[2])
        ue = (e1 * yt + e0 * U).contiguous() if s.lsend else None
        nacc = (ut - yn).view(nN, 3).norm(dim=1).contiguous()
        nvel = ((1.0 - af) * dt * vk).view(nN, 3).norm(dim=1).contiguous()
        s._s13nf.zero_(); s._s13cnt.zero_(); s._s13kq.zero_(); s._s13kk.zero_()
        fw = getattr(s, "g5w", None); wid = getattr(s, "l3id", None); fm = getattr(s, "g3fm", None)
        mbar = float(getattr(s, "mthr", 0.0) or 0.0)
        with RTC._Stream():
            if fw is not None and wid is not None and fm is not None and nC > 0:
                _fn(s, "s13_nf")(((nC + 127) // 128,), (128,), (_P(s.IP), _P(s.cb), _P(fm), _P(fw), _P(wid), _P(s._s13nf)))
            if nQ > 0:
                _fn(s, "s13_inv_q")(((nQ + 127) // 128,), (128,), (_P(s.IP), _P(s.cb), _P(s.BG), _P(s.Fq), _P(s.Vq), _P(yz), _P(dy.contiguous()), _P(s._s13kk),
                                                                   _P(s.U) if s.lsend else np.uint64(0), np.float32(e1), np.float32(e0), _P(s._s13kq)))
                _fn(s, "s13_q")(((nQ + 127) // 128,), (128,), (_P(s.IP), _P(s.cb), _P(s.BG), _P(s.Fq), _P(s.Vq), _P(yz), _P(yt), _P(yh), _P(ut), _P(yn), _P(ue),
                                                               _P(s._s13kq), _P(s._s13nf), _P(s.mn), np.float32(mbar), _P(s.an), _P(s.sk), _P(nacc), _P(nvel),
                                                               _P(s._s13cnt), _P(s._s13rows), np.int32(CAP)))
        torch.cuda.synchronize()
        cnt = s._s13cnt.cpu().tolist()
        nr = min(int(cnt[31]), CAP)
        R = s._s13rows[:nr * 16].view(nr, 16).cpu().numpy() if nr else np.zeros((0, 16), np.float32)
        nf = s._s13nf[:nN] > 0
        act = s.mn[:nN] > 0
        oth = act & ~nf

        def st(t):
            if t.numel() == 0:
                return None
            t = t.double()
            return [round(float(t.median()), 7), round(float(t.mean()), 7), round(float(t.max()), 7)]
        K = R[R[:, 1] > 0] if nr else R
        def qd(col, A=K):
            if len(A) == 0:
                return None
            v = A[:, col].astype(np.float64)
            return [round(float(np.quantile(v, p)), 5) for p in (0.0, 0.5, 1.0)]
        beta = float(s.awb[0].item()) if getattr(s, "awb", None) is not None else float("nan")
        rec = dict(fr=fr, trig=int(trig), beta=beta, kmax=int(K[:, 1].max()) if len(K) else 0, cnt=cnt,
                   nnode=int(act.sum()), nnf=int((nf & act).sum()),
                   acc_nf=st(nacc[:nN][nf & act]), acc_oth=st(nacc[:nN][oth]), vel_nf=st(nvel[:nN][nf & act]), vel_oth=st(nvel[:nN][oth]),
                   kJt=qd(3), kJh=qd(4), kJu=qd(5), kJn=qd(6), kJe=qd(7), kmm=qd(10), kam=qd(13), kvm=qd(14), kz=qd(15),
                   zall=qd(15, R) if nr else None, ex=[[round(float(x), 5) for x in r] for r in K[np.argsort(-K[:, 1])][:24]] if len(K) else [])
        C.log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        C.log.flush()
        C.nrec += 1
    except Exception:
        C.nerr += 1
        if C.nerr <= 3:
            import traceback
            with open(os.path.join(C.out, "s13_err.txt"), "a") as f_:
                f_.write(traceback.format_exc()[-3000:] + "\n")


class _XBN:
    """s._xbn 감쌈: replay 의 'r' 경계 판정 읽기 (int(s._xbn[3])) 순간 진단 (speed12 와 같은 꼴)."""
    def __init__(self, s, a):
        self.s, self.a = s, a

    def __getitem__(self, i):
        v = self.a[i]
        if i == 3 and int(self.a[2]) == 0:
            rdiag(self.s, int(v))
        return v

    def __setitem__(self, i, v):
        self.a[i] = v

    def __len__(self):
        return len(self.a)


def _install_dg():
    from . import fem_speed4 as S4
    o = S4._xb_buf

    def _xb_buf(s):
        o(s)
        if not isinstance(s._xbn, _XBN):
            s._xbn = _XBN(s, s._xbn)
    S4._xb_buf = _xb_buf


def _glob(s, name):
    mods = [s.mod] + (list(s._mods()) if hasattr(s, "_mods") else [])
    for m in mods:
        try:
            return m.get_global(name)
        except Exception:
            continue
    raise RuntimeError(f"{name} 없음")


def _bind_x(s):
    """캡처 밖: 노드 버퍼 (용량 바뀌면 새로) 와 장치 포인터 S13Q·S13M, s13_sum 손잡이."""
    import cupy as cp
    cap = int(s.capN)
    if getattr(s, "_s13Qb", None) is None or s._s13Qb.numel() < cap:
        s._s13Qb = torch.zeros(cap, device=s.dev, dtype=torch.float64)
        s._s13Mb = torch.zeros(cap, device=s.dev, dtype=torch.int32)
    key = (id(s.mod), int(s._s13Qb.data_ptr()), int(s._s13Mb.data_ptr()))
    if getattr(s, "_s13key", None) != key:
        torch.cuda.synchronize()
        cp.ndarray((1,), dtype=cp.uint64, memptr=_glob(s, "S13Q"))[0] = int(s._s13Qb.data_ptr())
        cp.ndarray((1,), dtype=cp.uint64, memptr=_glob(s, "S13M"))[0] = int(s._s13Mb.data_ptr())
        cp.cuda.runtime.deviceSynchronize()
        s._s13sum = _fn(s, "s13_sum")
        s._s13key = key
        C.nbind = getattr(C, "nbind", 0) + 1


def _install_ctl():
    from . import fem_ao_rt as AORT
    from . import fem_rt_cuda as RTC
    cls = AORT.AORTSolver
    o_ctl = cls.ctl

    def ctl(s, op, it=0):
        cap = getattr(s, "_c3cap", None) is not None
        if not cap and getattr(s, "mod", None) is not None:
            try:
                _bind_x(s)
            except Exception as e:                                                     # 모듈이 아직 없을 때 (첫 컴파일 앞) 는 다음 부름에서
                C.berr = repr(e)[:300]
        if C.x and op == 3 and it == 0 and cap and getattr(s, "_x1K", 0) == 1 and getattr(s, "_xbd", None) is not None and getattr(s, "awb", None) is not None:
            assert getattr(s, "_s13key", None) is not None, "절 CM86 X: S13Q·S13M 포인터를 캡처 앞에 못 묶음 " + str(getattr(C, "berr", ""))
            with RTC._Stream():
                s._s13sum((((int(s.capN) + 255) // 256),), (256,), (_P(s.IP),))
            C.nsum = getattr(C, "nsum", 0) + 1
        return o_ctl(s, op, it)
    cls.ctl = ctl


def install(out, dg=0, x=0):
    """실행기를 runpy 로 띄우는 순간 (speed12 설치 뒤 = 가장 바깥) 에 부름."""
    C.out, C.dg, C.x = out, int(dg), int(x)
    if C.dg or C.x:
        _install_src()
    if C.x:
        from . import fem_speed12 as S12
        assert S12.C.z == 1, "--s13x 는 H8 (speed12 z 1) 위"
        _install_ctl()
    if C.dg:
        C.log = open(os.path.join(out, "s13.jsonl"), "w")
        _install_dg()
    import atexit

    def _fin():
        with open(os.path.join(out, "s13info.json"), "w") as f_:
            json.dump(dict(dg=C.dg, x=C.x, done=sorted(C.done), nbind=getattr(C, "nbind", 0), nsum=getattr(C, "nsum", 0), berr=getattr(C, "berr", None), nrec=C.nrec, nerr=C.nerr), f_)
    _fin()
    atexit.register(_fin)
    print(f"[절 CM86] speed13 켬 진단 {C.dg} X {C.x}", flush=True)
