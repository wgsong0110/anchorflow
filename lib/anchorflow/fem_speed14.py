"""절 CM88 묶음 speed14 (11-08 위임, 수박 > 16.7 ms 프레임): 가벼움 벌점 강성이 바닥 띠 무거운 칸의 HHT 가속도 기억을 키우는 경로 진단 + 고침.
감싸개 exe/speed14_wrap.py 가 부름 (exe/speed13_wrap.py 바깥). 옛 실행기 (수박, AORTSolver) 만.

배경 (절 CM86): H9 의 남은 β < 1 걸음 (나) = 바닥 띠 무거운 칸에서 관성 예측 ŷ (Ut) 자체가 눌림, 그 압축의 약 절반이 가속도 기억 a_n 몫
  Ut − Yn = c_a a_n,  c_a = (1 − α_f) Δt² (½ − β)  (HHT α_m = 0; ao_step0_ga 의 ys 식).
  걸음 끝 a_{n+1} = (y − ys)/τ² (ao_stepend_ga 의 Newmark 속도 식과 같은 값) = g + M⁻¹ f(y) (풀이가 맞으면; f = α 가중 위치 y 의 힘 = −∇E(y)).
  -> a_{n+1} 의 벌점 몫 = −M⁻¹ ∇E_벌점(y) 이고 이것이 다음 걸음 ŷ 에 c_a 를 곱해 들어감.

(D) --s14dg 1 (결과 불변; speed13 진단 '@s13dg 1' 위에서만):
    걸음 끝 (그래프 안, 갱신 커널 바로 뒤): 벌점 강성 힘 ∇E_K(y) 를 gq_force (ghost6 커널 글자 그대로, cm 0 = 고스트 질량 몫 없음) 로 두 번 --
      가벼움 몫 (면 가중 fw − 1[P1X], lightg3/lightg4 의 r κ λ 몫) · 강성 전체 (면 가중 fw) -- 노드 가속도 a_L = −∇E_L/m, a_K = −∇E_K/m (노드 버퍼),
      프레임 끝 G2P (ao_g2p_a 와 같은 커널·같은 가중) 로 입자 버퍼 pL·pK 에 (a_n 이 입자 pa 로 넘어가는 것과 같은 길).
    다음 'r' 경계 (그래프 밖, speed13 rdiag 뒤): pL·pK·pa 를 입자 질량 가중 P2G (ao_p2g_a, 분모 = 같은 커널로 1 을 모음) 로 지금 노드에 -> a_L'·a_K'·a_n'
      (실행기의 a_n 은 --gq_mass 가우스 점 경로라 a_n' 은 대조용 근사; 표에 실제 gaA 와의 차를 적음).
      필터를 자른 점 (speed13 의 점별 k > 0) 마다: J (Ut) · J (Ut − c_a a_L') · J (Ut − c_a a_K') -- 가속도 기억에서 벌점 몫을 빼면 안 뒤집히는지,
      27 노드 c_a |a_n| · c_a |a_L'| · c_a |a_K'| 최대, a_n 과 a_L' 의 방향 (cos), 칸에서 가장 가까운 벌점 면 칸까지 체비셰프 거리 (칸; 가벼움 면 · 강성 전체 면).
      몸 전체: 무거운 노드 (m ≥ 100 m̄) · 나머지 활성 노드의 |a_L'|/|a_n'| 중앙.
    -> <out>/s14.jsonl (걸음마다 한 줄).
(A) --s14a 1|2 (결과 바뀜; 고침): 걸음 끝 가속도 기억 a_{n+1} 에서 벌점 강성 몫을 뺌 -- ga ← ga − a_벌점 (a_벌점 = −∇E_벌점(y)/m, 위와 같은 커널).
      1 = 가벼움 몫만 (fw − 1[P1X]) · 2 = 벌점 강성 전체 (fw). 속도·변위 갱신은 그대로 (그 걸음 풀이의 힘을 다 씀) -- 다음 걸음 관성 예측 ŷ 의 가속도 몫에만 벌점이 안 들어감.
      벌점 = 수치 안정화 몫 (물리 힘 아님) 이라 HHT 가 '지난 걸음 가속도' 로 기억해 다음 걸음에 다시 밀지 않게 함.
      갱신 커널이 기억을 이미 0 으로 둔 노드 (고착·구속·후퇴 오일러 노드; 세 성분 모두 0) 는 그대로 0. 새 문턱·클램프 없음.
(M) --s14m 1 (결과 바뀜; 고침 M, H10 = H9 + M): 가속도 기억을 입자로 넘기는 G2P (ao_g2p_a: pa = Σ_K w_K a_K) 를 노드 질량 가중으로
      pa = Σ_K w_K m_K a_K / Σ_K w_K m_K  (운동량 (힘) 일치 꼴; 분모 0 이면 0). 가벼운 노드 (m ~ m̄) 는 벌점 강성에 끌려 가속도가 크지만 (그 노드의 힘은
      다른 힘과 맞서 균형) 운동량은 작음 -- 무게 없이 섞으면 그 큰 가속도가 바닥 띠 무거운 칸 입자의 기억으로 건너가 다음 걸음 ŷ 를 누름.
      속도·위치·F G2P 는 그대로, 기억 a_n 의 건너감만. 새 문턱·클램프 없음. 진단 판에서는 같은 커널로 입자 버퍼 pM 도 만들어 'r' 경계에서
      J (Ut + c_a (a_M' − a_P')) (a_P' = 지금 pa 를 같은 P2G 로 옮긴 값) 와 가벼운 노드 몫을 봄.
(T) --s14ct 1 (결과 불변; 단독 판 시간 가르기): 프레임 끝 G2P (ao_g2p_a) 바로 뒤 그래프 안에서 제어 셈 CT (뉴턴 반복 CT[2] · PCG 반복 CT[3] · PCG 시작 CT[7] · 걸음 CT[6])
      를 장치 고리 버퍼에 차례로 복사 (작은 커널 하나, 호스트 동기화 없음) -> 끝에 <out>/s14ct.json (기록 순서 = 프레임·시도 순서; 그래프 밖 데움 record 포함).
"""
import json
import os

import numpy as np
import torch


class C:
    out = "."
    dg = 0
    a = 0
    m = 0
    ct = 0
    log = None
    nend = 0
    nfix = 0
    ng2p = 0
    nrec = 0
    nerr = 0
    ebuf = None
    done = set()


STEPEND = ("ao_stepend_ga", "ao_stepend_uf", "ao_stepend_ufl", "en_stepend")
GQF_SIG = "gq_force(const int* IP, const int* cb, const unsigned char* fm, const float* fw, float k, float cm, const float* y, const float* yh, const float* y0, double* f, float* B, double* E)"

S14_SRC = r"""
// ---- 절 CM88 speed14 (lib/anchorflow/fem_speed14.py) ----
__global__ void s14_wd(const int* IP, const float* fw, const float* wid, int mode, float* wd) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= IP[11]) return;
    for (int a=0;a<3;a++) { float w = fw ? fw[c*3+a] : 1.f; float b = wid ? wid[c*3+a] : 0.f; wd[c*3+a] = (mode == 1) ? (w - b) : w; }
}
__global__ void s14_ct(const int* CT, int* ring, int* idx, int cap) {
    if (blockIdx.x*blockDim.x + threadIdx.x != 0) return;
    int i = idx[0];
    if (i < cap) for (int t=0;t<16;t++) ring[i*16+t] = CT[t];
    idx[0] = i + 1;
}
// 질량 가중 가속도 G2P (고침 M): pa = Σ w m a / Σ w m
__global__ void s14_g2pm(int N, const int* IP, const int* pb, const float* fx, const float* ga, const float* mn, float* pa) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    double s[3] = {0.0, 0.0, 0.0}, sw = 0.0;
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        double ww = (double)(w[0][i]*w[1][j]*w[2][k])*(double)mn[nid];
        sw += ww;
        for (int a=0;a<3;a++) s[a] += ww*(double)ga[nid*3+a];
    }
    for (int a=0;a<3;a++) pa[p*3+a] = (sw > 0.0) ? (float)(s[a]/sw) : 0.f;
}
// 진단: 입자마다 기억 |Σ w a| 중 가벼운 노드 (m < 100 m̄) 몫의 크기 |Σ_{가벼움} w a| -> pl (1 성분)
__global__ void s14_lshare(int N, const int* IP, const int* pb, const float* fx, const float* ga, const float* mn, float mcut, float* pl) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float s[3] = {0.f, 0.f, 0.f};
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        if (!(mn[nid] < mcut)) continue;
        float ww = w[0][i]*w[1][j]*w[2][k];
        for (int a=0;a<3;a++) s[a] += ww*ga[nid*3+a];
    }
    pl[p] = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
}
__global__ void s14_zd(const int* IP, double* f) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]*3) return;
    f[n] = 0.0;
}
// a_벌점 = −∇E_벌점/m (ap 에 씀); fix 면 기억이 남은 노드 (세 성분 중 하나라도 0 아님) 의 ga 에서 뺌
__global__ void s14_ap(const int* IP, const float* mn, const double* f, float* ga, float* ap, int fix) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n];
    bool kept = (ga[n*3] != 0.f) || (ga[n*3+1] != 0.f) || (ga[n*3+2] != 0.f);
    for (int a=0;a<3;a++) {
        float v = (m > 0.f) ? (float)(-f[n*3+a]/(double)m) : 0.f;
        if (ap) ap[n*3+a] = v;
        if (fix && kept) ga[n*3+a] -= v;
    }
}
"""

S14_DG_SRC = r"""
__global__ void s14_q(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const int* kq,
                      const float* ut, const float* ut2, const float* ut3, const float* ga, const float* nl, const float* na, const float* np_, float ca,
                      int* cnt, float* rows, int cap) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    if (!(Vq[q] > 0.f)) return;
    int k = kq[q]; if (k <= 0) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int t=0;t<9;t++) F[t] = Fq[(size_t)q*9+t];
    double Fd[9], Cf[9]; for (int t=0;t<9;t++) Fd[t] = (double)F[t]; aw_cof(Fd, Cf);
    double dF = Fd[0]*Cf[0] + Fd[1]*Cf[1] + Fd[2]*Cf[2];
    double Ju = s13_det(ids, oc, BG, F, ut)/dF, Ju2 = s13_det(ids, oc, BG, F, ut2)/dF, Ju3 = s13_det(ids, oc, BG, F, ut3)/dF;
    float am = 0.f, lm = 0.f, km = 0.f, pm = 0.f; double dot = 0.0, g2 = 0.0, l2 = 0.0;
    for (int o=0;o<27;o++) { int id = ids[o]; float g[3], l[3], A[3], p[3];
        for (int a=0;a<3;a++) { g[a] = ga[id*3+a]; l[a] = nl[id*3+a]; A[a] = na[id*3+a]; p[a] = np_[id*3+a]; }
        am = fmaxf(am, sqrtf(g[0]*g[0]+g[1]*g[1]+g[2]*g[2])); lm = fmaxf(lm, sqrtf(l[0]*l[0]+l[1]*l[1]+l[2]*l[2]));
        km = fmaxf(km, sqrtf(A[0]*A[0]+A[1]*A[1]+A[2]*A[2])); pm = fmaxf(pm, sqrtf(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]));
        for (int a=0;a<3;a++) { dot += (double)g[a]*l[a]; g2 += (double)g[a]*g[a]; l2 += (double)l[a]*l[a]; } }
    atomicAdd(&cnt[0], 1); if (Ju <= 0.0) atomicAdd(&cnt[1], 1); if (Ju2 <= 0.0) atomicAdd(&cnt[2], 1); if (Ju3 <= 0.0) atomicAdd(&cnt[3], 1);
    int r = atomicAdd(&cnt[31], 1);
    if (r < cap) { float* w = rows + (size_t)r*12; int c = q >> 3;
        w[0] = (float)q; w[1] = (float)k; w[2] = (float)Ju; w[3] = (float)Ju2; w[4] = (float)Ju3;
        w[5] = ca*am; w[6] = ca*lm; w[7] = ca*km; w[8] = ca*pm; w[9] = (g2 > 0.0 && l2 > 0.0) ? (float)(dot/sqrt(g2*l2)) : 0.f; w[10] = (float)c; w[11] = (float)cb[c*3+2]; }
}
"""


def _patch(src):
    if not isinstance(src, str) or "aw_y0z_a(" not in src or "s14_" in src:
        return src
    assert src.count(GQF_SIG) == 1, "절 CM88: ghost6 gq_force 글자가 없음/여럿"
    add = S14_SRC
    if C.dg:
        assert "s13_det(" in src, "절 CM88 진단은 speed13 진단 ('@s13dg 1') 위"
        add += S14_DG_SRC
    i = src.rstrip().rfind("}")
    C.done.add("커널")
    return src[:i] + add + "\n}\n"


def _install_src():
    from . import fem_ao_rt as AORT
    oa = AORT._src

    def az(*a, **kw):
        r = _patch(oa(*a, **kw))
        print(f"[절 CM88] speed14 소스 {sorted(C.done)}", flush=True)
        return r
    AORT._src = az


def _P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _fn(s, name):
    if not hasattr(s, "_s14fn"):
        s._s14fn = {}
    key = (id(s.mod), name)
    if key not in s._s14fn:
        mods = [s.mod] + (list(s._mods()) if hasattr(s, "_mods") else [])
        h = None
        for m in mods:
            try:
                h = m.get_function(name)
                break
            except Exception:
                continue
        if h is None:
            raise RuntimeError(f"{name} 없음")
        s._s14fn[key] = h
    return s._s14fn[key]


def _cap():
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _alloc(s):
    """캡처 밖에서만: 노드 힘 (f64) · 가중 · 노드 a_L·a_K · 입자 pL·pK."""
    capN, capC, N = int(s.capN), int(s.capQc), int(s.N)
    ok = (getattr(s, "_s14f", None) is not None and s._s14f.numel() >= capN * 3 and s._s14wd.numel() >= capC * 3
          and (not C.dg or (s._s14pL.numel() >= N * 3 and s._s14aL.numel() >= capN * 3)))
    if ok:
        return True
    if _cap():
        return False
    dev = s.dev
    s._s14f = torch.zeros(capN * 3, device=dev, dtype=torch.float64)
    s._s14wd = torch.zeros(capC * 3, device=dev, dtype=torch.float32)
    if C.dg:
        s._s14aL = torch.zeros(capN * 3, device=dev, dtype=torch.float32)
        s._s14aK = torch.zeros(capN * 3, device=dev, dtype=torch.float32)
        s._s14pL = torch.zeros(N * 3, device=dev, dtype=torch.float32)
        s._s14pK = torch.zeros(N * 3, device=dev, dtype=torch.float32)
    return True


def _pen(s, mode, ap, fix):
    """∇E_벌점(y) (gq_force, ghost6 강성 몫만) -> a_벌점 = −∇E/m 를 ap 에 (없으면 안 씀), fix 면 ga 에서 뺌. 그래프 안에서도 (지금 스트림)."""
    from . import fem_ghost6 as G6
    from . import fem_rt_cuda as RTC
    capN, capC = int(s.capN), int(s.capQc)
    fw = getattr(s, "g5w", None)
    wid = getattr(s, "l3id", None)
    Y0 = _P(s.g6u) if G6.CONS else np.uint64(0)
    k = float(getattr(s, "_gpk", 0.0) or 0.0)
    if mode == 1 and wid is None:                                                       # 가벼움 벌점이 없는 씬 (κ 0·m̄ 0): 가벼움 몫 0
        return
    with RTC._Stream():
        _fn(s, "s14_wd")(((capC + 127) // 128,), (128,), (_P(s.IP), G6.FWP(s), _P(wid) if mode == 1 else np.uint64(0), np.int32(mode), _P(s._s14wd)))
        _fn(s, "s14_zd")(((capN * 3 + 255) // 256,), (256,), (_P(s.IP), _P(s._s14f)))
        _fn(s, "gq_force")(((capC + 127) // 128,), (128,), (_P(s.IP), _P(s.cb), _P(s.g3fm), _P(s._s14wd), np.float32(k), np.float32(0.0), _P(s.y), np.uint64(0), Y0,
                                                             _P(s._s14f), np.uint64(0), np.uint64(0)))
        _fn(s, "s14_ap")(((capN + 255) // 256,), (256,), (_P(s.IP), _P(s.mn), _P(s._s14f), _P(s.gaA), _P(ap), np.int32(1 if fix else 0)))


def _after_end(s):
    if not (C.dg or C.a):
        return
    if getattr(s, "g3fm", None) is None or getattr(s, "_gpk", None) is None:
        return
    if not _alloc(s):
        raise RuntimeError("절 CM88: 캡처 중 버퍼 용량 부족 (캡처 밖 데움에서 못 잡음)")
    if C.dg:
        _pen(s, 1, s._s14aL, fix=(C.a == 1))
        _pen(s, 2, s._s14aK, fix=(C.a == 2))
    else:
        _pen(s, C.a, None, fix=True)
    C.nend += 1
    if C.a:
        C.nfix += 1


def _pre_g2p(s, args):
    """진단: 실제 G2P 앞에 같은 노드 기억 a 로 질량 가중 G2P (pM) 와 가벼운 노드 몫 (pLs) 을 입자 버퍼에 (그래프 안)."""
    from . import fem_rt_cuda as RTC
    if not C.dg:
        return
    N = int(s.N)
    if getattr(s, "_s14pM", None) is None or s._s14pM.numel() < N * 3:
        if _cap():
            raise RuntimeError("절 CM88: 캡처 중 pM 버퍼 없음")
        s._s14pM = torch.zeros(N * 3, device=s.dev, dtype=torch.float32)
        s._s14pLs = torch.zeros(N, device=s.dev, dtype=torch.float32)
    mbar = float(getattr(s, "mthr", 0.0) or 0.0)
    with RTC._Stream():
        _fn(s, "s14_g2pm")(((N + 127) // 128,), (128,), (np.int32(N), args[1], args[2], args[3], args[4], _P(s.mn), _P(s._s14pM)))
        _fn(s, "s14_lshare")(((N + 127) // 128,), (128,), (np.int32(N), args[1], args[2], args[3], args[4], _P(s.mn), np.float32(100.0 * mbar), _P(s._s14pLs)))


CTCAP = 8192


def _after_g2p_ct(s):
    from . import fem_rt_cuda as RTC
    if getattr(s, "_s14ring", None) is None:
        if _cap():
            raise RuntimeError("절 CM88 (T): 캡처 중 고리 버퍼 없음")
        s._s14ring = torch.zeros(CTCAP * 16, device=s.dev, dtype=torch.int32)
        s._s14idx = torch.zeros(1, device=s.dev, dtype=torch.int32)
        C.ctsolver = s
    with RTC._Stream():
        _fn(s, "s14_ct")((1,), (32,), (_P(s.CT), _P(s._s14ring), _P(s._s14idx), np.int32(CTCAP)))
    C.nct = getattr(C, "nct", 0) + 1


def _after_g2p(s):
    from . import fem_rt_cuda as RTC
    if C.ct:
        _after_g2p_ct(s)
    if not C.dg or getattr(s, "_s14pL", None) is None:
        return
    N = int(s.N)
    with RTC._Stream():
        for a_, p_ in ((s._s14aL, s._s14pL), (s._s14aK, s._s14pK)):
            _fn(s, "ao_g2p_a")(((N + 127) // 128,), (128,), (np.int32(N), _P(s.IP), _P(s.Np), _P(s.fx), _P(a_), _P(p_)))
    C.ng2p += 1


def _install_k():
    from . import fem_ao_rt as AORT
    cls = AORT.AORTSolver
    o_k = cls.k

    def k(s, name, n, args, b=128):
        if name == "ao_g2p_a" and (C.m or C.dg):
            _pre_g2p(s, args)
        if name == "ao_g2p_a" and C.m:
            from . import fem_rt_cuda as RTC
            N = int(s.N)
            with RTC._Stream():
                _fn(s, "s14_g2pm")(((N + 127) // 128,), (128,), (np.int32(N), args[1], args[2], args[3], args[4], _P(s.mn), args[5]))
            C.ng2pm = getattr(C, "ng2pm", 0) + 1
            r = None
        else:
            r = o_k(s, name, n, args, b)
        if name in STEPEND:
            _after_end(s)
        elif name == "ao_g2p_a":
            _after_g2p(s)
        return r
    cls.k = k


CAP = 1 << 15


def _p2g(s, pa, out, den):
    from . import fem_rt_cuda as RTC
    N = int(s.N)
    out.zero_()
    with RTC._Stream():
        _fn(s, "ao_p2g_a")(((N + 127) // 128,), (128,), (np.int32(N), _P(s.IP), _P(s.Np), _P(s.fx), _P(s.m), _P(pa), _P(out)))
    torch.cuda.synchronize()
    o = out.view(-1, 3)
    d = den.view(-1, 3)[:, :1]
    return torch.where(d > 0, o / d.clamp_min(1e-30), torch.zeros_like(o))


def rdiag(s, trig):
    """speed13 rdiag 뒤 ('r' 경계, 그래프 밖, 읽기만)."""
    from . import fem_rt_cuda as RTC
    try:
        from . import fem_speed6 as S6
        fr = int(getattr(S6.W, "fr", -1))
    except Exception:
        fr = -1
    try:
        if getattr(s, "_s14pL", None) is None or getattr(s, "_s13kq", None) is None:
            return
        IPh = s.IP.cpu().tolist() if torch.is_tensor(s.IP) else list(s.IP)
        nN, nC, nQ = int(IPh[9]), int(IPh[11]), int(IPh[12])
        dev = s.dev
        N = int(s.N)
        cap3 = int(s.capN) * 3
        if getattr(s, "_s14nb", None) is None or s._s14nb.numel() < cap3:
            s._s14nb = torch.zeros(cap3, device=dev, dtype=torch.float32)
            s._s14dn = torch.zeros(cap3, device=dev, dtype=torch.float32)
            s._s14cnt = torch.zeros(32, device=dev, dtype=torch.int32)
            s._s14rows = torch.zeros(CAP * 12, device=dev, dtype=torch.float32)
        if getattr(s, "_s14one", None) is None or s._s14one.numel() < N * 3:
            s._s14one = torch.ones(N * 3, device=dev, dtype=torch.float32)
        s._s14dn.zero_()
        with RTC._Stream():
            _fn(s, "ao_p2g_a")(((N + 127) // 128,), (128,), (np.int32(N), _P(s.IP), _P(s.Np), _P(s.fx), _P(s.m), _P(s._s14one), _P(s._s14dn)))
        torch.cuda.synchronize()
        den = s._s14dn.clone()
        aL = _p2g(s, s._s14pL, s._s14nb, den).reshape(-1).clone()
        aK = _p2g(s, s._s14pK, s._s14nb, den).reshape(-1).clone()
        aP = _p2g(s, s.pa, s._s14nb, den).reshape(-1).clone()
        aM = _p2g(s, s._s14pM, s._s14nb, den).reshape(-1).clone() if getattr(s, "_s14pM", None) is not None else aP.clone()
        ga_c = s.ga_c
        af, be = float(ga_c["af"]), float(ga_c["be"])
        dt = float(s.dt)
        ca = (1.0 - af) * dt * dt * (0.5 - be)
        n3 = nN * 3
        ut = s.Ut.reshape(-1)[:cap3].contiguous()
        ut2 = (ut - ca * aL[:cap3]).contiguous()
        ut3 = (ut - ca * aK[:cap3]).contiguous()
        utM = (ut + ca * (aM[:cap3] - aP[:cap3])).contiguous()
        ga = s.gaA.reshape(-1)[:cap3].contiguous()
        aLc, aKc, aPc = aL[:cap3].contiguous(), aK[:cap3].contiguous(), aP[:cap3].contiguous()
        s._s14cnt.zero_()
        with RTC._Stream():
            if nQ > 0:
                _fn(s, "s14_q")(((nQ + 127) // 128,), (128,), (_P(s.IP), _P(s.cb), _P(s.BG), _P(s.Fq), _P(s.Vq), _P(s._s13kq), _P(ut), _P(ut2), _P(ut3), _P(ga),
                                                               _P(aLc), _P(aKc), _P(aPc), np.float32(ca), _P(s._s14cnt), _P(s._s14rows), np.int32(CAP)))
        torch.cuda.synchronize()
        cnt = s._s14cnt.cpu().tolist()
        # 고침 M 대조: 같은 자른 점에서 J (Ut + c_a (a_M' − a_P')) -- s14_q 를 ut2 자리에 utM 으로 한 번 더 (cnt 2 = J ≤ 0 수, 행 3 = J)
        cM = torch.zeros(32, device=dev, dtype=torch.int32); rM = torch.zeros(CAP * 12, device=dev, dtype=torch.float32)
        with RTC._Stream():
            if nQ > 0:
                _fn(s, "s14_q")(((nQ + 127) // 128,), (128,), (_P(s.IP), _P(s.cb), _P(s.BG), _P(s.Fq), _P(s.Vq), _P(s._s13kq), _P(ut), _P(utM), _P(utM), _P(ga),
                                                               _P(aLc), _P(aKc), _P(aPc), np.float32(ca), _P(cM), _P(rM), np.int32(CAP)))
        torch.cuda.synchronize()
        cMl = cM.cpu().tolist(); nrM = min(int(cMl[31]), CAP)
        JM = rM[:nrM * 12].view(nrM, 12)[:, 3].double() if nrM else None
        # 가벼운 노드 몫: 자른 칸 입자의 기억 중 가벼운 노드 몫 -- 몸 전체 입자 중앙 |pLs|/|pa| (입자 기준)
        try:
            pa_n = s.pa.view(-1, 3).norm(dim=1); okp = pa_n > 0
            lsh = float((s._s14pLs[:N][okp] / pa_n[okp]).double().median()) if okp.any() else None
            mMs = float(((s._s14pM.view(-1, 3).norm(dim=1)[okp]) / pa_n[okp]).double().median()) if okp.any() else None
        except Exception:
            lsh, mMs = None, None
        nr = min(int(cnt[31]), CAP)
        R = s._s14rows[:nr * 12].view(nr, 12) if nr else None
        mn = s.mn[:nN]
        act = mn > 0
        mbar = float(getattr(s, "mthr", 0.0) or 0.0)
        heavy = act & (mn >= 100.0 * mbar) if mbar > 0 else act
        nrm = lambda t: t[:n3].view(nN, 3).norm(dim=1)
        gN, lN, kN, pN = nrm(ga), nrm(aLc), nrm(aKc), nrm(aPc)

        def med(t):
            return round(float(t.double().median()), 6) if t.numel() else None
        okh = heavy & (gN > 0)
        oko = act & ~heavy & (gN > 0)
        rec = dict(fr=fr, trig=int(trig), beta=float(s.awb[0].item()) if getattr(s, "awb", None) is not None else None, ca=ca, ncut=int(cnt[0]),
                   cut_Ju_le0=int(cnt[1]), cut_JuL_le0=int(cnt[2]), cut_JuK_le0=int(cnt[3]),
                   heavy_rL=med(lN[okh] / gN[okh]), heavy_rK=med(kN[okh] / gN[okh]), oth_rL=med(lN[oko] / gN[oko]), oth_rK=med(kN[oko] / gN[oko]),
                   p2g_rel=med((pN[act & (gN > 0)] - gN[act & (gN > 0)]).abs() / gN[act & (gN > 0)]), nheavy=int(heavy.sum()), nact=int(act.sum()),
                   cut_JuM_le0=int(cMl[2]), JuM=[round(float(torch.quantile(JM, p)), 6) for p in (0.0, 0.5, 1.0)] if JM is not None and JM.numel() else None,
                   p_light_share=lsh, p_M_over_pa=mMs,
                   heavy_rM=med((nrm(aM[:cap3].contiguous()) - pN).abs()[okh] / gN[okh]) if okh.any() else None)
        if nr:
            Rc = R.double()
            def q3(col):
                v = Rc[:, col]
                return [round(float(torch.quantile(v, p)), 6) for p in (0.0, 0.5, 1.0)]
            rec.update(Ju=q3(2), JuL=q3(3), JuK=q3(4), am=q3(5), aLm=q3(6), aKm=q3(7), aPm=q3(8), cosL=q3(9), z=q3(11))
            # 벌점 면 칸까지 체비셰프 거리 (칸): 면 (c, a) 가중 > 0 이면 칸 c 와 c + e_a
            fm = s.g3fm.reshape(-1)[:nC].long()
            fw = s.g5w.reshape(-1)[:nC * 3].view(nC, 3) if getattr(s, "g5w", None) is not None else torch.ones(nC, 3, device=dev)
            wid = s.l3id.reshape(-1)[:nC * 3].view(nC, 3) if getattr(s, "l3id", None) is not None else torch.zeros(nC, 3, device=dev)
            cb = s.cb.reshape(-1)[:nC * 3].view(nC, 3).long()
            cut = cb[R[:, 10].long()].double()
            for nm_, wv in (("dL", fw - wid), ("dK", fw)):
                pts = []
                for a in range(3):
                    on = ((fm >> a) & 1).bool() & (wv[:, a] > 0)
                    if on.any():
                        e = torch.zeros(3, dtype=torch.long, device=dev); e[a] = 1
                        pts += [cb[on], cb[on] + e]
                if pts:
                    Pc = torch.cat(pts).double().unique(dim=0)
                    d = torch.cdist(cut, Pc, p=float("inf")).min(dim=1).values
                    rec[nm_] = [round(float(torch.quantile(d, p)), 2) for p in (0.0, 0.5, 1.0)]
                    rec[nm_ + "_n"] = int(Pc.shape[0])
                else:
                    rec[nm_] = None
        C.log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        C.log.flush()
        C.nrec += 1
    except Exception:
        C.nerr += 1
        if C.nerr <= 3:
            import traceback
            with open(os.path.join(C.out, "s14_err.txt"), "a") as f_:
                f_.write(traceback.format_exc()[-3000:] + "\n")


def _install_dg():
    from . import fem_speed13 as S13
    assert S13.C.dg == 1, "--s14dg 는 speed13 진단 ('@s13dg 1') 위"
    o = S13.rdiag

    def rd(s, trig):
        o(s, trig)
        rdiag(s, trig)
    S13.rdiag = rd


def install(out, dg=0, a=0, ct=0, m=0):
    """실행기를 runpy 로 띄우는 순간 (speed13 설치 뒤 = 가장 바깥) 에 부름."""
    C.out, C.dg, C.a, C.ct, C.m = out, int(dg), int(a), int(ct), int(m)
    assert C.a in (0, 1, 2)
    if C.dg or C.a or C.ct or C.m:
        _install_src()
        _install_k()
    if C.dg:
        C.log = open(os.path.join(out, "s14.jsonl"), "w")
        _install_dg()
    import atexit

    def _fin():
        ctn = None
        s = getattr(C, "ctsolver", None)
        if s is not None:
            try:
                torch.cuda.synchronize()
                n = min(int(s._s14idx.item()), CTCAP)
                R = s._s14ring[:n * 16].view(n, 16).cpu().tolist()
                with open(os.path.join(out, "s14ct.json"), "w") as f_:
                    json.dump(dict(n=int(s._s14idx.item()), cols="CT[0..15]: 2 뉴턴 반복 · 3 PCG 반복 · 4 · 5 · 6 걸음 · 7 PCG 시작", rows=R), f_)
                ctn = n
            except Exception as e:
                ctn = repr(e)[:200]
        with open(os.path.join(out, "s14info.json"), "w") as f_:
            json.dump(dict(dg=C.dg, a=C.a, ct=C.ct, m=C.m, ng2pm=getattr(C, "ng2pm", 0), done=sorted(C.done), nend=C.nend, nfix=C.nfix, ng2p=C.ng2p, nct=getattr(C, "nct", 0), ctn=ctn, nrec=C.nrec, nerr=C.nerr), f_)
    _fin()
    atexit.register(_fin)
    print(f"[절 CM88] speed14 켬 진단 {C.dg} 고침 A {C.a} 고침 M {C.m} 셈 T {C.ct}", flush=True)
