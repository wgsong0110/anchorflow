"""절 CM106 (사용자 11-10): 사면체 판 TC -- 질량·운동량·F 를 모두 사면체 (셀) 단위로 모으고 노드에는 lumped ¼ 로.

정의 (doc/fem_realtime.md 절 CM106.0; 바탕 T9 = 절 CM102, 벌점·장벽·정규화 없음):
- m_T = 사면체 T 안 입자 질량 합, p_T = Σ_{p∈T} m_p v_p, V_T = 입자 부피 합, F_T = 그 사면체 입자 F 평균 (rt_gauss 그대로).
- 노드 m_i = Σ_{T∋i} m_T / 4, p_i = Σ_{T∋i} p_T / 4 (HHT 입자 가속도 몫도 같은 경로).
- 구현은 실행기에 이미 있는 경로의 조합: --gq_mass (절 CM9: 입자 -> 자기 적분점 -> 노드, 노드 가중 = 기저 표 BW = 사면체 꼭짓점 ¼)
  + 점유 가중 κ 0 (--gq_oct 0: V_q = 사면체 입자 부피, m_q = 사면체 입자 질량, p_q = 사면체 입자 운동량; 빈 사면체 0)
  + 절 CM102 사면체 기저 (tet1_wrap --tet 1). 이 모듈은 그 조합을 확인 (모드 1) 하고 기록만 한다 (결과 불변).
- 프레임 끝 입자 갱신 (위치·속도 λ 보간, F·logJp 사면체 값) 은 그대로.

기록 (tetc.jsonl, 프레임마다; 결과 불변 -- 별도 버퍼에만 씀):
- 보존: P2G 직후 (rt_vn 바로 앞, 그래프 안) Σ_p m_p · Σ_p m_p v_p 와 Σ_n m_n · Σ_n (mv)_n (double 합).
- 질량 대 강성 (몇 프레임): 노드 대각 강성 k_n = (λ + 2μ) Σ_q V_q |F_qᵀ ∇w_n(x_q)|² (등방 선형 접선의 대각 덩어리 대각합 / 3 근사),
  국소 진동수 ω_n = sqrt(k_n / m_n), 프레임 시간 h 곱 ω_n h 의 분위; k_n > 0 인 노드 중 m_n ≤ 0 · < 1e-3 · < 1e-6 x 중앙 질량인 노드 수.
"""
import json
import os

import numpy as np
import torch

INFO = dict(mode=None, dg=None, n_inst=0, n_launch=0, err=None)
OUT = ["."]
DGF = {0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 99, 100, 124, 150, 200, 250, 299}

SRC = r'''
extern "C" {
__global__ void tc_zero(double* o) { if (threadIdx.x < 16) o[threadIdx.x] = 0.0; }
__device__ __forceinline__ double wsum(double x) { for (int d = 16; d > 0; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d); return x; }
__global__ void tc_sum(int N, int capN, const float* m, const float* v, const int* IP, const float* mn, const float* mv, double* o) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13]) return;
    double a[10];
    for (int k=0;k<10;k++) a[k] = 0.0;
    if (i < N) { double mp = m[i]; double vx = v[i*3], vy = v[i*3+1], vz = v[i*3+2];
        a[0] = mp; a[1] = mp*vx; a[2] = mp*vy; a[3] = mp*vz; a[4] = mp*sqrt(vx*vx + vy*vy + vz*vz); }
    if (i < IP[9] && i < capN) { double mm = mn[i]; a[5] = mm; a[6] = mv[i*3]; a[7] = mv[i*3+1]; a[8] = mv[i*3+2]; a[9] = (mm > 0.0) ? 1.0 : 0.0; }
    for (int k=0;k<10;k++) { double s = wsum(a[k]); if ((threadIdx.x & 31) == 0 && s != 0.0) atomicAdd(&o[k], s); }
}
}
'''


class Diag:
    def __init__(s, solver):
        import cupy as cp
        s.buf = torch.zeros(16, dtype=torch.float64, device="cuda")
        s.mod = cp.RawModule(code=SRC)
        s.fz = s.mod.get_function("tc_zero")
        s.fs = s.mod.get_function("tc_sum")
        s.ink = False
        s.ins = False

    def launch(s, sv):
        from . import fem_rt_cuda as RTC
        P = lambda t: (t.data_ptr() if t is not None else 0)
        N = int(sv.N); capN = int(sv.capN); n = max(N, capN)
        with RTC._Stream():
            s.fz((1,), (32,), (P(s.buf),))
            s.fs(((n + 255) // 256,), (256,), (np.int32(N), np.int32(capN), P(sv.m), P(sv.v), P(sv.IP), P(sv.mn), P(sv.mv), P(s.buf)))
        INFO["n_launch"] += 1


def _ratio(sv):
    """노드 질량 대 대각 강성 (프레임 끝 버퍼 = 이 프레임 P2G 값)."""
    IP = sv.IP.detach().cpu().numpy().astype(np.int64)
    Q, nN, nb1, nb2 = int(IP[12]), int(IP[9]), int(IP[7]), int(IP[8])
    dev = sv.Vq.device
    cb = sv.cb[:Q // 8].long()
    Vq = sv.Vq[:Q].double()
    Fq = sv.Fq[:Q].double().view(Q, 3, 3)
    BG = sv.BG.double().view(8, 27, 3)
    q = torch.arange(Q, device=dev)
    c, oc = q >> 3, q & 7
    Kd = torch.zeros(nN, dtype=torch.float64, device=dev)
    for o in range(27):
        i, j, k = o // 9, (o // 3) % 3, o % 3
        bg = BG[oc, o]                                                                     # [Q,3]
        if not bool((bg != 0).any()):
            continue
        g = torch.einsum("qba,qb->qa", Fq, bg)                                             # F_qᵀ ∇w (gqn 과 같은 식)
        w = Vq * (g * g).sum(-1)
        nid = ((cb[c, 0] + i) * nb1 + (cb[c, 1] + j)) * nb2 + (cb[c, 2] + k)
        ok = (w > 0) & (nid >= 0) & (nid < nN)
        Kd.index_add_(0, nid[ok], w[ok])
    mn = sv.mn[:nN].double()
    E = float(2.0 * float(sv.mu) + float(sv.lam))
    sel = Kd > 0
    ms, ks = mn[sel], Kd[sel]
    pos = mn > 0
    med = float(mn[pos].median()) if bool(pos.any()) else 0.0
    o = dict(nK=int(sel.sum()), nM=int(pos.sum()), nM_noK=int((pos & ~sel).sum()), m_med=med,
             n0=int((ms <= 0).sum()), n3=int((ms < 1e-3 * med).sum()), n6=int((ms < 1e-6 * med).sum()), E=E, h=float(sv.h))
    pk = ms > 0
    if bool(pk.any()):
        om = torch.sqrt(E * ks[pk] / ms[pk]) * float(sv.h)
        qs = torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float64, device=dev)
        v = torch.quantile(om[:min(om.numel(), 16_000_000)], qs).tolist() if om.numel() <= 16_000_000 else [float("nan")] * 4
        o.update(wh50=v[0], wh90=v[1], wh99=v[2], wh999=v[3], whmax=float(om.max()), nwh10=int((om > 10).sum()), nwh100=int((om > 100).sum()))
    return o


def _write(rec):
    try:
        with open(os.path.join(OUT[0], "tetc.jsonl"), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e_:
        INFO["err"] = ("write " + str(e_))[:300]


def _dump():
    try:
        json.dump(INFO, open(os.path.join(OUT[0], "tetcinfo.json"), "w"), ensure_ascii=False, indent=1, default=str)
    except Exception:
        pass


def _wrap_k(k0):
    def k(s, name, n, args, b=128):
        d = getattr(s, "_tcx", None)
        if d is not None and name == "rt_vn" and not d.ink and getattr(s, "_pkc", None) is None:
            d.launch(s)
            d.ink = True
            try:
                return k0(s, name, n, args, b)
            finally:
                d.ink = False
        return k0(s, name, n, args, b)
    k._tetc = True
    return k


def _wrap_step(st0):
    def step(s, t, *a, **kw):
        d = getattr(s, "_tcx", None)
        if d is None or d.ins:
            return st0(s, t, *a, **kw)
        d.ins = True                                                                         # 안쪽 (상위 클래스) step 은 그대로 통과
        try:
            r = st0(s, t, *a, **kw)
        finally:
            d.ins = False
        try:
            torch.cuda.synchronize()
            b = d.buf.detach().cpu().numpy()
            fr = int(round(float(t) / float(s.h)))
            mp, Pp, Pa, mN, PN = float(b[0]), b[1:4], float(b[4]), float(b[5]), b[6:9]
            rec = dict(t=float(t), f=fr, m_p=mp, m_n=mN, P_p=[float(x) for x in Pp], P_n=[float(x) for x in PN], Pabs=Pa, nm=int(b[9]),
                       em=abs(mN - mp) / mp if mp > 0 else None, eP=float(np.abs(PN - Pp).max() / Pa) if Pa > 0 else None)
            if d.dg and fr in DGF and bool(torch.isfinite(s.x[:1]).all()):
                try:
                    rec["rat"] = _ratio(s)
                except Exception as e_:
                    rec["rat_err"] = repr(e_)[:300]
            _write(rec)
        except Exception as e_:
            INFO["err"] = ("step " + repr(e_))[:400]; _dump()
        return r
    step._tetc = True
    return step


def install(out, mode, dg):
    OUT[0] = out
    INFO["mode"] = int(mode); INFO["dg"] = int(dg)
    from . import fem_an_rt as ART
    mods = [ART]
    for nm in ("fem_an_rt_v0", "fem_ao_rt", "fem_mat_rt", "fem_det_rt"):
        try:
            mods.append(__import__("anchorflow." + nm, fromlist=[nm]))
        except Exception as e_:
            INFO.setdefault("imp_err", []).append(f"{nm}: {e_}"[:200])
    for M in mods:
        for cn in ("RTSolver", "AORTSolver", "RTAdapt", "MatRT", "DetMixin"):
            C = getattr(M, cn, None)
            if C is None:
                continue
            if "k" in C.__dict__ and not getattr(C.__dict__["k"], "_tetc", False):
                C.k = _wrap_k(C.__dict__["k"])
            if "step" in C.__dict__ and not getattr(C.__dict__["step"], "_tetc", False):
                C.step = _wrap_step(C.__dict__["step"])
            if "__init__" in C.__dict__ and not getattr(C.__init__, "_tetc", False):
                f0 = C.__init__

                def init(s, *a, _f0=f0, **kw):
                    _f0(s, *a, **kw)
                    if getattr(s, "_tcx", None) is None:
                        s._tcx = Diag(s)
                        s._tcx.dg = int(INFO["dg"])
                        INFO["n_inst"] += 1
                        INFO["cls"] = type(s).__name__
                        INFO["gqm"] = bool(getattr(s, "gqm", False)); INFO["goct"] = getattr(s, "goct", None)
                        _dump()
                init._tetc = True
                C.__init__ = init
    print(f"[절 CM106] tetc 켬 -- 모드 {mode} ({'TC: 사면체 단위 질량·운동량, 노드 ¼' if mode == 1 else '기록만'}), 기록 {dg}", flush=True)
    _dump()
