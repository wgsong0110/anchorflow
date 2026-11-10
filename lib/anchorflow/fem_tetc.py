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


def _q_nodes(sv, Q, nb1, nb2):
    """적분점 q 의 27 자리 노드 번호 [Q,27] 과 기저 기울기 [Q,27,3] (BG 표, 사면체는 4 자리만 0 아님)."""
    dev = sv.Vq.device
    cb = sv.cb[:Q // 8].long()
    q = torch.arange(Q, device=dev); c, oc = q >> 3, q & 7
    o = torch.arange(27, device=dev); i, j, k = o // 9, (o // 3) % 3, o % 3
    nid = ((cb[c, 0:1] + i) * nb1 + (cb[c, 1:2] + j)) * nb2 + (cb[c, 2:3] + k)
    bg = sv.BG.float().view(8, 27, 3)[oc]
    return nid, bg


def _divdiag(sv, x0):
    """절 CM106.D (조정자 11-10 정정): 발산 시작 노드 진단 (결과 불변, 프레임 끝 버퍼 읽기만).
    고르는 노드: (a) |U| (프레임 노드 이동) 최대 5 개, (b) 걸음 끝 det X ≤ 0 사면체 (X = (I + Σ U ⊗ ∇w) F_q, V_q > 0) 중 det 작은 4 개의 꼭짓점,
    (c) 이탈 입자 (걸음 끝 위치가 0.1/99.9 % 분위 상자 밖 4 칸 넘게) 중 먼 8 개가 프레임 시작에 속한 사면체의 꼭짓점.
    노드마다: m_i, 질량 > 0 노드 중 백분위·중앙 대비 비, λ_max(K_ii) (K_ii = Σ_q V_q (μ|g|² I + (μ+λ) g gᵀ), g = F_qᵀ∇w_i -- 등방 선형 접선),
    이웃 사면체 입자 수, 바닥 장벽 (이 프레임 걸린 표본점 bcf&1 또는 걸음 시작 활성 bact) 이웃 사면체 수, |U|·|v|."""
    IP = sv.IP.detach().cpu().numpy().astype(np.int64)
    Q, nN, nb1, nb2 = int(IP[12]), int(IP[9]), int(IP[7]), int(IP[8])
    if Q > 4_000_000:
        return dict(skip=Q, nN=nN)
    dev = sv.Vq.device
    nid, bg = _q_nodes(sv, Q, nb1, nb2)
    Vq = sv.Vq[:Q].float(); Fq = sv.Fq[:Q].float().view(Q, 3, 3)
    U = sv.U[:nN].float(); vk = sv.vk[:nN].float(); mn = sv.mn[:nN].float()
    ok = (bg.abs().sum(-1) > 0) & (nid >= 0) & (nid < nN)
    nidc = nid.clamp(0, nN - 1)
    Gd = torch.einsum("qoa,qob->qab", U[nidc] * ok[..., None], bg)                     # Σ_o U_o ⊗ ∇w_o
    X = (torch.eye(3, device=dev) + Gd) @ Fq
    dX = torch.linalg.det(X.double()).float()
    act = Vq > 0
    inv = act & (dX <= 0)
    # 입자 -> 프레임 시작 적분점 (fx·pcd 는 프레임 시작 값)
    N = int(sv.N)
    fx = sv.fx[:N].float(); cidx = sv.cidx.long(); pcd = sv.pcd[:N].long()
    srt = torch.sort(fx, dim=1, descending=True, stable=True).indices
    tt = 2 * srt[:, 0] + (srt[:, 1] > srt[:, 2]).long()
    pq = cidx[pcd] * 8 + tt
    pq = torch.where((pq >= 0) & (pq < Q), pq, torch.full_like(pq, -1))
    npq = torch.bincount(pq[pq >= 0], minlength=Q)[:Q]
    x = sv.x[:N].float(); okx = torch.isfinite(x).all(1)
    far_i = torch.empty(0, dtype=torch.long, device=dev); nfar = 0
    if bool(okx.any()):
        xs = x[okx][::7]
        q0, q1 = torch.quantile(xs, 0.001, dim=0), torch.quantile(xs, 0.999, dim=0)
        cell = float(torch.linalg.norm(torch.linalg.inv(sv.Ai.double().reshape(3, 3).cpu())[:, 0]))
        d = torch.linalg.norm(torch.clamp(torch.maximum(q0 - x, x - q1), min=0), dim=1)
        d = torch.where(okx, d, torch.zeros_like(d))
        fm = d > 4 * cell; nfar = int(fm.sum())
        if nfar:
            far_i = torch.nonzero(fm).flatten()
            far_i = far_i[torch.argsort(d[far_i], descending=True)[:8]]
    bar = None
    if getattr(sv, "bcf", None) is not None:
        bar = (sv.bcf[:Q] & 1) > 0
    if getattr(sv, "bact", None) is not None:
        b2 = sv.bact[:Q] > 0
        bar = b2 if bar is None else (bar | b2)
    sel = {}
    un = torch.linalg.norm(U, dim=1); un = torch.where(torch.isfinite(un), un, torch.full_like(un, 3e38))
    for n_ in torch.topk(un, min(5, nN)).indices.tolist():
        sel.setdefault(n_, []).append("U")
    if bool(inv.any()):
        qi = torch.nonzero(inv).flatten(); qi = qi[torch.argsort(dX[qi])[:4]]
        for q_ in qi.tolist():
            for n_ in nid[q_][ok[q_]].tolist():
                sel.setdefault(n_, []).append("inv")
    for p_ in far_i.tolist():
        q_ = int(pq[p_])
        if q_ >= 0:
            for n_ in nid[q_][ok[q_]].tolist():
                sel.setdefault(n_, []).append("far")
    mpos = mn[mn > 0]; med = float(mpos.median()) if mpos.numel() else 0.0
    msort = torch.sort(mpos).values
    mu_, la_ = float(sv.mu), float(sv.lam)
    nodes = []
    for n_, why in sel.items():
        hit = (nid == n_) & ok                                                             # [Q,27]
        qs, os_ = torch.nonzero(hit, as_tuple=True)
        g = torch.einsum("qba,qb->qa", Fq[qs], bg[qs, os_])
        V = Vq[qs]
        K = (V[:, None, None] * (mu_ * (g * g).sum(-1)[:, None, None] * torch.eye(3, device=dev) + (mu_ + la_) * g[:, :, None] * g[:, None, :])).sum(0)
        lmax = float(torch.linalg.eigvalsh(K.double()).max()) if qs.numel() else 0.0
        m = float(mn[n_])
        rank = float(torch.searchsorted(msort, torch.tensor([m], device=dev)).item()) / max(1, msort.numel())
        aq = qs[V > 0]
        nodes.append(dict(n=int(n_), why=sorted(set(why)), m=m, m_pct=100 * rank, m_rel=(m / med if med > 0 else None), lmax=lmax,
                          ntet=int(qs.numel()), ntet_act=int(aq.numel()), npart=[int(v) for v in npq[qs].tolist()],
                          nbar=(int(bar[qs].sum()) if bar is not None else None), U=float(un[n_]), v=float(torch.linalg.norm(vk[n_]))))
    lm_all = None
    return dict(nN=nN, m_med=med, Umax=float(un.max()), ninv=int(inv.sum()), dmin=float(dX[act].min()) if bool(act.any()) else None,
                nfar=nfar, nodes=nodes)


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
            if d.dg >= 2:
                try:
                    rec["dv"] = _divdiag(s, None)
                except Exception as e_:
                    rec["dv_err"] = repr(e_)[:300]
            if d.dg and fr in DGF and bool(torch.isfinite(s.x[:1]).all()):
                try:
                    rec["rat"] = _ratio(s)
                except Exception as e_:
                    rec["rat_err"] = repr(e_)[:300]
            _write(rec)
            if fr % 10 == 0:
                _dump()
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
