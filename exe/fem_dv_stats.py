"""사용자 10-29 "발산한 셀들과 정상 셀들의 질량이랑 힘을 비교해 달라" -- --lagdump (10-30 확장판) 의 lagdump.pt 로
폭주 노드 vs 정상 노드 (질량 m_n·내부력 |f_n|·가속도 |f_n|/m_n·속도), 폭주 노드 이웃 셀 vs 정상 셀 (m_T·점유율·‖P‖·det A_T·모양 품질·‖∇λ‖·셀이 노드에 주는 힘) 통계.
계산 경로는 건드리지 않는다 (덤프만 읽는다). GPU 에서 돈다 (잡 안).

정의 (모두 덤프된 값에서 커널과 같은 식으로):
  A_T = I + Σ_i U_i ⊗ ∇λ_i (∇λ = 프레임 시작 Kuhn 사면체 기준, gl), X = A_T F̄_T, P = 2μ (X − R(X)) + λ (J − 1) sgn(det X) cof X
  (R = 커널과 같은 스케일 Newton 극분해 10 번, J = Π max(σ_k, 0.01)), 셀이 꼭짓점 i 에 주는 힘 f_Ti = −V_T P F̄_Tᵀ ∇λ_i (ex_tet_force_body 그대로).
  확인: Σ_T f_Ti 를 노드로 모은 값 vs 덤프된 걸음 직전 내부력 f (상대 오차 기록).
  현재 모양: ∇λ_i^cur = A_T⁻ᵀ ∇λ_i, ‖∇λ‖ 비 = max_i ‖∇λ_i^cur‖ / max_i ‖∇λ_i‖ (찌그러질수록 커짐), 품질 = (최소 높이 / 최대 모서리) / 기준 값 (1 = Kuhn 기준, 0 = 납작),
  최소 높이 = 1 / max_i ‖∇λ_i^cur‖.
  m_T = 셀 안 입자 질량 합, 점유율 = V_T / 사면체 부피 (프레임 시작), 노드 질량 m_n = 덤프 그대로 (--mass tet: Σ m_T/4, --p2g bs: B-스플라인).
폭주 노드: 전역 상태 (프레임 k, 서브스텝 상태 s = 0..N) 를 이어 붙여, t_div = 처음 |v_n| 비유한 또는 > 1e4 m/s 인 상태,
  g_last = t_div − 1. 폭주 = g_last 에서 |v_n| > 5 × q90(|v|, 살아있는 노드) 인 노드 중 |v| 상위 TOP 개.
  각 폭주 노드의 시작 = g_last 까지 끊기지 않고 5 × q90 을 넘은 구간의 첫 상태. 무리 시작 g_on = 그 최솟값.
정상: 같은 프레임의 살아있는 노드 중 폭주 노드와 그 1-이웃 (같은 사면체) 을 뺀 무작위 표본 (NN 개), 정상 셀 = 폭주 노드에 안 닿는 점유 사면체 무작위 (NT 개).
시점: 시작 전 = g_on − PRE, 시작 = g_on, 직전 2 = g_last − 1, 직전 1 = g_last.
사용: python exe/fem_dv_stats.py <실행 디렉토리>...   -> <실행>/dv_stats.npz, dv_stats.json
"""
import argparse
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--top", type=int, default=40)
ap.add_argument("--nn", type=int, default=3000)
ap.add_argument("--nt", type=int, default=5000)
ap.add_argument("--pre", type=int, default=10)
ap.add_argument("--k_abn", type=float, default=5.0)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def cross_rows(X):
    """cof3 (커널과 같은 여인수 행렬: det = Σ X[0,k] C[0,k])."""
    r0, r1, r2 = X[..., 0, :], X[..., 1, :], X[..., 2, :]
    return torch.stack([torch.cross(r1, r2, dim=-1), torch.cross(r2, r0, dim=-1), torch.cross(r0, r1, dim=-1)], -2)


def det3(X):
    return (X[..., 0, :] * torch.cross(X[..., 1, :], X[..., 2, :], dim=-1)).sum(-1)


def polar(X):
    Q = X.clone()
    for _ in range(10):
        dt = det3(Q)
        ds = torch.where(dt.abs() > 1e-30, dt, torch.full_like(dt, 1e-30))
        gm = ds.abs().pow(-1.0 / 3.0)
        C = cross_rows(Q)
        Q = 0.5 * (gm[..., None, None] * Q + C / (ds * gm)[..., None, None])
    return Q


def symeig3(C):
    """커널 symeig3 와 같은 닫힌 식 (오름차순)."""
    a_, b_, c_, d_, e_, f_ = C[..., 0, 0], C[..., 1, 1], C[..., 2, 2], C[..., 0, 1], C[..., 1, 2], C[..., 0, 2]
    q = (a_ + b_ + c_) / 3
    p1 = d_ * d_ + e_ * e_ + f_ * f_
    p2 = (a_ - q) ** 2 + (b_ - q) ** 2 + (c_ - q) ** 2 + 2 * p1
    p = (p2 / 6).clamp_min(1e-30).sqrt()
    ip = 1 / p
    B00, B11, B22, B01, B12, B02 = (a_ - q) * ip, (b_ - q) * ip, (c_ - q) * ip, d_ * ip, e_ * ip, f_ * ip
    r = 0.5 * (B00 * (B11 * B22 - B12 * B12) - B01 * (B01 * B22 - B12 * B02) + B02 * (B01 * B12 - B11 * B02))
    phi = torch.acos(r.clamp(-1, 1)) / 3
    l3 = q + 2 * p * torch.cos(phi)
    l1 = q + 2 * p * torch.cos(phi + 2.0 * np.pi / 3.0)
    return torch.stack([l1, 3 * q - l1 - l3, l3], -1)


def qtl(x, qs):
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return [float("nan")] * len(qs)
    return torch.quantile(x.double().cpu(), torch.tensor(qs, dtype=torch.float64)).tolist()


def frame_arrays(r):
    """한 프레임: 상태별 노드·셀 지표 (GPU)."""
    N = int(r["N"])
    S = N + 1
    mu, lam = float(r["mu"]), float(r["lam"])
    npos = r["npos"].to(dev).float()
    mn = r["mn"].to(dev).float()
    M = mn.shape[0]
    tets = r["tets"].to(dev).long()
    T = tets.shape[0]
    G = r["gl"].to(dev).float().view(T, 4, 3)
    FT = r["FT"].to(dev).float().view(T, 3, 3)
    VT = r["VT"].to(dev).float()
    V0 = r["V0"].to(dev).float()
    occT = VT > 0
    U = r["U"].to(dev).float()
    Vn = r["V"].to(dev).float()
    fcap = r["f"].to(dev).float()
    idx = tets.reshape(-1)
    gref = G.norm(dim=-1).max(-1).values                                             # [T]
    P4 = npos[idx].view(T, 4, 3)
    ed = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    def maxedge(P):
        return torch.stack([(P[:, i] - P[:, j]).norm(dim=-1) for i, j in ed], -1).max(-1).values
    q0 = (1.0 / gref) / maxedge(P4)                                                  # 기준 품질
    TA = dict(Pn=torch.full((S, T), float("nan"), device=dev), detA=torch.full((S, T), float("nan"), device=dev),
              qual=torch.full((S, T), float("nan"), device=dev), grat=torch.full((S, T), float("nan"), device=dev),
              sminX=torch.full((S, T), float("nan"), device=dev), fvmax=torch.full((S, T), float("nan"), device=dev))
    NA = dict(nP=torch.full((S, M), float("nan"), device=dev), nG=torch.full((S, M), float("nan"), device=dev),
              nD=torch.full((S, M), float("nan"), device=dev), nQ=torch.full((S, M), float("nan"), device=dev),
              nFV=torch.full((S, M), float("nan"), device=dev), nSX=torch.full((S, M), float("nan"), device=dev))
    ferr = []
    occ4 = occT[:, None].expand(T, 4).reshape(-1)
    for s in range(S):
        Us = U[s]
        if not bool(torch.isfinite(Us).all()):
            continue
        A = torch.eye(3, device=dev) + torch.einsum("tia,tib->tab", Us[idx].view(T, 4, 3), G)
        X = A @ FT
        lX = symeig3((X.transpose(-1, -2) @ X).double()).float()
        sX = lX.clamp_min(1e-12).sqrt()
        J = sX.clamp_min(0.01).prod(-1)
        cf = cross_rows(X)
        dX = (X[:, 0, :] * cf[:, 0, :]).sum(-1)
        sg = torch.sign(dX)
        Q = polar(X)
        P = 2 * mu * (X - Q) + lam * ((J - 1) * sg)[:, None, None] * cf
        Gm = P @ FT.transpose(-1, -2)
        fv = -VT[:, None, None] * torch.einsum("tcb,tib->tic", Gm, G)                 # [T,4,3]
        fv = torch.where(occT[:, None, None], fv, torch.zeros_like(fv))
        fvn = fv.norm(dim=-1)
        cA = cross_rows(A)
        dA = (A[:, 0, :] * cA[:, 0, :]).sum(-1)
        gcur = torch.einsum("tab,tib->tia", cA / dA[:, None, None], G)
        gmax = gcur.norm(dim=-1).max(-1).values
        Pc = npos[idx].view(T, 4, 3) + Us[idx].view(T, 4, 3)
        qual = ((1.0 / gmax) / maxedge(Pc)) / q0
        qual = torch.where(dA > 0, qual, -qual)                                       # 뒤집힌 셀은 음수
        Pn = P.flatten(1).norm(dim=-1)
        nan = torch.full_like(Pn, float("nan"))
        TA["Pn"][s] = torch.where(occT, Pn, nan)
        TA["detA"][s] = torch.where(occT, dA, nan)
        TA["qual"][s] = torch.where(occT, qual, nan)
        TA["grat"][s] = torch.where(occT, gmax / gref, nan)
        TA["sminX"][s] = torch.where(occT, sX[:, 0], nan)
        TA["fvmax"][s] = torch.where(occT, fvn.max(-1).values, nan)
        # 노드별 이웃 (점유) 셀 최대·최소
        def agg(vals, red, init):
            src = vals[:, None].expand(T, 4).reshape(-1)
            src = torch.where(occ4, src, torch.full_like(src, init))
            o = torch.full((M,), init, device=dev)
            return o.scatter_reduce_(0, idx, src, reduce=red, include_self=True)
        NA["nP"][s] = agg(Pn, "amax", -float("inf"))
        NA["nG"][s] = agg(gmax / gref, "amax", -float("inf"))
        NA["nD"][s] = agg(dA, "amin", float("inf"))
        NA["nQ"][s] = agg(qual, "amin", float("inf"))
        NA["nSX"][s] = agg(sX[:, 0], "amin", float("inf"))
        src = torch.where(occ4, fvn.reshape(-1), torch.full_like(fvn.reshape(-1), -float("inf")))
        NA["nFV"][s] = torch.full((M,), -float("inf"), device=dev).scatter_reduce_(0, idx, src, reduce="amax", include_self=True)
        if s < N and s % 16 == 0:
            fre = torch.zeros(M, 3, device=dev).index_add_(0, idx, fv.reshape(-1, 3))
            live = (mn > 0) & torch.isfinite(fcap[s]).all(-1)
            den = fcap[s][live].norm()
            ferr.append([s, float((fre[live] - fcap[s][live]).norm() / den.clamp_min(1e-30)), float(den),
                         float((torch.where(occT, dA, nan) - r["detA"][s].to(dev))[occT].abs().max())])
    for d_ in NA.values():
        d_[~torch.isfinite(d_)] = float("nan")
    sp = Vn.norm(dim=-1)                                                             # [S,M]
    fn = torch.cat([fcap.norm(dim=-1), torch.full((1, M), float("nan"), device=dev)], 0)   # 상태 N 은 걸음 없음
    nbrT = torch.zeros(M, device=dev).index_add_(0, idx, occT[:, None].expand(T, 4).reshape(-1).float())
    mT = r["mT"].to(dev).float()
    nb_mT = torch.zeros(M, device=dev).index_add_(0, idx, torch.where(occT, mT, torch.zeros_like(mT))[:, None].expand(T, 4).reshape(-1))
    return dict(N=N, S=S, M=M, T=T, mn=mn, npos=npos, tets=tets, occT=occT, VT=VT, V0=V0, mT=mT, npt=r["npt"].to(dev).float(), occ=VT / V0,
                rn=r["rn"].to(dev).float(), occn=r["occn"].to(dev).float(), sp=sp, fn=fn, TA=TA, NA=NA, ferr=ferr, nbrT=nbrT, nb_mT=nb_mT,
                key=None)


def node_keys(r, F_):
    Ai = r["Ai"].to(dev).float()
    y = torch.round(F_["npos"] @ Ai.T).long()
    return ((y[:, 0] + 100000) * 200000 + (y[:, 1] + 100000)) * 200000 + (y[:, 2] + 100000)


for run in a.runs:
    R = torch.load(os.path.join(run, "lagdump.pt"), map_location="cpu", weights_only=False)
    if "f" not in R[0]:
        raise SystemExit(f"{run}: 10-30 확장 전 덤프 (f 없음)")
    K = len(R)
    FR = []
    for k, r in enumerate(R):
        F_ = frame_arrays(r)
        F_["key"] = node_keys(r, F_)
        F_["t"] = int(r["t"])
        FR.append(F_)
        print(f"[프레임 {r['t'] + 1}] M {F_['M']} T {F_['T']} 힘 재계산 상대오차 (서브스텝, 오차, |f|, det 차) {F_['ferr'][:4]}", flush=True)
    N = FR[0]["N"]
    S = N + 1
    Gtot = K * S
    gfr = lambda g: (g // S, g % S)
    # 전역 상태별 q90 (살아있는 노드 |v|)
    q90 = np.full(Gtot, np.nan)
    vmax = np.full(Gtot, np.nan)
    t_div = None
    for g in range(Gtot):
        k, s = gfr(g)
        F_ = FR[k]
        live = F_["mn"] > 0
        x = F_["sp"][s][live]
        bad = (~torch.isfinite(x)) | (x > 1e4)
        if bool(bad.any()) and t_div is None:
            t_div = g
        xf = x[torch.isfinite(x)]
        if xf.numel():
            q90[g] = float(torch.quantile(xf, 0.9))
            vmax[g] = float(xf.max())
    if t_div is None:
        t_div = Gtot
    g_last = t_div - 1
    kL, sL = gfr(g_last)
    FL = FR[kL]
    liveL = FL["mn"] > 0
    spL = torch.where(liveL & torch.isfinite(FL["sp"][sL]), FL["sp"][sL], torch.full_like(FL["sp"][sL], -1.0))
    thr = a.k_abn * q90[g_last]
    exc = spL > thr
    order = torch.argsort(-spL)
    run_idx = order[: a.top]
    run_idx = run_idx[exc[run_idx]]
    n_exc = int(exc.sum())
    run_keys = FL["key"][run_idx]
    nR = run_idx.numel()
    # 프레임마다 폭주 노드 번호 (격자 키로 대응), 없으면 -1
    def map_keys(F_, keys):
        srt, perm = torch.sort(F_["key"])
        pos = torch.searchsorted(srt, keys).clamp_max(srt.numel() - 1)
        hit = srt[pos] == keys
        return torch.where(hit, perm[pos], torch.full_like(pos, -1))
    RI = [map_keys(F_, run_keys) for F_ in FR]                                       # [K][nR]
    # 전역 시계열 (폭주 노드)
    def series(name, sub=None):
        out = np.full((nR, Gtot), np.nan)
        for k, F_ in enumerate(FR):
            ix = RI[k]
            ok = ix >= 0
            if not bool(ok.any()):
                continue
            if name in ("mn", "rn", "occn", "nbrT", "nb_mT"):
                val = F_[name][ix.clamp_min(0)][None, :].expand(S, -1)
            elif name in ("sp", "fn"):
                val = F_[name][:, ix.clamp_min(0)]
            else:
                val = F_["NA"][name][:, ix.clamp_min(0)]
            val = val.detach().cpu().numpy().copy()
            val[:, ~ok.cpu().numpy()] = np.nan
            out[:, k * S:(k + 1) * S] = val.T
        return out
    TS = {n: series(n) for n in ("mn", "rn", "occn", "nbrT", "nb_mT", "sp", "fn", "nP", "nG", "nD", "nQ", "nFV", "nSX")}
    TS["acc"] = TS["fn"] / TS["mn"]
    # 각 폭주 노드의 시작 (g_last 까지 끊기지 않고 k_abn q90 초과)
    onset = np.full(nR, -1)
    for i in range(nR):
        g = g_last
        while g >= 0 and np.isfinite(TS["sp"][i, g]) and TS["sp"][i, g] > a.k_abn * q90[g]:
            g -= 1
        onset[i] = g + 1
    g_on = int(onset.min()) if nR else g_last
    # 정상 표본 (프레임마다): 폭주 노드·1-이웃 제외
    gen = torch.Generator(device="cpu").manual_seed(0)
    NORM, NORMT, RUNT = [], [], []
    for k, F_ in enumerate(FR):
        ix = RI[k][RI[k] >= 0]
        touch = torch.zeros(F_["T"], dtype=torch.bool, device=dev)
        if ix.numel():
            isr = torch.zeros(F_["M"], dtype=torch.bool, device=dev)
            isr[ix] = True
            touch = isr[F_["tets"]].any(-1)
        nb = torch.zeros(F_["M"], dtype=torch.bool, device=dev)
        nb[F_["tets"][touch].reshape(-1)] = True
        cand = torch.nonzero((F_["mn"] > 0) & ~nb).flatten().cpu()
        NORM.append(cand[torch.randperm(cand.numel(), generator=gen)[: a.nn]].to(dev))
        candT = torch.nonzero(F_["occT"] & ~touch).flatten().cpu()
        NORMT.append(candT[torch.randperm(candT.numel(), generator=gen)[: a.nt]].to(dev))
        RUNT.append(torch.nonzero(F_["occT"] & touch).flatten())
    # 정상 노드 지표 분위 (상태별) -- 순서 판정·띠 그림용
    def norm_q(name, qs):
        out = np.full((len(qs), Gtot), np.nan)
        for k, F_ in enumerate(FR):
            ix = NORM[k]
            if name in ("mn", "rn", "occn", "nb_mT"):
                val = F_[name][ix][None, :].expand(S, -1)
            elif name in ("sp", "fn"):
                val = F_[name][:, ix]
            elif name == "acc":
                val = F_["fn"][:, ix] / F_["mn"][ix][None, :]
            else:
                val = F_["NA"][name][:, ix]
            v_ = val.double()
            for s in range(S):
                out[:, k * S + s] = qtl(v_[s], qs)
        return out
    QS = [0.01, 0.5, 0.9, 0.99]
    NQ = {n: norm_q(n, QS) for n in ("mn", "sp", "fn", "acc", "nP", "nG", "nD", "nQ", "nFV", "nSX")}
    # 순서: 각 지표가 정상 표본 q99 (작을수록 나쁜 것은 q01) 를 넘은 마지막 연속 구간의 첫 상태 (g_last 에서 거꾸로)
    worse_hi = dict(fn=True, acc=True, nP=True, nG=True, nFV=True, sp=True, mn=False, nD=False, nQ=False, nSX=False)
    first = {}
    for n, hi in worse_hi.items():
        src = TS[n]
        ref = NQ[n][3] if hi else NQ[n][0]
        fs = np.full(nR, np.nan)
        abn_last = np.zeros(nR, dtype=bool)
        for i in range(nR):
            g = g_last
            # 힘 계열은 상태 N (걸음 없음) 이 nan -- 건너뛴다
            while g >= 0:
                v_, rf = src[i, g], ref[g]
                if not np.isfinite(v_) or not np.isfinite(rf):
                    if n in ("fn", "acc") and (g % S) == N:
                        g -= 1
                        continue
                    break
                if (v_ > rf) if hi else (v_ < rf):
                    g -= 1
                else:
                    break
            g0 = g + 1
            if g0 <= g_last:
                fs[i] = g0 - t_div
            gl_ = g_last if not (n in ("fn", "acc") and g_last % S == N) else g_last - 1
            v_, rf = src[i, gl_], ref[gl_]
            abn_last[i] = bool(np.isfinite(v_) and ((v_ > rf) if hi else (v_ < rf)))
        first[n] = dict(first_rel_median=float(np.nanmedian(fs)) if np.isfinite(fs).any() else None,
                        first_rel_min=float(np.nanmin(fs)) if np.isfinite(fs).any() else None,
                        frac_abn_at_last=float(abn_last.mean()) if nR else None, first_rel=fs.tolist())
    # 시점별 집단 비교
    tps = dict(pre=max(g_on - a.pre, 0), onset=g_on, last2=max(g_last - 1, 0), last1=g_last)
    TPS = {}
    QQ = [0.0, 0.1, 0.5, 0.9, 1.0]
    samp = {}
    for nm_, g in tps.items():
        k, s = gfr(g)
        F_ = FR[k]
        sf = s if s < N else N - 1                                                   # 힘은 걸음 직전 (상태 N 이면 N−1)
        ix = RI[k][RI[k] >= 0]
        nx = NORM[k]
        def nodes(ii):
            mm = F_["mn"][ii]
            fn_ = F_["fn"][sf][ii]
            return dict(m=mm, f=fn_, acc=fn_ / mm, v=F_["sp"][s][ii], rn=F_["rn"][ii], occn=F_["occn"][ii])
        def cells(tt):
            return dict(mT=F_["mT"][tt], occ=F_["occ"][tt], npt=F_["npt"][tt], P=F_["TA"]["Pn"][s][tt], detA=F_["TA"]["detA"][s][tt],
                        qual=F_["TA"]["qual"][s][tt], grat=F_["TA"]["grat"][s][tt], sminX=F_["TA"]["sminX"][s][tt], fv=F_["TA"]["fvmax"][s][tt])
        dn = dict(run=nodes(ix), norm=nodes(nx))
        dc = dict(run=cells(RUNT[k]), norm=cells(NORMT[k]))
        TPS[nm_] = dict(g=g, frame=FR[k]["t"] + 1, sub=s, rel=g - t_div, n_run_nodes=int(ix.numel()), n_run_cells=int(RUNT[k].numel()),
                        nodes={grp: {q: qtl(v_, QQ) for q, v_ in d_.items()} for grp, d_ in dn.items()},
                        cells={grp: {q: qtl(v_, QQ) for q, v_ in d_.items()} for grp, d_ in dc.items()})
        for grp, d_ in dn.items():
            for q, v_ in d_.items():
                samp[f"{nm_}_n_{grp}_{q}"] = v_.detach().cpu().numpy()
        for grp, d_ in dc.items():
            for q, v_ in d_.items():
                samp[f"{nm_}_c_{grp}_{q}"] = v_.detach().cpu().numpy()
    med_all = float(torch.median(FR[kL]["mn"][FR[kL]["mn"] > 0]))
    info = dict(run=run, cmd=open(os.path.join(run, "cmd.txt")).read().strip() if os.path.exists(os.path.join(run, "cmd.txt")) else "",
                frames=[F_["t"] + 1 for F_ in FR], N=N, t_div=t_div, t_div_frame=FR[gfr(t_div)[0]]["t"] + 1 if t_div < Gtot else None,
                t_div_sub=gfr(t_div)[1] if t_div < Gtot else None, g_last=g_last, g_on=g_on, g_on_rel=g_on - t_div,
                onset_rel=(onset - t_div).tolist(), q90_last=q90[g_last], thr=thr, n_exceed_last=n_exc, n_run=nR, m_med_last=med_all,
                mass_def=R[0].get("mass_def"), g2p=R[0].get("g2p"), force_check=[F_["ferr"] for F_ in FR], timepoints=TPS, first_abnormal=first,
                run_m_rel=(FL["mn"][run_idx] / med_all).cpu().tolist(), run_occn=FL["occn"][run_idx].cpu().tolist(), run_rn=FL["rn"][run_idx].cpu().tolist())
    json.dump(info, open(os.path.join(run, "dv_stats.json"), "w"), ensure_ascii=False, indent=1, default=float)
    np.savez(os.path.join(run, "dv_stats.npz"), q90=q90, vmax=vmax, onset=onset, S=S, N=N, t_div=t_div, g_last=g_last, g_on=g_on,
             frames=np.array([F_["t"] + 1 for F_ in FR]), meta=np.array(json.dumps(dict(timepoints={k_: dict(g=v_["g"], frame=v_["frame"], sub=v_["sub"], rel=v_["rel"]) for k_, v_ in TPS.items()}))),
             **{f"ts_{n}": v_ for n, v_ in TS.items()}, **{f"nq_{n}": v_ for n, v_ in NQ.items()}, **samp)
    print(f"[완료] {run}: t_div 프레임 {info['t_div_frame']} 서브스텝 {info['t_div_sub']} (전역 {t_div}/{Gtot}), 폭주 노드 {nR} (초과 {n_exc}, 기준 {thr:.3g} m/s), "
          f"무리 시작 {g_on - t_div} 상태 전, m/중앙 {np.round(info['run_m_rel'][:10], 3).tolist()}", flush=True)
    for n, d_ in first.items():
        print(f"   [순서] {n}: 첫 비정상 (t_div 기준) 중앙 {d_['first_rel_median']} 최소 {d_['first_rel_min']}, 직전 비정상 비율 {d_['frac_abn_at_last']}", flush=True)
    del FR
    torch.cuda.empty_cache()
