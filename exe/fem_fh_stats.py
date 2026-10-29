"""사용자 10-29 "셀 집중 질량 판 (절 R 설정, 제한 끔, lego N192) 에서 폭주 노드 이웃 셀의 응력이 왜 프레임 시작부터 큰가":
--lagdump (폭주 노드 판정, fem_dv_stats.py 와 같은 규칙) + --fhist (프레임마다 시작 입자 F·셀 F̄·끝 노드 U) 로
폭주 노드 이웃 셀에 든 입자들의 입자 F 이력을 발산 프레임 −W ~ 0 에 걸쳐 추적하고, 정상 입자 (아무 셀) · 희소 정상 (입자 1~2 개 셀, 폭주와 무관) 과 비교.
덤프만 읽는다 (계산 경로 불변). GPU (잡 안).

입자 지표 (프레임 t 시작 상태; 정답 = 같은 입자·같은 프레임 d["F"][t], d["x"][t]):
  σ(F_p) 세 개·det F_p, 이웃 벌어짐 ‖F_p − mean_{kNN} F_q‖_F (kNN = 같은 프레임 위치로 가까운 K 입자, 자기 제외; 정답은 정답 위치로),
  ‖F_p − F̄_T(p)‖ (자기 셀 평균과 차), 자기 셀 키 (월드 고정 Kuhn 사면체: 네 꼭짓점 격자 정수 좌표 합 -- 사면체마다 유일), 셀 입자 수, 셀 ‖P(F̄)‖·σ(F̄),
  바닥 높이 (x − p)·n·바닥 후보 (nidx), 프레임 끝 자기 셀 사상 A_T(p) (‖A − I‖, 이웃 A 차 ‖A_p − mean_kNN A_q‖), 끝 변위 이웃 차 ‖u_p − mean_kNN u_q‖.
  검산: F_p(t+1) = A_T(p)(t) F_p(t) (ex_g2p_lag 그대로) 의 상대 차.
P(F̄) = 커널과 같은 FCR (fem_dv_stats.py 와 같은 식, A = I 인 프레임 시작 응력).
사용: python exe/fem_fh_stats.py --data <L000.pt> <실행 디렉토리>...  -> <실행>/fh_stats.npz, fh_stats.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import deform_model as DM                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--top", type=int, default=40)
ap.add_argument("--k_abn", type=float, default=5.0)
ap.add_argument("--W", type=int, default=20, help="발산 프레임 앞 몇 프레임")
ap.add_argument("--knn", type=int, default=16)
ap.add_argument("--nn", type=int, default=3000, help="정상 입자 표본")
ap.add_argument("--ns", type=int, default=3000, help="희소 정상 (입자 1~2 개 셀) 입자 표본")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def cross_rows(X):
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


def Pnorm(X, mu, lam):
    """커널과 같은 FCR 응력 노름 (J = Π max(σ, 0.01), sgn det)."""
    X = X.double()
    sX = torch.linalg.svdvals(X)
    J = sX.clamp_min(0.01).prod(-1)
    cf = cross_rows(X)
    sg = torch.sign(det3(X))
    P = 2 * mu * (X - polar(X)) + lam * ((J - 1) * sg)[:, None, None] * cf
    return P.flatten(1).norm(dim=-1).float()


def ikeys(npos, Ai):
    return torch.round(npos.double() @ Ai.double().T).long()


def comp(y):
    return ((y[..., 0] + 1000000) * 2000003 + (y[..., 1] + 1000000)) * 2000003 + (y[..., 2] + 1000000)


def knn(q, allx, k):
    """q [n,3] 의 allx 안 최근접 k (자기 자신 = 거리 0 첫째 제외)."""
    out = []
    for s in range(0, q.shape[0], 256):
        d = torch.cdist(q[s:s + 256].double(), allx.double(), compute_mode="donot_use_mm_for_euclid_dist")
        out.append(torch.topk(d, k + 1, largest=False).indices[:, 1:])
    return torch.cat(out, 0)


def qtl(x, qs=(0.01, 0.1, 0.5, 0.9, 0.99)):
    x = x[torch.isfinite(x)] if torch.is_tensor(x) else x
    if x.numel() == 0:
        return [float("nan")] * len(qs)
    return torch.quantile(x.double().cpu(), torch.tensor(qs, dtype=torch.float64)).tolist()


d = torch.load(a.data, map_location="cpu", weights_only=False)
h = float(d["cfg"]["frame_dt"])
F_in = d["F"]
gt_id = float((F_in[:3].float() - torch.eye(3)).abs().max()) < 1e-6
GTF = DM.traj_F(d, h, "cpu", keep_gpu=False)
GTX = d["x"]
print(f"[정답] F {tuple(GTF.shape)} (궤적에 든 F 가 항등이라 위치에서 되살림: {gt_id and not d.get('F_is_true')}), x {tuple(GTX.shape)}", flush=True)

for run in a.runs:
    R = torch.load(os.path.join(run, "lagdump.pt"), map_location="cpu", weights_only=False)
    # ---- 폭주 노드 (fem_dv_stats.py 와 같은 규칙) ----
    K = len(R)
    S = int(R[0]["N"]) + 1
    sp = [r["V"].float().norm(dim=-1) for r in R]                                  # [S,M]
    t_div = None
    q90 = {}
    for k in range(K):
        live = R[k]["mn"] > 0
        for s in range(S):
            x_ = sp[k][s][live]
            if t_div is None and bool(((~torch.isfinite(x_)) | (x_ > 1e4)).any()):
                t_div = k * S + s
            xf = x_[torch.isfinite(x_)]
            q90[k * S + s] = float(torch.quantile(xf, 0.9)) if xf.numel() else float("nan")
    if t_div is None:
        t_div = K * S
    g_last = t_div - 1
    kL, sL = g_last // S, g_last % S
    rL = R[kL]
    liveL = rL["mn"] > 0
    spL = torch.where(liveL & torch.isfinite(sp[kL][sL]), sp[kL][sL], torch.full_like(sp[kL][sL], -1.0))
    order = torch.argsort(-spL)[: a.top]
    run_nodes_L = order[spL[order] > a.k_abn * q90[g_last]]
    fdiv = int(rL["t"])
    Ai = rL["Ai"].float()
    run_keys = comp(ikeys(rL["npos"][run_nodes_L].float(), Ai))
    print(f"[{os.path.basename(run)}] 발산 프레임 {fdiv + 1} (서브스텝 {sL + 1}), 폭주 노드 {run_nodes_L.numel()}", flush=True)
    fh = {int(os.path.basename(p)[1:4]): p for p in glob.glob(os.path.join(run, "fhist", "f*.pt"))}
    frames = [t for t in range(max(0, fdiv - a.W), fdiv + 1) if t in fh]
    nF = len(frames)

    def load(t):
        z = torch.load(fh[t], map_location="cpu", weights_only=False)
        for k_ in ("x", "v", "F", "npos", "gl", "FT", "VT", "mn", "Uend", "Vend", "Ai"):
            z[k_] = z[k_].to(dev).float()
        for k_ in ("pinv", "nidx", "tets"):
            z[k_] = z[k_].to(dev).long()
        return z

    # ---- 발산 프레임 시작: 폭주 셀·입자, 정상·희소 정상 표본 ----
    Zdiv = load(fdiv)
    Z = Zdiv
    mu, lam = Z["mu"], Z["lam"]
    Np = Z["x"].shape[0]
    T = Z["tets"].shape[0]
    nk = comp(ikeys(Z["npos"], Z["Ai"]))
    srt, perm = torch.sort(nk)
    pos = torch.searchsorted(srt, run_keys.to(dev)).clamp_max(srt.numel() - 1)
    rn = perm[pos][srt[pos] == run_keys.to(dev)]
    isr = torch.zeros(Z["npos"].shape[0], dtype=torch.bool, device=dev)
    isr[rn] = True
    touch = isr[Z["tets"]].any(-1)
    ring = torch.zeros_like(isr)
    ring[Z["tets"][touch].reshape(-1)] = True
    ring1 = ring[Z["tets"]].any(-1)                                                    # 폭주 노드 1-이웃까지 닿는 셀
    occ = Z["VT"] > 0
    npt = torch.bincount(Z["pinv"], minlength=T)
    PT = torch.full((T,), float("nan"), device=dev)
    PT[occ] = Pnorm(Z["FT"][occ].view(-1, 3, 3), mu, lam)
    normT = occ & ~ring1
    thrP = float(torch.quantile(PT[normT], 0.99))
    runT = torch.nonzero(occ & touch).flatten()
    hotT = runT[PT[runT] > thrP]
    gen = torch.Generator(device="cpu").manual_seed(0)
    pin = Z["pinv"]
    run_p = torch.nonzero(torch.isin(pin, runT)).flatten()
    hot_p = torch.nonzero(torch.isin(pin, hotT)).flatten()
    cand = torch.nonzero(normT[pin]).flatten().cpu()
    norm_p = cand[torch.randperm(cand.numel(), generator=gen)[: a.nn]].to(dev)
    cand = torch.nonzero((normT & (npt <= 2))[pin]).flatten().cpu()
    sparse_p = cand[torch.randperm(cand.numel(), generator=gen)[: a.ns]].to(dev)
    GR = dict(hot=hot_p, run=run_p, norm=norm_p, sparse=sparse_p)
    print(f"  폭주 셀 {runT.numel()} (‖P‖ > 정상 셀 99 % {thrP:.3g} Pa 인 '뜨거운' 셀 {hotT.numel()}), 입자: 뜨거운 {hot_p.numel()}·폭주 셀 {run_p.numel()}·"
          f"정상 {norm_p.numel()}·희소 정상 {sparse_p.numel()}; 폭주 셀 입자 수 분포 {torch.bincount(npt[runT]).tolist()}", flush=True)
    allp = torch.unique(torch.cat(list(GR.values())))
    nA = allp.numel()
    loc = {g: torch.searchsorted(allp, p) for g, p in GR.items()}
    hot_keys = comp(ikeys(Zdiv["npos"], Zdiv["Ai"])[Zdiv["tets"]].sum(1))[hotT]
    # ---- 프레임별 추적 ----
    names = ["s0", "s1", "s2", "detF", "mis", "dev_cell", "npt", "PT", "sTmin", "sTmax", "height", "height_end", "cand", "Aerr", "Amis", "umis",
             "gs0", "gs2", "gdet", "gmis", "xerr", "key_chg", "same_tet_nn"]
    H = {n: torch.full((nA, nF), float("nan"), device=dev) for n in names}
    KEYS = torch.zeros((nA, nF), dtype=torch.long, device=dev)
    cell_hist = []                                                                     # 뜨거운 셀 (발산 프레임 키) 의 프레임별 입자 수·‖P‖·σ_min(F̄)
    g2p_chk = []
    prevF = prevA = None
    for fi, t in enumerate(frames):
        Z = Zdiv if t == fdiv else load(t)
        T = Z["tets"].shape[0]
        occ = Z["VT"] > 0
        pin = Z["pinv"]
        npt = torch.bincount(pin, minlength=T)
        PT = torch.full((T,), float("nan"), device=dev)
        PT[occ] = Pnorm(Z["FT"][occ].view(-1, 3, 3), mu, lam)
        FTm = Z["FT"].view(T, 3, 3)
        sT = torch.full((T, 3), float("nan"), device=dev)
        sT[occ] = torch.linalg.svdvals(FTm[occ].double()).float()
        tk = comp(ikeys(Z["npos"], Z["Ai"])[Z["tets"]].sum(1))                        # 사면체 키 (꼭짓점 정수 좌표 합)
        G = Z["gl"].view(T, 4, 3)
        U = Z["Uend"][Z["tets"]]                                                       # [T,4,3]
        A = torch.eye(3, device=dev) + torch.einsum("tia,tib->tab", U, G)
        # 입자 끝 변위 u = Σ λ U (λ_i = δ_i0 + ∇λ_i·(x − 꼭짓점 0))
        x = Z["x"]
        F = Z["F"].view(-1, 3, 3)
        lm = torch.einsum("pib,pb->pi", G[pin], x - Z["npos"][Z["tets"][pin, 0]])
        lm[:, 0] += 1.0
        u = (lm[..., None] * U[pin]).sum(1)
        Ap = A[pin]
        if prevF is not None:
            pred = prevA @ prevF
            g2p_chk.append([t, float(((F[allp] - pred).flatten(1).norm(dim=-1) / pred.flatten(1).norm(dim=-1)).max())])
        prevF, prevA = F[allp].clone(), Ap[allp].clone()
        q = x[allp]
        nb = knn(q, x, a.knn)                                                          # [nA,K]
        sv = torch.linalg.svdvals(F[allp].double()).float()
        H["s0"][:, fi], H["s1"][:, fi], H["s2"][:, fi] = sv[:, 0], sv[:, 1], sv[:, 2]
        H["detF"][:, fi] = det3(F[allp])
        H["mis"][:, fi] = (F[allp] - F[nb].mean(1)).flatten(1).norm(dim=-1)
        H["dev_cell"][:, fi] = (F[allp] - FTm[pin[allp]]).flatten(1).norm(dim=-1)
        H["npt"][:, fi] = npt[pin[allp]].float()
        H["PT"][:, fi] = PT[pin[allp]]
        H["sTmin"][:, fi] = sT[pin[allp], 2]
        H["sTmax"][:, fi] = sT[pin[allp], 0]
        KEYS[:, fi] = tk[pin[allp]]
        if fi > 0:
            H["key_chg"][:, fi] = (KEYS[:, fi] != KEYS[:, fi - 1]).float()
        H["same_tet_nn"][:, fi] = (pin[nb] == pin[allp][:, None]).float().mean(1)
        if Z["plane"] is not None:
            pt_, nr_ = Z["plane"]
            H["height"][:, fi] = ((q - torch.tensor(pt_, device=dev)) * torch.tensor(nr_, device=dev)).sum(-1)
            H["height_end"][:, fi] = ((q + u[allp] - torch.tensor(pt_, device=dev)) * torch.tensor(nr_, device=dev)).sum(-1)
        isc = torch.zeros(Np, device=dev)
        isc[Z["nidx"]] = 1.0
        H["cand"][:, fi] = isc[allp]
        H["Aerr"][:, fi] = (Ap[allp] - torch.eye(3, device=dev)).flatten(1).norm(dim=-1)
        H["Amis"][:, fi] = (Ap[allp] - Ap[nb].mean(1)).flatten(1).norm(dim=-1)
        H["umis"][:, fi] = (u[allp] - u[nb].mean(1)).norm(dim=-1)
        # 정답 (같은 입자·같은 프레임)
        gx = GTX[t].to(dev).float()
        gF = GTF[t].to(dev).float().reshape(-1, 3, 3)
        gnb = knn(gx[allp], gx, a.knn)
        gsv = torch.linalg.svdvals(gF[allp].double()).float()
        H["gs0"][:, fi], H["gs2"][:, fi] = gsv[:, 0], gsv[:, 2]
        H["gdet"][:, fi] = det3(gF[allp])
        H["gmis"][:, fi] = (gF[allp] - gF[gnb].mean(1)).flatten(1).norm(dim=-1)
        H["xerr"][:, fi] = (q - gx[allp]).norm(dim=-1)
        # 뜨거운 셀 (발산 프레임 키 기준) 의 이력
        srt, perm = torch.sort(tk)
        pos = torch.searchsorted(srt, hot_keys).clamp_max(srt.numel() - 1)
        hit = srt[pos] == hot_keys
        ti = perm[pos]
        cell_hist.append(dict(t=t, npt=torch.where(hit, npt[ti].float(), torch.zeros_like(hot_keys, dtype=torch.float)).cpu(),
                              P=torch.where(hit & occ[ti], PT[ti], torch.full_like(PT[ti], float("nan"))).cpu(),
                              smin=torch.where(hit & occ[ti], sT[ti, 2], torch.full_like(PT[ti], float("nan"))).cpu()))
        hm = lambda g, n: float(H[n][loc[g], fi].median()) if loc[g].numel() else float("nan")
        print(f"  프레임 {t + 1}: 뜨거운 입자 이웃 벌어짐 중앙 {hm('hot', 'mis'):.3g} / 정상 {hm('norm', 'mis'):.3g}"
              f" / 희소 {hm('sparse', 'mis'):.3g}, σ_min 중앙 {hm('hot', 's2'):.3f} (정답 {hm('hot', 'gs2'):.3f}) / 정상 {hm('norm', 's2'):.3f} (정답 {hm('norm', 'gs2'):.3f}),"
              f" 뜨거운 셀 ‖P‖ 중앙 {float(torch.nanmedian(cell_hist[-1]['P'])) if cell_hist[-1]['P'].numel() else float('nan'):.3g} 입자 수 중앙 {float(cell_hist[-1]['npt'].median()) if cell_hist[-1]['npt'].numel() else float('nan'):.1f}", flush=True)
        del Z
    # ---- 요약 ----
    summ = dict(run=run, fdiv=fdiv + 1, sub_div=sL + 1, frames=[t + 1 for t in frames], n_run_nodes=int(run_nodes_L.numel()),
                n_run_cells=int(runT.numel()), n_hot_cells=int(hotT.numel()), thrP=thrP, g2p_check_maxrel=g2p_chk,
                n=dict((g, int(p.numel())) for g, p in GR.items()), q={})
    for g in GR:
        summ["q"][g] = {n: [qtl(H[n][loc[g], fi]) for fi in range(nF)] for n in names}
    # 첫 벌어짐: 뜨거운 입자마다 mis > 정상 99 % 가 발산 프레임까지 끊기지 않고 이어진 구간의 시작 (발산 프레임 기준)
    ref99 = torch.tensor([summ["q"]["norm"]["mis"][fi][4] for fi in range(nF)], device=dev)
    refs99 = torch.tensor([summ["q"]["sparse"]["mis"][fi][4] for fi in range(nF)], device=dev)
    def onset(vals, ref):
        out = []
        for i in range(vals.shape[0]):
            f = nF - 1
            while f >= 0 and bool(vals[i, f] > ref[f]):
                f -= 1
            out.append(f + 1 - (nF - 1) if f + 1 <= nF - 1 else None)
        return out
    summ["onset_mis_vs_norm99"] = onset(H["mis"][loc["hot"]], ref99)
    summ["onset_mis_vs_sparse99"] = onset(H["mis"][loc["hot"]], refs99)
    ch = torch.stack([c["npt"] for c in cell_hist], 1)                                 # [hot, nF]
    run12 = []
    for i in range(ch.shape[0]):
        f = nF - 1
        while f >= 0 and 0 < float(ch[i, f]) <= 2:
            f -= 1
        run12.append(nF - 1 - f)
    summ["hot_cell_frames_npt_le2_consecutive"] = run12
    summ["hot_cell_npt_hist"] = ch.tolist()
    summ["hot_cell_P_hist"] = torch.stack([c["P"] for c in cell_hist], 1).tolist()
    summ["hot_cell_smin_hist"] = torch.stack([c["smin"] for c in cell_hist], 1).tolist()
    # 셀 갈아탐 때 자기 셀 ‖P‖ 변화 (뜨거운 입자 / 정상)
    for g in ("hot", "norm", "sparse"):
        kc = H["key_chg"][loc[g]]
        P_ = H["PT"][loc[g]]
        jr, sr = [], []
        for fi in range(1, nF):
            r_ = torch.log10(P_[:, fi] / P_[:, fi - 1])
            jr += r_[kc[:, fi] > 0].tolist()
            sr += r_[kc[:, fi] == 0].tolist()
        summ[f"cross_{g}"] = dict(rate=float(torch.nanmean(kc[:, 1:])) if nF > 1 else None, dlogP_cross=qtl(torch.tensor(jr)) if jr else None,
                                  dlogP_stay=qtl(torch.tensor(sr)) if sr else None)
    json.dump(summ, open(os.path.join(run, "fh_stats.json"), "w"), ensure_ascii=False, indent=1, default=float)
    np.savez(os.path.join(run, "fh_stats.npz"), frames=np.array(frames) + 1, fdiv=fdiv + 1,
             **{f"{g}_{n}": H[n][loc[g]].cpu().numpy() for g in GR for n in names},
             hot_cell_npt=ch.numpy(), hot_cell_P=np.array(summ["hot_cell_P_hist"]), hot_cell_smin=np.array(summ["hot_cell_smin_hist"]))
    print(f"[완료] {run}: G2P 검산 최대 상대차 {max([c[1] for c in g2p_chk], default=float('nan')):.2e}, 뜨거운 입자 벌어짐 시작 (vs 정상 99 %) "
          f"{summ['onset_mis_vs_norm99']}, 뜨거운 셀 입자 ≤2 연속 프레임 {run12}", flush=True)
    for g in ("hot", "norm", "sparse"):
        print(f"   셀 갈아탐 {g}: {summ[f'cross_{g}']}", flush=True)
