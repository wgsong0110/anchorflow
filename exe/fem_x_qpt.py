"""절 X (사용자 10-29): lag 판 P2G 를 질량·운동량 Σλm (barycentric, 자기 사면체 네 노드), 셀 F̄_T (부피 가중)·V_T (부피 합), 적분점 = 셀 안 입자 질량 중심 (질량점) 으로.
이 스크립트는 덤프 (fem_ex_sim --lagdiag --lagdump K, 절 X 확장: pinv·mp 포함) 만 읽어 float64 로 검산한다 (계산 경로 불변, GPU 잡 안):

 (1) 적분점 위치가 힘을 바꾸지 않는가: 사면체마다 질량점 x_q = Σ m_p x_p / m_T 와 셀 (사면체) 중심 x_c 에서
     barycentric 함수 λ(x) 의 기울기 ∇λ 와 변위 기울기 ∇u (u(x) = Σ λ_i(x) U_i) 를 **그 점에서의 중앙 차분** 으로 구해
     F(x) = (I + ∇u(x)) F̄_T, P = FCR(F), f_i(x) = −V_T P F̄_Tᵀ ∇λ_i(x) 를 두 점에서 계산해 차이를 잰다 (상태: 프레임 시작 U = 0, 중간, 마지막 유한 상태).
 (2) 위치가 실제로 쓰이는 곳 -- 질량: Σ_p λ_i(x_p) m_p = Σ_T m_T λ_i^T(x_q,T) (λ 가 사면체 안에서 아핀이라 정확히 같다) -- 두 식과 덤프 m_n (커널 float32) 비교.
     x_q 가 사면체 안 (λ(x_q) ≥ 0) 인지, |x_q − x_c| / node_h 분포.
 (3) 노드 질량 분포: Σλm (이번 판) vs 절 R m_T/4 (같은 덤프 m_T 에서 Σ_T m_T/4), 중앙 대비 분위·2.5 % 미만 수, 국소 안정 비 r_n = h_s ω_n / 2
     (Gershgorin ω² ≤ Σ_T V_T (λ+2μ) |∇λ_n| Σ_j |∇λ_j| / m_n, fem_ex_sim 진단과 같은 식) 중앙·최대·필요 N (= N · 최대 r_n).
사용: python exe/fem_x_qpt.py <실행 디렉토리>...   -> <실행>/qpt.json
"""
import json
import os
import sys

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
dd = torch.float64
I3 = torch.eye(3, device=dev, dtype=dd)


def qtl(x, qs):
    x = x[torch.isfinite(x)]
    return torch.quantile(x.double().cpu(), torch.tensor(qs, dtype=dd)).tolist() if x.numel() else [float("nan")] * len(qs)


def polar_R(X):
    Us, _, Vh = torch.linalg.svd(X)
    dg = torch.ones(X.shape[0], 3, device=dev, dtype=dd)
    dg[:, 2] = torch.sign(torch.linalg.det(Us @ Vh))
    return Us @ torch.diag_embed(dg) @ Vh


def P_fcr(X, mu, lam):
    J = torch.linalg.det(X)
    cof = J[:, None, None] * torch.linalg.inv(X).transpose(-1, -2)
    return 2 * mu * (X - polar_R(X)) + lam * (J - 1)[:, None, None] * cof


def bary(P4, x):
    """P4 [T,4,3] 꼭짓점, x [T,3] -> λ [T,4] (아핀 barycentric, float64)."""
    D = (P4[:, 1:] - P4[:, :1]).transpose(-1, -2)                                  # 열 = P_k − P_0
    l = torch.linalg.solve(D, (x - P4[:, 0])[..., None])[..., 0]
    return torch.cat([1 - l.sum(-1, keepdim=True), l], -1)


def grad_at(P4, x, U4, eps):
    """점 x 에서 중앙 차분: ∇λ [T,4,3], ∇u [T,3,3] (∂u_a/∂x_b)."""
    gL = torch.zeros(x.shape[0], 4, 3, device=dev, dtype=dd)
    gU = torch.zeros(x.shape[0], 3, 3, device=dev, dtype=dd)
    for b in range(3):
        e = torch.zeros(3, device=dev, dtype=dd)
        e[b] = eps
        lp, lm = bary(P4, x + e), bary(P4, x - e)
        gL[:, :, b] = (lp - lm) / (2 * eps)
        gU[:, :, b] = (torch.einsum("ti,tia->ta", lp, U4) - torch.einsum("ti,tia->ta", lm, U4)) / (2 * eps)
    return gL, gU


def mass_stats(mn, med_ref=None):
    live = mn > 0
    med = float(mn[live].median())
    return dict(n_live=int(live.sum()), med=med, q_rel=qtl(mn[live] / med, [0.0, 1e-4, 1e-3, 0.01, 0.1, 0.5]),
                n_lt_2p5=int((mn[live] < 0.025 * med).sum()), n_lt_1e3=int((mn[live] < 1e-3 * med).sum()))


for run in sys.argv[1:]:
    R = torch.load(os.path.join(run, "lagdump.pt"), map_location="cpu", weights_only=False)
    r = R[0]
    if "pinv" not in r:
        raise SystemExit(f"{run}: 절 X 확장 전 덤프 (pinv 없음)")
    mu, lam, nh = float(r["mu"]), float(r["lam"]), float(r["node_h"])
    N, hs = int(r["N"]), float(r["hs"])
    npos = r["npos"].to(dev, dd)
    M = npos.shape[0]
    tets = r["tets"].to(dev).long()
    T = tets.shape[0]
    gl = r["gl"].to(dev, dd).view(T, 4, 3)
    FT = r["FT"].to(dev, dd).view(T, 3, 3)
    VT = r["VT"].to(dev, dd)
    occ = VT > 0
    xp = r["xp"].to(dev, dd)
    pinv = r["pinv"].to(dev).long()
    mp = r["mp"].to(dev, dd)
    rows = r["rows"].to(dev).long()
    lamb = r["lamb"].to(dev, dd)
    mT = torch.zeros(T, device=dev, dtype=dd).index_add_(0, pinv, mp)
    xq = torch.zeros(T, 3, device=dev, dtype=dd).index_add_(0, pinv, mp[:, None] * xp) / mT.clamp_min(1e-300)[:, None]
    P4 = npos[tets.reshape(-1)].view(T, 4, 3)
    xc = P4.mean(1)
    to = torch.nonzero(occ & (mT > 0)).flatten()
    P4o, xqo, xco = P4[to], xq[to], xc[to]
    lq = bary(P4o, xqo)
    dist = (xqo - xco).norm(dim=-1) / nh
    # 해석 ∇λ (꼭짓점에서 바로) vs 커널 gl (float32)
    gan, _ = grad_at(P4o, xco, torch.zeros_like(P4o), 1e-4 * nh)
    out = dict(run=run, frame=int(r["t"]) + 1, N=N, M=M, T_occ=int(to.numel()), mass_def=r.get("mass_def"),
               xq_inside_min_lambda=float(lq.min()), dist_xq_xc_over_h=qtl(dist, [0.0, 0.5, 0.9, 0.99, 1.0]),
               gl_kernel_vs_fd_rel=float((gan - gl[to]).norm() / gan.norm()))
    # (1) 상태별 힘 비교
    S_ = r["U"].shape[0]
    fin = [s for s in range(S_) if bool(torch.isfinite(r["U"][s]).all())]
    states = sorted(set([0, fin[len(fin) // 2], fin[-1]]))
    recs = []
    for s in states:
        U = r["U"][s].to(dev, dd)
        U4 = U[tets[to].reshape(-1)].view(-1, 4, 3)
        gq, uq = grad_at(P4o, xqo, U4, 1e-4 * nh)
        gc, uc = grad_at(P4o, xco, U4, 1e-4 * nh)
        Fq, Fc = (I3 + uq) @ FT[to], (I3 + uc) @ FT[to]
        good = torch.isfinite(Fq).all(-1).all(-1) & (torch.linalg.det(Fq).abs() > 1e-12) & (torch.linalg.det(Fc).abs() > 1e-12)
        Pq, Pc = P_fcr(Fq[good], mu, lam), P_fcr(Fc[good], mu, lam)
        V = VT[to][good][:, None, None]
        fq = -V * torch.einsum("tab,tib->tia", Pq @ FT[to][good].transpose(-1, -2), gq[good])
        fc = -V * torch.einsum("tab,tib->tia", Pc @ FT[to][good].transpose(-1, -2), gc[good])
        # 커널 식 (기준 ∇λ = gl, A = I + Σ U ⊗ gl) 과도 비교
        A = I3 + torch.einsum("tia,tib->tab", U4[good], gl[to][good])
        Pk = P_fcr(A @ FT[to][good], mu, lam)
        fk = -V * torch.einsum("tab,tib->tia", Pk @ FT[to][good].transpose(-1, -2), gl[to][good])
        fn = fc.norm(dim=-1).max(-1).values.clamp_min(1e-300)
        rel_t = (fq - fc).norm(dim=-1).max(-1).values / fn
        recs.append(dict(state=s, n_cells=int(good.sum()), f_scale=float(fc.norm()),
                         grad_lambda_rel=float((gq - gc).norm() / gc.norm()), grad_u_rel=float((uq - uc).norm() / max(float(uc.norm()), 1e-300)),
                         force_rel_global=float((fq - fc).norm() / fc.norm()), force_rel_cell_med=float(rel_t.median()), force_rel_cell_max=float(rel_t.max()),
                         kernel_formula_vs_center_rel=float((fk - fc).norm() / fc.norm())))
    out["force_check"] = recs
    # (2) 질량: Σ_p λ m_p vs Σ_T m_T λ(x_q) vs 덤프
    m_lam = torch.zeros(M, device=dev, dtype=dd).index_add_(0, rows.reshape(-1), (lamb * mp[:, None]).reshape(-1))
    m_q = torch.zeros(M, device=dev, dtype=dd).index_add_(0, tets[to].reshape(-1), (mT[to][:, None] * lq).reshape(-1))
    m_dump = r["mn"].to(dev, dd)
    live = m_lam > 0
    out["mass_identity"] = dict(sum_lam_vs_mT_lam_xq_max_rel=float(((m_lam - m_q).abs() / m_lam.clamp_min(1e-300))[live].max()),
                                sum_lam_vs_mT_lam_xq_med_rel=float(((m_lam - m_q).abs() / m_lam.clamp_min(1e-300))[live].median()),
                                dump_vs_sum_lam_max_rel=float(((m_dump - m_lam).abs() / m_lam.clamp_min(1e-300))[live].max()),
                                total_particles=float(mp.sum()), total_nodes=float(m_lam.sum()))
    # (3) 분포·국소 안정 비: Σλm vs m_T/4
    m_tet = torch.zeros(M, device=dev, dtype=dd).index_add_(0, tets.reshape(-1), (mT / 4)[:, None].expand(T, 4).reshape(-1))
    gn = gl.norm(dim=-1)
    w = VT[:, None] * (lam + 2 * mu) * gn * gn.sum(-1, keepdim=True)
    kd = torch.zeros(M, device=dev, dtype=dd).index_add_(0, tets.reshape(-1), w.reshape(-1))
    res = {}
    for nm, mm in (("sum_lambda_m", m_lam), ("tet_mT_over_4", m_tet)):
        lv = (mm > 0) & (kd > 0)
        rn = hs * (kd[lv] / mm[lv]).sqrt() / 2
        res[nm] = dict(mass_stats(mm), r_med=float(rn.median()), r_q999=qtl(rn, [0.999])[0], r_max=float(rn.max()), N_need=N * float(rn.max()),
                       n_r_gt1=int((rn > 1).sum()))
    both = (m_lam > 0) & (m_tet > 0)
    res["ratio_lam_over_tet"] = qtl(m_lam[both] / m_tet[both], [0.0, 0.01, 0.5, 0.99, 1.0])
    out["node_mass"] = res
    json.dump(out, open(os.path.join(run, "qpt.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[적분점] {run} 프레임 {out['frame']} 점유 사면체 {out['T_occ']}: x_q 안쪽 최소 λ {out['xq_inside_min_lambda']:.3e}, |x_q − x_c|/h 분위 (0,.5,.9,.99,1) "
          + " ".join(f"{q:.3f}" for q in out["dist_xq_xc_over_h"]) + f", 커널 gl vs 차분 ∇λ {out['gl_kernel_vs_fd_rel']:.2e}", flush=True)
    for q in recs:
        print(f"   상태 {q['state']}: ∇λ 차 {q['grad_lambda_rel']:.2e}, ∇u 차 {q['grad_u_rel']:.2e}, 힘 차 (전체 {q['force_rel_global']:.2e}, 셀 중앙 {q['force_rel_cell_med']:.2e} 최대 {q['force_rel_cell_max']:.2e}), "
              f"커널 식 vs 중심 {q['kernel_formula_vs_center_rel']:.2e}", flush=True)
    mi = out["mass_identity"]
    print(f"   질량: Σλm vs Σ m_T λ(x_q) 최대 {mi['sum_lam_vs_mT_lam_xq_max_rel']:.2e} (중앙 {mi['sum_lam_vs_mT_lam_xq_med_rel']:.2e}), 덤프 vs Σλm 최대 {mi['dump_vs_sum_lam_max_rel']:.2e}, "
          f"합 {mi['total_particles']:.6g} / {mi['total_nodes']:.6g}", flush=True)
    for nm in ("sum_lambda_m", "tet_mT_over_4"):
        q = res[nm]
        print(f"   {nm}: 노드 {q['n_live']}, 중앙 {q['med']:.3g}, 분위/중앙 " + " · ".join(f"{v:.2g}" for v in q["q_rel"])
              + f", 2.5 % 미만 {q['n_lt_2p5']}, 0.1 % 미만 {q['n_lt_1e3']}, r_n 중앙 {q['r_med']:.3g} / 99.9 % {q['r_q999']:.3g} / 최대 {q['r_max']:.3g} (필요 N {q['N_need']:.0f}, r>1 {q['n_r_gt1']})", flush=True)
    print(f"   Σλm / (m_T/4) 분위 (0,.01,.5,.99,1): " + " ".join(f"{v:.3g}" for v in res["ratio_lam_over_tet"]), flush=True)
