"""절 CM23.4 진단 (--frec, 결과 불변 -- 그래프 안 사본 버퍼를 프레임 뒤에 읽기만): 수박 K 1 첫 접촉 뒤 위쪽 운동량의 출처.

노드 힘 몫 (법선 n = 바닥 법선 = 위쪽, 단위 N; 풀이 기울기 g = Mi η + f_el + G n, Mi = m/τ², η = y − ŷ, ŷ 는 τ² g_grav 를 품음):
  탄성 = −f_el·n (aq_elem 기울기, 장벽 더하기 전), 장벽 = −G (rt_bar_q 노드 스칼라, f += G n), 관성 = −Mi η·n (중력 포함 관성 예측과의 차),
  중력 = m g·n, 잔차 = g·n (= 풀이가 못 맞춘 힘; 수렴하면 0). 거의 빈 노드 (m < 문턱, y = ŷ 고정) 는 힘 몫에서 뺌.
걸음 속도 변화 분해 (HHT, αm 0: v1 = α (y − z0) -> 노드마다 정확히
  v1 − v0 = dt (1 − γ) a0 [가속도 기억] + γ dt g_grav [중력] + γ dt (−f_el − G n + g)/m [탄성 + 장벽 + 잔차]; 고정 노드는 앞 둘만),
  질량 가중 합 / 질량 합 (m/s) -- 측정 Σ m Δv / Σ m 과 대조.
다음 걸음 가속도 기억 a1 = (v1 − v0 − dt (1−γ) a0)/(γ dt) = g_grav + (−f_el − G n + g)/m (법선 성분; 고착은 접선만 0).
장벽 표본점: x_q = 질량 중심 + Σ_o w (U + thinv (y − U)), U = 0 (K 1 첫 걸음), d = x_q·n − p; 걸림 = 활성 & d < d̂, 에너지 Σ −κ (d − d̂)² ln(d/d̂), 최소 d/d̂ (활성).
"""
import math

import torch


def _pct(t, qs=(0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)):
    t = t[torch.isfinite(t)].double()
    if t.numel() == 0:
        return None
    if t.numel() > 8_000_000:
        t = t[torch.randperm(t.numel(), device=t.device)[:8_000_000]]
    return [float(q) for q in torch.quantile(t, torch.tensor(qs, device=t.device, dtype=torch.float64))]


def _snh(F, vol, mu, lam):
    """SNH 탄성 에너지 Σ V Ψ, Ψ = μ/2 (I_C − 3) − μ (J − 1) + λ_s/2 (J − 1)², λ_s = λ + μ (풀이 판 재질식 그대로)."""
    F = F.reshape(-1, 3, 3).double()
    J = torch.linalg.det(F)
    Ic = (F * F).sum((1, 2))
    ls = lam + mu
    psi = 0.5 * mu * (Ic - 3.0) - mu * (J - 1.0) + 0.5 * ls * (J - 1.0) ** 2
    ok = torch.isfinite(psi)
    return float((vol.double()[ok] * psi[ok]).sum()), J


class FrecCtx:
    def __init__(s, S, a, FL, WB, m64, vol64, mu, lam, g, cell, mthr, Fgt, xgt):
        s.S, s.a, s.FL, s.WB = S, a, FL, WB
        s.m64, s.vol64, s.mu, s.lam = m64, vol64, mu, lam
        s.cell, s.mthr = cell, mthr
        dv = S.dev
        s.n = torch.tensor(FL["NR"] if FL else [0.0, 0.0, 1.0], device=dv, dtype=torch.float64)
        s.pn = float(FL["PN"]) if FL else 0.0
        s.gn = float(torch.tensor(g, dtype=torch.float64) @ s.n.cpu())
        s.Fgt, s.xgt = Fgt, xgt
        c = S.ga_c
        s.gm, s.af = float(c["gm"]), float(c["af"])
        s.dt = float(S.dt)
        s.it2 = 1.0 / (float(S.tau) ** 2)
        s.thinv = 1.0 / (1.0 - s.af)
        s.prev_a1 = None

    def _nodes(s):
        S = s.S
        IP = S.IP.cpu().tolist()
        nn = IP[9]; nb1, nb2 = IP[7], IP[8]
        idx = torch.arange(nn, device=S.dev)
        L = torch.stack([idx // (nb1 * nb2) + IP[0], (idx // nb2) % nb1 + IP[1], idx % nb2 + IP[2]], 1).double()
        X = L @ S.A.double().T
        return nn, X, IP

    def _bar(s, y, IP):
        S, WB = s.S, s.WB
        nq = IP[12]
        if WB is None or nq == 0:
            return None
        bx = S.bx[:4 * nq].view(nq, 4).double(); bm = bx[:, 3]
        act = (S.bact[:nq] > 0) & (bm > 0)
        q = torch.arange(nq, device=S.dev)
        cb = S.cb[q >> 3].long()
        o = torch.arange(27, device=S.dev)
        oi, oj, ok = o // 9, (o // 3) % 3, o % 3
        ids = ((cb[:, 0:1] + oi) * IP[7] + (cb[:, 1:2] + oj)) * IP[8] + (cb[:, 2:3] + ok)
        w = S.BW.view(8, 27)[(q & 7)].double()
        xq = torch.where(bm[:, None] > 0, bx[:, :3] / bm.clamp_min(1e-30)[:, None], torch.zeros_like(bx[:, :3]))
        xq = xq + (w[:, :, None] * (s.thinv * y.double()[ids])).sum(1)
        d = xq @ s.n - s.pn
        dh = float(WB["DH"])
        hit = act & (d < dh)
        kap = float(WB["KS"]) * S.Vq[:nq].double()
        dd = d - dh
        e = torch.where(hit & (d > 0), -kap * dd * dd * torch.log(d.clamp_min(1e-30) / dh), torch.zeros_like(d))
        neg = int((hit & (d <= 0)).sum())
        return dict(nsamp=int((bm > 0).sum()), nact=int(act.sum()), hit=int(hit.sum()), E=float(e.sum()), neg=neg,
                    dmin_dh=(float(d[act].min()) / dh if bool(act.any()) else None), dmin_all_dh=(float(d[bm > 0].min()) / dh if bool((bm > 0).any()) else None))

    def frame(s, fr):
        S = s.S
        nn, X, IP = s._nodes()
        n = s.n.float()
        mn = S.FRmn[:nn].double()
        free = mn >= max(s.mthr, 1e-30)
        hgt = X @ s.n - s.pn
        regs = dict(all=mn > 0, near=(mn > 0) & (hgt < s.a.frec_band * s.cell))
        out = dict(band_m=s.a.frec_band * s.cell, mass={k: float(mn[r].sum()) for k, r in regs.items()}, nnode={k: int(r.sum()) for k, r in regs.items()})
        ct3 = S.NRct3.cpu().numpy()
        its = []
        for it in range(S.nt + 1):
            fel = (S.FRfel[0, it, :nn].double() @ s.n.double())
            bg = S.FRbg[0, it, :nn].double()
            eta = S.FReta[0, it, :nn].double() @ s.n
            g = -(S.FRr[0, it, :nn].double() @ s.n)
            mi = mn * s.it2
            r_ = dict(it=it, end=(it == S.nt), ran=(True if it == S.nt else bool(int(ct3[it][0]) == 0)))
            for k, R in regs.items():
                Rf = R & free
                r_[k] = dict(el=float(-fel[Rf].sum()), bar=float(-bg[Rf].sum()), inert=float(-(mi * eta)[Rf].sum()), grav=float((mn[R] * s.gn).sum()), res=float(g[Rf].sum()),
                             el_abs=float(fel[Rf].abs().sum()))
            r_["barq"] = s._bar(S.FRy[0, it, :nn], IP)
            its.append(r_)
        out["it"] = its
        # 걸음 속도 변화 분해 (마지막 뉴턴 뒤 다시 잰 힘으로)
        E = S.nt
        fel = S.FRfel[0, E, :nn].double() @ s.n.double(); bg = S.FRbg[0, E, :nn].double(); g = -(S.FRr[0, E, :nn].double() @ s.n)
        v0 = S.FRv0[0, :nn].double() @ s.n; v1 = S.FRv1[0, :nn].double() @ s.n; a0 = S.FRa0[0, :nn].double() @ s.n; a1 = S.FRa1[0, :nn].double() @ s.n
        gd = s.gm * s.dt
        dv = {}
        for k, R in regs.items():
            M = float(mn[R].sum())
            if M <= 0:
                continue
            Rf = R & free
            d_ = dict(v0=float((mn * v0)[R].sum()) / M, v1=float((mn * v1)[R].sum()) / M, meas=float((mn * (v1 - v0))[R].sum()) / M,
                      mem=s.dt * (1.0 - s.gm) * float((mn * a0)[R].sum()) / M, grav=gd * s.gn,
                      el=gd * float(-fel[Rf].sum()) / M, bar=gd * float(-bg[Rf].sum()) / M, res=gd * float(g[Rf].sum()) / M,
                      a0=float((mn * a0)[R].sum()) / M, a1=float((mn * a1)[R].sum()) / M)
            d_["sum"] = d_["mem"] + d_["grav"] + d_["el"] + d_["bar"] + d_["res"]
            d_["chk"] = d_["meas"] - d_["sum"]
            # a1 의 몫 (다음 프레임 가속도 기억으로 넘어감): g_grav + (탄성 + 장벽 + 잔차)/m
            d_["a1_el"] = d_["el"] / gd; d_["a1_bar"] = d_["bar"] / gd; d_["a1_res"] = d_["res"] / gd
            dv[k] = d_
        out["dv"] = dv
        out["gm"], out["dt"], out["af"] = s.gm, s.dt, s.af
        # 프레임 끝 입자: det F 분포·SNH 에너지·질량 가중 법선 속도 (우리 / 정답)
        Ep, Jp = _snh(S.F, S.vol, s.mu, s.lam)
        mp = S.m.double()
        vz = float((mp * (S.v.double() @ s.n)).sum() / mp.sum())
        pe = dict(E=Ep, J=_pct(Jp), J_lt09=float((Jp < 0.9).double().mean()), J_lt095=float((Jp < 0.95).double().mean()), J_gt105=float((Jp > 1.05).double().mean()), vz=vz,
                  KE=float(0.5 * (mp * (S.v.double() ** 2).sum(1)).sum()))
        out["p"] = pe
        if s.Fgt is not None and fr < s.Fgt.shape[0]:
            Fg = s.Fgt[fr].to(S.dev)
            Eg, Jg = _snh(Fg, s.vol64.to(S.dev), s.mu, s.lam)
            out["gt"] = dict(E=Eg, J=_pct(Jg), J_lt09=float((Jg < 0.9).double().mean()), J_lt095=float((Jg < 0.95).double().mean()), J_gt105=float((Jg > 1.05).double().mean()))
        return out
