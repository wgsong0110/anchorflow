"""절 CM28 (1) 첫 접촉 비교 통계 (결과 불변, 읽기만): 우리 판 (--erec 의 en.fs) 과 정답 (exe/wmfrac_gt.py, GF h5 f_tensor = 탄성 F_e) 이 같은 함수.
J = det F (입자, 부피 가중 아님 -- 입자 수 몫), 탄성 에너지 SNH (풀이 판 식) · NHB (GF 식 Ψ = μ/2 (J^{-2/3} I_C − 3) + κ/4 (J² − 1 − 2 log J), J ≤ 0 은 뺌),
바닥 띠 = 바닥 위 높이 < band (0.03, 우리 칸 1 개 ≈ 정답 격자 4.5 칸) 입자의 SNH 에너지 몫, logJp 분포 (손상 > 0, 다져짐 < −0.2).
"""
import torch


def frame_stats(F, vol, x, lj, mu, lam, nrm=(0.0, 0.0, 1.0), pn=0.02, band=0.03):
    F = F.reshape(-1, 3, 3).double()
    vol = vol.double()
    J = torch.linalg.det(F)
    Ic = (F * F).sum((1, 2))
    ls, kap = lam + mu, 2.0 * mu / 3.0 + lam
    psn = 0.5 * mu * (Ic - 3.0) - mu * (J - 1.0) + 0.5 * ls * (J - 1.0) ** 2
    Jp = J.clamp_min(1e-30)
    pnh = 0.5 * mu * (Jp ** (-2.0 / 3.0) * Ic - 3.0) + 0.25 * kap * (Jp ** 2 - 1.0 - 2.0 * torch.log(Jp))
    ok = torch.isfinite(psn) & torch.isfinite(x.double()).all(-1)
    okn = ok & (J > 0) & torch.isfinite(pnh)
    h = x.double() @ torch.tensor(nrm, dtype=torch.float64, device=x.device) - pn
    bot = ok & (h < band)
    Jo = J[ok].float()
    qs = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], device=Jo.device)
    sub = Jo if Jo.numel() <= 8_000_000 else Jo[torch.randperm(Jo.numel(), device=Jo.device)[:8_000_000]]
    r = dict(Jq=[float(q) for q in torch.quantile(sub, qs)],
             J095=float((Jo < 0.95).double().mean()), J090=float((Jo < 0.90).double().mean()), J111=float((Jo > 1.11).double().mean()),
             Esnh=float((vol[ok] * psn[ok]).sum()), Enhb=float((vol[okn] * pnh[okn]).sum()), Ebot=float((vol[bot] * psn[bot]).sum()), nbot=int(bot.sum()),
             Jinv=int((J[ok] <= 0).sum()))
    if lj is not None:
        l = lj.double()[ok]
        l = l[torch.isfinite(l)]
        ls_ = l.float() if l.numel() <= 8_000_000 else l.float()[:8_000_000]
        r.update(ljq=[float(q) for q in torch.quantile(ls_, torch.tensor([0.01, 0.5, 0.99], device=ls_.device))],
                 lj_pos=float((l > 0).double().mean()), lj_c02=float((l < -0.2).double().mean()), lj_mean=float(l.mean()))
    return r
