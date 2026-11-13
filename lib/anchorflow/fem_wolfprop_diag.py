"""절 CM26 진단 --prec (새 실행기 exe/fem_an_rt_sim.py, 재질 판 MatRT; --nrec 위, K 1, HHT; 결과 불변 -- 그래프 안 사본 복사만, 읽기는 프레임 뒤 호스트에서).

질문: wolf 기본 판 (장벽 바닥) 이 장벽이 거의 안 걸리는 f15~22 에 무엇으로 몸을 받치는가 -- 같은 구간 (A) 가우스 점 고착 바닥은 무너짐.

노드 힘 몫 (y 공간, 단위 N, 법선 n = 바닥 법선 = 위쪽):
  풀이 변수 바꿈 y = h + T η (T = 구속·고착 사상: pm 0 세 성분 √k, pm 1 접선 √k·법선 1, pm 2 고정), 풀이 기울기 g = Mi η + T f (rt_grad), Mi = m/τ².
  날 잔차 R = Mi (y − Ũ) + f_el + G n (Ũ = 중력 포함 관성 예측) 를 R = T g + C 로 나눔: T g = 풀이가 못 맞춘 몫 (잔차, 수렴이면 0),
  C = 구속·고착이 흡수한 몫 (반력; 몸에 주는 힘 +C). 노드마다 정확히 Mi (y − Ũ) = −f_el − G n + C + T g.
  C 를 노드 종류로: 순차 방출 구속 노드 (k_구속 < 1, ao_pbc_k), 바닥 고착 노드 (k_구속 = 1 이고 √k < 1: (B) 띠 고착 접선 / (A) 가우스 점 고착 세 성분),
  고정 노드 (pm 2: 질량 < 문턱 관성 예측 고정).
  몸 합: 탄성 −Σ f_el·n (내력이라 몸 전체 합은 ≈ 0), 장벽 −Σ G, 구속 반력 Σ C·n, 고착 반력, 고정, 잔차 Σ T g·n, 관성 Σ Mi (y − Ũ)·n = Σ m (a1 − g), 무게 Σ m g·n.
  정지한 몸이면 관성 합 ≈ 0 -> 장벽 + 구속 + 고착 + 잔차 (+ 탄성 ≈ 0) = −무게: 무게를 누가 드는지.
걸음 속도 변화 (HHT, 노드마다 정확히): v1 − v0 = dt (1 − γ) a0 [가속도 기억] + γ dt g [중력] + γ dt (−f_el − G n + C + T g)/m, 질량 가중 평균 (m/s); 측정 Σ m Δv / Σ m 과 대조.
  걸음 끝 가속도 기억 지움 (구속 노드 a = 0, 고착 노드 a 접선/세 성분 0) 몫도: Σ m (a1_식 − a1_저장)·n.
영역: 몸 전체 (m > 0), 바닥 근처 (걸음 시작 노드 높이 < 1 칸 · 3 칸), 구속 노드 (k_구속 < ½) / 나머지.
입자 (프레임 끝): 방출 시각 trel 대비 묶임 (trel > 프레임 끝) · 이번 프레임 풀림 · 풀림 무리별 수·높이 (우리 / 정답)·처음 위치에서 내려간 양·이동량,
  풀린 입자 중 보간 가중의 반 넘게가 구속 노드인 수 (노드 구속에 사실상 묶인 풀린 입자), 묶인 입자 x 범위 (방출 앞면).
장벽 표본점 (B): 걸음 끝 y 의 d/d̂ 분포 (점유 표본점 전체·활성·바닥 근처), 걸린 수. 바닥 근처 노드 법선 속도 v1 분포.
"""
import numpy as np
import torch

ON = [False]
NM_N = ("y", "eta", "g0buf", "f", "bGn")          # 반복별 (뉴턴 반복 0..nt-1 + 걸음 끝 다시 잰 nt)
NM_S = ("U", "vk", "ak", "z0", "Ut", "mn", "sk", "pm", "an", "kb", "meff")   # 걸음 시작 (뉴턴 전)
NM_E = ("vk", "ak", "U")                            # 걸음 끝 (rt_ga_end·구속 가속도 0 뒤)


def alloc(s):
    """WW.alloc 끝에서 (--nrec 위, ON 일 때만). 다시 할당되면 크기 맞춰 새로."""
    if not ON[0]:
        return
    s.prec = True
    nt = s.nt
    s.PRn, s.PRs, s.PRe = {}, {}, {}
    for nm in NM_N:
        t = getattr(s, nm, None)
        if isinstance(t, torch.Tensor):
            s.PRn[nm] = torch.zeros((nt + 1,) + tuple(t.shape), device=t.device, dtype=t.dtype)
    for nm in NM_S:
        t = getattr(s, nm, None)
        if isinstance(t, torch.Tensor):
            s.PRs[nm] = torch.zeros_like(t)
    for nm in NM_E:
        s.PRe[nm] = torch.zeros_like(getattr(s, nm))


def _on(s, kstep):
    return getattr(s, "prec", False) and kstep == 0


def start(s, kstep):
    """걸음 시작 (시작점·변수 바꿈 정한 뒤, 뉴턴 전)."""
    if not _on(s, kstep):
        return
    for nm, b in s.PRs.items():
        if nm != "meff":
            s.dcopy(b, getattr(s, nm))


def fel(s, kstep, it):
    """요소 (탄성·소성) 기울기 f -- 장벽 더하기 전."""
    if _on(s, kstep):
        s.dcopy(s.PRn["f"][it], s.f)


def bar(s, kstep, it):
    if _on(s, kstep) and "bGn" in s.PRn:
        s.dcopy(s.PRn["bGn"][it], s.bGn)


def grad(s, kstep, it):
    """rt_grad 뒤: y·η·r0 (= −g)."""
    if not _on(s, kstep):
        return
    for nm in ("y", "eta", "g0buf"):
        s.dcopy(s.PRn[nm][it], getattr(s, nm))
    if it == 0:
        s.dcopy(s.PRs["meff"], s.meff)


def end(s, kstep):
    """걸음 끝 (rt_ga_end + 재질 걸음 끝 뒤)."""
    if not _on(s, kstep):
        return
    for nm, b in s.PRe.items():
        s.dcopy(b, getattr(s, nm))


def _T(sk, pm, n, v):
    """Tapply (pm 0: √k v, pm 1: 접선 √k + 법선 그대로, pm 2: 0)."""
    vn = (v * n).sum(1, keepdim=True)
    o = sk[:, None] * v
    o1 = sk[:, None] * (v - vn * n) + vn * n
    o = torch.where((pm == 1)[:, None], o1, o)
    o = torch.where((pm == 2)[:, None], torch.zeros_like(o), o)
    return o


def _pct(t, qs=(0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)):
    t = t[torch.isfinite(t)].double()
    if t.numel() == 0:
        return None
    if t.numel() > 4_000_000:
        t = t[torch.randperm(t.numel(), device=t.device)[:4_000_000]]
    return [round(float(q), 6) for q in torch.quantile(t, torch.tensor(qs, device=t.device, dtype=torch.float64))]


class Ctx:
    def __init__(s, S, FL, g, TREL, X0, h):
        s.S, s.FL = S, FL
        dv = S.dev
        s.n = torch.tensor(FL["NR"] if FL else [0.0, 0.0, 1.0], device=dv, dtype=torch.float64)
        s.pn = float(FL["PN"]) if FL else 0.0
        s.gn = float(torch.tensor(g, dtype=torch.float64) @ s.n.cpu())
        c = S.ga_c
        s.gm, s.af, s.be = float(c["gm"]), float(c["af"]), float(c["be"])
        s.dt = float(S.dt)
        sg = S.stages(S.K)[0]
        s.tau2 = float(sg["tau"]) ** 2
        s.am = float(c["am"])
        assert sg["kind"] == "ga" and abs(float(sg["gm"]) - s.gm) < 1e-12
        s.thinv = 1.0 / (1.0 - s.af)
        s.cell = float(S.A.double()[:, 0].norm())
        s.TREL = TREL                                                               # 원래 순서 (구속 없으면 None)
        s.X0 = X0.to(dv).double()
        s.h = h

    def _nodes(s, nn):
        S = s.S
        IP = S.IP.cpu().tolist()
        nb1, nb2 = IP[7], IP[8]
        idx = torch.arange(nn, device=S.dev)
        L = torch.stack([idx // (nb1 * nb2) + IP[0], (idx // nb2) % nb1 + IP[1], idx % nb2 + IP[2]], 1).double()
        return L @ S.A.double().T, IP

    def _barq(s, y, U, IP, near_h):
        S = s.S
        nq = IP[12]
        if not getattr(S, "bar", False) or nq == 0:
            return None
        bx = S.bx[:4 * nq].view(nq, 4).double(); bm = bx[:, 3]
        occ = bm > 0
        act = (S.bact[:nq] > 0) & occ
        q = torch.arange(nq, device=S.dev)
        cb = S.cb[q >> 3].long()
        o = torch.arange(27, device=S.dev)
        oi, oj, ok = o // 9, (o // 3) % 3, o % 3
        ids = ((cb[:, 0:1] + oi) * IP[7] + (cb[:, 1:2] + oj)) * IP[8] + (cb[:, 2:3] + ok)
        w = S.BW.view(8, 27)[(q & 7)].double()
        xq = torch.where(occ[:, None], bx[:, :3] / bm.clamp_min(1e-30)[:, None], torch.zeros_like(bx[:, :3]))
        Ud, yd = U.double(), y.double()
        xq = xq + (w[:, :, None] * (Ud[ids] + s.thinv * (yd[ids] - Ud[ids]))).sum(1)
        d = xq @ s.n - s.pn
        dh = float(s.FL["DH"])
        r = d / dh
        hit = act & (d < dh)
        nr_ = occ & (d < near_h)
        bins = [-1e30, 0.0, 1.0, 2.0, 4.0, 8.0, 1e30]
        hist = lambda m_: [int((m_ & (r >= bins[i]) & (r < bins[i + 1])).sum()) for i in range(len(bins) - 1)]
        return dict(nocc=int(occ.sum()), nact=int(act.sum()), hit=int(hit.sum()), dh=dh,
                    r_occ=_pct(r[occ]), r_act=_pct(r[act]), r_near=_pct(r[nr_]), n_near=int(nr_.sum()),
                    hist_bins="d/d̂ <0, 0~1, 1~2, 2~4, 4~8, ≥8", hist_occ=hist(occ), hist_act=hist(act))

    def frame(s, fi, gt_orig):
        S = s.S
        assert S.nstage == 1, "--prec 는 K 1 만"
        nn = int(S.IP[9])
        n3 = s.n
        X, IP = s._nodes(nn)
        Ps = s.S.PRs
        U = Ps["U"].view(-1)[:nn * 3].view(nn, 3).double()
        v0 = Ps["vk"].view(-1)[:nn * 3].view(nn, 3).double()
        a0 = Ps["ak"].view(-1)[:nn * 3].view(nn, 3).double()
        z0 = Ps["z0"].view(-1)[:nn * 3].view(nn, 3).double()
        Ut = Ps["Ut"].view(-1)[:nn * 3].view(nn, 3).double()
        mn = Ps["mn"].view(-1)[:nn].double()
        meff = Ps["meff"].view(-1)[:nn].double()
        sk = Ps["sk"].view(-1)[:nn].double()
        pm = Ps["pm"].view(-1)[:nn]
        an = Ps["an"].view(-1)[:nn].double() if "an" in Ps else torch.zeros_like(mn)
        kb = Ps["kb"].view(-1)[:nn].double() if "kb" in Ps else torch.ones_like(mn)
        Pe = S.PRe
        v1 = Pe["vk"].view(-1)[:nn * 3].view(nn, 3).double()
        a1s = Pe["ak"].view(-1)[:nn * 3].view(nn, 3).double()
        hgt = (X + U) @ n3 - s.pn
        body = mn > 0
        fix = body & (pm == 2)
        pin = body & ~fix & (kb < 1.0)
        stk = body & ~fix & ~pin & (sk < 1.0)
        regs = dict(all=body, near1=body & (hgt < s.cell), near3=body & (hgt < 3 * s.cell), pin=body & (kb < 0.5), rest=body & ~(kb < 0.5),
                    fh=body & (kb >= 0.99) & ~fix & (hgt > 0.32) & (X[:, 0] > 1.5 - (fi + 1) * s.h / 1.5 + 0.06))     # 절 CM137: 방출된 쪽 6 cm 너머, 높이 > 0.8 m, 자유 (구속 인자 1) 노드
        out = dict(cell=s.cell, t0=fi * s.h, nn=nn, mass={k: float(mn[r].sum()) for k, r in regs.items()}, nnode={k: int(r.sum()) for k, r in regs.items()},
                   nkind=dict(fix=int(fix.sum()), pin=int(pin.sum()), pin_half=int((body & (kb < 0.5)).sum()), stk=int(stk.sum()), free=int((body & ~fix & ~pin & ~stk).sum())),
                   mkind=dict(fix=float(mn[fix].sum()), pin=float(mn[pin].sum()), stk=float(mn[stk].sum())),
                   meff_eq=bool(torch.equal(meff, mn.to(meff.dtype))), weight=float(mn[body].sum() * s.gn), am=s.am, gm=s.gm, af=s.af, tau2=s.tau2)
        ct3 = S.NR_CT.cpu().numpy() if hasattr(S, "NR_CT") else None
        its = []
        for it in range(S.nt + 1):
            Pn = S.PRn
            y = Pn["y"][it].view(-1)[:nn * 3].view(nn, 3).double()
            r0 = Pn["g0buf"][it].view(-1)[:nn * 3].view(nn, 3).double()
            fe = Pn["f"][it].view(-1)[:nn * 3].view(nn, 3).double()
            bg = Pn["bGn"][it].view(-1)[:nn].double() if "bGn" in Pn else torch.zeros_like(mn)
            Mi = meff / s.tau2
            iner = Mi[:, None] * (y - Ut)
            R = iner + fe + bg[:, None] * n3[None]
            Tg = _T(sk, pm, n3, -r0)
            C = R - Tg
            ran = True if it == S.nt else (ct3 is None or int(ct3[2 * it][0]) == 0)
            r_ = dict(it=it, end=(it == S.nt), ran=ran)
            el_n, bar_n, C_n, Tg_n, in_n = -(fe @ n3), -bg, C @ n3, Tg @ n3, iner @ n3
            for k, Rg in regs.items():
                q = dict(el=float(el_n[Rg].sum()), bar=float(bar_n[Rg].sum()), pinC=float(C_n[Rg & pin].sum()), stkC=float(C_n[Rg & stk].sum()),
                         fixC=float(C_n[Rg & fix].sum()), res=float(Tg_n[Rg].sum()), inert=float(in_n[Rg].sum()), grav=float(mn[Rg].sum() * s.gn),
                         el_abs=float(el_n[Rg].abs().sum()), pinC_abs=float(C_n[Rg & pin].abs().sum()), res_tan=float((Tg - Tg_n[:, None] * n3)[Rg].norm(dim=1).sum()),
                         stkC_tan=float((C - C_n[:, None] * n3)[Rg & stk].norm(dim=1).sum()), pinC_tan=float((C - C_n[:, None] * n3)[Rg & pin].norm(dim=1).sum()))
                q["chk"] = q["inert"] - (q["el"] + q["bar"] + q["pinC"] + q["stkC"] + q["fixC"] + q["res"])
                r_[k] = q
            its.append(r_)
            if it == S.nt:
                yE, CE, TgE, feE, bgE = y, C, Tg, fe, bg
        out["it"] = its
        # 걸음 속도 변화 분해 (걸음 끝 다시 잰 힘)
        gd = s.gm * s.dt
        dvo = {}
        a1f = (yE - Ut) / s.tau2 + torch.tensor(S.g, device=S.dev, dtype=torch.float64)[None] - s.am * a0      # HHT: Ũ = ŷ + τ² (g − αm a0)
        for k, Rg in regs.items():
            M = float(mn[Rg].sum())
            if M <= 0:
                continue
            W = lambda t_: float((mn * t_)[Rg].sum()) / M
            mden = meff.clamp_min(1e-30)
            fz = lambda t_, mk: gd * float((mn * t_ / mden)[Rg & mk].sum()) / M
            d_ = dict(v0=W(v0 @ n3), v1=W(v1 @ n3), meas=W((v1 - v0) @ n3), mem=s.dt * (1.0 - s.gm) * W(a0 @ n3), grav=gd * s.gn,
                      el=fz(-(feE @ n3), body), bar=fz(-bgE, body), pinC=fz(CE @ n3, pin), stkC=fz(CE @ n3, stk), fixC=fz(CE @ n3, fix), res=fz(TgE @ n3, body),
                      a1_formula=W(a1f @ n3), a1_stored=W(a1s @ n3))
            d_["sum"] = d_["mem"] + d_["grav"] + d_["el"] + d_["bar"] + d_["pinC"] + d_["stkC"] + d_["fixC"] + d_["res"]
            d_["chk"] = d_["meas"] - d_["sum"]
            d_["a1_wipe"] = d_["a1_formula"] - d_["a1_stored"]          # 걸음 끝 가속도 기억을 지운 몫 (다음 걸음 기억에서 빠짐)
            dvo[k] = d_
        out["dv"] = dvo
        # 바닥 근처 노드 법선 속도
        for k in ("near1", "near3"):
            Rg = regs[k]
            out[f"vn_{k}"] = dict(v1=_pct((v1 @ n3)[Rg]), v1_free=_pct((v1 @ n3)[Rg & ~pin & ~fix]), v1_stk=_pct((v1 @ n3)[Rg & stk]),
                                  sk=_pct(sk[Rg]), n_pm1=int((Rg & (pm == 1)).sum()), n_pm0=int((Rg & (pm == 0)).sum()), n_an=int((Rg & (an > 0)).sum()))
        out["bar"] = s._barq(yE, Ps["U"].view(-1)[:nn * 3].view(nn, 3), IP, 3 * s.cell)
        # 입자 (프레임 끝)
        out["p"] = s._parts(fi, gt_orig, kb, nn, IP)
        return out

    def _parts(s, fi, gt_orig, kb, nn, IP):
        S = s.S
        if s.TREL is None:
            return None
        t0, t1 = fi * s.h, (fi + 1) * s.h
        perm = S.perm.long()
        tr = s.TREL[perm]                                                           # 정렬 순서
        x = S.x.double()
        x0 = s.X0[perm]
        gt = gt_orig[perm].double()
        n3 = s.n
        hz, hz0, hg = x @ n3 - s.pn, x0 @ n3 - s.pn, gt @ n3 - s.pn
        # 구속 노드 가중 (프레임 시작 P2G 의 기저 = Np·fx)
        fx = S.fx.double(); pb = S.Np.long()
        w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], 1)    # [N, 3 (o), 3 (축)]
        pinw = torch.zeros(x.shape[0], device=S.dev, dtype=torch.float64)
        kbp = (kb < 0.5).double()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    nid = ((pb[:, 0] + i) * IP[7] + (pb[:, 1] + j)) * IP[8] + (pb[:, 2] + k)
                    ok = (nid >= 0) & (nid < nn)
                    pinw += torch.where(ok, w[:, i, 0] * w[:, j, 1] * w[:, k, 2] * kbp[nid.clamp(0, nn - 1)], torch.zeros_like(pinw))
        grp = dict(bound=tr > t1, now=(tr > t0) & (tr <= t1), rel=tr <= t0)
        o = dict(t0=t0, t1=t1)
        okg = torch.isfinite(gt).all(1)
        for k, m_ in grp.items():
            mg = m_ & okg
            if int(m_.sum()) == 0:
                o[k] = dict(n=0)
                continue
            o[k] = dict(n=int(m_.sum()), h=float(hz[m_].mean()), h_gt=float(hg[mg].mean()) if int(mg.sum()) else None, h0=float(hz0[m_].mean()),
                        drop=float((hz0 - hz)[m_].mean()), drop_gt=float((hz0 - hg)[mg].mean()) if int(mg.sum()) else None,
                        disp=float((x - x0)[m_].norm(dim=1).mean()), disp_gt=float((gt - x0)[mg].norm(dim=1).mean()) if int(mg.sum()) else None,
                        pinw_mean=float(pinw[m_].mean()), n_pinw_half=int((m_ & (pinw > 0.5)).sum()), n_pinw_any=int((m_ & (pinw > 1e-6)).sum()))
        if int(grp["bound"].sum()):
            xb = x0[grp["bound"]]
            o["bound_x"] = [float(xb[:, 0].min()), float(xb[:, 0].max())]
        o["all_x"] = [float(x0[:, 0].min()), float(x0[:, 0].max())]
        o["all_h0"] = [float(hz0.min()), float(hz0.max())]
        o["h_pct"] = _pct(hz); o["h_pct_gt"] = _pct(hg[okg])
        o["n_low"] = int((hz < 3 * s.cell).sum()); o["n_low_gt"] = int((hg < 3 * s.cell).sum())
        return o
