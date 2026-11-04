"""절 AT 진단 (조정자 11-05): 충돌 프레임 G2P 에서 입자 F 가 퇴화하는지 직접 측정.

AORTSolver 의 커널 호출 (s.k) 을 인스턴스에서 감싸 그래프 밖 프레임에서 상태를 복사한다.
- 충돌 프레임 (phase "a"): 걸음마다 a_n (걸음 시작 가속도 상태)·U·vk·고착 몫, G2P 직전 입자 F_old·x·base·fx·Ust, G2P 직후 F_new.
- 다음 프레임 (phase "b", nanchk 와 함께): rt_gauss 직후 F_q·V_q·badq·칸 번호·입자 F (= F_new), 첫 걸음 시작 a_n·Ut·시작점.
풀이·결과는 바꾸지 않는다 (복사만).
"""
import json
import math

import torch

OFF = [(o // 9, (o // 3) % 3, o % 3) for o in range(27)]


def _bw(fx):
    w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2], -1)          # [P,3(축),3(노드)]
    dw = torch.stack([fx - 1.5, -2 * (fx - 1), fx - 0.5], -1)
    return w, dw


def part_nodes(pb, fx, nb1, nb2, Ai):
    """aq_g2p 와 같은 식: 27 노드 번호, 가중, 월드 기울기 gx (gx[b] = Σ_a Ai[a,b] gy[a])."""
    fx = fx.double(); pb = pb.long()
    w, dw = _bw(fx)
    nid, ww, gy = [], [], []
    for (i, j, k) in OFF:
        nid.append(((pb[:, 0] + i) * nb1 + (pb[:, 1] + j)) * nb2 + (pb[:, 2] + k))
        ww.append(w[:, 0, i] * w[:, 1, j] * w[:, 2, k])
        gy.append(torch.stack([dw[:, 0, i] * w[:, 1, j] * w[:, 2, k], w[:, 0, i] * dw[:, 1, j] * w[:, 2, k], w[:, 0, i] * w[:, 1, j] * dw[:, 2, k]], -1))
    nid = torch.stack(nid, 1); ww = torch.stack(ww, 1); gy = torch.stack(gy, 1)
    gx = gy @ Ai.double().reshape(3, 3)
    return nid, ww, gx


def grad_at(Un, nid, gx):
    """I + Σ_o U[nid_o] ⊗ gx_o (aq_g2p 의 A)."""
    A = torch.einsum("poa,pob->pab", Un.double()[nid], gx)
    return A + torch.eye(3, dtype=A.dtype, device=A.device)


def sv_signed(F):
    s = torch.linalg.svdvals(F)
    d = torch.linalg.det(F)
    s[:, 2] = torch.where(d < 0, -s[:, 2], s[:, 2])
    return s


class Diag:
    def __init__(s, S):
        s.S = S
        s.phase = None
        s.ks = -1
        s.a, s.b = {}, {}
        s.orig = S.k
        S.k = s.k

    def _c(s, nm):
        t = getattr(s.S, nm)
        return t.clone()

    def k(s, name, n, args, b=128):
        ph = s.phase
        if ph is None:
            return s.orig(name, n, args, b)
        S = s.S
        D = s.a if ph == "a" else s.b
        if name == "ao_step0_ga":
            s.ks += 1
            torch.cuda.synchronize()
            D.setdefault("a_n", []).append(S.gaA.clone()); D.setdefault("U_n", []).append(S.U.clone()); D.setdefault("v_n", []).append(S.vk.clone())
        if name == "aq_g2p":
            torch.cuda.synchronize()
            D["F_old"] = S.F.clone(); D["x_old"] = S.x.clone(); D["pb"] = S.Np.clone(); D["fx"] = S.fx.clone(); D["Ust"] = S.Ust.clone(); D["IP"] = S.IP.clone()
            D["mn"] = S.mn.clone()
        if name == "ao_stepend_ga":
            torch.cuda.synchronize()
            D.setdefault("y_end", []).append(S.y.clone())
        s.orig(name, n, args, b)
        torch.cuda.synchronize()
        if name == "rt_gauss":
            for nm in ("Fq", "Vq", "badq", "cidx", "pcd", "fx", "Np", "F", "IP", "cb", "Aacc"):
                D[nm + "_g"] = s._c(nm)
        if name == "ao_step0_ga":
            D.setdefault("Ut", []).append(S.Ut.clone()); D.setdefault("z0", []).append(S.z0.clone())
        if name == "ao_floor_n":
            D.setdefault("an", []).append(S.an.clone()); D.setdefault("mqn", []).append(S.mqn.clone()); D.setdefault("y_s", []).append(S.y.clone())
            D.setdefault("sk", []).append(S.sk.clone()); D.setdefault("pm", []).append(S.pm.clone())
        if name == "aw_y0z_b":
            D.setdefault("y0", []).append(S.y.clone())
        if name == "ao_stepend_ga":
            D.setdefault("U_e", []).append(S.U.clone()); D.setdefault("v_e", []).append(S.vk.clone()); D.setdefault("a_e", []).append(S.gaA.clone())
        if name == "aq_g2p":
            D["F_new"] = S.F.clone(); D["pa_new"] = S.pa.clone() if S.pa is not None else None
        if name == "ao_g2p_a":
            D["a_last"] = S.gaA.clone()

    # ------------------------------------------------------------------
    def analyze(s, hit, out):
        S = s.S; A_, B_ = s.a, s.b
        c = S.ga_c; dt = float(S.dt); af, be, gm = c["af"], c["be"], c["gm"]
        cell = float(S.A[:, 0].norm())
        ca_pred = (1 - af) * dt * dt * (0.5 - be)                                     # a_n 의 관성 예측 Ut 몫 (HHT am = 0)
        ca_U = dt * dt * ((0.5 - be) - be * (1 - gm) / gm)                            # a_n 의 걸음 끝 U_{n+1} 명시 몫 (Newmark 식에 a_{n+1} 식 대입)
        o = dict(dt=dt, cell=cell, af=af, be=be, gm=gm, ca_pred=ca_pred, ca_U=ca_U)
        ipa = A_["IP"].tolist(); nb1a, nb2a = ipa[7], ipa[8]; nna = ipa[9]
        K = len(A_["U_e"])
        # ---- (1) 전체 입자: F_old, A_K, F_new 행렬식 ----
        Np = S.N
        dFo = torch.empty(Np, dtype=torch.float64, device=S.dev); dFn = torch.empty_like(dFo); dAK = torch.empty_like(dFo); dRM = torch.empty_like(dFo)
        dAinc = torch.empty(K, Np, dtype=torch.float64, device=S.dev); dAa = torch.empty(K, Np, dtype=torch.float64, device=S.dev)
        mnA = A_["mn"]
        for i0 in range(0, Np, 200000):
            sl = slice(i0, min(Np, i0 + 200000))
            nid, ww, gx = part_nodes(A_["pb"][sl], A_["fx"][sl], nb1a, nb2a, S.Ai)
            Fo = A_["F_old"][sl].double().reshape(-1, 3, 3); Fn = A_["F_new"][sl].double().reshape(-1, 3, 3)
            Ap = None
            for kk in range(K):
                Ak = grad_at(A_["Ust"][kk], nid, gx)
                dAinc[kk, sl] = torch.linalg.det(Ak) / (torch.linalg.det(Ap) if Ap is not None else 1.0)
                Aa = grad_at(ca_U * A_["a_n"][kk], nid, gx)                             # a_n 명시 몫만의 변형 기울기
                dAa[kk, sl] = torch.linalg.det(Aa)
                Ap = Ak
            dFo[sl] = torch.linalg.det(Fo); dFn[sl] = torch.linalg.det(Fn); dAK[sl] = torch.linalg.det(Ap)
            dRM[sl] = dFn[sl] / (dAK[sl] * dFo[sl])
        # 마지막 걸음: 고착 노드 끝 변위를 풀이의 끝 위치 U_y = (y − αf U_n)/(1 − αf) 로 바꾸면 (처치 (b)·(c) 의 Newmark 끝 위치 대신)
        kl = K - 1
        Un_l = A_["U_n"][kl]; Uy_l = (A_["y_end"][kl] - af * Un_l) / (1 - af)
        stk_l = ((A_["mqn"][kl] > 0) & (A_["an"][kl] > 0))
        U_alt = torch.where(stk_l[:, None], Uy_l, A_["U_e"][kl])
        dAalt = torch.empty_like(dFo)
        for i0 in range(0, Np, 200000):
            sl = slice(i0, min(Np, i0 + 200000))
            nid, ww, gx = part_nodes(A_["pb"][sl], A_["fx"][sl], nb1a, nb2a, S.Ai)
            Apv = grad_at(A_["Ust"][kl - 1], nid, gx) if kl > 0 else None
            Aal = grad_at(U_alt, nid, gx)
            dAalt[sl] = torch.linalg.det(Aal) / (torch.linalg.det(Apv) if Apv is not None else 1.0)
        nna_ = nna
        dUe = (A_["U_e"][kl] - Un_l)[:nna_]; dUy = (Uy_l - Un_l)[:nna_]; sk_ = stk_l[:nna_]
        o["last_stuck"] = dict(n_stuck=int(sk_.sum()), Ue_minus_Uy_cell_max=float((dUe - dUy)[sk_].norm(dim=-1).max()) / cell if bool(sk_.any()) else None,
                               Ue_minus_Uy_cell_med=float((dUe - dUy)[sk_].norm(dim=-1).median()) / cell if bool(sk_.any()) else None,
                               nonstuck_Ue_minus_Uy_cell_max=float((dUe - dUy)[~sk_ & (mnA[:nna_] > 0)].norm(dim=-1).max()) / cell)
        s._dAalt = dAalt
        s._Ualt = (U_alt, Uy_l, Un_l, stk_l)
        def cnt(x):
            return dict(n_le0=int((x <= 0).sum()), n_lt1e_3=int((x < 1e-3).sum()), n_lt1e_6=int((x < 1e-6).sum()), min=float(x.min()), p001=float(x.quantile(1e-3)) if x.numel() < 16000000 else None, med=float(x.median()))
        o["all"] = dict(det_F_old=cnt(dFo), det_A_K=cnt(dAK), det_F_new=cnt(dFn), det_Ainc=[cnt(dAinc[kk]) for kk in range(K)], det_Aa=[cnt(dAa[kk]) for kk in range(K)])
        o["all"]["det_Ainc_last_stuckUy"] = cnt(dAalt)
        bad = dFn < 1e-3
        o["all"]["new_lt1e_3"] = dict(n=int(bad.sum()), old_lt1e_3=int((bad & (dFo < 1e-3)).sum()), AK_lt1e_3=int((bad & (dAK < 1e-3)).sum()),
                                     rm_changed=int((bad & ((dRM - 1).abs() > 1e-3)).sum()))
        # ---- (2) 걸음별 몸체 상태 ----
        st = []
        for kk in range(K):
            m = mnA[:nna]; act = m > S.mthr
            an = A_["an"][kk][:nna]; mq = A_["mqn"][kk][:nna]; stk = (mq > 0) & (an > 0)
            a_n = A_["a_n"][kk][:nna]; a_e = A_["a_e"][kk][:nna]; v_n = A_["v_n"][kk][:nna]; v_e = A_["v_e"][kk][:nna]
            dU = (A_["U_e"][kk] - (A_["U_e"][kk - 1] if kk > 0 else torch.zeros_like(A_["U_e"][kk])))[:nna]
            st.append(dict(k=kk, n_stuck=int((stk & act).sum()), a_n_med_z=float(a_n[act][:, 2].median()), a_e_med_z=float(a_e[act][:, 2].median()),
                           v_n_med_z=float(v_n[act][:, 2].median()), v_e_med_z=float(v_e[act][:, 2].median()),
                           pred_a_cell_med=float((ca_pred * a_n[act]).norm(dim=-1).median()) / cell, pred_a_cell_max=float((ca_pred * a_n[act]).norm(dim=-1).max()) / cell,
                           pred_a_n_gt1cell=int(((ca_pred * a_n[act]).norm(dim=-1) > cell).sum()), dU_cell_med=float(dU[act].norm(dim=-1).median()) / cell, n_act=int(act.sum())))
        o["steps_a"] = st
        # ---- (3) 다음 프레임 시작: 퇴화 가우스 점과 그 칸 입자 ----
        ipb = B_["IP_g"].tolist(); nq = ipb[12]; nb1b, nb2b = ipb[7], ipb[8]
        Fq = B_["Fq_g"][:nq].double().reshape(nq, 3, 3); JF = torch.linalg.det(Fq); Vq = B_["Vq_g"][:nq]
        vmed = float(Vq[Vq > 0].median())
        o["Fq_b"] = dict(nq=nq, n_le0=int(((JF <= 0) & (Vq > 0)).sum()), n_lt1e_3=int(((JF < 1e-3) & (Vq > 0)).sum()), n_lt1e_6=int(((JF < 1e-6) & (Vq > 0)).sum()),
                         F_same_as_new=bool(torch.equal(B_["F_g"], A_["F_new"])))
        qs = []
        try:
            qs = [int(p_["q"]) for p_ in hit["st"]["as_inv"]["pts"]]
        except Exception:
            pass
        JFv = torch.where(Vq > 0, JF, torch.full_like(JF, float("inf")))
        low = torch.argsort(JFv)[:16].tolist()
        o["as_inv_q"] = qs
        Q = list(dict.fromkeys(qs + [q for q in low if float(JFv[q]) < 1e-3]))
        cid = B_["cidx_g"][B_["pcd_g"].long()].long()
        oc_p = ((B_["fx_g"][:, 0] >= 1).long() * 4 + (B_["fx_g"][:, 1] >= 1).long() * 2 + (B_["fx_g"][:, 2] >= 1).long())
        cells = {}
        for q in Q:
            cells.setdefault(q >> 3, []).append(q)
        nidA_all = None
        # 다음 프레임 첫 걸음 시작 (노드 번호 = 다음 프레임 상자)
        a4 = B_["a_n"][0]; Ut4 = B_["Ut"][0]; y04 = B_["y0"][0] if B_.get("y0") else B_["y_s"][0]; ys4 = B_["y_s"][0]
        nnb = ipb[9]; mB = S.mn[:nnb]
        actB = mB > S.mthr
        o["start_b"] = dict(a_n_med_z=float(a4[:nnb][actB][:, 2].median()), pred_a_cell_med=float((ca_pred * a4[:nnb][actB]).norm(dim=-1).median()) / cell,
                            pred_a_n_gt1cell=int(((ca_pred * a4[:nnb][actB]).norm(dim=-1) > cell).sum()), n_act=int(actB.sum()))
        BG = S.BG.reshape(8, 27, 3).double(); cbB = B_["cb_g"]
        def Xdet(q, v):
            c_ = q >> 3; oc = q & 7; cbq = cbB[c_].long().tolist()
            ids = [((cbq[0] + i) * nb1b + (cbq[1] + j)) * nb2b + (cbq[2] + k) for (i, j, k) in OFF]
            A = torch.zeros(3, 3, dtype=torch.float64, device=S.dev)
            for o_, nd in enumerate(ids):
                A += v[nd].double()[:, None] * BG[oc, o_][None, :]
            Fq_ = Fq[q]
            return float(torch.linalg.det(Fq_ + A @ Fq_)), ids
        o["cells"] = []
        for c_, ql in list(cells.items())[:6]:
            pis = torch.nonzero(cid == c_).flatten()
            rec = dict(cell=c_, qs=[])
            for q in ql:
                J0, ids = Xdet(q, torch.zeros_like(Ut4))
                rec["qs"].append(dict(q=q, oc=q & 7, detFq=float(JF[q]), vq_rel=float(Vq[q]) / vmed, badq=int(B_["badq_g"][q]), w_oct=float(B_["Aacc_g"][q, 27]),
                                      J_y0=Xdet(q, y04)[0], J_ys=Xdet(q, ys4)[0], J_Ut=Xdet(q, Ut4)[0], J_Ut_noa=Xdet(q, Ut4 - ca_pred * a4)[0],
                                      a_cell_nodes=[round(float((ca_pred * a4[nd]).norm()) / cell, 3) for nd in ids],
                                      stuck_nodes=int(((B_["mqn"][0][ids] > 0) & (B_["an"][0][ids] > 0)).sum()), light_nodes=int((S.mn[ids] < S.mthr).sum())))
            # 칸 안 입자
            P = []
            Fp = B_["F_g"][pis].double().reshape(-1, 3, 3)
            nid, ww, gx = part_nodes(A_["pb"][pis], A_["fx"][pis], nb1a, nb2a, S.Ai)
            for ii, p in enumerate(pis.tolist()):
                Fo = A_["F_old"][p].double().reshape(3, 3)
                Aks = [grad_at(A_["Ust"][kk], nid[ii:ii + 1], gx[ii:ii + 1])[0] for kk in range(K)]
                inc = [float(torch.linalg.det(Aks[kk]) / (torch.linalg.det(Aks[kk - 1]) if kk > 0 else 1.0)) for kk in range(K)]
                Aa = [grad_at(ca_U * A_["a_n"][kk], nid[ii:ii + 1], gx[ii:ii + 1])[0] for kk in range(K)]
                nds = nid[ii].tolist()
                P.append(dict(p=p, oc=int(oc_p[p]), vol=float(S.vol[p]), detF_new=float(torch.linalg.det(Fp[ii])), sv_new=[float(q) for q in sv_signed(Fp[ii:ii + 1])[0]],
                              detF_old=float(torch.linalg.det(Fo)), sv_old=[float(q) for q in sv_signed(Fo[None])[0]],
                              detA_K=float(torch.linalg.det(Aks[-1])), sv_A_K=[float(q) for q in sv_signed(Aks[-1][None])[0]], detAinc=inc,
                              rm_ratio=float(torch.linalg.det(Fp[ii]) / (torch.linalg.det(Aks[-1]) * torch.linalg.det(Fo))),
                              detAa=[float(torch.linalg.det(q)) for q in Aa], normAa=[float((q - torch.eye(3, dtype=q.dtype, device=q.device)).norm()) for q in Aa],
                              nodes_a=[dict(k=kk, stuck=int(((A_["mqn"][kk][nds] > 0) & (A_["an"][kk][nds] > 0)).sum()), light=int((mnA[nds] < S.mthr).sum()), m0=int((mnA[nds] <= 0).sum()),
                                            a_cell_max=float((ca_pred * A_["a_n"][kk][nds]).norm(dim=-1).max()) / cell,
                                            dU_cell_max=float((A_["U_e"][kk][nds] - (A_["U_e"][kk - 1][nds] if kk > 0 else 0)).norm(dim=-1).max()) / cell,
                                            dU_spread_cell=float((A_["U_e"][kk][nds] - (A_["U_e"][kk - 1][nds] if kk > 0 else 0)).std(0).norm()) / cell) for kk in range(K)]))
                U_alt, Uy_l, Un_l, stk_l = s._Ualt
                nl_ = []
                for nd in nds:
                    nl_.append(dict(stuck=int(stk_l[nd]), light=int(mnA[nd] < S.mthr), m=float(mnA[nd]), Ue_cell=[round(float(q) / cell, 3) for q in (A_["U_e"][K - 1][nd] - Un_l[nd])],
                                    Uy_cell=[round(float(q) / cell, 3) for q in (Uy_l[nd] - Un_l[nd])], v_e=[round(float(q), 3) for q in A_["v_e"][K - 1][nd]], a_e=[round(float(q), 1) for q in A_["a_e"][K - 1][nd]]))
                P[-1]["nodes_last"] = nl_
                P[-1]["detAinc_last_stuckUy"] = float(s._dAalt[p])
            rec["particles"] = P
            # F_q 구성 다시: 팔분 영역 (없으면 칸) 입자의 기하 평균 det (회전·로그) 와 유클리드 평균 det
            for qr in rec["qs"]:
                sel = [pp for pp in P if pp["oc"] == qr["oc"]] or P
                w = torch.tensor([pp["vol"] for pp in sel], dtype=torch.float64)
                Fs = B_["F_g"][[pp["p"] for pp in sel]].double().reshape(-1, 3, 3).cpu()
                dd = torch.linalg.det(Fs)
                qr["from"] = "octant" if any(pp["oc"] == qr["oc"] for pp in P) else "cell"
                qr["n_part"] = len(sel)
                qr["det_eu"] = float(torch.linalg.det((w[:, None, None] * Fs).sum(0) / w.sum()))
                qr["det_geo"] = float(torch.exp((w * torch.log(dd.clamp_min(1e-300))).sum() / w.sum())) if bool((dd > 0).all()) else None
                qr["any_inv"] = bool((dd <= 0).any())
            o["cells"].append(rec)
        json.dump(o, open(out, "w"), indent=1)
        return o
