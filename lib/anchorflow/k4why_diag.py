"""절 S.9 진단 (--wdiag, 결과 불변): 우리 L000 K 4 발산 앞 몸통 안 가우스 점 뒤집힘의 '왜'.

풀이기 (fem_an_rt.RTSolver, wd=True) 가 그래프 안에서 복사만 해 두는 것: 걸음 시작점 y (WY0[k]), 뉴턴 반복 끝 y (WY[k nt + i]), 걸음 끝 U (WU[k]),
허용치 제어 카운터 CT·CD 사본 (WCT·WCD: rt_ctl op 3 뒤·op 7 뒤, 걸음 끝 op 8 뒤). 나머지는 이미 있는 진단 DG (뉴턴 반복별 SC 사본)·DGF (--gfin 걸음 끝 |g|²).
여기 (그래프 밖, 프레임 뒤) 에서:
- 가우스 점 q (부피 > 0) 마다 시험 X = F_q + (Σ_o y_o ⊗ ∇w_o(x_q)) F_q 의 det -- rt_elem 과 같은 식 -- 을 걸음 k 마다 [시작점, 뉴턴 1 끝, .., 뉴턴 nt 끝, 걸음 끝 U] 에서 (f64 로 다시 셈).
- 뉴턴 반복별 |g|/F (F² = CD[2] = 걸음 시작점 탄성·접촉 힘 몫 -- 허용치 규칙 |g| <= εN F 와 같은 척도), PCG 반복 수·상한 도달, √(r·z 끝 / r·z 처음), 선탐색 보폭, 걸음 끝 |g|/F (--gfin).
- 창 (wdiag <= 프레임 <= wdiag_end) 프레임은 점·노드·입자 원자료를 wd_f<프레임>.npz 로 (분석 exe/k4why_tab.py).
"""
import os

import numpy as np
import torch

OFF = 1 << 12


def _key(c):
    """전역 칸 좌표 [n,3] (정수) -> int64 열쇠."""
    c = c.long() + OFF
    return (c[:, 0] * (2 * OFF) + c[:, 1]) * (2 * OFF) + c[:, 2]


def part_q(S, x_orig):
    """입자 (원래 순서) -> 전역 칸 base·팔분 (rt_coord 와 같은 f32 식)."""
    Ai = S.Ai.float().reshape(3, 3)
    x = x_orig.float()
    y = torch.stack([Ai[a, 0] * x[:, 0] + Ai[a, 1] * x[:, 1] + Ai[a, 2] * x[:, 2] for a in range(3)], 1)
    fb = torch.floor(y - 0.5)
    fx = y - fb
    oc = (fx[:, 0] >= 1).long() * 4 + (fx[:, 1] >= 1).long() * 2 + (fx[:, 2] >= 1).long()
    return fb.long(), oc


def frame(S, fi, tolN, Jp0, Jp1, x0o, nr, pn, out=None):
    """프레임 fi (0 부터) 뒤 진단. Jp0·Jp1 = 프레임 시작·끝 입자 det F (원래 순서), x0o = 프레임 시작 입자 위치 (원래 순서), nr·pn = 바닥 법선·평면 값.
    돌려줌: fem.jsonl 용 요약 dict. out 이 있으면 원자료 npz."""
    dev = S.x.device
    ip = S.IP.cpu().tolist()
    nn, nc, nq = ip[9], ip[11], ip[12]
    nb1, nb2 = ip[7], ip[8]
    lo = torch.tensor(ip[0:3], device=dev)
    K, nt = S.nstage, S.nt
    cb = S.cb[:nc].long()
    ids = []
    for o in range(27):
        i_, j_, k_ = o // 9, (o // 3) % 3, o % 3
        ids.append(((cb[:, 0] + i_) * nb1 + (cb[:, 1] + j_)) * nb2 + (cb[:, 2] + k_))
    ids = torch.stack(ids, 1)                                                          # [nc, 27]
    qi = torch.arange(nq, device=dev)
    cq, oc = qi >> 3, qi & 7
    Fq = S.Fq[:nq].double().reshape(nq, 3, 3)
    Vq = S.Vq[:nq].float()
    on = Vq > 0
    BG = S.BG.double().reshape(8, 27, 3)
    snaps = []                                                                         # 걸음 k: [시작점, 뉴턴 1..nt 끝, 걸음 끝 U]
    for k in range(K):
        snaps.append(S.WY0[k])
        for i in range(nt):
            snaps.append(S.WY[k * nt + i])
        snaps.append(S.WU[k])
    NS = len(snaps)
    J = torch.empty(nq, NS, device=dev, dtype=torch.float64)
    smin = torch.empty(nq, NS, device=dev, dtype=torch.float64)                       # 부호 있는 최소 특이값 (det 부호 x 최소 |σ|)
    CH = 32768
    for s0 in range(0, nq, CH):
        s1 = min(nq, s0 + CH)
        idq = ids[cq[s0:s1]]                                                           # [m, 27]
        bg = BG[oc[s0:s1]]                                                             # [m, 27, 3]
        F = Fq[s0:s1]
        for si, Y in enumerate(snaps):
            yy = Y[:nn].double()[idq]                                                  # [m, 27, 3]
            A = torch.einsum("moa,moc->mac", yy, bg)
            X = F + A @ F
            d = torch.linalg.det(X)
            J[s0:s1, si] = d
            Xf = torch.where(torch.isfinite(X).all(-1).all(-1)[:, None, None], X, torch.zeros_like(X))   # 비유한 행은 svd 가 멈추므로 0 (det 는 nan 그대로)
            sv = torch.where(torch.isfinite(d), torch.linalg.svdvals(Xf)[:, 2], torch.full_like(d, float('nan')))
            smin[s0:s1, si] = torch.sign(d) * sv
    J0 = torch.linalg.det(Fq)
    Jn = torch.where(on[:, None], torch.nan_to_num(J, nan=-1e30), torch.full_like(J, 1e30))
    J0n = torch.where(on, torch.nan_to_num(J0, nan=-1e30), torch.full_like(J0, 1e30))
    jmin_s = Jn.min(0).values                                                          # 시점별 최소
    inv_any = (Jn <= 0).any(1)
    vmed = float(Vq[on].median())
    mn = S.mn[:nn].float()
    mmed = float(mn[mn > 0].median())
    # 뉴턴·PCG 반복별 (DG: [2i] = rt_grad 뒤 SC, [2i+1] = 선탐색 뒤 SC; WCT·WCD: op 3 뒤 [2i], op 7 뒤 [2i+1])
    DG = S.DG.cpu().numpy(); CT = S.WCT.cpu().numpy(); CD = S.WCD.cpu().numpy(); DGF = S.DGF.cpu().numpy()
    it = []
    for k in range(K):
        F2 = CD[2 * (k * nt)][2]
        for i in range(nt):
            a_, b_ = 2 * (k * nt + i), 2 * (k * nt + i) + 1
            ran = int(CT[a_][0]) == 0                                                  # 이 반복 PCG·갱신을 했는지 (op 3 에서 멈춤 표시 없음)
            stop_here = (not ran) and (i == 0 or int(CT[2 * (k * nt + i - 1)][0]) == 0)   # 이 반복의 |g| 판정에서 멈춤 (허용치 도달)
            g = float(np.sqrt(max(DG[a_][1], 0.0))) if (ran or stop_here) else None
            rec = dict(k=k, i=i, ran=ran, stop=bool(stop_here), g=g, gF=(g / np.sqrt(F2) if (g is not None and F2 > 0) else None))
            if ran:
                rec.update(nP=int(CT[b_][3] - CT[a_][3]), cap=int(CT[b_][5] - CT[a_][5]), rz=float(np.sqrt(max(DG[b_][0], 0.0) / DG[a_][0])) if DG[a_][0] > 0 else None,
                           st=float(DG[b_][14]))
            it.append(rec)
    gend = []
    for k in range(K):
        F2 = CD[2 * (k * nt)][2]
        g1 = DGF[k][1]
        gend.append(float(np.sqrt(max(g1, 0.0) / F2)) if F2 > 0 and g1 > 0 else None)
    nconv = sum(1 for q in gend if q is not None and q <= tolN)
    st_min = min([r["st"] for r in it if r.get("st") is not None] or [1.0])
    o = dict(Jq0min=float(J0n.min()), nJ0le0=int((J0n <= 0).sum()), Jqmin=float(jmin_s.min()), Jq_s=[float(q) for q in jmin_s.tolist()], nqinv=int(inv_any.sum()),
             nqlt01=int((Jn.min(1).values < 0.1).sum()), nbadq=int(S.nbad), Jp0min=float(torch.nan_to_num(Jp0, nan=-1e30).min()), Jp1min=float(torch.nan_to_num(Jp1, nan=-1e30).min()),
             nJp0le0=int((Jp0 <= 0).sum()), nJp1le0=int((Jp1 <= 0).sum()), it=it, gend=gend, nconv=nconv, st_min=st_min, vmed=vmed, mmed=mmed)
    if out is not None:
        # 입자 -> 점 (프레임 시작 위치): 팔분 입자 수·칸 입자 수·팔분 입자 det 최소
        pb, poc = part_q(S, x0o)
        gc = lo[None] + cb                                                             # 압축 칸 전역 좌표
        ck = _key(gc)
        so, si_ = torch.sort(ck)
        pk = _key(pb)
        pos = torch.searchsorted(so, pk).clamp(max=nc - 1)
        hit = so[pos] == pk
        pc = torch.where(hit, si_[pos], torch.full_like(pos, -1))                      # 입자의 압축 칸 (-1 = 상자 밖이면 안 생김)
        pq = torch.where(pc >= 0, pc * 8 + poc, torch.full_like(pc, -1))
        okp = pq >= 0
        qcnt = torch.bincount(pq[okp], minlength=nq)[:nq]
        ccnt = torch.bincount(pc[okp], minlength=nc)[:nc]
        qjp = torch.full((nq,), 1e30, device=dev, dtype=torch.float64)
        qjp.scatter_reduce_(0, pq[okp], torch.nan_to_num(Jp0[okp].double(), nan=-1e30), reduce="amin")
        Aa = S.Aacc[:nq].double()
        aw = Aa[:, 27]
        Rs = Aa[:, :9].reshape(nq, 3, 3) / aw.clamp_min(1e-30)[:, None, None]
        Rs = torch.where(torch.isfinite(Rs).all(-1).all(-1)[:, None, None], Rs, torch.zeros_like(Rs))
        rsv = torch.where(aw > 0, torch.linalg.svdvals(Rs)[:, 2], torch.full_like(aw, -1.0))   # 회전 평균 일치 (1 = 같은 회전, 작으면 팔분 입자 회전이 갈림)
        nmask = mn > 0
        nidx = torch.nonzero(nmask).squeeze(1)
        ngc = torch.stack([nidx // (nb1 * nb2), (nidx // nb2) % nb1, nidx % nb2], 1) + lo[None]
        sel = torch.nonzero(on).squeeze(1)
        xq = S.xq0[:nq].float()
        hq = xq.double() @ torch.tensor(nr, device=dev, dtype=torch.float64) - pn
        np.savez_compressed(
            out, frame=np.array([fi + 1]), K=np.array([K]), nt=np.array([nt]), lo=np.array(ip[0:3]), nb=np.array(ip[6:9]),
            q_qi=sel.int().cpu().numpy(), q_cell=gc[cq[sel]].int().cpu().numpy(), q_oc=oc[sel].to(torch.int8).cpu().numpy(), q_V=(Vq[sel] / vmed).cpu().numpy(), q_m=S.mq[:nq][sel].cpu().numpy(),
            q_J0=J0[sel].float().cpu().numpy(), q_J=J[sel].float().cpu().numpy(), q_smin=smin[sel].float().cpu().numpy(), q_bad=S.badq[:nq][sel].to(torch.int8).cpu().numpy(),
            q_np=qcnt[sel].int().cpu().numpy(), q_nc=ccnt[cq[sel]].int().cpu().numpy(), q_jp=qjp[sel].float().cpu().numpy(), q_rsv=rsv[sel].float().cpu().numpy(),
            q_x=xq[sel].cpu().numpy(), q_h=hq[sel].float().cpu().numpy(), q_ids=ids[cq[sel]].int().cpu().numpy(),
            n_idx=nidx.int().cpu().numpy(), n_cell=ngc.int().cpu().numpy(), n_m=(mn[nidx] / mmed).cpu().numpy(), n_mq=(S.mqn[:nn][nidx].float() / mmed).cpu().numpy(),
            n_U=torch.stack([S.WU[k][:nn][nidx] for k in range(K)], 0).cpu().numpy(), n_Y0=torch.stack([S.WY0[k][:nn][nidx] for k in range(K)], 0).cpu().numpy(),
            p_J0=Jp0.float().cpu().numpy(), p_J1=Jp1.float().cpu().numpy(), p_x0=x0o.float().cpu().numpy(), p_q=pq.int().cpu().numpy(),
            cellh=np.array([float(S.A.reshape(3, 3)[:, 0].norm())]), vmed=np.array([vmed]), mmed=np.array([mmed]))
    return o
