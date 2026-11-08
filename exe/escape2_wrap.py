"""절 CM43 묶음 escape2 감싸개 (수박 옛 실행기 입자 이탈): 진단 D2 (결과 불변, 읽기만) 와 고침 후보를 붙인 뒤 exe/escape1_wrap.py 로 넘김
(그쪽 진단 AF_E1_DG·후보 T2·I 인자는 그대로 넘어감).

사용: python exe/escape2_wrap.py [--e2c 후보] <escape1_wrap / speed3_wrap 인자...> <실행기.py> <실행기 인자...>
  환경 AF_E2_DG=1 이면 프레임 (모든 시도 끝) 마다:
    <out>/e2.jsonl  -- 프레임 요약: 고정 노드 (m_n < mthr, pm 2 = 관성 예측에 묶임) 수, 입자 27 노드 가중 중 고정 노드 몫 w_pin 분포 (> 0.5·= 1 입자 수),
                       w_pin > 0.5 입자의 속도 - 몸 질량 중심 속도 최대, 걸음 판정 (덜 풀림·‖g‖²/F²)
    새 이탈 입자 (escape1 과 같은 정의: 몸 상자 (0.1~99.9 백분위) 에서 4 칸 넘게; 이 프레임에 처음) 가 있으면 앞 R 프레임 (고리 버퍼) 의 그 입자 기록을
    <out>/e2_f<프레임>.npz 와 e2.jsonl 의 'esc' 줄에: 입자 시작·끝 위치·속도, logJp, J, SNH 1 차 PK 응력 주값 (μ σ_i + (λ_s (J − 1) − μ) σ_j σ_k, 입자 탄성 F),
    칸 질량 / 칸 질량 중앙, 27 노드의 가중·질량 비·고정 여부 (pm)·후퇴 오일러 집합 (ulm)·걸음 시작 속도 (mv/mn)·끝 속도 vk·가속도·변위 U,
    G2P 분해 (입자 속도 = Σ w vk 중 고정 노드 몫·풀린 노드 몫).
  진단 실패는 풀이에 영향 없음 (<out>/e2_err.txt).
"""
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
CAND = {}
while argv and argv[0].startswith("--e2"):                                              # 감싸개 인자 (실행기 앞에만)
    CAND[argv[0][2:]] = argv[1]
    del argv[:2]

if CAND:
    from anchorflow import fem_escape2 as E2
    E2.install(CAND, out)

if os.environ.get("AF_E2_DG", "0") == "1":
    import numpy as np
    import torch
    from anchorflow import fem_ao_rt as AORT
    from anchorflow import fem_wmrx as WRX

    LOG = open(os.path.join(out, "e2.jsonl"), "w")
    RING = int(os.environ.get("AF_E2_RING", "5"))
    ST = dict(dep=0, pre=None, ring=[], far=None, nesc=0)
    NODE_CAP = 6_000_000                                                                 # 노드 상자가 이보다 크면 노드 배열은 고리에 안 둠 (메모리)

    def _bw(fx):
        return torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], dim=-1)

    def _nid(pb, nb1, nb2):
        o = torch.arange(27, device=pb.device)
        i, j, k = o // 9, (o // 3) % 3, o % 3
        return ((pb[:, 0:1] + i) * nb1 + (pb[:, 1:2] + j)) * nb2 + (pb[:, 2:3] + k)

    def _w27(fx):
        w = _bw(fx)
        o = torch.arange(27, device=fx.device)
        i, j, k = o // 9, (o // 3) % 3, o % 3
        return w[:, 0, i] * w[:, 1, j] * w[:, 2, k]

    def _snh(F, mu, lam):
        """입자 탄성 F [n,9] -> (J, 주 신장 3, SNH 1 차 PK 주값 3, |P|)"""
        Fm = F.reshape(-1, 3, 3).double()
        U_, s, Vh = torch.linalg.svd(Fm)
        d = torch.det(Fm)
        s = s.clone()
        s[:, 2] = torch.where(d < 0, -s[:, 2], s[:, 2])
        J = s[:, 0] * s[:, 1] * s[:, 2]
        lss = lam + mu
        cc = lss * (J - 1.0) - mu
        P = torch.stack([mu * s[:, 0] + cc * s[:, 1] * s[:, 2], mu * s[:, 1] + cc * s[:, 0] * s[:, 2], mu * s[:, 2] + cc * s[:, 0] * s[:, 1]], -1)
        return J, s, P, P.norm(dim=-1)

    @torch.no_grad()
    def pre(s):
        N = int(getattr(s, "N", s.x.shape[0]))
        ST["pre"] = dict(x=s.x[:N].clone(), v=s.v[:N].clone(), lj=s.ljp[:N].clone(), F=s.F[:N].clone(), pa=(s.pa[:N].clone() if getattr(s, "pa", None) is not None else None))

    @torch.no_grad()
    def post(s, fi):
        IP = s.IP.cpu()
        nb1, nb2, nn = int(IP[7]), int(IP[8]), int(IP[9])
        N = int(getattr(s, "N", s.x.shape[0]))
        mn = s.mn[:nn].float()
        has = mn > 0
        med = float(mn[has].median()) if bool(has.any()) else 1.0
        mthr = float(getattr(s, "mthr", 0.0))
        pin = (s.pm[:nn] == 2) & has
        pb = s.Np.reshape(-1, 3)[:N].long()
        fx = s.fx.reshape(-1, 3)[:N].float()
        wpin = torch.zeros(N, device=fx.device)
        for c0 in range(0, N, 262144):                                                # 덩어리로 (임시 [n,27] 메모리)
            w_ = _w27(fx[c0:c0 + 262144])
            i_ = _nid(pb[c0:c0 + 262144], nb1, nb2).clamp(0, nn - 1)
            wpin[c0:c0 + 262144] = (w_ * pin[i_].float()).sum(-1)
        x = s.x[:N].float()
        v = s.v[:N].float()
        m = s.m[:N].float()
        ok = torch.isfinite(x).all(-1) & torch.isfinite(v).all(-1)
        vcm = (v[ok] * m[ok, None]).sum(0) / m[ok].sum()
        dv = (v - vcm).norm(dim=-1)
        hp = (wpin > 0.5) & ok
        c = s.cidx.long()[s.pcd[:N].long()]
        nc = int(IP[11])
        cm_all = s.Cacc.reshape(-1, 30)[:nc, 29].float()
        cmed = float(cm_all[cm_all > 0].median())
        cmr = cm_all[c.clamp(0, nc - 1)] / cmed
        rec = dict(f=fi + 1, nn=nn, mthr_ratio=mthr / med, npin=int(pin.sum()), nlt=int((has & (mn < mthr)).sum()),
                   n_wpin05=int(hp.sum()), n_wpin1=int(((wpin > 0.999) & ok).sum()), dv_max_wpin05=float(dv[hp].max()) if bool(hp.any()) else None,
                   dv_max_all=float(dv[ok].max()), dv_p999=float(torch.quantile(dv[ok][::7].double(), 0.999)), vcm=[float(q) for q in vcm])
        L_, G_ = getattr(s, "ufl", None), getattr(s, "ufg", None)
        K = int(getattr(s, "Kcur", getattr(s, "K", 1)))
        if L_ is not None and G_ is not None and L_.numel() >= K:
            rec["step"] = dict(K=K, fail=[int(q) for q in L_[:K].cpu()], g2=[float(q) for q in G_[:K].cpu()])
        # 고리 버퍼: 이 프레임 기록
        P0 = ST["pre"]
        ent = dict(f=fi + 1, x0=P0["x"], v0=P0["v"], lj0=P0["lj"], F0=P0["F"], pa0=P0["pa"], pb=pb.int(), fx=fx.clone(), nb=(nb1, nb2, nn), cmr=cmr.half(), wpin=wpin.half())
        if nn <= NODE_CAP:
            ulm = getattr(s, "ulm", None)
            ent.update(mn=(mn / med).clone(), pin=pin.clone(), vs=torch.where(has[:, None], s.mv.reshape(-1, 3)[:nn].float() / mn.clamp_min(1e-30)[:, None], torch.zeros(nn, 3, device=mn.device)),
                       vk=s.vk.reshape(-1, 3)[:nn].float().clone(), ga=(s.gaA.reshape(-1, 3)[:nn].float().clone() if getattr(s, "gaA", None) is not None else None),
                       U=s.U.reshape(-1, 3)[:nn].float().clone(), ulm=(ulm[:nn].clone() if ulm is not None and ulm.numel() >= nn else None))
        ST["ring"].append(ent)
        if len(ST["ring"]) > RING:
            ST["ring"].pop(0)
        # 이탈 (escape1 과 같은 정의)
        cell = float(s.A[:, 0].norm())
        xs = x[ok][::7].double()
        q = torch.quantile(xs, torch.tensor([0.001, 0.999], dtype=torch.float64, device=xs.device), dim=0).float()
        d = torch.clamp(torch.maximum(q[0] - x, x - q[1]), min=0).norm(dim=-1)
        d = torch.where(ok, d, torch.full_like(d, float("inf")))
        far = d > 4 * cell
        prev = ST["far"]
        new = far & (~prev if prev is not None and prev.numel() == far.numel() else torch.ones_like(far))
        ST["far"] = far
        rec.update(nfar=int(far.sum()), nnew=int(new.sum()), cell=cell)
        zf = float(os.environ.get("AF_E2_FLOORZ", "0.02"))                              # 수박 바닥 평면 z (--floor_plane 1,1,0.02,0,0,1)
        zz = x[:, 2]
        below = ok & (zz < zf - cell)                                                  # 바닥 아래 한 칸 넘게 (바닥을 뚫고 떨어진 입자)
        rec.update(zmin=float(zz[ok].min()), nbelow=int(below.sum()), nfar_below=int((below & far).sum()),
                   nfar_wpin05=int((far & (wpin > 0.5)).sum()))
        LOG.write(json.dumps(rec) + "\n")
        nnew = int(new.sum())
        if nnew == 0 or ST["nesc"] >= 40:                                              # 이탈 프레임 기록은 40 번까지
            LOG.flush()
            return
        ST["nesc"] += 1
        idx = torch.nonzero(new).squeeze(1)
        if idx.numel() > 64:
            idx = idx[torch.argsort(d[idx], descending=True)[:64]]
        mu, lam = float(s.prm[0]), float(s.prm[1])
        D = dict(f=np.int32(fi + 1), idx=idx.cpu().numpy(), d=d[idx].cpu().numpy(), cell=np.float32(cell), vcm=vcm.cpu().numpy())
        G = len(ST["ring"])
        frames = []
        for gi, e in enumerate(ST["ring"]):
            nxt = ST["ring"][gi + 1] if gi + 1 < G else None
            x1 = (nxt["x0"] if nxt is not None else s.x[:N])[idx].float()
            v1 = (nxt["v0"] if nxt is not None else s.v[:N])[idx].float()
            lj1 = (nxt["lj0"] if nxt is not None else s.ljp[:N])[idx].float()
            F1 = (nxt["F0"] if nxt is not None else s.F[:N])[idx]
            J0, s0, P0_, Pn0 = _snh(e["F0"][idx], mu, lam)
            J1, s1, P1_, Pn1 = _snh(F1, mu, lam)
            pre_ = f"g{gi}_"
            D.update({pre_ + "f": np.int32(e["f"]), pre_ + "x0": e["x0"][idx].cpu().numpy(), pre_ + "x1": x1.cpu().numpy(), pre_ + "v0": e["v0"][idx].cpu().numpy(), pre_ + "v1": v1.cpu().numpy(),
                      pre_ + "lj0": e["lj0"][idx].cpu().numpy(), pre_ + "lj1": lj1.cpu().numpy(), pre_ + "J0": J0.cpu().numpy(), pre_ + "J1": J1.cpu().numpy(),
                      pre_ + "s1": s1.cpu().numpy(), pre_ + "P1": P1_.cpu().numpy(), pre_ + "Pn0": Pn0.cpu().numpy(), pre_ + "Pn1": Pn1.cpu().numpy(),
                      pre_ + "cmr": e["cmr"][idx].float().cpu().numpy(), pre_ + "wpin": e["wpin"][idx].float().cpu().numpy(), })
            nid = _nid(e["pb"][idx].long(), e["nb"][0], e["nb"][1]).clamp(0, e["nb"][2] - 1)
            ww = _w27(e["fx"][idx])
            D[pre_ + "w"] = ww.cpu().numpy()
            fr = dict(f=e["f"], v0=float(e["v0"][idx].float().norm(dim=-1).median()), v1=float(v1.norm(dim=-1).median()), lj1_max=float(lj1.max()),
                      J1_min=float(J1.min()), J1_med=float(J1.median()), Pn1_med=float(Pn1.median()), Pn1_max=float(Pn1.max()), cmr_med=float(e["cmr"][idx].float().median()),
                      wpin_med=float(e["wpin"][idx].float().median()))
            if "mn" in e:
                pn = e["pin"][nid]
                vk = e["vk"][nid]                                                    # [n,27,3]
                vs = e["vs"][nid]
                g2p = (ww[..., None] * vk).sum(1)
                g2p_pin = (ww[..., None] * vk * pn[..., None].float()).sum(1)
                D.update({pre_ + "mn": e["mn"][nid].cpu().numpy(), pre_ + "pin": pn.cpu().numpy(), pre_ + "vs": vs.cpu().numpy(), pre_ + "vk": vk.cpu().numpy(),
                          pre_ + "U": e["U"][nid].cpu().numpy()})
                if e.get("ga") is not None:
                    D[pre_ + "ga"] = e["ga"][nid].cpu().numpy()
                if e.get("ulm") is not None:
                    D[pre_ + "ulm"] = e["ulm"][nid].cpu().numpy()
                    fr["ulm_w"] = float((ww * (e["ulm"][nid] != 0).float()).sum(1).median())
                spd_s, spd_e = vs.norm(dim=-1), vk.norm(dim=-1)
                big = ww > 1e-3
                fr.update(g2p_err=float((g2p - v1).norm(dim=-1).max()), vpin_share=float((g2p_pin.norm(dim=-1) / g2p.norm(dim=-1).clamp_min(1e-9)).median()),
                          vn_end_max_pin=float(torch.where(big & pn, spd_e, torch.zeros_like(spd_e)).max(1).values.median()),
                          vn_end_max_sol=float(torch.where(big & ~pn, spd_e, torch.zeros_like(spd_e)).max(1).values.median()),
                          vn_start_max=float(torch.where(big, spd_s, torch.zeros_like(spd_s)).max(1).values.median()),
                          dvn_max_sol=float(torch.where(big & ~pn, (vk - vs).norm(dim=-1), torch.zeros_like(spd_e)).max(1).values.median()),
                          dvn_max_pin=float(torch.where(big & pn, (vk - vs).norm(dim=-1), torch.zeros_like(spd_e)).max(1).values.median()),
                          mn_min_med=float(torch.where(big, e["mn"][nid], torch.full_like(ww, float("inf"))).min(1).values.median()))
            frames.append(fr)
        np.savez_compressed(os.path.join(out, f"e2_f{fi + 1:03d}.npz"), **D)
        LOG.write(json.dumps(dict(esc=fi + 1, n=int(idx.numel()), d_med=float(d[idx].median()), frames=frames)) + "\n")
        LOG.flush()

    def outer(fn):
        def w(s, t, h, *a, **kw):
            if ST["dep"] == 0:
                try:
                    pre(s)
                except Exception:
                    import traceback
                    with open(os.path.join(out, "e2_err.txt"), "a") as f_:
                        f_.write("pre\n" + traceback.format_exc()[-2000:] + "\n")
            ST["dep"] += 1
            try:
                res = fn(s, t, h, *a, **kw)
            finally:
                ST["dep"] -= 1
            if ST["dep"] == 0 and ST["pre"] is not None:
                try:
                    post(s, int(round(t / h)))
                except Exception:
                    import traceback
                    with open(os.path.join(out, "e2_err.txt"), "a") as f_:
                        f_.write(traceback.format_exc()[-2000:] + "\n")
            return res
        return w
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    print("[절 CM43 진단 D2] 프레임마다 고정 노드 몫·이탈 입자 앞 프레임 기록 -> e2.jsonl·e2_f*.npz (결과 불변)", flush=True)

sys.argv = [os.path.join(HERE, "escape1_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")
