"""절 CM49 묶음 escape4 감싸개: 표본점 기준 높이 이어 받기 R·상대 장벽 A (lib/anchorflow/fem_escape4.py) 와 진단 D5 를 붙인 뒤 exe/escape3_wrap.py 로 넘김
(그쪽 --e3w W̄·AF_E3_DG, escape2 의 --e2*·AF_E2_DG, escape1·speed3 인자는 그대로 넘어감).

사용: python exe/escape4_wrap.py [--e5r 1] [--e5a 1] <escape3_wrap 인자 (--e3w 1 필수)...> <실행기.py> <실행기 인자...>
  --e5r 1  R: 걸음 끝 장벽 기준 높이를 입자에 이어 받아 다음 프레임 표본점 시작 기준 높이로 (정의 변경)
  --e5a 1  A: 걸음 시작 기준 높이 ≤ 0 인 표본점을 빼지 않고 시작 위치를 원점으로 한 상대 장벽 (기준 높이 d̂ 로)
  환경 AF_E5_DG=1 (옛 실행기만, 결과 불변, 읽기만): 프레임마다 <out>/e5.jsonl --
    이 프레임 팔분 칸 q 의 걸음 시작 높이 (실제 질량 평균 d_q, 기준 d̃_q = d_q + δ_q) 가 ≤ 0 인 표본점을 앞 프레임 소속으로 가름:
      이어짐 = 칸 입자 중 앞 프레임에 빠진 (비활성) 표본점 소속이 있음, 끝 어긋남 = 앞 표본점은 활성인데 그 끝 높이 (실제 · R 이면 기준) 가 ≤ 0,
      다시 모음 = 모든 입자가 앞 프레임 활성·끝 높이 > 0 인 표본점에서 옴 (보장된 평균에서 새 칸으로 갈라져 생김).
    끝 어긋남 직접 확인 (K 1 판만): 장벽이 본 걸음 끝 높이 (x̄ + Σ W̄ thinv y, U0 = 0) 대 실제 G2P 끝 질량 평균 (+ δ), 활성 점 중 장벽 > 0 · 실제 ≤ 0 수.
    바닥 밑 입자 수·z 최소, 바닥과 팔분 경계 어긋남 (칸 단위), R 이면 장치 기준 높이와 파이썬 재계산 차, δ 통계 (질량 가중 합 · 최대 |δ|), 커널 셈 (d̃ ≤ 0 · A 로 옮김).
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
while argv and argv[0].startswith("--e5"):                                              # 감싸개 인자 (맨 앞에만)
    CAND[argv[0][2:]] = argv[1]
    del argv[:2]
R_ = int(float(CAND.get("e5r", 0)))
A_ = int(float(CAND.get("e5a", 0)))
W_ = int(float(argv[argv.index("--e3w") + 1])) if "--e3w" in argv else 0
if R_ or A_:
    assert W_ == 1, "--e5r·--e5a 는 W̄ (--e3w 1) 위에서만"
    from anchorflow import fem_escape4 as E5
    E5.install(r=R_, a=A_)
    print(f"[절 CM49] R {R_} · A {A_} (W̄ {W_})", flush=True)

if os.environ.get("AF_E5_DG", "0") == "1":
    import torch
    from anchorflow import fem_ao_rt as AORT
    from anchorflow import fem_wmrx as WRX

    LOG = open(os.path.join(out, "e5.jsonl"), "w")
    D = dict(dep=0, x0=None, prev=None)

    def _w27(fx):
        w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], dim=-1)
        o = torch.arange(27, device=fx.device)
        i, j, k = o // 9, (o // 3) % 3, o % 3
        return w[:, 0, i] * w[:, 1, j] * w[:, 2, k]

    def _nid(b, nb1, nb2):
        o = torch.arange(27, device=b.device)
        i, j, k = o // 9, (o // 3) % 3, o % 3
        return ((b[:, 0:1] + i) * nb1 + (b[:, 1:2] + j)) * nb2 + (b[:, 2:3] + k)

    @torch.no_grad()
    def post(s, fi):
        IP = s.IP.cpu()
        if int(IP[13]):
            LOG.write(json.dumps(dict(f=fi + 1, overflow=1)) + "\n"); LOG.flush()
            D["prev"] = None
            return
        W = getattr(s, "wbar", None) or s.floor
        nr = torch.tensor((s.floor or {}).get("NR", [0.0, 0.0, 1.0]), device=s.x.device, dtype=torch.float64)
        pn, dh = float(W["PN"]), float(W["DH"])
        nb1, nb2, nn, nq = int(IP[7]), int(IP[8]), int(IP[9]), int(IP[12])
        N = int(getattr(s, "N", s.x.shape[0]))
        x = s.x[:N].double()
        x0 = D["x0"].double()
        m = s.m[:N].double()
        fx = s.fx.reshape(-1, 3)[:N].float()
        c = s.cidx.long()[s.pcd[:N].long()]
        oc = (fx[:, 0] >= 1).long() * 4 + (fx[:, 1] >= 1).long() * 2 + (fx[:, 2] >= 1).long()
        q = c * 8 + oc
        ok = torch.isfinite(x).all(-1) & torch.isfinite(x0).all(-1)
        Z = lambda: torch.zeros(nq, device=x.device, dtype=torch.float64)
        M = Z().index_add_(0, q, m)
        hp0 = x0 @ nr - pn                                                              # 입자 시작 높이
        hp1 = x @ nr - pn                                                               # 입자 끝 높이
        da0 = Z().index_add_(0, q, m * hp0.nan_to_num(0)) / M.clamp_min(1e-30)          # 실제 질량 평균 시작 높이
        da1 = Z().index_add_(0, q, m * hp1.nan_to_num(0)) / M.clamp_min(1e-30)
        has = M > 0
        dq = getattr(s, "e5DQ", None)
        dl = dq[:nq].double() if dq is not None else Z()                                # δ_q (R·A 판), 없으면 0
        bx = s.bx.reshape(-1, 4)[:nq].double()
        dref_dev = (bx[:, :3] @ nr) / bx[:, 3].clamp_min(1e-30) - pn                     # 장치 표본점 기준 높이 (옮긴 뒤)
        bact = s.bact[:nq].long() if getattr(s, "bact", None) is not None else torch.ones(nq, device=x.device, dtype=torch.long)
        rec = dict(f=fi + 1, nq=int(has.sum()), K=int(getattr(s, "K", 1)), ncall=int(D.get("ncall", 1)))
        rec["ref_dev_err"] = float(((da0 + dl) - dref_dev).abs()[has].max()) if bool(has.any()) else None
        # 파이썬으로 다시 잰 이어 받은 시작 기준 (앞 프레임 끝 h_p 질량 평균) -- R 이 없는 판에서도 '이었다면' 값
        pv = D["prev"]
        dcar = None
        if pv is not None and pv["hr"].shape[0] == N:
            dcar = Z().index_add_(0, q, m * pv["hr"].nan_to_num(0)) / M.clamp_min(1e-30)
            if dq is not None and getattr(s, "e5hr", None) is not None:
                lifted = has & ~(dcar > 0)
                rec["car_dev_err"] = float(((dcar - da0) - dl).abs()[has & ~lifted].max()) if bool((has & ~lifted).any()) else None
        # 가르기: 시작 높이 ≤ 0 인 표본점 (실제 d_q; R 판이면 기준 d̃ 옮기기 전 = 이어 받은 값)
        cats = {}
        for nm, d0 in (("act", da0), ("car", dcar)):
            if d0 is None:
                continue
            V = has & ~(d0 > 0)
            rec[f"nv_{nm}"] = int(V.sum())
            if pv is None or pv["qa"].shape[0] != N or not bool(V.any()):
                continue
            inV = V[q] & ok
            pin = (pv["qa"] == 0)                                                       # 앞 표본점 비활성
            pmis = (pv["qa"] == 1) & ~(pv["de"] > 0)                                    # 앞 표본점 활성, 끝 높이 (실제 + δ) ≤ 0
            Zi = lambda: torch.zeros(nq, device=x.device, dtype=torch.long)
            a_in = Zi().index_add_(0, q[inV], pin[inV].long())
            a_mis = Zi().index_add_(0, q[inV], pmis[inV].long())
            ci = V & (a_in > 0)
            cm = V & ~(a_in > 0) & (a_mis > 0)
            cr = V & ~(a_in > 0) & ~(a_mis > 0)
            cats[nm] = dict(inherit=int(ci.sum()), mismatch=int(cm.sum()), regroup=int(cr.sum()))
            if nm == "act" and bool(cr.any()):
                qi = torch.nonzero(cr).squeeze(1)[:6]
                ex = []
                for qq in qi.tolist():                                                 # 다시 모음 예: 새 칸 입자들이 온 앞 표본점 수·그 끝 높이 범위·입자 높이 범위
                    mem = torch.nonzero((q == qq) & ok).squeeze(1)
                    oq = pv["qid"][mem]
                    ex.append(dict(q=qq, d0=round(float(da0[qq]), 5), np=int(mem.numel()), nold=int(torch.unique(oq).numel()), dold=[round(float(pv["de"][mem].min()), 5), round(float(pv["de"][mem].max()), 5)],
                                   hp=[round(float(hp0[mem].min()), 5), round(float(hp0[mem].max()), 5)], m=round(float(M[qq] / M[has].median()), 4)))
                rec["regroup_ex"] = ex
        rec["cat"] = cats
        # 끝 어긋남 직접 (K 1, 장벽이 본 끝 = x̄_ref + Σ W̄ (U0 + thinv (y − U0)), U0 = 0)
        wq = getattr(s, "e3wq", None)
        if wq is not None and int(getattr(s, "K", 1)) == 1 and getattr(s, "ga_c", None) is not None and int(D.get("ncall", 1)) == 1:   # 여러 번 부른 프레임 (첫 접촉 다시 풀기) 은 y 가 마지막 시도 것이라 뺌
            th = 1.0 / (1.0 - float(s.ga_c["af"]))
            cb = s.cb.reshape(-1, 3).long()
            qi = torch.arange(nq, device=x.device)
            ids = _nid(cb[qi // 8], nb1, nb2).clamp(0, nn - 1)
            yn = (s.y.reshape(-1, 3)[:nn].double() @ nr)
            Wq = wq.reshape(27, -1)[:, :nq].t().double()
            dbar = dref_dev + (Wq * (th * yn[ids])).sum(1)
            dend = da1 + dl
            A1 = has & (bact == 1)
            rec.update(bar_end_max=float((dbar - dend).abs()[A1].max()) if bool(A1.any()) else None,
                       bar_pos_act_le0=int((A1 & (dbar > 0) & ~(dend > 0)).sum()), act_end_le0=int((A1 & ~(dend > 0)).sum()))
        rec.update(nbelow=int((ok & (hp1 < 0)).sum()), zmin=float(hp1[ok].min()) if bool(ok.any()) else None, inact=int((has & (bact == 0)).sum()))
        bz = -hp1[ok & (hp1 < 0)]
        if bz.numel():
            qs = torch.quantile(bz[:: max(1, bz.numel() // 100000)].double(), torch.tensor([0.5, 0.9, 0.99], device=bz.device, dtype=torch.float64))
            rec["below_q"] = [round(float(v), 5) for v in qs]                           # 바닥 밑 입자 깊이 중앙 · 90 · 99 백분위 m
        if dq is not None and bool(has.any()):
            nearf = has & (da0 < 0.05)                                                  # 바닥 5 cm 안 표본점의 δ 분포 (진단 집계 범위만)
            if bool(nearf.any()):
                qs = torch.quantile(dl[nearf], torch.tensor([0.01, 0.5, 0.99], device=dl.device, dtype=torch.float64))
                rec["dnear_q"] = [round(float(v), 5) for v in qs]
        # 고정 노드 (pm 2 = m_n < mthr, 관성 예측에 묶임) 몫: 비활성 표본점 (W̄ 무게) · 깊은 입자 (바닥 밑 한 칸 넘음, 입자 B-스플라인 무게)
        pmv = getattr(s, "pm", None)
        if pmv is not None:
            fixn = (pmv[:nn] == 2).double()
            inq = torch.nonzero(has & (bact == 0)).squeeze(1)
            if inq.numel() and wq is not None:
                cb_ = s.cb.reshape(-1, 3).long()
                ids_ = _nid(cb_[inq // 8], nb1, nb2).clamp(0, nn - 1)
                Wi = wq.reshape(27, -1)[:, :nq].t().double()[inq]
                rec["inact_fix"] = [round(float(v), 3) for v in (Wi * fixn[ids_]).sum(1)[:8]]
                rec["inact_d0"] = [round(float(v), 4) for v in da0[inq][:8]]
                rec["inact_m"] = [round(float(v), 5) for v in (M[inq] / M[has].median())[:8]]
            deep = torch.nonzero(ok & (hp1 < -0.02)).squeeze(1)
            if deep.numel():
                pb_ = s.Np.reshape(-1, 3)[:N].long()[deep]
                w_ = _w27(fx[deep])
                fd = (w_.double() * fixn[_nid(pb_, nb1, nb2).clamp(0, nn - 1)]).sum(1)
                rec.update(ndeep=int(deep.numel()), deep_fix_med=round(float(fd.median()), 3), deep_fix_min=round(float(fd.min()), 3))
        if dq is not None:
            rec.update(dsum=float((M * dl)[has].sum() / M[has].sum()), dmax=float(dl[has].abs().max()), dneg=float(dl[has].min()), dpos=float(dl[has].max()))
            cn = getattr(s, "e5cnt", None)
            if cn is not None:
                rec["kcnt"] = [int(v) for v in cn.cpu()]
        # 바닥과 팔분 경계 (칸 단위 어긋남: 바닥 높이에서 가장 가까운 팔분 z 경계까지, 0 = 겹침)
        A = s.A.reshape(3, 3).double()
        h = float(A[2, 2].abs())
        if h > 0:
            zs = x0[:, 2] - (fx[:, 2].double() - 1.0) * h                               # 그 입자 칸의 팔분 z 경계 (세계)
            zs = zs[ok][:1000]
            zf = pn / float(nr[2]) if float(nr[2]) != 0 else float("nan")
            off = ((zf - zs) / (0.5 * h)) % 1.0
            rec["floor_oct_off"] = round(float(off.median()), 4)
        # 다음 프레임에 쓸 입자별 앞 표본점 정보
        de = da1 + dl
        hr = de[q]
        D["prev"] = dict(qa=bact[q].clone(), de=de[q].clone(), hr=hr.clone(), qid=q.clone())
        LOG.write(json.dumps(rec) + "\n")
        LOG.flush()

    def outer(fn):                                                                     # 프레임 첫 맨바깥 호출에서 시작 위치 (첫 접촉 다시 풀기·X3c 는 같은 프레임을 여러 번 부름)
        def w(s, t, h, *a, **kw):
            fi = int(round(t / h))
            if D["dep"] == 0:
                if D.get("fi") != fi:
                    D["fi"], D["ncall"] = fi, 0
                    D["x0"] = s.x[:int(getattr(s, "N", s.x.shape[0]))].float().clone()
                D["ncall"] += 1
            D["dep"] += 1
            try:
                return fn(s, t, h, *a, **kw)
            finally:
                D["dep"] -= 1
        return w
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    from anchorflow import fem_an_rt_v0 as V0
    _oc = V0.RTSolver.counts

    def counts(s):                                                                     # 프레임 끝 (모든 시도·되돌림 뒤, 실행기가 프레임마다 한 번 부름)
        r = _oc(s)
        if D.get("x0") is not None and D.get("fi") is not None and D.get("done") != D["fi"]:
            D["done"] = D["fi"]
            try:
                post(s, D["fi"])
            except Exception:
                import traceback
                with open(os.path.join(out, "e5_err.txt"), "a") as f_:
                    f_.write(traceback.format_exc()[-2000:] + "\n")
        return r
    V0.RTSolver.counts = counts
    print("[절 CM49 진단 D5] 걸음 시작 바닥 밑 표본점 가르기 (이어짐·끝 어긋남·다시 모음) · 끝 어긋남 직접 · δ -> e5.jsonl (결과 불변)", flush=True)

sys.argv = [os.path.join(HERE, "escape3_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")
