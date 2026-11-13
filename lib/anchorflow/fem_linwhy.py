"""절 CM127 묶음 linwhy (조정자 11-13, 사용자 wolf 완전 위임; 진단 -- 고른 프레임만 그래프 대신 같은 커널 순서를 바로 실행, 분석 뒤 버퍼 되돌림):
wolf 뉴턴 선형계 (확장 접은 Ã δ = r, MG V 순환 전처리 M) 를 실제 판 한가운데서 꺼내 잰다.

고른 프레임 (env LW_F = '8,20,30', 1 부터) 의 첫 걸음 (kstep 0), 고른 뉴턴 반복 (LW_IT = '0,10,40') 의 PCG 첫 반복에서
  (헤시안 곱 rt_ap 를 낸 뒤, rt_cg1m 직전) 멈춰:
  - 연산자 Aop(x) = 실행기가 그 반복에 낸 커널 그대로 (rt_pinit (+ 확장 펼치기 훅) -> KTp 0 -> 요소 곱 (+ 벌점·장벽 훅) -> rt_ap (+ 확장 접기 훅)).
    가장 바깥 R.k 에서 기록한 (이름, 인자) 를 s.k 로 다시 내므로 안쪽 감싸개 훅이 모두 같이 돎.
  - 전처리 Mop(r) = s.mg_vc(0, r, z) (실행기 V 순환 그대로, r 은 줄어든 자유도 밖 0).
  - 줄어든 자유도 = 활성 (m > 0) 이고 확장 외부 노드 (w8b ≥ 0) 가 아닌 노드의 3 성분.
  잼: 대칭 (u·Av − v·Au), 랑초시 (완전 재직교, f64) 로 Ã·M·확장 없는 A 의 극값, PCG (보통 β) 의 잔차 이력과 α·β 로 M⁻¹Ã 리츠 값,
      블록 야코비 PCG·전처리 없는 CG·FGMRES(M) 이력, 층별 평활 ω 와 거듭제곱 60 번 λmax (ω λmax > 2 면 그 층 평활 발산),
      새 ω (거듭제곱 60) 로 M 다시 -> 같은 측정, 노드별 잔차 위치 (실제 PCG 끝 r: 확장 주인 노드·장벽·꼭짓점 몫·높이).
  선택 (LW_ASM = 프레임, LW_R = 결합 반경 8): 색칠 탐침으로 Ã 를 희소 행렬로 조립 -> 조립 = 연산자 확인, ‖A − Aᵀ‖ 와 가장 큰 비대칭 성분 위치, 덤프.
출력: <out>/lw/fFFF_iII.npz (노드별 배열, 리츠 값, 이력) · <out>/lw.jsonl (요약 줄) · <out>/lwinfo.json.
훅: 실행기 runpy 순간 (가장 바깥). 그 밖 프레임은 그래프 재생 그대로.
"""
import json
import math
import os
import time

import numpy as np
import torch

ST = dict(out=".", F=[], IT=[], nana=0, err=None, frames_done=[], LZ=300, NPCG=300, ASM=-1, R=8)
LOG = [None]
f64 = torch.float64


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _dump():
    try:
        with open(os.path.join(ST["out"], "lwinfo.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def lanczos(op, n, k, dev, seed=0, v0=None, vec=0):
    """대칭 연산자 op (f64 n 벡터) 의 랑초시 k 단계 (완전 재직교 두 번). 돌려줌: 리츠 값 (오름), 단계 수."""
    g = torch.Generator(device=dev).manual_seed(seed)
    v = torch.randn(n, dtype=f64, device=dev, generator=g) if v0 is None else v0.clone().to(f64)
    v /= v.norm()
    V = torch.zeros(k + 1, n, dtype=f64, device=dev)
    V[0] = v
    al, be = [], []
    for j in range(k):
        w = op(V[j])
        a = float(torch.dot(w, V[j]))
        al.append(a)
        for _ in range(2):
            w -= V[:j + 1].T @ (V[:j + 1] @ w)
        b = float(w.norm())
        if not math.isfinite(b) or b < 1e-10 * max(abs(a), 1e-300):
            break
        be.append(b)
        V[j + 1] = w / b
    m = len(al)
    T = np.diag(al) + np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1)
    if vec:
        ev, Y = np.linalg.eigh(T)
        Yt = torch.from_numpy(Y[:, :vec]).to(dev)
        X = (V[:m].T @ Yt).T.contiguous()
        del V
        return ev, m, X
    ev = np.linalg.eigvalsh(T)
    del V
    return ev, m


def pcg(Aop, Mop, b, iters, rec_lz=True, snaps=(1, 2, 5, 10, 20, 50, 100, 200)):
    """보통 PCG (Fletcher-Reeves β = r·z 비). 잔차 이력 |r_i|/|b| 와 pAp ≤ 0 · rz ≤ 0 수, (α, β) 로 만든 M⁻¹A 리츠 값."""
    x = torch.zeros_like(b)
    xs = {}
    r = b.clone()
    z = Mop(r)
    p = z.clone()
    rz = float(torch.dot(r, z))
    nb = float(b.norm())
    hist = [1.0]
    phi = [0.0]
    al, be = [], []
    npap = nrz = 0
    rz0 = rz
    for i in range(iters):
        Ap = Aop(p)
        pAp = float(torch.dot(p, Ap))
        if not (pAp > 0):
            npap += 1
            break
        a = rz / pAp
        x += a * p
        r -= a * Ap
        hist.append(float(r.norm()) / nb)
        phi.append(-0.5 * float(torch.dot(x, b + r)))
        if (i + 1) in snaps:
            xs[i + 1] = x.float().cpu().numpy()
        z = Mop(r)
        rzn = float(torch.dot(r, z))
        if not (rzn > 0):
            nrz += 1
            al.append(a)
            break
        bt = rzn / rz
        al.append(a); be.append(bt)
        p = z + bt * p
        rz = rzn
    ritz = None
    if rec_lz and len(al) >= 2:
        m = len(al)
        d = np.zeros(m); e = np.zeros(m - 1)
        for j in range(m):
            d[j] = 1.0 / al[j] + (be[j - 1] / al[j - 1] if j > 0 else 0.0)
            if j < m - 1 and j < len(be):
                e[j] = math.sqrt(max(be[j], 0.0)) / al[j]
        T = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        ritz = np.linalg.eigvalsh(T)
    return dict(hist=np.array(hist), phi=np.array(phi), xs=xs, npap=npap, nrz=nrz, ritz=ritz, rz0=rz0, x=x)


def fgmres(Aop, Mop, b, m):
    """오른쪽 전처리 FGMRES(m) (M 대칭·정부호 가정 없음). 잔차 이력 |r_i|/|b|."""
    nb = float(b.norm())
    V = [b / nb]
    Z = []
    H = np.zeros((m + 1, m))
    hist = [1.0]
    for j in range(m):
        z = Mop(V[j]); Z.append(z)
        w = Aop(z)
        for i in range(j + 1):
            H[i, j] = float(torch.dot(w, V[i])); w = w - H[i, j] * V[i]
        for i in range(j + 1):
            c = float(torch.dot(w, V[i])); H[i, j] += c; w = w - c * V[i]
        H[j + 1, j] = float(w.norm())
        e1 = np.zeros(j + 2); e1[0] = nb
        y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)
        hist.append(float(np.linalg.norm(H[:j + 2, :j + 1] @ y - e1)) / nb)
        if not (H[j + 1, j] > 1e-30 * nb):
            break
        V.append(w / H[j + 1, j])
    return np.array(hist)


def install(out):
    from . import fem_an_rt as ART
    F = [int(q) for q in os.environ.get("LW_F", "20").split(",") if q.strip()]
    ITS = [int(q) for q in os.environ.get("LW_IT", "0").split(",") if q.strip()]
    ST.update(out=out, F=F, IT=ITS, LZ=int(os.environ.get("LW_LZ", "300")), NPCG=int(os.environ.get("LW_NPCG", "300")),
              ASM=int(os.environ.get("LW_ASM", "-1")), R=int(os.environ.get("LW_R", "8")), NGM=int(os.environ.get("LW_NGM", "100")))
    os.makedirs(os.path.join(out, "lw"), exist_ok=True)
    LOG[0] = open(os.path.join(out, "lw.jsonl"), "a")
    R = ART.RTSolver
    ok, octl, ost = R.k, R.ctl, R.step

    class Eager:
        def __init__(s_, sv, real):
            s_.sv, s_.real = sv, real

        def replay(s_):
            sv = s_.sv
            sv._lw_on = True
            try:
                sv.record(0.0, sv.Kcur)
            finally:
                sv._lw_on = False

    def step(s, t):
        fr = int(round(t / s.h)) + 1
        if fr not in ST["F"] or fr in ST["frames_done"] or (s.ga and getattr(s, "ga_first", False)):
            return ost(s, t)
        ST["frames_done"].append(fr)
        s._lw_fr = fr
        Kw = s.Kc if (s.Kc and getattr(s, "contact_prev", False)) else s.K
        if Kw != s.Kcur:
            s.Kcur = Kw
            s.graph = s.graphs.get(Kw)
        if s.graph is None:
            bk = s._bk(); s.capture(); s._rs(bk)
        real = s.graph
        s.graph = Eager(s, real)
        try:
            return ost(s, t)
        finally:
            if isinstance(s.graph, Eager):
                s.graph = real
            print(f"[절 CM127] linwhy f{fr} 분석 끝 ({ST['nana']} 개 누적)", flush=True)
            _dump()

    def ctl(s, op, it=0):
        if getattr(s, "_lw_on", False):
            dbg = ST.setdefault("dbg", {})
            kk = f"ctl{op}"
            dbg[kk] = dbg.get(kk, 0) + 1
            if op == 5 and it == 0:
                dbg.setdefault("c5", []).append([getattr(s, "_lw_k", None), getattr(s, "_lw_it", None)]) if len(dbg.get("c5", [])) < 20 else None
        if getattr(s, "_lw_on", False) and not getattr(s, "_lw_in", False):
            if op == 0:
                s._lw_k = -1
            elif op == 1:
                s._lw_k = getattr(s, "_lw_k", -1) + 1
            elif op == 2:
                s._lw_it = it
                s._lw_rec = None
                s._lw_pin = None
                s._lw_hist = []
            elif op == 7 and getattr(s, "_lw_pend", None) is not None:
                r = octl(s, op, it)
                try:
                    finish(s)
                except Exception:
                    import traceback
                    ST["err"] = "finish " + traceback.format_exc()[-1500:]
                    _dump()
                return r
        r = octl(s, op, it)
        if getattr(s, "_lw_on", False) and not getattr(s, "_lw_in", False) and op == 5 and it == 0 \
                and getattr(s, "_lw_k", -1) == 0 and getattr(s, "_lw_it", -1) in ST["IT"]:
            s._lw_rec = []
        return r

    def k(s, name, n, args, b=128):
        if not getattr(s, "_lw_on", False) or getattr(s, "_lw_in", False):
            return ok(s, name, n, args, b)
        dbg = ST.setdefault("dbgk", {})
        if len(dbg) < 200 or name in dbg:
            dbg[name] = dbg.get(name, 0) + 1
        if name == "rt_pinit":
            s._lw_pin = (name, n, tuple(args), b)
        if name == "rt_sc_roll" and getattr(s, "_lw_pend", None) is not None:
            torch.cuda.synchronize()
            s._lw_hist.append(float(s.SC[57]))
        rec = getattr(s, "_lw_rec", None)
        if rec is not None and name in ("rt_cg1m", "rt_cg1"):
            s._lw_rec = None
            ct = s.CT.cpu().tolist()
            ST.setdefault("ct_at", []).append(ct[:8])
            if not (ct[0] or ct[1]):
                try:
                    analyze(s, rec)
                except Exception:
                    import traceback
                    ST["err"] = "analyze " + traceback.format_exc()[-2000:]
                    print("[절 CM127] 분석 오류", ST["err"], flush=True)
                    _dump()
        elif rec is not None and name != "rt_ctl":
            rec.append((name, n, tuple(args), b))
        return ok(s, name, n, args, b)

    R.k, R.ctl, R.step = k, ctl, step

    # ------------------------------------------------------------------ 분석
    def analyze(s, seg):
        t0 = time.time()
        torch.cuda.synchronize()
        s._lw_in = True
        dev = s.dev
        capN = int(s.capN)
        IP = s.IP.cpu().numpy().copy()
        nn = int(IP[9])
        names = [q[0] for q in seg]
        # 되돌릴 버퍼
        keep = {}
        for nm in ("r", "z", "zo", "p", "Tp", "KTp", "Ap", "dl", "SC", "CT", "CD", "W", "m0u", "m0t", "m0KT", "w8acc"):
            t_ = getattr(s, nm, None)
            if torch.is_tensor(t_):
                keep[nm] = t_.clone()
                if nm in ("r", "z", "zo", "p", "Tp", "KTp", "Ap", "dl", "m0u", "m0t", "m0KT", "w8acc"):
                    keep[nm] = keep[nm].view(-1)
        mlv = [{kk: v.clone() for kk, v in L.items() if torch.is_tensor(v) and kk in ("z", "y", "KT", "t", "r")} for L in getattr(s, "ML", [])]
        mg_pow0 = s.mg_pow
        try:
            res = _analyze(s, seg, keep, IP, nn, capN, dev)
        finally:
            for nm, t_ in keep.items():
                getattr(s, nm).view(-1).copy_(t_.view(-1))
            for L, d in zip(getattr(s, "ML", []), mlv):
                for kk, v in d.items():
                    L[kk].copy_(v)
            s.mg_pow = mg_pow0
            torch.cuda.synchronize()
            s._lw_in = False
        res["names"] = names
        res["sec"] = time.time() - t0
        s._lw_pend = res
        ST["nana"] += 1
        print(f"[절 CM127] f{s._lw_fr} it{s._lw_it} 분석 {res['sec']:.0f} s: λÃ [{res['lzA'][0]:.3e}, {res['lzA'][1]:.3e}] · λM [{res['lzM'][0]:.3e}, {res['lzM'][1]:.3e}] · "
              f"대칭 A {res['symA']:.1e} M {res['symM']:.1e} · PCG100 {res['pcg_M'][min(100, len(res['pcg_M']) - 1)]:.2e}", flush=True)

    def _analyze(s, seg, keep, IP, nn, capN, dev):
        i32 = np.int32
        pin = s._lw_pin
        assert pin is not None, "rt_pinit 인자 없음"
        mn = s.mn.view(-1)[:capN].float()
        w8b = getattr(s, "w8b", None)
        act = torch.zeros(capN, dtype=torch.bool, device=dev)
        act[:nn] = mn[:nn] > 0
        outer = (w8b[:capN] >= 0) & act if w8b is not None else torch.zeros_like(act)
        inner = act & ~outer
        idx = torch.nonzero(inner.repeat_interleave(3)).squeeze(1)
        idx_all = torch.nonzero(act.repeat_interleave(3)).squeeze(1)
        nd = int(idx.numel())
        X = torch.zeros(capN * 3, dtype=torch.float32, device=dev)
        rin = torch.zeros(capN * 3, dtype=torch.float32, device=dev)
        zout = torch.zeros(capN * 3, dtype=torch.float32, device=dev)
        nmv = [0, 0]

        def Afull(xf):
            s.z.view(-1)[:capN * 3].copy_(xf)
            s.k(*pin)
            for (nm, n_, a_, b_) in seg:
                s.k(nm, n_, a_, b_)
            nmv[0] += 1
            return s.Ap.view(-1)[:capN * 3]

        def Aop(x):
            X.zero_(); X[idx] = x.float()
            y = Afull(X)
            return y[idx].to(f64)

        def Mfull(rf):
            s.mg_vc(0, rf, zout)
            nmv[1] += 1
            return zout

        def Mop(x):
            rin.zero_(); rin[idx] = x.float()
            return Mfull(rin)[idx].to(f64)

        Dv = s.Dinv.view(-1)[:capN * 9].view(capN, 3, 3)

        def Jop(x):
            rin.zero_(); rin[idx] = x.float()
            z = torch.bmm(Dv, rin.view(capN, 3, 1)).view(-1)
            return z[idx].to(f64)

        # 확장 없는 A (층 0 V 순환 연산자: Mi z + T K T z (+ 장벽 융합)) -- 활성 노드 전체
        nr = s.floor["NR"] if s.floor else [0.0, 0.0, 1.0]
        it2 = np.float32(1.0 / (getattr(s, 'mg_tau', s.tau) ** 2))
        KB = s.bKn if (getattr(s, "bar", False) and s.mgbar) else None
        Xa = torch.zeros(capN * 3, dtype=torch.float32, device=dev)
        resb = torch.zeros(capN * 3, dtype=torch.float32, device=dev)

        def A0op(x):
            Xa.zero_(); Xa[idx_all] = x.float()
            s.k("rt_mg_node", capN, (i32(0), P(s.LM), i32(3), P(s.meff), it2, P(s.sk), P(s.pm), *[np.float32(q) for q in nr], P(s.Dinv), P(s.W), P(None), P(None), P(Xa), P(s.m0u), P(None)), b=256)
            s.m0KT.zero_()
            s.mg_Az(0, Xa, s.m0KT, tz_ready=True)
            s.k("rt_mg_node", capN, (i32(0), P(s.LM), i32(2), P(s.meff), it2, P(s.sk), P(s.pm), *[np.float32(q) for q in nr], P(s.Dinv), P(s.W), P(None), P(s.m0KT), P(Xa), P(resb), P(KB)), b=256)
            return (-resb)[idx_all].to(f64)

        o = dict(f=s._lw_fr, it=s._lw_it, nn=nn, nact=int(act.sum()), ninner=int(inner.sum()), nouter=int(outer.sum()), nd=nd, nseg=len(seg))
        # 0. 연산자 확인: 기록 곱 = 실행기 Ap (p = z0)
        b = keep["r"][idx].to(f64)
        o["bnorm"] = float(b.norm())
        y0 = Afull(keep["z"].clone()).clone()
        dm = inner.repeat_interleave(3)
        o["chk_Ap"] = float((y0[idx] - keep["Ap"][idx]).norm() / keep["Ap"][idx].norm().clamp_min(1e-300))
        o["leakA"] = float(y0[~dm].abs().max() / y0[idx].abs().max().clamp_min(1e-300))
        zz = Mfull(keep["r"].clone()).clone()
        o["chk_Mz"] = float((zz[idx] - keep["z"][idx]).norm() / keep["z"][idx].norm().clamp_min(1e-300))
        # Mi (질량 대각) 범위
        Mi = (s.meff.view(-1)[:capN].float() * float(it2)).repeat_interleave(3)[idx]
        o["Mi"] = [float(Mi.min()), float(Mi.median()), float(Mi.max())]
        # 1. 대칭
        g = torch.Generator(device=dev).manual_seed(1)
        sa, sm = [], []
        for _ in range(3):
            u = torch.randn(nd, dtype=f64, device=dev, generator=g); v = torch.randn(nd, dtype=f64, device=dev, generator=g)
            Au, Av = Aop(u), Aop(v)
            sa.append(abs(float(u @ Av) - float(v @ Au)) / math.sqrt(abs(float(u @ Au)) * abs(float(v @ Av))))
            Mu, Mv = Mop(u), Mop(v)
            sm.append(abs(float(u @ Mv) - float(v @ Mu)) / math.sqrt(abs(float(u @ Mu)) * abs(float(v @ Mv)) + 1e-300))
        o["symA"], o["symM"] = max(sa), max(sm)
        # 2. 랑초시
        LZ = ST["LZ"]
        evA, mA = lanczos(Aop, nd, LZ, dev, 2)
        evM, mM = lanczos(Mop, nd, LZ, dev, 3)
        evA0, mA0 = lanczos(A0op, int(idx_all.numel()), LZ, dev, 4)
        o["lzA"] = [float(evA[0]), float(evA[-1])]; o["lzM"] = [float(evM[0]), float(evM[-1])]; o["lzA0"] = [float(evA0[0]), float(evA0[-1])]
        o["lzA_neg"] = int((evA < 0).sum()); o["lzM_neg"] = int((evM < 0).sum())
        o["lzA_belowMi"] = int((evA < 0.999 * o["Mi"][0]).sum())
        # 3. 실제 rhs 로 풀이 이력
        NP = ST["NPCG"]
        rM = pcg(Aop, Mop, b, NP)
        rJ = pcg(Aop, Jop, b, NP)
        rI = pcg(Aop, lambda x: x.clone(), b, min(NP, 200))
        o["pcg_M"] = rM["hist"].tolist(); o["pcg_J"] = rJ["hist"].tolist(); o["cg_I"] = rI["hist"].tolist()
        o["pcg_M_bad"] = [rM["npap"], rM["nrz"]]; o["pcg_J_bad"] = [rJ["npap"], rJ["nrz"]]
        o["ritz_M"] = [float(rM["ritz"][0]), float(rM["ritz"][-1])] if rM["ritz"] is not None else None
        o["ritz_J"] = [float(rJ["ritz"][0]), float(rJ["ritz"][-1])] if rJ["ritz"] is not None else None
        o["gm_M"] = fgmres(Aop, Mop, b, ST["NGM"]).tolist()
        # 4. 층별 ω: 지금 W 대 거듭제곱 60 번 (ω λmax)
        Wcur = s.W.clone()
        s.mg_pow = 60
        lam = []
        for l in range(s.nl):
            s.mg_power(l)
        torch.cuda.synchronize()
        Wf = s.W.clone()
        o["W_cur"] = Wcur.tolist(); o["W_fresh60"] = Wf.tolist()
        o["omega_lmax"] = [float(Wcur[l] / Wf[l]) * float(s.mg_omega) for l in range(s.nl)]      # = ω_cur λmax (ω = mg_omega 일 때 W = ω/λ)
        # 5. 새 ω 로 M -> 같은 측정 (W = 거듭제곱 60 값)
        evMf, _ = lanczos(Mop, nd, min(LZ, 150), dev, 5)
        rMf = pcg(Aop, Mop, b, NP)
        o["lzM_fresh"] = [float(evMf[0]), float(evMf[-1])]; o["pcg_Mf"] = rMf["hist"].tolist(); o["pcg_Mf_bad"] = [rMf["npap"], rMf["nrz"]]
        o["ritz_Mf"] = [float(rMf["ritz"][0]), float(rMf["ritz"][-1])] if rMf["ritz"] is not None else None
        o["gm_Mf"] = fgmres(Aop, Mop, b, ST["NGM"]).tolist()
        s.W.copy_(Wcur)
        # 6. 확장 반영 전처리 시험 (진단): Ã 블록 대각 D̃ (색칠 탐침 = 정확 · 근사 D_j + Σ_o c_oj² T D_o T) 야코비, 층 0 을 Ã·D̃ 로 바꾼 V 순환 (거친 층 그대로)
        try:
            ex = experiments(s, o, Afull, Aop, Mop, b, X, inner, outer, idx, capN, IP, dev, nr, it2, rM, rJ, rI, rMf)
            o.update(ex)
        except Exception:
            import traceback
            o["exp_err"] = traceback.format_exc()[-1500:]
            print("[절 CM127] 시험 오류", o["exp_err"], flush=True)
        o["nmv"] = list(nmv)
        # 노드 배열 (활성 노드만)
        nid = torch.nonzero(act).squeeze(1)
        arr = dict(nid=nid.cpu().numpy().astype(np.int32), IP=IP, A=s.A.float().cpu().numpy(), Ai=s.Ai.float().cpu().numpy(),
                   mn=mn[nid].cpu().numpy(), meff=s.meff.view(-1)[:capN][nid].float().cpu().numpy(), sk=s.sk.view(-1)[:capN][nid].cpu().numpy(), pm=s.pm.view(-1)[:capN][nid].cpu().numpy(),
                   w8b=(w8b[:capN][nid].cpu().numpy() if w8b is not None else np.zeros(len(nid), np.int32)),
                   bKn=(s.bKn.view(-1)[:capN][nid].float().cpu().numpy() if getattr(s, "bKn", None) is not None else np.zeros(len(nid), np.float32)),
                   r0=keep["r"][:capN * 3].view(capN, 3)[nid].cpu().numpy(), evA=evA, evM=evM, evA0=evA0, evMf=evMf,
                   ritzM=(rM["ritz"] if rM["ritz"] is not None else np.zeros(0)), ritzJ=(rJ["ritz"] if rJ["ritz"] is not None else np.zeros(0)),
                   ritzMf=(rMf["ritz"] if rMf["ritz"] is not None else np.zeros(0)),
                   xM=_scatter(rM["x"], idx, capN, nid), xMf=_scatter(rMf["x"], idx, capN, nid))
        # 꼭짓점 몫 (nlwhy 상태 버퍼가 있으면: 지금 반복 재질 평가의 DP 상태)
        st = getattr(s, "nlST", None)
        nq = int(IP[12])
        if st is not None and nq > 0:
            q = torch.arange(nq, device=dev)
            c = q >> 3
            cb = s.cb.view(-1, 3)[:nq // 8 + 1].long()
            nb1, nb2 = int(IP[7]), int(IP[8])
            Vq = s.Vq.view(-1)[:nq].float()
            sq = st[:nq]
            Va = torch.zeros(capN, device=dev, dtype=torch.float32); Vt = torch.zeros(capN, device=dev, dtype=torch.float32); Vl = torch.zeros(capN, device=dev, dtype=torch.float32)
            for o_ in range(27):
                i_, j_, k_ = o_ // 9, (o_ // 3) % 3, o_ % 3
                ids = ((cb[c, 0] + i_) * nb1 + (cb[c, 1] + j_)) * nb2 + (cb[c, 2] + k_)
                ok_ = (ids >= 0) & (ids < capN)
                Va.index_add_(0, ids[ok_], Vq[ok_]); Vt.index_add_(0, ids[ok_], (Vq * (sq == 2).float())[ok_]); Vl.index_add_(0, ids[ok_], (Vq * (sq == 1).float())[ok_])
            arr["fap"] = (Vt / Va.clamp_min(1e-30))[nid].cpu().numpy(); arr["ftail"] = (Vl / Va.clamp_min(1e-30))[nid].cpu().numpy(); arr["Vn"] = Va[nid].cpu().numpy()
            o["q_state"] = [int((sq == j).sum()) for j in (0, 1, 2)]
        # 확장 주인 노드 (외부 노드의 블록 노드) 표시
        if w8b is not None and int(outer.sum()) > 0:
            own = torch.zeros(capN, dtype=torch.int32, device=dev)
            on_ = torch.nonzero(outer).squeeze(1)
            b0 = w8b[on_].long()
            Wt = s.w8w.view(-1, 9)[on_]
            nb1, nb2 = int(IP[7]), int(IP[8])
            for a_ in range(3):
                for b_ in range(3):
                    for c_ in range(3):
                        ww = Wt[:, a_] * Wt[:, 3 + b_] * Wt[:, 6 + c_]
                        qn = b0 + (a_ * nb1 + b_) * nb2 + c_
                        m_ = ww != 0
                        own.index_add_(0, qn[m_], torch.ones(int(m_.sum()), dtype=torch.int32, device=dev))
            arr["own"] = own[nid].cpu().numpy()
        # 선택: 색칠 탐침 조립
        if ST["ASM"] == -2 or (ST["ASM"] == s._lw_fr and s._lw_it == ST["IT"][0]):
            arr.update(assemble(s, Afull, X, inner, idx, capN, IP, dev, o))
        if getattr(s, "_lw_sx", None) is not None:
            arr.update(s._lw_sx); s._lw_sx = None
            arr["b_red"] = b.float().cpu().numpy(); arr["idx_red"] = idx.cpu().numpy().astype(np.int32)
        if getattr(s, "_lw_low", None) is not None:
            arr["lowmode"] = s._lw_low[nid].float().cpu().numpy(); s._lw_low = None
        s._lw_arr = arr
        return o

    def Tmats(s, capN, nr, dev):
        sk = s.sk.view(-1)[:capN].float(); pm = s.pm.view(-1)[:capN]
        n = torch.tensor(nr, dtype=torch.float32, device=dev)
        I = torch.eye(3, device=dev).expand(capN, 3, 3)
        Pp = I - n[:, None] * n[None, :]
        T = torch.where((pm == 0)[:, None, None], sk[:, None, None] * I, sk[:, None, None] * Pp + (I - Pp))
        T = torch.where((pm == 2)[:, None, None], torch.zeros_like(T), T)
        return T

    def experiments(s, o, Afull, Aop, Mop, b, X, inner, outer, idx, capN, IP, dev, nr, it2, rM, rJ, rI, rMf):
        i32 = np.int32
        out = {}
        t0 = time.time()
        # (a) 정확 D̃: 반경 R 색칠 탐침 (같은 색 노드 사이 Ã 결합 0)
        Rr = ST["R"]; Pp = 2 * Rr + 1
        nb1, nb2 = int(IP[7]), int(IP[8])
        n = torch.arange(capN, device=dev)
        gi, gj, gk = n // (nb1 * nb2), (n // nb2) % nb1, n % nb2
        col = ((gi % Pp) * Pp + (gj % Pp)) * Pp + (gk % Pp)
        De = torch.zeros(capN, 3, 3, device=dev, dtype=torch.float64)
        for cc in torch.unique(col[inner]).tolist():
            sel = inner & (col == cc)
            for comp in range(3):
                X.zero_(); X.view(capN, 3)[sel, comp] = 1.0
                y = Afull(X).view(capN, 3)
                De[sel, :, comp] = y[sel].to(torch.float64)
        out["t_probe"] = time.time() - t0
        # (b) 근사 D̃ (절 CM94 wolfh13 정의): D_j + T_j (Σ_o c_oj² D_o) T_j, D = (Dinv)⁻¹ (rt_grad 의 Mi + T B T 블록)
        E3 = torch.eye(3, device=dev, dtype=torch.float64).expand(capN, 3, 3)
        Dsrc = s.Dinv.view(-1)[:capN * 9].view(capN, 3, 3).double()
        actn = (s.mn.view(-1)[:capN] > 0)
        okd = actn & torch.isfinite(Dsrc).flatten(1).all(1) & (torch.linalg.det(torch.where(torch.isfinite(Dsrc), Dsrc, torch.zeros_like(Dsrc))).abs() > 0)
        Dn = torch.linalg.inv(torch.where(okd[:, None, None], Dsrc, E3))
        Dn = torch.where(okd[:, None, None], Dn, torch.zeros_like(Dn))
        acc = torch.zeros(capN, 3, 3, device=dev, dtype=torch.float64)
        on_ = torch.nonzero(outer).squeeze(1)
        b0 = s.w8b[on_].long(); Wt = s.w8w.view(-1, 9)[on_].double()
        for a_ in range(3):
            for b_ in range(3):
                for c_ in range(3):
                    ww = Wt[:, a_] * Wt[:, 3 + b_] * Wt[:, 6 + c_]
                    qn = b0 + (a_ * nb1 + b_) * nb2 + c_
                    acc.index_add_(0, qn, (ww ** 2)[:, None, None] * Dn[on_])
        T = Tmats(s, capN, nr, dev).double()
        Da = Dn + T @ acc @ T
        Da = torch.where(inner[:, None, None], Da, torch.eye(3, device=dev, dtype=torch.float64).expand(capN, 3, 3))
        De2 = torch.where(inner[:, None, None], De, torch.eye(3, device=dev, dtype=torch.float64).expand(capN, 3, 3))
        De2 = 0.5 * (De2 + De2.transpose(1, 2))
        def sinv(Dm):
            dt = torch.linalg.det(Dm)
            good = torch.isfinite(dt) & (dt.abs() > 1e-300)
            return torch.where(good[:, None, None], torch.linalg.inv(torch.where(good[:, None, None], Dm, E3)), torch.zeros_like(Dm)), int((~good & inner).sum())
        (Dei, nbe), (Dai, nba) = sinv(De2), sinv(Da)
        out["n_sing"] = [nbe, nba]
        De2 = torch.where((torch.linalg.det(De2).abs() > 1e-300)[:, None, None], De2, E3)
        dd = (De2 - Da)[inner].flatten(1).norm(dim=1) / De2[inner].flatten(1).norm(dim=1)
        out["Dapprox_relerr"] = [float(torch.quantile(dd.float(), q)) for q in (0.5, 0.9, 0.99)]
        out["De_norm_ratio"] = float(De2[inner].flatten(1).norm(dim=1).median() / Dn[inner].flatten(1).norm(dim=1).median())
        idn = idx // 3; idc = idx % 3
        Z = torch.zeros(capN, 3, device=dev, dtype=torch.float64)

        def jac(Dinv_):
            def f(x):
                Z.zero_(); Z[idn, idc] = x
                return torch.bmm(Dinv_, Z[:, :, None]).squeeze(2)[idn, idc]
            return f
        JDe, JDa = jac(Dei), jac(Dai)

        # λ(D̃⁻¹Ã): 대칭형 L⁻¹ Ã L⁻ᵀ
        Le = torch.linalg.cholesky(De2); Lei = torch.linalg.inv(Le)
        def symDe(x):
            Z.zero_(); Z[idn, idc] = x
            y = torch.bmm(Lei.transpose(1, 2), Z[:, :, None]).squeeze(2)[idn, idc]
            y = Aop(y)
            Z.zero_(); Z[idn, idc] = y
            return torch.bmm(Lei, Z[:, :, None]).squeeze(2)[idn, idc]
        evS, _, Ys = lanczos(symDe, int(idx.numel()), 200, dev, 12, vec=3)
        out["lam_DeA"] = [float(evS[0]), float(evS[-1])]
        out["lam_DeA_low5"] = [float(q) for q in evS[:5]]
        # 가장 낮은 모드 위치: u = L⁻ᵀ y
        mn_ = s.mn.view(-1)[:capN].float()
        light = (mn_ < 1e-2 * mn_[inner].median()) & inner
        ownm = torch.zeros(capN, dtype=torch.bool, device=dev)
        for a_ in range(3):
            for b_ in range(3):
                for c_ in range(3):
                    ww = Wt[:, a_] * Wt[:, 3 + b_] * Wt[:, 6 + c_]
                    ownm[(b0 + (a_ * nb1 + b_) * nb2 + c_)[ww != 0]] = True
        bar_ = (s.bKn.view(-1)[:capN] > 0) if getattr(s, "bKn", None) is not None else torch.zeros_like(inner)
        pmn = s.pm.view(-1)[:capN] != 0
        lows = []
        for jv in range(Ys.shape[0]):
            Z.zero_(); Z[idn, idc] = Ys[jv]
            u = torch.bmm(Lei.transpose(1, 2), Z[:, :, None]).squeeze(2)
            un = (u ** 2).sum(1)
            tot = float(un.sum())
            lows.append(dict(lam=float(evS[jv]), own=float(un[ownm].sum() / tot), light=float(un[light].sum() / tot), bar=float(un[bar_].sum() / tot),
                             pm=float(un[pmn].sum() / tot), top10=float(torch.topk(un, 10).values.sum() / tot), top100=float(torch.topk(un, 100).values.sum() / tot),
                             nlight=int(light.sum()), nown=int((ownm & inner).sum())))
            if jv == 0:
                s._lw_low = un
        out["lowmodes"] = lows
        NP = ST["NPCG"]
        rDe = pcg(Aop, JDe, b, NP); rDa = pcg(Aop, JDa, b, NP)
        # (c) 층 0 을 Ã·D̃ 로 바꾼 V 순환 (거친 층 = 실행기 그대로): z = ω D̃⁻¹ r; z += Q Vc(Qᵀ (r − Ã z)); z += ω D̃⁻¹ (r − Ã z), Q = T P (내부 몫)
        Lc = s.ML[0]
        res_f = torch.zeros(capN * 3, dtype=torch.float32, device=dev)
        cor = torch.zeros(capN * 3, dtype=torch.float32, device=dev)
        LM = s.LM

        def coarse(e):
            res_f.zero_(); res_f.view(capN, 3)[idn, idc] = e.float()
            s.k("rt_mg_node", capN, (i32(0), P(LM), i32(4), P(s.meff), it2, P(s.sk), P(s.pm), *[np.float32(q) for q in nr], P(s.Dinv), P(s.W), P(res_f), P(None), P(None), P(s.m0u), P(None)), b=256)
            if s.mv2:
                s.k("rt_mg_restrict3", Lc["capn"] * 16, (i32(0), P(LM), P(s.m0u), P(Lc["r"]), i32(3)), b=256)
            else:
                s.k("rt_mg_restrict", Lc["capn"], (i32(0), P(LM), P(s.m0u), P(Lc["r"]), i32(3)), b=256)
            s.mg_vc(1, Lc["r"], Lc["z"])
            s.k("rt_mg_prolong", capN, (i32(0), P(LM), P(Lc["z"]), P(s.m0u), i32(0)), b=256)
            s.k("rt_mg_node", capN, (i32(0), P(LM), i32(4), P(s.meff), it2, P(s.sk), P(s.pm), *[np.float32(q) for q in nr], P(s.Dinv), P(s.W), P(s.m0u), P(None), P(None), P(cor), P(None)), b=256)
            return cor.view(capN, 3)[idn, idc].to(f64)

        def vmx(Jop, om):
            def f(r):
                z = om * Jop(r)
                z = z + coarse(r - Aop(z))
                return z + om * Jop(r - Aop(z))
            return f
        lD = out["lam_DeA"][1]
        om = 1.0 / lD
        # 근사 D̃ 의 λmax (거듭제곱 30)
        g = torch.Generator(device=dev).manual_seed(13)
        v = torch.randn(int(idx.numel()), dtype=f64, device=dev, generator=g)
        lam_a = 1.0
        for _ in range(30):
            w = JDa(Aop(v)); lam_a = float(torch.dot(v, w) / torch.dot(v, v)); v = w / w.norm()
        out["lam_DaA_pow30"] = lam_a
        VE, VA = vmx(JDe, om), vmx(JDa, 1.0 / lam_a)
        rVE = pcg(Aop, VE, b, min(NP, 200)); rVA = pcg(Aop, VA, b, min(NP, 200))
        # 기준 φ* (정확 D̃ V 순환 PCG 600)
        rRef = pcg(Aop, VE, b, 600)
        phis = [rRef["phi"].min(), rDe["phi"].min(), rM["phi"].min()]
        ph = float(min(phis))
        out["phi_ref"] = ph; out["ref_res"] = float(rRef["hist"][-1])

        def eA(r_):
            return [float(math.sqrt(max(q - ph, 0.0) / max(-ph, 1e-300))) for q in r_["phi"]]
        for nm, r_ in (("M", rM), ("J", rJ), ("I", rI), ("Mf", rMf), ("De", rDe), ("Da", rDa), ("VE", rVE), ("VA", rVA)):
            out["eA_" + nm] = eA(r_)
            out["res_" + nm] = r_["hist"].tolist()
            out["bad_" + nm] = [r_["npap"], r_["nrz"]]
        for nm, r_ in (("De", rDe), ("VE", rVE), ("M", rM)):
            out["ritz_" + nm] = [float(r_["ritz"][0]), float(r_["ritz"][-1])] if r_["ritz"] is not None else None
        sx = {}
        for nm, r_ in (("M", rM), ("I", rI), ("De", rDe), ("Da", rDa), ("VE", rVE), ("Mf", rMf), ("Ref", rRef)):
            for kk, xv in r_["xs"].items():
                sx[f"x_{nm}_{kk}"] = xv
        s._lw_sx = sx
        out["t_exp"] = time.time() - t0
        print(f"[절 CM127] 시험: D̃ 근사 오차 {out['Dapprox_relerr']} λ(D̃⁻¹Ã) {out['lam_DeA']} 낮은 모드 {out['lowmodes'][0]} | 에너지 오차 5·20 회: "
              + " ".join(f"{k}:{out['eA_' + k][min(5, len(out['eA_' + k]) - 1)]:.2g}/{out['eA_' + k][min(20, len(out['eA_' + k]) - 1)]:.2g}" for k in ("M", "J", "I", "De", "Da", "VE", "VA")), flush=True)
        return out

    def _scatter(x, idx, capN, nid):
        full = torch.zeros(capN * 3, dtype=f64, device=x.device)
        full[idx] = x
        return full.view(capN, 3)[nid].float().cpu().numpy()

    def assemble(s, Afull, X, inner, idx, capN, IP, dev, o):
        Rr = ST["R"]
        Pp = 2 * Rr + 1
        nb0, nb1, nb2 = int(IP[6]), int(IP[7]), int(IP[8])
        n = torch.arange(capN, device=dev)
        gi, gj, gk = n // (nb1 * nb2), (n // nb2) % nb1, n % nb2
        dofmask = inner.repeat_interleave(3)
        rows, cols, vals = [], [], []
        nleak = 0
        t0 = time.time()
        for ci in range(Pp):
            for cj in range(Pp):
                for ck in range(Pp):
                    sel = inner & (gi % Pp == ci) & (gj % Pp == cj) & (gk % Pp == ck)
                    if not bool(sel.any()):
                        continue
                    for comp in range(3):
                        X.zero_()
                        X.view(capN, 3)[sel, comp] = 1.0
                        y = Afull(X)
                        nzr = torch.nonzero((y != 0) & dofmask).squeeze(1)
                        if nzr.numel() == 0:
                            continue
                        rn = nzr // 3
                        di = (ci - gi[rn] + Rr) % Pp - Rr
                        dj = (cj - gj[rn] + Rr) % Pp - Rr
                        dk = (ck - gk[rn] + Rr) % Pp - Rr
                        cn = ((gi[rn] + di) * nb1 + (gj[rn] + dj)) * nb2 + (gk[rn] + dk)
                        good = (cn >= 0) & (cn < capN)
                        good &= sel[cn.clamp(0, capN - 1)]
                        nleak += int((~good).sum())
                        rows.append(nzr[good].int()); cols.append((cn[good] * 3 + comp).int()); vals.append(y[nzr[good]].clone())
        rows = torch.cat(rows); cols = torch.cat(cols); vals = torch.cat(vals)
        # 줄어든 번호
        mp = torch.full((capN * 3,), -1, dtype=torch.int64, device=dev)
        mp[idx] = torch.arange(idx.numel(), device=dev)
        ri, cj_ = mp[rows.long()], mp[cols.long()]
        okk = (ri >= 0) & (cj_ >= 0)
        ri, cj_, vv = ri[okk], cj_[okk], vals[okk].to(f64)
        nd = int(idx.numel())
        Asp = torch.sparse_coo_tensor(torch.stack([ri, cj_]), vv, (nd, nd)).coalesce()
        g = torch.Generator(device=dev).manual_seed(7)
        xr = torch.randn(nd, dtype=f64, device=dev, generator=g)
        X.zero_(); X[idx] = xr.float()
        ya = Afull(X)[idx].to(f64)
        yb = torch.sparse.mm(Asp, xr[:, None]).squeeze(1)
        o["asm_err"] = float((ya - yb).norm() / ya.norm())
        o["asm_nnz"] = int(Asp._nnz()); o["asm_leak"] = nleak; o["asm_sec"] = time.time() - t0
        At = Asp.t().coalesce()
        D = (Asp - At).coalesce()
        o["asm_asym"] = float(D.values().norm() / Asp.values().norm())
        dv = D.values().abs()
        top = torch.topk(dv, min(50, dv.numel()))
        I = D.indices()[:, top.indices]
        idxn = idx.cpu().numpy()
        return dict(asm_r=ri.cpu().numpy().astype(np.int32), asm_c=cj_.cpu().numpy().astype(np.int32), asm_v=vv.float().cpu().numpy(), asm_idx=idxn.astype(np.int32),
                    asm_top=np.stack([idxn[I[0].cpu().numpy()], idxn[I[1].cpu().numpy()]], 1).astype(np.int32), asm_topv=top.values.cpu().numpy())

    def finish(s):
        o = s._lw_pend
        s._lw_pend = None
        arr = s._lw_arr
        s._lw_arr = None
        torch.cuda.synchronize()
        capN = int(s.capN)
        nid = torch.from_numpy(arr["nid"]).long().to(s.dev)
        arr["rfin"] = s.r.reshape(-1, 3)[:capN][nid].cpu().numpy()
        arr["dlfin"] = s.dl.reshape(-1, 3)[:capN][nid].cpu().numpy()
        arr["real_r2"] = np.array(s._lw_hist, dtype=np.float64)
        o["real_prel"] = (np.sqrt(np.array(s._lw_hist)) / max(o["bnorm"], 1e-300)).tolist()
        o["real_ct"] = s.CT.cpu().tolist()
        np.savez_compressed(os.path.join(ST["out"], "lw", f"f{o['f']:03d}_i{o['it']:02d}.npz"), **arr)
        LOG[0].write(json.dumps(o, ensure_ascii=False, default=float) + "\n")
        LOG[0].flush()
        _dump()

    import atexit
    atexit.register(_dump)
    print(f"[절 CM127] linwhy 켬 -- 프레임 {F} 뉴턴 반복 {ITS} 선형계 측정 (그 프레임만 그래프 대신 바로 실행) -> {out}/lw", flush=True)
    _dump()
