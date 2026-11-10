"""절 CM104 진단 (사용자 11-10): 레고 사면체 판 (T9·T9S) 첫 프레임 걸음이 왜 안 풀리는가 -- 같은 걸음을 여러 판으로 다시 풂 (결과 불변 판 아님: 진단 프로세스는 첫 걸음 뒤 끝냄).

훅: 새 실행기 RTSolver 의 첫 프레임 (그래프 밖, ga_first) 첫 걸음 뉴턴 반복 0 의 ctl(3, 0) (rt_grad 직후 -- 시작점 y0, 관성 예측 ŷ, 유효 질량, 요소 헤시안,
블록 대각, 장벽 활성이 모두 실행기가 만든 그대로) 에서 아래를 하고 프로세스를 끝냄 (os._exit). 모든 연산은 실행기 커널 (s.k) 을 그대로 다시 부르거나 torch 로 읽기.

부품 (--tw 쉼표 목록):
  base  상태 지표 (노드 질량 0 인데 강성 있는 노드 Z, 적분점 det 분포, 시작점·예측 비교), 같은 상태 선형계 PCG (실행기 MG · 블록 야코비) 수렴·란초스 조건수,
        실행기와 같은 설정 (MG, PCG 상한 4, 허용치 0.1) 뉴턴 다시 (실행기 기록과 대조)
  a     뉴턴 + PCG 거의 정확 (실행기 MG, 상한 2000, 허용치 1e-6)
  b     뉴턴 + MG 끔 블록 야코비 (상한 4 · 2000)
  c     진단용 MG: 사면체 기저에 맞는 거친 층 (꼭짓점 맞춘 삼선형 전달 P, 갈러킨 Pᵀ H P, 4 층, 블록 야코비 평활 V(2,2), 가장 거친 층 직접) -- 선형계 수렴·조건수 + 뉴턴 (상한 4 · 2000)
  d     저질량 m̄ 를 사면체 노드 질량 기준으로 (m̄_T = 1e-3 x 사면체 노드 질량 중앙값; d1 = m < m̄_T 노드 관성 예측 고정 (재질 판 mfix 와 같은 T = 0), d2 = 유효 질량 하한 m̄_T)
  e     시작점 섞기 끔 (뉴턴 시작 = 섞기 전 관성 예측 ŷ, η = eS)
  g     (추가 진단) 질량 0 · 강성 있는 노드 Z 의 시작점 변위를 이웃 질량 가중 평균으로 (속도 0 으로 남은 노드가 시작점을 뒤집는지 가르기)
  f     헤시안 대칭·양정치 (명시 행렬 뽑기 -- 색칠 탐침 5³ x 3, 스텐실 폭), 기울기·헤시안 유한차분, 사면체 기저 독립 참값 (torch FCR 힘) 대조
결과: <out>/tetwhy.json (+ 로그 줄 '[tetwhy] ...').
"""
import json
import math
import os
import time
import traceback

import numpy as np
import torch

CFG = dict(parts=set(), out=".", done=False, tag="")


def install(out, parts, tag=""):
    from . import fem_an_rt as ART
    CFG.update(out=out, parts=set(parts), tag=tag)
    C = ART.RTSolver
    if getattr(C.__init__, "_tw", False):
        return
    f0 = C.__init__

    def init(s, *a, **kw):
        f0(s, *a, **kw)
        oc = s.ctl

        def ctl(op, it=0):
            if op == 3 and it == 0 and getattr(s, "ga_first", False) and not CFG["done"]:
                CFG["done"] = True
                rc = 0
                try:
                    run(s)
                except Exception:
                    rc = 3
                    TW.res["error"] = traceback.format_exc()[-3000:]
                    print("[tetwhy] 오류\n" + TW.res["error"], flush=True)
                finally:
                    TW.dump()
                    print(f"[tetwhy] 끝 ({time.time() - TW.t0:.1f} s) -> {TW.path}", flush=True)
                    os._exit(rc)
            return oc(op, it)
        s.ctl = ctl
    init._tw = True
    C.__init__ = init
    print(f"[절 CM104 tetwhy] 훅 설치 부품 {sorted(CFG['parts'])} 출력 {out}", flush=True)


class TW:
    res = {}
    t0 = time.time()
    path = None

    @staticmethod
    def dump():
        TW.path = os.path.join(CFG["out"], "tetwhy.json")
        with open(TW.path, "w") as fo:
            json.dump(TW.res, fo, ensure_ascii=False, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))


def say(*a):
    print("[tetwhy]", *a, f"({time.time() - TW.t0:.1f} s)", flush=True)


def _f(x):
    x = float(x)
    return x if math.isfinite(x) else str(x)


class Ctx:
    pass


def run(s):
    from . import fem_an_rt as ART
    P = ART.P
    TW.t0 = time.time()
    sync = torch.cuda.synchronize
    sync()
    dev = s.y.device
    i32, f32 = np.int32, np.float32
    c = Ctx()
    IPh = s.IP.cpu().numpy().astype(np.int64)
    c.nN, c.nq, c.nb1, c.nb2 = int(IPh[9]), int(IPh[12]), int(IPh[7]), int(IPh[8])
    c.nb0 = c.nN // (c.nb1 * c.nb2)
    nN, n3, nq = c.nN, 3 * c.nN, c.nq
    capN, capQ = s.capN, s.capQc * 8
    sg = s.stages(s.Kcur)[0]
    tau, thinv = float(sg["tau"]), float(sg["thinv"])
    it2 = 1.0 / (tau * tau)
    FL = s.floor
    nr = FL["NR"] if FL else [0.0, 0.0, 1.0]
    NR = [f32(q) for q in nr]
    BA = (*NR, f32(FL["PN"]), f32(FL["DH"]), f32(FL["KS"])) if (FL and s.bar) else None
    R = TW.res
    R.update(tag=CFG["tag"], parts=sorted(CFG["parts"]), nN=nN, nq=nq, nb=[c.nb0, c.nb1, c.nb2], capN=capN, capQ=capQ, tau=tau, it2=it2, thinv=thinv,
             fmv=s.fmv_name(), nl=int(s.nl), mgbar=int(s.mgbar), bar=bool(s.bar), bfull=bool(getattr(s, "bfull", False)), ncomp=bool(s.ncomp),
             mthr=float(getattr(s, "mthr", 0.0) or 0.0), mfloor=float(s.mfloor), goct=float(s.goct), y0loc=int(getattr(s, "y0loc", 0) or 0),
             tet=bool(getattr(s, "_tet1_done", False)))
    say("상태", {k: R[k] for k in ("nN", "nq", "nb", "fmv", "nl", "mgbar", "tet", "mthr")})
    # ---- 걸음 시작 상태 사본
    c.Y0 = s.y.view(-1)[:n3].clone(); c.E0 = s.eta.view(-1)[:n3].clone(); c.YH = s.yh.view(-1)[:n3].clone()
    c.U = s.U.view(-1)[:n3].clone(); c.VK = s.vk.view(-1)[:n3].clone(); c.MN = s.mn[:nN].clone(); c.ME0 = s.meff[:nN].clone()
    c.SK = s.sk[:nN].clone(); c.PM = s.pm[:nN].clone(); c.R0 = s.r.view(-1)[:n3].clone()
    c.YS = s.ySv.view(-1)[:n3].clone(); c.ES = s.eS.view(-1)[:n3].clone()
    c.B0 = s.B.view(-1)[:9 * nN].clone()
    c.ME = c.ME0.clone()

    def Tmats(SK, PM):
        nv = torch.tensor(nr, device=dev, dtype=torch.float32)
        I3 = torch.eye(3, device=dev)
        nn_ = torch.outer(nv, nv)
        T = SK[:, None, None] * I3
        m1 = PM == 1
        T[m1] = SK[m1, None, None] * (I3 - nn_) + nn_
        T[PM == 2] = 0.0
        return T
    c.T = Tmats(c.SK, c.PM)
    R["pm_counts"] = {str(k): int((c.PM == k).sum()) for k in range(4)}
    R["sk_lt1"] = int((c.SK < 1).sum())

    def Tap(T, v):
        return torch.bmm(T, v.view(-1, 3, 1)).view(-1)

    # ---- 버퍼
    Zf = lambda n: torch.zeros(n, device=dev, dtype=torch.float32)
    c.pb, c.Tp, c.KTp, c.Ap = Zf(capN * 3), Zf(capN * 3), Zf(capN * 3), Zf(capN * 3)
    c.rb, c.zb, c.yb = Zf(capN * 3), Zf(capN * 3), Zf(capN * 3)
    c.gn2, c.kn2 = Zf(capN), Zf(capN)
    c.SCd = torch.zeros(64, device=dev, dtype=torch.float64)
    c.meb = Zf(capN)
    c.meb.copy_(s.meff[:capN])
    Kb_arg = (lambda: P(s.bKn)) if (s.bar and s.mgbar and not getattr(s, "bfull", False)) else (lambda: P(None))

    # ---- 적분점 det (torch, 실행기와 같은 식 X = F_q + Σ_o y_o ⊗ F_qᵀ ∇w_o)
    q_ = torch.arange(nq, device=dev)
    cq = (q_ >> 3).long(); oc = (q_ & 7).long()
    cbq = s.cb[cq].long()
    off = torch.tensor([[o // 9, (o // 3) % 3, o % 3] for o in range(27)], device=dev, dtype=torch.long)
    nid = ((cbq[:, None, 0] + off[None, :, 0]) * c.nb1 + (cbq[:, None, 1] + off[None, :, 1])) * c.nb2 + (cbq[:, None, 2] + off[None, :, 2])   # (nq, 27)
    BGt = s.BG.view(8, 27, 3)[oc]                                                                   # (nq, 27, 3)
    Fq = s.Fq[:nq].view(nq, 3, 3).float()
    Gq = torch.einsum("qab,qoa->qob", Fq, BGt)                                                      # g_b = Σ_a F_ab bg_a
    Vq = s.Vq[:nq].clone()
    act_q = Vq > 0
    bgnz = (BGt.abs().sum(2) > 0)                                                                   # (nq, 27) 기저 기울기가 있는 노드
    c.nid, c.Gq, c.Fq, c.Vq, c.act_q, c.bgnz = nid, Gq, Fq, Vq, act_q, bgnz

    def Xq(y):
        Y = y.view(-1, 3)[nid]                                                                       # (nq, 27, 3)
        return Fq + torch.einsum("qoa,qob->qab", Y.float(), Gq)

    def Jq(y):
        return torch.linalg.det(Xq(y))

    def jstat(y):
        J = Jq(y)[act_q]
        return dict(min=_f(J.min()), n_le0=int((J <= 0).sum()), n_lt01=int((J < 0.1).sum()), nact=int(act_q.sum()))

    # ---- 노드 분류
    Btr = c.B0.view(nN, 9)[:, [0, 4, 8]].sum(1)
    mpos = c.MN[c.MN > 0]
    med = float(mpos.median()) if mpos.numel() else 0.0
    c.med = med
    Z = (c.MN <= 0) & (Btr > 0)
    L3 = (c.MN > 0) & (c.MN < 1e-3 * med)
    stiff = Btr > 0
    VKn = c.VK.view(nN, 3).norm(dim=1)
    D0 = (c.Y0 - c.U).view(nN, 3).norm(dim=1)
    DH = (c.YH - c.U).view(nN, 3).norm(dim=1)
    heavy = c.MN >= 1e-3 * med
    R["nodes"] = dict(stiff=int(stiff.sum()), mass_pos=int((c.MN > 0).sum()), med_mass=med, Z_massless_stiff=int(Z.sum()), L_lt1e3med=int(L3.sum()),
                      Z_T_identity=int((Z & (c.PM == 0) & (c.SK == 1)).sum()),
                      v_heavy_med=_f(VKn[heavy].median()) if heavy.any() else None, v_Z_max=_f(VKn[Z].max()) if Z.any() else None,
                      dY0_heavy_med=_f(D0[heavy].median()) if heavy.any() else None, dY0_Z_med=_f(D0[Z].median()) if Z.any() else None,
                      dYH_heavy_med=_f(DH[heavy].median()) if heavy.any() else None, dYH_Z_med=_f(DH[Z].median()) if Z.any() else None,
                      y0_vs_yhat=_f((c.Y0 - c.YH).norm() / max(float((c.YH - c.U).norm()), 1e-30)),
                      y0_vs_yS=_f((c.Y0 - c.YS).norm() / max(float((c.YS - c.U).norm()), 1e-30)))
    if getattr(s, "ykn", None) is not None:
        yk = s.ykn[:nN]
        R["nodes"]["y0loc_beta_lt1"] = int((yk > 0).sum())
    # 적분점: 시작점 det ≤ 0 인 점 중 질량 0 꼭짓점 있는 점
    J0 = Jq(c.Y0)
    inv0 = act_q & (J0 <= 0)
    vm = c.MN[nid.clamp(max=nN - 1)]                                                                # (nq, 27)
    hasZ = (bgnz & (vm <= 0)).any(1)
    minm = torch.where(bgnz, vm, torch.full_like(vm, 3e38)).min(1).values
    R["q_start"] = dict(J_y0=jstat(c.Y0), J_yhat=jstat(c.YS), J_U=jstat(c.U), inv_with_Zvertex=int((inv0 & hasZ).sum()), inv=int(inv0.sum()),
                        act_with_Zvertex=int((act_q & hasZ).sum()), inv_minm_over_med_med=_f((minm[inv0] / max(med, 1e-30)).median()) if inv0.any() else None,
                        Vq_zero_frac=_f(1 - act_q.float().mean()))
    say("노드", R["nodes"]); say("적분점", R["q_start"])

    # ---- 커널 래퍼
    def kel(y_ptr, mode, E_ptr, f_ptr, H_ptr, B_ptr):
        s.k("rt_elem", capQ, (P(s.IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), y_ptr, f32(s.mu), f32(s.lam), i32(mode), E_ptr, f_ptr, H_ptr, i32(capQ), B_ptr))

    def grad_at(eta, me=None):
        """η 에서 기울기 (실행기 반복 0 과 같은 커널 순서) -- s.y·η·f·B·H·Dinv·r 를 바꿈. 돌려줌: g (n3 f64), 에너지 (관성·탄성·장벽), 탄성 힘 f_el (f64)."""
        if me is not None:
            s.meff[:nN].copy_(me)
        s.eta.view(-1)[:n3].copy_(eta)
        s.y.view(-1)[:n3].copy_(c.YH + Tap(c.T, eta))
        sync()
        s.zf(s.f, 9, 3); s.zf(s.B, 9, 9)
        s.k("rt_sc_clear", 1, (P(s.SC), i32(0), i32(60)), b=1)
        kel(P(s.y), 1 | 2 | 8 | 4, P(s.SC[10:]), P(s.f), P(s.H), P(s.B))
        sync()
        Eel = float(s.SC[10]); fel = s.f.view(-1)[:n3].clone()
        if s.bar:
            s.zf(s.bGn); s.zf(s.bKn)
            s.k("rt_bar_q", capQ, (P(s.IP), P(s.cb), P(s.BW), P(s.Vq), P(s.y), P(s.U), f32(thinv), *BA, i32(1), P(s.bGn), P(s.bKn), P(s.SC[10:])))
            s.k("rt_bar_n", capN, (P(s.IP), P(s.bGn), P(s.bKn), *NR, P(s.f), P(s.B)), b=256)
            if s.mgbar >= 2 and s.nl > 1 and s.mv2:
                for l_ in range(1, s.nl):
                    Lc_ = s.ML[l_ - 1]
                    s.k("rt_mg_restrict3", Lc_["capn"] * 16, (i32(l_ - 1), P(s.LM), P(s.bKn if l_ == 1 else s.ML[l_ - 2]["Kb"]), P(Lc_["Kb"]), i32(1)), b=256)
                    s.k("rt_mg_dbar", Lc_["capn"], (i32(l_), P(s.LM), P(Lc_["D"]), P(Lc_["Kb"]), *NR, P(Lc_["Db"])), b=256)
        sync()
        Ebar = float(s.SC[10]) - Eel
        s.k("rt_grad", capN, (P(s.IP), P(s.meff), f32(it2), P(s.sk), P(s.pm), *NR, P(s.eta), P(s.f), P(s.B), P(s.Dinv), P(s.r), P(s.z), P(s.p), P(s.dl), P(s.Tp),
                                P(s.SC), P(s.g0buf)), b=256)
        sync()
        g = -s.r.view(-1)[:n3].double()
        Ein = float(0.5 * (s.meff[:nN].double() * it2 * (eta.double().view(-1, 3) ** 2).sum(1)).sum())
        return g, dict(Ein=Ein, Eel=Eel, Ebar=Ebar, E=Ein + Eel + Ebar), fel

    def energy_at(eta, me):
        """에너지만 (s.y·H·B 안 바꿈: 위치는 내 버퍼)."""
        c.yb.zero_(); c.yb[:n3] = c.YH + Tap(c.T, eta)
        c.SCd.zero_()
        sync()
        kel(P(c.yb), 4, P(c.SCd[10:]), P(s.f), P(s.H), P(s.B))
        if s.bar:
            c.gn2.zero_(); c.kn2.zero_()
            s.k("rt_bar_q", capQ, (P(s.IP), P(s.cb), P(s.BW), P(s.Vq), P(c.yb), P(s.U), f32(thinv), *BA, i32(1), P(c.gn2), P(c.kn2), P(c.SCd[11:])))
        sync()
        Ein = float(0.5 * (me.double() * it2 * (eta.double().view(-1, 3) ** 2).sum(1)).sum())
        return Ein + float(c.SCd[10]) + float(c.SCd[11]), float(c.SCd[10]), float(c.SCd[11])

    def hv(p, T=None):
        """헤시안 곱 (실행기 PCG 와 같은 커널: Mi p + T K T p + 장벽 덩어리)."""
        T = c.T if T is None else T
        c.Tp.zero_(); c.Tp[:n3] = Tap(T, p.float()); c.KTp.zero_(); c.pb.zero_(); c.pb[:n3] = p.float(); c.Ap.zero_()
        s.k(s.fmv_name(), capQ, (P(s.IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.fmv_H()), P(c.Tp), P(c.KTp), i32(capQ)))
        if s.bar and not s.mgbar and not getattr(s, "bfull", False):
            s.k("rt_bar_mv", capN, (P(s.IP), P(s.bKn), *NR, P(c.Tp), P(c.KTp)), b=256)
        s.k("rt_ap", capN, (P(s.IP), P(s.meff), f32(it2), P(s.sk), P(s.pm), *NR, P(c.pb), P(c.KTp), P(c.Ap), P(c.SCd), Kb_arg()), b=256)
        sync()
        return c.Ap[:n3].double()

    def kv(v):
        """탄성 헤시안 곱 K v 만 (T 없음, 관성·장벽 없음)."""
        c.Tp.zero_(); c.Tp[:n3] = v.float(); c.KTp.zero_()
        s.k(s.fmv_name(), capQ, (P(s.IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.fmv_H()), P(c.Tp), P(c.KTp), i32(capQ)))
        sync()
        return c.KTp[:n3].double()

    def prec_mg(r):
        c.rb.zero_(); c.rb[:n3] = r.float(); c.zb.zero_()
        s.mg_vc(0, c.rb, c.zb)
        sync()
        return c.zb[:n3].double()

    def prec_bj(r):
        D = s.Dinv.view(-1)[:9 * nN].view(nN, 3, 3).double()
        return torch.bmm(D, r.view(nN, 3, 1)).view(-1)

    def mg_ready():
        s.mg_tau = tau
        s.mg_power(0)
        sync()

    def pcg(M, b, maxit, tol, H=None, marks=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000)):
        H = hv if H is None else H
        x = torch.zeros_like(b); r = b.clone(); z = M(r); p = z.clone()
        rz = float(r @ z); rz0 = rz; r0n = float(r.norm())
        al, be, hist = [], [], {}
        out = dict(rz0=rz0, neg_pAp=0, neg_rz=0, it=0)
        if not (rz > 0):
            out["neg_rz"] = 1
        for k in range(maxit):
            Ap = H(p); pAp = float(p @ Ap)
            if not (pAp > 0):
                out["neg_pAp"] = 1; out["pAp"] = _f(pAp); break
            a = rz / pAp
            x += a * p; r -= a * Ap; z = M(r); rzn = float(r @ z)
            if not (rzn > 0):
                out["neg_rz"] = 1; out["it"] = k + 1; break
            bt = rzn / rz
            al.append(a); be.append(bt)
            rel = math.sqrt(rzn / rz0); reln = float(r.norm()) / max(r0n, 1e-300)
            if (k + 1) in marks:
                hist[k + 1] = [rel, reln]
            out["it"] = k + 1
            if rel <= tol:
                break
            p = z + bt * p; rz = rzn
        out["rel_rz"] = math.sqrt(max(rzn, 0) / rz0) if al else None
        out["rel_r"] = float(r.norm()) / max(r0n, 1e-300)
        out["hist"] = hist
        # 란초스 (PCG 계수) -> 전처리 연산자 고윳값 극값
        if len(al) >= 2:
            n_ = len(al)
            Tm = np.zeros((n_, n_))
            for i in range(n_):
                Tm[i, i] = 1.0 / al[i] + (be[i - 1] / al[i - 1] if i > 0 else 0.0)
                if i + 1 < n_:
                    Tm[i, i + 1] = Tm[i + 1, i] = math.sqrt(max(be[i], 0.0)) / al[i]
            ev = np.linalg.eigvalsh(Tm)
            out["lz_min"], out["lz_max"] = float(ev[0]), float(ev[-1])
            out["kappa"] = float(ev[-1] / ev[0]) if ev[0] > 0 else None
        return x, out

    # ---- 시작 기울기 재현 확인
    g0, en0, fel0 = grad_at(c.E0)
    R["repro"] = dict(r_rel=_f((g0 + c.R0.double()).norm() / max(float(c.R0.double().norm()), 1e-300)), F=_f(g0.norm()), **{k: _f(v) for k, v in en0.items()})
    gin = (c.ME.double() * it2).repeat_interleave(3) * c.E0.double()
    R["grad_split"] = dict(inertia=_f(gin.norm()), elastic_T=_f(Tap(c.T.double() if False else c.T, fel0.float()).double().norm()),
                           total=_f(g0.norm()), Z_share=_f(g0.view(nN, 3)[Z].norm() / max(float(g0.norm()), 1e-300)) if Z.any() else 0.0,
                           L_share=_f(g0.view(nN, 3)[L3].norm() / max(float(g0.norm()), 1e-300)) if L3.any() else 0.0)
    say("재현", R["repro"], R["grad_split"])
    mg_ready()

    # ---- 뉴턴 (내 고리: 아르미호 선탐색 2^-k k<10, 하강 보장 = 전처리 기울기)
    def newton(label, eta0, M, cap, tol, nit, me=None, T=None, Mbuild=None):
        Tsave = c.T
        if T is not None:
            c.T = T
        me_ = c.ME0 if me is None else me
        eta = eta0.clone().float()
        g, en, _ = grad_at(eta, me_)
        if M is prec_mg:
            mg_ready()
        F = float(g.norm()); recs = []
        rec0 = dict(it=-1, gF=1.0, E=_f(en["E"]), J=jstat(c.YH + Tap(c.T, eta)))
        recs.append(rec0)
        for it in range(nit):
            Mi = Mbuild() if Mbuild is not None else M
            d, info = pcg(Mi, -g, cap, tol)
            gd = float(g @ d)
            fix = False
            if not (gd < 0):
                d = -Mi(g); gd = float(g @ d); fix = True
            st, E1 = 0.0, None
            for k in range(10):
                a = 0.5 ** k
                Et, _, _ = energy_at((eta.double() + a * d).float(), me_)
                if math.isfinite(Et) and Et <= en["E"] + 1e-4 * a * gd:
                    st, E1 = a, Et; break
            if st > 0:
                eta = (eta.double() + st * d).float()
            g, en, _ = grad_at(eta, me_)
            recs.append(dict(it=it, gF=_f(g.norm() / max(F, 1e-300)), st=st, pcg_it=info["it"], pcg_rel=_f(info["rel_rz"]) if info.get("rel_rz") is not None else None,
                             pcg_rel_r=_f(info["rel_r"]), kappa=info.get("kappa"), neg=[info["neg_pAp"], info["neg_rz"]], fix=fix, E=_f(en["E"]),
                             J=jstat(c.YH + Tap(c.T, eta))))
            if st == 0:
                break
        c.T = Tsave
        grad_at(c.E0, c.ME0)                                                                         # 버퍼를 시작 상태로
        out = dict(F=F, recs=recs, gF_end=recs[-1]["gF"])
        R.setdefault("newton", {})[label] = out
        say("뉴턴", label, "F", f"{F:.4g}", "‖g‖/F", [r_["gF"] for r_ in recs], "J≤0", [r_["J"]["n_le0"] for r_ in recs])
        return out

    parts = CFG["parts"]
    LIN = R.setdefault("lin", {})

    def lin(label, M, cap=2000, tol=1e-6, H=None):
        _, info = pcg(M, -g0, cap, tol, H=H)
        LIN[label] = info
        say("선형", label, {k: info.get(k) for k in ("it", "rel_rz", "rel_r", "kappa", "lz_min", "lz_max", "neg_pAp", "neg_rz")}, "hist", info["hist"])
        return info

    if "base" in parts:
        # MG 전처리 대칭·양정치 (무작위 시험)
        gen = torch.Generator(device=dev); gen.manual_seed(0)
        act3 = stiff.repeat_interleave(3) | (c.MN > 0).repeat_interleave(3)
        sym, posm = [], []
        for _ in range(6):
            u = torch.randn(n3, device=dev, dtype=torch.float64, generator=gen) * act3
            v = torch.randn(n3, device=dev, dtype=torch.float64, generator=gen) * act3
            Mu, Mv = prec_mg(u), prec_mg(v)
            sym.append(abs(float(u @ Mv) - float(v @ Mu)) / max(float(u.norm() * Mv.norm()), 1e-300))
            posm.append(float(u @ Mu) / max(float(u.norm() * Mu.norm()), 1e-300))
        R["mg_prec"] = dict(sym_err_max=max(sym), cos_min=min(posm))
        Hs, Hp = [], []
        for _ in range(4):
            u = torch.randn(n3, device=dev, dtype=torch.float64, generator=gen) * act3
            v = torch.randn(n3, device=dev, dtype=torch.float64, generator=gen) * act3
            Hu, Hv_ = hv(u), hv(v)
            Hs.append(abs(float(u @ Hv_) - float(v @ Hu)) / max(float(u.norm() * Hv_.norm()), 1e-300))
            Hp.append(float(u @ Hu) / max(float(u.norm() * Hu.norm()), 1e-300))
        R["H_rand"] = dict(sym_err_max=max(Hs), cos_min=min(Hp))
        say("MG 전처리", R["mg_prec"], "H", R["H_rand"])
        lin("mg", prec_mg)
        lin("bj", prec_bj)
        lin("mg_cap4", prec_mg, cap=4, tol=0.1)
        newton("base_mg_cap4", c.E0, prec_mg, 4, 0.1, 6)
    if "a" in parts:
        newton("a_mg_exact", c.E0, prec_mg, 2000, 1e-6, 10)
    if "b" in parts:
        lin("bj_cap4", prec_bj, cap=4, tol=0.1)
        newton("b_bj_cap4", c.E0, prec_bj, 4, 0.1, 6)
        newton("b_bj_exact", c.E0, prec_bj, 2000, 1e-6, 10)
    if "d" in parts:
        mT = 1e-3 * med
        lowm = (c.MN < mT) & (stiff | (c.MN > 0))
        R["d"] = dict(mbar_T=mT, n_low=int(lowm.sum()), n_low_massless=int((lowm & (c.MN <= 0)).sum()), mthr_in_solver=R["mthr"],
                      ga_first_mlow="1e-3 x 지금 (사면체) 노드 질량 중앙값 = 이미 사면체 기준")
        T1 = c.T.clone(); T1[lowm] = 0.0                                                             # d1: 관성 예측 고정 (재질 판 'm < mthr -> pm 2')
        e1 = c.E0.clone().view(nN, 3); e1[lowm] = 0.0
        sk_save, pm_save = s.sk[:nN].clone(), s.pm[:nN].clone()
        s.pm[:nN][lowm] = 2
        newton("d1_fix_mg_cap4", e1.view(-1), prec_mg, 4, 0.1, 6, T=T1)
        newton("d1_fix_bj_exact", e1.view(-1), prec_bj, 2000, 1e-6, 10, T=T1)
        s.sk[:nN].copy_(sk_save); s.pm[:nN].copy_(pm_save)
        me2 = torch.maximum(c.ME0, torch.where(stiff | (c.MN > 0), torch.full_like(c.ME0, mT), torch.zeros_like(c.ME0)))
        newton("d2_floor_mg_cap4", c.E0, prec_mg, 4, 0.1, 6, me=me2)
        newton("d2_floor_bj_exact", c.E0, prec_bj, 2000, 1e-6, 10, me=me2)
        grad_at(c.E0, c.ME0)
    if "e" in parts:
        R["e"] = dict(J_yhat=jstat(c.YS), y0_minus_yhat=_f((c.Y0 - c.YS).norm()), E_yhat=_f(energy_at(c.ES, c.ME0)[0]), E_y0=_f(energy_at(c.E0, c.ME0)[0]))
        say("e", R["e"])
        newton("e_nomix_mg_cap4", c.ES, prec_mg, 4, 0.1, 6)
        newton("e_nomix_bj_exact", c.ES, prec_bj, 2000, 1e-6, 10)
    if "g" in parts:
        nb = (c.nb0, c.nb1, c.nb2)
        m3 = c.MN.view(1, 1, *nb).double()
        dsp = (c.Y0 - c.U).view(nN, 3).double()
        ker = torch.ones(1, 1, 3, 3, 3, device=dev, dtype=torch.float64)
        filled = (c.MN > 0).double().view(1, 1, *nb)
        wgt = m3.clone()
        dnew = dsp.clone()
        todo = Z.clone()
        for _ in range(4):
            if not todo.any():
                break
            den = torch.nn.functional.conv3d(wgt, ker, padding=1).view(-1)
            num = torch.stack([torch.nn.functional.conv3d((wgt.view(-1) * dnew[:, a]).view(1, 1, *nb), ker, padding=1).view(-1) for a in range(3)], 1)
            ok = todo & (den > 0)
            dnew[ok] = num[ok] / den[ok, None]
            wgt = wgt.view(-1).clone(); wgt[ok] = med * 1e-3; wgt = wgt.view(1, 1, *nb)
            todo = todo & ~ok
        ynew = c.U.view(nN, 3).double() + dnew
        eg = c.E0.clone().view(nN, 3)
        zi = Z & (c.PM == 0) & (c.SK == 1)
        eg[zi] = (ynew[zi] - c.YH.view(nN, 3)[zi].double()).float()
        R["g"] = dict(nZ=int(Z.sum()), nZ_T_identity=int(zi.sum()), n_unfilled=int(todo.sum()), J=jstat(c.YH + Tap(c.T, eg.view(-1))),
                      E=_f(energy_at(eg.view(-1), c.ME0)[0]))
        say("g", R["g"])
        newton("g_Zfill_mg_cap4", eg.view(-1), prec_mg, 4, 0.1, 6)
        newton("g_Zfill_bj_exact", eg.view(-1), prec_bj, 2000, 1e-6, 10)
    if "c" in parts or "f" in parts:
        say("명시 행렬 뽑기")
        Hx, ex = extract(c, hv, dev)
        R["extract"] = ex
        say("뽑기", ex)
        if "f" in parts:
            R["f"] = fcheck(c, Hx, grad_at, energy_at, hv, kv, Tap, Xq, dev, nN, n3, s, Z)
        if "c" in parts:
            MGT = TetMG(c, Hx, dev)
            R["tetmg"] = MGT.info
            say("사면체 MG", MGT.info)
            lin("tetmg", MGT.apply)
            lin("tetmg_cap4", MGT.apply, cap=4, tol=0.1)

            def mk():
                Hn, _ = extract(c, hv, dev, quick=True)
                return TetMG(c, Hn, dev).apply
            newton("c_tetmg_cap4", c.E0, None, 4, 0.1, 6, Mbuild=mk)
            newton("c_tetmg_exact", c.E0, None, 2000, 1e-6, 10, Mbuild=mk)
    R["sec"] = time.time() - TW.t0


def extract(c, hv, dev, quick=False):
    """색칠 탐침으로 연산자 hv 의 명시 행렬 (cupyx CSR, n3 x n3). 색 = 노드 격자 좌표 mod 5 (이웃 ±2 까지 구분)."""
    import cupy as cp
    import cupyx.scipy.sparse as csp
    nN, n3 = c.nN, 3 * c.nN
    ids = torch.arange(nN, device=dev)
    i = ids // (c.nb1 * c.nb2); j = (ids // c.nb2) % c.nb1; k = ids % c.nb2
    rows, cols, vals = [], [], []
    hist = torch.zeros(3, device=dev, dtype=torch.long)
    lost = 0.0
    tot = 0.0
    for col in range(125):
        cx, cy, cz = col // 25, (col // 5) % 5, col % 5
        sel = (i % 5 == cx) & (j % 5 == cy) & (k % 5 == cz)
        if not sel.any():
            continue
        dx = ((cx - i + 2) % 5) - 2; dy = ((cy - j + 2) % 5) - 2; dz = ((cz - k + 2) % 5) - 2
        ni, nj, nk = i + dx, j + dy, k + dz
        valid = (ni >= 0) & (ni < c.nb0) & (nj >= 0) & (nj < c.nb1) & (nk >= 0) & (nk < c.nb2)
        ncol = (ni * c.nb1 + nj) * c.nb2 + nk
        dmax = torch.maximum(torch.maximum(dx.abs(), dy.abs()), dz.abs())
        for a in range(3):
            p = torch.zeros(nN, 3, device=dev, dtype=torch.float64)
            p[sel, a] = 1.0
            Ap = hv(p.view(-1)).view(nN, 3)
            nzr = Ap != 0
            tot += float(Ap.abs().sum())
            lost += float(Ap[~valid].abs().sum())
            for b in range(3):
                m = nzr[:, b] & valid
                if not m.any():
                    continue
                mi = ids[m]
                rows.append(3 * mi + b); cols.append(3 * ncol[m] + a); vals.append(Ap[m, b])
                hist += torch.bincount(dmax[m], minlength=3)[:3]
    r_ = torch.cat(rows); c_ = torch.cat(cols); v_ = torch.cat(vals)
    H = csp.coo_matrix((cp.asarray(v_), (cp.asarray(r_), cp.asarray(c_))), shape=(n3, n3)).tocsr()
    H.sum_duplicates()
    info = dict(nnz=int(H.nnz), offs_by_cheb=[int(q) for q in hist.tolist()], lost_rel=lost / max(tot, 1e-300))
    if not quick:
        gen = torch.Generator(device=dev); gen.manual_seed(1)
        errs = []
        for _ in range(3):
            v = torch.randn(n3, device=dev, dtype=torch.float64, generator=gen)
            a1 = hv(v); a2 = torch.as_tensor(H @ cp.asarray(v), device=dev)
            errs.append(float((a1 - a2).norm() / max(float(a1.norm()), 1e-300)))
        info["matvec_rel_err"] = max(errs)
    return H, info


def fcheck(c, H, grad_at, energy_at, hv, kv, Tap, Xq, dev, nN, n3, s, Z):
    """대칭·양정치·유한차분·사면체 기저 독립 참값."""
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as csl
    out = {}
    d = H.diagonal()
    act = cp.nonzero(d != 0)[0]
    Ha = H[act][:, act].tocsr()
    HT = Ha.T.tocsr()
    dif = (Ha - HT)
    nf = float(cp.sqrt((Ha.data.astype(cp.float64) ** 2).sum()))
    out["sym_rel_fro"] = float(cp.sqrt((dif.data.astype(cp.float64) ** 2).sum())) / max(nf, 1e-300) if dif.nnz else 0.0
    out["n_act_dof"] = int(act.size)
    out["diag_min"] = float(d[act].min()); out["diag_neg"] = int((d[act] < 0).sum())
    # 블록 대각 최소 고윳값
    coo = Ha.tocoo()
    r_, c_, v_ = coo.row, coo.col, coo.data
    m = (r_ // 3) == (c_ // 3)
    nb = int(act.size) // 3
    Bd = cp.zeros((nb, 3, 3), dtype=cp.float64)
    Bd[r_[m] // 3, r_[m] % 3, c_[m] % 3] = v_[m]
    try:
        eb = cp.linalg.eigvalsh(0.5 * (Bd + Bd.transpose(0, 2, 1)))
        out["block_eig_min"] = float(eb[:, 0].min()); out["block_neg"] = int((eb[:, 0] <= 0).sum())
    except Exception as e_:
        out["block_eig_err"] = str(e_)[:200]
    Hs = (0.5 * (Ha + HT)).astype(cp.float64).tocsr()
    for wh in ("LA", "SA"):
        try:
            t0 = time.time()
            ev = csl.eigsh(Hs, k=1, which=wh, maxiter=4000, tol=1e-6)
            out["eig_" + wh] = float(ev[0]) if not isinstance(ev, tuple) else float(ev[0][0])
            out["eig_" + wh + "_sec"] = time.time() - t0
        except Exception as e_:
            out["eig_" + wh + "_err"] = str(e_)[:200]
    if "eig_LA" in out and "eig_SA" in out and out["eig_SA"] > 0:
        out["kappa_H"] = out["eig_LA"] / out["eig_SA"]
    # 유한차분: 전체 기울기 vs 에너지, 헤시안 vs 기울기 (시작점 · 휴지 위치 U)
    gen = torch.Generator(device=dev); gen.manual_seed(2)
    mact = torch.zeros(n3, device=dev, dtype=torch.float64); mact[torch.as_tensor(act, device=dev)] = 1.0
    T = c.T

    def fd_at(eta, name):
        o = {}
        g, en, fel = grad_at(eta)
        v = torch.randn(n3, device=dev, dtype=torch.float64, generator=gen) * mact
        v = v / v.abs().max() * 1e-3                                                                  # 노드 변위 최대 1 mm
        gv = float(g @ v)
        best = None
        for eps in (1.0, 0.3, 0.1, 0.03, 0.01):
            Ep, _, _ = energy_at((eta.double() + eps * v).float(), c.ME0)
            Em, _, _ = energy_at((eta.double() - eps * v).float(), c.ME0)
            fd = (Ep - Em) / (2 * eps)
            e = abs(fd - gv) / max(abs(gv), 1e-300)
            best = e if best is None else min(best, e)
            o.setdefault("grad_fd", []).append([eps, fd, gv, e])
        o["grad_fd_best"] = best
        Hv_ = hv(v)
        bestH = None
        for eps in (1.0, 0.3, 0.1, 0.03):
            gp, _, _ = grad_at((eta.double() + eps * v).float())
            gm, _, _ = grad_at((eta.double() - eps * v).float())
            fd = (gp - gm) / (2 * eps)
            e = float((fd - Hv_).norm() / max(float(Hv_.norm()), 1e-300))
            bestH = e if bestH is None else min(bestH, e)
            o.setdefault("hess_fd", []).append([eps, e])
        o["hess_fd_best"] = bestH
        grad_at(eta)
        o["J"] = dict(min=float(torch.linalg.det(Xq(c.YH + Tap(T, eta)))[c.act_q].min()))
        out[name] = o
    fd_at(c.E0, "fd_y0")
    # 휴지 위치 (y = U, 변형 = F_q 그대로): η = T⁻¹ (U − ŷ) (T 가 0 인 노드는 그대로)
    Tp = torch.linalg.pinv(T.double())
    eU = torch.bmm(Tp, (c.U - c.YH).double().view(-1, 3, 1)).view(-1).float()
    fd_at(eU, "fd_U")
    # 사면체 기저 독립 참값 (torch f64): FCR 힘 P = 2μ (X − R) + λ (J − 1) cof X, f_o = Σ_q V_q P g_o -- 실행기 탄성 힘과 대조 (시작점·휴지 위치)
    mu, lam = float(s.mu), float(s.lam)

    def fref(y):
        X = Xq(y).double()
        Uu, S, Vh = torch.linalg.svd(X)
        dU = torch.linalg.det(Uu); dV = torch.linalg.det(Vh)
        Sg = S.clone()
        flip = (dU * dV) < 0
        Uu = Uu.clone(); Uu[flip, :, 2] *= -1
        Sg[flip, 2] *= -1
        Rr = Uu @ Vh
        J = torch.linalg.det(X)
        cof = J[:, None, None] * torch.linalg.inv(X).transpose(1, 2)
        Pk = 2 * mu * (X - Rr) + lam * (J - 1)[:, None, None] * cof
        Pk = Pk * c.Vq.double()[:, None, None]
        Pk[~c.act_q] = 0.0
        fo = torch.einsum("qab,qob->qoa", Pk, c.Gq.double())                                          # (nq, 27, 3)
        fn = torch.zeros(nN, 3, device=dev, dtype=torch.float64)
        fn.index_add_(0, c.nid.reshape(-1).clamp(max=nN - 1), fo.reshape(-1, 3))
        e = (c.Vq.double() * (mu * ((Sg - 1) ** 2).sum(1) + 0.5 * lam * (J - 1) ** 2))[c.act_q].sum()
        return fn.view(-1), float(e)
    for nm, eta in (("y0", c.E0), ("U", eU)):
        g, en, fel = grad_at(eta)
        y = c.YH + Tap(T, eta)
        fr, er = fref(y)
        out["ref_" + nm] = dict(f_rel=float((fel - fr).norm() / max(float(fr.norm()), 1e-300)), E_kernel=en["Eel"], E_ref=er, fnorm=float(fr.norm()))
        # 참 헤시안 (참값 힘 유한차분, f64) 대 실행기 탄성 곱 (투영 헤시안)
        v = torch.randn(n3, device=dev, dtype=torch.float64, generator=gen) * mact
        v = v / v.abs().max() * 1e-4
        fp, _ = fref(y.double() + v); fm, _ = fref(y.double() - v)
        Kt = (fp - fm) / 2
        Kk = kv(v)
        out["ref_" + nm]["K_rel"] = float((Kk - Kt).norm() / max(float(Kt.norm()), 1e-300))
        # 뒤집힌 점이 닿지 않는 노드로만
        bad = torch.zeros(nN, device=dev, dtype=torch.bool)
        Jy = torch.linalg.det(Xq(y))
        qb = c.act_q & (Jy <= 0.2)
        bad[c.nid[qb].reshape(-1).clamp(max=nN - 1)] = True
        w = v.view(nN, 3).clone(); w[bad] = 0.0; w = w.view(-1)
        if float(w.abs().max()) > 0:
            fp, _ = fref(y.double() + w); fm, _ = fref(y.double() - w)
            Kt2 = (fp - fm) / 2; Kk2 = kv(w)
            out["ref_" + nm]["K_rel_noninv"] = float((Kk2 - Kt2).norm() / max(float(Kt2.norm()), 1e-300))
        out["ref_" + nm]["n_q_Jle02"] = int(qb.sum())
    grad_at(c.E0)
    say("f", {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("grad_fd", "hess_fd")}) for k, v in out.items()})
    return out


class TetMG:
    """진단용 사면체 MG: 거친 층 = 꼭짓점 맞춘 격자 (간격 2^l), 삼선형 전달 P, 갈러킨 A_l = Pᵀ A_{l−1} P, 블록 야코비 평활 (ω = 4 / (3 ρ(D⁻¹A))) V(2,2), 가장 거친 층 직접 (유사역)."""

    def __init__(s_, c, H, dev, levels=4, nu=2):
        import cupy as cp
        import cupyx.scipy.sparse as csp
        s_.cp, s_.csp, s_.dev, s_.nu = cp, csp, dev, nu
        n3 = H.shape[0]
        d = H.diagonal()
        act = cp.nonzero(d != 0)[0]
        s_.n3, s_.act = n3, act
        A = H[act][:, act].tocsr().astype(cp.float64)
        A = (0.5 * (A + A.T)).tocsr()
        nodes = cp.unique(act // 3)
        # 노드 좌표 (고운 층)
        ids = nodes
        i = ids // (c.nb1 * c.nb2); j = (ids // c.nb2) % c.nb1; k = ids % c.nb2
        coords = cp.stack([i, j, k], 1)
        s_.L = [dict(A=A)]
        info = dict(levels=[])
        dofmap = act                                                                                  # 이 층 dof -> (노드 위치 순서)
        nodepos = {}
        # dof -> (노드 번호 in 'nodes', 성분)
        node_of = cp.searchsorted(nodes, act // 3)
        comp = act % 3
        crd = coords
        nod_of_dof, comp_of_dof = node_of, comp
        for l in range(1, levels):
            # 거친 노드 = 좌표 // 2 (짝수는 하나, 홀수는 양쪽 둘): 삼선형 가중
            cc = []
            ww = []
            for di in (0, 1):
                for dj in (0, 1):
                    for dk in (0, 1):
                        ci = cp.where(crd[:, 0] % 2 == 0, crd[:, 0] // 2, crd[:, 0] // 2 + di)
                        cj = cp.where(crd[:, 1] % 2 == 0, crd[:, 1] // 2, crd[:, 1] // 2 + dj)
                        ck = cp.where(crd[:, 2] % 2 == 0, crd[:, 2] // 2, crd[:, 2] // 2 + dk)
                        wi = cp.where(crd[:, 0] % 2 == 0, (di == 0) * 1.0, 0.5)
                        wj = cp.where(crd[:, 1] % 2 == 0, (dj == 0) * 1.0, 0.5)
                        wk = cp.where(crd[:, 2] % 2 == 0, (dk == 0) * 1.0, 0.5)
                        cc.append(cp.stack([ci, cj, ck], 1)); ww.append(wi * wj * wk)
            CC = cp.stack(cc, 1).reshape(-1, 3); WW = cp.stack(ww, 1).reshape(-1)
            nf = crd.shape[0]
            fidx = cp.repeat(cp.arange(nf), 8)
            keep = WW > 0
            CC, WW, fidx = CC[keep], WW[keep], fidx[keep]
            mx = CC.max(0) + 1
            key = (CC[:, 0] * mx[1] + CC[:, 1]) * mx[2] + CC[:, 2]
            ukey, cidx = cp.unique(key, return_inverse=True)
            ncn = int(ukey.size)
            # dof 단위 P: 고운 dof (노드 f, 성분 a) -> 거친 dof (노드 cidx, 같은 성분)
            ndf = int(nod_of_dof.size)
            # 고운 노드 f 의 dof 목록: 각 dof 마다 그 노드의 거친 몫 전부
            order = cp.argsort(fidx)
            fidx_s, cidx_s, WW_s = fidx[order], cidx[order], WW[order]
            starts = cp.searchsorted(fidx_s, cp.arange(nf))
            ends = cp.searchsorted(fidx_s, cp.arange(nf), side="right")
            cnt = ends - starts
            rep = cnt[nod_of_dof]
            dof_rows = cp.repeat(cp.arange(ndf), rep.get().tolist() if False else rep)
            # 각 dof 행의 몫 위치
            offs = cp.arange(int(rep.sum())) - cp.repeat(cp.cumsum(rep) - rep, rep)
            src = cp.repeat(starts[nod_of_dof], rep) + offs
            pc = cidx_s[src] * 3 + cp.repeat(comp_of_dof, rep)
            pw = WW_s[src]
            Pm = csp.coo_matrix((pw.astype(cp.float64), (dof_rows, pc)), shape=(ndf, ncn * 3)).tocsr()
            # 빈 거친 dof 빼기
            used = cp.unique(pc)
            Pm = Pm[:, used].tocsr()
            Ac = (Pm.T @ s_.L[-1]["A"] @ Pm).tocsr()
            s_.L[-1]["P"] = Pm
            s_.L.append(dict(A=Ac))
            # 다음 층 좌표·dof 정보
            cn_of_used = used // 3
            nod_of_dof = cp.searchsorted(cp.unique(cn_of_used), cn_of_used)
            comp_of_dof = used % 3
            ucn = cp.unique(cn_of_used)
            kz = ukey[ucn]
            crd = cp.stack([kz // (mx[1] * mx[2]), (kz // mx[2]) % mx[1], kz % mx[2]], 1)
            info["levels"].append(dict(ndof=int(Ac.shape[0]), nnz=int(Ac.nnz)))
        # 평활기 (블록 야코비) · ω
        for l, Lv in enumerate(s_.L):
            A = Lv["A"]
            n = A.shape[0]
            coo = A.tocoo()
            # 블록 = 연속 3 dof 가 한 노드라는 가정이 거친 층에서도 맞도록 dof 순서는 (노드, 성분) 정렬 -- 고운 층은 act 오름차순, 거친 층은 used 오름차순
            r_, c_, v_ = coo.row, coo.col, coo.data
            nb_ = n // 3
            m = (r_ // 3) == (c_ // 3)
            Bd = cp.zeros((nb_, 3, 3), dtype=cp.float64)
            Bd[r_[m] // 3, r_[m] % 3, c_[m] % 3] = v_[m]
            Bd = Bd + cp.eye(3)[None] * 1e-30
            Di = cp.linalg.inv(Bd)
            Lv["Di"] = Di
            x = cp.random.RandomState(3).standard_normal(n)
            lam = 1.0
            for _ in range(30):
                y = s_.bj(Di, A @ x)
                lam = float(cp.linalg.norm(y) / max(float(cp.linalg.norm(x)), 1e-300))
                x = y / max(float(cp.linalg.norm(y)), 1e-300)
            Lv["w"] = 4.0 / (3.0 * lam)
            info.setdefault("rho_DinvA", []).append(lam)
        Ad = s_.L[-1]["A"].toarray()
        ew, ev = cp.linalg.eigh(0.5 * (Ad + Ad.T))
        tol_ = float(ew.max()) * 1e-10
        inv = cp.where(ew > tol_, 1.0 / ew, 0.0)
        s_.L[-1]["Cinv"] = (ev * inv[None, :]) @ ev.T
        info["coarse_eig_min"] = float(ew.min()); info["coarse_n"] = int(Ad.shape[0])
        info["fine_ndof"] = int(n3 if False else s_.L[0]["A"].shape[0])
        s_.info = info

    def bj(s_, Di, r):
        cp = s_.cp
        return cp.einsum("nab,nb->na", Di, r.reshape(-1, 3)).reshape(-1)

    def vc(s_, l, r):
        Lv = s_.L[l]
        if l == len(s_.L) - 1:
            return Lv["Cinv"] @ r
        A, Di, w = Lv["A"], Lv["Di"], Lv["w"]
        x = w * s_.bj(Di, r)
        for _ in range(s_.nu - 1):
            x = x + w * s_.bj(Di, r - A @ x)
        rc = Lv["P"].T @ (r - A @ x)
        x = x + Lv["P"] @ s_.vc(l + 1, rc)
        for _ in range(s_.nu):
            x = x + w * s_.bj(Di, r - A @ x)
        return x

    def apply(s_, r):
        cp = s_.cp
        rr = cp.asarray(r.contiguous())
        z = cp.zeros(s_.n3, dtype=cp.float64)
        z[s_.act] = s_.vc(0, rr[s_.act])
        return torch.as_tensor(z, device=s_.dev)
