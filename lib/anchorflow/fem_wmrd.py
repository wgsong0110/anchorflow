"""절 CM38 진단 (수박 재설계, 결과 불변 -- 그래프 밖 한 프레임을 같은 커널 순서로 돌리며 읽기만).

무너진 걸음 (‖g‖²/F² > 1e6) 의 원인 가르기: 뉴턴 반복마다
  - 잔차 분해: g = M η/τ² (관성) + T f (탄성·장벽); F = 걸음 첫 반복 ‖T f‖ (판정 분모), 관성 척도 W = 걸음 첫 반복 ‖M η/τ²‖.
  - 선형계: PCG 끝 상대 잔차 ‖r‖/‖r0‖ (Jacobi 전처리, --cg 회).
  - 뉴턴 방향 δ (칸 단위): 최대·99 %·1 칸 넘는 노드 수·질량 몫, 최대 노드의 질량 / 질량 중앙값·대각 강성 / (m/τ²)·|g_n|.
  - 접선 일관성: 요소 커널 (재질 RM 포함 힘) 의 δ 방향 중심 차분 −(f(y+εδ) − f(y−εδ))/2ε 대 투영 접선 곱 K δ (ao_mv_q) 상대 오차·cos,
    온 걸음 할선 −(f(y+δ) − f(y)) 대 K δ (걸음 크기에 대한 비선형), y+δ 에서 비유한 힘 노드 수.
  - 실제 보폭: 다음 반복 시작 Δy 와 δ 의 비 (노드 중앙·최소·전체 노름비).
상태 (f·B·H) 는 대조 뒤 되돌림. 결정적 판 아님 (비결정 판만)."""
import json
import math

import numpy as np
import torch

from . import fem_ao_rt as AORT

P = AORT.P


def _n(t):
    return float(t.double().norm())


class Rec:
    def __init__(s, S, out, fd=True):
        s.S, s.fo, s.fd = S, open(out, "a"), fd
        s.on, s.fr = False, -1
        s.orig = S.k                                                                    # 인스턴스 감싸기 (클래스 k -- --sp2 가로채기 포함 -- 를 그대로 부름)
        S.k = s.k

    def k(s, name, n, args, b=128):
        S = s.S
        s.orig(name, n, args, b)
        if not s.on:
            return
        try:
            if name in ("ao_step0_ga", "rt_step0"):
                s.ks += 1; s.it = -1; s.y_prev = None; s.W = None; s.F = None; s.cur = None
            elif name in ("rt_grad", "rt_grad_c"):
                s.close()
                torch.cuda.synchronize()
                if int(S.CT[0]) == 0:                                                   # 허용치·--xit_from 으로 멈춘 반복은 커널이 건너뜀 (버퍼 낡음)
                    s.grad(args)
            elif name == "rt_ctl" and int(args[0]) == 7:
                s.pcg_end()
            elif name.startswith("ao_stepend") or name in ("rt_stepend", "en_stepend"):
                s.close()
        except Exception as e:                                                          # 진단 실패는 결과에 영향 없음
            s.fo.write(json.dumps(dict(fr=s.fr, ks=s.ks, it=s.it, err=repr(e)[:300])) + "\n")

    def begin(s, fr):
        s.fr, s.ks, s.it, s.on = fr, -1, -1, True
        s.cur, s.y_prev, s.d_prev = None, None, None

    def close(s):
        """앞 반복의 실제 보폭 (지금 y − 반복 시작 y 대 δ) 을 붙여 내보냄."""
        S = s.S
        if s.cur is None:
            return
        if s.y_prev is not None and s.d_prev is not None:
            torch.cuda.synchronize()
            nn = s.y_prev.shape[0]
            dy = (S.y.reshape(-1, 3)[:nn] - s.y_prev).double()
            d = s.d_prev
            dn = d.norm(dim=-1); m_ = dn > 1e-12 * max(float(dn.max()), 1e-30)
            if bool(m_.any()):
                rat = dy.norm(dim=-1)[m_] / dn[m_]
                s.cur.update(step_norm=_n(dy) / max(_n(d), 1e-300), step_med=float(rat.median()), step_min=float(rat.min()),
                             step_frac_lt1=float((rat < 0.999).double().mean()))
        s.emit()

    def end(s):
        s.on = False
        s.fo.flush()

    def grad(s, args):
        S = s.S
        torch.cuda.synchronize()
        nn = int(S.IP[9])
        if nn <= 0:
            return
        s.it += 1
        s.d_prev = None
        it2 = float(args[2])
        mn = S.mn[:nn].double()
        eta = S.eta.reshape(-1, 3)[:nn].double()
        g = -S.g0buf.reshape(-1, 3)[:nn].double()
        gI = mn[:, None] * it2 * eta
        gF = g - gI
        cell = float(S.A[:, 0].norm())
        if s.it == 0:
            s.W = _n(gI); s.F = _n(gF)
        md = float(mn[mn > 0].median()) if bool((mn > 0).any()) else 1.0
        gn = g.norm(dim=-1)
        j = int(gn.argmax())
        s.cur = dict(fr=s.fr, ks=s.ks, it=s.it, nn=nn, tau=1.0 / math.sqrt(it2), cell=cell, g=_n(g), gI=_n(gI), gF=_n(gF), F0=s.F, W0=s.W,
                     r_F=_n(g) / max(s.F, 1e-300), r_W=_n(g) / max(s.W, 1e-300), SC1=float(S.SC[1]), SC56=float(S.SC[56]),
                     gmax=float(gn[j]), gmax_mrel=float(mn[j]) / md, light=int(((mn > 0) & (mn < 1e-2 * md)).sum()),
                     g_light=_n(g[(mn < 1e-2 * md)]) / max(_n(g), 1e-300))
        if s.it == 0:                                                                  # 걸음 시작: η 가 어디서 오나 (관성 예측 ŷ − U, 시작점 y − U, 노드 속도·가속도 기억)
            U = S.U.reshape(-1, 3)[:nn].double()
            dt = float(S.dt)
            def st_(v, sc):
                vn = v.norm(dim=-1) / sc
                fin = torch.isfinite(vn)
                if not bool(fin.any()):
                    return dict(nonfin=int((~fin).sum()))
                vv = torch.where(fin, vn, torch.zeros_like(vn)); j_ = int(vv.argmax())
                return dict(max=float(vv[j_]), q99=float(torch.quantile(vv.float()[fin][:16000000], 0.99)), gt1=int((vv > 1).sum()),
                            gt1_mass=float(mn[vv > 1].sum() / mn.sum().clamp_min(1e-300)), arg_mrel=float(mn[j_]) / md, nonfin=int((~fin).sum()))
            yh = S.yh.reshape(-1, 3)[:nn].double(); y = S.y.reshape(-1, 3)[:nn].double()
            s.cur["start"] = dict(eta=st_(eta, cell), yhU=st_(yh - U, cell), yU=st_(y - U, cell), vdt=st_(S.vk.reshape(-1, 3)[:nn].double() * dt, cell),
                                  adt2=st_(S.gaA.reshape(-1, 3)[:nn].double() * dt * dt, cell) if getattr(S, "gaA", None) is not None else None,
                                  mmin_rel=float(mn[mn > 0].min()) / md if bool((mn > 0).any()) else 0.0, dt=dt)
            sk = S.sk[:nn].double(); pm = S.pm[:nn]; en = eta.norm(dim=-1) / cell; je = int(torch.where(torch.isfinite(en), en, torch.zeros_like(en)).argmax())
            bigE = en > 1
            s.cur["start"]["eta_arg"] = dict(sk=float(sk[je]), pm=int(pm[je]), r=float(S.an[je] / S.mqn[je]) if float(S.mqn[je]) > 0 else -1.0)
            s.cur["start"]["eta_big_sk"] = dict(n=int(bigE.sum()), sk_lt1e2=int((bigE & (sk < 1e-2)).sum()), sk_lt1=int((bigE & (sk < 0.999)).sum()), pm0=int((bigE & (pm == 0)).sum()))
            s.cur["start"]["band"] = dict(n_sk_lt1=int((sk < 0.999).sum()), n_sk_lt1e2=int((sk < 1e-2).sum()), n_sk_lt1e6=int((sk < 1e-6).sum()))
            if s.ks == 0:                                                              # 프레임 시작 입자 속도·가속도
                pv = S.v.reshape(-1, 3).double().norm(dim=-1)
                s.cur["start"]["pv_max"] = float(pv[torch.isfinite(pv)].max())
                if getattr(S, "pa", None) is not None:
                    pa = S.pa.reshape(-1, 3).double().norm(dim=-1)
                    s.cur["start"]["pa_max"] = float(pa[torch.isfinite(pa)].max())
        s.y_prev = S.y.reshape(-1, 3)[:nn].clone()

    def pcg_end(s):
        S = s.S
        torch.cuda.synchronize()
        nn = int(S.IP[9])
        if nn <= 0 or s.cur is None:
            return
        skip = int(S.CT[0])
        c = s.cur
        c["skip"] = skip
        if skip:
            s.d_prev = torch.zeros(nn, 3, dtype=torch.float64, device=S.y.device)
            return
        d = S.dl.reshape(-1, 3)[:nn].double()
        r = S.r.reshape(-1, 3)[:nn].double(); r0 = S.g0buf.reshape(-1, 3)[:nn].double()
        c["lin_rel"] = _n(r) / max(_n(r0), 1e-300)
        cell = c["cell"]
        mn = S.mn[:nn].double(); md = float(mn[mn > 0].median()) if bool((mn > 0).any()) else 1.0
        dn = d.norm(dim=-1) / cell
        c["d_max"] = float(dn.max()); c["d_q99"] = float(torch.quantile(dn.float()[:min(nn, 16000000)], 0.99)) if nn > 0 else 0.0
        big = dn > 1.0
        c["d_gt1"] = int(big.sum()); c["d_gt1_mass"] = float(mn[big].sum() / mn.sum().clamp_min(1e-300))
        j = int(dn.argmax())
        Bn = S.B.reshape(-1, 9)[:nn]
        it2 = 1.0 / c["tau"] ** 2
        c["d_argmax"] = dict(mrel=float(mn[j]) / md, Bdiag=float((Bn[j, 0] + Bn[j, 4] + Bn[j, 8]) / 3), Mi=float(mn[j]) * it2,
                             g=float(-S.g0buf.reshape(-1, 3)[j].double().norm()))
        s.d_prev = d.clone()
        if s.fd and bool(torch.isfinite(d).all()):
            s.fdchk(d, nn, c)

    def fdchk(s, d, nn, c):
        S = s.S
        bk = (S.f.clone(), S.B.clone(), S.H.clone())
        try:
            dd = torch.zeros(S.capN, 3, device=S.dev, dtype=torch.float32)
            dd[:nn] = d.float()
            y0 = S.y.clone()
            f0, _ = S._elem_at(y0, "aq")                                                 # 같은 y 의 힘·H (풀이가 쓴 것과 같음)
            Kd = S._mv_at(dd).double()
            dm = float(d.norm(dim=-1).max())
            for tag, ep in (("e3", 1e-3 * c["cell"] / max(dm, 1e-30)), ("e5", 1e-5 * c["cell"] / max(dm, 1e-30))):
                fp, _ = S._elem_at((y0 + ep * dd.reshape(y0.shape)).contiguous(), "aq")
                fm, _ = S._elem_at((y0 - ep * dd.reshape(y0.shape)).contiguous(), "aq")
                jd = -(fp.double() - fm.double()) / (2 * ep)
                ok = torch.isfinite(jd).all(-1)
                c["fd_" + tag] = dict(rel=_n(Kd[ok] - jd[ok]) / max(_n(jd[ok]), 1e-300), cos=float((Kd[ok] * jd[ok]).sum() / max(_n(Kd[ok]) * _n(jd[ok]), 1e-300)),
                                      dKd=float((d * Kd).sum()), dJd=float((d[ok] * jd[ok]).sum()), nonfin=int((~ok).sum()))
            fp, _ = S._elem_at((y0 + dd.reshape(y0.shape)).contiguous(), "aq")         # 온 걸음 할선
            js = -(fp.double() - f0.double())
            ok = torch.isfinite(js).all(-1)
            c["sec"] = dict(rel=_n(Kd[ok] - js[ok]) / max(_n(js[ok]), 1e-300), dKd=float((d * Kd).sum()), dJs=float((d[ok] * js[ok]).sum()),
                            nonfin=int((~ok).sum()), Ks=_n(js[ok]), Kd=_n(Kd))
        finally:
            S.f.copy_(bk[0]); S.B.copy_(bk[1]); S.H.copy_(bk[2])
            torch.cuda.synchronize()

    def emit(s):
        if s.cur is not None:
            s.fo.write(json.dumps(s.cur) + "\n")
        s.cur = None

    def frame(s, fi, t):
        """그래프 밖 한 프레임 (record 같은 순서). 넘치면 되돌리고 False (호출자가 S.step 으로)."""
        S = s.S
        bk = S._bk()
        if S.graph is None:
            S.capture(); S._rs(bk)
        S.set_bc(t)
        s.begin(fi + 1)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); S.record(t, S.Kcur); e1.record()
        torch.cuda.synchronize()
        s.close()
        s.fo.write(json.dumps(dict(fr=fi + 1, frame_end=1, K=int(S.Kcur), ufg=[float(q) for q in S.ufg[:S.Kcur]] if getattr(S, "ufg", None) is not None else None,
                                   ufl=[int(q) for q in S.ufl[:S.Kcur]] if getattr(S, "ufl", None) is not None else None)) + "\n")
        s.end()
        if int(S.IP[13]) != 0:
            S._rs(bk)
            return False
        S.last_ms = e0.elapsed_time(e1)
        return True
