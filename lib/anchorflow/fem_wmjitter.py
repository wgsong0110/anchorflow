"""절 CM90 묶음 wmjitter (사용자 11-09 직접 요청): H9 수박이 바닥에 놓인 뒤 계속 꿈틀거림 -- 물리 탄성 진동인지 수치 흔들림인지 가르는 진단 기록 (풀이 결과 불변, 읽기만).

가벼운 판 (exe/s6 실행기 사본, fem_speed6.Defer, '--s6l 1') 의 Defer.keep · Defer.finish 를 감쌈 (감싸개 exe/wmjitter_wrap.py '--wj 1').
keep 뒤 (프레임 끝, 쪼개기 함수가 돌아온 뒤):
  시뮬 스트림 = 원 배열 장치 사본만 (입자 F, 가우스 점 F_q·V_q·칸 좌표 cb·IP, 걸음 ufl·ufg·ufs, 시간 예산 전역 TB_CUT·TB_GATES·TB_T0·TB_TE).
  셈 = Defer 복사 스트림 (호스트 동기화 없음):
    입자 (Defer 가 이미 복사한 x·v 사본 + F 사본): 운동 에너지 KE, 운동량, 질량중심, 각운동량·관성 -> 강체 병진·회전 몫과 변형 몫 KE_def = KE − KE_t − KE_r,
      강체 운동을 뺀 속도 RMS (질량 가중), SNH 탄성 에너지 (실행기 --erec 와 같은 식), J 중앙·최소, 프레임 간 ΔJ RMS·평균|ΔJ|·|ΔJ| > 0.05 수.
    가우스 점 (칸 절대 좌표 IP[0:3] + cb 와 팔분 번호로 프레임 사이 같은 점을 맞춤): 점 수, J_q ≤ 0 수·최소·평균, 맞춘 점 수, ΔJ_q RMS·평균·최대·> 0.05·> 0.2 수.
    부분 표본 시계열: 입자 J (원래 순번 sub 마다 하나), 가우스 점 J_q (프레임 f0+1 에 있던 점을 키 순으로 gsub 마다 하나, 이후 프레임에서 같은 키를 찾음).
finish 뒤: Defer.X (프레임 끝 입자 위치, 원래 순서) 로 프레임 간 가속도 a = x(n+1) − 2 x(n) + x(n−1) 의 몸 전체 RMS·분위 (질량중심 가속도 뺀 것 포함), 질량중심,
  부분 표본 위치 -> <out>/wj.npz, <out>/wjinfo.json.
"""
import json
import os

import numpy as np
import torch

C = dict(out=None, sub=64, gsub=8, f0=29, on=False)
_ST = {}
COLS = ["KE", "Px", "Py", "Pz", "M", "KEt", "KEr", "KEdef", "vdef_rms", "cx", "cy", "cz", "wx", "wy", "wz", "Eel", "Jmed", "Jmin", "dJrms", "dJmean", "dJn05", "nfin",
        "Gn", "Gle0", "Gmin", "Gmean", "Gmatch", "GdJrms", "GdJmean", "GdJmax", "GdJn05", "GdJn20"]
ACOLS = ["arms", "arms_rel", "a50", "a90", "a99", "amax", "acx", "acy", "acz", "nfin"]
KC = 16
I64MAX = torch.iinfo(torch.int64).max


def _det(F):
    a, b, c, d, e, f, g, h, i = F.unbind(-1)
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _inv3(A):
    """3x3 역행렬 (수반 행렬 / det) -- linalg 의 오류 검사 동기화를 피함."""
    a, b, c, d, e, f, g, h, i = A.reshape(9).unbind(-1)
    D = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    adj = torch.stack([e * i - f * h, c * h - b * i, b * f - c * e, f * g - d * i, a * i - c * g, c * d - a * f, d * h - e * g, b * g - a * h, a * e - b * d]).reshape(3, 3)
    return adj / D


class Rec:
    def __init__(s, D, S):
        dev = S.x.device
        s.dev = dev
        s.L, s.Np = D.L, int(S.x.shape[0])
        s.m = S.m.double()
        s.vol = S.vol.double()
        s.mu, s.lam = float(S.mu), float(S.lam)
        perm = S.perm.long()
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(s.Np, device=dev)
        s.perm_ref = S.perm
        s.sub_o = torch.arange(0, s.Np, int(C["sub"]), device=dev)
        s.sub_i = inv[s.sub_o]
        s.T = torch.full((s.L, len(COLS)), float("nan"), device=dev, dtype=torch.float64)
        s.JS = torch.full((s.L, s.sub_o.numel()), float("nan"), device=dev, dtype=torch.float32)
        s.UF = torch.full((s.L, 3, KC), float("nan"), device=dev, dtype=torch.float64)
        s.TBv = torch.zeros((s.L, 4), device=dev, dtype=torch.int64)
        s.Jp = None
        s.PK = s.PJ = None
        s.SK = None
        s.GS = None
        s.tbp = None
        s.err = []
        s.nperm_change = 0
        s.capq = []
        try:
            from . import fem_speed3 as S3
            if S3.TB.mod is not None:
                s.tbp = [int(S3.TB.mod.get_global(nm).ptr) for nm in ("TB_CUT", "TB_GATES", "TB_T0", "TB_TE")]
        except Exception as ex:                                                          # 예산 모듈 없음 (예산 끈 판도 모듈은 있음) -- 기록만 빠짐
            s.err.append(f"tb {ex!r}")

    def frame(s, D, fi, S, k):
        cur = torch.cuda.current_stream()
        # ---- 시뮬 스트림: 사본만
        Fs = S.F.reshape(-1, 9).clone()
        nq_cap = int(S.Fq.shape[0])
        Fq = S.Fq.reshape(nq_cap, 9).clone(); Vq = S.Vq[:nq_cap].clone(); cb = S.cb.clone(); IP = S.IP.clone().long()
        uf = []
        for j, nm in enumerate(("ufl", "ufg", "ufs")):
            t = getattr(S, nm, None)
            if t is not None:
                n = min(int(t.numel()), KC)
                s.UF[fi, j, :n] = t[:n].double()
        if s.tbp is None:
            try:
                from . import fem_speed3 as S3
                if S3.TB.mod is not None:
                    s.tbp = [int(S3.TB.mod.get_global(nm).ptr) for nm in ("TB_CUT", "TB_GATES", "TB_T0", "TB_TE")]
            except Exception:
                pass
        if s.tbp is not None:
            import cupy as cp
            base = s.TBv.data_ptr() + fi * 4 * 8
            for j, (p, nb) in enumerate(zip(s.tbp, (4, 4, 8, 8))):
                cp.cuda.runtime.memcpyAsync(base + 8 * j, p, nb, cp.cuda.runtime.memcpyDeviceToDevice, cur.cuda_stream)
        if S.perm is not s.perm_ref:                                                   # 결정적 판만 프레임마다 정렬 (이 묶음은 비결정 판) -- 바뀌면 ΔJ 기준을 버림
            s.nperm_change += 1
            perm = S.perm.long(); inv = torch.empty_like(perm); inv[perm] = torch.arange(s.Np, device=s.dev)
            s.sub_i = inv[s.sub_o]; s.perm_ref = S.perm; s.Jp = None
        ev = torch.cuda.Event(); ev.record(cur)
        cs = D.cs
        cs.wait_event(ev)
        for t in (Fs, Fq, Vq, cb, IP):
            t.record_stream(cs)
        with torch.cuda.stream(cs):
            T = s.T[fi]
            x = D.XR[k].double(); v = D.VR[k].double()
            okp = torch.isfinite(x).all(-1) & torch.isfinite(v).all(-1)
            m = torch.where(okp, s.m, torch.zeros_like(s.m))
            x = torch.where(okp[:, None], x, torch.zeros_like(x)); v = torch.where(okp[:, None], v, torch.zeros_like(v))
            M = m.sum()
            P = (m[:, None] * v).sum(0)
            c = (m[:, None] * x).sum(0) / M
            KE = 0.5 * (m * (v * v).sum(-1)).sum()
            r = x - c
            Lm = (m[:, None] * torch.linalg.cross(r, v)).sum(0)
            rr = (r * r).sum(-1)
            I = torch.eye(3, device=s.dev, dtype=torch.float64) * (m * rr).sum() - (m[:, None] * r).T @ r
            w = _inv3(I) @ Lm
            Vc = P / M
            KEt = 0.5 * (P * P).sum() / M
            KEr = 0.5 * (w * Lm).sum()
            vd = v - Vc - torch.linalg.cross(w.expand_as(r), r)
            T[0] = KE; T[1:4] = P; T[4] = M; T[5] = KEt; T[6] = KEr; T[7] = KE - KEt - KEr
            T[8] = torch.sqrt((m * (vd * vd).sum(-1)).sum() / M)
            T[9:12] = c; T[12:15] = w
            F = Fs.double()
            J = _det(F)
            ok = torch.isfinite(J) & okp
            Ic = (F * F).sum(-1)
            psi = 0.5 * s.mu * (Ic - 3.0) - s.mu * (J - 1.0) + 0.5 * (s.lam + s.mu) * (J - 1.0) ** 2
            okE = ok & torch.isfinite(psi)
            T[15] = torch.where(okE, s.vol * psi, torch.zeros_like(psi)).sum()
            Jn = torch.where(ok, J, torch.full_like(J, float("nan")))
            T[16] = Jn.nanmedian(); T[17] = torch.where(ok, J, torch.full_like(J, float("inf"))).min()
            if s.Jp is not None:
                okd = ok & torch.isfinite(s.Jp)
                dJ = torch.where(okd, J - s.Jp, torch.zeros_like(J))
                nd = okd.sum().double()
                T[18] = torch.sqrt((dJ * dJ).sum() / nd); T[19] = dJ.abs().sum() / nd; T[20] = (dJ.abs() > 0.05).sum().double()
            T[21] = ok.sum().double()
            s.Jp = Jn
            s.JS[fi] = Jn[s.sub_i].float()
            # ---- 가우스 점
            q = torch.arange(nq_cap, device=s.dev)
            valid = (q < IP[12]) & (Vq > 0)
            cc = (q // 8).clamp(max=cb.shape[0] - 1)
            ax = cb[cc].long() + IP[0:3][None, :] + (1 << 17)
            key = ((ax[:, 0] * (1 << 18) + ax[:, 1]) * (1 << 18) + ax[:, 2]) * 8 + (q % 8)
            Jq = _det(Fq.double())
            valid = valid & torch.isfinite(Jq)
            key = torch.where(valid, key, torch.full_like(key, I64MAX))
            sk, si = key.sort()
            sj = Jq[si]
            sv = sk != I64MAX
            nv = sv.sum().double()
            T[22] = nv; T[23] = (sv & (sj <= 0)).sum().double()
            T[24] = torch.where(sv, sj, torch.full_like(sj, float("inf"))).min(); T[25] = torch.where(sv, sj, torch.zeros_like(sj)).sum() / nv
            if s.PK is not None:
                ix = torch.searchsorted(s.PK, sk).clamp(max=s.PK.numel() - 1)
                mt = sv & (s.PK[ix] == sk)
                d = torch.where(mt, sj - s.PJ[ix], torch.zeros_like(sj))
                nm_ = mt.sum().double()
                T[26] = nm_; T[27] = torch.sqrt((d * d).sum() / nm_); T[28] = d.abs().sum() / nm_; T[29] = d.abs().max()
                T[30] = (d.abs() > 0.05).sum().double(); T[31] = (d.abs() > 0.2).sum().double()
            s.PK, s.PJ = sk, sj
            if fi == int(C["f0"]):
                s.SK = sk[::int(C["gsub"])].clone()
                s.GS = torch.full((s.L, s.SK.numel()), float("nan"), device=s.dev, dtype=torch.float32)
            if s.SK is not None:
                ix = torch.searchsorted(sk, s.SK).clamp(max=sk.numel() - 1)
                mt = (sk[ix] == s.SK) & (s.SK != I64MAX)
                s.GS[fi] = torch.where(mt, sj[ix], torch.full_like(sj[ix], float("nan"))).float()
            e2 = torch.cuda.Event(); e2.record(cs)
        D.evd[k] = e2                                                                      # Defer 의 세 칸 고리: 다음에 이 칸을 덮기 전에 이 셈까지 기다림
        s.capq.append(nq_cap)

    def finish(s, D, h):
        torch.cuda.synchronize(); D.cs.synchronize()
        n = int(D.n)
        out = C["out"]
        m_o = torch.empty_like(s.m); m_o[s.perm_ref.long()] = s.m
        A = np.full((n, len(ACOLS)), np.nan)
        COM = np.full((n, 3), np.nan)
        Xd = lambda i: D.X[i].to(s.dev).double()
        prev, cu = None, Xd(0)
        for fi in range(n):
            ok0 = torch.isfinite(cu).all(-1)
            mm = torch.where(ok0, m_o, torch.zeros_like(m_o))
            COM[fi] = ((mm[:, None] * torch.where(ok0[:, None], cu, torch.zeros_like(cu))).sum(0) / mm.sum()).cpu().numpy()
            nx = Xd(fi + 1) if fi + 1 < n else None
            if prev is not None and nx is not None:
                a = nx - 2.0 * cu + prev
                ok = torch.isfinite(a).all(-1)
                a = a[ok]; mo = m_o[ok]
                an = a.norm(dim=-1)
                ac = (mo[:, None] * a).sum(0) / mo.sum()
                ar = (a - ac).norm(dim=-1)
                qs = torch.quantile(an.float(), torch.tensor([0.5, 0.9, 0.99], device=s.dev)).double()
                A[fi] = [float(torch.sqrt((an * an).mean())), float(torch.sqrt((ar * ar).mean())), float(qs[0]), float(qs[1]), float(qs[2]), float(an.max()),
                         float(ac[0]), float(ac[1]), float(ac[2]), float(ok.sum())]
            prev, cu = cu, nx
        so = s.sub_o.cpu().numpy()
        XS = D.X[:n][:, so].numpy().astype(np.float32)
        z = dict(T=s.T[:n].cpu().numpy(), cols=np.array(COLS), A=A, acols=np.array(ACOLS), COM=COM, XS=XS, sub_o=so, JS=s.JS[:n].cpu().numpy(),
                 UF=s.UF[:n].cpu().numpy(), TB=s.TBv[:n].cpu().numpy(), h=np.array([h]), K=np.array([q.get("K") or -1 for q in D.meta[:n]]),
                 ms=np.array([q.get("ms") for q in D.meta[:n]], dtype=np.float64))
        if s.GS is not None:
            vk = (s.SK != I64MAX).cpu().numpy()
            z["GS"] = s.GS[:n].cpu().numpy()[:, vk]
        np.savez_compressed(os.path.join(out, "wj.npz"), **z)
        info = dict(n=n, Np=s.Np, sub=int(C["sub"]), nsub=int(so.size), gsub=int(C["gsub"]), f0=int(C["f0"]) + 1, nG=int(z["GS"].shape[1]) if "GS" in z else 0,
                    tb=s.tbp is not None, perm_change=s.nperm_change, capq=[min(s.capq), max(s.capq)] if s.capq else None, err=s.err,
                    tries=[q.get("tries") for q in D.meta[:n]], ok=True)
        json.dump(info, open(os.path.join(out, "wjinfo.json"), "w"), ensure_ascii=False)
        print(f"[절 CM90 wmjitter] 기록 {n} 프레임, 입자 부분 표본 {so.size}, 가우스 점 표본 {info['nG']}, 예산 기록 {info['tb']}, 오류 {s.err}", flush=True)


def install(out, sub=64, gsub=8, f0=29):
    from . import fem_speed6 as S6
    C.update(out=out, sub=int(sub), gsub=int(gsub), f0=int(f0), on=True)
    D0 = S6.Defer
    ok_keep, ok_fin = D0.keep, D0.finish

    def keep(D, fi, S, redo, K, tries):
        r = ok_keep(D, fi, S, redo, K, tries)
        try:
            R = _ST.get("R")
            if R is None:
                R = _ST["R"] = Rec(D, S)
            R.frame(D, fi, S, fi % 3)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            with open(os.path.join(out, "wj_err.txt"), "a") as f:
                f.write(f"f{fi + 1} {ex!r}\n")
        return r

    def finish(D, *a, **kw):
        r = ok_fin(D, *a, **kw)
        try:
            R = _ST.get("R")
            if R is not None:
                R.finish(D, float("nan"))
        except Exception as ex:
            import traceback
            traceback.print_exc()
            with open(os.path.join(out, "wj_err.txt"), "a") as f:
                f.write(f"finish {ex!r}\n")
        return r

    D0.keep, D0.finish = keep, finish
    print(f"[절 CM90] wmjitter 켬 (가벼운 판 Defer 기록 감쌈; 입자 표본 1/{sub}, 가우스 점 표본 1/{gsub}, 가우스 점 표본 고르는 프레임 f{f0 + 1})", flush=True)
