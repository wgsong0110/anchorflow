"""절 CM96 묶음 convwhy (사용자 11-10 지시, 진단만 -- 결과 불변): wolf 접촉 걸음이 뉴턴 반복 상한 안에서 안 풀리는 원인 가르기.

두 방식 (새 실행기 exe/fem_an_rt_sim.py, 재질 판 MatRT 모래):
- 저장 (--cwsave 'fi,fi,..', 0 부터 프레임 번호): 실행기의 프레임 함수 (ART.split_step / WRX.split_col) 첫 호출 때 그 프레임 시작 입자 상태
  (s._bk() 이름 그대로: x·v·F·m·vol·perm·pa (+ trel)) 와 contact_prev·Kcur·t 를 st_fNN.pt (NN = 풀 프레임, 1 부터) 에. 풀이는 그대로 (읽기만).
- 다시 풀기 (--cwload 디렉토리): 실행기 프레임 i 마다 st_*.pt i 번째를 입자 버퍼에 넣고 (같은 이름·모양 확인) contact_prev 를 되돌린 뒤
  K 1 걸음 하나를 s.step(t_저장) 으로 풂 (쪼개기·무너진 프레임 재시도 없음). 실행기 설정 (데이터 첫 프레임에서 정하는 격자·빈 노드 문턱·κ) 은 저장 판과 같음.
기록 (다시 풀기 판만, 그래프 안 복사만 -- 커널·값 그대로; 저장 판은 버퍼·복사 없음):
- 뉴턴 반복 it 마다 rt_ctl(3) 뒤 SC·CD·CT 사본 (‖g‖² = SC1, F² = CD2, 멈춤 CT0), rt_ctl(7) 뒤 CT·CD (PCG 반복 누적 CT3, 목표 CD1, 도달 CT1),
  PCG 반복마다 rt_sc_roll 앞 SC0 (rz), SC2 (pAp), SC3 (새 rz), SC4, SC57 (|r|²) 과 CT0·CT1 (돌았는지) -> 랑초시 삼중대각 (α = rz/pAp, β = rz'/rz) 로 전처리 연산자 조건수 추정,
  걸음 시작점 k (Y0K), 반복별 장벽 실현 가능 보폭 k (BSK), 걸음 끝 잔차 비 (cjg).
바꾸는 판 (다시 풀기 전용): --cwks c = 장벽 강성 κ × c (FL["KS"], 캡처 전), --cwnopl 1 = Drucker-Prager 반환 사상 끔 (ao_dpf 가 늘 탄성 0 을 돌려줌; 소스 바꿈).
출력: cw.jsonl (상태마다 한 줄), cwinfo.json (훅 수·오류).
"""
import glob
import json
import math
import os

import numpy as np
import torch

ST = dict(save=[], load="", ks=1.0, nopl=0, out=".", nsave=0, nload=0, nframe=0, nctl3=0, nctl7=0, nroll=0, nalloc=0, nopl_src=0, err=None, states=[])
LOG = [None]
NOPL_OLD = "    if (!(dg > 0.f)) return 0;\n    if (tr > 0.f) { s = 0.f; return 2; }"
NOPL_NEW = "    if (1) return 0;   // 절 CM96 소성 끔 (진단)\n    if (tr > 0.f) { s = 0.f; return 2; }"


def _dump():
    try:
        with open(os.path.join(ST["out"], "cwinfo.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def _kmax(s):
    return 4 * max(s.K, s.Kc or 0, getattr(s, "kcap", 0))


def _cgmax(s):
    return max(s.cg_list) if getattr(s, "cg_list", None) else int(s.cg)


def _alloc_bufs(s):
    KK, nt, cg = _kmax(s), int(s.nt), _cgmax(s)
    dev = s.dev
    s.cwS3 = torch.zeros(KK * nt, 64, device=dev, dtype=torch.float64)
    s.cwD3 = torch.zeros(KK * nt, 4, device=dev, dtype=torch.float64)
    s.cwT3 = torch.zeros(KK * nt, 16, device=dev, dtype=torch.int32)
    s.cwD7 = torch.zeros(KK * nt, 4, device=dev, dtype=torch.float64)
    s.cwT7 = torch.zeros(KK * nt, 16, device=dev, dtype=torch.int32)
    s.cwG = torch.zeros(KK * nt, cg, 6, device=dev, dtype=torch.float64)
    s.cwGT = torch.zeros(KK * nt, cg, 2, device=dev, dtype=torch.int32)
    s._cw_k, s._cw_it, s._cw_ci = 0, 0, 0
    ST["nalloc"] += 1


def _clear(s):
    for nm in ("cwS3", "cwD3", "cwT3", "cwD7", "cwT7", "cwG", "cwGT"):
        getattr(s, nm).zero_()
    s.cwGT.fill_(-1)
    s.cwT3[:, 0] = -1


def _lanczos(al, be):
    """PCG α_j, β_j (Fletcher-Reeves β = rz_{j+1}/rz_j) -> 랑초시 삼중대각 고유값 (전처리 연산자 M⁻¹A 의 근사 극값)."""
    m = len(al)
    if m == 0:
        return None
    T = np.zeros((m, m))
    for j in range(m):
        T[j, j] = 1.0 / al[j] + (be[j - 1] / al[j - 1] if j > 0 else 0.0)
        if j + 1 < m:
            T[j, j + 1] = T[j + 1, j] = math.sqrt(max(be[j], 0.0)) / al[j]
    ev = np.linalg.eigvalsh(T)
    return float(ev[0]), float(ev[-1])


def _read(s, label, K):
    """걸음 (kstep < K) 마다 뉴턴 반복 기록을 호스트로."""
    nt = int(s.nt)
    S3, D3, T3 = s.cwS3.cpu().numpy(), s.cwD3.cpu().numpy(), s.cwT3.cpu().numpy()
    D7, T7 = s.cwD7.cpu().numpy(), s.cwT7.cpu().numpy()
    G, GT = s.cwG.cpu().numpy(), s.cwGT.cpu().numpy()
    BSK = s.BSK.cpu().numpy() if getattr(s, "BSK", None) is not None else None
    Y0K = s.Y0K.cpu().numpy() if getattr(s, "Y0K", None) is not None else None
    cjg = s.cjg.cpu().numpy().tolist() if getattr(s, "cjg", None) is not None else None
    steps = []
    for k in range(K):
        its = []
        F2 = None
        for it in range(nt):
            i = k * nt + it
            if T3[i, 0] < 0:                                                           # 이 반복 rt_ctl(3) 기록 없음
                break
            if it == 0:
                F2 = float(D3[i, 2])
            g2 = float(S3[i, 1])
            stop = int(T3[i, 0])
            o = dict(it=it, r=(math.sqrt(g2 / F2) if (F2 and F2 > 0 and g2 >= 0) else None), stop=stop)
            if stop:                                                                   # 이 반복 시작에서 허용치 도달 (또는 앞에서 멈춤) -> 이 반복은 안 돎
                its.append(o)
                break
            nP = int(T7[i, 3]) - int(T3[i, 3])
            o.update(nP=nP, Preach=int(T7[i, 1]), Ptgt=float(D7[i, 1]))
            al, be, rr = [], [], []
            for ci in range(G.shape[1]):
                if GT[i, ci, 0] != 0 or GT[i, ci, 1] != 0:
                    break
                rz, pAp, rz1, r2 = G[i, ci, 0], G[i, ci, 2], G[i, ci, 3], G[i, ci, 5]
                if not (pAp > 0 and rz > 0):
                    break
                al.append(rz / pAp); be.append(rz1 / rz); rr.append(r2)
            o["nL"] = len(al)
            if al:
                lam = _lanczos(al, be)
                if lam:
                    o.update(lmin=lam[0], lmax=lam[1], kap=(lam[1] / lam[0] if lam[0] > 0 else None))
                o["Prel"] = math.sqrt(rr[-1] / g2) if g2 > 0 else None
            if BSK is not None:
                o["bk"] = int(BSK[i, 0])
            its.append(o)
        rs = [q["r"] for q in its if q.get("r") is not None]
        steps.append(dict(k=k, its=its, F2=F2, y0k=(int(Y0K[k, 0]) if Y0K is not None else None), g2end=(cjg[k] if cjg and k < len(cjg) else None), rs=rs))
    return steps


def _write(o):
    if LOG[0] is not None:
        LOG[0].write(json.dumps(o, ensure_ascii=False) + "\n")
        LOG[0].flush()


def _patch_src(code):
    if ST["nopl"] and NOPL_OLD in code:
        ST["nopl_src"] += code.count(NOPL_OLD)
        code = code.replace(NOPL_OLD, NOPL_NEW)
    return code


def install(out, save="", load="", ks=1.0, nopl=0):
    from . import fem_an_rt as ART
    if load and str(load).isdigit():                                                   # 묶음 표기 (경로는 묶음 셸이 지우므로 숫자): 0 = 연기 저장 판 k8, n = 저장 판 sv<n> (출력 디렉토리 옆)
        load = os.path.join(os.path.dirname(os.path.abspath(out)), {"0": "k8"}.get(str(load), "sv" + str(load)))
    ST.update(out=out, save=[int(q) for q in str(save).split(",") if q.strip() != ""], load=load, ks=float(ks), nopl=int(nopl))
    LOG[0] = open(os.path.join(out, "cw.jsonl"), "w")
    if ST["load"]:
        ST["states"] = sorted(glob.glob(os.path.join(ST["load"], "st_f*.pt")))
        assert ST["states"], f"상태 없음 {ST['load']}"
    if ST["nopl"]:
        cpm = ART.cp
        orm = cpm.RawModule

        def RawModule(*a, **kw):
            if "code" in kw:
                kw["code"] = _patch_src(kw["code"])
            elif a:
                a = (_patch_src(a[0]),) + tuple(a[1:])
            return orm(*a, **kw)
        cpm.RawModule = RawModule
    R = ART.RTSolver
    oa, octl, ok = R.alloc, R.ctl, R.k

    def alloc(s, need):
        r = oa(s, need)
        if not ST["load"]:                                                             # 저장 판은 버퍼·복사 없음 (H8 그대로)
            return r
        try:
            _alloc_bufs(s)
        except Exception as e:
            ST["err"] = "alloc " + repr(e)
        return r

    def ctl(s, op, it=0):
        r = octl(s, op, it)
        if getattr(s, "cwS3", None) is None:
            return r
        if op == 0:
            s._cw_k = -1
        elif op == 1:
            s._cw_k += 1
        elif op == 2:
            s._cw_it = it
        elif op == 5:
            s._cw_ci = it
        elif op in (3, 7):
            i = max(s._cw_k, 0) * int(s.nt) + it
            if op == 3:
                s.dcopy(s.cwS3[i], s.SC); s.dcopy(s.cwD3[i], s.CD); s.dcopy(s.cwT3[i], s.CT)
                ST["nctl3"] += 1
            else:
                s.dcopy(s.cwD7[i], s.CD); s.dcopy(s.cwT7[i], s.CT)
                ST["nctl7"] += 1
        return r

    def k(s, name, n, args, b=128):
        if name == "rt_sc_roll" and getattr(s, "cwS3", None) is not None:
            i = max(s._cw_k, 0) * int(s.nt) + s._cw_it
            ci = s._cw_ci
            if ci < s.cwG.shape[1]:
                s.dcopy(s.cwG[i, ci, 0:5], s.SC[0:5]); s.dcopy(s.cwG[i, ci, 5:6], s.SC[57:58]); s.dcopy(s.cwGT[i, ci], s.CT[0:2])
                ST["nroll"] += 1
        return ok(s, name, n, args, b)

    R.alloc, R.ctl, R.k = alloc, ctl, k

    osplit = ART.split_step

    def split_step(s, t, h, depth):
        fi = int(round(t / h))
        if ST["load"]:
            return _load_frame(s, h)
        if fi in ST["save"] and fi not in ST.get("_saved", set()):
            ST.setdefault("_saved", set()).add(fi)
            _save(s, t, h)
            r = osplit(s, t, h, depth)
            _write(dict(mode="save", f=fi + 1, K=int(s.Kcur), tries=[list(q) for q in r[2]], cjg=(s.cjg[:int(s.Kcur)].cpu().tolist() if getattr(s, "cjg", None) is not None else None)))
            return r
        return osplit(s, t, h, depth)

    ART.split_step = split_step

    try:
        from . import fem_wmrx as WRX
        ocol = WRX.split_col

        def split_col(s, t, h, sub_dt, depth, fn=None):
            fi = int(round(t / h))
            if ST["load"]:
                WRX.on(s, 0)
                r = _load_frame(s, h)
                return r[0], r[1], [tuple(q) + (0,) for q in r[2]]
            return ocol(s, t, h, sub_dt, depth, fn)
        WRX.split_col = split_col
    except Exception as e:
        ST["err"] = "wrx " + repr(e)

    def _save(s, t, h):
        nm = list(s._bkn()) if hasattr(s, "_bkn") else ["x", "v", "F", "m", "vol", "perm", "pa"]
        d = dict(names=nm, t=float(t), h=float(h), contact_prev=bool(getattr(s, "contact_prev", False)), Kcur=int(s.Kcur), K=int(s.K),
                 **{"p_" + q: getattr(s, q).detach().cpu().clone() for q in nm})
        fn = os.path.join(out, "st_f%02d.pt" % (int(round(t / h)) + 1))
        torch.save(d, fn)
        ST["nsave"] += 1
        _dump()

    def _load_frame(s, h):
        i = ST["nframe"]
        ST["nframe"] += 1
        if i >= len(ST["states"]):
            raise SystemExit(0)
        fn = ST["states"][i]
        d = torch.load(fn, map_location="cpu", weights_only=False)
        nm = list(s._bkn()) if hasattr(s, "_bkn") else ["x", "v", "F", "m", "vol", "perm", "pa"]
        assert nm == d["names"], (nm, d["names"])
        for q in nm:
            dst, src = getattr(s, q), d["p_" + q]
            assert tuple(dst.shape) == tuple(src.shape) and dst.dtype == src.dtype, (q, dst.shape, src.shape)
            dst.copy_(src.to(dst.device))
        s.contact_prev = d["contact_prev"]
        if ST["ks"] != 1.0 and not ST.get("_ks_done"):
            ST["_ks_done"] = True
            ST["KS0"] = float(s.floor["KS"])
            s.floor["KS"] = float(s.floor["KS"]) * ST["ks"]
            if "KSN" in s.floor:
                s.floor["KSN"] = float(s.floor["KSN"]) * ST["ks"]
            ST["KS1"] = float(s.floor["KS"])
        ART.set_K_an(s, 1, h)
        if getattr(s, "cwS3", None) is not None:
            _clear(s)
        redo = s.step(d["t"])
        ms = float(s.last_ms)
        torch.cuda.synchronize()
        ST["nload"] += 1
        o = dict(mode="load", f=int(round(d["t"] / d["h"])) + 1, src=os.path.basename(fn), K=int(s.Kcur), redo=bool(redo), ms=ms,
                 steps=_read(s, None, int(s.Kcur)), fin=bool(torch.isfinite(s.x).all()))
        try:
            c = s.counts()
            o["counts"] = {q: c.get(q) for q in ("nn", "nq", "it", "q_tail", "q_apex", "p_rm", "nlow", "nact")}
        except Exception as e:
            o["counts_err"] = repr(e)
        _write(o)
        _dump()
        nf = int(s.cjl[:1].sum()) if getattr(s, "cjl", None) is not None else 0
        return redo, 1, [(1, nf, ms)]

    print(f"[절 CM96] convwhy 켬 -- 저장 {ST['save']} 다시 풀기 '{ST['load']}' ({len(ST['states'])} 상태) 장벽 강성 × {ST['ks']} 소성 끔 {ST['nopl']}", flush=True)
    _dump()
