"""절 CM65 묶음 speed6 (11-08 위임, 렌더 포함 60 fps 를 벽시계로): 공통 후보 G' (= F + A3 + P1X + H5 + Rf) + 겹치기 O (절 CM62) 의
프레임 벽시계와 GPU 합 차이 가르기 + 결과 불변 호스트 일 걷어내기. 감싸개 exe/speed6_wrap.py 가 부름 (인자 없으면 이 모듈을 안 부름).

(W) 프레임 벽시계 기록 (--s6w 1, 결과 불변; 다른 s6 인자가 있으면 늘 켬) -> <out>/s6.jsonl
    가장 바깥 쪼개기 함수 (ART·AORT split_step, WRX split_col) 앞뒤: 호스트 시각 h0·h1 (perf_counter) + CUDA 이벤트 ev0·ev1 (시뮬 스트림)
    + 래스터화 끝 이벤트 (겹치기 O 면 기본 스트림의 래스터화 끝, 아니면 ev1 = 렌더 포함 끝).
    GPU 이벤트 -> 벽시계: 첫 프레임에서 장치 동기화 뒤 기준 이벤트 E_ref 를 기록·동기화하고 그 순간 perf_counter = t_ref -> wall(ev) = t_ref + E_ref→ev.
    기록 칸 (ms): host = h1 − h0 (호스트가 쪼개기 함수 안에 있던 시간), per = ev0 → ev1 (GPU 시각, 절 CM62 와 같은 정의), P = 그 프레임 그림이 다 그려진 벽시계,
    W = P(t) − P(t−1) (벽시계 프레임 간격 = 판정 기준), lat = P(t) − h0(t) (호스트가 프레임을 시작한 때부터 그림까지), gap = h0(t) − h1(t−1) (실행기의 프레임 사이 코드),
    lag = wall(ev0) − h0 (호스트가 띄운 첫 GPU 일이 실제로 시작하기까지), ng = 그래프 재생 수, nsync = 장치·스트림 동기화 호출 수.
    이벤트는 끝난 것만 그때그때 읽고 (기다리지 않음) 나머지는 끝에서 한꺼번에 -- 기록이 프레임 시간에 끼지 않게.

(L) 가벼운 판 (--s6l 1, 결과 불변): 감싸개가 설치되기 전에 (speed6_wrap 맨 앞) 기록 전용 프레임 훅을 같은 자리·같은 장치 일만 하는 판으로 바꿈
    - fem_speed3.frame_hook_install: 프레임 시작 표시 TB_NEWF = 1 (시간 예산 시계 시작, 결과에 쓰임) 만 -- s3.jsonl·t2c.jsonl 읽기·쓰기 없음
    - fem_speed4.frame_install: 없음 (s4.jsonl 기록 전용)
    - fem_hhtc2._wrap_step: 걸음 셈 버퍼 _hcst 0 으로 (장치 일, H5 판정 행 번호에 쓰임) 만 -- hc.jsonl 읽기 없음
    - 진단 E1 (AF_E1_DG) 은 끔 (stab 스크립트), 실행기는 exe/s6/ 사본 (프레임 사이 평가·기록을 끝으로 미룸, Defer 아래)
    렌더 (rtrender1) 와 Rf 는 그대로 (절 CM54·56 정의).

(P) 프로파일 (--s6prof F0,F1): 프레임 F0 시작 ~ F1 시작 사이를 torch.profiler (CPU·CUDA, 파이썬 호출 쌓기) 로 -> <out>/trace.json.gz
    + 분석 analyze() -> <out>/s6prof.json (프레임별 GPU 빈 시간을 그때 호스트가 돌던 파이썬 함수·CUDA API 로 나눔).
"""
import gzip
import json
import os
import time

import numpy as np
import torch


class C:
    out = "."
    on = False
    o = 0
    lean = 0
    prof = None


# ======================= (L) 가벼운 판 프레임 훅 (감싸개 설치 전에 바꿈) =======================
def lean_preinstall():
    from . import fem_hhtc2 as HC2
    from . import fem_speed3 as S3
    from . import fem_speed4 as S4

    def frame_hook_install(out):
        """fem_speed3.frame_hook_install 과 같은 자리 (가장 바깥 쪼개기 함수 들어갈 때) 에 TB_NEWF = 1 만 (기록 없음)."""
        from . import fem_an_rt as ART
        from . import fem_ao_rt as AORT
        from . import fem_wmrx as WRX
        dep = [0]

        def outer(fn):
            def w(s, *a, **kw):
                if dep[0] == 0 and S3.TB.mod is not None:
                    S3.TB.g("TB_NEWF", np.int32)[0] = 1
                dep[0] += 1
                try:
                    return fn(s, *a, **kw)
                finally:
                    dep[0] -= 1
            return w
        ART.split_step = outer(ART.split_step)
        AORT.split_step = outer(AORT.split_step)
        WRX.split_col = outer(WRX.split_col)
    S3.frame_hook_install = frame_hook_install

    def frame_install(out):
        return None
    S4.frame_install = frame_install

    def _wrap_step(cls):
        o_s = cls.step

        def step(s, t, *a, **kw):
            d = getattr(s, "_hcdep", 0)
            s._hcdep = d + 1
            try:
                r = o_s(s, t, *a, **kw)
            finally:
                s._hcdep = d
            if d == 0 and getattr(s, "_hcst", None) is not None:
                s._hcst.zero_()
            return r
        cls.step = step
    HC2._wrap_step = _wrap_step
    print("[절 CM65] 가벼운 판: speed3·speed4·hhtc2 프레임 기록 훅을 장치 일만 하는 판으로", flush=True)


# ======================= 실행기 사본 (exe/s6/) 의 프레임 사이 평가 미루기 =======================
def lean_runner():
    return C.lean == 1 or os.environ.get("AF_S6_LEAN", "0") == "1"


class Defer:
    """프레임마다 입자 위치 (원래 순서) [+ logJp, 속도 최대] 를 복사 스트림으로 고정 메모리에 (시뮬 스트림은 기다리지 않음), 끝에서 오차·손상·발산·이탈을 셈."""

    def __init__(s, S, L, ljp=False, vmax=False):
        Np = int(S.x.shape[0])
        s.L, s.Np = L, Np
        s.X = torch.empty(L, Np, 3, dtype=torch.float32, pin_memory=True)
        s.LJ = torch.empty(L, Np, dtype=torch.float32, pin_memory=True) if ljp else None
        s.VM = torch.full((L,), float("nan"), device=S.x.device, dtype=torch.float32) if vmax else None
        s.cs = torch.cuda.Stream()
        s.meta = []
        s.n = 0
        s.cell = float(S.A.reshape(3, 3)[:, 0].norm()) if torch.is_tensor(S.A) else None
        print(f"[절 CM65] 실행기 사본: 프레임 사이 평가를 끝으로 (고정 메모리 {s.X.numel() * 4 / 1e9 + (s.LJ.numel() * 4 / 1e9 if ljp else 0):.2f} GB)", flush=True)

    def keep(s, fi, S, redo, K, tries):
        cur = torch.cuda.current_stream()
        xo = torch.empty_like(S.x)
        xo[S.perm] = S.x
        lo = None
        if s.LJ is not None:
            lo = torch.empty_like(S.ljp)
            lo[S.perm] = S.ljp
        if s.VM is not None:
            s.VM[fi] = torch.linalg.vector_norm(S.v, dim=-1).max()
        s.cs.wait_stream(cur)
        with torch.cuda.stream(s.cs):
            s.X[fi].copy_(xo, non_blocking=True)
            if lo is not None:
                s.LJ[fi].copy_(lo, non_blocking=True)
        xo.record_stream(s.cs)
        if lo is not None:
            lo.record_stream(s.cs)
        s.meta.append(dict(t=fi, ms=float(S.last_ms), redo=bool(redo), K=int(K) if K is not None else None, tries=[list(q) for q in tries] if tries else None))
        s.n = fi + 1

    def finish(s, gtx, EXT, flog, dev, fhash=False, vlim=None, e1=None):
        """돌려줌: errs, ms, div_at (실행기 고리와 같은 정의: 비유한·오차 > 2·(vlim 있으면) 속도 최대 > vlim 인 첫 프레임에서 끊음)."""
        import hashlib
        import math
        torch.cuda.synchronize()
        s.cs.synchronize()
        vm = s.VM.cpu().tolist() if s.VM is not None else None
        errs, ms, div_at = [], [], None
        prev_far = None
        for fi in range(s.n):
            xo = s.X[fi].to(dev)
            gt = gtx[fi + 1].to(dev).float()
            ok = torch.isfinite(gt).all(-1)
            e = float((xo[ok] - gt[ok]).norm(dim=-1).mean()) / EXT
            fin = bool(torch.isfinite(xo).all()) and math.isfinite(e)
            errs.append(e if fin else float("nan"))
            m = s.meta[fi]
            ms.append(m["ms"])
            rec = dict(t=fi, err=e, ms=m["ms"], redo=m["redo"], K=m["K"], lean=1)
            if m["tries"] is not None:
                rec["ks"] = dict(K=m["K"], tries=[q[:3] for q in m["tries"]])
                if any(len(q) > 3 for q in m["tries"]):
                    rec["ks"]["on"] = [q[3] if len(q) > 3 else None for q in m["tries"]]
            if fhash:
                rec["xh"] = hashlib.sha1(s.X[fi].numpy().tobytes()).hexdigest()[:10]
            okp = torch.isfinite(xo).all(-1)
            if s.LJ is not None:
                lj = s.LJ[fi].to(dev)
                rec["en"] = dict(lj0=float((lj[okp] > 0).double().mean()) if bool(okp.any()) else None, nfin=int(okp.sum()))
            if vm is not None:
                rec["vmax"] = vm[fi]
            if e1 is not None and s.cell:                                                # 절 CM42 E1 의 이탈 셈과 같은 식 (원래 순서 입자로)
                x = xo.float()
                xs = x[okp][::7].double()
                if xs.shape[0] > 10:
                    q = torch.quantile(xs, torch.tensor([0.001, 0.999], dtype=torch.float64, device=xs.device), dim=0).float()
                    d = torch.clamp(torch.maximum(q[0] - x, x - q[1]), min=0).norm(dim=-1)
                    d = torch.where(okp, d, torch.full_like(d, float("inf")))
                    far = d > 4 * s.cell
                    nw = int((far & ~prev_far).sum()) if prev_far is not None else int(far.sum())
                    prev_far = far
                    e1.write(json.dumps(dict(f=fi + 1, nfar=int(far.sum()), dmax=float(d[okp].max()) if bool(okp.any()) else None, nonfin=int((~okp).sum()), new=dict(n=nw), lean=1)) + "\n")
            flog.write(json.dumps(rec) + "\n")
            bad = (not fin) or e > 2.0 or (vlim is not None and vm is not None and vm[fi] is not None and math.isfinite(vm[fi]) and vm[fi] > vlim) or (vlim is not None and vm is not None and not math.isfinite(vm[fi]))
            if bad:
                div_at = fi + 1
                print(f"[발산] 프레임 {div_at} (끝에서 판정)", flush=True)
                break
        flog.flush()
        if e1 is not None:
            e1.flush()
        return errs, ms, div_at


# ======================= (W) 프레임 벽시계 기록 =======================
class W:
    q = []
    recs = []
    ref = None
    tref = None
    prev_P = None
    prev_h1 = None
    fr = 0
    dep = 0
    ng = 0
    nsync = 0
    prof = None


def _wall(ev):
    return W.tref + W.ref.elapsed_time(ev)                                               # ms (perf_counter 기준 ms)


def _resolve(force=False):
    while W.q:
        rec, ev0, ev1, ras = W.q[0]
        last = ras if ras is not None else ev1
        if not force and not last.query():
            break
        W.q.pop(0)
        if force:
            last.synchronize()
            ev1.synchronize()
        rec["per"] = round(ev0.elapsed_time(ev1), 4)
        P = _wall(last)
        rec["P"] = round(P, 4)
        rec["lat"] = round(P - rec["h0"], 4)
        rec["lag"] = round(_wall(ev0) - rec["h0"], 4)
        if ras is not None:
            rec["ras_end_after_ev1"] = round(ev1.elapsed_time(ras), 4)
        if W.prev_P is not None:
            rec["W"] = round(P - W.prev_P, 4)
        W.prev_P = P
        W.recs.append(rec)


def _flush_file():
    _resolve(True)
    with open(os.path.join(C.out, "s6.jsonl"), "w") as f:
        for r in W.recs:
            f.write(json.dumps(r) + "\n")


def _counters():
    og = torch.cuda.CUDAGraph.replay

    def replay(self, *a, **kw):
        W.ng += 1
        return og(self, *a, **kw)
    torch.cuda.CUDAGraph.replay = replay
    osy = torch.cuda.synchronize

    def synchronize(*a, **kw):
        W.nsync += 1
        return osy(*a, **kw)
    torch.cuda.synchronize = synchronize


def frame_install(out):
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX
    S5 = None
    if C.o:
        from . import fem_speed5 as S5

    def outer(fn):
        def w(s, *a, **kw):
            top = W.dep == 0
            if top:
                if W.ref is None:
                    torch.cuda.synchronize()
                    W.ref = torch.cuda.Event(enable_timing=True)
                    W.ref.record()
                    W.ref.synchronize()
                    W.tref = 1e3 * time.perf_counter()
                if C.prof and W.fr == C.prof[0]:
                    _prof_start()
                h0 = 1e3 * time.perf_counter()
                ev0 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                ng0, ns0 = W.ng, W.nsync
                if S5 is not None:
                    S5.O5.pend = []
                if W.prof is not None:
                    W.rf = torch.profiler.record_function(f"s6f{W.fr}")
                    W.rf.__enter__()
            W.dep += 1
            try:
                r = fn(s, *a, **kw)
            finally:
                W.dep -= 1
            if top:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                h1 = 1e3 * time.perf_counter()
                if W.prof is not None:
                    W.rf.__exit__(None, None, None)
                rec = dict(t=W.fr, o=int(C.o), a=float(getattr(s, "last_ms", float("nan"))), h0=round(h0, 4), host=round(h1 - h0, 4),
                           ng=W.ng - ng0, nsync=W.nsync - ns0)
                if W.prev_h1 is not None:
                    rec["gap"] = round(h0 - W.prev_h1, 4)
                W.prev_h1 = h1
                ras = S5.O5.pend[-1][1] if (S5 is not None and S5.O5.pend) else None
                W.q.append((rec, ev0, ev1, ras))
                W.fr += 1
                _resolve()
                if C.prof and W.fr == C.prof[1]:
                    _prof_stop()
            return r
        return w
    ART.split_step = outer(ART.split_step)
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    import atexit
    atexit.register(_flush_file)


# ======================= (P) 프로파일 =======================
def _prof_start():
    torch.cuda.synchronize()
    W.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_stack=True, record_shapes=False)
    W.prof.__enter__()
    print(f"[절 CM65] 프로파일 시작 프레임 {W.fr}", flush=True)


def _prof_stop():
    torch.cuda.synchronize()
    p = W.prof
    W.prof = None
    p.__exit__(None, None, None)
    fn = os.path.join(C.out, "trace.json")
    p.export_chrome_trace(fn)
    try:
        r = analyze(fn)
        json.dump(r, open(os.path.join(C.out, "s6prof.json"), "w"), ensure_ascii=False, indent=1)
        print(f"[절 CM65] 프로파일 분석 -> s6prof.json (프레임 {len(r.get('frames', []))})", flush=True)
    except Exception:
        import traceback
        open(os.path.join(C.out, "s6prof_err.txt"), "w").write(traceback.format_exc())
    with open(fn, "rb") as f, gzip.open(fn + ".gz", "wb") as g:
        g.write(f.read())
    os.remove(fn)


def _merge(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


class _Gaps:
    """GPU 빈 구간 (정렬, 겹침 없음) 위 누적 길이 -> 임의 구간 [t0, t1] 과 겹치는 빈 시간 O(log n)."""
    def __init__(s, gaps):
        import bisect
        s.bis = bisect
        s.gs = [a for a, b in gaps]
        s.ge = [b for a, b in gaps]
        s.cum = [0.0]
        for a, b in gaps:
            s.cum.append(s.cum[-1] + (b - a))

    def F(s, t):
        k = s.bis.bisect_right(s.gs, t) - 1
        if k < 0:
            return 0.0
        return s.cum[k] + max(0.0, min(t, s.ge[k]) - s.gs[k])

    def ov(s, t0, t1):
        return max(0.0, s.F(t1) - s.F(t0))


def analyze(fn, top=40):
    """크롬 추적 -> 프레임 (s6f 주석 시작 ~ 다음 s6f 시작) 마다 GPU 가 아무 일도 안 한 구간 (모든 스트림 합집합의 빈틈) 을
    그 구간에 호스트에서 돌던 파이썬 함수 (자기 몫 = 자식 뺀 몫; 포함 몫) 와 CUDA API 로 나눔. 단위 ms. 쪼개기 함수 안 (s6f 구간) / 밖 (실행기 프레임 사이) 따로."""
    import bisect
    ev = json.load(open(fn))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    X = [e for e in ev if e.get("ph") == "X" and "dur" in e]
    del ev
    gpu = [(float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    fr = sorted([(float(e["ts"]), float(e["ts"]) + float(e["dur"]), e["name"]) for e in X if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("s6f")])
    if not fr:
        return dict(n=0, err="s6f 주석 없음")
    T0 = fr[0][0]
    T1 = max(max((b for a, b in gpu), default=fr[-1][1]), fr[-1][1])
    gm = _merge(gpu)
    gaps, cur = [], T0
    for a, b in gm:
        if b <= cur:
            continue
        if a > cur:
            gaps.append((cur, min(a, T1)))
        cur = max(cur, b)
    if cur < T1:
        gaps.append((cur, T1))
    G = _Gaps(gaps)
    starts = [f[0] for f in fr]
    wend = starts[1:] + [T1]

    def fidx(t):
        k = bisect.bisect_right(starts, t) - 1
        return k if 0 <= k < len(fr) else None

    def where(t):                                                                       # 0 = 쪼개기 함수 안, 1 = 프레임 사이
        k = fidx(t)
        return None if k is None else (0 if t < fr[k][1] else 1)
    nF = len(fr)
    FR = [dict(name=fr[k][2], win_ms=(wend[k] - fr[k][0]) / 1e3, split_ms=(fr[k][1] - fr[k][0]) / 1e3,
               idle_in_ms=G.ov(fr[k][0], fr[k][1]) / 1e3, idle_out_ms=G.ov(fr[k][1], wend[k]) / 1e3,
               graph_launch=0, sync=0, memcpy=0, launch=0, self={}, api={}) for k in range(nF)]
    for k in range(nF):
        FR[k]["gpu_busy_ms"] = FR[k]["win_ms"] - FR[k]["idle_in_ms"] - FR[k]["idle_out_ms"]
        FR[k]["n_gap_in"] = sum(1 for a, b in gaps if fr[k][0] <= a < fr[k][1] and (b - a) > 5.0)
    by_tid = {}
    for e in X:
        if e.get("cat") == "python_function":
            by_tid.setdefault(e.get("tid"), []).append(e)
    tot_self, tot_incl = {}, {}
    for tid, L in by_tid.items():
        L.sort(key=lambda e: (float(e["ts"]), -float(e["dur"])))
        st = []                                                                          # [t0, t1, 이름, 사슬, 자식 빈 합]
        done = []
        for e in L:
            t0, t1 = float(e["ts"]), float(e["ts"]) + float(e["dur"])
            while st and st[-1][1] <= t0:
                done.append(st.pop())
            inc = G.ov(t0, t1)
            nd = [t0, t1, e["name"], [q[2] for q in st], 0.0, inc]
            if st:
                st[-1][4] += inc
            st.append(nd)
        done += st
        for t0, t1, name, chain, chi, inc in done:
            if inc <= 0:
                continue
            k = fidx(t0)
            if k is None:
                continue
            tot_incl[name] = tot_incl.get(name, 0.0) + inc
            sf = inc - chi
            if sf > 1e-9:
                key = " <- ".join([name] + chain[::-1][:3])
                tag = ("안 " if where(t0) == 0 else "밖 ") + key
                FR[k]["self"][tag] = FR[k]["self"].get(tag, 0.0) + sf
                tot_self[tag] = tot_self.get(tag, 0.0) + sf
    tot_api = {}
    for e in X:
        if e.get("cat") not in ("cuda_runtime", "cuda_driver"):
            continue
        a0, a1, name = float(e["ts"]), float(e["ts"]) + float(e["dur"]), e["name"]
        k = fidx(a0)
        if k is None:
            continue
        if "GraphLaunch" in name:
            FR[k]["graph_launch"] += 1
        if "ynchronize" in name:
            FR[k]["sync"] += 1
        if "Memcpy" in name:
            FR[k]["memcpy"] += 1
        if "LaunchKernel" in name:
            FR[k]["launch"] += 1
        v = G.ov(a0, a1)
        if v > 0:
            tag = ("안 " if where(a0) == 0 else "밖 ") + name
            FR[k]["api"][tag] = FR[k]["api"].get(tag, 0.0) + v
            tot_api[tag] = tot_api.get(tag, 0.0) + v
    for q in FR:
        q["top_self"] = sorted(((k_, round(v / 1e3, 4)) for k_, v in q.pop("self").items()), key=lambda z: -z[1])[:15]
        q["top_api"] = sorted(((k_, round(v / 1e3, 4)) for k_, v in q.pop("api").items()), key=lambda z: -z[1])[:10]
        for kk in ("win_ms", "split_ms", "idle_in_ms", "idle_out_ms", "gpu_busy_ms"):
            q[kk] = round(q[kk], 4)
    n = max(1, nF)
    return dict(n=nF, frames=FR,
                self_per_frame=sorted(((k, round(v / 1e3 / n, 4)) for k, v in tot_self.items()), key=lambda q: -q[1])[:top],
                incl_per_frame=sorted(((k, round(v / 1e3 / n, 4)) for k, v in tot_incl.items()), key=lambda q: -q[1])[:top],
                api_per_frame=sorted(((k, round(v / 1e3 / n, 4)) for k, v in tot_api.items()), key=lambda q: -q[1])[:20])


def install(out, o=0, lean=0, prof=None):
    C.out, C.o, C.lean, C.on = out, int(o), int(lean), True
    if prof:
        C.prof = tuple(int(q) for q in str(prof).split(","))
    if C.o:
        from . import fem_speed5 as S5
        S5.ov_install()
    _counters()
    frame_install(out)
    print(f"[절 CM65] speed6 켬 겹치기 {C.o} 가벼운 판 {C.lean} 프로파일 {C.prof} 시뮬 스트림 {torch.cuda.current_stream()}", flush=True)
