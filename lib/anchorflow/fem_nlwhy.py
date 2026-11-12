"""절 CM119 묶음 nlwhy (조정자 11-12, 진단 -- 결과 불변): 수렴 판 (cvb) 의 뉴턴 반복이 어디에 쓰이는가.

전체 판 (다시 풀기 아님) 에서 프레임마다 (실행기 프레임 함수 ART.split_step · WRX.split_col 를 감쌈) 걸음·뉴턴 반복별로 남김 -> nl.jsonl:
  - fem_convwhy 와 같은 그래프 안 사본: rt_ctl(3) 뒤 ‖g‖²·F²·멈춤, rt_ctl(7) 뒤 PCG 누적·도달, PCG 반복마다 rz·pAp·|r|² (-> PCG 상대 잔차 Prel, 랑초시 κ),
    반복별 장벽 실현 가능 보폭 k (BSK), 걸음 시작점 k (Y0K) -- 읽기는 fem_convwhy._read 그대로.
  - 선탐색 보폭: rt_lsup 앞 SC[14] (rt_lspick(N) 이 고른 c; full 이면 1) 과 full 표시.
  - 소성 상태: 재질 커널 (ao_elem_s / ao_elem) 이 불릴 때마다 같은 인자 (Fq, y) 로 가우스 점마다 시험 X = F_q + Σ y ⊗ g 의 ao_dpf 상태
    (0 탄성 · 1 꼬리 · 2 꼭짓점) 를 다시 계산해 상태별 수와 앞 평가 대비 바뀐 점 수 (전체 · 탄성<->소성 · 꼬리<->꼭짓점) 를 (걸음, 반복) 줄에 원자 합. 상태 버퍼는 따로 (풀이 안 건드림).
훅: 실행기 runpy 순간 (가장 바깥). 커널 글자는 ao_commit 앞에 넣음 (fem_wolfh12 와 같은 자리; qnodes·gqn·ssvdf·ao_dpf 를 씀).
"""
import json
import os

import numpy as np
import torch

ST = dict(out=".", nsrc=0, nalloc=0, nframe=0, npl=0, nls=0, err=None)
LOG = [None]
MARK = "// ---- 절 CM119 nlwhy"
HEAD = "__global__ void ao_commit("
NL_SRC = r"""
// ---- 절 CM119 nlwhy: 뉴턴 반복별 가우스 점 소성 상태 (읽기만, 상태 버퍼 따로) ----
__global__ void nl_pst(const int* IP, const int* cb, const float* BG, const float* Fq, const float* y, float cdp, int* stb, int* row) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    if (q == 0) atomicAdd(&row[7], 1);
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    for (int o_=0;o_<27;o_++) { float g[3]; gqn(F, BG + (oc*27+o_)*3, g); int id = ids[o_];
        for (int a=0;a<3;a++) { float ua = y[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; } }
    float Uu[9], S[3], V[9];
    ssvdf(X, Uu, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    atomicAdd(&row[st], 1);
    int old = stb[q]; stb[q] = st;
    if (old >= 0 && old != st) {
        atomicAdd(&row[3], 1);
        if ((old == 0) != (st == 0)) atomicAdd(&row[4], 1);
        else atomicAdd(&row[5], 1);
    }
}
"""


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def patch(code):
    if MARK in code or HEAD not in code or "ao_dpf(" not in code:
        return code
    i = code.index(HEAD)
    ST["nsrc"] += 1
    return code[:i] + MARK + "\n" + NL_SRC + code[i:]


def _dump():
    try:
        with open(os.path.join(ST["out"], "nlinfo.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def install(out):
    from . import fem_an_rt as ART
    from . import fem_convwhy as CW
    ST.update(out=out)
    LOG[0] = open(os.path.join(out, "nl.jsonl"), "w")
    cpm = ART.cp
    orm = cpm.RawModule

    def RawModule(*a, **kw):
        if "code" in kw:
            kw["code"] = patch(kw["code"])
        elif a:
            a = (patch(a[0]),) + tuple(a[1:])
        return orm(*a, **kw)
    cpm.RawModule = RawModule
    R = ART.RTSolver
    oa, octl, ok = R.alloc, R.ctl, R.k

    def alloc(s, need):
        r = oa(s, need)
        try:
            CW._alloc_bufs(s)
            KK, nt = CW._kmax(s), int(s.nt)
            s.nlLS = torch.zeros(KK * nt, 2, device=s.dev, dtype=torch.float64)
            s.nlPL = torch.zeros(KK * nt, 8, device=s.dev, dtype=torch.int32)
            capQ = s.Fq.numel() // 9
            if getattr(s, "nlST", None) is None or s.nlST.numel() != capQ:
                s.nlST = torch.full((capQ,), -1, device=s.dev, dtype=torch.int32)
            ST["nalloc"] += 1
        except Exception as e:
            ST["err"] = "alloc " + repr(e)
        return r

    def row(s):
        return max(s._cw_k, 0) * int(s.nt) + s._cw_it

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
            else:
                s.dcopy(s.cwD7[i], s.CD); s.dcopy(s.cwT7[i], s.CT)
        return r

    def k(s, name, n, args, b=128):
        if getattr(s, "cwS3", None) is not None:
            if name == "rt_sc_roll":
                i = row(s); ci = s._cw_ci
                if ci < s.cwG.shape[1]:
                    s.dcopy(s.cwG[i, ci, 0:5], s.SC[0:5]); s.dcopy(s.cwG[i, ci, 5:6], s.SC[57:58]); s.dcopy(s.cwGT[i, ci], s.CT[0:2])
            elif name == "rt_lsup":
                i = row(s)
                s.dcopy(s.nlLS[i, 0:1], s.SC[14:15])
                if int(args[-1]):
                    s._nl_full = getattr(s, "_nl_full", set()) | {i}
                ST["nls"] += 1
            elif name in ("ao_elem_s", "ao_elem") and getattr(s, "nlST", None) is not None:
                i = row(s)
                ok(s, "nl_pst", n, (args[0], args[1], args[2], args[3], args[5], args[8], P(s.nlST), P(s.nlPL[i])), 128)
                ST["npl"] += 1
        return ok(s, name, n, args, b)

    R.alloc, R.ctl, R.k = alloc, ctl, k

    def frame(fn, s, fi, *a):
        if getattr(s, "_nl_in", False):
            return fn(s, *a)
        s._nl_in = True
        try:
            return frame1(fn, s, fi, *a)
        finally:
            s._nl_in = False

    def frame1(fn, s, fi, *a):
        if getattr(s, "cwS3", None) is not None:
            CW._clear(s); s.nlLS.zero_(); s.nlPL.zero_()
        if not hasattr(s, "_nl_full"):
            s._nl_full = set()
        r = fn(s, *a)
        try:
            if getattr(s, "cwS3", None) is not None:
                torch.cuda.synchronize()
                K = int(s.Kcur)
                steps = CW._read(s, None, K)
                LS, PL = s.nlLS.cpu().numpy(), s.nlPL.cpu().numpy()
                nt = int(s.nt)
                for st_ in steps:
                    for o in st_["its"]:
                        i = st_["k"] * nt + o["it"]
                        o["ls"] = float(LS[i, 0]); o["full"] = int(i in s._nl_full)
                        o["pl"] = PL[i].tolist()                                       # [탄성, 꼬리, 꼭짓점, 바뀜, 탄성<->소성, 꼬리<->꼭짓점, -, 평가 수]
                c = {}
                try:
                    c = s.counts()
                except Exception:
                    pass
                LOG[0].write(json.dumps(dict(f=fi + 1, K=K, steps=steps, ms=float(getattr(s, "last_ms", 0.0)),
                                             nq=c.get("nq"), q_tail=c.get("q_tail"), q_apex=c.get("q_apex")), ensure_ascii=False, default=float) + "\n")
                LOG[0].flush()
        except Exception as e:
            ST["err"] = "read " + repr(e)
        ST["nframe"] += 1
        if ST["nframe"] in (1, 3) or ST["nframe"] % 10 == 0:
            _dump()
        return r

    osplit = ART.split_step

    def split_step(s, t, h, depth):
        if depth > 0:
            return osplit(s, t, h, depth)
        return frame(lambda s_, *a: osplit(s_, *a), s, int(round(t / h)), t, h, depth)
    ART.split_step = split_step
    try:
        from . import fem_wmrx as WRX
        ocol = WRX.split_col

        def split_col(s, t, h, sub_dt, depth, fn=None):
            if depth > 0:
                return ocol(s, t, h, sub_dt, depth, fn)
            return frame(lambda s_, *a: ocol(s_, *a), s, int(round(t / h)), t, h, sub_dt, depth, fn)
        WRX.split_col = split_col
    except Exception as e:
        ST["err"] = "wrx " + repr(e)
    import atexit
    atexit.register(_dump)
    print("[절 CM119] nlwhy 켬 -- 뉴턴 반복별 |g|·PCG·선탐색·장벽 보폭·소성 상태 바뀜 기록 (결과 불변)", flush=True)
    _dump()
