"""절 CM131 묶음 resmode (조정자 11-13, 진단 -- 결과 불변): 같은 예산에서 H5 는 걸음 잔차가 빨리 줄고 H9 는 정체하는 이유 -- 뉴턴 반복별 노드 잔차 기록.

프레임 창 [F0, F1] (env RM_F0·RM_F1, 1 부터) 의 프레임마다 <out>/rm/fNNN.npz (그래프 안 커널 사본, 풀이 버퍼는 읽기만):
  걸음 k (rt_ga_step0 / ao_step0_ga 로 셈, fem_escwhy 와 같은 번호) · 평가 e (rt_grad 가 불린 차례: 뉴턴 반복 시작 0..nt-1, 걸음 끝 다시 잼은 nt):
    R0 = rt_grad 의 r0 (= −T g, 안 접은 노드 잔차 -- 절 CM117 |g|/|g0| (fem_startpt sp_g2) 와 같은 버퍼),
    RF = rt_grad 뒤 r (확장 판은 접은 잔차 = 뉴턴이 실제로 보는 것; 확장 없으면 R0 와 같음),
    Y  = 그 평가의 y, DL = 그 반복 PCG 끝 방향 dl (rt_ctl(7) 뒤),
    OK = 그 평가가 실제로 돌았는지 (AF_SKIP 이면 커널이 안 써서 0) -- sp_g2 와 같은 판정.
  걸음 k 첫 평가: B (rt_grad 의 노드 대각 블록 T B T 앞 B, 탄성 + 고스트 + 장벽), ME (유효 질량 meff), IT2 (1/τ², 파이썬 값).
  프레임: 고스트 면 표지 g3fm (칸마다 +x·+y·+z 비트)·경계 칸 g3cut·두꺼운 칸 g3thk·칸 점유 g5phi·면 가중 g5w (fem_ghost4/5, 있으면), nid (mn > 0 노드, fem_escwhy 와 같은 순서).
훅: 실행기 runpy 순간 (가장 바깥). 커널 글자는 ao_commit 앞 (fem_escwhy 와 같은 자리). 버퍼 합이 RM_MAXB (기본 6e9 바이트) 를 넘으면 그 할당에서는 기록 안 함.
"""
import json
import os

import numpy as np
import torch

ST = dict(out=".", nsrc=0, nalloc=0, nsave=0, nskip=0, ngrad=0, ndl=0, err=None, F0=1, F1=10 ** 9, on_alloc=None, it2=[])
MARK = "// ---- 절 CM131 resmode"
HEAD = "__global__ void ao_commit("
RM_SRC = r"""
// ---- 절 CM131 resmode: 노드 버퍼 사본 (읽기만; AF_SKIP 이면 안 씀 -- 그 평가가 안 돈 것) ----
__global__ void rm_cpw(const int* IP, const float* src, float* dst, int w, int* flg) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i == 0 && flg) *flg = 1;
    if (i >= IP[9]*w) return;
    dst[i] = src[i];
}
"""


def P(t):
    if t is None:
        return np.uint64(0)
    return np.uint64(t.data_ptr()) if hasattr(t, "data_ptr") else np.uint64(t)               # 실행기 인자는 이미 포인터 (np.uint64)


def patch(code):
    if MARK in code or HEAD not in code:
        return code
    i = code.index(HEAD)
    ST["nsrc"] += 1
    return code[:i] + MARK + "\n" + RM_SRC + code[i:]


def _dump():
    try:
        with open(os.path.join(ST["out"], "rminfo.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def install(out):
    from . import fem_an_rt as ART
    ST.update(out=out, F0=int(os.environ.get("RM_F0", "1")), F1=int(os.environ.get("RM_F1", str(10 ** 9))))
    maxb = float(os.environ.get("RM_MAXB", "6e9"))
    os.makedirs(os.path.join(out, "rm"), exist_ok=True)
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
    oa, ok, octl, ost = R.alloc, R.k, R.ctl, R.step

    def alloc(s, need):
        r = oa(s, need)
        try:
            KK = max(int(s.K), int(getattr(s, "kcap", 0) or 0), int(getattr(s, "Kc", 0) or 0), 1)
            NE = int(s.nt) + 1
            capN = int(s.capN)
            nb = KK * NE * capN * 4 * 12 + KK * capN * 4 * 10
            s._rm = None
            torch.cuda.empty_cache()
            if nb > maxb:
                ST["on_alloc"] = f"끔 (버퍼 {nb / 1e9:.2f} GB > {maxb / 1e9:.1f})"
            else:
                Z = lambda *sh: torch.zeros(*sh, device=s.dev, dtype=torch.float32)
                s._rm = dict(R0=Z(KK, NE, capN, 3), RF=Z(KK, NE, capN, 3), Y=Z(KK, NE, capN, 3), DL=Z(KK, NE, capN, 3),
                             B=Z(KK, capN, 9), ME=Z(KK, capN), OK=torch.zeros(KK, NE, 4, device=s.dev, dtype=torch.int32))
                s._rm_KK, s._rm_NE = KK, NE
                ST["on_alloc"] = f"켬 (버퍼 {nb / 1e9:.2f} GB, K {KK}, 평가 {NE}, capN {capN})"
            ST["nalloc"] += 1
        except Exception as e:
            s._rm = None
            ST["err"] = "alloc " + repr(e)
        _dump()
        return r

    def cpw(s, src, dst, w, flg):
        ok(s, "rm_cpw", int(s.capN) * w, (P(s.IP), P(src), P(dst), np.int32(w), P(flg)), 256)

    def k(s, name, n, args, b=128):
        E = getattr(s, "_rm", None)
        if E is None:
            return ok(s, name, n, args, b)
        if name == "rt_reset":
            s._rm_k, s._rm_e = -1, 0
        r = ok(s, name, n, args, b)
        if name in ("rt_ga_step0", "ao_step0_ga"):
            s._rm_k = getattr(s, "_rm_k", -1) + 1
            s._rm_e = 0
        elif name == "rt_grad":                                                       # args: IP mn it2 sk pm n0 n1 n2 eta f B Dinv r z p dl Tp SC r0
            kk, e = getattr(s, "_rm_k", -1), getattr(s, "_rm_e", 0)
            if 0 <= kk < s._rm_KK and 0 <= e < s._rm_NE:
                fl = E["OK"][kk, e]
                cpw(s, args[18], E["R0"][kk, e], 3, fl[0:1])
                cpw(s, args[12], E["RF"][kk, e], 3, fl[1:2])
                cpw(s, s.y, E["Y"][kk, e], 3, None)
                if e == 0:
                    cpw(s, args[10], E["B"][kk], 9, None)
                    cpw(s, args[1], E["ME"][kk], 1, None)
                    it2 = float(args[2])
                    if len(ST["it2"]) < 64 and it2 not in ST["it2"]:
                        ST["it2"].append(it2)
                    s._rm_it2 = getattr(s, "_rm_it2", {})
                    s._rm_it2[kk] = it2
                ST["ngrad"] += 1
            s._rm_e = e + 1
        return r

    def ctl(s, op, it=0):
        r = octl(s, op, it)
        E = getattr(s, "_rm", None)
        if E is None:
            return r
        if op == 2:
            s._rm_e = int(it)
        elif op == 7:
            kk = getattr(s, "_rm_k", -1)
            if 0 <= kk < s._rm_KK and 0 <= it < s._rm_NE:
                cpw(s, s.dl, E["DL"][kk, it], 3, E["OK"][kk, it][2:3])
                ST["ndl"] += 1
        elif op == 8:
            s._rm_e = s._rm_NE - 1
        return r

    def step(s, t):
        fr = int(round(t / s.h)) + 1
        win = ST["F0"] <= fr <= ST["F1"]
        E = getattr(s, "_rm", None)
        if win and E is not None:
            E["OK"].zero_()
        r = ost(s, t)
        if win:
            try:
                save(s, fr)
            except Exception:
                import traceback
                ST["err"] = f"save f{fr} " + traceback.format_exc()[-800:]
                _dump()
        return r
    R.alloc, R.k, R.ctl, R.step = alloc, k, ctl, step

    def save(s, fr):
        torch.cuda.synchronize()
        E = getattr(s, "_rm", None)
        IP = s.IP.cpu().numpy().copy()
        o = dict(IP=IP, K=np.int32(getattr(s, "last_K", s.K)), rec=np.int32(E is not None))
        nn, nq = int(IP[9]), int(IP[12])
        if E is not None and nn > 0 and int(IP[13]) == 0:
            K = min(int(o["K"]), s._rm_KK)
            nid = torch.nonzero(s.mn[:nn] > 0).squeeze(1)
            o["nid"] = nid.cpu().numpy().astype(np.int32)
            for nm in ("R0", "RF", "Y", "DL"):
                o[nm] = E[nm][:K, :, :nn][:, :, nid].cpu().numpy()
            o["B"] = E["B"][:K, :nn][:, nid].cpu().numpy()
            o["ME"] = E["ME"][:K, :nn][:, nid].cpu().numpy()
            o["OK"] = E["OK"][:K].cpu().numpy()
            o["IT2"] = np.array([getattr(s, "_rm_it2", {}).get(k_, np.nan) for k_ in range(K)], dtype=np.float64)
            nc = nq // 8
            for nm, w_ in (("g3fm", 1), ("g3cut", 1), ("g3thk", 1), ("g5phi", 1), ("g5w", 3)):
                t_ = getattr(s, nm, None)
                if t_ is not None:
                    o[nm] = t_[:nc * w_].cpu().numpy()
            o["gpk"] = np.float64(getattr(s, "_gpk", np.nan) or 0.0)
            o["g3kM"] = np.float64(getattr(s, "_g3kM", np.nan) or 0.0)
            ST["nsave"] += 1
        else:
            ST["nskip"] += 1
        np.savez_compressed(os.path.join(ST["out"], "rm", f"f{fr:03d}.npz"), **o)
        _dump()
    import atexit
    atexit.register(_dump)
    print(f"[절 CM131] resmode 켬 -- 프레임 {ST['F0']}~{ST['F1']} 뉴턴 평가별 노드 잔차 (안 접은 r0 · 접은 r)·y·dl·대각 B·meff 기록 -> {out}/rm (결과 불변)", flush=True)
    _dump()
