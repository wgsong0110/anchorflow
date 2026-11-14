"""절 CM160 tear 진단 (결과 불변, 읽기만): bread 찢어짐 구간의 노드 탄성력·가우스 점 응력 기록.
  - 모듈 소스 (aq_elem 이 있는 판) 에 전역 TR_Q (가우스 점마다 12 float) · TR_ID (가우스 점 중심 노드 번호) 와 커널 tr_cp·tr_df 를 더하고,
    aq_elem 의 'Pm *= vq' 바로 앞에 코시 응력 6 성분 (Pm Xᵀ / J)·시험 X 의 특이값 3·J·ys·vq 를 TR_Q 에 쓰는 줄만 끼움 (풀이 버퍼는 안 건드림).
  - aq_elem 부르는 앞뒤로 tr_cp (f 사본) · tr_df (f − 사본 = 탄성 노드 힘) 를 부름 -> 마지막 평가의 탄성 힘 · 응력이 남음 (AF_SKIP 이면 둘 다 안 돎).
  - AF_TEAR_FR (쉼표 프레임 목록) 의 프레임 끝에 (다음 프레임 첫 step 또는 종료 때) 노드 (질량 > 0) 위치·질량·탄성 힘·구속 몫 kb, 가우스 점 응력을 <out>/tear/f<프레임>.npz 로.
훅: 실행기 runpy 순간 (가장 바깥; exe/tear_wrap.py)."""
import json
import math
import os

import numpy as np
import torch

INFO = dict(patched=0, heads=0, n_flush=0, err=None, calls=0)
OUT = ["."]
FR = set()
MARK = "// ---- 절 CM160 tear"
KHEAD = "__global__ void ao_commit("
EHEAD = "__global__ void __launch_bounds__(128) aq_elem("
EOLD = "    for (int k=0;k<9;k++) Pm[k] *= vq;\n"
ENEW = (
    "    if (TR_Q && on) {                                                         // 절 CM160: 응력 기록 (읽기만)\n"
    "        float tj_ = ss[0]*ss[1]*ss[2]; float tij_ = (fabsf(tj_) > 1e-12f) ? 1.f/tj_ : 0.f;\n"
    "        float* to_ = TR_Q + (size_t)qq*12;\n"
    "        const int tra_[6] = {0,1,2,0,0,1}, trb_[6] = {0,1,2,1,2,2};\n"
    "        for (int tk_=0; tk_<6; tk_++) { int ta_ = tra_[tk_], tb_ = trb_[tk_]; to_[tk_] = (Pm[ta_*3]*X[tb_*3] + Pm[ta_*3+1]*X[tb_*3+1] + Pm[ta_*3+2]*X[tb_*3+2])*tij_; }\n"
    "        to_[6] = S[0]; to_[7] = S[1]; to_[8] = S[2]; to_[9] = tj_; to_[10] = ys; to_[11] = vq;\n"
    "        TR_ID[qq] = ids[13];\n"
    "    }\n" + EOLD)
KSRC = r"""
// ---- 절 CM160 tear: 탄성 노드 힘 = aq_elem 앞뒤 f 차 ----
__global__ void tr_cp(const int* IP, const double* f, double* prev) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= IP[9]) return;
    for (int a=0;a<3;a++) prev[n*3+a] = f[n*3+a];
}
__global__ void tr_df(const int* IP, const double* f, const double* prev, double* out) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= IP[9]) return;
    for (int a=0;a<3;a++) out[n*3+a] = f[n*3+a] - prev[n*3+a];
}
"""
GDECL = "__device__ float* TR_Q;\n__device__ int* TR_ID;\n"


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def patch(code):
    if MARK in code or KHEAD not in code or EHEAD not in code or "__device__ float* AQ_DBG;" not in code:
        return code
    INFO["heads"] += code.count(EHEAD)
    out, pos = [], 0
    while True:
        i = code.find(EHEAD, pos)
        if i < 0:
            break
        j = code.find("__global__", i + len(EHEAD))
        j = len(code) if j < 0 else j
        seg = code[i:j]
        assert seg.count(EOLD) == 1, ("tear: aq_elem 'Pm *= vq' 줄 수", seg.count(EOLD))
        out.append(code[pos:i]); out.append(seg.replace(EOLD, ENEW, 1)); pos = j
    out.append(code[pos:])
    code = "".join(out)
    code = code.replace("__device__ float* AQ_DBG;", "__device__ float* AQ_DBG;\n" + GDECL, 1)
    k = code.index(KHEAD)
    INFO["patched"] += 1
    return code[:k] + MARK + "\n" + KSRC + code[k:]


def _dump():
    try:
        with open(os.path.join(OUT[0], "tearinfo.json"), "w") as f_:
            json.dump(INFO, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def _bind(s):
    g1, g2 = s.mod.get_global("TR_Q"), s.mod.get_global("TR_ID")
    import cupy as cp
    cp.ndarray((1,), dtype=cp.uint64, memptr=g1)[0] = int(s.trQ.data_ptr())
    cp.ndarray((1,), dtype=cp.uint64, memptr=g2)[0] = int(s.trID.data_ptr())


def _flush(s, t):
    if t is None:
        return
    fr = int(round(float(t) / float(s.h))) + 1
    if fr not in FR:
        return
    try:
        torch.cuda.synchronize()
        if not bool(torch.isfinite(s.x[:1]).all()):
            return
        IP = s.IP[:16].cpu().numpy().astype(np.int64)
        nn = int(IP[9]); nb = IP[6:9]; bmin = IP[0:3]; nb1, nb2 = int(nb[1]), int(nb[2])
        dev = s.U.device
        A = s.A.double().reshape(3, 3)
        ids = torch.arange(nn, device=dev, dtype=torch.int64)
        ijk = torch.stack([ids // (nb1 * nb2), (ids // nb2) % nb1, ids % nb2], 1) + torch.tensor(bmin, device=dev)
        lat = ijk.double() @ A.T
        U = s.U[:nn].double()
        pos = lat + U
        mn = s.mn[:nn].double()
        act = mn > 0
        fel = s.trD[:nn * 3].view(nn, 3)
        kb = getattr(s, "kb", None)
        o = dict(fr=fr, dx=float(A[:, 0].norm()), mthr=float(getattr(s, "mthr", 0.0)), nn=nn,
                 aid=torch.nonzero(act).flatten().int().cpu().numpy(),
                 pos=pos[act].float().cpu().numpy(), lat=lat[act].float().cpu().numpy(), m=mn[act].float().cpu().numpy(),
                 fel=fel[act].float().cpu().numpy(), kb=(kb[:nn][act].float().cpu().numpy() if kb is not None else np.zeros(int(act.sum()), np.float32)),
                 ijk=ijk[act].int().cpu().numpy())
        nq = int(IP[12])
        Q = s.trQ[:nq * 12].view(nq, 12)
        cid = s.trID[:nq].long()
        ok = (cid >= 0) & (cid < nn) & (Q[:, 11] > 0)
        o["q_pos"] = pos[cid[ok]].float().cpu().numpy()
        o["q_lat"] = lat[cid[ok]].float().cpu().numpy()
        o["q"] = Q[ok].cpu().numpy()
        o["q_id"] = cid[ok].int().cpu().numpy()
        os.makedirs(os.path.join(OUT[0], "tear"), exist_ok=True)
        np.savez_compressed(os.path.join(OUT[0], "tear", f"f{fr:03d}.npz"), **o)
        INFO["n_flush"] += 1
    except Exception as e:
        INFO["err"] = "flush: " + repr(e)[:400]
    _dump()


def install(out):
    global FR
    OUT[0] = out
    FR = {int(q) for q in os.environ.get("AF_TEAR_FR", "").split(",") if q.strip()}
    from . import fem_an_rt as ART
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
    oa, ok, os_ = R.alloc, R.k, R.step

    def alloc(s, need):
        r = oa(s, need)
        try:
            if hasattr(s, "mod") and getattr(s, "capQc", None) is not None and INFO["patched"]:
                capQ, capN = int(s.capQc) * 8, int(s.capN)
                dev = s.dev
                if getattr(s, "trQ", None) is None or s.trQ.numel() != capQ * 12 or s.trD.numel() != capN * 3:
                    s.trQ = torch.zeros(capQ * 12, device=dev, dtype=torch.float32)
                    s.trID = torch.full((capQ,), -1, device=dev, dtype=torch.int32)
                    s.trP = torch.zeros(capN * 3, device=dev, dtype=torch.float64)
                    s.trD = torch.zeros(capN * 3, device=dev, dtype=torch.float64)
                _bind(s)
        except Exception as e:
            INFO["err"] = "alloc: " + repr(e)[:400]
        return r

    def k(s, name, n, args, b=128):
        if name == "aq_elem" and getattr(s, "trP", None) is not None:
            ok(s, "tr_cp", s.capN, (args[0], args[10], P(s.trP)), 256)
            r = ok(s, name, n, args, b)
            ok(s, "tr_df", s.capN, (args[0], args[10], P(s.trP), P(s.trD)), 256)
            INFO["calls"] += 1
            return r
        return ok(s, name, n, args, b)

    def step(s, t, *a, **kw):
        pend = getattr(s, "_tr_pend", None)
        if pend is not None and pend != t:
            _flush(s, pend)
            s._tr_pend = None
        r = os_(s, t, *a, **kw)
        s._tr_pend = t
        if not getattr(s, "_tr_atexit", False):
            import atexit
            s._tr_atexit = True
            atexit.register(lambda s_=s: _flush(s_, getattr(s_, "_tr_pend", None)))
        return r

    R.alloc, R.k, R.step = alloc, k, step
    print(f"[절 CM160] tear 켬 -- 프레임 {sorted(FR)} 노드 탄성 힘·가우스 점 응력 기록 (결과 불변)", flush=True)
    _dump()
