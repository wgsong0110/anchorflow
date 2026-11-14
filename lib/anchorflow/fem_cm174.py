"""절 CM174 묶음 cm174 (조정자 11-14, 진단 -- 결과 불변; fem_nlwhy 위, fem_escwhy 와 같이 쓴다): 뉴턴 반복별 노드 힘 분해 + 가우스 점 상태.
창 프레임 (env CM174_F0~F1, 1 부터) 마다 <out>/cm174/fNNN.npz 에 남김 (그래프 안 사본·커널, 풀이 버퍼는 읽기만).
행 = 걸음 k x nt + 반복 it (fem_nlwhy 의 _cw_k·_cw_it). 재질 요소 ao_elem_s 마다 (그 반복의 y 에서):
  노드 (질량 > 0 만 저장): y, 탄성 몫 힘 (탄성 상태 점의 에너지 기울기 기여, 부호 = 에너지 기울기이므로 물리 힘은 −), 꼬리 몫 힘 (DP 꼬리 상태 점의 기여),
                           (꼭짓점 점은 응력 0 이라 기여 0), rt_grad 뒤: 전체 f (장벽 포함, f64 -> f32), g0buf (그 반복 시작 기울기).
  가우스 점: 상태 st (0 탄성 1 꼬리 2 꼭짓점), det X (그 반복 y 의 시험 X).
  걸음 단위 사본: cwS3 (SC, 반복별), cwT3 (CT), rls 기록 _rl_hist (있으면).
"""
import json
import os

import numpy as np
import torch

ST = dict(out=".", nsrc=0, nalloc=0, nsave=0, nskip=0, err=None, F0=1, F1=0, on_alloc=None)
MARK = "// ---- 절 CM174 cm174"
HEAD = "__global__ void ao_commit("
SRC = r"""
// ---- 절 CM174 cm174: ao_elem_s 의 힘 부분을 상태별로 나눠 노드에 (읽기만; 탄성 -> fe, 꼬리 -> ft, 꼭짓점은 응력 0) + 점 상태·det X ----
__global__ void cm174_f(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, float mu, float lam, float cdp,
                        float* fe, float* ft, signed char* stq, float* detq) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float A[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; int id = ids[o];
        for (int a=0;a<3;a++) { float ya = y[id*3+a]; A[a*3] += ya*bg[0]; A[a*3+1] += ya*bg[1]; A[a*3+2] += ya*bg[2]; } }
    float X[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) X[a*3+b] = F[a*3+b] + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b];
    float U[9], S[3], V[9];
    ssvdf(X, U, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    stq[q] = (signed char)st;
    detq[q] = X[0]*(X[4]*X[8]-X[5]*X[7]) - X[1]*(X[3]*X[8]-X[5]*X[6]) + X[2]*(X[3]*X[7]-X[4]*X[6]);
    if (st == 2) return;
    float K3 = lam + 2.f*mu/3.f;
    float p[3];
    for (int i=0;i<3;i++) {
        float tau = (st == 1) ? (K3*tr + 2.f*mu*s*e[i]) : (2.f*mu*ep[i] + lam*tr);
        p[i] = tau/(sgn[i]*sg[i]);
    }
    float vq = Vq[q];
    float Pm[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Pm[a*3+b] = vq*(U[a*3]*p[0]*V[b*3] + U[a*3+1]*p[1]*V[b*3+1] + U[a*3+2]*p[2]*V[b*3+2]);
    float Qf[9];
    for (int a=0;a<3;a++) for (int c=0;c<3;c++) Qf[a*3+c] = Pm[a*3]*F[c*3] + Pm[a*3+1]*F[c*3+1] + Pm[a*3+2]*F[c*3+2];
    float* out = (st == 1) ? ft : fe;
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3;
        for (int a=0;a<3;a++) atomicAdd(&out[ids[o]*3+a], Qf[a*3]*bg[0] + Qf[a*3+1]*bg[1] + Qf[a*3+2]*bg[2]); }
}
"""
NODE = ("NY", "NG", "NE", "NT")


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def patch(code):
    if MARK in code or HEAD not in code or "ao_dpf(" not in code:
        return code
    i = code.index(HEAD)
    ST["nsrc"] += 1
    return code[:i] + MARK + "\n" + SRC + code[i:]


def _dump():
    try:
        with open(os.path.join(ST["out"], "cm174info.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def install(out):
    from . import fem_an_rt as ART
    ST.update(out=out, F0=int(os.environ.get("CM174_F0", "1")), F1=int(os.environ.get("CM174_F1", "0")))
    maxb = float(os.environ.get("CM174_MAXB", "6e9"))
    os.makedirs(os.path.join(out, "cm174"), exist_ok=True)
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
    oa, ok, ost = R.alloc, R.k, R.step

    def alloc(s, need):
        r = oa(s, need)
        try:
            KK = max(int(s.K), int(getattr(s, "kcap", 0) or 0), int(getattr(s, "Kc", 0) or 0), 1)
            nt, capN, capQ = int(s.nt), int(s.capN), int(s.Fq.numel() // 9)
            RR = KK * nt
            nb = RR * (capN * 3 * 4 * 4 + capN * 3 * 8 + capQ * 5)
            s._c174 = None
            torch.cuda.empty_cache()
            if nb > maxb:
                ST["on_alloc"] = f"끔 (버퍼 {nb / 1e9:.2f} GB > {maxb / 1e9:.1f})"
            else:
                dev = s.dev
                E = {nm: torch.zeros(RR, capN, 3, device=dev, dtype=torch.float32) for nm in NODE}
                E["NF"] = torch.zeros(RR, capN, 3, device=dev, dtype=torch.float64)
                E["STq"] = torch.zeros(RR, capQ, device=dev, dtype=torch.int8)
                E["DQ"] = torch.zeros(RR, capQ, device=dev, dtype=torch.float32)
                s._c174 = E
                s._c174_KK = KK
                ST["on_alloc"] = f"켬 (버퍼 {nb / 1e9:.2f} GB, 행 {RR}, capN {capN}, capQ {capQ})"
            ST["nalloc"] += 1
        except Exception as e:
            s._c174 = None
            ST["err"] = "alloc " + repr(e)
        _dump()
        return r

    def cp_(s, dst, src):
        if src is None:
            return
        n = min(dst.numel(), src.numel())
        s.dcopy(dst.view(-1)[:n], src.reshape(-1)[:n])

    def row(s):
        return max(getattr(s, "_cw_k", 0), 0) * int(s.nt) + getattr(s, "_cw_it", 0)

    def k(s, name, n, args, b=128):
        r = ok(s, name, n, args, b)
        E = getattr(s, "_c174", None)
        if E is None or getattr(s, "cwS3", None) is None:
            return r
        i = row(s)
        if i >= E["NY"].shape[0]:
            return r
        if name in ("ao_elem_s", "ao_elem"):
            E["NE"][i].zero_(); E["NT"][i].zero_()
            ok(s, "cm174_f", n, (args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8],
                                 P(E["NE"][i]), P(E["NT"][i]), P(E["STq"][i]), P(E["DQ"][i])), 128)
            cp_(s, E["NY"][i], s.y)
        elif name.startswith("rt_grad"):
            cp_(s, E["NF"][i], s.f)
            cp_(s, E["NG"][i], getattr(s, "g0buf", None))
        return r

    def save(s, fr):
        torch.cuda.synchronize()
        E = s._c174
        IP = s.IP.cpu().numpy().copy()
        nt = int(s.nt)
        Kc = int(getattr(s, "last_K", s.K))
        RR = min(Kc * nt, E["NY"].shape[0])
        nn, nq = int(IP[9]), int(IP[12])
        o = dict(IP=IP, A=s.A.float().cpu().numpy(), K=np.int32(Kc), nt=np.int32(nt), h=np.float32(s.h))
        if nn > 0 and int(IP[13]) == 0:
            nid = torch.nonzero(s.mn[:nn] > 0).squeeze(1)
            o["nid"] = nid.cpu().numpy().astype(np.int32)
            o["mn"] = s.mn[:nn][nid].cpu().numpy()
            for nm in ("NY", "NG", "NE", "NT"):
                o[nm] = E[nm][:RR, :nn][:, nid].cpu().numpy()
            o["NF"] = E["NF"][:RR, :nn][:, nid].float().cpu().numpy()
            o["STq"] = E["STq"][:RR, :nq].cpu().numpy()
            o["DQ"] = E["DQ"][:RR, :nq].cpu().numpy()
            o["cb"] = s.cb.reshape(-1, 3)[:nq // 8].cpu().numpy()
            o["xq0"] = s.xq0.reshape(-1, 3)[:nq].cpu().numpy()
            o["Vq"] = s.Vq[:nq].cpu().numpy()
            if getattr(s, "cwS3", None) is not None:
                o["cwS3"] = s.cwS3[:RR, :16].cpu().numpy(); o["cwT3"] = s.cwT3[:RR].cpu().numpy()
            if getattr(s, "_rl_hist", None) is not None:
                o["rlh"] = s._rl_hist[:RR].cpu().numpy()
            ST["nsave"] += 1
        else:
            ST["nskip"] += 1
        np.savez_compressed(os.path.join(ST["out"], "cm174", f"f{fr:03d}.npz"), **o)
        _dump()

    def step(s, t):
        fr = int(round(t / s.h)) + 1
        r = ost(s, t)
        if ST["F0"] <= fr <= ST["F1"] and getattr(s, "_c174", None) is not None:
            try:
                save(s, fr)
            except Exception:
                import traceback
                ST["err"] = f"save f{fr} " + traceback.format_exc()[-800:]
                _dump()
        return r
    R.alloc, R.k, R.step = alloc, k, step
    import atexit
    atexit.register(_dump)
    print(f"[절 CM174] cm174 켬 -- 프레임 {ST['F0']}~{ST['F1']} 뉴턴 반복별 노드 힘 (탄성·꼬리) 분해·가우스 점 상태 -> {out}/cm174 (결과 불변)", flush=True)
    _dump()
