"""절 CM41 수박 깨짐 모양 2 -- C2 (--rm_sub n, 걸음 안 RM n 등분) 의 손상 과다 원인 진단 + 고침 후보 (수박 옛 실행기 exe/fem_ao_rt_sim.py 만, --rm_sub 판 위).
플래그를 안 주면 이 모듈을 안 부름 (생성 소스·커널 열·프레임 경로 그대로). 소스 바꾸기는 fem_wmfrac.apply (rmsub) 결과에 글자 치환으로만.

진단 --ws2 1 (결과 불변 읽기): 커밋 (aq_commit) 의 n 등분 RM 각 호출을 NACC 갈래로 나눠 logJp 증분을 셈 (장치 전역 AF_W2B, 프레임마다 읽고 0; 상태 되돌림 _rs 때도 0 -> 받아들인 시도만)
  갈래 b: 0 압축 꼭짓점 (p_tr > p0, 다져짐) · 1 인장 꼭짓점 (p_tr < −β p0) · 2 타원 전단 인장 쪽 (p_tr < p_c = (p0 − β p0)/2; 경화식이 logJp 를 올림 = 연화)
          · 3 타원 전단 압축 쪽 (p_tr ≥ p_c; 내림) · 4 안 (바뀜 없음). 칸 [b*4 + 0] Σ 양 증분 · [1] Σ 음 증분 · [2] 바뀐 호출 수 · [3] 그 호출에서 logJp 가 0 을 넘은 수.
  [24..39] 양 증분 크기 log10 막대 (−6 ~ 2, 0.5 간격) · [40..55] 가우스 점 걸음 변형 증분 ‖sym(∇U − ∇U0)‖ log10 막대 (−5 ~ 3)
  [56] 이 커밋에서 logJp 가 0 을 넘은 가우스 점 수 · [57] 그 점들의 변형 증분 합 · [58] 등분 n > 1 인 점 수 · [59] Σ n · [60] 변형 증분 최대 (int 비트 최대).
  + 입자: 손상 입자·새 손상 입자의 탄성 J·전체 부피 비 J·Jp 분위, logJp 크기 구간 (0~0.01·0.01~0.1·0.1~0.4·> 0.4) 몫.
후보 (a) --ws2_eps ε (> 0): 가우스 점마다 등분 수 n = clamp(⌈‖sym(∇U − ∇U0)‖ / ε⌉, 1, --rm_sub) (elem·commit 같은 식이라 같은 n; --rm_sub 가 상한).
후보 (b) --ws2_fc 'N,C' : 첫 접촉 프레임 (프레임 끝 입자 최저 높이 < --ws2_fch 가 처음인 프레임) 을 되돌려 뉴턴·PCG 상한 N·C 로 다시 (그래프 따로),
  그 결과 입자 에너지 (운동 + 중력 위치 + SNH 탄성) 가 프레임 시작보다 --ws2_fctol J 넘게 늘면 버리고 보통 결과를 받음 (에너지 생성 없음 조건).
"""
import math

import numpy as np
import torch

NB = 64
DECL = "\n__device__ float* AF_W2B;"
BR_FN = r"""
// ---- 절 CM41 (fem_wmshape2): NACC 갈래 (pl_rm mat 7 과 같은 식으로 판정만) · 등분 수 ----
__device__ int w2_br(const float* s, const float* prm, float lj) {
    float mu = prm[0], kap = prm[2], M = prm[4], beta = prm[5], xi = prm[6];
    float sg[3]; for (int i=0;i<3;i++) sg[i] = fmaxf(s[i], 0.f);
    float p0 = kap*(0.00001f + sinhf(xi*fmaxf(-lj, 0.f)));
    float J = sg[0]*sg[1]*sg[2];
    float B[3]; for (int i=0;i<3;i++) B[i] = sg[i]*sg[i];
    float Bm = (B[0] + B[1] + B[2])/3.f;
    float Jp = powf(J, -2.f/3.f);
    float sh[3]; for (int i=0;i<3;i++) sh[i] = mu*Jp*(B[i] - Bm);
    float ptr = -kap/2.f*(J - 1.f/J)*J;
    float ysh = 1.5f*(1.f + 2.f*beta);
    float yph = M*M*(ptr + beta*p0)*(ptr - p0);
    float y = ysh*(sh[0]*sh[0] + sh[1]*sh[1] + sh[2]*sh[2]) + yph;
    float pmin = beta*p0;
    if (ptr > p0) return 0;
    if (ptr < -pmin) return 1;
    if (y >= 1e-4f) return (ptr < 0.5f*(p0 - pmin)) ? 2 : 3;
    return 4;
}
__device__ float w2_de(const float* A0, const float* A1) {
    float e = 0.f;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float d = 0.5f*((A1[a*3+b] - A0[a*3+b]) + (A1[b*3+a] - A0[b*3+a])); e += d*d; }
    return sqrtf(e);
}
__device__ int w2_n(const float* A0, const float* A1, const float* prm) {
    int n = (int)(prm[12] + 0.5f); if (n < 1) n = 1;
#if AF_W2EPS_ON
    float e = w2_de(A0, A1);
    int m = (int)ceilf(e / AF_W2EPS);
    if (!(m >= 1)) m = 1;
    if (m < n) n = m;
#endif
    return n;
}
__device__ __forceinline__ int w2_bin(float v, float lo) { int b = (int)floorf((log10f(fmaxf(v, 1e-30f)) - lo)*2.f); return b < 0 ? 0 : (b > 15 ? 15 : b); }
"""
N_OLD = "    int n = (int)(prm[12] + 0.5f); if (n < 1) n = 1;\n"
N_NEW = "    int n = w2_n(A0, A1, prm);                                                   // 절 CM41: 등분 수 (--ws2_eps 이면 변형 증분에서)\n"
RM_OLD = "        bool ch = (prm[15] > 0.5f) ? false : pl_rm(mat, se, prm, ys, lj, al, dt);\n        any = any || ch;\n"
RM_DIAG = ("        int br_ = (mat == 7) ? w2_br(se, prm, lj) : 4; float lj0_ = lj;\n"
           "        bool ch = (prm[15] > 0.5f) ? false : pl_rm(mat, se, prm, ys, lj, al, dt);\n        any = any || ch;\n"
           "        if (AF_W2B && mat == 7) { float dl_ = lj - lj0_;                       // 절 CM41 진단: 갈래별 logJp 증분\n"
           "            if (dl_ > 1e-7f || dl_ < -1e-7f) { atomicAdd(&AF_W2B[br_*4 + (dl_ > 0.f ? 0 : 1)], dl_); atomicAdd(&AF_W2B[br_*4 + 2], 1.f);\n"
           "                if (lj0_ <= 0.f && lj > 0.f) atomicAdd(&AF_W2B[br_*4 + 3], 1.f);\n"
           "                if (dl_ > 0.f) atomicAdd(&AF_W2B[24 + w2_bin(dl_, -6.f)], 1.f); } }\n")
SUB_HEAD = "__device__ bool rms_sub("
COMMIT_CALL = "rms_sub(mat, prm, ys, lj, al, dt, F, A, ids, BG, oc, Uu, s, Vv);"
COMMIT_DIAG = ("w2_rms_c(mat, prm, ys, lj, al, dt, F, A, ids, BG, oc, Uu, s, Vv);")
COMMIT_PRE = "    float Uu[9], s[3], Vv[9];\n    float ys = Sq[(size_t)q*4], lj = Sq[(size_t)q*4+1], al = Sq[(size_t)q*4+2];\n"
COMMIT_POST_OLD = "    Sq[(size_t)q*4] = ys; Sq[(size_t)q*4+1] = lj; Sq[(size_t)q*4+2] = al;\n"
COMMIT_POST_NEW = ("    if (AF_W2B && mat == 7) { float lq0_ = Sq[(size_t)q*4+1];                      // 절 CM41 진단: 점별 변형 증분·0 넘음·등분 수\n"
                   "        float A0_[9] = {0,0,0,0,0,0,0,0,0};\n"
                   "        for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; int id = ids[o]; for (int a=0;a<3;a++) { float ua = AQ_U0[id*3+a]; A0_[a*3] += ua*bg[0]; A0_[a*3+1] += ua*bg[1]; A0_[a*3+2] += ua*bg[2]; } }\n"
                   "        float de_ = w2_de(A0_, A); int nn_ = w2_n(A0_, A, prm);\n"
                   "        atomicAdd(&AF_W2B[40 + w2_bin(de_, -5.f)], 1.f);\n"
                   "        if (lq0_ <= 0.f && lj > 0.f) { atomicAdd(&AF_W2B[56], 1.f); atomicAdd(&AF_W2B[57], de_); }\n"
                   "        if (nn_ > 1) atomicAdd(&AF_W2B[58], 1.f); atomicAdd(&AF_W2B[59], (float)nn_);\n"
                   "        atomicMax((int*)&AF_W2B[60], __float_as_int(de_)); }\n"
                   + COMMIT_POST_OLD)


def _rep(s, o, n, cnt=1):
    assert s.count(o) == cnt, (s.count(o), o[:120])
    return s.replace(o, n)


def _fn_body(r, head):
    i = r.index(head)
    j = r.index("\n}\n", i) + 3
    return i, j


def src_apply(r, diag, eps):
    """r = fem_wmfrac.apply 결과 (rmsub 판). diag = 진단, eps = 후보 (a) 문턱 (0 = 끔)."""
    assert SUB_HEAD in r, "절 CM41 은 --rm_sub 판 위에만"
    r = r.replace("extern \"C\" {", "extern \"C\" {" + DECL + "\n#define AF_W2EPS_ON %d\n#define AF_W2EPS %.9ef" % (1 if eps > 0 else 0, float(eps) if eps > 0 else 1.0), 1)
    i, j = _fn_body(r, SUB_HEAD)
    fn = r[i:j]
    fn = _rep(fn, N_OLD, N_NEW)
    fc = fn.replace(SUB_HEAD, "__device__ bool w2_rms_c(", 1)
    fc = _rep(fc, RM_OLD, RM_DIAG) if diag else fc
    r = r[:i] + BR_FN + fn + fc + r[j:]
    k = r.index("__global__ void aq_commit(")
    e = r.find("__global__", k + 10)
    body = r[k:e]
    body = _rep(body, COMMIT_CALL, COMMIT_DIAG)
    if diag:
        body = _rep(body, COMMIT_POST_OLD, COMMIT_POST_NEW)
    return r[:k] + body + r[e:]


CFG = dict(diag=0, eps=0.0, fc=None, fch=0.02, fctol=0.01, fcn=1)


def install(a):
    """소스·alloc·bind·_rs 감싸기 (S 만들기 전에)."""
    from . import fem_ao_rt as AORT
    from . import fem_wmfrac as WF
    CFG.update(diag=int(a.ws2), eps=float(a.ws2_eps), fch=float(a.ws2_fch), fctol=float(a.ws2_fctol), fcn=int(a.ws2_fcn))
    if a.ws2_fc:
        CFG["fc"] = tuple(int(q) for q in a.ws2_fc.split(","))
    if getattr(AORT, "_ws2", False):
        return
    AORT._ws2 = True
    o_apply = WF.apply

    def apply(r, wf):
        r = o_apply(r, wf)
        if int(wf.get("rmsub", 0)) > 0:
            r = src_apply(r, CFG["diag"], CFG["eps"])
        return r
    WF.apply = apply
    o_alloc = AORT.AORTSolver.alloc

    def alloc(s, need):
        nt0 = getattr(s, "nt", None)
        if CFG["fc"] and nt0 is not None:
            s.nt = max(nt0, CFG["fc"][0])                                              # 후보 (b): 반복별 버퍼는 큰 상한으로
        try:
            o_alloc(s, need)
        finally:
            if nt0 is not None:
                s.nt = nt0
        s.w2_graphs = {}                                                               # 다시 할당 뒤 후보 (b) 그래프도 무효
        if not hasattr(s, "w2b"):
            s.w2b = torch.zeros(NB, device=s.dev, dtype=torch.float32)
    AORT.AORTSolver.alloc = alloc
    o_bind = AORT.AORTSolver.bind_act

    def bind_act(s):
        o_bind(s)
        if CFG["diag"] and hasattr(s, "w2b"):
            g = s.mod.get_global("AF_W2B")
            AORT.ART.cp.ndarray((1,), dtype=AORT.ART.cp.uint64, memptr=g)[0] = int(s.w2b.data_ptr())
    AORT.AORTSolver.bind_act = bind_act
    o_rs = AORT.AORTSolver._rs

    def _rs(s, bk):
        o_rs(s, bk)
        if hasattr(s, "w2b"):
            s.w2b.zero_()
    AORT.AORTSolver._rs = _rs
    if CFG["fc"]:
        from . import fem_wmrx as WRX
        WRX.split_col = _fc_wrap(WRX.split_col)


ST = dict(S=None, mu=None, lam=None, g=None, NR=None, PN=None, done=0, seen=False, log=[], lprev=None)


def setup(S, mu, lam, g, FL):
    ST.update(S=S, mu=float(mu), lam=float(lam), g=[float(q) for q in g], NR=FL["NR"], PN=float(FL["PN"]))


def energy(S):
    from . import fem_wmkick_diag as FRD
    mp, xp, vp = S.m.double(), S.x.double(), S.v.double()
    ok = torch.isfinite(xp).all(-1) & torch.isfinite(vp).all(-1)
    gv = torch.tensor(ST["g"], device=xp.device, dtype=torch.float64)
    ke = float(0.5 * (mp[ok] * (vp[ok] ** 2).sum(1)).sum())
    pe = float(-(mp[ok] * (xp[ok] @ gv)).sum())
    ee = FRD._snh(S.F[ok], S.vol[ok], ST["mu"], ST["lam"])[0]
    return ke + pe + ee, ke, ee


def hmin(S):
    nr = torch.tensor(ST["NR"], device=S.x.device, dtype=torch.float64)
    d = S.x.double() @ nr - ST["PN"]
    d = d[torch.isfinite(d)]
    return float(d.min()) if d.numel() else float("nan")


def _fc_wrap(orig):
    def split_col(s, t, h, sub_dt, depth, fn=None):
        if ST["done"] >= CFG["fcn"] or ST["S"] is None:
            return orig(s, t, h, sub_dt, depth, fn)
        bk = s._bk()
        E0 = energy(s)
        r = orig(s, t, h, sub_dt, depth, fn)
        if not (ST["seen"] or hmin(s) < CFG["fch"]):
            return r
        ST["seen"] = True
        ms0 = float(s.last_ms)
        Ea = energy(s)
        bka = s._bk()
        s._rs(bk)
        base = (s.nt, s.cg, s.graphs, s.graph)
        s.nt, s.cg = CFG["fc"]
        gb = s.graphs = getattr(s, "w2_graphs", {})
        s.graph = s.graphs.get(s.Kcur)
        try:
            r2 = orig(s, t, h, sub_dt, depth, fn)
        finally:
            realloc = s.graphs is not gb                                               # 용량 늘려 다시 할당했으면 (alloc 이 graphs 를 새 dict 로) 옛 그래프 모두 무효
            s.w2_graphs = s.graphs
            s.nt, s.cg, s.graphs, s.graph = base
            if realloc:
                s.graphs = {}
            s.graph = s.graphs.get(s.Kcur)
        ms1 = float(s.last_ms)
        Eb = energy(s)
        acc = math.isfinite(Eb[0]) and Eb[0] <= E0[0] + CFG["fctol"]
        if not acc:
            s._rs(bka)
            r2 = r
        s.last_ms = ms0 + ms1
        ST["done"] += 1
        ST["log"].append(dict(t=int(round(t / h)) + 1, E0=E0, Ea=Ea, Eb=Eb, acc=bool(acc), ms=[ms0, ms1]))
        print(f"[절 CM41 첫 접촉] 프레임 {int(round(t / h)) + 1}: E 시작 {E0[0]:.3f} 보통 {Ea[0]:.3f} (운동 {Ea[1]:.3f} 탄성 {Ea[2]:.3f}) "
              f"뉴턴 {CFG['fc'][0]}·PCG {CFG['fc'][1]} {Eb[0]:.3f} (운동 {Eb[1]:.3f} 탄성 {Eb[2]:.3f}) -> {'받음' if acc else '버림'}", flush=True)
        return r2
    return split_col


def _q(t, qs):
    if t.numel() == 0:
        return None
    t = t.float()
    if t.numel() > 8_000_000:
        t = t[torch.randperm(t.numel(), device=t.device)[:8_000_000]]
    return [round(float(v), 5) for v in torch.quantile(t, torch.tensor(qs, device=t.device))]


def frame(fi):
    """프레임 끝 (읽기만): 커밋 갈래 기록 + 입자 손상 상태. 돌려줌 dict (fem.jsonl ws2)."""
    S = ST["S"]
    o = {}
    if CFG["diag"] and hasattr(S, "w2b"):
        b = S.w2b.cpu().numpy().astype(np.float64)
        o["br"] = [[float(b[k * 4 + j]) for j in range(4)] for k in range(5)]
        o["hdl"] = [int(q) for q in b[24:40]]
        o["hde"] = [int(q) for q in b[40:56]]
        o["x0"] = [int(b[56]), float(b[57])]
        o["nsub"] = [int(b[58]), float(b[59])]
        o["demax"] = float(b[60])                                                     # atomicMax 로 float 비트를 넣었으므로 float 로 읽으면 값
        S.w2b.zero_()
    if int(S.IP[13]) == 0 and getattr(S, "ljp", None) is not None:
        ok = torch.isfinite(S.x).all(-1) & torch.isfinite(S.F.reshape(S.F.shape[0], -1)).all(-1)
        lj = S.ljp[ok].double()
        J = torch.linalg.det(S.F[ok].reshape(-1, 3, 3).double())
        Jt = J * torch.exp(lj + 0.04)
        dm = lj > 0
        lo = torch.empty_like(S.ljp); lo[S.perm] = S.ljp
        o["lb"] = [float(((lj > lo_) & (lj <= hi_)).double().mean()) for lo_, hi_ in ((0.0, 0.01), (0.01, 0.1), (0.1, 0.4), (0.4, 1e9))]
        o["dJ"] = _q(J[dm], [0.1, 0.5, 0.9])
        o["dJt"] = _q(Jt[dm], [0.1, 0.5, 0.9])
        if ST["lprev"] is not None:
            lp = ST["lprev"][S.perm][ok].double()
            nw = dm & ~(lp > 0)
            o["nJ"] = _q(J[nw], [0.1, 0.5, 0.9])
            o["nJt"] = _q(Jt[nw], [0.1, 0.5, 0.9])
            dl = lj - lp
            o["dlq"] = _q(dl, [0.01, 0.5, 0.99])
        ST["lprev"] = lo.clone()
    if ST["log"]:
        o["fc"] = ST["log"][:]
        ST["log"].clear()
    return o
