"""절 CM92 묶음 finterp (11-09 위임): 입자 F 보간 교체 (--fp_interp, fem_fpi.fpi_p) 의 부피 손실 확인 + 부피를 보존하는 보간 정의 두 판.

배경: H8 wolf 는 뒤집힘이 없어도 (H8N) 입자 det F 중앙 f50 0.28~0.31. 가설 = 프레임 끝 입자 F ← 가우스 점 F 보간이 부피를 잃음.
지금 정의 (fem_fpi, 절 CM20): 점마다 X_q = 프레임 끝 가우스 점 F (재질 판은 커밋 RM 뒤), 부호 있는 SVD X = U S Vᵀ, R = U Vᵀ, L = V log S Vᵀ,
  입자 F_p = polar(Σ w R) · exp(Σ w L) (w = 칸 안 삼선형 가중) -- 곧 '극분해 뒤 U 를 로그 공간에서, R 은 회전 평균' 판.
  det F_p = exp(Σ w log J_q) = 기하 평균 ≤ 산술 평균 Σ w J_q (같음은 J_q 가 모두 같을 때만). 뒤집힌 점이 끼면 유클리드 평균 Σ w X (rl 규칙).
두 수정 판 (회전·등적 부분은 지금 정의 그대로, 부피 J 만 바꿈; 새 문턱·클램프 없음; 뒤집힌 점이 낀 입자·유클리드 대체 입자는 지금 정의 그대로):
  모드 1 (J 산술): F_p = polar(Σ w R) · exp(dev Σ w L) · (Σ w J_q)^{1/3}        -- 부피 J 를 산술 (부피 선형) 보간, 등적 부분은 로그 평균.
  모드 2 (J 증분): F_p = polar(Σ w R) · exp(dev Σ w L) · (J_p⁰ · Σ w J_q / J_q⁰)^{1/3} -- 입자 자기 부피 J_p⁰ (프레임 시작) 에 칸 점 부피 비
                   (프레임 끝 / 프레임 시작 P2G 가우스 점) 의 산술 보간을 곱함 (입자 사이 부피 차를 평균으로 지우지 않음).
                   J_q⁰ ≤ 0 이나 J_p⁰ ≤ 0 이면 (로그 정의역 밖) 그 입자는 지금 정의 그대로.
  모드 0 = 지금 정의 그대로 (fpi_p 그대로, 진단만).
진단 (dg 1, 결과 불변 -- 읽기만, 출력 버퍼 따로): 프레임마다 fi.jsonl
  가우스 점 (질량 m_q 가중 평균): J_q⁰ (프레임 시작 P2G 뒤 rt_gauss F_q), J_qX (마지막 커밋 RM 앞 A F_q), J_qE (보간 원천 X_q = 커밋 뒤),
  입자 (질량 m_p 가중 평균): J_p⁰ (프레임 시작), J_pG (보간 직전 = G2P 누적 판 값, 새 실행기 탄성 판은 J_p⁰ 와 같음), J_pA (보간 직후),
  J_ar = Σ_p m_p Σ w J_q / Σ m_p (같은 가중의 산술 보간 -- 보간 규칙만의 손실 = J_ar − J_pA), J_base (모드 0 정의였다면), 유클리드 대체 질량 몫, 중앙값.
훅 (실행기 runpy 순간, 가장 바깥): ART.build_src·AORT._src 끝에 FI_SRC, RTSolver.alloc (버퍼), RTSolver.k 의 rt_gauss 뒤 · ao/aq_commit 앞 · fpi_p 자리, RTSolver.step (기록).
"""
import json
import os

import numpy as np
import torch

ST = dict(mode=0, dg=0, out=".", nsrc=0, nalloc=0, ngauss=0, ncommit=0, nfpi=0, nframe=0, err=None)
MARK = "// ---- 절 CM92 finterp"
LOG = [None]


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


FI_SRC = r"""
// ---- 절 CM92 finterp: 보간 교체의 부피 진단 + 부피 보존 보간 (모드 1 J 산술, 모드 2 J 증분) -- fpi_p 앞에 넣음 ----
__device__ __forceinline__ float fi_det(const float* M) { return M[0]*(M[4]*M[8]-M[5]*M[7]) - M[1]*(M[3]*M[8]-M[5]*M[6]) + M[2]*(M[3]*M[7]-M[4]*M[6]); }
// 프레임 시작 (rt_gauss 뒤): 가우스 점 J_q0, 입자 J_p0
__global__ void fi_jg(const int* IP, const float* Fq, float* Jq0) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    Jq0[q] = fi_det(Fq + (size_t)q*9);
}
__global__ void fi_jp(int N, const float* F, float* Jp0) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N) return;
    Jp0[p] = fi_det(F + (size_t)p*9);
}
// 커밋 앞: J(A F_q) (ao_commit·aq_commit·fpi_q 와 같은 X)
__global__ void fi_jx(const int* IP, const int* cb, const float* BG, const float* Fq, const float* U, float* JqX) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9], X[9]; for (int k=0;k<9;k++) { F[k] = Fq[(size_t)q*9+k]; X[k] = F[k]; }
    for (int o=0;o<27;o++) { float g[3]; gqn(F, BG + (oc*27+o)*3, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ua = U[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; } }
    JqX[q] = fi_det(X);
}
"""
# fpi_p 글자 고침 (모든 다른 고침 뒤 최종 글자에서; 모드 0 이면 out 그대로 -> F·상태 결과 같음)
SIG_ADD = ", int fi_mode, const float* fi_Jq0, const float* fi_Jp0, float* fi_dg"
C_OLD = "    if (c < 0) return;\n"
C_NEW = ("    if (c < 0) return;\n"
         "    float fi_sJ = 0.f, fi_sJr = 0.f, fi_ws = 0.f; bool fi_j0b = false; float fi_jpG = fi_det(F + (size_t)p*9);   // 절 CM92\n")
E_OLD = "        const float* e = Fe + (size_t)(c*8 + oc)*28;\n"
E_NEW = ("        const float* e = Fe + (size_t)(c*8 + oc)*28;\n"
         "        { float jq_ = fi_det(e + 18); fi_ws += w; fi_sJ += w*jq_; if (fi_Jq0) { float j0_ = fi_Jq0[c*8 + oc]; if (j0_ > 0.f) fi_sJr += w*jq_/j0_; else fi_j0b = true; } }   // 절 CM92\n")
OK_OLD = "    bool ok = true;\n"
OK_NEW = ("    {   // 절 CM92: 부피 J 만 모드별 (회전·로그 경로이고 뒤집힌 점이 안 낀 입자만; 유클리드 대체·뒤집힘은 지금 정의 그대로)\n"
          "        if (fi_ws > 0.f) { fi_sJ /= fi_ws; fi_sJr /= fi_ws; }\n"
          "        float jb_ = fi_det(out), jp0_ = fi_Jp0 ? fi_Jp0[p] : 0.f;\n"
          "        bool rsp_ = !anyb && !bad;\n"
          "        if (rsp_) {\n"
          "            float trL_ = sL[0] + sL[4] + sL[8];\n"
          "            if (fi_mode == 1) { float sc_ = expf((logf(fi_sJ) - trL_) / 3.f); for (int k=0;k<9;k++) out[k] *= sc_; }\n"
          "            else if (fi_mode == 2 && !fi_j0b && jp0_ > 0.f) { float sc_ = expf((logf(jp0_ * fi_sJr) - trL_) / 3.f); for (int k=0;k<9;k++) out[k] *= sc_; }\n"
          "        }\n"
          "        if (fi_dg) { float* d_ = fi_dg + (size_t)p*8; d_[0] = jp0_; d_[1] = fi_jpG; d_[2] = fi_det(out); d_[3] = fi_sJ; d_[4] = jb_; d_[5] = rsp_ ? 0.f : 1.f; d_[6] = anyb ? 1.f : 0.f; d_[7] = fi_j0b ? 1.f : 0.f; }\n"
          "    }\n"
          "    bool ok = true;\n")
HEAD = "__global__ void fpi_p("


def patch(code):
    """최종 CUDA 글자 (cupy RawModule 직전) 의 fpi_p 에 절 CM92 고침 + 앞에 FI_SRC."""
    if MARK in code or HEAD not in code:
        return code
    assert code.count(HEAD) == 1
    i = code.index(HEAD)
    j = code.find("__global__", i + len(HEAD)); j = len(code) if j < 0 else j
    b = code[i:j]
    k = b.index(") {")
    b = b[:k] + SIG_ADD + b[k:]
    for o_, n_ in ((C_OLD, C_NEW), (E_OLD, E_NEW), (OK_OLD, OK_NEW)):
        assert b.count(o_) == 1, ("절 CM92 fpi_p 고칠 글자", o_, b.count(o_))
        b = b.replace(o_, n_, 1)
    ST["nsrc"] += 1
    return code[:i] + MARK + "\n" + FI_SRC + b + code[j:]


def _dump():
    try:
        with open(os.path.join(ST["out"], "fiinfo.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def _wm(v, w):
    ok = torch.isfinite(v) & (w > 0)
    if not bool(ok.any()):
        return None
    v = v[ok].double(); w = w[ok].double()
    return float((v * w).sum() / w.sum())


def _hook(R, fa):
    oa, ok, os_ = R.alloc, R.k, R.step

    def alloc(s, need):
        r = oa(s, need); fa(s)
        return r

    def on(s):
        return getattr(s, "fiJ0", None) is not None and getattr(s, "Fe", None) is not None

    def k(s, name, n, args, b=128):
        if not on(s):
            return ok(s, name, n, args, b)
        i32 = np.int32
        if name == "rt_gauss" and (ST["mode"] == 2 or ST["dg"]):                       # J_q0·J_p0 는 모드 2 와 진단만 씀
            r = ok(s, name, n, args, b)
            ok(s, "fi_jg", n, (args[0], P(s.Fq), P(s.fiJ0)), 128)
            ok(s, "fi_jp", s.N, (i32(s.N), P(s.F), P(s.fiJp0)), 256)
            ST["ngauss"] += 1
            return r
        if name in ("ao_commit", "aq_commit") and ST["dg"]:
            ok(s, "fi_jx", n, tuple(args[:5]) + (P(s.fiJX),), 128)
            ST["ncommit"] += 1
            return ok(s, name, n, args, b)
        if name == "fpi_p":                                                            # args: N IP fx pcd cidx Fe rl guard F Sq ysp ljp alp (+ 절 CM92 넷)
            ST["nfpi"] += 1
            return ok(s, name, n, tuple(args) + (i32(ST["mode"]), P(s.fiJ0), P(s.fiJp0), P(s.fiD)), b)
        return ok(s, name, n, args, b)

    def step(s, *a, **kw):
        r = os_(s, *a, **kw)
        if on(s):
            ST["nframe"] += 1
            if LOG[0] is not None and s.fiD is not None:
                try:
                    nq = int(s.IP[12].item()); N = s.N
                    mq = s.mq[:nq]; m = s.m[:N]
                    D = s.fiD[:N * 8].view(N, 8)
                    JE = torch.linalg.det(s.Fe[:nq, 18:27].reshape(nq, 3, 3).double()).float()
                    jn = D[:, 2]
                    fin = torch.isfinite(jn)
                    o = dict(f=ST["nframe"], t=float(a[0]) if a else None, nq=nq, N=N,
                             Jq0=_wm(s.fiJ0[:nq], mq), JqX=_wm(s.fiJX[:nq], mq), JqE=_wm(JE, mq),
                             Jp0=_wm(D[:, 0], m), JpG=_wm(D[:, 1], m), JpA=_wm(jn, m), Jar=_wm(D[:, 3], m), Jbase=_wm(D[:, 4], m),
                             euc=_wm(D[:, 5], m), invq=_wm(D[:, 6], m), j0b=_wm(D[:, 7], m),
                             medA=float(jn[fin].median()) if bool(fin.any()) else None, med0=float(D[:, 0].median()),
                             nneg=int((jn <= 0).sum()), qneg=int(((JE <= 0) & (mq > 0)).sum()))
                    LOG[0].write(json.dumps(o) + "\n"); LOG[0].flush()
                except Exception as e:                                                 # 진단만 (결과 불변)
                    ST["err"] = repr(e)
            if ST["nframe"] in (1, 2, 5) or ST["nframe"] % 25 == 0:
                _dump()
        return r

    R.alloc, R.k, R.step = alloc, k, step


def install(out, mode=0, dg=0):
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    ST.update(mode=int(mode), dg=int(dg), out=out)
    if ST["dg"]:
        LOG[0] = open(os.path.join(out, "fi.jsonl"), "w")
    cpm = ART.cp
    orm = cpm.RawModule

    def RawModule(*a, **kw):                                                           # 최종 글자 (다른 묶음 고침 모두 뒤) 에서 fpi_p 만
        if "code" in kw:
            kw["code"] = patch(kw["code"])
        elif a:
            a = (patch(a[0]),) + tuple(a[1:])
        return orm(*a, **kw)
    cpm.RawModule = RawModule

    def fa(s):                                                                         # --fp_interp 판만 (Fe 있음), 용량 바뀌면 다시
        if getattr(s, "Fe", None) is None:
            return
        capQ = s.Fe.shape[0]; NP = s.F.numel() // 9
        if getattr(s, "fiJ0", None) is not None and s.fiJ0.numel() == capQ and s.fiJp0.numel() == NP:
            return
        Z = lambda *sh: torch.zeros(*sh, device=s.dev, dtype=torch.float32)
        s.fiJ0 = Z(capQ); s.fiJX = Z(capQ); s.fiJp0 = Z(NP)
        s.fiD = Z(NP * 8) if ST["dg"] else None
        ST["nalloc"] += 1
    from . import fem_an_rt_v0 as ART0                                                # 옛 실행기 (수박) 의 바탕 RTSolver 는 fem_an_rt_v0 (fem_ao_rt·fem_det_rt)
    for R in (ART.RTSolver, ART0.RTSolver):
        _hook(R, fa)
    oao = AORT.AORTSolver.alloc

    def alloc_ao(s, need):                                                             # 옛 실행기: Fe 는 RTSolver.alloc 뒤에 생김
        r = oao(s, need); fa(s)
        return r

    AORT.AORTSolver.alloc = alloc_ao
    print(f"[절 CM92] finterp 켬 -- 모드 {ST['mode']} (0 지금 정의, 1 J 산술, 2 J 증분), 진단 {ST['dg']}", flush=True)
    _dump()
