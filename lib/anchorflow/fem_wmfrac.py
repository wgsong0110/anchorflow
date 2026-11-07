"""절 CM28 (사용자 위임 11-08 정의 변경, 옛 재질 실행기 수박 재질 7 만): 원본 GF NACC 와 이식의 차이를 하나씩 없애는 판.
fem_ao_rt._src 가 만든 소스에 글자 치환으로만 넣는다 -- 모두 끄면 (wf 비어 있음) 소스 글자 그대로.

(a) seedb  씨 영역 β: 입자 ys 자리 (재질 7 은 ys 를 안 씀) 에 입자 β (1 또는 prm[11] = 3e9, GF 러너 판정 exe/wm_seedmask.py) 를 두고,
           가우스 점 ys = 팔분 입자 부피 가중 평균 (지금 상태 규칙 그대로) -> RM 의 β = (ys ≥ (prm[5] + prm[11]) / 2) ? prm[11] : prm[5]
           (= 씨 입자 부피 과반이면 씨 β). 입자 쪽 RM 도 같은 식 (입자 ys 는 1 또는 3e9 그대로). fpi_p 가 ys 를 보간으로 덮지 않음 (β 는 입자 고정값).
(b) ljq    가우스 점 logJp 모으기: "p0" = 응집력 평균 -- S = 부피 가중 평균 sinh(ξ max(−logJp, 0)) (p0 = κ (1e-5 + S) 의 평균과 같음),
           S > 0 이면 logJp_q = −asinh(S)/ξ (그 응집력을 주는 logJp), 팔분 입자가 모두 logJp ≥ 0 (S = 0) 이면 산술 평균 (≥ 0, 손상 크기 유지).
           "max" = 팔분 입자 logJp 최대 (가장 손상된 입자). 두 판 모두 al 자리를 모으기에 쓰고 가우스 점 al = 1 (재질 7 은 al 을 안 씀).
(c) rmsub  걸음 안 소성 갱신 잘게 (가우스 점만; 입자는 --fp_interp 가 프레임 끝 가우스 점 보간으로 덮음): 걸음 시작 변위 U0 -> 시험 변위 y 를
           n = prm[12] 등분, A(s) = ∇U0 + s (∇y − ∇U0), X(s) = (I + A(s)) F_q, F_e ← RM(X(s_i) X(s_{i−1})⁻¹ F_e) 를 n 번 (logJp 누적),
           F_e 처음 = X(0) (앞 걸음 커밋 뒤 탄성 상태). 응력 P = τ(F_e) X⁻ᵀ (τ = U_e diag(τ̂) U_eᵀ); 접선은 지금처럼 시험 σ 의 탄성 접선.
           커밋 (aq_commit) 도 U0 -> U 끝 같은 n 등분. n = 1 이면 지금과 같은 식 (반올림만 다름).
(d) rmsnh  RM 판정·투영 식을 SNH 탄성과 맞춤: 키르히호프 τ_i = μ σ_i² + (λ_s (J − 1) − μ) J (λ_s = λ + μ) ->
           p = −μ B̄ − (λ_s (J − 1) − μ) J, ŝ_i = μ (B_i − B̄) (J^{-2/3} 없음), 항복면 투영 B_new = ŝ_new/μ + B̄,
           꼭짓점·경화의 Je = p_iso(Je) = p 의 근 (p_iso(J) = −μ J^{2/3} − (λ_s (J − 1) − μ) J, J ≥ 0.5 줄어드는 갈래 이분법 40 회; NHB 의 √(1 − 2p/κ) 자리).
           J ≤ 0 (뒤집혀 σ 가 0 으로 깎인 점) 은 logJp 갱신 안 함 (NHB 판은 NaN 으로 갱신이 안 되던 자리).
"""

PLRM_HEAD = "__device__ __forceinline__ bool pl_rm(int mat, float* s, const float* prm, float& ys, float& lj, float& al, float dt) {"
RMS_HELP = r"""
// 절 CM28 (d): SNH 등방 상태 압력 p_iso(J) = −μ J^{2/3} − (λ_s (J − 1) − μ) J 와 p_iso(Je) = pt 의 근 (Je ≥ 0.5, p_iso 가 줄어드는 갈래, 이분법)
__device__ __forceinline__ float rms_piso(float J, float mu, float ls) { return -mu*powf(J, 2.f/3.f) - (ls*(J - 1.f) - mu)*J; }
__device__ float rms_je(float pt, float mu, float ls) {
    float lo = 0.5f, hi = 1.f;
    if (!(pt == pt)) return 1.f;
    if (rms_piso(lo, mu, ls) <= pt) return lo;
    while (rms_piso(hi, mu, ls) > pt && hi < 1e7f) { lo = hi; hi *= 2.f; }
    for (int it=0; it<40; it++) { float md = 0.5f*(lo + hi); if (rms_piso(md, mu, ls) > pt) lo = md; else hi = md; }
    return 0.5f*(lo + hi);
}
"""
SEED_OLD = "        float kap = prm[2], M = prm[4], beta = prm[5], xi = prm[6];\n"
SEED_NEW = "        float kap = prm[2], M = prm[4], beta = (ys >= 0.5f*(prm[5] + prm[11])) ? prm[11] : prm[5], xi = prm[6];   // 절 CM28 (a) 씨 영역 β\n"
SNH_RM = [
    ("        float Jp = powf(J, -2.f/3.f);\n        float sh[3]; for (int i=0;i<3;i++) sh[i] = mu*Jp*(B[i] - Bm);\n        float prime = kap/2.f*(J - 1.f/J);\n        float ptr = -prime*J;\n",
     "        float lss_ = prm[1] + mu;                                         // 절 CM28 (d): SNH 키르히호프의 p·ŝ\n"
     "        float sh[3]; for (int i=0;i<3;i++) sh[i] = mu*(B[i] - Bm);\n        float ptr = -mu*Bm - (lss_*(J - 1.f) - mu)*J;\n"),
    ("            float Je = sqrtf(-2.f*p0/kap + 1.f);\n", "            float Je = rms_je(p0, mu, lss_);\n"),
    ("            float Je = sqrtf(2.f*pmin/kap + 1.f);\n", "            float Je = rms_je(-pmin, mu, lss_);\n"),
    ("            float J2 = powf(J, 2.f/3.f);\n            float sc = J2/mu*sqf/shn;\n", "            float sc = sqf/(mu*shn);\n"),
    ("                float Jef = sqrtf(fabsf(-2.f*pf/kap + 1.f));\n                if (Jef > 1e-4f && hard) ljn = lj + logf(J/Jef);\n",
     "                float Jef = rms_je(pf, mu, lss_);\n                if (Jef > 1e-4f && hard && J > 0.f) ljn = lj + logf(J/Jef);\n"),
]
SNH_HARD_OLD = "            if (hard) ljn = lj + logf(J/Je);\n"
SNH_HARD_NEW = "            if (hard && J > 0.f) ljn = lj + logf(J/Je);\n"
FPI_YS_OLD = "    if (Sq) { ysp[p] = s3[0]; ljp[p] = s3[1]; alp[p] = s3[2]; }\n"
FPI_YS_NEW = "    if (Sq) { ljp[p] = s3[1]; alp[p] = s3[2]; }                           // 절 CM28 (a): ys = 입자 β 고정\n"
P2G_OLD = "    float v4[4] = {w*ys[p], w*lj[p], w*al[p], w};\n    for (int k=0;k<4;k++) { atomicAdd(&Sa[(size_t)q*4+k], v4[k]); atomicAdd(&Sc[(size_t)c*4+k], v4[k]); }\n"
P2G_P0 = ("    float v4[4] = {w*ys[p], w*lj[p], w*sinhf(AF_LJQ_XI*fmaxf(-lj[p], 0.f)), w};          // 절 CM28 (b) p0: al 자리에 응집력 몫 sinh(ξ max(−logJp, 0))\n"
          "    for (int k=0;k<4;k++) { atomicAdd(&Sa[(size_t)q*4+k], v4[k]); atomicAdd(&Sc[(size_t)c*4+k], v4[k]); }\n")
P2G_MAX = ("    float v4[4] = {w*ys[p], w*lj[p], 0.f, w};                               // 절 CM28 (b) max: al 자리에 max(logJp + 64) (음 아닌 float 은 int 순서 = 값 순서)\n"
           "    for (int k=0;k<4;k++) if (k != 2) { atomicAdd(&Sa[(size_t)q*4+k], v4[k]); atomicAdd(&Sc[(size_t)c*4+k], v4[k]); }\n"
           "    { int mv_ = __float_as_int(fmaxf(lj[p] + 64.f, 0.f)); atomicMax((int*)&Sa[(size_t)q*4+2], mv_); atomicMax((int*)&Sc[(size_t)c*4+2], mv_); }\n")
GS_OLD = "    Sq[(size_t)q*4] = a[0]/w; Sq[(size_t)q*4+1] = a[1]/w; Sq[(size_t)q*4+2] = a[2]/w; Sq[(size_t)q*4+3] = 0.f;\n"
GS_P0 = "    { float S_ = a[2]/w; Sq[(size_t)q*4] = a[0]/w; Sq[(size_t)q*4+1] = (S_ > 0.f) ? -asinhf(S_)/AF_LJQ_XI : a[1]/w; Sq[(size_t)q*4+2] = 1.f; Sq[(size_t)q*4+3] = 0.f; }\n"
GS_MAX = "    Sq[(size_t)q*4] = a[0]/w; Sq[(size_t)q*4+1] = a[2] - 64.f; Sq[(size_t)q*4+2] = 1.f; Sq[(size_t)q*4+3] = 0.f;\n"
ELEM_HEAD = "__global__ void __launch_bounds__(128) aq_elem("
RMSUB_FN = r"""
// 절 CM28 (c): 걸음 안 소성 갱신 n 등분 (가우스 점). A0 = ∇U0 (AQ_U0 = 걸음 시작 변위 사본), A1 = 시험 변위 기울기, X(s) = (I + A0 + s (A1 − A0)) F.
//   F_e = X(0) 에서 시작, i = 1..n: F_tr = X(s_i) X(s_{i−1})⁻¹ F_e, RM -> F_e = U diag(σ̂) Vᵀ. 끝 SVD (Ue, se, Ve) 를 돌려줌.
__device__ bool rms_sub(int mat, const float* prm, float& ys, float& lj, float& al, float dt, const float* F, const float* A1, const int* ids, const float* BG, int oc,
                        float* Ue, float* se, float* Ve) {
    float A0[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; int id = ids[o]; for (int a=0;a<3;a++) { float ua = AQ_U0[id*3+a]; A0[a*3] += ua*bg[0]; A0[a*3+1] += ua*bg[1]; A0[a*3+2] += ua*bg[2]; } }
    int n = (int)(prm[12] + 0.5f); if (n < 1) n = 1;
    float Xp[9], Fe[9];
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Xp[a*3+b] = F[a*3+b] + A0[a*3]*F[b] + A0[a*3+1]*F[3+b] + A0[a*3+2]*F[6+b];
    for (int k=0;k<9;k++) Fe[k] = Xp[k];
    bool any = false;
    for (int i=1;i<=n;i++) {
        float t = (float)i/(float)n, Ai[9], Xi[9], Xpi[9], T1[9], Ft[9];
        for (int k=0;k<9;k++) Ai[k] = A0[k] + t*(A1[k] - A0[k]);
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) Xi[a*3+b] = F[a*3+b] + Ai[a*3]*F[b] + Ai[a*3+1]*F[3+b] + Ai[a*3+2]*F[6+b];
        inv3(Xp, Xpi); ao_mm3(Xi, Xpi, T1); ao_mm3(T1, Fe, Ft);
        ssvdf(Ft, Ue, se, Ve);
        bool ch = (prm[15] > 0.5f) ? false : pl_rm(mat, se, prm, ys, lj, al, dt);
        any = any || ch;
        pl_usv(Ue, se, Ve, Fe);
        for (int k=0;k<9;k++) Xp[k] = Xi[k];
    }
    return any;
}
"""
ELEM_RM_OLD = "    bool ch = (prm[15] > 0.5f) ? false : pl_rm(mat, se, prm, ys, lj, al, dt);\n    if (cnt && on && ch && mat != 7) atomicAdd(&cnt[0], 1);\n"
ELEM_RM_NEW = ("    float Ue_[9], Ve_[9];\n    bool ch = rms_sub(mat, prm, ys, lj, al, dt, F, A, ids, BG, oc, Ue_, se, Ve_);          // 절 CM28 (c): se = 끝 탄성 σ̂, Ue_ = 그 U\n"
               "    if (cnt && on && ch && mat != 7) atomicAdd(&cnt[0], 1);\n")
ELEM_PM_OLD = "    for (int a=0;a<3;a++) for (int b=0;b<3;b++) Pm[a*3+b] = U[a*3]*p[0]*V[b*3] + U[a*3+1]*p[1]*V[b*3+1] + U[a*3+2]*p[2]*V[b*3+2];\n"
ELEM_PM_NEW = ("    { float R_[9], tq[3];                                                     // 절 CM28 (c): P = τ X⁻ᵀ = Ue diag(τ̂) Ueᵀ U diag(1/σ) Vᵀ\n"
               "      for (int i=0;i<3;i++) { tq[i] = Pe[i]*se[i]; for (int j=0;j<3;j++) R_[i*3+j] = Ue_[i]*U[j] + Ue_[3+i]*U[3+j] + Ue_[6+i]*U[6+j]; }\n"
               "      for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float acc = 0.f; for (int i=0;i<3;i++) for (int j=0;j<3;j++) acc += Ue_[a*3+i]*tq[i]*R_[i*3+j]/ss[j]*V[b*3+j]; Pm[a*3+b] = acc; } }\n")
COMMIT_OLD = ("    float Uu[9], s[3], Vv[9]; ssvdf(X, Uu, s, Vv);\n    if (mat == 7 && prm[14] > 0.f) for (int i=0;i<3;i++) s[i] = fmaxf(s[i], prm[14]);\n"
              "    float ys = Sq[(size_t)q*4], lj = Sq[(size_t)q*4+1], al = Sq[(size_t)q*4+2];\n    bool ch = (prm[15] > 0.5f) ? false : pl_rm(mat, s, prm, ys, lj, al, dt);\n")
COMMIT_NEW = ("    float Uu[9], s[3], Vv[9];\n    float ys = Sq[(size_t)q*4], lj = Sq[(size_t)q*4+1], al = Sq[(size_t)q*4+2];\n"
              "    bool ch = rms_sub(mat, prm, ys, lj, al, dt, F, A, ids, BG, oc, Uu, s, Vv);             // 절 CM28 (c): 걸음 시작 U0 -> 끝 U 를 n 등분\n")


def _rep(s, o, n, cnt=1):
    assert s.count(o) == cnt, (s.count(o), o[:120])
    return s.replace(o, n)


def _rep_in(s, head, o, n):
    """head 로 시작하는 커널 안에서만 (다음 __global__ 앞까지) 한 번 치환."""
    i = s.index(head)
    j = s.find("__global__", i + len(head))
    j = len(s) if j < 0 else j
    return s[:i] + _rep(s[i:j], o, n) + s[j:]


def apply(r, wf):
    """r = fem_ao_rt._src 의 결과 (skipify 앞), wf = dict(seedb, ljq, ljq_xi, rmsub, rmsnh)."""
    if not wf:
        return r
    if wf.get("rmsnh"):
        i = r.index(PLRM_HEAD)
        r = r[:i] + RMS_HELP + r[i:]
        for o_, n_ in SNH_RM:
            r = _rep(r, o_, n_)
        r = _rep(r, SNH_HARD_OLD, SNH_HARD_NEW, 2)
    if wf.get("seedb"):
        r = _rep(r, SEED_OLD, SEED_NEW)
        if FPI_YS_OLD in r:
            r = _rep(r, FPI_YS_OLD, FPI_YS_NEW)
    if wf.get("ljq"):
        assert wf["ljq"] in ("p0", "max")
        r = _rep(r, P2G_OLD, P2G_P0 if wf["ljq"] == "p0" else P2G_MAX)
        r = _rep(r, GS_OLD, GS_P0 if wf["ljq"] == "p0" else GS_MAX)
        r = r.replace("extern \"C\" {", "extern \"C\" {\n#define AF_LJQ_XI %.9ef" % float(wf.get("ljq_xi", 3.0)), 1)
    if int(wf.get("rmsub", 0)) > 0:
        r = r.replace("extern \"C\" {", "extern \"C\" {\n__device__ float* AQ_U0;", 1)
        i = r.index(ELEM_HEAD)
        r = r[:i] + RMSUB_FN + r[i:]
        r = _rep_in(r, ELEM_HEAD, ELEM_RM_OLD, ELEM_RM_NEW)
        r = _rep_in(r, ELEM_HEAD, ELEM_PM_OLD, ELEM_PM_NEW)
        r = _rep_in(r, "__global__ void aq_commit(", COMMIT_OLD, COMMIT_NEW)
    return r
