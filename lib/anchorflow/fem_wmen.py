"""절 CM33 (조정자 결정 11-08, 정의 변경 + 진단): 수박 (2) 판 (절 CM32 fi + CM27 (d) 쪼개기 끔) 의 에너지 생성 -- 몫 분해 진단 (--enrec) 과 고침 후보.
옛 재질 실행기 (fem_ao_rt.AORTSolver) 에서 s.wf["en"] 이 있을 때만 쓰임 -- 없으면 실행·생성 소스 글자 그대로.

고침 후보 (소스 바꾸기 = 이 파일에서만, 끄면 소스 그대로):
  ro  덜 풀린 걸음 판정을 잔차만으로 (--uf_ro, --uf_res 위): uf_end 의 '보폭 최소 < 1' 조건을 뺌 -> 덜 풀림 = 허용치로 안 멈췄고 마지막으로 돈 뉴턴 반복 시작 ‖g‖ > F.
  nl  국소 보폭으로 덜 간 노드만 걸음 끝 갱신을 실제 이동 기반으로 (--ue_nl, --ws_bar·--ue be 위): 이 걸음 어느 뉴턴 반복에서든 노드 k > 0 (2^-k 보폭) 이었던 노드 (ecut) 는
      위치는 HHT 그대로 U1 = U + dt v + dt² ((½−β) a0 + β a1) (= U + (y − U)/(1 − αf)), 속도 v1 = (U1 − U)/dt, 가속도 기억 0. 나머지 노드는 HHT (덜 풀린 걸음은 원래대로 후퇴 오일러).
  mo  (d) 진단이 가리킨 것 -- 뉴턴 반복마다 증분 포텐셜 Φ(y) = ½ Σ m/τ² |y − ŷ|² + Σ V_q Ψ_SNH(RM(X_q(y))) + 장벽 (풀이와 같은 α 점·RM·장벽 식) 이 반복 시작보다
      커지면 (상대 1e-5 넘게) 이 반복의 갱신 (노드별 국소 보폭 포함) 을 반복 시작 쪽으로 반씩 되돌림 (최대 6 번, 그래도 크면 갱신 버림, 보폭 SC[14] 도 같이 줄임) (--en_mo).
  xf  덜 풀린 걸음에서만 뉴턴 반복을 늘림 (--xit_from j, --newton N): 반복 j 의 시작 잔차를 잰 뒤 (uf_g) 그때까지 판정이 '덜 풀림' 이 아니면 그 걸음 나머지 반복을 건너뜀
      (허용치 멈춤과 같은 CT[0] = 1·AF_SKIP). ro 와 함께면 잔차만, 아니면 보폭 최소 < 1 또는 잔차.
진단 (--enrec, 결과 불변 -- 그래프 안 사본 (Fq 커밋 앞) 과 절 CM23.4 --frec 버퍼를 프레임 뒤에 읽기만; 아래 EnCtx).
"""
import math

import numpy as np
import torch

from .fem_ao_rt import P

EN_SRC = r"""
// ---- 절 CM33 (lib/anchorflow/fem_wmen.py) ----
__global__ void en_cut(const int* IP, const int* kn, int* cut) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (kn[n] > cut[n]) cut[n] = kn[n];
}
__global__ void en_mo_set(int* MO, double* PH) { MO[0] = 1; PH[1] = 0.0; }
__global__ void en_ph01(double* PH) { PH[0] = PH[1]; }
// Φ 의 가우스 점 몫 (RM 뒤 SNH 탄성 + 장벽) -- aq_elem 과 같은 시험 X·RM (상태 지역 사본), rt_bar_q 와 같은 장벽 식
__global__ void __launch_bounds__(128) en_phi_q(const int* IP, const int* cb, const float* BG, const float* BW, const float* Fq, const float* Vq, const float* y, const float* U, float thinv,
                                                int mat, const float* prm, const float* Sq, float dt, float n0, float n1, float n2, float pn, float dh, float ks, const int* MO, double* PH) {
    if (!MO[0]) return;
    if (blockIdx.x*blockDim.x >= IP[12]) return;
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0.0;
    if (q < IP[12]) {
        int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
        float vq = Vq[q];
        if (vq > 0.f) {
            float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
            float A[9] = {0,0,0,0,0,0,0,0,0};
            for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; int id = ids[o]; for (int a=0;a<3;a++) { float ya = y[id*3+a]; A[a*3] += ya*bg[0]; A[a*3+1] += ya*bg[1]; A[a*3+2] += ya*bg[2]; } }
            float X[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) X[a*3+b] = F[a*3+b] + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b];
            float Uu[9], S[3], V[9]; ssvdf(X, Uu, S, V);
            float sev = (mat == 7) ? prm[14] : 0.f;
            float se[3]; for (int i=0;i<3;i++) se[i] = (sev > 0.f) ? fmaxf(S[i], sev) : S[i];
            float ys = Sq[(size_t)q*4], lj = Sq[(size_t)q*4+1], al = Sq[(size_t)q*4+2];
            if (!(prm[15] > 0.5f)) pl_rm(mat, se, prm, ys, lj, al, dt);
            double mu = prm[0], ls = (double)prm[1] + mu;
            double J = (double)se[0]*se[1]*se[2], ic = (double)se[0]*se[0] + (double)se[1]*se[1] + (double)se[2]*se[2];
            double ps = 0.5*mu*(ic - 3.0) - mu*(J - 1.0) + 0.5*ls*(J - 1.0)*(J - 1.0);
            e += (ps == ps) ? (double)vq*ps : 1e30;
        }
        float bm = AF_BX[q*4+3];
        if (bm > 0.f && AF_BACT[q]) {
            float xq[3] = {AF_BX[q*4]/bm, AF_BX[q*4+1]/bm, AF_BX[q*4+2]/bm};
            for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o]; for (int a=0;a<3;a++) xq[a] += w*(U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a])); }
            float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;
            if (d < dh) {
                double kap = (double)ks*(double)vq;
                if (d > 0.f) { double dd = (double)d - dh; e += -kap*dd*dd*log((double)d/dh); } else e += 1e30;
            }
        }
    }
    blocksum_add(e, &PH[1]);
}
__global__ void en_phi_n(const int* IP, const float* mn, float it2, const float* y, const float* yh, const int* MO, double* PH) {
    if (!MO[0]) return;
    if (blockIdx.x*blockDim.x >= IP[9]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0.0;
    if (n < IP[9]) { double m = 0.5*(double)mn[n]*it2; for (int a=0;a<3;a++) { double d = (double)y[n*3+a] - yh[n*3+a]; e += m*d*d; } }
    blocksum_add(e, &PH[1]);
}
__global__ void en_bt_chk(int* MO, const double* PH, double rtol) {
    if (!MO[0]) return;
    double a = PH[0], b = PH[1];
    if (b == b && b <= a + rtol*fabs(a) + 1e-9) MO[0] = 0; else MO[1]++;
}
__global__ void en_bt_mix(const int* IP, const int* MO, const float* ysv, const float* esv, float* y, float* eta, float c) {
    if (!MO[0]) return;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<3;a++) { y[n*3+a] = ysv[n*3+a] + c*(y[n*3+a] - ysv[n*3+a]); eta[n*3+a] = esv[n*3+a] + c*(eta[n*3+a] - esv[n*3+a]); }
}
__global__ void en_bt_sc(const int* MO, double* SC, double* PH) { if (!MO[0]) return; SC[14] *= 0.5; PH[1] = 0.0; }
__global__ void en_bt_fin(int* MO, double* SC, double* PH) { if (MO[0]) { SC[14] = 0.0; PH[1] = PH[0]; MO[2]++; MO[0] = 0; } }
__global__ void en_gate(int* CT, const float* ufs, const float* ufg, int k, int res, int ro) {
    if (CT[0]) return;
    bool cut = (!ro) && ufs[k] < 1.f, big = res && ufg[k] > 1.f;
    if (!(cut || big)) { CT[0] = 1; AF_SKIP = 1; }
}
"""
RO_OLD = "    bool cut = ufs[k] < 1.f, big = res && ufg[k] > 1.f;\n"
RO_NEW = "    bool cut = false && ufs[k] < 1.f, big = res && ufg[k] > 1.f;                  // 절 CM33 (a) --uf_ro: 잔차만\n"
NL_SIG_OLD = "float damp, const int* ufl, int um) {"
NL_SIG_NEW = "float damp, const int* ufl, int um, const int* ecut) {"
NL_N_OLD = "    if (n >= IP[9]) return;\n"
NL_N_NEW = "    if (n >= IP[9]) return;\n    bool cn_ = ecut[n] > 0;                                                  // 절 CM33 (b): 이 걸음 국소 보폭으로 자른 노드\n"
NL_U_OLD = "} else U[n*3+a] = u + dt*v + dt*dt*((0.5f - be)*an + be*a1);"
NL_U_NEW = "} else { U[n*3+a] = u + dt*v + dt*dt*((0.5f - be)*an + be*a1); if (cn_) v1 = (U[n*3+a] - u)/dt; }"
NL_A_OLD = "        if (uf_) a1 = 0.f;"
NL_A_NEW = "        if (uf_ || cn_) a1 = 0.f;"


def _rep(s, o, n):
    assert s.count(o) == 1, (o[:80], s.count(o))
    return s.replace(o, n, 1)


def apply(r, en):
    """r = fem_ao_rt._src 결과 (skipify 앞). en = dict(ro, nl, xf, diag)."""
    if not en:
        return r
    if en.get("ro"):
        r = _rep(r, RO_OLD, RO_NEW)
    add = EN_SRC
    if en.get("nl"):
        i = r.index("__global__ void ao_stepend_uf(")
        j = r.index("\n}\n", i) + 3
        k = r[i:j].replace("ao_stepend_uf(", "en_stepend(", 1)
        for o_, n_ in ((NL_SIG_OLD, NL_SIG_NEW), (NL_N_OLD, NL_N_NEW), (NL_U_OLD, NL_U_NEW), (NL_A_OLD, NL_A_NEW)):
            k = _rep(k, o_, n_)
        add = add + k
    i = r.rfind("}")
    return r[:i] + add + r[i:]


def get(s):
    return getattr(s, "wf", {}).get("en") or None


def alloc(s):
    """fem_ao_rt.AORTSolver.alloc 끝에서 (용량 바뀔 때마다)."""
    en = get(s)
    if not en:
        return
    s.ENcut = torch.zeros(s.capN, device=s.dev, dtype=torch.int32)
    s.ENmo = torch.zeros(4, device=s.dev, dtype=torch.int32); s.ENph = torch.zeros(2, device=s.dev, dtype=torch.float64)
    if en.get("mo"):
        s.ENysv = torch.zeros(s.capN, 3, device=s.dev, dtype=torch.float32); s.ENesv = torch.zeros(s.capN, 3, device=s.dev, dtype=torch.float32)
    if en.get("diag"):
        s.ENFq = torch.zeros_like(s.Fq)


def step0(s):
    s.zf(s.ENcut)
    if get(s).get("mo"):
        s.zf(s.ENmo)


def _phi(s, capN, capQ, thv, WB, nr):
    f = np.float32
    s.k("en_mo_set", 1, (P(s.ENmo), P(s.ENph)), b=1)
    s.k("en_phi_q", capQ, (P(s.IP), P(s.cb), P(s.BG), P(s.BW), P(s.Fq), P(s.Vq), P(s.y), P(s.U), thv, np.int32(s.mat), P(s.prm), P(s.Sq), f(s.dt), *[f(q) for q in nr],
                           f(WB["PN"]), f(WB["DH"]), f(WB["KS"]), P(s.ENmo), P(s.ENph)), b=128)
    s.k("en_phi_n", capN, (P(s.IP), P(s.mn), f(1.0 / (s.tau * s.tau)), P(s.y), P(s.yh), P(s.ENmo), P(s.ENph)), b=256)


def mo_pre(s, it, capN, capQ, thv, WB, nr):
    """(d) 이 반복 갱신 앞: y·η 사본, 반복 시작 Φ (첫 반복은 새로 재고, 뒤는 앞 반복 끝 값)."""
    s.ENysv.copy_(s.y); s.ENesv.copy_(s.eta)
    if it == 0:
        _phi(s, capN, capQ, thv, WB, nr)
    s.k("en_ph01", 1, (P(s.ENph),), b=1)


def mo_post(s, capN, capQ, thv, WB, nr, R=6):
    """(d) 갱신 뒤 Φ 를 재고, 시작보다 크면 반씩 되돌림 (최대 R 번), 그래도 크면 갱신 버림."""
    _phi(s, capN, capQ, thv, WB, nr)
    for _ in range(R):
        s.k("en_bt_chk", 1, (P(s.ENmo), P(s.ENph), np.float64(1e-5)), b=1)
        s.k("en_bt_mix", capN, (P(s.IP), P(s.ENmo), P(s.ENysv), P(s.ENesv), P(s.y), P(s.eta), np.float32(0.5)), b=256)
        s.k("en_bt_sc", 1, (P(s.ENmo), P(s.SC), P(s.ENph)), b=1)
        s.k("en_phi_q", capQ, (P(s.IP), P(s.cb), P(s.BG), P(s.BW), P(s.Fq), P(s.Vq), P(s.y), P(s.U), thv, np.int32(s.mat), P(s.prm), P(s.Sq), np.float32(s.dt), *[np.float32(q) for q in nr],
                               np.float32(WB["PN"]), np.float32(WB["DH"]), np.float32(WB["KS"]), P(s.ENmo), P(s.ENph)), b=128)
        s.k("en_phi_n", capN, (P(s.IP), P(s.mn), np.float32(1.0 / (s.tau * s.tau)), P(s.y), P(s.yh), P(s.ENmo), P(s.ENph)), b=256)
    s.k("en_bt_chk", 1, (P(s.ENmo), P(s.ENph), np.float64(1e-5)), b=1)
    s.k("en_bt_mix", capN, (P(s.IP), P(s.ENmo), P(s.ENysv), P(s.ENesv), P(s.y), P(s.eta), np.float32(0.0)), b=256)
    s.k("en_bt_fin", 1, (P(s.ENmo), P(s.SC), P(s.ENph)), b=1)


def gate(s, kstep, it):
    en = get(s)
    if int(en.get("xf") or 0) > 0 and it == int(en["xf"]):
        s.k("en_gate", 1, (P(s.CT), P(s.ufs), P(s.ufg), np.int32(kstep), np.int32(1 if s.ufr else 0), np.int32(1 if en.get("ro") else 0)), b=1)


def after_newton(s, capN):
    s.k("en_cut", capN, (P(s.IP), P(s.wskn), P(s.ENcut)), b=256)


def pre_commit(s):
    if get(s).get("diag"):
        s.ENFq.copy_(s.Fq)


# ======================= 진단 (--enrec): 프레임 (= K 1 한 걸음) 에너지 몫 분해 =======================
# 입자 에너지 E_p = 운동 + 중력 위치 + SNH 탄성 (입자 F; --erec 와 같은 식). 프레임 사슬:
#   입자 (시작) -[P2G]-> 격자 시작 (노드 운동 ½ Σ m v0², 노드 중력 위치 −Σ m g·X, 가우스 점 탄성 Σ V_q Ψ(F_q))
#   -[걸음]-> 격자 끝 (v1, U1, 가우스 점 시험 X = F_q + ∇U1 F_q) -[커밋 RM]-> X = RM(X) -[G2P·F 보간]-> 입자 (끝).
# 걸음 몫 (Δx = U1 − U0 = U1, 노드 합; 정확한 항등식):
#   ΔKE + ΔPE + ΔE_el + ΔB = W_res + W_con + C_el + C_bar + R_upd
#   g = m/τ² (y − ŷ) + f_el(y) + G(y) n  (풀이 기울기 물리 좌표, 마지막 뉴턴 뒤 다시 잰 것; 수렴이면 자유 노드에서 0)
#   W_res = Σ_자유 Δx·g (잔차 몫: 풀이가 못 맞춘 힘이 한 일), W_con = Σ_고착·고정 Δx·g (띠 고착·빈 노드 고정 구속의 일)
#   C_el = ΔE_el − Σ Δx·f_el(y) (탄성 몫: 에너지 변화와 풀이가 쓴 탄성 힘의 일 차 -- α 점·비볼록·RM 응력),  C_bar = ΔB − Σ Δx·G n
#   R_upd = ΔKE − Σ Δx·(m g + m/τ² (y − ŷ)) (갱신식 몫). 후퇴 오일러 갱신 (v1 = Δx/dt) 이면 R_upd = Σ Δx·g_upd − ½ Σ m |v1 − v0|²,
#     g_upd = m (v1 − v0)/dt − m g − m/τ² (y − ŷ) (HHT 로 푼 y 를 후퇴 오일러로 받은 불일치), ½ Σ m |Δv|² = 수치 감쇠 (≥ 0).
# 입자 에너지 변화 = P2G 몫 + W_res + W_con + C_el + R_upd + W_bar (= −Σ Δx·G n, 장벽 힘이 몸에 한 일) + 커밋 몫 + G2P 몫.
def _snh_sum(F, V, mu, lam):
    F = F.reshape(-1, 3, 3).double()
    J = torch.linalg.det(F)
    Ic = (F * F).sum((1, 2))
    psi = 0.5 * mu * (Ic - 3.0) - mu * (J - 1.0) + 0.5 * (lam + mu) * (J - 1.0) ** 2
    ok = torch.isfinite(psi) & (V > 0)
    return float((V.double()[ok] * psi[ok]).sum()), int((ok & (J <= 0)).sum())


class EnCtx:
    def __init__(s, S, FL, WB, mu, lam, g, cell):
        s.S, s.FL, s.WB = S, FL, WB
        s.mu, s.lam, s.cell = float(mu), float(lam), float(cell)
        dv = S.dev
        s.n = torch.tensor(FL["NR"] if FL else [0.0, 0.0, 1.0], device=dv, dtype=torch.float64)
        s.pn = float(FL["PN"]) if FL else 0.0
        s.g = torch.tensor([float(q) for q in g], device=dv, dtype=torch.float64)
        s.p0 = None

    def particles(s):
        S = s.S
        m = S.m.double(); x = S.x.double(); v = S.v.double()
        ok = torch.isfinite(x).all(-1) & torch.isfinite(v).all(-1)
        Ee, _ = _snh_sum(S.F[ok], S.vol[ok], s.mu, s.lam)
        return dict(KE=float(0.5 * (m[ok] * (v[ok] ** 2).sum(1)).sum()), PE=float(-(m[ok] * (x[ok] @ s.g)).sum()), El=Ee)

    def pre(s):
        """프레임 시작 (step 앞) 입자 에너지."""
        s.p0 = s.particles()

    def _q(s):
        S = s.S
        IP = S.IP.cpu().tolist()
        nq = IP[12]
        q = torch.arange(nq, device=S.dev)
        cb = S.cb[q >> 3].long()
        o = torch.arange(27, device=S.dev)
        oi, oj, ok = o // 9, (o // 3) % 3, o % 3
        ids = ((cb[:, 0:1] + oi) * IP[7] + (cb[:, 1:2] + oj)) * IP[8] + (cb[:, 2:3] + ok)
        return nq, q, ids, IP

    def eq(s, Fq, disp, nq, q, ids, ch=1 << 18):
        """Σ_q V_q Ψ_SNH(F_q + A(disp) F_q), A = Σ_o disp_o ⊗ ∇w_o (aq_elem 과 같은 식). 뒤집힌 점 수도."""
        S = s.S
        BG = S.BG.view(8, 27, 3).double()
        E, ninv = 0.0, 0
        for a in range(0, nq, ch):
            b = min(nq, a + ch)
            F = Fq[a:b].reshape(-1, 3, 3).double()
            if disp is None:
                X = F
            else:
                bg = BG[(q[a:b] & 7)]                                                   # (n, 27, 3)
                d = disp.double()[ids[a:b]]                                             # (n, 27, 3)
                A = torch.einsum("noa,nob->nab", d, bg)
                X = F + A @ F
            e, ni = _snh_sum(X, S.Vq[a:b], s.mu, s.lam)
            E += e; ninv += ni
        return E, ninv

    def bar(s, disp, nq, q, ids):
        """장벽 에너지 Σ −κ (d − d̂)² ln(d/d̂) (d < d̂, κ = KS V_q), 표본점 = 프레임 시작 칸 입자 질량 중심 + Σ w disp; 활성 = 걸음 시작 판정 (bact)."""
        S, WB = s.S, s.WB
        if WB is None or nq == 0:
            return 0.0, 0, 0
        bx = S.bx[:4 * nq].view(nq, 4).double(); bm = bx[:, 3]
        act = (S.bact[:nq] > 0) & (bm > 0)
        w = S.BW.view(8, 27)[(q & 7)].double()
        xq = torch.where(bm[:, None] > 0, bx[:, :3] / bm.clamp_min(1e-30)[:, None], torch.zeros_like(bx[:, :3]))
        if disp is not None:
            xq = xq + (w[:, :, None] * disp.double()[ids]).sum(1)
        d = xq @ s.n - s.pn
        dh = float(WB["DH"])
        hit = act & (d < dh)
        kap = float(WB["KS"]) * S.Vq[:nq].double()
        dd = d - dh
        e = torch.where(hit & (d > 0), -kap * dd * dd * torch.log(d.clamp_min(1e-30) / dh), torch.zeros_like(d))
        return float(e.sum()), int(hit.sum()), int((hit & (d <= 0)).sum())

    def frame(s, fr):
        S = s.S
        nq, q, ids, IP = s._q()
        nn = IP[9]
        nt = S.nt
        tau2 = float(S.tau) ** 2; dt = float(S.dt)
        thinv = 1.0 / (1.0 - float(S.ga_c["af"]))
        m = S.FRmn[:nn].double()
        has = m > 0
        v0 = S.FRv0[0, :nn].double(); v1 = S.FRv1[0, :nn].double(); U1 = S.FRU1[0, :nn].double()
        y = S.FRy[0, nt, :nn].double(); yh = S.FRyh[0, :nn].double()
        fel = S.FRfel[0, nt, :nn].double(); bgn = S.FRbg[0, nt, :nn].double()[:, None] * s.n[None, :]
        gph = (m / tau2)[:, None] * (y - yh) + fel + bgn
        pm = S.pm[:nn]; sk = S.sk[:nn].double()
        free = has & ((pm == 0) | (pm == 1)) & (sk >= 1.0 - 1e-6)                     # T = I (장벽 바닥 판은 모든 노드 pm 1: 접선 고착 sk < 1 만 구속)
        con = has & ~free
        cut = (S.ENcut[:nn] > 0) & has
        L = torch.stack([torch.arange(nn, device=S.dev) // (IP[7] * IP[8]) + IP[0], (torch.arange(nn, device=S.dev) // IP[8]) % IP[7] + IP[1], torch.arange(nn, device=S.dev) % IP[8] + IP[2]], 1).double()
        X = L @ S.A.double().T
        near = has & ((X @ s.n - s.pn) < 3.0 * s.cell)
        dx = U1
        mg = m[:, None] * s.g[None, :]
        KE0 = float(0.5 * (m * (v0 ** 2).sum(1)).sum()); KE1 = float(0.5 * (m * (v1 ** 2).sum(1)).sum())
        PE0 = float(-(m * (X @ s.g)).sum()); dPE = float(-(m * (dx @ s.g)).sum())
        dot = lambda A, B, R=None: float(((A * B).sum(1)[R] if R is not None else (A * B).sum(1)[has]).sum())
        Wf_el = dot(dx, fel); Wf_bar = dot(dx, bgn)
        Wres = dot(dx, gph, free); Wcon = dot(dx, gph, con)
        inert = (m / tau2)[:, None] * (y - yh)
        Rupd = (KE1 - KE0) - dot(dx, mg + inert)
        gupd = m[:, None] * (v1 - v0) / dt - mg - inert
        Wupd = dot(dx, gupd); Dnum = float(0.5 * (m * ((v1 - v0) ** 2).sum(1)).sum())
        reg = lambda T, R: float(((T).sum(1)[R]).sum())
        # 노드 몫 영역 나눔: 자른 노드 (이 걸음 어느 반복이든 국소 보폭 2^-k, k > 0) · 바닥 근처 (노드 높이 < 3 칸)
        pr = dx * gph
        pu = dx * gupd
        prr = (0.5 * m)[:, None] * (v1 ** 2 - v0 ** 2) - dx * (mg + inert)
        split = {}
        for nm_, R in (("cut", cut), ("uncut", has & ~cut), ("near", near), ("far", has & ~near)):
            split[nm_] = dict(n=int(R.sum()), m=float(m[R].sum()), Wres=reg(pr, R & free), Wcon=reg(pr, R & con), Rupd=reg(prr, R), Wupd=reg(pu, R),
                              dKE=float((0.5 * m * ((v1 ** 2).sum(1) - (v0 ** 2).sum(1)))[R].sum()))
        if getattr(S, "ulm", None) is not None:                                        # 절 CM35 진단: 후퇴 오일러 노드 집합 S / S 밖 (자른·안 자른) 나눔, HHT 속도 뒤집기 척도
            from . import fem_common2 as C2
            split["S2"] = C2.split_s(S, m, has, v0, v1, y, U1, dt, free, cut, pr, pu, prr, gph)
        Fq0 = S.ENFq[:nq]; Fqc = S.Fq[:nq]
        E0, i0 = s.eq(Fq0, None, nq, q, ids)
        E1, i1 = s.eq(Fq0, U1, nq, q, ids)
        Ey, iy = s.eq(Fq0, y, nq, q, ids)
        E1c, i1c = s.eq(Fqc, U1, nq, q, ids)
        B0, h0, _ = s.bar(None, nq, q, ids)
        B1, h1, ng1 = s.bar(U1, nq, q, ids)
        Cel = (E1 - E0) - Wf_el
        Cbar = (B1 - B0) - Wf_bar
        # 뉴턴 반복별 증분 포텐셜 Φ(y) = ½ Σ m/τ² |y − ŷ|² + Σ V Ψ(X(y)) + B(U + (y − U)/(1−αf)) (−Σ m g·ŷ 는 상수라 뺌), y_0 = 시작점, y_nt = 끝
        phi = []
        for it in range(nt + 1):
            yi = S.FRy[0, it, :nn].double()
            ine = float(0.5 * ((m / tau2) * ((yi - yh) ** 2).sum(1)).sum())
            ei, _ = s.eq(Fq0, yi, nq, q, ids)
            bi, _, _ = s.bar(thinv * yi, nq, q, ids)
            phi.append(dict(inert=ine, el=ei, bar=bi, tot=ine + ei + bi))
        eh, _ = s.eq(Fq0, yh, nq, q, ids)
        bh, _, _ = s.bar(thinv * yh, nq, q, ids)
        p1 = s.particles()
        p0 = s.p0 or p1
        PE1n = PE0 + dPE
        out = dict(
            p0=p0, p1=p1, dEp=(p1["KE"] + p1["PE"] + p1["El"]) - (p0["KE"] + p0["PE"] + p0["El"]),
            grid=dict(KE0=KE0, KE1=KE1, PE0=PE0, dPE=dPE, E0=E0, E1=E1, Ey=Ey, E1c=E1c, B0=B0, B1=B1, inv=[i0, i1, iy, i1c], barhit=[h0, h1, ng1]),
            p2g=dict(KE=KE0 - p0["KE"], PE=PE0 - p0["PE"], El=E0 - p0["El"]),
            step=dict(Wres=Wres, Wcon=Wcon, Cel=Cel, Cbar=Cbar, Rupd=Rupd, Wupd=Wupd, Dnum=Dnum, Wbar=-Wf_bar, Wf_el=Wf_el,
                      dKE=KE1 - KE0, dEl=E1 - E0, dB=B1 - B0, chk=(KE1 - KE0) + dPE + (E1 - E0) + (B1 - B0) - (Wres + Wcon + Cel + Cbar + Rupd)),
            commit=E1c - E1,
            g2p=dict(KE=p1["KE"] - KE1, PE=p1["PE"] - PE1n, El=p1["El"] - E1c),
            split=split,
            phi=phi, phi_hat=dict(el=eh, bar=bh, tot=eh + bh),
            uf=int(S.ufl[0]) if getattr(S, "ufl", None) is not None else None,
            ncut=int(cut.sum()), mcut=float(m[cut].sum()) / max(float(m[has].sum()), 1e-30),
        )
        st = out["step"]
        out["sum"] = out["p2g"]["KE"] + out["p2g"]["PE"] + out["p2g"]["El"] + st["Wres"] + st["Wcon"] + st["Cel"] + st["Rupd"] + st["Wbar"] + out["commit"] + out["g2p"]["KE"] + out["g2p"]["PE"] + out["g2p"]["El"]
        out["sum_chk"] = out["dEp"] - out["sum"]
        return out
