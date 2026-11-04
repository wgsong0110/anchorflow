"""절 CM2 재질 판 (wolf 모래·bread plasticine) 을 이 가지의 실시간 판 (lib/anchorflow/fem_an_rt.py RTSolver, 결정적 판 fem_an_det.DetMixin) 위로 옮긴 판.
lego·ficus 와 같은 실행기 경로·같은 인자 글자 (장벽 바닥 --fl_bar --fl_bst, 점유 가중 --gq_oct, MG 층·--mgbar 2, --mv2, --lsfast, --fk, 허용치) 를 쓴다.
재질 몫만 바꿈 (커널은 lib/anchorflow/fem_ao_rt.py AO_SRC 를 글자 그대로 이 모듈에 함께 컴파일):
- 요소: 모래 ao_elem (PG Drucker-Prager RM 응력 + 시험 X 의 Hencky 탄성 접선, 81 성분 헤시안 -- rt_mv·rt_mv2·MG 가 그대로 씀),
  plasticine aq_elem (정답 pl_rm, 차분 접선) 의 주값 분해 30 성분을 mt_h81 이 같은 81 성분 꼴로 펼침 (같은 선형 사상: dP = U C(Uᵀ dX V) Vᵀ / V_q, V = F_q⁻¹ M).
- 걸음 사이 소성 커밋 (ao_commit·aq_commit), G2P 입자 F 걸음별 RM (ao_g2p·aq_g2p), 입자 가속도 ao_g2p_a.
- 순차 방출 구속 (ao_vzero·aq_vset·ao_azero·ao_pbc_acc·ao_pbc_k, 노드 목표 속도 aq_p2g_v·aq_vbar), 거의 빈 노드 관성 예측 고정, 질량 0 노드 속도·가속도 이웃 평균 (aq_vfill).
- 선탐색: 재질 판은 퍼텐셜이 없어 (비결합 흐름·소성) 에너지 선탐색을 하지 않는다 (절 AO.0 정의 그대로) -- 장벽 바닥이면 보폭 = 활성 표본점 d > 0 인 최대 2^-k (rt_bar_kfe).
  --ls_n·--descent·--lsfast 는 같은 글자로 받지만 선탐색 경로가 없어 이 판에서는 쓰이지 않는다 (--fk 의 MG 거듭제곱법 몫은 쓰임).
- 첫 프레임 입자 가속도 = g (정지 상태; 절 AO 정의 그대로 -- 탄성 판의 a0 = g − ∇Ψ/m 계산은 하지 않음).
"""
import numpy as np
import torch

from . import fem_an_rt as ART
from . import fem_an_det as ADET
from . import fem_ao_rt as AORT

P = ART.P

MT_SRC = r'''
// ---- 절 CM2 재질 판 연결 커널 ----
// rt_floor_n (이 가지) + 재질 판 몫 (절 AO ao_floor_n 과 같은 식): 구속 인자 k_구속 < 1 이면 k *= k_구속, 세 성분 모두 (p = 0), 목표 속도 vbt;
// 질량 < 문턱 노드는 관성 예측 Ũ 에 고정 (pm 2).
__global__ void mt_floor_n(const int* IP, const float* an, const float* mqn, float kex, int floor_, int vout, float al, float n0, float n1, float n2,
                           const unsigned char* fixd, const float* vbc, const float* U, float tau, const float* Ut, const float* z0, float* y, float* sk, int* pm, float* yh, float* eta,
                           const float* kb, const float* mn, float mthr, const float* vbt) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float k = 1.f; int p = 0;
    if (floor_) {
        float r = (mqn[n] > 0.f) ? fminf(an[n]/mqn[n], 1.f) : 0.f;
        k = expf(kex*log1pf(-r));
#ifdef AF_FLJ
        k = 1.f - r;
#endif
        float vv = al*((y[n*3]-z0[n*3])*n0 + (y[n*3+1]-z0[n*3+1])*n1 + (y[n*3+2]-z0[n*3+2])*n2);
        p = (vout && vv > 0.f) ? 1 : 0;
#ifdef AF_BAR
        p = 1;
#endif
    }
    float vt[3] = {0.f, 0.f, 0.f};
    if (kb) { float b = kb[n]; if (b < 1.f) { k *= b; p = 0; if (vbt) for (int a=0;a<3;a++) vt[a] = vbt[n*3+a]/al; } }
    if (fixd && fixd[n]) p = 2;
    if (p == 2) { for (int a=0;a<3;a++) { float yy = U[n*3+a] + tau*vbc[n*3+a]; yh[n*3+a] = yy; eta[n*3+a] = 0.f; y[n*3+a] = yy; } sk[n] = 0.f; pm[n] = 2; return; }
    if (mn[n] < mthr) { for (int a=0;a<3;a++) { float yy = Ut[n*3+a]; yh[n*3+a] = yy; eta[n*3+a] = 0.f; y[n*3+a] = yy; } sk[n] = 0.f; pm[n] = 2; return; }
    float s = sqrtf(k);
    float d[3], Pd[3];
    for (int a=0;a<3;a++) d[a] = z0[n*3+a] + vt[a] - Ut[n*3+a];
    if (p == 0) { for (int a=0;a<3;a++) Pd[a] = d[a]; }
    else { float dn = d[0]*n0 + d[1]*n1 + d[2]*n2; Pd[0] = d[0]-dn*n0; Pd[1] = d[1]-dn*n1; Pd[2] = d[2]-dn*n2; }
    float h[3]; for (int a=0;a<3;a++) h[a] = Ut[n*3+a] + (1.f-k)*Pd[a];
    float e1[3], e2[3];
    for (int a=0;a<3;a++) { e1[a] = y[n*3+a] - Ut[n*3+a]; e2[a] = y[n*3+a] - h[a]; }
    float et[3];
    if (p == 0) { for (int a=0;a<3;a++) et[a] = s*e2[a]; }
    else { float c1 = e1[0]*n0 + e1[1]*n1 + e1[2]*n2; float c2 = e2[0]*n0 + e2[1]*n1 + e2[2]*n2;
           for (int a=0;a<3;a++) { float nn_ = (a==0) ? n0 : ((a==1) ? n1 : n2); et[a] = c1*nn_ + s*(e2[a] - c2*nn_); } }
    float ty[3]; Tapply(s, p, n0, n1, n2, et, ty);
    for (int a=0;a<3;a++) { yh[n*3+a] = h[a]; eta[n*3+a] = et[a]; y[n*3+a] = h[a] + ty[a]; }
    sk[n] = s; pm[n] = p;
}
// 구속 노드 (k_구속 < 1) 걸음 끝 가속도 0 (절 AO 처치 (b) 의 구속 몫; 바닥 몫은 rt_ga_end)
__global__ void mt_akb(const int* IP, const float* kb, float* ak) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (kb[n] < 1.f) { ak[n*3] = 0.f; ak[n*3+1] = 0.f; ak[n*3+2] = 0.f; }
}
// 주값 분해 30 성분 (aq_elem: U 9, M = F V 9, C 12 -- V_q 곱함) -> 81 성분 dP/dX (rt_mv 꼴: dP = V_q H dX, dX = A F)
__global__ void mt_h81(const int* IP, const float* Fq, const float* Vq, const float* H30, float* H, int capQ) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float U[9], M[9], C[12], F[9], Fi[9], V[9];
    for (int k=0;k<9;k++) { U[k] = H30[(size_t)k*capQ + q]; M[k] = H30[(size_t)(9+k)*capQ + q]; F[k] = Fq[(size_t)q*9+k]; }
    for (int k=0;k<12;k++) C[k] = H30[(size_t)(18+k)*capQ + q];
    inv3(F, Fi);
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) V[a*3+b] = Fi[a*3]*M[b] + Fi[a*3+1]*M[3+b] + Fi[a*3+2]*M[6+b];
    float vq = Vq[q]; float iv = (vq > 0.f) ? 1.f/vq : 0.f;
    for (int c=0;c<9;c++) {
        int i0 = c/3, j0 = c%3;
        float Xh[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) Xh[a*3+b] = U[i0*3+a]*V[j0*3+b];
        float Ph[9];
        Ph[0] = C[0]*Xh[0] + C[3]*Xh[4] + C[4]*Xh[8];
        Ph[4] = C[3]*Xh[0] + C[1]*Xh[4] + C[5]*Xh[8];
        Ph[8] = C[4]*Xh[0] + C[5]*Xh[4] + C[2]*Xh[8];
        Ph[1] = C[6]*Xh[1] + C[9]*Xh[3];  Ph[3] = C[9]*Xh[1] + C[6]*Xh[3];
        Ph[2] = C[7]*Xh[2] + C[10]*Xh[6]; Ph[6] = C[10]*Xh[2] + C[7]*Xh[6];
        Ph[5] = C[8]*Xh[5] + C[11]*Xh[7]; Ph[7] = C[11]*Xh[5] + C[8]*Xh[7];
        float W[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) W[a*3+b] = U[a*3]*Ph[b] + U[a*3+1]*Ph[3+b] + U[a*3+2]*Ph[6+b];
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) H[(size_t)((a*3+b)*9 + c)*capQ + q] = iv*(W[a*3]*V[b*3] + W[a*3+1]*V[b*3+1] + W[a*3+2]*V[b*3+2]);
    }
}
'''


def mat_module_src(mg2):
    src = ADET.det_module_src(mg2).rstrip()
    i = src.rfind("}")
    body = src[:i]
    decl = "\n__device__ int* AO_ACT;\n__device__ float* AQ_DBG;\n__device__ int AQ_DBGQ;"
    body = body.replace('extern "C" {', 'extern "C" {' + decl, 1)
    return "#include <cuda_fp16.h>\n" + body + AORT._pl_funcs() + ADET.det_ao(AORT.AO_SRC) + MT_SRC + "\n}\n"


class MatRT(ADET.DetMixin):
    """재질 판. 추가 인자: mat (2 모래, 5 plasticine, ...), cdp (모래 DP c), prm (재질 매개, fem_pl_cuda.params), visc (등가 점성), trel (입자 구속 끝 시각) / pvb (구속 속도),
    pbc_ns·dt_gt (구속 표본), mthr (거의 빈 노드 질량 문턱), ys0·lj0 (재질 입자 상태 처음 값), vfill, ar (구속 노드 가속도 0), det."""

    def __init__(s, *args, mat=2, cdp=1.0, prm=None, visc=0.0, trel=None, pvb=None, pbc_ns=4, dt_gt=2e-5, mthr=0.0, ys0=0.0, lj0=0.0, vfill=True, ar=True, det=False, **kw):
        s.det = bool(det)
        s.ao_ready = False
        s._mat, s.cdp, s.prm, s.visc = int(mat), float(cdp), prm, float(visc)
        s.aq = s._mat != 2
        s.has_pbc = trel is not None
        s.pbc_ns, s.dt_gt, s.mthr = int(pbc_ns), float(dt_gt), float(mthr)
        s.vfill, s.ar = bool(vfill), bool(ar)
        s._pvb_on = pvb is not None
        super().__init__(*args, **kw)
        assert s.ga, "재질 판은 --ga (HHT) 만"
        assert not s.bc, "재질 판은 고정 구속 상자 없는 씬만"
        s.matk = s._mat
        s.ga_first = False                                                             # 첫 프레임 입자 가속도 = g (절 AO)
        s.pa.copy_(torch.tensor(s.g, device=s.dev, dtype=torch.float32).expand(s.N, 3))
        s.trel = trel[s.perm].float().contiguous() if s.has_pbc else None
        s.pvb = None if pvb is None else pvb[s.perm].float().contiguous()
        if s.pvb is not None:
            s.pvc = torch.zeros(s.N, 3, device=s.dev, dtype=torch.float32); s.pmk = torch.zeros(s.N, 3, device=s.dev, dtype=torch.float32)
        if s.aq:
            s.ysp = torch.full((s.N,), float(ys0), device=s.dev, dtype=torch.float32)
            s.ljp = torch.full((s.N,), float(lj0), device=s.dev, dtype=torch.float32)
            s.alp = torch.ones(s.N, device=s.dev, dtype=torch.float32)
        s.extra_p = (("ysp", "ljp", "alp") if s.aq else ()) + (("pvb", "pvc", "pmk") if s.pvb is not None else ())
        s.tbuf = torch.zeros(1, device=s.dev, dtype=torch.float32)
        src = mat_module_src(getattr(s, "mg2", False))
        s.mod = ART.cp.RawModule(code=ART.build_src(src, s.adapt, s.goct, s.floor, fk=getattr(s, "fk", False), hspec=getattr(s, "hspec", False), ncomp=getattr(s, "ncomp", False)), options=("--std=c++14",))
        s._src0 = src
        s.fn = {}
        s.ao_ready = True
        s.det_bind()
        s.oct_bind()
        s.mat_bind()
        s.graph = None
        s.graphs = {}

    def alloc(s, need):
        super().alloc(need)
        s.det_alloc()
        Z = lambda *sh: torch.zeros(*sh, device=s.dev, dtype=torch.float32)
        I = lambda n_: torch.zeros(n_, device=s.dev, dtype=torch.int32)
        capQ = s.capQc * 8
        s.Ust = Z(s.K, s.capN, 3)
        s.H30 = Z(capQ * 30) if s.aq else None
        s.Hs = s.H30 if (s.aq and s.hspec) else None                                   # 절 CM3 --hspec: plasticine 은 aq_elem 의 30 성분을 곱이 바로 (mt_h81 펼침 없음); 모래는 ao_elem_s 가 H 앞 30 성분에
        s.cnt = I(4); s.qb = I(4); s.act = I(s.capN)
        s.Sa = Z(s.capQc * 32); s.Sc = Z(s.capQc * 4); s.Sq = Z(s.capQc * 32)
        s.gaM = Z(s.capN)
        if s.has_pbc:
            s.pacc = Z(s.capN * s.K * s.pbc_ns); s.kb = Z(s.capN)
        if s._pvb_on:
            s.vm = Z(s.capN, 3); s.cm = Z(s.capN, 3)
        if getattr(s, "ao_ready", False):
            s.det_bind()
            s.mat_bind()

    def mat_bind(s):
        for nm, t in (("AO_ACT", s.act), ("AQ_DBG", None)):
            g = s.mod.get_global(nm)
            ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=g)[0] = int(t.data_ptr()) if t is not None else 0

    # 결정적 판 입자 정렬·백업에 재질 입자 배열도
    def det_sort(s):
        o = torch.sort(s.pcd, stable=True).indices
        for nm in ("x", "v", "F", "m", "vol", "fx", "Np", "pcd", "perm", "pa") + (("trel",) if s.trel is not None else ()) + tuple(s.extra_p):
            t = getattr(s, nm)
            t.copy_(t[o])

    def _bkn(s):
        return ("x", "v", "F", "m", "vol", "perm", "pa") + (("trel",) if getattr(s, "trel", None) is not None else ()) + tuple(getattr(s, "extra_p", ()))

    def _bk(s):
        return tuple(getattr(s, nm).clone() for nm in s._bkn())

    def _rs(s, bk):
        for nm, t in zip(s._bkn(), bk):
            getattr(s, nm).copy_(t)

    def set_bc(s, t):
        super().set_bc(t)
        s.tbuf.fill_(float(t))

    def k(s, name, n, args, b=128):
        super().k(name, n, args, b)
        if getattr(s, "det", False) and name in ("aq_elem", "ao_elem_s"):                            # 재질 커널 칸 버퍼 -> 노드 (고정 순서; 절 AQ 결정적 판과 같음)
            ART.RTSolver.k(s, "ao_gath_d3", s.capN, (P(s.IP), P(s.cidx), args[10]), 256)
            ART.RTSolver.k(s, "ao_gath_f", s.capN, (P(s.IP), P(s.cidx), np.int32(9), P(s.d_cb), args[13]), 256)

    # ---- RTSolver.record 의 재질 몫 (s.matk) ----
    def mat_pre_p2g(s):
        N, i32 = s.N, np.int32
        s.zf(s.cnt)
        if s.aq:
            s.zf(s.qb)
        if s.has_pbc:
            if s.pvb is not None:
                s.k("aq_vset", N, (i32(N), P(s.trel), P(s.tbuf), P(s.pvb), P(s.v), P(s.pvc), P(s.pmk)))
            else:
                s.k("ao_vzero", N, (i32(N), P(s.IP), P(s.trel), P(s.tbuf), P(s.v)))
            s.k("ao_azero", N, (i32(N), P(s.trel), P(s.tbuf), P(s.pa)))

    def mat_post_vn(s):
        N, IP, i32, f = s.N, s.IP, np.int32, np.float32
        capN, capQ = s.capN, s.capQc * 8
        if s.aq:                                                                       # 가우스 점 소성 상태 (입자 부피 가중 평균)
            s.zf(s.Sa); s.zf(s.Sc)
            if s.det:
                s.k("aq_p2g_sd", s.capQc, (P(IP), P(s.d_st), P(s.fx), P(s.vol), P(s.ysp), P(s.ljp), P(s.alp), P(s.Sa), P(s.Sc)), b=128)
            else:
                s.k("aq_p2g_s", N, (i32(N), P(IP), P(s.fx), P(s.pcd), P(s.cidx), P(s.vol), P(s.ysp), P(s.ljp), P(s.alp), P(s.Sa), P(s.Sc)))
            s.k("aq_gauss_s", capQ, (P(IP), P(s.Sa), P(s.Sc), P(s.Sq)))
        if s.pvb is not None:                                                          # 노드 구속 목표 속도
            s.zf(s.vm, 9, 3); s.zf(s.cm, 9, 3)
            if s.det:
                s.zf(s.gaM, 9, 1)
                s.k("ao_p2g_n", capN, (P(IP), P(s.cidx), P(s.d_st), P(s.fx), P(s.m), P(s.pvc), P(s.gaM), P(s.vm), P(None), P(None), i32(1), i32(1), f(0.0), P(None)), b=128)
                s.zf(s.gaM, 9, 1)
                s.k("ao_p2g_n", capN, (P(IP), P(s.cidx), P(s.d_st), P(s.fx), P(s.m), P(s.pmk), P(s.gaM), P(s.cm), P(None), P(None), i32(1), i32(1), f(0.0), P(None)), b=128)
            else:
                s.k("aq_p2g_v", N, (i32(N), P(IP), P(s.Np), P(s.fx), P(s.m), P(s.pvc), P(s.pmk), P(s.vm), P(s.cm)))
            s.k("aq_vbar", capN, (P(IP), P(s.cm), P(s.vm)), b=256)
        if s.has_pbc:
            s.zf(s.pacc, 9, s.K * s.pbc_ns)
            s.k("ao_pbc_acc", N, (i32(N), P(IP), P(s.Np), P(s.fx), P(s.m), P(s.trel), P(s.tbuf), i32(s.K), i32(s.pbc_ns), f(s.dt), P(s.pacc)))

    def mat_post_anode(s):
        if s.aq and s.vfill:
            s.k("aq_vfill", s.capN, (P(s.IP), P(s.mn), P(s.vk), P(s.ak)), b=256)

    def mat_floor_n(s, FA, kstep, capN):
        i32, f = np.int32, np.float32
        if s.has_pbc:
            s.k("ao_pbc_k", capN, (P(s.IP), P(s.mn), P(s.pacc), i32(s.K), i32(s.pbc_ns), i32(kstep), f(s.dt / s.pbc_ns / s.dt_gt), P(s.kb)), b=256)
        s.k("mt_floor_n", capN, tuple(FA) + (P(s.kb if s.has_pbc else None), P(s.mn), f(s.mthr), P(s.vm if s.pvb is not None else None)), b=256)

    def mat_elem(s, kstep, it, nst, capQ):
        i32, f = np.int32, np.float32
        IP = s.IP
        last = kstep == nst - 1 and it == s.nt - 1
        if s.aq:
            al = s.ga_c["gm"] * s.dt / ((1 - s.ga_c["af"]) * s.ga_c["be"] * s.dt * s.dt)
            s.k("aq_elem", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), i32(s._mat), P(s.prm), P(s.Sq), f(s.dt), P(s.f), P(s.H30), i32(capQ), P(s.B),
                                  P(s.cnt if last else None), P(s.z0), f(al), f(s.visc), P(s.qb)))
            if not s.hspec:
                s.k("mt_h81", capQ, (P(IP), P(s.Fq), P(s.Vq), P(s.H30), P(s.H), i32(capQ)))
        else:
            s.k("ao_elem_s" if s.hspec else "ao_elem", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), f(s.mu), f(s.lam), f(s.cdp), i32(1 | 2 | 8), P(s.f), P(s.H), i32(capQ), P(s.B),
                                  P(s.cnt if last else None)))

    def mat_step_end(s, kstep, nst, capN, capQ):
        i32, f = np.int32, np.float32
        IP = s.IP
        if s.has_pbc and s.ar:
            s.k("mt_akb", capN, (P(IP), P(s.kb), P(s.ak)), b=256)
        s.dcopy(s.Ust[kstep], s.U)
        if kstep < nst - 1:
            if s.aq:
                s.k("aq_commit", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.U), i32(s._mat), P(s.prm), P(s.Sq), f(s.dt)))
            else:
                s.k("ao_commit", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.U), f(s.cdp)))

    def mat_g2p(s):
        N, IP, i32, f = s.N, s.IP, np.int32, np.float32
        s.k("ao_g2p_a", N, (i32(N), P(IP), P(s.Np), P(s.fx), P(s.ak), P(s.pa)))
        if s.aq:
            s.k("aq_g2p", N, (i32(N), P(IP), P(s.Ai), P(s.Np), P(s.fx), P(s.Ust), i32(s.K), i32(s.capN), P(s.vk), i32(s._mat), P(s.prm), f(s.dt), P(s.x), P(s.v), P(s.F),
                              P(s.ysp), P(s.ljp), P(s.alp), P(s.cnt)))
        else:
            s.k("ao_g2p", N, (i32(N), P(IP), P(s.Ai), P(s.Np), P(s.fx), P(s.Ust), i32(s.K), i32(s.capN), P(s.vk), f(s.cdp), P(s.x), P(s.v), P(s.F), P(s.cnt)))

    def counts(s):
        o = super().counts()
        c = s.cnt.cpu().tolist()
        o.update(q_tail=c[0], q_apex=c[1], p_rm=c[2])
        return o
