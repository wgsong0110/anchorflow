"""절 CM9.3 진단 (--nan_step; 결과 불변 -- 풀이 버퍼를 읽기만 하고 따로 둔 표시 버퍼에만 씀): 프레임 안 걸음·뉴턴 반복마다 처음 비유한·뒤집힘이 된 시점을 노드·가우스 점별로 기록.

옛 재질 실행기 (lib/anchorflow/fem_ao_rt.py AORTSolver) 의 record() 에 그래프 안 검사 커널을 끼운다 (끄면 record() 가 내는 커널 열이 그대로).
시점 부호 code = ((k + 1) * 8 + ph) * 2 + sub  (k = 걸음 0..K-1, P2G 는 k = -1):
  ph 0 sub 0 = (P2G: 프레임 시작 상태) / (걸음: 시작점 y -- 관성 예측·바닥·시작점 필터 뒤)
  ph i+1 sub 0 = 뉴턴 반복 i 의 기울기 (rt_grad 뒤 노드 힘 f, 전역 스칼라 SC)
  ph i+1 sub 1 = 뉴턴 반복 i 끝 (rt_lsup 뒤 y)
  ph nt+1 = 걸음 끝 (U, 속도 vk, HHT 가속도 gaA), ph nt+2 = 커밋 (가우스 점 F_q)
기록 (프레임마다 처음으로 되돌림; 값 = 그 성질이 처음 참이 된 code, 아니면 UNSET):
  노드 mN: 변위 (y 또는 U) 비유한, mV: 속도·가속도 비유한, mF: 기울기 (노드 힘 f, f64) 비유한, jm/jk: 걸음 사이 변위 증분 |U_k − U_{k−1}| 최대 (칸 단위) 와 그 걸음
  가우스 점 (부피 > 0 인 점만 뒤집힘·응력): mJ: 시험 X = F_q + ∇U F_q 의 det 이 유한한데 ≤ 0, mS: 응력 (aq_elem 과 같은 식으로 다시 계산한 P = U diag(p) Vᵀ, 재질 RM 은 지역 사본) 또는 SVD 비유한,
  mQ: 가우스 점 상태 (F_q, S_q) 비유한, jmin: 프레임 동안 det X 최솟값
  전역 gsc: SC[0..15] (|g|²·r·z 등 PCG 스칼라) 비유한
"""
import numpy as np
import torch

from . import fem_an_rt_v0 as ART

P = ART.P
UNSET = 0x7FFFFFFF

NSD_SRC = r'''
#define NSD_UNSET 0x7fffffff
__global__ void nsd_fill_i(int n, int* a, int v) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = v; }
__global__ void nsd_fill_f(int n, float* a, float v) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) a[i] = v; }
__global__ void nsd_node(const int* IP, const float* v, int* mk, int code) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || n >= IP[9] || mk[n] != NSD_UNSET) return;
    if (!isfinite(v[n*3]) || !isfinite(v[n*3+1]) || !isfinite(v[n*3+2])) mk[n] = code;
}
__global__ void nsd_node_d(const int* IP, const double* v, int* mk, int code) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || n >= IP[9] || mk[n] != NSD_UNSET) return;
    if (!isfinite(v[n*3]) || !isfinite(v[n*3+1]) || !isfinite(v[n*3+2])) mk[n] = code;
}
__global__ void nsd_jump(const int* IP, const float* U, const float* Up, float cinv, float* jm, int* jk, int k) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || n >= IP[9]) return;
    float d0 = U[n*3] - (Up ? Up[n*3] : 0.f), d1 = U[n*3+1] - (Up ? Up[n*3+1] : 0.f), d2 = U[n*3+2] - (Up ? Up[n*3+2] : 0.f);
    float d = sqrtf(d0*d0 + d1*d1 + d2*d2)*cinv;
    if (isfinite(d) && d > jm[n]) { jm[n] = d; jk[n] = k; }
}
__global__ void nsd_sc(const double* SC, int* g, int code) {
    if (g[0] != NSD_UNSET) return;
    for (int i=0;i<16;i++) if (!isfinite(SC[i])) { g[0] = code; return; }
}
// 가우스 점: X = F_q + A F_q (A = Σ_o y_o ⊗ ∇w_o, aq_elem 과 같은 식), y = 0 이면 X = F_q
__global__ void nsd_q(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, const float* Sq,
                      int mat, const float* prm, float dt, int dostress, int code, int* mJ, int* mS, int* mQ, float* jmin) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || q >= IP[12]) return;
    int oc = q & 7;
    float F[9]; bool qbad = false;
    for (int k=0;k<9;k++) { F[k] = Fq[(size_t)q*9+k]; if (!isfinite(F[k])) qbad = true; }
    if (Sq) for (int k=0;k<3;k++) if (!isfinite(Sq[(size_t)q*4+k])) qbad = true;
    if (qbad && mQ[q] == NSD_UNSET) mQ[q] = code;
    float vq = Vq[q];
    if (!(vq > 0.f)) return;
    float A[9] = {0,0,0,0,0,0,0,0,0};
    if (y) { int ids[27]; qnodes(q, IP, cb, ids);
        for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; int id = ids[o];
            for (int a=0;a<3;a++) { float ya = y[id*3+a]; A[a*3] += ya*bg[0]; A[a*3+1] += ya*bg[1]; A[a*3+2] += ya*bg[2]; } } }
    float X[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) X[a*3+b] = F[a*3+b] + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b];
    double J = (double)X[0]*((double)X[4]*X[8] - (double)X[5]*X[7]) - (double)X[1]*((double)X[3]*X[8] - (double)X[5]*X[6]) + (double)X[2]*((double)X[3]*X[7] - (double)X[4]*X[6]);
    if (isfinite(J)) { if ((float)J < jmin[q]) jmin[q] = (float)J; if (J <= 0.0 && mJ[q] == NSD_UNSET) mJ[q] = code; }
    if (!dostress || mS[q] != NSD_UNSET) return;
    float U[9], S[3], V[9];
    ssvdf(X, U, S, V);
    bool bad = false;
    for (int k=0;k<9;k++) if (!isfinite(U[k]) || !isfinite(V[k])) bad = true;
    float ys = Sq[(size_t)q*4], lj = Sq[(size_t)q*4+1], al = Sq[(size_t)q*4+2];
    float sev = (mat == 7) ? prm[14] : 0.f;
    float se[3]; for (int i=0;i<3;i++) { se[i] = (sev > 0.f) ? fmaxf(S[i], sev) : S[i]; if (!isfinite(S[i])) bad = true; }
    if (!(prm[15] > 0.5f)) pl_rm(mat, se, prm, ys, lj, al, dt);
    float Pe[3]; pl_pi(mat, se, prm, al, Pe);
    float ss[3], p[3];
    for (int i=0;i<3;i++) { ss[i] = (sev > 0.f) ? fmaxf(S[i], sev) : ((fabsf(S[i]) > 1e-6f) ? S[i] : ((S[i] < 0.f) ? -1e-6f : 1e-6f)); p[i] = Pe[i]*se[i]/ss[i]; }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) { float t = U[a*3]*p[0]*V[b*3] + U[a*3+1]*p[1]*V[b*3+1] + U[a*3+2]*p[2]*V[b*3+2]; if (!isfinite(t)) bad = true; }
    if (bad) mS[q] = code;
}
'''


def code(k, ph, sub=0):
    return ((k + 1) * 8 + ph) * 2 + sub


def decode(c, nt):
    """code -> (걸음 k, 단계 이름) -- k = -1 은 P2G."""
    c = int(c)
    if c == UNSET:
        return None
    sub = c & 1; c >>= 1; ph = c % 8; k = c // 8 - 1
    if k < 0:
        return (k, "P2G (프레임 시작)")
    if ph == 0:
        return (k, "시작점")
    if 1 <= ph <= nt:
        return (k, f"뉴턴 {ph} " + ("기울기" if sub == 0 else "끝"))
    if ph == nt + 1:
        return (k, "걸음 끝")
    return (k, "커밋")


class NSD:
    """AORTSolver 에 붙이는 진단 상태. s = 실행기."""

    def __init__(s_, S, src):
        s_.S = S
        r = src.rstrip()
        assert r.endswith("}")
        s_.mod = ART.cp.RawModule(code=r[:-1] + NSD_SRC + "\n}\n", options=("--std=c++14",))
        s_.fn = {}
        s_.alloc()

    def alloc(s_):
        S = s_.S; dev = S.dev
        capN, capQ = S.capN, S.capQc * 8
        I = lambda n: torch.full((n,), UNSET, device=dev, dtype=torch.int32)
        s_.mN, s_.mV, s_.mF, s_.jk = I(capN), I(capN), I(capN), I(capN)
        s_.jm = torch.zeros(capN, device=dev, dtype=torch.float32)
        s_.mJ, s_.mS, s_.mQ = I(capQ), I(capQ), I(capQ)
        s_.jmin = torch.full((capQ,), 1e30, device=dev, dtype=torch.float32)
        s_.gsc = I(1)
        s_.capN, s_.capQ = capN, capQ

    def k(s_, name, n, args, b=256):
        if name not in s_.fn:
            s_.fn[name] = s_.mod.get_function(name)
        with ART.RTC._Stream():
            s_.fn[name](((int(n) + b - 1) // b,), (b,), args)

    # ---- record() 안에서 부르는 것 (그래프에 들어감)
    def reset(s_):
        i32 = np.int32
        for t in (s_.mN, s_.mV, s_.mF, s_.jk, s_.mJ, s_.mS, s_.mQ, s_.gsc):
            s_.k("nsd_fill_i", t.numel(), (i32(t.numel()), P(t), i32(UNSET)))
        s_.k("nsd_fill_f", s_.capN, (i32(s_.capN), P(s_.jm), np.float32(0.0)))
        s_.k("nsd_fill_f", s_.capQ, (i32(s_.capQ), P(s_.jmin), np.float32(1e30)))

    def node(s_, t, mk, c):
        S = s_.S
        nm = "nsd_node_d" if t.dtype == torch.float64 else "nsd_node"
        s_.k(nm, S.capN, (P(S.IP), P(t), P(mk), np.int32(c)))

    def q(s_, y, c):
        S = s_.S
        aq = bool(getattr(S, "aq", False))
        s_.k("nsd_q", s_.capQ, (P(S.IP), P(S.cb), P(S.BG), P(S.Fq), P(S.Vq), P(y), P(S.Sq if aq else None), np.int32(S.mat), P(S.prm if aq else None), np.float32(S.dt),
                                np.int32(1 if aq else 0), np.int32(c), P(s_.mJ), P(s_.mS), P(s_.mQ), P(s_.jmin)), b=128)

    def sc(s_, c):
        s_.k("nsd_sc", 1, (P(s_.S.SC), P(s_.gsc), np.int32(c)), b=1)

    def jump(s_, kstep):
        S = s_.S
        s_.k("nsd_jump", S.capN, (P(S.IP), P(S.U), P(S.Ust[kstep - 1] if kstep > 0 else None), np.float32(1.0 / float(S.A[:, 0].norm())), P(s_.jm), P(s_.jk), np.int32(kstep)))

    # ---- 프레임 뒤 (그래프 밖)
    def frame_counts(s_):
        S = s_.S; ip = S.IP.cpu().tolist(); nn, nq = ip[9], ip[12]
        f = lambda t, n: int((t[:n] != UNSET).sum())
        return dict(nN=f(s_.mN, nn), nV=f(s_.mV, nn), nF=f(s_.mF, nn), nJ=f(s_.mJ, nq), nS=f(s_.mS, nq), nQ=f(s_.mQ, nq), gsc=int(s_.gsc[0]),
                    jmax=float(s_.jm[:nn].max()) if nn else 0.0, nj1=int((s_.jm[:nn] > 1.0).sum()))

    def dump(s_, path, x_prev_orig, extra=None):
        """발산 프레임 뒤: 입자마다 자기 칸 (27 노드·8 가우스 점) 의 처음 시점·원인 분류 + 노드·점 원자료를 npz 로."""
        S = s_.S; dev = S.dev
        ip = S.IP.cpu().tolist(); nn, nc, nq = ip[9], ip[11], ip[12]; nb1, nb2 = ip[7], ip[8]
        cb = S.cb[:nc].long()
        ids = []
        for o in range(27):
            i_, j_, k_ = o // 9, (o // 3) % 3, o % 3
            ids.append(((cb[:, 0] + i_) * nb1 + (cb[:, 1] + j_)) * nb2 + (cb[:, 2] + k_))
        ids = torch.stack(ids, 1)                                                     # [nc, 27]
        mn = S.mn[:nn].float(); med = float(mn[mn > 0].median())
        cN = s_.mN[ids].min(1).values; cF = s_.mF[ids].min(1).values; cV = s_.mV[ids].min(1).values
        jmc, ja = s_.jm[ids].max(1); jn = ids.gather(1, ja[:, None])[:, 0]
        cJ = s_.mJ[:nq].reshape(nc, 8).min(1).values; cS = s_.mS[:nq].reshape(nc, 8).min(1).values; cQ = s_.mQ[:nq].reshape(nc, 8).min(1).values
        cJmin = s_.jmin[:nq].reshape(nc, 8).min(1).values
        # 입자 -> 칸 (압축 번호): pcd = 상자 칸 번호, cidx = 압축 번호 (프레임 시작 rt_coord·rt_mark 값; G2P 뒤에도 그대로)
        pc = S.cidx[S.pcd.long()].long()
        inv = torch.empty_like(S.perm); inv[S.perm] = torch.arange(S.N, device=dev)
        po = pc[inv]                                                                  # 원래 입자 순서
        g = lambda t: t[po].cpu().numpy()
        xo = S.x_orig()
        Fo = torch.empty_like(S.F); Fo[S.perm] = S.F
        out = dict(x_prev=x_prev_orig.float().cpu().numpy(), nan=(~torch.isfinite(xo).all(-1)).cpu().numpy(), Fnan=(~torch.isfinite(Fo).all(-1)).cpu().numpy(),
                   p_cell=po.cpu().numpy().astype(np.int32),
                   p_cN=g(cN), p_cF=g(cF), p_cV=g(cV), p_cJ=g(cJ), p_cS=g(cS), p_cQ=g(cQ), p_jmax=g(jmc), p_jstep=g(s_.jk[jn]), p_jmrel=g(mn[jn] / med), p_Jmin=g(cJmin),
                   n_mN=s_.mN[:nn].cpu().numpy(), n_mF=s_.mF[:nn].cpu().numpy(), n_mV=s_.mV[:nn].cpu().numpy(), n_jm=s_.jm[:nn].cpu().numpy(), n_jk=s_.jk[:nn].cpu().numpy(),
                   n_mrel=(mn / med).cpu().numpy(), n_lo=np.array(ip[0:3], np.int64), n_nb=np.array(ip[6:9], np.int64),
                   q_mJ=s_.mJ[:nq].cpu().numpy(), q_mS=s_.mS[:nq].cpu().numpy(), q_mQ=s_.mQ[:nq].cpu().numpy(), q_jmin=s_.jmin[:nq].cpu().numpy(), q_vq=S.Vq[:nq].cpu().numpy(),
                   c_cb=cb.cpu().numpy().astype(np.int32), A=S.A.cpu().numpy(), gsc=np.array([int(s_.gsc[0])]), IP=np.array(ip), K=np.array([S.K]), nt=np.array([S.nt]),
                   mthr_rel=np.array([S.mthr / med]), cell=np.array([float(S.A[:, 0].norm())]))
        if extra:
            out.update(extra)
        np.savez_compressed(path, **out)
        return out
