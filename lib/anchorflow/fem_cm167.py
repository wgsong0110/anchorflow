"""절 CM167 묶음 cm167 (조정자 11-14, 진단 -- 결과 불변; fem_nlwhy 위): 윗덩어리 (걸음 시작 z > ZC 노드·가우스 점) 의 뉴턴 반복별 기록.
반복마다 (그래프 안, rt_grad 뒤·재질 커널 뒤 에 작은 커널을 더 부름; 풀이 버퍼는 읽기만):
  노드 (질량 > 3e-5 kg, 걸음 시작 z > ZC): 질량 M, 부호 있는 순 잔차 Σ g_z (g0buf = 그 반복 시작 기울기), Σ|g|, Σ m (y − Ũ)_z, Σ m y_z, Σ g_x, Σ g_y.
  가우스 점 (시작 위치 z > ZC): 시험 X 의 상태별 수 (0 탄성·1 꼬리·2 꼭짓점) 와 앞 평가 대비 바뀐 점 (전체·탄성<->소성·꼬리<->꼭짓점).
  전체 (실행기 진단 s.DG 그대로): 선탐색 직후 SC -- 증분 포텐셜 E0 = 관성 SC[20] + 탄성 SC[10], 후보 c = 2^-i 의 E_i = SC[21+i] + SC[40+i], g·δ SC[9], 고른 보폭 SC[14], 반복 시작 |g|² SC[1].
창 프레임 (env CM167_F0~F1, 1 부터) 의 첫 걸음 (K 1) 마다 <out>/cm167.jsonl 한 줄. fem_nlwhy 가 설치돼 있어야 함 (반복 번호 s._cw_k·_cw_it 를 그쪽 ctl 훅이 셈).
"""
import json
import os

import numpy as np
import torch

ST = dict(out=".", nsrc=0, nalloc=0, nsum=0, npst=0, nframe=0, nwrite=0, err=None, F0=1, F1=0)
LOG = [None]
MARK = "// ---- 절 CM167 cm167"
HEAD = "__global__ void ao_commit("
SRC = r"""
// ---- 절 CM167 cm167: 윗덩어리 노드 합 (읽기만) ----
__global__ void cm167_sum(const int* IP, const float* A, const float* U, const float* mn, const float* g, const float* y, const float* Ut, float zc, float thr, double* out) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n]; if (!(m > thr)) return;
    int nb1 = IP[7], nb2 = IP[8];
    float L0 = (float)(n/(nb1*nb2) + IP[0]), L1 = (float)((n/nb2)%nb1 + IP[1]), L2 = (float)(n%nb2 + IP[2]);
    float z = A[6]*L0 + A[7]*L1 + A[8]*L2 + U[n*3+2];
    if (!(z > zc)) return;
    atomicAdd(&out[0], (double)m);
    atomicAdd(&out[1], (double)g[n*3+2]);
    atomicAdd(&out[2], (double)sqrtf(g[n*3]*g[n*3] + g[n*3+1]*g[n*3+1] + g[n*3+2]*g[n*3+2]));
    atomicAdd(&out[3], (double)m*(double)(y[n*3+2] - Ut[n*3+2]));
    atomicAdd(&out[4], (double)m*(double)y[n*3+2]);
    atomicAdd(&out[5], (double)g[n*3]);
    atomicAdd(&out[6], (double)g[n*3+1]);
}
// ---- 윗덩어리 가우스 점 상태 바뀜 (fem_nlwhy nl_pst 와 같은 식, 시작 위치 z > ZC 점만) ----
__global__ void cm167_pst(const int* IP, const int* cb, const float* BG, const float* Fq, const float* y, float cdp, const float* xq0, float zc, int* stb, int* row) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    if (!(xq0[q*3+2] > zc)) return;
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
    return code[:i] + MARK + "\n" + SRC + code[i:]


def _dump():
    try:
        with open(os.path.join(ST["out"], "cm167info.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def install(out):
    from . import fem_an_rt as ART
    ST.update(out=out, F0=int(os.environ.get("CM167_F0", "1")), F1=int(os.environ.get("CM167_F1", "0")))
    ZC = float(os.environ.get("CM167_ZC", "0.9"))
    LOG[0] = open(os.path.join(out, "cm167.jsonl"), "w")
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
    oa, ok = R.alloc, R.k

    def alloc(s, need):
        r = oa(s, need)
        try:
            from . import fem_convwhy as CW
            KK, nt = CW._kmax(s), int(s.nt)
            s.c167S = torch.zeros(KK * nt, 8, device=s.dev, dtype=torch.float64)
            s.c167P = torch.zeros(KK * nt, 8, device=s.dev, dtype=torch.int32)
            capQ = s.Fq.numel() // 9
            s.c167T = torch.full((capQ,), -1, device=s.dev, dtype=torch.int32)
            s.c167A = s.A.float().contiguous().reshape(-1).clone()
            ST["nalloc"] += 1
        except Exception as e:
            ST["err"] = "alloc " + repr(e)
        return r

    def row(s):
        return max(getattr(s, "_cw_k", 0), 0) * int(s.nt) + getattr(s, "_cw_it", 0)

    def k(s, name, n, args, b=128):
        r = ok(s, name, n, args, b)
        if getattr(s, "c167S", None) is not None and getattr(s, "cwS3", None) is not None:
            if name == "rt_grad":
                i = row(s)
                ok(s, "cm167_sum", s.capN, (P(s.IP), P(s.c167A), P(s.U), P(s.mn), P(s.g0buf), P(s.y), P(s.Ut), np.float32(ZC), np.float32(3e-5), P(s.c167S[i])), 256)
                ST["nsum"] += 1
            elif name in ("ao_elem_s", "ao_elem"):
                i = row(s)
                ok(s, "cm167_pst", n, (args[0], args[1], args[2], args[3], args[5], args[8], P(s.xq0), np.float32(ZC), P(s.c167T), P(s.c167P[i])), 128)
                ST["npst"] += 1
        return r
    R.alloc, R.k = alloc, k

    def frame(fn, s, fi, *a):
        if getattr(s, "_c167_in", False):
            return fn(s, *a)
        s._c167_in = True
        try:
            f_ = fi + 1
            win = ST["F0"] <= f_ <= ST["F1"]
            if getattr(s, "c167S", None) is not None:
                s.c167S.zero_(); s.c167P.zero_(); s.c167T.fill_(-1)
            r = fn(s, *a)
            if win and getattr(s, "c167S", None) is not None and getattr(s, "cwS3", None) is not None:
                try:
                    torch.cuda.synchronize()
                    nt = int(s.nt)
                    S_, P_ = s.c167S.cpu().numpy(), s.c167P.cpu().numpy()
                    D = s.DG.cpu().numpy()
                    its = []
                    for it in range(nt):
                        if S_[it, 0] <= 0 and D[2 * it, 1] == 0:
                            break
                        d0, d1 = D[2 * it], D[2 * it + 1]
                        o = dict(it=it, M=float(S_[it, 0]), gz=float(S_[it, 1]), gabs=float(S_[it, 2]), ymu=float(S_[it, 3]), yz=float(S_[it, 4]), gx=float(S_[it, 5]), gy=float(S_[it, 6]),
                                 pl=P_[it].tolist(), g2=float(d0[1]), E0=float(d1[20] + d1[10]), gd=float(d1[9]), c=float(d1[14]),
                                 Ec=[float(d1[21 + j] + d1[40 + j]) for j in range(4)])
                        its.append(o)
                    LOG[0].write(json.dumps(dict(f=f_, K=int(s.Kcur), its=its), ensure_ascii=False) + "\n"); LOG[0].flush()
                    ST["nwrite"] += 1
                except Exception as e:
                    ST["err"] = "read " + repr(e)[:300]
            ST["nframe"] += 1
            _dump()
            return r
        finally:
            s._c167_in = False

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
    print(f"[절 CM167] cm167 켬 -- 프레임 {ST['F0']}~{ST['F1']} 윗덩어리 (z>{ZC}) 뉴턴 반복별 순 잔차·증분 포텐셜·상태 바뀜 (결과 불변)", flush=True)
    _dump()
