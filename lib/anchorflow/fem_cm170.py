"""절 CM170 묶음 cm170 (조정자 11-14, 진단 -- 결과 불변; fem_nlwhy 위): 먼 쪽 (구속면 x = 1.0 에서 21 cm 너머, 층 z 0.5~1.2 자유 노드) 의 뉴턴 반복별 평형 잔차.
반복마다 (그래프 안, 읽기만):
  - f 를 건드리는 커널 (인자에 풀이 버퍼 s.f 가 있는 커널) 마다 먼 쪽 노드의 Σ f_z · Σ f_x (그 반복의 누적 기울기; 커널 이름 순서 = 항 분해: 탄성 -> 장벽 -> 고스트 -> 구속 ...) 를 칸 하나씩에.
  - rt_grad 뒤: 질량 M, Σ m (y − Ũ)_z · _x, Σ Mi η_z (풀이가 쓰는 관성 항, Mi = meff/τ²), 점 수, 1/τ².
  - rt_lsup 뒤 (그 반복 갱신 직후): 같은 관성 합 (갱신된 y).
평형 잔차 (위 +) = Σ m (y − Ũ)_z / τ² + Σ f_z (f = 에너지 기울기) 를 M 으로 나눠 m/s². 프레임 창 (env CM170_F0~F1, 1 부터) 의 첫 걸음마다 <out>/cm170.jsonl 한 줄.
"""
import json
import os

import numpy as np
import torch

LSN = ("rt_lsyN", "rt_fixdir", "rt_sc_clear_g", "rt_elem_lsN", "rt_bar_lsN", "rt_lsgate", "rt_lspickN", "rt_sc_clear")
ST = dict(lsn={}, out=".", nsrc=0, nalloc=0, nsum=0, nframe=0, nwrite=0, err=None, F0=1, F1=0, names={}, itau2={})
LOG = [None]
MARK = "// ---- 절 CM170 cm170"
HEAD = "__global__ void ao_commit("
NS = 16
SRC = r"""
// ---- 절 CM170 cm170: 먼 쪽 노드 합 (읽기만) ----
__global__ void cm170_sum(const int* IP, const float* A, const float* mn, const float* meff, const double* f, const float* y, const float* Ut, const float* eta,
                          const float* sk, const int* pm, const float* dl, float itau2, float dmin, float zlo, float zhi, float thr, double* out) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float m = mn[n]; if (!(m > thr)) return;
    if (!(sk[n] > 0.999f) || pm[n] < 0 || pm[n] == 2) return;   // 고착 1 = T = I (pm 0·1 둘 다; 먼 쪽 노드는 pm 1)
    int nb1 = IP[7], nb2 = IP[8];
    float L0 = (float)(n/(nb1*nb2) + IP[0]), L1 = (float)((n/nb2)%nb1 + IP[1]), L2 = (float)(n%nb2 + IP[2]);
    float x = A[0]*L0 + A[1]*L1 + A[2]*L2, z = A[6]*L0 + A[7]*L1 + A[8]*L2 - 0.5f;
    float d = (x - 1.0f)*100.f;
    if (!(d >= dmin) || !(z >= zlo) || !(z < zhi)) return;
    atomicAdd(&out[0], (double)m);
    atomicAdd(&out[1], (double)m*(double)(y[n*3+2] - Ut[n*3+2]));
    atomicAdd(&out[2], (double)meff[n]*(double)itau2*(double)eta[n*3+2]);
    atomicAdd(&out[3], f[n*3+2]);
    atomicAdd(&out[4], f[n*3]);
    atomicAdd(&out[5], (double)m*(double)(y[n*3] - Ut[n*3]));
    atomicAdd(&out[6], (double)meff[n]*(double)itau2*(double)dl[n*3+2]);
    atomicAdd(&out[7], 1.0);
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
        with open(os.path.join(ST["out"], "cm170info.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def install(out):
    from . import fem_an_rt as ART
    ST.update(out=out, F0=int(os.environ.get("CM170_F0", "1")), F1=int(os.environ.get("CM170_F1", "0")))
    DMIN = float(os.environ.get("CM170_DMIN", "21")); ZLO = float(os.environ.get("CM170_ZLO", "0.5")); ZHI = float(os.environ.get("CM170_ZHI", "1.2"))
    LOG[0] = open(os.path.join(out, "cm170.jsonl"), "w")
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
    oa, ok, octl = R.alloc, R.k, R.ctl

    def alloc(s, need):
        r = oa(s, need)
        try:
            from . import fem_convwhy as CW
            KK, nt = CW._kmax(s), int(s.nt)
            s.c170S = torch.zeros(KK * nt, NS, 8, device=s.dev, dtype=torch.float64)
            s.c170A = s.A.float().contiguous().reshape(-1).clone()
            s.c170L = torch.zeros((KK * nt,) + tuple(s.SC.shape), device=s.dev, dtype=torch.float64)
            s.c170W = torch.zeros(KK * nt, 14, 8, device=s.dev, dtype=torch.float64)
            s.c170X = torch.tensor([9, 10, 14, 20, 21, 40, 59, 63], device=s.dev)
            s._c170_w = {}
            s._c170_on = False; s._c170_post = False; s._c170_wsnap = False; s._c170_seq = {}
            ST["nalloc"] += 1
        except Exception as e:
            ST["err"] = "alloc " + repr(e)
        return r

    def row(s):
        return max(getattr(s, "_cw_k", 0), 0) * int(s.nt) + getattr(s, "_cw_it", 0)

    def ctl(s, op, it=0):
        r = octl(s, op, it)
        if getattr(s, "c170S", None) is not None and getattr(s, "cwS3", None) is not None:
            if op == 2:
                s._c170_on = True; s._c170_post = False; s._c170_wsnap = False; s._c170_seq[row(s)] = 0; ST["names"][row(s)] = []
            elif op == 3:
                s._c170_on = False; s._c170_post = False; s._c170_wsnap = True; s._c170_w[row(s)] = 0; ST["lsn"][row(s)] = []
        return r

    def summ(s, name, args, slot, itau2):
        i = row(s)
        ok(s, "cm170_sum", s.capN, (P(s.IP), P(s.c170A), P(s.mn), P(s.meff), P(s.f), P(s.y), P(s.Ut), P(s.eta), P(s.sk), P(s.pm), P(s.dl), np.float32(itau2),
                                    np.float32(DMIN), np.float32(ZLO), np.float32(ZHI), np.float32(3e-5), P(s.c170S[i, slot])), 256)
        ST["nsum"] += 1

    def k(s, name, n, args, b=128):
        r = ok(s, name, n, args, b)
        if getattr(s, "c170S", None) is not None and getattr(s, "cwS3", None) is not None:
            i = row(s)
            if name in LSN and not s._c170_post and s._c170_wsnap:
                q = s._c170_w.get(i, 0)
                if q < 14:
                    s.c170W[i, q].copy_(s.SC[s.c170X]); s._c170_w[i] = q + 1
                    ST["lsn"].setdefault(i, []).append(name)
            if name == "rt_lsup" and s._c170_wsnap:
                s._c170_post = True; s._c170_wsnap = False
                s.dcopy(s.c170L[i], s.SC)                                          # 선탐색 판정 직후 (nlwhy 도 이 시점 SC[14] 를 읽음)
            if name == "rt_grad":
                it2 = float(args[2]); ST["itau2"][i] = it2
                summ(s, name, args, NS - 1, it2)
            elif s._c170_on:
                fp = int(s.f.data_ptr())
                if any(isinstance(a, np.uint64) and int(a) == fp for a in args):
                    q = s._c170_seq.get(i, 0)
                    if q < NS - 2:
                        summ(s, name, args, q, 0.0)
                        ST["names"].setdefault(i, []).append(name)
                        s._c170_seq[i] = q + 1
            elif name == "rt_lsup" and s._c170_post:
                summ(s, name, args, NS - 2, ST["itau2"].get(i, 0.0))
        return r
    R.alloc, R.k, R.ctl = alloc, k, ctl

    def frame(fn, s, fi, *a):
        if getattr(s, "_c170_in", False):
            return fn(s, *a)
        s._c170_in = True
        try:
            f_ = fi + 1
            win = ST["F0"] <= f_ <= ST["F1"]
            if getattr(s, "c170S", None) is not None:
                s.c170S.zero_(); s.c170L.zero_()
            r = fn(s, *a)
            if win and getattr(s, "c170S", None) is not None and getattr(s, "cwS3", None) is not None:
                try:
                    torch.cuda.synchronize()
                    nt = int(s.nt)
                    S_ = s.c170S.cpu().numpy()
                    D = s.DG.cpu().numpy()
                    L_ = s.c170L.cpu().numpy(); W_ = s.c170W.cpu().numpy()
                    its = []
                    try:
                        G_ = s.cwG.cpu().numpy()
                    except Exception:
                        G_ = None

                    def cgi(it):
                        if G_ is None:
                            return None
                        i_ = max(getattr(s, "_cw_k", 0), 0) * 0 + it
                        g = G_[i_]
                        nz = [j for j in range(g.shape[0]) if g[j, 0] != 0]
                        return dict(n=len(nz), pap=[float(g[j, 2]) for j in nz], rz=[float(g[j, 0]) for j in nz], pr=[float(g[j, 4]) for j in nz], rz0=float(g[nz[0], 0]) if nz else 0.0, rz1=float(g[nz[-1], 0]) if nz else 0.0, rr1=float(g[nz[-1], 5]) if nz else 0.0)
                    for it in range(nt):
                        g_ = S_[it, NS - 1]
                        if g_[0] <= 0:
                            break
                        M = float(g_[0]); it2 = ST["itau2"].get(it, 0.0)
                        nm = ST["names"].get(it, [])
                        fz = [float(S_[it, j, 3] / M) for j in range(len(nm))]
                        fx = [float(S_[it, j, 4] / M) for j in range(len(nm))]
                        p_ = S_[it, NS - 2]
                        o = dict(it=it, M=M, n=float(g_[7]), itau2=it2, inert_def=float(g_[1] * it2 / M), inert_slv=float(g_[2] / M), inert_x=float(g_[5] * it2 / M),
                                 names=nm, fz=fz, fx=fx, resid=float(g_[1] * it2 / M + (fz[-1] if fz else 0.0)),
                                 post_inert=float(p_[1] * it2 / M) if p_[0] > 0 else None, dl_slv=float(p_[6] / M) if p_[0] > 0 else None, cg=cgi(it), sc1=[float(v) for v in D[2 * it + 1][:64]], scL=[float(v) for v in L_[it][:64]], lsn=ST["lsn"].get(it, []), lsw=[[float(v) for v in W_[it, j]] for j in range(len(ST["lsn"].get(it, [])))], g2=float(D[2 * it, 1]), rz=float(D[2 * it, 0]))
                        its.append(o)
                    LOG[0].write(json.dumps(dict(f=f_, K=int(s.Kcur), its=its), ensure_ascii=False) + "\n"); LOG[0].flush()
                    ST["nwrite"] += 1
                except Exception as e:
                    ST["err"] = "read " + repr(e)[:300]
            ST["nframe"] += 1
            _dump()
            return r
        finally:
            s._c170_in = False

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
    print(f"[절 CM170] cm170 켬 -- 프레임 {ST['F0']}~{ST['F1']} 먼 쪽 (d>={DMIN} cm) 노드 뉴턴 반복별 평형 잔차 (결과 불변)", flush=True)
    _dump()
