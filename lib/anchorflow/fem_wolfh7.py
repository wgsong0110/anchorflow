"""절 CM79 묶음 wolfh7 (11-08 위임): wolf 높게 쌓임 7 -- P1X 강성을 칸 재질 항복 몫으로 연속 가중 (Y7, --w7y 1). 문턱·클램프 없음.

배경 (절 CM77): P1X 강성 k (ghost penalty 강성 벌점) 을 wolf 전체에서 빼면 (K, κ 0) 매달림은 풀리나 (낙하 몫 0.04 -> 0.31) 가벼운 노드 풀이가 불안정 (4/4 발산).
  남은 갈래 = 강성을 재질 상태로 연속 가중 -- 모래가 흐르는 칸에서만 빼고 탄성 (쌓여 눌린) 칸은 그대로.
정의:
  가우스 점 q 의 한 걸음 (커밋 = 걸음 끝 반환 사상) 에서 시험 탄성 X = (I + Σ_o U_o ⊗ ∇w_o) F_q (커밋 커널 ao_commit·aq_commit 과 같은 식), 반환 뒤 탄성 Y,
  걸음 시작 탄성 F_q 의 로그 주신장 (부호 있는 SVD 주값 |s| 의 log, 큰 것부터) ε_tr = log s(X), ε_1 = log s(Y), ε_0 = log s(F_q).
    소성 증분 a = ‖ε_tr − ε_1‖ (등방 반환 사상이라 X·Y 가 같은 주축 -> 정확한 주값 차), 탄성 증분 b = ‖ε_1 − ε_0‖ (주값 거리),
    항복 몫 η_q = a / (a + b)  (0 = 탄성 걸음, 1 = 증분이 모두 소성 = 흐름; 정의상 [0,1], 클램프 없음).
    커밋이 F_q 를 안 바꾼 점 (반환 없음) 은 a = 0 그대로 (Y = X 로 둠 -- 반올림 잡음 없이 η 0). a + b = 0 (변형 없음) 이면 η 0. 비유한 값이면 η 0 (강성 유지).
    Y 는 커밋 뒤 F_q' 에서 Y = X F_q⁻¹ F_q' (두 커밋 커널 모두 F_q' = F_q X⁻¹ Y 로 씀).
  칸 η_c = Σ_q V_q η_q / Σ_q V_q (칸 8 가우스 점, 부피 가중; 부피 0 이면 0) -> 입자 η_p = 그 칸 η_c (걸음 끝, G2P 앞의 칸 번호).
  다음 프레임 (칸 번호가 바뀌므로 입자가 나름): 칸 η̄_c = Σ_p vol_p η_p / Σ_p vol_p (rt_gauss 뒤, ghost6 면 표지 다음).
  면 F (칸 a 와 +축 이웃 b) 의 강성 가중 w_F = (1 − η̄_a)(1 − η̄_b)  (이웃 없으면 1 − η̄_a) -> 강성 벌점 k 만 k w_F (힘·헤시안 곱·블록 대각·에너지·선탐색 몫).
    고스트 질량 k_M 은 그대로 (절 CM70·CM77 (K) 와 같음). 첫 프레임은 η_p = 0 (w 1 = P1X 그대로).
  탄성 재질 (레고·ficus) 은 반환 사상이 없어 η ≡ 0 -> 설치 안 함 (같은 경로). 재질 판 (새 실행기 MatRT: wolf 모래·bread) 만. 옛 실행기 (수박) 는 아직 없음 (켜면 멈춤).
훅: 실행기 runpy 순간 (다른 감싸개 뒤, 가장 바깥):
  (1) ART.build_src·AORT._src 결과에 gq_force·gq_mv·gq_ls·gq_lsfin·gq_phi 의 복사본 *_w7 (강성 몫에만 kw7[c*3+a]) + 항복 몫 커널 w7_* 를 덧붙임.
  (2) RTSolver.k 를 감쌈: ao_commit·aq_commit 앞 w7_save (F_q 사본), 뒤 w7_eta (점 η) · w7_cell (칸 η) · w7_p (입자 η); rt_gauss 뒤 w7_z · w7_pc · w7_fw (면 가중);
      s.fn[gq_*] 를 '같은 인자 + kw7 (+ gq_mv 는 k)' 로 *_w7 을 띄우는 호출체로 둠 (ghost6 가 안쪽 _ok 로 띄워도 s.fn 을 거침; 모듈 바뀌면 다시).
  (3) 입자 η 배열 g7eta 를 재질 판 입자 백업 목록 (extra_p) 에 넣음 (데움 실행·넘침 다시 하기 뒤 되돌림).
기록: w7info.json (설치·고친 커널·호출체 수) + w7.jsonl (프레임 10·20·30·40·49 뒤: 입자 부피 가중 η 평균, η > ½ 부피 몫 (기록용 셈), 면 가중 평균·최소).
"""
import json
import os

import numpy as np
import torch

ST = dict(Y=0, out=".", nsrc=0, nw7=0, nlaunch={}, fnset=0, miss=[], said=False, rec=[], nalloc=0)

W7_K = r'''
// ---- 절 CM79 wolfh7: 항복 몫 η (lib/anchorflow/fem_wolfh7.py) ----
__global__ void w7_save(const int* IP, const float* Fq, float* Fo) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || q >= IP[12]) return;
    for (int k=0;k<9;k++) Fo[(size_t)q*9+k] = Fq[(size_t)q*9+k];
}
__device__ __forceinline__ void w7_ls(const float* M, float* e) {
    float U_[9], s_[3], V_[9]; ssvdf(M, U_, s_, V_);
    for (int i=0;i<3;i++) e[i] = logf(fabsf(s_[i]));
}
__global__ void w7_eta(const int* IP, const int* cb, const float* BG, const float* Fo, const float* Fq, const float* U, float* eq) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || q >= IP[12]) return;
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fo[(size_t)q*9+k];
    float A[9] = {0,0,0,0,0,0,0,0,0};
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; int id = ids[o]; for (int a=0;a<3;a++) { float ua = U[id*3+a]; A[a*3] += ua*bg[0]; A[a*3+1] += ua*bg[1]; A[a*3+2] += ua*bg[2]; } }
    float X[9]; for (int a=0;a<3;a++) for (int b=0;b<3;b++) X[a*3+b] = F[a*3+b] + A[a*3]*F[b] + A[a*3+1]*F[3+b] + A[a*3+2]*F[6+b];
    bool ch = false; for (int k=0;k<9;k++) if (Fq[(size_t)q*9+k] != F[k]) ch = true;
    float Y[9];
    if (ch) { float Fi[9], T[9], F1[9]; for (int k=0;k<9;k++) F1[k] = Fq[(size_t)q*9+k]; inv3(F, Fi); ao_mm3(Fi, F1, T); ao_mm3(X, T, Y); }
    else for (int k=0;k<9;k++) Y[k] = X[k];
    float et[3], e1[3], e0[3]; w7_ls(X, et); w7_ls(Y, e1); w7_ls(F, e0);
    float a2 = 0.f, b2 = 0.f;
    if (ch) for (int i=0;i<3;i++) { float d = et[i] - e1[i]; a2 += d*d; }
    for (int i=0;i<3;i++) { float d = e1[i] - e0[i]; b2 += d*d; }
    float a = sqrtf(a2), b = sqrtf(b2);
    float eta = (a + b > 0.f) ? a/(a + b) : 0.f;
    if (!isfinite(eta)) eta = 0.f;
    eq[q] = eta;
}
__global__ void w7_cell(const int* IP, const float* Vq, const float* eq, float* ec) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    float sv = 0.f, se = 0.f;
    for (int o=0;o<8;o++) { float v = Vq[c*8+o]; sv += v; se += v*eq[c*8+o]; }
    ec[c] = (sv > 0.f) ? se/sv : 0.f;
}
__global__ void w7_p(int N, const int* IP, const int* pcd, const int* cidx, const float* ec, float* etap) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    if (c < 0) return;
    etap[p] = ec[c];
}
__global__ void w7_z(const int* IP, float* acc) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    acc[c*2] = 0.f; acc[c*2+1] = 0.f;
}
__global__ void w7_pc(int N, const int* IP, const int* pcd, const int* cidx, const float* vol, const float* etap, float* acc) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    if (c < 0) return;
    float v = vol[p];
    atomicAdd(&acc[c*2], v*etap[p]); atomicAdd(&acc[c*2+1], v);
}
__global__ void w7_fw(const int* IP, const int* cb, const int* cidx, const float* acc, float* kw) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n[3] = {IP[6]-2, IP[7]-2, IP[8]-2};
    float ea = (acc[c*2+1] > 0.f) ? acc[c*2]/acc[c*2+1] : 0.f;
    for (int a=0;a<3;a++) {
        int x[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x[a] += 1;
        int c2 = (x[a] < n[a]) ? cidx[(x[0]*n[1] + x[1])*n[2] + x[2]] : -1;
        float eb = (c2 >= 0 && acc[c2*2+1] > 0.f) ? acc[c2*2]/acc[c2*2+1] : 0.f;
        kw[c*3+a] = (1.f - ea)*(1.f - eb);
    }
}
'''

WF = "float wf = fw ? fw[c*3+a] : 1.f;"
WFD = "double wf = fw ? (double)fw[c*3+a] : 1.0;"


def _span(src, head):
    i = src.index(head)
    j = src.index("\n}\n", i) + 3
    return src[i:j]


def _sig(k, nm, extra):
    k = k.replace(nm + "(", nm + "_w7(", 1)
    i = k.index(") {")
    return k[:i] + extra + k[i:]


def _rep(k, o, n, cnt, tag):
    assert k.count(o) == cnt, ("wolfh7 글자", tag, o, k.count(o))
    return k.replace(o, n)


def _force_like(k, tag):
    k = _rep(k, WF, WF + " float wk7 = kw7 ? kw7[c*3+a] : 1.f;", 2, tag)
    k = _rep(k, "Dz[i*3+j] = wf*(k*Dy[i*3+j] + (hm ? cm*Dd[i*3+j] : 0.f));", "Dz[i*3+j] = wf*(k*wk7*Dy[i*3+j] + (hm ? cm*Dd[i*3+j] : 0.f));", 1, tag)
    k = _rep(k, "e += 0.5*(double)k*(double)wf*(double)s;", "e += 0.5*(double)k*(double)wk7*(double)wf*(double)s;", 1, tag)
    k = _rep(k, "float mm = kk*wf*GQ_M[i*4]*GQ_M[j*4];", "float mm = (k*wk7 + (hm ? cm : 0.f))*wf*GQ_M[i*4]*GQ_M[j*4];", 1, tag)
    return k


def w7_src(src):
    if "gq_force(" not in src or "gq_force_w7(" in src:
        return src
    add = []
    k = _span(src, "__global__ void __launch_bounds__(128) gq_force(")
    add.append(_force_like(_sig(k, "gq_force", ", const float* kw7"), "gq_force"))
    k = _span(src, "__global__ void __launch_bounds__(128) gq_phi(")
    add.append(_force_like(_sig(k, "gq_phi", ", const float* kw7"), "gq_phi"))
    k = _sig(_span(src, "__global__ void __launch_bounds__(128) gq_mv("), "gq_mv", ", const float* kw7, float k7")
    k = _rep(k, "float v = kk*wf*MD[i*3+j];", "float v = (kk + k7*((kw7 ? kw7[c*3+a] : 1.f) - 1.f))*wf*MD[i*3+j];", 1, "gq_mv")
    add.append(k)
    k = _sig(_span(src, "__global__ void __launch_bounds__(128) gq_ls("), "gq_ls", ", const float* kw7")
    k = _rep(k, WFD, WFD + " double wk7 = kw7 ? (double)kw7[c*3+a] : 1.0;", 1, "gq_ls")
    k = _rep(k, "double qa = 0.0, qab = 0.0, qb = 0.0, qe = 0.0, qeb = 0.0;", "double qa = 0.0, qab = 0.0, qb = 0.0, qe = 0.0, qeb = 0.0, qbk7 = 0.0;", 1, "gq_ls")
    k = _rep(k, "qa += wf*sa; qab += wf*sab; qb += wf*sb; qe += wf*se; qeb += wf*seb;", "qa += wf*wk7*sa; qab += wf*wk7*sab; qb += wf*sb; qbk7 += wf*wk7*sb; qe += wf*se; qeb += wf*seb;", 1, "gq_ls")
    k = _rep(k, "blocksum_add(qeb, &GQ[4]);", "blocksum_add(qeb, &GQ[4]); blocksum_add(qbk7, &GQ[5]);", 1, "gq_ls")
    add.append(k)
    k = _sig(_span(src, "__global__ void gq_lsfin("), "gq_lsfin", "")
    k = _rep(k, "0.5*(double)k*(GQ[0] + 2.0*cd*GQ[1] + cd*cd*GQ[2])", "0.5*(double)k*(GQ[0] + 2.0*cd*GQ[1] + cd*cd*GQ[5])", 1, "gq_lsfin")
    k = _rep(k, "GQ[4] = 0.0;", "GQ[4] = 0.0; GQ[5] = 0.0;", 1, "gq_lsfin")
    add.append(k)
    if "aq_commit(" in src and "ssvdf(" in src and "ao_mm3(" in src:
        add.append(W7_K)
    ST["nsrc"] += 1
    ST["nw7"] += 1 if "aq_commit(" in src else 0
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + "\n".join(add) + "\n}\n"


def P(t):
    return np.uint64(0 if t is None else t.data_ptr())


EXTRA = {"gq_force": lambda s: (P(s.g7kw),), "gq_phi": lambda s: (P(s.g7kw),), "gq_ls": lambda s: (P(s.g7kw),),
         "gq_mv": lambda s: (P(s.g7kw), np.float32(float(s._gpk))), "gq_lsfin": lambda s: ()}


class _Launch:
    def __init__(s_, s, nm):
        s_.s, s_.mod, s_.nm, s_.f = s, s.mod, nm, s.mod.get_function(nm + "_w7")

    def __call__(s_, grid, block, args):
        ST["nlaunch"][s_.nm] = ST["nlaunch"].get(s_.nm, 0) + 1
        return s_.f(grid, block, tuple(args) + EXTRA[s_.nm](s_.s))


def _ensure_fn(s):
    fn = s.fn
    for nm in EXTRA:
        cur = dict.get(fn, nm)
        if isinstance(cur, _Launch) and cur.mod is s.mod and cur.s is s:
            continue
        try:
            fn[nm] = _Launch(s, nm)
            ST["fnset"] += 1
        except Exception as e_:
            ST["miss"].append(f"{nm}: {type(e_).__name__} {str(e_)[:120]}")


def _on(s):
    return bool(getattr(s, "_g7on", False))


def _alloc(s):
    cq = int(s.capQc)
    if getattr(s, "g7kw", None) is None or s.g7kw.numel() < cq * 3:
        s.g7kw = torch.ones(cq * 3, device=s.dev, dtype=torch.float32)
        s.g7acc = torch.zeros(cq * 2, device=s.dev, dtype=torch.float32)
        s.g7ec = torch.zeros(cq, device=s.dev, dtype=torch.float32)
        s.g7eq = torch.zeros(cq * 8, device=s.dev, dtype=torch.float32)
        s.g7Fo = torch.zeros(cq * 8 * 9, device=s.dev, dtype=torch.float32)
        ST["nalloc"] += 1


def install(out, Y=0):
    ST.update(Y=int(Y), out=out)
    if not ST["Y"]:
        return
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_mat_rt as FMR
    o_b, o_s = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: w7_src(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: w7_src(o_s(*a, **kw))

    o_init = FMR.MatRT.__init__

    def __init__(s, *a, **kw):
        o_init(s, *a, **kw)
        s._g7on = True
        s.g7eta = torch.zeros(s.N, device=s.dev, dtype=torch.float32)
        s.extra_p = tuple(s.extra_p) + ("g7eta",)
        print(f"[절 CM79 wolfh7] (Y7) 재질 {s._mat} 판에 설치 -- 입자 η {s.N} 개 (백업 목록에 넣음), P1X 강성 k 는 면마다 w_F = (1 − η̄_a)(1 − η̄_b) 배 (고스트 질량 그대로)", flush=True)
        _dump()
    FMR.MatRT.__init__ = __init__

    o_asolve = getattr(AORT, "AORTSolver", None)
    if o_asolve is not None:
        o_ainit = o_asolve.__init__

        def __ainit__(s, *a, **kw):
            raise NotImplementedError("wolfh7 (Y7): 옛 실행기 (수박) 는 아직 없음")
        o_asolve.__init__ = __ainit__

    ok = ART.RTSolver.k
    i32 = np.int32

    def k(s, name, n, args, b=128):
        if not _on(s):
            return ok(s, name, n, args, b)
        if name in ("ao_commit", "aq_commit"):
            _alloc(s)
            capQ = int(s.capQc) * 8
            ok(s, "w7_save", capQ, (P(s.IP), args[3], P(s.g7Fo)), 128)
            r = ok(s, name, n, args, b)
            ok(s, "w7_eta", capQ, (P(s.IP), args[1], args[2], P(s.g7Fo), args[3], args[4], P(s.g7eq)), 128)
            ok(s, "w7_cell", int(s.capQc), (P(s.IP), P(s.Vq), P(s.g7eq), P(s.g7ec)), 128)
            ok(s, "w7_p", int(s.N), (i32(int(s.N)), P(s.IP), P(s.pcd), P(s.cidx), P(s.g7ec), P(s.g7eta)), 128)
            return r
        if name == "rt_gauss":
            _alloc(s)
            _ensure_fn(s)
            r = ok(s, name, n, args, b)
            cq = int(s.capQc)
            ok(s, "w7_z", cq, (P(s.IP), P(s.g7acc)), 128)
            ok(s, "w7_pc", int(s.N), (i32(int(s.N)), P(s.IP), P(s.pcd), P(s.cidx), P(s.vol), P(s.g7eta), P(s.g7acc)), 128)
            ok(s, "w7_fw", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g7acc), P(s.g7kw)), 128)
            if not ST["said"]:
                ST["said"] = True
                print("[절 CM79 wolfh7] (Y7) 훅: 커밋 앞뒤 w7_save·w7_eta·w7_cell·w7_p, rt_gauss 뒤 w7_z·w7_pc·w7_fw, s.fn[gq_*] -> *_w7", flush=True)
                _dump()
            return r
        return ok(s, name, n, args, b)
    ART.RTSolver.k = k

    o_step = ART.RTSolver.step

    def step(s, t, *a, **kw):
        r = o_step(s, t, *a, **kw)
        if _on(s) and getattr(s, "g7kw", None) is not None:
            fr = int(round(t / s.h))
            if fr in (10, 20, 30, 40, 49):
                try:
                    _stat(s, fr)
                except Exception as e_:
                    ST["miss"].append(f"stat {fr}: {type(e_).__name__} {str(e_)[:120]}")
        return r
    ART.RTSolver.step = step
    print("[절 CM79] wolfh7 켬 -- (Y7) P1X 강성 × 면 항복 가중 (1 − η̄_a)(1 − η̄_b), η = 소성 증분 / (소성 + 탄성 증분)", flush=True)
    import atexit
    atexit.register(_dump)


def _stat(s, fr):
    torch.cuda.synchronize()
    N = int(s.N)
    e = s.g7eta[:N].double()
    v = s.vol[:N].double()
    IP = s.IP.cpu().tolist()
    nc = int(IP[11])
    fm = getattr(s, "g3fm", None)
    d = dict(f=fr, eta_mean=float((e * v).sum() / v.sum()), eta_gt_half_vol=float((v * (e > 0.5)).sum() / v.sum()), eta_max=float(e.max()))
    if fm is not None and nc > 0:
        m = fm[:nc].long()
        kw = s.g7kw[:nc * 3].view(nc, 3).double()
        sel = torch.stack([((m >> a) & 1).bool() for a in range(3)], 1)
        if bool(sel.any()):
            w = kw[sel]
            d.update(nface=int(sel.sum()), w_mean=float(w.mean()), w_min=float(w.min()), w_lt_half=float((w < 0.5).double().mean()))
    ST["rec"].append(d)
    with open(os.path.join(ST["out"], "w7.jsonl"), "a") as f_:
        f_.write(json.dumps(d) + "\n")


def _dump():
    try:
        d = {k: v for k, v in ST.items() if k not in ("rec",)}
        with open(os.path.join(ST["out"], "w7info.json"), "w") as f_:
            json.dump(d, f_, ensure_ascii=False, default=str)
    except Exception:
        pass
