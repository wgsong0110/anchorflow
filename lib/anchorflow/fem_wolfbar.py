"""절 CM107 (사용자 11-11): wolf 육면체 판 부피 붕괴·덜 풀린 걸음이 바닥 장벽 걸린 칸에서 시작하는가 -- 기록 (결과 불변) + 진단 판 장벽 강성 배수.

기록 (dg 1, 결과 불변: 풀이 버퍼는 읽기만, 이 모듈 버퍼에만 씀; 별도 cupy 모듈, 같은 스트림·그래프 안):
- 걸음마다 (ao_commit 앞·뒤) 가우스 점 q 마다: J0 = det F_q (커밋 앞), JX = det(F_q + Σ U ⊗ g) (풀이 걸음 끝 시험 F, 절 CM92 J_qX),
  커밋 뒤 det F_q' -> JY = det F_q' · JX / J0 (반환 사상 뒤, 절 CM92 J_qE), 점 높이 d = (x_q0 + Σ BW U)·n − p (걸음 끝),
  장벽 힘 |b'(d)| = κ |2 (d − d̂) ln(d/d̂) + (d − d̂)²/d| (κ = ks V_q, 표본점 위치 = 장벽 판과 같은 bx 질량 중심; 활성 bact 이고 0 < d < d̂, d ≤ 0 이면 −1 표시),
  V_q, m_q, 표식 (bact | bcf << 1; bcf bit 0 = 이 프레임 어느 뉴턴 반복에서든 장벽이 걸림 (--bar_cnt)).
- 걸음 시작 장벽 실현 가능 보폭 (rt_bar_kfe, y = ŷ 쪽) 과 뉴턴 반복 보폭 자름 (rt_bar_kfe, y -> yc) 을 점마다 다시 계산한 k_q (2^-k) 의 걸음 최대.
- 뉴턴 반복 시작 잔차 r0 (rt_grad 의 g0buf, cj_g 때 사본 -> 걸음 끝에는 마지막 반복 = fem.jsonl uf.g2 와 같은 반복), 시작점 섞기 노드 k (y0loc ykn, cj_init 때 사본).
- 호스트 (s.step 뒤): 칸 = 이차 B-스플라인 칸 (8 팔분 점); 장벽 활성 칸 = 8 점 중 하나라도 bcf bit 0. 점 무리 A (활성 칸) · N (나머지 점유 칸),
  노드 무리 = 활성 칸 27 노드 (A) · 나머지 점유 칸 노드 (N). 무리별 부피·J 변화·뒤집힌 점·장벽 힘·보폭 자름 점·잔차 몫·시작점 β<1 노드, 높이 띠별 부피 변화 -> wb.jsonl.
진단 판 (결과 바꿈, 기본 판에 안 넣음): ks = 장벽 강성 κ 배수 (FL["KS"] × ks, 첫 걸음 캡처 전). d̂ 는 실행기 --fl_dhm (절 CM6), 단순 미끄럼 바닥은 실행기 --fl_nclamp (절 CM16).
"""
import json
import math
import os

import numpy as np
import torch

INFO = dict(dg=0, ks=1.0, ks0=None, ks1=None, n_inst=0, n_alloc=0, n_pre=0, n_post=0, n_kq=0, n_kqs=0, n_r=0, n_yk=0, n_rec=0, n_line=0, err=None, names={})
OUT = ["."]
LOG = [None]
KS_MAX = 8
NB = [-1e9, 0.0, None, 0.02, 0.05, 0.10, 1e9]                                       # 높이 띠 (m): 바닥 아래 · [0, d̂) · [d̂, 2 cm) · [2, 5) · [5, 10) · ≥ 10 cm (None = d̂)

SRC = r'''
extern "C" {
__device__ __forceinline__ void qn(int q, const int* IP, const int* cb, int* ids) {
    int c = q >> 3; int nb1 = IP[7], nb2 = IP[8];
    for (int o=0;o<27;o++) { int i = o/9, j = (o/3)%3, k = o%3; ids[o] = ((cb[c*3]+i)*nb1 + (cb[c*3+1]+j))*nb2 + (cb[c*3+2]+k); }
}
__device__ __forceinline__ float det3(const float* X) {
    return X[0]*(X[4]*X[8]-X[5]*X[7]) - X[1]*(X[3]*X[8]-X[5]*X[6]) + X[2]*(X[3]*X[7]-X[4]*X[6]);
}
__global__ void wb_cpf(int n, const float* a, float* b) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) b[i] = a[i]; }
__global__ void wb_cpi(int n, const int* a, int* b) { int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) b[i] = a[i]; }
// 커밋 앞: o[q*10 + ...] = J0, JX, d, fb, V, m, 표식, (뒤) J1, d_b, 0
__global__ void wb_pre(const int* IP, const int* cb, const float* BG, const float* BW, const float* Fq, const float* Vq, const float* mq, const float* xq0, const float* U,
                       const float* bx, const int* bact, const int* bcf, float n0, float n1, float n2, float pn, float dh, float ks, float* o) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7; int ids[27]; qn(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    float xq[3] = {xq0[q*3], xq0[q*3+1], xq0[q*3+2]};
    float xb[3] = {0.f, 0.f, 0.f}; float bm = bx ? bx[q*4+3] : 0.f;
    if (bm > 0.f) for (int a=0;a<3;a++) xb[a] = bx[q*4+a]/bm;
    for (int o_=0;o_<27;o_++) { const float* bg = BG + (oc*27+o_)*3;
        float g[3] = {F[0]*bg[0] + F[3]*bg[1] + F[6]*bg[2], F[1]*bg[0] + F[4]*bg[1] + F[7]*bg[2], F[2]*bg[0] + F[5]*bg[1] + F[8]*bg[2]};
        int id = ids[o_]; float w = BW[oc*27+o_];
        for (int a=0;a<3;a++) { float ua = U[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; xq[a] += w*ua; xb[a] += w*ua; } }
    float d = xq[0]*n0 + xq[1]*n1 + xq[2]*n2 - pn;
    float db = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn;
    int act = (bact && bm > 0.f) ? bact[q] : 0;
    float fb = 0.f;
    if (act && db < dh) {
        if (db > 0.f) { double kap = (double)ks*(double)Vq[q], dd = (double)db - dh, L = log((double)db/dh);
            fb = (float)fabs(kap*(2.0*dd*L + dd*dd/db)); }
        else fb = -1.f;
    }
    float* r = o + (size_t)q*10;
    r[0] = det3(F); r[1] = det3(X); r[2] = d; r[3] = fb; r[4] = Vq[q]; r[5] = mq[q];
    r[6] = (float)(act | ((bcf ? (bcf[q] & 3) : 0) << 1)); r[8] = (bm > 0.f) ? db : 0.f; r[9] = 0.f;
}
__global__ void wb_post(const int* IP, const float* Fq, float* o) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    o[(size_t)q*10+7] = det3(Fq + (size_t)q*9);
}
// rt_bar_kfe 와 같은 식을 점마다: y0 -> y1 의 β = 2^-k 에서 d(β) > 0 인 가장 작은 k -> kq[q] = max
__global__ void wb_kq(const int* IP, const int* cb, const float* BW, const float* y0, const float* y1, const float* U, float thinv,
                      float n0, float n1, float n2, float pn, const float* bx, const int* bact, int* kq) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = bx[q*4+3];
    if (!(bm > 0.f) || !bact[q]) return;
    int oc = q & 7; int ids[27]; qn(q, IP, cb, ids);
    float xa[3], xb[3];
    for (int a=0;a<3;a++) { xa[a] = bx[q*4+a]/bm; xb[a] = xa[a]; }
    for (int o=0;o<27;o++) { float w = BW[oc*27+o]; int id = ids[o];
        for (int a=0;a<3;a++) { xa[a] += w*(U[id*3+a] + thinv*(y0[id*3+a] - U[id*3+a])); xb[a] += w*(U[id*3+a] + thinv*(y1[id*3+a] - U[id*3+a])); } }
    float d0 = xa[0]*n0 + xa[1]*n1 + xa[2]*n2 - pn, d1 = xb[0]*n0 + xb[1]*n1 + xb[2]*n2 - pn;
    if (d1 > 0.f) return;
    int k = 0; float c = 1.f;
    while (true) { c *= 0.5f; k++; if (c == 0.f) break; if (d0 + c*(d1 - d0) > 0.f) break; }
    if (k > kq[q]) kq[q] = k;
}
}
'''


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _dump():
    try:
        with open(os.path.join(OUT[0], "wbinfo.json"), "w") as f_:
            json.dump(INFO, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


class Buf:
    def __init__(s):
        import cupy as cp
        s.mod = cp.RawModule(code=SRC)
        s.fn = {n: s.mod.get_function(n) for n in ("wb_cpf", "wb_cpi", "wb_pre", "wb_post", "wb_kq")}
        s.capQ = s.capN = 0
        s.cur = -1

    def alloc(s, sv):
        capQ, capN = int(sv.capQc) * 8, int(sv.capN)
        if capQ == s.capQ and capN == s.capN:
            return
        dev = sv.dev
        s.capQ, s.capN = capQ, capN
        s.q = torch.zeros(KS_MAX, capQ, 10, device=dev, dtype=torch.float32)
        s.kq = torch.zeros(KS_MAX, capQ, device=dev, dtype=torch.int32)
        s.kqs = torch.zeros(KS_MAX, capQ, device=dev, dtype=torch.int32)
        s.r = torch.zeros(KS_MAX, capN, 3, device=dev, dtype=torch.float32)
        s.yk = torch.zeros(KS_MAX, capN, device=dev, dtype=torch.int32)
        s.has = torch.zeros(KS_MAX, 4, dtype=torch.int32)                                # 호스트: 슬롯별 (q·r·yk·kq) 기록 여부 (캡처 때 정함)
        INFO["n_alloc"] += 1

    def go(s, name, n, args, b=128):
        from . import fem_rt_cuda as RTC
        with RTC._Stream():
            s.fn[name](((int(n) + b - 1) // b,), (b,), tuple(args))


def _fl(sv):
    FL = sv.floor or {}
    nr = FL.get("NR", [0.0, 0.0, 1.0])
    return [float(q) for q in nr], float(FL.get("PN", 0.0)), float(FL.get("DH", 0.0)), float(FL.get("KS", 0.0))


def _nm(name):
    INFO["names"][name] = INFO["names"].get(name, 0) + 1


def _wrap(R):
    ok, os_, oa, orc = R.k, R.step, R.alloc, R.record

    def alloc(s, need):
        r = oa(s, need)
        if INFO["dg"]:
            if getattr(s, "_wb", None) is None:
                s._wb = Buf(); INFO["n_inst"] += 1; INFO["cls"] = type(s).__name__
            s._wb.alloc(s)
        return r

    def record(s, *a, **kw):
        if getattr(s, "_wb", None) is not None:
            s._wb.cur = -1
        return orc(s, *a, **kw)

    def k(s, name, n, args, b=128):
        W = getattr(s, "_wb", None)
        if W is None or not W.capQ:
            return ok(s, name, n, args, b)
        f32, i32 = np.float32, np.int32
        try:
            if name == "cj_init":
                W.cur = int(args[-1])
                if W.cur < KS_MAX and getattr(s, "ykn", None) is not None and getattr(s, "y0loc", 0):
                    W.go("wb_cpi", W.capN, (i32(W.capN), P(s.ykn), P(W.yk[W.cur])), 256); W.has[W.cur, 2] = 1; INFO["n_yk"] += 1; _nm(name)
            elif name == "ao_commit" and 0 <= W.cur < KS_MAX:
                nr, pn, dh, ks = _fl(s)
                bx = getattr(s, "bx", None) if getattr(s, "bar", False) else None
                W.go("wb_pre", W.capQ, (P(s.IP), P(s.cb), P(s.BG), P(s.BW), P(s.Fq), P(s.Vq), P(s.mq), P(s.xq0), args[4], P(bx), P(getattr(s, "bact", None) if bx is not None else None),
                                        P(getattr(s, "bcf", None) if (bx is not None and getattr(s, "bcnt", False)) else None), *[f32(q) for q in nr], f32(pn), f32(dh), f32(ks), P(W.q[W.cur])))
                INFO["n_pre"] += 1; _nm(name)
        except Exception as e:
            INFO["err"] = "k_pre " + name + ": " + repr(e)[:300]
        r = ok(s, name, n, args, b)
        try:
            if name == "ao_commit" and 0 <= W.cur < KS_MAX:
                W.go("wb_post", W.capQ, (P(s.IP), P(s.Fq), P(W.q[W.cur])))
                W.has[W.cur, 0] = 1; INFO["n_post"] += 1
            elif name == "cj_g":
                kk = int(args[-1])
                if kk < KS_MAX:
                    W.go("wb_cpf", W.capN * 3, (i32(W.capN * 3), P(s.g0buf), P(W.r[kk])), 256); W.has[kk, 1] = 1; INFO["n_r"] += 1; _nm(name)
            elif name == "rt_bar_kfe" and getattr(s, "bar", False):
                y1 = int(args[4])
                if getattr(s, "yc", None) is not None and y1 == s.yc.data_ptr() and 0 <= W.cur < KS_MAX:
                    W.go("wb_kq", W.capQ, tuple(args[:11]) + (P(s.bx), P(s.bact), P(W.kq[W.cur]))); W.has[W.cur, 3] = 1; INFO["n_kq"] += 1; _nm(name + ":yc")
                elif getattr(s, "ySv", None) is not None and y1 == s.ySv.data_ptr() and W.cur + 1 < KS_MAX:
                    W.go("wb_kq", W.capQ, tuple(args[:11]) + (P(s.bx), P(s.bact), P(W.kqs[W.cur + 1]))); INFO["n_kqs"] += 1; _nm(name + ":ySv")
                else:
                    _nm(name + ":other")
        except Exception as e:
            INFO["err"] = "k_post " + name + ": " + repr(e)[:300]
        return r

    def step(s, t, *a, **kw):
        if INFO["ks"] != 1.0 and not getattr(s, "_wb_ks", False) and isinstance(getattr(s, "floor", None), dict) and "KS" in s.floor:
            s._wb_ks = True
            INFO["ks0"] = float(s.floor["KS"]); s.floor["KS"] = float(s.floor["KS"]) * INFO["ks"]; INFO["ks1"] = float(s.floor["KS"])
            if "KSN" in s.floor:
                s.floor["KSN"] = float(s.floor["KSN"]) * INFO["ks"]
            print(f"[절 CM107 wolfbar] 장벽 강성 κ x {INFO['ks']}: KS {INFO['ks0']:.6g} -> {INFO['ks1']:.6g}", flush=True)
            _dump()
        r = os_(s, t, *a, **kw)
        W = getattr(s, "_wb", None)
        if W is not None and W.capQ and LOG[0] is not None:
            try:
                torch.cuda.synchronize()
                o = _analyze(s, W, t)
                LOG[0].write(json.dumps(o) + "\n"); LOG[0].flush(); INFO["n_line"] += 1
            except Exception as e:
                INFO["err"] = "analyze: " + repr(e)[:300]
            W.kq.zero_(); W.kqs.zero_(); W.q[..., 6] = 0                                   # 다음 프레임 (그래프 재생이 다시 씀)
            INFO["n_rec"] += 1
            if INFO["n_rec"] in (1, 2, 5) or INFO["n_rec"] % 25 == 0:
                _dump()
        return r

    for f_ in (alloc, record, k, step):
        f_._wb = True
    R.alloc, R.record, R.k, R.step = alloc, record, k, step


def _grp(m, J0, JX, JY, fb, V, mq, kq, kqs):
    n = int(m.sum())
    if n == 0:
        return dict(n=0)
    v = V[m]; j0, jx, jy = J0[m], JX[m], JY[m]
    f = fb[m]
    o = dict(n=n, V=float(v.sum()), VJ0=float((v * j0).sum()), dVs=float((v * (jx - j0)).sum()), dVr=float((v * (jy - jx)).sum()),
             inv=int((jx <= 0).sum()), inv0=int((j0 <= 0).sum()), c09=int((jx < 0.9 * j0).sum()), W=float(mq[m].sum() * 9.8),
             fb=float(f[f > 0].sum()), fbmax=float(f.max()) if n else 0.0, nfb=int((f > 0).sum()), nfbin=int((f < 0).sum()),
             nkq=int((kq[m] > 0).sum()), kqmax=int(kq[m].max()), nkqs=int((kqs[m] > 0).sum()), kqsmax=int(kqs[m].max()))
    return o


def _analyze(s, W, t):
    IP = s.IP.detach().cpu().numpy().astype(np.int64)
    nq, nn, nb1, nb2 = int(IP[12]), int(IP[9]), int(IP[7]), int(IP[8])
    K = int(getattr(s, "nstage", s.K))
    nc = nq // 8
    dev = s.Fq.device
    cb = s.cb.reshape(-1)[:nc * 3].view(nc, 3).long()                                   # s.cb 는 [capC, 3]
    o3 = torch.arange(27, device=dev); oi, oj, ok_ = o3 // 9, (o3 // 3) % 3, o3 % 3
    nid = ((cb[:, 0:1] + oi) * nb1 + (cb[:, 1:2] + oj)) * nb2 + (cb[:, 2:3] + ok_)               # [nc, 27]
    nid = nid.clamp(0, max(nn - 1, 0))
    mn = s.mn[:nn].double()
    cjg = s.cjg[:K].cpu().tolist() if getattr(s, "cjg", None) is not None else [None] * K
    cjl = s.cjl[:K].cpu().tolist() if getattr(s, "cjl", None) is not None else [None] * K
    cjs = s.cjs[:K].cpu().tolist() if getattr(s, "cjs", None) is not None else [None] * K
    nr, pn, dh, ks = _fl(s)
    edges = [dh if q is None else q for q in NB]
    steps = []
    for k in range(min(K, KS_MAX)):
        h = W.has[k].tolist()
        st = dict(k=k, has=h, g2=cjg[k], fail=cjl[k], smin=cjs[k])
        if h[0]:
            B = W.q[k, :nq].double()
            J0, JX, d, fb, V, mq, fl, J1, db = B[:, 0], B[:, 1], B[:, 2], B[:, 3], B[:, 4], B[:, 5], B[:, 6].long(), B[:, 7], B[:, 8]
            JY = torch.where(J0 != 0, J1 * JX / J0, JX)
            val = V > 0
            bcf1 = ((fl >> 1) & 1) > 0
            cellA = (bcf1 & val).view(nc, 8).any(1)
            cellO = val.view(nc, 8).any(1)
            pA = cellA.repeat_interleave(8)
            kq = W.kq[k, :nq].long(); kqs = W.kqs[k, :nq].long()
            A, N = val & pA, val & ~pA
            st["A"] = _grp(A, J0, JX, JY, fb, V, mq, kq, kqs)
            st["N"] = _grp(N, J0, JX, JY, fb, V, mq, kq, kqs)
            st["ncell"] = [int(cellA.sum()), int((cellO & ~cellA).sum())]
            st["nbact"] = int(((fl & 1) > 0).sum()); st["nbcf"] = int((bcf1 & val).sum())
            bands = []
            for i in range(len(edges) - 1):
                mb = val & (d >= edges[i]) & (d < edges[i + 1])
                bands.append([int(mb.sum()), float(V[mb].sum()), float((V[mb] * J0[mb]).sum()), float((V[mb] * (JX[mb] - J0[mb])).sum()), float((V[mb] * (JY[mb] - JX[mb])).sum()),
                              int((mb & (JX <= 0)).sum()), int((mb & pA).sum())])
            st["bands"] = bands
            # 노드 무리
            nA = torch.zeros(nn, dtype=torch.bool, device=dev); nO = torch.zeros(nn, dtype=torch.bool, device=dev)
            if nn:
                nA[nid[cellA].reshape(-1)] = True; nO[nid[cellO].reshape(-1)] = True
            nN = nO & ~nA
            st["nnode"] = [int(nA.sum()), int(nN.sum())]
            st["mnode"] = [float(mn[nA].sum()), float(mn[nN].sum())]
            if h[1]:
                g2 = (W.r[k, :nn].double() ** 2).sum(-1)
                st["r2"] = [float(g2[nA].sum()), float(g2[nN].sum()), float(g2[nO].sum())]
            if h[2]:
                yk = W.yk[k, :nn].long()
                bet = torch.pow(2.0, -yk.double().clamp(max=200))
                st["yk"] = [int(((yk > 0) & nA).sum()), int(((yk > 0) & nN).sum()), float((mn * bet)[nA].sum() / mn[nA].sum().clamp_min(1e-30)),
                            float((mn * bet)[nN].sum() / mn[nN].sum().clamp_min(1e-30)), int(yk[nO].max()) if bool(nO.any()) else 0,
                            float(mn[nA & (yk > 0)].sum()), float(mn[nN & (yk > 0)].sum())]
        steps.append(st)
    o = dict(t=float(t), fr=int(round(float(t) / float(s.h))), K=K, nq=nq, nn=nn, steps=steps)
    try:
        o["bk"] = s.BSK[:K * s.nt, 0].cpu().tolist(); o["y0k"] = s.Y0K[:K, 0].cpu().tolist()
    except Exception:
        pass
    return o


def install(out, dg=1, ks=1.0):
    from . import fem_an_rt as ART
    OUT[0] = out
    INFO.update(dg=int(dg), ks=float(ks))
    if INFO["dg"]:
        LOG[0] = open(os.path.join(out, "wb.jsonl"), "w")
    _wrap(ART.RTSolver)
    print(f"[절 CM107] wolfbar 켬 -- 기록 {INFO['dg']}, 장벽 강성 배수 {INFO['ks']}", flush=True)
    _dump()
