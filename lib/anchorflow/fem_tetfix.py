"""절 CM178 (사용자 11-15 지시): 사면체 TC 판 레고 발산 -- 처음 뒤집히는 걸음 추적 (결과 불변 기록) + 원리적 수정 새 플래그 (기본 0).

기록 `--tfx 1` (tetfix.jsonl; 프레임 창 --tfw f0:f1 안의 프레임마다, 한 프레임 = 한 걸음):
  실행기 걸음 안 버퍼를 절 CM26 진단 훅 (fem_wolfprop_diag, PRD) 의 사본 경로로 받음 (그래프 안 dcopy 만, 읽기는 프레임 뒤 호스트):
  뉴턴 반복 it 마다 y (α-수준 노드 변위), η, g0buf (rt_grad 잔차), f (요소 기울기, 장벽 전), bGn (장벽 노드 스칼라), 걸음 시작 U·Ut·mn·sk·pm·meff.
  각 사면체 (적분점 q, 꼭짓점 4) 에 대해:
    det X(y_it) (요소가 보는 모양), det X(끝 외삽 U0 + thinv (y_it - U0)) (프레임 끝 모양), det X(Ut) (관성 예측), det X(U) (걸음 끝)
    표본점 (장벽 판 팔분 점 = 사면체 입자 질량 중심) 의 바닥 거리 d_it (= 질량 중심 + ¼ Σ 꼭짓점 끝 외삽 변위) 와 자기 장벽 힘 b'(d)
    꼭짓점: 질량 m, 유효 질량 meff, 높이 z (바닥 평면 기준, 시작 · it 별 끝 외삽), 구속·고착 (sk, pm), 노드 장벽 스칼라 bGn_it, 잔차 |g0buf_it|, 요소 기울기 법선 f_it·n
  전체: 반복별 det ≤ 0 사면체 수·최소 det, 장벽 걸린 표본점 수, 접촉 띠 사면체 중 꼭짓점이 평면 아래인 비율 (질량 중심은 평면 위)
  결과는 <out>/tetfix.jsonl.

수정 플래그 `--tff 1` (아래 install_ff).
"""
import json
import math
import os

import numpy as np
import torch

from . import fem_wolfprop_diag as PRD
from .fem_tetc import _q_nodes

INFO = dict(mode=None, err=None, n_rec=0, ff=0, ff_mods=[])
OUT = ["."]
WIN = [0, -1]
WORST = 4


def _det_tets(D, nid, bg, ok, Fq, I3):
    """D [nN,3] 변위장 -> det((I + Σ D ⊗ ∇w) F) [Q]."""
    nidc = nid.clamp(0, D.shape[0] - 1)
    Gd = torch.einsum("qoa,qob->qab", D[nidc] * ok[..., None], bg)
    X = (I3 + Gd) @ Fq
    return torch.linalg.det(X.double()).float()


def _bprime(d, dh, kap):
    """장벽 기울기 b'(d) (0 < d < d̂), d ≤ 0 이면 inf."""
    dd = d - dh
    L = torch.log(d.clamp_min(1e-30) / dh)
    b = -kap * (2.0 * dd * L + dd * dd / d.clamp_min(1e-30))
    b = torch.where(d >= dh, torch.zeros_like(b), b)
    return torch.where(d <= 0, torch.full_like(b, float("inf")), b)


def trace(sv, fr):
    IP = sv.IP.detach().cpu().numpy().astype(np.int64)
    Q, nN, nb1, nb2 = int(IP[12]), int(IP[9]), int(IP[7]), int(IP[8])
    dev = sv.Vq.device
    if Q > 4_000_000 or not hasattr(sv, "PRn"):
        return dict(skip=Q, has_prn=hasattr(sv, "PRn"))
    nt = int(sv.nt)
    FL = sv.floor
    n = torch.tensor(FL["NR"], device=dev, dtype=torch.float32)
    pn, dh, ks = float(FL["PN"]), float(FL["DH"]), float(FL["KS"])
    sg = sv.stages(sv.K)[0]
    thinv = float(sg["thinv"])
    nid, bg = _q_nodes(sv, Q, nb1, nb2)
    ok = (bg.abs().sum(-1) > 0) & (nid >= 0) & (nid < nN)
    Vq = sv.Vq[:Q].float()
    Fq = sv.Fq[:Q].float().view(Q, 3, 3)
    I3 = torch.eye(3, device=dev)
    act = Vq > 0
    U0 = sv.PRs["U"][:nN].float()
    Ut = sv.PRs["Ut"][:nN].float()
    Uend = sv.U[:nN].float()
    ys = [sv.PRn["y"][i][:nN].float() for i in range(nt + 1)]
    # 노드 격자 위치 -> 바닥 높이
    idx = torch.arange(nN, device=dev)
    L = torch.stack([idx // (nb1 * nb2) + IP[0], (idx // nb2) % nb1 + IP[1], idx % nb2 + IP[2]], 1).double()
    x0 = (L @ sv.A.double().reshape(3, 3).T).float()
    z0 = x0 @ n - pn                                                                       # 노드 격자 높이 (변위 0)
    mn = sv.mn[:nN].float()
    meff = sv.PRs["meff"][:nN].float()
    sk = sv.PRs["sk"][:nN].float(); pm = sv.PRs["pm"][:nN].float()
    # 반복별 det
    dets_y, dets_e = [], []
    for i in range(nt + 1):
        dets_y.append(_det_tets(ys[i], nid, bg, ok, Fq, I3))
        dets_e.append(_det_tets(U0 + thinv * (ys[i] - U0), nid, bg, ok, Fq, I3))
    d_pred = _det_tets(Ut, nid, bg, ok, Fq, I3)
    d_end = _det_tets(Uend, nid, bg, ok, Fq, I3)
    # 표본점 거리·장벽 힘
    bx = sv.bx[:Q * 4].view(Q, 4)
    has = bx[:, 3] > 0
    xmc = bx[:, :3] / bx[:, 3:4].clamp_min(1e-30)
    BW = sv.BW.float().view(8, 27)[(torch.arange(Q, device=dev) & 7)]
    nidc = nid.clamp(0, nN - 1)
    bact = (sv.bact[:Q] > 0) if getattr(sv, "bact", None) is not None else torch.zeros(Q, dtype=torch.bool, device=dev)

    def d_of(Dend):
        dq = xmc + torch.einsum("qo,qoa->qa", BW * ok, Dend[nidc])
        return dq @ n - pn

    d_its = [d_of(U0 + thinv * (ys[i] - U0)) for i in range(nt + 1)]
    kap = ks * Vq
    glob = []
    for i in range(nt + 1):
        a = act & (dets_y[i] <= 0); b = act & (dets_e[i] <= 0)
        band = has & act & bact & (d_its[i] < dh)
        glob.append(dict(it=i, ninv_y=int(a.sum()), dmin_y=float(dets_y[i][act].min()), ninv_end=int(b.sum()), dmin_end=float(dets_e[i][act].min()),
                         nband=int(band.sum()), nband_neg=int((has & act & bact & (d_its[i] <= 0)).sum()),
                         bmax=(float(_bprime(d_its[i][band], dh, kap[band]).max()) if bool(band.any()) else 0.0)))
    # 접촉 띠 사면체 (끝 외삽 높이) 꼭짓점 평면 아래 비율
    ze = []
    for i in (0, nt):
        De = U0 + thinv * (ys[i] - U0)
        zv = z0 + (De @ n)
        zq = torch.where(ok, zv[nidc], torch.full((Q, 27), 1e30, device=dev))
        zmin = zq.min(1).values
        band = has & act & bact & (d_its[i] < dh)
        ze.append(dict(it=i, nband=int(band.sum()), nvert_below=int((band & (zmin < 0)).sum()), zmin_med=(float(zmin[band].median()) if bool(band.any()) else None),
                       zmin_min=(float(zmin[band].min()) if bool(band.any()) else None)))
    # 대표 사면체: 걸음 끝 det 작은 4 + 반복 중 최소 det 작은 4
    sel = []
    for src in (d_end, torch.stack([dets_e[i] for i in range(nt + 1)]).min(0).values):
        dd = torch.where(act, src, torch.full_like(src, 1e30))
        for q_ in torch.argsort(dd)[:WORST].tolist():
            if float(dd[q_]) <= 0.0 and q_ not in sel:
                sel.append(q_)
    sel = sel[:2 * WORST]
    d0 = torch.where(act, dets_y[0], torch.full_like(dets_y[0], 1e30))
    sel0 = [q_ for q_ in torch.argsort(d0)[:WORST].tolist() if float(d0[q_]) <= 0.0]                # 시작점 (뉴턴 반복 0 의 y) 이 이미 뒤집힌 사면체
    sel = sel + [q_ for q_ in sel0 if q_ not in sel]
    g0 = sv.PRn["g0buf"].float(); fe = sv.PRn["f"].float(); bgn = sv.PRn["bGn"].float()
    tets = []
    for q_ in sel:
        vn = nid[q_][ok[q_]]
        t = dict(q=int(q_), start_inv=(q_ in sel0), npart_samp=float(bx[q_, 3]), V=float(Vq[q_]), bact=bool(bact[q_]),
                 det_pred=float(d_pred[q_]), det_end=float(d_end[q_]), det_y=[float(dets_y[i][q_]) for i in range(nt + 1)], det_e=[float(dets_e[i][q_]) for i in range(nt + 1)],
                 d=[float(d_its[i][q_]) for i in range(nt + 1)], dh=dh, bprime=[float(_bprime(d_its[i][q_:q_ + 1], dh, kap[q_:q_ + 1])) for i in range(nt + 1)],
                 xmc_z=float(xmc[q_] @ n - pn), verts=[])
        for v_ in vn.tolist():
            De = [U0[v_] + thinv * (ys[i][v_] - U0[v_]) for i in range(nt + 1)]
            t["verts"].append(dict(n=int(v_), m=float(mn[v_]), meff=float(meff[v_]), sk=float(sk[v_]), pm=int(pm[v_]), z0=float(z0[v_]),
                                   z=[float(z0[v_] + De[i] @ n) for i in range(nt + 1)],
                                   bGn=[float(bgn[i][v_]) for i in range(nt + 1)], g=[float(torch.linalg.norm(g0[i][v_])) for i in range(nt + 1)],
                                   fn=[float(fe[i][v_] @ n) for i in range(nt + 1)], ft=[float(torch.linalg.norm(fe[i][v_] - (fe[i][v_] @ n) * n)) for i in range(nt + 1)],
                                   vy=float(sv.PRs["vk"][v_] @ n), ykn=(float(sv.ykn[v_]) if getattr(sv, "ykn", None) is not None else None), dUt=float(torch.linalg.norm(Ut[v_] - U0[v_])), dU=[float(torch.linalg.norm(ys[i][v_] - U0[v_])) for i in range(nt + 1)]))
        tets.append(t)
    return dict(glob=glob, band=ze, tets=tets, nsel=len(sel), Q=Q, nN=nN, thinv=thinv, U0max=float(U0.abs().max()))


def _write(rec):
    try:
        with open(os.path.join(OUT[0], "tetfix.jsonl"), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        INFO["n_rec"] += 1
    except Exception as e_:
        INFO["err"] = ("write " + str(e_))[:300]


def _wrap_step(st0):
    def step(s, t, *a, **kw):
        if getattr(s, "_tfin", False):
            return st0(s, t, *a, **kw)
        s._tfin = True
        try:
            r = st0(s, t, *a, **kw)
        finally:
            s._tfin = False
        try:
            fr = int(round(float(t) / float(s.h)))
            if WIN[0] <= fr <= WIN[1] and bool(torch.isfinite(s.x[:1]).all()) and getattr(s, "prec", False):
                torch.cuda.synchronize()
                _write(dict(f=fr, **trace(s, fr)))
        except Exception as e_:
            import traceback
            _write(dict(f=int(round(float(t) / float(s.h))), err=traceback.format_exc()[-1500:]))
        return r
    step._tfx = True
    return step


# ---- 수정 --tff 1 (사면체 뒤집힘 불가능 선탐색) · --tff 2 (시작점 섞기 점검) · 3 = 둘 다: 선탐색 후보 (rt_elem_lsN, 시작점 후보 rt_y0pick 도 같은 커널) 의 탄성 에너지를
# 후보 y 의 사면체 det(X) <= 0 이거나 걸음 끝 외삽 상태 (U + thinv (y - U), 걸음 시작 U = 0 이라 F + thinv (X - F)) 의 det <= 0 이면 무한대 (1e30) 로.
# 장벽 에너지가 d <= 0 에서 1e30 인 것 (rt_bar_lsN) 과 같은 방식 -- 새 항·벌점·임계값·클램프 없음; 입자가 있는 사면체 (V_q > 0) 만.
_RX = r"(?<![\w.])blocksum_add\(en, &SC\[40\+i\]\);"
_FF = r"""if (on && vq > 0.f) { float Xe_[9]; for (int k_=0;k_<9;k_++) Xe_[k_] = F[k_] + (%sf)*(X[k_] - F[k_]);
            float dY_ = X[0]*(X[4]*X[8]-X[5]*X[7]) - X[1]*(X[3]*X[8]-X[5]*X[6]) + X[2]*(X[3]*X[7]-X[4]*X[6]);
            float dE_ = Xe_[0]*(Xe_[4]*Xe_[8]-Xe_[5]*Xe_[7]) - Xe_[1]*(Xe_[3]*Xe_[8]-Xe_[5]*Xe_[6]) + Xe_[2]*(Xe_[3]*Xe_[7]-Xe_[4]*Xe_[6]);
            if (!(dY_ > 0.f) || !(dE_ > 0.f)) en = 1e30; }
        blocksum_add(en, &SC[40+i]);"""


_TFCHK = r"""
// 절 CM178 --tff 2: 시작점 섞기 (--y0loc) 의 한 회전 점검 (yl_chk 와 같은 규칙) 에 사면체 뒤집힘 추가. 입자 있는 사면체 (V_q > 0) 의 X(y) 또는 걸음 끝 외삽 X(U + thinv (y - U))
// 의 det <= 0 이면 위반 -> 그 사면체 꼭짓점 넷의 k 를 (넷의 최대 k) + 1 로 (yl_merge 가 합침), 위반 수를 fl 에.
__global__ void __launch_bounds__(128) tf_chk(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* y, const float* U, float thinv,
                         const int* kn, int* kn2, int* fl) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    if (!(Vq[q] > 0.f)) return;
    int oc = q & 7; if (oc >= 6) return;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9], X[9], Xe[9];
    for (int k=0;k<9;k++) { F[k] = Fq[(size_t)q*9+k]; X[k] = F[k]; Xe[k] = F[k]; }
    for (int o=0;o<27;o++) {
        const float* bg = BG + (oc*27+o)*3;
        if (bg[0] == 0.f && bg[1] == 0.f && bg[2] == 0.f) continue;
        float g[3]; gqn(F, bg, g); int id = ids[o];
        for (int a=0;a<3;a++) { float ya = y[id*3+a]; float ye = U[id*3+a] + thinv*(ya - U[id*3+a]);
            for (int b=0;b<3;b++) { X[a*3+b] += ya*g[b]; Xe[a*3+b] += ye*g[b]; } }
    }
    float dY = X[0]*(X[4]*X[8]-X[5]*X[7]) - X[1]*(X[3]*X[8]-X[5]*X[6]) + X[2]*(X[3]*X[7]-X[4]*X[6]);
    float dE = Xe[0]*(Xe[4]*Xe[8]-Xe[5]*Xe[7]) - Xe[1]*(Xe[3]*Xe[8]-Xe[5]*Xe[6]) + Xe[2]*(Xe[3]*Xe[7]-Xe[4]*Xe[6]);
    if (dY > 0.f && dE > 0.f) return;
    atomicAdd(fl, 1);
    int km = 0;
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; if (bg[0] != 0.f || bg[1] != 0.f || bg[2] != 0.f) km = max(km, kn[ids[o]]); }
    int kt = (km >= 999) ? 1000 : km + 1;
    for (int o=0;o<27;o++) { const float* bg = BG + (oc*27+o)*3; if (bg[0] != 0.f || bg[1] != 0.f || bg[2] != 0.f) atomicMax(&kn2[ids[o]], kt); }
}
"""
_RYM = "__global__ void yl_merge("


def _ff_transform(code, thinv, ff):
    import re
    if "rt_elem_lsN" not in code:
        return code, 0
    cnt = 0
    if ff & 1:
        m = list(re.finditer(_RX, code))
        if len(m) != 1:
            return code, -len(m)
        code = re.sub(_RX, lambda _m: _FF % repr(float(thinv)), code)
        cnt += 1
    if ff & 2:
        if code.count(_RYM) != 1:
            return code, -10
        code = code.replace(_RYM, _TFCHK + _RYM)
        cnt += 2
    return code, cnt


def _wrap_k(k0):
    def k(s, name, n, args, b=128):
        r = k0(s, name, n, args, b)
        if name == "yl_chk" and getattr(s, "_tff2", False) and s._pkc is None:
            from . import fem_an_rt as ART
            P = ART.P if hasattr(ART, "P") else (lambda t: (t.data_ptr() if t is not None else 0))
            k0(s, "tf_chk", n, (args[0], args[1], P(s.BG), P(s.Fq), P(s.Vq), args[3], args[4], args[5], args[10], args[11], args[12]), b)
            INFO["n_chk"] = INFO.get("n_chk", 0) + 1
        return r
    k._tff = True
    return k


def install_ff(thinv, ff):
    import cupy as cp
    INFO["ff"] = int(ff)
    base = cp.RawModule
    if not getattr(base, "_tff", False):
        def RawModule(*a, **kw):
            a = list(a)
            code = kw.get("code", a[0] if a else None)
            if isinstance(code, str):
                code2, cnt = _ff_transform(code, INFO["thinv"], INFO["ff"])
                INFO["ff_mods"].append(cnt)
                if cnt > 0:
                    if "code" in kw:
                        kw["code"] = code2
                    else:
                        a[0] = code2
            return base(*a, **kw)
        RawModule._tff = True
        cp.RawModule = RawModule
    INFO["thinv"] = float(thinv)
    if ff & 2:
        from . import fem_an_rt as ART
        C = ART.RTSolver
        if "k" in C.__dict__ and not getattr(C.__dict__["k"], "_tff", False):
            C.k = _wrap_k(C.__dict__["k"])
        f0 = C.__init__
        if not getattr(f0, "_tff", False):
            def init(s, *a, _f0=f0, **kw):
                _f0(s, *a, **kw)
                s._tff2 = True
            init._tff = True
            C.__init__ = init
    print(f"[절 CM178] --tff {ff}: " + ("사면체 뒤집힘 불가능 선탐색 " if ff & 1 else "") + ("시작점 섞기에 사면체 뒤집힘 점검 " if ff & 2 else "") + f"(thinv {thinv})", flush=True)


def install(out, mode, win, ff=0, thinv=1.125):
    OUT[0] = out
    INFO["mode"] = int(mode)
    WIN[0], WIN[1] = win
    if ff:
        install_ff(thinv, ff)
    if mode >= 1:
        from . import fem_an_rt as ART
        PRD.ON[0] = True                                                                   # 절 CM26 사본 훅 (alloc 은 --nrec 이 부름)
        C = ART.RTSolver
        if "step" in C.__dict__ and not getattr(C.__dict__["step"], "_tfx", False):
            C.step = _wrap_step(C.__dict__["step"])
    print(f"[절 CM178] tetfix 켬 -- 모드 {mode}, 창 {win}, tff {ff}", flush=True)
