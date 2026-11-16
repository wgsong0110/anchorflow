"""절 CM101 모래시계 모드 세기 (진단, 읽기만): 실제 씬 프레임의 첫 반복 탄성 헤시안 (점유 가중 포함) 의 영·근사 영 고유 모드 수와 자리.

헤시안 (y = 0 -> X = F_q, 실행기 rt_elem 의 첫 반복 자리): K = Σ_q V_q G_qᵀ C G_q, G_qn = F_qᵀ ∇w_n(x_q) (실행기 gqn 과 같음), C = 등방 선형 탄성 접선 (씬 μ·λ).
  가우스 점 2x2x2 (팔분, 무게 ⅛) 또는 3x3x3 (fem_q27 정의: 축마다 5/18·8/18·5/18 몫 칸). V_q = (1 − κ) 몫 칸 입자 부피 + κ W_o 칸 입자 부피 (κ = 씬 --gq_oct, 수박 0),
  F_q = 몫 칸 입자 F 산술 평균 (빈 몫 칸은 칸 평균; 실행기는 회전·로그 신장 평균 -- 모드 수에는 영향 거의 없음). 벌점 항 (P1X·가벼움) 은 넣지 않음 (탄성 헤시안만).
고유값: 야코비 배율 Â = D^-½ K D^-½ (지지 없는 자유도 뺌; 칸 81x81 강성을 모은 희소 행렬, cuSPARSE 곱), 체비쇼프 거름 부분공간 반복 (블록) + 레일리-리츠 (f64). λ̂ < 1e-5 λ̂_max = 영, < 1e-3 = 근사 영.
자리: 영 모드 부분공간의 노드 몫 Σ|u_n|² (u = D^-½ z, 모드마다 단위) 을 노드 '지지 칸 점유 수' (노드를 27 노드 판에 넣는 칸 27 개 중 점유 칸 수) 구간별로.
사용: python exe/q27_eig.py --pred pred.pt --solver q27solver.pt --frames 0,c --q 1|8|27|64 --out out.json [--kb 2400]
"""
import argparse
import json
import math
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--solver", required=True)
ap.add_argument("--frames", default="0,c")
ap.add_argument("--q", type=int, default=8)
ap.add_argument("--kb", type=int, default=2400)
ap.add_argument("--deg", type=int, default=24)
ap.add_argument("--iters", type=int, default=14)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)
SV = torch.load(a.solver, map_location="cpu", weights_only=False)
D = torch.load(a.pred, map_location="cpu", weights_only=False)
XP = D["pred"]; FP = D.get("F")
A = SV["A"].double().reshape(3, 3).to(dev); Ai = SV["Ai"].double().reshape(3, 3).to(dev)
vol = SV["vol"].double().reshape(-1).to(dev)
mu = float(SV["mu"].double().mean()) if torch.is_tensor(SV["mu"]) else float(SV["mu"])
lam = float(SV["lam"].double().mean()) if torch.is_tensor(SV["lam"]) else float(SV["lam"])
kap = SV.get("goct"); kap = float(kap) if (kap is not None and float(kap) >= 0) else 0.0
cell = float(A.norm(dim=1).mean())


def contact_frame():
    zmin = torch.stack([torch.nan_to_num(XP[f, :, 2].float(), nan=1e9).min() for f in range(XP.shape[0])])
    lo = float(zmin.min())
    for f in range(XP.shape[0]):
        if float(zmin[f]) <= lo + 0.5 * cell:
            return f
    return int(zmin.argmin())


def axis_tab(q):
    if q == 1:                                                                          # 절 CM179: 칸 가운데 1 점
        d = [0.0]; w = [1.0]; edges = []
    elif q == 8:
        d = [-0.5 / math.sqrt(3), 0.5 / math.sqrt(3)]; w = [0.5, 0.5]; edges = [1.0]
    elif q == 64:                                                                        # 절 CM195: 가우스 4x4x4 (fem_q27b.py --q27n 4 와 같은 점·무게·몫 칸 경계 = 누적 무게)
        ga = math.sqrt(3 / 7 - 2 / 7 * math.sqrt(6 / 5)); gb = math.sqrt(3 / 7 + 2 / 7 * math.sqrt(6 / 5))
        wa = (18 + math.sqrt(30)) / 36; wb = (18 - math.sqrt(30)) / 36
        d = [-0.5 * gb, -0.5 * ga, 0.5 * ga, 0.5 * gb]; w = [0.5 * wb, 0.5 * wa, 0.5 * wa, 0.5 * wb]
        edges = [0.5 + w[0], 0.5 + w[0] + w[1], 0.5 + w[0] + w[1] + w[2]]
    else:
        s = 0.5 * math.sqrt(0.6); d = [-s, 0.0, s]; w = [5 / 18, 8 / 18, 5 / 18]; edges = [0.5 + 5 / 18, 0.5 + 13 / 18]
    return d, w, edges


def n_components(ids_c, nn):
    """절 CM195: 노드를 함께 쓰는 칸끼리 이은 연결 성분 수 (성분마다 강체 모드 6 개 = 탄성 헤시안의 자명한 영 모드)."""
    lab = torch.arange(nn, device=ids_c.device)
    for _ in range(100000):
        cl = lab[ids_c].min(1).values
        new = lab.clone().scatter_reduce_(0, ids_c.reshape(-1), cl[:, None].expand_as(ids_c).reshape(-1), reduce="amin")
        new = new[new]                                                                   # 포인터 건너뛰기
        if torch.equal(new, lab):
            break
        lab = new
    return int(torch.unique(lab[ids_c.reshape(-1)]).numel())


def build(fi, q):
    x = XP[fi].to(dev).double()
    F = FP[fi].to(dev).double().reshape(-1, 3, 3) if FP is not None else torch.eye(3, device=dev, dtype=torch.float64).expand(x.shape[0], 3, 3)
    ok = torch.isfinite(x).all(1) & torch.isfinite(F.reshape(-1, 9)).all(1)
    x, F, v = x[ok], F[ok], vol[ok]
    y = x @ Ai.T
    b = torch.floor(y - 0.5); fx = y - b; b = b.long()
    bmin = b.min(0).values
    b = b - bmin
    nbx = (b.max(0).values + 3).tolist()
    ck = (b[:, 0] * (nbx[1]) + b[:, 1]) * (nbx[2]) + b[:, 2]
    cu, cinv = torch.unique(ck, return_inverse=True)
    nc = cu.numel()
    cb = torch.stack([cu // (nbx[1] * nbx[2]), (cu // nbx[2]) % nbx[1], cu % nbx[2]], 1)
    d1, w1, edges = axis_tab(q)
    n1 = len(d1)
    sub = torch.zeros_like(fx, dtype=torch.long)
    for e in edges:
        sub += (fx >= e).long()
    so = (sub[:, 0] * n1 + sub[:, 1]) * n1 + sub[:, 2]
    nq1 = n1 ** 3
    qi = cinv * nq1 + so
    Vs = torch.zeros(nc * nq1, device=dev, dtype=torch.float64).index_add_(0, qi, v)
    Fs = torch.zeros(nc * nq1, 9, device=dev, dtype=torch.float64).index_add_(0, qi, v[:, None] * F.reshape(-1, 9))
    Vc = torch.zeros(nc, device=dev, dtype=torch.float64).index_add_(0, cinv, v)
    Fc = torch.zeros(nc, 9, device=dev, dtype=torch.float64).index_add_(0, cinv, v[:, None] * F.reshape(-1, 9))
    Wo = torch.tensor([w1[i] * w1[j] * w1[k] for i in range(n1) for j in range(n1) for k in range(n1)], device=dev, dtype=torch.float64)
    Vq = (1 - kap) * Vs + kap * (Vc[:, None] * Wo[None, :]).reshape(-1)
    Fq = torch.where((Vs > 0)[:, None], Fs / Vs.clamp_min(1e-300)[:, None], (Fc / Vc[:, None]).repeat_interleave(nq1, 0)).reshape(-1, 3, 3)
    # 기저 기울기 표 (점 o, 노드 27)
    BG = torch.zeros(nq1, 27, 3, device=dev, dtype=torch.float64)
    for o in range(nq1):
        ii = (o // (n1 * n1), (o // n1) % n1, o % n1)
        fxv = [1.0 + d1[ii[t]] for t in range(3)]
        W = [[0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2] for f in fxv]
        dW = [[f - 1.5, -2 * (f - 1), f - 0.5] for f in fxv]
        for n in range(27):
            i, j, k = n // 9, (n // 3) % 3, n % 3
            gy = torch.tensor([dW[0][i] * W[1][j] * W[2][k], W[0][i] * dW[1][j] * W[2][k], W[0][i] * W[1][j] * dW[2][k]], device=dev, dtype=torch.float64)
            BG[o, n] = gy @ Ai
    # 노드 번호
    off = torch.tensor([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], device=dev)
    nodes = (cb[:, None, :] + off[None]).reshape(-1, 3)
    nk = (nodes[:, 0] * (nbx[1]) + nodes[:, 1]) * (nbx[2]) + nodes[:, 2]
    nu, ninv = torch.unique(nk, return_inverse=True)
    ids_c = ninv.reshape(nc, 27)
    keep = Vq > 0
    qcell = torch.arange(nc * nq1, device=dev) // nq1
    qo = torch.arange(nc * nq1, device=dev) % nq1
    qcell, qo, Vq, Fq = qcell[keep], qo[keep], Vq[keep], Fq[keep]
    G = torch.einsum("qij,qni->qnj", Fq, BG[qo])                                         # g = Fᵀ bg
    ids = ids_c[qcell]
    # 노드 지지 칸 점유 수
    occ = torch.zeros(nu.numel(), device=dev, dtype=torch.long).index_add_(0, ids_c.reshape(-1), torch.ones(ids_c.numel(), device=dev, dtype=torch.long))
    I3 = torch.eye(3, device=dev, dtype=torch.float64)
    f0 = dict(F_dev_max=float((F - I3).abs().amax()), x_sum=float(x.sum()), F_sum=float(F.sum()))     # 절 CM195: 같은 첫 상태인지 판끼리 대조
    return dict(nc=nc, nq=int(keep.sum()), nn=int(nu.numel()), ids=ids, G=G.float(), V=Vq.float(), occ=occ, np=int(ok.sum()), qcell=qcell, ids_c=ids_c,
                ncomp=n_components(ids_c, int(nu.numel())), f0=f0)


def make_op(S):
    """칸마다 81x81 강성 (점 합) 을 모아 희소 행렬 (CSR) 로 -- 야코비 배율 Â = D^-½ K D^-½ 곱은 cuSPARSE SpMM."""
    nn = S["nn"]; ids, G, V = S["ids"], S["G"].double(), S["V"].double()
    nq = ids.shape[0]
    qc = S["qcell"]
    nc = S["nc"]
    Kc = torch.zeros(nc, 27, 3, 27, 3, device=dev, dtype=torch.float32)
    I3 = torch.eye(3, device=dev, dtype=torch.float64)
    CH = 4096
    for q0 in range(0, nq, CH):
        q1 = min(nq, q0 + CH)
        g = G[q0:q1]; v = V[q0:q1]
        t1 = lam * torch.einsum("qna,qmb->qnamb", g, g)
        t3 = mu * torch.einsum("qnb,qma->qnamb", g, g)
        gg = torch.einsum("qnj,qmj->qnm", g, g)
        t2 = mu * gg[:, :, None, :, None] * I3[None, None, :, None, :]
        Kc.index_add_(0, qc[q0:q1], ((t1 + t2 + t3) * v[:, None, None, None, None]).float())
        del t1, t2, t3, gg
    ic = S["ids_c"]                                                                     # (nc, 27)
    dof = (ic[:, :, None] * 3 + torch.arange(3, device=dev)[None, None, :]).reshape(nc, 81)
    rows = dof[:, :, None].expand(nc, 81, 81).reshape(-1)
    cols = dof[:, None, :].expand(nc, 81, 81).reshape(-1)
    vals = Kc.reshape(-1)
    del Kc
    n = nn * 3
    K = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n)).coalesce()
    del rows, cols, vals
    idx = K.indices(); val = K.values()
    dg = torch.zeros(n, device=dev, dtype=torch.float64)
    m = idx[0] == idx[1]
    dg.index_add_(0, idx[0][m], val[m].double())
    act = dg > 1e-12 * dg.max()
    s = torch.zeros_like(dg); s[act] = dg[act].rsqrt()
    val = (val.double() * s[idx[0]] * s[idx[1]]).float()
    A = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce().to_sparse_csr()
    S["nnz"] = int(val.numel())
    del idx, val, K

    def Ah(Z):
        return torch.sparse.mm(A, Z)
    return Ah, act, s


KB = [a.kb]


def chfsi(Ah, act, n, kb):
    na = int(act.sum())
    kb = min(kb, na - 1)
    KB[0] = kb
    am = act.float()
    x = torch.randn(n, 1, device=dev) * am[:, None]
    for _ in range(40):
        x = Ah(x); x = x / x.norm()
    lmax = float((x * Ah(x)).sum()) * 1.05
    X = torch.randn(n, kb, device=dev) * am[:, None]
    X, _ = torch.linalg.qr(X.double()); X = X.float()
    cut = 0.02 * lmax
    hist = []
    for it in range(a.iters):
        e = (lmax - cut) / 2; c = (lmax + cut) / 2
        Y0 = X; Y1 = (Ah(X) - c * X) / e
        for _ in range(2, a.deg + 1):
            Y2 = 2 * (Ah(Y1) - c * Y1) / e - Y0
            Y0, Y1 = Y1, Y2
            nr = Y1.norm(dim=0).max()
            if nr > 1e15:
                Y0 = Y0 / nr; Y1 = Y1 / nr
        Q, _ = torch.linalg.qr(Y1.double())
        Q = Q.float()
        AQ = Ah(Q)
        Hs = (Q.double().T @ AQ.double()); Hs = 0.5 * (Hs + Hs.T)
        ev, W = torch.linalg.eigh(Hs)
        X = (Q.double() @ W).float()
        AX = (AQ.double() @ W).float()
        res = (AX - X * ev.float()[None, :]).norm(dim=0) / lmax
        cut = float(ev[-1]) if float(ev[-1]) > 0 else cut
        nz5 = int((ev < 1e-5 * lmax).sum()); nz3 = int((ev < 1e-3 * lmax).sum())
        hist.append(dict(it=it, cut_rel=cut / lmax, n_lt1e5=nz5, n_lt1e3=nz3, res_max_lt1e3=float(res[ev < 1e-3 * lmax].max()) if nz3 else None))
        print("[q27 eig] it", it, hist[-1], flush=True)
    return ev.cpu().numpy(), X, res.cpu().numpy(), lmax, hist


t0 = time.time()
fr = []
for t in a.frames.split(","):
    fr.append(contact_frame() if t == "c" else int(t))
OUT = dict(pred=a.pred, q=a.q, mu=mu, lam=lam, kap=kap, cell=cell, cases=[])
for fi in fr:
    S = build(fi, a.q)
    Ah, act, sc = make_op(S)
    n = S["nn"] * 3
    ev, X, res, lmax, hist = chfsi(Ah, act, n, a.kb)
    z5 = ev < 1e-5 * lmax; z3 = ev < 1e-3 * lmax
    conv = res < 1e-4
    # 영 모드 자리: u = D^-½ z, 모드마다 단위 -> 노드 몫
    loc = {}
    occ = S["occ"]
    bins = [(1, 8), (9, 17), (18, 26), (27, 27)]
    for nm, msk in (("zero", z5), ("near", z3 & ~z5)):
        k = int(msk.sum())
        row = dict(n=k)
        if k:
            U = (sc.float()[:, None] * X[:, torch.from_numpy(msk).to(dev)])
            U = U / U.norm(dim=0, keepdim=True).clamp_min(1e-30)
            wn = (U * U).sum(1).reshape(-1, 3).sum(1) / k
            for lo, hi in bins:
                m = (occ >= lo) & (occ <= hi)
                row["%d-%d" % (lo, hi)] = dict(node_frac=float(m.float().mean()), mode_frac=float(wn[m].sum()))
        loc[nm] = row
    rec = dict(frame=fi, frame_label="f%d" % (fi + 1), nc=S["nc"], nq=S["nq"], nn=S["nn"], np=S["np"], nnz=S.get("nnz"), ndof_act=int(act.sum()), kb=KB[0], lmax=lmax,
               n_zero=int(z5.sum()), ncomp=S["ncomp"], n_rigid=6 * S["ncomp"], n_zero_nonrigid=int(z5.sum()) - 6 * S["ncomp"], f0=S["f0"], n_near=int((z3 & ~z5).sum()), n_zero_conv=int((z5 & conv).sum()), all_kb_small=bool(z3.all()),
               ev_head=[float(q) / lmax for q in ev[:12]], ev_at=[float(ev[min(len(ev) - 1, j)]) / lmax for j in (6, 30, 100, 300, 1000)], loc=loc, hist=hist[-3:], sec=round(time.time() - t0, 1))
    OUT["cases"].append(rec)
    print("[q27 eig] 프레임", fi, "q", a.q, {k: rec[k] for k in ("nc", "nq", "nn", "n_zero", "ncomp", "n_zero_nonrigid", "n_near", "n_zero_conv", "all_kb_small")}, flush=True)
    json.dump(OUT, open(a.out, "w"), indent=1)
    del Ah, X
    torch.cuda.empty_cache()
print("[q27 eig] 끝", round(time.time() - t0, 1), "s")
