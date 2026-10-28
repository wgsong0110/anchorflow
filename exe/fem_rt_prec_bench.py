"""PCG 전처리·행렬 정밀도 탐색 (doc/fem_realtime.md 절 J, 사용자 10-28 "할 수 있는 거 다"): 정답 프레임 t 상태의 입자 바닥 계
(최종 판과 같은 행렬: 워프 사면체 헤시안 + 바닥 후보 f32 + |λ| 투영 바닥 1e-6, 대칭 반쪽 블록 ELL) 를 float64 torch 로 풀어 **같은 rtol 1e-6 반복 수** 를 잰다.

  bj            블록 야코비 (기준 = 최종 판 전처리)
  bjS<s>[c]     노드 묶음 블록 야코비: 격자 좌표 gi//s 묶음 (s=2: 최대 8 노드 24x24, s=3: 27 노드 81x81) 의 주 부분행렬 역; c = 접촉 영역 묶음만 (나머지 3x3)
  ssor / dilu   8 색 (Kuhn 15 점 스텐실 이웃은 모두 다른 색) 대칭 블록 가우스-자이델 / 블록 D-ILU (IC0 의 대각 수정판, 같은 희소 패턴)
  add<s><t|r>   2 단계 가법: D⁻¹ + Z E⁻¹ Zᵀ (Z = gi//s 묶음의 평행이동 t 또는 강체 6 모드 r), E = Zᵀ A Z
  defl<s><t|r>  같은 Z 로 deflated PCG (Saad et al. 2000)
  dc<s><t|n>    접촉 노드 (바닥 높이 ≤ --c_eps 입자의 노드) 만의 묶음 Z: 접선 2 방향 (t) 또는 3 방향 (n), deflated PCG
  ac<s><t|n>    같은 접촉 Z 로 2 단계 가법
  eig<k>        (참고, --eig) D⁻¹A 의 가장 작은 고유벡터 k 개로 deflation (상한)
  mix_<p>       혼합 정밀도 반복 개선 모방 (커널 pcg_t32 mode 1 과 같은 식): 안쪽 float32 벡터 + 행렬 p (f32 / f16s / bf16s / f16c = 야코비 대칭 척도 S A S 를
                16 비트로, c = 평행이동 모드 보정 대각), 신뢰 갱신 r = b − A₃₂ x (float64) 로만 멈춤 (실제 잔차 ≤ rtol)
--eig k: scipy eigsh (shift-invert) 로 D⁻¹A 의 가장 작은 k 개 고유값 + 접촉 노드·접선 성분 비율 (낮은 모드가 어디 있는지).
사용: python exe/fem_rt_prec_bench.py --t 0 10 40 70 --out <폴더>
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--t", type=int, nargs="+", default=[0, 10, 40, 70])
ap.add_argument("--out", required=True)
ap.add_argument("--rtol", type=float, default=1e-6)
ap.add_argument("--maxit", type=int, default=20000)
ap.add_argument("--eig", type=int, default=24)
ap.add_argument("--c_eps", type=float, default=0.02, help="접촉 입자: 바닥 높이 s ≤ c_eps (stick ε 0.01 의 2 배)")
ap.add_argument("--variants", default="bj,bjS2,bjS2c,bjS3c,ssor,dilu,add4t,add4r,add6r,defl4t,defl4r,defl6r,defl8r,dc1t,dc1n,dc2n,dc4n,ac1n,ac2n,"
                                      "mix_f32,mix_f16s,mix_bf16s,mix_f16c")
a = ap.parse_args()
dev = "cuda"
os.makedirs(a.out, exist_ok=True)
f8 = torch.float64
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).float()
vol = m / float(cfg["density"])
N = int(d["x"].shape[1])
g = torch.as_tensor(cfg["g"], device=dev, dtype=torch.float32)
bc = [q for q in cfg["boundary_conditions"] if q.get("type") == "surface_collider"][0]
P0 = torch.as_tensor(bc["point"], device=dev, dtype=f8)
NR = torch.as_tensor(bc["normal"], device=dev, dtype=f8)
NR = NR / NR.norm()
_e = torch.eye(3, device=dev, dtype=f8)[int(NR.abs().argmin())]
T1 = torch.linalg.cross(NR, _e)
T1 = T1 / T1.norm()
T2 = torch.linalg.cross(NR, T1)
print(f"[장치] {torch.cuda.get_device_name(0)}  바닥 법선 {NR.tolist()}", flush=True)
RES = []


def build(t):
    x = d["x"][t].to(dev).float()
    v = d["v"][t].to(dev).float()
    F = DM.traj_F(d, h, dev)[t].to(dev).float().reshape(N, 3, 3)
    G = RTC.GridK(x, 0.028284)
    U, ws = RT.node_inertial(x, v, m, G, g, h)
    CF = RTC.CudaTetqK(G, x, v, F, m, vol, cfg, h, U)
    CF.warp, CF.near32, CF.sym = True, True, 1
    PC = RTC.CudaPcg(60000, dev)
    PC.sym = 1
    slots = RTC.slots_k(G)
    PC.B.zero_()
    CF.cnt.zero_()
    CF.hess_into(U, slots, PC.B, 1e-6, Ms=PC.Mmax)
    RTC.asm_k(G, PC, 1, PC.Mmax)
    b = CF.energy_grad(U)[1].reshape(-1).double()
    Ms, M = PC.Mmax, G.M
    Bh = PC.B.view(15, 9, Ms)[:8, :, :M].permute(2, 0, 1).reshape(M, 8, 3, 3)          # float32 (저장 값)
    nb = PC.nb.view(15, Ms)[:8, :M].t().long()
    # 블록 쌍 (I, J, blk) 전체 (양쪽 방향)
    ii = torch.arange(M, device=dev).unsqueeze(1).expand(M, 8)
    ok = nb >= 0
    I0, J0, B0 = ii[ok], nb[ok], Bh[ok].double()
    k_ = torch.arange(8, device=dev).unsqueeze(0).expand(M, 8)[ok]
    off = k_ > 0
    I = torch.cat([I0, J0[off]])
    J = torch.cat([J0, I0[off]])
    Bk = torch.cat([B0, B0[off].transpose(1, 2)])
    # 접촉 노드: 바닥 높이 ≤ c_eps 인 입자 (걸음 시작 U 로 옮긴 위치) 의 네 노드
    u = (G.lam.unsqueeze(-1) * U.float()[G.rows]).sum(1)
    s = ((x + u).double() - P0) @ NR
    cpart = s <= a.c_eps
    cnode = torch.zeros(M, dtype=torch.bool, device=dev)
    cnode[G.rows[cpart].reshape(-1)] = True
    return dict(G=G, CF=CF, PC=PC, b=b, M=M, I=I, J=J, Bk=Bk, Bh=Bh, nb=nb, cnode=cnode, npos=G.npos.double(), gi=G.gi, ncp=int(cpart.sum()),
                nnear=CF.nn)


def to_csr(I, J, Bk, n3):
    r = (3 * I).unsqueeze(1).unsqueeze(2) + torch.arange(3, device=dev).view(1, 3, 1)
    c = (3 * J).unsqueeze(1).unsqueeze(2) + torch.arange(3, device=dev).view(1, 1, 3)
    r, c = r.expand_as(Bk).reshape(-1), c.expand_as(Bk).reshape(-1)
    A = torch.sparse_coo_tensor(torch.stack([r, c]), Bk.reshape(-1), (n3, n3)).coalesce()
    return A.to_sparse_csr()


def mv(A, x):
    return (A @ x.unsqueeze(-1)).squeeze(-1)


def pcg(A, prec, b, Z=None, AZ=None, Einv=None, maxit=None, rtol=None):
    maxit = maxit or a.maxit
    rtol = rtol or a.rtol
    bn = float(b.norm())
    if Z is not None:
        x = Z @ (Einv @ (Z.t() @ b))
        r = b - mv(A, x)
    else:
        x = torch.zeros_like(b)
        r = b.clone()
    z = prec(r)
    p = z - Z @ (Einv @ (AZ.t() @ z)) if Z is not None else z.clone()
    rz = float(r @ z)
    it = 0
    for it in range(1, maxit + 1):
        Ap = mv(A, p)
        pAp = float(p @ Ap)
        if not pAp > 0:
            break
        al = rz / pAp
        x += al * p
        r -= al * Ap
        if float(r.norm()) < rtol * bn:
            break
        z = prec(r)
        rz2 = float(r @ z)
        be = rz2 / rz
        rz = rz2
        p = be * p + z
        if Z is not None:
            p -= Z @ (Einv @ (AZ.t() @ z))
    tr = float((b - mv(A, x)).norm()) / bn
    return x, it, tr


def blk_apply(Dinv, r):
    M = Dinv.shape[0]
    return (Dinv @ r.view(M, 3, 1)).view(-1)


def color_of(gi):
    return ((gi[:, 0] + 2 * gi[:, 1] + 4 * gi[:, 2]) % 8).long()


def agg_of(gi, s, sel=None):
    """gi//s 묶음 번호 (sel 이 있으면 그 노드만, 나머지 -1) 와 묶음 수."""
    key = torch.div(gi - gi.min(0).values, s, rounding_mode="floor")
    lin = (key[:, 0] * 100003 + key[:, 1]) * 100003 + key[:, 2]
    if sel is not None:
        lin = torch.where(sel, lin, torch.full_like(lin, -1))
    u, inv = torch.unique(lin, return_inverse=True)
    if sel is not None:
        has_neg = bool((u == -1).any())
        inv = inv - (1 if has_neg else 0)
        inv = torch.where(sel, inv, torch.full_like(inv, -1))
        return inv, int(u.numel()) - (1 if has_neg else 0)
    return inv, int(u.numel())


def zbasis(S, agg, na, mode):
    """Z [3M, k] dense: 묶음마다 평행이동 3 (t) / 강체 6 (r) / 접선 2 (tg) / 3 방향 (n)."""
    M = agg.shape[0]
    pos = S["npos"]
    ok = agg >= 0
    if mode == "tg":
        dirs = [T1, T2]
    else:
        dirs = [torch.eye(3, device=dev, dtype=f8)[q] for q in range(3)]
    nc = len(dirs) + (3 if mode == "r" else 0)
    Z = torch.zeros(M, 3, na * nc, device=dev, dtype=f8)
    idx = torch.nonzero(ok).squeeze(1)
    ag = agg[idx]
    for q, e in enumerate(dirs):
        Z[idx, :, ag * nc + q] = e.unsqueeze(0).expand(idx.numel(), 3)
    if mode == "r":
        cen = torch.zeros(na, 3, device=dev, dtype=f8).index_add_(0, ag, pos[idx])
        cnt = torch.zeros(na, device=dev, dtype=f8).index_add_(0, ag, torch.ones_like(ag, dtype=f8))
        cen = cen / cnt.clamp_min(1).unsqueeze(1)
        rel = pos[idx] - cen[ag]
        for q in range(3):
            e = torch.eye(3, device=dev, dtype=f8)[q].expand_as(rel)
            Z[idx, :, ag * nc + 3 + q] = torch.linalg.cross(e, rel)
    Z = Z.reshape(3 * M, na * nc)
    keep = Z.abs().sum(0) > 0
    return Z[:, keep]


def coarse(A, Z):
    AZ = (A @ Z)
    E = Z.t() @ AZ
    E = 0.5 * (E + E.t())
    ev = torch.linalg.eigvalsh(E)
    tol = 1e-12 * float(ev.abs().max())
    w, V = torch.linalg.eigh(E)
    wi = torch.where(w > tol, 1.0 / w, torch.zeros_like(w))
    Einv = (V * wi) @ V.t()
    return AZ, Einv, int((w <= tol).sum())


def bigblock(S, A, s, contact_only):
    """gi//s 묶음 주 부분행렬 (밀집) 의 역 -- 묶음에 든 노드만, 없는 자리는 단위."""
    M, gi = S["M"], S["gi"]
    sel = S["cnode"] if contact_only else None
    if contact_only:
        # 접촉 노드를 품은 묶음 전체 (그 묶음의 비접촉 노드도 같이)
        agg_all, na_all = agg_of(gi, s)
        hit = torch.zeros(na_all, dtype=torch.bool, device=dev)
        hit[agg_all[S["cnode"]]] = True
        sel = hit[agg_all]
    agg, na = agg_of(gi, s, sel)
    loc = (gi - gi.min(0).values) % s
    loc = (loc[:, 0] * s + loc[:, 1]) * s + loc[:, 2]
    nl = s ** 3
    I, J, Bk = S["I"], S["J"], S["Bk"]
    same = (agg[I] >= 0) & (agg[I] == agg[J])
    Bb = torch.zeros(na, nl, 3, nl, 3, device=dev, dtype=f8)
    Bb[agg[I[same]], loc[I[same]], :, loc[J[same]], :] = Bk[same]
    Bb = Bb.reshape(na, 3 * nl, 3 * nl)
    occ = torch.zeros(na, nl, dtype=torch.bool, device=dev)
    occ[agg[agg >= 0], loc[agg >= 0]] = True
    eye = torch.eye(3 * nl, device=dev, dtype=f8)
    emp = (~occ).repeat_interleave(3, 1).to(f8)
    Bb = Bb + eye * emp.unsqueeze(1)
    L, info = torch.linalg.cholesky_ex(Bb)
    nbad = int((info != 0).sum())
    Binv = torch.cholesky_inverse(L)
    Dinv = S["Dinv"]

    def prec(r, Binv=Binv, agg=agg, loc=loc, na=na, nl=nl):
        y = blk_apply(Dinv, r).view(M, 3)
        ok = agg >= 0
        R = torch.zeros(na, nl, 3, device=dev, dtype=f8)
        rv = r.view(M, 3)
        R[agg[ok], loc[ok]] = rv[ok]
        Y = (Binv @ R.reshape(na, 3 * nl, 1)).view(na, nl, 3)
        y = y.clone()
        y[ok] = Y[agg[ok], loc[ok]]
        return y.reshape(-1)
    return prec, dict(n_agg=na, size=3 * nl, chol_fail=nbad, nodes_in=int((agg >= 0).sum()))


def sweeps(S, A, kind):
    """8 색 대칭 블록 GS (ssor) / 블록 D-ILU (dilu): (D̃ + L) D̃⁻¹ (D̃ + Lᵀ)."""
    M, I, J, Bk = S["M"], S["I"], S["J"], S["Bk"]
    col = color_of(S["gi"])
    off = I != J
    Dt = S["D"].clone()
    if kind == "dilu":
        for c in range(8):
            sel = off & (col[I] == c) & (col[J] < c)
            if sel.any():
                Bs = Bk[sel]
                contrib = Bs @ torch.linalg.inv(Dt[J[sel]]) @ Bs.transpose(1, 2)
                Dt.index_add_(0, I[sel], -contrib)
    ev = torch.linalg.eigvalsh(0.5 * (Dt + Dt.transpose(1, 2)))
    nneg = int((ev[:, 0] <= 0).sum())
    Dti = torch.linalg.inv(Dt)
    n3 = 3 * M
    Ac = []
    for c in range(8):
        sel = off & (col[I] == c)
        Ac.append((to_csr(I[sel], J[sel], Bk[sel], n3), (col == c)))

    def prec(r):
        y = torch.zeros(M, 3, device=dev, dtype=f8)
        rv = r.view(M, 3)
        for Ac_, cm in Ac:                                        # 앞 쓸기 (낮은 색부터): y_c = D̃_c⁻¹ (r_c - Σ_{낮은 색} A y)
            q = mv(Ac_, y.view(-1)).view(M, 3)
            y[cm] = (Dti[cm] @ (rv[cm] - q[cm]).unsqueeze(-1)).squeeze(-1)
        w = (Dt @ y.unsqueeze(-1)).squeeze(-1)
        xx = torch.zeros(M, 3, device=dev, dtype=f8)
        for Ac_, cm in reversed(Ac):                              # 뒤 쓸기
            q = mv(Ac_, xx.view(-1)).view(M, 3)
            xx[cm] = (Dti[cm] @ (w[cm] - q[cm]).unsqueeze(-1)).squeeze(-1)
        return xx.reshape(-1)
    return prec, dict(neg_diag=nneg)


def mix_emul(S, A, b, kind, delta=0.1):
    """pcg_t32 mode 1 모방: 안쪽 float32 (벡터·블록 야코비·행렬 A_lo), 신뢰 갱신 float64 r = b − A x_hi (A = 저장 float32 값)."""
    M = S["M"]
    Dinv = S["Dinv"]
    diag = torch.diagonal(S["D"], dim1=1, dim2=2).reshape(-1)
    sc = diag.clamp_min(1e-300).rsqrt()                           # S = diag(1/√a_ii)
    vals = A.values()
    crow, colx = A.crow_indices(), A.col_indices()
    rowx = torch.repeat_interleave(torch.arange(3 * M, device=dev), crow[1:] - crow[:-1])
    if kind == "f32":
        vlo = vals.float()
    else:
        sv = vals * sc[rowx] * sc[colx]                            # S A S (대각 1)
        dt = torch.float16 if kind in ("f16s", "f16c") else torch.bfloat16
        q = sv.to(dt).double()
        if kind == "f16c":
            selfb = torch.div(rowx, 3, rounding_mode="floor") == torch.div(colx, 3, rounding_mode="floor")
            q = torch.where(selfb, sv, q)                          # 자기 블록 (칸 0) 은 float32 로 둔다
            # 평행이동 모드 보정: 행마다 (같은 성분 열) 반올림 오차 합을 그 행 대각에 되돌려 A_lo·(평행이동) = A·(평행이동) 이 되게
            # (대각 3x3 블록은 커널에서 float32 로 둔다고 보고 그 대각만 보정 값을 갖는다)
            err = (q - sv) / (sc[rowx] * sc[colx])                 # 원래 척도의 반올림 오차
            same = (rowx % 3) == (colx % 3)
            fix = torch.zeros(3 * M, device=dev, dtype=f8).index_add_(0, rowx[same], err[same])
            dg = rowx == colx
            q = q.clone()
            q[dg] = q[dg] - fix[rowx[dg]] * sc[rowx[dg]] ** 2
        vlo = (q / (sc[rowx] * sc[colx])).float()
        bad = ~torch.isfinite(vlo)
        if bad.any():
            print(f"    [{kind}] 비유한 {int(bad.sum())}", flush=True)
    Alo = torch.sparse_csr_tensor(crow, colx, vlo, (3 * M, 3 * M))
    Di32 = Dinv.float()
    bn = float(b.norm())
    bs = b / bn
    xh = torch.zeros_like(bs)
    x = torch.zeros(3 * M, device=dev, dtype=torch.float32)
    r = bs.float()
    z = blk_apply(Di32, r)
    dd = z.clone()
    Ad = mv(Alo, dd)
    rz = float((r.double() * z.double()).sum())
    dAd = float((dd.double() * Ad.double()).sum())
    rmax, nup, it, thr = 1.0, 0, 0, a.rtol ** 2
    tr = 1.0
    while it < a.maxit:
        if not dAd > 0:
            break
        al = rz / dAd
        x += al * dd
        r -= al * Ad
        z = blk_apply(Di32, r)
        rr = float((r.double() ** 2).sum())
        rz2 = float((r.double() * z.double()).sum())
        it += 1
        if rr < delta * delta * rmax or rr < thr:
            xh += x.double()
            x.zero_()
            rt = bs - mv(A, xh)
            r = rt.float()
            z = blk_apply(Di32, r)
            rr = float((rt * rt).sum())
            rz2 = float((rt * z.double()).sum())
            nup += 1
            rmax = rr
            tr = math.sqrt(rr)
            if rr < thr:
                break
        elif rr > rmax:
            rmax = rr
        be = rz2 / rz
        rz = rz2
        q = mv(Alo, z)
        dd = z + be * dd
        Ad = q + be * Ad
        dAd = float((dd.double() * Ad.double()).sum())
    xx = (xh + x.double()) * bn
    return xx, it, float((b - mv(A, xx)).norm()) / bn, nup


def eig_low(S, A, k):
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    M = S["M"]
    Ac = A.to_sparse_coo().coalesce()
    ij = Ac.indices().cpu().numpy()
    As = sp.csc_matrix((Ac.values().cpu().numpy(), (ij[0], ij[1])), shape=(3 * M, 3 * M))
    D = S["D"].cpu().numpy()
    Ds = sp.block_diag([D[i] for i in range(M)], format="csc")
    t0 = time.time()
    lam, V = sla.eigsh(As, k=k, M=Ds, sigma=0.0, which="LM")
    lmax = sla.eigsh(As, k=1, M=Ds, which="LA", return_eigenvectors=False)[0]
    o = np.argsort(lam)
    lam, V = lam[o], V[:, o]
    cn = S["cnode"].cpu().numpy()
    Vt = torch.as_tensor(V, device=dev)
    Dg = S["D"]
    info = []
    t1n, t2n, nn = T1.cpu().numpy(), T2.cpu().numpy(), NR.cpu().numpy()
    for q in range(k):
        vv = V[:, q].reshape(M, 3)
        e = np.einsum("ma,mab,mb->m", vv, D, vv)
        et = e.sum()
        fc = e[cn].sum() / et
        tg = (vv[cn] @ t1n) ** 2 + (vv[cn] @ t2n) ** 2
        nr = (vv[cn] @ nn) ** 2
        ftg = tg.sum() / max((tg + nr).sum(), 1e-300)
        info.append(dict(lam=float(lam[q]), f_contact=float(fc), f_tangent_in_contact=float(ftg)))
    return lam, float(lmax), info, Vt, time.time() - t0


for t in a.t:
    S = build(t)
    M = S["M"]
    n3 = 3 * M
    A = to_csr(S["I"], S["J"], S["Bk"], n3)
    D = S["Bh"][:, 0].double()
    S["D"] = D
    S["Dinv"] = torch.linalg.inv(D)
    b = S["b"]
    print(f"\n[t={t}] 노드 {M}, 접촉 노드 {int(S['cnode'].sum())} (입자 {S['ncp']} 이 s ≤ {a.c_eps}), 바닥 후보 입자 {S['nnear']}, nnz {A.values().numel()}",
          flush=True)
    bjp = lambda r: blk_apply(S["Dinv"], r)
    x0, it0, tr0 = pcg(A, bjp, b)
    print(f"  bj (기준): {it0} 회, 실제 잔차 {tr0:.2e}", flush=True)
    row = dict(t=t, M=M, n_contact=int(S["cnode"].sum()), bj=it0)
    eV = None
    if a.eig > 0:
        try:
            lam, lmax, info, eV, te = eig_low(S, A, a.eig)
            print(f"  [고유] D⁻¹A 가장 작은 {a.eig} 개 ({te:.1f} s): λ_max {lmax:.3e}, κ ≈ {lmax / lam[0]:.3e}", flush=True)
            for q, inf in enumerate(info):
                print(f"    λ{q:2d} {inf['lam']:.3e}  접촉 노드 몫 {inf['f_contact']:.3f}  (접촉 노드 안 접선 몫 {inf['f_tangent_in_contact']:.3f})", flush=True)
            row["eig"] = dict(lmax=lmax, low=info)
        except Exception as ex:                                     # noqa: BLE001
            print(f"  [고유] 실패: {ex!r}", flush=True)
    for vn in [q for q in a.variants.split(",") if q]:
        t0 = time.time()
        extra = {}
        try:
            if vn == "bj":
                continue
            if vn.startswith("bjS"):
                s = int(vn[3])
                pr, extra = bigblock(S, A, s, vn.endswith("c"))
                xx, it, tr = pcg(A, pr, b)
            elif vn in ("ssor", "dilu"):
                pr, extra = sweeps(S, A, vn)
                xx, it, tr = pcg(A, pr, b)
            elif vn[:3] in ("add", "def") or vn[:2] in ("dc", "ac"):
                if vn.startswith("defl"):
                    s, md, kind = int(vn[4:-1]), vn[-1], "defl"
                    agg, na = agg_of(S["gi"], s)
                elif vn.startswith("add"):
                    s, md, kind = int(vn[3:-1]), vn[-1], "add"
                    agg, na = agg_of(S["gi"], s)
                else:
                    s, md, kind = int(vn[2:-1]), vn[-1], ("defl" if vn.startswith("dc") else "add")
                    agg, na = agg_of(S["gi"], s, S["cnode"])
                Z = zbasis(S, agg, na, {"t": "t", "r": "r", "n": "t3"}[md] if md != "t" or vn[:2] not in ("dc", "ac") else "tg")
                AZ, Einv, nnull = coarse(A, Z)
                extra = dict(k=int(Z.shape[1]), n_agg=na, null=nnull)
                if kind == "defl":
                    xx, it, tr = pcg(A, bjp, b, Z=Z, AZ=AZ, Einv=Einv)
                else:
                    pr = lambda r, Z=Z, Einv=Einv: bjp(r) + Z @ (Einv @ (Z.t() @ r))
                    xx, it, tr = pcg(A, pr, b)
                del Z, AZ, Einv
            elif vn.startswith("eig") and eV is not None:
                k = int(vn[3:])
                Z = eV[:, :k].contiguous()
                AZ, Einv, nnull = coarse(A, Z)
                xx, it, tr = pcg(A, bjp, b, Z=Z, AZ=AZ, Einv=Einv)
                extra = dict(k=k)
            elif vn.startswith("mix_"):
                xx, it, tr, nup = mix_emul(S, A, b, vn[4:])
                extra = dict(n_upd=nup)
            else:
                continue
        except Exception as ex:                                     # noqa: BLE001
            print(f"  {vn}: 실패 {ex!r}", flush=True)
            row[vn] = dict(fail=repr(ex))
            torch.cuda.empty_cache()
            continue
        dx = float((xx - x0).norm() / x0.norm())
        print(f"  {vn}: {it} 회 ({it / it0:.2f} x 기준), 실제 잔차 {tr:.2e}, 해 차 {dx:.1e}  {extra}  [{time.time() - t0:.1f} s]", flush=True)
        row[vn] = dict(it=it, res=tr, dx=dx, **extra)
        torch.cuda.empty_cache()
    RES.append(row)
    json.dump(RES, open(os.path.join(a.out, "prec_bench.json"), "w"), ensure_ascii=False, indent=1)
print("[끝] prec_bench", flush=True)
