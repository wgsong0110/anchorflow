"""Worsey–Farin C1 3 차 매크로 요소 (--interp c1wf).

Kuhn 사면체 하나를 내심 z 와 면마다 한 점 w_f 로 12 조각으로 나누고, 조각마다
3 차 Bernstein–Bézier 다항식을 둔다. 자유도 (성분마다):
  꼭짓점: 값 1 + 격자좌표 기울기 ∂u/∂y 3         -> 4 x 4 = 16
  변:     중점에서 변에 가로지르는 미분 2 방향      -> 6 x 2 = 12      (합 28)
변 방향은 Kuhn 격자에서 7 종류(0/1 벡터)뿐이고, 가로 방향 쌍은 종류마다 고정
(_TRANS) 이라 변을 공유하는 모든 사면체가 같은 자유도를 본다.

w_f 는 면을 공유하는 두 사면체의 내심을 잇는 선분이 그 면과 만나는 점이다
(Worsey–Farin 조건; 이래야 면을 가로질러 C1). Kuhn 분할은 축 순열·평행이동에
불변이라 6 종류 사면체 표만 있으면 되고, 격자좌표 y 에서 만들면 월드 좌표로의
아핀 사상이 C1 을 그대로 지킨다.

노드 출력 채널 (C1_NOUT = 54):
  [0:3]   u (변위)
  [3:12]  ∂u_c/∂y_d  (c*3 + d)
  [12:54] 변 (종류 k, 방향 s, 성분 c) -> 12 + (k*2 + s)*3 + c, 변은 **아래쪽 끝 노드**가 든다

det(I + ∇u) 는 조각마다 6 차 다항식 -> Bernstein 계수 84 개. 계수가 모두 양수면
조각 전체에서 det>0 (볼록 결합). 표는 import 때 numpy 로 한 번 만든다.
"""
import itertools
import math

import numpy as np

C1_NOUT = 54
CLASSES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], float)


def _trans_pairs():
    E = np.eye(3); out = []
    for d in CLASSES:
        for a_, b_ in ((0, 1), (0, 2), (1, 2)):
            if abs(np.linalg.det(np.stack([d, E[a_], E[b_]]))) > 0.5:
                out.append((E[a_], E[b_])); break
    return np.array(out)                         # [7,2,3]


_TRANS = _trans_pairs()
PL_DEDUP = [False]     # pl_c1_dofs 중복 쓰기 원천 하나만 (측정용, 기본 꺼짐 = 실행 그대로)


def multi4(n):
    return [m for m in itertools.product(range(n + 1), repeat=4) if sum(m) == n][::-1]


I1, I2, I3, I4, I6 = multi4(1), multi4(2), multi4(3), multi4(4), multi4(6)
ID3 = {m: k for k, m in enumerate(I3)}


def mcoef(m):
    return math.factorial(sum(m)) / np.prod([math.factorial(x) for x in m])


def bern_np(n, L):
    L = np.atleast_2d(L)
    return np.stack([mcoef(m) * np.prod([L[:, i] ** m[i] for i in range(4)], 0) for m in multi4(n)], 1)


def _add(m, e):
    return tuple(x + y for x, y in zip(m, e))


EYE4 = [tuple(int(i == j) for j in range(4)) for i in range(4)]

PERMS = list(itertools.permutations(range(3)))  # 사면체 종류 = 축 순열 (locate 의 perm)


def chain(p):
    v = [np.zeros(3)]
    for r in range(3):
        v.append(v[-1] + np.eye(3)[p[r]])
    return np.array(v)


def incenter(V):
    A = []
    for i in range(4):
        f = [V[j] for j in range(4) if j != i]
        A.append(0.5 * np.linalg.norm(np.cross(f[1] - f[0], f[2] - f[0])))
    A = np.array(A); return (A[:, None] * V).sum(0) / A.sum()


def neighbor_vertex(V, i):
    """Kuhn 뒤집기: 면 (i 제외) 건너편 사면체의 새 꼭짓점."""
    return V[(i - 1) % 4] + V[(i + 1) % 4] - V[i]      # 순환 첨자 (Freudenthal 피벗)


def bary4(P, x):
    """사면체 P[4,3] 에서 점 x 의 barycentric."""
    A = np.vstack([P.T, np.ones(4)]); return np.linalg.solve(A, np.r_[x, 1.0])


def build_type(p):
    """종류 p 하나의 WF 요소 -> dict(sub [12,4,3], ord [12,20,28], face_bary, ...)"""
    V = chain(p); z = incenter(V)
    W = {}
    for i in range(4):
        F = [V[j] for j in range(4) if j != i]
        Vn = np.array(F + [neighbor_vertex(V, i)]); zn = incenter(Vn)
        # z + t (zn - z) 가 면 평면 위
        nrm = np.cross(F[1] - F[0], F[2] - F[0]); t = np.dot(F[0] - z, nrm) / np.dot(zn - z, nrm)
        w = z + t * (zn - z)
        lb = np.linalg.lstsq(np.vstack([np.array(F).T, np.ones(3)]), np.r_[w, 1.0], rcond=None)[0]
        assert lb.min() > 1e-6, ("면 점이 면 밖", p, i, lb)
        W[i] = (w, lb)
    subs = []                                    # (a, b, w_i, z), a,b 는 면 i 의 꼭짓점 번호
    sub_lbl = []
    for i in range(4):
        f = [j for j in range(4) if j != i]
        for k in range(3):
            a_, b_ = f[k], f[(k + 1) % 3]
            subs.append(np.array([V[a_], V[b_], W[i][0], z])); sub_lbl.append((a_, b_, i))
    subs = np.array(subs)
    # 영역점
    key = lambda x: tuple(np.round(x * 3e6).astype(np.int64))
    pts = {}; sidx = np.zeros((12, 20), int)
    for s, S in enumerate(subs):
        for k, m in enumerate(I3):
            x = (np.array(m)[:, None] * S).sum(0) / 3.0
            kk = key(x)
            if kk not in pts: pts[kk] = len(pts)
            sidx[s, k] = pts[kk]
    U = len(pts)
    rows, rhs = [], []                           # rhs: 28 열
    # C1 조건 (조각끼리 공유하는 면)
    for s1 in range(12):
        for s2 in range(s1 + 1, 12):
            k1 = {key(x): j for j, x in enumerate(subs[s1])}
            sh = [key(x) for x in subs[s2] if key(x) in k1]
            if len(sh) != 3: continue
            e2 = [j for j, x in enumerate(subs[s2]) if key(x) not in k1][0]
            lam = bary4(subs[s1], subs[s2][e2])
            sh1 = [k1[kk] for kk in sh]                       # s1 의 공유 꼭짓점 번호
            sh2 = [j for j, x in enumerate(subs[s2]) if key(x) in k1]
            for mu in itertools.product(range(3), repeat=3):
                if sum(mu) != 2: continue
                # s2 쪽: mu on shared + 1 at e2 ; 점으로 찾는다
                base = sum(mu[q] * subs[s1][sh1[q]] for q in range(3))
                r = np.zeros(U)
                r[pts[key((base + subs[s2][e2]) / 3.0)]] -= 1
                for j in range(4):
                    r[pts[key((base + subs[s1][j]) / 3.0)]] += lam[j]
                rows.append(r); rhs.append(np.zeros(28))
    nC = len(rows)
    # 보간 조건. 자유도 번호: 꼭짓점 i -> 4i (값), 4i+1+d (∂/∂y_d); 변 e(i<j) -> 16 + 2e + s
    EDG = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    def sub_with(*verts):
        for s, S in enumerate(subs):
            ks = {key(x) for x in S}
            if all(key(v) in ks for v in verts): return s
        raise RuntimeError
    for i in range(4):
        r = np.zeros(U); r[pts[key(V[i])]] = 1; e = np.zeros(28); e[4 * i] = 1; rows.append(r); rhs.append(e)
        s = sub_with(V[i]); S = subs[s]
        for q in S:
            if np.allclose(q, V[i]): continue
            r = np.zeros(U); r[pts[key((2 * V[i] + q) / 3)]] += 3; r[pts[key(V[i])]] -= 3
            e = np.zeros(28); e[4 * i + 1:4 * i + 4] = q - V[i]; rows.append(r); rhs.append(e)
    cls_of = []
    for ei, (i, j) in enumerate(EDG):
        d = V[j] - V[i]; c = int(np.where((np.abs(CLASSES - d).sum(1) < 1e-9))[0][0]); cls_of.append(c)
        mid = 0.5 * (V[i] + V[j]); s = sub_with(V[i], V[j]); S = subs[s]
        for sdir in range(2):
            t = _TRANS[c, sdir]
            # 방향 t 의 barycentric 방향좌표
            dl = np.linalg.solve(np.vstack([S.T, np.ones(4)]), np.r_[t, 0.0])
            Lm = bary4(S, mid); B2 = bern_np(2, Lm)[0]
            r = np.zeros(U)
            for g_, gm in enumerate(I2):
                for j4 in range(4):
                    r[pts[key((np.array(_add(gm, EYE4[j4]))[:, None] * S).sum(0) / 3)]] += 3 * B2[g_] * dl[j4]
            e = np.zeros(28); e[16 + 2 * ei + sdir] = 1; rows.append(r); rhs.append(e)
    A = np.array(rows); B = np.array(rhs)
    sol, res, rk, sv = np.linalg.lstsq(A, B, rcond=None)
    assert rk == U, ("WF 계가 유일해를 안 준다", rk, U)
    resid = float(np.abs(A @ sol - B).max())
    assert resid < 1e-9, ("WF 계 잔차", resid)
    ordm = sol[sidx]                             # [12,20,28]
    # 기울기 Bernstein (y 좌표) : [12,10,3,28]
    GD = np.zeros((12, 10, 3, 28))
    for s, S in enumerate(subs):
        Em = np.stack([S[1] - S[0], S[2] - S[0], S[3] - S[0]], 1); Ei = np.linalg.inv(Em)
        for g_, gm in enumerate(I2):
            D = np.stack([3 * (ordm[s, ID3[_add(gm, EYE4[k])]] - ordm[s, ID3[_add(gm, EYE4[0])]]) for k in (1, 2, 3)], 0)  # [3(edge),28]
            GD[s, g_] = Ei.T @ D                 # ∇_y = Ei^T [D1 D2 D3]
    Hinv = np.stack([np.linalg.inv(np.vstack([S.T, np.ones(4)])) for S in subs])   # [12,4,4]
    return dict(V=V, z=z, W=W, sub=subs, ord=ordm, GD=GD, Hinv=Hinv, cls=np.array(cls_of), EDG=EDG, U=U, nC=nC,
                resid=resid, sub_lbl=sub_lbl)


TYPES = [build_type(p) for p in PERMS]


def _prod_table(na, nb):
    Ia, Ib, Ic = multi4(na), multi4(nb), multi4(na + nb)
    ic = {m: k for k, m in enumerate(Ic)}
    A, B, C, Wt = [], [], [], []
    for a_, ma in enumerate(Ia):
        for b_, mb in enumerate(Ib):
            mc = _add(ma, mb); A.append(a_); B.append(b_); C.append(ic[mc]); Wt.append(mcoef(ma) * mcoef(mb) / mcoef(mc))
    return np.array(A), np.array(B), np.array(C), np.array(Wt), len(Ic)


P22 = _prod_table(2, 2)
P42 = _prod_table(4, 2)


def perm_type_index():
    """locate 의 perm (순위->축) 을 종류 번호로: 키 perm0*3+perm1."""
    t = -np.ones(9, int)
    for k, p in enumerate(PERMS):
        t[p[0] * 3 + p[1]] = k
    return t


# --------------------------------------------------------------- numpy 평가 (시험용)
def np_eval(ti, yl, dofs):
    """종류 ti, 셀 국소 좌표 yl [P,3], 국소 자유도 dofs [28, C] -> 값 [P,C], ∇_y [P,C,3], 조각 [P]."""
    T = TYPES[ti]
    H = np.einsum("sij,pj->psi", T["Hinv"], np.c_[yl, np.ones(len(yl))])   # [P,12,4]
    s = H.min(2).argmax(1); L = H[np.arange(len(yl)), s]
    val = np.einsum("pk,pkn,nc->pc", bern_np(3, L), T["ord"][s], dofs)
    gr = np.einsum("pk,pkdn,nc->pcd", bern_np(2, L), T["GD"][s], dofs)
    return val, gr, s


# --------------------------------------------------------------- torch
_TT = {}


def tables(device, dtype):
    import torch
    k = (str(device), dtype)
    if k not in _TT:
        f = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=dtype)
        li = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=torch.long)
        _TT[k] = dict(
            ord=f(np.stack([t["ord"] for t in TYPES])),          # [6,12,20,28]
            GD=f(np.stack([t["GD"] for t in TYPES])),            # [6,12,10,3,28]
            Hinv=f(np.stack([t["Hinv"] for t in TYPES])),        # [6,12,4,4]
            cls=li(np.stack([t["cls"] for t in TYPES])),         # [6,6]
            edg=li(np.array(TYPES[0]["EDG"])),                   # [6,2] (모든 종류 같은 순서)
            ptype=li(perm_type_index()),
            p22=tuple(li(x) for x in P22[:3]) + (f(P22[3]), P22[4]),
            p42=tuple(li(x) for x in P42[:3]) + (f(P42[3]), P42[4]),
            I2=li(np.array(I2)), I3=li(np.array(I3)), I6=li(np.array(I6)),
            mc2=f([mcoef(m) for m in I2]), mc3=f([mcoef(m) for m in I3]))
    return _TT[k]


def _bern_t(L, I, mc):
    import torch
    # L [N,4], I [K,4] -> [N,K]
    return mc * torch.prod(L.unsqueeze(1) ** I.unsqueeze(0).to(L.dtype), -1)


def particle_struct(q, lo, lat, nn, aux, rows):
    """입자마다: 종류 [N], 조각 [N], 값 가중 [N,28], 기울기 가중 [N,28,3] (y 좌표), 자유도 모으기 색인."""
    import torch
    T = tables(q.device, q.dtype)
    ci, rank, perm = aux[0], aux[1], aux[2]
    y = (q - lo) @ lat.Ai.T
    yl = (y - ci.to(y.dtype)).clamp(0.0, 1.0)
    ty = T["ptype"][perm[:, 0] * 3 + perm[:, 1]]
    Hh = torch.einsum("nsij,nj->nsi", T["Hinv"][ty], torch.cat([yl, torch.ones_like(yl[:, :1])], 1))   # [N,12,4]
    sb = Hh.min(2).values.argmax(1)
    L = Hh[torch.arange(q.shape[0], device=q.device), sb]                                                  # [N,4]
    wv = torch.einsum("nk,nkd->nd", _bern_t(L, T["I3"], T["mc3"]), T["ord"][ty, sb])                    # [N,28]
    wg = torch.einsum("nk,nkcd->ndc", _bern_t(L, T["I2"], T["mc2"]), T["GD"][ty, sb])                    # [N,28,3]
    return dict(ty=ty, sb=sb, wv=wv.detach(), wg=wg.detach(), L=L)


def gather_dofs(dpn, rows, ty):
    """노드 출력 [M,54] -> 입자(또는 사면체)별 국소 자유도 [N,28,3]. rows [N,4] (사슬 순서 v0..v3)."""
    import torch
    T = tables(dpn.device, dpn.dtype)
    N = rows.shape[0]
    nd = dpn[rows]                                                       # [N,4,54]
    val = nd[..., 0:3]                                                   # [N,4,3]
    gr = nd[..., 3:12].reshape(N, 4, 3, 3).transpose(-1, -2)             # [N,4,3(d),3(c)]
    vtx = torch.cat([val.unsqueeze(2), gr], 2).reshape(N, 16, 3)        # 꼭짓점 i: [값, d0, d1, d2]
    cls = T["cls"][ty]                                                   # [N,6]
    st = rows[:, T["edg"][:, 0]]                                         # [N,6] 아래쪽 끝 노드
    ch = 12 + (cls.unsqueeze(-1) * 2 + torch.arange(2, device=dpn.device)).unsqueeze(-1) * 3 \
        + torch.arange(3, device=dpn.device)                             # [N,6,2,3]
    ed = dpn[st.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 3), ch]    # [N,6,2,3]
    return torch.cat([vtx, ed.reshape(N, 12, 3)], 1)                     # [N,28,3]


def jac_pre(rows, ws, lat, dpn):
    """값 u [N,3], ∇_x u [N,3,3] (g2p_jac_pre 와 같은 규약: 단위행렬 없음)."""
    import torch
    D = gather_dofs(dpn, rows, ws["ty"])
    u = torch.einsum("nk,nkc->nc", ws["wv"].to(D.dtype), D)
    Gy = torch.einsum("nkd,nkc->ncd", ws["wg"].to(D.dtype), D)
    return u, Gy @ lat.Ai.to(Gy.dtype)


def tet_struct(rows, ty, lat):
    """점유 사면체 (중복 제거): rows [T,4], 종류 [T], 조각 부피 [T,12]."""
    import torch
    key = torch.cat([rows, ty.unsqueeze(1)], 1)
    uq, inv = torch.unique(key, dim=0, return_inverse=True)
    T = tables(rows.device, lat.A.dtype)
    trows, tty = uq[:, :4], uq[:, 4]
    detA = abs(float(torch.linalg.det(lat.A.double())))
    sv = []
    for t in TYPES:
        S = t["sub"]; sv.append([abs(np.linalg.det(np.stack([S_[1] - S_[0], S_[2] - S_[0], S_[3] - S_[0]], 1))) / 6 for S_ in S])
    svol = torch.as_tensor(np.array(sv), device=rows.device, dtype=lat.A.dtype)[tty] * detA   # [T,12]
    return dict(rows=trows, ty=tty, svol=svol)


def _bprod(a, b, tab):
    """Bernstein 곱: a [...,Ka], b [...,Kb] -> [...,Kc]."""
    import torch
    ia, ib, ic, w, nc = tab
    t = a[..., ia] * b[..., ib] * w
    out = torch.zeros(t.shape[:-1] + (nc,), device=t.device, dtype=t.dtype)
    return out.index_add_(t.dim() - 1, ic, t)


def det_coefs(dpn, ts, lat):
    """점유 사면체의 모든 조각에서 det(I + ∇_x u) 의 6 차 Bernstein 계수 [T,12,84]."""
    import torch
    T = tables(dpn.device, dpn.dtype)
    D = gather_dofs(dpn, ts["rows"], ts["ty"])                           # [T,28,3]
    Gy = torch.einsum("tsgdk,tkc->tsgcd", T["GD"][ts["ty"]], D)         # [T,12,10,3c,3d]
    Gx = Gy @ lat.Ai.to(Gy.dtype)
    F = Gx + torch.eye(3, device=dpn.device, dtype=dpn.dtype)          # 상수 I 의 Bernstein 계수는 모두 1
    F = F.permute(0, 1, 3, 4, 2)                                         # [T,12,3,3,10]
    p22, p42 = T["p22"], T["p42"]
    c0 = _bprod(F[..., 1, 1, :], F[..., 2, 2, :], p22) - _bprod(F[..., 1, 2, :], F[..., 2, 1, :], p22)
    c1 = _bprod(F[..., 1, 2, :], F[..., 2, 0, :], p22) - _bprod(F[..., 1, 0, :], F[..., 2, 2, :], p22)
    c2 = _bprod(F[..., 1, 0, :], F[..., 2, 1, :], p22) - _bprod(F[..., 1, 1, :], F[..., 2, 0, :], p22)
    return _bprod(c0, F[..., 0, 0, :], p42) + _bprod(c1, F[..., 0, 1, :], p42) + _bprod(c2, F[..., 0, 2, :], p42)


def barrier_energy(c, svol, kappa, jhat):
    """진짜 로그 장벽 Σ_조각 (V_s/84) Σ_k κ b(c_k),  b(c) = -(c-ĉ)² ln(c/ĉ)  (c<ĉ), c<=0 이면 +inf."""
    import torch
    if bool((c <= 0).any()):
        return torch.full((), float("inf"), device=c.device, dtype=c.dtype)
    cc = torch.minimum(c, torch.full_like(c, jhat))
    b = -((cc - jhat) ** 2) * torch.log(cc / jhat)
    return kappa * ((svol / 84.0).unsqueeze(-1) * b).sum()


# --------------------------------------------------------------- 요소 말림 감시 (det>0 이어도 휘어진 요소가 스스로 겹치는 것)
def _face_tess(T):
    """요소 경계(4 면)를 WF 면 분할 3 삼각형 x 2 단계 세분 = 면당 12, 요소당 48 삼각형으로.
    -> 점 (조각, barycentric) [P], 삼각형 [Q,3], 변 [E,2], (변,삼각형) 유효쌍 마스크 [E,Q] (꼭짓점 공유 제외)."""
    key = lambda x: tuple(np.round(x * 3e6).astype(np.int64))
    pts, P_sub, P_L, tris = {}, [], [], []
    for s, (a_, b_, i) in enumerate(T["sub_lbl"]):
        S = T["sub"][s]                                   # (V_a, V_b, w_i, z) -- 앞 셋이 경계 면 위
        loc = {}
        for (p, q, r) in ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (0, 1, 1), (1, 0, 1)):
            L = np.array([p, q, r, 0]) / 2.0; x = L @ S; k = key(x)
            if k not in pts:
                pts[k] = len(pts); P_sub.append(s); P_L.append(L)
            loc[(p, q, r)] = pts[k]
        for tr in (((2, 0, 0), (1, 1, 0), (1, 0, 1)), ((1, 1, 0), (0, 2, 0), (0, 1, 1)), ((1, 0, 1), (0, 1, 1), (0, 0, 2)),
                   ((1, 1, 0), (0, 1, 1), (1, 0, 1))):
            tris.append([loc[t] for t in tr])
    tris = np.array(tris)
    E = sorted({tuple(sorted((t[j], t[(j + 1) % 3]))) for t in tris for j in range(3)})
    E = np.array(E)
    ok = np.ones((len(E), len(tris)), bool)
    for e, (p, q) in enumerate(E):
        for t, tr in enumerate(tris):
            if p in tr or q in tr:
                ok[e, t] = False
    return np.array(P_sub), np.array(P_L), tris, E, ok


_TESS = [_face_tess(t) for t in TYPES]


def tess_tables(device, dtype):
    import torch
    k = ("tess", str(device), dtype)
    if k not in _TT:
        P_sub = np.stack([t[0] for t in _TESS]); P_L = np.stack([t[1] for t in _TESS])
        W = np.stack([np.einsum("pk,pkn->pn", bern_np(3, P_L[ti]), TYPES[ti]["ord"][P_sub[ti]]) for ti in range(6)])   # [6,P,28]
        Y = np.stack([np.einsum("pj,pjd->pd", P_L[ti], TYPES[ti]["sub"][P_sub[ti]]) for ti in range(6)])               # [6,P,3] 셀 국소 y
        f = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=dtype)
        li = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=torch.long)
        _TT[k] = dict(W=f(W), Y=f(Y), tri=li(np.stack([t[2] for t in _TESS])), E=li(np.stack([t[3] for t in _TESS])),
                      ok=torch.as_tensor(np.stack([t[4] for t in _TESS]), device=device))
    return _TT[k]


def curl_check(dpn, wt, lo, lat, nn, uniq, chunk=512, eps=1e-6, coplanar=1e-6):
    """요소 말림 근사 검사: 각 점유 사면체의 상(경계 면 48 삼각형 근사)에서, 꼭짓점을 공유하지 않는 변-삼각형 쌍이
    교차하면 그 요소의 경계 면이 스스로 교차한 것(요소가 겹쳐 말림)으로 센다 (Möller–Trumbore, 엄격한 내부).
    -> (말린 요소 수, 교차 쌍 수)."""
    import torch
    with torch.no_grad():
        dpn = dpn.double()                                       # float32 에서는 같은 면 위 쌍이 반올림으로 '교차' 로 잡힌다
        lo = lo.double(); lat = type(lat)(lat.A.double(), lat.Ai.double(), lat.s, lat.hz)
        TT = tess_tables(dpn.device, dpn.dtype)
        nnl = [int(nn[k_]) for k_ in range(3)]
        fl = uniq[wt["rows"][:, 0]]
        ijk = torch.stack([fl // (nnl[1] * nnl[2]), (fl // nnl[2]) % nnl[1], fl % nnl[2]], -1).to(dpn.dtype)
        base = ijk @ lat.A.to(dpn.dtype).T + lo.to(dpn.dtype)
        n_el, n_pair = 0, 0
        for s0 in range(0, wt["rows"].shape[0], chunk):
            sl = slice(s0, s0 + chunk)
            ty = wt["ty"][sl]
            D = gather_dofs(dpn, wt["rows"][sl], ty)                                         # [C,28,3]
            X = base[sl].unsqueeze(1) + TT["Y"][ty] @ lat.A.to(dpn.dtype).T + torch.einsum("cpn,cnd->cpd", TT["W"][ty], D)
            tri, E, ok = TT["tri"][ty], TT["E"][ty], TT["ok"][ty]                            # [C,Q,3] [C,E,2] [C,E,Q]
            g = lambda idx: torch.gather(X, 1, idx.reshape(idx.shape[0], -1, 1).expand(-1, -1, 3)).reshape(idx.shape + (3,))
            T3 = g(tri)                                                                       # [C,Q,3,3]
            Pe = g(E)                                                                         # [C,E,2,3]
            o = Pe[:, :, 0].unsqueeze(2); dd = (Pe[:, :, 1] - Pe[:, :, 0]).unsqueeze(2)    # [C,E,1,3]
            v0, e1, e2 = T3[:, :, 0].unsqueeze(1), (T3[:, :, 1] - T3[:, :, 0]).unsqueeze(1), (T3[:, :, 2] - T3[:, :, 0]).unsqueeze(1)
            dd, e1, e2, v0 = torch.broadcast_tensors(dd, e1, e2, v0)                         # [C,E,Q,3]
            o = o.expand_as(dd)
            pv = torch.cross(dd, e2, dim=-1); det = (e1 * pv).sum(-1)                        # [C,E,Q]
            sc = (e1.norm(dim=-1) * e2.norm(dim=-1) * dd.norm(dim=-1)).clamp_min(1e-30)
            good = ok & (det.abs() > coplanar * sc)
            inv = 1.0 / torch.where(good, det, torch.ones_like(det))
            tv = o - v0; u = (tv * pv).sum(-1) * inv
            qv = torch.cross(tv, e1, dim=-1); vv = (dd * qv).sum(-1) * inv; tt = (e2 * qv).sum(-1) * inv
            hit = good & (u > eps) & (vv > eps) & (u + vv < 1 - eps) & (tt > eps) & (tt < 1 - eps)
            n_pair += int(hit.sum()); n_el += int(hit.flatten(1).any(1).sum())
        return n_el, n_pair


def pl_c1_dofs(val, wt, M):
    """노드 값 [M,3] 의 barycentric(PL) 장에서 C1 기울기·변 채널 [M,51] 을 만든다 (선형 연산, 미분 가능).
    노드 ∂u/∂y = 점유 사면체 PL 기울기의 평균(Kuhn 사면체는 부피가 모두 같다), 변 가로 미분 = 두 끝 노드 기울기 평균 · t
    (끝 노드가 점유 사면체에 없으면 시작 노드 것). 아핀 값이면 정확한 아핀 C1 자유도가 된다.
    --interp c1wf 의 기본 매개화: 기울기·변 채널 = 이 값 + 변수 (변수 0 이면 '매끄럽게 한 PL')."""
    import torch
    T = tables(val.device, val.dtype)
    rows, ty = wt["rows"], wt["ty"]
    perm = torch.as_tensor(np.array(PERMS), device=val.device)[ty]                       # [T,3] 순위 -> 축
    f = val[rows]                                                                          # [T,4,3]
    st = (f[:, 1:] - f[:, :-1])                                                            # [T,3(r),3(c)]
    Gy = torch.zeros_like(st).scatter_(1, perm.unsqueeze(-1).expand(-1, -1, 3), st)      # [T,3(d),3(c)] ∂u_c/∂y_d
    Gy = Gy.transpose(1, 2)                                                                # [T,3c,3d]
    cnt = torch.zeros(M, device=val.device, dtype=val.dtype).index_add_(0, rows.reshape(-1), torch.ones(rows.numel(), device=val.device, dtype=val.dtype))
    Gn = torch.zeros(M, 9, device=val.device, dtype=val.dtype).index_add_(
        0, rows.reshape(-1), Gy.reshape(-1, 1, 9).expand(-1, 4, -1).reshape(-1, 9)) / cnt.clamp_min(1.0).unsqueeze(-1)
    out = torch.cat([Gn, torch.zeros(M, 42, device=val.device, dtype=val.dtype)], 1)
    TR = torch.as_tensor(_TRANS, device=val.device, dtype=val.dtype)                     # [7,2,3]
    e0, e1 = T["edg"][:, 0], T["edg"][:, 1]
    sN, eN = rows[:, e0], rows[:, e1]                                                      # [T,6]
    cls = T["cls"][ty]                                                                     # [T,6]
    Gm = 0.5 * (Gn[sN] + Gn[eN]).reshape(sN.shape + (3, 3))                              # [T,6,3c,3d]
    dv = torch.einsum("tecd,tesd->tesc", Gm, TR[cls])                                      # [T,6,2,3c]
    ch = 9 + (cls.unsqueeze(-1) * 2 + torch.arange(2, device=val.device)).unsqueeze(-1) * 3 + torch.arange(3, device=val.device)
    idx_r = sN.unsqueeze(-1).unsqueeze(-1).expand_as(ch)
    _ir, _ic, _dv = idx_r.reshape(-1), ch.reshape(-1), dv.reshape(-1)
    if PL_DEDUP[0]:
        # 측정용 (AF_VARGRAD): 같은 (노드, 채널) 자리에 여러 사면체가 같은 값을 쓴다. index_put(accumulate=False) 의 역전파는
        # 중복된 모든 원천에 기울기를 다 주어 그 경로의 기울기를 중복 수만큼 부풀린다. 첫 원천 하나만 남겨 참 기울기를 만든다
        _lin = _ir * out.shape[1] + _ic
        _u, _inv = torch.unique(_lin, return_inverse=True)
        _first = torch.full((_u.numel(),), _lin.numel(), device=_lin.device, dtype=torch.long).scatter_reduce(
            0, _inv, torch.arange(_lin.numel(), device=_lin.device), reduce="amin")
        _ir, _ic, _dv = _ir[_first], _ic[_first], _dv[_first]
    out = out.index_put((_ir, _ic), _dv)              # 같은 변은 같은 값 (끝 노드로만 정해짐)
    return out
