"""절 CM176: 2D 모래 외팔보 numpy 모형 -- 뉴턴 변형 비교 (초기 강성 / ct1 / 반매끈 정확 접선 + 전역화).
고정 격자 (Q1, 가우스 2x2), 점마다 탄성 F 상태, 시험 X = F + A F (A = Σ y ⊗ ∇w), Drucker-Prager 반환 (헨키 변형률, 꼭짓점 응력 0, 꼬리 = 편차만 줄임) = 3D 코드 ao_dpf 와 같은 사상.
왼쪽 열 노드 고정, 나머지 자유, 중력. 한 걸음 (또는 여러 걸음 + F 갱신) 의 뉴턴을 반복별 기록."""
import numpy as np, sys, json, argparse

def build(nx, ny, h, rho, E, nu, phi_deg=35.0):
    lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
    s = np.sin(np.radians(phi_deg)); al = np.sqrt(2/3)*2*s/(3-s)
    cdp = (3*lam+2*mu)/(2*mu)*al
    nid = lambda i, j: i*(ny+1)+j
    N = (nx+1)*(ny+1)
    g = 1/np.sqrt(3); gp = [(-g,-g),(g,-g),(-g,g),(g,g)]
    conn, GW = [], []
    for i in range(nx):
        for j in range(ny):
            nodes = [nid(i,j), nid(i+1,j), nid(i,j+1), nid(i+1,j+1)]
            for (xi, et) in gp:
                # Q1 shape grads (reference [-1,1]^2 -> physical h)
                dN = np.array([[-(1-et), -(1-xi)], [(1-et), -(1+xi)], [-(1+et), (1-xi)], [(1+et), (1+xi)]])/4.0 * (2.0/h)
                conn.append(nodes); GW.append(dN)
    conn = np.array(conn); GW = np.array(GW)
    Vq = np.full(len(conn), h*h/4.0)
    mass = np.zeros(N)
    for q in range(len(conn)):
        for o in range(4): mass[conn[q, o]] += rho*Vq[q]/4.0
    return dict(nx=nx, ny=ny, h=h, N=N, conn=conn, GW=GW, Vq=Vq, mass=mass, lam=lam, mu=mu, cdp=cdp, nid=nid)

def tau_map(S, M, mode):
    """S (...,2) 특이값 -> 키르히호프 응력 (...,2), 상태 (0 탄성 1 꼬리 2 꼭짓점). mode 'el' = 탄성만"""
    lam, mu, cdp = M['lam'], M['mu'], M['cdp']
    ep = np.log(np.maximum(S, 1e-14)); tr = ep.sum(-1); e = ep - tr[..., None]/2; rho = np.linalg.norm(e, axis=-1)
    tel = 2*mu*ep + lam*tr[..., None]
    if mode == 'el': return tel, np.zeros(tr.shape, int)
    dg = rho + cdp*tr
    st = np.where(dg > 0, np.where(tr > 0, 2, 1), 0)
    sc = np.where(rho > 0, -cdp*tr/np.maximum(rho, 1e-30), 0.0)
    tcone = (lam+mu)*tr[..., None] + 2*mu*sc[..., None]*e
    tau = np.where((st == 0)[..., None], tel, np.where((st == 1)[..., None], tcone, 0.0))
    return tau, st

def P_map(X, M, mode='dp'):
    U, S, Vt = np.linalg.svd(X)
    tau, st = tau_map(S, M, mode)
    P = np.einsum('qai,qi,qib->qab', U, tau/S, Vt)
    return P, st

def forces(y, Fq, M, mode='dp'):
    """내력 (노드 기울기 G = +dΨ/dy 부호: 잔차에 더함). f_node = V P Fᵀ ∇w (힘의 반대 = 내부 저항)"""
    Y = y.reshape(-1, 2)
    A = np.einsum('qoa,qoc->qac', Y[M['conn']], M['GW'])
    X = Fq + np.einsum('qac,qcb->qab', A, Fq)
    P, st = P_map(X, M, mode)
    Qf = M['Vq'][:, None, None]*np.einsum('qae,qce->qac', P, Fq)
    g = np.zeros((M['N'], 2))
    np.add.at(g, M['conn'], np.einsum('qac,qoc->qoa', Qf, M['GW']))
    return g.reshape(-1), X, st

def tangent(y, Fq, M, mode, fd=1e-6):
    """모드: 'el' 탄성 접선 (초기 강성법, 현재 X 에서) / 'ct1' 꼭짓점만 0 / 'ct2' 정확 접선 (활성 집합의 일반화 미분) ; 모두 PSD 투영 (코드 dm=max(D,0))"""
    Y = y.reshape(-1, 2)
    A = np.einsum('qoa,qoc->qac', Y[M['conn']], M['GW'])
    X = Fq + np.einsum('qac,qcb->qab', A, Fq)
    nq = len(X)
    pm = 'el' if mode == 'el' else 'dp'
    _, st = P_map(X, M, 'dp')
    T = np.zeros((nq, 4, 4))
    for k in range(4):
        d = np.zeros((2, 2)).ravel(); d[k] = fd*max(1.0, np.abs(X).max()); d = d.reshape(2, 2)
        Pp, _ = P_map(X + d, M, pm); Pm, _ = P_map(X - d, M, pm)
        T[:, :, k] = ((Pp - Pm)/(2*d.ravel()[k])).reshape(nq, 4)
    if mode == 'ct1': T[st == 2] = 0.0
    if mode != 'ex':
        T = 0.5*(T + T.transpose(0, 2, 1))
        w, V = np.linalg.eigh(T); w = np.maximum(w, 0.0)
        T = np.einsum('qij,qj,qkj->qik', V, w, V)
    T = T.reshape(nq, 2, 2, 2, 2)  # a e b g
    R = np.einsum('qaebg,qce,qdg->qacbd', T, Fq, Fq)
    Kl = np.einsum('q,qoc,qacbd,qpd->qoapb', M['Vq'], M['GW'], R, M['GW'])  # q o a p b
    K = np.zeros((M['N']*2, M['N']*2))
    for q in range(nq):
        idx = (M['conn'][q][:, None]*2 + np.arange(2)[None, :]).ravel()
        K[np.ix_(idx, idx)] += Kl[q].reshape(8, 8)
    return K, st

def retmap_F(X, M):
    U, S, Vt = np.linalg.svd(X)
    lam, mu, cdp = M['lam'], M['mu'], M['cdp']
    ep = np.log(np.maximum(S, 1e-14)); tr = ep.sum(-1); e = ep - tr[..., None]/2; rho = np.linalg.norm(e, axis=-1)
    dg = rho + cdp*tr; st = np.where(dg > 0, np.where(tr > 0, 2, 1), 0)
    sc = np.where(rho > 0, -cdp*tr/np.maximum(rho, 1e-30), 0.0)
    epn = np.where((st == 0)[:, None], ep, np.where((st == 1)[:, None], tr[:, None]/2 + sc[:, None]*e, 0.0))
    return np.einsum('qai,qi,qib->qab', U, np.exp(epn), Vt)

def solve_step(M, Fq, Ut, free, y0, mode, glob, dt, nit, tol, rec):
    """반복: r(y) = m (y − Ũ)/dt² + g(y) (g = 내부 저항 기울기: V P Fᵀ ∇w 합, 부호 +), 걸음 H d = −r.  glob: 'none' | 'arm' (비단조 Armijo 잔차)"""
    m = np.repeat(M['mass'], 2); it2 = 1.0/dt**2
    y = y0.copy(); y[~free] = 0.0
    def res(yy):
        g, X, st = forces(yy, Fq, M); r = m*(yy - Ut)*it2 + g; r[~free] = 0.0; return r, st
    r, st = res(y); hist = []
    r0 = np.sqrt(np.sum(r**2/(m*it2))) 
    for k in range(nit):
        nr = np.sqrt(np.sum(r**2/(m*it2)))
        hist.append(dict(k=k, nr=nr, a=0, nst=np.bincount(st, minlength=3).tolist()))
        if nr <= tol*r0 and k > 0: break
        K, _ = tangent(y, Fq, M, mode)
        H = K + np.diag(m*it2)
        d = np.zeros_like(y); d[free] = np.linalg.solve(H[np.ix_(free, free)], -r[free])
        a = 1.0
        if glob == 'arm':
            for _ in range(30):
                rn, stn = res(y + a*d); nn = np.sqrt(np.sum(rn**2/(m*it2)))
                if nn <= (1 - 1e-4*a)*nr: break
                a *= 0.5
            hist[-1]['a'] = a
        y = y + a*d; r, st = res(y)
    nr = np.sqrt(np.sum(r**2/(m*it2))); hist.append(dict(k=len(hist), nr=nr, a=0, nst=np.bincount(st, minlength=3).tolist()))
    return y, hist

def run(args, mode, glob, nit, tol=1e-6, nsteps=3):
    M = build(args.nx, args.ny, args.h, args.rho, args.E, args.nu)
    nq = len(M['conn']); Fq = np.tile(np.eye(2), (nq, 1, 1))
    free = np.ones(M['N']*2, bool)
    ncol = args.nx+1 if args.sup < 0 else args.sup+1           # --sup K: 바닥 고정은 앞 K 칸만 (나머지는 허공에 걸침)
    for i in range(ncol): free[M['nid'](i, 0)*2:M['nid'](i, 0)*2+2] = False   # 바닥 행 고정
    dt = args.dt; g = np.zeros(M['N']*2); g[1::2] = -9.8
    v = np.zeros(M['N']*2); v[1::2] = -args.v0; v[~free] = 0; out = []
    for s in range(nsteps):
        Ut = dt*v + dt*dt*g; Ut[~free] = 0
        y, hist = solve_step(M, Fq, Ut, free, Ut.copy(), mode, glob, dt, nit, tol, None)
        # F 갱신 (반환 사상) + 속도
        Y = y.reshape(-1, 2); A = np.einsum('qoa,qoc->qac', Y[M['conn']], M['GW']); X = Fq + np.einsum('qac,qcb->qab', A, Fq)
        Fq = retmap_F(X, M); v = y/dt; v[~free] = 0
        top = v.reshape(-1, 2)[[M['nid'](M['nx'], j) for j in range(M['ny']+1)], 1].mean()/dt/9.8
        out.append(dict(step=s, top_vy=top, hist=hist))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=8); ap.add_argument('--ny', type=int, default=24)
    ap.add_argument('--h', type=float, default=0.02); ap.add_argument('--rho', type=float, default=2200.)
    ap.add_argument('--E', type=float, default=3.5e8); ap.add_argument('--nu', type=float, default=0.3)
    ap.add_argument('--dt', type=float, default=1/60); ap.add_argument('--nsteps', type=int, default=2)
    ap.add_argument('--nit', type=int, default=40); ap.add_argument('--v0', type=float, default=1.0); ap.add_argument('--sup', type=int, default=-1)
    a = ap.parse_args()
    for mode, glob in [('el', 'none'), ('ct1', 'none'), ('ct2', 'none'), ('ct2', 'arm')]:
        o = run(a, mode, glob, a.nit, nsteps=a.nsteps)
        for s in o:
            h = s['hist']; ns = [x['nr'] for x in h]
            print(f"{mode:4s} {glob:4s} step{s['step']} iters={len(h)-1} nr0={ns[0]:.3e} nr[5]={ns[min(5,len(ns)-1)]:.3e} nr[-1]={ns[-1]:.3e} far_acc/g={s['top_vy']:.3f} apex@end={h[-1]['nst'][2]} cone={h[-1]['nst'][1]}", flush=True)


def solve_homotopy(M, Fq, Ut, free, y0, mode, dt, tol=1e-6, maxJ=400):
    """경로 추종: r(y) − (1−λ) r(y_0) = 0, λ 0→1 (적응 보폭, 각 점 뉴턴 ≤ 8 반복·잔차 tol_λ 이내); 야코비안 풀이 횟수 반환"""
    m = np.repeat(M['mass'], 2); it2 = 1.0/dt**2
    def res(yy):
        g, X, st = forces(yy, Fq, M); r = m*(yy - Ut)*it2 + g; r[~free] = 0.0; return r, st
    y = y0.copy(); y[~free] = 0.0
    r0, _ = res(y); nr0 = np.sqrt(np.sum(r0**2/(m*it2)))
    lam, dl, nJ, trace = 0.0, 0.25, 0, []
    while lam < 1.0 and nJ < maxJ:
        l2 = min(1.0, lam + dl); yt = y.copy(); ok = False
        for k in range(8):
            r, st = res(yt); r = r - (1 - l2)*r0; r[~free] = 0
            nr = np.sqrt(np.sum(r**2/(m*it2)))
            if nr <= tol*nr0: ok = True; break
            K, _ = tangent(yt, Fq, M, mode); nJ += 1
            H = K + np.diag(m*it2); d = np.zeros_like(y); d[free] = np.linalg.solve(H[np.ix_(free, free)], -r[free]); yt = yt + d
        if ok: y, lam, dl = yt, l2, min(dl*2, 1.0)
        else: dl *= 0.5
        trace.append((lam, dl, nJ))
        if dl < 1e-4: break
    r, st = res(y); return y, np.sqrt(np.sum(r**2/(m*it2)))/nr0, nJ, lam
