"""앵커가 (회전벡터, 로그신축) 을 상태로 들고 가우시안의 F 를 만든다. 그 역도 푼다.

`anchor_frame.AnchorFrame` 은 회전을 쿼터니언으로 들고, 역문제를 쿼터니언 4 성분의
최소제곱으로 푼 뒤 앵커마다 정규화했다. 그 정규화가 최소제곱을 무효로 만든다 --
역블러링은 병조건이라 해가 크게 진동하는데, 진동하는 4 벡터를 개별 정규화하면
방향이 통째로 바뀐다. 실측으로 그 인코더의 F 잔차는 항등원의 99.3% 로, 아무것도
담지 못했다(같은 상태를 경사하강으로 다듬으면 39.5% 까지 간다).

여기서는 두 가지를 바꾼다.

1. 회전을 **회전벡터** u_a in R^3 으로 든다. 정규화 단계가 없어 위 문제가 원인부터
   사라지고, 블렌딩이 상태에 대해 정확히 선형이 된다.

       u_g = sum_a w_ga u_a,   s_g = sum_a w_ga s_a,   F_g = exp([u_g]x) diag(e^{s_g})

2. 인코더가 (u, s) **각자의 공간**이 아니라 **F 위에서** 푼다:

       min_{u,s}  sum_g m_g || F_g(u,s) - F_g^MPM ||_F^2

   가우스-뉴턴이고, 야코비안은 짝 목록을 통해 희소하다. 미지수가 6M(=3072) 뿐이라
   3072x3072 을 조립할 수도 있지만, 쌍-쌍 누적이 4e8 개라 무행렬 켤레기울기로 푼다.

증분의 미분은 좌섭동 convention 이다: exp(u+d) ~ exp([J_l(u) d]x) exp(u) 이므로

       K z = [J_l(u_g) z_u]x F  +  F diag(z_s)

이고, 그 수반은 z_u <- J_l^T sum_k F[:,k] x Y[:,k],  z_s <- F[:,k] . Y[:,k] 이다.
"""
from __future__ import annotations

import torch


def skew(a):
    """[...,3] -> [...,3,3], skew(a) b = a x b"""
    z = torch.zeros_like(a[..., 0])
    return torch.stack([
        torch.stack([z, -a[..., 2], a[..., 1]], -1),
        torch.stack([a[..., 2], z, -a[..., 0]], -1),
        torch.stack([-a[..., 1], a[..., 0], z], -1)], -2)


def expmap(u):
    """회전벡터 -> 회전행렬. 로드리게스, 원점 근처는 급수."""
    th2 = u.pow(2).sum(-1, keepdim=True).clamp(min=0)
    th = th2.clamp(min=1e-24).sqrt()
    small = th2 < 1e-12
    a = torch.where(small, 1.0 - th2 / 6.0, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24.0, (1.0 - torch.cos(th)) / th2.clamp(min=1e-24))
    S = skew(u)
    eye = torch.eye(3, device=u.device, dtype=u.dtype).expand_as(S)
    return eye + a.unsqueeze(-1) * S + b.unsqueeze(-1) * (S @ S)


def logmap(R):
    """회전행렬 -> 회전벡터."""
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    c = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    th = torch.acos(c).unsqueeze(-1)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], -1)
    sn = torch.sin(th)
    # th->0 이면 v/(2 sin th) -> v/2; th->pi 는 sin 이 0 이라 위험하므로 clamp 로 막는다
    sc = torch.where(th < 1e-4, torch.full_like(th, 0.5), th / (2.0 * sn.clamp(min=1e-7)))
    return v * sc


def left_jacobian(u):
    """SO(3) 좌 야코비안. exp(u+d) ~ exp([J_l(u) d]x) exp(u)."""
    th2 = u.pow(2).sum(-1, keepdim=True)
    th = th2.clamp(min=1e-24).sqrt()
    small = th2 < 1e-10
    a = torch.where(small, 0.5 - th2 / 24.0,
                    (1.0 - torch.cos(th)) / th2.clamp(min=1e-24))
    b = torch.where(small, 1.0 / 6.0 - th2 / 120.0,
                    (th - torch.sin(th)) / (th2 * th).clamp(min=1e-24))
    S = skew(u)
    eye = torch.eye(3, device=u.device, dtype=u.dtype).expand_as(S)
    return eye + a.unsqueeze(-1) * S + b.unsqueeze(-1) * (S @ S)


def polar_target(F0, iters=8):
    """MPM 의 F -> 가우시안별 (회전벡터, 로그신축) 목표. 초기값 전용이라 미분 없음."""
    from .anchor_fit import closest_rotation
    with torch.no_grad():
        R = closest_rotation(F0, iters, 1e-6)
        U = R.transpose(-1, -2) @ F0
        sv = U.diagonal(dim1=-2, dim2=-1)
        ls = torch.log(torch.nn.functional.softplus(sv * 4.0) / 4.0 + 1e-6)
        return logmap(R), ls


class FrameState:
    """짝 구조가 고정된 동안 재사용되는 인코더/디코더.

    가중치 w 는 매 호출 인자로 받는다 -- 기하가 학습되면 매 iter 바뀌기 때문이다.
    """

    def __init__(self, pair_g, pair_a, N, M, mass):
        self.pair_g, self.pair_a = pair_g, pair_a
        self.N, self.M = int(N), int(M)
        self.dev = pair_g.device
        self.mass = mass
        self.msum = mass.sum()
        # [N,K] 조밀 배치: 정규방정식 조립과 초기값 최소제곱에 쓴다
        order = torch.argsort(pair_g)
        pg, pa = pair_g[order], pair_a[order]
        cnt = torch.bincount(pg, minlength=self.N)
        off = torch.cat([torch.zeros(1, dtype=torch.long, device=self.dev), cnt.cumsum(0)])
        K = int(cnt.max())
        sel = torch.arange(K, device=self.dev).unsqueeze(0) + off[:-1].unsqueeze(1)
        self.valid = torch.arange(K, device=self.dev).unsqueeze(0) < cnt.unsqueeze(1)
        self.sel = sel.clamp(max=pa.shape[0] - 1)
        self.order = order
        self.sa = torch.where(self.valid, pa[self.sel], torch.zeros_like(self.sel))
        self.K = K
        # 짝-짝 인덱스 [N,K,K] 를 미리 만들면 안 된다. 프로브의 고정 이웃 8 개와 달리
        # 실제 피팅은 마할라노비스 지지라 가우시안당 앵커가 평균 20.9 개이고, N·K^2 이
        # 5.6 GB 로 뛴다(실측 OOM). gram() 에서만 쓰이므로 가우시안을 쪼개 누적한다.
        # 미분 가능 경로에서는 청크마다 [chunk,K,K] 외적이 그래프에 남는다. 앵커가
        # 1024 로 늘자 이것이 OOM 의 한 축이 됐다(44.5GB 중 43.96GB). 2^22 로 낮추면
        # 중간값이 4 분의 1 이고 커널 호출만 네 배가 된다 -- 계산 결과는 동일하다.
        self.chunk = max(1, (1 << 22) // max(K * K, 1))
        # 미분이 필요 없는 경로(라벨 생성)에서는 청크를 잘게 썰 이유가 없다.
        # K=45 일 때 2^22 청크면 171,553 행을 83 번에 나눠 index_add 하는데 그것이
        # gram 조립 594 ms 의 대부분이었다 -- 인코더 비용의 94%.
        self.chunk_nograd = max(1, (1 << 26) // max(K * K, 1))
        # 기하가 고정인 동안(궤적 생성·DAgger) gram 과 촐레스키는 매번 같은 값이다.
        # w 의 신원으로 열쇠를 삼아 한 번만 계산한다.
        self._gcache = None

    # ---- 복호 -------------------------------------------------------------
    def blend(self, u_a, s_a, w):
        ug = torch.zeros(self.N, 3, device=self.dev, dtype=u_a.dtype).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * u_a[self.pair_a])
        sg = torch.zeros(self.N, 3, device=self.dev, dtype=s_a.dtype).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * s_a[self.pair_a])
        return ug, sg

    def decode(self, u_a, s_a, w):
        ug, sg = self.blend(u_a, s_a, w)
        return expmap(ug) * sg.exp().unsqueeze(-2)      # F = R diag(e^s)

    # ---- 야코비안 (무행렬) -------------------------------------------------
    @staticmethod
    def _Kz(F, Jl, zg):
        """[N,6] 증분 -> [N,3,3] 의 1 차 변화"""
        a = torch.einsum("nij,nj->ni", Jl, zg[:, :3])
        return skew(a) @ F + F * zg[:, 3:].unsqueeze(-2)

    @staticmethod
    def _KTy(F, Jl, Y):
        """[N,3,3] -> [N,6] 수반"""
        c = torch.cross(F, Y, dim=-2).sum(-1)           # sum_k F[:,k] x Y[:,k]
        gu = torch.einsum("nji,nj->ni", Jl, c)          # J_l^T c
        gs = (F * Y).sum(-2)                            # F[:,k] . Y[:,k]
        return torch.cat([gu, gs], -1)

    def _gather(self, z, w):
        return torch.zeros(self.N, 6, device=self.dev, dtype=z.dtype).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * z[self.pair_a])

    def _scatter(self, y, w):
        return torch.zeros(self.M, 6, device=self.dev, dtype=y.dtype).index_add_(
            0, self.pair_a, w.unsqueeze(-1) * y[self.pair_g])

    def _Hz(self, F, Jl, w, z):
        t = self._Kz(F, Jl, self._gather(z, w))
        return self._scatter(self._KTy(F, Jl, self.mass.reshape(-1, 1, 1) * t), w)

    def _diagH(self, F, Jl, w):
        """야코비 전처리자. 대각 (a,i) = sum_g m_g w_ga^2 |K_g e_i|^2"""
        e = torch.eye(6, device=self.dev, dtype=F.dtype)
        n2 = torch.stack([
            self.mass * self._Kz(F, Jl, e[i].expand(self.N, 6)).pow(2).sum((-1, -2))
            for i in range(6)], -1)                                  # [N,6]
        return torch.zeros(self.M, 6, device=self.dev, dtype=F.dtype).index_add_(
            0, self.pair_a, (w * w).unsqueeze(-1) * n2[self.pair_g])

    # ---- 초기값: 로그 공간 최소제곱 ---------------------------------------
    def gram_mat(self, w, ridge=1e-4):
        """W^T M W. [N,M] 을 세우지 않고 짝 구조로 조립한다. 미분 가능하다."""
        sw = torch.where(self.valid, w[self.order][self.sel], torch.zeros_like(w[:1]))
        sw = sw * self.mass.sqrt().unsqueeze(1)
        G = torch.zeros(self.M * self.M, device=self.dev)
        step = self.chunk if torch.is_grad_enabled() else self.chunk_nograd
        for i in range(0, self.N, step):
            c = slice(i, min(i + step, self.N))
            a = self.sa[c]
            G.index_add_(0, (a.unsqueeze(-1) * self.M + a.unsqueeze(-2)).reshape(-1),
                         (sw[c].unsqueeze(-1) * sw[c].unsqueeze(-2)).reshape(-1))
        G = G.reshape(self.M, self.M)
        # 릿지 배율은 detach 한다 -- 정규화 세기가 학습 대상이 되면 그것을 줄이는
        # 방향으로 흐른다
        return G + ridge * G.diagonal().mean().detach() * torch.eye(self.M, device=self.dev)

    def gram(self, w, ridge=1e-4):
        return torch.linalg.cholesky(self.gram_mat(w, ridge).double())

    def gram_cached(self, w, fixed=None, ridge=1e-4):
        """기하가 고정인 동안 재사용하는 (G, L, L_free). grad 가 켜져 있으면 캐시하지
        않는다 -- 피팅은 매 iter 다른 w 를 보고 그래프도 흘려야 한다."""
        if torch.is_grad_enabled():
            G = self.gram_mat(w, ridge)
            Lf = None
            if fixed is not None and bool(fixed.any()):
                Lf = torch.linalg.cholesky(G[~fixed][:, ~fixed].double())
            return G, torch.linalg.cholesky(G.double()), Lf
        key = (w.data_ptr(), int(w.shape[0]), float(ridge))
        if self._gcache is None or self._gcache[0] != key:
            G = self.gram_mat(w, ridge)
            L = torch.linalg.cholesky(G.double())
            Lf = None
            if fixed is not None and bool(fixed.any()):
                Lf = torch.linalg.cholesky(G[~fixed][:, ~fixed].double())
            self._gcache = (key, G, L, Lf)
        return self._gcache[1], self._gcache[2], self._gcache[3]

    def _wls(self, t, w, L):
        """min_x sum_g m_g |W x - t_g|^2 의 해. t 는 [N,d]."""
        mw = self.mass[self.pair_g] * w
        rhs = torch.zeros(self.M, t.shape[-1], device=self.dev).index_add_(
            0, self.pair_a, mw.unsqueeze(-1) * t[self.pair_g])
        return torch.cholesky_solve(rhs.double(), L).float()

    def _ls_init(self, ug_t, sg_t, w, ridge=1e-4, L=None):
        """min_x sum_g m_g |W x - t_g|^2. 정규화 단계가 없어 최소제곱이 살아남는다."""
        if L is None:
            L = self.gram(w, ridge)
        return [self._wls(t, w, L) for t in (ug_t, sg_t)]

    # ---- 인코더 -----------------------------------------------------------
    def encode(self, F0, w, iters=6, cg_iters=20, lam=1e-3, init=None, verbose=False):
        """MPM 의 F [N,3,3] -> 앵커의 (u_a, s_a). 목적함수 위의 가우스-뉴턴.

        기본값 (GN 6, CG 20) 은 비용 스윕에서 고른 것이다: F 잔차 0.625 에 창당
        615 ms. GN 10/CG 20 이면 0.611/904 ms, GN 8/CG 40 이면 0.617/1258 ms 로
        더 써도 3% 안쪽이다. 반면 _ls_init 을 빼고 0 에서 시작하면 GN 12 를 돌려도
        0.849 에 머문다 -- 초기값이 이 풀이의 절반이다.
        """
        with torch.no_grad():
            if init is None:
                ug_t, sg_t = polar_target(F0)
                u, s = self._ls_init(ug_t, sg_t, w)
            else:
                u, s = init[0].clone(), init[1].clone()

            def loss_of(u_, s_):
                F = self.decode(u_, s_, w)
                return float((self.mass * (F - F0).pow(2).sum((-1, -2))).sum() / self.msum), F

            L0, F = loss_of(u, s)
            hist = [L0]
            for it in range(iters):
                ug, _ = self.blend(u, s, w)
                Jl = left_jacobian(ug)
                r = self.mass.reshape(-1, 1, 1) * (F - F0)
                b = -self._scatter(self._KTy(F, Jl, r), w)
                D = self._diagH(F, Jl, w)
                D = D.clamp(min=1e-10 * D.mean().clamp(min=1e-30))
                # 감쇠 켤레기울기: (H + lam D) z = b, 전처리자 1/(D(1+lam))
                z = torch.zeros_like(b)
                rr = b.clone()
                Pm = 1.0 / (D * (1.0 + lam))
                zz = Pm * rr
                pp = zz.clone()
                rz = (rr * zz).sum()
                for _ in range(cg_iters):
                    Ap = self._Hz(F, Jl, w, pp) + lam * D * pp
                    denom = (pp * Ap).sum()
                    if not torch.isfinite(denom) or denom.abs() < 1e-30:
                        break
                    al = rz / denom
                    z = z + al * pp
                    rr = rr - al * Ap
                    zz = Pm * rr
                    rz2 = (rr * zz).sum()
                    if rz2 <= 1e-14 * rz.abs().clamp(min=1e-30):
                        break
                    pp = zz + (rz2 / rz) * pp
                    rz = rz2
                Ln, Fn = loss_of(u + z[:, :3], s + z[:, 3:])
                if Ln < L0:
                    u, s, L0, F = u + z[:, :3], s + z[:, 3:], Ln, Fn
                    lam = max(lam * 0.4, 1e-8)
                else:
                    lam = min(lam * 6.0, 1e4)
                hist.append(L0)
                if verbose:
                    print(f"    GN {it+1:2d}  손실 {L0:.6e}  lam {lam:.2e}", flush=True)
            return u, s, hist

    # ---- 결합 인코더: (p, u, s) 를 왕복 손실 위에서 함께 푼다 ----------------
    #
    # F 항만 최소화하면 피팅의 손실(x + v + F)과 다른 함수를 푸는 셈이라, 인코더
    # 출력을 detach 하고 복호만 미분하는 근거(포락선 정리)가 무너진다 -- x 항을
    # 통과하는 d(u,s)/d(기하) 경로가 통째로 빠진다. 그래서 x 와 F 를 함께 푼다.
    # v 는 v_a 만으로 정해지는 별도의 선형 최소제곱이므로 project_v_ls 가 그대로
    # 정확해(같은 C 행렬), 여기서 건드리지 않는다.
    #
    # 손실이 제곱합의 **제곱근** 들의 합(_mw 가 sqrt 를 포함한다)이므로 순수
    # 가우스-뉴턴이 아니다: L = sum_k c_k sqrt(S_k) 의 그래디언트가
    # sum_k (c_k / 2 sqrt(S_k)) grad S_k 이므로, 매 바깥 반복에서 그 계수를 다시
    # 재고 가중 제곱합 위에서 GN 을 돈다(IRLS). 고정점이 L 의 정류점이다.

    def _gather9(self, z, w):
        return torch.zeros(self.N, 9, device=self.dev, dtype=z.dtype).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * z[self.pair_a])

    def _scatter9(self, y, w):
        return torch.zeros(self.M, 9, device=self.dev, dtype=y.dtype).index_add_(
            0, self.pair_a, w.unsqueeze(-1) * y[self.pair_g])

    def _apply(self, F, Jl, Yg, w, z):
        """z [M,9] = (dp, du, ds) -> 잔차 공간의 1 차 변화 (dx [N,3], dF [N,3,3])"""
        zg = self._gather9(z, w)
        dF = self._Kz(F, Jl, zg[:, 3:])
        return zg[:, :3] + torch.einsum("nij,nj->ni", dF, Yg), dF

    def _applyT(self, F, Jl, Yg, w, gx, gF):
        """(gx, gF) -> [M,9]. <dx,gx> + <dF,gF> = <z, 이 결과> 를 만족한다."""
        us = self._KTy(F, Jl, gx.unsqueeze(-1) * Yg.unsqueeze(-2) + gF)
        return self._scatter9(torch.cat([gx, us], -1), w)

    def decode_x(self, p, u, s, w, Yg):
        F = self.decode(u, s, w)
        cc = torch.zeros(self.N, 3, device=self.dev, dtype=p.dtype).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * p[self.pair_a])
        return cc + torch.einsum("nij,nj->ni", F, Yg), F

    def encode_joint(self, x0, F0, w, Yg, c_x=1.0, c_F=1.0, fixed=None, p_fix=None,
                     iters=6, cg_iters=20, lam=1e-3, ridge=1e-4, init=None,
                     verbose=False):
        """MPM 의 (x, F) -> 앵커의 (p, u, s). 피팅과 같은 목적함수 위의 GN.

        c_x, c_F 는 피팅의 항 계수 그대로 준다: c_x = RT_W[0]/grid_lim,
        c_F = RT_W[2]·r_ref/grid_lim.
        """
        with torch.no_grad():
            L = self.gram(w, ridge)
            if init is None:
                ug_t, sg_t = polar_target(F0)
                u, s = self._ls_init(ug_t, sg_t, w, L=L)
                Fh = self.decode(u, s, w)
                p = self._wls(x0 - torch.einsum("nij,nj->ni", Fh, Yg), w, L)
            else:
                p, u, s = [t.clone() for t in init]
            if fixed is not None and bool(fixed.any()):
                p = torch.where(fixed.unsqueeze(-1), p_fix, p)
                msk = torch.ones(self.M, 9, device=self.dev)
                msk[:, :3] = (~fixed).unsqueeze(-1).float()
            else:
                msk = None

            def terms(p_, u_, s_):
                xh, F = self.decode_x(p_, u_, s_, w, Yg)
                rx, rF = xh - x0, F - F0
                Sx = float((self.mass * rx.pow(2).sum(-1)).sum() / self.msum)
                SF = float((self.mass * rF.pow(2).sum((-1, -2))).sum() / self.msum)
                return Sx, SF, rx, rF, F

            Sx, SF, rx, rF, F = terms(p, u, s)
            L0 = c_x * Sx ** 0.5 + c_F * SF ** 0.5
            hist = [L0]
            for it in range(iters):
                # sqrt 의 미분에서 오는 항별 계수 (IRLS)
                bx = c_x / (2.0 * max(Sx, 1e-30) ** 0.5) / float(self.msum)
                bF = c_F / (2.0 * max(SF, 1e-30) ** 0.5) / float(self.msum)
                wx = bx * self.mass.unsqueeze(-1)                   # [N,1]
                wF = bF * self.mass.reshape(-1, 1, 1)
                ug, _ = self.blend(u, s, w)
                Jl = left_jacobian(ug)
                b = -self._applyT(F, Jl, Yg, w, wx * rx, wF * rF)
                # 야코비 전처리자
                e = torch.eye(9, device=self.dev, dtype=F.dtype)
                cols = []
                for i in range(9):
                    dx_, dF_ = self._apply(F, Jl, Yg, w, e[i].expand(self.M, 9))
                    cols.append((wx.squeeze(-1) * dx_.pow(2).sum(-1)
                                 + wF.reshape(-1) * dF_.pow(2).sum((-1, -2))))
                D = torch.zeros(self.M, 9, device=self.dev, dtype=F.dtype).index_add_(
                    0, self.pair_a, (w * w).unsqueeze(-1) * torch.stack(cols, -1)[self.pair_g])
                D = D.clamp(min=1e-12 * D.mean().clamp(min=1e-30))
                if msk is not None:
                    b = b * msk

                def Hz(z):
                    dx_, dF_ = self._apply(F, Jl, Yg, w, z)
                    out = self._applyT(F, Jl, Yg, w, wx * dx_, wF * dF_) + lam * D * z
                    return out * msk if msk is not None else out

                z = torch.zeros_like(b); rr = b.clone()
                Pm = 1.0 / (D * (1.0 + lam))
                zz = Pm * rr; pp = zz.clone(); rz = (rr * zz).sum()
                for _ in range(cg_iters):
                    Ap = Hz(pp)
                    den = (pp * Ap).sum()
                    if not torch.isfinite(den) or den.abs() < 1e-30:
                        break
                    al = rz / den
                    z = z + al * pp; rr = rr - al * Ap
                    zz = Pm * rr; rz2 = (rr * zz).sum()
                    if rz2 <= 1e-14 * rz.abs().clamp(min=1e-30):
                        break
                    pp = zz + (rz2 / rz) * pp; rz = rz2
                Sx2, SF2, rx2, rF2, F2 = terms(p + z[:, :3], u + z[:, 3:6], s + z[:, 6:])
                Ln = c_x * Sx2 ** 0.5 + c_F * SF2 ** 0.5
                if Ln < L0:
                    p, u, s = p + z[:, :3], u + z[:, 3:6], s + z[:, 6:]
                    Sx, SF, rx, rF, F, L0 = Sx2, SF2, rx2, rF2, F2, Ln
                    lam = max(lam * 0.4, 1e-8)
                else:
                    lam = min(lam * 6.0, 1e4)
                hist.append(L0)
                if verbose:
                    print(f"    GN {it+1:2d}  L {L0:.6e}  (x {Sx**0.5:.4e}, "
                          f"F {SF**0.5:.4e})  lam {lam:.2e}", flush=True)
            return p, u, s, hist

    # ---- 닫힌 형태 인코더 --------------------------------------------------
    #
    # 가우시안별 목표를 먼저 쪼개면 남는 것이 전부 선형이라 반복이 사라진다.
    #
    #   1) 극분해로 가우시안별 (회전벡터, 로그신축) 목표. F^MPM 만의 함수이므로
    #      기하 파라미터와 무관한 **상수**다 -- 미분할 것이 없고, 앵커 겹침과도
    #      무관한 고정 비용이다.
    #   2) 블렌드가 u_g = sum_a w_ga u_a, s_g = sum_a w_ga s_a 로 둘 다 선형이므로
    #      각 공간의 최소제곱 역이 (W^T M W) x = W^T M t, 촐레스키 한 번이다.
    #   3) (u,s) 가 정해지면 F_g 가 정해지고, x_g = sum_a w_ga p_a + F_g(X-r) 은
    #      p 에 선형이라 **같은 인수분해**를 재사용한다.
    #
    # 가우스-뉴턴판보다 F 잔차는 크지만(초기 기하에서 0.895 대 0.617), 그 대가로
    # 얻는 것이 크다: 포락선 정리가 필요 없다. 인코더를 detach 하지 않고 그냥
    # 통과해 미분하면 되므로 "안쪽 풀이가 수렴했는가" 라는 전제 자체가 사라지고,
    # 겹침이 늘수록 풀이가 나빠져 그래디언트가 틀어지는 되먹임도 없다. 그리고
    # 실제로 쓸 인코더가 이것이므로, 기하를 이 인코더에 맞춰 맞추는 편이 맞다.

    def encode_closed(self, x0, F0, w, Yg, fixed=None, p_fix=None, ridge=1e-4,
                      targets=None):
        """MPM 의 (x, F) -> 앵커의 (p, u, s). 촐레스키 두 번, 반복 없음."""
        if targets is None:
            with torch.no_grad():
                targets = polar_target(F0)
        ug_t, sg_t = targets
        G = self.gram_mat(w, ridge)
        L = torch.linalg.cholesky(G.double())
        u = self._wls(ug_t, w, L)
        s = self._wls(sg_t, w, L)
        F = self.decode(u, s, w)
        rhs_x = x0 - torch.einsum("nij,nj->ni", F, Yg)
        if fixed is not None and bool(fixed.any()):
            # 고정 앵커의 p 는 파라미터 그대로 두고 자유 블록만 푼다. p_fix 를
            # 살아 있는 텐서로 받으면 그 경로의 미분도 함께 흐른다.
            # **고정 앵커의 기여만** 빼야 한다. p_fix 를 그대로 흩뿌리면 자유
            # 앵커의 현재 위치까지 상수항으로 들어가 우변이 통째로 틀어진다
            # (ficus 는 512 중 129 가 고정이라 mwRMS x 가 1.55 까지 갔다).
            pfx = torch.where(fixed.unsqueeze(-1), p_fix, torch.zeros_like(p_fix))
            cc = torch.zeros(self.N, 3, device=self.dev, dtype=w.dtype).index_add_(
                0, self.pair_g, w.unsqueeze(-1) * pfx[self.pair_a])
            free = ~fixed
            idx = torch.nonzero(free, as_tuple=False).squeeze(-1)
            Gf = G[free][:, free]
            Lf = torch.linalg.cholesky(Gf.double())
            mw = self.mass[self.pair_g] * w
            rhs = torch.zeros(self.M, 3, device=self.dev, dtype=w.dtype).index_add_(
                0, self.pair_a, mw.unsqueeze(-1) * (rhs_x - cc)[self.pair_g])
            pf = torch.cholesky_solve(rhs[free].double(), Lf).float()
            p = pfx.clone().index_put((idx,), pf)
        else:
            p = self._wls(rhs_x, w, L)
        return p, u, s

    # ---- 자유 F: 앵커가 3x3 을 그대로 들고 선형 블렌딩 ---------------------
    #
    # F 를 회전과 신축으로 쪼개지 않는다. R diag(e^s) 형태는 6 자유도라 전단을 못
    # 담고, 회전벡터가 2pi 로 감겨 상태가 최대 80 rad 까지 갔다. 행렬은 감기지
    # 않으므로 그 병이 원천적으로 없다.
    #
    # 실측(ficus, fs_sc3000): F 잔차 0.2908 대 0.7128(회전+신축), 스텝 변화 꼬리
    # 30 배 대 162 배, 비용 30 ms 대 1551 ms -- 정확도·조건화·속도 전부 이긴다.
    #
    # 대가는 부피다. 성분별 평균이라 이웃 회전이 어긋나면 행렬식이 줄어든다:
    # 중앙값 0.99 로 멀쩡하지만 하위 1% 가 0.23, 0.1% 가 0.018 이다. 음수는 없으니
    # 뒤집히지는 않고 납작해진다. 렌더링에서 문제가 되는지는 학생을 돌려봐야 안다.

    def decode_F(self, Fa, w):
        """[M,9] -> [N,3,3], 성분별 가중평균"""
        return torch.zeros(self.N, 9, device=self.dev, dtype=Fa.dtype).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * Fa[self.pair_a]).view(-1, 3, 3)

    def decode_x_F(self, p, Fa, w, Yg):
        F = self.decode_F(Fa, w)
        cc = torch.zeros(self.N, 3, device=self.dev, dtype=p.dtype).index_add_(
            0, self.pair_g, w.unsqueeze(-1) * p[self.pair_a])
        return cc + torch.einsum("nij,nj->ni", F, Yg), F

    def encode_closed_F(self, x0, F0, w, Yg, fixed=None, p_fix=None, ridge=1e-4):
        """MPM 의 (x, F) -> 앵커의 (p, F_a). 촐레스키 한 번(자유 블록은 두 번)."""
        G, L, Lf = self.gram_cached(w, fixed, ridge)
        Fa = self._wls(F0.reshape(-1, 9), w, L)
        F = self.decode_F(Fa, w)
        rhs_x = x0 - torch.einsum("nij,nj->ni", F, Yg)
        if fixed is not None and bool(fixed.any()):
            pfx = torch.where(fixed.unsqueeze(-1), p_fix, torch.zeros_like(p_fix))
            cc = torch.zeros(self.N, 3, device=self.dev, dtype=w.dtype).index_add_(
                0, self.pair_g, w.unsqueeze(-1) * pfx[self.pair_a])
            free = ~fixed
            idx = torch.nonzero(free, as_tuple=False).squeeze(-1)
            mw = self.mass[self.pair_g] * w
            rhs = torch.zeros(self.M, 3, device=self.dev, dtype=w.dtype).index_add_(
                0, self.pair_a, mw.unsqueeze(-1) * (rhs_x - cc)[self.pair_g])
            pf = torch.cholesky_solve(rhs[free].double(), Lf).float()
            p = pfx.clone().index_put((idx,), pf)
        else:
            p = self._wls(rhs_x, w, L)
        return p, Fa
