"""실현 가능성(유한 에너지)을 지키는 L-BFGS (--interp c1wf 의 진짜 로그 장벽용).

torch.optim.LBFGS 의 strong_wolfe 는 에너지가 +inf 인 시도점을 만나면 3 차 보간이
nan 걸음을 낸다. 진짜 로그 장벽(계수<=0 이면 +inf)과 함께 쓰려면 시도점이 무한이면
걸음을 줄이는 되추적이 필요하다 (IPC 의 선 탐색 거름과 같은 구실).

같은 쓰임새: opt.step(closure), closure 는 0 으로 비운 뒤 E 를 계산하고 backward 한다.
  - 방향: 두 고리 L-BFGS (history_size), 곡률 sᵀy<=0 이면 그 짝은 버린다
  - 첫 걸음 t0 = lr·min(1, 1/|g|₁) (torch 와 같다), 이후 t0 = lr
  - 되추적 Armijo (c1=1e-4, 반감); 시도점 에너지가 유한하지 않으면 무조건 반감
  - max_iter 번 반복, 평가 상한 max_eval (기본 max_iter*5/4, torch 와 같다)
"""
import math

import torch


class FeasLBFGS:
    def __init__(self, params, lr=1.0, max_iter=20, max_eval=None, history_size=50,
                 tolerance_grad=0.0, tolerance_change=0.0, c1=1e-4, max_ls=40):
        self.params = [p for p in params]
        self.lr, self.max_iter = float(lr), int(max_iter)
        self.max_eval = int(max_eval) if max_eval else int(self.max_iter * 5 / 4)
        self.m, self.tg, self.tc, self.c1, self.max_ls = int(history_size), float(tolerance_grad), float(tolerance_change), float(c1), int(max_ls)
        self.stats = {}

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def _flat_grad(self):
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in self.params])

    def _flat(self):
        return torch.cat([p.detach().reshape(-1) for p in self.params])

    def _set(self, x):
        o = 0
        with torch.no_grad():
            for p in self.params:
                n = p.numel(); p.copy_(x[o:o + n].view_as(p)); o += n

    def step(self, closure):
        nev = 0
        E = float(closure()); nev += 1
        if not math.isfinite(E):
            self.stats = dict(evals=nev, iters=0, infeasible_start=True, E=E, stop="infeasible_start")
            return E
        g = self._flat_grad().detach().clone()
        g = torch.nan_to_num(g, nan=0.0)
        x = self._flat()
        S, Y = [], []
        it, n_bt, n_inf = 0, 0, 0
        stop = "max_iter"                      # 종료 사유 (기록용, 계산에는 안 쓴다)
        for it in range(1, self.max_iter + 1):
            if self.tg > 0 and float(g.abs().max()) <= self.tg:
                stop = "grad_tol"; break
            q = g.clone(); al = []
            for s, y in reversed(list(zip(S, Y))):
                r = 1.0 / float(y @ s); a_ = r * float(s @ q); q -= a_ * y; al.append((r, a_, s, y))
            if S:
                q *= float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1])
            for r, a_, s, y in reversed(al):
                b_ = r * float(y @ q); q += s * (a_ - b_)
            d = -q
            gtd = float(g @ d)
            if not (gtd < 0):
                d = -g; gtd = float(g @ d); S, Y = [], []
            t = self.lr * (min(1.0, 1.0 / float(g.abs().sum().clamp_min(1e-30))) if it == 1 else 1.0)
            ok = False; ls_cap = "max_ls"
            for _ in range(self.max_ls):
                if nev >= self.max_eval:
                    ls_cap = "max_eval_in_ls"; break
                self._set(x + t * d)
                En = float(closure()); nev += 1
                if math.isfinite(En) and En <= E + self.c1 * t * gtd:
                    ok = True; break
                if not math.isfinite(En):
                    n_inf += 1
                n_bt += 1
                t *= 0.5
            if not ok:
                self._set(x)
                stop = "ls_fail_" + ls_cap         # 선 탐색 실패 (되추적 max_ls 소진 또는 평가 상한)
                break
            gn = torch.nan_to_num(self._flat_grad().detach().clone(), nan=0.0)
            xn = x + t * d
            s, y = xn - x, gn - g
            if float(s @ y) > 1e-20:
                S.append(s); Y.append(y)
                if len(S) > self.m:
                    S.pop(0); Y.pop(0)
            dE = abs(E - En)
            x, g, E = xn, gn, En
            if self.tc > 0 and dE < self.tc:
                stop = "change_tol"; break
            if nev >= self.max_eval:
                stop = "max_eval"; break
        # 마지막 x 의 기울기를 params 에 남긴다 (호출부가 읽을 수 있게)
        self._set(x)
        self.stats = dict(evals=nev, iters=it, backtracks=n_bt, infeasible_trials=n_inf, E=E, stop=stop,
                          gnorm=float(g.norm()), ginf=float(g.abs().max()))
        return E
