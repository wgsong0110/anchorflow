"""절 CM146: 잔차 노름 판정 선탐색 `--rls 1` (기본 0 = 옛 판). 에너지 (탄성 포텐셜) 대신 반환 사상 뒤 실제 기울기 잔차 ||g|| 의 Armijo 감소로 뉴턴 보폭을 판정한다
(비결합 소성은 포텐셜이 아니므로 잔차가 맞는 판정 기준). 이 판은 에너지 선탐색 (--ls_n)·--en_mo·--descent 를 끈 명령 위에서만 (exe/cm140_gen.sh 토큰 rl 이 뺌).
정의 (걸음 안, 반복 k): 기준점 (eta_b, g2_b = ||g(eta_b)||^2, 기준 방향 d_b = 그 점의 뉴턴 방향), 시험 보폭 a (처음 1). 반복마다 현재 점 (= eta_b + a d_b) 의 ||g||^2 = g2 (rt_ctl 3 의 SC[1]) 를
  받아들임 <=> sqrt(g2) <= (1 - c1 a) sqrt(g2_b)  (c1 = 1e-4, 표준 Armijo 상수)  -> 기준점 = 현재 점, 기준 방향 = 이 반복의 뉴턴 방향 (dl 그대로), a = 1.
  거절 -> a <- a/2 (반씩 되돌림), 이 반복 방향을 기준점에서 새 시험 점 eta_b + a d_b 로 가는 변위로 바꿈 (PCG 가 현재 점에서 푼 방향은 버림).
  첫 반복 (k = 0) 은 기준점 설정만. 허용치로 끝난 반복 (CT[0]) 은 그대로. 임계값·클램프 없음 (보폭 줄임은 Armijo 가 정하고, 하한은 걸음 끝).
설치: fem_an_rt.RTSolver.ctl 의 op 3 (|g|² 직후) 에서 g2 를 저장, op 7 (PCG 끝, 보폭 쓰기 전) 에서 판정·방향 바꿈."""
import torch

MODE = [0]
C1 = 1e-4


def install(mode):
    MODE[0] = int(mode)
    from . import fem_an_rt as ART
    _wrap(ART.RTSolver)
    return {ART.RTSolver}


def _wrap(c):
    octl = c.ctl
    oalloc = c.__dict__.get("alloc")

    def ctl(s, op, it=0):
        r = octl(s, op, it)
        if MODE[0] == 0 or getattr(s, "_rl_in", False) or not getattr(s, "adapt", True):
            return r
        if op == 3:
            _bufs(s); s._rl_g2.copy_(s.SC[1:2])
        elif op == 7:
            s._rl_in = True
            try:
                _step(s, it)
            finally:
                s._rl_in = False
        return r
    c.ctl = ctl
    if oalloc is not None:
        def alloc(s, need):
            r = oalloc(s, need)
            if MODE[0]:
                s._rl_n = None; _bufs(s)
            return r
        c.alloc = alloc


def _bufs(s):
    n = s.dl.numel()
    if getattr(s, "_rl_n", None) == n:
        return
    dev = s.dl.device
    s._rl_n = n
    s._rl_eb = torch.zeros(n, device=dev, dtype=torch.float32); s._rl_db = torch.zeros(n, device=dev, dtype=torch.float32)
    s._rl_g2 = torch.zeros(1, device=dev, dtype=torch.float64); s._rl_g2b = torch.zeros(1, device=dev, dtype=torch.float64)
    s._rl_a = torch.ones(1, device=dev, dtype=torch.float64)


def _step(s, it):
    _bufs(s)
    d = s.dl.view(-1); e = s.eta.view(-1)
    run = (s.CT[0] == 0)
    if it == 0:
        s._rl_eb.copy_(torch.where(run, e, s._rl_eb)); s._rl_db.copy_(torch.where(run, d, s._rl_db))
        s._rl_g2b.copy_(torch.where(run, s._rl_g2, s._rl_g2b)); s._rl_a.copy_(torch.where(run, torch.ones_like(s._rl_a), s._rl_a))
        return
    a = s._rl_a
    acc = s._rl_g2 <= (1.0 - C1 * a) ** 2 * s._rl_g2b                 # sqrt 양변: ||g|| <= (1 - c1 a) ||g_b||
    a_new = torch.where(acc, torch.ones_like(a), a * 0.5)
    trial = s._rl_eb + a_new.float() * s._rl_db - e                    # 거절 시 새 시험 점으로 가는 변위
    d_new = torch.where(acc, d, trial)
    eb_new = torch.where(acc, e, s._rl_eb); db_new = torch.where(acc, d, s._rl_db); g2b_new = torch.where(acc, s._rl_g2, s._rl_g2b)
    d.copy_(torch.where(run, d_new, d))
    s._rl_eb.copy_(torch.where(run, eb_new, s._rl_eb)); s._rl_db.copy_(torch.where(run, db_new, s._rl_db))
    s._rl_g2b.copy_(torch.where(run, g2b_new, s._rl_g2b)); s._rl_a.copy_(torch.where(run, a_new, a))
