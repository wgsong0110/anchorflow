"""절 CM140: 뉴턴 방향에 Anderson 가속 (깊이 1, 걸음 안 반복끼리) -- `--anx 1`, 기본 0 = 옛 판과 비트 같음.
배경 (절 CM138): 뉴턴이 푸는 선형계의 헤시안이 탄성 접선 (초기 강성법) 이라 소성 흐름 방식에서는 진짜 접선보다 훨씬 뻣뻣하다 -> 반복이 보폭 1 로도 느린 선형 수렴 (평형 근처에서 정체).
  이 반복 eta <- eta + dl 은 고정점 반복 G(eta) = eta + dl(eta) 이고, 정체는 G 의 느린 모드 (고유값이 1 에 가까움). Anderson 가속은 임계값·클램프 없이 이 느린 모드를 최소제곱으로 없앤다.
정의: 반복 k (k >= 1) 에서 dl_k = G(eta_k) - eta_k 를 (원래 값) 저장, theta = <dl_k, dl_k - dl_{k-1}> / |dl_k - dl_{k-1}|^2 (잔차 |(1-theta) dl_k + theta dl_{k-1}| 최소),
  새 방향 dl_k' = dl_k - theta [ (eta_k - eta_{k-1}) + (dl_k - dl_{k-1}) ]  (= (1-theta) G(eta_k) + theta G(eta_{k-1}) - eta_k).
  이후 뒤집힘 필터·장벽 보폭 (SC[14])·rt_lsup 은 이 방향을 그대로 씀 (실현 가능성 검사는 가속된 방향으로).
`--anx 2` 는 내적을 노드 질량 mn 으로 가중 (에너지 꼴에 가까운 노름). 걸음 첫 반복 (it 0) 은 건드리지 않음.
설치: install() 이 솔버 클래스의 alloc·ctl 을 감쌈 (ctl op 7 = PCG 끝, 보폭 쓰기 전; 허용치로 끝난 반복 (CT[0]) 은 그대로)."""
import torch

MODE = [0]
ST = {"n": 0}


def install(mode):
    MODE[0] = int(mode)
    from . import fem_an_rt as ART                                       # 새 실행기 (fem_an_rt_sim) 가 쓰는 솔버 = ART.RTSolver (nlwhy 와 같은 클래스)
    seen = {ART.RTSolver}
    _wrap(ART.RTSolver)
    return seen


def _wrap(c):
    octl = c.ctl
    oalloc = c.__dict__.get("alloc")

    def ctl(s, op, it=0):
        r = octl(s, op, it)
        if op != 7 or getattr(s, "_an_in", False) or MODE[0] == 0:
            return r
        if not getattr(s, "adapt", True) or not getattr(s, "ao_ready", True):
            return r
        s._an_in = True
        try:
            _acc(s, it)
        finally:
            s._an_in = False
        return r
    c.ctl = ctl
    if oalloc is not None:
        def alloc(s, need):
            r = oalloc(s, need)
            if MODE[0]:
                s._an_ea = torch.zeros_like(s.dl).view(-1); s._an_da = torch.zeros_like(s.dl).view(-1)
            return r
        c.alloc = alloc


def _acc(s, it):
    if getattr(s, "_an_ea", None) is None or s._an_ea.numel() != s.dl.numel():
        s._an_ea = torch.zeros_like(s.dl).view(-1); s._an_da = torch.zeros_like(s.dl).view(-1)
    d = s.dl.view(-1); e = s.eta.view(-1); ea, da = s._an_ea, s._an_da
    if it > 0:
        dd = d - da
        if MODE[0] == 2:
            w = s.mn.view(-1)
            w = w.repeat_interleave(3)[: d.numel()] if w.numel() * 3 == d.numel() else None
        else:
            w = None
        num = ((d * dd) if w is None else (w * d * dd)).double().sum()
        den = ((dd * dd) if w is None else (w * dd * dd)).double().sum()
        th = torch.where(den > 0, num / den, torch.zeros_like(den)).float()
        run = (s.CT[0] == 0).float()                                 # 허용치로 이미 끝났으면 이 반복은 건너뛰는 중 (dl 은 옛 값)
        new = d - th * ((e - ea) + dd)
        raw = d.clone()
        d.copy_(torch.where(run > 0, new, d))
        ea.copy_(e); da.copy_(raw)
    else:
        ea.copy_(e); da.copy_(d)
