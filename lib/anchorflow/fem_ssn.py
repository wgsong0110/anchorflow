"""절 CM176 --ssn 1: fem_rls (절 CM146 잔차 노름 Armijo) 의 받음 판정을 비단조 (Grippo) 로: 기준 = 최근 받은 NM 개 점의 ‖g‖² 최댓값 (NM = 환경 AF_RLS_NM, 1 이면 fem_rls 와 똑같음).
새 상수 없음 (c1 1e-4 는 fem_rls 의 것, NM 은 기억 길이). fem_rls 의 `_step`·`_bufs` 를 모듈 수준에서 바꿔치기 (fem_rls 파일은 안 건드림)."""
import os
import torch
from . import fem_rls as RL

NM = int(os.environ.get("AF_RLS_NM", "1"))
_obufs, _ostep = RL._bufs, RL._step


def _bufs(s):
    first = getattr(s, "_rl_n", None) != s.dl.numel()
    _obufs(s)
    if first or not hasattr(s, "_rl_nm"):
        s._rl_nm = torch.zeros(max(NM, 1), device=s.dl.device, dtype=torch.float64)


def _step(s, it):
    if NM <= 1:
        return _ostep(s, it)
    _bufs(s)
    run = (s.CT[0] == 0)
    if it == 0:
        s._rl_nm.copy_(torch.where(run, s._rl_g2.expand_as(s._rl_nm), s._rl_nm))
        return _ostep(s, it)
    ref = s._rl_nm.max().view(1)
    old_g2b = s._rl_g2b.clone()
    s._rl_g2b.copy_(ref)                                     # fem_rls 의 판정이 기준으로 쓰는 칸을 비단조 기준으로 (판정 직후 되돌림)
    a = s._rl_a.clone()
    acc = s._rl_g2 <= (1.0 - RL.C1 * a) ** 2 * ref
    _ostep(s, it)
    s._rl_g2b.copy_(torch.where(run & ~acc, old_g2b, s._rl_g2b))   # 거절이면 기준점의 노름² 은 그대로 (fem_rls 가 안 바꿈)
    s._rl_nm.copy_(torch.where(run & acc, torch.cat([s._rl_nm[1:], s._rl_g2.view(1)]), s._rl_nm))


def patch():
    RL._bufs, RL._step = _bufs, _step
