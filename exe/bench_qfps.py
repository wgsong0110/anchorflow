"""품질 기준을 만족하는 설정의 FPS를 잰다 (방법별 자기 수렴해 기준).

규약
  1. 손잡이 s(프레임당 스텝 수)를 2 배씩 올리며 궤적을 만든다.
  2. ||X(2s) - X(s)||_RMS / L < tau_conv 가 3 단계 연속 유지되면 그 X 를 그
     방법의 **수렴해**로 고정한다 (L = 초기 물체 지름).
  3. 수렴해 대비 **한 프레임 오차 <= tol** 이고 누적이 발산하지 않는 가장 작은
     s 를 이분 탐색으로 찾는다.
  4. 그 설정으로 다시 돌려 프레임당 시간을 재고 FPS 를 적는다.

이 파일은 궤적 생성 함수를 주입받는다 (방법마다 실행 방식이 다르다).
"""
from __future__ import annotations

import json
import time

import numpy as np


def rms_rel(a, b, L):
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean())) / L


def ladder(run, s0, tau_conv, L, max_mul=16, log=print):
    """s 를 2 배씩 올리며 수렴해를 찾는다 -> (X_conv, s_conv, 기록)."""
    hist = []
    s = s0
    X_prev, t_prev = run(s)
    hist.append((s, t_prev))
    ok = 0
    while s * 2 <= s0 * max_mul:
        s2 = s * 2
        X2, t2 = run(s2)
        d = rms_rel(X2[:len(X_prev)], X_prev, L)
        hist.append((s2, t2))
        log(f"  [사다리] s {s} -> {s2}: 변화 {100 * d:.4f}% "
            f"(기준 {100 * tau_conv:.3f}%), {t2:.1f}초")
        ok = ok + 1 if d < tau_conv else 0
        X_prev, s = X2, s2
        if ok >= 3:
            return X_prev, s, hist
    return X_prev, s, hist


def search(run, X_ref, L, tol, s_lo, s_hi, log=print):
    """수렴해 대비 tol 을 만족하는 가장 작은 s 를 이분 탐색."""
    best = None
    while s_lo < s_hi:
        mid = (s_lo + s_hi) // 2
        X, t = run(mid)
        e1 = rms_rel(X[1], X_ref[1], L)                 # 한 프레임 오차
        eT = rms_rel(X[-1], X_ref[-1], L)               # 누적 오차
        okay = (e1 <= tol) and np.isfinite(eT) and eT < 10 * tol * len(X)
        log(f"  [탐색] s {mid}: 한프레임 {100 * e1:.4f}% 누적 "
            f"{100 * eT:.4f}% -> {'통과' if okay else '미달'} ({t:.1f}초)")
        if okay:
            best = (mid, t, e1, eT)
            s_hi = mid
        else:
            s_lo = mid + 1
    return best
