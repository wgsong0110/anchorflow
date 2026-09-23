"""평행 집게(그리퍼)로 잡고 움직이는 궤적.

지금까지 쓰던 제어점은 반경 안 입자를 **평행이동**만 시켰다. 그건 손가락 모양도
자세도 없는 이상화라, 실제 조작에 더 가깝게 두 가지를 바꾼다.

  모양  -- 구가 아니라 마주 보는 두 판(Franka Panda Hand 식 평행 집게) 사이에
          낀 덩어리를 잡는다. 판의 법선·접선이 곧 집게 자세다.
  운동  -- 평행이동만이 아니라 SE(3), 즉 회전까지 준다. 잡은 덩어리가
          `x = P(t) + R(t) (x0 - c0)` 로 움직이므로 비틀기가 표현된다.

찢기처럼 방향이 정해진 조작은 웨이포인트를 무작위로 뽑는 것보다 **양 끝을 잡고
반대로 당기는** 결정적 궤적이 낫다. 그래서 여기서는 주축을 찾아 양 끝에 집게를
하나씩 놓고, 서로 반대 방향으로 당기면서 축 둘레로 비튼다.
"""
from __future__ import annotations

import numpy as np


def _frame_from(axis, second):
    """axis 를 첫 축으로 하는 정규직교 기저 [3,3] (행이 축)."""
    e0 = axis / (np.linalg.norm(axis) + 1e-12)
    s = second - e0 * float(second @ e0)
    if np.linalg.norm(s) < 1e-8:
        s = np.eye(3)[int(np.argmin(np.abs(e0)))]
        s = s - e0 * float(s @ e0)
    e1 = s / (np.linalg.norm(s) + 1e-12)
    e2 = np.cross(e0, e1)
    return np.stack([e0, e1, e2])


def principal_axes(X):
    """주축 세 개 (분산이 큰 순서). 물체의 길이 방향을 찾는 데 쓴다."""
    X = X[np.isfinite(X).all(1)] if not np.isfinite(X).all() else X
    Xc = X - X.mean(0)
    C = (Xc.T @ Xc) / max(len(X), 1)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    return V[:, o].T, w[o]


def pick_pinch_grips(X, jaw=0.12, pad_w=0.35, pad_d=0.18, grab=0.12,
                     n_grips=2):
    """양 끝에 평행 집게를 하나씩 놓고, 집게 사이에 낀 입자를 고른다.

    jaw    집게 간격 (물체 지름 대비) -- 두 판 사이 두께
    pad_w  판의 폭   (물체 지름 대비)
    pad_d  판의 깊이 (당기는 축 방향 두께)
    grab   양 끝에서 얼마나 안쪽까지를 집게 자리로 볼지 (물체 길이 대비)

    반환: grips = [{c, R, members, off}], 여기서 off 는 **집게 좌표계** 기준
    상대위치라 회전을 주면 그대로 비틀림이 된다.
    """
    ax, _ = principal_axes(X)
    a = ax[0]                              # 길이 방향 = 당기는 축
    ext = float(np.linalg.norm(X.max(0) - X.min(0)))
    t = X @ a
    lo, hi = t.min(), t.max()
    span = hi - lo
    grips = []
    for s in ([+1, -1] if n_grips >= 2 else [+1])[:n_grips]:
        # 끝에서 grab 만큼 안쪽 구간의 무게중심을 집게 중심으로
        sel = (t > hi - grab * span) if s > 0 else (t < lo + grab * span)
        if sel.sum() < 10:
            sel = (t > hi - 0.25 * span) if s > 0 else (t < lo + 0.25 * span)
        c = X[sel].mean(0)
        # 집게가 물릴 방향: 그 자리에서 가장 얇은 방향을 집게 축으로 잡는다
        loc, w = principal_axes(X[sel] - c)
        R = _frame_from(a * s, loc[0] if abs(loc[0] @ a) < 0.9 else loc[1])
        R = np.stack([R[0], R[2], R[1]])   # (당기는 축, 폭, 집게 축)
        d = (X - c) @ R.T                  # 집게 좌표계
        m = np.flatnonzero((np.abs(d[:, 0]) < pad_d * ext)
                           & (np.abs(d[:, 1]) < pad_w * ext)
                           & (np.abs(d[:, 2]) < 0.5 * jaw * ext))
        grips.append(dict(c=c, R=R, members=m, off=(X[m] - c)))
    return grips, a, ext


def tear_traj(grips, axis, ext, steps, dt, speed=0.15, twist=0.0):
    """양쪽으로 당기면서(속도 speed) 축 둘레로 비트는(각속도 twist) SE(3) 궤적.

    speed, twist 는 각각 물체 지름/초, 라디안/초. 집게 표면 속도가
    speed + twist * (판 반경) 이므로 둘을 같이 올리면 빨라진다.
    """
    P = np.zeros((steps, len(grips), 3), np.float64)
    R = np.zeros((steps, len(grips), 3, 3), np.float64)
    V = np.zeros((steps, len(grips), 3), np.float64)
    W = np.zeros((steps, len(grips), 3), np.float64)
    for k, g in enumerate(grips):
        s = float(np.sign(g["R"][0] @ axis)) or 1.0
        u = axis * s
        for i in range(steps):
            tt = i * dt
            P[i, k] = g["c"] + u * (speed * ext * tt)
            th = twist * tt * s
            ca, sa = np.cos(th), np.sin(th)
            K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            R[i, k] = np.eye(3) + sa * K + (1 - ca) * (K @ K)   # 로드리게스
            V[i, k] = u * (speed * ext)
            W[i, k] = u * (twist * s)
    return P, R, V, W


def pick_pad_at(X, center, normal, jaw=0.10, pad_w=0.25, pad_d=0.25, ext=None):
    """표면 한 점에 집게 패드를 대고 그 아래 덩어리를 잡는다.

    법선을 첫 축으로 하는 국소 좌표계를 만들고, 법선 방향으로 얕게(jaw), 접선
    방향으로 넓게(pad_w, pad_d) 뻗은 상자 안 입자를 고른다. 손가락 끝이 표면을
    집었을 때 딸려오는 살점에 해당한다.
    """
    ext = ext if ext is not None else float(np.linalg.norm(X.max(0) - X.min(0)))
    R = _frame_from(normal, np.eye(3)[int(np.argmin(np.abs(normal)))])
    d = (X - center) @ R.T
    m = np.flatnonzero((np.abs(d[:, 0]) < jaw * ext)
                       & (np.abs(d[:, 1]) < pad_w * ext)
                       & (np.abs(d[:, 2]) < pad_d * ext))
    return m, R


def random_moves(rng, n, speed_lo=0.3, speed_hi=0.7, twist_hi=1.5):
    """에피소드마다 방향(단위벡터)·속도·비틀기를 뽑는다."""
    out = []
    for _ in range(n):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v) + 1e-12
        out.append(dict(dir=v, speed=float(rng.uniform(speed_lo, speed_hi)),
                        twist=float(rng.uniform(-twist_hi, twist_hi))))
    return out
