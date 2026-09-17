"""입자를 그대로 그리는 최소 렌더러.

학습이 보는 것은 입자 위치뿐이고 원본 h5 는 압축 뒤 지웠으므로, 공식 래스터라이저를
다시 태우는 대신 그 입자를 직접 그린다 -- 학습이 보는 것과 영상이 보는 것을
일치시키기 위해서다.

두 가지 규칙이 이 렌더러의 전부다.

  색은 **정준 위치**로 고정한다. 그러면 갈라져 나간 조각이 원래 어디 있던 재질인지
  드러난다. 현재 위치로 칠하면 조각이 움직이는 동안 색이 따라 변해 그 정보가 사라진다.

  화면 범위는 **궤적 전체**로 한 번만 잡는다. 프레임마다 맞추면 물체가 떨어지는
  것인지 카메라가 따라가는 것인지 구별할 수 없다.
"""
from __future__ import annotations

import numpy as np
import torch


def camera(elev, azim, device="cuda"):
    """정사영 카메라의 회전. 행이 (right, up, forward)."""
    e, z = np.radians(elev), np.radians(azim)
    fwd = np.array([np.cos(e) * np.cos(z), np.cos(e) * np.sin(z), np.sin(e)])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return torch.tensor(np.stack([right, up, fwd]), dtype=torch.float32,
                        device=device)


def frame_box(pts, R, width, margin=0.55):
    """-> (중심, 반폭, W, H). pts 는 [*,3] 전체 궤적."""
    q = pts.reshape(-1, 3) @ R.T
    lo, hi = q[:, :2].min(0).values, q[:, :2].max(0).values
    ctr = 0.5 * (lo + hi)
    half = margin * float((hi - lo).max())
    H = int(round(width * float(hi[1] - lo[1] + 1e-6)
                  / float(hi[0] - lo[0] + 1e-6)))
    return ctr, half, int(width), max(min(H, 2 * width), width // 2)


def canon_color(X0):
    """정준 위치를 색으로. [N,3] -> [N,3] in [0.25, 0.95]"""
    c = X0 - X0.min(0).values
    return 0.25 + 0.7 * c / c.max(0).values.clamp(min=1e-9)


def splat(x, col, R, ctr, half, W, H, point=1, bg=1.0):
    """[N,3] 점을 [H,W,3] 이미지로. 뒤에서 앞으로 그려 앞의 점이 이긴다."""
    p = x @ R.T
    u = ((p[:, 0] - ctr[0]) / half * 0.5 + 0.5) * (W - 1)
    v = (0.5 - (p[:, 1] - ctr[1]) / half * 0.5) * (H - 1)
    ok = torch.isfinite(u) & torch.isfinite(v)
    ui = u.round().long().clamp(0, W - 1)[ok]
    vi = v.round().long().clamp(0, H - 1)[ok]
    o = torch.argsort(p[:, 2][ok], descending=True)
    ui, vi, cc = ui[o], vi[o], col[ok][o]
    img = torch.full((H, W, 3), bg, device=x.device)
    flat = img.reshape(-1, 3)
    for dy in range(-point, point + 1):
        for dx in range(-point, point + 1):
            flat[(vi + dy).clamp(0, H - 1) * W + (ui + dx).clamp(0, W - 1)] = cc
    return img.clamp(0, 1)
