"""학습에 실제로 들어가는 궤적을 그대로 그린다.

원본 h5 는 한 궤적에 9.4GB 라 압축 뒤 지웠고, 학습이 보는 것은 입자 40,000 개의
위치뿐이다. 그래서 공식 렌더러를 다시 태우는 대신 **그 입자를 직접** 그린다 --
학습이 보는 것과 영상이 보는 것이 같아야 하기 때문이다.

색은 **정준 위치**로 고정한다. 그러면 갈라져 나간 조각이 원래 어디 있던 재질인지
그대로 드러나서, 파괴가 실제로 일어나는지 눈으로 확인할 수 있다. 현재 위치로
칠하면 조각이 움직이는 동안 색이 따라 변해 그 정보가 사라진다.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help=".pt 하나 또는 그것들이 든 디렉토리")
ap.add_argument("--out", required=True)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--point", type=int, default=1, help="점 반경(픽셀)")
ap.add_argument("--elev", type=float, default=12.0)
ap.add_argument("--azim", type=float, default=35.0)
ap.add_argument("--stride", type=int, default=1)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
import imageio.v2 as imageio                                    # noqa: E402
from PIL import Image, ImageDraw                                # noqa: E402

files = ([a.data] if a.data.endswith(".pt")
         else sorted(glob.glob(os.path.join(a.data, "*.pt"))))
os.makedirs(a.out, exist_ok=True)


def camera(X0, elev, azim):
    """물체를 담는 정사영 카메라. 궤적 전체를 담도록 여유를 준다."""
    e, z = np.radians(elev), np.radians(azim)
    fwd = np.array([np.cos(e) * np.cos(z), np.cos(e) * np.sin(z), np.sin(e)])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return torch.tensor(np.stack([right, up, fwd]), dtype=torch.float32,
                        device=dev)


for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    X = d["x"][::a.stride]
    T, N, _ = X.shape
    tag = os.path.splitext(os.path.basename(f))[0]
    Xc = X[0].to(dev)
    R = camera(Xc, a.elev, a.azim)
    # 화면 범위는 **궤적 전체**로 잡는다. 프레임마다 다시 맞추면 물체가 떨어지는
    # 것인지 카메라가 따라가는 것인지 구별할 수 없다.
    allp = X.reshape(-1, 3).to(dev) @ R.T
    lo = allp[:, :2].min(0).values
    hi = allp[:, :2].max(0).values
    ctr = 0.5 * (lo + hi)
    half = 0.55 * float((hi - lo).max())
    W = a.width
    H = int(round(W * float(hi[1] - lo[1] + 1e-6) / float(hi[0] - lo[0] + 1e-6)))
    H = max(min(H, 2 * W), W // 2)

    # 정준 위치 -> 색
    c0 = Xc - Xc.min(0).values
    c0 = c0 / c0.max(0).values.clamp(min=1e-9)
    col = (0.25 + 0.7 * c0)

    frames = []
    for t in range(T):
        p = X[t].to(dev) @ R.T
        u = ((p[:, 0] - ctr[0]) / half * 0.5 + 0.5) * (W - 1)
        v = (0.5 - (p[:, 1] - ctr[1]) / half * 0.5) * (H - 1)
        ok = torch.isfinite(u) & torch.isfinite(v)
        ui = u.round().long().clamp(0, W - 1)[ok]
        vi = v.round().long().clamp(0, H - 1)[ok]
        depth = p[:, 2][ok]
        cc = col[ok]
        # 앞에 있는 점이 이기도록 깊이 순으로 그린다 (뒤에서 앞으로)
        o = torch.argsort(depth, descending=True)
        ui, vi, cc = ui[o], vi[o], cc[o]
        img = torch.ones(H, W, 3, device=dev)
        idx = vi * W + ui
        for dy in range(-a.point, a.point + 1):
            for dx in range(-a.point, a.point + 1):
                jj = ((vi + dy).clamp(0, H - 1) * W + (ui + dx).clamp(0, W - 1))
                img.reshape(-1, 3)[jj] = cc
        arr = (img.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        im = Image.fromarray(arr); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W, 16], fill=(0, 0, 0))
        c = d["cfg"]
        dr.text((4, 3), f"{tag}  E={c['E']:g} nu={c['nu']:g} xi={c.get('xi',0):g}"
                        f"  f{t*a.stride:03d}/{X.shape[0]*a.stride}",
                fill=(255, 255, 255))
        frames.append(np.array(im))
    p_out = os.path.join(a.out, f"{tag}.mp4")
    imageio.mimsave(p_out, frames, fps=a.fps, quality=8)
    print(f"[저장] {p_out}  {T} 프레임 x {N} 입자, {W}x{H}", flush=True)
print("RENDER_OK")
