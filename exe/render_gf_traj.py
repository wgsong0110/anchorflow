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

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

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

from anchorflow import ptrender                                 # noqa: E402

files = ([a.data] if a.data.endswith(".pt")
         else sorted(glob.glob(os.path.join(a.data, "*.pt"))))
os.makedirs(a.out, exist_ok=True)


for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    X = d["x"][::a.stride]
    T, N, _ = X.shape
    tag = os.path.splitext(os.path.basename(f))[0]
    Xc = X[0].to(dev)
    R = ptrender.camera(a.elev, a.azim, dev)
    # 화면 범위와 색은 라이브러리 규칙을 그대로 쓴다 (정준색, 궤적 전체 범위)
    ctr, half, W, H = ptrender.frame_box(X.to(dev), R, a.width)
    col = ptrender.canon_color(Xc)

    frames = []
    for t in range(T):
        img = ptrender.splat(X[t].to(dev), col, R, ctr, half, W, H, a.point)
        arr = (img.cpu().numpy() * 255).astype("uint8")
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
