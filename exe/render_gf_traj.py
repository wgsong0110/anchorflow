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
import time
import glob
import os
import sys

import numpy as np
import torch

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None, help=".pt 하나 또는 그것들이 든 디렉토리")
ap.add_argument("--h5_dir", default=None,
                help="압축 전 sim_*.h5 를 바로 그린다. 시뮬을 막 돌리고 .pt 로 "
                     "줄이기 전에 눈으로 확인할 때 쓴다")
ap.add_argument("--h5_pts", type=int, default=60000, help="h5 에서 뽑을 입자 수")
ap.add_argument("--tag", default="run")
ap.add_argument("--out", required=True)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--point", type=int, default=1, help="점 반경(픽셀)")
ap.add_argument("--elev", type=float, default=12.0)
ap.add_argument("--azim", type=float, default=35.0)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--up", default="z", choices=("x", "y", "z"),
                help="그 씬에서 **위**가 어느 축인지. 렌더러의 카메라는 세계 +z 를 "
                     "위로 잡으므로, 중력이 -y 인 씬(타이치 공식 mpm3d)을 그냥 "
                     "그리면 옆으로 떨어지는 것처럼 보인다")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
import imageio.v2 as imageio                                    # noqa: E402
from PIL import Image, ImageDraw                                # noqa: E402

from anchorflow import ptrender                                 # noqa: E402

os.makedirs(a.out, exist_ok=True)
_T0 = time.time()


def load_h5_traj(h5_dir, n_pts, stride):
    """sim_*.h5 를 .pt 와 같은 모양으로 읽는다."""
    import h5py
    fs = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))[::stride]
    if not fs:
        raise SystemExit(f"h5 가 없다: {h5_dir}")

    def rd(p):
        with h5py.File(p, "r") as h:
            d = np.array(h["x"])
        return torch.from_numpy(d.T if d.shape[0] == 3 else d).float()

    X0 = rd(fs[0]); XL = rd(fs[-1])
    ok = torch.isfinite(X0).all(1) & torch.isfinite(XL).all(1)
    cand = torch.nonzero(ok).squeeze(-1)
    g = torch.Generator().manual_seed(0)
    sel = cand[torch.randperm(cand.numel(), generator=g)[:n_pts]].sort().values
    X = torch.stack([rd(p)[sel] for p in fs])
    bad = ~torch.isfinite(X).all(-1)
    for t in range(1, X.shape[0]):
        if bad[t].any():
            X[t][bad[t]] = X[t - 1][bad[t]]
    print(f"[h5] {len(fs)} 프레임 x {sel.numel()} 입자 (전체 {X0.shape[0]})",
          flush=True)
    return X


if a.h5_dir:
    files = [None]
elif a.data:
    files = ([a.data] if a.data.endswith(".pt")
             else sorted(glob.glob(os.path.join(a.data, "*.pt"))))
else:
    raise SystemExit("--data 나 --h5_dir 중 하나는 있어야 한다")

UPPERM = {"z": [0, 1, 2], "y": [2, 0, 1], "x": [1, 2, 0]}[a.up]

for f in files:
    if f is None:
        d = {"cfg": {}}
        X = load_h5_traj(a.h5_dir, a.h5_pts, a.stride)
    else:
        d = torch.load(f, map_location="cpu", weights_only=False)
        X = d["x"][::a.stride]
    X = X[..., UPPERM]           # 씬의 위 축을 세계 +z 로 돌린다
    T, N, _ = X.shape
    tag = a.tag if f is None else os.path.splitext(os.path.basename(f))[0]
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
        mat = (f"E={c['E']:g} nu={c['nu']:g} xi={c.get('xi',0):g}" if c else "")
        dr.text((4, 3), f"{tag}  {mat}"
                        f"  f{t*a.stride:03d}/{X.shape[0]*a.stride}",
                fill=(255, 255, 255))
        frames.append(np.array(im))
    _t_draw = time.time() - _T0
    p_out = os.path.join(a.out, f"{tag}.mp4")
    _t_enc = time.time()
    imageio.mimsave(p_out, frames, fps=a.fps, quality=8)
    _t_enc = time.time() - _t_enc
    print(f"[시간] 읽기+그리기 {_t_draw:.1f}s ({_t_draw / max(T,1)*1e3:.0f} ms/프레임)"
          f" + 인코딩 {_t_enc:.1f}s", flush=True)
    print(f"[저장] {p_out}  {T} 프레임 x {N} 입자, {W}x{H}", flush=True)
print("RENDER_OK")
