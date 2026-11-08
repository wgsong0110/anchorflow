"""절 CM54: 실시간 렌더 확인 PNG (<run>/rrimg/<h5 번호 4 자리>.png) 대 영상 렌더 경로 PNG (같은 덤프를 lb_roll_to_h5 + lb_render/wm_render 로) 화면 대조 -> json.

  python exe/rtrender1_cmp.py --a RUN/rrimg --b OFFDIR --bfmt %04d
"""
import argparse
import glob
import json
import os

import imageio.v2 as imageio
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--bfmt", default="%04d")
a = ap.parse_args()
R = []
for f in sorted(glob.glob(os.path.join(a.a, "*.png"))):
    t = int(os.path.basename(f)[:-4])
    g = os.path.join(a.b, (a.bfmt % t) + ".png")
    if not os.path.exists(g):
        R.append(dict(t=t, miss=True)); continue
    x = imageio.imread(f).astype(np.int32)[..., :3]; y = imageio.imread(g).astype(np.int32)[..., :3]
    if x.shape != y.shape:
        R.append(dict(t=t, shape=[list(x.shape), list(y.shape)])); continue
    d = np.abs(x - y)
    mse = float((d.astype(np.float64) ** 2).mean())
    R.append(dict(t=t, max=int(d.max()), mean=float(d.mean()), n_px_diff=int((d > 0).any(2).sum()), n_px_gt2=int((d > 2).any(2).sum()),
                  psnr=(99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))), hw=list(x.shape[:2])))
ok = [r for r in R if "max" in r]
print(json.dumps(dict(n=len(R), n_cmp=len(ok), max=max([r["max"] for r in ok], default=None), psnr_min=min([r["psnr"] for r in ok], default=None),
                      px_diff_max=max([r["n_px_diff"] for r in ok], default=None), px_gt2_max=max([r["n_px_gt2"] for r in ok], default=None), frames=R), ensure_ascii=False))
