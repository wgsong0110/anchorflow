#!/usr/bin/env python3
"""절 CM194: 같은 프레임의 1x1x1 대 2x2x2 렌더 확대 비교 -> fig/gq_closeup_<장면>.png.
  python exe/gqj_closeup.py <장면> <q1 png 폴더> <def png 폴더> <q1 gq.json> <출력 png> [정답 png 폴더]
프레임 = 1x1x1 판의 모래시계 에너지 비가 가장 큰 프레임 (같은 프레임 번호의 두 렌더); 확대 자리 = 두 렌더 차이 (상자 평균) 가 가장 큰 곳."""
import glob, json, os, sys
import numpy as np
from PIL import Image
class imageio:
    imread = staticmethod(lambda p: np.asarray(Image.open(p).convert("RGB")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
for fp in (os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf"), "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp); plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name(); break
sc, P1, P8, J, OUT = sys.argv[1:6]
REF = sys.argv[6] if len(sys.argv) > 6 else ""
f1 = sorted(glob.glob(f"{P1}/*.png")); f8 = sorted(glob.glob(f"{P8}/*.png"))
nf = min(len(f1), len(f8))
dif = [float(np.abs(imageio.imread(f1[i])[::4, ::4, :3].astype(np.float32) - imageio.imread(f8[i])[::4, ::4, :3].astype(np.float32)).mean()) for i in range(nf)]
fi = int(np.argmax(dif))                                                              # 두 렌더 차이가 가장 큰 프레임
hg = json.load(open(J)).get("hgm_frames") or []
A = imageio.imread(f1[fi])[..., :3].astype(np.float32) / 255; B = imageio.imread(f8[fi])[..., :3].astype(np.float32) / 255
D = np.abs(A - B).sum(-1)
H, W = D.shape; w = max(64, min(H, W) // 5)
cs = np.pad(np.cumsum(np.cumsum(D, 0), 1), ((1, 0), (1, 0)))
S = cs[w:, w:] - cs[:-w, w:] - cs[w:, :-w] + cs[:-w, :-w]
y0, x0 = np.unravel_index(np.argmax(S), S.shape)
imgs = [("$1^3$ (칸당 1 점)", A), ("$2^3$ (기본)", B)]
if REF and os.path.isdir(REF):
    fr_ = sorted(glob.glob(f"{REF}/*.png"))
    if fi < len(fr_):
        imgs.append(("정답 MPM", imageio.imread(fr_[fi])[..., :3].astype(np.float32) / 255))
fig, axs = plt.subplots(2, len(imgs), figsize=(3.2 * len(imgs), 6.4), constrained_layout=True)
for j, (nm, I) in enumerate(imgs):
    axs[0, j].imshow(I); axs[0, j].add_patch(Rectangle((x0, y0), w, w, fill=False, ec="red", lw=1.5)); axs[0, j].set_title(nm, fontsize=11); axs[0, j].axis("off")
    axs[1, j].imshow(I[y0:y0 + w, x0:x0 + w], interpolation="nearest"); axs[1, j].set_title("확대", fontsize=10); axs[1, j].axis("off")
_h = hg[fi - 1] if 0 < fi <= len(hg) and hg[fi - 1] is not None else None
fig.suptitle(f"{'Lego' if sc == 'L000' else sc}, 프레임 {fi} (두 렌더 차이가 가장 큰 프레임" + (f", 1$^3$ 모래시계 에너지 비 {_h:.2f}" if _h is not None else "") + ")", fontsize=11)
fig.savefig(OUT, dpi=150)
print("ok", sc, fi, x0, y0, w)
