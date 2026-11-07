"""절 CM23 (로컬): 수박 손상 logJp 상위 10·20·30 % 표시 비교 영상 합성 -- 2 줄 x 2 칸 (각 600x600) + 칸마다 그 프레임 문턱·실제 몫·비유한 몫 글자 + 범례.
  윗줄: 정답 (자기 분포 백분위) | 우리 기본 판 K1 (자기 분포 백분위)
  아랫줄: 정답 일반 3DGS 색 | 우리 기본 판 K1 (정답 같은 프레임 문턱 값을 그대로 적용)
  python3 exe/wm_dmgtop_compose.py --dir <vid 폴더 (gt_top·ours_top·ours_gtthr·gt_plain .mp4, dmgtop.json)> --out out.mp4
"""
import argparse
import json
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402
from matplotlib.patches import Patch                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--font", default="/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf")
ap.add_argument("--title", default="수박 손상 변수 logJp 상위 10·20·30 % 입자 -- 정답 대 우리 기본 판 (프레임당 1 걸음)")
a = ap.parse_args()
font_manager.fontManager.addfont(a.font)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=a.font).get_name()
J = json.load(open(os.path.join(a.dir, "dmgtop.json")))
T = tempfile.mkdtemp()
V = dict(gt="gt_top", ours="ours_top", gtp="gt_plain", ogt="ours_gtthr")
n = 10 ** 9
for k, v in V.items():
    os.makedirs(f"{T}/{k}")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", os.path.join(a.dir, v + ".mp4"), f"{T}/{k}/%04d.png"], check=True)
    n = min(n, len(os.listdir(f"{T}/{k}")))
os.makedirs(f"{T}/o")
PAL = [("상위 10 %", (0.85, 0.10, 0.10)), ("10~20 %", (1.00, 0.55, 0.00)), ("20~30 %", (0.95, 0.85, 0.05)), ("나머지 (반투명)", (0.55, 0.55, 0.55)), ("비유한 logJp (백분위에서 뺌)", (0.55, 0.15, 0.75))]


def txt(r, own):
    q = r["thr"]; f = r["frac"]
    s = ("문턱 (%s) q90 %.3g · q80 %.3g · q70 %.3g" % ("자기 분포" if own else "정답 문턱 그대로", q[0], q[1], q[2]))
    s += "\n실제 몫 빨강 %.1f · 주황 %.1f · 노랑 %.1f %% · 비유한 %.1f %%" % (100 * f[0], 100 * f[1], 100 * f[2], 100 * r["nf"])
    return s


for i in range(n):
    fig = plt.figure(figsize=(12, 13.6), dpi=100)
    fig.text(0.5, 0.985, a.title, ha="center", va="top", fontsize=15)
    fig.text(0.5, 0.958, "프레임 %d · 색 = 그 프레임 logJp 백분위 범주 (3DGS 알파블렌딩, 가우시안 색만 바꿈); 정답 = 원래 정답 위치·F + GF 다시 돌린 logJp" % i, ha="center", va="top", fontsize=10)
    cells = [("gt", "정답 (자기 분포 백분위)", J["gt"], True), ("ours", "우리 기본 판 K1 (자기 분포 백분위)", J["ours"], True),
             ("gtp", "정답 일반 색", None, None), ("ogt", "우리 기본 판 K1 (정답 문턱 적용)", J["ours_gtthr"], False)]
    for j, (k, lab, L, own) in enumerate(cells):
        x0 = 0.0 + 0.5 * (j % 2); y0 = 0.50 - 0.465 * (j // 2)
        ax = fig.add_axes([x0, y0, 0.5, 0.42]); ax.axis("off")
        ax.imshow(plt.imread(f"{T}/{k}/{i + 1:04d}.png"))
        fig.text(x0 + 0.25, y0 + 0.423, lab, ha="center", va="bottom", fontsize=13)
        if L is not None:
            r = L[min(i, len(L) - 1)]
            fig.text(x0 + 0.01, y0 + 0.005, txt(r, own), ha="left", va="bottom", fontsize=9.5, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.legend(handles=[Patch(color=c, label=l) for l, c in PAL], loc="lower center", ncol=5, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(f"{T}/o/{i:04d}.png"); plt.close(fig)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(a.fps), "-i", f"{T}/o/%04d.png", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", a.out], check=True)
print("영상", a.out, n, "프레임")
