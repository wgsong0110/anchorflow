"""절 CM13.4: 손상 색 비교 영상 합성 (로컬) -- 정답 손상 | 우리 손상 + 색 막대 + 제목·범례 (글자는 그림으로 만들어 얹음).

  python3 exe/wmd_compose.py --gt gt_lj.mp4 --ours ours_lj.mp4 --label_gt .. --label_ours .. --note .. --out out.mp4
"""
import argparse
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402
import numpy as np                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True)
ap.add_argument("--ours", required=True)
ap.add_argument("--label_gt", required=True)
ap.add_argument("--label_ours", required=True)
ap.add_argument("--note", required=True)
ap.add_argument("--vmin", type=float, default=-0.2)
ap.add_argument("--vmax", type=float, default=0.4)
ap.add_argument("--cmap", default="viridis")
ap.add_argument("--fps", type=int, default=32)
ap.add_argument("--font", default="/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf")
ap.add_argument("--out", required=True)
a = ap.parse_args()
font_manager.fontManager.addfont(a.font)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=a.font).get_name()
plt.rcParams["axes.unicode_minus"] = False
D = tempfile.mkdtemp()
W, H, TOP, BOT, CB = 600, 600, 60, 70, 240
# 색 막대
fig = plt.figure(figsize=(CB / 100, (H) / 100), dpi=100)
cax = fig.add_axes([0.12, 0.06, 0.16, 0.88])
sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(a.vmin, a.vmax), cmap=a.cmap)
cb = fig.colorbar(sm, cax=cax, extend="both")
cb.set_ticks([-0.2, -0.04, 0.0, 0.2, 0.4])
cb.set_ticklabels(["−0.2", "−0.04 (처음)", "0 (응집력 0)", "0.2", "0.4 (GF 숨김)"])
cb.ax.tick_params(labelsize=10)
cb.set_label("logJp (NACC 경화) -- 모든 프레임 공통 범위", fontsize=10)
fig.savefig(f"{D}/cb.png", facecolor="white"); plt.close(fig)


def txt(path, w, h, s, fs):
    f = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    f.text(0.5, 0.5, s, ha="center", va="center", fontsize=fs, wrap=True)
    f.savefig(path, facecolor="white"); plt.close(f)


txt(f"{D}/t1.png", W, TOP, a.label_gt, 13)
txt(f"{D}/t2.png", W, TOP, a.label_ours, 13)
txt(f"{D}/nb.png", 2 * W + CB, BOT, a.note, 10)
TW = 2 * W + CB; TH = TOP + H + BOT
fc = (f"color=white:s={TW}x{TH}:r={a.fps}[bg];[0:v]scale={W}:{H}[g];[1:v]scale={W}:{H}[o];"
      f"[bg][g]overlay=0:{TOP}:shortest=1[a1];[a1][o]overlay={W}:{TOP}:shortest=1[a2];"
      f"[a2][2:v]overlay=0:0[a3];[a3][3:v]overlay={W}:0[a4];[a4][4:v]overlay={2 * W}:{TOP}[a5];[a5][5:v]overlay=0:{TOP + H}[a6];"
      f"[a6]drawtext=fontfile={a.font}:text='프레임 %{{eif\\:n\\:d}}':x={2 * W + 20}:y=15:fontsize=22:fontcolor=black,format=yuv420p[v]")
NF = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", a.gt],
                        capture_output=True, text=True).stdout.strip())
NF = min(NF, int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", a.ours],
                                capture_output=True, text=True).stdout.strip()))
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", a.gt, "-i", a.ours, "-loop", "1", "-i", f"{D}/t1.png", "-loop", "1", "-i", f"{D}/t2.png",
       "-loop", "1", "-i", f"{D}/cb.png", "-loop", "1", "-i", f"{D}/nb.png", "-filter_complex", fc, "-map", "[v]", "-frames:v", str(NF),
       "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
subprocess.run(cmd, check=True)
print("[합성]", a.out, f"{TW}x{TH}")
