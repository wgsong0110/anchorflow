"""여러 mp4 를 같은 프레임에 맞춘 격자로 합친다. 칸마다 제목 띠 (두 줄: 이름 / 부제), 맨 위 머리 줄.

칸 크기 = 첫 칸 영상 크기 (다르면 그 크기로 맞추고 알린다). 프레임 수가 다르면 가장 짧은 것에 맞추고 알린다. 빈 칸은 흰색.

  python exe/allmodels_grid.py --cell "정답 (MPM)||gt.mp4" --cell "모델|오차 1.0% · 반복 15000|m.mp4" --cols 4 \
      --font NotoSansCJKkr-Regular.otf --head "궤적 L000" --out grid.mp4
"""
import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--cell", action="append", required=True, help="제목|부제|mp4")
ap.add_argument("--cols", type=int, default=4)
ap.add_argument("--font", required=True)
ap.add_argument("--head", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=0, help="0 이면 첫 영상 fps")
a = ap.parse_args()

cells = [c.split("|", 2) for c in a.cell]
vids = []
fps = a.fps
for t, s, p in cells:
    r = imageio.get_reader(p)
    fr = [np.asarray(f)[..., :3] for f in r]
    if not fps:
        fps = int(round(r.get_meta_data().get("fps", 30)))
    r.close()
    vids.append(fr)
    print(f"[칸] {t} | {s} | {p}: {len(fr)} 프레임 {fr[0].shape[1]}x{fr[0].shape[0]}", flush=True)
H, Wd = vids[0][0].shape[:2]
T = min(len(v) for v in vids)
if len({len(v) for v in vids}) > 1:
    print(f"[주의] 프레임 수가 다르다 {[len(v) for v in vids]} -> {T} 에 맞춤", flush=True)
for i, v in enumerate(vids):
    if v[0].shape[:2] != (H, Wd):
        print(f"[주의] 칸 {i} 크기 {v[0].shape[1]}x{v[0].shape[0]} -> {Wd}x{H}", flush=True)
        vids[i] = [np.asarray(Image.fromarray(f).resize((Wd, H), Image.BICUBIC)) for f in v]
nc = a.cols; nr = (len(cells) + nc - 1) // nc
fs = max(14, Wd // 22)
f1 = ImageFont.truetype(a.font, fs); f2 = ImageFont.truetype(a.font, int(fs * 0.85)); fh = ImageFont.truetype(a.font, int(fs * 1.1))
TB = int(fs * 2.7)                 # 제목 띠 높이
HD = int(fs * 1.8) if a.head else 0
OW, OH = nc * Wd, HD + nr * (TB + H)
OW += OW % 2; OH += OH % 2
# 글자 판 (프레임마다 같은 것)
base = Image.new("RGB", (OW, OH), (255, 255, 255))
dr = ImageDraw.Draw(base)
if a.head:
    dr.text((10, int(fs * 0.3)), a.head, fill=(0, 0, 0), font=fh)


def fit(text, font, width):
    while font.getlength(text) > width - 8 and len(text) > 4:
        text = text[:-2] + "…"
    return text


for i, (t, s, _) in enumerate(cells):
    r, c = divmod(i, nc)
    x, y = c * Wd, HD + r * (TB + H)
    dr.text((x + 6, y + 2), fit(t, f1, Wd), fill=(0, 0, 0), font=f1)
    if s:
        dr.text((x + 6, y + 2 + int(fs * 1.25)), fit(s, f2, Wd), fill=(150, 0, 0), font=f2)
base = np.asarray(base).copy()
w = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=8, macro_block_size=1)
for k in range(T):
    fr = base.copy()
    for i in range(len(cells)):
        r, c = divmod(i, nc)
        x, y = c * Wd, HD + r * (TB + H) + TB
        fr[y:y + H, x:x + Wd] = vids[i][k]
    w.append_data(fr)
w.close()
print(f"[저장] {a.out} {T} 프레임 {OW}x{OH} fps {fps}, 칸 {len(cells)} ({nr}행 {nc}열)", flush=True)
