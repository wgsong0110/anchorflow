"""[stab4] 장면마다 최소 N 비교 영상: 3x2 칸 (정답 = PG N1024 | PG | i-PG / GF | OmniPhysGS | 우리), 칸 위에 한국어 이름·N.
입력 = 드라이버 vis 단계 영상 (<장면>/vis/<이름>.mp4, 같은 렌더러 rep_ip F 반영 3DGS). 없는 칸은 회색 + '없음'. 가로 2400 이하, yuv420p, H.264 faststart.
  python exe/stab4_vid.py --vis <장면 vis 폴더> --scene 레고 --items 'pg_N1024=정답 (PG N=1024)' 'pg_N43=PG N=43' ... --font <otf> --out out.mp4
"""
import argparse
import os
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--vis", required=True)
ap.add_argument("--scene", required=True)
ap.add_argument("--items", nargs="+", required=True)
ap.add_argument("--font", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pw", type=int, default=800)
ap.add_argument("--cols", type=int, default=3)
a = ap.parse_args()
vids = []
for it in a.items:
    nm, lab = it.split("=", 1)
    p = f"{a.vis}/{nm}.mp4"
    fr = [np.asarray(f)[..., :3] for f in imageio.get_reader(p)] if os.path.exists(p) else None
    vids.append((lab, fr))
T = max(len(f) for _, f in vids if f)
h0, w0 = next(f[0].shape[:2] for _, f in vids if f)
PH = int(round(a.pw * h0 / w0 / 2)) * 2; BAR = 44
font = ImageFont.truetype(a.font, 30)
rows = (len(vids) + a.cols - 1) // a.cols
W, H = a.pw * a.cols, (PH + BAR) * rows
cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", a.out]
pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for t in range(T):
    can = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(can)
    for i, (lab, fr) in enumerate(vids):
        x, y = (i % a.cols) * a.pw, (i // a.cols) * (PH + BAR)
        if fr:
            im = Image.fromarray(fr[min(t, len(fr) - 1)]).resize((a.pw, PH), Image.BILINEAR)
            can.paste(im, (x, y + BAR))
        else:
            d.rectangle([x, y + BAR, x + a.pw - 1, y + BAR + PH - 1], fill=(200, 200, 200)); lab += " (없음)"
        d.text((x + 10, y + 6), lab, fill=(0, 0, 0), font=font)
    d.text((W - 260, H - 40), f"{a.scene}  프레임 {t}", fill=(80, 80, 80), font=ImageFont.truetype(a.font, 24))
    pr.stdin.write(np.asarray(can).tobytes())
pr.stdin.close(); pr.wait()
print("ok", a.out, W, H, T)
