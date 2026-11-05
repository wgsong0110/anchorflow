"""절 CM14.3 영상 합성 (인스턴스, python + ffmpeg): 칸 여러 개를 두 줄 격자로 (가로 4096 이하 -- 브라우저 h264 한도), 오른쪽에 장벽 에너지 색 막대 범례.
칸 위 글자 띠 = 판 이름·오차 (fem_barnode_compose.py 와 같은 꼴), 칸 영상은 비율 그대로 높이 HP 로, 짧은 영상·발산한 판은 마지막 프레임에서 정지.
색 막대는 exe/lb_render.py --cells 와 같은 사상 (log10 에너지 -> inferno 0.05~0.9, 고정 범위 --erange).
사용: python exe/fem_bener_compose.py <출력 mp4> <제목> <글꼴 Bold> <글꼴 Regular> <erange lo,hi> <범례 설명 (| 로 줄 나눔)> <열 수> [<칸 이름> <영상> <summary.json 또는 ->]..."""
import json
import os
import sys

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

out, title, FB, FR, ER, LEG, NCOL = sys.argv[1:8]
rest = sys.argv[8:]
assert len(rest) % 3 == 0, rest
ELO, EHI = [float(q) for q in ER.split(",")]
NCOL = int(NCOL)
CMAP = matplotlib.colormaps["inferno"]
TOP, CBW, TT, WMAX = 110, 420, 70, 4096


def er(p):
    if p == "-":
        return None, None
    s = json.load(open(p))
    dv = s.get("diverged")
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    return f"오차 {100 * e:.2f}%" + (f" · 프레임 {dv} 에서 발산 (뒤는 정지)" if dv else ""), dv


cells = []
for i in range(0, len(rest), 3):
    nm, v, s = rest[i:i + 3]
    if not os.path.exists(v):
        print("[칸 빠짐: 영상 없음]", nm, v)
        continue
    fr = [np.asarray(f)[..., :3] for f in imageio.get_reader(v)]
    cells.append((nm, er(s)[0], fr))
n = len(cells)
nrow = (n + NCOL - 1) // NCOL
asp = [c[2][0].shape[1] / c[2][0].shape[0] for c in cells]
HP = 720
while True:                                                                         # 가로 합 (가장 넓은 줄 + 색 막대) <= 4096 이 되게 칸 높이를 줄임
    ws = [int(round(HP * a_ / 2)) * 2 for a_ in asp]
    rw = max(sum(ws[r * NCOL:(r + 1) * NCOL]) for r in range(nrow))
    if rw + CBW <= WMAX:
        break
    HP -= 8
Wt = rw + CBW; Wt += Wt % 2
Ht = TT + nrow * (HP + TOP); Ht += Ht % 2
fb = lambda z: ImageFont.truetype(FB, z)
fr_ = lambda z: ImageFont.truetype(FR, z)
base = Image.new("RGB", (Wt, Ht), "white")
dr = ImageDraw.Draw(base)
dr.text((14, 14), title, fill=(68, 68, 68), font=fr_(20))
pos = []
for k, (nm, e_, _) in enumerate(cells):
    r, c = divmod(k, NCOL)
    x = sum(ws[r * NCOL:r * NCOL + c]); y = TT + r * (HP + TOP)
    dr.text((x + 14, y + 14), nm, fill=(0, 0, 0), font=fb(26))
    if e_:
        dr.text((x + 14, y + 58), e_, fill=(0x1f, 0x4f, 0xbf), font=fb(20))
    pos.append((x, y + TOP, ws[k]))
# 색 막대 (세로, 위 = 큰 에너지)
cx, cy, ch, cw = rw + 40, TT + 120, min(nrow * (HP + TOP) - 400, 900), 46
t = np.linspace(1, 0, ch)[:, None]
bar = (np.array(CMAP(0.05 + 0.85 * t))[..., :3] * 255).astype(np.uint8).repeat(cw, 1)
base.paste(Image.fromarray(bar), (cx, cy))
dr.rectangle([cx, cy, cx + cw, cy + ch], outline=(0, 0, 0))
dr.text((cx - 10, cy - 100), "장벽 에너지\n(로그 눈금)", fill=(0, 0, 0), font=fb(24))
for k_ in range(int(np.ceil(ELO)), int(np.floor(EHI)) + 1):
    yy = cy + int(round((EHI - k_) / (EHI - ELO) * (ch - 1)))
    dr.line([cx + cw, yy, cx + cw + 10, yy], fill=(0, 0, 0), width=2)
    dr.text((cx + cw + 16, yy - 13), f"1e{k_}", fill=(0, 0, 0), font=fr_(22))
yl = cy + ch + 30
for line in LEG.split("|"):
    dr.text((rw + 14, yl), line, fill=(40, 40, 40), font=fr_(18))
    yl += 26
base = np.asarray(base)
L = max(len(c[2]) for c in cells)
W = imageio.get_writer(out, fps=30, codec="libx264", pixelformat="yuv420p", macro_block_size=1, ffmpeg_params=["-crf", "20", "-movflags", "+faststart"])
for fi in range(L):
    im = base.copy()
    for (nm, e_, frs), (x, y, w) in zip(cells, pos):
        f = frs[min(fi, len(frs) - 1)]
        if f.shape[0] != HP or f.shape[1] != w:
            f = np.asarray(Image.fromarray(f).resize((w, HP), Image.LANCZOS))
        im[y:y + HP, x:x + w] = f
    W.append_data(im)
W.close()
print("[저장]", out, n, "칸", f"{Wt}x{Ht}", L, "프레임", "칸 높이", HP)
