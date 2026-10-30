"""사용자 10-30 발산 셀 입자 영상 합치기: 위 줄 = 정답 | 우리 롤아웃 (전체, 표시 입자 자홍), 아래 줄 = 같은 두 칸의 투시 렌더 (나머지 불투명도를 낮춘 것) 를
표시 입자 화면 위치 중심으로 잘라 확대. 프레임 번호·한국어 라벨. 색은 렌더 그대로 (칸별 색 범위 조정 없음).

  python exe/fem_dc_compose.py --gt A.mp4 --ours B.mp4 --gt_fade C.mp4 --ours_fade D.mp4 --gt_px C.json --ours_px D.json --title "..." --out X.mp4
"""
import argparse
import json

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
for k in ("gt", "ours", "gt_fade", "ours_fade", "gt_px", "ours_px", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--title", required=True)
ap.add_argument("--sub", default="", help="제목 아래 둘째 줄")
ap.add_argument("--label_gt", default="정답 (같은 입자 자홍)")
ap.add_argument("--label_ours", default="우리 롤아웃 (발산 전까지)")
ap.add_argument("--fps", type=float, default=4)
ap.add_argument("--hold", type=float, default=2.0, help="마지막 프레임을 이 초만큼 더")
ap.add_argument("--panel", type=int, default=600)
ap.add_argument("--zmin", type=float, default=0.18, help="확대 칸 최소 자르기 폭 (화면 폭 비)")
a = ap.parse_args()


def font(sz):
    from matplotlib import font_manager as fm
    c = [f for f in fm.findSystemFonts() if "Noto" in f and ("CJK" in f or "KR" in f)] or [fm.findfont("DejaVu Sans")]
    return ImageFont.truetype(sorted(c)[0], sz)


def frames(p):
    return [f for f in iio.get_reader(p)]


G, O, GF, OF = frames(a.gt), frames(a.ours), frames(a.gt_fade), frames(a.ours_fade)
n = min(len(G), len(O), len(GF), len(OF))
PG, PO = json.load(open(a.gt_px)), json.load(open(a.ours_px))
H0, W0 = G[0].shape[:2]


def centers(P):
    out, last = [], None
    for q in P["frames"][:n]:
        if q.get("med") is not None:
            last = q
        out.append(last)
    return out


CG, CO = centers(PG), centers(PO)
ext = [max(q["hi"][0] - q["lo"][0], q["hi"][1] - q["lo"][1]) for q in CG + CO if q is not None]
Z = int(min(W0, max(a.zmin * W0, 1.5 * (max(ext) if ext else 0))))          # 두 칸 같은 자르기 폭
print(f"[확대] 자르기 {Z}px / {W0} (x{W0 / Z:.2f})", flush=True)


def crop(img, c):
    cx, cy = (W0 / 2, H0 / 2) if c is None else c["med"]
    x0 = int(np.clip(cx - Z / 2, 0, W0 - Z)); y0 = int(np.clip(cy - Z / 2, 0, H0 - Z))
    return img[y0:y0 + Z, x0:x0 + Z]


P = a.panel
F1, F2, F3 = font(int(P * 0.08)), font(int(P * 0.04)), font(int(P * 0.033))
TB = int(P * 0.16)
wr = iio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=1)
canvas = None
for i in range(n):
    cv = Image.new("RGB", (2 * P, 2 * P + TB), (255, 255, 255))
    tiles = [(G[i], 0, 0, a.label_gt), (O[i], P, 0, a.label_ours),
             (crop(GF[i], CG[i]), 0, P, a.label_gt + f" -- 확대 x{W0 / Z:.1f}, 나머지 반투명"),
             (crop(OF[i], CO[i]), P, P, a.label_ours + f" -- 확대 x{W0 / Z:.1f}, 나머지 반투명")]
    dr = ImageDraw.Draw(cv)
    for im, x, y, lb in tiles:
        cv.paste(Image.fromarray(np.ascontiguousarray(im)).resize((P, P), Image.LANCZOS), (x, y + TB))
        dr.text((x + int(P * 0.02), y + TB + int(P * 0.01)), lb, fill=(0, 0, 0), font=F3)
        dr.text((x + int(P * 0.02), y + TB + int(P * 0.05)), f"{i}", fill=(230, 0, 0), font=F1)
    dr.line([(P, TB), (P, 2 * P + TB)], fill=(120, 120, 120), width=2)
    dr.line([(0, P + TB), (2 * P, P + TB)], fill=(120, 120, 120), width=2)
    dr.text((int(P * 0.02), int(P * 0.015)), a.title, fill=(200, 0, 0), font=F2)
    if a.sub:
        dr.text((int(P * 0.02), int(P * 0.08)), a.sub, fill=(0, 0, 0), font=F3)
    canvas = np.asarray(cv)
    wr.append_data(canvas)
for _ in range(int(a.hold * a.fps)):
    wr.append_data(canvas)
wr.close()
print(f"[합침] {a.out} ({n} 프레임 + 정지 {a.hold} 초)", flush=True)
