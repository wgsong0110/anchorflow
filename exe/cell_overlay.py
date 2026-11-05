"""절 CM14.3: 칸 상자 반투명 겹치기 (numpy + PIL 만; exe/lb_render.py --cells 가 씀, 로컬 시험 가능).
draw_cells(fr, px, dep, cols, alpha): fr [H,W,3] uint8, px [n,8,2] 픽셀 (꼭짓점 cn = 4i + 2j + k), dep [n,8] 시점 깊이 (클수록 멂), cols [n,3] 0~255.
먼 칸부터, 칸 안에서는 먼 면부터 면마다 알파 합성 (화가 순서) + 테두리 (색 x 0.7). 가우시안과의 깊이 가림은 없음 (렌더 위에 겹침)."""
import numpy as np
from PIL import Image, ImageDraw

FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


def draw_cells(fr, px, dep, cols, alpha):
    out = fr.astype(np.float32)
    H, W = fr.shape[:2]
    for ci in np.argsort(-dep.mean(1)):
        P8, d8 = px[ci], dep[ci]
        if not np.isfinite(P8).all():
            continue
        c = np.asarray(cols[ci], np.float32)
        for fc in sorted(FACES, key=lambda q: -d8[list(q)].mean()):
            poly = P8[list(fc)]
            x0, y0 = np.floor(poly.min(0)).astype(int); x1, y1 = np.ceil(poly.max(0)).astype(int)
            x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W - 1), min(y1, H - 1)
            if x1 < x0 or y1 < y0:
                continue
            m = Image.new("L", (x1 - x0 + 1, y1 - y0 + 1), 0)
            dr = ImageDraw.Draw(m)
            pts = [(float(p[0] - x0), float(p[1] - y0)) for p in poly]
            dr.polygon(pts, fill=1)
            mk = np.asarray(m).astype(bool)
            sub = out[y0:y1 + 1, x0:x1 + 1]
            sub[mk] = sub[mk] * (1 - alpha) + c * alpha
            e = Image.new("L", m.size, 0)
            ImageDraw.Draw(e).line(pts + [pts[0]], fill=1, width=1)
            ek = np.asarray(e).astype(bool)
            sub[ek] = c * 0.7
    return np.clip(out, 0, 255).astype(np.uint8)
