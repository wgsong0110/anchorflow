"""절 CM11 wolf 영상 합성 (인스턴스, ffmpeg): 정답 | 안 띄운 (B) | 띄운 (B), 형식은 exe/fem_barx_compose.py 그대로 (칸 800x800 + 위 200 글자 띠, 판 이름·오차, 위 제목; 색 범위 조정 없음).
사용: python exe/fem_lift_compose.py <정답 영상> <안 띄운 영상> <안 띄운 summary.json> <띄운 영상> <띄운 summary.json> <글꼴 Bold> <글꼴 Regular> <출력 mp4> <임시 폴더>"""
import json
import os
import subprocess
import sys

gt, v0, s0p, v1, s1p, FB, FR, out, TMP = sys.argv[1:10]
os.makedirs(TMP, exist_ok=True)


def er(p, lift=False):
    s = json.load(open(p))
    dv = s.get("diverged")
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    t = f"오차 {100 * e:.2f}%"
    if lift and s.get("lift"):
        t = f"띄움 {1000 * s['lift']['dist']:.1f} mm (h x {s['lift']['n']}) · 오차 {100 * e:.2f}% (뺀 값 {100 * s['lift']['err_lift_mean_done']:.2f}%)"
    return t + (f" · 프레임 {dv} 에서 발산" if dv else ""), dv


e0, d0 = er(s0p)
e1, d1 = er(s1p, True)
title = "wolf 50 프레임 · 채택 (B) (장벽 + --y0eb, HHT ρ 0.8, K 2, εN 1e-2 · εP 1e-1, 상한 뉴턴 2 x PCG 5, 결정적) · 처음 관통 칸 0 이 되게 띄운 판 비교"
cells = [("정답 (PG 모래)", None, gt, None), ("우리 (B) 안 띄움", e0, v0, d0), ("우리 (B) 띄움", e1, v1, d1)]


def tf(name, text):
    p = f"{TMP}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(f, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


ins, fl = [], []
for k, (nm, e_, path, dv) in enumerate(cells):
    ins += ["-i", path]
    f_ = f"[{k}:v]scale=800:800,pad=800:1000:0:200:white," + ("tpad=stop_mode=clone:stop=-1," if dv else "") + dt(tf(f"c{k}a", nm), FB, 32, 16, 70)
    if e_:
        f_ += "," + dt(tf(f"c{k}b", e_), FB, 24, 16, 124, "0x1f4fbf")
    fl.append(f_ + f"[c{k}]")
fl.append("[c0][c1][c2]hstack=inputs=3,drawbox=x=799:y=60:w=2:h=940:color=0xdddddd:t=fill,drawbox=x=1599:y=60:w=2:h=940:color=0xdddddd:t=fill,"
          + dt(tf("title", title), FR, 20, 16, 16, "0x444444") + "[o]")
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", "-shortest", out], check=True)
print("[저장]", out)
