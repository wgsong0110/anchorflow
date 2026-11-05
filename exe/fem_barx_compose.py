"""절 CM10 wolf 영상 합성 (인스턴스, ffmpeg): 정답 | 우리 (--bar_cross 판 x_wolf), 형식은 exe/fem_cm4_compose.py (fem_cm_wolf_cmp.mp4) 그대로 -- 칸 800x800 + 위 200 글자 띠, 판 이름·오차, 위 제목 (색 범위 조정 없음).
사용: python exe/fem_barx_compose.py <정답 영상> <우리 영상> <우리 summary.json> <글꼴 Bold> <글꼴 Regular> <출력 mp4> <임시 폴더>"""
import json
import os
import subprocess
import sys

gt, ours, summ, FB, FR, out, TMP = sys.argv[1:8]
os.makedirs(TMP, exist_ok=True)
s = json.load(open(summ))
dv = s.get("diverged")
e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
er = f"오차 {100 * e:.2f}%" + (f" · 프레임 {dv} 에서 발산 (그 뒤 정지 화면)" if dv else "")
title = "wolf 50 프레임 · 암시적 B-스플라인 실시간 판 채택 (B) + 관통 칸 장벽 --bar_cross (HHT ρ 0.8, K 2, εN 1e-2 · εP 1e-1, 상한 뉴턴 2 x PCG 5) · 결정적"
cells = [("정답 (PG 모래)", None, gt), ("우리 (B) + 관통 칸 장벽", er, ours)]


def tf(name, text):
    p = f"{TMP}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(f, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


ins, fl = [], []
for k, (nm, e_, path) in enumerate(cells):
    ins += ["-i", path]
    f_ = f"[{k}:v]scale=800:800,pad=800:1000:0:200:white," + (("tpad=stop_mode=clone:stop=-1," if k == 1 and dv else "")) + dt(tf(f"c{k}a", nm), FB, 32, 16, 70)
    if e_:
        f_ += "," + dt(tf(f"c{k}b", e_), FB, 32, 16, 124, "0x1f4fbf")
    fl.append(f_ + f"[c{k}]")
fl.append("[c0][c1]hstack=inputs=2,drawbox=x=799:y=60:w=2:h=940:color=0xdddddd:t=fill,"
          + dt(tf("title", title), FR, 20, 16, 16, "0x444444") + "[o]")
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", "-shortest", out], check=True)
print("[저장]", out)
