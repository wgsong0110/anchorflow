"""절 CM25 wolf 영상 합성 (로컬 ffmpeg): 정답 | 지금 기본 판 | 무너진 판 ... -- 칸 너비 = min(600, 2400 / 칸 수) + 위 150 글자 띠 (가로 ≤ 2400), yuv420p, faststart; 색 범위 조정 없음.
사용: python exe/fem_wolfwhy_compose.py <글꼴 Bold> <글꼴 Regular> <출력 mp4> <임시 폴더> <제목> <정답 영상> [<이름> <영상> <summary>]..."""
import json
import os
import subprocess
import sys

FB, FR, out, TMP, title, gt = sys.argv[1:7]
RS = sys.argv[7:]
os.makedirs(TMP, exist_ok=True)
CW = min(600, (2400 // (1 + len(RS) // 3)) // 2 * 2)


def er(p):
    s = json.load(open(p))
    dv = s.get("diverged")
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    return f"오차 {100 * e:.2f}%" + (f" · 프레임 {dv} 에서 발산" if dv else ""), dv


cells = [("정답 (PG 모래)", None, gt, None)]
for nm, v, s in zip(RS[0::3], RS[1::3], RS[2::3]):
    e_, dv = er(s)
    cells.append((nm, e_, v, dv))


def tf(name, text):
    p = f"{TMP}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(f, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


ins, fl = [], []
for k, (nm, e_, path, dv) in enumerate(cells):
    ins += ["-r", "30", "-i", path]
    f_ = f"[{k}:v]scale={CW}:{CW},setsar=1,pad={CW}:{CW + 150}:0:150:white," + ("tpad=stop_mode=clone:stop=-1," if dv else "") + dt(tf(f"c{k}a", nm), FB, 26, 12, 56)
    if e_:
        f_ += "," + dt(tf(f"c{k}b", e_), FB, 20, 12, 98, "0x1f4fbf")
    fl.append(f_ + f"[c{k}]")
n = len(cells)
fl.append("".join(f"[c{k}]" for k in range(n)) + f"hstack=inputs={n}:shortest=1,"
          + ",".join(f"drawbox=x={CW * k - 1}:y=46:w=2:h={CW + 104}:color=0xdddddd:t=fill" for k in range(1, n)) + ","
          + dt(tf("title", title), FR, 16, 12, 14, "0x444444") + "[o]")
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
                "-movflags", "+faststart", "-shortest", out], check=True)
print("[저장]", out)
