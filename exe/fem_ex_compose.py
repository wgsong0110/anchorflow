"""절 L 명시적 판 영상 합성 (로컬 ffmpeg drawtext + Noto Sans CJK KR; % 가 지워지지 않게 textfile + expansion=none).
칸마다 위 띠: 판 이름 / "프레임당 XX.X ms · XX.X fps · 오차 XX.XX%" (정답 칸은 이름만), 렌더의 프레임 번호는 그대로.
사용: python exe/fem_ex_compose.py <출력.mp4> "<제목 줄>" "이름|summary.json 또는 -|영상" ...   (칸 800x800 + 띠 160)"""
import json
import os
import subprocess
import sys

TMP = "/tmp/fem_ex_compose"
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
os.makedirs(TMP, exist_ok=True)
out, title, cells = sys.argv[1], sys.argv[2], [c.split("|") for c in sys.argv[3:]]


def tf(name, text):
    p = f"{TMP}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(textfile, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={textfile}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


ins, fl = [], []
n = len(cells)
for k, (nm, sj, path) in enumerate(cells):
    ins += ["-i", path]
    a = tf(f"c{k}a", nm)
    sz = 30 if len(nm) <= 24 else (26 if len(nm) <= 30 else 22)
    f_ = f"[{k}:v]scale=800:800,pad=800:1000:0:200:white," + dt(a, FB, sz, 16, 70)
    if sj != "-":
        s = json.load(open(sj))
        err = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
        b = tf(f"c{k}b", f"프레임당 {s['ms_mean']:.1f} ms · {s['fps']:.1f} fps · 오차 {100 * err:.2f}%")
        f_ += "," + dt(b, FB, 32, 16, 128, "0x1f4fbf")
    fl.append(f_ + f"[c{k}]")
t0 = tf("title", title)
lines = "".join(f"drawbox=x={800 * k - 1}:y=60:w=2:h=940:color=0xdddddd:t=fill," for k in range(1, n))
fl.append("".join(f"[c{k}]" for k in range(n)) + f"hstack=inputs={n}," + lines + dt(t0, FR, 26, 16, 16, "0x444444") + "[o]")
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-r", "30", out], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "2", "-i", out, "-frames:v", "1", f"{TMP}/chk_{os.path.basename(out)}.png"], check=True)
print(f"[저장] {out}  (확인 그림 {TMP}/chk_{os.path.basename(out)}.png)")
