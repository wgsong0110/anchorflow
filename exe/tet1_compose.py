"""절 CM102 영상 합성 (exe/gpl_compose.py 의 build 그대로, 4 칸 600 폭 = 2400 폭): 정답 | H9 (육면체) | 사면체 T9 | 사면체 T9S.
사용: python exe/tet1_compose.py <출력.mp4> <제목> <칸 파일:이름표:요약 json 또는 -> x4
  요약 json 이 있으면 오차·발산을 둘째 줄로. 정답 프레임 수만큼 (짧은 칸은 끝 프레임 정지), yuv420p, faststart. 마지막 프레임 PNG 도 (<출력>_last.png)."""
import json
import os
import subprocess
import sys

OUT, TITLE = sys.argv[1], sys.argv[2]
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
cells = []
for a in sys.argv[3:]:
    p, name, sj = a.split(":", 2)
    b = ""
    if sj != "-" and os.path.exists(sj):
        s = json.load(open(sj))
        e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
        b = f"오차 {100 * e:.2f}%" + (f" (f{s['diverged']} 발산)" if s.get("diverged") else "")
    cells.append((p, name, b))
fps, nf = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames", "-show_entries", "stream=r_frame_rate,nb_read_frames", "-of", "csv=p=0", cells[0][0]],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
T = OUT + "_txt"
os.makedirs(T, exist_ok=True)


def tf(name, text):
    p = f"{T}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(f, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


def build(cells, W, title, last=False):
    ins, fl = [], []
    for k, (path, a, b) in enumerate(cells):
        ins += (["-sseof", "-0.05"] if last else []) + ["-i", path]
        f_ = f"[{k}:v]" + ("" if last else f"fps={fps},tpad=stop=-1:stop_mode=clone,") + f"scale={W}:{W},setsar=1,pad={W}:{W + 100}:0:100:white," + dt(tf(f"c{k}a{W}{int(last)}", a), FB, int(W * 0.042), 12, 10)
        if b:
            f_ += "," + dt(tf(f"c{k}b{W}{int(last)}", b), FB, int(W * 0.038), 12, 55, "0x1f4fbf" if k else "0x444444")
        fl.append(f_ + f"[c{k}]")
    n = len(cells)
    fl.append("".join(f"[c{j}]" for j in range(n)) + f"hstack=inputs={n}," + ",".join(f"drawbox=x={W * j - 1}:y=0:w=2:h={W + 100}:color=0xdddddd:t=fill" for j in range(1, n))
              + f",pad={W * n}:{W + 150}:0:50:white," + dt(tf(f"title{W}{n}{int(last)}", title), FR, 20, 16, 14, "0x444444") + "[o]")
    return ins, ";".join(fl)


W = 2400 // len(cells)
ins, fl = build(cells, W, TITLE)
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", fl, "-map", "[o]", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", fps, "-frames:v", nf, OUT], check=True)
ins, fl = build(cells, W, TITLE + " · 마지막 프레임", last=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", fl, "-map", "[o]", "-frames:v", "1", OUT[:-4] + "_last.png"], check=True)
print("[저장]", OUT, fps, nf)
