"""절 CM89 wolf 영상·그림 합성 (exe/wolfh9_compose.py 복사; 영상 = 정답 | 최선, 통과할 때만) (로컬 ffmpeg; exe/ext2v_compose.py 방식).
사용: python exe/wolfh10_compose.py <자료 디렉토리> <최선 판 이름 (vA|vB)> <최선 이름표> [--no_video]
  자료: gt_gt_wolf.mp4 (R2 lbours/fem_ag_wolf/gt), vW5·vW (R2 lbours/ext2/vid: 이전 H5 · H7 = H8 같은 계산), v<최선> (R2 lbours/wolfh10)
출력: fem_H10_wolf_cmp.mp4 = 정답 | 최선 (2400 폭, yuv420p, faststart; 정답 프레임 수만큼, 짧은 칸은 끝 프레임 정지)
      wolfh10_last.png = 마지막 프레임 정답 | H5 | H8 | 최선 (칸 600 폭)"""
import json
import os
import subprocess
import sys

D, BEST, BL = sys.argv[1], sys.argv[2], sys.argv[3]
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")


def er(r):
    s = json.load(open(f"{D}/{r}/summary.json"))
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    return f"오차 {100 * e:.2f}%" + (f" (f{s['diverged']} 발산)" if s.get("diverged") else "")


GT = (f"{D}/gt_gt_wolf.mp4", "wolf · 정답 (PG 모래)", "")
H5 = (f"{D}/vW5/vj/fem_wolf.mp4", "wolf · 이전 H5", er("vW5"))
H8 = (f"{D}/vW/vj/fem_wolf.mp4", "wolf · H8 (= H9 wolf 계산)", er("vW"))
BS = (f"{D}/{BEST}/vj/fem_wolf.mp4", f"wolf · {BL}", er(BEST))
fps, nf = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames", "-show_entries", "stream=r_frame_rate,nb_read_frames", "-of", "csv=p=0", GT[0]],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
T = f"{D}/txt_w10"
os.makedirs(T, exist_ok=True)


def tf(name, text):
    p = f"{T}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(f, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


def build(cells, W, title, extra_in=(), last=False):
    ins, fl = [], []
    for k, (path, a, b) in enumerate(cells):
        ins += (["-sseof", "-0.034"] if last else []) + ["-i", path]
        f_ = f"[{k}:v]" + ("" if last else f"fps={fps},tpad=stop=-1:stop_mode=clone,") + f"scale={W}:{W},setsar=1,pad={W}:{W + 120}:0:120:white," + dt(tf(f"c{k}a{W}", a), FB, int(W * 0.04), 16, 10)
        if b:
            f_ += "," + dt(tf(f"c{k}b{W}", b), FB, int(W * 0.0375), 16, 60, "0x1f4fbf" if k else "0x444444")
        fl.append(f_ + f"[c{k}]")
    n = len(cells)
    fl.append("".join(f"[c{j}]" for j in range(n)) + f"hstack=inputs={n}," + ",".join(f"drawbox=x={W * j - 1}:y=0:w=2:h={W + 120}:color=0xdddddd:t=fill" for j in range(1, n))
              + f",pad={W * n}:{W + 170}:0:50:white," + dt(tf(f"title{W}{n}", title), FR, 20, 16, 14, "0x444444") + "[o]")
    return ins, ";".join(fl)


title = f"wolf · 정답 | {BL} · 덤프 판 1 개 (비결정)"
if "--no_video" not in sys.argv:
  ins, fl = build([GT, BS], 1200, title)
  out = f"{D}/fem_H10_wolf_cmp.mp4"
  subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", fl, "-map", "[o]", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", fps, "-frames:v", nf, out], check=True)
  print("[저장]", out)
ins, fl = build([GT, H5, H8, BS], 600, f"wolf 마지막 프레임 (f50) · 정답 | 이전 H5 | H8 | {BL}", last=True)
png = f"{D}/wolfh10_last.png"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", fl, "-map", "[o]", "-frames:v", "1", "-update", "1", png], check=True)
print("[저장]", png)
