"""절 CM101 영상 합성 (로컬 ffmpeg; exe/ext2v_compose.py 방식): 정답 | H5 2x2x2 | 최선 3x3x3 판 -- 칸 800, 3 칸 = 2400 폭.
수박은 위 줄 일반 3DGS, 아래 줄 손상 색 (logJp viridis −0.2~0.4; 정답 R2 lbours/fem_cm/wmJd/gt_new_plain.mp4·gt_lj.mp4).
사용: python exe/q27_compose.py <wolf|wm> <자료 디렉토리 (정답 gt_*.mp4 + 판 디렉토리 둘)> <H5 판> <3x3x3 판> <3x3x3 이름표>
출력 <자료>/fem_q27_<씬>_cmp.mp4 (yuv420p, faststart; 정답 프레임 수만큼, 짧은 칸은 마지막 프레임 정지) + q27_<씬>_last.png"""
import json
import os
import subprocess
import sys

sc, D, RC, RB, LB = sys.argv[1:6]
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
NAME = {"wolf": "wolf", "wm": "수박"}[sc]
GTN = {"wolf": "정답 (PG 모래)", "wm": "정답 (GF)"}[sc]


def er(r):
    s = json.load(open(f"{D}/{r}/summary.json"))
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    return f"오차 {100 * e:.2f}%" + (f" (f{s['diverged']} 발산)" if s.get("diverged") else "")


if sc == "wm":
    rows = [("일반 렌더", f"{D}/gt_new_plain.mp4", "vj/plain.mp4"), ("손상 색 logJp", f"{D}/gt_lj.mp4", "vj/lj.mp4")]
else:
    rows = [("", f"{D}/gt_wolf.mp4", "vj/fem_wolf.mp4")]
fps, nf = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames", "-show_entries", "stream=r_frame_rate,nb_read_frames", "-of", "csv=p=0", rows[0][1]],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
title = f"{NAME} · 정답 | H5 (칸마다 가우스 2x2x2) | {LB} (칸마다 가우스 3x3x3 = 27 점 완전 적분) · 덤프 판 각 1 개 (비결정)"
T = f"{D}/txt_q27_{sc}"
os.makedirs(T, exist_ok=True)


def tf(name, text):
    p = f"{T}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(f, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


def build(last):
    ins, fl, k = [], [], 0
    for ri, (rn, gt, sub) in enumerate(rows):
        cells = [(gt, f"{NAME} · {GTN}", rn), (f"{D}/{RC}/{sub}", f"{NAME} · H5 2x2x2", er(RC) + (f" · {rn}" if rn else "")),
                 (f"{D}/{RB}/{sub}", f"{NAME} · {LB}", er(RB) + (f" · {rn}" if rn else ""))]
        for ci, (path, a, b) in enumerate(cells):
            ins += (["-sseof", "-0.034"] if last else []) + ["-i", path]
            f_ = f"[{k}:v]" + ("" if last else f"fps={fps},tpad=stop=-1:stop_mode=clone,") + "scale=800:800,setsar=1,pad=800:920:0:120:white," + dt(tf(f"c{k}a", a), FB, 32, 16, 10)
            if b:
                f_ += "," + dt(tf(f"c{k}b", b), FB, 30, 16, 60, "0x1f4fbf" if ci else "0x444444")
            fl.append(f_ + f"[c{k}]")
            k += 1
        fl.append("".join(f"[c{k - 3 + j}]" for j in range(3)) + "hstack=inputs=3," + ",".join(f"drawbox=x={800 * j - 1}:y=0:w=2:h=920:color=0xdddddd:t=fill" for j in (1, 2)) + f"[r{ri}]")
    if len(rows) == 1:
        fl.append("[r0]pad=2400:970:0:50:white," + dt(tf("title", title), FR, 20, 16, 14, "0x444444") + "[o]")
    else:
        fl.append("[r0][r1]vstack=inputs=2,pad=2400:1890:0:50:white," + dt(tf("title", title), FR, 20, 16, 14, "0x444444") + "[o]")
    return ins, ";".join(fl)


ins, fl = build(False)
out = f"{D}/fem_q27_{sc}_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", fl, "-map", "[o]", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", fps, "-frames:v", nf, out], check=True)
print("[저장]", out)
ins, fl = build(True)
png = f"{D}/q27_{sc}_last.png"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", fl, "-map", "[o]", "-frames:v", "1", "-update", "1", png], check=True)
print("[저장]", png)
