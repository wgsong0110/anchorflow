"""사용자 11-09 영상 합성 (로컬 ffmpeg; exe/fem_cm4_compose.py 방식): 정답 | H7 (절 CM84 공통 후보, 묶음 ext2 단계 v 덤프 판), 칸마다 씬 이름·오차.
수박은 위 줄 일반 3DGS, 아래 줄 손상 색 (logJp viridis −0.2~0.4; 정답 R2 lbours/fem_cm/wmJd/gt_new_plain.mp4·gt_lj.mp4, 절 CM80 영상과 같은 칸).
사용: python exe/ext2v_compose.py <씬 lego|ficus|wolf|bread|wm> <자료 디렉토리 (R2 lbours/ext2/vid/v<X> 를 받은 곳 + 정답 gt_*.mp4)>
출력 <자료>/fem_H7_<씬>_cmp.mp4 (2400 폭 이하, yuv420p, faststart; 정답 프레임 수만큼, 짧은 칸은 마지막 프레임 정지)"""
import json
import os
import re
import subprocess
import sys

sc, D = sys.argv[1], sys.argv[2]
W3 = sc == "wolf3"                                     # 조정자 11-09: wolf 정답 | 이전 H5 (단계 v5 vW5) | H7 (= H8: 새 실행기 씬은 같은 계산) -> fem_H8_wolf_cmp.mp4
sc = "wolf" if W3 else sc
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
RUN = {"lego": "vL", "ficus": "vF", "wolf": "vW", "bread": "vB", "wm": "vM"}[sc]
NAME = {"lego": "레고", "ficus": "ficus", "wolf": "wolf", "bread": "bread", "wm": "수박"}[sc]
GTN = {"lego": "정답 (MPM)", "ficus": "정답 (PG)", "wolf": "정답 (PG 모래)", "bread": "정답 (PG plasticine)", "wm": "정답 (GF)"}[sc]
R = f"{D}/{RUN}"
s = json.load(open(f"{R}/summary.json"))
e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
er = f"오차 {100 * e:.2f}%"
if s.get("diverged"):
    er += f" (f{s['diverged']} 발산)"
if sc == "ficus" and os.path.exists(f"{R}/period.txt"):
    m = re.findall(r"정답 대비 ([0-9.]+)\)", open(f"{R}/period.txt").read())
    if m:
        er += f" · 주기 비 {m[-1]}"
if sc == "wm":
    rows = [("일반 렌더", f"{D}/gt_gt_new_plain.mp4", f"{R}/vj/plain.mp4"), ("손상 색 logJp", f"{D}/gt_gt_lj.mp4", f"{R}/vj/lj.mp4")]
else:
    sh = {"lego": "lego"}.get(sc, sc)
    rows = [("", f"{D}/gt_gt_{sh}.mp4", f"{R}/vj/fem_{sh}.mp4")]
H5C = None
if W3:
    s5 = json.load(open(f"{D}/vW5/summary.json"))
    e5 = s5.get("err_mean") if s5.get("err_mean") is not None else s5.get("err_mean_done")
    H5C = (f"{D}/vW5/vj/fem_wolf.mp4", f"{NAME} · 이전 H5", f"오차 {100 * e5:.2f}%" + (f" (f{s5['diverged']} 발산)" if s5.get("diverged") else ""))
fps, nf = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames", "-show_entries", "stream=r_frame_rate,nb_read_frames", "-of", "csv=p=0", rows[0][1]],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
title = f"{NAME} · 정답 | H7 (절 CM84 공통 후보: H5 + S_구속 + C + κ 0 노드만 확장 T1E2 + P1X 강성 × κ) · 덤프 판 1 개 (비결정)"
if W3:
    title = "wolf · 정답 | 이전 공통 H5 (절 CM79) | H7 = H8 (절 CM85 의 고침은 수박 실행기만 바꿈 -> wolf 는 같은 계산) · 덤프 판 각 1 개 (비결정)"
T = f"{D}/txt_{sc}"
os.makedirs(T, exist_ok=True)


def tf(name, text):
    p = f"{T}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(f, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


ins, fl, k = [], [], 0
for ri, (rn, gt, ours) in enumerate(rows):
    cells = [(gt, f"{NAME} · {GTN}", rn)] + ([H5C] if H5C else []) + [(ours, f"{NAME} · H7" + (" (= H8, 같은 계산)" if W3 else ""), (er + (f" · {rn}" if rn else "")))]
    for ci, (path, a, b) in enumerate(cells):
        ins += ["-i", path]
        f_ = f"[{k}:v]fps={fps},tpad=stop=-1:stop_mode=clone,scale=800:800,setsar=1,pad=800:920:0:120:white," + dt(tf(f"c{k}a", a), FB, 32, 16, 10)
        if b:
            f_ += "," + dt(tf(f"c{k}b", b), FB, 30, 16, 60, "0x1f4fbf" if ci else "0x444444")
        fl.append(f_ + f"[c{k}]")
        k += 1
    nc = len(cells)
    fl.append("".join(f"[c{k - nc + j}]" for j in range(nc)) + f"hstack=inputs={nc}," + ",".join(f"drawbox=x={800 * j - 1}:y=0:w=2:h=920:color=0xdddddd:t=fill" for j in range(1, nc)) + f"[r{ri}]")
if len(rows) == 1:
    fl.append(f"[r0]pad={800 * nc}:970:0:50:white," + dt(tf("title", title), FR, 20, 16, 14, "0x444444") + "[o]")
else:
    fl.append("[r0][r1]vstack=inputs=2,pad=1600:1890:0:50:white," + dt(tf("title", title), FR, 20, 16, 14, "0x444444") + "[o]")
out = f"{D}/fem_H8_wolf_cmp.mp4" if W3 else f"{D}/fem_H7_{sc}_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", fps, "-frames:v", nf, out], check=True)
print("[저장]", out, er)
