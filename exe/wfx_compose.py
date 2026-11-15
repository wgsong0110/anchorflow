"""wolffix (11-15) 비교 영상 합성 (로컬 ffmpeg; exe/mlow_compose.py 방식): 정답 | 기준 판 실행 1 (B_d1) | 기준 판 실행 2 (B_t1) | 무너진 프레임 다시 풀기 (S2T_1).
사용: python exe/wfx_compose.py <자료 디렉토리> (gt_wolf.mp4 B_d1.mp4 B_t1.mp4 S2T_1.mp4; R2 lbours/fem_ag_wolf/gt · lbours/wolffix1115/{d1,r1}/<판>/vj/fem_wolf.mp4)
 -> fem_wolf_wolffix_cmp.mp4 (칸 600, 가로 2400, yuv420p, faststart, 51 프레임)"""
import os, subprocess, sys
D = sys.argv[1]; FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
cells = ["정답 (PG 모래)", "기준 판 (실행 1, 왼쪽 튐)", "기준 판 (실행 2, 뒤집힌 덩어리)", "무너진 프레임 다시 풀기 (K 2·4)"]
files = ["gt_wolf.mp4", "B_d1.mp4", "B_t1.mp4", "S2T_1.mp4"]
W = 600; ins = []; fl = []
for k, (a, f) in enumerate(zip(cells, files)):
    ins += ["-i", f"{D}/{f}"]; p = f"{D}/c{k}.txt"; open(p, "w").write(a)
    fl.append(f"[{k}:v]fps=30,tpad=stop=-1:stop_mode=clone,scale={W}:{W},setsar=1,pad={W}:{W+60}:0:60:white,drawtext=fontfile={FB}:textfile={p}:fontsize=24:fontcolor=black:expansion=none:x=12:y=16[c{k}]")
fl.append("[c0][c1][c2][c3]hstack=inputs=4," + ",".join(f"drawbox=x={W*j-1}:y=0:w=2:h={W+60}:color=0xdddddd:t=fill" for j in range(1, 4)) + "[o]")
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-r", "30", "-frames:v", "51", f"{D}/fem_wolf_wolffix_cmp.mp4"], check=True)
