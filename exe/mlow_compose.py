"""wolf 비교 영상 합성 (로컬 ffmpeg; exe/wolfcv_compose.py 방식): 정답 | 기준 | mlow 판 1 | 판 2.
사용: python exe/mlow_compose.py <자료 디렉토리>  (gt_wolf.mp4 Kacdhnz.mp4 c.mp4 d.mp4) -> fem_wolf_mlow_cmp.mp4 (칸 600, 가로 2400, yuv420p, faststart, 51 프레임)"""
import os, subprocess, sys
D = sys.argv[1]
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
T = f"{D}/txt"; os.makedirs(T, exist_ok=True)
cells = ["정답 (PG 모래)", "기준 (확장·S_구속·C)", "mlow 1e-20 (판 1)", "mlow 1e-20 (판 2)"]
files = ["gt_wolf.mp4", "Kacdhnz.mp4", "c.mp4", "d.mp4"]
W = 600
ins, fl = [], []
for k, (a, f) in enumerate(zip(cells, files)):
    ins += ["-i", f"{D}/{f}"]
    p = f"{T}/c{k}.txt"; open(p, "w").write(a)
    fl.append(f"[{k}:v]fps=30,tpad=stop=-1:stop_mode=clone,scale={W}:{W},setsar=1,pad={W}:{W+60}:0:60:white,"
              f"drawtext=fontfile={FB}:textfile={p}:fontsize=26:fontcolor=black:expansion=none:x=12:y=14[c{k}]")
fl.append("[c0][c1][c2][c3]hstack=inputs=4," + ",".join(f"drawbox=x={W*j-1}:y=0:w=2:h={W+60}:color=0xdddddd:t=fill" for j in range(1, 4)) + "[o]")
out = f"{D}/fem_wolf_mlow_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "30", "-frames:v", "51", out], check=True)
print("[저장]", out)
