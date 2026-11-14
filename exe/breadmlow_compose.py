"""bread 비교 영상 합성 (로컬 ffmpeg; exe/mlow_compose.py 방식): 정답 | A (공통 기본 H9) | B (확장·S_구속·C 끔 + mlow 1e-20).
사용: python exe/breadmlow_compose.py <자료 디렉토리>  (gt_bread.mp4 BA_1.mp4 BB_1.mp4) -> fem_bread_mlow_cmp.mp4 (칸 800, 가로 2400, yuv420p, faststart)"""
import os, subprocess, sys
D = sys.argv[1]
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
T = f"{D}/txt"; os.makedirs(T, exist_ok=True)
cells = ["정답 (PG 빵)", "A: 공통 기본 H9", "B: 확장·S_구속·C 끔 + mlow 1e-20"]
files = ["gt_bread.mp4", "BA_1.mp4", "BB_1.mp4"]
W = 800
ins, fl = [], []
for k, (a, f) in enumerate(zip(cells, files)):
    ins += ["-i", f"{D}/{f}"]
    p = f"{T}/c{k}.txt"; open(p, "w").write(a)
    fl.append(f"[{k}:v]fps=30,tpad=stop=-1:stop_mode=clone,scale={W}:{W},setsar=1,pad={W}:{W+70}:0:70:white,"
              f"drawtext=fontfile={FB}:textfile={p}:fontsize=32:fontcolor=black:expansion=none:x=14:y=16[c{k}]")
fl.append("[c0][c1][c2]hstack=inputs=3," + ",".join(f"drawbox=x={W*j-1}:y=0:w=2:h={W+70}:color=0xdddddd:t=fill" for j in range(1, 3)) + "[o]")
out = f"{D}/fem_bread_mlow_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "30", "-frames:v", "301", out], check=True)
print("[저장]", out)
