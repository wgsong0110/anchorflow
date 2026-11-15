"""절 CM190 대표 비교 영상 합성 (로컬 ffmpeg): 레고 정답 | 기본 (가우스 2x2x2) | 가우스 1x1x1 -- 칸 640, 가로 1920, yuv420p, faststart.
사용: python exe/fpa_compose.py <렌더 mp4 폴더 (gt.mp4 def.mp4 q1.mp4)> <metrics 캐시 runs 폴더> <출력 mp4>"""
import glob, json, os, statistics as S, subprocess, sys
D, RUNS, OUT = sys.argv[1:4]
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
T = os.path.join(os.path.dirname(os.path.abspath(OUT)), "txt_fpa"); os.makedirs(T, exist_ok=True)
def lab(v):
    m = [json.load(open(f)) for f in glob.glob(f"{RUNS}/L000_{v}_r*/metrics.json")]
    m = [q for q in m if q.get("err") is not None]
    return "오차 %.2f%%  %.1f ms" % (100 * S.median(q["err"] for q in m), S.median(q["ms"] for q in m)) if m else ""
V = [("gt", "정답 (PhysGaussian MPM)", ""), ("def", "우리 기본 (가우스점 2x2x2)", lab("def")), ("q1", "가우스점 1x1x1", lab("q1"))]
nf = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", f"{D}/def.mp4"], capture_output=True, text=True, check=True).stdout.strip()
ins, fl = [], []
for k, (v, a, b) in enumerate(V):
    ins += ["-i", f"{D}/{v}.mp4"]
    pa, pb = f"{T}/c{k}a.txt", f"{T}/c{k}b.txt"
    open(pa, "w").write(f"레고 · {a}"); open(pb, "w").write(b or " ")
    fl.append(f"[{k}:v]fps=30,tpad=stop=-1:stop_mode=clone,scale=640:640,setsar=1,pad=640:730:0:90:white,"
              f"drawtext=fontfile={FB}:textfile={pa}:fontsize=22:fontcolor=black:expansion=none:x=10:y=12,"
              f"drawtext=fontfile={FB}:textfile={pb}:fontsize=22:fontcolor=0x1f4fbf:expansion=none:x=10:y=50[c{k}]")
fl.append("[c0][c1][c2]hstack=inputs=3[o]")
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "30", "-frames:v", nf, OUT], check=True)
print("[저장]", OUT, nf)
