"""절 CM183 A 비교 영상 합성 (로컬 ffmpeg): 정답 | 기본 | 가우스 1 점 | ρ∞=0 | ρ∞=1 -- 칸 480, 가로 2400, yuv420p, faststart.
사용: python exe/abl_compose.py <장면> <렌더 mp4 폴더 (gt.mp4 def.mp4 q1.mp4 rho0.mp4 rho1.mp4)> <출력 mp4>   (오차는 R2 metrics.json 중앙값 /tmp/ablruns 에서)"""
import json, os, subprocess, sys, glob, statistics as S
sc, D, OUT = sys.argv[1:4]
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
T = os.path.join(os.path.dirname(os.path.abspath(OUT)), "txt_abl_" + sc); os.makedirs(T, exist_ok=True)
def err(v):
    e = []
    for f in glob.glob(f"/tmp/ablruns/{sc}_{v}_r*/metrics.json"):
        m = json.load(open(f))
        if m.get("err") is not None: e.append(m["err"])
    return "오차 %.2f%%" % (100 * S.median(e)) if e else "오차 없음"
V = [("gt", "정답", ""), ("def", "기본 (가우스 2x2x2, HHT 0.8)", err("def")), ("q1", "가우스 1x1x1", err("q1")), ("rho0", "ρ∞=0 (일반화-α)", err("rho0")), ("rho1", "ρ∞=1 (중점)", err("rho1"))]
nf = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", f"{D}/def.mp4"], capture_output=True, text=True, check=True).stdout.strip()
ins, fl = [], []
for k, (v, a, b) in enumerate(V):
    ins += ["-i", f"{D}/{v}.mp4"]
    pa, pb = f"{T}/c{k}a.txt", f"{T}/c{k}b.txt"
    open(pa, "w").write(f"{sc} · {a}"); open(pb, "w").write(b or " ")
    fl.append(f"[{k}:v]fps=30,tpad=stop=-1:stop_mode=clone,scale=480:480,setsar=1,pad=480:560:0:80:white,"
              f"drawtext=fontfile={FB}:textfile={pa}:fontsize=18:fontcolor=black:expansion=none:x=8:y=10,"
              f"drawtext=fontfile={FB}:textfile={pb}:fontsize=20:fontcolor=0x1f4fbf:expansion=none:x=8:y=42[c{k}]")
fl.append("[c0][c1][c2][c3][c4]hstack=inputs=5[o]")
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "30", "-frames:v", nf, OUT], check=True)
print("[저장]", OUT, nf)
