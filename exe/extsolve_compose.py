"""절 CM117 wolf 영상 합성 (로컬 ffmpeg; exe/w5s0_compose.py 일반화): 정답 | 판들, 칸 W, 가로 ≤ 2400, yuv420p, faststart, 51 프레임.
사용: python exe/extsolve_compose.py <자료 디렉토리> <출력.mp4> <제목> <판:이름> [<판:이름> ...]
  자료 디렉토리: gt_wolf.mp4, extsolve_an.json, <판>/vj/fem_wolf.mp4"""
import json, os, subprocess, sys

D, OUT, TITLE = sys.argv[1], sys.argv[2], sys.argv[3]
RUNS = [a.split(":", 1) for a in sys.argv[4:]]
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf"); FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
T = f"{D}/txt"; os.makedirs(T, exist_ok=True)
A = json.load(open(f"{D}/extsolve_an.json"))
NF = 51
NC = len(RUNS) + 1
W = min(800, (2400 // NC) // 2 * 2)


def tf(name, text):
    p = f"{T}/{name}.txt"; open(p, "w").write(text); return p


def dt(f, font, size, x, y, color="black", en=None):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}" + (f":enable='{en}'" if en else "")


fs = max(16, W // 25)
cells = [("wolf · 정답 (PG 모래)", "", f"{D}/gt_wolf.mp4", None)]
for key, name in RUNS:
    a = A[key]
    e_ = f"오차 {100 * a['err']:.2f}%" if a.get("err") is not None else f"발산 f{a.get('diverged')}"
    cells.append((f"wolf · {name}", f"{e_} · 프레임 {a['ms_mean']:.0f} ms", f"{D}/{key}/vj/fem_wolf.mp4", key))
ref = RUNS[0][0]
ins, fl = [], []
for k, (a, b, path, key) in enumerate(cells):
    ins += ["-i", path]
    f_ = f"[{k}:v]fps=30,tpad=stop=-1:stop_mode=clone,scale={W}:{W},setsar=1,pad={W}:{W + 120}:0:120:white," + dt(tf(f"c{k}a", a), FB, fs + 4, 12, 8)
    if b:
        f_ += "," + dt(tf(f"c{k}b", b), FB, fs, 12, 48, "0x1f4fbf")
    for n in range(NF):
        if key is None:
            h = A[ref]["h"].get(str(n - 1)) if n > 0 else A[ref]["h"].get("0")
            txt = f"프레임 {n} · 질량중심 {h[1]:.0f} mm" if h else f"프레임 {n}"
        else:
            hk = A[key]["h"].get(str(max(n - 1, 0)))
            rr = A[key]["r07"].get(str(n), 0) if n > 0 else 0
            txt = f"프레임 {n} · 질량중심 {hk[0]:.0f} mm · r<0.7 칸 {rr}" if hk else f"프레임 {n}"
        f_ += "," + dt(tf(f"c{k}t{n}", txt), FR, fs - 2, 12, 86, "0x333333", f"eq(n,{n})")
    fl.append(f_ + f"[c{k}]")
fl.append("".join(f"[c{k}]" for k in range(NC)) + f"hstack=inputs={NC}," + ",".join(f"drawbox=x={W * j - 1}:y=0:w=2:h={W + 120}:color=0xdddddd:t=fill" for j in range(1, NC))
          + f",pad={NC * W}:{W + 170}:0:50:white," + dt(tf("title", TITLE), FR, 20, 16, 14, "0x444444") + "[o]")
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "30", "-frames:v", str(NF), OUT], check=True)
print("[저장]", OUT)
