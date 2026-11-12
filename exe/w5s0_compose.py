"""절 CM116 wolf 영상 합성 (로컬 ffmpeg; exe/wolfcv_compose.py 방식): 정답 | H9 | H9 --w5s 0, 칸 800, 2400 폭, yuv420p, faststart, 51 프레임.
사용: python exe/w5s0_compose.py <자료 디렉토리>   (gt_wolf.mp4, w5s0_an.json, w9_1/vj/fem_wolf.mp4, w0_1/vj/fem_wolf.mp4)"""
import json, os, subprocess, sys

D = sys.argv[1]
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf"); FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
T = f"{D}/txt"; os.makedirs(T, exist_ok=True)
A = json.load(open(f"{D}/w5s0_an.json"))
NF = 51


def tf(name, text):
    p = f"{T}/{name}.txt"; open(p, "w").write(text); return p


def dt(f, font, size, x, y, color="black", en=None):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}" + (f":enable='{en}'" if en else "")


def er(k):
    return f"오차 {100 * A[k]['err']:.2f}% · 프레임 {A[k]['ms_mean']:.1f} ms"


cells = [("wolf · 정답 (PG 모래)", "", f"{D}/gt_wolf.mp4", None),
         ("wolf · H9 (기본)", er("w9_1"), f"{D}/w9_1/vj/fem_wolf.mp4", "w9_1"),
         ("wolf · H9 --w5s 0 (S_구속 끔)", er("w0_1"), f"{D}/w0_1/vj/fem_wolf.mp4", "w0_1")]
W = 800; ins, fl = [], []
for k, (a, b, path, key) in enumerate(cells):
    ins += ["-i", path]
    f_ = f"[{k}:v]fps=30,tpad=stop=-1:stop_mode=clone,scale={W}:{W},setsar=1,pad={W}:{W + 120}:0:120:white," + dt(tf(f"c{k}a", a), FB, 32, 16, 10)
    if b:
        f_ += "," + dt(tf(f"c{k}b", b), FB, 28, 16, 56, "0x1f4fbf")
    for n in range(NF):
        h = A["w9_1"]["h"].get(str(n - 1)) if n > 0 else A["w9_1"]["h"].get("0")
        if key is None:
            txt = f"프레임 {n} · 질량중심 {h[1]:.0f} mm (바닥 위)" if h else f"프레임 {n}"
        else:
            hk = A[key]["h"].get(str(max(n - 1, 0)))
            rr = A[key]["r07"].get(str(n), 0) if n > 0 else 0
            txt = f"프레임 {n} · 질량중심 {hk[0]:.0f} mm · r<0.7 칸 {rr}" if hk else f"프레임 {n}"
        f_ += "," + dt(tf(f"c{k}t{n}", txt), FR, 24, 16, 92, "0x333333", f"eq(n,{n})")
    fl.append(f_ + f"[c{k}]")
fl.append("[c0][c1][c2]hstack=inputs=3," + ",".join(f"drawbox=x={W * j - 1}:y=0:w=2:h={W + 120}:color=0xdddddd:t=fill" for j in range(1, 3))
          + f",pad={3 * W}:{W + 170}:0:50:white," + dt(tf("title", "wolf 모래 · 정답 | H9 | H9 에서 --w5s 0 만 바꿈 (비교용 진단, 기본 판 안 바꿈), 각 1 판 (비결정)"), FR, 20, 16, 14, "0x444444") + "[o]")
out = f"{D}/fem_wolf_w5s0_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "30", "-frames:v", str(NF), out], check=True)
print("[저장]", out)
