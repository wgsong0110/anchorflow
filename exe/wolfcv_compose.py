"""절 CM93 wolf 영상 합성 (로컬 ffmpeg; exe/wolfh10_compose.py 방식): 정답 | H8 | 수렴 판 (뉴턴 20 · PCG 40, f27 발산 -> 마지막 프레임 정지 + '프레임 27 발산').
사용: python exe/wolfcv_compose.py <자료 디렉토리>
  자료: gt_wolf.mp4 (R2 lbours/fem_ag_wolf/gt), stat.json (프레임별 det F 중앙값 · 바닥 위 질량중심 mm: 정답 wolf.pt F · an.json, 판 pred.pt F · an.json), aC_1/vj/fem_wolf.mp4 · h8_1/vj/fem_wolf.mp4 · summary.json (R2 lbours/wolfcv, lbours/wolfh12)
출력: fem_wolf_conv_cmp.mp4 (칸 800, 2400 폭, yuv420p, faststart, 51 프레임)"""
import json
import os
import subprocess
import sys

D = sys.argv[1]
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
T = f"{D}/txt"
os.makedirs(T, exist_ok=True)
S = json.load(open(f"{D}/stat.json"))
NF = 51
DIV = int(json.load(open(f"{D}/aC_1/summary.json"))["diverged"])        # 발산 프레임 (그리지 못함); 마지막으로 그린 프레임 = DIV - 1


def er(r):
    s = json.load(open(f"{D}/{r}/summary.json"))
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    return f"오차 {100 * e:.2f}%" + (f" (f{DIV - 1} 까지)" if s.get("diverged") else "")


cells = [("wolf · 정답 (PG 모래)", "", f"{D}/gt_wolf.mp4", "gt"),
         ("wolf · H8 (뉴턴 6 · PCG 5)", er("h8_1"), f"{D}/h8_1/vj/fem_wolf.mp4", "h8_1"),
         ("wolf · 수렴 판 (뉴턴 20 · PCG 40)", er("aC_1"), f"{D}/aC_1/vj/fem_wolf.mp4", "aC_1")]


def tf(name, text):
    p = f"{T}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(f, font, size, x, y, color="black", en=None):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}" + (f":enable='{en}'" if en else "")


W = 800
ins, fl = [], []
for k, (a, b, path, key) in enumerate(cells):
    ins += ["-i", path]
    f_ = f"[{k}:v]fps=30,tpad=stop=-1:stop_mode=clone,scale={W}:{W},setsar=1,pad={W}:{W + 120}:0:120:white," + dt(tf(f"c{k}a", a), FB, 32, 16, 10)
    if b:
        f_ += "," + dt(tf(f"c{k}b", b), FB, 28, 16, 56, "0x1f4fbf")
    last = len(S[key]["det"]) - 1                                         # 마지막으로 그린 프레임
    for n in range(NF):
        i = min(n, last)
        txt = f"프레임 {i} · det F 중앙값 {S[key]['det'][i]:.3f} · 질량중심 {S[key]['com'][i]:.0f} mm (바닥 위)"
        f_ += "," + dt(tf(f"c{k}t{n}", txt), FR, 24, 16, 92, "0x333333", f"eq(n,{n})")
    if last < NF - 1:
        f_ += "," + dt(tf(f"c{k}d", f"프레임 {last + 1} 발산 (이후 프레임 {last} 정지)"), FB, 36, 16, 860, "red", f"gte(n,{last + 1})")
    fl.append(f_ + f"[c{k}]")
fl.append("[c0][c1][c2]hstack=inputs=3," + ",".join(f"drawbox=x={W * j - 1}:y=0:w=2:h={W + 120}:color=0xdddddd:t=fill" for j in range(1, 3))
          + f",pad={3 * W}:{W + 170}:0:50:white," + dt(tf("title", "wolf 모래 · 정답 | H8 | 수렴 판 (예산 끔 · 뉴턴 20 · PCG 40), 각 덤프 판 1 개 (비결정)"), FR, 20, 16, 14, "0x444444") + "[o]")
out = f"{D}/fem_wolf_conv_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "30", "-frames:v", str(NF), out], check=True)
print("[저장]", out)
