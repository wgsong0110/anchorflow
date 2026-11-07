"""절 CM22 wolf 영상 합성 (로컬 ffmpeg): 정답 | (A) 가우스 점 고착 | (B) 장벽 | (B)+보간 교체 -- 칸 600x600 + 위 150 글자 띠 (가로 2400), yuv420p, faststart; 색 범위 조정 없음.
사용: python exe/fem_wolfnew_compose.py <정답> <(A) 영상> <(A) summary> <(B) 영상> <(B) summary> <(B)+보간 영상> <(B)+보간 summary> <글꼴 Bold> <글꼴 Regular> <출력 mp4> <임시 폴더>"""
import json
import os
import subprocess
import sys

gt, va, sa, vb, sb, vn, sn, FB, FR, out, TMP = sys.argv[1:12]
os.makedirs(TMP, exist_ok=True)
CW = 600


def er(p):
    s = json.load(open(p))
    dv = s.get("diverged")
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    return f"오차 {100 * e:.2f}%" + (f" · 프레임 {dv} 에서 발산" if dv else ""), dv


title = "wolf 50 프레임 · HHT ρ 0.8, K 2, 결정적 · (A) 가우스 점 고착 (PCG 3) · (B) 장벽 + --y0eb (PCG 5) · (B)+보간 교체 = (B) + --fp_interp"
cells = [("정답 (PG 모래)", None, gt, None)]
for nm, v, s in (("(A) 가우스 점 고착", va, sa), ("(B) 장벽", vb, sb), ("(B)+보간 교체 (새 기본 판)", vn, sn)):
    e_, dv = er(s)
    cells.append((nm, e_, v, dv))


def tf(name, text):
    p = f"{TMP}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(f, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


ins, fl = [], []
for k, (nm, e_, path, dv) in enumerate(cells):
    ins += ["-r", "30", "-i", path]
    f_ = f"[{k}:v]scale={CW}:{CW},setsar=1,pad={CW}:{CW + 150}:0:150:white," + ("tpad=stop_mode=clone:stop=-1," if dv else "") + dt(tf(f"c{k}a", nm), FB, 26, 12, 56)
    if e_:
        f_ += "," + dt(tf(f"c{k}b", e_), FB, 20, 12, 98, "0x1f4fbf")
    fl.append(f_ + f"[c{k}]")
n = len(cells)
fl.append("".join(f"[c{k}]" for k in range(n)) + f"hstack=inputs={n}:shortest=1,"
          + ",".join(f"drawbox=x={CW * k - 1}:y=46:w=2:h={CW + 104}:color=0xdddddd:t=fill" for k in range(1, n)) + ","
          + dt(tf("title", title), FR, 16, 12, 14, "0x444444") + "[o]")
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
                "-movflags", "+faststart", "-shortest", out], check=True)
print("[저장]", out)
