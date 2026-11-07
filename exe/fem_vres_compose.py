"""절 CM15 영상 합성 (인스턴스, ffmpeg): exe/fem_barnode_compose.py 그대로에 (1) hstack shortest=1 (발산 칸 정지 화면이 끝나지 않던 것, 절 CM12.2), (2) 가로 4096 넘으면 비율 그대로 줄임, (3) faststart.
칸 여러 개를 가로로 (칸 위 글자 띠에 판 이름·오차, 맨 위 제목; 색 범위 조정 없음).
칸 높이 720 에 맞춰 비율 그대로 (lego·wolf 정답 영상 크기가 우리 렌더와 다를 수 있음), 발산한 판은 마지막 프레임에서 정지 화면.
사용: python exe/fem_vres_compose.py <출력 mp4> <제목> <글꼴 Bold> <글꼴 Regular> <임시 폴더> [<칸 이름> <영상> <summary.json 또는 ->]..."""
import json
import os
import subprocess
import sys

out, title, FB, FR, TMP = sys.argv[1:6]
rest = sys.argv[6:]
assert len(rest) % 3 == 0, rest
os.makedirs(TMP, exist_ok=True)
H, TOP = 720, 180
MAXW = int(os.environ.get("CMP_MAXW", "4096"))                                     # 절 CM17: 가로 상한 (기본 4096 그대로)


def er(p):
    if p == "-":
        return None, None
    s = json.load(open(p))
    dv = s.get("diverged")
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    return f"오차 {100 * e:.2f}%" + (f" · 프레임 {dv} 에서 발산 (뒤는 정지)" if dv else ""), dv


def tf(name, text):
    p = f"{TMP}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(f, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


cells = []
for i in range(0, len(rest), 3):
    nm, v, s = rest[i:i + 3]
    if not os.path.exists(v):
        print("[칸 빠짐: 영상 없음]", nm, v)
        continue
    e_, dv = er(s)
    cells.append((nm, e_, v, dv))
ins, fl = [], []
for k, (nm, e_, path, dv) in enumerate(cells):
    ins += (["-r", os.environ["CMP_IN_R"]] if os.environ.get("CMP_IN_R") else []) + ["-i", path]          # 절 CM20: 입력 프레임률을 출력 30 으로 다시 읽음 (수박 32 fps 렌더의 프레임 빠짐 막기; 없으면 그대로)
    f_ = f"[{k}:v]scale=-2:{H},pad=iw:{H + TOP}:0:{TOP}:white," + ("tpad=stop_mode=clone:stop=-1," if dv else "") + dt(tf(f"c{k}a", nm), FB, 26, 14, 62)
    if e_:
        f_ += "," + dt(tf(f"c{k}b", e_), FB, 20, 14, 110, "0x1f4fbf")
    fl.append(f_ + f"[c{k}]")
n = len(cells)
fl.append("".join(f"[c{k}]" for k in range(n)) + f"hstack=inputs={n}:shortest=1," + dt(tf("title", title), FR, 18, 14, 14, "0x444444") + f",scale=w='min(iw,{MAXW})':h=-2[o]")
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", "-movflags", "+faststart", "-shortest", out], check=True)
print("[저장]", out, n, "칸")
