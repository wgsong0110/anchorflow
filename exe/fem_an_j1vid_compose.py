"""절 AN.14 j1 비교 영상 합성 (로컬 ffmpeg, R2 마운트): 정답 | bej | hhj, 칸 위에 판 이름·오차 (색 범위 조정 없음).
사용: python exe/fem_an_j1vid_compose.py lego0003|lego0016|ficus   (R2 lbours/fem_an/j1vid/END 가 있어야 한다)"""
import json
import os
import subprocess
import sys

L = os.path.expanduser("~/r2/result/anchorflow/lbours")
V = f"{L}/fem_an/j1vid"
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
TMP = "/tmp/fem_an_j1vid_compose"
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
os.makedirs(TMP, exist_ok=True)
sc = sys.argv[1]
if sc == "ficus":
    t, sh, gt = "ficus", "ficus", f"{L}/fem3_ficus/gt_ficus.mp4"
    title = "ficus F_pg_v0 125 프레임 · 암시적 B-스플라인 실시간 판 j1 (공통 허용치 εN 1e-2 · εP 1e-1, 상한 뉴턴 2 x PCG 10) · 결정적"
else:
    t, sh = sc[4:], "lego"
    gt = f"{L}/fem_af_lego/gt_{t}/gt_lego.mp4"
    title = f"lego 홀드아웃 {t} 80 프레임 · 암시적 B-스플라인 실시간 판 j1 (공통 허용치 εN 1e-2 · εP 1e-1, 상한 뉴턴 1 x PCG 5) · 결정적"
NM = {"bej": "bej: 후퇴 오일러 K 2", "hhj": "hhj: HHT ρ 0.7 K 2"}
cells = [("정답 (MPM)" if sh == "lego" else "정답 (PG)", None, gt)]
for v in ("bej", "hhj"):
    s = json.load(open(f"{V}/{v}_{t}/summary.json"))
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    cells.append((NM[v], f"오차 {100 * e:.2f}%", f"{V}/{v}_{t}/vj/{v}_{sh}.mp4"))


def tf(name, text):
    p = f"{TMP}/{sc}_{name}.txt"
    open(p, "w").write(text)
    return p


def dt(f, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


ins, fl = [], []
for k, (nm, er, path) in enumerate(cells):
    ins += ["-i", path]
    f_ = f"[{k}:v]scale=800:800,pad=800:1000:0:200:white," + dt(tf(f"c{k}a", nm), FB, 32, 16, 70)
    if er:
        f_ += "," + dt(tf(f"c{k}b", er), FB, 36, 16, 124, "0x1f4fbf")
    fl.append(f_ + f"[c{k}]")
fl.append("[c0][c1][c2]hstack=inputs=3,drawbox=x=799:y=60:w=2:h=940:color=0xdddddd:t=fill,drawbox=x=1599:y=60:w=2:h=940:color=0xdddddd:t=fill,"
          + dt(tf("title", title), FR, 26, 16, 16, "0x444444") + "[o]")
out = f"{OUT}/fem_an_j1_{sc}_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", "-shortest", out], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "2", "-i", out, "-frames:v", "1", f"{TMP}/chk_{sc}.png"], check=True)
print("[저장]", out)
