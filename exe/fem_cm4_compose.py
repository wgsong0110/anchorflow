"""절 CM4 영상 합성 (로컬 ffmpeg, R2 마운트): 정답 | 우리 (절 CM3 채택 공통 설정, 묶음 cm4 v_<씬>), 칸 위에 판 이름·오차 (색 범위 조정 없음).
사용: python exe/fem_cm4_compose.py lego_L000|lego_0016|ficus|wolf|bread   (R2 lbours/fem_cm/marks/cm4.END 가 있어야 한다)
출력 ~/workspace/result/anchorflow/v20261009/fem_cm_<씬>_cmp.mp4"""
import json
import os
import subprocess
import sys

L = os.path.expanduser("~/r2/result/anchorflow/lbours")
V = f"{L}/fem_cm/cm4"
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
TMP = "/tmp/fem_cm4_compose"
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
os.makedirs(TMP, exist_ok=True)
sc = sys.argv[1]
CAP = {"ficus": 24, "wolf": 5, "bread": 2}
GT = {"lego_L000": (f"{L}/fem2_lego/gt_lego.mp4", "정답 (MPM)", "lego L000 80 프레임"),
      "lego_0016": (f"{L}/fem_af_lego/gt_0016/gt_lego.mp4", "정답 (MPM)", "lego 홀드아웃 0016 80 프레임"),
      "ficus": (f"{L}/fem3_ficus/gt_ficus.mp4", "정답 (PG)", "ficus F_pg_v0 125 프레임"),
      "wolf": (f"{L}/fem_ag_wolf/gt/gt_wolf.mp4", "정답 (PG 모래)", "wolf 50 프레임"),
      "bread": (f"{L}/fem_ag_bread/gt/gt_bread.mp4", "정답 (PG plasticine)", "bread 300 프레임")}
gt, gtn, sname = GT[sc]
t = sc[5:] if sc.startswith("lego_") else sc
sh = "lego" if sc.startswith("lego_") else sc
s = json.load(open(f"{V}/v_{t}/summary.json"))
e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
er = f"오차 {100 * e:.2f}%"
if t == "ficus" and os.path.exists(f"{V}/v_ficus/period.txt"):
    import re
    m = re.findall(r"정답 대비 ([0-9.]+)\)[^\n]*?진폭 [0-9.]+ m \(비 ([0-9.]+)\)", open(f"{V}/v_ficus/period.txt").read())
    if m:
        er += f" · 주기 비 {m[-1][0]} · 진폭 비 {m[-1][1]}"
c = CAP.get(t, 6)
title = f"{sname} · 암시적 B-스플라인 실시간 판 공통 설정 (HHT ρ 0.8, K 2, εN 1e-2 · εP 1e-1, 상한 뉴턴 2 x PCG {c}) · 결정적"
cells = [(gtn, None, gt), ("우리 (공통 설정)", er, f"{V}/v_{t}/vj/fem_{sh}.mp4")]


def tf(name, text):
    p = f"{TMP}/{sc}_{name}.txt"
    open(p, "w").write(text)
    return p


def dt(f, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


ins, fl = [], []
for k, (nm, e_, path) in enumerate(cells):
    ins += ["-i", path]
    f_ = f"[{k}:v]scale=800:800,pad=800:1000:0:200:white," + dt(tf(f"c{k}a", nm), FB, 32, 16, 70)
    if e_:
        f_ += "," + dt(tf(f"c{k}b", e_), FB, 32, 16, 124, "0x1f4fbf")
    fl.append(f_ + f"[c{k}]")
fl.append("[c0][c1]hstack=inputs=2,drawbox=x=799:y=60:w=2:h=940:color=0xdddddd:t=fill,"
          + dt(tf("title", title), FR, 20, 16, 16, "0x444444") + "[o]")
out = f"{OUT}/fem_cm_{sc}_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", "-shortest", out], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", out, "-frames:v", "1", f"{TMP}/chk_{sc}.png"], check=True)
print("[저장]", out)
