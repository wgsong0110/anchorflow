"""절 J 최종 판 영상 합성 (로컬 ffmpeg drawtext + Noto Sans CJK KR; % 가 지워지지 않게 textfile + expansion=none).
  (1) 정답 | 최종 판 (1600x960, 상단 띠: 번호·판 이름 / "프레임당 XX.X ms · XX.X fps · 오차 XX.XX%")
  (2) 정답 | ② 이전 기준 판 (i_tmix_asm_r3_F) | 최종 판 (2400x960, 칸마다 같은 띠)
사용: python exe/fem_rt_compose_j.py <최종 태그> "<최종 판 이름>" <출력 접두>"""
import json
import os
import subprocess
import sys

R2 = os.path.expanduser("~/r2/result/anchorflow/lbours")
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
TMP = "/tmp/fem_j_compose"
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
GT = f"{R2}/fem2_lego/gt_lego.mp4"
TAG, NAME, PFX = sys.argv[1], sys.argv[2], sys.argv[3]
PREV = ("i_tmix_asm_r3_F", "입자 바닥 + 혼합 정밀도 PCG (이전 기준 판)", f"{R2}/fem_rt_lego/i_tmix_asm_r3_F/v7/fem_lego.mp4")
os.makedirs(TMP, exist_ok=True)


def stats(tag):
    s = json.load(open(f"{R2}/fem_rt_lego/{tag}/summary.json"))
    return s["ms_mean"], s["fps"], 100 * s["err_mean"]


def tf(name, text):
    p = f"{TMP}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(textfile, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={textfile}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


def run(cmd):
    print(" ".join(cmd[:3]), "...", cmd[-1], flush=True)
    subprocess.run(cmd, check=True)


ms, fps, err = stats(TAG)
pms, pfps, perr = stats(PREV[0])
src = f"{R2}/fem_rt_lego/{TAG}/vj/fem_lego.mp4"
# (1) 정답 | 최종
t1 = tf("a1", f"⑧ {NAME}")
t2 = tf("a2", f"프레임당 {ms:.1f} ms · {fps:.1f} fps · 오차 {err:.2f}%")
tl = tf("al", "왼쪽: 정답    오른쪽: ⑧")
out1 = f"{OUT}/{PFX}.mp4"
fc = ("[0:v][1:v]hstack=inputs=2[s];[s]pad=1600:960:0:160:white,"
      + dt(t1, FB, 50, 30, 18) + "," + dt(t2, FB, 46, 30, 92, "0x1f4fbf") + "," + dt(tl, FR, 26, 1290, 20, "0x555555") + "[o]")
run(["ffmpeg", "-v", "error", "-y", "-i", GT, "-i", src, "-filter_complex", fc, "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", out1])
# (2) 정답 | 이전 기준 | 최종 (칸 800x800 + 위 띠 160)
cells = [("정답 (MPM)", None, GT), (f"② {PREV[1]}", (pms, pfps, perr), PREV[2]), (f"⑧ {NAME}", (ms, fps, err), src)]
ins, fl = [], []
for k, (nm, st, path) in enumerate(cells):
    ins += ["-i", path]
    a = tf(f"c{k}a", nm)
    sz = 30 if len(nm) <= 26 else 25
    f_ = f"[{k}:v]scale=800:800,pad=800:960:0:160:white," + dt(a, FB, sz, 16, 14)
    if st is not None:
        b = tf(f"c{k}b", f"프레임당 {st[0]:.1f} ms · {st[1]:.1f} fps · 오차 {st[2]:.2f}%")
        f_ += "," + dt(b, FB, 36, 16, 84, "0x1f4fbf")
    fl.append(f_ + f"[c{k}]")
fl.append("[c0][c1][c2]hstack=inputs=3,drawbox=x=799:y=0:w=2:h=960:color=0xdddddd:t=fill,drawbox=x=1599:y=0:w=2:h=960:color=0xdddddd:t=fill[o]")
out2 = f"{OUT}/{PFX}_vs_prev.mp4"
run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", out2])
for o in (out1, out2):
    run(["ffmpeg", "-v", "error", "-y", "-ss", "2", "-i", o, "-frames:v", "1", f"{TMP}/chk_{os.path.basename(o)}.png"])
print(json.dumps(dict(final=TAG, ms=ms, fps=fps, err=err, prev=PREV[0], pms=pms, pfps=pfps, perr=perr, out=[out1, out2]), ensure_ascii=False))
