"""절 CM24 영상 (로컬): 수박 정답 | 지금 기본 판 | 고친 판 둘셋 -- 2 줄 x 3 칸 (칸 600x600 렌더) + 요약 칸.
  칸마다 판 이름 · 이 프레임 오차 · 그때까지 튐 (질량중심 z 가 그때까지 최저점보다 오른 최대, exe/wmk1fix_tab.py 와 같은 정의, erec zc).
  발산 판은 발산 프레임 뒤로 그 프레임에서 정지 (렌더는 stab_wmk1fix_vid.sh 가 마지막 유한 프레임으로 채움) + "발산 fN -- 정지".
  python3 exe/wmk1fix_compose.py --dir <gt_plain.mp4·<판>.mp4·<판>/fem.jsonl·summary.json> --runs base_a:지금 기본 판,ks3_a:... --out out.mp4
"""
import argparse
import json
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--runs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="수박 기본 판 (프레임당 1 걸음) 첫 접촉 걸음 덜 풀림 고치기 -- 3DGS 렌더 (일반 색, GF 카메라)")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--font", default="/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf")
a = ap.parse_args()
font_manager.fontManager.addfont(a.font)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=a.font).get_name()
RUNS = [tuple(q.split(":", 1)) for q in a.runs.split(",")]
assert len(RUNS) <= 4


def bounce(zs):
    lo, b, at = float("inf"), 0.0, None
    for i, z in enumerate(zs):
        if z is None:
            continue
        lo = min(lo, z)
        if z - lo > b:
            b, at = z - lo, i + 1
    return b, at


def res(s):
    return ("발산 f%d" % s["diverged"]) if s.get("diverged") else ("완주 %d" % s["frames_done"])


D = {}
for k, _ in RUNS:
    rows = [json.loads(l) for l in open(os.path.join(a.dir, k, "fem.jsonl"))]
    D[k] = (rows, json.load(open(os.path.join(a.dir, k, "summary.json"))))
lng = max((D[k][0] for k, _ in RUNS), key=len)
GZ = [r.get("com", {}).get("gz") for r in lng]
GB, GBat = bounce(GZ)
SUM = []
for k, lab in RUNS:
    rv, sv = D[k]
    b, ba = bounce([r.get("en", {}).get("zc") for r in rv])
    mv = 100 * (sv["err_mean"] if sv.get("err_mean") is not None else sv["err_mean_done"])
    SUM.append("%s: %s · 평균 오차 %.1f %% · 최대 튐 %.3f m" % (lab, res(sv), mv, b))
T = tempfile.mkdtemp()
VID = [("gt", "gt_plain")] + [(k, k) for k, _ in RUNS]
n = 10 ** 9
for k, v in VID:
    os.makedirs(f"{T}/{k}")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", os.path.join(a.dir, v + ".mp4"), f"{T}/{k}/%04d.png"], check=True)
    n = min(n, len(os.listdir(f"{T}/{k}")))
os.makedirs(f"{T}/o")
BOX = dict(facecolor="white", alpha=0.85, edgecolor="none")
for i in range(n):
    fig = plt.figure(figsize=(18, 13.6), dpi=100)
    fig.text(0.5, 0.992, a.title, ha="center", va="top", fontsize=16)
    fig.text(0.5, 0.962, "프레임 %d / %d · 튐 = 질량중심 z 가 그때까지 최저점보다 오른 최대 · 오차 = 정답 대비 입자 위치 오차" % (i, n - 1), ha="center", va="top", fontsize=11)
    cells = [("gt", "정답 (GF)")] + list(RUNS)
    for j, (k, lab) in enumerate(cells):
        x0 = j % 3 / 3.0; y0 = 0.48 - 0.47 * (j // 3)
        ax = fig.add_axes([x0 + 0.002, y0, 1 / 3.0 - 0.004, 0.44]); ax.axis("off")
        ax.imshow(plt.imread(f"{T}/{k}/{i + 1:04d}.png"))
        fig.text(x0 + 1 / 6.0, y0 + 0.443, lab, ha="center", va="bottom", fontsize=14, weight="bold")
        col = "black"
        if k == "gt":
            b, _ = bounce(GZ[:min(i, len(GZ))])
            s = "튐 (그때까지) %.3f m\n100 프레임 튐 %.3f m (f%s)" % (b, GB, GBat)
        else:
            rv, sv = D[k]
            dv = sv.get("diverged")
            tt = min(i, len(rv))
            b, _ = bounce([r.get("en", {}).get("zc") for r in rv[:tt]])
            e = ("%.1f %%" % (100 * rv[tt - 1]["err"])) if tt >= 1 else "-"
            s = "프레임 %d 오차 %s · 튐 (그때까지) %.3f m" % (tt, e, b)
            if dv and i >= dv:
                s = ("발산 프레임 f%d" % dv if i == dv else "발산 f%d -- 이후 정지" % dv) + "\n" + s
                col = "#c00000"
        fig.text(x0 + 0.006, y0 + 0.004, s, ha="left", va="bottom", fontsize=11, color=col, bbox=BOX)
    if len(cells) < 6:
        x0, y0 = 2 / 3.0, 0.01
        fig.text(x0 + 0.01, y0 + 0.43, "요약 (100 프레임)", ha="left", va="top", fontsize=12, weight="bold")
        fig.text(x0 + 0.01, y0 + 0.395, "\n\n".join(SUM) + "\n\n정답 튐 %.3f m (f%s)" % (GB, GBat), ha="left", va="top", fontsize=10.5, wrap=True)
    fig.savefig(f"{T}/o/{i:04d}.png"); plt.close(fig)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(a.fps), "-i", f"{T}/o/%04d.png", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", a.out], check=True)
print("영상", a.out, n, "프레임")
