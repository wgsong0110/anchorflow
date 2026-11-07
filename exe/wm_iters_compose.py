"""절 CM23.3 영상 (로컬): 수박 기본 판 K 1 반복 상한별 비교 -- 2 줄 x 3 칸 (각 600x600): 정답 | 2x5 | 4x10 / 8x20 | 16x40 | 요약 글자.
  칸마다 판 이름 · 이 프레임 오차 · 그때까지 튐 (질량중심 z 가 그때까지 최저점보다 오른 최대, exe/wmk1conv_tab.py 와 같은 정의).
  발산 판은 발산 프레임 뒤로 그 프레임에서 정지 (렌더는 stab_wmk1vid.sh 가 마지막 유한 프레임으로 채움) + "발산 fN -- 정지" 표시.
  요약 칸: 덤프 판 (v_*, 렌더한 판) 대 표 판 (d_*, 절 CM23.3 표) 오차 곡선 대조.
  python3 exe/wm_iters_compose.py --dir <gt_plain.mp4 · N2P5.mp4 ... · v_*/ d_* fem.jsonl·summary.json> --out out.mp4 --check check.md
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
ap.add_argument("--out", required=True)
ap.add_argument("--check", required=True)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--font", default="/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf")
a = ap.parse_args()
font_manager.fontManager.addfont(a.font)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=a.font).get_name()
RUNS = [("N2P5", "2 × 5 (기본)"), ("N4P10", "4 × 10"), ("N8P20", "8 × 20"), ("N16P40", "16 × 40")]


def load(n):
    rows = [json.loads(l) for l in open(os.path.join(a.dir, n, "fem.jsonl"))]
    s = json.load(open(os.path.join(a.dir, n, "summary.json")))
    return rows, s


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
    D[k] = dict(v=load("v_" + k), d=load("d_" + k))
# 정답 질량중심 z: 가장 긴 판의 gz (모든 판 같은 정답)
lng = max((D[k]["v"][0] for k, _ in RUNS), key=len)
GZ = [r.get("com", {}).get("gz") for r in lng]
GB, GBat = bounce(GZ)
# ---- 덤프 판 대 표 판 대조
L = ["# 절 CM23.3 영상 -- 덤프 판 (v_*, 렌더한 판 = 진단 판 명령 + --dump) 대 표 판 (d_*, wmk1conv_table.md 진단 판)", "",
     "비결정 실행이라 같은 명령도 조금 다를 수 있음. 오차 차 = 공통 프레임 |오차 v - 오차 d| (%p).", "",
     "| 상한 | 덤프 판 결과 · 평균 오차 % | 표 판 결과 · 평균 오차 % | 오차 차 최대 %p (프레임) · f1~4 최대 %p | 오차 % f4 · f8 · f12 · f20 (덤프 판) | 튐 m 덤프 판 · 표 판 (정답 " + "%.3f) |" % GB,
     "|---|---|---|---|---|---|"]
SUM = []
for k, lab in RUNS:
    (rv, sv), (rd, sd) = D[k]["v"], D[k]["d"]
    ev = [r["err"] for r in rv]; ed = [r["err"] for r in rd]
    m = min(len(ev), len(ed))
    df = [abs(ev[i] - ed[i]) * 100 for i in range(m)]
    j = max(range(m), key=lambda i: df[i])
    bv, bva = bounce([r.get("com", {}).get("z") for r in rv]); bd, bda = bounce([r.get("com", {}).get("z") for r in rd])
    mv = 100 * (sv["err_mean"] if sv.get("err_mean") is not None else sv["err_mean_done"])
    md = 100 * (sd["err_mean"] if sd.get("err_mean") is not None else sd["err_mean_done"])
    pick = " · ".join(("%.1f" % (100 * ev[f - 1])) if f <= len(ev) else "-" for f in (4, 8, 12, 20))
    L.append("| %s | %s · %.2f | %s · %.2f | %.2f (f%d) · %.3f | %s | %.3f (f%s) · %.3f (f%s) |" % (
        lab, res(sv), mv, res(sd), md, df[j], j + 1, max(df[:4]), pick, bv, bva, bd, bda))
    SUM.append("%s  덤프 판 %s · 평균 %.1f %% · 튐 %.2f m  |  표 판 %s · %.1f %%  |  오차 차 최대 %.1f %%p" % (lab, res(sv), mv, bv, res(sd), md, df[j]))
open(a.check, "w").write("\n".join(L) + "\n")
print("\n".join(L))
# ---- 영상
T = tempfile.mkdtemp()
VID = [("gt", "gt_plain")] + [(k, k) for k, _ in RUNS]
n = 10 ** 9
for k, v in VID:
    os.makedirs(f"{T}/{k}")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", os.path.join(a.dir, v + ".mp4"), f"{T}/{k}/%04d.png"], check=True)
    n = min(n, len(os.listdir(f"{T}/{k}")))
os.makedirs(f"{T}/o")
BOX = dict(facecolor="white", alpha=0.85, edgecolor="none")
for i in range(n):                                        # 영상 프레임 i (0 = 처음 상태, i >= 1 = 시뮬 프레임 i = jsonl t i-1)
    fig = plt.figure(figsize=(18, 13.6), dpi=100)
    fig.text(0.5, 0.992, "수박 기본 판 (프레임당 1 걸음) -- 반복 상한 뉴턴 × PCG 별 (허용치 εN 1e-2 · εP 1e-1 그대로)", ha="center", va="top", fontsize=17)
    fig.text(0.5, 0.962, "프레임 %d / %d · 3DGS 알파블렌딩 렌더 (일반 색, GF 카메라) · 튐 = 질량중심 z 가 그때까지 최저점보다 오른 최대" % (i, n - 1),
             ha="center", va="top", fontsize=11)
    cells = [("gt", "정답 (GF)")] + list(RUNS)
    for j, (k, lab) in enumerate(cells):
        x0 = j % 3 / 3.0; y0 = 0.48 - 0.47 * (j // 3)
        ax = fig.add_axes([x0 + 0.002, y0, 1 / 3.0 - 0.004, 0.44]); ax.axis("off")
        ax.imshow(plt.imread(f"{T}/{k}/{i + 1:04d}.png"))
        fig.text(x0 + 1 / 6.0, y0 + 0.443, lab, ha="center", va="bottom", fontsize=15, weight="bold")
        if k == "gt":
            tt = min(i, len(GZ))
            b, _ = bounce(GZ[:tt])
            s = "튐 (그때까지) %.3f m\n100 프레임 튐 %.3f m (f%s)" % (b, GB, GBat)
            col = "black"
        else:
            rv, sv = D[k]["v"]
            dv = sv.get("diverged")
            tt = min(i, len(rv))                          # 지금까지 보인 시뮬 프레임 수
            b, _ = bounce([r.get("com", {}).get("z") for r in rv[:tt]])
            e = ("%.1f %%" % (100 * rv[tt - 1]["err"])) if tt >= 1 else "-"
            fb, fba = bounce([r.get("com", {}).get("z") for r in rv])
            mv = 100 * (sv["err_mean"] if sv.get("err_mean") is not None else sv["err_mean_done"])
            s = "프레임 %s 오차 %s · 튐 (그때까지) %.2f m\n%s · 평균 오차 %.1f %% · 최대 튐 %.2f m (f%s)" % (tt if tt else 0, e, b, res(sv), mv, fb, fba)
            col = "black"
            if dv and i > dv:
                s = "발산 f%d -- 이후 정지\n" % dv + s
                col = "#c00000"
            elif dv and i == dv:
                s = "발산 프레임 f%d\n" % dv + s
                col = "#c00000"
        fig.text(x0 + 0.006, y0 + 0.004, s, ha="left", va="bottom", fontsize=11, color=col, bbox=BOX)
    # 요약 칸
    x0, y0 = 2 / 3.0, 0.01
    fig.text(x0 + 0.01, y0 + 0.43, "요약 (렌더한 덤프 판 대 절 CM23.3 표 판, 같은 명령 비결정 두 번)", ha="left", va="top", fontsize=12, weight="bold")
    fig.text(x0 + 0.01, y0 + 0.395, "\n\n".join(SUM).replace("  |  ", "\n    ") +
             "\n\n정답 튐 %.3f m · 오차 = 정답 대비 입자 위치 오차 (%%)" % GB, ha="left", va="top", fontsize=10.5)
    fig.savefig(f"{T}/o/{i:04d}.png"); plt.close(fig)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(a.fps), "-i", f"{T}/o/%04d.png", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", a.out], check=True)
print("영상", a.out, n, "프레임")
