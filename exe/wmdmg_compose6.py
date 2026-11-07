"""절 CM19 (사용자 11-07): 셀별 여섯 칸 손상 비교 영상 (로컬) -- 2 줄 x 3 칸 + 색 막대 둘, 가로 2340.

  윗줄: 정답 (logJp) | SNH (logJp) | CDM 정수압 (d)
  아랫줄: CDM 단축 (d) | 위상장 정수압 (d) | 위상장 단축 (d)
  logJp 칸은 −0.2~0.4, d 칸은 0~1 -- 모든 프레임·칸 공통 범위 (프레임별 정규화 없음).
  --div 칸별 발산 프레임 (0 = 완주): 그 프레임부터 칸 위에 "프레임 N 발산 -- 멈춤" (렌더 칸은 마지막 유한 프레임에서 멈춰 있음).
  python3 exe/wmdmg_compose6.py --v 6개 --lab 6개 --div 6개 --title .. --note .. --out out.mp4
"""
import argparse
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--v", nargs=6, required=True)
ap.add_argument("--lab", nargs=6, required=True)
ap.add_argument("--div", nargs=6, type=int, required=True)
ap.add_argument("--title", required=True)
ap.add_argument("--note", required=True)
ap.add_argument("--fps", type=int, default=32)
ap.add_argument("--font", default="/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf")
ap.add_argument("--out", required=True)
a = ap.parse_args()
font_manager.fontManager.addfont(a.font)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=a.font).get_name()
plt.rcParams["axes.unicode_minus"] = False
D = tempfile.mkdtemp()
P, TT, LB, BOT, CB = 700, 50, 50, 60, 240
TW = 3 * P + CB
TH = TT + 2 * (LB + P) + BOT


def cbar(path, vmin, vmax, ticks, labels, title):
    fig = plt.figure(figsize=(CB / 100, (LB + P) / 100), dpi=100)
    cax = fig.add_axes([0.12, 0.05, 0.16, 0.88])
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin, vmax), cmap="viridis")
    cb = fig.colorbar(sm, cax=cax, extend="both" if vmin < 0 else "neither")
    cb.set_ticks(ticks); cb.set_ticklabels(labels); cb.ax.tick_params(labelsize=10)
    cb.set_label(title, fontsize=10)
    fig.savefig(path, facecolor="white"); plt.close(fig)


def txt(path, w, h, s, fs, bg="white", fg="black"):
    f = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    f.text(0.5, 0.5, s, ha="center", va="center", fontsize=fs, wrap=True, color=fg)
    f.savefig(path, facecolor=bg); plt.close(f)


cbar(f"{D}/cb1.png", -0.2, 0.4, [-0.2, -0.04, 0.0, 0.2, 0.4], ["−0.2", "−0.04 (처음)", "0 (응집력 0)", "0.2", "0.4 (GF 숨김)"], "정답·SNH 칸: logJp (NACC 경화), 공통 범위")
cbar(f"{D}/cb2.png", 0.0, 1.0, [0, 0.5, 1], ["0", "0.5 (손상 문턱)", "1"], "CDM·위상장 칸: 손상 d, 공통 범위")
txt(f"{D}/tt.png", TW, TT, a.title, 14)
for i, s in enumerate(a.lab):
    txt(f"{D}/l{i}.png", P, LB, s, 11)
    txt(f"{D}/d{i}.png", P, 40, f"프레임 {a.div[i]} 발산 -- 그 뒤 멈춤" if a.div[i] else "-", 14, bg="#c0392b", fg="white")
txt(f"{D}/nb.png", TW, BOT, a.note, 10)
ins = []
for v in a.v:
    ins += ["-i", v]
imgs = ["tt", "cb1", "cb2", "nb"] + [f"l{i}" for i in range(6)] + [f"d{i}" for i in range(6)]
for m in imgs:
    ins += ["-loop", "1", "-i", f"{D}/{m}.png"]
fc = [f"color=white:s={TW}x{TH}:r={a.fps}[bg]"]
for i in range(6):
    fc.append(f"[{i}:v]scale={P}:{P}[v{i}]")
cur = "bg"
k = 0


def ov(src, x, y, en=None):
    global cur, k
    fc.append(f"[{cur}][{src}]overlay={x}:{y}:shortest=1" + (f":enable='gte(n,{en})'" if en is not None else "") + f"[o{k}]"); cur = f"o{k}"; k += 1


def pos(i):
    r, c = divmod(i, 3)
    return c * P, TT + r * (LB + P)


for i in range(6):
    x, y = pos(i); ov(f"v{i}", x, y + LB)
ov("6:v", 0, 0)
ov("7:v", 3 * P, TT)
ov("8:v", 3 * P, TT + LB + P)
ov("9:v", 0, TT + 2 * (LB + P))
for i in range(6):
    x, y = pos(i); ov(f"{10 + i}:v", x, y)
for i in range(6):
    if a.div[i]:
        x, y = pos(i); ov(f"{16 + i}:v", x, y + LB + P - 40, en=a.div[i])   # 영상 프레임 n = 시뮬 프레임 n (0 = 처음 상태)
fc.append(f"[{cur}]drawtext=fontfile={a.font}:text='프레임 %{{eif\\:n\\:d}}':x={3 * P + 20}:y={TT + 2 * (LB + P) + 15}:fontsize=22:fontcolor=black,format=yuv420p[v]")


def nfr(p):
    return int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", p],
                              capture_output=True, text=True).stdout.strip())


NF = min(nfr(v) for v in a.v)
cmd = ["ffmpeg", "-y", "-loglevel", "error"] + ins + ["-filter_complex", ";".join(fc), "-map", "[v]", "-frames:v", str(NF),
                                                      "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
subprocess.run(cmd, check=True)
print("[합성]", a.out, f"{TW}x{TH}", NF, "프레임")
