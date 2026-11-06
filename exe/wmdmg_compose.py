"""절 CM19: 수박 손상 모델 비교 영상 합성 (로컬) -- 2 줄 x 4 칸 + 색 막대 둘, 가로 2400.

  윗줄 (logJp 색, −0.2~0.4 공통 범위): 정답 | SNH | CDM | 위상장
  아랫줄: 정답 일반 3DGS | SNH 일반 3DGS | CDM 손상 d 색 (0~1) | 위상장 손상 d 색 (0~1)
  python3 exe/wmdmg_compose.py --v gt_lj.mp4 snh_lj.mp4 cdm_lj.mp4 pf_lj.mp4 gt_plain.mp4 snh_plain.mp4 cdm_d.mp4 pf_d.mp4 --lab 'l1' ... (8 개) --title .. --note .. --out out.mp4
"""
import argparse
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--v", nargs=8, required=True)
ap.add_argument("--lab", nargs=8, required=True)
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
P, TT, LB, BOT, CB = 540, 50, 44, 60, 240
TW = 4 * P + CB
TH = TT + 2 * (LB + P) + BOT


def cbar(path, vmin, vmax, ticks, labels, title):
    fig = plt.figure(figsize=(CB / 100, (LB + P) / 100), dpi=100)
    cax = fig.add_axes([0.12, 0.05, 0.16, 0.88])
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin, vmax), cmap="viridis")
    cb = fig.colorbar(sm, cax=cax, extend="both" if vmin < 0 else "neither")
    cb.set_ticks(ticks); cb.set_ticklabels(labels); cb.ax.tick_params(labelsize=10)
    cb.set_label(title, fontsize=10)
    fig.savefig(path, facecolor="white"); plt.close(fig)


def txt(path, w, h, s, fs):
    f = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    f.text(0.5, 0.5, s, ha="center", va="center", fontsize=fs, wrap=True)
    f.savefig(path, facecolor="white"); plt.close(f)


cbar(f"{D}/cb1.png", -0.2, 0.4, [-0.2, -0.04, 0.0, 0.2, 0.4], ["−0.2", "−0.04 (처음)", "0 (응집력 0)", "0.2", "0.4 (GF 숨김)"], "윗줄: logJp (NACC 경화), 공통 범위")
cbar(f"{D}/cb2.png", 0.0, 1.0, [0, 0.5, 1], ["0", "0.5 (손상 문턱)", "1"], "아랫줄 오른쪽 둘: 손상 d, 공통 범위")
txt(f"{D}/tt.png", TW, TT, a.title, 14)
for i, s in enumerate(a.lab):
    txt(f"{D}/l{i}.png", P, LB, s, 11)
txt(f"{D}/nb.png", TW, BOT, a.note, 10)
ins = []
for v in a.v:
    ins += ["-i", v]
imgs = ["tt", "cb1", "cb2", "nb"] + [f"l{i}" for i in range(8)]
for m in imgs:
    ins += ["-loop", "1", "-i", f"{D}/{m}.png"]
fc = [f"color=white:s={TW}x{TH}:r={a.fps}[bg]"]
for i in range(8):
    fc.append(f"[{i}:v]scale={P}:{P}[v{i}]")
cur = "bg"
k = 0


def ov(src, x, y):
    global cur, k
    fc.append(f"[{cur}][{src}]overlay={x}:{y}:shortest=1[o{k}]"); cur = f"o{k}"; k += 1


for i in range(8):
    r, c = divmod(i, 4)
    ov(f"v{i}", c * P, TT + r * (LB + P) + LB)
ov("8:v", 0, 0)
ov("9:v", 4 * P, TT)
ov("10:v", 4 * P, TT + LB + P)
ov("11:v", 0, TT + 2 * (LB + P))
for i in range(8):
    r, c = divmod(i, 4)
    ov(f"{12 + i}:v", c * P, TT + r * (LB + P))
fc.append(f"[{cur}]drawtext=fontfile={a.font}:text='프레임 %{{eif\\:n\\:d}}':x={4 * P + 20}:y={TT + 2 * (LB + P) + 15}:fontsize=22:fontcolor=black,format=yuv420p[v]")


def nfr(p):
    return int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", p],
                              capture_output=True, text=True).stdout.strip())


NF = min(nfr(v) for v in a.v)
cmd = ["ffmpeg", "-y", "-loglevel", "error"] + ins + ["-filter_complex", ";".join(fc), "-map", "[v]", "-frames:v", str(NF),
                                                      "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
subprocess.run(cmd, check=True)
print("[합성]", a.out, f"{TW}x{TH}", NF, "프레임")
