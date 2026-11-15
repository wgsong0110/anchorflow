"""wolf 정답 | 기준 (CM139 Kacdhnz) | CM177 뉴턴 완전 수렴 K4 (TVB+k4) 3 칸 비교 + 에너지 색 영상 합성 (로컬 ffmpeg).
사용: python exe/wolfbest_compose.py <자료 디렉토리> (gt_wolf.mp4 Kacdhnz.mp4 k4.mp4 gt_en.mp4 a_en.mp4 k4_en.mp4 en139.json en177.json) -> fem_wolf_best_cmp.mp4, fem_wolf_best_energy.mp4"""
import json, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import glob
D = sys.argv[1]
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf"); FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
fm.fontManager.addfont(FR); plt.rcParams["font.family"] = "Noto Sans CJK KR"
cells = [("정답 (PG 모래)", ""), ("기준 (H9 − 확장·S_구속·C)", ""), ("뉴턴을 걸음마다 완전히 푼 K4", "서브스텝 4 · 완전 수렴(실시간 아님)")]
# --- 3DGS 비교
W = 780; T = f"{D}/txt"; os.makedirs(T, exist_ok=True)
files = ["gt_wolf.mp4", "Kacdhnz.mp4", "k4.mp4"]; ins, fl = [], []
for k, ((a, b), f) in enumerate(zip(cells, files)):
    ins += ["-i", f"{D}/{f}"]
    p1 = f"{T}/c{k}.txt"; open(p1, "w").write(a); p2 = f"{T}/s{k}.txt"; open(p2, "w").write(b or " ")
    fl.append(f"[{k}:v]fps=30,tpad=stop=-1:stop_mode=clone,scale={W}:{W},setsar=1,pad={W}:{W+90}:0:90:white,"
              f"drawtext=fontfile={FB}:textfile={p1}:fontsize=28:fontcolor=black:expansion=none:x=12:y=12,"
              f"drawtext=fontfile={FB}:textfile={p2}:fontsize=22:fontcolor=0x444444:expansion=none:x=12:y=52[c{k}]")
fl.append("[c0][c1][c2]hstack=inputs=3," + ",".join(f"drawbox=x={W*j-1}:y=0:w=2:h={W+90}:color=0xdddddd:t=fill" for j in range(1, 3)) + "[o]")
out = f"{D}/fem_wolf_best_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "30", "-frames:v", "51", out], check=True)
print("저장", out, flush=True)
# --- 에너지
J1 = json.load(open(f"{D}/en139.json")); J2 = json.load(open(f"{D}/en177.json"))
assert J1["lo"] == J2["lo"] and J1["hi"] == J2["hi"]; lo, hi = J1["lo"], J1["hi"]
E = [J1["E"]["gt"], J1["E"]["a"], J2["E"]["k4"]]
vids = ["gt_en.mp4", "a_en.mp4", "k4_en.mp4"]
CW = 700; BAR = 240; TOP = 130
fr = []
for k, v in enumerate(vids):
    d = f"{D}/ef{k}"; os.makedirs(d, exist_ok=True)
    for p in glob.glob(f"{d}/*.png"): os.remove(p)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{D}/{v}", f"{d}/%04d.png"], check=True)
    fr.append(sorted(glob.glob(f"{d}/*.png")))
Tn = min(len(x) for x in fr); assert Tn == 51, Tn
w0, h0 = Image.open(fr[0][0]).size; CH = round(CW * h0 / w0); Wd = 3 * CW + BAR; H = TOP + CH
fig = plt.figure(figsize=(BAR / 100, CH / 100), dpi=100)
ax = fig.add_axes([0.12, 0.16, 0.2, 0.70])
ax.imshow(np.linspace(1, 0, 256)[:, None], aspect="auto", cmap="turbo", extent=[0, 1, lo, hi]); ax.set_xticks([])
ks = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1))
ax.set_yticks(ks); ax.set_yticklabels([f"$10^{{{k}}}$" for k in ks], fontsize=15); ax.yaxis.tick_right(); ax.set_ylim(lo, hi)
fig.text(0.04, 0.93, "탄성 에너지 밀도 ψ\n(J/m³, 로그 눈금)", fontsize=14, va="center", ha="left")
fig.text(0.04, 0.015, f"세 칸·전 프레임 공통 고정\n{10**lo:.2g} ~ {10**hi:.2g} J/m³ (밖은 끝 색)", fontsize=10.5, va="bottom", ha="left", color="#444")
fig.canvas.draw(); bar = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert("RGB"); plt.close(fig)
f1, f2, f3 = ImageFont.truetype(FB, 24), ImageFont.truetype(FR, 19), ImageFont.truetype(FB, 24)
os.makedirs(f"{D}/eout", exist_ok=True)
for t in range(Tn):
    im = Image.new("RGB", (Wd, H), "white"); dr = ImageDraw.Draw(im)
    for k, (nm, sub) in enumerate(cells):
        x = k * CW
        im.paste(Image.open(fr[k][t]).convert("RGB").resize((CW, CH), Image.LANCZOS), (x, TOP))
        dr.text((x + 12, 8), nm, fill="black", font=f1)
        dr.text((x + 12, 40), sub, fill="#444", font=f2)
        dr.text((x + 12, 70), f"프레임 {t}", fill=(200, 0, 0), font=f3)
        dr.text((x + 12, 102), f"전체 탄성 에너지 {E[k][t]:.4g} J", fill="black", font=f2)
    im.paste(bar, (3 * CW, TOP))
    dr.text((3 * CW + 12, 10), "wolf 50 프레임", fill="black", font=f2)
    dr.text((3 * CW + 12, 46), "입자별 탄성 에너지 색", fill="black", font=f2)
    im.save(f"{D}/eout/{t:04d}.png")
out = f"{D}/fem_wolf_best_energy.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "10", "-i", f"{D}/eout/%04d.png", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
print("저장", out, flush=True)
