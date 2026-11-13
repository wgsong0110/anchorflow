"""wolf 탄성 에너지 색 비교 합성 (로컬 ffmpeg + PIL, exe/wolfen_compose.py 꼴): 정답 | H9 | 뉴턴 60 (cvb) 세 칸 모두 3DGS 에너지 색 + 오른쪽 로그 색 막대 (J/m^3),
칸마다 프레임 번호·전체 탄성 에너지. 색 범위 = en.json lo·hi (모든 칸·프레임 공통 고정).
사용: python exe/cvben_compose.py <en.json> <gt_en.mp4> <nb9_en.mp4> <ncvb_en.mp4> <출력 mp4> <임시 폴더>"""
import json, os, subprocess, sys, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
en, vg, v1, v2, out, TMP = sys.argv[1:7]
os.makedirs(TMP, exist_ok=True)
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf"); FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
fm.fontManager.addfont(FR); plt.rcParams["font.family"] = "Noto Sans CJK KR"
J = json.load(open(en)); lo, hi = J["lo"], J["hi"]; E = J["E"]
CW = 700; BAR = 300; TOP = 130
names = [("정답", "", vg, E["gt"]), ("H9", "(뉴턴 6 · PCG 5)", v1, E["nb9"]), ("뉴턴 60 (cvb)", "(뉴턴 60 · PCG 100)", v2, E["ncvb"])]
fr = {}
for k, (_, _, v, _) in enumerate(names):
    d = f"{TMP}/f{k}"; os.makedirs(d, exist_ok=True)
    for p in glob.glob(f"{d}/*.png"): os.remove(p)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v, f"{d}/%04d.png"], check=True)
    fr[k] = sorted(glob.glob(f"{d}/*.png"))
T = min(len(x) for x in fr.values()); assert T == J["L"] + 1, (T, J["L"])
w0, h0 = Image.open(fr[0][0]).size
CH = round(CW * h0 / w0); W = 3 * CW + BAR; H = TOP + CH
print(f"프레임 {T} 칸 {CW}x{CH} 전체 {W}x{H}", flush=True)
fig = plt.figure(figsize=(BAR / 100, CH / 100), dpi=100)
ax = fig.add_axes([0.12, 0.16, 0.2, 0.70])
ax.imshow(np.linspace(1, 0, 256)[:, None], aspect="auto", cmap="turbo", extent=[0, 1, lo, hi]); ax.set_xticks([])
ks = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1))
ax.set_yticks(ks); ax.set_yticklabels([f"$10^{{{k}}}$" for k in ks], fontsize=15); ax.yaxis.tick_right(); ax.set_ylim(lo, hi)
fig.text(0.04, 0.93, "탄성 에너지 밀도 ψ\n(J/m³, 로그 눈금)", fontsize=14, va="center", ha="left")
fig.text(0.04, 0.015, f"세 칸·전 프레임 공통 고정\n{10**lo:.2g} ~ {10**hi:.2g} J/m³ (밖은 끝 색)", fontsize=10.5, va="bottom", ha="left", color="#444")
fig.canvas.draw(); bar = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert("RGB"); plt.close(fig)
f1, f2, f3 = (ImageFont.truetype(FB, 30), ImageFont.truetype(FR, 24), ImageFont.truetype(FB, 24))
os.makedirs(f"{TMP}/out", exist_ok=True)
for t in range(T):
    im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
    for k, (nm, sub, _, Ek) in enumerate(names):
        x = k * CW
        im.paste(Image.open(fr[k][t]).convert("RGB").resize((CW, CH), Image.LANCZOS), (x, TOP))
        dr.text((x + 14, 10), nm, fill="black", font=f1)
        if sub: dr.text((x + 14 + dr.textlength(nm, font=f1) + 12, 16), sub, fill="#444", font=f2)
        dr.text((x + 14, 52), f"프레임 {t}", fill=(200, 0, 0), font=f3)
        dr.text((x + 14, 88), f"전체 탄성 에너지 {Ek[t]:.4g} J", fill="black", font=f2)
    im.paste(bar, (3 * CW, TOP))
    dr.text((3 * CW + 12, 10), "wolf 50 프레임", fill="black", font=f2)
    dr.text((3 * CW + 12, 46), "입자별 탄성 에너지 색", fill="black", font=f2)
    im.save(f"{TMP}/out/{t:04d}.png")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "10", "-i", f"{TMP}/out/%04d.png", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
print("저장", out, flush=True)
