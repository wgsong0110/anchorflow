"""wolf H9 칸 모은 영상 합성 (ffmpeg + PIL): 윗줄 우리 H9 (일반 3DGS | 탄성 에너지 밀도 | 운동 에너지 | 칸 높이 비), 아랫줄 정답 (일반 | 탄성 | 운동 | 해당 없음).
색 범위는 fields.json 의 두 판·전 프레임 고정값, 색 막대에 값·단위, 칸마다 프레임 번호와 전체값 (에너지 합, 높이 비 중앙값).
사용: python exe/wolffields_compose.py <fields.json> <vid 디렉토리 (gt_plain gt_en gt_ke h9_plain h9_en h9_ke h9_r .mp4)> <출력 mp4> <임시 폴더>"""
import json, os, subprocess, sys, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fj, vd, out, TMP = sys.argv[1:5]
os.makedirs(TMP, exist_ok=True)
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf"); FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
fm.fontManager.addfont(FR); plt.rcParams["font.family"] = "Noto Sans CJK KR"
J = json.load(open(fj))
CW, BW, RB, HEAD = 500, 120, 44, 62
need = ["gt_plain", "gt_en", "gt_ke", "h9_plain", "h9_en", "h9_ke", "h9_r"]
fr = {}
for k in need:
    d = f"{TMP}/f_{k}"; os.makedirs(d, exist_ok=True)
    for p in glob.glob(f"{d}/*.png"): os.remove(p)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{vd}/{k}.mp4", f"{d}/%04d.png"], check=True)
    fr[k] = sorted(glob.glob(f"{d}/*.png"))
T = min(len(x) for x in fr.values()); assert T == len(J["E_el_gt"]), (T, len(J["E_el_gt"]))
w0, h0 = Image.open(fr["gt_plain"][0]).size
CH = round(CW * h0 / w0); ROWH = RB + HEAD + CH
W = 4 * CW + 3 * BW; H = 2 * ROWH
print(f"프레임 {T} 칸 {CW}x{CH} 전체 {W}x{H}", flush=True)


def bar(title, unit, lo, hi, cmap, ticks, labels, note):
    fig = plt.figure(figsize=(BW / 100, H / 100), dpi=100)
    top = 0.10; bot = 0.07
    ax = fig.add_axes([0.10, bot, 0.26, 1 - top - bot - 0.03])
    ax.imshow(np.linspace(1, 0, 256)[:, None], aspect="auto", cmap=cmap, extent=[0, 1, lo, hi]); ax.set_xticks([])
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=13); ax.yaxis.tick_right(); ax.set_ylim(lo, hi)
    fig.text(0.05, 0.985, f"{title}\n({unit})", fontsize=12.5, va="top", ha="left", linespacing=1.3)
    fig.text(0.05, 0.005, note, fontsize=9, va="bottom", ha="left", color="#444")
    fig.canvas.draw(); im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert("RGB"); plt.close(fig); return im


def logbar(title, unit, lo, hi, cm, note):
    ks = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1)); st = max(1, len(ks) // 8)
    ks = ks[::st]
    return bar(title, unit, lo, hi, cm, ks, [f"$10^{{{k}}}$" for k in ks], note)


B_el = logbar("탄성 에너지 밀도", "J/m³, 로그", J["e_lo"], J["e_hi"], "turbo", "두 판·전 프레임\n공통 고정")
B_ke = logbar("운동 에너지", "J/입자, 로그", J["k_lo"], J["k_hi"], "turbo", "두 판·전 프레임\n공통 고정")
rt = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1]
# 발산형 색표: 1 이 흰색 (0.3~1 구간과 1~1.1 구간을 각각 반씩) -> 그림용 y 축은 색 위치(0~1)로 두고 눈금만 값으로
posf = lambda v: 0.5 * (v - 0.3) / 0.7 if v < 1 else 0.5 + 0.5 * (v - 1) / 0.1
B_r = bar("칸 높이 비 r", "h끝/h시작", 0, 1, "RdBu", [posf(v) for v in rt], [f"{v:g}" for v in rt], "0.3~1.1 고정\n회색: 칸 없음")
BARS = {"en": B_el, "ke": B_ke, "r": B_r}
f1, f2, f3 = ImageFont.truetype(FB, 30), ImageFont.truetype(FB, 24), ImageFont.truetype(FR, 22)
cols = [("plain", "일반 3DGS"), ("en", "탄성 에너지 밀도"), ("ke", "운동 에너지"), ("r", "칸 높이 비")]
rows = [("h9", "우리 H9"), ("gt", "정답")]
os.makedirs(f"{TMP}/out", exist_ok=True)
for t in range(T):
    im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
    for ri, (rk, rn) in enumerate(rows):
        y = ri * ROWH
        dr.rectangle([0, y, W, y + RB - 4], fill=(232, 236, 244))
        dr.text((14, y + 5), rn, fill="black", font=f1)
        for ci, (ck, cn) in enumerate(cols):
            x = ci * CW + max(0, ci - 1) * BW
            yi = y + RB + HEAD
            dr.text((x + 10, y + RB + 2), cn, fill="black", font=f2)
            if ck == "r" and rk == "gt":
                dr.rectangle([x, yi, x + CW - 1, yi + CH - 1], fill=(225, 225, 225))
                dr.text((x + 10, y + RB + 31), f"프레임 {t}", fill=(200, 0, 0), font=f3)
                dr.text((x + CW // 2 - 55, yi + CH // 2 - 30), "해당 없음", fill=(60, 60, 60), font=f1)
                dr.text((x + CW // 2 - 105, yi + CH // 2 + 12), "(정답에는 격자가 없음)", fill=(90, 90, 90), font=f3)
                continue
            im.paste(Image.open(fr[f"{rk}_{ck}"][t]).convert("RGB").resize((CW, CH), Image.LANCZOS), (x, yi))
            if ck == "plain": val = ""
            elif ck == "en": val = f" · 합 {J['E_el_' + ('pred' if rk == 'h9' else 'gt')][t]:.3g} J"
            elif ck == "ke": val = f" · 합 {J['E_ke_' + ('pred' if rk == 'h9' else 'gt')][t]:.3g} J"
            else: val = f" · 중앙값 {J['r_med'][t]:.3f}"
            dr.text((x + 10, y + RB + 31), f"프레임 {t}{val}", fill=(200, 0, 0), font=f3)
    for ci, ck in ((1, "en"), (2, "ke"), (3, "r")):
        im.paste(BARS[ck], ((ci + 1) * CW + (ci - 1) * BW, 0))
    im.save(f"{TMP}/out/{t:04d}.png")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "10", "-i", f"{TMP}/out/%04d.png", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                "-c:v", "libx264", "-crf", "18", "-movflags", "+faststart", out], check=True)
print("저장", out, flush=True)
