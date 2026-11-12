"""wolf H9 증분 포텐셜 색 영상 합성 (PIL + ffmpeg): 한 줄 다섯 칸 일반 3DGS | 합 | 관성 | 탄성 | 장벽+정규화, 칸마다 그 항의 전체 합과 프레임 번호, 칸 아래 색 막대 (전 프레임 고정 범위).
사용: python exe/wolfip_compose.py <ip.json> <vid 디렉토리 (plain sum in el rb .mp4)> <출력 mp4> <임시 폴더>"""
import glob, json, os, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fj, vd, out, TMP = sys.argv[1:5]
os.makedirs(TMP, exist_ok=True)
FF = shutil.which("ffmpeg")
if not FF:
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
FD = os.environ.get("WIP_FONT_DIR", os.path.expanduser("~/.fonts"))
FB, FR = f"{FD}/NotoSansCJKkr-Bold.otf", f"{FD}/NotoSansCJKkr-Regular.otf"
fm.fontManager.addfont(FR); plt.rcParams["font.family"] = "Noto Sans CJK KR"; plt.rcParams["axes.unicode_minus"] = False
J = json.load(open(fj)); R = J["range"]; T = J["T"]
CW, HEAD, BARH, TOP = 460, 74, 78, 50
cols = [("plain", "일반 3DGS", None), ("sum", "합 (관성+탄성+중력+장벽+정규화)", "E_sum"), ("in", "관성 ½m|y − y^|²/τ²", "E_in"),
        ("el", "탄성 V_q ψ(F_q)", "E_el"), ("rb", "장벽 + 정규화 (고스트 질량)", "E_rb")]
fr = {}
for k, _, _ in cols:
    d = f"{TMP}/f_{k}"; os.makedirs(d, exist_ok=True)
    for p in glob.glob(f"{d}/*.png"): os.remove(p)
    subprocess.run([FF, "-v", "error", "-y", "-i", f"{vd}/{k}.mp4", f"{d}/%04d.png"], check=True)
    fr[k] = sorted(glob.glob(f"{d}/*.png"))
    assert len(fr[k]) == T, (k, len(fr[k]), T)
# 같은 자르기 상자 (일반 3DGS 전 프레임에서 배경과 다른 픽셀의 합집합 + 여백) 를 모든 칸·프레임에
BOX = None
for p_ in fr["plain"]:
    a_ = np.asarray(Image.open(p_).convert("RGB")).astype(np.int16)
    bg = a_[0, 0]
    m_ = np.abs(a_ - bg).max(2) > 12
    if m_.any():
        ys_, xs_ = np.nonzero(m_); b_ = (xs_.min(), ys_.min(), xs_.max() + 1, ys_.max() + 1)
        BOX = b_ if BOX is None else (min(BOX[0], b_[0]), min(BOX[1], b_[1]), max(BOX[2], b_[2]), max(BOX[3], b_[3]))
w0, h0 = Image.open(fr["plain"][0]).size
if BOX is None:
    BOX = (0, 0, w0, h0)
mx_, my_ = int(0.06 * (BOX[2] - BOX[0])) + 4, int(0.06 * (BOX[3] - BOX[1])) + 4
BOX = (max(0, BOX[0] - mx_), max(0, BOX[1] - my_), min(w0, BOX[2] + mx_), min(h0, BOX[3] + my_))
bw_, bh_ = BOX[2] - BOX[0], BOX[3] - BOX[1]
CH = min(560, round(CW * bh_ / bw_)); CH -= CH % 2
IW = min(CW, round(CH * bw_ / bh_))
print(f"자르기 {BOX} (원래 {w0}x{h0})", flush=True)
NOTE = ["입자로 나누기 -- 노드 항 (관성·중력): 입자 B-스플라인 가중 w_pn 비례 (노드마다 Σ_p w_pn 으로 나눔) · 가우스 점 항 (탄성): 그 팔분 칸 입자에 부피 비례",
        "(입자 없는 팔분 칸 몫은 같은 칸 입자에) · 장벽: 표본점 (팔분 칸) 입자에 부피 비례 · 면 항 (정규화): 양쪽 칸에 반씩 나눈 뒤 칸 안 입자에 부피 비례.",
        "값 = 프레임 (마지막 걸음) 끝 해. y = HHT 풀이 변수, y^ = 관성 예측, 중력 = −m g·(걸음 끝 − 걸음 시작 노드 변위), 탄성 = Hencky ψ (커밋 뒤 F_q).",
        "켜진 정규화 항: P1X 면 고스트 질량 ½ (k_M/τ²) (y − y^)^T G (y − y^) 만 (wolf κ 0 이라 P1X 강성 k = 0, 가벼움 벌점 0; 확장 B-스플라인은 제약이라 에너지에서 뺌).",
        "색 범위는 전 프레임 공통 고정 (프레임별 정규화 없음). 프레임 0 = 걸음 전 (회색)."]
_rc = [r for r in J["rec"] if r]
if _rc:
    _k = max(abs(r["face"]["k"]) for r in _rc); _kM = _rc[-1]["face"]["kM"]
    _Ek = sum(abs(r["face"]["E_k"]) for r in _rc); _Em = sum(r["face"]["E_m"] for r in _rc)
    NOTE.append(f"실측 (이 판 기록): 강성 k 최대 {_k:.3g}, 강성 몫 |합| 전 프레임 {_Ek:.3g} J · 고스트 질량 k_M {_kM:.3g} kg, 질량 몫 합 전 프레임 {_Em:.3g} J · "
                f"P1X 면 수 프레임 1 {_rc[0]['face']['nface']} -> 프레임 {_rc[-1]['f']} {_rc[-1]['face']['nface']}, 면 가중 " + ("g5w" if _rc[-1]['face']['fwp'] else "없음 (모두 1; κ 0 이라 lightg3 꺼짐)") + ".")
NH = 30 * len(NOTE) + 16
W = 5 * CW; H = TOP + HEAD + CH + BARH + NH
print(f"프레임 {T} 칸 {CW}x{CH} 전체 {W}x{H}", flush=True)


def hbar(cmap, ticks, labels, title):
    fig = plt.figure(figsize=(CW / 100, BARH / 100), dpi=100)
    ax = fig.add_axes([0.06, 0.52, 0.88, 0.22])
    ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap, extent=[0, 1, 0, 1]); ax.set_yticks([])
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=10); ax.set_xlim(0, 1)
    fig.text(0.5, 0.80, title, fontsize=11, ha="center", va="bottom")
    fig.canvas.draw(); im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert("RGB"); plt.close(fig); return im


def logbar(lo, hi, title):
    ks = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1)); st = max(1, int(np.ceil(len(ks) / 7))); ks = ks[::st]
    return hbar("YlOrRd", [(k - lo) / (hi - lo) for k in ks], [f"$10^{{{k}}}$" for k in ks], title + " (J/입자, 로그; 0 이하 = 맨 왼쪽 색)")


lim, lt = R["sum_lim"], R["sum_lt"]
sl = lambda v: 0.5 + 0.5 * np.sign(v) * np.log10(1 + abs(v) / lt) / np.log10(1 + lim / lt)
kt = int(np.floor(np.log10(lim))); kk = [kt - 3, kt - 1, kt]
tv = [-10.0 ** k for k in kk[::-1]] + [0.0] + [10.0 ** k for k in kk]
BARS = {"sum": hbar("RdBu_r", [sl(v) for v in tv], ["0" if v == 0 else (f"$-10^{{{int(round(np.log10(-v)))}}}$" if v < 0 else f"$10^{{{int(round(np.log10(v)))}}}$") for v in tv],
                    f"합 (J/입자, 대칭 로그, 0 흰색, ±{lim:.2g} 고정)"),
        "in": logbar(*R["in"], "관성"), "el": logbar(*R["el"], "탄성"), "rb": logbar(*R["rb"], "장벽+정규화")}
f1, f2, f3, f4 = ImageFont.truetype(FB, 28), ImageFont.truetype(FB, 21), ImageFont.truetype(FR, 21), ImageFont.truetype(FR, 18)
os.makedirs(f"{TMP}/out", exist_ok=True)
for p in glob.glob(f"{TMP}/out/*.png"): os.remove(p)
for t in range(T):
    im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W, TOP - 6], fill=(232, 236, 244))
    dr.text((14, 8), f"wolf H9 증분 포텐셜 항별 (입자에 나눈 값)   ·   프레임 {t}", fill="black", font=f1)
    for ci, (ck, cn, ek) in enumerate(cols):
        x = ci * CW
        dr.text((x + 10, TOP + 4), cn, fill="black", font=f2)
        if ek is None:
            val = f"프레임 {t}"
        elif t == 0:
            val = f"프레임 0 · 걸음 전"
        else:
            val = f"프레임 {t} · 전체 합 {J[ek][t]:.4g} J"
        dr.text((x + 10, TOP + 38), val, fill=(200, 0, 0), font=f3)
        im.paste(Image.open(fr[ck][t]).convert("RGB").crop(BOX).resize((IW, CH), Image.LANCZOS), (x + (CW - IW) // 2, TOP + HEAD))
        if ck in BARS:
            im.paste(BARS[ck], (x, TOP + HEAD + CH))
    for i, ln in enumerate(NOTE):
        dr.text((14, TOP + HEAD + CH + BARH + 8 + 30 * i), ln, fill=(60, 60, 60), font=f4)
    im.save(f"{TMP}/out/{t:04d}.png")
subprocess.run([FF, "-v", "error", "-y", "-framerate", "10", "-i", f"{TMP}/out/%04d.png", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                "-c:v", "libx264", "-crf", "18", "-movflags", "+faststart", out], check=True)
r = subprocess.run([FF, "-v", "error", "-i", out, "-f", "null", "-"], capture_output=True, text=True)
print("전체 디코드", "정상" if (r.returncode == 0 and not r.stderr.strip()) else f"오류 rc={r.returncode} {r.stderr[:300]}", flush=True)
print("저장", out, W, H, flush=True)
