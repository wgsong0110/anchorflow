"""절 CM110 floorpen 그래프 (로컬, 작은 json 만): python exe/floorpen_plot.py <floorpen 디렉토리> <출력 png>"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf", "/home/wgsong/.fonts/NotoSansCJKkr-Bold.otf"):
    fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK KR"; plt.rcParams["axes.unicode_minus"] = False
D = {n: json.load(open(f"{sys.argv[1]}/floorpen_{n}.json")) for n in ("h8_1", "aC_1")}
C = {"h8_1": "#1f77b4", "aC_1": "#d55e00"}; L = {"h8_1": "H8 (h8_1)", "aC_1": "수렴 판 (aC_1, 27 프레임 발산)"}
dh = D["aC_1"]["dh"] * 1e3
fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
for n, d in D.items():
    r = d["rows"]; f = [x["f"] for x in r]
    ax[0].plot(f, [x["n_under"] for x in r], color=C[n], label=L[n], lw=2)
    ax[1].plot(f, [x["depth_max_mm"] for x in r], color=C[n], lw=2)
    ax[2].plot(f, [x.get("n_newfar", 0) for x in r], color=C[n], lw=2)
    ax[3].plot(f, [x.get("n_jump3") or 0 for x in r], color=C[n], lw=2, label=f"{L[n]}: 속도 급증 > 3 m/s")
    ax[3].plot(f, [x.get("n_jump1") or 0 for x in r], color=C[n], lw=1.2, ls="--", label=f"{L[n]}: 속도 급증 > 1 m/s")
ax[1].axhline(dh, color="gray", ls=":"); ax[1].text(1, dh * 1.04, f"장벽 두께 $\hat{{d}}$ = {dh:.0f} mm", color="gray", fontsize=9)
for a in ax: a.axvline(27, color=C["aC_1"], ls=":", alpha=.6)
ax[0].text(26.8, ax[0].get_ylim()[1] * .9, "aC_1 발산 f27", color=C["aC_1"], ha="right", fontsize=9)
for a, t in zip(ax, ["바닥 밑 입자 수 (개)", "바닥 밑 최대 깊이 (mm)", "새 이탈 입자: 몸 상자 4칸 밖 (개)", "속도 급증 입자 (개, 로그)"]):
    a.set_ylabel(t, fontsize=9); a.grid(alpha=.25)
ax[0].legend(fontsize=9, loc="upper left"); ax[3].legend(fontsize=8, loc="upper right"); ax[3].set_yscale("symlog", linthresh=1); ax[3].set_ylim(0, 3000)
ax[3].set_xlabel("프레임")
fig.suptitle("wolf 프레임별 바닥 밑 입자와 튄 입자 (바닥 z = 0.48 m, 장벽 두께 $\hat{d}$ = 5 mm)", fontsize=11)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130)
