"""smin.json -> 프레임별 전체 탄성 에너지 PNG (한국어). 사용: python exe/smin_png.py <smin.json> <out.png>"""
import json, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont(os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")); plt.rcParams["font.family"] = "Noto Sans CJK KR"
d = json.load(open(sys.argv[1])); x = range(len(d["pred"]))
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(x, [max(r["E"], 1e-3) for r in d["pred"]], color="#2a6fdb", lw=2, label="예측 (H8)")
ax.plot(x, [max(r["E"], 1e-3) for r in d["gt"]], color="#777", lw=2, label="정답 (정답 h5 의 F, 거의 직교라 0 근처)")
ax.set_yscale("log"); ax.set_xlabel("프레임"); ax.set_ylabel("전체 탄성 에너지 Σ V_p ψ(F_p) [J]")
ax.set_title("wolf H8 (=H9) 프레임별 전체 탄성 에너지 (Hencky, E=%.3g Pa, ν=%s; 0 은 1e-3 으로 표시)" % (d["E"], d["nu"]), fontsize=10)
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130)
