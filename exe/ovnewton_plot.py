"""ovnewton (출력만 최적화 + 뉴턴 걸음 프레임당 40 회) records.json -> 프레임별 E/E*·위치 오차·|p| 그래프 (글자 한국어).

  python exe/ovnewton_plot.py --rec L000=DIR/records.json --rec 0004=DIR/records.json --out PNG
  망 nt 비교선: --nt L000=DIR/records.json (있으면 같은 색 점선)
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf",):
    try:
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--rec", action="append", required=True)
ap.add_argument("--nt", action="append", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()
COL = {"L000": "#2a78d6", "0004": "#eb6834"}
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
nts = dict(s.split("=", 1) for s in a.nt)
for s in a.rec:
    T, p = s.split("=", 1)
    R = json.load(open(p)); c = COL.get(T, "#1baf7a")
    ok = [r for r in R if not r.get("nan")]
    fr = [r["t"] + 1 for r in ok]
    ax[0].plot(fr, [r["ratio_E_Estar"] for r in ok], color=c, lw=2, label=f"{T} 출력 뉴턴 40걸음")
    ax[1].plot(fr, [100 * r["pos_err_model"] for r in ok], color=c, lw=2, label=f"{T} 출력 뉴턴 40걸음")
    ax[1].plot(fr, [100 * r["ref_err"] for r in ok if "ref_err" in r], color=c, lw=1, alpha=.55, ls="--",
               label=f"{T} 출력만 최적화 L-BFGS 500")
    xs, ys = [], []
    for r in R:
        for q in r["newton"]["log"]:
            xs.append(r["t"] + 1 + q["k"] / max(len(r["newton"]["log"]), 1)); ys.append(q.get("p_norm", float("nan")))
    ax[2].plot(xs, ys, color=c, lw=1, label=f"{T}")
    for r in R:
        if r.get("nan"):
            for k_ in range(3):
                ax[k_].axvline(r["t"] + 1, color=c, lw=1, ls=":")
    if T in nts:
        Q = [r for r in json.load(open(nts[T])) if r.get("ratio_E_Estar") is not None]
        ax[0].plot([r["t"] + 1 for r in Q], [r["ratio_E_Estar"] for r in Q], color=c, lw=1, ls=":", label=f"{T} 망 nt")
ax[0].set_yscale("log"); ax[0].set_title("프레임별 E/E* (E* = 같은 상태 출력만 최적화 L-BFGS 500)")
ax[0].axhline(1, color="#888", lw=.8)
ax[1].set_title("프레임별 위치 오차 (정답 대비, %)")
ax[2].set_yscale("log"); ax[2].set_title("뉴턴 걸음별 |p| / EXT (프레임 안 40 걸음)")
for x_ in ax:
    x_.set_xlabel("프레임"); x_.grid(alpha=.25); x_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(a.out, dpi=130)
print("[저장]", a.out)
