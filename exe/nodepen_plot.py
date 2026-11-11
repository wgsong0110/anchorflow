"""절 CM111 nodepen 그래프 (로컬, 작은 json 만): python exe/nodepen_plot.py <nodepen_*.json 디렉토리> <출력 png>"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf", "/home/wgsong/.fonts/NotoSansCJKkr-Bold.otf"):
    fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK KR"; plt.rcParams["axes.unicode_minus"] = False
D = {n: json.load(open(f"{sys.argv[1]}/nodepen_{n}.json")) for n in ("nA1", "nA2", "nH1")}
C = {"nA1": "#d55e00", "nA2": "#cc79a7", "nH1": "#1f77b4"}
L = {"nA1": "수렴 판 1 (50 프레임)", "nA2": "수렴 판 2 (f18 발산)", "nH1": "H8 (50 프레임)"}
dh, deep = D["nA1"]["dh_mm"], D["nA1"]["deep_mm"]
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
a = ax[0, 0]
for n, d in D.items():
    r = d["rows"]; f = [x["f"] for x in r]
    a.plot(f, [x["n_cross_C"] for x in r], color=C[n], lw=2, label=L[n])
    a.plot(f, [x["n_deep_E"] for x in r], color=C[n], lw=1.2, ls="--")
a.set_yscale("symlog", linthresh=10); a.set_xlabel("프레임"); a.set_ylabel("노드 수 (개, 로그)"); a.grid(alpha=.25)
a.set_title("(가) 바닥 밑으로 내려간 노드 (실선: 걸음 시작 바닥 위 -> 끝 바닥 아래,\n점선: 깊이 > 1.5 칸 = %.0f mm)" % deep, fontsize=10); a.legend(fontsize=8)
a = ax[0, 1]
for n, d in D.items():
    r = d["rows"]; f = [x["f"] for x in r]
    a.plot(f, [max(x["depth_max_C_mm"], 0.5) for x in r], color=C[n], lw=2, label=L[n])
a.axhline(dh, color="gray", ls=":"); a.text(1, dh * 1.1, "장벽 두께 $\\hat{d}$ = %.0f mm" % dh, color="gray", fontsize=8)
a.axhline(deep, color="gray", ls="--", lw=.8); a.text(1, deep * 1.1, "1.5 칸 = %.0f mm" % deep, color="gray", fontsize=8)
a.set_yscale("log"); a.set_xlabel("프레임"); a.set_ylabel("내려간 노드의 바닥 밑 최대 깊이 (mm, 로그)"); a.grid(alpha=.25)
a.set_title("(나) 내려간 노드가 얼마나 깊이 들어갔나", fontsize=10)
# (다) 튄 입자 중 받침에 바닥 밑 노드가 있던 비율 (직전 포함 3 걸음 창) vs 같은 프레임·높이 구간 튀지 않은 입자로 기대한 비율
a = ax[1, 0]
rows = [("nA1", "cohort_newfar", "새 이탈"), ("nA2", "cohort_newfar", "새 이탈"), ("nA1", "cohort_jump3", "급증 > 3 m/s"), ("nA2", "cohort_jump3", "급증 > 3 m/s"), ("nA1", "cohort_jump1", "급증 > 1 m/s"), ("nA2", "cohort_jump1", "급증 > 1 m/s"), ("nH1", "cohort_jump1", "급증 > 1 m/s")]
x = np.arange(len(rows)); w = .38
ob = [D[n][c]["W3"]["obs"]["E"] / max(D[n][c]["W3"]["n"], 1) * 100 for n, c, _ in rows]
ex = [D[n][c]["W3"]["exp"]["E"] / max(D[n][c]["W3"]["n"], 1) * 100 for n, c, _ in rows]
a.bar(x - w / 2, ob, w, color=[C[n] for n, _, _ in rows], label="튄 입자 (관측)")
a.bar(x + w / 2, ex, w, color="#bbbbbb", label="같은 프레임·높이의 튀지 않은 입자로 기대")
for i, (n, c, t) in enumerate(rows): a.text(i - w / 2, ob[i] + 1, "%d/%d" % (D[n][c]["W3"]["obs"]["E"], D[n][c]["W3"]["n"]), ha="center", fontsize=7)
a.set_xticks(x); a.set_xticklabels(["%s\n%s" % (L[n].split(" (")[0], t) for n, _, t in rows], fontsize=7)
a.set_ylabel("받침에 바닥 밑 노드가 있던 비율 (%)"); a.grid(alpha=.25, axis="y"); a.legend(fontsize=8)
a.set_title("(다) 튄 입자의 받침 노드 (튀는 걸음 포함 3 걸음 안)", fontsize=10)
# (라) 시점: 튀는 걸음(W1) vs 그 앞 1~3 걸음(P3)
a = ax[1, 1]
rows2 = [("nA1", "cohort_newfar", "새 이탈"), ("nA2", "cohort_newfar", "새 이탈"), ("nA2", "cohort_jump3", "급증 > 3 m/s")]
x = np.arange(len(rows2)); w = .26
for k, (kind, lab, hat) in enumerate((("W1", "그 걸음", ""), ("P3", "앞 1~3 걸음", "//"))):
    v = [D[n][c][kind]["obs"]["E"] / max(D[n][c][kind]["n"], 1) * 100 for n, c, _ in rows2]
    a.bar(x + (k - .5) * w, v, w, color=[C[n] for n, _, _ in rows2], hatch=hat, edgecolor="white", label=lab)
for k, kind in enumerate(("W1", "P3")):
    for i, (n, c, t) in enumerate(rows2):
        e = D[n][c][kind]["exp"]["E"] / max(D[n][c][kind]["n"], 1) * 100
        a.plot([i + (k - .5) * w - w / 2, i + (k - .5) * w + w / 2], [e, e], color="k", lw=1.5)
a.plot([], [], color="k", lw=1.5, label="기대 (검은 선)")
a.set_xticks(x); a.set_xticklabels(["%s\n%s" % (L[n].split(" (")[0], t) for n, _, t in rows2], fontsize=8)
a.set_ylabel("받침에 바닥 밑 노드가 있던 비율 (%)"); a.grid(alpha=.25, axis="y"); a.legend(fontsize=8)
a.set_title("(라) 튀기 전에 이미 있었나: 튀는 걸음 vs 앞 1~3 걸음", fontsize=10)
fig.suptitle("wolf: 튄 입자의 받침 노드가 바닥 (z = 0.48 m) 밑으로 내려갔었나 (노드 덤프, 결과 불변)", fontsize=12)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130)
