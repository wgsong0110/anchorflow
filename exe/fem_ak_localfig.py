"""절 AK.15 그림: 최대 압축 뒤 15 프레임 평균 입자 탄성 (J) 의 몸 좌표 높이 띠 x 수평 부위 지도 -- 궤적마다 우리 / 정답 / 차 (모든 칸 공통 색 범위). 사용: python exe/fem_ak_localfig.py <출력 png>"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                       # noqa: E402
import numpy as np                                                                    # noqa: E402
from matplotlib import font_manager                                                   # noqa: E402

font_manager.fontManager.addfont(os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf"))
plt.rcParams["font.family"] = "Noto Sans CJK KR"
L = os.path.expanduser("~/r2/result/anchorflow/lbours/fem_ak_lego")
TS = ["0001", "0003", "0016", "0017", "0018", "0005", "0013"]
D = {t: json.load(open(f"{L}/{t}_pqf3_z_F1/local.json")) for t in TS}
vmax = max(max(np.max(D[t]["map"]["ours"]), np.max(D[t]["map"]["gt"])) for t in TS)
dmax = max(np.max(np.abs(np.array(D[t]["map"]["ours"]) - np.array(D[t]["map"]["gt"]))) for t in TS)
fig, ax = plt.subplots(3, len(TS), figsize=(2.2 * len(TS), 7.5), constrained_layout=True)
REG = ["-x-y", "-x+y", "+x-y", "+x+y"]
for j, t in enumerate(TS):
    o, g = np.array(D[t]["map"]["ours"]), np.array(D[t]["map"]["gt"])
    for i, (M, cm, lo, hi, nm) in enumerate(((o, "viridis", 0, vmax, "우리"), (g, "viridis", 0, vmax, "정답"), (o - g, "RdBu_r", -dmax, dmax, "우리 − 정답"))):
        im = ax[i, j].imshow(M, origin="lower", cmap=cm, vmin=lo, vmax=hi, aspect="auto")
        ax[i, j].set_xticks(range(4))
        ax[i, j].set_xticklabels(REG, fontsize=7)
        ax[i, j].set_yticks(range(8))
        ax[i, j].set_yticklabels([f"{k + 1}" for k in range(8)], fontsize=7)
        if j == 0:
            ax[i, j].set_ylabel(f"{nm}\n몸 높이 띠 (1 = 바닥 쪽)", fontsize=8)
        if i == 0:
            ax[i, j].set_title(f"{t}{' (대조)' if t in ('0005', '0013') else ''}", fontsize=9)
        if j == len(TS) - 1:
            fig.colorbar(im, ax=ax[i, :], shrink=0.8, label="입자 탄성 (J)" if i < 2 else "차 (J)")
fig.suptitle("최대 압축 뒤 15 프레임 평균 입자 탄성 V·ψ 의 몸 좌표 분포 (부위 = 몸 x·y 반쪽)", fontsize=11)
fig.savefig(sys.argv[1], dpi=130)
print("[저장]", sys.argv[1])
