"""목적함수 다섯 비교 (exe/pf3obj.sh) 프레임별 곡선 -> PNG.

  python exe/pf3obj_plot.py <내려받은 lbours/pf3obj 디렉토리> <출력.png> [목적함수 목록, 쉼표 구분 (기본 전부)]
열 = 궤적 (L000, 0004), 행:
  1) 위치 오차 (정답 대비, %) -- 네 목적함수 모델 롤아웃 + 출력만 최적화
  2) E/E* -- 같은 모델 상태에서 망 출력의 증분 포텐셜 / 출력만 최적화 최솟값 (로그 축)
  3) 목적함수 끝/처음 (프레임 안 최적화로 줄어든 비, 로그 축)
  4) 모델 상태와 출력만 최적화 상태의 거리 (%)
모든 칸 공통 축 범위 (행별로 두 궤적 같은 범위).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
for _p in (os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf"),):
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
U, OUT = sys.argv[1], sys.argv[2]
TR = ("L000", "0004")
KS = [("ip", "증분 포텐셜", "#2a78d6"), ("nt", "뉴턴 목표 L2", "#8a4fc9"),
      ("l2ov", "출력만 최적화 위치 L2", "#e08a1e"), ("l2gt", "정답 위치 L2", "#2f9e57"), ("nl2", "모델 상태 출력만 최적화 노드 L2", "#c0392b")]
if len(sys.argv) > 3:
    _sel = sys.argv[3].split(",")
    KS = [x for x in KS if x[0] in _sel]
    assert len(KS) == len(_sel), f"모르는 목적함수: {_sel}"
R = {(k, T): json.load(open(f"{U}/{k}/{T}/records.json")) for k, _, _ in KS for T in TR}
fig, ax = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
rows = [("위치 오차 (정답 대비, %)", lambda q: 100 * q["pos_err_model"], False),
        ("E / E* (같은 모델 상태)", lambda q: q["ratio_E_Estar"], True),
        ("목적함수 끝 / 처음 (프레임 안)", lambda q: q["obj_last"] / q["obj_first"] if q["obj_first"] else np.nan, True),
        ("모델 상태 - 출력만 최적화 상태 거리 (%)", lambda q: 100 * q["dist_model_ref"], False)]
for j, T in enumerate(TR):
    for i, (yl, f, lg) in enumerate(rows):
        a = ax[i, j]
        for k, lab, c in KS:
            r = R[(k, T)]
            a.plot([q["t"] + 1 for q in r], [f(q) for q in r], color=c, lw=1.6, label=lab)
        if i == 0:
            r = R[("ip", T)]
            a.plot([q["t"] + 1 for q in r], [100 * q["ref_err"] for q in r], color="#555555", lw=1.4, ls="--", label="출력만 최적화")
        if i == 1:
            a.axhline(1.0, color="#999999", lw=0.8)
        if lg:
            a.set_yscale("log")
        a.set_ylabel(yl)
        a.grid(alpha=0.3)
        if i == 0:
            a.set_title(f"{T}" + (" (다섯 궤적 학습에 안 쓴 궤적)" if T == "0004" else ""))
    ax[-1, j].set_xlabel("프레임")
for i in range(len(rows)):
    lo = min(ax[i, 0].get_ylim()[0], ax[i, 1].get_ylim()[0]); hi = max(ax[i, 0].get_ylim()[1], ax[i, 1].get_ylim()[1])
    ax[i, 0].set_ylim(lo, hi); ax[i, 1].set_ylim(lo, hi)
ax[0, 0].legend(fontsize=9)
fig.suptitle(f"망 파라미터 프레임별 최적화 · 목적함수 {'하나 둘 셋 넷 다섯'.split()[len(KS) - 1]} (Adam 1e-3, 프레임당 2000 회, 새 망)")
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print("저장", OUT)
