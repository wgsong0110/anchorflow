"""AF_OV_CURVE 로 남긴 프레임별 증분 포텐셜 곡선을 그린다."""
from __future__ import annotations
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import os as _os
for _p in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf",
           _os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")):
    try: fm.fontManager.addfont(_p)
    except Exception: pass
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--curve", nargs="+", required=True,
                help="json 경로. 'name=경로' 로 이름을 붙일 수 있다")
ap.add_argument("--out", required=True)
ap.add_argument("--terms", action="store_true",
                help="첫 곡선 파일을 **항별로** 나눠 그린다 (전체·관성·탄성·접촉)")
a = ap.parse_args()

if a.terms:
    name, _, path = a.curve[0].partition("=")
    if not path:
        name, path = _os.path.basename(a.curve[0]).replace(".json", ""), a.curve[0]
    D = json.load(open(path))
    fr = D["frames"]
    KS = [("E", "전체 E (정규화)"), ("in", "관성 (원시)"),
          ("el", "탄성 (원시)"), ("bc", "접촉 (원시)")]
    fig, ax = plt.subplots(1, 4, figsize=(19.0, 4.3), dpi=120)
    cm = plt.get_cmap("viridis")
    for k, (key, lab) in enumerate(KS):
        q = ax[k]
        for i, f in enumerate(fr):
            y = np.asarray(f.get(key, []), dtype=float)
            if y.size == 0:
                continue
            q.plot(np.arange(1, len(y) + 1), np.abs(y), lw=1.0,
                   color=cm(i / max(len(fr) - 1, 1)))
        q.set_xscale("log")
        if np.nanmax([np.max(np.abs(f.get(key, [0]))) for f in fr]) > 0:
            q.set_yscale("log")
        q.set_xlabel("반복 (함수 호출 수)"); q.set_title(lab, fontsize=11)
        q.grid(alpha=.3, which="both")
    ax[0].set_ylabel(f"{name}   (색: 어두운 쪽이 앞 프레임)")
    fig.tight_layout(); fig.savefig(a.out)
    print(f"[저장] {a.out}", flush=True)
    raise SystemExit(0)

fig, ax = plt.subplots(1, len(a.curve), figsize=(5.4 * len(a.curve), 4.4),
                       dpi=120, squeeze=False)
for k, spec in enumerate(a.curve):
    name, _, path = spec.partition("=")
    if not path:
        name, path = _os.path.basename(spec).replace(".json", ""), spec
    D = json.load(open(path))
    fr = D["frames"]
    cm = plt.get_cmap("viridis")
    q = ax[0][k]
    for i, f in enumerate(fr):
        E = np.asarray(f["E"], dtype=float)
        q.plot(np.arange(1, len(E) + 1), E, lw=1.0,
               color=cm(i / max(len(fr) - 1, 1)))
    q.set_xscale("log"); q.set_yscale("log")
    q.set_xlabel("반복 (함수 호출 수)")
    q.set_ylabel("증분 포텐셜 E")
    q.set_title(f"{name}  (프레임 {len(fr)} 개, 색: 이름 -> 나중 프레임)",
                fontsize=10)
    q.grid(alpha=.3, which="both")
    # 프레임마다 처음/끝을 적어 둔다
    b = np.median([f["E"][0] for f in fr])
    e = np.median([f["E"][-1] for f in fr])
    q.text(.02, .03, f"중앙값 {b:.3e} -> {e:.3e}  ({100*(1-e/b):.1f}% 감소)",
           transform=q.transAxes, fontsize=9)
fig.tight_layout()
fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)
