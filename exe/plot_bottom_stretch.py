"""바닥 쪽 입자의 변형구배가 실제로 늘어나는지 본다 (PG 와 나란히).

tr(C) = 주신축의 제곱합이라 변형이 없으면 3 이고, 늘어나면 커진다. 덤프의
`fscal` 은 (det F, tr C, ‖E‖_F) 이고, PG 는 궤적이 자기 F 를 들고 있으므로
같은 양을 직접 계산해 맞댈 수 있다.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
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
ap.add_argument("--dump", required=True)
ap.add_argument("--traj", required=True, help="PG 궤적 (.pt, F 를 들고 있다)")
ap.add_argument("--frac", type=float, default=0.1, help="아래쪽 몇 할을 바닥군으로")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


D = L(a.dump)
X0 = np.asarray(D["x0"], dtype=np.float32)
t0 = int(D["t0"])
fs = D.get("fscal")
if fs is None:
    raise SystemExit("덤프에 fscal 이 없다 -- --ov_roll 로 다시 덤프할 것")
fs = np.asarray(fs)                                  # [T,N,3]
T = fs.shape[0]

z = X0[:, 2]
zq = np.quantile(z, a.frac)
bot = z <= zq
top = z >= np.quantile(z, 1 - a.frac)
mid = ~bot & ~top
print(f"[무리] 바닥군 {int(bot.sum())} (z<={zq:.4f}), 중간 {int(mid.sum())}, "
      f"위 {int(top.sum())}", flush=True)

d = L(a.traj)
FP = d["F"].float()                                   # [Tt,N,3,3]
def trC_pg(t):
    F = FP[min(t, FP.shape[0] - 1)]
    C = F.transpose(-1, -2) @ F
    return C.diagonal(dim1=-2, dim2=-1).sum(-1).numpy()

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4), dpi=120)
names = [("바닥군", bot, "tab:red"), ("중간", mid, "tab:gray"),
         ("위", top, "tab:blue")]
ts = np.arange(T)
for q, (key, lab) in zip(ax, [(1, "tr(C)   (3 = 변형 없음)"),
                              (0, "det(F)  (1 = 부피 보존)"),
                              (2, "‖E‖_F  (0 = 변형 없음)")]):
    for nm, m, c in names:
        q.plot(ts, [np.median(fs[t][m, key]) for t in ts], "-", color=c,
               label=f"출력만 최적화 {nm}")
    for nm, m, c in names:
        q.plot(ts, [np.median(trC_pg(t0 + t + 1)[m]) for t in ts], "--",
               color=c, alpha=.7, label=f"PG {nm}") if key == 1 else None
    q.set_xlabel("프레임"); q.set_title(lab, fontsize=11); q.grid(alpha=.3)
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)
for nm, m, _ in names:
    print(f"  {nm:4} tr(C) 중앙  처음 {np.median(fs[0][m,1]):.4f} -> 끝 "
          f"{np.median(fs[-1][m,1]):.4f}   |  PG {np.median(trC_pg(t0+1)[m]):.4f}"
          f" -> {np.median(trC_pg(t0+T)[m]):.4f}", flush=True)
