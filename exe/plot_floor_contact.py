"""바닥에 닿는 순간부터 **맨 아래 셀**의 탄성 에너지와 세로 찌그러짐.

세로 찌그러짐은 sqrt(C_zz) 로 잰다 -- 처음에 z 를 향하던 섬유의 신축이라
1 보다 작으면 세로로 눌린 것이다. 탄성 에너지는 psi (에너지 밀도) 다.
접촉 시점은 바닥군 입자의 최저 z 가 바닥면에 닿는 첫 프레임으로 잡는다.
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
ap.add_argument("--traj", required=True)
ap.add_argument("--floor", type=float, required=True)
ap.add_argument("--frac", type=float, default=0.1)
ap.add_argument("--phase", type=int, nargs="+", default=[],
                help="손잡이 명령 구간 경계 프레임 (예: 10 28 = 10 부터 하강, "
                     "28 부터 상승)")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


D = L(a.dump)
X0 = np.asarray(D["x0"], dtype=np.float32)
P = D["pred"].float().numpy()
fs = np.asarray(D["fscal"])
if fs.shape[-1] < 5:
    raise SystemExit("덤프에 C_zz/psi 가 없다 -- 다시 덤프할 것")
T = fs.shape[0]
bot = X0[:, 2] <= np.quantile(X0[:, 2], a.frac)
print(f"[바닥군] {int(bot.sum())} 개 (초기 z <= "
      f"{np.quantile(X0[:,2], a.frac):.4f}), 바닥 z={a.floor}", flush=True)

# 접촉 시점: 바닥군 최저 z 가 바닥에 닿는 첫 프레임
zmin = np.array([P[t][bot, 2].min() for t in range(T)])
hit = int(np.argmax(zmin <= a.floor + 1e-4)) if (zmin <= a.floor + 1e-4).any() \
    else -1
print(f"[접촉] 프레임 {hit} (바닥군 최저 z {zmin.min():.4f})", flush=True)

d = L(a.traj)
FP = d["F"].float()
t0 = int(D["t0"])
pg_czz, pg_psi = [], []
for t in range(T):
    F = FP[min(t0 + t + 1, FP.shape[0] - 1)]
    C = (F.transpose(-1, -2) @ F)[:, 2, 2].numpy()
    pg_czz.append(np.median(C[bot]))
from anchorflow import phys_resid
for t in range(T):
    F = FP[min(t0 + t + 1, FP.shape[0] - 1)][bot]
    psi, _ = phys_resid.psi_of(F, d["cfg"], float(d["cfg"]["frame_dt"]))
    pg_psi.append(float(np.median(psi.numpy())))

ts = np.arange(T)
fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.4), dpi=120)
ax[0].plot(ts, [np.median(fs[t][bot, 4]) for t in ts], "-", color="tab:red",
           label="출력만 최적화")
ax[0].plot(ts, pg_psi, "--", color="tab:blue", label="PG MPM")
ax[0].set_yscale("log")
ax[0].set_xlabel("프레임"); ax[0].set_ylabel("탄성 에너지 밀도 psi (중앙값)")
ax[0].set_title("맨 아래 셀의 탄성 에너지", fontsize=11)
ax[1].plot(ts, [np.median(np.sqrt(fs[t][bot, 3])) for t in ts], "-",
           color="tab:red", label="출력만 최적화")
ax[1].plot(ts, np.sqrt(pg_czz), "--", color="tab:blue", label="PG MPM")
ax[1].axhline(1.0, color="0.5", lw=1)
ax[1].set_xlabel("프레임")
ax[1].set_ylabel("세로 신축 sqrt(C_zz)  (<1 이면 눌림)")
ax[1].set_title("맨 아래 셀의 세로 찌그러짐", fontsize=11)
_lab = ["하강 시작", "상승 시작", "구간"]
for q in ax:
    _y = q.get_ylim()[1]
    for _i, _p in enumerate(a.phase):
        q.axvline(_p, color="tab:green", ls="-.", lw=1.3, alpha=.8)
        q.text(_p, _y, " " + (_lab[_i] if _i < 2 else _lab[2]), va="top",
               fontsize=9, color="tab:green")
    if hit >= 0:
        q.axvline(hit, color="k", ls=":", lw=1.6)
        q.text(hit, _y, " 접촉", va="top", fontsize=9)
    q.grid(alpha=.3); q.legend(fontsize=9)
fig.tight_layout(); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)
print(f"  psi  중앙 접촉전 {np.median([np.median(fs[t][bot,4]) for t in range(max(hit,1))]):.3e}"
      f" -> 끝 {np.median(fs[-1][bot,4]):.3e}   PG 끝 {pg_psi[-1]:.3e}")
print(f"  sqrt(C_zz) 끝: 우리 {np.median(np.sqrt(fs[-1][bot,3])):.4f}  "
      f"PG {np.sqrt(pg_czz[-1]):.4f}")
