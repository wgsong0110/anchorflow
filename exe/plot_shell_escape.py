"""흘러내린 입자가 **바깥 껍질 셀** 출신인지 본다.

표면에 걸친 셀은 셀의 일부만 물체 안이라 입자가 적게 들고, 그만큼 탄성 기여가
약하다. 그 약한 셀의 입자가 먼저 떨어져 나가는지 확인한다.

껍질 깊이: 프레임 0 의 점유 셀 집합에서 **빈 셀에 닿은 점유 셀**을 깊이 1 로
두고, 그것을 벗겨내며 깊이 2, 3, ... 을 매긴다 (침식). 입자는 자기 셀의 깊이를
물려받는다.

흘러내림: 마지막 프레임에 중심 입자로부터 반지름 밖이면서 **중심보다 아래**.

비교는 개수가 아니라 **깊이별 비율**로 한다 -- 표면 셀은 애초에 입자가 적어
개수로 세면 반드시 오해한다.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn.functional as Fn
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

from anchorflow import simplex as SX

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True, help="롤아웃 덤프 (.pt)")
ap.add_argument("--n_nodes", type=float, required=True,
                help="그 실행이 쓴 격자 (깊이를 같은 격자에서 매긴다)")
ap.add_argument("--hz_ratio", type=float, default=SX.HZ_CUBE)
ap.add_argument("--r", type=float, default=0.0)
ap.add_argument("--out", required=True)
ap.add_argument("--npz", default="", help="깊이를 npz 로 남긴다 (영상 색칠용)")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


D = L(a.dump)
X0 = torch.as_tensor(np.asarray(D["x0"], dtype=np.float32)).to(dev)
P = D["pred"].float().numpy()

# --- 1. 껍질 깊이 ---------------------------------------------------------
lo, lat, nn = SX.grid_for_nodes(X0, a.n_nodes, hz_ratio=a.hz_ratio)
_i, _l, aux = SX.locate(X0, lo, lat, nn)
ci = aux[0].long()
lo_i = ci.min(0).values
ci0 = ci - lo_i
dims = (ci0.max(0).values + 1).tolist()
occ = torch.zeros(dims, device=dev)
occ.index_put_((ci0[:, 0], ci0[:, 1], ci0[:, 2]),
               torch.ones(ci0.shape[0], device=dev), accumulate=True)
cur = (occ > 0).float()
depth = torch.zeros_like(cur)
k = 0
while cur.sum() > 0 and k < 64:
    k += 1
    # 6-이웃 중 하나라도 비면 이번 층의 껍질이다
    pad = Fn.pad(cur.reshape(1, 1, *dims), (1, 1, 1, 1, 1, 1))
    nb = (pad[:, :, :-2, 1:-1, 1:-1] + pad[:, :, 2:, 1:-1, 1:-1]
          + pad[:, :, 1:-1, :-2, 1:-1] + pad[:, :, 1:-1, 2:, 1:-1]
          + pad[:, :, 1:-1, 1:-1, :-2] + pad[:, :, 1:-1, 1:-1, 2:]
          ).reshape(*dims)
    sh = (cur > 0) & (nb < 6)
    depth[sh] = k
    cur = cur * (~sh).float()
pdepth = depth[ci0[:, 0], ci0[:, 1], ci0[:, 2]].cpu().numpy().astype(int)

# --- 2. 흘러내린 입자 -----------------------------------------------------
X0n = X0.cpu().numpy()
c0 = X0n.mean(0)
cid = int(np.linalg.norm(X0n - c0, axis=-1).argmin())
rad = a.r if a.r > 0 else float(np.linalg.norm(X0n - X0n[cid], axis=-1).max())
cen = P[-1][cid]
dist = np.linalg.norm(P[-1] - cen, axis=-1)
down = (dist > rad) & (P[-1][:, 2] < cen[2])
print(f"[기준] 중심 입자 {cid}, 반지름 {rad:.4f}, 마지막 중심 z {cen[2]:.4f} "
      f"-> 흘러내림 {int(down.sum())}/{len(down)} ({100*down.mean():.2f}%)",
      flush=True)

ks = np.arange(1, pdepth.max() + 1)
tot = np.array([int((pdepth == q).sum()) for q in ks])
esc = np.array([int(((pdepth == q) & down).sum()) for q in ks])
rate = 100.0 * esc / np.maximum(tot, 1)
for q, t_, e_, r_ in zip(ks, tot, esc, rate):
    print(f"  깊이 {q}: 입자 {t_:6d}  흘러내림 {e_:6d}  비율 {r_:6.2f}%",
          flush=True)

fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=120)
ax[0].bar(ks, rate, color="tab:red")
ax[0].set_xlabel("껍질 깊이 (1 = 가장 바깥)")
ax[0].set_ylabel("흘러내린 비율 (%)")
ax[0].set_title(f"깊이별 흘러내림 비율  (전체 {100*down.mean():.2f}%)",
                fontsize=11)
ax[0].axhline(100 * down.mean(), color="0.4", ls="--", lw=1)
ax[0].grid(alpha=.3, axis="y")
ax[1].bar(ks - 0.2, tot, width=.4, label="그 깊이 입자 수", color="0.7")
ax[1].bar(ks + 0.2, esc, width=.4, label="그중 흘러내린 수",
          color="tab:red")
ax[1].set_xlabel("껍질 깊이"); ax[1].set_ylabel("입자 수")
ax[1].set_title("깊이별 개수 (비율과 함께 봐야 한다)", fontsize=11)
ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(a.out)
print(f"[저장] {a.out}", flush=True)
if a.npz:
    np.savez(a.npz, depth=pdepth, down=down)
    print(f"[npz] {a.npz}", flush=True)
