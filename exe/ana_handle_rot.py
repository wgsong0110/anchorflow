"""초기에 손잡이 안이던 입자들의 회전량과 상대 위치.

회전은 프레임 0 의 상대 위치 대비 **현재** 상대 위치로부터 바로 잰다 (프레임 간
증분을 적분하지 않는다). 아핀 적합 뒤 극분해로 신축을 떼어내고 회전만 남긴다.
상대 위치는 세 방향 투영으로 프레임마다 그려 영상으로 낸다 -- 색은 각 입자의
**초기** 상대 위치, 점선 원이 손잡이 경계.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import imageio.v2 as imageio
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
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--t0", type=int, default=0)
ap.add_argument("--lim", type=float, default=2.2, help="축 범위를 R 의 몇 배로")
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--out_fig", required=True)
ap.add_argument("--out_vid", required=True)
a = ap.parse_args()


def L(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")


D = L(a.dump)
P = np.asarray(D["pred"], dtype=np.float64)      # [T,N,3] 출력만 최적화
G = np.asarray(D["gt"], dtype=np.float64)        # [T,N,3] PG
X0 = np.asarray(D["x0"], dtype=np.float64)
R = float(np.asarray(D["ctrl_R"]).reshape(-1)[0])
cid = int(np.asarray(D["ctrl_id"]).reshape(-1)[0])
T = P.shape[0]
c0 = X0[cid]
sel = np.linalg.norm(X0 - c0, axis=1) <= R       # 초기에 손잡이 안이던 입자
n = int(sel.sum())
print(f"손잡이 R {R:.4f}, 중심입자 {cid}, 초기에 안이던 입자 {n} 개, {T} 프레임",
      flush=True)

# 중심은 각 방법이 쓰는 그 방법 자신의 손잡이 입자 현재 위치
def rel(A):
    return A[:, sel, :] - A[:, cid:cid + 1, :]


Ro = rel(P)          # [T,n,3]
Rg = rel(G)
r0 = X0[sel] - c0    # 초기 상대 위치 (두 방법 공통)


def fit(r_now, r_ini):
    """r_now ~ A r_ini 최소제곱 -> 극분해 A=QS, 회전각/축/잔차."""
    H = r_now.T @ r_ini
    M = r_ini.T @ r_ini
    A = H @ np.linalg.pinv(M)
    U, s, Vt = np.linalg.svd(A)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:                      # 반사를 회전으로 만든다
        U[:, -1] *= -1.0
        Q = U @ Vt
    cth = np.clip((np.trace(Q) - 1.0) / 2.0, -1.0, 1.0)
    th = np.degrees(np.arccos(cth))
    w = np.array([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
    nw = np.linalg.norm(w)
    ax = w / nw if nw > 1e-12 else np.array([0.0, 0.0, 1.0])
    # Procrussian 적합 (신축 없이 바로 직교) 교차 확인
    Up, sp, Vtp = np.linalg.svd(r_now.T @ r_ini)
    Qp = Up @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Up @ Vtp))]) @ Vtp
    thp = np.degrees(np.arccos(np.clip((np.trace(Qp) - 1.0) / 2.0, -1.0, 1.0)))
    res = np.sqrt(np.mean(np.sum((r_now - r_ini @ Qp.T) ** 2, 1)))
    # 적합 축 기준 입자별 회전각
    e3 = ax
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(e1 @ e3) > 0.9: e1 = np.array([0.0, 1.0, 0.0])
    e1 = e1 - (e1 @ e3) * e3; e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    def ang(v):
        return np.arctan2(v @ e2, v @ e1)
    da = np.degrees(np.unwrap(ang(r_now) - ang(r_ini)))
    rad = np.linalg.norm(r_ini - (r_ini @ e3)[:, None] * e3, axis=1)
    k = rad > 0.2 * R                             # 축에 붙은 입자는 각도가 무의미
    pp = (np.percentile(da[k], [5, 50, 95]) if k.sum() > 4
          else np.array([np.nan] * 3))
    return th, ax, res / R, thp, pp, np.sort(s)[::-1]


rows = {"우리": [], "PG": []}
for nm, RR in (("우리", Ro), ("PG", Rg)):
    for t in range(T):
        rows[nm].append(fit(RR[t], r0))
TH = {k: np.array([x[0] for x in v]) for k, v in rows.items()}
AX = {k: np.array([x[1] for x in v]) for k, v in rows.items()}
RS = {k: np.array([x[2] for x in v]) for k, v in rows.items()}
THP = {k: np.array([x[3] for x in v]) for k, v in rows.items()}
PP = {k: np.array([x[4] for x in v]) for k, v in rows.items()}
SV = {k: np.array([x[5] for x in v]) for k, v in rows.items()}
for k in TH:
    print(f"  [{k}] 회전각 최대 {np.nanmax(TH[k][a.t0:]):.2f}도 "
          f"(프레임 {a.t0 + int(np.nanargmax(TH[k][a.t0:]))}), 마지막 "
          f"{TH[k][-1]:.2f}도, Procrustes 최대 {np.nanmax(THP[k][a.t0:]):.2f}도, "
          f"잔차/R 최대 {np.nanmax(RS[k][a.t0:]):.3f}, 특이값 마지막 "
          f"{SV[k][-1][0]:.3f}/{SV[k][-1][1]:.3f}/{SV[k][-1][2]:.3f}", flush=True)

# ---- 그림: 프레임에 따른 회전각 / 축 / 잔차 ----
fr = np.arange(T)
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4), dpi=120)
for k, col in (("우리", "crimson"), ("PG", "royalblue")):
    ax[0].plot(fr[a.t0:], TH[k][a.t0:], color=col, lw=2, label=f"{k} 아핀 극분해")
    ax[0].plot(fr[a.t0:], THP[k][a.t0:], color=col, lw=1, ls="--", alpha=0.7,
               label=f"{k} Procrustes")
    ax[0].fill_between(fr[a.t0:], PP[k][a.t0:, 0], PP[k][a.t0:, 2],
                       color=col, alpha=0.12)
    ax[2].plot(fr[a.t0:], RS[k][a.t0:], color=col, lw=2, label=k)
    for j, (c2, ls) in enumerate(zip("xyz", ["-", "--", ":"])):
        ax[1].plot(fr[a.t0:], AX[k][a.t0:, j], color=col, ls=ls, lw=1.6,
                   alpha=0.85, label=f"{k} {c2}")
ax[0].set_title("손잡이 안이던 입자의 회전각 (프레임 0 기준)\n띠는 입자별 5~95%")
ax[0].set_ylabel("도")
ax[1].set_title("회전축 성분"); ax[1].set_ylim(-1.05, 1.05)
ax[2].set_title("강체 적합 잔차 / R")
for q in ax:
    q.set_xlabel("프레임"); q.grid(alpha=0.25); q.legend(fontsize=8)
fig.tight_layout(); fig.savefig(a.out_fig)
print(f"[저장] {a.out_fig}", flush=True)

# ---- 영상: 상대 위치 세 투영 ----
col = np.clip((r0 / R + 1.0) / 2.0, 0.0, 1.0)     # 초기 상대 위치 -> RGB
PRJ = [((0, 1), "x", "y", "z 방향으로 투영"),
       ((1, 2), "y", "z", "x 방향으로 투영"),
       ((0, 2), "x", "z", "y 방향으로 투영")]
lim = a.lim * R
th_c = np.linspace(0, 2 * np.pi, 200)
fig2, axs = plt.subplots(2, 3, figsize=(13.2, 9.0), dpi=100)
wr = imageio.get_writer(a.out_vid, fps=a.fps, codec="libx264", quality=8,
                        macro_block_size=1)
for t in tqdm(range(a.t0, T), ncols=70):
    for i, (nm, RR) in enumerate((("출력만 최적화", Ro), ("PG MPM", Rg))):
        for j, ((u, v), lu, lv, ttl) in enumerate(PRJ):
            q = axs[i][j]; q.clear()
            q.plot(R * np.cos(th_c), R * np.sin(th_c), "k--", lw=1.2, alpha=0.8)
            q.scatter(RR[t][:, u], RR[t][:, v], s=10, c=col, lw=0)
            q.axhline(0, color="gray", lw=0.5); q.axvline(0, color="gray", lw=0.5)
            q.set_xlim(-lim, lim); q.set_ylim(-lim, lim)
            q.set_aspect("equal"); q.grid(alpha=0.2)
            q.set_xlabel(lu); q.set_ylabel(lv)
            q.set_title(f"{nm}  {ttl}", fontsize=10)
    fig2.suptitle(f"초기에 손잡이 안이던 입자 {n}개의 상대 위치   프레임 {t}   "
                  f"회전 우리 {TH['우리'][t]:.1f}도 / PG {TH['PG'][t]:.1f}도   "
                  f"(점선 원 = 손잡이 경계 R={R:.3f})", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.canvas.draw()
    wr.append_data(np.ascontiguousarray(
        np.asarray(fig2.canvas.buffer_rgba())[..., :3]))
wr.close()
print(f"[저장] {a.out_vid}  {T - a.t0} 프레임 (프레임 {a.t0}~{T - 1})", flush=True)
