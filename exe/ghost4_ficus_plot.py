"""절 CM53 ficus 벌점 칸·면 위치 그림: python exe/ghost4_ficus_plot.py <g4cells_f*.npz> <출력.png> [제목] [x]
칸 중심 (격자 좌표 IP[0:3] + cb + 1 -> 세계 A·y), 입자 (회색, 밀도 영역별 색), 방식 0~3 의 벌점 면이 걸린 칸 (면 표지 fm ≠ 0 인 칸과 그 +축 이웃 칸).
영역 = 입자 밀도 (--region_density 의 ρ) 값별; 칸 영역 = 그 칸 안 입자의 가장 많은 영역. 표: 방식별 벌점 면 수를 영역별로 (면의 두 칸 영역 쌍).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for f_ in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf",):
    try:
        font_manager.fontManager.addfont(f_)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f_).get_name()
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

p, out = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else p
FX = len(sys.argv) > 4 and sys.argv[4] == "x"                                              # x = 구속 노드에 닿는 면 뺀 표지 (fmx*)
d = np.load(p)
IP, cb, A, x, rho, cut = d["IP"], d["cb"].astype(np.int64), d["A"].astype(np.float64), d["x"].astype(np.float64), d["rho"], d["cut"]
Ai = np.linalg.inv(A)
nc = cb.shape[0]
yc = cb + IP[0:3][None] + 1.0
xc = yc @ A.T
# 입자 -> 칸
b = np.floor(x @ Ai.T - 0.5).astype(np.int64) - IP[0:3][None]
key = lambda q: (q[:, 0] * 100003 + q[:, 1]) * 100019 + q[:, 2]
ck = key(cb)
order = np.argsort(ck)
pk = key(b)
pos = np.searchsorted(ck[order], pk)
pos = np.clip(pos, 0, nc - 1)
pc = np.where(ck[order][pos] == pk, order[pos], -1)
# 영역 = 밀도 값 (반올림)
rr = np.round(rho.astype(np.float64), 1)
vals, inv, cnt = np.unique(rr, return_inverse=True, return_counts=True)
print("입자 밀도 영역:", {float(v): int(c) for v, c in zip(vals, cnt)})
print("칸에 안 든 입자:", int((pc < 0).sum()), "/", len(pc))
zr = [(float(v), float(x[inv == i, 2].min()), float(x[inv == i, 2].max())) for i, v in enumerate(vals)]
print("영역별 z 범위:", zr)
creg = np.full(nc, -1)
ok = pc >= 0
M = np.zeros((nc, len(vals)), dtype=np.int64)
np.add.at(M, (pc[ok], inv[ok]), 1)
creg = M.argmax(1)
creg[M.sum(1) == 0] = -1


def face_cells(fm):
    a_, b_ = [], []
    lut = {k: i for i, k in enumerate(ck.tolist())}
    for ax in range(3):
        cs = np.nonzero((fm >> ax) & 1)[0]
        for c in cs:
            q = cb[c].copy(); q[ax] += 1
            c2 = lut.get(int(key(q[None])[0]), -1)
            a_.append(c); b_.append(c2)
    return np.array(a_, dtype=np.int64), np.array(b_, dtype=np.int64)


cols = ["tab:brown", "tab:green", "tab:olive", "tab:purple", "tab:cyan", "tab:pink"]
fig, axs = plt.subplots(2, 4, figsize=(20, 11))
NM = {0: "방식 0 (절 CM47 면 전부)", 1: "방식 1 (3x3x3 열림, 절 CM51)", 2: "방식 2 (5x5x5 열림)", 3: "방식 3 (내부 칸-경계 칸 면)"}
sub = np.random.default_rng(0).choice(len(x), min(len(x), 40000), replace=False)
rows = []
for t in range(4):
    kk = f"fmx{t}" if FX else f"fm{t}"
    fm = d[kk] if kk in d else None
    if fm is None:
        continue
    a_, b_ = face_cells(fm.astype(np.int64))
    pen = np.unique(np.concatenate([a_, b_[b_ >= 0]])) if len(a_) else np.zeros(0, dtype=np.int64)
    # 영역 쌍별 면 수
    pr = {}
    for c1, c2 in zip(a_, b_):
        k_ = tuple(sorted((float(vals[creg[c1]]) if creg[c1] >= 0 else -1.0, float(vals[creg[c2]]) if c2 >= 0 and creg[c2] >= 0 else -1.0)))
        pr[k_] = pr.get(k_, 0) + 1
    zf = xc[a_, 2] if len(a_) else np.zeros(0)
    rows.append((t, len(a_), len(pen), pr, (float(zf.min()), float(np.median(zf)), float(zf.max())) if len(zf) else None))
    for j, (i0, i1, lab) in enumerate(((0, 2, "x-z (옆)"), (1, 2, "y-z (옆)"))):
        ax = axs[j, t]
        for i, v in enumerate(vals):
            m_ = inv[sub] == i
            ax.scatter(x[sub][m_, i0], x[sub][m_, i1], s=0.2, c=cols[i % len(cols)], alpha=0.25, label=f"입자 ρ {v:g}")
        ax.scatter(xc[:, i0], xc[:, i1], s=1.5, c="0.6", alpha=0.3, label=f"활성 칸 {nc}")
        if len(pen):
            ax.scatter(xc[pen, i0], xc[pen, i1], s=10, c="red", label=f"벌점 면 칸 {len(pen)} (면 {len(a_)})")
        ax.set_title(f"{NM[t]}{' + 구속 노드 면 빼기' if FX else ''} · {lab}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel(["x", "y", "z"][i0] + " (m)"); ax.set_ylabel("z (m)")
        if t == 0:
            ax.legend(fontsize=7, markerscale=6, loc="upper right")
        elif len(pen):
            ax.legend(fontsize=7, markerscale=2, loc="upper right", handles=[ax.collections[-1]])
fig.suptitle(title, fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=110)
for t, nf, ncl, pr, zf in rows:
    print(f"방식 {t}: 벌점 면 {nf}, 칸 {ncl}, 면 z 최소·중앙·최대 {zf}, 영역 쌍별 면 {pr}")
