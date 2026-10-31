"""토폴로지 추적 FEM 표현 판 (2026-10-31, rep_track2 --method fem --riem_partF --feas) 결과 표 -> LaTeX 행.

  python exe/topo_fem31_table.py --dir ~/workspace/result/anchorflow/topo/fem31/main
metrics 열: t, RMSE/L, CD/L, EMD, 입자 det F 최소, det F ≤ 0 비율, 평균 |x - x_gt|/L
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
a = ap.parse_args()
SC = ["wolf", "bread", "watermelon", "ficus", "hotdog"]
ME = [("fem", "Ours (FEM 표현)"), ("tetmesh", "사면체 메쉬"), ("grid", "그리드")]


def fmt_det(v):
    if not np.isfinite(v):
        return "--"
    if abs(v) >= 1e3 or (abs(v) < 1e-3 and v != 0):
        e = int(np.floor(np.log10(abs(v))))
        return f"${v / 10 ** e:.1f}\\times10^{{{e}}}$"
    return f"{v:.4f}"


rows, frames = [], []
for s in SC:
    for m, name in ME:
        p = os.path.join(a.dir, f"{s}_{m}.npz")
        if not os.path.exists(p):
            rows.append(f"{s} & {name} & \\multicolumn{{7}}{{l}}{{(실행 중)}} \\\\")
            continue
        D = np.load(p, allow_pickle=True)
        R = D["metrics"]
        nf = int(D["nfeas"]) if "nfeas" in D.files else -1
        name_ = name + ("$^\\ddagger$" if (s, m) == ("bread", "tetmesh") else "")
        rows.append(f"{s} & {name_} & {100 * R[:, 1].mean():.3f} & {100 * R[:, 2].mean():.3f} & "
                    f"{100 * R[:, 6].mean():.3f} & {fmt_det(R[:, 4].min())} & {100 * R[:, 5].max():.2f} & "
                    f"{D['dof'].mean():,.0f}".replace(",", "{,}") + f" & {nf} \\\\")
        T = R.shape[0]
        pick = [t for t in (1, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 180) if t <= T]
        frames.append((s, name, {t: 100 * R[t - 1, 1] for t in pick}))
    rows.append("\\midrule")
print("%% 요약 행")
print("\n".join(rows[:-1]))
print("%% 프레임별 RMSE")
for s, name, d in frames:
    print(f"{s} & {name} & " + " & ".join(f"{t}: {v:.3f}" for t, v in d.items()) + " \\\\")
