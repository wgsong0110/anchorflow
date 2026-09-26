"""지표 CSV 들을 모아 표로 낸다 (조합별 + 솔버별 요약)."""
import argparse
import csv
import glob
import os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default="bench_met/metrics_*.csv")
ap.add_argument("--visual", default="bench_met/visual_*.csv")
ap.add_argument("--out", default="bench/table.md")
ap.add_argument("--csv", default="bench/table.csv")
a = ap.parse_args()

rows = []
for f in sorted(glob.glob(a.glob)):
    with open(f) as fh:
        rows += list(csv.DictReader(fh))
vis = defaultdict(dict)
for f in sorted(glob.glob(a.visual)):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            vis[(r["solver"], r["combo"], r["seed"])] = r
if not rows:
    raise SystemExit("지표 CSV 가 없다")

NUM = ["CD", "EMD", "residual", "vol_ratio", "det_neg_mean", "det_neg_max",
       "mom_span", "ene_span", "penetration", "penetration_max", "fps", "wall_s"]
VNUM = ["PSNR", "SSIM", "LPIPS", "flicker"]


def agg(rs, keys):
    out = {}
    for k in keys:
        v = [float(r[k]) for r in rs
             if r.get(k) not in (None, "", "nan") and r[k] == r[k]]
        v = [x for x in v if x == x]
        out[k] = sum(v) / len(v) if v else float("nan")
    return out


bysolver = defaultdict(list)
bycombo = defaultdict(list)
for r in rows:
    bysolver[r["solver"]].append(r)
    bycombo[(r["solver"], r["combo"])].append(r)

L = ["# 베이스라인 벤치 (기준 PG)", "",
     "지표는 시드 평균이다. CD/EMD 는 PG 대비, 잔차는 바닥 접촉항을 포함한 i-PG "
     "정류 잔차를 길이로 환산해 물체 크기로 나눈 무차원 값이다.", "",
     "## 솔버별 요약", "",
     "| 솔버 | n | CD | EMD | 잔차 | 부피비 | detF<0 평균 | 관통 | 관통 최대 | "
     "운동량 변동 | 에너지 변동 | FPS | PSNR | SSIM | LPIPS |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for s, rs in sorted(bysolver.items()):
    m = agg(rs, NUM)
    vv = [vis[(s, r["combo"], r["seed"])] for r in rs
          if (s, r["combo"], r["seed"]) in vis]
    v = agg(vv, VNUM) if vv else {k: float("nan") for k in VNUM}
    L.append(f"| {s} | {len(rs)} | {m['CD']:.3e} | {m['EMD']*100:.3f}% | "
             f"{m['residual']:.3e} | {m['vol_ratio']:.3f} | "
             f"{m['det_neg_mean']*100:.3f}% | {m['penetration']*100:.3f}% | "
             f"{m['penetration_max']*100:.3f}% | {m['mom_span']:.3e} | "
             f"{m['ene_span']:.3e} | {m['fps']:.2f} | {v['PSNR']:.2f} | "
             f"{v['SSIM']:.4f} | {v['LPIPS']:.4f} |")

L += ["", "## 조합별", "",
      "| 솔버 | 조합 | n | CD | EMD | 잔차 | 부피비 | detF<0 | 관통 | FPS |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for (s, c), rs in sorted(bycombo.items()):
    m = agg(rs, NUM)
    L.append(f"| {s} | {c} | {len(rs)} | {m['CD']:.3e} | {m['EMD']*100:.3f}% | "
             f"{m['residual']:.3e} | {m['vol_ratio']:.3f} | "
             f"{m['det_neg_mean']*100:.3f}% | {m['penetration']*100:.3f}% | "
             f"{m['fps']:.2f} |")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
open(a.out, "w").write("\n".join(L) + "\n")
with open(a.csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("\n".join(L[:40]))
print(f"\n[저장] {a.out} / {a.csv} ({len(rows)} 행)")
