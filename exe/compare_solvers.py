"""물성을 똑같이 맞췄을 때 두 솔버가 같은 궤적을 내는지 잰다.

PhysGaussian 과 GaussianFluent 는 같은 코드에서 갈라졌지만 전달 방식과 감쇠가
다르다. 같은 물성을 줘도 궤적이 갈리면 두 쪽에서 만든 데이터를 한 모델에
먹일 수 없으므로, 먼저 **무엇을 맞춰야 같아지는지**를 확인한다.

맞춰야 하는 것들:
  flip_pic_ratio   GF 는 FLIP/PIC 혼합, PhysGaussian 은 APIC 다. 0 으로 두면 같다
  rpic_damping     둘 다 0
  grid_v_damping   둘 다 1 (감쇠 없음)
  n_grid, grid_lim, substep_dt, frame_dt, 재질, E, nu, density, g
  particle_filling 끔 (채우기가 입자 수와 순서를 바꾼다)

두 출력은 같은 ply 에서 같은 전처리를 거치므로 입자 순서가 같다. 그래서 프레임마다
입자별 차이를 그대로 잴 수 있다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="솔버 A 의 h5 디렉토리")
ap.add_argument("--b", required=True, help="솔버 B 의 h5 디렉토리")
ap.add_argument("--tag_a", default="A")
ap.add_argument("--tag_b", default="B")
ap.add_argument("--out", default=None)
ap.add_argument("--every", type=int, default=20)
a = ap.parse_args()


def frames(d):
    # sim_*.h5 만 프레임이다. 이어 돌리기용 state_last.h5 가 섞이면
    # 마지막 프레임이 하나 더 있는 것처럼 보인다.
    return sorted(glob.glob(os.path.join(d, "**", "sim_*.h5"), recursive=True))


def rd(p):
    with h5py.File(p, "r") as h:
        x = np.array(h["x"])
    return (x.T if x.shape[0] == 3 else x).astype(np.float64)


fa, fb = frames(a.a), frames(a.b)
n = min(len(fa), len(fb))
if n == 0:
    raise SystemExit("h5 가 없다")
x0 = rd(fa[0])
EXT = float(np.linalg.norm(x0.max(0) - x0.min(0)))
print(f"[비교] {a.tag_a} {len(fa)} 프레임 vs {a.tag_b} {len(fb)} 프레임, "
      f"입자 {x0.shape[0]} vs {rd(fb[0]).shape[0]}, 물체 {EXT:.4f}", flush=True)
if rd(fb[0]).shape[0] != x0.shape[0]:
    raise SystemExit("입자 수가 다르다 -- 전처리(채우기·불투명도 문턱)가 어긋났다")
d0 = np.abs(rd(fa[0]) - rd(fb[0])).max()
print(f"[0 프레임] 최대 차 {d0:.3e} (전처리가 같으면 0 이어야 한다)", flush=True)

rows = []
for i in range(0, n, a.every):
    xa, xb = rd(fa[i]), rd(fb[i])
    ok = np.isfinite(xa).all(1) & np.isfinite(xb).all(1)
    d = np.linalg.norm(xa[ok] - xb[ok], axis=1)
    mv = np.linalg.norm(xa[ok] - x0[ok], axis=1)
    rows.append(dict(f=i, rel=float(d.mean() / EXT),
                     p99=float(np.quantile(d, 0.99) / EXT),
                     moved=float(mv.mean() / EXT), bad=int((~ok).sum())))
    print(f"  f{i:4d}  차이 평균 {100*rows[-1]['rel']:6.3f}%  p99 "
          f"{100*rows[-1]['p99']:6.3f}%  (그동안 움직인 거리 평균 "
          f"{100*rows[-1]['moved']:6.3f}%)  비유한 {rows[-1]['bad']}", flush=True)

last = rows[-1]
print(f"\n[판정] 마지막 프레임 차이 {100*last['rel']:.3f}% / 이동 "
      f"{100*last['moved']:.3f}% = {last['rel']/max(last['moved'],1e-12):.3f}"
      f"  ({'일치' if last['rel'] < 0.05 * last['moved'] else '갈린다'})",
      flush=True)
if a.out:
    json.dump(dict(a=a.tag_a, b=a.tag_b, ext=EXT, rows=rows),
              open(a.out, "w"), indent=1)
    print(f"[저장] {a.out}", flush=True)
print("CMP_OK", flush=True)
