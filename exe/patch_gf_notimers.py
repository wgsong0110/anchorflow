"""솔버 안의 프로파일 타이머를 끈다.

`p2g2p` 는 단계마다 `wp.ScopedTimer(..., synchronize=True)` 로 감싸여 있다.
재는 데는 좋지만 돌릴 때는 **서브스텝마다 일곱 번 장치 동기화**가 걸려
파이프라인이 매번 비워진다. 프레임당 458 서브스텝이면 3206 번이다.

`--on` 으로 되돌릴 수 있다 (프로파일할 때는 켜야 한다).

  python exe/patch_gf_notimers.py --gf <GaussianFluent> [--on]
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
ap.add_argument("--on", action="store_true", help="다시 켠다")
a = ap.parse_args()

p = os.path.join(a.gf, "mpm_solver_warp", "mpm_solver_warp.py")
s = open(p).read()
old, new = ("synchronize=True", "synchronize=False")
if a.on:
    old, new = new, old
n = s.count(old)
s = s.replace(old, new)
open(p, "w").write(s)
print(f"{'켰다' if a.on else '껐다'}: {n} 곳")
print("GFTIMER_OK")
