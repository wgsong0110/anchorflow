"""타이치 공식 예제 `mpm3d.py` 를 **물리 부분은 손대지 않고** 돌려 궤적을 뽑는다.

직접 짠 유체가 발산해서(입자 11.5 만 중 11.4 만이 비유한값) 검증된 구현으로
바꾼다. 타이치가 함께 배포하는 `taichi/examples/simulation/mpm3d.py` 는 저자들의
3D MLS-MPM 레퍼런스(약압축성 유체, E=400, p_rho=1)다.

예제 파일은 맨 아래에 GUI 루프가 있고 `if __name__ == "__main__"` 로 막혀 있어서
**그냥 import 하면 물리 정의(init, substep)만 들어온다**. 그래서 한 줄도 고치지
않고 불러다 쓰고, 프레임마다 위치·속도를 h5 로 떨군다 (키 이름은 기성 러너와 맞춘다).
n_grid 같은 설정은 예제 상단의 상수라 바꾸려면 파일을 복사해 고쳐야 하므로,
기본값(32 격자, 16384 입자)을 그대로 쓰고 해상도만 --copy_with 로 바꾼다.
"""
import argparse
import importlib.util
import os
import re
import shutil
import sys

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=240)
ap.add_argument("--n_grid", type=int, default=0,
                help="0 이면 예제 기본값(32). 바꾸면 예제 파일을 복사해 그 줄만 고친다")
a = ap.parse_args()

import taichi                                                     # noqa: E402

src = os.path.join(os.path.dirname(taichi.__file__), "examples",
                   "simulation", "mpm3d.py")
if not os.path.exists(src):
    raise SystemExit(f"공식 예제를 못 찾았다: {src}")

use = src
if a.n_grid:
    # 예제 상단의 `dim, n_grid, steps, dt = 3, 32, 25, 4e-4` 한 줄만 바꾼다.
    # dt 는 예제가 해상도별로 주석에 적어둔 값을 그대로 따른다.
    DT = {32: 4e-4, 64: 2e-4, 128: 8e-5}
    if a.n_grid not in DT:
        raise SystemExit(f"예제가 적어둔 해상도만 쓴다: {sorted(DT)}")
    use = os.path.join(os.path.dirname(a.out) or ".",
                       f"mpm3d_n{a.n_grid}.py")
    os.makedirs(os.path.dirname(use) or ".", exist_ok=True)
    txt = open(src).read()
    new = f"dim, n_grid, steps, dt = 3, {a.n_grid}, 25, {DT[a.n_grid]:g}"
    txt2 = re.sub(r"^dim, n_grid, steps, dt = 3, 32, 25, 4e-4$", new, txt,
                  count=1, flags=re.M)
    if txt2 == txt:
        raise SystemExit("해상도 줄을 못 찾았다 -- 예제가 바뀌었다")
    open(use, "w").write(txt2)
    print(f"[복사] {use} (격자 {a.n_grid})", flush=True)

spec = importlib.util.spec_from_file_location("mpm3d_official", use)
m = importlib.util.module_from_spec(spec)
sys.modules["mpm3d_official"] = m
spec.loader.exec_module(m)                 # GUI 루프는 __main__ 가드 뒤에 있다
print(f"[공식] {src}", flush=True)
print(f"[설정] 격자 {m.n_grid}, 입자 {m.n_particles}, steps {m.steps}, "
      f"dt {m.dt:g}, E {m.E}, rho {m.p_rho}", flush=True)

m.init()
os.makedirs(a.out, exist_ok=True)
for f in range(a.frames + 1):
    x = m.F_x.to_numpy()
    v = m.F_v.to_numpy()
    J = m.F_J.to_numpy()
    with h5py.File(os.path.join(a.out, "sim_%010d.h5" % f), "w") as h:
        h.create_dataset("x", data=x.T.astype(np.float32))
        h.create_dataset("v", data=v.T.astype(np.float32))
        # 이 예제의 유체는 변형구배를 부피비 J 하나로만 들고 있다
        F = np.zeros((len(x), 9), np.float32)
        s = np.cbrt(np.maximum(J, 1e-6))
        F[:, 0] = F[:, 4] = F[:, 8] = s
        h.create_dataset("f_tensor", data=F.T)
        h.create_dataset("J", data=J.astype(np.float32))
        h.create_dataset("time", data=np.array([[f * m.steps * m.dt]]))
    if f == a.frames:
        break
    for _ in range(m.steps):
        m.substep()
    if f % 40 == 0:
        fin = np.isfinite(x).all(1)
        print(f"  f{f:4d}  비유한 {int((~fin).sum())}/{len(x)}  "
              f"x[{x[fin].min():.3f},{x[fin].max():.3f}]  "
              f"J 중앙 {np.median(J):.3f}", flush=True)
print(f"[저장] {a.out}  {a.frames + 1} 프레임", flush=True)
print("MPM3D_OK", flush=True)
