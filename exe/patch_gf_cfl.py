"""시간 간격의 CFL 계수를 config 로 열어 둔다.

씬 러너는 `cfl = 0.6` 이 박혀 있고 `substep_dt = cfl * dx / 음속` 으로 프레임당
서브스텝 수가 정해진다 (수박: 458 개). 프레임 시간의 거의 전부가 여기서 나온다.

계수를 올리면 그만큼 빨라지지만 **정확도와 안정성을 바꾸는 손잡이**다. 그래서
config 키로 열어 두고, 올린 값이 원래 궤적과 같은지 같은 잣대로 확인한 뒤에만 쓴다.
키가 없으면 0.6 그대로라 기존 동작은 안 바뀐다.

  python exe/patch_gf_cfl.py --gf <GaussianFluent>
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
a = ap.parse_args()

MARK = "# [anchorflow] cfl 을 config 에서"
OLD = "    cfl = 0.6"
NEW = ('    ' + MARK + '\n'
       '    import json as _jcfl\n'
       '    cfl = float(_jcfl.load(open(args.config)).get("cfl", 0.6))\n')

n = 0
for rel in ("gs_simulation.py",
            os.path.join("gs_simulation", "watermelon", "gs_simulation_watermelon.py")):
    p = os.path.join(a.gf, rel)
    if not os.path.exists(p):
        continue
    s = open(p).read()
    if MARK in s or OLD not in s:
        continue
    s = s.replace(OLD, NEW, 1)
    open(p, "w").write(s)
    print(f"고쳤다: {p}")
    n += 1
print(f"GFCFL_OK ({n} 곳)")
