"""GF 모래의 **빠진 Jp 항**을 되살린다 (Klar et al. 2016 / Genesis 와 같게).

GF 의 `sand_return_mapping` 은 이렇게 되어 있다.

    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]

`Jp` 가 **주석 처리**돼 있고 갱신도 안 한다. 원본에서 Jp 는 당겨져 벌어진 만큼을
기억해 두는 값이라, 이게 없으면 압력이 낮은 더미 **표면에서 전단 저항이 0 에
가까워** 계속 흘러내린다. 실측으로 마찰각을 50 -> 70 도로 20 도나 올려도 안식각이
19.4 -> 22.6 도로 3 도밖에 안 움직였다 -- 재질 손잡이가 아니라 이 항이 막고 있었다.

Genesis 의 같은 재질(`materials/MPM/sand.py`)은 `tr = epsilon.sum() + Jp` 이고
`Jp_new = tr if tr >= 0 else 0` 으로 갱신한다. 그쪽에 맞춘다.

  python exe/patch_gf_sand_jp.py --gf <GaussianFluent>
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
a = ap.parse_args()

MARK = "# [anchorflow] 모래의 Jp 항 복원"
p = os.path.join(a.gf, "mpm_solver_warp", "mpm_utils.py")
s = open(p).read()
if MARK in s:
    print("이미 되어 있다"); raise SystemExit(0)

OLD = "    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]"
NEW = ("    " + MARK + " (Klar et al. / Genesis 와 같게)\n"
       "    tr = epsilon[0] + epsilon[1] + epsilon[2] + state.particle_Jp[p]")
if OLD not in s:
    raise SystemExit("모래의 tr 줄을 못 찾았다")
s = s.replace(OLD, NEW, 1)

# Jp 갱신: 당겨져 벌어지면(tr>=0) 그만큼 기억하고, 눌리면 0 으로 되돌린다
OLD2 = """    if delta_gamma > 0 and tr > 0:
        F_elastic = U * wp.transpose(V)"""
NEW2 = """    if tr >= 0.0:
        state.particle_Jp[p] = tr
    else:
        state.particle_Jp[p] = 0.0

    if delta_gamma > 0 and tr > 0:
        F_elastic = U * wp.transpose(V)"""
if OLD2 not in s:
    raise SystemExit("모래의 갈래를 못 찾았다")
s = s.replace(OLD2, NEW2, 1)
open(p, "w").write(s)
print(f"고쳤다: {p}")
print("GFSANDJP_OK")
