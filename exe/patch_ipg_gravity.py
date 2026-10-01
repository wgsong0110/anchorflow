"""i-PG implicit 솔버가 **가벼운 격자점을 통째로 버려 중력이 사라지는** 버그를 고친다.

증상: 몸체에서 떨어져 나간 입자가 중력을 전혀 받지 않는다. 실측(교사 궤적,
dv_z/dt 분위) --

    PG   : 1% 분위 -9.84,  -9.8+-2 인 비율 2.52%,  v_z 최소 -1.672
    i-PG : 1% 분위 -6.17,  -9.8+-2 인 비율 0.38%,  v_z 최소 -0.875

원인:

    mass_thresh = max(1e-10, 1e-2 * mass_n.max())    # **최대 질량의 1%**
    mask = mass_n > mass_thresh
    free_mask = mask & (~bc_mask)
    ...
    R[~free_mask] = 0.0

0 으로 나누기를 막으려는 장치인데 문턱이 **상대값**이다. 몸체 내부 격자점은
여러 입자가 겹쳐 질량이 크고 이탈 입자는 혼자 싣기 때문에 두 자릿수 작다 --
그래서 1% 문턱이 정확히 이탈 입자의 격자점만 골라 잘라낸다. 잘린 격자점은
`R=0` 이 되어 운동방정식이 사라지고 속도가 `v^n` 에 얼어붙는다. `f_ext = m*g`
도 함께 지워지므로 **중력이 통째로 없어진다**.

고치는 방법: 문턱을 수치 보호 목적에 맞는 **절대 기준**으로 내린다. 질량이
0 에 가까운 격자점만 빼고(0 나눗셈 방지), 실제 입자가 있는 가벼운 격자점은
방정식을 살려 둔다. 나눗셈 자리에는 하한을 둬서 보호를 유지한다.

여러 번 돌려도 안전하다.

  python exe/patch_ipg_gravity.py --ipg /home/dkta/work/i-physgaussian
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--ipg", required=True)
ap.add_argument("--rel", type=float, default=1e-8,
                help="최대 질량 대비 절대 문턱 비율 (기본 1e-8). 1e-2 가 기본값"
                     "이었고 그것이 이탈 입자를 잘라냈다")
a = ap.parse_args()
p = os.path.join(a.ipg, "implicit_mpm_solver.py")
s = open(p).read()
if "AF_MASS_THRESH" in s:
    print("[패치] 이미 들어 있다")
    raise SystemExit(0)

old = ("        mass_thresh = max(1e-10, 1e-2 * float(mass_n.max()))"
       " if mass_n.max() > 0 else 1e-10\n"
       "        mask = mass_n > mass_thresh\n")
assert old in s, "mass_thresh 줄을 못 찾았다"
new = (
    "        # AF_MASS_THRESH: 문턱은 **0 나눗셈 방지용** 이라 절대값이어야 한다.\n"
    "        # 원래 1e-2*max 였는데, 몸체 내부 격자점(여러 입자 겹침)보다 두\n"
    "        # 자릿수 가벼운 **이탈 입자의 격자점**이 거기 걸려 잘려 나갔다.\n"
    "        # 잘리면 R=0 이 되어 운동방정식이 사라지고 f_ext=m*g 까지 지워져\n"
    "        # 중력을 전혀 받지 못한다 (실측: 이탈 입자 dv_z/dt ~ 0, PG 는 -9.8).\n"
    "        _af_rel = float(_af_os_g.environ.get('AF_MASS_THRESH', '%g'))\n"
    "        mass_thresh = (max(1e-30, _af_rel * float(mass_n.max()))\n"
    "                       if mass_n.max() > 0 else 1e-30)\n"
    "        mask = mass_n > mass_thresh\n" % a.rel)
s = s.replace(old, new, 1)

# 나눗셈 보호: 질량으로 나누는 자리에 하한을 둔다 (문턱을 내린 대신)
old2 = ("        v_grid_n = np.zeros_like(momentum_n)\n"
        "        for d in range(3):\n"
        "            v_grid_n[..., d][mask] = momentum_n[..., d][mask]"
        " / mass_n[mask]\n")
assert old2 in s, "v_grid_n 나눗셈을 못 찾았다"
new2 = ("        v_grid_n = np.zeros_like(momentum_n)\n"
        "        # 문턱을 내렸으므로 나눗셈 자리에서 보호한다\n"
        "        _af_m = np.maximum(mass_n[mask], 1e-30)\n"
        "        for d in range(3):\n"
        "            v_grid_n[..., d][mask] = momentum_n[..., d][mask] / _af_m\n")
s = s.replace(old2, new2, 1)

if "import os as _af_os_g" not in s:
    s = s.replace("import numpy as np\n", "import numpy as np\nimport os as _af_os_g\n", 1)
open(p, "w").write(s)
print(f"[패치] 질량 문턱을 1e-2*max -> {a.rel:g}*max 로 내리고 나눗셈 보호 추가")
print(f"        -> {p}")
