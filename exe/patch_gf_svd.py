"""수박(재질 7)에서 **쓰이지도 않는 SVD** 한 번을 뺀다. 답은 그대로다.

`compute_stress_from_F_trial` 은 재질을 가리지 않고 `wp.svd3` 를 한 번 푼다.
그런데 재질 7 의 응력(`kirchoff_stress_neoHookeanBoarden`)은 F, J, mu, kappa 만
쓰고 U/sigma/V 를 쓰지 않는다. 게다가 같은 커널 안에서 NACC 되돌림이 이미 SVD 를
한 번 풀었다. 즉 입자마다 매 서브스텝 SVD 를 두 번 푸는데 하나는 버려진다.

  python exe/patch_gf_svd.py --gf <GaussianFluent>
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
a = ap.parse_args()

MARK = "# [anchorflow] 재질 7 은 U/sigma/V 를 안 쓴다"
p = os.path.join(a.gf, "mpm_solver_warp", "mpm_utils.py")
s = open(p).read()
if MARK in s:
    print("이미 되어 있다"); raise SystemExit(0)

OLD = "        wp.svd3(state.particle_F[p], U, sig, V)"
NEW = ("        " + MARK + "\n"
       "        if model.material[p] != 7:\n"
       "            wp.svd3(state.particle_F[p], U, sig, V)")
if s.count(OLD) != 1:
    raise SystemExit(f"svd3 호출이 {s.count(OLD)} 곳이다 -- 한 곳이어야 한다")
s = s.replace(OLD, NEW, 1)
open(p, "w").write(s)
print(f"고쳤다: {p}")
print("GFSVD_OK")
