"""i-PG Newton-GMRES 에 float32 응력 잡음 바닥을 다루는 두 선택지를 넣는다 (환경 변수로만 켜짐, 기본 = 원래 동작 그대로).

진단 (exe/diag_ipg_wolf.py, wolf E 5e7 모래): F_trial = I 정확히여도 warp svd3 반올림으로 모든 입자 Kirchhoff 응력이 11.3 Pa
-> 첫 서브스텝 잔차 |R0| = |f_int| = 0.24 (전체 |m g| 4.46 의 5 %) 가 전부 이 잡음이다. 상대 기준 |R| < 1e-4 |R0| 은 잡음 아래라
닿을 수 없고, Armijo 줄 탐색이 10 번 모두 실패한 뒤에도 마지막 alpha 로 갱신해 (원 코드) k>=5 에서 첫 프레임에 발산한다.

  AF_NEWTON_ATOL_FEXT=c : |R| < c * |f_ext(자유점)| 이면 수렴 (절대 기준을 중력 하중에 맞춤). 0/없음 = 끔.
  AF_LS_REJECT=1        : 줄 탐색이 10 번 모두 Armijo 를 못 맞추면 그 갱신을 버리고 Newton 을 멈춤 (원 코드는 alpha=1/512 로 갱신).

여러 번 돌려도 안전하다.  python exe/patch_ipg_noisefloor.py --ipg /home/dkta/work/i-physgaussian
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--ipg", required=True)
a = ap.parse_args()
p = os.path.join(a.ipg, "implicit_mpm_solver.py")
s = open(p).read()
if "AF_NEWTON_ATOL_FEXT" in s:
    print("[이미 적용] patch_ipg_noisefloor")
    raise SystemExit(0)

old1 = """            if rel_res < _newton_tol or res_norm < 1e-10:
                converged = True"""
new1 = """            if newton_it == 0:                                   # [anchorflow noisefloor] 절대 기준 (중력 하중 대비)
                _af_atol_c = float(_os.environ.get("AF_NEWTON_ATOL_FEXT", "0") or 0)
                _af_atol = (_af_atol_c * 9.8 * float(cp.sqrt(cp.sum((mass_gpu * free_mask_gpu) ** 2)))) if _af_atol_c > 0 else 0.0
            if rel_res < _newton_tol or res_norm < 1e-10 or res_norm < _af_atol:
                converged = True"""
assert s.count(old1) == 1, "old1"
s = s.replace(old1, new1)

old2 = """                if phi_trial <= phi_0 + c1 * alpha * dphi_0:
                    break
                alpha *= 0.5

            du_k = du_k + alpha * delta_du
"""
new2 = """                if phi_trial <= phi_0 + c1 * alpha * dphi_0:
                    break
                alpha *= 0.5
            else:                                                # [anchorflow noisefloor] 줄 탐색 실패 -> 갱신 버림
                if _os.environ.get("AF_LS_REJECT") == "1":
                    final_residual = res_norm
                    break

            du_k = du_k + alpha * delta_du
"""
assert s.count(old2) == 1, "old2"
s = s.replace(old2, new2)
open(p, "w").write(s)
print("[적용] patch_ipg_noisefloor")
