"""i-PG 러너가 config 의 `init_velocity` 를 읽게 한다 (PG 러너에 넣은 것과 같은 패치).

PhysGaussian 쪽 gs_simulation.py 에는 이 패치가 들어가 있었는데 i-PG 쪽에는 없어서,
파괴 시연에서 초기 하강속도가 **조용히 무시**됐다 (궤적의 첫 프레임 vz 가 0, 중력으로만
천천히 떨어졌다). finalize_mu_lam 바로 뒤, 모든 입자 속도를 그 값으로 둔다.
여러 번 돌려도 안전하다.

  python exe/patch_ipg_initv.py --ipg /home/dkta/work/i-physgaussian
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--ipg", required=True)
a = ap.parse_args()

p = os.path.join(a.ipg, "gs_simulation.py")
s = open(p).read()
if "[anchorflow] init v" in s:
    print("[패치] 이미 있다")
else:
    anchor = "    mpm_solver.finalize_mu_lam()\n"
    assert s.count(anchor) == 1, "finalize_mu_lam 위치가 하나가 아니다"
    add = anchor + '''    import json as _j3
    _v0 = _j3.load(open(args.config)).get("init_velocity", None)
    if _v0 is not None:
        mpm_solver.import_particle_v_from_torch(
            torch.zeros(mpm_init_pos.shape[0], 3, device="cuda").add_(
                torch.tensor(_v0, device="cuda", dtype=torch.float32)))
        print("[anchorflow] init v", _v0, flush=True)
'''
    s = s.replace(anchor, add, 1)
    open(p, "w").write(s)
    print(f"[패치] {p}")
