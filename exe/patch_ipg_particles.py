"""PG/i-PG 가 **주어진 입자 집합**을 그대로 시뮬레이션하게 만든다. 멱등이다.

gs_simulation 은 입자를 3DGS 모델의 가우시안 중심에서 가져온다 (채우기를 끄면
가우시안 중심 그 자체다). 합성한 장면 -- 예를 들어 공중의 공 -- 을 MPM 으로
돌려 보려면 그 자리에 우리 입자를 넣어야 한다.

AF_PARTICLES_NPY 가 있으면 좌표 변환(transform2origin / shift2center111) **뒤에**
입자를 갈아끼운다. 즉 npy 는 시뮬레이션 좌표계(도메인 [0, grid_lim]) 그대로다.
가우시안 속성(shs/opacity/cov)은 렌더용이라 첫 행을 복제해 길이만 맞춘다 --
우리가 쓰는 것은 --output_h5 로 나오는 입자 궤적뿐이다.

  python exe/patch_ipg_particles.py --ipg /home/dkta/work/i-physgaussian
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--ipg", required=True)
a = ap.parse_args()
p = os.path.join(a.ipg, "gs_simulation.py")
s = open(p).read()
if "AF_PARTICLES_NPY" in s:
    print("[패치] 이미 들어 있다")
    raise SystemExit(0)

old = """    # fill particles if needed
    gs_num = transformed_pos.shape[0]
"""
new = """    # --- AF: 입자 집합을 통째로 갈아끼운다 (합성 장면용) ----------------
    _afp = os.environ.get("AF_PARTICLES_NPY")
    if _afp:
        import numpy as _afnp
        _P = _afnp.load(_afp)
        _P = torch.tensor(_P, dtype=transformed_pos.dtype,
                          device=transformed_pos.device)
        print(f"[AF입자] {_afp} -> {_P.shape[0]} 개로 갈아끼운다 "
              f"(가우시안 {transformed_pos.shape[0]} 개 대신) AF_PARTICLES_NPY",
              flush=True)
        _n = _P.shape[0]
        transformed_pos = _P
        init_cov = init_cov[:1].repeat(_n, 1)
        init_opacity = init_opacity[:1].repeat(_n, 1)
        init_shs = init_shs[:1].repeat(
            *([_n] + [1] * (init_shs.dim() - 1)))
    # fill particles if needed
    gs_num = transformed_pos.shape[0]
"""
assert s.count(old) == 1, "채우기 앞 지점을 못 찾았다"
if "\nimport os" not in s and "\nimport os\n" not in s:
    s = "import os\n" + s
open(p, "w").write(s.replace(old, new, 1))
print(f"[패치] 입자 갈아끼우기 추가 -> {p}")
