"""입자 수를 config 로 줄일 수 있게 한다 (`particle_subsample: k` -> k 개 중 1 개).

시간의 90% 가 입자 작업(p2g 78% + g2p + 응력)이라 입자를 줄이면 거의 비례해서
빨라진다. 다만 이것도 **이산화를 바꾸는 손잡이**라 시간 간격과 똑같이 결과를
바꾼다. 그래서 둘을 같은 잣대(원래 궤적과의 차이, 재현 바닥 8.42%)로 재서 고른다.

부피는 건드리지 않아도 된다 -- `get_particle_volume` 이 **셀 안 개수로 나누므로**
솎아낸 구름으로 다시 세면 입자마다 부피가 그만큼 커져 질량이 보존된다.

솎는 것은 `0::k` 로 **정해진 자리**라, 원래 궤적의 같은 자리와 바로 견줄 수 있다.

  python exe/patch_gf_subsample.py --gf <GaussianFluent>
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
a = ap.parse_args()

MARK = "# [anchorflow] 입자 솎기"
IMPORT = "import torch as _tsub\n"
OLD = """    mpm_init_vol = get_particle_volume("""
NEW = ('    ' + MARK + ': config 의 particle_subsample 만큼 k 개 중 1 개\n'
       '    import json as _jsub\n'
       '    _sub = int(_jsub.load(open(args.config)).get("particle_subsample", 1))\n'
       '    if _sub > 1:\n'
       '        # 가우시안에서 온 배열들도 **같은 자리**로 함께 솎아야 한다.\n'
       '        # 위치만 줄이면 뒤에서 공분산을 넣을 때 길이가 안 맞는다.\n'
       '        _keep = _tsub.arange(0, gs_num, _sub, device=mpm_init_pos.device)\n'
       '        _rest = mpm_init_pos[gs_num:][::_sub]\n'
       '        mpm_init_pos = _tsub.cat([mpm_init_pos[_keep], _rest], 0).contiguous()\n'
       '        init_cov = init_cov[_keep].contiguous()\n'
       '        init_shs = init_shs[_keep].contiguous()\n'
       '        init_opacity = init_opacity[_keep].contiguous()\n'
       '        gs_num = int(_keep.shape[0])\n'
       '        print(f"[anchorflow] 입자를 {_sub} 개 중 1 개로 솎았다 -> "\n'
       '              f"{mpm_init_pos.shape[0]} (가우시안 {gs_num})", flush=True)\n'
       '    mpm_init_vol = get_particle_volume(')

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
    # 껍질 beta 마스크는 **원래 가우시안 수**로 만들어진다. 솎았으면 같이 솎아야
    # beta 배열과 길이가 맞는다.
    MOLD = "     new_mask[all_neighbors] = True"
    if MOLD in s:
        s = s.replace(MOLD, MOLD + "\n     if _sub > 1:\n"
                      "         new_mask = new_mask[::_sub]", 1)
    if IMPORT not in s:
        s = s.replace("import torch\n", "import torch\n" + IMPORT, 1)
    open(p, "w").write(s)
    print(f"고쳤다: {p}")
    n += 1
print(f"GFSUB_OK ({n} 곳)")
