"""GF 러너에서 **배경 가우시안 불러오기와 그리기**를 끄는 스위치를 단다.

`gs_simulation.py` 는 프레임마다 `model/garden_ours`(702MB)를 배경으로 불러온다.
그림에만 쓰는 것이라 h5 만 필요할 때는 짐이다 -- 파일도 옮겨야 하고, 프레임마다
체크포인트를 다시 읽어 느리다. 시뮬레이션 상태에는 전혀 닿지 않는다.

`--no_render` 를 주면 그 줄들을 건너뛴다. 물리는 그대로다.

  python exe/patch_gf_norender.py --gf <GaussianFluent>
"""
import argparse
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
a = ap.parse_args()

p = os.path.join(a.gf, "gs_simulation.py")
s = open(p).read()
if "[anchorflow] no_render" in s:
    print("이미 되어 있다"); raise SystemExit(0)

# 1) 인자 추가 -- argparse 블록의 마지막 add_argument 뒤에 붙인다
m = list(re.finditer(r"^\s*parser\.add_argument\(.*?\)\s*$", s, re.M | re.S))
if not m:
    raise SystemExit("argparse 블록을 못 찾았다")
i = m[-1].end()
s = (s[:i] + '\n    # [anchorflow] no_render: 배경 불러오기와 그리기를 건너뛴다\n'
     '    parser.add_argument("--no_render", action="store_true")' + s[i:])

# 2) 배경 체크포인트 불러오는 블록을 감싼다
old = '''        gaussians2 = load_checkpoint('''
if old not in s:
    raise SystemExit("배경 불러오는 줄을 못 찾았다")
j = s.index(old)
# 뒤에 오는 `if args.render_img:` 는 **줄바꿈과 들여쓰기까지 포함해서** 찾는다.
# 들여쓰기를 빼고 자르면 그 줄이 컬럼 0 으로 밀려 파일이 깨진다 (한 번 겪었다).
tail = "\n        if args.render_img:"
k = s.index(tail, j)
block = s[j:k]
wrapped = ("        if not args.no_render:\n"
           + "\n".join(("    " + ln) if ln.strip() else ln
                       for ln in block.rstrip("\n").split("\n")))
s = (s[:j] + wrapped
     + "\n        if args.render_img and not args.no_render:"
     + s[k + len(tail):])

open(p, "w").write(s)
print(f"고쳤다: {p}")
print("GFNORENDER_OK")
