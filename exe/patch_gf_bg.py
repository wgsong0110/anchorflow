"""GF 최상위 러너(gs_simulation.py)의 프레임 루프 안 배경 가우시안 읽기 (저자 로컬 경로 .../model/garden_ours) 를 --render_img 일 때만 돌게 감싼다.
입자 상태에 닿지 않는 그림용 배경이다 (h5 만 필요할 때 경로가 없어 프레임 1 직후 죽던 문제). 시뮬 코드는 그대로.
  python patch_gf_bg.py SRC DST"""
import sys
src = open(sys.argv[1]).read().split("\n"); out = []; i = 0; n = 0
while i < len(src):
    ln = src[i]
    if ln.strip().startswith('gaussians2 = load_checkpoint(') and not ln.strip().startswith("#"):
        ind = ln[:len(ln) - len(ln.lstrip())]
        out.append(f"{ind}if args.render_img:   # [anchorflow] 배경은 그릴 때만")
        j = i
        while j < len(src) and "shs_render2 =" not in src[j]:
            out.append("    " + src[j]); j += 1
        out.append("    " + src[j]); i = j + 1; n += 1; continue
    out.append(ln); i += 1
txt = "\n".join(out)
if "--nobg" in sys.argv:        # 영상용: 배경 가우시안을 빼고 (공식 렌더 경로 그대로, 배경 항만 없음), 래스터라이저 반환 개수 (2 또는 3) 둘 다 받게
    import re
    txt = txt.replace("if args.render_img:   # [anchorflow] 배경은 그릴 때만\n            gaussians2", "if False:   # [anchorflow] 영상용 배경 없음\n            gaussians2")
    txt = re.sub(r"torch\.concat\(\[(\w+)\s*,\s*\w+2\],\s*dim\s*=\s*0\s*\)", r"\1", txt)
    txt = txt.replace("rendering, raddi, point_xy = rasterize(", "rendering, raddi, point_xy = _r3(rasterize)(")
    txt = txt.replace("\nif __name__", "\ndef _r3(f):\n    def g(**k):\n        o = f(**k); return o if len(o) == 3 else (o[0], o[1], None)\n    return g\n\nif __name__", 1)
open(sys.argv[2], "w").write(txt); print("감쌈", n)
