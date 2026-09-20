"""씬 전용 러너(`gs_simulation/<씬>/gs_simulation_<씬>.py`)에서 **배경 가우시안**을
끄고, 저자 로컬에만 있는 파일 의존을 없앤다.

수박 러너는 시작할 때 두 가지를 저자의 절대경로에서 읽는다.

  opcity_zero_mask.pt   쓰는 곳이 **주석 처리된 한 줄뿐**이라 빈 텐서면 충분하다
  model/garden          배경으로 같이 그리는 3DGS. 그림에만 쓴다

둘 다 시뮬레이션 상태에 닿지 않는다. h5 만 필요할 때는 수백 MB 를 옮길 이유가 없다.
배경 블록은 `--render_img` 를 줄 때만 돌게 감싼다 (기본값이 꺼짐이라 그대로 꺼진다).

  python exe/patch_gf_scene_norender.py --runner <gs_simulation_watermelon.py>
"""
import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument("--runner", required=True)
a = ap.parse_args()

src = open(a.runner).read()
if "[anchorflow] 배경은 그릴 때만" in src:
    print("이미 되어 있다"); raise SystemExit(0)
lines = src.split("\n")

# 1) opa_mask -- 빈 텐서로
for i, ln in enumerate(lines):
    if re.match(r"\s*opa_mask\s*=\s*torch\.load\(", ln):
        ind = ln[:len(ln) - len(ln.lstrip())]
        lines[i] = (f"{ind}# [anchorflow] 저자 로컬 파일. 쓰는 곳이 주석 한 줄뿐이다.\n"
                    f"{ind}opa_mask = torch.zeros(0, dtype=torch.bool, device=\"cuda\")")
        break

# 2) 배경 블록 -- gaussians2 부터 그것을 쓰는 마지막 줄까지
beg = next((i for i, ln in enumerate(lines)
            if re.match(r"\s*gaussians2\s*=\s*load_checkpoint\(", ln)), None)
if beg is None:
    raise SystemExit("배경 블록을 못 찾았다")
NAMES = ("gaussians2", "pos2", "cov3D2", "rot2", "opacity_render2",
         "shs_render2", "combined_mask", "transform_matrix")
end = beg
for i in range(beg, min(beg + 60, len(lines))):
    if any(n in lines[i] for n in NAMES):
        end = i
ind = lines[beg][:len(lines[beg]) - len(lines[beg].lstrip())]
body = [(ind + "    " + ln.lstrip()) if ln.strip() else ln
        for ln in lines[beg:end + 1]]
lines[beg:end + 1] = ([ind + "# [anchorflow] 배경은 그릴 때만 불러온다",
                       ind + "if args.render_img:"] + body)

open(a.runner, "w").write("\n".join(lines))
print(f"고쳤다: {a.runner}  (배경 {end - beg + 1} 줄을 감쌌다)")
print("GFSCENE_OK")
