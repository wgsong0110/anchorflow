"""i-PhysGaussian config 를 DreamPhysics 솔버가 먹는 형식으로 바꾼다.

E 의 단위가 다르다. DreamPhysics 솔버는 config 의 E 에 1e7 을 곱하고, i-PG 는 곱하지
않는다. 그래서 i-PG 의 물리 단위 E 를 그대로 넣으면 1e7 배가 되어 터진다. 예전에
i-PG 재현에서 이것 때문에 k=1 대조가 24.82% 로 나왔고, 고치자 2.68% 가 됐다.

`additional_material_params` 안의 E 도 각각 나눠야 한다 -- 그걸 빠뜨린 것이 그때의
실제 실수였다.
"""
from __future__ import annotations

import argparse, copy, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--scale", type=float, default=1e7, help="DreamPhysics 가 곱하는 배수")
args = ap.parse_args()

c = json.load(open(args.src))
before = c.get("E")
c["E"] = c["E"] / args.scale
for blk in c.get("additional_material_params", []) or []:
    if "E" in blk:
        blk["E"] = blk["E"] / args.scale
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(c, open(args.out, "w"), indent=4)
print(f"{os.path.basename(args.src)}: E {before} -> {c['E']}, "
      f"추가 재질 블록 {len(c.get('additional_material_params') or [])}개, "
      f"재질 {c.get('material')}, n_grid {c.get('n_grid')}, g {c.get('g')}")
print(f"저장 {args.out}")
