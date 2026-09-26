"""프레임 수를 줄인 벤치 설정을 만든다 (i-PG 처럼 비싼 솔버용).

i-PG 는 프레임당 substep 이 많아 (dt x20 에서 42, x8 에서 104) 240 프레임이
현실적으로 불가능하다 -- 실측 substep 당 9.5 초다. 프레임 수를 줄여 재고 그
비용을 표에 그대로 적는다.
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--frames", type=int, required=True)
a = ap.parse_args()
os.makedirs(a.dst, exist_ok=True)
for f in sorted(glob.glob(os.path.join(a.src, "*.json"))):
    c = json.load(open(f))
    c["frame_num"] = a.frames
    json.dump(c, open(os.path.join(a.dst, os.path.basename(f)), "w"))
    print(os.path.basename(f), a.frames)
