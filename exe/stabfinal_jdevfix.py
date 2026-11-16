"""[stabfinal 11-16] 이미 만든 extra.json 의 한 항목에 stab_extra.py 의 부피 변화 규칙 (영역 상자 밖 입자가 있는 판은 jdev_max 평균에서 뺌) 을 다시 적용.
판별 npz (stab_mpm/stab_omni 출력: stats, keys) 의 프레임 통계 jdev·out 만 씀 (다시 돌리지 않음).
  python exe/stabfinal_jdevfix.py <extra.json> <항목 이름, 예 pg_N45> <판 a npz> [판 b npz] [판 c npz]
"""
import json, sys
import numpy as np

ex, key, fs = sys.argv[1], sys.argv[2], sys.argv[3:]
d = json.load(open(ex)); e = d["phys"][key]
runs = []
for f in fs:
    z = np.load(f, allow_pickle=True); k = [str(x) for x in z["keys"]]; s = z["stats"]
    q = dict(jdev_max=float(np.nanmax(s[:, k.index("jdev")])), out_max=float(np.nanmax(s[:, k.index("out")])))
    print(f, q); runs.append(q)
clean = [q for q in runs if q["out_max"] == 0.0]
e["out_max"] = float(np.mean([q["out_max"] for q in runs]))
if clean and len(clean) < len(runs):
    e["jdev_max_all"] = float(np.mean([q["jdev_max"] for q in runs]))
    e["jdev_max"] = float(np.mean([q["jdev_max"] for q in clean]))
e["jdev_runs"] = len(clean) if clean else len(runs)
print(key, e)
json.dump(d, open(ex, "w"), indent=1)
