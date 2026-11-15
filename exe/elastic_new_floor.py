"""elastic_new: PG 전처리 (불투명도 걸러냄 -> config 회전 -> transform2origin -> +1) 와 같은 식으로 가우시안 최저점 z 를 구해
sticky 바닥 (surface_collider) z 를 최저점 - DROP 으로 덮어쓴다.  python elastic_new_floor.py MODEL_DIR CFG [DROP=0.3]"""
import json, math, os, sys
import numpy as np
from plyfile import PlyData
md, cf = sys.argv[1], sys.argv[2]; drop = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
it = max(int(q.split("_")[-1]) for q in os.listdir(f"{md}/point_cloud") if q.startswith("iteration_"))
v = PlyData.read(f"{md}/point_cloud/iteration_{it}/point_cloud.ply")["vertex"]
x = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32); op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
c = json.load(open(cf)); x = x[op > c["opacity_threshold"]]
for dg, ax in zip(c["rotation_degree"], c["rotation_axis"]):
    cs, sn = math.cos(dg / 180.0 * 3.1415926), math.sin(dg / 180.0 * 3.1415926)
    R = {0: [[1, 0, 0], [0, cs, -sn], [0, sn, cs]], 1: [[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]], 2: [[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]]}[int(ax)]
    x = x @ np.array(R, np.float32).T
lo, hi = x.min(0), x.max(0); x = (x - (lo + hi) / 2) * (float(c.get("scale", 1.0)) / (hi - lo).max()) + 1.0
zmin = float(x[:, 2].min()); zf = round(zmin - drop, 4)
for b in c["boundary_conditions"]:
    if b["type"] == "surface_collider": b["point"][2] = zf
json.dump(c, open(cf, "w"), indent=1, ensure_ascii=False)
print(f"[바닥] 가우시안 {len(x)} 최저점 z {zmin:.4f} -> 바닥 z {zf} (낙하 {drop}); 상자 {x.min(0).round(3)} ~ {x.max(0).round(3)}")
