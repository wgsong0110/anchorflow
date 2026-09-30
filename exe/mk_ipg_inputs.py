"""i-PG 입력(시나리오 npz + 해상도별 config)을 만든다. 멱등이다.

해상도 기준:
  원본  n_grid 100 -> dx 0.020 (교사 PG 가 쓴 값)
  우리  n_grid 58  -> dx 0.0345 = 학생의 노드 간격 hn (자유도 간격을 맞춘 것)

`particle_filling.n_grid` 은 **둘 다 100 으로 고정**한다 -- 입자 집합이 같아야
손잡이 색인과 비교가 성립하고 채움 캐시를 공유할 수 있다.
"""
import argparse, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_grids", default="100,58")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
d = torch.load(a.traj, map_location="cpu", weights_only=False)

ci = d["ctrl_id"]
hid = (ci[0] if ci.dim() == 2 else ci).reshape(-1).numpy().astype(np.int64)
assert len(np.unique(hid)) == len(hid), f"손잡이 색인 중복: {hid}"
vel = d["ctrl_vel"].numpy().astype(np.float32)
R = float(d["ctrl_R"].reshape(-1)[0])
sp = os.path.join(a.out, "scen_s400706.npz")
np.savez(sp, hid=hid, vel=vel)
print(f"시나리오 -> {sp}  손잡이 {hid.tolist()}  vel {vel.shape}  R {R}")

c = {k: v for k, v in dict(d["cfg"]).items() if v is not None}
c["grid_lim"] = 2.0          # gs_simulation 이 직접 읽는다 (도메인 [0,2])
for ng in [int(q) for q in a.n_grids.split(",")]:
    cc = dict(c)
    cc["n_grid"] = ng
    cc["particle_filling"] = dict(cc["particle_filling"])
    cc["particle_filling"]["n_grid"] = 100        # 채움은 고정
    p = os.path.join(a.out, f"cfg_ng{ng}.json")
    json.dump(cc, open(p, "w"), indent=2)
    print(f"n_grid {ng} (dx {2.0 / ng:.4f}), 채움 100 -> {p}")
