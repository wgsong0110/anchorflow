"""train_deform 롤아웃 덤프(AF_ROLL_DUMP, AF_ROLL_DUMP_F=1) -> lb_render.py 가 읽는 궤적 디렉토리.

프레임 0 은 그 궤적의 처음 상태, 그 뒤는 모델 예측 위치와 모델 변형장 야코비안 누적 F.
lb_gen 처럼 렌더용 F 에는 초기 회전 R 을 싣는다 (F_render = F_sim · R). meta.json 은 원래 궤적 것을 그대로 복사.

  python exe/lb_roll_to_h5.py --dump roll.pt --traj /home/dkta/work/lbdata/test/0000 --out DIR
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--traj", required=True, help="원래 lbdata 궤적 디렉토리 (meta.json, sim_0000000000.h5)")
ap.add_argument("--out", required=True)
ap.add_argument("--first_from_dump", action="store_true",
                help="프레임 0 을 원래 궤적 0 프레임 대신 덤프 pred[0]·F[0] 으로 (이어 돌리기 덤프: pred[0] = 시작 상태), 그 뒤 pred[1:]")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
d = torch.load(a.dump, weights_only=False)
assert d.get("F") is not None, "덤프에 F 가 없다 -- AF_ROLL_DUMP_F=1 로 다시"
meta = json.load(open(f"{a.traj}/meta.json"))
R = torch.as_tensor(np.array(meta["R"]), dtype=torch.float32)
json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
P, F = d["pred"].float(), d["F"].float()
if a.first_from_dump:
    P, F = P[1:], F[1:]
    with h5py.File(f"{a.out}/sim_0000000000.h5", "w") as o:
        o.create_dataset("x", data=d["pred"][0].float().numpy().astype(np.float32))
        o.create_dataset("F", data=(d["F"][0].float() @ R).reshape(-1, 9).numpy().astype(np.float32))
else:
    with h5py.File(f"{a.traj}/sim_0000000000.h5") as h, h5py.File(f"{a.out}/sim_0000000000.h5", "w") as o:
        for k in ("x", "F"):
            o.create_dataset(k, data=np.array(h[k]))
for i in range(P.shape[0]):
    Fr = (F[i] @ R).reshape(-1, 9)
    with h5py.File(f"{a.out}/sim_{i + 1:010d}.h5", "w") as o:
        o.create_dataset("x", data=P[i].numpy().astype(np.float32))
        o.create_dataset("F", data=Fr.numpy().astype(np.float32))
print(f"[저장] {a.out}  {P.shape[0] + 1} 프레임, det F 범위 {float(torch.linalg.det(F[-1]).min()):.3f}~{float(torch.linalg.det(F[-1]).max()):.3f}", flush=True)
