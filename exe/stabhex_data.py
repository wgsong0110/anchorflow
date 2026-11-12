"""서브스텝 안정성 '우리(육면체)' 입력 만들기 (11-12 지시, 묶음 stabhex): 안정성 실험 장면 (doc/stab_plan.md) 의 0 프레임 입자 + config 를
육면체 실행기 exe/fem_an_rt_sim.py 의 --data 형식 (.pt: cfg, x [T,N,3], v, F) 으로.

- 입자·순서·설정 = /home/dkta/work/stab/<형상>/init/sim_0000000000.h5 · config.json 그대로 (다른 방법과 같은 입력 번호 -> stabstat.subset 같은 번호).
- 정답 궤적은 없다: x 는 0 프레임을 61 번 펼친 보기 (실행기의 err 는 '처음 위치 대비 평균 이동/EXT' 가 되어 뜻 없음; 발산 판정 e > 2 만 남음).
- F = 항등 (정지 초기 상태 -- F_is_true 로 위치에서 되살리지 않게), v = h5 v (없으면 0).

  python exe/stabhex_data.py <형상 폴더> <출력 .pt> [프레임 수 60]
"""
import json
import sys

import h5py
import numpy as np
import torch

src, out = sys.argv[1], sys.argv[2]
T = int(sys.argv[3]) if len(sys.argv) > 3 else 60


def rd(h, k):
    if k not in h:
        return None
    v = np.array(h[k])
    return v.T if (v.ndim == 2 and v.shape[0] in (3, 9) and v.shape[0] != v.shape[-1]) else v


with h5py.File(f"{src}/init/sim_0000000000.h5", "r") as h:
    x0 = rd(h, "x").astype(np.float32)
    v0 = rd(h, "v")
cfg = json.load(open(f"{src}/config.json"))
N = x0.shape[0]
v0 = np.zeros_like(x0) if v0 is None else v0.reshape(N, 3).astype(np.float32)
X = torch.from_numpy(x0)[None].expand(T + 1, N, 3)                       # 저장은 N x 3 하나 (펼친 보기)
d = dict(cfg=cfg, x=X, v=torch.from_numpy(v0)[None], F=torch.eye(3).reshape(1, 1, 3, 3).expand(1, N, 3, 3).contiguous(), F_is_true=True,
         stabhex=dict(src=src, frames=T, note="x 는 0 프레임 반복 (정답 없음)"))
torch.save(d, out)
print(f"[stabhex 입력] {src} -> {out}: 입자 {N}, 프레임 {T}, |v0| 최대 {float(np.abs(v0).max()):.3g}, z 최소 {float(x0[:, 2].min()):.4f}, "
      f"바닥 {[b for b in cfg['boundary_conditions'] if b['type'] == 'surface_collider'][0]['point'][2]}", flush=True)
