"""토폴로지 추적 실험: 시뮬 h5 궤적 -> rep_track2 입력 (--flow, --aux).

목표 점 = 원본 3DGS 가우시안 중심 (공식 불투명도 거르기 뒤 전부, 내부 채움 없음, 부분표본 없음).
시뮬 러너(PhysGaussian·GaussianFluent)는 가우시안을 거른 순서 그대로 입자 배열 앞에 두고 채움 입자를 뒤에 붙인다.
러너와 같은 변환(회전·transform2origin·shift2center111)으로 가우시안을 시뮬 좌표로 옮겨 0 프레임 입자 앞부분과
대조하고(어긋나면 멈춤), 그 입자들의 프레임별 위치를 쓴다. 정규화는 gauss_flow·rep_aux 와 같다
(0 프레임 전체 입자 lo·s, 짧은 축 가운데 맞춤). 어느 프레임에서든 값이 유한하지 않은 가우시안은 뺀다.

  python exe/topo_flow.py --h5_dir topo/pg_wolf_sand --model pgmodel/wolf_whitebg-trained \
      --config PhysGaussian/config/wolf_config.json --out topo/flow_wolf.npz --aux_out topo/aux_wolf.npz
"""
import argparse
import glob
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--h5_dir", required=True)
ap.add_argument("--model", required=True, help="3DGS 모델 디렉토리 (point_cloud/iteration_*/point_cloud.ply)")
ap.add_argument("--config", required=True, help="시뮬에 쓴 설정 json (회전·크기·불투명도 문턱)")
ap.add_argument("--out", required=True)
ap.add_argument("--aux_out", required=True)
ap.add_argument("--every", type=int, default=1, help="이 프레임 간격으로만 (목표 프레임 수 줄이기)")
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
a = ap.parse_args()
sys.path.append(a.pg)
sys.path.append(os.path.join(a.pg, "gaussian-splatting"))
_cwd = os.getcwd()
os.chdir(a.pg)

import h5py                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from utils.transformation_utils import (                          # noqa: E402
    generate_rotation_matrices, apply_rotations, transform2origin, shift2center111)
from scene.gaussian_model import GaussianModel                   # noqa: E402
os.chdir(_cwd)

cfg = json.load(open(a.config))
it = sorted(glob.glob(f"{a.model}/point_cloud/iteration_*/point_cloud.ply"),
            key=lambda p: int(p.split("iteration_")[1].split("/")[0]))[-1]
gs = GaussianModel(3)
gs.load_ply(it)
op = gs.get_opacity.detach()[:, 0]
keep = op > float(cfg.get("opacity_threshold", 0.02))
pos = gs.get_xyz.detach()[keep]
R = generate_rotation_matrices(torch.tensor(cfg.get("rotation_degree", [0.0])), cfg.get("rotation_axis", [0]))
rot = apply_rotations(pos, R)
tpos, so, omp = transform2origin(rot, float(cfg.get("scale", 1.0)))
tpos = shift2center111(tpos).cpu().numpy().astype(np.float64)
NG = tpos.shape[0]

files = sorted(glob.glob(f"{a.h5_dir}/**/sim_*.h5", recursive=True))
assert files, f"h5 가 없다: {a.h5_dir}"
files = files[::a.every]


def rd(p):
    with h5py.File(p, "r") as h:
        x = np.array(h["x"])
    return x.T if (x.ndim == 2 and x.shape[0] == 3 and x.shape[1] != 3) else x


X0 = rd(files[0]).astype(np.float64)
err = float(np.abs(X0[:NG] - tpos).max())
print(f"[대응] 가우시안 {NG}  0 프레임 입자 {X0.shape[0]}  앞 {NG} 개와 위치 차 최대 {err:.2e}", flush=True)
assert err < 1e-4, "시뮬 입자 앞부분이 거른 가우시안과 같은 순서가 아니다 -- 멈춘다"
T = len(files)
traj = np.empty((T, NG, 3), np.float32)
for i, f in enumerate(files):
    traj[i] = rd(f)[:NG]
ok = np.isfinite(traj).all((0, 2))
lo = X0.min(0)
s = float((X0.max(0) - lo).max())
off = (1.0 - ((X0.max(0) - lo) / s)) / 2.0
Gn = (X0[:NG] - lo) / s + off                      # 0 프레임 입자 위치 그대로 (rep_track2 가 목표 0 프레임과 정확히 같은지 본다)
trajn = (traj.astype(np.float64) - lo) / s + off
idx = np.nonzero(ok)[0]
print(f"[목표] 프레임 {T} (간격 {a.every})  가우시안 {NG} 중 끝까지 유한 {len(idx)}  "
      f"평균 변위 {np.linalg.norm(trajn[-1, idx] - trajn[0, idx], axis=1).mean():.4f} (정규화 단위)", flush=True)
np.savez_compressed(a.out, X0=Gn[idx].astype(np.float32), traj=trajn[:, idx].astype(np.float32),
                    idx=idx, lo=lo, s=s, src=a.h5_dir, every=a.every)
np.savez(a.aux_out, G=Gn.astype(np.float32), gidx=torch.nonzero(keep).squeeze(1).cpu().numpy(),
         lo=lo, s=s, off=off, scale_origin=float(so), mean=omp.reshape(-1).cpu().numpy(),
         model=a.model, opacity_threshold=float(cfg.get("opacity_threshold", 0.02)),
         rotation_degree=np.array(cfg.get("rotation_degree", [0.0])), rotation_axis=np.array(cfg.get("rotation_axis", [0])))
print(f"[저장] {a.out}  {a.aux_out}", flush=True)
