"""PhysTwin 공식 절차로 질량-스프링의 물성을 정한다 -- 우리 목표 궤적에 그대로.

PhysTwin 은 강성을 설정 파일 값으로 쓰지 않고 두 단계로 **정한다** (사용자 지시 2026-10-14: 논문 방식 그대로):
  1 단계 optimize_cma.py: CMA-ES (20 세대) 로 전역 강성·스프링 연결 반경·이웃 수·충돌 탄성/마찰·감쇠 12 개
  2 단계 train_warp.py  : 스프링마다 log Y 를 Adam (lr 1e-3, configs/real.yaml 의 200 반복) 으로
둘 다 자기 미분가능 시뮬레이터 롤아웃과 관측의 오차를 줄인다. 우리 궤적은 모든 점의 정확한 3D 위치라
PhysTwin 의 공식 `synthetic` 데이터 경로(전체 점 궤적, 손잡이 없음, 추적 오차)를 그대로 쓴다.

맞춘 것 (PhysTwin 쪽 규약):
  - 중력은 코드에 -9.8 (z 아래) 로 박혀 있고 바닥은 z = 0 -> 궤적을 시뮬 좌표 그대로(정규화 × s, 중력 9.8 이 맞는
    단위) 두고 z 만 전 프레임 최저점이 0 이 되게 옮긴다 (물체가 바닥에 놓이는 장면). reverse_z = False.
  - FPS = 1 / 궤적 프레임 간격, num_substeps = 프레임 간격 / dt (dt 는 real.yaml 의 5e-5 그대로).
  - 질량점 = rep_track2 의 PhysTwin 표현과 같은 점 (가우시안 중심 11024 개 표본 -> 복셀 평균), 그 궤적은 같은 묶음의 평균.
  - 초기 속도 = 첫 두 프레임 차분 (공식 velocity_path 인자).
  - 학습 프레임 = 전 프레임 (우리 궤적에는 공식 데이터의 train/test 분할이 없다).
끈 것: 카메라 영상 시각화와 wandb (계산 경로와 무관; 카메라 정보가 없다).

출력 (--out_dir): pts.npy, vel.npy, optimizeCMA/optimal_params.pkl, train/best_*.pth, fit.npz (질량점 정지 위치,
스프링 짝·정지 길이·학습된 강성, 최적 매개변수) -- rep_track2 의 PhysTwin 표현이 이것을 읽는다.
"""
import argparse
import glob
import json
import os
import pickle
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--flow", required=True)
ap.add_argument("--frame_dt", type=float, required=True, help="목표 궤적 한 프레임의 시간 (원 시뮬 frame_dt × 간격)")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--phystwin", default="/home/dkta/work/PhysTwin")
ap.add_argument("--cma_iter", type=int, default=20, help="optimize_cma.py 기본값")
ap.add_argument("--train_iters", type=int, default=0, help="시험용: 0 이면 real.yaml 의 iterations (200)")
a = ap.parse_args()

os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, a.phystwin)
os.chdir(a.phystwin)                                         # configs/real.yaml 상대 경로

import numpy as np                                           # noqa: E402
import torch                                                 # noqa: E402

os.makedirs(a.out_dir, exist_ok=True)
OUT = os.path.abspath(a.out_dir)

# ---- 질량점 궤적 (rep_track2 PhysTwin 과 같은 점)
D = np.load(a.flow)
traj = D["traj"].astype(np.float64)                          # [T+1,N,3] 정규화
s = float(D["s"])
T1, N, _ = traj.shape
g = torch.Generator(device="cpu").manual_seed(0)
psel = torch.randperm(N, generator=g)[:11024].numpy()
P0 = traj[0, psel]
vox = 0.005 / 0.25
key = np.floor((P0 - P0.min(0)) / vox).astype(np.int64)
_, inv = np.unique(key, axis=0, return_inverse=True)
inv = inv.reshape(-1)
M = int(inv.max()) + 1
cnt = np.bincount(inv, minlength=M).astype(np.float64)
B = np.zeros((T1, M, 3))
for d in range(3):
    for t in range(T1):
        B[t, :, d] = np.bincount(inv, weights=traj[t, psel, d], minlength=M) / cnt
B *= s                                                       # 시뮬 좌표 (중력 9.8 단위)
ZSHIFT = float(B[..., 2].min())
B[..., 2] -= ZSHIFT                                          # 바닥 z = 0
vel = (B[1] - B[0]) / a.frame_dt
np.save(f"{OUT}/pts.npy", B.astype(np.float32))
np.save(f"{OUT}/vel.npy", vel.astype(np.float32))
print(f"[질량점] {M} 개, 프레임 {T1}, 크기 {np.ptp(B[0], 0).round(3).tolist()}, 초기 속력 최대 "
      f"{np.linalg.norm(vel, axis=1).max():.3f}", flush=True)

# ---- 렌더링·대화형 전용 import 는 빈 모듈로 (학습기 trainer_warp.py 가 영상 렌더러·키보드 입력까지 불러온다;
#      강성 결정 계산 경로는 쓰지 않는다)
import types                                                 # noqa: E402
from unittest import mock                                    # noqa: E402


class _Stub(types.ModuleType):
    def __getattr__(self, k):
        return mock.MagicMock(name=f"{self.__name__}.{k}")


for _m in ["pynput", "gaussian_splatting", "gaussian_splatting.scene", "gaussian_splatting.scene.gaussian_model",
           "gaussian_splatting.scene.cameras", "gaussian_splatting.gaussian_renderer", "gaussian_splatting.dynamic_utils",
           "gaussian_splatting.utils", "gaussian_splatting.utils.graphics_utils", "gaussian_splatting.rotation_utils",
           "gs_render"]:
    sys.modules.setdefault(_m, _Stub(_m))

# open3d 0.18 의 Vector3dVector 는 float32 배열을 받으면 세그폴트 (cma_optimize_warp.py:89 에서 실측) -> float64 로
import open3d as _o3d                                        # noqa: E402
_v3 = _o3d.utility.Vector3dVector
_o3d.utility.Vector3dVector = lambda x: _v3(np.ascontiguousarray(np.asarray(x), dtype=np.float64))

# ---- PhysTwin 설정 (configs/real.yaml 그대로 + synthetic 규약)
from qqtt.utils import cfg                                   # noqa: E402
cfg.load_from_yaml("configs/real.yaml")
cfg.data_type = "synthetic"
cfg.reverse_z = False
cfg.FPS = 1.0 / a.frame_dt
cfg.num_substeps = round(a.frame_dt / cfg.dt)
print(f"[설정] FPS {cfg.FPS:.2f}  dt {cfg.dt}  substeps {cfg.num_substeps}  iterations {cfg.iterations}  "
      f"init_spring_Y {cfg.init_spring_Y}", flush=True)

import qqtt.data.simple_data as _sd                          # noqa: E402
_sd.SimpleData.visualize_data = lambda self, visualize=False: None
from qqtt.engine.cma_optimize_warp import OptimizerCMA       # noqa: E402
from qqtt.engine.trainer_warp import InvPhyTrainerWarp       # noqa: E402
import wandb                                                 # noqa: E402
wandb.log = lambda *x, **k: None
_ef = OptimizerCMA.error_func
OptimizerCMA.error_func = lambda self, p, visualize=False, video_path=None: _ef(self, p, False, None)
InvPhyTrainerWarp.visualize_sim = lambda self, *x, **k: None

# ---- 1 단계: CMA-ES
if not os.path.exists(f"{OUT}/optimizeCMA/optimal_params.pkl"):
    opt = OptimizerCMA(data_path=f"{OUT}/pts.npy", base_dir=OUT, train_frame=T1, velocity_path=f"{OUT}/vel.npy")
    opt.optimize(max_iter=a.cma_iter)
    if os.path.exists(f"{OUT}/optimal_params.pkl"):
        os.replace(f"{OUT}/optimal_params.pkl", f"{OUT}/optimizeCMA/optimal_params.pkl")
with open(f"{OUT}/optimizeCMA/optimal_params.pkl", "rb") as f:
    optimal = pickle.load(f)
print(f"[1 단계] {optimal}", flush=True)

# ---- 2 단계: 스프링별 강성 경사 학습
cfg.load_from_yaml("configs/real.yaml")
cfg.data_type = "synthetic"
cfg.reverse_z = False
cfg.FPS = 1.0 / a.frame_dt
cfg.num_substeps = round(a.frame_dt / cfg.dt)
cfg.set_optimal_params(dict(optimal))
if a.train_iters:
    cfg.iterations = a.train_iters
tr = InvPhyTrainerWarp(data_path=f"{OUT}/pts.npy", base_dir=OUT, train_frame=T1, velocity_path=f"{OUT}/vel.npy")
if not glob.glob(f"{OUT}/train/best_*.pth"):
    tr.train()
best = sorted(glob.glob(f"{OUT}/train/best_*.pth"), key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]
ck = torch.load(best, map_location="cpu")
Y = ck["spring_Y"].float().numpy()
springs = tr.init_springs.cpu().numpy() if torch.is_tensor(tr.init_springs) else np.asarray(tr.init_springs)
L0 = tr.init_rest_lengths.cpu().numpy() if torch.is_tensor(tr.init_rest_lengths) else np.asarray(tr.init_rest_lengths)
X0 = tr.init_vertices.cpu().numpy() if torch.is_tensor(tr.init_vertices) else np.asarray(tr.init_vertices)
np.savez(f"{OUT}/fit.npz", X0=X0, springs=springs, rest_lengths=L0, spring_Y=Y, psel=psel, inv=inv, s=s,
         zshift=ZSHIFT, best=best,
         optimal=json.dumps({k: float(v) for k, v in optimal.items()}))
print(f"[2 단계] {best}  스프링 {len(springs)}  강성 중앙 {np.median(Y):.4g}  범위 {Y.min():.4g}~{Y.max():.4g}", flush=True)
print("PTFIT_OK", flush=True)
