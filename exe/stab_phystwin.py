"""안정 최소 서브스텝 표 (doc/pareto.md 절 S.3): PhysTwin 공식 코드 (Jianghanxiao/PhysTwin 81c7187) 실행기.

  --mode fit  물성을 논문 절차로 정함 (사용자 지시: 베이스라인 물성은 논문 절차대로) -- 공식 1 단계 optimize_cma.py (CMA-ES 20 세대,
              전역 강성·스프링 반경·이웃 수·조작점 반경·이웃 수·충돌·감쇠 12 개) -> 2 단계 train_warp.py (스프링별 log Y, Adam lr 1e-3,
              real.yaml 200 반복). 궤적 = 정답 입자 궤적 전 프레임 (L000 은 공식 synthetic 경로 = 전체 점 궤적·추적 손실,
              ficus 는 화분 고정 때문에 공식 조작점이 있는 real 경로 = chamfer + 추적 + 가속도 손실, 가시성·추적 유효 전부 참).
              서브스텝 = 공식 dt 5e-5 를 프레임 간격에 맞춘 N0 = round(프레임 간격 / 5e-5) (dt = 프레임 간격 / N0).
  --mode run  맞춘 물성 그대로 프레임당 서브스텝 N (dt = 프레임 간격 / N) 으로 롤아웃 (공식 InvPhyTrainerWarp pure_inference + test() 의
              체크포인트 적재 + visualize_sim 의 프레임 루프) -> 전 입자 위치 = 공식 대화형 데모의 가우시안 이동 (interpolate_motions_speedup,
              이웃 16, 첫 프레임에 한 번 정한 이웃) -> 오차·발산·시간 (forward_graph + 보간, 프레임 앞뒤 동기화 perf_counter)·덤프 (F = 누적 회전).

공식 밖 처리 (doc/pareto.md 절 S.3 목록):
  (1) 단위: PhysTwin 의 반경 범위 (0.01~0.05 m)·dt·복셀 0.005 m 는 실측 물체 ~0.25 m 기준 -> 길이를 K = 0.25 / (0 프레임 상자 최대 변) 배,
      시간을 c = sqrt(K |g| / 9.8) 배 (닮음 변환: 중력이 코드에 박힌 9.8 과 같아짐; exe/phystwin_fit.py 와 같은 처리). 바닥 z = 0 이 되게 z 이동
      (L000: config 바닥 0.48 -> 0; ficus: 바닥 없음 -> 전 프레임 최저점보다 아래).
  (2) ficus 화분 고정 = 공식 조작점 (controller points) 을 화분 상자 안 질량점으로, 모든 프레임 같은 위치 (새 기제 아님, 손잡이를 고정으로).
      그래서 ficus 는 조작점이 있는 공식 real 경로 (손실 정의가 L000 과 다름).
  (3) 초기 속도 = 정답 0 프레임 입자 속도 (공식 velocity_path 인자; real 경로는 이 인자가 없어 시뮬레이터의 초기 속도 배열에 같은 값을 넣음).
  (4) 질량점 = 정답 입자에서 무작위 11024 개 (rep_track2·fps_phystwin 과 같은 수, 씨앗 0) -> 공식 data_process_sample 복셀 0.005 m 처음 점.
      가우시안 이동의 뼈대에 ficus 조작점 (움직임 0) 을 더함 (화분 입자가 화분 조작점을 따라가게).
  (5) 렌더·대화형 전용 import (pynput·gaussian_splatting 렌더러·pyrender) 는 빈 모듈, ipdb.set_trace 는 예외 (보간 SVD 실패 = 발산으로 셈).

  python exe/stab_phystwin.py --mode fit --data L000.pt --tag L000 --out FIT --frames 80 --floor vel
  python exe/stab_phystwin.py --mode run --fit FIT --N 167 --data L000.pt --tag L000 --out RUN --frames 80 --floor vel [--dump --dump_F]
"""
import argparse
import glob
import importlib.util
import json
import os
import pickle
import sys
import time
import types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stab_bl_common as C                                   # noqa: E402
if os.environ.get("WARP_CACHE_PATH"):                         # 다른 warp 판 (PG 0.10 등) 과 커널 캐시를 나눔
    os.environ["WARP_CACHE_PATH"] += "_phystwin"

ap = argparse.ArgumentParser()
C.add_scene_args(ap)
ap.add_argument("--mode", choices=["fit", "run"], required=True)
ap.add_argument("--phystwin", default="/home/dkta/work/PhysTwin")
ap.add_argument("--fit", default="", help="run: fit 출력 폴더")
ap.add_argument("--N", type=int, default=0, help="run: 프레임당 서브스텝 (0 = 맞춤 때의 N0)")
ap.add_argument("--nobs", type=int, default=11024)
ap.add_argument("--cma_iter", type=int, default=20, help="optimize_cma.py 기본값")
ap.add_argument("--train_iters", type=int, default=0, help="0 = real.yaml iterations (200)")
ap.add_argument("--knn", type=int, default=16, help="가우시안 이동 이웃 (공식 데모 16)")
a = ap.parse_args()
a.out = os.path.abspath(a.out)
FIT = os.path.abspath(a.fit or a.out)
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np                                           # noqa: E402
import torch                                                 # noqa: E402

S = C.load_scene(a)
L, x0 = S["L"], S["x0"]
P = x0.shape[0]
# ---------- (1) 단위·좌표 ----------
ext_side = float((x0.max(0).values - x0.min(0).values).max())
K = 0.25 / ext_side
c = (K * abs(S["g"][2]) / 9.8) ** 0.5
FDT = S["frame_dt"] * c
if S["FZ"] is not None:
    Z0 = S["FZ"]
else:
    Z0 = float(S["X"][..., 2].min()) - 0.05 * ext_side
N0 = max(1, round(FDT / 5e-5))


def to_pt(x):
    """시뮬 좌표 -> PhysTwin 좌표."""
    y = x.double().clone()
    y[..., 2] -= Z0
    return y * K


def from_pt(y):
    x = y.double() / K
    x[..., 2] += Z0
    return x


# ---------- (4) 질량점 ----------
g_ = torch.Generator().manual_seed(0)
sel = torch.randperm(P, generator=g_)[:min(a.nobs, P)]
Q0 = to_pt(x0[sel])
key = torch.floor((Q0 - Q0.min(0).values) / 0.005).long()
seen, first = set(), []
for i, k in enumerate(map(tuple, key.tolist())):                # data_process_sample: 순서대로 복셀마다 처음 점
    if k not in seen:
        seen.add(k); first.append(i)
psel = sel[torch.as_tensor(first)]
ctrl = C.in_boxes(x0[psel], S["BOX"]) if S["BOX"] else torch.zeros(len(psel), dtype=torch.bool)
REAL = bool(ctrl.any())
obj_idx, ctl_idx = psel[~ctrl], psel[ctrl]
B = to_pt(S["X"][:, obj_idx])                                  # [L+1, M, 3]
CP = to_pt(x0[ctl_idx])                                        # [Nc, 3] 고정
vel = S["v0"][obj_idx] * (K / c)
print(f"[PhysTwin] {a.tag} 입자 {P}, 질량점 {B.shape[1]} + 조작점 {CP.shape[0]} ({'real' if REAL else 'synthetic'} 경로), K {K:.4g} 시간 {c:.4g} 배, "
      f"프레임 {FDT:.5g} s (PhysTwin 단위) N0 {N0}, 바닥 z {Z0:.4g}, 초기 속력 최대 {float(vel.norm(dim=1).max()):.3g}", flush=True)
print(f"[명령] {' '.join(sys.argv)}", flush=True)
os.makedirs(FIT, exist_ok=True)
os.makedirs(a.out, exist_ok=True)
DP = f"{FIT}/final_data.pkl" if REAL else f"{FIT}/pts.npy"
if a.mode == "fit":
    np.save(f"{FIT}/vel.npy", vel.float().numpy())
    if REAL:
        T1, M = B.shape[0], B.shape[1]
        pickle.dump(dict(object_points=B.float().numpy(), object_colors=np.ones((T1, M, 3), np.float32),
                         object_visibilities=np.ones((T1, M), bool), object_motions_valid=np.ones((T1, M), bool),
                         controller_points=CP[None].repeat(T1, 1, 1).float().numpy(), surface_points=np.zeros((0, 3)),
                         interior_points=np.zeros((0, 3))), open(DP, "wb"))
    else:
        np.save(DP, B.float().numpy())

# ---------- (5) 렌더·대화형 import 막기 ----------


class _Stub(types.ModuleType):
    def __getattr__(self, k):
        return mock.MagicMock(name=f"{self.__name__}.{k}")


sys.path.insert(0, a.phystwin)
os.chdir(a.phystwin)                                           # configs/real.yaml 상대 경로
for _m in ["pynput", "gaussian_splatting", "gaussian_splatting.scene", "gaussian_splatting.scene.gaussian_model",
           "gaussian_splatting.scene.cameras", "gaussian_splatting.gaussian_renderer", "gaussian_splatting.utils",
           "gaussian_splatting.utils.graphics_utils", "gaussian_splatting.rotation_utils", "gs_render"]:
    sys.modules.setdefault(_m, _Stub(_m))
try:
    import pyrender                                            # noqa: F401
except Exception:                                              # noqa: BLE001  (화면 없는 노드 -- 시각화 전용)
    sys.modules["pyrender"] = _Stub("pyrender")
try:
    import wandb                                               # noqa: F401
except Exception:                                              # noqa: BLE001  (기록 전용 -- WANDB_MODE disabled 와 같은 효과)
    _nop = lambda *x, **k: None                                 # noqa: E731
    sys.modules["wandb"] = types.SimpleNamespace(init=_nop, log=_nop, finish=_nop, Video=_nop)
_sp = importlib.util.spec_from_file_location("gaussian_splatting.dynamic_utils", f"{a.phystwin}/gaussian_splatting/dynamic_utils.py")
DU = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(DU)
sys.modules["gaussian_splatting.dynamic_utils"] = DU


class _IpdbStop(RuntimeError):
    pass


def _no_trace(*x, **k):
    raise _IpdbStop("dynamic_utils 의 ipdb.set_trace (SVD 실패·det 이상)")


sys.modules["ipdb"] = types.SimpleNamespace(set_trace=_no_trace)
import open3d as _o3d                                          # noqa: E402
_v3 = _o3d.utility.Vector3dVector
_o3d.utility.Vector3dVector = lambda x: _v3(np.ascontiguousarray(np.asarray(x), dtype=np.float64))   # phystwin_fit 과 같음 (float32 세그폴트)
import warp as wp                                              # noqa: E402
from qqtt.utils import cfg                                     # noqa: E402
import qqtt.data.simple_data as _sd                            # noqa: E402
import qqtt.data.real_data as _rd                              # noqa: E402
_sd.SimpleData.visualize_data = lambda self, visualize=False: None
_rd.RealData.visualize_data = lambda self, visualize=False, save_gt=True: None
from qqtt.engine.cma_optimize_warp import OptimizerCMA         # noqa: E402
from qqtt.engine.trainer_warp import InvPhyTrainerWarp         # noqa: E402
import wandb                                                   # noqa: E402
wandb.log = lambda *x, **k: None
wandb.Video = lambda *x, **k: None
InvPhyTrainerWarp.visualize_sim = lambda self, *x, **k: None
_ef = OptimizerCMA.error_func
OptimizerCMA.error_func = lambda self, p, visualize=False, video_path=None: _ef(self, p, False, None)


def set_cfg(N):
    cfg.load_from_yaml("configs/real.yaml")
    cfg.data_type = "real" if REAL else "synthetic"
    cfg.reverse_z = False
    cfg.FPS = 1.0 / FDT
    cfg.num_substeps = int(N)
    cfg.dt = FDT / int(N)


def set_vel(sim):
    """(3) real 경로는 velocity_path 를 안 받음 -> 시뮬레이터 초기 속도 배열에 직접."""
    if REAL:
        wp.copy(sim.wp_init_velocities, wp.from_torch(vel.float().cuda().contiguous(), dtype=wp.vec3))


VP = f"{FIT}/vel.npy"
if a.mode == "fit":
    set_cfg(N0)
    t0 = time.time()
    if not os.path.exists(f"{FIT}/optimizeCMA/optimal_params.pkl"):
        opt = OptimizerCMA(data_path=DP, base_dir=FIT, train_frame=L + 1, velocity_path=None if REAL else VP)
        if REAL:
            opt.init_velocities = vel.float().cuda()
        opt.optimize(max_iter=a.cma_iter)
        os.replace(f"{FIT}/optimal_params.pkl", f"{FIT}/optimizeCMA/optimal_params.pkl")
    optimal = pickle.load(open(f"{FIT}/optimizeCMA/optimal_params.pkl", "rb"))
    print(f"[1 단계 CMA-ES] {optimal} ({time.time() - t0:.0f}s)", flush=True)
    set_cfg(N0)
    cfg.set_optimal_params(dict(optimal))
    if a.train_iters:
        cfg.iterations = a.train_iters
    tr = InvPhyTrainerWarp(data_path=DP, base_dir=FIT, train_frame=L + 1, velocity_path=None if REAL else VP)
    set_vel(tr.simulator)
    if not glob.glob(f"{FIT}/train/best_*.pth"):
        tr.train()
    best = sorted(glob.glob(f"{FIT}/train/best_*.pth"), key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]
    Y = torch.load(best, map_location="cpu")["spring_Y"].float()
    json.dump(dict(tag=a.tag, N0=N0, K=K, c=c, Z0=Z0, FDT=FDT, real=REAL, n_mass=int(B.shape[1]), n_ctrl=int(CP.shape[0]),
                   optimal={k: float(v) for k, v in optimal.items()}, best=best, iterations=int(cfg.iterations), lr=float(cfg.base_lr),
                   spring_Y_median=float(Y.median()), spring_Y_min=float(Y.min()), spring_Y_max=float(Y.max()), n_springs=int(Y.numel()),
                   sec=time.time() - t0), open(f"{FIT}/fit.json", "w"), indent=1)
    print(f"[2 단계 경사 학습] {best} 스프링 {Y.numel()} 강성 중앙 {float(Y.median()):.4g} ({time.time() - t0:.0f}s)\nPTFIT_OK", flush=True)
    sys.exit(0)

# ---------- run ----------
fj = json.load(open(f"{FIT}/fit.json"))
N = a.N or N0
set_cfg(N)
cfg.set_optimal_params(dict(fj["optimal"]))
tr = InvPhyTrainerWarp(data_path=DP, base_dir=a.out, train_frame=L + 1, velocity_path=None if REAL else VP, pure_inference_mode=True)
ck = torch.load(fj["best"], map_location=cfg.device)                                   # test() 와 같은 적재
sim = tr.simulator
assert len(ck["spring_Y"]) == sim.n_springs, "맞춤 때와 스프링 연결이 다름"
sim.set_spring_Y(torch.log(ck["spring_Y"]).detach().clone())
sim.set_collide(ck["collide_elas"].detach().clone(), ck["collide_fric"].detach().clone())
sim.set_collide_object(ck["collide_object_elas"].detach().clone(), ck["collide_object_fric"].detach().clone())
set_vel(sim)
dev = "cuda"
xyz = to_pt(x0).float().to(dev)                                 # 전 입자 (공식 데모의 가우시안 자리)
quat = torch.zeros(P, 4, device=dev); quat[:, 0] = 1
CPd = CP.float().to(dev)
sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
prev = torch.cat([wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).clone(), CPd])
rel = DU.get_topk_indices(prev, K=a.knn)
widx = torch.cat([DU.knn_weights_sparse(prev, xyz[i:i + 4000], K=a.knn)[1] for i in range(0, P, 4000)])   # 공식 knn_weights_sparse 를 나눠서 (CPU 메모리)
R = C.Roll(a, S, N, extra=dict(method="phystwin", N0=N0, dt=cfg.dt, real=REAL, n_mass=int(B.shape[1]), n_ctrl=int(CP.shape[0])))
print(f"[run] N {N} dt {cfg.dt:.4g} (PhysTwin 단위), 스프링 {sim.n_springs}", flush=True)
for f in range(1, L + 1):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if REAL:
        sim.set_controller_target(f, pure_inference=True)
    if sim.object_collision_flag:
        sim.update_collision_graph()
    wp.capture_launch(sim.forward_graph)
    x = torch.cat([wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False), CPd])
    sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
    if not torch.isfinite(x).all():
        R.fail(f, "질량점 비유한"); break
    try:
        w = DU.calc_weights_vals_from_indices(prev, xyz, widx)
        xyz, quat, _ = DU.interpolate_motions_speedup(bones=prev, motions=x - prev, relations=rel, weights=w, weights_indices=widx, xyz=xyz, quat=quat)
    except Exception as ex:                                     # noqa: BLE001
        R.fail(f, f"가우시안 이동 실패 ({type(ex).__name__}: {ex})"); break
    prev = x.clone()
    ms = C.sync_ms(t0)
    F = DU.quat2mat(quat) if a.dump_F else None
    if not R.frame(f, from_pt(xyz), F=F, ms=ms):
        break
R.finish()
