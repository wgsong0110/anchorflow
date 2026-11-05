"""안정 최소 서브스텝 표 (doc/pareto.md 절 S.3): Spring-Gaus 공식 코드 (Colmar-zlicheng/Spring-Gaus 62a1bb5) 실행기.

  --mode prep   exe/stab_mv_render.py 다시점 영상 -> 공식 mpm_synthetic 형식 데이터 (camera.json + cam_<i>/<프레임>.png) 와 설정 yaml
                (DEFAULT = 공식 config/mpm_synthetic/default.yaml 상속, DATA 절만 우리 씬). 학습 프레임 = 전 프레임 (N_FRAME = FRAME_ALL = L+1).
  --mode train  공식 train.py 의 정적 (3DGS 30k) -> 속도 (100 반복) -> 동역학 (300 반복, N_STEP 100 에서 반복 100·200 에 +50) 를 그대로
                (train.py 를 모듈로 불러 __main__ 블록과 같은 순서로 부름; 끝의 공식 평가 eval_fitting·eval_prediction 은 그쪽 자체 지표라
                건너뜀 -- 예전 벤치에서 open3d 세그폴트로 죽던 곳). 끝나면 SG_TRAIN_DONE.
  --mode run    학습 결과 (공식 test 규약: --dy_reload <exp>/checkpoints_dynamic/checkpoint/Spring_Mass.pth.tar) 를 get_simulator 로 적재,
                n_step = N (dt = 프레임 간격 / N) 만 바꿔 롤아웃 -> 우리 입자 = 공식 결속 set_all_particle (K_BINDING 16, 1/√d 가중) 로
                앵커에 묶고 이동 = 결속 가중 앵커 변위 (공식 interpolate 의 velocity 단계 식) -> 오차·발산 (공식 is_nan 포함)·시간·덤프 (F = I:
                Spring-Gaus 가우시안은 등방·회전 없음).
공식 밖 처리 (doc/pareto.md 절 S.3):
  (1) 단위: 공식 합성 데이터는 물체 ~0.4 를 [0,1] 상자 안에 (XYZ 0.3~0.7), 중력 9.8 -> 길이 Ks = 0.4 / (0 프레임 상자 최대 변) 배 (중심 0.5),
      시간 c = sqrt(Ks |g| / 9.8) 배 (닮음 변환; 영상은 카메라 위치만 같이 옮기면 그대로). 중력 z 축 (공식 코드가 BC 법선 축으로 z 지원).
      config 중력 0 (ficus) 이면 c = 1 (길이만) 이고 단계 설정 G = [0, 0, 0] (정답 config 대로; 사용자 결정 11-05).
  (2) ficus: 바닥 없음 -> BC 바닥을 전 프레임 최저점 아래에 (닿지 않음). 화분 상자 안 앵커는 서브스텝마다 처음 위치·속도 0 (공식에 점별 고정 없음).
      초기 충격이 국소라 공식 균일 INIT_VELOCITY 로 표현 안 됨 -> 속도 단계를 건너뛰고 (공식 규약: init_velocity_100.json 이 있으면 건너뜀, 0 으로 씀)
      앵커 초기 속도 init_v = 정답 0 프레임 입자 속도의 앵커 최근접 8 개 평균 (학습 안 함).
  (3) 시각화 open3d ply 덤프 간격만 끔 (VIZ_INTERVAL 999999; 예전 노드에서 첫 o3d 호출 세그폴트 -- exe/mk_sg_config.py 와 같음).

  python exe/stab_sg.py --mode prep  --mv MVDIR --fit FIT --data L000.pt --tag L000 --out FIT --frames 80 --floor vel
  python exe/stab_sg.py --mode train --fit FIT --data L000.pt --tag L000 --out FIT --frames 80 --floor vel
  python exe/stab_sg.py --mode run --fit FIT --N 100 --data L000.pt --tag L000 --out RUN --frames 80 --floor vel [--dump --dump_F]
"""
import argparse
import glob
import json
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stab_bl_common as C                                   # noqa: E402

ap = argparse.ArgumentParser()
C.add_scene_args(ap)
ap.add_argument("--mode", choices=["prep", "train", "run"], required=True)
ap.add_argument("--sg", default="/home/dkta/work/Spring-Gaus")
ap.add_argument("--fit", required=True, help="학습 작업 폴더 (데이터·설정·checkpoints·exp)")
ap.add_argument("--mv", default="", help="prep: exe/stab_mv_render.py 출력")
ap.add_argument("--N", type=int, default=0, help="run: n_step (0 = 학습 끝 n_step)")
ap.add_argument("--smoke", type=int, default=0, help="prep: 연기 시험 -- 학습 반복 수만 아주 작게 (정적 300·속도 2·동역학 2·다듬기 10; 결과는 경로 확인용)")
a = ap.parse_args()
a.out = os.path.abspath(a.out)
FIT = os.path.abspath(a.fit)
SG = os.path.abspath(a.sg)

import numpy as np                                           # noqa: E402
import torch                                                 # noqa: E402

S = C.load_scene(a)
L, x0 = S["L"], S["x0"]
os.makedirs(FIT, exist_ok=True)
print(f"[명령] {' '.join(sys.argv)}", flush=True)
XF = f"{FIT}/af_xf.json"
if a.mode == "prep":
    mvm = json.load(open(f"{a.mv}/mv_meta.json"))
    assert mvm["n_frames"] == L + 1, f"다시점 영상 프레임 수 {mvm['n_frames']} != L+1 {L + 1}"
    side = float((x0.max(0).values - x0.min(0).values).max())
    Ks = 0.4 / side
    c = C.time_scale(Ks, S["g"])                                                                # 중력 0 (ficus) 이면 1
    ctr = [float(q) for q in mvm["ctr"]]
    json.dump(dict(Ks=Ks, c=c, ctr=ctr, side=side), open(XF, "w"), indent=1)
xf = json.load(open(XF))
Ks, c, CTR = xf["Ks"], xf["c"], torch.tensor(xf["ctr"], dtype=torch.float64)


def to_sg(x):
    return (x.double().cpu() - CTR) * Ks + 0.5


def from_sg(y):
    return (y.double().cpu() - 0.5) / Ks + CTR


DT = S["frame_dt"] * c
OBJ = a.tag
CFG = f"{FIT}/{OBJ}.yaml"
BCJ = f"{FIT}/af_bc.json"
if a.mode == "prep":
    D = f"{FIT}/data/{OBJ}"
    os.makedirs(D, exist_ok=True)
    cams = json.load(open(f"{a.mv}/cams_sim.json"))
    for cm in cams:
        M = np.array(cm["c2w"]); M[:3, 3] = (M[:3, 3] - np.array(xf["ctr"])) * Ks + 0.5          # 닮음 변환: 위치만 (회전·초점 그대로)
        cm["c2w"] = M.tolist()
        lk = f"{D}/{cm['camera']}"
        if not os.path.exists(lk):
            os.symlink(os.path.abspath(f"{a.mv}/{cm['camera']}"), lk)
    json.dump(cams, open(f"{D}/camera.json", "w"), indent=1)
    Xs = to_sg(S["X"].reshape(-1, 3)).reshape(S["X"].shape)
    lo, hi = Xs.reshape(-1, 3).min(0).values, Xs.reshape(-1, 3).max(0).values
    pad = 0.1 * float((hi - lo).max())
    lo, hi = (lo - pad).tolist(), (hi + pad).tolist()
    fix = bool(S["BOX"])
    if S["FZ"] is not None:
        gz = float(to_sg(torch.tensor([[0.0, 0.0, S["FZ"]]]))[0, 2])
        zmin = Xs[..., 2].amin(dim=1)                                                           # 프레임별 최저점
        hit = next((f for f in range(1, L + 1) if float(zmin[f]) <= gz + 0.005), L)
    else:
        gz = float(Xs[..., 2].min()) - 0.05
        hit = L
    hit = max(2, min(hit, L))
    GV = "[0, 0, -9.8]" if abs(S["g"][2]) > 0 else "[0, 0, 0]"                               # 닮음 변환 뒤 중력 (c 로 9.8 에 맞춤) / config 중력 0 이면 0
    VIT = 2 if a.smoke else 100                                                                 # 공식 VELOCITY.ITERATIONS
    SMK = "" if not a.smoke else f"""
STATIC:
  ITERATIONS: 300
  TEST_ITERATIONS: [300]
  SAVE_ITERATIONS: [300]
  CHECKPOINT_ITERATIONS: [300]
"""
    CK = f"{FIT}/checkpoints/{OBJ}"
    os.makedirs(CK, exist_ok=True)
    y = f"""DEFAULT: {SG}/config/mpm_synthetic/default.yaml
CHECKPOINTS_ROOT: {CK}

DATA:
  TYPE: MPM_Synthetic
  DATA_ROOT: {FIT}/data
  OBJ_NAME: {OBJ}
  N_CAM: {len(cams)}
  N_FRAME: {L + 1}
  FRAME_ALL: {L + 1}
  HIT_FRAME: {hit}
  H: {cams[0]['height']}
  W: {cams[0]['width']}
  WHITE_BKG: False
  XYZ_MIN: {[round(q, 5) for q in lo]}
  XYZ_MAX: {[round(q, 5) for q in hi]}
  BC: [[[0, 0, {gz:.6f}], [0, 0, 1]]]

  DT: {DT:.8f}
  GLOBAL_M: 1
  GLOBAL_K: 1000
  GLOBAL_DAMP: 0.1

  EVAL_FREQ: {1.0 / DT:.6f}
  EVAL_FRAME: {L + 1}

VELOCITY:
  G: {GV}
  VIZ_INTERVAL: 999999
""" + (f"""  ITERATIONS: {VIT}
  SAVE_INTERVAL: 1
""" if a.smoke else "") + f"""
DYNAMIC:
  G: {GV}
  VIZ_INTERVAL: 999999
""" + ("""  ITERATIONS: 2
  ITER_REFINE: 10
  SAVE_INTERVAL: 1
""" if a.smoke else "") + SMK
    open(CFG, "w").write(y)
    bc = dict(fix_boxes_sg=[], v0=None)
    if fix:
        for (pt, sz) in S["BOX"]:
            bc["fix_boxes_sg"].append((to_sg(torch.tensor([pt]))[0].tolist(), [float(q) * Ks for q in sz]))
        torch.save(dict(x0=to_sg(x0).float(), v0=(S["v0"] * (Ks / c)).float()), f"{FIT}/af_v0.pt")
        bc["v0"] = f"{FIT}/af_v0.pt"
        json.dump([0.0, 0.0, 0.0], open(f"{CK}/init_velocity_{VIT}.json", "w"))                 # 속도 단계 건너뜀 (공식 규약), 균일 속도 0
    json.dump(bc, open(BCJ, "w"), indent=1)
    print(f"[prep] {CFG}\n{y}\n[BC] {bc}\nSGPREP_OK", flush=True)
    sys.exit(0)

# ---------- 공식 코드 불러오기 (train·run) ----------
os.chdir(FIT)                                                                                   # exp/·checkpoints 는 작업 폴더에
sys.path.insert(0, SG)
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
BC = json.load(open(BCJ))
import lib.models.spring_mass.Spring_Mass as SMM                                              # noqa: E402
if BC["fix_boxes_sg"] or BC["v0"]:                                                              # (2) ficus 화분 고정·초기 속도장
    _V = torch.load(BC["v0"])
    _init0, _step0 = SMM.Spring_Mass.initialize, SMM.Spring_Mass.step

    def _initialize(self, xyz):
        _init0(self, xyz)
        X0 = _V["x0"].to(xyz.device)
        V0 = _V["v0"].to(xyz.device)
        idx = torch.cat([torch.cdist(self.init_xyz[i:i + 256].float(), X0).topk(8, largest=False).indices for i in range(0, self.n_points, 256)])
        self.af_fix = C.in_boxes(self.init_xyz.float(), BC["fix_boxes_sg"])
        v = V0[idx].mean(1)
        v[self.af_fix] = 0.0
        self.init_v = v.to(self.init_xyz.dtype)
        print(f"[ficus 처리] 앵커 {self.n_points}, 고정 {int(self.af_fix.sum())}, 초기 속력 최대 {float(v.norm(dim=1).max()):.3g}", flush=True)

    def _step(self, xyz, v, K, m, rebound_k, fric_k, damp, dt):
        xyz, v = _step0(self, xyz, v, K, m, rebound_k, fric_k, damp, dt)
        fx = self.af_fix[:, None]
        return torch.where(fx, self.init_xyz, xyz), torch.where(fx, torch.zeros_like(v), v)
    SMM.Spring_Mass.initialize = _initialize
    SMM.Spring_Mass.step = _step
import train as T                                                                               # noqa: E402  (공식 train.py, __main__ 블록은 안 돎)

if a.mode == "train":
    sys.argv = ["train.py", "-g", GPU, "--cfg", CFG, "--exp_id", f"stab_{OBJ}", "--eval_cam", "0"]
    arg = T.config_parser()
    cfg = T.get_config_merge_default(config_file=arg.cfg, arg=arg)
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id                                            # 공식과 같음 (이미 같은 값)
    import open3d as o3d                                                                         # noqa: E402
    T.o3d = o3d
    T.setup_seed(cfg.SEED)
    recorder = T.Recorder(arg.exp_id, cfg, rank=0, time_f=time.time())
    T.recorder = recorder
    T.cfg, T.arg = cfg, arg                                                                      # 공식 __main__ 의 모듈 전역 (render_step·eval 이 전역 cfg 를 읽음; 절 S.8 셋째 연기 실패)
    exp_path = f"{recorder.exp_id}_{recorder.timestamp}"
    os.makedirs("checkpoints", exist_ok=True)
    static = os.path.exists(os.path.join(cfg.CHECKPOINTS_ROOT, "static_gaussians"))
    scene = T.Scene(cfg, exp_path, shuffle=False, load_static=not static)
    T.scene = scene                                                                              # 공식 __main__ 전역 scene (render_step)
    t0 = time.time()
    if not static:
        T.logger.info("Begin Train Static...")
        T.train_static(arg, cfg, scene, cfg.CHECKPOINTS_ROOT)
    if not os.path.exists(os.path.join(cfg.CHECKPOINTS_ROOT, f"init_velocity_{cfg.VELOCITY.ITERATIONS}.json")):
        T.logger.info("Begin Train Velocity...")
        T.train_velocity(arg, cfg, scene, recorder)
    T.eval_fitting = T.eval_prediction = lambda *x, **k: T.logger.info("[stab] 공식 평가 건너뜀")
    T.train_dynamic(arg, cfg, scene, recorder)
    cks = sorted(glob.glob(f"{FIT}/exp/{exp_path}/checkpoints_dynamic/checkpoint/Spring_Mass.pth.tar"))
    assert cks, "동역학 체크포인트 없음"
    json.dump(dict(tag=OBJ, exp=f"{FIT}/exp/{exp_path}", dy_reload=cks[-1], n_step=json.load(open(f"{FIT}/exp/{exp_path}/checkpoints_dynamic/checkpoint/dy_n_step.json")),
                   init_velocity=json.load(open(os.path.join(cfg.CHECKPOINTS_ROOT, f"init_velocity_{cfg.VELOCITY.ITERATIONS}.json"))), sec=time.time() - t0),
              open(f"{FIT}/sg_train.json", "w"), indent=1)
    print(f"[학습 끝] {cks[-1]} ({time.time() - t0:.0f}s)\nSG_TRAIN_DONE", flush=True)
    sys.exit(0)

# ---------- run ----------
tj = json.load(open(f"{FIT}/sg_train.json"))
sys.argv = ["test.py", "-g", GPU, "--cfg", CFG, "--exp_id", f"stabrun_{OBJ}", "--dy_reload", tj["dy_reload"]]
arg = T.config_parser()
cfg = T.get_config_merge_default(config_file=arg.cfg, arg=arg)


class _Scene:
    """get_simulator 가 쓰는 set_gaussians 만 (합성 데이터 경로 = load_ply; 영상 적재 생략)."""

    def __init__(s, cfg):
        s.cfg = cfg

    def set_gaussians(s, gaussians, load_path=None):
        s.gaussians = gaussians
        gaussians.load_ply(load_path)


load_velocity = json.load(open(os.path.join(cfg.CHECKPOINTS_ROOT, f"init_velocity_{cfg.VELOCITY.ITERATIONS}.json")))
sim, gaussians = T.get_simulator(arg, cfg, _Scene(cfg), cfg_stage=cfg.DYNAMIC, init_velocity=load_velocity, load_g=None)
sim.eval()
NT = int(sim.n_step)
N = a.N or NT
dev = "cuda"
ours0 = to_sg(x0).float().to(dev)
sim.set_all_particle(ours0)                                                                     # 공식 결속을 우리 입자에
sim.stage = "velocity"                                                                          # interpolate 의 변위 식 (forward 의 다른 곳엔 영향 없음)
sim.n_step = N
R = C.Roll(a, S, N, extra=dict(method="springgaus", n_step_train=NT, dt=sim.dt / N, n_anchor=int(sim.n_points), init_velocity=load_velocity,
                               dy_reload=tj["dy_reload"]))
print(f"[run] 학습 n_step {NT} -> {N}, 앵커 {sim.n_points}, 입자 {ours0.shape[0]}, 프레임 간격 {sim.dt:.5g} (Spring-Gaus 단위)", flush=True)
with torch.no_grad():
    xyz_all, xyz, v = ours0.clone(), sim.init_xyz.detach().clone(), sim.init_v.detach().clone()
    for f in range(1, L + 1):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        xyz_all, xyz, v, is_nan = sim(xyz_all, xyz, v, f)
        ms = C.sync_ms(t0)
        if bool(is_nan):
            R.fail(f, "공식 is_nan (앵커 NaN)"); break
        if not R.frame(f, from_sg(xyz_all), F=None, ms=ms):
            break
R.finish()
