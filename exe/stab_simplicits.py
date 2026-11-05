"""안정 최소 서브스텝 표 (doc/pareto.md 절 S.3): Simplicits (kaolin 공식 구현 v0.18.0, kaolin.physics.simplicits easy_api) 실행기.

kaolin 은 warp 1.x 가 필요해 별도 환경 (simp) 에서 돈다 (exe/stab_bl_setup.sh 가 확인·설치).

  --mode weights  스키닝 가중치 학습: SimplicitsObject.create_trained 를 **기본 인자** 로 (핸들 10, 표본 1000, 층 6, 1 만 스텝, le 0.1, lo 1e6;
                  exe/simplicits_weights.py 와 같은 호출). 점 = 정답 0 프레임 전 입자 (공식 예제가 3DGS 를 부피 표본으로 채운 점을 쓰는 것과 같은 역할),
                  물성 = 정답 config E·ν·밀도 (논문은 물성을 주어진 값으로 씀; --region_density 면 영역 밀도), 부피 = 입자 부피 합 (PG 규약).
                  학습된 물체를 torch.save (공식 예제의 캐시와 같음).
  --mode run      SimplicitsScene 기본 설정 (뉴턴 5 회, 직접 풀이, 정규화 1e-4) 에 timestep = 프레임 간격 / N, 적분점 add_object 기본 1000,
                  중력 = config g (set_scene_gravity), L000 바닥 = set_scene_floor (높이 = config 바닥 0.48, z 축, 기본 벌점),
                  ficus 화분 = set_object_boundary_condition (상자 안 적분점 고정, 기본 벌점). 프레임마다 run_sim_step N 회 ->
                  전 입자 = 공식 예제 transform_gaussians_lbs 와 같은 점별 변환 (I + Σ w_h T_h, 가중치는 처음에 한 번) -> 위치·F (= 점별 변환의 3x3).
                  시간 = (run_sim_step N 회 + 전 입자 변환), 프레임 앞뒤 동기화 perf_counter.
공식 밖 처리 (doc/pareto.md 절 S.3): 초기 속도 -- 공식 API 에 없음 -> 장면을 다 세운 뒤 sim_z_dot (핸들 변환 속도) 에 정답 0 프레임 입자 속도를
  적분점에서 최소제곱으로 투영 (B ż ≈ v, B = 장면의 LBS 행렬). 감쇠 없음 (공식에 없음 -- ficus 의 PG 격자 감쇠는 안 넣음).

  python exe/stab_simplicits.py --mode weights --data L000.pt --tag L000 --out FIT --frames 80 --floor vel
  python exe/stab_simplicits.py --mode run --fit FIT --N 4 --data L000.pt --tag L000 --out RUN --frames 80 --floor vel [--dump --dump_F]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stab_bl_common as C                                   # noqa: E402
if os.environ.get("WARP_CACHE_PATH"):                         # 다른 warp 판 (PG 0.10 등) 과 커널 캐시를 나눔
    os.environ["WARP_CACHE_PATH"] += "_simplicits"

ap = argparse.ArgumentParser()
C.add_scene_args(ap)
ap.add_argument("--mode", choices=["weights", "run"], required=True)
ap.add_argument("--fit", default="", help="run: weights 출력 폴더")
ap.add_argument("--N", type=int, default=1)
ap.add_argument("--seed", type=int, default=0, help="적분점 무작위 표본 씨앗 (add_object 의 randperm)")
a = ap.parse_args()
FIT = os.path.abspath(a.fit or a.out)

import torch                                                 # noqa: E402
import kaolin                                                # noqa: E402
import warp as wp                                            # noqa: E402
from kaolin.physics.simplicits import SimplicitsObject, SimplicitsScene   # noqa: E402

dev = "cuda"
S = C.load_scene(a)
L, x0 = S["L"], S["x0"]
P = x0.shape[0]
pts = x0.float().to(dev)
os.makedirs(FIT, exist_ok=True)
os.makedirs(a.out, exist_ok=True)
print(f"[Simplicits] kaolin {kaolin.__version__} warp {wp.config.version} {a.tag} 입자 {P}, E {S['E']:g} ν {S['nu']} 밀도 {float(S['rho'].min()):g}~{float(S['rho'].max()):g}, "
      f"부피 {float(S['vol'].sum()):.4g}, 바닥 {S['FZ']}, 고정 상자 {len(S['BOX'])}, 프레임 {L}", flush=True)
print(f"[명령] {' '.join(sys.argv)}", flush=True)
OBJ = f"{FIT}/simp_obj.pt"
if a.mode == "weights":
    torch.manual_seed(a.seed)
    t0 = time.time()
    obj = SimplicitsObject.create_trained(pts, S["E"], S["nu"], S["rho"].float().to(dev), float(S["vol"].sum()))
    torch.save(obj, OBJ)
    json.dump(dict(tag=a.tag, kaolin=kaolin.__version__, n_pts=P, num_handles=int(obj.num_handles), sec=time.time() - t0), open(f"{FIT}/weights.json", "w"), indent=1)
    print(f"[가중치] 핸들 열 {obj.num_handles} (상수 1 열 포함), {time.time() - t0:.0f}s -> {OBJ}\nSIMP_OK", flush=True)
    sys.exit(0)

obj = torch.load(OBJ, map_location=dev, weights_only=False)
torch.manual_seed(a.seed)
dt = S["frame_dt"] / a.N
scene = SimplicitsScene(timestep=dt)
oid = scene.add_object(obj)
scene.set_scene_gravity(acc_gravity=torch.tensor([0.0, 0.0, 0.0 - S["g"][2]]))  # 정답 config g (ficus 0); kaolin 규약: 양수 = 그 축 아래 방향 (공식 예제 [0, 0, 9.8])
if S["FZ"] is not None:
    scene.set_scene_floor(floor_height=S["FZ"], floor_axis=2)
for bi, box in enumerate(S["BOX"]):
    scene.set_object_boundary_condition(oid, f"fix{bi}", lambda x, box=box: C.in_boxes(x, [box]))
# 초기 속도 (공식 밖): ż = argmin |B ż − v0(적분점)|
so = scene.get_object(oid)
from kaolin.physics.utils import warp_utilities as WU       # noqa: E402
Bd = WU._bsr_to_torch(scene.sim_B).to_dense().float()
vq = S["v0"][so.sample_indices.cpu()].float().to(dev).reshape(-1)
zd = torch.linalg.lstsq(Bd, vq[:, None]).solution[:, 0]
vres = float((Bd @ zd - vq).norm() / vq.norm().clamp(min=1e-30))
scene.sim_z_dot = wp.from_torch(zd.contiguous())
Wall = torch.cat([obj.skinning_weight_function(pts[i:i + 100000]).detach() for i in range(0, P, 100000)])   # 공식 예제: 렌더 점 가중치는 한 번
EYE = torch.eye(4, device=dev)


def deform():
    T = scene.get_object_transforms(oid)                                          # [H,4,4] 상대 변환
    A = torch.einsum("ph,hij->pij", Wall, T) + EYE                                 # transform_gaussians_lbs 와 같은 점별 변환
    return (A[:, :3, :3] @ pts[:, :, None])[..., 0] + A[:, :3, 3], A[:, :3, :3]


R = C.Roll(a, S, a.N, extra=dict(method="simplicits", dt=dt, num_qp=int(so.num_qp), handles=int(obj.num_handles), v0_lstsq_rel_res=vres))
print(f"[run] N {a.N} dt {dt:.4g}, 적분점 {so.num_qp}, 초기 속도 투영 상대 잔차 {vres:.3g}", flush=True)
for f in range(1, L + 1):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    try:
        for _ in range(a.N):
            scene.run_sim_step()
        x, F = deform()
    except Exception as ex:                                                         # noqa: BLE001  (뉴턴 안 선형 풀이 실패 등)
        R.fail(f, f"{type(ex).__name__}: {ex}"); break
    ms = C.sync_ms(t0)
    if not R.frame(f, x, F=F if a.dump_F else None, ms=ms):
        break
R.finish()
