"""수박 파괴 궤적(h5 프레임)을 GaussianFluent 의 전처리·카메라 그대로, 배경 없이 흰 바탕으로 렌더한다.

GF 공식 러너(gs_simulation_watermelon.py)와 Fracture-GS 재구현(collision_mpm.py) 두 궤적을 **같은 렌더러**로
그리기 위한 것. 렌더 규약은 GF 러너를 따른다:
  - 가우시안 = 채우기 결과의 앞 gs_num 입자 (불투명도·sim_area 거른 뒤 순서)
  - 위치: undoshift2center111 -> undotransform2origin -> 역회전,  공분산: F Σ0 Fᵀ / scale²
  - 색: convert_SH (보는 방향 + F 의 극분해 회전),  카메라: get_camera_view (config 의 방위·고도·반경)
GF 러너와 다른 점: garden 배경을 넣지 않는다, logJp > 0.4 입자를 숨기는 규칙은 쓰지 않는다 (두 궤적 모두 같게).

  cd GaussianFluent && python <anchorflow>/exe/wm_render.py --model_path M --config C --sim DIR --out V.mp4
"""
import argparse
import glob
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--gf", default="/home/dkta/work/GaussianFluent")
ap.add_argument("--model_path", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--sim", required=True, help="h5 프레임 폴더 (GF: x + f_tensor, collision_mpm: x + F)")
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=32)
ap.add_argument("--frames", type=int, default=0)
ap.add_argument("--meta", default="", help="여러 벌 장면 (wm_pair_init.py 의 meta.json): 벌마다 시뮬 이동량")
ap.add_argument("--radius_scale", type=float, default=1.0, help="카메라 반경 배수 (여러 벌이 다 들어오게)")
a = ap.parse_args()
sys.path.insert(0, a.gf); sys.path.insert(1, os.path.join(a.gf, "gaussian-splatting"))
os.chdir(a.gf)

import h5py                                                      # noqa: E402
import imageio.v2 as imageio                                     # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from utils.decode_param import decode_param_json                 # noqa: E402
from utils.transformation_utils import *                         # noqa: E402,F403
from utils.camera_view_utils import *                            # noqa: E402,F403
from utils.render_utils import *                                 # noqa: E402,F403

dev = "cuda"
material_params, bc_params, time_params, pp, camera_params = decode_param_json(a.config)
gaussians = load_checkpoint(a.model_path)                        # noqa: F405
pipeline = PipelineParamsNoparse(); pipeline.compute_cov3D_python = True   # noqa: F405
background = torch.tensor([1, 1, 1], dtype=torch.float32, device=dev)
params = load_params_from_gs(gaussians, pipeline)                # noqa: F405
init_pos, init_cov, init_opacity, init_shs = params["pos"], params["cov3D_precomp"], params["opacity"], params["shs"]
init_screen_points = params["screen_points"]
m = init_opacity[:, 0] > pp["opacity_threshold"]
init_pos, init_cov, init_opacity, init_shs, init_screen_points = (init_pos[m], init_cov[m], init_opacity[m],
                                                                  init_shs[m], init_screen_points[m])
for k in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
    setattr(gaussians, k, getattr(gaussians, k)[m])
R = generate_rotation_matrices(torch.tensor(pp["rotation_degree"]), pp["rotation_axis"])   # noqa: F405
rp = apply_rotations(init_pos, R)                                                           # noqa: F405
if pp["sim_area"] is not None:
    b = pp["sim_area"]; m = torch.ones(rp.shape[0], dtype=torch.bool, device=dev)
    for i in range(3):
        m = m & (rp[:, i] > b[2 * i]) & (rp[:, i] < b[2 * i + 1])
    rp, init_cov, init_opacity, init_shs = rp[m], init_cov[m], init_opacity[m], init_shs[m]
tp, scale_origin, mean_pos = transform2origin(rp)                 # noqa: F405
tp = shift2center111(tp)                                          # noqa: F405
init_cov = apply_cov_rotations(init_cov, R) * scale_origin * scale_origin   # noqa: F405
gs_num = tp.shape[0]
C0 = torch.zeros(gs_num, 3, 3, device=dev)
c = init_cov
C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c[:, 0], c[:, 1], c[:, 2]
C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c[:, 3], c[:, 4], c[:, 5]
C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c[:, 1], c[:, 2], c[:, 4]
vc, oc = get_center_view_worldspace_and_observant_coordinate(    # noqa: F405
    torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda(),
    torch.tensor(camera_params["mpm_space_vertical_upward_axis"]).reshape((1, 3)).cuda(),
    R, scale_origin, mean_pos)
fs = sorted(glob.glob(f"{a.sim}/sim_*.h5"))
if a.frames:
    fs = fs[:a.frames + 1]
with h5py.File(fs[0], "r") as h:
    x0 = np.array(h["x"]); x0 = x0.T if x0.shape[0] == 3 else x0
import json as _js
SHIFTS = np.array(_js.load(open(a.meta))["shifts"]) if a.meta else np.zeros((1, 3))
K = len(SHIFTS); NP1 = x0.shape[0] // K
SC = SHIFTS.mean(0)                                                # 모든 벌에 같은 이동을 빼서 상대 배치를 그대로 둔다
GIDX = np.concatenate([np.arange(gs_num) + k * NP1 for k in range(K)])
for k in range(K):
    d0 = float(np.abs(x0[k * NP1:k * NP1 + gs_num] - SHIFTS[k] - tp.cpu().numpy()).max())
    print(f"[대응] 벌 {k}: 가우시안 {gs_num}, 입자 {NP1}, 0 프레임 가우시안 위치 차이 최대 {d0:.2e}", flush=True)
init_shs = init_shs.repeat(K, 1, 1) if init_shs.dim() == 3 else init_shs.repeat(K, 1)
init_opacity = init_opacity.repeat(K, 1); C0 = C0.repeat(K, 1, 1)
WR = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8)
fr_dir = os.path.splitext(a.out)[0] + "_frames"; os.makedirs(fr_dir, exist_ok=True)
for t, fp in enumerate(fs):
    with h5py.File(fp, "r") as h:
        x = np.array(h["x"]); x = x.T if x.shape[0] == 3 else x
        F = np.array(h["f_tensor"] if "f_tensor" in h else h["F"]).reshape(-1, 3, 3)
    X = torch.as_tensor(x[GIDX] - SC, device=dev).float(); Fg = torch.as_tensor(F[GIDX], device=dev).float()
    ok = torch.isfinite(X).all(1) & torch.isfinite(Fg).all(2).all(1)
    cov = Fg @ C0 @ Fg.transpose(1, 2)
    c6 = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
    U, _, Vh = torch.linalg.svd(Fg)
    rot = U @ Vh
    pos = apply_inverse_rotations(undotransform2origin(undoshift2center111(X), scale_origin, mean_pos), R)   # noqa: F405
    c6 = apply_inverse_cov_rotations(c6 / (scale_origin * scale_origin), R)                                   # noqa: F405
    cam = get_camera_view(a.model_path, default_camera_index=camera_params["default_camera_index"],          # noqa: F405
                          center_view_world_space=vc, observant_coordinates=oc, show_hint=camera_params["show_hint"],
                          init_azimuthm=camera_params["init_azimuthm"], init_elevation=camera_params["init_elevation"],
                          init_radius=camera_params["init_radius"] * a.radius_scale, move_camera=camera_params["move_camera"],
                          current_frame=t, delta_a=camera_params["delta_a"], delta_e=camera_params["delta_e"],
                          delta_r=camera_params["delta_r"], width=600, height=600)
    rast = initialize_resterize(cam, gaussians, pipeline, background)          # noqa: F405
    col = convert_SH(init_shs[ok], cam, gaussians, pos[ok], rot[ok])           # noqa: F405
    out = rast(means3D=pos[ok], means2D=torch.zeros_like(pos[ok]), shs=None, colors_precomp=col.float(),
               opacities=init_opacity[ok], scales=None, rotations=None, cov3D_precomp=c6[ok])
    img = out[0]
    im = (img.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    WR.append_data(im)
    if t % 10 == 0:
        imageio.imwrite(f"{fr_dir}/{t:03d}.png", im)
WR.close()
print(f"[영상] {a.out} ({len(fs)} 프레임)", flush=True)
