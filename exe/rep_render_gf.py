"""repflow A GF 장면: 각 방법의 위치 궤적 + 변형 기울기 (rep_track_sg --save_traj --traj32 --save_F K) 를 GF 공식 렌더 경로로 그린다 (exe/gfs_render.py 와 같은 코드·같은 카메라·같은 전처리, 정답 h5 대신 방법 결과를 넣음).
가우시안 = 채우기 결과의 앞 gs_num 입자; 위치 y (정규화) -> 시뮬 좌표 x = (y - off) s + lo -> undoshift2center111 -> undotransform2origin -> 역회전, 공분산 F Σ0 Fᵀ / scale², 색 convert_SH (F 극분해 회전).
phystwin 은 F 대신 공식 뼈 회전 R 로 (rep_track 의 렌더 규약). 정답은 exe/gfs_render.py 로 같은 PNG 이름 규칙 (<out>_frames/<t:03d>.png) 으로 그린다.
  cd GaussianFluent && python <anchorflow>/exe/rep_render_gf.py --model_path M --config C --res R.npz --aux A.npz --out V.mp4 [--count 1]"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--gf", default="/home/dkta/work/GaussianFluent")
ap.add_argument("--model_path", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--res", default="")
ap.add_argument("--aux", default="")
ap.add_argument("--out", default="")
ap.add_argument("--fps", type=int, default=32)
ap.add_argument("--count", type=int, default=0, help="1 이면 가우시안 수 gs_num 만 출력하고 끝")
a = ap.parse_args()
sys.path.insert(0, a.gf); sys.path.insert(1, os.path.join(a.gf, "gaussian-splatting"))
os.chdir(a.gf)

import imageio.v2 as imageio                                     # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from utils.decode_param import decode_param_json                 # noqa: E402
from utils.transformation_utils import *                         # noqa: E402,F403
from utils.camera_view_utils import *                            # noqa: E402,F403
from utils.render_utils import *                                 # noqa: E402,F403
from scene.gaussian_model import GaussianModel                   # noqa: E402
from utils.system_utils import searchForMaxIteration             # noqa: E402


class PipelineParamsNoparse:                                     # GF gs_simulation.py 와 같다
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


torch.set_grad_enabled(False)
dev = "cuda"
material_params, bc_params, time_params, pp, camera_params = decode_param_json(a.config)
d_ = os.path.join(a.model_path, "point_cloud")
gaussians = GaussianModel(3); gaussians.load_ply(os.path.join(d_, f"iteration_{searchForMaxIteration(d_)}", "point_cloud.ply"))
pipeline = PipelineParamsNoparse(); pipeline.compute_cov3D_python = True
background = torch.tensor([1, 1, 1], dtype=torch.float32, device=dev)
params = load_params_from_gs(gaussians, pipeline)                # noqa: F405
init_pos, init_cov, init_opacity, init_shs = params["pos"], params["cov3D_precomp"], params["opacity"], params["shs"]
m = init_opacity[:, 0] > pp["opacity_threshold"]
init_pos, init_cov, init_opacity, init_shs = init_pos[m], init_cov[m], init_opacity[m], init_shs[m]
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
if a.count:
    print(f"GS_NUM {gs_num}", flush=True); sys.exit(0)
C0 = torch.zeros(gs_num, 3, 3, device=dev); c = init_cov
C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c[:, 0], c[:, 1], c[:, 2]
C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c[:, 3], c[:, 4], c[:, 5]
C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c[:, 1], c[:, 2], c[:, 4]
vc, oc = get_center_view_worldspace_and_observant_coordinate(    # noqa: F405
    torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda(),
    torch.tensor(camera_params["mpm_space_vertical_upward_axis"]).reshape((1, 3)).cuda(), R, scale_origin, mean_pos)
Z = np.load(a.res); AX = np.load(a.aux, allow_pickle=True)
lo, s_, off = AX["lo"], float(AX["s"]), AX["off"]
TR, FS = Z["traj"], Z["Fs"]                                       # [T, N, 3] (정규화, 프레임 1..T), [T, K, 9] float16 (앞 K 점의 F 또는 phystwin R)
assert FS.shape[1] >= gs_num, ("--save_F K 가 가우시안 수보다 작다", FS.shape, gs_num)
T1 = TR.shape[0] + 1
x0_sim = (AX["G"][:gs_num].astype(np.float64) - off) * s_ + lo
d0 = float(np.abs(x0_sim - tp.cpu().numpy()).max())
print(f"[대응] 가우시안 {gs_num}, 입자 {AX['G'].shape[0]}, 0 프레임 가우시안 위치 차이 최대 {d0:.2e} (정규화 {d0 / s_:.2e})", flush=True)
fr_dir = os.path.splitext(a.out)[0] + "_frames"; os.makedirs(fr_dir, exist_ok=True)
WR = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8)
for t in range(T1):
    if t == 0:
        x = x0_sim; Fg = np.tile(np.eye(3, dtype=np.float32), (gs_num, 1, 1))
    else:
        x = (TR[t - 1][:gs_num].astype(np.float64) - off) * s_ + lo
        Fg = FS[t - 1][:gs_num].astype(np.float32).reshape(-1, 3, 3)
    X = torch.as_tensor(x, device=dev).float(); Fg = torch.as_tensor(Fg, device=dev).float()
    ok = torch.isfinite(X).all(1) & torch.isfinite(Fg).all(2).all(1)
    cov = Fg @ C0 @ Fg.transpose(1, 2)
    c6 = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
    U, _, Vh = torch.linalg.svd(torch.nan_to_num(Fg, nan=0.0, posinf=0.0, neginf=0.0).cpu() + 1e-12 * torch.eye(3))
    rot = (U @ Vh).to(dev)
    pos = apply_inverse_rotations(undotransform2origin(undoshift2center111(X), scale_origin, mean_pos), R)   # noqa: F405
    c6 = apply_inverse_cov_rotations(c6 / (scale_origin * scale_origin), R)                                   # noqa: F405
    cam = get_camera_view(a.model_path, default_camera_index=camera_params["default_camera_index"],          # noqa: F405
                          center_view_world_space=vc, observant_coordinates=oc, show_hint=camera_params["show_hint"],
                          init_azimuthm=camera_params["init_azimuthm"], init_elevation=camera_params["init_elevation"],
                          init_radius=camera_params["init_radius"], move_camera=camera_params["move_camera"],
                          current_frame=t, delta_a=camera_params["delta_a"], delta_e=camera_params["delta_e"],
                          delta_r=camera_params["delta_r"], width=600, height=600)
    rast = initialize_resterize(cam, gaussians, pipeline, background)          # noqa: F405
    col = convert_SH(init_shs[ok], cam, gaussians, pos[ok], rot[ok])           # noqa: F405
    out = rast(means3D=pos[ok], means2D=torch.zeros_like(pos[ok]), shs=None, colors_precomp=col.float(),
               opacities=init_opacity[ok], scales=None, rotations=None, cov3D_precomp=c6[ok])
    im = (out[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    WR.append_data(im); imageio.imwrite(f"{fr_dir}/{t:03d}.png", im)
WR.close()
print(f"[영상] {a.out} ({T1} 프레임, 비유한 {int((~ok).sum())} 마지막 프레임)", flush=True)
