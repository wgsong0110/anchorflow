"""MPMAvatar 공식 코드(이방성 천 MPM)로 Actor01 옷을 SMPL-X 없이 시뮬하고 저자 렌더(AO 준그림자)로 영상을 만든다.

run_demo.py 의 Trainer 를 그대로 따르되 바꾼 것만:
  - 몸: SMPL-X 대신 아바타 메쉬의 몸 부분(학습 프레임 460 자세)을 **정지** 메쉬 충돌체로
  - 옷: 몸에 붙은 joint 꼭짓점·삼각형은 속도 0 으로 고정 (공식 particle mover 에 0 속도)
  - 모래·의자 없음
  - 초기 조건 (--mode):
      a  옷 전체(고정점 제외)에 몸 좌우 방향 --vel m/s 초기 속도
      b  옷자락 앞쪽을 엉덩이 높이 축으로 --angle 도 접어 올린 모양에서 정지 상태로 놓기
         (회전각은 몸 앞뒤 좌표로 부드럽게 0 -> --angle, 뒤쪽은 그대로 -- 다리를 뚫지 않게)
  - 카메라: 회전 대신 테스트 카메라 하나 고정
물성은 저자 학습값 (demo/a1_phys_param.npz 의 D, E, H), ν·γ·κ·서브스텝은 코드 기본값.

  cd MPMAvatar && python <anchorflow>/exe/mpma_cloth.py <scripts/sim 의 인자들> --mode a --out_dir DIR
"""
import argparse
import math
import os
import sys

MP = "/home/dkta/work/MPMAvatar"
sys.path.insert(0, MP)
os.chdir(MP)

import numpy as np                                               # noqa: E402
try:
    import imageio_ffmpeg as _iff; FFMPEG = _iff.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"
import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402
from PIL import Image                                            # noqa: E402
from glob import glob                                            # noqa: E402
from tqdm import tqdm                                            # noqa: E402
import warp as wp                                                # noqa: E402

from arguments import ModelParams, PipelineParams, OptimizationParams   # noqa: E402
from scene import Scene, MeshGaussianModel                        # noqa: E402
from gaussian_renderer import render                              # noqa: E402
from utils.general_utils import read_obj                          # noqa: E402
from utils.sh_utils import eval_sh                                # noqa: E402
from utils.demo_utils import prune_faces                          # noqa: E402
from warp_mpm.mpm_data_structure import MPMStateStruct, MPMModelStruct   # noqa: E402
from warp_mpm.mpm_solver import MPMWARP                           # noqa: E402

parser = argparse.ArgumentParser()
lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
parser.add_argument("--mode", choices=["a", "b"], required=True)
parser.add_argument("--vel", type=float, default=2.0)
parser.add_argument("--angle", type=float, default=60.0)
parser.add_argument("--frames", type=int, default=50, help="25 fps")
parser.add_argument("--out_dir", required=True)
parser.add_argument("--skip_render", action="store_true")
parser.add_argument("--dump_only", action="store_true", help="초기 상태(입자·고정점·축)만 저장하고 끝 (PG 비교용)")
args = parser.parse_args(sys.argv[1:])
A = lp.extract(args); PIPE = pp.extract(args)
dev = "cuda:0"
OUT = args.out_dir; os.makedirs(OUT, exist_ok=True)


def convert_SH(shs_view, cam, pc, position, rotation=None):
    shs_view = shs_view.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = position - cam.camera_center.repeat(shs_view.shape[0], 1)
    if rotation is not None:
        dir_pp = torch.matmul(rotation, dir_pp.unsqueeze(2)).squeeze(2)
    dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    return torch.clamp_min(eval_sh(pc.active_sh_degree, shs_view, dir_pp) + 0.5, 0.0)


# ------------------------------------------------------------------ 아바타 (공식 로더 그대로)
split_idx = np.load(A.split_idx_path)
num_joint_v, num_joint_f = int(split_idx["num_joint_v"]), int(split_idx["num_joint_f"])
cloth_v = torch.tensor(split_idx["reordered_cloth_v_idx"]).long().cuda()
cloth_f = torch.tensor(split_idx["reordered_cloth_f_idx"]).long().cuda()
human_v = torch.tensor(split_idx["reordered_human_v_idx"]).long().cuda()
new_cloth_faces = torch.tensor(split_idx["new_cloth_faces"]).long().cuda()
new_human_faces = torch.tensor(split_idx["new_human_faces"]).long().cuda()
gaussians = MeshGaussianModel(A.sh_degree, device="cuda")
scene = Scene(A, gaussians, return_type="image", device="cuda", load_timestep=-1)
verts0 = (gaussians.verts_orig[0] + gaussians.verts_offset[0]).detach()          # 학습 프레임 460 메쉬
best = {k: v for k, v in np.load(os.path.join(A.dataset_dir, "demo/a1_phys_param.npz")).items()}
D_, E_, H_ = float(best["D"]), float(best["E"]), float(best["H"])
print(f"[물성] 저자 학습값 D {D_:.4g}  E {E_:.4g}  H {H_:.4g} | 기본값 ν {A.init_nu} γ {A.init_gamma} κ {A.init_kappa} "
      f"서브스텝 {A.substep}", flush=True)

cverts = verts0[cloth_v]
min_pos, max_pos = cverts.min(0)[0], cverts.max(0)[0]
mean_pos = (min_pos + max_pos) / 2.0
scale = 1.0 / (2.2 - mean_pos[1])                                               # run_demo 와 같은 월드 -> 시뮬
shift = torch.tensor([[1.0, 1.0, 1.0]]).float().cuda() - mean_pos * scale
wld2sim = lambda p: p * scale + shift                                           # noqa: E731
sim2wld = lambda p: (p - shift) / scale                                         # noqa: E731

# 몸 좌우·앞뒤 축 (y 위): 어깨 높이 몸 단면의 넓은 수평 주축 = 좌우, 발끝 쪽 = 앞
hv = verts0[human_v]
yl, yh = float(hv[:, 1].min()), float(hv[:, 1].max())
sh = hv[(hv[:, 1] > yl + 0.78 * (yh - yl)) & (hv[:, 1] < yl + 0.85 * (yh - yl))][:, [0, 2]]
ev, evec = torch.linalg.eigh(torch.cov(sh.T))
side2 = evec[:, -1]; fwd2 = evec[:, 0]
feet = hv[hv[:, 1] < yl + 0.03 * (yh - yl)][:, [0, 2]]; ank = hv[(hv[:, 1] > yl + 0.05 * (yh - yl)) & (hv[:, 1] < yl + 0.10 * (yh - yl))][:, [0, 2]]
if float(((feet.mean(0) - ank.mean(0)) * fwd2).sum()) < 0:
    fwd2 = -fwd2
SIDE = torch.tensor([float(side2[0]), 0.0, float(side2[1])], device="cuda")
FWD = torch.tensor([float(fwd2[0]), 0.0, float(fwd2[1])], device="cuda")
UP = torch.tensor([0.0, 1.0, 0.0], device="cuda")
print(f"[축] 좌우 {SIDE.cpu().numpy().round(3)} 앞 {FWD.cpu().numpy().round(3)}  키 {yh - yl:.3f} m", flush=True)

# ------------------------------------------------------------------ 초기 옷 모양 / 속도
init_w = cverts.clone()
jmask = torch.zeros(cverts.shape[0], dtype=torch.bool, device="cuda"); jmask[:num_joint_v] = True   # 몸에 붙은 꼭짓점
if args.mode == "b":
    cy0, cy1 = float(cverts[:, 1].min()), float(cverts[:, 1].max())
    hip = yl + 0.53 * (yh - yl)                                                  # 엉덩이 높이 (키의 53%)
    piv = torch.tensor([float(cverts[:, 0].mean()), hip, float(cverts[:, 2].mean())], device="cuda")
    rel = cverts - piv
    f = (rel * FWD).sum(1)
    fr, bk = float(torch.quantile(f, 0.9)), float(torch.quantile(f, 0.1))
    wgt = ((f - bk) / max(fr - bk, 1e-6)).clamp(0, 1); wgt = wgt * wgt * (3 - 2 * wgt)        # 뒤 0 -> 앞 1
    below = (cverts[:, 1] < hip) & (~jmask)
    th = math.radians(args.angle) * wgt * below.float()
    # 좌우 축 둘레로 앞-위로 접기:  r' = r cosθ + (k×r) sinθ + k (k·r)(1-cosθ),  k = -SIDE 또는 SIDE 중 앞-위로 가는 쪽
    k = SIDE if float((torch.cross(SIDE, -UP, dim=0) * FWD).sum()) > 0 else -SIDE
    ct, st = torch.cos(th)[:, None], torch.sin(th)[:, None]
    kr = torch.cross(k.expand_as(rel), rel, dim=1); kd = (rel * k).sum(1, keepdim=True)
    init_w = piv + rel * ct + kr * st + k * kd * (1 - ct)
    print(f"[초기 b] 엉덩이 높이 {hip:.3f} m 축으로 앞쪽을 최대 {args.angle}° 접음, 움직인 꼭짓점 {int((th > 1e-3).sum())}/{cverts.shape[0]}",
          flush=True)
init_verts = wld2sim(init_w)
faces = new_cloth_faces
n_elements, n_vertices = faces.shape[0], init_verts.shape[0]
n_particles = n_elements + n_vertices


def compute_dir_vol(vertices, faces, thickness):
    d1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]; d2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    d3 = d1.cross(d2); d3 /= d3.norm(dim=1, keepdim=True)
    init_dir = torch.stack([d1, d2, d3], -1)
    R11 = d1.norm(dim=1); R12 = (d1 * d2).sum(dim=1) / R11; R22 = (d2 - (R12 / R11)[:, None] * d1).norm(dim=1)
    area = 0.5 * torch.norm(d1.cross(d2), dim=1); element_vol = 0.25 * thickness * area
    vertex_vol = torch.zeros(vertices.shape[0]).cuda().float()
    vertex_vol.index_add_(0, faces.reshape(-1), element_vol[:, None].repeat(1, 3).reshape(-1))
    return init_dir, torch.stack([R11, R12, R22], -1), element_vol, vertex_vol


def rest_dir_inv_from_vf(vertices, faces):
    d1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]; d2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    R11 = d1.norm(dim=1); R12 = (d1 * d2).sum(dim=1) / R11; R22 = (d2 - (R12 / R11)[:, None] * d1).norm(dim=1)
    iR11, iR22 = 1.0 / R11, 1.0 / R22
    return torch.stack([iR11, -R12 * iR11 * iR22, iR22], -1)


rest_verts = wld2sim(cverts)                                                     # 정지 모양 = 학습 프레임 메쉬
if args.dump_only:
    torch.save(dict(init_w=init_w.cpu(), rest_w=cverts.cpu(), faces=faces.cpu(), num_joint_v=num_joint_v,
                    human_w=hv.cpu(), human_faces=new_human_faces.cpu(), SIDE=SIDE.cpu(), FWD=FWD.cpu(),
                    scale=float(scale), shift=shift.cpu(), D=D_, E=E_, H=H_), f"{OUT}/init_state.pt")
    print(f"[저장] {OUT}/init_state.pt", flush=True)
    raise SystemExit(0)

# ------------------------------------------------------------------ 시뮬 (run_demo.setup_simulation 그대로, 모래·의자·SMPL-X 빼고)
grid_size = 250
elts0 = init_verts[faces].mean(1)
pos0 = torch.cat([elts0, init_verts], 0)
_, _, ev_, vv_ = compute_dir_vol(rest_verts, faces, thickness=1e-5)
d0, rd0, _, _ = compute_dir_vol(rest_verts, faces, thickness=1e-5)
vol0 = torch.cat([ev_, vv_], 0)
wp.init()
mpm_state = MPMStateStruct(); mpm_state.init(n_particles, n_elements, n_vertices, device=dev, requires_grad=True)
p_trad = np.zeros(n_particles, np.int32); p_vert = np.zeros(n_particles, np.int32); p_vert[n_elements:] = 1
p_elem = np.zeros(n_particles, np.int32); p_elem[:n_elements] = 1
mpm_state.from_torch(pos0.clone(), vol0.clone(), torch.linalg.inv(d0).clone(), rest_dir_inv_from_vf(rest_verts, faces).clone(),
                     faces.int().clone(), p_trad, p_vert, p_elem, torch.zeros((n_particles - n_vertices, 6)).to(dev),
                     device=dev, requires_grad=True, n_grid=grid_size, grid_lim=2.0)
mpm_model = MPMModelStruct(); mpm_model.init(n_particles, device=dev, requires_grad=True)
mpm_model.init_other_params(n_grid=grid_size, grid_lim=2.0, device=dev)
material_params = {"material": "sand", "g": [0.0, -9.8, 0.0], "density": 1.0, "grid_v_damping_scale": 1.1,
                   "friction_angle": A.friction_angle}
body_x = wld2sim(hv).cpu().numpy(); body_f = new_human_faces.cpu().numpy()
solver = MPMWARP(n_particles, n_elements, n_vertices, n_grid=grid_size, grid_lim=2.0, mesh_vertices=body_x, mesh_faces=body_f,
                 num_joint_t=0, num_joint_v=num_joint_v, num_joint_f=num_joint_f, device=dev)
solver.set_parameters_dict(mpm_model, mpm_state, material_params)
dens = torch.ones(n_particles, device=dev) * D_
E = torch.ones(n_particles, device=dev) * E_
nu = torch.ones(n_particles, device=dev) * A.init_nu
gam = torch.ones(n_particles, device=dev) * A.init_gamma
kap = torch.ones(n_particles, device=dev) * A.init_kappa
mpm_state.reset_density(dens.clone(), torch.ones_like(dens).int(), dev, update_mass=True)
solver.set_E_nu_from_torch(mpm_model, E.clone(), nu.clone(), gam.clone(), kap.clone(), dev)
solver.prepare_mu_lam(mpm_model, mpm_state, dev)
solver.add_surface_collider([0., 0.1, 0.], [0., 1., 0.])
solver.add_mesh_collider(solver.mesh.id, n_grid=mpm_model.n_grid, friction=A.mesh_friction_coeff)
solver.add_particle_mover(n_grid=mpm_model.n_grid)

# demo() 의 reset_state 와 같게: 현재 방향은 초기 모양에서, 정지 방향은 학습 메쉬(높이 배율 H)에서
pd, _, _, _ = compute_dir_vol(init_verts, faces, thickness=1e-5)
rv = wld2sim(cverts)
R_inv = rest_dir_inv_from_vf(torch.stack([rv[:, 0], rv[:, 1] * H_, rv[:, 2]], 1), faces)
vel = torch.zeros_like(pos0)
if args.mode == "a":
    vv = (SIDE * args.vel * scale).float()
    vel[:] = vv
    vel[n_elements:][jmask] = 0.0
    vel[:num_joint_f] = 0.0
mpm_state.reset_state(n_vertices, pos0.clone(), pd.clone(), None, vel.clone(), tensor_R_inv=R_inv.clone(), device=dev, requires_grad=True)
mpm_state.reset_density(dens.clone(), None, dev, update_mass=True)
solver.set_E_nu_from_torch(mpm_model, E.clone(), nu.clone(), gam.clone(), kap.clone(), dev)
solver.prepare_mu_lam(mpm_model, mpm_state, dev)

dt_f = 1.0 / 25; sub = dt_f / A.substep
body_x_t = torch.as_tensor(body_x, device=dev); body_v_t = torch.zeros_like(body_x_t)
jv0 = torch.zeros(num_joint_v, 3, device=dev); jf0 = torch.zeros(num_joint_f, 3, device=dev)
vt_f, ft = [], []
with open(A.uv_path, "r") as f:
    for line in f:
        if line[:2] == "vt":
            vt_f.append(line)
        elif line[:2] == "f ":
            parts = line.strip().split()
            ft.append([int(parts[1].split("/")[1]), int(parts[2].split("/")[1]), int(parts[3].split("/")[1])])
vt_f += [f"f {v[0]}/{vt[0]} {v[1]}/{vt[1]} {v[2]}/{vt[2]}\n" for v, vt in zip(gaussians.faces.cpu().numpy() + 1, ft)]
mesh_dir = os.path.join(OUT, "uvmesh"); os.makedirs(mesh_dir, exist_ok=True)


def full_verts(cloth_w):
    v = verts0.clone(); v[cloth_v] = cloth_w
    return v


all_verts = [full_verts(init_w).detach()]
for i in tqdm(range(args.frames), desc="시뮬"):
    for s in range(A.substep):
        solver.p2g2p(mpm_model, mpm_state, sub, mesh_x=body_x_t, mesh_v=body_v_t, joint_traditional_v=None,
                     joint_verts_v=jv0, joint_faces_v=jf0, device=dev)
    pos = wp.to_torch(mpm_state.particle_x).clone()
    cw = sim2wld(pos[n_elements:]).detach()
    if not torch.isfinite(cw).all():
        print(f"[발산] 프레임 {i + 1}", flush=True); break
    all_verts.append(full_verts(cw))
for i, v in enumerate(all_verts):
    with open(os.path.join(mesh_dir, f"{i:03d}.obj"), "w") as f:
        f.writelines([f"v {x[0]} {x[1]} {x[2]}\n" for x in v.cpu().numpy()]); f.writelines(vt_f)
torch.save(dict(verts=torch.stack(all_verts).cpu(), cloth_v=cloth_v.cpu()), f"{OUT}/sim_verts.pt")
print(f"[시뮬] {len(all_verts)} 프레임 -> {OUT}", flush=True)
if args.skip_render:
    raise SystemExit(0)

# ------------------------------------------------------------------ 렌더 (demo 그대로: Blender AO -> shadow_net, 카메라는 고정)
rc = os.system(f"{os.environ.get('BLENDER', 'blender')} -b -P blender/bake.py -- --output_path {OUT} > {OUT}/bake.log 2>&1")
print(f"[AO 굽기] rc={rc}", flush=True)
ao = [np.array(Image.open(p).convert("L")).astype(np.float32) / 255. for p in sorted(glob(os.path.join(OUT, "aomap/*.png")))]
ao = torch.from_numpy(np.array(ao)).unsqueeze(1).contiguous().float().cuda()
with torch.no_grad():
    prune_faces(gaussians, os.path.join(A.dataset_dir, "demo/a1_prune_f_idx.npy"))
cam = scene.test_dataset.camera_list[0]; cam_idx = scene.test_camera_index[0]
bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
imgdir = os.path.join(OUT, "frames"); os.makedirs(imgdir, exist_ok=True)
with torch.no_grad():
    for i in tqdm(range(len(all_verts)), desc="렌더"):
        gaussians.set_mesh_by_verts(all_verts[i])
        shadow_map = gaussians.shadow_net(ao[i:i + 1] if ao.shape[0] > i else ao[-1:])["shadow_map"]
        shadow = F.grid_sample(shadow_map, gaussians.uv_coord, mode="bilinear", align_corners=False).squeeze()[..., None][gaussians.binding]
        col = shadow * convert_SH(gaussians.get_features, cam, gaussians, gaussians.get_xyz)
        pkg = render(cam, gaussians, PIPE, bg, override_color=col)
        img = pkg["render"] * torch.exp(gaussians.cam_m[cam_idx])[:, None, None] + gaussians.cam_c[cam_idx][:, None, None]
        img = img * pkg["mask"] + (1.0 - pkg["mask"])
        Image.fromarray((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(f"{imgdir}/{i:04d}.png")
os.system(f"{FFMPEG} -y -hide_banner -loglevel error -framerate 25 -i {imgdir}/%04d.png -pix_fmt yuv420p "
          f"-vf scale='trunc(iw/2)*2:trunc(ih/2)*2' {OUT}/video.mp4")
print(f"[영상] {OUT}/video.mp4", flush=True)
