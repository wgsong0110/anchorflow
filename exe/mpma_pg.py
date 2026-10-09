"""MPMAvatar 비교용: 같은 Actor01 옷을 PhysGaussian 공식 MPM(등방성 jelly)으로 같은 초기 조건에서 시뮬·렌더한다.

PG 에는 천(코드원) 모델도 메쉬 충돌체도 없으므로:
  - 입자 = 아바타 가우시안 (옷 면에 붙은 것 + 몸 면에 붙은 것), 부피 = dx^3 / 칸 안 개수 (PG 정의)
  - 재질: PG jelly (fixed corotated), E·밀도는 MPMAvatar 저자 학습값, ν 0.3
  - 몸 가우시안과 옷의 joint 면(몸에 붙은 면) 가우시안은 매 서브스텝 처음 위치·속도 0 으로 묶는다
  - 초기 조건 a/b 는 mpma_cloth.py 와 같은 축·각도 (가우시안에 같은 회전 / 속도)
  - 렌더: MPMAvatar 렌더와 같은 카메라·색 보정, 준그림자는 첫 프레임 값 고정 (입자 구름이라 AO 를 다시 구울 수 없다)
세 단계 (환경이 다르다):
  --stage dump    (mpma env)  아바타에서 PG 입력을 뽑는다
  --stage sim     (af env)    PG 공식 솔버로 시뮬
  --stage render  (mpma env)  렌더
"""
import argparse
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["dump", "sim", "render"], required=True)
ap.add_argument("--mode", choices=["a", "b"], required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--vel", type=float, default=2.0)
ap.add_argument("--angle", type=float, default=60.0)
ap.add_argument("--frames", type=int, default=50)
ap.add_argument("--substep", type=int, default=400)
ap.add_argument("--n_grid", type=int, default=250)
a, rest = ap.parse_known_args()
os.makedirs(a.work, exist_ok=True)
import numpy as np                                               # noqa: E402
try:
    import imageio_ffmpeg as _iff; FFMPEG = _iff.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"
import torch                                                     # noqa: E402

if a.stage in ("dump", "render"):
    MP = "/home/dkta/work/MPMAvatar"; sys.path.insert(0, MP); os.chdir(MP)
    import torch.nn.functional as F                              # noqa: E402
    from arguments import ModelParams, PipelineParams            # noqa: E402
    from scene import Scene, MeshGaussianModel                    # noqa: E402
    from utils.sh_utils import eval_sh                            # noqa: E402
    from utils.demo_utils import prune_faces                      # noqa: E402
    p2 = argparse.ArgumentParser(); lp = ModelParams(p2); pp = PipelineParams(p2)
    A2 = p2.parse_args(rest); A = lp.extract(A2); PIPE = pp.extract(A2)
    gaussians = MeshGaussianModel(A.sh_degree, device="cuda")
    scene = Scene(A, gaussians, return_type="image", device="cuda", load_timestep=-1)
    split_idx = np.load(A.split_idx_path)
    cloth_f = torch.tensor(split_idx["reordered_cloth_f_idx"]).long().cuda()
    human_f = torch.tensor(split_idx["reordered_human_f_idx"]).long().cuda()
    num_joint_f = int(split_idx["num_joint_f"])
    cam = scene.test_dataset.camera_list[0]; cam_idx = scene.test_camera_index[0]

if a.stage == "dump":
    st = torch.load(f"{a.work}/../init_state.pt")              # mpma_cloth --dump_only 결과 (축·회전 기준)
    verts0 = (gaussians.verts_orig[0] + gaussians.verts_offset[0]).detach()
    gaussians.set_mesh_by_verts(verts0)
    prune_faces(gaussians, os.path.join(A.dataset_dir, "demo/a1_prune_f_idx.npy"))
    xyz = gaussians.get_xyz.detach(); cov = gaussians.get_covariance().detach()
    op = gaussians.get_opacity.detach(); bind = gaussians.binding
    keep = op[:, 0] > 0.02
    is_cloth = torch.isin(bind, cloth_f); is_joint = torch.isin(bind, cloth_f[:num_joint_f])
    # 첫 프레임 준그림자·색 (고정 카메라)
    _ap = f"{a.work}/../aomap/000.png"                          # MPMAvatar 실행이 구운 첫 프레임 AO
    ao0 = None
    if os.path.exists(_ap):
        from PIL import Image as _I
        ao0 = torch.from_numpy(np.array(_I.open(_ap).convert("L")).astype(np.float32) / 255.)[None, None].cuda()
    print(f"[준그림자] 첫 프레임 AO {'있음' if ao0 is not None else '없음 (그림자 없이)'}", flush=True)
    shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
    d = xyz - cam.camera_center[None]; d = d / d.norm(dim=1, keepdim=True)
    col = torch.clamp_min(eval_sh(gaussians.active_sh_degree, shs_view, d) + 0.5, 0.0)
    if ao0 is not None:
        sm = gaussians.shadow_net(ao0)["shadow_map"]
        col = col * F.grid_sample(sm, gaussians.uv_coord, mode="bilinear", align_corners=False).squeeze()[..., None][bind]
    X = xyz.clone()
    SIDE, FWD = st["SIDE"].cuda(), st["FWD"].cuda()
    UP = torch.tensor([0.0, 1.0, 0.0], device="cuda")
    V = torch.zeros_like(X)
    if a.mode == "b":                                          # mpma_cloth 와 같은 접기 (가우시안 위치에)
        hv = st["human_w"].cuda(); yl, yh = float(hv[:, 1].min()), float(hv[:, 1].max())
        cv = st["rest_w"].cuda()
        hip = yl + 0.53 * (yh - yl)
        piv = torch.tensor([float(cv[:, 0].mean()), hip, float(cv[:, 2].mean())], device="cuda")
        fc = ((cv - piv) * FWD).sum(1); fr, bk = float(torch.quantile(fc, 0.9)), float(torch.quantile(fc, 0.1))
        rel = X - piv; f = (rel * FWD).sum(1)
        w = ((f - bk) / max(fr - bk, 1e-6)).clamp(0, 1); w = w * w * (3 - 2 * w)
        th = math.radians(a.angle) * w * ((X[:, 1] < hip) & is_cloth & ~is_joint).float()
        k = SIDE if float((torch.cross(SIDE, -UP, dim=0) * FWD).sum()) > 0 else -SIDE
        ct, s_ = torch.cos(th)[:, None], torch.sin(th)[:, None]
        X = piv + rel * ct + torch.cross(k.expand_as(rel), rel, dim=1) * s_ + k * (rel * k).sum(1, keepdim=True) * (1 - ct)
        # 공분산도 같은 회전으로
        R = torch.eye(3, device="cuda").expand(X.shape[0], 3, 3).clone()
        K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device="cuda")
        R = R + s_[..., None] * K + (1 - ct)[..., None] * (K @ K)
        C = torch.zeros(cov.shape[0], 3, 3, device="cuda")
        C[:, 0, 0], C[:, 0, 1], C[:, 0, 2], C[:, 1, 1], C[:, 1, 2], C[:, 2, 2] = cov.T
        C[:, 1, 0], C[:, 2, 0], C[:, 2, 1] = cov[:, 1], cov[:, 2], cov[:, 4]
        C = R @ C @ R.transpose(1, 2)
        cov = torch.stack([C[:, 0, 0], C[:, 0, 1], C[:, 0, 2], C[:, 1, 1], C[:, 1, 2], C[:, 2, 2]], 1)
    else:
        V[is_cloth & ~is_joint] = SIDE * a.vel
    sc, shf = float(st["scale"]), st["shift"].cuda()
    pin = (~is_cloth) | is_joint
    torch.save(dict(x=(X * sc + shf)[keep].cpu(), v=(V * sc)[keep].cpu(), cov=(cov * sc * sc)[keep].cpu(),
                    pin=pin[keep].cpu(), col=col[keep].cpu(), op=op[keep].cpu(), scale=sc, shift=shf.cpu(),
                    E=st["E"], D=st["D"]), f"{a.work}/pg_in.pt")
    print(f"[PG 입력] 가우시안 {int(keep.sum())} (옷 {int((is_cloth & keep).sum())}, 고정 {int((pin & keep).sum())})", flush=True)

elif a.stage == "sim":
    PG = "/home/dkta/work/i-physgaussian"; sys.path.insert(0, PG); os.chdir(PG)
    import warp as wp
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
    wp.init()
    I = torch.load(f"{a.work}/pg_in.pt")
    x = I["x"].cuda().float(); n = x.shape[0]; dx = 2.0 / a.n_grid
    cell = (x / dx).floor().long(); key = (cell[:, 0] * a.n_grid + cell[:, 1]) * a.n_grid + cell[:, 2]
    _, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    vol = dx ** 3 / cnt[inv].float()                                        # PG get_particle_volume 와 같은 정의
    sol = MPM_Simulator_WARP(10)
    sol.load_initial_data_from_torch(x, vol, I["cov"].cuda().float(), n_grid=a.n_grid, grid_lim=2.0)
    sol.set_parameters_dict({"material": "jelly", "E": float(I["E"]), "nu": 0.3, "density": float(I["D"]),
                             "g": [0.0, -9.8, 0.0], "grid_v_damping_scale": 1.1})
    sol.add_surface_collider((0.0, 0.1, 0.0), (0.0, 1.0, 0.0), "sticky", 0.0)
    sol.finalize_mu_lam()
    sol.import_particle_v_from_torch(I["v"].cuda().float())
    pin = wp.from_torch(I["pin"].cuda().int().contiguous()); x0 = wp.from_torch(x.contiguous(), dtype=wp.vec3)

    @wp.kernel
    def hold(px: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3), x0: wp.array(dtype=wp.vec3), m: wp.array(dtype=int)):
        p = wp.tid()
        if m[p] == 1:
            px[p] = x0[p]; pv[p] = wp.vec3(0.0, 0.0, 0.0)

    sub = (1.0 / 25) / a.substep
    XS, CS = [x.cpu()], [I["cov"].float()]
    from tqdm import tqdm
    for f in tqdm(range(a.frames), desc="PG 시뮬"):
        for s in range(a.substep):
            wp.launch(hold, dim=n, inputs=[sol.mpm_state.particle_x, sol.mpm_state.particle_v, x0, pin])
            sol.p2g2p(f, sub)
        xt = sol.export_particle_x_to_torch().clone().cpu()
        if not torch.isfinite(xt).all():
            print(f"[발산] 프레임 {f + 1}", flush=True); break
        XS.append(xt); CS.append(sol.export_particle_cov_to_torch().clone().cpu().reshape(-1, 6))
    torch.save(dict(x=torch.stack(XS), cov=torch.stack(CS)), f"{a.work}/pg_traj.pt")
    print(f"[PG 시뮬] {len(XS)} 프레임", flush=True)

else:
    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
    from PIL import Image
    I = torch.load(f"{a.work}/pg_in.pt"); T = torch.load(f"{a.work}/pg_traj.pt")
    sc, shf = I["scale"], I["shift"].cuda()
    col, op = I["col"].cuda(), I["op"].cuda()
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    rs = GaussianRasterizationSettings(image_height=int(cam.image_height), image_width=int(cam.image_width),
                                       tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5), bg=bg,
                                       scale_modifier=1.0, viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
                                       sh_degree=3, campos=cam.camera_center, prefiltered=False, debug=False)
    rast = GaussianRasterizer(raster_settings=rs)
    imgdir = f"{a.work}/frames"; os.makedirs(imgdir, exist_ok=True)
    with torch.no_grad():
        for i in range(T["x"].shape[0]):
            X = (T["x"][i].cuda() - shf) / sc; C = T["cov"][i].cuda() / (sc * sc)
            out = rast(means3D=X, means2D=torch.zeros_like(X), shs=None, colors_precomp=col, opacities=op,
                       scales=None, rotations=None, cov3Ds_precomp=C)
            img, mask = out[0], out[3]
            img = img * torch.exp(gaussians.cam_m[cam_idx])[:, None, None] + gaussians.cam_c[cam_idx][:, None, None]
            img = img * mask + (1.0 - mask)
            Image.fromarray((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(f"{imgdir}/{i:04d}.png")
    os.system(f"{FFMPEG} -y -hide_banner -loglevel error -framerate 25 -i {imgdir}/%04d.png -pix_fmt yuv420p "
              f"-vf scale='trunc(iw/2)*2:trunc(ih/2)*2' {a.work}/video.mp4")
    print(f"[영상] {a.work}/video.mp4", flush=True)
