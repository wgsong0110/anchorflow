#!/usr/bin/env python
"""Profiling script: time each component of one training step."""
import sys, os, math, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np
from plyfile import PlyData
from omegaconf import OmegaConf

sys.path.append("/workspace/SC-GS")
from scene.gaussian_model import GaussianModel
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

def ts(label, t0):
    torch.cuda.synchronize()
    dt = time.time() - t0
    mem = torch.cuda.memory_allocated() // (1024**2)
    print(f"  [{label}] {dt*1000:.0f} ms   gpu_alloc={mem} MiB")
    return time.time()

# ── config ──────────────────────────────────────────────────────────────────
PLY  = "/workspace/lego_canonical.ply"
COL  = "/workspace/kitchen_colmap"
CFG  = "cfg/anchorflow_kitchen.yaml"
dev  = "cuda"

cfg = OmegaConf.load(CFG)
T   = cfg.model.n_frames   # 25
res = cfg.model.res        # 192

# ── Gaussian attrs ───────────────────────────────────────────────────────────
print("\n=== Loading Gaussians ===")
t0 = time.time()
names = [p.name for p in PlyData.read(PLY)["vertex"].properties if p.name.startswith("f_rest_")]
sh    = min(int(math.sqrt((len(names) + 3) // 3)) - 1 if names else 0, 3)
g     = GaussianModel(sh); g.load_ply(PLY); g.active_sh_degree = sh
SH_C0 = 0.28209479177387814
canonical_xyz = g.get_xyz.detach().to(dev)
opacities     = g.get_opacity.detach().to(dev)
scales        = g.get_scaling.detach().to(dev)
rotations     = g.get_rotation.detach().to(dev)
colors        = (SH_C0 * g._features_dc.detach()[:,0,:] + 0.5).clamp(0,1).to(dev)
bg            = torch.zeros(3, device=dev)
t0 = ts("load gaussians", t0)
print(f"  G={canonical_xyz.shape[0]}")

# ── Camera ───────────────────────────────────────────────────────────────────
class Cam:
    def __init__(self, R, T, fovx, fovy, W, H):
        self.image_width, self.image_height = W, H
        self.FoVx, self.FoVy = fovx, fovy
        self.znear, self.zfar = 0.01, 100.0
        w2v  = torch.tensor(getWorld2View2(R, T)).T.cuda()
        proj = getProjectionMatrix(self.znear, self.zfar, fovx, fovy).T.cuda()
        self.world_view_transform = w2v
        self.full_proj_transform  = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center        = w2v.inverse()[3, :3]
        self.tanfovx = math.tan(fovx * 0.5)
        self.tanfovy = math.tan(fovy * 0.5)

extr  = read_extrinsics_binary(f"{COL}/sparse/0/images.bin")
intr  = read_intrinsics_binary(f"{COL}/sparse/0/cameras.bin")
items = sorted(extr.values(), key=lambda x: x.name)
im = items[0]; cam_data = intr[im.camera_id]
f  = cam_data.params[0]
R  = qvec2rotmat(im.qvec).astype(np.float32)
Tv = np.array(im.tvec, dtype=np.float32)
fov = 2 * math.atan(min(cam_data.width, cam_data.height) / (2 * f))
cam = Cam(R, Tv, fov, fov, res, res)

def render_gaussians(cam, xyz):
    cfg2 = GaussianRasterizationSettings(
        image_height=cam.image_height, image_width=cam.image_width,
        tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
        bg=bg, scale_modifier=1.0,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        sh_degree=0, campos=cam.camera_center,
        prefiltered=False, debug=False,
    )
    rast = GaussianRasterizer(raster_settings=cfg2)
    m2d  = torch.zeros_like(xyz, requires_grad=True)
    img, _ = rast(means3D=xyz, means2D=m2d, shs=None, colors_precomp=colors,
                  opacities=opacities, scales=scales, rotations=rotations, cov3D_precomp=None)
    return img

# ── Render warmup ─────────────────────────────────────────────────────────────
print("\n=== Rasterizer warmup ===")
t0 = time.time()
_ = render_gaussians(cam, canonical_xyz)
torch.cuda.synchronize()
t0 = ts("render warmup", t0)

# ── tokens_to_nodes ───────────────────────────────────────────────────────────
print("\n=== tokens_to_nodes ===")
t0 = time.time()
from anchorflow.tokens_to_nodes import tokens_to_nodes
import anchorflow.tokens_to_nodes as t2n_mod

cameras_t2n = []
for i in range(min(4, len(items))):
    im2 = items[i]; cd2 = intr[im2.camera_id]
    f2  = cd2.params[0]
    R2  = qvec2rotmat(im2.qvec).astype(np.float32)
    T2  = np.array(im2.tvec, dtype=np.float32)
    fov2 = 2 * math.atan(min(cd2.width, cd2.height) / (2 * f2))
    cameras_t2n.append(Cam(R2, T2, fov2, fov2, res, res))

def render_fn(c):
    with torch.no_grad():
        return render_gaussians(c, canonical_xyz)

node_pos = tokens_to_nodes(canonical_xyz, opacities, render_fn, cameras_t2n,
                           n_nodes=cfg.model.n_nodes, device=dev)
if t2n_mod._dino_model is not None:
    del t2n_mod._dino_model; t2n_mod._dino_model = None
import gc; gc.collect(); torch.cuda.empty_cache()
t0 = ts("tokens_to_nodes (total)", t0)
print(f"  nodes={node_pos.shape[0]}")

# ── NodeFlow init ─────────────────────────────────────────────────────────────
print("\n=== NodeFlow init ===")
t0 = time.time()
from anchorflow.nodeflow import NodeFlow
model = NodeFlow(
    canonical_xyz=canonical_xyz, node_positions=node_pos,
    n_nodes=cfg.model.n_nodes, n_frames=T,
    hidden=cfg.model.hidden, n_gnn_layers=cfg.model.n_gnn_layers,
    k_node=cfg.model.k_node, k_gauss=cfg.model.k_gauss, z0_dim=cfg.model.z0_dim,
).to(dev)
K  = model.n_nodes
B  = cfg.train.z0_bank_size
z0_bank = torch.nn.Parameter(torch.randn(B, K, cfg.model.z0_dim, device=dev) * 0.01)
t0 = ts("NodeFlow init", t0)

# ── SVD load ──────────────────────────────────────────────────────────────────
print("\n=== SVD load ===")
t0 = time.time()
from anchorflow.sds import SVDGuidance
svd = SVDGuidance(
    sigma_min=cfg.mds.sigma_min, sigma_max=cfg.mds.sigma_max,
    guidance_scale=cfg.mds.guidance_scale,
    motion_bucket_id=cfg.mds.motion_bucket_id,
    grad_clip=cfg.mds.grad_clip, device=dev,
)
torch.cuda.synchronize()
t0 = ts("SVD load", t0)
mem_after_svd = torch.cuda.memory_allocated() // (1024**2)
print(f"  GPU alloc after SVD load: {mem_after_svd} MiB")

# ── One step profiling ────────────────────────────────────────────────────────
print("\n=== One training step (profiling) ===")
z0 = z0_bank[0]

# 1. GNN encode
torch.cuda.synchronize(); t0 = time.time()
h = model.encode(z0)
t0 = ts("1. GNN encode", t0)

# 2. decode_batch (all T-1 frames)
torch.cuda.synchronize(); t0 = time.time()
t_vals = torch.arange(1, T, dtype=torch.float32, device=dev)
all_disps = model.decode_batch(h, t_vals)
t0 = ts(f"2. decode_batch (T-1={T-1})", t0)

# 3. render frame0 (no grad)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    frame0 = render_gaussians(cam, canonical_xyz).clamp(0, 1)
t0 = ts("3. render frame0", t0)

# 4. render deformed frames (with grad)
torch.cuda.synchronize(); t0 = time.time()
frames = [frame0]
for i in range(T - 1):
    xyz_def = canonical_xyz + all_disps[i]
    frames.append(render_gaussians(cam, xyz_def))
frames_t = torch.stack(frames, dim=0)
t0 = ts(f"4. render {T-1} deformed frames", t0)

# 5. VAE encode_frames (batch)
torch.cuda.synchronize(); t0 = time.time()
x0 = svd.encode_frames(frames_t)
t0 = ts("5. VAE encode_frames (batch)", t0)

# 6. static lat0 encode
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    lat0    = svd.vae.encode((frames_t[0:1] * 2 - 1).float()).latent_dist.mode() * svd.vae_scale
    x0_stat = lat0.unsqueeze(0).expand(1, T, -1, -1, -1)
t0 = ts("6. static lat0 encode", t0)

# 7. cond encode (_clip_embed + _cond_latent)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    cond_lat, img_emb, time_ids = svd._cond(frame0, T)
t0 = ts("7. _cond (clip+vae cond)", t0)

# 8. sigma + noise
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    sigma = svd._sample_sigma()
    noise = torch.randn_like(x0)
    z_dyn  = x0      + sigma * noise
    z_stat = x0_stat + sigma * noise
t0 = ts("8. sigma + noise", t0)

# 9a. UNet forward batch=4 (fused MDS)
print("  [9a. UNet batch=4 fused] ...")
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    try:
        zeros_cl  = torch.zeros_like(cond_lat)
        zeros_emb = torch.zeros_like(img_emb)
        B4_z   = torch.cat([z_dyn,  z_dyn,  z_stat, z_stat], dim=0)
        B4_cl  = torch.cat([zeros_cl, cond_lat, zeros_cl, cond_lat], dim=0)
        B4_emb = torch.cat([zeros_emb, img_emb, zeros_emb, img_emb], dim=0)
        B4_tid = time_ids.expand(4, -1)
        v4 = svd._unet_forward(B4_z, B4_cl, B4_emb, B4_tid, sigma)
        t0 = ts("9a. UNet batch=4 (fused dyn+stat)", t0)
        del v4; torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  OOM at batch=4: {e}")
        torch.cuda.empty_cache()

# 9b. UNet forward batch=2 x2
print("  [9b. UNet batch=2 x2] ...")
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    eps_dyn  = svd._eps_pred_single(z_dyn,  sigma, cond_lat, img_emb, time_ids)
    eps_stat = svd._eps_pred_single(z_stat, sigma, cond_lat, img_emb, time_ids)
t0 = ts("9b. UNet batch=2 x2 (sequential)", t0)

# 10. backward
torch.cuda.synchronize(); t0 = time.time()
grad = eps_dyn - eps_stat
loss = svd._apply(x0, grad)
loss.backward()
t0 = ts("10. backward", t0)

print("\n=== DONE ===")
print(f"GPU peak alloc: {torch.cuda.max_memory_allocated() // (1024**2)} MiB")
print(f"GPU peak reserved: {torch.cuda.max_memory_reserved() // (1024**2)} MiB")
