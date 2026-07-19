#!/usr/bin/env python
"""Profile the real SSM training step: rollout / binding / lbs_warp / render / MDS."""
from __future__ import annotations
import argparse, json, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
sys.path.append("/workspace/SC-GS")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as _render_scgs

def render(cam, g, pipe, bg):
    zeros = torch.zeros_like(g.get_xyz)
    return _render_scgs(cam, g, pipe, bg, d_xyz=zeros, d_rotation=0.0, d_scaling=zeros)

from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow.anchors import AnchorSet
from anchorflow.ssm_dynamics import SSMDynamics, ssm_rollout
from anchorflow import warp as W
from anchorflow import geom

class Cam:
    def __init__(s, R, T, fx, fy, Wd, Hd):
        s.image_width, s.image_height = Wd, Hd; s.FoVx, s.FoVy = fx, fy
        s.znear, s.zfar = 0.01, 100.0
        w2v = torch.tensor(getWorld2View2(R, T)).T.cuda()
        proj = getProjectionMatrix(s.znear, s.zfar, fx, fy).T.cuda()
        s.world_view_transform = w2v
        s.full_proj_transform = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        s.camera_center = w2v.inverse()[3, :3]
class Pipe:
    convert_SHs_python = False; compute_cov3D_python = True
    debug = False; antialiasing = False

def tm(label, fn, n=1):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): out = fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"  {label:44s} {dt:8.1f} ms  mem={torch.cuda.memory_allocated()//(1024**2):6d} MiB")
    return out, dt

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--cfg", required=True)
a = ap.parse_args()
cfg = OmegaConf.load(a.cfg); T = cfg.model.n_frames
g = GaussianModel(3)
g.load_ply(f"{a.model}/point_cloud/iteration_30000/point_cloud.ply"); g.active_sh_degree = 3
for p in (g._features_dc, g._features_rest, g._opacity, g._scaling, g._rotation, g._xyz):
    p.requires_grad_(False)
canon = g.get_xyz.detach().clone()
cov6 = W.cov_from_scale_rot(g.get_scaling.detach(), g._rotation.detach()).detach()
G = canon.shape[0]; print(f"gaussians={G}")
cj = json.load(open(f"{a.model}/cameras.json"))[0]
rot = np.array(cj["rotation"], dtype=np.float32); pos = np.array(cj["position"], dtype=np.float32)
Wd, Hd = cj["width"], cj["height"]; s_ = cfg.model.res / max(Wd, Hd)
cam = Cam(rot, -rot.T@pos, focal2fov(cj["fx"],Wd), focal2fov(cj["fy"],Hd),
          max(8,int(round(Wd*s_/8))*8), max(8,int(round(Hd*s_/8))*8))
bg = torch.zeros(3, device="cuda")
_q = canon.float()
extent = float((torch.quantile(_q,0.99,dim=0)-torch.quantile(_q,0.01,dim=0)).norm()); del _q
M = cfg.model.n_nodes
from anchorflow.anchors import fps
anchors = AnchorSet.from_trajectory(canon[fps(canon, M)].clone(),
    latent_dim=cfg.model.z_dim, e_dim=cfg.model.e_dim, K=cfg.model.k_gauss).cuda()
model = SSMDynamics(hidden=cfg.model.hidden, mp_steps=cfg.model.mp_steps,
    ssm_dim=cfg.model.ssm_dim, e_dim=cfg.model.e_dim, z_dim=cfg.model.z_dim,
    accel_scale=cfg.model.accel_scale*extent).cuda()
gcfg = {"graph":"knn","k":cfg.model.k_node}; dt = cfg.model.dt
z = torch.nn.Parameter(0.01*torch.randn(anchors.num, cfg.model.z_dim, device="cuda"))
v0 = torch.nn.Parameter(torch.randn(anchors.num,3,device="cuda")*0.24)

print("\n=== 컴포넌트 ===")
p_seq,_ = tm("1. ssm_rollout [T,M,3]", lambda: ssm_rollout(model, anchors.canonical, v0,
    anchors.e, z, init_vel=v0, init_pos=anchors.canonical, steps=T-1, cfg=gcfg, dt=dt, grad=True))
(wb, ib), _ = tm("2. cal_nn_weight (learnable rho)", lambda: anchors.cal_nn_weight(canon))
aR, _ = tm("3. anchor_rotations (Procrustes)", lambda: W.anchor_rotations(anchors.canonical, p_seq[-1]))
_, t_pos = tm("4. lbs pos only (CUDA kernel)", lambda: W._lbs_blend(canon, wb, ib, anchors.canonical, p_seq[-1], aR), n=3)
def cov_part():
    quat = geom.matrix_to_quat(aR); qg = W._blend_quat(quat, wb, ib)
    Rg = geom.quat_to_matrix(qg); S = W.cov6_to_mat3(cov6)
    return W.mat3_to_cov6(Rg @ S @ Rg.transpose(-1,-2))
_, t_cov = tm("5. cov warp (quat blend + RSR^T) [torch]", cov_part, n=3)
_, t_full = tm("6. lbs_warp 전체 (1프레임)", lambda: W.lbs_warp(canon, cov6, wb, ib, anchors.canonical, p_seq[-1]), n=3)
with torch.no_grad():
    _, t_rend = tm("7. render 1프레임 (no_grad)", lambda: render(cam, g, Pipe(), bg)["render"], n=3)
print(f"\n  => lbs_warp 중 cov 비중: {t_cov/t_full*100:.0f}%  (pos {t_pos/t_full*100:.0f}%)")
print(f"  => 25프레임 추정: lbs_warp {t_full*25:.0f}ms, render {t_rend*25:.0f}ms")
