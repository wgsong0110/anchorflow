"""학생 한 스텝의 전체 비용을 잰다: P2G + 신경망 + G2P + 3DGS 렌더."""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--res", type=int, default=50)
ap.add_argument("--arch", default="unet",
                choices=("conv", "unet", "conv_sep", "unet_sep",
                         "conv_par", "unet_par"))
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--feat", type=int, default=62)
ap.add_argument("--nctrl", type=int, default=4)
ap.add_argument("--ens", type=int, default=1,
                help="원점을 어긋나게 둔 격자를 몇 개 앙상블할지")
ap.add_argument("--rep", type=int, default=20)
a = ap.parse_args()

dev = "cuda:0"
torch.set_grad_enabled(False)
from scene.gaussian_model import GaussianModel
from utils.camera_view_utils import get_camera_view
from utils.decode_param import decode_param_json
from utils.render_utils import convert_SH, initialize_resterize, load_params_from_gs
from utils.system_utils import searchForMaxIteration
from utils.transformation_utils import (apply_cov_rotations, apply_rotations,
                                        generate_rotation_matrices,
                                        get_center_view_worldspace_and_observant_coordinate,
                                        shift2center111, transform2origin)
from anchorflow import trilinear as TRI, vox_anchor
from anchorflow.conv_stepper import ConvStepper


class _P:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


(mat, bc, tp, pre, cam_p) = decode_param_json(a.config)
g = GaussianModel(3)
it = searchForMaxIteration(os.path.join(a.model, "point_cloud"))
g.load_ply(os.path.join(a.model, "point_cloud", f"iteration_{it}", "point_cloud.ply"))
p = load_params_from_gs(g, _P())
pos, cov, opa, shs = p["pos"], p["cov3D_precomp"], p["opacity"], p["shs"]
keep = opa[:, 0] > pre["opacity_threshold"]
pos, cov, opa, shs = pos[keep], cov[keep], opa[keep], shs[keep]
for nm in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
    setattr(g, nm, getattr(g, nm)[keep])
rots = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]), pre["rotation_axis"])
x = shift2center111(transform2origin(apply_rotations(pos, rots), float(pre.get("scale", 1.0)))[0]).to(dev)
cov = apply_cov_rotations(cov, rots).to(dev)
N = x.shape[0]
v = torch.zeros_like(x)
X = x.clone()
m = torch.full((N,), 1e-6, device=dev)
print(f"[가우시안] {N}", flush=True)

lo, h, n3 = vox_anchor.grid_for(x, a.res ** 3)
grid = tuple(int(t) for t in n3)


def timeit(fn, rep):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(rep): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/rep*1e3

def cell_step():
    crow, cw, ncell = TRI.cell_index(x, lo, h, n3)
    Mc = ncell[0] * ncell[1] * ncell[2]
    cen = torch.zeros(Mc, 3, device=dev)
    f = TRI.tri_feats(x, v, X, m, crow, cw, Mc, cen, float(h))
    return f, ncell, Mc

f0, ncell, M_cell = cell_step()
F_CELL = f0.shape[-1]
flat_c, w_c = TRI.corners(x, lo, h, n3)
grid_pts = tuple(int(t) for t in n3)

_arch = a.arch.replace("conv", "plain")
net = ConvStepper(a.feat, a.hidden, a.depth, arch=_arch).to(dev)
fin = torch.randn(M_cell, a.feat, device=dev)
dp_pts = torch.randn(int(n3[0]) * int(n3[1]) * int(n3[2]), 3, device=dev)

# 손잡이 특징: 가우시안별 계산 + 셀 집계
K = a.nctrl
hc = torch.rand(K, 3, device=dev)
hd = torch.rand(K, 3, device=dev) * 0.01
def ctrl_step():
    rel = (x.unsqueeze(1) - hc.unsqueeze(0)) / float(h)
    frc = hd.unsqueeze(0).expand(x.shape[0], K, 3) / float(h)
    q = ((x.unsqueeze(1) - hc.unsqueeze(0)).norm(dim=-1) / 0.15).clamp(0, 1)
    w = (1.0 - q * q) ** 2
    cf = torch.cat([rel.reshape(x.shape[0], -1), frc.reshape(x.shape[0], -1), w], -1)
    crow, cw, ncell2 = TRI.cell_index(x, lo, h, n3)
    Mc = ncell2[0] * ncell2[1] * ncell2[2]
    wm = cw * m.unsqueeze(1)
    num = torch.zeros(Mc, cf.shape[-1], device=dev)
    num.index_add_(0, crow.reshape(-1),
                   (wm.unsqueeze(-1) * cf.unsqueeze(1)).reshape(-1, cf.shape[-1]))
    den = torch.zeros(Mc, 1, device=dev)
    den.index_add_(0, crow.reshape(-1), wm.reshape(-1, 1))
    return num / den.clamp(min=1e-12)

# 앙상블: 원점을 어긋나게 둔 격자 여러 개의 변위장을 평균
if a.ens > 1:
    import math as _m
    offs = [(lo + (torch.rand(3, device=dev) - 0.5) * float(h)) for _ in range(a.ens)]

    def ens_step():
        acc = torch.zeros_like(x)
        for o in offs:
            cr, cwe, nc = TRI.cell_index(x, o, h, n3)
            Mc = nc[0] * nc[1] * nc[2]
            cen = torch.zeros(Mc, 3, device=dev)
            f = TRI.tri_feats(x, v, X, m, cr, cwe, Mc, cen, float(h))
            fi = torch.randn(Mc, a.feat, device=dev)
            d3 = net(None, fi, 1/60., grid_pts, cells=tuple(nc))[0]
            fl, wq = TRI.corners(x, o, h, n3)
            acc = acc + TRI.g2p(fl, wq, d3)
        return acc / a.ens

    t_ens = timeit(ens_step, a.rep)
    print(f"[앙상블] {a.ens} x res {a.res}: 전달+신경망 {t_ens:.2f} ms", flush=True)

t_ctrl = timeit(ctrl_step, a.rep)
t_p2g = timeit(cell_step, a.rep)
t_cor = timeit(lambda: TRI.corners(x, lo, h, n3), a.rep)
t_net = timeit(lambda: net(None, fin, 1/60., grid_pts, cells=tuple(ncell)), a.rep)
t_g2p = timeit(lambda: TRI.g2p(flat_c, w_c, dp_pts), a.rep)

view_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape(1, 3).cuda()
up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(1, 3).cuda()
vcw, obs = get_center_view_worldspace_and_observant_coordinate(
    view_c, up, rots, transform2origin(apply_rotations(pos, rots), float(pre.get("scale", 1.0)))[1],
    transform2origin(apply_rotations(pos, rots), float(pre.get("scale", 1.0)))[2])
cam = get_camera_view(a.model, default_camera_index=0, center_view_world_space=vcw,
                      observant_coordinates=obs, show_hint=False, init_azimuthm=None,
                      init_elevation=None, init_radius=None, move_camera=False,
                      current_frame=0, delta_a=0, delta_e=0, delta_r=0)
bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=dev)
rast = initialize_resterize(cam, g, _P(), bg)
R = torch.eye(3, device=dev).repeat(N, 1, 1)
col = convert_SH(shs.to(dev), cam, g, x, R)
scr = torch.zeros((N, 3), device=dev)

def render():
    return rast(means3D=x.contiguous(), means2D=scr, shs=None,
                colors_precomp=col.float().contiguous(), opacities=opa.to(dev).contiguous(),
                scales=None, rotations=None, cov3D_precomp=cov.contiguous())

t_rnd = timeit(render, a.rep)
tot = t_p2g + t_ctrl + t_cor + t_net + t_g2p + t_rnd
print(f"[격자] 격자점 {grid_pts} 셀 {tuple(ncell)} 셀특징 {F_CELL}채널", flush=True)
print(f"셀집계 {t_p2g:.2f} | 손잡이 {t_ctrl:.2f} | corners {t_cor:.2f} | "
      f"신경망 {t_net:.2f} | G2P {t_g2p:.2f} | 렌더 {t_rnd:.2f} ms", flush=True)
print(f"[합계] {tot:.2f} ms  ->  {1000/tot:.1f} FPS "
      f"(렌더 제외 {tot-t_rnd:.2f} ms, {1000/(tot-t_rnd):.1f} FPS)", flush=True)
print("STEPBENCH_OK", flush=True)
