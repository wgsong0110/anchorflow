"""형상별 trilinear 활성 칸 비율을 잰다 (원본 3DGS / PG 내부채움 둘 다)."""
from __future__ import annotations
import argparse, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--pgfill", default="", help="PG 내부채움 입자 .npy")
ap.add_argument("--name", default="")
ap.add_argument("--res", type=int, nargs="+", default=[32, 50])
a = ap.parse_args()

dev = "cuda:0"
torch.set_grad_enabled(False)
from scene.gaussian_model import GaussianModel
from utils.decode_param import decode_param_json
from utils.render_utils import load_params_from_gs
from utils.system_utils import searchForMaxIteration
from utils.transformation_utils import (apply_rotations, generate_rotation_matrices,
                                        shift2center111, transform2origin)
from anchorflow import trilinear as TRI, vox_anchor


class _P:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


(mat, bc, tp, pre, cam_p) = decode_param_json(a.config)
g = GaussianModel(3)
it = searchForMaxIteration(os.path.join(a.model, "point_cloud"))
g.load_ply(os.path.join(a.model, "point_cloud", f"iteration_{it}", "point_cloud.ply"))
p = load_params_from_gs(g, _P())
pos, opa = p["pos"], p["opacity"]
pos = pos[opa[:, 0] > pre["opacity_threshold"]]
rots = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]), pre["rotation_axis"])
x_raw = shift2center111(transform2origin(apply_rotations(pos, rots),
                                         float(pre.get("scale", 1.0)))[0]).to(dev)
sets = [("원본 3DGS", x_raw)]
if a.pgfill and os.path.exists(a.pgfill):
    sets.append(("PG 내부채움", torch.from_numpy(np.load(a.pgfill)).float().to(dev)))

for nm, X in sets:
    for R in a.res:
        lo, h, n3 = vox_anchor.grid_for(X, R ** 3)
        tot = int(n3[0] * n3[1] * n3[2])
        flat, w = TRI.corners(X, lo, h, n3)
        act = int(torch.unique(flat.reshape(-1)).numel())
        print(f"{a.name:10} {nm:12} res {R:3d}: 입자 {X.shape[0]:7d} | 전체 {tot:7d} "
              f"| 활성 {act:6d} ({100*act/tot:5.1f}%)", flush=True)
print("RATE_OK", flush=True)
