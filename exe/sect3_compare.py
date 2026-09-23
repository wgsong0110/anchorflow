"""3DGS 내부를 입자셋으로 채운 뒤, **xy 평행 평면 단면**을 맨 밑에서 맨 위까지
훑으면서 원본 3DGS 와 나란히 비교한다.

각 프레임은 z = z0 근방의 얇은 슬랩(두께 --thick)에 속한 커널만 골라 바로 위에서
내려다본 것이다. 원본은 껍데기라 테두리만 보이고, 채운 쪽은 속이 메워져 보인다.

  python exe/fill_section_compare.py --model <3DGS> --config <씬 config> \
      --out out.mp4 --spacing 0.02 --frames 90
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from tqdm import tqdm
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--spacing", type=float, default=0.02, help="입자 간격 (시뮬 단위)")
ap.add_argument("--grid", type=int, default=160, help="점유 격자 해상도")
ap.add_argument("--level", type=float, default=0.5, help="점유 문턱 (분위)")
ap.add_argument("--frames", type=int, default=90)
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--res", type=int, default=560)
ap.add_argument("--thick", type=float, default=0.0, help="슬랩 두께 (0이면 spacing)")
ap.add_argument("--axis", type=int, default=2, help="단면 법선축 (2=z, xy 평행)")
ap.add_argument("--elev", type=float, default=88.0, help="내려다보는 고도각")
ap.add_argument("--close", type=float, default=0.03,
                help="껍데기 구멍 막는 닫음 반경 (시뮬 단위)")
ap.add_argument("--white_bg", type=int, default=1)
ap.add_argument("--dot", type=int, default=2, help="점 반지름 (픽셀)")
ap.add_argument("--pg_h5", required=True, help="PG 내부채우기 결과 h5")
ap.add_argument("--anchor_npy", required=True, help="앵커 입자셋 npy")
a = ap.parse_args()

dev = "cuda:0"
torch.set_grad_enabled(False)
import imageio.v2 as imageio                                      # noqa: E402
from PIL import Image, ImageDraw                                  # noqa: E402
from scipy.ndimage import (binary_dilation, binary_erosion,
                           binary_fill_holes, gaussian_filter)      # noqa: E402
from scipy.spatial import cKDTree                                 # noqa: E402

from scene.gaussian_model import GaussianModel                    # noqa: E402
from utils.camera_view_utils import (                             # noqa: E402
    get_camera_view)
from utils.decode_param import decode_param_json                  # noqa: E402
from utils.render_utils import (convert_SH, initialize_resterize,  # noqa: E402
                                load_params_from_gs)
from utils.system_utils import searchForMaxIteration               # noqa: E402
from utils.transformation_utils import (                          # noqa: E402
    apply_cov_rotations, apply_inverse_cov_rotations,
    apply_inverse_rotations, apply_rotations, generate_rotation_matrices,
    get_center_view_worldspace_and_observant_coordinate, shift2center111,
    transform2origin, undoshift2center111, undotransform2origin)


class _Pipe:
    convert_SHs_python = False
    compute_cov3D_python = True
    debug = False


(mat, bc, tp, pre, cam_p) = decode_param_json(a.config)
g = GaussianModel(3)
it = searchForMaxIteration(os.path.join(a.model, "point_cloud"))
g.load_ply(os.path.join(a.model, "point_cloud", f"iteration_{it}", "point_cloud.ply"))
pipe = _Pipe()
bg = torch.tensor([1, 1, 1] if a.white_bg else [0, 0, 0], dtype=torch.float32,
                  device=dev)
p = load_params_from_gs(g, pipe)
pos, cov, opa, shs = p["pos"], p["cov3D_precomp"], p["opacity"], p["shs"]
keep = opa[:, 0] > pre["opacity_threshold"]
pos, cov, opa, shs = pos[keep], cov[keep], opa[keep], shs[keep]
for nm in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
           "_rotation"):
    setattr(g, nm, getattr(g, nm)[keep])
rots = generate_rotation_matrices(torch.tensor(pre["rotation_degree"]),
                                  pre["rotation_axis"])
rotated = apply_rotations(pos, rots)
trans, scale_origin, mean_pos = transform2origin(rotated,
                                                 float(pre.get("scale", 1.0)))
trans = shift2center111(trans)
cov = apply_cov_rotations(cov, rots)
cov = scale_origin * scale_origin * cov
S = trans.detach().cpu().numpy().astype(np.float64)
print(f"[표면] 가우시안 {S.shape[0]}", flush=True)

# ---------------------------------------------- 내부를 균일 격자로 채운다
N = a.grid
lo, hi = S.min(0) - 0.04, S.max(0) + 0.04
h = (hi - lo) / N
gi = np.floor((S - lo) / h).astype(int).clip(0, N - 1)
vol = np.zeros((N, N, N), np.float32)
np.add.at(vol, (gi[:, 0], gi[:, 1], gi[:, 2]), 1.0)
vol = gaussian_filter(vol, sigma=1.0)
thr = float(np.quantile(vol[vol > 1e-6], a.level))
occ = vol > thr
# 3DGS 껍데기는 구멍이 뚫려 있어 그냥 binary_fill_holes 를 하면 바깥에서 물이 새
# 들어와 속이 안 메워진다. 먼저 닫음(dilate->fill->erode)으로 틈을 막고,
# 세 축 단면별 2D 채움 중 둘 이상에서 안쪽이면 내부로 본다.
kc = max(1, int(round(a.close / float(h.min()))))
occ_c = binary_dilation(occ, iterations=kc)


def _fill2d(m, ax):
    out = np.zeros_like(m)
    for i in range(m.shape[ax]):
        sl = [slice(None)] * 3
        sl[ax] = i
        out[tuple(sl)] = binary_fill_holes(m[tuple(sl)])
    return out


f0 = _fill2d(occ_c, 0).astype(np.uint8)
f1 = _fill2d(occ_c, 1).astype(np.uint8)
f2 = _fill2d(occ_c, 2).astype(np.uint8)
solid = binary_fill_holes(occ_c) | ((f0 + f1 + f2) >= 2)
solid = binary_erosion(solid, iterations=kc + 1)      # 닫음 만큼 되돌리고 경계 한 칸 더
print(f"[격자] {N}^3 점유 {int(occ.sum())} -> 닫음 {int(occ_c.sum())} "
      f"-> 속 채움 {int(solid.sum())} 칸 (닫음 반경 {kc}칸)", flush=True)

import h5py                                                        # noqa: E402

with h5py.File(a.pg_h5, "r") as _h:
    PGP = np.array(_h["x"])
if PGP.shape[0] == 3 and PGP.shape[1] != 3:
    PGP = PGP.T
ANP = np.load(a.anchor_npy)
print(f"[입자] 3DGS {S.shape[0]}, PG 채움 {PGP.shape[0]}, 앵커 {ANP.shape[0]}",
      flush=True)

C0 = 0.28209479177387814
col_s = (0.5 + C0 * shs[:, 0, :].detach().cpu().numpy()).clip(0, 1)
tree = cKDTree(S)
col_p = col_s[tree.query(PGP, k=1)[1]]
col_a = col_s[tree.query(ANP, k=1)[1]]

AX = a.axis
U, V = [k for k in (0, 1, 2) if k != AX]
thick = a.thick if a.thick > 0 else 0.012
ALL = np.concatenate([S, PGP, ANP], 0)
z_lo, z_hi = float(ALL[:, AX].min()), float(ALL[:, AX].max())
u_lo, u_hi = ALL[:, U].min() - 0.03, ALL[:, U].max() + 0.03
v_lo, v_hi = ALL[:, V].min() - 0.03, ALL[:, V].max() + 0.03
span = max(u_hi - u_lo, v_hi - v_lo)
uc, vc = 0.5 * (u_lo + u_hi), 0.5 * (v_lo + v_hi)
u_lo, v_lo = uc - span / 2, vc - span / 2
R = a.res
dd = a.dot
oy, ox = np.mgrid[-dd:dd + 1, -dd:dd + 1]
disk = (ox ** 2 + oy ** 2) <= dd ** 2
oy, ox = oy[disk], ox[disk]


def scatter(P, C):
    img = np.ones((R, R, 3), np.float32) * (1.0 if a.white_bg else 0.0)
    if len(P):
        px = ((P[:, U] - u_lo) / span * (R - 1)).astype(int)
        py = (((v_lo + span) - P[:, V]) / span * (R - 1)).astype(int)
        for k in range(len(ox)):
            img[(py + oy[k]).clip(0, R - 1), (px + ox[k]).clip(0, R - 1)] = C
    return (img * 255).astype(np.uint8)


frames = []
for i in tqdm(range(a.frames), desc="단면", ncols=80):
    z0 = z_lo + (z_hi - z_lo) * i / max(a.frames - 1, 1)
    ms = np.abs(S[:, AX] - z0) < thick * 0.5
    mp = np.abs(PGP[:, AX] - z0) < thick * 0.5
    ma = np.abs(ANP[:, AX] - z0) < thick * 0.5
    row = []
    for nm, im in ((f"원본 3DGS   z={z0:.3f}   {int(ms.sum())}", scatter(S[ms], col_s[ms])),
                   (f"PG 내부채우기   {int(mp.sum())}", scatter(PGP[mp], col_p[mp])),
                   (f"앵커 입자셋   {int(ma.sum())}", scatter(ANP[ma], col_a[ma]))):
        pil = Image.fromarray(im)
        dr = ImageDraw.Draw(pil)
        dr.rectangle([0, 0, pil.width, 16], fill=(0, 0, 0))
        dr.text((4, 2), nm, fill=(255, 255, 255))
        row.append(np.array(pil))
    frames.append(np.concatenate(row, 1))

imageio.mimsave(a.out, frames, fps=a.fps, quality=8)
print(f"[저장] {a.out}  {len(frames)} 프레임", flush=True)
print("SECT3_OK", flush=True)
