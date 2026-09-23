"""3DGS 내부를 **균일 격자**로 채우고, 원본과 360도 렌더를 나란히 비교한다.

PhysGaussian 의 `particle_filling` 은 레이캐스팅 + 밀도 문턱으로 채우는데 칸당
한 점이라 분포가 표면 밀도를 따라간다. 여기서는 점유 격자에 구멍을 메워
(binary_fill_holes) **내부 전체를 같은 간격으로** 채운다.

새 점의 겉모습(SH·불투명도·공변)은 가장 가까운 표면 가우시안에서 복사한다 --
PhysGaussian 의 `init_filled_particles` 와 같은 근거다.

  python exe/fill_uniform_compare.py --model <3DGS> --config <씬 config> \
      --out out.mp4 --spacing 0.02 --frames 60
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
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--fps", type=int, default=20)
ap.add_argument("--res", type=int, default=560)
ap.add_argument("--close", type=float, default=0.03,
                help="껍데기 구멍 막는 닫음 반경 (시뮬 단위)")
ap.add_argument("--white_bg", type=int, default=1)
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
# 격자가 아니라 **입자 셋**으로 채운다: 부피 안에 무작위로 뿌리고 최소 간격을
# 지키도록 솎는다 (포아송 디스크). 격자로 채우면 줄무늬가 렌더에 그대로 보인다.
rng = np.random.default_rng(0)
vol_cells = int(solid.sum())
cell_v = float(h[0] * h[1] * h[2])
target = int(vol_cells * cell_v / (a.spacing ** 3))
# 후보를 넉넉히(목표의 수십 배) 뿌려야 포아송 디스크가 목표 밀도까지 찬다.
chunks = []
need = max(target * 60, 200000)
got = 0
while got < need:
    c = rng.uniform(lo, hi, size=(2_000_000, 3))
    ci = np.floor((c - lo) / h).astype(int).clip(0, N - 1)
    c = c[solid[ci[:, 0], ci[:, 1], ci[:, 2]]]
    if c.shape[0] == 0:
        break
    chunks.append(c)
    got += c.shape[0]
cand = np.concatenate(chunks, 0)[:need]
print(f"[후보] 부피 안 무작위 점 {cand.shape[0]} 개 (목표 {target})", flush=True)
# 포아송 디스크: 최소 간격 r 을 지키며 순서대로 채택 (공간 해시, 벡터화)
r_min = 0.85 * a.spacing
order = rng.permutation(len(cand))
gk = np.floor(cand / r_min).astype(int)
grid_key = {}
acc = []
offs = [(dx_, dy_, dz_) for dx_ in (-1, 0, 1) for dy_ in (-1, 0, 1) for dz_ in (-1, 0, 1)]
for i in tqdm(order, desc="포아송 디스크", ncols=80):
    k0, k1, k2 = gk[i]
    nb = []
    for dx_, dy_, dz_ in offs:
        v = grid_key.get((k0 + dx_, k1 + dy_, k2 + dz_))
        if v:
            nb.extend(v)
    if nb:
        if (np.abs(cand[nb] - cand[i]).max(1) < r_min).any():
            dd = np.linalg.norm(cand[nb] - cand[i], axis=1)
            if (dd < r_min).any():
                continue
    grid_key.setdefault((k0, k1, k2), []).append(i)
    acc.append(i)
L = cand[np.array(acc)]
n_before = L.shape[0]
tree = cKDTree(S)
d, nn = tree.query(L, k=1)
L = L[d > 0.3 * a.spacing]                     # 표면 가우시안과 겹치지 않게
d, nn = tree.query(L, k=1)
print(f"[채움] 포아송 {n_before} -> 표면필터 후 {L.shape[0]} 개, "
      f"최소 간격 {r_min:.4f}, 목표 {target}", flush=True)

Lt = torch.from_numpy(L).float().to(dev)
cov_f = cov[nn]                                      # 가장 가까운 표면에서 복사
shs_f = shs[nn]
opa_f = opa[nn]

view_c = torch.tensor(cam_p["mpm_space_viewpoint_center"]).reshape(1, 3).cuda()
up = torch.tensor(cam_p["mpm_space_vertical_upward_axis"]).reshape(1, 3).cuda()
vcw, obs = get_center_view_worldspace_and_observant_coordinate(
    view_c, up, rots, scale_origin, mean_pos)


def to_world(x):
    return apply_inverse_rotations(
        undotransform2origin(undoshift2center111(x), scale_origin, mean_pos), rots)


def render(xs, cs, ops, ss, az):
    cam = get_camera_view(a.model, default_camera_index=-1,
                          center_view_world_space=vcw, observant_coordinates=obs,
                          show_hint=False, init_azimuthm=az,
                          init_elevation=float(cam_p.get("init_elevation") or 15),
                          init_radius=float(cam_p.get("init_radius") or 2.5),
                          move_camera=False, current_frame=0, delta_a=0,
                          delta_e=0, delta_r=0)
    rast = initialize_resterize(cam, g, pipe, bg)
    pw = to_world(xs)
    cw = apply_inverse_cov_rotations(cs / (scale_origin * scale_origin), rots)
    R = torch.eye(3, device=dev).repeat(pw.shape[0], 1, 1)
    col = convert_SH(ss, cam, g, pw, R)
    scr = torch.zeros((pw.shape[0], 3), dtype=torch.float32, device=dev)
    out = rast(means3D=pw.contiguous(), means2D=scr, shs=None,
               colors_precomp=col.float().contiguous(),
               opacities=ops.contiguous(), scales=None, rotations=None,
               cov3D_precomp=cw.contiguous())
    img = out[0] if isinstance(out, (tuple, list)) else out
    img = torch.nan_to_num(img, nan=float(bg[0])).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


X_all = torch.cat([trans, Lt])
C_all = torch.cat([cov, cov_f])
O_all = torch.cat([opa, opa_f])
S_all = torch.cat([shs, shs_f])
frames = []
for i in range(a.frames):
    az = 360.0 * i / a.frames
    a_img = render(trans, cov, opa, shs, az)              # 원본
    b_img = render(X_all, C_all, O_all, S_all, az)        # 내부 채움
    row = []
    for nm, im in (("원본 3DGS", a_img), (f"내부 입자셋 채움 (+{L.shape[0]})", b_img)):
        pil = Image.fromarray(im)
        dr = ImageDraw.Draw(pil)
        dr.rectangle([0, 0, pil.width, 16], fill=(0, 0, 0))
        dr.text((4, 2), nm, fill=(255, 255, 255))
        row.append(np.array(pil))
    frames.append(np.concatenate(row, 1))
    if i % 10 == 0:
        print(f"  {i}/{a.frames} (방위 {az:.0f}도)", flush=True)

imageio.mimsave(a.out, frames, fps=a.fps, quality=8)
print(f"[저장] {a.out}  {len(frames)} 프레임", flush=True)
print("FILLCMP_OK", flush=True)
