"""lbdata 궤적을 NGFF(Neural Gaussian Force Field) 학습 형식으로 바꾼다.

NGFF 형식 (R/dynamic_models/DGSDataset.py _process_seq_dir): 장면 디렉토리에
  0000.h5 ...  : pos (N,3) / cov3D (N,6 위삼각 xx,xy,xz,yy,yz,zz) / rot (N,3,3), float16
  opacity.h5   : opacity (N,1)      shs.h5 : shs (N,SH,3)
장면 이름은 basename 으로 캐시가 갈리므로 학습 r0000.., 평가 t0000.. 로 겹치지 않게 한다.

가우시안만 쓴다 (앞 NG=297877 개, 불투명도 필터 뒤 lego 3DGS 와 같은 번호, lb_render.py 와 같은 규약).
좌표는 시뮬 좌표 그대로 (상자 [0,2]^3, 바닥 z=0.48).
  cov3D = SO^2 · F C0 F^T  (C0 = 세계 좌표 3DGS 공분산, SO = transform2origin 축척; sim = (world-MEAN)·SO + 1)
  rot   = polar(F) · R_gauss
**용량 절약**: cov3D·rot 는 0 프레임에만 쓴다. NGFF 학습은 points/com/com_vel/angle/angle_vel/padding/knn 만
쓰고 cov3D·rot 는 안 쓴다. 패치한 DGSDataset 은 1 프레임 이후 키가 없으면 0 프레임 값을 쓴다.
opacity.h5·shs.h5 는 모든 장면이 같으므로 common/ 에 한 번 쓰고 심링크한다.
초기 강체 속도: init_state.json 의 v0 (시뮬 단위/초), frame_dt.
scene_configs/lb_<split>.json 의 scene_object_idxs 는 [] -> NGFF group_objects 가 전부 객체 0 으로 둔다 (단일 객체).

  cd /home/dkta/work/i-physgaussian && python exe/lb_to_ngff.py --split train --ids 0       # 하나 먼저
  ... --split train --ids all ; --split test --ids all
"""
import argparse
import json
import os
import sys

W = "/home/dkta/work"
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=f"{W}/lbdata")
ap.add_argument("--out", default=f"{W}/lb/ngff_data")
ap.add_argument("--split", required=True, choices=["train", "test"])
ap.add_argument("--ids", default="all", help="all 또는 0,5,7")
ap.add_argument("--frames", type=int, default=80, help="0..frames-1 (NGFF 공식 --steps 80)")
ap.add_argument("--model", default=f"{W}/pgmodel/lego_whitebg-trained")
a = ap.parse_args()
import h5py                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from tqdm import tqdm                                            # noqa: E402
torch.set_grad_enabled(False)
dev = "cuda"
cfg = json.load(open(f"{a.src}/config.json"))
sys.path.append(f"{W}/i-physgaussian"); sys.path.append(f"{W}/i-physgaussian/gaussian-splatting")
_cwd = os.getcwd(); os.chdir(f"{W}/i-physgaussian")
from scene.gaussian_model import GaussianModel                       # noqa: E402
from utils.transformation_utils import transform2origin, shift2center111   # noqa: E402
os.chdir(_cwd)

gs = GaussianModel(3)
gs.load_ply(f"{a.model}/point_cloud/iteration_30000/point_cloud.ply")
KIDX = torch.nonzero(gs.get_opacity[:, 0] > float(cfg.get("opacity_threshold", 0.02))).squeeze(1)
TP, SO, MEAN = transform2origin(gs.get_xyz[KIDX], float(cfg.get("scale", 1.0)))
TP = shift2center111(TP)
NG = KIDX.numel()
SO = float(SO)
c6 = gs.get_covariance()[KIDX]
C0 = torch.zeros(NG, 3, 3, device=dev)
C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c6[:, 0], c6[:, 1], c6[:, 2]
C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c6[:, 3], c6[:, 4], c6[:, 5]
C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c6[:, 1], c6[:, 2], c6[:, 4]
q = torch.nn.functional.normalize(gs._rotation[KIDX], dim=1)          # (r,x,y,z), 3DGS build_rotation 과 같은 식
r, x, y, z = q.unbind(1)
RG = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
                  2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
                  2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
print(f"[가우시안] NG={NG} SO={SO:.6f} SH={tuple(gs.get_features.shape[1:])}", flush=True)

out = a.out
os.makedirs(f"{out}/common", exist_ok=True)
os.makedirs(f"{out}/scene_configs", exist_ok=True)
if not os.path.exists(f"{out}/common/shs.h5"):
    with h5py.File(f"{out}/common/opacity.h5.tmp", "w") as h:
        h["opacity"] = gs.get_opacity[KIDX].cpu().numpy().astype(np.float16)
    with h5py.File(f"{out}/common/shs.h5.tmp", "w") as h:
        h["shs"] = gs.get_features[KIDX].cpu().numpy().astype(np.float16)
    os.replace(f"{out}/common/opacity.h5.tmp", f"{out}/common/opacity.h5")
    os.replace(f"{out}/common/shs.h5.tmp", f"{out}/common/shs.h5")

src = f"{a.src}/{a.split}"
alln = sorted(d for d in os.listdir(src) if d.isdigit() and os.path.exists(f"{src}/{d}/meta.json"))
ids = alln if a.ids == "all" else [f"{int(i):04d}" for i in a.ids.split(",")]
pre = "r" if a.split == "train" else "t"
grp = f"lb_{a.split}"
os.makedirs(f"{out}/mpm/{grp}", exist_ok=True)
cfgp = f"{out}/scene_configs/{grp}.json"
sc = json.load(open(cfgp)) if os.path.exists(cfgp) else {}


def up6(c):
    return torch.stack([c[:, 0, 0], c[:, 0, 1], c[:, 0, 2], c[:, 1, 1], c[:, 1, 2], c[:, 2, 2]], 1)


for nid in tqdm(ids, desc=a.split):
    name = f"{pre}{nid}"
    sc[name] = {"scene_object_idxs": [], "src": f"{src}/{nid}"}
    d = f"{out}/mpm/{grp}/{name}"
    if os.path.exists(f"{d}/DONE"):
        continue
    os.makedirs(d, exist_ok=True)
    meta = json.load(open(f"{src}/{nid}/meta.json"))
    assert int(meta["gaussians_first"]) == NG, (meta["gaussians_first"], NG)
    for fn in ("opacity.h5", "shs.h5"):
        if not os.path.lexists(f"{d}/{fn}"):
            os.symlink(f"../../../common/{fn}", f"{d}/{fn}")
    for t in range(a.frames):
        with h5py.File(f"{src}/{nid}/sim_{t:010d}.h5") as h:
            xs = np.array(h["x"][:NG])
            Fm = np.array(h["F"][:NG]) if t == 0 else None
        assert xs.shape == (NG, 3), xs.shape
        with h5py.File(f"{d}/{t:04d}.h5", "w") as h:
            h["pos"] = xs.astype(np.float16)
            if t == 0:
                Ft = torch.as_tensor(Fm, device=dev).float().reshape(-1, 3, 3)
                cov = (SO ** 2) * (Ft @ C0 @ Ft.transpose(1, 2))
                U, _, Vh = torch.linalg.svd(Ft.double())
                Rp = U @ Vh
                Rp = torch.where((torch.linalg.det(Rp) < 0)[:, None, None],
                                 U @ torch.diag(torch.tensor([1., 1., -1.], device=dev, dtype=torch.float64)) @ Vh, Rp)
                h["cov3D"] = up6(cov).cpu().numpy().astype(np.float16)
                h["rot"] = (Rp.float() @ RG).cpu().numpy().astype(np.float16)
    json.dump({"v0": meta["v0"], "frame_dt": meta["frame_dt"], "src": f"{src}/{nid}", "commit": meta.get("commit"),
               "frames": a.frames, "NG": NG, "SO": SO,
               "note": "cov3D/rot 는 0000.h5 에만 (NGFF 학습 미사용)"}, open(f"{d}/init_state.json", "w"), indent=1)
    open(f"{d}/DONE", "w").write("ok\n")
json.dump(sc, open(cfgp + ".tmp", "w"), indent=1)
os.replace(cfgp + ".tmp", cfgp)
print(f"[완료] {a.split} {len(ids)} 개 -> {out}/mpm/{grp}", flush=True)
