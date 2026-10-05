"""GASP 칸의 **시각 품질**(FID/FVD/KVD)과 **물리 잔차** 입력을 한 번에 만든다.

PG 쪽 `bench_vq.py` 와 같은 규약:
  * 한 칸마다 두 번 굴린다 -- 자기 수렴 서브스텝 s_conv(참조)과 기준 통과 s(대상).
  * 프레임 0 부터 `--frames` 장을 렌더한다 (앞 24 장을 버리지 않는다. PG 와 같다).
  * 렌더는 **GASP 공식 경로** 그대로: 시뮬된 가짜메시 꼭짓점 -> faces.pt 로 삼각형
    -> GaMeS `gs_flat` 모델의 renderer (`gaussian_points_animated_renderer`).
  * 참조 패스에서 앞 35 프레임은 h5 로도 떨군다 (잔차용). 같은 궤적을 두 번
    굴리지 않기 위함이다.

  python exe/bench_vq_gasp.py --shape mic --material elastic --run run5 --frames 60
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from tqdm import tqdm

W = "/home/dkta/work"
GASP = f"{W}/GASP"

ap = argparse.ArgumentParser()
ap.add_argument("--shape", required=True)
ap.add_argument("--material", default="elastic",
                choices=["elastic", "elastoplastic", "viscoplastic"])
ap.add_argument("--run", default="run5")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--resid_frames", type=int, default=35, help="h5 로 떨굴 앞부분")
ap.add_argument("--floor", type=float, default=0.1)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--s", type=int, default=0)
ap.add_argument("--s_ref", type=int, default=0)
ap.add_argument("--cam", type=int, default=4, help="학습 카메라 번호 (공식 기본 4)")
ap.add_argument("--win", type=int, default=16)
ap.add_argument("--only_test", action="store_true")
a = ap.parse_args()

O = f"{W}/bench/{a.run}"
SJ = f"{O}/gasp_{a.shape}_{a.material}.json"
d = json.load(open(SJ)) if os.path.exists(SJ) else {}
s_test = a.s or d.get("s")
s_ref = a.s_ref or d.get("s_conv")
if not s_test or not s_ref:
    raise SystemExit(f"[건너뜀] 서브스텝을 모른다 ({SJ}). --s/--s_ref 로 줄 것")
s_test, s_ref = int(s_test), int(s_ref)

# --- 꼭짓점과 좌표 변환 (bench_gasp.py 와 **똑같이**) ------------------
VP = f"{W}/gamesout/{a.shape}/pseudomesh_info/ours_30000/vertices.pt"
V = torch.load(VP, map_location="cpu").cpu().numpy().reshape(-1, 3)
X0 = np.load(f"{W}/anfill_{a.shape}.npy")
lo_t, hi_t = X0.min(0), X0.max(0)
lo_v, hi_v = V.min(0), V.max(0)
sc = float((hi_t - lo_t).max() / (hi_v - lo_v).max())
cen_t, cen_v = (lo_t + hi_t) / 2.0, (lo_v + hi_v) / 2.0
P0 = ((V - cen_v) * sc + cen_t).astype(np.float32)


def to_model(P):
    """시뮬 좌표 -> 모델(GaMeS) 좌표. 위 변환의 정확한 역이다."""
    return (P - cen_t) / sc + cen_v


# --- GASP 공식 렌더 준비 ---------------------------------------------
os.chdir(GASP)
sys.path.insert(0, GASP)
sys.path.insert(0, f"{GASP}/games_submodule")
import torchvision                                      # noqa: E402
from games_submodule.scene import Scene                  # noqa: E402
from games_submodule.games.flat_splatting.scene.points_gaussian_model \
    import PointsGaussianModel                           # noqa: E402
from games_submodule.renderer.gaussian_points_animated_renderer \
    import render as gs_render                            # noqa: E402

DS = SimpleNamespace(sh_degree=3, source_path=f"{W}/gamesdata/{a.shape}",
                     model_path=f"{W}/gamesout/{a.shape}", images="images",
                     resolution=-1, white_background=True, data_device="cuda",
                     eval=True, gs_type="gs_flat", num_splats=[2], meshes=[])
PIPE = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                       debug=False, antialiasing=False)

with torch.no_grad():
    gaussians = PointsGaussianModel(DS.sh_degree)
    scene = Scene(DS, gaussians, load_iteration=30000, shuffle=False)
    gaussians.prepare_vertices()
    gaussians.prepare_scaling_rot()
BG = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
faces = torch.load(f"{W}/gamesout/{a.shape}/pseudomesh_info/ours_30000/"
                   f"faces.pt").long().cuda()
views = scene.getTrainCameras()
view = views[a.cam % len(views)]
print(f"[렌더 준비] 가우시안 {gaussians.get_xyz.shape[0]}  면 {faces.shape[0]}  "
      f"카메라 {a.cam}/{len(views)}", flush=True)

# --- taichi MPM (bench_gasp.py 와 같은 설정) --------------------------
sys.path.insert(0, f"{W}/taichi_elements")
import taichi as ti                                     # noqa: E402
from engine.mpm_solver import MPMSolver                  # noqa: E402

ti.init(arch=ti.gpu, log_level=ti.ERROR,
        device_memory_fraction=float(os.environ.get("AF_TI_FRAC", 0.5)))
MATID = {"elastic": MPMSolver.material_elastic,
         "elastoplastic": MPMSolver.material_snow,
         "viscoplastic": MPMSolver.material_sand}
MATNAME = {"elastic": "jelly", "elastoplastic": "ti_snow",
           "viscoplastic": "ti_sand"}


def go(s, tag, h5dir=""):
    """서브스텝 s 로 굴리며 프레임마다 렌더한다 (필요하면 h5 도 떨군다)."""
    od = f"{O}/vq/gasp_{a.shape}_{a.material}_{tag}_{s}"
    shutil.rmtree(od, ignore_errors=True)
    os.makedirs(od, exist_ok=True)
    if h5dir:
        shutil.rmtree(h5dir, ignore_errors=True)
        os.makedirs(h5dir, exist_ok=True)
    dx = 2.0 / a.n_grid
    dt_scale = (1.0 / 60.0) / (2e-2 * dx / 2.0) / float(s)
    mpm = MPMSolver(res=(a.n_grid,) * 3, size=2, dt_scale=dt_scale,
                    E_scale=1.0, unbounded=False, support_plasticity=True)
    mpm.set_gravity((0.0, 0.0, -9.8))
    mpm.add_surface_collider(point=(0.0, 0.0, a.floor), normal=(0.0, 0.0, 1.0),
                             surface=MPMSolver.surface_sticky)
    mpm.add_particles(particles=P0, material=MATID[a.material])
    N = int(mpm.n_particles[None])
    fld = ti.Vector.field(9, dtype=ti.f32, shape=N) if h5dir else None
    jfld = ti.field(dtype=ti.f32, shape=N) if h5dir else None

    if h5dir:
        @ti.kernel
        def grab(n: ti.i32):
            for p in range(n):
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        fld[p][3 * i + j] = mpm.F[p][i, j]
                jfld[p] = mpm.Jp[p]

    for f in tqdm(range(a.frames), desc=f"{tag} s={s}"):
        pi = mpm.particle_info()
        P = pi["position"][:N]
        with torch.no_grad():
            Vm = torch.as_tensor(to_model(P), dtype=torch.float32,
                                 device="cuda")
            img = gs_render(Vm[faces], view, gaussians, PIPE, BG)["render"]
            torchvision.utils.save_image(img, f"{od}/{f:04d}.png")
        if h5dir and f < a.resid_frames:
            grab(N)
            with h5py.File(f"{h5dir}/{f:04d}.h5", "w") as hf:
                hf["x"] = P.astype(np.float32)
                hf["v"] = pi["velocity"][:N].astype(np.float32)
                hf["f_tensor"] = fld.to_numpy()[:N].astype(np.float32)
                hf["jp"] = jfld.to_numpy()[:N].astype(np.float32)
        mpm.step(1.0 / 60.0)
    print(f"[렌더] {tag} s={s} -> png {len(os.listdir(od))} 장  {od}",
          flush=True)
    return od


def mk_mp4(src, mp4):
    subprocess.run(["python", "-u", f"{W}/anchorflow/exe/pngs2mp4.py",
                    "--dir", src, "--out", mp4, "--fps", "30"],
                   cwd=f"{W}/anchorflow")


print(f"[칸] gasp {a.shape} {a.material}  대상 s={s_test} / 참조 s_conv={s_ref}"
      f"  프레임 {a.frames}", flush=True)
os.makedirs(f"{O}/vq", exist_ok=True)
if a.only_test:
    dt = go(s_test, "test")
    mk_mp4(dt, f"{O}/vq/gasp_{a.shape}_{a.material}_test.mp4")
    print("VQ_CELL_DONE", flush=True)
    raise SystemExit(0)

H5 = f"{W}/gaspsim/{a.shape}_{a.material}_s{s_ref}"
dr = go(s_ref, "ref", h5dir=f"{H5}/simulation_ply")
cfg = dict(material=MATNAME[a.material], E=2e6, nu=0.2, density=1000.0,
           n_grid=a.n_grid, grid_lim=2.0, frame_dt=1.0 / 60.0,
           g=[0.0, 0.0, -9.8], vol_mode="uniform",
           boundary_conditions=[dict(type="surface_collider",
                                     point=[0.0, 0.0, a.floor],
                                     normal=[0.0, 0.0, 1.0], surface="sticky",
                                     start_time=0.0, end_time=1e9)])
json.dump(cfg, open(f"{H5}/cfg.json", "w"), indent=1)
dt = go(s_test, "test")

mj = f"{O}/vq/gasp_{a.shape}_{a.material}_vq.json"
subprocess.run(["python", "-u", f"{W}/anchorflow/exe/vq_metrics.py",
                "--ref", dr, "--test", dt, "--out", mj, "--win", str(a.win),
                "--label", f"gasp {a.shape} {a.material} s={s_test} "
                           f"vs s_conv={s_ref}"],
               cwd=f"{W}/anchorflow",
               env=dict(os.environ, PYTHONPATH=f"{W}/anchorflow/lib",
                        PYTHONUTF8="1"))
for src, tag in ((dt, "test"), (dr, "ref")):
    mk_mp4(src, f"{O}/vq/gasp_{a.shape}_{a.material}_{tag}.mp4")

# 잔차는 참조 궤적으로 (PG 와 같다)
rj = f"{O}/resid/gasp_{a.shape}_{a.material}.json"
os.makedirs(f"{O}/resid", exist_ok=True)
subprocess.run(["python", "-u", f"{W}/anchorflow/exe/ana_traj_resid.py",
                "--dir", f"{H5}/simulation_ply", "--cfg", f"{H5}/cfg.json",
                "--out", rj, "--png", rj.replace(".json", ".png"),
                "--label", f"GASP {a.shape} {a.material} reference "
                           f"(s={s_ref})"],
               cwd=f"{W}/anchorflow",
               env=dict(os.environ, PYTHONPATH=f"{W}/anchorflow/lib",
                        PYTHONUTF8="1"))
print("VQ_CELL_DONE", flush=True)
