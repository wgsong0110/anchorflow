"""베이스라인들의 한 프레임 비용을 같은 기준으로 잰다.

`bench_rollout.py` 는 우리 학생과 MPM 만 잰다. 여기서는 소성 비교에 쓰는 베이스라인을
같은 씬·같은 프레임 정의로 나란히 잰다:

  - PhysGaussian / DreamPhysics MPM  (warp)
  - PhysDreamer 시뮬레이터           (warp, 같은 구성식·다른 구현)
  - meshgs 사면체 FEM                (torch)

**한 프레임**은 물리 시간 `frame_dt` 를 뜻한다. 솔버마다 서브스텝 수가 다르므로
서브스텝당이 아니라 프레임당으로 재야 비교가 된다. 래스터화는 세 방법이 같은
가우시안 집합을 그리므로 따로 재서 한 번만 보고한다 (합산할 때는 각자에 더한다).

경고: 여기 숫자는 **실행 속도**일 뿐 정확도와 무관하다. 발산하는 설정도 빠르게
발산한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--scgs", required=True)
ap.add_argument("--anchorflow", required=True)
ap.add_argument("--physgaussian", default=None)
ap.add_argument("--physdreamer", default=None)
ap.add_argument("--meshgs", default=None, help="meshgs lib 경로")
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=30, help="측정 프레임 수")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--width", type=int, default=800)
ap.add_argument("--height", type=int, default=800)
ap.add_argument("--student_ckpt", default=None,
                help="영상학습 학생 체크포인트")
ap.add_argument("--n_grid", type=int, default=100)
a = ap.parse_args()

sys.path.insert(0, a.anchorflow)
sys.path.insert(0, a.scgs)
dev = "cuda"
torch.set_grad_enabled(False)

import warp as wp  # noqa: E402

wp.init()
from anchorflow import scene_setup  # noqa: E402

sc = scene_setup.build(a.ply, a.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
cfg = sc.cfg
FRAME_DT = float(cfg.get("frame_dt", 0.01))
SUB_DT = float(cfg.get("substep_dt", 1e-4))
n_sub = max(1, int(round(FRAME_DT / SUB_DT)))
mat = sc.pos[sc.keep].contiguous()
N = mat.shape[0]
print(f"[씬] 물질 입자 {N}, frame_dt {FRAME_DT}, 서브스텝/프레임 {n_sub}", flush=True)

res = {"n_particles": N, "frame_dt": FRAME_DT, "substeps_per_frame": n_sub,
       "methods": {}}


def timed(fn, frames, warmup, label):
    """프레임당 ms. 앞 warmup 회는 버린다."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(frames):
        fn()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / frames * 1000.0
    print(f"  {label:34s} {ms:8.2f} ms/frame  ({1000.0/ms:7.1f} fps)", flush=True)
    return ms


# ---------------- 1. 우리 MPM 교사 (PhysGaussian 계열 warp 솔버) ----------------
if a.physgaussian:
    sys.path.insert(0, a.physgaussian)
try:
    from anchorflow.mpm_teacher import MPMTeacher
    # horizon 이 길면 회전 구동기 두 개가 한 창에 들어와 warp 커널 이름이 충돌한다.
    # 속도만 재므로 짧은 창이면 충분하다.
    T = MPMTeacher(sc, horizon=FRAME_DT * 2)
    T._set(T.pos_m.clone(), torch.zeros(T.n, 3, device=dev), T.eye.clone(),
           torch.zeros_like(T.eye))

    def step_mpm():
        for _ in range(n_sub):
            T.solver.p2g2p(None, float(sc.sub_dt), device=T.wp_dev)

    print("[MPM (PhysGaussian warp 솔버)]", flush=True)
    res["methods"]["mpm_physgaussian"] = timed(
        step_mpm, a.frames, a.warmup, "물리 스텝")
except Exception as e:
    print("  MPM 실패:", type(e).__name__, e, flush=True)

# ---------------- 2. PhysDreamer 시뮬레이터 ----------------
if a.physdreamer:
    try:
        # PhysGaussian 의 mpm_solver_warp 가 같은 이름의 warp_utils 를 들고 있어
        # 먼저 임포트되면 PhysDreamer 쪽이 그것을 먹는다. 경로를 앞에 두고
        # 캐시를 지운 뒤 임포트한다.
        for _m in [k for k in sys.modules
                   if k.split(".")[0] in ("warp_utils", "mpm_utils",
                                          "mpm_data_structure")]:
            del sys.modules[_m]
        sys.path.insert(0, os.path.join(a.physdreamer, "physdreamer", "warp_mpm"))
        sys.path.insert(0, a.physdreamer)
        from physdreamer.warp_mpm.mpm_data_structure import (MPMModelStruct,
                                                             MPMStateStruct)
        from physdreamer.warp_mpm.mpm_solver_diff import MPMWARPDiff
        st = MPMStateStruct()
        st.init(N, device=dev, requires_grad=False)
        st.from_torch(mat.clone(), sc.volume[sc.keep].contiguous().clone(), None,
                      device=dev, requires_grad=False, n_grid=a.n_grid,
                      grid_lim=2.0)
        md = MPMModelStruct()
        md.init(N, device=dev, requires_grad=False)
        md.init_other_params(n_grid=a.n_grid, grid_lim=2.0, device=dev)
        sv = MPMWARPDiff(N, n_grid=a.n_grid, grid_lim=2.0, device=dev)
        sv.set_parameters_dict(md, st, {
            "material": cfg.get("material", "metal"),
            "g": list(cfg.get("g", [0.0, 0.0, 0.0])),
            "density": float(cfg.get("density", 1000)),
            "grid_v_damping_scale": 1.1})
        sv.set_E_nu(md, 1e5, float(cfg.get("nu", 0.3)), device=dev)
        sv.prepare_mu_lam(md, st, device=dev)

        def step_pd():
            for k in range(n_sub):
                sv.p2g2p(md, st, k, SUB_DT, device=dev)

        print("[PhysDreamer 시뮬레이터]", flush=True)
        res["methods"]["physdreamer"] = timed(
            step_pd, a.frames, a.warmup, "물리 스텝")
    except Exception as e:
        print("  PhysDreamer 실패:", type(e).__name__, e, flush=True)

# ---------------- 3. meshgs 사면체 FEM ----------------
if a.meshgs:
    try:
        sys.path.insert(0, a.meshgs)
        from meshgs.fem import TetFEM
        from meshgs.tetcage import build_tet_mesh, dilate_fill, occupancy
        g = torch.stack(torch.meshgrid(
            *[torch.linspace(-0.4, 0.4, 16, device=dev)] * 3, indexing="ij"),
            -1).reshape(-1, 3)
        occ, org, h = occupancy(g, res=32)
        occ = dilate_fill(occ, 1)
        V, Tt = build_tet_mesh(occ, org, h)
        f = TetFEM(V, Tt, density=200.0, E=1e5, nu=0.3, plastic="von_mises",
                   yield_stress=1e3, damping=8.0)
        Vc, vc = V.clone(), torch.zeros_like(V)
        # FEM 은 dt 가 훨씬 작다. 같은 물리 시간(frame_dt)을 덮도록 서브스텝을 맞춘다.
        fem_dt = 2e-4
        fem_sub = max(1, int(round(FRAME_DT / fem_dt)))

        def step_fem():
            global Vc, vc
            for _ in range(fem_sub):
                Vc, vc = f.step(Vc, vc, fem_dt, fixed=None)

        print(f"[meshgs FEM] 정점 {V.shape[0]} 사면체 {Tt.shape[0]}, "
              f"서브스텝/프레임 {fem_sub}", flush=True)
        res["methods"]["meshgs_fem"] = timed(
            step_fem, max(5, a.frames // 3), 2, "물리 스텝")
        res["meshgs_verts"] = int(V.shape[0])
        res["meshgs_tets"] = int(Tt.shape[0])
    except Exception as e:
        print("  meshgs 실패:", type(e).__name__, e, flush=True)

# ---------------- 3.5 우리 학생 (영상학습 스테퍼) ----------------
if a.student_ckpt:
    try:
        from anchorflow.nextstate import NextStep, apply_step
        st_ck = torch.load(a.student_ckpt, map_location=dev, weights_only=False)
        net = NextStep(hidden=128, depth=4, heads=4, use_accel=False,
                       scale=float(sc.extent),
                       vel_scale=float(sc.extent) / max(FRAME_DT, 1e-6),
                       zero_init=True).to(dev)
        net.load_state_dict(st_ck["net"])
        net.eval()
        AC = st_ck["ac"].to(dev)
        v_s = st_ck["v0"].to(dev)
        gp_s = sc.pos.clone()
        # 학생은 chunk 개 스텝을 한 번에 내므로 프레임당 순전파는 1/chunk 회다.
        # 여기서는 프레임 하나를 온전히 만드는 비용(순전파 + 스키닝)을 잰다.
        _p, _v, _g = AC.clone(), v_s.clone(), gp_s.clone()

        def step_student():
            global _p, _v, _g
            q, d = apply_step(net, _p, _v, None, FRAME_DT, sc.fixed_mask)[-1]
            _p, _v = q, d / FRAME_DT
            _g = sc.skin(_p, _g)

        print(f"[우리 학생] 앵커 {AC.shape[0]}", flush=True)
        res["methods"]["student"] = timed(step_student, a.frames, a.warmup,
                                          "순전파 + 스키닝")
        res["n_anchors"] = int(AC.shape[0])
    except Exception as e:
        print("  학생 실패:", type(e).__name__, e, flush=True)

# ---------------- 4. 래스터화 (공통) ----------------
try:
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render as _render
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2

    class MiniCam:
        def __init__(self, w, h, fy, fx, zn, zf, wvt, fp):
            self.image_width, self.image_height = w, h
            self.FoVy, self.FoVx, self.znear, self.zfar = fy, fx, zn, zf
            self.world_view_transform, self.full_proj_transform = wvt, fp
            self.camera_center = wvt.inverse()[3, :3]

    class _P:
        debug = False
        compute_cov3D_python = False
        convert_SHs_python = False

    gs = GaussianModel(3, fea_dim=0)
    gs.load_ply(a.ply)
    NA = gs._xyz.shape[0]
    R = np.eye(3); Tv = np.array([0.0, 0.0, 3.0])
    wvt = torch.tensor(getWorld2View2(R, Tv)).transpose(0, 1).float().to(dev)
    pmx = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=0.8,
                              fovY=0.8).transpose(0, 1).to(dev)
    cam = MiniCam(a.width, a.height, 0.8, 0.8, 0.01, 100.0, wvt,
                  (wvt.unsqueeze(0).bmm(pmx.unsqueeze(0))).squeeze(0))
    ZR = torch.zeros(NA, 4, device=dev); ZR[:, 0] = 1.0
    ZS = torch.zeros(NA, 3, device=dev)
    dx = torch.zeros(NA, 3, device=dev, dtype=sc.pos.dtype)

    def step_render():
        _render(cam, gs, _P(), torch.ones(3, device=dev), dx, ZR, ZS,
                d_rot_as_res=True)

    print(f"[래스터화] 가우시안 {NA}, {a.width}x{a.height}", flush=True)
    res["methods"]["rasterize"] = timed(step_render, a.frames, a.warmup,
                                        "래스터화")
    res["n_gaussians"] = int(NA)
except Exception as e:
    print("  래스터화 실패:", type(e).__name__, e, flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print(f"\n[저장] {a.out}", flush=True)
print("BENCH_OK")
