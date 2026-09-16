"""GaussianFluent 가 떨군 입자 궤적으로 Scene 을 세운다.

`scene_setup.build` 는 PLY 와 PhysGaussian 전처리(회전 -> sim_area 크롭 ->
정규화 -> 내부 채움)를 전제한다. 여기 입력은 저쪽 CD-MPM 이 그 전처리를 이미
마치고 내놓은 MPM 입자라 그 경로를 다시 태울 것이 없다 -- 같은 원시 요소
(AnchorSet, AnchorElasticSim)로 Scene 만 직접 세운다.

두 스크립트가 이 구성을 공유한다: 기하를 학습하는 `exe/fit_gf_geom.py` 와
그 기하의 지지 크기를 재는 `exe/plot_anchor_support.py`. 같은 앵커·같은 표본
위에서 재지 않으면 둘의 수치를 나란히 읽을 수 없어서 여기로 뺐다.
"""
from __future__ import annotations

import glob
import json
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .anchor_mpm import AnchorElasticSim, lame_from_E_nu
from .anchors import AnchorSet
from .scene_setup import Scene


def load_x(p):
    """h5 한 장의 입자 위치 [N,3]. 저쪽은 [3,N] 으로 쓴다."""
    with h5py.File(p, "r") as f:
        d = np.array(f["x"])
    return d.T if d.shape[0] == 3 else d


def load_traj(h5_dir, stride=2, n_pts=40000, seed=0, dev="cuda", quiet=False):
    """-> (TR [T,n,3] on dev, EXT, cfg 없이). 프레임과 입자를 모두 부분표본한다.

    이 MPM 은 진행하면서 입자를 비유한값으로 떨군다 (watermelon 101 프레임에
    33,646/1,381,100). 처음과 끝 모두 유한한 것만 뽑고, 중간에 떨어진 것은 직전
    값으로 채운다 -- 그 프레임 전체를 버리는 것보다 낫다.
    """
    files = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))[::stride]
    if not files:
        raise SystemExit(f"h5 가 없다: {h5_dir}")
    X0f = torch.from_numpy(load_x(files[0])).float()
    XLf = torch.from_numpy(load_x(files[-1])).float()
    ok = torch.isfinite(X0f).all(1) & torch.isfinite(XLf).all(1)
    cand = torch.nonzero(ok).squeeze(-1)
    g = torch.Generator().manual_seed(seed)
    sel = cand[torch.randperm(cand.numel(), generator=g)[:n_pts]].sort().values
    if not quiet:
        print(f"[입자] 전체 {X0f.shape[0]}, 두 끝 모두 유한 {cand.numel()}, "
              f"표본 {sel.numel()}", flush=True)
    TR = torch.stack([torch.from_numpy(load_x(p)).float()[sel]
                      for p in tqdm(files, desc="궤적", ncols=80,
                                    disable=quiet)]).to(dev)
    bad = ~torch.isfinite(TR).all(-1)
    if bad.any():
        if not quiet:
            print(f"[보정] 중간 프레임의 비유한값 {int(bad.sum())} 개를 직전 값으로",
                  flush=True)
        for t in range(1, TR.shape[0]):
            m = bad[t]
            if m.any():
                TR[t][m] = TR[t - 1][m]
    X0 = TR[0].contiguous()
    EXT = float((X0.max(0).values - X0.min(0).values).norm())
    if not quiet:
        print(f"[씬] 프레임 {TR.shape[0]}, 물체 대각 {EXT:.4f}, 최대 변위 "
              f"{100*float((TR-X0).norm(dim=-1).max())/EXT:.1f}%", flush=True)
    return TR, EXT


def build_scene(X0, cfg, n_anchors=512, K=8, eig_floor=0.02, dev="cuda",
                quiet=False):
    """정준 배치 X0 [N,3] 과 씬 config 로 Scene 하나."""
    N = X0.shape[0]
    n_grid = int(cfg.get("n_grid", 100))
    dx = float(cfg.get("grid_lim", 2.0)) / n_grid
    vi = (X0 / dx).long().clamp(0, n_grid - 1)
    flat = (vi[:, 0] * n_grid + vi[:, 1]) * n_grid + vi[:, 2]
    cnt = torch.zeros(n_grid ** 3, device=dev).index_add_(
        0, flat, torch.ones(N, device=dev))
    volume = ((dx ** 3) / cnt[flat]).contiguous()
    E = torch.full((N,), float(cfg["E"]), device=dev)
    nu = torch.full((N,), float(cfg["nu"]), device=dev)
    dens = torch.full((N,), float(cfg["density"]), device=dev)
    mu, lam = lame_from_E_nu(E, nu)
    keep = torch.ones(N, dtype=torch.bool, device=dev)

    aset, _ = AnchorSet.from_gaussians(X0, node_num=n_anchors, latent_dim=0,
                                       e_dim=0, K=K)
    ac = aset.canonical.clone().contiguous()
    radius = AnchorElasticSim(X0, ac, K=K).radius
    sim = AnchorElasticSim(X0, ac, K=K, radius=radius)
    sim.eig_floor = eig_floor
    sim.rot_fallback = True
    w0 = sim._weights(X0, ac)
    M = ac.shape[0]
    mass = torch.zeros(M, device=dev).index_add_(
        0, sim.nn_idx.reshape(-1),
        ((dens * volume).unsqueeze(-1) * w0).reshape(-1)).clamp(min=1e-12)
    fixed = torch.zeros(M, dtype=torch.bool, device=dev)
    for bc in cfg.get("boundary_conditions", []):
        if bc["type"] == "cuboid":
            c_ = torch.tensor(bc["point"], device=dev)
            s_ = torch.tensor(bc["size"], device=dev)
            fixed |= ((ac - c_).abs() <= s_).all(-1)
    if not quiet:
        print(f"[앵커] {M} 개, 반경 {radius:.5f}, 고정 {int(fixed.sum())}",
              flush=True)
    return Scene(cfg=cfg, xyz_world=X0, pos=X0, keep=keep, volume=volume, mu=mu,
                 lam=lam, crop=None, anchor_canonical=ac, mass=mass,
                 fixed_mask=fixed, sim=sim,
                 gravity=torch.tensor(cfg["g"], dtype=torch.float32, device=dev),
                 n_grid=n_grid, sub_dt=float(cfg.get("substep_dt", 1e-4)),
                 damping=float(cfg.get("grid_v_damping_scale", 1.0)),
                 to_mpm=lambda x: x, undo=lambda x: x)


def read_cfg(p):
    return json.load(open(p))
