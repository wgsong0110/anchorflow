#!/usr/bin/env python
"""Profile ONE training step of train_anchorflow.py, component by component.

Mirrors the real training path exactly (official model + cameras.json +
official render + checkpointed render + fused LBS + cached conditioning) so the
numbers are actionable rather than indicative.

    python exe/profile_step.py --model /workspace/gs_official/kitchen \
        --cfg cfg/anchorflow_kitchen.yaml
"""
from __future__ import annotations

import argparse, json, os, sys, time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf

sys.path.append("/workspace/gaussian-splatting")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow.nodeflow import NodeFlow, _LBS_CUDA
from anchorflow.sds import SVDGuidance


class Cam:
    def __init__(self, R, T, fovx, fovy, W, H):
        self.image_width, self.image_height = W, H
        self.FoVx, self.FoVy = fovx, fovy
        self.znear, self.zfar = 0.01, 100.0
        w2v = torch.tensor(getWorld2View2(R, T)).T.cuda()
        proj = getProjectionMatrix(self.znear, self.zfar, fovx, fovy).T.cuda()
        self.world_view_transform = w2v
        self.full_proj_transform = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center = w2v.inverse()[3, :3]


class Pipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


class Timer:
    def __init__(self): self.rows = []
    def __call__(self, label, fn, n=1):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = None
        for _ in range(n):
            out = fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n * 1000
        mem = torch.cuda.memory_allocated() // (1024**2)
        self.rows.append((label, dt, mem))
        print(f"  {label:38s} {dt:8.1f} ms   mem={mem:6d} MiB")
        return out
    def report(self):
        tot = sum(r[1] for r in self.rows)
        print("\n=== 비중 ===")
        for label, dt, _ in sorted(self.rows, key=lambda r: -r[1]):
            print(f"  {label:38s} {dt:8.1f} ms  {dt/tot*100:5.1f}%  "
                  f"{'#'*int(dt/tot*50)}")
        print(f"  {'합계':38s} {tot:8.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cfg",   required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    dev, T = "cuda", cfg.model.n_frames
    tm = Timer()

    print("=== 로드 ===")
    g = GaussianModel(3)
    g.load_ply(f"{args.model}/point_cloud/iteration_30000/point_cloud.ply")
    g.active_sh_degree = 3
    canon = g.get_xyz.detach().clone()
    G = canon.shape[0]
    print(f"  gaussians={G}  LBS_CUDA={_LBS_CUDA}")

    cj = json.load(open(f"{args.model}/cameras.json"))[0]
    rot = np.array(cj["rotation"], dtype=np.float32); pos = np.array(cj["position"], dtype=np.float32)
    W, H = cj["width"], cj["height"]
    fovx, fovy = focal2fov(cj["fx"], W), focal2fov(cj["fy"], H)
    s = cfg.model.res / max(W, H)
    W8, H8 = max(8, int(round(W*s/8))*8), max(8, int(round(H*s/8))*8)
    cam = Cam(rot, -rot.T @ pos, fovx, fovy, W8, H8)
    print(f"  render {W8}x{H8}")

    bg = torch.zeros(3, device=dev)
    model = NodeFlow(canonical_xyz=canon, n_nodes=cfg.model.n_nodes, n_frames=T,
                     hidden=cfg.model.hidden, n_gnn_layers=cfg.model.n_gnn_layers,
                     k_node=cfg.model.k_node, k_gauss=cfg.model.k_gauss,
                     dt=float(cfg.model.get("dt", 0.1))).to(dev)
    K = model.n_nodes
    z0 = torch.nn.Parameter(torch.randn(K, 3, device=dev) * 0.24)

    svd = SVDGuidance(sigma_min=cfg.mds.sigma_min, sigma_max=cfg.mds.sigma_max,
                      guidance_scale=cfg.mds.guidance_scale,
                      motion_bucket_id=cfg.mds.motion_bucket_id,
                      grad_clip=cfg.mds.grad_clip, device=dev)

    def render_at(c, xyz):
        g._xyz = xyz
        return render(c, g, Pipe(), bg)["render"]

    # warmup
    with torch.no_grad():
        frame0 = render_at(cam, canon).clamp(0, 1)
    cond = svd.precompute_cond(frame0, T)

    print("\n=== 컴포넌트별 (forward) ===")
    h = tm("1. GNN encode_scene", lambda: model.encode_scene())
    nd = tm("2. rollout_nodes [T-1,K,3]", lambda: model.rollout_nodes(h, z0))

    tm("3. LBS 1프레임 [G,3]", lambda: model.lbs_frame(nd[0]), n=5)
    model._use_lbs_cuda = False
    tm("3b. LBS 1프레임 (torch 참조)", lambda: model.lbs_frame(nd[0]), n=5)
    model._use_lbs_cuda = _LBS_CUDA

    with torch.no_grad():
        tm("4. render 1프레임 (no_grad)", lambda: render_at(cam, canon), n=5)

    def render_all_ckpt():
        fr = [frame0]
        for i in range(T - 1):
            d = model.lbs_frame(nd[i])
            fr.append(checkpoint(lambda x: render_at(cam, x), canon + d,
                                 use_reentrant=False))
        return torch.stack(fr, 0)

    def render_all_plain():
        fr = [frame0]
        for i in range(T - 1):
            d = model.lbs_frame(nd[i])
            fr.append(render_at(cam, canon + d))
        return torch.stack(fr, 0)

    frames = tm(f"5. LBS+render {T}프레임 (checkpoint)", render_all_ckpt)
    peak_ckpt = torch.cuda.max_memory_allocated() // (1024**2)

    tm("6. VAE encode_frames (grad)", lambda: svd.encode_frames(frames))

    print("\n=== MDS 전체 (forward) ===")
    loss = tm("7. mds_loss (cache 사용)",
              lambda: svd.mds_loss(frames, cond_image=frame0,
                                   w_power=cfg.mds.w_power, cond_cache=cond))
    tm("7b. mds_loss (cache 없이)",
       lambda: svd.mds_loss(frames, cond_image=frame0, w_power=cfg.mds.w_power))

    print("\n=== backward ===")
    tm("8. loss.backward()", lambda: loss.backward(retain_graph=True))

    tm.report()

    print("\n=== 메모리 ===")
    print(f"  peak allocated: {torch.cuda.max_memory_allocated()//(1024**2)} MiB")
    print(f"  peak reserved:  {torch.cuda.max_memory_reserved()//(1024**2)} MiB")
    print(f"  (checkpoint 경로 peak: {peak_ckpt} MiB)")


if __name__ == "__main__":
    main()
