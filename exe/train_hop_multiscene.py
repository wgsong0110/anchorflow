#!/usr/bin/env python
"""
Multi-scene HopDynamics training: ONE shared GNN+SSM dynamics core (HopDynamics)
trained jointly across multiple scenes' anchor trajectories, each scene keeping
its own AnchorSet (canonical positions + per-anchor embedding `e`) and its own
learned init_h -- testing the hypothesis that the underlying dynamics (how an
anchor's motion evolves hop-to-hop) are scene-agnostic, so sharing the SSM
across scenes should train faster / generalize better than training one SSM
per scene from scratch (see train_hop_autoreg_anchortraj.py's single-scene
density-curriculum baseline for direct comparison).

Same "density" curriculum as train_hop_autoreg_anchortraj.py (always cover the
full per-scene trajectory from step 0, equally-spaced hop count ramps
sparse-to-dense) -- applied per scene on whichever scene's turn it is
(round-robin), using a GLOBAL step count for the shared curriculum schedule.

No photometric loss here on purpose: this experiment isolates the "shared SSM
across scenes" effect against the anchor-space-only baseline, not photometric
supervision (a separate, already-running axis of variation).

Usage:
  python exe/train_hop_multiscene.py \\
      --scene jumpingjacks=/workspace/scgs_jump_anchor_traj_v2.pt \\
      --scene hellwarrior=/workspace/scgs_hellwarrior_anchor_traj.pt \\
      --out /workspace/hop_multiscene \\
      --iters 20000
"""
from __future__ import annotations

import sys, os, argparse, subprocess, math, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import trange

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)
from anchorflow.anchors import AnchorSet
from anchorflow.ssm_dynamics import HopDynamics, build_graph


def quat_mse_loss(pred, target):
    """See train_hop_autoreg_anchortraj.py's quat_mse_loss docstring: plain
    MSE, not normalize()-based, to avoid a dead-gradient fixed point at the
    rot_decoder's zero-init."""
    return torch.nn.functional.mse_loss(pred, target)


class Scene:
    def __init__(self, name, traj_path, args, dev):
        self.name = name
        data = torch.load(traj_path, map_location=dev)
        self.canonical = data["canonical_nodes"].to(dev).float()
        self.teacher_pos = data["anchor_pos"].to(dev).float()
        self.teacher_rot = data["anchor_rot"].to(dev).float()
        teacher_local_rot = data.get("anchor_local_rot")
        self.teacher_local_rot = teacher_local_rot.to(dev).float() if teacher_local_rot is not None else None
        self.times_f = data["times"]
        self.T, self.M = self.teacher_pos.shape[0], self.teacher_pos.shape[1]
        self.dt_base = self.times_f[1] - self.times_f[0]
        self.max_len = self.T - 1

        self.anchors = AnchorSet.from_trajectory(self.canonical, latent_dim=args.latent_dim,
                                                   e_dim=args.e_dim, K=args.anchor_k).to(dev)
        self.edge_index = build_graph(self.anchors.canonical.detach(), {"graph": "knn", "k": args.k_graph})
        self.init_h = torch.nn.Parameter(0.01 * torch.randn(self.M, args.ssm_dim, device=dev))
        print(f"[scene] {name}: T={self.T} M={self.M} dt_base={self.dt_base:.6f} "
              f"has_local_rot={self.teacher_local_rot is not None}")

    def params(self):
        p = [self.init_h]
        if self.anchors.e is not None:
            p.append(self.anchors.e)
        return p

    def state_dict(self):
        return {"anchors": self.anchors.state_dict(), "init_h": self.init_h.detach().cpu()}

    def load_state_dict(self, sd, dev):
        self.anchors.load_state_dict(sd["anchors"])
        self.init_h.data.copy_(sd["init_h"].to(dev))


def frame_indices_for(scene, n):
    T, max_len = scene.T, scene.max_len
    if n >= max_len:
        return list(range(1, T))
    if n <= 1:
        return [T - 1]
    return sorted(set(1 + round(i * (T - 2) / (n - 1)) for i in range(n)))


def rollout_and_loss(model, scene, frame_indices, args, dev):
    spatial = model.spatial_embed(scene.anchors.e, scene.edge_index, scene.anchors.canonical)
    p, h, prev_t = scene.anchors.canonical, scene.init_h, 0
    pred_pos_list, pred_rot_list, pred_lrot_list = [], [], []
    for hop_i, cur_t in enumerate(frame_indices, start=1):
        dt_hop = (cur_t - prev_t) * scene.dt_base
        d_p, d_rot, h = model.hop(spatial, h, scene.anchors.e, dt_hop)
        p = p + d_p
        pred_pos_list.append(p)
        pred_rot_list.append(d_rot)
        pred_lrot_list.append(model.local_rotation(h))
        prev_t = cur_t
        if args.tbptt_chunk > 0 and hop_i % args.tbptt_chunk == 0:
            h = h.detach()
            p = p.detach()

    pred_pos = torch.stack(pred_pos_list, dim=0)
    pred_rot = torch.stack(pred_rot_list, dim=0)
    idx_t = torch.tensor(frame_indices, device=dev)
    tgt_pos = scene.teacher_pos[idx_t]
    tgt_rot = scene.teacher_rot[idx_t]

    loss_pos = torch.nn.functional.mse_loss(pred_pos, tgt_pos)
    loss_rot = quat_mse_loss(pred_rot.reshape(-1, 4), tgt_rot.reshape(-1, 4))
    loss = loss_pos + args.lambda_rot * loss_rot

    if scene.teacher_local_rot is not None:
        pred_local_rot = torch.stack(pred_lrot_list, dim=0)
        tgt_local_rot = scene.teacher_local_rot[idx_t]
        loss_local_rot = quat_mse_loss(pred_local_rot.reshape(-1, 4), tgt_local_rot.reshape(-1, 4))
        loss = loss + args.lambda_local_rot * loss_local_rot
    else:
        loss_local_rot = torch.zeros((), device=dev)

    return loss, loss_pos, loss_rot, loss_local_rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", action="append", required=True, metavar="NAME=TRAJ_PATH",
                     help="repeatable; e.g. --scene jumpingjacks=/workspace/scgs_jump_anchor_traj_v2.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_final_ratio", type=float, default=0.05)
    ap.add_argument("--lambda_rot", type=float, default=0.1)
    ap.add_argument("--lambda_local_rot", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mp_steps", type=int, default=6)
    ap.add_argument("--ssm_dim", type=int, default=128)
    ap.add_argument("--e_dim", type=int, default=8)
    ap.add_argument("--latent_dim", type=int, default=8)
    ap.add_argument("--n_time_freqs", type=int, default=6)
    ap.add_argument("--k_graph", type=int, default=8)
    ap.add_argument("--anchor_k", type=int, default=4)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--curriculum_start_n", type=int, default=8)
    ap.add_argument("--curriculum_iters", type=int, default=0, help="0 = iters//2")
    ap.add_argument("--tbptt_chunk", type=int, default=50)
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__),
                             capture_output=True, text=True).stdout.strip()
    print(f"[train_hop_multiscene] commit={commit} args={vars(args)}")
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    scenes = []
    for spec in args.scene:
        name, path = spec.split("=", 1)
        scenes.append(Scene(name, path, args, dev))
    n_scenes = len(scenes)

    model = HopDynamics(hidden=args.hidden, mp_steps=args.mp_steps, ssm_dim=args.ssm_dim,
                         e_dim=args.e_dim, n_time_freqs=args.n_time_freqs).to(dev)

    params = list(model.parameters())
    for sc in scenes:
        params += sc.params()
    opt = torch.optim.Adam(params, lr=args.lr)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ckpt["model"])
        for sc in scenes:
            sc.load_state_dict(ckpt["scenes"][sc.name], dev)
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"] + 1
        print(f"[resume] from step {start_step}")

    lr_final = args.lr * args.lr_final_ratio
    curriculum_iters = args.curriculum_iters or (args.iters // 2)
    ema_loss = {sc.name: None for sc in scenes}

    pbar = trange(start_step, args.iters, desc="hop_multiscene")
    for step in pbar:
        t = min(step / max(args.iters - 1, 1), 1.0)
        lr = math.exp(math.log(args.lr) * (1 - t) + math.log(lr_final) * t)
        for g in opt.param_groups:
            g["lr"] = lr

        scene = scenes[step % n_scenes]
        frac = min(step / max(curriculum_iters, 1), 1.0)
        n = min(scene.max_len, args.curriculum_start_n + int(frac * (scene.max_len - args.curriculum_start_n)))
        frame_indices = frame_indices_for(scene, n)

        loss, loss_pos, loss_rot, loss_local_rot = rollout_and_loss(model, scene, frame_indices, args, dev)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        ema_loss[scene.name] = loss.item() if ema_loss[scene.name] is None else 0.98 * ema_loss[scene.name] + 0.02 * loss.item()

        if step % 20 == 0:
            pbar.set_postfix({"scene": scene.name, "n": len(frame_indices), "loss": f"{loss.item():.6f}",
                               "pos": f"{loss_pos.item():.6f}", "rot": f"{loss_rot.item():.6f}",
                               "lrot": f"{loss_local_rot.item():.6f}", "lr": f"{lr:.2e}"}, refresh=False)

        if step % args.ckpt_every == 0 or step == args.iters - 1:
            ckpt = {"step": step, "model": model.state_dict(),
                    "scenes": {sc.name: sc.state_dict() for sc in scenes},
                    "opt": opt.state_dict(), "args": vars(args), "commit": commit}
            torch.save(ckpt, os.path.join(args.out, "ckpt_last.pt"))
            torch.save(ckpt, os.path.join(args.out, f"ckpt_{step:06d}.pt"))
            ema_str = " ".join(f"{k}={v:.6f}" if v is not None else f"{k}=n/a" for k, v in ema_loss.items())
            print(f"\n[step {step}] saved -- ema_loss: {ema_str}", flush=True)

    print("[train] done.")


if __name__ == "__main__":
    main()
