#!/usr/bin/env python
"""
Autoregressive HopDynamics training via anchor-trajectory distillation.

Instead of photometric (render) supervision, this trains HopDynamics directly
in anchor space against a teacher trajectory extracted from a converged
SC-GS checkpoint (see exe/*_anchor_traj extraction snippet, saved as a .pt
with canonical_nodes/anchor_pos/anchor_rot/times). SC-GS's own control nodes
are used AS anchorflow's anchor set (AnchorSet.from_trajectory) so there is
no anchor-correspondence/registration problem: the teacher targets map 1:1
to our own anchors by construction.

No rendering, no rasterizer, no Gaussians -- pure anchor-space regression,
so this is much cheaper per step than photometric HopDynamics training.

Usage:
  python exe/train_hop_autoreg_anchortraj.py \\
      --traj /workspace/scgs_jump_anchor_traj.pt \\
      --out  /workspace/jump_hop_autoreg \\
      --iters 20000
"""
from __future__ import annotations

import sys, os, argparse, subprocess
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import trange

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)
from anchorflow.anchors import AnchorSet
from anchorflow.ssm_dynamics import HopDynamics, build_graph, hop_rollout


def quat_mse_loss(pred, target):
    """Plain MSE on raw (unnormalized) quaternion components [N,4].

    NOT normalize()-based on purpose: rot_decoder's last layer is
    zero-initialized (outputs exactly [0,0,0,0] at step 0), and
    F.normalize's gradient at the exact zero vector vanishes (norm()'s
    x/||x|| gradient is 0/0 there) -- a normalize()-based geodesic loss
    gets permanently stuck at that fixed point since it never receives a
    nonzero gradient to escape zero-init. Plain MSE has no such singularity:
    its gradient at pred=0 is simply -2*target, always well-defined and
    nonzero whenever target != 0. Confirmed empirically (2026-07-28):
    training with the normalize()-based loss left rot loss frozen at
    exactly 1.000000 for 145+ steps while pos loss moved fine.
    """
    return torch.nn.functional.mse_loss(pred, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="path to *_anchor_traj.pt (canonical_nodes/anchor_pos/anchor_rot/times)")
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
    ap.add_argument("--frames_per_step", type=int, default=0,
                     help="if >0, visit a random subset of this many frame indices per step "
                          "instead of all T-1 (hop_rollout already takes the actual gap between "
                          "visited frames as dt, so subsampling is a free ~T/frames_per_step "
                          "speedup, not an approximation of the dynamics -- every frame still "
                          "gets supervised, just spread across different random subsets over "
                          "many steps instead of all-at-once every step).")
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__),
                             capture_output=True, text=True).stdout.strip()
    print(f"[train_hop_autoreg_anchortraj] commit={commit} args={vars(args)}")

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    data = torch.load(args.traj, map_location=dev)
    canonical = data["canonical_nodes"].to(dev).float()          # [M,3]
    teacher_pos = data["anchor_pos"].to(dev).float()              # [T,M,3]
    teacher_rot = data["anchor_rot"].to(dev).float()              # [T,M,4]
    teacher_local_rot = data.get("anchor_local_rot")              # [T,M,4] or None (older .pt files)
    if teacher_local_rot is not None:
        teacher_local_rot = teacher_local_rot.to(dev).float()
    times_f = data["times"]                                       # list[float], len T, times_f[0]==0.0
    T, M = teacher_pos.shape[0], teacher_pos.shape[1]
    assert canonical.shape[0] == M, (canonical.shape, teacher_pos.shape)
    dt_base = times_f[1] - times_f[0]
    print(f"[data] T={T} M={M} dt_base={dt_base:.6f} has_local_rot={teacher_local_rot is not None}")

    anchors = AnchorSet.from_trajectory(canonical, latent_dim=args.latent_dim, e_dim=args.e_dim, K=args.anchor_k).to(dev)
    graph_cfg = {"graph": "knn", "k": args.k_graph}
    edge_index = build_graph(anchors.canonical.detach(), graph_cfg)

    model = HopDynamics(hidden=args.hidden, mp_steps=args.mp_steps, ssm_dim=args.ssm_dim,
                         e_dim=args.e_dim, n_time_freqs=args.n_time_freqs).to(dev)
    init_h = torch.nn.Parameter(0.01 * torch.randn(M, args.ssm_dim, device=dev))

    params = list(model.parameters()) + [init_h]
    if anchors.e is not None:
        params += [anchors.e]
    opt = torch.optim.Adam(params, lr=args.lr)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ckpt["model"])
        init_h.data.copy_(ckpt["init_h"])
        anchors.load_state_dict(ckpt["anchors"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"] + 1
        print(f"[resume] from step {start_step}")

    lr_final = args.lr * args.lr_final_ratio
    all_frames = list(range(1, T))  # hop_rollout visits t=1..T-1; t=0 is implicit canonical/identity
    import random, math

    pbar = trange(start_step, args.iters, desc="hop_autoreg")
    for step in pbar:
        t = min(step / max(args.iters - 1, 1), 1.0)
        lr = math.exp(math.log(args.lr) * (1 - t) + math.log(lr_final) * t)
        for g in opt.param_groups:
            g["lr"] = lr

        if args.frames_per_step > 0 and args.frames_per_step < len(all_frames):
            frame_indices = sorted(random.sample(all_frames, args.frames_per_step))
        else:
            frame_indices = all_frames

        p_by_t, rot_by_t, h_by_t = hop_rollout(model, anchors.canonical, anchors.e, edge_index,
                                                times=frame_indices, dt_base=dt_base, grad=True, h0=init_h)

        pred_pos = torch.stack([p_by_t[i] for i in frame_indices], dim=0)   # [len(frame_indices),M,3]
        pred_rot = torch.stack([rot_by_t[i] for i in frame_indices], dim=0)
        idx_t = torch.tensor(frame_indices, device=dev)
        tgt_pos = teacher_pos[idx_t]
        tgt_rot = teacher_rot[idx_t]

        loss_pos = torch.nn.functional.mse_loss(pred_pos, tgt_pos)
        loss_rot = quat_mse_loss(pred_rot.reshape(-1, 4), tgt_rot.reshape(-1, 4))
        loss = loss_pos + args.lambda_rot * loss_rot

        if teacher_local_rot is not None:
            pred_local_rot = torch.stack([model.local_rotation(h_by_t[i]) for i in frame_indices], dim=0)
            tgt_local_rot = teacher_local_rot[idx_t]
            loss_local_rot = quat_mse_loss(pred_local_rot.reshape(-1, 4), tgt_local_rot.reshape(-1, 4))
            loss = loss + args.lambda_local_rot * loss_local_rot
        else:
            loss_local_rot = torch.zeros((), device=dev)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.6f}", "pos": f"{loss_pos.item():.6f}",
                               "rot": f"{loss_rot.item():.6f}", "lrot": f"{loss_local_rot.item():.6f}",
                               "lr": f"{lr:.2e}"})

        if step % args.ckpt_every == 0 or step == args.iters - 1:
            ckpt = {"step": step, "model": model.state_dict(), "init_h": init_h.detach().cpu(),
                    "anchors": anchors.state_dict(), "opt": opt.state_dict(),
                    "args": vars(args), "commit": commit}
            torch.save(ckpt, os.path.join(args.out, "ckpt_last.pt"))
            torch.save(ckpt, os.path.join(args.out, f"ckpt_{step:06d}.pt"))
            print(f"\n[step {step}] saved (pos={loss_pos.item():.6f} rot={loss_rot.item():.6f} "
                  f"local_rot={loss_local_rot.item():.6f})")

    print("[train] done.")


if __name__ == "__main__":
    main()
