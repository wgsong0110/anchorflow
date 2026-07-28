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
from anchorflow.ssm_dynamics import HopDynamics, build_graph


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
    ap.add_argument("--curriculum_start_len", type=int, default=8,
                     help="rollout length (consecutive next-frame hops from t=0) at step 0")
    ap.add_argument("--curriculum_iters", type=int, default=0,
                     help="steps to linearly ramp rollout length from curriculum_start_len to T-1 "
                          "(full trajectory); 0 = default to iters//2. Held at T-1 after this point.")
    ap.add_argument("--tbptt_chunk", type=int, default=50,
                     help="detach (h, p) from the autograd graph every this many hops during the "
                          "rollout, bounding backprop-through-time length once curriculum length "
                          "exceeds it (each hop's d_p/d_rot/local_rotation is still directly "
                          "supervised via the loss at every visited frame regardless -- this only "
                          "cuts how far a later frame's loss can backprop into earlier hops' "
                          "weights). 0 disables (full BPTT through the whole rollout).")
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
    curriculum_iters = args.curriculum_iters or (args.iters // 2)
    max_len = T - 1  # full trajectory: t=1..T-1
    import math

    pbar = trange(start_step, args.iters, desc="hop_autoreg")
    for step in pbar:
        t = min(step / max(args.iters - 1, 1), 1.0)
        lr = math.exp(math.log(args.lr) * (1 - t) + math.log(lr_final) * t)
        for g in opt.param_groups:
            g["lr"] = lr

        # Curriculum: always hop from t=0, one dt_base-sized next-frame step at a
        # time (no variable dt) -- ramp the rollout length L linearly from
        # curriculum_start_len up to the full trajectory (max_len) over
        # curriculum_iters steps, then hold at max_len. This directly trains the
        # exact regime needed at dense-inference time (many consecutive small-dt
        # hops), instead of a random-dt/random-subset regime that never covered
        # long consecutive chains (see 2026-07-28 divergence diagnosis).
        frac = min(step / max(curriculum_iters, 1), 1.0)
        L = min(max_len, args.curriculum_start_len + int(frac * (max_len - args.curriculum_start_len)))

        spatial = model.spatial_embed(anchors.e, edge_index, anchors.canonical)
        p, h = anchors.canonical, init_h
        pred_pos_list, pred_rot_list, pred_lrot_list = [], [], []
        for i in range(1, L + 1):
            d_p, d_rot, h = model.hop(spatial, h, anchors.e, dt_base)
            p = p + d_p
            pred_pos_list.append(p)
            pred_rot_list.append(d_rot)
            pred_lrot_list.append(model.local_rotation(h))
            if args.tbptt_chunk > 0 and i % args.tbptt_chunk == 0:
                h = h.detach()
                p = p.detach()

        pred_pos = torch.stack(pred_pos_list, dim=0)   # [L,M,3]
        pred_rot = torch.stack(pred_rot_list, dim=0)
        tgt_pos = teacher_pos[1:L + 1]
        tgt_rot = teacher_rot[1:L + 1]

        loss_pos = torch.nn.functional.mse_loss(pred_pos, tgt_pos)
        loss_rot = quat_mse_loss(pred_rot.reshape(-1, 4), tgt_rot.reshape(-1, 4))
        loss = loss_pos + args.lambda_rot * loss_rot

        if teacher_local_rot is not None:
            pred_local_rot = torch.stack(pred_lrot_list, dim=0)
            tgt_local_rot = teacher_local_rot[1:L + 1]
            loss_local_rot = quat_mse_loss(pred_local_rot.reshape(-1, 4), tgt_local_rot.reshape(-1, 4))
            loss = loss + args.lambda_local_rot * loss_local_rot
        else:
            loss_local_rot = torch.zeros((), device=dev)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0:
            pbar.set_postfix({"L": L, "loss": f"{loss.item():.6f}", "pos": f"{loss_pos.item():.6f}",
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
