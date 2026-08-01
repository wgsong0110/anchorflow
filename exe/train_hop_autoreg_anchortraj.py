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
from anchorflow.ssm_dynamics import (
    HopDynamics, build_graph, rest_edge_lengths, xpbd_distance_project, xpbd_directional_project,
)


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
    ap.add_argument("--lambda_anchor", type=float, default=1.0,
                     help="weight on the anchor-space distillation loss (pos+rot+local_rot vs the "
                          "SC-GS teacher trajectory). Set to 0 with --photo_model_path for a "
                          "photometric-loss-ONLY run (no anchor-trajectory supervision at all).")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mp_steps", type=int, default=6)
    ap.add_argument("--ssm_dim", type=int, default=128)
    ap.add_argument("--e_dim", type=int, default=8)
    ap.add_argument("--latent_dim", type=int, default=8)
    ap.add_argument("--n_time_freqs", type=int, default=6)
    ap.add_argument("--k_graph", type=int, default=8)
    ap.add_argument("--anchor_k", type=int, default=4)
    ap.add_argument("--no_ssm", action="store_true",
                     help="ablation: make hop() memoryless -- h_next = u (this hop's raw hop_in "
                          "output) instead of the SSM's decayed recurrent blend. Drops any momentum/"
                          "gait-phase signal beyond the immediately previous hop, but also removes "
                          "the recurrent hidden state as a channel through which error can compound "
                          "over a long autoregressive rollout. Ignored if --stateless_pv is set "
                          "(that mode has no h at all, this flag doesn't apply).")
    ap.add_argument("--stateless_pv", action="store_true",
                     help="use HopDynamics.hop_pv() instead of hop(): explicit (p,v) state, NO "
                          "recurrent h whatsoever, GNN message passing RE-RUN every hop grounded in "
                          "the CURRENT position/velocity (not a one-time canonical spatial embedding "
                          "+ separately-evolving h) -- lets neighbors coordinate continuously "
                          "throughout the rollout instead of only once at t=0. The network predicts "
                          "velocity directly (d_p = v_next * dt) instead of a raw per-hop "
                          "displacement, so position updates scale with elapsed time by construction. "
                          "Costs a full mp_steps-layer GNN pass every hop (like the "
                          "--spatial_from_current_pos experiment that was ~3.6x slower, but this ALSO "
                          "removes hop_in+ssm's cost, so net slowdown may differ).")
    ap.add_argument("--gnn_layer", choices=["interaction", "mpnn"], default="interaction",
                     help="'interaction': InteractionNetwork (GNS-lineage, residual edge state "
                          "carried/evolved layer to layer). 'mpnn': MPNNLayer (Gilmer et al. "
                          "textbook message passing -- edge features are the raw input at every "
                          "layer, no evolving edge state).")
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--curriculum_mode", choices=["window", "density"], default="density",
                     help="'window': always dt=dt_base, ramp how much of the trajectory (from t=0) "
                          "is covered, short-to-long. 'density': always cover the FULL trajectory "
                          "(t=1..T-1) from step 0, ramp how many EQUALLY-SPACED hops are used to "
                          "get there, sparse/large-dt-to-dense/dt_base -- every frame gets gradient "
                          "signal from step 0, and dt shrinks over training instead of the rollout "
                          "horizon growing.")
    ap.add_argument("--curriculum_start_len", type=int, default=8,
                     help="[window mode] rollout length (consecutive next-frame hops from t=0) at step 0")
    ap.add_argument("--curriculum_start_n", type=int, default=8,
                     help="[density mode] number of equally-spaced hops spanning the full "
                          "trajectory at step 0")
    ap.add_argument("--curriculum_iters", type=int, default=0,
                     help="steps to linearly ramp window length / hop count to the full trajectory; "
                          "0 = default to iters//2. Held at the max after this point.")
    ap.add_argument("--tbptt_chunk", type=int, default=50,
                     help="detach (h, p) from the autograd graph every this many hops during the "
                          "rollout, bounding backprop-through-time length once curriculum length "
                          "exceeds it (each hop's d_p/d_rot/local_rotation is still directly "
                          "supervised via the loss at every visited frame regardless -- this only "
                          "cuts how far a later frame's loss can backprop into earlier hops' "
                          "weights). 0 disables (full BPTT through the whole rollout).")
    ap.add_argument("--photo_model_path", type=str, default=None,
                     help="SC-GS checkpoint dir -- if set, adds a photometric L1 rendering loss each "
                          "step using SC-GS's own (frozen) Gaussians/renderer as a differentiable "
                          "decoder of our predicted anchor pos/rot/local_rotation, supervised against "
                          "the REAL GT training images (unlike the anchor-space loss, this needs the "
                          "script run with CWD = the SC-GS repo root, same as eval_hop_autoreg_psnr.py).")
    ap.add_argument("--photo_source_path", type=str, default=None, help="dataset dir (required with --photo_model_path)")
    ap.add_argument("--photo_iteration", type=int, default=-1)
    ap.add_argument("--lambda_photo", type=float, default=1.0)
    ap.add_argument("--photo_views_per_step", type=int, default=2)
    ap.add_argument("--spatial_from_current_pos", action="store_true",
                     help="recompute the GNN spatial embedding every hop from the CURRENT (already "
                          "deformed) absolute anchor position, instead of computing it once from "
                          "canonical before the rollout and reusing it unchanged at every hop. With "
                          "this on, the network directly observes where it actually is at each step "
                          "(grounded, more Markovian) instead of relying purely on the recurrent "
                          "hidden state h to implicitly track how far anchors have moved from "
                          "canonical. Costs one extra spatial_embed (GNN message-passing) call per "
                          "hop.")
    ap.add_argument("--xpbd", action="store_true",
                     help="apply the XPBD distance-constraint projection (structural no-splitting "
                          "guarantee, see ssm_dynamics.xpbd_distance_project) after every hop's raw "
                          "d_p update, during TRAINING (not just eval) -- so the model learns "
                          "against trajectories that already respect the constraint, instead of the "
                          "constraint only being applied post-hoc at inference.")
    ap.add_argument("--xpbd_mode", choices=["distance", "directional", "none"], default="distance",
                     help="'distance': free 3D projection to the closest constraint-satisfying point "
                          "(xpbd_distance_project) -- with many simultaneous neighbor constraints this "
                          "can end up moving an anchor in a direction quite different from its "
                          "predicted d_p, since it only minimizes distance to the (infeasible) "
                          "proposal, not direction fidelity. 'directional': restrict each anchor's "
                          "correction to its OWN fixed direction (d_p normalized) via "
                          "xpbd_directional_project -- guarantees the actual displacement has cosine "
                          "similarity 1 with d_p (the network's intended direction is never "
                          "overridden, only its magnitude is adjusted to fit neighbors), at the cost "
                          "of a weaker constraint-violation bound (fewer degrees of freedom to work "
                          "with per sweep). 'none': apply NO projection at all -- positions are used "
                          "raw (p = p + d_p every hop, no correction, matches eval too). Only a soft "
                          "hinge-loss penalty (see lambda_xpbd_reg) pushes the network to respect "
                          "distances on its own via gradient descent. No structural guarantee "
                          "whatsoever (an untrained/adversarial model can still violate arbitrarily), "
                          "but avoids running gradients through the iterative projection every hop, "
                          "which may itself be part of what was destabilizing training.")
    ap.add_argument("--xpbd_tolerance", type=float, default=0.15,
                     help="allowed edge stretch/compression vs canonical length, e.g. 0.15 = ±15%%")
    ap.add_argument("--xpbd_iters", type=int, default=20,
                     help="Jacobi sweeps per hop -- see demo_xpbd_rollout.py's convergence sweep "
                          "(4->5.3x, 20->2.3x, 50->1.5x max stretch on this same anchor graph)")
    ap.add_argument("--lambda_xpbd_reg", type=float, default=0.1,
                     help="[--xpbd only] weight on ‖xpbd_distance_project(p+d_p) - (p+d_p)‖^2 -- the "
                          "residual the projection had to correct. XPBD is a fixed-iteration Jacobi "
                          "solve, not a closed-form one, so it never perfectly satisfies the distance "
                          "constraint in one hop; this regularizer pushes the network's raw d_p to "
                          "already be close to constraint-satisfying, so the same iteration budget "
                          "converges tighter over training instead of relying on brute-forcing more "
                          "iterations. Keep small relative to lambda_anchor -- the teacher trajectory "
                          "itself may have edges that legitimately stretch beyond xpbd_tolerance "
                          "(real non-rigid deformation in the source video), and this loss must not "
                          "fight that.")
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
    xpbd_rest_len = rest_edge_lengths(anchors.canonical.detach(), edge_index) if args.xpbd else None
    xpbd_src, xpbd_dst = edge_index if args.xpbd else (None, None)

    model = HopDynamics(hidden=args.hidden, mp_steps=args.mp_steps, ssm_dim=args.ssm_dim,
                         e_dim=args.e_dim, n_time_freqs=args.n_time_freqs,
                         use_ssm=not args.no_ssm, stateless=args.stateless_pv,
                         gnn_layer=args.gnn_layer).to(dev)
    # stateless_pv has no recurrent h to initialize -- init_h is created either
    # way to keep the checkpoint format / optimizer param-list code uniform,
    # but it receives no gradient at all in that mode (never referenced).
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
    import math, random

    photo_enabled = args.photo_model_path is not None
    if photo_enabled:
        assert args.photo_source_path, "--photo_source_path required with --photo_model_path"
        sys.path.insert(0, os.getcwd())  # SC-GS repo root -- run this script with CWD there
        from scene import Scene as SCGSScene, DeformModel as SCGSDeformModel
        from gaussian_renderer import GaussianModel as SCGSGaussianModel, render as scgs_render
        from arguments import ModelParams as SCGSModelParams, PipelineParams as SCGSPipelineParams, get_combined_args

        sys.argv = [sys.argv[0], "--model_path", args.photo_model_path, "--source_path", args.photo_source_path,
                    "--deform_type", "node"]
        _parser = argparse.ArgumentParser()
        _model_p = SCGSModelParams(_parser, sentinel=True)
        _pipeline_p = SCGSPipelineParams(_parser)
        _parser.add_argument("--iteration", default=-1, type=int)
        _cargs = get_combined_args(_parser)
        _cargs.source_path = args.photo_source_path
        photo_dataset = _model_p.extract(_cargs)
        photo_pipeline = _pipeline_p.extract(_cargs)

        photo_deform = SCGSDeformModel(K=photo_dataset.K, deform_type=photo_dataset.deform_type, is_blender=photo_dataset.is_blender,
                                        skinning=photo_dataset.skinning, hyper_dim=photo_dataset.hyper_dim, node_num=photo_dataset.node_num,
                                        pred_opacity=photo_dataset.pred_opacity, pred_color=photo_dataset.pred_color,
                                        use_hash=photo_dataset.use_hash, hash_time=photo_dataset.hash_time,
                                        d_rot_as_res=photo_dataset.d_rot_as_res, local_frame=photo_dataset.local_frame,
                                        progressive_brand_time=photo_dataset.progressive_brand_time, max_d_scale=photo_dataset.max_d_scale)
        photo_deform.load_weights(photo_dataset.model_path, iteration=args.photo_iteration)
        photo_deform.deform.eval()
        for pp in photo_deform.deform.parameters():
            pp.requires_grad_(False)

        gs_fea_dim = photo_deform.deform.node_num if photo_dataset.skinning and photo_deform.name == "node" else photo_dataset.hyper_dim
        photo_gaussians = SCGSGaussianModel(photo_dataset.sh_degree, fea_dim=gs_fea_dim, with_motion_mask=photo_dataset.gs_with_motion_mask)
        photo_scene = SCGSScene(photo_dataset, photo_gaussians, load_iteration=args.photo_iteration, shuffle=False)
        for attr in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "feature"]:
            t_ = getattr(photo_gaussians, attr, None)
            if t_ is not None and hasattr(t_, "requires_grad_"):
                t_.requires_grad_(False)
        bg_color = [1, 1, 1] if photo_dataset.white_background else [0, 0, 0]
        photo_bg = torch.tensor(bg_color, dtype=torch.float32, device=dev)
        photo_views = photo_scene.getTrainCameras()
        # readNerfSyntheticInfo appends test cams after the real train cams when
        # cfg_args has eval=False (true for our jumpingjacks checkpoint) -- so
        # getTrainCameras() can be longer than T (200 real + 20 test = 220), but
        # the first T entries are still the real train frames in original order,
        # matching our extraction's frame indices 1:1. Only require >=, not ==.
        assert len(photo_views) >= T, (len(photo_views), T, "SC-GS must have at least as many train cameras as our extraction's T")
        print(f"[photo] enabled: {len(photo_views)} train views from {args.photo_model_path}, "
              f"lambda={args.lambda_photo} views_per_step={args.photo_views_per_step}")

    pbar = trange(start_step, args.iters, desc="hop_autoreg")
    for step in pbar:
        t = min(step / max(args.iters - 1, 1), 1.0)
        lr = math.exp(math.log(args.lr) * (1 - t) + math.log(lr_final) * t)
        for g in opt.param_groups:
            g["lr"] = lr

        frac = min(step / max(curriculum_iters, 1), 1.0)

        if args.curriculum_mode == "window":
            # Always dt=dt_base, one next-frame hop at a time from t=0 -- ramp
            # how much of the trajectory (the rollout horizon L) is covered,
            # short-to-long. Frames beyond L get no gradient signal until the
            # window reaches them.
            L = min(max_len, args.curriculum_start_len + int(frac * (max_len - args.curriculum_start_len)))
            frame_indices = list(range(1, L + 1))
        else:
            # "density": always cover the FULL trajectory (t=1..T-1) from step
            # 0 -- every frame gets gradient signal from the start, like the
            # original frames_per_step scheme -- but the number of EQUALLY
            # SPACED hops used to span it ramps from curriculum_start_n up to
            # max_len (=every single frame, dt=dt_base). dt shrinks over
            # training instead of the horizon growing.
            n = min(max_len, args.curriculum_start_n + int(frac * (max_len - args.curriculum_start_n)))
            if n >= max_len:
                frame_indices = list(range(1, T))
            elif n <= 1:
                frame_indices = [T - 1]
            else:
                frame_indices = sorted(set(1 + round(i * (T - 2) / (n - 1)) for i in range(n)))

        if not args.stateless_pv and not args.spatial_from_current_pos:
            spatial = model.spatial_embed(anchors.e, edge_index, anchors.canonical)
        p, h, prev_t = anchors.canonical, init_h, 0
        v = torch.zeros_like(anchors.canonical)            # stateless_pv only: starts at rest
        pred_pos_list, pred_rot_list, pred_lrot_list = [], [], []
        xpbd_residual_list = []
        for hop_i, cur_t in enumerate(frame_indices, start=1):
            dt_hop = (cur_t - prev_t) * dt_base
            if args.stateless_pv:
                d_p, d_rot, v, lrot, h = model.hop_pv(anchors.e, edge_index, p, v, dt_hop, h=h)
            else:
                if args.spatial_from_current_pos:
                    spatial = model.spatial_embed(anchors.e, edge_index, p)
                d_p, d_rot, h = model.hop(spatial, h, anchors.e, dt_hop)
            p_pre = p + d_p
            if args.xpbd and args.xpbd_mode == "directional":
                p = xpbd_directional_project(p_pre, edge_index, xpbd_rest_len, direction=d_p,
                                              tolerance=args.xpbd_tolerance, iters=args.xpbd_iters)
                xpbd_residual_list.append(p - p_pre)
            elif args.xpbd and args.xpbd_mode == "distance":
                p = xpbd_distance_project(p_pre, edge_index, xpbd_rest_len,
                                           tolerance=args.xpbd_tolerance, iters=args.xpbd_iters)
                xpbd_residual_list.append(p - p_pre)
            else:
                p = p_pre                          # xpbd_mode == "none", or --xpbd not set at all
            pred_pos_list.append(p)
            pred_rot_list.append(d_rot)
            pred_lrot_list.append(lrot if args.stateless_pv else model.local_rotation(h))
            prev_t = cur_t
            if args.tbptt_chunk > 0 and hop_i % args.tbptt_chunk == 0:
                p = p.detach()
                if args.stateless_pv:
                    v = v.detach()
                    if h is not None:
                        h = h.detach()
                else:
                    h = h.detach()

        pred_pos = torch.stack(pred_pos_list, dim=0)   # [len(frame_indices),M,3]
        pred_rot = torch.stack(pred_rot_list, dim=0)
        idx_t = torch.tensor(frame_indices, device=dev)
        tgt_pos = teacher_pos[idx_t]
        tgt_rot = teacher_rot[idx_t]

        loss_pos = torch.nn.functional.mse_loss(pred_pos, tgt_pos)
        loss_rot = quat_mse_loss(pred_rot.reshape(-1, 4), tgt_rot.reshape(-1, 4))
        anchor_loss = loss_pos + args.lambda_rot * loss_rot

        if teacher_local_rot is not None:
            pred_local_rot = torch.stack(pred_lrot_list, dim=0)
            tgt_local_rot = teacher_local_rot[idx_t]
            loss_local_rot = quat_mse_loss(pred_local_rot.reshape(-1, 4), tgt_local_rot.reshape(-1, 4))
            anchor_loss = anchor_loss + args.lambda_local_rot * loss_local_rot
        else:
            loss_local_rot = torch.zeros((), device=dev)

        loss = args.lambda_anchor * anchor_loss

        if args.xpbd and args.xpbd_mode == "none":
            # no projection happened -- penalize the RAW positions' actual
            # edge-length violation directly (hinge: zero inside the
            # tolerance band, quadratic outside it), instead of a residual
            # from a projection that never ran.
            pred_pos_stacked = torch.stack(pred_pos_list, dim=0)         # [n_hops,M,3]
            edge_dist = (pred_pos_stacked[:, xpbd_dst] - pred_pos_stacked[:, xpbd_src]).norm(dim=-1)
            stretch_ratio = edge_dist / xpbd_rest_len.unsqueeze(0)
            violation = torch.relu(stretch_ratio - (1 + args.xpbd_tolerance)) \
                + torch.relu((1 - args.xpbd_tolerance) - stretch_ratio)
            loss_xpbd_reg = (violation ** 2).mean()
            loss = loss + args.lambda_xpbd_reg * loss_xpbd_reg
        elif args.xpbd and xpbd_residual_list:
            xpbd_residual = torch.stack(xpbd_residual_list, dim=0)   # [n_hops,M,3]
            loss_xpbd_reg = (xpbd_residual ** 2).sum(-1).mean()
            loss = loss + args.lambda_xpbd_reg * loss_xpbd_reg
        else:
            loss_xpbd_reg = torch.zeros((), device=dev)

        if photo_enabled:
            n_sample = min(args.photo_views_per_step, len(frame_indices))
            sample_local_idxs = random.sample(range(len(frame_indices)), n_sample)
            photo_loss_sum = torch.zeros((), device=dev)
            for si in sample_local_idxs:
                fi = frame_indices[si]  # dataset frame index, 1..T-1
                pos_i, rot_i, lrot_i = pred_pos_list[si], pred_rot_list[si], pred_lrot_list[si]
                d_xyz_i = pos_i - canonical

                class _PhotoNodeDeform:
                    def __call__(self, t, **kwargs):
                        return {"d_xyz": d_xyz_i, "d_rotation": rot_i,
                                "d_scaling": torch.zeros(M, 3, device=dev), "local_rotation": lrot_i,
                                "hidden": None, "d_opacity": None, "d_color": None}

                photo_deform.deform.node_deform = _PhotoNodeDeform()
                view = photo_views[fi]
                if photo_dataset.load2gpu_on_the_fly:
                    view.load2device()
                time_input = photo_deform.deform.expand_time(view.fid)
                d_values = photo_deform.step(photo_gaussians.get_xyz.detach(), time_input, feature=photo_gaussians.feature,
                                              motion_mask=photo_gaussians.motion_mask, is_training=False)
                results = scgs_render(view, photo_gaussians, photo_pipeline, photo_bg, d_values["d_xyz"], d_values["d_rotation"],
                                       d_values["d_scaling"], d_opacity=d_values["d_opacity"], d_color=d_values["d_color"],
                                       d_rot_as_res=photo_deform.d_rot_as_res)
                image = torch.clamp(results["render"], 0.0, 1.0)
                gt_image = torch.clamp(view.original_image.to(dev), 0.0, 1.0)
                photo_loss_sum = photo_loss_sum + torch.nn.functional.l1_loss(image, gt_image)
            photo_loss = photo_loss_sum / n_sample
            loss = loss + args.lambda_photo * photo_loss
        else:
            photo_loss = torch.zeros((), device=dev)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0:
            pbar.set_postfix({"n": len(frame_indices), "loss": f"{loss.item():.6f}", "pos": f"{loss_pos.item():.6f}",
                               "rot": f"{loss_rot.item():.6f}", "lrot": f"{loss_local_rot.item():.6f}",
                               "photo": f"{photo_loss.item():.6f}", "xpbd_reg": f"{loss_xpbd_reg.item():.6f}",
                               "lr": f"{lr:.2e}"}, refresh=False)

        if step % args.ckpt_every == 0 or step == args.iters - 1:
            ckpt = {"step": step, "model": model.state_dict(), "init_h": init_h.detach().cpu(),
                    "anchors": anchors.state_dict(), "opt": opt.state_dict(),
                    "args": vars(args), "commit": commit}
            torch.save(ckpt, os.path.join(args.out, "ckpt_last.pt"))
            torch.save(ckpt, os.path.join(args.out, f"ckpt_{step:06d}.pt"))
            print(f"\n[step {step}] saved (pos={loss_pos.item():.6f} rot={loss_rot.item():.6f} "
                  f"local_rot={loss_local_rot.item():.6f} photo={photo_loss.item():.6f})", flush=True)

    print("[train] done.")


if __name__ == "__main__":
    main()
