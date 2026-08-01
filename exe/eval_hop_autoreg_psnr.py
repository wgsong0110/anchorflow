#!/usr/bin/env python
"""
PSNR/SSIM eval for a HopDynamics anchor-trajectory-distilled model, rendered
through SC-GS's OWN gaussian_renderer + DeformNetworkNode.forward() -- not a
reimplementation of the local_frame/LBS composition math.

Approach: load the real SC-GS DeformModel/GaussianModel/Scene from the
checkpoint exactly like SC-GS's own render.py does, then monkeypatch
`deform.deform.node_deform` (the raw per-node network call) to return our
HopDynamics model's predictions instead of SC-GS's own node network's output.
Everything downstream of that call -- cal_nn_weight (trained skinning),
the local_frame SVD-composition (Ax formula), d_rot_as_res branching, scale
blending -- is SC-GS's real, unmodified code. This avoids re-deriving those
formulas by hand and guarantees fidelity to how SC-GS actually renders.

Must be run with CWD = the SC-GS repo root (so `from scene import ...` etc.
resolve), e.g.:

  cd /workspace/SC-GS && python /workspace/anchorflow/exe/eval_hop_autoreg_psnr.py \\
      --model_path /workspace/scgs_jump_node \\
      --source_path /workspace/dnerf_jumpingjacks \\
      --traj /workspace/scgs_jump_anchor_traj_v2.pt \\
      --hop_ckpt /workspace/jump_hop_autoreg_v2/ckpt_last.pt
"""
import sys, os, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True, help="SC-GS checkpoint dir (e.g. /workspace/scgs_jump_node)")
ap.add_argument("--source_path", required=True, help="dataset dir (overrides the checkpoint's saved source_path)")
ap.add_argument("--iteration", type=int, default=-1)
ap.add_argument("--traj", required=True, help="*_anchor_traj*.pt used to train the HopDynamics model")
ap.add_argument("--hop_ckpt", required=True, help="HopDynamics ckpt_*.pt from train_hop_autoreg_anchortraj.py")
ap.add_argument("--split", default="test", choices=["train", "test"])
ap.add_argument("--save_video", default=None, help="if set, path to write a render|gt side-by-side mp4")
ap.add_argument("--fixed_view_video", default=None,
                 help="if set, path to write a SINGLE-camera video that sweeps the full trajectory "
                      "(camera pose held fixed, time varies over every frame in --traj) instead of "
                      "the per-view test/train split, which changes camera every frame")
ap.add_argument("--fixed_view_idx", type=int, default=0, help="which train-camera pose to hold fixed")
ap.add_argument("--fixed_view_stride", type=int, default=1,
                 help="subsample the trajectory by this stride before rendering (fewer, bigger hops -- "
                      "e.g. stride=5 over T=200 frames gives ~40 hops, matching typical "
                      "--frames_per_step training chain length, instead of 199 consecutive hops)")
ap.add_argument("--fixed_view_random_n", type=int, default=0,
                 help="if >0, hop through a RANDOM subset of this many frames (matching "
                      "train_hop_autoreg_anchortraj.py's `sorted(random.sample(all_frames, "
                      "frames_per_step))` exactly) instead of the uniform --fixed_view_stride grid. "
                      "Overrides --fixed_view_stride when set.")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fixed_view_consecutive_n", type=int, default=0,
                 help="if >0, hop through N CONSECUTIVE frame indices starting at "
                      "--fixed_view_start (dt=dt_base every hop, smallest possible dt, same "
                      "hop count as --fixed_view_random_n but no large-dt hops mixed in -- "
                      "tests whether small-dt specifically is undertrained, vs. hop-count alone). "
                      "Overrides --fixed_view_random_n and --fixed_view_stride when set.")
ap.add_argument("--fixed_view_start", type=int, default=0)
ap.add_argument("--multiscene_scene", default=None,
                 help="if --hop_ckpt is a train_hop_multiscene.py checkpoint (has a 'scenes' dict "
                      "instead of top-level 'anchors'/'init_h'), which scene's anchors/init_h to use "
                      "alongside the shared model.")
ap.add_argument("--use_teacher", action="store_true",
                 help="render with SC-GS's OWN trained node_deform network (no HopDynamics "
                      "monkeypatch) -- lets --fixed_view_video show the teacher's own "
                      "reconstruction at the same synthetic fixed-camera/swept-time query as a "
                      "reference, since no real GT exists for those (camera, time) combos.")
args = ap.parse_args()

_lib = "/workspace/anchorflow/lib"
sys.path.insert(0, _lib)
sys.path.insert(0, os.getcwd())  # SC-GS repo root (`scene`, `gaussian_renderer`, ...) -- script runs from elsewhere
from anchorflow.anchors import AnchorSet
from anchorflow.ssm_dynamics import (
    HopDynamics, build_graph, hop_rollout, rest_edge_lengths, xpbd_distance_project, xpbd_directional_project,
)

import torch
from scene import Scene, DeformModel
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.image_utils import psnr as psnr_fn
from utils.image_utils import ssim as ssim_fn

sys.argv = [sys.argv[0], "--model_path", args.model_path, "--source_path", args.source_path,
            "--deform_type", "node"]
parser = argparse.ArgumentParser()
model_p = ModelParams(parser, sentinel=True)
pipeline_p = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
cargs = get_combined_args(parser)
cargs.source_path = args.source_path  # explicit override, checkpoint's saved path may not exist here
dataset = model_p.extract(cargs)
pipeline = pipeline_p.extract(cargs)

dev = "cuda"
torch.set_grad_enabled(False)

deform = DeformModel(K=dataset.K, deform_type=dataset.deform_type, is_blender=dataset.is_blender,
                      skinning=dataset.skinning, hyper_dim=dataset.hyper_dim, node_num=dataset.node_num,
                      pred_opacity=dataset.pred_opacity, pred_color=dataset.pred_color, use_hash=dataset.use_hash,
                      hash_time=dataset.hash_time, d_rot_as_res=dataset.d_rot_as_res, local_frame=dataset.local_frame,
                      progressive_brand_time=dataset.progressive_brand_time, max_d_scale=dataset.max_d_scale)
deform.load_weights(dataset.model_path, iteration=args.iteration)
deform.deform.eval()

gs_fea_dim = deform.deform.node_num if dataset.skinning and deform.name == "node" else dataset.hyper_dim
gaussians = GaussianModel(dataset.sh_degree, fea_dim=gs_fea_dim, with_motion_mask=dataset.gs_with_motion_mask)
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device=dev)

print(f"[eval] local_frame={dataset.local_frame} d_rot_as_res={dataset.d_rot_as_res} "
      f"skinning={dataset.skinning} node_num={dataset.node_num}")

# ---- load HopDynamics model + trajectory metadata ----
traj = torch.load(args.traj, map_location=dev)
canonical = traj["canonical_nodes"].to(dev).float()
times_f = traj["times"]
dt_base = times_f[1] - times_f[0]
M = canonical.shape[0]
assert M == deform.deform.nodes.shape[0], (M, deform.deform.nodes.shape)

ckpt = torch.load(args.hop_ckpt, map_location=dev)
hargs = ckpt["args"]
anchors = AnchorSet.from_trajectory(canonical, latent_dim=hargs["latent_dim"], e_dim=hargs["e_dim"],
                                     K=hargs["anchor_k"]).to(dev)
if args.multiscene_scene:
    scene_sd = ckpt["scenes"][args.multiscene_scene]
    anchors.load_state_dict(scene_sd["anchors"])
    init_h = scene_sd["init_h"].to(dev)
else:
    anchors.load_state_dict(ckpt["anchors"])
    init_h = ckpt["init_h"].to(dev)
graph_cfg = {"graph": "knn", "k": hargs["k_graph"]}
edge_index = build_graph(anchors.canonical.detach(), graph_cfg)
model = HopDynamics(hidden=hargs["hidden"], mp_steps=hargs["mp_steps"], ssm_dim=hargs["ssm_dim"],
                     e_dim=hargs["e_dim"], n_time_freqs=hargs["n_time_freqs"]).to(dev)
model.load_state_dict(ckpt["model"])
model.eval()

# match training's XPBD projection at rollout time -- a model trained WITH
# the constraint but evaluated/rendered WITHOUT it is a train/eval mismatch
# (the checkpoint's own saved args say whether training used it, and which
# mode -- older checkpoints predate --xpbd_mode and default to "distance").
xpbd_mode = hargs.get("xpbd_mode", "distance") if hargs.get("xpbd") else None
if xpbd_mode == "none":
    project_fn = None   # trained raw (no projection at all) -- eval must match, not add one
    print("[eval] XPBD mode=none: no projection applied (matches training's raw rollout)")
elif hargs.get("xpbd"):
    xpbd_rest_len = rest_edge_lengths(anchors.canonical.detach(), edge_index)
    if xpbd_mode == "directional":
        project_fn = lambda p, d_p: xpbd_directional_project(
            p, edge_index, xpbd_rest_len, direction=d_p,
            tolerance=hargs["xpbd_tolerance"], iters=hargs["xpbd_iters"])
    else:
        project_fn = lambda p, d_p: xpbd_distance_project(
            p, edge_index, xpbd_rest_len, tolerance=hargs["xpbd_tolerance"], iters=hargs["xpbd_iters"])
    print(f"[eval] XPBD projection ENABLED (mode={xpbd_mode} tolerance={hargs['xpbd_tolerance']} "
          f"iters={hargs['xpbd_iters']}) -- matches training")
else:
    project_fn = None

print(f"[eval] hop ckpt step={ckpt['step']} commit={ckpt.get('commit')} scene={args.multiscene_scene}")

# ---- one rollout covering every fid we'll be asked to render ----
views = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
fids = sorted(set(float(v.fid) for v in views) | {0.0})
labels = [f / dt_base for f in fids]
label_of_fid = {f: (f / dt_base) for f in fids}

p_by_t, rot_by_t, h_by_t = hop_rollout(model, anchors.canonical, anchors.e, edge_index,
                                        times=labels, dt_base=dt_base, grad=False, h0=init_h,
                                        project_fn=project_fn)


class HopNodeDeform:
    """Drop-in replacement for DeformNetworkNode.node_deform: same return
    dict shape, but values come from our HopDynamics rollout instead of
    SC-GS's own per-node network."""

    def __call__(self, t, **kwargs):
        fid_val = float(t.reshape(-1)[0].item())
        label = label_of_fid[round(fid_val, 6)] if round(fid_val, 6) in label_of_fid else fid_val / dt_base
        pos = p_by_t[label]
        rot = rot_by_t[label]
        lrot = model.local_rotation(h_by_t[label])
        return {
            "d_xyz": pos - canonical,
            "d_rotation": rot,
            "d_scaling": torch.zeros(M, 3, device=dev),
            "local_rotation": lrot,
            "hidden": h_by_t[label],
            "d_opacity": None,
            "d_color": None,
        }


if args.use_teacher:
    print("[eval] --use_teacher set: rendering with SC-GS's own node network, HopDynamics unused")
else:
    deform.deform.node_deform = HopNodeDeform()

# ---- render + measure ----
views_sorted = sorted(views, key=lambda v: float(v.fid))
psnr_list, ssim_list = [], []
frames = []
for view in views_sorted:
    if dataset.load2gpu_on_the_fly:
        view.load2device()
    fid = view.fid
    time_input = deform.deform.expand_time(fid)
    d_values = deform.step(gaussians.get_xyz.detach(), time_input, feature=gaussians.feature,
                            motion_mask=gaussians.motion_mask, is_training=False)
    d_xyz, d_rotation, d_scaling = d_values["d_xyz"], d_values["d_rotation"], d_values["d_scaling"]
    d_opacity, d_color = d_values["d_opacity"], d_values["d_color"]
    results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling,
                      d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)
    image = torch.clamp(results["render"], 0.0, 1.0)
    gt_image = torch.clamp(view.original_image.to(dev), 0.0, 1.0)
    psnr_list.append(psnr_fn(image[None], gt_image[None]).mean())
    ssim_list.append(ssim_fn(image[None], gt_image[None], data_range=1.0).mean())
    if args.save_video:
        side_by_side = torch.cat([image, gt_image[:3]], dim=2)  # concat along width
        frames.append((side_by_side.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))

psnr_mean = torch.stack(psnr_list).mean().item()
ssim_mean = torch.stack(ssim_list).mean().item()
print(f"[eval] split={args.split} n_views={len(views)} PSNR={psnr_mean:.4f} SSIM={ssim_mean:.4f}")

if args.save_video:
    import imageio
    os.makedirs(os.path.dirname(args.save_video) or ".", exist_ok=True)
    imageio.mimwrite(args.save_video, frames, fps=10, quality=8)
    print(f"[eval] wrote {len(frames)} frames (render|gt) -> {args.save_video}")

if args.fixed_view_video:
    import imageio
    # Full trajectory rollout at every original extraction frame index, all
    # under ONE fixed camera pose -- unlike the per-split loop above (which
    # renders each view's own camera, so consecutive frames jump between
    # different viewpoints AND times). This is the equivalent of SC-GS's own
    # render.py interpolate_time(): fix the camera, sweep only time.
    T = len(times_f)
    if args.fixed_view_consecutive_n > 0:
        start, n = args.fixed_view_start, args.fixed_view_consecutive_n
        end = min(start + n, T)
        # hop consecutively from t=0 through `end-1` so the hidden state is built up
        # correctly if start > 0; render only the requested [start, end) window.
        frame_indices = list(range(1, end))
        render_indices = list(range(start, end))
    elif args.fixed_view_random_n > 0:
        import random
        random.seed(args.seed)
        all_frames = list(range(1, T))
        frame_indices = sorted(random.sample(all_frames, min(args.fixed_view_random_n, len(all_frames))))
        render_indices = [0] + frame_indices
    else:
        render_indices = list(range(0, T, args.fixed_view_stride))
        if render_indices[-1] != T - 1:
            render_indices.append(T - 1)
        frame_indices = [i for i in render_indices if i != 0]
    if not args.use_teacher:
        p_full, rot_full, h_full = hop_rollout(model, anchors.canonical, anchors.e, edge_index,
                                                times=frame_indices, dt_base=dt_base, grad=False, h0=init_h,
                                                project_fn=project_fn)
        p_full[0], rot_full[0], h_full[0] = anchors.canonical, torch.zeros(M, 4, device=dev), init_h
    print(f"[eval] fixed_view_video: {len(render_indices)} rendered frames via {len(frame_indices)} hops "
          f"(stride={args.fixed_view_stride}, random_n={args.fixed_view_random_n}, "
          f"consecutive_n={args.fixed_view_consecutive_n}, start={args.fixed_view_start}, "
          f"use_teacher={args.use_teacher})")

    fixed_view = scene.getTrainCameras()[args.fixed_view_idx]
    if dataset.load2gpu_on_the_fly:
        fixed_view.load2device()

    frames2 = []
    for i in render_indices:
        if not args.use_teacher:
            pos, rot, h = p_full[i], rot_full[i], h_full[i]
            lrot = model.local_rotation(h)

            class _Fixed:
                def __call__(self, t, **kwargs):
                    return {"d_xyz": pos - canonical, "d_rotation": rot,
                            "d_scaling": torch.zeros(M, 3, device=dev), "local_rotation": lrot,
                            "hidden": h, "d_opacity": None, "d_color": None}

            deform.deform.node_deform = _Fixed()
        # else: deform.deform.node_deform is untouched -- SC-GS's own trained network
        fid = torch.tensor([times_f[i]], device=dev)
        time_input = deform.deform.expand_time(fid)
        d_values = deform.step(gaussians.get_xyz.detach(), time_input, feature=gaussians.feature,
                                motion_mask=gaussians.motion_mask, is_training=False)
        results = render(fixed_view, gaussians, pipeline, background, d_values["d_xyz"], d_values["d_rotation"],
                          d_values["d_scaling"], d_opacity=d_values["d_opacity"], d_color=d_values["d_color"],
                          d_rot_as_res=deform.d_rot_as_res)
        image = torch.clamp(results["render"], 0.0, 1.0)
        frames2.append((image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))

    os.makedirs(os.path.dirname(args.fixed_view_video) or ".", exist_ok=True)
    fps = max(2, round(20 * len(render_indices) / T))
    imageio.mimwrite(args.fixed_view_video, frames2, fps=fps, quality=8)
    print(f"[eval] wrote {len(frames2)} frames (fixed camera idx={args.fixed_view_idx}, fps={fps}) -> {args.fixed_view_video}")
