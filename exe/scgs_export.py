#!/usr/bin/env python
"""Export a trained SC-GS reconstruction into anchorflow's portable format.

Runs inside the SC-GS image (PYTHONPATH=/opt/SC-GS). SC-GS (yihua7/SC-GS,
"Sparse-Controlled Gaussian Splatting", CVPR'24) represents a dynamic scene as a
canonical 3DGS + a set of sparse *control nodes* (`ControlNodeWarp.nodes`,
[M, 3+hyper]) whose per-time motion is produced by a deformation MLP
(`utils.time_utils.DeformNetwork`: (node_xyz, t) -> d_xyz + d_rotation(quat) +
d_scaling). Dense Gaussians are bound to the K nearest control nodes with an RBF
kernel weight (LBS skinning, `ControlNodeWarp.cal_nn_weight`). The control nodes
ARE anchorflow's anchors and the kNN/RBF weights ARE its LBS binding, so this
script drops SC-GS straight into anchorflow.

A trained SC-GS run is saved in TWO pieces under --model_path (see
scene/__init__.py:Scene.save + scene/deform_model.py:DeformModel.save_weights):
    point_cloud/iteration_<it>/point_cloud.ply   canonical dense 3DGS (+ fea_* cols)
    deform/iteration_<it>/deform.pth             ControlNodeWarp state_dict
                                                 (nodes, _node_radius, _node_weight,
                                                  network.*, gs_* node-gaussians)
plus a `cfg_args` file (a Namespace repr) with the exact model flags.

Writes into --out:
    node_traj.npy   [T, M, 3]   control-node (anchor) positions per timestep
    node_rot.npy    [T, M, 4]   control-node orientation quaternions (wxyz) per t
    canonical.ply               canonical 3DGS in INRIA layout (matches mosca_export)
    lbs_weight.npz              per-Gaussian -> node skinning: nn_idx[N,K], nn_weight[N,K]
Prints shapes + SCGS_EXPORT_OK.

Time in SC-GS is normalized fid in [0, 1] (fid = frame_index / (num_frames-1);
see scene/dataset_readers.py). We query the MLP at t = linspace(0, 1, T) which is
exactly the set of training timestamps -> anchorflow uses dt = 1/(T-1).

    python exe/scgs_export.py --model_path outputs/jumpingjacks_node \
        --num_frames 150 --out /workspace/scgs_out   [--iteration -1]
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace  # noqa: F401  (used by eval of cfg_args)

import numpy as np
import torch


# INRIA 3DGS .ply field order (matches exe/mosca_export.py canonical.ply layout);
# everything SC-GS's gaussian_model.save_ply writes EXCEPT the trailing fea_* cols.
def _read_scgs_ply(path):
    from plyfile import PlyData

    ply = PlyData.read(path)
    v = ply["vertex"]
    props = [p.name for p in v.properties]
    inria = [p for p in props if not p.startswith("fea")]     # drop hyper/motion feats
    fea = [p for p in props if p.startswith("fea")]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    feature = (np.stack([np.asarray(v[f]) for f in fea], axis=1)
               if fea else np.zeros((xyz.shape[0], 0), np.float32))
    inria_data = np.stack([np.asarray(v[p]) for p in inria], axis=1).astype(np.float32)
    return xyz.astype(np.float32), feature.astype(np.float32), inria, inria_data


def _write_inria_ply(path, field_names, data):
    from plyfile import PlyData, PlyElement

    elem = np.empty(data.shape[0], dtype=[(f, "f4") for f in field_names])
    elem[:] = list(map(tuple, data))
    PlyData([PlyElement.describe(elem, "vertex")]).write(path)


def _load_cfg(model_path):
    """Recover the exact model flags SC-GS trained with (train_gui.py writes a
    Namespace repr to <model_path>/cfg_args; see prepare_output_and_logger)."""
    p = os.path.join(model_path, "cfg_args")
    if os.path.exists(p):
        with open(p) as f:
            return eval(f.read())            # noqa: S307  (trusted, our own run)
    return Namespace()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="SC-GS output dir (…_node)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_frames", type=int, required=True,
                    help="T: number of video timesteps (= distinct fids)")
    ap.add_argument("--iteration", type=int, default=-1,
                    help="checkpoint iteration; -1 = latest")
    ap.add_argument("--t_chunk", type=int, default=16, help="time chunk for MLP query")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    assert torch.cuda.is_available(), "scgs_export needs a CUDA device (pytorch3d/simple-knn)"

    from scene.deform_model import DeformModel
    from utils.system_utils import searchForMaxIteration

    cfg = _load_cfg(args.model_path)
    g = lambda k, d: getattr(cfg, k, d)  # noqa: E731

    # d_rot_as_res is the *effective* residual-rotation flag SC-GS uses at runtime.
    d_rot_as_res = bool(g("d_rot_as_res", True)) and not bool(g("d_rot_as_rotmat", False))
    is_scene_static = bool(g("is_scene_static", False))

    # ---- rebuild + load the control-node deformation model -----------------
    deform = DeformModel(
        deform_type=g("deform_type", "node"),
        is_blender=bool(g("is_blender", False)),
        d_rot_as_res=d_rot_as_res,
        K=int(g("K", 3)),
        hyper_dim=int(g("hyper_dim", 8)),
        node_num=int(g("node_num", 512)),
        skinning=bool(g("skinning", False)),
        pred_opacity=bool(g("pred_opacity", False)),
        pred_color=bool(g("pred_color", False)),
        use_hash=bool(g("use_hash", False)),
        hash_time=bool(g("hash_time", False)),
        local_frame=bool(g("local_frame", False)),
        progressive_brand_time=bool(g("progressive_brand_time", False)),
        max_d_scale=float(g("max_d_scale", -1.0)),
        is_scene_static=is_scene_static,
        with_arap_loss=False,
    )
    assert deform.name == "node", f"expected deform_type=node, got {deform.name}"
    ok = deform.load_weights(args.model_path, iteration=args.iteration)
    assert ok, f"could not load deform.pth under {args.model_path}/deform"
    cn = deform.deform                       # ControlNodeWarp
    cn.eval()
    if not cn.skinning:
        cn.hyper_dim = cn.nodes.shape[1] - 3  # realign after any densification
    M = cn.nodes.shape[0]
    print(f"[scgs_export] loaded {M} control nodes, hyper_dim={cn.hyper_dim}, "
          f"K={cn.K}, d_rot_as_res={cn.d_rot_as_res}, skinning={cn.skinning}")

    # ---- canonical dense Gaussians (INRIA .ply) ----------------------------
    it_pc = args.iteration
    if it_pc == -1:
        it_pc = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{it_pc}",
                            "point_cloud.ply")
    xyz_np, fea_np, inria_fields, inria_data = _read_scgs_ply(ply_path)
    _write_inria_ply(os.path.join(args.out, "canonical.ply"), inria_fields, inria_data)
    N = xyz_np.shape[0]
    xyz = torch.from_numpy(xyz_np).cuda()
    feature = torch.from_numpy(fea_np).cuda() if fea_np.shape[1] else None

    # ---- control-node trajectory + rotation (query MLP at each timestep) ----
    T = args.num_frames
    t_lin = torch.linspace(0, 1, T).cuda()
    rot_bias = torch.tensor([1.0, 0, 0, 0], dtype=torch.float32, device="cuda")
    traj_chunks, rot_chunks = [], []
    with torch.no_grad():
        s = 0
        while s < T:
            e = min(s + args.t_chunk, T)
            t_samp = t_lin[None, s:e, None].expand(M, e - s, 1)          # [M, c, 1]
            nd = cn.node_deform(t=t_samp)                               # detach_node=True
            traj = cn.nodes[:, None, :3].detach() + nd["d_xyz"]         # [M, c, 3]
            traj_chunks.append(traj.permute(1, 0, 2).contiguous())     # [c, M, 3]
            drot = nd.get("d_rotation", None)
            if drot is not None:
                q = torch.nn.functional.normalize(rot_bias + drot, dim=-1)  # [M, c, 4]
                rot_chunks.append(q.permute(1, 0, 2).contiguous())     # [c, M, 4]
            s = e
    node_traj = torch.cat(traj_chunks, dim=0).cpu().numpy()             # [T, M, 3]
    np.save(os.path.join(args.out, "node_traj.npy"), node_traj)

    have_rot = len(rot_chunks) > 0 and not is_scene_static
    if have_rot:
        node_rot = torch.cat(rot_chunks, dim=0).cpu().numpy()          # [T, M, 4] wxyz
        np.save(os.path.join(args.out, "node_rot.npy"), node_rot)

    # ---- LBS skinning: Gaussian -> control-node weights (reuse SC-GS binding) --
    with torch.no_grad():
        nn_weight, _, nn_idx = cn.cal_nn_weight(x=xyz, feature=feature)
    nn_weight_np = nn_weight.detach().cpu().numpy()
    if cn.skinning:
        # skinning: dense softmax over ALL nodes; nn_idx is arange(M) (shared).
        nn_idx_np = nn_idx.detach().cpu().numpy().astype(np.int64)      # [M]
        K = int(nn_weight_np.shape[-1])
    else:
        nn_idx_np = nn_idx.detach().cpu().numpy().astype(np.int64)      # [N, K]
        K = int(cn.K)
    np.savez(os.path.join(args.out, "lbs_weight.npz"),
             nn_idx=nn_idx_np, nn_weight=nn_weight_np,
             K=K, node_num=M, skinning=bool(cn.skinning))

    # ---- report ------------------------------------------------------------
    print(f"[scgs_export] canonical.ply  N={N}  fields={len(inria_fields)}")
    print(f"[scgs_export] node_traj.npy  {node_traj.shape}  (T,M,3)")
    if have_rot:
        print(f"[scgs_export] node_rot.npy   {node_rot.shape}  (T,M,4 wxyz)")
    else:
        print("[scgs_export] node_rot.npy   (skipped: static scene)")
    print(f"[scgs_export] lbs_weight.npz  nn_idx={nn_idx_np.shape} "
          f"nn_weight={nn_weight_np.shape} K={K} skinning={bool(cn.skinning)}")
    print("SCGS_EXPORT_OK")


if __name__ == "__main__":
    main()
