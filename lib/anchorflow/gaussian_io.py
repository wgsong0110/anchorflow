"""Load / save a 3D Gaussian Splatting scene in the standard INRIA `.ply` format.

Compatible with vanilla 3DGS, SC-GS, and TRELLIS exports. TRELLIS writes SH
**degree 0** (only `f_dc_0..2`, no `f_rest_*`) and stores everything
*pre-activation* (opacity pre-sigmoid, scale in log-space, raw quaternion), plus
an optional Y-up→Z-up axis transform — so this loader must tolerate a missing
`f_rest_*` block and returns raw params with activation helpers applied at use.

Returns a plain dict of torch tensors:
    xyz        [N,3]      world positions
    f_dc       [N,3]      SH DC term (raw)
    f_rest     [N,R,3]    higher SH (R = (deg+1)^2 - 1; R=0 for TRELLIS)
    opacity    [N,1]      raw (apply sigmoid)
    scaling    [N,3]      raw log-scale (apply exp)
    rotation   [N,4]      raw quaternion (apply normalize)
"""

from __future__ import annotations

import numpy as np
import torch


def load_ply(path, device="cpu"):
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"]
    names = set(v.data.dtype.names)

    def col(n):
        return torch.tensor(np.asarray(v[n]), dtype=torch.float32)

    xyz = torch.stack([col("x"), col("y"), col("z")], dim=-1)
    f_dc = torch.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], dim=-1)

    rest_names = sorted([n for n in names if n.startswith("f_rest_")],
                        key=lambda s: int(s.split("_")[-1]))
    if rest_names:
        rest = torch.stack([col(n) for n in rest_names], dim=-1)     # [N, 3R]
        f_rest = rest.reshape(rest.shape[0], -1, 3)                  # [N, R, 3]
    else:
        f_rest = torch.zeros(xyz.shape[0], 0, 3)

    opacity = col("opacity").unsqueeze(-1)
    scaling = torch.stack([col(f"scale_{i}") for i in range(3)], dim=-1)
    rot_names = sorted([n for n in names if n.startswith("rot_")],
                       key=lambda s: int(s.split("_")[-1]))
    rotation = torch.stack([col(n) for n in rot_names], dim=-1)      # [N,4]

    out = dict(xyz=xyz, f_dc=f_dc, f_rest=f_rest, opacity=opacity,
               scaling=scaling, rotation=rotation)
    return {k: t.to(device) for k, t in out.items()}


def save_ply(path, g):
    """Inverse of load_ply. `g` is the dict returned by load_ply."""
    from plyfile import PlyData, PlyElement
    xyz = g["xyz"].detach().cpu().numpy()
    f_dc = g["f_dc"].detach().cpu().numpy()
    f_rest = g["f_rest"].detach().cpu().numpy().reshape(xyz.shape[0], -1)
    opacity = g["opacity"].detach().cpu().numpy()
    scaling = g["scaling"].detach().cpu().numpy()
    rotation = g["rotation"].detach().cpu().numpy()

    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    fields += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    fields += ["opacity", "scale_0", "scale_1", "scale_2",
               "rot_0", "rot_1", "rot_2", "rot_3"]
    dtype = [(f, "f4") for f in fields]
    normals = np.zeros_like(xyz)
    data = np.concatenate([xyz, normals, f_dc, f_rest, opacity, scaling, rotation],
                          axis=1)
    elem = np.empty(xyz.shape[0], dtype=dtype)
    elem[:] = list(map(tuple, data))
    PlyData([PlyElement.describe(elem, "vertex")]).write(path)


# --- activations (apply at render time) ----------------------------------- #
def activate(g):
    return dict(
        xyz=g["xyz"],
        f_dc=g["f_dc"],
        f_rest=g["f_rest"],
        opacity=torch.sigmoid(g["opacity"]),
        scaling=torch.exp(g["scaling"]),
        rotation=torch.nn.functional.normalize(g["rotation"], dim=-1),
    )


AXIS_ZUP_FROM_YUP = torch.tensor([[1., 0, 0], [0, 0, -1], [0, 1, 0]])


def apply_axis_transform(g, R):
    """Rotate positions (and quaternions if needed) by a 3x3 R. TRELLIS uses
    AXIS_ZUP_FROM_YUP; pass R=None or identity to keep the native frame."""
    if R is None:
        return g
    g = dict(g)
    g["xyz"] = g["xyz"] @ R.to(g["xyz"]).T
    return g
