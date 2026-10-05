"""`pytorch3d` 의 **순수 파이썬 부분만** 토치로 대신 깐다 (CUDA 확장 안 짓는다).

Spring-Gaus 가 쓰는 것은 회전 변환 아홉 개(`pytorch3d.transforms`)와
`pytorch3d.ops.knn_points` 뿐이다. 둘 다 토치로 똑같이 쓸 수 있다.
빌드가 금지된 환경이라 이 길을 택한다 (simple_knn 과 같은 처리).

  python exe/make_pytorch3d_shim.py
"""
from __future__ import annotations

import os
import site
import sys

TRANSFORMS = '''"""pytorch3d.transforms 대체 (회전 변환만). [anchorflow]"""
import torch
import torch.nn.functional as F


def quaternion_to_matrix(q):
    r, i, j, k = torch.unbind(q, -1)
    s = 2.0 / (q * q).sum(-1)
    o = torch.stack((
        1 - s * (j * j + k * k), s * (i * j - k * r), s * (i * k + j * r),
        s * (i * j + k * r), 1 - s * (i * i + k * k), s * (j * k - i * r),
        s * (i * k - j * r), s * (j * k + i * r), 1 - s * (i * i + j * j),
    ), -1)
    return o.reshape(q.shape[:-1] + (3, 3))


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    pos = x > 0
    ret[pos] = torch.sqrt(x[pos])
    return ret


def matrix_to_quaternion(m):
    bd = m.shape[:-2]
    m9 = m.reshape(bd + (9,))
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(m9, -1)
    q_abs = _sqrt_positive_part(torch.stack([
        1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1))
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], -1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], -1),
    ], dim=-2)
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :]
    return out.reshape(bd + (4,))


def axis_angle_to_quaternion(aa):
    angles = torch.norm(aa, p=2, dim=-1, keepdim=True)
    half = angles * 0.5
    eps = 1e-6
    small = angles.abs() < eps
    sin_half_over_angle = torch.empty_like(angles)
    sin_half_over_angle[~small] = (torch.sin(half[~small]) / angles[~small])
    sin_half_over_angle[small] = 0.5 - (angles[small] * angles[small]) / 48
    return torch.cat([torch.cos(half), aa * sin_half_over_angle], dim=-1)


def quaternion_to_axis_angle(q):
    norms = torch.norm(q[..., 1:], p=2, dim=-1, keepdim=True)
    half = torch.atan2(norms, q[..., :1])
    eps = 1e-6
    small = half.abs() < eps
    sin_half_over_angle = torch.empty_like(half)
    sin_half_over_angle[~small] = (torch.sin(half[~small]) / (2 * half[~small]))
    sin_half_over_angle[small] = 0.5 - (half[small] * half[small]) / 48
    return q[..., 1:] / sin_half_over_angle


def axis_angle_to_matrix(aa):
    return quaternion_to_matrix(axis_angle_to_quaternion(aa))


def matrix_to_axis_angle(m):
    return quaternion_to_axis_angle(matrix_to_quaternion(m))


def _axis_angle_rotation(axis, angle):
    cos, sin = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(angle), torch.zeros_like(angle)
    if axis == "X":
        mf = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        mf = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        mf = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("축은 X/Y/Z 여야 한다")
    return torch.stack(mf, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention):
    ms = [_axis_angle_rotation(c, e)
          for c, e in zip(convention, torch.unbind(euler_angles, -1))]
    return torch.matmul(torch.matmul(ms[0], ms[1]), ms[2])


def _angle_from_tan(axis, other_axis, data, horizontal, tait_bryan):
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter):
    return {"X": 0, "Y": 1, "Z": 2}[letter]


def matrix_to_euler_angles(matrix, convention):
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central = torch.asin(matrix[..., i0, i2]
                             * (-1.0 if i0 - i2 in [-1, 2] else 1.0))
    else:
        central = torch.acos(matrix[..., i0, i0])
    o = (_angle_from_tan(convention[0], convention[1], matrix[..., i2],
                         False, tait_bryan),
         central,
         _angle_from_tan(convention[2], convention[1], matrix[..., i0, :],
                         True, tait_bryan))
    return torch.stack(o, -1)


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix):
    return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))
'''

OPS = '''"""pytorch3d.ops 대체 (knn_points 만). [anchorflow]"""
from collections import namedtuple

import torch

_KNN = namedtuple("KNN", "dists idx knn")


def knn_points(p1, p2, lengths1=None, lengths2=None, K=1, version=-1,
               return_nn=False, return_sorted=True, norm=2):
    """[B,N,D] 와 [B,M,D] 의 K 최근접. dists 는 **제곱거리** (원본과 같다)."""
    d = torch.cdist(p1, p2, p=float(norm))
    dists, idx = torch.topk(d, min(K, p2.shape[1]), dim=-1, largest=False,
                            sorted=return_sorted)
    if norm == 2:
        dists = dists ** 2
    knn = None
    if return_nn:
        B, N, Kk = idx.shape
        knn = torch.gather(
            p2.unsqueeze(1).expand(-1, N, -1, -1), 2,
            idx.unsqueeze(-1).expand(-1, -1, -1, p2.shape[-1]))
    return _KNN(dists=dists, idx=idx, knn=knn)
'''

sp = [p for p in sys.path if p.endswith("site-packages")]
root = os.path.join(sp[0] if sp else site.getsitepackages()[0], "pytorch3d")
if os.path.exists(os.path.join(root, "_C.so")):
    print(f"진짜 pytorch3d 가 있다: {root} -- 건드리지 않는다")
    raise SystemExit(0)
os.makedirs(root, exist_ok=True)
open(os.path.join(root, "__init__.py"), "w").write(
    '"""[anchorflow] 순수 파이썬 대체본 (회전 변환 + knn)."""\n'
    "__version__ = '0.0.0+anchorflow'\n")
open(os.path.join(root, "transforms.py"), "w").write(TRANSFORMS)
open(os.path.join(root, "ops.py"), "w").write(OPS)
print(f"설치: {root} (transforms, ops)")
