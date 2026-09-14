"""GausSim 공개 저장소에 필요한 수정들을 한 번에 적용한다.

`setup_instance.sh` 는 vast 인스턴스용으로 패키지 설치까지 같이 하는데, 클러스터에는
`gssim` env 가 이미 있어 소스 수정만 필요하다. 그 부분만 떼어낸 것이고 멱등하다.

메우는 것 (전부 공개 저장소의 빠진 조각이거나 버전 차이):
  1. `torch._six` -- torch 2.x 에서 삭제
  2. `PointCloudViewer` -- 저장소에 없는 클래스인데 임포트만 되어 있다
  3. blender 카메라 경로가 `Camera(static_img_path=...)` 를 안 넘긴다
  4. 래스터라이저 반환 개수 (이미지/환경에 따라 2 개 또는 4 개)
  5. `_edge_theta` 의 0/0 -- 노드가 자기 앵커와 같은 자리면 방향이 정의되지 않는다.
     방어가 없으면 그 간선이 NaN 이 되고 합쳐지는 노드의 임베딩이 통째로 오염된다
  6. `denom` / `spatial_lr_scale` -- 3DGS 의 training_setup() 에서만 만들어진다
"""
from __future__ import annotations

import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="GausSim_ICCV2025 경로")
a = ap.parse_args()
R = a.root


def edit(rel, old, new, tag):
    p = os.path.join(R, rel)
    if not os.path.exists(p):
        print("  없음:", rel)
        return
    s = open(p).read()
    if new.strip().split("\n")[0] in s or old not in s:
        print("  이미 적용:", tag)
        return
    open(p, "w").write(s.replace(old, new, 1))
    print("  적용:", tag)


print("[1] torch._six")
for p in glob.glob(os.path.join(R, "**", "*.py"), recursive=True):
    s = open(p).read()
    if "from torch._six import" in s:
        open(p, "w").write(s.replace(
            "from torch._six import inf",
            "from torch import inf  # torch 2.x 에서 torch._six 가 사라졌다"))
        print("  ", os.path.relpath(p, R))

print("[2] PointCloudViewer")
edit("mmgs/models/simulators/gs_simulator_hierarchy.py",
     "from mmgs.utils import PointCloudViewer",
     "# from mmgs.utils import PointCloudViewer  # 저장소에 없는 클래스, 쓰이지도 않는다",
     "죽은 임포트 제거")

print("[3] static_img_path")
edit("mmgs/datasets/multiview_video_dataset.py",
     """                    FoVy=FovY,
                    FoVx=FovX,
                    img_path=img_path,
                    img_hw=img_hw,""",
     """                    FoVy=FovY,
                    FoVx=FovX,
                    img_path=img_path,
                    # blender 경로는 이 인자를 안 넘긴다 -- colmap 경로만 시험된 흔적
                    static_img_path=img_path,
                    img_hw=img_hw,""",
     "blender 카메라")

print("[4] 래스터라이저 반환 개수")
for p in glob.glob(os.path.join(R, "mmgs/models/utils/*.py")):
    s = open(p).read()
    if "rendered_image, radii = rasterizer(" not in s:
        continue
    lines = s.replace("rendered_image, radii = rasterizer(",
                      "_ras_out = rasterizer(").split("\n")
    out, i = [], 0
    while i < len(lines):
        out.append(lines[i])
        if "_ras_out = rasterizer(" in lines[i]:
            ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            depth = lines[i].count("(") - lines[i].count(")")
            while depth > 0:
                i += 1
                out.append(lines[i])
                depth += lines[i].count("(") - lines[i].count(")")
            out.append(ind + "rendered_image, radii = _ras_out[0], _ras_out[1]")
        i += 1
    open(p, "w").write("\n".join(out))
    print("  적용:", os.path.relpath(p, R))

print("[5] _edge_theta 0/0")
edit("mmgs/models/backbones/meshgraphnet_hie.py",
     """        normed_recv = recv_vec / torch.linalg.norm(recv_vec, dim=-1, keepdim=True)
        normed_send = send_vec / torch.linalg.norm(send_vec, dim=-1, keepdim=True)""",
     """        # 노드가 자기 앵커와 같은 자리면 이 벡터가 0 이고 0/0 -> NaN 이 된다.
        # 그 간선이 합쳐지는 노드의 임베딩이 통째로 오염되므로, 방향이 정의되지
        # 않을 때는 0 벡터로 둔다 (cos=sin=0).
        _eps = 1e-8
        _rn = torch.linalg.norm(recv_vec, dim=-1, keepdim=True)
        _sn = torch.linalg.norm(send_vec, dim=-1, keepdim=True)
        normed_recv = torch.where(_rn > _eps, recv_vec / _rn.clamp(min=_eps),
                                  torch.zeros_like(recv_vec))
        normed_send = torch.where(_sn > _eps, send_vec / _sn.clamp(min=_eps),
                                  torch.zeros_like(send_vec))""",
     "퇴화 방향 방어")

print("[6] denom / training_setup")
edit("mmgs/utils/physdreamer_utils.py",
     """def apply_mask_gaussian(gaussian, mask):
    new_xyz = gaussian._xyz[mask]""",
     """def apply_mask_gaussian(gaussian, mask):
    # denom / xyz_gradient_accum / spatial_lr_scale 은 3DGS 의 training_setup()
    # 에서 한꺼번에 만들어진다. 시뮬레이션용으로 ply 만 읽으면 하나도 없으므로
    # 최소 인자로 한 번 불러 채운다. densify 통계라 추론에는 쓰이지 않는다.
    if getattr(gaussian, "denom", None) is None:
        if getattr(gaussian, "spatial_lr_scale", None) is None:
            gaussian.spatial_lr_scale = 1.0
        from argparse import Namespace
        _op = Namespace(position_lr_init=0.0, position_lr_final=0.0,
                        position_lr_delay_mult=1.0, position_lr_max_steps=1,
                        feature_lr=0.0, opacity_lr=0.0, scaling_lr=0.0,
                        rotation_lr=0.0, percent_dense=0.01)
        gaussian.training_setup(_op)
    new_xyz = gaussian._xyz[mask]""",
     "densify 통계 채우기")

print("완료")
