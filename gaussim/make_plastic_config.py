"""GausSim 의 pudding config 에서 우리 plastic 씬용 config 를 만든다.

pudding 은 콜맵으로 찍은 실사 씬이고 우리 것은 blender 형식 transforms 를 쓴다.
바꾸는 것은 씬 이름, 카메라 파일 형식, 프레임 간격 셋뿐이고 나머지 하이퍼파라미터는
그쪽 기본값을 그대로 둔다 -- 베이스라인이므로 임의로 손대지 않는다.
"""
from __future__ import annotations

import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="configs/gssim/iccv/pudding.py")
ap.add_argument("--dst", default="configs/gssim/iccv/plastic.py")
ap.add_argument("--scene", default="plastic")
ap.add_argument("--frame_dt", type=float, default=0.01,
                help="한 프레임의 물리 시간. plane_dp.json 의 frame_dt 와 맞춘다")
ap.add_argument("--resolution", default=None,
                help="렌더 해상도 'H,W'. 데이터셋 프레임 크기와 같아야 손실에서 "
                     "shape 이 안 맞는다")
ap.add_argument("--radius", default=None,
                help="계층별 상호작용 반경, 쉼표로 구분. 주지 않으면 그쪽 값 그대로. "
                     "pudding 기준 값(0.03/0.4/5.0)은 우리 씬 스케일에서 너무 작아 "
                     "노드의 1/3 이 간선을 하나도 못 얻고, 그 노드의 임베딩이 NaN 이 된다")
a = ap.parse_args()

s = open(a.src).read()
s = re.sub(r"pudding", a.scene, s)

# real_dt: pudding 은 1/50 이었다. 우리 씬의 frame_dt 로 바꾼다.
s = re.sub(r"%s=1/\d+" % a.scene, "%s=%r" % (a.scene, a.frame_dt), s, count=1)

# 데이터 로더가 blender 형식을 타게 한다. 기본 config 는 colmap 씬용이라
# cam_transform_fn=None 이고, 그러면 sparse/0 을 찾다가 죽는다.
s = s.replace("            data_dir=data_dir,",
              "            data_dir=data_dir,\n"
              "            cam_transform_fn=cam_transform_fn,")
s = ("# 이 데이터는 blender 형식 transforms_train.json 을 쓴다 (colmap sparse 없음)\n"
     "cam_transform_fn = 'transforms_train.json'\n") + s

if a.resolution:
    # resolution 은 _base_ 쪽 env_cfg 에 있어서 이 파일만 고쳐서는 안 바뀐다.
    # 데이터셋 dict 마다 명시적으로 넘겨 base 를 덮는다.
    h, w = [int(v) for v in a.resolution.split(",")]
    s = s.replace("            data_dir=data_dir,",
                  "            data_dir=data_dir,\n"
                  "            resolution=resolution,")
    s = ("# 데이터셋 프레임 크기. base 의 [960, 540] 을 덮는다\n"
         "resolution = [%d, %d]\n" % (h, w)) + s
    print("resolution -> [%d, %d]" % (h, w))

if a.radius:
    # radius 는 그쪽 config 에서도 씬마다 다르게 주는 값이다. 우리 씬의 레벨별
    # 16번째 이웃 거리 p99 보다 크게 잡아야 고립 노드가 안 생긴다.
    rad = "[" + ", ".join(a.radius.split(",")) + "]"
    s = re.sub(r"radius=\[[^\]]*\]", "radius=" + rad, s, count=1)
    print("radius ->", rad)

open(a.dst, "w").write(s)
print("생성:", a.dst)

from mmcv import Config  # noqa: E402

c = Config.fromfile(a.dst)
print("  scene_list:", c.scene_list)
print("  real_dt:", c.real_dt)
print("  cam_transform_fn:", c.data.train.dataset.env_cfg.get("cam_transform_fn"))
print("  max_rollout_step:", c.max_rollout_step)
print("  radius:", c.model.preprocessor.radius
      if hasattr(c.model, "preprocessor") else "(확인 불가)")
