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

open(a.dst, "w").write(s)
print("생성:", a.dst)

from mmcv import Config  # noqa: E402

c = Config.fromfile(a.dst)
print("  scene_list:", c.scene_list)
print("  real_dt:", c.real_dt)
print("  cam_transform_fn:", c.data.train.dataset.env_cfg.get("cam_transform_fn"))
print("  max_rollout_step:", c.max_rollout_step)
