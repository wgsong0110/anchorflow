"""우리 PG 궤적을 GausSim 이 읽는 배치로 만든다 (그쪽 정의 그대로).

GausSim 의 감독은 **다시점 렌더 손실**이다. 위치를 직접 감독하게 고치면 그쪽
방법이 아니게 되므로 손대지 않고, 대신 PG 궤적을 여러 카메라로 렌더해 넣는다.

읽는 배치 (Blender/NeRF 경로 -- COLMAP 보다 간단하고 같은 코드 경로다):
  {scene}/transforms_train.json   camera_angle_x + frames[{file_path, transform_matrix}]
  {scene}/transforms_test.json
  {scene}/images/seq_%05d_frame_%05d.jpg          카메라별 기준 한 장
  {scene}/video_images/seq_%05d_frame_%05d/%05d.jpg  그 카메라의 프레임들
  {scene}/point_cloud.ply                          3DGS 원본
  {scene}/clean_object_points.ply, moving_part_points.ply
  {scene}/pin_mask.json                            고정 입자 색인

file_path 는 `seq` 로 시작해야 하고 마지막 다섯 자리가 카메라 번호다
(`_read_camera_transforms` 가 그렇게 뜯는다).

  python exe/make_gaussim_data.py --dumps bench_pg --combo mic_clayC \
      --model pgmodel/mic_whitebg-trained --cfg bench_cfg/mic_clayC.json \
      --out gaussim_data/mic_clayC --seeds 0-5 --n_cam 8 --stride 4
"""
import argparse
import json
import os
import shutil

import sys

import numpy as np
import torch

# 렌더는 PG 클론의 utils/scene 을 쓴다. 경로를 여기서 붙여 준다 (러너가
# PYTHONPATH 를 잡아 주지 않아도 돌게).
_PG = os.environ.get("AF_PG", os.path.join(
    os.environ.get("AF_WORK", "/root/work"), "PG_pgtraj"))
for _p in (_PG, os.path.join(_PG, "gaussian-splatting")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from anchorflow.gsrender import GSScene            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dumps", required=True, help="PG 덤프 폴더")
ap.add_argument("--combo", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--cfg", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", default="0-5")
ap.add_argument("--n_cam", type=int, default=8)
ap.add_argument("--stride", type=int, default=4, help="프레임 간격")
ap.add_argument("--frames", type=int, default=240)
ap.add_argument("--test_seqs", type=int, default=1)
ap.add_argument("--white_bg", type=int, default=1)
a = ap.parse_args()

dev = "cuda"
os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
os.makedirs(os.path.join(a.out, "video_images"), exist_ok=True)
lo, hi = (a.seeds.split("-") + [a.seeds])[:2]
seeds = list(range(int(lo), int(hi) + 1))

import imageio.v2 as imageio

d0 = torch.load(os.path.join(a.dumps, f"{a.combo}_s{seeds[0]:02d}.pt"),
                map_location="cpu", weights_only=False)
X0 = d0["x"][0].float().to(dev)
scene = GSScene(a.model, a.cfg, X0, device=dev, white_bg=bool(a.white_bg))
print(f"[대응] 가우시안 {scene.gs_num} / 입자 {X0.shape[0]}, 첫 프레임 최대 차 "
      f"{scene.fit:.3e}", flush=True)

# 카메라는 물체를 둘러싼 고리에서 고르게 뽑는다. GSScene 이 config 의 카메라
# 규약을 이미 들고 있으므로 frame 번호로 골라 쓰는 대신 여기서 직접 만든다.
cam_json = {"camera_angle_x": None, "frames": []}
test_json = {"camera_angle_x": None, "frames": []}
# 카메라는 config 의 방위·고도만 바꿔 고리에서 고르게 뽑는다 (그쪽 규약 유지).
BASE_AZ = float(scene.cam["init_azimuthm"])
BASE_EL = float(scene.cam["init_elevation"])
VIEWS = [(BASE_AZ + 360.0 * c / a.n_cam, BASE_EL + 20.0 * ((c % 3) - 1))
         for c in range(a.n_cam)]

for si, sd in enumerate(seeds):
    p = os.path.join(a.dumps, f"{a.combo}_s{sd:02d}.pt")
    if not os.path.exists(p):
        print(f"[없음] {p}")
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    xs, Fs = d["x"].float(), (d["F"].float() if d.get("F") is not None else None)
    T = min(xs.shape[0], a.frames + 1)
    tgt = test_json if si < a.test_seqs else cam_json
    for c in range(a.n_cam):
        name = f"seq_{si:05d}_frame_{c:05d}"
        vdir = os.path.join(a.out, "video_images", name)
        os.makedirs(vdir, exist_ok=True)
        az, el = VIEWS[c]
        scene.set_view(azim=az, elev=el)
        c2w, fovx = scene.blender_c2w(0)
        if cam_json["camera_angle_x"] is None:
            cam_json["camera_angle_x"] = fovx
            test_json["camera_angle_x"] = fovx
        k = 0
        for t in range(0, T, a.stride):
            with torch.no_grad():
                img = scene.render(xs[t], Fs[t].to(dev) if Fs is not None else None,
                                   frame=t)
            if torch.is_tensor(img):
                img = img.clamp(0, 1)
                if img.dim() == 3 and img.shape[0] == 3:
                    img = img.permute(1, 2, 0)
                arr = (img.detach().cpu().numpy() * 255).astype(np.uint8)
            else:
                arr = np.asarray(img)
            imageio.imwrite(os.path.join(vdir, f"{k:05d}.jpg"), arr, quality=95)
            if k == 0:
                imageio.imwrite(os.path.join(a.out, "images", name + ".jpg"),
                                arr, quality=95)
            k += 1
        tgt["frames"].append(dict(file_path=f"images/{name}",
                                  transform_matrix=c2w.tolist()))
        print(f"[렌더] {name}: {k} 프레임", flush=True)

json.dump(cam_json, open(os.path.join(a.out, "transforms_train.json"), "w"))
json.dump(test_json, open(os.path.join(a.out, "transforms_test.json"), "w"))
# 3DGS 원본과 물체 점군, 고정 입자
it = max(int(x.split("_")[-1]) for x in
         os.listdir(os.path.join(a.model, "point_cloud")))
shutil.copy(os.path.join(a.model, "point_cloud", f"iteration_{it}",
                         "point_cloud.ply"),
            os.path.join(a.out, "point_cloud.ply"))
json.dump([], open(os.path.join(a.out, "pin_mask.json"), "w"))
print(f"[완료] {a.out}", flush=True)
