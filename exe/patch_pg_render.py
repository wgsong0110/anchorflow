"""PG 클론에 **다중 카메라 프레임 저장**을 넣는다. 여러 번 돌려도 안전하다.

두 가지를 한 번에 얻는다: (1) GausSim 재학습용 다시점 영상, (2) 시각품질의
기준 영상. 렌더는 가우시안과 1:1 인 전체 입자로만 제대로 되므로 시뮬 **중에**
그려야 한다 -- 2 만 개로 줄인 덤프에서는 그릴 수 없다.

환경변수
  AF_R_OUT    프레임을 쓸 폴더 (없으면 아무 것도 안 한다)
  AF_R_CAMS   "방위,고도;방위,고도;..." 목록. 비우면 config 카메라 한 대
  AF_R_EVERY  이 간격의 프레임만 저장 (기본 1)

  python exe/patch_pg_render.py --pg /root/work/PG_pgtraj
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()
p = os.path.join(a.pg, "gs_simulation.py")
s = open(p).read()
if "AF_R_OUT" in s:
    print("[패치] 이미 들어 있다")
    raise SystemExit(0)

old = "    for frame in tqdm(range(frame_num)):\n"
new = ('''    # [anchorflow] 다중 카메라 프레임 저장 준비
    _af_rout = _af_os.environ.get("AF_R_OUT")
    _af_rcams = []
    if _af_rout:
        _af_os.makedirs(_af_rout, exist_ok=True)
        _spec = _af_os.environ.get("AF_R_CAMS", "")
        for _t in [q for q in _spec.split(";") if q]:
            _aa, _ee = _t.split(",")
            _af_rcams.append((float(_aa), float(_ee)))
        if not _af_rcams:
            _af_rcams = [(camera_params["init_azimuthm"],
                          camera_params["init_elevation"])]
        _af_revery = int(_af_os.environ.get("AF_R_EVERY", 1))
        import imageio.v2 as _af_iio
        for _ci in range(len(_af_rcams)):
            _af_os.makedirs(_af_os.path.join(_af_rout, f"cam_{_ci:05d}"),
                            exist_ok=True)
        print(f"[렌더] {_af_rout} 카메라 {len(_af_rcams)} 대, "
              f"{_af_revery} 프레임마다", flush=True)
    for frame in tqdm(range(frame_num)):
''')
assert old in s
s = s.replace(old, new, 1)

# 손잡이 표식을 붙이기 **전에** 그린다 -- 학습 데이터에 표식이 들어가면 안 된다.
old2 = """            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
            if _af_on:"""
new2 = """            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
            if _af_rout and (frame % _af_revery == 0):
                # [anchorflow] 표식 없는 깨끗한 프레임을 카메라마다 저장한다.
                # SH 는 카메라에 따라 달라지므로 카메라별로 다시 계산한다.
                for _ci, (_aa, _ee) in enumerate(_af_rcams):
                    _cam2 = get_camera_view(
                        model_path,
                        default_camera_index=camera_params["default_camera_index"],
                        center_view_world_space=viewpoint_center_worldspace,
                        observant_coordinates=observant_coordinates,
                        show_hint=False, init_azimuthm=_aa, init_elevation=_ee,
                        init_radius=camera_params["init_radius"],
                        move_camera=False, current_frame=0,
                        delta_a=0.0, delta_e=0.0, delta_r=0.0)
                    _rast2 = initialize_resterize(_cam2, gaussians, pipeline,
                                                  background)
                    _col2 = convert_SH(shs, _cam2, gaussians, pos, rot)
                    _out2 = _rast2(means3D=pos, means2D=init_screen_points,
                                   shs=None, colors_precomp=_col2,
                                   opacities=opacity, scales=None,
                                   rotations=None, cov3D_precomp=cov3D)
                    _arr = (_out2[0].clamp(0, 1).permute(1, 2, 0)
                            .detach().cpu().numpy() * 255).astype("uint8")
                    _af_iio.imwrite(_af_os.path.join(
                        _af_rout, f"cam_{_ci:05d}",
                        f"{frame // _af_revery:05d}.jpg"), _arr, quality=95)
            if _af_on:"""
assert old2 in s
s = s.replace(old2, new2, 1)
open(p, "w").write(s)
print("[패치] AF_R_OUT 다중 카메라 렌더 추가 완료")
