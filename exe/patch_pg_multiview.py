"""PG/i-PG 의 `gs_simulation.py` 에 **다중시점 렌더**를 붙인다.

왜: Spring-Gaus 는 다중시점 **영상**에서 물성을 맞춘다. 우리 형상에는 그런 관측이
없으므로, MPM 기준 궤적을 여러 카메라에서 렌더해 그 입력을 만든다 (사용자 지시).

원본은 프레임마다 카메라 하나만 렌더한다. `AF_MV_CAMS=<cameras.json>` 이 있으면
그 목록의 카메라 전부로 렌더해 `<output>/cam_<i>/<frame:03>.png` 로 저장한다
(Spring-Gaus 의 MPM_Synthetic 이 기대하는 배치 그대로).

  python exe/patch_pg_multiview.py --repo /home/dkta/work/PhysGaussian
"""
from __future__ import annotations

import argparse
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
a = ap.parse_args()

p = os.path.join(a.repo, "gs_simulation.py")
s = open(p).read()
if "AF_MV_CAMS" in s:
    print(f"이미 패치됨: {p}")
    raise SystemExit(0)
bak = p + ".premv"
if not os.path.exists(bak):
    shutil.copy(p, bak)

SETUP = '''
    # [anchorflow] 다중시점 렌더 준비 (Spring-Gaus 입력 생성용)
    _af_mv_path = os.environ.get("AF_MV_CAMS")
    _af_mv_cams = []
    if _af_mv_path:
        import json as _afj
        from scene.cameras import Camera as _AFCam
        from utils.graphics_utils import focal2fov as _aff2f
        for _ci, _c in enumerate(_afj.load(open(_af_mv_path))):
            _Rc2w = np.array(_c["rotation"], dtype=np.float64)
            _pos = np.array(_c["position"], dtype=np.float64)
            _T = -_Rc2w.T @ _pos
            _af_mv_cams.append((_ci, _AFCam(
                colmap_id=_ci, R=_Rc2w, T=_T,
                FoVx=_aff2f(_c["fx"], _c["width"]),
                FoVy=_aff2f(_c["fy"], _c["height"]),
                image=torch.zeros(3, _c["height"], _c["width"]),
                gt_alpha_mask=None, image_name=f"cam_{_ci}", uid=_ci,
                data_device="cuda")))
        print(f"[AF다중시점] 카메라 {len(_af_mv_cams)} 대  {_af_mv_path}",
              flush=True)
        # Spring-Gaus 는 바닥면과 중력축을 **월드 좌표**로 받는다. 시뮬 공간의
        # 바닥(z=floor)과 위 방향을 그대로 역변환해 적어 둔다.
        _af_pls = torch.tensor([[1.0, 1.0, 0.1], [1.0, 1.0, 1.1]],
                               device="cuda", dtype=torch.float32)
        _af_plw = apply_inverse_rotations(
            undotransform2origin(undoshift2center111(_af_pls), scale_origin,
                                 original_mean_pos), rotation_matrices)
        _af_p0 = _af_plw[0].tolist()
        _af_up = (_af_plw[1] - _af_plw[0])
        _af_up = (_af_up / _af_up.norm()).tolist()
        _af_bb = apply_inverse_rotations(
            undotransform2origin(undoshift2center111(transformed_pos),
                                 scale_origin, original_mean_pos),
            rotation_matrices)
        _afj2 = __import__("json")
        _afj2.dump(dict(floor_point=_af_p0, up=_af_up,
                        xyz_min=_af_bb.min(0).values.tolist(),
                        xyz_max=_af_bb.max(0).values.tolist(),
                        frame_dt=float(time_params["frame_dt"]),
                        n_frames=int(frame_num)),
                   open(os.path.join(args.output_path, "sgmeta.json"), "w"),
                   indent=1)
        print(f"[AF다중시점] 바닥점 {_af_p0}, 위 {_af_up}", flush=True)

'''
mark = "    for frame in tqdm(range(frame_num)):"
s = s.replace(mark, SETUP + mark, 1)

RENDER = '''
        if _af_mv_cams:
            # 프레임마다 **모든 카메라**로 렌더해 저장한다
            _pos_mv = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
            _cov_mv = mpm_solver.export_particle_cov_to_torch()
            _rot_mv = mpm_solver.export_particle_R_to_torch()
            _cov_mv = _cov_mv.view(-1, 6)[:gs_num].to(device)
            _rot_mv = _rot_mv.view(-1, 3, 3)[:gs_num].to(device)
            _pos_mv = apply_inverse_rotations(
                undotransform2origin(undoshift2center111(_pos_mv),
                                     scale_origin, original_mean_pos),
                rotation_matrices)
            _cov_mv = _cov_mv / (scale_origin * scale_origin)
            _cov_mv = apply_inverse_cov_rotations(_cov_mv, rotation_matrices)
            for _ci, _cam in _af_mv_cams:
                _rast = initialize_resterize(_cam, gaussians, pipeline,
                                             background)
                _cols = convert_SH(shs_render, _cam, gaussians, _pos_mv,
                                   _rot_mv)
                _o = _rast(means3D=_pos_mv, means2D=init_screen_points,
                           shs=None, colors_precomp=_cols,
                           opacities=opacity_render, scales=None,
                           rotations=None, cov3D_precomp=_cov_mv)[0]
                _img = _o.permute(1, 2, 0).detach().cpu().numpy()
                _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
                _d = os.path.join(args.output_path, f"cam_{_ci}")
                os.makedirs(_d, exist_ok=True)
                cv2.imwrite(os.path.join(_d, f"{frame:03d}.png"), 255 * _img)
'''
mark2 = "        if args.render_img:"
s = s.replace(mark2, RENDER + mark2, 1)
open(p, "w").write(s)
print(f"패치 완료: {p} (원본 {bak})")
