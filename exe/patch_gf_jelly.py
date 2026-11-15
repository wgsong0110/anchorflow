"""GF 젤리 장면 러너 (gs_simulation/jelly/gs_simulation_jellynacc.py) 를 고친 사본을 만든다. 시뮬레이션 코드는 건드리지 않는다.
  --mode run   : 프레임 루프 앞 배경(garden) 읽기를 --render_img 일 때만 돌게 감싼다 (저자 로컬 경로가 없어 죽던 곳)
  --mode nobg  : run + 영상용으로 배경 가우시안을 합치지 않고 (배경 없음), 래스터라이저 반환 2/3 개 둘 다 받게
  --mode gen   : 총알 자산 -> save_bullet_prop.pt 생성기. 러너 안 주석 처리된 저장 블록 (scale_factor 1/3, y += 0.7, save_core_init_render_vars) 을
                 그대로 되살리고 (러너의 총알 합치는 줄 대신), 그 위치에서 저장한 뒤 끝낸다. 젤리 config 의 속도 구속 상자 안에 총알 입자가 들어오는지 검산을 찍는다.
  python patch_gf_jelly.py SRC DST --mode run|nobg|gen"""
import re, sys
src = open(sys.argv[1]).read().split("\n"); mode = sys.argv[sys.argv.index("--mode") + 1]
out = []; i = 0
while i < len(src):
    ln = src[i]; st = ln.strip()
    if mode == "gen" and st == "save_filepath = 'save_bullet_prop.pt'":
        j = i
        while "mpm_init_cov = torch.concat([mpm_init_cov, loaded_data" not in src[j]: j += 1
        out += ['    scale_factor = 1 / 3          # 러너 주석 그대로 (총알을 1/3 로 줄여 +0.7 올림)',
                '    mpm_init_cov *= scale_factor ** 2',
                '    mpm_init_pos = mpm_init_pos.mean(dim=0) + scale_factor * (mpm_init_pos - mpm_init_pos.mean(dim=0))',
                '    mpm_init_pos[:, 1] += 0.7',
                "    save_core_init_render_vars(filepath='save_bullet_prop.pt', mpm_init_pos=mpm_init_pos, mpm_init_vol=mpm_init_vol, mpm_init_cov=mpm_init_cov, opacity_render=opacity, shs_render=shs)",
                '    _p = mpm_init_pos + torch.tensor([0.0, 2.0, 0.0], device=mpm_init_pos.device)      # 러너가 합친 뒤 y += 2.0',
                '    print("[총알] 입자", _p.shape[0], "bbox min", _p.min(0).values.tolist(), "max", _p.max(0).values.tolist(), "평균", _p.mean(0).tolist(), flush=True)',
                '    for _b in bc_params if isinstance(bc_params, list) else []: pass',
                '    _cj = json.load(open(args.config))',
                '    for _b in _cj["boundary_conditions"]:',
                '        if _b["type"] == "enforce_particle_translation":',
                '            _in = ((_p - torch.tensor(_b["point"], device=_p.device)).abs() < torch.tensor(_b["size"], device=_p.device)).all(-1)',
                '            print("[총알 검산] 속도 구속 상자 point", _b["point"], "size", _b["size"], "-> 상자 안 총알 입자", int(_in.sum()), "/", _p.shape[0], flush=True)',
                '    sys.exit(0)']
        i = j + 1; continue
    if mode in ("run", "nobg") and st.startswith('gaussians2 = load_checkpoint(') and not out_wrapped if False else False:
        pass
    out.append(ln); i += 1
txt = "\n".join(out)
if mode in ("run", "nobg"):
    L = txt.split("\n"); o2 = []; k = 0; done = False
    while k < len(L):
        s = L[k].strip()
        if not done and s.startswith('gaussians2 = load_checkpoint("/root/autodl-tmp') :
            ind = L[k][:len(L[k]) - len(L[k].lstrip())]
            o2.append(f"{ind}if {'False' if mode == 'nobg' else 'args.render_img'}:   # [anchorflow] 배경은 그릴 때만 (nobg 는 안 그림)")
            while "shs_render2 =" not in L[k] or L[k].strip().startswith("#"):
                o2.append("    " + L[k]); k += 1
            o2.append("    " + L[k]); k += 1; done = True; continue
        o2.append(L[k]); k += 1
    txt = "\n".join(o2)
    assert done, "배경 블록 못 찾음"
if mode == "nobg":
    txt = re.sub(r"torch\.concat\(\[\s*([^\[\]]+?)\s*,\s*\w+2\s*\]\s*,\s*dim\s*=\s*0\s*\)", r"\1", txt)
    txt = txt.replace("rendering, raddi, point_xy = rasterize(", "rendering, raddi, point_xy = _r3(rasterize)(").replace("rendering2, raddi2, point_xy2 = rasterize2(", "rendering2, raddi2, point_xy2 = _r3(rasterize2)(")
    # 래스터라이저는 means2D 를 입자 수만큼 쓴다 (graphdeco 판은 화면 좌표를 거기에 씀). 원본은 젤리 가우시안 수 (init_screen_points) 를 주는데 총알을 합쳐 입자가 더 많아 범위 밖 쓰기 -> 메모리 폭발. wm_render 와 같이 입자 수만큼 0 텐서
    txt = txt.replace("means2D=init_screen_points", "means2D=torch.zeros_like(pos_)")
    if "--traj" in sys.argv:      # 영상용: 시뮬 대신 h5 궤적 (환경변수 GFS_TRAJ 디렉토리, sim_<프레임+1>.h5) 의 x·F 를 솔버 상태에 넣고 공식 렌더 경로 (공분산 F C0 F^T, 회전 R) 를 그대로 -- 정답·우리 궤적을 같은 렌더러·카메라로
        loader = ("\nclass _TRJ:\n"
                  "    d = os.environ.get('GFS_TRAJ', '')\n"
                  "    @staticmethod\n"
                  "    def load(sol, frame):\n"
                  "        import h5py\n"
                  "        with h5py.File(os.path.join(_TRJ.d, 'sim_%010d.h5' % (frame + 1)), 'r') as h:\n"
                  "            x = np.array(h['x']); F = np.array(h['f_tensor'])\n"
                  "        x = x.T if x.shape[0] == 3 else x; F = (F.T if F.shape[0] == 9 else F).reshape(-1, 3, 3)\n"
                  "        sol.mpm_state.particle_x.assign(np.ascontiguousarray(x.astype(np.float32))); sol.mpm_state.particle_F.assign(np.ascontiguousarray(F.astype(np.float32)))\n")
        txt = txt.replace("\nif __name__", loader + "\nif __name__", 1)
        txt = txt.replace("mpm_solver.p2g2p(step, substep_dt, device=device, flip_pic_ratio=material_params['flip_pic_ratio'])", "pass")
        txt = txt.replace("                pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)", "                _TRJ.load(mpm_solver, frame)\n                pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)", 1)
    if "--dropbig" in sys.argv:   # 영상 전용: 총알 가우시안 하나 (공분산 trace 8117, 불투명도 0.0044, 나머지 trace 중앙값 1e-19) 가 래스터라이저를 터뜨림 -> 그림 쪽 불투명도만 0 (시뮬·정답 궤적에는 안 닿음)
        fix = ("            _trb = cov3D_[:, 0] + cov3D_[:, 3] + cov3D_[:, 5]\n"
               "            opacity_ = torch.where((_trb > 100).reshape(-1, 1), torch.zeros_like(opacity_), opacity_)\n")
        txt = txt.replace("            rendering, raddi, point_xy = _r3(rasterize)(", fix + "            rendering, raddi, point_xy = _r3(rasterize)(", 1)
    if "--dbg" in sys.argv:       # 렌더 직전 입력 통계 (프레임 0, 래스터라이저 메모리 폭발 원인 찾기용; 결과 불변)
        dbg = ("            if frame < 2:\n"
               "                print('[dbg] frame', frame, 'pos_', tuple(pos_.shape), 'nan', int((~torch.isfinite(pos_)).sum()), 'min', pos_.min(0).values.tolist(), 'max', pos_.max(0).values.tolist(), flush=True)\n"
               "                _tr = cov3D_[:, 0] + cov3D_[:, 3] + cov3D_[:, 5]; _big = _tr > 100; _ix = torch.nonzero(_big).flatten()\n"
               "                print('[dbg] big cov (trace>100)', int(_big.sum()), 'index min/max', (int(_ix.min()), int(_ix.max())) if _ix.numel() else None, 'n_jelly', int(gs_num) - 83580, 'opacity of big', opacity_[_big].flatten()[:8].tolist(), 'pos of big', pos_[_big][:4].tolist(), 'trace max', float(_tr.max()), flush=True)\n"
               "                _wv = current_camera.world_view_transform.to(pos_.device).float(); _p4 = torch.cat([pos_, torch.ones_like(pos_[:, :1])], 1) @ _wv; _zz = _p4[:, 2]; _fx = float(current_camera.image_width) / (2 * math.tan(current_camera.FoVx * 0.5)); _rad = 3 * torch.sqrt(torch.clamp(_tr / 3, min=0)) * _fx / _zz.clamp(min=1e-3)\n"
               "                print('[dbg] cam', 'W', current_camera.image_width, 'H', current_camera.image_height, 'z min', float(_zz.min()), 'z<0.2', int((_zz < 0.2).sum()), 'rad>1e3', int(((_rad > 1e3) & (_zz > 0.2)).sum()), 'rad>1e5', int(((_rad > 1e5) & (_zz > 0.2)).sum()), 'rad max', float(_rad[_zz > 0.2].max()), 'camera_center', current_camera.camera_center.tolist(), flush=True)\n"
               "                for _nm, _sl in (('jelly', slice(0, int(gs_num) - 83580)), ('bullet', slice(int(gs_num) - 83580, None)), ('jelly_half1', slice(0, (int(gs_num) - 83580) // 2)), ('jelly_half2', slice((int(gs_num) - 83580) // 2, int(gs_num) - 83580))):\n"
               "                    try:\n"
               "                        _o = _r3(rasterize)(means3D=pos_[_sl], means2D=torch.zeros_like(pos_[_sl]), shs=None, colors_precomp=colors_precomp[_sl].float(), opacities=opacity_[_sl], scales=None, rotations=None, cov3D_precomp=cov3D_[_sl]); torch.cuda.synchronize(); print('[dbg] subset', _nm, 'ok', flush=True)\n"
               "                    except Exception as _e:\n"
               "                        print('[dbg] subset', _nm, 'fail', str(_e)[:90].replace(chr(10), ' '), flush=True)\n"
               "                print('[dbg] cov3D_', tuple(cov3D_.shape), 'nan', int((~torch.isfinite(cov3D_)).sum()), 'absmax', float(cov3D_.abs().max()), 'median', float(cov3D_.abs().median()), 'opacity', float(opacity_.min()), float(opacity_.max()), 'colors', tuple(colors_precomp.shape), 'gs_num', gs_num, 'scale_origin', float(scale_origin), flush=True)\n")
        txt = txt.replace("            rendering, raddi, point_xy = _r3(rasterize)(", dbg + "            rendering, raddi, point_xy = _r3(rasterize)(", 1)
    txt = txt.replace("\nif __name__", "\ndef _r3(f):\n    def g(**k):\n        o = f(**k); return o if len(o) == 3 else (o[0], o[1], None)\n    return g\n\nif __name__", 1)
open(sys.argv[2], "w").write(txt); print("완료", mode)
