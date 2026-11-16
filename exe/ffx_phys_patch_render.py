"""[final_fix_phys] 코드 사본 (R2 tar) 의 exe/lb_render.py 에 PG 전처리 sim_area 자르기를 pillow2sofa (vasedeck·plane) 에 넣음 -- exe/pr2_patch_models.py 가 stab_mv_render.py 에 넣은 것과 같은 식
(자르지 않으면 정답 h5 와 가우시안 번호 대응 오차 0.88 로 lb_render 대응 검사에서 멈춤, 자르면 0). 저장소 lb_render.py 는 안 건드림; 인스턴스 코드 폴더에서 한 번 부름."""
f = "exe/lb_render.py"
s = open(f).read()
old = '''TP, SO, MEAN = transform2origin(gs.get_xyz[KIDX] @ _RT.T if ROT else gs.get_xyz[KIDX], float(cfg.get("scale", 1.0)))\n'''
new = '''_XR = gs.get_xyz[KIDX] @ _RT.T if ROT else gs.get_xyz[KIDX]
if a.shape in ("vasedeck", "plane", "pillow2sofa") and cfg.get("sim_area") is not None:                     # [final_fix_phys] PG 전처리 sim_area 자르기
    _b = cfg["sim_area"]
    _m = torch.ones(_XR.shape[0], dtype=torch.bool, device=_XR.device)
    for _i in range(3):
        _m = _m & (_XR[:, _i] > _b[2 * _i]) & (_XR[:, _i] < _b[2 * _i + 1])
    KIDX = KIDX[_m]
    _XR = _XR[_m]
TP, SO, MEAN = transform2origin(_XR, float(cfg.get("scale", 1.0)))
'''
if "_XR" not in s:
    assert old in s
    open(f, "w").write(s.replace(old, new, 1)); print("패치 sim_area", f)
