"""[paper_remeasure2] 코드 사본 (R2 tar) 의 3DGS 모델 이름 표에 PhysGaussian 논문 장면 셋을 더함 (저장소 파일은 안 건드림; 부팅 스크립트가 tar 푼 뒤 부름)."""
import re
for f, key in (("lib/anchorflow/rtrender1.py", "MODEL"), ("exe/lb_render.py", "MODEL"), ("exe/stab_mv_render.py", "MODEL")):
    s = open(f).read()
    if "pillow2sofa_whitebg-trained" in s:
        continue
    s = re.sub(r'(MODEL = \{)', r'\1"vasedeck": "vasedeck_whitebg-trained", "plane": "plane-trained", "pillow2sofa": "pillow2sofa_whitebg-trained", ', s, count=1)
    open(f, "w").write(s)
    print("패치", f)

# PG 전처리의 sim_area 자르기 (vasedeck · plane 의 공식 config 에 있음; 시뮬 입자 = 가우시안 중 영역 안의 것) 를 시뮬 안 렌더 (rtrender1) 가 반영하게 -- 새 장면에서만
f = "lib/anchorflow/rtrender1.py"
s = open(f).read()
old = '    TP, SO, MEAN = transform2origin(gs.get_xyz[KIDX] @ RT.T if ROT else gs.get_xyz[KIDX], float(cfg.get("scale", 1.0)))\n'
new = '''    _XR = gs.get_xyz[KIDX] @ RT.T if ROT else gs.get_xyz[KIDX]
    if C.shape in ("vasedeck", "plane") and cfg.get("sim_area") is not None:
        _b = cfg["sim_area"]
        _m = torch.ones(_XR.shape[0], dtype=torch.bool, device=_XR.device)
        for _i in range(3):
            _m = _m & (_XR[:, _i] > _b[2 * _i]) & (_XR[:, _i] < _b[2 * _i + 1])
        KIDX = KIDX[_m]
        _XR = _XR[_m]
    TP, SO, MEAN = transform2origin(_XR, float(cfg.get("scale", 1.0)))
'''
if old in s and "_XR" not in s:
    open(f, "w").write(s.replace(old, new, 1)); print("패치 sim_area", f)

# 다시점 렌더 (stab_mv_render.py, Spring-Gaus 준비 판) 도 같은 sim_area 자르기 (vasedeck·plane; final_pa_bl: pillow2sofa 도 -- 자르지 않으면 정답 h5 와 가우시안 번호 대응 오차 0.88, 자르면 0)
f = "exe/stab_mv_render.py"
s = open(f).read()
old = '''TP, SO, MEAN = transform2origin(gs.get_xyz[KIDX] @ _RT.T if ROT else gs.get_xyz[KIDX], float(cfg.get("scale", 1.0)))\n'''
new = '''_XR = gs.get_xyz[KIDX] @ _RT.T if ROT else gs.get_xyz[KIDX]
if a.shape in ("vasedeck", "plane", "pillow2sofa") and cfg.get("sim_area") is not None:                     # PG 전처리 sim_area 자르기 (paper_remeasure2)
    _b = cfg["sim_area"]
    _m = torch.ones(_XR.shape[0], dtype=torch.bool, device=_XR.device)
    for _i in range(3):
        _m = _m & (_XR[:, _i] > _b[2 * _i]) & (_XR[:, _i] < _b[2 * _i + 1])
    KIDX = KIDX[_m]
    _XR = _XR[_m]
TP, SO, MEAN = transform2origin(_XR, float(cfg.get("scale", 1.0)))
'''
if old in s and "_XR" not in s:
    open(f, "w").write(s.replace(old, new, 1)); print("패치 sim_area", f)
