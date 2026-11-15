"""절 CM189 (조정자 11-15 사용자 지시 "pillow2sofa·wolf 뺀 모든 장면을 레고와 똑같은 방식으로 시작"): 점소성 세 장면 config 를 레고 L000 세팅으로.
레고 세팅 = elastic_new ship_plastic_lego.json 과 같은 키 (lbdata/config.json 의 substep_dt 2.0008e-5·frame_dt 1/60·n_grid 100·bounding_box + sticky surface_collider,
프레임 80 (L000 궤적), 초기 속도 0, 회전 = L000 초기 회전 R (x 119.5764, y -6.4234, z 138.3083), 바닥 = 최저점 - lift 0.00373 (_floor_gap, exe/elastic_new_floor.py)).
재질 키 (material·E·nu·density·yield_stress·plastic_viscosity·scale)·채우기는 각 장면 그대로. 사용: python exe/plastfix_lego_cfg.py (exe/nshape_cfg/<장면>_lego.json 셋 생성)"""
import copy, json, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nshape_cfg")
LEGO = dict(rotation_degree=[119.5764, -6.4234, 138.3083], rotation_axis=[0, 1, 2], substep_dt=2.000800320128051e-05, frame_dt=1 / 60, frame_num=80,
            n_grid=100, _floor_gap=0.003729709149228453)
BC = [{"type": "bounding_box"}, {"type": "surface_collider", "point": [1, 1, 0.2], "normal": [0.0, 0.0, 1.0], "surface": "sticky", "friction": 0.0, "start_time": 0, "end_time": 1000.0}]
for sc in ("chair_paste", "drums_honey", "bread_foam_drop"):
    c = json.load(open(f"{D}/{sc}.json"))
    c.update(copy.deepcopy(LEGO)); c["boundary_conditions"] = copy.deepcopy(BC)
    c["_plastfix_lego"] = (f"{sc}.json 에서 레고 L000 세팅으로 (조정자 11-15): 회전 L000 초기 R, substep_dt 2.0008e-5, frame_dt 1/60, 80 프레임, 초기 속도 0, n_grid 100, "
                           "경계 = bounding_box + sticky 바닥 (z = 최저점 - lift 0.00373, 실행 때 elastic_new_floor.py), 재질·채우기는 원래 장면 그대로")
    json.dump(c, open(f"{D}/{sc}_lego.json", "w"), indent=1, ensure_ascii=False)
    print(sc, {k: c[k] for k in ("material", "E", "nu", "density", "yield_stress", "rotation_degree", "frame_num")}, c.get("plastic_viscosity"), c.get("scale"))
