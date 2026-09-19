"""GF 이식본이 **어떤 씬에서도** 같은 답을 내는지 볼 스윕 config 를 찍어낸다.

"임의의 씬" 은 결국 솔버가 밟는 갈래의 조합이다. 그래서 모델 하나를 놓고
재질 여섯 갈래 x 전달 방식 둘 x 경계·구동 조건을 갈라서 덮는다.

  0 jelly        FCR
  1 metal        von Mises + StVK
  2 sand         Drucker-Prager (부피를 평균으로 쓰는 갈래도 여기서만 탄다)
  3 foam         점소성 + StVK
  5 plasticine   von Mises + 손상(softening)
  7 watermelon   비연관 Cam-Clay + neoHookeanBoarden

프레임 수를 적게 두는 것은 일부러다 -- 갈리는 솔버는 첫 몇 프레임에서 이미 갈린다.
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n_grid", type=int, default=100)
ap.add_argument("--frames", type=int, default=12)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

BASE = {
    "opacity_threshold": 0.02,
    "rotation_degree": [0.0], "rotation_axis": [0],
    "substep_dt": 1e-4, "frame_dt": 1e-2, "frame_num": a.frames,
    "E": 2e3, "nu": 0.38, "density": 200.0,
    "material": "jelly", "grid_lim": 2.0, "n_grid": a.n_grid, "scale": 1.0,
    "rpic_damping": 0.0, "grid_v_damping_scale": 1.0,
    "flip_pic_ratio": 0.0,
    "g": [0.0, 0.0, -9.8],
    "init_velocity": [0.0, 0.0, -2.0],
    "use_config_dt": True,
    "boundary_conditions": [{"type": "bounding_box"}],
    "mpm_space_vertical_upward_axis": [0, 0, 1],
    "mpm_space_viewpoint_center": [1, 1, 1],
    "default_camera_index": -1, "show_hint": False,
    "init_azimuthm": 55, "init_elevation": 10, "init_radius": 3.0,
    "move_camera": False, "delta_a": 0.0, "delta_e": 0.0, "delta_r": 0.0,
}

FLOOR = {"type": "surface_collider", "point": [1, 1, 0.35],
         "normal": [0.0, 0.0, 1.0], "surface": "sticky", "friction": 0.0,
         "start_time": 0, "end_time": 1000.0}

SCENES = {
    # --- 재질 여섯 갈래 ---
    "m0_jelly":      dict(material="jelly"),
    "m1_metal":      dict(material="metal", yield_stress=1e3, hardening=1.0, xi=1.0),
    "m2_sand":       dict(material="sand", friction_angle=35.0),
    "m3_foam":       dict(material="foam", yield_stress=3e2, plastic_viscosity=1.0),
    "m5_plasticine": dict(material="plasticine", yield_stress=5e2, softening=0.1,
                          hardening=0.0),
    "m7_watermelon": dict(material="watermelon", E=2e3, nu=0.38, density=1,
                          friction_angle=45.0, beta=1.0, xi=3.0, hardening=1.0,
                          alpha_0=-0.04, g=[0.0, 0.0, -15.0],
                          init_velocity=[0.0, 0.0, -6.0], flip_pic_ratio=0.7),
    # --- 전달 방식 ---
    "t_apic_rpic":   dict(material="jelly", rpic_damping=0.3),
    "t_pic":         dict(material="jelly", rpic_damping=-1.0),
    "t_flip03":      dict(material="jelly", flip_pic_ratio=0.3),
    "t_flip09":      dict(material="jelly", flip_pic_ratio=0.9),
    # --- 격자 감쇠 ---
    "g_damp":        dict(material="jelly", grid_v_damping_scale=0.95),
    # --- 경계·구동 조건 ---
    "b_floor":       dict(material="jelly", _bc=[{"type": "bounding_box"}, FLOOR]),
    "b_slip":        dict(material="jelly", _bc=[
        {"type": "bounding_box"},
        dict(FLOOR, surface="slip", friction=0.3)]),
    "b_cut":         dict(material="jelly", _bc=[
        {"type": "bounding_box"},
        dict(FLOOR, surface="cut", point=[1, 1, 0.45])]),
    "b_cuboid":      dict(material="jelly", _bc=[
        {"type": "bounding_box"},
        {"type": "cuboid", "point": [1, 1, 1], "size": [0.3, 0.3, 0.3],
         "velocity": [1.0, 0.0, 0.0], "start_time": 0.0, "end_time": 0.05,
         "reset": 1}]),
    "b_impulse":     dict(material="jelly", _bc=[
        {"type": "bounding_box"},
        {"type": "particle_impulse", "force": [0.0, 400.0, 0.0],
         "point": [1, 1, 1], "size": [1, 1, 1], "num_dt": 40,
         "start_time": 0.02}]),
    "b_translate":   dict(material="jelly", _bc=[
        {"type": "bounding_box"},
        {"type": "enforce_particle_translation", "point": [1, 1, 1.3],
         "size": [1, 1, 0.25], "velocity": [0.0, 0.0, 0.0],
         "start_time": 0.0, "end_time": 0.08}]),
    "b_rotate":      dict(material="jelly", _bc=[
        {"type": "bounding_box"},
        {"type": "enforce_particle_velocity_rotation", "point": [1, 1, 1],
         "normal": [0, 0, 1], "half_height_and_radius": [0.5, 0.5],
         "rotation_scale": 6.0, "translation_scale": 0.0,
         "start_time": 0.0, "end_time": 0.06}]),
    "b_release":     dict(material="jelly", _bc=[
        {"type": "bounding_box"},
        {"type": "release_particles_sequentially", "normal": [0, 0, 1],
         "start_position": 1.4, "end_position": 0.6, "num_layers": 10,
         "start_time": 0.0, "end_time": 0.1}]),
    # --- 구역별 물성 ---
    "p_region":      dict(material="jelly", additional_material_params=[
        {"point": [1, 1, 1.2], "size": [1, 1, 0.3],
         "E": 2e4, "nu": 0.3, "density": 50.0}]),
}

for tag, over in SCENES.items():
    cfg = dict(BASE)
    bc = over.pop("_bc", None)
    cfg.update(over)
    if bc is not None:
        cfg["boundary_conditions"] = bc
    json.dump(cfg, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
print(f"[저장] {a.out} 에 {len(SCENES)} 개: {', '.join(SCENES)}")
