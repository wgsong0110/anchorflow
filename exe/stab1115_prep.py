"""[stab1115] 서브스텝 안정성 PG 논문 장면 판 입력 만들기: <out>/<씬>/{config.json, init/sim_0000000000.h5}.
낙하 장면 정의 = 기존 stab (ficus drop config: jelly E 1e6 nu 0.3 rho 1000, g -9.8, 바닥 z 0.48 sticky, 격자 100 grid_lim 2, 1/60 s x 60 프레임) 그대로.
입자 = 그 씬의 PG 공식 채우기 0 프레임 (R2 paper_remeasure2/scenes/<씬>/h5 의 sim_0), 변환 필드 (회전·sim_area·scale 등, 렌더 대응용) = 공식 config 에서.
사용: python exe/stab1115_prep.py <ficus 낙하 config.json> <공식 config 폴더> <h5 폴더들 루트> <out>
"""
import copy, json, os, shutil, sys
base = json.load(open(sys.argv[1])); off, h5r, out = sys.argv[2], sys.argv[3], sys.argv[4]
KEEP = ("rotation_degree", "rotation_axis", "sim_area", "scale", "opacity_threshold", "particle_filling", "mpm_space_vertical_upward_axis",
        "mpm_space_viewpoint_center", "default_camera_index", "init_azimuthm", "init_elevation", "init_radius", "move_camera", "delta_a", "delta_e", "delta_r", "show_hint")
for s in sys.argv[5:]:
    if s in ("ship_plastic_lego", "mic_metal_lego"):  # [stabfinal] 새 형상 = 정답 PG 사본 (R2 nshape/gt_pg/<장면>) 의 config (바닥 해석된 값) · 0 프레임 그대로
        c = json.load(open(f"{h5r}/{s}/gt/config.json"))
        c.setdefault("grid_lim", 2.0)                # PG decode_param 기본값 (config 에 없음) 을 명시 (pg_h5_to_pt 와 같음)
        c["stab_official"] = True                    # stab_mpm·stab_omni: 재질·감쇠·경계조건을 공식 함수로
        c["drop_note"] = f"레고 L000 세팅 + PG 재질 ({c.get('material')}); 정답 PG 사본 config·0 프레임 그대로"
        os.makedirs(f"{out}/{s}/init", exist_ok=True)
        json.dump(c, open(f"{out}/{s}/config.json", "w"), indent=1)
        if not os.path.exists(f"{out}/{s}/init/sim_0000000000.h5"):
            shutil.copyfile(f"{h5r}/{s}/gt/simulation_ply/sim_0000000000.h5", f"{out}/{s}/init/sim_0000000000.h5")
        print("[prep]", s, flush=True)
        continue
    if s in ("ficus", "lego"):                       # lego = 기존 stab (11-12 표) 과 같은 낙하 config (물리·회전 0 동일)
        c = copy.deepcopy(base)
    else:
        o = json.load(open(f"{off}/{s}_config.json")); c = copy.deepcopy(base)
        for k in KEEP:
            if k in o:
                c[k] = o[k]
            elif k in c and k in ("sim_area", "particle_filling", "scale"):
                c.pop(k, None)
        c["drop_note"] = f"PG 공식 {s} 0 프레임 입자 + 낙하 물리 (ficus 기존 stab config 와 같음)"
    os.makedirs(f"{out}/{s}/init", exist_ok=True)
    json.dump(c, open(f"{out}/{s}/config.json", "w"), indent=1)
    src = f"{h5r}/{s}/init/sim_0000000000.h5"
    if not os.path.exists(f"{out}/{s}/init/sim_0000000000.h5"):
        shutil.copyfile(src, f"{out}/{s}/init/sim_0000000000.h5")
    print("[prep]", s, flush=True)
