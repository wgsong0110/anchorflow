"""warp MPM 만으로 돌릴 네 씬의 config 를 찍어낸다.

경계 구역은 입자 구름의 실제 크기에서 뽑아야 하므로, h5 를 읽어 bbox 를 보고
숫자를 채운다. 어휘는 GF 의 config 그대로다 (`enforce_particle_translation`,
`bounding_box`, `surface_collider`).

  모래  wolf 채운 구름 + 재질 2(Drucker-Prager)
  조작  wolf 구름 + 재질 2, 다지기/밀기/쓸기를 시간 창마다 가한다
  인열  빵 채운 구름 + 노치 + 재질 7, 양끝을 붙잡고 당긴다
  접합  빵 구름 둘 + 재질 7, 눌러 붙인 뒤 **끊어질 때까지 당긴다**
  찰흙  빵 채운 구름 + 재질 5, 주물러 모양을 바꾸고 떼어 본다
"""
import argparse
import json
import os

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--wolf", required=True, help="wolf 채운 구름 h5")
ap.add_argument("--bread", required=True, help="빵 채운 구름 h5")
ap.add_argument("--out", required=True)
ap.add_argument("--beta", type=float, default=0.25,
                help="NACC 의 인장 항복면. 순수입자 실험에서 0.22~0.27 이 인열 문턱이었다")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def bbox(p):
    with h5py.File(p, "r") as h:
        x = np.array(h["x"])
    x = (x.T if x.shape[0] == 3 else x).astype(np.float64)
    x = x[np.isfinite(x).all(1)]
    return x.min(0), x.max(0), len(x)


BASE = dict(opacity_threshold=0.02, rotation_degree=[0.0], rotation_axis=[0],
            grid_lim=2.0, rpic_damping=0.0, grid_v_damping_scale=1.0,
            mpm_space_vertical_upward_axis=[0, 0, 1],
            mpm_space_viewpoint_center=[1, 1, 0.5], default_camera_index=-1,
            show_hint=False, init_azimuthm=265, init_elevation=30.0,
            init_radius=7.5, move_camera=False, delta_a=0, delta_e=0, delta_r=0.0)
BBOX = {"type": "bounding_box"}


def floor(z):
    return {"type": "surface_collider", "point": [1, 1, z], "normal": [0.0, 0.0, 1.0],
            "surface": "sticky", "friction": 0.0, "start_time": 0, "end_time": 1e3}


def grip(lo, hi, axis, vel, t0, t1):
    """상자 구역 하나를 속도로 붙잡는다 (GF 의 enforce_particle_translation)."""
    c = 0.5 * (lo + hi)
    s = 0.5 * (hi - lo)
    return {"type": "enforce_particle_translation",
            "point": [float(v) for v in c], "size": [float(v) for v in s],
            "velocity": [float(v) for v in vel],
            "start_time": float(t0), "end_time": float(t1)}


out = {}

# ------------------------------------------------------------------ 모래
lo, hi, n = bbox(a.wolf)
out["flow_wolf"] = dict(BASE, clouds=[{"h5": a.wolf}], material="sand",
                        E=5.0, nu=0.3, density=2000, friction_angle=30,
                        n_grid=200, substep_dt=2e-5, frame_dt=4e-2, frame_num=50,
                        flip_pic_ratio=0.0, g=[0.0, 0.0, -9.8],
                        init_velocity=[0.0, 0.0, 0.0],
                        boundary_conditions=[BBOX, floor(float(lo[2]) - 0.01)])

# ------------------------------------------------------------------ 인열
lo, hi, n = bbox(a.bread)
# 수평축(x,y) 중 긴 쪽으로 당긴다. z 를 고르면 두 덩이를 떼었을 때
# 격자 위아래로 삐져나가 CUDA 가 죽는다 (겪었다).
ax = int(np.argmax((hi - lo)[:2]))
L = hi[ax] - lo[ax]
pad = 0.12 * L
g_lo, g_hi = lo.copy(), hi.copy()
g_hi[ax] = lo[ax] + pad
h_lo, h_hi = lo.copy(), hi.copy()
h_lo[ax] = hi[ax] - pad
vL = np.zeros(3); vL[ax] = -0.15
vR = np.zeros(3); vR[ax] = 0.15
out["tear_bread"] = dict(
    BASE, material="watermelon", E=2e3, nu=0.38, density=1, friction_angle=45.0,
    beta=a.beta, xi=3.0, hardening=1.0, alpha_0=-0.04,
    n_grid=200, substep_dt=1e-4, frame_dt=2e-2, frame_num=200,
    flip_pic_ratio=0.7, auto_dt=True, g=[0.0, 0.0, 0.0],
    init_velocity=[0.0, 0.0, 0.0],
    clouds=[{"h5": a.bread,
             "notch": {"axis": int(ax), "at": float(0.5 * (lo[ax] + hi[ax])),
                       "depth": 0.35, "width": float(0.02 * L)}}],
    boundary_conditions=[BBOX,
                         grip(g_lo, g_hi, ax, vL, 0.0, 1e3),
                         grip(h_lo, h_hi, ax, vR, 0.0, 1e3)])

# ------------------------------------------------------------------ 접합
gap = 0.04 * L
half = 0.5 * (L + gap)
t_press, t_pull = 0.8, 4.0
tL = np.zeros(3); tL[ax] = -half
tR = np.zeros(3); tR[ax] = +half
# 떼어 놓은 뒤 전체를 격자 한가운데로 다시 민다
cen = 0.5 * ((lo + tL) + (hi + tR))
shift = np.array([1.0, 1.0, 1.0]) - cen
tL = tL + shift; tR = tR + shift
lo2, hi2 = lo + tL, hi + tR       # 두 덩이를 합친 바깥 상자
gl_lo, gl_hi = lo + tL, hi + tL
gl_hi[ax] = (lo + tL)[ax] + pad
gr_lo, gr_hi = lo + tR, hi + tR
gr_lo[ax] = (hi + tR)[ax] - pad
pL = np.zeros(3); pL[ax] = +0.25
pR = np.zeros(3); pR[ax] = -0.25
out["bond_bread"] = dict(
    BASE, material="watermelon", E=2e3, nu=0.38, density=1, friction_angle=45.0,
    beta=a.beta, xi=3.0, hardening=1.0, alpha_0=-0.04,
    n_grid=200, substep_dt=1e-4, frame_dt=2e-2, frame_num=250,
    flip_pic_ratio=0.7, auto_dt=True, g=[0.0, 0.0, 0.0],
    init_velocity=[0.0, 0.0, 0.0],
    clouds=[{"h5": a.bread, "translate": [float(v) for v in tL]},
            {"h5": a.bread, "translate": [float(v) for v in tR]}],
    boundary_conditions=[BBOX,
                         grip(gl_lo, gl_hi, ax, pL, 0.0, t_press),
                         grip(gr_lo, gr_hi, ax, pR, 0.0, t_press),
                         grip(gl_lo, gl_hi, ax, -pL, t_press, t_pull),
                         grip(gr_lo, gr_hi, ax, -pR, t_press, t_pull)])

# ------------------------------------------------------- 찰흙 주무르기
# 재질 5(plasticine, von Mises + 연화). 도구는 GF 가 이미 가진 `cuboid` --
# 격자 속도 구역을 시간 창마다 눌러 넣는다. 마지막에 도구를 떼고 가만히 둬서
# **모양이 안 돌아오는지**(소성) 본다.
lo, hi, n = bbox(a.bread)
c = 0.5 * (lo + hi)
sz = 0.5 * (hi - lo)


def press(axis, sign, speed, t0, t1, reach=0.45):
    """한 면에서 눌러 들어오는 판. 구역은 물체 바깥에서 시작해 안으로 겹친다."""
    pt = c.copy()
    pt[axis] = c[axis] + sign * (sz[axis] * (1.0 + reach))
    s2 = sz * 1.3
    s2[axis] = sz[axis] * reach
    v = [0.0, 0.0, 0.0]
    v[axis] = -sign * speed
    return {"type": "cuboid", "point": [float(x) for x in pt],
            "size": [float(x) for x in s2], "velocity": v,
            "start_time": float(t0), "end_time": float(t1), "reset": 0}


out["knead_bread"] = dict(
    BASE, clouds=[{"h5": a.bread}], material="plasticine",
    E=2e3, nu=0.3, density=1, yield_stress=8.0, softening=0.0, hardening=0.0,
    xi=0.0, n_grid=200, substep_dt=1e-4, frame_dt=2e-2, frame_num=260,
    flip_pic_ratio=0.0, auto_dt=True, g=[0.0, 0.0, 0.0],
    init_velocity=[0.0, 0.0, 0.0],
    boundary_conditions=[BBOX,
                         press(2, +1, 0.30, 0.0, 1.0),     # 위에서 누르고
                         press(0, +1, 0.30, 1.4, 2.4),     # 옆에서 누르고
                         press(1, -1, 0.30, 2.8, 3.8)])    # 뗀 뒤 가만히 둔다

# --------------------------------------------------- 모래에 여러 조작 가하기
# GF 의 `cuboid` 는 **고정된 구역**에 속도를 주는 것이라 움직이는 판이 아니다.
# 그래서 자리를 옮긴 상자 여러 개를 시간 창으로 이어 붙여 쓸고 지나가게 한다.
lo, hi, n = bbox(a.wolf)
c = 0.5 * (lo + hi)
sz = 0.5 * (hi - lo)


def region(pt, size, vel, t0, t1, reset=0):
    return {"type": "cuboid", "point": [float(x) for x in pt],
            "size": [float(x) for x in size], "velocity": [float(x) for x in vel],
            "start_time": float(t0), "end_time": float(t1), "reset": int(reset)}


def sweep(axis, t0, t1, speed, n_step=6, thick=0.12):
    """자리를 옮긴 상자들을 이어 붙여 판이 지나가는 것처럼 만든다."""
    out_bc = []
    dt = (t1 - t0) / n_step
    for i in range(n_step):
        pt = c.copy()
        pt[axis] = lo[axis] - thick * sz[axis] + (i / (n_step - 1)) * (hi[axis] - lo[axis])
        s2 = sz * 1.4
        s2[axis] = thick * sz[axis]
        v = [0.0, 0.0, 0.0]
        v[axis] = speed
        out_bc.append(region(pt, s2, v, t0 + i * dt, t0 + (i + 1) * dt))
    return out_bc


press_top = region([c[0], c[1], hi[2] + 0.3 * sz[2]],
                   [sz[0] * 1.3, sz[1] * 1.3, 0.35 * sz[2]],
                   [0.0, 0.0, -0.25], 1.2, 2.2)          # 위에서 다진다
push_side = region([lo[0] - 0.2 * sz[0], c[1], c[2]],
                   [0.3 * sz[0], sz[1] * 1.3, sz[2] * 1.3],
                   [0.25, 0.0, 0.0], 2.6, 3.6)           # 옆에서 민다

out["manip_wolf"] = dict(
    BASE, clouds=[{"h5": a.wolf}], material="sand",
    E=5.0, nu=0.3, density=2000, friction_angle=30,
    n_grid=200, substep_dt=2e-5, frame_dt=4e-2, frame_num=150,
    flip_pic_ratio=0.0, g=[0.0, 0.0, -9.8], init_velocity=[0.0, 0.0, 0.0],
    boundary_conditions=([BBOX, floor(float(lo[2]) - 0.01), press_top, push_side]
                         + sweep(1, 4.0, 5.2, 0.30)))     # 가로질러 쓴다

for k, v in out.items():
    json.dump(v, open(os.path.join(a.out, f"{k}.json"), "w"), indent=1)
    print(f"[저장] {k}.json  재질 {v['material']} beta {v.get('beta')} "
          f"구름 {len(v['clouds'])} 프레임 {v['frame_num']}")
print("WARP_SCENES_OK")
