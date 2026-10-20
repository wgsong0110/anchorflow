"""PG particle_impulse 와 같은 총운동량의 **균일** 초기 속도 (사용자 10-20 "둘 다 초기 속도 부여").

  v_init = Σ_{p∈자유} F·substep_dt·num_dt / Σ_{p∈자유} m_p
자유 = cuboid(reset) 고정 상자 밖 입자 (PG 는 상자 안 격자 속도를 0 으로 박으니 상자 안 입자는 어차피 멈춘다).
m_p = PG 그대로: 부피 dx³/(같은 칸 입자 수) (particle_filling.get_particle_volume, floor(x/dx)) x 밀도
      (additional_material_params 상자 안이면 그 밀도, 엄격 부등식 -- mpm_utils.apply_additional_params).
충격 상자 = |x0-point|<size (point·size 기본 [1,1,1], selection_add_impulse_on_particles).
cuboid 는 격자 노드 판정(|i·dx-point|<size)이지만 여기서는 입자 0 프레임 위치로 자유 입자를 가른다.

  python exe/ficus_v_init.py --h5 W/topo/pg_ficus_elastic/simulation_ply/sim_0000000000.h5 \
      --config W/PhysGaussian/config/ficus_config.json [--check cfg/ficus_v0_config.json]
"""
import argparse
import json

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--h5", required=True, help="PG 0 프레임 h5 (x (3,N))")
ap.add_argument("--config", required=True, help="PG 공식 config (particle_impulse 포함)")
ap.add_argument("--check", default="", help="enforce_particle_translation 속도가 이 값과 같은지 확인할 config")
a = ap.parse_args()
c = json.load(open(a.config))
x = np.array(h5py.File(a.h5, "r")["x"], dtype=np.float32).T
N = x.shape[0]
ng = int(c.get("n_grid", 50)); dx = np.float32(float(c.get("grid_lim", 2.0)) / ng)
vi = np.floor(x / dx).astype(np.int64)
fl = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
vol = dx ** 3 / np.bincount(fl, minlength=ng ** 3)[fl]
rho = np.full(N, float(c["density"]))
for r in c.get("additional_material_params") or []:
    q, s = np.array(r["point"]), np.array(r["size"])
    rho[((x > q - s) & (x < q + s)).all(1)] = float(r["density"])
m = vol * rho
fixed = np.zeros(N, bool)
P = np.zeros(3)
for b in c["boundary_conditions"]:
    if b["type"] == "cuboid":
        fixed |= (np.abs(x - np.array(b["point"])) < np.array(b["size"])).all(1)
free = ~fixed
for b in c["boundary_conditions"]:
    if b["type"] == "particle_impulse":
        assert float(b.get("start_time", 0.0)) == 0.0
        q, s = np.array(b.get("point", [1, 1, 1])), np.array(b.get("size", [1, 1, 1]))
        inb = (np.abs(x - q) < s).all(1)
        P += np.array(b["force"]) * float(c["substep_dt"]) * int(b.get("num_dt", 1)) * float((inb & free).sum())
v = P / m[free].sum()
dv = float(c["boundary_conditions"][-1]["force"][0]) * float(c["substep_dt"]) / m[free]
print(f"[입자] {N} 개, 고정 상자 안 {int(fixed.sum())}, 자유 {int(free.sum())}; 질량 합 전체 {m.sum():.6g} 자유 {m[free].sum():.6g}")
print(f"[충격 운동량] 자유 입자 Σ F·dt = {P.tolist()}")
print(f"[v_init] {v.tolist()}  (= 충격 속도의 자유 입자 질량 가중 평균; 충격 v0 x 범위 {dv.min():.4g} ~ {dv.max():.4g}, 질량비 {m[free].max() / m[free].min():.0f})")
if a.check:
    cc = json.load(open(a.check))
    vv = [b["velocity"] for b in cc["boundary_conditions"] if b["type"] == "enforce_particle_translation"]
    assert len(vv) == 1 and np.allclose(vv[0], v, rtol=1e-6, atol=1e-7), (vv, v)
    print(f"[확인] {a.check} 의 속도 {vv[0]} 와 일치")
