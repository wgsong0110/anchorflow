"""GaussianFluent 의 MPM 솔버를 **가우시안 없이** 타이치로 그대로 옮긴 것.

`exe/synth_mpm.py` 는 내가 쓴 MLS-MPM 이라 물성을 같게 줘도 GF 와 궤적이 갈린다.
같은 물성으로 같은 결과를 내려면 물성 말고도 다음이 같아야 한다 -- 실제로 이
넷이 전부 달랐다.

  1. 전달 방식      GF 는 FLIP/PIC 혼합(수박은 0.7), 내 것은 APIC
  2. 격자 힘 항     GF 는 B-스플라인 기울기 dweight, 내 것은 MLS 의 4/dx^2 dpos
  3. 입자 부피      GF 는 셀마다 세어서 dx^3/(셀 안 개수), 내 것은 간격^3
  4. 경계           GF 는 padding 3 칸의 한쪽 방향 속도 0

그래서 씬마다 맞추는 대신 솔버째로 옮겼다. config(json)와 초기 상태(h5)만 주면
어떤 GF 씬이든 같은 궤적이 나와야 한다.

  python exe/gf_mpm.py --config <GF config.json> --h5 <sim_0000000000.h5> --out DIR
"""
import argparse, json, os, time
import numpy as np
import h5py
import taichi as ti
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="GF 의 씬 config json")
ap.add_argument("--h5", required=True, help="초기 상태 h5 (x, v 를 읽는다)")
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=None, help="없으면 config 의 frame_num")
ap.add_argument("--stride", type=int, default=1, help="h5 입자 솎기 (검증용)")
ap.add_argument("--f64", action="store_true")
ap.add_argument("--flip", default="auto", choices=("auto", "on", "off"),
                help="auto 는 flip_pic_ratio>0 (gs_simulation.py 의 규칙). 씬 전용 "
                     "러너로 뽑은 궤적과 맞출 때는 on 을 쓴다")
ap.add_argument("--resume", action="store_true",
                help="out 의 state_last.h5 에서 이어 돌린다. 위치뿐 아니라 "
                     "F 와 logJp 까지 들고 있어야 궤적이 이어진다")
ap.add_argument("--auto_dt", action="store_true",
                help="GF 의 씬 러너처럼 substep_dt 를 CFL 로 다시 계산한다 "
                     "(gs_simulation_watermelon.py:416). config 값은 무시된다")
a = ap.parse_args()

cfg = json.load(open(a.config))
ti.init(arch=ti.gpu, default_fp=ti.f64 if a.f64 else ti.f32,
        device_memory_fraction=0.85)

# ------------------------------------------------------------------ 상수
# material_2_num (mpm_solver_warp.py:255). foam 이 3, snow 가 4, plasticine 이 5 다.
# 4 는 되돌림도 응력도 갈래가 없어 GF 에서 응력이 0 으로 남는다 -- 그대로 옮긴다.
MAT = {"jelly": 0, "metal": 1, "sand": 2, "foam": 3, "snow": 4,
       "plasticine": 5, "watermelon": 7}
material = MAT[cfg["material"]]
n_grid = int(cfg.get("n_grid", 100))
grid_lim = float(cfg.get("grid_lim", 2.0))
dx = grid_lim / n_grid
inv_dx = 1.0 / dx
E, nu = float(cfg["E"]), float(cfg["nu"])
mu0 = E / (2.0 * (1.0 + nu))
lam0 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
kappa0 = 2.0 * mu0 / 3.0 + lam0
_phi = float(cfg.get("friction_angle", 45.0))
_sf = np.sin(_phi / 180.0 * 3.14159265)
ALPHA = float(np.sqrt(2.0 / 3.0) * 2.0 * _sf / (3.0 - _sf))
M_CD = float(ALPHA * 3.0 / np.sqrt(2.0 / (6.0 - 3.0)))   # GF: alpha*dim/sqrt(2/(6-dim))
XI = float(cfg.get("xi", 0.0))
BETA = float(cfg.get("beta", 1.0))
HARDENING = float(cfg.get("hardening", 0.0))
YIELD0 = float(cfg.get("yield_stress", 0.0))
SOFTENING = float(cfg.get("softening", 0.1))
PLASTIC_VISC = float(cfg.get("plastic_viscosity", 0.0))
# mpm_solver_warp.py:109 -- particle_Jp 의 기본값은 0 이 아니라 -0.04 다.
# 0 으로 두면 p0 이 거의 0 이라 첫 스텝부터 전부 항복한다.
ALPHA0 = float(cfg.get("alpha_0", -0.04))
RPIC = float(cfg.get("rpic_damping", 0.0))
# 기본값은 1.1 이고, 1 보다 작을 때만 감쇠 커널이 돈다
GRID_DAMP = float(cfg.get("grid_v_damping_scale", 1.1))
# gs_simulation.py:429 -- 키가 없으면 0.7 로 보고 FLIP 을 켠다. 0 이면 APIC 이다.
# 다만 씬 전용 러너(gs_simulation_watermelon.py)는 flip_pic 인자를 넘기지 않아
# **비율과 무관하게 항상 FLIP** 이다. 그래서 --flip 으로 덮어쓸 수 있게 둔다.
FLIP = float(cfg.get("flip_pic_ratio", 0.7))
USE_FLIP = {"auto": FLIP > 0.0, "on": True, "off": False}[a.flip]
density = float(cfg["density"])
substep_dt = float(cfg["substep_dt"])
frame_dt = float(cfg["frame_dt"])
n_frames = a.frames if a.frames is not None else int(cfg.get("frame_num", 100))
if a.auto_dt:
    _c = np.sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * density))
    substep_dt = 0.6 * dx / _c
# GF 는 int() 로 버린다 -- 한 프레임이 frame_dt 보다 살짝 짧다. 그대로 따른다.
nsub = max(1, int(frame_dt / substep_dt))
G = np.array(cfg.get("g", [0.0, 0.0, -9.8]), np.float64)
# gs_simulation.py:374 가 config 에 없으면 [0,0,-6] 을 그대로 박아 넣는다
V0 = np.array(cfg.get("init_velocity", [0.0, 0.0, -6.0]), np.float64)

# ------------------------------------------------------------------ 초기 상태
with h5py.File(a.h5, "r") as h:
    X0 = np.array(h["x"]);  X0 = (X0.T if X0.shape[0] == 3 else X0).astype(np.float64)
    V_h5 = np.array(h["v"]) if "v" in h else None
X0 = X0[::a.stride]
ok = np.isfinite(X0).all(1)
X0 = X0[ok]
N = len(X0)
if V_h5 is not None:
    V_h5 = (V_h5.T if V_h5.shape[0] == 3 else V_h5).astype(np.float64)[::a.stride][ok]
    V_init = V_h5 if np.abs(V_h5).max() > 0 else np.tile(V0, (N, 1))
else:
    V_init = np.tile(V0, (N, 1))

# particle_filling.get_particle_volume 와 같은 정의: 셀마다 세고 dx^3/개수
cell = np.floor(X0 / dx).astype(np.int64)
cell = np.clip(cell, 0, n_grid - 1)
flat = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
cnt = np.bincount(flat, minlength=n_grid ** 3)
VOL = (dx ** 3) / cnt[flat].astype(np.float64)
if cfg["material"] == "sand":              # GF 는 모래만 평균 부피를 쓴다
    VOL = np.full(N, VOL.mean())
# additional_material_params: 상자 안 입자만 E/nu/density 를 갈아 끼운다
# (mpm_utils.py:886). GF 는 이 뒤에 질량과 mu/lam 을 다시 계산한다.
Ep = np.full(N, E); NUp = np.full(N, nu); DENp = np.full(N, density)
for prm in cfg.get("additional_material_params", []):
    _pt = np.array(prm["point"], np.float64); _sz = np.array(prm["size"], np.float64)
    _m = np.all((X0 > _pt - _sz) & (X0 < _pt + _sz), axis=1)
    Ep[_m] = prm["E"]; NUp[_m] = prm["nu"]; DENp[_m] = prm["density"]
    print(f"[구역 물성] {int(_m.sum())} 입자에 E={prm['E']} nu={prm['nu']} "
          f"rho={prm['density']}", flush=True)
MU = Ep / (2.0 * (1.0 + NUp))
LM = Ep * NUp / ((1.0 + NUp) * (1.0 - 2.0 * NUp))
KP = 2.0 * MU / 3.0 + LM
MASS = DENp * VOL
print(f"[초기] {N} 입자, 범위 {np.round(X0.min(0),3)}~{np.round(X0.max(0),3)}\n"
      f"       재질 {cfg['material']}({material}) 격자 {n_grid} dx {dx:.5f} "
      f"부피 중앙 {np.median(VOL):.3e} 질량합 {MASS.sum():.4f}\n"
      f"       dt {substep_dt} x {nsub} = 프레임 {frame_dt}, "
      f"{'FLIP '+str(FLIP) if USE_FLIP else 'APIC'}, v0 {V_init[0]}", flush=True)

# ------------------------------------------------------------------ 필드
rt = ti.f64 if a.f64 else ti.f32
x = ti.Vector.field(3, rt, N); v = ti.Vector.field(3, rt, N)
C = ti.Matrix.field(3, 3, rt, N)
F = ti.Matrix.field(3, 3, rt, N); Ftr = ti.Matrix.field(3, 3, rt, N)
St = ti.Matrix.field(3, 3, rt, N)
Jp = ti.field(rt, N); ys = ti.field(rt, N)
mu_p = ti.field(rt, N); lam_p = ti.field(rt, N); kap_p = ti.field(rt, N)
vol = ti.field(rt, N); mass = ti.field(rt, N)
alive = ti.field(ti.i32, N)
x0f = ti.Vector.field(3, rt, N)
gvin = ti.Vector.field(3, rt, (n_grid,) * 3)
gvout = ti.Vector.field(3, rt, (n_grid,) * 3)
gm = ti.field(rt, (n_grid,) * 3)

# ------------------------------------------------------------ 경계·구동 조건
# utils/decode_param.py:248 이 받는 일곱 가지를 전부 옮긴다. 하나라도 빠지면
# 그걸 쓰는 씬에서 궤적이 갈린다.
CFG_DT = float(cfg["substep_dt"])          # 임펄스 길이는 config 값으로 잰다
_gc, _pc = [], []
for bc in cfg.get("boundary_conditions", []):
    ty = bc["type"]
    t0 = float(bc.get("start_time", 0.0)); t1 = float(bc.get("end_time", 999.0))
    if ty == "bounding_box":
        _gc.append(dict(k=0, t0=0.0, t1=999.0))
    elif ty == "surface_collider":
        nv = np.array(bc["normal"], np.float64); nv = nv / np.linalg.norm(nv)
        _gc.append(dict(k=1, pt=bc["point"], n=nv, t0=t0, t1=t1,
                        sty={"sticky": 0, "slip": 1, "cut": 11}.get(bc["surface"], 2),
                        fr=float(bc.get("friction", 0.0))))
    elif ty == "cuboid":
        _gc.append(dict(k=2, pt=bc["point"], sz=bc["size"], vel=bc["velocity"],
                        t0=t0, t1=t1, rs=int(bc.get("reset", 0))))
    elif ty == "particle_impulse":
        _pc.append(dict(k=0, pt=bc.get("point", [1, 1, 1]),
                        sz=bc.get("size", [1, 1, 1]), vel=bc["force"],
                        t0=t0, t1=t0 + CFG_DT * int(bc.get("num_dt", 1))))
    elif ty == "enforce_particle_translation":
        _pc.append(dict(k=1, pt=bc["point"], sz=bc["size"], vel=bc["velocity"],
                        t0=t0, t1=t1))
    elif ty == "release_particles_sequentially":
        # mpm_solver_warp.py:1176 -- num_layers 인자를 무시하고 항상 50 개의
        # "속도 0 고정" 구역으로 펼친다. 그 상수까지 그대로 따른다.
        nl = 50
        nv = bc["normal"]
        pt = [0.0, 0.0, 0.0]; sz = [0.0, 0.0, 0.0]; ax = -1
        for i in range(3):
            if nv[i] == 0:
                pt[i] = 1.0; sz[i] = 1.0
            else:
                ax = i; pt[i] = float(bc["end_position"])
        half = abs(bc["start_position"] - bc["end_position"]) / nl
        for i in range(nl):
            sz2 = list(sz); sz2[ax] = half * (nl - i)
            _pc.append(dict(k=1, pt=list(pt), sz=sz2, vel=[0.0, 0.0, 0.0],
                            t0=t0, t1=t1 / nl * (i + 1)))
    elif ty == "enforce_particle_velocity_rotation":
        nv = np.array(bc["normal"], np.float64); nv = nv / np.linalg.norm(nv)
        h1 = np.array([1.0, 1.0, 1.0])
        if abs(float(nv @ h1)) < 0.01:
            h1 = np.array([0.72, 0.37, -0.67])
        h1 = h1 - float(h1 @ nv) * nv; h1 = h1 / np.linalg.norm(h1)
        h2 = np.cross(h1, nv)
        _pc.append(dict(k=2, pt=bc["point"], n=nv, h1=h1, h2=h2,
                        hr=bc["half_height_and_radius"],
                        rs=float(bc["rotation_scale"]),
                        ts=float(bc["translation_scale"]), t0=t0, t1=t1))
    else:
        raise TypeError(f"모르는 경계 종류: {ty}")

NGC, NPC = max(1, len(_gc)), max(1, len(_pc))
gc_k = ti.field(ti.i32, NGC); gc_sty = ti.field(ti.i32, NGC)
gc_rs = ti.field(ti.i32, NGC)
gc_pt = ti.Vector.field(3, rt, NGC); gc_n = ti.Vector.field(3, rt, NGC)
gc_sz = ti.Vector.field(3, rt, NGC); gc_vel = ti.Vector.field(3, rt, NGC)
gc_fr = ti.field(rt, NGC); gc_t0 = ti.field(rt, NGC); gc_t1 = ti.field(rt, NGC)
pc_k = ti.field(ti.i32, NPC)
pc_pt = ti.Vector.field(3, rt, NPC); pc_sz = ti.Vector.field(3, rt, NPC)
pc_vel = ti.Vector.field(3, rt, NPC); pc_n = ti.Vector.field(3, rt, NPC)
pc_h1 = ti.Vector.field(3, rt, NPC); pc_h2 = ti.Vector.field(3, rt, NPC)
pc_hr = ti.Vector.field(2, rt, NPC)
pc_rs = ti.field(rt, NPC); pc_ts = ti.field(rt, NPC)
pc_t0 = ti.field(rt, NPC); pc_t1 = ti.field(rt, NPC)


def _pack(items, key, dim, default=0.0):
    out = np.full((max(1, len(items)), dim), default, np.float64)
    for i, d in enumerate(items):
        if key in d:
            out[i] = np.array(d[key], np.float64).reshape(-1)[:dim]
    return out


def _packs(items, key, default=0.0):
    out = np.full(max(1, len(items)), default, np.float64)
    for i, d in enumerate(items):
        if key in d:
            out[i] = float(d[key])
    return out


gc_k.from_numpy(_packs(_gc, "k", -1).astype(np.int32))
gc_sty.from_numpy(_packs(_gc, "sty", 0).astype(np.int32))
gc_rs.from_numpy(_packs(_gc, "rs", 0).astype(np.int32))
gc_pt.from_numpy(_pack(_gc, "pt", 3)); gc_n.from_numpy(_pack(_gc, "n", 3))
gc_sz.from_numpy(_pack(_gc, "sz", 3)); gc_vel.from_numpy(_pack(_gc, "vel", 3))
gc_fr.from_numpy(_packs(_gc, "fr")); gc_t0.from_numpy(_packs(_gc, "t0"))
gc_t1.from_numpy(_packs(_gc, "t1", 999.0))
pc_k.from_numpy(_packs(_pc, "k", -1).astype(np.int32))
pc_pt.from_numpy(_pack(_pc, "pt", 3)); pc_sz.from_numpy(_pack(_pc, "sz", 3))
pc_vel.from_numpy(_pack(_pc, "vel", 3)); pc_n.from_numpy(_pack(_pc, "n", 3))
pc_h1.from_numpy(_pack(_pc, "h1", 3)); pc_h2.from_numpy(_pack(_pc, "h2", 3))
pc_hr.from_numpy(_pack(_pc, "hr", 2))
pc_rs.from_numpy(_packs(_pc, "rs")); pc_ts.from_numpy(_packs(_pc, "ts"))
pc_t0.from_numpy(_packs(_pc, "t0")); pc_t1.from_numpy(_packs(_pc, "t1", 999.0))
NG, NP = len(_gc), len(_pc)
print(f"[경계] 격자 {NG} 개 {[d['k'] for d in _gc]}, 입자 {NP} 개 "
      f"{sorted(set(d['k'] for d in _pc))}", flush=True)

GRAV = ti.Vector(list(G))


@ti.kernel
def init(X: ti.types.ndarray(), V: ti.types.ndarray(),
         VO: ti.types.ndarray(), MA: ti.types.ndarray(),
         MU: ti.types.ndarray(), LM: ti.types.ndarray(),
         KP: ti.types.ndarray()):
    for p in range(N):
        for d in ti.static(range(3)):
            x[p][d] = ti.cast(X[p, d], rt); v[p][d] = ti.cast(V[p, d], rt)
            x0f[p][d] = ti.cast(X[p, d], rt)
        F[p] = ti.Matrix.identity(rt, 3); Ftr[p] = ti.Matrix.identity(rt, 3)
        C[p] = ti.Matrix.zero(rt, 3, 3); St[p] = ti.Matrix.zero(rt, 3, 3)
        Jp[p] = ALPHA0; ys[p] = YIELD0
        mu_p[p] = ti.cast(MU[p], rt); lam_p[p] = ti.cast(LM[p], rt)
        kap_p[p] = ti.cast(KP[p], rt)
        vol[p] = ti.cast(VO[p], rt); mass[p] = ti.cast(MA[p], rt); alive[p] = 1


@ti.func
def _wts(xp):
    gp = xp * inv_dx
    base = ti.cast(gp - 0.5, ti.i32)
    fx = gp - ti.cast(base, rt)
    wa = 1.5 - fx; wb = fx - 1.0; wc = fx - 0.5
    w = ti.Matrix.zero(rt, 3, 3)
    for d in ti.static(range(3)):
        w[0, d] = 0.5 * wa[d] * wa[d]
        w[1, d] = 0.75 - wb[d] * wb[d]
        w[2, d] = 0.5 * wc[d] * wc[d]
    dw = ti.Matrix.zero(rt, 3, 3)
    for d in ti.static(range(3)):
        dw[0, d] = fx[d] - 1.5
        dw[1, d] = -2.0 * (fx[d] - 1.0)
        dw[2, d] = fx[d] - 0.5
    return base, fx, w, dw


@ti.kernel
def zero_grid():
    for I in ti.grouped(gm):
        gm[I] = 0.0
        gvin[I] = ti.Vector.zero(rt, 3); gvout[I] = ti.Vector.zero(rt, 3)


@ti.kernel
def p2g(dt: rt):
    for p in range(N):
        if alive[p] == 0:
            continue
        base, fx, w, dw = _wts(x[p])
        stress = St[p]
        Cp = C[p]
        if ti.static(not USE_FLIP):
            Cp = (1.0 - RPIC) * Cp + RPIC / 2.0 * (Cp - Cp.transpose())
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            ix, iy, iz = base[0] + i, base[1] + j, base[2] + k
            if 0 <= ix < n_grid and 0 <= iy < n_grid and 0 <= iz < n_grid:
                wt = w[i, 0] * w[j, 1] * w[k, 2]
                dwt = ti.Vector([dw[i, 0] * w[j, 1] * w[k, 2],
                                 w[i, 0] * dw[j, 1] * w[k, 2],
                                 w[i, 0] * w[j, 1] * dw[k, 2]]) * inv_dx
                ef = -vol[p] * (stress @ dwt)
                if ti.static(USE_FLIP):
                    gvin[ix, iy, iz] += wt * mass[p] * v[p]
                    gvout[ix, iy, iz] += wt * dt * ef
                else:
                    dpos = (ti.Vector([float(i), float(j), float(k)]) - fx) * dx
                    gvin[ix, iy, iz] += (wt * mass[p] * (v[p] + Cp @ dpos)
                                         + dt * ef)
                gm[ix, iy, iz] += wt * mass[p]


@ti.func
def _gpos(I):
    return ti.Vector([ti.cast(I[0], rt), ti.cast(I[1], rt),
                      ti.cast(I[2], rt)]) * dx


@ti.kernel
def grid_op(dt: rt, t: rt):
    for I in ti.grouped(gm):
        if gm[I] > 1e-15:
            vo = (gvin[I] + gvout[I]) / gm[I] + dt * GRAV
            if ti.static(GRID_DAMP < 1.0):
                vo = vo * GRID_DAMP
            gvout[I] = vo
    for I in ti.grouped(gm):
        for c in range(NG):
            k = gc_k[c]
            inwin = (gc_t0[c] <= t) and (t < gc_t1[c])
            if k == 0:
                # add_bounding_box: padding 3 칸, 안으로 들어오는 성분만 0
                vo = gvout[I]
                for d in ti.static(range(3)):
                    if I[d] < 3 and vo[d] < 0:
                        vo[d] = 0.0
                    if I[d] >= n_grid - 3 and vo[d] > 0:
                        vo[d] = 0.0
                gvout[I] = vo
            elif k == 1 and inwin:
                off = _gpos(I) - gc_pt[c]
                if off.dot(gc_n[c]) < 0.0:
                    if gc_sty[c] == 11:
                        # cut: z 창 밖이면 정지, 안이면 y 를 죽이고 0.3 배
                        zz = ti.cast(I[2], rt) * dx
                        if zz < 0.4 or zz > 0.53:
                            gvout[I] = ti.Vector.zero(rt, 3)
                        else:
                            vi = gvout[I]
                            gvout[I] = ti.Vector(
                                [vi[0], ti.cast(0.0, rt), vi[2]]) * 0.3
                    else:
                        # sticky(0) 뿐 아니라 slip(1)·마찰(2) 갈래도 GF 는
                        # 마지막 줄에서 0 으로 덮어쓴다 (solver:781). 그대로 둔다.
                        gvout[I] = ti.Vector.zero(rt, 3)
            elif k == 2:
                # set_velocity_on_cuboid
                if inwin:
                    off = _gpos(I) - gc_pt[c]
                    if (ti.abs(off[0]) < gc_sz[c][0]
                            and ti.abs(off[1]) < gc_sz[c][1]
                            and ti.abs(off[2]) < gc_sz[c][2]):
                        gvout[I] = gc_vel[c]
                elif gc_rs[c] == 1 and t < gc_t1[c] + 15.0 * dt:
                    # reset 은 상자 밖에도 걸린다 -- 원본이 그렇다
                    gvout[I] = ti.Vector.zero(rt, 3)


@ti.kernel
def particle_bc(dt: rt, t: rt):
    # pre_p2g_operations(임펄스) 와 particle_velocity_modifiers 를 한 번에.
    # 어느 입자가 대상인지는 **초기 위치**로 한 번 정해지고 바뀌지 않는다.
    for p in range(N):
        if alive[p] == 0:
            continue
        for c in range(NP):
            if not ((pc_t0[c] <= t) and (t < pc_t1[c])):
                continue
            k = pc_k[c]
            off0 = x0f[p] - pc_pt[c]
            if k == 0 or k == 1:
                if (ti.abs(off0[0]) < pc_sz[c][0]
                        and ti.abs(off0[1]) < pc_sz[c][1]
                        and ti.abs(off0[2]) < pc_sz[c][2]):
                    if k == 0:
                        v[p] = v[p] + (pc_vel[c] / mass[p]) * dt
                    else:
                        v[p] = pc_vel[c]
            else:
                nv = pc_n[c]
                vd = ti.abs(off0.dot(nv))
                hd0 = (off0 - off0.dot(nv) * nv).norm()
                if vd < pc_hr[c][0] and hd0 < pc_hr[c][1]:
                    off = x[p] - pc_pt[c]
                    hd = (off - off.dot(nv) * nv).norm()
                    th = ti.acos(off.dot(pc_h1[c]) / hd)
                    if off.dot(pc_h2[c]) <= 0:
                        th = -th
                    v[p] = (-hd * ti.sin(th) * pc_rs[c] * pc_h1[c]
                            + hd * ti.cos(th) * pc_rs[c] * pc_h2[c]
                            + pc_ts[c] * nv)


@ti.kernel
def g2p(dt: rt, flip: rt):
    for p in range(N):
        if alive[p] == 0:
            continue
        base, fx, w, dw = _wts(x[p])
        nv = ti.Vector.zero(rt, 3); ov = ti.Vector.zero(rt, 3)
        nC = ti.Matrix.zero(rt, 3, 3); nF = ti.Matrix.zero(rt, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            ix, iy, iz = base[0] + i, base[1] + j, base[2] + k
            if 0 <= ix < n_grid and 0 <= iy < n_grid and 0 <= iz < n_grid:
                wt = w[i, 0] * w[j, 1] * w[k, 2]
                gv = gvout[ix, iy, iz]
                nv += gv * wt
                dwt = ti.Vector([dw[i, 0] * w[j, 1] * w[k, 2],
                                 w[i, 0] * dw[j, 1] * w[k, 2],
                                 w[i, 0] * w[j, 1] * dw[k, 2]]) * inv_dx
                nF += gv.outer_product(dwt)
                if ti.static(USE_FLIP):
                    if gm[ix, iy, iz] > 1e-15:
                        ov += gvin[ix, iy, iz] * wt / gm[ix, iy, iz]
                else:
                    dpos = ti.Vector([float(i), float(j), float(k)]) - fx
                    nC += gv.outer_product(dpos) * (wt * inv_dx * 4.0)
        if ti.static(USE_FLIP):
            v[p] = v[p] * flip + nv - flip * ov
        else:
            v[p] = nv; C[p] = nC
        x[p] = x[p] + dt * nv
        Ftr[p] = (ti.Matrix.identity(rt, 3) + nF * dt) @ F[p]


@ti.kernel
def stress_kernel(dt: rt):
    for p in range(N):
        if alive[p] == 0:
            continue
        Fn = Ftr[p]
        # ---------------- return mapping ----------------
        if ti.static(material == 7):
            U, sg, V = ti.svd(Fn, rt)
            s0 = ti.max(sg[0, 0], 0.0); s1 = ti.max(sg[1, 1], 0.0)
            s2 = ti.max(sg[2, 2], 0.0)
            logJp = Jp[p]
            z = XI * ti.max(-logJp, 0.0)
            p0 = kap_p[p] * (1e-5 + 0.5 * (ti.exp(z) - ti.exp(-z)))   # sinh
            Jd = s0 * s1 * s2
            b0, b1, b2 = s0 * s0, s1 * s1, s2 * s2
            bm = (b0 + b1 + b2) / 3.0
            jp = ti.pow(Jd, -2.0 / 3.0)
            sh0 = mu_p[p] * jp * (b0 - bm)
            sh1 = mu_p[p] * jp * (b1 - bm)
            sh2 = mu_p[p] * jp * (b2 - bm)
            p_tr = -(kap_p[p] / 2.0 * (Jd - 1.0 / Jd)) * Jd
            ysc = (6.0 - 3.0) / 2.0 * (1.0 + 2.0 * BETA)
            yph = M_CD * M_CD * (p_tr + BETA * p0) * (p_tr - p0)
            ssq = sh0 * sh0 + sh1 * sh1 + sh2 * sh2
            y = ysc * ssq + yph
            f0, f1, f2 = s0, s1, s2
            lj = logJp
            p_min = BETA * p0
            if p_tr > p0:
                Je = ti.sqrt(-2.0 * p0 / kap_p[p] + 1.0)
                f0 = ti.pow(Je, 1.0 / 3.0); f1 = f0; f2 = f0
                if HARDENING > 0.5:
                    lj = logJp + ti.log(Jd / Je)
            elif p_tr < -p_min:
                Je = ti.sqrt(2.0 * p_min / kap_p[p] + 1.0)
                f0 = ti.pow(Je, 1.0 / 3.0); f1 = f0; f2 = f0
                if HARDENING > 0.5:
                    lj = logJp + ti.log(Jd / Je)
            elif y >= 1e-4:
                sn = ti.max(ti.sqrt(ssq), 1e-10)
                sf = ti.sqrt(-yph / ysc)
                sc = ti.pow(Jd, 2.0 / 3.0) / mu_p[p] * sf / sn
                f0 = ti.sqrt(sc * sh0 + bm)
                f1 = ti.sqrt(sc * sh1 + bm)
                f2 = ti.sqrt(sc * sh2 + bm)
                # 항복면 위의 경화. 이게 logJp 를 키워 p0 를 끌어내리고, 그래서
                # 재료가 물러진다 -- 수박이 깨지는 것은 이 갈래가 만든다.
                if (HARDENING > 0.5 and p0 > 1e-4 and p_tr < p0 - 1e-4
                        and p_tr > 1e-4 - p_min):
                    pc = (p0 - p_min) * 0.5
                    qt = ti.sqrt((6.0 - 3.0) / 2.0) * sn
                    dp = pc - p_tr
                    dq = -qt
                    dn = ti.max(ti.sqrt(dp * dp + dq * dq), 1e-10)
                    dp = dp / dn
                    Cq = M_CD * M_CD * (pc + BETA * p0) * (pc - p0)
                    Bq = M_CD * M_CD * dp * (2.0 * pc - p0 + BETA * p0)
                    Aq = (M_CD * M_CD * dp * dp
                          + (1.0 + 2.0 * BETA) * dq * dq)
                    disc = Bq * Bq - 4.0 * Aq * Cq
                    l1 = (-Bq + ti.sqrt(disc)) / (2.0 * Aq)
                    l2 = (-Bq - ti.sqrt(disc)) / (2.0 * Aq)
                    p1 = pc + l1 * dp
                    p2 = pc + l2 * dp
                    pf = p2
                    if (p_tr - pc) * (p1 - pc) > 0.0:
                        pf = p1
                    jef = ti.sqrt(ti.abs(-2.0 * pf / kap_p[p] + 1.0))
                    if jef > 1e-4:
                        lj = logJp + ti.log(Jd / jef)
            Jp[p] = lj
            sg[0, 0] = f0; sg[1, 1] = f1; sg[2, 2] = f2
            F[p] = U @ sg @ V.transpose()
        elif ti.static(material == 2):
            U, sg, V = ti.svd(Fn, rt)
            e = ti.Vector([ti.log(ti.max(ti.abs(sg[0, 0]), 1e-14)),
                           ti.log(ti.max(ti.abs(sg[1, 1]), 1e-14)),
                           ti.log(ti.max(ti.abs(sg[2, 2]), 1e-14))])
            tr = e[0] + e[1] + e[2]
            eh = e - ti.Vector([tr / 3.0] * 3)
            ehn = eh.norm()
            dg = ehn + (3.0 * lam_p[p] + 2.0 * mu_p[p]) / (2.0 * mu_p[p]) * tr * ALPHA
            Fe = Fn
            if dg > 0 and tr > 0:
                Fe = U @ V.transpose()
            elif dg > 0 and tr <= 0:
                H = e - eh * (dg / ehn)
                sn = ti.Matrix.zero(rt, 3, 3)
                for d in ti.static(range(3)):
                    sn[d, d] = ti.exp(H[d])
                Fe = U @ sn @ V.transpose()
            F[p] = Fe
        elif ti.static(material == 1 or material == 5):
            U, sg, V = ti.svd(Fn, rt)
            s = ti.Vector([ti.max(sg[0, 0], 0.01), ti.max(sg[1, 1], 0.01),
                           ti.max(sg[2, 2], 0.01)])
            e = ti.Vector([ti.log(s[0]), ti.log(s[1]), ti.log(s[2])])
            tm = (e[0] + e[1] + e[2]) / 3.0
            tau = 2.0 * mu_p[p] * e + lam_p[p] * (e[0] + e[1] + e[2]) * ti.Vector([1.0] * 3)
            st = tau[0] + tau[1] + tau[2]
            cond = tau - ti.Vector([st / 3.0] * 3)
            Fe = Fn
            if cond.norm() > ys[p] and ys[p] > 0:
                eh = e - ti.Vector([tm] * 3)
                ehn = eh.norm() + 1e-6
                dg = ehn - ys[p] / (2.0 * mu_p[p])
                e = e - (dg / ehn) * eh
                if ti.static(material == 5):
                    ys[p] = ys[p] - SOFTENING * ((dg / ehn) * eh).norm()
                    if ys[p] <= 0:
                        mu_p[p] = 0.0; lam_p[p] = 0.0
                sn = ti.Matrix.zero(rt, 3, 3)
                for d in ti.static(range(3)):
                    sn[d, d] = ti.exp(e[d])
                Fe = U @ sn @ V.transpose()
                if HARDENING == 1.0:
                    ys[p] = ys[p] + 2.0 * mu_p[p] * XI * dg
            F[p] = Fe
        elif ti.static(material == 3):
            U, sg, V = ti.svd(Fn, rt)
            s = ti.Vector([ti.max(sg[0, 0], 0.01), ti.max(sg[1, 1], 0.01),
                           ti.max(sg[2, 2], 0.01)])
            bt = ti.Vector([s[0] * s[0], s[1] * s[1], s[2] * s[2]])
            e = ti.Vector([ti.log(s[0]), ti.log(s[1]), ti.log(s[2])])
            tr = e[0] + e[1] + e[2]
            eh = e - ti.Vector([tr / 3.0] * 3)
            s_tr = 2.0 * mu_p[p] * eh
            sn_ = s_tr.norm()
            yv = sn_ - ti.sqrt(2.0 / 3.0) * ys[p]
            Fe = Fn
            if yv > 0:
                mh = mu_p[p] * (bt[0] + bt[1] + bt[2]) / 3.0
                snn = sn_ - yv / (1.0 + PLASTIC_VISC / (2.0 * mh * dt))
                s_new = (snn / sn_) * s_tr
                en = 1.0 / (2.0 * mu_p[p]) * s_new + ti.Vector([tr / 3.0] * 3)
                sm = ti.Matrix.zero(rt, 3, 3)
                for d in ti.static(range(3)):
                    sm[d, d] = ti.exp(en[d])
                Fe = U @ sm @ V.transpose()
            F[p] = Fe
        else:
            F[p] = Fn
        # ---------------- stress ----------------
        Fc = F[p]
        J = Fc.determinant()
        U, sg, V = ti.svd(Fc, rt)
        stress = ti.Matrix.zero(rt, 3, 3)
        if ti.static(material == 7):
            B = Fc @ Fc.transpose()
            btr = B[0, 0] + B[1, 1] + B[2, 2]
            devB = B - ti.Matrix.identity(rt, 3) * (btr / 3.0)
            prime = kap_p[p] / 2.0 * (J - 1.0 / J)
            stress = (mu_p[p] * ti.pow(J, -2.0 / 3.0) * devB
                      + ti.Matrix.identity(rt, 3) * (J * prime))
        elif ti.static(material == 0 or material == 5):
            R = U @ V.transpose()
            stress = (2.0 * mu_p[p] * (Fc - R) @ Fc.transpose()
                      + ti.Matrix.identity(rt, 3) * lam_p[p] * J * (J - 1.0))
        elif ti.static(material == 1 or material == 3):
            s = ti.Vector([ti.max(sg[0, 0], 0.01), ti.max(sg[1, 1], 0.01),
                           ti.max(sg[2, 2], 0.01)])
            ls = ti.log(s[0]) + ti.log(s[1]) + ti.log(s[2])
            tau = ti.Matrix.zero(rt, 3, 3)
            for d in ti.static(range(3)):
                tau[d, d] = 2.0 * mu_p[p] * ti.log(s[d]) + lam_p[p] * ls
            stress = U @ tau @ V.transpose() @ Fc.transpose()
        elif ti.static(material == 2):
            ls = ti.log(sg[0, 0]) + ti.log(sg[1, 1]) + ti.log(sg[2, 2])
            ctr = ti.Matrix.zero(rt, 3, 3)
            for d in ti.static(range(3)):
                ctr[d, d] = (2.0 * mu_p[p] * ti.log(sg[d, d]) / sg[d, d]
                             + lam_p[p] * ls / sg[d, d])
            stress = U @ ctr @ V.transpose() @ Fc.transpose()
        St[p] = (stress + stress.transpose()) / 2.0


@ti.kernel
def quarantine(lo: rt, hi: rt):
    for p in range(N):
        if alive[p] == 1:
            bad = 0
            for d in ti.static(range(3)):
                if not (lo < x[p][d] < hi):
                    bad = 1
                if not (ti.abs(v[p][d]) < 1e5):
                    bad = 1
            if bad == 1:
                alive[p] = 0


init(X0, V_init, VOL, MASS, MU, LM, KP)
os.makedirs(a.out, exist_ok=True)


def dump(f):
    with h5py.File(os.path.join(a.out, f"sim_{f:010d}.h5"), "w") as h:
        h.create_dataset("x", data=x.to_numpy().T.astype(np.float32))
        h.create_dataset("v", data=v.to_numpy().T.astype(np.float32))
        h.create_dataset("time", data=np.array([[f * frame_dt]]))


STATE = os.path.join(a.out, "state_last.h5")


def save_state(f, t):
    tmp = STATE + ".tmp"
    with h5py.File(tmp, "w") as h:
        h.create_dataset("frame", data=np.array([f]))
        h.create_dataset("t", data=np.array([t]))
        for k, fl in (("x", x), ("v", v), ("F", F), ("Ftr", Ftr), ("C", C),
                      ("Jp", Jp), ("ys", ys), ("mu", mu_p), ("lam", lam_p),
                      ("alive", alive)):
            h.create_dataset(k, data=fl.to_numpy())
    os.replace(tmp, STATE)


@ti.kernel
def _load(X: ti.types.ndarray(), V: ti.types.ndarray(), FF: ti.types.ndarray(),
          FT: ti.types.ndarray(), CC: ti.types.ndarray(), J: ti.types.ndarray(),
          Y: ti.types.ndarray(), M: ti.types.ndarray(), L: ti.types.ndarray(),
          A: ti.types.ndarray()):
    for p in range(N):
        for d in ti.static(range(3)):
            x[p][d] = ti.cast(X[p, d], rt); v[p][d] = ti.cast(V[p, d], rt)
            x0f[p][d] = ti.cast(X[p, d], rt)
            for e in ti.static(range(3)):
                F[p][d, e] = ti.cast(FF[p, d, e], rt)
                Ftr[p][d, e] = ti.cast(FT[p, d, e], rt)
                C[p][d, e] = ti.cast(CC[p, d, e], rt)
        Jp[p] = ti.cast(J[p], rt); ys[p] = ti.cast(Y[p], rt)
        mu_p[p] = ti.cast(M[p], rt); lam_p[p] = ti.cast(L[p], rt)
        alive[p] = A[p]


f0 = 0
t = 0.0
if a.resume and os.path.exists(STATE):
    with h5py.File(STATE, "r") as h:
        f0 = int(np.array(h["frame"])[0]); t = float(np.array(h["t"])[0])
        _load(np.array(h["x"]), np.array(h["v"]), np.array(h["F"]),
              np.array(h["Ftr"]), np.array(h["C"]), np.array(h["Jp"]),
              np.array(h["ys"]), np.array(h["mu"]), np.array(h["lam"]),
              np.array(h["alive"]))
    print(f"[이어감] {STATE} 의 프레임 {f0} (t={t:.4f}) 에서", flush=True)
else:
    dump(0)
t0 = time.time()
for f in tqdm(range(f0 + 1, n_frames + 1), desc="frames", ncols=78):
    for _ in range(nsub):
        quarantine(-0.5 * grid_lim, 1.5 * grid_lim)
        if NP:
            particle_bc(substep_dt, t)
        stress_kernel(substep_dt)
        zero_grid()
        p2g(substep_dt)
        grid_op(substep_dt, t)
        g2p(substep_dt, FLIP)
        t += substep_dt
    dump(f); save_state(f, t)
    if f % 5 == 0 or f == 1:
        xn = x.to_numpy(); al = alive.to_numpy()
        jp = Jp.to_numpy()[al == 1]
        print(f"  f{f:4d}  살아있음 {int(al.sum())}/{N}  "
              f"logJp 중앙 {np.median(jp):+.4f} p1 {np.percentile(jp,1):+.4f}  "
              f"x[{xn[al==1].min():.3f},{xn[al==1].max():.3f}]  "
              f"{time.time()-t0:.0f}s", flush=True)
print(f"[저장] {a.out}  {n_frames+1} 프레임  {time.time()-t0:.0f}s", flush=True)
