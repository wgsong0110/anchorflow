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

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="GF 의 씬 config json")
ap.add_argument("--h5", required=True, help="초기 상태 h5 (x, v 를 읽는다)")
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=None, help="없으면 config 의 frame_num")
ap.add_argument("--stride", type=int, default=1, help="h5 입자 솎기 (검증용)")
ap.add_argument("--f64", action="store_true")
ap.add_argument("--auto_dt", action="store_true",
                help="GF 의 씬 러너처럼 substep_dt 를 CFL 로 다시 계산한다 "
                     "(gs_simulation_watermelon.py:416). config 값은 무시된다")
a = ap.parse_args()

cfg = json.load(open(a.config))
ti.init(arch=ti.gpu, default_fp=ti.f64 if a.f64 else ti.f32,
        device_memory_fraction=0.85)

# ------------------------------------------------------------------ 상수
MAT = {"jelly": 0, "metal": 1, "sand": 2, "visplas": 3, "foam": 4,
       "snow": 5, "plasticine": 5, "watermelon": 7}
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
XI = float(cfg.get("xi", 3.0))
BETA = float(cfg.get("beta", 1.0))
HARDENING = float(cfg.get("hardening", 0.0))
YIELD0 = float(cfg.get("yield_stress", 0.0))
SOFTENING = float(cfg.get("softening", 0.1))
PLASTIC_VISC = float(cfg.get("plastic_viscosity", 0.0))
RPIC = float(cfg.get("rpic_damping", 0.0))
GRID_DAMP = float(cfg.get("grid_v_damping_scale", 1.0))
FLIP = float(cfg.get("flip_pic_ratio", 0.0))
USE_FLIP = "flip_pic_ratio" in cfg and FLIP > 0.0
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
MASS = density * VOL
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
mu_p = ti.field(rt, N); lam_p = ti.field(rt, N)
vol = ti.field(rt, N); mass = ti.field(rt, N)
alive = ti.field(ti.i32, N)
gvin = ti.Vector.field(3, rt, (n_grid,) * 3)
gvout = ti.Vector.field(3, rt, (n_grid,) * 3)
gm = ti.field(rt, (n_grid,) * 3)

# 경계: bounding_box 는 무조건, 평면 충돌자는 config 에서 읽는다
SURF = {"sticky": 0, "slip": 1, "cut": 2}
planes = [bc for bc in cfg.get("boundary_conditions", [])
          if bc["type"] == "surface_collider"]
NPL = max(1, len(planes))
pl_pt = ti.Vector.field(3, rt, NPL); pl_n = ti.Vector.field(3, rt, NPL)
pl_ty = ti.field(ti.i32, NPL); pl_fr = ti.field(rt, NPL)
pl_t0 = ti.field(rt, NPL); pl_t1 = ti.field(rt, NPL)
HAS_BBOX = any(bc["type"] == "bounding_box" for bc in cfg.get("boundary_conditions", []))
_pp = np.zeros((NPL, 3)); _pn = np.zeros((NPL, 3)); _pn[:, 2] = 1.0
_pt = np.zeros(NPL, np.int32); _pf = np.zeros(NPL); _p0 = np.zeros(NPL); _p1 = np.zeros(NPL)
for i, bc in enumerate(planes):
    nrm = np.array(bc["normal"], np.float64); nrm = nrm / np.linalg.norm(nrm)
    _pp[i] = bc["point"]; _pn[i] = nrm
    _pt[i] = SURF.get(bc.get("surface", "sticky"), 0)
    _pf[i] = bc.get("friction", 0.0)
    _p0[i] = bc.get("start_time", 0.0); _p1[i] = bc.get("end_time", 1e9)
pl_pt.from_numpy(_pp); pl_n.from_numpy(_pn); pl_ty.from_numpy(_pt)
pl_fr.from_numpy(_pf); pl_t0.from_numpy(_p0); pl_t1.from_numpy(_p1)
NPLANE = len(planes)

GRAV = ti.Vector(list(G))


@ti.kernel
def init(X: ti.types.ndarray(), V: ti.types.ndarray(),
         VO: ti.types.ndarray(), MA: ti.types.ndarray()):
    for p in range(N):
        for d in ti.static(range(3)):
            x[p][d] = X[p, d]; v[p][d] = V[p, d]
        F[p] = ti.Matrix.identity(rt, 3); Ftr[p] = ti.Matrix.identity(rt, 3)
        C[p] = ti.Matrix.zero(rt, 3, 3); St[p] = ti.Matrix.zero(rt, 3, 3)
        Jp[p] = 0.0; ys[p] = YIELD0
        mu_p[p] = mu0; lam_p[p] = lam0
        vol[p] = VO[p]; mass[p] = MA[p]; alive[p] = 1


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


@ti.kernel
def grid_op(dt: rt, t: rt):
    for I in ti.grouped(gm):
        if gm[I] > 1e-15:
            vo = (gvin[I] + gvout[I]) / gm[I] + dt * GRAV
            gvout[I] = vo * GRID_DAMP
    for I in ti.grouped(gm):
        # --- add_bounding_box: padding 3 칸, 들어오는 방향만 0 ---
        if ti.static(HAS_BBOX):
            vo = gvout[I]
            for d in ti.static(range(3)):
                if I[d] < 3 and vo[d] < 0:
                    vo[d] = 0.0
                if I[d] >= n_grid - 3 and vo[d] > 0:
                    vo[d] = 0.0
            gvout[I] = vo
        # --- add_surface_collider ---
        for c in range(NPLANE):
            if pl_t0[c] <= t < pl_t1[c]:
                off = ti.cast(I, rt) * dx - pl_pt[c]
                nrm = pl_n[c]
                if off.dot(nrm) < 0.0:
                    if pl_ty[c] == 0:
                        gvout[I] = ti.Vector.zero(rt, 3)
                    else:
                        # GF 의 slip/cut 갈래는 계산을 해놓고 마지막에 0 으로
                        # 덮어쓴다 (mpm_solver_warp.py:781). 그대로 옮긴다.
                        gvout[I] = ti.Vector.zero(rt, 3)


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
            p0 = kappa0 * (1e-5 + 0.5 * (ti.exp(z) - ti.exp(-z)))   # sinh
            Jd = s0 * s1 * s2
            b0, b1, b2 = s0 * s0, s1 * s1, s2 * s2
            bm = (b0 + b1 + b2) / 3.0
            jp = ti.pow(Jd, -2.0 / 3.0)
            sh0 = mu_p[p] * jp * (b0 - bm)
            sh1 = mu_p[p] * jp * (b1 - bm)
            sh2 = mu_p[p] * jp * (b2 - bm)
            p_tr = -(kappa0 / 2.0 * (Jd - 1.0 / Jd)) * Jd
            ysc = (6.0 - 3.0) / 2.0 * (1.0 + 2.0 * BETA)
            yph = M_CD * M_CD * (p_tr + BETA * p0) * (p_tr - p0)
            ssq = sh0 * sh0 + sh1 * sh1 + sh2 * sh2
            y = ysc * ssq + yph
            f0, f1, f2 = s0, s1, s2
            lj = logJp
            p_min = BETA * p0
            if p_tr > p0:
                Je = ti.sqrt(-2.0 * p0 / kappa0 + 1.0)
                f0 = ti.pow(Je, 1.0 / 3.0); f1 = f0; f2 = f0
                if HARDENING > 0.5:
                    lj = logJp + ti.log(Jd / Je)
            elif p_tr < -p_min:
                Je = ti.sqrt(2.0 * p_min / kappa0 + 1.0)
                f0 = ti.pow(Je, 1.0 / 3.0); f1 = f0; f2 = f0
                if HARDENING > 0.5:
                    lj = logJp + ti.log(Jd / Je)
            elif y >= 1e-4:
                sn = ti.sqrt(ssq)
                sf = ti.sqrt(-yph / ysc)
                sc = ti.pow(Jd, 2.0 / 3.0) / mu_p[p] * sf / sn
                f0 = ti.sqrt(ti.max(sc * sh0 + bm, 1e-12))
                f1 = ti.sqrt(ti.max(sc * sh1 + bm, 1e-12))
                f2 = ti.sqrt(ti.max(sc * sh2 + bm, 1e-12))
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
            prime = kappa0 / 2.0 * (J - 1.0 / J)
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


init(X0, V_init, VOL, MASS)
os.makedirs(a.out, exist_ok=True)


def dump(f):
    with h5py.File(os.path.join(a.out, f"sim_{f:010d}.h5"), "w") as h:
        h.create_dataset("x", data=x.to_numpy().T.astype(np.float32))
        h.create_dataset("v", data=v.to_numpy().T.astype(np.float32))
        h.create_dataset("time", data=np.array([[f * frame_dt]]))


dump(0)
t = 0.0
t0 = time.time()
for f in range(1, n_frames + 1):
    for _ in range(nsub):
        quarantine(-0.5 * grid_lim, 1.5 * grid_lim)
        stress_kernel(substep_dt)
        zero_grid()
        p2g(substep_dt)
        grid_op(substep_dt, t)
        g2p(substep_dt, FLIP)
        t += substep_dt
    dump(f)
    if f % 5 == 0 or f == 1:
        xn = x.to_numpy(); al = alive.to_numpy()
        jp = Jp.to_numpy()[al == 1]
        print(f"  f{f:4d}  살아있음 {int(al.sum())}/{N}  "
              f"logJp 중앙 {np.median(jp):+.4f} p1 {np.percentile(jp,1):+.4f}  "
              f"x[{xn[al==1].min():.3f},{xn[al==1].max():.3f}]  "
              f"{time.time()-t0:.0f}s", flush=True)
print(f"[저장] {a.out}  {n_frames+1} 프레임  {time.time()-t0:.0f}s", flush=True)
