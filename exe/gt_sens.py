"""정답 MPM (PG 공식 warp 솔버, lb_gen.py 와 같은 경로) 의 민감도 -- lego L000 을 조건 하나씩 바꿔 다시 굴려 정답과의 차를 잰다 (doc/fem_realtime.md 절 N.1).

정답 L000 = lb_gen.py (i-physgaussian MPM_Simulator_WARP, jelly = 고정 FCR, APIC rpic_damping 0, n_grid 100 · grid_lim 2 (dx 0.02),
substep_dt 2.0008e-5 (833/프레임), 바닥 surface_collider sticky (격자 노드 z < 0.48 -> 격자 속도 0), bounding_box, 감쇠 없음 (scale 1.1)).
초기 상태는 정답 0 프레임 h5 (x, v) 와 같은 값, 입자 부피 = dx0³ / (dx0 = 0.02 셀의 입자 수) -- **격자를 바꿔도 질량·부피는 정답 그대로** (격자 효과만 분리).
오차 = 프레임마다 입자 평균 |x - x_gt| / EXT 의 80 프레임 평균 (fem_ex_sim 과 같은 지표, EXT = L000.pt _ext).

  cd i-physgaussian && python <anchorflow>/exe/gt_sens.py --out DIR [--nsub 833] [--rpic 0|-1] [--n_grid 100] [--floor_z 0.48] [--surface sticky|slip]
출력 DIR/: x.pt (float32 [80, N, 3]), err.json (프레임별 오차, 요약)
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--gt", default="/home/dkta/work/lbdata/test_low/L000", help="정답 h5 폴더 (sim_XXXXXXXXXX.h5, meta.json)")
ap.add_argument("--pt", default="/home/dkta/work/lbpt/test_low/L000.pt", help="EXT (_ext) 를 읽을 곳")
ap.add_argument("--config", default="/home/dkta/work/lbdata/config.json")
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--nsub", type=int, default=0, help="프레임당 서브스텝 (0 = config 그대로 833)")
ap.add_argument("--rpic", type=float, default=0.0, help="PG rpic_damping: 0 = APIC (정답), -1 = PIC")
ap.add_argument("--n_grid", type=int, default=0, help="0 = config 그대로 (100)")
ap.add_argument("--floor_z", type=float, default=None, help="바닥 평면 높이 (정답 0.48)")
ap.add_argument("--surface", default=None, help="바닥 종류 (정답 sticky)")
ap.add_argument("--no_dump", action="store_true")
ap.add_argument("--flip", type=float, default=0.0, help="절 AK.14 진단: 서브스텝마다 FLIP 섞기 v = v_PIC + α (v_old − v_s), v_s = 같은 서브스텝 시작 상태의 운동량만 P2G (APIC 같은 식, 힘·중력·경계 없음) -> G2P 속도 (0 = 원래 정답 솔버)")
ap.add_argument("--diag", action="store_true", help="절 AK.14: 프레임마다 몸 좌표 높이 (0 프레임 대비 Kabsch)·운동 에너지·입자 탄성 Σ V ψ(F) -> err.json diag")
a = ap.parse_args()
sys.path.insert(0, a.pg)
os.chdir(a.pg)
import h5py                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import warp as wp                                                # noqa: E402
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP   # noqa: E402

cfg = json.load(open(a.config))
meta = json.load(open(os.path.join(a.gt, "meta.json")))
GL = float(cfg["grid_lim"])
n_grid0 = int(cfg["n_grid"])
dx0 = GL / n_grid0
n_grid = a.n_grid or n_grid0
frame_dt = float(cfg["frame_dt"])
nsub = a.nsub or int(round(frame_dt / float(cfg["substep_dt"])))
dt = frame_dt / nsub
G = [float(q) for q in cfg["g"]]
floor = dict([b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0])
if a.floor_z is not None:
    floor["point"] = [floor["point"][0], floor["point"][1], float(a.floor_z)]
if a.surface is not None:
    floor["surface"] = a.surface
d = torch.load(a.pt, map_location="cpu", weights_only=False, mmap=True)
x0d = d["x"][0].float()
EXT = float((x0d.max(0).values - x0d.min(0).values).norm())       # deform_model.extent (바운딩박스 대각선) 와 같다 -- fem_ex_sim 의 EXT
del d, x0d


def h5x(t, key="x"):
    with h5py.File(os.path.join(a.gt, f"sim_{t:010d}.h5")) as h:
        return np.array(h[key])


x0 = h5x(0).astype(np.float32)
v0 = h5x(0, "v").astype(np.float32)
N = x0.shape[0]
L = min(a.frames, int(meta["frames"]))
X = torch.as_tensor(x0).cuda().contiguous()
cell = (X / dx0).floor().long()
key = (cell[:, 0] * n_grid0 + cell[:, 1]) * n_grid0 + cell[:, 2]
_, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
VOL = (dx0 ** 3 / cnt[inv].float()).contiguous()
print(f"[정답 민감도] 입자 {N}, n_grid {n_grid} (dx {GL / n_grid:.5f}), 서브스텝 {dt:.4e} x {nsub}, rpic {a.rpic}, 바닥 {floor}, EXT {EXT:.6f}", flush=True)
print(f"[명령] {' '.join(sys.argv)}", flush=True)
wp.init()


@wp.kernel
def ak_p2g_mom(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), C: wp.array(dtype=wp.mat33), mass: wp.array(dtype=float), sel: wp.array(dtype=int),
               inv_dx: float, dx: float, gv: wp.array3d(dtype=wp.vec3), gm: wp.array3d(dtype=float)):
    p = wp.tid()
    if sel[p] == 0:
        gp = x[p] * inv_dx
        bx = wp.int(gp[0] - 0.5)
        by = wp.int(gp[1] - 0.5)
        bz = wp.int(gp[2] - 0.5)
        fx = gp - wp.vec3(wp.float(bx), wp.float(by), wp.float(bz))
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(wp.cw_mul(wa, wa) * 0.5, wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75), wp.cw_mul(wc, wc) * 0.5)
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    dpos = (wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx) * dx
                    weight = w[0, i] * w[1, j] * w[2, k]
                    wp.atomic_add(gv, bx + i, by + j, bz + k, weight * mass[p] * (v[p] + C[p] * dpos))
                    wp.atomic_add(gm, bx + i, by + j, bz + k, weight * mass[p])


@wp.kernel
def ak_g2p_v(x: wp.array(dtype=wp.vec3), sel: wp.array(dtype=int), inv_dx: float, gv: wp.array3d(dtype=wp.vec3), gm: wp.array3d(dtype=float),
             vs: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    if sel[p] == 0:
        gp = x[p] * inv_dx
        bx = wp.int(gp[0] - 0.5)
        by = wp.int(gp[1] - 0.5)
        bz = wp.int(gp[2] - 0.5)
        fx = gp - wp.vec3(wp.float(bx), wp.float(by), wp.float(bz))
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(wp.cw_mul(wa, wa) * 0.5, wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75), wp.cw_mul(wc, wc) * 0.5)
        nv = wp.vec3(0.0, 0.0, 0.0)
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    weight = w[0, i] * w[1, j] * w[2, k]
                    m_ = gm[bx + i, by + j, bz + k]
                    if m_ > 1e-15:
                        nv = nv + gv[bx + i, by + j, bz + k] * (weight / m_)
        vs[p] = nv

sol = MPM_Simulator_WARP(10)
sol.load_initial_data_from_torch(X, VOL, None, n_grid=n_grid, grid_lim=GL)
sol.set_parameters_dict({"material": "jelly", "E": float(cfg["E"]), "nu": float(cfg["nu"]), "density": float(cfg["density"]),
                         "g": G, "n_grid": n_grid, "grid_lim": GL, "grid_v_damping_scale": 1.1, "rpic_damping": float(a.rpic)})
sol.add_bounding_box()
sol.add_surface_collider(tuple(floor["point"]), tuple(floor["normal"]), floor["surface"], float(floor.get("friction", 0.0)))
sol.finalize_mu_lam()
sol.import_particle_v_from_torch(torch.as_tensor(v0).cuda().contiguous())
os.makedirs(a.out, exist_ok=True)
XD = None if a.no_dump else torch.empty(L, N, 3, dtype=torch.float32)
errs = []
t0 = time.time()
step = 0
if a.flip > 0:
    _st, _md = sol.mpm_state, sol.mpm_model
    _gv = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=wp.vec3, device="cuda:0")
    _gm = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=float, device="cuda:0")
    _vs = wp.zeros(shape=N, dtype=wp.vec3, device="cuda:0")
    print(f"[절 AK.14] FLIP 섞기 α {a.flip}", flush=True)
DIAG = []
if a.diag:
    import math as _m
    _x0 = torch.as_tensor(x0).cuda().double()
    _q0 = _x0 - _x0.mean(0)
    _mass = (VOL.double() * float(cfg["density"]))
    _E, _nu = float(cfg["E"]), float(cfg["nu"])
    _mu, _lam = _E / (2 * (1 + _nu)), _E * _nu / ((1 + _nu) * (1 - 2 * _nu))

    def _diag(f):
        xx = sol.export_particle_x_to_torch().double()
        vv = wp.to_torch(sol.mpm_state.particle_v).double()
        FF = wp.to_torch(sol.mpm_state.particle_F_trial if hasattr(sol.mpm_state, "particle_F_trial") else sol.mpm_state.particle_F).double().reshape(-1, 3, 3)
        c = xx.mean(0)
        U_, S_, Vh_ = torch.linalg.svd(_q0.T @ (xx - c))
        Dg = torch.eye(3, device="cuda", dtype=torch.float64)
        Dg[2, 2] = torch.sign(torch.det(Vh_.T @ U_.T))
        Rm = Vh_.T @ Dg @ U_.T
        b = (xx - c) @ Rm
        res = ((xx - c) - _q0 @ Rm.T).norm(dim=-1)
        sv = torch.linalg.svdvals(FF)
        J = torch.linalg.det(FF)
        psi = _mu * ((sv - 1) ** 2).sum(-1) + 0.5 * _lam * (J - 1) ** 2
        DIAG.append(dict(f=f, height=float(b[:, 2].max() - b[:, 2].min()), res=100 * float(res.pow(2).mean().sqrt()) / EXT, comz=float(c[2]),
                         ke=float(0.5 * (_mass * (vv ** 2).sum(-1)).sum()), ep=float((VOL.double() * psi).sum())))
for f in range(1, L + 1):
    for s in range(nsub):
        if a.flip > 0:                                                                # v_s: 서브스텝 시작 상태의 운동량만 왕복
            _gv.zero_()
            _gm.zero_()
            wp.launch(ak_p2g_mom, dim=N, inputs=[_st.particle_x, _st.particle_v, _st.particle_C, _st.particle_mass, _st.particle_selection,
                                                 _md.inv_dx, _md.dx, _gv, _gm], device="cuda:0")
            wp.launch(ak_g2p_v, dim=N, inputs=[_st.particle_x, _st.particle_selection, _md.inv_dx, _gv, _gm, _vs], device="cuda:0")
            v_old = wp.to_torch(_st.particle_v).clone()
            v_s = wp.to_torch(_vs).clone()
        sol.p2g2p(step, dt)
        if a.flip > 0:
            vt = wp.to_torch(_st.particle_v)
            vt.add_(a.flip * (v_old - v_s))
        step += 1
    if a.diag:
        _diag(f)
    xt = sol.export_particle_x_to_torch()
    gt = torch.as_tensor(h5x(f)).cuda()
    e = float((xt - gt).norm(dim=-1).mean()) / EXT
    if not torch.isfinite(xt).all():
        e = float("nan")
    errs.append(e)
    if XD is not None:
        XD[f - 1] = xt.cpu()
    print(f"  프레임 {f}/{L} 오차 {100 * e:.3f}% (누적 평균 {100 * np.nanmean(errs):.3f}%) 최저 {float(xt[:, 2].min()):.5f} {time.time() - t0:.0f}s", flush=True)
    if not np.isfinite(e):
        print(f"[발산] 프레임 {f}", flush=True)
        break
summ = dict(args=vars(a), nsub=nsub, dt=dt, n_grid=n_grid, floor=floor, EXT=EXT, frames_done=len(errs), err_mean=float(np.mean(errs)),
            err=errs, sec=time.time() - t0, diag=DIAG)
json.dump(summ, open(os.path.join(a.out, "err.json"), "w"), ensure_ascii=False, indent=1)
if XD is not None:
    torch.save(XD, os.path.join(a.out, "x.pt"))
print(f"[정답 민감도 결과] {os.path.basename(a.out)}: 평균 오차 {100 * summ['err_mean']:.3f}% ({len(errs)}/{L} 프레임), {summ['sec']:.0f}s", flush=True)
