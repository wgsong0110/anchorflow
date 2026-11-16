"""[final_fix_mpmbl] 파레토 MPM 베이스라인 실행기: i-PhysGaussian (i-PG) · OmniPhysGS 를 파레토 장면 (레고 L000·pillow2sofa·bread·wolf) 의 정답 MPM 과 같은 입력으로.

틀 = exe/pareto_mpm.py (정답 MPM 파레토 행 실행기) 그대로 -- 같은 초기 상태 (정답 0 프레임 h5 x·v), 같은 입자 부피 (dx0^3 / 칸 입자 수),
같은 재질·경계조건 경로 (--pg_cfg: PG decode_param.set_boundary_conditions, 레고: jelly + bounding_box + 바닥 sticky), 같은 오차
(프레임마다 입자 평균 |x - x_gt| / EXT, EXT = 자료 .pt 0 프레임 바운딩박스 대각선), 같은 시간 (--time: 서브스텝 루프 앞뒤 동기화 벽시계, 첫 2 프레임 뺀 평균).
바뀌는 것은 솔버만:
  ipg  = i-physgaussian ImplicitMPMSolver.p2g2p_newton_gmres (안정성 실험 stab_mpm.py --method ipg 와 같은 호출). AF_IPG_FUSED 는 환경 변수로 (탄성 장면만 1).
  omni = OmniPhysGS 공식 MPMModel.p2g2p (stab_omni.py 와 같은 길이 s=0.5 축소: x'=s x, v'=s v, g'=s g, E'=s^2 E, 항복 응력 s^2, 입자 부피 = 평균 x s^3),
         구성모델 = 공식 물리 구성모델 목록: jelly = Corotated + 항등, plasticine = Corotated + VonMises, sand = StVK + DruckerPrager (마찰각 = 정답 경로 값, 점착 0),
         경계조건 = 공식 set_boundary_conditions 목록 (cuboid·enforce_particle_translation·surface_collider; 좌표·속도 s 배), bounding_box = 공식 clip_bound,
         release_particles_sequentially (공식은 NotImplementedError) = PG 구현과 같은 50 겹 enforce_particle_translation (속도 0) 으로 펼침.
  pg   = 점검용 (정답 MPM 경로 그대로; --mat_extra 로 재질 키 확인)
출력 DIR/err.json (+ --dump 이면 pred.pt {pred [T,N,3] 프레임 1..T, F [T,N,3,3] = F_sim}; ffx_phys.py·lb_roll_to_h5.py 입력 형식)

  python exe/pareto_mpmbl.py --method ipg --pg <i-physgaussian> --gt H5DIR --config CFG --pt DATA.pt --frames 300 --nsub 10 --pg_cfg --time --out DIR [--dump]
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["ipg", "omni", "pg"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gt", default="/home/dkta/work/lbdata/test_low/L000")
ap.add_argument("--pt", default="/home/dkta/work/lbpt/test_low/L000.pt")
ap.add_argument("--config", default="/home/dkta/work/lbdata/config.json")
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--omni", default="/home/dkta/work/OmniPhysGS")
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--nsub", type=int, required=True)
ap.add_argument("--pg_cfg", action="store_true")
ap.add_argument("--mat_extra", default="", help="점검용: PG 재질 키 추가 (쉼표, 예 yield_stress,friction_angle) -- 정답 경로 (pareto_mpm) 는 안 넘김")
ap.add_argument("--omni_yield", type=float, default=None, help="omni plasticine 항복 응력 (원 단위; 기본 = 정답 경로 실제 값)")
ap.add_argument("--omni_fric", type=float, default=None, help="omni sand 마찰각 (도; 기본 = 정답 경로 실제 값)")
ap.add_argument("--time", action="store_true")
ap.add_argument("--dump", action="store_true")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
import h5py                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

cfg = json.load(open(a.config))
cfg.setdefault("n_grid", 50)
cfg.setdefault("grid_lim", 2.0)
meta = json.load(open(os.path.join(a.gt, "meta.json"))) if os.path.exists(os.path.join(a.gt, "meta.json")) else {}
GL = float(cfg["grid_lim"]); n_grid = int(cfg["n_grid"]); dx0 = GL / n_grid
frame_dt = float(cfg["frame_dt"]); nsub = a.nsub; dt = frame_dt / nsub
G = [float(q) for q in cfg["g"]]
_fl = [b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"]
floor = dict(_fl[0]) if _fl else None
d = torch.load(a.pt, map_location="cpu", weights_only=False, mmap=True)
x0d = d["x"][0].float()
EXT = float((x0d.max(0).values - x0d.min(0).values).norm())
NPT = int(x0d.shape[0])
del d, x0d


def h5x(t, key="x"):
    with h5py.File(os.path.join(a.gt, f"sim_{t:010d}.h5")) as h:
        r = np.array(h[key])
    return r.T.copy() if (r.ndim == 2 and r.shape[0] == 3 and r.shape[1] != 3) else r


x0 = h5x(0).astype(np.float32)
try:
    v0 = h5x(0, "v").astype(np.float32)
except KeyError:
    v0 = np.zeros_like(x0)
if v0.shape != x0.shape:
    v0 = v0.T.copy()
N = x0.shape[0]
assert N == NPT, f"정답 h5 입자 {N} != 자료 .pt 입자 {NPT}"
L = min(a.frames, int(meta.get("frames", cfg.get("frame_num", a.frames))))
X = torch.as_tensor(x0).cuda().contiguous()
cell = (X / dx0).floor().long()
key = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
_, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
VOL = (dx0 ** 3 / cnt[inv].float()).contiguous()
MAT = cfg.get("material", "jelly") if a.pg_cfg else "jelly"
print(f"[{a.method}] 입자 {N}, n_grid {n_grid}, 서브스텝 {dt:.4e} x {nsub}, 재질 {MAT}, 바닥 {floor}, EXT {EXT:.6f}, 프레임 {L}, "
      f"AF_IPG_FUSED={os.environ.get('AF_IPG_FUSED')}", flush=True)
print(f"[명령] {' '.join(sys.argv)}", flush=True)
import warp as wp                                                # noqa: E402
wp.init()

# ------------------------------------------------------------------------------------------------ 솔버 세우기
if a.method in ("ipg", "pg"):
    sys.path.insert(0, a.pg); os.chdir(a.pg)
    if a.method == "ipg":
        from implicit_mpm_solver import ImplicitMPMSolver     # noqa: E402
        sol = ImplicitMPMSolver(n_particles=10, n_grid=n_grid, grid_lim=GL)
    else:
        from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP   # noqa: E402
        sol = MPM_Simulator_WARP(10)
    sol.load_initial_data_from_torch(X, VOL, None, n_grid=n_grid, grid_lim=GL)
    if a.pg_cfg:                                             # pareto_mpm --pg_cfg 와 같은 순서·같은 키 (재질 -> 경계조건 -> mu·lam)
        from utils.decode_param import set_boundary_conditions   # noqa: E402
        ks = ["material", "E", "nu", "density", "g", "grid_v_damping_scale", "additional_material_params"] + [k for k in a.mat_extra.split(",") if k]
        _mp = {k: cfg[k] for k in ks if k in cfg}
        _mp.update({"n_grid": n_grid, "grid_lim": GL, "rpic_damping": 0.0})
        sol.set_parameters_dict(_mp)
        set_boundary_conditions(sol, cfg.get("boundary_conditions", []), {"substep_dt": dt, "frame_dt": frame_dt, "frame_num": L})
        if not np.any(v0) and cfg.get("init_velocity_bc"):
            set_boundary_conditions(sol, cfg["init_velocity_bc"], {"substep_dt": dt, "frame_dt": frame_dt, "frame_num": L})
        sol.finalize_mu_lam()
    else:
        sol.set_parameters_dict({"material": "jelly", "E": float(cfg["E"]), "nu": float(cfg["nu"]), "density": float(cfg["density"]),
                                 "g": G, "n_grid": n_grid, "grid_lim": GL, "grid_v_damping_scale": 1.1, "rpic_damping": 0.0})
        sol.add_bounding_box()
        sol.add_surface_collider(tuple(floor["point"]), tuple(floor["normal"]), floor["surface"], float(floor.get("friction", 0.0)))
        sol.finalize_mu_lam()
    sol.import_particle_v_from_torch(torch.as_tensor(v0).cuda().contiguous())
    _mm = sol.mpm_model
    try:
        print(f"[재질 확인] material {int(_mm.material)}, friction_angle {float(getattr(_mm, 'friction_angle', float('nan')))}, "
              f"yield_stress {float(wp.to_torch(_mm.yield_stress)[0])}, softening {getattr(_mm, 'softening', None)}, "
              f"damping {getattr(_mm, 'grid_v_damping_scale', None)}", flush=True)
    except Exception as _e:
        print("[재질 확인] 실패", repr(_e)[:120], flush=True)

    def substep(step):
        if a.method == "ipg":
            sol.p2g2p_newton_gmres(step, dt)
        else:
            sol.p2g2p(step, dt)

    def state_x():
        return sol.export_particle_x_to_torch()

    def state_F():
        return sol.export_particle_F_to_torch().reshape(-1, 3, 3)
else:
    sys.path.insert(0, a.omni); os.chdir(a.omni)
    from src.mpm_core import MPMModel, set_boundary_conditions as omni_bc   # noqa: E402
    from src.constitutive_models.physical_constitutive_models import (CorotatedElasticity, StVKElasticity,   # noqa: E402
                                                                       VonMisesPlasticity, DruckerPragerPlasticity)
    S = 0.5
    assert abs(GL * S - 1.0) < 1e-9
    E, nu = float(cfg["E"]), float(cfg["nu"])
    # 정답 경로 (pareto_mpm) 의 실제 재질 값: PG 모델 기본값 (항복 응력 0, 마찰각 25) 은 키를 안 넘기면 그대로 -- 점검 판 (--method pg --mat_extra) 으로 확인한 값을 인자로
    sig_y = a.omni_yield if a.omni_yield is not None else 0.0
    fric = a.omni_fric if a.omni_fric is not None else 25.0
    DAMP = float(cfg.get("grid_v_damping_scale", 1.0)) if a.pg_cfg else 1.0
    DAMP = DAMP if DAMP < 1.0 else 1.0
    vol_p = float(VOL.mean()) * S ** 3
    sim = {"num_grids": n_grid, "dt": dt, "gravity": [g * S for g in G], "clip_bound": 0.5, "damping": DAMP}
    mat = {"center": [0.5] * 3, "size": [vol_p * N, 1.0, 1.0], "rho": float(cfg["density"])}
    model = MPMModel(sim, mat, init_pos=X * S, enable_train=False, device=torch.device("cuda"))
    bcs = []
    for b in cfg["boundary_conditions"]:
        t = b["type"]
        if t == "bounding_box":
            continue                                              # 공식 clip_bound
        if t == "surface_collider":
            bcs.append(dict(type=t, point=[p * S for p in b["point"]], normal=b["normal"], surface=b["surface"], friction=float(b.get("friction", 0.0)),
                            start_time=float(b.get("start_time", 0.0)), end_time=float(b.get("end_time", 1000.0))))
        elif t == "cuboid":
            bcs.append(dict(type=t, point=[p * S for p in b["point"]], size=[q * S for q in b["size"]], velocity=[q * S for q in b["velocity"]],
                            start_time=float(b.get("start_time", 0.0)), end_time=float(b.get("end_time", 1000.0)), reset=bool(b.get("reset", False))))
        elif t == "enforce_particle_translation":
            bcs.append(dict(type=t, point=[p * S for p in b["point"]], size=[q * S for q in b["size"]], velocity=[q * S for q in b["velocity"]],
                            start_time=float(b["start_time"]), end_time=float(b["end_time"])))
        elif t == "release_particles_sequentially":                  # PG mpm_solver_warp.release_particles_sequentially 와 같은 펼침
            nl = 50; point = [0, 0, 0]; size = [0, 0, 0]; axis = -1
            for i in range(3):
                if b["normal"][i] == 0:
                    point[i] = 1; size[i] = 1
                else:
                    axis = i; point[i] = b["end_position"]
            half = abs(b["start_position"] - b["end_position"]) / nl; etp = b["end_time"] / nl
            for i in range(nl):
                sz = list(size); sz[axis] = half * (nl - i)
                bcs.append(dict(type="enforce_particle_translation", point=[p * S for p in point], size=[q * S for q in sz], velocity=[0.0, 0.0, 0.0],
                                start_time=float(b["start_time"]), end_time=float(etp * (i + 1))))
        else:
            raise SystemExit(f"[omni] 경계조건 {t} 대응 없음")
    if not a.pg_cfg:                                                 # 레고: 정답 경로와 같이 바닥만 (재질 jelly)
        bcs = [q for q in bcs if q["type"] == "surface_collider"]
    omni_bc(model, bcs)
    elas = (StVKElasticity() if MAT in ("metal", "sand") else CorotatedElasticity()).cuda()
    elas.log_E = torch.tensor([E * S * S], device="cuda").log(); elas.nu = torch.tensor([nu], device="cuda")
    plas = None
    if MAT in ("metal", "plasticine"):
        plas = VonMisesPlasticity().cuda()
        plas.log_E = torch.tensor([E * S * S], device="cuda").log(); plas.nu = torch.tensor([nu], device="cuda")
        plas.sigma_y = torch.tensor([sig_y * S * S], device="cuda")
    elif MAT == "sand":
        plas = DruckerPragerPlasticity().cuda()
        plas.log_E = torch.tensor([E * S * S], device="cuda").log(); plas.nu = torch.tensor([nu], device="cuda")
        plas.friction_angle = torch.tensor([fric], device="cuda"); plas.cohesion = torch.tensor([0.0], device="cuda")
    elif MAT != "jelly":
        raise SystemExit(f"[omni] 재질 {MAT} 대응 없음")
    print(f"[omni] 재질 {MAT}: 탄성 {type(elas).__name__}, 소성 {type(plas).__name__ if plas is not None else '항등'}, 감쇠 {DAMP}, "
          f"항복 {sig_y if MAT in ('metal', 'plasticine') else '-'}, 마찰각 {fric if MAT == 'sand' else '-'}, 경계조건 {[q['type'] for q in bcs]} ({len(bcs)} 개), 입자 부피 {vol_p:.3e}", flush=True)
    ST = dict(x=X * S, v=torch.as_tensor(v0).cuda() * S, C=torch.zeros(N, 3, 3, device="cuda"), F=torch.eye(3, device="cuda").repeat(N, 1, 1))

    def substep(step):
        with torch.no_grad():
            stress = elas(ST["F"])
            ST["x"], ST["v"], ST["C"], ST["F"] = model(ST["x"], ST["v"], ST["C"], ST["F"], stress)
            if plas is not None:
                ST["F"] = plas(ST["F"])

    def state_x():
        return ST["x"] / S

    def state_F():
        return ST["F"]

# ------------------------------------------------------------------------------------------------굴리기
os.makedirs(a.out, exist_ok=True)
PX = torch.empty(L, N, 3, dtype=torch.float32) if a.dump else None
PF = torch.empty(L, N, 3, 3, dtype=torch.float32) if a.dump else None
errs, MS = [], []
DIV = None
step = 0
t0 = time.time()
for f in range(1, L + 1):
    torch.cuda.synchronize(); wp.synchronize(); _tf = time.perf_counter()
    try:
        for s in range(nsub):
            substep(step)
            step += 1
    except Exception as e:                                           # i-PG GMRES 안 SVD 미수렴·OmniPhysGS 유한성 단언 등 -> 발산
        DIV = dict(frame=f, substep=s + 1, reason=f"예외 {repr(e)[:200]}")
    wp.synchronize(); torch.cuda.synchronize(); MS.append(1000.0 * (time.perf_counter() - _tf))
    if DIV is not None:
        errs.append(float("nan")); print(f"[발산] 프레임 {f}: {DIV['reason']}", flush=True); break
    xt = state_x()
    gt = torch.as_tensor(h5x(f)).cuda()
    e = float((xt - gt).norm(dim=-1).mean()) / EXT
    if not torch.isfinite(xt).all():
        e = float("nan")
    errs.append(e)
    if a.dump:
        PX[f - 1] = xt.float().cpu(); PF[f - 1] = state_F().float().reshape(N, 3, 3).cpu()
    print(f"  프레임 {f}/{L} 오차 {100 * e:.3f}% (누적 평균 {100 * np.nanmean(errs):.3f}%) {MS[-1]:.1f} ms 최저 z {float(xt[:, 2].min()):.5f} {time.time() - t0:.0f}s", flush=True)
    if not np.isfinite(e) or e > 2.0:
        DIV = dict(frame=f, substep=nsub, reason="비유한" if not np.isfinite(e) else f"오차 {100 * e:.1f}% > 200%")
        print(f"[발산] 프레임 {f}", flush=True); break
summ = dict(args=vars(a), method=a.method, nsub=nsub, dt=dt, n_grid=n_grid, material=MAT, EXT=EXT, n_particles=N, frames_total=L, frames_done=len(errs),
            err_mean=float(np.mean(errs)) if errs else None, err_pct=(100 * float(np.mean(errs))) if (errs and DIV is None) else None, div=DIV,
            err=errs, sec=time.time() - t0, ms=MS, ms_mean=(float(np.mean(MS[2:])) if len(MS) > 2 else None), ms_max=(float(np.max(MS[2:])) if len(MS) > 2 else None),
            af_ipg_fused=os.environ.get("AF_IPG_FUSED"), af_ipg_gpu=os.environ.get("AF_IPG_GPU"), gpu=torch.cuda.get_device_name(0))
json.dump(summ, open(os.path.join(a.out, "err.json"), "w"), ensure_ascii=False, indent=1)
if a.dump:
    nd = len(errs) if DIV is None else max(len(errs) - 1, 0)
    torch.save(dict(pred=PX[:nd].clone() if nd < L else PX, F=PF[:nd].clone() if nd < L else PF, part=nd < L, diverged=DIV,
                    method=a.method, nsub=nsub), os.path.join(a.out, "pred.pt"))
print(f"[결과] {a.method} N {nsub}: 평균 오차 {summ['err_pct']}% ({len(errs)}/{L} 프레임), 프레임당 {summ['ms_mean']} ms, {summ['sec']:.0f}s", flush=True)
