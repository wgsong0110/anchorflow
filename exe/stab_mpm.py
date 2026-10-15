"""서브스텝 안정성 실험: PG(공식 명시 MPM) / i-PG(암시 MPM, newton_gmres) 를 프레임당 N 서브스텝으로.

입력: 초기 상태 h5 (x, v) + config (탄성 jelly). 출력 npz: 프레임별 측정(stabstat) + 고정 부분표본 위치 + 시간.

  cd i-physgaussian && python <anchorflow>/exe/stab_mpm.py --method pg --h5 init.h5 --config cfg.json --substeps 64 --out r.npz
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["pg", "ipg"], required=True)
ap.add_argument("--h5", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--substeps", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--ckpt_every", type=int, default=5, help="이어 돌리기 체크포인트 간격 (프레임). <out>.ckpt.pt, 0 이면 끔")
ap.add_argument("--safe", type=int, default=0,
                help="1 이면 발산 방지 보정판 (lib/anchorflow/stabstat.py safe_*): 발산 조짐에서만 걸리고 정상 실행엔 영향 없음")
ap.add_argument("--frames", type=int, default=0, help="0 이면 설정의 frame_num (검사용으로 짧게)")
ap.add_argument("--dump", default="", help="영상용: 프레임마다 전체 입자 h5 (x, F) 를 이 폴더에")
a = ap.parse_args()
sys.path.insert(0, a.pg); os.chdir(a.pg)
import h5py                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import warp as wp                                                # noqa: E402
from tqdm import tqdm                                            # noqa: E402
from anchorflow import stabstat as ss                            # noqa: E402
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP   # noqa: E402

cfg = json.load(open(a.config))
n_grid, GL = int(cfg["n_grid"]), float(cfg["grid_lim"]); dx = GL / n_grid
frame_dt, NF = float(cfg["frame_dt"]), int(cfg["frame_num"])
if a.frames:
    NF = int(a.frames)
dt = frame_dt / a.substeps
G = [float(q) for q in cfg["g"]]
floor = ([b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"] or [None])[0]
FZ = float(floor["point"][2]) if floor is not None else 0.0      # 바닥이 없는 씬(plane·pillow2sofa)은 관통 지표를 안 쓴다
OFFICIAL_BC = any(b["type"] not in ("bounding_box", "surface_collider") for b in cfg["boundary_conditions"])
with h5py.File(a.h5) as h:
    x = np.array(h["x"]); x = np.ascontiguousarray((x.T if x.shape[0] == 3 else x).astype(np.float32))
    v = np.array(h["v"]) if "v" in h else np.zeros_like(x.T)
    v = np.ascontiguousarray((v.T if v.shape[0] == 3 else v).astype(np.float32))
N = x.shape[0]
X = torch.as_tensor(x).cuda()
cell = (X / dx).floor().long(); key = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
_, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
VOL = (dx ** 3 / cnt[inv].float()).contiguous()
MASS = VOL.double().cpu().numpy() * float(cfg["density"])
SUB = ss.subset(N)
wp.init()
if a.method == "ipg":
    from implicit_mpm_solver import ImplicitMPMSolver
    sol = ImplicitMPMSolver(n_particles=10, n_grid=n_grid, grid_lim=GL)
else:
    sol = MPM_Simulator_WARP(10)
sol.load_initial_data_from_torch(X, VOL, None, n_grid=n_grid, grid_lim=GL)
if OFFICIAL_BC:
    # 공식 씬 (plane·pillow2sofa 등): 재질·감쇠·경계조건을 **공식 gs_simulation 과 같은 함수**로 넣는다
    from utils.decode_param import set_boundary_conditions
    _mp = {k: cfg[k] for k in ("material", "E", "nu", "density", "yield_stress", "hardening", "xi", "friction_angle",
                                "grid_v_damping_scale", "softening", "beta", "mass") if k in cfg}
    _mp.update({"g": G, "n_grid": n_grid, "grid_lim": GL})
    sol.set_parameters_dict(_mp)
    set_boundary_conditions(sol, cfg["boundary_conditions"], {"substep_dt": dt, "frame_dt": frame_dt, "frame_num": NF})
    print(f"[공식 설정] 재질 {_mp.get('material')}, 경계조건 {[b['type'] for b in cfg['boundary_conditions']]}", flush=True)
else:
    sol.set_parameters_dict({"material": "jelly", "E": float(cfg["E"]), "nu": float(cfg["nu"]), "density": float(cfg["density"]),
                             "g": G, "n_grid": n_grid, "grid_lim": GL, "grid_v_damping_scale": 1.1})
    sol.add_bounding_box()
    sol.add_surface_collider(tuple(floor["point"]), tuple(floor["normal"]), floor["surface"], float(floor.get("friction", 0.0)))
sol.finalize_mu_lam()
sol.import_particle_v_from_torch(torch.as_tensor(v).cuda())
print(f"[{a.method}] 입자 {N}, dx {dx:.4f}, N {a.substeps} (dt {dt:.3e}), {NF} 프레임", flush=True)
if a.dump:
    os.makedirs(a.dump, exist_ok=True)
    with h5py.File(f"{a.dump}/sim_{0:010d}.h5", "w") as h:
        h.create_dataset("x", data=x); h.create_dataset("F", data=np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (N, 1)))
_VOLN = VOL.double().cpu().numpy(); _EK = dict(vol=_VOLN, E=float(cfg["E"]), nu=float(cfg["nu"]))
rows = [ss.frame_stats(x, v, np.tile(np.eye(3), (N, 1, 1)), MASS, G, FZ, **_EK)]
XS = [x[SUB]]; T = []
step = 0
# ---- 이어 돌리기: 입자 상태 전부 + i-PG 의 직전 속도(_prev_particle_v) + 누적 기록
CKPT = a.out + ".ckpt.pt"
_PF = [k for k in ("particle_x", "particle_v", "particle_F", "particle_F_trial", "particle_C", "particle_R", "particle_cov",
                   "particle_stress", "particle_Jp", "particle_selection") if getattr(sol.mpm_state, k, None) is not None]


def _save_ckpt(f):
    st = {k: wp.to_torch(getattr(sol.mpm_state, k)).clone().cpu() for k in _PF}
    pv = getattr(sol, "_prev_particle_v", None)
    st["_prev_particle_v"] = (wp.to_torch(pv).clone().cpu() if isinstance(pv, wp.array) else
                              (pv.clone().cpu() if torch.is_tensor(pv) else None))
    _pa = getattr(sol, "_particle_a", None)                   # i-PG Newmark 입자 가속도 (다음 스텝 a^n)
    st["_particle_a"] = wp.to_torch(_pa).clone().cpu() if isinstance(_pa, wp.array) else None
    torch.save(dict(f=f, step=step, state=st, rows=rows, XS=XS, T=T, safe_n=list(SAFE_N)), CKPT + ".tmp")
    os.replace(CKPT + ".tmp", CKPT)


# ---- 보정판: 정상 실행에서는 걸리지 않는 조건에서만 개입한다 -------------------------------------------
SAFE_N = [0, 0, 0]                                           # [클램프·되돌림 입자 수, 반으로 나눠 다시 푼 횟수, 관성 대체 횟수]
if a.safe:
    _SP = ss.safe_params(x, FZ, G, float(cfg["grid_lim"]), int(cfg["n_grid"]))
    print(f"[보정판] 상자 [{_SP['lo']:.4f}, {_SP['hi']:.4f}], 속력 상한 {_SP['vcap']:.2f}", flush=True)


def _snap():
    st = {k: wp.to_torch(getattr(sol.mpm_state, k)).clone() for k in _PF}
    pv = getattr(sol, "_prev_particle_v", None)
    st["_pv"] = wp.to_torch(pv).clone() if isinstance(pv, wp.array) else (pv.clone() if torch.is_tensor(pv) else None)
    pa = getattr(sol, "_particle_a", None)
    st["_pa"] = wp.to_torch(pa).clone() if isinstance(pa, wp.array) else None
    return st


def _restore(st):
    for k in _PF:
        wp.to_torch(getattr(sol.mpm_state, k)).copy_(st[k])
    pv = getattr(sol, "_prev_particle_v", None)
    if st["_pv"] is not None:
        if isinstance(pv, wp.array):
            wp.to_torch(pv).copy_(st["_pv"])
        else:
            sol._prev_particle_v = st["_pv"].clone()
    if st["_pa"] is not None and isinstance(getattr(sol, "_particle_a", None), wp.array):
        wp.to_torch(sol._particle_a).copy_(st["_pa"])


def _pre_clamp(h):
    """서브스텝 전: 이번 스텝에 영역 상자 밖으로 나갈 입자의 속도만 자른다 (정상 실행은 상자 안이라 0 건)."""
    X_ = wp.to_torch(sol.mpm_state.particle_x); V_ = wp.to_torch(sol.mpm_state.particle_v)
    nx = X_ + h * V_
    out = ((nx < _SP["lo"]) | (nx > _SP["hi"])).any(1) | ~torch.isfinite(nx).all(1)
    k = int(out.sum())
    if k:
        tgt = nx[out].clamp(_SP["lo"], _SP["hi"])
        tgt = torch.where(torch.isfinite(tgt), tgt, X_[out])
        V_[out] = (tgt - X_[out]) / h
    return k


def _one(h, depth=0):
    """한 서브스텝 (보정판). 예외·비유한이면 되돌려 반으로 두 번, 깊이 6 (1/64) 까지. 끝내 실패하면 관성만으로 전진."""
    if a.safe:
        wp.synchronize(); torch.cuda.synchronize()            # warp·torch 스트림이 달라도 순서를 보장
    pre = _snap() if a.safe else None
    if a.safe:
        SAFE_N[0] += _pre_clamp(h)
        torch.cuda.synchronize()
    try:
        if a.method == "ipg":
            sol.p2g2p_newton_gmres(step, h)
        else:
            sol.p2g2p(step, h)
        ok = True
    except Exception as e:                                   # i-PG: GMRES 안 SVD 미수렴 등
        if not a.safe:
            raise
        ok = False
    if a.safe and ok:
        wp.synchronize()
        # 비유한이 생긴 입자는 입자 상태 **전부**(F_trial·C 등 포함)를 직전 값으로, 속도 0. 하나라도 남기면 다음 스텝에 다시 퍼진다
        bad = None
        for k in ("particle_x", "particle_v", "particle_F", "particle_F_trial", "particle_C"):
            if k in _PF:
                t_ = wp.to_torch(getattr(sol.mpm_state, k)).reshape(N, -1)
                b_ = ~torch.isfinite(t_).all(1)
                bad = b_ if bad is None else (bad | b_)
        nb = int(bad.sum())
        if nb:
            for k in _PF:
                t_ = wp.to_torch(getattr(sol.mpm_state, k))
                if t_.dtype.is_floating_point:                   # 공분산처럼 입자당 여러 값이 1 차원으로 펼쳐진 필드도 있다
                    t_.view(N, -1)[bad] = pre[k].view(N, -1)[bad]
            wp.to_torch(sol.mpm_state.particle_v)[bad] = 0.0
            for nm, key in (("_prev_particle_v", "_pv"), ("_particle_a", "_pa")):
                q = getattr(sol, nm, None)
                if pre[key] is not None and q is not None:
                    (wp.to_torch(q) if isinstance(q, wp.array) else q).view(N, -1)[bad] = pre[key].view(N, -1)[bad]
        X_ = wp.to_torch(sol.mpm_state.particle_x); V_ = wp.to_torch(sol.mpm_state.particle_v)
        SAFE_N[0] += nb + ss.safe_fix(X_, V_, None, pre["particle_x"], _SP)
        for _fk in ("particle_F", "particle_F_trial"):          # ④ F 특이값 상한 100
            if _fk in _PF:
                SAFE_N[0] += ss.clamp_F(wp.to_torch(getattr(sol.mpm_state, _fk)))
        torch.cuda.synchronize()
    if ok or not a.safe:
        return
    _restore(pre)
    if depth < 6:
        SAFE_N[1] += 1
        _one(h / 2, depth + 1); _one(h / 2, depth + 1)
        return
    SAFE_N[2] += 1                                           # 관성 대체: x += h v, v += h g (영역 안으로)
    X_ = wp.to_torch(sol.mpm_state.particle_x); V_ = wp.to_torch(sol.mpm_state.particle_v)
    V_ += h * torch.as_tensor(G, device=V_.device, dtype=V_.dtype)
    X_.add_(h * V_).clamp_(_SP["lo"], _SP["hi"])


F_START = 1
if a.ckpt_every > 0 and os.path.exists(CKPT):
    ck = torch.load(CKPT, weights_only=False)
    for k in _PF:
        wp.to_torch(getattr(sol.mpm_state, k)).copy_(ck["state"][k].cuda())
    if ck["state"]["_prev_particle_v"] is not None:
        pv = getattr(sol, "_prev_particle_v", None)
        if isinstance(pv, wp.array):
            wp.to_torch(pv).copy_(ck["state"]["_prev_particle_v"].cuda())
        else:
            sol._prev_particle_v = (wp.from_torch(ck["state"]["_prev_particle_v"].cuda().contiguous(), dtype=wp.vec3)
                                    if a.method == "ipg" and ck["state"]["_prev_particle_v"].dim() == 2 else ck["state"]["_prev_particle_v"].cuda())
    if ck["state"].get("_particle_a") is not None:
        sol._particle_a = wp.from_torch(ck["state"]["_particle_a"].cuda().contiguous(), dtype=wp.vec3)
    elif a.method == "ipg" and ck["f"] > 0:
        print("[이어 돌리기] 경고: 체크포인트에 i-PG 입자 가속도가 없다 (a^n 패치 전 체크포인트)", flush=True)
    rows, XS, T, step, F_START = ck["rows"], ck["XS"], ck["T"], ck["step"], ck["f"] + 1
    SAFE_N[:] = list(ck.get("safe_n") or [0, 0, 0])
    print(f"[이어 돌리기] {CKPT} 프레임 {ck['f']} 부터 (step {step})", flush=True)
for f in tqdm(range(F_START, NF + 1), desc=f"{a.method} N{a.substeps}"):
    t0 = time.time()
    for s in range(a.substeps):
        _one(dt)
        step += 1
    torch.cuda.synchronize(); T.append(time.time() - t0)
    xt = sol.export_particle_x_to_torch().cpu().numpy()
    vt = wp.to_torch(sol.mpm_state.particle_v).cpu().numpy()
    Ft = sol.export_particle_F_to_torch().cpu().numpy()
    rows.append(ss.frame_stats(xt, vt, Ft, MASS, G, FZ, **_EK))
    rows[-1]["safe_n"], rows[-1]["safe_split"], rows[-1]["safe_inert"] = (float(q) for q in SAFE_N)
    if a.dump:
        with h5py.File(f"{a.dump}/sim_{f:010d}.h5", "w") as h:
            h.create_dataset("x", data=xt.astype(np.float32)); h.create_dataset("F", data=Ft.reshape(N, 9).astype(np.float32))
    XS.append(xt[SUB])
    if rows[-1]["nan"] > 0.5 and not a.safe:                     # 대부분 발산했으면 더 돌 이유가 없다
        print(f"  발산 프레임 {f}", flush=True); break
    if a.ckpt_every > 0 and f % a.ckpt_every == 0 and f < NF:
        _save_ckpt(f)
np.savez_compressed(a.out, sub=SUB, x=np.stack(XS).astype(np.float32), t=np.array(T),
                    stats=np.array([[r.get(k, np.nan) for k in ss.KEYS + ss.SAFE_KEYS] for r in rows]),
                    keys=np.array(ss.KEYS + ss.SAFE_KEYS), substeps=a.substeps, dt=dt, safe=a.safe)
print(f"[저장] {a.out}  프레임당 {np.mean(T):.3f}s", flush=True)
