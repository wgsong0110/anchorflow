"""GaussianFluent 수박 낙하 장면을 PhysGaussian 공식 MPM / MPMAvatar MPM 으로 돌린다 (재질은 GF 에 최대한 맞춤).

입력은 GF 공식 러너가 저장한 0 프레임(GF 내부 채우기 포함 138 만 입자)과 그 config.
GF 와 같게 둔 것: E 2000, ν 0.38, 밀도 1, 마찰각 45°, g (0,0,-15), 초기 속도 (0,0,-6), 격자 300 / 상자 2,
  서브스텝 0.6 dx / 음속 (GF 수박 러너와 같은 식), 프레임 0.03 s × 100, 경계는 상자 (GF 의 z=0 충돌면은 격자 밖이라 무효).
못 맞춘 것: 두 솔버 모두 NACC(CD-MPM) 재질이 없다 -> 공식 재질 중 같은 마찰각을 쓰는 sand(Drucker–Prager) 로.
  전달은 두 솔버 모두 APIC (GF 는 FLIP 0.7). GF 의 씨앗 영역 β 강화는 재질이 달라 옮길 수 없다.
출력: 프레임마다 sim_XXXXXXXXXX.h5 (x (3,N), F (N,9)) -- exe/wm_render.py 로 GF 와 같은 렌더.

  python exe/wm_other.py --solver pg|mpma --h5 <GF sim_0000000000.h5> --config <GF config> --out DIR
"""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--solver", choices=["pg", "mpma"], required=True)
ap.add_argument("--h5", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=0)
ap.add_argument("--material", choices=["sand", "nacc"], default="sand", help="nacc: GF 와 같은 CD-MPM(NACC) 소성 (β·ξ·경화·초기 log Jp 는 GF config 그대로). MPMAvatar 는 AF_MP_ROOT=MPMAvatar_nacc")
ap.add_argument("--check", type=int, default=0, help="이 서브스텝마다 동기화하고 x·F NaN 검사 (디버그)")
ap.add_argument("--drop_isolated", type=int, default=0, help="3×3×3 칸 안에 혼자인 입자(GF 채우기의 외톨이)는 솔버에서 빼고 자유낙하로 채운다 -- APIC+NACC 에서 혼자 발산해 NaN 이 격자 인덱스를 깬다")
ap.add_argument("--E", type=float, default=0, help="E 를 GF config 대신 이 값으로 (0 이면 config)")
a = ap.parse_args()
import h5py                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from tqdm import tqdm                                            # noqa: E402

cfg = json.load(open(a.config))
E, nu, rho = (a.E or float(cfg["E"])), float(cfg["nu"]), float(cfg["density"])
n_grid, GL = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
dx = GL / n_grid
c = math.sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * rho))
sub = 0.6 * dx / c
frame_dt = float(cfg["frame_dt"]); nsub = int(frame_dt / sub)
NF = a.frames or int(cfg["frame_num"])
G = [float(g) for g in cfg["g"]]
with h5py.File(a.h5) as h:
    x = np.array(h["x"]); x = np.ascontiguousarray((x.T if x.shape[0] == 3 else x).astype(np.float32))
NALL = x.shape[0]
XA = torch.as_tensor(x).cuda()
cell = (XA / dx).floor().long(); key = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
_, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
VOLA = (dx ** 3 / cnt[inv].float()).contiguous()                 # GF/PG get_particle_volume 와 같은 정의 (빼기 전 전체로)
keep = torch.ones(NALL, dtype=torch.bool, device="cuda")
if a.drop_isolated:
    occ = torch.zeros(n_grid + 2, n_grid + 2, n_grid + 2, device="cuda")
    c1 = (cell + 1).clamp(0, n_grid + 1); occ.index_put_((c1[:, 0], c1[:, 1], c1[:, 2]), torch.ones(NALL, device="cuda"), accumulate=True)
    nb = torch.nn.functional.conv3d(occ[None, None], torch.ones(1, 1, 3, 3, 3, device="cuda"), padding=1)[0, 0]
    keep = nb[c1[:, 0], c1[:, 1], c1[:, 2]] > 1.5
X, VOL = XA[keep].contiguous(), VOLA[keep].contiguous()
N = X.shape[0]
print(f"외톨이 입자 {NALL - N} 개를 자유낙하로 (전체 {NALL})", flush=True)
V0 = torch.tensor([0.0, 0.0, -6.0], device="cuda").expand(N, 3).contiguous()   # GF 수박 러너 하드코딩
print(f"[{a.solver}] 입자 {N}, 격자 {n_grid} dx {dx:.5f}, 서브스텝 {sub:.3e} × {nsub}, {NF} 프레임, 재질 {a.material} φ "
      f"{cfg.get('friction_angle', 45)} β {cfg.get('beta')} ξ {cfg.get('xi')} 경화 {cfg.get('hardening')}", flush=True)
os.makedirs(a.out, exist_ok=True)


KP = keep.cpu().numpy(); XD = x[~KP]


def dump(f, xt, Ft):
    """빠진 외톨이는 x0 + v0 t + g t²/2 (상자 안으로 자름), F = I 로 채워 GF 와 같은 입자 순서로 저장."""
    t = f * frame_dt
    xo = np.empty((NALL, 3), np.float32); Fo = np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (NALL, 1))
    xo[KP] = xt; Fo[KP] = Ft.reshape(-1, 9)
    xo[~KP] = np.clip(XD + np.array([0.0, 0.0, -6.0]) * t + 0.5 * np.array(G) * t * t, 3 * dx, GL - 3 * dx)
    with h5py.File(f"{a.out}/sim_{f:010d}.h5", "w") as h:
        h.create_dataset("x", data=xo.T.astype(np.float32))
        h.create_dataset("F", data=Fo.astype(np.float32))


mat = {"material": "sand", "E": E, "nu": nu, "density": rho, "friction_angle": float(cfg.get("friction_angle", 45.0)),
       "g": G, "grid_v_damping_scale": 1.1, "n_grid": n_grid, "grid_lim": GL}
if a.material == "nacc":                                        # GF CD-MPM 그대로 (GF config 값, 없으면 GF 기본값)
    mat.update(material="watermelon", beta=float(cfg.get("beta", 1.0)), xi=float(cfg.get("xi", 0.0)),
               hardening=float(cfg.get("hardening", 0.0)), alpha_0=float(cfg.get("alpha_0", -0.04)))
if a.solver == "pg":
    PG = "/home/dkta/work/i-physgaussian"; sys.path.insert(0, PG); os.chdir(PG)
    import warp as wp
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
    from mpm_solver_warp.mpm_utils import quarantine_nonfinite as QUAR
    wp.init()
    sol = MPM_Simulator_WARP(10)
    sol.load_initial_data_from_torch(X, VOL, None, n_grid=n_grid, grid_lim=GL)
    sol.set_parameters_dict(mat)
    sol.add_bounding_box()
    sol.finalize_mu_lam()
    sol.import_particle_v_from_torch(V0)
    dump(0, x[KP], np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)))
    for f in tqdm(range(1, NF + 1), desc="PG"):
        for s in range(nsub):
            if a.check:
                ms = sol.mpm_state; prev = {k: wp.to_torch(getattr(ms, k)).clone() for k in ("particle_F_trial", "particle_F", "particle_Jp", "particle_C", "particle_v")}
            wp.launch(QUAR, dim=N, inputs=[sol.mpm_state, -0.5 * GL, 1.5 * GL, 1e5])   # GF quarantine 과 같은 자리·같은 문턱
            sol.p2g2p(f, sub)
            if a.check:
                wp.synchronize(); S_ = wp.to_torch(sol.mpm_state.particle_stress).reshape(N, -1)
                bad_ = (~torch.isfinite(S_)).any(1)
                if bad_.any():
                    i = int(bad_.nonzero()[0]); torch.set_printoptions(precision=10)
                    print(f"[첫 비유한 응력] f{f} s{s} 입자 {i}", {k: v[i].tolist() for k, v in prev.items()},
                          "F_after", wp.to_torch(sol.mpm_state.particle_F)[i].tolist(), "Jp_after", float(wp.to_torch(sol.mpm_state.particle_Jp)[i]),
                          "mu", float(wp.to_torch(sol.mpm_model.mu)[i]), "kappa", float(wp.to_torch(sol.mpm_model.kappa)[i]),
                          "beta", float(wp.to_torch(sol.mpm_model.beta)[i]), "M", sol.mpm_model.M, flush=True)
                    torch.save(prev, f"{a.out}/prev_bad.pt"); raise SystemExit(1)
            if a.check and (s % a.check == 0 or s >= 95):
                wp.synchronize(); xx = sol.export_particle_x_to_torch(); FF = sol.export_particle_F_to_torch()
                ms = sol.mpm_state
                for nm in ("particle_v", "particle_stress", "particle_C", "particle_Jp"):
                    tt = wp.to_torch(getattr(ms, nm)).reshape(N, -1)
                    bad = (~torch.isfinite(tt)).any(1)
                    print(f"    {nm} 비유한 {int(bad.sum())} |max| {float(tt.nan_to_num(0).abs().max()):.3g}"
                          + (f" 첫 {int(bad.nonzero()[0])}" if bad.any() else ""), flush=True)
                print(f"  f{f} s{s}: x 비유한 {int((~torch.isfinite(xx)).any(1).sum())} 범위 {float(xx.nan_to_num(0).min()):.3f}~{float(xx.nan_to_num(0).max()):.3f}"
                      f" F 비유한 {int((~torch.isfinite(FF)).reshape(N, -1).any(1).sum())}", flush=True)
        xt = sol.export_particle_x_to_torch().cpu().numpy()
        dump(f, xt, sol.export_particle_F_to_torch().cpu().numpy())
else:
    MP = os.environ.get("AF_MP_ROOT", "/home/dkta/work/MPMAvatar"); sys.path.insert(0, MP); os.chdir(MP)
    import warp as wp
    wp.config.enable_backward = False                            # NACC 필드를 더하면 역전파 커널 인자가 4KB 를 넘는다 -- 순전파만 쓴다
    from warp_mpm.mpm_data_structure import MPMStateStruct, MPMModelStruct
    from warp_mpm.mpm_solver import MPMWARP

    @wp.kernel
    def quar_mpma(state: MPMStateStruct, lo: float, hi: float, vmax: float):
        """GF quarantine: 상자 밖이거나 |v| ≥ vmax (NaN 포함) 인 입자를 시뮬에서 빼고 속도를 지운다."""
        p = wp.tid()
        if state.particle_selection[p] == 0:
            x = state.particle_x[p]
            v = state.particle_v[p]
            ok = (x[0] > lo) and (x[0] < hi) and (x[1] > lo) and (x[1] < hi) and (x[2] > lo) and (x[2] < hi) \
                and (v[0] > -vmax) and (v[0] < vmax) and (v[1] > -vmax) and (v[1] < vmax) and (v[2] > -vmax) and (v[2] < vmax)
            if not ok:
                state.particle_selection[p] = 1
                state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)
    wp.init()
    dev = "cuda:0"
    st = MPMStateStruct(); st.init(N, 0, 0, device=dev, requires_grad=False)
    ones = np.ones(N, np.int32); zeros = np.zeros(N, np.int32)
    st.from_torch(X.clone(), VOL.clone(), torch.zeros(0, 3, 3).cuda(), torch.zeros(0, 3).cuda(), torch.zeros(0, 3).cuda(),
                  ones, zeros, zeros, torch.zeros(N, 6).cuda(), V0.clone(), device=dev, requires_grad=False,
                  n_grid=n_grid, grid_lim=GL)
    md = MPMModelStruct(); md.init(N, device=dev, requires_grad=False); md.init_other_params(n_grid=n_grid, grid_lim=GL, device=dev)
    sol = MPMWARP(N, 0, 0, n_grid=n_grid, grid_lim=GL, device=dev)
    sol.set_parameters_dict(md, st, {k: v for k, v in mat.items() if k not in ("E", "nu", "n_grid", "grid_lim")})
    one = torch.ones(N, device=dev)
    st.reset_state(0, X.clone(), torch.zeros(0, 3, 3).cuda(), None, V0.clone(), device=dev, requires_grad=False)
    st.reset_density((one * rho).clone(), None, dev, update_mass=True)
    _kb = one * 0.0
    if a.material == "nacc":                                     # NACC 의 체적 탄성계수 (PG compute_mu_lam_from_E_nu 와 같은 식)
        _mu, _la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)); _kb = one * (2.0 * _mu / 3.0 + _la)
    sol.set_E_nu_from_torch(md, one * E, one * nu, one * 0.0, _kb, dev)
    sol.prepare_mu_lam(md, st, dev)
    sol.add_bounding_box()
    dump(0, x[KP], np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)))
    for f in tqdm(range(1, NF + 1), desc="MPMAvatar"):
        for s in range(nsub):
            wp.launch(quar_mpma, dim=N, inputs=[st, -0.5 * GL, 1.5 * GL, 1e5], device=dev)
            sol.p2g2p(md, st, sub, device=dev)
        xt = wp.to_torch(st.particle_x).cpu().numpy()
        dump(f, xt, wp.to_torch(st.particle_F).cpu().numpy())
print(f"[저장] {a.out} ({NF + 1} 프레임)", flush=True)
