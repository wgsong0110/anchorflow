"""궤적 하나의 **물리 잔차와 증분 포텐셜**을 프레임마다 잰다.

쓰는 곳: 품질기준 FPS 벤치의 **수렴 궤적** (각 방법이 자기 서브스텝 사다리에서
수렴시킨 것). h5 에 x, v, f_tensor(=F) 가 다 들어 있으므로 F 를 되살릴 필요가
없다 -- 그 방법이 실제로 쓴 변형구배를 그대로 Psi 에 넣는다.

    E(x^{n+1}) = Σ m_p/(2h²)‖Δu_p − h v_p − h² g‖² + Σ V_p Ψ(F_p^{n+1}) + E_bc
    r_p = ∂E/∂x_p · h²/m_p          (길이 단위)

잔차는 **고정 상수**로만 나눈다 (--ext, 기본은 프레임 0 의 물체 지름).
자기 변위로 나누면 프레임마다 기준이 달라져 비교가 안 된다.

부피는 PG/GF 의 `get_particle_volume` 정의 그대로 (셀마다 세고 dx³/개수).

  python exe/ana_traj_resid.py --dir <h5디렉토리> --cfg <bench cfg.json> \
      --out out.json --png out.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow import phys_resid as pr        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="h5 가 든 디렉토리 (simulation_ply)")
ap.add_argument("--cfg", required=True, help="그 실행에 쓴 config json")
ap.add_argument("--out", default="")
ap.add_argument("--png", default="")
ap.add_argument("--ext", type=float, default=0.0, help="잔차 정규화 상수 (기본: 지름)")
ap.add_argument("--label", default="")
ap.add_argument("--dev", default="cuda:0")
a = ap.parse_args()

cfg = json.load(open(a.cfg))
d = a.dir if os.path.basename(a.dir) == "simulation_ply" else \
    os.path.join(a.dir, "simulation_ply")
fs = sorted(glob.glob(os.path.join(d, "*.h5")))
if len(fs) < 2:
    raise SystemExit(f"h5 가 부족하다: {d} ({len(fs)} 개)")


def rd(p):
    with h5py.File(p, "r") as f:
        x = np.array(f["x"]); v = np.array(f["v"]); F = np.array(f["f_tensor"])
    x = x.T if x.shape[0] == 3 else x
    v = v.T if v.shape[0] == 3 else v
    F = (F.T if F.shape[0] == 9 else F).reshape(-1, 3, 3)
    return x.astype(np.float32), v.astype(np.float32), F.astype(np.float32)


dev = torch.device(a.dev if torch.cuda.is_available() else "cpu")
X0, _, _ = rd(fs[0])
N = X0.shape[0]
L = float(np.linalg.norm(X0.max(0) - X0.min(0)))
ext = a.ext if a.ext > 0 else L

# --- 부피/질량: get_particle_volume 정의 그대로 -----------------------
n_grid = int(cfg["n_grid"]); grid_lim = float(cfg["grid_lim"])
dx = grid_lim / n_grid
cell = np.clip(np.floor(X0 / dx).astype(np.int64), 0, n_grid - 1)
flat = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
_uq, _inv, _cnt = np.unique(flat, return_inverse=True, return_counts=True)
vol = torch.as_tensor((dx ** 3) / _cnt[_inv], dtype=torch.float32, device=dev)
mass = vol * float(cfg["density"])
g = torch.as_tensor(cfg.get("g", [0.0, 0.0, -9.8]), dtype=torch.float32,
                    device=dev)
h = float(cfg["frame_dt"])

print(f"[궤적] {d}  입자 {N}  프레임 {len(fs)}  지름 L {L:.4f}\n"
      f"       재질 {cfg.get('material')}  E {cfg.get('E')}  nu {cfg.get('nu')}  "
      f"h(프레임) {h:.5f}  dx {dx:.5f}  부피 평균 {float(vol.mean()):.3e}",
      flush=True)
if pr.mat_name(cfg) == "watermelon":
    pr.cdmpm_reset(N, dev)

rows = []
xn, vn, Fn = rd(fs[0])
xn = torch.as_tensor(xn, device=dev)
vn = torch.as_tensor(vn, device=dev)
for i in tqdm(range(1, len(fs)), desc="프레임"):
    xn1, vn1, Fn1 = rd(fs[i])
    xn1 = torch.as_tensor(xn1, device=dev)
    Fn1t = torch.as_tensor(Fn1, device=dev)
    du = (xn1 - xn).detach().requires_grad_(True)
    E, pl, _F, parts = pr.pts_ip_energy(
        xn, du, vn, Fn1t, None, mass, vol, cfg, h, n_grid, grid_lim, g=g)
    # r_p = |dE/dx_p| * h^2 / m_p  (길이 단위) -> 고정 상수(지름)로 나눈다.
    # phys_resid.residual 은 **정규화된** E 를 받도록 쓰여 있어 그대로 쓰면
    # h^2 만큼(=3600 배) 부풀어 나온다.
    gx, = torch.autograd.grad(E, du)
    r = gx.norm(dim=-1) * (h * h) / mass.clamp_min(1e-20) / ext
    with torch.no_grad():
        rq = torch.quantile(r.float(), torch.tensor([0.5, 0.95, 1.0],
                                                    device=dev))
        rows.append(dict(frame=i,
                         E=float(E), e_in=float(parts[0]), e_el=float(parts[1]),
                         e_g=float(parts[2]), e_bc=float(parts[3]),
                         r_mean=float(r.mean()), r_med=float(rq[0]),
                         r_p95=float(rq[1]), r_max=float(rq[2]),
                         du_mean=float(du.norm(dim=-1).mean())))
    if pl is not None:
        pr.plastic_step(Fn1t, pl)          # 경화 상태만 이어 나른다
    xn, vn = xn1, torch.as_tensor(vn1, device=dev)

E_m = float(np.mean([q["E"] for q in rows]))
r_m = float(np.mean([q["r_med"] for q in rows]))
print(f"[요약] 증분 포텐셜 평균 {E_m:.4e}  (관성 "
      f"{np.mean([q['e_in'] for q in rows]):.3e}, 탄성 "
      f"{np.mean([q['e_el'] for q in rows]):.3e}, 경계 "
      f"{np.mean([q['e_bc'] for q in rows]):.3e})\n"
      f"       물리 잔차 중앙값 평균 {100 * r_m:.4f}% (지름 대비), "
      f"p95 평균 {100 * np.mean([q['r_p95'] for q in rows]):.4f}%", flush=True)

if a.out:
    json.dump(dict(dir=d, cfg=a.cfg, label=a.label, n_particles=N, L=L,
                   ext=ext, h=h, material=cfg.get("material"),
                   E_mean=E_m, r_med_mean=r_m,
                   r_p95_mean=float(np.mean([q["r_p95"] for q in rows])),
                   frames=rows), open(a.out, "w"), indent=1)
    print(f"[저장] {a.out}", flush=True)

if a.png:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = np.array([q["frame"] for q in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(f, [q["E"] for q in rows], "-o", ms=3, label="total")
    ax[0].plot(f, [q["e_in"] for q in rows], "--", label="inertia")
    ax[0].plot(f, [q["e_el"] for q in rows], "--", label="elastic")
    ax[0].plot(f, [q["e_bc"] for q in rows], ":", label="contact")
    ax[0].set_yscale("log"); ax[0].set_xlabel("frame")
    ax[0].set_ylabel("incremental potential"); ax[0].legend(fontsize=7)
    ax[1].plot(f, [100 * q["r_med"] for q in rows], "-o", ms=3, label="median")
    ax[1].plot(f, [100 * q["r_p95"] for q in rows], "--", label="p95")
    ax[1].set_yscale("log"); ax[1].set_xlabel("frame")
    ax[1].set_ylabel("physics residual (% of diameter)"); ax[1].legend(fontsize=7)
    fig.suptitle(a.label or d, fontsize=9)
    fig.tight_layout(); fig.savefig(a.png, dpi=130)
    print(f"[저장] {a.png}", flush=True)
