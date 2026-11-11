"""절 CM106.7 TC ficus 판 (vCF) 입자별 탄성 에너지 밀도 ψ(F_T) (J/m^3) -> psi.npy [T,Np] + 색 colors.npy [T+1,Np,3] (프레임 0 = ψ 0).
ψ = μ Σ(σ_i−1)² + λ/2 (Πσ_i−1)² (jelly 고정 공회전; exe/cell_locked_energy.py psi_sv 와 같은 식, σ = F 특이값, 자르지 않음).
색 범위 = 0 ~ 전체 프레임 유한 ψ 의 99 백분위 (고정, 프레임마다 안 맞춤), 선형 viridis.
  python exe/tetc_en_psi.py PRED.pt CONFIG.json OUTDIR
"""
import json, sys
import numpy as np, torch

pp, cfgp, od = sys.argv[1:4]
cfg = json.load(open(cfgp)); E, nu = float(cfg["E"]), float(cfg["nu"])
mu = E / (2 * (1 + nu)); lam = E * nu / ((1 + nu) * (1 - 2 * nu))
d = torch.load(pp, map_location="cpu", weights_only=False)
F = d["F"]; T, Np = F.shape[0], F.shape[1]
print("프레임", T, "입자", Np, "part", d.get("part"), "diverged", d.get("diverged"), flush=True)
psi = np.empty((T, Np), np.float32)
for t in range(T):
    s = torch.linalg.svdvals(F[t].double())
    J = s.prod(-1)
    psi[t] = (mu * ((s - 1) ** 2).sum(-1) + 0.5 * lam * (J - 1) ** 2).float().numpy()
fin = np.isfinite(psi)
v = psi[fin]
vmax = float(np.percentile(v[::max(1, v.size // 5000000)], 99.0))
st = dict(E=E, nu=nu, mu=mu, lam=lam, frames=T, np=Np, nonfinite=int((~fin).sum()), vmin=0.0, vmax=vmax, p50=float(np.percentile(v[::50], 50)),
          p999=float(np.percentile(v[::50], 99.9)), max=float(v.max()), per_frame_p99=[float(np.percentile(psi[t][fin[t]], 99)) for t in range(T)],
          per_frame_mean=[float(psi[t][fin[t]].mean()) for t in range(T)])
print({k: st[k] for k in ("mu", "lam", "nonfinite", "vmax", "p50", "p999", "max")}, flush=True)
json.dump(st, open(f"{od}/psi_stats.json", "w"))
np.save(f"{od}/psi.npy", psi)
import matplotlib
lut = matplotlib.colormaps["viridis"](np.linspace(0, 1, 1024))[:, :3].astype(np.float32)
col = np.empty((T + 1, Np, 3), np.float32)
col[0] = lut[0]
for t in range(T):
    x = np.nan_to_num(psi[t] / vmax, nan=0.0, posinf=1.0).clip(0, 1)
    col[t + 1] = lut[(x * 1023).astype(np.int64)]
np.save(f"{od}/colors.npy", col)
print("저장", col.shape, flush=True)
