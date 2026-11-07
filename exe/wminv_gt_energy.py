"""정답 에너지 증가 가르기 (읽기 전용, numpy+h5py). 수박 (GF) f0~f8 + 레고 (PG) 전 프레임. 출력 /tmp/wminv_energy/*.json"""
import json, os, sys
import numpy as np, h5py

R2 = os.path.expanduser("~/r2/result/anchorflow/_vastmirror")
OUT = "/tmp/wminv_energy"; os.makedirs(OUT, exist_ok=True)

def cell_vol(x, dx):
    idx = np.floor(x.astype(np.float64) / dx).astype(np.int64)
    key = (idx[:, 0] * 100003 + idx[:, 1]) * 100003 + idx[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return dx ** 3 / cnt[inv], cnt

def lame(E, nu): return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

def snh(F, mu, lam):
    J = np.linalg.det(F); Ic = (F * F).sum((1, 2))
    return 0.5 * mu * (Ic - 3) - mu * (J - 1) + 0.5 * (lam + mu) * (J - 1) ** 2, J, Ic

def nhb(J, Ic, mu, lam):
    kap = 2 * mu / 3 + lam; ok = J > 0; Jp = np.where(ok, J, 1.0)
    return np.where(ok, 0.5 * mu * (Jp ** (-2 / 3) * Ic - 3) + 0.25 * kap * (Jp ** 2 - 1 - 2 * np.log(Jp)), 0.0)

def fcr(F, mu, lam):
    s = np.linalg.svd(F, compute_uv=False); J = np.linalg.det(F)
    return mu * ((s - 1) ** 2).sum(1) + 0.5 * lam * (J - 1) ** 2

def common(m, v, x, g, V, psi, J):
    ok = np.isfinite(x).all(1) & np.isfinite(v).all(1)
    ke = 0.5 * (m[ok] * (v[ok].astype(np.float64) ** 2).sum(1)).sum()
    pe = -(m[ok] * (x[ok].astype(np.float64) @ g)).sum()
    return ok, ke, pe

# ---------------- 수박 ----------------
def watermelon():
    d = R2 + "/wm2/gf"; cfg = json.load(open(d + "/config.json"))
    mu, lam = lame(cfg["E"], cfg["nu"]); g = np.array(cfg["g"], float); rho = cfg["density"]
    dx = 2.0 / cfg["n_grid"]  # grid_lim 기본 2.0 (decode_param.py:20)
    ld = lambda f, k, n: np.ascontiguousarray(h5py.File(f"{d}/simulation_ply/sim_{f:010d}.h5")[k][:].T).reshape(n)
    x0 = np.ascontiguousarray(h5py.File(f"{d}/simulation_ply/sim_0000000000.h5")["x"][:].T)
    V, cnt = cell_vol(x0, dx); m = V * rho
    res = dict(mu=mu, lam=lam, kappa=2 * mu / 3 + lam, N=len(m), dx=dx, M=float(m.sum()), Vsum=float(V.sum()),
               cnt_stats=[int(cnt.min()), float(cnt.mean()), int(cnt.max())], frames=[])
    for f in range(9):
        h = h5py.File(f"{d}/simulation_ply/sim_{f:010d}.h5")
        F = np.ascontiguousarray(h["f_tensor"][:].T).reshape(-1, 3, 3).astype(np.float64)
        x = np.ascontiguousarray(h["x"][:].T); v = np.ascontiguousarray(h["v"][:].T); t = float(np.array(h["time"]).ravel()[0])
        psn, J, Ic = snh(F, mu, lam); pnh = nhb(J, Ic, mu, lam)
        ok, ke, pe = common(m, v, x, g, V, psn, J)
        z = x[ok, 2]; xs = x[ok, 0]; ys = x[ok, 1]
        q = lambda a: [float(c) for c in np.percentile(a, [1, 50, 99])]
        xd = x[ok].astype(np.float64)
        r = dict(f=f, time=t, KE=float(ke), PE=float(pe), Esnh=float((V * psn)[ok].sum()), Enhb=float((V * pnh)[ok & (J > 0)].sum()),
                 Jq=[float(c) for c in np.percentile(J[ok], [1, 50, 99])], Jmean_volw=float((V * J)[ok].sum() / V[ok].sum()),
                 zq=q(z), zmin=float(z.min()), xq=q(xs), yq=q(ys),
                 wx=float(np.percentile(xs, 99) - np.percentile(xs, 1)), wy=float(np.percentile(ys, 99) - np.percentile(ys, 1)),
                 band003=float((z < 0.03).mean()), zc=float((m[ok] * z).sum() / m[ok].sum()),
                 vzc=float((m[ok] * v[ok, 2]).sum() / m[ok].sum()),
                 KEz=float(0.5 * (m[ok] * v[ok, 2].astype(np.float64) ** 2).sum()), n_Jle0=int((J[ok] <= 0).sum()))
        r["height"] = r["zq"][2] - r["zq"][0]; r["tot_snh"] = r["KE"] + r["PE"] + r["Esnh"]; r["tot_nhb"] = r["KE"] + r["PE"] + r["Enhb"]
        res["frames"].append(r)
        print(f"wm f{f} t{t:.3f} KE {ke:.3f} PE {pe:.3f} SNH {r['Esnh']:.3f} NHB {r['Enhb']:.3f} tot_snh {r['tot_snh']:.3f} Jq {r['Jq']} h {r['height']:.4f} w {r['wx']:.3f},{r['wy']:.3f} band {r['band003']:.3f} zc {r['zc']:.4f} vz {r['vzc']:.3f}", flush=True)
    json.dump(res, open(OUT + "/wm_gt_energy.json", "w"), indent=1)

# ---------------- 레고 ----------------
def lego():
    d = R2 + "/lbdata/test_low/L000"; cfg = json.load(open(R2 + "/lbdata/config.json"))
    mu, lam = lame(cfg["E"], cfg["nu"]); g = np.array(cfg["g"], float); rho = cfg["density"]
    dx = cfg["grid_lim"] / cfg["n_grid"]; zf = 0.48
    h0 = h5py.File(f"{d}/sim_0000000000.h5"); x0 = h0["x"][:]; V, cnt = cell_vol(x0, dx); m = V * rho
    F0 = h0["F"][:].reshape(-1, 3, 3).astype(np.float64)
    res = dict(mu=mu, lam=lam, N=len(m), dx=dx, M=float(m.sum()), Vsum=float(V.sum()), cnt_stats=[int(cnt.min()), float(cnt.mean()), int(cnt.max())],
               F0_minus_I_max=float(np.abs(F0 - np.eye(3)).max()), F0_detq=[float(c) for c in np.percentile(np.linalg.det(F0), [0, 50, 100])],
               F0_eq_R=bool(np.allclose(F0, np.array(json.load(open(d + "/meta.json"))["R"]), atol=1e-5)), frames=[])
    for f in range(81):
        h = h5py.File(f"{d}/sim_{f:010d}.h5")
        F = h["F"][:].reshape(-1, 3, 3).astype(np.float64); x = h["x"][:]; v = h["v"][:]
        psn, J, Ic = snh(F, mu, lam); pfc = fcr(F, mu, lam)
        ok, ke, pe = common(m, v, x, g, V, psn, J)
        z = x[:, 2].astype(np.float64)
        r = dict(f=f, KE=float(ke), PE=float(pe), Esnh=float((V * psn).sum()), Efcr=float((V * pfc).sum()),
                 zmin=float(z.min()), zc=float((m * z).sum() / m.sum()), vzc=float((m * v[:, 2]).sum() / m.sum()),
                 nband=int((z < zf + 0.03).sum()), nbelow=int((z < zf).sum()), Jq=[float(c) for c in np.percentile(J, [1, 50, 99])])
        r["tot_fcr"] = r["KE"] + r["PE"] + r["Efcr"]; r["tot_snh"] = r["KE"] + r["PE"] + r["Esnh"]
        res["frames"].append(r)
        print(f"lego f{f} KE {r['KE']:.3f} PE {r['PE']:.3f} SNH {r['Esnh']:.3f} FCR {r['Efcr']:.3f} tot_fcr {r['tot_fcr']:.3f} zmin {r['zmin']:.4f} nband {r['nband']} vzc {r['vzc']:.3f} Jq {r['Jq']}", flush=True)
    json.dump(res, open(OUT + "/lego_pg_energy.json", "w"), indent=1)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "wm"): watermelon()
    if which in ("all", "lego"): lego()
