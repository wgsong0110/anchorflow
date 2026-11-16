"""[stab1115] 분석 (한 씬, CPU): 사다리·확정 판 -> 판정, 기준 궤적, N-오차 곡선, 확정 N 물리 정합성 -> an_<씬>.json.
정의는 exe/stabhex_an.py 와 같다 (판정: 60 프레임 끝 + 에너지 변화 <= 1 + 관통 <= 0.5 L; 기준 = 2 배 짝 RMSE/L < 0.5 %; 물리 문턱 exe/stab_scan_table.py).
  python exe/stab1115_an.py --scene ficus --out an_ficus.json
"""
import argparse, glob, json, os, sys
import numpy as np
import h5py
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stab1115_lib as L
import stab_metrics as sm
from anchorflow import stabstat as ss

ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--tol", type=float, default=0.005); ap.add_argument("--hextag", default="")
a = ap.parse_args()
S = a.scene; ST = L.ST
cfg, zf, g, dt, dxg = L.scene_consts(S); FR = L.nframes(S)
with h5py.File(f"{ST}/{S}/init/sim_0000000000.h5") as h:
    x0 = np.array(h["x"]); x0 = x0.T if x0.shape[0] == 3 else x0
sub = ss.subset(len(x0)); Lc = float(np.linalg.norm(x0[sub].max(0) - x0[sub].min(0)))
H = float(x0[:, 2].max() - zf); vcap = 10.0 * np.sqrt(2 * abs(g) * H)
Mtot = float(cfg["density"]) * dxg ** 3 * len(np.unique(np.floor(x0 / dxg).astype(np.int64), axis=0))
refr = L.load(S, "pg", 1024, "a")
PREF = refr["X"]; LP = float(np.linalg.norm(PREF[0].max(0) - PREF[0].min(0)))
REFM = sm.metrics(refr, PREF, LP, zf, g, dt, Mtot, dxg)
THR = dict(detneg=0.0, pen_mult=2.0, pen_floor=0.01, jit_mult=2.0, ff=0.05, cd=0.05)
SO = dict(scene=S, L=Lc, n_particles=int(len(x0)), floor_z=zf, phys_ref=REFM, methods={})


def rmse(XA, XB):
    T = min(len(XA), len(XB))
    return np.sqrt(((XA[:T].astype(np.float64) - XB[:T].astype(np.float64)) ** 2).sum(-1).mean(-1)) / Lc


def all_ns(m):
    ns = set()
    for rep in L.REPS:
        if m == "ours":
            for p in glob.glob(f"{ST}/{S}/hex/{S}_N*_{rep}{a.hextag}"):
                try: ns.add(int(os.path.basename(p).split("_N")[1].split("_")[0]))
                except Exception: pass
        else:
            for p in glob.glob(L.res_dir(S, rep) + f"/{m}_N*.fin"):
                ns.add(int(os.path.basename(p)[len(m) + 2:-4]))
    return sorted(ns)


def loadrep(m, n, rep):
    if m == "ours":
        return L.hex_load(L.hex_dir(S, n, rep) + a.hextag, FR)
    return L.load(S, m, n, rep)


for m in ("pg", "ipg", "gf", "omni", "ours"):
    ns = all_ns(m)
    if not ns:
        continue
    M = dict(runs={}, ref=None, ref_pairs=[], err={}, floor={}, N_min=None)
    byN = {}
    for n in ns:
        for rep in L.REPS:
            r = loadrep(m, n, rep)
            if r is None or r["status"] == "진행":
                continue
            ok, mt = L.judge(r, S)
            rec = dict(N=n, rep=rep, status=r["status"], stable=ok, metrics={k: (float(v) if isinstance(v, (int, float, np.floating)) and v is not None else v) for k, v in mt.items()})
            if r["status"] == "끝":
                mt2 = sm.metrics(r, PREF, LP, zf, g, dt, Mtot, dxg)
                rec["phys"] = {
                    "detneg": [mt2.get("detneg_max"), (mt2.get("detneg_max") or 0.0) <= THR["detneg"]],
                    "pen": [mt2.get("pen_depth_max"), (mt2.get("pen_depth_max") or 0.0) <= max(THR["pen_mult"] * (REFM.get("pen_depth_max") or 0.0), THR["pen_floor"])],
                    "jit": [mt2.get("jitter_d2"), (mt2.get("jitter_d2") or 0.0) <= THR["jit_mult"] * REFM["jitter_d2"]],
                    "ff": [mt2.get("freefall_mom_err"), None if mt2.get("freefall_mom_err") is None else mt2["freefall_mom_err"] <= THR["ff"]],
                    "cd": [mt2.get("cd_mpm"), (mt2.get("cd_mpm") or 1.0) <= THR["cd"]]}
            M["runs"][f"{m}_N{n}_{rep}"] = rec
            byN.setdefault(n, []).append((rep, r, rec))
    st = {}
    for n, lst in byN.items():
        if all(rec["status"] != "메모리부족" for _, _, rec in lst):
            st[n] = [bool(rec["stable"]) for _, _, rec in lst]
    M["stable_by_N"] = {n: [int(sum(v)), len(v)] for n, v in sorted(st.items())}
    good = [n for n, v in sorted(st.items()) if all(v) and len(v) >= (3 if m != "x" else 1)]
    pl = f"{ST}/{S}/plan_{m}{a.hextag if m == 'ours' else ''}.json"
    if os.path.exists(pl):
        M["N_min"] = json.load(open(pl)).get("N_min")
    if M["N_min"] is None and good:
        M["N_min"] = good[0]
    fin = {n: [r for rep, r, rec in lst if rep == "a"][0]["X"] for n, lst in byN.items()
           if any(rep == "a" and r["status"] == "끝" and np.isfinite(r["X"]).all() for rep, r, rec in lst) and (n & (n - 1)) == 0}
    Ns = sorted(fin)
    for n in Ns:
        if 2 * n in fin:
            d = float(rmse(fin[n], fin[2 * n])[1:FR + 1].mean()); M["ref_pairs"].append([n, 2 * n, d])
            if M["ref"] is None and d < a.tol:
                M["ref"] = 2 * n
    M["ref_converged"] = M["ref"] is not None
    if M["ref"] is None and Ns:
        c_ = Ns[0]
        while 2 * c_ in fin:
            c_ *= 2
        M["ref"] = c_
    if M["ref"] is not None:
        XR = fin[M["ref"]]
        for n in Ns:
            e = rmse(fin[n], XR); M["err"][n] = dict(mean=float(e[1:FR + 1].mean()), last=float(e[min(FR, len(e) - 1)]))
    for n, lst in byN.items():                               # 재현성 바닥: 같은 N 의 끝난 두 판
        ok_ = [r for rep, r, rec in lst if r["status"] == "끝"]
        if len(ok_) >= 2:
            e = rmse(ok_[0]["X"], ok_[1]["X"]); M["floor"][n] = dict(mean=float(e[1:FR + 1].mean()), max=float(e[1:FR + 1].max()))
    SO["methods"][m] = M
    print(S, m, "N_min", M["N_min"], "기준", M["ref"], "수렴" if M["ref_converged"] else "미수렴", flush=True)


def _j(o):
    if isinstance(o, dict): return {str(k): _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_j(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.bool_): return bool(o)
    return o


json.dump(_j(SO), open(a.out, "w"), ensure_ascii=False)
print("[stab1115 분석] ->", a.out)
