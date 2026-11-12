"""서브스텝 안정성 '우리(육면체)' 분석 (11-12 지시, 묶음 stabhex; 노드 CPU). 결과 JSON 하나 -> 표·그림은 exe/stabhex_tab.py.

- 안정 판정 (N_min 표, 기존 표와 같은 정의 = exe/stab_is_stable.py): 60 프레임 끝 (NaN 없음) + 운동+위치 에너지 변화 최대 ≤ 1 + 바닥 관통 깊이 ≤ 0.5 L
  (exe/stab_metrics.py metrics 그대로, 탄성 에너지 열은 빼고). 참고로 고정 문턱 판정 stabstat.unstable 의 첫 프레임도 남김 (doc/stab_plan.md).
- 기준 궤적 (doc/stab_plan.md): 방법마다 끝난 N (2 의 거듭제곱) 을 두 배씩 올려 N 과 2N 의 부분표본 RMSE/L (프레임 평균) < 0.5 % 가 되는 첫 2N.
- 오차: 기준 대비 부분표본 (32768, stabstat.subset 같은 번호) RMSE/L -- 프레임별·평균 (프레임 1~60). L = 처음 부분표본 대각선.
- 재현성 바닥: 같은 설정 두 번 (우리(육면체) a·b, PG·GF N1024 와 N1024r).
기존 방법 (PG·GF·i-PG·우리(사면체) ovs·rep_ip ours) 은 /home/dkta/work/stab/<형상>/res 의 결과를 그대로 읽음 (다시 안 돌림).

  python exe/stabhex_an.py --root /home/dkta/work/stab --hex /home/dkta/work/stabhex --jobs exe/stabhex_jobs.txt --out stabhex_an.json
"""
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
import stab_metrics as sm                                          # noqa: E402
from anchorflow import stabstat as ss                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/dkta/work/stab")
ap.add_argument("--hex", default="/dev/shm/dkta/stabhex/work")
ap.add_argument("--jobs", default="exe/stabhex_jobs.txt")
ap.add_argument("--out", required=True)
ap.add_argument("--tol", type=float, default=0.005)
a = ap.parse_args()
POW = [2 ** i for i in range(12)]
OLD = ("pg", "gf", "ipg", "ovs", "ours")
JOBS = [l.split() for l in open(a.jobs) if l.strip() and not l.startswith("#")]


def hex_load(d):
    """육면체 판 폴더 -> stab_metrics.load 와 같은 dict (+ 반복 수·끝 까닭)"""
    p = f"{d}/res.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    keys = [str(k) for k in z["keys"]]
    rows = {k: z["stats"][:, i] for i, k in enumerate(keys)}
    st = "끝" if (len(z["x"]) >= 61 and np.nanmax(rows["nan"]) == 0 and np.isfinite(z["x"]).all()) else "발산"
    stop = str(z["stop"]) if "stop" in z.files else ""
    if st == "발산" and "out of memory" in stop and np.isfinite(z["x"]).all() and np.nanmax(rows["nan"]) == 0:
        st = "메모리부족"                                           # 큰 N 의 프레임 CUDA 그래프 캡처가 GPU 메모리 부족 -- 발산 아님 (안정 판정·N_min 에서 뺌)
    it = z["it"] if "it" in z.files else np.zeros((0, 6))
    r = dict(status=st, rows=rows, X=z["x"], frames=int(len(z["x"]) - 1), stop=str(z["stop"]) if "stop" in z.files else "",
             newton=int(z["newton"]), cg=int(z["cg"]), tolN=float(z["tolN"]), tolP=float(z["tolP"]), K=int(z["substeps"]))
    if len(it):
        r["it"] = dict(nN_mean=float(it[:, 0].mean()), nP_mean=float(it[:, 1].mean()), nN_max=float(it[:, 0].max()), nP_max=float(it[:, 1].max()),
                       capN_frames=int((it[:, 2] > 0).sum()), capP_frames=int((it[:, 3] > 0).sum()), capN_sum=float(it[:, 2].sum()), capP_sum=float(it[:, 3].sum()),
                       steps_mean=float(it[:, 4].mean()), solves_mean=float(it[:, 5].mean()))
    sj = f"{d}/summary.json"
    if os.path.exists(sj):
        s = json.load(open(sj))
        r["summary"] = {k: s.get(k) for k in ("frames_done", "diverged", "ms_mean", "ms_p90", "x_sha1")}
    return r


def judge(r, zf, g, dt, dxg, vcap):
    """-> (표 판정 안정, 지표 dict, stabstat 첫 불안정 프레임)"""
    if r is None:
        return None, {}, None
    rows = r["rows"]
    unst = None
    if rows is not None:
        T = len(rows["ke"])
        R = [{k: float(rows[k][t]) for k in rows} for t in range(T)]
        e0 = R[0]["ke"] + R[0]["pe"]
        unst = ss.unstable(R, e0, vcap)
    if r["status"] != "끝":
        return False, {}, unst
    X = r["X"]
    L = float(np.linalg.norm(X[0].max(0) - X[0].min(0)))
    rr = dict(r); rr["rows"] = {k: v for k, v in rows.items() if k != "ee"} if rows is not None else None
    mt = sm.metrics(rr, None, L, zf, g, dt, 1.0, dxg)
    ok = bool(np.isfinite(mt.get("e_drift_max", 0.0)) and mt.get("e_drift_max", 0.0) <= 1.0 and (mt.get("pen_depth_max") or 0.0) <= 0.5)
    return ok, {k: (float(v) if isinstance(v, (int, float, np.floating)) and v is not None else v) for k, v in mt.items()}, unst


def rmse(XA, XB, L):
    T = min(len(XA), len(XB))
    e = np.sqrt(((XA[:T].astype(np.float64) - XB[:T].astype(np.float64)) ** 2).sum(-1).mean(-1)) / L
    return e


OUT = dict(tol=a.tol, shapes={})
for S in ("lego", "mic", "ficus"):
    res = f"{a.root}/{S}/res"
    cfg = json.load(open(f"{a.root}/{S}/config.json"))
    zf = float([b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]["point"][2])
    g = float(cfg["g"][2]); dt = float(cfg["frame_dt"]); dxg = float(cfg["grid_lim"]) / int(cfg["n_grid"])
    with h5py.File(f"{a.root}/{S}/init/sim_0000000000.h5") as h:
        x0 = np.array(h["x"]); x0 = x0.T if x0.shape[0] == 3 else x0
    sub = ss.subset(len(x0))
    L = float(np.linalg.norm(x0[sub].max(0) - x0[sub].min(0)))
    H = float(x0[:, 2].max() - zf)
    vcap = 10.0 * np.sqrt(2 * abs(g) * H)
    SO = dict(L=L, n_particles=int(len(x0)), floor_z=zf, vcap=vcap, methods={})
    Mtot = float(cfg["density"]) * dxg ** 3 * len(np.unique(np.floor(x0 / dxg).astype(np.int64), axis=0))     # exe/stab_phys_confirmed.py 와 같음
    refr = sm.load(res, "pg", 1024)
    PREF = refr["X"]; LP = float(np.linalg.norm(PREF[0].max(0) - PREF[0].min(0)))
    REFM = sm.metrics(refr, PREF, LP, zf, g, dt, Mtot, dxg)
    SO["phys_ref"] = {k: v for k, v in REFM.items()}
    THR = dict(detneg=0.0, pen_mult=2.0, pen_floor=0.01, jit_mult=2.0, ff=0.05, cd=0.05)                    # exe/stab_scan_table.py 문턱
    runs = {}                                                       # 방법 -> {N: [판 dict ...]} (오차는 첫 판)
    for m in OLD:
        runs[m] = {}
        for n in POW:
            r = sm.load(res, m, n)
            if r is not None:
                runs[m].setdefault(n, []).append(dict(r=r, name=f"{m}_N{n}"))
            rr = f"{res}/{m}_N{n}r.npz"
            if os.path.exists(rr):
                z = np.load(rr); keys = [str(k) for k in z["keys"]]
                rows = {k: z["stats"][:, i] for i, k in enumerate(keys)}
                st = "끝" if (len(z["x"]) >= 61 and np.nanmax(rows["nan"]) == 0) else "발산"
                runs[m].setdefault(n, []).append(dict(r=dict(status=st, rows=rows, X=z["x"]), name=f"{m}_N{n}r"))
    runs["hex"] = {}
    for n_, s_, k_, rep, dp in JOBS:
        if s_ != S:
            continue
        r = hex_load(f"{a.hex}/{n_}")
        if r is not None:
            runs["hex"].setdefault(int(k_), []).append(dict(r=r, name=n_, rep=rep))
    for m, byN in runs.items():
        M = dict(runs={}, ref=None, ref_pairs=[], err={}, floor={})
        for n, lst in sorted(byN.items()):
            for it in lst:
                ok, mt, un = judge(it["r"], zf, g, dt, dxg, vcap)
                rec = dict(N=n, status=it["r"]["status"], stable=ok, unstable_frame=un, metrics=mt)
                if m == "hex" and it["r"]["status"] == "끝":                # 물리 정합성 (exe/stab_phys_confirmed.py 와 같은 지표·문턱, 기준 PG N1024)
                    mt2 = sm.metrics(it["r"], PREF, LP, zf, g, dt, Mtot, dxg)
                    rec["phys"] = {
                        "detneg": [mt2.get("detneg_max"), (mt2.get("detneg_max") or 0.0) <= THR["detneg"]],
                        "pen": [mt2.get("pen_depth_max"), (mt2.get("pen_depth_max") or 0.0) <= max(THR["pen_mult"] * (REFM.get("pen_depth_max") or 0.0), THR["pen_floor"])],
                        "jit": [mt2.get("jitter_d2"), (mt2.get("jitter_d2") or 0.0) <= THR["jit_mult"] * REFM["jitter_d2"]],
                        "ff": [mt2.get("freefall_mom_err"), None if mt2.get("freefall_mom_err") is None else mt2["freefall_mom_err"] <= THR["ff"]],
                        "cd": [mt2.get("cd_mpm"), (mt2.get("cd_mpm") or 1.0) <= THR["cd"]]}
                for k in ("frames", "stop", "newton", "cg", "tolN", "tolP", "K", "it", "summary", "rep"):
                    if k in it["r"] or k in it:
                        rec[k] = it["r"].get(k, it.get(k))
                M["runs"][it["name"]] = rec
        fin = {n: lst[0]["r"]["X"] for n, lst in byN.items() if lst[0]["r"]["status"] == "끝" and np.isfinite(lst[0]["r"]["X"]).all()}
        Ns = sorted(fin)
        for n in Ns:                                                # 두 배 짝 차이
            if 2 * n in fin:
                d = float(rmse(fin[n], fin[2 * n], L)[1:61].mean())
                M["ref_pairs"].append([n, 2 * n, d])
                if M["ref"] is None and d < a.tol:
                    M["ref"] = 2 * n
        M["ref_converged"] = M["ref"] is not None
        if M["ref"] is None and Ns:
            M["ref"] = Ns[-1]
        if M["ref"] is not None:
            XR = fin[M["ref"]]
            for n in Ns:
                e = rmse(fin[n], XR, L)
                M["err"][n] = dict(mean=float(e[1:61].mean()), last=float(e[min(60, len(e) - 1)]), frames=[float(q) for q in e[:61]])
        for n, lst in byN.items():                                  # 재현성 바닥: 같은 N 두 판
            ok_ = [it for it in lst if it["r"]["status"] == "끝" and it.get("rep") != "d"]
            if len(ok_) >= 2:
                e = rmse(ok_[0]["r"]["X"], ok_[1]["r"]["X"], L)
                M["floor"][n] = dict(a=ok_[0]["name"], b=ok_[1]["name"], mean=float(e[1:61].mean()), max=float(e[1:61].max()))
        if m == "hex":                                              # 그래프 없이 띄운 확인 판 (d) 대 같은 N 그래프 판 (a)
            M["direct_check"] = {}
            for n, lst in byN.items():
                da = [it for it in lst if it.get("rep") == "a" and it["r"]["status"] == "끝"]
                dd = [it for it in lst if it.get("rep") == "d" and it["r"]["status"] == "끝"]
                if da and dd:
                    e = rmse(da[0]["r"]["X"], dd[0]["r"]["X"], L)
                    M["direct_check"][n] = dict(mean=float(e[1:61].mean()), max=float(e[1:61].max()))
        if m == "hex":                                              # N_min: 그 N 의 모든 판이 안정한 가장 작은 N (판 수 함께)
            st = {}
            for nm, rec in M["runs"].items():
                if rec["status"] != "메모리부족":
                    st.setdefault(rec["N"], []).append(bool(rec["stable"]))
            M["stable_by_N"] = {n: [int(sum(v)), len(v)] for n, v in sorted(st.items())}
            good = [n for n, v in sorted(st.items()) if all(v)]
            M["N_min"] = good[0] if good else None
            M["N_min_runs"] = st.get(M["N_min"], []) if good else []
            M["unstable_above_Nmin"] = [n for n, v in st.items() if good and n > good[0] and not all(v)]
        SO["methods"][m] = M
        print(S, m, "기준", M["ref"], "수렴" if M["ref_converged"] else "미수렴", "짝", [(p[0], round(100 * p[2], 3)) for p in M["ref_pairs"]],
              "N_min", M.get("N_min"), flush=True)
    OUT["shapes"][S] = SO


def _j(o):
    if isinstance(o, dict):
        return {str(k): _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(_j(OUT), open(a.out, "w"), ensure_ascii=False)
print("[stabhex 분석] ->", a.out, flush=True)
