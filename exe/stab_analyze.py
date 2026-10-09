"""서브스텝 안정성 실험 정리: 방법별 안정 판정, 기준 궤적(가장 큰 N) 대비 오차, 재현성 바닥, 그림.

  python exe/stab_analyze.py --dir /home/dkta/work/stab/lego
결과: <dir>/summary.md, <dir>/summary.json, <dir>/err_vs_N.png, <dir>/err_vs_time.png
"""
import argparse
import glob
import json
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import stabstat as ss  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
a = ap.parse_args()
D = a.dir
cfg = json.load(open(f"{D}/config.json"))
with h5py.File(f"{D}/init/sim_0000000000.h5") as h:
    x0 = np.array(h["x"]); x0 = x0.T if x0.shape[0] == 3 else x0
L = float(np.linalg.norm(x0.max(0) - x0.min(0)))                 # 고정 정규화 상수 (처음 물체 대각선)
fz = [b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]["point"][2]
gz = -float(cfg["g"][2])
H = float(x0[:, 2].max() - fz)
VCAP = 10.0 * np.sqrt(2 * gz * H)                                # 자유낙하 상한의 10 배
NAMES = {"pg": "PG", "ipg": "i-PG", "gf": "GaussianFluent", "ours": "우리"}
runs = {}
for f in glob.glob(f"{D}/res/*_N*.log"):
    m = re.match(r"(\w+?)_N(\d+)(r?)\.log", os.path.basename(f))
    if not m:
        continue
    meth, n, rep = m.group(1), int(m.group(2)), m.group(3) == "r"
    npz = f[:-4] + ".npz"
    log = open(f, errors="ignore").read()
    r = dict(method=meth, N=n, rep=rep, crashed=not os.path.exists(npz),
             crash_msg=("illegal memory" if "illegal memory" in log else ("오류" if "rror" in log else "")))
    if os.path.exists(npz):
        z = np.load(npz)
        keys = [str(k) for k in z["keys"]]
        rows = [dict(zip(keys, s)) for s in z["stats"]]
        e0 = rows[0]["ke"] + rows[0]["pe"]
        r.update(x=z["x"], t=z["t"], rows=rows, fail=ss.unstable(rows, e0, VCAP), nfr=len(rows) - 1,
                 tpf=float(np.mean(z["t"])) if len(z["t"]) else float("nan"))
    runs[(meth, n, rep)] = r


def rmse(xa, xb):
    k = min(len(xa), len(xb))
    d = np.sqrt(((xa[:k] - xb[:k]) ** 2).sum(-1).mean(-1)) / L        # 프레임별
    return d


out = {"L": L, "vcap": VCAP, "methods": {}}
lines = [f"# 서브스텝 안정성 ({os.path.basename(D)})", "",
         f"L (처음 물체 대각선) {L:.4f}, 속력 상한 {VCAP:.2f}, frame_dt {cfg['frame_dt']:.5f}, 60 프레임", ""]
for meth in ("pg", "gf", "ipg", "ours"):
    Ns = sorted(n for (m, n, rp) in runs if m == meth and not rp)
    if not Ns:
        continue
    ref = runs[(meth, Ns[-1], False)]
    info = dict(N=Ns, ref_N=Ns[-1], rows=[])
    if (meth, Ns[-1], True) in runs and "x" in runs[(meth, Ns[-1], True)] and "x" in ref:
        info["repro_floor"] = float(rmse(ref["x"], runs[(meth, Ns[-1], True)]["x"]).mean())
    if len(Ns) > 1 and "x" in ref and "x" in runs[(meth, Ns[-2], False)]:
        info["conv_half"] = float(rmse(ref["x"], runs[(meth, Ns[-2], False)]["x"]).mean())
    lines += [f"## {NAMES[meth]}  (기준 N = {Ns[-1]}, 재현성 바닥 {100 * info.get('repro_floor', np.nan):.3f}%, "
              f"N/2 와 차이 {100 * info.get('conv_half', np.nan):.3f}%)", "",
              "| N | dt | 안정 | 실패 프레임 | 평균 오차 %L | 마지막 오차 %L | 프레임당 s |", "|---|---|---|---|---|---|---|"]
    for n in Ns:
        r = runs[(meth, n, False)]
        dt = cfg["frame_dt"] / n
        if r["crashed"]:
            row = dict(N=n, stable=False, fail=0, err_mean=None, err_last=None, tpf=None, crash=r["crash_msg"])
            lines.append(f"| {n} | {dt:.2e} | ✗ (중단: {r['crash_msg']}) | 0 | - | - | - |")
        else:
            st = r["fail"] < 0 and r["nfr"] >= 60
            e = rmse(r["x"], ref["x"]) if "x" in ref else np.array([np.nan])
            ok = np.isfinite(e)
            row = dict(N=n, stable=bool(st), fail=int(r["fail"]), err_mean=float(e[ok].mean()) if ok.any() else None,
                       err_last=float(e[-1]) if ok.all() else None, tpf=r["tpf"])
            lines.append(f"| {n} | {dt:.2e} | {'✓' if st else '✗'} | {r['fail'] if r['fail'] >= 0 else '-'} | "
                         f"{100 * row['err_mean']:.3f} | {100 * row['err_last']:.3f} | {r['tpf']:.3f} |"
                         if row["err_mean"] is not None and row["err_last"] is not None else
                         f"| {n} | {dt:.2e} | {'✓' if st else '✗'} | {r['fail'] if r['fail'] >= 0 else '-'} | - | - | {r['tpf']:.3f} |")
        info["rows"].append(row)
    stab = [rw["N"] for rw in info["rows"] if rw["stable"]]
    info["N_min"] = min(stab) if stab else None
    lines += ["", f"최소 안정 N = {info['N_min']}", ""]
    out["methods"][meth] = info
open(f"{D}/summary.md", "w").write("\n".join(lines))
json.dump(out, open(f"{D}/summary.json", "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for meth, info in out["methods"].items():
        rs = [r for r in info["rows"] if r["err_mean"] is not None and r["N"] != info["ref_N"]]
        if rs:
            ax[0].loglog([r["N"] for r in rs], [100 * r["err_mean"] for r in rs], "o-", label=NAMES[meth])
            ax[1].loglog([r["tpf"] for r in rs], [100 * r["err_mean"] for r in rs], "o-", label=NAMES[meth])
        bad = [r["N"] for r in info["rows"] if not r["stable"]]
        for n in bad:
            ax[0].axvline(n, color="0.85", lw=0.5)
    ax[0].set_xlabel("프레임당 서브스텝 N"); ax[0].set_ylabel("기준 대비 평균 오차 (%L)"); ax[0].legend()
    ax[1].set_xlabel("프레임당 시간 (s)"); ax[1].set_ylabel("기준 대비 평균 오차 (%L)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{D}/err_vs_N.png", dpi=110)
except Exception as e:                                           # 그림은 부가물
    print("그림 실패", e)
print("\n".join(lines))
