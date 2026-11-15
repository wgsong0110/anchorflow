"""[substep_fix] 서브스텝 강인성 분석 (CPU): <res 루트>/<TAG>/<장면>_N<N>_<rep>/res.npz (stabhex 형식) 를 읽어
 (1) 완주 (frames_done >= FR, NaN 없음, 운동+위치 에너지 변화 <= 1, 바닥 관통 <= 0.5 L; exe/stab1115_lib.judge 와 같은 정의)
 (2) 기준 궤적 (그 장면에서 완주한 가장 큰 N 의 a 판) 대비 부분표본 RMSE/L 프레임 평균 (N 별, 같은 N 두 판은 따로)
 (3) 같은 N 두 판 사이 RMSE/L (재현성)
-> <out>.json, <out>.md 표 (한국어).  사용: python exe/ss_an.py --res <루트> --tag fix --scenes ficus,vasedeck --stab /home/dkta/work/stab --frames 50 --out /tmp/x"""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--scenes", default="ficus,vasedeck,plane,bread")
ap.add_argument("--stab", default="/home/dkta/work/stab"); ap.add_argument("--frames", type=int, default=50); ap.add_argument("--out", required=True)
ap.add_argument("--ns", default="1,2,4,8,16,32,64,128,256"); ap.add_argument("--cmp_tag", default="", help="N=1 비교 기준 태그 (수정 없는 판)")
a = ap.parse_args()
os.environ["S1115_ROOT"] = a.stab
import stab1115_lib as L  # noqa: E402
Ns = [int(x) for x in a.ns.split(",")]
out = {}
def load(s, n, rep):
    d = f"{a.res}/{a.tag}/{s}_N{n}_{rep}"
    z = d + "/res.npz"
    if not os.path.exists(z):
        return None
    r = L.hex_load(d)
    if r is None:
        return None
    z = np.load(z); fd = int(z["frames_done"])
    ok = fd >= a.frames
    if ok:
        r["X"] = r["X"][: a.frames + 1]
        rows = {k: v[: a.frames + 1] for k, v in r["rows"].items()}
        r["rows"] = rows; r["status"] = "끝" if (np.nanmax(rows["nan"]) == 0 and np.isfinite(r["X"]).all()) else "발산"
        try:
            ok, mt = L.judge(r, s)
        except Exception as e:
            ok, mt = None, {"err": str(e)[:80]}
    else:
        mt = {}
    r["fd"], r["ok"], r["mt"] = fd, bool(ok), mt
    return r
for s in a.scenes.split(","):
    R = {(n, rep): load(s, n, rep) for n in Ns for rep in "ab"}
    ref = None
    for n in sorted(Ns, reverse=True):
        r = R.get((n, "a"))
        if r is not None and r["ok"]:
            ref = (n, r); break
    o = dict(ref_N=ref[0] if ref else None, runs={})
    for n in Ns:
        for rep in "ab":
            r = R[(n, rep)]
            if r is None:
                continue
            e = None
            if r["ok"] and ref is not None:
                X, Y = r["X"], ref[1]["X"]; T = min(len(X), len(Y))
                Lx = float(np.linalg.norm(Y[0].max(0) - Y[0].min(0)))
                e = float(np.mean([np.sqrt(np.mean(np.sum((X[t] - Y[t]) ** 2, -1))) / Lx for t in range(1, T)]))
            tt = None
            try:
                tt = float(np.mean(np.load(f"{a.res}/{a.tag}/{s}_N{n}_{rep}/res.npz")["t"][1:]) * 1000)   # 프레임당 벽시계 ms (그래프 재생 + 기록; 대략)
            except Exception:
                pass
            o["runs"][f"N{n}_{rep}"] = dict(ms=tt, frames_done=r["fd"], ok=r["ok"], stop=r["stop"][:60], err_vs_ref=e, e_drift=r["mt"].get("e_drift_max"), pen=r["mt"].get("pen_depth_max"))
    if a.cmp_tag:                                                                      # N=1: 수정 판 대 수정 없는 판 부분표본 RMSE/L (프레임 평균) 과 프레임 시간
        import copy
        b1 = None
        try:
            zz = np.load(f"{a.res}/{a.cmp_tag}/{s}_N1_a/res.npz"); b1 = zz["x"]; tb = float(np.mean(zz["t"][1:]) * 1000); fb = int(zz["frames_done"])
        except Exception:
            tb = None; fb = None
        r1 = R.get((1, "a"))
        if b1 is not None and r1 is not None and fb >= a.frames:
            T = min(len(b1), len(r1["X"])); Lx = float(np.linalg.norm(b1[0].max(0) - b1[0].min(0)))
            o["n1_vs_base"] = dict(rmse_L=float(np.mean([np.sqrt(np.mean(np.sum((b1[t] - r1["X"][t]) ** 2, -1))) / Lx for t in range(1, T)])), ms_base=tb, ms_fix=o["runs"].get("N1_a", {}).get("ms"))
        elif b1 is None:
            o["n1_vs_base"] = "기준 판 없음 (수정 없는 N=1 이 발산했거나 안 돌림)"
    out[s] = o
json.dump(out, open(a.out + ".json", "w"), ensure_ascii=False, indent=1)
with open(a.out + ".md", "w") as f:
    for s, o in out.items():
        f.write(f"\n### {s} (기준 궤적 N={o['ref_N']})\n\n| N | 판 a | 판 b | 오차 a (%L) | 오차 b (%L) | 프레임당 ms (a) |\n|---|---|---|---|---|---|\n")
        for n in Ns:
            ra, rb = o["runs"].get(f"N{n}_a"), o["runs"].get(f"N{n}_b")
            c = lambda r: "-" if r is None else (f"완주" if r["ok"] else f"발산 (프레임 {r['frames_done']})")
            e = lambda r: "-" if (r is None or r["err_vs_ref"] is None) else f"{100 * r['err_vs_ref']:.2f}"
            ms_ = "-" if (ra is None or ra.get("ms") is None) else f"{ra['ms']:.0f}"
            f.write(f"| {n} | {c(ra)} | {c(rb)} | {e(ra)} | {e(rb)} | {ms_} |\n")
        if "n1_vs_base" in o:
            f.write(f"\nN=1 수정 판 대 수정 없는 판: {o['n1_vs_base']}\n")
print(open(a.out + ".md").read())
