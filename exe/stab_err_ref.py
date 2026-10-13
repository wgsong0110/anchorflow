"""서브스텝 안정성: 방법마다 자기 기준 N(기본 192) 궤적 대비 각 N 의 위치 오차.

모든 실행이 같은 번호의 고정 부분표본(stabstat.subset, 32768 개)을 60 프레임 저장하므로 같은 입자끼리 비교한다.
오차 = 입자 평균 |x_N − x_ref| / L (L = 형상의 고정 길이: 초기 부분표본의 대각선, 방법과 무관) -- 프레임별, 그 평균·마지막.
60 프레임을 끝내지 못했거나 값이 유한하지 않은 실행(발산)은 뺀다.
"""
import argparse, json, os
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/dkta/work/stab")
ap.add_argument("--ref", type=int, default=192)
ap.add_argument("--out", required=True)
ap.add_argument("--status", default="/home/dkta/work/stab/metrics.json", help="stab_metrics 결과 -- 발산 판정을 같은 기준으로")
ap.add_argument("--emd_n", type=int, default=2048, help="EMD 부분표본 (같은 번호, 모든 실행 공통)")
ap.add_argument("--emd_every", type=int, default=6)
a = ap.parse_args()
NS = (1, 2, 4, 8, 16, 32, 64, 128)


def load(p):
    if not os.path.exists(p):
        return None
    z = np.load(p)
    x = z["x"].astype(np.float64)
    if len(x) < 61 or not np.isfinite(x).all():
        return None
    if "keys" in z.files:
        st = dict(zip([str(k) for k in z["keys"]], z["stats"].T))
        if np.nanmax(st.get("nan", [0])) > 0:
            return None
    return x[:61]


ST = json.load(open(a.status)) if os.path.exists(a.status) else {}
EI = np.sort(np.random.default_rng(0).choice(32768, a.emd_n, replace=False))


def metr(x, ref, L):
    d = np.linalg.norm(x - ref, axis=-1)                      # [T,S] 같은 입자끼리
    mean = d.mean(1) / L
    rmse = np.sqrt((d ** 2).mean(1)) / L
    cd = np.zeros(len(x)); emd = []
    for t in range(1, len(x)):
        ka, kb = cKDTree(x[t]), cKDTree(ref[t])
        cd[t] = 0.5 * (kb.query(x[t])[0].mean() + ka.query(ref[t])[0].mean()) / L
        if t % a.emd_every == 0:
            A, B = x[t][EI], ref[t][EI]
            C = np.linalg.norm(A[:, None] - B[None], axis=-1)
            r, c = linear_sum_assignment(C)
            emd.append(C[r, c].mean() / L)
    return dict(mean=float(mean[1:].mean()), rmse=float(rmse[1:].mean()), cd=float(cd[1:].mean()), emd=float(np.mean(emd)),
                last=float(mean[-1]), rmse_last=float(rmse[-1]), cd_last=float(cd[-1]), emd_last=float(emd[-1]),
                per_frame=[float(v) for v in mean])


res = {}
for S in ("lego", "mic", "ficus"):
    R = f"{a.root}/{S}/res"
    for m in ("pg", "gf", "ipg", "ovs"):
        ref = load(f"{R}/{m}_N{a.ref}.npz")
        if ref is None:
            continue
        L = float(np.linalg.norm(np.ptp(ref[0], 0)))
        for N in NS:
            if N >= a.ref:
                continue
            x, safe = load(f"{R}/{m}_N{N}.npz"), False
            if ST.get(S, {}).get(m, {}).get(str(N), {}).get("status") == "발산":
                x = None
            if x is None:                                     # 원래 실행이 발산했으면 보정판 (그림에서 다르게 표시)
                x, safe = load(f"{R}/{m}_safe_N{N}.npz"), True
            if x is None:
                continue
            r = metr(x, ref, L); r.update(L=L, safe=safe)
            res[f"{S}|{m}|{N}"] = r
            print(S, m, N, "보정판" if safe else "", "평균거리 %.3f%%  RMSE %.3f%%  CD %.3f%%  EMD %.3f%%" % (100 * r["mean"], 100 * r["rmse"], 100 * r["cd"], 100 * r["emd"]), flush=True)
json.dump(res, open(a.out, "w"))
print("ERRREF_DONE")
