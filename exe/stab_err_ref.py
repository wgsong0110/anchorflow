"""서브스텝 안정성: 방법마다 자기 기준 N(기본 192) 궤적 대비 각 N 의 위치 오차.

모든 실행이 같은 번호의 고정 부분표본(stabstat.subset, 32768 개)을 60 프레임 저장하므로 같은 입자끼리 비교한다.
오차 = 입자 평균 |x_N − x_ref| / L (L = 형상의 고정 길이: 초기 부분표본의 대각선, 방법과 무관) -- 프레임별, 그 평균·마지막.
60 프레임을 끝내지 못했거나 값이 유한하지 않은 실행(발산)은 뺀다.
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/dkta/work/stab")
ap.add_argument("--ref", type=int, default=192)
ap.add_argument("--out", required=True)
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
            if x is None:                                     # 원래 실행이 발산했으면 보정판 (그림에서 다르게 표시)
                x, safe = load(f"{R}/{m}_safe_N{N}.npz"), True
            if x is None:
                continue
            e = np.linalg.norm(x - ref, axis=-1).mean(1) / L
            res[f"{S}|{m}|{N}"] = dict(mean=float(e[1:].mean()), last=float(e[-1]), per_frame=[float(v) for v in e], L=L, safe=safe)
            print(S, m, N, "보정판" if safe else "", "평균 %.4f%%  마지막 %.4f%%" % (100 * e[1:].mean(), 100 * e[-1]), flush=True)
json.dump(res, open(a.out, "w"))
print("ERRREF_DONE")
