"""절 CM158: 정답·격자 변경 판의 프레임별 질량 중심 높이 (h5 의 모든 입자 z 평균 - 바닥 0.48 m, 입자 질량 같음 (모래 uniform volume)).
사용: python -I cm158_an.py <정답 simulation_ply 디렉토리> <판 상위 디렉토리> 판이름...   판 h5 = <상위>/<판>/simulation_ply/sim_*.h5"""
import sys, os, json
import numpy as np, h5py
gtd, top, *names = sys.argv[1:]
FL = 0.48; FR = (15, 20, 25, 30, 50)
def com(d):
    o = {}; n = 0
    for f in range(51):
        p = os.path.join(d, f"sim_{f:010d}.h5")
        if not os.path.exists(p): continue
        with h5py.File(p, "r") as h: x = np.array(h["x"])
        if x.shape[0] == 3 and x.shape[1] != 3: x = x.T
        o[f] = (float(x[:, 2].mean() - FL) * 1000, float(np.percentile(x[:, 2], 99) - FL) * 1000, x.shape[0], bool(np.isfinite(x).all()))
    return o
res = {"정답": com(gtd)}
for n in names: res[n] = com(os.path.join(top, n, "simulation_ply"))
json.dump(res, open(os.path.join(top, "cm158_com.json"), "w"), ensure_ascii=False, indent=1)
print("질량 중심 높이 (mm, 바닥 0.48 m 기준) / 입자 수")
print("| 판 | 입자 수 | " + " | ".join(f"f{f}" for f in FR) + " |"); print("|---|---|" + "---|" * len(FR))
for k, v in res.items():
    npt = v[0][2] if 0 in v else "-"
    print(f"| {k} | {npt} | " + " | ".join((f"{v[f][0]:.1f}" if f in v else "-") for f in FR) + " |")
print("\n99 백분위 높이 (mm)")
for k, v in res.items(): print(f"| {k} | " + " | ".join((f"{v[f][1]:.1f}" if f in v else "-") for f in FR) + " |")
