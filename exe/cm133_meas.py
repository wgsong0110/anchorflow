"""절 CM133 할 일 1 (로컬, 결과 불변 측정): 예산 안 H9 에서 프레임별 내부 칸 수 · 외부/고립 노드 수 · 외부 질량 몫 · 층 두께.
  - 분류 수·질량 몫: 판의 w8.jsonl (걸음 시작 분류 기록, 실행기 그대로).
  - 층 두께: pred.pt 입자 위치 (torch 없이 exe/cm130_ptload), 수평 2 cm 기둥마다 두께 = 기둥 최고 입자 높이 − 0.48 (바닥), 입자 가중 중앙 · 기둥 안 입자 몫.
  - 기둥별 T1 내부 칸 유무 (esc 덤프가 있는 프레임만): cb (점유 칸) 로 T1 (26 이웃 모두 점유) 다시 계산, 내부 칸 가운데 (A (b + 1 + IP0..2)) 가 든 기둥 표시
    -> 두께 구간별 '내부 칸 없는 기둥의 입자 몫'.
사용: python3 exe/cm133_meas.py <판 디렉토리 (w8.jsonl·pred.pt)> [esc 디렉토리 (fNNN.npz)] > 표
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cm130_ptload as L  # noqa: E402

PZ = 0.48
BIN = 0.02
D = sys.argv[1]
ESC = sys.argv[2] if len(sys.argv) > 2 else None
W8 = [json.loads(l) for l in open(os.path.join(D, "w8.jsonl"))]
X = L.load(os.path.join(D, "pred.pt"))["pred"]
TB = [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 1e9]


def cols(x):
    k = np.floor(x[:, :2] / BIN).astype(np.int64)
    key = (k[:, 0] - k[:, 0].min()) * 100000 + (k[:, 1] - k[:, 1].min())
    u, inv = np.unique(key, return_inverse=True)
    top = np.full(len(u), -1e9)
    np.maximum.at(top, inv, x[:, 2] - PZ)
    return inv, top, k


def interior(cb, IP):
    c = IP[6:9] - 2
    occ = np.zeros(c + 2, bool)
    occ[cb[:, 0] + 1, cb[:, 1] + 1, cb[:, 2] + 1] = True
    ic = occ.copy()
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for d in (-1, 0, 1):
                ic &= np.roll(np.roll(np.roll(occ, a, 0), b, 1), d, 2)
    ic[0, :, :] = ic[-1, :, :] = ic[:, 0, :] = ic[:, -1, :] = ic[:, :, 0] = ic[:, :, -1] = False
    return np.argwhere(ic[1:-1, 1:-1, 1:-1])


print("| f | 내부 칸 | 점유 칸 | 내부 노드 | 외부 노드 (묶임) | 고립 노드 | 질량 몫 내부 · 외부 · 고립 | 두께 입자 중앙 · p10 cm | 두께 < 6 cm 기둥 입자 몫 |")
print("|---|---|---|---|---|---|---|---|---|")
rows = []
for f in range(1, X.shape[0] + 1):
    w = W8[f] if f < len(W8) else W8[-1]
    x = np.asarray(X[f - 1], dtype=np.float64)
    inv, top, _ = cols(x)
    tp = top[inv]
    mi = 1 - w["mf_outer"] - w["mf_iso"]
    r = dict(f=f, icell=w["icell"], ncell=w["ncell"], inner=w["inner"], outer=w["outer"], iso=w["iso"], mf_inner=mi, mf_outer=w["mf_outer"], mf_iso=w["mf_iso"],
             th_med=float(np.median(tp)), th_p10=float(np.quantile(tp, 0.1)), thin6=float((tp < 0.06).mean()))
    rows.append(r)
    if f % 5 == 0 or f in (1, 2, 3, 36, 39, 42, 47):
        print(f"| {f} | {r['icell']} | {r['ncell']} | {r['inner']} | {r['outer']} | {r['iso']} | {mi:.2f} · {r['mf_outer']:.2f} · {r['mf_iso']:.2f} | "
              f"{100*r['th_med']:.1f} · {100*r['th_p10']:.1f} | {r['thin6']:.2f} |")
json.dump(rows, open(os.path.join(os.environ.get("OUT", "."), "cm133_meas.json"), "w"))

if ESC:
    print("\n기둥 두께 구간별 'T1 내부 칸이 하나도 없는 기둥' 의 입자 몫 (esc 덤프, 걸음 시작 p0_x · cb)")
    print("| f | " + " | ".join(f"<{int(100*t)}" if t < 1 else "≥14" for t in TB) + " cm |")
    print("|---|" + "---|" * len(TB))
    for fn in sorted(glob.glob(os.path.join(ESC, "f*.npz"))):
        f = int(os.path.basename(fn)[1:4])
        if f % 5:
            continue
        z = np.load(fn)
        IP, A, cb = z["IP"], z["A"].astype(np.float64), z["cb"]
        x = z["p0_x"].astype(np.float64)
        inv, top, k = cols(x)
        icb = interior(cb, IP)
        cen = (icb + IP[0:3] + 1.0) @ A.T
        kc = np.floor(cen[:, :2] / BIN).astype(np.int64)
        kmin = k.min(0)
        key = (k[:, 0] - kmin[0]) * 100000 + (k[:, 1] - kmin[1])
        keyc = (kc[:, 0] - kmin[0]) * 100000 + (kc[:, 1] - kmin[1])
        has = np.isin(key, keyc)
        tp = top[inv]
        lo = 0.0
        out = []
        for t in TB:
            s = (tp >= lo) & (tp < t)
            out.append(f"{(~has[s]).mean():.2f} ({s.mean():.2f})" if s.any() else "-")
            lo = t
        print(f"| {f} | " + " | ".join(out) + " |")
    print("(괄호 = 그 두께 구간에 든 입자 몫)")
