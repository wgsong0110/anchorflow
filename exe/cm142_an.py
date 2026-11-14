"""절 CM142 집계 (CPU): 앞면 고정 + 들어올림 판 (F 접두; wolf.pt x[0] + 0.5, x > 1.0 해제, 나머지 영구 구속) -- 해제된 쪽 질량중심 가속 (a_com, 해석 -9.8) 을 구속면 거리 띠 (cm) x 높이 층별로.
사용: python exe/cm142_an.py <O> <판 ...>  -> 프레임 1~7 (바닥 닿기 전) 평균 가속도 한 줄씩"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cm130_ptload as L
O = sys.argv[1]; h = 0.04
w = L.load("/dev/shm/dkta/ag_data/wolf.pt") if os.path.exists("/dev/shm/dkta/ag_data/wolf.pt") else L.load("/dev/shm/cm134a/wolf.pt")
x0 = np.array(w["x"][0], dtype=np.float32); x0[:, 2] += 0.5
rel = x0[:, 0] > 1.0; dist = (x0[:, 0] - 1.0) * 100
for n in sys.argv[2:]:
    P = np.array(L.load(os.path.join(O, n, "pred.pt"))["pred"], dtype=np.float32); X = np.concatenate([x0[None], P], 0)
    print(f"[{n}]")
    def acc(m):
        z = X[:, m, 2].mean(1); a = np.diff(np.diff(z) / h) / h; return a
    for nm, m in (("해제 전체", rel), ("구속 몸", ~rel)):
        a = acc(m); print(f"  {nm:8s} n={m.sum():6d} a_com 프레임별 {np.round(a[:8],2).tolist()} 평균(1~6) {a[:6].mean():.2f}")
    for lo, hi in ((0, 3), (3, 6), (6, 11), (11, 21), (21, 99)):
        for zl, zh in ((0.5, 1.2), (1.2, 9)):
            m = rel & (dist >= lo) & (dist < hi) & (x0[:, 2] - 0.5 >= zl - 0.0) & (x0[:, 2] - 0.5 < zh) if False else rel & (dist >= lo) & (dist < hi) & ((x0[:, 2] >= zl + 0.5) & (x0[:, 2] < zh + 0.5))
            if m.sum() < 50: continue
            a = acc(m); print(f"  띠 [{lo},{hi}) cm 높이 [{zl},{zh}) n={m.sum():6d} a 평균(1~6) {a[:6].mean():.2f}", flush=True)
