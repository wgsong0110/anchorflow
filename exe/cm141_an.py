"""절 CM141 집계 (CPU): 동시 해제 + 들어올림 자유낙하 판 (R 접두) -- 질량중심 z(f), 프레임 평균 속도·가속도 (m/s, m/s^2), 위쪽 1/4·아래쪽 1/4 층 가속, 입자 F 압력 rms. 자유낙하 해석해 a = -9.8 와 비교.
사용: python exe/cm141_an.py <O> <판 ...>   (pred.pt 는 첫 프레임 제외 50 개 중 --frames 만큼; 첫 프레임 x0 = wolf.pt x[0] + 0.5)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cm130_ptload as L
O = sys.argv[1]; h = 0.04
w = L.load("/dev/shm/dkta/ag_data/wolf.pt") if os.path.exists("/dev/shm/dkta/ag_data/wolf.pt") else L.load("/dev/shm/cm134a/wolf.pt")
x0 = np.array(w["x"][0], dtype=np.float32); x0[:, 2] += 0.5
q = np.quantile(x0[:, 2], [0.25, 0.75]); top = x0[:, 2] > q[1]; bot = x0[:, 2] < q[0]
for n in sys.argv[2:]:
    P = np.array(L.load(os.path.join(O, n, "pred.pt"))["pred"], dtype=np.float32); X = np.concatenate([x0[None], P], 0)
    com = X[:, :, 2].mean(1); ct = X[:, top, 2].mean(1); cb = X[:, bot, 2].mean(1); zmin = X[:, :, 2].min(1)
    v = np.diff(com) / h; a = np.diff(v) / h
    print(f"[{n}] 프레임 {len(X)-1}  질량중심 z {np.round(com,4).tolist()}")
    print(f"   바닥 최저 z {np.round(zmin,3).tolist()}")
    print(f"   v_com {np.round(v,3).tolist()}  (자유낙하 {np.round(-9.8*h*(np.arange(len(v))+0.5),3).tolist()})")
    print(f"   a_com {np.round(a,2).tolist()}  (해석 -9.8)  위층 a {np.round(np.diff(np.diff(ct)/h)/h,2).tolist()}  아래층 a {np.round(np.diff(np.diff(cb)/h)/h,2).tolist()}", flush=True)
