"""[CM192] stabstat.frame_stats 최적화 (탄성 에너지 SVD 를 GPU float64 로) 전후 같은지: 같은 입력에 STABSTAT_CPU=1 (예전 numpy) 과 GPU 판을 비교.
입력 = 실제 실행의 입자 상태 모양 (ship 495058 입자) 으로 만든 F (회전 x 늘임 + 소성 같은 큰 변형 일부 + det<=0 몇 개).
  python exe/sf_eqtest.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import stabstat as SS
rng = np.random.default_rng(0); N = 495058
q = rng.normal(size=(N, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
w, x, y, z = q.T
R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
              2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).reshape(N, 3, 3)
S = np.exp(rng.normal(scale=0.05, size=(N, 3))); S[:1000] *= rng.uniform(0.2, 3.0, size=(1000, 3)); S[:10, 0] *= -1
F = R @ (S[:, :, None] * np.eye(3)[None]) @ R.transpose(0, 2, 1)
xx = rng.uniform(0.5, 1.5, size=(N, 3)); v = rng.normal(size=(N, 3)); m = rng.uniform(1e-6, 2e-6, N); vol = m / 1000
res = {}
for mode in ("1", "0"):
    os.environ["STABSTAT_CPU"] = mode
    t0 = time.time(); st = SS.frame_stats(xx, v, F, m, np.array([0, 0, -9.8]), 0.5, vol=vol, E=2e6, nu=0.3); res[mode] = (st, time.time() - t0)
a, b = res["1"][0], res["0"][0]
md = max(abs(a[k] - b[k]) / max(abs(a[k]), 1e-300) for k in a)
print("CPU(옛) %.3f s  GPU(새) %.3f s" % (res["1"][1], res["0"][1]))
for k in a:
    print(f"  {k:7s} 옛 {a[k]:.12e}  새 {b[k]:.12e}  상대차 {abs(a[k] - b[k]) / max(abs(a[k]), 1e-300):.2e}")
print("최대 상대차", md, "-> 1e-6 이내" if md < 1e-6 else "-> 넘음")
