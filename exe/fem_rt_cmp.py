"""두 실시간 판 실행의 위치 궤적 대조 (결과 불변 확인): 프레임마다 max|x_a - x_b| / EXT, 평균 |x_a - x_b| / EXT.
사용: python exe/fem_rt_cmp.py <폴더 a> <폴더 b>   (pred.pt 또는 roll_fem.pt)"""
import os
import sys

import torch


def load(d):
    f = os.path.join(d, "pred.pt") if os.path.exists(os.path.join(d, "pred.pt")) else os.path.join(d, "roll_fem.pt")
    D = torch.load(f, map_location="cpu", weights_only=False)
    return D["pred"].float(), float(D["EXT"])


A, E = load(sys.argv[1])
B, _ = load(sys.argv[2])
dd = (A - B).norm(dim=-1)
mx = dd.max(1).values / E
mn = dd.mean(1) / E
for t in list(range(0, A.shape[0], 10)) + [A.shape[0] - 1]:
    print(f"프레임 {t + 1}: max {mx[t]:.3e} 평균 {mn[t]:.3e}")
print(f"[대조] 전체 max|Δx|/EXT {float(mx.max()):.3e}, 프레임 평균 |Δx|/EXT 의 평균 {float(mn.mean()):.3e}")
