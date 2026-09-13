"""학습 대상 궤적에서 위치·회전·신축의 **변화량 분포**를 견준다.

학생의 출력 스케일(DISP_SCALE / DU_SCALE / DS_SCALE)은 연속 프레임 차이의 전체
평균이다. 회전은 임펄스 직후 0 -> 0.44 rad 로 뛰고 그 뒤로는 훨씬 천천히 변하므로,
첫 스텝이 평균을 끌어올리면 망의 출력이 그 배수만큼 과하게 스케일된다 -- 롤아웃에서
움직임이 107~144% 로 과잉인 것의 후보다.

상태 자체의 크기도 함께 낸다. |u_a| 가 2pi 를 넘으면 회전벡터가 감긴 것이라 정준형이
아니고, 블렌드가 u_a 에 선형이므로 감긴 큰 값들이 섞이면 결과가 달라진다.
"""
from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--max_traj", type=int, default=60)
args = ap.parse_args()

blob = torch.load(args.traj, map_location="cpu", weights_only=False)
tr = blob["trajs"][:args.max_traj]
n_min = min(x.shape[0] for x in tr)
T = torch.stack([t[:n_min] for t in tr]).float()
C = T.shape[-1]
print(f"[data] 궤적 {len(tr)}, 프레임 {n_min}, 앵커 {T.shape[2]}, 채널 {C}", flush=True)
BLK = [("위치 p", T[..., :3])]
if C >= 9:
    BLK += [("회전벡터 u", T[..., 3:6]), ("로그신축 s", T[..., 6:9])]

Q = [0.5, 0.9, 0.99, 0.999, 1.0]
print(f"\n== 상태 크기 |x| ==")
print(f"{'항':<12}{'평균':>10}" + "".join(f"{'p'+str(int(q*1000)/10):>10}" for q in Q))
for name, X in BLK:
    v = X.reshape(-1, 3).norm(dim=-1)
    qs = torch.quantile(v[torch.randperm(v.numel())[:200000]], torch.tensor(Q))
    print(f"{name:<12}{float(v.mean()):>10.4f}" + "".join(f"{float(a):>10.4f}" for a in qs))
if C >= 9:
    u = T[..., 3:6].reshape(-1, 3).norm(dim=-1)
    print(f"  |u| > pi 비율 {100*float((u > 3.14159).float().mean()):.2f}%,"
          f"  |u| > 2pi 비율 {100*float((u > 6.2832).float().mean()):.2f}%")

print(f"\n== 스텝 변화량 |dx| ==")
print(f"{'항':<12}{'전체평균':>11}{'첫스텝제외':>11}{'부풀림':>8}"
      + "".join(f"{'p'+str(int(q*1000)/10):>10}" for q in Q))
for name, X in BLK:
    d = (X[:, 1:] - X[:, :-1]).norm(dim=-1)
    allm = float(d.mean()); tail = float(d[:, 1:].mean())
    f = d.reshape(-1)
    qs = torch.quantile(f[torch.randperm(f.numel())[:200000]], torch.tensor(Q))
    print(f"{name:<12}{allm:>11.5f}{tail:>11.5f}{allm/max(tail,1e-12):>7.2f}x"
          + "".join(f"{float(a):>10.5f}" for a in qs))

print(f"\n== 프레임별 평균 변화량 ==")
hdr = [0, 1, 2, 3, 5, 10, 20, n_min - 2]
print(f"{'항':<12}" + "".join(f"{str(k)+'->'+str(k+1):>11}" for k in hdr))
for name, X in BLK:
    d = (X[:, 1:] - X[:, :-1]).norm(dim=-1)
    print(f"{name:<12}" + "".join(f"{float(d[:, k].mean()):>11.5f}"
                                   for k in hdr if k < d.shape[1]))
print("\nSCALE_DONE")
