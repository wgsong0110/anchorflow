"""학습 경로의 Gregory·방사형 스키닝(lib/anchorflow/skin.py)이 출력만 최적화의 repmaps.Lattice 와 같은 값을 내는지.

같은 입자·같은 노드 변위 u·반경 ρ_raw 를 넣고 위치 변위와 야코비안을 비교한다.
  python exe/test_skin.py --pt /home/dkta/work/lbpt/test/0000.pt
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                     # noqa: E402
from anchorflow import repmaps as rm                             # noqa: E402
from anchorflow.skin import gregory_skin                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pt", required=True)
ap.add_argument("--n", type=int, default=50000)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
d = torch.load(a.pt, map_location="cpu", weights_only=False, mmap=True)
P = d["x"][0][:a.n].float().to(dev)
h = (2.0 / 100.0) * (2.0 * math.sqrt(2.0)) ** (1.0 / 3.0)             # rep_ip 의 Lattice h 와 같다 (= node_h 0.028284)
L = rm.Lattice(P, h=h)
L.rebind(P)
with torch.no_grad():
    L.u.copy_(0.01 * torch.randn_like(L.u))
    L.rho_raw.copy_(torch.randn_like(L.rho_raw))
    y0, J0 = L.yJ(P, torch.arange(P.shape[0], device=dev))
    y1, D1 = gregory_skin(P, L.Xn, L.rows, L.u, L.rho_raw, h)
    J1 = D1 + torch.eye(3, device=dev)                                 # 스키닝은 ∇u 만 낸다
print(f"[스키닝 일치] 노드 {L.Xn.shape[0]}, 입자 {P.shape[0]}, 변위 최대차 {float((y0 - y1).abs().max()):.3e} "
      f"(변위 크기 {float(y0.abs().max()):.3e}), 야코비안 최대차 {float((J0 - J1).abs().max()):.3e}", flush=True)
