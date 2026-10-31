"""절 AK.24 단위 시험 자료: 공중의 균일 블록 (lego 재질 E 1e6, ν 0.3, ρ 1000, 입자 간격 0.01 m = 정답 칸당 8 입자, 한 변 L) 에 균일 아핀 변형 A 를 준 상태 (x = c + A (X − c), F = A, v = 0, 중력 0, 바닥 없음).
fem_ex_sim 이 읽는 .pt 형식 (정답 x 는 쓰지 않으므로 초기 상태 복사). 사용: python exe/fem_ak_block.py <출력 폴더> [L 1.0] [변형 0.03] [입자 간격 0.0125]"""
import json
import os
import sys

import torch

out = sys.argv[1]
Lb = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
eps = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
os.makedirs(out, exist_ok=True)
cfg = json.load(open("/home/dkta/work/lbdata/config.json"))
cfg["g"] = [0.0, 0.0, 0.0]
cfg["boundary_conditions"] = []
sp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0125                            # 입자 간격 (정답 칸 0.02 에 약 4 입자)
n = int(round(Lb / sp))
g1 = (torch.arange(n, dtype=torch.float64) + 0.5) * sp - Lb / 2
X = torch.stack(torch.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
c = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
T = 81
for nm, A in (("stretch", torch.diag(torch.tensor([1 + eps, 1.0, 1.0], dtype=torch.float64))),
              ("shear", torch.tensor([[1.0, eps, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64))):
    x0 = (c + X @ A.T).float()
    N = x0.shape[0]
    d = dict(cfg=cfg, x=x0[None].expand(T, N, 3), v=torch.zeros(1, N, 3).expand(T, N, 3), F=A.float()[None, None].expand(1, N, 3, 3).contiguous(),
             F_is_true=True, A=A, L=Lb, eps=eps)                                     # 정답 x 는 초기 상태 복사 (보기만 -- 저장은 한 벌)
    torch.save(d, os.path.join(out, f"{nm}.pt"))
    print(nm, "입자", N, "A", A.tolist(), flush=True)
