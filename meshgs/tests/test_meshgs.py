"""사면체 케이지·결속·FEM 단위 테스트."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch

from meshgs.tetcage import (bind_gaussians, build_tet_mesh, dilate_fill,
                            occupancy, skin, tet_F)
from meshgs.fem import TetFEM, return_map, pk1_neohookean

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

print("[1] 점유 격자 -> 사면체 메시")
g = torch.stack(torch.meshgrid(*[torch.linspace(-0.4, 0.4, 14, device=dev)] * 3,
                               indexing="ij"), -1).reshape(-1, 3)
xyz = g + 0.005 * torch.randn_like(g)
occ, org, h = occupancy(xyz, res=32)
occ = dilate_fill(occ, 1)
V, T = build_tet_mesh(occ, org, h)
print(f"    정점 {V.shape[0]}, 사면체 {T.shape[0]}")
assert V.shape[0] > 100 and T.shape[0] > 100

print("[2] 결속 후 스키닝이 원래 위치를 복원하는가")
tid, bw = bind_gaussians(xyz, V, T)
back = skin(V, T, tid, bw)
err = float((back - xyz).norm(dim=-1).max())
print(f"    최대 복원 오차 {err:.3e}  (결속 실패 {int((tid < 0).sum())} 개)")
assert err < 1e-4

print("[3] 강체 변환에서 F = R 인가")
ang = torch.tensor(0.7, device=dev, dtype=torch.float64)
R = torch.tensor([[torch.cos(ang), -torch.sin(ang), 0],
                  [torch.sin(ang), torch.cos(ang), 0], [0, 0, 1]],
                 device=dev, dtype=torch.float64).float()
V2 = V @ R.t() + torch.tensor([0.3, -0.2, 0.1], device=dev)
F, _ = tet_F(V, V2, T)
print(f"    |F - R| 최대 {float((F - R).abs().max()):.3e}")
assert float((F - R).abs().max()) < 1e-4

print("[4] 강체에서 탄성력이 0 인가")
fem = TetFEM(V, T, E=1e5, nu=0.3, damping=0.0)
vel = torch.zeros_like(V2)
V3, _ = fem.step(V2.clone(), vel, 1e-4)
print(f"    강체 변환 후 이동 최대 {float((V3 - V2).norm(dim=-1).max()):.3e}")
assert float((V3 - V2).norm(dim=-1).max()) < 1e-6

print("[5] 리턴 매핑: 항복 이하는 그대로, 이상은 투영")
Fs = torch.eye(3, device=dev).expand(6, 3, 3).clone()
Fs[3:, 0, 0] = 2.0                       # 큰 신장
Fe, over, C = return_map(Fs, "von_mises", 3.8e4, yield_stress=1e2)
print(f"    소성 발생 {over.tolist()}")
assert (~over[:3]).all() and over[3:].all()
assert float((Fe[:3] - Fs[:3]).abs().max()) < 1e-6
# 보정 행렬이 정의를 만족하는가: F_trial C = F_e
print(f"    |F C - Fe| 최대 {float((Fs @ C - Fe).abs().max()):.3e}")
assert float((Fs @ C - Fe).abs().max()) < 1e-5

print("[6] 제하 후 잔류 변형: 탄성은 0 으로 돌아가고 소성은 남는다")
# **변위 제어**로 누른다. 힘으로 주면 강성에 따라 변형량이 달라져 항복 여부가
# 재질마다 달라지고, 크게 주면 한 스텝에 요소가 뒤집혀 발산한다. 위를 정해진
# 만큼 내렸다가 놓는 것이 항복 시험의 표준 절차다.
top = V[:, 2] > V[:, 2].max() - 1.5 * h
bot = V[:, 2] < V[:, 2].min() + 1.5 * h
H = float(V[:, 2].max() - V[:, 2].min())
PRESS, HOLD, REL, DT = 600, 200, 2000, 2e-4
v_press = 0.20 * H / (PRESS * DT)            # 높이의 20% 압축
res = {}
for kind, ys in (("none", 1e9), ("von_mises", 5e3)):
    f2 = TetFEM(V, T, density=200.0, E=1e5, nu=0.3, plastic=kind,
                yield_stress=ys, damping=8.0)
    Vc, vc = V.clone(), torch.zeros_like(V)
    for s in range(PRESS + HOLD + REL):
        hold = (s < PRESS + HOLD)
        Vc, vc = f2.step(Vc, vc, DT, fixed=(bot | top) if hold else bot)
        if s < PRESS:                         # 위를 일정 속도로 내린다
            Vc[top, 2] -= v_press * DT
        if not torch.isfinite(Vc).all():
            break
    ok = bool(torch.isfinite(Vc).all())
    r = float((Vc - V).norm(dim=-1).mean()) / H if ok else float("nan")
    inv, dmin, ar = f2.quality(Vc) if ok else (float("nan"),) * 3
    res[kind] = r
    print(f"    {kind:<10} 유한 {ok}  잔류 {100*r:6.2f}% of 높이  "
          f"뒤집힘 {inv:.3f}  detF 최소 {dmin:.3f}")
assert res["none"] == res["none"] and res["von_mises"] == res["von_mises"], "발산"
assert res["von_mises"] > 3 * res["none"], \
    f"소성이 탄성보다 잔류가 크지 않다 ({res})"

print("\nTEST_OK")
