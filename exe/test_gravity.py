"""우리 목적함수에서 **자유 입자가 자유낙하하는지** 검사한다.

i-PG implicit 솔버는 격자점 질량이 최대의 1% 미만이면 그 격자점을 잔차에서
빼버려(`R=0`) 이탈 입자가 중력을 전혀 못 받았다. 우리 쪽은 중력이 격자를 거치지
않고 **입자별 관성항**에 들어가므로(`d = du - h*v - h^2*g`) 같은 함정이 없어야
한다 -- 단정하지 않고 확인한다.

  자유 입자(구속 아님, 탄성 결합 없음)의 증분 포텐셜 최소점은
      du* = h*v + h^2*g          (자유낙하)
  이고, 이탈 입자(혼자 떨어진 입자)도 같아야 한다.
"""
from __future__ import annotations
import argparse, os, sys

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import torch
from anchorflow import phys_resid as PR

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = a.dev
torch.manual_seed(0)
OK = [0, 0]


def chk(name, cond, info=""):
    OK[1] += 1
    OK[0] += bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  {info}" if info else ""))


cfg = {"E": 2e6, "nu": 0.3, "material": "metal", "yield_stress": 4e4,
       "density": 1000, "n_grid": 100, "grid_lim": 2.0, "bound": 3,
       "boundary_conditions": []}          # 바닥 없음 -- 순수 자유낙하
h = 1.0 / 60
g = torch.tensor([0.0, 0.0, -9.8], device=dev)

# 입자 셋: 뭉친 것 100 개 + **혼자 떨어진 것** 1 개 (질량 대비가 크다)
x = torch.cat([torch.rand(100, 3, device=dev) * 0.05 + 1.0,
               torch.tensor([[1.6, 1.6, 1.6]], device=dev)], 0)
v = torch.zeros_like(x)
F = torch.eye(3, device=dev).expand(x.shape[0], 3, 3).contiguous()
# 질량: 뭉친 입자는 서로 겹쳐 무겁게, 이탈 입자는 가볍게 (i-PG 가 자른 상황)
mass = torch.cat([torch.full((100,), 1.0, device=dev),
                  torch.full((1,), 1e-3, device=dev)])
vol = mass / float(cfg["density"])
chk("이탈 입자 질량이 최대의 1% 미만 (i-PG 가 잘라낸 조건)",
    float(mass[-1] / mass.max()) < 1e-2,
    f"비 {float(mass[-1]/mass.max()):.1e}")

# du 를 자유변수로 두고 증분 포텐셜을 최소화한다 (망 없이, 매개화 없이)
du = torch.zeros_like(x).requires_grad_(True)
opt = torch.optim.Adam([du], lr=3e-3)
for it in range(4000):
    E, _, _, _ = PR.pts_ip_energy(x, du, v, F, None, mass, vol, cfg, h,
                                  int(cfg["n_grid"]),
                                  float(cfg["grid_lim"]), g=g)
    opt.zero_grad(set_to_none=True)
    E.backward()
    opt.step()
want = (h * v + (h * h) * g)
err = (du.detach() - want).norm(dim=-1) / max(float(want.norm(dim=-1).max()), 1e-12)
chk("뭉친 입자의 최소점이 자유낙하", float(err[:100].max()) < 0.05,
    f"상대오차 최대 {float(err[:100].max()):.3f}")
chk("**이탈 입자**의 최소점도 자유낙하", float(err[-1]) < 0.05,
    f"상대오차 {float(err[-1]):.3f}  (du_z {float(du[-1,2]):.3e} vs "
    f"{float(want[-1,2]):.3e})")
# 중력을 끄면 변위가 0 이어야 한다 (중력이 실제로 그 항을 만든다는 확인)
du0 = torch.zeros_like(x).requires_grad_(True)
opt0 = torch.optim.Adam([du0], lr=3e-3)
for it in range(2000):
    E0, _, _, _ = PR.pts_ip_energy(x, du0, v, F, None, mass, vol, cfg, h,
                                   int(cfg["n_grid"]),
                                   float(cfg["grid_lim"]), g=None)
    opt0.zero_grad(set_to_none=True); E0.backward(); opt0.step()
chk("중력을 끄면 변위가 0", float(du0.detach().norm(dim=-1).max()) < 1e-4,
    f"최대 {float(du0.detach().norm(dim=-1).max()):.2e}")
print(f"\n{OK[0]}/{OK[1]}  " + ("ALL-OK" if OK[0] == OK[1] else "SOME-FAIL"))
