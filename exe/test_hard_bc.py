"""하드 구속(바닥 사영)과 소성·에너지의 수학을 검사한다.

지금 test_simplex 는 격자 기하와 C0 까지만 본다. 여기서는 바꾼 물리 쪽을
직접 검사한다:

  bc_project  관통이 정확히 0 이 되는가, 자유 입자는 건드리지 않는가,
              멱등인가(두 번 걸어도 같은가), sticky 접선이 0 인가
  psi_of      항등 변형에서 Psi=0, dlog=0
  회전 불변   Psi(RF) == Psi(F)
  체적 반응   압축/팽창에서 Psi 가 증가하고 최소가 F=I 인가
"""
from __future__ import annotations
import argparse, os, sys

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import torch
from anchorflow import phys_resid as PR

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50000)
ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = a.dev
torch.manual_seed(0)
OK = [0, 0]


def chk(name, cond, info=""):
    OK[1] += 1
    OK[0] += bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  {info}" if info else ""))


# 바닥 z=0.4, 법선 +z, sticky. PG 설정과 같은 꼴.
cfg = {"E": 2e6, "nu": 0.3, "material": "metal", "yield_stress": 4e4,
       "density": 1000, "n_grid": 100, "grid_lim": 2.0, "bound": 3,
       "boundary_conditions": [
           {"type": "surface_collider", "point": [0.0, 0.0, 0.4],
            "normal": [0.0, 0.0, 1.0], "surface": "sticky"}]}
GL, NG = 2.0, 100
dx = GL / NG

# 바닥 위·바닥 근처·바닥 아래로 갈 점을 섞는다
x = torch.rand(a.n, 3, device=dev) * 0.8 + 0.6
x[:, 2] = torch.rand(a.n, device=dev) * 0.5 + 0.40      # z in [0.40, 0.90]
du = (torch.rand(a.n, 3, device=dev) - 0.5) * 0.1
duP, act = PR.bc_project(x, du, cfg, 1 / 60, GL, NG)

sd2 = (x + duP)[:, 2] - 0.4
chk("사영 후 바닥 아래가 없다", float((-sd2).clamp_min(0).max()) < 1e-6,
    f"최대 침투 {float((-sd2).clamp_min(0).max()):.2e}")
# 활성 집합 밖은 건드리지 않는다
chk("비활성 입자는 du 가 비트 단위로 그대로",
    bool((duP[~act] == du[~act]).all()),
    f"활성 {int(act.sum())}/{a.n} ({100*float(act.float().mean()):.1f}%)")
# 멱등: 한 번 사영한 것을 다시 걸어도 같아야 한다
duP2, act2 = PR.bc_project(x, duP, cfg, 1 / 60, GL, NG)
chk("멱등 (두 번 걸어도 같다)",
    float((duP2 - duP).abs().max()) < 1e-6,
    f"차 {float((duP2 - duP).abs().max()):.2e}")
# sticky: 면에 닿은 입자는 접선 변위가 0
touch = (x[:, 2] - 0.4) < dx
if int(touch.sum()):
    tang = duP[touch][:, :2].abs().max()
    chk("sticky 접선 변위가 0", float(tang) < 1e-6,
        f"닿은 입자 {int(touch.sum())}, 접선 최대 {float(tang):.2e}")
else:
    chk("sticky 접선 변위가 0", False, "닿은 입자가 없다 -- 시험 설계 오류")

# ---------------------------------------------------------------- 물성
I3 = torch.eye(3, device=dev).expand(a.n, 3, 3).contiguous()
psi0, pl0 = PR.psi_of(I3.clone(), cfg, 1 / 60)
chk("항등 변형에서 Psi = 0", float(psi0.abs().max()) < 1e-6,
    f"{float(psi0.abs().max()):.2e}")
chk("항등 변형에서 dlog = 0", float(pl0.dlog.abs().max()) == 0.0,
    f"{float(pl0.dlog.abs().max()):.1e}")

# 회전 불변: Psi(R F) == Psi(F)
Fr = I3 + 0.2 * torch.randn(a.n, 3, 3, device=dev)
q = torch.randn(a.n, 4, device=dev)
q = q / q.norm(dim=-1, keepdim=True)
w, xq, yq, zq = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
R = torch.stack([
    torch.stack([1 - 2*(yq*yq+zq*zq), 2*(xq*yq-zq*w), 2*(xq*zq+yq*w)], -1),
    torch.stack([2*(xq*yq+zq*w), 1 - 2*(xq*xq+zq*zq), 2*(yq*zq-xq*w)], -1),
    torch.stack([2*(xq*zq-yq*w), 2*(yq*zq+xq*w), 1 - 2*(xq*xq+yq*yq)], -1)], -2)
p1, _ = PR.psi_of(Fr, cfg, 1 / 60)
p2, _ = PR.psi_of(R @ Fr, cfg, 1 / 60)
rel = ((p2 - p1).abs() / p1.abs().clamp_min(1e-6)).median()
chk("회전 불변 Psi(RF) == Psi(F)", float(rel) < 1e-4,
    f"상대오차 중앙 {float(rel):.2e}")

# 체적: F = s I 에서 s=1 이 최소
ss = torch.tensor([0.85, 0.95, 1.0, 1.05, 1.15], device=dev)
ps = []
for s_ in ss:
    Fs = (s_ * torch.eye(3, device=dev)).expand(64, 3, 3).contiguous()
    ps.append(float(PR.psi_of(Fs, cfg, 1 / 60)[0].mean()))
chk("Psi 최소가 F = I", min(range(5), key=lambda i: ps[i]) == 2,
    " ".join(f"s={float(ss[i]):.2f}:{ps[i]:.3e}" for i in range(5)))
print(f"\n{OK[0]}/{OK[1]}  " + ("ALL-OK" if OK[0] == OK[1] else "SOME-FAIL"))
