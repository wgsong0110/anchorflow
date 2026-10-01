"""공중에 뜬 공 하나로 **자유낙하 기준 궤적**을 만든다.

중력이 증분 포텐셜에 제대로 들어가는지 재려면, 정답이 닫힌 형식으로 알려진
장면이 필요하다. 손잡이도 바닥도 없고 F = I 인 공은 중력만 받으므로 정답이
    x(t) = x0 + ½ g t²
이고, 증분 포텐셜의 최소점도 정확히 그것이다 (관성항의 목표가 x + hv + h²g 이고
변형이 없으면 탄성항이 0 이다). 그래서 출력만 최적화가 이 궤적을 못 내면 중력이
안 걸린 것이다 -- 기존 찰흙 궤적은 fp16 로 저장돼 2 차 차분이 뭉개져 기준이 못
된다.
"""
from __future__ import annotations
import argparse
import math
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--ref", default="", help="cfg 를 베껴올 궤적 (물성·dt)")
ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--r", type=float, default=0.1)
ap.add_argument("--center", type=float, nargs=3, default=(1.0, 1.0, 1.4))
ap.add_argument("--frames", type=int, default=13)
ap.add_argument("--material", default="jelly")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cfg = dict(E=2e6, nu=0.3, g=[0.0, 0.0, -9.8], frame_dt=1.0 / 24.0,
           density=1000.0, n_grid=100, grid_lim=2.0, material=a.material,
           boundary_conditions=[])
if a.ref:
    try:
        rc = torch.load(a.ref, map_location="cpu", weights_only=False)["cfg"]
    except TypeError:
        rc = torch.load(a.ref, map_location="cpu")["cfg"]
    for k in ("E", "nu", "frame_dt", "density", "n_grid", "grid_lim"):
        if k in rc:
            cfg[k] = rc[k]
    print(f"[cfg] {a.ref} 에서 베낀 것: "
          + ", ".join(f"{k}={cfg[k]}" for k in
                      ("E", "nu", "frame_dt", "density", "n_grid")))
# 공 안에 고르게 (거절 표집이 아니라 반지름^(1/3) 로 -- 치우치지 않는다)
g_ = torch.Generator().manual_seed(a.seed)
u = torch.rand(a.n, generator=g_)
th = torch.rand(a.n, generator=g_) * 2 * math.pi
cz = torch.rand(a.n, generator=g_) * 2 - 1
rr = a.r * u.pow(1.0 / 3.0)
sz = (1 - cz * cz).clamp_min(0).sqrt()
x0 = torch.stack([rr * sz * th.cos(), rr * sz * th.sin(), rr * cz], -1)
x0 = x0 + torch.tensor(a.center, dtype=torch.float32)

dt = float(cfg["frame_dt"])
gv = torch.tensor(cfg["g"], dtype=torch.float32)
T = int(a.frames)
# 정답: 정지에서 출발한 자유낙하. t0=0 에서 v=0 이 되도록 프레임 0 을 정지점으로.
xs = torch.stack([x0 + 0.5 * gv * (i * dt) ** 2 for i in range(T)], 0)
d = dict(x=xs, F=torch.eye(3).expand(T, a.n, 3, 3).clone(), cfg=cfg,
         sel=torch.arange(a.n), n_full=a.n)
torch.save(d, a.out)
drop = float(0.5 * 9.8 * ((T - 1) * dt) ** 2)
print(f"[저장] {a.out}  {T} 프레임 x {a.n} 입자, 반지름 {a.r}, "
      f"중심 {tuple(a.center)}, {T-1} 프레임 낙하량 {drop:.3f} "
      f"(공 지름의 {drop/(2*a.r):.1f} 배)")
