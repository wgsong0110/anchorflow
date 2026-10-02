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
ap.add_argument("--shape", choices=["ball", "cyl"], default="ball",
                help="cyl = **원기둥** (축은 --handle_axis 와 같은 축)")
ap.add_argument("--cyl_len", type=float, default=0.40, help="원기둥 길이")
ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--n_add", type=int, default=0,
                help="**기존 입자는 그대로 두고** 그 위에 더 뽑는다 (시드를 "
                     "따로 쓴다). 앞 --n 개는 --n_add 0 일 때와 비트까지 같다")
ap.add_argument("--r", type=float, default=0.1)
ap.add_argument("--center", type=float, nargs=3, default=(1.0, 1.0, 1.4))
ap.add_argument("--frames", type=int, default=13)
ap.add_argument("--material", default="jelly")
ap.add_argument("--E", type=float, default=0.0,
                help="영률. 0 이면 --ref 의 값을 쓴다. 키우면 응력 전달이 "
                     "세진다 (정적 처짐이 1/E 로 준다)")
ap.add_argument("--nu", type=float, default=-1.0)
ap.add_argument("--scheme", choices=["implicit", "analytic", "hold"],
                default="implicit",
                help="기준 궤적의 적분. implicit=증분 포텐셜과 **같은 이산해** "
                     "x_n = x0 + h²g n(n+1)/2 (기본), analytic=연속해 ½gt². "
                     "연속해를 쓰면 적분 차이 h²gN/2 가 그대로 오차로 잡힌다 "
                     "(실측 12 프레임에서 4.77% -- 출력만 최적화 탓이 "
                     "아니다), hold=정지 자세 그대로 (손잡이 씬처럼 닫힌 해가 "
                     "없을 때 쓴다 -- **해가 아니라 기준점일 뿐이다**)")
ap.add_argument("--handle", choices=["none", "top", "topbot"], default="none",
                help="top=최상단 입자 하나, topbot=**최상단·최하단 둘**을 잡고 "
                     "서로 반대 방향으로 끈다 (양쪽으로 늘리기)")
ap.add_argument("--handle_r", type=float, default=0.04)
ap.add_argument("--n_grid", type=int, default=0,
                help="MPM 격자. 0 이면 --ref 값(보통 100). 손잡이 반경보다 "
                     "dx 가 크면 구 안에 격자점이 안 들어가 손잡이가 먹지 않는다")
ap.add_argument("--handle_axis", type=int, default=2, choices=[0, 1, 2],
                help="topbot 에서 두 손잡이를 고를 축 (0=x, 1=y, 2=z)")
ap.add_argument("--handle_vel", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                help="손잡이 명령 속도. 0 이면 **붙잡고 있는다**")
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
    for k in ("E", "nu", "frame_dt", "density", "n_grid", "grid_lim"):  # ref
        if k in rc:
            cfg[k] = rc[k]
if a.n_grid > 0:
    cfg["n_grid"] = int(a.n_grid)
if a.E > 0:
    cfg["E"] = float(a.E)
if a.nu >= 0:
    cfg["nu"] = float(a.nu)
print(f"[cfg] 최종: "
      + ", ".join(f"{k}={cfg[k]}" for k in
                  ("E", "nu", "frame_dt", "density", "n_grid")))
# 공 안에 고르게 (거절 표집이 아니라 반지름^(1/3) 로 -- 치우치지 않는다)
def _ball(n, seed):
    g_ = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=g_)
    th = torch.rand(n, generator=g_) * 2 * math.pi
    cz = torch.rand(n, generator=g_) * 2 - 1
    rr = a.r * u.pow(1.0 / 3.0)
    sz = (1 - cz * cz).clamp_min(0).sqrt()
    return torch.stack([rr * sz * th.cos(), rr * sz * th.sin(), rr * cz], -1)

def _cyl(n, seed):
    """축(--handle_axis)을 따라 길이 cyl_len, 반지름 r 인 원기둥."""
    g_ = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=g_).sqrt() * a.r
    th = torch.rand(n, generator=g_) * 2 * math.pi
    zz = (torch.rand(n, generator=g_) - 0.5) * a.cyl_len
    _ax = int(a.handle_axis)
    _o1, _o2 = [k for k in range(3) if k != _ax]
    p = torch.zeros(n, 3)
    p[:, _ax] = zz
    p[:, _o1] = u * th.cos()
    p[:, _o2] = u * th.sin()
    return p


x0 = _cyl(a.n, a.seed) if a.shape == "cyl" else _ball(a.n, a.seed)
n_base = a.n
if a.n_add > 0:
    # 기존 입자를 건드리지 않고 **덧붙인다** -- 같은 시드로 개수만 늘리면
    # 호출마다 난수 흐름이 밀려 앞쪽 입자까지 전부 달라진다.
    x0 = torch.cat([x0, (_cyl if a.shape == "cyl" else _ball)(
        a.n_add, a.seed + 1000003)], 0)
    print(f"[입자] 기존 {n_base} + 추가 {a.n_add} = {x0.shape[0]} 개 "
          f"(앞 {n_base} 개는 그대로다)")
a.n = x0.shape[0]
x0 = x0 + torch.tensor(a.center, dtype=torch.float32)

dt = float(cfg["frame_dt"])
gv = torch.tensor(cfg["g"], dtype=torch.float32)
T = int(a.frames)
# 정답: 정지에서 출발한 자유낙하. t0=0 에서 v=0 이 되도록 프레임 0 을 정지점으로.
# 증분 포텐셜의 한 스텝은 Delta u = h v + h²g, 즉 v_{n+1} = v_n + h g 이고
# x_{n+1} = x_n + h v_{n+1} 이다 -> x_n = x0 + h²g n(n+1)/2.
if a.scheme == "hold":
    xs = x0.unsqueeze(0).expand(T, a.n, 3).clone()
elif a.scheme == "implicit":
    xs = torch.stack([x0 + (dt * dt) * gv * (i * (i + 1) / 2)
                      for i in range(T)], 0)
else:
    xs = torch.stack([x0 + 0.5 * gv * (i * dt) ** 2 for i in range(T)], 0)
d = dict(x=xs, F=torch.eye(3).expand(T, a.n, 3, 3).clone(), cfg=cfg,
         sel=torch.arange(a.n), n_full=a.n)
if a.handle == "topbot":
    # 양쪽으로 늘린다: 위는 +v, 아래는 -v. 중심은 각 입자의 현재 위치를 따른다.
    _ax = int(a.handle_axis)
    cid_t = int(x0[:n_base, _ax].argmax())
    cid_b = int(x0[:n_base, _ax].argmin())
    hv = torch.tensor(a.handle_vel, dtype=torch.float32)
    d["ctrl_id"] = torch.tensor([cid_t, cid_b], dtype=torch.long)
    d["ctrl_vel"] = torch.stack([hv, -hv], 0).reshape(1, 2, 3).expand(
        T, 2, 3).clone()
    d["ctrl_R"] = torch.tensor([a.handle_r], dtype=torch.float32)
    d["ctrl_pos"] = torch.stack([xs[:, cid_t], xs[:, cid_b]], 1).clone()
    _nt = int(((x0 - x0[cid_t]).norm(dim=-1) < a.handle_r).sum())
    _nb = int(((x0 - x0[cid_b]).norm(dim=-1) < a.handle_r).sum())
    print(f"[손잡이] 축 {'xyz'[_ax]}: 한쪽 {cid_t} "
          f"({'xyz'[_ax]}={float(x0[cid_t,_ax]):.4f}) +v, 반대쪽 {cid_b} "
          f"({float(x0[cid_b,_ax]):.4f}) -v, 반경 {a.handle_r}, "
          f"명령 {tuple(a.handle_vel)}, 반경 안 입자 {_nt} / {_nb}")
elif a.handle == "top":
    # 최상단 입자를 손잡이로. 중심은 **그 입자의 현재 위치**를 따른다 (i-PG 가
    # collider 의 point 를 매 프레임 그렇게 갱신한다). 기준 궤적에서는 그 입자가
    # 기준대로 움직이므로 ctrl_pos 도 그 궤적을 그대로 쓴다.
    # 손잡이 입자는 **기존 집합 안에서** 고른다 -- 추가 입자 때문에 손잡이가
    # 다른 입자로 바뀌면 같은 장면이 아니게 된다.
    cid = int(x0[:n_base, 2].argmax())
    hv = torch.tensor(a.handle_vel, dtype=torch.float32)
    d["ctrl_id"] = torch.tensor([cid], dtype=torch.long)
    d["ctrl_vel"] = hv.reshape(1, 1, 3).expand(T, 1, 3).clone()
    d["ctrl_R"] = torch.tensor([a.handle_r], dtype=torch.float32)
    d["ctrl_pos"] = xs[:, cid].reshape(T, 1, 3).clone()
    _in = int(((x0 - x0[cid]).norm(dim=-1) < a.handle_r).sum())
    print(f"[손잡이] 입자 {cid} (z={float(x0[cid,2]):.4f}), 반경 "
          f"{a.handle_r}, 명령 속도 {tuple(a.handle_vel)}, "
          f"초기 반경 안 입자 {_in} 개")
torch.save(d, a.out)
drop = float(abs(float(xs[-1, 0, 2] - xs[0, 0, 2])))
print(f"[저장] {a.out}  {T} 프레임 x {a.n} 입자, 반지름 {a.r}, "
      f"중심 {tuple(a.center)}, 적분 {a.scheme}, {T-1} 프레임 낙하량 {drop:.3f} "
      f"(공 지름의 {drop/(2*a.r):.1f} 배)")
