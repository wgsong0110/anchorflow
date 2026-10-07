"""단일 가우시안 속도장으로 목표 변형을 만든다 (표현력 비교 실험의 정답 궤적).

속도장은 **한 번에 하나**다. 기본은 가우시안 v(x) = A·d·exp(-|x-c|²/σ²) 이고,
위상 변화를 내려면 끊기는 장(가르기·근원·비틀어 찢기·흡입, KINDS 참고)을 창마다
돌아가며 섞는다.

일정 주기(--period 프레임)마다 (c, d, σ, A) 를 새로 뽑는다. 적분 스텝은 그 창의
값으로 **Δt < σ²/A** 를 지킨다 (Δt = safety·σ²/A). 이 조건 아래에서는 한 스텝의
변위 기울기 |∇v|·Δt 가 작아 흐름 사상이 접히지 않는다 -- 생성 중 det F 를 직접
재서 확인한다.

모든 형상은 bbox 를 **[0,1]³ 정규화한 좌표**에서 같은 시드의 같은 속도장 열을
받는다. 그래서 형상만 다르고 걸리는 변형은 같다.

출력 (npz):
  X0      [N,3]  정규화된 정지 위치 (부분표본)
  traj    [T+1,N,3] 프레임별 목표 위치
  F       [T+1,N,3,3] 목표 변형 기울기 (det F 확인용, 저장은 --save_F 일 때만)
  idx     [N]    채우기 입자 집합에서 뽑은 인덱스 (모든 방법이 같은 인덱스를 쓴다)
  lo, s   정규화 상수 (x_sim = lo + s * x_norm)
  field   [W,9]  창별 (c3, d3, σ, A, 종류)

  python exe/gauss_flow.py --fill pgfill_wolf.npy --out flow_wolf.npz
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch


KINDS = ["smooth", "split", "point_src", "line_src", "twist_tear", "sink"]
#   smooth      v = A d g                         부드러운 밀기 (위상 보존)
#   split       v = A d sign((x-c)·d) g           평면 양쪽이 벌어진다 (갈라짐)
#   point_src   v = A n̂ g,  n̂=(x-c)/|x-c|         중심에서 찢어져 빈 공간이 열린다
#   line_src    v = A p̂ g⊥, p̂=x⊥/|x⊥|             축 d 를 따라 관통 구멍이 뚫린다
#   twist_tear  v = A sign((x-c)·d) (d×(x-c))/σ g  평면 양쪽이 반대로 비틀려 찢어진다
#   sink        v = -A n̂ g · cut(r/ε)              한 점으로 빨려 들어가 합쳐진다
# g = exp(-|x-c|²/σ²) (line_src 는 축까지 거리 g⊥). 끊기는 곳은 측도 0 인 집합이다.
SINK_EPS = 0.02
STEP_DET_MIN = float("inf")      # 지금까지 모든 스텝 사상의 det 최솟값


def field_seq(seed, n_win, c_lo=0.2, c_hi=0.8, sig=(0.15, 0.35),
              amp=(0.3, 0.8), kinds=None):
    """창별 (c3, d3, σ, A, 종류). 시드가 같으면 형상과 무관하게 같다.

    kinds 를 주면 창마다 그 종류를 차례로 돈다 (기본: 부드러운 가우시안만).
    """
    r = np.random.default_rng(seed)
    out = []
    for w in range(n_win):
        c = r.uniform(c_lo, c_hi, 3)
        d = r.normal(size=3)
        d /= np.linalg.norm(d)
        k = KINDS.index(kinds[w % len(kinds)]) if kinds else 0
        out.append(np.concatenate([c, d, [r.uniform(*sig), r.uniform(*amp), k]]))
    return np.array(out, dtype=np.float64)


def velocity(x, f):
    """창 f 의 속도 v(x) (x: [N,3])."""
    c, d, sg, A = f[:3], f[3:6], f[6], f[7]
    kind = KINDS[int(round(float(f[8])))] if f.shape[0] > 8 else "smooth"
    dx = x - c
    r2 = (dx * dx).sum(1)
    g = torch.exp(-r2 / (sg * sg))
    if kind == "smooth":
        return A * g[:, None] * d[None]
    if kind == "split":
        return A * (g * torch.sign(dx @ d))[:, None] * d[None]
    if kind == "point_src":
        n = dx / r2.sqrt().clamp_min(1e-9)[:, None]
        return A * g[:, None] * n
    if kind == "line_src":
        xp = dx - (dx @ d)[:, None] * d[None]
        rp2 = (xp * xp).sum(1)
        n = xp / rp2.sqrt().clamp_min(1e-9)[:, None]
        return A * torch.exp(-rp2 / (sg * sg))[:, None] * n
    if kind == "twist_tear":
        rot = torch.cross(d.expand_as(dx), dx, dim=1) / sg
        return A * (g * torch.sign(dx @ d))[:, None] * rot
    if kind == "sink":
        r = r2.sqrt()
        n = dx / r.clamp_min(1e-9)[:, None]
        cut = (r / SINK_EPS).clamp(0.0, 1.0)              # 중심에 닿으면 멈춘다
        return -A * (g * cut)[:, None] * n
    raise ValueError(kind)


def vel_and_grad(x, f):
    """v(x) 와 ∇v(x) -- 자동미분 (끊기는 곳 밖에서 정확)."""
    with torch.enable_grad():
        xx = x.detach().requires_grad_(True)
        v = velocity(xx, f)
        J = torch.stack([torch.autograd.grad(v[:, i].sum(), xx,
                                             retain_graph=(i < 2))[0]
                         for i in range(3)], 1)          # [N,3,3] J[i,j]=∂v_i/∂x_j
    return v.detach(), J.detach()


def dt_limit(f, safety):
    """Δt < σ²/A (흡입은 중심을 넘어가지 않게 Δt < ε/A 도)."""
    sg, A = float(f[6]), float(f[7])
    dt = safety * sg * sg / A
    if f.shape[0] > 8 and KINDS[int(round(float(f[8])))] == "sink":
        dt = min(dt, safety * SINK_EPS / A)
    return dt


def advance(x, F, f, T, safety=0.5):
    """창 f 로 시간 T 만큼 RK4 적분 (Δt = safety·σ²/A 이하). F 도 함께."""
    dt_max = dt_limit(f, safety)                          # Δt < σ²/A
    n = max(1, math.ceil(T / dt_max))
    dt = T / n
    global STEP_DET_MIN
    for _ in range(n):
        # 한 스텝 사상의 야코비안 (RK4 를 F=I 로 한 번 돌린 것) 의 det 을 잰다
        Fi = torch.eye(3, dtype=x.dtype, device=x.device).expand(x.shape[0], 3, 3)
        def rhs(xx, FF):
            v, J = vel_and_grad(xx, f)
            return v, J @ FF
        k1x, k1F = rhs(x, F)
        k2x, k2F = rhs(x + 0.5 * dt * k1x, F + 0.5 * dt * k1F)
        k3x, k3F = rhs(x + 0.5 * dt * k2x, F + 0.5 * dt * k2F)
        k4x, k4F = rhs(x + dt * k3x, F + dt * k3F)
        _, s1 = rhs(x, Fi)
        _, s2 = rhs(x + 0.5 * dt * k1x, Fi + 0.5 * dt * s1)
        _, s3 = rhs(x + 0.5 * dt * k2x, Fi + 0.5 * dt * s2)
        _, s4 = rhs(x + dt * k3x, Fi + dt * s3)
        Fstep = Fi + dt / 6.0 * (s1 + 2 * s2 + 2 * s3 + s4)
        STEP_DET_MIN = min(STEP_DET_MIN, float(torch.linalg.det(Fstep).min()))
        x = x + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        F = F + dt / 6.0 * (k1F + 2 * k2F + 2 * k3F + k4F)
    return x, F, n, dt


def run_flow(x0, field, frames, period, frame_dt, safety=0.5, keep_F=False):
    """x0 [N,3] 텐서를 frames 프레임 흘린다 -> traj [T+1,N,3], F, 기록."""
    x = x0.clone()
    F = torch.eye(3, dtype=x.dtype, device=x.device).expand(x.shape[0], 3, 3) \
        .clone()
    traj, Fs, log = [x.clone()], ([F.clone()] if keep_F else None), []
    for t in range(frames):
        f = torch.as_tensor(field[t // period], dtype=x.dtype, device=x.device)
        x, F, n, dt = advance(x, F, f, frame_dt, safety)
        traj.append(x.clone())
        if keep_F:
            Fs.append(F.clone())
        det = torch.linalg.det(F)
        log.append((t, n, dt, float(det.min()), float(det.max())))
    return torch.stack(traj), (torch.stack(Fs) if keep_F else None), log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", required=True, help="PG 공식 채우기 입자 (npy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30000, help="부분표본 입자 수")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--period", type=int, default=10, help="가우시안 교체 주기(프레임)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0, help="속도장 시드 (형상 공통)")
    ap.add_argument("--sub_seed", type=int, default=0, help="부분표본 시드")
    ap.add_argument("--safety", type=float, default=0.5, help="Δt = safety·σ²/A")
    ap.add_argument("--save_F", action="store_true")
    ap.add_argument("--amp", type=float, nargs=2, default=[2.0, 5.0],
                    help="세기 A 범위 (정규화 단위/초)")
    ap.add_argument("--sig", type=float, nargs=2, default=[0.1, 0.3],
                    help="폭 σ 범위")
    ap.add_argument("--kinds", default=",".join(KINDS),
                    help="창마다 돌아가며 쓸 장의 종류 (쉼표). 'smooth' 만이면 위상 보존")
    a = ap.parse_args()

    X = np.load(a.fill).astype(np.float64)
    lo = X.min(0)
    s = float((X.max(0) - lo).max())
    Xn = (X - lo) / s                                   # [0,1]³ 안 (긴 축이 1)
    # 짧은 축은 가운데로 (형상마다 같은 위치에 오도록)
    Xn = Xn + (1.0 - Xn.max(0)) / 2.0
    r = np.random.default_rng(a.sub_seed)
    idx = np.sort(r.choice(len(Xn), min(a.n, len(Xn)), replace=False))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x0 = torch.as_tensor(Xn[idx], dtype=torch.float64, device=dev)
    field = field_seq(a.seed, (a.frames + a.period - 1) // a.period,
                      sig=tuple(a.sig), amp=tuple(a.amp),
                      kinds=a.kinds.split(","))
    traj, Fs, log = run_flow(x0, field, a.frames, a.period, 1.0 / a.fps,
                             a.safety, keep_F=a.save_F)
    dmin = min(q[3] for q in log)
    nsub = [q[1] for q in log]
    disp = float((traj[-1] - traj[0]).norm(dim=1).mean())
    print(f"[흐름] 입자 {len(idx)}  프레임 {a.frames}  주기 {a.period}  창 "
          f"{len(field)}  서브스텝 {min(nsub)}~{max(nsub)}  det F 최소 {dmin:.4f}  "
          f"평균 변위 {disp:.4f} (정규화 단위)", flush=True)
    print(f"[스텝] 한 스텝 사상 det 최소 {STEP_DET_MIN:.4f} (모든 입자·모든 스텝)",
          flush=True)
    assert dmin > 0 and STEP_DET_MIN > 0, "det <= 0 -- 흐름이 접혔다"
    np.savez_compressed(
        a.out, X0=Xn[idx].astype(np.float32),
        traj=traj.cpu().numpy().astype(np.float32),
        idx=idx, lo=lo, s=s, field=field,
        **({"F": Fs.cpu().numpy().astype(np.float32)} if a.save_F else {}))
    print(f"[저장] {a.out}", flush=True)
