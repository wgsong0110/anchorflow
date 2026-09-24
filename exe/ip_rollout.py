"""한 스텝을 **증분 포텐셜 최소화**로 풀어 궤적을 만든다 (학생 없이).

Phase 2 가 손실로 쓰는 바로 그 목적함수를, 학습 대신 **직접 최적화**한다:

    x^{n+1} = argmin_x  sum_p m_p/(2h^2)|x_p - xtil_p|^2 + sum_p V_p Psi(F_p(x)) - m g.x
    xtil = x^n + h v^n

i-PhysGaussian 이 암시적 MPM 으로 하는 일을, 격자 대신 **입자 이웃의 최소제곱**으로
변형구배를 잡아 무격자로 하는 셈이다. 이 궤적이 좋으면 Phase 2 의 목적함수가 맞다는
뜻이고, 나쁘면 목적함수 자체가 교사와 다른 해를 가리킨다는 뜻이다 -- 학생의 학습
문제와 목적함수의 문제를 갈라 보기 위한 도구다.

손잡이는 교사와 같은 Dirichlet: 반경 안 입자는 궤적의 제어점 위치로 **덮어쓰고**
관성·중력 항에서 뺀다. 그 위치는 Psi 를 통해 나머지를 끌어당긴다.
"""
import argparse
import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from anchorflow import phys_resid                                # noqa: E402
from anchorflow import trilinear as TRI                          # noqa: E402
from anchorflow import vox_anchor                                # noqa: E402
from anchorflow.deform import skin_with_jacobian                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True, help="교사 궤적 (.pt)")
ap.add_argument("--t0", type=int, default=3)
ap.add_argument("--len", type=int, default=40)
ap.add_argument("--iters", type=int, default=60, help="스텝당 L-BFGS 반복")
ap.add_argument("--sub", type=int, default=1,
                help="프레임을 이만큼 쪼개 푼다. dt 를 줄여 PG 로 수렴하는지 보는 "
                     "것이 구현 오류와 적분기 감쇠를 가르는 시험이다")
ap.add_argument("--k", type=int, default=16, help="변형구배 최소제곱 이웃 수")
ap.add_argument("--lr", type=float, default=1.0)
ap.add_argument("--out", required=True, help="덤프 경로 (.pt)")
ap.add_argument("--var", default="grid", choices=("grid", "pts"),
                help="최적화 변수: grid=격자점 변위(모델과 같은 가설 공간), "
                     "pts=입자 위치 직접")
ap.add_argument("--vox_res", type=int, default=32, help="격자 한 변 칸 수")
ap.add_argument("--plast", default="frame", choices=("frame", "sub"),
                help="소성 사영을 상태에 반영하는 주기. F 가 프레임 해상도로 "
                     "복원된 값이라, 서브스텝마다 반영하면 사영 횟수만큼 소성이 "
                     "과하게 쌓인다 (실측: dt 를 줄일수록 교사에서 멀어졌다)")
ap.add_argument("--drive", default="dirichlet", choices=("dirichlet", "force"),
                help="구동 방식. dirichlet=손잡이 위치를 덮어쓴다(KIN), "
                     "force=가한 가속도를 **외력 항**으로 넣고 입자는 자유롭게 둔다")
ap.add_argument("--no_bc", action="store_true", help="경계조건을 끈다 (대조용)")
ap.add_argument("--dev", default="cuda")
a = ap.parse_args()

D = torch.load(a.traj, map_location="cpu", weights_only=False)
dev = a.dev
X = D["x"].float().to(dev)
cfg = D["cfg"]
h = float(cfg["frame_dt"])
EXT = float((X[0].max(0).values - X[0].min(0).values).norm())
N = X.shape[1]
g = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)

# 질량: 교사와 같은 방식(격자 점유로 부피, config 밀도)
ng = int(cfg.get("n_grid", 100))
dx = float(cfg.get("grid_lim", 2.0)) / ng
vi = (X[0] / dx).long().clamp(0, ng - 1)
flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cnt = torch.zeros(ng ** 3, device=dev).index_add_(
    0, flat, torch.ones(N, device=dev))
mass = ((dx ** 3) / cnt[flat]) * float(cfg["density"])
vol = mass / float(cfg["density"])

# 변형구배용 이웃 (기준 배치에서 한 번)
idx = torch.empty(N, a.k, dtype=torch.long, device=dev)
for s in range(0, N, 4096):
    e = min(s + 4096, N)
    idx[s:e] = torch.cdist(X[0][s:e], X[0]).topk(
        a.k + 1, largest=False).indices[:, 1:]

# 손잡이: 궤적의 제어점 무리와 그 상대 위치 (교사가 강체로 끌고 간다)
P = D["ctrl_pos"].float().to(dev)                  # [T,k,3] 실제 제어 입자 위치
R = D["ctrl_R"].float().to(dev)
hid = D["ctrl_id"].long().to(dev)
mem, off = [], []
for kk in range(P.shape[1]):
    r = float(R[0] if R.ndim == 1 else R[0, kk])
    sel = torch.nonzero((X[0] - P[0, kk]).norm(dim=-1) < r).squeeze(-1)
    mem.append(sel)
    off.append(X[0][sel] - P[0, kk])
free = torch.ones(N, dtype=torch.bool, device=dev)
if a.drive == "dirichlet":
    for sel in mem:
        free[sel] = False
# 힘 모드의 감쇠 가중: 교사와 같은 (1-q^2)^2
WFAL = []
for kk in range(P.shape[1]):
    r = float(R[0] if R.ndim == 1 else R[0, kk])
    q2 = ((X[0][mem[kk]] - P[0, kk]).norm(dim=-1) / max(r, 1e-9)).clamp(max=1.0) ** 2
    WFAL.append((1.0 - q2) ** 2)
ACC = (D.get("ctrl_vel").float().to(dev) if D.get("ctrl_vel") is not None
       else torch.zeros(1, P.shape[1], 3, device=dev))
print(f"[ip] 입자 {N}, 손잡이 {int((~free).sum())}, 구동 {a.drive}, "
      f"h {h:.5f}, 물체 {EXT:.4f}", flush=True)


BC = []
for _b in cfg.get("boundary_conditions", []):
    if _b.get("type") == "surface_collider":
        BC.append(("plane",
                   torch.tensor(_b["point"], device=dev, dtype=torch.float32),
                   torch.tensor(_b["normal"], device=dev, dtype=torch.float32),
                   _b.get("surface", "sticky")))
    elif _b.get("type") == "bounding_box":
        BC.append(("box", None, None, None))
_pad = 3.0 * dx


def apply_bc(xq):
    """PG 와 같은 경계: 바닥 평면은 sticky, 격자 가장자리는 상자.

    이것이 빠져 있으면 물체가 바닥을 뚫고 지나가 교사와 갈라진다 -- 목적함수가
    아니라 **구속이 빠진** 것이다.
    """
    if a.no_bc:
        return xq
    for kind, pt, nl, surf in BC:
        if kind == "plane":
            d = ((xq - pt) * nl).sum(-1, keepdim=True)
            xq = torch.where(d < 0, xq - d * nl, xq)
        else:
            lim = float(cfg.get("grid_lim", 2.0))
            xq = xq.clamp(_pad, lim - _pad)
    return xq


def defgrad(x_new, x_old, F_old, dt):
    """F <- (I + dt grad v) F. **속도 구배**로 민다.

    위치 차이로 J 를 직접 맞추면 dt 가 작을 때 d1 과 d0 이 거의 같아져 항등에 가까운
    행렬을 "거의 같은 두 값의 차" 로 구하게 되고, float32 에서 유효숫자가 날아간다.
    그 오차가 스텝 수만큼 쌓여 **dt 를 줄일수록 나빠지는** 거동이 된다 (실측했다).
    속도 구배는 O(1) 이라 같은 문제가 없다. 누적은 float64 로 한다.
    """
    d0 = (x_old[idx] - x_old.unsqueeze(1)).double()
    dv = ((x_new[idx] - x_new.unsqueeze(1)).double() - d0) / dt
    w = 1.0 / (d0.norm(dim=-1, keepdim=True) ** 2 + 1e-14)
    w = w / w.sum(1, keepdim=True)
    A = torch.einsum("nkc,nki,nkj->nij", w, dv, d0)
    B = torch.einsum("nkc,nki,nkj->nij", w, d0, d0)
    B = B + 1e-12 * torch.eye(3, device=dev, dtype=torch.float64)
    gv = A @ torch.linalg.inv(B)                          # grad v [N,3,3]
    I3 = torch.eye(3, device=dev, dtype=torch.float64)
    return ((I3 + dt * gv) @ F_old.double()).to(F_old.dtype)


x = X[a.t0].clone()
v = (x - X[max(a.t0 - 1, 0)]) / h
# 궤적이 교사의 진짜 F 를 담고 있으면 그것을 쓴다. 예전 궤적은 f_tensor 키를
# 잘못 읽어 항등만 들어 있고, PG 는 첫 프레임을 채우기 전에 덤프해 0 이 들어간다.
_Fd = D.get("F")
_ok = False
if _Fd is not None and _Fd.shape[0] > a.t0:
    _f0 = _Fd[a.t0].float().to(dev)
    _ok = (float(_f0.abs().max()) > 1e-6
           and float((_f0 - torch.eye(3, device=dev)).abs().max()) > 1e-6)
if _ok:
    F = _f0
    print("[ip] 궤적의 F 를 그대로 쓴다", flush=True)
else:
    F = phys_resid.rebuild_F(X[:a.t0 + 1], cfg, h, k=a.k)[a.t0].float().to(dev)
    print("[ip] 궤적에 쓸 만한 F 가 없어 위치에서 복원한다", flush=True)
preds, gts = [], []
t_start = time.time()
hs = h / a.sub
# 정규화는 **프레임 간격으로 고정**한다. 서브스텝 dt 로 잡으면 dt 를 줄일수록
# 목적함수와 그 기울기가 dt^-2 로 작아져, L-BFGS 가 수렴 허용오차에 먼저 걸려
# 거의 풀지 않고 끝난다 -- dt 를 줄였더니 오차가 커지던 것이 이것이었다.
NORM = float(mass.sum()) * EXT ** 2 / h ** 2
for i in tqdm(range(a.len), desc="암시적 스텝", ncols=80):
    t = a.t0 + i
    if t + 1 >= X.shape[0]:
        break
    for _si in range(a.sub):
        xtil = (x + hs * v).detach()
        x_old = x.detach()
        F_old = F.detach()
        # 손잡이는 프레임 안에서 선형으로 보간해 끌고 간다
        _al = (_si + 1.0) / a.sub
        _pc = ((1.0 - _al) * P[min(t, P.shape[0] - 1)]
               + _al * P[min(t + 1, P.shape[0] - 1)])
        tgt = {kk: _pc[kk] + off[kk]
               for kk in range(len(mem)) if mem[kk].numel()}

        def _fix(xq):
            """Dirichlet 대입 + 경계. 에너지는 **이 배치에서** 잰다."""
            if a.drive == "dirichlet":
                for kk, tv in tgt.items():
                    xq = xq.index_copy(0, mem[kk], tv)
            return apply_bc(xq)

        if a.var == "grid":
            # 학생과 **같은 가설 공간**: 격자를 현재 배치에 맞춰 잡고, 변수는
            # 격자점 변위 dp 다. 가우시안은 그 dp 를 꼭짓점 8 개 스키닝으로 받고,
            # F 도 그 사상의 해석적 야코비안으로 민다 -- step_once 와 같은 경로다.
            lo, hh, nn3 = vox_anchor.grid_for(x_old, a.vox_res ** 3)
            sidx, _w8 = TRI.corners(x_old, lo, hh, nn3)
            gpos = (torch.stack(torch.meshgrid(
                *[torch.arange(int(nn3[d]), device=dev, dtype=x_old.dtype)
                  for d in range(3)], indexing="ij"), -1).reshape(-1, 3)
                ) * hh + lo
            log_r = torch.full((gpos.shape[0],), float(torch.log(
                torch.tensor(float(hh)))), device=dev)
            log_t = torch.zeros_like(log_r)
            dp = torch.zeros(gpos.shape[0], 3, device=dev, requires_grad=True)
            var = [dp]

            def _state():
                xs, _w, J = skin_with_jacobian(
                    x_old, gpos, dp, log_r, log_t, sidx, float(hh))
                return _fix(xs), J
        else:
            q = xtil.clone().requires_grad_(True)
            var = [q]

            def _state():
                xs = _fix(q.clone())
                return xs, None

        opt = torch.optim.LBFGS(var, lr=a.lr, max_iter=a.iters,
                                history_size=20,
                                tolerance_grad=1e-14, tolerance_change=1e-16,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            xf, J = _state()
            F_tr = (J @ F_old) if J is not None else defgrad(xf, x_old, F_old, hs)
            E, _dl, _pt = phys_resid.ip_energy(
                xf, xtil, F_tr, mass, vol, cfg, hs, free=free, g=g, norm=NORM)
            if a.drive == "force":
                # 외력은 중력과 같은 자리에 들어간다: -sum m_p w_p a . x
                for kk in range(len(mem)):
                    if not mem[kk].numel():
                        continue
                    acc = ACC[min(t, ACC.shape[0] - 1), kk]
                    E = E - (mass[mem[kk]] * WFAL[kk]
                             * (xf[mem[kk]] * acc).sum(-1)).sum() / NORM
            E.backward()
            return E

        opt.step(closure)
        xf, J = _state()
        x_new = xf.detach()
        F_tr = ((J @ F_old) if J is not None
                else defgrad(x_new, x_old, F_old, hs)).detach()
        if a.plast == "sub" or _si == a.sub - 1:
            _psi, dlog = phys_resid.psi_of(F_tr, cfg, hs)
            F = phys_resid.plastic_step(F_tr, dlog)
        else:
            F = F_tr                      # 응력은 사영된 변형률로 이미 쟀다
        v = ((x_new - x_old) / hs).detach()
        x = x_new
    preds.append(x.detach().cpu())
    gts.append(X[t + 1].detach().cpu())
    err = float((x - X[t + 1]).norm(dim=-1).mean()) / EXT
    tqdm.write(f"  t={t:3d}  교사 대비 {100*err:.3f}%")

P_ = torch.stack(preds)
G_ = torch.stack(gts)
torch.save({"pred": P_, "gt": G_, "ctrl_pos": D.get("ctrl_pos"),
            "t0": a.t0, "EXT": EXT, "tag": D.get("tag", "ip")}, a.out)
e = (P_ - G_).norm(dim=-1).mean(-1) / EXT
print(f"[ip] 서브스텝 {a.sub} (dt {hs:.2e}), {len(preds)} 프레임, "
      f"교사 대비 평균 {100*float(e.mean()):.3f}% 끝 {100*float(e[-1]):.3f}%  "
      f"({time.time()-t_start:.0f}초)", flush=True)
