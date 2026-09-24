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

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True, help="교사 궤적 (.pt)")
ap.add_argument("--t0", type=int, default=3)
ap.add_argument("--len", type=int, default=40)
ap.add_argument("--iters", type=int, default=60, help="스텝당 L-BFGS 반복")
ap.add_argument("--k", type=int, default=16, help="변형구배 최소제곱 이웃 수")
ap.add_argument("--lr", type=float, default=1.0)
ap.add_argument("--out", required=True, help="덤프 경로 (.pt)")
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
for sel in mem:
    free[sel] = False
print(f"[ip] 입자 {N}, 손잡이 {int((~free).sum())}, h {h:.5f}, 물체 {EXT:.4f}",
      flush=True)


def defgrad(x_new, x_old, F_old):
    """국소 최소제곱으로 한 스텝 야코비안을 잡고 F 를 밀어 준다."""
    d0 = x_old[idx] - x_old.unsqueeze(1)
    d1 = x_new[idx] - x_new.unsqueeze(1)
    w = 1.0 / (d0.norm(dim=-1, keepdim=True) ** 2 + 1e-12)
    w = w / w.sum(1, keepdim=True)
    A = torch.einsum("nkc,nki,nkj->nij", w, d1, d0)
    B = torch.einsum("nkc,nki,nkj->nij", w, d0, d0)
    B = B + 1e-10 * torch.eye(3, device=dev)
    return (A @ torch.linalg.inv(B)) @ F_old


x = X[a.t0].clone()
v = (x - X[max(a.t0 - 1, 0)]) / h
F = phys_resid.rebuild_F(X[:a.t0 + 1], cfg, h, k=a.k)[a.t0].float().to(dev)
preds, gts = [], []
t_start = time.time()
for i in tqdm(range(a.len), desc="암시적 스텝", ncols=80):
    t = a.t0 + i
    if t + 1 >= X.shape[0]:
        break
    xtil = (x + h * v).detach()
    x_old = x.detach()
    F_old = F.detach()
    # 손잡이의 이번 스텝 목표 위치 (교사가 실제로 끌고 간 곳)
    tgt = {}
    for kk in range(len(mem)):
        if mem[kk].numel():
            tgt[kk] = P[min(t + 1, P.shape[0] - 1), kk] + off[kk]
    q = (xtil.clone()).requires_grad_(True)
    opt = torch.optim.LBFGS([q], lr=a.lr, max_iter=a.iters,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        xf = q.clone()
        for kk, tv in tgt.items():                 # Dirichlet 은 대입한다
            xf = xf.index_copy(0, mem[kk], tv)
        F_tr = defgrad(xf, x_old, F_old)
        E, _dl, _pt = phys_resid.ip_energy(
            xf, xtil, F_tr, mass, vol, cfg, h, free=free, g=g,
            norm=float(mass.sum()) * EXT ** 2 / h ** 2)
        E.backward()
        return E

    opt.step(closure)
    with torch.no_grad():
        x_new = q.detach()
        for kk, tv in tgt.items():
            x_new = x_new.index_copy(0, mem[kk], tv)
        F_tr = defgrad(x_new, x_old, F_old)
        _psi, dlog = phys_resid.psi_of(F_tr, cfg, h)
        F = phys_resid.plastic_step(F_tr, dlog)
        v = (x_new - x_old) / h
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
print(f"[ip] {len(preds)} 프레임, 교사 대비 평균 {100*float(e.mean()):.3f}% "
      f"끝 {100*float(e[-1]):.3f}%  ({time.time()-t_start:.0f}초)", flush=True)
