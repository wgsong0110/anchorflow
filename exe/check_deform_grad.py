"""변형 모델의 기울기가 실제로 맞는지, 갱신이 실제로 일어나는지 확인한다.

러닝 커브가 내려가지 않고 옆으로 누워 있었다. 그런 모양은 (a) 최적화가 어려운
것과 (b) 기울기 경로 어딘가가 끊긴 것이 똑같이 만들어내므로, 둘을 갈라야 한다.

네 가지를 차례로 본다. 앞의 것이 깨지면 뒤는 볼 필요가 없다.

  1. 도달   모든 파라미터에 0 이 아닌 기울기가 오는가. 한 덩어리라도 정확히 0 이면
            그 경로가 끊긴 것이다 (detach, 정수 색인, clamp 포화 등).
  2. 정확성 해석적 기울기가 유한차분과 맞는가. 무작위 방향으로 사영해 한 번에 본다.
  3. 갱신   optimizer.step() 뒤 파라미터가 실제로 움직이는가.
  4. 과적합 **창 하나**를 고정해 두고 손실을 떨어뜨릴 수 있는가. 이것이 결정적이다 --
            데이터가 하나뿐인데 못 내려가면 학습률이나 잡음 문제가 아니라 버그다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--n_pts", type=int, default=4000)
ap.add_argument("--lambda_J", type=float, default=0.1)
ap.add_argument("--overfit_iters", type=int, default=400)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--t0", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(a.seed)
from anchorflow.deform import (DeformNet, aggregate, bc_features,   # noqa: E402
                               grid_knn, jacobian_of, skin)

f = sorted(glob.glob(os.path.join(a.data, "*.pt")))[0]
d = torch.load(f, map_location="cpu", weights_only=False)
cfg = d["cfg"]
FRAME_DT = float(cfg["frame_dt"])
X0 = d["x"][0]
EXT = float((X0.max(0).values - X0.min(0).values).norm())
N_FULL = X0.shape[0]
print(f"[데이터] {os.path.basename(f)}, 프레임 {d['x'].shape[0]}, 물체 {EXT:.4f}",
      flush=True)


def fps(x, M, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.zeros(M, dtype=torch.long, device=x.device)
    idx[0] = torch.randint(x.shape[0], (1,), generator=g, device=x.device)
    dd = (x - x[idx[0]]).norm(dim=-1)
    for i in range(1, M):
        idx[i] = dd.argmax()
        dd = torch.minimum(dd, (x - x[idx[i]]).norm(dim=-1))
    return idx


X0d = X0.to(dev)
AIDX = fps(X0d, a.n_anchors, a.seed)
H = float(torch.cdist(X0d[AIDX], X0d[AIDX]).topk(2, largest=False).values[:, 1].median())
ng = int(cfg.get("n_grid", 100)); dx = float(cfg.get("grid_lim", 2.0)) / ng
vi = (X0d / dx).long().clamp(0, ng - 1)
flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cnt = torch.zeros(ng ** 3, device=dev).index_add_(0, flat, torch.ones(N_FULL, device=dev))
MASS = ((dx ** 3) / cnt[flat]) * float(cfg["density"])
VEL_SCALE = EXT / FRAME_DT
MAT = torch.cat([torch.tensor(
    [np.log(float(cfg["E"])), float(cfg["nu"]), float(cfg.get("xi", 0.0)),
     np.log(float(cfg["density"]))], device=dev, dtype=torch.float32),
    torch.tensor(cfg["g"], device=dev, dtype=torch.float32) / 15.0])

g = torch.Generator(device=dev).manual_seed(a.seed)
GS = torch.randperm(N_FULL, generator=g, device=dev)[:a.n_pts].sort().values


def take(t, i):
    return t[i.cpu()].to(dev)


XC = take(d["x"][0], GS)
x0 = take(d["x"][a.t0], GS)
v0 = (x0 - take(d["x"][a.t0 - 1], GS)) / FRAME_DT
p0 = take(d["x"][a.t0], AIDX)
gt = take(d["x"][a.t0 + 1], GS)
F0 = take(d["F"][a.t0], GS)
F1 = take(d["F"][a.t0 + 1], GS)
Jgt = F1 @ torch.linalg.inv(F0 + 1e-4 * torch.eye(3, device=dev))
disp = float((gt - x0).norm(dim=-1).mean())
print(f"[창] t0={a.t0}, 입자 {a.n_pts}, 앵커 간격 {H:.5f}, "
      f"한 프레임 평균 변위 {disp:.6f} = 물체의 {100*disp/EXT:.4f}%", flush=True)

idx, _ = grid_knn(x0, p0, a.k)
feat, _ = aggregate(x0, v0 / VEL_SCALE, XC, MASS[GS], idx, p0.shape[0], H, pa=p0)
n_bc = bc_features(p0[:2], cfg).shape[-1]
n_feat = feat.shape[-1] + MAT.numel() + n_bc
net = DeformNet(n_feat=n_feat, hidden=a.hidden, depth=a.depth, heads=a.heads,
                scale=0.02 * EXT, h=H, ext=EXT, seed=a.seed).to(dev)
print(f"[모델] 입력 {n_feat}, 파라미터 "
      f"{sum(q.numel() for q in net.parameters())/1e6:.2f}M, "
      f"출력 변위 단위 {0.02*EXT:.5f} (실제 변위의 {0.02*EXT/max(disp,1e-12):.1f}배)",
      flush=True)


def loss_fn(net_, lam=None):
    lam = a.lambda_J if lam is None else lam
    fe, _ = aggregate(x0, v0 / VEL_SCALE, XC, MASS[GS], idx, p0.shape[0], H, pa=p0)
    ex = torch.cat([MAT.reshape(1, -1).expand(p0.shape[0], -1),
                    bc_features(p0, cfg) / H], -1)
    dp, lr_, lt_ = net_(p0, torch.cat([fe, ex], -1), FRAME_DT)
    x2, _ = skin(x0, p0, dp, lr_, lt_, idx, H)
    lx = ((x2 - gt) ** 2).sum(-1).mean() / (EXT ** 2)
    if lam > 0:
        J = jacobian_of(lambda q: skin(q, p0, dp, lr_, lt_, idx, H)[0], x0)
        lJ = ((J - Jgt) ** 2).sum((-1, -2)).mean()
    else:
        lJ = torch.zeros((), device=dev)
    return lx + lam * lJ, lx, lJ, dp


rep = {}

# ---------------------------------------------------------------- 1. 도달
tot, lx, lJ, dp0 = loss_fn(net)
tot.backward()
dead, alive = [], []
for n, q in net.named_parameters():
    gnorm = 0.0 if q.grad is None else float(q.grad.norm())
    (dead if gnorm == 0.0 else alive).append((n, gnorm))
print(f"\n[1 도달] 손실 {float(tot):.4e} (위치 {float(lx):.4e}, J {float(lJ):.4e}), "
      f"초기 변위 크기 {float(dp0.norm(dim=-1).mean()):.3e}", flush=True)
print(f"  기울기 있음 {len(alive)} / 없음 {len(dead)}", flush=True)
for n, v in dead:
    print(f"    [끊김] {n}", flush=True)
for n, v in sorted(alive, key=lambda t: -t[1])[:6]:
    print(f"    {n:<40} |g| {v:.3e}", flush=True)
rep["dead"] = [n for n, _ in dead]
rep["alive"] = {n: v for n, v in alive}

# ---------------------------------------------------------------- 2. 정확성
# 무작위 방향 u 로 사영: (L(theta+eps u) - L(theta-eps u)) / 2eps 가 g.u 와 맞아야 한다
ps = [q for q in net.parameters() if q.grad is not None]
u = [torch.randn_like(q) for q in ps]
un = torch.sqrt(sum((t * t).sum() for t in u))
u = [t / un for t in u]
gdotu = float(sum((q.grad * t).sum() for q, t in zip(ps, u)))
print(f"\n[2 정확성] 해석적 방향도함수 {gdotu:.6e}", flush=True)
for eps in (1e-3, 1e-4, 1e-5):
    with torch.no_grad():
        for q, t in zip(ps, u):
            q += eps * t
    lp = float(loss_fn(net)[0])
    with torch.no_grad():
        for q, t in zip(ps, u):
            q -= 2 * eps * t
    lm = float(loss_fn(net)[0])
    with torch.no_grad():
        for q, t in zip(ps, u):
            q += eps * t
    fd = (lp - lm) / (2 * eps)
    err = abs(fd - gdotu) / max(abs(gdotu), 1e-30)
    print(f"  eps {eps:.0e}: 유한차분 {fd:.6e}  상대오차 {err:.3e}"
          f"{'  <- 불일치' if err > 0.05 else ''}", flush=True)
    rep[f"fd_{eps:g}"] = dict(fd=fd, rel_err=err)
rep["g_dot_u"] = gdotu

# ---------------------------------------------------------------- 3. 갱신
before = {n: q.detach().clone() for n, q in net.named_parameters()}
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
for _ in range(5):
    opt.zero_grad(set_to_none=True)
    loss_fn(net)[0].backward()
    opt.step()
moved = {n: float((q.detach() - before[n]).norm() / before[n].norm().clamp(min=1e-12))
         for n, q in net.named_parameters()}
still = [n for n, r in moved.items() if r == 0.0]
print(f"\n[3 갱신] 5 스텝 뒤 움직인 파라미터 {len(moved)-len(still)}/{len(moved)}",
      flush=True)
for n in still:
    print(f"    [안 움직임] {n}", flush=True)
for n, r in sorted(moved.items(), key=lambda t: -t[1])[:5]:
    print(f"    {n:<40} 상대변화 {r:.3e}", flush=True)
rep["not_moved"] = still

# ---------------------------------------------------------------- 4. 과적합
net2 = DeformNet(n_feat=n_feat, hidden=a.hidden, depth=a.depth, heads=a.heads,
                 scale=0.02 * EXT, h=H, ext=EXT, seed=a.seed).to(dev)
opt2 = torch.optim.Adam(net2.parameters(), lr=a.lr)
still_err = float(((x0 - gt) ** 2).sum(-1).mean()) / (EXT ** 2)
curve = []
for i in range(a.overfit_iters):
    opt2.zero_grad(set_to_none=True)
    tot, lx, lJ, _ = loss_fn(net2)
    tot.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(net2.parameters(), 1.0))
    opt2.step()
    curve.append((float(lx), float(lJ), gn))
    if i % max(a.overfit_iters // 10, 1) == 0 or i == a.overfit_iters - 1:
        print(f"  it {i:4d}  위치 RMSE {100*float(lx)**0.5:7.4f}%  "
              f"(정지 {100*still_err**0.5:.4f}%)  J {float(lJ):.3e}  |g| {gn:.2e}",
              flush=True)
c = np.array(curve)
print(f"\n[4 과적합] 위치 RMSE {100*c[0,0]**0.5:.4f}% -> {100*c[-1,0]**0.5:.4f}% "
      f"(최저 {100*c[:,0].min()**0.5:.4f}%), 정지 기준선 {100*still_err**0.5:.4f}%",
      flush=True)
ok = c[:, 0].min() < 0.25 * c[0, 0]
print(f"  창 하나도 못 맞추면 버그다 -> {'통과' if ok else '실패'}", flush=True)
rep["overfit"] = dict(first=float(c[0, 0]), last=float(c[-1, 0]),
                      best=float(c[:, 0].min()), still=still_err, pass_=bool(ok))

if a.out:
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, "grad_check.json"), "w"), indent=1,
              ensure_ascii=False)
print("GRADCHK_OK")
