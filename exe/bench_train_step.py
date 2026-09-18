"""학습 한 스텝의 시간이 어디로 가는지 쪼갠다.

unroll 16 으로 바꾸고 나서 한 스텝이 6.34 초다. 창 길이가 4 배가 됐는데 10 배
느려졌으므로 스텝당 비용 자체에도 뭔가 있다. 후보는 셋이다.

  자료 접근   궤적이 CPU 에 있어서 스텝마다 20000 행을 CPU 에서 모아 GPU 로 옮긴다
  야코비안    모양 손실(bures)이 J 를 요구해 자동미분용 순수 토치 경로를 탄다
  본 계산     kNN + 집계 + 어텐션 + 스키닝의 순전파/역전파

셋을 따로 재서 어디를 손대야 하는지 정한다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--traj", default="wolf_a")
ap.add_argument("--n_pts", type=int, default=20000)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--reps", type=int, default=20)
a = ap.parse_args()

dev = "cuda"
from anchorflow.deform import (DeformNet, aggregate, anchor_knn,  # noqa: E402
                               bc_features, bures_w2_sq, fps, skin,
                               skin_with_jacobian)

d = torch.load(os.path.join(a.data, a.traj + ".pt"), map_location="cpu",
               weights_only=False)
cfg = d["cfg"]; DT = float(cfg["frame_dt"])
X = d["x"]; F = d["F"]
EXT = float((X[0].max(0).values - X[0].min(0).values).norm())
N_FULL = X.shape[1]
gsel = torch.randperm(N_FULL)[:a.n_pts].sort().values
print(f"[설정] 궤적 {tuple(X.shape)}, F {tuple(F.shape)}, 표본 {a.n_pts}, "
      f"앵커 {a.n_anchors}", flush=True)


def timeit(fn, n=None):
    n = n or a.reps
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


# ---- 1. 자료 접근: CPU 상주 대 GPU 상주
def take_cpu(t, i):
    return t[i.cpu() if torch.is_tensor(i) else i].to(dev)


Xg = X.to(dev); Fg = F.to(dev); gg = gsel.to(dev)
print(f"[메모리] 궤적 하나를 GPU 에 올리면 "
      f"{(X.numel() + F.numel()) * 4 / 1e6:.0f} MB", flush=True)
t_cpu_x = timeit(lambda: take_cpu(X[5], gsel))
t_gpu_x = timeit(lambda: Xg[5][gg])
t_cpu_F = timeit(lambda: take_cpu(F[5], gsel))
t_gpu_F = timeit(lambda: Fg[5][gg])
print(f"[자료] 위치 한 번: CPU 상주 {t_cpu_x:.2f} ms -> GPU 상주 {t_gpu_x:.3f} ms", flush=True)
print(f"[자료] F  한 번: CPU 상주 {t_cpu_F:.2f} ms -> GPU 상주 {t_gpu_F:.3f} ms", flush=True)

# ---- 2. 한 스텝 순전파/역전파, J 있을 때와 없을 때
x = Xg[2][gg].clone()
v = (Xg[2][gg] - Xg[1][gg]) / DT
p = x[fps(x, a.n_anchors)]
m = torch.ones(a.n_pts, device=dev)
MAT = torch.zeros(7, device=dev)
idx, _ = anchor_knn(x, p, a.k)
feat, _ = aggregate(x, v, x, m, idx, a.n_anchors, 0.07, pa=p)
n_feat = feat.shape[-1] + MAT.numel() + bc_features(p[:2], cfg).shape[-1]
net = DeformNet(n_feat=n_feat, hidden=128, depth=4, heads=4,
                scale=0.02 * EXT, h=0.07, ext=EXT).to(dev)
ex = torch.cat([MAT.reshape(1, -1).expand(a.n_anchors, -1),
                bc_features(p, cfg) / 0.07], -1)
gt = Xg[3][gg]
F0 = Fg[2][gg].reshape(-1, 3, 3); F1 = Fg[3][gg].reshape(-1, 3, 3)
SIG0 = 0.005


def step(need_J, shape):
    net.zero_grad(set_to_none=True)
    ii, _ = anchor_knn(x, p, a.k)
    ff, _ = aggregate(x, v, x, m, ii, a.n_anchors, 0.07, pa=p)
    dp, lr_, lt_ = net(p, torch.cat([ff, ex], -1), DT)
    if need_J:
        x2, _, J = skin_with_jacobian(x, p, dp, lr_, lt_, ii, 0.07)
    else:
        x2, _ = skin(x, p, dp, lr_, lt_, ii, 0.07)
        J = None
    loss = ((x2 - gt) ** 2).sum(-1).mean() / (EXT ** 2)
    if shape and J is not None:
        Jgt = F1 @ torch.linalg.inv(F0 + 1e-4 * torch.eye(3, device=dev))
        Lt = SIG0 * F0
        loss = loss + bures_w2_sq(x2, gt, J @ Lt, Jgt @ Lt).mean() / (EXT ** 2)
    loss.backward()


t_plain = timeit(lambda: step(False, False))
t_J = timeit(lambda: step(True, False))
t_bures = timeit(lambda: step(True, True))
print(f"[스텝] 위치 손실만          {t_plain:6.2f} ms", flush=True)
print(f"[스텝] + 야코비안            {t_J:6.2f} ms  (+{t_J - t_plain:.2f})", flush=True)
print(f"[스텝] + 야코비안 + bures    {t_bures:6.2f} ms  (+{t_bures - t_J:.2f})", flush=True)
B, L = 4, 16
print(f"\n[추정] 배치 {B} x unroll {L} = {B * L} 스텝이면", flush=True)
print(f"  지금대로   {(t_bures * B * L + (t_cpu_x * 2 + t_cpu_F * 2) * B * L) / 1000:.2f} s/it",
      flush=True)
print(f"  자료를 GPU 로   {(t_bures * B * L) / 1000:.2f} s/it", flush=True)
print(f"  + 모양 손실을 4 스텝마다 "
      f"{((t_J * 3 + t_bures) / 4 * B * L) / 1000:.2f} s/it", flush=True)
print("TRAINBENCH_OK", flush=True)
