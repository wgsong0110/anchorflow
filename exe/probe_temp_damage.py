"""앵커 온도가 파괴와 함께 내려가는지, 손상 변수와 얼마나 같이 움직이는지 잰다.

온도가 무엇을 하는지부터: 스키닝 가중치가 w = softmax(g / tau), g = -d^2/(2 r^2)
이고 tau 는 그 자리 앵커들의 온도를 섞은 값이다. tau 가 크면 분포가 납작해져 여러
앵커가 함께 끌고(연속체), 작으면 뾰족해져 한 앵커가 독점한다(조각이 따로 움직일 수
있다). 그래서 "찢어지는 자리에서 온도가 내려간다" 는 검증할 수 있는 예측이다.

손상 변수는 GT 위치에서 만든다. t0 배치에서 각 앵커의 이웃 입자 32 개를 붙들고,
그 이웃들이 t0 이후 얼마나 흩어졌는지를 두 가지로 잰다.

  stretch   이웃들로 최소제곱한 국소 변형구배 F 의 최대 특이값 (1 이면 안 늘어남)
  sep       t0 에서 서로 이웃이던 쌍들의 거리가 t0 대비 몇 배가 되었나의 중앙값

둘 다 '재질이 그 자리에서 끊어졌나' 의 대리값이고, 앵커마다 한 숫자다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--traj", default="watermelon_h")
ap.add_argument("--out", required=True)
ap.add_argument("--t0", type=int, default=3)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--n_pts", type=int, default=20000)
ap.add_argument("--nb", type=int, default=32, help="앵커마다 붙들 이웃 입자 수")
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
from anchorflow.deform import (DeformNet, _outer_sum, aggregate,   # noqa: E402
                               anchor_knn, bc_features, dense_knn,
                               skin_with_jacobian)

d = torch.load(os.path.join(a.data, a.traj + ".pt"), map_location="cpu",
               weights_only=False)
cfg = d["cfg"]; FRAME_DT = float(cfg["frame_dt"])
X0 = d["x"][0]
EXT = float((X0.max(0).values - X0.min(0).values).norm())
N_FULL = X0.shape[0]
GS = torch.arange(min(a.n_pts, N_FULL))
ng = int(cfg.get("n_grid", 100)); dx = float(cfg.get("grid_lim", 2.0)) / ng
X0d = X0.to(dev)
vi = (X0d / dx).long().clamp(0, ng - 1)
flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cnt = torch.zeros(ng ** 3, device=dev).index_add_(0, flat,
                                                  torch.ones(N_FULL, device=dev))
MASS = ((dx ** 3) / cnt[flat]) * float(cfg["density"])
VEL = EXT / FRAME_DT
MAT = torch.cat([torch.tensor([np.log(float(cfg["E"])), float(cfg["nu"]),
                               float(cfg.get("xi", 0.)), np.log(float(cfg["density"]))],
                              device=dev, dtype=torch.float32),
                 torch.tensor(cfg["g"], device=dev, dtype=torch.float32) / 15.0])


def take(t, i):
    return t[i.cpu() if torch.is_tensor(i) else i].to(dev)


st = torch.load(a.ckpt, map_location=dev, weights_only=False)
ta_ = st["args"]; AIDX = st["aidx"].to(dev); H = float(st["H"]); K = int(ta_["k"])
net = DeformNet(n_feat=int(st["n_feat"]), hidden=int(ta_["hidden"]),
                depth=int(ta_["depth"]), heads=int(ta_["heads"]),
                scale=0.02 * EXT, h=H, ext=EXT, seed=int(ta_["seed"])).to(dev)
net.load_state_dict(st["net"]); net.eval()

# --- 앵커가 붙들 GT 이웃을 t0 배치에서 한 번 정한다 (막이 바뀌면 비교가 안 된다)
Xr = take(d["x"][a.t0], GS)
PA0 = take(d["x"][a.t0], AIDX)
NB, _ = dense_knn(PA0, Xr, a.nb)                 # [M, nb] 앵커별 이웃 입자
M = PA0.shape[0]
dX = Xr[NB] - Xr[NB].mean(1, keepdim=True)       # [M,nb,3] 이웃 무게중심 기준
D0 = _outer_sum(dX, dX)
eps = 1e-6 * torch.diagonal(D0, dim1=-2, dim2=-1).sum(-1) / 3.0
D0i = torch.linalg.inv(D0 + eps[:, None, None] * torch.eye(3, device=dev))
# 이웃 쌍 거리(같은 앵커에 속한 이웃끼리) -- 찢어지면 이 거리가 벌어진다
PD0 = torch.cdist(Xr[NB], Xr[NB])                # [M,nb,nb]
print(f"[설정] 앵커 {M}, 앵커별 이웃 {a.nb}, 물체 {EXT:.4f}", flush=True)

# --- 모델 롤아웃
x = take(d["x"][a.t0], GS)
v = (x - take(d["x"][max(a.t0 - 1, 0)], GS)) / FRAME_DT
p = PA0.clone(); XC = take(d["x"][0], GS)
rows = []
TT, RR, SS, SEP = [], [], [], []
for i in range(a.frames):
    idx, _ = anchor_knn(x, p, K)
    feat, _ = aggregate(x, v / VEL, XC, MASS[GS.to(dev)], idx, M, H, pa=p)
    ex = torch.cat([MAT.reshape(1, -1).expand(M, -1), bc_features(p, cfg) / H], -1)
    dp, lr_, lt_ = net(p, torch.cat([feat, ex], -1), FRAME_DT)
    x2, _, _J = skin_with_jacobian(x, p, dp, lr_, lt_, idx, H)
    v, p, x = (x2 - x) / FRAME_DT, p + dp, x2

    # 같은 프레임의 GT 에서 손상 변수
    xg = take(d["x"][a.t0 + i + 1], GS)
    dxc = xg[NB] - xg[NB].mean(1, keepdim=True)
    F = _outer_sum(dxc, dX) @ D0i
    s1 = torch.linalg.svdvals(F)[..., 0]
    pdt = torch.cdist(xg[NB], xg[NB])
    ratio = (pdt / PD0.clamp(min=1e-9))
    iu = torch.triu_indices(a.nb, a.nb, offset=1, device=dev)
    sep = ratio[:, iu[0], iu[1]].median(dim=-1).values

    TT.append(lt_.clone()); RR.append(lr_.exp().clone() / H)
    SS.append(s1.clone()); SEP.append(sep.clone())
    rows.append(dict(f=a.t0 + i + 1,
                     t_med=float(lt_.exp().median()),
                     t_mean=float(lt_.exp().mean()),
                     logt_mean=float(lt_.mean()),
                     r_med=float((lr_.exp() / H).median()),
                     s1_med=float(s1.median()), s1_p99=float(s1.quantile(0.99)),
                     sep_med=float(sep.median()),
                     sep_p99=float(sep.quantile(0.99))))

T = torch.stack(TT); R = torch.stack(RR); S = torch.stack(SS); P = torch.stack(SEP)


def corr(u, w):
    u = u.reshape(-1).double(); w = w.reshape(-1).double()
    u = u - u.mean(); w = w - w.mean()
    return float((u * w).sum() / (u.norm() * w.norm()).clamp(min=1e-30))


def spearman(u, w):
    ru = u.reshape(-1).argsort().argsort().double()
    rw = w.reshape(-1).argsort().argsort().double()
    return corr(ru, rw)


print("\n[프레임별] f  온도중앙  반경중앙  sigma1중앙(p99)  분리중앙(p99)", flush=True)
for r in rows:
    print(f"  {r['f']:3d}  {r['t_med']:8.3f}  {r['r_med']:7.3f}  "
          f"{r['s1_med']:6.3f} ({r['s1_p99']:7.3f})  "
          f"{r['sep_med']:6.3f} ({r['sep_p99']:7.3f})", flush=True)

print("\n[상관] 앵커·프레임을 전부 모아서 (n = %d)" % T.numel(), flush=True)
for nm, dmg in (("sigma1", S), ("분리비", P)):
    print(f"  log t_a  vs  {nm}:  피어슨 {corr(T, dmg):+.3f}  "
          f"스피어만 {spearman(T, dmg):+.3f}", flush=True)
    print(f"  r_a/h    vs  {nm}:  피어슨 {corr(R, dmg):+.3f}  "
          f"스피어만 {spearman(R, dmg):+.3f}", flush=True)

# 프레임 안에서만 본 상관 (프레임끼리의 추세를 빼고 앵커 사이 대비만)
pf = [spearman(T[i] - T[i].mean(), S[i] - S[i].mean()) for i in range(len(rows))]
print(f"\n[프레임 내부 상관] log t_a vs sigma1, 스피어만 중앙 {np.median(pf):+.3f} "
      f"(최소 {min(pf):+.3f}, 최대 {max(pf):+.3f})", flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(dict(traj=a.traj, t0=a.t0, rows=rows,
               corr_logt_s1=corr(T, S), corr_logt_sep=corr(T, P),
               spear_logt_s1=spearman(T, S), spear_logt_sep=spearman(T, P),
               corr_r_s1=corr(R, S), spear_r_s1=spearman(R, S),
               per_frame_spear=pf),
          open(a.out, "w"), indent=1, ensure_ascii=False)
print(f"\n[저장] {a.out}", flush=True)
print("TEMPDMG_OK", flush=True)
