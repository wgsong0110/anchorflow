"""초기 배치 대비 누적 변형장을 단면 히트맵으로 그린다 (최대 특이값 또는 변형률 노름).

sigma_1(F_t) 는 그 자리에서 재질이 가장 크게 늘어난 방향의 배율이고, Green-Lagrange
변형률 E = (F^T F - I)/2 의 노름은 "변형이 전혀 없으면 0" 이라는 기준점을 갖는다 --
sigma_1 은 변형이 없어도 1 이라 배경과 변형을 눈으로 가르기 어렵다.

sigma_1(F_t) 는 그 자리에서 재질이 가장 크게 늘어난 방향의 배율이다. 1 이면 그
방향으로 원래 길이 그대로, 2 면 두 배로 늘어난 것이고, 갈라지는 자리에서 치솟는다.
회전은 특이값을 바꾸지 않으므로 순수한 변형만 남고, 물체가 통째로 떨어지는 성분도
미분에서 자동으로 사라진다 -- 한 스텝 벡터장 영상(render_deform_field.py)에서
중심을 빼줘야 했던 것과 달리 여기서는 뺄 것이 없다.

GT 는 MPM 이 입자마다 들고 있는 변형구배 f_tensor 를 그대로 쓰고, 모델은 매 프레임
야코비안을 곱해 누적한 것을 쓴다. 단면과 격자는 벡터장 영상과 같은 규칙이다.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", default=None)
ap.add_argument("--traj", default="watermelon_h")
ap.add_argument("--out", required=True)
ap.add_argument("--t0", type=int, default=3)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--n_pts", type=int, default=20000)
ap.add_argument("--slices", type=int, default=5)
ap.add_argument("--grid", type=int, default=48)
ap.add_argument("--band", type=float, default=0.06)
ap.add_argument("--metric", default="sigma1", choices=("sigma1", "strain"),
                help="sigma1 은 F 의 최대 특이값(가장 크게 늘어난 배율), strain 은 "
                     "Green-Lagrange 변형률 E=(F^T F - I)/2 의 Frobenius 노름. "
                     "둘 다 회전에 불변이고, strain 은 변형이 없으면 정확히 0 이다")
ap.add_argument("--vmax", type=float, default=0.0,
                help="색 상한. 0 이면 전체 프레임의 p99 로 한 번 고정한다")
ap.add_argument("--mask", type=float, default=1.2,
                help="격자점에서 가장 가까운 입자까지 이 배(복셀 한 변 기준)를 넘으면 "
                     "재질이 없는 자리로 보고 가린다")
ap.add_argument("--cell", type=float, default=0.0,
                help="가림 판정의 길이 단위. 0 이면 입자 간격 중앙값에서 잡는다")
ap.add_argument("--tile", type=int, default=240)
ap.add_argument("--fps", type=int, default=10)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anchorflow import ptrender
from anchorflow.deform import (DeformNet, aggregate, anchor_knn, bc_features,
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
cnt = torch.zeros(ng ** 3, device=dev).index_add_(0, flat, torch.ones(N_FULL, device=dev))
MASS = ((dx ** 3) / cnt[flat]) * float(cfg["density"])
VEL = EXT / FRAME_DT
MAT = torch.cat([torch.tensor([np.log(float(cfg["E"])), float(cfg["nu"]),
                               float(cfg.get("xi", 0.)), np.log(float(cfg["density"]))],
                              device=dev, dtype=torch.float32),
                 torch.tensor(cfg["g"], device=dev, dtype=torch.float32) / 15.0])


def take(t, i):
    return t[i.cpu() if torch.is_tensor(i) else i].to(dev)


xc0 = take(d["x"][0], GS); c0 = xc0.mean(0); rel0 = xc0 - c0
zlo, zhi = float(rel0[:, 2].min()), float(rel0[:, 2].max())
edges = np.linspace(zlo, zhi, a.slices + 1)
ZS = [(edges[i] + edges[i + 1]) / 2 for i in range(a.slices)]
xy_lo = rel0[:, :2].min(0).values.cpu().numpy()
xy_hi = rel0[:, :2].max(0).values.cpu().numpy()
gx = np.linspace(xy_lo[0], xy_hi[0], a.grid)
gy = np.linspace(xy_lo[1], xy_hi[1], a.grid)
GX, GY = np.meshgrid(gx, gy, indexing="ij")
print(f"[단면] 상대 z {zlo:.3f}~{zhi:.3f} {a.slices} 등분, 격자 {a.grid}x{a.grid}",
      flush=True)


def measure(F):
    """F 에서 한 스칼라. 회전에 불변한 것만 쓴다."""
    if a.metric == "sigma1":
        return torch.linalg.svdvals(F)[..., 0]
    I = torch.eye(3, device=F.device)
    E = 0.5 * (F.transpose(-1, -2) @ F - I)
    return E.reshape(*E.shape[:-2], 9).norm(dim=-1)


# 가림 판정의 길이 단위: 입자 간격 중앙값 (없으면 물체의 2%)
if a.cell > 0:
    CELL = a.cell
else:
    _s = xc0[torch.randperm(xc0.shape[0], device=dev)[:2000]]
    CELL = float(torch.cdist(_s, _s).topk(2, largest=False).values[:, 1].median())
print(f"[가림] 길이 단위 {CELL:.5f}, 문턱 {a.mask}배", flush=True)


def to_grid(vals, pos_rel, z):
    """입자 값을 단면 격자로 옮긴다. 재질이 없는 자리는 NaN 으로 남겨 가린다.

    가리지 않으면 역거리 가중이 물체 밖까지 값을 퍼뜨려, 수박이 없는 배경이
    변형된 것처럼 보인다.
    """
    sel = (pos_rel[:, 2] - z).abs() < a.band * EXT
    if int(sel.sum()) < 8:
        return None
    q = torch.from_numpy(np.stack([GX, GY], -1).reshape(-1, 2)).float().to(dev)
    dd = torch.cdist(q, pos_rel[sel][:, :2])
    w = 1.0 / (dd + 0.02 * EXT) ** 2
    w = w / w.sum(1, keepdim=True)
    g = (w @ vals[sel].unsqueeze(-1)).squeeze(-1)
    g = torch.where(dd.min(1).values < a.mask * CELL, g,
                    torch.full_like(g, float("nan")))
    return g.reshape(a.grid, a.grid).cpu().numpy()


# 모델 롤아웃: 야코비안을 누적한다
model = None
if a.ckpt:
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ta = st["args"]; AIDX = st["aidx"].to(dev); H = float(st["H"]); K = int(ta["k"])
    net = DeformNet(n_feat=int(st["n_feat"]), hidden=int(ta["hidden"]),
                    depth=int(ta["depth"]), heads=int(ta["heads"]),
                    scale=0.02 * EXT, h=H, ext=EXT, seed=int(ta["seed"])).to(dev)
    net.load_state_dict(st["net"]); net.eval()
    x = take(d["x"][a.t0], GS)
    v = (x - take(d["x"][max(a.t0 - 1, 0)], GS)) / FRAME_DT
    p = take(d["x"][a.t0], AIDX); XC = take(d["x"][0], GS)
    Jacc = torch.eye(3, device=dev).expand(x.shape[0], 3, 3).contiguous()
    model = []
    for i in range(a.frames):
        idx, _ = anchor_knn(x, p, K)
        feat, _ = aggregate(x, v / VEL, XC, MASS[GS.to(dev)], idx, p.shape[0], H, pa=p)
        ex = torch.cat([MAT.reshape(1, -1).expand(p.shape[0], -1),
                        bc_features(p, cfg) / H], -1)
        dp, lr_, lt_ = net(p, torch.cat([feat, ex], -1), FRAME_DT)
        x2, _, J = skin_with_jacobian(x, p, dp, lr_, lt_, idx, H)
        Jacc = J @ Jacc
        v, p, x = (x2 - x) / FRAME_DT, p + dp, x2
        model.append((x.clone(), measure(Jacc).clone()))

# GT: MPM 의 f_tensor 를 t0 기준으로 다시 잡는다 (t0 에서 시작하는 롤아웃과 맞추려고)
F0 = take(d["F"][a.t0], GS)
F0i = torch.linalg.inv(F0 + 1e-4 * torch.eye(3, device=dev))
gt = []
for i in range(a.frames):
    t = a.t0 + i + 1
    xg = take(d["x"][t], GS)
    gt.append((xg, measure(take(d["F"][t], GS) @ F0i)))

vmax = a.vmax
if vmax <= 0:
    allv = torch.cat([s for _x, s in gt] + ([s for _x, s in model] if model else []))
    vmax = float(allv.quantile(0.99))
print(f"[색 범위] {1.0 if a.metric=='sigma1' else 0.0} ~ {vmax:.3f} "
      f"(전체 프레임 p99 로 고정), 지표 {a.metric}", flush=True)

# 롤아웃 패널 준비: 단면과 **같은 프레임**을 왼쪽에 둔다
RCAM = ptrender.camera(12.0, 35.0, dev)
_all = torch.stack([g[0] for g in gt])
RCTR, RHALF, RW, RH = ptrender.frame_box(_all, RCAM, 220)
RCOL = ptrender.canon_color(xc0)

os.makedirs(a.out, exist_ok=True)
rows = 1 if model is None else 2
frames = []
for i in range(a.frames):
    fig, ax = plt.subplots(rows, a.slices + 1,
                           figsize=((a.slices + 1) * a.tile / 100,
                                    rows * a.tile / 100),
                           squeeze=False)
    for r in range(rows):
        xx, ss = (gt[i] if r == 0 else model[i])
        rel = xx - xx.mean(0)
        A0 = ax[r][0]; A0.set_xticks([]); A0.set_yticks([])
        A0.imshow(ptrender.splat(xx, RCOL, RCAM, RCTR, RHALF, RW, RH, 1)
                  .cpu().numpy())
        A0.set_ylabel("GT" if r == 0 else "model", fontsize=9)
        if r == 0:
            A0.set_title("rollout", fontsize=8)
        for s, z in enumerate(ZS):
            A = ax[r][s + 1]; A.set_xticks([]); A.set_yticks([])
            g = to_grid(ss, rel, z)
            if g is None:
                A.text(.5, .5, "-", ha="center"); continue
            im = A.imshow(g.T, origin="lower", cmap="magma",
                          vmin=(1.0 if a.metric == "sigma1" else 0.0), vmax=vmax)
            if r == 0:
                A.set_title(f"rel z={z:+.2f}", fontsize=8)

    lbl = ("largest singular value of F" if a.metric == "sigma1"
           else "|Green-Lagrange strain|_F")
    fig.suptitle(f"{a.traj}  frame {a.t0+i+1:03d}  {lbl} (since frame {a.t0})",
                 fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    w_, h_ = fig.canvas.get_width_height()
    # matplotlib 3.10 에서 tostring_rgb 가 사라졌다. argb 로 받아 채널을 돌린다.
    frames.append(np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                  .reshape(h_, w_, 4)[..., 1:].copy())
    plt.close(fig)
    if i % 10 == 0:
        print(f"  {i}/{a.frames}", flush=True)

p_out = os.path.join(a.out, f"{a.metric}_{a.traj}_t{a.t0}.mp4")
imageio.mimsave(p_out, frames, fps=a.fps, quality=8)
print(f"[저장] {p_out}  {len(frames)} 프레임", flush=True)
print("STRETCH_OK")
