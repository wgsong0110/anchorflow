"""수박의 xy 단면마다 한 스텝 변형 벡터장을 그린다.

무엇을 보는가. 앵커가 그 평면 위의 점들을 **한 프레임 동안 어디로 옮기는지**다:

    u(p) = (phi_t(p) - c_{t+1}) - (p - c_t)

c 는 물체의 질량중심이고, 이것을 빼야 "수박이 통째로 떨어지는" 강체 성분이 사라져
변형만 남는다. 단면도 절대 z 가 아니라 **현재 중심 기준 상대 z** 로 자른다 --
물체가 내려가도 같은 높이대를 계속 보기 위해서다.

벡터를 가우시안마다 그리지 않고 평면 위에 깐 **정규 격자점**에서 평가한다. 변형
사상은 공간의 함수이므로, 입자가 성긴 곳에서도 장이 어떻게 생겼는지 보이려면
격자에서 재는 것이 맞다.

GT 는 입자에서 만든 변위를 격자로 보간해 같은 자리에서 비교한다.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", default=None, help="없으면 GT 변형장만 그린다")
ap.add_argument("--traj", default="watermelon_h")
ap.add_argument("--out", required=True)
ap.add_argument("--t0", type=int, default=3)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--n_pts", type=int, default=20000)
ap.add_argument("--slices", type=int, default=5, help="상대 z 단면 개수")
ap.add_argument("--grid", type=int, default=24, help="단면당 격자 해상도")
ap.add_argument("--band", type=float, default=0.06,
                help="단면 두께 (물체 크기 대비). GT 보간에 쓸 입자를 고른다")
ap.add_argument("--scale", type=float, default=12.0, help="화살표 길이 배율")
ap.add_argument("--mask", type=float, default=1.2,
                help="격자점에서 가장 가까운 입자까지 이 배를 넘으면 재질이 없는 "
                     "자리로 보고 화살표를 그리지 않는다")
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


# 단면은 **정준 배치의 상대 z** 구간으로 정하고, 격자의 xy 범위도 거기서 잡는다.
# (구간 자체는 고정이되, 매 프레임 현재 중심을 더해 평면을 놓는다.)
xc0 = take(d["x"][0], GS)
c0 = xc0.mean(0)
rel0 = xc0 - c0
zlo, zhi = float(rel0[:, 2].min()), float(rel0[:, 2].max())
edges = np.linspace(zlo, zhi, a.slices + 1)
ZS = [(edges[i] + edges[i + 1]) / 2 for i in range(a.slices)]
xy_lo = rel0[:, :2].min(0).values.cpu().numpy()
xy_hi = rel0[:, :2].max(0).values.cpu().numpy()
gx = np.linspace(xy_lo[0], xy_hi[0], a.grid)
gy = np.linspace(xy_lo[1], xy_hi[1], a.grid)
GX, GY = np.meshgrid(gx, gy, indexing="ij")
_s = rel0[torch.randperm(rel0.shape[0], device=dev)[:2000]]
CELL = float(torch.cdist(_s, _s).topk(2, largest=False).values[:, 1].median())
print(f"[단면] 상대 z {zlo:.3f}~{zhi:.3f} 를 {a.slices} 등분, "
      f"격자 {a.grid}x{a.grid}, 물체 {EXT:.4f}, 가림 단위 {CELL:.5f}", flush=True)


def occupancy(rel, z):
    """그 단면에서 재질이 있는 격자점만 참. 밖까지 화살표를 그리면 수박이 없는
    배경이 변형된 것처럼 보인다."""
    sel = (rel[:, 2] - z).abs() < a.band * EXT
    if int(sel.sum()) < 8:
        return None
    q = torch.from_numpy(np.stack([GX, GY], -1).reshape(-1, 2)).float().to(dev)
    dd = torch.cdist(q, rel[sel][:, :2]).min(1).values
    return (dd < a.mask * CELL).reshape(a.grid, a.grid).cpu().numpy()


def grid_points(z, c):
    """상대 z 인 평면 위 격자점을 **현재 중심 기준 절대 좌표**로."""
    p = np.stack([GX, GY, np.full_like(GX, z)], -1).reshape(-1, 3)
    return torch.from_numpy(p).float().to(dev) + c


def gt_field(t, c_t, c_n, z):
    """GT 변위를 격자로 옮긴다. 단면 두께 안의 입자만 쓰고, 역거리 가중으로 보간."""
    xt = take(d["x"][t], GS); xn = take(d["x"][t + 1], GS)
    u = (xn - c_n) - (xt - c_t)                  # 입자별 한 스텝 상대 변위
    rel = xt - c_t
    sel = (rel[:, 2] - z).abs() < a.band * EXT
    if int(sel.sum()) < 8:
        return None
    q = grid_points(z, c_t)
    dd = torch.cdist(q[:, :2], (xt[sel] - c_t)[:, :2])
    w = 1.0 / (dd + 0.02 * EXT) ** 2
    w = w / w.sum(1, keepdim=True)
    return (w @ u[sel]).cpu().numpy()


net = None
if a.ckpt:
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ta = st["args"]; AIDX = st["aidx"].to(dev); H = float(st["H"])
    net = DeformNet(n_feat=int(st["n_feat"]), hidden=int(ta["hidden"]),
                    depth=int(ta["depth"]), heads=int(ta["heads"]),
                    scale=0.02 * EXT, h=H, ext=EXT, seed=int(ta["seed"])).to(dev)
    net.load_state_dict(st["net"]); net.eval()
    K = int(ta["k"])


def model_field(p_anchor, dp, lr_, lt_, H, z, c_t, c_n):
    """앵커가 격자점을 어디로 옮기는지. 스키닝을 **격자점에** 그대로 적용한다."""
    q = grid_points(z, c_t)
    idx, _ = anchor_knn(q, p_anchor, K)
    q2, _, _ = skin_with_jacobian(q, p_anchor, dp, lr_, lt_, idx, H)
    return ((q2 - c_n) - (q - c_t)).cpu().numpy()


# 모델 롤아웃 (있으면)
seq = None
if net is not None:
    x = take(d["x"][a.t0], GS)
    v = (x - take(d["x"][max(a.t0 - 1, 0)], GS)) / FRAME_DT
    p = take(d["x"][a.t0], AIDX)
    XC = take(d["x"][0], GS)
    seq = []
    for i in range(a.frames):
        idx, _ = anchor_knn(x, p, K)
        feat, _ = aggregate(x, v / VEL, XC, MASS[GS.to(dev)], idx, p.shape[0], H, pa=p)
        ex = torch.cat([MAT.reshape(1, -1).expand(p.shape[0], -1),
                        bc_features(p, cfg) / H], -1)
        dp, lr_, lt_ = net(p, torch.cat([feat, ex], -1), FRAME_DT)
        x2, _, _ = skin_with_jacobian(x, p, dp, lr_, lt_, idx, H)
        seq.append((p.clone(), dp.clone(), lr_.clone(), lt_.clone(),
                    x.mean(0).clone(), x2.mean(0).clone(), x.clone()))
        v, p, x = (x2 - x) / FRAME_DT, p + dp, x2

RCAM = ptrender.camera(12.0, 35.0, dev)
_allx = torch.stack([take(d["x"][a.t0 + j], GS) for j in range(a.frames + 1)])
RCTR, RHALF, RW, RH = ptrender.frame_box(_allx, RCAM, 220)
RCOL = ptrender.canon_color(xc0)

os.makedirs(a.out, exist_ok=True)
rows = 1 if net is None else 2
frames = []
for i in range(a.frames):
    t = a.t0 + i
    c_t = take(d["x"][t], GS).mean(0)
    c_n = take(d["x"][t + 1], GS).mean(0)
    fig, ax = plt.subplots(rows, a.slices + 1,
                           figsize=((a.slices + 1) * a.tile / 100,
                                    rows * a.tile / 100),
                           squeeze=False)
    xs_gt = take(d["x"][t], GS)
    xs_md = seq[i][6] if (net is not None and len(seq[i]) > 6) else None
    for r in range(rows):
        A0 = ax[r][0]; A0.set_xticks([]); A0.set_yticks([])
        xx = xs_gt if r == 0 else (xs_md if xs_md is not None else xs_gt)
        A0.imshow(ptrender.splat(xx, RCOL, RCAM, RCTR, RHALF, RW, RH, 1)
                  .cpu().numpy())
        A0.set_ylabel("GT" if r == 0 else "model", fontsize=9)
        if r == 0:
            A0.set_title("rollout", fontsize=8)
    for s, z in enumerate(ZS):
        for r in range(rows):
            u = (gt_field(t, c_t, c_n, z) if r == 0 else
                 model_field(*seq[i][:4], H, z, seq[i][4], seq[i][5]))
            A = ax[r][s + 1]
            A.set_xticks([]); A.set_yticks([])
            if u is None:
                A.text(.5, .5, "-", ha="center"); continue
            U = u.reshape(a.grid, a.grid, 3)
            occ = occupancy((xs_gt - c_t) if r == 0 else (xs_md - seq[i][4]
                            if xs_md is not None else xs_gt - c_t), z)
            if occ is not None:
                U = np.where(occ[..., None], U, np.nan)
                A.set_facecolor("0.96")                # 수박 밖은 회색 바탕
                A.contour(GX, GY, occ.astype(float), levels=[0.5],
                          colors="#d00000", linewidths=1.0)
            mag = np.linalg.norm(U, axis=-1)
            A.quiver(GX, GY, U[..., 0], U[..., 1], mag, cmap="viridis",
                     scale=1.0 / max(a.scale, 1e-6), scale_units="xy",
                     width=0.006)
            if r == 0:
                A.set_title(f"rel z={z:+.2f}", fontsize=8)

    fig.suptitle(f"{a.traj}  frame {t:03d}  one-step deformation "
                 f"(centre-removed, xy slices)", fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    w_, h_ = fig.canvas.get_width_height()
    # matplotlib 3.10 에서 tostring_rgb 가 사라졌다. argb 로 받아 채널을 돌린다.
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h_, w_, 4)[..., 1:]
    frames.append(img.copy())
    plt.close(fig)
    if i % 10 == 0:
        print(f"  {i}/{a.frames}", flush=True)

p_out = os.path.join(a.out, f"field_{a.traj}_t{a.t0}.mp4")
imageio.mimsave(p_out, frames, fps=a.fps, quality=8)
print(f"[저장] {p_out}  {len(frames)} 프레임", flush=True)
print("FIELD_OK")
