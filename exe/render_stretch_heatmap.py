"""초기 배치 대비 누적 변형장의 **최대 특이값** 을 단면 히트맵으로 그린다.

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
ap.add_argument("--vmax", type=float, default=0.0,
                help="색 상한. 0 이면 전체 프레임의 p99 로 한 번 고정한다")
ap.add_argument("--tile", type=int, default=240)
ap.add_argument("--fps", type=int, default=10)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def sigma1(F):
    return torch.linalg.svdvals(F)[..., 0]


def to_grid(vals, pos_rel, z):
    """입자 값을 단면 격자로 옮긴다 (두께 안의 입자만, 역거리 가중)."""
    sel = (pos_rel[:, 2] - z).abs() < a.band * EXT
    if int(sel.sum()) < 8:
        return None
    q = torch.from_numpy(np.stack([GX, GY], -1).reshape(-1, 2)).float().to(dev)
    dd = torch.cdist(q, pos_rel[sel][:, :2])
    w = 1.0 / (dd + 0.02 * EXT) ** 2
    w = w / w.sum(1, keepdim=True)
    return (w @ vals[sel].unsqueeze(-1)).squeeze(-1).reshape(a.grid, a.grid).cpu().numpy()


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
        model.append((x.clone(), sigma1(Jacc).clone()))

# GT: MPM 의 f_tensor 를 t0 기준으로 다시 잡는다 (t0 에서 시작하는 롤아웃과 맞추려고)
F0 = take(d["F"][a.t0], GS)
F0i = torch.linalg.inv(F0 + 1e-4 * torch.eye(3, device=dev))
gt = []
for i in range(a.frames):
    t = a.t0 + i + 1
    xg = take(d["x"][t], GS)
    gt.append((xg, sigma1(take(d["F"][t], GS) @ F0i)))

vmax = a.vmax
if vmax <= 0:
    allv = torch.cat([s for _x, s in gt] + ([s for _x, s in model] if model else []))
    vmax = float(allv.quantile(0.99))
print(f"[색 범위] 1.0 ~ {vmax:.3f} (전체 프레임 p99 로 고정)", flush=True)

os.makedirs(a.out, exist_ok=True)
rows = 1 if model is None else 2
frames = []
for i in range(a.frames):
    fig, ax = plt.subplots(rows, a.slices,
                           figsize=(a.slices * a.tile / 100, rows * a.tile / 100),
                           squeeze=False)
    for r in range(rows):
        xx, ss = (gt[i] if r == 0 else model[i])
        rel = xx - xx.mean(0)
        for s, z in enumerate(ZS):
            A = ax[r][s]; A.set_xticks([]); A.set_yticks([])
            g = to_grid(ss, rel, z)
            if g is None:
                A.text(.5, .5, "-", ha="center"); continue
            im = A.imshow(g.T, origin="lower", cmap="magma", vmin=1.0, vmax=vmax)
            if r == 0:
                A.set_title(f"rel z={z:+.2f}", fontsize=8)
            if s == 0:
                A.set_ylabel("GT" if r == 0 else "model", fontsize=9)
    fig.suptitle(f"{a.traj}  frame {a.t0+i+1:03d}  largest singular value of F "
                 f"(since frame {a.t0})", fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    w_, h_ = fig.canvas.get_width_height()
    frames.append(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                  .reshape(h_, w_, 3).copy())
    plt.close(fig)
    if i % 10 == 0:
        print(f"  {i}/{a.frames}", flush=True)

p_out = os.path.join(a.out, f"stretch_{a.traj}_t{a.t0}.mp4")
imageio.mimsave(p_out, frames, fps=a.fps, quality=8)
print(f"[저장] {p_out}  {len(frames)} 프레임", flush=True)
print("STRETCH_OK")
