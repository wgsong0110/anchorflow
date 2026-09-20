"""학습된 변형 모델의 롤아웃을 GT 와 나란히 그린다.

숫자로는 "홀드아웃 15 프레임 평균 2.0% (정지 7.1%)" 같은 것만 보이는데, 그 오차가
전체가 조금씩 어긋난 것인지 한 조각이 통째로 날아간 것인지는 구별되지 않는다.
그래서 같은 화면에 GT / 예측 / 정지(아무것도 안 함) 셋을 나란히 둔다 -- 정지를
같이 두는 이유는, 이 궤적의 뒷부분이 거의 멈춰 있어서 "가만히 있기"가 이미 꽤
좋은 답이기 때문이다.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="궤적 .pt 들이 있는 디렉토리")
ap.add_argument("--ckpt", required=True, nargs="+", help="체크포인트 .pt (여러 개 가능)")
ap.add_argument("--traj", default="watermelon_h", help="어느 궤적에서 굴릴지")
ap.add_argument("--out", required=True)
ap.add_argument("--t0", type=int, default=3)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--n_pts", type=int, default=20000)
ap.add_argument("--width", type=int, default=420)
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--point", type=int, default=1)
ap.add_argument("--elev", type=float, default=12.0)
ap.add_argument("--azim", type=float, default=35.0)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
import imageio.v2 as imageio                                     # noqa: E402
from PIL import Image, ImageDraw                                 # noqa: E402

from anchorflow import ptrender                                  # noqa: E402
from anchorflow.deform import (DeformNet, aggregate, bc_features,  # noqa: E402
                               gauss_stretch, grid_knn, skin)

d = torch.load(os.path.join(a.data, a.traj + ".pt"), map_location="cpu",
               weights_only=False)
cfg = d["cfg"]
FRAME_DT = float(cfg["frame_dt"])
X0 = d["x"][0]
EXT = float((X0.max(0).values - X0.min(0).values).norm())
N_FULL = X0.shape[0]
GS = torch.arange(min(a.n_pts, N_FULL))
ng = int(cfg.get("n_grid", 100))
dx = float(cfg.get("grid_lim", 2.0)) / ng
X0d = X0.to(dev)
vi = (X0d / dx).long().clamp(0, ng - 1)
flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cnt = torch.zeros(ng ** 3, device=dev).index_add_(
    0, flat, torch.ones(N_FULL, device=dev))
MASS = ((dx ** 3) / cnt[flat]) * float(cfg["density"])
VEL_SCALE = EXT / FRAME_DT
MAT = torch.cat([torch.tensor(
    [np.log(float(cfg["E"])), float(cfg["nu"]), float(cfg.get("xi", 0.0)),
     np.log(float(cfg["density"]))], device=dev, dtype=torch.float32),
    torch.tensor(cfg["g"], device=dev, dtype=torch.float32) / 15.0])


# --------------------------------------------------- 제어점 (학습과 같은 규칙)
CTRL = None
if "ctrl_mem" in d and "ctrl_pos" in d:
    full = [torch.zeros(N_FULL, dtype=torch.bool) for _ in d["ctrl_mem"]]
    for k_, mm in enumerate(d["ctrl_mem"]):
        full[k_][mm] = True
    CTRL = []
    for k_, fl in enumerate(full):
        loc = torch.nonzero(fl[GS]).squeeze(-1).to(dev)
        g = GS[loc.cpu()]
        off = (d["x"][0][g] - d["x"][0][d["ctrl"][k_]]).to(dev)
        CTRL.append((loc, off))
    CP = d["ctrl_pos"].to(dev)
    FREE = torch.ones(GS.numel(), dtype=torch.bool, device=dev)
    for loc, _ in CTRL:
        FREE[loc] = False
    print(f"[제어점] {len(CTRL)} 개, 강제 입자 {int((~FREE).sum())}/{GS.numel()} "
          f"-- 예측에서 덮어쓰고 오차에서 뺀다", flush=True)
else:
    CP = None
    FREE = None


def force_ctrl(x2, t):
    """교사가 박은 입자는 예측 대신 궤적 값으로. 학습 때와 같아야 한다."""
    if CTRL is None:
        return x2
    t1 = min(t, CP.shape[0] - 1)
    x2 = x2.clone()
    for k_, (loc, off) in enumerate(CTRL):
        if k_ < CP.shape[1] and loc.numel():
            x2[loc] = CP[t1, k_] + off
    return x2


def take(t, i):
    """CPU 에 있는 궤적에서 색인해 GPU 로. 색인은 CPU 에서 해야 한다."""
    return t[i.cpu() if torch.is_tensor(i) else i].to(dev)


def rollout(ck):
    """-> [T,N,3] 예측 궤적"""
    st = torch.load(ck, map_location=dev, weights_only=False)
    ta = st["args"]
    AIDX = st["aidx"].to(dev)
    H = float(st["H"])
    dmg_on = bool(ta.get("damage", False))
    net = DeformNet(n_feat=int(st["n_feat"]), hidden=int(ta["hidden"]),
                    depth=int(ta["depth"]), heads=int(ta["heads"]),
                    scale=0.02 * EXT, h=H, ext=EXT, seed=int(ta["seed"]),
                    damage=dmg_on).to(dev)
    net.load_state_dict(st["net"])
    net.eval()
    k = int(ta["k"])
    x = take(d["x"][a.t0], GS)
    v = (x - take(d["x"][max(a.t0 - 1, 0)], GS)) / FRAME_DT
    p = take(d["x"][a.t0], AIDX)
    XC = take(d["x"][0], GS)
    out = [x.clone()]
    x0r, p0r = x.clone(), p.clone()
    dmg = None
    for _ in range(a.frames):
        idx, _ = grid_knn(x, p, k)
        feat, _ = aggregate(x, v / VEL_SCALE, XC, MASS[GS.to(dev)], idx,
                            p.shape[0], H, pa=p)
        extra = torch.cat([MAT.reshape(1, -1).expand(p.shape[0], -1),
                           bc_features(p, cfg) / H], -1)
        o = net(p, torch.cat([feat, extra], -1), FRAME_DT)
        dp, lr_, lt_ = o[0], o[1], o[2]
        if dmg_on:
            if dmg is None:
                dmg = torch.zeros(x.shape[0], device=dev)
            ref = (x0r.unsqueeze(1) - p0r[idx]).norm(dim=-1)
            w0 = skin(x, p, dp, lr_, lt_, idx, H, dmg=dmg)[1]
            dmg = (dmg + FRAME_DT * (w0 * o[3][idx]).sum(1)
                   * gauss_stretch(x, p, idx, ref, w0)).clamp(max=1.0)
        x2, _ = skin(x, p, dp, lr_, lt_, idx, H, dmg=dmg)
        x2 = force_ctrl(x2, a.t0 + len(out))
        v, p, x = (x2 - x) / FRAME_DT, p + dp, x2
        out.append(x.clone())
    return torch.stack(out), os.path.splitext(os.path.basename(ck))[0]


# 궤적보다 더 길게 굴릴 수 있게 한다. GT 가 떨어지면 마지막 프레임을 그대로
# 들고 있고(제목에 표시), 그 뒤 구간의 오차는 참고값이다 -- 모델이 결국 무너지는지
# 아니면 형상을 붙들고 버티는지 보려는 것이다.
NT = d["x"].shape[0]
GT = torch.stack([take(d["x"][min(a.t0 + i, NT - 1)], GS)
                  for i in range(a.frames + 1)])
GT_END = max(0, NT - 1 - a.t0)
if a.frames > GT_END:
    print(f"[주의] GT 는 {GT_END} 프레임까지다. 그 뒤는 마지막 프레임을 고정해 "
          f"비교한다", flush=True)
STILL = GT[:1].expand_as(GT)
preds = [rollout(c) for c in a.ckpt]
cols = ["GT", "정지"] + [n for _, n in preds]
seqs = [GT, STILL] + [p for p, _ in preds]

R = ptrender.camera(a.elev, a.azim, dev)
ctr, half, W, H = ptrender.frame_box(GT, R, a.width)
col = ptrender.canon_color(take(d["x"][0], GS))

os.makedirs(a.out, exist_ok=True)
frames = []
for t in range(GT.shape[0]):
    tiles = []
    for name, sq in zip(cols, seqs):
        img = ptrender.splat(sq[t], col, R, ctr, half, W, H, a.point)
        arr = (img.cpu().numpy() * 255).astype("uint8")
        im = Image.fromarray(arr)
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W, 15], fill=(0, 0, 0))
        e = "" if name in ("GT", "정지") else \
            f"  err {100*float((sq[t]-GT[t])[FREE if FREE is not None else slice(None)].norm(dim=-1).mean())/EXT:.2f}%"
        if name == "정지":
            e = f"  err {100*float((sq[t]-GT[t])[FREE if FREE is not None else slice(None)].norm(dim=-1).mean())/EXT:.2f}%"
        dr.text((4, 2), name + e, fill=(255, 255, 255))
        tiles.append(np.array(im))
    row = np.concatenate(tiles, 1)
    strip = Image.fromarray(row)
    dr = ImageDraw.Draw(strip)
    note = "" if t <= GT_END else "  (GT 끝, 마지막 프레임 고정)"
    dr.text((4, H - 14), f"{a.traj}  t0={a.t0}  frame {t:02d}{note}",
            fill=(0, 0, 0))
    frames.append(np.array(strip))
p_out = os.path.join(a.out, f"rollout_{a.traj}_t{a.t0}.mp4")
imageio.mimsave(p_out, frames, fps=a.fps, quality=8)
print(f"[저장] {p_out}  {len(frames)} 프레임, {row.shape[1]}x{row.shape[0]}",
      flush=True)
_fm = FREE if FREE is not None else slice(None)
for (sq, name) in preds:
    e = float((sq - GT)[:, _fm].norm(dim=-1).mean()) / EXT
    print(f"  {name}: 평균 오차 {100*e:.3f}%", flush=True)
print(f"  정지: 평균 오차 {100*float((STILL-GT)[:, FREE if FREE is not None else slice(None)].norm(dim=-1).mean())/EXT:.3f}%",
      flush=True)
print("ROLLVID_OK")
