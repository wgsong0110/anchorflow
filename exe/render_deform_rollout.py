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
ap.add_argument("--no_mat", action="store_true")
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
MAT = torch.zeros(0, device=dev) if a.no_mat else torch.cat([torch.tensor(
    [np.log(float(cfg["E"])), float(cfg["nu"]), float(cfg.get("xi", 0.0)),
     np.log(float(cfg["density"])),
     np.log1p(float(cfg.get("yield_stress", 0.0))),
     float(cfg.get("friction_angle", 0.0)) / 45.0]
    + [1.0 if cfg.get("material", "jelly") == _m else 0.0
       for _m in ("jelly", "metal", "foam", "sand")],
    device=dev, dtype=torch.float32),
    torch.tensor(cfg["g"], device=dev, dtype=torch.float32) / 15.0])


# --------------------------------------------------- 제어점 (학습과 같은 규칙)
# 궤적은 half 로 저장돼 있다 -- 학습과 같이 float32 로 올린다
for _k in ("x", "v", "F"):
    if _k in d and torch.is_tensor(d[_k]) and d[_k].dtype == torch.float16:
        d[_k] = d[_k].float()

CTRL = None
if "ctrl_pos" in d and "ctrl_mem" not in d:
    # 손잡이 궤적: 소속 목록 없이 위치만 있다 (학습도 ctrl_pos 만 쓴다)
    CP = d["ctrl_pos"].to(dev)
    FREE = None
    print(f"[손잡이] {CP.shape[1]} 개, 위치만 조건으로 쓴다", flush=True)
elif "ctrl_mem" in d and "ctrl_pos" in d:
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


def ctrl_feat(t, pa, n_ctrl):
    """앵커마다 (제어점 상대위치, 이번 스텝 강제 변위). 학습과 같은 식이어야 한다."""
    A = pa.shape[0]
    if CP is None:
        return torch.zeros(A, 7 * n_ctrl, device=pa.device, dtype=pa.dtype)
    tt = min(t, CP.shape[0] - 1)
    c = CP[tt]
    dc = CP[min(tt + 1, CP.shape[0] - 1)] - c
    k = c.shape[0]
    if k < n_ctrl:
        z = torch.zeros(n_ctrl - k, 3, device=pa.device, dtype=pa.dtype)
        c, dc = torch.cat([c, z]), torch.cat([dc, z])
    c, dc = c[:n_ctrl], dc[:n_ctrl]
    rel = (pa.unsqueeze(1) - c.unsqueeze(0)) / H_GLOBAL
    frc = dc.unsqueeze(0).expand(A, n_ctrl, 3) / H_GLOBAL
    R = d.get("ctrl_R")
    if R is not None:
        R = R.to(pa.device, pa.dtype)
        Rt = R[min(t, R.numel() - 1)].clamp(min=1e-6)
        q = ((pa.unsqueeze(1) - c.unsqueeze(0)).norm(dim=-1) / Rt).clamp(0, 1)
        w = (1.0 - q * q) ** 2
    else:
        w = torch.zeros(A, n_ctrl, device=pa.device, dtype=pa.dtype)
    return torch.cat([rel.reshape(A, -1), frc.reshape(A, -1),
                      w.reshape(A, -1)], -1)


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
    globals()["H_GLOBAL"] = H
    use_ctrl = bool(ta.get("control", False))
    n_ctrl = int(ta.get("n_ctrl", 4))
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
        if use_ctrl:
            extra = torch.cat([extra, ctrl_feat(a.t0 + len(out) - 1, p, n_ctrl)], -1)
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
        if use_ctrl:
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
# 제어점이 잡은 입자를 빨갛게 -- 어디를 조작하는지 안 보이면 영상만으로는 못 읽는다
CTRL_COL = None
if FREE is not None:
    CTRL_COL = ~FREE
preds = [rollout(c) for c in a.ckpt]
cols = ["GT", "정지"] + [n for _, n in preds]
seqs = [GT, STILL] + [p for p, _ in preds]

R = ptrender.camera(a.elev, a.azim, dev)
ctr, half, W, H = ptrender.frame_box(GT, R, a.width)
col = ptrender.canon_color(take(d["x"][0], GS))
if CTRL_COL is not None:
    col = col.clone()
    col[CTRL_COL] = torch.tensor([0.95, 0.2, 0.15], device=col.device,
                                 dtype=col.dtype)
    print(f"[표시] 제어점이 잡은 입자 {int(CTRL_COL.sum())} 개를 빨갛게",
          flush=True)

# 손잡이 표시: 중심을 색 구슬로, 반경을 옅은 구면으로 그린다
HB = None
if CP is not None:
    _pal = torch.tensor([[0.90, 0.10, 0.10], [0.10, 0.65, 0.20],
                         [0.15, 0.35, 0.95], [0.85, 0.55, 0.05]], device=dev)
    _k = torch.arange(160, device=dev, dtype=torch.float32)
    _z = 1.0 - 2.0 * (_k + 0.5) / 160
    _rr = (1.0 - _z * _z).clamp(min=0).sqrt()
    _ph = _k * 2.399963229728653
    SPH = torch.stack([_rr * torch.cos(_ph), _rr * torch.sin(_ph), _z], -1)
    BALL = SPH * 0.012
    RAD = d.get("ctrl_R")
    RAD = RAD.to(dev) if RAD is not None else None
    HB = (_pal, BALL, SPH, RAD)

os.makedirs(a.out, exist_ok=True)
frames = []
for t in range(GT.shape[0]):
    tiles = []
    for name, sq in zip(cols, seqs):
        pts, pcol = sq[t], col
        if HB is not None:
            _pal, BALL, SPH, RAD = HB
            tt = min(a.t0 + t, CP.shape[0] - 1)
            hp = CP[tt].to(dev)
            nb = min(hp.shape[0], _pal.shape[0])
            ex, ec = [], []
            rr = float(RAD[min(tt, RAD.numel() - 1)]) if RAD is not None else 0.0
            for kk in range(nb):
                ex.append(hp[kk] + BALL)
                ec.append(_pal[kk].expand(BALL.shape[0], 3))
                if rr > 0:
                    ex.append(hp[kk] + SPH * rr)
                    ec.append(_pal[kk].expand(SPH.shape[0], 3) * 0.35 + 0.65)
            pts = torch.cat([sq[t]] + ex, 0)
            pcol = torch.cat([col] + ec, 0)
        img = ptrender.splat(pts, pcol, R, ctr, half, W, H, a.point)
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
