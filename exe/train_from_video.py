"""다중뷰 영상만으로 앵커 기하와 학생 스테퍼를 학습한다 (소성 씬 포함).

3D 궤적을 직접 보지 않는다. 같은 동역학을 여러 카메라에서 본 영상이 전부이고,
지도 신호는 가우시안 중심을 각 시점으로 투영한 **2D 궤적 + 가시성 마스크**다.

초기 속도는 **시퀀스당 하나**다 -- 시점이 여럿이어도 촬영된 동역학은 하나이므로
영상마다 따로 두면 안 된다. 관측 불가능한 양이라 학습 변수로 두고 롤아웃 전체를
역전파해 맞춘다. 앵커 초기 위치는 정지 3DGS 에서 그대로 온다.

오차는 이미지 폭이라는 고정 상수로 나눈다. 궤적 자신의 변위로 나누지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True, help="소성 씬 config (metal/sand 등)")
ap.add_argument("--cameras", required=True, help="데이터셋 cameras.json")
ap.add_argument("--dreamphysics", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--n_views", type=int, default=4)
ap.add_argument("--n_track", type=int, default=20000)
ap.add_argument("--frames", type=int, default=40)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--iters", type=int, default=3000)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--lr_geom", type=float, default=1e-4)
ap.add_argument("--lr_v0", type=float, default=1e-2)
ap.add_argument("--window", type=int, default=10, help="한 스텝에 역전파할 프레임 수")
ap.add_argument("--fit", default=None, help="있으면 이 기하에서 출발한다")
ap.add_argument("--freeze_geom", action="store_true")
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--ckpt_every", type=int, default=200)
ap.add_argument("--resume", default=None)
a = ap.parse_args()

dev = "cuda"
os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, a.dreamphysics)
import warp as wp

wp.init()
from anchorflow import scene_setup
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.nextstate import NextStep, apply_step

LOG = open(os.path.join(a.out, "train.log"), "a")


def log(*x):
    m = " ".join(str(t) for t in x)
    print(m, flush=True)
    LOG.write(m + "\n"); LOG.flush()


sc = scene_setup.build(a.ply, a.config, a.n_anchors, a.K, device=dev,
                       frozen_weights=True, rot_fallback=True,
                       eig_floor=a.eig_floor)
EXT = float(sc.extent)
mat = sc.cfg.get("material", "jelly")
log(f"[씬] 재질 {mat}, 가우시안 {int(sc.keep.sum())}, 앵커 {sc.M}, 물체 {EXT:.4f}")

# ---------------- 정답: MPM 을 한 번 굴려 다시점 2D 궤적으로 ----------------
FRAME_DT = float(sc.cfg.get("frame_dt", 0.04))
T = MPMTeacher(sc, horizon=a.frames * FRAME_DT)
n_sub = max(1, int(round(FRAME_DT / float(sc.sub_dt))))
BASE = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        BASE = torch.tensor(bc["force"], device=dev)
v0_particles = torch.zeros(T.n, 3, device=dev)
if BASE is not None:
    f = BASE.unsqueeze(0).expand(sc.pos.shape[0], 3).contiguous()
    dv = sc.impulse_dv(f)
    v0_particles = (T.w.unsqueeze(-1) * dv[T.idx]).sum(1).contiguous()
T._set(T.pos_m.clone(), v0_particles, T.eye.clone(), torch.zeros_like(T.eye))

rng = np.random.RandomState(0)
tid = torch.from_numpy(np.sort(rng.choice(T.n, min(a.n_track, T.n),
                                          replace=False))).long().to(dev)
xs = [T.pos_m[tid].clone()]
t0 = time.time()
# 격자 이탈을 미리 잡는다. 그냥 굴리면 warp 가 불법 메모리 접근으로 죽고
# 그 시점 이후의 상태는 쓸 수 없다(실측: plane 에서 CUDA error 700).
for fr in range(a.frames - 1):
    bad = False
    for k in range(n_sub):
        T.solver.p2g2p(None, float(sc.sub_dt), device=T.wp_dev)
        if (k + 1) % 4 == 0 and (not T._in_domain() or not T._vel_safe(4)):
            bad = True
            break
    if bad:
        log(f"[mpm] 프레임 {fr+1} 에서 격자 이탈 -- 여기까지만 쓴다")
        break
    xs.append(T.solver.export_particle_x_to_torch()[tid].clone())
TRUTH = torch.stack(xs)                                   # [T, n_track, 3]
a.frames = TRUTH.shape[0]
d_max = float((TRUTH - TRUTH[0]).norm(dim=-1).max())
log(f"[mpm] {a.frames} 프레임 {time.time()-t0:.1f}s, 최대 변위 {d_max:.4f} "
    f"(물체의 {100*d_max/EXT:.2f}%)")

# ---------------- 카메라: 물체가 크게 담기고 서로 떨어진 시점 ----------------
cams = json.load(open(a.cameras))
xw = sc.undo(sc.pos[sc.keep]).double().cpu().numpy()
xsub = xw[rng.choice(xw.shape[0], min(4000, xw.shape[0]), replace=False)]
score = []
for i, c in enumerate(cams):
    w0, h0 = int(c["width"]), int(c["height"])
    R = np.array(c["rotation"]); t_ = -R.T @ np.array(c["position"])
    cx = xsub @ R + t_
    z = cx[:, 2]; fr = z > 1e-6
    if fr.sum() < 16:
        continue
    u = c["fx"] * cx[fr, 0] / z[fr] + w0 / 2
    v = c["fy"] * cx[fr, 1] / z[fr] + h0 / 2
    ins = (u >= 0) & (u < w0) & (v >= 0) & (v < h0)
    if ins.sum() < 16:
        continue
    score.append((float(ins.mean()) * float(fr.mean())
                  * min((np.ptp(u[ins]) / w0) * (np.ptp(v[ins]) / h0), 1.0), i))
score.sort(reverse=True)
cand = [i for _, i in score[: max(6 * a.n_views, a.n_views)]]
pos = np.array([cams[i]["position"] for i in cand])
sel = [0]
while len(sel) < min(a.n_views, len(cand)):
    d = np.linalg.norm(pos[:, None] - pos[None, sel], axis=-1).min(1)
    d[sel] = -1
    sel.append(int(d.argmax()))
VIEWS = [cand[i] for i in sel]
CAM = []
for i in VIEWS:
    c = cams[i]
    R = np.array(c["rotation"])
    CAM.append({"R": torch.tensor(R, device=dev, dtype=torch.float32),
                "t": torch.tensor(-R.T @ np.array(c["position"]), device=dev,
                                  dtype=torch.float32),
                "fx": float(c["fx"]), "fy": float(c["fy"]),
                "w": int(c["width"]), "h": int(c["height"])})
WPIX = float(np.mean([c["w"] for c in CAM]))
log(f"[카메라] {VIEWS} (폭 평균 {WPIX:.0f})")


def project(x_mpm, cam):
    xw_ = sc.undo(x_mpm)
    cx = xw_ @ cam["R"] + cam["t"]
    z = cx[:, 2].clamp(min=1e-6)
    return torch.stack([cam["fx"] * cx[:, 0] / z + cam["w"] / 2,
                        cam["fy"] * cx[:, 1] / z + cam["h"] / 2], -1)


with torch.no_grad():
    UV, VIS = [], []
    for cam in CAM:
        uv = torch.stack([project(TRUTH[t], cam) for t in range(a.frames)])
        u, v = uv[..., 0], uv[..., 1]
        VIS.append((u >= 0) & (u < cam["w"]) & (v >= 0) & (v < cam["h"]))
        UV.append(uv)
    UV = torch.stack(UV)                                  # [V, T, N, 2]
    VIS = torch.stack(VIS)
log(f"[정답] 2D 궤적 {tuple(UV.shape)}, 가시 비율 {float(VIS.float().mean()):.3f}")

# ---------------- 학습 대상 ----------------
# use_accel=False: 탄성 가속도는 앵커 시뮬레이터에서 오는 입력인데, 영상만으로
# 학습할 때는 그 시뮬레이터를 쓰지 않으므로 상태에서 뺀다 (--no_accel 과 같다).
net = NextStep(hidden=a.hidden, depth=a.depth, heads=a.heads,
               use_accel=False, scale=EXT, vel_scale=EXT / max(FRAME_DT, 1e-6),
               zero_init=True).to(dev)
GP0 = sc.pos.clone()
AC = sc.anchor_canonical.clone()
v0 = torch.nn.Parameter(torch.zeros(sc.M, 3, device=dev))   # 시퀀스당 하나
geom_params = []
if not a.freeze_geom:
    # 앵커 기하도 영상 손실로 민다. 정지 위치는 3DGS 에서 왔으므로 여기서는
    # 앵커 자체의 배치(위치)만 미세 조정한다.
    AC = torch.nn.Parameter(AC)
    geom_params = [AC]
groups = [{"params": list(net.parameters()), "lr": a.lr},
          {"params": [v0], "lr": a.lr_v0}]
if geom_params:
    groups.append({"params": geom_params, "lr": a.lr_geom})
opt = torch.optim.Adam(groups)
dt = FRAME_DT
FIX = sc.fixed_mask
log(f"[학습] 파라미터: 학생 {sum(p.numel() for p in net.parameters())}, "
    f"초기속도 {v0.numel()} (시퀀스당 1 개), 기하 "
    f"{sum(p.numel() for p in geom_params)}")

it0 = 0
CK = os.path.join(a.out, "ckpt.pt")
if a.resume or os.path.exists(CK):
    p_ = a.resume or CK
    st = torch.load(p_, map_location=dev, weights_only=False)
    net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"])
    with torch.no_grad():
        v0.copy_(st["v0"])
        if geom_params and "ac" in st:
            AC.copy_(st["ac"])
    it0 = int(st["iter"])
    log(f"[resume] {p_} iter {it0}")


def sub(p, v, gp):
    """프레임 하나: 학생 한 스텝 + 스키닝."""
    out = apply_step(net, p, v, None, dt, FIX)
    q, d = out[-1]
    return q, d / dt, sc.skin(q, gp, grad=True)


def rollout(n_frames, start=0, detach_geom=False):
    p = (AC if not detach_geom else AC.detach()).clone()
    v = v0.clone()
    gp = GP0.clone()
    outs = []
    for f in range(1, n_frames):
        p, v, gp = checkpoint(sub, p, v, gp, use_reentrant=False)
        outs.append(gp[sc.keep][tid])
    return outs


def loss_2d(outs, f_start=1):
    tot, cnt = 0.0, 0.0
    for i, x in enumerate(outs):
        t = f_start + i
        for ci, cam in enumerate(CAM):
            m = VIS[ci, t]
            if m.sum() == 0:
                continue
            e = ((project(x, cam) - UV[ci, t]) ** 2).sum(-1)
            tot = tot + (e * m).sum()
            cnt = cnt + m.sum()
    return tot / cnt.clamp(min=1)


hist = []
t_start = time.time()
for it in range(it0 + 1, a.iters + 1):
    opt.zero_grad(set_to_none=True)
    n = min(a.window, a.frames)
    outs = rollout(n)
    mse = loss_2d(outs)
    (mse / (WPIX ** 2)).backward()
    torch.nn.utils.clip_grad_norm_(
        [p for g in opt.param_groups for p in g["params"]], 1.0)
    opt.step()
    px = float(mse.detach().sqrt())
    hist.append({"iter": it, "px": px, "rel": px / WPIX})
    if it % 20 == 0 or it == 1:
        log(f"[{it:5d}/{a.iters}] 2D {px:8.2f} px ({100*px/WPIX:5.2f}% of 폭)  "
            f"|v0| {float(v0.norm()):.4f}  {(time.time()-t_start)/60:.1f}분")
    if it % a.ckpt_every == 0 or it == a.iters:
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                    "v0": v0.detach(), "ac": AC.detach(), "iter": it,
                    "hist": hist, "views": VIEWS}, CK)
        json.dump(hist, open(os.path.join(a.out, "hist.json"), "w"), indent=1)

# ---------------- 전체 롤아웃 평가 ----------------
with torch.no_grad():
    outs = rollout(a.frames, detach_geom=True)
    full = float(loss_2d(outs).sqrt())
    xs_pred = torch.stack(outs)
    e3 = float((xs_pred - TRUTH[1:]).norm(dim=-1).mean()) / EXT
log(f"\n[전체 {a.frames} 프레임] 2D {full:.2f} px ({100*full/WPIX:.2f}% of 폭), "
    f"3D 참고 {100*e3:.2f}% of 물체")
torch.save({"pred": xs_pred.cpu(), "truth": TRUTH.cpu(), "tid": tid.cpu(),
            "views": VIEWS}, os.path.join(a.out, "rollout.pt"))
log("TRAIN_FROM_VIDEO_OK")
