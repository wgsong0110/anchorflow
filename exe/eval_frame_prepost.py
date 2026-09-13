"""기하 피팅 전후의 성능을, 형상 매칭과 프레임 상태에 대해 각각 잰다.

평가 집합은 **이 자리에서 새로 만든다** -- 캐시의 held-out 10 개는 F 를 저장하지
않아(keep_fc 없이 만들어졌다) F 항을 못 재기 때문이다. 학습에 쓴 것과 같은 분포
(K 와 r 이 각각 로그균등, mp_kmax/mp_rmin 동일)에서 다른 씨앗으로 뽑는다.

궤적은 **학습 전 기하**로 한 번만 만들어 네 설정이 모두 같은 것을 본다. 임펄스를
입자 속도로 퍼뜨리는 데 기하가 쓰이므로, 설정마다 새로 만들면 비교가 섞인다.
"""
from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--fit_base", default=None, help="형상 매칭으로 학습한 기하")
ap.add_argument("--fit_frame", default=None, help="프레임 상태로 학습한 기하")
ap.add_argument("--n_eval", type=int, default=10)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--n_win", type=int, default=4, help="궤적당 평가 프레임 수")
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--mp_kmax", type=int, default=32)
ap.add_argument("--mp_rmin", type=float, default=0.125)
ap.add_argument("--impulse_range", type=float, default=16.0)
ap.add_argument("--seed", type=int, default=99991)
ap.add_argument("--gn", type=int, default=6); ap.add_argument("--cg", type=int, default=20)
ap.add_argument("--anchors", type=int, default=512)
ap.add_argument("--traj_out", default=None)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import AnchorSparse, Traj, load_fitted
from anchorflow.mpm_teacher import MPMTeacher
from anchorflow.frame_encode import FrameState
sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, args.anchors, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
MASS = sc.volume[sc.keep].clone()
DT_C = args.dt_mult * sc.sub_dt
fit0 = AnchorSparse(sc, c=0.25, eig_floor=0.02).to(dev)
fit0.init_from_geometry()
cache0 = fit0.prepare()
R_REF = float((fit0.Xc - cache0[1]).norm(dim=-1).pow(2).mean().sqrt())
C_X, C_V, C_F = 1.0 / args.grid_lim, DT_C / args.grid_lim, R_REF / args.grid_lim
print(f"[setup] c_x {C_X:.5g}, c_v {C_V:.5g}, c_F {C_F:.5g}", flush=True)

base = None
for bc in sc.cfg.get("boundary_conditions", []):
    if "force" in bc:
        base = torch.tensor(bc["force"], device=dev); break
if base is None:
    base = torch.tensor([0.0, 0.0, 1.0], device=dev)

# ---- 평가 궤적 (학습에 안 쓴 씨앗, F 저장) --------------------------------
T = MPMTeacher(sc)


@torch.no_grad()
def make_traj(force):
    dv = fit0.impulse_dv(force, cache0)
    v0 = torch.zeros(fit0.N, 3, device=dev).index_add_(
        0, fit0.pair_g, cache0[0].unsqueeze(-1) * dv[fit0.pair_a])
    T._set(T.pos_m.clone(), v0.contiguous(), T.eye.clone(), torch.zeros_like(T.eye))
    xs, vs, fs_ = [T.pos_m.half().cpu()], [v0.half().cpu()], [T.eye.half().cpu()]
    for _ in range(args.frames):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None
        xs.append(T.solver.export_particle_x_to_torch().half().cpu())
        vs.append(T.solver.export_particle_v_to_torch().half().cpu())
        fs_.append(T.solver.export_particle_F_to_torch().reshape(-1, 9).half().cpu())
    return Traj(torch.stack(xs)), Traj(torch.stack(vs)), Traj(torch.stack(fs_))


g = torch.Generator(device=dev); g.manual_seed(args.seed)
EV = []
from tqdm import tqdm
pbar = tqdm(total=args.n_eval, desc="평가 궤적 MPM", ncols=88)
tries = 0
while len(EV) < args.n_eval and tries < args.n_eval * 5:
    tries += 1
    uk = torch.rand(1, device=dev, generator=g).item()
    kk = max(1, int(round(args.mp_kmax ** uk)))
    ur = torch.rand(1, device=dev, generator=g).item()
    lo, hi = sc.sim.radius * args.mp_rmin, sc.extent
    rad = lo * ((hi / lo) ** ur)
    us = torch.rand(1, device=dev, generator=g).item()
    mag = base.norm().item() * 0.5 * (args.impulse_range ** us)
    t = make_traj(sc.random_multi_poke(g, kk, rad, mag))
    if t is not None:
        EV.append(t); pbar.update(1)
pbar.close()
print(f"[data] 평가 궤적 {len(EV)} (시도 {tries})", flush=True)
if args.traj_out:
    torch.save({"ev": EV}, args.traj_out)

WIN = []
gw = torch.Generator(device="cpu"); gw.manual_seed(20260907)
for i in range(len(EV)):
    for _ in range(args.n_win):
        WIN.append((i, int(torch.randint(max(1, args.frames - 6), (1,), generator=gw).item())))
print(f"[data] 평가 창 {len(WIN)}", flush=True)


def mw(e2): return float(((MASS * e2).sum() / MASS.sum()).sqrt())


def evaluate(f, frame):
    cache = f.prepare()
    fac = f.ls_factor(cache)
    Yg = f.Xc - cache[1]
    FS = FrameState(f.pair_g, f.pair_a, f.N, f.M, MASS) if frame else None
    acc = [0.0, 0.0, 0.0]
    for i, t in WIN:
        x0 = EV[i][0][t].to(dev, torch.float32)
        v0 = EV[i][1][t].to(dev, torch.float32)
        F0m = EV[i][2][t].to(dev, torch.float32).view(-1, 3, 3)
        v = f.project_v_ls(v0, cache, fac)
        if frame:
            pj, uj, sj = FS.encode_closed(x0, F0m, cache[0], Yg,
                                           fixed=f.fixed, p_fix=f.pos)
            xh, Fh = FS.decode_x(pj, uj, sj, cache[0], Yg)
            vl = f.lift(pj, v, cache)[1]
        else:
            p = f.project_ls(x0, cache, fac)
            xh, vl, Fl, _ = f.lift(p, v, cache)
            Fh = Fl.view(-1, 3, 3)
        acc[0] += mw((xh - x0).pow(2).sum(-1))
        acc[1] += mw((vl - v0).pow(2).sum(-1))
        acc[2] += mw((Fh - F0m).pow(2).sum((-1, -2)))
    return [a / len(WIN) for a in acc], f.M


ROWS = []
def run(tag, path, frame):
    if path is None:
        f = fit0
    elif not os.path.exists(path):
        print(f"  [건너뜀] {tag}: {path} 없음", flush=True); return
    else:
        f = load_fitted(sc, path, dev)[0].fit
    r, M = evaluate(f, frame)
    L = C_X * r[0] + C_V * r[1] + C_F * r[2]
    ROWS.append((tag, M, r[0], r[1], r[2], L))
    print(f"  {tag}: L={L:.5e}  x={r[0]:.4e} v={r[1]:.4e} F={r[2]:.4f}  앵커 {M}", flush=True)


run("학습 전 · 형상 매칭", None, False)
run("학습 후 · 형상 매칭", args.fit_base, False)
run("학습 전 · 프레임 상태", None, True)
run("학습 후 · 프레임 상태", args.fit_frame, True)

print(f"\n{'설정':<24}{'앵커':>6}{'mwRMS x':>12}{'mwRMS v':>12}{'mwRMS F':>10}"
      f"{'왕복 손실 L':>14}{'학습 전 대비':>12}")
pre = {False: None, True: None}
for k, (tag, M, ex, ev_, ef, L) in enumerate(ROWS):
    fam = "프레임" in tag
    if "학습 전" in tag: pre[fam] = L
    rel = f"{L/pre[fam]:.3f}" if pre[fam] else "-"
    print(f"{tag:<24}{M:>6}{ex:12.4e}{ev_:12.4e}{ef:10.4f}{L:14.5e}{rel:>12}")
print("\nPREPOST_DONE")
