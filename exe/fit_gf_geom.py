"""GaussianFluent 의 파괴 궤적 위에서 앵커 기하를 학습하고, 그 궤적을 얼마나 잘
**표현**하는지 잘린 가우시안 가중치와 kNN-softmax 가중치로 나눠 잰다.

왜 이 궤적인가. 앞서 bread 로 하려던 실험은 그 씬의 재질이 jelly 라 소성도 파괴도
없어 무효였다. GaussianFluent 의 watermelon 은 초기 k-NN 이웃의 13.4%(1.5배 기준),
3.4%(3배 기준)가 영구히 떨어져 나간다 -- 고정 연결성이 표현할 수 없는 양이 실제로
있는 궤적이다 (`measure_gf_tearing.py`).

무엇을 재는가. 궤적은 저쪽 CD-MPM 이 만든 것이라 우리 솔버에 되돌려 넣을 수 없다.
그래서 매 프레임 한 번씩 접었다 편 잔차를 본다:

    x_t  ->  project_ls  ->  앵커 상태  ->  gaussian_pos  ->  x̂_t

지표는 대응을 가정하지 않는 둘이다. 찢어지면 어느 점이 어느 점에 대응하는지가
흐려지므로 같은 입자끼리의 오차는 그 상황에서 읽기 어렵다.

    Chamfer      각 점에서 상대 구름의 최근접까지, 양방향 평균
    Earth Mover  최적 일대일 대응의 평균 이동량 (O(n^3) 이라 부분표본)

둘 다 물체 크기로 나눈다. 자기 변위로 나누면 덜 움직인 쪽이 유리해진다.

학습 전과 후를 모두 재고, 두 가중치 방식에 대해 따로 돌린다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import h5py
import numpy as np
import torch
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--h5_dir", required=True, help="GaussianFluent 가 떨군 sim_*.h5")
ap.add_argument("--config", required=True, help="그 씬의 config json")
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="run")
ap.add_argument("--softmax_w", action="store_true",
                help="가중치를 kNN 위 softmax 로. 기본은 잘린 가우시안 G(x)-c")
ap.add_argument("--softmax_k", type=int, default=16)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--n_pts", type=int, default=40000,
                help="입자 부분표본. 전체 138 만은 Chamfer 도 피팅도 못 버틴다")
ap.add_argument("--stride", type=int, default=2, help="프레임 간격")
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--batch", type=int, default=4, help="한 스텝에 쓰는 프레임 수")
ap.add_argument("--lr_pos", type=float, default=3e-4)
ap.add_argument("--lr_scale", type=float, default=1e-2)
ap.add_argument("--lr_quat", type=float, default=1e-2)
ap.add_argument("--refresh_every", type=int, default=20)
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--emd_sample", type=int, default=2048)
ap.add_argument("--cd_chunk", type=int, default=4096)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(a.seed)

from anchorflow.anchor_mpm import AnchorElasticSim, lame_from_E_nu  # noqa: E402
from anchorflow.anchor_sparse import AnchorSparse                  # noqa: E402
from anchorflow.anchors import AnchorSet                           # noqa: E402
from anchorflow.scene_setup import Scene                           # noqa: E402

# ---------------------------------------------------------------- 궤적 읽기


def load_x(p):
    with h5py.File(p, "r") as f:
        d = np.array(f["x"])
    return d.T if d.shape[0] == 3 else d


files = sorted(glob.glob(os.path.join(a.h5_dir, "*.h5")))[::a.stride]
if not files:
    raise SystemExit(f"h5 가 없다: {a.h5_dir}")
cfg = json.load(open(a.config))
print(f"[입력] {len(files)} 프레임 (stride {a.stride}), 재질 {cfg.get('material')}",
      flush=True)

X0f = torch.from_numpy(load_x(files[0])).float()
XLf = torch.from_numpy(load_x(files[-1])).float()
# 이 MPM 은 진행하면서 입자를 비유한값으로 떨군다. 처음과 끝 모두 유한한 것만 쓴다.
ok = torch.isfinite(X0f).all(1) & torch.isfinite(XLf).all(1)
cand = torch.nonzero(ok).squeeze(-1)
g = torch.Generator().manual_seed(a.seed)
sel = cand[torch.randperm(cand.numel(), generator=g)[:a.n_pts]].sort().values
print(f"[입자] 전체 {X0f.shape[0]}, 두 끝 모두 유한 {cand.numel()}, "
      f"표본 {sel.numel()}", flush=True)

TR = []
for p in tqdm(files, desc="궤적", ncols=80):
    x = torch.from_numpy(load_x(p)).float()[sel]
    TR.append(x)
TR = torch.stack(TR).to(dev)
# 남은 비유한값이 있으면 그 프레임의 해당 점을 직전 값으로 채운다 (전체를 버리지 않는다)
bad = ~torch.isfinite(TR).all(-1)
if bad.any():
    print(f"[보정] 중간 프레임의 비유한값 {int(bad.sum())} 개를 직전 값으로 채움",
          flush=True)
    for t in range(1, TR.shape[0]):
        m = bad[t]
        if m.any():
            TR[t][m] = TR[t - 1][m]
X0 = TR[0].contiguous()
EXT = float((X0.max(0).values - X0.min(0).values).norm())
print(f"[씬] 프레임 {TR.shape[0]}, 물체 대각 {EXT:.4f}, 최대 변위 "
      f"{100*float((TR-X0).norm(dim=-1).max())/EXT:.1f}%", flush=True)

# ---------------------------------------------------------------- 씬 구성
# scene_setup.build 는 PLY 와 PhysGaussian 전처리를 전제한다. 여기 입자는 저쪽
# 내부 채움까지 끝난 MPM 입자라 그 경로를 다시 태울 것이 없다 -- 같은 원시 요소
# (AnchorSet, AnchorElasticSim)로 Scene 만 직접 세운다.
N = X0.shape[0]
n_grid = int(cfg.get("n_grid", 100))
dx = float(cfg.get("grid_lim", 2.0)) / n_grid
vi = (X0 / dx).long().clamp(0, n_grid - 1)
flat = (vi[:, 0] * n_grid + vi[:, 1]) * n_grid + vi[:, 2]
cnt = torch.zeros(n_grid ** 3, device=dev).index_add_(
    0, flat, torch.ones(N, device=dev))
volume = ((dx ** 3) / cnt[flat]).contiguous()
E = torch.full((N,), float(cfg["E"]), device=dev)
nu = torch.full((N,), float(cfg["nu"]), device=dev)
dens = torch.full((N,), float(cfg["density"]), device=dev)
mu, lam = lame_from_E_nu(E, nu)
keep = torch.ones(N, dtype=torch.bool, device=dev)

aset, _ = AnchorSet.from_gaussians(X0, node_num=a.n_anchors, latent_dim=0,
                                   e_dim=0, K=a.K)
ac = aset.canonical.clone().contiguous()
radius = AnchorElasticSim(X0, ac, K=a.K).radius
sim = AnchorElasticSim(X0, ac, K=a.K, radius=radius)
sim.eig_floor = a.eig_floor
sim.rot_fallback = True
w0 = sim._weights(X0, ac)
M = ac.shape[0]
mass = torch.zeros(M, device=dev).index_add_(
    0, sim.nn_idx.reshape(-1),
    ((dens * volume).unsqueeze(-1) * w0).reshape(-1)).clamp(min=1e-12)
fixed = torch.zeros(M, dtype=torch.bool, device=dev)
for bc in cfg.get("boundary_conditions", []):
    if bc["type"] == "cuboid":
        c_ = torch.tensor(bc["point"], device=dev)
        s_ = torch.tensor(bc["size"], device=dev)
        fixed |= ((ac - c_).abs() <= s_).all(-1)

sc = Scene(cfg=cfg, xyz_world=X0, pos=X0, keep=keep, volume=volume, mu=mu,
           lam=lam, crop=None, anchor_canonical=ac, mass=mass, fixed_mask=fixed,
           sim=sim, gravity=torch.tensor(cfg["g"], dtype=torch.float32, device=dev),
           n_grid=n_grid, sub_dt=float(cfg.get("substep_dt", 1e-4)),
           damping=float(cfg.get("grid_v_damping_scale", 1.0)),
           to_mpm=lambda x: x, undo=lambda x: x)
print(f"[앵커] {M} 개, 반경 {radius:.5f}, 고정 {int(fixed.sum())}", flush=True)

fit = AnchorSparse(sc, c=a.c, eig_floor=a.eig_floor,
                   softmax_w=a.softmax_w, softmax_k=a.softmax_k).to(dev)
fit.refresh()
fit.set_B_ref()
print(f"[가중치] {'kNN softmax k=%d' % a.softmax_k if a.softmax_w else '잘린 가우시안 c=%g' % a.c}"
      f", 짝 {fit.pair_g.shape[0]}", flush=True)

# ---------------------------------------------------------------- 지표


def chamfer(p, q, chunk):
    """양방향 평균 최근접 거리. cdist 를 한 번에 못 잡으므로 행으로 쪼갠다."""
    def one(u, v):
        s, n = 0.0, u.shape[0]
        for i in range(0, n, chunk):
            s += float(torch.cdist(u[i:i + chunk], v).min(1).values.sum())
        return s / n
    return 0.5 * (one(p, q) + one(q, p))


def emd(p, q, n, seed):
    """부분표본 위 최적 일대일 대응의 평균 이동량.

    두 구름에서 **서로 다른** 인덱스를 뽑으면, 같은 모양에서 뽑은 두 부분표본
    사이에도 남는 표본 간격이 값에 그대로 실린다. 40000 점에서 2048 을 뽑을 때
    그 바닥이 1.9% 라 재구성 오차(0.5%)를 통째로 덮었다. 같은 인덱스를 쓰면
    대응 자체는 여전히 헝가리안이 자유롭게 고르되 그 바닥이 사라진다.
    """
    from scipy.optimize import linear_sum_assignment
    gg = torch.Generator().manual_seed(seed)
    i = torch.randperm(p.shape[0], generator=gg)[:n].to(p.device)
    d = torch.cdist(p[i], q[i]).double().cpu().numpy()
    r, c_ = linear_sum_assignment(d)
    return float(d[r, c_].mean())


@torch.no_grad()
def evaluate(name):
    fit.refresh()
    cache = fit.prepare()
    fac = fit.ls_factor(cache)
    rows = []
    for t in range(TR.shape[0]):
        x = TR[t]
        xh = fit.gaussian_pos(fit.project_ls(x, cache, fac), cache)
        rows.append(dict(frame=t,
                         cd=chamfer(xh, x, a.cd_chunk) / EXT,
                         emd=emd(xh, x, a.emd_sample, t) / EXT,
                         rmse=float((xh - x).norm(dim=-1).pow(2).mean().sqrt()) / EXT))
    cd = float(np.mean([r["cd"] for r in rows]))
    em = float(np.mean([r["emd"] for r in rows]))
    rm = float(np.mean([r["rmse"] for r in rows]))
    print(f"[{name}] CD {100*cd:.4f}%  EMD {100*em:.4f}%  RMSE {100*rm:.4f}%  "
          f"(마지막 프레임 CD {100*rows[-1]['cd']:.4f}%)", flush=True)
    return dict(cd_mean=cd, emd_mean=em, rmse_mean=rm,
                cd_last=rows[-1]["cd"], emd_last=rows[-1]["emd"], rows=rows)


before = evaluate("학습 전")

# ---------------------------------------------------------------- 학습
opt = torch.optim.Adam([
    {"params": [fit.pos], "lr": a.lr_pos},
    {"params": [fit.log_s, fit.log_amp], "lr": a.lr_scale},
    {"params": [fit.quat], "lr": a.lr_quat},
])
gen = torch.Generator(device=dev).manual_seed(a.seed)
hist = []
for it in tqdm(range(a.iters), desc="학습", ncols=80):
    if it % a.refresh_every == 0:
        fit.refresh()
    cache = fit.prepare()
    fac = fit.ls_factor(cache)
    idx = torch.randint(0, TR.shape[0], (a.batch,), generator=gen, device=dev)
    loss = 0.0
    for i in idx.tolist():
        x = TR[i]
        xh = fit.gaussian_pos(fit.project_ls(x, cache, fac), cache)
        loss = loss + (xh - x).pow(2).sum(-1).mean()
    loss = loss / a.batch / (EXT ** 2)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([fit.pos, fit.log_s, fit.quat, fit.log_amp], 1.0)
    opt.step()
    fit.clamp_()
    hist.append(float(loss))
    if it % 50 == 0 or it == a.iters - 1:
        tqdm.write(f"  it {it:4d}  loss {float(loss):.3e}  "
                   f"(rmse {100*float(loss)**0.5:.3f}%)")

after = evaluate("학습 후")

os.makedirs(a.out, exist_ok=True)
torch.save({"pos": fit.pos.detach().cpu(), "log_s": fit.log_s.detach().cpu(),
            "quat": fit.quat.detach().cpu(), "log_amp": fit.log_amp.detach().cpu(),
            "args": vars(a)}, os.path.join(a.out, f"geom_{a.tag}.pt"))
json.dump(dict(tag=a.tag, softmax_w=a.softmax_w, softmax_k=a.softmax_k,
               h5_dir=a.h5_dir, n_pts=int(N), n_anchors=int(M), extent=EXT,
               frames=int(TR.shape[0]), iters=a.iters,
               before=before, after=after, loss_hist=hist[::10]),
          open(os.path.join(a.out, f"geom_{a.tag}.json"), "w"), indent=1,
          ensure_ascii=False)
print(f"\n[요약] {a.tag}: CD {100*before['cd_mean']:.4f}% -> "
      f"{100*after['cd_mean']:.4f}% | EMD {100*before['emd_mean']:.4f}% -> "
      f"{100*after['emd_mean']:.4f}%", flush=True)
print(f"[저장] {a.out}", flush=True)
print("GFGEOM_OK")
