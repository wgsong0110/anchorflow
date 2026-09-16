"""앵커 변형 모델을 GaussianFluent 파괴 궤적 위에서 학습한다.

한 프레임이 한 스텝이다 -- 서브스텝은 없다. 모델은 앵커 상태와 앵커별로 집계한
가우시안 상태를 받아 앵커마다 (변위, 반경, 온도) 를 내고, 그것으로 만든 변형 사상이
다음 프레임의 가우시안 위치와 모양을 정한다 (lib/anchorflow/deform.py).

앵커 초기화가 이 설계의 이점 하나를 그대로 쓴다: 앵커는 FPS 로 고른 **가우시안**
이므로, 어느 프레임에서 시작하든 그 프레임의 가우시안 위치가 곧 앵커의 초기
위치다. 그래서 임의의 시점에서 창을 열어 몇 프레임 펼치는 학습이 자연스럽다.

손실은 둘이다.

  위치   다음 프레임 가우시안 위치. 물체 크기로 나눈다.
  모양   변형 사상의 야코비안 J 를 GT 변형구배의 한 프레임 증분 F_{t+1} F_t^-1 에
         맞춘다. 위치만 채점하면 모양은 자유롭게 틀려도 되는데, 모양은 렌더링에
         곧바로 들어간다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import numpy as np
import torch
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="gen_gf_trajs.py 가 만든 .pt 들의 디렉토리")
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="deform")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--iters", type=int, default=4000)
ap.add_argument("--batch", type=int, default=1,
                help="한 스텝에 평균 낼 창의 수. 창마다 손실이 수십 배 다르므로 "
                     "하나만 쓰면 기울기가 그 차이에 휘둘린다")
ap.add_argument("--unroll", type=int, default=1, help="한 창에서 펼칠 프레임 수")
ap.add_argument("--unroll_final", type=int, default=4)
ap.add_argument("--unroll_at", type=float, default=0.4,
                help="이 비율을 지나면 unroll 을 unroll_final 로 늘린다")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--lambda_J", type=float, default=0.1)
ap.add_argument("--shape_loss", default="frob", choices=("frob", "bures", "none"),
                help="모양을 어떻게 채점할지. frob 은 J 와 GT 증분의 Frobenius, "
                     "bures 는 위치까지 포함한 입자별 Bures-Wasserstein (길이^2 "
                     "단위라 lambda_J 가 필요 없다), none 은 위치만")
ap.add_argument("--sigma0", type=float, default=0.5,
                help="정준 가우시안의 등방 표준편차를 입자 간격의 몇 배로 볼지. "
                     "이 씬의 ply 는 sigma ~ 1e-9 로 사실상 점이라, 모양 항을 쓰려면 "
                     "부피에서 온 크기를 줘야 한다")
ap.add_argument("--n_pts", type=int, default=20000, help="한 스텝에 쓰는 가우시안 수")
ap.add_argument("--eval_t0", type=int, nargs="+", default=[5, 40, 80],
                help="롤아웃을 시작할 프레임들. 이 궤적은 충돌 직후와 안정된 뒤의 "
                     "프레임당 변위가 수십 배 달라서, 한 구간만 보면 오해한다")
ap.add_argument("--eval_len", type=int, default=15)
ap.add_argument("--hold_last", type=int, default=20,
                help="각 궤적의 마지막 몇 프레임을 평가용으로 뗀다")
ap.add_argument("--hold_traj", default=None,
                help="통째로 홀드아웃할 궤적 태그 (쉼표로 구분)")
ap.add_argument("--save_every", type=int, default=500)
ap.add_argument("--resume", default=None)
ap.add_argument("--r2", default=None, help="체크포인트를 올릴 R2 경로")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(a.seed)

from anchorflow import deform                                  # noqa: E402
from anchorflow.deform import (DeformNet, aggregate, bc_features,   # noqa: E402
                               bures_w2_sq, grid_knn, jacobian_of, skin)

# ---------------------------------------------------------------- 데이터
files = sorted(glob.glob(os.path.join(a.data, "*.pt")))
if not files:
    raise SystemExit(f"궤적이 없다: {a.data}")
hold = set((a.hold_traj or "").split(",")) - {""}
TR, held = [], []
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    tag = os.path.splitext(os.path.basename(f))[0]
    (held if tag in hold else TR).append((tag, d))
if not TR:
    raise SystemExit("학습할 궤적이 없다")
print(f"[데이터] 학습 {len(TR)} 궤적, 홀드아웃 {len(held)} 궤적", flush=True)
for tag, d in TR + held:
    c = d["cfg"]
    print(f"  {tag}: {d['x'].shape[0]} 프레임 x {d['x'].shape[1]} 입자, "
          f"E={c['E']:g} nu={c['nu']:g} xi={c.get('xi', 0):g} g={c['g']}", flush=True)

cfg0 = TR[0][1]["cfg"]
FRAME_DT = float(cfg0["frame_dt"])
X0 = TR[0][1]["x"][0]
EXT = float((X0.max(0).values - X0.min(0).values).norm())
N_FULL = X0.shape[0]

# 앵커는 t=0 배치에서 FPS 로 고른 **가우시안**이다. 인덱스로 들고 있으면 어느
# 프레임에서든 그 프레임의 가우시안 위치로 앵커를 초기화할 수 있다.


def fps(x, M, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.zeros(M, dtype=torch.long, device=x.device)
    idx[0] = torch.randint(x.shape[0], (1,), generator=g, device=x.device)
    d = (x - x[idx[0]]).norm(dim=-1)
    for i in range(1, M):
        idx[i] = d.argmax()
        d = torch.minimum(d, (x - x[idx[i]]).norm(dim=-1))
    return idx


X0d = X0.to(dev)
AIDX = fps(X0d, a.n_anchors, a.seed)
H = float(torch.cdist(X0d[AIDX], X0d[AIDX]).topk(
    2, largest=False).values[:, 1].median())          # 앵커 간격
print(f"[앵커] {a.n_anchors} 개 (FPS), 간격 {H:.5f}, 물체 {EXT:.4f}", flush=True)

# 질량: 격자 점유로 부피를 재고 config 의 밀도를 곱한다 (MPM 이 하는 것과 같다)
ng = int(cfg0.get("n_grid", 100))
dx = float(cfg0.get("grid_lim", 2.0)) / ng
vi = (X0d / dx).long().clamp(0, ng - 1)
flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cnt = torch.zeros(ng ** 3, device=dev).index_add_(
    0, flat, torch.ones(N_FULL, device=dev))
MASS = ((dx ** 3) / cnt[flat]) * float(cfg0["density"])

# 물성은 앵커 특징에 들어간다. 궤적마다 다르므로 궤적별로 만든다.
def mat_feat(cfg):
    # 중력은 방향이라 평행이동 등변성을 깨지 않는다. 궤적마다 다르므로 넣는다.
    g = torch.tensor(cfg["g"], device=dev, dtype=torch.float32) / 15.0
    # np.log 는 float64 를 돌려주고, 목록에 섞이면 텐서가 통째로 double 이 된다
    return torch.cat([torch.tensor(
        [np.log(float(cfg["E"])), float(cfg["nu"]), float(cfg.get("xi", 0.0)),
         np.log(float(cfg["density"]))], device=dev, dtype=torch.float32), g])


VEL_SCALE = EXT / FRAME_DT
# 정준 공분산. ply 의 것을 쓸 수 없어(사실상 0) 입자 간격에서 만든다 -- MPM 이
# 부피를 쓰는 것과 같은 근거다. 등방이므로 L0 = sigma0 * I.
SIG0 = a.sigma0 * float(dx)
N_MAT = 7
n_bc = bc_features(X0d[:2], cfg0).shape[-1]
n_feat_probe = None

# ---------------------------------------------------------------- 모델
net = None
opt = None
step0 = 0


def build(n_feat):
    global net, opt
    net = DeformNet(n_feat=n_feat, hidden=a.hidden, depth=a.depth, heads=a.heads,
                    scale=0.02 * EXT, h=H, ext=EXT, seed=a.seed).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    n = sum(p.numel() for p in net.parameters())
    print(f"[모델] 입력 {n_feat}, 파라미터 {n/1e6:.2f}M", flush=True)


def take(t_cpu, idx_gpu):
    """CPU 에 있는 궤적에서 부분표본을 떼어 GPU 로. 색인을 CPU 에서 하지 않으면
    torch 가 장치가 섞였다고 거부한다 -- 궤적 전체(궤적 7 개 x 101 x 40000 x 3)를
    GPU 에 올릴 수는 없으므로 CPU 색인이 맞다."""
    return t_cpu[idx_gpu.cpu()].to(dev, non_blocking=True)


def step_once(d, t, gsel, p, x, v, need_J=True):
    """한 프레임. -> (x_next, p_next, v_next, J, w)"""
    cfg = d["cfg"]
    X = take(d["x"][0], gsel)
    idx, _ = grid_knn(x, p, a.k)
    feat, _ = aggregate(x, v / VEL_SCALE, X, MASS[gsel], idx, p.shape[0], H,
                        pa=p)
    extra = torch.cat([mat_feat(cfg).reshape(1, N_MAT).expand(p.shape[0], N_MAT),
                       bc_features(p, cfg) / H], -1)
    dp, log_r, log_t = net(p, torch.cat([feat, extra], -1), FRAME_DT)
    x2, w = skin(x, p, dp, log_r, log_t, idx, H)
    J = None
    if need_J:
        J = jacobian_of(lambda q: skin(q, p, dp, log_r, log_t, idx, H)[0], x)
    return x2, p + dp, (x2 - x) / FRAME_DT, J, w


def window(d, t0, L, gsel):
    """궤적 d 의 t0 에서 L 프레임. 앵커는 그 프레임의 GT 가우시안으로 초기화.

    같은 창에서 **아무것도 안 했을 때**의 오차도 함께 낸다. 이 궤적은 충돌 직후와
    안정된 뒤의 프레임당 변위가 수십 배 차이라, 손실의 절대값만 보면 어려운 창을
    뽑았는지 모델이 나빠졌는지 구별할 수 없다. 정지 기준선과의 비를 봐야 한다
    (기준선은 모델과 무관한 양이므로 자체 변위 정규화가 아니다).
    """
    x = take(d["x"][t0], gsel)
    v = (x - take(d["x"][max(t0 - 1, 0)], gsel)) / FRAME_DT
    p = take(d["x"][t0], AIDX)
    loss_x = loss_J = 0.0
    still = 0.0
    x_still = x.clone()
    for i in range(L):
        x2, p, v, J, _ = step_once(d, t0 + i, gsel, p, x, v,
                                   need_J=a.lambda_J > 0)
        gt = take(d["x"][t0 + i + 1], gsel)
        loss_x = loss_x + ((x2 - gt) ** 2).sum(-1).mean() / (EXT ** 2)
        still = still + float(((x_still - gt) ** 2).sum(-1).mean()) / (EXT ** 2)
        if a.shape_loss != "none" and a.lambda_J > 0:
            F0 = take(d["F"][t0 + i], gsel)
            F1 = take(d["F"][t0 + i + 1], gsel)
            Jgt = F1 @ torch.linalg.inv(F0 + 1e-4 * torch.eye(3, device=dev))
            if a.shape_loss == "frob":
                loss_J = loss_J + ((J - Jgt) ** 2).sum((-1, -2)).mean()
            else:
                # 현재 프레임 가우시안의 인수 L_t = sigma0 * F_t. 예측/정답 공분산은
                # 각각 (J L_t)(J L_t)^T, (Jgt L_t)(Jgt L_t)^T 이므로 인수만 넘기면 된다.
                Lt = SIG0 * F0
                loss_J = loss_J + (bures_w2_sq(x2, gt, J @ Lt, Jgt @ Lt).mean()
                                   / (EXT ** 2))
        x = x2
    return (loss_x / L,
            (loss_J / L if a.lambda_J > 0 else torch.zeros((), device=dev)),
            still / L)


# 특징 차원을 한 번 재서 모델을 세운다
with torch.no_grad():
    _d = TR[0][1]
    _g = torch.arange(min(a.n_pts, N_FULL), device=dev)
    _x = take(_d["x"][0], _g)
    _p = take(_d["x"][0], AIDX)
    _i, _ = grid_knn(_x, _p, a.k)
    _f, _ = aggregate(_x, torch.zeros_like(_x), _x, MASS[_g], _i, _p.shape[0],
                      H, pa=_p)
    n_feat = _f.shape[-1] + N_MAT + n_bc
build(n_feat)

# 입력 표준화 통계는 실제로 뽑는 것과 같은 분포에서 모은다. 채널 스케일이 네 자릿수
# 넘게 벌어져 있어서(질량은 log 라 -16, 다음 변위를 결정하는 속도는 7e-4) 그냥 넣으면
# 정작 중요한 채널이 첫 Linear 에서 묻힌다.
with torch.no_grad():
    samp = []
    gstat = torch.Generator(device=dev).manual_seed(a.seed + 7)
    for _ in range(64):
        _tag, _dd = TR[int(torch.randint(len(TR), (1,), generator=gstat, device=dev))]
        _t = int(torch.randint(1, _dd["x"].shape[0] - 2, (1,), generator=gstat,
                               device=dev))
        _gs = torch.randperm(N_FULL, generator=gstat,
                             device=dev)[:min(4000, N_FULL)].sort().values
        _x = take(_dd["x"][_t], _gs)
        _v = (_x - take(_dd["x"][_t - 1], _gs)) / FRAME_DT
        _pp = take(_dd["x"][_t], AIDX)
        _ii, _ = grid_knn(_x, _pp, a.k)
        _ff, _ = aggregate(_x, _v / VEL_SCALE, take(_dd["x"][0], _gs), MASS[_gs],
                           _ii, _pp.shape[0], H, pa=_pp)
        _ee = torch.cat([mat_feat(_dd["cfg"]).reshape(1, N_MAT).expand(_pp.shape[0],
                                                                      N_MAT),
                         bc_features(_pp, _dd["cfg"]) / H], -1)
        samp.append(torch.cat([_ff, _ee], -1))
    samp = torch.cat(samp, 0)
    net.set_input_stats(samp)
    print(f"[표준화] 표본 {samp.shape[0]} x {samp.shape[1]}, 채널 표준편차 "
          f"최소 {float(net.in_sd.min()):.2e} 최대 {float(net.in_sd.max()):.2e}, "
          f"평균 절대값 최대 {float(net.in_mu.abs().max()):.2e}", flush=True)

if a.resume and os.path.exists(a.resume):
    ck = torch.load(a.resume, map_location=dev, weights_only=False)
    net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
    step0 = int(ck["step"])
    print(f"[재개] {a.resume} step {step0}", flush=True)

os.makedirs(a.out, exist_ok=True)
gen = torch.Generator(device=dev).manual_seed(a.seed)
hist = []
t_start = time.time()
pbar = tqdm(range(step0, a.iters), desc="학습", ncols=90)
for it in pbar:
    L = a.unroll if it < a.unroll_at * a.iters else a.unroll_final
    opt.zero_grad(set_to_none=True)
    lx = lJ = 0.0
    still = 0.0
    for _ in range(a.batch):
        tag, d = TR[int(torch.randint(len(TR), (1,), generator=gen, device=dev))]
        T = d["x"].shape[0] - a.hold_last
        t0 = int(torch.randint(1, max(T - L - 1, 2), (1,), generator=gen,
                               device=dev))
        gsel = torch.randperm(N_FULL, generator=gen,
                              device=dev)[:a.n_pts].sort().values
        wx, wJ, wst = window(d, t0, L, gsel)
        # 창마다 바로 역전파해 누적한다 -- 창 여러 개의 그래프를 동시에 들고 있으면
        # 야코비안까지 붙어 메모리가 배치 수만큼 늘어난다
        ((wx + a.lambda_J * wJ) / a.batch).backward()
        lx = lx + float(wx) / a.batch
        lJ = lJ + float(wJ) / a.batch
        still = still + wst / a.batch
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    hist.append((lx, lJ, still))
    if it % 20 == 0:
        pbar.set_postfix(x=f"{100*lx**0.5:.3f}%", 정지=f"{100*still**0.5:.3f}%",
                         비=f"{(lx/max(still,1e-20))**0.5:.2f}",
                         J=f"{lJ:.1e}", L=L, gn=f"{float(gn):.1e}")
    if (it + 1) % a.save_every == 0 or it == a.iters - 1:
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                    "step": it + 1, "aidx": AIDX.cpu(), "H": H, "EXT": EXT,
                    "n_feat": n_feat, "args": vars(a)},
                   os.path.join(a.out, f"{a.tag}_last.pt"))
        if a.r2:
            os.system(f"rclone copy {a.out} {a.r2} --include '*.pt' "
                      f"--include '*.json' >/dev/null 2>&1 &")

# ---------------------------------------------------------------- 평가
@torch.no_grad()
def rollout(d, t0, L, gsel):
    x = take(d["x"][t0], gsel)
    v = (x - take(d["x"][max(t0 - 1, 0)], gsel)) / FRAME_DT
    p = take(d["x"][t0], AIDX)
    errs, stills = [], []
    x_still = x.clone()
    for i in range(L):
        with torch.enable_grad():
            x2, p, v, _, _ = step_once(d, t0 + i, gsel, p, x, v, need_J=False)
        x2 = x2.detach(); p = p.detach(); v = v.detach()
        gt = take(d["x"][t0 + i + 1], gsel)
        errs.append(float((x2 - gt).norm(dim=-1).mean()) / EXT)
        stills.append(float((x_still - gt).norm(dim=-1).mean()) / EXT)
        x = x2
    return errs, stills


gsel = torch.arange(min(a.n_pts, N_FULL), device=dev)
rows = {}
for tag, d in TR + held:
    T = d["x"].shape[0]
    rows[tag] = dict(held=(tag in hold), windows={})
    for t0 in a.eval_t0:
        L = min(a.eval_len, T - t0 - 1)
        if L < 2:
            continue
        e, st = rollout(d, t0, L, gsel)
        rows[tag]["windows"][t0] = dict(L=L, err=e, still=st,
                                        err_mean=float(np.mean(e)),
                                        still_mean=float(np.mean(st)))
        print(f"[롤아웃] {tag}{' (홀드아웃)' if tag in hold else ''} t0={t0:3d}: "
              f"{L} 프레임, 평균 {100*np.mean(e):.3f}% "
              f"(정지 {100*np.mean(st):.3f}%, 비 "
              f"{np.mean(e)/max(np.mean(st),1e-12):.2f})", flush=True)
r_all = [(w["err_mean"], w["still_mean"]) for r in rows.values()
         for w in r["windows"].values()]
print(f"\n[요약] 전체 창 평균 {100*np.mean([x for x,_ in r_all]):.3f}% "
      f"(정지 {100*np.mean([y for _,y in r_all]):.3f}%, 비 "
      f"{np.mean([x/max(y,1e-12) for x,y in r_all]):.2f}) "
      f"-- 비가 1 보다 작아야 도움이 된 것이다", flush=True)

json.dump(dict(tag=a.tag, args=vars(a), extent=EXT, h=H, n_feat=n_feat,
               minutes=(time.time() - t_start) / 60,
               loss_hist=hist[::20], rollout=rows),
          open(os.path.join(a.out, f"{a.tag}.json"), "w"), indent=1,
          ensure_ascii=False)
print(f"[저장] {a.out}", flush=True)
print("DEFORM_OK")
