"""Spring-Gaus 를 우리 방법과 **같은 지도 신호**로 학습시킨다.

앞서는 물성을 공식 기본값(K=1000, 감쇠 0.1)으로 고정해 재서 그쪽에 불리했다.
원래 Spring-Gaus 는 다시점 영상으로 물성을 최적화하는 방법이고, 우리 변형 모델은
같은 궤적의 3D 위치로 학습한다. 둘을 견주려면 **같은 것을 보고 배우게** 해야 한다.

그래서 구조 하이퍼파라미터는 공식 default.yaml 그대로 두고(질량점 2048, 이웃 256,
프레임당 서브스텝 100), 학습 가능한 파라미터만 우리와 같은 창 표본추출·같은 위치
손실로 최적화한다. 남는 차이는 방법 자체의 차이다 -- 탄성 전용이라 소성·파괴를
구조적으로 표현할 수 없다는 것.
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time
_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)
import numpy as np
import torch
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--spring_gaus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="sg_trained")
ap.add_argument("--hold_traj", default="watermelon_h")
ap.add_argument("--n_pts", type=int, default=20000)
ap.add_argument("--sg_points", type=int, default=2048, help="공식 N_SAMPLE")
ap.add_argument("--sg_neighbors", type=int, default=256, help="공식 K_NEIGHBORS")
ap.add_argument("--sg_nstep", type=int, default=100, help="공식 N_STEP")
ap.add_argument("--iters", type=int, default=3000)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--unroll", type=int, default=1)
ap.add_argument("--unroll_final", type=int, default=4)
ap.add_argument("--unroll_at", type=float, default=0.35)
ap.add_argument("--motion_frac", type=float, default=0.8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--hold_last", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(a.seed)
sys.path.insert(0, a.spring_gaus)
from lib.models.spring_mass.Spring_Mass import Spring_Mass      # noqa: E402
from yacs.config import CfgNode as CN                            # noqa: E402

files = sorted(glob.glob(os.path.join(a.data, "*.pt")))
TR, held = [], []
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    tag = os.path.splitext(os.path.basename(f))[0]
    (held if tag == a.hold_traj else TR).append((tag, d))
cfg = TR[0][1]["cfg"]
FRAME_DT = float(cfg["frame_dt"])
X0 = TR[0][1]["x"][0]
EXT = float((X0.max(0).values - X0.min(0).values).norm())
N_FULL = X0.shape[0]
print(f"[데이터] 학습 {len(TR)} 궤적, 홀드아웃 {len(held)}, 물체 {EXT:.4f}", flush=True)

for _t, _d in TR + held:
    s_ = torch.randperm(_d["x"].shape[1])[:2000]
    _d["motion"] = (_d["x"][1:, s_] - _d["x"][:-1, s_]).norm(dim=-1).mean(1)

g = torch.Generator().manual_seed(a.seed)
PI = torch.randperm(N_FULL, generator=g)[:a.sg_points].sort().values
GS = torch.arange(min(a.n_pts, N_FULL))


def take(t, i):
    return t[i].to(dev)


sc = CN()
sc.K_NEIGHBORS = a.sg_neighbors
sc.K_BINDING = 16
sc.N_STEP = a.sg_nstep
sc.INIT_VELOCITY = [0, 0, 0]
sc.G = list(cfg.get("g", [0.0, 0.0, 0.0]))
sc.PRETRAINED = None
sc.DATA = CN()
sc.DATA.DT = FRAME_DT
sc.DATA.BC = [[[0, 0.3, 0], [0, 1, 0]]]
sc.DATA.GLOBAL_M = 1
sc.DATA.GLOBAL_K = 1000
sc.DATA.GLOBAL_DAMP = 0.1
pts0 = take(X0, PI)
sim = Spring_Mass(sc, pts0.clone()).to(dev)
if hasattr(sim, "device"):
    sim.device = dev
sim.set_dt(dt=FRAME_DT)
sim.set_all_particle(pts0.clone())
sim.stage = "dynamic"
params = [p for p in sim.parameters() if p.requires_grad]
print(f"[모델] 학습 가능한 텐서 {len(params)} 개, "
      f"{sum(p.numel() for p in params)} 원소", flush=True)
if not params:
    raise SystemExit("학습 가능한 파라미터가 없다 -- 구조를 다시 봐야 한다")
opt = torch.optim.Adam(params, lr=a.lr)

gen = torch.Generator().manual_seed(a.seed)
hist = []
pbar = tqdm(range(a.iters), desc="학습", ncols=90)
for it in pbar:
    L = a.unroll if it < a.unroll_at * a.iters else a.unroll_final
    opt.zero_grad(set_to_none=True)
    lx = still = 0.0
    for _ in range(a.batch):
        _tag, d = TR[int(torch.randint(len(TR), (1,), generator=gen))]
        T = d["x"].shape[0] - a.hold_last
        hi = max(T - L - 1, 2)
        if float(torch.rand(1, generator=gen)) < a.motion_frac:
            w_ = d["motion"][1:hi].clamp(min=1e-12)
            t0 = 1 + int(torch.multinomial(w_, 1, generator=gen))
        else:
            t0 = int(torch.randint(1, hi, (1,), generator=gen))
        x = take(d["x"][t0], PI)
        v = (x - take(d["x"][t0 - 1], PI)) / FRAME_DT
        x_still = x.clone()
        # Spring-Gaus 의 train.py 를 그대로 따른다: **프레임마다** 역전파하고
        # 상태를 끊는다 (그쪽은 xyz/v 를 매 프레임 detach().clone() 한다). 창 전체로
        # 이어 붙이면 그쪽 forward 안의 deepcopy(xyz) 가 비-리프 텐서에서 터진다.
        # forward 는 (xyz_all, xyz, v, is_nan) 을 준다 -- 속도는 세 번째다.
        for i in range(L):
            xa, xo, vo, _nan = sim(x, x, v, frame_id=i + 1)
            gt = take(d["x"][t0 + i + 1], PI)
            l1 = ((xo - gt) ** 2).sum(-1).mean() / (EXT ** 2)
            (l1 / L / a.batch).backward()
            lx += float(l1) / L / a.batch
            still += float(((x_still - gt) ** 2).sum(-1).mean()) / (EXT ** 2) / L
            x, v = xo.detach().clone(), vo.detach().clone()
    still /= a.batch
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist.append((lx, still))
    if it % 20 == 0:
        pbar.set_postfix(x=f"{100*lx**0.5:.3f}%", 정지=f"{100*still**0.5:.3f}%",
                         비=f"{(lx/max(still,1e-20))**0.5:.2f}", L=L,
                         gn=f"{float(gn):.1e}")
    if (it + 1) % 500 == 0 or it == a.iters - 1:
        os.makedirs(a.out, exist_ok=True)
        torch.save({"sim": sim.state_dict(), "pi": PI, "cfg": dict(sc),
                    "step": it + 1, "args": vars(a)},
                   os.path.join(a.out, f"{a.tag}_last.pt"))

os.makedirs(a.out, exist_ok=True)
json.dump(dict(tag=a.tag, args=vars(a), extent=EXT, loss_hist=hist[::20]),
          open(os.path.join(a.out, f"{a.tag}.json"), "w"), indent=1,
          ensure_ascii=False)
print(f"[저장] {a.out}", flush=True)
print("SGTRAIN_OK")
