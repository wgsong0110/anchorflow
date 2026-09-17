"""프레임 하나를 화면에 내기까지의 전 구간을 잰다.

지금까지는 조각별로만 재서 "실시간" 을 조각의 합으로 주장했다. 실제로는 앵커 수가
달라지면 어텐션 비용도 달라지고(복셀 앙상블은 앵커가 4 배다), 래스터화도 붙는다.
그래서 여기서는 구성마다 **한 프레임 전체**를 재고 비교 대상도 함께 둔다.

  기존        FPS 로 앵커를 다시 뽑고 kNN 으로 소속을 찾는다.
  복셀        점유 복셀이 앵커. 소속은 이웃 칸 색인.
  복셀 앙상블 오프셋이 다른 격자 L 개를 한 커널로. 떨림이 절반이 되는 대신 앵커가 L 배.
  학생        고정 연결성 스테퍼. 앵커 선정도 소속 탐색도 집계도 없다 -- 이 방법이
              동적 결합을 위해 치르는 값이 무엇인지 보여주는 기준선이다.
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time
_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ply", default=None)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--ens", type=int, default=4)
ap.add_argument("--width", type=int, default=800)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--out", default=None)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
from anchorflow import voxel
from anchorflow.deform import (DeformNet, aggregate, anchor_knn, bc_features,
                               fps, skin_with_jacobian)

f = sorted(glob.glob(os.path.join(a.data, "*.pt")))[0]
d = torch.load(f, map_location="cpu", weights_only=False)
cfg = d["cfg"]; X0 = d["x"][0]
EXT = float((X0.max(0).values - X0.min(0).values).norm())
FRAME_DT = float(cfg["frame_dt"])
if a.ply:
    from plyfile import PlyData
    v_ = PlyData.read(a.ply)["vertex"]
    t_ = torch.from_numpy(np.stack([v_["x"], v_["y"], v_["z"]], 1)).float()
    t_ = t_ - t_.mean(0)
    t_ = t_ / float((t_.max(0).values - t_.min(0).values).norm()) * EXT
    x = (t_ + X0.mean(0)).to(dev).contiguous()
else:
    x = X0.to(dev).contiguous()
N = x.shape[0]; X = x.clone()
v = torch.randn_like(x) * 0.01
m = torch.rand(N, device=dev) + 0.1
MAT = torch.cat([torch.tensor([np.log(float(cfg["E"])), float(cfg["nu"]),
                               float(cfg.get("xi", 0.)), np.log(float(cfg["density"]))],
                              device=dev, dtype=torch.float32),
                 torch.tensor(cfg["g"], device=dev, dtype=torch.float32) / 15.0])
aid0 = fps(x, a.n_anchors)
H = float(torch.cdist(x[aid0], x[aid0]).topk(2, largest=False).values[:, 1].median())
LO = (d["x"].reshape(-1, 3).min(0).values - 2 * H).to(dev)
OFFS = np.array([np.random.RandomState(i).rand(3) for i in range(a.ens)])
print(f"[씬] 입자 {N}, 물체 {EXT:.4f}, 복셀 한 변 {H:.5f}, "
      f"프레임 간격 {FRAME_DT*1000:.1f} ms", flush=True)


def timeit(fn, n=a.reps):
    for _ in range(4): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3


def make_net(M, n_feat):
    return DeformNet(n_feat=n_feat, hidden=128, depth=4, heads=4,
                     scale=0.02 * EXT, h=H, ext=EXT).to(dev).eval()


def raster(Ng):
    from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                             GaussianRasterizer)
    tanfov = float(np.tan(np.radians(30.)))
    vt = torch.eye(4, device=dev); vt[3, 2] = 4.0
    means = (x[:Ng] - x[:Ng].mean(0)).contiguous()
    cov = torch.zeros(Ng, 6, device=dev); cov[:, 0] = cov[:, 3] = cov[:, 5] = (0.004*EXT)**2
    opa = torch.full((Ng, 1), .8, device=dev); col = torch.rand(Ng, 3, device=dev)
    scr = torch.zeros_like(means)
    st = GaussianRasterizationSettings(image_height=a.width, image_width=a.width,
        tanfovx=tanfov, tanfovy=tanfov, bg=torch.ones(3, device=dev),
        scale_modifier=1., viewmatrix=vt, projmatrix=vt, sh_degree=3,
        campos=torch.zeros(3, device=dev), prefiltered=False, debug=False)
    r = GaussianRasterizer(raster_settings=st)
    return lambda: r(means3D=means, means2D=scr, shs=None, colors_precomp=col,
                     opacities=opa, scales=None, rotations=None, cov3D_precomp=cov)


t_ras = timeit(raster(N), 10)
rows = {}

# ---- 기존: FPS + kNN ----
p0 = x[aid0].contiguous()
i0, _ = anchor_knn(x, p0, a.k)
f0, _ = aggregate(x, v, X, m, i0, a.n_anchors, H, pa=p0)
net0 = make_net(a.n_anchors, f0.shape[-1] + MAT.numel() + bc_features(p0[:2], cfg).shape[-1])
e0 = torch.cat([MAT.reshape(1, -1).expand(a.n_anchors, -1), bc_features(p0, cfg)/H], -1)
dp0, lr0, lt0 = net0(p0, torch.cat([f0, e0], -1), FRAME_DT)


def frame_old():
    ai = fps(x, a.n_anchors); pp = x[ai]
    ii, _ = anchor_knn(x, pp, a.k)
    ff, _ = aggregate(x, v, X, m, ii, a.n_anchors, H, pa=pp)
    ee = torch.cat([MAT.reshape(1, -1).expand(a.n_anchors, -1),
                    bc_features(pp, cfg)/H], -1)
    dd, rr, tt = net0(pp, torch.cat([ff, ee], -1), FRAME_DT)
    skin_with_jacobian(x, pp, dd, rr, tt, ii, H)


rows["매 프레임 FPS + kNN (앵커 512)"] = timeit(frame_old) + t_ras


# ---- t=0 에만 FPS, 이후 앵커는 모델 출력으로 갱신, 소속은 매 프레임 kNN ----
# 앵커 선정만 빠지고 kNN 과 집계는 그대로 든다. 복셀이 셋을 한 번에 접는 것과
# 견주려면 이 구성이 맞는 비교 대상이다.
def frame_track():
    ii, _ = anchor_knn(x, p0, a.k)
    ff, _ = aggregate(x, v, X, m, ii, a.n_anchors, H, pa=p0)
    ee = torch.cat([MAT.reshape(1, -1).expand(a.n_anchors, -1),
                    bc_features(p0, cfg) / H], -1)
    dd, rr, tt = net0(p0, torch.cat([ff, ee], -1), FRAME_DT)
    skin_with_jacobian(x, p0, dd, rr, tt, ii, H)


rows["t=0 FPS 후 추적 + kNN (앵커 512)"] = timeit(frame_track) + t_ras

# ---- 복셀 ----
# 앙상블은 격자 수만큼 앵커가 늘어난다. 같은 앵커 예산에서 견주려면 칸을 L^(1/3)
# 배 성기게 해야 한다 -- 그래야 "떨림을 줄이려고 해상도를 내준" 값이 드러난다.
H_ENS = H * (a.ens ** (1.0 / 3.0))
VARIANTS = [("복셀 (격자 1)", None, H),
            (f"복셀 앙상블 (격자 {a.ens}, 같은 해상도)", OFFS, H),
            (f"복셀 앙상블 (격자 {a.ens}, 앵커수 보정)", OFFS, H_ENS)]
for name, offs, cell in VARIANTS:
    vb = voxel.build(x, X, v, m, cell, lo=LO, offsets=offs)
    gi, _ = voxel.neighbors(x, vb, a.k)
    nf = f0.shape[-1] + MAT.numel() + bc_features(vb.pos[:2], cfg).shape[-1]
    net = make_net(vb.M, nf)

    def frame_vox(offs=offs, net=net, nf=nf, cell=cell):
        vb = voxel.build(x, X, v, m, cell, lo=LO, offsets=offs)
        gi, _ = voxel.neighbors(x, vb, a.k)
        feat = torch.zeros(vb.M, nf - MAT.numel()
                           - bc_features(vb.pos[:2], cfg).shape[-1], device=dev)
        ee = torch.cat([feat, MAT.reshape(1, -1).expand(vb.M, -1),
                        bc_features(vb.pos, cfg)/cell], -1)
        dd, rr, tt = net(vb.pos, ee, FRAME_DT)
        skin_with_jacobian(x, vb.pos, dd, rr, tt, gi, H)

    rows[name + f" 앵커 {vb.M}"] = timeit(frame_vox) + t_ras

# ---- 학생 기준선 ----
from anchorflow.nextstate import NextStep, apply_step
stu = NextStep(hidden=128, depth=4, heads=4, use_accel=False, scale=EXT,
               vel_scale=EXT/FRAME_DT, zero_init=True).to(dev).eval()
ps = x[aid0].contiguous(); vs = torch.zeros_like(ps)
fixed = torch.zeros(a.n_anchors, dtype=torch.bool, device=dev)
i_fix, _ = anchor_knn(x, ps, a.k)
dpf = torch.randn(a.n_anchors, 3, device=dev) * 0.01
lrf = torch.full((a.n_anchors,), float(np.log(H)), device=dev)
ltf = torch.zeros(a.n_anchors, device=dev)


def frame_student():
    apply_step(stu, ps, vs, None, FRAME_DT, fixed)
    skin_with_jacobian(x, ps, dpf, lrf, ltf, i_fix, H)   # 고정 짝, 탐색 없음


rows["학생 (고정 연결성)"] = timeit(frame_student) + t_ras

print(f"\n래스터화만 ({a.width}x{a.width}, 가우시안 {N}): {t_ras:.2f} ms\n", flush=True)
for k_, v_ in rows.items():
    print(f"  {k_:<34} {v_:7.2f} ms = {1000/v_:6.1f} fps  "
          f"{'실시간' if v_ < FRAME_DT*1000 else '실시간 아님'}", flush=True)
if a.out:
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(N=N, raster=t_ras, frame_dt=FRAME_DT, rows=rows),
              open(os.path.join(a.out, "frame_e2e.json"), "w"), indent=1,
              ensure_ascii=False)
print("E2E_OK")
