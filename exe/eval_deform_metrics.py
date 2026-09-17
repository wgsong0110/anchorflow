"""학습된 변형 모델을 3D(CD, EMD)와 2D(PSNR, SSIM) 양쪽에서 잰다.

지금까지 본 것은 위치 RMSE 하나였는데, 그것은 입자 대응을 전제한다. 찢어지면
대응이 흐려지므로 대응을 가정하지 않는 CD/EMD 가 필요하고, 최종 산출물은 결국
화면이므로 렌더 지표도 있어야 한다.

CD/EMD 의 정의는 Spring-Gaus(ECCV 2024) 를 따른다 -- 베이스라인과 같은 자로 재야
비교가 되기 때문이다 (exe/../doc 에 정의 근거를 적어 둔다).

PSNR/SSIM 은 예측과 정답을 **같은 카메라, 같은 가우시안 외형**으로 래스터화해
위치·모양 차이만 보이게 한 뒤 잰다.
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time
_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", nargs="+", default=[])
ap.add_argument("--ply", default=None, help="외형(색·불투명도·크기)을 가져올 ply")
ap.add_argument("--traj", default="watermelon_h")
ap.add_argument("--out", required=True)
ap.add_argument("--t0", type=int, nargs="+", default=[3, 10, 20])
ap.add_argument("--frames", type=int, default=15)
ap.add_argument("--n_pts", type=int, default=20000)
ap.add_argument("--cd_pts", type=int, default=2048,
                help="CD/EMD 표본 크기 (EMD 가 O(n^3) 이라 필요하다)")
ap.add_argument("--width", type=int, default=600)
ap.add_argument("--spring_gaus", default=None,
                help="Spring-Gaus 클론 경로. 주면 같은 궤적·같은 지표로 함께 잰다")
ap.add_argument("--sg_points", type=int, default=2048, help="공식 N_SAMPLE")
ap.add_argument("--sg_neighbors", type=int, default=256, help="공식 K_NEIGHBORS")
ap.add_argument("--sg_nstep", type=int, default=100, help="공식 N_STEP")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)
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


# ------------------------------------------------------------ 3D 지표
def chamfer(p, q, chunk=4096):
    """양방향 평균 최근접 **제곱** 거리의 합 (Spring-Gaus 관례)."""
    def one(u, v):
        s, n = 0.0, u.shape[0]
        for i in range(0, n, chunk):
            dd = torch.cdist(u[i:i + chunk], v)
            s += float((dd.min(1).values ** 2).sum())
        return s / n
    return one(p, q) + one(q, p)


def emd(p, q, seed):
    """최적 일대일 대응의 평균 이동량. 두 구름에서 **같은 인덱스**를 뽑는다 --
    따로 뽑으면 같은 모양에서 뽑은 부분표본 사이의 간격이 값에 남는다."""
    from scipy.optimize import linear_sum_assignment
    g = torch.Generator().manual_seed(seed)
    i = torch.randperm(p.shape[0], generator=g)[:a.cd_pts].to(p.device)
    dd = torch.cdist(p[i], q[i]).double().cpu().numpy()
    r, c = linear_sum_assignment(dd)
    return float(dd[r, c].mean())


# ------------------------------------------------------------ 2D 지표
RAST = None
if a.ply:
    from plyfile import PlyData
    from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                             GaussianRasterizer)
    v_ = PlyData.read(a.ply)["vertex"]
    sel = d["sel"][GS].numpy()
    def gat(*names):
        return torch.from_numpy(np.stack([v_[n][sel] for n in names], 1)).float().to(dev)
    OPA = torch.sigmoid(gat("opacity"))
    SCA = torch.exp(gat("scale_0", "scale_1", "scale_2")).clamp(min=1e-4 * EXT)
    ROT = torch.nn.functional.normalize(gat("rot_0", "rot_1", "rot_2", "rot_3"))
    COL = (0.5 + 0.2820948 * gat("f_dc_0", "f_dc_1", "f_dc_2")).clamp(0, 1)
    print(f"[외형] ply 에서 {sel.shape[0]} 개, 크기 중앙 "
          f"{float(SCA.median()):.2e} (물체 {EXT:.3f})", flush=True)

    def cov_from(J):
        """Sigma = (J R S)(J R S)^T -> 상삼각 6 성분"""
        w, xq, yq, zq = ROT.unbind(-1)
        R = torch.stack([
            1 - 2*(yq*yq + zq*zq), 2*(xq*yq - w*zq), 2*(xq*zq + w*yq),
            2*(xq*yq + w*zq), 1 - 2*(xq*xq + zq*zq), 2*(yq*zq - w*xq),
            2*(xq*zq - w*yq), 2*(yq*zq + w*xq), 1 - 2*(xq*xq + yq*yq)], -1
        ).reshape(-1, 3, 3)
        L = R * SCA.unsqueeze(-2)
        if J is not None:
            L = J @ L
        S = L @ L.transpose(-1, -2)
        return torch.stack([S[:, 0, 0], S[:, 0, 1], S[:, 0, 2],
                            S[:, 1, 1], S[:, 1, 2], S[:, 2, 2]], -1).contiguous()

    def render(pos, J):
        c = X0d[GS.to(dev)].mean(0)
        tanf = float(np.tan(np.radians(25.)))
        vt = torch.eye(4, device=dev); vt[3, 2] = 3.0 * EXT
        st = GaussianRasterizationSettings(
            image_height=a.width, image_width=a.width, tanfovx=tanf, tanfovy=tanf,
            bg=torch.ones(3, device=dev), scale_modifier=1.0, viewmatrix=vt,
            projmatrix=vt, sh_degree=0, campos=torch.zeros(3, device=dev),
            prefiltered=False, debug=False)
        r = GaussianRasterizer(raster_settings=st)
        out = r(means3D=(pos - c).contiguous(), means2D=torch.zeros_like(pos),
                shs=None, colors_precomp=COL, opacities=OPA, scales=None,
                rotations=None, cov3D_precomp=cov_from(J))
        return out[0].clamp(0, 1)
    RAST = render


def psnr(a_, b_):
    return float(-10.0 * torch.log10(((a_ - b_) ** 2).mean().clamp(min=1e-12)))


def ssim(a_, b_, C1=0.01 ** 2, C2=0.03 ** 2):
    """가우시안 창 11x11, sigma 1.5 (3DGS/Spring-Gaus 가 쓰는 표준 형태)."""
    import torch.nn.functional as F
    g = torch.arange(11, device=a_.device).float() - 5
    g = torch.exp(-g ** 2 / (2 * 1.5 ** 2)); g = (g / g.sum())
    w = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11).contiguous()
    A, B = a_.unsqueeze(0), b_.unsqueeze(0)
    mu1 = F.conv2d(A, w, padding=5, groups=3); mu2 = F.conv2d(B, w, padding=5, groups=3)
    s11 = F.conv2d(A * A, w, padding=5, groups=3) - mu1 * mu1
    s22 = F.conv2d(B * B, w, padding=5, groups=3) - mu2 * mu2
    s12 = F.conv2d(A * B, w, padding=5, groups=3) - mu1 * mu2
    return float((((2*mu1*mu2 + C1) * (2*s12 + C2))
                  / ((mu1**2 + mu2**2 + C1) * (s11 + s22 + C2))).mean())


def rollout(ck, t0, L):
    st = torch.load(ck, map_location=dev, weights_only=False)
    ta = st["args"]; AIDX = st["aidx"].to(dev); H = float(st["H"])
    net = DeformNet(n_feat=int(st["n_feat"]), hidden=int(ta["hidden"]),
                    depth=int(ta["depth"]), heads=int(ta["heads"]),
                    scale=0.02 * EXT, h=H, ext=EXT, seed=int(ta["seed"])).to(dev)
    net.load_state_dict(st["net"]); net.eval()
    k = int(ta["k"])
    x = take(d["x"][t0], GS)
    v = (x - take(d["x"][max(t0 - 1, 0)], GS)) / FRAME_DT
    p = take(d["x"][t0], AIDX)
    XC = take(d["x"][0], GS)
    Jacc = torch.eye(3, device=dev).expand(x.shape[0], 3, 3).contiguous()
    out = [(x.clone(), Jacc.clone())]
    for _ in range(L):
        idx, _ = anchor_knn(x, p, k)
        feat, _ = aggregate(x, v / VEL, XC, MASS[GS.to(dev)], idx, p.shape[0], H, pa=p)
        ex = torch.cat([MAT.reshape(1, -1).expand(p.shape[0], -1),
                        bc_features(p, cfg) / H], -1)
        dp, lr_, lt_ = net(p, torch.cat([feat, ex], -1), FRAME_DT)
        x2, _, J = skin_with_jacobian(x, p, dp, lr_, lt_, idx, H)
        Jacc = J @ Jacc
        v, p, x = (x2 - x) / FRAME_DT, p + dp, x2
        out.append((x.clone(), Jacc.clone()))
    return out



def rollout_sg(t0, L):
    """Spring-Gaus 의 스프링-질량 시뮬레이터를 같은 조건에서 굴린다.

    값은 공식 config 그대로다 (mpm_synthetic/default.yaml): 질량점 2048 개,
    이웃 256, 프레임당 서브스텝 100. 물성은 학습하지 않는다 -- 다른 베이스라인과
    같은 조건에서 **시뮬레이터만** 재는 것이 목적이다. 이 방법은 탄성 전용이라
    소성·파괴가 구조적으로 표현되지 않는데, 그것이 이 비교로 보이려는 것이다.
    """
    sys.path.insert(0, a.spring_gaus)
    from lib.models.spring_mass.Spring_Mass import Spring_Mass
    from yacs.config import CfgNode as CN
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
    x0 = take(d["x"][t0], GS)
    g = torch.Generator().manual_seed(a.seed)
    pi = torch.randperm(x0.shape[0], generator=g)[:a.sg_points].to(dev)
    sim = Spring_Mass(sc, x0[pi].clone()).to(dev)
    if hasattr(sim, "device"):
        sim.device = dev
    sim.set_dt(dt=FRAME_DT)
    # 시뮬 대상은 질량점 2048 개다. 가우시안 결속은 아래에서 직접 한다 --
    # set_all_particle 에 전체를 넘기면 forward 가 전체를 돌려주어 상태가 섞인다.
    sim.set_all_particle(x0[pi].clone())
    sim.stage = "dynamic"
    xs = x0[pi].clone()
    vs = ((x0 - take(d["x"][max(t0 - 1, 0)], GS)) / FRAME_DT)[pi].clone()
    # 질량점 변위를 가우시안으로 옮기는 결속. 공식 K_BINDING 과 같은 16 이웃이다.
    dd = torch.cdist(x0, x0[pi])
    wv, ii = dd.topk(16, largest=False)
    wv = torch.softmax(-wv / wv[:, :1].clamp(min=1e-9), 1)
    out = [(x0.clone(), None)]
    for i in range(1, L + 1):
        o = sim(xs, xs, vs, frame_id=i)
        xs, vs = o[0].detach(), o[1].detach()
        out.append((x0 + ((xs - x0[pi])[ii] * wv.unsqueeze(-1)).sum(1), None))
    return out


rows = {}
if a.spring_gaus:
    per = []
    for t0 in a.t0:
        L = min(a.frames, d["x"].shape[0] - t0 - 1)
        seq = rollout_sg(t0, L)
        for i in range(1, len(seq)):
            xp, _ = seq[i]
            xg = take(d["x"][t0 + i], GS)
            m = dict(t0=t0, f=i, cd=chamfer(xp, xg) / (EXT ** 2),
                     emd=emd(xp, xg, t0 * 1000 + i) / EXT)
            if RAST is not None:
                ip, ig = RAST(xp, None), RAST(xg, None)
                m["psnr"] = psnr(ip, ig); m["ssim"] = ssim(ip, ig)
            per.append(m)
    rows["Spring-Gaus"] = per
    agg = {k: float(np.mean([r[k] for r in per])) for k in per[0]
           if k not in ("t0", "f")}
    print(f"[Spring-Gaus] CD {agg['cd']:.3e}  EMD {100*agg['emd']:.3f}%"
          + (f"  PSNR {agg['psnr']:.2f} dB  SSIM {agg['ssim']:.4f}"
             if "psnr" in agg else ""), flush=True)

for ck in a.ckpt:
    name = os.path.splitext(os.path.basename(ck))[0]
    per = []
    for t0 in a.t0:
        L = min(a.frames, d["x"].shape[0] - t0 - 1)
        seq = rollout(ck, t0, L)
        gtF = d["F"][t0].to(dev)
        for i in range(1, len(seq)):
            xp, Jp = seq[i]
            xg = take(d["x"][t0 + i], GS)
            m = dict(t0=t0, f=i,
                     cd=chamfer(xp, xg) / (EXT ** 2),
                     emd=emd(xp, xg, t0 * 1000 + i) / EXT)
            if RAST is not None:
                Jg = (d["F"][t0 + i].to(dev)
                      @ torch.linalg.inv(gtF + 1e-4 * torch.eye(3, device=dev)))
                ip, ig = RAST(xp, Jp), RAST(xg, None)
                m["psnr"] = psnr(ip, ig); m["ssim"] = ssim(ip, ig)
            per.append(m)
    rows[name] = per
    agg = {k: float(np.mean([r[k] for r in per])) for k in per[0] if k not in ("t0", "f")}
    print(f"[{name}] CD {agg['cd']:.3e}  EMD {100*agg['emd']:.3f}%"
          + (f"  PSNR {agg['psnr']:.2f} dB  SSIM {agg['ssim']:.4f}"
             if "psnr" in agg else ""), flush=True)

os.makedirs(a.out, exist_ok=True)
json.dump(dict(traj=a.traj, extent=EXT, n_pts=int(GS.numel()),
               cd_pts=a.cd_pts, rows=rows),
          open(os.path.join(a.out, "deform_metrics.json"), "w"), indent=1)
print("METRICS_OK")
