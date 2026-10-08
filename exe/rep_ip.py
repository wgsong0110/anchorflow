"""표현력 비교 B: 각 표현으로 **증분 포텐셜**을 매 프레임 최소화하는 암시적 시간적분 (repflow).

L2 추적(rep_track2) 대신, 시뮬레이터(i-PG · GaussianFluent · Fracture-GS)의 장면과 구성식으로
  x_{n+1} = argmin_θ  Σ m/(2h²)|x(θ) − x_n − h v_n|² + Σ V Ψ(F_trial(θ)) − Σ m g·x + E_바닥 + E_접촉
를 푼다. 입자는 시뮬레이터 입자 전체 (표면 가우시안 + PG 공식 내부 채움), 표현은 그 전체에 건다.
F_trial = J_inc(θ) F_e,n (J 는 표현 사상의 해석적 야코비안), 소성은 프레임 끝에 phys_resid 로 사영.
최적화는 A 와 같이 각 방법의 기하 위에서 (riem: Δ = -(G + ε λmax I)⁻¹ ∇E, G 는 매 스텝 재계산),
GausSim 은 제약 없는 Adam. 목적은 Σm/h² 로 나눠 (질량가중 평균 이동²) A 의 L2 와 같은 크기로 둔다.

기준 궤적(시뮬레이터 출력 h5)은 비교 측정에만 쓴다 -- 최적화에는 쓰지 않는다.

  python exe/rep_ip.py --sim repip/ipg_lego_visco_drop/simulation_ply --cfg repip/ipg_lego_visco_drop.json \
      --shape lego --method ours --out repip/res/ipg_lego_ours.npz --video ...mp4
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
from anchorflow import repmaps as rm                                  # noqa: E402
from anchorflow import phys_resid as pr                               # noqa: E402

W = "/home/dkta/work"
MODEL = {"lego": "lego_whitebg-trained", "mic": "mic_whitebg-trained",
         "ficus": "ficus_whitebg-trained", "wolf": "wolf_whitebg-trained"}
ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True, help="기준 시뮬 h5 프레임 폴더")
ap.add_argument("--cfg", required=True, help="그 시뮬의 설정 json")
ap.add_argument("--shape", required=True)
ap.add_argument("--method", required=True, choices=["ours", "phystwin", "vrgs", "gaussim", "simplicits"])
ap.add_argument("--simp", default="", help="simplicits 가중치 함수 (.pt, 이 시뮬 입자로 학습)")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--iters0", type=int, default=400)
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--riem_eps", type=float, default=1e-2)
ap.add_argument("--riem_lr", type=float, default=1.0)
ap.add_argument("--riem_cg", type=int, default=50)
ap.add_argument("--ls_max", type=int, default=30, help="Armijo 되돌림 최대 횟수")
ap.add_argument("--riem_lr_max", type=float, default=1e8, help="적응 보폭 상한")
ap.add_argument("--dbg", action="store_true")
ap.add_argument("--iso", type=float, default=0.0, help="렌더 공분산을 등방형 (iso × 축 길이 중앙값)² I 로")
ap.add_argument("--render_npz", default="", help="저장된 결과(npz traj)를 영상으로만 (최적화 없음)")
ap.add_argument("--old_order", action="store_true",
                help="render_npz 가 예전 순서(시뮬 입자 앞 N 개 = 가우시안 가정)로 저장된 경우")
ap.add_argument("--lr", type=float, default=1e-3, help="gaussim Adam")
ap.add_argument("--k_floor", type=float, default=1e3, help="바닥 관통 벌점 (관성항 대비 배수)")
ap.add_argument("--k_contact", type=float, default=1e3, help="물체 간 접촉 벌점 (관성항 대비 배수)")
ap.add_argument("--out", required=True)
ap.add_argument("--video", default="")
ap.add_argument("--render_ref", action="store_true", help="기준 궤적만 같은 렌더러로 영상으로 (최적화 없음)")
ap.add_argument("--tb", default="auto")
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)

cfg = json.load(open(a.cfg))
files = sorted(glob.glob(os.path.join(a.sim, "sim_*.h5")))
assert len(files) > a.frames, f"기준 프레임 부족: {len(files)}"


def rd(p, k):
    with h5py.File(p, "r") as f:
        if k not in f:
            return None
        v = np.array(f[k])
    return v.T if (v.ndim == 2 and v.shape[0] in (3, 9) and v.shape[0] != v.shape[-1]) else v


X0 = torch.as_tensor(rd(files[0], "x"), dtype=torch.float32, device=dev)
N = X0.shape[0]
OBJ = rd(files[0], "obj")
OBJ = torch.zeros(N, dtype=torch.long, device=dev) if OBJ is None else \
    torch.as_tensor(OBJ.reshape(-1), dtype=torch.long, device=dev)
NOBJ = int(OBJ.max()) + 1
V0 = rd(files[0], "v")
V0 = torch.zeros_like(X0) if V0 is None else torch.as_tensor(V0, dtype=torch.float32, device=dev)
if float(V0.abs().max()) == 0 and "init_velocity" in cfg:
    V0 = torch.as_tensor(cfg["init_velocity"], dtype=torch.float32, device=dev).expand(N, 3).clone()
h = float(cfg["frame_dt"])
g = torch.as_tensor(cfg.get("g", [0.0, 0.0, -9.8]), dtype=torch.float32, device=dev)
n_grid, grid_lim = int(cfg.get("n_grid", 100)), float(cfg.get("grid_lim", 2.0))
dx = grid_lim / n_grid
# 질량·부피: 시뮬레이터와 같은 정의 (셀마다 세고 dx³/개수)
cell = torch.floor(X0 / dx).long()
key = (cell[:, 0] * (n_grid + 8) + cell[:, 1]) * (n_grid + 8) + cell[:, 2]
_, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
VOL = dx ** 3 / cnt[inv].float()
MASS = float(cfg["density"]) * VOL
MSUM = float(MASS.sum())
NORM = 2.0 * h * h / MSUM                                    # 목적 정규화 (질량가중 평균 이동² 크기)
FLOOR = [b for b in cfg.get("boundary_conditions", []) if b["type"] == "surface_collider"]
ZF = float(FLOOR[0]["point"][2]) if FLOOR else None

# ---------------------------------------------------------------- 가우시안 (렌더·측정 대상)
sys.path.append(f"{W}/i-physgaussian"); sys.path.append(f"{W}/i-physgaussian/gaussian-splatting")
_cwd = os.getcwd(); os.chdir(f"{W}/i-physgaussian")
from scene.gaussian_model import GaussianModel                       # noqa: E402
from utils.transformation_utils import (transform2origin, shift2center111)  # noqa: E402
os.chdir(_cwd)
mp = f"{W}/pgmodel/{MODEL[a.shape]}"
gs = GaussianModel(3)
gs.load_ply(f"{mp}/point_cloud/iteration_30000/point_cloud.ply")
_op = gs.get_opacity.detach()[:, 0]
KIDX = torch.nonzero(_op > float(cfg.get("opacity_threshold", 0.02))).squeeze(1)
_TP, SO, MEAN = transform2origin(gs.get_xyz.detach()[KIDX], float(cfg.get("scale", 1.0)))
_TP = shift2center111(_TP)                                        # 가우시안의 시뮬 좌표 (회전 항등)
NEACH = N // NOBJ
# 물체마다 시뮬 좌표 -> 모델 좌표 (충돌 씬은 두 벌을 옮겨 놓았다)
SHIFT = torch.zeros(NOBJ, 3, device=dev)
if NOBJ > 1:
    note = cfg["repflow_note"]
    c = -torch.as_tensor(note["center_shift"], dtype=torch.float32, device=dev)
    for o in range(NOBJ):
        SHIFT[o] = X0[OBJ == o].mean(0) - torch.as_tensor(np.load(f"{W}/pgfill_{a.shape}.npy"),
                                                           device=dev).mean(0)
    del c


# 시뮬 입자 중 가우시안 찾기: PG 채우기는 경계 상자 밖 가우시안을 버리고 순서를 바꿀 수 있어
# (ficus 렌더가 망가졌다) 번호를 가정하지 않고 좌표로 짝을 맞춘다 (같은 점이면 거리 ~0)
_X0o = X0[:NEACH] - SHIFT[0]
_mi, _md = [], []
for _s in range(0, _TP.shape[0], 8192):
    _d, _j = torch.cdist(_TP[_s:_s + 8192].to(dev), _X0o, compute_mode="donot_use_mm_for_euclid_dist").min(1)
    _md.append(_d); _mi.append(_j)
_md, _mi = torch.cat(_md), torch.cat(_mi)
_ok = _md < 1e-5                        # 같은 점이면 정확히 0 (행렬곱 cdist 는 ~5e-4 오차라 쓰지 않는다)
KIDX = KIDX[_ok]
NG1 = KIDX.numel()
GI = torch.cat([_mi[_ok] + o * NEACH for o in range(NOBJ)])     # 전체 중 가우시안 번호
print(f"[가우시안 대응] 불투명도 통과 {_TP.shape[0]} 중 시뮬 입자와 일치 {NG1} (최대 거리 {float(_md[_ok].max()):.2e})",
      flush=True)
if a.dbg:
    _q = torch.quantile(_md[~_ok][:100000].float(), torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=dev)) if (~_ok).any() else None
    print(f"  안 맞는 가우시안 거리 분위수 (0/10/50/90/100%): {None if _q is None else [f'{v:.2e}' for v in _q.tolist()]}  "
          f"dx {dx:.4f}", flush=True)


def to_model(P, ob):
    return (P - SHIFT[ob] - 1.0) / SO + MEAN


L = float((X0[GI].max(0).values - X0[GI].min(0).values).norm())   # 고정 정규화 상수
print(f"[장면] {a.shape} 입자 {N} (물체 {NOBJ}, 가우시안 {GI.numel()})  h {h:.5f}  재질 "
      f"{cfg['material']}  질량합 {MSUM:.4f}  바닥 {ZF}  지름 L {L:.4f}", flush=True)

# ---------------------------------------------------------------- 렌더러 (rep_track2 와 같은 규약)
RENDER = bool(a.video)
if RENDER:
    import imageio.v2 as imageio
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    c6 = gs.get_covariance()[KIDX].detach()
    C0 = torch.zeros(NG1, 3, 3, device=dev)
    C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c6[:, 0], c6[:, 1], c6[:, 2]
    C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c6[:, 3], c6[:, 4], c6[:, 5]
    C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c6[:, 1], c6[:, 2], c6[:, 4]
    C0_ALL, SHS_ALL, OPA_ALL = C0, gs.get_features[KIDX].detach(), gs.get_opacity[KIDX].detach()
    C0 = C0.repeat(NOBJ, 1, 1)
    SHS = SHS_ALL.repeat(NOBJ, 1, 1)
    OPA = OPA_ALL.repeat(NOBJ, 1)
    cam = json.load(open(f"{mp}/cameras.json"))[0]
    Rw, pos = np.array(cam["rotation"]), np.array(cam["position"])
    W2C = np.linalg.inv(np.block([[Rw, pos[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
    fx_ = 2 * math.atan(cam["width"] / (2 * cam["fx"])); fy_ = 2 * math.atan(cam["height"] / (2 * cam["fy"]))
    wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
    pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fx_, fovY=fy_).transpose(0, 1).to(dev).float()
    RAST = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
        image_height=int(cam["height"]), image_width=int(cam["width"]),
        tanfovx=math.tan(fx_ * 0.5), tanfovy=math.tan(fy_ * 0.5),
        bg=torch.ones(3, device=dev, dtype=torch.float32), scale_modifier=1.0, viewmatrix=wv,
        projmatrix=(wv[None] @ pj[None])[0], sh_degree=3, campos=wv.inverse()[3, :3],
        prefiltered=False, debug=False))
    WR = imageio.get_writer(a.video, fps=30, codec="libx264", quality=8)

    _SMED = float(gs.get_scaling[KIDX].detach().median())

    def render(Pg, Fg, gidx=None, obj=None):
        """gidx: 그릴 가우시안의 모델 번호 순서 (KIDX 안 위치), obj: 각 점의 물체 번호."""
        sh = SHS if gidx is None else SHS_ALL[gidx]
        op = OPA if gidx is None else OPA_ALL[gidx]
        c0 = C0 if gidx is None else C0_ALL[gidx]
        Pm = to_model(Pg, OBJ[GI] if obj is None else obj)
        if a.iso > 0:
            cov = torch.eye(3, device=dev).expand(Pg.shape[0], 3, 3) * (a.iso * _SMED) ** 2
        else:
            cov = Fg @ c0 @ Fg.transpose(1, 2)
        c6_ = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2],
                           cov[:, 2, 2]], 1)
        with torch.no_grad():
            img = RAST(means3D=Pm, means2D=torch.zeros_like(Pm), shs=sh, colors_precomp=None,
                       opacities=op, scales=None, rotations=None, cov3D_precomp=c6_)[0]
        WR.append_data((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

if a.render_ref:                                               # 기준 궤적만 영상으로
    for t in range(a.frames + 1):
        x = torch.as_tensor(rd(files[t], "x"), dtype=torch.float32, device=dev)
        Fr = rd(files[t], "F")
        Fr = torch.eye(3, device=dev).expand(N, 3, 3) if Fr is None else \
            torch.as_tensor(Fr.reshape(-1, 3, 3), dtype=torch.float32, device=dev)
        render(x[GI], Fr[GI])
    WR.close()
    print(f"[기준 영상] {a.video}", flush=True)
    raise SystemExit(0)

if a.render_npz:                                               # 저장된 결과를 영상으로만
    Zr = np.load(a.render_npz)
    TR = torch.as_tensor(Zr["traj"].astype(np.float32), device=dev)         # [T, n, 3]
    if a.old_order:
        # 예전 판은 물체마다 시뮬 입자 앞 n0 개를 가우시안으로 가정해 저장했다 -> 그 입자들을 좌표로 모델에 짝짓는다
        n0 = TR.shape[1] // NOBJ
        GIo = torch.cat([torch.arange(n0, device=dev) + o * NEACH for o in range(NOBJ)])
        P0 = X0[GIo] - SHIFT[OBJ[GIo]]
        dd, jj = [], []
        for _s in range(0, P0.shape[0], 8192):
            _d, _j = torch.cdist(P0[_s:_s + 8192], _TP.to(dev), compute_mode="donot_use_mm_for_euclid_dist").min(1); dd.append(_d); jj.append(_j)
        dd, jj = torch.cat(dd), torch.cat(jj)
        keep = dd < 1e-5
        sel, gidx, ob = torch.nonzero(keep).squeeze(1), jj[keep], OBJ[GIo][keep]
        # jj 는 _TP(=KIDX 전체 순서) 안 위치. KIDX 는 일치 집합으로 줄었으니 원래 순서 배열로 다시 만든다
        _full = torch.nonzero(_op > float(cfg.get("opacity_threshold", 0.02))).squeeze(1)
        SHS_ALL, OPA_ALL = gs.get_features[_full].detach(), gs.get_opacity[_full].detach()
        _c6 = gs.get_covariance()[_full].detach()
        C0_ALL = torch.zeros(_full.numel(), 3, 3, device=dev)
        C0_ALL[:, 0, 0], C0_ALL[:, 0, 1], C0_ALL[:, 0, 2] = _c6[:, 0], _c6[:, 1], _c6[:, 2]
        C0_ALL[:, 1, 1], C0_ALL[:, 1, 2], C0_ALL[:, 2, 2] = _c6[:, 3], _c6[:, 4], _c6[:, 5]
        C0_ALL[:, 1, 0], C0_ALL[:, 2, 0], C0_ALL[:, 2, 1] = _c6[:, 1], _c6[:, 2], _c6[:, 4]
        print(f"[예전 순서] 저장 점 {GIo.numel()} 중 모델 가우시안과 일치 {int(keep.sum())}", flush=True)
        I3 = torch.eye(3, device=dev).expand(int(keep.sum()), 3, 3)
        render(X0[GIo][keep], I3, gidx, ob)
        for t in range(TR.shape[0]):
            render(TR[t][keep], I3, gidx, ob)
    else:
        I3 = torch.eye(3, device=dev).expand(TR.shape[1], 3, 3)
        render(X0[GI], I3)
        for t in range(TR.shape[0]):
            render(TR[t], I3)
    WR.close()
    print(f"[결과 영상] {a.video}", flush=True)
    raise SystemExit(0)

# ---------------------------------------------------------------- 표현 (물체마다)
reps, idx = [], []
for o in range(NOBJ):
    ii = torch.nonzero(OBJ == o).squeeze(1)
    Xo = X0[ii]
    surf = X0[_mi[_ok] + o * NEACH]                 # 이 물체의 표면 가우시안 (좌표 대응)
    if a.method == "ours":
        r = rm.Lattice(Xo, h=(2.0 / 100.0) * (2.0 * math.sqrt(2.0)) ** (1.0 / 3.0))
    elif a.method == "phystwin":
        r = rm.PhysTwin(Xo)
    elif a.method == "gaussim":
        r = rm.GausSim(Xo)
    elif a.method == "simplicits":
        r = rm.Simplicits(Xo, a.simp, center=Xo.mean(0))
    else:
        r = rm.VRGS(Xo, surf)
    reps.append(r); idx.append(ii)
REP = rm.Multi(reps, idx)
REB = REP.rebind_each_frame

# ---------------------------------------------------------------- 물리 상태
xn, vn = X0.clone(), V0.clone()
Fe = torch.eye(3, device=dev).expand(N, 3, 3).clone()          # 탄성 F
Fcum = torch.eye(3, device=dev).expand(N, 3, 3).clone()        # 정지 대비 전체 사상 (렌더·det)
Jprev = torch.eye(3, device=dev).expand(N, 3, 3).clone()       # 비재결합 방법의 직전 누적 야코비안
if pr.mat_name(cfg) == "watermelon":
    pr.cdmpm_reset(N, dev, torch.float32, float(cfg.get("alpha_0", -0.04)))
if a.method == "gaussim":
    OPT = torch.optim.Adam(REP.params(), lr=a.lr)


def contact_pairs(x):
    """물체 사이 가까운 짝 (프레임 시작에 한 번): 반지름 2dx 안의 다른 물체 최근접 1 개."""
    if NOBJ < 2:
        return None
    i0 = torch.nonzero(OBJ == 0).squeeze(1); i1 = torch.nonzero(OBJ == 1).squeeze(1)
    pa, pb = [], []
    for s in range(0, i0.numel(), 8192):
        d, j = torch.cdist(x[i0[s:s + 8192]], x[i1]).min(1)
        m = d < 2 * dx
        pa.append(i0[s:s + 8192][m]); pb.append(i1[j[m]])
    return torch.cat(pa), torch.cat(pb)


def energy(dy, J, X_, xtil, pairs, parts=False):
    x = X_ + dy
    Finc = J if REB else rm.mm3(J, JPI[0])
    Ftr = rm.mm3(Finc, Fe)
    psi, pl = pr.psi_of(Ftr, cfg, h)
    e_el = (VOL * psi).sum()
    d = x - xtil
    e_in = 0.5 / (h * h) * (MASS * (d * d).sum(1)).sum()
    e_g = -(MASS * (d * g).sum(1)).sum()
    e_f = torch.zeros((), device=dev)
    if ZF is not None:
        e_f = a.k_floor * 0.5 / (h * h) * (MASS * (ZF - x[:, 2]).clamp_min(0) ** 2).sum()
    e_c = torch.zeros((), device=dev)
    if pairs is not None and pairs[0].numel():
        dd = (x[pairs[0]] - x[pairs[1]]).norm(dim=1)
        e_c = a.k_contact * 0.5 / (h * h) * (MASS[pairs[0]] * (dx - dd).clamp_min(0) ** 2).sum()
    tot = e_in + e_el + e_g + e_f + e_c
    if parts:
        return tot, pl, Ftr, dict(inertia=float(e_in), elastic=float(e_el), gravity=float(e_g),
                                  floor=float(e_f), contact=float(e_c))
    return tot, pl, Ftr


# ---------------------------------------------------------------- 계량 곱
from torch.func import jvp, vjp                                     # noqa: E402
GNP = torch.compile(rm.gn_prod, dynamic=True)


def flat(ts):
    return torch.cat([q.reshape(-1) for q in ts])


TB = None
if a.tb != "none":
    from torch.utils.tensorboard import SummaryWriter
    _tb = os.path.join(f"{W}/tbrf", "ip_" + os.path.splitext(os.path.basename(a.out))[0])
    TB = SummaryWriter(_tb)
rows, PHYS, TRAJ, IPV, NBAD = [], [], [], [], []
EMDP = []
t0 = time.time()
if RENDER:
    render(X0[GI], Fcum[GI])
for t in range(1, a.frames + 1):
    if REB:
        REP.rebind(xn)
        X_ = xn
    else:
        X_ = X0
    xtil = xn + h * vn
    JPI = [None if REB else torch.linalg.inv(Jprev)]
    pairs = contact_pairs(xn)
    PL = REP.params()
    shapes, sizes = [q.shape for q in PL], [q.numel() for q in PL]
    METS = REP.metrics() if a.method != "gaussim" else None
    psizes = [sum(q.numel() for q in r.params()) for r in reps]
    if a.method == "gaussim" and REB:
        OPT = torch.optim.Adam(PL, lr=a.lr)
    if METS is not None:
        # λmax (거듭제곱법) 로 ε 척도 -- 블록 대각 (물체마다)
        def Hv(th, u):
            outs, s0 = [], 0
            for (fn, args), n_ in zip(METS, psizes):
                outs.append(GNP(fn, th[s0:s0 + n_], u[s0:s0 + n_], *args)); s0 += n_
            return torch.cat(outs)
        with torch.no_grad():
            th0 = flat([q.detach() for q in PL])
            u = torch.randn_like(th0)
            for _ in range(20):
                u = Hv(th0, u); lmax = float(u.norm()); u = u / max(lmax, 1e-30)
        eps = a.riem_eps * max(lmax, 1e-12)
    nit = a.iters0 if t == 1 else a.iters
    STP = [a.riem_lr / 2.0]
    for it_ in range(nit):
        for q in PL:
            q.grad = None
        dy, J = REP.yJ(X_)
        E, _, _ = energy(dy, J, X_, xtil, pairs)
        (E * NORM).backward()
        if METS is None:
            OPT.step()
            continue
        gk = flat([q.grad if q.grad is not None else torch.zeros_like(q) for q in PL]).detach()
        theta = flat([q.detach() for q in PL])
        with torch.no_grad():
            x = torch.zeros_like(gk); r = gk.clone(); pdir = r.clone(); rr = (r * r).sum()
            for _ in range(a.riem_cg):
                Gp = Hv(theta, pdir) + eps * pdir
                al = rr / (pdir * Gp).sum().clamp_min(1e-30)
                x += al * pdir; r -= al * Gp
                rr_new = (r * r).sum()
                if rr_new.sqrt() < 1e-4 * gk.norm():
                    break
                pdir = r + (rr_new / rr) * pdir; rr = rr_new
            # Armijo 되돌림 선탐색 (A 와 같다): 보폭 η 에서 시작해 목적이 충분히 줄 때까지 반으로
            def setp(vec):
                s0 = 0
                for q, n_ in zip(PL, sizes):
                    q.copy_(vec[s0:s0 + n_].reshape(q.shape)); s0 += n_
            # 적응 보폭: 직전 반복의 보폭을 두 배로 시도 (계량의 크기와 목적의 크기가 장면마다
            # 달라 고정 시작 보폭은 강체 이동조차 못 따라갔다 -- i-PG lego 의 ours·GS-Verse)
            f0 = float(E) * NORM; sl = float((gk * x).sum())
            stp = min(STP[0] * 2.0, a.riem_lr_max)
            for _bt in range(a.ls_max):
                setp(theta - stp * x)
                dyt, Jt = REP.yJ(X_)
                ft = float(energy(dyt, Jt, X_, xtil, pairs)[0]) * NORM
                if ft == ft and ft <= f0 - 1e-4 * stp * sl:
                    break
                stp *= 0.5
            else:
                setp(theta); stp = STP[0] * 0.25
            STP[0] = stp
            if a.dbg and (it_ < 10 or it_ % 50 == 0):
                print(f"    it {it_:3d} f0 {f0:.6e} ft {ft:.6e} 보폭 {stp:.3e} |g| {float(gk.norm()):.3e} "
                      f"|x| {float(x.norm()):.3e} g·x {sl:.3e} eps {eps:.3e} lmax {lmax:.3e}", flush=True)
    # ---- 프레임 마무리: 상태 갱신 (소성 사영), 측정
    with torch.no_grad():
        dy, J = REP.yJ(X_)
        E, pl, Ftr, parts = energy(dy, J, X_, xtil, pairs, parts=True)
        x1 = X_ + dy
        Fe = pr.plastic_step(Ftr, pl)
        if REB:
            Fcum = rm.mm3(J, Fcum)
        else:
            Fcum = J
            Jprev = J.clone()
        vn = (x1 - xn) / h
        xn = x1
        ref = torch.as_tensor(rd(files[t], "x"), dtype=torch.float32, device=dev)
        yg, rg = xn[GI], ref[GI]
        ok = torch.isfinite(rg).all(1)                      # 기준 시뮬이 격리(발산)한 입자는 뺀다
        rmse = float(((yg[ok] - rg[ok]) ** 2).sum(1).mean().sqrt()) / L
        NBAD.append(int((~ok).sum()))
        cdv = 0.0
        for P_, Q_ in ((yg[ok], rg[ok]), (rg[ok], yg[ok])):
            cdv += float(sum(torch.cdist(P_[s:s + 4096], Q_).min(1).values.sum()
                             for s in range(0, P_.shape[0], 4096)) / P_.shape[0]) * 0.5
        Jd = rm.det3(Fcum)
        rows.append((t, rmse, cdv / L, float("nan"), float(Jd.min()), float((Jd <= 0).float().mean())))
        com = (MASS[:, None] * xn).sum(0) / MSUM
        PHYS.append((t, float(0.5 * (MASS * (vn * vn).sum(1)).sum()), *(MASS[:, None] * vn).sum(0).tolist(),
                     *(MASS[:, None] * torch.cross(xn - com, vn, dim=-1)).sum(0).tolist(),
                     float((VOL * Jd).sum() / VOL.sum())))
        IPV.append((t, float(E), *parts.values()))
        TRAJ.append(yg.half().cpu().numpy())
        if t % 10 == 0 or t == a.frames:
            EMDP.append((t, yg[ok].float().cpu().numpy(), rg[ok].float().cpu().numpy()))
        if RENDER:
            render(yg, Fcum[GI])
        if TB is not None:
            TB.add_scalar("frame/RMSE_pct", 100 * rmse, t); TB.add_scalar("frame/CD_pct", 100 * cdv / L, t)
            TB.add_scalar("frame/detJ_min", rows[-1][4], t); TB.add_scalar("frame/inverted_pct", 100 * rows[-1][5], t)
            TB.add_scalar("frame/IP", float(E), t); TB.add_scalar("frame/KE", PHYS[-1][1], t)
            TB.flush()
    if t % 10 == 0 or t == 1:
        print(f"  [t={t:3d}] RMSE {100 * rmse:.3f}%  CD {100 * cdv / L:.3f}%  IP {float(E):.4e} "
              f"({', '.join(f'{k} {v:.2e}' for k, v in parts.items())})  det 최소 {rows[-1][4]:.3f} "
              f"(≤0 {100 * rows[-1][5]:.2f}%)  자유도 {REP.dof}  {time.time() - t0:.0f}s", flush=True)
if RENDER:
    WR.close()
R = np.array(rows)
print(f"[요약] {a.method}  RMSE {100 * R[:, 1].mean():.3f}%  CD {100 * R[:, 2].mean():.3f}%  "
      f"det 최소 {R[:, 4].min():.4f}  뒤집힘 최대 {100 * R[:, 5].max():.2f}%", flush=True)
np.savez_compressed(a.out, metrics=R, L=L, phys=np.array(PHYS), ip=np.array(IPV),
                    traj=np.stack(TRAJ), emd_t=np.array([q[0] for q in EMDP]),
                    emd_y=np.stack([q[1] for q in EMDP]), emd_tgt=np.stack([q[2] for q in EMDP]),
                    dof=REP.dof, ref_bad=np.array(NBAD))
print(f"[저장] {a.out}", flush=True)
