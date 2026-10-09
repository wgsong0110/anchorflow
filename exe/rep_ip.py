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
ap.add_argument("--substeps", type=int, default=1, help="프레임당 서브스텝 (증분 포텐셜을 dt = frame_dt / N 으로 N 번)")
ap.add_argument("--stab_out", default="", help="안정성 실험: 프레임별 측정·부분표본 npz")
ap.add_argument("--no_ref", action="store_true", help="기준 궤적 없이 (--sim 은 0 프레임만 쓴다)")
ap.add_argument("--ckpt_every", type=int, default=5, help="이어 돌리기 체크포인트 간격 (프레임). <out>.ckpt.pt")
ap.add_argument("--no_resume", action="store_true", help="체크포인트가 있어도 처음부터")
ap.add_argument("--own", action="store_true",
                help="탄성 항과 계량을 각 방법 고유 탄성 모델로: ours·GausSim·Simplicits 입자 고정공회전(det<0 정의), "
                     "GS-Verse StVK 막(두께 = 부피/겉넓이), PhysTwin 스프링(Y = E V/ΣL0). 소성 없음. E·ν 는 시뮬 설정")
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
assert a.no_ref or len(files) > a.frames, f"기준 프레임 부족: {len(files)}"


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
h = float(cfg["frame_dt"]) / a.substeps
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


# 가우시안 ↔ 시뮬 입자: 가우시안마다 가장 가까운 시뮬 입자(정확한 거리)에 붙이고 상대 위치를 그 입자의 F 로 옮긴다
#   x_g = x_p + F_p (X_g - X_p).  lego·mic 는 채움 캐시 앞부분이 가우시안 그대로라 상대 위치 0 (같은 번호),
#   ficus 는 캐시가 가우시안을 반 칸 안에서 옮겨 두어 (중앙값 6e-4) 번호 대응이 깨졌었다.
_X0o = X0[:NEACH] - SHIFT[0]
_mi, _md = [], []
for _s in range(0, _TP.shape[0], 8192):
    _P = _TP[_s:_s + 8192].to(dev)                                 # 행렬곱 cdist 로 후보 8 개 -> 직접 거리
    _c = torch.cdist(_P, _X0o).topk(8, largest=False).indices
    _dd = (_P[:, None] - _X0o[_c]).norm(dim=-1)
    _d, _k = _dd.min(1); _j = _c.gather(1, _k[:, None]).squeeze(1)
    _md.append(_d); _mi.append(_j)
_md, _mi = torch.cat(_md), torch.cat(_mi)
NG1 = KIDX.numel()
GI = torch.cat([_mi + o * NEACH for o in range(NOBJ)])          # 각 가우시안이 붙은 입자 번호 (전체 배열)
OFFG = (_TP.to(dev) - _X0o[_mi]).repeat(NOBJ, 1)                # 정지 상대 위치
print(f"[가우시안 대응] {NG1} 개, 붙은 입자까지 거리 중앙 {float(_md.median()):.2e} 최대 {float(_md.max()):.2e}",
      flush=True)


def gpos(xall, Fall):
    """가우시안 위치 = 붙은 입자 위치 + 그 입자 F · 정지 상대 위치."""
    return xall[GI] + (Fall[GI] @ OFFG[..., None]).squeeze(-1)


# 렌더용: 모든 물체에 **같은** 이동을 빼서 시뮬 배치(서로 떨어진 두 물체)를 그대로 둔다.
#   예전에는 물체마다 SHIFT[o] 를 빼서 두 물체가 처음부터 모델 원점에 겹쳐 그려졌다 (시뮬은 1.0 떨어져 있었다).
SHIFT_R = SHIFT.mean(0)


def to_model(P, ob):
    return (P - SHIFT_R - 1.0) / SO + MEAN


_G0 = X0[GI] + OFFG
L = float((_G0.max(0).values - _G0.min(0).values).norm())   # 고정 정규화 상수
print(f"[장면] {a.shape} 입자 {N} (물체 {NOBJ}, 가우시안 {GI.numel()})  h {h:.5f}  재질 "
      f"{cfg['material']}  질량합 {MSUM:.4f}  바닥 {ZF}  지름 L {L:.4f}", flush=True)

# ---------------------------------------------------------------- 렌더러 (rep_track2 와 같은 규약)
RENDER = bool(a.video)
FRAME_DIR, FRAME_IX = [None], [0]
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
    if NOBJ > 1:
        # 두 물체가 다 들어오게 카메라를 모델 중심에서 바라보는 방향 그대로 뒤로 뺀다 (배율 = 전체 폭 / 한 물체 폭)
        _span1 = float((X0[OBJ == 0].max(0).values - X0[OBJ == 0].min(0).values).max())
        _spanA = float((X0.max(0).values - X0.min(0).values).max())
        _k = 1.1 * _spanA / _span1
        _ctr = MEAN.detach().cpu().numpy().astype(np.float64)
        pos = _ctr + _k * (pos - _ctr)
        print(f"[카메라] 두 물체가 들어오게 {_k:.2f} 배 뒤로", flush=True)
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
    WR = imageio.get_writer(a.video, fps=30, codec="libx264", quality=8) if (a.render_ref or a.render_npz) else None

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
        im = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        if FRAME_DIR[0] is not None:                          # 본 실행: 프레임을 PNG 로 (이어 돌리기에도 영상이 이어진다)
            imageio.imwrite(f"{FRAME_DIR[0]}/{FRAME_IX[0]:04d}.png", im); FRAME_IX[0] += 1
        else:
            WR.append_data(im)

if a.render_ref:                                               # 기준 궤적만 영상으로
    for t in range(a.frames + 1):
        x = torch.as_tensor(rd(files[t], "x"), dtype=torch.float32, device=dev)
        Fr = rd(files[t], "F")
        Fr = torch.eye(3, device=dev).expand(N, 3, 3) if Fr is None else \
            torch.as_tensor(Fr.reshape(-1, 3, 3), dtype=torch.float32, device=dev)
        render(gpos(x, Fr), Fr[GI])
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
            _P = P0[_s:_s + 8192]; _c = torch.cdist(_P, _TP.to(dev)).topk(8, largest=False).indices
            _dd = (_P[:, None] - _TP.to(dev)[_c]).norm(dim=-1); _d, _k = _dd.min(1)
            dd.append(_d); jj.append(_c.gather(1, _k[:, None]).squeeze(1))
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
        render(_G0, I3)
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
    surf = _TP.to(dev) + SHIFT[o]                   # 이 물체의 표면 가우시안 (정지 위치)
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
if a.own:
    for r, ii in zip(reps, idx):
        Vo = float(VOL[ii].sum())
        if a.method == "phystwin":
            r.Ystf = float(cfg["E"]) * Vo / float(r.L0.sum())          # 에너지 밀도가 E 와 맞게
        elif a.method == "vrgs":
            r.thick = Vo / float(r.A0.sum())                            # 막 두께 = 부피 / 겉넓이
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

# ---------------------------------------------------------------- 이어 돌리기
CKPT = a.out + ".ckpt.pt"
rows, PHYS, TRAJ, IPV, NBAD = [], [], [], [], []
EMDP = []
T_START = 1
if os.path.exists(CKPT) and not a.no_resume:
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    xn, vn, Fe, Fcum, Jprev = ck["xn"], ck["vn"], ck["Fe"], ck["Fcum"], ck["Jprev"]
    reps = ck["reps"]; REP = rm.Multi(reps, idx)
    if ck["JP"] is not None:
        pr._JP[0] = ck["JP"]
    if a.method == "gaussim":
        OPT = torch.optim.Adam(REP.params(), lr=a.lr); OPT.load_state_dict(ck["opt"])
    rows, PHYS, TRAJ, IPV, NBAD, EMDP = ck["rows"], ck["PHYS"], ck["TRAJ"], ck["IPV"], ck["NBAD"], ck["EMDP"]
    T_START = ck["t"] + 1
    print(f"[이어 돌리기] {CKPT} 프레임 {ck['t']} 부터", flush=True)


def save_ckpt(t):
    torch.save(dict(t=t, xn=xn, vn=vn, Fe=Fe, Fcum=Fcum, Jprev=Jprev, reps=reps,
                    JP=pr._JP[0], opt=OPT.state_dict() if a.method == "gaussim" else None,
                    rows=rows, PHYS=PHYS, TRAJ=TRAJ, IPV=IPV, NBAD=NBAD, EMDP=EMDP), CKPT + ".tmp")
    os.replace(CKPT + ".tmp", CKPT)                           # 쓰다 죽어도 이전 체크포인트는 남는다


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
    if a.own:
        pl = None
        e_el = own_elastic(Ftr)
    else:
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


MU_E, LA_E = pr.lame(cfg["E"], cfg["nu"])
_sla_m = math.sqrt(float(cfg["nu"]) / (1 - 2 * float(cfg["nu"])))   # μ 를 1 로 둔 계량의 √(λ/2)


def own_elastic(Ftr):
    """각 방법 고유 탄성 에너지 (소성 없음)."""
    if a.method in ("ours", "gaussim", "simplicits"):
        return (VOL * rm.fcr_psi(Ftr, MU_E, LA_E)).sum()
    e = torch.zeros((), device=dev)
    for r, ii in zip(reps, idx):
        if a.method == "phystwin":
            Bn = r.B + r.m
            Ln = (Bn[r.rel] - Bn[:, None]).norm(dim=-1)
            e = e + 0.5 * r.Ystf * (r.L0 * (Ln / r.L0 - 1.0) ** 2).sum()
        else:                                                         # vrgs
            e = e + r.thick * rm.stvk_membrane(r.v, r.f, r.D0i, r.A0, MU_E, LA_E)
    return e


# ---------------------------------------------------------------- 계량 곱
from torch.func import jvp, vjp                                     # noqa: E402
GNP = torch.compile(rm.gn_prod, dynamic=True)


def flat(ts):
    return torch.cat([q.reshape(-1) for q in ts])


TB = None
if a.tb != "none":
    from torch.utils.tensorboard import SummaryWriter
    _tb = a.tb if a.tb != "auto" else os.path.join(f"{W}/tbrf", "ip_" + os.path.splitext(os.path.basename(a.out))[0])
    TB = SummaryWriter(_tb, purge_step=T_START)
t0 = time.time()
if a.stab_out:
    from anchorflow import stabstat as _ss
    _SUB = torch.as_tensor(_ss.subset(N), device=dev)
    _MASSN = MASS.double().cpu().numpy(); _FZ = ZF if ZF is not None else 0.0
    STROWS = [_ss.frame_stats(X0.cpu().numpy(), V0.cpu().numpy(), None, _MASSN, g.tolist(), _FZ)]
    STX = [X0[_SUB].cpu().numpy()]; STT = []
if RENDER:
    FRAME_DIR[0] = a.video + ".frames"; os.makedirs(FRAME_DIR[0], exist_ok=True)
    FRAME_IX[0] = T_START                                     # 프레임 0 = 정지, t 번째 = t
    if T_START == 1:
        FRAME_IX[0] = 0
        render(gpos(X0, Fcum), Fcum[GI])
for t in range(T_START, a.frames + 1):
  _tf = time.time()
  for ks in range(a.substeps):
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
        if a.own:
            if a.method == "phystwin":
                METS = [(rm.res_spring, (r.B, r.rel, r.L0)) for r in reps]
            elif a.method == "vrgs":
                METS = [(rm.res_tri_stvk, (r.f, r.D0i, r.A0, 1.0, _sla_m)) for r in reps]
            else:
                METS = "particle"                                         # 매 반복 극분해 R 을 다시 구한다

        def particle_mets():
            """입자 고정공회전 계량 (물체마다): 현재 매개(선형화 점)의 F 로 R 을 구해 순수 잔차 인자로."""
            with torch.no_grad():
                out = []
                for r, ii in zip(reps, idx):
                    Xo = X_[ii]
                    dyo, Jo = r.yJ(Xo, torch.arange(ii.numel(), device=dev))
                    Fp = Fe[ii] if REB else rm.mm3(JPI[0][ii], Fe[ii])
                    F0 = rm.mm3(Jo, Fp)
                    R0 = rm.polar_R(F0)
                    wv = (VOL[ii] / VOL[ii].sum()).sqrt()
                    if a.method == "ours":
                        out.append((rm.res_lattice_fcr, (r.u.numel(), r.rows, r.lam, r.dlam, r.r, r.dr, r.w, r.dw,
                                                         r.h, r.a, Fp, R0, wv, 1.0, _sla_m)))
                    elif a.method == "gaussim":
                        out.append((rm.res_gaussim_fcr, (r.p2.shape[0], r.lab, Fp, R0, wv, 1.0, _sla_m)))
                    else:
                        out.append((rm.res_simp_fcr, (r.W, r.dW, r.Xh, Fp, R0, wv, 1.0, _sla_m)))
            return out
        psizes = [sum(q.numel() for q in r.params()) for r in reps]
        if a.method == "gaussim" and REB and not a.own:
            OPT = torch.optim.Adam(PL, lr=a.lr)
        if METS is not None:
            # λmax (거듭제곱법) 로 ε 척도 -- 블록 대각 (물체마다)
            MC = [particle_mets() if METS == "particle" else METS]

            def Hv(th, u):
                outs, s0 = [], 0
                for (fn, args), n_ in zip(MC[0], psizes):
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
            if METS == "particle":
                MC[0] = particle_mets()                                   # 반복마다 R 을 다시
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
            Fe = Ftr if a.own else pr.plastic_step(Ftr, pl)
            if REB:
                Fcum = rm.mm3(J, Fcum)
            else:
                Fcum = J
                Jprev = J.clone()
            vn = (x1 - xn) / h
            xn = x1
            if ks < a.substeps - 1:
                continue
            if a.stab_out:
                torch.cuda.synchronize(); STT.append(time.time() - _tf)
                STROWS.append(_ss.frame_stats(xn.cpu().numpy(), vn.cpu().numpy(), Fe.cpu().numpy(), _MASSN, g.tolist(), _FZ))
                STX.append(xn[_SUB].cpu().numpy())
            ref = xn if a.no_ref else torch.as_tensor(rd(files[t], "x"), dtype=torch.float32, device=dev)
            Fr_ = None if a.no_ref else rd(files[t], "F")
            Fr_ = torch.eye(3, device=dev).expand(N, 3, 3) if Fr_ is None else \
                torch.as_tensor(Fr_.reshape(-1, 3, 3), dtype=torch.float32, device=dev)
            yg, rg = gpos(xn, Fcum), gpos(ref, Fr_)
            ok = torch.isfinite(rg).all(1)                      # 기준 시뮬이 격리(발산)한 입자는 뺀다
            rmse = float(((yg[ok] - rg[ok]) ** 2).sum(1).mean().sqrt()) / L
            NBAD.append(int((~ok).sum()))
            def _nn(P_, Q_, ch=4096, k=8):                          # 정확한 최근접 거리 (행렬곱 후보 + 직접 거리)
                out = []
                for s_ in range(0, P_.shape[0], ch):
                    Pi = P_[s_:s_ + ch]
                    j_ = torch.cdist(Pi, Q_).topk(min(k, Q_.shape[0]), largest=False).indices
                    out.append((Pi[:, None] - Q_[j_]).norm(dim=-1).min(1).values)
                return torch.cat(out)
            cdv = 0.5 * float(_nn(yg[ok], rg[ok]).mean() + _nn(rg[ok], yg[ok]).mean())
            Jd = rm.det3(Fcum)
            rows.append((t, rmse, cdv / L, float("nan"), float(Jd.min()), float((Jd <= 0).float().mean())))
            com = (MASS[:, None] * xn).sum(0) / MSUM
            PHYS.append((t, float(0.5 * (MASS * (vn * vn).sum(1)).sum()), *(MASS[:, None] * vn).sum(0).tolist(),
                         *(MASS[:, None] * torch.cross(xn - com, vn, dim=-1)).sum(0).tolist(),
                         float((VOL * Jd).sum() / VOL.sum())))
            IPV.append((t, float(E), *parts.values()))
            TRAJ.append(yg.float().cpu().numpy())
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
        if a.ckpt_every > 0 and t % a.ckpt_every == 0 and t < a.frames:
            save_ckpt(t)
if RENDER:
    import glob as _glob
    import shutil as _shutil
    WR = imageio.get_writer(a.video, fps=30, codec="libx264", quality=8)
    for fp_ in sorted(_glob.glob(f"{FRAME_DIR[0]}/*.png"))[:a.frames + 1]:
        WR.append_data(imageio.imread(fp_))
    WR.close()
if a.stab_out:
    _K = ("ke", "pe", "nan", "out", "vmax", "detneg", "detmin")
    np.savez_compressed(a.stab_out, sub=_SUB.cpu().numpy(), x=np.stack(STX).astype(np.float32), t=np.array(STT), keys=np.array(_K),
                        substeps=a.substeps, dt=h, stats=np.array([[r.get(k, np.nan) for k in _K] for r in STROWS]),
                        iters0=a.iters0, iters=a.iters)
    print(f"[안정성 저장] {a.stab_out}  프레임당 {np.mean(STT):.3f}s", flush=True)
R = np.array(rows)
print(f"[요약] {a.method}  RMSE {100 * R[:, 1].mean():.3f}%  CD {100 * R[:, 2].mean():.3f}%  "
      f"det 최소 {R[:, 4].min():.4f}  뒤집힘 최대 {100 * R[:, 5].max():.2f}%", flush=True)
np.savez_compressed(a.out, metrics=R, L=L, phys=np.array(PHYS), ip=np.array(IPV),
                    traj=np.stack(TRAJ), emd_t=np.array([q[0] for q in EMDP]),
                    emd_y=np.stack([q[1] for q in EMDP]), emd_tgt=np.stack([q[2] for q in EMDP]),
                    dof=REP.dof, ref_bad=np.array(NBAD))
print(f"[저장] {a.out}", flush=True)
if os.path.exists(CKPT):
    os.remove(CKPT)
if RENDER:
    _shutil.rmtree(FRAME_DIR[0], ignore_errors=True)
