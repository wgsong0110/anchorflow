"""Spring-Gaus 로 소성·점성·파괴 **시연 영상**을 뽑는다 (3DGS 알파블렌딩 렌더).

i-PG 쪽(`exe/demo_vid_ipg.py`)과 **같은 세 씬**을 같은 구동 규약으로 만든다.

  소성  물체를 반으로 나눠 양쪽 절반을 반대 방향으로 당긴다. 중력 없이 순수 인장.
  점성  물체 위에 같은 물체를 떨어뜨려 **붙기를 기다린 뒤** 두 덩이를 반대로 당긴다.
  파괴  위에서 바닥으로 던진다 (초기 하강속도 -6).

그쪽 모델에 없는 것을 둘 더한다 (둘 다 우리 확장이고 코드가 여기 다 보인다).

  1) **운동학 손잡이** -- 주축 양 끝 반경 R 안의 앵커를 명령 속도로 끌고 간다.
     i-PG 의 하드 Dirichlet 손잡이와 같은 역할이다 (힘이 아니라 속도를 박는다).
  2) **접촉 스프링** -- 두 덩이가 닿으면 사이에 스프링을 새로 엮는다. 쉬는 길이를
     **닿은 순간의 거리**로 잡아 충격 없이 붙는다. 스프링-질량 모델은 초기 kNN
     으로 망이 고정돼 있어 이것 없이는 두 덩이가 영원히 합쳐지지 않는다.

물성(항복 변형률·파괴 임계)은 벤치와 같은 `exe/sg_materials.py` 를 쓴다.

  python exe/demo_vid_sg.py --scene plastic --shape lego
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

W = "/home/dkta/work"
SG = f"{W}/Spring-Gaus"

ap = argparse.ArgumentParser()
ap.add_argument("--scene", choices=["plastic", "viscous", "fracture"],
                required=True)
ap.add_argument("--shape", default="lego")
ap.add_argument("--exp", default="", help="피팅 결과 (기본: 최신 af*_<형상>*)")
ap.add_argument("--out", default=f"{W}/demo")
ap.add_argument("--frames", type=int, default=0)
ap.add_argument("--n_step", type=int, default=0, help="프레임당 서브스텝")
ap.add_argument("--acc", type=float, default=0.0,
                help="양쪽에 주는 가속도 (0 이면 소성 80, 점성 20 -- i-PG 와 같은 값)")
ap.add_argument("--force_frames", type=int, default=0,
                help="힘을 주는 프레임 수 (0 이면 소성 20, 점성 40)")
ap.add_argument("--rupture", type=float, default=0.5,
                help="소성: 처음 길이 대비 이만큼 늘어난 스프링은 끊긴다 (연성 파단)")
ap.add_argument("--hold", type=int, default=40, help="점성: 붙기를 기다릴 프레임")
ap.add_argument("--gap", type=float, default=0.15, help="점성: 두 물체 간격")
ap.add_argument("--contact", type=float, default=0.0,
                help="접촉 스프링이 생기는 거리 (0 이면 앵커 간격의 1.2 배)")
ap.add_argument("--eps_break", type=float, default=0.10)
ap.add_argument("--v0", type=float, default=-6.0, help="파괴: 초기 하강속도")
ap.add_argument("--cam", type=int, default=-1,
                help="평가 카메라 번호 (기본: 당기는 씬은 4 -- 옆에서 봐서 당기는 "
                     "축이 화면 가로로 보인다, 파괴는 0)")
ap.add_argument("--cfg_name", default="",
                help="그쪽 설정 이름 (기본: <형상>_g -- 중력을 z 로 고쳐 다시 피팅한 것)")
# 두 덩이를 쌓으면 장면이 두 배로 높아지는데 그쪽 카메라는 물체에 바싹 붙어
# 있어 위 덩이가 화면 밖(심지어 카메라 뒤)으로 나가고, 그러면 아래 덩이까지
# 포함해 **전체가 백지로** 렌더된다 (실측: 간격 2.0 부터 흐려지고 3.68 에서 백지).
# 그래서 카메라를 뒤로 빼고 위로 올린다.
ap.add_argument("--cam_scale", type=float, default=0.0,
                help="카메라를 원점에서 몇 배 멀리 (0 이면 점성 2.2, 나머지 1.0)")
ap.add_argument("--cam_up", type=float, default=-999,
                help="카메라를 수직으로 얼마나 올리는가 (기본: 쌓은 높이의 절반)")
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()

MATOF = {"plastic": "elastoplastic", "viscous": "viscoplastic",
         "fracture": "fracture"}
CAMI = a.cam if a.cam >= 0 else (0 if a.scene == "fracture" else 4)
FRAMES = a.frames or {"plastic": 90, "viscous": 140, "fracture": 60}[a.scene]
OD = f"{a.out}/sg_{a.shape}_{a.scene}"
os.makedirs(a.out, exist_ok=True)

# ------------------------------------------------------------- 그쪽 코드 적재
# 임포트 경로는 bench_sg.py 와 **똑같이** 잡는다 (그쪽 train.py 가
# config_parser 와 get_simulator 를 함께 내보낸다)
# 중력을 고친 재피팅(af4_*) 을 쓴다
cs = sorted(glob.glob(f"{SG}/exp/af4_{a.shape}_*/checkpoints_dynamic/"
                      f"checkpoint/dy_n_step.json"))
if a.exp:
    cs = sorted(glob.glob(f"{a.exp}/checkpoints_dynamic/checkpoint/"
                          f"dy_n_step.json")) or cs
if not cs:
    raise SystemExit(f"[중단] {a.shape} 피팅 결과가 없다 ({SG}/exp)")
ck = os.path.join(os.path.dirname(cs[-1]), "Spring_Mass.pth.tar")
exp = os.path.dirname(os.path.dirname(os.path.dirname(cs[-1])))
if not os.path.exists(ck):
    raise SystemExit(f"[중단] 가중치가 없다: {ck}")
os.chdir(SG)
sys.path.insert(0, SG)
CFGN = a.cfg_name or f"{a.shape}_g"
sys.argv = ["demo.py", "--cfg", f"config/anchorflow/{CFGN}.yaml",
            "--exp_id", f"demo_{a.shape}", "--dy_reload", ck, "-g", "0"]
from train import config_parser, get_simulator                   # noqa: E402
from lib.utils.config import get_config_merge_default            # noqa: E402
from lib.models.gaus import Scene, render                       # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_materials                                              # noqa: E402

arg = config_parser()
cfg = get_config_merge_default(config_file=arg.cfg, arg=arg)
os.makedirs(f"{SG}/exp/demo_{a.shape}", exist_ok=True)
scene = Scene(cfg, f"demo_{a.shape}", shuffle=False, load_static=False)
with open(f"{cfg.CHECKPOINTS_ROOT}/init_velocity_"
          f"{cfg.VELOCITY.ITERATIONS}.json") as f:
    load_velocity = json.load(f)
sim, gaussians = get_simulator(arg, cfg, scene, cfg_stage=cfg.DYNAMIC,
                               init_velocity=load_velocity, load_g=None)
sim.eval()

# 그쪽 정적 단계는 가우시안을 **고정 크기 CONST_SCALE(0.003)** 로 학습한다
# (train.py: GaussianModel_isotropic(const_scale=cfg.STATIC.CONST_SCALE)). 그런데
# get_simulator 는 인자 없이 만들어 학습 안 된 _scaling(중앙값 0.035, 12 배)으로
# 그린다 -- 그래서 모든 렌더가 녹은 덩어리처럼 나왔다 (실측). 학습 때와 같게 맞춘다.
gaussians.const_scale = float(cfg.STATIC.CONST_SCALE)
# 중력: 그쪽 시뮬레이터는 cfg.DYNAMIC.G 를 읽는데 기본값이 [0,-9.8,0] (y 가 위)
# 이고, 우리 장면은 z 가 위라 g_f=[0,0,1] 과 곱해져 **유효 중력이 0** 이었다
# (우리 설정은 MODEL.G 에 넣어 먹히지 않았다). 시연에서는 z 중력을 직접 넣는다.
print(f"[중력] 피팅 값 g={sim.g.tolist()} g_f={sim.g_f.tolist()} -> 유효 "
      f"{(sim.g * sim.g_f).tolist()}", flush=True)
sim.g = torch.tensor([0.0, 0.0, -9.8], dtype=torch.float32)

# 렌더는 **정적 단계 가우시안**으로 한다. 동역학 체크포인트의 가우시안은 그쪽
# 동역학 단계가 색·불투명도까지 고쳐 반점이 많다(실측 비교: GT 대비 정적이 확연히
# 깨끗하다). 보간 계수는 그쪽 함수(set_all_particle)로 다시 잡는다.
from lib.models.gaus import GaussianModel_isotropic               # noqa: E402
_gs = GaussianModel_isotropic(const_scale=float(cfg.STATIC.CONST_SCALE))
scene.set_gaussians(_gs, f"{cfg.CHECKPOINTS_ROOT}/static_gaussians/point_cloud.ply")
with torch.no_grad():
    sim.set_all_particle(_gs.get_xyz.detach())
gaussians = _gs
print(f"[렌더] 정적 가우시안 {gaussians.get_xyz.shape[0]} 개", flush=True)

BG = torch.ones(3, dtype=torch.float32, device="cuda")       # 흰 배경
from lib.models.gaus.utils.graphics_utils import getWorld2View2   # noqa: E402
import copy as _copy                                              # noqa: E402


def pull_back(cam, scale, up):
    """카메라를 원점에서 `scale` 배 멀리, 수직으로 `up` 만큼 올린 사본."""
    if scale == 1.0 and up == 0.0:
        return cam
    c = _copy.copy(cam)
    tr = np.array([0.0, 0.0, 0.0]); tr[2] = up
    c.world_view_transform = torch.tensor(
        getWorld2View2(cam.R, cam.T, tr, scale)).transpose(0, 1).cuda()
    c.full_proj_transform = (c.world_view_transform.unsqueeze(0).bmm(
        cam.projection_matrix.unsqueeze(0))).squeeze(0)
    c.camera_center = c.world_view_transform.inverse()[3, :3]
    return c
print(f"[적재] {exp}\n       앵커 {sim.init_xyz.shape[0]}  가우시안 "
      f"{sim.init_xyz_all.shape[0]}  학습 n_step {int(sim.n_step)}", flush=True)


# --------------------------------------------------------- 1) 두 덩이로 복제
def duplicate(sim, gaussians, gap):
    """앵커·가우시안을 수직으로 옮겨 한 벌 더 얹는다 (위에서 떨어뜨릴 물체)."""
    N = int(sim.init_xyz.shape[0])
    ax = sim.ground_axis
    ext = float(sim.init_xyz[:, ax].max() - sim.init_xyz[:, ax].min())
    off = torch.zeros(3, device=sim.init_xyz.device)
    off[ax] = ext * (1.0 + gap)

    def cat2(t, add_off=False, add_idx=0):
        b = t.clone()
        if add_off:
            b = b + off
        if add_idx:
            b = b + add_idx
        return torch.cat([t, b], 0)

    M = int(sim.init_xyz_all.shape[0])
    with torch.no_grad():
        # 앵커 크기(N) 와 가우시안 크기(M) 를 **따로** 본다. 한 가지 크기로만
        # 거르면 가우시안·보간 텐서가 조용히 안 늘어나 한 덩이만 렌더된다
        for nm, is_x, idx in (("init_xyz", True, 0), ("init_v", False, 0),
                              ("init_xyz_all", True, 0),
                              ("knn_index", False, N), ("origin_len", False, 0),
                              ("intrp_index", False, N),
                              ("intrp_coef", False, 0),
                              ("global_k", False, 0), ("global_m", False, 0),
                              ("damp", False, 0), ("g_f", False, 0)):
            t = getattr(sim, nm, None)
            if not torch.is_tensor(t) or t.shape[0] not in (N, M):
                continue
            new = cat2(t.detach(), add_off=is_x, add_idx=idx)
            if isinstance(t, torch.nn.Parameter):
                setattr(sim, nm, torch.nn.Parameter(new, requires_grad=False))
            else:
                setattr(sim, nm, new)
        # 가우시안은 전부 두 벌로 (위치만 옮기고 색·불투명도·크기는 그대로)
        for nm in ("_xyz", "_color", "_scaling", "_rotation", "_opacity"):
            t = getattr(gaussians, nm)
            d = t.detach()
            new = torch.cat([d, d + (off if nm == "_xyz" else 0)], 0)
            setattr(gaussians, nm, torch.nn.Parameter(new,
                                                      requires_grad=False))
    sim.n_points = int(sim.init_xyz.shape[0])
    sim.n_all = int(sim.init_xyz_all.shape[0])
    print(f"[복제] 앵커 {N} -> {sim.n_points}, 가우시안 -> {sim.n_all}, "
          f"{ax} 축으로 {float(off[ax]):.4f} 띄웠다", flush=True)
    return N


NB = duplicate(sim, gaussians, a.gap) if a.scene == "viscous" else 0
CAM_S = a.cam_scale or (2.2 if a.scene == "viscous" else 1.0)
CAM_U = (a.cam_up if a.cam_up != -999 else
         (float(sim.init_xyz[:, 2].max() - sim.init_xyz[:, 2].min()) * 0.25
          if a.scene == "viscous" else 0.0))
print(f"[카메라] 거리 {CAM_S} 배, 수직 이동 {CAM_U:.3f}", flush=True)

# ------------------------------------------------------------- 2) 물성 확장
sg_materials.attach(sim, MATOF[a.scene], eps_break=a.eps_break)

X0 = sim.init_xyz.detach().clone()
L = float(torch.norm(X0.max(0).values - X0.min(0).values))
if a.n_step:
    sim.n_step = int(a.n_step)
NS = int(sim.n_step)

# ---------------------------------------------- 3) 힘 (절반/덩이 전체에 반대 방향)
# 속도를 박지 않고 **가속도**를 준다 (i-PG 의 particle_impulse 와 같은 규약).
# 소성: 수평 주축 기준 왼쪽 절반 전체에 -A, 오른쪽 절반 전체에 +A.
# 점성: 아래 덩이 전체에 -A, 위 덩이 전체에 +A (붙기를 기다린 뒤).
HD = None
if a.scene in ("plastic", "viscous"):
    g_ax = int(sim.ground_axis)
    Xc = (X0 - X0.mean(0)).cpu().numpy()
    Xc[:, g_ax] = 0.0
    w, V = np.linalg.eigh(Xc.T @ Xc / len(Xc))
    ax = V[:, int(np.argmax(w))]
    ax[g_ax] = 0.0
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    k = int(np.argmax(np.abs(ax)))
    e = np.zeros(3); e[k] = 1.0                 # i-PG 와 같은 좌표축 방향
    et = torch.as_tensor(e, dtype=torch.float32, device=X0.device)
    t = X0[:, k]
    if a.scene == "plastic":
        cut = t.median()
        mA, mB = t < cut, t >= cut
        hold = 0
    else:
        mA = torch.zeros_like(t, dtype=torch.bool); mA[:NB] = True
        mB = ~mA
        hold = a.hold
    ACC = a.acc or (80.0 if a.scene == "plastic" else 20.0)
    NF = a.force_frames or (20 if a.scene == "plastic" else 40)
    # 그쪽 장면 단위가 i-PG 시뮬 공간보다 크다 (lego 지름 SG 6.2 / i-PG 1.34).
    # 같은 **상대** 운동이 되도록 가속도를 길이 비로 맞춘다.
    SCL = L / 1.3418
    HD = dict(mA=mA, mB=mB, aA=-et * ACC * SCL, aB=et * ACC * SCL,
              f0=hold, f1=hold + NF)
    print(f"[힘] 축 {'xyz'[k]}  A {int(mA.sum())} / B {int(mB.sum())} 앵커  가속도 "
          f"{ACC} x 길이비 {SCL:.2f}  {hold}~{hold + NF} 프레임", flush=True)

# --------------------------------------------- 4) 접촉 스프링 (두 덩이 붙이기)
CT = None
if a.scene == "viscous":
    with torch.no_grad():
        d0 = float(sim.origin_len.mean())
        rc = a.contact or 1.2 * d0
        kc = float((10.0 ** sim.global_k).mean() / max(d0, 1e-9))
    CT = dict(rc=rc, kc=kc, pairs=None, l0=None, every=20)
    print(f"[접촉] 거리 {rc:.4f} 안이면 스프링을 엮는다 (강성 {kc:.3e}, "
          f"쉬는 길이는 닿은 순간 거리)", flush=True)


# ------------------------------------------------------------- 5) step 감싸기
orig_step = sim.step
state = dict(frame=0, sub=0, hxA=None, hxB=None)


def step(self, xyz, v, K, m, rebound_k, fric_k, damp, dt):
    global CT
    if CT is not None and CT["pairs"] is None or (
            CT is not None and state["sub"] % CT["every"] == 0):
        with torch.no_grad():
            # 아래 덩이 - 위 덩이 사이에서 가까운 짝을 찾는다 (한쪽 kNN 1 개)
            A, B = xyz[:NB], xyz[NB:]
            d = torch.cdist(A, B)
            best, j = d.min(1)
            i = torch.where(best < CT["rc"])[0]
            if i.numel():
                pr = torch.stack([i, j[i] + NB], 1)
                l0 = best[i]
                if CT["pairs"] is None:
                    CT["pairs"], CT["l0"] = pr, l0
                else:
                    key_old = CT["pairs"][:, 0] * self.n_points + CT["pairs"][:, 1]
                    key_new = pr[:, 0] * self.n_points + pr[:, 1]
                    keep = ~torch.isin(key_new, key_old)
                    if keep.any():
                        CT["pairs"] = torch.cat([CT["pairs"], pr[keep]], 0)
                        CT["l0"] = torch.cat([CT["l0"], l0[keep]], 0)
    xyz, v = orig_step(xyz=xyz, v=v, K=K * self._k_mask, m=m,
                       rebound_k=rebound_k, fric_k=fric_k, damp=damp, dt=dt)
    with torch.no_grad():
        # (a) 접촉 스프링 힘
        if CT is not None and CT["pairs"] is not None:
            i, j = CT["pairs"][:, 0], CT["pairs"][:, 1]
            dv = xyz[j] - xyz[i]
            dist = torch.norm(dv, dim=1, keepdim=True)
            u = dv / (dist + 1e-12)
            f = CT["kc"] * (dist.squeeze(1) - CT["l0"]).unsqueeze(1) * u
            mm = m if m.dim() == 2 else m.unsqueeze(1)
            acc = torch.zeros_like(v)
            acc.index_add_(0, i, f)
            acc.index_add_(0, j, -f)
            v = v + acc * dt / mm
            xyz = xyz + acc * dt * dt / mm
        # (b) 소성 상태 갱신 (sg_materials 와 같은 규약)
        cur = torch.norm(xyz[self.knn_index] - xyz.unsqueeze(1), dim=2)
        st = (cur - self.origin_len) / (self.origin_len + self.eps)
        cfg = self._af_cfg
        if cfg["kind"] == "break":
            self._k_mask *= (st.abs() <= cfg["eps_break"]).to(st.dtype)
        else:
            over = (st.abs() - cfg["eps_y"]).clamp_min(0.0) * torch.sign(st)
            rate = 1.0 if cfg["kind"] == "plastic" else dt / (dt + cfg["tau"])
            self.origin_len += rate * over * self.origin_len
        # (c) 연성 파단 -- 처음 길이 대비 rupture 넘게 늘어난 스프링은 끊는다.
        #     소성은 쉬는 길이가 따라가 변형률이 항복값에 머무르므로, 끊김은
        #     **누적 신장**(처음 길이 기준)으로 판정해야 한다.
        if cfg["kind"] == "plastic" and a.rupture > 0:
            tot = cur / (self._l0_ref + self.eps) - 1.0
            self._k_mask *= (tot <= a.rupture).to(tot.dtype)
        # (d) 힘 -- 정해진 프레임 동안 두 무리에 반대 방향 가속도
        if HD is not None and HD["f0"] <= state["frame"] < HD["f1"]:
            v[HD["mA"]] = v[HD["mA"]] + HD["aA"] * dt
            v[HD["mB"]] = v[HD["mB"]] + HD["aB"] * dt
    state["sub"] += 1
    return xyz, v


import types                                                     # noqa: E402
sim.step = types.MethodType(step, sim)

# ------------------------------------------------------------------ 6) 롤아웃
shutil.rmtree(OD, ignore_errors=True)
os.makedirs(OD, exist_ok=True)
import imageio.v2 as imageio                                     # noqa: E402
from tqdm import tqdm                                            # noqa: E402

with torch.no_grad():
    if a.scene == "fracture":
        sim.init_v = sim.init_v.detach().clone()
        sim.init_v[:, 2] = a.v0
        print(f"[초기속도] z = {a.v0}", flush=True)
    if a.scene == "plastic":
        sim.g_f = torch.zeros_like(sim.g_f) if torch.is_tensor(
            getattr(sim, "g_f", None)) else sim.g_f
        sim.g = torch.zeros_like(sim.g)
        print("[중력] 꺼서 순수 인장으로 본다", flush=True)
    xyz = sim.init_xyz.detach().clone()
    v = sim.init_v.detach().clone()
    xyz_all = torch.sum(xyz[sim.intrp_index] * sim.intrp_coef.unsqueeze(-1),
                        dim=1)
    gaussians._xyz = xyz_all
    for f in tqdm(range(FRAMES), desc="프레임"):
        state["frame"] = f
        cam = pull_back(scene.getEvalCameras(0, CAMI), CAM_S, CAM_U)
        img = render(cam, gaussians, BG, override_color=gaussians.get_color,
                     debug=False, compute_cov3D_python=False,
                     convert_SHs_python=False)["render"]
        imageio.imwrite(f"{OD}/{f:04d}.png",
                        (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                         * 255).astype(np.uint8))
        xyz_all, xyz, v, _ = sim(xyz_all, xyz, v, f + 1)
        gaussians._xyz = xyz_all
        if (f + 1) % 10 == 0:
            brk = float(1.0 - sim._k_mask.mean())
            dl0 = float((sim.origin_len / sim._l0_ref - 1).abs().max())
            npr = 0 if CT is None or CT["pairs"] is None else len(CT["pairs"])
            zc = [float(xyz[:NB, 2].mean()), float(xyz[NB:, 2].mean())] if NB \
                else [float(xyz[:, 2].mean())]
            print(f"  [{f + 1}] 끊긴 {100 * brk:.2f}%  쉬는길이 변화 "
                  f"{100 * dl0:.2f}%  접촉 스프링 {npr}  덩이 높이 "
                  f"{[round(q, 3) for q in zc]}", flush=True)

n = len(glob.glob(f"{OD}/*.png"))
mp4 = f"{a.out}/sg_{a.shape}_{a.scene}.mp4"
subprocess.run(["python", "-u", f"{W}/anchorflow/exe/pngs2mp4.py",
                "--dir", OD, "--out", mp4, "--fps", str(a.fps)],
               cwd=f"{W}/anchorflow")
print(f"[영상] {mp4}  ({n} 프레임)", flush=True)
print("DEMO_DONE", flush=True)
