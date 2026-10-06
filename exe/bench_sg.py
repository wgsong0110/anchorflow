"""Spring-Gaus 를 **같은 품질기준 규약**으로 잰다 (FPS · 잔차 · FID/FVD/KVD).

Spring-Gaus 는 스프링-질량이라 **탄성만** 있다 (소성·점소성·파괴 구성식이 없다
-- 그 칸은 미지원으로 적는다). 서브스텝 손잡이는 그쪽 시뮬레이터의
`n_step` 이다 (`dt = self.dt / self.n_step`).

규약은 PG 쪽과 같다:
  * 사다리로 최고정밀 궤적을 잡고, 그 대비 **한 프레임 오차 <= 0.5%** 인 가장
    작은 n_step 을 이분 탐색한다. 비교는 앵커 위치, 정규화는 **프레임 0 의
    지름**(고정 상수).
  * 시간은 시뮬레이터 전진만 잰다 (렌더 제외). 앵커 전진 + 가우시안 보간이
    모두 동역학 비용이라 둘 다 포함한다.
  * 잔차는 그쪽 **스프링 에너지 + 접촉 벌점**으로 증분 포텐셜을 세워 잰다
        E = Σ m/(2h²)|Δu − h v − h² g|² + Σ ½K(|Δx|−l0)² + E_bc
  * 시각 품질은 그쪽 렌더러로 참조(n_step 수렴)·대상(통과 n_step) 두 벌을 찍어
    FID/FVD/KVD.

  python exe/bench_sg.py --shape mic --run run5 --phase all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

W = "/home/dkta/work"
SG = f"{W}/Spring-Gaus"

ap = argparse.ArgumentParser()
ap.add_argument("--shape", required=True)
ap.add_argument("--material", default="elastic",
                choices=["elastic", "elastoplastic", "viscoplastic",
                         "fracture"])
ap.add_argument("--eps_break", type=float, default=0.10,
                help="파괴: 스프링이 끊기는 변형률 (유일한 자유 손잡이)")
ap.add_argument("--run", default="run5")
ap.add_argument("--exp", default="", help="피팅 결과 exp 디렉토리 (기본: 최신 af2_*)")
ap.add_argument("--frames", type=int, default=10, help="비교할 충돌 프레임 수")
ap.add_argument("--skip", type=int, default=24)
ap.add_argument("--vq_frames", type=int, default=34)
ap.add_argument("--tol", type=float, default=5e-3)
ap.add_argument("--tau", type=float, default=1e-3)
ap.add_argument("--s0", type=int, default=0, help="사다리 시작 n_step (기본: 학습값)")
ap.add_argument("--max_mul", type=int, default=8)
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--win", type=int, default=16)
ap.add_argument("--phase", default="all",
                choices=["search", "time", "resid", "vq", "all"])
a = ap.parse_args()

O = f"{W}/bench/{a.run}"
SJ = f"{O}/sg_{a.shape}_{a.material}.json"
os.makedirs(O, exist_ok=True)

# --- 그쪽 학습 결과 찾기 ----------------------------------------------
exp = a.exp
cs = sorted(glob.glob(f"{SG}/exp/af2_{a.shape}_*/checkpoints_dynamic/"
                      f"checkpoint/dy_n_step.json"))
if not cs:
    cs = sorted(glob.glob(f"{SG}/exp/af_{a.shape}_*/checkpoints_dynamic/"
                          f"checkpoint/dy_n_step.json"))
if exp:                                   # 디렉토리를 직접 준 경우
    cs = sorted(glob.glob(f"{exp}/checkpoints_dynamic/checkpoint/"
                          f"dy_n_step.json")) or cs
if not cs:
    raise SystemExit(f"[건너뜀] {a.shape} 의 피팅 결과가 없다 ({SG}/exp)")
# 그쪽 규약: dy_reload 의 두 단계 위에 'checkpoint/' 가 붙는다
#   <exp>/checkpoints_dynamic/checkpoint/Spring_Mass.pth.tar
DY = os.path.join(os.path.dirname(cs[-1]), "Spring_Mass.pth.tar")
exp = os.path.dirname(os.path.dirname(os.path.dirname(cs[-1])))
if not os.path.exists(DY):
    raise SystemExit(f"[건너뜀] 가중치가 없다: {DY}")
print(f"[피팅] {exp}\n       가중치 {DY}", flush=True)

os.chdir(SG)
sys.path.insert(0, SG)
sys.argv = ["test.py", "--cfg", f"config/anchorflow/{a.shape}.yaml",
            "--exp_id", f"bench_{a.shape}", "--dy_reload", DY, "-g", "0"]
from train import config_parser, get_simulator                 # noqa: E402
from lib.utils.config import get_config_merge_default          # noqa: E402
from lib.models.gaus import Scene, render                      # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import sg_materials                                            # noqa: E402

arg = config_parser()
cfg = get_config_merge_default(config_file=arg.cfg, arg=arg)
# Scene 이 cameras.json 을 exp_path 에 쓴다. 보통은 Recorder 가 만들어 주는데
# 여기서는 Recorder 를 안 쓰므로 직접 만든다.
os.makedirs(f"{SG}/exp/bench_{a.shape}", exist_ok=True)
scene = Scene(cfg, f"bench_{a.shape}", shuffle=False, load_static=False)
with open(f"{cfg.CHECKPOINTS_ROOT}/init_velocity_"
          f"{cfg.VELOCITY.ITERATIONS}.json") as f:
    load_velocity = json.load(f)
simulator, gaussians = get_simulator(arg, cfg, scene, cfg_stage=cfg.DYNAMIC,
                                     init_velocity=load_velocity, load_g=None)
simulator.eval()
# 그쪽에 없는 물성(소성·점소성·파괴)은 스프링 수준 확장으로 붙인다.
sg_materials.attach(simulator, a.material, eps_break=a.eps_break)
BG = torch.tensor(scene.dataset.bg, dtype=torch.float32, device="cuda")
S_TRAIN = int(simulator.n_step)
print(f"[시뮬] 앵커 {simulator.init_xyz.shape[0]}  가우시안 "
      f"{simulator.init_xyz_all.shape[0]}  학습 n_step {S_TRAIN}  "
      f"dt {simulator.dt}", flush=True)

X0 = simulator.init_xyz.detach().clone()
L = float(torch.norm(X0.max(0).values - X0.min(0).values))
NF = a.skip + a.frames


@torch.no_grad()
def rollout(s, frames=NF, render_dir="", timeit=False, keep_state=False):
    """n_step=s 로 frames 장 전진한다 -> (앵커 궤적 [T,N,3], 시뮬 시간)."""
    simulator.n_step = int(s)
    sg_materials.reset(simulator)        # 소성 상태는 롤아웃마다 초기화
    L0, MK = [], []
    xyz = simulator.init_xyz.detach().clone()
    v = simulator.init_v.detach().clone()
    xyz_all = torch.sum(xyz[simulator.intrp_index]
                        * simulator.intrp_coef.unsqueeze(-1), dim=1)
    gaussians._xyz = xyz_all
    if render_dir:
        shutil.rmtree(render_dir, ignore_errors=True)
        os.makedirs(render_dir, exist_ok=True)
        import imageio
    X, V, t_sim = [], [], 0.0
    for f in range(frames):
        X.append(xyz.detach().cpu().numpy().copy())
        V.append(v.detach().cpu().numpy().copy())
        if keep_state:
            L0.append(simulator.origin_len.detach().clone())
            MK.append(getattr(simulator, "_k_mask",
                              torch.ones_like(simulator.origin_len)).clone())
        if render_dir:
            cam = scene.getEvalCameras(0, a.cam)
            img = render(cam, gaussians, BG, override_color=gaussians.get_color,
                         debug=False, compute_cov3D_python=False,
                         convert_SHs_python=False)["render"]
            imageio.imwrite(f"{render_dir}/{f:04d}.png",
                            (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                             * 255).astype(np.uint8))
        if timeit:
            torch.cuda.synchronize()
            t0 = time.time()
        xyz_all, xyz, v, _ = simulator(xyz_all, xyz, v, f + 1)
        if timeit:
            torch.cuda.synchronize()
            t_sim += time.time() - t0
        gaussians._xyz = xyz_all
    if keep_state:
        return np.stack(X), np.stack(V), t_sim, L0, MK
    return np.stack(X), np.stack(V), t_sim


def rms_rel(A, B):
    return float(np.sqrt(((A - B) ** 2).sum(-1).mean()) / L)


d = json.load(open(SJ)) if os.path.exists(SJ) else {}

# --- 1) 사다리 + 이분 탐색 --------------------------------------------
if a.phase in ("search", "all"):
    s0 = a.s0 or S_TRAIN
    hist, s, Xc = [], s0, None
    Xp, _, _ = rollout(s)
    for _ in range(int(np.log2(a.max_mul)) + 1):
        s2 = 2 * s
        X2, _, _ = rollout(s2)
        ch = rms_rel(X2[-1], Xp[-1])
        hist.append((s2, 100 * ch))
        print(f"  [사다리] n_step {s} -> {s2}: 변화 {100 * ch:.4f}% "
              f"(기준 {100 * a.tau:.3f}%)", flush=True)
        Xp, s = X2, s2
        if ch <= a.tau:
            break
    Xc, s_conv = Xp, s
    print(f"[기준] n_step={s_conv} 최고정밀 궤적 확보", flush=True)
    lo, hi, best = max(1, s0 // 4), s_conv, None
    while lo < hi:
        mid = (lo + hi) // 2
        Xm, _, _ = rollout(mid)
        e1 = rms_rel(Xm[a.skip], Xc[a.skip])            # 충돌 첫 프레임
        eT = rms_rel(Xm[-1], Xc[-1])
        okay = e1 <= a.tol and np.isfinite(eT) and eT < 10 * a.tol * len(Xm)
        print(f"  [탐색] n_step {mid}: 한프레임 {100 * e1:.4f}% 누적 "
              f"{100 * eT:.4f}% -> {'통과' if okay else '미달'}", flush=True)
        if okay:
            best, hi = (mid, e1, eT), mid
        else:
            lo = mid + 1
    d.update(method="sg", shape=a.shape, material=a.material,
             n_particles=int(X0.shape[0]),
             n_gaussians=int(simulator.init_xyz_all.shape[0]), L=L,
             s_conv=int(s_conv), s_train=S_TRAIN, frames=a.frames,
             tol=a.tol, tau=a.tau, ladder=hist)
    if best is None:
        print("[결과] 합격 설정 없음 (미달)", flush=True)
        d["s"] = None
    else:
        d.update(s=int(best[0]), e1=float(best[1]), eT=float(best[2]))
        print(f"[탐색] n_step={best[0]}, 한프레임 {100 * best[1]:.4f}%",
              flush=True)
    json.dump(d, open(SJ, "w"), indent=1)

# --- 2) 시간 ---------------------------------------------------------
if a.phase in ("time", "all") and d.get("s"):
    s = int(d["s"])
    _, _, tL = rollout(s, frames=60, timeit=True)
    t_per = tL / 60.0
    d.update(ms_per_frame=1000 * t_per, ms_per_substep=1000 * t_per / s,
             fps=1.0 / t_per)
    json.dump(d, open(SJ, "w"), indent=1)
    print(f"[결과] sg {a.shape}: n_step={s}, {1.0 / t_per:.2f} FPS "
          f"({1000 * t_per:.1f} ms/프레임)", flush=True)

# --- 3) 물리 잔차 / 증분 포텐셜 ----------------------------------------
if a.phase in ("resid", "all"):
    s_ref = int(d.get("s_conv") or S_TRAIN)
    simulator.n_step = s_ref
    X, V, _, L0H, MKH = rollout(s_ref, frames=NF + 1, keep_state=True)
    h = float(simulator.dt)
    g = torch.as_tensor(cfg.MODEL.G, dtype=torch.float32, device="cuda")
    knn = simulator.knn_index
    l0 = simulator.origin_len
    with torch.no_grad():
        K = (10 ** simulator.global_k).reshape(1, 1).expand_as(l0) \
            if simulator.global_k.numel() == 1 else \
            (10 ** simulator.global_k).unsqueeze(1).expand_as(l0)
        m = (10 ** simulator.m).reshape(-1) if hasattr(simulator, "m") and \
            torch.is_tensor(simulator.m) else torch.full(
                (X.shape[1],), float(cfg.DATA.GLOBAL_M), device="cuda")
        if m.numel() == 1:
            m = m.expand(X.shape[1]).contiguous()
    k_bc = float(10 ** simulator.k_bc) / (float(l0.mean()) + 1e-8) \
        if getattr(simulator, "spring_bc", False) else 0.0
    ground = float(simulator.ground)
    gax = int(simulator.ground_axis)
    edge = float(getattr(simulator, "edge", 0.0))
    pw = float(getattr(simulator, "power", 0.0)) \
        if getattr(simulator, "unlinear_foce", False) else 0.0
    rows = []
    for i in range(1, X.shape[0]):
        xn = torch.as_tensor(X[i - 1], device="cuda")
        vn = torch.as_tensor(V[i - 1], device="cuda")
        xn1 = torch.as_tensor(X[i], device="cuda")
        du = (xn1 - xn).detach().requires_grad_(True)
        x2 = xn + du
        xtil = xn + h * vn + (h ** 2) * g
        e_in = (m * ((x2 - xtil) ** 2).sum(-1)).sum() / (2 * h * h)
        l0f = L0H[i]                      # 그 프레임의 쉬는 길이(소성 반영)
        mk = MKH[i]                       # 끊긴 스프링은 힘을 안 낸다
        dl = torch.norm(x2[knn] - x2.unsqueeze(1), dim=2) - l0f
        dl = torch.where(dl.abs() < edge, torch.zeros_like(dl), dl)
        e_sp = 0.5 * (K * mk * dl ** 2).sum()
        pen = (ground - x2[:, gax]).clamp_min(0.0)
        e_bc = (k_bc / (2.0 + pw)) * (pen ** (2.0 + pw)).sum() if k_bc else \
            torch.zeros((), device="cuda")
        E = e_in + e_sp + e_bc
        gx, = torch.autograd.grad(E, du)
        r = gx.norm(dim=-1) * (h * h) / m.clamp_min(1e-20) / L
        rq = torch.quantile(r.detach().float(),
                            torch.tensor([0.5, 0.95], device="cuda"))
        rows.append(dict(frame=i, E=float(E), e_in=float(e_in),
                         e_el=float(e_sp), e_bc=float(e_bc),
                         r_med=float(rq[0]), r_p95=float(rq[1])))
    rj = f"{O}/resid/sg_{a.shape}_{a.material}.json"
    os.makedirs(f"{O}/resid", exist_ok=True)
    out = dict(label=f"Spring-Gaus {a.shape} {a.material} reference "
                     f"(n_step={s_ref})", n_particles=int(X.shape[1]), L=L,
               ext=L, h=h, material=f"spring_mass/{a.material}",
               E_mean=float(np.mean([q["E"] for q in rows])),
               r_med_mean=float(np.mean([q["r_med"] for q in rows])),
               r_p95_mean=float(np.mean([q["r_p95"] for q in rows])),
               frames=rows)
    json.dump(out, open(rj, "w"), indent=1)
    print(f"[요약] 증분 포텐셜 평균 {out['E_mean']:.4e}  물리 잔차 중앙값 평균 "
          f"{100 * out['r_med_mean']:.4f}% (지름 대비)\n[저장] {rj}", flush=True)

# --- 4) 시각 품질 -----------------------------------------------------
if a.phase in ("vq", "all"):
    s_ref = int(d.get("s_conv") or S_TRAIN)
    s_test = int(d.get("s") or S_TRAIN)
    os.makedirs(f"{O}/vq", exist_ok=True)
    dr = f"{O}/vq/sg_{a.shape}_{a.material}_ref_{s_ref}"
    dt = f"{O}/vq/sg_{a.shape}_{a.material}_test_{s_test}"
    rollout(s_ref, frames=a.vq_frames, render_dir=dr)
    rollout(s_test, frames=a.vq_frames, render_dir=dt)
    mj = f"{O}/vq/sg_{a.shape}_{a.material}_vq.json"
    subprocess.run(["python", "-u", f"{W}/anchorflow/exe/vq_metrics.py",
                    "--ref", dr, "--test", dt, "--out", mj,
                    "--win", str(a.win),
                    "--label", f"sg {a.shape} {a.material} n_step={s_test} "
                               f"vs {s_ref}"],
                   cwd=f"{W}/anchorflow",
                   env=dict(os.environ, PYTHONPATH=f"{W}/anchorflow/lib",
                            PYTHONUTF8="1"))
    for src, tag in ((dt, "test"), (dr, "ref")):
        subprocess.run(["python", "-u", f"{W}/anchorflow/exe/pngs2mp4.py",
                        "--dir", src, "--out",
                        f"{O}/vq/sg_{a.shape}_{a.material}_{tag}.mp4",
                        "--fps", "20"], cwd=f"{W}/anchorflow")
print("SG_CELL_DONE", flush=True)
