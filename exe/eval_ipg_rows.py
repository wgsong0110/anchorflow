"""저장된 정답 위에서 i-PhysGaussian 을 채점한다.

exe/eval_ref_dump.py 가 확정해 둔 정답과 초기 속도를 그대로 불러오므로, 두 저장소의
mpm_solver_warp 사본이 달라도 기준이 흔들리지 않는다. 지표도 같다 -- 프레임마다 입자
평균 거리를 그 궤적의 MPM 자체 최대 변위로 나눈 값.

행이 셋이다.

  i-PG k=1   그쪽 **명시적** 솔버. 타임스텝을 키우지 않았으므로 우리 정답과의 차이는
             오롯이 두 코드베이스의 차이다 -- 대조군이고, 이것이 크면 아래 두 행의
             해석에서 그만큼을 빼고 읽어야 한다.
  i-PG k>1   그쪽 암시적 솔버. 이것이 그들이 지불하는 것: 물리도 상태도 그대로 두고
             타임스텝만 k 배로 키운다. 우리 학생은 반대로 상태를 줄이고(입자 171,553
             -> 앵커 512) 타임스텝은 그대로다.

지표를 둘 다 낸다.

  고정      mean_t mean_i ||pred - ref|| / grid_lim      <- 기본. i-PhysGaussian 이 쓰는
            정규화이고, 논문이 이유를 밝혀 두었다: 거의 정지한 구간에서 나눗셈이
            불안정해지는 것을 피하려고.
  자체변위  ... / (그 궤적의 MPM 자체 최대 변위)          <- 이전 격자가 쓰던 것

자체 변위로 나누면 거의 안 움직인 궤적에서 값이 폭발한다. 실측: 같은 12 칸 안에서
MPM 최대 변위가 0.021 ~ 1.004 로 48 배 벌어지고(전체 격자로는 150 배), log 변위와
log 오차의 상관이 네 모델 모두 -0.88 ~ -0.93 이었다. K=1 과 작은 반경이 취약해
보이던 것의 상당 부분이 물리가 아니라 이 분모였다. K=1, r=0.75x 에서는 MPM 최대
변위가 0.0006 이라 오차가 458% 로 찍혔다.

E 스케일을 맞춘다. DreamPhysics 커널은 mu/lam 을 만들 때 E 에 1e7 을 곱하고 i-PG 사본은
곱하지 않으므로, 같은 config 를 그대로 주면 재질이 1e7 배 물러진다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--ipg", required=True)
ap.add_argument("--dump", required=True)
ap.add_argument("--k", type=int, nargs="+", default=[1, 4, 20])
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--grid_lim", type=float, default=2.0)
ap.add_argument("--out", default=None)
args = ap.parse_args()

sys.path.insert(0, args.ipg)          # mpm_solver_warp 를 i-PG 사본으로 잡는다
import warp as wp

dev = "cuda"
torch.set_grad_enabled(False)
wp.init()
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from implicit_mpm_solver import ImplicitMPMSolver

sc0 = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                        frozen_weights=True, rot_fallback=True, eig_floor=0.02)
# 정답을 만든 MPMTeacher 와 같은 방식으로 물질 입자를 고른다(mpm_teacher.py:64)
MAT = torch.nonzero(sc0.keep, as_tuple=False).squeeze(-1)
X0 = sc0.pos[MAT].contiguous()
VOL = sc0.volume[MAT].contiguous()
N = X0.shape[0]
GRID_LIM, N_GRID = 2.0, int(getattr(sc0, "n_grid", None) or 100)
E_SCALED = float(sc0.cfg["E"]) * 1e7
print(f"[setup] 입자 {N}, n_grid {N_GRID}, E {sc0.cfg['E']} -> {E_SCALED:.4g}, "
      f"k={args.k}", flush=True)


def make(k):
    cls = MPM_Simulator_WARP if k == 1 else ImplicitMPMSolver
    s = cls(N, n_grid=N_GRID, grid_lim=GRID_LIM)
    s.load_initial_data_from_torch(X0.clone(), VOL.clone(),
                                    torch.zeros((N, 6), device=dev),
                                    n_grid=N_GRID, grid_lim=GRID_LIM)
    mp = {kk: sc0.cfg[kk] for kk in ("nu", "density", "material") if kk in sc0.cfg}
    mp["E"] = E_SCALED
    mp.update({"n_grid": N_GRID, "grid_lim": GRID_LIM,
               "g": sc0.cfg.get("g", [0, 0, 0]),
               "grid_v_damping_scale": sc0.cfg.get("grid_v_damping_scale", 1.0)})
    if "additional_material_params" in sc0.cfg:
        # 이 블록도 E 를 덮어쓴다(ficus 는 잎 영역을 E=0.01 로). 최상위만 스케일하면
        # 그 영역만 1e7 배 물러져 궤적이 완전히 달라진다 -- k=1 대조군이 15~25% 로
        # 나온 원인이었다.
        extra = []
        for blk in sc0.cfg["additional_material_params"]:
            blk = dict(blk)
            if "E" in blk:
                blk["E"] = float(blk["E"]) * 1e7
            extra.append(blk)
        mp["additional_material_params"] = extra
        print(f"[setup] 추가 재질 블록 {len(extra)}개의 E 도 1e7 배: "
              + ", ".join(f"{b.get('E'):.4g}" for b in extra if "E" in b), flush=True)
    s.set_parameters_dict(mp)
    s.finalize_mu_lam()
    return s


SOLV = {}
def run(v0, k):
    if k not in SOLV:
        SOLV[k] = make(k)
    s = SOLV[k]
    eye = torch.eye(3, device=dev).reshape(1, 9).repeat(N, 1).contiguous()
    s.import_particle_x_from_torch(X0.clone())
    s.import_particle_v_from_torch(v0.clone())
    s.import_particle_F_from_torch(eye.clone())
    s.import_particle_C_from_torch(torch.zeros_like(eye))
    sub = args.dt_mult if k == 1 else max(1, args.dt_mult // k)
    dt = sc0.sub_dt if k == 1 else sc0.sub_dt * k
    xs = [X0.clone()]
    step = 0
    for _ in range(args.frames):
        for _ in range(sub):
            if k == 1:
                s.p2g2p(None, dt, device=str(dev))
            else:
                s.p2g2p_implicit(step, dt)
            step += 1
        x = s.export_particle_x_to_torch().clone()
        if not torch.isfinite(x).all():
            return None
        xs.append(x)
    return torch.stack(xs)


files = sorted(glob.glob(os.path.join(args.dump, "*.pt")))
res = {f"i-PG k={k}": {} for k in args.k}
# 궤적 하나가 몇 분씩 걸린다. 중간에 죽어도 이어갈 수 있도록 매번 저장하고,
# 이미 채점한 궤적은 건너뛴다.
if args.out and os.path.exists(args.out):
    old_ = json.load(open(args.out))
    for r in res:
        res[r].update(old_.get(r, {}))
    done = set(res[list(res)[0]]) if res else set()
    print(f"[재개] 이미 채점된 궤적 {len(done)}개", flush=True)
else:
    done = set()
print(f"[data] 정답 {len(files)}개, 남은 것 {len(files) - len(done)}개", flush=True)
for fp in files:
    if os.path.basename(fp).replace(".pt", "") in done:
        continue
    b = torch.load(fp, map_location="cpu", weights_only=False)
    tag = os.path.basename(fp).replace(".pt", "")
    v0 = b["v0"].to(dev)
    ref = b["ref"].to(dev).float()
    span = float((ref - ref[0]).norm(dim=-1).max())
    line = [f"변위 {span:.4f}"]
    for k in args.k:
        got = run(v0, k)
        if got is None:
            res[f"i-PG k={k}"][tag] = None
            line.append(f"k={k} 발산")
            continue
        dist = float((got - ref).norm(dim=-1).mean())
        res[f"i-PG k={k}"][tag] = {"fixed": 100 * dist / args.grid_lim,
                                    "span": 100 * dist / max(span, 1e-12),
                                    "dist": dist, "mpm_span": span}
        line.append(f"k={k} {100*dist/args.grid_lim:5.2f}%(고정)/"
                    f"{100*dist/max(span,1e-12):6.1f}%(자체)")
    print(f"  {tag:20s} " + "  ".join(line), flush=True)
    del ref, v0
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)

print(f"\n{'행':>12} {'고정 평균':>10} {'고정 최악':>10} {'자체 평균':>10} "
      f"{'자체 최악':>10} {'발산':>6}")
for r in res:
    v = [x for x in res[r].values() if x is not None]
    bad = sum(1 for x in res[r].values() if x is None)
    if v:
        fx = [x["fixed"] for x in v]; sp = [x["span"] for x in v]
        print(f"{r:>12} {sum(fx)/len(fx):9.3f}% {max(fx):9.3f}% "
              f"{sum(sp)/len(sp):9.2f}% {max(sp):9.2f}% {bad:5d}")
    else:
        print(f"{r:>12} {'전부 발산':>10} {'':>10} {'':>10} {'':>10} {bad:5d}")
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"저장: {args.out}")
print("\nIPG_ROWS_DONE")
