"""det F < 0 과 그 뒤의 NaN 이 어디서 오는지 한 요소까지 좁혀서 추적한다.

plastic_probe.py 는 조합별로 "완주/발산"만 찍는다. 그것으로는 뒤집힘이 NaN 보다
먼저 온다는 **순서**밖에 못 읽는다. 여기서는 두 번 굴린다 (FEM 이 결정적이라 같은
궤적이 재현된다):

  1차 -- 전체를 굴려 처음 뒤집히는 요소와 그 스텝, 그리고 NaN 이 터지는 스텝을 찾는다.
  2차 -- 그 요소 하나를 0 스텝부터 따라가며 스텝마다 다음을 찍는다.

      det F_total    사면체 기하 자체가 뒤집혔는가
      det F_e_trial  소성 누적기를 통과한 뒤 (= F_total @ Fp_inv)
      det Fp_inv     소성 누적기 자체가 무너졌는가
      S_raw          _svd3 가 돌려준 특이값 (반사 보정이면 마지막이 음수)
      S              clamp(min=1e-4) 통과 후 -- 음수 부호가 여기서 사라진다
      S_new/S        보정 행렬 C 의 배율. 이게 폭주하면 Fp_inv 가 오염된다
      refl           반사 보정 발동 여부
      |dx|/h         스텝당 정점 변위를 요소 크기로 나눈 값 (적분 오버슈트 판정)
      rest vol/AR    그 요소의 쉬는 상태 품질 (슬리버 판정)

이 셋이 서로 배타적이라 어느 것이 먼저 임계를 넘는지 보면 원인이 갈린다:
슬리버(처음부터 납작) / 적분 오버슈트(|dx|/h 가 1 근처) / 리턴 매핑(clamp + C 폭주).

NaN 이 나오면 **그 스텝에서 즉시 멈추고** 직전 스텝과 실패 스텝을 통째로 덤프한다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "lib"))

import numpy as np
import torch

from meshgs.fem import TetFEM
from meshgs.tetcage import build_tet_mesh, dilate_fill, occupancy

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=32)
ap.add_argument("--E", type=float, default=1e5)
ap.add_argument("--nu", type=float, default=0.3)
ap.add_argument("--density", type=float, default=200.0)
ap.add_argument("--compress", type=float, default=0.20)
ap.add_argument("--dt", type=float, default=2e-4)
ap.add_argument("--yield_stress", type=float, default=1e3)
ap.add_argument("--damping", type=float, default=8.0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def log(*s):
    print(*s, flush=True)


# ---------------- 씬: plastic_probe.py 와 같은 압축 시험 ----------------
g = torch.stack(torch.meshgrid(*[torch.linspace(-0.4, 0.4, 16, device=dev)] * 3,
                               indexing="ij"), -1).reshape(-1, 3)
occ, org, h = occupancy(g, res=a.res)
occ = dilate_fill(occ, 1)
V, T = build_tet_mesh(occ, org, h)
H = float(V[:, 2].max() - V[:, 2].min())
top = V[:, 2] > V[:, 2].max() - 1.5 * h
bot = V[:, 2] < V[:, 2].min() + 1.5 * h
PRESS = int(round(0.12 / a.dt))
HOLD = PRESS // 3
REL = PRESS * 3
NSTEP = PRESS + HOLD + REL
v_press = a.compress * H / (PRESS * a.dt)
c_wave = (a.E / a.density) ** 0.5
log(f"[설정] 정점 {V.shape[0]} 사면체 {T.shape[0]} h {h:.4f} 높이 {H:.3f}")
log(f"[설정] dt {a.dt:.1e}  yield {a.yield_stress:.1e}  스텝 {NSTEP}")
log(f"[설정] 파속 {c_wave:.1f} m/s -> 명시적 CFL 한계 dt < {h/c_wave:.2e}")


def new_fem():
    return TetFEM(V, T, density=a.density, E=a.E, nu=a.nu, plastic="von_mises",
                  yield_stress=a.yield_stress, damping=a.damping)


# ---------------- 1차: 첫 뒤집힘 요소와 NaN 스텝 찾기 ----------------
f = new_fem()
Vc, vc = V.clone(), torch.zeros_like(V)
first_inv_step, first_inv_elem, nan_step = -1, -1, -1
for s in range(NSTEP):
    hold = s < PRESS + HOLD
    Vc, vc = f.step(Vc, vc, a.dt, fixed=(bot | top) if hold else bot)
    if s < PRESS:
        Vc[top, 2] -= v_press * a.dt
    if not torch.isfinite(Vc).all():
        nan_step = s
        log(f"[1차] NaN 스텝 {s}")
        break
    if first_inv_step < 0:
        det = torch.linalg.det(f.deform_grad(Vc))
        if (det <= 0).any():
            first_inv_step = s
            first_inv_elem = int(det.argmin())
            log(f"[1차] 첫 뒤집힘 스텝 {s}, 요소 {first_inv_elem}, "
                f"det {float(det.min()):.4e}, 뒤집힌 요소 수 {int((det<=0).sum())}")
if first_inv_step < 0:
    log("[1차] 뒤집힘 없음 -- 이 조합은 완주한다. 다른 dt/yield 로 볼 것")
    sys.exit(0)
if nan_step < 0:
    log(f"[1차] NaN 없이 {NSTEP} 스텝 완주 (뒤집힘은 {first_inv_step} 에 있었다)")

EL = first_inv_elem
tv = T[EL]
log(f"[대상] 요소 {EL}, 정점 {tv.tolist()}")

# 그 요소의 쉬는 상태 품질
D0 = torch.stack([V[tv[0]] - V[tv[3]], V[tv[1]] - V[tv[3]],
                  V[tv[2]] - V[tv[3]]], -1)
vol0 = float(torch.linalg.det(D0).abs() / 6.0)
sv0 = torch.linalg.svdvals(D0)
ar0 = float(sv0.max() / sv0.min().clamp(min=1e-12))
vol_med = float((torch.linalg.det(torch.stack(
    [V[T][:, 0] - V[T][:, 3], V[T][:, 1] - V[T][:, 3],
     V[T][:, 2] - V[T][:, 3]], -1)).abs() / 6.0).median())
log(f"[대상] 쉬는 상태 부피 {vol0:.4e} (전체 중앙값 {vol_med:.4e}, "
    f"비 {vol0/vol_med:.3f}), 종횡비 {ar0:.3f}, 변 길이 척도 h {h:.4f}")

# ---------------- 2차: 그 요소를 0 스텝부터 추적 ----------------
f = new_fem()
Vc, vc = V.clone(), torch.zeros_like(V)
rows = []
fail = None
log("")
log(f"{'step':>5} {'detF':>11} {'detFe_tr':>11} {'detFp_inv':>11} "
    f"{'S_raw_min':>10} {'S_min':>9} {'C_max':>10} {'refl':>5} {'|dx|/h':>8} "
    f"{'|f|max':>10} {'inv%':>6}")
for s in range(NSTEP):
    hold = s < PRESS + HOLD
    d = {}
    Vprev = Vc.clone()
    Vc, vc = f.step(Vc, vc, a.dt, fixed=(bot | top) if hold else bot, diag=d)
    if s < PRESS:
        Vc[top, 2] -= v_press * a.dt

    detF = float(torch.linalg.det(d["F_total"][EL]))
    detFe = float(torch.linalg.det(d["F_e_trial"][EL]))
    detFp = float(torch.linalg.det(d["Fp_inv"][EL]))
    S_raw = d["S_raw"][EL].tolist()
    S_cl = d["S"][EL].tolist()
    ratio = (d["S_new"][EL] / d["S"][EL]).tolist()
    refl = bool(d["refl"][EL])
    fmax = float(d["force"].norm(dim=-1).max())
    dx = float((Vc[tv] - Vprev[tv]).norm(dim=-1).max())
    det_all = torch.linalg.det(d["F_total"])
    inv_frac = float((det_all <= 0).float().mean())

    row = dict(step=s, detF=detF, detFe_trial=detFe, detFp_inv=detFp,
               S_raw=S_raw, S_clamped=S_cl, C_ratio=ratio, refl=refl,
               dx_over_h=dx / h, force_max=fmax, inv_frac=inv_frac,
               over=bool(d["over"][EL]))
    rows.append(row)

    bad_V = not torch.isfinite(Vc).all()
    bad_el = not all(map(math.isfinite, [detF, detFe, detFp, fmax]))
    if bad_V or bad_el or s <= 2 or s % 25 == 0 or s >= first_inv_step - 3:
        log(f"{s:5d} {detF:11.3e} {detFe:11.3e} {detFp:11.3e} "
            f"{min(S_raw):10.3e} {min(S_cl):9.3e} {max(ratio):10.3e} "
            f"{str(refl):>5} {dx/h:8.4f} {fmax:10.3e} {100*inv_frac:6.2f}")
    if bad_V or bad_el:
        # NaN 발생 -- 여기서 즉시 멈춘다.
        which = []
        for nm in ("F_total", "F_e_trial", "F_e", "P", "Fp_inv", "force"):
            t_ = d[nm]
            n_bad = int((~torch.isfinite(t_.reshape(t_.shape[0], -1))).any(-1).sum())
            which.append(f"{nm}={n_bad}")
        fail = dict(step=s, in_vertices=bool(bad_V), in_element=bool(bad_el),
                    nonfinite_elements=which,
                    n_bad_vertices=int((~torch.isfinite(Vc)).any(-1).sum()))
        log("")
        log(f"[NaN] 스텝 {s} 에서 비유한값 발생 -- 즉시 중단")
        log(f"[NaN] 어디에: {', '.join(which)}  (정점 {fail['n_bad_vertices']}개)")
        if len(rows) >= 2:
            pr = rows[-2]
            log(f"[NaN] 직전 스텝 {pr['step']}: detF {pr['detF']:.3e}  "
                f"detFp_inv {pr['detFp_inv']:.3e}  C_max {max(pr['C_ratio']):.3e}  "
                f"|f|max {pr['force_max']:.3e}  뒤집힘 {100*pr['inv_frac']:.2f}%")
        break

out = a.out or os.path.join(_here, "..", "..", "trace_out")
os.makedirs(out, exist_ok=True)
rec = dict(args=vars(a), n_vert=int(V.shape[0]), n_tet=int(T.shape[0]), h=h,
           height=H, n_step=NSTEP, cfl_limit=h / c_wave,
           first_inv_step=first_inv_step, first_inv_elem=EL, nan_step=nan_step,
           rest_vol=vol0, rest_vol_median=vol_med, rest_aspect=ar0,
           fail=fail, rows=rows)
p_json = os.path.join(out, "trace.json")
json.dump(rec, open(p_json, "w"), indent=1)
log("")
log(f"[저장] {p_json}  ({len(rows)} 스텝)")
log("NAN_DETECTED" if fail is not None else "NO_NAN")
log("TRACE_OK")
