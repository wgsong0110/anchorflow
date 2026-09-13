"""리턴 매핑이 뒤집힘의 원인인지 직접 가른다.

plastic_trace.py 는 소성 쪽 한 번만 굴려서 "탄성은 완주했다"는 별도 실행과
정황으로 비교했다. 그것만으로는 리턴 매핑이 원인이라고 말할 수 없다.

여기서는 **같은 압축 궤적을 두 번** 굴린다. 다른 것은 리턴 매핑 하나뿐이다:

  A) plastic = von_mises   -- 리턴 매핑 켬
  B) plastic = none        -- 리턴 매핑 끔 (탄성 대조군), dt / 압축 / 감쇠 전부 동일

같은 스텝에서 두 실행의 det F 를 나란히 보면, 뒤집힘이 리턴 매핑 때문인지
압축 자체 때문인지 갈린다. B 도 같이 뒤집히면 리턴 매핑은 무죄다.

그리고 A 에서는 스텝마다 리턴 매핑이 실제로 F 를 얼마나 바꿨는지 잰다:
리턴 매핑은 U, V 를 건드리지 않고 특이값만 바꾸므로, 들어간 특이값 S_raw 와
나온 특이값 S_new 를 직접 비교하면 그것이 곧 왜곡의 전부다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "lib"))

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
ap.add_argument("--elem", type=int, default=166818,
                help="추적할 요소. plastic_trace 가 찾은 첫 뒤집힘 요소")
ap.add_argument("--out", default=None)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def log(*s):
    print(*s, flush=True)


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
EL = min(a.elem, T.shape[0] - 1)
log(f"[설정] 정점 {V.shape[0]} 사면체 {T.shape[0]} h {h:.4f} dt {a.dt:.1e} "
    f"스텝 {NSTEP} 추적요소 {EL}")


def run(kind):
    """한 번 굴린다. 뒤집힘/NaN 이 나면 그 스텝을 기록하되 끝까지 가지 않는다."""
    f = TetFEM(V, T, density=a.density, E=a.E, nu=a.nu, plastic=kind,
               yield_stress=a.yield_stress, damping=a.damping)
    Vc, vc = V.clone(), torch.zeros_like(V)
    rows, first_inv, nan_step = [], -1, -1
    for s in range(NSTEP):
        hold = s < PRESS + HOLD
        d = {}
        Vprev = Vc.clone()
        Vc, vc = f.step(Vc, vc, a.dt, fixed=(bot | top) if hold else bot, diag=d)
        if s < PRESS:
            Vc[top, 2] -= v_press * a.dt
        det_all = torch.linalg.det(d["F_total"])
        row = dict(step=s,
                   detF=float(det_all[EL]),
                   detF_min=float(det_all.min()),
                   inv_frac=float((det_all <= 0).float().mean()),
                   dx_over_h=float((Vc[T[EL]] - Vprev[T[EL]]).norm(dim=-1).max()) / h,
                   force_max=float(d["force"].norm(dim=-1).max()))
        if kind != "none":
            # 리턴 매핑이 실제로 바꾼 양: 들어간 특이값 대 나온 특이값
            s_in = d["S_raw"][EL]
            s_out = d["S_new"][EL]
            row.update(S_in=s_in.tolist(), S_out=s_out.tolist(),
                       distort=float((s_out / s_in.clamp(min=1e-4)).max()),
                       yielded=bool(d["over"][EL]),
                       yielded_frac=float(d["over"].float().mean()))
        rows.append(row)
        if first_inv < 0 and float(det_all.min()) <= 0:
            first_inv = s
            log(f"  [{kind}] 첫 뒤집힘 스텝 {s} "
                f"(det최소 {float(det_all.min()):.3e}, {int((det_all<=0).sum())}개)")
        if not torch.isfinite(Vc).all() or not math.isfinite(row["force_max"]):
            nan_step = s
            log(f"  [{kind}] NaN 스텝 {s} -- 중단")
            break
    if first_inv < 0:
        log(f"  [{kind}] {NSTEP} 스텝 뒤집힘 없이 완주")
    if nan_step < 0 and first_inv >= 0:
        log(f"  [{kind}] NaN 없이 완주 (뒤집힘만 있었다)")
    return rows, first_inv, nan_step


log("")
log("A) 리턴 매핑 켬 (von_mises)")
A, A_inv, A_nan = run("von_mises")
log("")
log("B) 리턴 매핑 끔 (탄성 대조군, 나머지 조건 동일)")
B, B_inv, B_nan = run("none")

log("")
log("같은 스텝 나란히 보기 (추적 요소 / 전체 최소 det F)")
log(f"{'step':>5} | {'A detF':>10} {'A detmin':>10} {'A 왜곡':>8} {'A 항복%':>8} "
    f"| {'B detF':>10} {'B detmin':>10}")
marks = sorted(set([0, 50, 100, 150, 200, 225]
                   + list(range(max(0, A_inv - 4), min(len(A), A_inv + 3)))
                   if A_inv >= 0 else [0, 50, 100, 150, 200]))
for s in marks:
    if s >= len(A):
        continue
    ra = A[s]
    rb = B[s] if s < len(B) else None
    bstr = (f"| {rb['detF']:10.3e} {rb['detF_min']:10.3e}" if rb else "| (끝남)")
    log(f"{s:5d} | {ra['detF']:10.3e} {ra['detF_min']:10.3e} "
        f"{ra.get('distort', float('nan')):8.3f} "
        f"{100*ra.get('yielded_frac', 0):8.2f} {bstr}")

log("")
if A_inv >= 0 and B_inv < 0:
    log("[판정] 리턴 매핑을 켠 쪽만 뒤집혔다 -- 리턴 매핑이 원인이다.")
elif A_inv >= 0 and B_inv >= 0:
    log(f"[판정] 둘 다 뒤집혔다 (A {A_inv} / B {B_inv}) -- "
        "압축 자체가 원인이고 리턴 매핑은 시점만 당긴다.")
elif A_inv < 0:
    log("[판정] 소성 쪽도 안 뒤집혔다 -- 이 조합으로는 재현이 안 된다.")

out = a.out or os.path.join(_here, "..", "..", "ablate_out")
os.makedirs(out, exist_ok=True)
json.dump(dict(args=vars(a), elem=EL, n_tet=int(T.shape[0]), h=h,
               A=dict(first_inv=A_inv, nan=A_nan, rows=A),
               B=dict(first_inv=B_inv, nan=B_nan, rows=B)),
          open(os.path.join(out, "ablate.json"), "w"), indent=1)
log(f"[저장] {os.path.join(out, 'ablate.json')}")
log("ABLATE_OK")
