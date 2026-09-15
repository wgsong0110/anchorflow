"""방어를 하나씩 켜가며 소성 발산이 실제로 막히는지 가른다.

`plastic_ablate.py` 가 보인 것은 "리턴 매핑을 켜면 뒤집히고 끄면 완주한다"였다.
`plastic_trace.py` 가 보인 사슬은 이랬다:

  리턴 매핑이 요소를 납작하게 만듦 -> det F 가 0 으로 밀림 -> neo-Hookean 의
  F^-T 항이 1/det 로 발산 -> 명시적 스텝이 요소 크기의 2.88 배를 한 번에 이동해
  관통(det F < 0) -> _svd3 의 음수 특이값을 clamp 가 지움 -> 보정 배율 1e4 폭주 ->
  Fp_inv 오염 -> 힘 1e13 -> inv(F) 가 float32 에서 inf -> inf - inf = NaN

그 사슬의 각 고리에 대응하는 방어가 있다. 어느 고리를 끊어야 실제로 막히는지는
하나씩 켜 봐야 안다 -- 전부 켜서 "NaN 이 안 난다"만 보이면 무엇이 효과였는지 모른다.

판정 기준 세 가지를 같이 본다:
  완주 여부 / 뒤집힌 요소 비율 / **잔류 변형** -- 소성이 목적이므로 눌렀다 놓은 뒤
  영구 변형이 남아야 한다. NaN 만 막고 탄성처럼 되돌아가면 소성을 표현한 게 아니다.
"""
from __future__ import annotations

import argparse
import json
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
log(f"[설정] 정점 {V.shape[0]} 사면체 {T.shape[0]} dt {a.dt:.1e} "
    f"yield {a.yield_stress:.1e} 스텝 {NSTEP}")

# 사슬의 고리마다 하나씩. 마지막은 전부 켠 것.
CASES = [
    ("방어 없음 (기준)",       dict()),
    ("clamp 부호 보존",        dict(guard_sign=True)),
    ("보정 배율 상한 10",       dict(guard_ratio=10.0)),
    ("응력 inv 방어",          dict(guard_inv=True)),
    ("스텝 이동 제한 0.3h",     dict(guard_ccd=0.3)),
    ("부호+배율",              dict(guard_sign=True, guard_ratio=10.0)),
    ("부호+배율+inv",          dict(guard_sign=True, guard_ratio=10.0,
                                   guard_inv=True)),
    ("전부",                   dict(guard_sign=True, guard_ratio=10.0,
                                   guard_inv=True, guard_ccd=0.3)),
    ("이동제한+inv",           dict(guard_ccd=0.3, guard_inv=True)),
]

log("")
log(f"{'방어':<22} {'결과':>8} {'실패스텝':>8} {'첫뒤집힘':>9} "
    f"{'뒤집힘%':>8} {'잔류%':>8} {'detF최소':>10}")
rows = []
for name, kw in CASES:
    f = TetFEM(V, T, density=a.density, E=a.E, nu=a.nu, plastic="von_mises",
               yield_stress=a.yield_stress, damping=a.damping, **kw)
    Vc, vc = V.clone(), torch.zeros_like(V)
    fail, first_inv = -1, -1
    for s in range(NSTEP):
        hold = s < PRESS + HOLD
        Vc, vc = f.step(Vc, vc, a.dt, fixed=(bot | top) if hold else bot)
        if s < PRESS:
            Vc[top, 2] -= v_press * a.dt
        if not torch.isfinite(Vc).all():
            fail = s
            break
        iv, dmin, _ = f.quality(Vc)
        if first_inv < 0 and iv > 0:
            first_inv = s
    if fail < 0:
        iv, dmin, _ = f.quality(Vc)
        resid = float((Vc - V).norm(dim=-1).mean()) / H
        res = "완주"
    else:
        iv = dmin = float("nan")
        resid = float("nan")
        res = "발산"
    log(f"{name:<22} {res:>8} {fail if fail>=0 else '-':>8} "
        f"{first_inv if first_inv>=0 else '-':>9} "
        f"{100*iv if iv==iv else float('nan'):8.2f} "
        f"{100*resid if resid==resid else float('nan'):8.3f} "
        f"{dmin if dmin==dmin else float('nan'):10.3f}")
    rows.append(dict(name=name, guards=kw, diverged=fail >= 0, fail_step=fail,
                     first_inv=first_inv,
                     inverted_frac=None if iv != iv else float(iv),
                     residual=None if resid != resid else float(resid),
                     det_min=None if dmin != dmin else float(dmin)))

out = a.out or os.path.join(_here, "..", "..", "guards_out")
os.makedirs(out, exist_ok=True)
json.dump(dict(args=vars(a), n_vert=int(V.shape[0]), n_tet=int(T.shape[0]),
               n_step=NSTEP, cases=rows),
          open(os.path.join(out, "guards.json"), "w"), indent=1,
          ensure_ascii=False)
log("")
ok = [r for r in rows if not r["diverged"]]
log(f"[요약] 완주 {len(ok)}/{len(rows)}")
for r in ok:
    log(f"  {r['name']}: 잔류 {100*r['residual']:.3f}%, "
        f"뒤집힘 {100*r['inverted_frac']:.2f}%")
log(f"[저장] {os.path.join(out, 'guards.json')}")
log("GUARDS_OK")
