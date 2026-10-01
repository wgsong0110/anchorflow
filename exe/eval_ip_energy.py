"""궤적 하나하나에 **우리 증분 포텐셜**을 그대로 먹여 프레임별 값을 남긴다.

PG·i-PG·출력만 최적화를 같은 자로 재려면 세 결과를 같은 식에 넣어야 한다.

    E_t = Σ m/(2h²)‖Δu − h v − h²g‖² + Σ V Ψ(F_{t+1}) + E_c

탄성항은 궤적에 저장된 F 를 그대로 쓴다 (PG/i-PG 는 자기 F 를 h5 에 남긴다).
변형장 야코비안이 필요 없으므로 어떤 궤적이든 잴 수 있다. 손잡이 입자는
관성항에서 뺀다 -- 그 잔차는 반력이 실어 나르는 것이라 누구의 책임도 아니다.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch

from anchorflow import phys_resid

ap = argparse.ArgumentParser()
ap.add_argument("--traj", nargs="+", required=True,
                help="'이름=경로' 꼴도 된다")
ap.add_argument("--out", required=True, help="json 경로")
ap.add_argument("--t0", type=int, default=0)
ap.add_argument("--len", type=int, default=24)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def L(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


out = {}
for spec in a.traj:
    name, _, path = spec.partition("=")
    if not path:
        name, path = path or spec, spec
    d = L(path)
    cfg = d["cfg"]
    h = float(cfg["frame_dt"])
    X = d["x"].to(dev).float()
    gv = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)
    x0 = X[0]
    ng_, gl_ = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
    dx_ = gl_ / ng_
    vi = (x0 / dx_).long().clamp(0, ng_ - 1)
    fl = (vi[:, 0] * ng_ + vi[:, 1]) * ng_ + vi[:, 2]
    cn = torch.zeros(ng_ ** 3, device=dev).index_add_(
        0, fl, torch.ones(x0.shape[0], device=dev))
    mass = ((dx_ ** 3) / cn[fl]) * float(cfg["density"])
    vol = mass / float(cfg["density"])
    ext = float((x0.max(0).values - x0.min(0).values).norm())
    nrm = float(mass.sum()) * (ext ** 2) / (h ** 2)
    F = d["F"].to(dev).float() if "F" in d else None
    # 손잡이 입자는 관성항에서 뺀다 (매 프레임 중심에서 반경 안)
    cp = d.get("ctrl_pos")
    R = float(d["ctrl_R"].reshape(-1)[0]) if "ctrl_R" in d else 0.0

    rows = []
    for i in range(a.len):
        t = a.t0 + i
        if t + 1 >= X.shape[0]:
            break
        x, x2 = X[t], X[t + 1]
        v = (x - X[max(t - 1, 0)]) / h
        free = None
        if cp is not None and R > 0:
            c = cp[min(t, cp.shape[0] - 1)].to(dev).float()
            dmin = (x.unsqueeze(1) - c.unsqueeze(0)).norm(dim=-1).min(1).values
            free = (dmin >= R).float()
        du = x2 - x
        w = torch.ones_like(mass) if free is None else free
        e_in = float((0.5 * w * mass / (h * h)
                      * ((du - h * v - (h * h) * gv) ** 2).sum(-1)).sum())
        if F is not None:
            psi, _ = phys_resid.psi_of(F[t + 1], cfg, h)
            e_el = float((vol * psi).sum())
        else:
            e_el = float("nan")
        e_bc = float(phys_resid.bc_energy(x, du, mass, cfg, h, gl_, ng_))
        rows.append(dict(t=t, E=(e_in + e_el + e_bc) / nrm,
                         ein=e_in, eel=e_el, ebc=e_bc))
    out[name] = dict(path=path, norm=nrm, ext=ext, h=h, frames=rows)
    _m = np.mean([q["E"] for q in rows])
    print(f"[{name}] {len(rows)} 프레임, E 평균 {_m:.4e} "
          f"(관성 {np.mean([q['ein'] for q in rows]):.4e} / 탄성 "
          f"{np.mean([q['eel'] for q in rows]):.4e})", flush=True)

json.dump(out, open(a.out, "w"))
print(f"[저장] {a.out}", flush=True)
