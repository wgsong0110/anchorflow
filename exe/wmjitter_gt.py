"""절 CM90 묶음 wmjitter 정답 (GF) 쪽 지표 (읽기만): wm.pt 의 프레임별 입자 x·v·F 로 lib/anchorflow/fem_wmjitter.py 와 같은 입자 지표·가속도·부분 표본 -> <out>/gt.npz, gtinfo.json.
배열 첫 축 = 우리 프레임 번호 fi (0 = f1) = 정답 프레임 fi + 1 (실행기 오차와 같은 짝). 가우스 점은 정답에 없음 (MPM).
사용: python exe/wmjitter_gt.py --data wm.pt --out DIR [--sub 64]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import deform_model as DM  # noqa: E402
from anchorflow import phys_resid as PR  # noqa: E402
from anchorflow import fem_wmjitter as WJ  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sub", type=int, default=64)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
try:
    d = torch.load(a.data, map_location="cpu", weights_only=False, mmap=True)
except Exception as ex:
    print("[정답] mmap 안 됨", repr(ex), flush=True)
    d = torch.load(a.data, map_location="cpu", weights_only=False)
print("[정답] 키", {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in d.items()}, flush=True)
cfg = d["cfg"]
print("[정답] cfg", json.dumps({k: cfg[k] for k in cfg if not isinstance(cfg[k], (list, dict)) or k in ("g",)}, ensure_ascii=False, default=str), flush=True)
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).double().to(dev)
vol = m / float(cfg["density"])
mu, lam = PR.lame(cfg["E"], cfg["nu"])
X, V = d["x"], d.get("v")
FF = d.get("F")
Lf = int(X.shape[0]) - 1
Np = int(X.shape[1])
so = torch.arange(0, Np, a.sub)
T = np.full((Lf, len(WJ.COLS)), np.nan)
A = np.full((Lf, len(WJ.ACOLS)), np.nan)
COM = np.full((Lf, 3), np.nan)
JS = np.full((Lf, so.numel()), np.nan, dtype=np.float32)
Jp = None
c_ = {k: i for i, k in enumerate(WJ.COLS)}
for fi in range(Lf):
    t = fi + 1
    x = X[t].to(dev).double()
    v = V[t].to(dev).double() if V is not None and V.shape[0] > t else None
    okp = torch.isfinite(x).all(-1) & (torch.isfinite(v).all(-1) if v is not None else True)
    mm = torch.where(okp, m, torch.zeros_like(m))
    x0 = torch.where(okp[:, None], x, torch.zeros_like(x))
    M = mm.sum(); c = (mm[:, None] * x0).sum(0) / M
    T[fi, c_["M"]] = float(M); T[fi, c_["cx"]:c_["cz"] + 1] = c.cpu().numpy(); COM[fi] = c.cpu().numpy()
    if v is not None:
        v0 = torch.where(okp[:, None], v, torch.zeros_like(v))
        P = (mm[:, None] * v0).sum(0); KE = 0.5 * (mm * (v0 * v0).sum(-1)).sum()
        r = x0 - c; Lm = (mm[:, None] * torch.linalg.cross(r, v0)).sum(0)
        I = torch.eye(3, device=dev, dtype=torch.float64) * (mm * (r * r).sum(-1)).sum() - (mm[:, None] * r).T @ r
        w = torch.linalg.solve(I, Lm); KEt = 0.5 * (P * P).sum() / M; KEr = 0.5 * (w * Lm).sum()
        vd = v0 - P / M - torch.linalg.cross(w.expand_as(r), r)
        T[fi, c_["KE"]] = float(KE); T[fi, c_["Px"]:c_["Pz"] + 1] = P.cpu().numpy(); T[fi, c_["KEt"]] = float(KEt); T[fi, c_["KEr"]] = float(KEr)
        T[fi, c_["KEdef"]] = float(KE - KEt - KEr); T[fi, c_["vdef_rms"]] = float(torch.sqrt((mm * (vd * vd).sum(-1)).sum() / M)); T[fi, c_["wx"]:c_["wz"] + 1] = w.cpu().numpy()
    if FF is not None:
        F = FF[t].to(dev).double().reshape(-1, 9)
        J = WJ._det(F); ok = torch.isfinite(J) & okp
        Ic = (F * F).sum(-1)
        psi = 0.5 * mu * (Ic - 3.0) - mu * (J - 1.0) + 0.5 * (lam + mu) * (J - 1.0) ** 2
        okE = ok & torch.isfinite(psi)
        T[fi, c_["Eel"]] = float(torch.where(okE, vol * psi, torch.zeros_like(psi)).sum())
        Jn = torch.where(ok, J, torch.full_like(J, float("nan")))
        T[fi, c_["Jmed"]] = float(Jn.nanmedian()); T[fi, c_["Jmin"]] = float(Jn[ok].min())
        if Jp is not None:
            okd = ok & torch.isfinite(Jp); dJ = (J - Jp)[okd]
            T[fi, c_["dJrms"]] = float(torch.sqrt((dJ * dJ).mean())); T[fi, c_["dJmean"]] = float(dJ.abs().mean()); T[fi, c_["dJn05"]] = float((dJ.abs() > 0.05).sum())
        T[fi, c_["nfin"]] = float(ok.sum())
        Jp = Jn
        JS[fi] = Jn[so.to(dev)].float().cpu().numpy()
    if 1 <= fi and t + 1 <= Lf:
        xm, xp = X[t - 1].to(dev).double(), X[t + 1].to(dev).double()
        ac_ = xp - 2.0 * x + xm; ok = torch.isfinite(ac_).all(-1); ac_ = ac_[ok]; mo = m[ok]
        an = ac_.norm(dim=-1); acm = (mo[:, None] * ac_).sum(0) / mo.sum(); ar = (ac_ - acm).norm(dim=-1)
        qs = torch.quantile(an.float(), torch.tensor([0.5, 0.9, 0.99], device=dev)).double()
        A[fi] = [float(torch.sqrt((an * an).mean())), float(torch.sqrt((ar * ar).mean())), float(qs[0]), float(qs[1]), float(qs[2]), float(an.max()), float(acm[0]), float(acm[1]), float(acm[2]), float(ok.sum())]
XS = X[1:Lf + 1][:, so].numpy().astype(np.float32)
x0 = X[0].double(); ok0 = torch.isfinite(x0).all(-1)
ext = lambda x: (x[torch.isfinite(x).all(-1)].amax(0) - x[torch.isfinite(x).all(-1)].amin(0)).tolist()
info = dict(Np=Np, frames=Lf, h=h, E=float(cfg["E"]), nu=float(cfg["nu"]), density=float(cfg["density"]), g=[float(q) for q in cfg["g"]], mu=mu, lam=lam,
            vol_total=float(vol.sum()), mass_total=float(m.sum()), ext_f0=ext(x0), ext_f50=ext(X[50].double()), ext_f100=ext(X[Lf].double()),
            substep_dt=float(cfg.get("substep_dt", float("nan"))), keys=list(d.keys()), haveF=FF is not None, haveV=V is not None)
np.savez_compressed(os.path.join(a.out, "gt.npz"), T=T, cols=np.array(WJ.COLS), A=A, acols=np.array(WJ.ACOLS), COM=COM, XS=XS, sub_o=so.numpy(), JS=JS, h=np.array([h]))
json.dump(info, open(os.path.join(a.out, "gtinfo.json"), "w"), ensure_ascii=False, indent=1, default=str)
print("[정답 지표 끝]", json.dumps(info, ensure_ascii=False, default=str)[:600], flush=True)
