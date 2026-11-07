"""절 CM28 (1): 정답 첫 접촉 통계 -- GF h5 (원래 정답, f_tensor = particle_F = RM 뒤 탄성 F_e) + 다시 돌린 정답 logJp (lbours/fem_cm/wmJd/gt_jp).
우리 판 (--erec en.fs) 과 같은 함수 (lib/anchorflow/wmfrac_stats.py). 부피 = 정답 입자 질량 / 밀도 (exe/wmk1fix_tab.py gt_energy 와 같음).
사용: python exe/wmfrac_gt.py <wm.pt> <h5 디렉토리> <gt_jp 디렉토리> <출력 json> [프레임 수 12]
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import deform_model as DM  # noqa: E402
from anchorflow import phys_resid as PR  # noqa: E402
from anchorflow import wmfrac_stats as WFS  # noqa: E402
import h5py  # noqa: E402

WM, H5, JP, OUT = sys.argv[1:5]
NF = int(sys.argv[5]) if len(sys.argv) > 5 else 12
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load(WM, map_location="cpu", weights_only=False)
cfg = d["cfg"]; mu, lam = PR.lame(cfg["E"], cfg["nu"])
m = DM.traj_mass(d, "cpu").double()
vol = (m / float(cfg["density"])).to(dev)
g_ = torch.tensor([float(q) for q in cfg["g"]], dtype=torch.float64, device=dev)
out = []
for f in range(NF + 1):
    hp = os.path.join(H5, "sim_%010d.h5" % f)
    with h5py.File(hp, "r") as h:
        F = torch.from_numpy(np.ascontiguousarray(h["f_tensor"][:].T)).to(dev).reshape(-1, 3, 3)
        x = torch.from_numpy(np.ascontiguousarray(h["x"][:].T)).to(dev)
        v = torch.from_numpy(np.ascontiguousarray(h["v"][:].T)).to(dev).double()
        tm = float(np.array(h["time"]).ravel()[0])
    jp = os.path.join(JP, "sc_%04d.npy" % f)
    lj = torch.from_numpy(np.load(jp)).to(dev) if os.path.exists(jp) else None
    r = WFS.frame_stats(F, vol, x, lj, float(mu), float(lam))
    md = m.to(dev)
    ok = torch.isfinite(x).all(-1) & torch.isfinite(v).all(-1)
    r.update(t=f, time=tm, KE=float(0.5 * (md[ok] * (v[ok] ** 2).sum(1)).sum()), PE=float(-(md[ok] * (x[ok].double() @ g_)).sum()),
             zc=float((md[ok] * x[ok][:, 2].double()).sum() / md[ok].sum()), vz=float((md[ok] * v[ok][:, 2]).sum() / md[ok].sum()), M=float(md[ok].sum()))
    out.append(r)
    print(f"정답 f{f} t {tm:.5f} J 중앙 {r['Jq'][3]:.3f} J<0.9 {r['J090']:.3f} SNH {r['Esnh']:.2f} NHB {r['Enhb']:.2f} J 운동 {r['KE']:.2f} 위치 {r['PE']:.2f} 손상 {r.get('lj_pos')}", flush=True)
json.dump(dict(frames=out, mu=float(mu), lam=float(lam)), open(OUT, "w"), indent=1)
