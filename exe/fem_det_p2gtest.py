"""결정적 P2G 두 판 대조: 칸 버퍼 (AO_DET_P2GC=1) 대 노드 모으기 (0) -- 같은 첫 프레임에서 노드 질량·운동량·노드 상자 정보를 노드별로."""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch
from anchorflow import deform_model as DM, phys_resid as PR, simplex as SX, fem_ao_rt as AORT, fem_rt_cuda as RTC
RTC.RAWPTR[0] = True
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = d["cfg"]; dev = "cuda"
m64 = DM.traj_mass(d, dev).double(); vol = m64 / float(cfg["density"])
mu, lam = PR.lame(cfg["E"], cfg["nu"])
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100)); nh = float(sys.argv[2]) * DXG * math.sqrt(2.0)
lat = SX.lattice(nh, SX.HZ_CUBE * nh, device=dev, dtype=torch.float64)
for fr in range(0, 50, 5):
    x = d["x"][fr].to(dev); v = (d["x"][fr + 1].to(dev) - x) / float(cfg["frame_dt"])
    Np = x.shape[0]; F0 = torch.eye(3, device=dev, dtype=torch.float64).expand(Np, 3, 3).reshape(Np, 9)
    S = AORT.AORTSolver(x, v, F0, m64, vol, lat.A, lat.Ai, mu, lam, [0.0, 0.0, -9.8], 0.04, 1, "mid", 1, 1, 0, floor=None, det=True)
    out = {}
    for mode in ("0", "1"):
        os.environ["AO_DET_P2GC"] = mode
        AORT.frame_vmax(S, 0.0)
        nn = int(S.IP[9]); out[mode] = (S.mn[:nn].clone(), S.mv[:nn].clone(), int(S.IP[13]))
    a, b = out["0"], out["1"]
    dm = (a[0] - b[0]).abs(); dv = (a[1] - b[1]).abs().max(1).values
    print("프레임", fr, "넘침", a[2], b[2], "질량 합", float(a[0].sum()), float(b[0].sum()), "질량 최대 상대 차 %.2e" % float(dm.max() / a[0].max()), "운동량 최대 차 %.2e" % float(dv.max()), "|v| 최대 %.2f" % float(a[1].norm(dim=1).max() / a[0].clamp_min(1e-30).max()), flush=True)
