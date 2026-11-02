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
x = d["x"][int(sys.argv[3])].to(dev); v = (d["x"][int(sys.argv[3]) + 1].to(dev) - x) / float(cfg["frame_dt"])
Np = x.shape[0]; F0 = torch.eye(3, device=dev, dtype=torch.float64).expand(Np, 3, 3).reshape(Np, 9)
S = AORT.AORTSolver(x, v, F0, m64, vol, lat.A, lat.Ai, mu, lam, [0.0, 0.0, -9.8], 0.04, 1, "mid", 1, 1, 0, floor=None, det=True)
out = {}
for mode in ("0", "1"):
    os.environ["AO_DET_P2GC"] = mode
    S.k("rt_reset", 1, (AORT.P(S.IP),), b=1)
    vm = AORT.frame_vmax(S, 0.0)
    nn = int(S.IP[9]); out[mode] = (S.mn[:nn].clone(), S.mv[:nn].clone(), int(S.IP[11]), S.IP[:16].clone())
a, b = out["0"], out["1"]
dm = (a[0] - b[0]).abs(); dv = (a[1] - b[1]).abs().max(1).values
print("칸 수", a[2], b[2], "IP 같음", bool((a[3] == b[3]).all()))
print("질량 합", float(a[0].sum()), float(b[0].sum()), "최대 차", float(dm.max()), "상대", float(dm.max() / a[0].max()))
print("운동량 최대 차", float(dv.max()), "노드 수 질량 > 0:", int((a[0] > 0).sum()), int((b[0] > 0).sum()))
bad = torch.nonzero(dm > 1e-6 * a[0].max()).flatten()[:10]
nb = S.IP[6:9].tolist()
for n in bad.tolist():
    print("  노드", n, "좌표", (n // (nb[1] * nb[2]), (n // nb[2]) % nb[1], n % nb[2]), "질량", float(a[0][n]), float(b[0][n]))
