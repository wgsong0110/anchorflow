"""절 AO 결정적 판 대조 (모래 AORTSolver·탄성 RTAdapt): 같은 정답 상태에서 비결정·결정적 판을 step() 한 번 (그래프 캡처 데움 실행 포함) 풀고
(1) 입자 질량·부피·구속 끝 시각이 원래 입자에 그대로 붙어 있는지 (perm 으로 되돌려 입력과 비트 비교), (2) 노드 질량 상대 차, (3) 한 프레임 뒤 입자 위치 차.
사용: python exe/fem_ao_dettest.py <data.pt> <셀 배수> <K> <뉴턴> <PCG> <프레임,...> [pbc]"""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch
from anchorflow import deform_model as DM, phys_resid as PR, simplex as SX, fem_ao_rt as AORT, fem_ao_cuda as AO, fem_rt_cuda as RTC
RTC.RAWPTR[0] = True
dp, cell, K, nt, cg = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
frames = [int(q) for q in sys.argv[6].split(",")]; PBC = len(sys.argv) > 7
d = torch.load(dp, map_location="cpu", weights_only=False); cfg = d["cfg"]; dev = "cuda"; h = float(cfg["frame_dt"])
m64 = DM.traj_mass(d, dev).double(); vol = m64 / float(cfg["density"])
mu, lam = PR.lame(cfg["E"], cfg["nu"])
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100)); nh = cell * DXG * math.sqrt(2.0)
lat = SX.lattice(nh, SX.HZ_CUBE * nh, device=dev, dtype=torch.float64)
SAND = cfg.get("material") == "sand"
for fr in frames:
    x = d["x"][fr].to(dev); v = (d["x"][fr + 1].to(dev) - x) / h; Np = x.shape[0]
    F0 = torch.eye(3, device=dev, dtype=torch.float64).expand(Np, 3, 3).reshape(Np, 9)
    trel = (torch.linspace(0.0, 2.0, Np, device=dev, dtype=torch.float64)[torch.randperm(Np, generator=torch.Generator().manual_seed(0)).to(dev)] if PBC else None)
    res = {}
    for det in (False, True):
        if SAND:
            S = AORT.AORTSolver(x, v, F0, m64, vol, lat.A, lat.Ai, mu, lam, [0.0, 0.0, -9.8], h, K, "mid", nt, cg, 0, floor=None, cdp=AO.cdp_of(cfg, float(mu), float(lam))[0],
                                trel=trel, dt_gt=float(cfg["substep_dt"]), det=det)
        else:
            S = AORT.RTAdapt(x, v, F0, m64, vol, lat.A, lat.Ai, mu, lam, [0.0, 0.0, -9.8], h, K, "mid", nt, cg, 1, floor=None, compact=True, warm=True, det=det)
        S.step(fr * h)
        o = lambda t: (lambda z: (z.__setitem__(S.perm, t), z)[1])(torch.empty_like(t))
        ok_m = bool((o(S.m) == m64.float()).all()); ok_v = bool((o(S.vol) == vol.float()).all())
        ok_t = bool((o(S.trel) == trel.float()).all()) if (PBC and SAND) else None
        nn = int(S.IP[9])
        res[det] = (S.mn[:nn].clone(), S.x_orig().clone(), ok_m, ok_v, ok_t)
    a, b = res[False], res[True]
    dm = (a[0] - b[0]).abs().max() / a[0].abs().max() if a[0].shape == b[0].shape else float("nan")
    dx = (a[1] - b[1]).norm(dim=-1)
    print(f"프레임 {fr}: 질량·부피·끝 시각 제자리 (비결정/결정) {a[2]}/{b[2]} {a[3]}/{b[3]} {a[4]}/{b[4]}; 노드 질량 최대 상대 차 {float(dm):.2e}; "
          f"한 프레임 뒤 위치 차 최대 {float(dx.max()):.2e} m 평균 {float(dx.mean()):.2e} m", flush=True)
