"""절 AN 결정적 판 대조 (implicit-wolf exe/fem_det_p2gtest.py 의 생각을 넓힘): 같은 정답 상태에서 비결정 판 (RTSolver) 과 결정적 판 (DetRT) 을 한 프레임씩 풀고
(1) P2G 산출 (노드 질량·운동량, 가우스 점 F·부피·질량, 고착 질량 몫), (2) MG 거친 층 (가우스 점 F·부피, 질량, 대각 역행렬, 헤시안),
(3) 같은 y 에서 요소 기울기 f64·노드 대각·헤시안, (4) 같은 벡터의 고운 층·거친 층 헤시안 곱, (5) 한 프레임 뒤 입자 위치 (원래 순번) 와 정답 대비 오차 를 노드별로 비교한다.
상대 차 = max|a − b| / max|a| (와 평균 |a − b| / 평균 |a|). float32 누적이라 반올림 수준은 대략 1e-6 에서 1e-5.
사용: python exe/fem_an_dettest.py <data.pt> <셀 배수> <MG 층> <뉴턴> <PCG> <프레임,...>"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np                                                                    # noqa: E402
import torch                                                                         # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_an_det as ADET                                           # noqa: E402
from anchorflow import fem_an_rt as ART                                             # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

RTC.RAWPTR[0] = True
P, i32, f = ART.P, np.int32, np.float32
dpath, cell, mgl, nt, cg = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
frames = [int(q) for q in sys.argv[6].split(",")]
dev = "cuda"
d = torch.load(dpath, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m64 = DM.traj_mass(d, dev).double()
vol64 = m64 / float(cfg["density"])
mu, lam = PR.lame(cfg["E"], cfg["nu"])
EXT = float(d["_ext"])
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
NH = cell * DXG * math.sqrt(2.0)
lat = SX.lattice(NH, SX.HZ_CUBE * NH, device=dev, dtype=torch.float64)
pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
NR = np.array(pl["normal"], dtype=np.float64); NR = NR / np.linalg.norm(NR)
PT = [float(q) for q in pl["point"]]
FL = dict(PT=PT, NR=NR.tolist(), PN=float(np.dot(PT, NR)), DXG=DXG, KEX=h / float(cfg["substep_dt"]), vout=True)
FALL = DM.traj_F(d, h, dev)


def rel(a, b):
    a = a.double().reshape(-1); b = b.double().reshape(-1)
    da = (a - b).abs()
    return float(da.max() / a.abs().max().clamp_min(1e-300)), float(da.mean() / a.abs().mean().clamp_min(1e-300))


def line(name, a, b):
    mx, mn = rel(a, b)
    flag = "" if mx < 1e-4 else "  <-- 큼"
    print(f"    {name:34s} 최대 상대 차 {mx:.2e}, 평균 {mn:.2e}{flag}", flush=True)
    return mx


for fr in frames:
    x = d["x"][fr].to(dev); v = d["v"][fr].to(dev)
    Np = x.shape[0]
    F = FALL[fr].to(dev).double().reshape(Np, 9)
    kw = dict(floor=FL, mg_levels=mgl, ls_n=10, descent=True)
    args = (x, v, F, m64, vol64, lat.A, lat.Ai, mu, lam, [float(q) for q in cfg["g"]], h, 1, "mid", nt, cg, 1)
    S0 = ART.RTSolver(*args, **kw)
    S1 = ADET.DetRT(*args, det=True, **kw)
    for S in (S0, S1):
        S.step(fr * h)
    print(f"## 프레임 {fr} (셀 {cell}, MG 층 {mgl}, 뉴턴 {nt} x PCG {cg}): 노드 {int(S0.IP[9])} / {int(S1.IP[9])}, 가우스 점 {int(S0.IP[12])} / {int(S1.IP[12])}, 넘침 {int(S0.IP[13])} / {int(S1.IP[13])}", flush=True)
    nn, nq = int(S0.IP[9]), int(S0.IP[12])
    worst = 0.0
    print("  (1) P2G", flush=True)
    for nm, a, b in (("노드 질량 mn", S0.mn[:nn], S1.mn[:nn]), ("노드 운동량 mv", S0.mv[:nn], S1.mv[:nn]), ("가우스 점 F", S0.Fq[:nq], S1.Fq[:nq]),
                     ("가우스 점 부피 Vq", S0.Vq[:nq], S1.Vq[:nq]), ("가우스 점 질량 mq", S0.mq[:nq], S1.mq[:nq]), ("노드 고착 질량 몫 mqn", S0.mqn[:nn], S1.mqn[:nn])):
        worst = max(worst, line(nm, a, b))
    if mgl > 1:
        print("  (2) MG 거친 층 (프레임 시작 F 로 구성)", flush=True)
        for l in range(1, min(S0.nl, S1.nl)):
            L0, L1 = S0.ML[l - 1], S1.ML[l - 1]
            nnl, nql = int(S0.LM[l * 8 + 3]), int(S0.LM[l * 8 + 7])
            for nm, a, b in ((f"층 {l} 가우스 점 F", L0["Fq"][:nql], L1["Fq"][:nql]), (f"층 {l} 부피", L0["Vq"][:nql], L1["Vq"][:nql]),
                             (f"층 {l} 질량", L0["mc"][:nnl], L1["mc"][:nnl]), (f"층 {l} 대각 역행렬 D", L0["D"][:nnl], L1["D"][:nnl])):
                worst = max(worst, line(nm, a, b))
    # (3) 같은 y 에서 요소
    print("  (3) 같은 y (비결정 판 걸음 끝 y) 에서 요소", flush=True)
    yy = S0.y.clone()
    capQ = S0.capQc * 8
    res = []
    for S in (S0, S1):
        S.y.copy_(yy); S.f.zero_(); S.B.zero_(); S.SC.zero_()
        S.k("rt_elem", capQ, (P(S.IP), P(S.cb), P(S.BG), P(S.Fq), P(S.Vq), P(S.y), f(S.mu), f(S.lam), i32(1 | 2 | 8 | 4), P(S.SC[10:]), P(S.f), P(S.H), i32(capQ), P(S.B)))
        torch.cuda.synchronize()
        res.append((S.f[:nn].clone(), S.B[:nn].clone(), S.H[:, :nq].clone(), float(S.SC[10])))
    worst = max(worst, line("요소 기울기 f (f64)", res[0][0], res[1][0]))
    worst = max(worst, line("노드 대각 B", res[0][1], res[1][1]))
    worst = max(worst, line("가우스 점 헤시안 H", res[0][2], res[1][2]))
    print(f"    탄성 에너지 {res[0][3]:.10e} / {res[1][3]:.10e} (상대 차 {abs(res[0][3] - res[1][3]) / max(abs(res[0][3]), 1e-300):.2e})", flush=True)
    # (4) 헤시안 곱 (같은 H 를 쓰게 결정적 판에 비결정 판 H 를 복사)
    print("  (4) 같은 벡터의 헤시안 곱 (결정적 판에 비결정 판 H 복사)", flush=True)
    S1.H.copy_(S0.H)
    g = torch.Generator(device=dev).manual_seed(0)
    dy = torch.randn(nn, 3, device=dev, generator=g).float() * float(lat.A[:, 0].norm())
    outs = []
    for S in (S0, S1):
        S.Tp.zero_(); S.Tp[:nn].copy_(dy); S.KTp.zero_()
        S.k("rt_mv", capQ, (P(S.IP), P(S.cb), P(S.BG), P(S.Fq), P(S.Vq), P(S.H), P(S.Tp), P(S.KTp), i32(capQ)))
        torch.cuda.synchronize()
        outs.append(S.KTp[:nn].clone())
    worst = max(worst, line("고운 층 헤시안 곱", outs[0], outs[1]))
    if mgl > 1:
        for l in range(1, min(S0.nl, S1.nl)):
            L0, L1 = S0.ML[l - 1], S1.ML[l - 1]
            nnl = int(S0.LM[l * 8 + 3])
            L1["H"].copy_(L0["H"]); L1["Fq"].copy_(L0["Fq"]); L1["Vq"].copy_(L0["Vq"])
            z = torch.randn(nnl, 3, device=dev, generator=g).float()
            oo = []
            for S, L in ((S0, L0), (S1, L1)):
                L["z"].zero_(); L["z"][:nnl].copy_(z); L["KT"].zero_()
                S.k("rt_mg_mv", L["capq"], (i32(l), P(S.LM), P(S.BG), f(0.5 ** l), P(L["Fq"]), P(L["Vq"]), P(L["H"]), i32(L["capq"]), P(L["z"]), P(L["KT"])))
                torch.cuda.synchronize()
                oo.append(L["KT"][:nnl].clone())
            worst = max(worst, line(f"층 {l} 헤시안 곱", oo[0], oo[1]))
    # (5) 한 프레임 뒤
    x0, x1 = S0.x_orig(), S1.x_orig()
    gt = d["x"][fr + 1].to(dev).float()
    e0 = float((x0 - gt).norm(dim=-1).mean()) / EXT; e1 = float((x1 - gt).norm(dim=-1).mean()) / EXT
    dx = float((x0 - x1).norm(dim=-1).max()) / EXT
    print(f"  (5) 한 프레임 뒤: 입자 위치 최대 차 / EXT {dx:.2e}, 정답 대비 오차 {100 * e0:.4f} / {100 * e1:.4f} %", flush=True)
    print(f"  => 단계별 최대 상대 차 중 가장 큰 값 {worst:.2e} ({'반올림 수준' if worst < 1e-4 else '반올림보다 큼 -- 확인 필요'})", flush=True)
