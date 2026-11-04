"""절 AO 실시간 판 롤아웃 (lib/anchorflow/fem_ao_rt.py = 절 AN 실시간 판 + 모래). exe/fem_an_rt_sim.py 를 복사해 모래·구속·거의 빈 노드 인자를 더함. 원래 설명: 절 AN 실시간 판 롤아웃 (lib/anchorflow/fem_an_rt.py): 프레임마다 CUDA 그래프 재생 한 번 (P2G -> 고정 예산 풀이 -> G2P), 블록 야코비 전처리.
정의는 exe/fem_an_sim.py --fast --pc jacobi 와 같다 (정밀도: 곱·누적 f32, 기울기·내적·에너지 f64). 프레임 시간 = 재생 GPU 시간 (CUDA 이벤트), 오차 평가는 재생 밖."""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np                                                                    # noqa: E402
import torch                                                                         # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_an_rt as ART                                             # noqa: E402
from anchorflow import fem_ao_rt as AORT                                            # noqa: E402
from anchorflow import fem_ao_cuda as AO                                            # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--cell", type=float, default=1.0)
ap.add_argument("--floor", choices=["vel", "none"], default="vel")
ap.add_argument("--cfg_bc", action="store_true")
ap.add_argument("--region_density", action="store_true")
ap.add_argument("--pg_vdamp", action="store_true")
ap.add_argument("--K", type=int, default=1)
ap.add_argument("--integ", choices=["mid", "be"], default="mid")
ap.add_argument("--newton", type=int, default=2)
ap.add_argument("--cg", type=int, default=5)
ap.add_argument("--ls", type=int, default=1)
ap.add_argument("--no_vfl_out", action="store_true")
ap.add_argument("--mg_levels", type=int, default=1, help="1 = 블록 야코비, 2 이상 = 그 층 수의 기하 멀티그리드 (거친 층 조밀 상자)")
ap.add_argument("--mg_nc", type=int, default=8)
ap.add_argument("--mg_pow", type=int, default=6)
ap.add_argument("--dump", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--descent", action="store_true", help="하강 방향 보장: g·δ ≥ 0 이면 δ ← α0 M r0 (PCG 첫 방향)")
ap.add_argument("--diag", action="store_true", help="프레임마다 뉴턴 반복별 진단 (에너지·|g|·g·δ·보폭·PCG 잔차·고착 불일치) 을 fem.jsonl 에")
ap.add_argument("--prof", type=int, default=0, help="이 프레임마다 커널별 시간 (그래프 밖 한 번, 상태 되돌림) 을 fem.jsonl 에")
ap.add_argument("--pbc", action="store_true", help="정답 config 입자 속도 구속 (순차 방출, 속도 0)")
ap.add_argument("--pbc_ns", type=int, default=4)
ap.add_argument("--mlow", type=float, default=1e-8, help="거의 빈 노드 문턱 = mlow x 첫 프레임 노드 질량 중앙값 (관성 예측 고정)")
ap.add_argument("--dt_step", type=float, default=0.0, help="공통 걸음 간격 (s): 주면 K = ceil(frame_dt / dt_step) (K 인자 무시)")
ap.add_argument("--adaptK", type=float, default=0.0, help="적응 걸음 수: K = ceil(v_max · frame_dt / (c · 칸)), c = 이 값 (0 = 끔), v_max = 프레임 시작 노드 속도 최대")
ap.add_argument("--kmax", type=int, default=0, help="걸음 수 상한 (0 = 정답 MPM 프레임당 서브스텝 수 frame_dt / substep_dt; 발산 판정과 따로)")
ap.add_argument("--no_compact", action="store_true")
ap.add_argument("--h16", action="store_true", help="헤시안 위 삼각 f16 저장 (--hpack --mvc 와 함께; 결과가 바뀜 -- 대조용)")
ap.add_argument("--mvc", action="store_true", help="헤시안 곱 칸 공유 메모리 판 (--hpack 과 함께)")
ap.add_argument("--bdsh", action="store_true", help="모래 요소 커널 노드 대각을 공유 메모리 판으로 (합 순서 바뀜 -- 반올림 수준)")
ap.add_argument("--hspec", action="store_true", help="헤시안을 주값 분해 (U, F V, 계수 12) 30 개로 저장, 곱·노드 대각·힘을 칸 공유 메모리로 (ao_elem_s·ao_mv_q; 합 순서 바뀜 -- 반올림 수준)")
ap.add_argument("--check_hspec", type=str, default="", help="이 프레임들 (쉼표) 뒤 기존 판과 주값 분해 판의 힘·대각·곱 상대 차를 출력")
ap.add_argument("--bench_at", type=str, default="", help="이 프레임들 (쉼표) 뒤 곱·요소 커널 변형 벤치 (회당 ms)")
ap.add_argument("--hpack", action="store_true", help="모래 판 헤시안 위 삼각 45 개 저장·곱 (결과 불변)")
ap.add_argument("--det", action="store_true", help="결정적 합산 (lib/anchorflow/fem_det_rt.py): 같은 명령 두 번이 비트 단위로 같게")
ap.add_argument("--ao_fast", action="store_true", help="탄성 씬에도 노드 압축·PCG 첫 추정 (절 AO.11; 기본은 절 AN 그대로)")
ap.add_argument("--no_warm", action="store_true")
ap.add_argument("--F0_eye", action="store_true", help="초기 탄성 F = I (정답 MPM 시작 상태; 자료 F 가 항등이라 위치에서 되살린 F 대신)")
ap.add_argument("--tolN", type=float, default=-1.0, help="공통 허용치 (절 AN.14 규칙): 뉴턴 |g_k| <= εN |T f(y0)| 이면 걸음 끝 (--newton 은 상한, 음수 = 끔)")
ap.add_argument("--tolP", type=float, default=-1.0, help="PCG |r| <= εP min(|g_k|, |T f(y0)|) 이면 풀이 끝 (--cg 는 상한)")
ap.add_argument("--nanchk", type=int, default=0, help="진단: 이 프레임 (1 부터) 을 그래프 밖에서 커널마다 비유한 검사 (그 프레임은 재생 대신)")
ap.add_argument("--nhb_eps", type=float, default=0.0, help="수박 (NHB): 뒤집힌 시험 상태 바닥 -- 주값 σ̂ = max(σ, ε) 에서 RM·응력·접선, 커밋·G2P 도 (0 = 끔, 정답 식 그대로)")
ap.add_argument("--no_vfill", action="store_true", help="질량 0 노드 속도·가속도 이웃 평균 끔 (예전 판: 0)")
ap.add_argument("--mat_as", type=str, default="", help="진단: config 재질 이름을 바꿔 돌림 (jelly = 절 AN 탄성 판 RTAdapt, plasticine 등)")
ap.add_argument("--no_rm", action="store_true", help="진단: 재질 판 RM 끔 (탄성 응력만, prm[15])")
ap.add_argument("--ys_big", action="store_true", help="진단: 항복 응력을 아주 크게 (소성 없이 탄성만)")
ap.add_argument("--visc_rpic", action="store_true", help="정답 config rpic_damping r -> 등가 점성 η = ρ r (2−r) Δx_gt² / (16 Δt_gt) (절 AG.20)")
ap.add_argument("--floor_plane", type=str, default="", help="바닥 평면 px,py,pz,nx,ny,nz (주면 config surface_collider 대신; 수박 = GF bounding_box)")
ap.add_argument("--floor_slip", action="store_true", help="미끄럼 바닥 (들어가는 법선 성분만 고착 몫)")
ap.add_argument("--ga", choices=["hht", "ga"], default=None, help="HHT-α 또는 generalized-α (절 AN.err3 정의 + 처치 셋: 시작점 걸음 시작 위치, 끝 위치 Newmark 식, 고착·구속 노드 가속도 0)")
ap.add_argument("--rho", type=float, default=0.7)
ap.add_argument("--ga_y0ut", action="store_true", help="처치 (a) 끔: 뉴턴 시작점 관성 예측")
ap.add_argument("--ga_noar", action="store_true", help="처치 (b) 끔: 고착·구속 노드 가속도 0 안 함")
ap.add_argument("--aq_sand", action="store_true", help="진단 (절 AR): 모래 씬을 재질 커널 (aq_elem·aq_commit·aq_g2p, 차분 접선) 로 -- 해석 접선 모래 커널과 대조")
ap.add_argument("--check_aq", type=int, default=0, help="진단 (절 AR): 이 프레임 (1 부터) 을 그래프 밖에서 돌리며 요소 커널 호출마다 접선 곱 대 힘 중심 차분·대칭·(모래면) 해석 커널 대조")
ap.add_argument("--ls_inv", action="store_true", help="절 AS: 뉴턴 보폭 = 모든 가우스 점 (부피 > 0) 의 J > 0 이 [0, 보폭] 에서 유지되는 최대 2^-k (IPC 식 뒤집힘 없는 선탐색 필터; 풀리는 식은 그대로)")
ap.add_argument("--y0u", action="store_true", help="절 AS: --y0z 의 기준을 걸음 시작 위치 U 로 (y = U 이면 X = 커밋된 F_q; z0 기준은 공중에서도 뒤집힌 점이 있었다)")
ap.add_argument("--y0l", action="store_true", help="절 AS: 시작점 보간에 거의 빈 노드 (관성 예측 고정) 도 넣음 -- 고정 위치 = 보간된 시작점 (보폭 1 이면 그대로)")
ap.add_argument("--fq_rl", action="store_true", help="절 AS: 뒤집힌 입자가 있는 가우스 점 F_q 의 유클리드 평균 대체를 그 평균이 뒤집히지 않을 때만 (뒤집히면 회전·로그 평균)")
ap.add_argument("--ls_end", action="store_true", help="절 AS: 보폭·시작점 필터가 걸음 끝 변위 U_{n+1} (HHT 외삽) 의 X 도 J > 0 로 유지")
ap.add_argument("--y0z", action="store_true", help="절 AS: 뉴턴 시작점 = 영속도 위치 y_z (z0) 에서 지금 시작점 (관성 예측·고착) 쪽으로 뒤집힘 없는 최대 2^-k 만큼 (뒤집힘 없으면 그대로)")
ap.add_argument("--theta", type=float, default=None, help="θ 방식 (절 AN.err2 정의; 0.5 = 중점, 1 = 후퇴 오일러). 안 주면 --integ 그대로 (비트 동일)")
a = ap.parse_args()
if a.adaptK > 0:
    a.K = 8                                                                       # 버퍼·첫 캡처는 상한으로, 프레임마다 다시 정함
dev = "cuda"
RTC.RAWPTR[0] = True
os.makedirs(a.out, exist_ok=True)
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
if a.mat_as:
    cfg = dict(cfg, material=a.mat_as)
h = float(cfg["frame_dt"])
DM.REGION_DENSITY[0] = bool(a.region_density)
m64 = DM.traj_mass(d, dev).double()
EXT = float(d["_ext"])
rho = (d["_rho"].to(dev).double()) if a.region_density else None
vol64 = (m64 / rho) if a.region_density else (m64 / float(cfg["density"]))
mu, lam = PR.lame(cfg["E"], cfg["nu"])
Np = int(d["x"].shape[1])
L = min(a.frames, int(d["x"].shape[0]) - 1)
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
NODE_H = a.cell * DXG * math.sqrt(2.0)
lat = SX.lattice(NODE_H, SX.HZ_CUBE * NODE_H, device=dev, dtype=torch.float64)
if a.dt_step > 0:
    a.K = int(math.ceil(h / a.dt_step - 1e-9))
dt = h / a.K
VD_S = float(cfg.get("grid_v_damping_scale", 1.0))
DAMP = (VD_S ** (dt / float(cfg["substep_dt"]))) if (a.pg_vdamp and VD_S < 1.0) else 1.0
FL = None
if a.floor == "vel":
    pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
    NR = np.array(pl["normal"], dtype=np.float64); NR = NR / np.linalg.norm(NR)
    PT = [float(q) for q in pl["point"]]
    if a.floor_plane:
        _fp = [float(q) for q in a.floor_plane.split(",")]
        PT = _fp[:3]; NR = np.array(_fp[3:], dtype=np.float64); NR = NR / np.linalg.norm(NR)
    FL = dict(PT=PT, NR=NR.tolist(), PN=float(np.dot(PT, NR)), DXG=DXG, KEX=dt / float(cfg["substep_dt"]), vout=not a.no_vfl_out)
BC = None
if a.cfg_bc:
    BC = []
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") == "cuboid":
            t0, t1 = float(bc.get("start_time", 0.0)), float(bc.get("end_time", 1e3))
            BC.append(([float(q) for q in bc["point"]], [float(q) for q in bc["size"]], [float(q) for q in bc["velocity"]], t0, t1))
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") == "cuboid" and int(bc.get("reset", 0)) == 1:
            t1 = float(bc.get("end_time", 1e3))
            BC.append(([0.0, 0.0, 0.0], [1e9, 1e9, 1e9], [0.0, 0.0, 0.0], t1, t1 + 15.0 * h))
    BC = BC or None
F0 = (torch.eye(3, device=dev, dtype=torch.float64).reshape(1, 9).repeat(Np, 1) if a.F0_eye else DM.traj_F(d, h, dev)[0].to(dev).double().reshape(Np, 9))
SAND = cfg.get("material") == "sand"
TREL = None
PVB = None
if a.pbc:                                                                              # exe/fem_ao_sim.py 와 같은 구속 입자·끝 시각
    _x0b = d["x"][0].to(dev).double(); _bcs = []
    for _bc in (cfg.get("boundary_conditions") or []):
        _ty = _bc.get("type")
        if _ty == "release_particles_sequentially":
            _nl, _nv = 50, _bc["normal"]; _pt, _sz, _ax = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], -1
            for _k in range(3):
                if _nv[_k] == 0:
                    _pt[_k], _sz[_k] = 1.0, 1.0
                else:
                    _ax, _pt[_k] = _k, float(_bc["end_position"])
            _hl = abs(float(_bc["start_position"]) - float(_bc["end_position"])) / _nl
            for _k in range(_nl):
                _s2 = list(_sz); _s2[_ax] = _hl * (_nl - _k)
                _bcs.append((list(_pt), _s2, float(_bc["end_time"]) / _nl * (_k + 1)))
        elif _ty == "enforce_particle_translation":
            assert float(_bc["start_time"]) <= 0.0
            _bcs.append((_bc["point"], _bc["size"], float(_bc["end_time"]), [float(q) for q in _bc["velocity"]]))
    TREL = torch.full((Np,), -1e30, device=dev, dtype=torch.float64)
    PVB = torch.zeros(Np, 3, device=dev, dtype=torch.float64)
    for b_ in _bcs:
        m_ = ((_x0b - torch.tensor(b_[0], device=dev, dtype=torch.float64)).abs() < torch.tensor(b_[1], device=dev, dtype=torch.float64)).all(-1)
        TREL = torch.where(m_, torch.maximum(TREL, torch.tensor(b_[2], device=dev, dtype=torch.float64)), TREL)
        if len(b_) > 3:
            PVB[m_] = torch.tensor(b_[3], device=dev, dtype=torch.float64)
    if not bool((PVB != 0).any()):
        PVB = None
# 거의 빈 노드 문턱: 첫 프레임 노드 질량 (2 차 B-스플라인 P2G) 중앙값 x mlow
_y = d["x"][0].to(dev).double() @ lat.Ai.T
_b = torch.floor(_y - 0.5).long(); _fx = _y - _b; _b = _b - _b.min(0).values; _nb = (_b.max(0).values + 3).tolist()
_w = torch.stack([0.5 * (1.5 - _fx) ** 2, 0.75 - (_fx - 1) ** 2, 0.5 * (_fx - 0.5) ** 2], 1)
_mn = torch.zeros(_nb[0] * _nb[1] * _nb[2], device=dev, dtype=torch.float64)
for _i in range(3):
    for _j in range(3):
        for _k in range(3):
            _mn.index_add_(0, ((_b[:, 0] + _i) * _nb[1] + _b[:, 1] + _j) * _nb[2] + _b[:, 2] + _k, _w[:, _i, 0] * _w[:, _j, 1] * _w[:, _k, 2] * m64)
MTHR = float(a.mlow * _mn[_mn > 0].median()) if a.mlow > 0 else 0.0
del _y, _b, _fx, _w, _mn
MATK = {"foam": 3, "plasticine": 5, "watermelon": 7}.get(cfg.get("material"))
if MATK is not None:                                                                    # 절 AQ: 재질 판 (정답 pl_rm·pl_pi)
    from anchorflow import fem_pl_cuda as PLC
    _mat, PRM, PRD = PLC.params(cfg, float(mu), float(lam), dev)
    assert _mat == MATK
    if a.ys_big:
        PRD["yield_stress"] = 1e30
    if a.no_rm:
        PRM[15] = 1.0
    if a.nhb_eps > 0:
        PRM[14] = float(a.nhb_eps)
    VISC = 0.0
    if a.visc_rpic and float(cfg.get("rpic_damping", 0.0)) > 0:
        _r = float(cfg["rpic_damping"])
        VISC = float(cfg["density"]) * _r * (2 - _r) * DXG ** 2 / (16.0 * float(cfg["substep_dt"]))
    print(f"[절 AQ 재질 판] 재질 {cfg.get('material')} ({MATK}) 항복 {PRD['yield_stress']} logJp0 {PRD['logJp0']} 등가 점성 {VISC:.4g} 미끄럼 {a.floor_slip} 구속 속도 {PVB is not None}", flush=True)
    S = AORT.AORTSolver(d["x"][0].to(dev), d["v"][0].to(dev), F0, m64, vol64, lat.A, lat.Ai, mu, lam, [float(q) for q in cfg["g"]], h, a.K, a.integ, a.newton, a.cg, 0,
                        floor=FL, damp=DAMP, bc=BC, cdp=1.0, trel=TREL, pbc_ns=a.pbc_ns, dt_gt=float(cfg["substep_dt"]), mthr=MTHR, compact=not a.no_compact, warm=not a.no_warm, det=a.det,
                        hspec=True, theta=a.theta, ga=(dict(kind=a.ga, rho=a.rho, y0uk=not a.ga_y0ut, ar=not a.ga_noar) if a.ga else None),
                        mat=MATK, prm=PRM, eta=VISC, pvb=PVB, ys0=PRD["yield_stress"], lj0=(PRD["logJp0"] if MATK == 7 else 0.0), slip=a.floor_slip, tolN=a.tolN, tolP=a.tolP, vfill=not a.no_vfill, lsinv=a.ls_inv, y0z=a.y0z, y0u=a.y0u, y0l=a.y0l, fqrl=a.fq_rl, lsend=a.ls_end, mg_levels=a.mg_levels, mg_nc=a.mg_nc, mg_pow=a.mg_pow)
elif SAND:
    CDP = AO.cdp_of(cfg, float(mu), float(lam))[0]
    _sk = {}
    if a.aq_sand:                                                                      # 절 AR 진단: 같은 PG 모래를 재질 커널로
        from anchorflow import fem_pl_cuda as PLC
        _m2, _prm2, _ = PLC.params(cfg, float(mu), float(lam), dev)
        assert _m2 == 2
        _sk = dict(mat=2, prm=_prm2, aqk=True, vfill=not a.no_vfill)
        print(f"[절 AR] 모래를 재질 커널로 (cdp {CDP:.5f}, prm α {float(_prm2[3]):.5f} -> (3λ+2μ)α/(2μ) {float((3 * lam + 2 * mu) / (2 * mu) * float(_prm2[3])):.5f})", flush=True)
    S = AORT.AORTSolver(d["x"][0].to(dev), d["v"][0].to(dev), F0, m64, vol64, lat.A, lat.Ai, mu, lam, [float(q) for q in cfg["g"]], h, a.K, a.integ, a.newton, a.cg, 0,
                        floor=FL, damp=DAMP, bc=BC, cdp=CDP, trel=TREL, pbc_ns=a.pbc_ns, dt_gt=float(cfg["substep_dt"]), mthr=MTHR, compact=not a.no_compact, warm=not a.no_warm, det=a.det, hpack=a.hpack, bdsh=a.bdsh, mvc=a.mvc, h16=a.h16, theta=a.theta, hspec=a.hspec, ga=(dict(kind=a.ga, rho=a.rho, y0uk=not a.ga_y0ut, ar=not a.ga_noar) if a.ga else None), mg_levels=a.mg_levels, mg_nc=a.mg_nc, mg_pow=a.mg_pow, tolN=a.tolN, tolP=a.tolP, lsinv=a.ls_inv, y0z=a.y0z, y0u=a.y0u, y0l=a.y0l, fqrl=a.fq_rl, lsend=a.ls_end, **_sk)
else:
  S = AORT.RTAdapt(d["x"][0].to(dev), d["v"][0].to(dev), F0, m64, vol64, lat.A, lat.Ai, mu, lam, [float(q) for q in cfg["g"]], h, a.K, a.integ, a.newton, a.cg, a.ls,
                 floor=FL, damp=DAMP, bc=BC, mg_levels=a.mg_levels, mg_nc=a.mg_nc, mg_pow=a.mg_pow, descent=a.descent,
                 compact=a.ao_fast, warm=a.ao_fast and not a.no_warm, det=a.det)
if a.kmax <= 0:
    a.kmax = int(math.ceil(h / float(cfg["substep_dt"]) - 1e-9))
S.kcap = a.K
CELL = float(lat.A[:, 0].norm())
print(f"[절 AO 실시간 판] 적응 걸음 수 c {a.adaptK} 상한 {a.kmax}, 칸 {CELL:.4f} m")
print(f"[절 AO 실시간 판] 모래 {SAND} 구속 {a.pbc} 빈 노드 문턱 {MTHR:.3e} (mlow {a.mlow})", flush=True)
print(f"[절 AN 실시간 판] 입자 {Np}, 프레임 {L}, 셀 {float(lat.A[:, 0].norm()):.4f} m (정답 dx {DXG} 의 {a.cell} 배), K {a.K} {a.integ}, 뉴턴 {a.newton} x PCG {a.cg}, 선탐색 {a.ls}, MG 층 {a.mg_levels}, "
      f"바닥 {a.floor}, 구속 {len(BC) if BC else 0}", flush=True)
flog = open(os.path.join(a.out, "fem.jsonl"), "w")
_v0 = d["v"][0].double()
print(f"[처음 속도] 평균 {[round(float(q), 4) for q in _v0.mean(0)]} 표준편차 {[round(float(q), 4) for q in _v0.std(0)]} 최대 편차 {float((_v0 - _v0.mean(0)).norm(dim=-1).max()):.4f}", flush=True)
errs, ms = [], []
ITS = []
div_at = None
PRED = torch.empty(L, Np, 3) if a.dump else None
FDMP = torch.empty(L, Np, 3, 3) if a.dump else None
for fi in range(L):
    kp = S.profile(fi * h) if (a.prof and fi >= a.warm and (fi - a.warm) % a.prof == 0) else None
    torch.cuda.synchronize(); _t0 = time.perf_counter()
    if a.adaptK > 0:
        redo, KF, VMX = AORT.adapt_step(S, fi * h, h, CELL, float(cfg["substep_dt"]), a.kmax, a.adaptK)
    else:
        if a.check_aq and fi + 1 == a.check_aq:
            if S.graph is None:
                _bk0 = S._bk(); S.capture(); S._rs(_bk0)
            _ck = S.chkframe(fi * h); S.last_ms = float("nan")
            for _c in _ck:
                print(f"[대조 aq] 프레임 {fi + 1} {json.dumps(_c)}", flush=True)
            redo, KF, VMX = False, a.K, None
        elif a.nanchk and fi + 1 == a.nanchk:
            if S.graph is None:
                _bk0 = S._bk(); S.capture(); S._rs(_bk0)
            _hit = S.nanchk(fi * h); S.last_ms = float("nan")
            print(f"[비유한 검사] 프레임 {fi + 1}: {json.dumps(_hit)}", flush=True)
            redo, KF, VMX = False, a.K, None
        else:
            redo, KF, VMX = S.step(fi * h), a.K, None
    torch.cuda.synchronize(); WALL = 1e3 * (time.perf_counter() - _t0)
    gt = d["x"][fi + 1].to(dev).float()
    xo = S.x_orig()
    _ok = torch.isfinite(gt).all(-1)                                                   # 정답 비유한 입자 (GF quarantine) 는 지표에서 거름
    e = float((xo[_ok] - gt[_ok]).norm(dim=-1).mean()) / EXT
    fin = bool(torch.isfinite(xo).all()) and math.isfinite(e)
    errs.append(e if fin else float("nan")); ms.append(WALL if a.adaptK > 0 else S.last_ms)
    c = S.counts()
    _nn = int(S.IP[9])
    if _nn > 0 and int(S.IP[13]) == 0:                                                 # 발산 진단: 노드 최대 변위 (칸 단위) 와 그 노드 질량 / 중앙값, 입자 최대 속력
        _ua = S.U[:_nn].norm(dim=-1); _ui = int(_ua.argmax()); _mm = S.mn[:_nn]; _md = float(_mm[_mm > 0].median())
        c.update(umax_cell=float(_ua[_ui]) / CELL, umax_mrel=float(_mm[_ui]) / _md, vpmax=float(S.v.norm(dim=-1).max()), mrel_min=float(_mm[_mm > 0].min()) / _md)
    if a.diag:
        c["diag"] = S.diag()
    if "it" in c:
        ITS.append(c["it"])
    flog.write(json.dumps(dict(t=fi, err=e, ms=S.last_ms, wall_ms=WALL, K=KF, vmax=VMX, redo=redo, **c, **({"kernel_ms": kp} if kp else {}))) + "\n"); flog.flush()
    print(f"  프레임 {fi + 1}/{L} K {KF} v_max {VMX if VMX is None else round(VMX, 3)} 벽시계 {WALL:.1f} ms 재생 {S.last_ms:.2f} ms 오차 {100 * e:.3f}%{' (용량 늘려 다시)' if redo else ''} 점 {c['nq']} 노드 상자 {c['nn']} 유클리드 평균 점 {c['nbad']} 꼬리·꼭짓점 점 {c.get('q_tail')}·{c.get('q_apex')} 입자 RM {c.get('p_rm')}{(' 보폭(줄임·최대k·뒤집힌점·호출) ' + str(c.get('ls_inv')) + ' 시작 ' + str(c.get('y0z'))) if 'ls_inv' in c else ''}", flush=True)
    if a.bench_at and str(fi + 1) in a.bench_at.split(","):
        print(f"  [벤치] 프레임 {fi + 1} 점 {int(S.IP[12])} {json.dumps(S.bench(['ao_mv_c', 'ao_mv_q', 'ao_mv_q_na', 'ao_mv_q_ng', 'ao_mv_q_nh', 'ao_elem', 'ao_elem_s']))}", flush=True)
    if a.check_hspec and str(fi + 1) in a.check_hspec.split(","):
        print(f"  [대조 hspec] 프레임 {fi + 1} {json.dumps(S.check_hspec(fi))}", flush=True)
    if PRED is not None:
        PRED[fi] = xo.cpu()
        _Fo = torch.empty_like(S.F); _Fo[S.perm] = S.F
        FDMP[fi] = _Fo.reshape(Np, 3, 3).cpu()
    if not fin or e > 2.0 or float(S.v.norm(dim=-1).max()) > 200.0:
        div_at = fi + 1
        _nf = {}
        for _nm in ("x", "v", "F", "U", "y", "f", "Fq", "Sq", "H", "B", "mn", "ljp", "ysp", "alp", "pa", "gaA"):
            _t = getattr(S, _nm, None)
            if torch.is_tensor(_t) and _t.is_floating_point():
                _nf[_nm] = [int((~torch.isfinite(_t)).sum()), int(_t.numel())]
        print(f"[비유한 배열] {json.dumps(_nf)}", flush=True)
        print(f"[발산] 프레임 {div_at}", flush=True)
        break
sc = [q for i, q in enumerate(ms) if i >= a.warm]
summ = dict(args=vars(a), frames=L, frames_done=len(errs), diverged=div_at, err_mean=(float(np.mean(errs)) if div_at is None else None),
            err_mean_done=float(np.nanmean(errs)), ms_mean=float(np.mean(sc)) if sc else None, ms_min=float(np.min(sc)) if sc else None)
if ITS:                                                                               # 허용치 판 반복 수 분포 (프레임당)
    for _k in ("nN", "nP", "capN", "capP"):
        _v = np.array([q[_k] for q in ITS], dtype=np.float64)
        summ["it_" + _k] = dict(mean=float(_v.mean()), med=float(np.median(_v)), p90=float(np.percentile(_v, 90)), max=float(_v.max()))
_xo = S.x_orig().contiguous()
import hashlib
summ["x_sha1"] = hashlib.sha1(_xo.cpu().numpy().tobytes()).hexdigest()
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
if PRED is not None:
    torch.save({"pred": PRED[:len(errs)], "F": FDMP[:len(errs)]}, os.path.join(a.out, "pred.pt"))
print(f"[롤아웃] {a.tag} 실시간 판 셀 {a.cell} K {a.K} 뉴턴 {a.newton} PCG {a.cg}: {len(errs)}/{L} 프레임{' (발산 %d)' % div_at if div_at else ''}, 평균 {100 * summ['err_mean_done']:.3f}%, "
      f"프레임당 {summ['ms_mean']} ms (재생 GPU 시간)", flush=True)
