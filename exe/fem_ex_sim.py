"""명시적 FEM 단독 시뮬레이터 (사용자 10-28 지시, doc/fem_realtime.md 절 L) -- lib/anchorflow/fem_ex_cuda.py.

암시적 실시간 판 (exe/fem_rt_sim.py) 과 같은 모델 (사면체 적분점 F̄_T, 고정 FCR, 같은 관성·중력·감쇠, 입자 바닥 stick·softplus 매핑,
--cfg_bc·--region_density·--pg_vdamp 같은 정의) 을 뉴턴 대신 symplectic Euler 서브스텝 N 번으로 (h_s = h/N). 식은 fem_ex_cuda 머리말.

  --mode sub : 매 서브스텝 P2G·G2P (MPM 식)
  --mode lag : 프레임 동안 격자 위 라그랑주식 (U 누적, 셀 F = A_T(U) F̄_T), 바닥 후보 입자만 서브스텝마다 보간·매핑 갱신, 프레임 끝 G2P
  --opt      : 최적화 판 -- 셀 점유 격자 (호스트 동기화 프레임당 2 번), P2G·분할·F̄ 커널, 탄성 힘 float32 대수, 크기를 장치에서 읽는 커널로
               서브스텝 N 번 전체를 CUDA 그래프 하나로 (버퍼 용량 미리 할당, 넘치면 늘리고 다시 잡는다), 저장은 측정 밖
  (없으면) 기준 판 -- fem_rt 격자 (GridK) + CudaTetqK (암시적 판과 같은 P2G·분할, float64) + 같은 서브스텝 커널 float64 대수, 서브스텝마다 파이썬 루프
오차 = 프레임마다 입자 평균 |x - x_gt| / EXT 의 L 프레임 평균 (fem_rt_sim 과 같은 지표). 발산 = 값이 유한하지 않거나 바운딩박스 대각/EXT > --div 이면 그 프레임에서 멈춘다.
출력 <out>/: fem.jsonl, summary.json, roll_fem.pt (fem_rt_sim 과 같은 형식).
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np                                                                  # noqa: E402
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import fem_ex_cuda as EX                                            # noqa: E402
from anchorflow import fem_bs_cuda as BSC                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--node_h", type=float, default=0.028284)
ap.add_argument("--floor", choices=["particle", "none", "vel"], default="particle",
                help="vel (절 O, --opt --mode lag 만): 서브스텝마다 속도 단위 바닥 -- 바닥 후보 입자의 지금 보간 위치로 접촉 판정, 걸린 노드 속도에 정답 sticky 고착 + 비관통; 에너지 바닥 힘 끔")
ap.add_argument("--vfl_mode", choices=["frac", "full"], default="frac",
                help="--floor vel 고착 분배: frac = 노드 속도 x (1 - 접촉 입자 질량 몫) (접촉 입자 운동량만 0), full = 접촉 입자가 걸린 노드 속도 0 (정답 노드 sticky 와 같은 꼴)")
ap.add_argument("--vfl_np", type=int, default=1, help="--floor vel: 1 = 고착 뒤 접촉 노드 속도의 안쪽 법선 성분 제거 (비관통), 0 = 끔")
ap.add_argument("--vfl_margin", type=float, default=0.2, help="--floor vel: 바닥 후보 = 프레임 관성 예측 끝 높이 ≤ 이 값 (m) -- 기존 분할 s/ε ≤ 20, ε 0.01 과 같은 0.2")
ap.add_argument("--cfg_bc", action="store_true")
ap.add_argument("--region_density", action="store_true")
ap.add_argument("--pg_vdamp", action="store_true")
ap.add_argument("--tag", default="L000")
ap.add_argument("--eps_sp", type=float, default=0.01)
ap.add_argument("--eps_st", type=float, default=0.01)
ap.add_argument("--mode", choices=["sub", "lag"], default="lag")
ap.add_argument("--N", type=int, required=True, help="프레임당 서브스텝 수")
ap.add_argument("--opt", action="store_true")
ap.add_argument("--dil", type=int, default=-1, help="--opt --mode sub: 셀 팽창 (셀 수; -1 = 프레임 최대 이동으로 자동)")
ap.add_argument("--mass", choices=["bary", "tet"], default="bary",
                help="노드 질량: bary = Σ λ m_p (암시적 판 관성 행렬의 행 합, 기본), tet = 사면체 등분 Σ_T m_T/4 (정의 변경: 선형 사면체 표준 질량 집중)")
ap.add_argument("--near32", action="store_true", help="--opt: 바닥 후보 힘 float32 대수 (암시적 판 --near32 와 같은 방식)")
ap.add_argument("--tprof", type=int, default=-1, help="이 프레임 하나를 torch.profiler 로 (커널별 GPU 시간 표; --no_graph 와 함께)")
ap.add_argument("--no_fuse", action="store_true", help="--opt --mode sub: 융합 커널 끄기 (G2P·locate·P2G·분할 따로, F̄ 나누기·채우기 따로, G2P float64) -- 대조")
ap.add_argument("--no_sort", action="store_true", help="--opt --mode sub 융합 판: 입자 정렬·워프 구간 합 끄기 -- 대조")
ap.add_argument("--no_graph", action="store_true", help="--opt 에서 CUDA 그래프 끄기 (대조)")
ap.add_argument("--div", type=float, default=3.0)
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--no_dump", action="store_true")
ap.add_argument("--dump_f32", action="store_true")
ap.add_argument("--dump_pred", action="store_true")
ap.add_argument("--interp", choices=["bary", "bspline2"], default="bary",
                help="bspline2: MPM 표준 2 차 B-스플라인 (절 M, lib/anchorflow/fem_bs_cuda.py) -- 질량 Σ w m (P2G), G2P PIC, --mode sub 만")
ap.add_argument("--quad", choices=["particle", "cell"], default="particle", help="--interp bspline2 탄성 적분점: 입자 (MPM 식) / 셀 대표 F̄ (K 점)")
ap.add_argument("--psort", action="store_true", help="--interp bspline2 --opt: 프레임마다 입자를 B-스플라인 셀 번호로 정렬 (P2G·G2P 원자 합·읽기 지역성; 계산은 같고 합 순서만 바뀜)")
ap.add_argument("--apic", action="store_true", help="--interp bspline2 --opt: G2P/P2G 를 APIC 로 (정답 PG 와 같은 rpic_damping 0 식, 입자 C 운반; 절 N)")
ap.add_argument("--K", type=int, default=8, help="--quad cell: 셀 안 ∇w 평균 점 수 (1 중심, 8 가우스 2x2x2)")
ap.add_argument("--affine", action="store_true", help="--opt --mode lag: 프레임 끝 G2P 에서 입자 속도 기울기 C 를 남겨 다음 프레임 P2G 를 아핀 전달 (APIC 꼴, 절 N.3)")
ap.add_argument("--near_c0", action="store_true", help="--apic/--affine: 프레임 끝 바닥 후보 입자 (매핑으로 위치·속도가 바뀐 입자) 의 C 를 0 으로 (PIC 로 되돌림, 절 N.3)")
ap.add_argument("--near_diag", default="", help="진단 전용 (절 N.2): 바닥 후보 힘 항 배율 '관성,변위탄성,기울기탄성' (예 0,1,1) -- 원래 식은 1,1,1")
ap.add_argument("--lagdiag", action="store_true", help="--opt --mode lag 진단 (절 N.2): 그래프 없이 서브스텝마다 최대 |v_n| (8 표본·노드 높이·질량), 프레임 끝 사면체 det A_T 분포, 바닥 힘 크기 -> lagdiag.jsonl")
ap.add_argument("--lagdump", type=int, default=0,
                help="--lagdiag 덤프 (사용자 10-29 '날아가는 노드' 그림용, 계산 불변): 마지막 K 프레임의 서브스텝별 노드 U·v (시작 + 매 서브스텝 뒤), "
                     "사면체 det A_T, 노드 접촉 질량 비 α (그 서브스텝 ex_vfl_contact 판정과 같은 식), 노드 위치·질량·점유율, 사면체, 입자 x·rows·λ "
                     "-> 실행 끝 (발산 포함) 에 <out>/lagdump.pt (0 = 끔). 10-30 확장: 서브스텝별 노드 내부력 f (걸음 직전), 프레임 F̄_T·∇λ (gl)·셀 입자 질량 m_T·입자 수·격자 Ai·μ·λ "
                     "(질량 vs 힘 진단, exe/fem_dv_stats.py; --p2g bs 판에도)")
ap.add_argument("--rep", type=int, default=0,
                help="절 P (--opt --mode lag 만, 사용자 10-29): K 서브스텝마다 입자에서 다시 평균 (0 = 끔, 기존 lag 판). 바닥·질량·에너지 정의 그대로")
ap.add_argument("--rep_mode", choices=["full", "vel"], default="full",
                help="--rep: full = G2P (입자 x·v·F 갱신) -> 재배치 P2G (노드 질량·속도·셀 F̄, U = 0); vel = 노드 속도만 (v_p = Σλ v_n -> v_n = Σ w v_p / m_n, 이동·F̄ 그대로)")
ap.add_argument("--rep_w", choices=["std", "tet4", "lam"], default="std",
                help="절 Q (a) 속도 전달 가중 맞추기 (--mode lag --opt --mass tet): std = P2G m_p/4 · G2P λ (절 P 그대로, 불일치), "
                     "tet4 = G2P 속도도 ¼ Σ v_n (사면체 등분, 질량 정의 그대로), lam = P2G 속도 가중도 λ m_p (정규화 Σλm, 동역학 노드 질량 m_T/4 그대로) -- "
                     "재평균·프레임 경계 G2P/P2G 모두")
ap.add_argument("--fb", type=float, default=0.0,
                help="절 R (1) (--opt --mode lag): FCR 에 log J 장벽 κ μ b(J; Ĵ) 더하기 (기존 F 장벽 phys_resid._barrier_b, 연장 linear) -- κ (0 = 끔, 기존 기본 1.0)")
ap.add_argument("--fb_jhat", type=float, default=0.3, help="--fb 장벽이 켜지는 J (기존 --inv_jhat 기본 0.3)")
ap.add_argument("--fb_on", choices=["X", "A"], default="X", help="--fb 의 J: X = det(A_T F̄_T) (셀 변형 전체, 기존 F 장벽 꼴), A = det A_T (프레임 안 셀 사상)")
ap.add_argument("--lim", type=float, default=-1.0,
                help="절 R (2) (--opt --mode lag): 서브스텝마다 노드 이동을 줄여 점유 사면체 det A_T ≥ 이 값을 지킨다 (Kuhn 홀짝 8 색 Gauss-Seidel, 닫힌 꼴 α) -- 음수 = 끔")
ap.add_argument("--lim_mode", choices=["gs", "box"], default="box",
                help="--lim: gs = Kuhn 홀짝 8 색 Gauss-Seidel (노드별 닫힌 꼴 α, 서브스텝당 커널 8 개), box = 사면체별 상자 꼭짓점 판정 (det 이 α_i 마다 아핀 -> 상자 최소 = 꼭짓점; "
                     "노드 α = min s_T, 커널 2 개, Jacobi)")
ap.add_argument("--lim_v", choices=["scale", "keep"], default="scale", help="--lim: scale = 속도도 α 배 (이동과 맞춤, 줄인 운동량·에너지 기록), keep = 속도 그대로")
ap.add_argument("--rstat", action="store_true", help="절 R 통계 (뒤집힌 셀 수 등) -- --fb·--lim 이 없어도 같은 사면체 힘 커널 (ex_tet_force_R, κ 0) 로 (기준 판 비교용)")
ap.add_argument("--p2g", choices=["bary", "bs"], default="bary",
                help="절 S (--opt --mode lag, 사용자 10-29): bs = 프레임 시작 P2G 를 2 차 B-스플라인으로 -- 노드 질량 m_n = Σ w(x_n - x_p) m_p·운동량 Σ w m_p v_p (27 노드), "
                     "셀 F̄_c = Σ w(x_c - x_p) V_p F_p / Σ w V_p·V_c = Σ w V_p (셀 중심, 주변 27 셀; 분할 1 이라 Σ V_c = Σ V_p), 사면체 V_T = V_c/6; 메시 = 점유 셀 1 칸 팽창")
ap.add_argument("--bs_mesh", choices=["dil", "occ"], default="dil",
                help="--p2g bs 메시: dil = 점유 셀 1 칸 팽창 (절 S), occ = 절 R 판과 같은 메시 (입자가 든 셀과 그 꼭짓점만, 팽창 없음; 사용자 10-29) -- "
                     "B-스플라인 가중은 메시에 있는 노드·셀에만 쓰고 입자마다 메시 안 가중 합으로 나눠 정규화 (Σ_n w = 1, Σ_c w = 1; 절 T). G2P (bs)·바닥 접촉 가중도 같은 정규화")
ap.add_argument("--g2p", choices=["bary", "bs"], default="bary", help="--p2g bs 의 프레임 끝 G2P: bary = 사면체 barycentric (기존 ex_g2p_lag), bs = B-스플라인 (질량 가중과 같은 w)")
ap.add_argument("--bsdiag", type=int, default=0, help="--p2g bs 진단 (측정 밖): K 프레임마다 노드 질량 분포 (B-스플라인 / 셀 집중 m_T/4 / Σλm), 국소 안정 비 -> bsdiag.jsonl (0 = 첫 프레임만, -1 = 끔)")
ap.add_argument("--nint", action="store_true",
                help="절 V (--opt --mode lag, 사용자 10-29): 노드 적분 FEM (평균 노드 변형 사면체, Bonet & Burton 1998 / Dohrmann et al. 2000) -- "
                     "서브스텝마다 F_n = Σ_{T∋n} V_T F_T / Σ V_T, P_n = P_FCR(F_n), f_a = -Σ_T (V_T/4) Σ_{n∈T} P_n F̄_Tᵀ ∇λ_a (= -∂/∂U Σ_n V_n ψ(F_n), V_n = Σ V_T/4). "
                     "질량·바닥·F̄·G2P 는 절 R 판 그대로")
ap.add_argument("--nint_beta", type=float, default=0.0,
                help="--nint 영에너지 모드 안정화 선택지: 에너지 (1-β) Σ_n V_n ψ(F_n) + β Σ_T V_T ψ(F_T) (셀 적분 혼합). 0 = 순수 노드 적분")
ap.add_argument("--nint_check", type=int, default=0,
                help="--nint 검산 (측정 밖): 처음 K 프레임 끝 상태에서 커널 힘 vs float64 노드 에너지의 중앙 차분 (방향 미분·노드 성분), 운동량·각운동량 합 -> nint_check.jsonl")
ap.add_argument("--diag", action="store_true", help="첫 프레임: 노드별 국소 안정 걸음 2/ω (ω² = Σ_T V_T (λ+2μ)|∇λ_i|² / m_i) 분포, 서브스텝마다 최대 |v|")
a = ap.parse_args()

dev = "cuda"
os.makedirs(a.out, exist_ok=True)
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
hs = h / a.N
DM.REGION_DENSITY[0] = bool(a.region_density)
m = DM.traj_mass(d, dev).float().contiguous()
EXT = float(d["_ext"])
rho = (d["_rho"].to(dev).float()) if a.region_density else None
vol = ((m / rho) if a.region_density else (m / float(cfg["density"]))).contiguous()
BC_SEL = RT.cfg_bc_sel(cfg, d["x"][0].to(dev).float()) if a.cfg_bc else None
if a.cfg_bc and any(bc.get("type") in ("enforce_particle_translation", "enforce_particle_velocity_rotation") for bc, _ in BC_SEL) and a.opt:
    raise SystemExit("--opt --cfg_bc: 입자 구속은 아직 (cuboid 만)")
VD_S = float(cfg.get("grid_v_damping_scale", 1.0))
DAMP = (VD_S ** (hs / float(cfg["substep_dt"]))) if (a.pg_vdamp and VD_S < 1.0) else 1.0   # 서브스텝당 (프레임 합 = fem_rt_sim 의 VDAMP)
Np = int(d["x"].shape[1])
L = min(int(a.frames), int(d["x"].shape[0]) - 1)
x = d["x"][0].to(dev).float().contiguous()
v = d["v"][0].to(dev).float().contiguous()
F = DM.traj_F(d, h, dev)[0].to(dev).float().reshape(Np, 3, 3).contiguous()
GT = d["x"][1:L + 1].to(dev).float()
gv = [float(q) for q in cfg["g"]]
mu, lam = PR.lame(cfg["E"], cfg["nu"])
planes = [b for b in (cfg.get("boundary_conditions") or []) if b.get("type") == "surface_collider"]
FLOOR = a.floor == "particle" and bool(planes)                                     # 에너지 바닥 (입자 stick·softplus 매핑의 프레임 에너지 힘)
VFL = a.floor == "vel" and bool(planes)                                             # 절 O 속도 단위 바닥
HASPL = FLOOR or VFL
if a.floor in ("particle", "vel") and not planes:
    raise SystemExit(f"--floor {a.floor} 인데 surface_collider 가 없다")
if VFL and not (a.opt and a.mode == "lag"):
    raise SystemExit("--floor vel 은 --opt --mode lag 만")
REP = int(a.rep) if (a.rep and a.rep < a.N) else 0                                   # 절 P: K 서브스텝마다 재평균 (K ≥ N 이면 끔과 같다)
REPF = bool(REP) and a.rep_mode == "full"
if REP and not (a.opt and a.mode == "lag" and a.interp == "bary"):
    raise SystemExit("--rep 은 --opt --mode lag 만")
if a.lagdump and not (a.lagdiag and a.opt and a.mode == "lag"):
    raise SystemExit("--lagdump 은 --opt --mode lag --lagdiag 와 함께")
if REP and (FLOOR or a.lagdiag):
    raise SystemExit("--rep 은 에너지 바닥 (--floor particle)·--lagdiag 와 함께 못 쓴다 (프레임 단위 변위 U 를 0 으로 되돌리므로) -- --floor vel 또는 none")
# 절 Q: --affine + --rep = 재평균 APIC (b) -- full: 블록 G2P 가 C 를 남기고 재배치 P2G 가 m_p (v_p + C_p (x_n - x_p)); vel: 커널 안에서 C (항등이 된다)
RW4 = a.rep_w == "tet4"
RWL = a.rep_w == "lam"
if a.rep_w != "std" and not (a.opt and a.mode == "lag" and a.mass == "tet" and a.interp == "bary"):
    raise SystemExit("--rep_w tet4|lam 은 --opt --mode lag --mass tet 만")
if RW4 and a.affine:
    raise SystemExit("--rep_w tet4 + --affine 은 정의가 맞지 않는다 (APIC G2P 는 λ 보간 + C)")
NINT = bool(a.nint)                                                                  # 절 V: 노드 적분
if NINT and not (a.opt and a.mode == "lag" and a.interp == "bary" and not REP and a.p2g == "bary" and a.fb <= 0 and not a.affine):
    raise SystemExit("--nint 은 --opt --mode lag (재평균·B-스플라인 P2G·장벽·아핀 없음) 만")
if a.nint_check and not NINT:
    raise SystemExit("--nint_check 은 --nint 와 함께")
RNEW = a.fb > 0 or a.lim >= 0 or a.rstat or a.p2g == "bs" or NINT                                         # 절 R 경로 (사면체 힘 ex_tet_force_R + 통계)
LIM = a.lim >= 0
BSL = a.p2g == "bs"                                                                  # 절 S: lag 판 B-스플라인 P2G
G2PBS = BSL and a.g2p == "bs"
BSN = 1 if (BSL and a.bs_mesh == "occ") else 0                                       # 절 T: 팽창 없는 메시 + 메시 안 가중 정규화
if a.bs_mesh == "occ" and not BSL:
    raise SystemExit("--bs_mesh occ 는 --p2g bs 와 함께")
if BSL and not (a.opt and a.mode == "lag" and a.interp == "bary" and not REP and not a.affine and not FLOOR):
    raise SystemExit("--p2g bs 는 --opt --mode lag (재평균·아핀·에너지 바닥 없음) 만")
if BSL and a.lagdiag and not VFL:
    raise SystemExit("--p2g bs --lagdiag 는 --floor vel 만 (진단 루프가 같은 _step_lag 를 부르는 경로)")
if a.g2p == "bs" and not BSL:
    raise SystemExit("--g2p bs 는 --p2g bs 와 함께")
if RNEW and not (a.opt and a.mode == "lag" and a.interp == "bary" and not REP):
    raise SystemExit("--fb/--lim/--rstat 은 --opt --mode lag (재평균 없음) 만")
NB = -(-a.N // REP) if REP else 1                                                   # 프레임당 블록 수 (마지막 블록은 짧을 수 있다)
if HASPL:
    pt_ = [float(q) for q in planes[0]["point"]]
    nr_ = np.array(planes[0]["normal"], dtype=np.float64)
    nr_ = (nr_ / np.linalg.norm(nr_)).tolist()
    PL = [np.float64(q) for q in pt_ + nr_]
    PLf = [np.float32(q) for q in pt_ + nr_]
else:
    PL = [np.float64(0)] * 6
    PLf = [np.float32(0)] * 6
XN = 1.0 if a.eps_st > 0 else 0.0
# CFL: 파속 √((λ+2μ)/ρ) (가장 가벼운 밀도), 길이 = Kuhn 사면체 최소 높이 (첫 격자에서)
rho_min = float(rho.min()) if rho is not None else float(cfg["density"])
cwave = math.sqrt((lam + 2.0 * mu) / rho_min)
G0 = RTC.GridK(x, a.node_h)
lmin = EX.kuhn_min_altitude(G0.gl8)
dt_cfl = lmin / cwave
del G0
BSP = a.interp == "bspline2"
if BSP:
    if a.mode != "sub":
        raise SystemExit("--interp bspline2 는 --mode sub 만 (MPM 식)")
    _latd = SX.lattice(float(a.node_h), SX.HZ_CUBE * float(a.node_h), device=dev, dtype=torch.float64)
    lmin = float(_latd.A[:, 0].norm())                                              # 정육면체 격자 간격 dx (B-스플라인 CFL 길이)
    dt_cfl = lmin / cwave
print(f"[명시적 FEM] 입자 {Np}, 프레임 {L}, h {h:.6f}, 서브스텝 {a.N} (h_s {hs:.3e}), 판 {a.mode} {'최적화' if a.opt else '기준'}, "
      + (f"보간 2 차 B-스플라인 (적분점 {a.quad}{' K %d' % a.K if a.quad == 'cell' else ''}), 노드 질량 Σ w m (P2G), 전달 {'APIC' if a.apic else 'PIC'}, " if BSP else
         f"노드 질량 {'Σλm (기본)' if a.mass == 'bary' else '사면체 등분 m_T/4 (정의 변경)'}, ") + f"E {cfg['E']:g} ν {cfg['nu']:g} "
      f"(μ {mu:.4g} λ {lam:.4g}), EXT {EXT:.6f}, 바닥 {'입자 (stick·softplus)' if FLOOR else ('속도 단위 (입자 판정, 고착 %s, 비관통 %d, 후보 여유 %g)' % (a.vfl_mode, a.vfl_np, a.vfl_margin) if VFL else '없음')}, 감쇠 {DAMP:.8f}/서브스텝"
      + (f", 재평균 {a.rep_mode} {REP} 서브스텝마다 (블록 {NB})" if REP else "") + (", 아핀 전달 (APIC)" if a.affine else "")
      + (f", 전달 가중 {a.rep_w}" if a.rep_w != "std" else "")
      + (f", [절 S] P2G 2 차 B-스플라인 (노드 질량 Σ w m_p, 셀 F̄·V 셀 중심 w), G2P {'B-스플라인' if G2PBS else '사면체 barycentric'}, 메시 {'절 R 그대로 (팽창 없음, 메시 안 가중 정규화 -- 절 T)' if BSN else '1 칸 팽창'}" if BSL else "")
      + (f", [절 V] 노드 적분 FEM (F_n = Σ V_T F_T / Σ V_T, 노드 FCR 응력, β {a.nint_beta:g})" if NINT else "")
      + (f", [절 R] log J 장벽 κ {a.fb:g} Ĵ {a.fb_jhat:g} J=det({'A F̄' if a.fb_on == 'X' else 'A'})" if a.fb > 0 else "")
      + (f", [절 R] 이동 제한 det A_T ≥ {a.lim:g} ({'상자 꼭짓점 Jacobi' if a.lim_mode == 'box' else '8 색 Gauss-Seidel'}, 속도 {a.lim_v})" if LIM else "") + (", [절 R] 통계" if RNEW else ""), flush=True)
print(f"[CFL] 파속 √((λ+2μ)/ρ_min) = {cwave:.2f} m/s (ρ_min {rho_min:g}), {'격자 간격 dx' if BSP else 'Kuhn 최소 높이'} {lmin:.5f} (node_h {a.node_h}), "
      f"Δt_CFL = {dt_cfl:.3e} s -> N_CFL = h/Δt_CFL = {h / dt_cfl:.1f} (지금 h_s/Δt_CFL = {hs / dt_cfl:.3f})", flush=True)
print(f"[명령] {' '.join(sys.argv)}", flush=True)

I3 = torch.eye(3, device=dev)


def fixd_of(npos, rows_, lam_, now):
    if BC_SEL is None:
        return None, None
    fx, vb = RT.cfg_bc_nodes(cfg, BC_SEL, npos, rows_, lam_, m, now, h)
    if fx is None:
        return None, None
    return fx.to(torch.uint8).contiguous(), vb.float().contiguous()


def node_mv(G, x_, v_):
    w = (torch.full_like(G.lam, 0.25) if a.mass == "tet" else G.lam) * m.unsqueeze(-1)
    ws = torch.zeros(G.M, device=dev).index_add_(0, G.rows.reshape(-1), w.reshape(-1))
    vs = torch.zeros(G.M, 3, device=dev).index_add_(0, G.rows.reshape(-1), (w.unsqueeze(-1) * v_.unsqueeze(1)).reshape(-1, 3))
    return ws.contiguous(), (vs / ws.clamp_min(1e-30).unsqueeze(-1)).contiguous()


def near_args(n_near):
    return PL + [np.float64(a.eps_st), np.float64(a.eps_sp), np.float64(XN)]


# ---------------------------------------------------------------- 기준 판 (fem_rt 격자 + CudaTetqK, float64)
NEAR_N = []


def tet_fv(G, F_, far):
    """사면체 V_T·F̄_T (float64, 항등 입자만) -- CudaTetqK 와 같은 식."""
    Vf = vol.double() * far
    VT = torch.zeros(G.T, device=dev, dtype=torch.float64).index_add_(0, G.pinv, Vf)
    FT = torch.zeros(G.T, 3, 3, device=dev, dtype=torch.float64).index_add_(0, G.pinv, Vf[:, None, None] * F_.double())
    FT = torch.where((VT > 0)[:, None, None], FT / VT.clamp_min(1e-300)[:, None, None], torch.eye(3, device=dev, dtype=torch.float64))
    return VT.contiguous(), FT.contiguous()


def frame_base(i, x, v, F):
    st = dict()
    G = RTC.GridK(x, a.node_h)
    mn, vn = node_mv(G, x, v)
    U0 = h * vn + (h * h) * torch.tensor(gv, device=dev)
    CF = RTC.CudaTetqK(G, x, v, F, m, vol, cfg, h, U0, eps_sp=a.eps_sp, eps_st=a.eps_st, stick_xn=True, floor=FLOOR)   # 프레임 분할 (암시적 판과 같다)
    ni, nn = CF.ni, CF.nn
    far = torch.ones(Np, device=dev, dtype=torch.float64)
    far[ni] = 0.0
    Fv = F.reshape(Np, 9).contiguous()
    x0c, F0c, F0i, cc = (torch.zeros(max(nn, 1), k, device=dev) for k in (3, 9, 9, 3))
    if nn:
        EX.launch("ex_near_init", nn, (EX.P(None), np.int32(nn), EX.P(ni), EX.P(x), EX.P(v), EX.P(Fv), np.float32(h), np.float32(gv[0]), np.float32(gv[1]),
                                       np.float32(gv[2]), EX.P(x0c), EX.P(F0c), EX.P(F0i), EX.P(cc)))

    def near_force(G_, gn, Uc=None, xc=None, Fc=None):
        if nn:
            EX.launch("ex_near_force", nn, (EX.P(None), np.int32(nn), EX.P(ni), EX.P(x0c), EX.P(F0c), EX.P(cc), EX.P(Uc), EX.P(xc), EX.P(Fc), EX.P(F0i),
                                            EX.P(G_.rows), EX.P(G_.lam), EX.P(G_.gl4), EX.P(G_.pinv), EX.P(m), EX.P(vol), np.float64(h), *near_args(nn),
                                            np.float64(mu), np.float64(lam), EX.P(gn)))
    if a.mode == "lag":
        G.gl4 = G.gl8.reshape(G.T, 12).float().contiguous()
        fixd, vbc = fixd_of(G.npos, G.rows, G.lam, i * h)
        VT, FT = tet_fv(G, F, far)
        Uc = torch.zeros(G.M, 3, device=dev)
        Vn = vn.clone()
        gl8 = G.gl8.reshape(G.T, 12).contiguous()
        if a.diag and i == 0:
            kd = torch.zeros(G.M, device=dev, dtype=torch.float64).index_add_(
                0, G.tets.reshape(-1), (VT[:, None] * (lam + 2 * mu) * (G.gl8 ** 2).sum(-1)).reshape(-1))
            dtl = 2.0 / (kd / mn.double().clamp_min(1e-30)).sqrt().clamp_min(1e-30)
            ok = kd > 0
            qs = torch.quantile(dtl[ok].float().cpu(), torch.tensor([0.0, 1e-4, 1e-3, 0.01, 0.1, 0.5]))
            print(f"[진단] 국소 안정 걸음 2/ω 분위 (0, 1e-4, 1e-3, 1e-2, 0.1, 0.5): " + ", ".join(f"{float(q):.2e}" for q in qs)
                  + f" s -> 필요 N (최소) {h / float(qs[0]):.0f}; h_s 보다 짧은 노드 {int(((dtl < hs) & ok).sum())}", flush=True)
        for j in range(a.N):
            f = torch.zeros(G.M, 3, device=dev)
            EX.launch("ex_tet_force_dd", G.T, (EX.P(None), np.int32(G.T), EX.P(G.tets), EX.P(Uc), EX.P(gl8), EX.P(FT), EX.P(VT),
                                               np.float64(mu), np.float64(lam), EX.P(f)))
            gn = torch.zeros(G.M, 3, device=dev)
            near_force(G, gn, Uc=Uc)
            dU = torch.empty(G.M, 3, device=dev)
            EX.launch("ex_node_step", G.M, (EX.P(None), np.int32(G.M), EX.P(mn), EX.P(f), EX.P(gn), EX.P(Vn), EX.P(dU), EX.P(Uc), np.float32(gv[0]),
                                            np.float32(gv[1]), np.float32(gv[2]), np.float32(hs), np.float32(DAMP), EX.P(fixd), EX.P(vbc)))
            if a.diag and i == 0 and (j < 4 or j % max(1, a.N // 8) == 0):
                print(f"    [진단] 서브스텝 {j}: 최대 |v_n| {float(Vn.norm(dim=-1).max()):.3e}, 최대 |f/m| {float((f.norm(dim=-1) / mn.clamp_min(1e-30)).max()):.3e}, "
                      f"최대 |바닥 g/m| {float((gn.norm(dim=-1) / mn.clamp_min(1e-30)).max()):.3e}, 비유한 {int((~torch.isfinite(Vn)).sum())}", flush=True)
        x2, v2, F2 = torch.empty_like(x), torch.empty_like(x), torch.empty(Np, 9, device=dev)
        EX.launch("ex_g2p_lag", Np, (np.int32(Np), EX.P(x), EX.P(Uc), EX.P(Vn), EX.P(G.rows), EX.P(G.lam), EX.P(G.gl4), EX.P(G.pinv),
                                     EX.P(Fv), EX.P(x2), EX.P(v2), EX.P(F2), EX.P(None), np.int32(0)), b=256)
        if nn:
            EX.launch("ex_near_fin", nn, (EX.P(None), np.int32(nn), EX.P(ni), EX.P(x0c), EX.P(F0c), EX.P(Uc), EX.P(None), EX.P(None), EX.P(F0i),
                                          EX.P(G.rows), EX.P(G.lam), EX.P(G.gl4), EX.P(G.pinv), *near_args(nn), np.float64(h), EX.P(x2), EX.P(v2), EX.P(F2)))
        st.update(M=G.M, T=G.T, n_near=nn)
        return x2, v2, F2.view(Np, 3, 3), st
    # sub: 매 서브스텝 격자·P2G·선형 G2P, 바닥 후보 분할은 프레임 것
    xc, vc, Fc = x, v, Fv
    for j in range(a.N):
        Gj = G if j == 0 else RTC.GridK(xc, a.node_h)
        Gj.gl4 = Gj.gl8.reshape(Gj.T, 12).float().contiguous()
        mnj, vnj = (mn, vn) if j == 0 else node_mv(Gj, xc, vc)
        fixd, vbc = fixd_of(Gj.npos, Gj.rows, Gj.lam, i * h + j * hs)
        VT, FT = tet_fv(Gj, Fc.view(Np, 3, 3), far)
        f = torch.zeros(Gj.M, 3, device=dev)
        EX.launch("ex_tet_force_dd", Gj.T, (EX.P(None), np.int32(Gj.T), EX.P(Gj.tets), EX.P(None), EX.P(Gj.gl8.reshape(Gj.T, 12).contiguous()), EX.P(FT),
                                            EX.P(VT), np.float64(mu), np.float64(lam), EX.P(f)))
        gn = torch.zeros(Gj.M, 3, device=dev)
        near_force(Gj, gn, xc=xc, Fc=Fc)
        dU = torch.empty(Gj.M, 3, device=dev)
        Vn = vnj.clone()
        EX.launch("ex_node_step", Gj.M, (EX.P(None), np.int32(Gj.M), EX.P(mnj), EX.P(f), EX.P(gn), EX.P(Vn), EX.P(dU), EX.P(None), np.float32(gv[0]),
                                         np.float32(gv[1]), np.float32(gv[2]), np.float32(hs), np.float32(1.0), EX.P(fixd), EX.P(vbc)))
        x2, v2, F2 = torch.empty_like(x), torch.empty_like(x), torch.empty(Np, 9, device=dev)
        EX.launch("ex_g2p_sub", Np, (np.int32(Np), EX.P(xc), EX.P(dU), EX.P(Gj.rows), EX.P(Gj.lam), EX.P(Gj.gl4), EX.P(Gj.pinv), EX.P(Fc), np.int32(0), *PL,
                                     np.float64(a.eps_st), np.float64(a.eps_sp), np.float64(XN), np.float64(hs), np.float64(DAMP), EX.P(x2), EX.P(v2), EX.P(F2)), b=256)
        xc, vc, Fc = x2, v2, F2
    if nn:
        EX.launch("ex_near_fin", nn, (EX.P(None), np.int32(nn), EX.P(ni), EX.P(x0c), EX.P(F0c), EX.P(None), EX.P(xc), EX.P(Fc), EX.P(F0i),
                                      EX.P(G.rows), EX.P(G.lam), EX.P(G.gl4), EX.P(G.pinv), *near_args(nn), np.float64(h), EX.P(xc), EX.P(vc), EX.P(Fc)))
    st.update(M=G.M, T=G.T, n_near=nn)
    return xc, vc, Fc.view(Np, 3, 3), st



def _det3(Q):
    """[.., 3, 3] 행렬식 (삼중곱 -- 비유한 값에도 예외 없이; --lagdump 전용)."""
    return (Q[..., 0, :] * torch.cross(Q[..., 1, :], Q[..., 2, :], dim=-1)).sum(-1)


# ---------------------------------------------------------------- 최적화 판
class Opt:
    def __init__(self):
        self.lat = SX.lattice(float(a.node_h), SX.HZ_CUBE * float(a.node_h), device=dev, dtype=torch.float32)
        self.Ai = self.lat.Ai.float().contiguous()
        self.gpar = torch.zeros(12, device=dev)
        self.ipar = torch.zeros(4, dtype=torch.int32, device=dev)
        self.sz = torch.zeros(4, dtype=torch.int32, device=dev)
        self.Mcap = self.Tcap = self.Pcap = 0
        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=dev, dtype=dt)
        self.rows = z(Np, 4, dt=torch.long)
        self.lamb = z(Np, 4)
        self.pinv = z(Np, dt=torch.long)
        self.nslot = z(Np, dt=torch.int32)
        self.nidx = z(Np, dt=torch.long)
        self.x0c, self.F0c, self.F0i, self.cc = z(Np, 3), z(Np, 9), z(Np, 9), z(Np, 3)
        self.X = [z(Np, 3), z(Np, 3)]
        self.V = [z(Np, 3), z(Np, 3)]
        self.FF = [z(Np, 9), z(Np, 9)]
        self.graph = None
        self.n_cap = 0
        self.SORT = (a.mode == "sub" and not a.no_fuse and not a.no_sort)
        self.mA, self.volA = m, vol
        if self.SORT:
            self.perm = torch.zeros(Np, dtype=torch.long, device=dev)
            self.mA, self.volA = m.clone(), vol.clone()
            self.OX, self.OV, self.OF = z(Np, 3), z(Np, 3), z(Np, 9)
        self.fixd = None
        self._fhook = None                                                          # --lagdump: 서브스텝 노드 걸음 직전 내부력 f 복사 (진단 루프만, 그래프 판은 None)
        self.ncon = torch.zeros(1, dtype=torch.int32, device=dev)                    # --floor vel: 프레임 동안 접촉 (입자·서브스텝) 수
        self.bst = z(4)                                                              # 절 S: [0] 질량 0 노드 때문에 뺀 사면체 부피, [1] 그 사면체 수; 절 T: [2] 정규화로 메시 안 노드에 옮긴 질량 Σ(1-Win)m_p, [3] 옮긴 부피 Σ(1-Cin)V_p
        self.rst = z(8)                                                              # 절 R 통계 (프레임마다 비움): [0] 뒤집힌 점유 사면체 x 서브스텝, [1] 제한 노드 x 서브스텝, [2:5] 줄인 운동량, [5] 줄인 KE, [6] Σ|줄인 운동량|
        self.novf = torch.zeros(1, dtype=torch.int32, device=dev)                    # 노드 -> 사면체 목록 넘침 (24 초과 -- 없어야 한다)
        self.CL = z(Np, 9) if a.affine else None                                     # --affine: 입자 속도 기울기 (프레임 사이)
        if a.affine and (a.mode != "lag" or self.SORT):
            raise SystemExit("--affine 은 --opt --mode lag 만")

    def _alloc(self, M, T, P):
        new = False
        if M > self.Mcap:
            self.Mcap = int(M * 1.3) + 64
            z = lambda *s: torch.zeros(*s, device=dev)
            self.mn, self.mv, self.vn, self.f, self.gn, self.dU, self.Uc = z(self.Mcap), z(self.Mcap, 3), z(self.Mcap, 3), z(self.Mcap, 3), \
                z(self.Mcap, 3), z(self.Mcap, 3), z(self.Mcap, 3)
            self.fx = torch.zeros(self.Mcap, dtype=torch.uint8, device=dev)
            self.vbc = z(self.Mcap, 3)
            self.acc = z(self.Mcap)                                                  # --floor vel: 노드별 접촉 입자 질량 (쓴 뒤 비움)
            self.ml = z(self.Mcap) if RWL else None                                  # 절 Q --rep_w lam: 속도 전달 정규화 질량 Σλm
            if LIM:
                self.alph = torch.full((self.Mcap,), 1065353216, dtype=torch.int32, device=dev)   # box: 노드 α (1.0f 비트)
                self.ncol = torch.zeros(self.Mcap, dtype=torch.uint8, device=dev)   # 절 R (2): Kuhn 격자 지표 홀짝 색 (0..7)
                self.ntc = torch.zeros(self.Mcap, dtype=torch.int32, device=dev)
                self.nt = torch.zeros(self.Mcap * 24, dtype=torch.int32, device=dev)
            if NINT:                                                                 # 절 V: 노드 부피 V_n, 노드 응력 P_n, 노드 -> 사면체 목록
                self.Vnd, self.Pnd = z(self.Mcap), z(self.Mcap, 9)
                self.ntc_u = torch.zeros(self.Mcap, dtype=torch.int32, device=dev)
                self.nt_u = torch.zeros(self.Mcap * 24, dtype=torch.int32, device=dev)
            self.nposb = z(self.Mcap, 3)                                             # 노드 위치 상주 버퍼 (절 Q: 그래프 안 APIC P2G 가 주소를 고정해 읽는다)
            new = True
        if T > self.Tcap:
            self.Tcap = int(T * 1.3) + 64
            self.VTs = torch.zeros(self.Tcap, device=dev)
            self.FTs = torch.zeros(self.Tcap, 9, device=dev)
            self.FT = torch.zeros(self.Tcap, 9, device=dev)
            self.gl = torch.zeros(self.Tcap, 12, device=dev)
            self.tets = torch.zeros(self.Tcap, 4, dtype=torch.long, device=dev)
            self.tcell = torch.zeros(self.Tcap, dtype=torch.long, device=dev)
            if NINT:
                self.XTu = torch.zeros(self.Tcap, 9, device=dev)                     # 절 V: 사면체 (V_T/4) F_T
            new = True
        if P > self.Pcap:
            self.Pcap = int(P * 1.3) + 64
            self.cm0 = torch.zeros(self.Pcap, dtype=torch.uint8, device=dev)
            self.cm1 = torch.zeros(self.Pcap, dtype=torch.uint8, device=dev)
            self.nmark = torch.zeros(self.Pcap, dtype=torch.uint8, device=dev)
            self.tmark = torch.zeros(self.Pcap * 9, dtype=torch.uint8, device=dev)
            self.ncm = torch.zeros(self.Pcap, dtype=torch.long, device=dev)
            self.tcm = torch.zeros(self.Pcap * 9, dtype=torch.long, device=dev)
            if BSL:
                self.cV = torch.zeros(self.Pcap, device=dev)
                self.cVF = torch.zeros(self.Pcap * 9, device=dev)
            new = True
        if new:
            self.graph = None

    def grid(self, x, v, now):
        """셀 점유 격자 (월드 고정 Kuhn 격자, GridK 와 같은 격자점·같은 사면체 꼭짓점 순서). 동기화 2 번 (범위, 개수)."""
        lat = self.lat
        lo_ = x.min(0).values
        hi_ = x.max(0).values
        vmax = v.norm(dim=-1).max()
        ext = (hi_ - lo_).max().clamp_min(1e-12)
        dil = a.dil
        pad = 0.05 * ext
        if a.mode == "sub" or REPF:                                                # 절 P full: 프레임 안 재배치 -> sub 와 같은 여유·팽창
            pad = pad + vmax * h * 1.5 + 2.0 * a.node_h
        b0, b1 = lo_ - pad, hi_ + pad
        cor = torch.stack([torch.stack([b0[0] if (mm & 1) else b1[0], b0[1] if (mm & 2) else b1[1], b0[2] if (mm & 4) else b1[2]]) for mm in range(8)], 0)
        y = cor @ lat.Ai.T
        imin = torch.floor(y.min(0).values).long() - 1
        imax = torch.ceil(y.max(0).values).long() + 1
        ii = torch.cat([imin, imax, (vmax * h / lmin).ceil().long().view(1)]).tolist()                       # 동기화 1
        if float(ii[6]) * lmin > 0.25 * EXT:
            raise _Diverge(f"프레임 이동 max|v|·h > 0.25 EXT ({ii[6]} 셀)")
        nnl = [max(ii[3 + k] - ii[k] + 1, 2) for k in range(3)]
        if (a.mode == "sub" or REPF) and dil < 0:
            dil = int(ii[6]) + 2                                                       # 프레임 최대 이동 (셀) + 2
        if a.mode == "lag" and not REPF:
            dil = 1 if (BSL and not BSN) else 0                                        # 절 S: B-스플라인 지지 (노드 27·셀 27) 가 메시 안에 들게 1 칸 팽창 (절 T --bs_mesh occ: 팽창 없음, 절 R 메시)
        n0, n1, n2 = nnl
        Pn = n0 * n1 * n2
        if not hasattr(self, "P0"):
            self.P0 = Pn
        if Pn > 8 * self.P0:
            raise _Diverge(f"격자 상자 {Pn} > 8 x 첫 프레임 {self.P0}")
        lo = (lat.A @ imin.to(lat.A.dtype))
        self.gpar[:3] = lo
        self.gpar[3:] = self.Ai.reshape(-1)
        self.ipar.copy_(torch.tensor([n0, n1, n2, 0], dtype=torch.int32), non_blocking=True)
        self.dims = (n0, n1, n2)
        self._alloc(0, 0, Pn)
        cm0, cm1 = self.cm0[:Pn], self.cm1[:Pn]
        cm0.zero_()
        EX.launch("ex_mark_cells", Np, (np.int32(Np), EX.P(x), EX.P(self.gpar), EX.P(self.ipar), EX.P(cm0)), b=256)
        if dil > 0:
            EX.launch("ex_dilate", Pn, (EX.P(self.ipar), np.int32(dil), EX.P(cm0), EX.P(cm1)), b=256)
            cm = cm1
        else:
            cm = cm0
        nm, tm = self.nmark[:Pn], self.tmark[:Pn * 9]
        nm.zero_()
        tm.zero_()
        EX.launch("ex_mark_nt", Pn, (EX.P(self.ipar), EX.P(cm), EX.P(nm), EX.P(tm)), b=256)
        ncs = torch.cumsum(nm, 0, dtype=torch.int64)
        tcs = torch.cumsum(tm, 0, dtype=torch.int64)
        MT = torch.stack([ncs[-1], tcs[-1]]).tolist()                                 # 동기화 2
        M, T = int(MT[0]), int(MT[1])
        self._alloc(M, T, Pn)
        nm, tm = self.nmark[:Pn], self.tmark[:Pn * 9]
        self.ncm[:Pn] = torch.where(nm.bool(), ncs - 1, torch.full_like(ncs, -1))
        self.tcm[:Pn * 9] = tcs - 1
        EX.launch("ex_build_tets", Pn * 9, (EX.P(self.ipar), EX.P(tm), EX.P(self.tcm), EX.P(self.ncm), EX.P(self.tets)), b=256)
        if BSL:
            self.cmk = cm
            EX.launch("ex_tet_cell", Pn * 9, (EX.P(self.ipar), EX.P(tm), EX.P(self.tcm), EX.P(self.tcell)), b=256)
            self.Pn = Pn
        ar = torch.arange(Pn, device=dev)
        uniq = torch.empty(M + 1, dtype=torch.long, device=dev).scatter_(0, torch.where(nm.bool(), ncs - 1, torch.full_like(ncs, M)), ar)[:M]
        k2 = uniq % n2
        k1 = (uniq // n2) % n1
        k0 = uniq // (n1 * n2)
        npos = (torch.stack([k0, k1, k2], -1).to(lat.A.dtype) @ lat.A.T + lo).float().contiguous()
        self.sz.copy_(torch.tensor([M, T, 0, 0], dtype=torch.int32), non_blocking=True)
        EX.launch("ex_tet_geo32", T, (EX.P(self.sz), EX.P(self.tets), EX.P(npos), EX.P(self.gl)))
        self.M, self.T = M, T
        self.nposb[:M].copy_(npos)
        self.npos = self.nposb[:M]
        if LIM:
            self.alph.fill_(1065353216)
        if NINT:                                                                    # 절 V: 노드 -> 사면체 목록 (프레임마다)
            self.ntc_u[:M].zero_()
            EX.launch("ex_node_tets", T, (EX.P(self.sz), EX.P(self.tets), EX.P(self.ntc_u), EX.P(self.nt_u), EX.P(self.novf)), b=256)
        if LIM and a.lim_mode == "gs":
            # 색 = 전역 격자 지표 (imin + k) 의 홀짝 -- 한 사면체 네 노드는 지표 차 {0,1}³\0 이라 모두 다른 색
            gk = torch.stack([k0, k1, k2], -1) + imin.to(k0.device).view(1, 3)
            self.ncol[:M] = ((gk[:, 0] & 1) | ((gk[:, 1] & 1) << 1) | ((gk[:, 2] & 1) << 2)).to(torch.uint8)
            self.ntc[:M].zero_()
            EX.launch("ex_node_tets", T, (EX.P(self.sz), EX.P(self.tets), EX.P(self.ntc), EX.P(self.nt), EX.P(self.novf)), b=256)
        self.fixd = None
        if BC_SEL is not None:
            # cuboid 만 (입자 구속 없음 -- 위에서 막았다): train_deform.cfg_bc_nodes 의 cuboid 판정 그대로 (노드 위치)
            fx, vb = RT.cfg_bc_nodes(cfg, [], npos, torch.zeros(1, 4, dtype=torch.long, device=dev), torch.zeros(1, 4, device=dev), m[:1], now, h)
            if fx is not None:
                self.fx[:M] = fx.to(torch.uint8)
                self.vbc[:M] = vb.float()
                self.fixd = True
        return M, T

    # ---- 커널 묶음 (그래프에 들어가는 부분은 크기를 self.sz 에서 읽는다) ----
    def _p2g(self, x, v, F, H, keep):
        Mc, Tc = self.Mcap, self.Tcap
        self.mn.zero_()
        self.mv.zero_()
        if RWL:
            self.ml.zero_()
        self.VTs.zero_()
        self.FTs.zero_()
        if not keep:
            self.sz[2:3].zero_()
        EX.launch("ex_locate", Np, (np.int32(Np), EX.P(x), EX.P(self.gpar), EX.P(self.ipar), EX.P(self.ncm), EX.P(self.tcm), EX.P(self.tmark),
                                    EX.P(self.rows), EX.P(self.lamb), EX.P(self.pinv), EX.P(self.sz)), b=256)
        aff = self.CL if (a.affine and a.mode == "lag") else None
        EX.launch("ex_p2g_mv", Np, (np.int32(Np), EX.P(self.rows), EX.P(self.lamb), EX.P(self.mA), EX.P(v), EX.P(self.mn), EX.P(self.mv), np.int32(1 if a.mass == "tet" else 0),
                                    EX.P(aff), EX.P(x if aff is not None else None), EX.P(self.npos if aff is not None else None), EX.P(self.ml if RWL else None)), b=256)
        EX.launch("ex_vfin", Mc, (EX.P(self.sz), np.int32(0), EX.P(self.mn), EX.P(self.mv), EX.P(self.vn), EX.P(self.ml if RWL else None)), b=256)
        EX.launch("ex_split", Np, (np.int32(Np), EX.P(x), EX.P(self.rows), EX.P(self.lamb), EX.P(self.pinv), EX.P(self.vn), np.float32(H),
                                   np.float32(gv[0]), np.float32(gv[1]), np.float32(gv[2]), np.int32(1 if HASPL else 0), *PLf,
                                   np.float32(a.vfl_margin / 20.0 if VFL else a.eps_sp), np.float32(0.0 if VFL else a.eps_st), EX.P(self.volA), EX.P(F),
                                   EX.P(self.VTs), EX.P(self.FTs), EX.P(self.nslot), EX.P(self.nidx), EX.P(self.sz), np.int32(1 if keep else 0),
                                   np.int32(1 if VFL else 0)), b=256)
        EX.launch("ex_ffin", Tc, (EX.P(self.sz), np.int32(0), EX.P(self.VTs), EX.P(self.FTs), EX.P(self.FT)), b=256)

    def _p2g_bs(self, x, v, F):
        """절 S 프레임 시작 P2G (B-스플라인): 노드 질량·운동량 (27 노드), 셀 V·VF (셀 중심 27 셀) -> 사면체 V_T = V_c/6·F̄; 사면체 locate (barycentric G2P·진단용)."""
        Pn = self.Pn
        self.mn.zero_()
        self.mv.zero_()
        self.cV[:Pn].zero_()
        self.cVF[:Pn * 9].zero_()
        self.bst.zero_()
        self.sz[2:3].zero_()
        EX.launch("ex_locate", Np, (np.int32(Np), EX.P(x), EX.P(self.gpar), EX.P(self.ipar), EX.P(self.ncm), EX.P(self.tcm), EX.P(self.tmark),
                                    EX.P(self.rows), EX.P(self.lamb), EX.P(self.pinv), EX.P(self.sz)), b=256)
        EX.launch("bsl_p2g", Np, (np.int32(Np), EX.P(x), EX.P(v), EX.P(F), EX.P(self.mA), EX.P(self.volA), EX.P(self.gpar), EX.P(self.ipar), EX.P(self.ncm),
                                  EX.P(self.cmk), EX.P(self.mn), EX.P(self.mv), EX.P(self.cV), EX.P(self.cVF), EX.P(self.sz), np.int32(1 if VFL else 0),
                                  np.float32(h), np.float32(gv[0]), np.float32(gv[1]), np.float32(gv[2]), *PLf, np.float32(a.vfl_margin), EX.P(self.nidx),
                                  np.int32(BSN), EX.P(self.bst)), b=128)
        EX.launch("ex_vfin", self.Mcap, (EX.P(self.sz), np.int32(0), EX.P(self.mn), EX.P(self.mv), EX.P(self.vn), EX.P(None)), b=256)
        EX.launch("bsl_tfin", self.Tcap, (EX.P(self.sz), EX.P(self.tcell), EX.P(self.tets), EX.P(self.mn), EX.P(self.cV), EX.P(self.cVF),
                                          EX.P(self.VTs), EX.P(self.FT), EX.P(self.bst)), b=256)

    def _step(self, Ucum, xc, Fc):
        """한 서브스텝 힘 + 노드 걸음 (크기는 self.sz 에서)."""
        Mc, Tc = self.Mcap, self.Tcap
        self.f.zero_()
        EX.launch("ex_tet_force_ff", Tc, (EX.P(self.sz), np.int32(0), EX.P(self.tets), EX.P(Ucum), EX.P(self.gl), EX.P(self.FT), EX.P(self.VTs),
                                          np.float32(mu), np.float32(lam), EX.P(self.f)))
        if FLOOR:
            self.gn.zero_()
            EX.launch("ex_near_force32" if a.near32 else "ex_near_force", Np, (EX.P(self.sz), np.int32(0), EX.P(self.nidx), EX.P(self.x0c), EX.P(self.F0c), EX.P(self.cc), EX.P(Ucum),
                                            EX.P(xc), EX.P(Fc), EX.P(self.F0i), EX.P(self.rows), EX.P(self.lamb), EX.P(self.gl), EX.P(self.pinv), EX.P(self.mA),
                                            EX.P(self.volA), np.float64(h), *near_args(0), np.float64(mu), np.float64(lam), EX.P(self.gn)), b=64)
        EX.launch("ex_node_step", Mc, (EX.P(self.sz), np.int32(0), EX.P(self.mn), EX.P(self.f), EX.P(self.gn if FLOOR else None), EX.P(self.vn),
                                       EX.P(self.dU), EX.P(Ucum), np.float32(gv[0]), np.float32(gv[1]), np.float32(gv[2]), np.float32(hs),
                                       np.float32(DAMP if a.mode == "lag" else 1.0), EX.P(self.fx if self.fixd else None), EX.P(self.vbc)), b=256)

    def _step_lag(self, xcur=None):
        """lag 한 서브스텝 (절 O 판): 사면체 힘 -> [에너지 바닥 후보 힘] -> [속도 바닥 접촉 판정] -> 노드 걸음 (+ 고착·비관통, f·gn·acc 비움)."""
        Mc, Tc = self.Mcap, self.Tcap
        if NINT:                                                                    # 절 V: 노드 적분 내부력 (커널 3 개)
            self._nint_force()
        elif RNEW:
            EX.launch("ex_tet_force_R", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT), EX.P(self.VTs),
                                             np.float32(mu), np.float32(lam), EX.P(self.f), np.float32(a.fb), np.float32(a.fb_jhat),
                                             np.int32(1 if a.fb_on == "A" else 0), EX.P(self.rst)))
        else:
            EX.launch("ex_tet_force_ff", Tc, (EX.P(self.sz), np.int32(0), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT), EX.P(self.VTs),
                                              np.float32(mu), np.float32(lam), EX.P(self.f)))
        if FLOOR:
            EX.launch("ex_near_force32" if a.near32 else "ex_near_force", Np, (EX.P(self.sz), np.int32(0), EX.P(self.nidx), EX.P(self.x0c), EX.P(self.F0c),
                                            EX.P(self.cc), EX.P(self.Uc), EX.P(None), EX.P(None), EX.P(self.F0i), EX.P(self.rows), EX.P(self.lamb), EX.P(self.gl),
                                            EX.P(self.pinv), EX.P(self.mA), EX.P(self.volA), np.float64(h), *near_args(0), np.float64(mu), np.float64(lam),
                                            EX.P(self.gn)), b=64)
        if VFL and BSL:
            EX.launch("bsl_vfl_contact", Np, (EX.P(self.sz), EX.P(self.nidx), EX.P(self.Xf if xcur is None else xcur), EX.P(self.Uc), EX.P(self.rows), EX.P(self.lamb),
                                              EX.P(self.gpar), EX.P(self.ipar), EX.P(self.ncm), EX.P(self.mA), np.int32(1 if G2PBS else 0), *PLf,
                                              EX.P(self.acc), EX.P(self.ncon), np.int32(BSN)), b=128)
        elif VFL:
            EX.launch("ex_vfl_contact", Np, (EX.P(self.sz), EX.P(self.nidx), EX.P(self.Xf if xcur is None else xcur), EX.P(self.Uc), EX.P(self.rows), EX.P(self.lamb), EX.P(self.mA),
                                             np.int32(1 if a.mass == "tet" else 0), *PLf, EX.P(self.acc), EX.P(self.ncon)), b=128)
        if self._fhook is not None:                                                 # 진단 읽기만 (걸음이 f 를 비우기 전)
            self._fhook()
        EX.launch("ex_node_step_lag", Mc, (EX.P(self.sz), EX.P(self.mn), EX.P(self.f), EX.P(self.gn if FLOOR else None), EX.P(self.vn), EX.P(self.Uc),
                                           np.float32(gv[0]), np.float32(gv[1]), np.float32(gv[2]), np.float32(hs), np.float32(DAMP),
                                           EX.P(self.fx if self.fixd else None), EX.P(self.vbc), EX.P(self.acc if VFL else None),
                                           np.int32(1 if a.vfl_mode == "full" else 0), np.int32(a.vfl_np), *PLf[3:], np.int32(1 if LIM else 0)), b=256)
        if LIM and a.lim_mode == "box":
            EX.launch("ex_lim_box_tet", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.gl), EX.P(self.VTs), EX.P(self.vn), EX.P(self.Uc), np.float32(hs),
                                             np.float32(a.lim), EX.P(self.alph), EX.P(self.rst)))
            EX.launch("ex_lim_box_apply", Mc, (EX.P(self.sz), EX.P(self.mn), EX.P(self.vn), EX.P(self.Uc), np.float32(hs), EX.P(self.alph),
                                               np.int32(1 if a.lim_v == "keep" else 0), EX.P(self.fx if self.fixd else None), EX.P(self.rst)), b=256)
        elif LIM:
            for c in range(8):
                EX.launch("ex_lim_color", Mc, (EX.P(self.sz), np.int32(c), EX.P(self.ncol), EX.P(self.ntc), EX.P(self.nt), EX.P(self.tets), EX.P(self.gl),
                                               EX.P(self.VTs), EX.P(self.mn), EX.P(self.vn), EX.P(self.Uc), np.float32(hs), np.float32(a.lim),
                                               np.int32(1 if a.lim_v == "keep" else 0), EX.P(self.fx if self.fixd else None), EX.P(self.rst)), b=256)

    def _nint_force(self):
        Mc, Tc = self.Mcap, self.Tcap
        EX.launch("nint_cell", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT), EX.P(self.VTs), EX.P(self.XTu), EX.P(self.rst)))
        EX.launch("nint_node", Mc, (EX.P(self.sz), EX.P(self.ntc_u), EX.P(self.nt_u), EX.P(self.VTs), EX.P(self.XTu), EX.P(self.Pnd), EX.P(None),
                                    np.float32(mu), np.float32(lam)), b=128)
        EX.launch("nint_force", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT), EX.P(self.VTs), EX.P(self.Pnd),
                                     np.float32(mu), np.float32(lam), np.float32(a.nint_beta), EX.P(self.f)))

    def nint_check(self, i):
        """절 V 검산 (측정 밖, 프레임 끝 상태 U 와 그것에 무작위 변위 (0.01 node_h) 를 더한 상태 두 가지):
        (a) 에너지 일관성: float64 노드 에너지 E(U) = (1-β) Σ_n V_n ψ(F_n) + β Σ_T V_T ψ(F_T) (ψ = μ|F - R|² + λ/2 (det F - 1)², R = 극분해) 의
            중앙 차분 (무작위 방향·힘 큰 노드 성분) vs 같은 식의 float64 해석 힘 f = -Σ_T (V_T/4) [(1-β) Σ_n P_n + 4β P_T] F̄ᵀ ∇λ
        (b) 커널 (float32) 힘 vs float64 해석 힘: |f_k - f_64| / |f_64| (전체 노름)
        (c) 내력 합·모멘트 합 (운동량·각운동량)."""
        M, T = self.M, self.T
        dd = torch.float64
        occ = self.VTs[:T] > 0
        tets = self.tets[:T][occ]
        gl = self.gl[:T].view(T, 4, 3)[occ].to(dd)
        FT = self.FT[:T].view(T, 3, 3)[occ].to(dd)
        VT = self.VTs[:T][occ].to(dd)
        bt = float(a.nint_beta)
        Vn = torch.zeros(M, device=dev, dtype=dd).index_add_(0, tets.reshape(-1), (VT / 4).repeat_interleave(4))
        live = Vn > 0
        I3d = torch.eye(3, device=dev, dtype=dd)

        def polar_R(X):
            Us, S_, Vh = torch.linalg.svd(X)
            dg = torch.ones(X.shape[0], 3, device=dev, dtype=dd)
            dg[:, 2] = torch.sign(torch.linalg.det(Us @ Vh))
            return Us @ torch.diag_embed(dg) @ Vh

        def psi(X):
            return mu * ((X - polar_R(X)) ** 2).sum((-1, -2)) + 0.5 * lam * (torch.linalg.det(X) - 1) ** 2

        def Pf(X):
            J = torch.linalg.det(X)
            cof = J[:, None, None] * torch.linalg.inv(X).transpose(-1, -2)
            return 2 * mu * (X - polar_R(X)) + lam * (J - 1)[:, None, None] * cof

        def state(U_):
            X = (I3d + torch.einsum("tia,tib->tab", U_[tets], gl)) @ FT
            Sn = torch.zeros(M, 9, device=dev, dtype=dd).index_add_(0, tets.reshape(-1), ((VT / 4)[:, None, None] * X).repeat_interleave(4, 0).reshape(-1, 9))
            Fn = torch.zeros(M, 3, 3, device=dev, dtype=dd)
            Fn[live] = (Sn[live] / Vn[live, None]).view(-1, 3, 3)
            return X, Fn

        def energy(U_):
            X, Fn = state(U_)
            E = (1 - bt) * (Vn[live] * psi(Fn[live])).sum()
            if bt > 0:
                E = E + bt * (VT * psi(X)).sum()
            return float(E)

        def force64(U_):
            X, Fn = state(U_)
            Pn = torch.zeros(M, 3, 3, device=dev, dtype=dd)
            Pn[live] = Pf(Fn[live])
            S = (1 - bt) * Pn[tets].sum(1)
            if bt > 0:
                S = S + 4 * bt * Pf(X)
            Gm = S @ FT.transpose(-1, -2)                                             # [T,3,3]
            fT = -(VT / 4)[:, None, None] * torch.einsum("tcb,tib->tic", Gm, gl)      # [T,4,3]
            return torch.zeros(M, 3, device=dev, dtype=dd).index_add_(0, tets.reshape(-1), fT.reshape(-1, 3))

        def kforce(U_):
            Uk = self.Uc[:M].clone()
            self.Uc[:M].copy_(U_.float())
            self.f.zero_()
            self._nint_force()
            torch.cuda.synchronize()
            fk_ = self.f[:M].to(dd).clone()
            self.f.zero_()
            self.Uc[:M].copy_(Uk)
            return fk_
        g = torch.Generator(device=dev).manual_seed(1234 + i)
        U0 = self.Uc[:M].to(dd).clone()
        Up = U0 + 0.01 * float(a.node_h) * torch.randn(M, 3, device=dev, dtype=dd, generator=g) * live[:, None]
        recs = []
        for nm, Ud in (("U", U0), ("U+δ", Up)):
            f64 = force64(Ud)
            fk = kforce(Ud)
            sc = float(f64.norm())
            dirs = []
            for k in range(4):
                dv = torch.randn(M, 3, device=dev, dtype=dd, generator=g) * live[:, None]
                dv = dv / dv.norm()
                eps = 1e-6 * float(a.node_h)
                fd = (energy(Ud + eps * dv) - energy(Ud - eps * dv)) / (2 * eps)
                an = -float((f64 * dv).sum())
                dirs.append(dict(fd=fd, an=an, rel=abs(fd - an) / max(abs(fd), abs(an), 1e-300)))
            comp = []
            for n_ in torch.argsort(f64.norm(dim=-1), descending=True)[:3].tolist():
                for c in range(3):
                    e_ = torch.zeros(M, 3, device=dev, dtype=dd)
                    e_[n_, c] = 1.0
                    eps = 1e-6 * float(a.node_h)
                    fd = -(energy(Ud + eps * e_) - energy(Ud - eps * e_)) / (2 * eps)
                    comp.append(dict(n=n_, c=c, fd=fd, an=float(f64[n_, c]), rel=abs(fd - float(f64[n_, c])) / max(float(f64[n_].norm()), 1e-300)))
            xc = self.npos[:M].to(dd) + Ud
            fm_ = f64.norm(dim=-1)
            recs.append(dict(state=nm, E=energy(Ud), f_norm=sc, fd_dir_rel=max(q["rel"] for q in dirs), fd_comp_rel=max(q["rel"] for q in comp),
                             kern_rel=float((fk - f64).norm()) / max(sc, 1e-300), mom_rel=float(f64.sum(0).norm()) / max(float(fm_.sum()), 1e-300),
                             ang_rel=float(torch.cross(xc, f64, dim=-1).sum(0).norm()) / max(float((xc.norm(dim=-1) * fm_).sum()), 1e-300),
                             kmom_rel=float(fk.sum(0).norm()) / max(float(fk.norm(dim=-1).sum()), 1e-300), dirs=dirs, comp=comp))
        rec = dict(t=i, M=M, T_occ=int(occ.sum()), beta=bt, fixd=bool(self.fixd), states=recs)
        with open(os.path.join(a.out, "nint_check.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        for r in recs:
            print(f"    [V 검산] t {i + 1} {r['state']}: E {r['E']:.6g} |f| {r['f_norm']:.4g}; 에너지 중앙 차분 vs float64 해석 힘 상대차 방향 {r['fd_dir_rel']:.2e} 성분 {r['fd_comp_rel']:.2e}; "
                  f"커널 (float32) vs float64 {r['kern_rel']:.2e}; |Σf|/Σ|f| {r['mom_rel']:.2e} |Σx×f|/Σ|x||f| {r['ang_rel']:.2e}", flush=True)

    def _substeps(self):
        if a.mode == "lag":
            self.f.zero_()
            self.gn.zero_()
            if VFL:
                self.acc.zero_()
            if not REP:
                for j in range(a.N):
                    self._step_lag()
                return
            # 절 P: 블록 b (K 서브스텝) 마다 재평균. 입자 상태는 X/V/FF[b % 2] (프레임 시작 = [0])
            for b in range(NB):
                cur = b % 2 if REPF else 0
                for j in range(b * REP, min((b + 1) * REP, a.N)):
                    self._step_lag(self.X[cur])
                if b == NB - 1:
                    break
                if REPF:
                    nx = 1 - cur
                    EX.launch("ex_g2p_lag", Np, (np.int32(Np), EX.P(self.X[cur]), EX.P(self.Uc), EX.P(self.vn), EX.P(self.rows), EX.P(self.lamb), EX.P(self.gl),
                                                 EX.P(self.pinv), EX.P(self.FF[cur]), EX.P(self.X[nx]), EX.P(self.V[nx]), EX.P(self.FF[nx]),
                                                 EX.P(self.CL if a.affine else None), np.int32(1 if RW4 else 0)), b=256)
                    self._p2g(self.X[nx], self.V[nx], self.FF[nx], h, True)          # 재배치 P2G (질량·속도·F̄; 바닥 후보 표시는 프레임 것)
                    self.Uc.zero_()
                else:
                    self.mv.zero_()
                    EX.launch("ex_rep_vel_p2g", Np, (np.int32(Np), EX.P(self.rows), EX.P(self.lamb), EX.P(self.mA), np.int32(1 if (a.mass == "tet" and not RWL) else 0),
                                                     EX.P(self.vn), EX.P(self.mv), np.int32(1 if RW4 else 0), EX.P(self.gl if a.affine else None),
                                                     EX.P(self.pinv), EX.P(self.npos), EX.P(self.X[0])), b=256)
                    EX.launch("ex_rep_vel_fin", self.Mcap, (EX.P(self.sz), EX.P(self.ml if RWL else self.mn), EX.P(self.mv), EX.P(self.vn)), b=256)
        elif not a.no_fuse:
            # 융합 판: [사면체 힘 (F̄ 나누기 포함, 쓴 뒤 비움)] [바닥 후보 힘] [노드 걸음 (속도 나누기 포함, 쓴 뒤 비움)] [G2P + locate + P2G + 분할]
            Mc, Tc = self.Mcap, self.Tcap
            self.f.zero_()
            self.gn.zero_()
            for j in range(a.N):
                xa, Fa = self.X[j % 2], self.FF[j % 2]
                xb, vb, Fb = self.X[(j + 1) % 2], self.V[(j + 1) % 2], self.FF[(j + 1) % 2]
                EX.launch("ex_tet_force_sub", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.gl), EX.P(self.FTs), EX.P(self.VTs), np.float32(mu),
                                                   np.float32(lam), EX.P(self.f)))
                if FLOOR:
                    EX.launch("ex_near_force32" if a.near32 else "ex_near_force", Np, (EX.P(self.sz), np.int32(0), EX.P(self.nidx), EX.P(self.x0c),
                              EX.P(self.F0c), EX.P(self.cc), EX.P(None), EX.P(xa), EX.P(Fa), EX.P(self.F0i), EX.P(self.rows), EX.P(self.lamb),
                              EX.P(self.gl), EX.P(self.pinv), EX.P(self.mA), EX.P(self.volA), np.float64(h), *near_args(0), np.float64(mu), np.float64(lam),
                              EX.P(self.gn)), b=64)
                EX.launch("ex_node_step_sub", Mc, (EX.P(self.sz), EX.P(self.mn), EX.P(self.mv), EX.P(self.f), EX.P(self.gn if FLOOR else None),
                                                   EX.P(self.dU), np.float32(gv[0]), np.float32(gv[1]), np.float32(gv[2]), np.float32(hs),
                                                   EX.P(self.fx if self.fixd else None), EX.P(self.vbc)), b=256)
                EX.launch("ex_g2p_p2g_seg" if self.SORT else "ex_g2p_p2g", Np, (np.int32(Np), EX.P(xa), EX.P(Fa), EX.P(self.dU), EX.P(self.rows), EX.P(self.lamb), EX.P(self.pinv),
                                             EX.P(self.gl), np.float32(hs), np.float32(DAMP), EX.P(xb), EX.P(vb), EX.P(Fb), np.int32(1 if j < a.N - 1 else 0),
                                             EX.P(self.gpar), EX.P(self.ipar), EX.P(self.ncm), EX.P(self.tcm), EX.P(self.tmark), EX.P(self.mA), EX.P(self.volA),
                                             EX.P(self.nslot), np.int32(1 if a.mass == "tet" else 0), EX.P(self.mn), EX.P(self.mv), EX.P(self.VTs),
                                             EX.P(self.FTs), EX.P(self.sz)), b=256)
        else:
            for j in range(a.N):
                xa, va, Fa = self.X[j % 2], self.V[j % 2], self.FF[j % 2]
                xb, vb, Fb = self.X[(j + 1) % 2], self.V[(j + 1) % 2], self.FF[(j + 1) % 2]
                if j > 0:                                                           # 서브스텝 0 의 P2G 는 프레임 분할과 함께 (frame)
                    self._p2g(xa, va, Fa, hs, True)
                self._step(None, xa, Fa)
                EX.launch("ex_g2p_sub", Np, (np.int32(Np), EX.P(xa), EX.P(self.dU), EX.P(self.rows), EX.P(self.lamb), EX.P(self.gl), EX.P(self.pinv),
                                             EX.P(Fa), np.int32(0), *PL, np.float64(a.eps_st), np.float64(a.eps_sp), np.float64(XN),
                                             np.float64(hs), np.float64(DAMP), EX.P(xb), EX.P(vb), EX.P(Fb)), b=256)

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            g.capture_begin()
            self._substeps()
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = g
        self.n_cap += 1

    def _lagdiag(self, i):
        """lag 진단: 서브스텝을 파이썬으로 돌며 최대 |v_n| 표본, 프레임 끝 det A_T (점유 사면체) 분포."""
        M, T = self.M, self.T
        mn = self.mn[:M]
        z = self.npos[:, 2]
        zf = float(PL[2]) if HASPL else float("nan")
        smp = []
        K = max(1, a.N // 8)
        first_bad = None
        acc_s = None
        # 절 Q 메시 진단 (재평균 없음): 노드 질량 분포, 이웃 셀 점유율, 노드별 국소 안정 비 r_n = h_s ω_n / 2 (Gershgorin: ω² ≤ Σ_T V_T (λ+2μ) |∇λ_i| Σ_j |∇λ_j| / m_n),
        # 서브스텝 표본마다 최대 |v| 노드의 점유율·r_n·접촉 질량 비 α (= 접촉 입자 ¼ m_p 합 / m_n)
        tets_, gl_ = self.tets[:T], self.gl[:T].view(T, 4, 3)
        VT_ = self.VTs[:T]
        P4 = self.npos[tets_.reshape(-1)].view(T, 4, 3)
        V0 = torch.linalg.det(P4[:, 1:] - P4[:, :1]).abs() / 6.0
        gn_ = gl_.norm(dim=-1)                                                      # [T,4]
        w_ = VT_[:, None] * (lam + 2 * mu) * gn_ * gn_.sum(-1, keepdim=True)
        om2 = torch.zeros(M, device=dev).index_add_(0, tets_.reshape(-1), w_.reshape(-1)) / mn.clamp_min(1e-30)
        rn = hs * om2.sqrt() / 2
        vo = torch.zeros(M, device=dev).index_add_(0, tets_.reshape(-1), VT_[:, None].expand(T, 4).reshape(-1))
        v0 = torch.zeros(M, device=dev).index_add_(0, tets_.reshape(-1), V0[:, None].expand(T, 4).reshape(-1))
        occn = vo / v0.clamp_min(1e-30)
        live = mn > 0
        med = mn[live].median()
        light = live & (mn < 0.025 * med)
        nn_ = int(self.sz[2])
        ni_ = self.nidx[:nn_]

        def contact_alpha():
            if not VFL or nn_ == 0:
                return torch.zeros(M, device=dev)
            r_ = self.rows[ni_]
            u_ = (self.lamb[ni_][..., None] * self.Uc[r_]).sum(1)
            s_ = ((self.Xf[ni_] + u_ - torch.tensor(pt_, device=dev)) * torch.tensor(nr_, device=dev)).sum(-1)
            c_ = s_ < 0
            accn = torch.zeros(M, device=dev).index_add_(0, r_[c_].reshape(-1), (0.25 * self.mA[ni_][c_])[:, None].expand(-1, 4).reshape(-1))
            return accn / mn.clamp_min(1e-30)
        mstat = dict(m_med=float(med), n_live=int(live.sum()), n_light=int(light.sum()), n_vlight=int((live & (mn < 0.001 * med)).sum()),
                     occ_light_med=float(occn[light].median()) if light.any() else None, occ_all_med=float(occn[live].median()),
                     r_max=float(rn[live].max()), r_q999=float(torch.quantile(rn[live], 0.999)), r_light_max=float(rn[light].max()) if light.any() else None)
        al0 = contact_alpha()
        mstat.update(n_contact_nodes=int((al0 > 0).sum()), n_contact_light=int(((al0 > 0) & light).sum()))
        msmp = []
        LD = a.lagdump > 0
        if LD:                                                                      # 덤프 (읽기만 -- 계산 경로 불변)
            def detA_now():
                Q = torch.eye(3, device=dev) + torch.einsum("tia,tib->tab", self.Uc[tets_.reshape(-1)].view(T, 4, 3), gl_)
                return _det3(Q)
            dU_ = torch.empty(a.N + 1, M, 3, device=dev)
            dV_ = torch.empty(a.N + 1, M, 3, device=dev)
            dD_ = torch.empty(a.N + 1, T, device=dev)
            dAl_ = torch.zeros(a.N, M, device=dev)
            dF_ = torch.full((a.N, M, 3), float("nan"), device=dev)                # 서브스텝 j 의 내부력 (상태 j 의 U 로 계산된 것, 걸음 직전)
            self._fj = 0
            self._fhook = lambda: dF_[self._fj].copy_(self.f[:M])
            dU_[0], dV_[0], dD_[0] = self.Uc[:M], self.vn[:M], detA_now()
        if VFL:                                                                     # 절 O 판: 같은 서브스텝 커널 (_step_lag) -- f 는 걸음 안에서 비워져 |f/m| 표본은 0
            self.f.zero_()
            self.gn.zero_()
            self.acc.zero_()
        for j in range(a.N):
            if LD:
                self._fj = j
                dAl_[j] = contact_alpha()                                           # 이 서브스텝 접촉 판정 (서브스텝 시작 U 기준) 의 α
            if VFL:
                self._step_lag()
                if first_bad is None and not bool(torch.isfinite(self.vn[:M]).all()):
                    first_bad = j
            else:
                self._step(self.Uc, None, None)
            if LD:
                dU_[j + 1], dV_[j + 1], dD_[j + 1] = self.Uc[:M], self.vn[:M], detA_now()
            if j % K == 0 or j == a.N - 1:
                sp = self.vn[:M].norm(dim=-1)
                k = int(sp.argmax())
                gnm = (self.gn[:M].norm(dim=-1) / mn.clamp_min(1e-30)).max() if FLOOR else torch.zeros(())
                fm = (self.f[:M].norm(dim=-1) / mn.clamp_min(1e-30)).max()
                smp.append([j, float(sp.max()), float(z[k] - zf), float(mn[k] / mn[mn > 0].median()), float(fm), float(gnm), int((~torch.isfinite(sp)).sum())])
                al = contact_alpha()
                msmp.append([j, float(sp[k]), float(mn[k] / med), float(occn[k]), float(rn[k]), float(al[k]), int(((al > 0) & light).sum())])
        self._fhook = None
        tets, gl = self.tets[:T], self.gl[:T].view(T, 4, 3)
        U = self.Uc[tets.reshape(-1)].view(T, 4, 3)
        A = torch.eye(3, device=dev) + torch.einsum("tia,tib->tab", U, gl)
        dA = torch.linalg.det(A)
        occ = self.VTs[:T] > 0
        do = dA[occ]
        FTo = self.FT[:T].view(T, 3, 3)[occ]
        dF = torch.linalg.det(FTo)
        fin_ = lambda Mx: Mx[torch.isfinite(Mx).all(-1).all(-1)].double()
        sv_ = lambda Mx: (torch.linalg.svdvals(fin_(Mx)) if fin_(Mx).shape[0] else torch.full((1, 3), float("nan"), device=dev, dtype=torch.float64))
        sv = sv_(FTo)
        svA = sv_(A[occ] @ FTo)
        svP = sv_(self._Fp)                                                         # 프레임 시작 입자 F
        ke = float((mn * (self.vn[:M] ** 2).sum(-1)).sum() * 0.5)
        rec = dict(t=i, samples=smp, detA_min=float(do.min()) if do.numel() else None, detA_lt05=int((do < 0.5).sum()), detA_le0=int((do <= 0).sum()),
                   detFT_min=float(dF.min()) if dF.numel() else None, detFT_le0=int((dF <= 0).sum()), n_occ=int(occ.sum()), ke=ke,
                   svFT_max=float(sv[:, 0].max()), svFT_min=float(sv[:, -1].min()), svFT_q999=float(torch.quantile(sv[:, 0].float(), 0.999)),
                   svAF_max=float(svA[:, 0].max()), svAF_min=float(svA[:, -1].min()), first_bad_sub=first_bad,
                   svFp_max=float(svP[:, 0].max()), svFp_min=float(svP[:, -1].min()), svFp_q999=float(torch.quantile(svP[:, 0].float(), 0.999)),
                   Umax=float(self.Uc[:M].norm(dim=-1).max()), mesh=mstat, mesh_samples=msmp)
        if LD:
            if not hasattr(self, "_ring"):
                import collections
                self._ring = collections.deque(maxlen=int(a.lagdump))
            cpu = lambda q: q.detach().cpu()
            self._ring.append(dict(t=i, N=a.N, h=h, hs=hs, plane=(pt_, nr_) if HASPL else None, m_med=float(med), first_bad_sub=first_bad,
                                   npos=cpu(self.npos), mn=cpu(mn), occn=cpu(occn), rn=cpu(rn), tets=cpu(tets_).int(), V0=cpu(V0), VT=cpu(VT_),
                                   detFT=cpu(_det3(self.FT[:T].view(T, 3, 3))),
                                   xp=cpu(self._xp), rows=cpu(self.rows).int(), lamb=cpu(self.lamb), nidx=cpu(ni_),
                                   U=cpu(dU_), V=cpu(dV_), detA=cpu(dD_), alpha=cpu(dAl_),
                                   # 10-30 확장 (질량 vs 힘 진단): 걸음 직전 내부력, 프레임 셀 상태·기준 ∇λ, 셀 안 입자 질량·수, 격자
                                   f=cpu(dF_), FT=cpu(self.FT[:T]), gl=cpu(self.gl[:T]), Ai=cpu(self.Ai), mu=float(mu), lam=float(lam), node_h=float(a.node_h),
                                   mT=cpu(torch.zeros(T, device=dev).index_add_(0, self.pinv[:Np], m)),
                                   npt=cpu(torch.zeros(T, device=dev).index_add_(0, self.pinv[:Np], torch.ones_like(m))),
                                   mass_def=("bspline" if BSL else a.mass), g2p=(a.g2p if BSL else "bary")))
        with open(os.path.join(a.out, "lagdiag.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"    [lag 진단] t {i + 1}: 최대|v_n| 표본 " + " ".join(f"{q[1]:.3g}" for q in smp) + f" (마지막: 노드 높이-바닥 {smp[-1][2]:+.4f}, 질량/중앙 {smp[-1][3]:.3g}), "
              f"최대 |f/m| {smp[-1][4]:.3g}, 최대 |바닥 g/m| {smp[-1][5]:.3g}, det A_T 최소 {rec['detA_min']}, <0.5 {rec['detA_lt05']}, ≤0 {rec['detA_le0']}, "
              f"det F̄ 최소 {rec['detFT_min']}, σ(F̄) 최대 {rec['svFT_max']:.3g} (99.9% {rec['svFT_q999']:.3g}) 최소 {rec['svFT_min']:.3g}, "
              f"[메시] 질량 중앙 {mstat['m_med']:.3g}, 가벼운 (<2.5%) {mstat['n_light']}/{mstat['n_live']} (점유율 중앙 {mstat['occ_light_med']}, 전체 {mstat['occ_all_med']:.3g}), "
              f"r_n 최대 {mstat['r_max']:.3g} (99.9% {mstat['r_q999']:.3g}, 가벼운 {mstat['r_light_max']}), 접촉 노드 {mstat['n_contact_nodes']} (가벼운 {mstat['n_contact_light']}), "
              f"최대|v| 노드 [j,|v|,m/중앙,점유,r,α] {msmp[-1][:6]} "
              f"σ(A F̄) 최대 {rec['svAF_max']:.3g} 최소 {rec['svAF_min']:.3g}, 입자 F σ 최대 {rec['svFp_max']:.3g} (99.9% {rec['svFp_q999']:.3g}) 최소 {rec['svFp_min']:.3g}, 첫 비유한 서브스텝 {first_bad}, KE {ke:.4g}, 최대|U| {rec['Umax']:.4g}", flush=True)

    def bsdiag(self, i):
        """절 S 진단 (측정 밖, 이 프레임 P2G 직후 상태 -- mn·VTs·rows·λ 는 서브스텝이 바꾸지 않는다):
        노드 질량 세 정의 (B-스플라인 Σ w m / 셀 집중 m_T/4 (절 R lag 판) / Σλm) 분포, 국소 안정 비 (Gershgorin r_n = h_s ω_n / 2,
        ω_n² ≤ Σ_T V_T (λ+2μ) |∇λ_n| Σ_j |∇λ_j| / m_n) 를 새 정의 (V_T = V_c/6, m_n B-스플라인) 와 절 R 정의 (V_T = 셀 안 입자 부피, m_T/4) 로."""
        M, T = self.M, self.T
        mb = self.mn[:M].double()
        r_, l_, pv = self.rows, self.lamb.double(), self.pinv
        md = m.double()
        mlam = torch.zeros(M, device=dev, dtype=torch.float64).index_add_(0, r_.reshape(-1), (l_ * md[:, None]).reshape(-1))
        mT = torch.zeros(T, device=dev, dtype=torch.float64).index_add_(0, pv, md)
        VTo = torch.zeros(T, device=dev, dtype=torch.float64).index_add_(0, pv, vol.double())
        tets_ = self.tets[:T]
        mt4 = torch.zeros(M, device=dev, dtype=torch.float64).index_add_(0, tets_.reshape(-1), (mT[:, None] / 4).expand(T, 4).reshape(-1))
        gl_ = self.gl[:T].view(T, 4, 3).double()
        gn_ = gl_.norm(dim=-1)
        k_ = (lam + 2 * mu) * gn_ * gn_.sum(-1, keepdim=True)

        def stab(VT_, mn_):
            om2 = torch.zeros(M, device=dev, dtype=torch.float64).index_add_(0, tets_.reshape(-1), (VT_[:, None] * k_).reshape(-1))
            act = (om2 > 0) & (mn_ > 0)
            rn = hs * (om2[act] / mn_[act]).sqrt() / 2
            q = torch.quantile(rn.float(), torch.tensor([0.5, 0.99, 0.999, 1.0], device=dev)).tolist() if rn.numel() else [float("nan")] * 4
            return dict(r_q50=q[0], r_q99=q[1], r_q999=q[2], r_max=q[3], N_need=a.N * q[3], n_r_gt1=int((rn > 1).sum()), n_act=int(act.sum()))

        def dist(mm):
            live = mm > 0
            med = mm[live].median()
            rel = (mm[live] / med).float()
            q = torch.quantile(rel, torch.tensor([0.0, 1e-4, 1e-3, 0.01, 0.1, 0.5], device=dev)).tolist()
            return dict(n_live=int(live.sum()), med=float(med), q_rel=q, n_lt025=int((rel < 0.025).sum()), n_lt001=int((rel < 0.001).sum()), tot=float(mm.sum()))
        VTn = self.VTs[:T].double()
        live_t = VTn > 0
        zn = (mb <= 0)
        # 질량 0 노드가 V>0 셀에 걸린 수 (bsl_tfin 이 뺀 사면체)
        rec = dict(t=i, M=M, T=T, T_live_new=int(live_t.sum()), T_live_old=int((VTo > 0).sum()),
                   m_bs=dist(mb), m_t4=dist(mt4), m_lam=dist(mlam), m_p=float(md.sum()),
                   V_p=float(vol.double().sum()), V_new=float(VTn.sum()), V_drop=float(self.bst[0]), T_drop=int(self.bst[1]),
                   n_zero_mass_nodes=int(zn.sum()), m_redist=float(self.bst[2]), V_redist=float(self.bst[3]),
                   stab_new=stab(VTn, mb), stab_old=stab(VTo, mt4))
        # 새 정의에서 가장 가벼운 노드들 (상대 질량 < 2.5 %) 의 이웃 셀 부피 / 질량 비 vs 중앙
        with open(os.path.join(a.out, "bsdiag.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        mbd, mtd, mld, sn, so = rec["m_bs"], rec["m_t4"], rec["m_lam"], rec["stab_new"], rec["stab_old"]
        fq = lambda d_: "/".join(f"{q:.2g}" for q in d_["q_rel"])
        print(f"    [S 진단] t {i + 1}: 노드 질량 /중앙 분위 (0,1e-4,1e-3,.01,.1,.5) B-스플라인 {fq(mbd)} (<2.5% {mbd['n_lt025']}/{mbd['n_live']}) | "
              f"m_T/4 {fq(mtd)} (<2.5% {mtd['n_lt025']}/{mtd['n_live']}) | Σλm {fq(mld)} (<2.5% {mld['n_lt025']}/{mld['n_live']}); "
              f"총질량 입자 {rec['m_p']:.6g} 노드 B {mbd['tot']:.6g} (정규화로 옮긴 질량 {rec['m_redist']:.4g}); 부피 입자 {rec['V_p']:.6g} 사면체 {rec['V_new']:.6g} (뺀 {rec['V_drop']:.3g}, 사면체 {rec['T_drop']}, 정규화로 옮긴 부피 {rec['V_redist']:.4g}); "
              f"국소 안정 r_n (h_s ω/2) 중앙/99.9%/최대 새 {sn['r_q50']:.3g}/{sn['r_q999']:.3g}/{sn['r_max']:.3g} (필요 N ≈ {sn['N_need']:.0f}, r>1 {sn['n_r_gt1']}) "
              f"절 R 정의 {so['r_q50']:.3g}/{so['r_q999']:.3g}/{so['r_max']:.3g} (필요 N ≈ {so['N_need']:.0f})", flush=True)

    def frame(self, i, x, v, F):
        self.grid(x, v, i * h)
        Fv = F.reshape(Np, 9)
        lagm = a.mode == "lag"
        if lagm:
            xa, va, Fa = x, v, Fv
            if VFL:
                self.Xf = self.X[0]                                                 # 프레임 시작 입자 위치 (상주 버퍼 -- 그래프 주소 고정; 접촉 판정 = x + Σλ U)
                self.Xf.copy_(x)
            if REP:                                                                 # 절 P: 입자 상태 상주 버퍼 (블록 사이 G2P 가 [0]<->[1] 로 오간다)
                xa, va, Fa = self.X[0], self.V[0], self.FF[0]
                xa.copy_(x)
                va.copy_(v)
                Fa.copy_(Fv)
        else:
            xa, va, Fa = self.X[0], self.V[0], self.FF[0]
            if self.SORT:
                # 입자를 사면체 번호로 정렬 (프레임마다 한 번) -> 서브스텝 P2G 를 워프 구간 합으로
                EX.launch("ex_locate", Np, (np.int32(Np), EX.P(x), EX.P(self.gpar), EX.P(self.ipar), EX.P(self.ncm), EX.P(self.tcm), EX.P(self.tmark),
                                            EX.P(self.rows), EX.P(self.lamb), EX.P(self.pinv), EX.P(self.sz)), b=256)
                self.perm.copy_(torch.sort(self.pinv)[1])
                torch.index_select(x, 0, self.perm, out=xa)
                torch.index_select(v, 0, self.perm, out=va)
                torch.index_select(Fv, 0, self.perm, out=Fa)
                torch.index_select(m, 0, self.perm, out=self.mA)
                torch.index_select(vol, 0, self.perm, out=self.volA)
            else:
                xa.copy_(x)
                va.copy_(v)
                Fa.copy_(Fv)
        if BSL:
            self._p2g_bs(xa, va, Fa)
        else:
            self._p2g(xa, va, Fa, h, False)                                         # 프레임 분할 (관성 예측 h v_n + h² g 의 끝 높이)
        if FLOOR:
            EX.launch("ex_near_init", Np, (EX.P(self.sz), np.int32(0), EX.P(self.nidx), EX.P(xa), EX.P(va), EX.P(Fa), np.float32(h), np.float32(gv[0]),
                                           np.float32(gv[1]), np.float32(gv[2]), EX.P(self.x0c), EX.P(self.F0c), EX.P(self.F0i), EX.P(self.cc)), b=256)
        if lagm:
            self.Uc.zero_()
            if VFL:
                self.ncon.zero_()
            self.rst.zero_()
        if lagm and a.lagdiag:
            self._Fp = Fv.view(Np, 3, 3)
            self._xp = x
            self._lagdiag(i)
        elif a.no_graph:
            self._substeps()
        else:
            if self.graph is None:
                bufs = (self.vn, self.Uc, self.mn, self.mv, self.VTs, self.FTs, self.FT, self.rows, self.lamb, self.pinv)
                if REP:
                    bufs = bufs + tuple(self.X) + tuple(self.V) + tuple(self.FF) + (self.sz,)
                if a.affine:
                    bufs = bufs + (self.CL,)
                if RWL:
                    bufs = bufs + (self.ml,)
                snap = [t.clone() for t in bufs]
                xs_snap = [t.clone() for t in (xa, va, Fa)]
                self._substeps()                                                    # 데우기 (NVRTC 컴파일·지연 할당) -- 결과 버림
                for t, s_ in zip(bufs, snap):
                    t.copy_(s_)
                if not lagm:
                    for t, s_ in zip((xa, va, Fa), xs_snap):
                        t.copy_(s_)
                self.ncon.zero_()
                self.rst.zero_()
                if LIM:
                    self.alph.fill_(1065353216)
                self.capture()
            self.graph.replay()
        if lagm:
            ci = ((NB - 1) % 2) if REPF else 0                                       # 마지막 블록의 입자 상태 버퍼
            xin, Fin = (self.X[ci], self.FF[ci]) if REP else (x, Fv)
            x2, v2, F2 = self.X[1 - ci], self.V[1 - ci], self.FF[1 - ci]
            if G2PBS:
                EX.launch("bsl_g2p", Np, (np.int32(Np), EX.P(xin), EX.P(self.Uc), EX.P(self.vn), EX.P(self.gpar), EX.P(self.ipar), EX.P(self.ncm),
                                          EX.P(Fin), EX.P(x2), EX.P(v2), EX.P(F2), np.int32(BSN)), b=128)
            else:
                EX.launch("ex_g2p_lag", Np, (np.int32(Np), EX.P(xin), EX.P(self.Uc), EX.P(self.vn), EX.P(self.rows), EX.P(self.lamb), EX.P(self.gl),
                                             EX.P(self.pinv), EX.P(Fin), EX.P(x2), EX.P(v2), EX.P(F2), EX.P(self.CL if a.affine else None), np.int32(1 if RW4 else 0)), b=256)
            uc, xc_, Fc_ = self.Uc, None, None
        else:
            k = a.N % 2
            x2, v2, F2 = self.X[k], self.V[k], self.FF[k]
            uc, xc_, Fc_ = None, x2, F2
        if FLOOR:
            EX.launch("ex_near_fin", Np, (EX.P(self.sz), np.int32(0), EX.P(self.nidx), EX.P(self.x0c), EX.P(self.F0c), EX.P(uc), EX.P(xc_), EX.P(Fc_),
                                          EX.P(self.F0i), EX.P(self.rows), EX.P(self.lamb), EX.P(self.gl), EX.P(self.pinv), *near_args(0), np.float64(h),
                                          EX.P(x2), EX.P(v2), EX.P(F2)), b=128)
            if a.affine and a.near_c0:
                nn_ = int(self.sz[2])
                self.CL[self.nidx[:nn_]] = 0.0
        if self.SORT:
            self.OX.index_copy_(0, self.perm, x2)
            self.OV.index_copy_(0, self.perm, v2)
            self.OF.index_copy_(0, self.perm, F2)
            x2, v2, F2 = self.OX, self.OV, self.OF
        return x2, v2, F2.view(Np, 3, 3)



# ---------------------------------------------------------------- 2 차 B-스플라인 판 (--interp bspline2, 절 M)
class BS:
    """밀집 격자 (월드 고정 정육면체 격자 = 기존 노드 격자, 상자만 프레임/서브스텝마다) 위 MPM 식 명시적 서브스텝 -- 식은 fem_bs_cuda 머리말.
    opt  : float32 대수, G2P + 다음 P2G 융합, 격자 상자 프레임당 한 번 (서브스텝 재사용, 동기화 1 번), 서브스텝 N 번 + 마지막 G2P 를 CUDA 그래프 하나로
    기준 : float64 대수, 커널 따로 (G2P / P2G / 바닥 후보 / 셀 / 격자), 격자 상자 서브스텝마다 다시 (동기화), 파이썬 서브스텝 루프"""

    def __init__(self, opt):
        self.opt = opt
        self.R = torch.float32 if opt else torch.float64
        self.sfx = "f" if opt else "d"
        self.lat = SX.lattice(float(a.node_h), SX.HZ_CUBE * float(a.node_h), device=dev, dtype=torch.float64)
        self.dx = float(self.lat.A[:, 0].norm())
        self.gpar = torch.zeros(12, device=dev, dtype=self.R)
        self.ipar = torch.zeros(4, dtype=torch.int32, device=dev)
        self.sz = torch.zeros(4, dtype=torch.int32, device=dev)
        self.QUAD = 0 if a.quad == "particle" else 1
        self.gK = BSC.cell_grad_table(self.lat.Ai, a.K).to(dev, self.R).contiguous() if self.QUAD == 1 else None
        self.Pcap = 0
        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=dev, dtype=dt)
        self.nslot = z(Np, dt=torch.int32) - 1
        self.nidx = z(Np, dt=torch.long)
        self.x0c, self.F0c, self.F0i, self.cc = z(Np, 3), z(Np, 9), z(Np, 9), z(Np, 3)
        self.X = [z(Np, 3), z(Np, 3)]
        self.V = [z(Np, 3), z(Np, 3)]
        self.FF = [z(Np, 9), z(Np, 9)]
        self.APIC = bool(a.apic)
        if self.APIC and (not opt or a.psort):
            raise SystemExit("--apic 는 --opt 판 (정렬 없음) 만")
        self.CC = [z(Np, 9), z(Np, 9)] if self.APIC else [None, None]
        self.Cnow = 0                                                               # 지금 상태의 C 가 든 칸
        self.graph = None
        self.n_cap = 0
        self.bc_key = None
        self.M, self.T = 0, 0
        self.PLr = [np.float32(q) for q in PLf] if opt else PL
        self.SORT = bool(a.psort and opt)
        self.mA, self.volA = m, vol
        if self.SORT:
            self.mA, self.volA = m.clone(), vol.clone()
            self.OX, self.OV, self.OF = z(Np, 3), z(Np, 3), z(Np, 9)

    def _alloc(self, Pn):
        if Pn <= self.Pcap:
            return
        self.Pcap = int(Pn * 1.3) + 64
        z = lambda n, dt=self.R: torch.zeros(n, device=dev, dtype=dt)
        self.gm, self.gmv, self.vg = z(self.Pcap), z(self.Pcap * 3), z(self.Pcap * 3)
        self.cV = z(self.Pcap) if self.QUAD == 1 else None
        self.cVF = z(self.Pcap * 9) if self.QUAD == 1 else None
        self.fx = torch.zeros(self.Pcap, dtype=torch.uint8, device=dev)
        self.vbc = torch.zeros(self.Pcap * 3, device=dev)
        self.bc_key = None
        self.graph = None

    def box(self, x, v, now, motion):
        """밀집 상자: 입자 범위 + 여백 (motion: 프레임 동안 이동 max|v|·h·1.5) + 3 dx, base ∈ [0, n-3] 이 되게 정수 범위 ±2. 동기화 1 번."""
        lat = self.lat
        lo_ = x.min(0).values.double()
        hi_ = x.max(0).values.double()
        vmax = v.norm(dim=-1).max().double()
        ext = (hi_ - lo_).max().clamp_min(1e-12)
        pad = 0.05 * ext + 3.0 * self.dx + (vmax * h * 1.5 if motion else 0.0)
        b0, b1 = lo_ - pad, hi_ + pad
        cor = torch.stack([torch.stack([b0[0] if (mm & 1) else b1[0], b0[1] if (mm & 2) else b1[1], b0[2] if (mm & 4) else b1[2]]) for mm in range(8)], 0)
        y = cor @ lat.Ai.T
        imin = torch.floor(y.min(0).values).long() - 2
        imax = torch.ceil(y.max(0).values).long() + 2
        ii = torch.cat([imin, imax, (vmax * h / self.dx).ceil().long().view(1)]).tolist()       # 동기화
        if motion and float(ii[6]) * self.dx > 0.25 * EXT:
            raise _Diverge(f"프레임 이동 max|v|·h > 0.25 EXT ({ii[6]} 셀)")
        dims = [max(ii[3 + k] - ii[k] + 1, 4) for k in range(3)]
        Pn = dims[0] * dims[1] * dims[2]
        if not hasattr(self, "P0"):
            self.P0 = Pn
        if Pn > 8 * self.P0:
            raise _Diverge(f"격자 상자 {Pn} > 8 x 첫 프레임 {self.P0}")
        imin_l = ii[:3]
        lo = lat.A @ torch.tensor(imin_l, device=dev, dtype=torch.float64)
        self.gpar[:3] = lo.to(self.R)
        self.gpar[3:] = lat.Ai.reshape(-1).to(self.R)
        self.ipar.copy_(torch.tensor(dims + [0], dtype=torch.int32), non_blocking=True)
        self._alloc(Pn)
        self.M = Pn
        if BC_SEL is not None:
            # cuboid 만 (입자 구속 없음): train_deform.cfg_bc_nodes 의 cuboid 판정 그대로 (상자 안 노드 = v_bc) -- 밀집 상자의 모든 노드 위치에서
            act = tuple((float(bc.get("start_time", 0.0)) <= now < float(bc.get("end_time", 1e3)),
                         int(bc.get("reset", 0)) == 1 and float(bc.get("end_time", 1e3)) <= now < float(bc.get("end_time", 1e3)) + 15.0 * h)
                        for bc in (cfg.get("boundary_conditions") or []) if bc.get("type") == "cuboid")
            key = (tuple(imin_l), tuple(dims), act)
            if key != self.bc_key:
                ar = [torch.arange(dims[k], device=dev) for k in range(3)]
                ijk = torch.stack(torch.meshgrid(*ar, indexing="ij"), -1).reshape(-1, 3).double()
                npos = ((ijk + torch.tensor(imin_l, device=dev, dtype=torch.float64)) @ lat.A.T).float()
                fx, vb = RT.cfg_bc_nodes(cfg, [], npos, torch.zeros(1, 4, dtype=torch.long, device=dev), torch.zeros(1, 4, device=dev), m[:1], now, h)
                self.fx.zero_()
                self.vbc.zero_()
                if fx is not None:
                    self.fx[:Pn] = fx.to(torch.uint8)
                    self.vbc[:Pn * 3] = vb.float().reshape(-1)
                self.bc_key = key
        return Pn

    def _k(self, name):
        return f"{name}_{self.sfx}"

    def _fix(self):
        return self.fx if BC_SEL is not None else None

    def _near(self, xc, Fc):
        if FLOOR:
            BSC.launch(self._k("bs_near"), Np, (BSC.P(self.sz), BSC.P(self.nidx), BSC.P(self.x0c), BSC.P(self.F0c), BSC.P(self.cc), BSC.P(xc), BSC.P(Fc),
                                                 BSC.P(self.F0i), BSC.P(self.mA), BSC.P(self.volA), np.float64(h), *PL, np.float64(a.eps_st), np.float64(a.eps_sp),
                                                 np.float64(XN), np.float64(mu), np.float64(lam), BSC.P(self.gpar), BSC.P(self.ipar),
                                                 self._c(hs), BSC.P(self.gmv)), b=64)

    def _c(self, val):
        return np.float32(val) if self.opt else np.float64(val)

    def _grid_step(self, hs_, damp, fix):
        if self.QUAD == 1:
            BSC.launch(self._k("bs_cell"), self.Pcap, (BSC.P(self.ipar), BSC.P(self.cV), BSC.P(self.cVF), BSC.P(self.gK), self._c(mu), self._c(lam), self._c(hs_),
                                                       BSC.P(self.gmv)), b=128)
        BSC.launch(self._k("bs_node"), self.Pcap, (BSC.P(self.ipar), BSC.P(self.gm), BSC.P(self.gmv), BSC.P(self.vg), self._c(gv[0]), self._c(gv[1]),
                                                   self._c(gv[2]), self._c(hs_), self._c(damp), BSC.P(fix), BSC.P(self.vbc)), b=256)

    def _p2g_d(self, xa, va, Fa, quad, hs_):
        BSC.launch("bs_p2g_d", Np, (np.int32(Np), BSC.P(xa), BSC.P(va), BSC.P(Fa), BSC.P(self.mA), BSC.P(self.volA), BSC.P(self.nslot if FLOOR else None), np.int32(quad),
                                    BSC.P(self.gpar), BSC.P(self.ipar), np.float64(hs_), np.float64(mu), np.float64(lam), BSC.P(self.gm), BSC.P(self.gmv),
                                    BSC.P(self.cV), BSC.P(self.cVF), BSC.P(self.sz)), b=128)

    def _g2p_p2g_f(self, xa, va, Fa, xb, vb, Fb, do_g2p, do_p2g, quad, hs_, Ca=None, Cb=None):
        BSC.launch("bs_g2p_p2g_f", Np, (np.int32(Np), BSC.P(xa), BSC.P(va), BSC.P(Fa), BSC.P(self.vg), BSC.P(self.gpar), BSC.P(self.ipar), np.float32(hs_),
                                        np.int32(do_g2p), np.int32(do_p2g), BSC.P(xb), BSC.P(vb), BSC.P(Fb), BSC.P(self.mA), BSC.P(self.volA),
                                        BSC.P(self.nslot if FLOOR else None), np.int32(quad), np.float32(mu), np.float32(lam), BSC.P(self.gm), BSC.P(self.gmv),
                                        BSC.P(self.cV), BSC.P(self.cVF), BSC.P(self.sz), BSC.P(self.gK), BSC.P(Ca), BSC.P(Cb)), b=128)

    def _substeps(self):
        """opt: 서브스텝 N 번 (융합 G2P+P2G -> 바닥 후보 -> 셀 -> 격자) + 마지막 G2P. 상태 S[j%2] = 서브스텝 j 시작 상태."""
        S = lambda k: (self.X[k % 2], self.V[k % 2], self.FF[k % 2])
        CS = lambda k: self.CC[k % 2]
        for j in range(a.N):
            if j == 0:
                xa, va, Fa = S(0)
                self._g2p_p2g_f(xa, va, Fa, xa, va, Fa, 0, 1, self.QUAD, hs, CS(0), None)
            else:
                xa, va, Fa = S(j - 1)
                xb, vb, Fb = S(j)
                self._g2p_p2g_f(xa, va, Fa, xb, vb, Fb, 1, 1, self.QUAD, hs, CS(j - 1), CS(j))
            xc, _, Fc = S(j)
            self._near(xc, Fc)
            self._grid_step(hs, DAMP, self._fix())
        xa, va, Fa = S(a.N - 1)
        xb, vb, Fb = S(a.N)
        self._g2p_p2g_f(xa, va, Fa, xb, vb, Fb, 1, 0, self.QUAD, hs, CS(a.N - 1), CS(a.N))

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            g.capture_begin()
            self._substeps()
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = g
        self.n_cap += 1

    def frame(self, i, x, v, F):
        now = i * h
        Fv = F.reshape(Np, 9)
        xa, va, Fa = self.X[0], self.V[0], self.FF[0]
        xa.copy_(x)
        va.copy_(v)
        Fa.copy_(Fv)
        if self.APIC and self.Cnow != 0:
            self.CC[0].copy_(self.CC[self.Cnow])                                   # 지난 프레임 끝 C -> 칸 0
        self.box(xa, va, now, True if self.opt else False)
        if self.SORT:
            # B-스플라인 셀 번호 (base) 로 입자 정렬 -- 같은 27 노드를 쓰는 입자가 이웃 스레드에
            n_ = self.ipar[:3].long()
            yy = (xa.double() - self.gpar[:3].double()) @ self.gpar[3:].double().view(3, 3).T
            bb = torch.floor(yy - 0.5).long()
            key = (bb[:, 0] * n_[1] + bb[:, 1]) * n_[2] + bb[:, 2]
            self.perm = torch.argsort(key)
            for t_, src in ((xa, x), (va, v), (Fa, Fv), (self.mA, m), (self.volA, vol)):
                torch.index_select(src, 0, self.perm, out=t_)
        self.sz[2:3].zero_()
        # 프레임 분할: 프레임 상태 P2G (운동량만) -> 격자 속도 (중력·감쇠·고정 없음, ex_vfin 과 같다) -> 바닥 후보
        if self.opt:
            self._g2p_p2g_f(xa, va, Fa, xa, va, Fa, 0, 1, -1, 0.0, self.CC[0], None)
        else:
            self._p2g_d(xa, va, Fa, -1, 0.0)
        BSC.launch(self._k("bs_node"), self.Pcap, (BSC.P(self.ipar), BSC.P(self.gm), BSC.P(self.gmv), BSC.P(self.vg), self._c(0), self._c(0), self._c(0),
                                                   self._c(0), self._c(1), BSC.P(None), BSC.P(self.vbc)), b=256)
        BSC.launch(self._k("bs_split"), Np, (np.int32(Np), BSC.P(xa), BSC.P(self.vg), BSC.P(self.gpar), BSC.P(self.ipar), self._c(h), self._c(gv[0]),
                                             self._c(gv[1]), self._c(gv[2]), np.int32(1 if FLOOR else 0), *[self._c(q) for q in PL], self._c(a.eps_sp),
                                             self._c(a.eps_st), BSC.P(self.nslot), BSC.P(self.nidx), BSC.P(self.sz)), b=256)
        if FLOOR:
            EX.launch("ex_near_init", Np, (EX.P(self.sz), np.int32(0), EX.P(self.nidx), EX.P(xa), EX.P(va), EX.P(Fa), np.float32(h), np.float32(gv[0]),
                                           np.float32(gv[1]), np.float32(gv[2]), EX.P(self.x0c), EX.P(self.F0c), EX.P(self.F0i), EX.P(self.cc)), b=256)
        if self.opt:
            if a.no_graph:
                self._substeps()
            else:
                if self.graph is None:
                    bufs = [self.vg, self.gm, self.gmv, self.sz] + [t for t in (self.cV, self.cVF) if t is not None] + self.X + self.V + self.FF \
                        + [t for t in self.CC if t is not None]
                    snap = [t.clone() for t in bufs]
                    self._substeps()                                            # 데우기 (NVRTC 컴파일·지연 할당) -- 결과 버림
                    for t, s_ in zip(bufs, snap):
                        t.copy_(s_)
                    self.capture()
                self.graph.replay()
            k = a.N % 2
            self.Cnow = k
        else:
            S = lambda k_: (self.X[k_ % 2], self.V[k_ % 2], self.FF[k_ % 2])
            for j in range(a.N):
                if j > 0:
                    xa_, va_, Fa_ = S(j - 1)
                    xb, vb, Fb = S(j)
                    BSC.launch("bs_g2p_d", Np, (np.int32(Np), BSC.P(xa_), BSC.P(Fa_), BSC.P(self.vg), BSC.P(self.gpar), BSC.P(self.ipar), np.float64(hs),
                                                BSC.P(xb), BSC.P(vb), BSC.P(Fb), BSC.P(self.sz), BSC.P(self.gK),
                                                BSC.P(self.nslot if FLOOR else None)), b=128)
                    self.box(xb, vb, now + j * hs, False)                       # 격자 상자 서브스텝마다 다시
                xc, vc, Fc = S(j)
                self._p2g_d(xc, vc, Fc, self.QUAD, hs)
                self._near(xc, Fc)
                self._grid_step(hs, DAMP, self._fix())
            xa_, va_, Fa_ = S(a.N - 1)
            xb, vb, Fb = S(a.N)
            BSC.launch("bs_g2p_d", Np, (np.int32(Np), BSC.P(xa_), BSC.P(Fa_), BSC.P(self.vg), BSC.P(self.gpar), BSC.P(self.ipar), np.float64(hs),
                                        BSC.P(xb), BSC.P(vb), BSC.P(Fb), BSC.P(self.sz), BSC.P(self.gK),
                                                BSC.P(self.nslot if FLOOR else None)), b=128)
            k = a.N % 2
        x2, v2, F2 = self.X[k], self.V[k], self.FF[k]
        if FLOOR:
            EX.launch("ex_near_fin", Np, (EX.P(self.sz), np.int32(0), EX.P(self.nidx), EX.P(self.x0c), EX.P(self.F0c), EX.P(None), EX.P(x2), EX.P(F2),
                                          EX.P(self.F0i), EX.P(None), EX.P(None), EX.P(None), EX.P(None), *near_args(0), np.float64(h),
                                          EX.P(x2), EX.P(v2), EX.P(F2)), b=128)
            if self.APIC and a.near_c0:
                nn_ = int(self.sz[2])
                self.CC[k][self.nidx[:nn_]] = 0.0
        if self.SORT:
            self.OX.index_copy_(0, self.perm, x2)
            self.OV.index_copy_(0, self.perm, v2)
            self.OF.index_copy_(0, self.perm, F2)
            x2, v2, F2 = self.OX, self.OV, self.OF
        return x2, v2, F2.view(Np, 3, 3)


class _Recapture(Exception):
    pass


class _Diverge(Exception):
    pass


OPT = BS(a.opt) if BSP else (Opt() if a.opt else None)
if a.near_diag:
    EX.set_diag([float(q) for q in a.near_diag.split(",")])
    BSC.set_diag([float(q) for q in a.near_diag.split(",")])
    print(f"[진단 배율] 바닥 후보 힘 (관성, 변위 탄성, 기울기 탄성) = {a.near_diag}", flush=True)
if OPT is not None:
    RTC.RAWPTR[0] = True
PRED = torch.empty(L, Np, 3, device=dev) if (not a.no_dump or a.dump_pred) else None
VD = torch.empty(L, Np, 3, device=dev, dtype=torch.float16) if not a.no_dump else None
FD = torch.empty(L, Np, 3, 3, device=dev, dtype=(torch.float32 if a.dump_f32 else torch.float16)) if not a.no_dump else None
errs, recs = [], []
flog = open(os.path.join(a.out, "fem.jsonl"), "w")
div_at = None
for i in range(L):
    _TP = None
    if i == a.tprof:
        from torch.profiler import profile, ProfilerActivity
        _TP = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        _TP.__enter__()
    if float(v.norm(dim=-1).max()) * h > 0.25 * EXT:                              # 측정 밖 기준과 같은 판정 (기준 판 격자 폭증 방지)
        div_at = i + 1
        print(f"[발산] 프레임 {div_at}: 이전 프레임 속도 max|v|·h > 0.25 EXT -- 멈춤", flush=True)
        break
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        if OPT is None:
            x2, v2, F2, st = frame_base(i, x, v, F)
        else:
            ncap0 = OPT.n_cap
            x2, v2, F2 = OPT.frame(i, x, v, F)
            st = dict(M=OPT.M, T=OPT.T, recap=OPT.n_cap - ncap0)
    except _Diverge as ex_:
        div_at = i + 1
        print(f"[발산] 프레임 {div_at}: {ex_} -- 멈춤", flush=True)
        break
    torch.cuda.synchronize()
    t_frame = time.perf_counter() - t0
    if _TP is not None:
        _TP.__exit__(None, None, None)
        print(_TP.key_averages().table(sort_by="self_cuda_time_total", row_limit=25), flush=True)
    # ---- 측정 밖 ----
    if OPT is not None:
        szh = OPT.sz.tolist()
        st.update(n_near=int(szh[2]), overflow=int(szh[3]))
        if VFL:
            st.update(n_contact=int(OPT.ncon.item()))                               # 프레임 동안 접촉 (입자 x 서브스텝) 수
        if RNEW:                                                                    # 절 R 통계 (측정 밖)
            rs = OPT.rst.tolist()
            T_ = OPT.T
            occ_ = OPT.VTs[:T_] > 0
            A_ = torch.eye(3, device=dev) + torch.einsum("tia,tib->tab", OPT.Uc[OPT.tets[:T_].reshape(-1)].view(T_, 4, 3), OPT.gl[:T_].view(T_, 4, 3))
            dA_ = _det3(A_)[occ_]
            dX_ = _det3(A_[occ_] @ OPT.FT[:T_].view(T_, 3, 3)[occ_])
            dFp = _det3(F2.view(Np, 3, 3))
            ke_p = float(0.5 * (m * (v2 ** 2).sum(-1)).sum())
            st.update(inv_sub=int(rs[0]), inv_end=int((~(dA_ > 0)).sum()), invX_end=int((~(dX_ > 0)).sum()), detA_min=float(dA_.min()) if dA_.numel() else None,
                      detX_min=float(dX_.min()) if dX_.numel() else None, Fp_inv=int((~(dFp > 0)).sum()), Fp_detmin=float(dFp.min()),
                      lim_n=int(rs[1]), lim_dp=[rs[2], rs[3], rs[4]], lim_dke=rs[5], lim_dpabs=rs[6], lim_tets=int(rs[7]), ke_p=ke_p,
                      p_tot=[float(q) for q in (m[:, None] * v2).sum(0).tolist()], nt_ovf=int(OPT.novf.item()))
        if NINT and i < a.nint_check:                                               # 절 V 검산 (측정 밖, 프레임 끝 U)
            OPT.nint_check(i)
        if BSL:                                                                     # 절 S: 질량·운동량 보존 (P2G 전후, 서브스텝 끝, G2P 뒤)
            M_ = OPT.M
            mnb, mvb, vnb = OPT.mn[:M_].double(), OPT.mv[:M_].double(), OPT.vn[:M_].double()
            md_ = m.double()
            st.update(mass_p=float(md_.sum()), mass_n=float(mnb.sum()),
                      P_p0=(md_[:, None] * v.double()).sum(0).tolist(), P_n0=mvb.sum(0).tolist(), P_n1=(mnb[:, None] * vnb).sum(0).tolist(),
                      P_p1=(md_[:, None] * v2.double()).sum(0).tolist(), V_drop=float(OPT.bst[0]), T_drop=int(OPT.bst[1]),
                      V_T=float(OPT.VTs[:OPT.T].double().sum()), V_pt=float(vol.double().sum()), m_redist=float(OPT.bst[2]), V_redist=float(OPT.bst[3]))
            if a.bsdiag >= 0 and (i == 0 or (a.bsdiag > 0 and i % a.bsdiag == 0)):
                OPT.bsdiag(i)
        if szh[3]:
            print(f"  [경고] 프레임 {i + 1}: 팽창 영역 밖 입자 {szh[3]} (격자 넘침)", flush=True)
            OPT.sz[3:4].zero_()
    e = float((x2 - GT[i]).norm(dim=-1).mean()) / EXT
    spread = float((x2.max(0).values - x2.min(0).values).norm()) / EXT
    fin = bool(torch.isfinite(x2).all()) and math.isfinite(e)
    smin = nbelow = None
    if HASPL:
        s = ((x2 - torch.tensor(pt_, device=dev)) * torch.tensor(nr_, device=dev)).sum(-1)
        smin, nbelow = float(s.min()), int((s < 0).sum())
    errs.append(e if fin else float("nan"))
    rec = dict(t=i, sec=t_frame, err=e, spread=spread, s_min=smin, n_below=nbelow, **st)
    flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
    flog.flush()
    recs.append(rec)
    print(f"  프레임 {i + 1}/{L} {1e3 * t_frame:.2f} ms 오차 {100 * e:.3f}% (누적 평균 {100 * np.nanmean(errs):.3f}%) 퍼짐 {spread:.3f} "
          f"노드 {st.get('M')} 사면체 {st.get('T')} 바닥 후보 {st.get('n_near')}" + (f" 최저 입자 {smin:+.5f} 아래 {nbelow}" if HASPL else "") + (f" 접촉 {st.get('n_contact')}" if VFL else "")
          + (f" [R] 뒤집힘 (사면체x서브스텝) {st['inv_sub']} 끝 {st['inv_end']} det A 최소 {(st['detA_min'] if st['detA_min'] is not None else float('nan')):.3g} 입자 F 뒤집힘 {st['Fp_inv']} 제한 {st['lim_n']} 줄인 KE {st['lim_dke']:.3g} KE {st['ke_p']:.4g}" if RNEW else ""), flush=True)
    if PRED is not None:
        PRED[i] = x2
    if VD is not None:
        VD[i] = v2.half()
        FD[i] = F2.to(FD.dtype)
    x, v, F = x2.clone(), v2.clone(), F2.clone()
    if not fin or spread > a.div:
        div_at = i + 1
        print(f"[발산] 프레임 {div_at}: {'값 비유한' if not fin else '퍼짐 %.2f > %.1f' % (spread, a.div)} -- 멈춤", flush=True)
        break

flog.close()
if a.lagdump and getattr(OPT, "_ring", None):
    torch.save(list(OPT._ring), os.path.join(a.out, "lagdump.pt"))
    print(f"[lag 덤프] {os.path.join(a.out, 'lagdump.pt')} 프레임 {[r['t'] + 1 for r in OPT._ring]}", flush=True)
W = a.warm
sec = [r["sec"] for r in recs[W:]]
med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
done = len(recs)
summ = dict(out=a.out, args=vars(a), frames=L, frames_done=done, diverged=div_at, err_mean=(sum(errs) / len(errs) if div_at is None else None),
            err_mean_done=float(np.nanmean(errs)), ms_mean=1e3 * sum(sec) / max(len(sec), 1), ms_med=1e3 * med(sec), fps=len(sec) / max(sum(sec), 1e-9),
            N=a.N, hs=hs, mass=a.mass, dt_cfl=dt_cfl, N_cfl=h / dt_cfl, cwave=cwave, lmin=lmin, mode=a.mode, opt=a.opt,
            s_min=(min(r["s_min"] for r in recs) if HASPL else None), n_below_max=(max(r["n_below"] for r in recs) if HASPL else None),
            floor=a.floor, vfl_mode=(a.vfl_mode if VFL else None), vfl_np=(a.vfl_np if VFL else None), affine=bool(a.affine),
            n_contact_max=(max(r.get("n_contact") or 0 for r in recs) if VFL else None),
            spread_last=recs[-1]["spread"], n_near_med=med([r.get("n_near") or 0 for r in recs]),
            overflow_frames=sum(1 for r in recs if r.get("overflow")), gpu=torch.cuda.get_device_name(0), tag=a.tag, damp_sub=DAMP,
            fb=a.fb, fb_jhat=a.fb_jhat, fb_on=a.fb_on, lim=a.lim, lim_v=a.lim_v, lim_mode=a.lim_mode,
            inv_sub_tot=(sum(r.get("inv_sub", 0) for r in recs) if RNEW else None), inv_frames=(sum(1 for r in recs if r.get("inv_sub", 0) > 0) if RNEW else None),
            inv_end_max=(max(r.get("inv_end", 0) for r in recs) if RNEW else None), Fp_inv_max=(max(r.get("Fp_inv", 0) for r in recs) if RNEW else None),
            detA_min=(min((r["detA_min"] for r in recs if r.get("detA_min") is not None and math.isfinite(r["detA_min"])), default=None) if RNEW else None),
            lim_n_tot=(sum(r.get("lim_n", 0) for r in recs) if RNEW else None), lim_dke_tot=(sum(r.get("lim_dke", 0.0) for r in recs) if RNEW else None),
            lim_dpabs_tot=(sum(r.get("lim_dpabs", 0.0) for r in recs) if RNEW else None), nt_ovf=(max(r.get("nt_ovf", 0) for r in recs) if RNEW else None),
            p2g=a.p2g, g2p=a.g2p, bs_mesh=(a.bs_mesh if BSL else None), interp=a.interp, rep=REP, rep_mode=(a.rep_mode if REP else None), quad=(a.quad if BSP else None), K=(a.K if BSP and a.quad == "cell" else None), apic=bool(a.apic), rep_w=a.rep_w,
            nint=NINT, nint_beta=(a.nint_beta if NINT else None))
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
print(f"[롤아웃] {a.tag} 서브스텝 {a.N} {a.mode}{' 최적화' if a.opt else ''}: {done}/{L} 프레임{' (발산 %d)' % div_at if div_at else ''}, 평균 {100 * summ['err_mean_done']:.3f}%  "
      f"프레임당 {summ['ms_mean']:.2f} ms (중앙 {summ['ms_med']:.2f}, {summ['fps']:.2f} fps, 앞 {W} 프레임 제외) 퍼짐 {summ['spread_last']:.3f} "
      f"최저 입자 {summ['s_min']} 아래 최대 {summ['n_below_max']}", flush=True)
if PRED is not None and div_at is None:
    if a.dump_pred:
        torch.save({"pred": PRED.cpu(), "EXT": EXT}, os.path.join(a.out, "pred.pt"))
    if not a.no_dump:
        torch.save({"pred": PRED.cpu(), "gt": GT.cpu(), "x0": d["x"][0].float(), "EXT": EXT, "t0": 0, "tag": a.tag,
                    "F": FD.float().cpu(), "v": VD.float().cpu(), "ctrl_pos": d.get("ctrl_pos"), "ctrl_R": d.get("ctrl_R"), "ctrl_id": d.get("ctrl_id"),
                    "nodes": None, "fscal": None, "gt_fscal": None}, os.path.join(a.out, "roll_fem.pt"))
        print(f"[롤아웃 덤프] {os.path.join(a.out, 'roll_fem.pt')}", flush=True)
