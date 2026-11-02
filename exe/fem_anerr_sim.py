"""절 AN.err (오차 가르기): exe/fem_an_sim.py 복사본 + 새 플래그 -- --integ bdf2 (입자 이력으로 프레임 경계를 넘는 BDF2), --quad part (입자 적분점; 칸마다 8 의 배수로 부피 0 더미 채움),
--fl_src part (바닥 고착을 입자 위치로 판정, 명시적 판 bs_contact 와 같은 질량 몫 Σ w m_p w(s) / m_n), --fl_npd δ (명시적 판 비관통: 점 하나라도 s < −δ 인 노드의 안쪽 법선 속도 0),
--en (프레임마다 입자 탄성 Σ V ψ(F)·운동 에너지·몸 높이 (질량 중심 법선 높이, 법선 높이 0.1-99.9 % 범위) 우리 / 정답을 fem.jsonl 에). 나머지 정의는 원본 그대로.
원본 설명: 절 AN: 암시적 B-스플라인 하이브리드 판 (검증판: 틀은 torch, 적분점 커널은 lib/anchorflow/fem_an_cuda.py, 전부 float64).
프레임: P2G 한 번 (입자 -> 노드 질량·운동량, 입자 -> 칸 가우스 점 F_q = 절 AI A1 회전·신장 분리 평균, 가우스 점 부피·질량 = 칸 입자 합 / 8)
  -> 풀이 K 걸음 (입자 배열을 읽거나 쓰지 않음; 노드·가우스 점 배열만) -> G2P 한 번 (x += Σ w U, v = Σ w v_n, F_p ← (I + Σ U ⊗ ∇w) F_p).
걸음 (길이 Δt = h / K): 후퇴 오일러 (be, τ = Δt, y = U_{k+1}) 또는 암시적 중점 (mid, τ = Δt/2, y = (U_k + U_{k+1})/2, U_{k+1} = 2y − U_k, v_{k+1} = 2(y − U_k)/τ − v_k).
  둘 다 같은 최소화: E(y) = Σ_n m_n/(2τ²) |y − Ũ|² + Σ_q V_q ψ(X_q(y)) + 바닥 고착 항, Ũ = U_k + τ v_k + τ² g.
바닥 (노드 바닥 아님, 입자 판정 아님): 가우스 점 q 의 걸음 끝 위치 부호 거리 s_q, 정답 sticky 가중 w_st(s_q) (정답 격자 dx 의 평면 아래 노드 몫, s_q < 0 일 때만 = 절 AH 띠 0),
  노드 몫 r_n = Σ_q w_qn m_q w_st / Σ_q w_qn m_q, 걸음 고착 인자 k_n = (1 − r_n)^(Δt / dt_정답) (절 AH 정의 그대로), 고착은 v_{k+1} 의 접선·안쪽 법선 성분에만 (vfl_out, 절 AK.31).
  고착 항은 v_{k+1} 에 대한 2 차 감쇠 (계수 = 고착 인자를 정확히 내는 값) 이고, 변수 바꿈 y = ŷ + T η (T = √k P + (I − P)) 로 무한대 없이 푼다 (k = 0 이면 그 성분 고정).
뉴턴: 투영 헤시안 (FCR 해석 고유계, 음의 고유값 0), 행렬 없는 PCG (블록 야코비 전처리), 되돌이 선탐색 (아르미호), 고정 반복 예산, 첫 추정 = Ũ.
고착 몫 (r_n, P) 은 뉴턴 반복마다 그 반복의 y 로 다시 잰다 (반복 안에서는 고정)."""
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
from anchorflow import fem_an_cuda as AK                                            # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--cell", type=float, default=1.0, help="셀 크기 = 정답 MPM dx 의 몇 배 (씬 공통 규칙)")
ap.add_argument("--floor", choices=["vel", "none"], default="vel")
ap.add_argument("--cfg_bc", action="store_true")
ap.add_argument("--region_density", action="store_true")
ap.add_argument("--pg_vdamp", action="store_true")
ap.add_argument("--K", type=int, default=2, help="프레임당 암시적 걸음 수")
ap.add_argument("--integ", choices=["mid", "be", "bdf2", "theta", "trbdf2", "ga", "hht"], default="mid")
ap.add_argument("--y0", choices=["ut", "uk"], default="ut", help="뉴턴 시작점: 관성 예측 Ũ (원본) 또는 걸음 시작 위치 (같은 최소화, 시작점만 -- 큰 a_n 으로 예측이 요소를 뒤집을 때). 고착 판정은 그대로 관성 예측으로")
ap.add_argument("--rho", type=float, default=0.7, help="--integ ga (Chung-Hulbert generalized-α) / hht (HHT-α) 의 고주파 극한 스펙트럼 반경 ρ∞ (1 = 감쇠 없음, ga 0 에서 1, hht 0.5 에서 1)")
ap.add_argument("--theta", type=float, default=0.55, help="--integ theta 의 θ: 힘을 y = U_k + θ (U_{k+1} − U_k) 점에서 (0.5 = 중점, 1 = 후퇴 오일러); v_{k+1} = v_k + Δt a(y), U_{k+1} = U_k + Δt ((1−θ) v_k + θ v_{k+1})")
ap.add_argument("--mlow", type=float, default=0.0, help=">0 이면 절 AH.18 관성 질량 하한 m' = max(m, κ x 첫 프레임 노드 질량 중앙값) (관성에만; 중력은 실제 질량)")
ap.add_argument("--mfl_n", type=int, default=0, help=">0 이면 절 AG.15 국소 CFL 관성 질량을 명시적 N 서브스텝 기준으로: m' = max(m, (h/(2N))² k_n) (K 와 무관한 고정 하한; 관성에만)")
ap.add_argument("--fl_avg", type=int, default=0, help=">0 이면 (정의 변경, 제안 3) 고착 인자를 걸음 안에서 평균: k = exp(Δt/dt_정답 · mean_s log(1 − r_s)), r_s = 걸음 시작 위치의 몫 r0 와 걸음 끝 예측 위치의 몫 r1 의 선형 보간, s 중점 n 개")
ap.add_argument("--kad", type=int, default=0, help=">0 이면 (제안 2) 적응 걸음: 프레임 시작 입자 중 바닥 법선 거리 < --kad_d 이고 안쪽 법선 속력 또는 바깥 법선 속력이 --kad_v 를 넘는 입자가 있으면 그 프레임 K = kad, 아니면 --K (실행 중 값만)")
ap.add_argument("--kad_d", type=float, default=0.04, help="적응 걸음 바닥 근접 거리 (m) + 이번 프레임 이동 거리 |v·n| h")
ap.add_argument("--kad_v", type=float, default=0.1, help="적응 걸음 법선 속력 문턱 (m/s): 근접 입자의 |v·n| 최대가 이보다 크면 충격·반동 중으로 봄")
ap.add_argument("--quad", choices=["gauss", "part"], default="gauss", help="탄성 적분점: 칸 2x2x2 가우스 (원본) 또는 입자 (명시적 bnv 와 같은 적분점)")
ap.add_argument("--fl_src", choices=["q", "part"], default="q", help="바닥 고착 판정 점: 적분점 (원본) 또는 입자 (명시적 판 정의)")
ap.add_argument("--fl_npd", type=float, default=0.0, help=">0 이면 명시적 판 비관통: 판정 점 하나라도 s < −δ 인 노드는 예측 속도가 안쪽이면 법선 속도 0 (걸음당 한 번 정함)")
ap.add_argument("--ls_diff", action="store_true", help="가속 판 선탐색을 점별 에너지 차 Σ_q (ψ_q(후보) − ψ_q(지금)) + 관성 차 ½ Σ Mi c δ (2η + c δ) 로 판정 (총 에너지끼리 빼면 τ 가 작을 때 f64 반올림에 묻혀 걸음 0 만 고름)")
ap.add_argument("--mfl", type=float, default=0.0, help=">0 이면 관성 질량 하한 m' = max(m, β (Δt/2)² k_n), k_n = Σ_q V_q (λ+2μ) |∇w_qn| Σ_n' |∇w_qn'| (명시적 판 국소 CFL 관성 질량과 같은 식, β 1 = 같은 값) -- 거의 0 질량 가장자리 노드의 특이에 가까운 뉴턴 계 처치 (정의 변경)")
ap.add_argument("--en", action="store_true", help="프레임마다 에너지·높이 진단 (우리 / 정답)")
ap.add_argument("--newton", type=int, default=6)
ap.add_argument("--cg", type=int, default=100)
ap.add_argument("--cg_tol", type=float, default=1e-4, help="PCG 상대 잔차 (전처리 노름) 조기 종료")
ap.add_argument("--nt_tol", type=float, default=1e-6, help="뉴턴 조기 종료: |g_η| / |g_η 첫 반복| (고정 예산 안에서)")
ap.add_argument("--no_vfl_out", action="store_true")
ap.add_argument("--pc", choices=["jacobi", "mg"], default="jacobi", help="PCG 전처리: 노드 블록 야코비 또는 기하 멀티그리드 (세계 고정 격자, 간격 2 배씩)")
ap.add_argument("--mg_levels", type=int, default=5)
ap.add_argument("--mg_min", type=int, default=300, help="가장 거친 층 노드 수가 이 아래면 멈춤")
ap.add_argument("--mg_omega", type=float, default=1.0, help="감쇠 블록 야코비 평활 계수 = mg_omega / λ_max(D⁻¹A) (층마다 거듭제곱법 mg_pow 번으로 잼)")
ap.add_argument("--mg_pow", type=int, default=10)
ap.add_argument("--mg_nc", type=int, default=10, help="가장 거친 층 평활 횟수")
ap.add_argument("--dump", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--fast", action="store_true", help="가속 판: 헤시안 float32 저장·곱 (기울기·잔차 float64), MG 계층 프레임당 한 번 (뉴턴마다 0 층 대각만 갱신), "
                "고정 예산 (조기 종료 없음), 호스트 동기화 없음 (스칼라는 장치에), 장치 쪽 선탐색 후보 1·½·¼ (--fast_ls 0 이면 온 걸음)")
ap.add_argument("--fast_ls", type=int, default=1)
ap.add_argument("--mg_dense", type=int, default=1, help="가속 판 MG 거친 층을 압축 없는 조밀 상자로 (호스트 동기화 없음), λ 는 앞 프레임 값")
ap.add_argument("--fl_iter", action="store_true", help="(옛 판) 고착 상태를 뉴턴 반복마다 다시 잼 -- 풀이 문제가 반복마다 바뀌어 뉴턴이 수렴하지 않음 (절 AN 진단). 기본은 걸음당 한 번 (관성 예측 y = Ũ 의 걸음 끝 위치) 정하고 고정")
ap.add_argument("--no_proj", action="store_true", help="진단: 가우스 점 헤시안 고유값 투영 끔 (원래 헤시안)")
ap.add_argument("--diag", action="store_true", help="진단: 뉴턴 반복마다 |g|/|g0|, 보폭, PCG 반복·최종 상대 잔차, 란초스 고유값 추정 (전처리 계 λmin·λmax) 을 fem.jsonl 에")
ap.add_argument("--gfin", action="store_true", help="걸음 끝 y 에서 기울기를 한 번 더 재 |g|/|g0| 기록 (구간 '검사', 풀이 비용 아님)")
ap.add_argument("--sec", action="store_true", help="구간별 시간 (구간 경계마다 동기화 -- 합이 동기화 없는 프레임 시간보다 조금 크다): p2g·바닥·요소 (기울기+헤시안)·대각·멀티그리드 구성·PCG (헤시안 곱·사이클)·선탐색·G2P")
ap.add_argument("--prof", action="store_true", help="커널별 GPU 시간 (CUDA 이벤트) 을 fem.jsonl 에 -- 단독 카드에서만 의미")
ap.add_argument("--check", action="store_true", help="프레임 0 에서 기울기·헤시안 곱을 유한 차분과 대조하고 끝")
a = ap.parse_args()
dev = "cuda"
RTC.RAWPTR[0] = True
D64 = torch.float64
os.makedirs(a.out, exist_ok=True)
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
DT = h / a.K
TH0 = {"be": 1.0, "mid": 0.5, "theta": a.theta}.get(a.integ)
GAM = 2.0 - math.sqrt(2.0)                                                             # TR-BDF2 첫 단계 몫
SG = {}
GS = [None]                                                                            # 노드별 중력 몫 m / m' (관성 질량 하한일 때)


def ga_par():
    """generalized-α 계수 (x_{n+1−αf} 형): Chung-Hulbert αm = (2ρ−1)/(ρ+1), αf = ρ/(ρ+1); HHT αm = 0, αf = (1−ρ)/(1+ρ); γ = ½ − αm + αf, β = ¼ (1 − αm + αf)²."""
    r = a.rho
    if a.integ == "hht":
        am, af = 0.0, (1.0 - r) / (1.0 + r)
    else:
        am, af = (2 * r - 1) / (r + 1), r / (r + 1)
    return am, af, 0.5 - am + af, 0.25 * (1 - am + af) ** 2


GA_A = [None]                                                                          # 이번 걸음 시작 노드 가속도 a_n


def set_stage(kind, dt, th=None, c1=None, c0=None, tau=None):
    """한 풀이 단계의 계수 (실시간 판에서는 커널 인자 몇 개): θ 단계 (τ = θ dt) 또는 2 걸음 BDF 단계 (Ub = c1 U_a − c0 U_n, τ)."""
    global TAU, KEX, DAMP
    SG.clear()
    SG.update(kind=kind, dt=dt, th=th, c1=c1, c0=c0)
    if kind == "ga":
        am, af, gm, be = ga_par()
        SG.update(am=am, af=af, gm=gm, be=be)
        tau = math.sqrt((1 - af) * be * dt * dt / (1 - am))
    TAU = th * dt if kind == "th" else tau
    KEX = dt / float(cfg["substep_dt"])
    DAMP = (VD_S ** (dt / float(cfg["substep_dt"]))) if (a.pg_vdamp and VD_S < 1.0) else 1.0


TAU = DT if a.integ == "be" else (0.5 * DT if a.integ == "mid" else 2.0 * DT / 3.0)
DM.REGION_DENSITY[0] = bool(a.region_density)
m64 = DM.traj_mass(d, dev).double()
EXT = float(d["_ext"])
rho = (d["_rho"].to(dev).double()) if a.region_density else None
vol64 = (m64 / rho) if a.region_density else (m64 / float(cfg["density"]))
mu, lam = PR.lame(cfg["E"], cfg["nu"])
mu, lam = float(mu), float(lam)
gv = torch.tensor([float(q) for q in cfg["g"]], device=dev, dtype=D64)
Np = int(d["x"].shape[1])
L = min(a.frames, int(d["x"].shape[0]) - 1)
x = d["x"][0].to(dev).double()
v = d["v"][0].to(dev).double()
F = DM.traj_F(d, h, dev)[0].to(dev).double().reshape(Np, 9)
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))                  # 정답 MPM dx
NODE_H = a.cell * DXG * math.sqrt(2.0)                                                # 회전 정육면체 격자 칸 = node_h / √2
VD_S = float(cfg.get("grid_v_damping_scale", 1.0))
DAMP = (VD_S ** (DT / float(cfg["substep_dt"]))) if (a.pg_vdamp and VD_S < 1.0) else 1.0
lat = SX.lattice(NODE_H, SX.HZ_CUBE * NODE_H, device=dev, dtype=D64)
A64, Ai64 = lat.A, lat.Ai
dxl = float(A64[:, 0].norm())
FLOOR = a.floor == "vel"
VOUT = not a.no_vfl_out
if FLOOR:
    pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
    PT = torch.tensor([float(q) for q in pl["point"]], device=dev, dtype=D64)
    NR = torch.tensor(pl["normal"], device=dev, dtype=D64)
    NR = NR / NR.norm()
    PN = float(PT @ NR)
    KEX = DT / float(cfg["substep_dt"])
cwave = math.sqrt((lam + 2 * mu) / float((rho.min() if rho is not None else cfg["density"])))
print(f"[절 AN 암시적 B-스플라인] 입자 {Np}, 프레임 {L}, 걸음 {a.K}/프레임 ({a.integ}, Δt {DT:.4e}, CFL 수 {DT * cwave / dxl:.1f}), 셀 {dxl:.5f} m (정답 dx {DXG:.4f} 의 {a.cell} 배), "
      f"뉴턴 {a.newton} x PCG {a.cg}, 바닥 {a.floor}{' vfl_out' if (FLOOR and VOUT) else ''}", flush=True)
OFF = torch.stack(torch.meshgrid(torch.arange(3), torch.arange(3), torch.arange(3), indexing="ij"), -1).reshape(27, 3).to(dev)
GG = 0.5 / math.sqrt(3.0)
I3 = torch.eye(3, device=dev, dtype=D64)


def bsw(fx, Ai_=None):
    Ai_ = Ai64 if Ai_ is None else Ai_
    w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2], 1)
    dw = torch.stack([fx - 1.5, -2 * (fx - 1), fx - 0.5], 1)
    wi, wj, wk = w[:, OFF[:, 0], 0], w[:, OFF[:, 1], 1], w[:, OFF[:, 2], 2]
    W = wi * wj * wk
    GW = torch.stack([dw[:, OFF[:, 0], 0] * wj * wk, wi * dw[:, OFF[:, 1], 1] * wk, wi * wj * dw[:, OFF[:, 2], 2]], -1) @ Ai_     # ∇_x = Aiᵀ ∇_y
    return W, GW


def rs_prep(Fp):
    n = Fp.shape[0]
    Fc = Fp.contiguous()
    R = torch.empty(n, 9, device=dev, dtype=D64)
    LS = torch.empty(n, 9, device=dev, dtype=D64)
    neg = torch.empty(n, device=dev, dtype=torch.uint8)
    AK.launch("an_rsp", n, (np.int32(n), AK.P(Fc), AK.P(R), AK.P(LS), AK.P(neg)))
    return Fc, R, LS, neg


def rs_mean(Fp, w, inv, nq, count=True, prep=None):
    """절 AI A1 회전·신장 분리 평균 (float64, CUDA 커널): 뒤집힌 입자가 든 점·비유한 점은 유클리드 평균 (절 AK.33 과 같은 정의). prep = rs_prep(Fp) 을 넘기면 입자 준비를 다시 하지 않는다."""
    Fc, R, LS, neg = prep if prep is not None else rs_prep(Fp)
    W = w[:, None]
    sR = torch.zeros(nq, 9, device=dev, dtype=D64).index_add_(0, inv, W * R)
    sL = torch.zeros(nq, 9, device=dev, dtype=D64).index_add_(0, inv, W * LS)
    sF = torch.zeros(nq, 9, device=dev, dtype=D64).index_add_(0, inv, W * Fc)
    sw = torch.zeros(nq, device=dev, dtype=D64).index_add_(0, inv, w)
    bad = torch.zeros(nq, device=dev, dtype=torch.bool).index_put_((inv[neg.bool()],), torch.tensor(True, device=dev))
    RS = torch.empty(nq, 9, device=dev, dtype=D64)
    nf = torch.empty(nq, device=dev, dtype=torch.uint8)
    AK.launch("an_rsf", nq, (np.int32(nq), AK.P(sR), AK.P(sL), AK.P(sw), AK.P(RS), AK.P(nf)))
    bad = bad | nf.bool()
    rs_mean.last_bad = bad
    return torch.where(bad[:, None], sF / sw.clamp_min(1e-300)[:, None], RS), (int(bad.sum()) if count else None)


def wstk(s):
    """정답 sticky 가중 (절 AH 띠 0, fem_bsel_cuda.wstk 와 같은 식): 부호 거리 s 의 점이 정답 격자 (dx DXG) 에서 평면 아래 노드에 주는 2 차 B-스플라인 몫, s < 0 일 때만."""
    u = (s + PN) / DXG
    bf = torch.floor(u - 0.5)
    fx = u - bf
    w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2], -1)
    kk = bf[:, None] + torch.arange(3, device=dev, dtype=D64)[None]
    o = (w * ((kk * DXG - PN) < 0).double()).sum(-1)
    return torch.where(s < 0, o, torch.zeros_like(o))


class Elem:
    """가우스 점 집합 위 탄성 (커널 호출)."""

    def __init__(s, qn, G, Fq, Vq, na):
        s.qn, s.G, s.Fq, s.Vq, s.na, s.nq = qn, G, Fq, Vq, na, qn.shape[0]
        s.E = torch.empty(s.nq, device=dev, dtype=D64)
        s.H = torch.empty(s.nq, 81, device=dev, dtype=D64)
        s.nmv = 0

    def eval(s, y, mode, dev_e=False):
        f = torch.zeros(s.na, 3, device=dev, dtype=D64)
        if (mode & 8) and not hasattr(s, "H32"):
            s.H32 = torch.empty(s.nq, 81, device=dev, dtype=torch.float32)
        AK.launch("an_elem", s.nq, (np.int32(s.nq), AK.P(s.qn), AK.P(s.G), AK.P(s.Fq), AK.P(s.Vq), AK.P(y.contiguous()), np.float64(mu), np.float64(lam),
                                    np.int32(mode), AK.P(s.E), AK.P(f), AK.P(s.H), AK.P(getattr(s, "H32", None))))
        if not (mode & 4):
            return None, f
        return (s.E.sum() if dev_e else float(s.E.sum())), f

    def f32(s):
        if not hasattr(s, "G32"):
            s.G32 = s.G.float().contiguous()
            s.V32 = s.Vq.float().contiguous()
        return s

    def mv32(s, dy):
        with tsec("헤시안곱" if s.na == STAT.get("na") else "헤시안곱_거친층"):
            o = torch.zeros(s.na, 3, device=dev, dtype=torch.float32)
            AK.launch("an_mv32", s.nq, (np.int32(s.nq), AK.P(s.qn), AK.P(s.G32), AK.P(s.H32), AK.P(s.V32), AK.P(dy.contiguous()), AK.P(o)))
            s.nmv += 1
            return o

    def bd32(s):
        B = torch.zeros(s.na, 9, device=dev, dtype=torch.float32)
        AK.launch("an_bd32", s.nq, (np.int32(s.nq), AK.P(s.qn), AK.P(s.G32), AK.P(s.H32), AK.P(s.V32), AK.P(B)))
        return B.reshape(s.na, 3, 3)

    def mv(s, dy):
        with tsec("헤시안곱" if s.na == STAT.get("na") else "헤시안곱_거친층"):
            return s._mv(dy)

    def _mv(s, dy):
        o = torch.zeros(s.na, 3, device=dev, dtype=D64)
        AK.launch("an_mv", s.nq, (np.int32(s.nq), AK.P(s.qn), AK.P(s.G), AK.P(s.H), AK.P(s.Vq), AK.P(dy.contiguous()), AK.P(o)))
        s.nmv += 1
        return o

    def bd(s):
        B = torch.zeros(s.na, 9, device=dev, dtype=D64)
        AK.launch("an_bd", s.nq, (np.int32(s.nq), AK.P(s.qn), AK.P(s.G), AK.P(s.H), AK.P(s.Vq), AK.P(B)))
        return B.reshape(s.na, 3, 3)


SEC = {}


class tsec:
    """구간 시간 (ms) 누적: --sec 일 때만 경계에서 동기화."""

    def __init__(s, nm):
        s.nm = nm

    def __enter__(s):
        if a.sec:
            torch.cuda.synchronize()
            s.t = time.perf_counter()

    def __exit__(s, *e):
        if a.sec:
            torch.cuda.synchronize()
            t, c = SEC.get(s.nm, (0.0, 0))
            SEC[s.nm] = (t + 1e3 * (time.perf_counter() - s.t), c + 1)


_CK = {}


def ckern(ch):
    """2 차 B-스플라인 두 배 관계 계수 (1, 3, 3, 1)/4 의 텐서곱 (groups = ch)."""
    if ch not in _CK:
        c = torch.tensor([1.0, 3.0, 3.0, 1.0], device=dev, dtype=D64) / 4
        k3 = c[:, None, None] * c[None, :, None] * c[None, None, :]
        _CK[ch] = k3[None, None].repeat(ch, 1, 1, 1, 1).contiguous()
    return _CK[ch]


def restrict_dense(xd, nbf, Nc):
    """조밀 배열 (nx, ny, nz, ch) -> 거친 층 (Nc..., ch): r_c = Pᵀ r. 거친 노드 J 는 고운 층 노드 번호 2J − 3/2 자리, 고운 노드 2J − 3 + k (k = 0..3)."""
    ch = xd.shape[-1]
    t = xd.permute(3, 0, 1, 2)[None]
    npad = [2 * n - 4 for n in Nc]
    t = torch.nn.functional.pad(t, (0, npad[2] - nbf[2], 0, npad[1] - nbf[1], 0, npad[0] - nbf[0]))
    o = torch.nn.functional.conv3d(t, ckern(ch), stride=2, padding=3, groups=ch)
    return o[0].permute(1, 2, 3, 0)


def prolong_dense(xc, nbf):
    ch = xc.shape[-1]
    o = torch.nn.functional.conv_transpose3d(xc.permute(3, 0, 1, 2)[None], ckern(ch), stride=2, padding=3, groups=ch)
    return o[0, :, :nbf[0], :nbf[1], :nbf[2]].permute(1, 2, 3, 0)


class MG:
    """기하 멀티그리드 전처리 (대칭 V 순환). 층 ℓ+1 = 간격 2 배 세계 고정 격자의 2 차 B-스플라인 (두 배 관계로 고운 층 기저의 정확한 선형 결합 = 연장 P),
    거친 층 연산자 = 다시 이산화: 거친 칸 2x2x2 가우스 점, F = 그 팔분 영역 고운 점의 지금 X 의 회전·신장 분리 평균 (절 AI A1, 같은 rs_mean), 부피 = 칸 합 / 8, 투영 헤시안 (같은 커널);
    질량 = Pᵀ m (집중). 0 층 연산자 = Mi + T K T 그대로, 0 -> 1 전달은 T 를 끼운 Pᵀ T / T P. 평활 = 감쇠 블록 야코비 (ω), 가장 거친 층 = 평활 mg_nc 번."""

    def __init__(s, ctx, y, el, Mi, T, Bd):
        s.T = T
        s.L = [dict(el=el, Mi=Mi, D=torch.linalg.inv(Mi[..., None] * I3 + T @ Bd @ T), nb=ctx["nb"], act=ctx["act"])]
        X = el.Fq + torch.einsum("qna,qnb->qab", y[el.qn.long()], el.G).reshape(-1, 9)
        tq, V, nbf, mdn = ctx["tq"], el.Vq, ctx["nb"], ctx["mnd"].reshape(*ctx["nb"], 1)
        Ai_ = Ai64
        for lv in range(1, a.mg_levels):
            Nc = [n // 2 + 3 for n in nbf]
            tc = (tq + 1.5) / 2
            base = torch.floor(tc - 0.5).long()
            fx = tc - base
            cid = (base[:, 0] * Nc[1] + base[:, 1]) * Nc[2] + base[:, 2]
            uc, cinv = torch.unique(cid, return_inverse=True)
            nc = uc.numel()
            oi = ((fx >= 1.0).long() * torch.tensor([4, 2, 1], device=dev)).sum(1)
            qinv = cinv * 8 + oi
            nq = nc * 8
            Fc, _ = rs_mean(X, V, qinv, nq)
            cF, _ = rs_mean(X, V, cinv, nc)
            has = torch.zeros(nq, device=dev, dtype=torch.bool).index_put_((qinv,), torch.tensor(True, device=dev))
            Fc = torch.where(has[:, None], Fc, cF.repeat_interleave(8, 0)).contiguous()
            Vc = (torch.zeros(nc, device=dev, dtype=D64).index_add_(0, cinv, V).repeat_interleave(8) / 8.0).contiguous()
            o8 = torch.arange(8, device=dev)
            qfx = (1.0 + GG * (2 * torch.stack([o8 // 4, (o8 % 4) // 2, o8 % 2], 1).double() - 1)).repeat(nc, 1)
            Ai_ = Ai_ / 2
            _, QGW = bsw(qfx, Ai_)
            cb = torch.stack([uc // (Nc[1] * Nc[2]), (uc // Nc[2]) % Nc[1], uc % Nc[2]], 1).repeat_interleave(8, 0)
            qnid = ((cb[:, None, 0] + OFF[None, :, 0]) * Nc[1] + cb[:, None, 1] + OFF[None, :, 1]) * Nc[2] + cb[:, None, 2] + OFF[None, :, 2]
            mdc = restrict_dense(mdn, nbf, Nc)
            mflat = mdc.reshape(-1)
            act = torch.nonzero(mflat > 0).flatten()
            na = act.numel()
            cmp_ = torch.full((mflat.numel(),), -1, device=dev, dtype=torch.long)
            cmp_[act] = torch.arange(na, device=dev)
            qn = cmp_[qnid]
            assert bool((qn >= 0).all())
            G = torch.einsum("qab,qna->qnb", Fc.reshape(nq, 3, 3), QGW).contiguous()
            elc = Elem(qn.int().contiguous(), G, Fc, Vc, na)
            elc.eval(torch.zeros(na, 3, device=dev, dtype=D64), 2)
            Mic = (mflat[act] / (TAU * TAU))[:, None]
            s.L.append(dict(el=elc, Mi=Mic, D=torch.linalg.inv(Mic[..., None] * I3 + elc.bd()), nb=Nc, act=act))
            tq, X, V, nbf, mdn = (cb.double() + qfx), Fc, Vc, Nc, mdc
            if na < a.mg_min:
                break
        s.nl = len(s.L)
        for l in range(s.nl):                                                          # 층마다 λ_max(D⁻¹A) 거듭제곱법 -> ω_l = mg_omega / λ
            na_ = s.L[l]["Mi"].shape[0]
            z = torch.randn(na_, 3, device=dev, dtype=D64, generator=torch.Generator(device=dev).manual_seed(l))
            lm = 1.0
            for _ in range(a.mg_pow):
                w_ = s.sm(l, s.op(l, z))
                lm = float((z * w_).sum() / (z * z).sum())                             # 레일리 몫 (D 노름 아님, 추정)
                z = w_ / w_.norm()
            s.L[l]["w"] = a.mg_omega / max(lm, 1e-300)
        s.lam = [round(1.0 / L_["w"], 2) for L_ in s.L]

    def op(s, l, z):
        lv = s.L[l]
        if l == 0:
            return lv["Mi"] * z + (s.T @ lv["el"].mv((s.T @ z[..., None]).squeeze(-1))[..., None]).squeeze(-1)
        return lv["Mi"] * z + lv["el"].mv(z)

    def sm(s, l, r):
        return (s.L[l]["D"] @ r[..., None]).squeeze(-1)

    def tr(s, l, r):
        """층 l 의 압축 벡터 -> 층 l+1 (Pᵀ, 0 층은 Tᵀ 먼저)."""
        A_, B_ = s.L[l], s.L[l + 1]
        if l == 0:
            r = (s.T @ r[..., None]).squeeze(-1)
        dn = torch.zeros(int(np.prod(A_["nb"])), 3, device=dev, dtype=D64)
        dn[A_["act"]] = r
        o = restrict_dense(dn.reshape(*A_["nb"], 3), A_["nb"], B_["nb"]).reshape(-1, 3)
        return o[B_["act"]]

    def pr(s, l, zc):
        A_, B_ = s.L[l], s.L[l + 1]
        dn = torch.zeros(int(np.prod(B_["nb"])), 3, device=dev, dtype=D64)
        dn[B_["act"]] = zc
        z = prolong_dense(dn.reshape(*B_["nb"], 3), A_["nb"]).reshape(-1, 3)[A_["act"]]
        if l == 0:
            z = (s.T @ z[..., None]).squeeze(-1)
        return z

    def vc(s, l, r):
        w = s.L[l]["w"]
        z = w * s.sm(l, r)
        if l == s.nl - 1:
            for _ in range(a.mg_nc - 1):
                z = z + w * s.sm(l, r - s.op(l, z))
            return z
        z = z + s.pr(l, s.vc(l + 1, s.tr(l, r - s.op(l, z))))
        return z + w * s.sm(l, r - s.op(l, z))

    def __call__(s, r):
        with tsec("MG사이클"):
            return s.vc(0, r)


def ckern32(ch):
    k = ("f", ch)
    if k not in _CK:
        _CK[k] = ckern(ch).float().contiguous()
    return _CK[k]


def restrict32(xd, nbf, Nc):
    ch = xd.shape[-1]
    t = xd.permute(3, 0, 1, 2)[None]
    npad = [2 * n - 4 for n in Nc]
    t = torch.nn.functional.pad(t, (0, npad[2] - nbf[2], 0, npad[1] - nbf[1], 0, npad[0] - nbf[0]))
    return torch.nn.functional.conv3d(t, ckern32(ch), stride=2, padding=3, groups=ch)[0].permute(1, 2, 3, 0)


def prolong32(xc, nbf):
    ch = xc.shape[-1]
    o = torch.nn.functional.conv_transpose3d(xc.permute(3, 0, 1, 2)[None], ckern32(ch), stride=2, padding=3, groups=ch)
    return o[0, :, :nbf[0], :nbf[1], :nbf[2]].permute(1, 2, 3, 0)


class MG32(MG):
    """가속 판 멀티그리드: 계층 (거친 층 가우스 점·헤시안·대각·ω) 을 프레임 시작 (y = 0, X = F_q) 에서 한 번 만들고, 뉴턴마다 0 층의 T·대각만 바꾼다. 연산 float32."""

    def __init__(s, ctx, el, Mi32, T32, D0):
        s.T = T32
        s.L = [dict(el=el, Mi=Mi32, D=D0, nb=ctx["nb"], act=ctx["act"])]
        X, tq, V, nbf = el.Fq, ctx["tq"], el.Vq, ctx["nb"]
        mdn = ctx["mnd"].reshape(*ctx["nb"], 1)
        Ai_ = Ai64
        for lv in range(1, a.mg_levels):
            Nc = [n // 2 + 3 for n in nbf]
            tc = (tq + 1.5) / 2
            base = torch.floor(tc - 0.5).long()
            fx = tc - base
            cid = (base[:, 0] * Nc[1] + base[:, 1]) * Nc[2] + base[:, 2]
            uc, cinv = torch.unique(cid, return_inverse=True)
            nc = uc.numel()
            oi = ((fx >= 1.0).long() * torch.tensor([4, 2, 1], device=dev)).sum(1)
            qinv = cinv * 8 + oi
            nq = nc * 8
            Fc, _ = rs_mean(X, V, qinv, nq)
            cF, _ = rs_mean(X, V, cinv, nc)
            has = torch.zeros(nq, device=dev, dtype=torch.bool).index_put_((qinv,), torch.tensor(True, device=dev))
            Fc = torch.where(has[:, None], Fc, cF.repeat_interleave(8, 0)).contiguous()
            Vc = (torch.zeros(nc, device=dev, dtype=D64).index_add_(0, cinv, V).repeat_interleave(8) / 8.0).contiguous()
            o8 = torch.arange(8, device=dev)
            qfx = (1.0 + GG * (2 * torch.stack([o8 // 4, (o8 % 4) // 2, o8 % 2], 1).double() - 1)).repeat(nc, 1)
            Ai_ = Ai_ / 2
            _, QGW = bsw(qfx, Ai_)
            cb = torch.stack([uc // (Nc[1] * Nc[2]), (uc // Nc[2]) % Nc[1], uc % Nc[2]], 1).repeat_interleave(8, 0)
            qnid = ((cb[:, None, 0] + OFF[None, :, 0]) * Nc[1] + cb[:, None, 1] + OFF[None, :, 1]) * Nc[2] + cb[:, None, 2] + OFF[None, :, 2]
            mdc = restrict_dense(mdn, nbf, Nc)
            mflat = mdc.reshape(-1)
            act = torch.nonzero(mflat > 0).flatten()
            na = act.numel()
            cmp_ = torch.full((mflat.numel(),), -1, device=dev, dtype=torch.long)
            cmp_[act] = torch.arange(na, device=dev)
            qn = cmp_[qnid]
            G = torch.einsum("qab,qna->qnb", Fc.reshape(nq, 3, 3), QGW).contiguous()
            elc = Elem(qn.int().contiguous(), G, Fc, Vc, na)
            elc.eval(torch.zeros(na, 3, device=dev, dtype=D64), 2 | 8)
            elc.f32()
            Mic = (mflat[act] / (TAU * TAU))[:, None]
            s.L.append(dict(el=elc, Mi=Mic, D=torch.linalg.inv(Mic[..., None] * I3 + elc.bd32().double()), nb=Nc, act=act))
            tq, X, V, nbf, mdn = (cb.double() + qfx), Fc, Vc, Nc, mdc
            if na < a.mg_min:
                break
        s.nl = len(s.L)
        for l in range(s.nl):
            na_ = s.L[l]["Mi"].shape[0]
            z = torch.randn(na_, 3, device=dev, dtype=D64, generator=torch.Generator(device=dev).manual_seed(l))
            for _ in range(a.mg_pow):
                w_ = s.sm(l, s.op(l, z))
                lm = (z * w_).sum() / (z * z).sum()
                z = w_ / w_.norm()
            s.L[l]["w"] = a.mg_omega / lm.clamp_min(1e-300)                             # 장치 스칼라 (동기화 없음)
        s.lam = None

    def set_fine(s, D0, T32):
        s.L[0]["D"] = D0
        s.T = T32

    def op(s, l, z):
        """벡터 float64, 헤시안 곱만 float32 (질량 1e-50 대 가장자리 노드가 float32 에서 0 이 되지 않게)."""
        lv = s.L[l]
        if l == 0:
            return lv["Mi"] * z + (s.T @ lv["el"].mv32((s.T @ z[..., None]).squeeze(-1).float()).double()[..., None]).squeeze(-1)
        return lv["Mi"] * z + lv["el"].mv32(z.float()).double()


LAMC = {}                                                                              # 층별 λ_max(D⁻¹A) -- 앞 프레임 값을 이어 씀 (첫 프레임만 거듭제곱법)


class MGD(MG32):
    """가속 판 멀티그리드 (호스트 동기화 없음): 거친 층 = 압축 없는 조밀 상자 (세계 고정 격자, 간격 2 배씩). 거친 칸 전부에 가우스 점 8 개 (빈 칸은 부피 0, F = I),
    노드도 상자 전부 (질량 0 노드는 대각 역행렬 0). 계층은 프레임당 한 번 (프레임 시작 X = F_q), 뉴턴마다 0 층 T·대각만 갱신. λ 는 앞 프레임 값."""

    def __init__(s, ctx, el, Mi, T, D0):
        s.T = T
        s.L = [dict(el=el, Mi=Mi, D=D0, nb=ctx["nb"], act=ctx["act"], dense=False)]
        X, tq, V, nbf = el.Fq, ctx["tq"], el.Vq, ctx["nb"]
        mdn = ctx["mnd"].reshape(*ctx["nb"], 1)
        Ai_ = Ai64
        I9 = torch.eye(3, device=dev, dtype=D64).reshape(9)
        o8 = torch.arange(8, device=dev)
        qb8 = (1.0 + GG * (2 * torch.stack([o8 // 4, (o8 % 4) // 2, o8 % 2], 1).double() - 1))
        for lv in range(1, a.mg_levels):
            Nc = [n // 2 + 3 for n in nbf]
            Cc = [n - 2 for n in Nc]
            ncell = Cc[0] * Cc[1] * Cc[2]
            tc = (tq + 1.5) / 2
            base = torch.floor(tc - 0.5).long()
            fx = tc - base
            cid = (base[:, 0] * Cc[1] + base[:, 1]) * Cc[2] + base[:, 2]
            oi = ((fx >= 1.0).long() * torch.tensor([4, 2, 1], device=dev)).sum(1)
            qinv = cid * 8 + oi
            nq = ncell * 8
            Fc, _ = rs_mean(X, V, qinv, nq, count=False)
            cF, _ = rs_mean(X, V, cid, ncell, count=False)
            has = torch.zeros(nq, device=dev, dtype=torch.bool).index_put_((qinv,), torch.tensor(True, device=dev))
            hasc = torch.zeros(ncell, device=dev, dtype=torch.bool).index_put_((cid,), torch.tensor(True, device=dev))
            cF = torch.where(hasc[:, None], cF, I9)
            Fc = torch.where(has[:, None], Fc, cF.repeat_interleave(8, 0)).contiguous()
            Vc = (torch.zeros(ncell, device=dev, dtype=D64).index_add_(0, cid, V).repeat_interleave(8) / 8.0).contiguous()
            qfx = qb8.repeat(ncell, 1)
            Ai_ = Ai_ / 2
            _, QGW8 = bsw(qb8, Ai_)
            uc = torch.arange(ncell, device=dev)
            cb = torch.stack([uc // (Cc[1] * Cc[2]), (uc // Cc[2]) % Cc[1], uc % Cc[2]], 1).repeat_interleave(8, 0)
            qn = ((cb[:, None, 0] + OFF[None, :, 0]) * Nc[1] + cb[:, None, 1] + OFF[None, :, 1]) * Nc[2] + cb[:, None, 2] + OFF[None, :, 2]
            mdc = restrict_dense(mdn, nbf, Nc)
            mflat = mdc.reshape(-1)
            na = mflat.numel()
            G = torch.einsum("cqab,qna->cqnb", Fc.reshape(ncell, 8, 3, 3), QGW8).reshape(nq, 27, 3).contiguous()
            elc = Elem(qn.int().contiguous(), G, Fc, Vc, na)
            elc.eval(torch.zeros(na, 3, device=dev, dtype=D64), 2 | 8)
            elc.f32()
            Mic = (mflat / (TAU * TAU))[:, None]
            z0m = (mflat == 0).double()[:, None, None]
            Dc = torch.linalg.inv(Mic[..., None] * I3 + elc.bd32().double() + z0m * I3) * (1 - z0m)
            s.L.append(dict(el=elc, Mi=Mic, D=Dc, nb=Nc, act=None, dense=True))
            tq, X, V, nbf, mdn = (cb.double() + qfx), Fc, Vc, Nc, mdc
            if na < a.mg_min:
                break
        s.nl = len(s.L)
        for l in range(s.nl):
            if (l, s.nl, TAU) in LAMC:
                s.L[l]["w"] = LAMC[(l, s.nl, TAU)]
                continue
            na_ = s.L[l]["Mi"].shape[0]
            z = torch.randn(na_, 3, device=dev, dtype=D64, generator=torch.Generator(device=dev).manual_seed(l))
            for _ in range(a.mg_pow):
                w_ = s.sm(l, s.op(l, z))
                lm = (z * w_).sum() / (z * z).sum()
                z = w_ / w_.norm()
            s.L[l]["w"] = a.mg_omega / lm.clamp_min(1e-300)
            LAMC[(l, s.nl, TAU)] = s.L[l]["w"]
        s.lam = None

    def tr(s, l, r):
        A_, B_ = s.L[l], s.L[l + 1]
        if l == 0:
            r = (s.T @ r[..., None]).squeeze(-1)
            dn = torch.zeros(int(np.prod(A_["nb"])), 3, device=dev, dtype=D64)
            dn[A_["act"]] = r
        else:
            dn = r
        return restrict_dense(dn.reshape(*A_["nb"], 3), A_["nb"], B_["nb"]).reshape(-1, 3)

    def pr(s, l, zc):
        A_, B_ = s.L[l], s.L[l + 1]
        z = prolong_dense(zc.reshape(*B_["nb"], 3), A_["nb"]).reshape(-1, 3)
        if l == 0:
            z = (s.T @ z[A_["act"]][..., None]).squeeze(-1)
        return z


def coefs(Uk, vk, hist):
    """적분기 계수: 최소화 E(y) = Σ m'/(2τ²)|y − Ũ|² + Ψ(y) 의 Ũ, 속도 v_{k+1} = α (y − z0) 의 z0·α. 단계 SG (set_stage) 로 정함; 중력은 실제 질량 (GS = m/m')."""
    g_ = gv if GS[0] is None else gv * GS[0]
    if SG["kind"] == "ga":                                                             # 힘을 y = (1−αf) U_{n+1} + αf U_n 에서; m (y − Ũ)/τ² + ∇Ψ(y) = 0
        am, af, gm, be, dt = SG["am"], SG["af"], SG["gm"], SG["be"], SG["dt"]
        an = GA_A[0]
        xp = Uk + dt * vk + dt * dt * (0.5 - be) * an
        ys = af * Uk + (1 - af) * xp
        c1 = (1 - af) * be * dt * dt                                                    # a_{n+1} = (y − ys) / c1
        al = gm * dt / c1                                                              # v_{n+1} = v_n + dt (1−γ) a_n + γ dt a_{n+1} = α (y − z0)
        z0 = ys - (vk + dt * (1 - gm) * an) / al
        return ys + TAU * TAU * (g_ - am * an), z0, al
    if SG["kind"] == "bdf":
        Um, vm = hist
        Ub = SG["c1"] * Uk - SG["c0"] * Um
        vb = SG["c1"] * vk - SG["c0"] * vm
        return Ub + TAU * vb + TAU * TAU * g_, Ub, 1.0 / TAU
    th = SG["th"]
    return Uk + TAU * vk + TAU * TAU * g_, Uk + (1.0 - th) * TAU * vk, 1.0 / (th * TAU)


def endpos(y, Uk, vk=None, z0=None, al=None):
    """걸음 끝 변위: BDF 단계 y, θ 단계 U_k + dt ((1−θ) v_k + θ v_{k+1}) (θ 0.5 이면 2y − U_k, 1 이면 y)."""
    if SG["kind"] == "ga":
        return (y - SG["af"] * Uk) / (1 - SG["af"])
    if SG["kind"] == "bdf" or SG["th"] == 1.0:
        return y
    if SG["th"] == 0.5:
        return 2 * y - Uk
    return Uk + SG["dt"] * ((1.0 - SG["th"]) * vk + SG["th"] * al * (y - z0))


def floor_frac(Ue, FS, na):
    FW, Fqn, Fx0, Fm, Fden = FS
    xs = Fx0 + (FW[..., None] * Ue[Fqn]).sum(1)
    sq = (xs - PT) @ NR
    wst = wstk(sq)
    an = torch.zeros(na, device=dev, dtype=D64).index_add_(0, Fqn.reshape(-1), (FW * (Fm * wst)[:, None]).reshape(-1))
    return torch.where(Fden > 0, an / Fden.clamp_min(1e-300), torch.zeros_like(an)).clamp(0.0, 1.0), sq


def floor_state(y, Uk, z0, Ut, al, na, FS, vk=None):
    """걸음 고착 상태 (T, ŷ, η 시작값). FS = (가중 [n,27], 노드 [n,27], 시작 위치, 질량, 노드 분모)."""
    FW, Fqn, Fx0, Fm, Fden = FS
    Ue = endpos(y, Uk, vk, z0, al)
    r, sq = floor_frac(Ue, FS, na)
    if a.fl_avg > 0:                                                                   # 제안 3: 걸음 안 평균 (시작 r0 -> 끝 r1 선형, 중점 규칙)
        r0, _ = floor_frac(Uk, FS, na)
        ls = torch.zeros_like(r)
        for j in range(a.fl_avg):
            sj = (j + 0.5) / a.fl_avg
            ls = ls + torch.log1p(-((1 - sj) * r0 + sj * r).clamp(0.0, 1.0)) / a.fl_avg
        k = torch.exp(KEX * ls)
        STAT["stuck0"] = int((r0 > 0).sum())                                           # 진단: 걸음 시작 위치로 이미 몫이 있는 노드
        STAT["stuck1"] = int((r > 0).sum())                                            # 걸음 끝 예측 위치로 몫이 있는 노드
        r = torch.maximum(r, r0)
    else:
        k = torch.exp(KEX * torch.log1p(-r))
    vn1 = al * (y - z0)
    Pm = torch.where(((vn1 @ NR) > 0)[:, None, None], I3 - NR[:, None] * NR[None, :], I3.expand(na, 3, 3)) if VOUT else I3.expand(na, 3, 3)
    sk = torch.sqrt(k)[:, None, None]
    T = sk * Pm + (I3 - Pm)
    yh = Ut + ((1 - k)[:, None, None] * Pm @ (z0 - Ut)[..., None]).squeeze(-1)
    eta = ((I3 - Pm) @ (y - Ut)[..., None]).squeeze(-1) + sk.squeeze(-1) * (Pm @ (y - yh)[..., None]).squeeze(-1)
    if a.y0 == "uk":                                                                   # 시작점 = 걸음 시작 위치 (T 의 역: 자유 성분 그대로, 고착 성분 /√k, k = 0 이면 고정)
        dd = Uk - yh
        isk = torch.where(sk > 1e-6, 1.0 / sk.clamp_min(1e-6), torch.zeros_like(sk))
        eta = ((I3 - Pm) @ dd[..., None]).squeeze(-1) + (isk * Pm @ dd[..., None]).squeeze(-1)
    if a.fl_npd > 0:                                                                   # 명시적 판 비관통 (실제 접촉 노드, 안쪽 예측 속도 -> 법선 속도 0)
        real = (sq < -a.fl_npd) & (Fm > 0)
        cn = torch.zeros(na, device=dev, dtype=D64).index_add_(0, Fqn.reshape(-1), ((FW > 0) & real[:, None]).double().reshape(-1))
        npm = (cn > 0) & ((vn1 @ NR) < 0)
        Nn = (NR[:, None] * NR[None, :]).expand(na, 3, 3)
        Q = torch.where(npm[:, None, None], I3 - Nn, I3.expand(na, 3, 3))
        T = T @ Q
        yh = yh + torch.where(npm[:, None], ((z0 - yh) @ NR)[:, None] * NR, torch.zeros_like(yh))
        eta = (Q @ eta[..., None]).squeeze(-1)
        STAT["npd"] = int(npm.sum())
    return T, yh, eta, (r > 0).sum()


def solve_step_fast(el, mn, Uk, vk, fixd, vbc, QW, qn, xq0, mq, ctx, mgh, hist=None, FS=None, mI=None):
    """가속 판 한 걸음: 고정 예산 뉴턴 a.newton x PCG a.cg, 호스트 동기화 없음 (스칼라 장치), 헤시안 float32, MG 계층은 프레임당 한 번 (mgh[0])."""
    na = mn.shape[0]
    Ut, z0, al = coefs(Uk, vk, hist)
    Mi = ((mn if mI is None else mI) / (TAU * TAU))[:, None]
    y = Ut.clone()
    if fixd is not None:
        y[fixd] = Uk[fixd] + TAU * vbc[fixd]
    el.f32()
    g0n = None
    for it in range(a.newton):
        with tsec("바닥"):
          if it == 0 or a.fl_iter:
            if FLOOR:
                T, yh, eta, STAT["stuck_t"] = floor_state(y, Uk, z0, Ut, al, na, FS, vk)
            else:
                T = I3.expand(na, 3, 3)
                yh = Ut
                eta = (Uk - Ut) if a.y0 == "uk" else (y - Ut)
            if fixd is not None:
                T = torch.where(fixd[:, None, None], torch.zeros_like(T), T)
                yh = torch.where(fixd[:, None], Uk + TAU * vbc, yh)
                eta = torch.where(fixd[:, None], torch.zeros_like(eta), eta)
            Tc = T.contiguous()
        yfun = lambda e_: yh + (Tc @ e_[..., None]).squeeze(-1)
        y = yfun(eta)
        with tsec("요소"):
            Ee, fe = el.eval(y, 1 | 2 | 8 | (4 if a.fast_ls else 0), dev_e=True)
            E0q = el.E.clone() if (a.fast_ls and a.ls_diff) else None
            g = Mi * eta + (Tc @ fe[..., None]).squeeze(-1)
            if it == 0:
                g0t = g.norm()
        with tsec("대각"):
            Bd = el.bd32().double()
            D0 = torch.linalg.inv(Mi[..., None] * I3 + Tc @ Bd @ Tc)
        if a.pc == "mg":
            if mgh[0] is None:
                with tsec("MG구성"):
                    mgh[0] = (MGD if a.mg_dense else MG32)(ctx, el, Mi, Tc, D0)
            else:
                mgh[0].set_fine(D0, Tc)
            prec = mgh[0]
        else:
            prec = lambda r_: (D0 @ r_[..., None]).squeeze(-1)
        Aop = lambda p_: Mi * p_ + (Tc @ el.mv32((Tc @ p_[..., None]).squeeze(-1).float()).double()[..., None]).squeeze(-1)
        with tsec("PCG"):
            rr = -g
            zz = prec(rr)
            rz = (rr * zz).sum()
            pp = zz.clone()
            dl = torch.zeros_like(rr)
            for ci in range(a.cg):
                Ap = Aop(pp)
                pAp = (pp * Ap).sum()
                alp = torch.where(pAp > 0, rz / pAp, torch.zeros_like(pAp))
                dl = dl + alp * pp
                rr = rr - alp * Ap
                zo = zz
                zz = prec(rr)
                bet = torch.where(rz > 0, (rr * (zz - zo)).sum() / rz, torch.zeros_like(rz))
                rz = (rr * zz).sum()
                pp = zz + bet * pp
        with tsec("선탐색"):
            if a.fast_ls:
                E0 = 0.5 * (Mi * eta * eta).sum() + Ee
                gd = (g * dl).sum()
                st = torch.zeros((), device=dev, dtype=D64)
                for c_ in (0.25, 0.5, 1.0):                                            # 큰 후보가 받아들여지면 덮어쓴다 = 받아들여지는 가장 큰 걸음
                    et = eta + c_ * dl
                    Et, _ = el.eval(yfun(et), 4, dev_e=True)
                    if E0q is not None:
                        dE = (el.E - E0q).sum() + 0.5 * (Mi * (c_ * dl) * (2 * eta + c_ * dl)).sum()
                        st = torch.where(torch.isfinite(dE) & (dE <= 1e-4 * c_ * gd), torch.full_like(st, c_), st)
                        continue
                    E1 = 0.5 * (Mi * et * et).sum() + Et
                    st = torch.where(torch.isfinite(E1) & (E1 <= E0 + 1e-4 * c_ * gd), torch.full_like(st, c_), st)
            else:
                st = 1.0
            STAT["st0"] = STAT.get("st0", 0) + (st == 0)
            eta = eta + st * dl
            y = yfun(eta)
    if a.gfin:
        with tsec("검사"):
            _, fe_ = el.eval(y, 1)
            STAT["gr"] = max(STAT.get("gr", 0.0), float((Mi * eta + (Tc @ fe_[..., None]).squeeze(-1)).norm() / g0t.clamp_min(1e-300)))
    U1 = endpos(y, Uk, vk, z0, al)
    v1 = al * (y - z0)
    if fixd is not None:
        v1 = torch.where(fixd[:, None], vbc, v1)
    v1 = v1 * DAMP
    STAT["nt"] = STAT.get("nt", 0) + a.newton
    STAT["cg"] = STAT.get("cg", 0) + a.newton * a.cg
    STAT.setdefault("nts", []).append(a.newton)
    STAT.setdefault("cgs", []).append([a.cg] * a.newton)
    STAT["lv"] = mgh[0].nl if mgh[0] is not None else 1
    return U1, v1


STAT = {}


def solve_step(el, mn, Uk, vk, fixd, vbc, QW, qn, xq0, mq, ctx):
    """한 걸음: y 를 구해 (U_{k+1}, v_{k+1}) 를 돌려준다. 입자 배열은 쓰지 않는다."""
    na = mn.shape[0]
    Ut = Uk + TAU * vk + TAU * TAU * gv
    z0 = Uk + (0.5 * TAU * vk if a.integ == "mid" else 0.0)                         # v_{k+1} = α (y − z0)
    al = (2.0 / TAU) if a.integ == "mid" else (1.0 / TAU)
    Mi = (mn / (TAU * TAU))[:, None]
    mq_n = torch.zeros(na, device=dev, dtype=D64).index_add_(0, qn.reshape(-1), (QW * mq[:, None]).reshape(-1)) if FLOOR else None
    y = Ut.clone()
    if fixd is not None:
        y[fixd] = Uk[fixd] + TAU * vbc[fixd]
    g0n = None
    nst = dict(nt=0, cg=0, ls=0, gr=None, stuck=0)
    for it in range(a.newton):
        # 고착 몫 (이 반복의 y 로) -> 변수 바꿈 y = ŷ + T η
        tfl = tsec("바닥"); tfl.__enter__()
        if it > 0 and not a.fl_iter:
            pass                                                                       # 걸음당 한 번 정한 고착 상태 (T, ŷ) 와 η 를 그대로
        elif FLOOR:
            Ue = (2 * y - Uk) if a.integ == "mid" else y
            xq = xq0 + (QW[..., None] * Ue[qn]).sum(1)
            sq = (xq - PT) @ NR
            wst = wstk(sq)
            an = torch.zeros(na, device=dev, dtype=D64).index_add_(0, qn.reshape(-1), (QW * (mq * wst)[:, None]).reshape(-1))
            r = torch.where(mq_n > 0, an / mq_n.clamp_min(1e-300), torch.zeros_like(an)).clamp(0.0, 1.0)     # 몫 (정의상 [0,1]; 반올림만)
            k = torch.exp(KEX * torch.log1p(-r))
            vn1 = al * (y - z0)
            if VOUT:
                outw = (vn1 @ NR) > 0
                Pm = torch.where(outw[:, None, None], I3 - NR[:, None] * NR[None, :], I3.expand(na, 3, 3))
            else:
                Pm = I3.expand(na, 3, 3)
            sk = torch.sqrt(k)[:, None, None]
            T = sk * Pm + (I3 - Pm)
            yh = Ut + ((1 - k)[:, None, None] * Pm @ (z0 - Ut)[..., None]).squeeze(-1)
            # 반복 시작점: 고착 성분은 y_P ← ŷ + k P (y − ŷ) (η_P = √k P (y − ŷ)) -- 새 부분 문제의 출발점 (k 가 바뀐 반복에서 1/√k 넘침 없음; k = 1 이면 y 그대로, k = 0 이면 ŷ)
            eta = ((I3 - Pm) @ (y - Ut)[..., None]).squeeze(-1) + sk.squeeze(-1) * (Pm @ (y - yh)[..., None]).squeeze(-1)
            nst["stuck"] = int((r > 0).sum())
        else:
            T = I3.expand(na, 3, 3)
            yh = Ut
            eta = y - Ut
        if fixd is not None and (it == 0 or a.fl_iter):
            T = torch.where(fixd[:, None, None], torch.zeros_like(T), T)
            yh = torch.where(fixd[:, None], Uk + TAU * vbc, yh)
            eta = torch.where(fixd[:, None], torch.zeros_like(eta), eta)
        Tc = T.contiguous()
        tfl.__exit__()
        yfun = lambda e_: yh + (Tc @ e_[..., None]).squeeze(-1)
        y = yfun(eta)
        with tsec("요소"):
            Ee, fe = el.eval(y, 1 | 2 | 4 | (16 if a.no_proj else 0))
        E0 = 0.5 * float((Mi * eta * eta).sum()) + Ee
        g = Mi * eta + (Tc @ fe[..., None]).squeeze(-1)
        gn = float(g.norm())
        if g0n is None:
            g0n = max(gn, 1e-300)
        nst["gr"] = gn / g0n
        if gn / g0n < a.nt_tol and it > 0:
            break
        # PCG: (Mi + Tᵀ K T) δ = −g, 블록 야코비 전처리
        with tsec("대각"):
            Bd = el.bd()
        if a.pc == "mg":
            with tsec("MG구성"):
                mg_ = MG(ctx, y, el, Mi, Tc, Bd)
            nst["lv"] = mg_.nl
            nst["mglam"] = mg_.lam
            prec = mg_
        else:
            Pc = torch.linalg.inv(Mi[..., None] * I3 + Tc @ Bd @ Tc)
            prec = lambda r_: (Pc @ r_[..., None]).squeeze(-1)
        Aop = lambda p_: Mi * p_ + (Tc @ el.mv((Tc @ p_[..., None]).squeeze(-1))[..., None]).squeeze(-1)
        dl = torch.zeros_like(g)
        rr = -g
        zz = prec(rr)
        rz = float((rr * zz).sum())
        if a.pc == "mg" and not (rz > 0):                                              # 전처리가 양의 정부호가 아니면 이 풀이는 블록 야코비로 (횟수 기록)
            Pc = torch.linalg.inv(Mi[..., None] * I3 + Tc @ Bd @ Tc)
            prec = lambda r_: (Pc @ r_[..., None]).squeeze(-1)
            zz = prec(rr)
            rz = float((rr * zz).sum())
            nst["mgfb"] = nst.get("mgfb", 0) + 1
        pp = zz.clone()
        rz0 = rz
        tpc = tsec("PCG"); tpc.__enter__()
        lz_a, lz_b = [], []
        for ci in range(a.cg):
            Ap = Aop(pp)
            pAp = float((pp * Ap).sum())
            if not (pAp > 0):
                nst["negc"] = nst.get("negc", 0) + 1
                break
            alp = rz / pAp
            lz_a.append(alp)
            dl = dl + alp * pp
            rr = rr - alp * Ap
            zo = zz
            zz = prec(rr)
            rzn = float((rr * zz).sum())
            nst["cg"] += 1
            if rzn <= (a.cg_tol ** 2) * rz0:
                break
            bet_ = float((rr * (zz - zo)).sum()) / rz
            lz_b.append(bet_)
            pp = zz + bet_ * pp                                                       # 유연 PCG (폴라크-리비에르)
            rz = rzn
        tpc.__exit__()
        nst.setdefault("cgs", []).append(ci + 1)
        if a.diag and lz_a:                                                            # 란초스 삼중대각 (CG 계수) -> 전처리 계 고유값 추정
            n_ = len(lz_a)
            Tl = np.zeros((n_, n_))
            for i_ in range(n_):
                Tl[i_, i_] = 1.0 / lz_a[i_] + (lz_b[i_ - 1] / lz_a[i_ - 1] if i_ > 0 else 0.0)
                if i_ + 1 < n_ and i_ < len(lz_b):
                    Tl[i_, i_ + 1] = Tl[i_ + 1, i_] = math.sqrt(max(lz_b[i_], 0.0)) / lz_a[i_]
            ev_ = np.linalg.eigvalsh(Tl)
            nst.setdefault("diag", []).append(dict(g=gn / g0n, cg=ci + 1, res=math.sqrt(max(rzn, 0.0) / max(rz0, 1e-300)) if lz_a else None, lmin=float(ev_[0]), lmax=float(ev_[-1])))
        # 되돌이 선탐색 (아르미호)
        tls = tsec("선탐색"); tls.__enter__()
        gd = float((g * dl).sum())
        st = 1.0
        ok = False
        for li in range(12):
            et = eta + st * dl
            Et, _ = el.eval(yfun(et), 4)
            E1 = 0.5 * float((Mi * et * et).sum()) + Et
            if math.isfinite(E1) and E1 <= E0 + 1e-4 * st * gd:
                ok = True
                break
            st *= 0.5
            nst["ls"] += 1
        if not ok:
            st = 0.0
        eta = eta + st * dl
        y = yfun(eta)
        tls.__exit__()
        if a.diag and nst.get("diag"):
            nst["diag"][-1]["st"] = st
            nst["diag"][-1]["dE"] = (E1 - E0) if ok else None
        nst["nt"] += 1
        if st == 0.0:
            break
    if a.gfin and g0n is not None:
        with tsec("검사"):
            _, fe_ = el.eval(y, 1)                                                     # 마지막 반복의 변수 바꿈 (T, ŷ) 그대로의 η 기울기
            nst["grf"] = float((Mi * eta + (Tc @ fe_[..., None]).squeeze(-1)).norm()) / g0n
    if a.integ == "mid":
        U1 = 2 * y - Uk
        v1 = al * (y - z0)
    else:
        U1 = y
        v1 = (y - Uk) / TAU
    if fixd is not None:
        v1 = torch.where(fixd[:, None], vbc, v1)
    v1 = v1 * DAMP
    for kk_ in ("nt", "cg", "ls"):
        STAT[kk_] = STAT.get(kk_, 0) + nst[kk_]
    STAT["gr"] = max(STAT.get("gr", 0.0), nst.get("grf", nst["gr"]) or 0.0)
    STAT["stuck"] = nst["stuck"]
    STAT["lv"] = nst.get("lv", 1)
    if a.diag:
        STAT.setdefault("diag", []).append(dict(it=nst.get("diag", []), grf=nst.get("grf"), negc=nst.get("negc", 0)))
    STAT.setdefault("nts", []).append(nst["nt"])
    STAT.setdefault("cgs", []).append(nst.get("cgs", []))
    STAT["mglam"] = nst.get("mglam")
    STAT["mgfb"] = STAT.get("mgfb", 0) + nst.get("mgfb", 0)
    return U1, v1


def frame(fi, x, v, F):
    tp = tsec("P2G"); tp.__enter__()
    y = x @ Ai64.T
    base = torch.floor(y - 0.5).long()
    bmin = base.min(0).values
    bb = base - bmin
    nb = (bb.max(0).values + 3).tolist()
    nbt = nb[0] * nb[1] * nb[2]
    cid = (bb[:, 0] * nb[1] + bb[:, 1]) * nb[2] + bb[:, 2]
    fx = y - base
    W, GW = bsw(fx)
    nid = ((bb[:, None, 0] + OFF[None, :, 0]) * nb[1] + bb[:, None, 1] + OFF[None, :, 1]) * nb[2] + bb[:, None, 2] + OFF[None, :, 2]
    if a.fast:                                                                         # P2G 커널 (기저를 커널 안에서; 상자 질량 -> 활성 노드 -> 압축 질량·운동량)
        nbT = torch.tensor(nb, device=dev, dtype=torch.int32)
        yc = y.contiguous()
        AiT = Ai64.contiguous()
        mnd = torch.zeros(nbt, device=dev, dtype=D64)
        AK.launch("an_p2g", Np, (np.int32(Np), AK.P(yc), AK.P(v), AK.P(m64), AK.P(AiT), AK.P(bmin), AK.P(nbT), AK.P(None), AK.P(mnd), AK.P(None)))
        act = torch.nonzero(mnd > 0).flatten()
        na = act.numel()
        cmp_ = torch.full((nbt,), -1, device=dev, dtype=torch.long)
        cmp_[act] = torch.arange(na, device=dev)
        mn = torch.zeros(na, device=dev, dtype=D64)
        mv = torch.zeros(na, 3, device=dev, dtype=D64)
        AK.launch("an_p2g", Np, (np.int32(Np), AK.P(yc), AK.P(v), AK.P(m64), AK.P(AiT), AK.P(bmin), AK.P(nbT), AK.P(cmp_), AK.P(mn), AK.P(mv)))
    else:
        mnd = torch.zeros(nbt, device=dev, dtype=D64).index_add_(0, nid.reshape(-1), (W * m64[:, None]).reshape(-1))
        act = torch.nonzero(mnd > 0).flatten()
        na = act.numel()
        cmp_ = torch.full((nbt,), -1, device=dev, dtype=torch.long)
        cmp_[act] = torch.arange(na, device=dev)
        nidc = cmp_[nid]
        mn = mnd[act]
        mv = torch.zeros(na, 3, device=dev, dtype=D64).index_add_(0, nidc.reshape(-1), (W[..., None] * (m64[:, None] * v)[:, None, :]).reshape(-1, 3))
    nidc = cmp_[nid]
    vn = mv / mn[:, None]
    # 가우스 점 (칸마다 8 점, 같은 가중 = 칸 입자 부피 / 8, 질량 = 칸 입자 질량 / 8)
    uc, cinv = torch.unique(cid, return_inverse=True)
    nc = uc.numel()
    oi = ((fx >= 1.0).long() * torch.tensor([4, 2, 1], device=dev)).sum(1)
    qinv = cinv * 8 + oi
    nq = nc * 8
    Vc = torch.zeros(nc, device=dev, dtype=D64).index_add_(0, cinv, vol64)
    Mc = torch.zeros(nc, device=dev, dtype=D64).index_add_(0, cinv, m64)
    qV = Vc.repeat_interleave(8) / 8.0
    mq = Mc.repeat_interleave(8) / 8.0
    prep = rs_prep(F)
    qF, nbad = rs_mean(F, vol64, qinv, nq, count=False, prep=prep)
    qbad = rs_mean.last_bad
    cF, _ = rs_mean(F, vol64, cinv, nc, count=False, prep=prep)
    has = torch.zeros(nq, device=dev, dtype=torch.bool).index_put_((qinv,), torch.tensor(True, device=dev))
    qF = torch.where(has[:, None], qF, cF.repeat_interleave(8, 0))
    nbad = qbad[has].sum()                                                             # 장치 값 (프레임 끝에 정수로)
    o8 = torch.arange(8, device=dev)
    qb = torch.stack([o8 // 4, (o8 % 4) // 2, o8 % 2], 1).double()
    qfx8 = 1.0 + GG * (2 * qb - 1)
    qfx = qfx8.repeat(nc, 1)
    QW8, QGW8 = bsw(qfx8)                                                              # 팔분 위치 8 가지뿐 -> 8 개만 계산해 넓혀 씀
    QW = QW8.repeat(nc, 1)
    cz = uc % nb[2]
    cy = (uc // nb[2]) % nb[1]
    cx = uc // (nb[1] * nb[2])
    cb = torch.stack([cx, cy, cz], 1).repeat_interleave(8, 0)
    qnid = ((cb[:, None, 0] + OFF[None, :, 0]) * nb[1] + cb[:, None, 1] + OFF[None, :, 1]) * nb[2] + cb[:, None, 2] + OFF[None, :, 2]
    qn = cmp_[qnid]
    if not a.fast:
        assert bool((qn >= 0).all())
    Fq3 = qF.reshape(nq, 3, 3)
    G = torch.einsum("cqab,qna->cqnb", Fq3.reshape(nc, 8, 3, 3), QGW8).reshape(nq, 27, 3).contiguous()   # G_qn = F_qᵀ ∇w_qn
    xq0 = ((cb + bmin).double() + qfx) @ A64.T
    tq = cb.double() + qfx
    if a.quad == "part":                                                               # 입자 적분점: 칸 순 정렬, 칸마다 8 의 배수로 부피 0 더미 (커널의 8 레인 구간 합이 같은 칸을 가정)
        o = torch.argsort(cid)
        cnt = torch.zeros(nc, device=dev, dtype=torch.long).index_add_(0, cinv, torch.ones_like(cinv))
        pc = (cnt + 7) // 8 * 8
        off = torch.cumsum(pc, 0) - pc
        st = torch.cumsum(cnt, 0) - cnt
        co = cinv[o]
        rank = torch.arange(Np, device=dev) - st[co]
        pos = off[co] + rank
        nq = int(pc.sum())
        first = o[st]                                                                  # 칸의 첫 입자
        slot_c = torch.repeat_interleave(torch.arange(nc, device=dev), pc)
        fp = first[slot_c]
        qn = nidc[fp].clone(); qn[pos] = nidc[o]
        G = torch.zeros(nq, 27, 3, device=dev, dtype=D64)
        G[pos] = torch.einsum("pab,pna->pnb", F.reshape(Np, 3, 3)[o], GW[o])
        qF = I3.reshape(1, 9).repeat(nq, 1); qF[pos] = F[o]
        qV = torch.zeros(nq, device=dev, dtype=D64); qV[pos] = vol64[o]
        mq = torch.zeros(nq, device=dev, dtype=D64); mq[pos] = m64[o]
        QW = torch.zeros(nq, 27, device=dev, dtype=D64); QW[pos] = W[o]
        xq0 = x[fp].clone(); xq0[pos] = x[o]
        tq = (bb.double() + fx)[fp].clone(); tq[pos] = (bb.double() + fx)[o]
        G = G.contiguous()
    el = Elem(qn.int().contiguous(), G, qF.contiguous(), qV.contiguous(), na)
    if a.check:
        torch.manual_seed(0)
        for sc_ in (1e-4, 1e-3):
            y0 = sc_ * dxl * torch.randn(na, 3, device=dev, dtype=D64)
            dd = dxl * torch.randn(na, 3, device=dev, dtype=D64)
            E0_, f0 = el.eval(y0, 1 | 2 | 4)
            ep = 1e-7
            Ep, fp = el.eval(y0 + ep * dd, 1 | 4)
            Em, fm = el.eval(y0 - ep * dd, 1 | 4)
            el.eval(y0, 2)
            Hd = el.mv(dd)
            fd = (fp - fm) / (2 * ep)
            print(f"[대조] 변위 {sc_} 칸: dE 유한 차분 {(Ep - Em) / (2 * ep):.6e} vs f·d {float((f0 * dd).sum()):.6e}; |Hd − 유한 차분| / |유한 차분| {float((Hd - fd).norm() / fd.norm()):.3e}; "
                  f"블록 대각 대칭 {float((el.bd() - el.bd().transpose(1, 2)).abs().max()):.2e}", flush=True)
        sys.exit(0)
    fixd = vbc = None
    if a.cfg_bc:
        kq = torch.stack([act // (nb[1] * nb[2]), (act // nb[2]) % nb[1], act % nb[2]], 1) + bmin
        npos = (kq.double() @ A64.T).float()
        fx_, vb_ = RT.cfg_bc_nodes(cfg, [], npos, torch.zeros(1, 4, dtype=torch.long, device=dev), torch.zeros(1, 4, device=dev), m64[:1].float(), fi * h, h)
        if fx_ is not None:
            fixd, vbc = fx_.bool(), vb_.double()
    STAT.clear()
    STAT.update(nq=nq, na=na, nbad=nbad)
    ctx = dict(nb=nb, act=act, mnd=mnd, tq=tq)
    U = torch.zeros(na, 3, device=dev, dtype=D64)
    FS = None
    if FLOOR:
        if a.fl_src == "part":
            FS = (W, nidc, x, m64, mn)
        else:
            FS = (QW, qn, xq0, mq, torch.zeros(na, device=dev, dtype=D64).index_add_(0, qn.reshape(-1), (QW * mq[:, None]).reshape(-1)))
    Kf = a.K
    if a.kad > 0 and FLOOR:                                                            # 제안 2: 적응 걸음 (프레임 시작 입자 위치·속도만)
        sn = (x - PT) @ NR
        vnn = v @ NR
        near = sn < a.kad_d + vnn.clamp(max=0).abs() * h
        if bool(near.any()) and float(vnn[near].abs().max()) > a.kad_v:
            Kf = a.kad
    STAT["Kf"] = Kf
    DTf = h / Kf
    MED0[0] = float(mn.median()) if MED0[0] is None else MED0[0]
    mI = None
    if a.mlow > 0:
        mI = torch.maximum(mn, torch.full_like(mn, a.mlow * MED0[0]))
    if a.mfl_n > 0:
        gq_ = torch.einsum("qna,qna->qn", el.G, el.G).sqrt()
        kn = torch.zeros(na, device=dev, dtype=D64).index_add_(0, qn.reshape(-1), (qV[:, None] * (lam + 2 * mu) * gq_ * gq_.sum(1, keepdim=True)).reshape(-1))
        mI = torch.maximum(mn if mI is None else mI, (h / (2 * a.mfl_n)) ** 2 * kn)
    if mI is not None:
        STAT["mI_n"] = int((mI > mn).sum())
        STAT["mI_add"] = float((mI - mn).sum() / mn.sum())
    if a.mfl > 0:
        gq_ = torch.einsum("qna,qna->qn", el.G, el.G).sqrt()                            # |F_qᵀ ∇w| -- 프레임 시작 적분점 기울기
        kn = torch.zeros(na, device=dev, dtype=D64).index_add_(0, qn.reshape(-1), (qV[:, None] * (lam + 2 * mu) * gq_ * gq_.sum(1, keepdim=True)).reshape(-1))
        mI = torch.maximum(mn if mI is None else mI, a.mfl * (DTf / 2) ** 2 * kn)
        STAT["mfl_n"] = int((mI > mn).sum())
        STAT["mfl_add"] = float((mI - mn).sum() / mn.sum())
    hist = None
    if a.integ == "bdf2":                                                              # 입자 이력 (앞 걸음 위치 − 프레임 시작 위치, 앞 걸음 속도) -> 노드 (질량 가중)
        hd = HIST[0] if HIST[0] is not None else (-DTf * v, v)
        mW = W * m64[:, None]
        Um = torch.zeros(na, 3, device=dev, dtype=D64).index_add_(0, nidc.reshape(-1), (mW[..., None] * hd[0][:, None, :]).reshape(-1, 3)) / mn[:, None]
        vm = torch.zeros(na, 3, device=dev, dtype=D64).index_add_(0, nidc.reshape(-1), (mW[..., None] * hd[1][:, None, :]).reshape(-1, 3)) / mn[:, None]
        hist = (Um, vm)
    tp.__exit__()
    GS[0] = None if mI is None else (mn / mI)[:, None]
    MGH = {}
    an_ = None
    if a.integ in ("ga", "hht"):                                                       # 노드 가속도 상태: 앞 프레임 입자 가속도 P2G, 첫 프레임은 a0 = g − ∇Ψ(0)/m (평형식)
        if HISTA[0] is not None:
            mW = W * m64[:, None]
            an_ = torch.zeros(na, 3, device=dev, dtype=D64).index_add_(0, nidc.reshape(-1), (mW[..., None] * HISTA[0][:, None, :]).reshape(-1, 3)) / mn[:, None]
        else:
            _, f0 = el.eval(torch.zeros(na, 3, device=dev, dtype=D64), 1)
            me = torch.maximum(mn if mI is None else mI, 1e-3 * mn.median())
            an_ = gv[None, :] - f0 / me[:, None]
        if fixd is not None:
            an_ = torch.where(fixd[:, None], torch.zeros_like(an_), an_)
    for _ in range(Kf):
        Up, vp = U, vn
        if a.fast and a.integ == "trbdf2":                                             # 단계 1: 중점 γΔt, 단계 2: BDF2 (U_n, U_γ -> U_{n+1})
            set_stage("th", GAM * DTf, th=0.5)
            Ua, va = solve_step_fast(el, mn, U, vn, fixd, vbc, QW, qn, xq0, mq, ctx, MGH.setdefault(TAU, [None]), FS=FS, mI=mI)
            dd = GAM * (2 - GAM)
            set_stage("bdf", (1 - GAM) * DTf, c1=1.0 / dd, c0=(1 - GAM) ** 2 / dd, tau=(1 - GAM) / (2 - GAM) * DTf)
            U, vn = solve_step_fast(el, mn, Ua, va, fixd, vbc, QW, qn, xq0, mq, ctx, MGH.setdefault(TAU, [None]), hist=(U, vn), FS=FS, mI=mI)
        elif a.fast and a.integ in ("ga", "hht"):
            set_stage("ga", DTf)
            GA_A[0] = an_
            U, vn = solve_step_fast(el, mn, U, vn, fixd, vbc, QW, qn, xq0, mq, ctx, MGH.setdefault(TAU, [None]), FS=FS, mI=mI)
            an1 = (vn - vp - DTf * (1 - SG["gm"]) * an_) / (SG["gm"] * DTf)              # Newmark 속도 식으로 (고착 뒤 속도와 맞게)
            U = Up + DTf * vp + DTf * DTf * ((0.5 - SG["be"]) * an_ + SG["be"] * an1)     # 끝 위치도 Newmark 위치 식 (고착 없으면 (y − αf U_n)/(1−αf) 와 같음; 고착이 속도만 깎아 위치가 바닥 밑으로 외삽되던 것 고침)
            an_ = an1
            if fixd is not None:
                an_ = torch.where(fixd[:, None], torch.zeros_like(an_), an_)
        elif a.fast:
            if a.integ == "bdf2":
                set_stage("bdf", DTf, c1=4.0 / 3.0, c0=1.0 / 3.0, tau=2.0 * DTf / 3.0)
            else:
                set_stage("th", DTf, th=TH0)
            U, vn = solve_step_fast(el, mn, U, vn, fixd, vbc, QW, qn, xq0, mq, ctx, MGH.setdefault(TAU, [None]), hist=hist, FS=FS, mI=mI)
        else:
            assert a.integ != "bdf2" and a.fl_src == "q" and a.fl_npd == 0, "새 플래그는 --fast 경로에만"
            U, vn = solve_step(el, mn, U, vn, fixd, vbc, QW, qn, xq0, mq, ctx)
        hist = (Up, vp)
    if an_ is not None:
        HISTA[0] = (W[..., None] * an_[nidc]).sum(1)
    if a.integ == "bdf2":
        HIST[0] = ((W[..., None] * (hist[0] - U)[nidc]).sum(1), (W[..., None] * hist[1][nidc]).sum(1))
    STAT["nmv"] = el.nmv
    if "st0" in STAT:
        STAT["st0"] = int(STAT["st0"])
    if "stuck_t" in STAT:
        STAT["stuck"] = int(STAT.pop("stuck_t"))
    STAT["nbad"] = int(STAT["nbad"])
    # G2P (프레임 끝 한 번)
    tg = tsec("G2P"); tg.__enter__()
    if a.fast:
        x2 = torch.empty_like(x); v2 = torch.empty_like(v); F2 = torch.empty_like(F)
        AK.launch("an_g2p", Np, (np.int32(Np), AK.P(yc), AK.P(x), AK.P(F.contiguous()), AK.P(AiT), AK.P(bmin), AK.P(nbT), AK.P(cmp_), AK.P(U.contiguous()), AK.P(vn.contiguous()),
                                 AK.P(x2), AK.P(v2), AK.P(F2)))
    else:
        Ug = U[nidc]
        x2 = x + (W[..., None] * Ug).sum(1)
        v2 = (W[..., None] * vn[nidc]).sum(1)
        Am = I3 + torch.einsum("pna,pnb->pab", Ug, GW)
        F2 = (Am @ F.reshape(Np, 3, 3)).reshape(Np, 9)
    tg.__exit__()
    return x2, v2, F2


HIST = [None]
HISTA = [None]
MED0 = [None]


def psi_sum(Fp):
    Fm = Fp.reshape(-1, 3, 3).double()
    S = torch.linalg.svdvals(Fm)
    sg = torch.sign(torch.det(Fm))
    S = torch.cat([S[:, :2], S[:, 2:] * sg[:, None]], 1)
    J = S.prod(1)
    return float((vol64 * (mu * ((S - 1) ** 2).sum(1) + 0.5 * lam * (J - 1) ** 2)).sum())


HN = NR if FLOOR else torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=D64)


def hstat(xx):
    z = (xx @ HN).float()
    q = torch.quantile(z[torch.randperm(z.numel(), generator=torch.Generator().manual_seed(0))[:1000000].to(dev)] if z.numel() > 1000000 else z, torch.tensor([0.001, 0.999], device=dev))
    return float((m64 * (xx @ HN)).sum() / m64.sum()), float(q[1] - q[0])


FGT = DM.traj_F(d, h, dev) if a.en else None
flog = open(os.path.join(a.out, "fem.jsonl"), "w")
errs, secs = [], []
div_at = None
PRED = torch.empty(L, Np, 3) if a.dump else None
AK.mod()
AK.PROF["on"] = bool(a.prof)
for fi in range(L):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x2, v2, F2 = frame(fi, x, v, F)
    torch.cuda.synchronize()
    dt_ = time.perf_counter() - t0
    gt = d["x"][fi + 1].to(dev).double()
    e = float((x2 - gt).norm(dim=-1).mean()) / EXT
    fin = bool(torch.isfinite(x2).all()) and math.isfinite(e)
    errs.append(e if fin else float("nan"))
    secs.append(dt_)
    rec = dict(t=fi, err=e, sec=dt_, **{k_: v_ for k_, v_ in STAT.items()})
    if a.en and fin:
        hc, hr = hstat(x2)
        hcg, hrg = hstat(gt)
        rec["en"] = dict(psi=psi_sum(F2), ke=float(0.5 * (m64 * (v2 * v2).sum(1)).sum()), hc=hc, hc_gt=hcg, hr=hr, hr_gt=hrg)
        if FGT.shape[0] > fi + 1:
            rec["en"]["psi_gt"] = psi_sum(FGT[fi + 1].to(dev))
        if d["v"].shape[0] > fi + 1:
            vg = d["v"][fi + 1].to(dev).double()
            rec["en"]["ke_gt"] = float(0.5 * (m64 * (vg * vg).sum(1)).sum())
    if a.prof:
        rec["kernel_ms"] = AK.prof_take()
    if a.sec:
        rec["sec_ms"] = {k_: [round(t_, 3), c_] for k_, (t_, c_) in SEC.items()}
        SEC.clear()
    flog.write(json.dumps(rec) + "\n")
    flog.flush()
    print(f"  프레임 {fi + 1}/{L} {dt_:.2f} s 오차 {100 * e:.3f}% 뉴턴 {STAT.get('nt')} PCG {STAT.get('cg')} 선탐색 {STAT.get('ls')} 층 {STAT.get('lv')} λ {STAT.get('mglam')} 대체 {STAT.get('mgfb')} 헤시안 곱 {STAT.get('nmv')} 최대 |g|/|g0| {STAT.get('gr', 0):.1e} "
          f"고착 노드 {STAT.get('stuck')} 점 {STAT.get('nq')} 노드 {STAT.get('na')} 유클리드 평균 점 {STAT.get('nbad')} max|v| {float(v2.norm(dim=-1).max()):.3g}", flush=True)
    if PRED is not None:
        PRED[fi] = x2.float().cpu()
    x, v, F = x2, v2, F2
    if not fin or e > 2.0:
        div_at = fi + 1
        print(f"[발산] 프레임 {div_at}", flush=True)
        break
sc = secs[a.warm:]
summ = dict(args=vars(a), frames=L, frames_done=len(errs), diverged=div_at, err_mean=(float(np.mean(errs)) if div_at is None else None),
            err_mean_done=float(np.nanmean(errs)), s_mean=float(np.mean(sc)) if sc else None, dt=DT, cell=dxl)
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
if PRED is not None and div_at is None:
    torch.save({"pred": PRED}, os.path.join(a.out, "pred.pt"))
print(f"[롤아웃] {a.tag} 암시적 B-스플라인 {a.integ} K {a.K} 셀 {a.cell}: {len(errs)}/{L} 프레임{' (발산 %d)' % div_at if div_at else ''}, 평균 {100 * summ['err_mean_done']:.3f}%, "
      f"프레임당 {summ['s_mean']} s (검증판)", flush=True)
