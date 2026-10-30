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
from anchorflow import fem_ah_cuda as AHC                                           # noqa: E402  (절 AH 바닥 정의)
from anchorflow import fem_c_cuda as CCU                                            # noqa: E402
from anchorflow import fem_pl_cuda as PLC                                           # noqa: E402  (절 AG 소성·점성·파괴)
from anchorflow import phys_resid as PR                                             # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--node_h", type=float, default=0.028284)
ap.add_argument("--floor", choices=["particle", "none", "vel"], default="particle",
                help="vel (절 O, --opt --mode lag 만): 서브스텝마다 속도 단위 바닥 -- 바닥 후보 입자의 지금 보간 위치로 접촉 판정, 걸린 노드 속도에 정답 sticky 고착 + 비관통; 에너지 바닥 힘 끔")
ap.add_argument("--vfl_mode", choices=["frac", "full", "slip"], default="frac",
                help="--floor vel 고착 분배: frac = 노드 속도 x (1 - 접촉 입자 질량 몫) (접촉 입자 운동량만 0), full = 접촉 입자가 걸린 노드 속도 0 (정답 노드 sticky 와 같은 꼴), "
                     "slip (절 AG) = 접선 그대로 + 안쪽 법선 성분만 제거 (정답 bounding_box 의 한쪽 방향 속도 0)")
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
ap.add_argument("--dump_div", action="store_true", help="절 AG: 발산해도 끝난 프레임까지 roll_fem.pt 를 저장 (진단 영상용)")
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
ap.add_argument("--fhist", action="store_true",
                help="--lagdiag 덤프 (사용자 10-29 '응력이 왜 프레임 시작부터 큰가', 계산 불변): 프레임마다 시작 입자 x·v·F (f32)·입자 -> 사면체 pinv, "
                     "사면체·노드 위치·∇λ (gl)·셀 F̄_T·V_T, 프레임 끝 노드 U·v -> <out>/fhist/fNNN.pt (exe/fem_fh_stats.py)")
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
ap.add_argument("--lim_sr", type=float, default=-1.0,
                help="절 Z (1) 클램프: 서브스텝 노드 이동 비율 α 로 점유 사면체 σ1(A_T)/σ3(A_T) ≤ c (상자 꼭짓점 이분 + Weyl 빠른 통과, Jacobi 노드 min). --lim Ĵ 를 같이 주면 det 하한도 (기존 상자 판), 안 주면 비율만")
ap.add_argument("--lim_sr_chk", action="store_true", help="절 Z (1) 검산: 이동 뒤 실제 비율 위반 (r > 1.001 c)·det ≤ 0 사면체 x 서브스텝 수 (커널 1 개 더 -- 측정 판에는 끔)")
ap.add_argument("--sb", type=float, default=0.0, help="절 Z (2) 로그 장벽 강성 κ: 사면체 에너지에 V_T κ μ b(c - σ1/σ3(A_T); d̂) (0 = 끔)")
ap.add_argument("--sb_c", type=float, default=2.0, help="--sb 의 비율 상한 c")
ap.add_argument("--sb_dhat", type=float, default=-1.0, help="--sb 의 d̂ (장벽이 켜지는 거리 c - r < d̂; 음수면 0.5 (c - 1))")
ap.add_argument("--cq", action="store_true",
                help="절 AA: 셀 에너지 = 셀 안 입자 에너지 평균의 2 차 근사 V_T [ψ(A F̄) + ½<∂²ψ(A F̄), (A⊗I) Σ_F (A⊗I)ᵀ>] (Σ_F = 셀 입자 F 공분산 9x9, P2G 에서 한 번)")
ap.add_argument("--cq_h", choices=["km", "cf"], default="km", help="--cq 의 헤시안: km = 기존 해석 헤시안 장치 함수 kmanual (기본), cf = 같은 헤시안 닫힌 식 (빠른 판)")
ap.add_argument("--cqchk", default="", help="--cq 검산 (측정 밖): 이 프레임들 (1 부터, 쉼표) 끝에서 셀 에너지 세 가지 (2 차 근사 / 입자 적분 / ψ(A F̄)) 상대 차, 힘 vs float64 자동 미분·중앙 차분 -> cqchk.jsonl")
ap.add_argument("--fspread", type=int, default=0, help="측정 밖: K 프레임마다 입자 F 이웃 벌어짐 ‖F_p − 16-NN 평균‖ (우리 / 정답, 전체 표본·입자 1~2 개 셀 표본; 절 Y 지표) -> fem.jsonl (0 = 끔)")
ap.add_argument("--sp", choices=["none", "merge", "cube", "blend", "stiff", "mlow", "vcap", "vpmesh"], default="none",
                help="절 AD 희소 셀 처리: merge = 희소 사면체를 입자 많은 면 이웃 쪽으로 묶어 한 셀 (F̄·V 합, Ā = 묶음 A 평균) 로 응력, cube = 희소 사면체가 있는 정육면체의 점유 사면체를 한 셀로, "
                     "blend = 희소 사면체 F̄ 를 면 이웃 F̄ 와 입자 부피 가중으로 섞음, stiff = 희소 사면체 응력에 점유율 가중, mlow = 노드 관성 질량 하한, "
                     "vpmesh (절 AE) = 입자 부피를 PG/정답 MPM 정의 (MPM 격자 칸 dx³ / 칸 안 입자 수) 대신 같은 정의를 우리 메시 칸에: 첫 프레임 V_p' = V_기하(T_p) / (T_p 안 입자 수) -- 이후 고정, 질량 (정답과 같은 ρ·V_p^MPM) 그대로, 상한 없음")
ap.add_argument("--sp_a", choices=["avg", "tet"], default="avg",
                help="--sp merge/cube 묶음의 셀 사상: avg = 묶음 하나의 Ā (구성 사면체 A 의 V_t 가중 평균 -- 묶음 안 반대 변형은 에너지 0), tet = 구성 사면체마다 자기 A_t 에 묶음 F̄ (부피 V_t 그대로)")
ap.add_argument("--sp_hg", type=float, default=0.0, help="--sp_a avg 의 모래시계 억제: 묶음 에너지에 γ μ Σ_t V_t |A_t - Ā|² (0 = 끔)")
ap.add_argument("--sp_vcap", type=float, default=1.0, help="--sp vcap: 모든 점유 사면체의 탄성 부피 V_T' = min(V_T, θ V_기하) (입자 부피 합이 사면체 기하 부피를 넘지 않게; 질량 그대로)")
ap.add_argument("--sp_vpm_mass", action="store_true", help="--sp vpmesh: 질량도 같은 정의로 m_p' = ρ_p V_p' (PG 규칙 '칸 부피/칸 입자 수, 질량 = ρ V' 를 메시 칸에 그대로 -- 총질량이 정답과 달라진다)")
ap.add_argument("--vfl_w", choices=["part", "gtk"], default="part",
                help="절 AH (--floor vel): 접촉 가중. part = 평면 아래 입자 질량 전부 (기존), gtk = 정답 MPM sticky 격자 노드의 2 차 B-스플라인 가중 합 w(s) (평면 위 0.5 dx_g 안도 일부; lib/anchorflow/fem_ah_cuda.py)")
ap.add_argument("--vfl_rate", action="store_true",
                help="절 AH (--floor vel): 서브스텝 고착 몫을 단위 시간당으로 -- (1 - a/m)^(h_s/h_ref), h_ref = 정답 substep_dt (기존은 서브스텝마다 (1 - a/m))")
ap.add_argument("--rbdiag", action="store_true", help="절 AH 진단 (측정 밖): 프레임마다 강체 지표 (질량 중심·속도, 운동 에너지, 질량 중심 각운동량, Kabsch 회전, 최저 높이) 우리 / 정답 -> fem.jsonl rb")
ap.add_argument("--vp_h5", default="", help="절 AE: 입자 부피·질량을 정답 0 프레임 h5 (float32 위치) 로 정답 생성기와 같은 식으로 (.pt 위치는 float16 반올림이라 칸 판정이 갈림)")
ap.add_argument("--sp_k", type=int, default=2, help="--sp: 희소 = 사면체 안 입자 수 ≤ k")
ap.add_argument("--sp_occ", type=float, default=0.0, help="--sp: 희소 = (입자 수 ≤ k) 또는 점유율 V_T/V_기하 < 이 값 (0 = 입자 수만)")
ap.add_argument("--sp_beta", type=float, default=1.0, help="--sp blend: 이웃 몫 배율 β (F̄' = (V F̄ + β Σ V_nb F̄_nb)/(V + β Σ V_nb))")
ap.add_argument("--sp_wocc", type=float, default=1.0, help="--sp stiff: 희소 사면체 응력 가중 w = min(점유율 / 이 값, 1)")
ap.add_argument("--sp_mlow", type=float, default=0.25, help="--sp mlow: 노드 관성 질량 하한 = 이 값 x 첫 프레임 노드 질량 중앙값 (가속도 f/m' 에만; 운동량 전달·중력 그대로)")
ap.add_argument("--sp_chk", default="", help="--sp 검산 (측정 밖): 이 프레임들 (1 부터) 끝 상태에서 내부력 합 |Σf|/Σ|f|, 돌림힘 |Σ x×f|/Σ|x-c||f| -> fem.jsonl")
ap.add_argument("--rstat", action="store_true", help="절 R 통계 (뒤집힌 셀 수 등) -- --fb·--lim 이 없어도 같은 사면체 힘 커널 (ex_tet_force_R, κ 0) 로 (기준 판 비교용)")
ap.add_argument("--p2g", choices=["bary", "bs"], default="bary",
                help="절 S (--opt --mode lag, 사용자 10-29): bs = 프레임 시작 P2G 를 2 차 B-스플라인으로 -- 노드 질량 m_n = Σ w(x_n - x_p) m_p·운동량 Σ w m_p v_p (27 노드), "
                     "셀 F̄_c = Σ w(x_c - x_p) V_p F_p / Σ w V_p·V_c = Σ w V_p (셀 중심, 주변 27 셀; 분할 1 이라 Σ V_c = Σ V_p), 사면체 V_T = V_c/6; 메시 = 점유 셀 1 칸 팽창")
ap.add_argument("--bs_mesh", choices=["dil", "occ"], default="dil",
                help="--p2g bs 메시: dil = 점유 셀 1 칸 팽창 (절 S), occ = 절 R 판과 같은 메시 (입자가 든 셀과 그 꼭짓점만, 팽창 없음; 사용자 10-29) -- "
                     "B-스플라인 가중은 메시에 있는 노드·셀에만 쓰고 입자마다 메시 안 가중 합으로 나눠 정규화 (Σ_n w = 1, Σ_c w = 1; 절 T). G2P (bs)·바닥 접촉 가중도 같은 정규화")
ap.add_argument("--bs_mass", choices=["node", "cell"], default="node",
                help="--p2g bs --bs_mesh occ 의 질량·운동량: node = 노드 커널 B-스플라인 (절 T), cell = 절 W (사용자 10-29): 질량·운동량도 F̄·V 와 같은 셀 중심 커널로 셀에 "
                     "(m_c = Σ w~ m_p, (mv)_c = Σ w~ m_p v_p), 노드는 일반 FEM 집중 질량 m_n = Σ_{T∋n} m_T/4 (m_T = m_c/6), 운동량도 같은 경로; 바닥 접촉 분배도 같은 경로")
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
ap.add_argument("--fupd", choices=["T", "bar", "nbar", "navg", "polar", "navgbar"], default="T",
                help="절 AB: 프레임 끝 G2P 입자 F 갱신 -- T = F <- A_T F (기존), bar = A_T[(1-β)F + β F̄_T] (셀 균질화·부분 이완), nbar = A_T[(1-β)F + β F̃] (F̃ = 노드 평균 F̄ 의 λ 보간), "
                     "navg = [(1-β)A_T + β Ã] F (Ã = 노드 평균 A_T 의 λ 보간, 매끈한 속도 기울기), polar = A_T R[(1-β)S + β S̃] (극분해, 신장만 F̃ 쪽으로)")
ap.add_argument("--fupd_b", type=float, default=1.0, help="--fupd 의 β (0 = 기존과 같음)")
ap.add_argument("--fupd_b2", type=float, default=0.2, help="--fupd navgbar: F <- [(1-β)A_T + βÃ][(1-β2)F + β2 F̃] 의 β2")
ap.add_argument("--cfast", action="store_true", help="절 C: 아래 네 가지 (--vfl_cull --sr_fast --tlist --fpolar) 모두")
ap.add_argument("--cfuse", action="store_true", help="절 C (5): --cfast + 서브스텝 커널 융합 (힘+접촉 / 판정 둘 (속도 그 자리) / 노드 걸음+적용) -- 같은 식, 커널 6 -> 4 개")
ap.add_argument("--sr_ef", action="store_true", help="절 C (6): 힘 커널이 ‖A-I‖_F² 를 남기고 클램프 빠른 판정이 그것을 쓴다 (U 다시 안 읽음, 같은 판정)")
ap.add_argument("--fpolar_off", action="store_true", help="--cfast/--cfuse 에서 (4) 빠른 극분해만 끈다")
ap.add_argument("--cfuse_mode", type=int, default=1, help="--cfuse: 1 = 커널 4 개 (판정이 속도를 그 자리 계산), 2 = 힘+접촉만 융합 (노드 걸음·판정·적용 따로, 커널 5 개)")
ap.add_argument("--adapt", choices=["none", "act", "stab"], default="none",
                help="절 C 적응 서브스텝 (--cfuse 만, 정의 변경 -- 결과 바뀜): act = 직전 프레임 클램프 작동 (제한 노드 x 서브스텝) > --ad_thr 이면 --ad_hi, 아니면 --ad_lo; "
                     "stab = 국소 안정 비 ω_f = max_i sqrt(Σ_T V_T (λ+2μ) |F̄_Tᵀ∇λ_i|² / m_i) 로 N_f = N · ω_f/ω_0 (ω_0 = 첫 프레임), [--ad_lo, --ad_hi] 로 자르고 --ad_q 배수로 올림")
ap.add_argument("--ad_lo", type=int, default=128)
ap.add_argument("--ad_hi", type=int, default=384)
ap.add_argument("--ad_thr", type=float, default=0.0)
ap.add_argument("--ad_q", type=int, default=32)
ap.add_argument("--secprof", action="store_true", help="절 C: 프레임 구간 (격자·P2G 준비·서브스텝 그래프·G2P) CUDA 이벤트 시간 -> fem.jsonl sec_* (동기화 추가 없음)")
ap.add_argument("--vfl_cull", action="store_true", help="절 C (1): 바닥 접촉 판정에서 s0 > D (프레임 안 노드 아래쪽 최대 변위) 인 후보 생략 -- 같은 접촉 집합 (lib/anchorflow/fem_c_cuda.py)")
ap.add_argument("--sr_fast", action="store_true", help="절 C (2): 클램프 판정을 빠른 보장 (‖A-I‖_F + ρ) + 목록 사면체당 워프 (기존과 같은 식·같은 s_T) 로")
ap.add_argument("--tlist", action="store_true", help="절 C (3): 프레임마다 점유 사면체 목록 -> 힘·클램프 판정 커널이 그 목록 위로")
ap.add_argument("--cchk", action="store_true", help="절 C 검산 (--no_graph 와, 측정 밖): 서브스텝마다 기존 커널도 같은 상태에서 돌려 클램프 α (비트)·접촉 수·접촉 질량을 대조 -> cchk.json")
ap.add_argument("--fpolar", action="store_true", help="절 C (4): 힘 커널 극분해 rcbrtf + 수렴 조기 종료 (float 반올림 수준 차이)")
ap.add_argument("--diag", action="store_true", help="첫 프레임: 노드별 국소 안정 걸음 2/ω (ω² = Σ_T V_T (λ+2μ)|∇λ_i|² / m_i) 분포, 서브스텝마다 최대 |v|")
ap.add_argument("--plast", action="store_true",
                help="절 AG: 정답 config 의 material 로 정답 솔버 (PG/GF) 재질 식 -- 프레임 끝 G2P 뒤 입자별 return mapping (F_e·ys·logJp·손상 갱신) + 서브스텝마다 셀 응력을 "
                     "셀 소성 상태 (입자 부피 가중 평균) 로 항복면 안에 투영, 응력 모델도 정답 것 (metal·foam: PG StVK, sand: PG Drucker-Prager, plasticine: FCR + 손상, watermelon: GF NACC + neo-Hookean Boarden). "
                     "plasticine 손상 입자 (ys ≤ 0 -> 정답 μ = λ = 0) 는 셀 F̄·탄성 부피에서 빠진다 (질량은 그대로). lib/anchorflow/fem_pl_cuda.py")
ap.add_argument("--pl_cell", choices=["inc", "total", "none"], default="inc",
                help="절 AG 서브스텝 안 셀 return mapping: inc = 정답처럼 서브스텝마다 증분 (셀 F_e·상태를 서브스텝마다 갱신, F_trial = X_j X_(j-1)⁻¹ F_e, 걸음 h_s; 프레임 시작 F_e = F̄, 상태 = 입자 평균), "
                     "total = 프레임 누적 X = A F̄ 를 프레임 시작 상태로 한 번에 투영 (첫 구현 -- 발산), none = 셀 투영 없음 (프레임 끝 입자만)")
ap.add_argument("--pl_diag", type=int, default=0, help="절 AG 진단 전용: 1 = 셀 투영 끔 (셀 응력은 재질 응력 모델만, return mapping 은 프레임 끝 입자에만), 2 = 셀 응력을 FCR 로 (투영은 그대로), 3 = 둘 다")
ap.add_argument("--visc", type=float, default=0.0, help="절 AG: 셀 점성 응력 τ_v = 2 η J sym(∇v) (Pa·s, 서브스텝마다 셀 속도 기울기) -- 정답 솔버 (PG/GF) 에는 없는 항, 구현만")
ap.add_argument("--pbc", action="store_true", help="절 AG: 정답 config 의 입자 속도 구속 (enforce_particle_translation, release_particles_sequentially -- PG 와 같은 50 구역) 을 "
                "노드 몫 섞기로: 프레임 가운데 시각에 걸린 입자의 P2G 질량 몫 α 로 서브스텝마다 노드 걸음 뒤 v <- (1 - α) v + α v̄_bc (선택은 0 프레임 위치, PG 와 같다)")
ap.add_argument("--floor_plane", default="", help="절 AG: 바닥 평면을 config surface_collider 대신 'px,py,pz,nx,ny,nz' 로 (GF 수박: config 의 z=0 sticky 는 GF 에서 작동하지 않고 "
                "바닥은 bounding_box 의 padding 3 칸 -- z = 3 dx)")
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
if a.vp_h5:                                                                          # 절 AE: .pt 의 0 프레임 위치는 float16 반올림 (최대 0.49 mm) -> 칸 판정이 정답 생성기와 갈린다.
    import h5py                                                                      # 정답 생성기 (lb_gen.py) 가 실제로 본 float32 위치로 같은 식 V_p = dx³ / 칸 입자 수, m_p = ρ V_p
    with h5py.File(a.vp_h5) as h_:
        xg_ = np.array(h_["x"])
    xg_ = torch.as_tensor(xg_.T if xg_.shape[0] == 3 else xg_, dtype=torch.float32, device=dev)
    assert xg_.shape[0] == m.shape[0] and float((xg_ - d["x"][0].to(dev).float()).abs().max()) < 1e-3, "vp_h5: 입자 수·순서가 다르다"
    ng_, dx_ = int(cfg.get("n_grid", 100)), float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
    c_ = (xg_ / dx_).floor().long()
    _, i_, n_ = torch.unique((c_[:, 0] * ng_ + c_[:, 1]) * ng_ + c_[:, 2], return_inverse=True, return_counts=True)
    v_ = (dx_ ** 3 / n_[i_].float())
    print(f"[절 AE vp_h5] V_p 를 정답 float32 위치로: 다른 입자 {float(((v_ - vol).abs() / v_ > 1e-4).float().mean()):.3f}, Σ V_p {float(vol.double().sum()):.6g} -> {float(v_.double().sum()):.6g}", flush=True)
    rr_ = (rho if rho is not None else float(cfg["density"]))
    m = (v_ * rr_).float().contiguous()
    vol = v_.float().contiguous()
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
GTOK = torch.isfinite(GT).all(-1)                                                    # 절 AG: GF 는 진행 중 입자를 비유한값으로 떨군다 (quarantine) -- 지표는 정답이 유한한 입자만
GT_ALLOK = bool(GTOK.all())
if not GT_ALLOK:
    print(f"[정답] 비유한 입자 (프레임 1 / 마지막): {int((~GTOK[0]).sum())} / {int((~GTOK[-1]).sum())} -- 오차는 정답이 유한한 입자만", flush=True)
gv = [float(q) for q in cfg["g"]]
mu, lam = PR.lame(cfg["E"], cfg["nu"])


def rb_stats(xa, va, xb, vb):
    """절 AH --rbdiag: 우리 (xa, va) / 정답 (xb, vb) 의 강체 지표. 질량 m (정답 V_p 질량), 0 프레임 기준 Kabsch 회전."""
    out = {}
    md_ = m.double()
    x0d = d["x"][0].to(dev).double()
    c0 = (md_[:, None] * x0d).sum(0) / md_.sum()
    Rs = []
    for k_, (xx, vv) in (("o", (xa, va)), ("g", (xb, vb))):
        xx, vv = xx.double(), vv.double()
        c = (md_[:, None] * xx).sum(0) / md_.sum()
        vc = (md_[:, None] * vv).sum(0) / md_.sum()
        r_ = xx - c
        Lc = (md_[:, None] * torch.cross(r_, vv - vc, dim=-1)).sum(0)
        H_ = ((x0d - c0) * md_[:, None]).T @ r_
        U_, S_, Vh_ = torch.linalg.svd(H_)
        D_ = torch.diag(torch.tensor([1.0, 1.0, float(torch.sign(torch.det(Vh_.T @ U_.T)))], device=dev, dtype=torch.float64))
        R_ = Vh_.T @ D_ @ U_.T
        Rs.append(R_)
        out[k_] = dict(c=c.tolist(), vc=vc.tolist(), ke=float(0.5 * (md_ * (vv ** 2).sum(-1)).sum()), ke_c=float(0.5 * md_.sum() * (vc ** 2).sum()),
                       L=Lc.tolist(), zmin=float(xx[:, 2].min()), up=R_[:, 2].tolist())
    cs = float(((Rs[0] @ Rs[1].T).trace() - 1) / 2)
    out["ang"] = float(math.degrees(math.acos(max(-1.0, min(1.0, cs)))))
    out["dc"] = float((torch.tensor(out["o"]["c"]) - torch.tensor(out["g"]["c"])).norm())
    return out
planes = [b for b in (cfg.get("boundary_conditions") or []) if b.get("type") == "surface_collider"]
if a.floor_plane:                                                                   # 절 AG: 바닥 평면 직접 (GF bounding_box 바닥)
    _fp = [float(q) for q in a.floor_plane.split(",")]
    planes = [dict(type="surface_collider", point=_fp[:3], normal=_fp[3:6], surface="slip" if a.vfl_mode == "slip" else "sticky")]
VFM = {"frac": 0, "full": 1, "slip": 2}[a.vfl_mode]
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
if a.fhist and not (a.lagdiag and a.opt and a.mode == "lag"):
    raise SystemExit("--fhist 은 --opt --mode lag --lagdiag 와 함께")
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
SRL = a.lim_sr > 0                                                                  # 절 Z (1) 비율 클램프
SB = a.sb > 0                                                                       # 절 Z (2) 비율 로그 장벽
SB_D = (a.sb_dhat if a.sb_dhat > 0 else 0.5 * (a.sb_c - 1.0)) if SB else 0.0
if SB and not (a.sb_c > 1.0 and SB_D > 0):
    raise SystemExit("--sb: c > 1, d̂ > 0 이어야")
if SRL and a.lim_mode != "box":
    raise SystemExit("--lim_sr 은 --lim_mode box (Jacobi 상자 판) 만")
CQ = bool(a.cq)                                                                     # 절 AA: 2 차 근사 셀 에너지
if CQ and (NINT or SB or a.fb > 0 or a.p2g != "bary" or a.rep or a.affine or a.mode != "lag" or not a.opt or a.interp != "bary" or a.floor == "particle"):
    raise SystemExit("--cq 는 --opt --mode lag --p2g bary (노드 적분·장벽·재평균·아핀·에너지 바닥 없음) 만")
if a.cqchk and not CQ:
    raise SystemExit("--cqchk 은 --cq 와 함께")
FUPD = {"T": 0, "bar": 1, "nbar": 2, "navg": 3, "polar": 4, "navgbar": 5}[a.fupd]                    # 절 AB: G2P 입자 F 갱신 방식
FUPDN = FUPD >= 2                                                                      # 노드 평균 (fa_tet_acc) 필요
if FUPD and not (a.opt and a.mode == "lag" and a.interp == "bary" and a.p2g == "bary" and not a.rep and not a.affine and a.rep_w == "std"):
    raise SystemExit("--fupd 는 --opt --mode lag --p2g bary (재평균·아핀 없음) 만")
if a.cchk and not a.no_graph:
    raise SystemExit("--cchk 는 --no_graph 와 함께")
if a.cfuse:
    a.cfast = True
if a.cfast:
    a.vfl_cull = a.sr_fast = a.tlist = a.fpolar = True
if a.fpolar_off:
    a.fpolar = False
CC = a.vfl_cull or a.sr_fast or a.tlist or a.fpolar                                  # 절 C 경로
SPM = a.sp
SPG = SPM in ("merge", "cube") and a.sp_a == "avg"                                                       # 절 AD: 묶음 힘 (커널 2 개)
if SPM != "none" and not (a.opt and a.mode == "lag" and a.interp == "bary" and not a.rep and a.p2g == "bary" and not NINT and not CQ and not SB and a.fb <= 0 and not a.affine):
    raise SystemExit("--sp 는 --opt --mode lag --p2g bary (절 R 경로, 재평균·노드 적분·2 차 근사·장벽·아핀 없음) 만")
if a.sp_chk and SPM == "none":
    raise SystemExit("--sp_chk 는 --sp 와 함께")
RNEW = SPM != "none" or a.fb > 0 or a.lim >= 0 or SRL or SB or a.rstat or a.p2g == "bs" or NINT or CQ                                         # 절 R 경로 (사면체 힘 ex_tet_force_R + 통계)
LIM = a.lim >= 0 or SRL
BSL = a.p2g == "bs"                                                                  # 절 S: lag 판 B-스플라인 P2G
G2PBS = BSL and a.g2p == "bs"
BSN = 1 if (BSL and a.bs_mesh == "occ") else 0                                       # 절 T: 팽창 없는 메시 + 메시 안 가중 정규화
BSCM = BSL and a.bs_mass == "cell"                                                    # 절 W: 질량·운동량도 셀 중심 커널로 셀에, 노드 m_T/4
if a.bs_mass == "cell" and not BSN:
    raise SystemExit("--bs_mass cell 은 --p2g bs --bs_mesh occ 와 함께")
if a.bs_mesh == "occ" and not BSL:
    raise SystemExit("--bs_mesh occ 는 --p2g bs 와 함께")
if BSL and not (a.opt and a.mode == "lag" and a.interp == "bary" and not REP and not a.affine and not FLOOR):
    raise SystemExit("--p2g bs 는 --opt --mode lag (재평균·아핀·에너지 바닥 없음) 만")
if BSL and a.lagdiag and not VFL:
    raise SystemExit("--p2g bs --lagdiag 는 --floor vel 만 (진단 루프가 같은 _step_lag 를 부르는 경로)")
if a.g2p == "bs" and not BSL:
    raise SystemExit("--g2p bs 는 --p2g bs 와 함께")
if (SRL or SB) and (NINT or BSL):
    raise SystemExit("--lim_sr/--sb 는 절 R 경로 (--p2g bary, 노드 적분 없음) 만")
if CC and not (a.opt and a.mode == "lag" and a.interp == "bary" and not REP and RNEW and not (NINT or CQ or SB or BSL) and not SPG and LIM and a.lim_mode == "box"):
    raise SystemExit("절 C (--cfast 등) 는 --opt --mode lag 상자 판 이동 제한 (--lim_sr 또는 --lim --lim_mode box) 경로만")
if a.adapt != "none" and not a.cfuse:
    raise SystemExit("--adapt 는 --cfuse 와 함께")
if a.sr_ef and not (a.sr_fast and (a.cfuse_mode == 2 or not a.cfuse)):
    raise SystemExit("--sr_ef 는 --sr_fast (--cfuse 면 --cfuse_mode 2) 와 함께")
if a.cfuse and SPM == "mlow" and a.cfuse_mode != 2:
    raise SystemExit("--sp mlow + --cfuse 는 --cfuse_mode 2 만 (노드 걸음 커널이 따로인 판)")
if a.cfuse and not (SRL and not FLOOR and not a.cchk):
    raise SystemExit("--cfuse 는 --lim_sr (에너지 바닥·검산 없음) 만")
if a.vfl_cull and not VFL and HASPL:
    raise SystemExit("--vfl_cull 은 --floor vel 만")
if RNEW and not (a.opt and a.mode == "lag" and a.interp == "bary" and not REP):
    raise SystemExit("--fb/--lim/--rstat 은 --opt --mode lag (재평균 없음) 만")
NB = -(-a.N // REP) if REP else 1                                                   # 프레임당 블록 수 (마지막 블록은 짧을 수 있다)
# ---- 절 AG: 소성·점성·파괴 (정답 솔버 재질 식), 입자 속도 구속 ----
PLAST, VISC, PBC = bool(a.plast), a.visc > 0, bool(a.pbc)
if (PLAST or VISC or PBC) and not (a.opt and a.mode == "lag" and a.cfuse and a.cfuse_mode == 2 and not a.sr_ef and SPM in ("none", "vcap")
                                   and not FUPD and not REP and not a.lagdiag and a.adapt == "none" and not a.affine and a.p2g == "bary"):
    raise SystemExit("절 AG (--plast/--visc/--pbc) 는 확정 경로 (--opt --mode lag --cfuse --cfuse_mode 2, --sp none|vcap, F 갱신·P2G 기본, 진단·적응 없음) 만")
MATN, PL_DMG = 0, False
PLINC = PLAST and a.pl_cell == "inc"
PRM = torch.zeros(16, device=dev)
PRM[0], PRM[1] = mu, lam
if PLAST:
    MATN, PRM, PLP = PLC.params(cfg, mu, lam, dev)
    PL_YS = torch.full((Np,), PLP["yield_stress"], device=dev)                      # 입자 항복 응력 (PG model.yield_stress, 시작 = config)
    PL_LJ = torch.full((Np,), PLP["logJp0"], device=dev)                            # 입자 log Jp (GF particle_Jp, 시작 -0.04)
    PL_AL = torch.ones(Np, device=dev)                                               # 1 = 살아 있음, 0 = 손상 (PG plasticine ys ≤ 0 -> μ = λ = 0)
    PL_DMG = MATN == 5
    PL_RMC = torch.zeros(2, device=dev)                                              # 프레임마다: [0] 투영된 입자 수, [1] 새로 손상된 입자 수
    print(f"[절 AG 소성] 재질 {cfg.get('material')} (번호 {MATN}), " + ", ".join(f"{k} {v:.6g}" for k, v in PLP.items()), flush=True)
if VISC:
    print(f"[절 AG 점성] 셀 점성 응력 η {a.visc:g} Pa·s (정답 솔버에는 없는 항)", flush=True)
if PBC:                                                                              # 프레임별 [L, Np] 구속 여부·마지막으로 건 조건 번호 (PG 는 조건 순서대로 덮어쓴다)
    _x0b = d["x"][0].to(dev).float()
    _bcs = []
    for _bc in (cfg.get("boundary_conditions") or []):
        _ty = _bc.get("type")
        if _ty == "enforce_particle_translation":
            _bcs.append((_bc["point"], _bc["size"], _bc["velocity"], float(_bc["start_time"]), float(_bc["end_time"])))
        elif _ty == "release_particles_sequentially":                                 # PG mpm_solver_warp.release_particles_sequentially 그대로 (num_layers 는 무시하고 50)
            _nl, _nv = 50, _bc["normal"]
            _pt, _sz, _ax = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], -1
            for _k in range(3):
                if _nv[_k] == 0:
                    _pt[_k], _sz[_k] = 1.0, 1.0
                else:
                    _ax, _pt[_k] = _k, float(_bc["end_position"])
            _hl = abs(float(_bc["start_position"]) - float(_bc["end_position"])) / _nl
            for _k in range(_nl):
                _s2 = list(_sz)
                _s2[_ax] = _hl * (_nl - _k)
                _bcs.append((list(_pt), _s2, [0.0, 0.0, 0.0], float(_bc["start_time"]), float(_bc["end_time"]) / _nl * (_k + 1)))
        elif _ty in ("particle_impulse", "enforce_particle_velocity_rotation", "enforce_particle_velocity_cylinder", "cuboid"):
            raise SystemExit(f"--pbc: {_ty} 는 아직 없다")
    if not _bcs:
        raise SystemExit("--pbc 인데 config 에 입자 속도 구속이 없다")
    PBC_V = torch.tensor([b_[2] for b_ in _bcs], device=dev, dtype=torch.float32)
    _msk = [((_x0b - torch.tensor(b_[0], device=dev)).abs() < torch.tensor(b_[1], device=dev)).all(-1) for b_ in _bcs]
    PBC_K = torch.full((L, Np), -1, dtype=torch.int16, device=dev)
    for _i in range(L):
        _t = (_i + 0.5) * h                                                          # 프레임 가운데 시각 (PG 는 서브스텝마다 time ∈ [start, end))
        for _k, b_ in enumerate(_bcs):
            if b_[3] <= _t < b_[4]:
                PBC_K[_i][_msk[_k]] = _k
    print(f"[절 AG 입자 구속] 조건 {len(_bcs)} 개, 프레임 0 구속 입자 {int((PBC_K[0] >= 0).sum())}, 마지막 프레임 {int((PBC_K[L - 1] >= 0).sum())} / {Np}", flush=True)
    del _msk
if HASPL:
    pt_ = [float(q) for q in planes[0]["point"]]
    nr_ = np.array(planes[0]["normal"], dtype=np.float64)
    nr_ = (nr_ / np.linalg.norm(nr_)).tolist()
    PL = [np.float64(q) for q in pt_ + nr_]
    PLf = [np.float32(q) for q in pt_ + nr_]
else:
    PL = [np.float64(0)] * 6
    PLf = [np.float32(0)] * 6
AHF = VFL and (a.vfl_w == "gtk" or a.vfl_rate)                                     # 절 AH 바닥 정의 (fem_ah_cuda)
AH_GTK = 1 if (VFL and a.vfl_w == "gtk") else 0
AH_DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))                # 정답 MPM 격자 dx_g
AH_PN = float(np.dot(pt_, nr_)) if HASPL else 0.0                                   # p·n (정답 격자 노드 법선 좌표와 비교)
AH_BAND = 0.5 * AH_DXG if AH_GTK else 0.0                                           # gtk: 평면 위 0.5 dx_g 까지 가중 > 0
AH_KEX = (hs / float(cfg["substep_dt"])) if (VFL and a.vfl_rate) else 1.0           # (1 - a/m)^(h_s/h_ref)
if AHF:
    if not (a.opt and a.mode == "lag" and a.cfuse and a.cfuse_mode == 2) or PLAST or VISC or PBC:
        raise SystemExit("절 AH 바닥 (--vfl_w gtk / --vfl_rate) 은 확정 경로 (--opt --mode lag --cfuse --cfuse_mode 2, 탄성) 만")
    if abs(sum(abs(q) for q in nr_) - 1.0) > 1e-6:
        raise SystemExit("--vfl_w gtk 는 축 정렬 평면만 (정답 격자 노드 좌표 비교)")
    print(f"[절 AH 바닥] 가중 {a.vfl_w} (정답 격자 dx_g {AH_DXG:g}, p·n {AH_PN:g}, 고착 격자 노드 최대 지표 {AHC.gt_floor_k0(AH_DXG, AH_PN) if AH_GTK else '-'}, 띠 {AH_BAND:g} m), "
          f"고착 몫 {'(1 - a/m)^(h_s/h_ref), h_s/h_ref = %.4g' % AH_KEX if a.vfl_rate else '(1 - a/m) 서브스텝마다 (기존)'}", flush=True)
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
      + (", [절 W] 질량·운동량도 셀 중심 커널로 셀에 -> 노드 집중 m_T/4 (가중 일치)" if BSCM else "")
      + (f", [절 V] 노드 적분 FEM (F_n = Σ V_T F_T / Σ V_T, 노드 FCR 응력, β {a.nint_beta:g})" if NINT else "")
      + (f", [절 R] log J 장벽 κ {a.fb:g} Ĵ {a.fb_jhat:g} J=det({'A F̄' if a.fb_on == 'X' else 'A'})" if a.fb > 0 else "")
      + (f", [절 R] 이동 제한 det A_T ≥ {a.lim:g} ({'상자 꼭짓점 Jacobi' if a.lim_mode == 'box' else '8 색 Gauss-Seidel'}, 속도 {a.lim_v})" if a.lim >= 0 else "")
      + (f", [절 Z] 비율 클램프 σ1/σ3(A_T) ≤ {a.lim_sr:g} (상자 꼭짓점 이분 + Weyl, Jacobi, 속도 {a.lim_v}){' + 검산' if a.lim_sr_chk else ''}" if SRL else "")
      + (f", [절 AA] 셀 에너지 2 차 근사 (F̄ + 공분산 Σ_F 9x9, 헤시안 {'kmanual' if a.cq_h == 'km' else '닫힌 식'})" if CQ else "")
      + (f", [절 AD] 희소 셀 {SPM} (희소 = 입자 ≤ {a.sp_k}{' 또는 점유율 < %g' % a.sp_occ if a.sp_occ > 0 else ''}"
         + (f", β {a.sp_beta:g}" if SPM == "blend" else "") + (f", w = min(점유율/{a.sp_wocc:g}, 1)" if SPM == "stiff" else "")
         + (f", 관성 질량 하한 {a.sp_mlow:g} x 첫 프레임 중앙" if SPM == "mlow" else "") + ")" if SPM != "none" else "")
      + (f", [절 Z] 비율 로그 장벽 κ {a.sb:g} c {a.sb_c:g} d̂ {SB_D:g}" if SB else "")
      + (f", [절 AB] G2P 입자 F 갱신 {a.fupd} β {a.fupd_b:g}" + (f" β2 {a.fupd_b2:g}" if FUPD == 5 else "") if FUPD else "") + (", [절 R] 통계" if RNEW else ""), flush=True)
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
        self.graphs = {}                                                            # 절 C 적응 서브스텝: N 별 그래프
        self.Ncur, self.hscur, self.dampcur = a.N, hs, DAMP
        self.om0 = None
        self.n_cap = 0
        self.SORT = (a.mode == "sub" and not a.no_fuse and not a.no_sort)
        self.mA, self.volA = m, vol
        if PL_DMG:                                                                  # 절 AG: 손상 입자 (al 0) 는 셀 F̄·탄성 부피에서 뺀다 -- 프레임마다 vol x al
            self.volW = vol.clone()
            self.volA = self.volW
        if self.SORT:
            self.perm = torch.zeros(Np, dtype=torch.long, device=dev)
            self.mA, self.volA = m.clone(), vol.clone()
            self.OX, self.OV, self.OF = z(Np, 3), z(Np, 3), z(Np, 9)
        self.fixd = None
        self._fhook = None                                                          # --lagdump: 서브스텝 노드 걸음 직전 내부력 f 복사 (진단 루프만, 그래프 판은 None)
        self.ncon = torch.zeros(1, dtype=torch.int32, device=dev)                    # --floor vel: 프레임 동안 접촉 (입자·서브스텝) 수
        self.bst = z(4)                                                              # 절 S: [0] 질량 0 노드 때문에 뺀 사면체 부피, [1] 그 사면체 수; 절 T: [2] 정규화로 메시 안 노드에 옮긴 질량 Σ(1-Win)m_p, [3] 옮긴 부피 Σ(1-Cin)V_p
        self.rst = z(12)                                                             # 절 R 통계 (프레임마다 비움): [0] 뒤집힌 점유 사면체 x 서브스텝, [1] 제한 노드 x 서브스텝, [2:5] 줄인 운동량, [5] 줄인 KE, [6] Σ|줄인 운동량|
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
            self.accN = z(self.Mcap)                                                 # 절 AH: 노드별 실제 접촉 (평면 아래) 입자 수 (비관통 판정, 쓴 뒤 비움)
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
            if CC:                                                                   # 절 C: [0] 클램프 목록 개수, [1] D (float 비트), [2] 점유 사면체 수
                self.cint = torch.zeros(4, dtype=torch.int32, device=dev)
                if a.cchk:
                    self.acc_ref = z(self.Mcap)
                    self.ncon_ref = torch.zeros(1, dtype=torch.int32, device=dev)
                    self.alph_ref = torch.full((self.Mcap,), 1065353216, dtype=torch.int32, device=dev)
                    self.rst_ref = z(12)
                    if not hasattr(self, "chk"):
                        self.chk = {k: torch.zeros((), dtype=torch.float64, device=dev) for k in
                                    ("ncon_diff", "ncon_ref", "acc_set_diff", "acc_absdiff", "acc_sum", "alpha_bit_diff", "alpha_lim_nodes", "list_len", "occ_tets")}
            if FUPDN:                                                                # 절 AB: 노드 V_T 가중 합 (9 + 무게)
                self.NA = z(self.Mcap, 10)
                self.NB = z(self.Mcap, 10) if FUPD == 5 else None
            if SPM == "mlow":
                self.mI = z(self.Mcap)                                               # 절 AD mlow: 관성 질량 max(m_n, m_low)
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
            if SPM != "none":                                                       # 절 AD: 묶음 (루트·A 합 두 벌·F̄·V·구성 수), 섞은 F̄, 응력 가중 부피
                self.grp = torch.full((self.Tcap,), -1, dtype=torch.int32, device=dev)
                self.AG = [torch.zeros(self.Tcap, 9, device=dev), torch.zeros(self.Tcap, 9, device=dev)]
                self.FG = torch.zeros(self.Tcap, 9, device=dev)
                self.VG = torch.zeros(self.Tcap, device=dev)
                self.nmem = torch.ones(self.Tcap, device=dev)
                self.VF = torch.zeros(self.Tcap, device=dev)
            if CC:                                                                   # 절 C: 클램프 목록 [개수, 사면체...], 점유 사면체 목록
                self.clst = torch.zeros(self.Tcap + 1, dtype=torch.int32, device=dev)
                self.tl = torch.zeros(self.Tcap, dtype=torch.int32, device=dev)
                self.eF = torch.zeros(self.Tcap, device=dev)                         # (6) ‖A - I‖_F²
            if CQ:
                self.CS = torch.zeros(self.Tcap, 45, device=dev)                     # 절 AA: 셀 입자 F 공분산 (위 삼각 45)
            if NINT:
                self.XTu = torch.zeros(self.Tcap, 9, device=dev)                     # 절 V: 사면체 (V_T/4) F_T
            new = True
        if CC and not hasattr(self, "s0c"):
            self.s0c = torch.zeros(Np, device=dev)                                   # 절 C (1): 바닥 후보 시작 높이
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
            if BSCM:                                                                  # 절 W: 셀 질량·운동량, 바닥 접촉 셀 질량
                self.cM = torch.zeros(self.Pcap, device=dev)
                self.cMV = torch.zeros(self.Pcap * 3, device=dev)
                self.accC = torch.zeros(self.Pcap, device=dev)
            new = True
        if new:
            self.graph = None
            self.graphs = {}

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
        if SPM == "cube":
            EX.launch("ex_tet_cell", Pn * 9, (EX.P(self.ipar), EX.P(tm), EX.P(self.tcm), EX.P(self.tcell)), b=256)
            self.Pn = Pn
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
        if CQ:                                                                       # 절 AA: 두 번째 패스 -- 같은 입자 집합의 F 공분산
            self.CS.zero_()
            EX.launch("ex_cov_acc", Np, (np.int32(Np), EX.P(self.pinv), EX.P(self.nslot), np.int32(1 if VFL else 0), EX.P(self.volA), EX.P(F),
                                         EX.P(self.FT), EX.P(self.CS)), b=256)
            EX.launch("ex_cov_fin", Tc, (EX.P(self.sz), EX.P(self.VTs), EX.P(self.CS)), b=256)

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
        if BSCM:                                                                     # 절 W: 셀 질량·운동량·V·VF (셀 중심 커널) -> 사면체 -> 노드 m_T/4
            self.cM[:Pn].zero_()
            self.cMV[:Pn * 3].zero_()
            EX.launch("bsc_p2g", Np, (np.int32(Np), EX.P(x), EX.P(v), EX.P(F), EX.P(self.mA), EX.P(self.volA), EX.P(self.gpar), EX.P(self.ipar),
                                      EX.P(self.cmk), EX.P(self.cM), EX.P(self.cMV), EX.P(self.cV), EX.P(self.cVF), EX.P(self.sz), np.int32(1 if VFL else 0),
                                      np.float32(h), np.float32(gv[0]), np.float32(gv[1]), np.float32(gv[2]), *PLf, np.float32(a.vfl_margin), EX.P(self.nidx),
                                      EX.P(self.bst)), b=128)
            EX.launch("bsc_tfin", self.Tcap, (EX.P(self.sz), EX.P(self.tcell), EX.P(self.tets), EX.P(self.cM), EX.P(self.cMV), EX.P(self.cV), EX.P(self.cVF),
                                              EX.P(self.VTs), EX.P(self.FT), EX.P(self.mn), EX.P(self.mv)), b=256)
            EX.launch("ex_vfin", self.Mcap, (EX.P(self.sz), np.int32(0), EX.P(self.mn), EX.P(self.mv), EX.P(self.vn), EX.P(None)), b=256)
            return
        EX.launch("bsl_p2g", Np, (np.int32(Np), EX.P(x), EX.P(v), EX.P(F), EX.P(self.mA), EX.P(self.volA), EX.P(self.gpar), EX.P(self.ipar), EX.P(self.ncm),
                                  EX.P(self.cmk), EX.P(self.mn), EX.P(self.mv), EX.P(self.cV), EX.P(self.cVF), EX.P(self.sz), np.int32(1 if VFL else 0),
                                  np.float32(h), np.float32(gv[0]), np.float32(gv[1]), np.float32(gv[2]), *PLf, np.float32(a.vfl_margin), EX.P(self.nidx),
                                  np.int32(BSN), EX.P(self.bst)), b=128)
        EX.launch("ex_vfin", self.Mcap, (EX.P(self.sz), np.int32(0), EX.P(self.mn), EX.P(self.mv), EX.P(self.vn), EX.P(None)), b=256)
        EX.launch("bsl_tfin", self.Tcap, (EX.P(self.sz), EX.P(self.tcell), EX.P(self.tets), EX.P(self.mn), EX.P(self.cV), EX.P(self.cVF),
                                          EX.P(self.VTs), EX.P(self.FT), EX.P(self.bst)), b=256)

    def _sparse(self):
        """절 AD: 프레임 시작 (P2G 뒤, 서브스텝 전) 희소 사면체 판정과 처리 버퍼 (그래프가 읽는 상주 버퍼에 제자리로). 호스트 동기화 없음."""
        T, M = self.T, self.M
        VT = self.VTs[:T]
        occm = VT > 0
        occf = occm.float()
        ar = torch.arange(T, device=dev)
        cnt = torch.zeros(T, device=dev).index_add_(0, self.pinv[:Np], torch.ones(Np, device=dev))   # 사면체 안 입자 수 (--floor vel/none: 모든 입자가 사면체 F̄ 에)
        G3 = self.gl[:T].view(T, 4, 3)[:, 1:, :].double()
        Vg = (1.0 / (6.0 * _det3(G3).abs())).float()                                # 사면체 기하 부피 (Kuhn: 모두 같다)
        occ = VT / Vg
        sp = occm & (cnt <= a.sp_k)
        if a.sp_occ > 0:
            sp = sp | (occm & (occ < a.sp_occ))
        self._spst = [occm.sum(), sp.sum(), (sp & (cnt <= 1)).sum()]
        self._vt0 = VT.sum()                                                          # 바꾸기 전 탄성 부피 합 (stiff/vcap 비 기록용)
        if SPM in ("merge", "blend"):
            tt = self.tets[:T]
            fi = torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], device=dev)
            fn = torch.sort(tt[:, fi], dim=-1).values                               # [T,4,3]
            key = (fn[..., 0] * M + fn[..., 1]) * M + fn[..., 2]
            big = M * M * M + 16
            key = torch.where(occm[:, None], key, big + ar[:, None] * 4 + torch.arange(4, device=dev)[None, :])   # 빈 사면체 면은 짝이 없게
            sk, order = torch.sort(key.reshape(-1))
            tid = order // 4
            eq = sk[1:] == sk[:-1]                                                   # 같은 면을 가진 사면체 짝 (내부 면은 정확히 둘)
            pa, pb = tid[:-1], tid[1:]
        if SPM == "merge":
            score = cnt.long() * (T + 1) + ar                                        # 입자 많은 쪽 (같으면 번호 큰 쪽) -- 점수가 늘기만 해 고리 없음
            best = torch.full((T,), -1, dtype=torch.long, device=dev)
            for u_, w_ in ((pa, pb), (pb, pa)):
                cand = torch.where(eq & sp[u_] & (score[w_] > score[u_]), score[w_], torch.full_like(score[w_], -1))
                best.scatter_reduce_(0, u_, cand, reduce="amax")
            root = torch.where(best >= 0, best % (T + 1), ar)
            for _ in range(14):                                                      # 포인터 건너뛰기 (사슬 깊이 2^14 까지)
                root = root[root]
        elif SPM == "cube":
            cube = self.tcell[:T]
            Pn = self.Pn
            hs_ = torch.zeros(Pn, device=dev).index_add_(0, cube, sp.float())
            mem = occm & (hs_[cube] > 0)
            cmin = torch.full((Pn,), T, dtype=torch.long, device=dev).scatter_reduce_(0, cube, torch.where(occm, ar, torch.full_like(ar, T)), reduce="amin")
            root = torch.where(mem, cmin[cube], ar)
        if SPM in ("merge", "cube") and not SPG:                                     # --sp_a tet: 묶음 F̄·V/n 을 구성 사면체마다 (자기 A_t) -- 기존 힘 커널, F̄ 는 FT 에·V 는 VF 에
            nmem = torch.zeros(T, device=dev).index_add_(0, root, occf)
            VG = torch.zeros(T, device=dev).index_add_(0, root, VT * occf)
            FGs = torch.zeros(T, 9, device=dev).index_add_(0, root, self.FTs[:T] * occf[:, None])
            ing = occm & (nmem[root] > 1.5)
            FGr = FGs[root] / VG[root].clamp_min(1e-30)[:, None]
            self.FT[:T] = torch.where(ing[:, None], FGr, self.FT[:T])
            self.VF[:T] = VT                                                          # V_t 그대로 (처음 판 V_g/n 은 가벼운 노드에 강성이 몰려 발산)
            self._spst += [ing.sum(), (ing & (root == ar)).sum()]
        if SPG:
            nmem = torch.zeros(T, device=dev).index_add_(0, root, occf)
            VG = torch.zeros(T, device=dev).index_add_(0, root, VT * occf)
            FGs = torch.zeros(T, 9, device=dev).index_add_(0, root, self.FTs[:T] * occf[:, None])
            I9 = torch.eye(3, device=dev).reshape(1, 9)
            self.FG[:T] = torch.where(VG[:, None] > 0, FGs / VG.clamp_min(1e-30)[:, None], I9)
            self.VG[:T] = VG
            self.nmem[:T] = nmem.clamp_min(1.0)
            ing = occm & (nmem[root] > 1.5)
            self.grp[:T] = torch.where(ing, root, torch.full_like(root, -1)).int()
            self.AG[0].zero_()
            self.AG[1].zero_()
            self._spst += [ing.sum(), (ing & (root == ar)).sum()]
        elif SPM == "blend":
            num = self.FTs[:T].clone()
            den = VT.clone()
            for u_, w_ in ((pa, pb), (pb, pa)):
                wp = (eq & sp[u_]).float() * a.sp_beta
                num.index_add_(0, u_, wp[:, None] * self.FTs[:T][w_])
                den.index_add_(0, u_, wp * VT[w_])
            self.FT[:T] = torch.where(sp[:, None], num / den.clamp_min(1e-30)[:, None], self.FT[:T])
        elif SPM == "stiff":
            w = (occ / a.sp_wocc).clamp(max=1.0)
            self.VF[:T] = torch.where(sp, VT * w, VT)
            self.VTs[:T] = self.VF[:T]                                                # 응력 부피를 제자리로 (VTs 는 F̄ 나누기 (ex_ffin) 뒤에는 힘 가중·점유 판정 (> 0) 에만 쓰인다 -> 절 C 빠른 경로와 같이 쓸 수 있게)
        elif SPM == "vcap":
            self.VF[:T] = torch.where(occm, torch.minimum(VT, a.sp_vcap * Vg), VT)
            self.VTs[:T] = self.VF[:T]                                                # stiff 와 같이 제자리 (절 C 빠른 경로 겸용)
        elif SPM == "vpmesh":                                                        # 절 AE: 첫 프레임에 입자 부피를 메시 칸 기준으로 다시 정의 (이후 프레임은 그 V_p 로 P2G -- 매 프레임 바꾸지 않는다)
            if not hasattr(self, "_vp_orig"):
                assert self.volA.data_ptr() == vol.data_ptr(), "vpmesh: lag 판 (입자 정렬 없음) 만"
                pv = self.pinv[:Np]
                vnew = (Vg[pv] / cnt[pv].clamp_min(1.0)).to(vol.dtype)
                self._vp_orig = vol.double().sum()
                print(f"[절 AE vpmesh] 입자 부피 다시 정의: Σ V_p (MPM 칸) {float(self._vp_orig):.6g} -> Σ V_p' (메시 칸) {float(vnew.double().sum()):.6g} "
                      f"(비 {float(vnew.double().sum() / self._vp_orig):.4f}), V_p'/V_p 분위 10·50·90 % "
                      f"{[round(q_, 3) for q_ in torch.quantile((vnew / vol).float()[:min(Np, 10_000_000)], torch.tensor([0.1, 0.5, 0.9], device=dev)).tolist()]}", flush=True)
                if a.sp_vpm_mass:
                    assert self.mA.data_ptr() == m.data_ptr()
                    m0 = m.double().sum()
                    m.mul_(vnew / vol)                                                    # m_p' = ρ_p V_p' (ρ_p = m_p / V_p, 영역 밀도 그대로)
                    print(f"[절 AE vpmesh] 질량도 m_p' = ρ_p V_p': Σ m {float(m0):.6g} -> {float(m.double().sum()):.6g} kg", flush=True)
                vol.copy_(vnew)
                self._vt0 = self._vp_orig
                return True                                                               # 첫 프레임 P2G 를 새 V_p (·m_p) 로 다시 (부르는 쪽)
            self._vt0 = self._vp_orig
        elif SPM == "mlow":
            mn = self.mn[:M]
            if not hasattr(self, "mlow_v"):
                self.mlow_v = float(a.sp_mlow * mn[mn > 0].median())
                print(f"[절 AD mlow] 관성 질량 하한 {self.mlow_v:.4g} kg (= {a.sp_mlow:g} x 첫 프레임 노드 질량 중앙)", flush=True)
            self.mI[:M] = torch.where(mn > 0, mn.clamp_min(self.mlow_v), mn)
        if not hasattr(self, "_spinfo"):
            self._spinfo = True
            q = torch.quantile(occ[occm], torch.tensor([0.01, 0.1, 0.5, 0.9], device=dev)).tolist()
            qs = torch.quantile(occ[sp], torch.tensor([0.1, 0.5, 0.9], device=dev)).tolist() if bool(sp.any()) else []
            print(f"[절 AD] 첫 프레임: 점유 사면체 {int(occm.sum())}, 희소 {int(sp.sum())} (입자 1 개 {int((sp & (cnt <= 1)).sum())}), 점유율 분위 1·10·50·90 % "
                  f"{[round(v_, 3) for v_ in q]}, 희소 점유율 10·50·90 % {[round(v_, 3) for v_ in qs]}, 기하 부피 {float(Vg[0]):.4g}"
                  + (f", 묶인 사면체 {int(self._spst[3])} 묶음 {int(self._spst[4])}" if len(self._spst) > 3 else ""), flush=True)

    def sp_check(self):
        """절 AD 검산 (측정 밖, 프레임 끝 U): 내부력 합·돌림힘 (운동량·각운동량 보존), mlow 는 Σ (m/m') f."""
        M, Tc = self.M, self.Tcap
        self.f.zero_()
        if SPG:
            self.AG[0].zero_()
            EX.launch("sp_force_g1", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT), EX.P(self.VTs), np.float32(mu), np.float32(lam),
                                          EX.P(self.f), EX.P(self.grp), EX.P(self.AG[0]), EX.P(None)))
            EX.launch("sp_force_g2", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.gl), np.float32(mu), np.float32(lam), EX.P(self.f), EX.P(self.grp),
                                          EX.P(self.AG[0]), EX.P(None), EX.P(self.FG), EX.P(self.VG), EX.P(self.VTs), EX.P(self.Uc), np.float32(a.sp_hg)))
            self.AG[0].zero_()
        else:
            EX.launch("ex_tet_force_R", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT), EX.P(self.VF if (SPM in ("stiff", "vcap") or (SPM in ("merge", "cube") and not SPG)) else self.VTs),
                                             np.float32(mu), np.float32(lam), EX.P(self.f), np.float32(0), np.float32(0.3), np.int32(0), EX.P(None)))
        f = self.f[:M].double()
        xn = (self.npos + self.Uc[:M]).double()
        c = xn.mean(0)
        r = xn - c
        fa = f.norm(dim=-1)
        out = dict(F_sum=float(f.sum(0).norm() / fa.sum().clamp_min(1e-300)), tq=float(torch.cross(r, f, dim=-1).sum(0).norm() / (r.norm(dim=-1) * fa).sum().clamp_min(1e-300)),
                   f_abs=float(fa.sum()))
        if SPM == "mlow":
            mn, mI = self.mn[:M].double(), self.mI[:M].double()
            fr = torch.where(mI > 0, mn / mI.clamp_min(1e-30), torch.ones_like(mn))[:, None] * f
            out.update(P_rate=float(fr.sum(0).norm() / fa.sum().clamp_min(1e-300)), n_low=int((mI > mn).sum()))
        self.f.zero_()
        return out

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

    def _ag_frame(self, i):
        """절 AG (프레임마다, 그래프 밖): 셀 소성 상태 (입자 부피 가중 평균 -- 손상 입자 빼고) 와 입자 속도 구속의 노드 몫 α·α v̄_bc."""
        T, M = self.T, self.M
        if PLAST:
            if getattr(self, "_agT", -1) != self.Tcap:
                self.ysT = torch.zeros(self.Tcap, device=dev)
                self.ljT = torch.zeros(self.Tcap, device=dev)
                if PLINC:
                    self.FeC = torch.zeros(self.Tcap, 9, device=dev)
                    self.XpC = torch.zeros(self.Tcap, 9, device=dev)
                    self.ysC = torch.zeros(self.Tcap, device=dev)
                    self.ljC = torch.zeros(self.Tcap, device=dev)
                self._agT = self.Tcap
                self.graph, self.graphs = None, {}
            pv_ = self.pinv[:Np]
            w_ = self.volA
            den = torch.zeros(T, device=dev).index_add_(0, pv_, w_).clamp_min(1e-30)
            if MATN in (1, 3, 5):
                self.ysT[:T] = torch.zeros(T, device=dev).index_add_(0, pv_, w_ * PL_YS) / den
            if MATN == 7:
                self.ljT[:T] = torch.zeros(T, device=dev).index_add_(0, pv_, w_ * PL_LJ) / den
        if PBC:
            if getattr(self, "_agM", -1) != self.Mcap:
                self.pa = torch.zeros(self.Mcap, device=dev)
                self.pv = torch.zeros(self.Mcap, 3, device=dev)
                self._agM = self.Mcap
                self.graph, self.graphs = None, {}
            k_ = PBC_K[i].long()
            c_ = (k_ >= 0).float()
            vb_ = PBC_V[k_.clamp_min(0)] * c_[:, None]
            W4 = (torch.full_like(self.lamb, 0.25) if a.mass == "tet" else self.lamb) * (m * c_)[:, None]          # P2G 질량 가중과 같은 몫
            r_ = self.rows.reshape(-1)
            acc_ = torch.zeros(M, device=dev).index_add_(0, r_, W4.reshape(-1))
            mvb_ = torch.zeros(M, 3, device=dev).index_add_(0, r_, (W4[..., None] * vb_[:, None, :]).reshape(-1, 3))
            mn_ = self.mn[:M].clamp_min(1e-30)
            self.pa.zero_()
            self.pv.zero_()
            self.pa[:M] = (acc_ / mn_).clamp(max=1.0)
            self.pv[:M] = mvb_ / mn_[:, None]

    def _step_fused(self):
        """절 C (5): ex_tet_force_R + ex_vfl_contact + ex_node_step_lag + 클램프 판정 + 적용과 같은 식을 커널 4 개로."""
        Mc, Tc = self.Mcap, self.Tcap
        nbT = (Tc + 127) // 128
        nbP = ((Np + 127) // 128) if VFL else 0
        hs_, damp_ = self.hscur, self.dampcur
        if PLAST or VISC:                                                           # 절 AG: 같은 틀, 사면체 블록만 재질 식 (셀 투영) + 점성
            PLC.launch("ag_fc", 0, (np.int32(nbT), EX.P(self.sz), EX.P(self.tl), EX.P(self.cint[2:]), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT),
                                    EX.P(self.VTs), np.float32(mu), np.float32(lam), EX.P(self.f), EX.P(self.rst), np.int32(1 if a.fpolar else 0),
                                    np.int32(MATN), EX.P(PRM), EX.P(self.ysT if PLAST else None), EX.P(self.ljT if PLAST else None),
                                    np.float32(hs_ if PLINC else (getattr(self, "_j", 0) + 1) * hs_), EX.P(self.vn if VISC else None), np.float32(a.visc),
                                    np.int32(a.pl_diag | (1 if a.pl_cell == "none" else 0)),
                                    EX.P(self.FeC if PLINC else None), EX.P(self.XpC if PLINC else None), EX.P(self.ysC if PLINC else None), EX.P(self.ljC if PLINC else None),
                                    EX.P(self.s0c), EX.P(self.cint[1:]), EX.P(self.nidx), EX.P(self.Xf if VFL else None), EX.P(self.rows), EX.P(self.lamb), EX.P(self.mA),
                                    np.int32(1 if a.mass == "tet" else 0), *PLf, EX.P(self.acc), EX.P(self.ncon)), b=128, grid=nbT + nbP)
        elif AHF:                                                                   # 절 AH: 같은 힘 + 바닥 접촉 가중 (gtk) / 실제 접촉 표시
            AHC.launch("ah_fc", 0, (np.int32(nbT), EX.P(self.sz), EX.P(self.tl), EX.P(self.cint[2:]), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT),
                                    EX.P(self.VTs), np.float32(mu), np.float32(lam), EX.P(self.f), EX.P(self.rst), np.int32(1 if a.fpolar else 0), EX.P(self.eF if a.sr_ef else None),
                                    EX.P(self.s0c), EX.P(self.cint[1:]), EX.P(self.nidx), EX.P(self.Xf if VFL else None), EX.P(self.rows), EX.P(self.lamb), EX.P(self.mA),
                                    np.int32(1 if a.mass == "tet" else 0), *PLf, EX.P(self.acc), EX.P(self.ncon),
                                    np.int32(AH_GTK), np.float32(AH_BAND), np.float32(AH_PN), np.float32(AH_DXG), EX.P(self.accN if AH_GTK else None)), b=128, grid=nbT + nbP)
        else:
            CCU.launch("c_fc", 0, (np.int32(nbT), EX.P(self.sz), EX.P(self.tl), EX.P(self.cint[2:]), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT),
                                   EX.P(self.VTs), np.float32(mu), np.float32(lam), EX.P(self.f), EX.P(self.rst), np.int32(1 if a.fpolar else 0), EX.P(self.eF if a.sr_ef else None),
                                   EX.P(self.s0c), EX.P(self.cint[1:]), EX.P(self.nidx), EX.P(self.Xf if VFL else None), EX.P(self.rows), EX.P(self.lamb), EX.P(self.mA),
                                   np.int32(1 if a.mass == "tet" else 0), *PLf, EX.P(self.acc), EX.P(self.ncon)), b=128, grid=nbT + nbP)
        if a.cfuse_mode == 2 and AHF:
            AHC.launch("ah_node_step", Mc, (EX.P(self.sz), EX.P(self.mn), EX.P(self.f), EX.P(self.vn), np.float32(gv[0]), np.float32(gv[1]), np.float32(gv[2]),
                                            np.float32(hs_), np.float32(damp_), EX.P(self.fx if self.fixd else None), EX.P(self.vbc), EX.P(self.acc),
                                            EX.P(self.accN if AH_GTK else None), np.int32(VFM), np.int32(a.vfl_np), *PLf[3:], np.float32(AH_KEX * hs_ / hs)), b=256)
        elif a.cfuse_mode == 2:
            EX.launch("ex_node_step_lag", Mc, (EX.P(self.sz), EX.P(self.mI if SPM == "mlow" else self.mn), EX.P(self.f), EX.P(None), EX.P(self.vn), EX.P(self.Uc),
                                               np.float32(gv[0]), np.float32(gv[1]), np.float32(gv[2]), np.float32(hs_), np.float32(damp_),
                                               EX.P(self.fx if self.fixd else None), EX.P(self.vbc), EX.P(self.acc if VFL else None),
                                               np.int32(VFM), np.int32(a.vfl_np), *PLf[3:], np.int32(1)), b=256)
            if PBC:                                                                 # 절 AG: 입자 속도 구속의 노드 몫 (노드 걸음 뒤, 클램프·이동 앞)
                PLC.launch("ag_pbc", Mc, (EX.P(self.sz), EX.P(self.vn), EX.P(self.pa), EX.P(self.pv)), b=256)
            jon = np.int32(1 if a.lim >= 0 else 0)
            CCU.launch("c_sr_pre", Tc, (EX.P(self.sz), EX.P(self.tl), EX.P(self.cint[2:]), EX.P(self.tets), EX.P(self.gl), EX.P(self.VTs), EX.P(self.vn), EX.P(self.Uc),
                                        np.float32(hs_), np.float32(a.lim_sr), jon, np.float32(a.lim), EX.P(self.clst), EX.P(self.eF if a.sr_ef else None)), b=128)
            CCU.launch("c_sr_warp", 0, (EX.P(self.clst), EX.P(self.tets), EX.P(self.gl), EX.P(self.vn), EX.P(self.Uc), np.float32(hs_), np.float32(a.lim_sr),
                                        jon, np.float32(a.lim), EX.P(self.alph), EX.P(self.rst)), b=128, grid=CCU.NBW)
            CCU.launch("c_lim_apply", Mc, (EX.P(self.sz), EX.P(self.mn), EX.P(self.vn), EX.P(self.Uc), np.float32(hs_), EX.P(self.alph),
                                           np.int32(1 if a.lim_v == "keep" else 0), EX.P(self.fx if self.fixd else None), EX.P(self.rst),
                                           EX.P(self.clst), EX.P(self.cint[1:] if VFL else None), *PLf[3:]), b=256)
            return
        S = CCU.cvs(self.mn, self.f, None, self.vn, gv, hs_, damp_, self.fx if self.fixd else None, self.vbc, self.acc if VFL else None,
                    VFM, a.vfl_np, PLf[3:])
        jon = np.int32(1 if a.lim >= 0 else 0)
        CCU.launch("c_sr_pre2", Tc, (EX.P(self.sz), EX.P(self.tl), EX.P(self.cint[2:]), EX.P(self.tets), EX.P(self.gl), EX.P(self.VTs), EX.P(self.Uc), S,
                                     np.float32(a.lim_sr), jon, np.float32(a.lim), EX.P(self.clst)), b=128)
        CCU.launch("c_sr_warp2", 0, (EX.P(self.clst), EX.P(self.tets), EX.P(self.gl), EX.P(self.Uc), S, np.float32(a.lim_sr), jon, np.float32(a.lim),
                                     EX.P(self.alph), EX.P(self.rst)), b=128, grid=CCU.NBW)
        CCU.launch("c_apply2", Mc, (EX.P(self.sz), S, EX.P(self.vn), EX.P(self.f), EX.P(None), EX.P(self.acc if VFL else None), EX.P(self.Uc), EX.P(self.alph),
                                    np.int32(1 if a.lim_v == "keep" else 0), EX.P(self.rst), EX.P(self.clst), EX.P(self.cint[1:] if VFL else None)), b=256)

    def _step_lag(self, xcur=None):
        """lag 한 서브스텝 (절 O 판): 사면체 힘 -> [에너지 바닥 후보 힘] -> [속도 바닥 접촉 판정] -> 노드 걸음 (+ 고착·비관통, f·gn·acc 비움)."""
        if CC and a.cfuse:
            return self._step_fused()
        Mc, Tc = self.Mcap, self.Tcap
        if NINT:                                                                    # 절 V: 노드 적분 내부력 (커널 3 개)
            self._nint_force()
        elif CQ:                                                                    # 절 AA: 2 차 근사 셀 에너지 힘
            EX.launch("ex_tet_force_CQ", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT), EX.P(self.VTs), EX.P(self.CS),
                                              np.float32(mu), np.float32(lam), EX.P(self.f), EX.P(self.rst), np.int32(0 if a.cq_h == "km" else 1), EX.P(None)))
        elif SB:                                                                    # 절 Z (2): ex_tet_force_R + 비율 장벽
            EX.launch("ex_tet_force_SR", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT), EX.P(self.VTs),
                                              np.float32(mu), np.float32(lam), EX.P(self.f), np.float32(a.fb), np.float32(a.fb_jhat),
                                              np.int32(1 if a.fb_on == "A" else 0), EX.P(self.rst), np.float32(a.sb), np.float32(a.sb_c), np.float32(SB_D)))
        elif SPG:                                                                   # 절 AD merge/cube: 홀로 사면체 힘 + 묶음 A 합 -> 묶음 응력·분배 (+ 다음 벌 비움)
            j_ = getattr(self, "_j", 0)
            EX.launch("sp_force_g1", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT), EX.P(self.VTs), np.float32(mu), np.float32(lam),
                                          EX.P(self.f), EX.P(self.grp), EX.P(self.AG[j_ % 2]), EX.P(self.rst)))
            EX.launch("sp_force_g2", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.gl), np.float32(mu), np.float32(lam), EX.P(self.f), EX.P(self.grp),
                                          EX.P(self.AG[j_ % 2]), EX.P(self.AG[(j_ + 1) % 2]), EX.P(self.FG), EX.P(self.VG), EX.P(self.VTs), EX.P(self.Uc), np.float32(a.sp_hg)))
        elif CC:                                                                    # 절 C: 같은 식 (점유 목록·빠른 극분해 선택)
            CCU.launch("c_tet_force_R", Tc, (EX.P(self.sz), EX.P(self.tl if a.tlist else None), EX.P(self.cint[2:] if a.tlist else None), EX.P(self.tets),
                                             EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT), EX.P(self.VTs), np.float32(mu), np.float32(lam), EX.P(self.f),
                                             np.float32(a.fb), np.float32(a.fb_jhat), np.int32(1 if a.fb_on == "A" else 0), EX.P(self.rst), np.int32(1 if a.fpolar else 0),
                                             EX.P(self.eF if a.sr_ef else None)))
        elif RNEW:
            EX.launch("ex_tet_force_R", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT), EX.P(self.VF if (SPM in ("stiff", "vcap") or (SPM in ("merge", "cube") and not SPG)) else self.VTs),
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
        if VFL and BSCM:                                                             # 절 W: 접촉 질량도 셀 -> 사면체 -> 노드 /4 (노드 질량과 같은 경로)
            EX.launch("bsc_vfl_contact", Np, (EX.P(self.sz), EX.P(self.nidx), EX.P(self.Xf if xcur is None else xcur), EX.P(self.Uc), EX.P(self.rows), EX.P(self.lamb),
                                              EX.P(self.gpar), EX.P(self.ipar), EX.P(self.ncm), EX.P(self.cmk), EX.P(self.mA), np.int32(1 if G2PBS else 0), *PLf,
                                              EX.P(self.accC), EX.P(self.ncon)), b=128)
            EX.launch("bsc_acc_tet", Tc, (EX.P(self.sz), EX.P(self.tcell), EX.P(self.tets), EX.P(self.accC), EX.P(self.acc)), b=256)
            self.accC.zero_()
        elif VFL and BSL:
            EX.launch("bsl_vfl_contact", Np, (EX.P(self.sz), EX.P(self.nidx), EX.P(self.Xf if xcur is None else xcur), EX.P(self.Uc), EX.P(self.rows), EX.P(self.lamb),
                                              EX.P(self.gpar), EX.P(self.ipar), EX.P(self.ncm), EX.P(self.mA), np.int32(1 if G2PBS else 0), *PLf,
                                              EX.P(self.acc), EX.P(self.ncon), np.int32(BSN)), b=128)
        elif VFL and a.vfl_cull:                                                    # 절 C (1): s0 > D 후보 생략
            if a.cchk:                                                              # 검산: 기존 판정을 따로 (acc_ref·ncon_ref) 돌린다
                self.acc_ref.zero_()
                self.ncon_ref.zero_()
                EX.launch("ex_vfl_contact", Np, (EX.P(self.sz), EX.P(self.nidx), EX.P(self.Xf if xcur is None else xcur), EX.P(self.Uc), EX.P(self.rows), EX.P(self.lamb), EX.P(self.mA),
                                                 np.int32(1 if a.mass == "tet" else 0), *PLf, EX.P(self.acc_ref), EX.P(self.ncon_ref)), b=128)
                n0_ = self.ncon.clone()
            CCU.launch("c_vfl_contact", Np, (EX.P(self.sz), EX.P(self.s0c), EX.P(self.cint[1:]), EX.P(self.nidx), EX.P(self.Xf if xcur is None else xcur), EX.P(self.Uc),
                                             EX.P(self.rows), EX.P(self.lamb), EX.P(self.mA), np.int32(1 if a.mass == "tet" else 0), *PLf, EX.P(self.acc), EX.P(self.ncon)), b=128)
            if a.cchk:
                M_ = self.M
                self.chk["ncon_diff"] += (self.ncon - n0_ - self.ncon_ref).abs().sum()
                self.chk["ncon_ref"] += self.ncon_ref.sum()
                self.chk["acc_set_diff"] += ((self.acc[:M_] > 0) != (self.acc_ref[:M_] > 0)).sum()
                self.chk["acc_absdiff"] += (self.acc[:M_] - self.acc_ref[:M_]).abs().sum()
                self.chk["acc_sum"] += self.acc_ref[:M_].sum()
        elif VFL:
            EX.launch("ex_vfl_contact", Np, (EX.P(self.sz), EX.P(self.nidx), EX.P(self.Xf if xcur is None else xcur), EX.P(self.Uc), EX.P(self.rows), EX.P(self.lamb), EX.P(self.mA),
                                             np.int32(1 if a.mass == "tet" else 0), *PLf, EX.P(self.acc), EX.P(self.ncon)), b=128)
        if self._fhook is not None:                                                 # 진단 읽기만 (걸음이 f 를 비우기 전)
            self._fhook()
        EX.launch("ex_node_step_lag", Mc, (EX.P(self.sz), EX.P(self.mI if SPM == "mlow" else self.mn), EX.P(self.f), EX.P(self.gn if FLOOR else None), EX.P(self.vn), EX.P(self.Uc),
                                           np.float32(gv[0]), np.float32(gv[1]), np.float32(gv[2]), np.float32(hs), np.float32(DAMP),
                                           EX.P(self.fx if self.fixd else None), EX.P(self.vbc), EX.P(self.acc if VFL else None),
                                           np.int32(VFM), np.int32(a.vfl_np), *PLf[3:], np.int32(1 if LIM else 0)), b=256)
        if CC:                                                                      # 절 C: 판정 (빠른 보장 + 목록 워프 / 기존 커널) -> 적용 (목록 비움, D 갱신)
            if SRL and a.sr_fast and a.cchk:                                        # 검산: 기존 커널 α 를 따로
                self.alph_ref.fill_(1065353216)
                EX.launch("ex_lim_sr_tet", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.gl), EX.P(self.VTs), EX.P(self.vn), EX.P(self.Uc), np.float32(hs),
                                                np.float32(a.lim_sr), np.int32(1 if a.lim >= 0 else 0), np.float32(a.lim), EX.P(self.alph_ref), EX.P(self.rst_ref)))
            if SRL and a.sr_fast:
                CCU.launch("c_sr_pre", Tc, (EX.P(self.sz), EX.P(self.tl if a.tlist else None), EX.P(self.cint[2:] if a.tlist else None), EX.P(self.tets), EX.P(self.gl),
                                            EX.P(self.VTs), EX.P(self.vn), EX.P(self.Uc), np.float32(hs), np.float32(a.lim_sr), np.int32(1 if a.lim >= 0 else 0),
                                            np.float32(a.lim), EX.P(self.clst), EX.P(self.eF if a.sr_ef else None)), b=128)
                CCU.launch("c_sr_warp", 0, (EX.P(self.clst), EX.P(self.tets), EX.P(self.gl), EX.P(self.vn), EX.P(self.Uc), np.float32(hs), np.float32(a.lim_sr),
                                            np.int32(1 if a.lim >= 0 else 0), np.float32(a.lim), EX.P(self.alph), EX.P(self.rst)), b=128, grid=CCU.NBW)
                if a.cchk:
                    M_ = self.M
                    self.chk["alpha_bit_diff"] += (self.alph[:M_] != self.alph_ref[:M_]).sum()
                    self.chk["alpha_lim_nodes"] += (self.alph_ref[:M_] != 1065353216).sum()
                    self.chk["list_len"] += self.clst[0]
                    self.chk["occ_tets"] += (self.VTs[:self.T] > 0).sum()
            elif SRL:
                EX.launch("ex_lim_sr_tet", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.gl), EX.P(self.VTs), EX.P(self.vn), EX.P(self.Uc), np.float32(hs),
                                                np.float32(a.lim_sr), np.int32(1 if a.lim >= 0 else 0), np.float32(a.lim), EX.P(self.alph), EX.P(self.rst)))
            else:
                EX.launch("ex_lim_box_tet", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.gl), EX.P(self.VTs), EX.P(self.vn), EX.P(self.Uc), np.float32(hs),
                                                 np.float32(a.lim), EX.P(self.alph), EX.P(self.rst)))
            CCU.launch("c_lim_apply", Mc, (EX.P(self.sz), EX.P(self.mn), EX.P(self.vn), EX.P(self.Uc), np.float32(hs), EX.P(self.alph),
                                           np.int32(1 if a.lim_v == "keep" else 0), EX.P(self.fx if self.fixd else None), EX.P(self.rst),
                                           EX.P(self.clst if (SRL and a.sr_fast) else None), EX.P(self.cint[1:] if (VFL and a.vfl_cull) else None), *PLf[3:]), b=256)
            if a.lim_sr_chk:
                EX.launch("ex_sr_viol", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.gl), EX.P(self.VTs), EX.P(self.Uc), np.float32(a.lim_sr), EX.P(self.rst)))
        elif SRL:                                                                   # 절 Z (1): 비율 (+ det) 클램프 -> 같은 상자 판 적용 커널
            EX.launch("ex_lim_sr_tet", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.gl), EX.P(self.VTs), EX.P(self.vn), EX.P(self.Uc), np.float32(hs),
                                            np.float32(a.lim_sr), np.int32(1 if a.lim >= 0 else 0), np.float32(a.lim), EX.P(self.alph), EX.P(self.rst)))
            EX.launch("ex_lim_box_apply", Mc, (EX.P(self.sz), EX.P(self.mn), EX.P(self.vn), EX.P(self.Uc), np.float32(hs), EX.P(self.alph),
                                               np.int32(1 if a.lim_v == "keep" else 0), EX.P(self.fx if self.fixd else None), EX.P(self.rst)), b=256)
            if a.lim_sr_chk:
                EX.launch("ex_sr_viol", Tc, (EX.P(self.sz), EX.P(self.tets), EX.P(self.gl), EX.P(self.VTs), EX.P(self.Uc), np.float32(a.lim_sr), EX.P(self.rst)))
        elif LIM and a.lim_mode == "box":
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
                if BSCM:
                    self.accC.zero_()
            if PLINC:                                                               # 절 AG: 셀 증분 상태를 프레임 시작 값으로 (그래프 안 -- 데우기가 바꾼 것도 되돌린다)
                PLC.launch("ag_cinit", self.Tcap, (EX.P(self.sz), EX.P(self.FT), EX.P(self.ysT), EX.P(self.ljT), EX.P(self.FeC), EX.P(self.XpC),
                                                   EX.P(self.ysC), EX.P(self.ljC)), b=256)
            if not REP:
                for j in range(self.Ncur):
                    self._j = j
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
        self.graphs[self.Ncur] = g
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
            wc_ = (torch.full_like(self.lamb[ni_][c_], 0.25) if a.mass == "tet" else self.lamb[ni_][c_]) * self.mA[ni_][c_][:, None]   # 접촉 분배 = P2G 질량 가중 (ex_vfl_contact 와 같은 식)
            accn = torch.zeros(M, device=dev).index_add_(0, r_[c_].reshape(-1), wc_.reshape(-1))
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
        LSTEP = VFL or RNEW                                                         # 절 X: 바닥 없는 절 R 경로 (ficus --rstat) 도 측정 판과 같은 _step_lag 로 (덤프 f 포함)
        if LSTEP:                                                                   # 절 O 판: 같은 서브스텝 커널 (_step_lag) -- f 는 걸음 안에서 비워져 |f/m| 표본은 0
            self.f.zero_()
            self.gn.zero_()
            if VFL:
                self.acc.zero_()
        for j in range(a.N):
            if LD:
                self._fj = j
                dAl_[j] = contact_alpha()                                           # 이 서브스텝 접촉 판정 (서브스텝 시작 U 기준) 의 α
            if LSTEP:
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
                                   mass_def=("bspline" if BSL else a.mass), g2p=(a.g2p if BSL else "bary"),
                                   pinv=cpu(self.pinv[:Np]).int(), mp=cpu(m)))                   # 절 X: 입자 -> 사면체, 입자 질량 (적분점·질량점 검산용)
        with open(os.path.join(a.out, "lagdiag.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"    [lag 진단] t {i + 1}: 최대|v_n| 표본 " + " ".join(f"{q[1]:.3g}" for q in smp) + f" (마지막: 노드 높이-바닥 {smp[-1][2]:+.4f}, 질량/중앙 {smp[-1][3]:.3g}), "
              f"최대 |f/m| {smp[-1][4]:.3g}, 최대 |바닥 g/m| {smp[-1][5]:.3g}, det A_T 최소 {rec['detA_min']}, <0.5 {rec['detA_lt05']}, ≤0 {rec['detA_le0']}, "
              f"det F̄ 최소 {rec['detFT_min']}, σ(F̄) 최대 {rec['svFT_max']:.3g} (99.9% {rec['svFT_q999']:.3g}) 최소 {rec['svFT_min']:.3g}, "
              f"[메시] 질량 중앙 {mstat['m_med']:.3g}, 가벼운 (<2.5%) {mstat['n_light']}/{mstat['n_live']} (점유율 중앙 {mstat['occ_light_med']}, 전체 {mstat['occ_all_med']:.3g}), "
              f"r_n 최대 {mstat['r_max']:.3g} (99.9% {mstat['r_q999']:.3g}, 가벼운 {mstat['r_light_max']}), 접촉 노드 {mstat['n_contact_nodes']} (가벼운 {mstat['n_contact_light']}), "
              f"최대|v| 노드 [j,|v|,m/중앙,점유,r,α] {msmp[-1][:6]} "
              f"σ(A F̄) 최대 {rec['svAF_max']:.3g} 최소 {rec['svAF_min']:.3g}, 입자 F σ 최대 {rec['svFp_max']:.3g} (99.9% {rec['svFp_q999']:.3g}) 최소 {rec['svFp_min']:.3g}, 첫 비유한 서브스텝 {first_bad}, KE {ke:.4g}, 최대|U| {rec['Umax']:.4g}", flush=True)

    def _fhist(self, i, v0):
        """--fhist (읽기만): 이 프레임 시작 입자 상태·셀 상태·프레임 끝 노드 U (진단 루프 뒤, G2P 앞) -> <out>/fhist/fNNN.pt."""
        M, T = self.M, self.T
        cpu = lambda q: q.detach().cpu()
        od = os.path.join(a.out, "fhist")
        os.makedirs(od, exist_ok=True)
        torch.save(dict(t=i, x=cpu(self._xp).float(), v=cpu(v0).float(), F=cpu(self._Fp).float().reshape(Np, 9), pinv=cpu(self.pinv[:Np]).int(),
                        nidx=cpu(self.nidx[:int(self.sz[2])]).int(), tets=cpu(self.tets[:T]).int(), npos=cpu(self.npos), gl=cpu(self.gl[:T]),
                        FT=cpu(self.FT[:T]), VT=cpu(self.VTs[:T]), mn=cpu(self.mn[:M]), Uend=cpu(self.Uc[:M]), Vend=cpu(self.vn[:M]), Ai=cpu(self.Ai),
                        plane=(pt_, nr_) if HASPL else None, mu=float(mu), lam=float(lam)), os.path.join(od, f"f{i:03d}.pt"))

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
        if BSCM:                                                                    # 절 W: 비교용 절 T 노드 커널 질량 (같은 입자 배치, 같은 메시·정규화, 버리는 버퍼)
            mT_ = torch.zeros(self.Mcap, device=dev)
            mvT_ = torch.zeros(self.Mcap, 3, device=dev)
            cV_, cVF_ = torch.zeros(self.Pn, device=dev), torch.zeros(self.Pn * 9, device=dev)
            sz_ = self.sz.clone()
            xs_ = self._xs_diag
            EX.launch("bsl_p2g", Np, (np.int32(Np), EX.P(xs_[0]), EX.P(xs_[1]), EX.P(xs_[2]), EX.P(self.mA), EX.P(self.volA), EX.P(self.gpar), EX.P(self.ipar), EX.P(self.ncm),
                                      EX.P(self.cmk), EX.P(mT_), EX.P(mvT_), EX.P(cV_), EX.P(cVF_), EX.P(sz_), np.int32(0),
                                      np.float32(h), np.float32(gv[0]), np.float32(gv[1]), np.float32(gv[2]), *PLf, np.float32(a.vfl_margin), EX.P(self.nidx),
                                      np.int32(1), EX.P(None)), b=128)
            mTd = mT_[:M].double()
            rec.update(m_bsT=dist(mTd), stab_T=stab(VTn, mTd), m_cell=float(self.cM[:self.Pn].double().sum()),
                       P_cell=self.cMV[:self.Pn * 3].view(-1, 3).double().sum(0).tolist(),
                       cV_vs_T=float((cV_.double().sum() - VTn.sum() * 1.0).abs()))
            # 노드별 질량 비 새 / 절 R (m_T/4), 새 / 절 T
            both = (mb > 0) & (mt4 > 0)
            rr = (mb[both] / mt4[both]).float()
            rec["ratio_new_R_q"] = torch.quantile(rr, torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], device=dev)).tolist()
            bt = (mb > 0) & (mTd > 0)
            rt = (mb[bt] / mTd[bt]).float()
            rec["ratio_new_T_q"] = torch.quantile(rt, torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], device=dev)).tolist()
        # 새 정의에서 가장 가벼운 노드들 (상대 질량 < 2.5 %) 의 이웃 셀 부피 / 질량 비 vs 중앙
        with open(os.path.join(a.out, "bsdiag.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        mbd, mtd, mld, sn, so = rec["m_bs"], rec["m_t4"], rec["m_lam"], rec["stab_new"], rec["stab_old"]
        fq = lambda d_: "/".join(f"{q:.2g}" for q in d_["q_rel"])
        if BSCM:
            mT2, sT = rec["m_bsT"], rec["stab_T"]
            print(f"    [W 진단] t {i + 1}: 노드 질량 /중앙 분위 (0,1e-4,1e-3,.01,.1,.5) 새 (셀 커널 m_T/4) {fq(mbd)} (<2.5% {mbd['n_lt025']}/{mbd['n_live']}) | "
                  f"절 T (노드 커널) {fq(mT2)} (<2.5% {mT2['n_lt025']}/{mT2['n_live']}) | 절 R m_T/4 {fq(mtd)} (<2.5% {mtd['n_lt025']}/{mtd['n_live']}); "
                  f"총질량 입자 {rec['m_p']:.7g} 셀 {rec['m_cell']:.7g} 노드 {mbd['tot']:.7g}; 셀 운동량 {[round(q, 4) for q in rec['P_cell']]}; "
                  f"새/절 R 질량 비 분위 (0,.01,.5,.99,1) {'/'.join(f'{q:.3g}' for q in rec['ratio_new_R_q'])}, 새/절 T {'/'.join(f'{q:.3g}' for q in rec['ratio_new_T_q'])}; "
                  f"국소 안정 r_n 중앙/99.9%/최대 새 {sn['r_q50']:.3g}/{sn['r_q999']:.3g}/{sn['r_max']:.3g} (필요 N ≈ {sn['N_need']:.0f}, r>1 {sn['n_r_gt1']}) "
                  f"절 T {sT['r_q50']:.3g}/{sT['r_q999']:.3g}/{sT['r_max']:.3g} (필요 N ≈ {sT['N_need']:.0f}) "
                  f"절 R {so['r_q50']:.3g}/{so['r_q999']:.3g}/{so['r_max']:.3g} (필요 N ≈ {so['N_need']:.0f})", flush=True)
            return
        print(f"    [S 진단] t {i + 1}: 노드 질량 /중앙 분위 (0,1e-4,1e-3,.01,.1,.5) B-스플라인 {fq(mbd)} (<2.5% {mbd['n_lt025']}/{mbd['n_live']}) | "
              f"m_T/4 {fq(mtd)} (<2.5% {mtd['n_lt025']}/{mtd['n_live']}) | Σλm {fq(mld)} (<2.5% {mld['n_lt025']}/{mld['n_live']}); "
              f"총질량 입자 {rec['m_p']:.6g} 노드 B {mbd['tot']:.6g} (정규화로 옮긴 질량 {rec['m_redist']:.4g}); 부피 입자 {rec['V_p']:.6g} 사면체 {rec['V_new']:.6g} (뺀 {rec['V_drop']:.3g}, 사면체 {rec['T_drop']}, 정규화로 옮긴 부피 {rec['V_redist']:.4g}); "
              f"국소 안정 r_n (h_s ω/2) 중앙/99.9%/최대 새 {sn['r_q50']:.3g}/{sn['r_q999']:.3g}/{sn['r_max']:.3g} (필요 N ≈ {sn['N_need']:.0f}, r>1 {sn['n_r_gt1']}) "
              f"절 R 정의 {so['r_q50']:.3g}/{so['r_q999']:.3g}/{so['r_max']:.3g} (필요 N ≈ {so['N_need']:.0f})", flush=True)

    def _choose_N(self, i):
        """절 C 적응 서브스텝 -- self.Ncur·hscur·dampcur 를 정한다 (동기화 1 번, 측정 안)."""
        if a.adapt == "act":
            Nf = a.ad_lo if (i > 0 and float(self.lim_prev) <= a.ad_thr) else a.ad_hi
        else:
            T_ = self.T
            occ = self.VTs[:T_] > 0
            gl_ = self.gl[:T_].view(T_, 4, 3)
            FtG = torch.einsum("tba,tib->tia", self.FT[:T_].view(T_, 3, 3), gl_)              # F̄ᵀ ∇λ_i
            kd = (self.VTs[:T_, None] * (lam + 2 * mu) * (FtG ** 2).sum(-1)) * occ[:, None]
            K = torch.zeros(self.M, device=dev).index_add_(0, self.tets[:T_].reshape(-1), kd.reshape(-1))
            om2 = float((K / self.mn[:self.M].clamp_min(1e-30)).max())
            if self.om0 is None:
                self.om0 = om2
            Nf = int(math.ceil(a.N * math.sqrt(om2 / self.om0)))
        q = max(1, a.ad_q)
        Nf = int(min(max(int(math.ceil(Nf / q)) * q, a.ad_lo), a.ad_hi))
        self.Ncur = Nf
        self.hscur = h / Nf
        self.dampcur = (VD_S ** (self.hscur / float(cfg["substep_dt"]))) if (a.pg_vdamp and VD_S < 1.0) else 1.0

    def _ev(self, k):
        if a.secprof:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.evs.append((k, e))

    def frame(self, i, x, v, F):
        self.evs = []
        self._ev("start")
        self.grid(x, v, i * h)
        self._ev("grid")
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
        if PL_DMG:
            torch.mul(vol, PL_AL, out=self.volW)
        if BSL:
            self._xs_diag = (xa, va, Fa)
            self._p2g_bs(xa, va, Fa)
        else:
            self._p2g(xa, va, Fa, h, False)                                         # 프레임 분할 (관성 예측 h v_n + h² g 의 끝 높이)
            if SPM != "none":
                if self._sparse():                                                  # 절 AE vpmesh: 첫 프레임에 V_p (·m_p) 를 바꿨다 -> 같은 상태로 P2G 다시
                    self._p2g(xa, va, Fa, h, False)
                    self._sparse()
        if PLAST or PBC:
            self._ag_frame(i)
        if FLOOR:
            EX.launch("ex_near_init", Np, (EX.P(self.sz), np.int32(0), EX.P(self.nidx), EX.P(xa), EX.P(va), EX.P(Fa), np.float32(h), np.float32(gv[0]),
                                           np.float32(gv[1]), np.float32(gv[2]), EX.P(self.x0c), EX.P(self.F0c), EX.P(self.F0i), EX.P(self.cc)), b=256)
        if lagm:
            self.Uc.zero_()
            if VFL:
                self.ncon.zero_()
            self.rst.zero_()
            if CC:                                                                  # 절 C: 목록 개수·D 비움, 점유 사면체 목록, 바닥 후보 시작 높이
                self.cint.zero_()
                self.clst[:1].zero_()
                if a.tlist:
                    CCU.launch("c_tet_list", self.Tcap, (EX.P(self.sz), EX.P(self.VTs), EX.P(self.tl), EX.P(self.cint[2:])), b=256)
                if VFL and a.vfl_cull:
                    CCU.launch("c_s0", Np, (EX.P(self.sz), EX.P(self.nidx), EX.P(self.Xf), *PLf, EX.P(self.s0c)), b=256)
        if a.adapt != "none":
            self._choose_N(i)
        self._ev("p2g")
        if lagm and a.lagdiag:
            self._Fp = Fv.view(Np, 3, 3)
            self._xp = x
            self._lagdiag(i)
            if a.fhist:
                self._fhist(i, va)
        elif a.no_graph:
            self._substeps()
        else:
            if a.adapt != "none":
                self.graph = self.graphs.get(self.Ncur)
                if self.graph is None and self.graphs:                             # 커널은 이미 컴파일됨 -- 데우기 없이 이 N 그래프만 잡는다
                    self.capture()
            if self.graph is None:
                bufs = (self.vn, self.Uc, self.mn, self.mv, self.VTs, self.FTs, self.FT, self.rows, self.lamb, self.pinv)
                if REP:
                    bufs = bufs + tuple(self.X) + tuple(self.V) + tuple(self.FF) + (self.sz,)
                if a.affine:
                    bufs = bufs + (self.CL,)
                if RWL:
                    bufs = bufs + (self.ml,)
                if SPG:
                    bufs = bufs + tuple(self.AG)
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
                if CC:                                                              # 절 C: 데우기가 남긴 목록 개수·D 비움 (점유 목록 tl·cint[2] 는 그대로)
                    self.cint[:2].zero_()
                    self.clst[:1].zero_()
                self.capture()
            self.graph.replay()
        if a.adapt == "act":
            self.lim_prev = self.rst[1].item()                                      # 동기화 (측정 안) -- 다음 프레임 N 결정
        self._ev("sub")
        if lagm:
            ci = ((NB - 1) % 2) if REPF else 0                                       # 마지막 블록의 입자 상태 버퍼
            xin, Fin = (self.X[ci], self.FF[ci]) if REP else (x, Fv)
            x2, v2, F2 = self.X[1 - ci], self.V[1 - ci], self.FF[1 - ci]
            if G2PBS:
                EX.launch("bsl_g2p", Np, (np.int32(Np), EX.P(xin), EX.P(self.Uc), EX.P(self.vn), EX.P(self.gpar), EX.P(self.ipar), EX.P(self.ncm),
                                          EX.P(Fin), EX.P(x2), EX.P(v2), EX.P(F2), np.int32(BSN)), b=128)
            elif FUPD:                                                              # 절 AB: F 갱신 방식만 바뀐 G2P
                if FUPDN:
                    self.NA[:self.M].zero_()
                    EX.launch("fa_tet_acc", self.Tcap, (EX.P(self.sz), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT), EX.P(self.VTs),
                                                        np.int32(0 if FUPD in (3, 5) else 1), EX.P(self.NA)), b=256)
                    if FUPD == 5:
                        self.NB[:self.M].zero_()
                        EX.launch("fa_tet_acc", self.Tcap, (EX.P(self.sz), EX.P(self.tets), EX.P(self.Uc), EX.P(self.gl), EX.P(self.FT), EX.P(self.VTs),
                                                            np.int32(1), EX.P(self.NB)), b=256)
                EX.launch("ex_g2p_lagF", Np, (np.int32(Np), EX.P(xin), EX.P(self.Uc), EX.P(self.vn), EX.P(self.rows), EX.P(self.lamb), EX.P(self.gl),
                                              EX.P(self.pinv), EX.P(Fin), EX.P(x2), EX.P(v2), EX.P(F2), EX.P(self.FT), EX.P(self.NA if FUPDN else None),
                                              np.int32(FUPD), np.float32(a.fupd_b), EX.P(self.NB if FUPD == 5 else None), np.float32(a.fupd_b2)), b=256)
            else:
                EX.launch("ex_g2p_lag", Np, (np.int32(Np), EX.P(xin), EX.P(self.Uc), EX.P(self.vn), EX.P(self.rows), EX.P(self.lamb), EX.P(self.gl),
                                             EX.P(self.pinv), EX.P(Fin), EX.P(x2), EX.P(v2), EX.P(F2), EX.P(self.CL if a.affine else None), np.int32(1 if RW4 else 0)), b=256)
            uc, xc_, Fc_ = self.Uc, None, None
            if PLAST:                                                               # 절 AG: 프레임 끝 입자별 정답 return mapping (F_e·상태 갱신, 걸음 dt = h)
                PLC.launch("ag_rm", Np, (np.int32(Np), EX.P(F2), EX.P(PL_YS), EX.P(PL_LJ), EX.P(PL_AL), np.int32(MATN), EX.P(PRM), np.float32(h), EX.P(PL_RMC)), b=128)
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
        if a.secprof:
            self._ev("g2p")
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


# ---------------------------------------------------------------- 절 AA 검산·이웃 벌어짐 (측정 밖)
def _cof64(Q):
    a_, b_, c_ = Q[..., 0, 0], Q[..., 0, 1], Q[..., 0, 2]
    d_, e_, f_ = Q[..., 1, 0], Q[..., 1, 1], Q[..., 1, 2]
    g_, h_, i_ = Q[..., 2, 0], Q[..., 2, 1], Q[..., 2, 2]
    return torch.stack([torch.stack([e_ * i_ - f_ * h_, f_ * g_ - d_ * i_, d_ * h_ - e_ * g_], -1),
                        torch.stack([c_ * h_ - b_ * i_, a_ * i_ - c_ * g_, b_ * g_ - a_ * h_], -1),
                        torch.stack([b_ * f_ - c_ * e_, c_ * d_ - a_ * f_, a_ * e_ - b_ * d_], -1)], -2)


def psi64(X):
    """커널과 같은 FCR (float64): μ‖X − Q‖² + λ/2 (|det X| − 1)², Q = 크기 조정 뉴턴 극분해 10 회 (polar 장치 함수 그대로)."""
    Q = X
    for _ in range(10):
        dt = _det3(Q)
        ds = torch.where(dt.abs() > 1e-30, dt, torch.full_like(dt, 1e-30))
        gm = ds.abs().pow(-1.0 / 3.0)
        Q = 0.5 * (gm[..., None, None] * Q + _cof64(Q) / (ds * gm)[..., None, None])
    return mu * ((X - Q) ** 2).sum((-1, -2)) + 0.5 * lam * (_det3(X).abs() - 1.0) ** 2


def _psi9(x9):
    return psi64(x9.view(3, 3))


_HESS = torch.func.vmap(torch.func.hessian(_psi9))


def cq_energy(A, Fb, CS9):
    """셀별 [ψ(A F̄), ½<H, S>] (float64). A·Fb [T,3,3], CS9 [T,9,9]."""
    X = A @ Fb
    H = _HESS(X.reshape(-1, 9))
    Sx = torch.einsum("trk,tsl,tkcld->trcsd", A, A, CS9.view(-1, 3, 3, 3, 3)).reshape(-1, 9, 9)
    return psi64(X), 0.5 * (H * Sx).sum((-1, -2))


def _cs_full(CS):
    iu = torch.triu_indices(9, 9, device=CS.device)
    M_ = torch.zeros(CS.shape[0], 9, 9, device=CS.device, dtype=torch.float64)
    M_[:, iu[0], iu[1]] = CS.double()
    M_[:, iu[1], iu[0]] = CS.double()
    return M_


def _tet_e_U(Uloc, gl4, Fb, CS9, V):
    """사면체 하나의 2 차 근사 에너지 (자동 미분용, float64): Uloc [4,3]."""
    A = torch.eye(3, dtype=torch.float64, device=Uloc.device) + torch.einsum("ic,ib->cb", Uloc, gl4)
    X = A @ Fb
    H = torch.func.hessian(_psi9)(X.reshape(9))
    Sx = torch.einsum("rk,sl,kcld->rcsd", A, A, CS9.view(3, 3, 3, 3)).reshape(9, 9)
    return V * (psi64(X) + 0.5 * (H * Sx).sum())


_TET_G = torch.func.vmap(torch.func.grad(_tet_e_U))
_TET_E = torch.func.vmap(_tet_e_U)


def cq_check(i, Fstart):
    """프레임 i 끝 (G2P 직전 U) 에서: (1) 셀 에너지 세 가지 -- 2 차 근사 / 입자 적분 (1/V_T) Σ V_p ψ(A F_p) / ψ(A F̄) -- 의 상대 차를
    입자 수 칸·Σ_F 큰 셀 칸별로, A = 프레임 끝 A_T 와 A = I 둘 다; (2) 힘: 커널 (km·cf) vs float64 자동 미분 (∂³ψ 포함) vs 에너지 중앙 차분."""
    T_ = OPT.T
    occ = OPT.VTs[:T_] > 0
    tid = torch.nonzero(occ).squeeze(1)
    U = OPT.Uc.double()
    tets = OPT.tets[:T_][tid]
    gl4 = OPT.gl[:T_][tid].view(-1, 4, 3).double()
    Fb = OPT.FT[:T_][tid].view(-1, 3, 3).double()
    VT = OPT.VTs[:T_][tid].double()
    CS9 = _cs_full(OPT.CS[:T_][tid])
    trS = CS9.diagonal(dim1=-2, dim2=-1).sum(-1)
    Aend = torch.eye(3, dtype=torch.float64, device=dev) + torch.einsum("tic,tib->tcb", U[tets.reshape(-1)].view(-1, 4, 3), gl4)
    # 입자 -> 점유 사면체 순번 (P2G 와 같은 집합: --floor vel·바닥 없음이면 전부)
    rank = torch.full((T_,), -1, dtype=torch.long, device=dev)
    rank[tid] = torch.arange(tid.numel(), device=dev)
    pin = OPT.pinv.clone()
    inc = (OPT.nslot < 0) | torch.tensor(bool(VFL), device=dev)
    pr = rank[pin]
    inc = inc & (pr >= 0)
    pr = pr[inc]
    Fp = Fstart.reshape(Np, 3, 3).double()[inc]
    Vp = vol.double()[inc]
    cnt = torch.bincount(pr, minlength=tid.numel())
    out = dict(t=i, T_occ=int(tid.numel()), cnt_q=[int(q) for q in torch.quantile(cnt.float(), torch.tensor([0.1, 0.5, 0.9], device=dev)).tolist()])
    eps = 1e-6 * mu
    bins = [("n1", cnt == 1), ("n2", cnt == 2), ("n3_5", (cnt >= 3) & (cnt <= 5)), ("n6_15", (cnt >= 6) & (cnt <= 15)), ("n16+", cnt >= 16)]
    hi = trS > torch.quantile(trS.float(), 0.9).double()
    bins += [("trS_top10", hi), ("trS_top1", trS > torch.quantile(trS.float(), 0.99).double()), ("all", torch.ones_like(hi))]
    for nmA, A in (("A_end", Aend), ("A_I", torch.eye(3, dtype=torch.float64, device=dev).expand_as(Aend).contiguous())):
        eb, eq = [], []
        for s0 in range(0, tid.numel(), 8192):
            e0, e1 = cq_energy(A[s0:s0 + 8192], Fb[s0:s0 + 8192], CS9[s0:s0 + 8192])
            eb.append(e0)
            eq.append(e0 + e1)
        eb, eq = torch.cat(eb), torch.cat(eq)
        ep = torch.zeros_like(eb).index_add_(0, pr, Vp * psi64(A[pr] @ Fp)) / VT
        r = {}
        for nb, mk in bins:
            if not bool(mk.any()):
                continue
            w = VT[mk]
            den = (w * ep[mk]).sum()
            rq = ((eq[mk] - ep[mk]).abs() / (ep[mk] + eps))
            rb = ((eb[mk] - ep[mk]).abs() / (ep[mk] + eps))
            qs = torch.tensor([0.5, 0.9, 0.99], device=dev, dtype=torch.float64)
            r[nb] = dict(n=int(mk.sum()), sum_quad=float(((w * eq[mk]).sum() - den) / den), sum_bar=float(((w * eb[mk]).sum() - den) / den),
                         cell_quad_q=[float(q) for q in torch.quantile(rq.float(), qs.float()).tolist()],
                         cell_bar_q=[float(q) for q in torch.quantile(rb.float(), qs.float()).tolist()],
                         e_part_med=float(ep[mk].median()))
        out[nmA] = r
    # (2) 힘: 부분 사면체 집합 (Σ_F 큰 1000 + 무작위 1000, 보정이 켜지는 셀) -- 커널은 그 사면체만 V 를 남긴 복사본으로
    g = torch.Generator(device="cpu").manual_seed(i)
    act = torch.nonzero((trS > 0)).squeeze(1)
    if act.numel():
        top = act[torch.argsort(trS[act], descending=True)[:1000]]
        rnd = act[torch.randperm(act.numel(), generator=g)[:1000].to(dev)]
        sub = torch.unique(torch.cat([top, rnd]))
        Vm = torch.zeros_like(OPT.VTs)
        Vm[tid[sub]] = OPT.VTs[tid[sub]]
        fk = {}
        gdb = torch.zeros(OPT.Tcap, 27, device=dev)
        for hm_, hv in (("km", 0), ("cf", 1), ("base", 0)):
            fbuf = torch.zeros_like(OPT.f)
            CSb = OPT.CS if hm_ != "base" else torch.zeros_like(OPT.CS)
            EX.launch("ex_tet_force_CQ", OPT.Tcap, (EX.P(OPT.sz), EX.P(OPT.tets), EX.P(OPT.Uc), EX.P(OPT.gl), EX.P(OPT.FT), EX.P(Vm), EX.P(CSb),
                                                    np.float32(mu), np.float32(lam), EX.P(fbuf), EX.P(None), np.int32(hv), EX.P(gdb if hm_ == "km" else None)))
            torch.cuda.synchronize()
            fk[hm_] = fbuf[:OPT.M].double()
        ts = tets[sub]
        Ul = U[ts.reshape(-1)].view(-1, 4, 3)
        gr = _TET_G(Ul, gl4[sub], Fb[sub], CS9[sub], VT[sub])                            # ∂E/∂U_loc (∂³ψ 포함, 자동 미분)
        fref = torch.zeros_like(fk["km"]).index_add_(0, ts.reshape(-1), -gr.reshape(-1, 3))
        nz = fref.norm(dim=-1) > 0
        rel = lambda f_: float((f_ - fref)[nz].norm() / fref[nz].norm())
        # 중앙 차분 (무작위 방향, 에너지 float64)
        dU = torch.randn(Ul.shape, generator=torch.Generator(device="cpu").manual_seed(100 + i), dtype=torch.float64).to(dev)
        dU = dU / dU.norm() * 1e-3 * a.node_h * math.sqrt(Ul.shape[0])
        Ef = lambda Ux: float(_TET_E(Ux, gl4[sub], Fb[sub], CS9[sub], VT[sub]).sum())
        cds = {}
        for ep_ in (1e-1, 1e-2):
            cds[f"{ep_:g}"] = (Ef(Ul + ep_ * dU) - Ef(Ul - ep_ * dU)) / (2 * ep_)
        dd_ad = float((gr * dU).sum())
        # 사면체별 ∂E/∂A 세 몫 (커널 gdbg) vs float64 자동 미분: ψ 몫, X 경유 q 몫 (S 고정), S 경유 q 몫 (H 고정)
        Ab = Aend[sub].clone()
        def _parts(A1, Fb1, CS1, which):
            X1 = A1 @ Fb1
            if which == "psi":
                return psi64(X1)
            A0 = A1.detach()
            if which == "qx":
                H = torch.func.hessian(_psi9)(X1.reshape(9))
                Sx = torch.einsum("rk,sl,kcld->rcsd", A0, A0, CS1.view(3, 3, 3, 3)).reshape(9, 9)
            else:
                H = torch.func.hessian(_psi9)((A0 @ Fb1).reshape(9))
                Sx = torch.einsum("rk,sl,kcld->rcsd", A1, A1, CS1.view(3, 3, 3, 3)).reshape(9, 9)
            return 0.5 * (H * Sx).sum()
        gk = gdb[tid[sub]].double().view(-1, 3, 9)
        prt = {}
        for j_, wh in enumerate(("psi", "qx", "qs")):
            gA = torch.func.vmap(torch.func.grad(lambda A1, F1, C1: _parts(A1, F1, C1, wh)))(Ab, Fb[sub], CS9[sub]).reshape(-1, 9)
            kk = gk[:, j_]
            prt[wh] = dict(rel=float((kk - gA).norm() / gA.norm().clamp_min(1e-30)), norm_ad=float(gA.norm()), norm_k=float(kk.norm()),
                           cos=float((kk * gA).sum() / (kk.norm() * gA.norm()).clamp_min(1e-30)))
        out["force_parts"] = prt
        out["force"] = dict(n_tets=int(sub.numel()), rel_km_vs_ad=rel(fk["km"]), rel_cf_vs_ad=rel(fk["cf"]), rel_base_vs_ad=rel(fk["base"]),
                            rel_km_vs_cf=float((fk["km"] - fk["cf"]).norm() / fk["cf"].norm()),
                            corr_over_base=float((fk["km"] - fk["base"]).norm() / fk["base"].norm()),
                            dirderiv_ad=dd_ad, dirderiv_cd=cds, rel_cd_vs_ad={k_: abs(v_ - dd_ad) / max(abs(dd_ad), 1e-30) for k_, v_ in cds.items()})
    with open(os.path.join(a.out, "cqchk.jsonl"), "a") as fo:
        fo.write(json.dumps(out, ensure_ascii=False) + "\n")
    fr = out.get("force", {})
    print(f"  [AA 검산] 프레임 {i + 1}: 셀 에너지 상대 차 (2 차 근사 / ψ(A F̄) vs 입자 적분, 셀 중앙·90 %) "
          + " ".join(f"{k_} {v_['cell_quad_q'][0]:.2e}/{v_['cell_bar_q'][0]:.2e} {v_['cell_quad_q'][1]:.2e}/{v_['cell_bar_q'][1]:.2e}" for k_, v_ in out["A_end"].items())
          + (f" | 힘 커널 km vs 자동미분 {fr['rel_km_vs_ad']:.2e} cf {fr['rel_cf_vs_ad']:.2e} 보정 끈 커널 {fr['rel_base_vs_ad']:.2e} 중앙차분 vs 자동미분 {fr['rel_cd_vs_ad']}" if fr else "")
          + (" | 몫별 " + json.dumps(out["force_parts"], ensure_ascii=False) if out.get("force_parts") else ""), flush=True)


FSP = None
if a.fspread:
    GTF = DM.traj_F(d, h, "cpu")
    GTF = GTF[:L + 1].reshape(-1, Np, 9).float() if torch.is_tensor(GTF) else torch.stack([q.reshape(Np, 9).float().cpu() for q in GTF[:L + 1]])
    _gsp = torch.Generator(device="cpu").manual_seed(7)
    FSP = torch.randperm(Np, generator=_gsp)[:3000].to(dev)


def _knn(q, allx, k=16):
    out = []
    for s0 in range(0, q.shape[0], 256):
        dd_ = torch.cdist(q[s0:s0 + 256].double(), allx.double(), compute_mode="donot_use_mm_for_euclid_dist")
        out.append(torch.topk(dd_, k + 1, largest=False).indices[:, 1:])
    return torch.cat(out)


def fspread(i, x2, F2):
    """프레임 i 끝 입자 F 이웃 벌어짐 (절 Y 지표): ‖F_p − mean_16NN F_q‖_F, 우리 (우리 위치로 이웃) / 정답 (정답 위치·정답 F). 표본 = 고정 무작위 3000 + 프레임 시작 입자 1~2 개 셀 입자 3000."""
    T_ = OPT.T
    pin = OPT.pinv
    cnt = torch.bincount(pin, minlength=T_)
    sp_ = torch.nonzero(cnt[pin] <= 2).squeeze(1)
    sp_ = sp_[torch.randperm(sp_.numel(), generator=torch.Generator(device="cpu").manual_seed(11 + i))[:3000].to(dev)] if sp_.numel() else sp_
    Fo = F2.reshape(Np, 9)
    gx = d["x"][i + 1].to(dev).float()
    gF = GTF[i + 1].to(dev)
    r = {}
    for nm_, ids in (("all", FSP), ("sparse", sp_)):
        if ids.numel() == 0:
            continue
        nb = _knn(x2[ids], x2)
        mo = (Fo[ids] - Fo[nb].mean(1)).norm(dim=-1)
        gnb = _knn(gx[ids], gx)
        mg = (gF[ids] - gF[gnb].mean(1)).norm(dim=-1)
        qs = torch.tensor([0.5, 0.9], device=dev)
        r[nm_] = dict(n=int(ids.numel()), ours=[float(q) for q in torch.quantile(mo, qs).tolist()], gt=[float(q) for q in torch.quantile(mg, qs).tolist()])
    return r

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
            st = dict(M=OPT.M, T=OPT.T, recap=OPT.n_cap - ncap0, Nf=OPT.Ncur)
            if a.secprof and len(getattr(OPT, "evs", [])) > 1:                      # 절 C 구간 시간 (이벤트, 프레임 끝 동기화 뒤에 읽는다)
                st["_evs"] = OPT.evs
    except _Diverge as ex_:
        div_at = i + 1
        print(f"[발산] 프레임 {div_at}: {ex_} -- 멈춤", flush=True)
        break
    torch.cuda.synchronize()
    t_frame = time.perf_counter() - t0
    if "_evs" in st:
        evs_ = st.pop("_evs")
        for (k0, e0), (k1, e1) in zip(evs_[:-1], evs_[1:]):
            st["sec_" + k1] = e0.elapsed_time(e1)
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
                      sr_viol=int(rs[8]), sr_inv=int(rs[9]), sb_act=int(rs[10]), cq_act=int(rs[11]),
                      srA_max=(float((lambda sv: (sv[:, 0] / sv[:, -1].clamp_min(1e-30)).nan_to_num(float("inf")).max())(torch.linalg.svdvals(A_[occ_].double()))) if bool(occ_.any()) else None),
                      p_tot=[float(q) for q in (m[:, None] * v2).sum(0).tolist()], nt_ovf=int(OPT.novf.item()))
        if RNEW:                                                                    # 절 AD: 폭주 노드 (프레임 끝 |v_n| > 5 x 90 % 분위, 절 U 규칙) -- 기준 판에도
            vnn = OPT.vn[:OPT.M].norm(dim=-1)
            q90 = torch.quantile(vnn[OPT.mn[:OPT.M] > 0].float(), 0.9)
            st.update(rn=int((vnn > 5.0 * q90).sum()), vn_max=float(vnn.max()), vn_q90=float(q90))
        if PLAST:                                                                   # 절 AG 소성 통계 (측정 밖)
            rmc_ = PL_RMC.tolist()
            PL_RMC.zero_()
            st.update(pl_rm=int(rmc_[0]), pl_newdmg=int(rmc_[1]), pl_dmg=int((PL_AL <= 0).sum()), pl_ys_mean=float(PL_YS.mean()),
                      pl_lj=[float(PL_LJ.min()), float(PL_LJ.mean()), float(PL_LJ.max())], pl_lj_pos=int((PL_LJ > 0).sum()))
        if SPM != "none":
            sps = [int(q_) for q_ in torch.stack(OPT._spst).tolist()]
            st.update(sp_occ=sps[0], sp_n=sps[1], sp_n1=sps[2], **(dict(sp_grouped=sps[3], sp_groups=sps[4]) if len(sps) > 3 else {}))
            if SPM in ("stiff", "vcap", "vpmesh"):
                st.update(sp_vratio=float(OPT.VTs[:OPT.T].sum() / OPT._vt0))           # 탄성 부피 합 비 (바꾼 뒤 / 입자 부피 합)
            if a.sp_chk and str(i + 1) in a.sp_chk.split(","):
                st.update(sp_chk=OPT.sp_check())
                print(f"  [절 AD 검산] 프레임 {i + 1}: {st['sp_chk']}", flush=True)
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
            if BSCM:                                                                # 절 W: 셀 단계 합 (P2G -> 셀, 셀 -> 노드)
                st.update(mass_c=float(OPT.cM[:OPT.Pn].double().sum()), P_c=OPT.cMV[:OPT.Pn * 3].view(-1, 3).double().sum(0).tolist())
            if a.bsdiag >= 0 and (i == 0 or (a.bsdiag > 0 and i % a.bsdiag == 0)):
                OPT.bsdiag(i)
        if szh[3]:
            print(f"  [경고] 프레임 {i + 1}: 팽창 영역 밖 입자 {szh[3]} (격자 넘침)", flush=True)
            OPT.sz[3:4].zero_()
        if CQ and a.cqchk and str(i + 1) in a.cqchk.split(","):                    # 절 AA 검산 (측정 밖, 프레임 끝 U·프레임 시작 입자 F)
            cq_check(i, F)
        if FSP is not None and (i == 0 or (i + 1) % a.fspread == 0):
            try:
                st.update(fspread=fspread(i, x2, F2))
            except Exception as ex_:                                               # 발산 직전 비유한 등 -- 기록만
                st.update(fspread=str(ex_)[:200])
    if GT_ALLOK:
        e = float((x2 - GT[i]).norm(dim=-1).mean()) / EXT
    else:
        e = float((x2[GTOK[i]] - GT[i][GTOK[i]]).norm(dim=-1).mean()) / EXT
    spread = float((x2.max(0).values - x2.min(0).values).norm()) / EXT
    if a.rbdiag and GT_ALLOK:                                                       # 절 AH 진단 (측정 밖): 강체 지표 우리 / 정답
        st.update(rb=rb_stats(x2, v2, GT[i], d["v"][i + 1].to(dev).float()))
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
          + (f" [R] 뒤집힘 (사면체x서브스텝) {st['inv_sub']} 끝 {st['inv_end']} det A 최소 {(st['detA_min'] if st['detA_min'] is not None else float('nan')):.3g} 입자 F 뒤집힘 {st['Fp_inv']} 제한 {st['lim_n']} 줄인 KE {st['lim_dke']:.3g} KE {st['ke_p']:.4g} 끝 σ1/σ3(A) 최대 {(st['srA_max'] if st['srA_max'] is not None else float('nan')):.4g}"
             + (f" 비율 위반 {st['sr_viol']} det≤0 {st['sr_inv']}" if a.lim_sr_chk else "") + (f" 장벽 활성 {st['sb_act']}" if SB else "") if RNEW else ""), flush=True)
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
            lim_sr=(a.lim_sr if SRL else None), sb=(a.sb if SB else None), sb_c=(a.sb_c if SB else None), sb_dhat=(SB_D if SB else None),
            sr_viol_tot=(sum(r.get("sr_viol", 0) for r in recs) if RNEW else None), sr_inv_tot=(sum(r.get("sr_inv", 0) for r in recs) if RNEW else None),
            sb_act_tot=(sum(r.get("sb_act", 0) for r in recs) if RNEW else None), lim_frames=(sum(1 for r in recs if r.get("lim_n", 0) > 0) if RNEW else None),
            srA_max=(max((r["srA_max"] for r in recs if r.get("srA_max") is not None), default=None) if RNEW else None),
            inv_sub_tot=(sum(r.get("inv_sub", 0) for r in recs) if RNEW else None), inv_frames=(sum(1 for r in recs if r.get("inv_sub", 0) > 0) if RNEW else None),
            inv_end_max=(max(r.get("inv_end", 0) for r in recs) if RNEW else None), Fp_inv_max=(max(r.get("Fp_inv", 0) for r in recs) if RNEW else None),
            detA_min=(min((r["detA_min"] for r in recs if r.get("detA_min") is not None and math.isfinite(r["detA_min"])), default=None) if RNEW else None),
            lim_n_tot=(sum(r.get("lim_n", 0) for r in recs) if RNEW else None), lim_dke_tot=(sum(r.get("lim_dke", 0.0) for r in recs) if RNEW else None),
            lim_dpabs_tot=(sum(r.get("lim_dpabs", 0.0) for r in recs) if RNEW else None), nt_ovf=(max(r.get("nt_ovf", 0) for r in recs) if RNEW else None),
            p2g=a.p2g, g2p=a.g2p, bs_mesh=(a.bs_mesh if BSL else None), bs_mass=(a.bs_mass if BSL else None), interp=a.interp, rep=REP, rep_mode=(a.rep_mode if REP else None), quad=(a.quad if BSP else None), K=(a.K if BSP and a.quad == "cell" else None), apic=bool(a.apic), rep_w=a.rep_w,
            nint=NINT, nint_beta=(a.nint_beta if NINT else None), fupd=a.fupd, fupd_b=(a.fupd_b if FUPD else None), fupd_b2=(a.fupd_b2 if FUPD == 5 else None),
            sp=SPM, sp_k=a.sp_k, sp_occ=a.sp_occ, sp_beta=a.sp_beta, sp_wocc=a.sp_wocc, sp_mlow=a.sp_mlow,
            rn_tot=(sum(r.get("rn", 0) for r in recs) if RNEW else None), rn_max=(max((r.get("rn", 0) for r in recs), default=None) if RNEW else None),
            sp_n_med=(med([r.get("sp_n", 0) for r in recs]) if SPM != "none" else None), sp_grouped_med=(med([r.get("sp_grouped", 0) for r in recs]) if SPM in ("merge", "cube") else None), sp_a=a.sp_a, sp_hg=a.sp_hg, sp_vcap=a.sp_vcap, sp_vratio_med=(med([r.get("sp_vratio", 1.0) for r in recs]) if SPM in ("stiff", "vcap", "vpmesh") else None))
if CC and a.cchk:
    ck = {k: float(t_) for k, t_ in OPT.chk.items()}
    json.dump(ck, open(os.path.join(a.out, "cchk.json"), "w"), indent=1)
    print(f"[절 C 검산] {ck}", flush=True)
summ.update(cfast=dict(vfl_cull=a.vfl_cull, sr_fast=a.sr_fast, tlist=a.tlist, fpolar=a.fpolar, cfuse=a.cfuse) if CC else None,
            adapt=(dict(rule=a.adapt, lo=a.ad_lo, hi=a.ad_hi, thr=a.ad_thr, q=a.ad_q, N_mean=float(np.mean([r.get("Nf", a.N) for r in recs])),
                        N_hist={str(k): sum(1 for r in recs if r.get("Nf") == k) for k in sorted(set(r.get("Nf", a.N) for r in recs))}) if a.adapt != "none" else None))
summ.update(plast=(dict(material=cfg.get("material"), mat=MATN, **PLP, dmg_final=int((PL_AL <= 0).sum()), lj_pos_final=int((PL_LJ > 0).sum())) if PLAST else None),
            visc=a.visc, pbc=PBC, floor_plane=a.floor_plane or None, gt_nonfinite_last=int((~GTOK[-1]).sum()))
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
print(f"[롤아웃] {a.tag} 서브스텝 {a.N} {a.mode}{' 최적화' if a.opt else ''}: {done}/{L} 프레임{' (발산 %d)' % div_at if div_at else ''}, 평균 {100 * summ['err_mean_done']:.3f}%  "
      f"프레임당 {summ['ms_mean']:.2f} ms (중앙 {summ['ms_med']:.2f}, {summ['fps']:.2f} fps, 앞 {W} 프레임 제외) 퍼짐 {summ['spread_last']:.3f} "
      f"최저 입자 {summ['s_min']} 아래 최대 {summ['n_below_max']}", flush=True)
if PRED is not None and (div_at is None or (a.dump_div and done > 0)):
    if div_at is not None:                                                          # 절 AG --dump_div: 끝난 프레임까지만
        PRED, GT, FD, VD = PRED[:done], GT[:done], FD[:done], VD[:done]
    if a.dump_pred:
        torch.save({"pred": PRED.cpu(), "EXT": EXT}, os.path.join(a.out, "pred.pt"))
    if not a.no_dump:
        torch.save({"pred": PRED.cpu(), "gt": GT.cpu(), "x0": d["x"][0].float(), "EXT": EXT, "t0": 0, "tag": a.tag,
                    "F": FD.float().cpu(), "v": VD.float().cpu(), "ctrl_pos": d.get("ctrl_pos"), "ctrl_R": d.get("ctrl_R"), "ctrl_id": d.get("ctrl_id"),
                    "nodes": None, "fscal": None, "gt_fscal": None}, os.path.join(a.out, "roll_fem.pt"))
        print(f"[롤아웃 덤프] {os.path.join(a.out, 'roll_fem.pt')}", flush=True)
