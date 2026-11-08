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
from anchorflow import fem_an_rt_v0 as ART                                           # noqa: E402
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
ap.add_argument("--dump_lj", action="store_true", help="절 CM13.4 (결과 불변): 프레임 끝 입자 logJp (NACC 경화 상태 ljp) 를 원래 입자 순서로 lj.pt 에, 손상 몫 al·ys 범위는 로그에")
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
ap.add_argument("--nan_step", action="store_true", help="절 CM9.3 진단 (결과 불변, 재질 판만): 걸음·뉴턴 반복마다 노드 변위·기울기·가우스 점 뒤집힘·응력의 처음 비유한 시점을 기록, 발산 프레임에 nan_step_f<프레임>.npz (직전 프레임 위치 포함)")
ap.add_argument("--nanchk", type=int, default=0, help="진단: 이 프레임 (1 부터) 을 그래프 밖에서 커널마다 비유한 검사 (그 프레임은 재생 대신)")
ap.add_argument("--nhb_eps", type=float, default=0.0, help="수박 (NHB): 뒤집힌 시험 상태 바닥 -- 주값 σ̂ = max(σ, ε) 에서 RM·응력·접선, 커밋·G2P 도 (0 = 끔, 정답 식 그대로)")
ap.add_argument("--no_vfill", action="store_true", help="질량 0 노드 속도·가속도 이웃 평균 끔 (예전 판: 0)")
ap.add_argument("--mat_as", type=str, default="", help="진단: config 재질 이름을 바꿔 돌림 (jelly = 절 AN 탄성 판 RTAdapt, plasticine 등)")
ap.add_argument("--no_rm", action="store_true", help="진단: 재질 판 RM 끔 (탄성 응력만, prm[15])")
ap.add_argument("--ys_big", action="store_true", help="진단: 항복 응력을 아주 크게 (소성 없이 탄성만)")
ap.add_argument("--visc_rpic", action="store_true", help="정답 config rpic_damping r -> 등가 점성 η = ρ r (2−r) Δx_gt² / (16 Δt_gt) (절 AG.20)")
ap.add_argument("--qjd", action="store_true", help="절 CM20 진단 (결과 불변, 재질 판): 프레임 끝 U 로 가우스 점 X_q = F_q + Σ U ⊗ G_qn 의 J (절 CM18 rt_qjd 그대로, 따로 컴파일한 읽기 전용 커널) -- J ≤ 0 수·최소를 fem.jsonl qj 에")
ap.add_argument("--fp_interp", action="store_true", help="절 CM20 (사용자 승인 11-07 정의 변경, 재질 판 수박): 마지막 걸음도 가우스 점에서 커밋 (RM·상태) 한 뒤 입자 F·(ys, logJp, al) 를 입자 G2P RM 누적 대신 프레임 끝 가우스 점 값의 칸 안 삼선형 보간 (회전·로그 신장 분리, --fq_rl 규칙) 으로 교체. 끄면 예전과 같음")
ap.add_argument("--gq_mass", action="store_true", help="절 CM9 (사용자 승인 정의 변경): 노드 질량·운동량·가속도 몫을 입자 -> 자기 팔분 가우스 점 -> 노드 (힘과 같은 점 x_q·같은 V_q 가중 (칸 합 / 8)·같은 기저) 로. 비결정 판만, 끄면 예전과 같음")
ap.add_argument("--gq_chk", action="store_true", help="절 CM9 진단 (결과 불변): 노드 질량·운동량·가속도 몫 합 대 입자 합 상대 차 (fem.jsonl gqchk; --pbc 씬은 뜻 없음)")
ap.add_argument("--floor_plane", type=str, default="", help="바닥 평면 px,py,pz,nx,ny,nz (주면 config surface_collider 대신; 수박 = GF bounding_box)")
ap.add_argument("--floor_slip", action="store_true", help="미끄럼 바닥 (들어가는 법선 성분만 고착 몫)")
ap.add_argument("--fl_bar", action="store_true", help="절 CM21 (사용자 승인 11-07 정의 변경, 재질 판): 장벽 바닥 = 레고 (B) 정의 -- 팔분 표본점 (칸 입자 질량 중심) IPC 거리 장벽 (d̂ = ½ dx_정답, κ = (λ+2μ) V_q / d̂²), 법선 자유·고착은 접선에만, "
                "보폭 = 뒤집힘 필터와 장벽 실현 가능 (활성 표본점 d > 0) 중 작은 쪽, HHT 고착 가속도 0 은 접선만 (--floor_slip 없이)")
ap.add_argument("--fl_bst", type=int, default=0, choices=[0, 1], help="절 CM21 (--fl_bar 위): 1 = 띠 고착 (절 AN.21 bst 1: 표본점 d < d̂ 이면 고착 몫 1, 밖 0). 이 실행기는 1 만")
ap.add_argument("--y0loc", type=int, default=0, help="절 CM27 (c) (--fl_bar 위, 새 실행기 exe/fem_an_rt_sim.py --y0loc 와 같은 정의·커널): 시작점 섞기 노드별 β, 확인 라운드 이 수 (0 = 끔, 소스 그대로)")
ap.add_argument("--y0eb", action="store_true", help="절 CM21 (--fl_bar 위, 절 CM8 (B)): 시작점 = U 에서 관성 예측 쪽으로 장벽 에너지가 E(U) 이하인 가장 큰 2^-k (실현 가능 k 부터)")
ap.add_argument("--bar_cnt", action="store_true", help="절 CM21 진단 (결과 불변, --fl_bar 위): 프레임마다 장벽 걸린 표본점 수 (어느 뉴턴 반복 기울기 위치에서든 활성이고 d < d̂) -> fem.jsonl bc")
ap.add_argument("--lj_interp", action="store_true", help="절 CM21 (사용자 승인 11-07 정의 변경, 재질 판 수박): 마지막 걸음도 가우스 점에서 커밋 (RM·logJp) 한 뒤 입자 logJp 를 프레임 끝 가우스 점 logJp 의 칸 안 삼선형 보간 (절 CM20 과 같은 가중) 으로 교체. F 는 G2P 누적 그대로")
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
ap.add_argument("--elx", choices=["", "jbar", "snh"], default="", help="절 CM13 (사용자 승인 정의 변경, 재질 판 수박만): jbar = 가우스 점 J 로그 장벽 κ_J V_q (−log J + J − 1) 을 에너지에 (기울기·투영 헤시안), snh = 탄성 부분을 Stable Neo-Hookean (μ_s = μ, λ_s = λ + μ; RM 그대로). 끄면 같은 소스")
ap.add_argument("--dmg", choices=["", "cdm", "pf"], default="", help="절 CM19 (사용자 승인 정의 변경, 수박 --elx snh 위에만): cdm = 연속체 손상 (입자·가우스 점 스칼라 d, ψ = g(d) ψ⁺ + ψ⁻, g = (1−d)², d = 이력 최대 국소 법칙), "
                "pf = 위상장 파괴 AT2 (노드 d, 이력장 H = max ψ⁺, 걸음마다 변위 풀이 뒤 d 풀이). NACC RM 그대로. 끄면 같은 소스")
ap.add_argument("--dmg_psic", choices=["tens", "uni"], default="tens", help="절 CM19 ψ_c 근거: tens = NACC 정수압 인장 꼭짓점 (J = Je = √(1 + 2βp0/κ)) 의 ψ⁺, uni = NACC 단축 인장 항복 상태의 ψ⁺ (p0 = 처음 logJp 값)")
ap.add_argument("--pf_l0", type=float, default=1.0, help="절 CM19 위상장 l0 = 이 값 x 칸 크기 (B-스플라인 노드 간격)")
ap.add_argument("--pf_it", type=int, default=30, help="절 CM19 위상장 d 풀이 PCG 반복 수 (고정, 그래프 안)")
ap.add_argument("--jbar", type=float, default=1.0, help="절 CM13 판 (1): κ_J = 이 값 x μ (재질 전단 강성)")
ap.add_argument("--nrec", action="store_true", help="절 CM23.3 진단 (결과 불변, 재질 판만): 걸음·뉴턴 반복별 |g|/F (허용치 규칙 척도), 멈춤, PCG 반복·상한, 보폭, 마지막 뉴턴 뒤 |g|/F (fem.jsonl nr), 입자 질량중심 z 우리·정답 (com)")
ap.add_argument("--jdiag", action="store_true", help="절 CM13 진단 (결과 불변, 재생 밖): 프레임 끝 가우스 점 F_q (부피 > 0) det ≤ 0 수·최소, 입자 F det 최소·≤ 0 수·< 1e-3 수 (fem.jsonl jd)")
ap.add_argument("--theta", type=float, default=None, help="θ 방식 (절 AN.err2 정의; 0.5 = 중점, 1 = 후퇴 오일러). 안 주면 --integ 그대로 (비트 동일)")
ap.add_argument("--frec", type=int, default=0, help="절 CM23.4 진단 (결과 불변, --nrec 위, K 1 만): 프레임 1~이 값의 뉴턴 반복별 노드 힘 몫 (탄성·장벽·관성·중력·잔차) 법선 성분 합 (몸 전체·바닥 위 --frec_band 칸), "
                "걸음 속도 변화 몫 분해, 장벽 표본점 (걸린 수·에너지·최소 d/d̂), 프레임 끝 입자 det F 분포·SNH 탄성 에너지 (우리·정답 F) -> fem.jsonl fr")
ap.add_argument("--frec_band", type=float, default=3.0, help="절 CM23.4: 바닥 근처 = 걸음 시작 노드 위치의 바닥 위 높이 < 이 값 x 칸")
ap.add_argument("--bar_kmul", type=float, default=1.0, help="절 CM23.4 대조 (진단용 정의 변경): 장벽 강성 κ x 이 값")
ap.add_argument("--bar_dhmul", type=float, default=1.0, help="절 CM23.4 대조 (진단용 정의 변경): 장벽 두께 d̂ x 이 값 (κ = (λ+2μ) V/d̂² 정의 그대로 따라 바뀜, --bar_dhk 면 κ 는 원래 d̂ 로)")
ap.add_argument("--bar_dhk", action="store_true", help="절 CM23.4: --bar_dhmul 에서 κ 를 원래 d̂ 값으로 고정")
ap.add_argument("--ksplit", type=int, default=0, help="절 CM24 (1b, 사용자 승인 11-07 정의 변경): 적응 걸음 쪼개기 -- K 1 로 풀고 덜 풀린 걸음 (판정 커널: 허용치 εN 으로 안 멈췄고 뉴턴 보폭 하나라도 < 1) 이 있으면 "
                "그 프레임을 버리고 K 두 배 (걸음 간격 반) 로 다시, 최대 K = 2^이 값 (0 = 끔)")
ap.add_argument("--ue_loc", type=int, default=0, help="절 CM31 (새 실행기 --ue_loc 와 같은 정의, 옛 실행기는 2 만): 덜 풀린 걸음 갱신 바꿈을 --y0loc β<1 노드에만 (0 = 몸 전체, 소스 그대로)")
ap.add_argument("--ls0_keep", action="store_true", help="절 CM31 (새 실행기와 같은 정의): 선탐색·장벽 보폭 0 반복은 η·y 그대로, 진단 fem.jsonl l0 (끄면 소스 그대로)")
ap.add_argument("--ue_bigT", type=float, default=1e6, help="절 CM35 --ue_loc 비트 32 문턱: 덜 풀린 걸음 ‖g‖²/F² 가 이 값 넘으면 몸 전체 후퇴 오일러")
ap.add_argument("--ue_dil", type=int, default=0, help="절 CM35 (--ue_loc 위, lib/anchorflow/fem_common2.py): 덜 풀린 걸음 후퇴 오일러 노드 집합 S 를 27 이웃으로 이 번 넓힘 (0 = 끔, 소스 그대로)")
ap.add_argument("--ue", choices=["", "be", "a0", "beall"], default="", help="절 CM24 (2)·(3) (사용자 승인 정의 변경, HHT 판): 덜 풀린 걸음 (같은 판정) 의 걸음 끝 갱신 -- be = 후퇴 오일러 (U1 = y, v1 = (y − U)/h, 가속도 기억 0), "
                "a0 = 속도·위치는 HHT 그대로·가속도 기억만 0, beall = 모든 걸음 후퇴 오일러 갱신 (비교). 빈 값 = 끔")
ap.add_argument("--uf_res", action="store_true", help="절 CM24 판정 2: 덜 풀린 걸음 조건에 '마지막으로 돈 뉴턴 반복 시작 잔차 ‖g‖ > F (허용치 규칙 척도)' 를 더함 (또는; 끄면 보폭 < 1 만)")
ap.add_argument("--ufrec", action="store_true", help="절 CM24 진단 (결과 불변): 판정 커널만 켜고 걸음별 덜 풀림·보폭 최소를 fem.jsonl uf 에")
ap.add_argument("--erec", action="store_true", help="절 CM24 진단 (결과 불변, 재생 밖): 프레임 끝 입자 운동·중력 위치·SNH 탄성 에너지, 질량중심 z·법선 속도 (fem.jsonl en)")
ap.add_argument("--seedb", type=str, default="", help="절 CM28 (a) (사용자 위임 11-08 정의 변경, 수박): 씨 영역 β -- exe/wm_seedmask.py 출력 npz (GF 러너 판정, 입자 순번) 의 씨 입자 β = --seedb_val, "
                "가우스 점은 팔분 씨 입자 부피 과반이면 씨 β (입자 ys 자리에 β). 빈 값 = 끔")
ap.add_argument("--seedb_val", type=float, default=3e9, help="절 CM28 (a): 씨 영역 β (GF 러너 3e9)")
ap.add_argument("--ljq", choices=["", "p0", "max"], default="", help="절 CM28 (b): 가우스 점 logJp 모으기 -- p0 = 응집력 평균 (평균 sinh(ξ max(−logJp,0)) 의 역, 팔분 입자 모두 ≥ 0 이면 산술 평균), max = 최대. 빈 값 = 산술 평균 그대로")
ap.add_argument("--rm_sub", type=int, default=0, help="절 CM28 (c): 걸음 안 소성 갱신을 가우스 점에서 n 등분 (걸음 시작 변위 -> 시험·끝 변위 선형 보간, 증분마다 RM; 0 = 끔)")
ap.add_argument("--rm_snh", action="store_true", help="절 CM28 (d): NACC RM 판정·투영 식을 SNH 탄성 (--elx snh) 의 키르히호프 p·ŝ 와 맞춤 (Je = SNH 등방 압력의 근)")
ap.add_argument("--ks_rho", type=float, default=-1.0, help="절 CM28 (e): --ksplit 쪼갠 시도 (K > 1) 의 HHT ρ∞ (음수 = 기본 --rho 그대로; 1 = 수치 감쇠 없음)")
ap.add_argument("--wi", choices=["", "node", "jbar", "rec"], default="", help="절 CM29 (조정자 위임 11-08 풀이 방식 변경, 수박 재질 판): 뉴턴 뒤집힘 필터 국소화 -- node = 노드별 보폭 (걸친 가우스 점의 필터 k 만큼만 자름, 확인 라운드 --wi_r, 어기면 전역 보폭), "
                "jbar = 필터 대신 그 반복 필터가 걸 점에 J 로그 장벽 (κ_J = --wi_kj x μ, 프레임 끝까지 유지; --ls_inv 없이), rec = 진단만 (결과 불변). lib/anchorflow/fem_wminv.py")
ap.add_argument("--wi_r", type=int, default=4, help="절 CM29 node: 확인 라운드 수")
ap.add_argument("--wi_kj", type=float, default=1.0, help="절 CM29 jbar: κ_J = 이 값 x μ")
ap.add_argument("--gq_oct", type=float, default=-1.0, help="절 CM30 (a) (조정자 결정 11-08 정의 변경, 수박 --gq_mass --fl_bar 위): 점유 가중 κ -- 가우스 점 부피·질량·운동량 = (1 − κ) 팔분 칸 실제 + κ 칸 합 / 8 (새 실행기 --gq_oct 와 같은 식), 입자 보간에서 빈 팔분 점 뺌; 음수 = 끔. lib/anchorflow/fem_wmsparse.py")
ap.add_argument("--gq_ored", action="store_true", help="절 CM30 (a) 재분배 판: 빈 팔분 점 0, 칸 합을 입자 있는 팔분 점에 똑같이 (운동량도), 입자 보간에서 빈 팔분 점 뺌")
ap.add_argument("--ws_bar", type=int, default=0, help="절 CM30 (b): 장벽 실현 가능 보폭 국소화 -- 관통할 표본점이 걸친 노드만 자름 (뒤집힘 필터는 원래 전역), 확인 라운드 이 수 (어기면 원래 전역 보폭); 0 = 끔")
ap.add_argument("--wi_rec", type=str, default="", help="절 CM29 진단 (결과 불변): 원래 필터가 거는 가우스 점 기록할 프레임 (예 3,4,5; 1 부터) -> fem.jsonl wi·wi_rec_f<프레임>.npz")
ap.add_argument("--ws_fi", action="store_true", help="절 CM32 (조정자 결정 11-08 정의 변경, --ws_bar 위): 뒤집힘 필터도 노드별 국소 (절 CM29 (a)) -- 노드 보폭 시작 0, 필터·장벽 확인 라운드 --ws_bar + 1, 어기면 원래 전역 보폭. lib/anchorflow/fem_wmfrac2.py")
ap.add_argument("--w2rec", action="store_true", help="절 CM32 진단 (결과 불변, --ws_bar 위): 시도 (K 1·2·4·8) 마다 뉴턴 반복별 전역 필터 kg·전역 장벽 kb·노드 k 최대·라운드별 어긴 표본점·가우스 점 수 -> fem.jsonl w2")
ap.add_argument("--ks_it", type=str, default="", help="절 CM28 (e): --ksplit 쪼갠 시도 (K > 1) 의 뉴턴,PCG 상한 (예 8,20; 빈 값 = 기본 그대로)")
ap.add_argument("--uf_ro", action="store_true", help="절 CM33 (a) (조정자 결정 11-08 정의 변경, --uf_res 위): 덜 풀린 걸음 판정을 잔차만으로 (보폭 최소 < 1 조건 뺌). lib/anchorflow/fem_wmen.py")
ap.add_argument("--ue_nl", action="store_true", help="절 CM33 (b) (--ws_bar·--ue be 위): 이 걸음 국소 보폭으로 자른 노드만 걸음 끝 속도 = (U1 − U)/dt (U1 = HHT 위치), 가속도 기억 0")
ap.add_argument("--xit_from", type=int, default=0, help="절 CM33 (c) (--uf_res·허용치 판 위): 반복 이 번째 (0 부터) 시작 잔차를 잰 뒤 그때까지 덜 풀림 판정이 아니면 나머지 반복 건너뜀 (--newton 이 상한); 0 = 끔")
ap.add_argument("--en_mo", action="store_true", help="절 CM33 (d) (--ws_bar 위): 뉴턴 반복마다 증분 포텐셜 Φ (관성 + RM 뒤 SNH 탄성 + 장벽) 가 반복 시작보다 늘면 그 반복 갱신을 반씩 되돌림 (최대 6 번, 그래도 늘면 버림)")
ap.add_argument("--enrec", type=int, default=0, help="절 CM33 진단 (결과 불변, --nrec 위, K 1 HHT): 프레임 1~이 값의 에너지 몫 분해 (P2G·잔차·고착·탄성·장벽·갱신식·커밋·G2P, 뉴턴 반복별 증분 포텐셜) -> fem.jsonl enb")
ap.add_argument("--sp2", type=int, default=0, help="절 CM36 (조정자 승인 11-08, 반올림 수준 커널 재작성): 수박 실행기 커널 비트 (1 칸 스캔 3 단계·2 0 블록 합 생략·4 칸 합 팔분 합·8 점->노드 칸 모으기·16 뒤집힘 필터 f32 증명; 0 = 끔, 글자 그대로). lib/anchorflow/fem_speed2.py")
ap.add_argument("--ks_big", action="store_true", help="절 CM37 (조정자 결정 11-08 정의 변경, --ksplit 위, lib/anchorflow/fem_common3.py): 쪼개기 판정을 '덜 풀린 걸음 중 걸음 끝 잔차 비 ‖g‖²/F² > --ue_bigT (비트 32 와 같은 판정, 비유한 포함)' 인 걸음이 있을 때만으로 (끄면 판정 그대로)")
ap.add_argument("--wrx", type=int, default=0, help="절 CM38 재설계 후보 비트 (lib/anchorflow/fem_wmrx.py; 0 이면 안 부름): 1 고착 상한 (√k < --wrx_skc 이면 0), 2 속도 수준 접촉 (띠 노드 들어가는 법선 이동을 띠 몫만큼 지움)")
ap.add_argument("--wrx_skc", type=float, default=1e-2, help="절 CM38 후보 1 문턱")
ap.add_argument("--wrx_col", type=int, default=0, help="절 CM38: 0 모든 걸음에 켬, 1 무너진 프레임만 (K 1 시도 ‖g‖²/F² > --ue_bigT 이면 되돌리고 켠 채 다시 쪼개기; --ksplit --ks_big 위)")
ap.add_argument("--wrd", default="", help="절 CM38 진단 (결과 불변, lib/anchorflow/fem_wmrd.py): 이 프레임들 (1 부터, 예 3-30 또는 4,8) 을 그래프 밖 같은 커널 순서로 돌리며 뉴턴 반복마다 잔차 분해·PCG 상대 잔차·방향 크기·접선 일관성 -> wrd.jsonl")
ap.add_argument("--wrd_fd", type=int, default=1, help="절 CM38: --wrd 에서 접선 차분 대조 (1) / 끔 (0)")
ap.add_argument("--xg", type=int, default=0, help="절 CM37 (결과 불변, --xit_from 위, lib/anchorflow/fem_common3.py): 프레임 그래프를 걸음마다 반복 --xit_from 시작·판정 뒤·뉴턴 끝에서 나눠 캡처, 뒤 구간은 CT[0] = 0 (안 멈춤) 일 때만 재생 (1); 2 = 프로파일 (늘 재생, 건너뛸 구간 GPU 시간 따로); 0 = 끔 (그래프 그대로)")
ap.add_argument("--wsh_rec", type=int, default=0, help="절 CM40 진단 (결과 불변, lib/anchorflow/fem_wmshape.py): 프레임마다 몸 모양·logJp·응력 통계, 이 프레임까지 가우스 점 통계 -> fem.jsonl wsh (0 = 끔)")
ap.add_argument("--wsh_sv", default="", help="절 CM40 진단: 이 프레임들 (쉼표) 입자 위치·logJp float16 -> wsh/xs_*.npz")
ap.add_argument("--wsh_pl", type=int, default=0, help="절 CM40 후보: 1 = 입자 logJp = 프레임 시작 값 + 가우스 점 증분 보간 (--fp_interp 위; 0 이면 모듈 안 부름)")
ap.add_argument("--ws2", type=int, default=0, help="절 CM41 진단 (결과 불변, lib/anchorflow/fem_wmshape2.py, --rm_sub 위): 커밋 RM 갈래별 logJp 증분·변형 증분 막대 -> fem.jsonl ws2")
ap.add_argument("--ws2_eps", type=float, default=0.0, help="절 CM41 후보 (a): 가우스 점 등분 수 = clamp(ceil(|sym(grad U - grad U0)| / 이 값), 1, --rm_sub) (0 = 끔)")
ap.add_argument("--ws2_yc", type=float, default=0.0, help="절 CM41 후보 (a2): 등분 수 = clamp(ceil((κ|tr Δε| + 2μ|dev Δε|) / (이 값 · p0(logJp))), 1, --rm_sub) (0 = 끔)")
ap.add_argument("--ws2_win", type=int, default=0, help="절 CM41 후보 (c): 첫 접촉부터 이 프레임 수만 --rm_sub 등분, 그 뒤 n = 1 (0 = 끔)")
ap.add_argument("--ws2_fc", default="", help="절 CM41 후보 (b): 'N,C' 첫 접촉 프레임만 뉴턴·PCG 상한 N·C 로 다시, 에너지가 늘면 버림 (--wrx_col 위)")
ap.add_argument("--ws2_fch", type=float, default=0.02, help="절 CM41 (b): 첫 접촉 = 프레임 끝 입자 최저 높이 < 이 값 (m) 이 처음인 프레임")
ap.add_argument("--ws2_fctol", type=float, default=0.01, help="절 CM41 (b): 받는 조건 E 끝 <= E 시작 + 이 값 (J)")
ap.add_argument("--ws2_fcn", type=int, default=1, help="절 CM41 (b): 첫 접촉부터 이 프레임 수")
ap.add_argument("--ws3", type=int, default=0, help="절 CM44 후보 (lib/anchorflow/fem_wmshape3.py): 수렴 안 한 걸음은 커밋에서 소성 상태 확정 안 함 -- 1 = 허용치 εN 판정, 2 = 절 CM24 덜 풀린 걸음 판정 (0 = 끔, 모듈 안 부름)")
ap.add_argument("--ws3_lj", type=int, default=0, help="절 CM44: 1 = 미룸 대신 RM 투영은 하고 손상 변수 (ys·logJp·al) 만 걸음 시작 값 유지")
ap.add_argument("--ws3_win", type=int, default=0, help="절 CM44: 1 = 절 CM41 충격 창 (--ws2_win) 끝난 다음 프레임부터만")
a = ap.parse_args()
if a.wrx:                                                                       # 절 CM38 후보 (소스 바꾸기는 모듈 만들기 전에)
    from anchorflow import fem_wmrx as WRX
    WRX.install()
if a.sp2:                                                                       # 절 CM36: 소스·커널 실행 감싸기 (끄면 안 부름)
    from anchorflow import fem_speed2 as SP2
    SP2.install(a.sp2)
if a.wsh_pl:                                                                    # 절 CM40 후보 (소스·커널 열 감싸기, 끄면 안 부름)
    from anchorflow import fem_wmshape as WSHM
    WSHM.install_pl()
WS2 = None
if a.ws2 or a.ws2_eps > 0 or a.ws2_yc > 0 or a.ws2_fc or a.ws2_win:                                          # 절 CM41 (소스·alloc 감싸기, 끄면 안 부름)
    assert a.rm_sub > 0 and (not a.ws2_fc or (a.wrx and a.wrx_col)), "절 CM41 은 --rm_sub 위 (후보 b 는 --wrx_col 위)"
    from anchorflow import fem_wmshape2 as WS2
    WS2.install(a)
WS3 = None
if a.ws3:                                                                        # 절 CM44 (소스·bind 감싸기, 끄면 안 부름)
    assert not a.ws3_win or a.ws2_win, "--ws3_win 은 --ws2_win 위"
    from anchorflow import fem_wmshape3 as WS3
    WS3.install(a)
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
WBAR = None
if a.fl_bar:                                                                          # 절 CM21: 장벽 바닥 (새 실행기 --fl_bar --fl_bst 1 --y0eb 와 같은 정의)
    assert FL is not None and not a.floor_slip and a.fl_bst == 1 and a.ga_y0ut and not a.det, "--fl_bar 는 --floor vel, --floor_slip 없이, --fl_bst 1, --ga_y0ut, 비결정 판"
    _dh = 0.5 * DXG * a.bar_dhmul                                                    # 절 CM23.4 대조: 기본 (x 1) 이면 같은 값
    WBAR = dict(NR=FL["NR"], PN=FL["PN"], DH=_dh, KS=(float(lam) + 2.0 * float(mu)) / ((0.5 * DXG) if a.bar_dhk else _dh) ** 2 * a.bar_kmul, y0eb=a.y0eb)
    if a.y0loc:                                                                        # 절 CM27 (c)
        WBAR["y0loc"] = int(a.y0loc)
    if a.bar_kmul != 1.0 or a.bar_dhmul != 1.0:
        print(f"[절 CM23.4 대조] 장벽 κ x {a.bar_kmul}, d̂ x {a.bar_dhmul} (κ {'원래 d̂ 로 고정' if a.bar_dhk else '정의대로 d̂ 따름'}) -> d̂ {_dh:.6g} κ/V {WBAR['KS']:.6g}", flush=True)
    print(f"[절 CM21 장벽 바닥] 평면 {FL['PT']} 법선 {FL['NR']} d̂ {0.5 * DXG:.6g} κ/V {WBAR['KS']:.6g} 띠 고착 bst 1 시작점 y0eb {a.y0eb}", flush=True)
else:
    assert not (a.fl_bst or a.y0eb or a.bar_cnt or a.bar_kmul != 1.0 or a.bar_dhmul != 1.0), "--fl_bst·--y0eb·--bar_cnt·--bar_kmul·--bar_dhmul 는 --fl_bar 위"
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
    if a.elx:
        assert MATK == 7, "--elx 는 수박 (watermelon) 재질 판만"
        if a.elx == "jbar":
            PRM[13] = float(a.jbar) * float(mu)
        print(f"[절 CM13] --elx {a.elx}: " + (f"J 로그 장벽 κ_J = {a.jbar} x μ = {float(PRM[13]):.6g} (NHB κ {float(PRM[2]):.6g}, 작은 변형 부피 강성 κ + κ_J)" if a.elx == "jbar" else f"Stable Neo-Hookean μ_s = μ = {float(mu):.6g}, λ_s = λ + μ = {float(lam) + float(mu):.6g}"), flush=True)
    WFD = {}
    if a.seedb or a.ljq or a.rm_sub > 0 or a.rm_snh:                                  # 절 CM28 (a)~(d)
        assert MATK == 7 and not a.dmg and not a.det, "절 CM28 판은 수박 비결정, 손상 없이"
        if a.seedb:
            WFD["seedb"] = True; PRM[11] = float(a.seedb_val)
        if a.ljq:
            WFD["ljq"] = a.ljq; WFD["ljq_xi"] = float(PRD["xi"])
        if a.rm_sub > 0:
            assert a.fp_interp, "--rm_sub 는 --fp_interp 판 (입자 F·상태 = 가우스 점 보간) 만"
            WFD["rmsub"] = int(a.rm_sub); PRM[12] = float(a.rm_sub)
        if a.rm_snh:
            assert a.elx == "snh", "--rm_snh 는 --elx snh 위에만"
            WFD["rmsnh"] = True
        print(f"[절 CM28] {WFD} β 씨 {a.seedb_val if a.seedb else '-'}", flush=True)
    if a.ks_rho >= 0 or a.ks_it:                                                      # 절 CM28 (e): 쪼갠 시도만
        assert a.ksplit > 0 and a.ga == "hht", "--ks_rho·--ks_it 는 --ksplit HHT 판만"
        if a.ks_rho >= 0:
            _r = float(a.ks_rho); _am, _af = 0.0, (1 - _r) / (1 + _r)
            WFD["ks_ga"] = dict(am=_am, af=_af, gm=0.5 - _am + _af, be=0.25 * (1 - _am + _af) ** 2)
        if a.ks_it:
            assert not a.nrec, "--ks_it 는 --nrec 없이 (반복 기록 버퍼가 기본 뉴턴 수)"
            WFD["ks_nt"], WFD["ks_cg"] = [int(q) for q in a.ks_it.split(",")]
        print(f"[절 CM28 (e)] 쪼갠 시도 설정 {WFD}", flush=True)
    if a.wi or a.wi_rec:                                                              # 절 CM29
        assert MATK == 7 and WBAR is not None and not a.det, "--wi 는 수박 장벽 바닥 비결정 판"
        assert a.wi != "node" or a.ls_inv, "--wi node 는 --ls_inv 위 (그 필터를 국소로)"
        assert a.wi != "jbar" or (not a.ls_inv and a.elx == "snh"), "--wi jbar 는 --ls_inv 없이, --elx snh 위"
        WFD["wi"] = dict(mode=("" if a.wi == "rec" else a.wi), R=int(a.wi_r), kj=float(a.wi_kj) * float(mu), rec=bool(a.wi_rec), rec_frames=tuple(int(q) for q in a.wi_rec.split(",") if q))
        print(f"[절 CM29] {WFD['wi']}", flush=True)
    if a.gq_oct >= 0 or a.gq_ored or a.ws_bar > 0:                                     # 절 CM30
        assert MATK == 7 and WBAR is not None and a.gq_mass and not a.det, "절 CM30 은 수박 --gq_mass --fl_bar 비결정 판"
        assert a.ws_bar == 0 or a.ls_inv, "--ws_bar 는 --ls_inv 위 (필터는 전역 그대로, 장벽만 국소로)"
        WFD["ws"] = dict(oct=(float(a.gq_oct) if a.gq_oct >= 0 else None), red=bool(a.gq_ored), bar=int(a.ws_bar))
        if a.ws_fi or a.w2rec:                                                        # 절 CM32 (끄면 키 없음 -- 그대로)
            assert a.ws_bar > 0, "--ws_fi·--w2rec 는 --ws_bar 위"
            WFD["ws"].update(fi=bool(a.ws_fi), rec=bool(a.w2rec))
        print(f"[절 CM30] {WFD['ws']}", flush=True)
    if a.uf_ro or a.ue_nl or a.xit_from > 0 or a.enrec > 0 or a.en_mo:                             # 절 CM33 (끄면 키 없음 -- 그대로)
        assert MATK == 7 and WBAR is not None and not a.det and a.ga == "hht", "절 CM33 은 수박 장벽 바닥 HHT 비결정 판"
        assert not a.uf_ro or (a.uf_res and (a.ue or a.ufrec or a.ksplit > 0)), "--uf_ro 는 --uf_res 판정 위"
        assert not a.ue_nl or (a.ue == "be" and a.ws_bar > 0), "--ue_nl 은 --ue be --ws_bar 위"
        assert a.xit_from <= 0 or (a.uf_res and a.tolN >= 0 and a.newton > a.xit_from and (a.ksplit == 0 or a.ks_big)), "--xit_from 은 --uf_res 허용치 판, --newton > 이 값, 쪼개기 없이 (절 CM37 --ks_big 은 허용)"
        assert a.enrec <= 0 or (a.nrec and a.K == 1 and a.ksplit == 0), "--enrec 는 --nrec 위, K 1, 쪼개기 없이"
        assert not a.en_mo or (a.ws_bar > 0 and a.elx == "snh"), "--en_mo 는 --ws_bar --elx snh 위"
        WFD["en"] = dict(ro=bool(a.uf_ro), nl=bool(a.ue_nl), xf=int(a.xit_from), diag=a.enrec > 0, mo=bool(a.en_mo))
        print(f"[절 CM33] {WFD['en']}", flush=True)
    DMGI = None
    if a.dmg:                                                                         # 절 CM19: 손상 매개변수를 GF 수박 재질에서 유도 (정답 궤적에 맞추지 않음)
        assert MATK == 7 and a.elx == "snh" and not a.det, "--dmg 는 수박 --elx snh, 비결정 판만"
        _mu, _la, _ka, _M, _be, _xi = float(mu), float(lam), float(PRD["kappa"]), float(PRD["M"]), float(PRD["beta"]), float(PRD["xi"])
        _ls = _la + _mu; _p0 = _ka * (1e-5 + math.sinh(_xi * max(-float(PRD["logJp0"]), 0.0)))

        def _psip(sg):                                                                # dmg_psip 와 같은 식 (f64)
            J = sg[0] * sg[1] * sg[2]
            ps = 0.5 * _mu * (sum(q * q for q in sg) - 3.0) - _mu * (J - 1.0) + 0.5 * _ls * (J - 1.0) ** 2
            return ps - (0.5 * _ka * (J - 1.0) ** 2 if J < 1.0 else 0.0)

        def _yl(sg):                                                                  # pl_rm 재질 7 의 항복 함수 y (NHB 식)
            J = sg[0] * sg[1] * sg[2]; B = [q * q for q in sg]; Bm = sum(B) / 3.0
            sh = [_mu * J ** (-2.0 / 3.0) * (q - Bm) for q in B]; pt = 0.5 * _ka * (1.0 - J * J)
            return 1.5 * (1 + 2 * _be) * sum(q * q for q in sh) + _M * _M * (pt + _be * _p0) * (pt - _p0)

        def _bis(fn, lo, hi):
            flo = fn(lo)
            for _ in range(200):
                md = 0.5 * (lo + hi); fm = fn(md)
                if (fm > 0) == (flo > 0):
                    lo, flo = md, fm
                else:
                    hi = md
            return 0.5 * (lo + hi)
        _Je = math.sqrt(1.0 + 2.0 * _be * _p0 / _ka)
        PS_T = _psip([_Je ** (1.0 / 3.0)] * 3)                                       # (가) 정수압 인장 꼭짓점

        def _tau(sg):                                                                 # NHB 키르히호프 주값 (옆 응력 0 조건)
            J = sg[0] * sg[1] * sg[2]; B = [q * q for q in sg]; Bm = sum(B) / 3.0
            return [_mu * J ** (-2.0 / 3.0) * (q - Bm) + 0.5 * _ka * (J * J - 1.0) for q in B]
        _lat = lambda x: _bis(lambda b: _tau([x, b, b])[1], 0.5, 1.5)
        _xu = _bis(lambda x: _yl([x, _lat(x), _lat(x)]), 1.0001, 1.5); _bu = _lat(_xu)
        PS_U = _psip([_xu, _bu, _bu])                                                 # (나) 단축 인장 항복
        _lsh = _bis(lambda x: _yl([x, 1.0 / x, 1.0]), 1.0001, 1.5)
        PS_S = _psip([_lsh, 1.0 / _lsh, 1.0])                                         # 참고: 순수 전단 (p = 0) 항복 = 탄성 영역 안 ψ⁺ 최대 근처
        PSC = PS_T if a.dmg_psic == "tens" else PS_U
        CELL0 = float(lat.A[:, 0].norm()); L0 = a.pf_l0 * CELL0
        GC = 512.0 / 27.0 * PSC * L0                                                 # AT2 1 차원 강도 σ_c = (9/16) √(E G_c / (3 l0)) 를 σ_c = √(2 E ψ_c) 에 맞춤
        PRM[10] = float(PSC if a.dmg == "cdm" else GC); PRM[11] = float(L0)
        DMGI = dict(kind=a.dmg, basis=a.dmg_psic, psi_tens=PS_T, psi_uni=PS_U, psi_shear=PS_S, psic=PSC, Je=_Je, p0=_p0, uni_stretch=[_xu, _bu], shear_stretch=_lsh, cell=CELL0, l0=L0, Gc=GC, pf_it=a.pf_it,
                    pf_peakH=GC / (6.0 * L0), pf_dmax_at_psishear=(2 * PS_S) / (2 * PS_S + GC / L0))
        print(f"[절 CM19] --dmg {a.dmg} 근거 {a.dmg_psic}: ψ⁺ 정수압 인장 꼭짓점 {PS_T:.4f} (Je {_Je:.5f}) · 단축 인장 항복 {PS_U:.4f} (늘임 {_xu:.5f}, {_bu:.5f}) · 순수 전단 항복 {PS_S:.4f}, p0 {_p0:.4f} -> ψ_c {PSC:.4f}"
              + (f", 위상장 l0 {L0:.5f} m (칸 {CELL0:.5f} x {a.pf_l0}), G_c = 512/27 ψ_c l0 = {GC:.5g}, 반복 {a.pf_it}" if a.dmg == "pf" else ""), flush=True)
    VISC = 0.0
    if a.visc_rpic and float(cfg.get("rpic_damping", 0.0)) > 0:
        _r = float(cfg["rpic_damping"])
        VISC = float(cfg["density"]) * _r * (2 - _r) * DXG ** 2 / (16.0 * float(cfg["substep_dt"]))
    print(f"[절 AQ 재질 판] 재질 {cfg.get('material')} ({MATK}) 항복 {PRD['yield_stress']} logJp0 {PRD['logJp0']} 등가 점성 {VISC:.4g} 미끄럼 {a.floor_slip} 구속 속도 {PVB is not None}", flush=True)
    S = AORT.AORTSolver(d["x"][0].to(dev), d["v"][0].to(dev), F0, m64, vol64, lat.A, lat.Ai, mu, lam, [float(q) for q in cfg["g"]], h, a.K, a.integ, a.newton, a.cg, 0,
                        floor=FL, damp=DAMP, bc=BC, cdp=1.0, trel=TREL, pbc_ns=a.pbc_ns, dt_gt=float(cfg["substep_dt"]), mthr=MTHR, compact=not a.no_compact, warm=not a.no_warm, det=a.det,
                        hspec=True, theta=a.theta, ga=(dict(kind=a.ga, rho=a.rho, y0uk=not a.ga_y0ut, ar=not a.ga_noar) if a.ga else None),
                        mat=MATK, prm=PRM, eta=VISC, pvb=PVB, ys0=PRD["yield_stress"], lj0=(PRD["logJp0"] if MATK == 7 else 0.0), slip=a.floor_slip, tolN=a.tolN, tolP=a.tolP, vfill=not a.no_vfill, lsinv=a.ls_inv, y0z=a.y0z, y0u=a.y0u, y0l=a.y0l, fqrl=a.fq_rl, lsend=a.ls_end, mg_levels=a.mg_levels, mg_nc=a.mg_nc, mg_pow=a.mg_pow, gqm=a.gq_mass, elx=a.elx, dmg=a.dmg, pf_it=a.pf_it, fpi=a.fp_interp, wbar=WBAR, lji=a.lj_interp, bcnt=a.bar_cnt,
                        uf=(dict(judge=True, ue=a.ue, res=a.uf_res, uel=a.ue_loc, ls0=a.ls0_keep, **({"dil": a.ue_dil} if a.ue_dil > 0 else {}), **({"bigT": a.ue_bigT} if a.ue_loc & 32 else {})) if (a.ksplit > 0 or a.ue or a.ufrec) else (dict(ls0=True) if a.ls0_keep else None)), wf=WFD)
    if a.seedb:                                                                       # 절 CM28 (a): 입자 β (원래 순번 -> 풀이 순서 perm)
        _sd = np.load(a.seedb)
        _seed = torch.from_numpy(_sd["seed"].astype(np.bool_))
        assert _seed.numel() == Np, ("씨 마스크 길이", _seed.numel(), Np)
        _xg = torch.from_numpy(_sd["xyz"]).double(); _x0 = d["x"][0].double()          # 순번 대응 확인: 정답 0 프레임 위치 = 배율 x 원래 3DGS 좌표 + 이동 (최소제곱)
        _xc, _gc = _x0 - _x0.mean(0), _xg - _xg.mean(0)
        _sc = float((_xc * _gc).sum() / (_gc * _gc).sum())
        _res = (_xc - _sc * _gc).norm(dim=1)
        print(f"[절 CM28 (a)] 씨 입자 {int(_seed.sum())} / {Np} ({100 * float(_seed.double().mean()):.2f} %), 순번 대응: 배율 {_sc:.5f}, 잔차 최대 {float(_res.max()):.3e} 중앙 {float(_res.median()):.3e} (장면 단위)", flush=True)
        _bp = torch.where(_seed, torch.tensor(float(a.seedb_val), dtype=torch.float64), torch.tensor(float(PRD["beta"]), dtype=torch.float64)).float()
        S.ysp.copy_(_bp.to(dev)[S.perm])
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
assert not a.fp_interp or getattr(S, "fpi", False), "--fp_interp 는 재질 판 (AORTSolver aq 경로) 만"
assert not (a.fl_bar or a.lj_interp) or MATK is not None, "--fl_bar·--lj_interp 는 재질 판 (AORTSolver) 만"
assert not (a.ksplit > 0 or a.ue or a.ufrec) or MATK is not None, "--ksplit·--ue·--ufrec 는 재질 판 (AORTSolver) 만"
if a.ksplit > 0:                                                                      # 절 CM24 (1b): 버퍼는 걸음 수 상한 2^ksplit 으로
    assert a.K == 1 and a.adaptK <= 0 and a.frec <= 0 and DAMP == 1.0 and a.ga, "--ksplit 은 K 1, 고정 걸음, --frec 없이, 감쇠 없는 HHT·GA 판만"
if a.dmg:                                                                             # 절 CM19: 처음 손상 d = 0 (cdm) / 이력 H = 0 (pf) -- 재질 7 은 al·ys 를 안 씀
    S.alp.zero_(); S.ysp.zero_()
if a.nrec:
    assert MATK is not None and a.adaptK <= 0 and a.tolN >= 0, "--nrec 는 재질 판 (AORTSolver), 고정 걸음, 허용치 판만"
    S.nrec_on()
    print("[절 CM23.3 진단] --nrec 켬 (뉴턴 반복별 |g|/F·PCG·보폭·마지막 뉴턴 뒤 |g|/F, 결과 불변)", flush=True)
if a.frec > 0:
    assert a.nrec and a.K == 1 and a.ga == "hht", "--frec 는 --nrec 위, K 1, HHT 만"
    S.frec_on()
    from anchorflow import fem_wmkick_diag as FRD
    FRC = FRD.FrecCtx(S, a, FL, WBAR, m64, vol64, float(mu), float(lam), [float(q) for q in cfg["g"]], float(lat.A[:, 0].norm()), MTHR,
                      DM.traj_F(d, h, dev) if not a.F0_eye else None, d["x"])
    print(f"[절 CM23.4 진단] --frec {a.frec} 켬 (노드 힘 몫·속도 변화 분해·장벽 표본점·det F·탄성 에너지, 결과 불변)", flush=True)
ENC = None
if a.enrec > 0:                                                                       # 절 CM33 진단 (결과 불변): --frec 사본 버퍼 + 커밋 앞 F_q 사본
    if a.frec <= 0:
        S.frec_on()
    from anchorflow import fem_wmen as _EN
    ENC = _EN.EnCtx(S, FL, WBAR, float(mu), float(lam), [float(q) for q in cfg["g"]], float(lat.A[:, 0].norm()))
    print(f"[절 CM33 진단] --enrec {a.enrec} 켬 (프레임 에너지 몫 분해, 결과 불변)", flush=True)
if a.nan_step:
    assert MATK is not None and a.adaptK <= 0 and not a.det, "--nan_step 은 재질 판 (AORTSolver), 고정 걸음, 비결정 판만"
    S.nsd_on()
    print("[절 CM9.3 진단] --nan_step 켬 (걸음·뉴턴 반복별 처음 비유한·뒤집힘 시점, 결과 불변)", flush=True)
NSD_DUMPS = 0
if a.kmax <= 0:
    a.kmax = int(math.ceil(h / float(cfg["substep_dt"]) - 1e-9))
S.kcap = a.K
if a.ksplit > 0:
    S.kcap = 1 << a.ksplit
    S.alloc(None)
if a.ks_big:                                                                          # 절 CM37: 무너진 걸음만 쪼개기 (문턱 = 비트 32 와 같은 --ue_bigT)
    assert a.ksplit > 0 and a.uf_res and a.tolN >= 0, "--ks_big 은 --ksplit·--uf_res 허용치 판 위"
    S.ksbig = float(a.ue_bigT)
if a.xg:                                                                              # 절 CM37: 나눈 그래프 (결과 불변)
    assert a.xit_from > 0 and a.tolN >= 0, "--xg 는 --xit_from 허용치 판 위"
    from anchorflow import fem_common3 as _C3
    _C3.install(S, a.xg)
if a.w2rec:                                                                           # 절 CM32 진단 버퍼
    from anchorflow import fem_wmfrac2 as _W2
    _W2.rec_alloc(S)
CELL = float(lat.A[:, 0].norm())
print(f"[절 AO 실시간 판] 적응 걸음 수 c {a.adaptK} 상한 {a.kmax}, 칸 {CELL:.4f} m")
print(f"[절 AO 실시간 판] 모래 {SAND} 구속 {a.pbc} 빈 노드 문턱 {MTHR:.3e} (mlow {a.mlow})", flush=True)
print(f"[절 AN 실시간 판] 입자 {Np}, 프레임 {L}, 셀 {float(lat.A[:, 0].norm()):.4f} m (정답 dx {DXG} 의 {a.cell} 배), K {a.K} {a.integ}, 뉴턴 {a.newton} x PCG {a.cg}, 선탐색 {a.ls}, MG 층 {a.mg_levels}, "
      f"바닥 {a.floor}, 구속 {len(BC) if BC else 0}", flush=True)
flog = open(os.path.join(a.out, "fem.jsonl"), "w")
_v0 = d["v"][0].double()
print(f"[처음 속도] 평균 {[round(float(q), 4) for q in _v0.mean(0)]} 표준편차 {[round(float(q), 4) for q in _v0.std(0)]} 최대 편차 {float((_v0 - _v0.mean(0)).norm(dim=-1).max()):.4f}", flush=True)
if a.wrx:                                                                             # 절 CM38 후보 장치 전역 (col 이면 처음 0, 무너진 프레임 다시 할 때만 1)
    WRX.setup(S, a.wrx, a.wrx_skc); WRX.on(S, 0 if a.wrx_col else 1)
WRD, WRD_F = None, set()
if a.wrd:                                                                             # 절 CM38 진단
    from anchorflow import fem_wmrd as _WRD
    for _p in a.wrd.split(","):
        _q = [int(q) for q in _p.split("-")]
        WRD_F.update(range(_q[0], _q[-1] + 1))
    WRD = _WRD.Rec(S, os.path.join(a.out, "wrd.jsonl"), fd=bool(a.wrd_fd))
WSH = None
if WS2 is not None:                                                                   # 절 CM41
    WS2.setup(S, mu, lam, cfg["g"], FL)
if WS3 is not None:                                                                   # 절 CM44
    WS3.setup(S)
if a.wsh_rec:                                                                         # 절 CM40 진단 (읽기만)
    from anchorflow import fem_wmshape as WSHM
    WSH = WSHM.Rec(S, a.out, mu, lam, float(PRD["xi"]), d["x"][0], a.wsh_rec, a.wsh_sv)
errs, ms = [], []
ITS = []
div_at = None
PRED = torch.empty(L, Np, 3) if a.dump else None
FDMP = torch.empty(L, Np, 3, 3) if a.dump else None
LJD = torch.empty(L, Np) if a.dump_lj else None
DD = torch.empty(L, Np) if (a.dump_lj and a.dmg) else None
for fi in range(L):
    kp = S.profile(fi * h) if (a.prof and fi >= a.warm and (fi - a.warm) % a.prof == 0) else None
    _xprev = S.x_orig().clone() if a.nan_step else None                               # 절 CM9.3: 이 프레임 시작 (= 직전 프레임 끝) 입자 위치, 원래 순서
    if ENC is not None and fi + 1 <= a.enrec:                                          # 절 CM33: 프레임 시작 입자 에너지 (재생 밖)
        ENC.pre()
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
        elif WRD is not None and fi + 1 in WRD_F and WRD.frame(fi, fi * h):           # 절 CM38 진단 (그래프 밖, 같은 커널 순서; 넘치면 False -> 아래 보통 경로)
            redo, KF, VMX = False, a.K, None
        elif a.nanchk and fi + 1 == a.nanchk:
            if S.graph is None:
                _bk0 = S._bk(); S.capture(); S._rs(_bk0)
            _hit = S.nanchk(fi * h); S.last_ms = float("nan")
            print(f"[비유한 검사] 프레임 {fi + 1}: {json.dumps(_hit)}", flush=True)
            redo, KF, VMX = False, a.K, None
        else:
            if a.gq_chk:                                                               # 절 CM9: 프레임 시작 입자 합 (f64)
                _m = S.m.double(); _pv = S.v.double(); _pp = S.pa.double() if S.pa is not None else torch.zeros_like(_pv)
                _ps = [float(_m.sum()), (_m[:, None] * _pv).sum(0), (_m[:, None] * _pp).sum(0), float((_m * _pv.norm(dim=-1)).sum()), float((_m * _pp.norm(dim=-1)).sum())]
            if a.ksplit > 0:                                                           # 절 CM24 (1b): 덜 풀린 걸음이 있으면 K 두 배로 프레임 다시 (시간 = 모든 시도의 재생 GPU 시간 합)
                if a.wrx and a.wrx_col:                                                # 절 CM38: 무너진 프레임만 후보 켬
                    redo, KF, _ktr4 = WRX.split_col(S, fi * h, h, float(cfg["substep_dt"]), a.ksplit); VMX = None
                    _ktr = [q[:3] for q in _ktr4]; WRX_TR = _ktr4
                else:
                    redo, KF, _ktr = AORT.split_step(S, fi * h, h, float(cfg["substep_dt"]), a.ksplit); VMX = None
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
    if a.gq_chk and '_ps' in dir() and _nn > 0:                                       # 노드 합 (이 프레임 P2G 결과, 풀이 중 안 바뀜; 가속도 상태 gaA 는 걸음마다 갱신되어 여기서는 못 잼 -> 새 실행기 판에서)
        _mn = S.mn[:_nn].double()
        c["gqchk"] = [abs(float(_mn.sum()) - _ps[0]) / _ps[0], float((S.mv[:_nn].double().sum(0) - _ps[1]).norm()) / max(_ps[3], 1e-30), None]
    if _nn > 0 and int(S.IP[13]) == 0:                                                 # 발산 진단: 노드 최대 변위 (칸 단위) 와 그 노드 질량 / 중앙값, 입자 최대 속력
        _ua = S.U[:_nn].norm(dim=-1); _ui = int(_ua.argmax()); _mm = S.mn[:_nn]; _md = float(_mm[_mm > 0].median())
        c.update(umax_cell=float(_ua[_ui]) / CELL, umax_mrel=float(_mm[_ui]) / _md, vpmax=float(S.v.norm(dim=-1).max()), mrel_min=float(_mm[_mm > 0].min()) / _md)
    if a.nan_step:
        c["nsd"] = S.nsdo.frame_counts()
        if (c["nsd"]["nN"] > 0 or not fin) and NSD_DUMPS < 3:                          # 노드 변위 비유한이 생긴 프레임 (발산 프레임 포함, 처음 3 개)
            _pz = os.path.join(a.out, f"nan_step_f{fi + 1}.npz")
            try:
                S.nsdo.dump(_pz, _xprev, extra=dict(frame=np.array([fi + 1])))
                print(f"[절 CM9.3 덤프] {_pz} {json.dumps(c['nsd'])}", flush=True)
            except Exception as _ex:
                import traceback; traceback.print_exc(); print(f"[절 CM9.3 덤프 실패] {_ex!r}", flush=True)
            NSD_DUMPS += 1
    if a.qjd and int(S.IP[13]) == 0:                                                  # 절 CM20 진단 (결과 불변): 프레임 끝 가우스 점 J (새 실행기 rt_qjd 와 같은 커널·같은 버퍼 배치)
        from anchorflow import fem_an_rt as _ARTN
        c["qj"] = _ARTN.RTSolver.qjd(S, [0.0, 0.0, 1.0], 0.0)
    if a.jdiag and int(S.IP[13]) == 0:                                                # 절 CM13 진단 (결과 불변): 프레임 끝 커밋된 F_q·입자 F 의 행렬식
        _nq = int(S.IP[12]); _vq = S.Vq[:_nq] > 0
        _dq = torch.linalg.det(S.Fq[:_nq].reshape(-1, 3, 3).double())[_vq]
        _dp = torch.linalg.det(S.F.reshape(-1, 3, 3).double())
        _fin = torch.isfinite(_dp)
        c["jd"] = dict(q_le0=int((_dq <= 0).sum()), q_nf=int((~torch.isfinite(_dq)).sum()), q_min=float(_dq[torch.isfinite(_dq)].min()) if bool(torch.isfinite(_dq).any()) else None,
                       p_min=float(_dp[_fin].min()) if bool(_fin.any()) else None, p_le0=int((_dp <= 0).sum()), p_lt1e3=int((_dp < 1e-3).sum()), p_nf=int((~_fin).sum()))
    if FL is not None and int(S.IP[13]) == 0:                                         # 절 CM21 지표 (결과 불변): 바닥 아래 입자 최대 깊이·수 (평면 부호 거리 < 0), 장벽 걸린 표본점, 걸음별 시작점 k
        _sd = xo.double() @ torch.tensor(FL["NR"], device=xo.device, dtype=torch.float64) - FL["PN"]
        _sd = _sd[torch.isfinite(_sd)]
        c["fd"] = dict(depth=float(max(0.0, -float(_sd.min()))) if _sd.numel() else None, nbelow=int((_sd < 0).sum()))
        if WBAR is not None:
            if a.bar_cnt:
                _nq = int(S.IP[12]); _bf = S.bcf[:_nq]; _bm = S.bx[:4 * _nq].view(-1, 4)[:, 3] > 0
                c["bc"] = dict(hit=int(((_bf & 1) > 0).sum()), nsamp=int(_bm.sum()), hit_cells=int(torch.unique(torch.nonzero((_bf & 1) > 0).flatten() // 8).numel()))
            c["y0k"] = [int(q) for q in S.y0k[:S.K].cpu()]
            if getattr(S, "ykn", None) is not None:                                        # 절 CM27 (c) 진단
                c["yl"] = S.YLS[:S.K, :5 + int(a.y0loc)].cpu().tolist()
    if a.dmg:                                                                         # 절 CM19 (결과 불변): 입자 손상 d (cdm = alp, pf = ysp 노드 d 보간), logJp, 이력 H (pf = alp)
        _okp = torch.isfinite(S.x).all(-1); _dp = (S.alp if a.dmg == "cdm" else S.ysp)[_okp]; _lj = S.ljp[_okp]; _nn_ = max(int(_okp.sum()), 1)
        c["dmg"] = dict(d05=float((_dp > 0.5).sum()) / _nn_, d01=float((_dp > 0.1).sum()) / _nn_, dmean=float(_dp.nan_to_num(0).mean()) if _dp.numel() else None, dmax=float(_dp.nan_to_num(0).max()) if _dp.numel() else None,
                        lj0=float((_lj > 0).sum()) / _nn_, uni=float(((_dp > 0.5) | (_lj > 0)).sum()) / _nn_)
        if a.dmg == "pf":
            _H = S.alp[_okp]
            c["dmg"].update(Hmax=float(_H.nan_to_num(0).max()) if _H.numel() else None, Hmean=float(_H.nan_to_num(0).mean()) if _H.numel() else None, pfres=[float(q) for q in S.pfres[:a.K].cpu()])
    if a.nrec and int(S.IP[13]) == 0:                                                # 절 CM23.3 (결과 불변): 걸음 k·뉴턴 i 별 기록
        _sc = S.NRsc.cpu().numpy(); _c3 = S.NRct3.cpu().numpy(); _c7 = S.NRct7.cpu().numpy(); _cd = S.NRcd.cpu().numpy(); _st = S.NRst.cpu().numpy(); _fn = S.NRfin.cpu().numpy()
        _nt = S.nt; _rows = []; _gend = []; _KN = S.K
        _si = S.NRsi.cpu().numpy(); _kb = S.NRkb.cpu().numpy(); _k0 = S.NRk0.cpu().numpy(); _y0b = S.NRy0b.cpu().numpy()
        for _k in range(_KN):
            _F2 = float(_cd[_k * _nt][2]); _F = math.sqrt(_F2) if _F2 > 0 else float("nan")
            for _i in range(_nt):
                _j = _k * _nt + _i
                _ran = int(_c3[_j][0]) == 0
                _stp = (not _ran) and (_i == 0 or int(_c3[_j - 1][0]) == 0)
                _g = math.sqrt(max(float(_sc[_j][1]), 0.0)) if (_ran or _stp) else None
                _r = dict(k=_k, i=_i, ran=_ran, stop=bool(_stp), gF=(_g / _F if _g is not None else None))
                if _ran:
                    _r.update(nP=int(_c7[_j][3] - _c3[_j][3]), capP=int(_c7[_j][5] - _c3[_j][5]), st=float(_st[_j]), si=float(_si[_j]), kb=int(_kb[_j]))
                _rows.append(_r)
            _gend.append(math.sqrt(max(float(_fn[_k][0]), 0.0)) / _F)
        c["nr"] = dict(F=[math.sqrt(max(float(_cd[_k * _nt][2]), 0.0)) for _k in range(_KN)], it=_rows, gend=_gend,
                       conv=[bool(any(r_["stop"] for r_ in _rows if r_["k"] == _k) or _gend[_k] <= a.tolN) for _k in range(_KN)],
                       k0=[int(q) for q in _k0[:_KN]], y0b=[float(q) for q in _y0b[:_KN]])
    if a.nrec:                                                                        # 질량중심 z (우리·정답, 정답 비유한 입자는 둘 다 뺌)
        _w = m64[_ok].double(); _w = _w / _w.sum()
        c["com"] = dict(z=float((_w * xo[_ok][:, 2].double()).sum()), gz=float((_w * gt[_ok][:, 2].double()).sum()))
    if getattr(S, "ufj", False):                                                     # 절 CM24 (결과 불변): 걸음별 덜 풀림·보폭 최소 (마지막 시도)
        c["uf"] = dict(fail=[int(q) for q in S.ufl[:S.K].cpu()], smin=[float(q) for q in S.ufs[:S.K].cpu()], g2=[float(q) for q in S.ufg[:S.K].cpu()] if S.ufr else None)
        if getattr(S, "uel", 0):
            c["uf"]["ul"] = S.ULS[:S.K, :3].cpu().tolist()
    if a.ls0_keep:                                                                     # 절 CM31: 이 프레임 보폭 0 반복 수·δ 비유한 노드 수 (읽고 0)
        _g = S.mod.get_global("AF_L0N"); _a = AORT.ART.cp.ndarray((2,), dtype=AORT.ART.cp.int32, memptr=_g); c["l0"] = [int(q) for q in _a.get()]; _a[:] = 0
    if a.ksplit > 0:
        c["ks"] = dict(K=KF, tries=[list(q) for q in _ktr])
        if a.wrx and a.wrx_col:                                                       # 절 CM38: 시도별 후보 켬
            c["ks"]["on"] = [q[3] for q in WRX_TR]
    if a.xg:                                                                          # 절 CM37: 조건 구간 [건너뜀, 돎] (+ --xg 2 GPU ms [늘, 돈 조건, 건너뛸 조건])
        c["xg"] = _C3.frame_stats(S)
    if a.wi or a.wi_rec:                                                             # 절 CM29 (결과 불변 읽기): 국소 필터 통계 (마지막 시도), 기록 프레임 요약
        from anchorflow import fem_wminv as _WI
        c["wi"] = dict(st=_WI.frame_stats(S))
        if S.wi_att:
            _mm = S.mn[:int(S.IP[9])]; _md = float(_mm[_mm > 0].median())
            _ci = S.cidx[S.pcd.long()].long(); _ci = _ci[_ci >= 0]
            _npc = torch.bincount(_ci, minlength=int(S.IP[12]) // 8 + 1).cpu().numpy()
            _ci2 = S.cidx[S.pcd.long()].long(); _ok3 = (_ci2 >= 0) & torch.isfinite(S.x).all(-1)        # 칸 입자 높이 평균 (프레임 끝 위치; 칸 번호는 이 프레임 것)
            _hz = torch.zeros(len(_npc), device=S.x.device, dtype=torch.float64).index_add_(0, _ci2[_ok3], S.x[_ok3].double() @ torch.tensor(FL["NR"], device=S.x.device, dtype=torch.float64))
            _hc = (_hz.cpu().numpy() / np.maximum(_npc, 1)) - float(FL["PN"])
            c["wi"]["rec"] = [_WI.rec_summary(_t, _md, _npc, float(FL["PN"]), FL["NR"], CELL, _hc) for _t in S.wi_att]
            np.savez_compressed(os.path.join(a.out, f"wi_rec_f{fi + 1}.npz"), **{f"K{_t['K']}_rows": _t["rows"] for _t in S.wi_att}, **{f"K{_t['K']}_ck": _t["ck"] for _t in S.wi_att})
            S.wi_att.clear()
    if a.ws_bar > 0:                                                                  # 절 CM30 (b) (결과 불변 읽기): 마지막 시도 통계
        from anchorflow import fem_wmsparse as _WS
        c["ws"] = dict(st=_WS.frame_stats(S))
        if a.w2rec:                                                                   # 절 CM32 진단: 이 프레임 모든 시도
            from anchorflow import fem_wmfrac2 as _W2
            c["w2"] = _W2.frame_rec(S, _ktr if a.ksplit > 0 else [(KF, None, None)])
    if a.erec and int(S.IP[13]) == 0:                                                # 절 CM24 (결과 불변, 읽기만): 입자 에너지·질량중심
        from anchorflow import fem_wmkick_diag as _FRD
        _mp = S.m.double(); _xp = S.x.double(); _vp = S.v.double(); _ok2 = torch.isfinite(_xp).all(-1) & torch.isfinite(_vp).all(-1)
        _gv = torch.tensor([float(q) for q in cfg["g"]], device=_xp.device, dtype=torch.float64)
        _nv = torch.tensor(FL["NR"] if FL else [0.0, 0.0, 1.0], device=_xp.device, dtype=torch.float64)
        _Ee = _FRD._snh(S.F[_ok2], S.vol[_ok2], float(mu), float(lam))[0] if a.elx == "snh" else None
        _M = float(_mp[_ok2].sum())
        c["en"] = dict(KE=float(0.5 * (_mp[_ok2] * (_vp[_ok2] ** 2).sum(1)).sum()), PE=float(-(_mp[_ok2] * (_xp[_ok2] @ _gv)).sum()), Eel=_Ee,
                       zc=float((_mp[_ok2] * (_xp[_ok2] @ _nv)).sum()) / _M, vz=float((_mp[_ok2] * (_vp[_ok2] @ _nv)).sum()) / _M, nfin=int(_ok2.sum()),
                       lj0=(float((S.ljp[_ok2] > 0).double().mean()) if MATK == 7 else None))     # 손상 몫 = logJp > 0 (분모 = 위치 유한 입자, exe/wmdmg_stats.py 와 같은 정의)
        if MATK == 7 and fi + 1 <= 12:                                               # 절 CM28 (1) 첫 접촉 비교 통계 (읽기만): J 분포·SNH/NHB 탄성 에너지·바닥 띠·logJp 분포
            from anchorflow import wmfrac_stats as _WFS
            c["en"]["fs"] = _WFS.frame_stats(S.F, S.vol, S.x, S.ljp, float(mu), float(lam), FL["NR"] if FL else (0.0, 0.0, 1.0), float(FL["PN"]) if FL else 0.0)
    if a.frec > 0 and fi + 1 <= a.frec and int(S.IP[13]) == 0:                        # 절 CM23.4 (결과 불변, 읽기만)
        try:
            c["fr"] = FRC.frame(fi + 1)
        except Exception as _ex:
            import traceback; traceback.print_exc(); c["fr"] = dict(err=repr(_ex))
    if a.en_mo:                                                                       # 절 CM33 (d) (읽기만): 이 프레임 되돌린 횟수·버린 반복 수
        c["enmo"] = [int(q) for q in S.ENmo[1:3].cpu()]
    if ENC is not None and fi + 1 <= a.enrec and int(S.IP[13]) == 0:                  # 절 CM33 (결과 불변, 읽기만)
        try:
            c["enb"] = ENC.frame(fi + 1)
        except Exception as _ex:
            import traceback; traceback.print_exc(); c["enb"] = dict(err=repr(_ex))
    if WSH is not None:                                                               # 절 CM40 진단 (읽기만)
        c["wsh"] = WSH.frame(fi)
    if WS2 is not None:                                                               # 절 CM41 진단 (읽기만)
        c["ws2"] = WS2.frame(fi)
    if WS3 is not None:                                                               # 절 CM44 (읽기 + 창 뒤 켜기)
        c["ws3"] = WS3.frame(fi)
    if a.diag:
        c["diag"] = S.diag()
    if "it" in c:
        ITS.append(c["it"])
    flog.write(json.dumps(dict(t=fi, err=e, ms=S.last_ms, wall_ms=WALL, K=KF, vmax=VMX, redo=redo, **c, **({"kernel_ms": kp} if kp else {}))) + "\n"); flog.flush()
    print(f"  프레임 {fi + 1}/{L} K {KF} v_max {VMX if VMX is None else round(VMX, 3)} 벽시계 {WALL:.1f} ms 재생 {S.last_ms:.2f} ms 오차 {100 * e:.3f}%{' (용량 늘려 다시)' if redo else ''} 점 {c['nq']} 노드 상자 {c['nn']} 유클리드 평균 점 {c['nbad']} 꼬리·꼭짓점 점 {c.get('q_tail')}·{c.get('q_apex')} 입자 RM {c.get('p_rm')}{(' 보폭(줄임·최대k·뒤집힌점·호출) ' + str(c.get('ls_inv')) + ' 시작 ' + str(c.get('y0z'))) if 'ls_inv' in c else ''}{(' 손상 ' + json.dumps(c['dmg'])) if 'dmg' in c else ''}", flush=True)
    if a.bench_at and str(fi + 1) in a.bench_at.split(","):
        print(f"  [벤치] 프레임 {fi + 1} 점 {int(S.IP[12])} {json.dumps(S.bench(['ao_mv_c', 'ao_mv_q', 'ao_mv_q_na', 'ao_mv_q_ng', 'ao_mv_q_nh', 'ao_elem', 'ao_elem_s']))}", flush=True)
    if a.check_hspec and str(fi + 1) in a.check_hspec.split(","):
        print(f"  [대조 hspec] 프레임 {fi + 1} {json.dumps(S.check_hspec(fi))}", flush=True)
    if PRED is not None:
        PRED[fi] = xo.cpu()
        _Fo = torch.empty_like(S.F); _Fo[S.perm] = S.F
        FDMP[fi] = _Fo.reshape(Np, 3, 3).cpu()
    if LJD is not None:                                                               # 절 CM13.4: 읽기만 (풀이 상태 안 건드림)
        _lo = torch.empty_like(S.ljp); _lo[S.perm] = S.ljp
        LJD[fi] = _lo.cpu()
        if DD is not None:
            _do = torch.empty_like(S.alp); _do[S.perm] = (S.alp if a.dmg == "cdm" else S.ysp)
            DD[fi] = _do.cpu()
        if fi in (0, L - 1) or fi % 20 == 0:
            print(f"  [logJp] 프레임 {fi + 1} 범위 {float(S.ljp.min()):.4g}~{float(S.ljp.max()):.4g} > 0 몫 {float((S.ljp > 0).float().mean()):.4f} | al 범위 {float(S.alp.min()):.4g}~{float(S.alp.max()):.4g} ys 범위 {float(S.ysp.min()):.4g}~{float(S.ysp.max()):.4g}", flush=True)
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
summ = dict(args=vars(a), dmg=DMGI if MATK is not None else None, frames=L, frames_done=len(errs), diverged=div_at, err_mean=(float(np.mean(errs)) if div_at is None else None),
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
if LJD is not None:
    torch.save({"logJp": LJD[:len(errs)], **({"d": DD[:len(errs)]} if DD is not None else {})}, os.path.join(a.out, "lj.pt"))
print(f"[롤아웃] {a.tag} 실시간 판 셀 {a.cell} K {a.K} 뉴턴 {a.newton} PCG {a.cg}: {len(errs)}/{L} 프레임{' (발산 %d)' % div_at if div_at else ''}, 평균 {100 * summ['err_mean_done']:.3f}%, "
      f"프레임당 {summ['ms_mean']} ms (재생 GPU 시간)", flush=True)
