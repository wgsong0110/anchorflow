"""절 AN 실시간 판 롤아웃 (lib/anchorflow/fem_an_rt.py): 프레임마다 CUDA 그래프 재생 한 번 (P2G -> 고정 예산 풀이 -> G2P), 블록 야코비 전처리.
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
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--cell", type=float, default=1.0)
ap.add_argument("--cell_auto", type=float, default=0.0, help="절 CM185 (공통 규칙, 기본 0 = 끔): 프레임 0 입자가 점유한 칸 수 (칸 = --cell 배) 가 N 을 넘으면 칸 배수를 (점유/N)^(1/3) 배로 키움 -- 실시간 해상도 상한. N 아래 장면은 그대로")
ap.add_argument("--box_auto", type=float, default=0.0, help="절 CM185 (공통 규칙, 기본 0 = 끔): 프레임 0 노드 상자 (rt_box 와 같은 정의, Π(최대−최소+3)) 가 B 를 넘으면 칸 배수를 상자 ≤ B 가 될 때까지 키움 -- PCG·MG 커널이 상자 크기로 돈다 (절 CM185 프로파일)")
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
ap.add_argument("--fl_bar", action="store_true", help="장벽 바닥 (절 AN.19, 정의 변경): 법선 = 팔분 표본점 IPC 거리 장벽 (d̂ = ½ dx_정답, κ = (λ+2μ) V_q / d̂²), 고착 = 정답 sticky 가중 (띠 제한 없음) 을 접선에만; --ls_n 선탐색 필요")
ap.add_argument("--fl_bst", type=int, default=0, choices=[0, 1, 2], help="절 AN.21 (--fl_bar 위, 정의 변경): 띠 고착 -- 장벽 띠 안 (d < d̂) 표본점의 고착 몫 = 1 (정답 sticky 노드 v = 0 처럼 완전 고착, 접선); 띠 밖 1 = 0, 2 = 정답 sticky 가중 그대로")
ap.add_argument("--fl_stk", action="store_true", help="절 CM12 (--fl_bar 위, 사용자 승인 정의 변경): 장벽 바닥 (B) 에 (A) 가우스 점 고착 (CM8 a 판 정의 그대로: 가우스 점 걸음 끝 위치 s < 0 의 정답 sticky 가중 -> 노드 k_A = (1 − r̄)^κ, 바깥 법선 이탈 아니면 세 성분, 고착 노드 가속도 0) 을 함께; 장벽 띠 고착 (접선) 과 곱")
ap.add_argument("--fl_bnd", type=int, default=0, choices=[0, 1], help="절 CM159 (--fl_bar --fl_bst 1 위, 정의 변경, 기본 0): 띠 고착 몫을 정답 sticky 정의대로 평면 아래 격자 노드 ((x_n − p)·n < 0, 격자 위치 A L) 에만 모음 -- 평면 위 노드는 고착 안 받음")
ap.add_argument("--bar_cross", action="store_true", help="절 CM10 (--fl_bar 위, 사용자 승인 정의 변경): 장벽을 바닥 평면을 관통하는 B-스플라인 칸 (칸 범위 = 코드의 칸 정의, 걸음 시작 위치에서 판정) 에 속한 표본점에만")
ap.add_argument("--bar_cnt", action="store_true", help="절 CM10 진단 (결과 불변, --fl_bar 위): 프레임마다 장벽이 걸린 표본점 수·전체 대비·칸 수, 관통 칸 수를 fem.jsonl bc 에")
ap.add_argument("--bar_node", default="", choices=["", "add", "only"], help="절 CM14 (--fl_bar 위, 사용자 승인 11-05 정의 변경): B-스플라인 노드 위치의 바닥 위 높이 d_n 에 같은 IPC 장벽 (κ_n = (λ+2μ) V_n / d̂_n², V_n = Σ_q w V_q; 노드 대각). add = 표본점 장벽에 더함, only = 노드 장벽만 (띠 고착은 표본점 그대로)")
ap.add_argument("--bar_ecell", action="store_true", help="절 CM14.3 진단 (결과 불변, --fl_bar 위): 걸음 끝 장벽 에너지를 칸별로 (표본점 -> 그 칸, 노드 -> 노드가 가운데인 칸) -> fem.jsonl be (합) + bener.pt (프레임별 칸 목록: 가운데 노드·프레임 끝 EL·프레임 안 걸음 끝 최대 EM·꼭짓점 8 개)")
ap.add_argument("--fl_vres", type=float, default=0.0, help="절 CM15 (사용자 승인 11-05 정의 변경, --fl_bar·--fl_bst·--y0eb 없이): 바닥 = 장벽·고착 대신 −z 저항 -- 걸음 시작 위치가 평면 아래인 표본점 (장벽과 같은 팔분 질량 중심) 의 법선 이동 s = (x(y) − x(U))·n < 0 에 E = ½ c m_q s², c = k / (τ² thinv²) (k = 이 값, 관성 계수 기준; 후퇴 오일러 k/h², HHT ρ 0.8 2.88 k/h²), 접선 자유")
ap.add_argument("--fl_nclamp", action="store_true", help="절 CM16 (사용자 지시 하드 클램프 11-05, --fl_bar·--fl_bst·--y0eb 없이): 바닥 = 걸음 시작 노드 위치 (A L + U) 가 평면 위 0 거리 이하인 노드의 걸음 변위 법선 성분이 음수면 0 (한쪽 구속, 활성 집합을 뉴턴 반복마다 갱신; 위로·접선 자유, 마찰 없음 -- 고착 KEX 0)")
ap.add_argument("--pen", action="store_true", help="절 CM15 진단 (결과 불변, 바닥 씬): 프레임마다 바닥 평면 아래 입자 수·최대 침투 깊이 (우리·정답), 저항 바닥이면 걸음 시작 위치가 평면 아래인 표본점 수 (마지막 걸음) -> fem.jsonl pen")
ap.add_argument("--bar_ndh", type=float, default=0.5, help="절 CM14: 노드 장벽 두께 d̂_n = 이 값 x dx_정답 (기본 ½ = 표본점 장벽 d̂ 와 같음)")
ap.add_argument("--bar_full", action="store_true", help="절 AN.21: 장벽 헤시안 곱을 덩어리 대신 점별 w wᵀ 참 꼴로 (블록 대각 전처리에는 덩어리 값)")
ap.add_argument("--fl_up", type=int, default=0, choices=[0, 1, 2], help="절 CM17 (사용자 승인 11-06 정의 변경, --floor vel 의 (A) 가우스 점 고착 판만): 고착 몫의 법선 성분을 한쪽 구속으로 -- 걸음 변위 법선 성분이 음수일 때만 법선 고착 (활성 집합, 뉴턴 반복마다 갱신), 위 (+n) 는 자유; 1 = 접선은 지금처럼 고착, 2 = 접선도 자유 (마찰 없음). 0 = 끔 (원래와 같은 기계어)")
ap.add_argument("--fl_pt", action="store_true", help="점별 고착 (절 AN.17, 정의 변경): 고착 거듭제곱을 가우스 점에서 한 뒤 노드로 질량 평균 (be·HHT·GA 처럼 단계 frac 1 판만, --pk·--K_contact 와 함께 쓰지 않음)")
ap.add_argument("--mg_levels", type=int, default=1, help="1 = 블록 야코비, 2 이상 = 그 층 수의 기하 멀티그리드 (거친 층 조밀 상자)")
ap.add_argument("--mg_nc", type=int, default=8)
ap.add_argument("--mg_pow", type=int, default=6)
ap.add_argument("--dump", action="store_true")
ap.add_argument("--dump_F", action="store_true", help="--dump 와 함께: 입자 변형 기울기 F 도 pred.pt 에 (영상 렌더용; 풀이 결과 불변)")
ap.add_argument("--dump_part", action="store_true", help="--dump 와 함께: 발산·GPU 메모리 부족으로 멈춰도 그때까지 프레임을 pred.pt 에 (영상용, 풀이 불변; 절 CM8 (A) 영상)")
ap.add_argument("--tag", default="")
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--no_fuse2", action="store_true", help="결과 불변 융합 2 판 끔 (비교용)")
ap.add_argument("--mg_csolve", action="store_true", help="가장 거친 층 한 블록 커널 (결과 불변, lego 셀 1 에서 더 느림)")
ap.add_argument("--mg_lreuse", action="store_true", help="정의 변경: 거친 층 ω 를 첫 프레임 값으로 재사용")
ap.add_argument("--mg_wreuse", action="store_true", help="정의 변경: 프레임 안 단계 τ 가 같으면 층 0 ω 재사용")
ap.add_argument("--gfin", action="store_true", help="걸음 끝 |g|/|g0| 를 fem.jsonl 에 (기울기 한 번 더 -- 시간 측정 판에서는 끌 것)")
ap.add_argument("--e_scale", type=float, default=1.0, help="절 CM134 진단 (기본 1 = 안 바뀜): 영률 E 에 곱함. 수렴 판 낙하 가속이 정답의 0.55 배인 까닭이 탄성 결합인지 가르는 용도")
ap.add_argument("--cap_mul", type=float, default=2.0, help="절 BD: 고정 용량 버퍼 = 첫 프레임 상자 x 이 배수 (기본 2.0 그대로). 결정적 판에서 결과 불변 (넘침 다시 할당만 덜 일어남)")
ap.add_argument("--det", action="store_true", help="결정적 합산 (같은 명령 두 번이 비트 단위로 같게; lib/anchorflow/fem_an_det.py, MG 와 함께 못 씀) -- 판정용, 속도는 비결정 판으로")
ap.add_argument("--ga", choices=["ga", "hht", "be"], default=None, help="be = 후퇴 오일러 (절 CM190 ablation, --ue beall 필요; --rho 무시). generalized-α (Chung-Hulbert) 또는 HHT-α (implicit-err 절 AN.err3 증분 포텐셜 꼴, 시작점 = 걸음 시작 위치, 끝 = Newmark 위치 식)")
ap.add_argument("--rho", type=float, default=0.7)
ap.add_argument("--fric_mul", type=float, default=1.0, help="절 CM152 진단 전용: 재질 판 마찰각 (cfg friction_angle, 없으면 PG 기본 25도) 에 곱함. 1 = 끔 (그대로). 고침 아님, 원인 가르기")
ap.add_argument("--no_areset", action="store_true", help="고착 노드 걸음 끝 가속도 0 처치 끔")
ap.add_argument("--scheme", choices=["theta", "trbdf2"], default=None, help="단계 방식 (안 주면 --integ: mid = θ ½, be = θ 1)")
ap.add_argument("--theta", type=float, default=0.5)
ap.add_argument("--mfloor", type=float, default=0.0, help="관성 질량 하한 계수 c: m' = max(m, c τ² tr(B)/3) (0 = 없음)")
ap.add_argument("--K_contact", type=int, default=0, help="접촉 적응 걸음: 앞 프레임에 고착 노드가 있으면 이 걸음 수 (0 = 없음)")
ap.add_argument("--ls_n", type=int, default=0, help="선탐색 후보 2^0 .. 2^-(ls_n−1) (0 = 옛 1·½·¼)")
ap.add_argument("--descent", action="store_true", help="하강 방향 보장: g·δ ≥ 0 이면 δ ← α0 M r0 (PCG 첫 방향)")
ap.add_argument("--diag", action="store_true", help="프레임마다 뉴턴 반복별 진단 (에너지·|g|·g·δ·보폭·PCG 잔차·고착 불일치) 을 fem.jsonl 에")
ap.add_argument("--tolN", type=float, default=-1.0, help="절 AN.14 허용치 조기 종료: 걸음마다 뉴턴을 |g|/|g0| <= tolN 이면 멈춤 (--newton 은 상한; 음수 = 고정 반복)")
ap.add_argument("--tolP", type=float, default=-1.0, help="절 AN.14: PCG 를 sqrt(r·z / r0·z0) <= tolP 이면 멈춤 (--cg 는 상한; 음수 = 고정 반복)")
ap.add_argument("--gq_oct", type=float, default=-1.0, help="정의 변경 (절 AN.err4): >= 0 이면 칸 가우스 점 가중 (부피·판정 질량) = (1 − κ) x 팔분 칸 실제 입자 부피 (질량) + κ x 칸 합 / 8. 음수 = 원래 (칸 합 / 8)")
ap.add_argument("--pk", action="store_true", help="절 AN.18 지속형 커널 판: 프레임 하나 = 협동 실행 커널 한 번 (결과 정의 같음)")
ap.add_argument("--pk_loc", type=int, default=2, help="지속형 판: 이 MG 층 이상은 블록 0 하나가 (격자 장벽 줄이기; 99 = 끔)")
ap.add_argument("--pkb", type=int, default=256, help="지속형 판 물리 블록 크기")
ap.add_argument("--pkminb", type=int, default=1, help="지속형 판 __launch_bounds__ 최소 블록 수 (레지스터 상한)")
ap.add_argument("--mv2", action="store_true", help="절 AN.20: 칸 협동 헤시안 곱 (고운·거친 층; 결과 정의 같음, 반올림 순서만 다름)")
ap.add_argument("--mg_occ", action="store_true", help="절 AN.20: 거친 층 곱을 점유 칸 목록만 (빈 칸은 부피 0 이라 연산자 같음; --mv2 포함)")
ap.add_argument("--mg_fuse", type=int, default=0, help="절 AN.20: 이 MG 층 이상 V 순환을 한 블록 커널 하나로 (0 = 끔; 결정적 판과 같은 비트)")
ap.add_argument("--kbench", type=int, default=0, help="절 AN.20: 이 프레임 수를 돈 뒤 커널별 그래프 재생 시간 (µs/회) 을 kbench.json 에 쓰고 끝냄")
ap.add_argument("--y0ls", action="store_true", help="절 AN.22 (결과 정의 같음): ga·장벽 판 뉴턴 시작점을 걸음 시작 U 와 관성 예측 Ũ 사이 후보 2^-i 중 전체 에너지 (관성 + 탄성 + 장벽) 최소로 (--ls_n 필요)")
ap.add_argument("--lsfast", action="store_true", help="절 AN.22 (결과 같음): 선탐색 후보 c = 1 먼저 -- 받아들여지면 나머지 후보 에너지 커널 생략, 하강 방향을 안 바꾼 반복의 관성 몫 다시 계산 생략")
ap.add_argument("--mgbar", nargs="?", const=1, default=0, type=int, help="절 AN.22 (전처리만, 풀리는 방정식 같음): MG V 순환 고운 층 연산자에 장벽 덩어리 헤시안 (원래는 블록 대각에만 있고 잔차 갱신 곱에는 없어 V 순환이 장벽 없는 계를 풀었음); 2 = 거친 층에도 (제한한 덩어리 강성, 블록 대각 셔먼-모리슨)")
ap.add_argument("--cg_list", default=None, help="절 AN.22 (풀이기 설정만): 뉴턴 반복별 PCG 상한 'a,b,..' (길이 = --newton; --cg 대신)")
ap.add_argument("--fk", action="store_true", help="절 AN.22 (같은 식, 반올림만 다름): 선탐색 관성 몫 후보를 이차식 합 넷으로, MG 거듭제곱법 f64 나눗셈 줄임")
ap.add_argument("--y0eb", action="store_true", help="절 CM8 (B, 정의 변경, --ga_y0ut --fl_bar 위): 시작점 섞기 β = 장벽 에너지가 걸음 시작 위치 (U) 의 장벽 에너지보다 크지 않은 가장 큰 2^-k (실현 가능 k 부터 2^-58 배까지, 없으면 U)")
ap.add_argument("--ga_y0ut", action="store_true", help="절 CM2 정의 변경 셋째: 뉴턴 시작점 = 관성 예측 ŷ (고착 반영; 장벽 판은 걸음 시작 U 에서 ŷ 쪽으로 활성 표본점 d > 0 인 최대 2^-k)")
ap.add_argument("--pbc", action="store_true", help="절 CM2 재질 판: 정답 config 입자 속도 구속 (순차 방출·병진, exe/fem_ao_rt_sim.py 와 같음)")
ap.add_argument("--pbc_ns", type=int, default=4)
ap.add_argument("--rel0", action="store_true", help="절 CM141 진단 (기본 끔): 순차 방출 구속을 없애 모든 입자를 t=0 에 동시에 해제 (구속 입자 시각 -1e30)")
ap.add_argument("--relfix", type=float, default=0.0, help="절 CM142 진단 (기본 0 끔): 앞면 x 를 고정 -- x 가 이 값보다 큰 입자만 t=0 에 해제, 나머지는 영구 구속 (시각 +1e30)")
ap.add_argument("--lift", type=float, default=0.0, help="절 CM141 진단 (기본 0): 모든 프레임 입자 z 에 더함 (바닥 위에서 자유낙하 거리를 벌림; 정답 대조는 무의미)")
ap.add_argument("--mlow", type=float, default=0.0, help="절 CM2 재질 판: 거의 빈 노드 문턱 = mlow x 첫 프레임 노드 질량 중앙값 (관성 예측 고정; 0 = 없음)")
ap.add_argument("--visc_rpic", action="store_true", help="절 CM2 재질 판: 정답 rpic_damping -> 등가 점성 (절 AG.20)")
ap.add_argument("--no_vfill", action="store_true")
ap.add_argument("--hspec", action="store_true", help="절 CM3 (결과 정의 같음, 반올림 순서만 다름): 고운 층 헤시안을 주값 분해 30 성분 (U, M = F V, 계수 12) 으로 저장·곱 (rt_mv2s; 모래는 ao_elem_s, plasticine 은 aq_elem 30 성분 그대로); --mv2 필요")
ap.add_argument("--ncomp", action="store_true", help="절 CM3 (결과 정의 같음): PCG·층 0 V 순환 노드 커널을 점유 칸 이웃 노드 목록만 (그 밖 노드는 질량·대각·힘·곱이 모두 0)")
ap.add_argument("--fl_nost", action="store_true", help="절 CM6 진단 전용 (채택 후보 아님): 바닥 고착 끔 (고착 지수 KEX 0 -> 노드 고착 인자 1; 장벽·법선은 그대로)")
ap.add_argument("--fl_dhm", type=float, default=1.0, help="절 CM6 진단 전용 (채택 후보 아님): 장벽 두께 d̂ = 이 값 x ½ dx_정답 (κ 도 같은 식 (λ+2μ) V_q / d̂²)")
ap.add_argument("--hgt", action="store_true", help="절 CM6 진단 (결과 불변): 프레임마다 정답·우리 높이 (질량 중심·최대·99 백분위, 바닥 법선 방향) 를 fem.jsonl hg·hp 에")
ap.add_argument("--fp_interp", action="store_true", help="절 CM20 (사용자 승인 11-07 정의 변경, --ga 만; 절 CM22 재질 판 (wolf 모래·bread) 도 -- 마지막 걸음도 가우스 점 커밋 (RM) 뒤 F·재질 상태 교체): 입자 F 를 G2P 누적 F ← A(x_p) F 대신 프레임 끝 가우스 점 F_q^end = A(x_q) F_q 의 칸 안 삼선형 보간 (회전·로그 신장 분리 평균, rt_gauss 규칙) 으로 교체. 끄면 예전과 비트 같음")
ap.add_argument("--fpi_el", type=int, default=0, help="절 CM191 (기본 0 = 소스 그대로): 1 이면 --fp_interp 를 탄성 재질에만 -- 재질 판 (sand·foam·plasticine·watermelon·metal, 반환 사상 있는 재질) 은 --fp_interp 를 끄고 입자 F = G2P 누적 (걸음별 RM, 입자 소성 이력) 을 씀")
ap.add_argument("--gq_mass", action="store_true", help="절 CM9 (사용자 승인 정의 변경): 노드 질량·운동량 (HHT 입자 가속도 몫도) 을 입자 -> 자기 팔분 가우스 점 -> 노드 (힘과 같은 점 x_q·같은 가중·같은 기저 w_n(x_q)) 로. 끄면 예전과 비트 같음")
ap.add_argument("--gq_chk", action="store_true", help="절 CM9 진단 (결과 불변): 프레임마다 노드 질량·운동량·가속도 몫 합 대 입자 합 상대 차를 fem.jsonl gqchk 에 (구속 입자 속도를 바꾸는 --pbc 씬은 뜻 없음)")
ap.add_argument("--lift_cells", type=int, default=0, help="절 CM11 진단 (사용자 지시): 처음 입자 위치 전체를 바닥 법선 방향으로 이 정수 x B-스플라인 칸 크기 h (= 격자 행렬 A 의 열 길이) 만큼 띄워 시작 (초기 속도·질량·F·구속 입자 선택·빈 노드 문턱은 원래 위치로 그대로); 오차 err 는 정답 그대로 대비, err_lift 는 띄운 거리 빼고")
ap.add_argument("--lift_auto", action="store_true", help="--lift_cells 와 함께: 처음 상태에서 바닥 평면을 관통하는 칸 (절 CM10 꼭짓점 판정, U = 0) 이 0 이 될 때까지 정수를 1 씩 올림")
ap.add_argument("--elx", default="fcr", choices=["fcr", "snh"], help="절 CM18 (사용자 승인 11-06 정의 변경, 탄성 판만): 가우스 점 탄성 -- fcr = 지금 (Ψ = μ Σ (σ − 1)² + λ/2 (J − 1)²), snh = Stable Neo-Hookean (절 CM13 (2) 수박 --elx snh 와 같은 식, μ_s = μ, λ_s = λ + μ; J ≤ 0 에서도 유한). fcr = 원래와 같은 기계어")
ap.add_argument("--qjd", action="store_true", help="절 CM18 진단 (결과 불변, 탄성 판): 프레임 끝 커밋된 U 로 가우스 점 J = det X_q (부피 > 0) -- J ≤ 0 수·비유한 수·최소 J 점 위치·높이를 fem.jsonl qj 에 (풀이와 따로 컴파일한 읽기 전용 커널)")
ap.add_argument("--pdiag", action="store_true", help="절 S.8 진단 (결과 불변): 프레임마다 입자 J = det F 최소·최대·J ≤ 0 수, 이상 입자 (J ≤ 0.05·J > 5·비유한) 수·바닥 위 높이·B-스플라인 칸 (바닥 관통 칸 여부)·오차 꼬리를 fem.jsonl pd 에, 처음 이상 프레임부터 5 프레임 이상 입자 번호를 pdiag_f<프레임>.npy 에")
ap.add_argument("--wdiag", type=int, default=-1, help="절 S.9 진단 (결과 불변, --tolN/--tolP 판): 프레임마다 가우스 점 det (걸음 시작점·뉴턴 반복 끝·걸음 끝 U)·뉴턴 |g|/F·PCG 반복·상한·보폭을 fem.jsonl wd 에, 이 프레임 (1 부터) 부터 --wdiag_end 까지 원자료 wd_f<프레임>.npz (lib/anchorflow/k4why_diag.py); --gfin 과 함께")
ap.add_argument("--wdiag_end", type=int, default=10 ** 6)
ap.add_argument("--wdiag_cap", type=int, default=2000000, help="절 S.10: 노드 용량이 이보다 크면 (상자 폭증) 진단 버퍼·복사를 끔 (메모리만, 결과 불변)")
ap.add_argument("--fhash", action="store_true", help="절 S.9 (결과 불변, 호스트만): 프레임마다 입자 위치 해시 (sha1 앞 10 자리) 를 fem.jsonl xh 에 -- 진단 켬·끔 대조")
ap.add_argument("--prof", type=int, default=0, help="이 프레임마다 커널별 시간 (그래프 밖 한 번, 상태 되돌림) 을 fem.jsonl 에")
ap.add_argument("--ksplit", type=int, default=0, help="절 CM24 (1b, 사용자 승인 11-07 정의 변경): 적응 걸음 쪼개기 -- K 1 로 풀고 덜 풀린 걸음 (판정 커널) 이 있으면 그 프레임을 버리고 K 두 배로 다시, 최대 K = 2^이 값 (0 = 끔)")
ap.add_argument("--uf_res", action="store_true", help="절 CM24 판정 2: 덜 풀린 걸음 = 허용치 미달 & (보폭 < 1 또는 마지막으로 돈 뉴턴 반복 시작 ‖g‖ > F) (끄면 보폭 < 1 만)")
ap.add_argument("--ue", choices=["", "be", "a0", "beall"], default="", help="절 CM24 (2)·(3) (사용자 승인 정의 변경, HHT 판): 덜 풀린 걸음 (판정 커널) 의 걸음 끝 갱신 -- be 후퇴 오일러 (U1 = y, v1 = (y − U)/h, 가속도 기억 0), a0 가속도 기억만 0, beall 모든 걸음 후퇴 오일러 갱신")
ap.add_argument("--ue_loc", type=int, default=0, help="절 CM31 (정의 변경, --ue be·a0 위, lib/anchorflow/fem_common1.py): 덜 풀린 걸음 갱신 바꿈을 노드 집합에만 (비트 합: 1 걸음 끝 장벽 걸린 표본점 노드, 2 --y0loc 시작점 β < 1 노드, 4 잔차 힘 > 노드 무게) -- 0 = 몸 전체 (끄면 소스 그대로)")
ap.add_argument("--uf_w", action="store_true", help="절 CM31 (정의 변경, --uf_res 위): 판정 2 의 힘 척도 F² = ‖T f_0‖² + Σ (m |g|)² (무게를 더함; 끄면 소스 그대로)")
ap.add_argument("--ls0_keep", action="store_true", help="절 CM31 (기본 판 비결정 발산 고침 후보, lib/anchorflow/fem_common1.py): 선탐색·장벽 보폭 0 인 뉴턴 반복은 η·y 그대로 (지금은 η += 0·δ 라 δ 비유한이면 NaN); δ 유한이면 비트 같음, 보폭 0 반복 수·δ 비유한 노드 수를 fem.jsonl l0 에 (끄면 소스 그대로)")
ap.add_argument("--xit_from", type=int, default=0, help="절 CM35 (절 CM33 (c) 를 새 실행기에, --uf_res 허용치 판 위, lib/anchorflow/fem_common2.py): 반복 이 번째 (0 부터) 시작 잔차를 잰 뒤 그때까지 덜 풀림 판정 (보폭 최소 < 1 또는 ‖g‖ > F) 이 아니면 나머지 반복 건너뜀 (--newton 이 덜 풀린 걸음 상한); 0 = 끔")
ap.add_argument("--en_mo", action="store_true", help="절 CM35 (절 CM33 (d) 를 새 실행기에): 새 실행기는 선탐색 (--ls_n 아르미호, 없으면 보폭 0) 이 이미 반복마다 Φ 를 안 올림 -> 소스 그대로, 선탐색 판인지만 확인")
ap.add_argument("--ue_bigT", type=float, default=1e6, help="절 CM35 --ue_loc 비트 32 문턱: 덜 풀린 걸음 ‖g‖²/F² 가 이 값 넘으면 몸 전체 후퇴 오일러")
ap.add_argument("--ue_dil", type=int, default=0, help="절 CM35 (--ue_loc 위, lib/anchorflow/fem_common2.py): 덜 풀린 걸음 후퇴 오일러 노드 집합 S 를 27 이웃으로 이 번 넓힘 (0 = 끔, 소스 그대로)")
ap.add_argument("--ufrec", action="store_true", help="절 CM24 진단 (결과 불변): 판정 커널만 켜고 걸음별 덜 풀림을 fem.jsonl uf 에")
ap.add_argument("--nrec", action="store_true", help="절 CM25 진단 (결과 불변, lib/anchorflow/fem_wolfwhy_diag.py): 걸음마다 시작점 실현 가능 k0·고른 k, 뉴턴 반복별 ‖g‖/F·멈춤·PCG 반복·상한·보폭 (자른 주체), 걸음 끝 ‖g‖/F (다시 잼, 버퍼 되돌림)·장벽 노드 기울기 합을 fem.jsonl nr 에")
ap.add_argument("--prec", type=str, default="", help="절 CM26 진단 (결과 불변, --nrec 위, 재질 판 K 1 HHT, lib/anchorflow/fem_wolfprop_diag.py): 'f0:f1' 프레임 (1 부터, 양끝 포함) 의 뉴턴 반복·걸음 끝 노드 힘 몫 (탄성·소성, 장벽, 순차 방출 구속 반력, 바닥 고착 반력, 고정 노드, 잔차, 관성, 중력) 법선 합·속도 변화 분해 (몸 전체·바닥 근처·구속 노드), 방출 무리별 입자 높이 (우리·정답), 장벽 표본점 d/d̂ 분포를 prec.jsonl 에")
ap.add_argument("--y0loc", type=int, default=0, help="절 CM27 (c) (정의 변경은 뉴턴 시작점뿐, --fl_bar --ga_y0ut 위): 시작점 섞기 β 를 몸 전체 스칼라 대신 노드별 -- 표본점마다 전역 규칙 (실현 가능 + --y0eb) 을 그 점에만 적용한 k 를 그 점이 걸친 노드에 (최대), 섞은 뒤 활성 표본점 d > 0 확인·고침 라운드 이 수, 그래도 어기면 전역 k (0 = 끔, 소스 그대로)")
ap.add_argument("--pbc_phi", action="store_true", help="절 CM27 (e) (정의 변경, 재질 판 순차 방출 구속): 구속 노드 이동도 k_구속 = e^-x 대신 정답 서브스텝 반복 (구속 입자 v = 0 -> 노드 (1 − α) 배, 힘은 매 서브스텝) 의 닫힌 꼴 φ(x) = 2 (x − 1 + e^-x)/x² (끄면 소스 그대로)")
ap.add_argument("--ks_big", action="store_true", help="절 CM37 (조정자 결정 11-08 정의 변경, --ksplit 위, lib/anchorflow/fem_common3.py): 쪼개기 판정을 '덜 풀린 걸음 중 걸음 끝 잔차 비 ‖g‖²/F² > --ue_bigT (비트 32 와 같은 판정, 비유한 포함)' 인 걸음이 있을 때만으로 (끄면 판정 그대로)")
ap.add_argument("--xg", type=int, default=0, help="절 CM37 (결과 불변, --xit_from 위, lib/anchorflow/fem_common3.py): 프레임 그래프를 걸음마다 반복 --xit_from 시작·판정 뒤·뉴턴 끝에서 나눠 캡처, 뒤 구간은 CT[0] = 0 (안 멈춤) 일 때만 재생 (1); 2 = 프로파일 (늘 재생, 건너뛸 구간 GPU 시간 따로); 0 = 끔 (그래프 그대로)")
ap.add_argument("--bar_nostep", action="store_true", help="절 CM25 대조 (진단 전용 정의 변경, 기본 판 아님): 선탐색 없는 장벽 판 (재질) 의 뉴턴 보폭 장벽 실현 가능 자름 (rt_bar_kfe) 끔 -- 보폭 1")
ap.add_argument("--wrx", type=int, default=0, help="절 CM38 재설계 후보 비트 (lib/anchorflow/fem_wmrx.py; 0 이면 안 부름): 1 고착 상한, 2 속도 수준 접촉 -- 바닥 노드 커널 rt_floor_n·mt_floor_n")
ap.add_argument("--wrx_skc", type=float, default=1e-2, help="절 CM38 후보 1 문턱")
ap.add_argument("--wrx_col", type=int, default=0, help="절 CM38: 0 모든 걸음, 1 무너진 프레임만 (--ksplit --ks_big 위)")
ap.add_argument("--be_sub", type=int, default=0, help="절 CM193 (기본 0 = 그대로): 프레임보다 짧은 걸음 (K > 1: ssN·kN·다시 풀기·쪼개기) 은 HHT 대신 후퇴 오일러 (α_m = α_f = 0, γ = β = 1, 매 걸음 U1 = y·v1 = (y − U)/h·a 0 = 판 be 와 같은 갱신, --ue_loc 노드 집합 없이 모든 노드). 1 = 모든 재질, 2 = 반환 사상 재질 (sand·foam·plasticine·watermelon·metal) 만. K = 1 걸음은 그대로")
ap.add_argument("--be_sub_mf", type=float, default=0.0, help="절 CM193 (--be_sub 위, 기본 0 = 그대로): K > 1 걸음에서 관성 질량 하한 m' = max(m, c τ² tr(B)/3) (--mfloor 와 같은 식, c = 이 값) -- 거의 빈 노드의 강성/질량 비 (ωτ) 를 묶음. K = 1 걸음은 그대로")
ap.add_argument("--wrx_el", type=int, default=0, help="절 CM192 (기본 0 = 소스 그대로): 1 이면 반환 사상 없는 탄성 재질은 --wrx_col 1 (X3 를 무너진 프레임 다시 풀기에서만), 반환 사상 재질 (sand·foam·plasticine·watermelon·metal) 은 주어진 --wrx_col 그대로")
a = ap.parse_args()
if a.wrx:                                                                             # 절 CM38: 소스 바꾸기는 모듈 만들기 전에
    from anchorflow import fem_wmrx as WRX
    WRX.install(new=True)
assert a.xit_from <= 0 or (a.uf_res and a.ue and a.tolN >= 0 and a.newton > a.xit_from and (a.ksplit == 0 or a.ks_big)), "--xit_from 은 --ue·--uf_res 허용치 판, --newton > 이 값, 쪼개기 없이 (절 CM37 --ks_big 은 허용)"   # 절 CM35
assert not a.en_mo or a.ls_n > 0, "--en_mo (새 실행기) 는 --ls_n 선탐색 판만 (선탐색이 (d) 를 이미 함)"                                          # 절 CM35
assert a.ue_dil <= 0 or a.ue_loc, "--ue_dil 은 --ue_loc 위"
dev = "cuda"
RTC.RAWPTR[0] = True
os.makedirs(a.out, exist_ok=True)
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
if a.fric_mul != 1.0:                                                                  # 절 CM152 진단
    cfg = dict(cfg); cfg["friction_angle"] = float(cfg.get("friction_angle", 25.0)) * a.fric_mul
    print(f"[절 CM152] 마찰각 x{a.fric_mul} -> {cfg['friction_angle']} 도", flush=True)
if a.lift != 0.0:
    d["x"] = d["x"].clone(); d["x"][..., 2] += a.lift
h = float(cfg["frame_dt"])
DM.REGION_DENSITY[0] = bool(a.region_density)
m64 = DM.traj_mass(d, dev).double()
EXT = float(d["_ext"])
rho = (d["_rho"].to(dev).double()) if a.region_density else None
vol64 = (m64 / rho) if a.region_density else (m64 / float(cfg["density"]))
mu, lam = PR.lame(float(cfg["E"]) * a.e_scale, cfg["nu"])
Np = int(d["x"].shape[1])
L = min(a.frames, int(d["x"].shape[0]) - 1)
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
NODE_H = a.cell * DXG * math.sqrt(2.0)
lat = SX.lattice(NODE_H, SX.HZ_CUBE * NODE_H, device=dev, dtype=torch.float64)
CAUTO = None
if a.cell_auto > 0 or a.box_auto > 0:                                                  # 절 CM185: 점유 칸 수 상한 (칸 정의 = 아래 _ncross 의 점유 칸과 같음)
    _x0c = d["x"][0].to(dev).double()
    _occ0 = int(torch.unique(torch.floor(_x0c @ lat.Ai.T - 0.5), dim=0).shape[0])
    CAUTO = dict(N=a.cell_auto, occ0=_occ0, cell0=a.cell)
    _occ, _itc = _occ0, 0
    while a.cell_auto > 0 and _occ > a.cell_auto and _itc < 50:                       # 상한 아래가 될 때까지 (점유 수는 c^-3 보다 느리게 줄 수 있음: 수박 1 번 만에 5.9 만)
        a.cell = a.cell * max((_occ / a.cell_auto) ** (1.0 / 3.0), 1.01); _itc += 1
        NODE_H = a.cell * DXG * math.sqrt(2.0)
        lat = SX.lattice(NODE_H, SX.HZ_CUBE * NODE_H, device=dev, dtype=torch.float64)
        _occ = int(torch.unique(torch.floor(_x0c @ lat.Ai.T - 0.5), dim=0).shape[0])
    CAUTO.update(occ1=_occ, itc=_itc)
    if a.box_auto > 0:
        def _box(lt):
            _b = torch.floor(_x0c @ lt.Ai.T - 0.5)
            _nb = (_b.max(0).values - _b.min(0).values + 3).long()
            return int(_nb.prod()), [int(q) for q in _nb]
        _bx, _nb0 = _box(lat); CAUTO.update(box0=_bx, nb0=_nb0, B=a.box_auto)
        _it = 0
        while _bx > a.box_auto and _it < 50:
            a.cell = a.cell * max((_bx / a.box_auto) ** (1.0 / 3.0), 1.01); _it += 1
            NODE_H = a.cell * DXG * math.sqrt(2.0)
            lat = SX.lattice(NODE_H, SX.HZ_CUBE * NODE_H, device=dev, dtype=torch.float64)
            _bx, _nb1 = _box(lat)
        CAUTO.update(box1=_bx, it=_it, occ2=int(torch.unique(torch.floor(_x0c @ lat.Ai.T - 0.5), dim=0).shape[0]))
    CAUTO["cell"] = a.cell; del _x0c
    print(f"[절 CM185 칸 상한] {json.dumps(CAUTO)}", flush=True)
    os.makedirs(a.out, exist_ok=True); json.dump(CAUTO, open(os.path.join(a.out, "cauto.json"), "w"))
dt = h / a.K
VD_S = float(cfg.get("grid_v_damping_scale", 1.0))
DAMP = (VD_S ** (dt / float(cfg["substep_dt"]))) if (a.pg_vdamp and VD_S < 1.0) else 1.0
FL = None
if a.floor == "vel":
    pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
    NR = np.array(pl["normal"], dtype=np.float64); NR = NR / np.linalg.norm(NR)
    PT = [float(q) for q in pl["point"]]
    FL = dict(PT=PT, NR=NR.tolist(), PN=float(np.dot(PT, NR)), DXG=DXG, KEX=dt / float(cfg["substep_dt"]), vout=not a.no_vfl_out, pt=a.fl_pt, bar=a.fl_bar, bst=a.fl_bst, bfull=a.bar_full, bcross=a.bar_cross, bcnt=a.bar_cnt, stk=a.fl_stk, bnd=a.fl_bnd, DH=0.5 * DXG, KS=(lam + 2.0 * mu) / (0.5 * DXG) ** 2)
    assert a.fl_bar or not a.fl_stk, "--fl_stk 는 --fl_bar 위 (절 CM12)"
    assert a.fl_bar or not a.fl_bnd, "--fl_bnd 는 --fl_bar 위 (절 CM159)"
    if a.fl_nost:
        FL["KEX"] = 0.0
    if a.fl_dhm != 1.0:
        FL["DH"] = a.fl_dhm * 0.5 * DXG; FL["KS"] = (lam + 2.0 * mu) / FL["DH"] ** 2
    if a.bar_node:                                                                     # 절 CM14: 노드 장벽 (표본점 장벽 d̂·κ·띠 고착 문턱은 그대로)
        assert a.fl_bar, "--bar_node 는 --fl_bar 위 (절 CM14)"
        FL["bnode"] = a.bar_node; FL["DHN"] = a.bar_ndh * DXG; FL["KSN"] = (lam + 2.0 * mu) / FL["DHN"] ** 2
    if a.y0loc:                                                                        # 절 CM27 (c)
        assert a.fl_bar and a.ga_y0ut and not a.bar_node and not a.fl_vres, "--y0loc 는 --fl_bar --ga_y0ut 표본점 장벽 판만"
        FL["y0loc"] = int(a.y0loc)
    if a.bar_ecell:                                                                    # 절 CM14.3 진단 (결과 불변)
        assert a.fl_bar, "--bar_ecell 은 --fl_bar 위"
        FL["becell"] = True
    if a.fl_vres > 0:                                                                  # 절 CM15: 표본점 버퍼만 장벽 판 것을 씀 (bar), 장벽 에너지·고착 (KEX 0 -> 노드 고착 인자 1) 없음
        assert not (a.fl_bar or a.fl_bst or a.fl_stk or a.bar_node or a.bar_cross or a.bar_cnt or a.bar_ecell or a.bar_full or a.fl_pt or a.y0eb), "--fl_vres 는 장벽·고착 플래그 없이 (절 CM15)"
        FL.update(bar=True, vres=float(a.fl_vres), KEX=0.0, bst=0)
    if a.fl_nclamp:                                                                    # 절 CM16: 노드 법선 하드 클램프 -- 장벽·고착 (KEX 0 -> 노드 고착 인자 1) 없음
        assert not (a.fl_bar or a.fl_bst or a.fl_stk or a.bar_node or a.bar_cross or a.bar_cnt or a.bar_ecell or a.bar_full or a.fl_pt or a.y0eb or a.fl_vres), "--fl_nclamp 는 장벽·고착·저항 플래그 없이 (절 CM16)"
        FL.update(nclamp=True, KEX=0.0, bst=0)
    if a.fl_up:                                                                        # 절 CM17: (A) 고착 몫 법선 한쪽 구속 -- 장벽·띠 고착·점별·클램프·저항 없이
        assert not (a.fl_bar or a.fl_bst or a.fl_stk or a.bar_node or a.bar_cross or a.bar_cnt or a.bar_ecell or a.bar_full or a.fl_pt or a.y0eb or a.fl_vres or a.fl_nclamp or a.fl_nost), "--fl_up 은 (A) 가우스 점 고착 판 (--floor vel) 만 (절 CM17)"
        FL["fup"] = int(a.fl_up)
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
F0 = DM.traj_F(d, h, dev)[0].to(dev).double().reshape(Np, 9)
X0 = d["x"][0].to(dev)
LIFT = None
if a.lift_cells or a.lift_auto:                                                       # 절 CM11: 물체 전체를 바닥 법선 방향으로 h 의 정수배 띄움 (질량·F·구속 선택·문턱은 원래 위치로 앞뒤에서 그대로)
    assert FL is not None, "--lift_cells 는 바닥 씬만"
    _nr = torch.tensor(FL["NR"], device=dev, dtype=torch.float64)
    HC = float(lat.A[:, 0].norm())
    _cs = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], device=dev, dtype=torch.float64)

    def _ncross(xx):                                                                   # 절 CM10 관통 칸 정의 (U = 0): 점유 칸 꼭짓점 A (b + ½ + {0,1}³) 평면 부호 거리 최소 < 0 < 최대
        _b = torch.unique(torch.floor(xx.double() @ lat.Ai.T - 0.5), dim=0)
        _dd = ((_b[:, None, :] + 0.5 + _cs[None]) @ lat.A.T) @ _nr - FL["PN"]
        _mn, _mx = _dd.min(1).values, _dd.max(1).values
        return dict(cross=int(((_mn < 0) & (_mx > 0)).sum()), occ=int(_b.shape[0]), below=int((_mx <= 0).sum()), dmin_part=float((xx.double() @ _nr - FL["PN"]).min()), dmin_corner=float(_mn.min()))
    _c0 = _ncross(X0)
    _n = max(a.lift_cells, 1 if a.lift_auto else 0); _tr = []
    while True:
        _ci = _ncross(X0.double() + _n * HC * _nr); _tr.append(dict(n=_n, **_ci))
        if not a.lift_auto or _ci["cross"] == 0:
            break
        _n += 1
    LOFF = (_n * HC * _nr).to(X0.dtype)
    X0 = X0 + LOFF
    LIFT = dict(n=_n, h=HC, dist=_n * HC, normal=FL["NR"], orig=_c0, tried=_tr)
    print(f"[절 CM11 띄움] 칸 h {HC:.5f} m x {_n} = {_n * HC:.5f} m (법선 {FL['NR']}); 처음 관통 칸 원래 {_c0['cross']} / 점유 {_c0['occ']} -> 띄운 뒤 {_tr[-1]['cross']} (시도 {[(q['n'], q['cross']) for q in _tr]})", flush=True)
if a.det:
    from anchorflow import fem_an_det as ADET
SOLVER = ADET.DetRT if a.det else ART.RTSolver
MATX = {}
SNHX = dict(snh=True) if a.elx == "snh" else {}                                    # 절 CM18: 탄성 판만 (MATX 에 안 넣음 -- MATX 가 차면 재질 판 발산 판정이 켜짐)
MATNAME = cfg.get("material")
if a.fpi_el and a.fp_interp and MATNAME in ("sand", "foam", "plasticine", "watermelon", "metal"):     # 절 CM191: 반환 사상 재질은 가우스 점 F 보간 교체를 하지 않음
    a.fp_interp = False
    print(f"[절 CM191 --fpi_el] 재질 {MATNAME}: --fp_interp 끔 (입자 F = G2P 누적)", flush=True)
if a.wrx_el and a.wrx and not a.wrx_col and MATNAME not in ("sand", "foam", "plasticine", "watermelon", "metal"):   # 절 CM192: X3 (고착 상한 + 속도 수준 비탄성 접촉) 는 소성 흐름 재질에서만 매 걸음
    a.wrx_col = 1
    print(f"[절 CM192 --wrx_el] 재질 {MATNAME}: --wrx_col 1 (X3 는 무너진 프레임 다시 풀기에서만)", flush=True)
if MATNAME in ("sand", "foam", "plasticine", "watermelon", "metal"):                            # 절 CM2: 재질 판 (lib/anchorflow/fem_mat_rt.py) -- 같은 실행기·같은 인자
    assert a.elx == "fcr", "--elx snh 는 탄성 판만 (재질 판 수박은 exe/fem_ao_rt_sim.py --elx, 절 CM13)"     # 절 CM22: --fp_interp·--qjd 재질 판도 (끄면 그대로)
    from anchorflow import fem_mat_rt as MRT
    from anchorflow import fem_ao_cuda as AO
    from anchorflow import fem_pl_cuda as PLC
    TREL = PVB = None
    if a.pbc:                                                                          # exe/fem_ao_rt_sim.py 와 같은 구속 입자·끝 시각
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
        if a.rel0:
            TREL = torch.full_like(TREL, -1e30)
        if a.relfix > 0:
            TREL = torch.where(_x0b[:, 0] > a.relfix, torch.full_like(TREL, -1e30), torch.full_like(TREL, 1e30))
    MTHR = 0.0
    if a.mlow > 0:                                                                     # 첫 프레임 노드 질량 중앙값 (exe/fem_ao_rt_sim.py 와 같은 식)
        _y = d["x"][0].to(dev).double() @ lat.Ai.T
        _b = torch.floor(_y - 0.5).long(); _fx = _y - _b; _b = _b - _b.min(0).values; _nb = (_b.max(0).values + 3).tolist()
        _w = torch.stack([0.5 * (1.5 - _fx) ** 2, 0.75 - (_fx - 1) ** 2, 0.5 * (_fx - 0.5) ** 2], 1)
        _mn = torch.zeros(_nb[0] * _nb[1] * _nb[2], device=dev, dtype=torch.float64)
        for _i in range(3):
            for _j in range(3):
                for _k in range(3):
                    _mn.index_add_(0, ((_b[:, 0] + _i) * _nb[1] + _b[:, 1] + _j) * _nb[2] + _b[:, 2] + _k, _w[:, _i, 0] * _w[:, _j, 1] * _w[:, _k, 2] * m64)
        MTHR = float(a.mlow * _mn[_mn > 0].median())
        del _y, _b, _fx, _w, _mn
    if MATNAME == "sand":
        MATX = dict(mat=2, cdp=AO.cdp_of(cfg, float(mu), float(lam))[0])
    else:
        _mat, PRM, PRD = PLC.params(cfg, float(mu), float(lam), dev)
        VISC = 0.0
        if a.visc_rpic and float(cfg.get("rpic_damping", 0.0)) > 0:
            _r = float(cfg["rpic_damping"])
            VISC = float(cfg["density"]) * _r * (2 - _r) * DXG ** 2 / (16.0 * float(cfg["substep_dt"]))
        MATX = dict(mat=_mat, prm=PRM, visc=VISC, ys0=PRD["yield_stress"], lj0=(PRD["logJp0"] if _mat == 7 else 0.0))
    MATX.update(trel=TREL, pvb=PVB, pbc_ns=a.pbc_ns, dt_gt=float(cfg["substep_dt"]), mthr=MTHR, vfill=not a.no_vfill, det=a.det)
    if a.pbc_phi and TREL is not None:                                                 # 절 CM27 (e) (순차 방출 구속 없는 씬은 할 일 없음 -- 같은 플래그를 모든 씬에)
        MATX["pbc_phi"] = True
    SOLVER = MRT.MatRT
    a.ls = 0                                                                           # 재질 판: 퍼텐셜 없음 -> 에너지 선탐색 없음 (절 AO.0; 장벽이면 실현 가능 보폭만)
    print(f"[절 CM2 재질 판] {MATNAME} {MATX.get('mat')} 구속 {TREL is not None} 구속 속도 {PVB is not None} 빈 노드 문턱 {MTHR:.3e} 점성 {MATX.get('visc', 0.0)}", flush=True)
ART.WW.ON[0] = bool(a.nrec)                                                          # 절 CM25: 버퍼는 alloc 에서
if a.prec:                                                                           # 절 CM26: 버퍼는 WW.alloc 끝에서
    assert a.nrec and a.K == 1 and a.ga == "hht" and MATX, "--prec 는 --nrec 위, K 1, HHT, 재질 판만"
    ART.PRD.ON[0] = True
    PR_F0, PR_F1 = [int(q) for q in a.prec.split(":")]
S = SOLVER(X0, d["v"][0].to(dev), F0, m64, vol64, lat.A, lat.Ai, mu, lam, [float(q) for q in cfg["g"]], h, a.K, a.integ, a.newton, a.cg, a.ls,
                 floor=FL, damp=DAMP, bc=BC, mg_levels=a.mg_levels, mg_nc=a.mg_nc, mg_pow=a.mg_pow, descent=a.descent, ls_n=a.ls_n, scheme=a.scheme, theta=a.theta, mfloor=a.mfloor, K_contact=a.K_contact or None, gfin=a.gfin, fuse2=not a.no_fuse2, mg_wreuse=a.mg_wreuse, mg_lreuse=a.mg_lreuse, mg_csolve=a.mg_csolve, ga=a.ga, rho=a.rho, areset=not a.no_areset, tolN=a.tolN, tolP=a.tolP, goct=a.gq_oct, pk=a.pk, pk_loc=a.pk_loc, pkb=a.pkb, pkminb=a.pkminb, mv2=a.mv2, mg_occ=a.mg_occ, mg_fuse=a.mg_fuse, y0ls=a.y0ls, lsfast=a.lsfast, mgbar=a.mgbar, cg_list=a.cg_list, fk=a.fk, y0ut=a.ga_y0ut, y0eb=a.y0eb, hspec=a.hspec, ncomp=a.ncomp, cap_mul=a.cap_mul, gqm=a.gq_mass, cj=({**(dict(judge=True, res=a.uf_res, ue=a.ue, **({"uel": a.ue_loc} if a.ue_loc else {}), **({"ufw": True} if a.uf_w else {}), **({"xf": a.xit_from} if a.xit_from > 0 else {}), **({"dil": a.ue_dil} if a.ue_dil > 0 else {}), **({"bigT": a.ue_bigT} if a.ue_loc & 32 else {})) if (a.ksplit > 0 or a.ufrec or a.ue) else {}), **({"ls0": True} if a.ls0_keep else {})} or None), **({"fpi": True} if a.fp_interp else {}), **({"wd": True, "wdcap": a.wdiag_cap} if a.wdiag >= 0 else {}), **MATX, **SNHX)
if a.bar_nostep:
    assert getattr(S, "bar", False) and not S.ls, "--bar_nostep 은 선탐색 없는 장벽 판 (재질) 만"
    S.nobk = True
if a.be_sub and (a.be_sub == 1 or MATNAME in ("sand", "foam", "plasticine", "watermelon", "metal")):   # 절 CM193: K > 1 걸음은 후퇴 오일러 (그래프는 K 마다 캡처되므로 K 별로 고정)
    assert S.ga and S.cue, "--be_sub 는 HHT·GA 판 + --ue (갱신 바꿈 커널 컴파일) 위"
    _BEC = dict(am=0.0, af=0.0, gm=1.0, be=1.0)

    def _st_be(K_):
        c0 = S.ga_c
        if K_ > 1:
            S.ga_c = _BEC
        try:
            return type(S).stages(S, K_)
        finally:
            S.ga_c = c0

    def _rec_be(t_, K_=None):
        Kx = K_ or S.K
        if Kx <= 1:
            return type(S).record(S, t_, K_)
        sv = (getattr(S, "uel", 0), S.cue, S.mfloor)
        S.uel = 0; S.cue = 3
        if a.be_sub_mf > 0:                                                            # 절 CM193 --be_sub_mf: K > 1 그래프에만 관성 질량 하한
            S.mfloor = max(S.mfloor, a.be_sub_mf)
        try:
            return type(S).record(S, t_, K_)
        finally:
            S.uel, S.cue, S.mfloor = sv
    S.stages = _st_be; S.record = _rec_be
    print(f"[절 CM193 --be_sub {a.be_sub}] 재질 {MATNAME}: K > 1 걸음은 후퇴 오일러" + (f", 관성 질량 하한 c {a.be_sub_mf:g}" if a.be_sub_mf > 0 else ""), flush=True)
if a.wrx:                                                                             # 절 CM38 후보 장치 전역 (col 이면 처음 0)
    WRX.setup(S, a.wrx, a.wrx_skc); WRX.on(S, 0 if a.wrx_col else 1)
_split = (lambda t_: WRX.split_col(S, t_, h, None, a.ksplit, fn=lambda d_: ART.split_step(S, t_, h, d_))) if (a.wrx and a.wrx_col) else (lambda t_: ART.split_step(S, t_, h, a.ksplit))
if a.ksplit > 0:                                                                      # 절 CM24 (1b): 버퍼는 걸음 수 상한 2^ksplit 으로, K 1 로 시작
    assert a.K == 1 and not a.K_contact and not a.pk and a.ga, "--ksplit 은 K 1, --K_contact·--pk 없이, HHT·GA 판만"
    S.kcap = 1 << a.ksplit
    S.alloc(None)
if a.ks_big:                                                                          # 절 CM37: 무너진 걸음만 쪼개기 (문턱 = 비트 32 와 같은 --ue_bigT)
    assert a.ksplit > 0 and a.uf_res and a.tolN >= 0, "--ks_big 은 --ksplit·--uf_res 허용치 판 위"
    S.ksbig = float(a.ue_bigT)
if a.xg:                                                                              # 절 CM37: 나눈 그래프 (결과 불변)
    assert a.xit_from > 0 and a.tolN >= 0 and not getattr(a, "pk", 0), "--xg 는 --xit_from 허용치 그래프 판 위"
    from anchorflow import fem_common3 as _C3
    _C3.install(S, a.xg)
print(f"[절 AN 실시간 판] 입자 {Np}, 프레임 {L}, 셀 {float(lat.A[:, 0].norm()):.4f} m (정답 dx {DXG} 의 {a.cell} 배), K {a.K} {a.integ}, 뉴턴 {a.newton} x PCG {a.cg}, 선탐색 {a.ls}, MG 층 {a.mg_levels}, 적분기 {a.ga + ' ρ ' + str(a.rho) if a.ga else a.integ}, "
      f"바닥 {a.floor}, 구속 {len(BC) if BC else 0}", flush=True)
flog = open(os.path.join(a.out, "fem.jsonl"), "w")
errs, ms = [], []
div_at = None
OOM = None
PD_ON = []
PRED = torch.empty(L, Np, 3) if a.dump else None
BEN = [] if a.bar_ecell else None                                                  # 절 CM14.3: 프레임별 칸 목록
PREDF = torch.empty(L, Np, 3, 3) if (a.dump and a.dump_F) else None
for fi in range(L):
    kp = ((S.pk_profile(fi * h) if a.pk else S.profile(fi * h)) if (a.prof and fi >= a.warm and (fi - a.warm) % a.prof == 0) else None)
    if a.gq_chk:                                                                       # 절 CM9: 프레임 시작 입자 합 (f64)
        _m = S.m.double(); _ps = [float(_m.sum()), (_m[:, None] * S.v.double()).sum(0), (_m[:, None] * S.pa.double()).sum(0), float((_m * S.v.double().norm(dim=-1)).sum()), float((_m * S.pa.double().norm(dim=-1)).sum())]
    if a.wdiag >= 0:                                                                   # 절 S.9: 프레임 시작 입자 det F·위치 (원래 순서)
        _Fs = torch.empty_like(S.F); _Fs[S.perm] = S.F
        _Jp0 = torch.linalg.det(_Fs.reshape(-1, 3, 3).double()); _x0o = S.x_orig().clone(); del _Fs
    if a.dump_part:                                                                    # 메모리 부족 = 그 프레임에서 멈춤 (발산 쪽으로 기록, 덤프는 앞 프레임까지)
        try:
            if a.ksplit > 0:                                                           # 절 CM37: 덤프 판도 쪼개기 경로 그대로
                redo, _KF, _ktr = _split(fi * h)
            else:
                redo = S.step(fi * h)
        except torch.OutOfMemoryError as _ex:
            div_at, OOM = fi + 1, str(_ex)[:200]
            print(f"[발산] 프레임 {div_at} (GPU 메모리 부족 -- 용량 늘려 다시 중)", flush=True)
            break
    elif a.ksplit > 0:                                                                 # 절 CM24 (1b): 시간 = 모든 시도의 재생 GPU 시간 합
        redo, _KF, _ktr = _split(fi * h)
    else:
        redo = S.step(fi * h)
    gt = d["x"][fi + 1].to(dev).float()
    xo = S.x_orig()
    _ok = torch.isfinite(gt).all(-1)                                                   # 정답 비유한 입자 (GF) 는 지표에서 거름 (탄성 씬은 모두 유한 -> 같은 값)
    e = float((xo[_ok] - gt[_ok]).norm(dim=-1).mean()) / EXT
    fin = bool(torch.isfinite(xo).all()) and math.isfinite(e)
    if LIFT is not None:                                                               # 절 CM11: 띄운 거리 빼고 잰 오차 (참고)
        c_lift = float(((xo[_ok] - LOFF.float()) - gt[_ok]).norm(dim=-1).mean()) / EXT
    errs.append(e if fin else float("nan")); ms.append(S.last_ms)
    c = S.counts()
    if getattr(S, "cjj", False):                                                      # 절 CM24 (결과 불변): 걸음별 덜 풀림·보폭 최소·잔차 비 (마지막 시도)
        _Kn = S.Kcur
        c["uf"] = dict(fail=[int(q) for q in S.cjl[:_Kn].cpu()], smin=[float(q) for q in S.cjs[:_Kn].cpu()], g2=[float(q) for q in S.cjg[:_Kn].cpu()] if S.cjr else None)
        if getattr(S, "uel", 0):                                                       # 절 CM31: 걸음별 [S 노드, 몸 노드, S∩몸]
            c["uf"]["ul"] = S.ULS[:_Kn, :3].cpu().tolist()
    if a.ksplit > 0:
        c["ks"] = dict(K=_KF, tries=[list(q) for q in _ktr])
    if a.xg:                                                                          # 절 CM37: 조건 구간 [건너뜀, 돎] (+ --xg 2 GPU ms [늘, 돈 조건, 건너뛸 조건])
        c["xg"] = _C3.frame_stats(S)
    if a.ls0_keep:                                                                     # 절 CM31: 이 프레임 보폭 0 반복 수·δ 비유한 노드 수 (읽고 0)
        _g = S.mod.get_global("AF_L0N"); _a = ART.cp.ndarray((2,), dtype=ART.cp.int32, memptr=_g); c["l0"] = [int(q) for q in _a.get()]; _a[:] = 0
    if a.diag:
        c["diag"] = S.diag()
    c["K"] = S.last_K
    if a.gq_chk:                                                                       # 노드 합 (이 프레임 P2G 결과, 풀이 중 안 바뀜)
        _nn = int(S.IP[9]); _mn = S.mn[:_nn].double()
        c["gqchk"] = [abs(float(_mn.sum()) - _ps[0]) / _ps[0], float((S.mv[:_nn].double().sum(0) - _ps[1]).norm()) / max(_ps[3], 1e-30),
                      float((S.ma[:_nn].double().sum(0) - _ps[2]).norm()) / max(_ps[4], 1e-30) if S.ga else None]
    if LIFT is not None:
        c["err_lift"] = c_lift
    if a.bar_cnt and getattr(S, "bar", False):                                         # 절 CM10 진단: 장벽 걸린 표본점·칸, 관통 칸
        c["bc"] = S.bar_counts()
    if BEN is not None and getattr(S, "bar", False):                                   # 절 CM14.3 진단: 칸별 걸음 끝 장벽 에너지
        c["be"], _cl = S.bar_ecell()
        BEN.append(_cl)
        if fi % 10 == 9 or fi + 1 == L:
            torch.save(dict(frames=BEN, A=S.A.float().reshape(3, 3).cpu()), os.path.join(a.out, "bener.pt"))
    if getattr(S, "bnode", ""):                                                        # 절 CM14 진단: 노드 장벽 걸린 노드 수·부피 > 0 노드 수·d ≤ 0 이라 빠진 노드 수
        c["bn"] = S.barn_counts()
    if a.pen and FL:                                                                   # 절 CM15 진단 (결과 불변): 바닥 평면 아래 입자 수·최대 깊이 (m), 저항 바닥 표본점 수
        _nr = torch.tensor(FL["NR"], device=dev, dtype=torch.float64)
        _hp = xo.double() @ _nr - FL["PN"]; _hg = gt[_ok].double() @ _nr - FL["PN"]
        _hp = _hp[torch.isfinite(_hp)]
        c["pen"] = dict(np=int((_hp < 0).sum()), dmax=float((-_hp.min()).clamp(min=0)) if _hp.numel() else None, gnp=int((_hg < 0).sum()), gdmax=float((-_hg.min()).clamp(min=0)))
        if getattr(S, "vres", 0.0):
            _nq = int(S.IP[12]); c["pen"]["nq"] = int(((S.bact[:_nq] > 0) & (S.bx[:_nq * 4].view(-1, 4)[:, 3] > 0)).sum())
    if getattr(S, "fup", 0):                                                           # 절 CM17 진단 (결과 불변): 걸음별 후보·넣기·빼기·걸음 끝 법선 고착 노드 수
        c["fup"] = S.fup_counts()
    if getattr(S, "ncl", False):                                                       # 절 CM16 진단 (결과 불변): 걸음별 후보·넣기·빼기·걸음 끝 활성 노드 수
        c["ncl"] = S.ncl_counts()
    if S.y0ut and getattr(S, "bar", False) and not getattr(S, "vres", 0.0):           # 절 CM2 진단: 걸음별 시작점 k (β = 2^-k, 0 = ŷ 그대로)
        c["y0k"] = S.Y0K[:S.nstage, 0].cpu().tolist()
        if not S.ls:                                                                   # 절 CM6 진단: 선탐색 없는 장벽 판 (재질) 뉴턴 반복별 보폭 k (상한까지 펼친 자리; 허용치로 건너뛴 반복은 앞 값)
            c["bk"] = S.BSK[:S.nstage * S.nt, 0].cpu().tolist()
    if getattr(S, "y0loc", 0):                                                         # 절 CM27 (c) 진단: 걸음별 [k>0 노드, 최대 k, β0 노드, 몸 노드, 라운드별 어긴 점 수 ...]
        c["yl"] = S.YLS[:S.nstage, :5 + S.y0loc].cpu().tolist()
    if a.prec and PR_F0 <= fi + 1 <= PR_F1:                                            # 절 CM26 진단 (결과 불변, 읽기만)
        if fi + 1 == PR_F0 or "PRC" not in globals():
            PRC = ART.PRD.Ctx(S, FL, [float(q) for q in cfg["g"]], TREL, X0.double(), h)
            plog = open(os.path.join(a.out, "prec.jsonl"), "a")
        try:
            _pr = PRC.frame(fi, d["x"][fi + 1].to(dev))
        except Exception:                                                              # 진단 계산 실패는 풀이에 영향 없음 -- 기록만
            import traceback as _tb
            _pr = dict(err=_tb.format_exc()[-1500:])
        plog.write(json.dumps(dict(f=fi + 1, **_pr)) + "\n"); plog.flush()
    if a.nrec:                                                                         # 절 CM25 진단 (결과 불변)
        c["nr"] = ART.WW.frame(S, a.tolN)
    if a.pdiag:                                                                        # 절 S.8 진단 (결과 불변, 상태 안 바꿈)
        _Fo = torch.empty_like(S.F); _Fo[S.perm] = S.F
        _J = torch.linalg.det(_Fo.reshape(-1, 3, 3).double())
        _ep = (xo.double() - gt.double()).norm(dim=-1) / EXT
        _nr = torch.tensor(FL["NR"] if FL else [0.0, 0.0, 1.0], device=dev, dtype=torch.float64)
        _hp = xo.double() @ _nr - (FL["PN"] if FL else 0.0)
        _bd = ~torch.isfinite(_J) | (_J <= 0.05) | (_J > 5.0)
        _pd = dict(Jmin=float(torch.nan_to_num(_J, nan=-1e30).min()), Jmax=float(torch.nan_to_num(_J, nan=1e30).max()), nJneg=int((_J <= 0).sum()), nbad=int(_bd.sum()),
                   hmin=float(_hp.min()), e99=float(torch.quantile(torch.nan_to_num(_ep[::7], nan=1e3).float(), 0.99)), emax=float(torch.nan_to_num(_ep, nan=1e3).max()), n_e20=int((torch.nan_to_num(_ep, nan=1e3) > 0.2).sum()))
        if int(_bd.sum()):
            _hb = _hp[_bd]
            _cl = torch.unique(torch.floor(xo.double()[_bd] @ lat.Ai.T.double() - 0.5), dim=0)
            _cs = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], device=dev, dtype=torch.float64)
            _dd = ((_cl[:, None, :] + 0.5 + _cs[None]) @ lat.A.T.double()) @ _nr - (FL["PN"] if FL else 0.0)
            _w = int(torch.argmin(torch.nan_to_num(_J, nan=-1e30)))
            _pd.update(bad_h=[float(_hb.min()), float(_hb.median()), float(_hb.max())], bad_ctr=xo.double()[_bd].mean(0).tolist(), bad_cells=int(_cl.shape[0]),
                       bad_cells_cross=int(((_dd.min(1).values < 0) & (_dd.max(1).values > 0)).sum()), bad_cells_below=int((_dd.max(1).values <= 0).sum()),
                       bad_in_band=int((_hb < FL["DH"]).sum()) if FL else None, worst=dict(p=_w, J=float(_J[_w]), x=xo[_w].tolist(), h=float(_hp[_w]), err=float(_ep[_w]),
                       cell=torch.floor(xo.double()[_w] @ lat.Ai.T.double() - 0.5).tolist()))
            PD_ON.append(fi + 1)
            if len(PD_ON) <= 5:
                np.save(os.path.join(a.out, f"pdiag_f{fi + 1}.npy"), torch.nonzero(_bd).squeeze(1).cpu().numpy()[:200000])
            print(f"  [pdiag] 프레임 {fi + 1}: J 최소 {_pd['Jmin']:.3g} (J≤0 {_pd['nJneg']}), 이상 입자 {_pd['nbad']} 높이 {_pd['bad_h']}, 칸 {_pd['bad_cells']} (관통 {_pd['bad_cells_cross']}, 아래 {_pd['bad_cells_below']}), 최악 {_pd['worst']}", flush=True)
        c["pd"] = _pd
    if a.qjd:                                                                          # 절 CM18 진단 (결과 불변): 프레임 끝 가우스 점 J
        c["qj"] = S.qjd(FL["NR"] if FL else [0.0, 0.0, 1.0], FL["PN"] if FL else 0.0)
    if a.gfin:
        c["grel"] = S.grel()
    if a.wdiag >= 0:                                                                   # 절 S.9 진단 (결과 불변, 상태 안 바꿈)
        from anchorflow import k4why_diag as KWD
        _Fe = torch.empty_like(S.F); _Fe[S.perm] = S.F
        _Jp1 = torch.linalg.det(_Fe.reshape(-1, 3, 3).double()); del _Fe
        _win = a.wdiag <= fi + 1 <= a.wdiag_end
        try:
            c["wd"] = KWD.frame(S, fi, a.tolN, _Jp0, _Jp1, _x0o, FL["NR"] if FL else [0.0, 0.0, 1.0], FL["PN"] if FL else 0.0,
                                out=os.path.join(a.out, f"wd_f{fi + 1}.npz") if _win else None)
        except torch.OutOfMemoryError as _ex:                                          # 절 S.10: 진단 계산만 메모리 부족 (풀이 상태 안 바뀜) -> 이 프레임 진단 없음
            c["wd"] = dict(off=True, oom=str(_ex)[:120])
        torch.cuda.empty_cache()                                                       # 진단이 잡은 메모리를 돌려줌 (풀이 메모리 여유를 원래 판과 같게)
        if _win and not c["wd"].get("off"):
            _w = c["wd"]
            print(f"  [wdiag] 프레임 {fi + 1}: 점 det 최소 {_w['Jqmin']:.3g} (F_q {_w['Jq0min']:.3g}, ≤0 점 {_w['nqinv']}), 입자 det 최소 시작 {_w['Jp0min']:.3g} 끝 {_w['Jp1min']:.3g}, 걸음 끝 |g|/F {_w['gend']}, 보폭 최소 {_w['st_min']:.3g}", flush=True)
    if a.fhash:
        import hashlib as _hl
        c["xh"] = _hl.sha1(xo.cpu().numpy().tobytes()).hexdigest()[:10]
    if a.hgt and fin:                                                                  # 절 CM6 진단: 높이 (바닥 법선 방향, 바닥 없으면 z) -- 질량 중심·최대·99 백분위, 정답과 우리
        _nh = torch.tensor(FL["NR"] if FL else [0.0, 0.0, 1.0], device=dev, dtype=torch.float32)
        _mw = m64.float() / float(m64.sum())
        for _nm, _xx in (("g", gt[_ok]), ("p", xo[_ok])):
            _z = _xx @ _nh
            c["h" + _nm] = [float((_z * _mw[_ok]).sum() / _mw[_ok].sum()), float(_z.max()), float(torch.quantile(_z[::7], 0.99))]
    flog.write(json.dumps(dict(t=fi, err=e, ms=S.last_ms, redo=redo, **c, **({"kernel_ms": kp} if kp else {}))) + "\n"); flog.flush()
    print(f"  프레임 {fi + 1}/{L} {S.last_ms:.2f} ms 오차 {100 * e:.3f}%{' (용량 늘려 다시)' if redo else ''} 점 {c['nq']} 노드 상자 {c['nn']} 유클리드 평균 점 {c['nbad']}{(' 반복 ' + str(c['it'])) if 'it' in c else ''}", flush=True)
    if PRED is not None:
        PRED[fi] = xo.cpu()
        if PREDF is not None:
            _fo = torch.empty_like(S.F); _fo[S.perm] = S.F
            PREDF[fi] = _fo.reshape(-1, 3, 3).cpu()
    if a.kbench and fi + 1 == a.kbench:
        kb = S.kbench()
        kb["_frame_ms"] = S.last_ms; kb["_counts"] = c; kb["_mg_cnt"] = S.mg_cnt.cpu().tolist() if hasattr(S, "mg_cnt") else None
        kb["_LM"] = S.LM.cpu().tolist() if hasattr(S, "LM") else None
        json.dump(kb, open(os.path.join(a.out, "kbench.json"), "w"), indent=1, ensure_ascii=False)
        print("[kbench]", json.dumps(kb, ensure_ascii=False), flush=True)
        L = fi + 1
        break
    if not fin or e > 2.0 or (MATX and float(S.v.norm(dim=-1).max()) > 200.0):
        div_at = fi + 1
        print(f"[발산] 프레임 {div_at}", flush=True)
        break
sc = [q for i, q in enumerate(ms) if i >= a.warm]
summ = dict(args=vars(a), frames=L, frames_done=len(errs), diverged=div_at, err_mean=(float(np.mean(errs)) if div_at is None else None),
            err_mean_done=float(np.nanmean(errs)), ms_mean=float(np.mean(sc)) if sc else None, ms_min=float(np.min(sc)) if sc else None)
if S.adapt:                                                                          # 반복 분포 (프레임별 뉴턴·PCG 합; 첫 warm 프레임 빼지 않음)
    its = [json.loads(l_).get("it") for l_ in open(os.path.join(a.out, "fem.jsonl"))]
    its = [q for q in its if q]
    if its:
        for kk in ("nN", "nP", "capN", "capP"):
            arr = np.array([q[kk] for q in its], dtype=float)
            summ["it_" + kk] = dict(mean=float(arr.mean()), p50=float(np.median(arr)), p90=float(np.percentile(arr, 90)), max=float(arr.max()), min=float(arr.min()))
if LIFT is not None:
    _el = [json.loads(l_).get("err_lift") for l_ in open(os.path.join(a.out, "fem.jsonl"))]
    LIFT["err_lift_mean_done"] = float(np.nanmean([q for q in _el if q is not None])) if _el else None
    summ["lift"] = LIFT
summ["pk"] = getattr(S, "pk_info", None)
if OOM:
    summ["oom"] = OOM
summ["ms_max"] = float(np.max(sc)) if sc else None
summ["ms_p90"] = float(np.percentile(sc, 90)) if sc else None
import hashlib                                                                        # noqa: E402
summ["x_sha1"] = None if OOM else hashlib.sha1(S.x_orig().cpu().numpy().tobytes()).hexdigest()[:10]     # 끝 상태 해시 (결정성 확인)
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
if BEN is not None:
    torch.save(dict(frames=BEN, A=S.A.float().reshape(3, 3).cpu()), os.path.join(a.out, "bener.pt"))
if PRED is not None and div_at is None:
    torch.save({"pred": PRED, **({"F": PREDF} if PREDF is not None else {})}, os.path.join(a.out, "pred.pt"))
elif PRED is not None and a.dump_part and len(errs):
    _n = len(errs)
    torch.save({"pred": PRED[:_n], **({"F": PREDF[:_n]} if PREDF is not None else {}), "part": True, "diverged": div_at}, os.path.join(a.out, "pred.pt"))
print(f"[롤아웃] {a.tag} 실시간 판 셀 {a.cell} K {a.K} 뉴턴 {a.newton} PCG {a.cg}: {len(errs)}/{L} 프레임{' (발산 %d)' % div_at if div_at else ''}, 평균 {100 * summ['err_mean_done']:.3f}%, "
      f"프레임당 {summ['ms_mean']} ms (재생 GPU 시간)", flush=True)
