# 속도-품질 파레토 비교 (MPM 계열 · 프록시 계열 · 우리) -- 가지 implicit-bspline, 커밋 6247369

조정자 11-05 지시 (사용자: "VR-GS 는 직접 구현하고 나머지도 적절하게 처리해서 측정 완료"). 씬은 탄성 lego L000 과 ficus (FPS 는 탄성만).
오차 = 롤아웃 평균 |x−x_gt|/EXT (우리 실행기와 같은 정의, EXT = 0 프레임 바운딩박스 대각선). 시간 = A6000 카드 0 단독 (gpu0.lock, 판마다 시작 때·실행 중 카드 0 의 다른 계산 프로세스 수를 card.txt 에 기록), 첫 2 프레임 뺀 평균·최대.

## P.0 진행 지점과 다음 할 일 (11-05 08:01 KST)
- 묶음 pareto1 을 잡 130567 (ece-a6gpu10) 에 띄움: `bash exe/pareto_launch.sh pareto1 'bash exe/pareto.sh 6247369'` (PA_DETACH=1). 시험 smoke1 (07:56) 통과.
- 결과: R2 lbours/pareto/ -- fit_<씬>_i<반복>/ (VR-GS E 맞춤 fit.json·log), t_<이름>/ (단독 시간 판: log·card.txt·apps_during.txt·summary.json/err.json/run.json), pareto.log, 표 pareto_table.md, 점 points.json, 끝 표식 pareto.END (12:12 KST 뒤 새 판 안 띄움, 안 띄운 판은 skipped.txt).
- **다음 할 일 (결과 받은 뒤)**:
  1. pareto.END 확인. 없거나 skipped.txt 에 판이 있으면 다음 잡에서 같은 커밋으로 다시 띄움 (excl 은 rc= 가 있는 판을 건너뜀 -- 단 /dev/shm 은 잡과 함께 사라지므로 R2 lbours/pareto/ 를 /dev/shm/dkta/lbours/pareto/ 로 먼저 내려받을 것; 마감 PA_UNTIL·PA_HARD 를 환경 변수로).
  2. 로컬에서 그림: `python3 exe/pareto_plot.py ~/r2/result/anchorflow/lbours/pareto/points.json <png>` (한국어 라벨 -- 기존 LaTeX 그림 fem_ak_local_map.png 이 한국어), /home/wgsong/workspace/writing/anchorflow/ 에 넣고 fem_results.tex 절 "속도-품질 파레토" 에 표·그림 덧붙임.
  3. 이 문서 P.4 에 표·결론, ~/workspace/result/anchorflow/outputs.md 에 결과 줄.
  4. Spring-Gaus (P.2) 는 사용자 결정 대기.

## P.1 VR-GS 직접 구현 (lib/anchorflow/vrgs_xpbd.py, 실행기 exe/pareto_vrgs.py)
논문 (arXiv 2401.16663v2, 11-05 다시 읽음) 에 있는 것: XPBD (Macklin 2016), 탄성 = "strain energy constraint", 가우시안 중심 = 사면체 무게중심 보간·공분산 F Σ0 Fᵀ, 케이지 정점 10K~30K (VDB -> marching cubes -> TetGen), 서브스텝당 반복 1~50 (표 1), PG 비교 설정 dt 1e-4 s·반복 11, 물성 E·ν·ρ (초기 E 1000·ν 0.3·ρ 1000, **E 는 바닥 자유 낙하 충돌 변형을 보고 손으로 조정, 10 회 안에**), 속도 감쇠 (계수 없음), Unity + CUDA 플러그인. 서브스텝 수·색칠·풀이 순서·감쇠 계수는 논문에 없음.
논문에 없어 표준 관행으로 메운 것 (정의):
- 케이지: 가우시안 중심 점유 격자 1 칸 팽창 -> 칸마다 Kuhn 6 사면체 (meshgs/tetcage.py 구조; TetGen 대신). 칸 크기 = 정점 ≤ 30,000 인 가장 작은 칸 -- lego 정점 29,999·사면체 143,694·칸 0.0184, ficus 29,985·127,734·0.0134 (smoke1).
- 변형 에너지 구속: 정답 재질 FCR 을 두 구속으로 나눠 에너지 정확히 일치 (C_R = ‖F−R‖, α_R = 1/(2μV); C_V = det F − 1, α_V = 1/(λV)), 사면체마다 2x2 결합 풀이, R = Müller 2016 반복 회전 추출 (사원수 따뜻한 시작 5 회).
- 가우스-자이델 + 탐욕 색칠 (lego 32 색, ficus 31 색), float32, 서브스텝 하나를 CUDA 그래프로 잡아 N 회 재생 + 중심 갱신 그래프. 시간 = 이것들의 GPU 시간 (이벤트).
- 질량 ρ V_T / 4 (ρ = 사면체 안 입자 밀도 평균; 케이지가 팽창돼 lego 케이지 질량 149 vs 입자 100.4, ficus 5.85 vs 7.87 -- 영역 밀도 차), 바닥 = 정점 비관통 z ≥ 0.48 반복마다 + 고착 (그 서브스텝 접선 이동 취소 = 정답 sticky), ficus 고정 상자 안 정점 역질량 0 (7,178 개), 감쇠 = config grid_v_damping_scale 같은 시간율 (ficus 0.9999 / 1e-4 s, lego 없음), 초기 속도 = 입자 속도의 질량·무게중심 가중 평균 (프레임 0 한 번). 서브스텝 안에 입자 없음.
- **물성 (논문 절차)**: 설정 (서브스텝 N x 반복) 마다 E 를 정답 궤적 롤아웃 오차 최소로 log10 E 황금분할 10 회 (정답 E 의 1e-2..1e2 배). ν·ρ 는 정답 config (lego E 1e6 ν 0.3, ficus E 2e6 ν 0.4 + 영역 밀도). smoke1 lego 15 프레임 2 회 평가에서 정답 E 의 0.34 배가 나음 (케이지 질량 1.5 배 탓으로 보임 -- 확인은 아님).
- 곡선: lego N {8,16,32,64,167 (dt 1e-4)} x 반복 {1,5,11}, ficus N {25,50,100,200,400 (dt 1e-4)} x 반복 {1,5,11}.
- 규모 대조 (논문 표 1, RTX 4090, 렌더 포함 fps): lego 케이지를 Horse 10,611·Bear 19,472·Fox 28,205 정점 이하로 짓고 가우시안 1.30M·2.53M·0.30M 의 중심 갱신 시간, 서브스텝 1 회 시간 (반복 1·5·11) -> 논문 프레임 시간 안에 들어가는 최대 서브스텝 수 (논문은 서브스텝 수를 안 밝혀 직접 fps 대조는 불가; 같은 차수인지 확인).

## P.2 프록시 계열 공개 코드 (PhysTwin·Simplicits 제외)
- **Spring-Gaus** (ECCV 2024): 코드 공개 (github Colmar-zlicheng/Spring-Gaus, Apache-2.0). 공식 절차 = 다시점 영상 -> 정적 3DGS 30k -> 속도 100 -> 동역학 300 반복, 서브스텝 n_step (dt = frame_dt/n_step), 바닥 투영 + 학습 마찰. 이 프로젝트에서 10-06 벤치에 다른 낙하 궤적 (taichi MPM, 34 프레임) 으로 돌린 적 있음 (exe/bench_sg.py, R2 bench-2026-10-06).
  - ficus: **공식 설정에 없는 것이 필요** -- 점별 고정 경계 (화분 cuboid) 없음 (FIX_* 는 학습 변수 동결), 충격은 균일 init_velocity 만. -> 실행 전 멈춤, 사용자 결정 필요.
  - lego L000: 바닥 낙하라 공식 범위에 가까우나 L000 다시점 렌더 (10 시점 x 80 프레임) + 정적·속도·동역학 피팅 (형상당 수 시간) 이 필요하고, 초기 속도가 균일 병진 하나로 표현되는지 (L000 초기 속도장) 확인 전. -> 실행 전 멈춤, 진행 여부 결정 필요.
- **GausSim**: 공개 저장소가 "Full code: TODO" 로 학습 코드가 없고 사전학습 가중치 데모만 있어 우리 궤적에 공식 절차로 맞출 수 없음 -> 제외 (우리 tools/train.py 는 재구성이라 GausSim 결과로 쓸 수 없음).
- Gaussian Splashing (PBD)·VR-GS: 공개 코드 확인 못함 (VR-GS 는 P.1 직접 구현).

## P.3 MPM 계열과 우리
- 정답 MPM: exe/pareto_mpm.py = exe/gt_sens.py (f605f25) 복사 + `--time` (서브스텝 루프 앞뒤 동기화 벽시계 -- PG warp 솔버 그대로라 파이썬 발사 포함), 격자 안 확인은 프레임 끝 한 번 (시간 밖). 공식 config (`--rpic_cfg`, ficus `--pg_cfg`) 에서 서브스텝 수만: lego {833 (공식), 416, 208, 104, 52, 26}, ficus {400 (공식), 200, 100, 50, 25}.
  큰 dt 쪽 CUDA 717/700 은 절 BD (bigdt2 에이전트) 결론 그대로: CFL 밖 폭주 -> 격자 밖 P2G (물리적 발산이 먼저). 그 조사는 하지 않음. smoke1: lego 104 서브스텝 3 프레임 0.009 %, 프레임 125~132 ms (비단독).
- 우리: 절 AN.22 최종 공통 설정 (`--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga hht --rho 0.8 --tolN 1e-2 --tolP 1e-1 --newton 2`, lego `--floor vel --fl_bar --fl_bst 1`, ficus `--floor none --cfg_bc --region_density --pg_vdamp`), 채택 K 2 · PCG lego 3 / ficus 20. 곡선: K {1,2,4} x PCG lego {1,2,3,5,10} / ficus {5,10,20,40}. 비결정 판 (시간 판의 오차를 그대로 씀).

## P.4 결과
(묶음 pareto1 결과 받은 뒤 채움)
