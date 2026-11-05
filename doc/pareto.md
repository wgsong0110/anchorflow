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

## S. 안정 최소 서브스텝 표 (논문용; 조정자 11-05 사용자 지시) -- 가지 implicit-bspline, 실행기 exe/stab.py · exe/stab.sh · exe/stab_launch.sh
지시: 방법별·씬별 "끝까지 발산하지 않는 최소 프레임당 서브스텝 수" N* 와 N* 에서의 지표 -- 위치 오차 (롤아웃 평균 |x−x_gt|/EXT), A6000 단독 프레임 시간, 시각 품질 PSNR·SSIM·LPIPS (3DGS 알파블렌딩, 정답 렌더와 같은 카메라). 씬 lego L000·ficus·wolf·bread (수박 제외). 탄성 전용 방법은 wolf·bread 에서 "해당 없음".
### S.0 정의
- 발산 = 기존 절 BD 정의 그대로: 프레임 위치 비유한 또는 프레임 오차 > 2 (= 200 %), 정답 MPM 은 추가로 서브스텝마다 입자가 격자 배열 밖 (`--sub_chk`). 실행기가 결과 파일 없이 죽으면 "중단" (발산 쪽으로 셈, 표에 따로).
- 탐색: 사다리 N = 1, 2, 4, … (방법별 상한 hi, 상한은 마지막에 그대로) -> 처음 안정한 N_s 와 그 앞 (발산) 사이 정수 이분 -> N*. 표에 경계 양쪽 (N* 안정 오차, 바로 아래 시험한 N 의 발산 프레임) 을 남김. 안정이 N 에 단조라는 보장이 없어 (절 BD 표: 명시적 B-스플라인 L000 K 1·2·8 완주, K 4 발산) **N* 뒤 2N*·4N* 도 돌려 발산이 있으면 † (비단조)**.
- N* 판의 오차 = 그 판정 판 (우리는 결정적 `--det`); 시간 = 같은 N 으로 카드 0 단독 판 (gpu0.lock, 시작 때 카드 0 의 다른 계산 프로세스 0 대기, 실행 중 5 초마다 기록, 첫 2 프레임 뺀 평균·최대; 우리는 비결정 판); 시각 품질 = 같은 N 의 덤프 판 (위치 + F) -> exe/lb_roll_to_h5.py -> exe/lb_render.py `--png` (정답 궤적과 같은 3DGS·같은 카메라 = cameras.json[0] 을 1.6 배 뒤로, 흰 배경) -> exe/stab_vis.py (프레임 1.. 평균, LPIPS alex). 정답 렌더 = 정답 h5 를 같은 렌더러로 (`--frames L`).
- 시간 정의는 실행기마다 절 P 와 같음: 정답 MPM = 서브스텝 루프 앞뒤 동기화 벽시계 (PG warp 파이썬 발사 포함), VR-GS = CUDA 이벤트 (그래프 재생 + 중심 갱신), 명시적 B-스플라인 = 프레임 앞뒤 동기화 perf_counter, 우리 = 재생 GPU 시간.
- 렌더 공분산: 각 방법 공식 방식 -- MPM·우리·명시적 = 입자 F Σ0 Fᵀ, VR-GS = 결속 사면체 F Σ0 Fᵀ (논문).
### S.1 방법별 설정 (상한 hi = 사다리 끝)
| 방법 | 실행기 | 서브스텝 손잡이 | 물성 | hi | 공식 밖 처리 |
|---|---|---|---|---|---|
| 정답 MPM (PG) | exe/gt_sens.py `--sub_chk --rpic_cfg` (+ 이번에 `--time` = pareto_mpm.py 와 같은 계시, `--dump_pred`) | `--nsub N` (dt = frame_dt/N) | 공식 config | 공식 서브스텝 L000 833 · ficus 400 · wolf 1999 · bread 100 | 없음 |
| VR-GS 직접 구현 | exe/pareto_vrgs.py (절 P.1) | 서브스텝 N, 반복 11 고정 (논문 PG 비교 설정) | **N 마다 E 를 논문 절차로 맞춤** (`--mode fit`, log10 E 황금분할 10 회; 안정 판정 = 맞춘 E 의 롤아웃) | L000 167 · ficus 400 (dt 1e-4 s) | 절 P.1 그대로 (케이지·ficus 고정 상자·초기 속도) |
| 명시적 B-스플라인 | exe/fem_bsel_cuda_sim.py `--theta 0 --vfl_out` (절 BD 의 bnv, + `--dump_F`) | `--N` | 정답 config | 1024 | 없음 (우리 표현 그대로) |
| 우리 (B) | exe/fem_an_rt_sim.py, 씬 인자 = exe/fem_cm8_scn.sh `cm_scene <씬> --v=b` (절 CM8 (B): 장벽 + `--y0eb`) + `--ga hht --rho 0.8 --newton 2 --tolN 1e-2 --tolP 1e-1`, PCG 상한 = cm8 (B) 고른 값 L000 4 · wolf 5, ficus 24 · bread 2 (cm4) | `--K N` (걸음 수) | 정답 config | 16 | 없음 |
| PhysTwin | exe/stab_phystwin.py (공식 81c7187; 준비 판 `--mode fit`, 판 `--mode run`) | num_substeps N (dt = 프레임 간격/N) | 준비 판에서 CMA-ES 20 세대 + 스프링별 경사 학습 200 반복 (논문 절차), 판은 맞춘 물성 그대로 | 맞춤 때 N0 (공식 dt 5e-5 를 프레임 간격에 맞춤; 단위 변환 뒤 L000 약 167 -- 실행 때 fit.json) | S.3 (1)~(5) |
| Simplicits | exe/stab_simplicits.py (kaolin v0.18.0 공식; `--mode weights`, `--mode run`) | timestep = 프레임 간격/N | 정답 config E·ν·밀도 (논문은 물성을 주어진 값으로 씀), 가중치 = create_trained 기본 인자 | 64 (공식 서브스텝 개념 없음) | S.3 (초기 속도 투영) |
| Spring-Gaus | exe/stab_sg.py (공식 62a1bb5; `--mode prep`·`train`·`run`) + exe/stab_mv_render.py (다시점 영상) | n_step N (dt = 프레임 간격/N) | 준비 판에서 공식 3 단계 학습 (정적 30k·속도 100·동역학 300), 판은 학습 물성 그대로 | 학습 끝 n_step (공식 일정이면 200 -- 실행 때 sg_train.json) | S.3 (1)~(3) |
| GausSim | 실행 안 함 (S.4) | | | | |
- 명시적 B-스플라인은 이 실행기에 재질 식이 없어 wolf·bread 를 "해당 없음 (실행기에 재질 없음)" 으로 적음 -- 탄성 전용은 아니므로, 재질을 넣어 돌릴지는 사용자 결정.
- 고친 실행기 (풀이 불변, 출력만): gt_sens.py `--time`·`--dump_pred`, fem_bsel_cuda_sim.py `--dump_F`, pareto_vrgs.py `--dump_F` (+ vrgs_xpbd.VRGS.F_part), lb_render.py `--png`.
### S.2 실행과 이어서 할 지점
- 로컬: `PA_DETACH=1 PA_JOB=<잡>.ECE-util1 PA_NODE=<노드> bash exe/stab_launch.sh stab1 'ST_UNTIL=<epoch> ST_HARD=<epoch> bash exe/stab.sh <커밋>'` (커밋 묶음 -> R2 lbours/stab/code -> 노드 /dev/shm/dkta/impwt/<커밋>). 먼저 연기 시험 `ST_OUT=/dev/shm/dkta/lbours/stab_smoke ST_ARGS="--smoke 3"` (방법 x 씬마다 상한 한 판 3 프레임 -> 덤프·렌더·시각·시간 경로) 를 같은 방법으로 띄우고 R2 lbours/stab_smoke/stab_table.md 로 확인.
- 비측정 판은 빈 카드 1..7 (anerr_claim 규약 -- 다른 에이전트와 공유, 동시 6), 시간 판은 카드 0 하나씩.
- 결과: R2 lbours/stab/ -- p/<방법>_<씬>_n<N>[_dump|_time]/ (log·res.json·실행기 출력·pred.pt·png·vis.json), gt_<씬>/png, stab.log·stab_main.log, 표 stab_table.md·stab_table.tex·stab.json (2 분마다), 끝 표식 stab.END.
- 이어 돌리기: stab.sh 가 시작 때 R2 의 p/*/res.json·vis.json·gt_*/png 를 내려받아 끝난 판은 건너뜀 (덤프가 없으면 덤프 판만 다시). 같은 커밋으로 다시 띄우면 됨.
- LaTeX: writing/anchorflow/fem_results.tex 끝 절 "안정 최소 서브스텝" 이 `~/r2/result/anchorflow/lbours/stab/stab_table.tex` 를 `\IfFileExists` 로 읽음 (잡 안 표가 갱신되면 다시 컴파일만).
### S.3 PhysTwin · Simplicits · Spring-Gaus (11-05 실행기 준비 -- GPU 실행 확인 전, 연기 시험이 첫 확인)
공통: exe/stab_bl_common.py (anchorflow 라이브러리 없이 -- simp 환경에서도 돌게; 씬 .pt 적재·PG 질량·EXT·고정 상자 판정·오차/발산 (프레임 오차 > 2 또는 비유한)·summary.json (bnv·우리와 같은 열쇠)·덤프 pred.pt {pred, F}). 오차는 모든 방법과 같게 **정답 전 입자** 평균 |x−x_gt|/EXT. 탄성 전용이라 wolf·bread 는 "해당 없음".
stab.py 연결: 방법 phystwin·simplicits·springgaus (기본 --methods 에 포함). 씬마다 사다리 앞에 준비 판 p/<방법>_<씬>_prep/ (빈 카드 하나, 상한 16 h, 끝나면 res.json {hi}) -> 상한 hi = PhysTwin fit.json N0 · Spring-Gaus sg_train.json n_step · Simplicits 64 -> 사다리·이분·단조 확인·덤프·시간 판은 다른 방법과 같음 (판 명령 = 각 실행기 `--mode run --fit <준비 판> --N <N>`). 연기 시험 (`--smoke 3`) 은 준비 판 반복만 줄임 (PhysTwin CMA 1 세대·학습 1 반복, Spring-Gaus 정적 300·속도 2·동역학 2·다듬기 10).
환경: exe/stab_bl_setup.sh (stab.sh 가 stab.py 앞에서 부름, 로그 <out>/bl_setup.log, 끝 줄 BLSETUP rc=) -- 공유 환경 af·simp 는 안 건드리고 모자란 것만 /dev/shm/dkta/pylib/{pt,sg,simp18} 에 pip --target --no-deps (PhysTwin: af 의 warp 가 1.x 아니면 warp-lang 1.x·cma·kornia 등; Spring-Gaus: pytorch3d·그쪽 서브모듈 래스터라이저·simple_knn 이 af 에 없으면 잡 안 빌드; Simplicits: simp 의 kaolin 이 0.18.0 아니면 kaolin 0.18.0 휠을 simp 의 torch·CUDA 에 맞춰). 공식 코드 사본이 고정 커밋이 아니면 /dev/shm/dkta/ext/ 에 따로 받음. 결과는 ST_* 환경 변수 (/dev/shm/dkta/pylib/bl_env.sh). 이어 돌리기: stab.sh 가 R2 의 p/*_prep/** (mv·data 빼고) 도 내려받음.
- **PhysTwin** (exe/stab_phystwin.py; 공식 Jianghanxiao/PhysTwin 81c7187, 로컬 ~/workspace/external/physics/PhysTwin, 클러스터 /home/dkta/work/PhysTwin): 준비 판 = 공식 OptimizerCMA (CMA-ES 20 세대, 12 매개) -> InvPhyTrainerWarp.train (스프링별 log Y, Adam lr 1e-3, real.yaml 200 반복) 을 정답 궤적 전 프레임에 (L000 = 공식 synthetic 경로 = 전체 점 궤적·추적 손실; ficus = 조작점이 있는 공식 real 경로 = chamfer + 추적 + 가속도 손실, 가시성·추적 유효 전부 참). 판 = 공식 pure_inference 트레이너 + test() 체크포인트 적재 + visualize_sim 프레임 루프 (forward_graph), 서브스텝만 N. 전 입자 위치 = 공식 대화형 데모의 가우시안 이동 (interpolate_motions_speedup, 이웃 16, 이웃은 첫 프레임에 한 번), F = 누적 회전 (렌더 R Σ0 Rᵀ). 시간 = forward_graph + 보간 (프레임 앞뒤 동기화).
  공식 밖: (1) 단위 -- 반경 범위·dt·복셀 0.005 m 가 실측 물체 ~0.25 m 기준이라 길이 K = 0.25/(0 프레임 상자 최대 변) 배·시간 c = √(K|g|/9.8) 배 (닮음 변환, 코드에 박힌 중력 9.8 과 맞음; exe/phystwin_fit.py 와 같은 처리), z 이동 (L000 바닥 0.48 -> 0, ficus 는 전 프레임 최저점 아래). 그래서 N0 = round(c·프레임 간격/5e-5) 이고 dt = c·프레임 간격/N0 (5e-5 를 반올림한 서브스텝 수로 프레임 간격에 정확히 맞춤 -- 예전 S.3 의 L000 333·ficus 800 은 단위 변환 전 값). (2) ficus 화분 = 공식 조작점을 화분 상자 안 질량점으로, 모든 프레임 같은 위치 (그래서 real 경로; 손실이 L000 과 다름). (3) 초기 속도 = 정답 0 프레임 입자 속도 (예전 계획의 첫 두 프레임 차분 대신 저장된 정확한 값; synthetic 은 공식 velocity_path, real 은 그 인자가 없어 시뮬레이터 초기 속도 배열에 같은 값). (4) 질량점 = 정답 입자 무작위 11024 개 (rep_track2·fps_phystwin 과 같은 수, 씨앗 0; 예전 공식 설정판은 가우시안 중심이었으나 여기 .pt 에는 가우시안 수가 없어 전 입자) -> 공식 data_process_sample 복셀 0.005 m 처음 점; 가우시안 이동 뼈대에 ficus 조작점 (움직임 0) 을 더함. (5) 렌더·대화형 import (pynput·gaussian_splatting 렌더러·gs_render·pyrender·wandb 없으면) 는 빈 모듈, dynamic_utils 의 ipdb.set_trace 는 예외 -> 그 프레임 발산 (보간 SVD 실패), knn_weights_sparse 는 4000 점씩 나눠 부름 (같은 계산, CPU 메모리).
  남은 갈래: dt 를 바꿀 때마다 다시 맞추는 판은 논문 절차가 아니라 안 함.
- **Simplicits** (exe/stab_simplicits.py; kaolin v0.18.0 kaolin.physics.simplicits.easy_api, 로컬 ~/workspace/external/physics/kaolin; 잡 안 simp 환경): 준비 판 = create_trained 기본 인자 (핸들 10, 표본 1000, 층 6, 1 만 스텝; exe/simplicits_weights.py 와 같은 호출), 점 = 정답 0 프레임 전 입자 (공식 예제가 3DGS 를 부피 표본으로 채운 점을 쓰는 역할), 물성 정답 config E·ν·밀도 (ficus 영역 밀도), 부피 = 입자 부피 합. 판 = SimplicitsScene 기본 설정 (뉴턴 5·직접 풀이) 에 timestep = 프레임 간격/N, add_object 기본 적분점 1000 (씨앗 0), set_scene_gravity (config g), L000 set_scene_floor (높이 0.48, z 축, 기본 벌점 1e4), ficus set_object_boundary_condition (상자 안 적분점, 기본 벌점). 전 입자 = 공식 예제 transform_gaussians_lbs 와 같은 점별 변환 I + Σ w T (가중치 한 번), F = 그 3x3. 시간 = run_sim_step N 회 + 전 입자 변환.
  공식 밖: 초기 속도 -- API 에 없어 장면을 다 세운 뒤 sim_z_dot 에 정답 0 프레임 속도를 적분점에서 최소제곱 투영 (B ż ≈ v, 상대 잔차를 summary 에 기록). 감쇠 없음 (공식에 없음; ficus PG 격자 감쇠 안 넣음).
- **Spring-Gaus** (exe/stab_sg.py + exe/stab_mv_render.py; 공식 Colmar-zlicheng/Spring-Gaus 62a1bb5, 로컬 ~/workspace/external/physics/Spring-Gaus, 클러스터 /home/dkta/work/Spring-Gaus): 준비 판 = (a) stab_mv_render: 정답 h5 궤적을 lb_render 와 같은 3DGS·공분산으로 10 시점 512x512 (공식 합성 데이터 N_CAM·H·W), 검정 배경 (공식 로더 bg 0) -> (b) `--mode prep`: 공식 mpm_synthetic 형식 (camera.json blender c2w + cam_<i>/<프레임>.png), 설정 = 공식 default.yaml 상속 + DATA 절 (N_FRAME = FRAME_ALL = 전 프레임, HIT_FRAME = 정답이 바닥에 처음 닿는 프레임, GLOBAL_M 1·K 1000·DAMP 0.1 = 공식 합성 config 값) -> (c) `--mode train`: 공식 train.py 의 train_static -> train_velocity -> train_dynamic 을 __main__ 블록과 같은 순서로 (끝의 공식 자체 평가 eval_fitting·eval_prediction 만 건너뜀). 판 = 공식 test 규약 (--dy_reload, get_simulator) 적재 뒤 n_step = N, 우리 입자를 공식 set_all_particle (K_BINDING 16, 1/√d) 로 앵커에 묶어 앵커 변위로 이동 (공식 interpolate 의 velocity 단계 식 -- dynamic 단계 식은 위치를 앵커 가중 평균으로 바꿔 우리 입자에 정지 편향이 생김), 공식 is_nan 은 발산, F = I (그쪽 가우시안은 등방·회전 없음).
  공식 밖: (1) 단위 -- 공식 합성 데이터 크기 (물체 ~0.4, [0,1] 상자) 에 맞춰 길이 Ks = 0.4/(0 프레임 상자 최대 변)·시간 √(Ks|g|/9.8) 배 (닮음 변환, 영상은 카메라 위치만 옮겨 그대로), 중력 z 축 (공식 코드가 BC 법선 z 를 지원; 시뮬레이터는 단계 설정 VELOCITY.G·DYNAMIC.G 를 읽고 cfg.MODEL 을 읽는 곳이 없음 (코드 확인) -- 예전 exe/mk_sg_config.py 는 MODEL.G 에만 적어 단계 G 가 기본 [0,−9.8,0] 이었고 BC 축 z 마스크와 곱해져 중력 0 이었던 것으로 보임 (코드 읽기로만, 실행 확인 아님); 여기서는 VELOCITY.G·DYNAMIC.G 에 [0,0,−9.8]). (2) ficus -- 바닥 없음이라 BC 바닥을 전 프레임 최저점 아래에 (안 닿음); 화분 상자 안 앵커는 서브스텝마다 처음 위치·속도 0 (Spring_Mass.step 뒤에 덮어씀; 공식에 점별 고정 없음); 초기 충격이 국소라 공식 균일 INIT_VELOCITY 로 표현 안 돼 속도 단계를 공식 규약대로 건너뛰고 (init_velocity_100.json = 0) 앵커 init_v = 정답 0 프레임 속도의 최근접 8 입자 평균 (학습 안 함). L000 은 초기 속도가 강체 병진이라 공식 속도 단계 그대로. (3) open3d ply 덤프 간격만 끔 (VIZ_INTERVAL 999999, 예전 노드 세그폴트).
  비용 추정 (예전 벤치 wolf 34 프레임 동역학 100 반복 52 분에서 비례, 확인 아님): L000 81 프레임 약 6 h, ficus 126 프레임 약 9~10 h + 정적 30k.
### S.4 GausSim -- 직접 구현 가능성·비용 (실행 안 함)
- 공개 저장소 (ftbabi/GausSim_ICCV2025 a85da5a, 로컬 ~/workspace/external/physics/GausSim_ICCV2025): README TODO "Full code" 미체크. 실제로 있는 것 -- 모델 (mmgs/models: 계층 MeshGraphNet backbones/meshgraphnet_hie.py, simulators/gs_simulator_hierarchy.py 의 forward_train, heads/acc_decoder.py 의 렌더·static·운동량 손실), 학습 API (mmgs/apis/train.py 141 행), 학습 config (configs/gssim/iccv/pudding.py: Adam 2e-4, 롤아웃 1 -> 16 단계 증가, EpochRunner), 다시점 영상 데이터셋 클래스. 없는 것 -- 학습 진입 스크립트 tools/train.py, 데이터 준비 (점 표본·계층 군집·pin_mask 만드는 코드; 공개 데이터는 pudding 데모 하나), 사전학습 가중치 외 체크포인트.
- 즉 "직접 구현" = 진입 스크립트 (mmcv 표준 train 런처, 수십 행) + 우리 씬을 그쪽 데이터 형식 (다시점 영상 + 가우시안 + 군집 계층 + pin) 으로 만드는 변환. 이 프로젝트에 예전 재구성이 있음 (imp_wt/gaussim/train.py·make_dataset_from_mpm.py·apply_patches.py) -- 메모리 규칙대로 결과는 "GausSim 공개 조각으로 재구성한 학습" 으로만 보고 가능.
- 서브스텝 개념: 학습된 시뮬레이터가 프레임 간격 (config dt 1/30) 한 걸음씩 -- 프레임당 서브스텝은 구조상 1 고정이라 "최소 서브스텝" 탐색 대상이 아님 (표에는 N = 1 의 발산 여부·지표만 가능).
- 비용 추정: 논문 A800 4 장 batch 4 로 22 시간 수렴 (메모리 gaussim-official-facts) -> A6000 4 장 기준 씬당 하루 안팎 (확인 아님), L000·ficus 두 씬 = 카드 4 장 x 이틀. 3090 24GB 에서 롤아웃 16 OOM 기록 -> A6000 48GB 는 들어갈 가능성 (확인 아님). 데이터 변환·진입 스크립트 작업 반나절. 탄성 전용이라 wolf·bread 해당 없음.
### S.5 진행 (11-05)
- 13:27·13:3x KST: 로그인 노드 SSH 가 연결 거부 (`kex_exchange_identification: Connection reset by peer`, ControlMaster 소켓도 응답 없음) -> 잡 상태 확인 못 함. 코드만 준비 (위 S.1 실행기 수정 + exe/stab.py·stab.sh·stab_launch.sh·stab_go_tmpl.sh), 로컬 표 생성 함수만 가짜 결과로 확인 (GPU 실행 확인은 아직 -- 연기 시험이 첫 확인).
- 13:53 KST 다시 시도 (15 분 뒤, ControlMaster 경유): 90 초 안에 응답 없음 -> 잡 확인·띄우기 못 함. **다음 할 일**: `ssh dkta@147.46.121.39 /opt/pbs/bin/qstat -answ1 -u dkta` 로 잡 (130784 등) 이 R 인지·노드 확인 -> nvidia-smi 로 빈 카드 확인 (cm9 수박 에이전트와 공유) -> 연기 시험 `PA_DETACH=1 PA_JOB=<id>.ECE-util1 PA_NODE=<노드> bash exe/stab_launch.sh stabsm 'ST_OUT=/dev/shm/dkta/lbours/stab_smoke ST_ARGS="--smoke 3" bash exe/stab.sh <커밋>'` -> R2 lbours/stab_smoke/stab_table.md 에 모든 칸 (오차·ms·PSNR) 이 차면 본 묶음 `... stab_launch.sh stab1 'ST_UNTIL=<잡 만료−10 h epoch> ST_HARD=<잡 만료−1 h epoch> bash exe/stab.sh <커밋>'`. 그다음 S.3 실행기 (PhysTwin → Simplicits → Spring-Gaus).
- 14:17 KST: S.3 실행기 세 개를 로컬에서 준비 (문법 검사만, GPU·클러스터 실행 없음 -- 클러스터 SSH 불통) 하고 stab.py 기본 방법에 연결 (위 S.3). **첫 잡에서**: 연기 시험 명령은 위와 같음 (`ST_ARGS="--smoke 3"`; 이제 세 베이스라인도 포함) -> 먼저 <out>/bl_setup.log 끝 줄 BLSETUP rc=0 확인 (실패면 그 항목만 고치고, 다른 방법은 그대로 돎) -> stab_table.md 의 PhysTwin·Simplicits·Spring-Gaus 칸과 p/<방법>_<씬>_prep/log_*.txt 확인 -> 본 묶음. 준비 판이 길어 (Spring-Gaus 씬당 6~10 h) 본 묶음을 일찍 띄울 것.
