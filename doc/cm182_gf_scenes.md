# 절 CM182 GaussianFluent 공개 장면 (watermelon 제외) -- 정답 궤적 + 우리 판 (2026-11-15)

## 1. 정답 궤적 (GF 공식 러너·공식 config 그대로)
자산: HF `hbpencil01/GaussianFluent` -> R2 `datasets/anchorflow/gf_scenes/{model,config}` (watermelon·garden 제외, 4.65 GB). 정답 h5 + 로그 -> `datasets/anchorflow/gf_scenes/gt/<장면>/`.
러너 = `gs_simulation.py.orig`. 달라진 것은 **배경 읽기만** (`exe/patch_gf_bg.py`: 저자 로컬 경로 `garden_ours` 를 `--render_img` 일 때만 읽게 감쌈, 시뮬 코드 그대로). 원본이 `IndexError` (채우기 입자 수 대 껍질 마스크) 를 내는 장면 (sand_castle) 만 R2 사본 `gs_simulation.py` (껍질 beta 블록 끔) 로.

| 장면 | 프레임 | 정답 이웃 이탈 최대 | 위상 변화 |
|---|---|---|---|
| sand_castle | 100 | 17.75 % (영구) | 있음 |
| pumkin | 100 | 4.96 % | 있음 |
| toast | 150 | 4.01 % | 있음 |
| lollipop | 90 | 5.56 % | 있음 |
| pineple | 60 | 10.96 % | 있음 |
| milk | 50 | 71.89 % | 있음 (유체 성격) |
| 젤리+총알 | 520 | 4.41 % (마지막 4.12 %, 잔존 93 %) | 있음 |
| bullet (총알 자산만) | 24 | 0.08 % | 없음 |
| dragonfruit, kiwi, cake | 60, 60, 24 | 0.00 % | 없음 |

이웃 이탈 = 초기 k-NN (k 16) 이웃이 초기 거리의 3 배 넘게 벌어진 비율 (`exe/measure_gf_tearing.py`). 장면 선정: 사용자 11-15 지시로 fruitninja·cookie·oreo·bowl 제외 (논문에 없음), toast 는 GF Table A1 `tosta` 와 이름·E·ν·M 이 같아 유지 (α0·Δx·ρ 는 표와 공개 config 가 완전히 같지 않다 -- watermelon 도 ρ 가 다름).

### 젤리+총알
공식 러너 `gs_simulation_jellynacc.py` 는 총알을 `save_bullet_prop.pt` (HF·저장소에 없음) 에서 읽는다. 러너 안 주석 처리된 생성 블록 (총알 1/3 축소·y+0.7·`save_core_init_render_vars`) 을 그대로 되살려 생성 (`exe/patch_gf_jelly.py --mode gen`). 검산: 총알 입자 83,580 개 중 83,578 개가 config 의 속도 구속 상자 안. 영상 `result/anchorflow/v20261009/gf_jellybullet_gt.mp4` (배경 없음, 총알 가우시안 1 개 (공분산 trace 8117, 불투명도 0.0044) 의 불투명도만 렌더에서 0). 공식 러너는 시뮬을 프레임 0~19, 200~239, 381 이후에만 진행한다.

## 2. 우리 판 (육면체 B-스플라인 2 차, 가우스점 2x2x2, K=1 HHT, 공통 기본 H9 명령)
명령 = R2 `lbours/wolfcv/cm157/BA_1/cmd.txt` 에서 `--data`·`--frames`·`--tag`·`--out`·바닥·`--rr`(렌더 대상 pg -> gf)·덤프만 바꿈 (`exe/gfs_cmd.py`). 뺀 것은 진단 기록 감싸개 startpt 하나 (입자 100 만 개 넘는 장면에서 44 GiB 버퍼). 렌더 훅 `--rr gf` 는 풀이 상태를 읽기만 한다.

장면별 대응 (모든 장면 같은 규칙):
- **바닥**: GF 는 격자 경계 padding 3 칸 (z = 3 dx) 이 실제 바닥, config 의 z=0 sticky 평면은 작동 안 함 -> 우리 판이 읽는 .pt 안 config 사본의 sticky 평면 높이를 max(원래, 3 dx) 로. 정답 시뮬 config 는 그대로.
- **cut (surface_type 11)**: 평면 아래 노드 속도를 매 서브스텝 0 (z<0.4) 또는 (vx,0,vz)x0.3 (슬랩). 프레임당 서브스텝 ~126 이면 0.3^126 = 0 이라 프레임 단위에서 sticky 와 같다 -> 사본 config 에서 같은 높이 sticky 평면으로 (kiwi, pineple, sand_castle). 슬랩의 y 성분 구속은 표현 안 됨 (근사). lollipop 은 cut 점 z=0 이라 GF 에서도 무효. (하위 에이전트가 만든 노드 상자 구속판 `exe/cutbc_wrap.py` 는 재질 판이 고정 구속 상자를 못 받아 (assert) 쓰지 못함.)
- **particle_impulse (sand_castle)**: `exe/impulse_wrap.py` (입자 속도 충격).
- **substep_dt**: GF 러너는 config 값을 무시하고 0.6 dx/음속 으로 다시 계산 (숨은 기본값 2번) -> 사본 config 의 `substep_dt` 를 GF 실제값으로 (격자 속도 감쇠 지수).
- **GF 의 FLIP/PIC 비율**: 우리 판에 대응 옵션이 없다 (milk 0.99, pineple 0.9 등 소산 차이로 남음).
- **n_grid 없는 config (cake)**: PG·GF 기본값 n_grid 50, grid_lim 2.0.
- 메모리: 입자 100 만 개 넘는 장면은 `--cap_mul=2` (용량 버퍼만, 결과 불변이라 코드에 적힘).
- **K 규칙** (사용자 11-15): 발산하면 K 를 2, 4, 8 순서로 올려 처음 완주하는 값. K>1 판은 `--ksplit 3 --ks_big` 를 뺌 (`--ksplit` 은 K=1 에서만). 완주한 K: pineple K=2. 완주 못한 K: pumkin K=2·4, milk K=2·4·8 (K=1 과 같은 시점대에서 발산).

## 3. 결과 (완주한 장면만, 표 `writing/anchorflow/gf_scenes_results.tex`)
표: R2 `lbours/gfscenes/ours/<실행>/{summary,topo,visual}.json` -> `exe/gfs_table.py`. 오차 = 정답 대비 입자 평균 위치 오차 (기존 정의). 프레임 시간은 진단·렌더 훅이 켜진 **비단독** 값 (단독 측정은 하지 못함).

발산 (부분 결과는 표에 쓰지 않음): toast 9/150, pumkin 9/100, lollipop 10/90, cake 16/24, milk 28/50, sand_castle 28/100, 젤리+총알 135/199 (프레임 136 GPU 메모리 부족, 노드 상자 2,597 만).

## 4. 발산 원인 (pumkin·pineple·toast 진단, 확인한 것만)
- 발산 직전 로그: pumkin 은 프레임 1~2 정상 (Newton 4, 3 회, 잔차비 1.0/0.52/0.44), **프레임 3 에서 바닥 근처 (z≈0.05, 평면 z=0.02) 가우스점 반전 8 개** 로 시작, 프레임 3 부터 Newton 잔차 NaN, 반전 입자 21,723 -> 237,160 -> 392,737, 입자 최대 높이 1.25 -> 2.12 (정답 1.25 고정), 노드 상자 600 만 -> 2,080 만 노드 -> 프레임 10 GPU 메모리 부족 (표시는 "발산"). 정답 이웃 이탈 2.7 %, 우리 37 %.
- pineple: config 가 β=0, 마찰각 0 (κ=0, 고스트 강성 0, 확장 켜짐) -- 프레임 1 부터 Newton NaN, 반전 38,681 개 (z=0.52, cut 을 바꾼 평면 바로 위). 퇴화 영역인지 바닥 충돌인지는 근거로 못 가름.
- 원인 분리 (pumkin 20 프레임): 기본 = 프레임 3 첫 반전·프레임 10 발산. (a) 초기 속도 0: 20 프레임 완주하나 낙하해 닿는 프레임 11 부터 같은 반전 (프레임 15 에 279,374). (b) 탄성 재질 (같은 E·ν): 프레임 3 첫 반전 (9 개), 프레임 5 에 197,603. (c) 미끄럼 바닥: 기본과 거의 같음 (프레임 3 에 7 개, 프레임 10 발산). => 재질 (NACC)·바닥 종류 (sticky 대 미끄럼) 는 원인이 아니고 충돌 속도는 시점만 바꾼다. 바닥 접촉 순간에 반전이 생긴다 (cake 탄성 장면 발산과도 일치).
- 후보 2 (옛 재질 실행기 `fem_ao_rt_sim.py --ls_inv` 등, 수박 M 판 명령): pumkin 16/100 (오차 101.9 %), toast 14/150 (103.2 %) 에서 발산 -- 실패. 후보 3 (β=0·φ=0 퇴화 경로) 은 시험 못함.
- 최대 속도는 로그에 직접 없다 (입자 최대 높이 증가로 ~14 m/s 이상 추정).
- 바닥 접촉 뒤집힘 수정과 NACC 장면 재실행은 다른 에이전트가 맡았다 (사용자 11-15).

## 5. 인스턴스·운영 메모
- 전력 한도 호스트 destroy·blocklist: machine 700, 59000, 138839, 123134, 123713, 43621, 29089, 154960, 38441 (+ CUDA 고장 150494, loading 정지 46238).
- **래스터라이저 wheel**: 릴리스 `cuda-3dgs-torch251` 의 wheel 은 `TORCH_CUDA_ARCH_LIST="8.6;8.9"` 라 A100 (sm_80) 에서 `torch.OutOfMemoryError ... 131070 GiB` (잘못된 값 읽기). 워크플로 (브랜치 `cuda-sm80-wheel`, 커밋 efe5fbc, `8.0;8.6;8.9;9.0`) 로 다시 빌드해 같은 태그에 덮어씀 (08:52 UTC). A100 에서 우리 판·렌더 정상 확인 (dragonfruit 3 프레임, PSNR 45.67).
- vast 크레딧은 다른 에이전트와 같은 계정이라 크게 흔들렸다 (0 으로 인스턴스가 exited 된 적 있음).
- 렌더 지표 (PSNR·SSIM·LPIPS) 는 발산 뒤 프레임을 마지막 유한 프레임으로 멈춰 그린 영상 기준이라 발산 장면에서는 낮다.
