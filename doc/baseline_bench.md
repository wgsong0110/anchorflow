# 베이스라인 벤치마크 계획 (PG 기준)

2026-09-26 작성. 목적: **i-PhysGaussian / GausSim / GS-Verse** 를 PhysGaussian(PG)
기준으로 12 개 형상-물성 조합에 대해 같은 자로 재고 표로 정리한다.

## 0. 원칙

- **기준(GT)은 PG** 다. 모든 오차 지표는 PG 의 같은 프레임과 비교한다.
- **입력을 완전히 통일**한다: 같은 채우기 입자 집합, 같은 물성 json, 같은 격자
  (100³, 영역 [0,2]³), 같은 프레임 수, 같은 손잡이 계획.
- **손잡이 계획은 학생 학습과 같은 규약**이다 (아래 1 절). 목표점은 유한 고정
  후보에서 뽑고, 이동은 **가속도 2.4 / 최고속도 0.6 고정**이며 목표에 닿으면
  다음 목표를 뽑는다. 고정 시간으로 끊지 않는다.
- 상대가 공개하지 않은 것은 **추측해서 채우지 않는다**. 못 잰 칸은 "코드 미공개"
  로 남기고 그 이유를 적는다.

## 1. 공통 시나리오 (제어점 이동 규약)

`lib/anchorflow/scene_pool.py` 의 `HandlePlan` 을 그대로 쓴다.

- 제어 입자: 손잡이 **2 개**, 서로 2R(=0.30) 이상 떨어진 입자에서 뽑는다. R=0.15.
- 목표점: `target_grid(domain=2.0, margin=0.30, n_side=5, clear=R)` 의 **유한 고정**
  후보 48 개 중 하나. 바닥면(z=0.48, sticky)에서 R 이상 띄운 점만 남긴다.
- 지령 속도: `s = min(vmax, a·t, sqrt(2 a d))`, a=2.4, vmax=0.6, 도달 허용 5e-3.
  → 출발에서 a 로 가속, vmax 로 순항, 남은 거리가 감속 거리에 들면 a 로 감속해
  목표에서 정지. 도달 시간은 거리에 따라 다르다 (0.46 → 57 프레임, 1.15 → 127).
- 도달하면 **그 자리에서** 새 제어 입자·목표점을 뽑는다.
- 감쇠 가중치는 PG 규약 `w=(1-q²)²`, q = ‖x-c‖/R.
- 씬당 시나리오 **8 개**(시드 고정), 각 **240 프레임**. 12 조합 × 8 = 96 회.

시나리오는 미리 한 번 만들어 파일로 고정한다 (`bench/scen_{조합}_{시드}.json`:
제어 입자 색인, 목표점 열, 프레임 수). 모든 솔버가 **같은 파일**을 읽는다.

## 2. 측정 항목

| 지표 | 정의 | 비고 |
|---|---|---|
| CD | PG 대비 Chamfer distance, 2048 점 부분표본 | 같은 색인으로 뽑는다 |
| EMD | PG 대비 Earth Mover, 같은 부분표본 | 같은 색인 |
| 물리잔차 | `r = ∂E/∂x`, `‖r‖h²/m/EXT` 의 평균 | i-PG 목적함수, **바닥 접촉항 포함** |
| 부피비 | 프레임별 `det F` 중앙값의 max/min | 1 에 가까울수록 좋다 |
| 운동량 | `Σ m v` 의 성분별 max−min, 그리고 PG 대비 비 | 손잡이가 외력이라 보존은 아니다 |
| 에너지 | 운동+중력+탄성 합의 max−min, PG 대비 비 | 같은 구성모델로 계산 |
| 영역 관통 | 바닥면 아래·경계 밖으로 나간 깊이의 질량가중 합 | 이번에 넣은 접촉항의 기준면 |
| det F < 0 | `det F` 가 음수인 입자 비율 (프레임 최대·평균) | 뒤집힌 원소 |
| FPS | 프레임당 시간의 역수, 워밍업 3 프레임 제외 | GPU 단독 점유에서 잰다 |
| 총 실행시간 | 240 프레임 전체 벽시계 | 초기화 포함/제외 둘 다 |
| 시각 품질 | 렌더 후 PG 렌더 대비 PSNR / SSIM / LPIPS | 같은 카메라·같은 프레임 |

시각 품질은 **PG 렌더를 기준 영상**으로 보고 프레임별 PSNR/SSIM/LPIPS 를 재고,
시간축 흔들림은 연속 프레임 차의 L1 (`temporal flicker`) 로 같이 낸다.

## 3. 대상과 실행 가능성

2026-09-26 정정: GausSim 과 GS-Verse 는 **둘 다 코드가 공개돼 있다**. 앞서
미공개로 적은 것은 잘못이다.

| 대상 | 코드 | 상태 |
|---|---|---|
| PG | `/home/dkta/work/PG_pgtraj` | 기준. 그대로 실행 |
| i-PhysGaussian | `/home/dkta/work/i-physgaussian` | Newton-GMRES (`AF_IPG_NEWTON=1`). 손잡이 기능이 없어 `exe/patch_ipg_scen.py` 로 같은 채우기·시나리오 구동을 넣었다. dt×8 기본, ×20(논문 세팅)도 잰다 |
| GausSim | [ftbabi/GausSim_ICCV2025](https://github.com/ftbabi/GausSim_ICCV2025) | 모델·학습·평가 전부 공개 (`mmgs/models/simulators/gs_simulator_hierarchy.py`, `mmgs/apis/train.py`). 빠진 것은 `tools/train.py` 진입점과 READY 데이터셋. 공개 가중치는 pudding 씬이라 **우리 12 조합으로 재학습**한다 |
| GS-Verse | [Anastasiya999/GS-Verse](https://github.com/Anastasiya999/GS-Verse) | C#/Unity. 배치 실행 경로가 없어 XPBD + GaMeS 메시-가우시안 결합을 **우리 파이프라인으로 포팅**한다 |
| 우리 학생 | `exe/train_deform.py --pool` | 학습 중인 18 개 중 최고 검증비 |

### GausSim 재학습 (그쪽 정의 그대로)

감독은 **다시점 렌더 손실**이다 (`encode_decode` 의 `gt_label` 이 카메라별
영상이다). 위치를 직접 감독하도록 고치면 그쪽 방법이 아니게 되므로 손대지
않는다. 따라서 우리 PG 궤적을 **다시점 영상으로 렌더**해 그쪽 데이터셋 배치로
넣는다.

1. `mmgs/datasets/multiview_video_dataset.py` 가 기대하는 배치를 확인한다.
2. 조합마다 PG 궤적을 카메라 4 대 x 프레임으로 렌더한다 (`GSScene.render`).
3. `tools/train.py` 를 공개된 `mmgs/apis/train_model` 로 20 줄 작성한다.
4. 조합별 학습 -> 벤치 시나리오 롤아웃 -> 같은 지표.
   탄성 전용이므로 clay/viscoplastic 은 모델 범위 밖임을 표에 명시한다.

### GS-Verse 포팅

Unity/C# 의 XPBD 와 GaMeS 결합을 `lib/anchorflow/gsverse.py` 로 옮긴다.
핵심 파일: `Assets/Scripts/BaseGSVerse.cs`, `SplatDeformate.cs`,
`GSVerseSegmented.cs`, `package/Runtime/GaMeS/GaMeSUtils.cs`.
메시는 3DGS 에서 뽑는다 (PG 의 `particle_filling` 이 이미 쓰는 marching cubes).
소성이 없으므로 clay/viscoplastic 은 범위 밖임을 명시한다.

## 4. 실행 계획 (병렬)

1. **시나리오 고정** — `exe/make_scen.py` 로 96 개 json 생성 (CPU, 즉시).
2. **PG 기준 궤적** — 12 조합 × 8 시나리오 × 240 프레임. GPU 1 장당 1 조합.
3. **i-PG** — 같은 시나리오, dt 4 수준. GPU 1 장당 1 조합-dt.
4. **지표 계산** — 궤적 덤프에서 CPU/GPU 혼합. 조합별 1 프로세스.
5. **렌더** — PG 와 각 대상의 같은 카메라 영상, 그다음 PSNR/SSIM/LPIPS.
6. **표** — `bench/table.md` 와 `bench/table.csv` 로 낸다.

자원: vast.ai 인스턴스를 필요한 만큼 띄워 **GPU 한 장당 한 작업**으로 병렬
실행한다 (지금 12 장 사용 중, 벤치용으로 추가 확보). 클러스터 A6000 은 학습이
점유 중이므로 벤치는 인스턴스에서 돈다.

## 5. 산출물

- `bench/scen_*.json` — 고정 시나리오
- `bench/dump_{대상}_{조합}_{시드}.pt` — 프레임별 위치·F·속도
- `bench/metrics_{대상}.csv` — 조합별 지표
- `bench/table.md` — 최종 표
- `~/workspace/result/anchorflow/bench/` 에 복사하고 `outputs.md` 에 기록
