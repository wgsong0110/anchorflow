# 표현력 비교 (repflow) — 남은 작업과 진행 상황

2026-10-07 사용자 지시로 정한 범위. 사용자는 더 관여하지 않으니, 이 문서를 기준으로 끝까지
진행하고 체크를 갱신한다. 결과 표는 `writing/anchorflow/` 의 LaTeX 파일로 낸다.

## 공통 규칙 (지시·메모리)

- 가우시안은 **공식 불투명도 필터 뒤 전부** (wolf 13.9 만). 부분표본 금지. 내부 채움 없음 (L2 추적).
- det 는 **각 표현 사상의 실제 야코비안**으로만. 이웃 최소제곱 det 추정 금지 (결과는 archive_knn_det).
- 비교 방법의 표현은 공식 설정. 공식에 없는 부분은 문서에 적는다.
- 속도 측정(FPS)은 단독 GPU, 탄성 하나로, **맨 마지막에**.
- 결과물은 R2 에 올리고 `~/workspace/result/anchorflow/outputs.md` 에 기록.
- 실행 전 커밋, 커밋 해시를 결과에 남긴다. TB: 클러스터 `/home/dkta/work/tbrf` (sshfs → `~/tbmnt/repflow`, 포트 6008).

## A. L2 추적 (wolf · bread · ship), 120 프레임, 속도장 10 프레임마다 교체

목표: 단일 가우시안 속도장 흐름(gauss_flow, Δt < σ²/A)을 각 표현이 입자 L2 로 따라간다.
최적화는 각 방법의 기하(계량 텐서장) 위에서: Δ = -(G + ε λmax I)⁻¹ ∇L, G 는 매 스텝 재계산.

| 방법 | 표현 | 최적화(계량) | 비고 |
|---|---|---|---|
| ours | 사면체 격자 + Gregory·방사형, 매 프레임 재설정 | 셀 det 로그 장벽 (riem logbarrier) | |
| PhysTwin | 질량점 + 공식 interpolate_motions (실시간 데모 대응) | 스프링 탄성 (riem elastic), 강성 k = 1, 10, 100, 1000 | |
| GS-Verse | 표면 메시 (n_grid 100) + 삼각형 국소 틀 결합 | 삼각형 셀 det 로그 장벽 | |
| GausSim | 3 단 계층, 부피 보존 F | 제약 없는 Adam (lr 1e-3, 400/200 반복) | det ≡ 1 |
| Simplicits | kaolin 기본 가중치 (핸들 10) | 입자 det 로그 장벽 | |
| (시간 남으면) 내부점법 | ours · GS-Verse · Simplicits | ipm (μ 1e-4 부터 1/10 씩 4 단계) | |

하지 않는 것 (지시): tet, ours 탄성(입자 F) 계량, 제약 없는 Simplicits.

측정: RMSE · CD · EMD(편향 제거 Sinkhorn, 전체 점) · det 최소/뒤집힘 비율(야코비안) ·
물리 지표(에너지·질량·운동량 보존, 물리 잔차) · 영상(3DGS 공식 래스터라이저) 시각 품질(목표 렌더 대비 PSNR/SSIM/LPIPS 등).

- [x] 해석 야코비안 + torch.compile, 작은 배치 행렬곱 원소별화 (검증 1e-7)
- [ ] 프로파일링 (prof3) → 느린 구간 최적화
- [ ] ε·η 고르기 (wolf 3 프레임): PhysTwin elastic ✓(ε1e-3, η10 최선), ours logbarrier, GS-Verse logbarrier, Simplicits logbarrier
- [ ] wolf 본 실행 (TB·영상)
- [ ] bread · ship: 흐름(gauss_flow --n 0), Simplicits 가중치 학습, 본 실행
- [ ] EMD 전체 재측정 (rep_emd)
- [ ] 물리 지표 · 시각 품질 측정
- [ ] (시간 남으면) 내부점법

## B. 증분 포텐셜 (lego · ficus · mic) — L2 대신 시뮬레이터의 증분 포텐셜을 매 스텝 최소화

각 표현을 매개변수 공간으로 두고 암시적 시간적분. 기준 궤적은 그 시뮬레이터가 직접 낸 것.

| 시뮬레이터 | 장면 (지시) | 파라미터 |
|---|---|---|
| i-PhysGaussian | 점소성 | 공식 레포/논문 값 |
| GaussianFluent | 바닥으로 던지기 | 공식 레포/논문 값 (숨은 기본값 포함) |
| Fracture-GS | 두 물체 빠른 충돌 | 공식 레포/논문 값. 공개 코드가 없다는 게 확실하면 직접 구현 |

- 내부 채움 필수 (PG/GF 공식 채우기만, 직접 만든 채우기 금지).

### 공식 자료 조사 (2026-10-07)

- **Fracture-GS** (Wang·Wu·Song·Xu, ICLR 2026, openreview zcAwK50ft0): 공개 코드 **없음** 확인 —
  GitHub 검색(Fracture-GS/FractureGS/Collision-MPM), 1 저자 GitHub(wangxiaogang866) 저장소 목록,
  ICLR 포스터 페이지, mlanthology, 논문 노트("Code: To be confirmed") 어디에도 없다 → 직접 구현.
  구성: Collision-MPM (물체별 독립 P2G, 정규화 질량 분포로 만든 운동량 보존 경계력) + NACC 구성식.
  논문 표의 물성 (E[MPa], ν, 밀도, NACC α β ξ M):
  Bowl 5e4/0.46/2/(0.98,0.5,1,2.36), Ficus leaf 8e4/0.39/0.6/(0.94,2,3,2.36),
  Ficus branch 1e6/0.39/5/(0.94,2,3,2.36), Ficus pot 2e4/0.39/2/(0.98,0.5,2,2.36),
  Teapot 5e5/0.46/5/(0.98,0.5,1,2.36), Table top 1.5e4/0.39/1/(0.99,0.5,1,2.36),
  Table leg 1e8/0.39/1000/(0.94,2,3,2.36). 충돌 속도·격자·dt 는 본문 표에 없음 → 정한 값을 적는다.
- **i-PhysGaussian** (arXiv 2602.17117): 공식 코드 미공개 (클러스터 i-physgaussian 은 비공식 재현,
  RobotiX101). 점소성 = `plasticine` (rate-dependent J2, Table 6). 장면별 수치는 "릴리스 설정 파일"
  참조라고만 하고 미공개 → 점소성 수치는 비공식 재현 코드의 구성식 + 정한 값을 적는다.
- **GaussianFluent**: 공식 설정 watermelon(CD-MPM, E 2e3, ν 0.38, 밀도 1, 마찰 45, β 1, ξ 3,
  경화 1, flip_pic 0.7, n_grid 300, g -15, 바닥) + 숨은 기본값 (초기속도 -6 등, patch_gf_match.py).
  lego·ficus·mic 전용 설정은 없음 → watermelon 설정으로 바닥 던지기.
- 측정: A 와 같음 + 프레임별 증분 포텐셜 값.

- [ ] Fracture-GS 공식 레포 찾기 / 받기
- [ ] 세 시뮬레이터의 공식 파라미터 확정 (문서화)
- [ ] 형상별 채움 (ficus 없음 → 공식 채우기로 생성)
- [ ] 기준 궤적 생성 (9 = 3 형상 × 3 시뮬레이터)
- [ ] 증분 포텐셜 최소화 추적기 (rep_track2 에 --obj ip)
- [ ] 본 실행 · 측정

## C. 결과 문서

- [ ] `writing/anchorflow/` 에 LaTeX 파일, 모든 결과 표
- [ ] (맨 마지막) FPS: 단독 GPU, 탄성

## 진행 기록

- 2026-10-07 21:10 문서 작성. 프로파일링·ε·η 고르기 진행 중.
