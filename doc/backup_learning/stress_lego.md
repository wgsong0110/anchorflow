# 셀 응력 모델 (lego, 사용자 10-27 확정 설계)

코드: `lib/anchorflow/stress_net.py` (모델·적분점·보간·생성기·통계·mom_core 어댑터), `exe/train_stress.py` (stats · check · train · eval · bench),
`exe/stress_lego.sh` (클러스터 드라이버: test · run). 출력 `lbours/stress_lego/` (R2 `r2:storage/result/anchorflow/lbours/stress_lego/`).

## 설계
- **셀** = 세계 고정 Kuhn 사면체 격자 (node_h 0.028284, hz = h/√6) 의 사면체 하나. 격자·셀 번호·꼭짓점 사슬 순서·∇λ 는 `mom_core.frame` 과 같다
  (셀 번호 = unique(tet_id) 순서 = `S.pinv`, 꼭짓점 = `S.tets`, ∇λ = `S.gl`; check 모드에서 대조).
- **입력** (셀 안 입자 집합, 순서 불변, 개수 가변): 셀 국소 좌표 (barycentric λ 4 + 격자 셀 좌표 f − ½ 3), F_p, 물성 (log μ_p/μ_ref, ν_p), log(V_p/V_tet).
- **회전 공변**: 셀 대표 회전 R = polar(F̄_T), F̄_T = Σ V_p F_p / V_T (det R = +1). 입력 F'_p = Rᵀ F_p, 출력 τ_q = R τ'_q Rᵀ.
  F_p → Q F_p 이면 R → Q R (극분해 유일성), F' 불변 → τ → Q τ Qᵀ (정확한 공변, 위치는 격자 고정이라 그대로).
- **집합 인코더**: 입자 MLP (3 층, 폭 128) ⇄ 셀 문맥 MLP 2 왕복 (풀링 = 부피 가중 평균 + V_tet 단위 부피 가중 합, 셀 특징 19: 사면체 방향 one-hot 6,
  F̄' − I 9, log V_T/V_tet, log(1+n)/log 64, 셀 물성 2). 적분점마다 주의 풀링 (입자 로짓 → 부피 가중 softmax) + 적분점 임베딩 → 출력 MLP → 대칭 6 성분 Δ_q.
  **τ'_q = τ_FCR(F̄'; μ_c, λ_c) + 0.1 μ_c S(Δ_q)** (출력 층 0 초기화 → 시작 = 평균 F 기준).
- **노드 힘**: f_n = −Σ_T Σ_q ω_q V_T τ_q ∇w_n(x_q), V_T = Σ_{p∈T} V_p (mom_core 의 VT), 적분점 위치·가중치 고정:
  K=1 셀 중심 (ω=1), K=4 2 차 정확 규칙 (λ = (0.585, 0.138, 0.138, 0.138) 순환, ω = ¼).
- **보간** `--interp`: `bary` (사면체 4 꼭짓점, ∇λ 셀 안 상수 → K=1 로도 정확할 수 있다), `bspline2` (MPM 2 차 B-스플라인 27 노드, simplex.bspline27 과 같은 식; ∇w 가 셀 안에서 변해 K 개 적분점 필요).
- **정답**: f_n^true = −Σ_p V_p τ_FCR(F_p) ∇w_n(x_p), τ_FCR = `deform_model.kirchhoff_tau` (ψ 가 (μ, λ) 에 선형이라 두 번 불러 입자별 물성으로 합성; check 에서 씬 cfg 직접 호출과 대조).
- **mom_core 에 끼우기** (bary): `MomStressAdapter(net).tau_T(S, x, F, Vp, mu, lam)` → [T,3,3] 유효 응력 Σ_q ω_q τ_q → `mom_core.node_force(S, VT, ·)` 그대로.
  bspline2 모델은 `quad_forces` + `global_assemble` (노드 = 격자 정수좌표).

## 손실·지표 (고정 상수 스케일)
- 손실 L = 평균_셀 Σ_n |f^모델 − f^정답|² / f_ref,c², f_ref,c = μ_c V_tet / node_h (μ_c = 셀 입력 물성, V_tet·node_h 고정 상수 — 응답·변위로 나누지 않는다).
- 지표: 집합 단위 상대 오차 (분모 = 시험 집합 전체 정답 힘의 합; 표본별 나눗셈 없음). 무작위 시험 셀은 μ_c 로 잰 무차원 힘, lego 상태는 셀 기여 (국소 힘) 와 전역 노드 힘 둘 다.
- 기준: 셀 평균 F 하나로 낸 FCR 응력 (적분점 1 개 셀 중심), bspline2 는 같은 응력을 K 적분점에 둔 것도.

## 자기지도 셀 생성 (`stress_net.CellGen`, 실제 궤적 무관)
위치 4 모드 (균등 / 한 점으로 수축 / 부분 사면체 / 정규 부분격자), 입자 수 logU·균등 [1, n_max], V_p logU, F_p = R_out R_c U_c (I + G·Δy)(I + σN)(I + ε_out),
물성 (씬 값 40 %, 나머지 E logU[E/100, 100E], ν U[0, 0.45], 10 % 셀은 두 물성 섞기). 실제 분포와의 비교는 `test_<커밋>/stats/cover.txt`.

## 시험 단계 (test_6818a52, ece-a6gpu10 GPU3)
검산 (`test_6818a52/check/check.txt`):
- FCR 분해 (입자별 μ·λ) vs kirchhoff_tau (씬 cfg): 상대차 2.5e-8. 보간 항등식 (float64): bary·bspline2 모두 |Σw−1|, |Σ∇w|h, |Σx⊗∇w − I| ≤ 7e-16.
- 보존 (무작위 셀 2048, float64, 출력 층 무작위): 셀별 |Σf|/Σ|f| 최대 ≤ 2e-16 (모델), 각운동량 |Σ(x−x̄)×f|/Σ|x−x̄||f| 최대 ≤ 6e-16 (모델; bary·bspline2, K=1·4).
  대조: τ 에 반대칭 성분을 더하면 0.83~0.89 -- 각운동량 보존은 τ 대칭에서 온다.
- 회전 공변: max_셀 ‖τ_q(QF) − Qτ_q(F)Qᵀ‖/‖τ_q‖ = 5e-5~1e-4 (float64). 출처: 극분해 뉴턴으로 낸 F'_p 차 ≤ 2e-6 (거의 특이한 셀), 이것이 기저 τ_FCR(F̄') 의
  내부 float32 특이값 계산 (phys_resid._sig, _SIG_DT = float32) 에서 커진다. 같은 kirchhoff_tau 자체의 입자 공변 오차가 최대 1e-2 (전체 1e-6) 이라 남은 차는 FCR 구현 정밀도.
- mom_core 규약: dump40 사면체 43,340 개 셀 번호·꼭짓점 사슬 순서 같음, ∇λ 상대차 6.8e-6 (float32), `mom_core.node_force(VT, 입자 τ 부피평균)` vs 입자 적분 정답 5.1e-6,
  어댑터 `MomStressAdapter.tau_T` → `mom_core.node_force` vs 이쪽 적분점 조립 8e-6.
- 이어 돌리기 (bspline2 K4, 20 → 30 vs 한 번에 30): 망 최대차 1.8e-7, 같은 설정 두 번 돈 잡음 바닥 2.6e-7, 생성기·CPU 난수 같음, 시험 평가값 같음.

분포 (`test_6818a52/stats/cover.txt`, 실제 = dump40 + fem2_lego 롤아웃 프레임 10~80 9 상태, 셀 38.8 만, 입자 180 만 표본; 생성 16384 셀):
- 실제: 셀 입자 수 중앙 2, 90 % 18, 99.9 % 163, 최대 584 / 생성 최대 640. 셀 점유 V_T/V_tet 실제 ≤ 22 / 생성 ≤ 30. |F'_p − F̄'| 실제 99.9 % 0.99 최대 2.65 / 생성 2.78, 4.36.
  입자 회전각 (셀 회전 뺀 뒤) 실제 최대 159° / 생성 180°. det F_p 실제 [−1.00, 2.56] / 생성 [−4.4, 9.3]. F̄' log 특이값, det F̄, 셀 중심·퍼짐 반경 모두 덮음.
- 덮지 못한 두 칸: (1) log10 V_p 상위 0.1 % 분위 (실제 −5.10 = 최대값에 0.1 % 이상 몰림 / 생성 −5.30; 생성 최대 −4.40 은 실제 최대를 넘는다),
  (2) |F'_p − F̄'| 하위 0.1 % (실제 정확히 0 / 생성 6e-8 -- F 가 같은 셀의 부동소수 잡음).

## 본 학습 (6818a52, 150000 반복, 셀 1024/반복, Adam 5e-4 코사인)
- `run_bary_K1` (GPU3), `run_bspline2_K4` (GPU4). 표식 `STRESS_bary_K1_DONE` / `STRESS_bspline2_K4_DONE` (실패 `_FAIL`).
- 첫 평가 (bary K1 반복 2000): 무작위 시험 8.2 % (기준 20.7 %), lego 9 상태 평균 셀 기여 99.8 % (기준 134.5 %), 전역 노드 144.2 % (기준 202.9 %).
- 첫 평가 (bspline2 K4 반복 2000~4000): 무작위 시험 45.7 % (기준 1 점 59.7 %, 기준 4 점 48.9 %), lego 셀 75.2 % (기준 145.9 %, 기준 4 점 111.0 %), 전역 83.4 % (기준 146.4 %).
- 기준 (평균 F 하나) 이 lego 실제 상태에서 100 % 를 넘는다: 셀 안 입자들의 회전이 섞여 τ(F̄) 가 입자 응력 평균과 방향부터 다르다 (doc/fem_realtime.md 의 에너지 24.5 % 차와 같은 현상).

## 추론 시간 (lego dump40, 사면체 43,340 · 입자 313,750, A6000 단독 카드, 동기화 중앙값)
| | bary K1 | bspline2 K4 |
|---|---|---|
| 모델 응력 (회전 정규화 + 망 + 기저 + 되돌림) | 46.3 ms | 52.7 ms |
| 모델 응력 + 노드 힘 조립 | 48.3 ms | 66.2 ms |
| 기준 (평균 F 적분점 1) + 조립 | 12.7 ms | 13.1 ms |
| 정답 (입자 FCR 적분) + 조립 | 43.5 ms | 59.2 ms |
구간: 셀 평균 0.16 ms, 극분해 (배율 뉴턴) 5.6 ms, 기저 τ_FCR(F̄') 8.5 ms, 나머지 ~32 ms 가 집합 인코더 (입자 31 만 개 MLP). 격자·셀 묶기 6.5 ms 별도.
지금 크기 (폭 128) 에서는 모델이 정확한 입자 적분보다 빠르지 않다.
