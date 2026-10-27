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

## 결과
(아래에 기록)
