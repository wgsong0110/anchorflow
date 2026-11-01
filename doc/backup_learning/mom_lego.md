# 운동량 잔차 변위 망 (lego L000, 사용자 10-27 확정 설계)

코드: `lib/anchorflow/mom_core.py` (시뮬레이터·손실), `lib/anchorflow/mom_data.py` (자기지도 상태 생성·분포 통계),
`exe/train_mom.py` (cover · train · roll), `exe/mom_lego.sh` (클러스터 드라이버: test · run). 출력 `lbours/mom_lego/` (R2 `r2:storage/result/anchorflow/lbours/mom_lego/`).

## 시뮬레이터 (MPM 식, 격자 = 세계 고정 Kuhn 사면체 격자, node_h 0.028284 -- 기존 lego 실행과 같은 격자)
한 스텝 (길이 h):
1. **P2G**: 입자 질량·운동량 -> 노드 (barycentric λ): m_n = Σλm_p, v_n = Σλm_p v_p / m_n. 사면체마다 F̄_T = Σ_{p∈T} V_p F_p / V_T, V_T = Σ_{p∈T} V_p (V_p = m_p/ρ, 초기 부피).
2. **응력 모델**: τ_T = P(F̄_T) F̄_Tᵀ, 고정 FCR (`phys_resid.psi_of` 를 미분한 `deform_model.kirchhoff_tau` 그대로). 자리 = `mom_core.FcrStress` 하나 (F̄ [T,3,3] -> τ [T,3,3]) -- 학습 응력 모델로 갈아 끼울 곳.
   노드 내부력 f_n = −Σ_{T∋n} V_T τ_T ∇λ_{n,T} (= −∂/∂Δx Σ_T V_T ψ(A_T F̄_T) 의 Δx = 0 값; `check_force` 자동미분 대조).
3. **변위 망 (학습 대상)**: Graph U-Net (gunet5_near 와 같은 몸통: 3 수준 h·2h·4h, 층 2+2+2+1+1, 폭 128, 거친 수준 바닥 near, dt FiLM, dt 비례 출력 크기 0.02·EXT·h/(1/60), 0 초기화 출력).
   **관성 예측 더하기 없음 -- 망 출력이 Δx_n 그 자체.**
   노드 입력 11 (고정 상수 변환만, 표본 자체 양으로 나누지 않음; 그 뒤 생성 상태 64 개로 잰 평균·표준편차로 표준화해 고정):
   symlog(f_n/m_n ; |g|) 3 (= sign·log(1 + |a|/|g|)), v_n [m/s] 3, log(m_n / m_full) 1 (m_full = ρ·격자점당 부피), 점유 = 노드에 걸린 점유 사면체 수 / 24 (Kuhn 격자 내부 노드 = 24) 1,
   log10(h / (1/60)) 1, 바닥 근접 [플래그, 거리/h_node] 2 (`--bc_feat near` 와 같은 `deform_model.bc_near_feats`).
4. **G2P**: x_p += Σλ Δx_n, v_p = Σλ Δx_n/h, F_p ← A_T F_p, A_T = I + Σ_i Δx_i ⊗ ∇λ_i.

## 손실 (탄성 에너지를 다시 계산하지 않는다 -- f 는 스텝 시작 상태에서 정해진 상수)
- 운동량 잔차 r_n = m_n(Δx_n − h v_n)/h² − f_n − m_n g. 노드 질량은 max(m_n, m_floor), m_floor = 1e-3·m_full (고정 상수) -- 입자가 꼭짓점·면 위에 있어 λ = 0 이면 m_n = 0 인 노드가 생겨 f/m 이 비유한이 된다 (첫 cover 실측). 입력 f/m·명시해·잔차 무게 모두 같은 바닥.
  **E_res = Σ_n h²|r_n|²/(2 m_n)** = Σ_n m_n/(2h²)|Δx_n − x̃_n|², x̃_n = h v_n + h²(f_n/m_n + g) (명시적 오일러 해).
  스케일: 잔차 제곱을 **노드 질량으로 나눈** M⁻¹ 노름에 h²/2 를 곱한 것 -- 증분 포텐셜 관성항과 같은 꼴·단위 [J] 라 아래 두 장벽과 단위가 맞고,
  망의 속도 출력 Δx/h 로 보면 ½Σ m|δv|² 라 dt 에 따라 무게가 바뀌지 않는다. h 는 표본의 조건 변수(입력)이지 표본 변위가 아니다.
- 셀 뒤집힘 금지 (det A_T > 0): **E_inv = Σ_T V_T κ_inv μ b(det A_T; Ĵ = 0.3, 선형 연장)** -- 기존 `--inv_barrier 1.0 --inv_jhat 0.3 --inv_ext linear` 의 꼴 (κμ b(det F)) 을 이번 스텝 셀 사상 det A_T 에, κ_inv = 1.
- 바닥 비관통: **E_bc = Σ_p κ_bc m_p/h² b(sd_p; d̂, 선형 연장)**, sd_p = 스텝 끝 입자 위치의 바닥 부호 거리 -- 기존 `--bc_mode barrier` (`phys_resid.bc_energy` 장벽 분기) 와 같은 꼴, κ_bc = 1, d̂ = 격자 dx/2 (기존 기본값).
  (기존 sticky 접선 항은 지시에 없어 넣지 않았다 -- 비관통만.)
- **손실 = (E_res + E_inv + E_bc) / E_ref, E_ref = M_tot |g| EXT** (lego 고정 상수). b(·) = −(d − d̂)² ln(d/d̂) (d < d̂), d ≤ 10⁻³ d̂ 는 선형 연장 (`phys_resid._barrier_b`).
- 반복마다 생성 상태 4 개 (`--batch 4`) 손실 평균으로 Adam (lr 3e-4, 기울기 노름 1 에서 자르기).

## 자기지도 상태 생성 (정답 궤적 무관, `mom_data.StateGen`)
lego 기준 배치 (L000 0 프레임 입자, 질량·부피 그대로) 위에서: 15 % 무변형 (회전만, 힘 0 근처), 회전 (30 % SO(3) 균등, 나머지 축 정렬 자세 24 개 중 하나 + 연직축 균등 + 기울기 logU[1e-4, 0.6] rad -- 바닥에 평평히), 매끈한 무작위 변형장 (푸리에 6 모드, 파장 logU[3 h_node, 3 EXT],
변형률 logU[1e-6, 0.2]) + 무작위 아핀 늘임 (logU[1e-6, 0.15]) + 25 % 바닥 법선 방향 눌림 비 U[0.4, 1] (정답 롤아웃은 바닥에서 높이가 1.056 -> 0.52 로 납작해진다), 입자별 F 퍼짐 (logU[1e-6, 0.05]) + 일부 입자 (≤ 3 %) 추가 회전 (각 logU[1e-3, 1.5]),
바닥 최소 거리 (15 % 약간 관통 [−0.5, 0] h_node, 55 % 근접 [0, 2] h_node, 나머지 [2 h_node, 0.5 EXT]), 속도 = 병진 logU[1e-3, 6] m/s + 회전 logU[1e-3, 10] rad/s
+ 매끈한 장 logU[1e-4, 2] m/s + 입자 잡음, dt = 25 % 1/60, 나머지 logU[1/60/64, 1/60]. 분포 비교는 `--mode cover` (`lbours/mom_lego/test_<커밋>/cover/cover.txt`).

## 평가
최저 오차 체크포인트로 (1)~(4) 롤아웃을 lego L000 80 프레임 (h = 1/60, 서브스텝 없음): 롤아웃 평균 |x − x_gt|/EXT (기존과 같은 지표),
뒤집힌 사면체 수 (det A_T ≤ 0), 뒤집힌 입자 수 (det F_p ≤ 0), 프레임당 시간 (동기화, 한 카드에 이것 하나). 학습 중 1000 회마다 같은 80 프레임 롤아웃 (검증 지평 = 평가 지평).
발산 (비유한 또는 시작 대비 이동 > 20 EXT) 하면 그 앞 상태에 멈춰 두고 남은 프레임도 같은 지표로 재서 80 프레임 평균을 낸다.
비교: PCG FEM 11.30 % · 0.731 s/프레임, Graph U-Net 풀 학습 (pool1_gunet) 22.85 %.
