# FEM pcg 판 실시간화 (lego L000, 2026-10-27~)

사용자 지시 (10-27): "lego FEM 프로파일링부터 하고 속도 가속할 수 있는 거 다 가속하고(커스텀 커널 작성 포함) 돌려서 실행 성능 측정".
기준 = 커밋 15250d4 수렴판 (`exe/fem_roll.sh lego … full`, FEM_SOLVER=pcg, FEM_PCG_MAX=10000, FEM_EIG_FLOOR=1e-6, FEM_LM=1,
FEM_LM_MUMIN=0, FEM_EIG_PROJ=abs, tol 1e-2), 결과 `lbours/fem2_lego/run`: **L000 80 프레임 오차 11.304 %**.
한 출력 프레임 = 증분 포텐셜 풀이 한 번 (h = 1/60, fem.jsonl 프레임당 기록 하나 -- 서브스텝 없음).
오차 = 롤아웃 평균 위치 오차 / EXT (고정 상수, 기존과 같은 지표).

## 새 선택지 (모두 환경변수, 안 주면 기존 계산 그대로)

| 변수 (fem_roll.sh) | train_deform | 내용 |
|---|---|---|
| FEM_PROF=1 | AF_FEM_PROF | 구간마다 CUDA 동기화해 정확한 시간: t_fwd0(파이프라인 순전파), t_eg(E+기울기), t_hess(헤시안), t_proj(사면체 합+eigh 투영), t_asm(투영+CSR), t_pcg, t_ls(선탐색, n_ls 회 E 평가) |
| FEM_DUMP=폴더 FEM_DUMP_T=프레임들 | AF_FEM_DUMP | 축약 검증 덤프 (입자 12x12 HL, Jd, ∂Jd/∂U, x·v·F·m·V, 노드 위치) |
| FEM_FIX_NEWTON=N | AF_FEM_FIX_NEWTON | 뉴턴 걸음 최대 N (수렴 판정·에너지·선탐색·LM 그대로, 일찍 수렴하면 먼저 멈춤). 시작점은 기존 `--ov_init predict` (이전 프레임 해의 속도로 낸 관성 예측 = warm start) |
| FEM_PCG_MAX=k | AF_FEM_PCG_MAX (기존) | PCG 반복 상한 |
| FEM_FAST=particle\|tet | AF_FEM_FAST | E·기울기·헤시안·선탐색 E 를 `lib/anchorflow/fem_fast.py` 로 (같은 식을 입자 단위 작은 함수로; 망·격자 탐색 없음). 둘째 뉴턴 걸음부터 파이프라인 순전파 생략. tet = 사면체 정확 축약 |
| FEM_FAST_CHECK=N | AF_FEM_FAST_CHECK | 처음 N 프레임 기존 파이프라인과 E·기울기·사면체 헤시안 대조 ([FEM 빠른 검산]) |
| FEM_PCG_CHECK=k | AF_FEM_PCG_CHECK | PCG 스칼라를 GPU 에 두고 k 회마다만 수렴 확인 (반복마다 동기화 2 회 제거) |
| FEM_CSR_CACHE=1 | AF_FEM_CSR_CACHE | 프레임 안 CSR 패턴(정렬·중복합 색인) 재사용, 값만 채움 |

## (0) 기준판 단계별 시간

### 기존 80 프레임 실행 로그에서 (15250d4, GPU1 단독 카드, 동기화 없는 시간 -- 비동기라 구간 경계가 약간 섞임)
프레임당 평균 120.2 s, 프레임당 뉴턴 걸음 평균 44.7 (중앙 42), 걸음당:

| 구간 | 걸음당 | 비고 |
|---|---|---|
| 헤시안 (이중 역전파 12 회, 314k 입자 12x12) | 1.502 s | |
| PCG (블록 야코비, CSR float64) | 0.848 s | 반복 평균 2873 (중앙 2891), 반복당 ~0.30 ms |
| 조립 (사면체 합 + eigh 투영 + CSR) | 0.094 s | 사면체 42,373, nnz 1.70 M |
| 순전파 + E + 기울기 | 0.085 s | |
| 선탐색·기타 (프레임 루프 쪽) | 0.160 s | |
| **걸음 합** | **2.69 s** | × 44.7 걸음 = 120 s / 프레임 |

(동기화 계측판 결과는 아래 "측정 기록" 에 추가)

## 측정 기록 (누적)
