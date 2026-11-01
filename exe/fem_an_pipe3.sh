#!/bin/bash
# 절 AN 파이프 3: 고착 상태 걸음당 한 번 판으로 진단 다시 (E_ref·A_full·C_nofloor) -> 가속 판 반복 예산 (구간별, 동기화 없음).
cd "$(dirname "$0")/.." || exit 1
AN_DIAG=diag2 AN_V1="E_ref C_nofloor" AN_V2="A_full" bash exe/fem_an_diag.sh
bash exe/fem_an_budget.sh fast3 --fast
AN_SECF=" " bash exe/fem_an_budget.sh fast3ns --fast
