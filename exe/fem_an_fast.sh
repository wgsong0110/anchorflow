#!/bin/bash
# 절 AN 가속 판 시험 (비측정): 같은 반복 예산 표 두 번 -- (1) 구간별 (--sec --gfin, 경계 동기화), (2) 동기화 없는 프레임 시간만.
cd "$(dirname "$0")/.." || exit 1
bash exe/fem_an_budget.sh fast1 --fast
AN_SECF=" " bash exe/fem_an_budget.sh fast1ns --fast
