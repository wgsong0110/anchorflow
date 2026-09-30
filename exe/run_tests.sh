#!/bin/bash
# 전체 검사를 한 번에. 앞으로 구조를 건드리면 이걸 돌린다.
#
# 2026-09-30 교훈: 나온 버그가 전부 "경로가 실행되지 않거나 전역이 오염돼 조용히
# 틀리는" 부류였다. 눈으로 읽어서는 계속 놓치므로 짧게라도 실제로 태운다.
#
# 사용: bash exe/run_tests.sh [GPU]
set -u
G=${1:-0}
R=$(cd "$(dirname "$0")/.." && pwd)
W=${AF_WORK:-/home/dkta/work}
export PATH=/home/dkta/.conda/envs/af/bin:$PATH
export AF_WORK=$W PYTHONPATH=$R/lib PYTHONIOENCODING=utf-8
export MPLCONFIGDIR=/home/dkta/.mplcache CUDA_VISIBLE_DEVICES=$G
FAIL=0
one () {
  local name=$1; shift
  printf "== %-22s " "$name"
  local out
  out=$(timeout 1200 python -u "$@" < /dev/null 2>&1)
  local last
  last=$(echo "$out" | grep -aoE "[0-9]+/[0-9]+  (ALL-OK|SOME-FAIL)" | tail -1)
  if echo "$last" | grep -q ALL-OK; then
    echo "$last"
  else
    echo "실패  ${last:-출력없음}"
    echo "$out" | tail -12 | sed 's/^/     /'
    FAIL=$((FAIL+1))
  fi
}
one 격자·복합체        $R/exe/test_simplex.py
one 변형장미분·야코비안  $R/exe/test_field_deriv.py
one 하드구속·물성      $R/exe/test_hard_bc.py
one 소성최적화동치     $R/exe/test_plastic_equiv.py
one 손실항반응        $R/exe/test_loss_terms.py --gpu $G
one 종단             $R/exe/test_e2e.py --gpu $G
echo
if [ $FAIL -eq 0 ]; then echo "전체 통과"; else echo "실패 $FAIL 개"; exit 1; fi
