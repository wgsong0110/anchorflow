#!/bin/bash
# 절 AN.err2 본 묶음 이어 돌기 (조정자 11-02: 카드 3·5 만 -- 0 측정, 1·2·4·6 은 L2 추적): 앞 풀의 남은 판이 끝나기를 기다린 뒤 같은 목록을 카드 3·5 에서 (끝난 판은 건너뜀)
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "fem_anerr/r2/" > /dev/null; do sleep 30; done
echo "[앞 판 끝] $(date +%T)"
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh r2 /dev/shm/dkta/anerr_r2.txt
