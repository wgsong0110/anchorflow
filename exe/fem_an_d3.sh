#!/bin/bash
# 절 AN.21 묶음 d3 (잡 안; d2 뒤): HHT ρ 0.8 + 장벽 + 띠 고착 (덩어리 헤시안) 의 lego 21 전체를 상한 2x10 (네 궤적 모두 < 10 %) 과 4x20 (수렴 쪽) 에서 -- 16.7 ms 밖 예산, 정의 자체의 lego 21 판정용.
# 결과 R2 lbours/fem_an/d2/ (d2 와 같은 묶음 디렉토리에 이어서), 표 marks/d3_table.md, 끝 표식 marks/d3.END
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_d2 XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
M=/dev/shm/dkta/lbours/fem_an/marks; MR=r2:storage/result/anchorflow/lbours/fem_an/marks; R=/dev/shm/dkta/lbours/fem_an/d2; mkdir -p $M /dev/shm/dkta/an_lists
echo "[d3 시작] $(cat .done) $(date '+%F %T')" > $M/d3.log; rclone copy $M/d3.log $MR -q
L=/dev/shm/dkta/an_lists/d3.txt; : > $L
for v in h8s1lc2x10 h8s1lc4x20; do A=$(grep -m1 "^$v 0016 " /dev/shm/dkta/an_lists/d2a.txt | cut -d' ' -f3-)
  for t in $(seq -f %04g 0 19); do case $t in 0016|0019) continue;; esac; echo "$v $t $A" >> $L; done; done
echo "목록 $(wc -l < $L) 줄: $(head -1 $L)" >> $M/d3.log
AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=5 bash exe/fem_an_pool.sh d2 $L > $M/d3_pool.log 2>&1
$PY exe/fem_an_tol_table.py $R > $M/d3_table.md 2>&1
echo "[d3 끝] $(date '+%F %T')" | tee -a $M/d3.log > $M/d3.END
rclone copy $M $MR -q --include "d3*"
