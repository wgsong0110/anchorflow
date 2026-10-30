#!/bin/bash
# 절 AI 스윕 드라이버: 카드 AI_CARDS (기본 1~6) x 일꾼 AI_W (기본 2) 동시. 사용: bash exe/fem_ai_sweep.sh <단계> [씬,...]
cd "$(dirname "$0")/.." || exit 1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
ST=${1:?단계}; SCS=${2:-}
CARDS=${AI_CARDS:-1 2 3 4 5 6}; WK=${AI_W:-2}
set -- $CARDS; NC=$#; K=$((NC * WK))
OD=/tmp/dkta/lbours/fem_ai_logs; mkdir -p $OD
LB=$(echo "$ST" | tr ':,' '__')
echo "[스윕 $ST] 커밋 $(git log -1 --oneline) 카드 $CARDS x $WK, 작업 $(python exe/fem_ai_jobs.py $ST $SCS | wc -l) $(date '+%F %T')"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
j=0
for G in $CARDS; do for w in $(seq 1 $WK); do
  bash exe/fem_ai_gpu.sh $G $j $K "$ST" "$SCS" > $OD/w_${LB}_${G}_$w.log 2>&1 &
  j=$((j + 1)); done; done
wait
cat $OD/w_${LB}_*.log | grep -v "^\[일꾼" | sort -k2
rclone copy $OD r2:storage/result/anchorflow/lbours/fem_ai_logs -q
echo "[스윕 끝 $ST] $(date +%T)"
