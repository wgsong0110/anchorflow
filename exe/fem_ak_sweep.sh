#!/bin/bash
# 절 AK 스윕 (비측정): 작업 목록 (exe/fem_ak_jobs.py <단계>) 을 카드 AK_CARDS (기본 1~6) x 일꾼 AK_W (기본 2) 로 동시에 (fem_ab_run.sh, NOT_TIMED; 출력 노드 /tmp + R2).
# 사용: bash exe/fem_ak_sweep.sh <단계> [씬,...]        (일꾼: bash exe/fem_ak_sweep.sh --w <카드> <k> <K> <단계> [씬,...])
cd "$(dirname "$0")/.." || exit 1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
if [ "$1" = "--w" ]; then
  G=$2; k=$3; K=$4; ST=$5; SCS=$6
  case $G in 1|2|3|4|5|6) ;; *) echo "[오류] 비측정은 카드 1~6 만"; exit 1;; esac
  echo "[일꾼 카드 $G ($k/$K)] 커밋 $(git log -1 --oneline) 시작 $(date +%T)"
  i=0
  while read -r ROOT TG ARGS; do
    if [ $((i % K)) -eq "$k" ]; then
      O=/tmp/dkta/lbours/$ROOT/$TG
      EX_ROOT=$ROOT bash exe/fem_ab_run.sh "$TG" "$G" $ARGS < /dev/null > /dev/null 2>&1
      echo "$(date +%T) $TG $( [ -f $O/EX_DONE ] && echo 끝 || echo 실패) $(grep -a '롤아웃\]\|\[발산\]' $O/v_fem.log 2>/dev/null | tail -2 | cut -c1-170 | tr '\n' ' ')"
    fi
    i=$((i + 1))
  done < <(python exe/fem_ak_jobs.py "$ST" $SCS)
  echo "[일꾼 카드 $G ($k/$K)] 끝 $(date +%T)"
  exit 0
fi
ST=${1:?단계}; SCS=${2:-}
CARDS=${AK_CARDS:-1 2 3 4 5 6}; WK=${AK_W:-2}
set -- $CARDS; NC=$#; K=$((NC * WK))
OD=/tmp/dkta/lbours/fem_ak_logs; mkdir -p $OD
LB=$(echo "$ST" | tr ':,@' '___' | cut -c1-60)
echo "[스윕 $ST] 커밋 $(git log -1 --oneline) 카드 $CARDS x $WK, 작업 $(python exe/fem_ak_jobs.py "$ST" $SCS | wc -l) $(date '+%F %T')"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
j=0
for G in $CARDS; do for w in $(seq 1 $WK); do
  bash exe/fem_ak_sweep.sh --w $G $j $K "$ST" "$SCS" > $OD/w_${LB}_${G}_$w.log 2>&1 &
  j=$((j + 1)); done; done
wait
cat $OD/w_${LB}_*.log | grep -v "^\[일꾼" | sort -k2
echo "[스윕 끝 $ST] $(date +%T)" | tee $OD/DONE_${LB}
rclone copy $OD r2:storage/result/anchorflow/lbours/fem_ak_logs -q
