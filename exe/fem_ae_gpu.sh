#!/bin/bash
# 절 AE 일꾼: 작업 목록에서 (줄 번호 % K == k) 인 실행만 카드 G 에서 차례로 -- 비측정 (fem_ab_run.sh, NOT_TIMED; 출력 노드 로컬 /tmp + R2).
# 카드 1~6 (GPU0 은 단독 측정 몫). ficus 는 끝나면 주기 계산 (period.json) -> R2. 사용: bash exe/fem_ae_gpu.sh <G> <k> <K> <작업 목록 .py>
G=${1:?GPU}; k=${2:?k}; K=${3:?K}; JL=${4:?목록}
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
case $G in 1|2|3|4|5|6) ;; *) echo "[오류] 비측정은 카드 1~6 만"; exit 1;; esac
echo "[일꾼 카드 $G ($k/$K)] 커밋 $(git log -1 --oneline) 시작 $(date +%T)"
i=0
while read -r ROOT TG ARGS; do
  if [ $((i % K)) -eq "$k" ]; then
    O=${AB_OUT:-/tmp/dkta/lbours}/$ROOT/$TG
    EX_ROOT=$ROOT bash exe/fem_ab_run.sh "$TG" "$G" $ARGS < /dev/null > /dev/null 2>&1
    if [ -f $O/pred.pt ]; then
      [ -f $O/period.json ] || python exe/fem_p_period.py --data $FD --pred $O/pred.pt > $O/period.txt 2>&1
      rclone copy $O r2:storage/result/anchorflow/lbours/$ROOT/$TG -q
    fi
    echo "$(date +%T) $TG $( [ -f $O/EX_DONE ] && echo 끝 || echo 실패) $(grep -a '롤아웃\]' $O/v_fem.log 2>/dev/null | cut -c1-170)"
  fi
  i=$((i + 1))
done < <(python $JL)
echo "[일꾼 카드 $G ($k/$K)] 끝 $(date +%T)"
