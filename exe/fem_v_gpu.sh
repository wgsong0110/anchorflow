#!/bin/bash
# 절 V 한 카드 담당: 작업 목록 (python <목록> [β]) 에서 (줄 번호 % 7 == G) 인 실행만 이 카드에서 차례로 -- 비측정 (EX_NOWAIT=1, NOT_TIMED 표식).
# 실행마다 그 카드가 비어 있는지 (500 MiB 아래) 확인하고 기다린다 (같은 잡의 다른 작업과 겹치지 않게).
# 사용: bash exe/fem_v_gpu.sh <G> <작업 목록 .py> [β]
G=${1:?GPU}; JL=${2:?작업 목록}; BT=${3:-0}
cd "$(dirname "$0")/.." || exit 1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[카드 $G] 커밋 $(git log -1 --oneline) 시작 $(date +%T)"
i=0; n=0
while read -r ROOT TG ARGS; do
  if [ $((i % 7)) -eq "$G" ]; then
    while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 500 ]; do echo "[대기] 카드 $G 사용 중 $(date +%T)"; sleep 30; done
    EX_ROOT=$ROOT EX_NOWAIT=1 bash exe/fem_ex_run.sh "$TG" "$G" $ARGS < /dev/null > /dev/null 2>&1
    n=$((n + 1))
    O=/home/dkta/work/lbours/$ROOT/$TG; echo "$(date +%T) $TG $( [ -f $O/EX_DONE ] && echo 끝 || echo 실패) $(grep -a '롤아웃\]' $O/v_fem.log 2>/dev/null | cut -c1-160)"
  fi
  i=$((i + 1))
done < <(python $JL $BT)
echo "[카드 $G] 끝 $n 개 $(date +%T)"
touch /home/dkta/work/lbours/fem_v_lego/GPU${G}_DONE_$(basename $JL .py)_b$BT
