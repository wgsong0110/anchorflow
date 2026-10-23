#!/bin/bash
# exe/pfnew.sh 실행 대기열 (사용자 10-23: 빈 카드에만, 모자라면 순서대로). 계산 노드 잡 안에서 exe/job_run.sh 로 띄운다.
#   대기열 파일 $Q 의 줄 "<판> <궤적> <워크트리>" 를 위에서부터, **빈 카드**(메모리 < 500 MiB, GPU 0 제외, 내 점유 표식 없음)가
#   생길 때마다 하나씩 그 카드에 띄운다 (bash <워크트리>/exe/pfnew.sh <판> <궤적> <GPU>). 줄은 나중에 덧붙여도 된다.
#   띄운 줄은 $Q.started 에 남는다. 대기열이 다 띄워지고 내 실행이 모두 끝나면 멈춘다 (AF_PFQ_HOLD 파일이 있으면 계속 기다린다).
W=/home/dkta/work; Q=${PFQ:-$W/lbours/pfnew_queue.txt}; L=$W/lbours/pfnew_logs; mkdir -p $L; touch $Q $Q.started
exec 8>$Q.lock; flock -n 8 || { echo "[중단] 대기열이 이미 돌고 있다"; exit 0; }
declare -A PID
echo "[대기열 시작] $(date '+%F %T') $(hostname)"
while true; do
  for g in "${!PID[@]}"; do kill -0 ${PID[$g]} 2>/dev/null || { echo "[끝] GPU$g pid ${PID[$g]} $(date '+%T')"; unset PID[$g]; }; done
  next=$(grep -vxF -f $Q.started $Q | grep -v '^\s*$' | head -1)
  if [ -n "$next" ]; then
    free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$1!=0 && $2<500{print $1}')
    for g in $free; do
      [ -n "${PID[$g]:-}" ] && continue
      read -r K T WT <<<"$next"
      [ -f $WT/exe/pfnew.sh ] || { echo "[오류] 워크트리 없음 $WT"; echo "$next" >> $Q.started; break; }
      bash $WT/exe/pfnew.sh $K $T $g > $L/${K}_$T.log 2>&1 < /dev/null &
      PID[$g]=$!; echo "$next" >> $Q.started
      echo "[띄움] $(date '+%T') $K $T GPU$g pid $! ($WT)"
      break
    done
  elif [ ${#PID[@]} = 0 ] && [ ! -f $W/lbours/AF_PFQ_HOLD ]; then
    echo "[대기열 끝] $(date '+%F %T')"; break
  fi
  sleep 60
done
