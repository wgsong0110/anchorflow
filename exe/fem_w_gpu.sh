#!/bin/bash
# 절 W 한 카드 담당: 작업 목록에서 (줄 번호 % 7 == G) 인 실행만 이 카드에서 차례로 -- 비측정 (EX_NOWAIT=1, NOT_TIMED 표식).
# 같은 잡의 다른 작업 (절 U 등) 과 겹치지 않게: 실행마다 카드가 비어 있을 때 (500 MiB 아래) 까지 기다리고, 카드 잠금 디렉토리 ($W/gpulock_w/G) 를 잡는다.
# 사용: bash exe/fem_w_gpu.sh <G> <작업 목록 .py> <완료 표식 디렉토리>
G=${1:?GPU}; JL=${2:?작업 목록}; DD=${3:?표식 디렉토리}
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
mkdir -p $W/gpulock_w $DD
echo "[카드 $G] 커밋 $(git log -1 --oneline) 시작 $(date +%T)"
i=0; n=0
while read -r ROOT TG ARGS; do
  if [ $((i % 7)) -eq "$G" ]; then
    O=$W/lbours/$ROOT/$TG
    if [ ! -f $O/EX_DONE ]; then
      until mkdir $W/gpulock_w/$G 2>/dev/null; do sleep 20; done
      while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 500 ]; do echo "[대기] 카드 $G 사용 중 $(date +%T)"; sleep 30; done
      EX_ROOT=$ROOT EX_NOWAIT=1 bash exe/fem_ex_run.sh "$TG" "$G" $ARGS < /dev/null > /dev/null 2>&1
      rmdir $W/gpulock_w/$G
    fi
    n=$((n + 1))
    echo "$(date +%T) $TG $( [ -f $O/EX_DONE ] && echo 끝 || echo 실패) $(grep -a '롤아웃\]\|\[발산\]' $O/v_fem.log 2>/dev/null | tail -1 | cut -c1-160)"
  fi
  i=$((i + 1))
done < <(python $JL)
echo "[카드 $G] 끝 $n 개 $(date +%T)"
touch $DD/GPU${G}_DONE_$(basename $JL .py)
