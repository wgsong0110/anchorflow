#!/bin/bash
# 절 Q 1 차 한 카드 담당: fem_q_jobs.py 목록에서 (줄 번호 % 7 == G) 인 실행만 이 카드 (CUDA_VISIBLE_DEVICES=G) 에서 차례로 -- 비측정 (EX_NOWAIT=1, NOT_TIMED 표식).
# 사용: bash exe/fem_q_gpu.sh <G>
G=${1:?GPU}
cd "$(dirname "$0")/.." || exit 1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[카드 $G] 커밋 $(git log -1 --oneline) 시작 $(date +%T)"
i=0; n=0
while read -r ROOT TG ARGS; do
  if [ $((i % 7)) -eq "$G" ]; then
    EX_ROOT=$ROOT EX_NOWAIT=1 bash exe/fem_ex_run.sh "$TG" "$G" $ARGS < /dev/null > /dev/null 2>&1
    n=$((n + 1))
    if [ $n -eq 1 ]; then echo "[카드 확인] 첫 실행 뒤 이 카드 상태: $(nvidia-smi -i $G --query-gpu=index,uuid,memory.used --format=csv,noheader)"; fi
    O=/home/dkta/work/lbours/$ROOT/$TG; echo "$(date +%T) $TG $( [ -f $O/EX_DONE ] && echo 끝 || echo 실패) $(grep -a '롤아웃\]' $O/v_fem.log 2>/dev/null | cut -c1-160)"
  fi
  i=$((i + 1))
done < <(python exe/fem_q_jobs.py)
echo "[카드 $G] 끝 $n 개 $(date +%T)"
touch /home/dkta/work/lbours/fem_q_lego/GPU${G}_DONE
