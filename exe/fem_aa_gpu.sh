#!/bin/bash
# 절 AA 한 카드 담당: 작업 목록 (exe/fem_aa_jobs.py) 에서 (줄 번호 % K == k) 인 실행만 카드 G 에서 차례로 -- 비측정 (EX_NOWAIT=1, NOT_TIMED 표식).
# 사용: bash exe/fem_aa_gpu.sh <G> <k> <K> [작업 목록 .py]
G=${1:?GPU}; k=${2:?k}; K=${3:?K}; JL=${4:-exe/fem_aa_jobs.py}
cd "$(dirname "$0")/.." || exit 1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[카드 $G ($k/$K)] 커밋 $(git log -1 --oneline) 시작 $(date +%T)"
i=0; n=0
while read -r ROOT TG ARGS; do
  if [ $((i % K)) -eq "$k" ]; then
    EX_ROOT=$ROOT EX_NOWAIT=1 bash exe/fem_ex_run.sh "$TG" "$G" $ARGS < /dev/null > /dev/null 2>&1
    n=$((n + 1))
    if [ $n -eq 1 ]; then echo "[카드 확인] 첫 실행 뒤: $(nvidia-smi -i $G --query-gpu=index,uuid,memory.used --format=csv,noheader)"; fi
    O=/home/dkta/work/lbours/$ROOT/$TG; echo "$(date +%T) $TG $( [ -f $O/EX_DONE ] && echo 끝 || echo 실패) $(grep -a '롤아웃\]' $O/v_fem.log 2>/dev/null | cut -c1-160)"
  fi
  i=$((i + 1))
done < <(python $JL)
echo "[카드 $G] 끝 $n 개 $(date +%T)"
mkdir -p /home/dkta/work/lbours/fem_aa_lego; touch /home/dkta/work/lbours/fem_aa_lego/GPU${G}_DONE_$(basename $JL .py)
