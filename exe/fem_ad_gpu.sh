#!/bin/bash
# 절 AD 일꾼: 작업 목록에서 (줄 번호 % K == k) 인 실행만 카드 G 에서 차례로 -- 비측정 (fem_c_run.sh, NOT_TIMED 표식).
# 사용: bash exe/fem_ad_gpu.sh <G> <k> <K> [작업 목록 .py]
G=${1:?GPU}; k=${2:?k}; K=${3:?K}; JL=${4:-exe/fem_ad_jobs.py}
cd "$(dirname "$0")/.." || exit 1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[카드 $G ($k/$K)] 커밋 $(git log -1 --oneline) 시작 $(date +%T)"
i=0; n=0
while read -r ROOT TG ARGS; do
  if [ $((i % K)) -eq "$k" ]; then
    EX_ROOT=$ROOT bash exe/fem_c_run.sh "$TG" "$G" $ARGS < /dev/null > /dev/null 2>&1
    n=$((n + 1))
    if [ $n -eq 1 ]; then echo "[카드 확인] 첫 실행 뒤: $(nvidia-smi -i $G --query-gpu=index,uuid,memory.used --format=csv,noheader)"; fi
    O=/home/dkta/work/lbours/$ROOT/$TG; echo "$(date +%T) $TG $( [ -f $O/EX_DONE ] && echo 끝 || echo 실패) $(grep -a '롤아웃\]' $O/v_fem.log 2>/dev/null | cut -c1-160)"
  fi
  i=$((i + 1))
done < <(case $JL in cat_*) cat ${JL#cat_};; *) python $JL;; esac)
echo "[카드 $G 일꾼 $k] 끝 $n 개 $(date +%T)"
