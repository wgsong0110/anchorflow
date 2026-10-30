#!/bin/bash
# 절 AI 일꾼 (비측정): 작업 목록에서 (줄 번호 % K == k) 인 실행만 카드 G 에서 차례로 (fem_ab_run.sh, NOT_TIMED; 출력 노드 /tmp + R2).
# 사용: bash exe/fem_ai_gpu.sh <G> <k> <K> <단계> [씬,...]
G=${1:?GPU}; k=${2:?k}; K=${3:?K}; ST=${4:?단계}; SCS=${5:-}
cd "$(dirname "$0")/.." || exit 1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
case $G in 1|2|3|4|5|6) ;; *) echo "[오류] 비측정은 카드 1~6 만 (0 은 단독 측정)"; exit 1;; esac
echo "[일꾼 카드 $G ($k/$K)] 커밋 $(git log -1 --oneline) 시작 $(date +%T)"
i=0
while read -r ROOT TG ARGS; do
  if [ $((i % K)) -eq "$k" ]; then
    O=/tmp/dkta/lbours/$ROOT/$TG
    EX_ROOT=$ROOT bash exe/fem_ab_run.sh "$TG" "$G" $ARGS < /dev/null > /dev/null 2>&1
    echo "$(date +%T) $TG $( [ -f $O/EX_DONE ] && echo 끝 || echo 실패) $(grep -a '롤아웃\]\|\[발산\]' $O/v_fem.log 2>/dev/null | tail -2 | cut -c1-170 | tr '\n' ' ')"
  fi
  i=$((i + 1))
done < <(python exe/fem_ai_jobs.py "$ST" $SCS)
echo "[일꾼 카드 $G ($k/$K)] 끝 $(date +%T)"
