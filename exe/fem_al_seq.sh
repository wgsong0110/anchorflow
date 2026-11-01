#!/bin/bash
# 절 AL: 여러 회를 차례로 (fem_al_run.sh). 사용: bash exe/fem_al_seq.sh "<태그>|<인자>" ...
cd "$(dirname "$0")/.." || exit 1
echo "[커밋] $(git log -1 --oneline) $(date '+%F %T')"
for it in "$@"; do TG=${it%%|*}; AR=${it#*|}; [ "$AR" = "$it" ] && AR=""
  bash exe/fem_al_run.sh $TG $AR
  O=/tmp/dkta/lbours/fem_al_wolf/$TG; echo "$(date +%T) $TG $(cat $O/err_fem.txt 2>/dev/null | tail -1 | cut -c1-200)"
done
echo "AL_SEQ_DONE $(date +%T)"
