#!/bin/bash
# 절 AG N 스윕 (비측정): AG_SWEEP="씬:N,N.. ..." 를 AG_CARDS 카드 x 일꾼 AG_W 개에 나눠 동시에 (fem_ag_gpu.sh) -> 씬별 N* (fem_ag_pick.py)
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
OS=/tmp/dkta/lbours/fem_ag_sum; RR=r2:storage/result/anchorflow/lbours; mkdir -p $OS
CARDS=(${AG_CARDS:-1}); WK=${AG_W:-2}; K=$(( ${#CARDS[@]} * WK )); TAG=${AG_TAG:-s1}
echo "[스윕 $TAG] $(git log -1 --oneline) $(date '+%F %T') 카드 ${CARDS[*]} x $WK, 목록 $(bash exe/fem_ag_jobs.sh | wc -l) 개: $AG_SWEEP"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
i=0
for G in ${CARDS[@]}; do for w in $(seq 1 $WK); do
  bash exe/fem_ag_gpu.sh $G $i $K < /dev/null > $OS/w_${TAG}_${G}_$w.log 2>&1 &
  i=$((i + 1))
done; done
wait
SC=$(for SN in $AG_SWEEP; do echo -n "${SN%%:*} "; done)
python exe/fem_ag_pick.py /tmp/dkta/lbours $OS/table_$TAG.md $OS/timed_$TAG.txt $SC
for S in $SC; do rclone copy /tmp/dkta/lbours/fem_ag_$S $RR/fem_ag_$S -q; done
rclone copy $OS $RR/fem_ag_sum -q
echo "AG_SWEEP_DONE $TAG $(date +%T)"; touch $OS/SWEEP_DONE_$TAG
