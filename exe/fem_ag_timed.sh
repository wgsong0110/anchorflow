#!/bin/bash
# 절 AG 단독 측정: AG_TIMED="씬:N ..." 를 회차 1~3 번갈아 GPU0 (gpu0.lock, fem_ab_run.sh EX_TIMED=1). 회마다 시작 전 카드 1~6 이 비기를 (≤ 500 MiB) 기다린다.
# 회차 1 은 --dump_f32 (F 덤프 -- 저장은 측정 밖), 2·3 은 --no_dump.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
source exe/fem_ag_scene.sh
OS=/tmp/dkta/lbours/fem_ag_sum; RR=r2:storage/result/anchorflow/lbours; mkdir -p $OS
echo "[단독 측정] $(git log -1 --oneline) $(date '+%F %T') $AG_TIMED"
idle() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$1!=0 && $2>500{b=1} END{exit b}'; }
for r in 1 2 3; do
  for SN in ${AG_TIMED:?}; do
    S=${SN%%:*}; N=${SN#*:}; ag_scene $S
    TG=ag_${S}_n${N}_t$r; D="--no_dump"; [ $r = 1 ] && D="--dump_f32"
    n=0; until idle; do sleep 30; n=$((n + 1)); [ $((n % 20)) = 0 ] && echo "  [대기] 다른 카드 사용 중 $(date +%T): $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"; done
    EX_TIMED=1 EX_ROOT=fem_ag_$S bash exe/fem_ab_run.sh $TG 0 $SARGS $REC $D --N $N < /dev/null > /dev/null 2>&1
    O=/tmp/dkta/lbours/fem_ag_$S/$TG
    echo "$(date +%T) $TG $(grep -a '롤아웃\]' $O/v_fem.log | cut -c1-200) | 다른 카드 $(awk -F, '$1!=0{print $2}' $O/gpu_at_start.txt | tr -d ' \n')"
  done
done
SC=$(for SN in $AG_TIMED; do echo -n "${SN%%:*} "; done)
python exe/fem_ag_table.py /tmp/dkta/lbours $SC > $OS/table_timed.md; cat $OS/table_timed.md
for S in $SC; do rclone copy /tmp/dkta/lbours/fem_ag_$S $RR/fem_ag_$S -q --exclude "roll_fem.pt"; done
rclone copy $OS $RR/fem_ag_sum -q
echo "AG_TIMED_DONE $(date +%T)"; touch $OS/TIMED_DONE
