#!/bin/bash
# 절 CM110 묶음 floorpen 진입점 (exe/go_wolfcv.sh 복사; CPU 만, 카드 안 씀). 코드 디렉토리에서. 사용: bash exe/go_floorpen.sh <커밋> <단계>
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
R=r2:storage/result/anchorflow/lbours/wolfcv/floorpen
O=/dev/shm/dkta/lbours/floorpen; C=$(pwd -P)
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 PYTHONUTF8=1
PY=/home/dkta/.conda/envs/af/bin/python
SRC=r2:storage/result/anchorflow/lbours/wolfh12
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STG; /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음"
CFG=/dev/shm/dkta/ag_data/gt_wolf/config.json; [ -f $CFG ] || CFG=/home/dkta/work/PhysGaussian/config/wolf_config.json
DATA=/dev/shm/dkta/ag_data/wolf.pt
say "cfg $CFG data $DATA $(ls -la $DATA 2>&1 | awk '{print $5}')"
for p in "aC_1 26" "h8_1 50"; do
  set -- $p; n=$1; nd=$2
  D=/dev/shm/dkta/floorpen_dump/$n; mkdir -p $D
  [ -s $D/pred.pt ] || rclone copyto $SRC/$n/pred.pt $D/pred.pt -q
  rclone copyto $SRC/$n/e1.jsonl $D/e1.jsonl -q
  $PY -u exe/floorpen.py $n $D/pred.pt $CFG $DATA $D/e1.jsonl $nd $O/floorpen_$n.json > $O/floorpen_$n.log 2>&1
  say "$n rc=$? $(tail -1 $O/floorpen_$n.log | cut -c1-200)"
  rm -rf $D
done
rclone copy $O $R --exclude "go_*.END" -q
echo "끝 $(date '+%F %T')" > $O/go_$STG.END; rclone copyto $O/go_$STG.END $R/go_$STG.END -q
say "끝"
