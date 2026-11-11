#!/bin/bash
# wolf H8(=H9) 판 입자 F σ_min 히스토그램 진입점. 사용: bash exe/go_smin.sh <커밋> <단계 (an)>. 노드 / ·/tmp 안 씀 (/dev/shm/dkta/smin).
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
R=r2:storage/result/anchorflow/lbours/wolfcv/smin_viewer
O=/dev/shm/dkta/smin; mkdir -p $O /dev/shm/dkta/tmp; export TMPDIR=/dev/shm/dkta/tmp PYTHONUTF8=1 OMP_NUM_THREADS=8
PY=/home/dkta/.conda/envs/af/bin/python
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H; /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음"
[ -s $O/pred.pt ] || rclone copyto r2:storage/result/anchorflow/lbours/wolfh12/h8_1/pred.pt $O/pred.pt
say "pred.pt $(du -sh $O/pred.pt | cut -f1)"
$PY -u exe/smin_an.py $O/pred.pt /dev/shm/dkta/ag_data/gt_wolf $O/smin.json > $O/smin.log 2>&1
say "an rc=$?"
rclone copyto $O/smin.log $R/smin.log -q; rclone copyto $O/smin.json $R/smin.json -q
say "끝"; echo done > $O/go_$STG.END; rclone copyto $O/go_$STG.END $R/go_$STG.END -q
