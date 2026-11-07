#!/bin/bash
# 절 CM26 묶음 wolfprop 진입점 (잡 133077 안 pbs_attach 로; exe/job_run.sh AF_DETACH=1). 코드 = R2 bundle -> /tmp, 결과 처음부터 /dev/shm -> R2 lbours/wolfprop.
# 사용: bash go_wolfprop.sh <커밋>
export PATH=/home/dkta/bin:$PATH
H=${1:?커밋}
R=r2:storage/result/anchorflow/lbours/wolfprop
O=/dev/shm/dkta/lbours/wolfprop; CB=/tmp/dkta/wolfprop_code; C=$CB/$H
mkdir -p $O $CB /dev/shm/dkta/cache /tmp/dkta; rm -f $O/wolfprop.END $O/wolfprop.FAIL
L=$O/go.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go.log -q; }
say "시작 $(hostname) 커밋 $H"
[ -s $CB/wolfprop_$H.bundle ] || rclone copyto $R/code/wolfprop_$H.bundle $CB/wolfprop_$H.bundle -q
[ -d $C/.git ] || git clone -q -b k1new $CB/wolfprop_$H.bundle $C
cd $C && git checkout -q $H || { say "코드 실패"; echo "코드 실패" | rclone rcat $R/wolfprop.END; exit 1; }
say "코드 $(git log --oneline -1 | cut -c1-100); /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wolfprop XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
WDP=/tmp/dkta/ag_data/wolf.pt; [ -s $WDP ] || WDP=/dev/shm/dkta/ag_data/wolf.pt
WW=/dev/shm/dkta/lbours/wolfwhy_ref; mkdir -p $WW
for q in base A_stick; do [ -s $WW/$q/fem.jsonl ] || rclone copyto r2:storage/result/anchorflow/lbours/wolfwhy/$q/fem.jsonl $WW/$q/fem.jsonl -q; done
bash exe/stab_wolfprop.sh $O $R < /dev/null > $O/stab_wolfprop.out 2>&1
say "stab_wolfprop 끝 rc=$? $(cat $O/wolfprop.FAIL 2>/dev/null)"
$PY exe/wolfprop_tab.py $O $WDP $WW > $O/wolfprop_table.md 2> $O/wolfprop_table.err
rclone copy $O $R --exclude wolfprop.END -q
D=$(rclone check $O $R --one-way --size-only --exclude wolfprop.END 2>&1 | grep -c ERROR)
say "R2 확인 차이 $D"
{ [ -f $O/wolfprop.FAIL ] && cat $O/wolfprop.FAIL || echo "끝 $(date '+%F %T') R2 차이 $D"; } > $O/wolfprop.END
rclone copyto $O/wolfprop.END $R/wolfprop.END -q
say "끝"
