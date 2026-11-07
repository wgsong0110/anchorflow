#!/bin/bash
# 절 CM25 묶음 wolfwhy 진입점 (잡 133077 안 pbs_attach 로; exe/job_run.sh AF_DETACH=1). 코드 = R2 bundle -> /dev/shm, 결과 처음부터 /dev/shm -> R2 lbours/wolfwhy.
export PATH=/home/dkta/bin:$PATH
H=__HASH__
R=r2:storage/result/anchorflow/lbours/wolfwhy
L2=r2:storage/result/anchorflow/lbours
O=/dev/shm/dkta/lbours/wolfwhy; CB=/tmp/dkta/wolfwhy_code; C=$CB/$H
mkdir -p $O $CB /dev/shm/dkta/cache /tmp/dkta; rm -f $O/wolfwhy.END
L=$O/go.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go.log -q; }
say "시작 $(hostname) 커밋 $H"
[ -s $CB/wolfwhy_$H.bundle ] || rclone copyto $R/code/wolfwhy_$H.bundle $CB/wolfwhy_$H.bundle -q
[ -d $C/.git ] || git clone -q -b k1new $CB/wolfwhy_$H.bundle $C
cd $C && git checkout -q $H || { say "코드 실패"; echo "코드 실패" | rclone rcat $R/wolfwhy.END; exit 1; }
say "코드 $(git log --oneline -1 | cut -c1-100); /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wolfwhy XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
WDP=/tmp/dkta/ag_data/wolf.pt; [ -s $WDP ] || WDP=/dev/shm/dkta/ag_data/wolf.pt
( while [ ! -f $O/wolfwhy.END ]; do sleep 300; $PY exe/wolfwhy_tab.py $O $WDP > $O/wolfwhy_table.md 2> $O/wolfwhy_table.err; rclone copy $O $R --exclude wolfwhy.END --exclude "pred.pt" -q; done ) &
bash exe/stab_wolfwhy.sh $O $R < /dev/null > $O/stab_wolfwhy.out 2>&1
say "stab_wolfwhy 끝 rc=$? $(cat $O/wolfwhy.FAIL 2>/dev/null)"
$PY exe/wolfwhy_tab.py $O $WDP > $O/wolfwhy_table.md 2> $O/wolfwhy_table.err
rclone copy $O $R --exclude wolfwhy.END -q
D=$(rclone check $O $R --one-way --size-only --exclude wolfwhy.END 2>&1 | grep -c ERROR)
say "R2 확인 차이 $D"
{ [ -f $O/wolfwhy.FAIL ] && cat $O/wolfwhy.FAIL || echo "끝 $(date '+%F %T') R2 차이 $D"; } > $O/wolfwhy.END
rclone copyto $O/wolfwhy.END $R/wolfwhy.END -q
say "끝"
