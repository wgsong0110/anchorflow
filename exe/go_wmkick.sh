#!/bin/bash
# 절 CM23.4 묶음 wmkick 진입점 (잡 133077 안 pbs_attach 로; exe/job_run.sh AF_DETACH=1). 코드 = R2 bundle -> /dev/shm, 결과 처음부터 /dev/shm -> R2 lbours/wmkick.
export PATH=/home/dkta/bin:$PATH
H=__HASH__
R=r2:storage/result/anchorflow/lbours/wmkick
L2=r2:storage/result/anchorflow/lbours
O=/dev/shm/dkta/lbours/wmkick; CB=/tmp/dkta/wmkick_code; C=$CB/$H
mkdir -p $O $CB /dev/shm/dkta/cache /tmp/dkta; rm -f $O/wmkick.END
L=$O/go.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go.log -q; }
say "시작 $(hostname) 커밋 $H"
[ -s $CB/wmkick_$H.bundle ] || rclone copyto $R/code/wmkick_$H.bundle $CB/wmkick_$H.bundle -q
[ -d $C/.git ] || git clone -q -b k1new $CB/wmkick_$H.bundle $C
cd $C && git checkout -q $H || { say "코드 실패"; echo "코드 실패" | rclone rcat $R/wmkick.END; exit 1; }
say "코드 $(git log --oneline -1 | cut -c1-100); /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wmkick XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
( while [ ! -f $O/wmkick.END ]; do sleep 300; $PY exe/wmkick_tab.py $O > $O/wmkick_table.md 2> $O/wmkick_table.err; rclone copy $O $R --exclude wmkick.END --exclude "pred.pt" -q; done ) &
bash exe/stab_wmkick.sh $O $R < /dev/null > $O/stab_wmkick.out 2>&1
say "stab_wmkick 끝 rc=$? $(cat $O/wmkick.FAIL 2>/dev/null)"
$PY exe/wmkick_tab.py $O > $O/wmkick_table.md 2> $O/wmkick_table.err
rclone copy $O $R --exclude wmkick.END -q
D=$(rclone check $O $R --one-way --size-only --exclude wmkick.END 2>&1 | grep -c ERROR)
say "R2 확인 차이 $D"
{ [ -f $O/wmkick.FAIL ] && cat $O/wmkick.FAIL || echo "끝 $(date '+%F %T') R2 차이 $D"; } > $O/wmkick.END
rclone copyto $O/wmkick.END $R/wmkick.END -q
say "끝"
