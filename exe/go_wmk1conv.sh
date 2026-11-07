#!/bin/bash
# 절 CM23.3 묶음 wmk1conv 진입점 (잡 133077 안 pbs_attach 로; exe/job_run.sh AF_DETACH=1). 코드 = R2 bundle -> /dev/shm, 결과 처음부터 /dev/shm -> R2 lbours/wmk1conv.
export PATH=/home/dkta/bin:$PATH
H=__HASH__
R=r2:storage/result/anchorflow/lbours/wmk1conv
L2=r2:storage/result/anchorflow/lbours
O=/dev/shm/dkta/lbours/wmk1conv; CB=/dev/shm/dkta/wmk1conv_code; C=$CB/$H
mkdir -p $O $CB /dev/shm/dkta/cache; rm -f $O/wmk1conv.END
L=$O/go.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go.log -q; }
say "시작 $(hostname) 커밋 $H"
[ -s $CB/wmk1conv_$H.bundle ] || rclone copyto $R/code/wmk1conv_$H.bundle $CB/wmk1conv_$H.bundle -q
[ -d $C/.git ] || git clone -q -b k1new $CB/wmk1conv_$H.bundle $C
cd $C && git checkout -q $H || { say "코드 실패"; echo "코드 실패" | rclone rcat $R/wmk1conv.END; exit 1; }
say "코드 $(git log --oneline -1 | cut -c1-100); /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wmk1conv XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
( while [ ! -f $O/wmk1conv.END ]; do sleep 300; $PY exe/wmk1conv_tab.py $O > $O/wmk1conv_table.md 2> $O/wmk1conv_table.err; rclone copy $O $R --exclude wmk1conv.END --exclude "pred.pt" -q; done ) &
bash exe/stab_wmk1conv.sh $O $R < /dev/null > $O/stab_wmk1conv.out 2>&1
say "stab_wmk1conv 끝 rc=$? $(cat $O/wmk1conv.FAIL 2>/dev/null)"
$PY exe/wmk1conv_tab.py $O > $O/wmk1conv_table.md 2> $O/wmk1conv_table.err
rclone copy $O $R --exclude wmk1conv.END -q
D=$(rclone check $O $R --one-way --size-only --exclude wmk1conv.END 2>&1 | grep -c ERROR)
say "R2 확인 차이 $D"
{ [ -f $O/wmk1conv.FAIL ] && cat $O/wmk1conv.FAIL || echo "끝 $(date '+%F %T') R2 차이 $D"; } > $O/wmk1conv.END
rclone copyto $O/wmk1conv.END $R/wmk1conv.END -q
say "끝"
