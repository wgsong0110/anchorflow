#!/bin/bash
# 절 CM27 묶음 wolffix 진입점 (잡 133077 안 pbs_attach 로; exe/job_run.sh AF_DETACH=1). 코드 = R2 bundle -> /tmp, 결과 처음부터 /dev/shm -> R2 lbours/wolffix.
# 사용: bash go_wolffix.sh <커밋>
export PATH=/home/dkta/bin:$PATH
H=${1:?커밋}
R=r2:storage/result/anchorflow/lbours/wolffix
O=/dev/shm/dkta/lbours/wolffix; CB=/tmp/dkta/wolffix_code; C=$CB/$H
mkdir -p $O $CB /dev/shm/dkta/cache /tmp/dkta; rm -f $O/wolffix.END $O/wolffix.FAIL
L=$O/go.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go.log -q; }
say "시작 $(hostname) 커밋 $H"
[ -s $CB/wolffix_$H.bundle ] || rclone copyto $R/code/wolffix_$H.bundle $CB/wolffix_$H.bundle -q
[ -d $C/.git ] || git clone -q -b k1new $CB/wolffix_$H.bundle $C
cd $C && git checkout -q $H || { say "코드 실패"; echo "코드 실패" | rclone rcat $R/wolffix.END; exit 1; }
say "코드 $(git log --oneline -1 | cut -c1-100); /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wolffix XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
WDP=/tmp/dkta/ag_data/wolf.pt; [ -s $WDP ] || WDP=/dev/shm/dkta/ag_data/wolf.pt
bash exe/stab_wolffix.sh $O $R < /dev/null > $O/stab_wolffix.out 2>&1
say "stab_wolffix 끝 rc=$? $(cat $O/wolffix.FAIL 2>/dev/null)"
$PY exe/wolffix_tab.py $O $WDP > $O/wolffix_table.md 2> $O/wolffix_table.err
rclone copy $O $R --exclude wolffix.END -q --transfers 8
D=$(rclone check $O $R --one-way --size-only --exclude wolffix.END 2>&1 | grep -c ERROR)
say "R2 확인 차이 $D"
{ [ -f $O/wolffix.FAIL ] && cat $O/wolffix.FAIL || echo "끝 $(date '+%F %T') R2 차이 $D"; } > $O/wolffix.END
rclone copyto $O/wolffix.END $R/wolffix.END -q
say "끝"
