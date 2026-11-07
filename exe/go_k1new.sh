#!/bin/bash
# 절 CM23 묶음 k1new 진입점 (잡 133077 안 pbs_attach 로; exe/job_run.sh AF_DETACH=1). 코드 = R2 bundle -> /dev/shm, 결과 처음부터 /dev/shm -> R2 lbours/k1new.
export PATH=/home/dkta/bin:$PATH
H=__HASH__
R=r2:storage/result/anchorflow/lbours/k1new
L2=r2:storage/result/anchorflow/lbours
O=/dev/shm/dkta/lbours/k1new; CB=/dev/shm/dkta/k1new_code; C=$CB/$H
mkdir -p $O $CB /dev/shm/dkta/cache; rm -f $O/k1new.END
L=$O/go.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go.log -q; }
say "시작 $(hostname) 커밋 $H"
[ -s $CB/k1new_$H.bundle ] || rclone copyto $R/code/k1new_$H.bundle $CB/k1new_$H.bundle -q
[ -d $C/.git ] || git clone -q -b k1new $CB/k1new_$H.bundle $C
cd $C && git checkout -q $H || { say "코드 실패"; echo "코드 실패" | rclone rcat $R/k1new.END; exit 1; }
say "코드 $(git log --oneline -1 | cut -c1-100); /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
# 재사용: 레고 K2 해시 대조용 fint K2_on, 수박 K3 = wmbar b3_a·b3_b (덤프 빼고)
mkdir -p $O/ref/L_K2_fint; rclone copyto $L2/fint/K2_on/fem.jsonl $O/ref/L_K2_fint/fem.jsonl -q
for r in a b; do [ -f $O/M_K3_$r/summary.json ] || { rclone copy $L2/wmbar/b3_$r $O/M_K3_$r -q --exclude pred.pt --exclude lj.pt; echo "재사용 R2 lbours/wmbar/b3_$r" > $O/M_K3_$r/reuse.txt; }; done
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_k1new XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
( while [ ! -f $O/k1new.END ]; do sleep 300; $PY exe/k1new_tab.py $O > $O/k1new_table.md 2> $O/k1new_table.err; rclone copy $O $R --exclude k1new.END --exclude "pred.pt" -q; done ) &
bash exe/stab_k1new.sh $O $R < /dev/null > $O/stab_k1new.out 2>&1
say "stab_k1new 끝 rc=$? $(cat $O/k1new.FAIL 2>/dev/null)"
$PY exe/k1new_tab.py $O > $O/k1new_table.md 2> $O/k1new_table.err
rclone copy $O $R --exclude k1new.END -q
D=$(rclone check $O $R --one-way --size-only --exclude k1new.END 2>&1 | grep -c ERROR)
say "R2 확인 차이 $D"
{ [ -f $O/k1new.FAIL ] && cat $O/k1new.FAIL || echo "끝 $(date '+%F %T') R2 차이 $D"; } > $O/k1new.END
rclone copyto $O/k1new.END $R/k1new.END -q
say "끝"
