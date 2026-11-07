#!/bin/bash
# 절 CM23.3 영상 진입점 (잡 133077 안 pbs_attach 로; exe/job_run.sh AF_DETACH=1). 코드 = R2 bundle -> /dev/shm, 결과 /dev/shm -> R2 lbours/wmk1conv (v_* 덤프 판, vid/).
export PATH=/home/dkta/bin:$PATH
H=__HASH__
R=r2:storage/result/anchorflow/lbours/wmk1conv
O=/dev/shm/dkta/lbours/wmk1conv; CB=/dev/shm/dkta/wmk1vid_code; C=$CB/$H
mkdir -p $O $CB /dev/shm/dkta/cache; rm -f $O/wmk1vid.END
L=$O/go_vid.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_vid.log -q; }
say "시작 $(hostname) 커밋 $H"
[ -s $CB/wmk1vid_$H.bundle ] || rclone copyto $R/code/wmk1vid_$H.bundle $CB/wmk1vid_$H.bundle -q
[ -d $C/.git ] || git clone -q -b k1new $CB/wmk1vid_$H.bundle $C
cd $C && git checkout -q $H || { say "코드 실패"; echo "코드 실패" | rclone rcat $R/wmk1vid.END; exit 1; }
say "코드 $(git log --oneline -1 | cut -c1-100); /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wmk1vid XDG_CACHE_HOME=/dev/shm/dkta/cache
bash exe/stab_wmk1vid.sh $O $R < /dev/null > $O/stab_wmk1vid.out 2>&1
say "stab_wmk1vid 끝 rc=$?"
rclone copy $O/stab_wmk1vid.out $R/ -q
echo "끝 $(date '+%F %T')" > $O/wmk1vid.END
rclone copyto $O/wmk1vid.END $R/wmk1vid.END -q
say "끝"
