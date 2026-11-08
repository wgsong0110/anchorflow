#!/bin/bash
# 수박 입자 날아감 비교 영상 진입점 (잡 133077 안, exe/job_run.sh 와 같은 pbs_attach 떼어냄으로 `rclone cat .../wmescape/code/wmescape_go.sh | bash`).
# 코드 = R2 lbours/wmescape/code/launch.txt 의 커밋 bundle -> /tmp, 결과 /tmp/dkta/lbours/wmescape -> R2 lbours/wmescape (판 끝날 때마다 + 5 분마다). 끝 표식 R2 wmescape.END.
export PATH=/home/dkta/bin:$PATH
R=r2:storage/result/anchorflow/lbours/wmescape
H=$(rclone cat $R/code/launch.txt 2>/dev/null | head -1 | awk '{print $1}')
O=/tmp/dkta/lbours/wmescape; CB=/tmp/dkta/wmescape_code; C=$CB/$H
mkdir -p $O $CB /dev/shm/dkta/cache; rm -f $O/wmescape.END
L=$O/go.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go.log -q; }
say "시작 $(hostname) 커밋 '$H'"
[ -n "$H" ] || { say "launch.txt 없음"; echo "코드 실패" | rclone rcat $R/wmescape.END; exit 1; }
[ -s $CB/wmescape_$H.bundle ] || rclone copyto $R/code/wmescape_$H.bundle $CB/wmescape_$H.bundle -q
[ -d $C/.git ] || git clone -q -b k1new $CB/wmescape_$H.bundle $C
cd $C && git checkout -q $H || { say "코드 실패"; echo "코드 실패" | rclone rcat $R/wmescape.END; exit 1; }
say "코드 $(git log --oneline -1 | cut -c1-100); /tmp $(df -h /tmp | tail -1 | awk '{print $4}') · /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | tr '\n' ' ')"
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wmescape XDG_CACHE_HOME=/dev/shm/dkta/cache
( while [ ! -f $O/wmescape.END ]; do sleep 300; [ -f $O/wmescape.END ] && break; rclone copy $O $R --exclude "pred.pt" --exclude "h5_fem/**" -q; done ) &
bash exe/wmescape_vid.sh $O $R < /dev/null > $O/vid.out 2>&1
say "vid 끝 rc=$?"
rclone copy $O $R --exclude "pred.pt" --exclude "h5_fem/**" -q
echo "끝 $(date '+%F %T')" > $O/wmescape.END; rclone copyto $O/wmescape.END $R/wmescape.END -q
say "끝"
