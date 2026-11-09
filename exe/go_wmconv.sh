#!/bin/bash
# 절 CM90 묶음 wmconv 진입점 (exe/wmconv_launch.sh 가 부름; 코드 디렉토리에서). 결과 /dev/shm/dkta/lbours/wmconv -> R2 lbours/wmconv (5 분마다 + 판 끝날 때마다), 끝 표식 R2 wmconv.END
export PATH=/home/dkta/bin:$PATH
H=$1
R=r2:storage/result/anchorflow/lbours/wmconv
O=/dev/shm/dkta/lbours/wmconv
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache; rm -f $O/wmconv.END
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wmconv XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go.log -q; }
say "시작 $(hostname) 커밋 $H 코드 $(pwd -P); / $(df -h / | tail -1 | awk '{print $4}'); /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4, $5}'); 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
( while [ ! -f $O/wmconv.END ]; do sleep 300; [ -f $O/wmconv.END ] && break; rclone copy $O $R --exclude pred.pt --exclude "h5_fem/**" --exclude "sc_lj/**" --exclude pred_v.pt -q; done ) &
bash exe/stab_wmconv.sh $O $R < /dev/null > $O/stab.out 2>&1
say "stab 끝 rc=$?"
rclone copy $O $R --exclude "h5_fem/**" --exclude "sc_lj/**" --exclude pred_v.pt -q
D=$(rclone check $O $R --one-way --size-only --exclude "h5_fem/**" --exclude "sc_lj/**" --exclude pred_v.pt 2>&1 | grep -c ERROR)
say "R2 확인 차이 $D"
echo "끝 $(date '+%F %T') R2 차이 $D" > $O/wmconv.END; rclone copyto $O/wmconv.END $R/wmconv.END -q
