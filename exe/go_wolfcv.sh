#!/bin/bash
# 절 CM93 영상 묶음 wolfcv 진입점 (exe/go_wolfh12.sh 복사; 덤프 렌더만, 재계산 없음). 코드 디렉토리에서. 사용: bash exe/go_wolfcv.sh <커밋> <단계>
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
R=r2:storage/result/anchorflow/lbours/wolfcv
O=/dev/shm/dkta/lbours/wolfcv; C=$(pwd -P)
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache /home/dkta/work/wolfconv
EN=go_$STG.END; rm -f $O/$EN
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STG 코드 $C; /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; /home $(df -h /home | tail -1 | awk '{print $4}') 남음; 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
export XDG_CACHE_HOME=/dev/shm/dkta/cache
bash exe/stab_wolfcv.sh $O $R $STG < /dev/null > $O/stab_$STG.out 2>&1
say "단계 $STG 끝 rc=$?"
rclone copy $O $R --exclude "go_*.END" -q
echo "끝 $(date '+%F %T')" > $O/$EN; rclone copyto $O/$EN $R/$EN -q
say "끝"
