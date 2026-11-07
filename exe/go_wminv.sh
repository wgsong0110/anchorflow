#!/bin/bash
# 절 CM29 묶음 wminv 진입점 (잡 133077 안 pbs_attach 로; exe/job_run.sh AF_DETACH=1). 코드 = R2 bundle -> /tmp, 결과 처음부터 /dev/shm -> R2 lbours/wminv (5 분마다 통째 + 판 끝날 때마다).
# 사용: bash go_wminv.sh <커밋> <단계 이름> <판 ...>   (끝 표식 R2 lbours/wminv/go_<단계>.END; 마지막 단계는 조정이 wminv.END 를 씀)
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2; shift 2; VS="$*"
R=r2:storage/result/anchorflow/lbours/wminv
O=/dev/shm/dkta/lbours/wminv; CB=/tmp/dkta/wminv_code; C=$CB/$H
EN=go_$STG.END
mkdir -p $O $CB /dev/shm/dkta/cache /tmp/dkta; rm -f $O/$EN $O/wminv.FAIL
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STG 판 $VS"
[ -s $CB/wminv_$H.bundle ] || rclone copyto $R/code/wminv_$H.bundle $CB/wminv_$H.bundle -q
[ -d $C/.git ] || git clone -q -b k1new $CB/wminv_$H.bundle $C
cd $C && git checkout -q $H || { say "코드 실패"; echo "코드 실패" | rclone rcat $R/$EN; exit 1; }
say "코드 $(git log --oneline -1 | cut -c1-100); /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음, 메모리 $(free -g | awk '/Mem/{print $7}') G 가용; 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wminv XDG_CACHE_HOME=/dev/shm/dkta/cache
( while [ ! -f $O/$EN ]; do sleep 300; [ -f $O/$EN ] && break; rclone copy $O $R --exclude "go_*.END" --exclude "pred.pt" --exclude "srcold/**" -q; done ) &
if [ "$STG" = vid ]; then bash exe/stab_wminv_vid.sh $O $R $VS < /dev/null > $O/stab_$STG.out 2>&1; else bash exe/stab_wminv.sh $O $R $STG $VS < /dev/null > $O/stab_$STG.out 2>&1; fi
say "stab_wminv 끝 rc=$? $(cat $O/wminv.FAIL 2>/dev/null)"
rclone copy $O $R --exclude "go_*.END" --exclude "srcold/**" -q
D=$(rclone check $O $R --one-way --size-only --exclude "go_*.END" --exclude "srcold/**" 2>&1 | grep -c ERROR)
say "R2 확인 차이 $D"
{ [ -f $O/wminv.FAIL ] && cat $O/wminv.FAIL || echo "끝 $(date '+%F %T') R2 차이 $D"; } > $O/$EN
rclone copyto $O/$EN $R/$EN -q
say "끝"
