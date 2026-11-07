#!/bin/bash
# 절 CM24 묶음 wmk1fix 진입점 (잡 133077 안 pbs_attach 로; exe/job_run.sh AF_DETACH=1). 코드 = R2 bundle -> /tmp, 결과 처음부터 /dev/shm -> R2 lbours/wmk1fix (5 분마다 통째 + 판 끝날 때마다).
# 사용: bash go_wmk1fix.sh <커밋> <단계 스크립트 (exe/stab_wmk1fix.sh 등)> <끝 표식 이름>
export PATH=/home/dkta/bin:$PATH
H=$1; ST=$2; EN=$3
R=r2:storage/result/anchorflow/lbours/wmk1fix
O=/dev/shm/dkta/lbours/wmk1fix; CB=/tmp/dkta/wmk1fix_code; C=$CB/$H
mkdir -p $O $CB /dev/shm/dkta/cache /tmp/dkta; rm -f $O/$EN
L=$O/go_$EN.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$EN.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $ST"
[ -s $CB/wmk1fix_$H.bundle ] || rclone copyto $R/code/wmk1fix_$H.bundle $CB/wmk1fix_$H.bundle -q
[ -d $C/.git ] || git clone -q -b k1new $CB/wmk1fix_$H.bundle $C
cd $C && git checkout -q $H || { say "코드 실패"; echo "코드 실패" | rclone rcat $R/$EN; exit 1; }
say "코드 $(git log --oneline -1 | cut -c1-100); /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음, 메모리 $(free -g | awk '/Mem/{print $7}') G 가용; 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wmk1fix XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
( while [ ! -f $O/$EN ]; do sleep 300; [ -f $O/$EN ] && break; rclone copy $O $R --exclude "$EN" --exclude "pred.pt" --exclude "srcold/**" -q; done ) &
bash $ST $O $R < /dev/null > $O/${EN%.END}.out 2>&1
say "$ST 끝 rc=$? $(cat $O/wmk1fix.FAIL 2>/dev/null)"
rclone copy $O $R --exclude "$EN" --exclude "srcold/**" -q
D=$(rclone check $O $R --one-way --size-only --exclude "$EN" --exclude "srcold/**" 2>&1 | grep -c ERROR)
say "R2 확인 차이 $D"
{ [ -f $O/wmk1fix.FAIL ] && cat $O/wmk1fix.FAIL || echo "끝 $(date '+%F %T') R2 차이 $D"; } > $O/$EN
rclone copyto $O/$EN $R/$EN -q
say "끝"
