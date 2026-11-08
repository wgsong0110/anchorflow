#!/bin/bash
# 절 CM54 묶음 rtrender1 진입점 (잡 133077 안 상주 시작점 exe/rtrender1_boot.sh 가 부름). 코드 = R2 bundle -> /dev/shm/dkta (노드 /tmp 98 % -- 쓰지 않음), 결과 /dev/shm -> R2 lbours/rtrender1 (5 분마다 + 판 끝날 때마다).
# 사용: bash go_rtrender1.sh <커밋> <단계[,단계]>   (끝 표식 R2 lbours/rtrender1/go_<단계>.END)
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
R=r2:storage/result/anchorflow/lbours/rtrender1
O=/dev/shm/dkta/lbours/rtrender1; CB=/dev/shm/dkta/rtrender1_code; C=$CB/$H
EN=go_${STG//,/_}.END
mkdir -p $O $CB /dev/shm/dkta/cache; rm -f $O/$EN $O/rtrender1.FAIL
L=$O/go_${STG//,/_}.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/$(basename $L) -q; }
say "시작 $(hostname) 커밋 $H 단계 $STG"
[ -s $CB/rtrender1_$H.bundle ] || rclone copyto $R/code/rtrender1_$H.bundle $CB/rtrender1_$H.bundle -q
[ -d $C/.git ] || git clone -q -b k1new $CB/rtrender1_$H.bundle $C
cd $C && git checkout -q $H || { say "코드 실패"; echo "코드 실패" | rclone rcat $R/$EN; exit 1; }
say "코드 $(git log --oneline -1 | cut -c1-100); /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_rtrender1 XDG_CACHE_HOME=/dev/shm/dkta/cache TMPDIR=/dev/shm/dkta/cache
( while [ ! -f $O/$EN ]; do sleep 300; [ -f $O/$EN ] && break; rclone copy $O $R --exclude "go_*.END" --exclude "pred.pt" --exclude "h5/**" --exclude "offpng/*.png" -q; done ) &
for st in ${STG//,/ }; do                                                                   # 단계 여럿 = 쉼표로 이어 차례로 (연기 실패면 멈춤)
  [ -f $O/rtrender1.FAIL ] && { say "실패 표식 있음 -- $st 안 함"; break; }
  say "단계 $st 시작"
  bash exe/stab_rtrender1.sh $O $R $st < /dev/null > $O/stab_$st.out 2>&1
  say "단계 $st 끝 rc=$?"
done
rclone copy $O $R --exclude "go_*.END" --exclude "pred.pt" --exclude "h5/**" --exclude "offpng/*.png" -q
D=$(rclone check $O $R --one-way --size-only --exclude "go_*.END" --exclude "pred.pt" --exclude "h5/**" --exclude "offpng/*.png" 2>&1 | grep -c ERROR)
say "R2 확인 차이 $D"
echo "끝 $(date '+%F %T') R2 차이 $D" > $O/$EN; rclone copyto $O/$EN $R/$EN -q
say "끝"
