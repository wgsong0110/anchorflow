#!/bin/bash
# 절 CM83 묶음 ext1 진입점 (exe/go_wolfh8.sh 복사; exe/ext1_boot.sh 가 부름; 코드 디렉토리에서). 코드 = R2 tar -> /dev/shm, 결과 /dev/shm -> R2 lbours/ext1 (5 분마다 + 판 끝날 때마다).
# 사용: bash go_ext1.sh <커밋> <단계>   (끝 표식 R2 lbours/ext1/go_<단계>.END; 마지막에 ext1.END)
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
R=r2:storage/result/anchorflow/lbours/ext1
O=/dev/shm/dkta/lbours/ext1; C=$(pwd -P)                                                  # 코드 = 지금 디렉토리 (ext1_boot.sh 가 R2 tar 를 /dev/shm 에 푼 곳; / 가 꽉 차 git clone 안 씀)
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR
EN=go_$STG.END
mkdir -p $O /dev/shm/dkta/cache; rm -f $O/$EN $O/ext1.FAIL
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STG 코드 $C"
say "/ $(df -h / | tail -1 | awk '{print $4}'); /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_ext1_$STG XDG_CACHE_HOME=/dev/shm/dkta/cache
( while [ ! -f $O/$EN ]; do sleep 300; [ -f $O/$EN ] && break; rclone copy $O $R --exclude "go_*.END" --exclude "pred.pt" -q; done ) &
for st in ${STG//,/ }; do                                                                   # 단계 여럿 = 쉼표로 이어 차례로 (연기 실패면 멈춤)
  [ -f $O/ext1.FAIL ] && { say "실패 표식 있음 -- $st 안 함"; break; }
  say "단계 $st 시작"
  bash exe/stab_ext1.sh $O $R $st < /dev/null > $O/stab_$st.out 2>&1
  say "단계 $st 끝 rc=$?"
done
rclone copy $O $R --exclude "go_*.END" -q
D=$(rclone check $O $R --one-way --size-only --exclude "go_*.END" 2>&1 | grep -c ERROR)
say "R2 확인 차이 $D"
echo "끝 $(date '+%F %T') R2 차이 $D" > $O/$EN; rclone copyto $O/$EN $R/$EN -q
say "끝"
