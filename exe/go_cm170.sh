#!/bin/bash
# 절 CM170: 먼 쪽 평형 잔차 기록 판들을 카드별로 돌리고 R2 lbours/wolfcv/cm170/<판> 로 올림 (아무것도 안 지움). 코드 디렉토리에서. 사용: bash exe/go_cm170.sh <태그> <카드0 판들> <카드1 판들>   (쉼표 구분)
export PATH=/home/dkta/bin:$PATH
H=$1; G0=$2; G1=$3
R=r2:storage/result/anchorflow/lbours/wolfcv/cm170
O=/home/dkta/work/cm170
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go_run.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_run.log -q; }
source exe/cm139_gen.sh; source exe/cm140_gen.sh; source exe/cm170_gen.sh
rclone cat r2:storage/result/anchorflow/lbours/wolfcv/cm131/k5/cmd.txt > $O/k5src.txt
[ $(wc -c < $O/k5src.txt) -gt 100 ] || { say "k5 명령 못 읽음"; exit 1; }
mk() { local n=$1; mkdir -p $O/$n; cm170_cmd $n $O/k5src.txt $O/$n > $O/$n/cmd.txt
  grep -q -- "--out $O/$n\$" $O/$n/cmd.txt || { say "$n 명령 확인 실패"; return 1; }
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
  say "$n 명령: $(grep -o 'cm170_wrap.py\|--newton [0-9]*\|--cg [0-9]*\|--frames [0-9]*\|--xg 1\|--xit_from [0-9]' $O/$n/cmd.txt | tr '\n' ' ')"; }
runone() { local n=$1 g=$2
  say "$n 카드 $g 시작 (계산 프로세스 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  ( while [ ! -f $O/$n/DONE ]; do sleep 60; tail -c 3000 $O/$n/log.txt 2>/dev/null | rclone rcat $R/$n/log_tail.txt; rclone copy $O/$n $R/$n -q --transfers 8 --exclude DONE; done ) &
  env CUDA_VISIBLE_DEVICES=$g timeout 5400 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; touch $O/$n/DONE
  say "$n rc=$rc $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ') $(grep -a '절 CM170' $O/$n/log.txt | head -1 | cut -c1-120)"
  rclone copy $O/$n $R/$n -q --transfers 8; say "$n R2 올림"; return 0; }
line() { local g=$1; shift; for n in "$@"; do runone $n $g; done; }
for n in ${G0//,/ } ${G1//,/ }; do [ "$n" = - ] || mk $n || exit 1; done
( [ "$G0" = - ] || line 0 ${G0//,/ } ) & P1=$!; sleep 15; ( [ "$G1" = - ] || line 1 ${G1//,/ } ) & P2=$!; wait $P1 $P2
say "끝 R2 대조 $(rclone check $O $R --one-way -q 2>&1 | grep -c ERROR) 개 차이"
