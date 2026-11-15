#!/bin/bash
# 절 CM180: 혼합 판을 카드 하나에서 돌리고 R2 lbours/tethex/<판> 로 올림 (아무것도 안 지움). 사용: bash exe/go_tethex.sh <카드> <판 이름들 쉼표>   (이름 첫 글자: A = --tmx 1 · B = --tmx 2 · C = --tmx 3 · 그 밖 (M·P: 앞 실행, e3_wq 표 번호 오류판) = 1·2)
export PATH=/home/dkta/bin:$PATH
G=$1; NAMES=$2
R=r2:storage/result/anchorflow/lbours/tethex
O=/home/dkta/work/tethex
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go_run_$G.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_run_$G.log -q; }
source exe/tethex_gen.sh
rclone cat r2:storage/result/anchorflow/lbours/tetfix/TC+ff3/cmd.txt > $O/tc.txt
[ $(wc -c < $O/tc.txt) -gt 100 ] || { say "TC+ff3 명령 못 읽음"; exit 1; }
for n in ${NAMES//,/ }; do
  mkdir -p $O/$n; th_cmd $O/tc.txt $O/$n $(case ${n:0:1} in B) echo 2;; C) echo 3;; *) echo 1;; esac) > $O/$n/cmd.txt
  grep -q -- "--out $O/$n *\$\|--out $O/$n " $O/$n/cmd.txt || { say "$n 명령 확인 실패"; exit 1; }
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
  say "$n 명령: $(grep -o 'tetfix_wrap.py\|tetc_wrap.py\|--tmx [0-9]\|--tet [0-9]\|--frames [0-9]*\|--gq_mass\|--gq_oct [0-9.]*\|--tff [0-9]\|--K [0-9]*' $O/$n/cmd.txt | tr '\n' ' ')"
  say "$n 카드 $G 시작 (계산 프로세스 $(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  ( while [ ! -f $O/$n/DONE ]; do sleep 60; tail -c 3000 $O/$n/log.txt 2>/dev/null | rclone rcat $R/$n/log_tail.txt; done ) &
  env CUDA_VISIBLE_DEVICES=$G timeout 5400 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  rc=$?
  touch $O/$n/DONE
  say "$n rc=$rc $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ') $(grep -a '롤아웃\|발산' $O/$n/log.txt | tail -1 | cut -c1-200)"
  rclone copy $O/$n $R/$n -q --transfers 8 --exclude DONE; say "$n R2 올림"
done
say "끝"
