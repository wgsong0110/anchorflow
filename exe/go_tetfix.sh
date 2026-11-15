#!/bin/bash
# 절 CM178: 판들을 (카드 하나에서 차례로) 돌리고 R2 lbours/tetfix/<판> 로 올림 (아무것도 안 지움). 사용: bash exe/go_tetfix.sh <카드> <판 이름들 쉼표>
export PATH=/home/dkta/bin:$PATH
G=$1; NAMES=$2
R=r2:storage/result/anchorflow/lbours/tetfix
O=/home/dkta/work/tetfix
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go_run_$G.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_run_$G.log -q; }
source exe/tetfix_gen.sh
S=r2:storage/result/anchorflow/lbours
rclone cat $S/tetc/cL_1/cmd.txt > $O/cL1.txt; rclone cat $S/tetc/dL_1/cmd.txt > $O/dL1.txt; rclone cat $S/speed13/oXL_1/cmd.txt > $O/oXL1.txt
for sc in W F B; do rclone cat $S/tetc/c${sc}_1/cmd.txt > $O/c${sc}1.txt; rclone cat $S/speed13/oX${sc}_1/cmd.txt > $O/oX${sc}1.txt; done
for f in cL1 dL1 oXL1 cW1 cF1 cB1 oXW1 oXF1 oXB1; do [ $(wc -c < $O/$f.txt) -gt 100 ] || { say "$f 명령 못 읽음"; exit 1; }; done
for n in ${NAMES//,/ }; do
  mkdir -p $O/$n; tf_cmd $n $O/cL1.txt $O/dL1.txt $O/$n $O/oXL1.txt > $O/$n/cmd.txt || exit 1
  grep -q -- "--out $O/$n *\$\|--out $O/$n " $O/$n/cmd.txt || { say "$n 명령 확인 실패"; exit 1; }
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
  say "$n 명령: $(grep -o 'tetfix_wrap.py\|tetc_wrap.py\|--frames [0-9]*\|--nrec\|--tfx [0-9]\|--tfw [0-9:]*\|--tcm [0-9]\|--gq_oct [0-9.]*\|--K [0-9]*' $O/$n/cmd.txt | tr '\n' ' ')"
  say "$n 카드 $G 시작 (계산 프로세스 $(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  ( while [ ! -f $O/$n/DONE ]; do sleep 60; tail -c 3000 $O/$n/log.txt 2>/dev/null | rclone rcat $R/$n/log_tail.txt; done ) &
  env CUDA_VISIBLE_DEVICES=$G timeout 5400 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  rc=$?
  touch $O/$n/DONE
  say "$n rc=$rc $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ') $(grep -a '롤아웃\|발산' $O/$n/log.txt | tail -1 | cut -c1-200)"
  rclone copy $O/$n $R/$n -q --transfers 8 --exclude DONE; say "$n R2 올림"
done
say "끝"
