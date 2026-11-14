#!/bin/bash
# 절 CM157: wolf 용으로 바꾼 설정 (확장 --w8e/d/t 와 S_구속·C --w5s/--w5c 끔 + --mlow 1e-20) 이 다른 장면 (레고·ficus·bread) 에서 결과를 바꾸는가.
# 코드 디렉토리에서. 사용: bash exe/go_cm157.sh <커밋> <카드> <판 목록 쉼표 (예 LA_1,LB_1,...)>   판 이름 = 장면 글자 (L 레고 · F ficus · B bread) + 판 (A = H9, B = H9 − 확장 − S_구속·C + mlow 1e-20) + _반복.
# 한 카드에서 차례로 (단독 시간). 판 디렉토리 /home/dkta/work/cm157/<판> -> R2 lbours/wolfcv/cm157/<판>.
export PATH=/home/dkta/bin:$PATH
H=$1; G=$2; RUNS=$3
R=r2:storage/result/anchorflow/lbours/wolfcv/cm157
O=/home/dkta/work/cm157
PY=/home/dkta/.conda/envs/af/bin/python
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go_run.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_run.log -q; }
for s in L F B; do rclone cat r2:storage/result/anchorflow/lbours/ext2/oA${s}_1/cmd.txt > $O/src$s.txt; [ $(wc -c < $O/src$s.txt) -gt 100 ] || { say "$s 원본 명령 못 읽음"; exit 1; }; done
runone() { local n=$1 s=${1:0:1} k=${1:1:1}
  mkdir -p $O/$n; python3 exe/cm157_gen.py $O/src$s.txt $s $k $O/$n > $O/$n/cmd.txt || { say "$n 명령 생성 실패"; return 1; }
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
  say "$n 시작 카드 $G (계산 프로세스 $(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -c .)) 확장 $(grep -c -- '--w8e 1' $O/$n/cmd.txt) S_구속 $(grep -c -- '--w5s 1' $O/$n/cmd.txt) mlow $(grep -o -- '--mlow [^ ]*' $O/$n/cmd.txt | head -1)"
  env CUDA_VISIBLE_DEVICES=$G timeout 3600 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --transfers 8; say "$n R2 올림"; return 0; }
for n in ${RUNS//,/ }; do runone $n; done
ALL=$(ls $O | grep -E '^[LFB][AB]_[0-9]+$' | tr '\n' ' ')
$PY -u exe/cm157_an.py $O $ALL > $O/an_$(date +%H%M).md 2>&1; say "an rc=$?"
rclone copy $O $R --include "an_*.md" -q
say "끝 R2 대조 $(rclone check $O $R --one-way -q 2>&1 | grep -c ERROR) 개 차이"
