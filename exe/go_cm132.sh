#!/bin/bash
# 절 CM132 묶음 (확장 T1E2 외부 노드 정의 고침, 예산 안 H9 + 수정, wolf 50 프레임). 코드 디렉토리에서. 사용: bash exe/go_cm132.sh <커밋> [카드0 판들] [카드1 판들]
#   원본 = R2 extsolve/nb9 (H9 예산) 명령 그대로, python -u 바로 뒤 exe/extfix_wrap.py <플래그> 를 ext2_wrap.py 앞에 끼우고 --out 만 바꿈.
#   판: b9 = 원본 그대로 (같은 카드 프레임 시간 기준) · fx = --xfl 1 (다) · la = --xfl 1 --xls 1 (다 + 가 순수 사영) · lb = --xfl 1 --xls 2 (다 + 가 아핀 재현 제약 사영)
#   판 디렉토리 /home/dkta/work/cm132/<판> -> R2 lbours/wolfcv/cm132/<판> (덤프 포함, 안 지움).
export PATH=/home/dkta/bin:$PATH
H=$1; G0=${2:-fx,lb}; G1=${3:-la,b9}
R=r2:storage/result/anchorflow/lbours/wolfcv/cm132
RX=r2:storage/result/anchorflow/lbours/wolfcv/extsolve
O=/home/dkta/work/cm132
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go_run.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_run.log -q; }
flags() { case $1 in b9) echo "";; fx) echo "--xfl 1";; la) echo "--xfl 1 --xls 1";; lb) echo "--xfl 1 --xls 2";; fl) echo "--xls 2";; *) echo "?";; esac; }
mk() { local n=$1 F C; F=$(flags $n); [ "$F" = "?" ] && { say "$n 모르는 판"; return 1; }
  C=$(rclone cat $RX/nb9/cmd.txt | tr -d "\n")
  [ ${#C} -gt 100 ] || { say "nb9 명령 못 읽음"; return 1; }
  mkdir -p $O/$n
  if [ -n "$F" ]; then C=$(echo "$C" | sed -e "s#exe/speed12_wrap.py --s12z 1 exe/ext2_wrap.py#exe/speed12_wrap.py --s12z 1 exe/extfix_wrap.py $F exe/ext2_wrap.py#"); fi
  echo "$C" | sed -e "s#--out [^ ]*\$#--out $O/$n#" > $O/$n/cmd.txt
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
  if [ -n "$F" ]; then grep -q "exe/extfix_wrap.py $F exe/ext2_wrap.py --w8k 1 --w8e 1 --w8d 2 --w8t 1" $O/$n/cmd.txt || { say "$n 명령 고침 확인 실패"; return 1; }; fi
  grep -q -- "--out $O/$n\$" $O/$n/cmd.txt || { say "$n --out 확인 실패"; return 1; }
  say "$n 명령: $(grep -o 'extfix_wrap.py[^e]*exe/ext2_wrap.py\|--frames [0-9]*\|--tb [0-9.]*\|--out [^ ]*' $O/$n/cmd.txt | tr '\n' ' ')"; }
runone() { local n=$1 g=$2
  say "$n 카드 $g 시작 (계산 프로세스 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  ( while [ ! -f $O/$n/DONE ]; do sleep 90; tail -c 3000 $O/$n/log.txt 2>/dev/null | rclone rcat $R/$n/log_tail.txt; rclone copy $O/$n $R/$n -q --transfers 8 --exclude DONE; done ) &
  env CUDA_VISIBLE_DEVICES=$g timeout 5400 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; touch $O/$n/DONE
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(cat $O/$n/xfinfo.json 2>/dev/null | cut -c1-300) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --transfers 8; say "$n R2 올림 (덤프 포함)"; }
for n in ${G0//,/ } ${G1//,/ }; do mk $n || exit 1; done
( for n in ${G0//,/ }; do runone $n 0; done ) & P1=$!; sleep 20; ( for n in ${G1//,/ }; do runone $n 1; done ) & P2=$!; wait $P1 $P2
say "끝 R2 대조 $(rclone check $O $R --one-way -q 2>&1 | grep -c ERROR) 개 차이"
