#!/bin/bash
# 절 CM130 묶음 (wolf 탄성 에너지 과다 원인 진단, 결과 불변 기록만). 코드 디렉토리에서. 사용: bash exe/go_cm130.sh <커밋>
#   판 e9 = R2 extsolve/nb9 (H9 예산) 명령, ec = extsolve/ncvb (cvb 수렴) 명령 -- 명령줄 python -u 바로 뒤에 exe/escwhy_wrap.py (절 CM126 기록) 한 겹만 끼우고
#   --frames 와 --out 만 바꿈 (e9 20 · ec 30 프레임), ESC_F0~F1 = 5~20 (e9) · 5~30 (ec). 카드 둘에 동시. 판 디렉토리 /home/dkta/work/cm130/<판> -> R2 lbours/wolfcv/cm130/<판> (덤프 포함, 안 지움).
export PATH=/home/dkta/bin:$PATH
H=$1
R=r2:storage/result/anchorflow/lbours/wolfcv/cm130
RX=r2:storage/result/anchorflow/lbours/wolfcv/extsolve
O=/home/dkta/work/cm130
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go_run.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_run.log -q; }
mk() { local n=$1 src=$2 fr=$3 f0=$4 f1=$5 C
  C=$(rclone cat $RX/$src/cmd.txt | tr -d "\n")
  [ ${#C} -gt 100 ] || { say "$src 명령 못 읽음"; return 1; }
  mkdir -p $O/$n
  echo "ESC_F0=$f0 ESC_F1=$f1 $C" | sed -e "s#/bin/python -u exe/#/bin/python -u exe/escwhy_wrap.py exe/#" -e "s#--frames [0-9]*#--frames $fr#" -e "s#--out [^ ]*\$#--out $O/$n#" > $O/$n/cmd.txt
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
  say "$n (원본 $src): $(grep -o 'escwhy_wrap.py exe/[a-z0-9_]*.py\|--frames [0-9]*\|--newton [0-9]*\|--cg [0-9]*\|--out [^ ]*\|ESC_F[01]=[0-9]*' $O/$n/cmd.txt | tr '\n' ' ')"; }
runone() { local n=$1 g=$2
  say "$n 카드 $g 시작 (계산 프로세스 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  ( while [ ! -f $O/$n/DONE ]; do sleep 90; tail -c 3000 $O/$n/log.txt 2>/dev/null | rclone rcat $R/$n/log_tail.txt; done ) &
  env CUDA_VISIBLE_DEVICES=$g timeout 5400 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; touch $O/$n/DONE
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) esc $(ls $O/$n/esc 2>/dev/null | wc -l) 개 $(cat $O/$n/escinfo.json 2>/dev/null | cut -c1-300) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --transfers 8; say "$n R2 올림 (덤프 포함, 노드 사본 남김)"; }
mk e9 nb9 20 5 20 && mk ec ncvb 30 5 30 || exit 1
runone e9 0 & P1=$!; sleep 20; runone ec 1 & P2=$!; wait $P1 $P2
say "끝 R2 대조 $(rclone check $O $R --one-way -q 2>&1 | grep -c ERROR) 개 차이"
