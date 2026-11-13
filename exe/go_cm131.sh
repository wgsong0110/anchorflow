#!/bin/bash
# 절 CM131 묶음 (H5 는 빨리 수렴, H9 는 정체 -- 뉴턴 평가별 노드 잔차 기록, 결과 불변 기록만). 코드 디렉토리에서. 사용: bash exe/go_cm131.sh <커밋>
#   2x2: h9 = R2 extsolve/nb9 (H9 예산) 명령 그대로 · g9 = nb9 + 고스트 강성 켬 (ext2 '--w8k 1' -> '--w8k 0', 확장 그대로)
#        h5 = R2 extsolve/e5 (H5) 명령 그대로 · k5 = e5 + 고스트 강성 끔 (ext2_wrap.py 바로 뒤 '--w8k 1' = fem_wolfh6 (K) P1X 강성 × κ (wolf κ 0), 확장 없음)
#   명령줄 python -u 바로 뒤에 exe/resmode_wrap.py exe/escwhy_wrap.py (기록 두 겹) 만 끼우고 --out 만 바꿈 (50 프레임 그대로), 기록 창 ESC·RM 5~30.
#   카드 둘: 카드 0 = h9 -> k5, 카드 1 = h5 -> g9 (판마다 카드 하나 단독). 판 디렉토리 /home/dkta/work/cm131/<판> -> R2 lbours/wolfcv/cm131/<판> (덤프 포함, 안 지움).
export PATH=/home/dkta/bin:$PATH
H=$1
R=r2:storage/result/anchorflow/lbours/wolfcv/cm131
RX=r2:storage/result/anchorflow/lbours/wolfcv/extsolve
O=/home/dkta/work/cm131
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go_run.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_run.log -q; }
mk() { local n=$1 src=$2 ed=$3 C
  C=$(rclone cat $RX/$src/cmd.txt | tr -d "\n")
  [ ${#C} -gt 100 ] || { say "$src 명령 못 읽음"; return 1; }
  mkdir -p $O/$n
  echo "ESC_F0=5 ESC_F1=30 RM_F0=5 RM_F1=30 $C" | sed -e "s#/bin/python -u exe/#/bin/python -u exe/resmode_wrap.py exe/escwhy_wrap.py exe/#" -e "s#--out [^ ]*\$#--out $O/$n#" | sed -e "${ed:-s#^##}" > $O/$n/cmd.txt
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
  say "$n (원본 $src): $(grep -o 'resmode_wrap.py exe/escwhy_wrap.py exe/[a-z0-9_]*.py\|ext2_wrap.py --w8[a-z] [0-9]* --w8[a-z] [0-9]*\|ext2_wrap.py --[a-z0-9]* [0-9]*\|--g4 [0-9e.-]*\|--frames [0-9]*\|--newton [0-9]*\|--cg [0-9]*\|--out [^ ]*\|ESC_F[01]=[0-9]*\|RM_F[01]=[0-9]*' $O/$n/cmd.txt | tr '\n' ' ')"; }
runone() { local n=$1 g=$2
  say "$n 카드 $g 시작 (계산 프로세스 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  ( while [ ! -f $O/$n/DONE ]; do sleep 90; tail -c 3000 $O/$n/log.txt 2>/dev/null | rclone rcat $R/$n/log_tail.txt; rclone copy $O/$n $R/$n -q --transfers 8 --exclude DONE; done ) &
  env CUDA_VISIBLE_DEVICES=$g timeout 5400 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; touch $O/$n/DONE
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) esc $(ls $O/$n/esc 2>/dev/null | wc -l) rm $(ls $O/$n/rm 2>/dev/null | wc -l) 개 $(cat $O/$n/rminfo.json 2>/dev/null | cut -c1-400) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --transfers 8; say "$n R2 올림 (덤프 포함, 노드 사본 남김)"; }
mk h9 nb9 "" && mk g9 nb9 "s#ext2_wrap.py --w8k 1 --w8e 1#ext2_wrap.py --w8k 0 --w8e 1#" && mk h5 e5 "" && mk k5 e5 "s#ext2_wrap.py --l4#ext2_wrap.py --w8k 1 --l4#" || exit 1
grep -q "ext2_wrap.py --w8k 0 --w8e 1" $O/g9/cmd.txt && grep -q "ext2_wrap.py --w8k 1 --l4" $O/k5/cmd.txt && grep -q "ext2_wrap.py --w8k 1 --w8e 1" $O/h9/cmd.txt && ! grep -q "w8k" $O/h5/cmd.txt || { say "명령 고침 확인 실패"; exit 1; }
( runone h9 0; runone k5 0 ) & P1=$!; sleep 20; ( runone h5 1; runone g9 1 ) & P2=$!; wait $P1 $P2
say "끝 R2 대조 $(rclone check $O $R --one-way -q 2>&1 | grep -c ERROR) 개 차이"
