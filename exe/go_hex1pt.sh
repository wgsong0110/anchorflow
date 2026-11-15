#!/bin/bash
# 절 CM179: 육면체 셀 가우스 적분 2x2x2 (기본) 대 1x1x1 실행 + 3DGS 렌더 + 모래시계 모드 분석. 코드 디렉토리에서.
# 사용: bash exe/go_hex1pt.sh <커밋> <카드> <판 목록 쉼표 (예 L2,L1,F2,F1)>   판 이름 = 장면 글자 (L 레고 · F ficus · B bread) + 적분 (2 = 2x2x2 기본, 1 = 1x1x1).
# 판마다: 실행 (--dump --dump_F --dump_part) -> exe/lb_roll_to_h5.py -> exe/lb_render.py (fem_<장면>.mp4) -> exe/q27_eig.py (영 에너지 모드, 처음 프레임·접촉 프레임) -> R2 lbours/hex1pt/<판>.
export PATH=/home/dkta/bin:$PATH
H=$1; G=$2; RUNS=$3
R=r2:storage/result/anchorflow/lbours/hex1pt
O=/home/dkta/work/hex1pt
W=/home/dkta/work; AG=/dev/shm/dkta/ag_data
PY=/home/dkta/.conda/envs/af/bin/python
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go_run_$G.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_run_$G.log -q; }
for s in L F B; do [ -s $O/src$s.txt ] || rclone cat r2:storage/result/anchorflow/lbours/wolfcv/cm157/${s}A_1/cmd.txt > $O/src$s.txt; [ $(wc -c < $O/src$s.txt) -gt 100 ] || { say "$s 원본 명령 못 읽음"; exit 1; }; done
runone() { local n=$1 s=${1:0:1} m=${1:1:1} GT CFG SH Q
  case $s in L) GT=$W/lbdata/test_low/L000; CFG=$W/lbdata/config.json; SH=lego;; F) GT=$W/lbours/ficus_v0/gt; CFG=$GT/config.json; SH=ficus;; B) GT=$AG/gt_bread; CFG=$GT/config.json; SH=bread;; esac
  [ $m = 1 ] && Q=1 || Q=8
  mkdir -p $O/$n; python3 exe/hex1pt_gen.py $O/src$s.txt $s $m $O/$n > $O/$n/cmd.txt || { say "$n 명령 생성 실패"; return 1; }
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
  say "$n 시작 카드 $G (계산 프로세스 $(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -c .)) q1pt $(grep -c -- '--q1pt 1' $O/$n/cmd.txt) 프레임 $(grep -o -- '--frames [0-9]*' $O/$n/cmd.txt)"
  env CUDA_VISIBLE_DEVICES=$G timeout 5400 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ') pred $(stat -c %s $O/$n/pred.pt 2>&1)"
  if [ -f $O/$n/pred.pt ]; then
    local HD=/dev/shm/dkta/hex1pt_h5/$n; mkdir -p $HD
    $PY -u exe/lb_roll_to_h5.py --dump $O/$n/pred.pt --traj $GT --out $HD/h5 > $O/$n/h5.log 2>&1
    CUDA_VISIBLE_DEVICES=$G $PY -u exe/lb_render.py --sim $HD/h5 --config $CFG --shape $SH --out $O/$n/fem_$SH.mp4 --frame_label 0 --caption " " < /dev/null > $O/$n/r.log 2>&1
    say "$n 영상 $(stat -c %s $O/$n/fem_$SH.mp4 2>&1) $(tail -1 $O/$n/r.log | cut -c1-160)"
    rm -rf $HD
    [ -f $O/$n/q27solver.pt ] && CUDA_VISIBLE_DEVICES=$G timeout 1500 $PY -u exe/q27_eig.py --pred $O/$n/pred.pt --solver $O/$n/q27solver.pt --frames 0,c --q $Q --kb $([ $s = B ] && echo 800 || echo 2400) --out $O/$n/eig.json > $O/$n/eig.log 2>&1
    say "$n eig rc=$? $(tail -2 $O/$n/eig.log | cut -c1-300 | tr '\n' ' ')"
  fi
  rclone copy $O/$n $R/$n -q --transfers 8; say "$n R2 올림"; return 0; }
for n in ${RUNS//,/ }; do runone $n; done
say "끝 카드 $G"
