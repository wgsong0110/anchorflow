#!/bin/bash
# 절 CM93 영상 묶음 wolfcv: R2 lbours/wolfh12 의 덤프 (수렴 판 aC_1 = 뉴턴 20·PCG 40 f27 발산, H8 h8_1) 를 3DGS 로 렌더만 한다 (재계산 없음; exe/stab_wolfh9.sh vid() 와 같은 절차).
# 덤프는 /home/dkta/work/wolfconv, h5 는 /dev/shm/dkta (렌더 뒤 지움). 카드는 공용 표식 디렉토리로 계산 프로세스 0 인 카드만. 잡 안, 코드 디렉토리에서.
# 사용: bash exe/stab_wolfcv.sh <출력> <R2> <단계>   (단계 v: aC_1 h8_1)
O=$1; RO=$2; STG=$3; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR
W=/home/dkta/work; CD=$(pwd -P); SRC=r2:storage/result/anchorflow/lbours/wolfh12
DD=$W/wolfconv; mkdir -p $DD
AG=/dev/shm/dkta/ag_data; mkdir -p $AG
GTW=$AG/gt_wolf
log() { echo "[$(date '+%F %T')] $*" >> $O/wolfcv_$STG.log; rclone copyto $O/wolfcv_$STG.log $RO/wolfcv_$STG.log -q; }
if [ ! -f $GTW/meta.json ] || [ ! -f $GTW/sim_0000000000.h5 ]; then
  GTW=/dev/shm/dkta/wolfcv_gt; mkdir -p $GTW
  CUDA_VISIBLE_DEVICES="" $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt $GTW/tmp.pt --out_dir $GTW --out_dir_n 1 > $O/mkgt.log 2>&1; rm -f $GTW/tmp.pt
fi
log "시작 $(hostname) 코드 $CD 정답 궤적 $GTW $(ls $GTW | head -3 | tr '\n' ' ')"
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL
NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
claim_wait() { local g; while :; do g=$(claim $1) && { echo $g; return; }; sleep 30; done; }
one() { local n=$1 g=$2 Od=$O/$1 V=$O/$1/vj HD=/dev/shm/dkta/wolfcv_vid/$1
  mkdir -p $V $HD $DD/$n
  [ -s $DD/$n/pred.pt ] || rclone copyto $SRC/$n/pred.pt $DD/$n/pred.pt -q
  log "$n 덤프 $(stat -c %s $DD/$n/pred.pt) 카드 $g"
  $PY -u exe/lb_roll_to_h5.py --dump $DD/$n/pred.pt --traj $GTW --out $HD/h5 > $V/h5.log 2>&1
  CUDA_VISIBLE_DEVICES=$g $PY -u exe/lb_render.py --sim $HD/h5 --config $GTW/config.json --shape wolf --out $V/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $V/r.log 2>&1
  log "$n 영상 $(stat -c %s $V/fem_wolf.mp4 2>&1) $(tail -1 $V/r.log | cut -c1-160) $(tail -1 $V/h5.log | cut -c1-160)"
  rm -rf $HD
  rclone copy $Od $RO/$n -q; }
for n in aC_1 h8_1; do
  ( me=$BASHPID; g=$(claim_wait $me); one $n $g; rm -f $CL/$g ) & sleep 3
done
wait
log "끝"
