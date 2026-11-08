#!/bin/bash
# 절 CM66 영상 (exe/stab_wolfh2_vid.sh 복사, 노드 /tmp 안 읽음·안 씀): 판정 판 덤프 (pred.pt, --dump_F) 를 3DGS 렌더 (exe/lb_render.py --shape wolf; 절 CM25 영상과 같은 절차). 잡 안, 커밋 디렉토리에서; 묶음 출력 $1, R2 $2, 판 이름 = 나머지 인자.
O=$1; RO=$2; shift 2; RUNS="$*"
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
W=/home/dkta/work
log() { echo "[$(date '+%F %T')] $*" >> $O/vid.log; rclone copyto $O/vid.log $RO/vid.log -q; }
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
RG=$(claim_wait $$); log "카드 $RG 판 $RUNS"
GTD=/dev/shm/dkta/ag_data/gt_wolf
if [ ! -f $GTD/meta.json ]; then mkdir -p /dev/shm/dkta/ag_data
  CUDA_VISIBLE_DEVICES=$RG $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /dev/shm/dkta/ag_data/wolf_rend_tmp.pt --out_dir $GTD > /dev/shm/dkta/ag_data/cv_wolf_rend.log 2>&1
  rm -f /dev/shm/dkta/ag_data/wolf_rend_tmp.pt; fi
for n in $RUNS; do
  Od=$O/vid/$n; mkdir -p $Od/vj
  [ -s $O/$n/pred.pt ] || rclone copyto $RO/$n/pred.pt $O/$n/pred.pt -q
  $PY -u exe/lb_roll_to_h5.py --dump $O/$n/pred.pt --traj $GTD --out $Od/h5 > $Od/h5.log 2>&1
  CUDA_VISIBLE_DEVICES=$RG $PY -u exe/lb_render.py --sim $Od/h5 --config $GTD/config.json --shape wolf --out $Od/vj/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $Od/vj/r.log 2>&1
  log "[렌더] $n $(ls -la $Od/vj/fem_wolf.mp4 2>&1 | cut -c1-160) $(tail -1 $Od/vj/r.log | cut -c1-200) $(tail -1 $Od/h5.log | cut -c1-200)"
  cp $O/$n/summary.json $Od/summary.json 2>/dev/null
  rclone copy $Od $RO/vid/$n -q --exclude "h5/**"
done
rm -f $CL/$RG
log "VID_END"
