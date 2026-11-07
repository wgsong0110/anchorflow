#!/bin/bash
# 절 CM25 영상 보충: 본 묶음 덤프에 F 가 없어 (--dump_F 빠짐) 렌더 실패 -> base·A_stick 를 같은 명령 + --dump_F 로 다시 (결과 불변: 끝 해시 대조), 렌더 (exe/lb_render.py --shape wolf), vj 만 R2.
O=$1; RO=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
W=/home/dkta/work
source <(sed -n '/^WD=/,/^  esac; }/p' exe/stab_wolfwhy.sh)
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
RG=$(claim_wait $$); log "카드 $RG"
AGD=/tmp/dkta/ag_data; GTD=$AGD/gt_wolf; [ -f $GTD/meta.json ] || GTD=/dev/shm/dkta/ag_data/gt_wolf
if [ ! -f $GTD/meta.json ]; then mkdir -p /dev/shm/dkta/ag_data
  CUDA_VISIBLE_DEVICES=$RG $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /dev/shm/dkta/ag_data/wolf_rend_tmp.pt --out_dir $GTD > /dev/shm/dkta/ag_data/cv_wolf_rend.log 2>&1
  rm -f /dev/shm/dkta/ag_data/wolf_rend_tmp.pt; fi
for n in base A_stick; do
  Od=$O/v_$n; mkdir -p $Od
  c="exe/fem_an_rt_sim.py $(cmdof $n) --out $Od"; echo "$PY -u $c" > $Od/cmd.txt
  CUDA_VISIBLE_DEVICES=$RG timeout 3600 $PY -u $c < /dev/null > $Od/log.txt 2>&1
  log "$n 덤프 rc=$? 해시 $(grep -o '"x_sha1": "[0-9a-f]*"' $Od/summary.json 2>/dev/null)"
  $PY -u exe/lb_roll_to_h5.py --dump $Od/pred.pt --traj $GTD --out $Od/h5 > $Od/h5.log 2>&1
  mkdir -p $Od/vj
  CUDA_VISIBLE_DEVICES=$RG $PY -u exe/lb_render.py --sim $Od/h5 --config $GTD/config.json --shape wolf --out $Od/vj/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $Od/vj/r.log 2>&1
  log "[렌더] $n $(ls -la $Od/vj/fem_wolf.mp4 2>&1 | cut -c1-160) $(tail -1 $Od/vj/r.log | cut -c1-200) $(tail -1 $Od/h5.log | cut -c1-200)"
  rclone copy $Od $RO/v_$n -q --exclude pred.pt --exclude "h5/**"
done
rm -f $CL/$RG
log "VID_END"
