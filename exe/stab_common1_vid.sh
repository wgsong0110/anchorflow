#!/bin/bash
# 절 CM31 영상: 레고·wolf 기본 판·후보 A2k 를 결정적 + 덤프 (--dump --dump_F; 결과 불변) 로 다시 돌려 3DGS 렌더 (exe/lb_render.py, 절 CM8a·CM27 영상과 같은 절차). 잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2.
O=$1; RO=$2
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
W=/home/dkta/work
source exe/fem_cm2_scn.sh
for t in L000 wolf; do cm_data $t; done
BA0="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8"
FBB="--floor vel --fl_bar --fl_bst 1 --y0eb"; TL="--tolN 1e-2 --tolP 1e-1"
CAND="--y0loc 3 --ue be --uf_res --ue_loc 2 --ls0_keep"
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
GTW=/tmp/dkta/ag_data/gt_wolf; [ -f $GTW/meta.json ] || GTW=/dev/shm/dkta/ag_data/gt_wolf
one() { local n=$1 sc=$2 x=$3 g=$4 Od=$O/vid/$1 c GTD SH CF
  mkdir -p $Od/vj
  if [ $sc = L ]; then c="exe/fem_an_rt_sim.py --data $SD/L000.pt --frames 80 $FBB $BA0 --newton 2 --cg 4 $TL --fp_interp --tag L000 --K 1"; GTD=$W/lbdata/test_low/L000; SH=lego; CF=$W/lbdata/config.json
  else c="exe/fem_an_rt_sim.py --data $AG/wolf.pt --frames 50 $FBB --pbc --mlow 1e-3 $BA0 --cell 2 --newton 2 --cg 5 $TL --fp_interp --tag wolf --K 1"; GTD=$GTW; SH=wolf; CF=$GTW/config.json; fi
  c="$c --det --fhash --dump --dump_F $x --out $Od"; echo "$PY -u $c" > $Od/cmd.txt
  CUDA_VISIBLE_DEVICES=$g timeout 3600 $PY -u $c < /dev/null > $Od/log.txt 2>&1
  $PY -u exe/lb_roll_to_h5.py --dump $Od/pred.pt --traj $GTD --out $Od/h5 > $Od/h5.log 2>&1
  CUDA_VISIBLE_DEVICES=$g $PY -u exe/lb_render.py --sim $Od/h5 --config $CF --shape $SH --out $Od/vj/fem_$SH.mp4 --frame_label 0 --caption " " < /dev/null > $Od/vj/r.log 2>&1
  log "[렌더] $n $(grep -ah '롤아웃\]' $Od/log.txt | tail -1 | cut -c1-160) $(ls -la $Od/vj/fem_$SH.mp4 2>&1 | cut -c1-120) $(tail -1 $Od/vj/r.log | cut -c1-160)"
  rclone copy $Od $RO/vid/$n -q --exclude "h5/**" --exclude pred.pt; }
for spec in "L_off L -" "L_A2k L $CAND" "W_off W -" "W_A2k W $CAND"; do set -- $spec; n=$1; sc=$2; shift 2; x="$*"; [ "$x" = - ] && x=""
  ( g=$(claim_wait $BASHPID); one $n $sc "$x" $g; rm -f $CL/$g ) & sleep 3; done
wait
log "VID_END"
