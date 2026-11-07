#!/bin/bash
# 절 CM24 영상: 수박 판정 판 덤프 (R2 lbours/wmk1fix/<판>/pred.pt) 를 3DGS 일반 색으로 렌더 (절 CM23.3 stab_wmk1vid.sh 와 같은 경로:
#   유한 프레임 뒤 마지막 프레임으로 100 까지 채움 -> lb_roll_to_h5 -> wm_render). 정답 칸은 R2 lbours/k1new/vid/gt_plain.mp4 재사용 (같은 렌더러·카메라), 합성은 로컬.
# 잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2, 판 이름들 = 나머지 인자. 판마다 카드 하나, 동시에.
O=$1; R=$2; shift 2; RUNS="$*"; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
V=$O/vid; mkdir -p $V
log() { echo "[$(date '+%F %T')] $*" >> $O/vid.log; rclone copyto $O/vid.log $R/vid.log -q; }
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
W=/home/dkta/work; CD=$(pwd -P)
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon; CFG=$W/wm2/gf/config.json; GTH=$W/wm2/gf/simulation_ply
GTD=/tmp/dkta/ag_data/gt_wm; [ -f $GTD/meta.json ] || GTD=/dev/shm/dkta/ag_data/gt_wm
if [ ! -f $GTD/meta.json ]; then
  g=$(claim_wait $$); mkdir -p /dev/shm/dkta/ag_data
  CUDA_VISIBLE_DEVICES=$g $PY -u exe/pg_h5_to_pt.py --h5_dir $GTH --config $CFG --out_pt /dev/shm/dkta/ag_data/wm_rend_tmp.pt --out_dir $GTD --out_dir_n 1 --allow_nonfinite > /dev/shm/dkta/ag_data/cv_wm_rend.log 2>&1
  rm -f /dev/shm/dkta/ag_data/wm_rend_tmp.pt $CL/$g; GTD=/dev/shm/dkta/ag_data/gt_wm; fi
log "정답 traj $GTD, 판 $RUNS"
one() { local n=$1 me=$BASHPID g D=$O/vtmp_$1
  [ -s $V/$n.mp4 ] && return
  g=$(claim_wait $me); log "$n 카드 $g"
  mkdir -p $D; rclone copyto $R/$n/pred.pt $D/pred.pt -q
  $PY - $D/pred.pt $D/pred_v.pt <<'PYX' > $V/pad_$n.log 2>&1
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False); P, F = d["pred"], d["F"]
n = 0
while n < P.shape[0] and bool(torch.isfinite(P[n]).all()) and bool(torch.isfinite(F[n]).all()): n += 1
L = 100
P, F = P[:n], F[:n]
if n < L:
    P = torch.cat([P, P[n - 1:n].expand(L - n, -1, -1)]); F = torch.cat([F, F[n - 1:n].expand(L - n, -1, -1, -1)])
torch.save({"pred": P.contiguous(), "F": F.contiguous()}, sys.argv[2]); print("덤프 프레임", d["pred"].shape[0], "유한 프레임", n, "->", P.shape[0])
PYX
  rm -f $D/pred.pt
  $PY -u exe/lb_roll_to_h5.py --dump $D/pred_v.pt --traj $GTD --out $D/h5_fem > $V/h5_$n.log 2>&1; rm -f $D/pred_v.pt
  (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$g $PY -u $CD/exe/wm_render.py --model_path $WMM --config $CFG --frames 100 --sim $D/h5_fem --out $V/$n.mp4 < /dev/null) > $V/r_$n.log 2>&1
  log "$n 렌더 $(tail -1 $V/pad_$n.log) $(tail -1 $V/h5_$n.log | cut -c1-150) $(tail -1 $V/r_$n.log | cut -c1-150) $(ls -la $V/$n.mp4 2>&1 | awk '{print $5}')"
  rm -rf $D; rm -f $CL/$g
  rclone copy $V $R/vid -q; }
for n in $RUNS; do one $n & sleep 3; done
wait
log "단계 끝 vid"
