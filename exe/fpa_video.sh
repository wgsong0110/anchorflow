#!/bin/bash
# 절 CM190 대표 비교 영상 렌더 (인스턴스, GPU 한 장): 레고 정답 + 기본 판 (가우스 2x2x2) + 가우스 1x1x1 판 3DGS 렌더 mp4 -> R2 lbours/final_pa/vid/L000/
# 사용: bash exe/fpa_video.sh <카드>   (코드 디렉토리에서; pred.pt = R2 lbours/final_pa/runs/L000_{def,q1}_r1/pred.pt)
export PATH=/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
G=${1:-0}; PY=/home/dkta/.conda/envs/af/bin/python; W=/home/dkta/work
RA=r2:storage/result/anchorflow/lbours/final_pa; O=/dev/shm/dkta/fpavid; mkdir -p $O /dev/shm/dkta/tmp; export TMPDIR=/dev/shm/dkta/tmp
GT=$W/lbdata/test_low/L000; CFG=$W/lbdata/config.json
for v in def q1; do
  rclone copyto $RA/runs/L000_${v}_r1/pred.pt $O/$v.pt -q || { echo "$v pred 없음"; continue; }
  $PY -u exe/lb_roll_to_h5.py --dump $O/$v.pt --traj $GT --out $O/h5_$v > $O/h5_$v.log 2>&1
  CUDA_VISIBLE_DEVICES=$G $PY -u exe/lb_render.py --sim $O/h5_$v --config $CFG --shape lego --out $O/$v.mp4 --frame_label 0 --caption " " < /dev/null > $O/r_$v.log 2>&1
  echo "$v $(stat -c %s $O/$v.mp4 2>&1)"; rm -rf $O/h5_$v $O/$v.pt
done
CUDA_VISIBLE_DEVICES=$G $PY -u exe/lb_render.py --sim $GT --config $CFG --shape lego --out $O/gt.mp4 --frame_label 0 --caption " " < /dev/null > $O/r_gt.log 2>&1
echo "gt $(stat -c %s $O/gt.mp4 2>&1)"
rclone copy $O $RA/vid/L000 -q --include "*.mp4" --include "*.log"
