#!/bin/bash
# 절 CM183 A 비교 영상용 렌더 (인스턴스에서, GPU 한 장): 장면 하나의 정답 + 판 (def q1 rho0 rho1) 3DGS 렌더 mp4 -> R2 lbours/ablation/vid/<장면>/
# 사용: bash exe/abl_video.sh <ficus|bread> <카드>   (코드 디렉토리에서; pred.pt 는 R2 lbours/ablation/runs/<장면>_<판>_r1/pred.pt)
export PATH=/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
S=$1; G=$2; PY=/home/dkta/.conda/envs/af/bin/python; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data
RA=r2:storage/result/anchorflow/lbours/ablation; O=/dev/shm/dkta/ablvid_$S; mkdir -p $O /dev/shm/dkta/tmp; export TMPDIR=/dev/shm/dkta/tmp
case $S in ficus) GT=$W/lbours/ficus_v0/gt; CFG=$GT/config.json;; bread) GT=$AG/gt_bread; CFG=$GT/config.json;; esac
for v in def q1 rho0 rho1; do
  SRC=${S}_${v}_r1; [ $v = def ] && SRC=${S}_defb_r1; rclone copyto $RA/runs/$SRC/pred.pt $O/$v.pt -q || { echo "$v pred 없음"; continue; }
  $PY -u exe/lb_roll_to_h5.py --dump $O/$v.pt --traj $GT --out $O/h5_$v > $O/h5_$v.log 2>&1
  CUDA_VISIBLE_DEVICES=$G $PY -u exe/lb_render.py --sim $O/h5_$v --config $CFG --shape $S --out $O/$v.mp4 --frame_label 0 --caption " " < /dev/null > $O/r_$v.log 2>&1
  echo "$v $(stat -c %s $O/$v.mp4 2>&1)"; rm -rf $O/h5_$v $O/$v.pt
done
# 정답: 정답 궤적 폴더의 h5 를 그대로 (sim_*.h5)
CUDA_VISIBLE_DEVICES=$G $PY -u exe/lb_render.py --sim $GT --config $CFG --shape $S --out $O/gt.mp4 --frame_label 0 --caption " " < /dev/null > $O/r_gt.log 2>&1
echo "gt $(stat -c %s $O/gt.mp4 2>&1)"
rclone copy $O $RA/vid/$S -q --include "*.mp4" --include "*.log"
