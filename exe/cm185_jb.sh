#!/bin/bash
source /root/xsenv.sh
export XS_CODE=r2:storage/result/anchorflow/lbours/gfscenes/code_cm185
export W=/root/gfs AF=/root/sv/af R=r2:storage/result/anchorflow; export RX=$R/lbours/gfscenes DS=r2:storage/datasets/anchorflow/gf_scenes GF=/root/gfs/GaussianFluent
export TAG=cm185s_$(hostname | cut -c1-6) LOG=/root/sv/cm185s.log
say() { echo "[$TAG $(date "+%F %T")] $*" | tee -a $LOG; rclone copyto $LOG $RX/$TAG.log -q 2>/dev/null; }
export -f say
rclone cat $XS_CODE/af_code.tar | tar -x -C $AF; rclone copy $XS_CODE/exe $AF/exe -q
export GFS_DROP="--pdiag --qjd --fhash --hgt --bar_cnt"
rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $W/font.otf -q
say "[젤리 b1 시작] $(grep -c box_auto $AF/exe/fem_an_rt_sim.py) GPU 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
XS_SFX=_b1 bash $AF/exe/gfs_jb_ours.sh "--cap_mul=2"
say "[젤리 b1 끝] cauto $(cat $W/ours/jellybullet_b1/run/cauto.json 2>/dev/null)"
