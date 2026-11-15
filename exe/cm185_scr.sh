#!/bin/bash
# 절 CM185 거르기 (지표·영상 없음, ABL0): bash scr.sh 꼬리표 '이름:모델:설정:프레임:추가' ...   (진단 플래그 뺌 = 단독 시간 판)
source /root/xsenv.sh
lab=$1; shift
export XS_CODE=r2:storage/result/anchorflow/lbours/gfscenes/code_cm185
export W=/root/gfs AF=/root/sv/af R=r2:storage/result/anchorflow; export RX=$R/lbours/gfscenes DS=r2:storage/datasets/anchorflow/gf_scenes GF=/root/gfs/GaussianFluent
export TAG=cm185s_$(hostname | cut -c1-6) LOG=/root/sv/cm185s.log
say() { echo "[$TAG $(date "+%F %T")] $*" | tee -a $LOG; rclone copyto $LOG $RX/$TAG.log -q 2>/dev/null; }
export -f say
rclone cat $XS_CODE/af_code.tar | tar -x -C $AF; rclone copy $XS_CODE/exe $AF/exe -q
export GFS_DROP="--pdiag --qjd --fhash --hgt --bar_cnt"
say "[거르기 시작] $lab $(grep -c box_auto $AF/exe/fem_an_rt_sim.py) $(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader) GPU 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) GFS_BAUTO=${GFS_BAUTO:-기본} GFS_CAUTO=${GFS_CAUTO:-기본}"
for e in "$@"; do IFS=: read n m c nf fl <<< "$e"; [ -d $W/hf/model/$m ] || rclone copy $DS/model/$m $W/hf/model/$m --transfers 16 -q; XS_SFX=_$lab bash $AF/exe/gfs_ours.sh $n $m $c $nf "ABL0+$fl"; say "  [$n] cauto $(cat $W/ours/${n}_$lab/run/cauto.json 2>/dev/null)"; done
say "[거르기 끝] $lab"
