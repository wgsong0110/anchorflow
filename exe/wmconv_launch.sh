#!/bin/bash
# 절 CM90 묶음 wmconv 띄우기 (잡 133077 안에서 pbs_attach 로, 상주 시작점 없이 한 번): R2 lbours/wmconv/code/launch.txt 의 커밋 tar 를 /dev/shm 에 풀고 exe/go_wmconv.sh
export PATH=/home/dkta/bin:$PATH
R=r2:storage/result/anchorflow/lbours/wmconv
mkdir -p /dev/shm/dkta
H=$(rclone cat $R/code/launch.txt | head -1)
CB=/dev/shm/dkta/wmconv_code/$H; mkdir -p $CB
echo "[launch $(date '+%F %T')] $(hostname) 커밋 $H" | rclone rcat $R/launch.log
rclone cat $R/code/wmconv_$H.tar | tar -x -C $CB && cd $CB && bash exe/go_wmconv.sh $H < /dev/null > /dev/shm/dkta/wmconv_go.out 2>&1
