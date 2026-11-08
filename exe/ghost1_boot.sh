#!/bin/bash
# 절 CM47 묶음 ghost1 잡 안 시작점 (/home/dkta/work/ghost1_boot.sh 로 두고 exe/job_run.sh AF_DETACH=1 로 부름; 인자 없음).
# 커밋·단계 = R2 lbours/ghost1/code/launch.txt ('<커밋> <단계>'), 코드 = R2 git bundle -> 노드 /tmp -> exe/go_ghost1.sh.
export PATH=/home/dkta/bin:$PATH
R=r2:storage/result/anchorflow/lbours/ghost1
read -r H STG <<< "$(rclone cat $R/code/launch.txt)"
CB=/tmp/dkta/ghost1_code; mkdir -p $CB
[ -s $CB/ghost1_$H.bundle ] || rclone copyto $R/code/ghost1_$H.bundle $CB/ghost1_$H.bundle -q
[ -d $CB/$H/.git ] || git clone -q -b k1new $CB/ghost1_$H.bundle $CB/$H
cd $CB/$H && git checkout -q $H && exec bash exe/go_ghost1.sh $H $STG
