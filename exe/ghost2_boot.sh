#!/bin/bash
# 절 CM48 묶음 ghost2 잡 안 시작점 (/home/dkta/work/ghost2_boot.sh 로 두고 exe/job_run.sh AF_DETACH=1 로 부름; 인자 없음).
# 커밋·단계 = R2 lbours/ghost2/code/launch.txt ('<커밋> <단계>'), 코드 = R2 git bundle -> 노드 /tmp -> exe/go_ghost2.sh.
export PATH=/home/dkta/bin:$PATH
R=r2:storage/result/anchorflow/lbours/ghost2
read -r H STG <<< "$(rclone cat $R/code/launch.txt)"
CB=/tmp/dkta/ghost2_code; mkdir -p $CB
[ -s $CB/ghost2_$H.bundle ] || rclone copyto $R/code/ghost2_$H.bundle $CB/ghost2_$H.bundle -q
[ -d $CB/$H/.git ] || git clone -q -b k1new $CB/ghost2_$H.bundle $CB/$H
cd $CB/$H && git checkout -q $H && exec bash exe/go_ghost2.sh $H $STG
