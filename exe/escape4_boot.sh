#!/bin/bash
# 절 CM49 묶음 escape4 잡 안 시작점 (/home/dkta/work/escape4_boot.sh 로 두고 exe/job_run.sh AF_DETACH=1 로 부름; 인자 없음).
# 커밋·단계 = R2 lbours/escape4/code/launch.txt ('<커밋> <단계>'), 코드 = R2 git bundle -> 노드 /tmp -> exe/go_escape4.sh.
export PATH=/home/dkta/bin:$PATH
R=r2:storage/result/anchorflow/lbours/escape4
read -r H STG <<< "$(rclone cat $R/code/launch.txt)"
CB=/tmp/dkta/escape4_code; mkdir -p $CB
[ -s $CB/escape4_$H.bundle ] || rclone copyto $R/code/escape4_$H.bundle $CB/escape4_$H.bundle -q
[ -d $CB/$H/.git ] || git clone -q -b k1new $CB/escape4_$H.bundle $CB/$H
cd $CB/$H && git checkout -q $H && exec bash exe/go_escape4.sh $H $STG
