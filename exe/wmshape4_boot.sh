#!/bin/bash
# 절 CM46 묶음 wmshape4 잡 안 시작점 (/home/dkta/work/wmshape4_boot.sh 로 두고 exe/job_run.sh AF_DETACH=1 로 부름; 인자 없음).
# 커밋·단계 = R2 lbours/wmshape4/code/launch.txt ('<커밋> <단계>'), 코드 = R2 git bundle -> 노드 /tmp -> exe/go_wmshape4.sh.
export PATH=/home/dkta/bin:$PATH
R=r2:storage/result/anchorflow/lbours/wmshape4
read -r H STG <<< "$(rclone cat $R/code/launch.txt)"
CB=/tmp/dkta/wmshape4_code; mkdir -p $CB
[ -s $CB/wmshape4_$H.bundle ] || rclone copyto $R/code/wmshape4_$H.bundle $CB/wmshape4_$H.bundle -q
[ -d $CB/$H/.git ] || git clone -q -b k1new $CB/wmshape4_$H.bundle $CB/$H
cd $CB/$H && git checkout -q $H && exec bash exe/go_wmshape4.sh $H $STG
