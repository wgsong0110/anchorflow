#!/bin/bash
# 절 AN.err 잡 안 진입점 (job_run.sh 가 부른다): /home/dkta/work/imp_boot4.sh 로 R2 lbours/fem_an/code/imp_<C>.tar.gz 를 노드 /tmp/dkta/impwt/<C> 에 풀고 스크립트를 돈다.
# 로그는 노드 /dev/shm 에, 3 분마다·끝에 R2 lbours/fem_anerr/marks/<이름>.log. 생성: sed 로 @C@ @NAME@ @SCRIPT@ 치환.
export PATH=$HOME/bin:$PATH
M=/dev/shm/dkta/lbours/fem_anerr/marks; mkdir -p $M; LG=$M/@NAME@.log
( while true; do sleep 180; rclone copyto $LG r2:storage/result/anchorflow/lbours/fem_anerr/marks/@NAME@.log -q; done ) & SY=$!
echo "[시작] $(date +%T)" > $LG
IMP_C=@C@ bash /home/dkta/work/imp_boot4.sh @SCRIPT@ >> $LG 2>&1
echo "[끝] rc=$? $(date +%T)" >> $LG
kill $SY; rclone copyto $LG r2:storage/result/anchorflow/lbours/fem_anerr/marks/@NAME@.log -q
