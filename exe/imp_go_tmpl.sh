#!/bin/bash
# 절 AN 잡 안 실행 틀 (job_run.sh 가 부른다): 부트 + 스크립트 출력을 노드 /tmp 로그에 쓰고, 3 분마다·끝에 R2 (lbours/fem_an/marks/<이름>.log) 로 올린다.
# 로컬은 ~/r2/result/anchorflow/lbours/fem_an/marks 에서 확인 (SSH 감시 안 함). 생성: sed 로 @C@ @NAME@ @SCRIPT@ @ENV@ 치환.
export PATH=$HOME/bin:$PATH
M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $M; LG=$M/@NAME@.log
( while true; do sleep 180; rclone copyto $LG r2:storage/result/anchorflow/lbours/fem_an/marks/@NAME@.log -q; done ) & SY=$!
echo "[시작] $(date +%T)" > $LG
env @ENV@ IMP_C=@C@ bash /home/dkta/work/imp_boot.sh @SCRIPT@ >> $LG 2>&1
echo "[끝] rc=$? $(date +%T)" >> $LG
kill $SY; rclone copyto $LG r2:storage/result/anchorflow/lbours/fem_an/marks/@NAME@.log -q
