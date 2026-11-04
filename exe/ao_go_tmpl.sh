#!/bin/bash
# 절 AO 잡 안 진입점 (job_run.sh 가 부른다, 클러스터 git 안 씀): R2 lbours/fem_ao/code/ao_<C>.tar.gz (로컬 git archive) 를 노드 /tmp/dkta/aowt/<C> 에 풀고 스크립트를 돈다.
# 로그는 노드 /tmp 에 쓰고 3 분마다·끝에 R2 lbours/fem_ao/marks/<이름>.log 로 (로컬은 ~/r2 로 확인). 생성: sed 로 @C@ @NAME@ @CMD@ 치환.
export PATH=$HOME/bin:$PATH
RM=r2:storage/result/anchorflow/lbours/fem_ao/marks
M=/tmp/dkta/lbours/fem_ao/marks; mkdir -p $M; LG=$M/@NAME@.log
( while true; do sleep 180; rclone copyto $LG $RM/@NAME@.log -q; done ) & SY=$!
echo "[시작] $(date '+%F %T') $(hostname)" > $LG
C=@C@; D=/tmp/dkta/aowt/$C
if [ ! -f $D/.done ]; then
  mkdir -p $D && rclone copyto r2:storage/result/anchorflow/lbours/fem_ao/code/ao_$C.tar.gz /tmp/dkta/aowt/ao_$C.tar.gz -q && tar -xzf /tmp/dkta/aowt/ao_$C.tar.gz -C $D && echo $C > $D/.done || echo "[오류] 묶음 $C" >> $LG
fi
cd $D && { @CMD@ ; } >> $LG 2>&1
echo "[끝] rc=$? $(date '+%F %T')" >> $LG
kill $SY; rclone copyto $LG $RM/@NAME@.log -q
