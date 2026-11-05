#!/bin/bash
# 안정 서브스텝 묶음 잡 안 진입점 (exe/stab_launch.sh, exe/pareto_go_tmpl.sh 복사 가 sed 로 @C@ @NAME@ @CMD@ 치환): R2 lbours/stab/code/imp_<C>.bundle 을 노드 /dev/shm/dkta/impwt/<C> 에 풀고 명령을 돈다 (클러스터 로그인 노드 git 안 씀).
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
RM=r2:storage/result/anchorflow/lbours/stab/marks
M=/dev/shm/dkta/lbours/stab/marks; mkdir -p $M; LG=$M/@NAME@.log
( while true; do sleep 180; rclone copyto $LG $RM/@NAME@.log -q; done ) & SY=$!
echo "[시작] $(date '+%F %T') $(hostname)" > $LG
C=@C@; D=/dev/shm/dkta/impwt/$C
if [ ! -f $D/.done ]; then
  rclone copyto r2:storage/result/anchorflow/lbours/stab/code/imp_$C.bundle /dev/shm/dkta/impwt/imp_$C.bundle -q && rm -rf $D && git clone -q /dev/shm/dkta/impwt/imp_$C.bundle $D && (cd $D && git checkout -q $C) && echo $C > $D/.done || echo "[오류] 묶음 $C" >> $LG
fi
cd $D && { @CMD@ ; } >> $LG 2>&1
echo "[끝] rc=$? $(date '+%F %T')" >> $LG
kill $SY; rclone copyto $LG $RM/@NAME@.log -q
