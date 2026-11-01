#!/bin/bash
# 절 AN.err 잡 안 부트 (클러스터 git 없음): R2 lbours/fem_an/code/imp_<C>.tar.gz 를 노드 /dev/shm/dkta/impwt/<C> 에 풀고 스크립트를 돈다 (노드 /tmp 가득 참 대비; imp_boot4.sh 와 같은 일, 위치만 다름).
export PATH=$HOME/bin:$PATH
C=${IMP_C:?커밋}; S=${1:?스크립트}; shift
D=/dev/shm/dkta/impwt/$C
if [ ! -f $D/.done ]; then
  mkdir -p $D && rclone copyto r2:storage/result/anchorflow/lbours/fem_an/code/imp_$C.tar.gz /dev/shm/dkta/impwt/imp_$C.tar.gz -q && tar -xzf /dev/shm/dkta/impwt/imp_$C.tar.gz -C $D && echo $C > $D/.done || { echo "[오류] 묶음 $C"; exit 1; }
fi
cd $D && echo "[부트] $C -> $S $* $(date +%T)" && { case $S in /*) exec bash $S "$@";; *) exec bash $D/exe/$S "$@";; esac; }
