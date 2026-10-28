#!/bin/bash
# 절 M 잡 안 진입점: 커밋 C 워크트리 ($W/wt/bs_<C>) 를 만들고 그 안의 스크립트를 돈다 (git 은 계산 노드에서).
# 사용 (로그인 노드에서): AF_JOB=.. AF_NODE=.. AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/fem_bs_boot.sh <로그>   -- 부트 파일은 BS_C·BS_S 를 박아 쓴다
W=/home/dkta/work; C=${BS_C:?커밋}; S=${BS_S:?스크립트}
WT=$W/wt/bs_$C
[ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $WT $C) || { echo "[오류] worktree $C"; exit 1; }
cd $WT && echo "[부트] $(git log -1 --oneline) -> $S" && exec bash $WT/exe/$S
