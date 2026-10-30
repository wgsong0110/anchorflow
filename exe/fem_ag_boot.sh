#!/bin/bash
# 절 AG 잡 안 진입점: 커밋 C 워크트리 ($W/wt/ag_<C>) 를 만들고 그 안의 스크립트를 돈다 (git 은 계산 노드에서).
# 사용: AG_C=<커밋> AG_S=<스크립트> [AG_ARGS=...] bash fem_ag_boot.sh
W=/home/dkta/work; C=${AG_C:?커밋}; S=${AG_S:?스크립트}
WT=$W/wt/ag_$C
[ -d $WT ] || (cd $W/anchorflow && flock $W/anchorflow/.git/c_fetch.lock git fetch -q origin && flock $W/anchorflow/.git/c_fetch.lock git worktree add -q --detach $WT $C) || { echo "[오류] worktree $C"; exit 1; }
cd $WT && echo "[부트] $(git log -1 --oneline) -> $S ${AG_ARGS:-} $(date +%T)" && exec bash $WT/exe/$S ${AG_ARGS:-}
