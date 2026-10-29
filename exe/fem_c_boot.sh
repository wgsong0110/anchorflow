#!/bin/bash
# 절 C (클램프 판 속도 최적화) 잡 안 진입점: 커밋 C 워크트리 ($W/wt/c_<C>) 를 만들고 그 안의 스크립트를 돈다 (git 은 계산 노드에서).
# 사용: CC_C=<커밋> CC_S=<스크립트> bash fem_c_boot.sh
W=/home/dkta/work; C=${CC_C:?커밋}; S=${CC_S:?스크립트}
WT=$W/wt/c_$C
[ -d $WT ] || (cd $W/anchorflow && flock $W/anchorflow/.git/c_fetch.lock git fetch -q origin && git worktree add -q --detach $WT $C) || { echo "[오류] worktree $C"; exit 1; }
cd $WT && echo "[부트] $(git log -1 --oneline) -> $S $(date +%T)" && exec bash $WT/exe/$S
