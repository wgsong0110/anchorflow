#!/bin/bash
# 절 AB (갈래 A, G2P 입자 F 갱신) 잡 안 진입점: 커밋 C 워크트리 ($W/wt/ab_<C>) 를 만들고 그 안의 스크립트를 돈다 (git 은 계산 노드에서, fetch 는 다른 갈래와 같은 잠금).
# 사용: AB_C=<커밋> AB_S=<스크립트> bash fem_ab_boot.sh
W=/home/dkta/work; C=${AB_C:?커밋}; S=${AB_S:?스크립트}
WT=$W/wt/ab_$C
[ -d $WT ] || (cd $W/anchorflow && flock $W/anchorflow/.git/c_fetch.lock git fetch -q origin && flock $W/anchorflow/.git/c_fetch.lock git worktree add -q --detach $WT $C) || { echo "[오류] worktree $C"; exit 1; }
cd $WT && echo "[부트] $(git log -1 --oneline) -> $S $(date +%T)" && exec bash $WT/exe/$S
