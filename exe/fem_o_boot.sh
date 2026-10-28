#!/bin/bash
# 절 O 잡 안 진입점: 커밋 C 워크트리 ($W/wt/o_<C>) 를 만들고 그 안의 스크립트를 돈다 (git 은 계산 노드에서). 부트 파일이 O_C·O_S 를 박아 쓴다.
W=/home/dkta/work; C=${O_C:?커밋}; S=${O_S:?스크립트}
WT=$W/wt/o_$C
[ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $WT $C) || { echo "[오류] worktree $C"; exit 1; }
cd $WT && echo "[부트] $(git log -1 --oneline) -> $S" && exec bash $WT/exe/$S
