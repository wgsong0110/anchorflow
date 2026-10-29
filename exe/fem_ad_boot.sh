#!/bin/bash
# 절 AD (갈래 B, 희소 셀 처리) 잡 안 진입점: 커밋 C 워크트리 ($W/wt/ad_<C>) 를 만들고 그 안의 스크립트를 돈다 (git 은 계산 노드에서). 사용: AD_C=<커밋> AD_S=<스크립트> bash fem_ad_boot.sh
W=/home/dkta/work; C=${AD_C:?커밋}; S=${AD_S:?스크립트}
WT=$W/wt/ad_$C
[ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $WT $C) || { echo "[오류] worktree $C"; exit 1; }
cd $WT && echo "[부트] $(git log -1 --oneline) -> $S" && exec bash $WT/exe/$S
