#!/bin/bash
# 조정자 11-09: 묶음 ext2 단계 v5 앞 단계 -- wolf H5 덤프 판 + 렌더 (exe/stab_ext2v.sh 단계 v5). 상주 시작점이 둘이라 잠금 디렉토리로 한 번만.
mkdir /dev/shm/dkta/ext2v_v5.lock 2>/dev/null || { echo "이미 돈다/돌았다 -- 건너뜀"; exit 0; }
bash exe/stab_ext2v.sh /dev/shm/dkta/lbours/ext2v r2:storage/result/anchorflow/lbours/ext2/vid v5 < /dev/null
rclone copy /dev/shm/dkta/lbours/ext2v/vW5 r2:storage/result/anchorflow/lbours/ext2/vid/vW5 -q --exclude pred.pt
echo "끝 $(date '+%F %T')" > /dev/shm/dkta/lbours/ext2v/vid5.END; rclone copyto /dev/shm/dkta/lbours/ext2v/vid5.END r2:storage/result/anchorflow/lbours/ext2/vid/vid5.END -q
