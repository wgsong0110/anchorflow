#!/bin/bash
# 사용자 11-09 영상: 묶음 ext2 단계 v 의 앞 단계 스크립트 (exe/stab_ext2.sh 가 부름; 단계 v 판정 판 목록은 비어 있음). H7 다섯 씬 덤프 판 + 3DGS 렌더 = exe/stab_ext2v.sh,
# 출력은 묶음 ext2 판정 결과와 섞지 않게 /dev/shm/dkta/lbours/ext2v -> R2 lbours/ext2/vid. 인자: <ext2 출력> <ext2 R2> (안 씀)
bash exe/stab_ext2v.sh /dev/shm/dkta/lbours/ext2v r2:storage/result/anchorflow/lbours/ext2/vid v < /dev/null
rclone copy /dev/shm/dkta/lbours/ext2v r2:storage/result/anchorflow/lbours/ext2/vid -q --exclude pred.pt
echo "끝 $(date '+%F %T')" > /dev/shm/dkta/lbours/ext2v/vid.END; rclone copyto /dev/shm/dkta/lbours/ext2v/vid.END r2:storage/result/anchorflow/lbours/ext2/vid/vid.END -q
