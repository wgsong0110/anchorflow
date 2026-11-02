# 잡 만료 전 마지막 R2 동기화: topo/fem31 전체 (체크포인트 포함)
~/bin/rclone copy /home/dkta/work/topo/fem31 r2:storage/result/anchorflow/topo/fem31 --transfers 8 2>&1 | tail -2
echo SYNC_DONE
