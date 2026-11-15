#!/bin/bash
# wolffix (11-15): 띄워 둔 vast 인스턴스에 새 커밋 코드 + 명세로 카드 하나 일 추가 (로컬에서). 사용: bash exe/wfx_remote.sh <ip> <port> <태그> <커밋> <카드> <로컬 명세 파일>
IP=$1; PORT=$2; TAG=$3; H=$4; G=$5; SPEC=$6
NM=$(basename $SPEC .txt)
scp -q -o StrictHostKeyChecking=no -P $PORT $SPEC root@$IP:/root/sv/spec_$NM.txt
ssh -o StrictHostKeyChecking=no -p $PORT root@$IP "D=/root/sv/code/$H; [ -d \$D/exe ] || { mkdir -p \$D; rclone cat r2:storage/result/anchorflow/lbours/wolffix1115/code/wfx_$H.tar | tar -x -C \$D; }; cd \$D && export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:\$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8; tmux new-session -d -s wfx${G}_$NM \"bash exe/go_wfx.sh $TAG $G /root/sv/spec_$NM.txt > /root/sv/go_${NM}.out 2>&1\"; sleep 2; tmux ls"
