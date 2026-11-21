#!/bin/bash
# final_fix_sg 재시작 뒤 대기열 다시 (onstart 가 부름): 지표 (sgfast, R2 DONE 없으면), 2·3 회 시간 (sgrep23, 안 끝났으면)
source /root/fbenv.sh; export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
sleep 30
echo "[requeue $(date)]" >> /root/sg_requeue.log
rclone lsf r2:storage/result/anchorflow/lbours/final_fix/phys/springgaus_pillow2sofa_nsg/DONE 2>/dev/null | grep -q . || tmux has-session -t sgfast 2>/dev/null || tmux new-session -d -s sgfast "bash /root/sg_p2s_fast.sh alias_sg_p2s_mq.sh"
[ -f /root/sg_rep23.done ] || tmux has-session -t sgrep23 2>/dev/null || tmux new-session -d -s sgrep23 "bash /root/sg_rep23.sh"
