# PGND lbdata 결과 -> R2 (체크포인트·설정·평가 산출·로그)
R=r2:storage/result/anchorflow/pgnd_lb
L=/home/dkta/work/lb/pgnd/experiments/log/lego
~/bin/rclone copy $L/train $R/train --exclude '*.tmp' -q
~/bin/rclone copy /home/dkta/work/lb/pgnd_lb_train.out $R -q
~/bin/rclone copy /home/dkta/work/lb/pgnd/PATCHES_anchorflow.md $R -q
