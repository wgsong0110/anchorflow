#!/bin/bash
# 사용자 요청(10-24): ficus 셀 안 F 주축 방향 분산 (정답 PG 대 선형 출력만 최적화)
W=/home/dkta/work; O=$W/lbours/ficus_axisvar; mkdir -p $O; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
export OMP_NUM_THREADS=8 PYTHONUTF8=1 PATH=$HOME/bin:$PATH
S=$W/ficus_axisvar.py; cp $S $O/ficus_axisvar.py
python -u $S --stage stats --out $O > $O/stats.log 2>&1 || { echo AXISVAR_FAIL stats; exit 1; }
python -u $S --stage plot --out $O --png $O/ficus_axisvar.png --hist_mp4 $O/ficus_axisvar_hist.mp4 > $O/plot.log 2>&1 || { echo AXISVAR_FAIL plot; exit 1; }
cd $W/wt/pf3obj_1f22538; GTD=$W/lbours/ficus_v0/gt
g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1); echo "GPU $g" > $O/gpu.txt
CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $GTD --config $GTD/config.json --shape ficus --colors $O/col_gt_long.npy --out $O/r_gt.mp4 > $O/r_gt.log 2>&1 || { echo AXISVAR_FAIL rgt; exit 1; }
CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $W/lbours/ficus_v0/h5_ov --config $GTD/config.json --shape ficus --colors $O/col_ov_long.npy --out $O/r_ov.mp4 > $O/r_ov.log 2>&1 || { echo AXISVAR_FAIL rov; exit 1; }
python -u $S --stage compose --out $O --left $O/r_gt.mp4 --right $O/r_ov.mp4 --render_mp4 $O/ficus_axisvar_render.mp4 > $O/compose.log 2>&1 || { echo AXISVAR_FAIL compose; exit 1; }
rclone copy $O r2:storage/result/anchorflow/lbours/ficus_axisvar -q || { echo AXISVAR_FAIL r2; exit 1; }
rclone copy $O/ficus_axisvar.png r2:storage/result/anchorflow/ovnet -q
rclone copy $O/ficus_axisvar_hist.mp4 r2:storage/result/anchorflow/v20261009 -q
rclone copy $O/ficus_axisvar_render.mp4 r2:storage/result/anchorflow/v20261009 -q
touch $O/AXISVAR_DONE; echo AXISVAR_DONE
