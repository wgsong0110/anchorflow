#!/bin/bash
# 사용자 요청(10-24): ficus 셀 안 F 주축 방향 분산 (정답 PG 대 선형 출력만 최적화). 주 결과 eps 1e-5, 첫 실행(eps 1e-3)은 eps1e-3/ 에 보관
W=/home/dkta/work; O=$W/lbours/ficus_axisvar; mkdir -p $O; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
export OMP_NUM_THREADS=8 PYTHONUTF8=1 PATH=$HOME/bin:$PATH
S=$W/ficus_axisvar.py; cp $S $O/ficus_axisvar.py
if [ ! -d $O/eps1e-3 ] && [ -f $O/info.json ]; then
  mkdir -p $O/eps1e-3; mv $O/axisvar.npz $O/col_gt_long.npy $O/col_ov_long.npy $O/info.json $O/quantiles.json $O/stats.log $O/plot.log $O/compose.log $O/r_gt.* $O/r_ov.* $O/ficus_axisvar.png $O/ficus_axisvar_hist.mp4 $O/ficus_axisvar_render.mp4 $O/gpu.txt $O/eps1e-3/
  rm -f $O/AXISVAR_DONE
  python -u $S --stage plot --out $O/eps1e-3 --png $O/eps1e-3/ficus_axisvar_linaxis.png --hist_mp4 $O/eps1e-3/ficus_axisvar_hist_linaxis.mp4 > $O/eps1e-3/plot_linaxis.log 2>&1 || { echo AXISVAR_FAIL plot_old; exit 1; }
fi
python -u $S --stage stats --eps 1e-5 --out $O > $O/stats.log 2>&1 || { echo AXISVAR_FAIL stats; exit 1; }
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
