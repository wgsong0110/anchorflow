#!/bin/bash
# 절 CM192: final_fix_bba -- 시뮬 대기열 (시간 측정) 이 끝난 뒤에만 같은 카드에서 지표 대기열 (exe/ffx_mq_worker.sh, final_fix_mq.txt) 일꾼 4 개. destroy 안 함 (사용자 지시).
export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH HOME=/root
source /root/xsenv.sh
W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; D=/root/pr/code/ce741cb4
while ! grep -q "일꾼 끝\]\|\[일꾼 끝\]" /root/pr/vast.log; do sleep 60; done
cd $D
pip install -q pyiqa -c <(pip freeze | grep -iE "^(torch|torchvision|numpy|scipy|scikit-image|opencv-python-headless)==") 2>&1 | tail -1
python -c "import pyiqa; print('[pyiqa]', pyiqa.__version__)"
[ -s $AG/gt_bread_full/config.json ] || python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/pr/gtbread_unused2.pt --out_dir $AG/gt_bread_full > /root/pr/gt_bread_full.log 2>&1
rclone cat r2:storage/result/anchorflow/lbours/final_fix/code/ffxphys_1717e403.tar 2>/dev/null | tar -x -C $D exe/ffx_mq_worker.sh
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pr XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext CUDA_VISIBLE_DEVICES=0
WP=""; for w in 0 1 2 3; do WID=$w FX_LABEL=final_fix_mq bash exe/ffx_mq_worker.sh > /root/pr/mq$w.log 2>&1 & WP="$WP $!"; sleep 10; done
wait $WP
for f in /root/pr/mq*.log; do rclone copyto $f r2:storage/result/anchorflow/lbours/final_fix/marks/final_fix_bba.$(basename $f) -q; done
echo "POST_SIM_DONE $(date)"
