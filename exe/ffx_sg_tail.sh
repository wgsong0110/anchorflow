#!/bin/bash
# 절 CM192: final_fix_sg 대기열 꼬리 -- (1) pr2_grid (Spring-Gaus pillow2sofa 공식 학습 + 단독 시간 3 회) 끝난 뒤 (2) blx Spring-Gaus bread 단독 시간 (준비 판 없으면 rc 3 -> 20 분 뒤 다시) (3) Spring-Gaus pillow2sofa 대표 설정 덤프 판 지표 (exe/ffx_mq_worker.sh sg). destroy 안 함 (사용자 지시).
source /root/fbenv.sh; export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
L=/root/sg_tail.log; MK=r2:storage/result/anchorflow/lbours/final_fix/marks/final_fix_sg_tail.log
say() { echo "[sg_tail $(date '+%F %T')] $*" >> $L; rclone copyto $L $MK -q; }
while pgrep -f pr2_grid.py >/dev/null; do sleep 120; done
say "pr2_grid 끝"
for i in $(seq 1 60); do
  rclone copyto r2:storage/result/anchorflow/lbours/final_pa/blx/code/blx_time_sgb.sh /workspace/blx_time_sgb.sh -q && bash /workspace/blx_time_sgb.sh af094e61 >> /root/blx_time_sgb.log 2>&1; rc=$?
  say "blx_time_sgb 시도 $i rc=$rc"; rclone copyto /root/blx_time_sgb.log r2:storage/result/anchorflow/lbours/final_fix/marks/final_fix_sg.blx_time_sgb.log -q
  [ $rc = 3 ] || break; sleep 1200
done
D=$(ls -d /root/pr/code/*/ | head -1); cd $D
rclone cat r2:storage/result/anchorflow/lbours/final_fix/code/ffxphys_1717e403.tar | tar -x -C $D
python exe/ffx_phys_patch_render.py >> $L 2>&1
pip install -q pyiqa -c <(pip freeze | grep -iE "^(torch|torchvision|numpy|scipy|scikit-image|opencv-python-headless)==") >> $L 2>&1
touch /root/pr/bl.done
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pr XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext CUDA_VISIBLE_DEVICES=0
WID=0 FX_LABEL=final_fix_sgmq bash exe/ffx_mq_worker.sh > /root/sgmq.log 2>&1
rclone copyto /root/sgmq.log r2:storage/result/anchorflow/lbours/final_fix/marks/final_fix_sg.sgmq.log -q
say "끝"
