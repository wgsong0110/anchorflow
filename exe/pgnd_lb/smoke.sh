# 스모크: 200 반복 (50 마다 저장, 100 마다 평가) -> 자동 재개로 300 까지 -> 그 체크포인트로 exe/pgnd_eval_lb.py
set -u
source /home/dkta/work/anchorflow/exe/pgnd_lb/common.sh
cd $W/anchorflow && git fetch -q origin && git merge -q --ff-only FETCH_HEAD; git log --oneline -1
cd $P; rm -rf experiments/log/lego_smoke
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
S=(train.name=lego_smoke/train train.dataset_name=lego_smoke/dataset train.iteration_save_interval=50 train.iteration_eval_interval=100)
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=5 python experiments/train/train_eval.py "${PGND_ARGS[@]}" "${S[@]}" train.num_iterations=200 2>&1 | stdbuf -oL tr "\r" "\n" | grep --line-buffered -v "it/s]$" 
echo "rc1=${PIPESTATUS[0]}  $(( $(date +%s)-t0 ))s"; ls experiments/log/lego_smoke/train/ckpt
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=5 python experiments/train/train_eval.py "${PGND_ARGS[@]}" "${S[@]}" train.num_iterations=300 2>&1 | stdbuf -oL tr "\r" "\n" | grep --line-buffered -v "it/s]$"
echo "rc2=${PIPESTATUS[0]}  $(( $(date +%s)-t0 ))s"; ls experiments/log/lego_smoke/train/ckpt
CUDA_VISIBLE_DEVICES=5 python -u $W/anchorflow/exe/pgnd_eval_lb.py --pgnd $P --train_name lego_smoke/train --iteration -1 --out $W/lb/pgnd_lb_smoke_eval
echo "rc3=$?"
echo SMOKE_DONE
