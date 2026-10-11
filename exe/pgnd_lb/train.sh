# PGND lbdata 본 학습 (sloth 공식 설정 100k 반복). 다시 띄우면 train.resume_iteration=-1 로 마지막 체크포인트에서 잇는다.
#   AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/anchorflow/exe/pgnd_lb/train.sh /home/dkta/work/lb/pgnd_lb_train.out
set -u
source /home/dkta/work/anchorflow/exe/pgnd_lb/common.sh
cd $W/anchorflow && git fetch -q origin && git merge -q --ff-only FETCH_HEAD
echo "[시작] $(date) anchorflow $(git log --oneline -1) pgnd $(git -C $P log --oneline -1) + anchorflow.patch"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# 클러스터 사본을 저장소의 패치와 같게 맞춘다
git -C $P checkout -q -- experiments/train/train_eval.py && git -C $P apply $W/anchorflow/exe/pgnd_lb/anchorflow.patch && cp $W/anchorflow/exe/pgnd_lb/PATCHES_anchorflow.md $P/
echo "PGND_NO_ANOMALY=${PGND_NO_ANOMALY:-0}"
cd $P
( while true; do sleep 3600; bash $W/anchorflow/exe/pgnd_lb/upload.sh; echo "[R2] $(date)"; done ) &
UP=$!
CUDA_VISIBLE_DEVICES=5 python experiments/train/train_eval.py "${PGND_ARGS[@]}" \
  train.name=lego/train train.dataset_name=lego/dataset \
  train.num_iterations=100000 train.iteration_eval_interval=1000 train.iteration_save_interval=1000 \
  2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "it/s]$"
echo "rc=${PIPESTATUS[0]} $(date)"
kill $UP
bash $W/anchorflow/exe/pgnd_lb/upload.sh
echo PGND_TRAIN_DONE
