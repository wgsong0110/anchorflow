#!/bin/bash
# lego 롤아웃 상태 대 학습 상태 400 개 최근접 거리 (exe/rollout_nn_dist.py). 기존 덤프만 읽는다, 시뮬·학습 없음.
# 클러스터 잡 안에서 job_run.sh(AF_DETACH=1)로. 빈 GPU(메모리 100 MiB 미만) 하나, 없으면 CPU.
# 출력 lbours/rollout_nn 통째 R2. 사용: C=<커밋> bash rollout_nn_run.sh
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
C=${C:?커밋}; WT=$W/wt/rolloutnn_$C
OD=$W/lbours/rollout_nn; R=r2:storage/result/anchorflow/lbours/rollout_nn; mkdir -p $OD
rm -f $OD/ROLLOUTNN_DONE $OD/ROLLOUTNN_FAIL
fail(){ echo "[실패] $1"; touch $OD/ROLLOUTNN_FAIL; rclone copy $OD $R -q; exit 1; }
[ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $WT $C) || fail worktree
cd $WT || fail cd
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<100{print $1; exit}')
echo "[시작] $(date) 커밋 $(git rev-parse --short HEAD) GPU ${G:-없음(CPU)}"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
if [ -n "$G" ]; then DEVA="--dev cuda"; export CUDA_VISIBLE_DEVICES=$G; else DEVA="--dev cpu"; export CUDA_VISIBLE_DEVICES=; fi
python -u exe/rollout_nn_dist.py --out $OD $DEVA < /dev/null > $OD/run.log 2>&1 || fail "계산 (run.log)"
grep -q ROLLOUTNN_JSON_OK $OD/log.txt || fail "json 없음"
python -u exe/rollout_nn_plot.py --json $OD/rollout_nn_dist.json --png $OD/lego_rollout_nn_dist.png --md $OD/lego_rollout_nn_dist.md > $OD/plot.log 2>&1 || fail 그림
cp exe/rollout_nn_dist.py exe/rollout_nn_plot.py $OD/; git rev-parse HEAD > $OD/commit.txt
rclone copy $OD $R -q || fail rclone
rclone copyto $OD/lego_rollout_nn_dist.png r2:storage/result/anchorflow/ovnet/lego_rollout_nn_dist.png -q
rclone copyto $OD/lego_rollout_nn_dist.md r2:storage/result/anchorflow/ovnet/lego_rollout_nn_dist.md -q
touch $OD/ROLLOUTNN_DONE; rclone copy $OD $R -q; echo "[끝] $(date)"; ls $OD
