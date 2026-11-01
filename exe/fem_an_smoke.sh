#!/bin/bash
# 절 AN 첫 시험: 기울기·헤시안 대조 (lego 0001, drop1, ficus) -> 짧은 롤아웃 (5 프레임). 끝나면 R2 표식.
cd "$(dirname "$0")/.." || exit 1
df -h /tmp | tail -1; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $M
AN_GPUS="${AN_GPUS:-4 5 6}" bash exe/fem_an_run.sh chk 0001,drop1,ficus --check
AN_GPUS="${AN_GPUS:-4 5 6}" bash exe/fem_an_run.sh smk 0001,drop1,ficus --frames 5
echo "[smoke 끝] $(date +%T)" > $M/smoke_done; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
