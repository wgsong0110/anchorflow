#!/bin/bash
# 절 AK.22: B-스플라인 요소 CUDA 판 (비측정; EX_TIMED=1 이면 GPU0 + gpu0.lock 단독). 사용: bash exe/fem_bsel_cuda_run.sh <판 이름> <궤적,...> [추가 인자]
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
VK=$1; TS=$(echo $2 | tr ',' ' '); shift 2; EXTRA="$*"
i=0
for t in $TS; do
  O=/tmp/dkta/lbours/fem_ak_bselc/${t}_$VK; mkdir -p $O
  if [ $t = ficus ]; then DA="--data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --N 640"
  elif [ $t = L000 ]; then DA="--data /home/dkta/work/one/L000/L000.pt --frames 80 --node_h 0.028284 --floor vel --N 267"
  else DA="--data /tmp/dkta/af_data/$t.pt --frames 80 --node_h 0.028284 --floor vel --N 267"; fi
  if [ -n "${EX_TIMED:-}" ]; then
    ( exec 9>/home/dkta/work/gpu0.lock; flock 9
      while [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 500 ]; do sleep 10; done
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > $O/gpu_at_start.txt
      CUDA_VISIBLE_DEVICES=0 python -u exe/fem_bsel_cuda_sim.py --out $O --tag $t $DA $EXTRA < /dev/null > $O/log.txt 2>&1 )
  else
    G=$((2 + i % 5)); i=$((i + 1))
    CUDA_VISIBLE_DEVICES=$G python -u exe/fem_bsel_cuda_sim.py --out $O --tag $t $DA $EXTRA < /dev/null > $O/log.txt 2>&1 &
  fi
done
wait
for t in $TS; do O=/tmp/dkta/lbours/fem_ak_bselc/${t}_$VK; echo "${t}_$VK $(grep -a '롤아웃\|Error\|error' $O/log.txt | tail -1 | cut -c1-160)"; rclone copy $O r2:storage/result/anchorflow/lbours/fem_ak_bselc/${t}_$VK -q; done
echo "[끝] $(date +%T)"
