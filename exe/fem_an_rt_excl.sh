#!/bin/bash
# 절 AN 실시간 판 단독 측정 (GPU0, gpu0.lock, 판마다 카드 0 의 다른 프로세스 확인): lego L000 20 프레임, 셀 1·1.5·2 x 예산 (1x5, 2x5, 2x10, 2x20, 2x50) x 2 회, 커널별 시간 (--prof 6).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
NM=${1:-rtx1}; R=/tmp/dkta/lbours/fem_an/$NM; M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $R $M
U0=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
others0() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$U0"; }
exec 9>/home/dkta/work/gpu0.lock; flock 9
while [ "$(others0)" -gt 0 ]; do sleep 10; done
echo "카드 0 $(cat .done) $(date +%T)" > $R/info.txt; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $R/info.txt
for rep in 1 2; do for c in 1 1.5 2; do for b in "1 5" "2 5" "2 10" "2 20" "2 50"; do set -- $b
  V=c$(echo $c | tr . p)_n$1_cg$2_r$rep; O=$R/$V; mkdir -p $O
  while [ "$(others0)" -gt 0 ]; do sleep 5; done
  echo "시작 때 카드 0 의 다른 프로세스 $(others0) 개" > $O/card.txt
  nvidia-smi -i 0 --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > $O/smi.csv 2>/dev/null & SM=$!
  CUDA_VISIBLE_DEVICES=0 python -u exe/fem_an_rt_sim.py --data /home/dkta/work/one/L000/L000.pt --out $O --tag L000 --frames 20 --cell $c --newton $1 --cg $2 --prof 6 < /dev/null > $O/log.txt 2>&1
  kill $SM; echo "$V $(grep -a '롤아웃\|rror' $O/log.txt | tail -1 | cut -c1-170)"
done; done; done > $R/run.out 2>&1
cp $R/run.out $M/${NM}.out
rclone copy $R r2:storage/result/anchorflow/lbours/fem_an/$NM -q; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
