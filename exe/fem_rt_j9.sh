#!/bin/bash
# doc/fem_realtime.md 절 K: 후반 퍼짐 (입자 날아감) 빈도 통계 -- 시간 측정 아님, GPU 0~5 에 나눠 동시에. 기준 6 / 최종 6 / 최종에서 --eg32 뺀 판 6.
# 사용: bash exe/fem_rt_j9.sh
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
L=$W/lbours/fem_rt_lego/k_stat_$C; R=r2:storage/result/anchorflow/lbours/fem_rt_lego/k_stat_$C; mkdir -p $L
BASE="--floor particle --fast --cuda --graph --lite --warp --near32 --sym --true_res --asm_k --pcg2 6 --mix_delta 0.1 --no_dump --dump_pred"
FIN="$BASE --rawptr --lowsync --gridk2 --p2gk --eg32 both --hgate"
NOE="$BASE --rawptr --lowsync --gridk2 --p2gk --hgate"
for g in 0 1 2 3 4 5; do
  ( bash exe/fem_rt_sim.sh ks_base_$g $g $BASE; bash exe/fem_rt_sim.sh ks_fin_$g $g $FIN; bash exe/fem_rt_sim.sh ks_noe_$g $g $NOE ) < /dev/null > $L/g$g.log 2>&1 &
done
wait
RT=$W/lbours/fem_rt_lego
python exe/fem_rt_spread.py $(for k in base fin noe; do for g in 0 1 2 3 4 5; do echo $RT/ks_${k}_$g; done; done) > $L/spread.log 2>&1
for k in base fin noe; do for g in 0 1 2 3 4 5; do echo "ks_${k}_$g $(head -1 $RT/ks_${k}_$g/err_fem.txt | grep -o '평균 [0-9.]*%')"; done; done > $L/err.log
touch $L/J9_DONE; rclone copy $L $R -q
