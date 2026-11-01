#!/bin/bash
# 절 AN 파이프 4 (비측정, 카드 AN_GPUS): (1) 실시간 판 커널별 시간 (L000 10 프레임, 셀 1·1.5·2, 뉴턴 1 x PCG 5) + ficus torch 가속 판 대조 (뉴턴 2 x PCG 10, 10 프레임)
# (2) 가망 조합 셋 검증 사다리 (실시간 판): r1n1c5 (셀 1, 1 x 5), r15n2c5 (셀 1.5, 2 x 5), r2n1c5 (셀 2, 1 x 5).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
R=/tmp/dkta/lbours/fem_an; M=$R/marks; mkdir -p $M
GL=(${AN_GPUS:-4 5 6})
j=0
for c in 1 1.5 2; do O=$R/rtprof/c$(echo $c | tr . p); mkdir -p $O
  ( CUDA_VISIBLE_DEVICES=${GL[$((j % ${#GL[@]}))]} python -u exe/fem_an_rt_sim.py --data /home/dkta/work/one/L000/L000.pt --out $O --tag L000 --frames 10 --cell $c --newton 1 --cg 5 --prof 4 < /dev/null > $O/log.txt 2>&1 ) &
  j=$((j + 1)); done
O=$R/ficus_torchfast_n2c10; mkdir -p $O
( CUDA_VISIBLE_DEVICES=${GL[0]} python -u exe/fem_an_sim.py --data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --out $O --tag ficus --frames 10 --floor none --cfg_bc --region_density --pg_vdamp \
   --fast --pc jacobi --K 1 --integ mid --newton 2 --cg 10 --cg_tol 0 --nt_tol 0 < /dev/null > $O/log.txt 2>&1 ) &
wait
rclone copy $R/rtprof r2:storage/result/anchorflow/lbours/fem_an/rtprof -q; rclone copy $O r2:storage/result/anchorflow/lbours/fem_an/ficus_torchfast_n2c10 -q
grep -a 롤아웃 $O/log.txt > $M/ficus_torchfast.txt; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
export AN_SIM=exe/fem_an_rt_sim.py
AN_GPUS="${GL[*]}" bash exe/fem_an_ladder.sh r1n1c5 --cell 1 --K 1 --integ mid --newton 1 --cg 5
AN_GPUS="${GL[*]}" bash exe/fem_an_ladder.sh r15n2c5 --cell 1.5 --K 1 --integ mid --newton 2 --cg 5
AN_GPUS="${GL[*]}" bash exe/fem_an_ladder.sh r2n1c5 --cell 2 --K 1 --integ mid --newton 1 --cg 5
