#!/bin/bash
# 절 AN 파이프 5 (비측정, 융합 실시간 판): 단위 시험 예산 탐색 (셀 1) 과 lego 21 (셀 1·1.5 x 뉴턴 2 x PCG 10·20) 을 함께.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
R=/tmp/dkta/lbours/fem_an; M=$R/marks; mkdir -p $M; df -h /tmp | tail -1
( AN_GPUS="${AN_GPUS:-4 5 6}" bash exe/fem_an_rt_unit.sh ru1 ) &
export AN_SIM=exe/fem_an_rt_sim.py
SC=L000,$(seq -f %04g 0 19 | paste -sd,)
for e in "rl1n2c10:--cell 1 --newton 2 --cg 10" "rl1n2c20:--cell 1 --newton 2 --cg 20" "rl15n2c10:--cell 1.5 --newton 2 --cg 10" "rl15n2c20:--cell 1.5 --newton 2 --cg 20"; do
  v=${e%%:*}; A=${e#*:}
  AN_GPUS="${AN_GPUS:-4 5 6}" bash exe/fem_an_run.sh $v $SC --K 1 --integ mid $A > $M/lego_$v.out 2>&1
done
python exe/fem_an_table.py $R r15n2c5,rl1n2c10,rl1n2c20,rl15n2c10,rl15n2c20 $SC > $M/lego_rt_table.md 2>&1
rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
wait
echo "[pipe5 끝] $(date +%T)" > $M/pipe5_done; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
